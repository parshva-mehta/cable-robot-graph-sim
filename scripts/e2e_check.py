#!/usr/bin/env python3
"""End-to-end check for the EKF -> ROS pipeline (this repo has TWO hook points).

Runs a real EKF through BOTH hook points with the sinks attached:

  * ``run_ekf_rollout`` -- the batch rollout (mirrors the reference repo), and
  * ``OnlineEKF``       -- the streaming wrapper unique to this repo.

Both write the ROS-readable rollout file and optionally stream live
``nav_msgs/Odometry`` to rosbridge. The script works with or without a trained
model and with or without the dataset:

  * No ``--model``: a lightweight STUB simulator is built from the robot config.
    The GNN forward is stubbed (identity dynamics), but the real EKF math runs
    -- gtsam predict/update, the quaternion tangent-space linearization in
    ``linearization.py``, and both hook points -- so the plumbing (sink fires on
    every frame) and the units cross-check are exercised end to end without a
    ``.pt`` file or a dataset.
  * ``--model PATH``: loads a real ``TensegrityGNNSimulator`` via
    ``load_simulator`` instead.
  * ``--data-dir DIR``: uses the real dataset; otherwise ``gt``/``extra`` are
    synthesized from the robot config.

Examples:
    # file only, stub simulator, synthetic data (no model, no dataset, no ROS)
    python3 scripts/e2e_check.py

    # also stream to a running rosbridge
    ROSBRIDGE_URL=ws://localhost:9090 python3 scripts/e2e_check.py --ros

    # real model + real dataset
    python3 scripts/e2e_check.py --ros --model best_model.pt \
        --data-dir /path/to/dataset/traj_6
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Allow running as `python3 scripts/e2e_check.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ekf import OnlineEKF, run_ekf_rollout
from sim_data_publisher import (STATE_DIM_PER_ROD, DEFAULT_POSITION_SCALE,
                                CompositeSink, RodStatePublisher,
                                RolloutStateFileWriter, rod_names_from_simulator,
                                build_odometry_msg, split_rod_states,
                                split_rod_covariances)

DEFAULT_CONFIG = "simulators/configs/3_bar_gnn_sim_config.json"


# ---------------------------------------------------------------------------
# Stub simulator: exercises the real EKF loops without a trained GNN.
# ---------------------------------------------------------------------------

class _StubMotorState:
    def __init__(self):
        self.omega_t = torch.zeros(1, 1, 1)


class _StubMotor:
    def __init__(self):
        self.motor_state = _StubMotorState()


class _StubCable:
    def __init__(self, rest_length):
        self._rest_length = torch.tensor(float(rest_length))
        self.actuation_length = torch.zeros(1, 1, 1)
        self.motor = _StubMotor()


class _StubRod:
    def __init__(self, name):
        self.name = name


class _StubRobot:
    def __init__(self, rod_names, cable_rest_lengths):
        self.rigid_bodies = OrderedDict((n, _StubRod(n)) for n in rod_names)
        self.actuated_cables = OrderedDict(
            (f"cable_{i}", _StubCable(rl))
            for i, rl in enumerate(cable_rest_lengths)
        )

    @property
    def rods(self):
        return self.rigid_bodies


class StubSimulator:
    """Identity-dynamics stand-in for TensegrityGNNSimulator.

    Implements only what ``ekf.py`` and ``linearization.py`` touch:
    ``step``, ``robot.{rods,rigid_bodies,actuated_cables}``, ``parameters``,
    ``ctrls_hist``, ``node_hidden_state``, ``dtype``, ``device``.
    """

    def __init__(self, rod_names, cable_rest_lengths):
        self.robot = _StubRobot(rod_names, cable_rest_lengths)
        self.ctrls_hist = None
        self.node_hidden_state = None
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def parameters(self):
        return iter(())  # no params -> linearize_dynamics falls back to f32/cpu

    def step(self, curr_state, ctrls=None, state_to_graph_kwargs=None):
        # Identity dynamics: next state == current state. Keeps quaternions
        # valid and gtsam stable while still driving every hook and the EKF
        # predict/update path for real.
        return curr_state.clone(), None


def build_stub_simulator(config_path):
    cfg = json.load(open(config_path))
    rods = cfg["tensegrity_cfg"]["rods"]
    rod_names = [r["name"] for r in rods]
    rest_lengths = [
        c["rest_length"] for c in cfg["tensegrity_cfg"]["cables"]
        if c.get("type") == "actuated_cable"
    ]
    return StubSimulator(rod_names, rest_lengths)


def load_real_simulator(model_path):
    from simulators.tensegrity_gnn_simulator import load_simulator
    try:
        sim = load_simulator(model_path, map_location=torch.device("cpu"),
                             cache_batch_sizes=[1])
    except TypeError:
        sim = load_simulator(model_path, map_location=torch.device("cpu"))
    sim = sim.to("cpu")
    sim.eval()
    return sim


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def synthetic_data(config_path, simulator, n_steps):
    """Build a minimal gt/extra pair from the robot config (no dataset needed).

    Each rod's COM is the midpoint of its two config ``end_pts``; orientation is
    identity, velocities zero. Every frame is identical, so the EKF settles onto
    the config COMs -- which is exactly what the units cross-check needs.
    """
    cfg = json.load(open(config_path))
    rods = cfg["tensegrity_cfg"]["rods"]

    pos, quat = [], []
    for rod in rods:
        a, b = rod["end_pts"]
        pos.append([(a[i] + b[i]) / 2 for i in range(3)])
        quat.append([1.0, 0.0, 0.0, 0.0])
    zeros3 = [[0.0] * 3 for _ in rods]

    cables = list(simulator.robot.actuated_cables.values())
    rest_lengths = [float(c._rest_length) for c in cables]

    # gt_data stores flat per-key arrays (pos: 3*n_rods, quat: 4*n_rods, ...).
    flat_pos = [v for p in pos for v in p]
    flat_quat = [v for q in quat for v in q]
    flat_zeros = [0.0] * (3 * len(rods))
    gt = [{"pos": flat_pos, "quat": flat_quat,
           "linvel": flat_zeros, "angvel": flat_zeros} for _ in range(n_steps + 1)]
    extra = [{"controls": [0.0] * len(cables),
              "rest_lengths": rest_lengths,
              "motor_speeds": [0.0] * len(cables)} for _ in range(n_steps)]
    return gt, extra


def load_dataset(data_dir, n_steps):
    data_dir = Path(data_dir)
    gt = json.load((data_dir / "processed_data.json").open("r"))
    extra_name = "extra_state_data.json"
    extra = json.load((data_dir / extra_name).open("r"))
    if n_steps > 0:
        gt, extra = gt[:n_steps + 1], extra[:n_steps]
    return gt, extra


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _build_start_state(gt0, n_rods):
    vals = []
    for r in range(n_rods):
        vals.extend(
            gt0["pos"][r * 3:(r + 1) * 3]
            + gt0["quat"][r * 4:(r + 1) * 4]
            + gt0["linvel"][r * 3:(r + 1) * 3]
            + gt0["angvel"][r * 3:(r + 1) * 3]
        )
    return torch.tensor(vals, dtype=torch.float32).reshape(1, -1, 1)


def _z_from_gt(gt_t, n_rods):
    pos = np.array(gt_t["pos"], np.float64).reshape(-1, 3)
    quat = np.array(gt_t["quat"], np.float64).reshape(-1, 4)
    lv = np.array(gt_t["linvel"], np.float64).reshape(-1, 3)
    av = np.array(gt_t["angvel"], np.float64).reshape(-1, 3)
    return np.hstack([pos, quat, lv, av]).reshape(-1)


def check_file(path, n_frames, n_rods, scale):
    """Validate the written file the way the ROS interface package reads it.

    ``scale`` is this robot's ``data_scale_factor`` (0.325/2.95), the value the
    ROS launch must use for this 2.95-unit robot -- NOT the reference's 0.10.
    """
    lines = Path(path).read_text().splitlines()
    widths = {len(line.split()) for line in lines}
    expected_cols = n_rods * STATE_DIM_PER_ROD

    ok = True
    print(f"  frames returned : {n_frames}")
    print(f"  file lines      : {len(lines)}")
    print(f"  columns/line    : {widths} (expected {{{expected_cols}}})")
    if len(lines) != n_frames:
        print("  FAIL: file line count != frame count"); ok = False
    if widths != {expected_cols}:
        print("  FAIL: unexpected column count"); ok = False

    try:
        from scipy.spatial.transform import Rotation as SciPyRot
    except ImportError:
        print("  (scipy missing -- skipping rotation validity check)")
        return ok

    values = [float(t) for t in lines[-1].split()]
    names = ["red", "green", "blue"]
    for r in range(n_rods):
        b = values[r * STATE_DIM_PER_ROD:(r + 1) * STATE_DIM_PER_ROD]
        q = np.array(b[3:7], np.float32)
        # create_transform() in interface/scripts/sim_data_publisher.py: W first
        R = SciPyRot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        com_m = np.array(b[0:3], np.float32) * scale  # ROS data_scale_factor
        resid = np.abs(R @ R.T - np.eye(3)).max()
        label = names[r] if r < len(names) else f"rod{r}"
        print(f"  {label:>5}: com(m)={np.round(com_m, 4)}  "
              f"det(R)={np.linalg.det(R):.6f}  |RR^T-I|={resid:.2e}")
        if abs(np.linalg.det(R) - 1.0) > 1e-5 or resid > 1e-5:
            print(f"  FAIL: {label} rotation is not orthonormal"); ok = False
    return ok


class _FanOut:
    """Fans publish_state to an (already-open) file writer and an optional
    shared, externally-managed live publisher. Does not open/close either --
    the file writer is managed by its own ``with`` block and the live publisher
    is connected/closed once by the caller (see the reactor note in main)."""

    def __init__(self, file_writer, live_pub=None):
        self.sinks = [s for s in (file_writer, live_pub) if s is not None]

    def publish_state(self, time, state, covariance=None):
        out = []
        for s in self.sinks:
            try:
                out.append(s.publish_state(time, state, covariance=covariance))
            except TypeError:
                out.append(s.publish_state(time, state))
        return out


class _CountingSink:
    """Wraps a sink and counts publish_state calls, to prove every frame fired.

    Also records the last (state, covariance) seen so the caller can verify the
    EKF covariance reached the sink and projects into a non-zero Odometry
    covariance."""

    def __init__(self, inner):
        self.inner = inner
        self.count = 0
        self.cov_frames = 0
        self.last_state = None
        self.last_cov = None

    def publish_state(self, time, state, covariance=None):
        self.count += 1
        if covariance is not None:
            self.cov_frames += 1
            self.last_state = state
            self.last_cov = covariance
        try:
            return self.inner.publish_state(time, state, covariance=covariance)
        except TypeError:
            return self.inner.publish_state(time, state)

    def open(self):
        opener = getattr(self.inner, "open", None) or getattr(self.inner, "connect", None)
        if opener:
            opener()
        return self

    def close(self):
        closer = getattr(self.inner, "close", None)
        if closer:
            closer()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="Trained .pt model; omit to use the stub simulator.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None,
                    help="Real dataset directory; omit to use synthetic data.")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", default="rollout_ekf.txt")
    ap.add_argument("--out-online", default="rollout_ekf_online.txt")
    ap.add_argument("--ros", action="store_true",
                    help="Also stream live Odometry to rosbridge.")
    ap.add_argument("--rosbridge-url", default=None,
                    help="Overrides $ROSBRIDGE_URL (default ws://localhost:9090).")
    ap.add_argument("--gnn-jacobian", action="store_true",
                    help="Linearize with the GNN Jacobian instead of finite diff.")
    args = ap.parse_args()

    if args.model:
        if not Path(args.model).exists():
            sys.exit(f"model not found: {args.model} (run from the repo root)")
        print(f"loading model {args.model} ...")
        sim = load_real_simulator(args.model)
    else:
        print(f"building stub simulator from {args.config} ...")
        sim = build_stub_simulator(args.config)

    rod_names = rod_names_from_simulator(sim)
    n_rods = len(rod_names)
    print(f"  simulator : {type(sim).__name__}")
    print(f"  rods      : {rod_names}")
    print(f"  scale     : {DEFAULT_POSITION_SCALE:.10f} (= 0.325/2.95)")

    if args.data_dir:
        gt, extra = load_dataset(args.data_dir, args.steps)
        print(f"  data      : {args.data_dir} ({len(extra)} steps)")
    else:
        gt, extra = synthetic_data(args.config, sim, args.steps)
        print(f"  data      : synthetic from {args.config} ({len(extra)} steps)")

    use_fd = not args.gnn_jacobian
    ok = True

    def report_covariance(counting, n_frames):
        """Verify the EKF covariance reached the sink and projects to a non-zero
        6x6 Odometry covariance for each rod."""
        nonlocal ok
        print(f"  covariance      : {counting.cov_frames}/{n_frames} frames "
              f"carried a covariance")
        if counting.cov_frames != n_frames:
            print("  FAIL: covariance missing on some frames"); ok = False
            return
        cov = np.asarray(counting.last_cov)
        if cov.shape != (n_rods * STATE_DIM_PER_ROD, n_rods * STATE_DIM_PER_ROD):
            print(f"  FAIL: covariance shape {cov.shape}"); ok = False
            return
        rods = split_rod_states(counting.last_state)
        blocks = split_rod_covariances(cov, n_rods)

        def fmt6(flat, indent="        "):
            # Scientific notation so tiny-but-nonzero variances (e.g. 3.6e-06)
            # stay visible instead of rounding to 0.000 -- this is proof output.
            m = np.asarray(flat, dtype=float).reshape(6, 6)
            body = np.array2string(
                m, max_line_width=200, separator="  ", prefix=indent,
                formatter={"float_kind": lambda x: f"{x: .2e}"})
            return indent + body

        for name, (pos, quat, lv, av), P in zip(rod_names, rods, blocks):
            msg = build_odometry_msg(name, 0.0, pos, quat, lv, av,
                                     twist_frame="body",
                                     position_scale=DEFAULT_POSITION_SCALE,
                                     rod_covariance=P)
            pose = np.asarray(msg["pose"]["covariance"]).reshape(6, 6)
            twist = np.asarray(msg["twist"]["covariance"]).reshape(6, 6)
            # Full 6x6 matrices, for proof (order: [x y z rot_x rot_y rot_z] /
            # [vx vy vz wx wy wz]).
            print(f"    {name}  pose.covariance 6x6 "
                  f"[x y z rot_x rot_y rot_z]:")
            print(fmt6(pose))
            print(f"    {name}  twist.covariance 6x6 "
                  f"[vx vy vz wx wy wz]:")
            print(fmt6(twist))
            if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(twist)):
                print(f"  FAIL: {name} covariance not finite"); ok = False
            if float(np.max(pose.diagonal())) <= 0.0:
                print(f"  FAIL: {name} pose covariance is all zero"); ok = False

    # ONE shared live publisher, connected once. roslibpy runs a Twisted reactor
    # that cannot be restarted within a process, so a second RodStatePublisher
    # (a second connect after the first terminated) would fail with
    # ReactorNotRestartable. Both hook paths reuse this single connection and it
    # is closed once at the very end.
    live_pub = None
    if args.ros:
        live_pub = RodStatePublisher(url=args.rosbridge_url,
                                     rod_names=rod_names, stamp_source="sim")
        live_pub.connect()
        print(f"  rosbridge : {live_pub.url}")

    try:
        # --- Path 1: run_ekf_rollout --------------------------------------
        print("\n[1/2] run_ekf_rollout ...")
        with RolloutStateFileWriter(args.out, expected_n_rods=n_rods) as fw:
            counting = _CountingSink(_FanOut(fw, live_pub))
            frames = run_ekf_rollout(sim, gt, extra, args.dt,
                                     use_finite_diff=use_fd, publisher=counting)
        print(f"  sink fired {counting.count} times for {len(frames)} frames")
        if counting.count != len(frames):
            print("  FAIL: sink did not fire on every frame"); ok = False
        ok = check_file(args.out, len(frames), n_rods, DEFAULT_POSITION_SCALE) and ok
        report_covariance(counting, len(frames))
        norm = np.linalg.norm(frames[-1]["state"].flatten().tolist()[3:7])
        print(f"  quat norm       : {norm:.6f}")
        if abs(norm - 1.0) > 1e-4:
            print("  FAIL: quaternion is not normalized"); ok = False

        # --- Path 2: OnlineEKF --------------------------------------------
        print("\n[2/2] OnlineEKF (streaming) ...")
        start_state = _build_start_state(gt[0], n_rods)
        with RolloutStateFileWriter(args.out_online, expected_n_rods=n_rods) as fw:
            online_counting = _CountingSink(_FanOut(fw, live_pub))
            ekf = OnlineEKF(sim, dt=args.dt, n_rods=n_rods, use_finite_diff=use_fd,
                            publisher=online_counting)
            ekf.initialize(start_state,
                           rest_lengths=extra[0]["rest_lengths"],
                           motor_speeds=extra[0]["motor_speeds"])
            n_online_frames = 1  # the initialize() publish
            for k, ex in enumerate(extra):
                have_meas = k + 1 < len(gt)
                z = _z_from_gt(gt[k + 1], n_rods) if have_meas else None
                ekf.step(z_t=z, u_t=ex["controls"], have_measurement=have_meas)
                n_online_frames += 1
        print(f"  sink fired {online_counting.count} times for {n_online_frames} frames")
        if online_counting.count != n_online_frames:
            print("  FAIL: OnlineEKF sink did not fire on every frame"); ok = False
        ok = check_file(args.out_online, n_online_frames, n_rods,
                        DEFAULT_POSITION_SCALE) and ok
        report_covariance(online_counting, n_online_frames)
    finally:
        if live_pub is not None:
            live_pub.close()

    print("\n" + ("PASS: both hook points ran, every frame reached every sink."
                  if ok else "FAIL: see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
