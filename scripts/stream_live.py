#!/usr/bin/env python3
"""Continuously stream EKF estimates to rosbridge for a live Foxglove/RViz view.

Unlike ``e2e_check.py`` (which publishes a handful of frames and exits, so a
consumer that connects afterward sees nothing -- nav_msgs/Odometry is not
latched), this keeps a single rosbridge connection open and publishes on a timer
until Ctrl-C (or ``--duration`` elapses). Connect Foxglove first, then run this.

It advertises three topics on ONE shared connection (roslibpy uses a single
global Twisted reactor, so multiple connections are fragile):

  /tensegrity/<rod>/odom      nav_msgs/Odometry     (pose + twist + covariance)
  /tensegrity/ekf/covariance  std_msgs/Float64MultiArray  (36x36 joint tangent)
  /tensegrity/ekf/jacobian    std_msgs/Float64MultiArray  (36x36 tangent df/dx)

Data source:
  * ``--data-dir DIR`` replays real ground-truth measurements (loops).
  * otherwise a synthetic trajectory is generated so the rods visibly move
    (gentle bounce + yaw), which is enough to confirm the live view, the
    covariance ellipsoids, and the matrix topics -- but it is NOT real motion.
  * ``--model PATH`` uses the trained GNN for the dynamics/linearization;
    otherwise a stub (identity dynamics) is used.

Examples:
    ROSBRIDGE_URL=ws://localhost:9090 python3 scripts/stream_live.py --rate 15
    python3 scripts/stream_live.py --model best.pt --data-dir data/traj_6 --rate 20
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ekf import OnlineEKF
from e2e_check import (build_stub_simulator, load_real_simulator,
                       load_dataset, _build_start_state)
from sim_data_publisher import (CompositeSink, RodStatePublisher,
                                MatrixStreamPublisher, rod_names_from_simulator,
                                split_rod_states, quat_wxyz_to_ros,
                                ros_time_from_seconds, DEFAULT_POSITION_SCALE)

DEFAULT_CONFIG = "simulators/configs/3_bar_gnn_sim_config.json"


def _build_tf_message(state_flat, rod_names, scale, frame_id="world"):
    """A tf2_msgs/TFMessage placing each rod frame under ``world`` from the EKF
    state. Foxglove's 3D panel needs a transform tree to render; nav_msgs/Odometry
    does not populate one, so without this the panel shows only the grid."""
    stamp = ros_time_from_seconds(__import__("time").time())
    transforms = []
    for name, (pos, quat, _lv, _av) in zip(rod_names, split_rod_states(state_flat)):
        transforms.append({
            "header": {"stamp": stamp, "frame_id": frame_id},
            "child_frame_id": name,
            "transform": {
                "translation": {"x": pos[0] * scale, "y": pos[1] * scale,
                                "z": pos[2] * scale},
                "rotation": quat_wxyz_to_ros(quat),
            },
        })
    return {"transforms": transforms}


def _build_path_message(history, frame_id="world"):
    """A nav_msgs/Path from a list of (stamp, position_m, ros_quat) samples.
    Foxglove's 3D panel draws Path as a line, which gives the position trail
    that a single Odometry message (one pose, latest only) cannot."""
    return {
        "header": {"stamp": history[-1][0], "frame_id": frame_id},
        "poses": [{
            "header": {"stamp": stamp, "frame_id": frame_id},
            "pose": {"position": {"x": p[0], "y": p[1], "z": p[2]},
                     "orientation": q},
        } for stamp, p, q in history],
    }


def _config_base_state(config_path, n_rods):
    """Per-rod COM (midpoint of config end_pts) with identity orientation."""
    cfg = json.load(open(config_path))
    rods = cfg["tensegrity_cfg"]["rods"]
    base = []
    for rod in rods[:n_rods]:
        a, b = rod["end_pts"]
        com = [(a[i] + b[i]) / 2 for i in range(3)]
        base.append(com)
    return base


def _synthetic_pose_measurement(base_coms, t, amplitude, yaw_amp, freq):
    """Pose-only measurement (7 per rod: pos + quat wxyz) that gently moves so a
    live viewer sees motion. Not real data -- a visibility aid."""
    w = 2.0 * math.pi * freq
    z = []
    for com in base_coms:
        dz = amplitude * math.sin(w * t)
        pos = [com[0], com[1], com[2] + dz]
        ang = yaw_amp * math.sin(w * t)          # yaw about +z
        quat = [math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]  # (w,x,y,z)
        z += pos + quat
    return np.array(z, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--best", action="store_true",
                    help="use the trained model + trajectory from eval.py's default "
                         "paths (best_rollout_model.pt + traj_6); fills --model and "
                         "--data-dir if unset")
    ap.add_argument("--rosbridge-url", default=None)
    ap.add_argument("--rate", type=float, default=10.0, help="publish rate (Hz)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to stream (0 = until Ctrl-C)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--amplitude", type=float, default=0.3,
                    help="synthetic vertical bounce amplitude (sim units)")
    ap.add_argument("--yaw", type=float, default=0.4,
                    help="synthetic yaw amplitude (radians)")
    ap.add_argument("--freq", type=float, default=0.2, help="synthetic motion Hz")
    ap.add_argument("--no-matrices", action="store_true",
                    help="skip the covariance/jacobian Float64MultiArray topics")
    ap.add_argument("--gnn-jacobian", action="store_true",
                    help="linearize with the GNN Jacobian (torch.func.jacrev) "
                         "instead of finite differences -- much faster with a real "
                         "--model (one pass vs ~78), so the stream can keep up")
    ap.add_argument("--no-tf", action="store_true",
                    help="skip the /tf world->rod broadcast (needed for the "
                         "Foxglove/RViz 3D panel to place the rods)")
    ap.add_argument("--trail", type=int, default=300,
                    help="poses kept in each rod's /tensegrity/<rod>/path trail "
                         "(0 = disable the trail topics)")
    ap.add_argument("--trail-every", type=int, default=3,
                    help="publish the trail every N frames (the Path message "
                         "grows with --trail, so this bounds the bandwidth)")
    args = ap.parse_args()

    if args.best:
        # Same default model/trajectory paths as eval.py (mac/windows aware).
        # Kept inline (not imported from eval.py) so resolving the paths does not
        # drag in the simulator stack -- the torch_geometric dependency then
        # surfaces at model load with an actionable message, below.
        import os as _os
        _mac_root = (
            "/Users/parshvamehta/Library/Mobile Documents/com~apple~CloudDocs/"
            "PRACSYS/cablegraphrobot/tensegrity")
        _paths = {
            "model_path": (
                r"C:\Users\parshva-mehta\OneDrive\Documents\Projects\PRACSYS\Tensegrity"
                r"\tensegrity\models\best_n_step_rollout_model.pt"
                if _os.name == "nt" else
                f"{_mac_root}/models/best_rollout_model.pt"),
            "data_dir": (
                r"C:\Users\parshva-mehta\OneDrive\Documents\Projects\PRACSYS\Tensegrity"
                r"\tensegrity\data_sets\3bar_new_platform_high_friction\dataset_0\traj_6"
                if _os.name == "nt" else
                f"{_mac_root}/data_sets/3bar_new_platform_high_friction/dataset_0/traj_6"),
        }
        args.model = args.model or _paths["model_path"]
        args.data_dir = args.data_dir or _paths["data_dir"]
        print(f"(--best) model={args.model}")
        print(f"(--best) data ={args.data_dir}")

    try:
        sim = (load_real_simulator(args.model) if args.model
               else build_stub_simulator(args.config))
    except ModuleNotFoundError as exc:
        if "torch_geometric" in str(exc):
            sys.exit(
                "error: loading the trained model needs torch_geometric, which is "
                "not in this Python env.\n"
                "  Activate the project env:  conda activate cable_robot_gnn\n"
                "  or install deps:           pip install -r requirements.txt\n"
                "  (the synthetic stub stream -- no --model/--best -- needs neither)."
            )
        raise
    rod_names = rod_names_from_simulator(sim)
    n_rods = len(rod_names)
    n_cables = len(sim.robot.actuated_cables)
    print(f"simulator : {type(sim).__name__}  rods={rod_names}")

    # Data source.
    gt = extra = None
    if args.data_dir:
        gt, extra = load_dataset(args.data_dir, 0)
        print(f"data      : {args.data_dir} ({len(gt)} frames, looping)")
    else:
        base_coms = _config_base_state(args.config, n_rods)
        print("data      : synthetic moving trajectory (visibility aid, not real)")

    # One shared rosbridge connection for all three topics.
    live = RodStatePublisher(url=args.rosbridge_url, rod_names=rod_names,
                             stamp_source="wall")
    live.connect()
    print(f"rosbridge : {live.url}")
    sinks = [live]
    if not args.no_matrices:
        sinks.append(MatrixStreamPublisher(form="tangent", ros=live.ros))
        sinks.append(MatrixStreamPublisher(topic="/tensegrity/ekf/jacobian",
                                           source="jacobian", ros=live.ros))
        for s in sinks[1:]:
            s.connect()
        print("topics    : /tensegrity/<rod>/odom, /tensegrity/ekf/covariance, "
              "/tensegrity/ekf/jacobian")
    else:
        print("topics    : /tensegrity/<rod>/odom")
    pub = CompositeSink(*sinks)

    # /tf broadcast so the 3D panel has a frame tree to place the rods in.
    tf_topic = None
    if not args.no_tf:
        import roslibpy
        tf_topic = roslibpy.Topic(live.ros, "/tf", "tf2_msgs/TFMessage")
        tf_topic.advertise()
        print("            /tf (world -> each rod, for the 3D panel)")

    # Per-rod trail: a rolling nav_msgs/Path so the 3D panel can draw where each
    # rod has been. Odometry alone renders only the latest pose.
    path_topics = {}
    trails = {}
    if args.trail > 0:
        import collections
        import roslibpy
        for name in rod_names:
            t = roslibpy.Topic(live.ros, f"/tensegrity/{name}/path", "nav_msgs/Path")
            t.advertise()
            path_topics[name] = t
            trails[name] = collections.deque(maxlen=args.trail)
        print("            /tensegrity/<rod>/path (position trail, "
              f"last {args.trail} poses)")

    # observe_pose_only keeps the synthetic measurement simple (pos+quat).
    ekf = OnlineEKF(sim, dt=args.dt, n_rods=n_rods,
                    use_finite_diff=not args.gnn_jacobian,
                    observe_pose_only=(gt is None), publisher=pub)

    if gt is not None:
        start = _build_start_state(gt[0], n_rods)
        rest = extra[0]["rest_lengths"]
        motors = extra[0]["motor_speeds"]
    else:
        # Build a start state (pos, identity quat, zero vel) from the config.
        vals = []
        for com in base_coms:
            vals += com + [1.0, 0.0, 0.0, 0.0] + [0.0] * 3 + [0.0] * 3
        import torch
        start = torch.tensor(vals, dtype=torch.float32).reshape(1, -1, 1)
        rest = [float(c._rest_length) for c in sim.robot.actuated_cables.values()]
        motors = [0.0] * n_cables
    ekf.initialize(start, rest_lengths=rest, motor_speeds=motors)

    period = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()
    k = 0
    print(f"streaming at {args.rate:g} Hz (Ctrl-C to stop) ...")
    try:
        while True:
            t = time.time() - t0
            if args.duration > 0 and t >= args.duration:
                break
            if gt is not None:
                idx = (k + 1) % len(gt)
                g = gt[idx]
                z = np.hstack([np.array(g["pos"]).reshape(-1, 3),
                               np.array(g["quat"]).reshape(-1, 4),
                               np.array(g["linvel"]).reshape(-1, 3),
                               np.array(g["angvel"]).reshape(-1, 3)]).reshape(-1)
                u = extra[idx % len(extra)]["controls"]
            else:
                z = _synthetic_pose_measurement(base_coms, t, args.amplitude,
                                                args.yaw, args.freq)
                u = [0.0] * n_cables
            ekf.step(z_t=z, u_t=u, have_measurement=True)
            if tf_topic is not None:
                import roslibpy
                st = ekf.state_torch.detach().cpu().numpy().reshape(-1)
                tf_topic.publish(roslibpy.Message(
                    _build_tf_message(st, rod_names, DEFAULT_POSITION_SCALE)))
            if path_topics:
                import roslibpy
                st = ekf.state_torch.detach().cpu().numpy().reshape(-1)
                stamp = ros_time_from_seconds(time.time())
                for name, (pos, quat, _lv, _av) in zip(rod_names,
                                                       split_rod_states(st)):
                    trails[name].append((
                        stamp,
                        [float(c) * DEFAULT_POSITION_SCALE for c in pos],
                        quat_wxyz_to_ros(quat)))
                if k % max(args.trail_every, 1) == 0:
                    for name, t in path_topics.items():
                        t.publish(roslibpy.Message(_build_path_message(trails[name])))
            k += 1
            if k % max(int(args.rate), 1) == 0:
                print(f"  t={t:6.1f}s  frames={k}", end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        for t in path_topics.values():
            try:
                t.unadvertise()
            except Exception:  # noqa: BLE001
                pass
        if tf_topic is not None:
            try:
                tf_topic.unadvertise()
            except Exception:  # noqa: BLE001
                pass
        # Close borrowers first (they do not terminate the shared connection),
        # then the owner.
        for s in reversed(sinks):
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"done. published {k} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
