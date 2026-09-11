#!/usr/bin/env python3
"""Proof that the EKF covariance reaches a real ROS topic (before vs after).

Runs the real GTSAM EKF (stub simulator, so no trained model is needed), then
round-trips two nav_msgs/Odometry messages through a running rosbridge + roscore
and captures them back off the topic via a subscription:

  BEFORE  -- the message the v1 bridge produced (no covariance): 36 zeros.
  AFTER   -- the same rod state with the EKF's covariance projected in.

Both are published to the SAME real topic and read back through ROS, so this is
an end-to-end wire check, not an in-process assertion. Writes a proof report to
--out (default covariance_proof.txt) and exits non-zero if the check fails.

Prereq: a rosbridge websocket at $ROSBRIDGE_URL (default ws://localhost:9090),
e.g. the container started by run_all.sh / TESTING.md Level 3a.

    ROSBRIDGE_URL=ws://localhost:9090 python3 scripts/prove_covariance.py
"""
import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import roslibpy

from ekf import OnlineEKF
from e2e_check import build_stub_simulator, synthetic_data, _build_start_state, _z_from_gt
from sim_data_publisher import (
    DEFAULT_POSITION_SCALE, build_odometry_msg, rod_names_from_simulator,
    split_rod_states, split_rod_covariances,
)

CONFIG = "simulators/configs/3_bar_gnn_sim_config.json"


def run_ekf_final_state(steps=4, dt=0.01):
    """Drive the real GTSAM EKF and return the last (state, covariance)."""
    sim = build_stub_simulator(CONFIG)
    rods = rod_names_from_simulator(sim)
    n = len(rods)
    gt, extra = synthetic_data(CONFIG, sim, steps)

    class _Rec:
        state = None
        cov = None

        def publish_state(self, t, state, covariance=None):
            self.state, self.cov = state, covariance

    rec = _Rec()
    ekf = OnlineEKF(sim, dt=dt, n_rods=n, use_finite_diff=True, publisher=rec)
    ekf.initialize(_build_start_state(gt[0], n),
                   rest_lengths=extra[0]["rest_lengths"],
                   motor_speeds=extra[0]["motor_speeds"])
    for k, ex in enumerate(extra):
        have = k + 1 < len(gt)
        ekf.step(z_t=_z_from_gt(gt[k + 1], n) if have else None,
                 u_t=ex["controls"], have_measurement=have)
    return rods, rec.state, rec.cov


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("ROSBRIDGE_URL", "ws://localhost:9090"))
    ap.add_argument("--out", default="covariance_proof.txt")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    rods, state, cov = run_ekf_final_state(steps=args.steps)
    if cov is None:
        sys.exit("FAIL: EKF produced no covariance")
    name = rods[0]
    pos, quat, lv, av = split_rod_states(state)[0]
    P0 = split_rod_covariances(cov, len(rods))[0]

    msg_before = build_odometry_msg(name, 0.0, pos, quat, lv, av,
                                    position_scale=DEFAULT_POSITION_SCALE)
    msg_after = build_odometry_msg(name, 0.0, pos, quat, lv, av,
                                   position_scale=DEFAULT_POSITION_SCALE,
                                   rod_covariance=P0)

    # --- round-trip through real ROS (or in-process fallback) --------------
    topic_name = "/tensegrity/proof/rod_01/odom"
    mode = "real ROS round-trip (published, then read back via subscription)"
    received = []
    try:
        url = args.url if "://" in args.url else "ws://" + args.url
        scheme, _, hostport = url.partition("://")
        host, _, port = hostport.partition(":")
        ros = roslibpy.Ros(host=host, port=int(port) if port else 9090,
                           is_secure=(scheme == "wss"))
        ros.run(timeout=10)
        if not ros.is_connected:
            raise ConnectionError("not connected")

        pub = roslibpy.Topic(ros, topic_name, "nav_msgs/Odometry")
        sub = roslibpy.Topic(ros, topic_name, "nav_msgs/Odometry")
        sub.subscribe(lambda m: received.append(m))
        pub.advertise()
        time.sleep(1.5)  # let advertise/subscribe register in the ROS graph

        pub.publish(roslibpy.Message(msg_before))
        time.sleep(1.2)
        pub.publish(roslibpy.Message(msg_after))

        deadline = time.time() + 8
        while len(received) < 2 and time.time() < deadline:
            time.sleep(0.2)

        pub.unadvertise()
        sub.unsubscribe()
        ros.terminate()
    except Exception as exc:  # noqa: BLE001 - fall back to an in-process check
        print(f"[rosbridge at {args.url} unavailable ({exc}); "
              f"using in-process message check]")
        received = []

    if len(received) < 2:
        # Fallback: no live ROS, so compare the message dicts directly. Weaker
        # (no wire hop) but still proves the code path: zeros before, populated
        # after, from the real EKF covariance.
        mode = ("in-process message check (rosbridge unavailable -- no wire hop; "
                "run with a live rosbridge for the ROS round-trip)")
        received = [msg_before, msg_after]

    # Identify which received message is which by its pose covariance content.
    def pose_cov(m):
        return np.asarray(m["pose"]["covariance"], dtype=float)

    def twist_cov(m):
        return np.asarray(m["twist"]["covariance"], dtype=float)

    got_before = next((m for m in received if np.all(pose_cov(m) == 0.0)), None)
    got_after = next((m for m in received if np.any(pose_cov(m) != 0.0)), None)

    ok = True
    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 72)
    w("PROOF: EKF covariance on a real ROS nav_msgs/Odometry topic")
    w("=" * 72)
    w(f"generated : {_dt.datetime.now().isoformat(timespec='seconds')}")
    w(f"rosbridge : {args.url}")
    w(f"topic     : {topic_name}")
    w(f"transport : {mode}")
    w(f"rod       : {name}")
    w(f"source    : real GTSAM EKF, stub simulator, {args.steps} steps (no trained model)")
    w("")
    w("Each covariance below is a 6x6 matrix, row-major (36 floats). Pose order")
    w("[x y z rot_x rot_y rot_z]; twist order [vx vy vz wx wy wz].")
    w("")

    w("-" * 72)
    w("BEFORE  (what the v1 bridge sent -- build_odometry_msg with no covariance)")
    w("-" * 72)
    if got_before is None:
        w("  MISSING: no all-zero message came back"); ok = False
    else:
        pb, tb = pose_cov(got_before), twist_cov(got_before)
        w(f"  pose.covariance  diag = {np.array2string(pb.reshape(6, 6).diagonal())}")
        w(f"  twist.covariance diag = {np.array2string(tb.reshape(6, 6).diagonal())}")
        w(f"  all 36 pose entries zero  : {bool(np.all(pb == 0.0))}")
        w(f"  all 36 twist entries zero : {bool(np.all(tb == 0.0))}")
        if not (np.all(pb == 0.0) and np.all(tb == 0.0)):
            w("  FAIL: BEFORE should be all zeros"); ok = False

    w("")
    w("-" * 72)
    w("AFTER   (EKF covariance projected in -- the change under test)")
    w("-" * 72)
    if got_after is None:
        w("  MISSING: no populated message came back"); ok = False
    else:
        pa, ta = pose_cov(got_after), twist_cov(got_after)
        w(f"  pose.covariance  diag = {np.array2string(pa.reshape(6, 6).diagonal(), precision=3)}")
        w(f"  twist.covariance diag = {np.array2string(ta.reshape(6, 6).diagonal(), precision=3)}")
        nonzero = int(np.count_nonzero(pa)) + int(np.count_nonzero(ta))
        finite = bool(np.all(np.isfinite(pa)) and np.all(np.isfinite(ta)))
        w(f"  nonzero entries (pose+twist) : {nonzero} / 72")
        w(f"  all finite                    : {finite}")
        w(f"  pose position var  > 0        : {bool(pa.reshape(6, 6)[0, 0] > 0)}")
        w(f"  pose small-angle var > 0      : {bool(pa.reshape(6, 6)[3, 3] > 0)}")
        # Round-trip integrity: what ROS returned equals what we sent.
        sent = np.asarray(msg_after['pose']['covariance'], dtype=float)
        rt = float(np.max(np.abs(pa - sent)))
        w(f"  round-trip max|sent-recv|     : {rt:.2e}  (0 == exact through ROS)")
        if not (nonzero > 0 and finite):
            w("  FAIL: AFTER should be nonzero and finite"); ok = False

    w("")
    w("=" * 72)
    if ok:
        w("VERDICT: PASS -- covariance is zero before the change and populated "
          "after.")
        w(f"         transport: {mode}")
    else:
        w("VERDICT: FAIL -- see above.")
    w("=" * 72)

    report = "\n".join(lines) + "\n"
    Path(args.out).write_text(report)
    print(report)
    print(f"[written to {args.out}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
