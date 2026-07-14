"""Convert raw ``new_platform_data`` trajectories into the trainable format.

Raw (per trajectory dir):
    config.json                     camera intrinsics/extrinsics, color->rod map, geometry
    data/NNNN.json                  per-frame telemetry (motors, sensors, imu, endcaps, header.secs)
    poses-proposed/{red,green,blue}.npy   (N,4,4) per-rod homogeneous transform (camera frame)
    poses-proposed/{0..5}_pos.npy         (N,3)   per-endcap position (camera frame)

Output (per segment dir), matching the schema the training engine consumes
(see nn_training/real_tensegrity_gnn_training_engine.py):
    processed_data.json   list of frames: {time, end_pts(6,3), pos(9), quat(12), linvel(9), angvel(9)}
    extra_state_data.json list of frames: {time, dt, controls(6), rest_lengths(6),
                                           motor_speeds(6), sensor_lens(9)}

The training engine derives rod COM/orientation from ``end_pts`` (NOT pos/quat), and for
real trajectories filters frames by comparing ``sensor_lens`` against endcap distances --
so both fields must be in sim units and geometrically consistent.

Frame / scale handling (see plan_newDS.md Phase 0.2, resolved empirically):
  1. camera -> world via config['cam_extr'] (proper rotation, gravity-aligned; rod length
     is preserved at the real 0.36 m).
  2. scale x (SIM_ROD_LEN / config['rod_length'])  ~= 8.19  -> sim units.
  3. z-lift so the trajectory's lowest endcap sits at the sim ground-contact height, and
     x,y recentred so the first frame's COM is at the origin (x,y are dynamically
     irrelevant -- the robot rolls freely over the ground plane).
No residual rotation is fit: cam_extr already yields a gravity-aligned frame (verified: the
transformed Z span equals one rod length and endcap-pair distance is a constant 0.36 m).

Control sign (plan Phase 0.3) is still unverified: control = motors.speed / max_speed, with
an optional global sign flip via --control-sign.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

# --- sim reference geometry (simulators/configs/3_bar_gnn_sim_config.json) ---
SIM_ROD_LEN = 2.95          # sim rod endcap-pair distance (verified)
SIM_SPHERE_RADIUS = 0.175   # sim endcap sphere radius == resting endcap-centre height
DT_SIM = 0.01

# endcap index -> rod: node_to_color [red,red,green,green,blue,blue],
# color_to_rod {red:[0,1], green:[2,3], blue:[4,5]}  =>  pairs (0,1)(2,3)(4,5) -> rods 0,1,2
ROD_COLORS = ["red", "green", "blue"]


def _apply(T, pts):
    """Apply 4x4 homogeneous transform T to points (..., 3)."""
    h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1)
    return (h @ T.T)[..., :3]


def _quat_from_R(R):
    """Rotation matrix (3,3) -> quaternion (qw,qx,qy,qz), canonical qw>=0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def _angvel_from_R(R0, R1, dt):
    """Body-frame-agnostic angular velocity (world) between rotations, via log map."""
    dR = R1 @ R0.T
    cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([dR[2, 1] - dR[1, 2],
                     dR[0, 2] - dR[2, 0],
                     dR[1, 0] - dR[0, 1]]) / (2.0 * np.sin(theta))
    return axis * (theta / dt)


def load_raw(traj_dir):
    traj_dir = Path(traj_dir)
    cfg = json.load(open(traj_dir / "config.json"))
    frame_files = sorted(glob.glob(str(traj_dir / "data" / "*.json")))
    frames = [json.load(open(f)) for f in frame_files]
    pose_dir = traj_dir / "poses-proposed"
    endcaps_cam = np.stack([np.load(pose_dir / f"{i}_pos.npy") for i in range(6)], axis=1)  # (N,6,3)
    rot_cam = np.stack([np.load(pose_dir / f"{c}.npy")[:, :3, :3] for c in ROD_COLORS], axis=1)  # (N,3,3,3)
    n = min(len(frames), endcaps_cam.shape[0], rot_cam.shape[0])
    return cfg, frames[:n], endcaps_cam[:n], rot_cam[:n]


def compute_transform(cfg, endcaps_cam, ground_z):
    """Return (scale, cam_extr, offset) mapping camera endcaps -> sim world."""
    scale = SIM_ROD_LEN / cfg["rod_length"]
    E = np.array(cfg["cam_extr"])
    ec_world = _apply(E, endcaps_cam)          # (N,6,3) gravity-aligned, real metres
    ec_scaled = ec_world * scale               # sim units
    z_lift = ground_z - ec_scaled[..., 2].min()
    com0 = ec_scaled[0].reshape(3, 2, 3).mean(axis=1).mean(axis=0)  # frame-0 mean rod COM
    offset = np.array([-com0[0], -com0[1], z_lift])
    return scale, E, offset


def convert_trajectory(cfg, frames, endcaps_cam, rot_cam, scale, E, offset, control_sign):
    """Build (processed_frames, extra_frames) lists for a contiguous segment."""
    R_frame = E[:3, :3]                                   # rotation part (proper rotation)
    ec_sim = _apply(E, endcaps_cam) * scale + offset      # (N,6,3) sim world
    rot_sim = np.einsum("ij,nrjk->nrik", R_frame, rot_cam)  # (N,3,3,3) sim-world rod rotations

    n = len(frames)
    secs = np.array([f["header"]["secs"] for f in frames])

    # Snap timestamps to the fixed 0.01 s sim grid (plan_newDS.md Phase 0.1 strategy B:
    # substep to fixed dt, two-sided). Each real interval becomes a whole number of 0.01 s
    # steps, min 1, so the engine's round(t/0.01) indexing is exact and control/measurement
    # step counts stay consistent. Sub-0.01 bursts are stretched to one step.
    times = [0.0]
    for i in range(1, n):
        step = max(1, int(round((secs[i] - secs[i - 1]) / DT_SIM)))
        times.append(times[-1] + step * DT_SIM)
    secs = np.array(times)

    # per-frame COM (3 rods) and quat
    coms = np.stack([(ec_sim[:, 2 * k] + ec_sim[:, 2 * k + 1]) / 2 for k in range(3)], axis=1)  # (N,3,3)
    quats = np.stack([[_quat_from_R(rot_sim[i, k]) for k in range(3)] for i in range(n)])        # (N,3,4)

    processed, extra = [], []
    for i in range(n):
        dt_prev = secs[i] - secs[i - 1] if i > 0 else (secs[1] - secs[0] if n > 1 else DT_SIM)
        dt_next = secs[i + 1] - secs[i] if i < n - 1 else dt_prev
        if i > 0:
            linvel = (coms[i] - coms[i - 1]) / dt_prev                       # (3,3)
            angvel = np.stack([_angvel_from_R(rot_sim[i - 1, k], rot_sim[i, k], dt_prev)
                               for k in range(3)])
        else:
            linvel = np.zeros((3, 3))
            angvel = np.zeros((3, 3))

        processed.append({
            "time": float(secs[i]),
            "end_pts": ec_sim[i].tolist(),                                   # (6,3)
            "pos": coms[i].reshape(-1).tolist(),                            # (9,)
            "quat": quats[i].reshape(-1).tolist(),                          # (12,)
            "linvel": linvel.reshape(-1).tolist(),                          # (9,)
            "angvel": angvel.reshape(-1).tolist(),                          # (9,)
        })

        m = frames[i]["motors"]
        s = frames[i]["sensors"]
        max_speed = cfg.get("max_speed", 70.0)
        extra.append({
            "time": float(secs[i]),
            "dt": float(dt_next),
            "controls": [control_sign * m[str(j)]["speed"] / max_speed for j in range(6)],
            "rest_lengths": [s[str(j)]["length"] / 1000.0 * scale for j in range(6)],
            "motor_speeds": [m[str(j)]["speed"] for j in range(6)],
            "sensor_lens": [s[str(j)]["length"] / 1000.0 * scale for j in range(9)],
        })
    return processed, extra


def split_segments(frames, endcaps_cam, rot_cam, gap_split, min_frames):
    """Split a trajectory at time gaps > gap_split (s). Yields index ranges."""
    secs = np.array([f["header"]["secs"] for f in frames])
    dt = np.diff(secs)
    breaks = [0] + [i + 1 for i, d in enumerate(dt) if d > gap_split] + [len(frames)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a >= min_frames:
            yield a, b


def convert(raw_dir, out_dir, ground_z=SIM_SPHERE_RADIUS, gap_split=5.0,
            min_frames=20, control_sign=1):
    cfg, frames, endcaps_cam, rot_cam = load_raw(raw_dir)
    scale, E, offset = compute_transform(cfg, endcaps_cam, ground_z)
    name = Path(raw_dir).name
    segs = list(split_segments(frames, endcaps_cam, rot_cam, gap_split, min_frames))
    written = []
    for si, (a, b) in enumerate(segs):
        suffix = f"_seg{si}" if len(segs) > 1 else ""
        seg_out = Path(out_dir) / f"real_{name}{suffix}"
        seg_out.mkdir(parents=True, exist_ok=True)
        processed, extra = convert_trajectory(
            cfg, frames[a:b], endcaps_cam[a:b], rot_cam[a:b],
            scale, E, offset, control_sign)
        json.dump(processed, open(seg_out / "processed_data.json", "w"))
        json.dump(extra, open(seg_out / "extra_state_data.json", "w"))
        written.append((str(seg_out), b - a))
        print(f"  wrote {seg_out.name}: {b - a} frames  (scale={scale:.4f})")
    if not written:
        print(f"  WARNING: no segment >= {min_frames} frames in {name}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw trajectory dir (has config.json, data/, poses-proposed/)")
    ap.add_argument("--out", required=True, help="output dataset dir (segments written as subdirs)")
    ap.add_argument("--ground-z", type=float, default=SIM_SPHERE_RADIUS)
    ap.add_argument("--gap-split", type=float, default=5.0, help="split trajectory at time gaps > this (s)")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--control-sign", type=int, default=1, choices=[-1, 1])
    args = ap.parse_args()
    print(f"Converting {args.raw}")
    convert(args.raw, args.out, args.ground_z, args.gap_split, args.min_frames, args.control_sign)


if __name__ == "__main__":
    main()
