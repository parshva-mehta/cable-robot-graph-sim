"""Phase 1 diagnostic: split raw-GNN rotation error into swing vs axial twist.

Background
----------
The GNN pipeline reconstructs each rod's quaternion with
compute_quat_btwn_z_and_vec(principal_axis) — the shortest-arc rotation from
world-z to the rod axis — so predicted quats carry ZERO axial twist by
construction.  MuJoCo ground-truth quats carry the rod's real spin.  The
evaluate() rotation metric (compute_angle_btwn_quats on full quats) therefore
mixes a large twist artifact into the reported number.  Additionally that
metric does not fold the quaternion double cover (no |w|), so antipodal
quat pairs report 2*pi - theta instead of theta.

This script, per trajectory and per rod:
  1. runs the raw GNN rollout (same reset/ctrl logic as eval.py --mode raw),
  2. computes per-step: full-quat angle (old metric), folded full angle,
     swing angle (between principal axes), relative twist angle, GT
     self-twist and pred self-twist content,
  3. verifies the (w,x,y,z) convention against GT end_pts geometry and
     counts double-cover sign flips,
  4. saves per-step arrays to logs/swing_twist_diag/<traj>.npz for reuse,
  5. prints a summary table + the Phase 1 deliverable line.

Run:
  conda run --no-capture-output -n cable_robot_gnn python -u \
      scripts/rotation_swing_twist_diag.py --device cpu
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulators.tensegrity_gnn_simulator import load_simulator
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE

DATA_ROOT = Path(
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/"
    "data_sets/3bar_new_platform_high_friction/dataset_0"
)
MODEL_PATH = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/"
    "models/best_rollout_model.pt"
)


def unit(v, axis=-1, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def quat_to_rot_z(q):
    """Third column (body z-axis in world frame) of R(q); q is (..., 4) wxyz."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        2 * (w * w + z * z) - 1,
    ], axis=-1)


def quat_mul(a, b):
    """Hamilton product, (..., 4) wxyz."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def folded_angle(q):
    """Rotation angle of q folded to [0, pi] (double-cover safe)."""
    vec = np.linalg.norm(q[..., 1:], axis=-1)
    return 2 * np.arctan2(vec, np.abs(q[..., 0]))


def unfolded_angle(q):
    """Old-metric angle: 2*atan2(|vec|, w) in [0, 2*pi) — NOT double-cover safe."""
    vec = np.linalg.norm(q[..., 1:], axis=-1)
    return 2 * np.arctan2(vec, q[..., 0])


def twist_about_z(q_rel):
    """Twist component of q_rel about the body z axis, folded to [0, pi]."""
    return 2 * np.arctan2(np.abs(q_rel[..., 3]), np.abs(q_rel[..., 0]))


def minimal_quat_for_axis(z_axis):
    """compute_quat_btwn_z_and_vec in numpy: twistless quat taking world-z to axis."""
    a = unit(z_axis)
    q = np.stack([
        1.0 + a[..., 2],
        -a[..., 1],
        a[..., 0],
        np.zeros_like(a[..., 0]),
    ], axis=-1)
    return unit(q)


def run_raw_rollout(simulator, gt_data, extra_data, device, dataset_idx):
    """Mirror eval.py raw-mode reset + rollout; return (T, n_rods, 7) pred poses."""
    num_rods = len(simulator.robot.rigid_bodies)

    init_rest_lengths = extra_data[0]['rest_lengths']
    init_motor_speeds = extra_data[0]['motor_speeds']
    cables = list(simulator.robot.actuated_cables.values())
    for i, cable in enumerate(cables):
        cable.actuation_length = cable._rest_length - torch.tensor(
            init_rest_lengths[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
        cable.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    d0 = gt_data[0]
    pos, quat = d0['pos'], d0['quat']
    linvel, angvel = d0['linvel'], d0['angvel']
    state_vals = []
    for r in range(num_rods):
        state_vals.extend(
            pos[r * 3:(r + 1) * 3] + quat[r * 4:(r + 1) * 4]
            + linvel[r * 3:(r + 1) * 3] + angvel[r * 3:(r + 1) * 3]
        )
    start_state = torch.tensor(
        state_vals, dtype=DEFAULT_DTYPE
    ).reshape(1, -1, 1).to(device)

    ctrls = torch.tensor(
        [e['controls'] for e in extra_data], dtype=DEFAULT_DTYPE
    ).T.unsqueeze(0).to(device)

    with torch.no_grad():
        all_states, _, _ = simulator.run(
            curr_state=start_state,
            ctrls=ctrls,
            state_to_graph_kwargs={
                'dataset_idx': torch.tensor(
                    [[dataset_idx]], dtype=torch.long
                ).to(device)
            },
            show_progress=True,
        )

    # all_states rows are (1, 13*n_rods, 1); prepend t=0.
    states = [start_state] + list(all_states)
    poses = np.stack([
        s.reshape(num_rods, 13)[:, :7].cpu().numpy() for s in states
    ])
    return poses  # (T, n_rods, 7)


def gt_poses_and_endpts(gt_data, n_rods):
    T = len(gt_data)
    poses = np.zeros((T, n_rods, 7))
    end_pts = np.zeros((T, n_rods, 2, 3))
    for t, d in enumerate(gt_data):
        for r in range(n_rods):
            poses[t, r, :3] = d['pos'][r * 3:(r + 1) * 3]
            poses[t, r, 3:] = d['quat'][r * 4:(r + 1) * 4]
            # end_pts is a list of 2*n_rods 3-vectors: [rod0_pt0, rod0_pt1, ...]
            ep = d['end_pts']
            end_pts[t, r, 0] = ep[r * 2]
            end_pts[t, r, 1] = ep[r * 2 + 1]
    return poses, end_pts


def analyze_traj(pred_poses, gt_poses, gt_end_pts):
    """Per-step per-rod metrics; arrays are (T, n_rods, ...). Skips t=0."""
    T = min(len(pred_poses), len(gt_poses))
    qp = unit(pred_poses[1:T, :, 3:])
    qg = unit(gt_poses[1:T, :, 3:])

    # --- Convention check: GT quat z-axis vs end_pts geometry -------------
    z_gt = quat_to_rot_z(qg)
    axis_geom = unit(gt_end_pts[1:T, :, 1] - gt_end_pts[1:T, :, 0])
    conv_dot = np.sum(z_gt * axis_geom, axis=-1)

    # --- Old metric (unfolded) vs folded full-quat angle -------------------
    q_rel = quat_mul(quat_conj(qg), qp)
    ang_old = unfolded_angle(q_rel)
    ang_folded = folded_angle(q_rel)
    sign_flip = (np.sum(qg * qp, axis=-1) < 0)

    # --- Swing: angle between principal axes ------------------------------
    z_pred = quat_to_rot_z(qp)
    swing = np.arccos(np.clip(np.sum(z_gt * z_pred, axis=-1), -1.0, 1.0))

    # --- Twist: axial component of the relative rotation ------------------
    twist_rel = twist_about_z(q_rel)

    # --- Self-twist content: q = q_minimal(axis) ∘ q_twist ----------------
    def self_twist(q):
        z = quat_to_rot_z(q)
        q_min = minimal_quat_for_axis(z)
        q_tw = quat_mul(quat_conj(q_min), q)
        return twist_about_z(q_tw)

    gt_self_twist = self_twist(qg)
    pred_self_twist = self_twist(qp)

    return {
        'ang_old': ang_old,
        'ang_folded': ang_folded,
        'swing': swing,
        'twist_rel': twist_rel,
        'gt_self_twist': gt_self_twist,
        'pred_self_twist': pred_self_twist,
        'sign_flip': sign_flip,
        'conv_dot': conv_dot,
        'com_sqerr': ((pred_poses[1:T, :, :3] - gt_poses[1:T, :, :3]) ** 2
                      ).mean(axis=-1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset_idx', type=int, default=0,
                        help='0 is correct for dataset_0 eval data; 3-9 are '
                             'null embeddings (see scripts/dataset_idx_sweep.py)')
    parser.add_argument('--trajs', type=str, default=None,
                        help='Comma-separated traj names (default: all traj_*)')
    parser.add_argument('--out_dir', type=str, default='logs/swing_twist_diag')
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trajs:
        traj_names = args.trajs.split(',')
    else:
        traj_names = sorted(
            p.name for p in DATA_ROOT.iterdir()
            if p.is_dir() and p.name.startswith('traj_')
        )

    simulator = load_simulator(
        MODEL_PATH, map_location=torch.device('cpu'), cache_batch_sizes=[1]
    )
    simulator = simulator.to(device)
    simulator.eval()
    n_rods = len(simulator.robot.rigid_bodies)

    rows = []
    agg = {k: [] for k in ('ang_old', 'ang_folded', 'swing', 'twist_rel',
                           'gt_self_twist', 'pred_self_twist')}
    total_flips, total_steps = 0, 0
    conv_min = 1.0

    for name in traj_names:
        tdir = DATA_ROOT / name
        with open(tdir / 'processed_data.json') as f:
            gt_data = json.load(f)
        with open(tdir / 'extra_state_data.json') as f:
            extra_data = json.load(f)

        print(f'\n=== {name} ({len(gt_data)} steps) ===', flush=True)
        pred_poses = run_raw_rollout(
            simulator, gt_data, extra_data, device, args.dataset_idx
        )
        gt_poses, gt_end_pts = gt_poses_and_endpts(gt_data, n_rods)
        m = analyze_traj(pred_poses, gt_poses, gt_end_pts)

        np.savez_compressed(
            out_dir / f'{name}.npz',
            pred_poses=pred_poses, gt_poses=gt_poses,
            **{k: v for k, v in m.items()},
        )

        for k in agg:
            agg[k].append(m[k].mean())
        total_flips += int(m['sign_flip'].sum())
        total_steps += m['sign_flip'].size
        conv_min = min(conv_min, float(m['conv_dot'].min()))

        rows.append((
            name, m['ang_old'].mean(), m['ang_folded'].mean(),
            m['swing'].mean(), m['twist_rel'].mean(),
            m['gt_self_twist'].mean(), m['pred_self_twist'].mean(),
            m['sign_flip'].mean(), m['com_sqerr'].mean(),
        ))
        r = rows[-1]
        print(f'  old={r[1]:.4f}  folded={r[2]:.4f}  swing={r[3]:.4f}  '
              f'twist={r[4]:.4f}  gt_selftw={r[5]:.4f}  pred_selftw={r[6]:.4f}  '
              f'flip%={100*r[7]:.1f}  com_mse={r[8]:.6f}', flush=True)

    print('\n===== Summary (mean rad over steps x rods) =====')
    hdr = (f'{"traj":<10}{"old":>9}{"folded":>9}{"swing":>9}{"twist":>9}'
           f'{"gt_stw":>9}{"pred_stw":>9}{"flip%":>8}{"com_mse":>11}')
    print(hdr)
    for r in rows:
        print(f'{r[0]:<10}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>9.4f}'
              f'{r[5]:>9.4f}{r[6]:>9.4f}{100*r[7]:>8.1f}{r[8]:>11.6f}')

    mean_old = np.mean(agg['ang_old'])
    mean_folded = np.mean(agg['ang_folded'])
    mean_swing = np.mean(agg['swing'])
    pct_artifact = 100.0 * (mean_old - mean_swing) / mean_old

    print(f'\nConvention check: min |dot(z(q_gt), axis(end_pts))| pairing = '
          f'{conv_min:.6f} (should stay ~+1)')
    print(f'Double-cover sign flips: {100.0*total_flips/total_steps:.1f}% of '
          f'rod-steps (inflate the old unfolded metric)')
    print(f'Pred self-twist (should be ~0): {np.mean(agg["pred_self_twist"]):.6f} rad')
    print(f'GT   self-twist (real spin)  : {np.mean(agg["gt_self_twist"]):.6f} rad')
    print(f'\nDELIVERABLE: {pct_artifact:.1f}% of the reported rotation error '
          f'({mean_old:.4f} rad) is twist/double-cover artifact; '
          f'true swing error = {mean_swing:.4f} rad '
          f'(folded full-quat = {mean_folded:.4f} rad)')


if __name__ == '__main__':
    main()
