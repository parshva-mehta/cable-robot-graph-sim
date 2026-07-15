"""B1: re-test GNN dynamics transfer to scaled real data with dataset_idx fixed.

The 2026-07-02 NO-GO (GNN 12-57x worse than a frozen-state baseline on
new_platform_processed real data) predates the discovery that dataset_idx 9
is a null embedding (idx 3-9 identical; training used 0/1/2).  This re-runs
the same protocol sweeping idx {0,1,2,9}:

  For start frames k (strided), initialize the sim state from real frame k,
  roll open-loop with per-frame controls substepped at dt_sim=0.01
  (n_sub = round(dt_frame/0.01), dt clipped to [0.005, 1.0]), and record
  COM position error (m, mean over rods) against the real pose at horizons
  of 1, 5, and 20 frames.  Baseline: frozen state at frame k.

Run:
  conda run --no-capture-output -n cable_robot_gnn python -u \
      scripts/real_transfer_retest.py --device cpu
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
from utilities.misc_utils import DEFAULT_DTYPE
from scripts.rotation_swing_twist_diag import MODEL_PATH, quat_to_rot_z, unit

REAL_ROOT = Path(
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/"
    "data_sets/new_platform_processed/dataset_0"
)
DT_SIM = 0.01


def state_from_frame(d, n_rods, device):
    vals = []
    for r in range(n_rods):
        vals.extend(
            d['pos'][r * 3:(r + 1) * 3] + d['quat'][r * 4:(r + 1) * 4]
            + d['linvel'][r * 3:(r + 1) * 3] + d['angvel'][r * 3:(r + 1) * 3]
        )
    return torch.tensor(vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)


def frame_pos(d, n_rods):
    return np.array(d['pos']).reshape(n_rods, 3)


def rollout_window(sim, gt, extra, k, n_frames, dataset_idx, device, n_rods):
    """Roll from frame k for n_frames real frames; return pred pos at each
    frame boundary as {frame_offset: (n_rods, 3)}."""
    # Reset cable/motor/recurrent state to frame k
    cables = list(sim.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            extra[k]['rest_lengths'][i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            extra[k]['motor_speeds'][i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
    sim.ctrls_hist = None
    sim.node_hidden_state = None

    curr = state_from_frame(gt[k], n_rods, device)
    out = {}
    idx_t = torch.tensor([[dataset_idx]], dtype=torch.long).to(device)
    with torch.no_grad():
        for j in range(n_frames):
            f = k + j
            dt = float(np.clip(gt[f + 1]['time'] - gt[f]['time'], 0.005, 1.0))
            n_sub = max(1, int(round(dt / DT_SIM)))
            ctrls = torch.tensor(
                extra[f]['controls'], dtype=DEFAULT_DTYPE
            ).reshape(1, -1, 1).repeat(1, 1, n_sub).to(device)
            states, _, _ = sim.run(
                curr_state=curr, ctrls=ctrls,
                state_to_graph_kwargs={'dataset_idx': idx_t},
                show_progress=False,
            )
            curr = states[-1]
            pose = curr.reshape(n_rods, 13).cpu().numpy()
            out[j + 1] = pose[:, :3].copy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--traj', type=str, default='real_2026-06-17_11-42-54')
    parser.add_argument('--idxs', type=str, default='0,1,2,9')
    parser.add_argument('--horizons', type=str, default='1,5,20')
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    idxs = [int(i) for i in args.idxs.split(',')]
    horizons = [int(h) for h in args.horizons.split(',')]
    max_h = max(horizons)

    sim = load_simulator(
        MODEL_PATH, map_location=torch.device('cpu'), cache_batch_sizes=[1]
    )
    sim = sim.to(device)
    sim.eval()
    n_rods = len(sim.robot.rigid_bodies)

    tdir = REAL_ROOT / args.traj
    with open(tdir / 'processed_data.json') as f:
        gt = json.load(f)
    with open(tdir / 'extra_state_data.json') as f:
        extra = json.load(f)
    T = min(len(gt), len(extra))
    starts = list(range(0, T - max_h - 1, args.stride))
    print(f'{args.traj}: {T} frames, {len(starts)} start windows, '
          f'horizons {horizons} frames', flush=True)

    # Frozen baseline (idx-independent)
    frozen = {h: [] for h in horizons}
    for k in starts:
        p0 = frame_pos(gt[k], n_rods)
        for h in horizons:
            pt = frame_pos(gt[k + h], n_rods)
            frozen[h].append(np.linalg.norm(pt - p0, axis=1).mean())

    print(f'\n{"idx":<6}' + ''.join(
        f'{f"gnn@{h}f":>11}{f"frozen@{h}f":>12}{f"ratio@{h}f":>11}'
        for h in horizons))
    for idx in idxs:
        errs = {h: [] for h in horizons}
        for k in starts:
            preds = rollout_window(sim, gt, extra, k, max_h, idx,
                                   device, n_rods)
            for h in horizons:
                pt = frame_pos(gt[k + h], n_rods)
                errs[h].append(np.linalg.norm(preds[h] - pt, axis=1).mean())
        row = f'{idx:<6}'
        for h in horizons:
            g, fz = np.mean(errs[h]), np.mean(frozen[h])
            row += f'{g:>11.4f}{fz:>12.4f}{g / fz:>11.2f}'
        print(row, flush=True)

    print('\nratio < 1.0 = GNN beats "robot frozen" baseline '
          '(2026-07-02 NO-GO had ratios 12/35/57 at idx 9).')


if __name__ == '__main__':
    main()
