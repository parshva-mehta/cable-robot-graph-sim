"""Phase 3: sweep the dataset_idx conditioning input of the raw GNN rollout.

The GNN is conditioned on a one-hot dataset index (NUM_DATASETS=10); eval.py
hard-codes dataset_idx=9.  Per-rod drift profiling shows the rollout drifts
as a whole body, i.e. a systematic dynamics bias — exactly what the dataset
conditioning encodes.  This sweeps idx 0..9 on a truncated horizon and
reports translation MSE + swing rotation per horizon.

Run:
  conda run --no-capture-output -n cable_robot_gnn python -u \
      scripts/dataset_idx_sweep.py --device cpu
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
from scripts.rotation_swing_twist_diag import (
    DATA_ROOT, MODEL_PATH, run_raw_rollout, gt_poses_and_endpts, analyze_traj,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trajs', type=str, default='traj_0,traj_5,traj_6',
                        help='Distinct trajectories (1/7 and 2/8 are near-dups)')
    parser.add_argument('--idxs', type=str, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--out', type=str, default='logs/dataset_idx_sweep.npz')
    args = parser.parse_args()

    device = torch.device(args.device)
    traj_names = args.trajs.split(',')
    idxs = [int(i) for i in args.idxs.split(',')]
    horizons = [100, 500, args.max_steps]

    simulator = load_simulator(
        MODEL_PATH, map_location=torch.device('cpu'), cache_batch_sizes=[1]
    )
    simulator = simulator.to(device)
    simulator.eval()
    n_rods = len(simulator.robot.rigid_bodies)

    data = {}
    for name in traj_names:
        tdir = DATA_ROOT / name
        with open(tdir / 'processed_data.json') as f:
            gt_data = json.load(f)[:args.max_steps + 1]
        with open(tdir / 'extra_state_data.json') as f:
            extra_data = json.load(f)[:args.max_steps + 1]
        data[name] = (gt_data, extra_data,
                      *gt_poses_and_endpts(gt_data, n_rods))

    results = {}  # (traj, idx) -> metrics dict
    for idx in idxs:
        for name in traj_names:
            gt_data, extra_data, gt_poses, gt_end_pts = data[name]
            pred = run_raw_rollout(simulator, gt_data, extra_data, device, idx)
            m = analyze_traj(pred, gt_poses, gt_end_pts)
            results[(name, idx)] = m
            line = f'idx={idx} {name}: ' + '  '.join(
                f'com@{h}={m["com_sqerr"][:h].mean():.6f}' for h in horizons
            ) + '  ' + '  '.join(
                f'swing@{h}={m["swing"][:h].mean():.4f}' for h in horizons
            )
            print(line, flush=True)

    print('\n===== dataset_idx sweep summary =====')
    print(f'{"idx":<5}' + ''.join(f'{f"com@{h}":>13}' for h in horizons)
          + ''.join(f'{f"swing@{h}":>12}' for h in horizons)
          + '   (mean over trajs)')
    summary = {}
    for idx in idxs:
        com_means = [np.mean([results[(n, idx)]['com_sqerr'][:h].mean()
                              for n in traj_names]) for h in horizons]
        swing_means = [np.mean([results[(n, idx)]['swing'][:h].mean()
                                for n in traj_names]) for h in horizons]
        summary[idx] = (com_means, swing_means)
        print(f'{idx:<5}' + ''.join(f'{v:>13.6f}' for v in com_means)
              + ''.join(f'{v:>12.4f}' for v in swing_means))

    best = min(summary, key=lambda i: summary[i][0][-1])
    print(f'\nBest idx by com@{args.max_steps}: {best} '
          f'(com={summary[best][0][-1]:.6f}); '
          f'current hard-coded idx 9 com={summary[9][0][-1]:.6f}'
          if 9 in summary else '')

    np.savez_compressed(
        args.out,
        **{f'{n}_idx{i}_com': results[(n, i)]['com_sqerr']
           for n in traj_names for i in idxs},
        **{f'{n}_idx{i}_swing': results[(n, i)]['swing']
           for n in traj_names for i in idxs},
    )
    print(f'saved per-step arrays to {args.out}')


if __name__ == '__main__':
    main()
