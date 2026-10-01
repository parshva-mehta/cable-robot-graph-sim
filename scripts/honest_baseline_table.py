"""Phase 2: honest raw-GNN-vs-GT baseline table from saved diagnostic arrays.

Reads the per-trajectory .npz files written by rotation_swing_twist_diag.py
(logs/swing_twist_diag/) and prints translation MSE + swing rotation error
at several rollout horizons, per trajectory.  No model runs needed.

Metrics at horizon H are means over steps 1..H (x rods):
  com_mse : mean squared COM error (m^2)  [same definition as eval.evaluate]
  swing   : mean principal-axis angle error (rad)
  legacy  : old unfolded full-quat angle, for contrast (rad)

Run:  python scripts/honest_baseline_table.py [--horizons 10,50,...]
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_dir', type=str, default='logs/swing_twist_diag')
    parser.add_argument('--horizons', type=str, default='10,50,100,500,1000,full')
    args = parser.parse_args()

    files = sorted(Path(args.diag_dir).glob('traj_*.npz'))
    if not files:
        raise SystemExit(f'no npz files in {args.diag_dir}; '
                         'run rotation_swing_twist_diag.py first')
    horizons = args.horizons.split(',')

    print(f'{"traj":<9}{"metric":<9}' + ''.join(f'{h:>12}' for h in horizons))
    per_h_com = {h: [] for h in horizons}
    per_h_swing = {h: [] for h in horizons}
    per_h_legacy = {h: [] for h in horizons}

    for f in files:
        d = np.load(f)
        com, swing, legacy = d['com_sqerr'], d['swing'], d['ang_old']
        T = com.shape[0]

        def at(arr, h):
            n = T if h == 'full' else min(int(h), T)
            return arr[:n].mean()

        com_row = {h: at(com, h) for h in horizons}
        swing_row = {h: at(swing, h) for h in horizons}
        legacy_row = {h: at(legacy, h) for h in horizons}
        for h in horizons:
            per_h_com[h].append(com_row[h])
            per_h_swing[h].append(swing_row[h])
            per_h_legacy[h].append(legacy_row[h])

        name = f.stem
        print(f'{name:<9}{"com_mse":<9}'
              + ''.join(f'{com_row[h]:>12.6f}' for h in horizons))
        print(f'{"":<9}{"swing":<9}'
              + ''.join(f'{swing_row[h]:>12.4f}' for h in horizons))
        print(f'{"":<9}{"legacy":<9}'
              + ''.join(f'{legacy_row[h]:>12.4f}' for h in horizons))

    print('-' * (18 + 12 * len(horizons)))
    print(f'{"MEAN":<9}{"com_mse":<9}'
          + ''.join(f'{np.mean(per_h_com[h]):>12.6f}' for h in horizons))
    print(f'{"":<9}{"swing":<9}'
          + ''.join(f'{np.mean(per_h_swing[h]):>12.4f}' for h in horizons))
    print(f'{"":<9}{"legacy":<9}'
          + ''.join(f'{np.mean(per_h_legacy[h]):>12.4f}' for h in horizons))


if __name__ == '__main__':
    main()
