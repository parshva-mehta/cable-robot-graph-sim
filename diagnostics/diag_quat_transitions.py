"""
Diagnostic: extract and validate per-body quaternion transitions from a rollout.

Output layout inside --output-dir
-----------------------------------
quaternions/
    quat_rod1.txt   — transitions for rod 1
    quat_rod2.txt   — transitions for rod 2
    quat_rod3.txt   — transitions for rod 3

Each row: t  qw_t  qx_t  qy_t  qz_t  qw_{t+1}  qx_{t+1}  qy_{t+1}  qz_{t+1}

Norms are validated; rows where |‖q‖ − 1| > 1e-4 are reported on stdout.
With --sign-fix, sign is flipped whenever dot(q_t, q_{t−1}) < 0.

Usage
-----
python diagnostics/diag_quat_transitions.py \\
    --input  rollout_states.txt \\
    --output-dir diagnostics \\
    [--sign-fix]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linearization import N_BODIES, BLOCK_SIZE, QUAT_OFFSET, QUAT_SIZE


def parse_args():
    p = argparse.ArgumentParser(
        description='Extract per-body quaternion transitions and validate norms.'
    )
    p.add_argument('--input',      default="../rollout_states.txt")
    p.add_argument('--output-dir', default="../diagnostics",
                   help='Root output directory (quaternions/ subfolder created inside)')
    p.add_argument('--sign-fix', action='store_true',
                   help='Flip sign when dot(q_t, q_{t-1}) < 0 (double-cover fix)')
    return p.parse_args()


def main():
    args = parse_args()

    print(f'Loading states from {args.input}')
    states = np.loadtxt(args.input, dtype=np.float64)
    if states.ndim == 1:
        states = states[np.newaxis, :]
    T = len(states)
    print(f'  {T} timesteps loaded')

    quat_dir = Path(args.output_dir) / 'quaternions'
    quat_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_BODIES):
        rod_num = i + 1   # 1-indexed to match image
        qs = i * BLOCK_SIZE + QUAT_OFFSET
        qe = qs + QUAT_SIZE
        quats = states[:, qs:qe].copy()   # (T, 4)  [w, x, y, z]

        # Norm validation
        norms    = np.linalg.norm(quats, axis=1)
        bad_mask = np.abs(norms - 1.0) > 1e-4
        n_bad    = int(bad_mask.sum())
        if n_bad:
            first = int(np.argmax(bad_mask))
            print(f'  Rod {rod_num}: {n_bad} timestep(s) with |norm-1| > 1e-4 '
                  f'(first at t={first}, norm={norms[first]:.6f})')
        else:
            print(f'  Rod {rod_num}: all quaternion norms within 1e-4 of 1.0')

        # Optional double-cover sign fix
        if args.sign_fix:
            for t in range(1, T):
                if float(np.dot(quats[t], quats[t - 1])) < 0.0:
                    quats[t] = -quats[t]

        # Write transition file
        # Columns: t  qw_t qx_t qy_t qz_t  qw_{t+1} qx_{t+1} qy_{t+1} qz_{t+1}
        out_path = quat_dir / f'quat_rod{rod_num}.txt'
        with open(out_path, 'w') as f:
            for t in range(T - 1):
                row = np.concatenate([[t], quats[t], quats[t + 1]])
                f.write(' '.join(f'{v:.8f}' for v in row) + '\n')

        print(f'  Rod {rod_num}: wrote {T - 1} transitions → {out_path.name}')

    print('Done.')


if __name__ == '__main__':
    main()
