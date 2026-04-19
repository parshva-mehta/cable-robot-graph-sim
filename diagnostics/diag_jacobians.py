"""
Diagnostic: compute and log per-timestep Jacobians along a rollout.

Output layout inside --output-dir
-----------------------------------
jacobians/
    jac_full_jacs.txt      — t + full 39×39 Jacobian (flattened, 1521 values)
    jac_full_quats.txt     — t + all 3 quaternion diagonal blocks concatenated
                             (3 × 4×4 = 48 values per row)
    jac_rod1.txt           — t + raw  quat block for rod 1 (16 values, 4×4)
    jac_rod2.txt           — t + raw  quat block for rod 2
    jac_rod3.txt           — t + raw  quat block for rod 3
    jac_reg_rod1.txt       — t + regularised quat block for rod 1
    jac_reg_rod2.txt       — t + regularised quat block for rod 2
    jac_reg_rod3.txt       — t + regularised quat block for rod 3

Usage
-----
python diagnostics/diag_jacobians.py \\
    --model   path/to/model.pt \\
    --input   rollout_states.txt \\
    --output-dir diagnostics \\
    --dt 0.01 \\
    --device cpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linearization import (
    N_BODIES,
    BLOCK_SIZE,
    QUAT_OFFSET,
    QUAT_SIZE,
    linearize_dynamics,
    _fix_jacobian_quaternion_rank,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute GNN Jacobians along a rollout and write diagnostic files.'
    )
    p.add_argument('--model', default='../../tensegrity/models/best_rollout_model.pt',
                   help='Path to saved simulator (.pt)')
    p.add_argument('--input', default="../rollout_states.txt", help='Path to rollout_states.txt')
    p.add_argument('--output-dir', default="../diagnostics",
                   help='Root output directory (jacobians/ subfolder created inside)')
    p.add_argument('--dt',    type=float, default=0.01,
                   help='Timestep (informational; model uses its own dt)')
    p.add_argument('--device', default='cpu', help='torch device (cpu / cuda)')
    return p.parse_args()


def _write_row(fh, values):
    fh.write(' '.join(f'{v:.8f}' for v in values) + '\n')


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f'Loading model from {args.model}')
    from simulators.tensegrity_gnn_simulator import load_simulator
    model = load_simulator(args.model, map_location=device, cache_batch_sizes=[1])
    model = model.to(device)
    model.eval()
    model.ctrls_hist        = None
    model.node_hidden_state = None

    print(f'Loading states from {args.input}')
    states = np.loadtxt(args.input, dtype=np.float64)
    if states.ndim == 1:
        states = states[np.newaxis, :]
    T = len(states)
    print(f'  {T} timesteps loaded')

    jac_dir = Path(args.output_dir) / 'jacobians'
    jac_dir.mkdir(parents=True, exist_ok=True)

    # Open output files
    f_full_jacs  = open(jac_dir / 'jac_full_jacs.txt',  'w')
    f_full_quats = open(jac_dir / 'jac_full_quats.txt', 'w')
    rod_files     = [open(jac_dir / f'jac_rod{i+1}.txt',     'w') for i in range(N_BODIES)]
    rod_reg_files = [open(jac_dir / f'jac_reg_rod{i+1}.txt', 'w') for i in range(N_BODIES)]

    try:
        for t in range(T - 1):
            if t % 500 == 0:
                print(f'  timestep {t} / {T - 1}')

            state = states[t]

            _, J = linearize_dynamics(model, state, dt=args.dt)
            J_reg         = _fix_jacobian_quaternion_rank(J, state, N_BODIES)

            # jac_full_jacs.txt — full 39×39 Jacobian flattened
            _write_row(f_full_jacs, np.concatenate([[t], J.flatten()]))

            # jac_full_quats.txt — all per-body quat blocks concatenated
            all_qblks = []
            for i in range(N_BODIES):
                qs = i * BLOCK_SIZE + QUAT_OFFSET
                qe = qs + QUAT_SIZE
                all_qblks.append(J[qs:qe, qs:qe].flatten())
            _write_row(f_full_quats, np.concatenate([[t], *all_qblks]))

            # per-rod files (1-indexed)
            for i in range(N_BODIES):
                qs = i * BLOCK_SIZE + QUAT_OFFSET
                qe = qs + QUAT_SIZE
                J_qblk     = J[qs:qe, qs:qe]
                J_reg_qblk = J_reg[qs:qe, qs:qe]
                _write_row(rod_files[i],     np.concatenate([[t], J_qblk.flatten()]))
                _write_row(rod_reg_files[i], np.concatenate([[t], J_reg_qblk.flatten()]))

    finally:
        f_full_jacs.close()
        f_full_quats.close()
        for f in rod_files + rod_reg_files:
            f.close()

    print(f'Done.  Results written to {jac_dir}')


if __name__ == '__main__':
    main()
