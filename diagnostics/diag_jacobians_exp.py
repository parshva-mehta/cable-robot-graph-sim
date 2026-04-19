"""
Diagnostic: compute and log per-timestep Jacobians (exp-map state space) along a rollout.

Output layout inside --output-dir
-----------------------------------
jacobians_exp/
    exp_jac_full_jacs.txt    — t + full 36×36 Jacobian (flattened, 1296 values)
    exp_jac_rod1.txt         — t + full 12×12 Jacobian diagonal block for rod 1
    exp_jac_rod2.txt         — t + full 12×12 Jacobian diagonal block for rod 2
    exp_jac_rod3.txt         — t + full 12×12 Jacobian diagonal block for rod 3
    exp_jac_rot_rod1.txt     — t + 3×3 exp_rot sub-block for rod 1 (9 values)
    exp_jac_rot_rod2.txt     — t + 3×3 exp_rot sub-block for rod 2
    exp_jac_rot_rod3.txt     — t + 3×3 exp_rot sub-block for rod 3

State layout per rod (EXP_BLOCK_SIZE = 12):
  indices 0:3  — pos     (x, y, z)
  indices 3:6  — exp_rot (axis * angle, 3D)
  indices 6:9  — linvel  (vx, vy, vz)
  indices 9:12 — angvel  (wx, wy, wz)

Usage
-----
python diagnostics/diag_jacobians_exp.py \\
    --model   path/to/model.pt \\
    --input   rollout_states.txt \\
    --output-dir diagnostics \\
    --device cpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linearization import N_BODIES
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    linearize_dynamics_exp,
    quat_state_to_exp_state,
)

EXP_ROT_OFFSET = 3   # offset of exp_rot within an EXP_BLOCK_SIZE block
EXP_ROT_SIZE   = 3   # 3D exp-map rotation


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute exp-map GNN Jacobians along a rollout and write diagnostic files.'
    )
    p.add_argument('--model', default='../../tensegrity/models/best_rollout_model.pt',
                   help='Path to saved simulator (.pt)')
    p.add_argument('--input', default="../rollout_states.txt",
                   help='Path to rollout_states.txt (quat-space, 39D per row)')
    p.add_argument('--output-dir', default="../diagnostics",
                   help='Root output directory (jacobians_exp/ subfolder created inside)')
    p.add_argument('--device', default='cpu', help='torch device (cpu / cuda)')
    p.add_argument('--finite-diff', action='store_true',
                   help='Use central finite differences instead of torch.func.jacrev')
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
    states_quat = np.loadtxt(args.input, dtype=np.float64)
    if states_quat.ndim == 1:
        states_quat = states_quat[np.newaxis, :]
    T = len(states_quat)
    print(f'  {T} timesteps loaded')

    # Convert all quat states (39D) → exp-map states (36D) up front
    try:
        ref = next(model.parameters())
        dtype, dev = ref.dtype, ref.device
    except StopIteration:
        dtype, dev = torch.float32, torch.device('cpu')

    states_quat_t = torch.tensor(states_quat, dtype=dtype, device=dev).unsqueeze(-1)  # (T, 39, 1)
    states_exp_t  = quat_state_to_exp_state(states_quat_t)                            # (T, 36, 1)
    states_exp    = states_exp_t[:, :, 0].detach().cpu().numpy().astype(np.float64)   # (T, 36)

    jac_dir = Path(args.output_dir) / 'jacobians_exp'
    jac_dir.mkdir(parents=True, exist_ok=True)

    # Open output files
    f_full_jacs  = open(jac_dir / 'exp_jac_full_jacs.txt',  'w')
    rod_files    = [open(jac_dir / f'exp_jac_rod{i+1}.txt',     'w') for i in range(N_BODIES)]
    rot_files    = [open(jac_dir / f'exp_jac_rot_rod{i+1}.txt', 'w') for i in range(N_BODIES)]

    try:
        for t in range(T - 1):
            if t % 500 == 0:
                print(f'  timestep {t} / {T - 1}')

            state_exp = states_exp[t]   # (36,)

            _, J = linearize_dynamics_exp(
                model, state_exp, use_finite_diff=args.finite_diff
            )
            # J is (36, 36) float64 — naturally full-rank, no regularisation needed

            # exp_jac_full_jacs.txt — full 36×36 Jacobian flattened
            _write_row(f_full_jacs, np.concatenate([[t], J.flatten()]))

            # Per-rod files (1-indexed)
            for i in range(N_BODIES):
                b_start = i * EXP_BLOCK_SIZE
                b_end   = b_start + EXP_BLOCK_SIZE

                # Full 12×12 diagonal block for rod i
                J_rod = J[b_start:b_end, b_start:b_end]
                _write_row(rod_files[i], np.concatenate([[t], J_rod.flatten()]))

                # 3×3 exp_rot sub-block for rod i
                rs = b_start + EXP_ROT_OFFSET
                re = rs + EXP_ROT_SIZE
                J_rot = J[rs:re, rs:re]
                _write_row(rot_files[i], np.concatenate([[t], J_rot.flatten()]))

    finally:
        f_full_jacs.close()
        for f in rod_files + rot_files:
            f.close()

    print(f'Done.  Results written to {jac_dir}')


if __name__ == '__main__':
    main()
