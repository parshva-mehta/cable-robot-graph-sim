"""
Diagnostic: exp-map linearization vs. quaternion T-projection linearization.

Implements the exp-map wrapper approach (convert exp→quat before model.step,
quat→exp after) and compares it against the current quaternion-based method
from linearization.py.

Diagnostics printed per timestep
---------------------------------
  rank_J_raw   — rank of raw 39×39 quat Jacobian (should be ≤36)
  rank_J_fix   — rank of T-projection-fixed 39×39 Jacobian (should be 39)
  rank_J_exp   — rank of 36×36 exp-map Jacobian (should be 36)
  sr_raw       — spectral radius of J_raw
  sr_fix       — spectral radius of J_fix (ambient)
  sr_exp       — spectral radius of J_exp
  pred_diff    — max |next_state_quat_from_exp - next_state_from_quat_method|
  roundtrip    — max exp-map round-trip error: |exp→quat→exp - exp|
  theta_max    — largest rotation angle across all rods (singularity proximity)
  jac_equiv    — max |J_exp - T_out @ J_fix @ T_in| (measures mathematical equivalence)

Aggregate summary printed at the end.

Output files (inside --output-dir/exp_jac/)
--------------------------------------------
  summary.txt         — per-timestep scalar metrics (one row per t)
  pred_diff.txt       — per-timestep per-rod predicted-state differences
  exp_jac_full.txt    — t + full 36×36 exp Jacobian flattened (1297 values)
  exp_jac_rod1.txt    — t + 3×3 exp-rot diagonal block for rod 1
  exp_jac_rod2.txt    — t + 3×3 exp-rot diagonal block for rod 2
  exp_jac_rod3.txt    — t + 3×3 exp-rot diagonal block for rod 3

Usage
-----
  cd cable-robot-graph-sim
  python diagnostics/diag_exp_vs_quat.py \\
      --model   ../../tensegrity/models/best_rollout_model.pt \\
      --input   rollout_states.txt \\
      --output-dir diagnostics \\
      --max-steps 50 \\
      --device cpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linearization import (
    N_BODIES,
    BLOCK_SIZE,
    STATE_DIM,
    linearize_dynamics,
    _fix_jacobian_quaternion_rank,
    _build_tangent_projections,
)
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    step_exp,
    linearize_dynamics_exp as _linearize_dynamics_exp,
)


def linearize_dynamics_exp(model, state_exp_np, dtype, dev,
                            sample_index=0, use_finite_diff=False):
    """Thin wrapper so the diagnostic can pass dtype/dev explicitly."""
    # The shared module infers dtype/dev from model parameters; dtype/dev args
    # are accepted here for API compatibility but the module handles them.
    return _linearize_dynamics_exp(
        model, state_exp_np,
        sample_index=sample_index,
        use_finite_diff=use_finite_diff,
    )

    return next_exp_np, J_np


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_angles_from_exp_state(state_exp_np: np.ndarray) -> np.ndarray:
    """Return the rotation angle (radians) for each rod from an exp-map state."""
    s = state_exp_np.reshape(N_BODIES, EXP_BLOCK_SIZE)
    angles = np.linalg.norm(s[:, 3:6], axis=1)   # one angle per rod
    return angles


def _safe_rank(mat: np.ndarray, tol: float = 1e-6) -> int:
    sv = np.linalg.svd(mat, compute_uv=False)
    return int(np.sum(sv > tol))


def _spectral_radius(mat: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(mat))))


def _print_header():
    cols = [
        f"{'t':>4}",
        f"{'rnk_raw':>8}", f"{'rnk_fix':>8}", f"{'rnk_exp':>8}",
        f"{'sr_raw':>9}",  f"{'sr_fix':>9}",  f"{'sr_exp':>9}",
        f"{'pred_diff':>10}", f"{'roundtrip':>10}",
        f"{'θ_max/π':>9}",  f"{'jac_equiv':>10}",
    ]
    print("  ".join(cols))
    print("-" * 110)


def _fmt_row(t, rnk_raw, rnk_fix, rnk_exp,
             sr_raw, sr_fix, sr_exp,
             pred_diff, roundtrip, theta_max_frac, jac_equiv):
    return (
        f"{t:4d}  "
        f"{rnk_raw:8d}  {rnk_fix:8d}  {rnk_exp:8d}  "
        f"{sr_raw:9.4f}  {sr_fix:9.4f}  {sr_exp:9.4f}  "
        f"{pred_diff:10.2e}  {roundtrip:10.2e}  "
        f"{theta_max_frac:9.4f}  {jac_equiv:10.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare exp-map vs. quaternion linearization diagnostics.'
    )
    p.add_argument('--model', default='../../tensegrity/models/best_rollout_model.pt',
                   help='Path to saved TensegrityGNNSimulator (.pt)')
    p.add_argument('--input', default='../rollout_states.txt',
                   help='Rollout states file (quat format, 39 values/row)')
    p.add_argument('--output-dir', default='.',
                   help='Root output dir; exp_vs_quat/ subfolder created inside')
    p.add_argument('--max-steps', type=int, default=20,
                   help='Max timesteps to process (0 = all)')
    p.add_argument('--finite-diff', action='store_true',
                   help='Use central finite differences instead of jacrev')
    p.add_argument('--sample-index', type=int, default=9,
                   help='dataset_idx forwarded to graph processor')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args  = parse_args()
    dev   = torch.device(args.device)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f'Loading model from {args.model}')
    from simulators.tensegrity_gnn_simulator import load_simulator
    model = load_simulator(args.model, map_location=dev, cache_batch_sizes=[1])
    model = model.to(dev)
    model.eval()
    model.ctrls_hist        = None
    model.node_hidden_state = None

    try:
        ref   = next(model.parameters())
        dtype = ref.dtype
    except StopIteration:
        dtype = torch.float32

    # ── Load rollout states (quat format) ────────────────────────────────────
    print(f'Loading states from {args.input}')
    states_quat = np.loadtxt(args.input, dtype=np.float64)
    if states_quat.ndim == 1:
        states_quat = states_quat[np.newaxis, :]
    T = len(states_quat)
    print(f'  {T} timesteps loaded (quat format, dim={states_quat.shape[1]})')

    max_t = (T - 1) if args.max_steps <= 0 else min(args.max_steps, T - 1)

    # ── Pre-convert all states to exp-map ────────────────────────────────────
    states_exp = np.zeros((T, EXP_STATE_DIM), dtype=np.float64)
    for i in range(T):
        s_t = torch.tensor(states_quat[i], dtype=dtype, device=dev).reshape(1, STATE_DIM, 1)
        states_exp[i] = quat_state_to_exp_state(s_t)[0, :, 0].cpu().numpy()

    print(f'  Converted to exp-map format (dim={EXP_STATE_DIM})\n')

    # ── Output files ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / 'exp_jac'
    out_dir.mkdir(parents=True, exist_ok=True)
    f_summary       = open(out_dir / 'summary.txt',      'w')
    f_pred_diff     = open(out_dir / 'pred_diff.txt',    'w')
    f_exp_full      = open(out_dir / 'exp_jac_full.txt', 'w')
    rod_jac_files   = [open(out_dir / f'exp_jac_rod{r+1}.txt', 'w')
                       for r in range(N_BODIES)]

    # Column headers
    f_summary.write(
        't rank_raw rank_fix rank_exp sr_raw sr_fix sr_exp '
        'pred_diff roundtrip theta_max_frac jac_equiv\n'
    )
    f_pred_diff.write('t  ' + '  '.join(f'rod{r+1}_diff' for r in range(N_BODIES)) + '\n')
    f_exp_full.write('# t  J_exp (36x36 flattened, 1296 values)\n')
    for r in range(N_BODIES):
        rod_jac_files[r].write(f'# t  J_exp rot block rod {r+1} (3x3 = 9 values)\n')

    # ── Per-timestep loop ────────────────────────────────────────────────────
    print(f'Running {max_t} timestep(s) (finite_diff={args.finite_diff})\n')
    _print_header()

    accum = {k: [] for k in [
        'rnk_raw', 'rnk_fix', 'rnk_exp',
        'sr_raw', 'sr_fix', 'sr_exp',
        'pred_diff', 'roundtrip', 'theta_max_frac', 'jac_equiv',
    ]}

    for t in range(max_t):
        s_quat_np = states_quat[t]   # (39,)
        s_exp_np  = states_exp[t]    # (36,)

        # ── Quat method (current) ────────────────────────────────────────────
        model.ctrls_hist = None
        model.node_hidden_state = None
        next_quat_np, J_raw = linearize_dynamics(
            model, s_quat_np,
            sample_index=args.sample_index,
            use_finite_diff=args.finite_diff,
        )

        J_fix = _fix_jacobian_quaternion_rank(J_raw, s_quat_np, N_BODIES)

        # ── Exp-map method (new) ─────────────────────────────────────────────
        model.ctrls_hist = None
        model.node_hidden_state = None
        next_exp_np, J_exp = linearize_dynamics_exp(
            model, s_exp_np, dtype, dev,
            sample_index=args.sample_index,
            use_finite_diff=args.finite_diff,
        )

        # ── Metric 1: Jacobian rank ──────────────────────────────────────────
        rnk_raw = _safe_rank(J_raw)
        rnk_fix = _safe_rank(J_fix)
        rnk_exp = _safe_rank(J_exp)

        # ── Metric 2: Spectral radius ────────────────────────────────────────
        sr_raw  = _spectral_radius(J_raw)
        sr_fix  = _spectral_radius(J_fix)
        sr_exp  = _spectral_radius(J_exp)

        # ── Metric 3: Forward prediction agreement ───────────────────────────
        # Convert next_exp → quat and compare to next_quat from quat method
        next_exp_t  = torch.tensor(next_exp_np, dtype=dtype, device=dev).reshape(1, EXP_STATE_DIM, 1)
        next_quat_from_exp = exp_state_to_quat_state(next_exp_t)[0, :, 0].cpu().numpy()
        pred_diff   = float(np.max(np.abs(next_quat_from_exp - next_quat_np[:STATE_DIM])))

        # Per-rod positional prediction differences for file output
        rod_diffs = []
        for r in range(N_BODIES):
            a = r * BLOCK_SIZE
            rod_diffs.append(float(np.max(np.abs(
                next_quat_from_exp[a:a+3] - next_quat_np[a:a+3]
            ))))

        # ── Metric 4: Round-trip fidelity exp→quat→exp ───────────────────────
        s_exp_t = torch.tensor(s_exp_np, dtype=dtype, device=dev).reshape(1, EXP_STATE_DIM, 1)
        s_rt    = quat_state_to_exp_state(exp_state_to_quat_state(s_exp_t))
        roundtrip = float(torch.max(torch.abs(s_rt - s_exp_t)).item())

        # ── Metric 5: Singularity proximity  θ_max / π ──────────────────────
        angles         = _rotation_angles_from_exp_state(s_exp_np)
        theta_max_frac = float(np.max(angles) / np.pi)

        # ── Metric 6: Jacobian equivalence  J_exp ≈ T_out @ J_fix @ T_in ────
        # T_out/T_in are built from current state (tangent at linearization point).
        # The equivalence holds exactly only when prev & next states are identical,
        # so this measures the linearization-point approximation error.
        T_out, T_in = _build_tangent_projections(s_quat_np, N_BODIES)
        J_projected  = T_out @ J_fix @ T_in    # (36, 36)
        jac_equiv    = float(np.max(np.abs(J_exp - J_projected)))

        # ── Accumulate ───────────────────────────────────────────────────────
        for k, v in zip(accum.keys(),
                        [rnk_raw, rnk_fix, rnk_exp,
                         sr_raw, sr_fix, sr_exp,
                         pred_diff, roundtrip, theta_max_frac, jac_equiv]):
            accum[k].append(v)

        # ── Print row ────────────────────────────────────────────────────────
        row = _fmt_row(t, rnk_raw, rnk_fix, rnk_exp,
                       sr_raw, sr_fix, sr_exp,
                       pred_diff, roundtrip, theta_max_frac, jac_equiv)
        print(row)

        # ── Write to files ───────────────────────────────────────────────────
        f_summary.write(
            f'{t} {rnk_raw} {rnk_fix} {rnk_exp} '
            f'{sr_raw:.6f} {sr_fix:.6f} {sr_exp:.6f} '
            f'{pred_diff:.6e} {roundtrip:.6e} '
            f'{theta_max_frac:.6f} {jac_equiv:.6e}\n'
        )
        f_pred_diff.write(
            f'{t}  ' + '  '.join(f'{d:.6e}' for d in rod_diffs) + '\n'
        )

        # Full 36×36 exp Jacobian (flattened)
        f_exp_full.write(
            ' '.join([f'{t}'] + [f'{v:.8f}' for v in J_exp.flatten()]) + '\n'
        )

        # Per-rod 3×3 exp-rotation diagonal blocks
        # Rotation indices in the 12D per-rod block are 3:6, so within the
        # 36D flattened Jacobian the diagonal block for rod r starts at
        # row = r*EXP_BLOCK_SIZE+3, col = r*EXP_BLOCK_SIZE+3
        for r in range(N_BODIES):
            rs = r * EXP_BLOCK_SIZE + 3   # row/col start of 3×3 rot block
            re = rs + 3
            blk = J_exp[rs:re, rs:re]
            rod_jac_files[r].write(
                ' '.join([f'{t}'] + [f'{v:.8f}' for v in blk.flatten()]) + '\n'
            )

    # ── Aggregate summary ────────────────────────────────────────────────────
    print('\n' + '=' * 110)
    print('AGGREGATE SUMMARY')
    print('=' * 110)

    labels = {
        'rnk_raw':        ('Jacobian rank — raw quat 39×39',         '.1f'),
        'rnk_fix':        ('Jacobian rank — T-proj fixed 39×39',     '.1f'),
        'rnk_exp':        ('Jacobian rank — exp-map 36×36',          '.1f'),
        'sr_raw':         ('Spectral radius — raw quat',             '.4f'),
        'sr_fix':         ('Spectral radius — T-proj fixed',         '.4f'),
        'sr_exp':         ('Spectral radius — exp-map',              '.4f'),
        'pred_diff':      ('Forward prediction agreement (max |Δ|)', '.2e'),
        'roundtrip':      ('Round-trip error exp→quat→exp (max)',    '.2e'),
        'theta_max_frac': ('Max rotation angle / π (0=identity, '
                           '1=singularity)',                         '.4f'),
        'jac_equiv':      ('Jacobian equivalence max |J_exp - '
                           'T_out@J_fix@T_in|',                     '.2e'),
    }

    for key, (label, fmt) in labels.items():
        vals = accum[key]
        mn   = np.mean(vals)
        mx   = np.max(vals)
        print(f'  {label}')
        fmt_str = f'    mean={mn:{fmt}}  max={mx:{fmt}}'
        print(fmt_str)

    print('\nNotes:')
    print('  rank_raw  < 39 confirms quaternion rank deficiency (expected: 36).')
    print('  rank_exp == 36 confirms exp-map has no rank deficiency.')
    print('  pred_diff ~ 0 means both methods predict the same next state.')
    print('  roundtrip ~ 0 means exp↔quat conversion is numerically stable.')
    print('  theta_max/π < 1 means no singularity encountered (π = 180°).')
    print('  jac_equiv ~ 0 means the two approaches are mathematically equivalent')
    print('            at the linearization point (non-zero = 1st-order approx gap).')

    f_summary.close()
    f_pred_diff.close()
    f_exp_full.close()
    for f in rod_jac_files:
        f.close()
    print(f'\nFiles written to {out_dir}/')


if __name__ == '__main__':
    main()
