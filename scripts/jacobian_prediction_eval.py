"""Jacobian prediction diagnostic over a full trajectory.

At each step k along a GNN rollout, this script:
  1. Linearizes the GNN at the current state x_k:
       f_ref, F_k = linearize_dynamics_exp(sim, x_k)
  2. Runs GNN(x_k) independently (same LSTM context) to verify that
       f_ref matches the direct GNN output (trivial error ≈ 0).
  3. Runs GNN(x_k + eps*v) from the same LSTM context to measure
       how well the linear prediction f_ref + F_k @ (eps*v)
       approximates the true perturbed output (linear prediction error).
  4. Records Jacobian statistics: spectral radius, condition number,
     Frobenius norm.

The trivial error (step 2) verifies that linearize_dynamics_exp and the
rollout GNN call use identical LSTM context.  The linear prediction error
(step 3) measures how well F_k captures the local sensitivity of the
dynamics — the same quantity that drives covariance propagation in the MEKF.

Usage:
    python scripts/jacobian_prediction_eval.py
    python scripts/jacobian_prediction_eval.py --n_steps 60 --eps 1e-3 --save_plots
    python scripts/jacobian_prediction_eval.py --use_autodiff
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulators.tensegrity_gnn_simulator import load_simulator
from utilities.misc_utils import DEFAULT_DTYPE
from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
    step_exp,
)
from ekf import _ensure_ctrl_for_step

_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATA = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0/traj_6"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load(model_path, data_dir, device):
    sim = load_simulator(model_path, map_location=torch.device('cpu'), cache_batch_sizes=[1])
    sim = sim.to(device).eval()
    p = Path(data_dir)
    with open(p / 'processed_data.json') as f:
        gt = json.load(f)
    with open(p / 'extra_state_data.json') as f:
        extra = json.load(f)
    return sim, gt, extra


def _reset_sim(sim, extra, device):
    cables = list(sim.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            extra[0]['rest_lengths'][i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            extra[0]['motor_speeds'][i], dtype=DEFAULT_DTYPE, device=device
        ).reshape(1, 1, 1)
    sim.ctrls_hist = None
    sim.node_hidden_state = None


def _make_start_state(gt_data, n_rods, device):
    d0 = gt_data[0]
    vals = []
    for r in range(n_rods):
        vals.extend(
            d0['pos'][r * 3:(r + 1) * 3]
            + d0['quat'][r * 4:(r + 1) * 4]
            + d0['linvel'][r * 3:(r + 1) * 3]
            + d0['angvel'][r * 3:(r + 1) * 3]
        )
    return torch.tensor(vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def run_jacobian_prediction_eval(
    sim, gt_data, extra_data, device,
    n_steps: int = None,
    eps: float = 1e-3,
    dataset_idx: int = 9,
    use_finite_diff: bool = True,
    max_spectral_radius: float = 1.0,
) -> dict:
    """Run the Jacobian prediction diagnostic over n_steps rollout steps.

    Context management
    ------------------
    At step k, let ctx_k be the LSTM state before the step.
    linearize_dynamics_exp internally saves/restores ctx_k and returns
    (f_ref, F_k).  The script then:
      - calls step_exp from ctx_k to get x_{k+1} and advance to ctx_{k+1}
      - restores ctx_k for the perturbed evaluation
      - restores ctx_{k+1} (nominal) for the next iteration

    This isolates the Jacobian test from LSTM drift while keeping the
    nominal rollout trajectory correct.

    Returns
    -------
    dict with keys:
      trivial_err      (n_steps,) float64  — ||f_ref - GNN(x_k)||
      linear_err       (n_steps,) float64  — ||f_ref + F@δ - GNN(x_k+δ)||
      rel_linear_err   (n_steps,) float64  — linear_err / ||GNN(x_k+δ) - f_ref||
      spectral_radius  (n_steps,) float64  — max|eig(F)|
      cond_F           (n_steps,) float64  — condition number of F
      norm_F           (n_steps,) float64  — Frobenius norm of F
    """
    dtype = DEFAULT_DTYPE
    n_rods = len(sim.robot.rigid_bodies)
    s2g = {'dataset_idx': torch.tensor([[dataset_idx]], dtype=torch.long, device=device)}

    x_k_quat = _make_start_state(gt_data, n_rods, device)

    if n_steps is None:
        n_steps = len(extra_data)
    n_steps = min(n_steps, len(extra_data))

    # Fixed position-only perturbation direction (avoids contact-tangent singularity).
    rng = np.random.default_rng(42)
    dv = np.zeros(EXP_STATE_DIM, dtype=np.float64)
    for r in range(EXP_STATE_DIM // EXP_BLOCK_SIZE):
        base = r * EXP_BLOCK_SIZE
        dv[base:base + 3] = rng.standard_normal(3)
    dv /= np.linalg.norm(dv)
    delta = eps * dv  # (36,)

    keys = ['trivial_err', 'linear_err', 'rel_linear_err', 'rel_pose_err',
            'spectral_radius', 'cond_F', 'norm_F']
    metrics = {k: [] for k in keys}

    print(f"\nRunning {n_steps} steps  (eps={eps:.0e}, {'FD' if use_finite_diff else 'autodiff'})")
    print(f"{'step':>5}  {'trivial_err':>12}  {'rel_all':>8}  {'rel_pose':>9}  "
          f"{'SR(F)':>8}  {'cond(F)':>10}  {'||F||_F':>10}")
    print("-" * 80)

    for k in range(n_steps):
        ctrl = _ensure_ctrl_for_step(extra_data[k]['controls'], sim)
        x_k_exp = quat_state_to_exp_state(x_k_quat)          # (1, 36, 1)
        x_k_exp_np = x_k_exp[0, :EXP_STATE_DIM, 0].cpu().numpy().astype(np.float64)

        # ctx_k: LSTM state before the step.
        # linearize_dynamics_exp saves/restores ctx_k internally, so after it
        # returns the LSTM is exactly at ctx_k.
        ctx_k = _save_model_ctx(sim)

        # Step 1: linearize at x_k.
        f_ref_np, F_k = linearize_dynamics_exp(
            sim, x_k_exp,
            sample_index=dataset_idx,
            use_finite_diff=use_finite_diff,
            ctrls=ctrl,
            max_spectral_radius=max_spectral_radius,
            verbose=False,
        )
        # LSTM is at ctx_k (restored by linearize_dynamics_exp).

        # Step 2: run GNN(x_k) from ctx_k — trivial check and rollout advance.
        with torch.no_grad():
            x_next_exp = step_exp(sim, x_k_exp, ctrl, s2g)   # advances to ctx_{k+1}
        x_next_np = x_next_exp[0, :EXP_STATE_DIM, 0].cpu().numpy().astype(np.float64)
        ctx_k1 = _save_model_ctx(sim)

        trivial_err = float(np.linalg.norm(f_ref_np - x_next_np))

        # Step 3: perturbed GNN(x_k + δ) from ctx_k.
        _restore_model_ctx(sim, ctx_k)
        x_pert_exp = torch.tensor(
            x_k_exp_np + delta, dtype=dtype, device=device
        ).reshape(1, EXP_STATE_DIM, 1)
        with torch.no_grad():
            x_pert_next_exp = step_exp(sim, x_pert_exp, ctrl, s2g)
        x_pert_next_np = x_pert_next_exp[0, :EXP_STATE_DIM, 0].cpu().numpy().astype(np.float64)

        # Linear prediction: f_ref + F @ δ.
        linear_pred_np = f_ref_np + F_k @ delta
        actual_diff = x_pert_next_np - f_ref_np   # true nonlinear deviation
        F_delta      = F_k @ delta                  # linearized prediction of deviation
        linear_err   = float(np.linalg.norm(linear_pred_np - x_pert_next_np))
        ref_norm     = max(float(np.linalg.norm(actual_diff)), 1e-15)
        rel_linear_err = linear_err / ref_norm

        # Pose-only (pos + exp_rot) relative error — velocity excluded because:
        # the angvel Jacobian is unreliable near stationary states due to the
        # acos(prev·curr)/dt singularity; pos+rot blocks are well-conditioned.
        pose_idx = np.array([r * EXP_BLOCK_SIZE + i
                              for r in range(n_rods) for i in range(6)])
        pose_actual = actual_diff[pose_idx]
        pose_pred   = F_delta[pose_idx]
        pose_norm   = max(float(np.linalg.norm(pose_actual)), 1e-15)
        rel_pose_err = float(np.linalg.norm(pose_pred - pose_actual)) / pose_norm

        # Jacobian statistics.
        sr = float(np.max(np.abs(np.linalg.eigvals(F_k))))
        try:
            sv = np.linalg.svd(F_k, compute_uv=False)
            cond_F = float(sv[0] / max(sv[-1], 1e-15))
        except np.linalg.LinAlgError:
            cond_F = float('inf')
        norm_F = float(np.linalg.norm(F_k, 'fro'))

        metrics['trivial_err'].append(trivial_err)
        metrics['linear_err'].append(linear_err)
        metrics['rel_linear_err'].append(rel_linear_err)
        metrics['rel_pose_err'].append(rel_pose_err)
        metrics['spectral_radius'].append(sr)
        metrics['cond_F'].append(cond_F)
        metrics['norm_F'].append(norm_F)

        print(f"{k:>5}  {trivial_err:>12.2e}  {rel_linear_err:>8.3f}  "
              f"{rel_pose_err:>9.3f}  {sr:>8.4f}  {cond_F:>10.2e}  {norm_F:>10.4f}")

        # Restore ctx_{k+1} (nominal) for the next iteration.
        _restore_model_ctx(sim, ctx_k1)
        x_k_quat = exp_state_to_quat_state(x_next_exp)

    for k in keys:
        metrics[k] = np.array(metrics[k], dtype=np.float64)

    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(metrics):
    n = len(metrics['trivial_err'])
    print(f"\n{'=' * 60}")
    print(f"Summary over {n} steps")
    print(f"{'=' * 60}")
    print(f"  trivial error   mean={np.mean(metrics['trivial_err']):.2e}  "
          f"max={np.max(metrics['trivial_err']):.2e}  "
          f"(should be machine-precision ≈ 0)")
    print(f"  linear pred err mean={np.mean(metrics['linear_err']):.4e}  "
          f"max={np.max(metrics['linear_err']):.4e}")
    print(f"  rel err (all)   mean={np.mean(metrics['rel_linear_err']):.3f}  "
          f"max={np.max(metrics['rel_linear_err']):.3f}  "
          f"(dominated by angvel acos singularity)")
    print(f"  rel err (pose)  mean={np.mean(metrics['rel_pose_err']):.3f}  "
          f"max={np.max(metrics['rel_pose_err']):.3f}  "
          f"(pos+rot only — MEKF-relevant; < 0.1 = good)")
    print(f"  spectral radius mean={np.mean(metrics['spectral_radius']):.4f}  "
          f"max={np.max(metrics['spectral_radius']):.4f}")
    print(f"  cond(F)         mean={np.mean(metrics['cond_F']):.2e}  "
          f"max={np.max(metrics['cond_F']):.2e}")
    print(f"  ||F||_F         mean={np.mean(metrics['norm_F']):.4f}  "
          f"max={np.max(metrics['norm_F']):.4f}")

    trivial_ok = np.max(metrics['trivial_err']) < 1e-5
    pose_ok    = np.mean(metrics['rel_pose_err']) < 0.1
    sr_ok      = np.max(metrics['spectral_radius']) <= 1.01
    print(f"\n  trivial error ≈ 0:          {'PASS' if trivial_ok else 'FAIL'}")
    print(f"  mean rel pose err < 0.1:    {'PASS' if pose_ok else 'FAIL'}")
    print(f"  spectral radius ≤ 1:        {'PASS' if sr_ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(metrics, out_dir='.', eps=1e-3):
    out_dir = Path(out_dir)
    steps = np.arange(len(metrics['trivial_err']))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Jacobian prediction diagnostic  (eps={eps:.0e})', fontsize=13)

    # (a) Trivial error: ||f_ref - GNN(x_k)||
    ax = axes[0, 0]
    ax.semilogy(steps, np.maximum(metrics['trivial_err'], 1e-20), 'b-o', ms=3)
    ax.axhline(1e-5, color='r', linestyle='--', label='1e-5 threshold', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('||f_ref - GNN(x_k)||')
    ax.set_title('(a) Trivial error\n(linearize_dynamics_exp vs direct GNN call)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # (b) Relative linear prediction error — all vs pose-only
    ax = axes[0, 1]
    ax.plot(steps, metrics['rel_linear_err'], 'g-o', ms=3, label='all (pos+rot+vel)', alpha=0.6)
    ax.plot(steps, metrics['rel_pose_err'],   'b-s', ms=3, label='pose only (pos+rot)')
    ax.axhline(0.1, color='r', linestyle='--', label='0.1 threshold', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('||F·δ - actual|| / ||actual||')
    ax.set_title('(b) Relative linear prediction error\n(pose-only is MEKF-relevant)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Spectral radius of F over trajectory
    ax = axes[1, 0]
    ax.plot(steps, metrics['spectral_radius'], 'm-o', ms=3)
    ax.axhline(1.0, color='r', linestyle='--', label='SR = 1.0 (stability)', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('max |eigenvalue|')
    ax.set_title('(c) Spectral radius of F_k\n(clamped to 1.0 by default)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Condition number and Frobenius norm
    ax = axes[1, 1]
    color1, color2 = 'tab:blue', 'tab:orange'
    ax2 = ax.twinx()
    ax.semilogy(steps, metrics['cond_F'], color=color1, marker='o', ms=3, label='cond(F)')
    ax2.plot(steps, metrics['norm_F'], color=color2, marker='s', ms=3, label='||F||_F')
    ax.set_xlabel('Step')
    ax.set_ylabel('cond(F)', color=color1)
    ax2.set_ylabel('||F||_F', color=color2)
    ax.set_title('(d) Condition number and Frobenius norm of F_k')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = out_dir / 'jacobian_prediction_eval.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Jacobian prediction diagnostic')
    parser.add_argument('--model_path',  default=_DEFAULT_MODEL)
    parser.add_argument('--data_dir',    default=_DEFAULT_DATA)
    parser.add_argument('--device',      default='cpu')
    parser.add_argument('--n_steps',     type=int, default=30,
                        help='Number of rollout steps to evaluate')
    parser.add_argument('--eps',         type=float, default=1e-3,
                        help='Perturbation magnitude for linear prediction test')
    parser.add_argument('--dataset_idx', type=int, default=9)
    parser.add_argument('--use_autodiff', action='store_true', default=False,
                        help='Use autodiff Jacobian (default: finite difference)')
    parser.add_argument('--max_sr',      type=float, default=1.0,
                        help='Spectral radius clamp for F')
    parser.add_argument('--save_plots',  action='store_true', default=False)
    parser.add_argument('--out_dir',     default='.')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == 'cpu' or not torch.cuda.is_available()
        else args.device
    )

    if not Path(args.model_path).exists():
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)
    if not Path(args.data_dir).exists():
        print(f"ERROR: data dir not found: {args.data_dir}")
        sys.exit(1)

    print(f"Loading model from {args.model_path}")
    sim, gt_data, extra_data = _load(args.model_path, args.data_dir, device)
    _reset_sim(sim, extra_data, device)
    print(f"Model loaded.  n_rods={len(sim.robot.rigid_bodies)}  device={device}")

    metrics = run_jacobian_prediction_eval(
        sim, gt_data, extra_data, device,
        n_steps=args.n_steps,
        eps=args.eps,
        dataset_idx=args.dataset_idx,
        use_finite_diff=not args.use_autodiff,
        max_spectral_radius=args.max_sr,
    )

    print_summary(metrics)

    if args.save_plots:
        save_plots(metrics, out_dir=args.out_dir, eps=args.eps)


if __name__ == '__main__':
    main()
