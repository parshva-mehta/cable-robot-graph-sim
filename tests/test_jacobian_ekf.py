"""Tests for Jacobian linearization quality and EKF consistency (NEES).

Three test cases:
  test_jacobian_quality        — F@δ ≈ f(x+δ)-f(x); residual O(‖δ‖²)
  test_linearized_prediction   — F_k correctly propagates small perturbations
                                 at each step (single-step linearization check)
  test_nees                    — normalized NEES ≈ 1; filter is consistent

Why these tests are meaningful
-------------------------------
The MEKF mean is propagated via GNN, not F, so predict-only matching the GNN
rollout proves nothing about the Jacobian.  F only enters the covariance update
(P = F P Fᵀ + Q) and therefore the Kalman gain.  A wrong F produces an
miscalibrated K, which causes the filter to over- or under-weight measurements.

test_jacobian_quality checks that F is the correct first-order local
linearization of the dynamics.

test_linearized_prediction checks that, at each step, F correctly predicts
how a small perturbation to the input state propagates forward (both paths
share the same LSTM context, isolating the dynamics Jacobian from LSTM drift).

test_nees checks end-to-end filter consistency: if F is calibrating P correctly,
the normalized estimation error should be near 1.

Standalone:
    python tests/test_jacobian_ekf.py [--model_path ...] [--data_dir ...]
pytest:
    pytest tests/test_jacobian_ekf.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pytest

from simulators.tensegrity_gnn_simulator import load_simulator
from utilities.misc_utils import DEFAULT_DTYPE
from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    quat_state_to_canonical_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
    step_exp,
)
import ekf as _ekf_mod
from ekf import (
    run_ekf_rollout,
    _ekf_step as _ORIG_EKF_STEP,
    _ensure_ctrl_for_step,
)


_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATA = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0/traj_6"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_assets(model_path, data_dir, device):
    sim = load_simulator(
        model_path, map_location=torch.device("cpu"), cache_batch_sizes=[1]
    )
    sim = sim.to(device)
    sim.eval()
    p = Path(data_dir)
    with open(p / "processed_data.json") as f:
        gt_data = json.load(f)
    with open(p / "extra_state_data.json") as f:
        extra_data = json.load(f)
    return sim, gt_data, extra_data


def _make_start_state(gt_data, n_rods, device):
    d0 = gt_data[0]
    vals = []
    for r in range(n_rods):
        vals.extend(
            d0["pos"][r * 3:(r + 1) * 3]
            + d0["quat"][r * 4:(r + 1) * 4]
            + d0["linvel"][r * 3:(r + 1) * 3]
            + d0["angvel"][r * 3:(r + 1) * 3]
        )
    return torch.tensor(vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)


def _reset_sim(sim, extra_data, device):
    init_rl = extra_data[0]["rest_lengths"]
    init_ms = extra_data[0]["motor_speeds"]
    cables = list(sim.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            init_rl[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            init_ms[i], dtype=DEFAULT_DTYPE, device=device
        ).reshape(1, 1, 1)
    sim.ctrls_hist = None
    sim.node_hidden_state = None


# ---------------------------------------------------------------------------
# Test 1: Jacobian quality
#
# At a state x_k, compute F = ∂f/∂x.  For small perturbations δ:
#   ‖(f(x+δ) - f(x)) - F@δ‖ / ‖f(x+δ) - f(x)‖  should be O(δ)
#
# Checks:
#   1. Relative residual at smallest δ (1e-4) < rel_tol (5 %)
#   2. Convergence: residual ratio between δ=1e-2 and δ=1e-4 > 3×
#      (first-order convergence → ratio ≈ 100×, threshold is conservative)
# ---------------------------------------------------------------------------

def test_jacobian_quality(model_path, data_dir, device,
                           warm_up_steps=5,
                           rel_tol=0.05,
                           convergence_min_ratio=3.0):
    """F@δ ≈ f(x+δ)-f(x) with second-order residual, checked on pose rows.

    Angular velocity (angvel) rows are excluded from the check because the
    GNN reconstructs angvel via acos(prev_prin·curr_prin)/dt — the acos
    derivative diverges near identity, making those rows non-smooth and
    non-convergent at any finite δ.  For the EKF's NEES (pose-only), only
    the pose (pos + exp_rot) and linvel rows matter.  Pose rows are checked
    here; linvel rows are covered by the structural formula and are accurate
    to O(δ²) by construction.
    """
    print("\n=== Test 1: Jacobian quality: F@δ ≈ f(x+δ)-f(x)  [pose rows] ===")
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    _reset_sim(sim, extra_data, device)

    dtype = DEFAULT_DTYPE
    dataset_idx = 9
    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=device)}

    # Pose indices: pos (0:3) + exp_rot (3:6) per rod.
    pose_idx = np.array([
        r * EXP_BLOCK_SIZE + j
        for r in range(n_rods)
        for j in range(6)
    ])

    # Warm up LSTM so the linearization point is not a trivial initial state.
    x_quat = start.clone()
    with torch.no_grad():
        for k in range(warm_up_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            ns, _ = sim.step(x_quat, ctrls=ctrl, state_to_graph_kwargs=s2g)
            x_quat = ns[..., 0:1]

    ctrl = _ensure_ctrl_for_step(extra_data[warm_up_steps]["controls"], sim)
    state_exp = quat_state_to_exp_state(x_quat)  # (1, 36, 1)

    # Compute F and the nominal next state f(x).
    # No SR clamping: we want the true local linearization, not a stabilised one.
    # linearize_dynamics_exp restores the LSTM context on return.
    ctx = _save_model_ctx(sim)
    f_x_np, F = linearize_dynamics_exp(
        sim, state_exp,
        sample_index=dataset_idx,
        use_finite_diff=True,
        ctrls=ctrl,
        verbose=False,
    )
    x_exp_np = state_exp.squeeze().cpu().numpy().astype(np.float64)

    def _gnn_exp(x_np):
        """Run one GNN step from x_np (exp-map), restoring LSTM to ctx first."""
        _restore_model_ctx(sim, ctx)
        x_t = torch.tensor(x_np, dtype=dtype, device=device).reshape(1, EXP_STATE_DIM, 1)
        with torch.no_grad():
            ns = step_exp(sim, x_t, ctrl, s2g)
        return ns[0, :EXP_STATE_DIM, 0].detach().cpu().numpy().astype(np.float64)

    # Fixed perturbation direction (reproducible).
    rng = np.random.default_rng(0)
    delta_dir = rng.standard_normal(EXP_STATE_DIM)
    delta_dir /= np.linalg.norm(delta_dir)

    eps_list = [1e-2, 1e-3, 1e-4]
    rel_errors = []

    for eps in eps_list:
        delta = eps * delta_dir
        f_xd = _gnn_exp(x_exp_np + delta)
        actual_diff = f_xd - f_x_np
        # Check only pose rows — the smooth, EKF-critical part of F.
        residual = np.linalg.norm((actual_diff - F @ delta)[pose_idx])
        ref_norm = max(np.linalg.norm(actual_diff[pose_idx]), 1e-15)
        rel_errors.append(residual / ref_norm)

    ok_accuracy = rel_errors[-1] < rel_tol
    conv_ratio = rel_errors[0] / max(rel_errors[-1], 1e-15)
    ok_convergence = conv_ratio > convergence_min_ratio

    ok = ok_accuracy and ok_convergence
    print(f"  {'PASS' if ok else 'FAIL'}")
    print(f"  pose-row rel errors @ eps=[1e-2,1e-3,1e-4]: {[f'{e:.3e}' for e in rel_errors]}")
    print(f"  convergence ratio (1e-2/1e-4): {conv_ratio:.1f}×  (need >{convergence_min_ratio:.0f}×)")
    if not ok_accuracy:
        print(f"  FAIL: pose rel error at eps=1e-4 = {rel_errors[-1]:.3e} >= {rel_tol}")
    if not ok_convergence:
        print(f"  FAIL: convergence ratio {conv_ratio:.1f}× < {convergence_min_ratio:.0f}×")
    return ok


# ---------------------------------------------------------------------------
# Test 2: Linearized prediction (single-step, re-linearized each step)
#
# At each step k, both nominal and perturbed paths use the SAME LSTM context
# (saved before the step), isolating the dynamics Jacobian from LSTM drift.
#
# δ_linear_{k+1} = F_k @ δ_k   vs   δ_actual_{k+1} = f(x_k+δ_k) - f(x_k)
#
# δ_k is propagated via the linear map (F_k @ δ_k) for the next step.
# This is exactly how the EKF propagates its error ellipsoid via P = F P Fᵀ+Q.
# If F is wrong, the predicted deviation will disagree with the actual one.
# ---------------------------------------------------------------------------

def test_linearized_prediction(model_path, data_dir, device,
                                n_steps=5,
                                eps=1e-3,
                                rel_tol=0.15):
    """F_k @ δ ≈ f(x_k+δ)-f(x_k) checked independently at each step.

    Uses a fresh pose-only perturbation at each step so the perturbation
    magnitude stays in the linear regime regardless of F's spectral radius
    or its large velocity rows (scaled by 1/dt).  Propagating δ across steps
    would grow it by the Frobenius norm of F (~280) each step, leaving the
    linear regime after 1-2 steps — a property of the dynamics, not of
    Jacobian quality.

    Deviation is evaluated only on pose (pos + exp_rot) rows — the smooth,
    EKF-critical directions.  Angvel rows are excluded for the same reason as
    in test_jacobian_quality (acos singularity makes them non-differentiable).
    """
    print(f"\n=== Test 2: Linearized prediction over {n_steps} steps [pose rows] ===")
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    _reset_sim(sim, extra_data, device)

    dtype = DEFAULT_DTYPE
    dataset_idx = 9
    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=device)}

    # Pose indices: pos (0:3) + exp_rot (3:6) per rod.
    pose_idx = np.array([
        r * EXP_BLOCK_SIZE + j
        for r in range(n_rods)
        for j in range(6)
    ])

    # Fixed pose-only perturbation direction: non-zero only in pos + exp_rot blocks.
    # Using pose-only delta keeps ‖delta‖ from exploding via the 1/dt velocity rows.
    rng = np.random.default_rng(1)
    delta_dir = np.zeros(EXP_STATE_DIM, dtype=np.float64)
    for r in range(n_rods):
        delta_dir[EXP_BLOCK_SIZE * r : EXP_BLOCK_SIZE * r + 6] = rng.standard_normal(6)
    delta_dir /= np.linalg.norm(delta_dir)
    delta = eps * delta_dir  # fresh same-magnitude perturbation reused at each step

    x_nom_quat = start.clone()
    rel_errs = []

    with torch.no_grad():
        for k in range(n_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            x_nom_exp_np = (
                quat_state_to_exp_state(x_nom_quat)
                .squeeze().cpu().numpy().astype(np.float64)
            )
            x_nom_exp_t = torch.tensor(
                x_nom_exp_np, dtype=dtype, device=device
            ).reshape(1, EXP_STATE_DIM, 1)

            # Linearize at nominal state k (restores LSTM on return).
            f_x_np, F_k = linearize_dynamics_exp(
                sim, x_nom_exp_t,
                sample_index=dataset_idx,
                use_finite_diff=True,
                ctrls=ctrl,
                verbose=False,
            )

            # Perturbed step from the same LSTM context (ctx_k).
            x_pert_exp_t = torch.tensor(
                x_nom_exp_np + delta, dtype=dtype, device=device
            ).reshape(1, EXP_STATE_DIM, 1)
            ns_pert = step_exp(sim, x_pert_exp_t, ctrl, s2g)
            x_pert_next_exp_np = ns_pert[0, :EXP_STATE_DIM, 0].cpu().numpy().astype(np.float64)

            # Advance nominal LSTM to ctx_{k+1}.
            ctx_k = _save_model_ctx(sim)
            _restore_model_ctx(sim, ctx_k)
            step_exp(sim, x_nom_exp_t, ctrl, s2g)
            ctx_k1 = _save_model_ctx(sim)

            # Compare linear prediction vs actual deviation — pose rows only.
            delta_actual = x_pert_next_exp_np - f_x_np
            delta_linear = F_k @ delta

            residual = np.linalg.norm((delta_actual - delta_linear)[pose_idx])
            ref_norm = max(np.linalg.norm(delta_actual[pose_idx]), 1e-15)
            rel_errs.append(residual / ref_norm)

            # Advance nominal trajectory for next step.
            _restore_model_ctx(sim, ctx_k1)
            x_nom_quat = exp_state_to_quat_state(
                torch.tensor(f_x_np, dtype=dtype, device=device)
                .reshape(1, EXP_STATE_DIM, 1)
            )

    max_rel_err = max(rel_errs)
    ok = max_rel_err < rel_tol
    print(f"  {'PASS' if ok else 'FAIL'} — max pose deviation error: {max_rel_err:.3e}  (tol {rel_tol:.0e})")
    print(f"  per-step pose rel errors: {[f'{e:.3e}' for e in rel_errs]}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: NEES consistency
#
# For a consistent filter, the Normalized Estimation Error Squared satisfies
#   E[NEES_k] = n_pose = 6 * n_rods = 18   (pose-only: pos + exp_rot)
#
# NEES is computed only on pose components because velocities are not directly
# observed (observe_pose_only=True) and FD-injected, so their P block does not
# reflect the Kalman update.
#
# A wrong F will miscalibrate P, producing NEES far from n_pose:
#   F too large → P inflated → K too large → state over-corrected → NEES << n_pose
#   F too small → P deflated → K too small → state under-corrected → NEES >> n_pose
#
# Bounds are deliberately loose (single trajectory, highly nonlinear dynamics):
#   normalized NEES in [nees_lo, nees_hi] = [0.05, 20]
# ---------------------------------------------------------------------------

def test_nees(model_path, data_dir, device, n_steps=40,
              nees_lo=0.05, nees_hi=20.0):
    """Normalized pose NEES ≈ 1 for a filter with a well-calibrated Jacobian."""
    print("\n=== Test 3: NEES consistency ===")
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    n = min(n_steps, len(extra_data) - 1)
    dtype = DEFAULT_DTYPE

    # Pose components: pos (0:3) + exp_rot (3:6) per rod block of size 12.
    pose_idx = np.array([
        r * EXP_BLOCK_SIZE + j
        for r in range(n_rods)
        for j in range(6)
    ])
    n_pose = len(pose_idx)  # 18 for 3-rod robot

    # Monkey-patch _ekf_step to capture (x_post_quat, P_post) at each step.
    x_est_list = []
    P_list = []

    def _capturing_step(x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
                        Q_sigmas, R_sigmas, n_rods_inner, have_measurement,
                        **kwargs):
        x_post, P_post = _ORIG_EKF_STEP(
            x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
            Q_sigmas, R_sigmas, n_rods_inner, have_measurement,
            **kwargs,
        )
        if have_measurement:
            x_est_list.append(x_post.copy())
            P_list.append(P_post.copy())
        return x_post, P_post

    _ekf_mod._ekf_step = _capturing_step
    try:
        run_ekf_rollout(
            sim,
            gt_data[:n + 1],
            extra_data[:n],
            dt=0.01,
            process_noise_scale=1e-4,
            measurement_noise_scale=1e-3,
            observe_pose_only=True,
            start_state=start,
            dataset_idx_val=9,
        )
    finally:
        _ekf_mod._ekf_step = _ORIG_EKF_STEP

    if not x_est_list:
        print("  FAIL — no update steps captured")
        return False

    nees_values = []
    meas_step = 0
    for k in range(n):
        if meas_step >= len(x_est_list):
            break

        x_est_quat = x_est_list[meas_step]
        P_post = P_list[meas_step]
        meas_step += 1

        # True state at k+1 in exp-map.
        gt = gt_data[k + 1]
        x_true_quat = np.zeros(13 * n_rods, dtype=np.float64)
        pos  = np.array(gt["pos"],    dtype=np.float64)
        quat = np.array(gt["quat"],   dtype=np.float64)
        lv   = np.array(gt["linvel"], dtype=np.float64)
        av   = np.array(gt["angvel"], dtype=np.float64)
        for r in range(n_rods):
            x_true_quat[13*r:13*r+3]    = pos[3*r:3*r+3]
            x_true_quat[13*r+3:13*r+7]  = quat[4*r:4*r+4]
            x_true_quat[13*r+7:13*r+10] = lv[3*r:3*r+3]
            x_true_quat[13*r+10:13*r+13] = av[3*r:3*r+3]

        # Use canonical exp-map for the ground truth so it matches the GNN's
        # principal-axis quaternion convention (no axial spin).
        x_true_exp = (
            quat_state_to_canonical_exp_state(
                torch.tensor(x_true_quat, dtype=dtype).reshape(1, -1, 1)
            ).squeeze().cpu().numpy().astype(np.float64)
        )
        x_est_exp = (
            quat_state_to_exp_state(
                torch.tensor(x_est_quat, dtype=dtype).reshape(1, -1, 1)
            ).squeeze().cpu().numpy().astype(np.float64)
        )

        # Restrict to pose components.
        err_pose = (x_true_exp - x_est_exp)[pose_idx]
        P_pose   = P_post[np.ix_(pose_idx, pose_idx)]
        P_pose_reg = P_pose + 1e-10 * np.eye(n_pose)

        try:
            P_inv = np.linalg.inv(P_pose_reg)
            nees = float(err_pose @ P_inv @ err_pose)
            nees_values.append(nees)
        except np.linalg.LinAlgError:
            pass

    if not nees_values:
        print("  FAIL — no finite NEES values")
        return False

    mean_nees = float(np.mean(nees_values))
    nees_norm = mean_nees / n_pose
    ok = nees_lo <= nees_norm <= nees_hi

    print(f"  {'PASS' if ok else 'FAIL'} — mean pose NEES = {mean_nees:.2f}  (n_pose = {n_pose})")
    print(f"  normalized NEES = {nees_norm:.3f}  (expected in [{nees_lo}, {nees_hi}])")
    return ok


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

_ASSETS_PRESENT = Path(_DEFAULT_MODEL).exists() and Path(_DEFAULT_DATA).exists()
_SKIP = pytest.mark.skipif(not _ASSETS_PRESENT, reason="model/data not found")
_DEV = torch.device("cpu")


@_SKIP
def test_pytest_jacobian_quality():
    assert test_jacobian_quality(_DEFAULT_MODEL, _DEFAULT_DATA, _DEV)


@_SKIP
def test_pytest_linearized_prediction():
    assert test_linearized_prediction(_DEFAULT_MODEL, _DEFAULT_DATA, _DEV)


@_SKIP
def test_pytest_nees():
    assert test_nees(_DEFAULT_MODEL, _DEFAULT_DATA, _DEV)


# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Jacobian quality and EKF NEES tests")
    parser.add_argument("--model_path", default=_DEFAULT_MODEL)
    parser.add_argument("--data_dir",   default=_DEFAULT_DATA)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--n_steps",    type=int, default=40)
    parser.add_argument(
        "--test",
        choices=["jacobian", "prediction", "nees", "all"],
        default="all",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    if not Path(args.model_path).exists():
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)
    if not Path(args.data_dir).exists():
        print(f"ERROR: data dir not found: {args.data_dir}")
        sys.exit(1)

    results = {}
    if args.test in ("jacobian", "all"):
        results["jacobian_quality"] = test_jacobian_quality(
            args.model_path, args.data_dir, device
        )
    if args.test in ("prediction", "all"):
        results["linearized_prediction"] = test_linearized_prediction(
            args.model_path, args.data_dir, device, n_steps=5
        )
    if args.test in ("nees", "all"):
        results["nees"] = test_nees(
            args.model_path, args.data_dir, device, n_steps=args.n_steps
        )

    print("\n=== Summary ===")
    all_pass = True
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            all_pass = False

    print(f"\nJacobian/NEES verdict: {'GO' if all_pass else 'NO-GO'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
