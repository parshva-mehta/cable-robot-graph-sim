"""Phase 3–5 GTSAM-MEKF refiner evaluation: sweeps, diagnostics, and ablations.

Parallel to scripts/mekf_refiner_eval.py but uses ekf_gtsam instead of ekf.
All phases, helper functions, and CLI options are identical; only the filter
backend changes from manual Kalman to GTSAM GaussianFactorGraph.

Usage:
    python scripts/gtsam_refiner_eval.py                         # all phases
    python scripts/gtsam_refiner_eval.py --phases 3
    python scripts/gtsam_refiner_eval.py --phases 4 --save_plots
    python scripts/gtsam_refiner_eval.py --phases 5
    python scripts/gtsam_refiner_eval.py --phases 3,4,5 --traj traj_6,traj_3
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from simulators.tensegrity_gnn_simulator import load_simulator
from utilities.misc_utils import DEFAULT_DTYPE
from utilities import torch_quaternion
from linearization_exp import (
    EXP_BLOCK_SIZE,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
)
import ekf_gtsam as _ekf_mod
from ekf_gtsam import (
    run_ekf_rollout,
    OnlineEKF,
    _ekf_step as _ORIG_EKF_STEP,
)
from ekf import (
    _ensure_ctrl_for_step,
    _structured_Q_sigmas,
    _structured_R_sigmas,
    _pose_quat_to_exp,
    _full_quat_state_to_exp_np,
)
from eval import evaluate, evaluate_from_frames

_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATASET_ROOT = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0"
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
            d0["pos"][r * 3 : (r + 1) * 3]
            + d0["quat"][r * 4 : (r + 1) * 4]
            + d0["linvel"][r * 3 : (r + 1) * 3]
            + d0["angvel"][r * 3 : (r + 1) * 3]
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


def _run_raw_gnn(sim, gt_data, extra_data, device):
    start = _make_start_state(gt_data, len(sim.robot.rigid_bodies), device)
    ctrls = torch.tensor(
        [e["controls"] for e in extra_data], dtype=DEFAULT_DTYPE
    ).T.unsqueeze(0).to(device)
    return evaluate(
        sim, gt_data, ctrls,
        extra_data[0]["rest_lengths"],
        extra_data[0]["motor_speeds"],
    )


def _run_ekf(sim, gt_data, extra_data, device, **ekf_kwargs):
    """Return (frames, com_err, rot_err, pen_err) for a GTSAM-MEKF rollout."""
    start = _make_start_state(gt_data, len(sim.robot.rigid_bodies), device)
    frames = run_ekf_rollout(
        sim, gt_data, extra_data,
        start_state=start,
        observe_pose_only=True,
        dataset_idx_val=9,
        **ekf_kwargs,
    )
    n_rods = len(sim.robot.rigid_bodies)
    com_e, rot_e, pen_e = evaluate_from_frames(frames, gt_data, n_rods, device, is_exp=True)
    return frames, com_e, rot_e, pen_e


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def _corrupt_gt(gt_data, sigma_pos, sigma_rot, rng):
    corrupted = []
    for d in gt_data:
        dc = dict(d)
        pos = np.array(d["pos"], dtype=np.float64)
        dc["pos"] = (pos + rng.normal(0.0, sigma_pos, pos.shape)).tolist()
        quats = np.array(d["quat"], dtype=np.float64).reshape(-1, 4)
        quats += rng.normal(0.0, sigma_rot, quats.shape)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        dc["quat"] = quats.reshape(-1).tolist()
        corrupted.append(dc)
    return corrupted


# ---------------------------------------------------------------------------
# Phase 3: Refiner behavior
# ---------------------------------------------------------------------------

def phase3_refiner_behavior(model_path, dataset_root, trajs, device,
                             process_noise=1e-6, measurement_noise=1e-1,
                             ratio_target=0.7):
    print("\n=== Phase 3: GTSAM Refiner behavior ===")
    rows = []
    header = f"{'traj':<10} {'raw_com':>10} {'gtsam_com':>12} {'ratio':>8} {'raw_rot':>10} {'gtsam_rot':>12}"
    print(header)
    print("-" * len(header))

    all_pass = True
    for traj in trajs:
        data_dir = Path(dataset_root) / traj
        if not data_dir.exists():
            print(f"  SKIP {traj}: not found")
            continue
        sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)

        _reset_sim(sim, extra_data, device)
        raw_com, raw_rot, raw_pen = _run_raw_gnn(sim, gt_data, extra_data, device)

        _, gtsam_com, gtsam_rot, gtsam_pen = _run_ekf(
            sim, gt_data, extra_data, device,
            dt=0.01,
            process_noise_scale=process_noise,
            measurement_noise_scale=measurement_noise,
            innovation_gate_sigma=3.0,
            jacobian_update_interval=5,
        )

        ratio = gtsam_com / max(raw_com, 1e-12)
        ok = ratio < ratio_target
        if not ok:
            all_pass = False
        tag = "" if ok else "  <- FAIL (ratio >= {:.2f})".format(ratio_target)
        print(
            f"{traj:<10} {raw_com:>10.4e} {gtsam_com:>12.4e} {ratio:>8.3f}"
            f" {raw_rot:>10.4e} {gtsam_rot:>12.4e}{tag}"
        )
        rows.append(dict(
            traj=traj, raw_com=raw_com, gtsam_com=gtsam_com, ratio=ratio,
            raw_rot=raw_rot, gtsam_rot=gtsam_rot,
        ))

    status = "PASS" if all_pass else "FAIL"
    print(f"\nPhase 3 verdict: {status}  (target ratio < {ratio_target})")
    return rows, all_pass


def phase3_noise_sweep(model_path, dataset_root, traj, device,
                        sigmas_pos=None, process_noise=1e-6, save_plots=False):
    if sigmas_pos is None:
        sigmas_pos = [0.0, 0.001, 0.005, 0.01, 0.05]

    print(f"\n=== Phase 3: GTSAM Noise sweep ({traj}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    rng = np.random.default_rng(42)

    print(f"  {'sigma_pos(m)':>12} {'raw_com':>12} {'gtsam_com':>12} {'meas_com':>12}")
    n_rods = len(sim.robot.rigid_bodies)
    raw_com_ref = None
    ekf_coms = []

    for sigma in sigmas_pos:
        corrupted = _corrupt_gt(gt_data, sigma_pos=sigma, sigma_rot=sigma * 5.0, rng=rng)

        meas_com_errs = []
        for i in range(1, len(corrupted)):
            for r in range(n_rods):
                pred = np.array(corrupted[i]["pos"][r * 3 : r * 3 + 3])
                true = np.array(gt_data[i]["pos"][r * 3 : r * 3 + 3])
                meas_com_errs.append(float(np.mean((pred - true) ** 2)))
        meas_com = float(np.mean(meas_com_errs))

        if raw_com_ref is None:
            _reset_sim(sim, extra_data, device)
            raw_com_ref, _, _ = _run_raw_gnn(sim, gt_data, extra_data, device)

        meas_noise = max(sigma ** 2, 1e-8)
        _, ekf_com, _, _ = _run_ekf(
            sim, corrupted, extra_data, device,
            dt=0.01,
            process_noise_scale=process_noise,
            measurement_noise_scale=meas_noise,
            innovation_gate_sigma=3.0,
            jacobian_update_interval=5,
        )
        ekf_coms.append(ekf_com)

        print(
            f"  {sigma:>12.3f} {raw_com_ref:>12.4e} {ekf_com:>12.4e} {meas_com:>12.4e}"
        )

    if save_plots:
        _plot_noise_sweep(sigmas_pos, raw_com_ref, ekf_coms, traj)

    return sigmas_pos, ekf_coms, raw_com_ref


def phase3_R_sweep(model_path, dataset_root, traj, device,
                    sigma_injected=0.01, R_values=None, save_plots=False):
    if R_values is None:
        R_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    print(f"\n=== Phase 3: GTSAM R-sweep  (sigma={sigma_injected:.3f} m, {traj}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    rng = np.random.default_rng(42)
    corrupted = _corrupt_gt(gt_data, sigma_pos=sigma_injected,
                             sigma_rot=sigma_injected * 5.0, rng=rng)

    true_var = sigma_injected ** 2
    print(f"  True sigma^2= {true_var:.2e}  (expect optimal R near this value)")
    print(f"  {'R':>10} {'gtsam_com':>12} {'note':>20}")

    best_R, best_err = None, np.inf
    for R in R_values:
        _, ekf_com, _, _ = _run_ekf(
            sim, corrupted, extra_data, device,
            dt=0.01, process_noise_scale=1e-6,
            measurement_noise_scale=R,
            innovation_gate_sigma=3.0, jacobian_update_interval=5,
        )
        mark = "  <- best" if ekf_com < best_err else ""
        if ekf_com < best_err:
            best_err, best_R = ekf_com, R
        print(f"  {R:>10.1e} {ekf_com:>12.4e}{mark}")

    print(f"\n  Best R={best_R:.1e}  (true sigma^2={true_var:.1e})")
    return R_values, best_R


# ---------------------------------------------------------------------------
# Phase 4: Per-step diagnostics
# ---------------------------------------------------------------------------

def _make_nees_capturing_wrapper(nees_records):
    """Wrap _ekf_step to compute per-step NEES (innovation chi-square)."""
    _orig = _ORIG_EKF_STEP

    def _nees_step(x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
                   Q_sigmas, R_sigmas, n_rods, have_measurement,
                   innovation_gate_sigma=np.inf, dataset_idx_val=9,
                   diagnostics=None, x_pred_quat_np=None):
        x_post, P_post = _orig(
            x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
            Q_sigmas, R_sigmas, n_rods, have_measurement,
            innovation_gate_sigma=innovation_gate_sigma,
            dataset_idx_val=dataset_idx_val,
            diagnostics=diagnostics,
            x_pred_quat_np=x_pred_quat_np,
        )

        if have_measurement and z_exp is not None:
            try:
                ref = next(simulator.parameters())
                dtype, dev = ref.dtype, ref.device
            except StopIteration:
                dtype, dev = DEFAULT_DTYPE, torch.device("cpu")

            quat_dim = x_quat_np.size
            state_dim = P_exp.shape[0]
            x_t = torch.tensor(x_quat_np, dtype=dtype, device=dev).reshape(1, quat_dim, 1)
            x_pred_exp = quat_state_to_exp_state(x_t)[0, :state_dim, 0].detach().cpu().numpy().astype(np.float64)

            Q_safe = np.maximum(Q_sigmas, 1e-9)
            Q = np.diag(Q_safe ** 2)
            P_pred = 0.5 * (F_exp @ P_exp @ F_exp.T + Q)
            P_pred = 0.5 * (P_pred + P_pred.T)

            R_mat = np.diag(np.maximum(R_sigmas, 1e-9) ** 2)
            S = H_np @ P_pred @ H_np.T + R_mat
            innov = z_exp.reshape(-1) - H_np @ x_pred_exp
            meas_dim = innov.size

            try:
                S_inv = np.linalg.inv(S)
                nees = float(innov @ S_inv @ innov)
                nees_records.append({"nees": nees, "meas_dim": meas_dim,
                                     "innov_norm": float(np.linalg.norm(innov)),
                                     "trace_P": float(np.trace(P_post))})
            except np.linalg.LinAlgError:
                pass

        return x_post, P_post

    return _nees_step


def phase4_diagnostics(model_path, dataset_root, traj, device,
                        process_noise=1e-6, measurement_noise=1e-1,
                        save_plots=False):
    print(f"\n=== Phase 4: GTSAM Per-step diagnostics ({traj}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)

    nees_records = []
    _ekf_mod._ekf_step = _make_nees_capturing_wrapper(nees_records)
    try:
        frames = run_ekf_rollout(
            sim, gt_data, extra_data,
            start_state=start, dt=0.01,
            process_noise_scale=process_noise,
            measurement_noise_scale=measurement_noise,
            observe_pose_only=True,
            dataset_idx_val=9,
            log_diagnostics=True,
            jacobian_update_interval=5,
        )
    finally:
        _ekf_mod._ekf_step = _ORIG_EKF_STEP

    if not nees_records:
        print("  No NEES records collected (no measurement steps).")
        return

    innov_norms = [r["innov_norm"] for r in nees_records]
    nees_vals   = [r["nees"] for r in nees_records]
    traces      = [r["trace_P"] for r in nees_records]
    meas_dim    = nees_records[0]["meas_dim"]

    mean_nees  = float(np.mean(nees_vals))
    nees_ratio = mean_nees / meas_dim
    nsteps     = len(nees_records)
    lo = float(np.percentile(nees_vals, 5))
    hi = float(np.percentile(nees_vals, 95))

    print(f"  Steps with measurements: {nsteps}")
    print(f"  Mean NEES:               {mean_nees:.2f}  (meas_dim={meas_dim})")
    print(f"  NEES / meas_dim:         {nees_ratio:.3f}  (consistent filter ~1.0)")
    print(f"  NEES 5th-95th pct:       [{lo:.1f}, {hi:.1f}]")
    print(f"  Mean innovation norm:    {np.mean(innov_norms):.4f} m")
    print(f"  Max  innovation norm:    {np.max(innov_norms):.4f} m")
    print(f"  Covariance trace trend:  start={traces[0]:.3e}  end={traces[-1]:.3e}")

    consistent = 0.1 < nees_ratio < 10.0
    print(f"\n  NEES consistency: {'PASS' if consistent else 'FAIL'} "
          f"(ratio {nees_ratio:.3f}, expected ~1.0)")

    if save_plots:
        _plot_diagnostics(innov_norms, nees_vals, traces, traj)

    return nees_ratio, consistent


# ---------------------------------------------------------------------------
# Phase 5: Robustness & ablations
# ---------------------------------------------------------------------------

def phase5_gating(model_path, dataset_root, traj, device,
                   outlier_step=10, outlier_magnitude=1.0):
    print(f"\n=== Phase 5: GTSAM Gating test (outlier +{outlier_magnitude}m at step {outlier_step}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)

    corrupted = deepcopy(gt_data)
    for r in range(n_rods):
        corrupted[outlier_step]["pos"][r * 3] += outlier_magnitude

    step_diags = []
    _orig = _ORIG_EKF_STEP

    def _capture_diag(x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
                      Q_sigmas, R_sigmas, n_rods_inner, have_measurement,
                      innovation_gate_sigma=np.inf, dataset_idx_val=9,
                      diagnostics=None, x_pred_quat_np=None):
        diag = {} if diagnostics is None else diagnostics
        x_post, P_post = _orig(
            x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
            Q_sigmas, R_sigmas, n_rods_inner, have_measurement,
            innovation_gate_sigma=innovation_gate_sigma,
            dataset_idx_val=dataset_idx_val,
            diagnostics=diag,
            x_pred_quat_np=x_pred_quat_np,
        )
        step_diags.append(dict(diag))
        return x_post, P_post

    _ekf_mod._ekf_step = _capture_diag
    try:
        frames_corrupt = run_ekf_rollout(
            sim, corrupted, extra_data,
            start_state=start, dt=0.01,
            process_noise_scale=1e-6,
            measurement_noise_scale=1e-1,
            observe_pose_only=True,
            innovation_gate_sigma=3.0,
            dataset_idx_val=9,
            log_diagnostics=True,
        )
    finally:
        _ekf_mod._ekf_step = _ORIG_EKF_STEP

    target_idx = outlier_step - 1
    if target_idx < len(step_diags):
        diag = step_diags[target_idx]
        gated = diag.get("gated", False)
        print(f"  Step {outlier_step} diagnostics: gated={gated}, "
              f"innov_norm={diag.get('pos_innovation_norm', 'n/a'):.3f}")
        status = "PASS" if gated else "FAIL"
        print(f"  {status} — outlier {'gated (state unaffected)' if gated else 'NOT gated (state corrupted)'}")
        return gated
    else:
        print(f"  SKIP — step {outlier_step} out of range (only {len(step_diags)} steps)")
        return False


def phase5_missing_measurements(model_path, dataset_root, traj, device,
                                  dropout=0.2, seed=42):
    print(f"\n=== Phase 5: GTSAM Missing measurements (dropout={dropout:.0%}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    rng = np.random.default_rng(seed)

    gt_sparse = deepcopy(gt_data)
    dropped = 0
    for i in range(1, len(gt_sparse)):
        if rng.random() < dropout:
            gt_sparse[i] = None
            dropped += 1

    print(f"  Dropped {dropped}/{len(gt_data)-1} measurement steps")

    sim2, gt_data2, extra_data2 = _load_assets(
        model_path, Path(dataset_root) / traj, device
    )
    online = OnlineEKF(
        simulator=sim2,
        dt=0.01,
        n_rods=len(sim2.robot.rigid_bodies),
        process_noise_scale=1e-6,
        measurement_noise_scale=1e-1,
        observe_pose_only=True,
        dataset_idx_val=9,
    )
    online.initialize(start.clone())

    frames_online = [{"state": quat_state_to_exp_state(start)}]
    for k, extra in enumerate(extra_data2):
        have_meas = (k + 1 < len(gt_sparse)) and (gt_sparse[k + 1] is not None)
        if have_meas:
            gt = gt_sparse[k + 1]
            pos = np.array(gt["pos"], dtype=np.float64)
            quat = np.array(gt["quat"], dtype=np.float64)
            z_t = np.hstack([
                pos.reshape(n_rods, 3), quat.reshape(n_rods, 4)
            ]).reshape(-1)
        else:
            z_t = None

        state_out = online.step(z_t=z_t, u_t=extra["controls"], have_measurement=have_meas)
        frames_online.append({"state": state_out.detach()})

    ekf_com, ekf_rot, _ = evaluate_from_frames(frames_online, gt_data2, n_rods, device, is_exp=True)

    _reset_sim(sim2, extra_data2, device)
    raw_com, raw_rot, _ = _run_raw_gnn(sim2, gt_data2, extra_data2, device)

    ratio = ekf_com / max(raw_com, 1e-12)
    status = "PASS" if ratio < 1.0 else "FAIL"
    print(f"  raw_com={raw_com:.4e}  gtsam_com(sparse)={ekf_com:.4e}  ratio={ratio:.3f}")
    print(f"  {status} — GTSAM-EKF {'beats' if ratio < 1.0 else 'does NOT beat'} raw GNN with {dropout:.0%} dropout")
    return ratio < 1.0


def phase5_jac_sweep(model_path, dataset_root, traj, device,
                      jac_periods=None):
    if jac_periods is None:
        jac_periods = [1, 5, 10, 20]

    print(f"\n=== Phase 5: GTSAM Jacobian update period sweep ({traj}) ===")
    data_dir = Path(dataset_root) / traj
    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)

    print(f"  {'jac_period':>12} {'gtsam_com':>12} {'ratio_vs_J1':>14}")
    base_err = None
    prev_err = None
    all_graceful = True

    for J in jac_periods:
        _, ekf_com, _, _ = _run_ekf(
            sim, gt_data, extra_data, device,
            dt=0.01,
            process_noise_scale=1e-6,
            measurement_noise_scale=1e-1,
            innovation_gate_sigma=3.0,
            jacobian_update_interval=J,
        )
        if base_err is None:
            base_err = ekf_com
        ratio = ekf_com / max(base_err, 1e-12)

        if prev_err is not None and ekf_com > prev_err * 20.0:
            all_graceful = False
        prev_err = ekf_com

        print(f"  {J:>12} {ekf_com:>12.4e} {ratio:>14.3f}")

    status = "PASS" if all_graceful else "FAIL"
    print(f"\n  {status} — degradation {'graceful' if all_graceful else 'NOT graceful (catastrophic jump)'}")
    return all_graceful


def phase5_streaming_parity(model_path, dataset_root, traj, device,
                              n_steps=40, tol=1e-5):
    print(f"\n=== Phase 5: GTSAM Streaming vs batch parity ({traj}) ===")
    data_dir = Path(dataset_root) / traj
    sim_batch, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    sim_online, _, _ = _load_assets(model_path, data_dir, device)
    n_rods = len(sim_batch.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    n = min(n_steps, len(extra_data))

    frames = run_ekf_rollout(
        sim_batch, gt_data[: n + 1], extra_data[:n],
        start_state=start.clone(),
        dt=0.01,
        process_noise_scale=1e-6,
        measurement_noise_scale=1e-1,
        observe_pose_only=True,
        dataset_idx_val=9,
        jacobian_update_interval=5,
    )

    online = OnlineEKF(
        simulator=sim_online,
        dt=0.01,
        n_rods=n_rods,
        process_noise_scale=1e-6,
        measurement_noise_scale=1e-1,
        observe_pose_only=True,
        dataset_idx_val=9,
        jacobian_update_interval=5,
        ema_alpha=1.0,
    )
    online.initialize(start.clone())

    max_pos_err = 0.0
    for k in range(n):
        extra = extra_data[k]
        have_meas = k + 1 < len(gt_data)
        if have_meas:
            gt = gt_data[k + 1]
            pos = np.array(gt["pos"], dtype=np.float64)
            quat = np.array(gt["quat"], dtype=np.float64)
            z_t = np.hstack([pos.reshape(n_rods, 3), quat.reshape(n_rods, 4)]).reshape(-1)
        else:
            z_t = None

        online_out = online.step(z_t=z_t, u_t=extra["controls"], have_measurement=have_meas)
        online_qs = exp_state_to_quat_state(online_out).squeeze().cpu().numpy()
        batch_qs  = exp_state_to_quat_state(frames[k + 1]["state"]).squeeze().cpu().numpy()

        for r in range(n_rods):
            err = float(np.linalg.norm(online_qs[13 * r : 13 * r + 3]
                                        - batch_qs[13 * r : 13 * r + 3]))
            max_pos_err = max(max_pos_err, err)

    status = "PASS" if max_pos_err < tol else "FAIL"
    print(f"  {status} — max pos error {max_pos_err:.2e} m  (tol {tol:.0e})")
    return max_pos_err < tol


# ---------------------------------------------------------------------------
# Phase 6: Cross-trajectory summary
# ---------------------------------------------------------------------------

def phase6_cross_traj(rows):
    if not rows:
        return
    ratios = [r["ratio"] for r in rows]
    print("\n=== Phase 6: GTSAM Cross-trajectory generalization ===")
    print(f"  Trajectories evaluated: {len(rows)}")
    print(f"  GTSAM-EKF/raw ratio  mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}")
    consistent = all(r < 1.0 for r in ratios)
    print(f"  All ratios < 1.0: {'YES - GO' if consistent else 'NO - NO-GO'}")
    return consistent


# ---------------------------------------------------------------------------
# Optional plot helpers
# ---------------------------------------------------------------------------

def _plot_noise_sweep(sigmas, ekf_coms, raw_com, traj):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available -- skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigmas, ekf_coms, "o-", label="GTSAM-EKF COM error")
    ax.axhline(raw_com, color="r", linestyle="--", label="Raw GNN COM error")
    ax.set_xlabel("Injected noise sigma_pos (m)")
    ax.set_ylabel("COM MSE (m^2)")
    ax.set_title(f"GTSAM Noise sweep -- {traj}")
    ax.legend()
    fig.tight_layout()
    out = f"gtsam_noise_sweep_{traj}.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def _plot_diagnostics(innov_norms, nees_vals, traces, traj):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available -- skipping plot)")
        return
    steps = np.arange(len(innov_norms))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, innov_norms)
    axes[0].set_ylabel("Innovation norm (m)")
    axes[0].set_title(f"GTSAM-MEKF diagnostics -- {traj}")

    axes[1].plot(steps, nees_vals)
    axes[1].axhline(np.mean(nees_vals), color="r", linestyle="--",
                    label=f"mean={np.mean(nees_vals):.1f}")
    axes[1].set_ylabel("NEES (chi-square)")
    axes[1].legend()

    axes[2].plot(steps, traces)
    axes[2].set_ylabel("Covariance trace")
    axes[2].set_xlabel("Step")

    fig.tight_layout()
    out = f"gtsam_diagnostics_{traj}.png"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GTSAM-MEKF Phase 3-5 evaluation")
    parser.add_argument("--model_path", default=_DEFAULT_MODEL)
    parser.add_argument("--dataset_root", default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--traj", default="traj_6,traj_3",
                        help="Comma-separated trajectory names")
    parser.add_argument("--phases", default="3,4,5",
                        help="Comma-separated phases to run: 3,4,5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--process_noise", type=float, default=1e-6)
    parser.add_argument("--measurement_noise", type=float, default=1e-1)
    parser.add_argument("--ratio_target", type=float, default=0.7)
    parser.add_argument("--save_plots", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    trajs = [t.strip() for t in args.traj.split(",")]
    phases = {int(p.strip()) for p in args.phases.split(",")}
    primary = trajs[0]

    if not Path(args.model_path).exists():
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)
    if not Path(args.dataset_root).exists():
        print(f"ERROR: dataset root not found: {args.dataset_root}")
        sys.exit(1)

    results = {}

    if 3 in phases:
        rows, p3_pass = phase3_refiner_behavior(
            args.model_path, args.dataset_root, trajs, device,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            ratio_target=args.ratio_target,
        )
        phase3_noise_sweep(
            args.model_path, args.dataset_root, primary, device,
            process_noise=args.process_noise, save_plots=args.save_plots,
        )
        phase3_R_sweep(
            args.model_path, args.dataset_root, primary, device,
        )
        results["phase3_refiner"] = p3_pass
        if len(rows) > 1:
            phase6_cross_traj(rows)

    if 4 in phases:
        nees_ratio, p4_pass = phase4_diagnostics(
            args.model_path, args.dataset_root, primary, device,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            save_plots=args.save_plots,
        )
        results["phase4_nees"] = p4_pass

    if 5 in phases:
        results["phase5_gating"] = phase5_gating(
            args.model_path, args.dataset_root, primary, device,
        )
        results["phase5_missing_meas"] = phase5_missing_measurements(
            args.model_path, args.dataset_root, primary, device,
        )
        results["phase5_jac_sweep"] = phase5_jac_sweep(
            args.model_path, args.dataset_root, primary, device,
        )
        results["phase5_streaming"] = phase5_streaming_parity(
            args.model_path, args.dataset_root, primary, device,
        )

    print("\n" + "=" * 60)
    print("FINAL RESULTS  (GTSAM-MEKF)")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            all_pass = False
    print(f"\nOverall: {'GO' if all_pass else 'NO-GO'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
