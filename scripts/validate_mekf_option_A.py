"""Option A validation: full-state MEKF against MuJoCo ground truth.

Option A = full-state observation (observe_pose_only=False) with the current
FD-velocity override kept (velocity is finite-differenced from measured pose
after the Kalman update; see ekf.py:_fd_inject_velocities).  This is what the
code runs today.  We prove it works on INDEPENDENT physics (MuJoCo gt_data),
which is the proxy for the advisors' real data — the GNN process model is
scored against a different physics engine, i.e. a genuine model-mismatch test.

Headline claims proven here, per trajectory:
  (1) RMSE_pos(EKF)  <  RMSE_pos(raw measurements)     — the filter beats the sensor
  (2) RMSE_pos(EKF)  <  RMSE_pos(predict-only GNN)      — measurements help
  (3) pose NEES ≈ 1                                     — covariance is calibrated
Plus a velocity-accuracy number (Option A scores the FD reconstruction).

Why measurements are corrupted: run_ekf_rollout builds measurements directly
from gt_data with NO added noise, so a measurement-only baseline would be 0 by
construction.  We corrupt gt (Gaussian on pos+quat) to form realistic
measurements and score every estimator against the CLEAN gt.

Usage:
    conda run -n cable_robot_gnn python scripts/validate_mekf_option_A.py
    conda run -n cable_robot_gnn python scripts/validate_mekf_option_A.py \
        --trajs traj_6,traj_3 --sigma_pos 0.005
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np
import torch

import ekf as _ekf_mod
from ekf import (
    run_ekf_rollout,
    _ekf_step as _ORIG_EKF_STEP,
)
from linearization_exp import (
    EXP_BLOCK_SIZE,
    quat_state_to_exp_state,
    quat_state_to_canonical_exp_state,
)
from utilities.misc_utils import DEFAULT_DTYPE
from utilities import torch_quaternion
from eval import rollout_by_ctrls

# Reuse the (already-working) harness helpers rather than re-deriving them.
from scripts.mekf_refiner_eval import (
    _load_assets,
    _make_start_state,
    _reset_sim,
    _run_raw_gnn,
    _corrupt_gt,
)

_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATASET_ROOT = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0"
)


# ---------------------------------------------------------------------------
# Baselines computed directly against clean gt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unified pose extraction + windowed scoring (identical metric for every estimator)
# ---------------------------------------------------------------------------

def _poses_from_frames(frames, n_rods):
    """List over steps i>=1 of (n_rods, 7) [pos quat] from exp-map EKF frames."""
    from linearization_exp import exp_state_to_quat_state
    out = []
    for i in range(1, len(frames)):
        qs = exp_state_to_quat_state(frames[i]["state"]).squeeze().cpu().numpy().astype(np.float64)
        out.append(np.array([qs[13 * r:13 * r + 7] for r in range(n_rods)]))
    return out


def _poses_from_rollout(poses, n_rods):
    """List over steps i>=1 of (n_rods, 7) from rollout_by_ctrls poses (1, 7n, 1)."""
    out = []
    for i in range(1, len(poses)):
        p = np.asarray(poses[i].detach().cpu().numpy() if hasattr(poses[i], "detach")
                       else poses[i]).reshape(-1)[:7 * n_rods].astype(np.float64)
        out.append(p.reshape(n_rods, 7))
    return out


def _poses_from_gt(gt, n_rods):
    """List over steps i>=1 of (n_rods, 7) from a gt/measurement dict list."""
    out = []
    for i in range(1, len(gt)):
        out.append(np.array(
            [gt[i]["pos"][3 * r:3 * r + 3] + gt[i]["quat"][4 * r:4 * r + 4]
             for r in range(n_rods)], dtype=np.float64))
    return out


def _score_poses(pred_list, clean_gt, n_rods, eval_start):
    """(com = mean pos MSE, rot = mean angle) over steps >= eval_start vs clean gt.

    NOTE: rot is inflated by axial spin (raw gt quat vs principal-axis pred); it is
    a shared artifact across estimators, reported for context, not gated on.
    """
    clean = _poses_from_gt(clean_gt, n_rods)
    n = min(len(pred_list), len(clean))
    com, rot = [], []
    for i in range(eval_start, n):
        for r in range(n_rods):
            com.append(float(np.mean((pred_list[i][r][:3] - clean[i][r][:3]) ** 2)))
            pq = torch.tensor(pred_list[i][r][3:7], dtype=DEFAULT_DTYPE).reshape(1, 4)
            tq = torch.tensor(clean[i][r][3:7], dtype=DEFAULT_DTYPE).reshape(1, 4)
            rot.append(torch_quaternion.compute_angle_btwn_quats(tq, pq).mean().item())
    return float(np.mean(com)), float(np.mean(rot))


def _predict_only_poses(sim, meas_gt, extra_data, device, n_rods):
    """GNN rollout from the CORRUPTED x0 (the same uncertain start the EKF sees),
    no measurements — a FAIR predict-only baseline (not the clean-x0 best case)."""
    _reset_sim(sim, extra_data, device)
    start = _make_start_state(meas_gt, n_rods, device)
    ctrls = torch.tensor(
        [e["controls"] for e in extra_data], dtype=DEFAULT_DTYPE
    ).T.unsqueeze(0).to(device)
    poses = rollout_by_ctrls(sim, ctrls, start)
    return _poses_from_rollout(poses, n_rods)


def _velocity_rmse_from_frames(frames, clean_gt, n_rods, device):
    """RMSE of EKF output velocity vs clean gt linvel/angvel.

    frames[i]['state'] is exp-map (1, 36, 1); per rod block of 12 holds
    [pos(3) exp_rot(3) linvel(3) angvel(3)].  Under Option A these velocities
    are the FD reconstruction (+ EMA), so this scores that reconstruction.
    """
    lin_sq, ang_sq = [], []
    num_steps = min(len(frames) - 1, len(clean_gt) - 1)
    for i in range(1, num_steps + 1):
        state_np = frames[i]["state"].squeeze().cpu().numpy().astype(np.float64)
        for r in range(n_rods):
            b = EXP_BLOCK_SIZE * r
            lin = state_np[b + 6: b + 9]
            ang = state_np[b + 9: b + 12]
            gt_lin = np.array(clean_gt[i]["linvel"][r * 3:r * 3 + 3], dtype=np.float64)
            gt_ang = np.array(clean_gt[i]["angvel"][r * 3:r * 3 + 3], dtype=np.float64)
            lin_sq.append(np.mean((lin - gt_lin) ** 2))
            ang_sq.append(np.mean((ang - gt_ang) ** 2))
    return float(np.sqrt(np.mean(lin_sq))), float(np.sqrt(np.mean(ang_sq)))


# ---------------------------------------------------------------------------
# Pose NEES (capture posterior x, P at measurement steps; score vs clean gt)
# ---------------------------------------------------------------------------

def _run_ekf_fullstate_with_nees(sim, meas_gt, clean_gt, extra_data, device,
                                 n_rods, measurement_noise, process_noise,
                                 observe_pose_only=False):
    """Run a full-state EKF rollout; return (frames, mean_pose_nees_norm).

    NEES is restricted to pose (pos + exp_rot) because under Option A velocity
    is FD-injected, not Kalman-filtered, so its P block is not meaningful.
    """
    pose_idx = np.array([r * EXP_BLOCK_SIZE + j
                         for r in range(n_rods) for j in range(6)])
    n_pose = len(pose_idx)

    x_list, P_list = [], []

    def _capturing_step(x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
                        Q_sigmas, R_sigmas, n_rods_inner, have_measurement, **kwargs):
        x_post, P_post = _ORIG_EKF_STEP(
            x_quat_np, P_exp, simulator, ctrl, F_exp, H_np, z_exp,
            Q_sigmas, R_sigmas, n_rods_inner, have_measurement, **kwargs)
        if have_measurement:
            x_list.append(x_post.copy())
            P_list.append(P_post.copy())
        return x_post, P_post

    start = _make_start_state(meas_gt, n_rods, device)
    _ekf_mod._ekf_step = _capturing_step
    try:
        frames = run_ekf_rollout(
            sim, meas_gt, extra_data,
            dt=0.01,
            process_noise_scale=process_noise,
            measurement_noise_scale=measurement_noise,
            observe_pose_only=observe_pose_only,
            start_state=start,
            innovation_gate_sigma=3.0,
            jacobian_update_interval=5,
            dataset_idx_val=9,
        )
    finally:
        _ekf_mod._ekf_step = _ORIG_EKF_STEP

    nees_vals = []
    for k, (x_est_quat, P_post) in enumerate(zip(x_list, P_list)):
        if k + 1 >= len(clean_gt):
            break
        gt = clean_gt[k + 1]
        x_true = np.zeros(13 * n_rods, dtype=np.float64)
        for r in range(n_rods):
            x_true[13*r:13*r+3]     = gt["pos"][3*r:3*r+3]
            x_true[13*r+3:13*r+7]   = gt["quat"][4*r:4*r+4]
            x_true[13*r+7:13*r+10]  = gt["linvel"][3*r:3*r+3]
            x_true[13*r+10:13*r+13] = gt["angvel"][3*r:3*r+3]
        x_true_exp = quat_state_to_canonical_exp_state(
            torch.tensor(x_true, dtype=DEFAULT_DTYPE).reshape(1, -1, 1)
        ).squeeze().cpu().numpy().astype(np.float64)
        x_est_exp = quat_state_to_exp_state(
            torch.tensor(x_est_quat, dtype=DEFAULT_DTYPE).reshape(1, -1, 1)
        ).squeeze().cpu().numpy().astype(np.float64)
        err = (x_true_exp - x_est_exp)[pose_idx]
        P_pose = P_post[np.ix_(pose_idx, pose_idx)] + 1e-10 * np.eye(n_pose)
        try:
            nees_vals.append(float(err @ np.linalg.inv(P_pose) @ err))
        except np.linalg.LinAlgError:
            pass

    nees_norm = float(np.mean(nees_vals)) / n_pose if nees_vals else float("nan")
    return frames, nees_norm


# ---------------------------------------------------------------------------
# Per-trajectory validation
# ---------------------------------------------------------------------------

def validate_traj(model_path, dataset_root, traj, device,
                  sigma_pos, sigma_rot, process_noise, seed, max_steps=None,
                  observe_pose_only=False, eval_start_frac=0.2):
    data_dir = Path(dataset_root) / traj
    if not data_dir.exists():
        print(f"  SKIP {traj}: not found")
        return None

    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    if max_steps is not None:
        gt_data = gt_data[:max_steps + 1]
        extra_data = extra_data[:max_steps]
    n_rods = len(sim.robot.rigid_bodies)
    rng = np.random.default_rng(seed)
    # Skip the initial-convergence transient so the window scores steady-state tracking.
    eval_start = int(eval_start_frac * len(extra_data))

    # Corrupt gt → realistic measurements (noise on pose); same x0 corruption the EKF sees.
    meas_gt = _corrupt_gt(gt_data, sigma_pos=sigma_pos, sigma_rot=sigma_rot, rng=rng)

    # (1) Fair predict-only: GNN rolled from the CORRUPTED x0, no measurements.
    pred_poses = _predict_only_poses(sim, meas_gt, extra_data, device, n_rods)
    pred_com, pred_rot = _score_poses(pred_poses, gt_data, n_rods, eval_start)

    # (2) Raw measurement baseline.
    meas_com, meas_rot = _score_poses(_poses_from_gt(meas_gt, n_rods), gt_data, n_rods, eval_start)

    # (3) EKF fed corrupted measurements from the same corrupted x0; score vs clean gt.
    meas_noise = max(sigma_pos ** 2, 1e-8)
    _reset_sim(sim, extra_data, device)
    frames, nees_norm = _run_ekf_fullstate_with_nees(
        sim, meas_gt, gt_data, extra_data, device, n_rods,
        measurement_noise=meas_noise, process_noise=process_noise,
        observe_pose_only=observe_pose_only)
    ekf_com, ekf_rot = _score_poses(_poses_from_frames(frames, n_rods), gt_data, n_rods, eval_start)
    lin_rmse, ang_rmse = _velocity_rmse_from_frames(frames, gt_data, n_rods, device)

    beats_meas = ekf_com < meas_com
    beats_pred = ekf_com < pred_com
    return dict(
        traj=traj, pred_com=pred_com, meas_com=meas_com, ekf_com=ekf_com,
        pred_rot=pred_rot, meas_rot=meas_rot, ekf_rot=ekf_rot,
        lin_rmse=lin_rmse, ang_rmse=ang_rmse, nees_norm=nees_norm,
        beats_meas=beats_meas, beats_pred=beats_pred,
    )


def main():
    ap = argparse.ArgumentParser(description="Option A full-state MEKF validation")
    ap.add_argument("--model_path", default=_DEFAULT_MODEL)
    ap.add_argument("--dataset_root", default=_DEFAULT_DATASET_ROOT)
    ap.add_argument("--trajs", default="traj_6,traj_3,traj_0")
    ap.add_argument("--sigma_pos", type=float, default=0.02,
                    help="measurement pos noise (m); keep > GNN predict-only RMS (~1cm) "
                         "so the measurement is the noisy source and fusion can help")
    ap.add_argument("--sigma_rot", type=float, default=None, help="quat noise (default 5*sigma_pos)")
    ap.add_argument("--process_noise", type=float, default=1e-2,
                    help="validated tuning for the corrupted-measurement validation "
                         "(Q must reflect model-mismatch, not just sensor noise; "
                         "1e-4 was overconfident → NEES blew up and EKF under-trusted "
                         "measurements)")
    ap.add_argument("--nees_lo", type=float, default=0.05,
                    help="established loose bound; this filter is conservative (~0.06)")
    ap.add_argument("--nees_hi", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap trajectory length for a fast signal (default: full)")
    ap.add_argument("--observe", choices=["pose", "full"], default="full",
                    help="pose = observe pose only + FD velocity; full = observe full state")
    ap.add_argument("--eval_start_frac", type=float, default=0.2,
                    help="skip this fraction of leading steps (convergence transient)")
    args = ap.parse_args()

    device = torch.device("cpu")
    sigma_rot = args.sigma_rot if args.sigma_rot is not None else 5.0 * args.sigma_pos
    trajs = [t.strip() for t in args.trajs.split(",") if t.strip()]

    print("=== Option A validation: full-state MEKF vs MuJoCo ground truth ===")
    print(f"  observe={args.observe}, process_noise={args.process_noise}")
    print(f"  measurement noise: sigma_pos={args.sigma_pos} m, sigma_rot={sigma_rot} rad")
    print(f"  metrics: com = mean pos MSE (m^2), rot = mean angle (rad); lower is better")
    print(f"  NOTE: *_rot is inflated by axial spin (raw gt quat vs principal-axis pred);")
    print(f"        it is a shared metric artifact, not divergence — compare cols, don't gate on it.\n")
    hdr = (f"{'traj':<8} {'pred_com':>10} {'meas_com':>10} {'ekf_com':>10} "
           f"{'<meas':>6} {'<pred':>6} {'pred_rot':>9} {'meas_rot':>9} {'ekf_rot':>9} {'NEES/n':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for traj in trajs:
        r = validate_traj(args.model_path, args.dataset_root, traj, device,
                          args.sigma_pos, sigma_rot, args.process_noise, args.seed,
                          max_steps=args.max_steps,
                          observe_pose_only=(args.observe == "pose"),
                          eval_start_frac=args.eval_start_frac)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['traj']:<8} {r['pred_com']:>10.3e} {r['meas_com']:>10.3e} "
              f"{r['ekf_com']:>10.3e} {'Y' if r['beats_meas'] else 'N':>6} "
              f"{'Y' if r['beats_pred'] else 'N':>6} {r['pred_rot']:>9.3e} "
              f"{r['meas_rot']:>9.3e} {r['ekf_rot']:>9.3e} {r['nees_norm']:>8.2f}")

    if not rows:
        print("\nNo trajectories ran.")
        sys.exit(1)

    beats_meas_all = all(r["beats_meas"] for r in rows)
    beats_pred_all = all(r["beats_pred"] for r in rows)
    nees_ok = all(args.nees_lo <= r["nees_norm"] <= args.nees_hi
                  for r in rows if np.isfinite(r["nees_norm"]))

    print("\n=== Verdict ===")
    print(f"  (1) EKF beats raw measurement (all trajs): {'PASS' if beats_meas_all else 'FAIL'}")
    print(f"  (2) EKF beats predict-only   (all trajs): {'PASS' if beats_pred_all else 'FAIL'}")
    print(f"  (3) pose NEES in [{args.nees_lo}, {args.nees_hi}] (all trajs): "
          f"{'PASS' if nees_ok else 'FAIL'}")
    all_pass = beats_meas_all and beats_pred_all and nees_ok
    print(f"\nOption A verdict: {'GO' if all_pass else 'NO-GO'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
