"""Grid sweep over MEKF noise parameters: how much to trust the GNN vs the sensor.

The Kalman trust balance is governed by two beliefs the filter holds:
  Q (process_noise_scale)      -- how much we DISTRUST the GNN process model
  R (measurement_noise_scale)  -- how much we DISTRUST the measurement

The trivial degenerate result is "trust measurements only": large Q + small R
drives the Kalman gain -> 1, so the EKF just copies the sensor (ekf_com ~ meas_com)
and the GNN adds nothing.  The opposite corner ("trust model only": small Q +
large R) ignores the sensor and drifts to predict-only.  The useful regime is in
between -- where fusion beats BOTH baselines.

Crucial difference from validate_mekf_option_A.py: there the filter's assumed R
is hard-wired to the TRUE injected sensor variance (sigma_pos**2), so R is
correctly specified and only Q is free.  Here we DECOUPLE them: the true sensor
noise (sigma_pos) is held fixed -- it is a property of the sensor, not a tuning
knob -- and we sweep the filter's *beliefs* Q and R independently.  That is what
"optimal noise parameters" actually means.

Per (Q, R) cell we report, scored against CLEAN gt over a steady-state window:
  ekf_com       -- EKF position MSE (m^2)
  copy          -- ekf_com / meas_com.  ~1.0  => trivially copying the sensor
  gain          -- min(meas_com, pred_com) / ekf_com.  >1 => fusion beats the
                   better of the two baselines (the non-trivial win we want)
  NEES/n        -- pose NEES normalized by pose dim; ~1 is calibrated, this
                   filter historically runs conservative (~0.06-12)

Headline: among NON-TRIVIAL cells (ekf_com strictly below BOTH meas_com and
pred_com by a margin), report the one with the lowest ekf_com and its Q, R, Q/R.

Usage:
    conda run -n cable_robot_gnn python scripts/mekf_noise_sweep.py
    conda run -n cable_robot_gnn python scripts/mekf_noise_sweep.py \
        --traj traj_6 --q_grid 1e-4,1e-3,1e-2,1e-1 --r_grid 1e-5,1e-4,1e-3,1e-2 \
        --sigma_pos 0.02 --max_steps 200
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np
import torch

from utilities.misc_utils import DEFAULT_DTYPE

from scripts.mekf_refiner_eval import (
    _load_assets,
    _make_start_state,
    _reset_sim,
    _corrupt_gt,
)
from scripts.validate_mekf_option_A import (
    _poses_from_gt,
    _score_poses,
    _predict_only_poses,
    _poses_from_frames,
    _run_ekf_fullstate_with_nees,
)

_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATASET_ROOT = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0"
)


def _parse_grid(s):
    return [float(x) for x in s.split(",") if x.strip()]


def sweep_traj(model_path, dataset_root, traj, device, sigma_pos, sigma_rot,
               q_grid, r_grid, seed, max_steps, observe_pose_only,
               eval_start_frac, nees_lo, nees_hi, margin):
    data_dir = Path(dataset_root) / traj
    if not data_dir.exists():
        print(f"SKIP {traj}: not found")
        return None

    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    if max_steps is not None:
        gt_data = gt_data[:max_steps + 1]
        extra_data = extra_data[:max_steps]
    n_rods = len(sim.robot.rigid_bodies)
    rng = np.random.default_rng(seed)
    eval_start = int(eval_start_frac * len(extra_data))

    # Fixed true sensor noise -> measurements. Same corruption seen at x0.
    meas_gt = _corrupt_gt(gt_data, sigma_pos=sigma_pos, sigma_rot=sigma_rot, rng=rng)

    # Baselines do NOT depend on (Q, R) -- compute once.
    pred_poses = _predict_only_poses(sim, meas_gt, extra_data, device, n_rods)
    pred_com, _ = _score_poses(pred_poses, gt_data, n_rods, eval_start)
    meas_com, _ = _score_poses(_poses_from_gt(meas_gt, n_rods), gt_data, n_rods, eval_start)
    best_baseline = min(meas_com, pred_com)

    print(f"\n=== {traj}  (sensor sigma_pos={sigma_pos} m, fixed) ===")
    print(f"  baselines:  measurement-only com = {meas_com:.3e}   "
          f"predict-only com = {pred_com:.3e}   best = {best_baseline:.3e}")
    print(f"  window: steps >= {eval_start} ({len(extra_data)} total), "
          f"observe={'pose' if observe_pose_only else 'full'}\n")

    hdr = (f"{'Q':>9} {'R':>9} {'Q/R':>9} {'ekf_com':>11} {'copy':>7} "
           f"{'gain':>7} {'NEES/n':>9} {'verdict':>12}")
    print(hdr)
    print("-" * len(hdr))

    cells = []
    for q in q_grid:
        for r in r_grid:
            _reset_sim(sim, extra_data, device)
            frames, nees_norm = _run_ekf_fullstate_with_nees(
                sim, meas_gt, gt_data, extra_data, device, n_rods,
                measurement_noise=r, process_noise=q,
                observe_pose_only=observe_pose_only)
            ekf_com, _ = _score_poses(
                _poses_from_frames(frames, n_rods), gt_data, n_rods, eval_start)

            copy = ekf_com / meas_com if meas_com > 0 else float("inf")
            gain = best_baseline / ekf_com if ekf_com > 0 else float("inf")
            nees_ok = (np.isfinite(nees_norm) and nees_lo <= nees_norm <= nees_hi)

            # Non-trivial fusion: beats BOTH baselines by `margin`, NOT just
            # copying the sensor, and covariance in a sane range.
            beats_both = (ekf_com < (1 - margin) * meas_com and
                          ekf_com < (1 - margin) * pred_com)
            # copy ~ 1 (within margin band) => statistically the same as the
            # sensor i.e. the trivial trust-measurements-only result.
            trivial_copy = abs(copy - 1.0) <= margin
            if beats_both and nees_ok:
                verdict = "FUSION"
            elif beats_both and not nees_ok:
                verdict = "miscalib"          # tracks well but covariance wrong
            elif trivial_copy:
                verdict = "copy-meas"         # indistinguishable from the sensor
            elif ekf_com >= pred_com:
                verdict = "drift"             # leans on the model, lost the sensor
            elif ekf_com >= meas_com:
                verdict = "no-help"           # worse than sensor, not drifting
            else:
                verdict = "weak"              # helps, but under the margin

            cells.append(dict(q=q, r=r, ekf_com=ekf_com, copy=copy, gain=gain,
                              nees_norm=nees_norm, verdict=verdict,
                              fusion=(verdict == "FUSION")))
            print(f"{q:>9.1e} {r:>9.1e} {q / r:>9.1e} {ekf_com:>11.3e} "
                  f"{copy:>7.3f} {gain:>7.2f} {nees_norm:>9.2f} {verdict:>12}")

    fusion_cells = [c for c in cells if c["fusion"]]
    print()
    if fusion_cells:
        best = min(fusion_cells, key=lambda c: c["ekf_com"])
        print(f"  >> NON-TRIVIAL OPTIMUM: Q={best['q']:.1e}  R={best['r']:.1e}  "
              f"Q/R={best['q'] / best['r']:.1e}")
        print(f"     ekf_com={best['ekf_com']:.3e} (gain {best['gain']:.2f}x over best "
              f"baseline, copy {best['copy']:.3f}, NEES/n {best['nees_norm']:.2f})")
    else:
        print("  >> NO non-trivial fusion cell found in this grid (all cells either "
              "copy the sensor, drift, or are miscalibrated).")
    return dict(traj=traj, meas_com=meas_com, pred_com=pred_com, cells=cells,
                fusion_cells=fusion_cells)


def main():
    ap = argparse.ArgumentParser(description="MEKF Q/R noise grid sweep")
    ap.add_argument("--model_path", default=_DEFAULT_MODEL)
    ap.add_argument("--dataset_root", default=_DEFAULT_DATASET_ROOT)
    ap.add_argument("--traj", default="traj_6", help="single trajectory to sweep")
    ap.add_argument("--q_grid", default="1e-4,1e-3,1e-2,1e-1",
                    help="process_noise (GNN distrust) values, comma-separated")
    ap.add_argument("--r_grid", default="1e-5,1e-4,1e-3,1e-2",
                    help="measurement_noise (sensor distrust) values, comma-separated")
    ap.add_argument("--sigma_pos", type=float, default=0.02,
                    help="TRUE injected sensor pos noise (m), held fixed across the grid")
    ap.add_argument("--sigma_rot", type=float, default=None,
                    help="true quat noise (default 5*sigma_pos)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap horizon for speed; default full (needed for predict-only to drift)")
    ap.add_argument("--observe", choices=["pose", "full"], default="pose",
                    help="pose = observe pose only + FD velocity (validated Option A)")
    ap.add_argument("--eval_start_frac", type=float, default=0.2)
    ap.add_argument("--nees_lo", type=float, default=0.05)
    ap.add_argument("--nees_hi", type=float, default=20.0)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="fractional margin: FUSION requires ekf_com < (1-margin)*baseline")
    args = ap.parse_args()

    device = torch.device("cpu")
    sigma_rot = args.sigma_rot if args.sigma_rot is not None else 5.0 * args.sigma_pos
    q_grid = _parse_grid(args.q_grid)
    r_grid = _parse_grid(args.r_grid)

    print("=== MEKF noise-trust grid sweep ===")
    print(f"  Q (GNN distrust)    grid: {q_grid}")
    print(f"  R (sensor distrust) grid: {r_grid}")
    print(f"  true sensor noise (FIXED): sigma_pos={args.sigma_pos} m, sigma_rot={sigma_rot} rad")
    print(f"  goal: find (Q,R) where the EKF beats BOTH the sensor and predict-only")
    print(f"        (the 'FUSION' verdict) -- excluding the trivial trust-measurements-only corner.")

    sweep_traj(args.model_path, args.dataset_root, args.traj, device,
               args.sigma_pos, sigma_rot, q_grid, r_grid, args.seed,
               args.max_steps, observe_pose_only=(args.observe == "pose"),
               eval_start_frac=args.eval_start_frac,
               nees_lo=args.nees_lo, nees_hi=args.nees_hi, margin=args.margin)


if __name__ == "__main__":
    main()
