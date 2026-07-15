"""Phase 4: GTSAM-MEKF validation under observation noise (swing-only metric).

Protocol (per trajectory, per sigma):
  1. Corrupt gt measurements: pos ~ N(0, sigma_pos), quat ~ N(0, 5*sigma_pos)
     re-normalized.  Score everything against the CLEAN gt.
  2. Measurement-only baseline: corrupted poses vs clean gt.
  3. Predict-only baseline: raw GNN rolled open-loop from the SAME corrupted
     x0 (fair comparison — see validate_mekf_option_A lessons).
  4. GTSAM-MEKF: pose-only observation of the corrupted measurements,
     R = sigma^2, Q = process_noise (1e-2 validated for model-mismatch),
     started from the same corrupted x0.  Per-step NEES captured.
  All scored over a convergence window that skips the initial 20% of steps.

Pass criteria (per run):
  ekf_com < meas_com  AND  ekf_com < pred_com  AND  NEES/dim in [0.05, 20].

Run:
  conda run --no-capture-output -n cable_robot_gnn python -u \
      scripts/phase4_ekf_noise_validation.py --device cpu
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulators.tensegrity_gnn_simulator import load_simulator
from utilities.misc_utils import DEFAULT_DTYPE
from linearization_exp import exp_state_to_quat_state
import ekf_gtsam as _ekf_mod
from ekf_gtsam import run_ekf_rollout
from eval import rollout_by_ctrls
from scripts.gtsam_refiner_eval import (
    _corrupt_gt, _reset_sim, _make_nees_capturing_wrapper, _ORIG_EKF_STEP,
)
from scripts.rotation_swing_twist_diag import (
    DATA_ROOT, MODEL_PATH, gt_poses_and_endpts, quat_to_rot_z, unit,
)


def swing_np(qa, qb):
    """Principal-axis angle between two (..., 4) wxyz quat arrays."""
    za, zb = quat_to_rot_z(unit(qa)), quat_to_rot_z(unit(qb))
    return np.arccos(np.clip(np.sum(za * zb, axis=-1), -1.0, 1.0))


def score(pred_poses, clean_poses, skip_frac=0.2):
    """Windowed (com_mse, swing) of (T, n_rods, 7) pose arrays vs clean gt."""
    T = min(len(pred_poses), len(clean_poses))
    lo = max(1, int(T * skip_frac))
    p, g = pred_poses[lo:T], clean_poses[lo:T]
    com = ((p[:, :, :3] - g[:, :, :3]) ** 2).mean()
    sw = swing_np(p[:, :, 3:], g[:, :, 3:]).mean()
    return float(com), float(sw)


def corrupted_start_state(corrupted, clean, n_rods, device):
    """13-stride start state: corrupted pos/quat, clean velocities."""
    d0c, d0 = corrupted[0], clean[0]
    vals = []
    for r in range(n_rods):
        vals.extend(
            d0c['pos'][r * 3:(r + 1) * 3] + d0c['quat'][r * 4:(r + 1) * 4]
            + d0['linvel'][r * 3:(r + 1) * 3] + d0['angvel'][r * 3:(r + 1) * 3]
        )
    return torch.tensor(vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)


def frames_to_poses(frames, n_rods):
    poses = []
    for fr in frames:
        s = exp_state_to_quat_state(fr['state']).reshape(n_rods, 13)
        poses.append(s[:, :7].detach().cpu().numpy())
    return np.stack(poses)


def rollout_poses_from(sim, extra_data, start_state, device, n_rods):
    ctrls = torch.tensor(
        [e['controls'] for e in extra_data], dtype=DEFAULT_DTYPE
    ).T.unsqueeze(0).to(device)
    _reset_sim(sim, extra_data, device)
    with torch.no_grad():
        rp = rollout_by_ctrls(sim, ctrls, start_state, dataset_idx=0)
    return np.stack([p.reshape(n_rods, 7).cpu().numpy() for p in rp])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trajs', type=str, default='traj_6,traj_3')
    parser.add_argument('--sigmas', type=str, default='0.005,0.02,0.05')
    parser.add_argument('--process_noise', type=float, default=1e-2,
                        help='Validated for model-mismatch (NEES-calibrated)')
    parser.add_argument('--q_sweep', type=str, default='1e-4,1e-1',
                        help='Extra Q values swept on the first traj at the '
                             'middle sigma (contrast to --process_noise)')
    parser.add_argument('--jac_interval', type=int, default=20)
    parser.add_argument('--skip_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    trajs = args.trajs.split(',')
    sigmas = [float(s) for s in args.sigmas.split(',')]
    q_extra = [float(q) for q in args.q_sweep.split(',')] if args.q_sweep else []

    sim = load_simulator(
        MODEL_PATH, map_location=torch.device('cpu'), cache_batch_sizes=[1]
    )
    sim = sim.to(device)
    sim.eval()
    n_rods = len(sim.robot.rigid_bodies)

    # Primary cells decide the verdict; Q-contrast cells only illustrate
    # calibration sensitivity (Q=1e-4 is EXPECTED to fail: overconfident
    # model -> NEES blowup, the known model-mismatch lesson).
    runs = [(t, s, args.process_noise, True) for t in trajs for s in sigmas]
    runs += [(trajs[0], sigmas[len(sigmas) // 2], q, False) for q in q_extra]

    print(f'{"traj":<8}{"sigma":>8}{"Q":>9}{"meas_com":>12}{"pred_com":>12}'
          f'{"ekf_com":>12}{"ekf_swing":>11}{"NEES/d":>9}{"verdict":>12}')
    all_pass = True
    for traj, sigma, q, primary in runs:
        tdir = DATA_ROOT / traj
        with open(tdir / 'processed_data.json') as f:
            gt_data = json.load(f)
        with open(tdir / 'extra_state_data.json') as f:
            extra_data = json.load(f)
        clean_poses, _ = gt_poses_and_endpts(gt_data, n_rods)

        rng = np.random.default_rng(args.seed)
        corrupted = _corrupt_gt(gt_data, sigma_pos=sigma,
                                sigma_rot=sigma * 5.0, rng=rng)
        corr_poses, _ = gt_poses_and_endpts(corrupted, n_rods)
        x0 = corrupted_start_state(corrupted, gt_data, n_rods, device)

        meas_com, meas_sw = score(corr_poses, clean_poses, args.skip_frac)

        pred_poses = rollout_poses_from(sim, extra_data, x0, device, n_rods)
        pred_com, pred_sw = score(pred_poses, clean_poses, args.skip_frac)

        nees_records = []
        _ekf_mod._ekf_step = _make_nees_capturing_wrapper(nees_records)
        try:
            _reset_sim(sim, extra_data, device)
            frames = run_ekf_rollout(
                sim, corrupted, extra_data,
                dt=0.01,
                process_noise_scale=q,
                measurement_noise_scale=max(sigma ** 2, 1e-8),
                observe_pose_only=True,
                start_state=x0,
                use_finite_diff=False,
                innovation_gate_sigma=3.0,
                dataset_idx_val=0,
                max_spectral_radius=1.0,
                jacobian_update_interval=args.jac_interval,
            )
        finally:
            _ekf_mod._ekf_step = _ORIG_EKF_STEP

        ekf_poses = frames_to_poses(frames, n_rods)
        ekf_com, ekf_sw = score(ekf_poses, clean_poses, args.skip_frac)

        if nees_records:
            nees_d = float(np.mean([r['nees'] for r in nees_records])
                           ) / nees_records[0]['meas_dim']
        else:
            nees_d = float('nan')

        ok = (ekf_com < meas_com and ekf_com < pred_com
              and 0.05 <= nees_d <= 20)
        if primary:
            all_pass = all_pass and ok
            tag = 'PASS' if ok else 'FAIL'
        else:
            tag = ('pass' if ok else 'fail') + '(ctr)'
        print(f'{traj:<8}{sigma:>8.3f}{q:>9.0e}{meas_com:>12.4e}'
              f'{pred_com:>12.4e}{ekf_com:>12.4e}{ekf_sw:>11.4f}'
              f'{nees_d:>9.2f}{tag:>12}', flush=True)

    print(f'\nPhase 4 verdict: {"PASS" if all_pass else "FAIL"} over primary '
          f'cells (EKF beats measurement-only AND predict-only, NEES/d in '
          f'[0.05,20], swing-only metric, window skips first '
          f'{args.skip_frac:.0%}); (ctr) rows are Q-calibration contrasts.')


if __name__ == '__main__':
    main()
