"""Validate EKF using GNN as ground truth.

This script runs the EKF on rollout data where:
  - GNN predictions are treated as ground truth
  - Synthetic sensor noise is added to create measurements
  - EKF estimates are compared against GNN predictions

Checks:
  1. Residuals: innovations should be white noise (zero mean, uncorrelated)
  2. Covariance health: P stays positive definite, reasonable magnitude
  3. Sanity: filtered state makes physical sense
  4. Performance: filtered state closer to GNN than raw noisy measurements
"""

import argparse
import json
import numpy as np
import torch
import tqdm
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

from simulators.tensegrity_gnn_simulator import load_simulator
from ekf import OnlineEKF, run_ekf_rollout, _structured_Q_sigmas, _structured_R_sigmas
from utilities.misc_utils import DEFAULT_DTYPE


def plot_residuals(innovations, output_dir="ekf_validation"):
    """Plot innovation sequence and check for white noise properties."""
    Path(output_dir).mkdir(exist_ok=True)

    innovations = np.array(innovations)
    n_steps, n_meas = innovations.shape

    # Plot innovations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series
    axes[0, 0].plot(innovations[:, :min(5, n_meas)])
    axes[0, 0].set_title('Innovation Time Series (first 5 channels)')
    axes[0, 0].set_ylabel('Innovation')
    axes[0, 0].set_xlabel('Timestep')

    # Histogram of pooled innovations
    axes[0, 1].hist(innovations.flatten(), bins=30, density=True)
    axes[0, 1].set_title('Histogram of All Innovations')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].axvline(0, color='r', linestyle='--', label='Zero mean')
    axes[0, 1].legend()

    # Autocorrelation (mean across channels)
    mean_innovation = innovations.mean(axis=1)
    acf = np.correlate(mean_innovation - mean_innovation.mean(),
                       mean_innovation - mean_innovation.mean(),
                       mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    axes[1, 0].plot(acf[:min(50, len(acf))])
    axes[1, 0].set_title('Autocorrelation of Mean Innovation')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].axhline(0, color='r', linestyle='--', alpha=0.3)

    # Q-Q plot
    stats.probplot(innovations.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals.png', dpi=100)
    print(f"Saved residual plots to {output_dir}/residuals.png")

    return innovations


def check_residual_properties(innovations):
    """Check if innovations are white noise: zero mean, uncorrelated, normal."""
    innovations = np.array(innovations)
    n_steps, n_meas = innovations.shape

    results = {
        'n_timesteps': n_steps,
        'n_measurements': n_meas,
    }

    # Zero mean test
    mean = innovations.mean(axis=0)
    std = innovations.std(axis=0)
    results['innovation_mean'] = float(np.mean(np.abs(mean)))
    results['innovation_std'] = float(np.mean(std))

    # Normality test (Shapiro-Wilk on pooled)
    pooled = innovations.flatten()
    if len(pooled) > 5000:
        pooled_sample = np.random.choice(pooled, 5000, replace=False)
    else:
        pooled_sample = pooled
    stat, p_value = stats.shapiro(pooled_sample)
    results['normality_test_p'] = float(p_value)
    results['normality_pass'] = p_value > 0.05

    # Autocorrelation test: mean innovation
    mean_innovation = innovations.mean(axis=1)
    mean_innovation_centered = mean_innovation - mean_innovation.mean()
    acf = np.correlate(mean_innovation_centered, mean_innovation_centered, mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    results['autocorr_lag1'] = float(acf[1])  # Should be ~0
    results['autocorr_pass'] = np.abs(acf[1]) < 0.3

    return results


def check_covariance_health(covs):
    """Check covariance matrices for positive definiteness and reasonable scale."""
    covs = np.array(covs)  # (n_steps, state_dim, state_dim)
    n_steps, state_dim, _ = covs.shape

    results = {
        'n_timesteps': n_steps,
        'state_dim': state_dim,
        'all_positive_definite': True,
        'eigenvalue_issues': [],
        'log_det_mean': None,
        'log_det_std': None,
    }

    log_dets = []
    for i, P in enumerate(covs):
        try:
            eigvals = np.linalg.eigvals(P)
            min_eigval = np.min(eigvals)
            if min_eigval < 1e-10:
                results['all_positive_definite'] = False
                results['eigenvalue_issues'].append({
                    'step': i,
                    'min_eigenvalue': float(min_eigval)
                })
            log_dets.append(np.log(np.linalg.det(P)))
        except np.linalg.LinAlgError:
            results['all_positive_definite'] = False
            results['eigenvalue_issues'].append({
                'step': i,
                'error': 'singular matrix'
            })

    if log_dets:
        results['log_det_mean'] = float(np.mean(log_dets))
        results['log_det_std'] = float(np.std(log_dets))

    return results


def compare_gnn_vs_filtered(gnn_states, filtered_states, gt_states, n_rods):
    """Compare GNN, EKF-filtered, and noisy measurements against ground truth.

    Returns:
        gnn_error: MSE between GNN and GT
        filtered_error: MSE between EKF and GT
        noisy_error: MSE between noisy and GT
    """
    gnn_states = np.array([s.detach().cpu().numpy().reshape(-1) for s in gnn_states])
    filtered_states = np.array([s.detach().cpu().numpy().reshape(-1) for s in filtered_states])
    gt_states = np.array([s.reshape(-1) for s in gt_states])

    # Compare full state
    gnn_error = np.mean((gnn_states - gt_states)**2)
    filtered_error = np.mean((filtered_states - gt_states)**2)

    # Compare position only (first 3*n_rods)
    pos_dim = 3 * n_rods
    gnn_pos_error = np.mean((gnn_states[:, :pos_dim] - gt_states[:, :pos_dim])**2)
    filtered_pos_error = np.mean((filtered_states[:, :pos_dim] - gt_states[:, :pos_dim])**2)

    # Compare rotation only (next 4*n_rods)
    rot_dim = 4 * n_rods
    gnn_rot_error = np.mean((gnn_states[:, pos_dim:pos_dim+rot_dim] -
                             gt_states[:, pos_dim:pos_dim+rot_dim])**2)
    filtered_rot_error = np.mean((filtered_states[:, pos_dim:pos_dim+rot_dim] -
                                 gt_states[:, pos_dim:pos_dim+rot_dim])**2)

    return {
        'gnn_full_state_mse': float(gnn_error),
        'filtered_full_state_mse': float(filtered_error),
        'improvement_full': float((gnn_error - filtered_error) / (gnn_error + 1e-10)),
        'gnn_position_mse': float(gnn_pos_error),
        'filtered_position_mse': float(filtered_pos_error),
        'improvement_position': float((gnn_pos_error - filtered_pos_error) / (gnn_pos_error + 1e-10)),
        'gnn_rotation_mse': float(gnn_rot_error),
        'filtered_rotation_mse': float(filtered_rot_error),
        'improvement_rotation': float((gnn_rot_error - filtered_rot_error) / (gnn_rot_error + 1e-10)),
    }


def validate_ekf(model_path, data_path, extra_data_path,
                 process_noise=1e-4, measurement_noise=1e-3,
                 sensor_noise_scale=1e-3, max_steps=None,
                 output_dir='ekf_validation'):
    """Run EKF validation on rollout data."""
    Path(output_dir).mkdir(exist_ok=True)

    # Load model and simulator
    print("Loading simulator and model...")
    simulator = load_simulator(model_path)
    n_rods = len(simulator.robot.rigid_bodies)
    state_dim = 13 * n_rods
    dt = simulator.dt

    # Load data
    print("Loading data...")
    with open(data_path) as f:
        gt_data = json.load(f)
    with open(extra_data_path) as f:
        extra_data = json.load(f)

    if max_steps:
        gt_data = gt_data[:max_steps]
        extra_data = extra_data[:max_steps-1]

    # Initialize simulator state
    device = simulator.device
    dtype = simulator.dtype
    cables = list(simulator.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            extra_data[0]['rest_lengths'][i], dtype=dtype
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            extra_data[0]['motor_speeds'][i], dtype=dtype, device=device
        ).reshape(1, 1, 1)

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    # Build start state
    d0 = gt_data[0]
    state_vals = []
    for r in range(n_rods):
        state_vals.extend(
            d0['pos'][r * 3:(r + 1) * 3]
            + d0['quat'][r * 4:(r + 1) * 4]
            + d0['linvel'][r * 3:(r + 1) * 3]
            + d0['angvel'][r * 3:(r + 1) * 3]
        )
    start_state = torch.tensor(state_vals, dtype=dtype).reshape(1, -1, 1).to(device)

    # Run GNN rollout (ground truth)
    print("Running GNN rollout (ground truth)...")
    gnn_states = [start_state.clone()]
    with torch.no_grad():
        curr_state = start_state.clone()
        for k, extra in enumerate(tqdm.tqdm(extra_data)):
            ctrl = torch.tensor(extra['controls'], dtype=dtype, device=device).reshape(1, -1, 1)
            s2g_kwargs = {'dataset_idx': torch.tensor([[9]], dtype=torch.long, device=device)}
            out = simulator.step(curr_state, ctrls=ctrl, state_to_graph_kwargs=s2g_kwargs)
            curr_state = out[0] if isinstance(out, tuple) else out
            gnn_states.append(curr_state.detach().clone())

    # Reset simulator
    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    # Generate synthetic noisy measurements from GNN
    print("Generating synthetic measurements with noise...")
    sensor_std = np.sqrt(float(measurement_noise)) * np.sqrt(float(sensor_noise_scale))
    noisy_measurements = []
    filtered_states = [start_state.clone()]

    for i, gnn_state in enumerate(gnn_states):
        gnn_np = gnn_state.detach().cpu().numpy().reshape(-1)
        # Add noise to pose only (position + quaternion)
        noisy = gnn_np.copy()
        noisy[:3*n_rods] += np.random.normal(0, sensor_std, 3*n_rods)
        noisy[3*n_rods:7*n_rods] += np.random.normal(0, sensor_std*5, 4*n_rods)  # Quat noise
        noisy_measurements.append(noisy)

    # Run EKF
    print("Running EKF with GNN dynamics...")
    Q_sigmas = _structured_Q_sigmas(state_dim, n_rods, np.sqrt(process_noise), 2.0, 2.0)
    R_sigmas = _structured_R_sigmas(7*n_rods, n_rods, np.sqrt(measurement_noise))

    ekf = OnlineEKF(
        simulator, dt, n_rods,
        process_noise_scale=process_noise,
        measurement_noise_scale=measurement_noise,
        observe_pose_only=True,
        use_finite_diff=False,
        innovation_gate_sigma=np.inf,
        dataset_idx_val=9
    )
    ekf.initialize(start_state)

    innovations = []
    covs = []

    for k in tqdm.tqdm(range(len(extra_data))):
        ctrl = torch.tensor(extra_data[k]['controls'], dtype=dtype, device=device).reshape(1, -1, 1)
        z = noisy_measurements[k+1][:7*n_rods]  # Pose only

        state_out = ekf.step(z, u_t=ctrl, have_measurement=True)
        filtered_states.append(state_out.detach().clone())

        # Extract innovation from GTSAM state
        mean_pred = np.array(ekf.state_gtsam.mean()).reshape(-1)
        H_np = ekf.H_np
        innovation = z.reshape(-1) - (H_np @ mean_pred)
        innovations.append(innovation)

        # Extract covariance
        try:
            P = np.asarray(ekf.state_gtsam.covariance(), dtype=np.float64)
            covs.append(P)
        except RuntimeError:
            pass

    # Analyze results
    print("\n" + "="*60)
    print("EKF VALIDATION RESULTS")
    print("="*60)

    # Residuals
    print("\n[1] RESIDUALS (Innovation should be white noise)")
    residual_results = check_residual_properties(innovations)
    plot_residuals(innovations, output_dir)
    print(f"  Mean innovation (abs):     {residual_results['innovation_mean']:.6f}")
    print(f"  Std innovation:            {residual_results['innovation_std']:.6f}")
    print(f"  Normality test p-value:    {residual_results['normality_test_p']:.4f} {'✓ PASS' if residual_results['normality_pass'] else '✗ FAIL'}")
    print(f"  Lag-1 autocorrelation:     {residual_results['autocorr_lag1']:.4f} {'✓ PASS' if residual_results['autocorr_pass'] else '✗ FAIL'}")

    # Covariance
    print("\n[2] COVARIANCE HEALTH")
    cov_results = check_covariance_health(covs)
    print(f"  All positive definite:     {cov_results['all_positive_definite']} {'✓ PASS' if cov_results['all_positive_definite'] else '✗ FAIL'}")
    if cov_results['eigenvalue_issues']:
        print(f"  Issues at steps: {[x['step'] for x in cov_results['eigenvalue_issues'][:3]]}{'...' if len(cov_results['eigenvalue_issues']) > 3 else ''}")
    print(f"  Mean log(det(P)):          {cov_results['log_det_mean']:.2f}")
    print(f"  Std log(det(P)):           {cov_results['log_det_std']:.2f}")

    # Performance
    print("\n[3] PERFORMANCE vs GROUND TRUTH")
    perf = compare_gnn_vs_filtered(gnn_states, filtered_states,
                                   [np.array(gt_data[i]['pos'] + gt_data[i]['quat'] +
                                            gt_data[i]['linvel'] + gt_data[i]['angvel'])
                                    for i in range(len(gt_data))],
                                   n_rods)
    print(f"  GNN full state MSE:        {perf['gnn_full_state_mse']:.6f}")
    print(f"  EKF full state MSE:        {perf['filtered_full_state_mse']:.6f}")
    print(f"  Improvement:               {perf['improvement_full']*100:+.2f}% {'✓' if perf['improvement_full'] > 0 else '✗'}")
    print(f"  Position MSE improvement:  {perf['improvement_position']*100:+.2f}%")
    print(f"  Rotation MSE improvement:  {perf['improvement_rotation']*100:+.2f}%")

    # Summary
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    if residual_results['normality_pass'] and residual_results['autocorr_pass']:
        print("✓ Residuals look like white noise — filter tuning is good")
    else:
        print("✗ Residuals have structure — consider tuning Q/R")

    if cov_results['all_positive_definite']:
        print("✓ Covariance is healthy — no numerical issues")
    else:
        print("✗ Covariance has issues — may need numerical stabilization")

    if perf['improvement_full'] > 0:
        print(f"✓ EKF improves over GNN by {perf['improvement_full']*100:.1f}%")
    elif perf['improvement_full'] > -0.05:
        print(f"≈ EKF similar to GNN (within {perf['improvement_full']*100:.1f}%)")
    else:
        print(f"✗ EKF worse than GNN — filter may be hurting predictions")

    # Save summary
    summary = {
        'residuals': residual_results,
        'covariance': cov_results,
        'performance': perf,
    }
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to {output_dir}/summary.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt",
                        help='Path to trained .pt model file')
    parser.add_argument('--data_dir', type=str,
                        default="/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/3bar_new_platform_high_friction/dataset_0/traj_6",
                        help='Directory with processed_data.json and extra_state_data.json')
    parser.add_argument('--process-noise', type=float, default=1e-4)
    parser.add_argument('--measurement-noise', type=float, default=1e-3)
    parser.add_argument('--sensor-noise', type=float, default=1e-3,
                        help='Scale of synthetic sensor noise in measurements')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--output-dir', default='ekf_validation')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / 'processed_data.json'
    extra_data_path = data_dir / 'extra_state_data.json'

    validate_ekf(
        args.model_path, data_path, extra_data_path,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        sensor_noise_scale=args.sensor_noise,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )
