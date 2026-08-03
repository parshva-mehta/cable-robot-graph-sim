import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import tqdm
import numpy as np

from simulators.tensegrity_gnn_simulator import TensegrityGNNSimulator, load_simulator
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE


def rollout_by_ctrls(simulator,
                     ctrls,
                     start_state,
                     dataset_idx=0):
    poses = []

    curr_state = start_state \
        if start_state is not None \
        else simulator.get_curr_state()
    pose = curr_state.reshape(-1, 13, 1)[:, :7].reshape(curr_state.shape[0], -1, 1)
    poses.append(pose)

    all_states, graphs, _ = simulator.run(
        curr_state=curr_state,
        ctrls=ctrls,
        state_to_graph_kwargs={'dataset_idx': torch.tensor([[dataset_idx]], dtype=torch.long, device=curr_state.device)},
        show_progress=True
    )
    poses.extend([s.reshape(-1, 13, 1)[:, :7].reshape(1, -1, 1) for s in all_states])

    return poses


def evaluate(simulator,
             gt_data,
             ctrls,
             init_rest_lengths,
             init_motor_speeds,
             dataset_idx=0):
    cables = list(simulator.robot.actuated_cables.values())
    dev = cables[0]._rest_length.device
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            init_rest_lengths[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(dev)
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(dev)

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    num_rods = len(simulator.robot.rigid_bodies)
    d0 = gt_data[0]
    pos, quat = d0['pos'], d0['quat']
    linvel, angvel = d0['linvel'], d0['angvel']
    state_vals = []
    for r in range(num_rods):
        state_vals.extend(
            pos[r * 3:(r + 1) * 3] + quat[r * 4:(r + 1) * 4]
            + linvel[r * 3:(r + 1) * 3] + angvel[r * 3:(r + 1) * 3]
        )
    start_state = torch.tensor(state_vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(dev)

    with torch.no_grad():
        rollout_poses = rollout_by_ctrls(simulator, ctrls, start_state, dataset_idx)

    num_steps = min(len(rollout_poses) - 1, len(gt_data) - 1)
    com_errs, swing_errs, rot_errs, pen_errs = [], [], [], []
    for i in range(1, num_steps + 1):
        # rollout_poses[i]: (1, 7*num_rods, 1)
        pose_tensor = rollout_poses[i]
        for r in range(num_rods):
            pred_pos = pose_tensor[:, r * 7:r * 7 + 3, 0]    # (1, 3)
            pred_quat = pose_tensor[:, r * 7 + 3:r * 7 + 7, 0]  # (1, 4)

            gt_pos = torch.tensor(
                gt_data[i]['pos'][r * 3:(r + 1) * 3],
                dtype=DEFAULT_DTYPE
            ).reshape(1, 3).to(dev)
            gt_quat = torch.tensor(
                gt_data[i]['quat'][r * 4:(r + 1) * 4],
                dtype=DEFAULT_DTYPE
            ).reshape(1, 4).to(dev)

            com_mse = ((gt_pos - pred_pos) ** 2).mean()
            swing_err = torch_quaternion.compute_swing_angle_btwn_quats(
                gt_quat, pred_quat
            )
            ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

            gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
            pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
            pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

            com_errs.append(com_mse.item())
            swing_errs.append(swing_err.mean().item())
            rot_errs.append(ang_err.mean().item())
            pen_errs.append(pen_err.mean().item())

    return {
        'com': sum(com_errs) / len(com_errs),
        'rot_swing': sum(swing_errs) / len(swing_errs),
        'rot_full': sum(rot_errs) / len(rot_errs),
        'pen': sum(pen_errs) / len(pen_errs),
    }


def evaluate_from_frames(frames, gt_data, n_rods, device, is_exp=False):
    """Compute COM, rotation, and penetration errors from EKF frames vs gt_data.

    frames[i]['state'] must be (1, state_dim, 1).  Pass is_exp=True for the
    exp-map EKF so the 36D state is converted to quat before extracting pos/quat.
    """
    if is_exp:
        from linearization_exp import exp_state_to_quat_state

    num_steps = min(len(frames) - 1, len(gt_data) - 1)
    com_errs, swing_errs, rot_errs, pen_errs = [], [], [], []
    for i in range(1, num_steps + 1):
        state_t = frames[i]['state']  # (1, state_dim, 1)
        if is_exp:
            state_t = exp_state_to_quat_state(state_t)  # → (1, 39, 1)
        state_np = state_t.squeeze().cpu()  # (39,)

        for r in range(n_rods):
            pred_pos = state_np[r * 13: r * 13 + 3].reshape(1, 3).to(device)
            pred_quat = state_np[r * 13 + 3: r * 13 + 7].reshape(1, 4).to(device)

            gt_pos = torch.tensor(
                gt_data[i]['pos'][r * 3:(r + 1) * 3], dtype=DEFAULT_DTYPE
            ).reshape(1, 3).to(device)
            gt_quat = torch.tensor(
                gt_data[i]['quat'][r * 4:(r + 1) * 4], dtype=DEFAULT_DTYPE
            ).reshape(1, 4).to(device)

            com_mse = ((gt_pos - pred_pos) ** 2).mean()
            swing_err = torch_quaternion.compute_swing_angle_btwn_quats(
                gt_quat, pred_quat
            )
            ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

            gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
            pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
            pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

            com_errs.append(com_mse.item())
            swing_errs.append(swing_err.mean().item())
            rot_errs.append(ang_err.mean().item())
            pen_errs.append(pen_err.mean().item())

    return {
        'com': sum(com_errs) / len(com_errs),
        'rot_swing': sum(swing_errs) / len(swing_errs),
        'rot_full': sum(rot_errs) / len(rot_errs),
        'pen': sum(pen_errs) / len(pen_errs),
    }


def _states_to_rod_poses(states, n_rods, stride):
    """Extract per-rod (pos+quat) poses from a list of flat state tensors.

    states: iterable of tensors that flatten to (n_rods*stride,), where each
            rod block begins with 3 pos + 4 quat values.
    Returns a list of (n_rods, 7) CPU tensors, one per timestep.
    """
    poses = []
    for s in states:
        flat = s.reshape(-1).cpu()
        poses.append(torch.stack([flat[r * stride:r * stride + 7] for r in range(n_rods)]))
    return poses


def _gt_to_rod_poses(gt_data, n_rods):
    """Build per-rod (pos+quat) poses from the gt_data json list."""
    poses = []
    for d in gt_data:
        rods = [
            torch.tensor(
                d['pos'][r * 3:(r + 1) * 3] + d['quat'][r * 4:(r + 1) * 4],
                dtype=DEFAULT_DTYPE,
            )
            for r in range(n_rods)
        ]
        poses.append(torch.stack(rods))
    return poses


def _frames_to_rod_poses(frames, n_rods, is_exp=True):
    """Extract per-rod (pos+quat) poses from EKF frames."""
    if is_exp:
        from linearization_exp import exp_state_to_quat_state
    states = []
    for frame in frames:
        state_t = frame['state']  # (1, state_dim, 1)
        if is_exp:
            state_t = exp_state_to_quat_state(state_t)  # → (1, 13*n_rods, 1)
        states.append(state_t)
    return _states_to_rod_poses(states, n_rods, stride=13)


def pairwise_pose_error(poses_a, poses_b, device):
    """COM (MSE) and rotation (mean angle) error between two pose sequences.

    poses_a / poses_b: lists of (n_rods, 7) tensors. Compares from t=1 onward
    (t=0 is the shared initial state). Returns (com_mse, rot_err) averaged over
    all rods and timesteps.
    """
    num_steps = min(len(poses_a), len(poses_b))
    com_errs, rot_errs = [], []
    for i in range(1, num_steps):
        pa, pb = poses_a[i], poses_b[i]
        for r in range(pa.shape[0]):
            pos_a = pa[r, :3].reshape(1, 3).to(device)
            pos_b = pb[r, :3].reshape(1, 3).to(device)
            quat_a = pa[r, 3:7].reshape(1, 4).to(device)
            quat_b = pb[r, 3:7].reshape(1, 4).to(device)

            com_errs.append(((pos_a - pos_b) ** 2).mean().item())
            ang = torch_quaternion.compute_angle_btwn_quats(quat_a, quat_b)
            rot_errs.append(ang.mean().item())

    if not com_errs:
        return 0.0, 0.0
    return sum(com_errs) / len(com_errs), sum(rot_errs) / len(rot_errs)


def run_raw_gnn_poses(simulator, gt_data, ctrls,
                      init_rest_lengths, init_motor_speeds, num_rods,
                      dataset_idx=0):
    """Reset sim state and run a raw GNN rollout; return per-rod pose sequence.

    Mirrors evaluate()'s reset logic so the rollout matches the raw baseline,
    but returns the (n_rods, 7) poses (incl. t=0) instead of aggregate errors.
    """
    cables = list(simulator.robot.actuated_cables.values())
    dev = cables[0]._rest_length.device
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            init_rest_lengths[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(dev)
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(dev)

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    d0 = gt_data[0]
    pos, quat = d0['pos'], d0['quat']
    linvel, angvel = d0['linvel'], d0['angvel']
    state_vals = []
    for r in range(num_rods):
        state_vals.extend(
            pos[r * 3:(r + 1) * 3] + quat[r * 4:(r + 1) * 4]
            + linvel[r * 3:(r + 1) * 3] + angvel[r * 3:(r + 1) * 3]
        )
    start_state = torch.tensor(state_vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(dev)

    with torch.no_grad():
        rollout_poses = rollout_by_ctrls(simulator, ctrls, start_state, dataset_idx)

    # rollout_by_ctrls returns (1, 7*num_rods, 1) per step (stride 7, de-interleaved)
    return _states_to_rod_poses(rollout_poses, num_rods, stride=7)


def write_frames_to_file(frames, output_path, mode):
    """Write EKF frames to a rollout_states-format text file."""
    if mode == 'ekf_exp':
        from linearization_exp import exp_state_to_quat_state

    with open(output_path, 'w') as f:
        for frame in frames:
            state_t = frame['state']  # (1, state_dim, 1)
            if mode == 'ekf_exp':
                state_t = exp_state_to_quat_state(state_t)  # → (1, 39, 1)
            row = state_t.squeeze().cpu().numpy()  # (39,)
            f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
    print(f'Wrote {len(frames)} timesteps to {output_path}')


def compute_end_pts_from_state(rod_pos_state, principal_axis, rod_length):
    """
    :param rod_pos_state: (x, y, z, quat.w, quat.x, quat.y, quat.z)
    :param principal_axis: tensor of vector(s)
    :param rod_length: length of rod
    :return: ((x1, y1, z1), (x2, y2, z2))
    """
    # Get position
    pos = rod_pos_state[:, :3, ...]

    # Compute half-length vector from principal axis
    half_length_vec = rod_length * principal_axis / 2

    # End points are +/- of half-length vector from COM
    end_pt1 = pos - half_length_vec
    end_pt2 = pos + half_length_vec

    return [end_pt1, end_pt2]


def batch_compute_end_pts(sim, batch_state: torch.Tensor):
    """
    Compute end pts for entire batch

    :param batch_state: batch of states
    :return: list of endpts
    """
    end_pts = []
    for i, rod in enumerate(sim.rigid_bodies.values()):
        pose = batch_state[:, i * 13: i * 13 + 7]
        principal_axis = torch_quaternion.quat_as_rot_mat(pose[:, 3:7])[..., 2:]
        end_pts.extend(compute_end_pts_from_state(pose, principal_axis, rod.length))

    return torch.hstack(end_pts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt",
                        #default="C:/Users/parshva-mehta/Documents/Projects/PRACSYS/Tensegrity/tensegrity/models/best_rollout_model.pt",
                        help='Path to trained .pt model file')
    parser.add_argument('--data_dir', type=str,
                        default="/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/3bar_new_platform_high_friction/dataset_0/traj_6",
                        #default="C:/Users/parshva-mehta/Documents/Projects/PRACSYS/Tensegrity/tensegrity/data_sets/3bar_new_platform_high_friction/dataset_0/traj_6",
                        help='Directory with processed_data.json and extra_state_data.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output rollout text file (default derived from --mode)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--torch_threads', type=int, default=1,
                        help='CPU threads for torch. The per-step tensors here are '
                             'small enough that thread sync costs more than it saves: '
                             'measured 16.9 ms/forward at 1 thread vs 22.1 ms at 8. '
                             'Use 0 to leave the torch default untouched.')
    parser.add_argument('--mode', choices=['raw', 'ekf', 'gtsam'], default='raw',
                        help='raw: pure GNN rollout; ekf: exp-map MEKF; gtsam: GTSAM-based MEKF')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Timestep used by EKF modes')
    parser.add_argument('--process_noise', type=float, default=1e-2,
                        help='Q scale: small = trust GNN model strongly (stable)')
    parser.add_argument('--measurement_noise', type=float, default=1e-1,
                        help='R scale: large = low Kalman gain, fewer jumps')
    parser.add_argument('--observe_pose_only', action='store_true', default=True,
                        help='Observe only pose (pos+quat); skip velocities')
    parser.add_argument('--innovation_gate', type=float, default=3.0,
                        help='Reject updates with ||innovation|| > gate*sqrt(meas_dim)')
    parser.add_argument('--jac_update_period', type=int, default=5,
                        help='Recompute EKF Jacobian every N steps (1=every step, 5=5x speedup)')
    parser.add_argument('--max_spectral_radius', type=float, default=1.0,
                        help='Clamp threshold for EKF Jacobian spectral radius')
    parser.add_argument('--dataset_idx', type=int, default=0,
                        help='Dataset one-hot index. Training used 0/1/2 for '
                             'dataset_0/1/2; unused indices 3-9 are null '
                             'embeddings (identical, degraded rollouts). '
                             'This eval data is dataset_0 -> idx 0.')
    parser.add_argument('--compare_raw', action='store_true', default=False,
                        help='Also run raw GNN rollout alongside EKF and print comparison '
                             '(ekf mode only; ratio≈1.0 means EKF not correcting)')
    parser.add_argument('--log_kalman_diagnostics', action='store_true', default=False,
                        help='Print per-step Kalman gain proxy for position block '
                             '(ekf mode only; shows innovation vs correction norms)')
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # Derive output filename from mode when not explicitly provided
    if args.output is None:
        output_names = {
            'raw':   'rollout_states.txt',
            'ekf':   'rollout_states_ekf.txt',
            'gtsam': 'rollout_states_gtsam.txt',
        }
        args.output = output_names[args.mode]

    # Always deserialize checkpoints onto CPU first to avoid backend-specific
    # restore issues (e.g., stale MPS/CUDA device tags across machines).
    load_device = torch.device('cpu')

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
    elif args.device == 'mps' and torch.backends.mps.is_available():
        # Note: measured ~4x SLOWER than CPU for this model (58.9 ms vs 14.1 ms
        # per forward). The GEMMs are too small to fill the GPU, so per-kernel
        # dispatch dominates. Selectable, but CPU is the faster choice on Apple.
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        if args.device != 'cpu':
            print(f"{args.device} unavailable, using CPU")

    simulator = load_simulator(args.model_path, map_location=load_device, cache_batch_sizes=[1])
    simulator = simulator.to(device)
    simulator.eval()

    data_dir = Path(args.data_dir)
    with open(data_dir / 'processed_data.json') as f:
        gt_data = json.load(f)
    with open(data_dir / 'extra_state_data.json') as f:
        extra_data = json.load(f)

    num_rods = len(simulator.robot.rigid_bodies)

    # Build initial state: interleave per-rod values from flat json arrays.
    # Each rod: x y z qw qx qy qz vx vy vz wx wy wz (13 values)
    d0 = gt_data[0]
    pos, quat = d0['pos'], d0['quat']
    linvel, angvel = d0['linvel'], d0['angvel']
    state_vals = []
    for r in range(num_rods):
        state_vals.extend(
            pos[r * 3:(r + 1) * 3] + quat[r * 4:(r + 1) * 4]
            + linvel[r * 3:(r + 1) * 3] + angvel[r * 3:(r + 1) * 3]
        )
    start_state = torch.tensor(state_vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)

    # Controls: (1, num_cables, T)
    ctrls = torch.tensor(
        [e['controls'] for e in extra_data], dtype=DEFAULT_DTYPE
    ).T.unsqueeze(0).to(device)

    # Reset cable and motor state to match start of trajectory
    init_rest_lengths = extra_data[0]['rest_lengths']
    init_motor_speeds = extra_data[0]['motor_speeds']
    cables = list(simulator.robot.actuated_cables.values())
    for i, cable in enumerate(cables):
        cable.actuation_length = cable._rest_length - torch.tensor(
            init_rest_lengths[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)
        cable.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i], dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1).to(device)

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    if args.mode == 'raw':
        with torch.no_grad():
            all_states, _, _ = simulator.run(
                curr_state=start_state,
                ctrls=ctrls,
                state_to_graph_kwargs={
                    'dataset_idx': torch.tensor([[args.dataset_idx]], dtype=torch.long).to(device)
                },
                show_progress=True
            )

        with open(args.output, 'w') as f:
            f.write(' '.join(f'{v:.8f}' for v in state_vals) + '\n')
            for state in all_states:
                row = state.squeeze().cpu().numpy()
                f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
        print(f'Wrote {len(all_states) + 1} timesteps to {args.output}')

        errs = evaluate(
            simulator, gt_data, ctrls, init_rest_lengths, init_motor_speeds,
            dataset_idx=args.dataset_idx
        )

    elif args.mode == 'ekf':
        from ekf import run_ekf_rollout

        frames = run_ekf_rollout(
            simulator, gt_data, extra_data,
            dt=args.dt,
            process_noise_scale=args.process_noise,
            measurement_noise_scale=args.measurement_noise,
            observe_pose_only=args.observe_pose_only,
            start_state=start_state,
            use_finite_diff=False,
            innovation_gate_sigma=args.innovation_gate,
            dataset_idx_val=args.dataset_idx,
            max_spectral_radius=args.max_spectral_radius,
            log_diagnostics=args.log_kalman_diagnostics,
        )
        write_frames_to_file(frames, args.output, mode='ekf_exp')
        errs = evaluate_from_frames(
            frames, gt_data, num_rods, device, is_exp=True
        )

    else:  # gtsam
        from ekf_gtsam import run_ekf_rollout as run_gtsam_rollout

        frames = run_gtsam_rollout(
            simulator, gt_data, extra_data,
            dt=args.dt,
            process_noise_scale=args.process_noise,
            measurement_noise_scale=args.measurement_noise,
            observe_pose_only=args.observe_pose_only,
            start_state=start_state,
            use_finite_diff=False,
            innovation_gate_sigma=args.innovation_gate,
            dataset_idx_val=args.dataset_idx,
            max_spectral_radius=args.max_spectral_radius,
            log_diagnostics=args.log_kalman_diagnostics,
        )
        write_frames_to_file(frames, args.output, mode='ekf_exp')
        errs = evaluate_from_frames(
            frames, gt_data, num_rods, device, is_exp=True
        )

    print(f'COM Error (MSE):            {errs["com"]:.6f} m\u00b2')
    print(f'Rotation Error (swing):     {errs["rot_swing"]:.6f} rad')
    print(f'Rotation Error (full-quat): {errs["rot_full"]:.6f} rad  '
          f'prev method of measurment')
    print(f'Penetration Error:          {errs["pen"]:.6f} m')

    if args.compare_raw and args.mode in ('ekf', 'gtsam'):
        # run_raw_gnn_poses() reinitializes cables/motor/LSTM before running, so
        # it is safe to call after the EKF has consumed the simulator.
        gnn_poses = run_raw_gnn_poses(
            simulator, gt_data, ctrls, init_rest_lengths, init_motor_speeds,
            num_rods, dataset_idx=args.dataset_idx
        )
        ekf_poses = _frames_to_rod_poses(frames, num_rods, is_exp=True)
        gt_poses = _gt_to_rod_poses(gt_data, num_rods)

        # Pairwise pose errors between the prediction algorithms and ground truth.
        ekf_gt_com, ekf_gt_rot = pairwise_pose_error(ekf_poses, gt_poses, device)
        gnn_gt_com, gnn_gt_rot = pairwise_pose_error(gnn_poses, gt_poses, device)
        ekf_gnn_com, ekf_gnn_rot = pairwise_pose_error(ekf_poses, gnn_poses, device)

        print(f'\n=== Pose-algorithm comparison ===')
        com_hdr = 'COM Error (MSE, m\u00b2)'.rjust(22)
        rot_hdr = 'Rotation Error (rad)'.rjust(22)
        print(f'{"":<16}{com_hdr}{rot_hdr}')
        print(f'{"EKF  vs GT":<16}{ekf_gt_com:>22.6f}{ekf_gt_rot:>22.6f}')
        print(f'{"GNN  vs GT":<16}{gnn_gt_com:>22.6f}{gnn_gt_rot:>22.6f}')
        print(f'{"EKF  vs GNN":<16}{ekf_gnn_com:>22.6f}{ekf_gnn_rot:>22.6f}')

        if gnn_gt_com > 1e-12:
            ratio = ekf_gt_com / gnn_gt_com
            print(f'\nEKF/GNN COM ratio (vs GT): {ratio:.3f}  '
                  f'(< 1.0 = EKF improves, \u2248 1.0 = EKF not correcting position)')


if __name__ == '__main__':
    main()

