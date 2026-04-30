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
                     start_state):
    poses = []

    curr_state = start_state \
        if start_state is not None \
        else simulator.get_curr_state()
    pose = curr_state.reshape(-1, 13, 1)[:, :7].reshape(curr_state.shape[0], -1, 1)
    poses.append(pose)

    all_states, graphs, _ = simulator.run(
        curr_state=curr_state,
        ctrls=ctrls,
        state_to_graph_kwargs={'dataset_idx': torch.tensor([[9]], dtype=torch.long, device=curr_state.device)},
        show_progress=True
    )
    poses.extend([s.reshape(-1, 13, 1)[:, :7].reshape(1, -1, 1) for s in all_states])

    return poses


def evaluate(simulator,
             gt_data,
             ctrls,
             init_rest_lengths,
             init_motor_speeds):
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
        rollout_poses = rollout_by_ctrls(simulator, ctrls, start_state)

    num_steps = min(len(rollout_poses) - 1, len(gt_data) - 1)
    com_errs, rot_errs, pen_errs = [], [], []
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
            ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

            gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
            pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
            pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

            com_errs.append(com_mse.item())
            rot_errs.append(ang_err.mean().item())
            pen_errs.append(pen_err.mean().item())

    avg_com_err = sum(com_errs) / len(com_errs)
    avg_rot_err = sum(rot_errs) / len(rot_errs)
    avg_pen_err = sum(pen_errs) / len(pen_errs)

    return avg_com_err, avg_rot_err, avg_pen_err


def evaluate_from_frames(frames, gt_data, n_rods, device, is_exp=False):
    """Compute COM, rotation, and penetration errors from EKF frames vs gt_data.

    frames[i]['state'] must be (1, state_dim, 1).  For ekf_exp mode pass
    is_exp=True so the 36D exp-map state is converted to quat before extracting
    pos/quat for metrics.
    """
    if is_exp:
        from linearization_exp import exp_state_to_quat_state

    num_steps = min(len(frames) - 1, len(gt_data) - 1)
    com_errs, rot_errs, pen_errs = [], [], []
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
            ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

            gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
            pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
            pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

            com_errs.append(com_mse.item())
            rot_errs.append(ang_err.mean().item())
            pen_errs.append(pen_err.mean().item())

    return (
        sum(com_errs) / len(com_errs),
        sum(rot_errs) / len(rot_errs),
        sum(pen_errs) / len(pen_errs),
    )


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
                        help='Path to trained .pt model file')
    parser.add_argument('--data_dir', type=str,
                        default="/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/3bar_new_platform_high_friction/dataset_0/traj_6",
                        help='Directory with processed_data.json and extra_state_data.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output rollout text file (default derived from --mode)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', choices=['raw', 'ekf', 'ekf_exp'], default='raw',
                        help='raw: pure GNN rollout; ekf: quat-space EKF; ekf_exp: exp-map EKF')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Timestep used by EKF modes')
    parser.add_argument('--process_noise', type=float, default=1e-4)
    parser.add_argument('--measurement_noise', type=float, default=1e-3)
    parser.add_argument('--observe_pose_only', action='store_true')
    parser.add_argument('--innovation_gate', type=float, default=float('inf'))
    parser.add_argument('--dataset_idx', type=int, default=9)
    args = parser.parse_args()

    # Derive output filename from mode when not explicitly provided
    if args.output is None:
        output_names = {
            'raw':     'rollout_states.txt',
            'ekf':     'rollout_states_ekf.txt',
            'ekf_exp': 'rollout_states_ekf_exp.txt',
        }
        args.output = output_names[args.mode]

    # Always deserialize checkpoints onto CPU first to avoid backend-specific
    # restore issues (e.g., stale MPS/CUDA device tags across machines).
    load_device = torch.device('cpu')

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
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

        com_err, rot_err, pen_err = evaluate(
            simulator, gt_data, ctrls, init_rest_lengths, init_motor_speeds
        )

    elif args.mode == 'ekf':
        from ekf import run_ekf_rollout as run_ekf_quat

        frames = run_ekf_quat(
            simulator, gt_data, extra_data,
            dt=args.dt,
            process_noise_scale=args.process_noise,
            measurement_noise_scale=args.measurement_noise,
            observe_pose_only=args.observe_pose_only,
            start_state=start_state,
            use_finite_diff=True,
            innovation_gate_sigma=args.innovation_gate,
            dataset_idx_val=args.dataset_idx,
        )
        write_frames_to_file(frames, args.output, mode='ekf')
        com_err, rot_err, pen_err = evaluate_from_frames(
            frames, gt_data, num_rods, device, is_exp=False
        )

    else:  # ekf_exp
        from ekf_alt import run_ekf_rollout as run_ekf_exp

        frames = run_ekf_exp(
            simulator, gt_data, extra_data,
            dt=args.dt,
            process_noise_scale=args.process_noise,
            measurement_noise_scale=args.measurement_noise,
            observe_pose_only=args.observe_pose_only,
            start_state=start_state,
            use_finite_diff=False,
            innovation_gate_sigma=args.innovation_gate,
            dataset_idx_val=args.dataset_idx,
        )
        write_frames_to_file(frames, args.output, mode='ekf_exp')
        com_err, rot_err, pen_err = evaluate_from_frames(
            frames, gt_data, num_rods, device, is_exp=True
        )

    print(f'COM Error (MSE):       {com_err:.6f} m\u00b2')
    print(f'Rotation Error (mean): {rot_err:.6f} rad')
    print(f'Penetration Error:     {pen_err:.6f} m')


if __name__ == '__main__':
    main()

