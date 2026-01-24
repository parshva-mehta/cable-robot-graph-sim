import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import tqdm
import numpy as np

from simulators.tensegrity_gnn_simulator import TensegrityGNNSimulator
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
        state_to_graph_kwargs={'dataset_idx': 9},
        show_progress=True
    )
    poses.extend([s.reshape(-1, 13, 1)[:, :7].reshape(1, -1, 1) for s in all_states])

    return poses


def evaluate(simulator,
             gt_data,
             ctrls,
             init_rest_lengths,
             init_motor_speeds):
    cables = simulator.actuated_cables.values()
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - init_rest_lengths[i]
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1)

    pos, quat = gt_data[0]['pos'], gt_data[0]['quat']
    linvel, angvel = gt_data[0]['linvel'], gt_data[0]['angvel']

    start_state = torch.tensor(
        pos + quat + linvel + angvel,
        dtype=DEFAULT_DTYPE
    ).reshape(1, 13, 1)

    rollout_poses = rollout_by_ctrls(
        simulator,
        ctrls,
        start_state
    )

    com_errs, rot_errs, pen_errs = [], [], []
    for i in range(1, len(gt_data)):
        gt_pos = torch.tensor(
            gt_data[i]['pos'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 3, 1)

        gt_quat = torch.tensor(
            gt_data[i]['quat'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 4, 1)

        pred_pos = rollout_poses[i]['pose'][:, :3]
        pred_quat = rollout_poses[i]['pose'][:, 3:7]

        com_mse = ((gt_pos - pred_pos) ** 2).mean()
        ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

        gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
        pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
        pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

        com_errs.append(com_mse.item())
        rot_errs.append(ang_err.item())
        pen_errs.append(pen_err.item())

    avg_com_err = sum(com_errs) / len(com_errs)
    avg_rot_err = sum(rot_errs) / len(rot_errs)
    avg_pen_err = sum(pen_errs) / len(pen_errs)

    return avg_com_err, avg_rot_err, avg_pen_err


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

