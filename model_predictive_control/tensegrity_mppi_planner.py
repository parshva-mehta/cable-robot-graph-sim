"""
Tensegrity robot motion planning using Model Predictive Path Integral (MPPI) control.

This module implements an MPPI-based planner for tensegrity cable-driven robots,
combining sampling-based trajectory optimization with grid-based heuristics for
obstacle avoidance and goal-directed navigation.
"""

from logging import Logger
from pathlib import Path

import numpy as np
import torch.nn
from torch.distributions import Uniform

from model_predictive_control.mppi_utils import wave_heuristic_dict_to_arr, snap_to_grid_torch, unsnap_to_grid_torch, \
    heuristic_dir2, fill_grid
from simulators.abstract_simulator import AbstractSimulator
from utilities import torch_quaternion


class TensegrityMPPIPlanner(torch.nn.Module):
    """
    Model Predictive Path Integral (MPPI) planner for tensegrity cable robots.

    This planner uses sampling-based trajectory optimization to generate control
    sequences for cable-driven tensegrity robots. It combines:
    - Grid-based heuristic costs for distance-to-goal and obstacle avoidance
    - Directional alignment costs for orientation control
    - Terminal rewards for goal achievement

    The planner samples multiple control trajectories, evaluates their costs through
    forward simulation, and selects or weights them to produce an optimal control action.
    """

    def __init__(self,
                 sim: AbstractSimulator | str | Path,
                 n_samples: int,
                 horizon: float,
                 act_interval: float,
                 ctrl_interval: float,
                 device: str = 'cpu',
                 u_bounds: tuple = (-1., 1.),
                 gamma: float = 1.0,
                 rest_len_bounds: tuple = (0.8, 2.1),
                 goal: tuple | None = None,
                 obstacles: tuple = (),
                 boundary: tuple = (),
                 logger: Logger | None = None,
                 strategy: 'str' = 'min',
                 cost_weights: tuple = (1.0, 0.0, 0.0),
                 grid_step=0.1,
                 tol: float = 0.5,
                 min_vel_dt=0.04):
        """
        Initialize the MPPI planner.

        Args:
            sim: Simulator instance or path to saved simulator model
            n_samples: Number of trajectory samples to generate per planning iteration
            horizon: Planning horizon in seconds
            act_interval: Action interval in seconds (interval at which actions are applied)
            ctrl_interval: Control interval in seconds (interval at which controls are sampled)
            device: Torch device ('cpu' or 'cuda')
            u_bounds: Control input bounds as (min, max) tuple
            gamma: Discount factor for future costs (1.0 = no discounting)
            rest_len_bounds: Cable rest length bounds as (min, max) tuple in meters
            goal: Goal position as (x, y, z) tuple
            obstacles: Tuple of box obstacles, each as (xmin, xmax, ymin, ymax)
            boundary: Planning boundary as (xmin, xmax, ymin, ymax)
            logger: Logger instance for output
            strategy: Trajectory selection strategy ('min' or 'weighted')
                - 'min': Select trajectory with minimum cost
                - 'weighted': Weight trajectories by exponential of negative cost
            cost_weights: Weights for (distance, direction, obstacle) costs
            grid_step: Grid resolution for heuristic cost computation in meters
            tol: Goal tolerance threshold in meters
            min_vel_dt: Minimum time difference for velocity estimation in seconds
        """
        super().__init__()
        self.logger = logger

        if isinstance(sim, str) or isinstance(sim, Path):
            self.sim = torch.load(sim, map_location='cpu', weights_only=False)
        else:
            self.sim = sim

        self.sim.to(device)
        self.dt = self.sim.data_processor.dt.item()

        self.min_vel_dt = min_vel_dt
        self.n_samples = n_samples
        self.horizon = round(horizon / self.dt)
        self.act_interval = round(act_interval / self.dt)
        self.ctrl_interval = round(ctrl_interval / self.dt)
        self.dtype = self.sim.dtype
        self.device = device

        self.goal = None
        self.set_goals([np.array(goal)])

        self.obs_cost_gain = 50.0
        self.obs_min_dist = 0.5
        self.terminal_reward = -10.0
        self.goal_threshold = tol

        assert strategy in ['min', 'weighted']
        self.strategy = strategy

        self.gamma = gamma
        self.gamma_seq = torch.cumprod(
            torch.full((1, 1, self.horizon),
                       gamma,
                       dtype=self.dtype,
                       device=self.device),
            dim=-1
        )

        self.ctrl_min, self.ctrl_max = u_bounds
        self.rest_min, self.rest_max = rest_len_bounds
        self.n_ctrls = len(self.sim.robot.actuated_cables)
        self.prev_ctrls = torch.zeros(
            (1, self.n_ctrls, self.horizon // self.ctrl_interval),
            dtype=self.dtype,
            device=self.device
        )

        self.cost_weights = cost_weights

        self.boundary = boundary
        self.box_obstacles = obstacles

        self.grid_step = grid_step
        h_val = fill_grid(
            goal,
            boundary,
            self.grid_step,
            obstacles
        )
        self.dist_cost_grid, self.obs_cost_grid = wave_heuristic_dict_to_arr(
            h_val,
            boundary,
            obstacles,
            self.grid_step,
            self.device,
            self.dtype
        )

    def to(self, device):
        """
        Move planner and all associated tensors to specified device.

        Args:
            device: Target device ('cpu' or 'cuda')

        Returns:
            Self for method chaining
        """
        self.device = device
        self.dist_cost_grid = self.dist_cost_grid.to(device)
        self.obs_cost_grid = self.obs_cost_grid.to(device)
        self.gamma_seq = self.gamma_seq.to(device)
        self.sim = self.sim.to(device)
        self.prev_ctrls = self.prev_ctrls.to(device)
        self.goal = self.goal.to(device)

        return self

    @staticmethod
    def _compute_weight(cost, beta, factor):
        """
        Compute trajectory weights for MPPI weighted strategy.

        Args:
            cost: Cost values, shape (nsamples,)
            beta: Minimum cost value for normalization
            factor: Temperature factor for exponential weighting

        Returns:
            Weight values, shape (nsamples,)
        """
        return torch.exp(-factor * (cost - beta))

    def map(self, data):
        """
        Convert input data to torch tensor with planner's device and dtype.

        Args:
            data: Input data (numpy array, list, or torch tensor)

        Returns:
            Torch tensor on planner's device and dtype
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, list):
            if isinstance(data[0], np.ndarray):
                data = np.vstack(data)
                data = torch.from_numpy(data)
            else:
                data = torch.tensor(data)

        data = data.to(self.sim.device).to(self.sim.dtype)
        return data

    @staticmethod
    def endpts_to_5d_pose(end_pts):
        """
        Convert robot endpoints to 5D pose representation (position + quaternion).

        Args:
            end_pts: Endpoint positions, shape (batch, 6)

        Returns:
            Pose tensor with position and quaternion, shape (batch, 7, 1)
        """
        end_pts_ = end_pts.reshape(-1, 6, 1)
        curr_pos = (end_pts_[:, :3] + end_pts_[:, 3:]) / 2.
        prin = (end_pts_[:, 3:] - end_pts_[:, :3])
        curr_quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin)

        return torch.hstack([curr_pos, curr_quat])

    def rollout(self, curr_state_, batch_actions):
        """
        Simulate forward dynamics for a batch of action sequences.

        Args:
            curr_state_: Current state tensor, shape (batch, state_dim, 1)
            batch_actions: Batch of action sequences, shape (batch, n_ctrls, horizon)

        Returns:
            Tuple of (states, costs) where:
                - states: Concatenated state trajectory, shape (batch, state_dim, horizon+1)
                - costs: Tuple of (total_costs, dist_costs, dir_costs, obs_costs) as lists
        """
        states = [curr_state_.clone()]
        state_to_graph_kwargs = {'dataset_idx': torch.tensor([8]).repeat(curr_state_.shape[0]).reshape(-1, 1)}
        sim_states, _, _ = self.sim.run(
            curr_state_,
            ctrls=batch_actions,
            state_to_graph_kwargs=state_to_graph_kwargs,  # HACK
            show_progress=True
        )

        costs, dist_costs, dir_costs, obs_costs = [], [], [], []
        curr_dir = self.get_goal_dir(curr_state_.reshape(-1, 13, 1)[:, :7])
        for state in sim_states:
            dist_cost, dir_cost, obs_cost = self.all_costs(state, curr_dir)
            cost = (self.cost_weights[0] * dist_cost
                    + self.cost_weights[1] * dir_cost
                    + self.cost_weights[2] * obs_cost)

            costs.append(cost)
            dist_costs.append(self.cost_weights[0] * dist_cost.detach())
            dir_costs.append(self.cost_weights[1] * dir_cost.detach())
            obs_costs.append(self.cost_weights[2] * obs_cost.detach())

        states.extend(sim_states)
        states = torch.concat(states, dim=-1)

        return states, (costs, dist_costs, dir_costs, obs_costs)

    def set_goals(self, com_goal):
        """
        Set goal position for planning.

        Args:
            com_goal: Goal center-of-mass position as numpy array or list
        """
        self.goal = self.map(com_goal)
        self.goal[:, 2] = 0.

    def heuristic_grid_cost(self, curr_state):
        """
        Compute grid-based heuristic costs for distance-to-goal and obstacles.

        Uses precomputed wavefront propagation grids to efficiently evaluate
        distance costs and obstacle costs based on robot center-of-mass and
        endpoint positions.

        Args:
            curr_state: Current state tensor

        Returns:
            Tuple of (dist_cost, obs_cost) tensors, shape (batch,)
        """
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))
        idx = snap_to_grid_torch(com[:, :2], self.grid_step, self.boundary).squeeze(-1)
        dist_cost = self.dist_cost_grid[idx[:, 0], idx[:, 1]]

        end_pts = self.compute_end_pts(curr_state)
        idx = snap_to_grid_torch(end_pts[:, :2], self.grid_step, self.boundary).squeeze(-1)
        obs_cost = self.obs_cost_grid[idx[:, 0], idx[:, 1]].max(dim=1).values

        return dist_cost, obs_cost

    def multi_goal_costs(self, curr_state):
        """
        Compute costs for multi-waypoint trajectory following.

        Evaluates costs for following a path through multiple goal waypoints,
        penalizing deviation from the path and distance to final goal.

        Args:
            curr_state: Current state tensor

        Returns:
            Tuple of (total_costs, zero_costs), each shape (batch, 1)
        """
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))

        com_vecs = com[:, :2] - self.goals[:, :2, :-1]
        com_vecs2 = com[:, :2] - self.goals[:, :2, 1:]

        avg_dists = 0.5 * (com_vecs.norm(dim=1) + com_vecs2.norm(dim=1)).unsqueeze(1)
        min_goal_idx = avg_dists.argmin(dim=-1).flatten()
        goal = self.goals.repeat(com.shape[0], 1, 1)[torch.arange(com.shape[0]), :, min_goal_idx].unsqueeze(-1)
        goal_vec = self.goal_vecs.repeat(com.shape[0], 1, 1)[torch.arange(com.shape[0]), :, min_goal_idx].unsqueeze(-1)
        goal_vec_norm = self.goal_vecs_norm.repeat(com.shape[0], 1, 1)[torch.arange(com.shape[0]), :,
                        min_goal_idx].unsqueeze(-1)
        com_vec = com_vecs[torch.arange(com.shape[0]), :, min_goal_idx].unsqueeze(-1)

        t = torch.linalg.vecdot(com_vec, goal_vec, dim=1).unsqueeze(1)
        d = goal[:, 2:] - self.cost_slope * t
        d_lim = self.goals[0, 2:, min_goal_idx + 1].T.unsqueeze(-1)
        d = torch.clamp_min(d, d_lim)
        v = com_vec - t * goal_vec / (goal_vec_norm ** 2)
        v_norm = v.norm(dim=1, keepdim=True)

        off_traj_costs = (self.off_traj_cost * v_norm + 1) ** 2
        ramp_costs = (self.off_traj_cost * v_norm + 1) ** 2 + d
        # ramp_costs = ramp_costs.min(dim=-1, keepdim=True).values

        final_goal_dist_cost = (com[:, :2] - self.goals[:, :2, -1:]).norm(dim=1, keepdim=True)
        # costs = torch.maximum(ramp_costs, final_goal_dist_cost)
        costs = ramp_costs + final_goal_dist_cost
        # if ramp_costs.min(dim=0).values.item() < final_goal_dist_cost.

        return costs, torch.zeros_like(costs)[:, :1]

    def compute_end_pts(self, curr_state):
        """
        Compute robot endpoint positions from state.

        Args:
            curr_state: Current state tensor

        Returns:
            Endpoint positions tensor, shape (batch, 3, n_endpoints)
        """
        return self.sim.robot.compute_end_pts(curr_state)

    @staticmethod
    def _dist2d_pt_to_line_seg(pt, line_seg):
        """
        Compute 2D distance from points to line segments.

        Args:
            pt: Point positions, shape (batch, 2, ...)
            line_seg: Tuple of (start, end) line segment tensors

        Returns:
            Distance tensor, shape (batch, 1, n_segments)
        """
        pt = pt.reshape(pt.shape[0], pt.shape[1], 1)
        v = line_seg[1] - line_seg[0]
        length = v.norm(dim=1, keepdim=True)

        rel_pt = pt - line_seg[0]
        proj = (v * rel_pt).sum(dim=1, keepdim=True) / length
        proj = torch.clamp(proj, 0.0, 1.0)

        closest_pt = line_seg[0] + proj * v
        dist = (closest_pt - pt).norm(dim=1, keepdim=True)

        return dist

    def box_obstacle_costs(self, curr_state):
        """
        Compute obstacle avoidance costs based on distances to box obstacles.

        Evaluates the minimum distance from robot endpoints to obstacle boundaries,
        accounting for robot sphere radius. Cost increases as inverse square root
        of distance to encourage obstacle avoidance.

        Args:
            curr_state: Current state tensor, shape (batch, state_dim)

        Returns:
            Obstacle costs, shape (batch,)
        """
        curr_state = self.map(curr_state)

        if len(self.box_obstacles) == 0:
            return torch.zeros_like(curr_state[:, :1]).flatten()

        end_pts = self.compute_end_pts(curr_state)[:, :2]
        n_end_pts = end_pts.shape[-1]
        end_pts = end_pts.transpose(1, 2).reshape(-1, 2)

        # sphere_r = 0.175
        sphere_r = self.sim.robot.sphere_radius

        # costs = torch.zeros_like(curr_state[:, :1])

        lines_0 = torch.tensor(
            [
                p
                for xmin, xmax, ymin, ymax in self.box_obstacles
                for p in [[xmin, ymin], [xmin, ymin], [xmax, ymax], [xmax, ymax]]
            ], dtype=self.dtype, device=self.device
        ).T.unsqueeze(0)

        lines_1 = torch.tensor(
            [
                p
                for xmin, xmax, ymin, ymax in self.box_obstacles
                for p in [[xmin, ymax], [xmax, ymin], [xmin, ymax], [xmax, ymin]]
            ], dtype=self.dtype, device=self.device
        ).T.unsqueeze(0)

        dists = self._dist2d_pt_to_line_seg(end_pts, (lines_0, lines_1))
        dists = dists.min(dim=2).values.reshape(-1, n_end_pts).min(dim=1).values

        dists = torch.clamp_min(dists - sphere_r - self.obs_min_dist, 1e-8)
        costs = self.obs_cost_gain / (dists ** 0.5)

        return costs.flatten()

    def _dist_costs(self, curr_state):
        """
        Compute distance and directional costs (alternative formulation).

        Args:
            curr_state: Current state tensor

        Returns:
            Tuple of (dist_costs, dir_costs), each shape (batch,)
        """
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))

        goal_dir = self.goals[:, :2] - com[:, :2]
        dist_costs = goal_dir.norm(dim=1, keepdim=True)
        goal_dir = goal_dir / dist_costs

        dist_costs = dist_costs ** 2

        end_pts = self.compute_end_pts(curr_state)
        mid_left = end_pts[..., ::2].mean(dim=-1, keepdim=True)
        mid_right = end_pts[..., 1::2].mean(dim=-1, keepdim=True)
        prin = mid_right - mid_left
        # prin = prin.mean(dim=1).unsqueeze(-1)
        prin[:, 2] = 0.0
        prin = prin / prin.norm(dim=1, keepdim=True)

        # goal_dir = torch.tensor([-1., 0.], dtype=prin.dtype, device=prin.device).reshape(1, 2, 1)
        dir_costs = torch.linalg.vecdot(goal_dir, prin[:, :2], dim=1).abs()

        return dist_costs.flatten(), dir_costs.flatten()

    def multi_goal_dists_cost(self, curr_state):
        """
        Compute distance costs for multiple goals (alternative formulation).

        Args:
            curr_state: Current state tensor

        Returns:
            Tuple of (total_dist, dir_costs), each shape (batch,)
        """
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))

        goal_dir = com[:, :2] - self.goals[:, :2]
        dists = goal_dir.norm(dim=1, keepdim=True)

        final_dist = dists[..., -1:]

        total_dist = final_dist

        goal_dir = goal_dir[..., -1:] / final_dist
        prin = torch_quaternion.quat_as_rot_mat(
            curr_state.reshape(-1, 13, 1)[:, 3:7]
        )[..., -1:].reshape(curr_state.shape[0], -1, 3)
        prin = prin.mean(dim=1).unsqueeze(-1)
        prin[:, 2] = 0.0
        prin = prin / prin.norm(dim=1, keepdim=True)

        z = torch.zeros_like(prin)
        z[:, 2] = 1.

        y = torch.cross(z, prin, dim=1)
        y = y / y.norm(dim=1, keepdim=True)

        dir_costs = -torch.linalg.vecdot(goal_dir, y[:, :2], dim=1)

        return total_dist.squeeze(-1), dir_costs

    def dir_cost(self, curr_state, curr_dir):
        """
        Compute directional alignment cost.

        Penalizes misalignment between robot's principal axis and desired direction.

        Args:
            curr_state: Current state tensor
            curr_dir: Desired direction vector, shape (batch, 2, 1), or None

        Returns:
            Directional cost tensor, shape (batch,)
        """
        if curr_dir is None:
            return torch.zeros_like(curr_state[:, :1]).flatten()

        prin = torch_quaternion.quat_as_rot_mat(
            curr_state.reshape(-1, 13, 1)[:, 3:7]
        )[..., -1:].reshape(curr_state.shape[0], -1, 3)
        prin = prin.mean(dim=1).unsqueeze(-1)
        prin[:, 2] = 0.0
        prin = prin / prin.norm(dim=1, keepdim=True)

        dir_cost = torch.linalg.vecdot(curr_dir, prin[:, :2], dim=1).abs()
        dir_cost = (dir_cost + 1.) ** 2

        return dir_cost.flatten()

    def terminal_cost(self, curr_state):
        """
        Compute terminal reward for reaching the goal.

        Provides negative cost (reward) when robot center-of-mass is within
        goal tolerance threshold.

        Args:
            curr_state: Current state tensor

        Returns:
            Terminal cost tensor, shape (batch,)
        """
        xy_com = curr_state.reshape(curr_state.shape[0], -1, 13)[..., :2].mean(dim=1)
        goal = self.goal[:, :2]
        dist = (goal - xy_com).norm(dim=1)
        close = dist < self.goal_threshold
        terminal = torch.ones_like(dist) * self.terminal_reward * close

        return terminal.flatten()

    def all_costs(self, curr_state, curr_dir=None):
        """
        Compute all cost components for a given state.

        Args:
            curr_state: Current state tensor
            curr_dir: Desired direction vector, or None

        Returns:
            Tuple of (dist_cost, dir_cost, obs_cost) tensors, each shape (batch,)
        """
        dist_cost, obs_cost = self.heuristic_grid_cost(curr_state)
        # _dist_cost, dir_cost = self._dist_costs(curr_state)
        # _obs_cost = self.box_obstacle_costs(curr_state)
        dir_cost = self.dir_cost(curr_state, curr_dir)
        terminal_cost = self.terminal_cost(curr_state)
        dist_cost = dist_cost + terminal_cost

        return dist_cost, dir_cost, obs_cost

    def reset_sim_pose(self, curr_pose_time, prev_pose_time, rest_lengths=None, motor_speeds=None, batch_size=1):
        """
        Reset simulator with pose observations and estimated velocities.

        Args:
            curr_pose_time: Tuple of (current_pose, timestamp)
            prev_pose_time: Tuple of (previous_pose, timestamp) for velocity estimation
            rest_lengths: Cable rest lengths
            motor_speeds: Motor angular velocities
            batch_size: Batch size for parallel simulation
        """
        rest_lengths = self.map(rest_lengths).unsqueeze(0)
        motor_speeds = self.map(motor_speeds).unsqueeze(0)

        curr_pose, curr_timestamp = curr_pose_time
        curr_pose = self.map(curr_pose).reshape(-1, 7, 1)
        curr_pos, curr_quat = curr_pose[:, :3], curr_pose[:, 3:7]

        if prev_pose_time and (curr_timestamp - prev_pose_time[1]) > 1e-6:
            prev_pose, prev_timestamp = prev_pose_time
            prev_pose = self.map(prev_pose).reshape(-1, 7, 1)
            prev_pos, prev_quat = prev_pose[:, :3], prev_pose[:, 3:7]

            dt = curr_timestamp - prev_timestamp

            lin_vel = (curr_pos - prev_pos) / dt
            ang_vel = torch_quaternion.compute_ang_vel_quat(prev_quat, curr_quat, dt)
        else:
            lin_vel = torch.zeros_like(curr_pose[:, :3])
            ang_vel = torch.zeros_like(curr_pose[:, :3])

        vels = torch.hstack([lin_vel, ang_vel])

        curr_state = torch.hstack([curr_pose, vels]).reshape(1, -1, 1)
        self.reset_sim_state(curr_state, motor_speeds, rest_lengths, batch_size)

    def reset_sim_state(self, curr_state, motor_speeds, rest_lengths, batch_size=1):
        """
        Reset simulator with full state information.

        Args:
            curr_state: Current state tensor, shape (1, state_dim, 1)
            motor_speeds: Motor angular velocities
            rest_lengths: Cable rest lengths
            batch_size: Batch size for parallel simulation
        """
        self.sim.reset()

        curr_state = self.map(curr_state).reshape(1, -1, 1)
        if rest_lengths is not None:
            rest_lengths = self.map(rest_lengths).reshape(1, -1, 1)
        if motor_speeds is not None:
            motor_speeds = self.map(motor_speeds).reshape(1, -1, 1)

        curr_state = curr_state.repeat(batch_size, 1, 1)
        self.sim.update_state(curr_state)
        self.reset_cables(rest_lengths, motor_speeds, batch_size)

    def reset_cables(self, rest_lengths, motor_speeds, batch_size):
        """
        Reset cable states for all actuated cables.

        Args:
            rest_lengths: Cable rest lengths, shape (1, n_cables, 1)
            motor_speeds: Motor angular velocities, shape (1, n_cables, 1)
            batch_size: Batch size for parallel simulation
        """
        cables = self.sim.robot.actuated_cables.values()
        for i, c in enumerate(cables):
            if rest_lengths is not None:
                c.set_rest_length(rest_lengths[:, i: i + 1].repeat(batch_size, 1, 1))
            if motor_speeds is not None:
                c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1].repeat(batch_size, 1, 1)

    def plan(self, prev_n_pose_time_tups, rest_lens, motor_speeds):
        """
        Generate optimal control sequence using MPPI.

        Args:
            prev_n_pose_time_tups: List of (pose, timestamp) tuples for velocity estimation
            rest_lens: Current cable rest lengths
            motor_speeds: Current motor angular velocities

        Returns:
            Tuple of (min_actions, min_act_states, batch_states) where:
                - min_actions: Optimal action sequence
                - min_act_states: States resulting from optimal actions
                - batch_states: All sampled trajectory states
        """
        curr_pose, curr_timestamp = prev_n_pose_time_tups[-1]
        prev_pose_timestep = None
        for prev_pose_timestep in prev_n_pose_time_tups[:-1][::-1]:
            prev_pose, prev_timestamp = prev_pose_timestep
            if (curr_timestamp - prev_timestamp) >= self.min_vel_dt:
                break

        self.reset_sim_pose(prev_n_pose_time_tups[-1], prev_pose_timestep, rest_lens, motor_speeds)

        curr_state = self.sim.get_curr_state()

        mode = 'simple'
        if mode == 'simple':
            mppi_func, nsamples = self.mppi_simple, self.n_samples
        elif mode == 'perturb':
            mppi_func, nsamples = self.mppi_perturb, self.n_samples

        min_actions, min_act_states, batch_states = mppi_func(
            curr_state,
            rest_lens,
            motor_speeds,
            nsamples
        )
        # self.U = min_actions[..., ::self.ctrl_interval].clone().to(self.device)

        return min_actions, min_act_states, batch_states

    def compute_ctrl_lims(self, rest_lens, motor_speeds):
        """
        Compute simple control limits based on rest length bounds.

        Args:
            rest_lens: Current cable rest lengths
            motor_speeds: Current motor angular velocities

        Returns:
            Tuple of (lower, upper) control limits, each shape (n_cables,)
        """
        rest_lens = self.map(rest_lens)
        upper = (rest_lens >= self.rest_min).to(self.dtype).flatten()
        lower = -(rest_lens <= self.rest_max).to(self.dtype).flatten()
        return lower, upper


    def compute_ctrl_lims2(self, rest_lens, motor_speeds):
        """
        Compute control limits considering motor dynamics and constraints.

        Accounts for motor speed limits, winch radius, and current motor state
        to compute feasible control bounds that respect physical constraints.

        Args:
            rest_lens: Current cable rest lengths
            motor_speeds: Current motor angular velocities

        Returns:
            Tuple of (lower, upper) control limits, each shape (n_cables,)
        """
        rest_lens = rest_lens.unsqueeze(-1)
        motor_speeds = motor_speeds.unsqueeze(-1)

        cables = self.sim.robot.actuated_cables.values()
        s = torch.vstack([c.motor.speed for c in cables])
        m = torch.vstack([c.motor.max_omega for c in cables])
        # s, m = 0.8, 220 * 2 * np.pi / 60
        sm_inv = 1. / (s * m)
        # r_w = 0.035
        r_w = torch.vstack([c.winch_r for c in cables])

        alpha = 2 * sm_inv / (self.dt * r_w)

        lower = alpha * (rest_lens - self.rest_max) - motor_speeds * sm_inv
        upper = alpha * (rest_lens - self.rest_min) - motor_speeds * sm_inv

        lower = torch.clamp(lower, self.ctrl_min, self.ctrl_max - 1e-2)
        upper = torch.clamp(upper, self.ctrl_min + 1e-2, self.ctrl_max)

        # lower = torch.clamp(lower, self.u_min, 0.0).flatten()
        # upper = torch.clamp(upper, 0.0, self.u_max).flatten()

        # if (lower > -1).any() or (upper < 1.).any():
        #     s=0

        return lower.flatten(), upper.flatten()

    def mppi_simple(self, curr_state, curr_rest_lens, curr_motor_speeds, nsamples):
        """
        MPPI with uniform random action sampling.

        Samples action sequences uniformly from feasible control bounds without
        using previous control sequence as a prior.

        Args:
            curr_state: Current state tensor
            curr_rest_lens: Current cable rest lengths
            curr_motor_speeds: Current motor angular velocities
            nsamples: Number of trajectory samples

        Returns:
            Tuple of (min_actions, min_act_states, batch_states)
        """
        lower, upper = self.compute_ctrl_lims(curr_rest_lens, curr_motor_speeds)
        # lower = -ones(6, ref_tensor=curr_state)
        # upper = ones(6, ref_tensor=curr_state)
        #
        dist = Uniform(lower, upper)
        batch_actions = dist.sample(
            (nsamples, self.horizon // self.ctrl_interval)
        ).to(self.device).to(self.dtype)
        batch_actions = (batch_actions.repeat(1, 1, self.ctrl_interval)
                         .reshape(-1, self.horizon, self.n_ctrls)
                         .transpose(1, 2))

        return self.mppi(batch_actions, curr_motor_speeds, curr_rest_lens, curr_state, nsamples)

    def mppi_perturb(self, curr_state, curr_rest_lens, curr_motor_speeds, nsamples):
        """
        MPPI with perturbation-based action sampling.

        Samples action sequences by perturbing the previous control sequence,
        which provides temporal consistency and warm-starting for faster convergence.

        Args:
            curr_state: Current state tensor
            curr_rest_lens: Current cable rest lengths
            curr_motor_speeds: Current motor angular velocities
            nsamples: Number of trajectory samples

        Returns:
            Tuple of (min_actions, min_act_states, batch_states)
        """
        lower, upper = self.compute_ctrl_lims(curr_rest_lens, curr_motor_speeds)
        batch_act_perturb = 0.1 * torch.randn((nsamples, self.n_ctrls, self.horizon // self.ctrl_interval),
                                              dtype=self.dtype, device=self.device)
        batch_actions = torch.roll(self.prev_ctrls, -1, -1) + batch_act_perturb
        batch_actions[..., -1:] = (Uniform(lower, upper).sample((nsamples, 1))
                                   .transpose(1, 2)
                                   .to(self.device)
                                   .to(self.dtype))
        # batch_actions = torch.clamp(batch_actions, -1., 1.)
        for i in range(batch_actions.shape[1]):
            batch_actions[:, i] = torch.clamp(batch_actions[:, i], lower[i], upper[i])
        batch_actions = (batch_actions.repeat(1, self.ctrl_interval, 1)
                         .transpose(1, 2)
                         .reshape(-1, self.horizon, self.n_ctrls)
                         .transpose(1, 2))

        min_actions, min_act_states, batch_states = (
            self.mppi(batch_actions,
                      curr_motor_speeds,
                      curr_rest_lens,
                      curr_state,
                      nsamples))

        return min_actions, min_act_states, batch_states

    def get_curr_dir(self, curr_pose):
        """
        Get current direction vector from robot pose.

        Computes the perpendicular direction to the robot's principal axis.

        Args:
            curr_pose: Current pose tensor, shape (batch, 7)

        Returns:
            Direction vector, shape (1, 2, 1)
        """
        curr_pose = self.map(curr_pose).reshape(-1, 7, 1)
        prin = torch_quaternion.compute_prin_axis(curr_pose[:, 3:7]).mean(dim=0, keepdim=True)[:, :2]
        prin /= prin.norm(dim=1, keepdim=True)
        curr_dir = torch.hstack([-prin[:, 1:], prin[:, :1]])
        return curr_dir

    def rel_2d_angle(self, curr_dir, goal_dir):
        """
        Compute relative 2D angle between current and goal directions.

        Args:
            curr_dir: Current direction vector
            goal_dir: Goal direction vector

        Returns:
            Angle in radians, shape (1, 1)
        """
        curr_dir = self.map(curr_dir).reshape(1, 2, 1)
        goal_dir = self.map(goal_dir).reshape(1, 2, 1)

        cross = goal_dir[:, 0] * curr_dir[:, 1] - goal_dir[:, 1] * curr_dir[:, 0]  # 2D cross product (scalar)
        dot = goal_dir[:, 0] * curr_dir[:, 0] + goal_dir[:, 1] * curr_dir[:, 1]
        angle = torch.atan2(cross, dot)

        return angle

    def get_goal_dir(self, curr_pose):
        """
        Compute goal direction from current position using gradient descent on cost grid.

        Uses heuristic cost grid to determine the best local direction toward the goal
        while avoiding obstacles.

        Args:
            curr_pose: Current pose tensor, shape (batch, 7, 1)

        Returns:
            Goal direction vector, shape (1, 2, 1)
        """
        curr_pose = self.map(curr_pose)
        start_com = curr_pose.reshape(-1, 7, 1)[:, :2].mean(dim=0, keepdim=True)
        snapped_com = snap_to_grid_torch(start_com, self.grid_step, self.boundary)
        cost_grid = self.cost_weights[0] * self.dist_cost_grid + self.cost_weights[2] * self.obs_cost_grid
        best_pt = unsnap_to_grid_torch(
            heuristic_dir2(cost_grid, snapped_com, 2),
            self.grid_step,
            self.boundary
        )
        curr_dir = torch.hstack([best_pt[0] - start_com[:, :1], best_pt[1] - start_com[:, 1:2]])
        curr_dir = curr_dir / curr_dir.norm(dim=1, keepdim=True)
        return curr_dir

    def mppi(self, batch_actions, curr_motor_speeds, curr_rest_lens, curr_state, nsamples):
        """
        Core MPPI algorithm for trajectory optimization.

        Evaluates sampled action sequences through forward simulation, computes
        their costs, and either selects the minimum cost trajectory or computes
        a weighted combination based on the strategy.

        Args:
            batch_actions: Batch of action sequences, shape (nsamples, n_ctrls, horizon)
            curr_motor_speeds: Current motor angular velocities
            curr_rest_lens: Current cable rest lengths
            curr_state: Current state tensor
            nsamples: Number of trajectory samples

        Returns:
            Tuple of (min_actions, min_act_states, batch_states) where:
                - min_actions: Selected or weighted optimal actions
                - min_act_states: States from optimal trajectory
                - batch_states: All sampled trajectory states
        """
        curr_state_ = curr_state.clone().repeat(
            (nsamples if curr_state.shape[0] == 1 else 1), 1, 1)

        batch_states, all_costs = self.rollout(curr_state_, batch_actions)
        costs, other_costs = all_costs[0], all_costs[1:]

        if self.strategy == 'weighted':
            costs = torch.stack(costs, dim=-1).cpu()
            costs = (self.gamma_seq * costs.reshape(costs.shape[0], 1, -1)).sum(dim=-1)
            beta = torch.min(costs)
            cost_total_non_zero = self._compute_weight(costs, beta, 0.)
            eta = torch.sum(cost_total_non_zero)
            omega = ((1. / eta) * cost_total_non_zero).reshape(-1, 1, 1)
            # print(omega.device, batch_actions.device)
            min_actions = (omega * batch_actions.cpu()).sum(dim=0, keepdim=True)

            device = self.device
            self.to('cpu')
            self.reset_sim_state(curr_state_[:1].cpu(),
                                 curr_motor_speeds.cpu(),
                                 curr_rest_lens.cpu())
            min_act_states, all_min_costs = (
                self.rollout(curr_state[:1].cpu(), min_actions.cpu()))
            min_costs, min_other_costs = all_min_costs[0], all_min_costs[1:]

            other_cost = [(self.gamma_seq * torch.vstack(c)).sum(dim=0).mean().cpu().item()
                          for c in min_other_costs]
            cost = (self.gamma_seq * torch.vstack(min_costs)).sum(dim=0).mean().cpu().item()
            # cost = self.weights[0] * dist_c + self.weights[1] * dir_c + self.weights[2] * obs_c

            self.to(device)
        else:
            batch_cost = torch.stack(costs, dim=-1).sum(dim=-1)
            idx = batch_cost.argmin(dim=0)

            min_actions = batch_actions[idx: idx + 1]
            min_act_states = batch_states[idx: idx + 1]
            other_cost = [c[self.act_interval - 1][idx].cpu().item()
                          for c in other_costs]
            cost = costs[self.act_interval - 1][idx].cpu().item()

        self.logger.info(f"Total: {cost}, "
                         f"Other: {other_cost}")

        return min_actions, min_act_states, batch_states
