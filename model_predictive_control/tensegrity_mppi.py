import multiprocessing

import torch.nn
import tqdm
from torch.distributions import Uniform

from model_predictive_control.mppi_utils import *
from model_predictive_control.mujoco_env import MJCSim
from mujoco_physics_engine.tensegrity_mjc_simulation import *
from state_objects.primitive_shapes import Cylinder
from utilities import torch_quaternion
from utilities.tensor_utils import zeros


class TensegrityGnnMPPI(torch.nn.Module):

    def __init__(self,
                 sim,
                 curr_state,
                 nsamples_cold,
                 nsamples_warm,
                 horizon,
                 act_interval,
                 sensor_interval,
                 ctrl_interval,
                 dt,
                 device='cpu',
                 u_min=-1.,
                 u_max=1.,
                 nu=6,
                 gamma=1.001,
                 rest_min=0.9,
                 rest_max=2.0,
                 goal=None,
                 obstacles=[],
                 boundaries=[],
                 logger=None,
                 strategy='min'):
        super().__init__()
        self.logger = logger

        self.sim = sim
        self.curr_pose = curr_state.reshape(-1, 13, 1)[:, :7]
        self.nsamples_cold = nsamples_cold
        self.nsamples_warm = nsamples_warm
        self.horizon = int(horizon / dt)
        self.act_interval = int(act_interval / dt)
        self.sensor_interval = int(sensor_interval / dt)
        self.ctrl_interval = int(ctrl_interval / dt)
        self.dt = dt
        self.dtype = self.sim.dtype
        self.device = device
        self.goals, self.goal_vecs, self.goal_vecs_norm = None, None, None

        self.obs_cost_gain = 50.0
        self.obs_min_dist = 0.5
        self.terminal_reward = -10.0
        self.goal_threshold = 0.5

        self.strategy = strategy

        self.gamma = gamma
        self.gamma_seq = torch.cumprod(
            torch.full((1, 1, self.horizon),
                       gamma,
                       dtype=self.dtype,
                       device=self.device),
            dim=-1
        )

        self.ctrl_min = u_min
        self.ctrl_max = u_max
        self.rest_min = rest_min
        self.rest_max = rest_max
        self.n_ctrls = nu

        self.prev_ctrls = zeros((1, nu, self.horizon // self.ctrl_interval), ref_tensor=self.curr_pose)

        self.weights = [1.0, 0.0, 0.0]
        self.cost_slope = 5.
        self.off_traj_cost = 5.

        self.set_goals([np.array(goal)])
        self.boundaries = boundaries
        self.box_obstacles = obstacles

        self.grid_step = 0.2
        self.dist_cost_grid, self.obs_cost_grid = fill_grid(
            goal,
            boundaries,
            self.grid_step,
            obstacles
        )
        self.dist_cost_grid, self.obs_cost_grid = (
            self.dist_cost_grid.to(self.dtype), self.obs_cost_grid.to(self.dtype))

    def to(self, device):
        self.device = device
        self.dist_cost_grid = self.dist_cost_grid.to(device)
        self.obs_cost_grid = self.obs_cost_grid.to(device)
        # self.gamma_seq = self.gamma_seq.to(device)
        self.sim = self.sim.to(device)
        self.curr_pose = self.curr_pose.to(device)
        self.prev_ctrls = self.prev_ctrls.to(device)
        if self.goals is not None:
            self.goals = self.goals.to(device)
            self.goal_vecs = self.goal_vecs.to(device)
            self.goal_vecs_norm = self.goal_vecs_norm.to(device)

        return self

    @staticmethod
    def _compute_weight(cost, beta, factor):
        return torch.exp(-factor * (cost - beta))

    def map(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        data = data.to(self.sim.device).to(self.sim.dtype)
        return data

    @staticmethod
    def endpts_to_5d_pose(end_pts):
        end_pts_ = end_pts.reshape(-1, 6, 1)
        curr_pos = (end_pts_[:, :3] + end_pts_[:, 3:]) / 2.
        prin = (end_pts_[:, 3:] - end_pts_[:, :3])
        curr_quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin)

        return torch.hstack([curr_pos, curr_quat])

    def set_goals(self, com_goal):
        if not isinstance(com_goal, list):
            com_goal = [com_goal]
        self.goals = torch.concat([
            self.map(c).reshape(-1, 3, 1)
            for c in com_goal
        ], dim=-1)
        self.goal_vecs = self.goals[:, :2, 1:] - self.goals[:, :2, :-1]
        self.goal_vecs_norm = self.goal_vecs.norm(dim=1, keepdim=True)

        self.goals[:, 2, -1] = 0.
        for i in range(self.goals.shape[2] - 2, -1, -1):
            d = self.goal_vecs_norm[..., i]
            self.goals[:, 2:, i] = self.goals[:, 2:, i + 1] + self.cost_slope * d
            # self.goal_vecs[..., i] /= d

    def heuristic_grid_cost(self, curr_state, grid_step=0.2):
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))
        idx = snap_to_grid_torch(com[:, :2], grid_step, self.boundaries).squeeze(-1)
        # idx[:, 0] = torch.round((idx[:, 0] - self.boundaries[0]) / grid_step)
        # idx[:, 1] = torch.round(-(idx[:, 1] - self.boundaries[3]) / grid_step)
        dist_cost, obs_cost = self.dist_cost_grid[idx[:, 0], idx[:, 1]], self.obs_cost_grid[idx[:, 0], idx[:, 1]]

        return dist_cost, obs_cost

    def multi_goal_costs(self, curr_state):
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
        return self.sim.robot.compute_end_pts(curr_state)

    def box_obstacle_costs2(self, curr_state):
        """
        end_pts: (batch_size, 3, num_end_pts)
        """
        curr_state = self.map(curr_state)
        end_pts = self.compute_end_pts(curr_state)[:, :3]
        sphere_r = 0.175
        # sphere_r = self.sim.robot.sphere_radius

        mid_left = end_pts[..., ::2].mean(dim=1, keepdim=True)
        mid_right = end_pts[..., 1::2].mean(dim=1, keepdim=True)
        prin = mid_right - mid_left
        prin[:, 2] = 0.
        prin = prin / prin.norm(dim=1, keepdim=True)

        z = torch.zeros_like(prin)
        z[:, 2] = 1.
        y = torch.cross(z, prin, dim=1)
        y = y / y.norm(dim=1, keepdim=True)

        costs = torch.zeros_like(curr_state[:, :1])
        for xmin, ymin, xmax, ymax in self.box_obstacles:
            lines = [
                [[xmin, ymin], [xmin, ymax]],
                [[xmin, ymin], [xmax, ymin]],
                [[xmax, ymax], [xmin, ymax]],
                [[xmax, ymax], [xmax, ymin]],
            ]

            dists = []
            for a, b in lines:
                v = torch.tensor([b[0] - a[0], b[1] - a[1]],
                                 dtype=curr_state.dtype,
                                 device=curr_state.device
                                 ).reshape(1, 2, 1)
                v = v / v.norm(dim=1, keepdim=True)
                u = end_pts.clone()
                u[:, 0] = u[:, 0] - a[0]
                u[:, 1] = u[:, 1] - a[1]

                w = torch.linalg.vecdot(u, v, dim=1).unsqueeze(1)
                w = w / v.norm(dim=1, keepdim=True)
                w = torch.clamp(w, 0.0, 1.0)

                pt = end_pts.clone()
                pt[:, :1] = a[0] + w * v[:, 0]
                pt[:, 1:] = a[1] + w * v[:, 1]

                d = end_pts - pt
                d[:, 0] = d[:, 0] * 10.
                d = d.norm(dim=1, keepdim=True)
                dists.append(d)

            dist = torch.concat(dists, dim=2).min(dim=2, keepdim=True).values
            #
            #
            # dx = torch.min(torch.hstack([
            #     (end_pts[:, :1] - xmin).abs(), (end_pts[:, :1] - xmax).abs()
            # ]), dim=1, keepdim=True).values
            # dy = torch.min(torch.hstack([
            #     (end_pts[:, 1:2] - ymin).abs(), (end_pts[:, 1:2] - ymax).abs()
            # ]), dim=1, keepdim=True).values
            # dist = torch.hstack([dx, dy]).norm(dim=1, keepdim=True).min(dim=2, keepdim=True).values

            dist = torch.clamp_min(dist - sphere_r - self.obs_min_dist, 1e-8)
            cost = self.obs_cost_gain / dist
            cost = torch.clamp_min(cost, 0.0)

            costs += cost

        return costs.flatten()

    def box_obstacle_costs(self, curr_state):
        """
        end_pts: (batch_size, 3, num_end_pts)
        """
        curr_state = self.map(curr_state)
        end_pts = self.compute_end_pts(curr_state)[:, :2]
        sphere_r = 0.175
        # sphere_r = self.sim.robot.sphere_radius

        costs = torch.zeros_like(curr_state[:, :1])
        for xmin, ymin, xmax, ymax in self.box_obstacles:
            lines = [
                [[xmin, ymin], [xmin, ymax]],
                [[xmin, ymin], [xmax, ymin]],
                [[xmax, ymax], [xmin, ymax]],
                [[xmax, ymax], [xmax, ymin]],
            ]

            dists = []
            for a, b in lines:
                v = torch.tensor([b[0] - a[0], b[1] - a[1]],
                                 dtype=curr_state.dtype,
                                 device=curr_state.device
                                 ).reshape(1, 2, 1)
                v = v / v.norm(dim=1, keepdim=True)
                u = end_pts.clone()
                u[:, 0] = u[:, 0] - a[0]
                u[:, 1] = u[:, 1] - a[1]

                w = torch.linalg.vecdot(u, v, dim=1).unsqueeze(1)
                w = w / v.norm(dim=1, keepdim=True)
                w = torch.clamp(w, 0.0, 1.0)

                pt = end_pts.clone()
                pt[:, :1] = a[0] + w * v[:, 0]
                pt[:, 1:] = a[1] + w * v[:, 1]

                d = end_pts - pt
                # d[:, 0] = d[:, 0] * 10.
                d = d.norm(dim=1, keepdim=True)
                dists.append(d)

            dist = torch.concat(dists, dim=2).min(dim=2, keepdim=True).values
            #
            #
            # dx = torch.min(torch.hstack([
            #     (end_pts[:, :1] - xmin).abs(), (end_pts[:, :1] - xmax).abs()
            # ]), dim=1, keepdim=True).values
            # dy = torch.min(torch.hstack([
            #     (end_pts[:, 1:2] - ymin).abs(), (end_pts[:, 1:2] - ymax).abs()
            # ]), dim=1, keepdim=True).values
            # dist = torch.hstack([dx, dy]).norm(dim=1, keepdim=True).min(dim=2, keepdim=True).values

            dist = torch.clamp_min(dist - sphere_r - self.obs_min_dist, 1e-8)
            cost = self.obs_cost_gain / (dist ** 0.5)
            cost = torch.clamp_min(cost, 0.0)

            costs += cost

        return costs.flatten()

    def _dist_costs(self, curr_state):
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))

        goal_dir = self.goals[:, :2] - com[:, :2]
        dist_costs = goal_dir.norm(dim=1, keepdim=True)
        goal_dir = goal_dir / dist_costs

        dist_costs = dist_costs ** 2

        # prin = torch_quaternion.quat_as_rot_mat(
        #     curr_state.reshape(-1, 13, 1)[:, 3:7]
        # )[..., -1:].reshape(curr_state.shape[0], -1, 3)
        # prin = prin.mean(dim=1).unsqueeze(-1)
        # prin[:, 2] = 0.0
        # prin /= prin.norm(dim=1, keepdim=True)

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
        curr_state = self.map(curr_state)
        com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :3]
               .mean(dim=1).unsqueeze(-1))

        goal_dir = com[:, :2] - self.goals[:, :2]
        dists = goal_dir.norm(dim=1, keepdim=True)

        # curr_dist = dists.topk(3, dim=-1, largest=False).values.sum(dim=-1, keepdim=True)
        # curr_dist = dists.topk(3, dim=-1, largest=False).values.sum(dim=-1, keepdim=True)
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
        xy_com = (curr_state.reshape(curr_state.shape[0], -1, 13)[..., :2]
                  .mean(dim=1).unsqueeze(-1))
        goal = self.goals[:, :2]
        dist = (goal - xy_com).norm(dim=1)
        close = dist < self.goal_threshold
        terminal = torch.ones_like(dist) * self.terminal_reward * close

        return terminal.flatten()

    def all_costs(self, curr_state, curr_dir=None):
        dist_cost, obs_cost = self.heuristic_grid_cost(curr_state)
        # _dist_cost, dir_cost = self._dist_costs(curr_state)
        # _obs_cost = self.box_obstacle_costs(curr_state)
        dir_cost = self.dir_cost(curr_state, curr_dir)
        terminal_cost = self.terminal_cost(curr_state)
        dist_cost = dist_cost + terminal_cost

        return dist_cost, dir_cost, obs_cost

    def set_pose_by_endpts(self, curr_end_pts):
        curr_end_pts = self.map(curr_end_pts)
        self.curr_pose = self.endpts_to_5d_pose(curr_end_pts)

    def reset_sim_pose(self, curr_pose, rest_lengths, motor_speeds, batch_size=1):
        curr_pose = self.map(curr_pose).reshape(-1, 7, 1)
        rest_lengths = self.map(rest_lengths).unsqueeze(0)
        motor_speeds = self.map(motor_speeds).unsqueeze(0)

        prev_pos, prev_quat = self.curr_pose[:, :3], self.curr_pose[:, 3:7]

        # lin_vel = (curr_pose[:, :3] - prev_pos) / (self.dt)
        lin_vel = (curr_pose[:, :3] - prev_pos) / (self.dt * self.sensor_interval)
        ang_vel = torch_quaternion.compute_ang_vel_quat(prev_quat,
                                                        curr_pose[:, 3:7],
                                                        # self.dt
                                                        self.dt * self.sensor_interval
                                                        )
        vels = torch.hstack([lin_vel, ang_vel])

        self.curr_pose = curr_pose
        curr_state = torch.hstack([curr_pose, vels]).reshape(1, -1, 1)

        self.reset_sim_state(curr_state, motor_speeds, rest_lengths, batch_size)

    def reset_sim_state(self, curr_state, motor_speeds, rest_lengths, batch_size=1):
        self.sim.reset()

        curr_state = self.map(curr_state).reshape(1, -1, 1)
        rest_lengths = self.map(rest_lengths).reshape(1, -1, 1)
        motor_speeds = self.map(motor_speeds).reshape(1, -1, 1)

        curr_state = curr_state.repeat(batch_size, 1, 1)
        self.sim.update_state(curr_state)
        self.reset_cables(rest_lengths, motor_speeds, batch_size)

    def reset_cables(self, rest_lengths, motor_speeds, batch_size):
        cables = self.sim.robot.actuated_cables.values()
        for i, c in enumerate(cables):
            c.set_rest_length(rest_lengths[:, i: i + 1].repeat(batch_size, 1, 1))
            c.motor.motor_state.omega_t = motor_speeds[:, i: i + 1].repeat(batch_size, 1, 1)

    def reset_sim_endpts(self, curr_end_pts, rest_lengths, motor_speeds, batch_size=1):
        curr_end_pts = self.map(curr_end_pts)
        curr_pose = self.endpts_to_5d_pose(curr_end_pts)

        self.reset_sim_pose(curr_pose, rest_lengths, motor_speeds, batch_size)

    def compute_actions(self,
                        curr_state,
                        curr_rest_lens,
                        curr_motor_speeds,
                        mode):
        curr_state = self.map(curr_state)
        curr_rest_lens = self.map(curr_rest_lens)
        curr_motor_speeds = self.map(curr_motor_speeds)
        mode = 'simple'
        if mode == 'simple':
            mppi_func, nsamples = self.mppi_simple, self.nsamples_cold
        elif mode == 'perturb':
            mppi_func, nsamples = self.mppi_perturb, self.nsamples_warm
        else:
            mppi_func, nsamples = self.mppi_diff, 1

        min_actions, min_act_states, batch_states = mppi_func(
            curr_state,
            curr_rest_lens,
            curr_motor_speeds,
            nsamples
        )
        # self.U = min_actions[..., ::self.ctrl_interval].clone().to(self.device)

        return min_actions, min_act_states, batch_states

    def compute_ctrl_lims(self, rest_lens, motor_speeds):
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

    def mppi_diff(self, curr_state, curr_rest_lens, curr_motor_speeds, nsamples):
        device = self.device
        self.to('cpu')
        self.sim.eval()

        lower, upper = self.compute_ctrl_lims(curr_rest_lens, curr_motor_speeds)
        dist = Uniform(lower, upper)
        # new_action = (dist.sample((1, 1))
        #               .to(self.device)
        #               .to(self.dtype)
        #               .transpose(1, 2))
        # with torch.no_grad():
        # actions = self.U[..., 1:].cpu()
        # batch_actions = (actions.repeat(1, 1, self.ctrl_interval)
        #                  .reshape(-1, self.horizon - self.ctrl_interval, self.nu)
        #                  .transpose(1, 2))
        #
        curr_state_ = curr_state.cpu().clone()
        all_states = [curr_state_.clone()]
        # for i in tqdm.tqdm(range(batch_actions.shape[2])):
        #     curr_state_, _ = self.sim.step(
        #         curr_state_,
        #         self.dt,
        #         control_signals=batch_actions[..., i].clone()
        #     )
        #     all_states.append(curr_state_.clone())
        #
        # curr_state_ = curr_state_.detach()
        rest_lengths = torch.hstack([c.rest_length
                                     for c in self.sim.robot.actuated_cables.values()]
                                    ).flatten().detach()
        motor_speeds = torch.hstack([c.motor.motor_state.omega_t
                                     for c in self.sim.robot.actuated_cables.values()]
                                    ).flatten().detach()

        # with torch.no_grad():
        #     # actions = (dist.sample((1000, 1))
        #     #            .to(self.dtype)
        #     #            .transpose(1, 2))
        #     i = 0
        #     acts = torch.linspace(-1., 1., 501).reshape(-1, 1, 1).to(torch.float32)
        #     actions = torch.zeros((acts.shape[0], 6, 1), dtype=torch.float32)
        #     actions[:, i: i + 1] = acts
        #     batch_actions = (actions.repeat(1, 1, self.ctrl_interval)
        #                      .reshape(-1, self.ctrl_interval, self.nu)
        #                      .transpose(1, 2))
        #
        #     self.reset_sim_state(curr_state_.cpu().clone(),
        #                          motor_speeds.cpu().clone(),
        #                          rest_lengths.cpu().clone(),
        #                          batch_actions.shape[0])
        #     states, (costs, dist_costs, dir_costs, obs_costs) = \
        #         self.rollout(self.sim.get_curr_state().cpu().clone(), batch_actions)
        #
        #     np_act = actions.detach().numpy().squeeze()
        #     total_costs = torch.stack(costs, dim=-1).mean(dim=-1).flatten().detach().numpy()
        # total_costs = costs[-1].flatten().detach().numpy()

        # import matplotlib.pyplot as plt
        #
        # act = np_act[:, i].tolist()
        # aug_act = [(j, act[j]) for j in range(len(act))]
        # aug_act = sorted(aug_act, key=lambda x: x[1])
        # idx = [a[0] for a in aug_act]
        # act = [a[1] for a in aug_act]
        # sorted_total_costs = total_costs.tolist()
        # sorted_total_costs = [sorted_total_costs[i] for i in idx]
        # plt.plot(act, sorted_total_costs)
        # plt.title(f'{i}')
        # plt.tight_layout()  # Adjust subplot spacing
        # plt.show()

        with torch.enable_grad():
            actions = (dist.sample((1, self.horizon // self.ctrl_interval))
                       .to(self.dtype)
                       .transpose(1, 2))
            actions = torch.nn.Parameter(actions)
            optimizer = torch.optim.Adam([actions], lr=1e-2)

            for i in range(100):
                prev_actions = actions.data.detach().clone()
                batch_actions = (actions.repeat(1, 1, self.ctrl_interval)
                                 .reshape(-1, self.horizon, self.n_ctrls)
                                 .transpose(1, 2))

                states, (costs, dist_costs, dir_costs, obs_costs) = \
                    self.rollout(curr_state_.cpu().clone(), batch_actions)

                total_costs = torch.stack(costs, dim=-1).sum()
                total_costs._backward()
                torch.nn.utils.clip_grad_norm_(actions,
                                               max_norm=100)
                optimizer.step()
                print(total_costs, actions.grad.mean().detach().item())
                optimizer.zero_grad()

                self.reset_sim_state(curr_state_.cpu().clone(),
                                     motor_speeds.cpu().clone(),
                                     rest_lengths.cpu().clone())

            min_actions = batch_actions.detach()
            min_act_states = all_states + [states]
            batch_states = all_states + [states]

        self.logger.info(f"Total: {costs[self.act_interval - 1]}, "
                         f"Obstacle: {obs_costs[self.act_interval - 1]}, "
                         f"Dist: {dist_costs[self.act_interval - 1]}, "
                         f"Dir: {dir_costs[self.act_interval - 1]}")
        self.sim.eval()
        self.to(device)

        return min_actions, min_act_states, batch_states

    def mppi_simple(self, curr_state, curr_rest_lens, curr_motor_speeds, nsamples):
        lower, upper = self.compute_ctrl_lims(curr_rest_lens, curr_motor_speeds)
        # lower = -ones(6, ref_tensor=curr_state)
        # upper = ones(6, ref_tensor=curr_state)

        dist = Uniform(lower, upper)
        batch_actions = dist.sample(
            (nsamples, self.horizon // self.ctrl_interval)
        ).to(self.device).to(self.dtype)
        batch_actions = (batch_actions.repeat(1, 1, self.ctrl_interval)
                         .reshape(-1, self.horizon, self.n_ctrls)
                         .transpose(1, 2))

        return self.mppi(batch_actions, curr_motor_speeds, curr_rest_lens, curr_state, nsamples)

    def mppi_perturb(self, curr_state, curr_rest_lens, curr_motor_speeds, nsamples):
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

    def proc_rollout(self, curr_state, actions, sim, idx, queue):
        act_lens = [c.actuation_length.clone() for c in sim.robot.actuated_cables.values()]
        motor_speeds = [c.motor.motor_state.omega_t.clone() for c in sim.robot.actuated_cables.values()]

        all_states = []
        for i in tqdm.tqdm(range(actions.shape[0])):
            ctrls = actions[i: i + 1].clone()
            for j, c in enumerate(sim.robot.actuated_cables.values()):
                c.actuation_length = act_lens[j].clone()
                c.motor.motor_state.omega_t = motor_speeds[j].clone()

            state_to_graph_kwargs = {'dataset_idx': torch.tensor([[9]])}
            sim_states, _ = self.sim.run(
                curr_state.clone(),
                ctrls=ctrls,
                dt=self.dt,
                state_to_graph_kwargs=state_to_graph_kwargs  # HACK
            )

            sim_states = torch.cat(sim_states, dim=-1)
            all_states.append(sim_states)
        all_states = torch.vstack(all_states)
        torch.save(all_states, Path(self.tmp_dir, f'tmp{idx}.pt'))
        queue.put(True)

    def multi_proc_rollout(self, curr_state_, batch_actions, num_proc=24):
        sim_cpys = [deepcopy(self.sim) for _ in range(num_proc)]
        queues = [multiprocessing.Queue() for _ in range(num_proc)]
        processes = [
            multiprocessing.Process(
                target=self.proc_rollout,
                args=(curr_state_[:1].clone(),
                      batch_actions[i::num_proc].clone(),
                      sim_cpys[i],
                      i,
                      queues[i])
            ) for i in range(num_proc)
        ]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        status = [False for _ in range(num_proc)]
        while not all(status):
            for i in range(num_proc):
                status[i] = queues[i].get()

        all_states = torch.zeros_like(curr_state_).repeat(1, 1, batch_actions.shape[-1])
        for j in range(num_proc):
            all_states[j::num_proc] = torch.load(Path(self.tmp_dir, f"tmp{j}.pt"))
        all_states = [all_states[..., j: j + 1] for j in range(all_states.shape[-1])]

        return all_states

    def rollout(self, curr_state_, batch_actions):
        states = [curr_state_.clone()]
        if self.multi_proc_mode:
            sim_states = self.multi_proc_rollout(curr_state_, batch_actions)
        else:
            state_to_graph_kwargs = {'dataset_idx': torch.tensor([7]).repeat(curr_state_.shape[0]).reshape(-1, 1)}
            sim_states, _ = self.sim.run(
                curr_state_,
                ctrls=batch_actions,
                dt=self.dt,
                state_to_graph_kwargs=state_to_graph_kwargs  # HACK
            )

        costs, dist_costs, dir_costs, obs_costs = [], [], [], []
        curr_dir = self.get_dir(curr_state_)
        for state in sim_states:
            dist_cost, dir_cost, obs_cost = self.all_costs(state, curr_dir)
            cost = (self.weights[0] * dist_cost
                    + self.weights[1] * dir_cost
                    + self.weights[2] * obs_cost)

            costs.append(cost)
            dist_costs.append(self.weights[0] * dist_cost.detach())
            dir_costs.append(self.weights[1] * dir_cost.detach())
            obs_costs.append(self.weights[2] * obs_cost.detach())

        states.extend(sim_states)
        states = torch.concat(states, dim=-1)

        return states, (costs, dist_costs, dir_costs, obs_costs)

    def get_dir(self, curr_state_):
        start_com = curr_state_[:1].reshape(-1, 13, 1)[:, :2].mean(dim=0, keepdim=True)
        snapped_com = snap_to_grid_torch(start_com, self.grid_step, self.boundaries)
        cost_grid = self.weights[0] * self.dist_cost_grid + self.weights[2] * self.obs_cost_grid
        best_pt = unsnap_to_grid_torch(
            heuristic_dir(cost_grid, snapped_com),
            self.grid_step,
            self.boundaries
        )
        curr_dir = torch.hstack([best_pt[0] - start_com[:, :1], best_pt[1] - start_com[:, 1:2]])
        curr_dir = curr_dir / curr_dir.norm(dim=1, keepdim=True)
        return curr_dir

    def mppi(self, batch_actions, curr_motor_speeds, curr_rest_lens, curr_state, nsamples):
        curr_state_ = curr_state.clone().repeat(
            (nsamples if curr_state.shape[0] == 1 else 1), 1, 1)

        batch_states, all_costs = self.rollout(curr_state_, batch_actions)
        costs, other_costs = all_costs[0], all_costs[1:]

        if self.strategy == 'wghted_avg':
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


class TensegrityMjcMPPI(TensegrityGnnMPPI):

    def __init__(self,
                 xml,
                 env_kwargs,
                 nsamples_cold,
                 nsamples_warm,
                 horizon,
                 act_interval,
                 sensor_interval,
                 ctrl_interval,
                 tmp_dir,
                 goal,
                 obstacles,
                 boundaries,
                 logger=None):
        sim = MJCSim(xml, tmp_dir=tmp_dir, **env_kwargs)
        nu = sim.env.n_actuators
        curr_state = sim.get_curr_state()
        super().__init__(sim,
                         curr_state,
                         nsamples_cold,
                         nsamples_warm,
                         horizon,
                         act_interval,
                         sensor_interval,
                         ctrl_interval,
                         sim.dt,
                         goal=goal,
                         obstacles=obstacles,
                         boundaries=boundaries,
                         nu=nu,
                         logger=logger)

    def reset(self):
        pass

    def rollout(self, curr_state_, batch_actions):
        next_states = self.sim.rollout(curr_state_, batch_actions)

        dist_costs, dir_costs, obs_costs, costs = [], [], [], []
        for i in range(next_states.shape[2]):
            d0, d1, d2 = self.all_costs(next_states[..., i: i + 1])
            # d0, d1 = self.multi_goal_costs(next_states[..., i: i + 1])
            c = self.weights[0] * d0 + self.weights[1] * d1 + self.weights[2] * d2
            dist_costs.append(d0)
            dir_costs.append(d1)
            obs_costs.append(d2)
            costs.append(c)

        # dist_costs = [dist_costs[..., i: i + 1] for i in range(dist_costs.shape[2])]
        # dir_costs = [dir_costs[..., i: i + 1] for i in range(dir_costs.shape[2])]
        # costs = [costs[..., i: i + 1] for i in range(costs.shape[2])]

        return next_states, (costs, dist_costs, dir_costs, obs_costs)

    def compute_end_pts(self, curr_state):
        end_pts = []
        length = 2.95
        for i in range(3):
            state = curr_state[:, i * 13: i * 13 + 7]
            principal_axis = torch_quaternion.compute_prin_axis(state[:, 3:7])
            end_pts.extend(Cylinder.compute_end_pts_from_state(state, principal_axis, length))

        return torch.concat(end_pts, dim=2)

    def reset_cables(self, rest_lengths, motor_speeds, batch_size):
        rest_lengths = rest_lengths.clone().numpy()
        motor_speeds = motor_speeds.clone().numpy()

        self.sim.reset_cables(rest_lengths, motor_speeds)
