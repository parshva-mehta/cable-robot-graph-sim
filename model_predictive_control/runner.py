import json
import logging
import math
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image
import tqdm

from model_predictive_control.mujoco_env import MJCTensegrityEnv
from model_predictive_control.tensegrity_mppi import TensegrityGnnMPPI, TensegrityMjcMPPI
from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer


class MPPIRunner:

    def __init__(self, cfg):
        self.logger = self._get_logger(cfg)

        self.cfg = cfg
        self.output = cfg['output']
        self.planner_type = cfg['planner_type'] if 'planner_type' in cfg else 'gnn'

        self.sim_dt = 0.01
        self.env_dt = 0.01

        self.sensor_interval = cfg['sensor_interval']
        self.env_act_interval = int(cfg['act_interval'] / self.env_dt)
        self.sim_act_interval = int(cfg['act_interval'] / self.sim_dt)
        self.max_steps = cfg['max_time'] / self.env_dt
        self.threshold = cfg['threshold']
        self.visualize = cfg['visualize']
        self.save_data = cfg['save_data']

        nsamples_cold, nsamples_warm = cfg['nsamples'], cfg['nsamples'] // 2

        self.xml = Path(cfg['xml_path'])
        self.env_kwargs = cfg['env_kwargs']
        self.env = MJCTensegrityEnv(self.xml, env_type=cfg['env_type'], **self.env_kwargs)

        if 'start' in cfg:
            self.shift_env_robot_to_start(cfg)

        # Initial env robot stabilization for 5s
        for _ in range(500):
            self.env.step(np.zeros((1, self.env.env.n_actuators)))

        self._init_planner(cfg, nsamples_cold, nsamples_warm)

        self.vis, self.frames_path = None, None
        if cfg['visualize']:
            self._init_visualizer(cfg)

    def _init_visualizer(self, cfg):
        self.vis = MuJoCoVisualizer()
        self.vis.set_xml_path(Path(cfg['vis_xml_path']))
        self.vis.mjc_model.site_pos[0] = cfg['goal']
        self.vis.set_camera("top")
        self.frames_path = Path(self.output, "frames/")
        self.frames_path.mkdir(exist_ok=True)
        Path(self.output, "vids").mkdir(exist_ok=True)

    def _init_planner(self, cfg, nsamples_cold, nsamples_warm):
        common_planner_kwargs = dict(
            nsamples_cold=nsamples_cold,
            nsamples_warm=nsamples_warm,
            horizon=cfg['horizon'],
            act_interval=cfg['act_interval'],
            ctrl_interval=cfg['ctrl_interval'],
            sensor_interval=self.sensor_interval,
            goal=cfg['goal'],
            obstacles=cfg['obstacles'],
            boundaries=cfg['boundaries'],
            logger=self.logger
        )

        if self.planner_type == 'gnn':
            sim = torch.load(Path(cfg['model_path']), map_location='cpu', weights_only=False)
            sim.to(cfg['device'])
            if 'torch_compile' in cfg and cfg['torch_compile']:
                sim.run_compile()

            for c in sim.robot.actuated_cables.values():
                c.min_length = 0.4
                c.max_length = 2.5

            start_pose = torch.tensor(self.env.env.mjc_data.qpos.reshape(-1, 7), dtype=sim.dtype)
            start_vels = torch.zeros_like(start_pose[:, :6])
            start_state = torch.hstack([start_pose, start_vels]).reshape(1, -1, 1)

            self.planner = TensegrityGnnMPPI(
                sim=sim,
                curr_state=start_state,
                dt=self.sim_dt,
                nu=self.env.env.n_actuators,
                **common_planner_kwargs
            )
            self.planner.to(cfg['device'])
            init_endpts, _, _ = self.env.sense()
            self.planner.set_pose_by_endpts(init_endpts)
        elif self.planner_type == 'mujoco':
            self.planner = TensegrityMjcMPPI(
                xml=self.xml,
                env_kwargs=self.env_kwargs,
                tmp_dir=Path(self.output, "temp/"),
                **common_planner_kwargs
            )
        else:
            raise Exception(f"Planner type {self.planner_type} invalid")
        self.planner.set_goals(np.array(cfg['goal'], dtype=np.float64))

    def _get_logger(self, cfg):
        logger = logging.Logger("logger")
        logger.setLevel(logging.DEBUG)  # Set the minimum logging level

        # Create handlers
        console_handler = logging.StreamHandler()  # Log to stdout
        console_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(Path(cfg['output'], "log.txt"))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def shift_env_robot_to_start(self, cfg):
        start = np.array(cfg['start']).reshape(1, 3)
        endpts = self.env.env.get_endpts()
        pos = self.env.env.mjc_data.qpos.reshape(-1, 7).copy()
        com = pos[:, :3].mean(axis=0, keepdims=True)
        pos[:, 0] -= com[:, 0] - start[:, 0]
        pos[:, 1] -= com[:, 1] - start[:, 1]
        pos[:, 2] = pos[:, 2] - endpts[:, 2].min() + 0.18
        self.env.env.mjc_data.qpos = pos.flatten()

    def rererun(self):
        all_processed_data = json.load(Path(self.output, 'processed_data.json').open('r'))
        all_extra_data = json.load(Path(self.output, 'extra_state_data.json').open('r'))

        processed_data1, extra_data1 = self.rerun(all_processed_data[0]['pos'],
                                                  all_processed_data[0]['quat'],
                                                  all_processed_data[0]['linvel'],
                                                  all_processed_data[0]['angvel'],
                                                  all_extra_data[0]['rest_lengths'],
                                                  all_extra_data[0]['motor_speeds'],
                                                  [[dd for dd in d['controls']] for d in all_extra_data])

        print(max([max([abs(a - b) for a, b in zip(p0['pos'], p1['pos'])]) for p0, p1 in
                   zip(all_processed_data, processed_data1)]))
        print(max([max([abs(a - b) for a, b in zip(p0['quat'], p1['quat'])]) for p0, p1 in
                   zip(all_processed_data, processed_data1)]))
        print(max([max([max([abs(a - b) for a, b in zip(p0['end_pts'][k], p1['end_pts'][k])])
                        for k in range(12)])
                   for p0, p1 in zip(all_processed_data, processed_data1)]))

    def step_env(self, step, actions, sim_poses):
        dt_ratio = int(self.sim_dt / self.env_dt)
        # actions = (actions.transpose((0, 2, 1))
        #            .repeat(dt_ratio, axis=2)
        #            .reshape(actions.shape[0], -1, actions.shape[1])
        #            .transpose((0, 2, 1)))

        reached_goal = False
        poses = []
        processed_data, extra_data = [], []
        # sim_poses[0]['pos'] = np.concatenate(
        #     [self.env.env.mjc_data.qpos.flatten(), sim_poses[0]['pos']])
        sim_poses[0]['pos'] = self.env.env.mjc_data.qpos.flatten()

        self.env.forward()
        for i in range(self.env_act_interval):
            if i == self.env_act_interval - int(self.sensor_interval / self.env_dt):
                prev_end_pts, _, _ = self.env.sense()
                self.planner.set_pose_by_endpts(prev_end_pts)

            e_data = {
                "time": step * self.env_dt,
                "dt": self.env_dt,
                "rest_lengths": self.env.env.mjc_model.tendon_lengthspring[:self.planner.n_ctrls, 0].flatten().tolist(),
                "motor_speeds": [c.motor_state.omega_t.flatten().item() for c in self.env.env.cable_motors],
                "controls": actions[0, :, i].flatten().tolist()
            }
            step += 1

            self.env.step(actions[:1, :, i].copy())

            p_data = {
                "time": step * self.env_dt,
                "end_pts": self.env.env.get_endpts().tolist(),
                "sites": [self.env.env.mjc_data.sensor(f"pos_{s}").data.flatten().tolist()
                          for sp in self.env.env.cable_sites for s in sp],
                "pos": self.env.env.mjc_data.qpos.reshape(-1, 7)[:, :3].flatten().tolist(),
                "quat": self.env.env.mjc_data.qpos.reshape(-1, 7)[:, 3:7].flatten().tolist(),
                "linvel": self.env.env.mjc_data.qvel.reshape(-1, 6)[:, :3].flatten().tolist(),
                "angvel": self.env.env.mjc_data.qvel.reshape(-1, 6)[:, 3:].flatten().tolist()
            }

            if self.save_data:
                processed_data.append(p_data)
                extra_data.append(e_data)

            if step % dt_ratio == 0:
                poses.append({"time": self.sim_dt * step,
                              "pos": self.env.env.mjc_data.qpos.copy().tolist(),
                              })
                idx = int((i + 1) / dt_ratio)
                sim_poses[idx]['pos'] = np.concatenate([
                    self.env.env.mjc_data.qpos.flatten()#, sim_poses[idx]['pos']
                ])

            mjc_state = np.hstack([
                self.env.env.mjc_data.qpos.reshape(-1, 7),
                self.env.env.mjc_data.qvel.reshape(-1, 6)
            ]).reshape(1, -1, 1)
            # cost, dir_cost = self.planner.multi_goal_dists_cost(mjc_state)
            # multi_goal_cost, _ = self.planner.multi_goal_costs(mjc_state)
            dist_to_goal, dir_cost = self.planner._dist_costs(mjc_state)
            dist_to_goal = dist_to_goal ** 0.5
            box_cost = self.planner.box_obstacle_costs(mjc_state)
            if dist_to_goal < self.threshold:
                reached_goal = True

        self.logger.info(f"{step} {dist_to_goal.cpu().item()} {dir_cost.cpu().item()} {box_cost.cpu().item()}")

        return poses, step, reached_goal, processed_data, extra_data

    def plan(self, step):
        if self.planner_type == 'gnn':
            end_pts, rest_lens, motor_speeds = self.env.sense()
            self.planner.reset_sim_endpts(end_pts.reshape(1, -1, 1), rest_lens, motor_speeds)
        else:
            state, rest_lens, motor_speeds = self.env.sense_full()
            self.planner.reset_sim_state(state, motor_speeds, rest_lens)

        mode = 'simple' if step == 0 or step % (1 * self.planner.horizon) == 0 else 'perturb'

        actions, states, batch_states = self.planner.compute_actions(
            self.planner.sim.robot.get_curr_state(),
            rest_lens,
            motor_speeds,
            mode
        )

        actions = actions.cpu().clone().numpy()

        return actions, states, batch_states

    def run_goal(self):
        all_extra_data, all_processed_data, poses = self._init_data()
        step = len(poses)
        reached_goal = False
        frames = []

        while not reached_goal and step < self.max_steps:
            actions, states, batch_states = self.plan(step)
            num_rods = self.planner.sim.num_bodies

            batch_pos = sum([batch_states[:, i * 13: (i + 1) * 13] for i in range(num_rods)]) / num_rods
            sim_poses = [
                {
                    'time': step * self.sim_dt,
                    'pos': states[:1, :, i].reshape(-1, 13)[:, :7].flatten().cpu().clone().numpy().tolist()
                }
                for i in range(self.sim_act_interval + 1)
            ]
            prev_step = step
            curr_poses, step, reached_goal, processed_data, extra_data = self.step_env(
                step,
                actions,
                sim_poses
            )

            if self.save_data:
                all_processed_data.extend(processed_data)
                all_extra_data.extend(extra_data)

            if self.visualize and self.vis and self.frames_path:
                batch_pos = batch_pos.transpose(1, 2).cpu().tolist()
                random.shuffle(batch_pos)
                self._vis_env_data(batch_pos, frames, num_rods, prev_step, sim_poses, states)

            poses.extend(curr_poses)
            if self.visualize and (step % int(10 / self.env_dt) == 0 or reached_goal):
                self.vis.save_video(Path(self.output, f"vids/{step}_vid.mp4"), frames)
                frames = []

            if self.save_data and (step % int(10 / self.env_dt) == 0 or reached_goal):
                self._save_train_data(all_extra_data, all_processed_data)

            if step % int(2 / self.env_dt) == 0 or reached_goal:
                self._save_pose_data(poses)

        if self.save_data:
            self._reproduce_traj_data(all_extra_data, all_processed_data)

        return poses, step

    def _reproduce_traj_data(self, all_extra_data, all_processed_data):
        processed_data1, extra_data1 = self.rerun(all_processed_data[0]['pos'],
                                                  all_processed_data[0]['quat'],
                                                  all_processed_data[0]['linvel'],
                                                  all_processed_data[0]['angvel'],
                                                  all_extra_data[0]['rest_lengths'],
                                                  all_extra_data[0]['motor_speeds'],
                                                  [[dd for dd in d['controls']] for d in all_extra_data])
        processed_data2, extra_data2 = self.rerun(processed_data1[0]['pos'],
                                                  processed_data1[0]['quat'],
                                                  processed_data1[0]['linvel'],
                                                  processed_data1[0]['angvel'],
                                                  extra_data1[0]['rest_lengths'],
                                                  extra_data1[0]['motor_speeds'],
                                                  [[dd for dd in d['controls']] for d in extra_data1])
        print(max([max([abs(a - b) for a, b in zip(p0['pos'], p1['pos'])]) for p0, p1 in
                   zip(all_processed_data, processed_data1)]))
        print(max([max([abs(a - b) for a, b in zip(p0['quat'], p1['quat'])]) for p0, p1 in
                   zip(all_processed_data, processed_data1)]))
        e1 = max([max([abs(a - b) for a, b in zip(p0['pos'], p1['pos'])]) for p0, p1 in
                  zip(processed_data2, processed_data1)])
        e2 = max([max([abs(a - b) for a, b in zip(p0['quat'], p1['quat'])]) for p0, p1 in
                  zip(processed_data2, processed_data1)])
        error = max(e1, e2)
        print(f'Reproducible error: {error}')
        if error < 1e-10:
            self._save_train_data(extra_data2, processed_data2)
        else:
            raise ValueError('error too high')

    def _vis_env_data(self, batch_pos, frames, num_rods, prev_step, sim_poses, states):
        for i, pose in enumerate(sim_poses[1:]):
            if i % 4 != 0:
                continue

            self.vis.mjc_data.qpos = pose['pos']
            mujoco.mj_forward(self.vis.mjc_model, self.vis.mjc_data)
            self.vis.renderer.update_scene(self.vis.mjc_data, "camera")
            self.add_goal()

            for path in batch_pos[:200]:
                self.vis.add_path_to_scene(path[::20], radius=0.02)

            path = sum([states[:1, i: i + 3] for i in range(0, 13 * num_rods, 13)]) / num_rods
            path = path.transpose(1, 2).cpu().tolist()[0]
            self.vis.add_path_to_scene(path[::10], radius=0.05, rgba=np.array([0., 1., 0., 1.]))

            # path = sum([states[1:, i: i + 3] for i in range(0, 13 * num_rods, 13)]) / num_rods
            # path = path.transpose(1, 2).cpu().tolist()[0]
            # self.vis.add_path_to_scene(path[::10], radius=0.1, rgba=np.array([1., 0., 0., 1.]))

            frame = self.vis.renderer.render().copy()
            frames.append(frame)
            Image.fromarray(frame).save(self.frames_path / f"{prev_step + i + 1}.png")

    def add_goal(self):
        goal_arr = np.array([self.cfg['goal'], self.cfg['goal']])
        goal_arr[1] += 0.05
        self.vis.add_path_to_scene(goal_arr,
                                   radius=0.4,
                                   rgba=np.array([0.0, 0.0, 1.0, 1.0]))

    def _init_data(self):
        if 'restart' in self.cfg and self.cfg['restart']:
            poses = json.load(Path(self.cfg['output'], 'poses.json').open('r'))
            all_processed_data = json.load(Path(self.cfg['output'], 'processed_data.json').open('r'))
            all_extra_data = json.load(Path(self.cfg['output'], 'extra_state_data.json').open('r'))

            pos, quat = all_processed_data[-1]['pos'], all_processed_data[-1]['quat']
            linvel, angvel = all_processed_data[-1]['linvel'], all_processed_data[-1]['angvel']
            self.env.env.mjc_data.qpos = np.hstack([
                np.array(pos, dtype=np.float64).reshape(-1, 3),
                np.array(quat, dtype=np.float64).reshape(-1, 4)
            ]).flatten()
            self.env.env.mjc_data.qvel = np.hstack([
                np.array(linvel, dtype=np.float64).reshape(-1, 3),
                np.array(angvel, dtype=np.float64).reshape(-1, 3)
            ]).flatten()
            self.env.forward()
        else:
            self.env.forward()
            poses, all_extra_data = [], []
            all_processed_data = [{
                "time": 0.0,
                "end_pts": self.env.env.get_endpts().tolist(),
                "sites": {s: self.env.env.mjc_data.sensor(f"pos_{s}").data.flatten().tolist()
                          for sp in self.env.env.cable_sites for s in sp},
                "pos": self.env.env.mjc_data.qpos.reshape(-1, 7)[:, :3].flatten().tolist(),
                "quat": self.env.env.mjc_data.qpos.reshape(-1, 7)[:, 3:7].flatten().tolist(),
                "linvel": self.env.env.mjc_data.qvel.reshape(-1, 6)[:, :3].flatten().tolist(),
                "angvel": self.env.env.mjc_data.qvel.reshape(-1, 6)[:, 3:].flatten().tolist()
            }]

        start_state = self.env.env.get_curr_state()
        start_rest_lens = self.env.env.get_rest_lengths()
        start_motor_speeds = self.env.env.get_motor_speeds()

        self.planner.reset_sim_state(start_state, start_motor_speeds, start_rest_lens)

        return all_extra_data, all_processed_data, poses

    def _save_pose_data(self, poses):
        with Path(self.output, "poses.json").open('w') as fp:
            json.dump(poses, fp)

    def _save_train_data(self, all_extra_data, all_processed_data):
        with Path(self.output, "processed_data.json").open('w') as fp:
            json.dump(all_processed_data[:-1], fp)
        with Path(self.output, "extra_state_data.json").open('w') as fp:
            json.dump(all_extra_data, fp)

    def rerun(self, init_pos, init_quat, init_linvel, init_angvel, init_rest, init_mspeeds, controls):
        env_copy = MJCTensegrityEnv(self.xml, env_type=self.cfg['env_type'], **self.env_kwargs)

        init_pos_arr = np.array(init_pos, dtype=np.float64).reshape(-1, 3)
        init_quat_arr = np.array(init_quat, dtype=np.float64).reshape(-1, 4)
        init_linvel_arr = np.array(init_linvel, dtype=np.float64).reshape(-1, 3)
        init_angvel_arr = np.array(init_angvel, dtype=np.float64).reshape(-1, 3)

        env_copy.env.mjc_data.qpos = np.hstack([init_pos_arr, init_quat_arr]).flatten()
        env_copy.env.mjc_data.qvel = np.hstack([init_linvel_arr, init_angvel_arr]).flatten()

        # init_rest = gt_extra_data[w]['rest_lengths']
        # init_mspeeds = gt_extra_data[w]['motor_speeds']

        for j, cable in enumerate(env_copy.env.cable_motors):
            cable.motor_state.omega_t[:] = deepcopy(init_mspeeds[j])
        #
        env_copy.env.mjc_model.tendon_lengthspring[:env_copy.env.n_actuators, 0] = deepcopy(init_rest)
        env_copy.env.mjc_model.tendon_lengthspring[:env_copy.env.n_actuators, 1] = deepcopy(init_rest)

        extra_data = []
        processed_data = [{
            "time": 0.0,
            "end_pts": env_copy.env.get_endpts().tolist(),
            "sites": [env_copy.env.mjc_data.sensor(f"pos_{s}").data.flatten().tolist()
                      for sp in env_copy.env.cable_sites for s in sp],
            "pos": init_pos,
            "quat": init_quat,
            "linvel": init_linvel,
            "angvel": init_angvel
        }]

        for i, c in enumerate(tqdm.tqdm(controls)):
            e_data = {
                "time": i * self.env_dt,
                "dt": self.env_dt,
                "rest_lengths": env_copy.env.mjc_model.tendon_lengthspring[:env_copy.env.n_actuators,
                                0].flatten().tolist(),
                "motor_speeds": [c.motor_state.omega_t.flatten().item() for c in env_copy.env.cable_motors],
                "controls": deepcopy(c)
            }

            env_copy.env.sim_step(np.array(c).reshape(1, -1))

            p_data = {
                "time": (i + 1) * self.env_dt,
                "end_pts": env_copy.env.get_endpts().tolist(),
                "sites": [env_copy.env.mjc_data.sensor(f"pos_{s}").data.flatten().tolist()
                          for sp in env_copy.env.cable_sites for s in sp],
                "pos": env_copy.env.mjc_data.qpos.reshape(-1, 7)[:, :3].flatten().tolist(),
                "quat": env_copy.env.mjc_data.qpos.reshape(-1, 7)[:, 3:7].flatten().tolist(),
                "linvel": env_copy.env.mjc_data.qvel.reshape(-1, 6)[:, :3].flatten().tolist(),
                "angvel": env_copy.env.mjc_data.qvel.reshape(-1, 6)[:, 3:].flatten().tolist()
            }

            processed_data.append(p_data)
            extra_data.append(e_data)

            # tmp_poses.append(env_copy.mjc_data.qpos.reshape(1, -1).copy())
            # tmp_vels.append(env_copy.mjc_data.qvel.reshape(1, -1).copy())

        # tmp_poses = np.vstack(tmp_poses)
        # tmp_vels = np.vstack(tmp_vels)
        #
        # gt_pose = np.vstack([
        #     np.hstack([
        #         np.array(d['pos'], np.float64).reshape(-1, 3),
        #         np.array(d['quat'], np.float64).reshape(-1, 4)
        #     ]).reshape(1, -1) for d in gt_processed_data[1:]
        # ])
        #
        # gt_vels = np.vstack([
        #     np.hstack([
        #         np.array(d['linvel'], np.float64).reshape(-1, 3),
        #         np.array(d['angvel'], np.float64).reshape(-1, 3)
        #     ]).reshape(1, -1) for d in gt_processed_data[1:]
        # ])
        #
        # error = np.abs(gt_pose - tmp_poses).max()
        #
        # logger.info(f'Rerun pose error: {np.abs(gt_pose - tmp_poses).max()}')
        # logger.info(f'Rerun vels error: {np.abs(gt_vels - tmp_vels).max()}')

        return processed_data, extra_data


def combine_videos(video_dir, output_path):
    """
    Combines multiple MP4 videos into a single MP4 video using OpenCV.

    :param video_paths: List of paths to the input MP4 videos.
    :param output_path: Path to save the combined MP4 video.
    """
    try:
        video_paths = [p.as_posix() for p in Path(video_dir).glob("*vid.mp4")]
        video_paths = sorted(video_paths, key=lambda p: int(p.split("/")[-1].split("_")[0]))
        # List to store video capture objects
        video_captures = [cv2.VideoCapture(path) for path in video_paths]

        # Get properties of the first video to determine output properties
        frame_width = int(video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_captures[0].get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (frame_width, frame_height))

        for cap in video_captures:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

        # Release all resources
        for cap in video_captures:
            cap.release()
        out.release()

        print(f"Successfully combined videos into {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_mppi(cfg):
    Path(cfg['output']).mkdir(exist_ok=True)

    gx, gy = 0, 0
    min_dist = 10
    while (gx ** 2 + gy ** 2) ** 0.5 < min_dist:
        gx = min_dist * (2 * random.random() - 1)
        gy = min_dist * (2 * random.random() - 1)
    cfg['goal'][0] = gx
    cfg['goal'][1] = gy

    mppi = MPPIRunner(cfg)

    mppi.run_goal()
    combine_videos(Path(cfg['output'], 'vids/'), Path(cfg['output'], 'vids/combined.mp4'))

    shutil.rmtree(Path(cfg['output'], 'frames/'))
    # os.rmdir(Path(cfg['output'], 'temp/'))
    del mppi
