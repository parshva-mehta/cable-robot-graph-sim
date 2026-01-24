import json
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import tqdm

from mujoco_physics_engine.cable_motor import DCMotor
from mujoco_physics_engine.mujoco_simulation import AbstractMuJoCoSimulator
from mujoco_physics_engine.pid import PID

import mujoco

from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer
from utilities import torch_quaternion

CENTER_SITES = [
    ("s3", "s5"),
    ("s1", "s3"),
    ("s5", "s1"),
    ("s0", "s2"),
    ("s4", "s0"),
    ("s2", "s4"),
]

KUN_SITES = [
    ("s_3_5", "s_5_3"),
    ("s_1_3", "s_3_1"),
    ("s_1_5", "s_5_1"),
    ("s_0_2", "s_2_0"),
    ("s_0_4", "s_4_0"),
    ("s_2_4", "s_4_2"),
    ("s_2_5", "s_5_2"),
    ("s_0_3", "s_3_0"),
    ("s_1_4", "s_4_1")
]

REAL_ATTACH_SITES = [
    ("s_3_b5", "s_b5_3"),
    ("s_1_b3", "s_b3_1"),
    ("s_5_b1", "s_b1_5"),
    ("s_0_b2", "s_b2_0"),
    ("s_4_b0", "s_b0_4"),
    ("s_2_b4", "s_b4_2"),
    ("s_3_5", "s_5_3"),
    ("s_1_3", "s_3_1"),
    ("s_1_5", "s_5_1"),
    ("s_0_2", "s_2_0"),
    ("s_0_4", "s_4_0"),
    ("s_2_4", "s_4_2"),
    ("s_2_5", "s_5_2"),
    ("s_0_3", "s_3_0"),
    ("s_1_4", "s_4_1")
]

SIX_BAR_SURFACE_SITES = [
    ("s_0_10", "s_10_0"),
    ("s_1_4", "s_4_1"),
    ("s_2_6", "s_6_2"),
    ("s_1_3", "s_3_1"),
    ("s_4_8", "s_8_4"),
    ("s_2_5", "s_5_2"),
    ("s_5_6", "s_6_5"),
    ("s_7_11", "s_11_7"),
    ("s_7_8", "s_8_7"),
    ("s_0_9", "s_9_0"),
    ("s_9_10", "s_10_9"),
    ("s_3_11", "s_11_3"),
    ("s_0_8", "s_8_0"),
    ("s_0_4", "s_4_0"),
    ("s_1_10", "s_10_1"),
    ("s_3_10", "s_10_3"),
    ("s_9_11", "s_11_9"),
    ("s_7_9", "s_9_7"),
    ("s_2_4", "s_4_2"),
    ("s_1_2", "s_2_1"),
    ("s_3_6", "s_6_3"),
    ("s_6_11", "s_11_6"),
    ("s_5_7", "s_7_5"),
    ("s_5_8", "s_8_5"),
]

SIX_BAR_CENTER_SITES = [
    ("s0", "s10"),
    ("s1", "s4"),
    ("s2", "s6"),
    ("s1", "s3"),
    ("s4", "s8"),
    ("s2", "s5"),
    ("s5", "s6"),
    ("s7", "s11"),
    ("s7", "s8"),
    ("s0", "s9"),
    ("s9", "s10"),
    ("s3", "s11"),
    ("s0", "s8"),
    ("s0", "s4"),
    ("s1", "s10"),
    ("s3", "s10"),
    ("s9", "s11"),
    ("s7", "s9"),
    ("s2", "s4"),
    ("s1", "s2"),
    ("s3", "s6"),
    ("s6", "s11"),
    ("s5", "s7"),
    ("s5", "s8"),
]

CABLE_SITES = {
    "center": CENTER_SITES,
    "kun": KUN_SITES,
    "real_attach": REAL_ATTACH_SITES,
    "six_bar_surface": SIX_BAR_SURFACE_SITES,
    "six_bar_center": SIX_BAR_CENTER_SITES,
}


class TensegrityMuJoCoSimulator(AbstractMuJoCoSimulator):

    def __init__(self,
                 xml_path: Path,
                 visualize: bool = True,
                 render_size: (int, int) = (1280, 1280),
                 render_fps: int = 50,
                 min_len=0.4,
                 max_len=2.7,
                 attach_type='center',
                 num_rods=3,
                 n_actuators=6,
                 sphere_radius=0.175,
                 motor_speed=0.8,
                 winch_r=0.035):
        super().__init__(xml_path, visualize, render_size, render_fps)
        self.num_rods = num_rods
        self.n_actuators = n_actuators
        self.sphere_radius = sphere_radius

        self.min_cable_rest_length = min_len
        self.max_cable_rest_length = max_len
        self.actuator_tendon_ids = list(range(self.n_actuators))
        self.curr_ctrl = [0.0 for _ in range(self.n_actuators)]
        self.pids = [PID() for _ in range(self.n_actuators)]
        self.pid_freq = 0.01
        self.rod_names = {
            0: "r01",
            1: "r23",
            2: "r45"
        }
        self.cable_sites = CABLE_SITES[attach_type]

        self.cable_map = {
            i: i + (self.n_actuators if attach_type == 'real_attach' else 0)
            for i in range(n_actuators)
        }

        self.end_pts = [
            f's{i}' for i in range(2 * self.num_rods)
        ]
        self.stiffness = self.mjc_model.tendon_stiffness.copy()
        self.cable_motors = [DCMotor(np.array(motor_speed)) for _ in range(self.n_actuators)]
        self.winch_r = np.array(winch_r, dtype=np.float64)

    def get_se2(self):
        self.forward()

        end_pts = self.get_endpts()
        com = end_pts.mean(axis=0, keepdims=True)[:, :2]

        left, right = end_pts[::2].mean(axis=0, keepdims=True), end_pts[1::2].mean(axis=0, keepdims=True)
        prin = (right - left)[:, :2]
        prin /= np.linalg.norm(prin, axis=1, keepdims=True)
        angle = np.arctan2(prin[:, 1], prin[:, 0])  # wrt to x-axis

        return com, angle

    def get_curr_state(self):
        pose = self.mjc_data.qpos.reshape(-1, 7)
        vels = self.mjc_data.qvel.reshape(-1, 6)
        state = np.hstack([pose, vels]).reshape(1, -1, 1)
        return state

    def get_rest_lengths(self):
        return self.mjc_model.tendon_lengthspring[:, :1].flatten().copy()

    def get_motor_speeds(self):
        return np.concatenate([c.motor_state.omega_t for c in self.cable_motors]).flatten().copy()

    def bring_to_grnd(self):
        self.forward()
        qpos = self.mjc_data.qpos.copy().reshape(-1, 7)
        end_pts = self.get_endpts().reshape(-1, 3)
        min_z = end_pts[:, 2].min()
        qpos[:, 2] -= min_z - self.sphere_radius
        self.mjc_data.qpos = qpos.reshape(1, -1)

    def reset(self):
        super().reset()
        self.bring_to_grnd()

        for motor in self.cable_motors:
            motor.reset_omega_t()

    def sim_step(self, controls=None):
        # if controls is None:
        #    controls = self.curr_ctrl.copy()

        mujoco.mj_forward(self.mjc_model, self.mjc_data)
        for i, sites in enumerate(self.cable_sites):
            rest_length = self.mjc_model.tendon_lengthspring[i, :1]

            s0 = self.mjc_data.sensor(f"pos_{sites[0]}").data
            s1 = self.mjc_data.sensor(f"pos_{sites[1]}").data
            dist = np.linalg.norm(s1 - s0, keepdims=True)

            self.mjc_model.tendon_stiffness[i] = np.zeros_like(self.stiffness[i]) \
                if dist < rest_length else self.stiffness[i]

            if controls is not None and i < self.n_actuators:
                dl = self.cable_motors[i].compute_cable_length_delta(
                    controls[:, i], self.winch_r, self.dt)

                rest_length = rest_length - dl
                if (self.min_cable_rest_length > rest_length) or (rest_length > self.max_cable_rest_length):
                    print(f"ERROR: rest length {rest_length} for cable {i} is out of bounds")

                rest_length = np.clip(
                    rest_length,
                    self.min_cable_rest_length,
                    self.max_cable_rest_length
                )

                self.mjc_model.tendon_lengthspring[i] = rest_length

        mujoco.mj_step(self.mjc_model, self.mjc_data)
        mujoco.mj_forward(self.mjc_model, self.mjc_data)

    def run_w_ctrls(self, ctrls):
        frames = [{'time': 0.0, 'pos': self.mjc_data.qpos.copy()}]

        for i, ctrl in enumerate(ctrls):
            self.sim_step(ctrl)
            frames.append({'time': i * self.dt, 'pos': self.mjc_data.qpos.copy()})

        return frames

    def get_endpts(self):
        self.forward()
        end_pts = []
        for end_pt_site in self.end_pts:
            end_pt = self.mjc_data.sensor(f"pos_{end_pt_site}").data.copy()
            end_pts.append(end_pt)

        end_pts = np.vstack(end_pts)
        return end_pts

    def detect_ground_endcaps(self):
        end_pts = self.get_endpts()
        aug_end_pts = [[(i, end_pts[i]), (i + 1, end_pts[i + 1])]
                       for i in range(0, len(end_pts), 2)]
        aug_end_pts = [min(e, key=lambda x: x[1].flatten()[2].item()) for e in aug_end_pts]

        ground_endcaps = tuple([a[0] for a in aug_end_pts])

        return ground_endcaps

    def run(self,
            end_time: float = None,
            num_steps: int = None,
            save_path: Path = None,
            pos_sensor_names: Optional[List] = None,
            quat_sensor_names: Optional[List] = None,
            linvel_sensor_names: Optional[List] = None,
            angvel_sensor_names: Optional[List] = None):

        mujoco.mj_forward(self.mjc_model, self.mjc_data)

        end_pts = [self.get_endpts()]
        pos = [self.mjc_data.qpos.copy()]
        vel = [self.mjc_data.qvel.copy()]
        frames = [self.render_frame("front")]
        num_steps_per_frame = int(1 / self.render_fps / self.dt)
        for n in range(num_steps):
            if (n + 1) % 100 == 0:
                print((n + 1) * self.dt)

            self.sim_step()

            mujoco.mj_forward(self.mjc_model, self.mjc_data)
            end_pts.append(self.get_endpts())
            pos.append(self.mjc_data.qpos.copy())
            vel.append(self.mjc_data.qvel.copy())

            if self.visualize and ((n + 1) % num_steps_per_frame == 0 or n == num_steps - 1):
                frame = self.render_frame()
                frames.append(frame.copy())

        self.save_video(Path(save_path, "gt_vid.mp4"), frames)

        return end_pts, pos, vel


class ThreeBarTensegrityMuJoCoSimulator(TensegrityMuJoCoSimulator):

    def run_primitive(self, prim_type, left_range=None, right_range=None):
        prim_gaits = {
            'cw': [[1., 1., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 1., 1., 0., 0.8, 0.]],
            'roll': [[1., 1., 0.1, 1., 1., 0.1], [0., 1., 1., 0., 1., 0.1]],
            'ccw': [[1., 1., 1., 0., 1., 1.], [1., 0., 1., 0., 1., 1.], [0., 0., 0., 0., 0., 0.]],
            # 'crawl': [[0.1, 0.0, 0.1, 0.0, 0.0, 0.0], [1.0, 0.1, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.1, 0.0, 0.0, 0.0]]
        }
        rest_target_gait = [1., 1., 1., 1., 1., 1.]

        if "ccw" == prim_type:
            min_length = 100
            range_ = 100
            tol = 0.1
        elif "roll" == prim_type:
            min_length = 90
            range_ = 100
            tol = 0.1
        elif "cw" == prim_type:
            min_length = 80
            range_ = 120
            tol = 0.1
        else:
            min_length = 100
            range_ = 100
            tol = 0.1

        left_range = range_ if left_range is None else left_range
        right_range = range_ if right_range is None else right_range

        for i in range(self.n_actuators):
            range_ = left_range if i % 2 == 0 else right_range
            self.pids[i].min_length = min_length / 100
            self.pids[i].RANGE = range_ / 100
            self.pids[i].tol = tol

        target_gaits = [rest_target_gait] + prim_gaits[prim_type] + [rest_target_gait]

        data, extra_info = self.run_w_target_gaits(target_gaits)
        return data, extra_info


    def run_w_target_gaits(self, target_gaits, save_path=None, max_time_per_gait=10):
        symmetry_mapping = {
            (0, 2, 5): [0, 1, 2, 3, 4, 5], (0, 3, 5): [0, 1, 2, 3, 4, 5],
            (1, 2, 4): [1, 2, 0, 4, 5, 3], (1, 2, 5): [1, 2, 0, 4, 5, 3],
            (0, 3, 4): [2, 0, 1, 5, 3, 4], (1, 3, 4): [2, 0, 1, 5, 3, 4]
        }
        max_steps = max_time_per_gait // self.dt

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)

        self.forward()
        data = [{
            "time": 0.0,
            "end_pts": [
                self.mjc_data.sensor(f"pos_{s}").data.tolist()
                for s in self.end_pts
            ],
            "sites": {
                s: self.mjc_data.sensor(f"pos_{s}").data.tolist()
                for c in self.cable_sites for s in c
            },
            "pos": self.mjc_data.qpos.reshape(-1, 7)[:, :3].flatten().tolist(),
            "quat": self.mjc_data.qpos.reshape(-1, 7)[:, 3:].flatten().tolist(),
            "linvel": self.mjc_data.qvel.reshape(-1, 6)[:, :3].flatten().tolist(),
            "angvel": self.mjc_data.qvel.reshape(-1, 6)[:, 3:].flatten().tolist(),
            # "init_rest_lengths": self.mjc_model.tendon_lengthspring[:6, 0].tolist(),
            "pid": {
                "min_length": self.pids[0].min_length,
                "RANGE": self.pids[0].RANGE,
                "tol": self.pids[0].tol,
                "motor_speed": self.cable_motors[0].speed.item()
            }
        }]
        target_gaits_dicts = []
        extra_data = []
        key_frame_ids = []

        if self.visualize:
            frames = [self.render_frame()]

        num_steps_per_frame = int(1 / self.render_fps / self.dt)
        global_steps = 0
        num_steps = []
        for target_gait in tqdm.tqdm(target_gaits):
            for pid in self.pids:
                pid.reset()
            # print(k)
            step = 0
            controls = [1.]

            ground_endcap_idx = self.detect_ground_endcaps()
            if ground_endcap_idx in symmetry_mapping:
                order = symmetry_mapping[ground_endcap_idx]
            elif target_gait == [1., 1., 1., 1., 1., 1.]:
                order = [0, 1, 2, 3, 4, 5]
            else:
                raise Exception(f"Ground endcaps {ground_endcap_idx} not in symmetry mapping")

            target_gait = [target_gait[o] for o in order]

            target_gaits_dicts.append({
                'idx': global_steps,
                'target_gait': target_gait,
                'info': {
                    'min_length': int(self.pids[0].min_length * 100),
                    'RANGE': int(self.pids[0].RANGE * 100),
                    'tol': self.pids[0].tol,
                    'P': self.pids[0].k_p,
                    'I': self.pids[0].k_i,
                    'D': self.pids[0].k_d,
                    'max_speed': int(self.cable_motors[0].speed * 100),

                }
            })

            while any([c != 0 for c in controls]) and step < max_steps:
                # print(step)
                step += 1
                global_steps += 1

                if step == max_steps:
                    print('reached max steps')
                    break

                # print(global_steps)
                mujoco.mj_forward(self.mjc_model, self.mjc_data)

                if global_steps % (self.pid_freq // self.dt) == 0 or step == 1:
                    controls = []
                    curr_lens = []
                    for i in range(len(target_gait)):
                        pid = self.pids[i]
                        gait = target_gait[i]

                        rest_length = self.mjc_model.tendon_lengthspring[i, 0]
                        key = self.cable_map[i] if hasattr(self, "cable_map") and self.cable_map else i
                        s0 = self.mjc_data.sensor(f"pos_{self.cable_sites[key][0]}").data
                        s1 = self.mjc_data.sensor(f"pos_{self.cable_sites[key][1]}").data
                        curr_length = np.linalg.norm(s1 - s0)

                        ctrl, _ = pid.update_control_by_target_gait(curr_length, gait, rest_length)
                        controls.append(ctrl)
                        curr_lens.append(curr_length)

                # print([c.item() for c in controls])
                # print([c.item() for c in curr_lens])
                # print(self.mjc_model.tendon_lengthspring[:self.n_actuators, 0].flatten())

                extra_data.append({
                    "time": round(self.dt * (global_steps - 1), 4),
                    "rest_lengths": self.mjc_model.tendon_lengthspring[:self.n_actuators, 0].copy().tolist(),
                    "motor_speeds": [c.motor_state.omega_t[0].copy() for c in self.cable_motors],
                    "controls": [c.copy().item() for c in controls]
                })

                self.sim_step(np.array(controls).reshape(1, -1))
                self.forward()

                data.append({
                    "time": round(self.dt * global_steps, 4),
                    "end_pts": [
                        self.mjc_data.sensor(f"pos_{s}").data.tolist()
                        for s in self.end_pts
                    ],
                    "sites": {
                        s: self.mjc_data.sensor(f"pos_{s}").data.tolist()
                        for c in self.cable_sites for s in c
                    },
                    "pos": self.mjc_data.qpos.reshape(-1, 7)[:, :3].flatten().tolist(),
                    "quat": self.mjc_data.qpos.reshape(-1, 7)[:, 3:].flatten().tolist(),
                    "linvel": self.mjc_data.qvel.reshape(-1, 6)[:, :3].flatten().tolist(),
                    "angvel": self.mjc_data.qvel.reshape(-1, 6)[:, 3:].flatten().tolist(),
                    "rest_lengths": self.mjc_model.tendon_lengthspring[:self.n_actuators, 0].copy().tolist(),
                    "motor_speeds": [c.motor_state.omega_t[0].copy() for c in self.cable_motors],
                })

                if self.visualize and (global_steps % num_steps_per_frame == 0):
                    frame = self.render_frame()
                    frames.append(frame)
            num_steps.append(step)
            key_frame_ids.append(global_steps)

        return data[:-1], extra_data

    def align_prin(self, new_prin, new_com):
        import torch
        from utilities import torch_quaternion

        new_prin = torch.from_numpy(new_prin)
        new_com = torch.from_numpy(new_com)
        new_prin[:, 2] = 0

        self.forward()

        pose = self.mjc_data.qpos.reshape(-1, 7, 1)
        pos, quat = pose[:, :3], pose[:, 3:]
        end_pts = self.get_endpts()

        pos = torch.from_numpy(pos)
        quat = torch.from_numpy(quat)
        end_pts = torch.from_numpy(end_pts)

        mid_left = end_pts[::2].mean(dim=0, keepdim=True)
        mid_right = end_pts[1::2].mean(dim=0, keepdim=True)
        prins = mid_right - mid_left
        prins[:, 2] = 0.
        prins = prins / prins.norm(dim=1, keepdim=True)

        curr_com = torch.mean(pos, dim=0, keepdim=True)

        rot_dir = torch.cross(prins, new_prin, dim=1)
        rot_dir /= rot_dir.norm(dim=1)

        angle = torch.linalg.vecdot(prins, new_prin, dim=1).unsqueeze(1)
        angle = torch.clamp(angle, -1, 1)
        angle = torch.acos(angle) / 2

        rot_quat = torch.hstack([torch.cos(angle), torch.sin(angle) * rot_dir])

        new_pos = torch_quaternion.rotate_vec_quat(rot_quat, pos - curr_com) + new_com
        new_quat = torch_quaternion.quat_prod(rot_quat, quat)

        self.mjc_data.qpos = torch.hstack([new_pos, new_quat]).flatten().numpy()
        self.forward()

    def flip_to_next_support_tri(self):
        self.forward()
        pose = self.mjc_data.qpos.reshape(-1, 7, 1)
        pos, quat = pose[:, :3], pose[:, 3:]
        end_pts = self.get_endpts()

        pos = torch.from_numpy(pos).reshape(-1, 3, 1)
        quat = torch.from_numpy(quat).reshape(-1, 4, 1)
        end_pts = torch.from_numpy(end_pts).reshape(-1, 3, 1)

        mid_left = end_pts[::2].mean(dim=0, keepdim=True)
        mid_right = end_pts[1::2].mean(dim=0, keepdim=True)
        prins = mid_right - mid_left
        prins = prins / prins.norm(dim=1, keepdim=True)

        curr_com = torch.mean(pos, dim=0, keepdim=True)

        angle = torch.tensor(torch.pi / 3, dtype=prins.dtype).reshape(1, 1, 1)
        rot_quat = torch.hstack([torch.cos(angle), prins * torch.sin(angle)])

        new_pos = torch_quaternion.rotate_vec_quat(rot_quat, pos - curr_com) + curr_com
        new_quat = torch_quaternion.quat_prod(rot_quat, quat)
        self.mjc_data.qpos = torch.hstack([new_pos, new_quat]).flatten().numpy()

        self.bring_to_grnd()
        self.forward()


class SixBarTensegrityMuJoCoSimulator(TensegrityMuJoCoSimulator):

    def __init__(self, xml_path: Path, attach_type='six_bar_center', visualize=True):
        super().__init__(xml_path,
                         visualize=visualize,
                         attach_type=attach_type,
                         num_rods=6,
                         n_actuators=24)
        self.rod_names = {
            0: "r01",
            1: "r23",
            2: "r45",
            3: "r67",
            4: "r89",
            5: "r1011"
        }
