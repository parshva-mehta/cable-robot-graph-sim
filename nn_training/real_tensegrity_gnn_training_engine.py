import json
import logging
import random
from copy import deepcopy
from pathlib import Path

import numpy as np

from nn_training.datasets.real_tensegrity_dataset import RealMultiSimTensegrityDataset
from nn_training.tensegrity_gnn_training_engine import TensegrityMultiSimMultiStepMotorGNNTrainingEngine
from simulators.tensegrity_gnn_simulator import *
from utilities import torch_quaternion


class RealTensegrityMultiSimMultiStepMotorGNNTrainingEngine(TensegrityMultiSimMultiStepMotorGNNTrainingEngine):
    target_dt: float
    min_dt: float
    max_dt: float
    vel_min_dt: float
    mix_ratio: float
    eval_weight_real: float

    def __init__(self,
                 training_config: Dict,
                 logger: logging.Logger):
        self.target_dt = training_config['target_dt']
        self.min_dt = max(self.target_dt - training_config['dt_delta'] - 1e-6, 0.01 - 1e-6)
        self.max_dt = self.target_dt + training_config['dt_delta'] + 1e-6
        self.vel_min_dt = training_config['vel_min_dt'] if 'vel_min_dt' in training_config else 0.0
        self.mix_ratio = training_config['mix_ratio']
        self.eval_weight_real = training_config.get('eval_weight_real', 0.5)

        training_config['num_steps_fwd'] = round(self.target_dt / training_config['dt'])
        super().__init__(training_config, logger)

    def _init_data(self, data_paths):
        data_dict = {}
        data_dict['names'] = [Path(p).name for p in data_paths]
        real_idxs = [i for i, n in enumerate(data_dict['names']) if 'real' in n]

        data_jsons, extra_state_infos, vis_data = self._load_json_files(data_paths)
        self._check_and_mod_data_length(data_jsons, extra_state_infos)

        extra_state_infos_cpy = deepcopy(extra_state_infos)
        real_data_jsons = [data_jsons[i] for i in real_idxs]
        real_extra_state_infos = [extra_state_infos[i] for i in real_idxs]
        real_data_jsons, real_extra_state_infos = self._filter_data(
            real_data_jsons, real_extra_state_infos, data_dict['names'])

        for i, real_idx in enumerate(real_idxs):
            data_jsons[real_idx] = real_data_jsons[i]
            extra_state_infos[real_idx] = real_extra_state_infos[i]

        self.logger.info("Loaded data jsons")

        max_times = [e[-1]['time'] for e in extra_state_infos]
        data_controls, data_act_lens = (
            self._interp_ctrls_act_lens(extra_state_infos_cpy, max_times))

        data_dict.update({
            'data_jsons': data_jsons,
            'extra_state_infos': extra_state_infos,
            'controls': data_controls,
            'act_lengths': data_act_lens,
            'vis_data': vis_data,
        })

        data_dict['gt_end_pts'] = self._get_end_pts(data_jsons)

        data_dict['times'] = [[d['time'] - data_json[0]['time'] for d in data_json]
                              for data_json in data_jsons]

        data_dict['states'], _ = self._data_json_to_states(data_jsons,
                                                           data_dict['gt_end_pts'],
                                                           data_dict['times'],
                                                           extra_state_infos,
                                                           data_dict['names'])
        self.logger.info("Computed states and controls")

        data_dict['dataset_idx'] = self._enum_dataset(data_paths)

        return data_dict

    def _get_dataset(self, data_dict):
        real_idxs = [i for i, n in enumerate(data_dict['names']) if 'real' in n]
        sim_idxs = [i for i, n in enumerate(data_dict['names']) if 'real' not in n]

        real_data_dict = {k: [v[i] for i in real_idxs] for k, v in data_dict.items()}
        sim_data_dict = {k: [v[i] for i in sim_idxs] for k, v in data_dict.items()}

        return RealMultiSimTensegrityDataset(
            real_data_dict,
            sim_data_dict,
            self.min_dt,
            self.max_dt,
            self.target_dt,
            self.mix_ratio,
            self.num_steps_fwd,
            self.num_ctrls_hist,
            self.dt,
            self.dtype
        )

    def _load_json_files(self, paths):
        data_jsons, extra_state_infos, vis_data = [], [], []
        for path in paths:
            processed_data_path = Path(path, 'processed_data.json')
            if not processed_data_path.exists():
                processed_data_path = Path(path, 'interp_processed_data.json')
            with processed_data_path.open('r') as fp:
                data_jsons.append(json.load(fp))

            extra_info_path = Path(path, f"extra_state_data.json")
            if not extra_info_path.exists():
                extra_info_path = Path(path, 'interp_extra_state_data.json')

            if extra_info_path.exists():
                with extra_info_path.open('r') as fp:
                    extra_state_infos.append(json.load(fp))
            else:
                extra_state_infos.append(None)

            vis_path = Path(path, 'vis_data_0.01.json')
            if vis_path.exists():
                with vis_path.open('r') as fp:
                    vis_data.append(json.load(fp))
            else:
                vis = deepcopy(data_jsons[-1])
                for i in range(len(vis)):
                    pos, quat = np.array(vis[i]['pos']), np.array(vis[i]['quat'])
                    pose = np.hstack([pos.reshape(-1, 3), quat.reshape(-1, 4)]).flatten()
                    vis[i]['pos'] = pose.tolist()
                vis_data.append(vis)

        return data_jsons, extra_state_infos, vis_data

    def _interp_ctrls_act_lens(self, extra_state_infos_cpy, max_times):
        data_controls = []
        data_act_lens = []
        for j, e_data in enumerate(extra_state_infos_cpy):
            controls, dl_rates = [], []
            for i in range(len(e_data) - 1):
                if e_data[i + 1]['time'] > max_times[j]:
                    continue

                dt = e_data[i + 1]['time'] - e_data[i]['time']
                num_steps = round(dt / self.dt)
                ctrl = torch.tensor(e_data[i]['controls'], dtype=self.dtype).reshape(1, -1, 1)
                controls.extend([ctrl] * num_steps)

            gt_rest_lens = []
            last_idx = -1
            for i in range(len(e_data) - 1):
                if e_data[i + 1]['time'] > max_times[j]:
                    last_idx = i
                    break

                r0, r1 = e_data[i]['rest_lengths'], e_data[i + 1]['rest_lengths']
                t0, t1 = e_data[i]['time'], e_data[i + 1]['time']
                n = round((t1 - t0) / 0.01)
                gt_rest_lens.append(r0)
                for k in range(1, n):
                    t = t0 + k * self.dt
                    w = (t1 - t) / (t1 - t0)
                    r = [w * r0[m] + (1 - w) * r1[m] for m in range(len(r0))]
                    gt_rest_lens.append(r)
            gt_rest_lens.append(e_data[last_idx]['rest_lengths'])

            rest_rest_lens = [c._rest_length for c in self.simulator.robot.actuated_cables.values()]
            data_act_lens.append([
                [r - g for g, r in zip(gt_rest_lens[i], rest_rest_lens)]
                for i in range(len(gt_rest_lens))
            ])

            data_controls.append(controls)
        return data_controls, data_act_lens

    def _data_json_to_states(self,
                             data_jsons,
                             gt_end_pts,
                             times,
                             extra_state_jsons=None,
                             data_names=None):
        data_pos, data_quats, data_controls = self._data_to_pos_quat_ctrls(
            data_jsons,
            gt_end_pts,
            extra_state_jsons
        )
        data_states = self._pos_quat_to_states(
            data_pos,
            data_quats,
            times,
            data_names
        )

        return data_states, data_controls

    def _pos_quat_to_states(self, data_pos, data_quats, times, data_names=None):
        num_rods = len(self.simulator.robot.rigid_bodies)

        data_states = []
        for k, (pos, quats, times) in enumerate(zip(data_pos, data_quats, times)):
            states = []
            for i in range(len(pos)):
                prev_i = max(i - 1, 0)
                if data_names is not None and 'real' in data_names[k]:
                    while round(times[i] - times[prev_i], 5) < self.vel_min_dt and prev_i > 0:
                        prev_i -= 1

                pos_0, pos_1 = pos[prev_i], pos[i]
                quat_0, quat_1 = quats[prev_i], quats[i]
                dt = times[i] - times[prev_i]

                if i > 0:
                    lin_vels = [(pos_1[j] - pos_0[j]) / dt for j in range(num_rods)]
                    ang_vels = [
                        torch_quaternion.compute_ang_vel_quat(
                            quat_0[j],
                            quat_1[j],
                            dt
                        ) for j in range(num_rods)
                    ]
                else:
                    lin_vels = [torch.zeros_like(pos_0[j]) for j in range(num_rods)]
                    ang_vels = [torch.zeros_like(pos_0[j]) for j in range(num_rods)]

                state = torch.hstack([
                    torch.hstack([pos_1[j], quat_1[j], lin_vels[j], ang_vels[j]])
                    for j in range(num_rods)
                ])
                states.append(state)
            data_states.append(states)

        return data_states

    def _filter_endcaps(self, all_end_caps, threshold_cycle=0.5, threshold_upright=0.5):
        keep_indices = []
        for i, end_caps in enumerate(all_end_caps):
            left = end_caps[::2].mean(dim=0, keepdim=True)
            right = end_caps[1::2].mean(dim=0, keepdim=True)
            prin = right - left
            prin[:, 2] = 0.
            prin = prin / prin.norm(dim=1, keepdim=True)

            # left cycle
            l0 = end_caps[2:3] - end_caps[:1]
            l1 = end_caps[4:5] - end_caps[:1]
            l2 = torch.cross(l1, l0, dim=1)
            l2_upright = l2[:, 2:].abs() / l2.norm(dim=1, keepdim=True)
            l2[:, 2] = 0.
            l2 = l2 / l2.norm(dim=1, keepdim=True)

            # right cycle
            r0 = end_caps[3:4] - end_caps[1:2]
            r1 = end_caps[5:] - end_caps[1:2]
            r2 = torch.cross(r1, r0, dim=1)
            r2_upright = r2[:, 2:].abs() / r2.norm(dim=1, keepdim=True)
            r2[:, 2] = 0.
            r2 = r2 / r2.norm(dim=1, keepdim=True)

            z = torch.zeros_like(l2)
            z[:, 2] = 1.

            l2_cycle = torch.linalg.vecdot(l2, prin, dim=1).unsqueeze(1)
            r2_cycle = torch.linalg.vecdot(r2, prin, dim=1).unsqueeze(1)

            if l2_cycle > threshold_cycle and r2_cycle > threshold_cycle \
                    and l2_upright < threshold_upright and r2_upright < threshold_upright:
                keep_indices.append(i)

        return keep_indices

    def _filter_mismatched_endcap_lens(self, all_end_caps, all_sensor_lens, threshold=0.5):
        sensor_order = [[3, 1, 1, 0, 0, 2, 2, 0, 1], [5, 3, 5, 2, 4, 4, 5, 3, 4]]
        keep_indices = []
        for j, end_caps in enumerate(all_end_caps):
            diffs = (end_caps[sensor_order[1]] - end_caps[sensor_order[0]]).norm(dim=1, keepdim=True)
            diffs = (diffs.flatten() - all_sensor_lens[j]).abs()
            if (diffs < threshold).all():
                keep_indices.append(j)

        return keep_indices

    def _filter_data(self, all_data_jsons, all_extra_state_jsons, all_data_names):
        all_filtered_data_jsons, all_filtered_extra_state_jsons = [], []
        for i in range(len(all_data_jsons)):
            data_json = all_data_jsons[i]
            extra_state_json = all_extra_state_jsons[i]

            filtered_data_json = data_json[:-10]
            filtered_extra_state_json = extra_state_json[:-10]

            end_caps = [torch.tensor(d['end_pts'], dtype=self.dtype)
                        for d in filtered_data_json]
            sensor_lens = [torch.tensor(e['sensor_lens'], dtype=self.dtype)
                           for e in filtered_extra_state_json]

            keep_indices = self._filter_mismatched_endcap_lens(end_caps, sensor_lens)
            filtered_end_caps = [end_caps[i] for i in keep_indices]
            filtered_data_json = [data_json[i] for i in keep_indices]
            filtered_extra_state_json = [extra_state_json[i] for i in keep_indices]

            if keep_indices[0] != 0:
                filtered_end_caps = [end_caps[0]] + filtered_end_caps
                filtered_data_json = [data_json[0]] + filtered_data_json
                filtered_extra_state_json = [extra_state_json[0]] + filtered_extra_state_json

            keep_indices = self._filter_endcaps(filtered_end_caps)
            filtered_data_json = [filtered_data_json[i] for i in keep_indices]
            filtered_extra_state_json = [filtered_extra_state_json[i] for i in keep_indices]

            if keep_indices[0] != 0:
                filtered_data_json = [data_json[0]] + filtered_data_json
                filtered_extra_state_json = [extra_state_json[0]] + filtered_extra_state_json

            all_filtered_data_jsons.append(filtered_data_json)
            all_filtered_extra_state_jsons.append(filtered_extra_state_json)

        return all_filtered_data_jsons, all_filtered_extra_state_jsons

    def _find_next_t(self, t, future_times):
        best_dt, best_next_t, best_next_idx = 999999., None, None
        for i, f_t in enumerate(future_times):
            dt = f_t - t
            if self.min_dt <= dt <= self.max_dt and abs(dt - self.target_dt) < best_dt:
                best_dt = dt
                best_next_t = f_t
                best_next_idx = i
            elif dt > self.max_dt:
                break

        return best_next_t, best_next_idx

    def evaluate_rollouts(self, data_dict, prefix=''):
        real_idxs = [i for i, n in enumerate(data_dict['names']) if 'real' in n]
        sim_idxs = [i for i, n in enumerate(data_dict['names']) if 'real' not in n]

        real_data_dict = {k: [v[i] for i in real_idxs] for k, v in data_dict.items()}
        sim_data_dict = {k: [v[i] for i in sim_idxs] for k, v in data_dict.items()}

        real_n_step_loss = self.eval_n_step_aheads(200, real_data_dict)
        real_com_loss, real_angle_loss = self.eval_rollout_w_ctrls(real_data_dict)

        sim_n_step_loss = self.eval_n_step_aheads(200, sim_data_dict)
        sim_com_loss, sim_angle_loss = self.eval_rollout_w_ctrls(sim_data_dict)

        full_traj_loss = (self.eval_weight_real * real_com_loss + (1 - self.eval_weight_real) * sim_com_loss)
        n_step_loss = (self.eval_weight_real * real_n_step_loss + (1 - self.eval_weight_real) * sim_n_step_loss)

        other_losses = [
            real_com_loss,
            real_angle_loss,
            real_n_step_loss,
            sim_com_loss,
            sim_angle_loss,
            sim_n_step_loss
        ]

        return full_traj_loss, n_step_loss, other_losses

    def eval_n_step_aheads(self, n, data_dict, num_samples=1000):
        trajs = data_dict['states']
        all_act_lengths = data_dict['act_lengths']
        all_controls = data_dict['controls']
        all_gt_endpts = data_dict['gt_end_pts']
        times = data_dict['times']
        data_names = data_dict['names']

        total_loss = 0.0
        total_com_loss, total_angle_loss = 0.0, 0.0
        for i in range(len(trajs)):
            states = [trajs[i][0].clone() for _ in range(self.num_hist - 1)] + trajs[i]
            act_lengths = torch.vstack([torch.hstack(a) for a in all_act_lengths[i]])

            curr_state = torch.vstack(states).to(self.device)

            idx_starts, idx_ends = self._compute_eval_n_steps_idxs(
                times[i], n * self.dt, max_sample=num_samples
            )
            num_steps = [round((times[i][e] - times[i][s]) / self.dt)
                         for s, e in zip(idx_starts, idx_ends)]
            max_steps, min_steps = max(num_steps), min(num_steps)
            num_steps = [num - min_steps for num in num_steps]

            all_ctrls = torch.vstack(
                all_controls[i] + [torch.zeros_like(all_controls[i][0])] * (max_steps - min_steps)
            ).to(self.device)
            ctrl_idx_starts = torch.tensor([round(times[i][idx] / self.dt) for idx in idx_starts])
            assert all_ctrls.shape[0] > max_steps

            self.simulator.reset(
                act_len=act_lengths[idx_starts].clone().to(self.device),
            )

            curr_state = curr_state[idx_starts]

            dataset_idx = torch.full(
                (curr_state.shape[0], 1),
                data_dict['dataset_idx'][i],
                dtype=torch.int
            )

            gt_act_lens, ctrls = self._get_n_step_act_lens_ctrls(
                act_lengths[i], all_controls[i], ctrl_idx_starts, max_steps
            )

            states, _ = self.simulator.run(
                curr_state=curr_state,
                dt=self.dt,
                ctrls=ctrls,
                gt_act_lens=gt_act_lens,
                state_to_graph_kwargs={'dataset_idx': dataset_idx},
            )

            pred_endpts = torch.concat([self._batch_compute_end_pts(s) for s in states[min_steps - 1:]], dim=2)
            pred_endpts = pred_endpts[torch.arange(curr_state.shape[0]), :, num_steps]
            pred_endpts = pred_endpts.unsqueeze(-1)

            gt_endpts = torch.vstack([
                e.reshape(1, -1, 1)
                for e in all_gt_endpts[i]
            ]).to(self.device)[idx_ends]

            loss = ((pred_endpts - gt_endpts) ** 2).mean().detach().item()

            com_loss, angle_loss = self.compute_com_and_rot_loss(pred_endpts, gt_endpts)
            com_loss, angle_loss = com_loss.detach().item(), angle_loss.detach().item()
            total_com_loss += com_loss
            total_angle_loss += angle_loss

            self.logger.info(f'Eval {n}-steps: {data_names[i]}, {loss}, {com_loss}, {angle_loss}')

        total_loss /= len(trajs) if len(trajs) > 0 else 1
        total_com_loss /= len(trajs) if len(trajs) > 0 else 1
        total_angle_loss /= len(trajs) if len(trajs) > 0 else 1

        self.logger.info(f'Avg {n}-steps: {total_loss}, {total_com_loss}, {total_angle_loss}')

        return total_loss

    def eval_rollout_w_ctrls(self, data_dict):
        states = data_dict['states']
        data_jsons = data_dict['data_jsons']
        all_act_lengths = data_dict['act_lengths']
        all_controls = data_dict['controls']
        all_gt_endpts = data_dict['gt_end_pts']
        data_names = data_dict['names']
        all_dataset_idxs = data_dict['dataset_idx']

        device = self.device
        self.device = 'cpu'
        self.to('cpu')
        torch.set_num_threads(8)

        total_loss = 0.0
        total_com_loss, total_angle_loss = 0.0, 0.0
        for i in range(len(states)):
            act_lengths = [torch.hstack(a) for a in all_act_lengths[i]]
            self.simulator.reset(
                act_lens=act_lengths[0].clone().to(self.device),
            )
            dataset_idx = torch.tensor([[all_dataset_idxs[i]]], dtype=torch.int)

            curr_state = states[i][0].clone().to(self.device)
            controls = torch.cat(all_controls[i], dim=-1).to(self.device) if not self.use_gt_act_lens else None
            gt_act_lens = torch.cat(act_lengths[1:], dim=-1).to(self.device) if self.use_gt_act_lens else None

            all_states, graphs, _ = self.simulator.run(
                curr_state=curr_state,
                dt=self.dt,
                ctrls=controls,
                gt_act_lens=gt_act_lens,
                state_to_graph_kwargs={'dataset_idx': dataset_idx},
                show_progress=True
            )

            idxs = [round(data_jsons[i][j]['time'] / self.dt) - 1
                    for j in range(1, len(data_jsons[i]))]

            pred_endpts = torch.vstack([
                self._batch_compute_end_pts(all_states[j])
                for j in idxs
            ])
            gt_endpts = torch.vstack([
                e.reshape(1, -1, 1)
                for e in all_gt_endpts[i][1:]
            ]).to(self.device)

            loss = ((pred_endpts - gt_endpts) ** 2).mean().detach().item()
            com_loss, mean_angle_loss = self.compute_com_and_rot_loss(pred_endpts, gt_endpts)

            self.logger.info(f"Eval Rollout: {data_names[i]}, "
                             f"{loss}, "
                             f"{com_loss}, "
                             f"{mean_angle_loss}")

            total_com_loss += com_loss
            total_angle_loss += mean_angle_loss

        total_loss /= len(states) if len(states) > 0 else 1
        total_com_loss /= len(states) if len(states) > 0 else 1
        total_angle_loss /= len(states) if len(states) > 0 else 1
        self.device = device
        self.to(device)

        self.logger.info(f'Total loss: {total_loss}, com loss: {total_com_loss}, angle loss: {total_angle_loss}')

        return total_com_loss, total_angle_loss

    def _compute_eval_n_steps_idxs(self, times, time_ahead, pct_tol=0.03, max_sample=200):
        min_dt, max_dt = time_ahead * (1 - pct_tol), time_ahead * (1 + pct_tol)

        idx_starts, idx_ends, tmp = [], [], []
        for i in range(len(times) - 1):
            t0 = times[i]
            best_, best_idx, best_dt = 99999., None, 99999.
            for j in range(i + 1, len(times)):
                t1 = times[j]
                dt = t1 - t0
                if min_dt <= dt <= max_dt <= (times[-1] - t1) and abs(dt - time_ahead) < best_:
                    best_, best_idx = abs(dt - time_ahead), j
                    best_dt = dt
            tmp.append(best_dt)
            if best_idx is not None:
                idx_starts.append(i)
                idx_ends.append(best_idx)

        idxs = list(range(len(idx_starts)))
        random.shuffle(idxs)
        idxs = idxs[:max_sample]
        idx_starts = [idx_starts[i] for i in idxs]
        idx_ends = [idx_ends[i] for i in idxs]

        return idx_starts, idx_ends

    def _accum_dv_normalizer(self, gt_nodes_pos):
        p_vel = (gt_nodes_pos[:, 6:] - gt_nodes_pos[:, 3:-3]) / self.dt
        vel = (gt_nodes_pos[:, 3:-3] - gt_nodes_pos[:, :-6]) / self.dt
        gt_dv = p_vel - vel
        self.simulator.data_processor.normalizers['node_dv'](gt_dv)

    def _compute_node_loss(self, graphs, gt_end_pts, dt, mask=None):
        def reshape_node_pos_tensor(pos):
            return pos.unsqueeze(1).reshape(pos.shape[0], -1, 3).transpose(1, 2)

        assert mask is not None, "Mask cannot be None"

        num_out_steps = self.simulator.num_out_steps \
            if hasattr(self.simulator, 'num_out_steps') else 1
        body_mask = graphs[0].body_mask.flatten()

        all_norm_gt_pos, all_norm_p_pos = [], []
        all_gt_pos, all_p_pos = [], []
        for i in range(len(graphs)):
            start = i * num_out_steps
            end = min(start + num_out_steps, gt_end_pts.shape[-1])

            gt_nodes_pos_lis = [graphs[i].pos[body_mask]]
            for j in range(start, end):
                end_pts = gt_end_pts[..., j:j + 1].reshape(-1, 6, 1)
                gt_pos, gt_quat = self.endpts2pos(end_pts[:, :3], end_pts[:, 3:])
                single_gt_nodes_pos = self.simulator.data_processor.pose2node(
                    gt_pos, gt_quat, gt_end_pts.shape[0]
                )
                gt_nodes_pos_lis.append(single_gt_nodes_pos)

            if i == 0 and not self.load_sim:
                gt_prev_pos = gt_nodes_pos_lis[0] - graphs[i].vel[body_mask] * self.dt
                gt_node_pos = torch.hstack([gt_prev_pos] + gt_nodes_pos_lis)
                self._accum_dv_normalizer(gt_node_pos)

            gt_nodes_pos = torch.hstack(gt_nodes_pos_lis[1:])

            dim = gt_nodes_pos.shape[1]
            pred_node_pos = graphs[i].p_pos[body_mask].transpose(1, 2).reshape(gt_nodes_pos.shape[0], -1)[:, :dim]

            dv_normalizer = self.simulator.data_processor.normalizers['node_dv']
            norm_gt_pos = gt_nodes_pos / dv_normalizer.std_w_eps[:, :dim] / self.dt
            norm_p_pos = pred_node_pos / dv_normalizer.std_w_eps[:, :dim] / self.dt

            all_gt_pos.append(reshape_node_pos_tensor(gt_nodes_pos))
            all_p_pos.append(reshape_node_pos_tensor(pred_node_pos))
            all_norm_gt_pos.append(reshape_node_pos_tensor(norm_gt_pos))
            all_norm_p_pos.append(reshape_node_pos_tensor(norm_p_pos))

        num_nodes = all_norm_p_pos[0].shape[0] // mask.shape[0]
        mask = mask.repeat_interleave(num_nodes, dim=0)

        all_norm_gt_pos = torch.concat(all_norm_gt_pos, dim=-1) * mask
        all_norm_p_pos = torch.concat(all_norm_p_pos, dim=-1) * mask
        loss = self.loss_fn(all_norm_p_pos, all_norm_gt_pos)

        with torch.no_grad():
            all_gt_pos = torch.concat(all_gt_pos, dim=-1) * mask
            all_p_pos = torch.concat(all_p_pos, dim=-1) * mask
            pos_loss = self.loss_fn(all_gt_pos, all_p_pos)

        return loss, pos_loss.detach().item()

    def _compute_cable_dl_loss(self, graphs, gt_cable_dls):
        pred_cable_dls = torch.hstack([g.act_cable_dl for g in graphs])
        gt_act_cable_dls = gt_cable_dls.repeat_interleave(2, dim=1).reshape(-1, gt_cable_dls.shape[-1])

        n_out = self.simulator.num_out_steps
        num_steps = math.ceil(round(gt_act_cable_dls.shape[1] / n_out, 2))
        normalizer = self.simulator.data_processor.normalizers['cable_dl']

        norm_gt_cable_dls = []
        for i in range(num_steps):
            cable_dls = gt_act_cable_dls[:, i * n_out: (i + 1) * n_out]
            dim = cable_dls.shape[1]
            step_diff = n_out - dim
            if step_diff > 0:
                pad = torch.zeros_like(cable_dls[:, :1]).repeat(1, step_diff)
                cable_dls = torch.hstack([cable_dls, pad])
            norm_cable_dls = normalizer(cable_dls)
            norm_gt_cable_dls.append(norm_cable_dls[:, :dim])
        norm_gt_cable_dls = torch.hstack(norm_gt_cable_dls)

        act_cable_mask = graphs[0].cable_actuated_mask.flatten()
        norm_pred_cable_dls = torch.hstack([g.cable_decode_output[act_cable_mask] for g in graphs])

        norm_cable_dl_loss = self.loss_fn(
            norm_pred_cable_dls[:, :gt_act_cable_dls.shape[1]],
            norm_gt_cable_dls
        )
        cable_dl_loss = self.loss_fn(
            pred_cable_dls[:, :gt_act_cable_dls.shape[1]],
            gt_act_cable_dls
        ).detach().item()

        return norm_cable_dl_loss, cable_dl_loss
