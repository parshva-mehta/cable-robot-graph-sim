import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data as Graph

from nn_training.datasets.tensegrity_dataset import TensegrityDataset
from simulators.tensegrity_gnn_simulator import *
from state_objects.primitive_shapes import Cylinder
from utilities import torch_quaternion
from utilities.misc_utils import save_curr_code, DEFAULT_DTYPE


MAX_EPOCH_NO_IMPROVE = 10
NUM_PRINT_STATUS_STEP = 100


class TensegrityMultiSimMultiStepMotorGNNTrainingEngine(nn.Module):
    config: dict
    sim_config: dict
    output_dir: str | Path
    data_root: str | Path
    dt: float
    dtype: torch.dtype
    simulator: LearnedSimulator
    num_steps_fwd: int
    num_hist: int
    batch_size_per_update: int
    batch_size_per_step: int
    num_grad_accum: int
    curr_accum_step: int
    load_sim: bool
    load_sim_path: str | Path | None
    output_dir: str | Path
    data_root: str | Path
    logger: logging.Logger
    best_val_loss: float
    best_rollout_loss: float
    best_n_step_rollout_loss: float
    best_train_loss: float
    num_no_improve: int
    eval_step_size: int
    epoch_num: int
    num_eval_n_steps: int
    train_data_dict: Dict
    train_dataset: TensegrityDataset
    train_dataloader: DataLoader
    val_data_dict: Dict
    val_dataset: TensegrityDataset
    val_dataloader: DataLoader
    trainable_params: torch.nn.ParameterList
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    loss_fn: nn.Module

    def __init__(self,
                 training_config: Dict,
                 logger: logging.Logger):
        super().__init__()

        # Load simulator config
        self.sim_config = training_config['sim_config']
        if isinstance(training_config['sim_config'], str):
            with open(training_config['sim_config'], "r") as j:
                self.sim_config = json.load(j)

        # Load relevant constants from cfg
        self.num_ctrls_hist = self.sim_config.get('num_ctrls_hist', 1)
        self.num_sims = training_config.get('num_sims', 10)
        self.dt = self.sim_config['dt']
        self.dtype = DEFAULT_DTYPE
        self.config = training_config
        self.num_steps_fwd = self.config.get('num_steps_fwd', 1)
        self.num_hist = self.config.get('num_hist', 1)
        self.use_gt_act_lens = self.config.get('use_gt_act_lens', False)
        self.batch_size_per_update = self.config.get('batch_size_per_update', 128)
        self.batch_size_per_step = self.config.get('batch_size_per_step', 128)
        self.num_grad_accum = round(self.batch_size_per_update / self.batch_size_per_step)
        self.num_eval_n_steps = self.config.get('num_eval_n_steps', 200)
        self.eval_step_size = self.config.get('eval_step_size', 20)
        self.load_sim = self.config.get('load_sim', False)
        self.load_sim_path = self.config.get('load_sim_path', None)

        assert self.batch_size_per_update % self.batch_size_per_step == 0
        self.sim_config['cache_batch_sizes'] = [1, self.batch_size_per_step]

        self.output_dir = self.config['output_path']
        Path(self.output_dir).mkdir(exist_ok=True)

        self.logger = logger

        # tracking vars
        self.device = 'cpu'
        self.curr_accum_step = 1
        self.best_val_loss = 1e20
        self.best_rollout_loss = 1e20
        self.best_n_step_rollout_loss = 1e20
        self.best_train_loss = 1e20
        self.num_no_improve = 10
        self.epoch_num = 0

        if self.config.get('save_code', True):
            self._save_code()

        # Load datasets and dataloaders
        self.data_root = training_config['data_root']

        train_data_paths = [Path(self.data_root, p) for p in training_config['train_data_paths']]
        self.train_data_dict = self._init_data(train_data_paths)
        self.train_dataset = self._get_dataset(self.train_data_dict)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_per_step,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        val_data_paths = [Path(self.data_root, p) for p in training_config['val_data_paths']]
        self.val_data_dict = self._init_data(val_data_paths)
        self.val_dataset = self._get_dataset(self.val_data_dict)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_step,
            shuffle=True,
            collate_fn=self.val_dataset.collate_fn
        )

        self.simulator = self._get_simulator()

        # Optimization params
        self.trainable_params = torch.nn.ParameterList(self.simulator.parameters())
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **training_config['optimizer_params']
        )
        self.loss_fn = torch.nn.MSELoss()

    def _init_data(self, data_paths):
        data_dict = {}
        data_dict['names'] = [p.name for p in data_paths]

        data_jsons, extra_state_infos = self._load_json_files(data_paths)
        self._check_and_mod_data_length(data_jsons, extra_state_infos)
        data_dict.update({'data_jsons': data_jsons, 'extra_state_infos': extra_state_infos})
        self.logger.info("Loaded data jsons")

        data_dict['gt_end_pts'] = self._get_end_pts(data_jsons)
        self.logger.info("Loaded end points")

        data_dict['times'] = [[d['time'] - data_json[0]['time'] for d in data_json]
                              for data_json in data_jsons]

        data_dict['states'], data_dict['controls'] = (
            self._data_json_to_states(data_jsons,
                                      data_dict['gt_end_pts'],
                                      data_dict['times'],
                                      extra_state_infos)
        )
        self.logger.info("Computed states and controls")

        data_dict['act_lengths'], data_dict['motor_omegas'] = (
            self._get_act_lens(extra_state_infos, data_dict['times'])
        )
        self.logger.info("Loaded act lengths and motor speeds")

        data_dict['dataset_idx'] = self._enum_dataset(data_paths)

        return data_dict

    def _check_and_mod_data_length(self, data_jsons, extra_state_infos):
        for i in range(len(data_jsons)):
            data_json, extra_state_info = data_jsons[i], extra_state_infos[i]
            if len(data_json) == len(extra_state_info):
                continue
            elif len(data_json) - len(extra_state_info) == 1:
                data_jsons[i] = data_json[:-1]
            elif len(data_json) - len(extra_state_info) == -1:
                extra_state_infos[i] = extra_state_info[:-1]
            else:
                raise Exception("Data lengths between processed data and extra state info does not match")

    def _enum_dataset(self, data_paths):
        dataset_idx = []
        for p in data_paths:
            p = Path(p)
            try:
                idx = int(p.parent.name.split("_")[-1])
            except Exception:
                self.logger.info(f"No dataset index for {p.parent.name}, defaulting to 0")
                idx = 0
            dataset_idx.append(idx)
        return dataset_idx

    def _compute_node_loss(self, graphs, gt_end_pts, dt):
        norm_pred_dv, norm_gt_dv = [], []
        body_mask = graphs[0].body_mask.flatten()

        num_inner_preds = graphs[0].p_dv.shape[-1]

        for i, graph in enumerate(graphs):
            norm_pred_dv.append(graph.decode_output[body_mask])

            graph_gt_dv = []
            prev_gt_pos = graphs[i].pos[body_mask]
            prev_gt_vel = graphs[i].vel[body_mask]
            for j in range(i * num_inner_preds, (i + 1) * num_inner_preds):
                if j >= gt_end_pts.shape[-1]:
                    break

                end_pts = gt_end_pts[..., j].reshape(-1, 6, 1)
                gt_pos, gt_quat = self.endpts2pos(end_pts[:, :3], end_pts[:, 3:])
                gt_nodes_pos = self.simulator.data_processor.pose2node(
                    gt_pos, gt_quat, gt_end_pts.shape[0],
                    augment_grnd=False
                )
                gt_node_vel = (gt_nodes_pos - prev_gt_pos) / dt
                gt_node_dv = gt_node_vel - prev_gt_vel
                graph_gt_dv.append(gt_node_dv)

                prev_gt_pos, prev_gt_vel = gt_nodes_pos, gt_node_vel

            graph_gt_dv = torch.hstack(graph_gt_dv)
            norm_gt_dv.append(self.simulator.data_processor.normalizers['node_dv'](graph_gt_dv))

        norm_gt_dv = torch.hstack(norm_gt_dv)
        norm_pred_dv = torch.hstack(norm_pred_dv)[:, :norm_gt_dv.shape[1]]

        loss = self.loss_fn(norm_pred_dv, norm_gt_dv)
        pos_loss = self.loss_fn(
            graphs[-1].p_pos[body_mask, :, -1],
            gt_nodes_pos
        ).detach().item()

        return loss, pos_loss

    @staticmethod
    def _get_act_lens(extra_state_jsons, max_act_rest_lens):
        data_act_lens = []

        for extra_state_json in extra_state_jsons:
            act_lens, motor_omegas = [], []

            for e in extra_state_json:
                act_lengths = torch.tensor(
                    [max_act_rest_lens[i] - e['rest_lengths'][i]
                     for i in range(len(e['rest_lengths']))]
                ).reshape(1, -1, 1)
                act_lens.append(act_lengths)

            data_act_lens.append(act_lens)

        return data_act_lens

    def _get_simulator(self):
        if self.load_sim and self.load_sim_path:
            sim = torch.load(self.load_sim_path, map_location="cpu", weights_only=False)
            sim.reset()
            sim.cpu()
            print("Loaded simulator")
        else:
            sim = TensegrityGNNSimulator(
                gnn_params=self.sim_config['gnn_params'],
                tensegrity_cfg=self.sim_config['tensegrity_cfg'],
                num_datasets=self.sim_config['num_sims'],
                num_ctrls_hist=self.sim_config['num_ctrls_hist'],
                dt=self.sim_config['dt'],
                cache_batch_sizes=[self.batch_size_per_step]
            )

        return sim

    def _compute_cable_dl_loss(self, graphs, gt_cable_dls):
        pred_cable_dls = torch.hstack([g.act_cable_dl for g in graphs])
        gt_act_cable_dls = gt_cable_dls.repeat_interleave(2, dim=1).reshape(-1, gt_cable_dls.shape[-1])

        n_out = self.simulator.num_out_steps
        num_steps = self.num_steps_fwd // n_out
        normalizer = self.simulator.data_processor.normalizers['cable_dl']
        norm_gt_cable_dls = torch.hstack([
            normalizer(gt_act_cable_dls[:, i * n_out: (i + 1) * n_out])
            for i in range(num_steps)
        ])

        act_cable_mask = graphs[0].cable_actuated_mask.flatten()
        norm_pred_cable_dls = torch.hstack([g.cable_decode_output[act_cable_mask] for g in graphs])

        norm_cable_dl_loss = self.loss_fn(norm_pred_cable_dls, norm_gt_cable_dls)
        cable_dl_loss = self.loss_fn(
            pred_cable_dls,
            gt_act_cable_dls
        ).detach().item()

        return norm_cable_dl_loss, cable_dl_loss

    def run_one_epoch(self,
                      batches,
                      grad_required=True,
                      rot_aug=False) -> List[float]:
        if grad_required:
            self.simulator.train()
        else:
            self.simulator.eval()

        total_loss, total_other_losses = 0.0, []
        num_train, curr_batch = 0, 0
        for batch in tqdm.tqdm(batches):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            curr_batch += 1
            num_train += batch['x'].shape[0]

            if rot_aug:
                batch['x'], batch['y'] = self.rotate_data_aug(batch['x'], batch['y'])

            graphs = self._batch_sim_ctrls(batch)
            losses = self._compute_node_loss(graphs, batch['y'], self.dt)

            if self.use_gt_act_lens:
                norm_cable_loss, cable_loss = 0.0, 0.0
            else:
                act_lens, next_act_lens = batch['act_len'], batch['next_act_lens']
                cable_dls = next_act_lens - torch.concat([act_lens, next_act_lens[..., :-1]], dim=2)
                norm_cable_loss, cable_loss = self._compute_cable_dl_loss(graphs, cable_dls)

            backward_loss = losses[0] + (5 / self.num_steps_fwd) * norm_cable_loss
            losses = [backward_loss, norm_cable_loss.detach().item(), cable_loss] + [l for l in losses[1:]]

            # If gradient updates required, run backward pass
            if grad_required:
                self._backward(backward_loss)

            total_loss += losses[0].detach().item() * batch['x'].shape[0]
            total_other_losses.append([
                l * batch['x'].shape[0] for l in losses[1:]
            ])

            if curr_batch % NUM_PRINT_STATUS_STEP == 0:
                avg_other_losses = [
                    sum(l) / num_train
                    for l in zip(*total_other_losses)
                ]
                print(total_loss / num_train, avg_other_losses)

        total_loss /= num_train
        avg_other_losses = [
            sum(l) / num_train
            for l in zip(*total_other_losses)
        ]

        losses = [total_loss] + avg_other_losses

        return losses

    def _batch_sim_ctrls(self, batch) -> List[Graph]:
        batch_state = batch['x']
        controls = batch['ctrl']
        act_lens = batch['act_len']
        next_act_lens = batch['next_act_lens']
        dataset_idx = batch['dataset_idx']

        self.simulator.reset(
            act_lens=act_lens[..., -1:].clone(),
            ctrls_hist=batch['ctrls_hist'].clone()
        )

        curr_state = batch_state[..., -1:].clone()
        if self.use_gt_act_lens:
            gt_act_lens, ctrls = [], None
            for i in range(self.num_steps_fwd):
                gt_act_lens.append(next_act_lens[:, :, i: i + 1].clone())
                ctrls = None
            gt_act_lens = torch.cat(gt_act_lens, dim=-1)
        else:
            gt_act_lens, ctrls = None, controls.clone()

        states, graphs = self.simulator.run(
            curr_state=curr_state,
            dt=self.dt,
            ctrls=ctrls,
            gt_act_lens=gt_act_lens,
            state_to_graph_kwargs={"dataset_idx": dataset_idx}
        )

        return graphs

    def eval_n_step_aheads(self, n, data_dict, num_samples=500):
        trajs = data_dict['states']
        act_lengths = data_dict['act_lengths']
        all_controls = data_dict['controls']
        all_gt_endpts = data_dict['gt_end_pts']
        all_dataset_idxs = data_dict['dataset_idx']

        total_loss = 0.0
        total_com_loss, total_angle_loss = 0.0, 0.0
        for i in range(len(trajs)):
            states = trajs[i]

            idxs = list(range(len(states) - n - 1))
            random.shuffle(idxs)
            idxs = idxs[:num_samples]

            dataset_idx = torch.tensor(
                [[all_dataset_idxs[i]]], dtype=torch.int
            ).repeat(len(idxs), 1)

            curr_state = torch.vstack(states[:-n]).to(self.device)[idxs]

            num_ctrls = len(all_controls[i])
            ctrls_hist = [torch.zeros((1, all_controls[i][0].shape[1], 1), dtype=curr_state.dtype)
                          for _ in range(self.num_ctrls_hist)]
            ctrls_hist = torch.vstack(ctrls_hist + all_controls[i])
            ctrls_hist = torch.concat([
                ctrls_hist[j: j + num_ctrls]
                for j in range(self.num_ctrls_hist)
            ], dim=2)[idxs].to(self.device)

            self.simulator.reset(
                act_lens=torch.vstack(act_lengths[i][:-n + 1])[idxs].to(self.device),
                ctrls_hist=ctrls_hist
            )

            gt_act_lens, ctrls = self._get_n_step_act_lens_ctrls(
                act_lengths[i],
                all_controls[i],
                idxs,
                n
            )

            states, _, _ = self.simulator.run(
                curr_state=curr_state,
                dt=self.dt,
                ctrls=ctrls,
                gt_act_lens=gt_act_lens,
                state_to_graph_kwargs={'dataset_idx': dataset_idx},
                show_progress=True
            )

            pred_endpts = self.batch_compute_end_pts(states[-1])
            gt_endpts = torch.vstack([
                torch.hstack(e)
                for e in all_gt_endpts[i][n:]
            ]).to(self.device)[idxs]

            loss = self.loss_fn(pred_endpts, gt_endpts).detach().item()
            total_loss += loss

            com_loss, angle_loss = self.compute_com_and_rot_loss(pred_endpts, gt_endpts)
            total_com_loss += com_loss
            total_angle_loss += angle_loss

            self.logger.info(f'{loss}, {com_loss}, {angle_loss}')

        total_loss /= len(trajs) if len(trajs) > 0 else 1
        total_com_loss /= len(trajs) if len(trajs) > 0 else 1
        total_angle_loss /= len(trajs) if len(trajs) > 0 else 1

        self.logger.info(f"Avg {n}-step: {total_com_loss}, {total_angle_loss}")

        return total_loss, total_com_loss, total_angle_loss

    def eval_rollout_w_ctrls(self, data_dict):
        states = data_dict['states']
        act_lengths = data_dict['act_lengths']
        motor_omegas = data_dict['motor_omegas']
        all_controls = data_dict['controls']
        all_gt_endpts = data_dict['gt_end_pts']
        all_dataset_idxs = data_dict['dataset_idx']

        device = self.device
        self.device = 'cpu'
        self.to('cpu')
        torch.set_num_threads(8)

        total_loss = 0.0
        total_com_loss, total_angle_loss = 0.0, 0.0
        for i in range(len(states)):
            self.simulator.reset(
                act_lens=act_lengths[i][0].clone().to(self.device),
                motor_speeds=motor_omegas[i][0].clone().to(self.device)
            )
            dataset_idx = torch.tensor([[all_dataset_idxs[i]]], dtype=torch.int)

            curr_state = states[i][0].clone().to(self.device)
            controls = torch.cat(all_controls[i], dim=-1).to(self.device) if not self.use_gt_act_lens else None
            gt_act_lens = torch.cat(act_lengths[i][1:], dim=-1).to(self.device) if self.use_gt_act_lens else None

            all_states, graphs, _ = self.simulator.run(
                curr_state=curr_state,
                dt=self.dt,
                ctrls=controls,
                gt_act_lens=gt_act_lens,
                state_to_graph_kwargs={'dataset_idx': dataset_idx},
                show_progress=True
            )

            pred_endpts = torch.vstack([self.batch_compute_end_pts(s) for s in all_states])
            gt_endpts = torch.vstack([
                torch.hstack(e)
                for e in all_gt_endpts[i][1:]
            ]).to(self.device)

            loss = self.loss_fn(pred_endpts, gt_endpts).detach().item()
            total_loss += loss

            com_loss, angle_loss = self.compute_com_and_rot_loss(pred_endpts, gt_endpts)
            total_com_loss += com_loss
            total_angle_loss += angle_loss

            self.logger.info(f'{loss}, {com_loss}, {angle_loss}')

        total_loss /= len(states) if len(states) > 0 else 1
        total_com_loss /= len(states) if len(states) > 0 else 1
        total_angle_loss /= len(states) if len(states) > 0 else 1
        self.device = device
        self.to(device)

        self.logger.info(f'Total loss: {total_loss}, '
                         f'Com loss: {total_com_loss}, '
                         f'Angle loss: {total_angle_loss}')

        return total_loss, total_com_loss, total_angle_loss

    def _get_dataset(self, data_dict):
        return TensegrityDataset(
            data_dict,
            num_steps_fwd=self.num_steps_fwd,
            dt=self.dt,
            num_ctrls_hist=self.num_ctrls_hist,
        )

    @staticmethod
    def rotate_data_aug(batch_x, gt_end_pts):
        n = gt_end_pts.shape[1] // 3

        angle = 2 * torch.pi * (torch.rand((batch_x.shape[0], 1, 1), device=batch_x.device) - 0.5)
        w = torch.cos(angle / 2)
        xyz = torch.tensor([0, 0, 1],
                           dtype=batch_x.dtype,
                           device=batch_x.device
                           ).reshape(1, 3, 1)
        xyz = (xyz.repeat(batch_x.shape[0], 1, 1) * torch.sin(angle / 2))
        q = torch.hstack([w, xyz]).repeat(1, n, 1).reshape(-1, 4, 1)

        batch_x_rots = []
        for i in range(batch_x.shape[2]):
            batch_x_i = batch_x[..., i: i + 1].reshape(-1, 13, 1)
            pos = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, :3])
            quat = torch_quaternion.quat_prod(q, batch_x_i[:, 3:7])
            linvel = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, 7:10])
            angvel = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, 10:])

            batch_x_rots.append(torch.hstack([
                pos, quat, linvel, angvel
            ]).reshape(-1, 13 * n, 1))

        batch_x_rots = torch.concat(batch_x_rots, dim=2)

        gt_end_pts_rots = []
        for i in range(gt_end_pts.shape[2]):
            gt_end_pts_ = gt_end_pts[:, :, i: i + 1].reshape(-1, 6, 1)
            endpt_0_rot = torch_quaternion.rotate_vec_quat(q, gt_end_pts_[:, :3])
            endpt_1_rot = torch_quaternion.rotate_vec_quat(q, gt_end_pts_[:, 3:])
            gt_end_pts_rot = torch.hstack([
                endpt_0_rot, endpt_1_rot
            ]).reshape(-1, 6 * n, 1)

            gt_end_pts_rots.append(gt_end_pts_rot)

        gt_end_pts_rots = torch.concat(gt_end_pts_rots, dim=2)

        return batch_x_rots, gt_end_pts_rots

    def train_epoch(self) -> Tuple:
        train_losses = self.run_one_epoch(self.train_dataloader, rot_aug=True)

        if train_losses[0] < self.best_train_loss:
            self.best_train_loss = train_losses[0]
            self.num_no_improve = 0
        else:
            self.num_no_improve += 1

        if self.num_no_improve > MAX_EPOCH_NO_IMPROVE:
            self.logger.info("No improvement, lowering learning rate")
            self.best_train_loss = train_losses[0]
            self.num_no_improve = 0
            for p in self.optimizer.param_groups:
                p['lr'] /= 2.0

        with torch.no_grad():
            if self.epoch_num % self.eval_step_size == 0:
                val_losses = self.run_one_epoch(self.val_dataloader, grad_required=False)

                self.simulator.eval()
                val_rollout_loss, val_n_steps_loss, val_other_rollout_losses = (
                    self.evaluate_rollouts(self.val_data_dict))

                self.simulator.train()
            else:
                val_losses = [-9.]
                val_rollout_loss = -9.
                val_n_steps_loss = -9.
                val_other_rollout_losses = []
        if -9. < val_losses[0] < self.best_val_loss:
            self.best_val_loss = val_losses[0]
            torch.save(
                self.simulator,
                Path(self.output_dir, "best_loss_model.pt")
            )
        if -9. < val_rollout_loss < self.best_rollout_loss:
            self.best_rollout_loss = val_rollout_loss
            torch.save(
                self.simulator,
                Path(self.output_dir, "best_rollout_model.pt")
            )
        if -9. < val_n_steps_loss < self.best_n_step_rollout_loss:
            self.best_n_step_rollout_loss = val_n_steps_loss
            torch.save(
                self.simulator,
                Path(self.output_dir, "best_n_step_rollout_model.pt")
            )

        losses = (
            train_losses,
            val_losses,
            [val_rollout_loss, val_n_steps_loss, *val_other_rollout_losses]
        )

        return losses

    def _log_status(self, losses: Tuple) -> None:
        """
        Method to print training status to console

        :param losses: Train loss, Val loss, Val rollout KF loss
        """
        losses = [f'{l:.4}' if isinstance(l, float) else [f'{ll:.4}' for ll in l] for l in losses]

        loss_file = Path(self.output_dir, "loss.txt")
        loss_msg = (f'Epoch {self.epoch_num}, '
                    f'"Train/Val/Val KF Losses": {losses}')

        try:
            with loss_file.open('a') as fp:
                fp.write(loss_msg + "\n")
        except:
            with loss_file.open('w') as fp:
                fp.write(loss_msg + "\n")

        self.logger.info(loss_msg)

    def _compute_init_losses(self):
        if not self.load_sim:
            self.simulator.data_processor.start_normalizers()

        train_loss = self.run_one_epoch(
            self.train_dataloader,
            grad_required=False,
            rot_aug=True
        )
        self.simulator.data_processor.stop_normalizers()

        self.logger.info(
            f"dv normalizer stats: "
            f"{self.simulator.data_processor.normalizers['node_dv'].mean} "
            f"{self.simulator.data_processor.normalizers['node_dv'].std_w_eps}"
        )

        val_loss = self.run_one_epoch(
            self.val_dataloader,
            grad_required=False
        )

        # if True:
        if self.load_sim:
            self.simulator.eval()
            val_rollout_loss = self.evaluate_rollouts(
                self.val_data_dict
            )
            self.simulator.train()
            self.logger.info(val_rollout_loss)

        losses = (train_loss, val_loss)

        return losses

    def run(self, num_epochs: int):
        """
        Method to run entire training

        :param num_epochs: Number of epochs to train
        :return: List of train losses, List of val losses, dictionary of trainable params and list of losses
        """
        with torch.no_grad():
            losses = self._compute_init_losses()
            self._log_status(losses)

        # Run training over num_epochs
        while self.epoch_num < num_epochs:
            self.epoch_num += 1

            # Run single epoch training and evaluation
            losses = self.train_epoch()
            self._log_status(losses)

        del self.logger

    def run_n_steps(self, num_steps: int):
        """
        Method to run entire training

        :param num_steps: Number of training steps
        :return: List of train losses, List of val losses, dictionary of trainable params and list of losses
        """
        num_epochs = num_steps // len(self.train_dataloader)
        return self.run(num_epochs)

    def _backward(self, loss: torch.Tensor) -> None:
        """
        Run back propagation with loss tensor

        :param loss: torch.Tensor
        """
        (loss / self.num_grad_accum).backward()

        if self.curr_accum_step >= self.num_grad_accum:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.curr_accum_step = 1
        else:
            self.curr_accum_step += 1

    def to(self, device):
        super().to(device)
        self.device = device
        self.simulator.to(device)

        return self

    def _save_code(self):
        code_dir_name = "tensegrity_physics_engine"
        curr_code_dir = os.getcwd()
        code_output = Path(self.output_dir, code_dir_name)
        save_curr_code(curr_code_dir, code_output)

    def _get_end_pts(self, data_jsons):
        with torch.no_grad():
            data_end_pts = []
            for data_json in data_jsons:
                end_pts = [
                    [torch.tensor(e, dtype=self.dtype).reshape(1, 3, 1)
                     for e in d['end_pts']]
                    for d in data_json
                ]
                data_end_pts.append(end_pts)
        return data_end_pts

    @staticmethod
    def _load_json_files(paths):
        data_jsons, extra_state_infos = [], []
        for path in paths:
            with Path(path, "processed_data.json").open('r') as fp:
                data_jsons.append(json.load(fp))

            extra_info_path = Path(path, f"extra_state_data.json")
            if extra_info_path.exists():
                with extra_info_path.open('r') as fp:
                    extra_state_infos.append(json.load(fp))
            else:
                extra_state_infos.append(None)

        return data_jsons, extra_state_infos

    def _data_to_pos_quat_ctrls(self, data_jsons, gt_end_pts, extra_state_jsons):
        data_pos, data_quats, data_controls = [], [], []
        for i, data_json in enumerate(data_jsons):
            pos, quats, controls = [], [], []
            for j, d in enumerate(data_json):
                end_pts = gt_end_pts[i][j]
                end_pts = [(end_pts[2 * k], end_pts[2 * k + 1])
                           for k in range(len(end_pts) // 2)]

                pos.append([(e[1] + e[0]) / 2 for e in end_pts])
                quats.append([
                    torch_quaternion.compute_quat_btwn_z_and_vec(
                        e[1] - e[0]
                    ) for e in end_pts
                ])

            data_pos.append(pos)
            data_quats.append(quats)

        if extra_state_jsons is not None:
            times = [[d['time'] - data_json[0]['time']
                      for d in data_json]
                     for data_json in data_jsons]
            data_controls = self._load_controls(extra_state_jsons, times)

        return data_pos, data_quats, data_controls

    @staticmethod
    def _pos_quat_to_states(data_pos, data_quats, times):
        num_rods = len(data_pos[0][0])

        data_states = []
        for k, (pos, quats, times) in enumerate(zip(data_pos, data_quats, times)):
            states = []
            for i in range(len(pos)):
                prev_i = max(i - 1, 0)
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
                    lin_vels = [torch.zeros_like(pos_0) for j in range(num_rods)]
                    ang_vels = [torch.zeros_like(pos_0) for j in range(num_rods)]

                state = torch.hstack([
                    torch.hstack([pos_1[j], quat_1[j], lin_vels[j], ang_vels[j]])
                    for j in range(num_rods)
                ])
                states.append(state)
            data_states.append(states)

        return data_states

    def _data_json_to_states(self,
                             data_jsons,
                             gt_end_pts,
                             times,
                             extra_state_jsons=None):
        data_pos, data_quats, data_controls = (
            self._data_to_pos_quat_ctrls(
                data_jsons,
                gt_end_pts,
                extra_state_jsons
            )
        )
        data_states = self._pos_quat_to_states(
            data_pos,
            data_quats,
            times
        )

        return data_states, data_controls

    @staticmethod
    def endpts2pos(endpt1, endpt2):
        pos = (endpt2 + endpt1) / 2.0
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(endpt2 - endpt1)

        return pos, quat

    def _load_controls(self, extra_state_jsons, traj_times):
        data_controls = []
        for i, times in enumerate(traj_times):
            traj_start_time = times[0]
            extra_state_start_time = extra_state_jsons[i][0]['time']
            curr_idx = 0
            controls = []

            for j in range(len(times) - 1):
                next_time = times[j + 1] - traj_start_time
                ctrls = []

                for k in range(curr_idx, len(extra_state_jsons[i])):
                    data_time = extra_state_jsons[i][k]['time'] - extra_state_start_time
                    if np.isclose(data_time, next_time):
                        curr_idx = k
                        break
                    ctrl = torch.tensor(extra_state_jsons[i][k]['controls'],
                                        dtype=self.dtype
                                        ).reshape(1, -1, 1)
                    ctrls.append(ctrl)

                control = torch.concat(ctrls, dim=2)
                controls.append(control)

            data_controls.append(controls)

        return data_controls

    @staticmethod
    def compute_com_and_rot_loss(pred_endpts, gt_endpts):
        def com_and_prin(end_pts):
            split_endpts = end_pts.reshape(-1, 6, 1)
            com = (split_endpts[:, 3:] + split_endpts[:, :3]) / 2
            prin = split_endpts[:, 3:] - split_endpts[:, :3]
            prin = prin / prin.norm(dim=1, keepdim=True)

            return com, prin

        pred_com, pred_prin = com_and_prin(pred_endpts)
        gt_com, gt_prin = com_and_prin(gt_endpts)

        loss = ((pred_com - gt_com) ** 2).mean().detach().item()
        rot_loss = torch.clamp(torch.linalg.vecdot(pred_prin, gt_prin, dim=1), -0.9999999, 0.9999999)
        mean_ang_loss = torch.rad2deg(torch.acos(rot_loss)).mean().detach()

        return loss, mean_ang_loss

    def batch_compute_end_pts(self, batch_state: torch.Tensor, sim=None) -> torch.Tensor:
        """
        Compute end pts for entire batch

        :param batch_state: batch of states
        :return: batch of endpts
        """
        if sim is None:
            sim = self.simulator

        end_pts = []
        for i, rod in enumerate(sim.robot.rigid_bodies.values()):
            state = batch_state[:, i * 13: i * 13 + 7]
            prin_axis = torch_quaternion.quat_as_rot_mat(state[:, 3:7])[..., 2:]
            end_pts.extend(Cylinder.compute_end_pts_from_state(state, prin_axis, rod.length))

        return torch.concat(end_pts, dim=1)

    def evaluate_rollouts(self, data_dict):
        n_step_loss, com_n_step_loss, angle_n_step_loss = self.eval_n_step_aheads(
            self.num_eval_n_steps,
            data_dict
        )

        rollout_loss, com_rollout_loss, angle_rollout_loss = (
            self.eval_rollout_w_ctrls(data_dict)
        )

        other_losses = [com_rollout_loss, angle_rollout_loss, com_n_step_loss, angle_n_step_loss]
        return rollout_loss, n_step_loss, other_losses

    def _get_n_step_act_lens_ctrls(self, act_lengths, all_controls, idxs, n):
        if self.use_gt_act_lens:
            gt_act_lens, ctrls = [], None
            for k in range(n):
                end_idx = k + len(all_controls) - n
                act_lens = torch.vstack(act_lengths[k + 1: end_idx + 1])[idxs].to(self.device)
                gt_act_lens.append(act_lens)
            gt_act_lens = torch.concat(gt_act_lens, dim=-1)
        else:
            gt_act_lens, ctrls = None, []
            for k in range(n):
                end_idx = k + len(all_controls) - n
                controls = torch.vstack(all_controls[k:end_idx])[idxs].to(self.device).squeeze(-1)
                ctrls.append(controls)
            ctrls = torch.stack(ctrls, dim=-1)

        return gt_act_lens, ctrls
