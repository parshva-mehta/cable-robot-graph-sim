import math
from typing import List, Dict

import torch
import tqdm
from torch_geometric.data import Data as GraphData

from gnn_physics.data_processors.graph_data_processor import GraphDataProcessor
from gnn_physics.gnn import EncodeProcessDecode, RecurrentType
from robots.tensegrity import TensegrityRobotGNN
from simulators.abstract_simulator import LearnedSimulator


class TensegrityGNNSimulator(LearnedSimulator):

    def __init__(
            self,
            gnn_params,
            tensegrity_cfg,
            dt=0.01,
            num_datasets: int = 10,
            num_ctrls_hist: int = 20,
            cache_batch_sizes: List[int] | None = None
    ):
        self.use_cable_decoder = gnn_params['use_cable_decoder']

        self.num_out_steps = gnn_params['n_fwd_pred_steps']
        self.num_ctrls_hist = num_ctrls_hist

        self.ctrls_hist = None
        self.node_hidden_state = None

        robot = TensegrityRobotGNN(tensegrity_cfg)
        data_processor_params = {
            'tensegrity': robot,
            'dt': dt,
            'cache_batch_sizes': cache_batch_sizes,
            'node_hidden_state_size': gnn_params['latent_dim'],
            'num_out_steps': self.num_out_steps,
            'num_ctrls_hist': self.num_ctrls_hist,
            'num_datasets': num_datasets
        }

        if gnn_params['recurrent_type'] == RecurrentType.LSTM.value:
            data_processor_params['node_hidden_state_size'] *= 2

        super().__init__(
            gnn_params,
            data_processor_params
        )

        self.robot = self.data_processor.robot

    def torch_compile(self):
        torch._dynamo.config.cache_size_limit = 512
        self.data_processor.compile(fullgraph=True)
        self._encode_process_decode.compile(fullgraph=True)

    def update_state(self, next_state: torch.Tensor) -> None:
        self.robot.update_state(next_state)

    def reset(self, **kwargs):
        if 'act_lens' in kwargs:
            cables = self.robot.actuated_cables.values()
            for i, cable in enumerate(cables):
                cable.actuation_length = kwargs['act_lens'][:, i: i + 1].clone()

        if 'motor_speeds' in kwargs:
            cables = self.robot.actuated_cables.values()
            for i, cable in enumerate(cables):
                cable.motor.motor_state.omega_t = kwargs['motor_speeds'][:, i: i + 1].clone()

        self.ctrls_hist = kwargs.get('ctrls_hist', None)
        self.node_hidden_state = kwargs.get('node_hidden_state', None)

    def _build_gnn(self, **kwargs):
        return EncodeProcessDecode(
            node_types=kwargs['node_types'],
            edge_types=kwargs['edge_types'],
            n_out=kwargs['n_out'] * kwargs['n_fwd_pred_steps'],
            latent_dim=kwargs['latent_dim'],
            nmessage_passing_steps=kwargs['nmessage_passing_steps'],
            nmlp_layers=kwargs['nmlp_layers'],
            mlp_hidden_dim=kwargs['mlp_hidden_dim'],
            processor_shared_weights=kwargs['processor_shared_weights'],
            recurrent_type=kwargs['recurrent_type'],
            use_cable_decoder=kwargs['use_cable_decoder']
        )

    def _get_data_processor(self):
        return GraphDataProcessor(**self.data_processor_params)

    def _generate_graph(self, state, **kwargs):
        batch_size = state.shape[0]
        if batch_size not in self.data_processor.cached_batch_size_keys:
            self.data_processor.precompute_and_cache_batch_sizes([batch_size])

        graph_feats, raw_feats = self.data_processor(state, **kwargs)
        graph = self.data_processor.feats2graph(graph_feats, raw_feats)
        graph = self._add_hidden_state(graph)

        return graph

    def _process_gnn(self, graph, **kwargs):
        graph = self._encode_process_decode(graph)

        dv_normalizer = self.data_processor.normalizers['node_dv']
        graph['p_dv'] = dv_normalizer.inverse(graph['decode_output'])
        graph['p_dv'] = graph.p_dv.unsqueeze(1).reshape(graph.p_dv.shape[0], -1, 3).transpose(1, 2)
        graph['p_vel'] = graph.vel.unsqueeze(-1) + torch.cumsum(graph.p_dv, dim=-1)
        graph['p_pos'] = graph.pos.unsqueeze(-1) + torch.cumsum(graph.p_vel * self.dt, dim=-1)

        num_steps = kwargs.get('num_steps', self.num_out_steps)

        if self.use_cable_decoder:
            cable_normalizer = self.data_processor.normalizers['cable_dl']
            cable_actuated_mask = graph['cable_actuated_mask'].flatten()
            graph['act_cable_dl'] = cable_normalizer.inverse(graph['cable_decode_output'][cable_actuated_mask])
            mean_act_cable_dl = torch.cumsum(
                graph['act_cable_dl']
                .reshape(-1, 2, self.num_out_steps)
                .mean(dim=1, keepdim=True)
                .reshape(-1, len(self.robot.actuated_cables), self.num_out_steps),
                dim=2
            )
            curr_rest_lens = torch.hstack([c.rest_length for c in self.robot.actuated_cables.values()])
            graph['next_rest_lens'] = curr_rest_lens - mean_act_cable_dl

            for i, c in enumerate(self.robot.actuated_cables.values()):
                c.set_rest_length(graph['next_rest_lens'][:, i: i + 1, num_steps - 1: num_steps])

        return graph

    def _add_hidden_state(self, graph):
        if self.node_hidden_state is not None:
            graph[f'node_hidden_state'] = self.node_hidden_state.clone()

        return graph

    def step(self,
             curr_state: torch.Tensor,
             ctrls: torch.Tensor | None = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs):
        if ctrls is not None:
            if ctrls.shape[-1] < self.num_out_steps:
                step_diff = self.num_out_steps - ctrls.shape[-1]
                pad = torch.zeros_like(ctrls[..., :1]).repeat(1, 1, step_diff)
                ctrls = torch.cat([ctrls, pad], dim=-1)

            if self.ctrls_hist is None:
                self.ctrls_hist = torch.zeros_like(ctrls[..., :1]).repeat(1, 1, self.num_ctrls_hist)

            ctrls = torch.concat([self.ctrls_hist, ctrls], dim=2)
            self.ctrls_hist = ctrls[..., -self.num_ctrls_hist:].clone()

        act_cables = list(self.robot.actuated_cables.values())
        if act_cables[0].rest_length.shape[0] == 1:
            for c in act_cables:
                rest_len = c.rest_length.repeat(ctrls.shape[0], 1, 1)
                c.set_rest_length(rest_len)

        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}

        state_to_graph_kwargs['ctrls'] = ctrls

        self.update_state(curr_state)

        graph = self._generate_graph(curr_state, **state_to_graph_kwargs)
        next_graph = self._process_gnn(graph, **gnn_kwargs)

        body_mask = next_graph.body_mask.flatten()
        next_state = self.data_processor.node2pose(
            next_graph.p_pos[body_mask],
            next_graph.pos[body_mask],
            self.robot.num_nodes_per_rod
        )

        self.node_hidden_state = next_graph['node_hidden_state'].clone()

        return next_state, next_graph

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int = 0,
            state_to_graph_kwargs: List[Dict] | Dict | None = None,
            gnn_kwargs: List[Dict] | Dict | None = None,
            gt_act_lens: torch.Tensor | None = None,
            show_progress=False,
            **kwargs):
        if state_to_graph_kwargs is None:
            state_to_graph_kwargs = {}
        if gnn_kwargs is None:
            gnn_kwargs = {}
        if isinstance(ctrls, list):
            ctrls = torch.cat(ctrls, dim=-1)

        num_steps = ctrls.shape[-1] if ctrls is not None else num_steps
        num_multi_steps = math.ceil(round(num_steps / self.num_out_steps, 5))
        if gt_act_lens is not None:
            assert num_steps == gt_act_lens.shape[-1]

        iterator = range(num_multi_steps)
        if show_progress:
            iterator = tqdm.tqdm(iterator)

        states, graphs, rest_lens = [], [], []
        for i in iterator:
            start = i * self.num_out_steps
            n = min(num_steps - start, self.num_out_steps)
            end = start + n

            s2g_dict = state_to_graph_kwargs[i] \
                if isinstance(state_to_graph_kwargs, list) \
                else state_to_graph_kwargs
            gnn_dict = gnn_kwargs[i] \
                if isinstance(gnn_kwargs, list) \
                else gnn_kwargs
            gnn_dict['num_steps'] = n

            if gt_act_lens is not None:
                cables = self.robot.actuated_cables.values()
                cable_init_rest_len = torch.hstack([c._rest_length for c in cables])
                for j, cable in enumerate(cables):
                    cable.actuation_length = gt_act_lens[:, j: j + 1, end - 1: end]
                self.data_processor.robot.gnn_rest_lens = cable_init_rest_len - gt_act_lens[..., start:end]
                controls = None
            else:
                controls = ctrls[..., start:end] if ctrls is not None else None
                step_diff = self.num_out_steps - controls.shape[-1]
                if step_diff > 0:
                    pad = torch.zeros_like(controls[..., :1]).repeat(1, 1, step_diff)
                    controls = torch.cat([controls, pad], dim=-1)

            curr_state, graph = self.step(
                curr_state[..., -1:],
                ctrls=controls,
                state_to_graph_kwargs=s2g_dict,
                gnn_kwargs=gnn_dict,
                **kwargs
            )

            graphs.append(graph.clone())
            states.extend([
                curr_state[..., i: i + 1].clone()
                for i in range(controls.shape[-1])
            ])
            rest_lens.extend([
                graph['next_rest_lens'][..., i: i + 1].clone()
                for i in range(controls.shape[-1])
            ])

        return states[:num_steps], graphs, rest_lens[:num_steps]
