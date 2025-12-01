from typing import Tuple, Union, Optional, List, Dict

import torch
import tqdm

from robots.tensegrity import TensegrityRobot
from state_objects.base_state_object import BaseStateObject


class AbstractSimulator(BaseStateObject):

    def __init__(self):
        super().__init__('simulator')

    def get_curr_state(self) -> torch.Tensor:
        """
        Method to get current state
        :return: Current state tensor
        """
        raise NotImplementedError()

    def update_state(self, next_state: torch.Tensor) -> None:
        """
        Method to update internal state values
        :param next_state: Next state
        """
        raise NotImplementedError()

    def apply_control(self, control_signals, dt) -> None:
        raise NotImplementedError()

    def step(self,
             curr_state: torch.Tensor,
             dt: torch.Tensor | float,
             ctrls: List[torch.Tensor] | torch.Tensor | None,
             **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def run(self, **kwargs):
        raise NotImplementedError()


class LearnedSimulator(AbstractSimulator):

    def __init__(
            self,
            gnn_params: Dict,
            data_processor_params: Dict,
    ):

        super().__init__()
        self.data_processor_params = data_processor_params
        self.data_processor = self._get_data_processor()

        self.node_types = {k: sum(v.values()) for k, v in self.data_processor.hier_node_feat_dict.items()}
        self.edge_types = {k: sum(v.values()) for k, v in self.data_processor.hier_edge_feat_dict.items()}
        self.dt = self.data_processor.dt

        # Initialize the GNN
        self._encode_process_decode = self._build_gnn(
            node_types=self.node_types,
            edge_types=self.edge_types,
            **gnn_params,
        )

    def reset(self, **kwargs):
        pass

    def _get_data_processor(self):
        raise NotImplementedError()

    def _build_gnn(self, **kwargs):
        raise NotImplementedError()

    def to(self, device):
        super().to(device)
        self.dt = self.dt.to(device)
        self._encode_process_decode = self._encode_process_decode.to(device)
        self.data_processor = self.data_processor.to(device)

        return self

    def forward(self,
                curr_state: torch.Tensor,
                ctrls: Optional[torch.Tensor] = None,
                state_to_graph_kwargs: Dict | None = None,
                gnn_kwargs: Dict | None = None,
                **kwargs):
        return self.step(curr_state,
                         ctrls,
                         state_to_graph_kwargs,
                         gnn_kwargs,
                         **kwargs)

    def process_gnn(self, graph, **kwargs):
        raise NotImplementedError()

    def step(self,
             curr_state: torch.Tensor,
             ctrls: Optional[torch.Tensor] = None,
             state_to_graph_kwargs: Dict | None = None,
             gnn_kwargs: Dict | None = None,
             **kwargs):
        raise NotImplementedError()

    def apply_controls(self, ctrls):
        raise NotImplementedError

    def run(self,
            curr_state: torch.Tensor,
            ctrls: List[torch.Tensor] | torch.Tensor | None,
            num_steps: int | None = None,
            **kwargs):
        raise NotImplementedError