from enum import Enum
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import MessagePassing


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU,
        dropout: Optional[float] = None) -> nn.Module:
    """
    Method to build MLP

    @param input_size: size of input
    @param hidden_layer_sizes: list of hidden layer sizes
    @param output_size: size of output
    @param output_activation: activation function specific for output
    @param activation: activation for all other layers
    @return: mlp object
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        if i > 0 and dropout is not None:
            mlp.add_module(f'Dropout-{i}', nn.Dropout(dropout))

        mlp.add_module(f"NN-{i}", nn.Linear(layer_sizes[i],
                                            layer_sizes[i + 1]))
        mlp.add_module(f"Act-{i}", act[i]())

    return mlp


class RecurrentType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    RNN = 'rnn'
    MLP = 'mlp'


class Encoder(nn.Module):

    def __init__(self,
                 n_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 node_types: Dict[str, int],
                 edge_types: Dict[str, int],
                 recurrent_type: RecurrentType | None = None,
                 ):
        """
        Encoder to take features to a higher dimensional space

        @param n_out: output size of encoder
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: dim of hidden layers in mlp
        @param node_types: dictionary of node types and their input sizes
        @param edge_types: dictionary of edge types and their input sizes
        """

        def mlp(in_feats, out_feats, num_layers=nmlp_layers):
            """
            method to quickly augment mlp with LayerNorm as last layer
            @param in_feats:
            @return:
            """
            return nn.Sequential(
                *[build_mlp(in_feats,
                            [mlp_hidden_dim
                             for _ in range(num_layers)],
                            out_feats),
                  nn.LayerNorm(out_feats)]
            )

        super().__init__()
        self.n_out = n_out
        self.enum_node_types = {n: i for i, n in enumerate(node_types.keys())}
        self.enum_edge_types = {n: i for i, n in enumerate(edge_types.keys())}

        self.node_encoders = nn.ParameterDict({
            name: mlp(in_dim, n_out) for name, in_dim in node_types.items()
        })

        self.edge_encoders = nn.ParameterDict({
            name: mlp(in_dim, n_out) for name, in_dim in edge_types.items()
        })

        self.recurrent_type = RecurrentType(recurrent_type) if recurrent_type else None

        recurr_block_fn = lambda cell: nn.Sequential(cell, nn.LayerNorm(n_out))
        self.node_recur_layer_norm = nn.LayerNorm(n_out)

        if self.recurrent_type == RecurrentType.MLP:
            self.node_recurrent_block = mlp(2 * n_out, n_out, 2)
            self.cable_recurrent_block = mlp(2 * n_out, n_out, 2)
            self.recur_fwd_fn = self.mlp_forward
        elif self.recurrent_type == RecurrentType.GRU:
            self.node_recurrent_block = recurr_block_fn(nn.RNNCell(n_out, n_out))
            self.cable_recurrent_block = recurr_block_fn(nn.RNNCell(n_out, n_out))
            self.recur_fwd_fn = self.rnn_gru_forward
        elif self.recurrent_type == RecurrentType.LSTM:
            self.node_recurrent_block = nn.LSTMCell(n_out, n_out)
            self.node_lstm_layer_norm = nn.LayerNorm(n_out)
            self.recur_fwd_fn = self.lstm_forward
        elif self.recurrent_type == RecurrentType.RNN:
            self.node_recurrent_block = recurr_block_fn(nn.GRUCell(n_out, n_out))
            self.cable_recurrent_block = recurr_block_fn(nn.GRUCell(n_out, n_out))
            self.recur_fwd_fn = self.rnn_gru_forward
        else:
            raise Exception("recurrent_type not valid")

    def mlp_forward(self, graph_batch: GraphData) -> GraphData:
        hidden_state = graph_batch[f'node_hidden_state']

        graph_batch['x'] = self.node_recurrent_block(torch.hstack([
            graph_batch['x'],
            hidden_state
        ]))
        graph_batch['x'] = self.node_recur_layer_norm(graph_batch['x'])

        graph_batch[f'node_hidden_state'] = graph_batch['x'].clone()

        return graph_batch

    def rnn_gru_forward(self, graph_batch: GraphData):
        hidden_state = graph_batch[f'node_hidden_state']
        graph_batch['x'] = self.node_recurrent_block(
            graph_batch['x'],
            hidden_state
        )
        graph_batch['x'] = self.node_recur_layer_norm(graph_batch['x'])
        graph_batch[f'node_hidden_state'] = graph_batch['x'].clone()

        return graph_batch

    def lstm_forward(self, graph_batch: GraphData):
        hidden_state = graph_batch[f'node_hidden_state'][:, :graph_batch['x'].shape[1]]
        memory = graph_batch[f'node_hidden_state'][:, graph_batch['x'].shape[1]:]
        graph_batch['x'], memory = self.node_recurrent_block(
            graph_batch['x'],
            (hidden_state, memory)
        )
        graph_batch['x'] = self.node_lstm_layer_norm(graph_batch['x'])

        graph_batch[f'node_hidden_state'] = torch.hstack([
            graph_batch['x'].clone(),
            memory.clone()
        ])

        return graph_batch

    def forward(self, graph: GraphData) -> GraphData:
        # Encode node features
        encoded = self.node_encoders['node'](graph['node_x'])
        graph['x'] = encoded

        # Encode edge features
        for edge_name, n in self.enum_edge_types.items():
            encoded = self.edge_encoders[edge_name](graph[edge_name + "_edge_attr"])
            graph[edge_name + "_edge_attr"] = encoded

        if self.recur_fwd_fn is not None:
            graph = self.recur_fwd_fn(graph)

        return graph


class BaseInteractionNetwork(MessagePassing):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            agg_method: str = 'add'
    ):
        """
        Base interaction network that does message passing on the lower level

        @param nnode_in: input node feat size
        @param nedge_in: input edge feat size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        """
        # Aggregate features from neighbors
        super().__init__(aggr=agg_method)
        # Edge MLP
        self.msg_fn = nn.Sequential(
            *[build_mlp(nnode_in + nnode_in + nedge_in,
                        [mlp_hidden_dim
                         for _ in range(nmlp_layers)],
                        n_out),
              nn.LayerNorm(n_out)]
        )
        self.tmp_edge_attr = None

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of model

        @param x: node feats
        @param edge_index: edge indices
        @param edge_attr: edge feats
        @return: updated node feats
        """
        x_prop, edge_prop = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr
        )

        return x_prop, self.tmp_edge_attr

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor
                ) -> torch.Tensor:
        """
        Message passing step
        @param x_i: node feats of one end of edges
        @param x_j: node feats of other end of edges
        @param edge_attr: edge feats
        @return: updated edges
        """
        concat_vec = torch.hstack([x_i, x_j, edge_attr])
        msg = self.msg_fn(concat_vec)
        edge_attr = edge_attr + msg

        self.tmp_edge_attr = edge_attr

        return edge_attr

    def update(self,
               x_updated: torch.Tensor,
               x: torch.Tensor,
               edge_attr: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Do nothing
        """
        return x_updated, edge_attr


class InteractionNetwork(nn.Module):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            edge_types: List[str],
            agg_method: str = "add"
    ):
        """
        Interaction network that does message passing on the higher level

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        @param edge_types: edge types in graph
        """
        # Aggregate features from neighbors
        super().__init__()

        # Node MLP
        self.update_fn = nn.Sequential(*[
            build_mlp(nnode_in + n_out * len(edge_types),
                      [mlp_hidden_dim for _ in range(nmlp_layers)],
                      n_out),
            nn.LayerNorm(n_out)
        ])

        # Edge message passing networks
        kwargs = {
            "nnode_in": nnode_in,
            "nedge_in": nedge_in,
            "n_out": n_out,
            "nmlp_layers": nmlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            'agg_method': agg_method
        }
        self.mp_dict = nn.ParameterDict({
            name: BaseInteractionNetwork(**kwargs)
            for name in edge_types
        })

    def forward(self, graph: GraphData) -> GraphData:
        """
        forward pass

        @param graph: batch graph data
        @return: updated graph
        """
        agg_xs = []
        for name, mp in self.mp_dict.items():
            edge_attr = graph[name + "_edge_attr"]
            edge_idx = graph[name + "_edge_index"]

            agg_x, updated_edge = mp(graph['x'], edge_idx, edge_attr)
            graph[name + "_edge_attr"] = updated_edge

            agg_xs.append(agg_x)

        agg_xs = torch.hstack(agg_xs)
        concat_vec = torch.hstack([graph['x'], agg_xs])
        graph['x'] = graph['x'] + self.update_fn(concat_vec)

        return graph


class Processor(nn.Module):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            edge_types: List[str],
            processor_shared_weights: bool = False,
    ):
        """
        Processor containing multiple interaction networks

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        @param edge_types: edge types in graph
        @param processor_shared_weights: flag to have shared or sep weights per interaction network
        """

        kwargs = {
            "nnode_in": nnode_in,
            "nedge_in": nedge_in,
            "n_out": n_out,
            "nmlp_layers": nmlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "edge_types": edge_types
        }

        super().__init__()
        self.num_msg_passes = nmessage_passing_steps
        self.shared_weights = processor_shared_weights
        # Create a stack of L Graph Networks GNs.
        if self.shared_weights:
            self.gnn_stacks = self._get_interaction_network(**kwargs)
        else:
            self.gnn_stacks = nn.ModuleList([
                self._get_interaction_network(**kwargs)
                for _ in range(nmessage_passing_steps)
            ])

        # self.gnn_stacks.compile()

    def _get_interaction_network(self, **kwargs):
        return InteractionNetwork(**kwargs)

    def forward(self, graph_batch: GraphData) -> GraphData:
        """
        Forward pass

        @param graph_batch: graph data
        @return: message passed graph
        """
        for i in range(self.num_msg_passes):
            graph_batch = self.gnn_stacks(graph_batch) \
                if self.shared_weights \
                else self.gnn_stacks[i](graph_batch)

        return graph_batch


class Decoder(nn.Module):
    def __init__(
            self,
            nnode_in: int,
            nnode_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            use_cable_decoder: bool,
    ):
        """
        Decoder that maps node latent vectors to normalized dv

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        """
        super().__init__()
        self.node_decode_fn = build_mlp(
            nnode_in,
            [mlp_hidden_dim for _ in range(nmlp_layers)],
            nnode_out
        )

        self.cable_decode_fn = None
        if use_cable_decoder:
            assert nnode_out % 3 == 0
            n_cable_out = nnode_out // 3
            self.cable_decode_fn = build_mlp(
                nnode_in,
                [mlp_hidden_dim for _ in range(nmlp_layers)],
                n_cable_out
            )

    def forward(self, graph_batch: GraphData) -> GraphData:
        """
        forward pass

        @param graph_batch: graph data
        @return: decoded graph data
        """
        graph_batch['decode_output'] = self.node_decode_fn(graph_batch['x'])
        if self.cable_decode_fn is not None:
            graph_batch['cable_decode_output'] = (
                self.cable_decode_fn(graph_batch['cable_edge_attr'])
            )

        return graph_batch


class EncodeProcessDecode(nn.Module):

    def __init__(
            self,
            node_types: Dict[str, int],
            edge_types: Dict[str, int],
            n_out: int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            processor_shared_weights: bool = False,
            recurrent_type: RecurrentType | None = None,
            use_cable_decoder: bool = False,
    ):
        """
        GNN class

        @param node_types: dictionary of node types to input vec size
        @param edge_types: dictionary of edge types to input vec size
        @param n_out: output size
        @param latent_dim: latent dim size
        @param nmessage_passing_steps: number of msg passing steps
        @param nmlp_layers: number of mlp layers per mlp
        @param mlp_hidden_dim: hidden dim size
        @param processor_shared_weights: flag for shared or sep weights for interaction networks
        """
        super().__init__()
        self._encoder = Encoder(
            n_out=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_types=node_types,
            edge_types=edge_types,
            recurrent_type=recurrent_type,
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nedge_in=latent_dim,
            n_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            processor_shared_weights=processor_shared_weights,
            edge_types=list(edge_types.keys())
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=n_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            use_cable_decoder=use_cable_decoder
        )

    def to(self, device):
        self._encoder = self._encoder.to(device)
        self._processor = self._processor.to(device)
        self._decoder = self._decoder.to(device)

        return self

    def forward(self,
                graph_batch: GraphData):
        """
        forward pass

        @param graph_batch: graph data
        @return: GNN processed graph data
        """
        graph_batch = self._encoder(graph_batch)
        graph_batch = self._processor(graph_batch)
        graph_batch = self._decoder(graph_batch)

        return graph_batch
