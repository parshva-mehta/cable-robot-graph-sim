"""TorchScript-compatible variant of EncodeProcessDecode.

Refactored from gnn_physics/gnn.py to remove three TorchScript blockers:
  1. torch_geometric.data.Data passed end-to-end -> explicit tensor I/O.
  2. nn.ParameterDict + dynamic string-key dispatch -> nn.ModuleDict with
     literal-key access (and fully unrolled per-edge-type calls).
  3. self.recur_fwd_fn = self.<method> bound-method dispatch -> Final[int]
     discriminator with static if/elif branches that TorchScript dead-code-
     eliminates.

PyG MessagePassing is replaced with a hand-rolled scatter using
torch.Tensor.index_add_ so libtorch builds do not need torch_scatter / PyG.

Edge types are fixed to ('body', 'cable', 'contact') and there is a single
node type 'node'. State-dict structure matches gnn.EncodeProcessDecode so
trained checkpoints can be loaded directly.

Limitations vs gnn.py (raise at construction time):
  - processor_shared_weights=True is not supported.
  - recurrent_type GRU/RNN is not supported (the original wired an
    nn.Sequential([cell, LayerNorm]) and called it with two positional args,
    which already errors in stock PyTorch; that path is dead in gnn.py and
    not worth replicating here).
"""
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn


def _check_no_recurrent(name: Optional[str]) -> None:
    """The scriptable variant only supports recurrent_type=None.

    TorchScript requires every attribute referenced in `forward` to exist on
    the module at script time, even inside dead branches guarded by Final[int]
    discriminators -- attribute resolution happens before constant folding.
    Supporting recurrent_type={mlp,lstm} would therefore force registering
    placeholder submodules that would either pollute the state_dict or
    type-mismatch the active call sites.

    Production training (3_bar_gnn_sim_config.json) uses recurrent_type=null,
    so this restriction is not blocking. Scripting an LSTM/MLP variant should
    be done in a separate subclass with its own structured state_dict.
    """
    if name is not None:
        raise NotImplementedError(
            f"scriptable_gnn does not support recurrent_type={name!r}; "
            f"only recurrent_type=None is currently scriptable. "
            f"Train without recurrence or extend ScriptableEncoder with a "
            f"per-mode subclass."
        )


def build_mlp(input_size: int,
              hidden_layer_sizes: List[int],
              output_size: Optional[int] = None,
              output_activation: type = nn.Identity,
              activation: type = nn.ReLU,
              dropout: Optional[float] = None) -> nn.Sequential:
    """Same as gnn.build_mlp but with TorchScript-friendly defaults."""
    layer_sizes = [input_size] + list(hidden_layer_sizes)
    if output_size is not None:
        layer_sizes.append(output_size)

    nlayers = len(layer_sizes) - 1
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation

    mlp = nn.Sequential()
    for i in range(nlayers):
        if i > 0 and dropout is not None:
            mlp.add_module(f'Dropout-{i}', nn.Dropout(dropout))
        mlp.add_module(f"NN-{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module(f"Act-{i}", act[i]())
    return mlp


def _mlp_block(in_feats: int,
               out_feats: int,
               num_layers: int,
               mlp_hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        build_mlp(in_feats,
                  [mlp_hidden_dim for _ in range(num_layers)],
                  out_feats),
        nn.LayerNorm(out_feats),
    )


class ScriptableEncoder(nn.Module):
    def __init__(self,
                 n_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 node_types: Dict[str, int],
                 edge_types: Dict[str, int],
                 recurrent_type: Optional[str] = None):
        super().__init__()
        _check_no_recurrent(recurrent_type)
        self.n_out = n_out

        # Encoders (ModuleDict, NOT ParameterDict): literal-key access scripts.
        self.node_encoders = nn.ModuleDict({
            name: _mlp_block(in_dim, n_out, nmlp_layers, mlp_hidden_dim)
            for name, in_dim in node_types.items()
        })
        self.edge_encoders = nn.ModuleDict({
            name: _mlp_block(in_dim, n_out, nmlp_layers, mlp_hidden_dim)
            for name, in_dim in edge_types.items()
        })

        # LayerNorm always present (matches gnn.Encoder state-dict layout).
        self.node_recur_layer_norm = nn.LayerNorm(n_out)

    def forward(self,
                node_x: torch.Tensor,
                body_edge_attr: torch.Tensor,
                cable_edge_attr: torch.Tensor,
                contact_edge_attr: torch.Tensor,
                node_hidden_state: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor,
                           torch.Tensor,
                           torch.Tensor,
                           torch.Tensor,
                           Optional[torch.Tensor]]:
        x = self.node_encoders['node'](node_x)
        body_e = self.edge_encoders['body'](body_edge_attr)
        cable_e = self.edge_encoders['cable'](cable_edge_attr)
        contact_e = self.edge_encoders['contact'](contact_edge_attr)
        # Hidden state is unused when recurrent_type=None; pass through so the
        # output signature matches the recurrent-supporting plan and the
        # caller does not have to special-case None vs. tensor.
        return x, body_e, cable_e, contact_e, node_hidden_state


class EdgeMessagePassing(nn.Module):
    """Replacement for gnn.BaseInteractionNetwork that does not depend on
    torch_geometric.nn.MessagePassing.

    Computes:
        msg = msg_fn(concat([x[dst], x[src], edge_attr]))
        edge_attr_new = edge_attr + msg
        agg[i] = sum_{e: dst(e)=i} edge_attr_new[e]    (index_add_)

    State-dict layout (`msg_fn.0.NN-*`) matches gnn.BaseInteractionNetwork so
    checkpoints transfer.
    """

    def __init__(self,
                 nnode_in: int,
                 nedge_in: int,
                 n_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int):
        super().__init__()
        self.msg_fn = nn.Sequential(
            build_mlp(nnode_in + nnode_in + nedge_in,
                      [mlp_hidden_dim for _ in range(nmlp_layers)],
                      n_out),
            nn.LayerNorm(n_out),
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # PyG default flow is source_to_target: x_j = x[edge_index[0]]
        # (source), x_i = x[edge_index[1]] (target). Aggregation happens at
        # target index.
        src = edge_index[0]
        dst = edge_index[1]

        x_j = x.index_select(0, src)
        x_i = x.index_select(0, dst)

        concat_vec = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.msg_fn(concat_vec)
        edge_attr_new = edge_attr + msg

        agg = torch.zeros(
            (x.shape[0], edge_attr_new.shape[-1]),
            dtype=edge_attr_new.dtype,
            device=edge_attr_new.device,
        )
        agg = agg.index_add(0, dst, edge_attr_new)
        return agg, edge_attr_new


class ScriptableInteractionNetwork(nn.Module):
    def __init__(self,
                 nnode_in: int,
                 nedge_in: int,
                 n_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 edge_types: List[str]):
        super().__init__()
        self.update_fn = nn.Sequential(
            build_mlp(nnode_in + n_out * len(edge_types),
                      [mlp_hidden_dim for _ in range(nmlp_layers)],
                      n_out),
            nn.LayerNorm(n_out),
        )
        self.mp_dict = nn.ModuleDict({
            name: EdgeMessagePassing(nnode_in, nedge_in, n_out,
                                     nmlp_layers, mlp_hidden_dim)
            for name in edge_types
        })

    def forward(self,
                x: torch.Tensor,
                body_edge_attr: torch.Tensor,
                body_edge_index: torch.Tensor,
                cable_edge_attr: torch.Tensor,
                cable_edge_index: torch.Tensor,
                contact_edge_attr: torch.Tensor,
                contact_edge_index: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        agg_body, body_e = self.mp_dict['body'](x, body_edge_index, body_edge_attr)
        agg_cable, cable_e = self.mp_dict['cable'](x, cable_edge_index, cable_edge_attr)
        agg_contact, contact_e = self.mp_dict['contact'](x, contact_edge_index, contact_edge_attr)

        agg = torch.cat([agg_body, agg_cable, agg_contact], dim=-1)
        concat = torch.cat([x, agg], dim=-1)
        x_new = x + self.update_fn(concat)
        return x_new, body_e, cable_e, contact_e


class ScriptableProcessor(nn.Module):
    num_msg_passes: torch.jit.Final[int]

    def __init__(self,
                 nnode_in: int,
                 nedge_in: int,
                 n_out: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 edge_types: List[str],
                 processor_shared_weights: bool = False):
        super().__init__()
        if processor_shared_weights:
            raise NotImplementedError(
                "processor_shared_weights=True not supported in scriptable_gnn"
            )
        self.num_msg_passes = nmessage_passing_steps
        self.gnn_stacks = nn.ModuleList([
            ScriptableInteractionNetwork(
                nnode_in=nnode_in,
                nedge_in=nedge_in,
                n_out=n_out,
                nmlp_layers=nmlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                edge_types=edge_types,
            )
            for _ in range(nmessage_passing_steps)
        ])

    def forward(self,
                x: torch.Tensor,
                body_edge_attr: torch.Tensor,
                body_edge_index: torch.Tensor,
                cable_edge_attr: torch.Tensor,
                cable_edge_index: torch.Tensor,
                contact_edge_attr: torch.Tensor,
                contact_edge_index: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_e = body_edge_attr
        cable_e = cable_edge_attr
        contact_e = contact_edge_attr
        for layer in self.gnn_stacks:
            x, body_e, cable_e, contact_e = layer(
                x,
                body_e, body_edge_index,
                cable_e, cable_edge_index,
                contact_e, contact_edge_index,
            )
        return x, body_e, cable_e, contact_e


class ScriptableDecoder(nn.Module):
    use_cable_decoder: torch.jit.Final[bool]

    def __init__(self,
                 nnode_in: int,
                 nnode_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 use_cable_decoder: bool):
        super().__init__()
        self.use_cable_decoder = use_cable_decoder
        self.node_decode_fn = build_mlp(
            nnode_in,
            [mlp_hidden_dim for _ in range(nmlp_layers)],
            nnode_out,
        )
        if use_cable_decoder:
            assert nnode_out % 3 == 0
            n_cable_out = nnode_out // 3
            self.cable_decode_fn = build_mlp(
                nnode_in,
                [mlp_hidden_dim for _ in range(nmlp_layers)],
                n_cable_out,
            )

    def forward(self,
                x: torch.Tensor,
                cable_edge_attr: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        decode_output = self.node_decode_fn(x)
        cable_decode_output: Optional[torch.Tensor] = None
        if self.use_cable_decoder:
            cable_decode_output = self.cable_decode_fn(cable_edge_attr)
        return decode_output, cable_decode_output


class ScriptableEncodeProcessDecode(nn.Module):
    """Drop-in scriptable replacement for gnn.EncodeProcessDecode.

    Forward takes 8 named tensors (instead of a GraphData) and returns
    (decode_output, cable_decode_output, new_node_hidden_state). The
    submodule names match gnn.EncodeProcessDecode (`_encoder`, `_processor`,
    `_decoder`) so a state_dict trained against gnn.EncodeProcessDecode
    transfers via load_state_dict.
    """

    def __init__(self,
                 node_types: Dict[str, int],
                 edge_types: Dict[str, int],
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights: bool = False,
                 recurrent_type: Optional[str] = None,
                 use_cable_decoder: bool = False):
        super().__init__()
        self._encoder = ScriptableEncoder(
            n_out=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_types=node_types,
            edge_types=edge_types,
            recurrent_type=recurrent_type,
        )
        self._processor = ScriptableProcessor(
            nnode_in=latent_dim,
            nedge_in=latent_dim,
            n_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            processor_shared_weights=processor_shared_weights,
            edge_types=list(edge_types.keys()),
        )
        self._decoder = ScriptableDecoder(
            nnode_in=latent_dim,
            nnode_out=n_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            use_cable_decoder=use_cable_decoder,
        )

    def forward(self,
                node_x: torch.Tensor,
                body_edge_attr: torch.Tensor,
                body_edge_index: torch.Tensor,
                cable_edge_attr: torch.Tensor,
                cable_edge_index: torch.Tensor,
                contact_edge_attr: torch.Tensor,
                contact_edge_index: torch.Tensor,
                node_hidden_state: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor,
                           Optional[torch.Tensor],
                           Optional[torch.Tensor]]:
        x, body_e, cable_e, contact_e, new_hidden = self._encoder(
            node_x, body_edge_attr, cable_edge_attr, contact_edge_attr,
            node_hidden_state,
        )
        x, body_e, cable_e, contact_e = self._processor(
            x,
            body_e, body_edge_index,
            cable_e, cable_edge_index,
            contact_e, contact_edge_index,
        )
        decode_output, cable_decode_output = self._decoder(x, cable_e)
        return decode_output, cable_decode_output, new_hidden


# ---------------------------------------------------------------------------
# Adapter: pull the named tensors out of a torch_geometric Data object so the
# scriptable model can be called from Python code that still produces
# GraphData (e.g. TensegrityGNNSimulator). Importing here is local-only since
# torch_geometric.data is not part of the scripted forward path.
# ---------------------------------------------------------------------------

def unpack_graph(graph) -> Tuple[torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor,
                                 Optional[torch.Tensor]]:
    """Extract the eight tensors needed by ScriptableEncodeProcessDecode.forward
    from a torch_geometric.data.Data instance.
    """
    hidden = graph['node_hidden_state'] if 'node_hidden_state' in graph else None
    return (
        graph['node_x'],
        graph['body_edge_attr'],
        graph['body_edge_index'],
        graph['cable_edge_attr'],
        graph['cable_edge_index'],
        graph['contact_edge_attr'],
        graph['contact_edge_index'],
        hidden,
    )
