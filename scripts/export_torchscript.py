"""Export a trained TensegrityGNNSimulator's GNN as a TorchScript artifact.

Usage:
    python scripts/export_torchscript.py \
        --checkpoint path/to/sim.ckpt \
        --output model.pt

The exported `model.pt` can be loaded in C++ via libtorch:

    auto module = torch::jit::load("model.pt");
    auto out = module.forward({node_x, body_e, body_idx,
                               cable_e, cable_idx,
                               contact_e, contact_idx,
                               node_hidden_state /* or torch::Tensor() */}).toTuple();

Output ordering: (decode_output, cable_decode_output_or_None,
                  new_node_hidden_state_or_None).
"""
import argparse
import sys
from pathlib import Path

import torch

# Make the repo root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_physics.scriptable_gnn import (  # noqa: E402
    ScriptableEncodeProcessDecode,
    unpack_graph,
)
from simulators.tensegrity_gnn_simulator import (  # noqa: E402
    TensegrityGNNSimulator,
    load_simulator,
)


def build_scriptable_from_sim(sim: TensegrityGNNSimulator) -> ScriptableEncodeProcessDecode:
    """Construct a ScriptableEncodeProcessDecode with shapes matching `sim`'s
    EncodeProcessDecode and copy weights into it.
    """
    p = sim.gnn_params if hasattr(sim, 'gnn_params') else None
    # The simulator stashes the raw gnn_params dict as the LearnedSimulator
    # constructor argument; we reconstruct what we need directly.
    inner = sim._encode_process_decode

    # If the simulator was built with use_scriptable_gnn=True, _encode_process_decode
    # is already the GraphData-adapted wrapper.
    if hasattr(inner, 'scriptable'):
        return inner.scriptable

    # Otherwise: construct a fresh ScriptableEncodeProcessDecode using the same
    # shapes and copy weights via load_state_dict (state-dict layout matches).
    encoder_cfg = inner._encoder
    processor_cfg = inner._processor
    decoder_cfg = inner._decoder

    n_out = decoder_cfg.node_decode_fn[-1].out_features
    latent_dim = encoder_cfg.n_out

    nmessage_passing_steps = (
        len(processor_cfg.gnn_stacks)
        if isinstance(processor_cfg.gnn_stacks, torch.nn.ModuleList)
        else processor_cfg.num_msg_passes
    )
    if not isinstance(processor_cfg.gnn_stacks, torch.nn.ModuleList):
        raise NotImplementedError(
            "checkpoint uses processor_shared_weights=True; "
            "scriptable_gnn does not support this configuration"
        )

    # Recover MLP shapes from the encoder's first node MLP.
    node_in_dim = encoder_cfg.node_encoders['node'][0][0].in_features
    body_in_dim = encoder_cfg.edge_encoders['body'][0][0].in_features
    cable_in_dim = encoder_cfg.edge_encoders['cable'][0][0].in_features
    contact_in_dim = encoder_cfg.edge_encoders['contact'][0][0].in_features

    # nmlp_layers: count Linear layers in the inner Sequential, minus output.
    inner_mlp = encoder_cfg.node_encoders['node'][0]
    n_linears = sum(1 for m in inner_mlp if isinstance(m, torch.nn.Linear))
    nmlp_layers = n_linears - 1
    mlp_hidden_dim = next(
        m.out_features for m in inner_mlp if isinstance(m, torch.nn.Linear)
    )

    use_cable_decoder = decoder_cfg.cable_decode_fn is not None

    # Recover recurrent type from the active block, if any.
    if hasattr(encoder_cfg, 'node_recurrent_block'):
        block = encoder_cfg.node_recurrent_block
        if isinstance(block, torch.nn.LSTMCell):
            recurrent_type = 'lstm'
        else:
            recurrent_type = 'mlp'
    else:
        recurrent_type = None

    scriptable = ScriptableEncodeProcessDecode(
        node_types={'node': node_in_dim},
        edge_types={
            'body': body_in_dim,
            'cable': cable_in_dim,
            'contact': contact_in_dim,
        },
        n_out=n_out,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        processor_shared_weights=False,
        recurrent_type=recurrent_type,
        use_cable_decoder=use_cable_decoder,
    )

    # State-dict layout matches; load directly.
    missing, unexpected = scriptable.load_state_dict(inner.state_dict(), strict=False)
    if missing:
        print(f"[export_torchscript] missing keys (zero-initialized): {missing}")
    if unexpected:
        print(f"[export_torchscript] unexpected keys (ignored): {unexpected}")
    return scriptable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    sim = load_simulator(args.checkpoint, map_location=args.device)
    scriptable = build_scriptable_from_sim(sim).to(args.device).eval()

    print("[export_torchscript] running torch.jit.script ...")
    scripted = torch.jit.script(scriptable)
    scripted.save(str(args.output))
    print(f"[export_torchscript] saved {args.output}")

    # Quick sanity: re-load and forward a zero input through both, compare.
    print("[export_torchscript] running parity check ...")
    reload = torch.jit.load(str(args.output), map_location=args.device)
    reload.eval()

    # Synthesize a minimal input matching expected feature widths.
    nodes = sim.robot.num_nodes + 1  # +1 for ground (contact node)
    n_node_in = scriptable._encoder.node_encoders['node'][0][0].in_features
    n_body_in = scriptable._encoder.edge_encoders['body'][0][0].in_features
    n_cable_in = scriptable._encoder.edge_encoders['cable'][0][0].in_features
    n_contact_in = scriptable._encoder.edge_encoders['contact'][0][0].in_features

    g = torch.Generator(device=args.device).manual_seed(0)
    node_x = torch.randn(nodes, n_node_in, device=args.device, generator=g)
    body_e = torch.randn(8, n_body_in, device=args.device, generator=g)
    body_idx = torch.randint(0, nodes, (2, 8), device=args.device, dtype=torch.long, generator=g)
    cable_e = torch.randn(12, n_cable_in, device=args.device, generator=g)
    cable_idx = torch.randint(0, nodes, (2, 12), device=args.device, dtype=torch.long, generator=g)
    contact_e = torch.randn(6, n_contact_in, device=args.device, generator=g)
    contact_idx = torch.randint(0, nodes, (2, 6), device=args.device, dtype=torch.long, generator=g)
    hidden = None

    with torch.no_grad():
        out_eager = scriptable(node_x, body_e, body_idx, cable_e, cable_idx,
                               contact_e, contact_idx, hidden)
        out_scripted = reload(node_x, body_e, body_idx, cable_e, cable_idx,
                              contact_e, contact_idx, hidden)

    for i, (a, b) in enumerate(zip(out_eager, out_scripted)):
        if a is None and b is None:
            continue
        max_diff = (a - b).abs().max().item()
        print(f"  output[{i}] max abs diff = {max_diff:.3e}")
        assert max_diff < 1e-5, f"parity violation at output {i}"
    print("[export_torchscript] parity OK")


if __name__ == '__main__':
    main()
