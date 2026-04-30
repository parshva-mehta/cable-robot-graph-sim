# TorchScript Refactor Plan: Porting `EncodeProcessDecode` to C++

Goal: make `gnn_physics/gnn.py` scriptable via `torch.jit.script()` so the model can be loaded and executed in C++ via `libtorch`. Edge types are fixed at three (`body`, `cable`, `contact`) and there is a single node type, which makes static specialization tractable.

## Scope

- In scope: `gnn_physics/gnn.py` (model), a thin TorchScript-friendly wrapper, and a Python-side adapter that flattens `GraphData` into plain tensors before calling the model.
- Out of scope: `GraphDataProcessor` itself stays in Python for training. We only need a C++-callable forward pass; feature construction can run in C++ separately or be ported later.

## Phase 1 — Replace `GraphData` with explicit tensor I/O

The model currently passes a `torch_geometric.data.Data` object end-to-end and mutates string-keyed entries. Replace with positional tensor arguments.

- Change every `forward(graph)` signature in `gnn.py` (`Encoder`, `InteractionNetwork`, `Processor`, `Decoder`, `EncodeProcessDecode`) to take/return a fixed tuple:
  - Inputs: `node_x`, `body_edge_attr`, `cable_edge_attr`, `contact_edge_attr`, `body_edge_index`, `cable_edge_index`, `contact_edge_index`, `node_hidden_state` (`Optional[Tensor]`).
  - Outputs: `decode_output`, `cable_decode_output` (`Optional[Tensor]`), `node_hidden_state` (`Optional[Tensor]`).
- Drop all `graph[...]` subscripting. Use named local tensors.
- Type-annotate every argument and return with `torch.Tensor` / `Optional[torch.Tensor]` so the scripter has no inference work to do.

## Phase 2 — Eliminate dynamic `ParameterDict` dispatch

`Encoder.node_encoders`, `Encoder.edge_encoders`, and `InteractionNetwork.mp_dict` are indexed by string at runtime via a `for edge_name, n in self.enum_edge_types.items()` loop (`gnn.py:179`, `gnn.py:328`). TorchScript cannot compile this.

- Replace `nn.ParameterDict` (which is also misused — these are `nn.Module`s, not parameters; should be `nn.ModuleDict`) with explicit named submodules:
  - `self.node_encoder` (single, since only one node type).
  - `self.body_edge_encoder`, `self.cable_edge_encoder`, `self.contact_edge_encoder`.
  - In `InteractionNetwork`: `self.body_mp`, `self.cable_mp`, `self.contact_mp`.
- Unroll the encode/message-passing loops into three explicit calls (one per edge type) and `torch.hstack` the three results.
- This sacrifices the "arbitrary edge types" generality, but the codebase only ever uses these three. Keep a Python-only generic version (`EncodeProcessDecodeGeneric`) for any future experimentation; the scriptable `EncodeProcessDecode` becomes the production path.

## Phase 3 — Make recurrent dispatch static

`self.recur_fwd_fn = self.lstm_forward` (`gnn.py:115–129`) stores a bound method as an instance attribute — not scriptable.

- Replace with an `int` discriminator stored as a buffer or a `Final[int]` class attribute set in `__init__`:
  - `0 = none, 1 = mlp, 2 = gru, 3 = lstm, 4 = rnn`.
- In `forward`, branch with `if self.recur_mode == 3: ...` (TorchScript handles static `if` chains fine).
- Always instantiate every variant's submodules conditionally; unused variants simply aren't constructed. Mark the discriminator `Final` so TorchScript dead-code-eliminates the inactive branches.
- Drop the `RecurrentType` enum from the scriptable model (TorchScript supports enums but the `RecurrentType(recurrent_type) if recurrent_type else None` coercion at `gnn.py:107` is awkward); convert at the config layer instead.

## Phase 4 — Replace `MessagePassing` with explicit scatter

`BaseInteractionNetwork` inherits from `torch_geometric.nn.MessagePassing` and uses `self.propagate(...)`. PyG's `propagate` does runtime introspection on `message`/`update` signatures (it inspects argument names like `x_i`, `x_j`) — not scriptable.

- Reimplement message passing directly:
  ```python
  src, dst = edge_index[0], edge_index[1]
  msg_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
  msg = self.msg_fn(msg_in)
  edge_attr_new = edge_attr + msg
  agg_x = torch.zeros_like(x).index_add_(0, dst, edge_attr_new)
  ```
- Use `index_add_` (or `scatter_add_`) for the `add` aggregation. If other aggregators are needed later, add them as explicit branches.
- Remove the `self.tmp_edge_attr` side-channel (`gnn.py:219, 258`) — it exists only because PyG's `propagate` separates `message` from the return value. With direct scatter, return `(agg_x, edge_attr_new)` cleanly.

## Phase 5 — Minor scripter-friendliness fixes

- `build_mlp` (`gnn.py:10`): `output_size: int = None` → `Optional[int] = None`, and replace `if output_size:` with `if output_size is not None:`. F-string layer names (`f"NN-{i}"`) are fine.
- Remove the lambda `recurr_block_fn` at `gnn.py:109`; inline `nn.Sequential(cell, nn.LayerNorm(n_out))`.
- The custom `EncodeProcessDecode.to(self, device)` override (`gnn.py:514`) shadows `nn.Module.to` and is unnecessary — child modules are already registered, so `nn.Module.to` handles them. Delete it.
- Replace `recurrent_type: RecurrentType | None` (PEP 604 syntax) with `Optional[RecurrentType]` — older TorchScript versions don't parse `|` unions.
- Replace `torch.hstack` with `torch.cat(..., dim=-1)` (more reliably scriptable across versions).

## Phase 6 — Adapter + export

- Add `gnn_physics/scriptable_gnn.py` with the refactored model. Keep the original `gnn.py` for now to avoid breaking training; once validated, switch `train_sim_data.py` / `train_real_data.py` to import the scriptable version.
- Add a Python-side `unpack_graph(graph_batch) -> Tuple[Tensor, ...]` helper that pulls the named entries out of `GraphData` and feeds them positionally. Use this in `TensegrityGNNSimulator` so the simulator works against the new signature.
- Add `scripts/export_torchscript.py`:
  ```python
  model = EncodeProcessDecode(...).eval()
  model.load_state_dict(torch.load(ckpt))
  scripted = torch.jit.script(model)
  scripted.save("model.pt")
  ```
- Validate: load `model.pt` in a Python `torch.jit.load()` test, run on a saved `GraphData` snapshot, and assert outputs match the original model to ~1e-6.

## Phase 7 — C++ side

- Build `libtorch` against the same PyTorch version used to script. Pin the version in `requirements.txt` and a `CMakeLists.txt` note.
- C++ caller constructs the eight input tensors (matching shapes/dtypes) and calls `module.forward({...})`. Returns a tuple — destructure with `.toTuple()->elements()`.
- For the dynamic contact graph (contact edges added/removed each step): the scripted model accepts variable-length `contact_edge_index` and `contact_edge_attr`, so this is a runtime concern, not an export concern. Verify by scripting once and calling with two different contact-edge counts.

## Validation checkpoints

1. After Phase 1–5: `torch.jit.script(EncodeProcessDecode(...))` succeeds without error.
2. Numerical parity vs. the original model on 100 random `GraphData` samples (max abs diff < 1e-5).
3. End-to-end MPPI rollout in C++ produces the same predicted accelerations as the Python `TensegrityGNNSimulator` on a fixed seed.

## Risk / unknowns

- **`MessagePassing` parity.** PyG's `propagate` has historically applied small implementation details (e.g., self-loop handling, fused kernels) that a naive scatter might miss. Phase 4 needs the parity check in validation step 2 to confirm.
- **Recurrent state shape.** `lstm_forward` slices `node_hidden_state` by `graph['x'].shape[1]` (`gnn.py:158`) — the hidden-state tensor packs (h, c) along dim 1. This survives the refactor unchanged but the C++ caller must allocate the doubled-width tensor for LSTM mode.
- **`use_cable_decoder`** branch: scriptable as long as `self.cable_decode_fn` is `Optional[nn.Module]` and the branch is `if self.cable_decode_fn is not None`.

## Estimated effort

- Phase 1–3: ~1 day (mechanical).
- Phase 4: ~1 day (the parity bug surface).
- Phase 5–6: ~0.5 day.
- Phase 7 (C++ harness): ~1–2 days depending on existing build setup.
