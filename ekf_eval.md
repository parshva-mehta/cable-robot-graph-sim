# Plan: EKF Rollout Outputs in eval.py

## Goal

Add two new output modes to `eval.py` that produce EKF-filtered rollout files using the same output pipeline (same 13-values-per-rod-per-line format) as the existing `rollout_states.txt`:

| Output file | Jacobian method | Module |
|---|---|---|
| `rollout_states.txt` | None (pure GNN rollout, existing) | `simulator.run()` |
| `rollout_states_ekf.txt` | Finite-diff in quat space (39D) | `ekf.run_ekf_rollout` |
| `rollout_states_ekf_exp.txt` | Exp-map Jacobian (36D) | `ekf_alt.run_ekf_rollout` |

## Current State

- **`eval.py`**: Loads model via `torch.load`, builds start state + controls from `processed_data.json` / `extra_state_data.json`, calls `simulator.run()`, writes `rollout_states.txt` (one line per timestep, 13 floats per rod: `x y z qw qx qy qz vx vy vz wx wy wz`), then runs `evaluate()` for metrics.
- **`ekf.py`**: `run_ekf_rollout(simulator, gt_data, extra_gt_data, dt, ..., use_finite_diff=True/False)` returns a list of frame dicts `{'time', 'pose', 'state'}`. `state` is `(1, 39, 1)` quat-format tensor. `pose` is flattened `(pos, quat)` per rod (7*n_rods).
- **`ekf_alt.py`**: Same API signature. `state` is `(1, 36, 1)` exp-map tensor. `pose` is still flattened `(pos, quat)` per rod (converted back from exp for compatibility).
- **Output format difference**: EKF frames store `pose` (7 values/rod) and `state`. The existing rollout_states.txt writes full 13-value state (pos + quat + vel). For EKF frames, the full 13-value state must be recovered from `frame['state']`.

## Design Decisions

### Argument interface

Add a single `--mode` argument with three choices rather than separate flags:

```
--mode raw        (default, existing behavior)
--mode ekf        (quat-space EKF, finite-diff Jacobian → rollout_states_ekf.txt)
--mode ekf_exp    (exp-map EKF → rollout_states_ekf_exp.txt)
```

The `--output` argument continues to work as an explicit override. When not provided, the output filename is derived from the mode:
- `raw` → `rollout_states.txt`
- `ekf` → `rollout_states_ekf.txt`
- `ekf_exp` → `rollout_states_ekf_exp.txt`

### EKF-specific arguments

Add optional arguments that map directly to `run_ekf_rollout` parameters:

```
--dt                  float   (default 0.01, used by EKF modes only)
--process_noise       float   (default 1e-4)
--measurement_noise   float   (default 1e-3)
--observe_pose_only   flag    (default False)
--innovation_gate     float   (default inf, no gating)
--dataset_idx         int     (default 9, also used by raw mode)
```

These are ignored in `raw` mode.

### Output format unification

Both EKF modules return frames where `frame['state']` contains the full filtered state. To write the same format as `rollout_states.txt`:

- **`ekf` mode**: `frame['state']` is `(1, 39, 1)` — squeeze to `(39,)` = 13 values * 3 rods. Write directly.
- **`ekf_exp` mode**: `frame['state']` is `(1, 36, 1)` in exp-map space. Must convert back to quat state via `exp_state_to_quat_state()` to get `(1, 39, 1)`, then write. This ensures the output file has the same quat-based format for downstream consumers.

### Metrics reuse

The existing `evaluate()` function re-runs a full GNN rollout internally (it calls `rollout_by_ctrls`). For EKF modes, metrics should be computed directly from the EKF frames against `gt_data`, not by re-running the GNN. Write a small helper `evaluate_frames(frames, gt_data, n_rods, device)` that computes the same COM/rotation/penetration errors from the frame list.

## Implementation Steps

### Step 1: Add new arguments to `argparse` in `main()`

In `eval.py:144-155`, add:

```python
parser.add_argument('--mode', choices=['raw', 'ekf', 'ekf_exp'], default='raw')
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--process_noise', type=float, default=1e-4)
parser.add_argument('--measurement_noise', type=float, default=1e-3)
parser.add_argument('--observe_pose_only', action='store_true')
parser.add_argument('--innovation_gate', type=float, default=float('inf'))
parser.add_argument('--dataset_idx', type=int, default=9)
```

Modify the existing `--output` default to `None` so it can be derived from mode.

### Step 2: Add imports for EKF modules

At the top of `eval.py`, conditionally or unconditionally add:

```python
from ekf import run_ekf_rollout as run_ekf_quat
from ekf_alt import run_ekf_rollout as run_ekf_exp
from linearization_exp import exp_state_to_quat_state
```

These imports pull in `gtsam`, `linearization`, and `linearization_exp`. Since eval.py already requires `torch` and the simulator, and gtsam is in `requirements.txt`, no new dependencies are needed.

### Step 3: Derive output filename from mode

After parsing args, if `args.output` is still `None`:

```python
output_names = {'raw': 'rollout_states.txt',
                'ekf': 'rollout_states_ekf.txt',
                'ekf_exp': 'rollout_states_ekf_exp.txt'}
args.output = output_names[args.mode]
```

### Step 4: Branch on mode in `main()`

After loading the model, data, and building `start_state` / `ctrls` / `init_rest_lengths` / `init_motor_speeds` (lines 157-198 — this setup code is shared across all modes):

**`raw` mode** (lines 199-219): No changes. Existing `simulator.run()` path.

**`ekf` mode**: Call `run_ekf_quat()` with the loaded data:

```python
frames = run_ekf_quat(
    simulator, gt_data, extra_data, dt=args.dt,
    process_noise_scale=args.process_noise,
    measurement_noise_scale=args.measurement_noise,
    observe_pose_only=args.observe_pose_only,
    start_state=start_state,
    use_finite_diff=True,
    innovation_gate_sigma=args.innovation_gate,
    dataset_idx_val=args.dataset_idx,
)
```

Then write output using the shared writer (Step 5).

**`ekf_exp` mode**: Call `run_ekf_exp()`:

```python
frames = run_ekf_exp(
    simulator, gt_data, extra_data, dt=args.dt,
    process_noise_scale=args.process_noise,
    measurement_noise_scale=args.measurement_noise,
    observe_pose_only=args.observe_pose_only,
    start_state=start_state,
    use_finite_diff=False,
    innovation_gate_sigma=args.innovation_gate,
    dataset_idx_val=args.dataset_idx,
)
```

Note: `ekf_alt.run_ekf_rollout` does not have `use_finite_diff` as a parameter name but `linearize_dynamics_exp` supports it. The default in `ekf_alt` is `False` (uses `torch.func.jacrev` on the exp-map wrapped step). This is the intended behavior — the exp-map variant's advantage is the clean 36x36 Jacobian via autodiff.

### Step 5: Write EKF frames to output file

Extract the shared file-writing logic into a helper, or inline it after the mode branch:

```python
def write_frames_to_file(frames, output_path, n_rods, mode):
    with open(output_path, 'w') as f:
        for frame in frames:
            state_t = frame['state']  # (1, state_dim, 1)
            if mode == 'ekf_exp':
                # Convert exp-map (1,36,1) → quat (1,39,1)
                state_t = exp_state_to_quat_state(state_t)
            row = state_t.squeeze().cpu().numpy()  # (39,) = 13 * n_rods
            f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
    print(f'Wrote {len(frames)} timesteps to {output_path}')
```

For `raw` mode, keep the existing write logic (which writes `state_vals` as the first line, then `all_states`).

### Step 6: Compute and print metrics from EKF frames

Write `evaluate_from_frames()` that mirrors the existing `evaluate()` but takes the frame list directly instead of re-running the simulator:

```python
def evaluate_from_frames(frames, gt_data, n_rods, device):
    """Compute COM, rotation, and penetration errors from EKF frames vs gt_data."""
    num_steps = min(len(frames) - 1, len(gt_data) - 1)
    com_errs, rot_errs, pen_errs = [], [], []
    for i in range(1, num_steps + 1):
        state_t = frames[i]['state']
        # For ekf_exp, state is (1,36,1) — need quat for metrics
        # Caller should convert before passing, or handle here
        ...  # Same per-rod error logic as existing evaluate()
    return avg_com_err, avg_rot_err, avg_pen_err
```

The frame's `state` must be in quat format (39D) before extracting pos/quat for metrics. For `ekf_exp` mode, convert via `exp_state_to_quat_state()` before passing to this function.

### Step 7: Refactor `main()` for clarity

After all three modes converge, print the same metrics summary:

```python
print(f'COM Error (MSE):       {com_err:.6f} m^2')
print(f'Rotation Error (mean): {rot_err:.6f} rad')
print(f'Penetration Error:     {pen_err:.6f} m')
```

## Key Concerns

### 1. Simulator state mutation

Both `run_ekf_quat` and `run_ekf_exp` call `simulator.step()` internally (via `linearize_dynamics` / `linearize_dynamics_exp`). They mutate `simulator.ctrls_hist` and `simulator.node_hidden_state`. Both EKF rollout functions already handle this correctly — they save/restore context via `_save_model_ctx` / `_restore_model_ctx` inside the linearization calls. However, the cable rest lengths and motor states are initialized at the top of each `run_ekf_rollout`, so the shared setup in `main()` (lines 190-198) will be overwritten. This is fine — the EKF functions handle their own init.

### 2. `extra_state_data.json` vs `extra_data`

`eval.py` currently loads `extra_state_data.json` into `extra_data`. The EKF functions expect this as `extra_gt_data` — same format (list of dicts with `controls`, `rest_lengths`, `motor_speeds`). No conversion needed.

### 3. `evaluate()` re-runs the simulator

In `raw` mode, `evaluate()` at line 224 re-runs a full GNN rollout via `rollout_by_ctrls`. For EKF modes, we should NOT call this function — the EKF frames are the rollout. Use `evaluate_from_frames()` instead.

### 4. EKF requires `gtsam`

`gtsam` is already in `requirements.txt`. But importing `ekf.py` / `ekf_alt.py` at the top of `eval.py` will fail if gtsam is not installed, even in `raw` mode. Solution: use lazy imports inside the mode branch, or guard with `if args.mode != 'raw'`.

### 5. Output consistency

For `ekf_exp` mode, the conversion `exp_state_to_quat_state()` recovers full 13-value-per-rod state vectors in quat format. The velocities (linvel, angvel) are the same in both representations (indices 6:12 in exp = indices 7:13 in quat). The conversion only affects the rotation representation (exp 3D → quat 4D). So the output file will have the same semantics as the raw rollout.

## Files Modified

| File | Changes |
|---|---|
| `eval.py` | Add `--mode` and EKF args; branch on mode; add `write_frames_to_file()` and `evaluate_from_frames()`; lazy-import `ekf` / `ekf_alt` |

No other files need modification. `ekf.py`, `ekf_alt.py`, `linearization.py`, and `linearization_exp.py` are used as-is.

## Verification

1. `python3 eval.py --mode raw ...` produces identical `rollout_states.txt` to current behavior
2. `python3 eval.py --mode ekf ...` produces `rollout_states_ekf.txt` with same line format
3. `python3 eval.py --mode ekf_exp ...` produces `rollout_states_ekf_exp.txt` with same line format
4. All three files have the same number of columns per line (13 * n_rods = 39)
5. Metrics are printed for all three modes
6. `--output custom_name.txt` overrides the default filename in any mode
