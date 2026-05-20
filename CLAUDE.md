# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNN-based physics simulator for cable-driven tensegrity robots. The learned model is used for dynamics prediction, state estimation (MEKF / GTSAM-MEKF), and MPPI-based control. Associated paper: arXiv 2602.21331.

## Setup

```bash
conda create --name cable_robot_gnn python=3.10
conda activate cable_robot_gnn
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `mujoco`, `dm-control`, `numpy-quaternion`, `gtsam`.

## Common Commands

```bash
# Train GNN (simulation data only)
python3 train_sim_data.py

# Train GNN (real robot + simulation data mixed)
python3 train_real_data.py

# Evaluate: raw GNN rollout
python3 eval.py --mode raw

# Evaluate: MEKF (manual Kalman, exp-map covariance)
python3 eval.py --mode ekf --process_noise 1e-6 --measurement_noise 1e-1

# Evaluate: GTSAM-MEKF (same filter, GTSAM GaussianFactorGraph backend)
python3 eval.py --mode gtsam

# Compare filter vs raw GNN in one run
python3 eval.py --mode ekf --compare_raw

# Run MPPI control on real robot
python3 mppi_run.py

# Phase 3–5 MEKF evaluation suite (noise sweep, NEES, gating, ablations)
python scripts/mekf_refiner_eval.py --phases 3,4,5 --traj traj_6,traj_3

# Same suite for GTSAM backend
python scripts/gtsam_refiner_eval.py --phases 3,4,5

# Run tests (require model + data paths to be present)
pytest tests/test_mekf_limits_v2.py          # preferred; supersedes v1
python tests/test_mekf_limits_v2.py          # standalone with default paths
```

## Configuration

Two config types must be kept consistent:

- **Training configs**: `nn_training/configs/*.json` — epochs, LR, batch size, data paths, `num_steps_fwd`
- **Simulator/robot configs**: `simulators/configs/*.json` — GNN architecture, robot geometry (rods/cables), physics params (gravity, dt, `num_ctrls_hist`)

Training scripts hard-code which config files to load. To switch, edit the file paths at the top of `train_sim_data.py` or `train_real_data.py`.

## Architecture

### Data Flow

```
Robot state (pos, quaternion, vel per rod)
    ↓ GraphDataProcessor (gnn_physics/data_processors/graph_data_processor.py)
Graph: nodes = rod bodies, edges = cable/contact/rigid constraints
    ↓ EncodeProcessDecode GNN (gnn_physics/gnn.py)
Predicted accelerations (3 DOF per node)
    ↓ Physics integrator
Next state
```

### Module Responsibilities

- **`gnn_physics/`** — GNN architecture (`EncodeProcessDecode`) + `AccumulatedNormalizer` + `GraphDataProcessor` that converts physics states into node/edge feature tensors. The data processor is the most complex component (~1300 lines).
- **`state_objects/`** — Physics state hierarchy: `PrimitiveShape` → `RigidObject` → `CompositeBody` → `TensegrityRod`. Cables are separate (`Cable`, `ActuatedCable`). All state is quaternion-based.
- **`robots/`** — `TensegrityRobot` assembles rods and cables from config. `TensegrityRobotGNN` precomputes inverse mass/inertia for graph features.
- **`simulators/`** — `TensegrityGNNSimulator` wraps the GNN for rollout; manages control history (`ctrls_hist`) and LSTM hidden states (`node_hidden_state`).
- **`nn_training/`** — Training engines: `TensegrityGNNTrainingEngine` (sim-only) and `RealTensegrityMultiSimMultiStepMotorGNNTrainingEngine` (real+sim with curriculum scheduling).
- **`actuation/`** — `DCMotor` model (RPM-based) + `PIDController`. Actuated cables change rest length to modulate tension.
- **`mujoco_physics_engine/`** — Ground-truth MuJoCo simulator for data generation only; not used in the GNN training loop.
- **`utilities/`** — Quaternion math (`torch_quaternion.py`), inertia tensors, tensor helpers.

### GNN Architecture

`EncodeProcessDecode` in `gnn_physics/gnn.py`:
- **Encoder**: Separate MLPs per node type and edge type
- **Processor**: N steps of message passing (default 4), optionally shared weights
- **Decoder**: Predicts 3-DOF accelerations; separate cable decoder available
- Recurrent variants (LSTM/GRU) via `recurrent_type` config field

### State Estimation Layer (MEKF / GTSAM-MEKF)

Two parallel implementations of the same Multiplicative Extended Kalman Filter:

| File | Backend |
|------|---------|
| `ekf.py` | Manual numpy Kalman gain + Joseph-form covariance |
| `ekf_gtsam.py` | GTSAM `GaussianFactorGraph` (QR/Cholesky solver) |

Both share the same public API (`run_ekf_rollout`, `OnlineEKF`) and all helper functions. `ekf_gtsam.py` imports helpers from `ekf.py` and replaces only `_gtsam_update()`.

**State representation** (both implementations):
- Mean: 39D quat space (3 pos + 4 quat + 3 linvel + 3 angvel per rod)
- Covariance: 36D exp-map space (3 pos + 3 exp_rot + 3 linvel + 3 angvel per rod)

**Supporting modules**:
- `linearization.py` — quaternion tangent-space projections; `_save_model_ctx` / `_restore_model_ctx` for LSTM state snapshots
- `linearization_exp.py` — exp-map state conversions (`quat_state_to_exp_state`, `exp_state_to_quat_state`), `linearize_dynamics_exp` (finite-diff or autograd Jacobian), spectral radius clamping

### LSTM Cadence

`simulator.step(x, ctrls)` advances `ctrls_hist` and `node_hidden_state` as a side effect. Any code calling `step` must either save/restore the context (`_save_model_ctx` / `_restore_model_ctx`) or own the GNN cadence explicitly. This is the primary source of subtle bugs — the EKF batch rollout (`run_ekf_rollout`) saves context before the batch GNN call and re-advances LSTM exactly once per batch to match `sim.run`'s cadence.

### Robot Structure

The 3-bar tensegrity has 3 rods, each a `CompositeBody` of cylinder + two spherical endcaps + motor housings. State per rod: 3 pos + 4 quat + 3 linvel + 3 angvel = 13 values. Six actuated cables (DC motor winches) control tension; passive cables provide structural constraints.

### Multi-Step Training

Progressive curriculum: phases increase `num_steps_fwd` (e.g., 4→4→8→8→16) with decreasing learning rates. Real-data training additionally schedules `mix_ratio` (real:sim) and `target_dt`.

## Key Design Decisions

- **Quaternion representation** throughout (not Euler angles). Use `utilities/torch_quaternion.py` for all rotation operations.
- **Graph features** encode physics properties (stiffness, damping, mass) directly as edge/node attributes — the GNN is physics-aware, not purely data-driven.
- **Contact edges** (rod-to-ground) are dynamically added/removed based on current state.
- **Control history** (`num_ctrls_hist` timesteps of past controls) is concatenated to node features.
- **Exp-map covariance**: The MEKF keeps mean in quat space (no round-trip error) and covariance in 36D exp-map space (naturally full-rank; no tangent projection or ε·qqᵀ regularization needed).
- **Monkey-patching `_ekf_step`**: `scripts/mekf_refiner_eval.py` and `tests/` replace `ekf._ekf_step` at module level to capture per-step diagnostics. Always restore the original in a `finally` block.

## Tests

Tests live in `tests/` and require the trained model and a data trajectory to be present at the default paths hard-coded in each file. `test_mekf_limits_v2.py` supersedes `test_mekf_limits.py` (fixes LSTM cadence and K→0 semantics).

Four test cases (runnable standalone or via pytest):
- `test_phase1_invariants` — covariance stays PD, state stays finite, quat stays unit-norm
- `test_zero_K` — P₀ ≪ R → K ≈ 0, EKF state ≈ raw GNN state
- `test_unity_K` — P₀ ≫ R → K ≈ I, EKF state ≈ measurement
- `test_predict_only` — no measurements → EKF predict path matches step-by-step GNN exactly

## Branch Structure

| Branch | Description |
|--------|-------------|
| `main` | Stable baseline |
| `MEKF` | Manual MEKF implementation (`ekf.py`) |
| `GSTAM` | GTSAM-MEKF parallel (`ekf_gtsam.py`, `scripts/gtsam_refiner_eval.py`) |
| `ekf-fixes` | EKF bug-fix history |
