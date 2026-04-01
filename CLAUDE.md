# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNN-based physics simulator for cable-driven tensegrity robots. The learned model is used for both dynamics prediction and MPPI-based control. Associated paper: arXiv 2602.21331.

## Setup

```bash
conda create --name cable_robot_gnn python=3.10
conda activate cable_robot_gnn
pip install -r requirements.txt
```

Key dependencies: `torch`, `torch-geometric`, `mujoco`, `dm-control`, `numpy-quaternion`.

## Common Commands

```bash
# Train GNN using simulation data only
python3 train_sim_data.py

# Train GNN mixing real robot + simulation data
python3 train_real_data.py

# Evaluate trained model
python3 eval.py

# Run MPPI control on real robot
python3 mppi_run.py
```

There is no test suite. Validation is done via `eval.py`.

## Configuration

Two config types must be consistent with each other:

- **Training configs**: `nn_training/configs/*.json` — epochs, learning rate, batch size, data paths, `num_steps_fwd`
- **Simulator/robot configs**: `simulators/configs/*.json` — GNN architecture, robot geometry (rods/cables), physics params (gravity, dt, `num_ctrls_hist`)

Training scripts hard-code which config files to load. To change configs, edit the file paths at the top of `train_sim_data.py` or `train_real_data.py`.

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
- **`simulators/`** — `TensegrityGNNSimulator` wraps the GNN model for rollout; manages control history and LSTM hidden states.
- **`nn_training/`** — Training engines handle multi-step rollout losses, optimizer updates, and checkpointing. `RealTensegrityMultiSimMultiStepMotorGNNTrainingEngine` adds real-data mixing with curriculum scheduling.
- **`actuation/`** — `DCMotor` model (RPM-based) + `PIDController`. Actuated cables change rest length to modulate tension.
- **`mujoco_physics_engine/`** — Ground-truth MuJoCo simulator used for data generation (not for GNN training loop).
- **`utilities/`** — Quaternion math (`torch_quaternion.py`), inertia tensors, tensor helpers.

### GNN Architecture

`EncodeProcessDecode` in `gnn_physics/gnn.py`:
- **Encoder**: Separate MLPs per node type and edge type
- **Processor**: N steps of message passing (default 4), optionally with shared weights
- **Decoder**: Predicts 3-DOF accelerations; separate cable decoder available
- Recurrent variants (LSTM/GRU) are supported via `recurrent_type` config field

### Robot Structure

The 3-bar tensegrity has 3 rods, each modeled as a `CompositeBody` of a cylinder + two spherical endcaps + motor housings. State per rod: 3 pos + 4 quat + 3 linvel + 3 angvel = 13 values. Six actuated cables (DC motor winches) control tension; passive cables provide structural constraints.

### Multi-Step Training

Training uses progressive curriculum: phases increase `num_steps_fwd` (e.g., 4→4→8→8→16) with decreasing learning rates. Real-data training additionally schedules `mix_ratio` (real:sim) and `target_dt`.

## Key Design Decisions

- **Quaternion representation** throughout (not Euler angles). Use `utilities/torch_quaternion.py` for all rotation operations.
- **Graph features** encode physics properties (stiffness, damping, mass) directly as edge/node attributes — the GNN is physics-aware, not purely data-driven.
- **Contact edges** (rod-to-ground) are dynamically added/removed based on current state.
- **Control history** (`num_ctrls_hist` timesteps of past controls) is concatenated to node features for the GNN input.
