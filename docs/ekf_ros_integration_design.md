# Design: Streaming EKF State to ROS Noetic (Docker) via roslibpy

## Context

`ekf.py` produces, at every timestep, a state estimate for each rod: 13 values
= position (3) + quaternion (4) + linear velocity (3) + angular velocity (3).
This repo has **two** places that produce such estimates:

- `run_ekf_rollout` (`ekf.py`) — batch rollout over a ground-truth trajectory
  (mirrors the reference repo `tensegrity_gnn_simulator_public`).
- `OnlineEKF` (`ekf.py`) — a **streaming** wrapper (`initialize`/`step`) that
  does not exist in the reference repo, and is the better fit for a live timer
  loop.

That state lives only as torch tensors inside the Python process that also runs
the GNN simulation/linearization (torch, gtsam — no ROS installed). The goal is
to stream each rod's estimated pose+velocity into ROS Noetic, running in a
separate Docker container, **without installing ROS/catkin into this repo's
environment**.

## Chosen architecture: roslibpy + rosbridge_suite

```
┌─────────────────────────────┐        websocket (port 9090)        ┌──────────────────────────────┐
│ This repo (host).           │ ───────────────────────────────────▶│ Docker: ROS Noetic container   │
│ Python 3.10, torch, gtsam.  │   roslibpy.Topic(...).publish(msg)  │  - roscore                     │
│ No ROS needed.              │                                      │  - rosbridge_websocket (9090)  │
│  ekf.py -> sim_data_        │                                      │  - foxglove_bridge (8765)      │
│  publisher.py               │                                      │  - RViz / other ROS nodes      │
└─────────────────────────────┘                                      └──────────────────────────────┘
```

This repo's process depends only on the pure-Python `roslibpy` package (added to
`requirements.txt`) — it talks JSON-over-websocket to `rosbridge_websocket`,
which republishes onto real ROS topics inside the container. No rospy, no catkin
workspace, no Python-version coupling to Noetic's Python 3.8. `roslibpy` is
imported **lazily** inside `RodStatePublisher.connect()`, so `sim_data_publisher`
and all its pure helpers import and unit-test without ROS or roslibpy present.

## Message design: `nav_msgs/Odometry`, one topic per rod

Each rod behaves like a free rigid body, and `nav_msgs/Odometry` carries exactly
pose (position+orientation) **and** twist (linear+angular velocity). It is a
stock message — nothing custom needs to be compiled inside the Noetic image.

The alternative, the `interface` package's `TensegrityBars`
(`PRX-Kinodynamic/tensegrity`), has pose+covariance but **no twist fields**, so
it cannot carry the EKF's velocity estimates. `Odometry` is the only stock
message that does. (For the pose-only path there is already a file route — see
"Two paths" below.)

Topic naming: `/tensegrity/<rod_name>/odom`, using the rod names from the robot
config in state-layout order (`rod_01`, `rod_23`, `rod_45` for the 3-bar
config).

Field mapping, per rod, per EKF state block `state[13*r : 13*r+13]`:

| EKF slice | Odometry field | Notes |
|---|---|---|
| `pos = state[0:3]` | `pose.pose.position.{x,y,z}` | scaled to meters (see Units) |
| `quat = state[3:7]` | `pose.pose.orientation.{x,y,z,w}` | **reorder** `(w,x,y,z) → (x,y,z,w)` |
| `linvel = state[7:10]` | `twist.twist.linear.{x,y,z}` | world→body rotation + scaled |
| `angvel = state[10:13]` | `twist.twist.angular.{x,y,z}` | world→body rotation, **not** scaled |
| — | `header.stamp` | wall-clock (default) or sim time |
| — | `header.frame_id` | `"world"` |
| — | `child_frame_id` | rod name |

Covariance: `pose.covariance` and `twist.covariance` are left zeroed. GTSAM's
state covariance is 13-dim per rod (quaternion included) and has no exact
closed-form projection onto the 6×6 (3 position + 3 small-angle) ordering ROS
expects.

## Hook points (both), with a duck-typed sink

`ekf.py` never imports roslibpy. Both hook points take an optional
`publisher=None`; when provided it is called as
`publisher.publish_state(time, state)` once for the initial state and once per
timestep, where `state` is the `(1, state_dim, 1)` tensor. Behavior is
**unchanged when `publisher is None`**, so existing offline callers (`eval.py`)
are untouched.

- `run_ekf_rollout(..., publisher=None)`: published after the initial frame and
  after each per-step frame.
- `OnlineEKF(..., publisher=None)`: `initialize()` publishes the initial state at
  `t=0`; each `step()` increments an internal `t += dt` and publishes the new
  estimate.

The sink is duck-typed (`publish_state(time, state)`), so `RodStatePublisher`,
`RolloutStateFileWriter`, and `CompositeSink` are all interchangeable, and a test
can pass a recording stub.

## Verified against this repo's code (not assumed)

Everything below was checked in this repo rather than copied from the reference,
because several details differ.

### 1. Units — RE-DERIVED; **0.1101694915**, not 0.1

The ROS reader (`interface/scripts/sim_data_publisher.py`) does two things with
positions:

- `create_transform` multiplies incoming COM positions by `data_scale_factor`
  (a launch param, default `0.1`, comment: *"Data is in cm, but need it in
  meters"*);
- it renders each rod's endcaps at **hardcoded** offsets `±0.325/2` along the
  rod axis (`self.offsetP`/`self.offsetM`) — i.e. a true physical rod length of
  **0.325 m**, independent of `data_scale_factor`.

So the correct sim-units→meters factor makes this simulator's rod length match
that 0.325 m:

```
scale = (ROS physical rod length) / (this sim's rod length)
      = 0.325 m / 2.95 units
      = 0.1101694915...
```

This robot's rods measure **2.95** sim units end-to-end — verified by computing
the Euclidean distance between each rod's two `end_pts` in
`simulators/configs/3_bar_gnn_sim_config.json` (note the filename differs from
the reference repo's `3_bar_tensegrity_gnn_sim_config.json`); all three rods are
exactly 2.95 apart.

The reference repo's rods are **3.25** units, giving `0.325/3.25 = 0.1` exactly —
which is why the reference could use `0.1` and it coincided with the ROS
`data_scale_factor` default. **Here 0.1 would be wrong**: 2.95-unit rods scaled
by 0.1 render at 0.295 m, ~10% short of the ROS endcaps. `DEFAULT_POSITION_SCALE`
is therefore `0.325/2.95`.

Consequence for the ROS side: when replaying **this** robot's file, launch with
`data_scale_factor:=0.11017` (not the 0.1 default). And for the live and file
paths to agree in meters, the live `position_scale` must equal the ROS reader's
`data_scale_factor` — both `0.325/2.95`.

### 2. Velocity frame — **world**, so the twist is rotated into the body frame

`nav_msgs/Odometry.twist` is conventionally expressed in `child_frame_id` (body)
axes. This repo's state velocities are **world-frame**, verified two ways:

- `gnn_physics/data_processors/graph_data_processor.py` `node2pose`:
  `lin_vel = (curr_com_pos - prev_com_pos)/dt` is a finite difference of world
  COM positions, and `ang_vel = compute_ang_vel_vecs(prev_prin, curr_prin, dt)`
  is built from world-frame principal axes — neither is expressed in a body
  frame.
- `utilities/torch_quaternion.update_quat` integrates
  `q_new = quat_exp(0.5*dt*ω) ⊗ q` — angular velocity on the **left**, the
  world/space-fixed convention (a body-rate would right-multiply). `update_quat2`
  and `compute_ang_vel_quat` (`q_curr ⊗ q_prev⁻¹`) agree.

So `RodStatePublisher` rotates the twist into the body frame by default
(`twist_frame="body"`), via `rotate_world_to_body`, which the unit tests assert
equals `torch_quaternion.rotate_vec_quat(inverse_unit_quat(q), v)`.
`twist_frame="world"` opts out (a documented deviation from the ROS convention).

### 3. Scaling applies to one path only

The ROS reader applies `data_scale_factor` to **file** data. So:

- `RolloutStateFileWriter` writes **raw** simulator units (the ROS side scales
  it; pre-scaling would double-convert).
- `RodStatePublisher` scales the live Odometry to meters itself (nothing
  downstream converts it).

The factor multiplies `pose.position` and `twist.linear` (a length per unit
time). `twist.angular` is rad/s and is scale-invariant, as is the orientation
quaternion. Because rotations are orthonormal, scaling commutes with the
world→body twist rotation (unit-tested).

### 4. Quaternion order — this repo is `(w, x, y, z)`

Verified in `utilities/torch_quaternion.py`: `quat_prod` treats `q[:, 0:1]` as
the scalar and `q[:, 1:]` as the vector part. ROS `geometry_msgs/Quaternion` is
`(x, y, z, w)`, so `quat_wxyz_to_ros` reorders. (The ROS reader's
`create_transform` also reads the file quaternion as `(w, x, y, z)`, so the file
path needs no reorder — consistent.)

### 5. Covariance — left zeroed (see above).

### 6. `header.stamp` — wall-clock by default

Simulated time starts at 0, which looks ancient to RViz/TF unless the ROS side
runs `use_sim_time` with a `/clock` source. `stamp_source="wall"` (default) is
safe; `stamp_source="sim"` publishes the rollout's simulated time for a
`use_sim_time` setup.

### 7. `robot.rods` accessor — a property here

The reference reads rod names from `simulator.robot.rods`. In this repo
`robot.rods` is a **property aliasing `rigid_bodies`** (an `OrderedDict` keyed by
rod name), so `rod_names_from_simulator` works unchanged and stays aligned with
the EKF state layout.

## Two paths, on purpose

- **Live Odometry (`RodStatePublisher`)** — the deliverable. Carries velocities;
  pre-scaled to meters. Use when you want the estimate *live* or need the twist.
- **File (`RolloutStateFileWriter`)** — writes the 39-column format the
  `interface` package already replays (`roslaunch interface
  simulated_data.launch data_file:=...`). This overlaps with what `eval.py`
  already writes to `rollout_states_ekf.txt`; the writer exists only to stream
  it **incrementally** (one line per frame, at full float64 precision) so it can
  run inside `CompositeSink` next to the live publisher. It is a convenience,
  not new capability — `eval.py` already owns the offline file path.

## Verification performed

See `TESTING.md` for the exact commands and real outputs. Summary:

1. **Unit (pytest, mock roslibpy)** — quaternion reorder, world→body rotation,
   position/velocity scaling (position & linvel scale; angvel & orientation do
   not), Odometry field mapping, topic lifecycle, file layout. Includes a test
   asserting `DEFAULT_POSITION_SCALE == 0.325/2.95` and that a 2.95-unit rod
   publishes 0.325 m apart.
2. **Wire (real roslibpy, stub websocket server)** — asserts the `advertise`
   and `publish` rosbridge frames on the wire.
3. **Real ROS container** — `ros:noetic-ros-base` + `ros-noetic-rosbridge-server`,
   confirmed real `nav_msgs/Odometry` messages arrive with correct values via
   `rostopic echo` / `rostopic type`.
4. **Real EKF loops** — both `run_ekf_rollout` and `OnlineEKF` driven with the
   publisher attached, asserting the sink fired on every frame. Uses a
   lightweight **stub simulator** (identity dynamics): the GNN forward is
   stubbed, but the real gtsam predict/update and the quaternion tangent-space
   linearization run. No trained `.pt` model or dataset ships in this repo (see
   "Not verified").
5. **Cross-check** — the file path (raw × `data_scale_factor`) and the live path
   (pre-scaled) land on the same meters, using this robot's `0.325/2.95`.

### Not verified

- **The trained GNN forward.** No `.pt` checkpoint or dataset is committed here,
  so Levels 4/5 use a stub simulator. The hook plumbing, EKF math, and units are
  all exercised; the learned dynamics are not. With a real model, run
  `scripts/e2e_check.py --model <path> --data-dir <dir>`.
- **RViz/Foxglove visuals** — not automatable; eyeball per `instructions.md`.
