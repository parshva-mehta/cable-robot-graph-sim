# Streaming EKF state to ROS over the rosbridge websocket

How to set up and run the live websocket link between this repo's EKF and a ROS
Noetic container. Design rationale (and everything that was verified) lives in
[`docs/ekf_ros_integration_design.md`](docs/ekf_ros_integration_design.md);
step-by-step verification with real outputs is in [`TESTING.md`](TESTING.md).

## How it works

```
┌─────────────────────────────────┐   websocket :9090   ┌──────────────────────────────┐
│ This repo (host)                │ ──────────────────► │ Docker: ROS Noetic            │
│ python 3.10, torch, gtsam       │  roslibpy publishes │  roscore                      │
│ NO ROS installed                │  JSON over ws       │  rosbridge_websocket  :9090   │
│                                 │                     │  (foxglove_bridge     :8765)  │
│ ekf.py ──► sim_data_publisher   │                     │  ► /tensegrity/<rod>/odom     │
└─────────────────────────────────┘                     └──────────────────────────────┘
```

The only host-side dependency is the pure-Python `roslibpy` package. rosbridge
republishes the JSON onto real ROS topics inside the container as
`nav_msgs/Odometry`, one topic per rod:

| Rod index | Rod name | Color (ROS side) | Topic |
|---|---|---|---|
| 0 | `rod_01` | red | `/tensegrity/rod_01/odom` |
| 1 | `rod_23` | green | `/tensegrity/rod_23/odom` |
| 2 | `rod_45` | blue | `/tensegrity/rod_45/odom` |

## Prerequisites

- This repo's Python env (`cable_robot_gnn`; see `CLAUDE.md` / `README.md`).
- Docker (for the ROS side).
- Optionally, a checkout of the companion catkin workspace
  ([`parshva-mehta/catkin_ws`](https://github.com/parshva-mehta/catkin_ws)) if
  you want its `TensegrityBars` visualization / foxglove bridge. **Not needed**
  for the live Odometry path — `nav_msgs/Odometry` is a stock message.

## 1. Install the host-side dependency

```bash
conda activate cable_robot_gnn
pip install roslibpy          # already listed in requirements.txt
```

## 2. Start a ROS container with rosbridge

For the live Odometry path a stock image is enough (lighter than the catkin
image, which compiles GTSAM from source):

```bash
docker run -d --name rosbridge -p 9090:9090 ros:noetic-ros-base bash -lc '
source /opt/ros/noetic/setup.bash
apt-get update -qq && apt-get install -y -qq --no-install-recommends ros-noetic-rosbridge-server
roscore & sleep 6
roslaunch --wait rosbridge_server rosbridge_websocket.launch address:=0.0.0.0 port:=9090
'
```

Wait for the bridge (the apt install takes a few minutes the first time):

```bash
docker logs rosbridge 2>&1 | grep "Rosbridge WebSocket server started"
# Rosbridge WebSocket server started at ws://0.0.0.0:9090
```

If `docker` reports *"Cannot connect to the Docker daemon"*, the daemon is not
running (distinct from Docker not installed): start it (`sudo dockerd &`, or
Docker Desktop) and retry.

### Alternative: the catkin workspace image (adds TensegrityBars + foxglove)

```bash
CATKIN_WS=../catkin_ws docker compose \
  -f docker/docker-compose.ros-noetic.yml up --build
```

Its default command (`start_bridges.sh`) starts roscore, rosbridge on **9090**,
and foxglove_bridge on **8765**. The `--build` compiles GTSAM 4.2.0 and pulls
LibTorch, so it is slow; only use it when you need the ROS-side visualization.
`--platform linux/amd64` (set in the compose file) is required on Apple Silicon.

## 3. Smoke-test the websocket

Confirm the link end to end with synthetic data — needs neither torch nor a
dataset:

```bash
export ROSBRIDGE_URL=ws://localhost:9090
python3 - <<'PY'
import json, time
from sim_data_publisher import RodStatePublisher

cfg = json.load(open('simulators/configs/3_bar_gnn_sim_config.json'))
rods = cfg['tensegrity_cfg']['rods']
state = []
for r in rods:
    a, b = r['end_pts']
    com = [(a[i] + b[i]) / 2 for i in range(3)]
    state += com + [1.0, 0.0, 0.0, 0.0] + [0.0, 1.0, 0.0] + [0.0, 0.0, 2.0]

with RodStatePublisher(rod_names=[r['name'] for r in rods], stamp_source='sim') as pub:
    for k in range(200):
        pub.publish_state(k * 0.01, state)
        time.sleep(0.02)
PY
```

Watch it from inside the container:

```bash
docker exec rosbridge bash -lc 'source /opt/ros/noetic/setup.bash; rostopic echo /tensegrity/rod_01/odom'
docker exec rosbridge bash -lc 'source /opt/ros/noetic/setup.bash; rostopic hz  /tensegrity/rod_01/odom'
```

## 4. Run the EKF with the publisher attached

This repo has **two** hook points; both take an optional `publisher=`.

### Batch: `run_ekf_rollout`

```python
import json
from pathlib import Path
import torch

from ekf import run_ekf_rollout
from simulators.tensegrity_gnn_simulator import load_simulator
from sim_data_publisher import RodStatePublisher, rod_names_from_simulator

simulator = load_simulator("path/to/model.pt", map_location=torch.device("cpu"),
                           cache_batch_sizes=[1]).to("cpu")
simulator.eval()

data_dir = Path("path/to/dataset/traj_6")
gt_data    = json.load((data_dir / "processed_data.json").open())
extra_data = json.load((data_dir / "extra_state_data.json").open())

with RodStatePublisher(rod_names=rod_names_from_simulator(simulator)) as pub:
    frames = run_ekf_rollout(simulator, gt_data, extra_data, dt=0.01, publisher=pub)
print(f"published {len(frames)} frames")
```

### Streaming: `OnlineEKF` (the better fit for a live loop)

```python
from ekf import OnlineEKF
from sim_data_publisher import RodStatePublisher, rod_names_from_simulator

with RodStatePublisher(rod_names=rod_names_from_simulator(simulator)) as pub:
    ekf = OnlineEKF(simulator, dt=0.01, n_rods=3, publisher=pub)
    ekf.initialize(start_state, rest_lengths=rest_lengths, motor_speeds=motor_speeds)
    for z_t, u_t in stream:          # your measurement/control source
        ekf.step(z_t=z_t, u_t=u_t)   # publishes each estimate as it is produced
```

`initialize()` publishes the initial state at `t=0`; each `step()` publishes the
new estimate. With `publisher=None` (the default) behavior is unchanged, so
existing offline callers such as `eval.py` are unaffected.

### One-command end-to-end check (no model, no dataset needed)

`scripts/e2e_check.py` runs **both** hook points with the sinks attached. Without
a model it builds a stub simulator; without `--data-dir` it synthesizes data
from the robot config:

```bash
python3 scripts/e2e_check.py                    # file only, stub sim, synthetic data
ROSBRIDGE_URL=ws://localhost:9090 \
  python3 scripts/e2e_check.py --ros            # also stream live Odometry
python3 scripts/e2e_check.py --ros \
  --model model.pt --data-dir path/to/traj_6    # real model + dataset
```

## Configuration

`RodStatePublisher` options, all optional:

| Argument | Default | Notes |
|---|---|---|
| `url` | `$ROSBRIDGE_URL`, else `ws://localhost:9090` | `ws://ros-noetic:9090` from a sibling container; `wss://` works too. |
| `rod_names` | derived `rod_0…rod_N` | Pass `rod_names_from_simulator(simulator)` to match the config. |
| `frame_id` | `"world"` | `header.frame_id`. |
| `topic_namespace` | `"/tensegrity"` | Topics are `<ns>/<rod_name>/odom`. |
| `stamp_source` | `"wall"` | `"sim"` stamps with the rollout's simulated time — only useful with `/use_sim_time` and a `/clock` source. |
| `twist_frame` | `"body"` | Rotates the twist into body axes, per the `Odometry` convention. `"world"` publishes raw world-frame velocities. |
| `position_scale` | `0.325/2.95 ≈ 0.11017` | Simulator-units-to-meters factor for `pose.position` and `twist.linear`. See **Units** below. `1.0` publishes raw units. |
| `queue_size` | `10` | Per-topic rosbridge queue size. |
| `connect_timeout` | `10.0` | Seconds for the websocket handshake. |

## Units (this robot: 2.95 units → 0.325 m, factor ≈ 0.11017)

Simulator lengths are ~10× meters. This robot's rods measure **2.95** sim units
end-to-end (`simulators/configs/3_bar_gnn_sim_config.json`), and the ROS side
hardcodes its endcap offsets at `±0.325/2` — a true rod length of `0.325 m`. So
the correct factor is `0.325 / 2.95 = 0.1101694915`, and `RodStatePublisher`
applies it by default so `pose.position` and `twist.linear` arrive in **true
meters**. `twist.angular` is rad/s and is never scaled.

> This differs from the reference repo (`tensegrity_gnn_simulator_public`),
> whose rods are 3.25 units → factor exactly `0.1`. **Do not use 0.1 here** — it
> would render this robot ~10% too small. See the design doc for the derivation.

The **file** path (`RolloutStateFileWriter`) is deliberately *not* scaled: the
ROS reader applies its own `data_scale_factor`, so pre-scaling would
double-convert. When replaying **this** robot's file, launch the ROS side with
`data_scale_factor:=0.11017` (its default is 0.1, calibrated for the 3.25 robot):

```bash
roslaunch interface simulated_data.launch \
  data_file:=/ws/rollout_ekf.txt data_scale_factor:=0.11017
```

## Visualizing

**Foxglove** (if you ran the catkin image, which runs `foxglove_bridge` on
8765): open [app.foxglove.dev](https://app.foxglove.dev), connect to
`ws://localhost:8765`, add a 3D panel — it renders `Odometry` natively. Rods
should be ~0.325 m long.

**RViz** needs a TF frame matching `header.frame_id`, which the bridges don't
publish. Add one, then set Fixed Frame to `world` and add an Odometry display
per topic:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world map'
```

## Troubleshooting

**`ConnectionError: could not connect to rosbridge at ...`** — the container
isn't up or the port isn't published. `docker ps` should show
`0.0.0.0:9090->9090/tcp`, and `docker logs rosbridge` should say the server
started.

**`Cannot connect to the Docker daemon`** — the daemon isn't running (distinct
from Docker not installed). Start Docker Desktop, or `sudo dockerd &` on a bare
Linux host.

**Connects, but `rostopic list` shows no `/tensegrity/...` topics** — rosbridge
advertises a topic only on first publish. Run a publisher and re-check while it
runs.

**`rostopic echo` prints nothing** — namespace mismatch. `rostopic list | grep
tensegrity` shows what actually exists; it must match `topic_namespace`.

**Rods look ~10% too small (or 10× too big)** — wrong scale. The live path must
use `position_scale=0.325/2.95` (the default); the file path must be replayed
with `data_scale_factor:=0.11017`, not the ROS default 0.1.

**Timestamps look ancient / RViz drops messages** — you're on
`stamp_source="sim"` without a `/clock` source. Use the default `"wall"`.
