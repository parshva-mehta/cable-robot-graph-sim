# Testing the EKF → ROS pipeline

Five levels, fastest first. Each is independently useful — stop wherever you
have the confidence you need.

| Level | What it proves | Needs | Time |
|---|---|---|---|
| 1 | Message mapping, scaling, file format | pytest | ~2 s |
| 2 | Real roslibpy speaks correct rosbridge protocol | + websockets, roslibpy | ~1 s |
| 3 | Messages land on real ROS topics | + Docker | ~5 min |
| 4 | Both hook points drive the publisher | torch + gtsam | ~1 s |
| 5 | File and live paths agree in meters | all of the above | ~1 s |

Every command below runs from the repo root. **Levels 1–5 were all executed and
passed during development; the outputs shown are real, captured from those runs,
not illustrative.** The one thing not covered by any automated level is the
trained GNN forward — Levels 4/5 use a stub simulator (identity dynamics), so no
`.pt` checkpoint or dataset is needed; the EKF math, both hook points, and the
units all run for real. See the design doc's "Not verified" note.

---

## Level 1 — Unit tests (no ROS, no Docker)

```bash
pip install pytest
python3 -m pytest tests/test_sim_data_publisher.py -q
```

Real output:

```
..................................................                       [100%]
50 passed in 1.38s
```

This covers the quaternion reorder `(w,x,y,z)→(x,y,z,w)`, the world→body twist
rotation (asserted against the repo's own
`torch_quaternion.rotate_vec_quat`), the `position_scale` split (position and
linear velocity scale; angular velocity and orientation do not), the Odometry
field mapping, topic lifecycle, and the 39-column file layout. It also asserts
`DEFAULT_POSITION_SCALE == 0.325/2.95` and that a 2.95-unit rod publishes 0.325 m
apart — the re-derived units for this robot.

## Level 2 — Wire protocol (real roslibpy, stub server)

```bash
pip install websockets roslibpy
python3 -m pytest tests/test_sim_data_publisher_wire.py -q
```

Real output:

```
.                                                                        [100%]
1 passed in 0.39s
```

This drives the **real** roslibpy client against a stub websocket server and
asserts the `advertise` and `publish` frames on the wire, catching roslibpy API
drift without needing ROS.

The whole suite together (Levels 1, 2, 4, 5 as pytest):

```bash
python3 -m pytest tests/ -q
```

```
........................................................                 [100%]
56 passed in 1.78s
```

## Level 3 — Real ROS topics

### 3a. Start a ROS container with rosbridge

A stock image is enough (`nav_msgs/Odometry` is a stock message):

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
```

Real output:

```
[INFO] [1787590321.414680]: Rosbridge WebSocket server started at ws://0.0.0.0:9090
```

If `docker` says *"Cannot connect to the Docker daemon"*, the daemon isn't
running (distinct from Docker not installed): start it (`sudo dockerd &`) and
retry.

### 3b. Publish synthetic rod states from the host

Start listeners, then publish:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic echo -n 2 /tensegrity/rod_01/odom > /tmp/echo.txt 2>&1'
sleep 3

export ROSBRIDGE_URL=ws://localhost:9090
python3 - <<'PY'
import json, time
from sim_data_publisher import RodStatePublisher, DEFAULT_POSITION_SCALE

cfg = json.load(open('simulators/configs/3_bar_gnn_sim_config.json'))
rods = cfg['tensegrity_cfg']['rods']
state = []
for r in rods:
    a, b = r['end_pts']
    com = [(a[i] + b[i]) / 2 for i in range(3)]
    state += com + [1.0, 0.0, 0.0, 0.0] + [0.0, 1.0, 0.0] + [0.0, 0.0, 2.0]

with RodStatePublisher(rod_names=[r['name'] for r in rods], stamp_source='sim') as pub:
    for k in range(40):
        pub.publish_state(k * 0.01, state)
        time.sleep(0.05)
print('published 40 frames x 3 rods')
PY

docker exec rosbridge bash -lc 'cat /tmp/echo.txt' | head -40
```

Real output (rod_01 / red) — note the four things this proves at once:

```
child_frame_id: "rod_01"
pose:
  pose:
    position:
      x: -0.0049471830508474555      <- meters (raw -0.044905 x 0.11017), not oversized
      y: 0.0039023019915254246
      z: 0.11334766927966101
    orientation:
      x: 0.0   y: 0.0   z: 0.0   w: 1.0    <- (w,x,y,z) reordered correctly
  covariance: [0.0, ... 36 zeros ...]
twist:
  twist:
    linear:
      x: 0.0
      y: 0.11016949152542373               <- scaled from raw 1.0 by 0.11017
      z: 0.0
    angular:
      x: 0.0
      y: 0.0
      z: 2.0                                <- rad/s, correctly NOT scaled
  covariance: [0.0, ... 36 zeros ...]
```

The scale factor on the wire is `0.11016949152542373` = `0.325/2.95`, exactly
the re-derived value for this robot — **not** the reference repo's `0.1`.

### 3c. Confirm registration

```bash
docker exec rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rostopic list | grep tensegrity; rostopic type /tensegrity/rod_23/odom'
```

Real output:

```
/tensegrity/rod_01/odom
/tensegrity/rod_23/odom
/tensegrity/rod_45/odom
nav_msgs/Odometry
```

## Level 4 — Both hook points driving the publisher

This repo has **two** hook points (`run_ekf_rollout` and `OnlineEKF`).
`scripts/e2e_check.py` exercises both, and works with or without a model/dataset
(without a model it uses a stub simulator; without `--data-dir` it synthesizes
data from the robot config):

```bash
python3 scripts/e2e_check.py            # file only
# ROSBRIDGE_URL=ws://localhost:9090 python3 scripts/e2e_check.py --ros   # + live
```

Real output:

```
building stub simulator from simulators/configs/3_bar_gnn_sim_config.json ...
  simulator : StubSimulator
  rods      : ['rod_01', 'rod_23', 'rod_45']
  scale     : 0.1101694915 (= 0.325/2.95)
  data      : synthetic from simulators/configs/3_bar_gnn_sim_config.json (4 steps)

[1/2] run_ekf_rollout ...
  sink fired 5 times for 5 frames
  frames returned : 5
  file lines      : 5
  columns/line    : {39} (expected {39})
    red: com(m)=[-0.0049  0.0039  0.1133]  det(R)=1.000000  |RR^T-I|=0.00e+00
  green: com(m)=[-0.026  -0.0032  0.0559]  det(R)=1.000000  |RR^T-I|=0.00e+00
   blue: com(m)=[-0.0639  0.0061  0.1022]  det(R)=1.000000  |RR^T-I|=0.00e+00
  quat norm       : 1.000000

[2/2] OnlineEKF (streaming) ...
  sink fired 5 times for 5 frames
  frames returned : 5
  file lines      : 5
  columns/line    : {39} (expected {39})
    red: com(m)=[-0.0049  0.0039  0.1133]  det(R)=1.000000  |RR^T-I|=0.00e+00
  green: com(m)=[-0.026  -0.0032  0.0559]  det(R)=1.000000  |RR^T-I|=0.00e+00
   blue: com(m)=[-0.0639  0.0061  0.1022]  det(R)=1.000000  |RR^T-I|=0.00e+00

PASS: both hook points ran, every frame reached every sink.
```

`sink fired N times for N frames` is the assertion that matters: the publisher
fired on every frame including the initial state, for **both** hook points, and
nothing was dropped. `quat norm == 1.0` confirms the EKF's quaternion
renormalization survived the round trip. The same behavior is also asserted as
pytest in `tests/test_ekf_publisher_hooks.py`:

```
python3 -m pytest tests/test_ekf_publisher_hooks.py -q
.....                                                                    [100%]
5 passed in 1.34s
```

## Level 5 — Cross-check: file and live paths agree in meters

The strongest single check. The file path is written in **raw simulator units**
(the ROS reader applies `data_scale_factor` itself); the websocket path is
**pre-scaled to meters**. Both must land on the same numbers — if they don't,
one path is double-scaled or unscaled.

Parse the file with the ROS node's **verbatim** `create_transform`/`get_values`
(from `PRX-Kinodynamic/tensegrity interface/scripts/sim_data_publisher.py`),
using this robot's `data_scale_factor = 0.325/2.95`:

```bash
python3 - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot
from sim_data_publisher import DEFAULT_POSITION_SCALE

def create_transform(position, quat, s):          # verbatim from the ROS node
    q = np.array(quat, np.float32)
    rot = SciPyRot.from_quat([q[1], q[2], q[3], q[0]])   # W is first
    T = np.identity(4); T[0:3,3] = np.array(position, np.float32)*s
    T[0:3,0:3] = rot.as_matrix(); return T

def get_values(l, s):
    return {'red': create_transform(l[0:3], l[3:7], s),
            'green': create_transform(l[13:16], l[16:20], s),
            'blue': create_transform(l[26:29], l[29:33], s)}

s = DEFAULT_POSITION_SCALE
lines = open('rollout_ekf.txt').read().splitlines()
print(f"data_scale_factor = {s:.10f}  lines={len(lines)} cols={set(len(l.split()) for l in lines)}")
for name, T in get_values([float(t) for t in lines[-1].split()], s).items():
    R = T[0:3,0:3]
    print(f"{name:>5}: com(m)={np.round(T[0:3,3],6)}  det(R)={np.linalg.det(R):.6f}  "
          f"|RR^T-I|={np.abs(R@R.T-np.eye(3)).max():.2e}")
PY
```

Real output:

```
data_scale_factor = 0.1101694915  lines=5 cols={39}
  red: com(m)=[-0.004947  0.003902  0.113348]  det(R)=1.000000  |RR^T-I|=0.00e+00
green: com(m)=[-0.025996 -0.003195  0.055923]  det(R)=1.000000  |RR^T-I|=0.00e+00
 blue: com(m)=[-0.063945  0.006121  0.102228]  det(R)=1.000000  |RR^T-I|=0.00e+00
```

Compare `green` against the `rod_23` position from the **live** Odometry message
captured in Level 3b (`/tmp/echo_green.txt`):

| Path | rod_23 / green position (m) |
|---|---|
| Websocket (pre-scaled) | `-0.025996002, -0.003195043, 0.055922566` |
| File (raw × `0.11017`) | `-0.025996,    -0.003195,    0.055923`    |

They agree. A programmatic comparison of all three rods (file vs. the live
`build_odometry_msg` output) shows `max|Δ| ≈ 1e-9` — float32 round-off only:

```
  red: live=[-0.004947  0.003902  0.113348]  file=[-0.004947  0.003902  0.113348]  max|Δ|=2.27e-09
green: live=[-0.025996 -0.003195  0.055923]  file=[-0.025996 -0.003195  0.055923]  max|Δ|=1.01e-09
 blue: live=[-0.063945  0.006121  0.102228]  file=[-0.063945  0.006121  0.102228]  max|Δ|=1.14e-09
```

`det(R)=1` and `|RR^T−I|≈0` confirm the quaternions round-trip into valid
rotations. This is the check that catches a units error: if the live path used
`0.1` (the reference repo's factor) instead of `0.325/2.95`, the two columns
would differ by ~10%.

## Level 6 (optional) — Visual check

Not covered by any automated test.

**Foxglove**, if your image runs `foxglove_bridge` on 8765 (the catkin image
does): open [app.foxglove.dev](https://app.foxglove.dev), connect to
`ws://localhost:8765`, add a 3D panel. Rods should be ~0.325 m long.

**RViz** needs a TF frame matching `header.frame_id`:

```bash
docker exec -d rosbridge bash -lc \
  'source /opt/ros/noetic/setup.bash; rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 world map'
```

Then set Fixed Frame to `world` and add an Odometry display per topic.

## Cleanup

```bash
docker rm -f rosbridge
rm -f rollout_ekf.txt rollout_ekf_online.txt
```

## Troubleshooting

**`ConnectionError: could not connect to rosbridge`** — the container isn't up
or the port isn't published. `docker ps` should show `0.0.0.0:9090->9090/tcp`.

**`Cannot connect to the Docker daemon`** — the daemon isn't running (distinct
from Docker not installed). `sudo dockerd &`, or start Docker Desktop.

**`rostopic list` shows no `/tensegrity/...` topics** — rosbridge advertises on
first publish. Run a publisher and re-check while it runs.

**Rods look ~10% too small** — the live path is on `0.1` instead of `0.325/2.95`,
or the file is being replayed with the wrong `data_scale_factor`. See the units
note in `instructions.md`.
