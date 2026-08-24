"""Ship EKF rod state estimates to the ROS Noetic side.

Two sinks, both exposing ``publish_state(time, state)`` so either can be passed
straight to ``run_ekf_rollout(publisher=...)`` / ``OnlineEKF(publisher=...)``,
or combined with ``CompositeSink``:

* :class:`RolloutStateFileWriter` -- writes the 39-column text format the
  `interface` package's own ``sim_data_publisher.py`` already replays via
  ``roslaunch interface simulated_data.launch data_file:=...``. Needs no ROS
  connection and no changes on the ROS side. This is the same layout
  ``eval.py`` already writes to ``rollout_states_ekf.txt``; the writer exists
  only to stream it *incrementally* during a rollout (see the module note on
  overlap with ``eval.py`` below).
* :class:`RodStatePublisher` -- streams live ``nav_msgs/Odometry`` over
  rosbridge, one topic per rod. Carries the velocity estimates that the
  file/``TensegrityBars`` path has no fields for. This is the live path -- the
  actual deliverable.

Units differ between the two on purpose. The file writer emits raw simulator
units because the ROS reader applies its own ``data_scale_factor``; the
Odometry path has no such consumer-side conversion, so it scales to meters
itself (see ``DEFAULT_POSITION_SCALE`` and its derivation below).

See ``docs/ekf_ros_integration_design.md`` for the architecture. In short: this
process holds no ROS dependency at all -- it speaks JSON-over-websocket to a
``rosbridge_websocket`` server running inside a ROS Noetic container, which
republishes onto real ROS topics. One ``nav_msgs/Odometry`` topic per rod,
named ``/tensegrity/<rod_name>/odom``.

The EKF state is 13 values per rod: pos (3), quat (4), linvel (3), angvel (3).
Two conventions differ between this repo and ROS and are handled here:

* **Quaternion order.** This repo stores ``(w, x, y, z)`` (verified in
  ``utilities/torch_quaternion.py``: ``quat_prod`` reads ``q[:, 0:1]`` as the
  scalar part and ``q[:, 1:]`` as the vector part); ``geometry_msgs/Quaternion``
  is ``(x, y, z, w)``.
* **Twist frame.** This repo's state velocities are world-frame. Verified in
  ``gnn_physics/data_processors/graph_data_processor.py`` ``node2pose``:
  ``lin_vel = (curr_com_pos - prev_com_pos) / dt`` is a finite difference of
  world positions, and ``ang_vel`` comes from the world-frame principal axes
  via ``compute_ang_vel_vecs`` -- neither is rotated into a body frame.
  ``torch_quaternion.update_quat`` integrates
  ``q_new = quat_exp(0.5*dt*w) (x) q`` -- angular velocity on the *left*, the
  world/space-fixed convention -- which agrees.  ``nav_msgs/Odometry.twist`` is
  conventionally expressed in ``child_frame_id`` (body) axes, so by default the
  twist is rotated into the body frame before publishing. Pass
  ``twist_frame="world"`` to publish the raw world-frame velocities instead (a
  documented deviation from the ROS convention).

Typical use::

    from sim_data_publisher import RodStatePublisher, rod_names_from_simulator

    with RodStatePublisher(rod_names=rod_names_from_simulator(simulator)) as pub:
        frames = run_ekf_rollout(simulator, gt_data, extra_gt_data, dt,
                                 publisher=pub)

``roslibpy`` is imported lazily, so this module (and every pure helper in it)
can be imported and unit-tested without ROS or roslibpy installed.

Overlap with ``eval.py``: ``eval.py`` (via ``write_frames_to_file``) already
writes the 39-column rollout file *after* a rollout finishes, and the committed
``rollout_states_ekf.txt`` is exactly that. ``RolloutStateFileWriter`` writes
the identical format, but *incrementally* -- one line per timestep as the frame
is produced -- so it can run inside ``CompositeSink`` alongside the live
publisher, and at full float64 precision (``eval.py`` truncates to ``%.8f``).
It is a convenience, not the deliverable; the live Odometry path is.
"""

import math
import os
import time as _time

# 13 = pos(3) + quat(4) + linvel(3) + angvel(3)
STATE_DIM_PER_ROD = 13

DEFAULT_ROSBRIDGE_URL = "ws://localhost:9090"
ODOMETRY_MSG_TYPE = "nav_msgs/Odometry"
DEFAULT_TOPIC_NAMESPACE = "/tensegrity"
DEFAULT_FRAME_ID = "world"

# ---------------------------------------------------------------------------
# Simulator-units-to-meters scale factor (RE-DERIVED for this robot).
#
# The ROS side (PRX-Kinodynamic/tensegrity, interface/scripts/sim_data_publisher.py)
# renders each rod with its endcaps hardcoded at +/-0.325/2 along the rod axis
# (`self.offsetP`/`self.offsetM`), i.e. a true physical rod length of 0.325 m,
# and it multiplies incoming COM positions by `data_scale_factor` (default 0.1).
#
# The correct scale converts THIS simulator's rod length to that 0.325 m so the
# COM spacing matches the hardcoded endcaps:
#
#     scale = (ROS physical rod length) / (this sim's rod length)
#           = 0.325 m / 2.95 units
#           = 0.1101694915...
#
# This robot's rods measure 2.95 sim units end-to-end -- verified from
# simulators/configs/3_bar_gnn_sim_config.json (note the filename differs from
# the reference repo's `3_bar_tensegrity_gnn_sim_config.json`): every rod's two
# `end_pts` are 2.95 apart.
#
# This is DIFFERENT from the reference repo, whose rods are 3.25 units, giving
# 0.325/3.25 = 0.1 exactly. Do not assume 0.1 here -- 2.95-unit rods scaled by
# 0.1 would render at 0.295 m, ~10% short of the ROS endcaps.
#
# The file writer deliberately does NOT apply this -- the ROS reader scales the
# file itself (and for THIS robot must be launched with
# `data_scale_factor:=0.11017` rather than its 0.1 default), so pre-scaling
# would double-convert. Only the live Odometry path, which nothing else scales,
# needs it. For the two paths to land on the same meters, the ROS reader's
# `data_scale_factor` must equal this factor.
ROS_PHYSICAL_ROD_LENGTH_M = 0.325
THIS_SIM_ROD_LENGTH_UNITS = 2.95
DEFAULT_POSITION_SCALE = ROS_PHYSICAL_ROD_LENGTH_M / THIS_SIM_ROD_LENGTH_UNITS

# nav_msgs/Odometry carries 6x6 pose and twist covariances. The EKF's covariance
# is 13-dim per rod (quaternion included) and has no exact closed-form
# projection onto the 3-position + 3-small-angle ordering ROS expects, so v1
# leaves them zeroed. See the design doc's "Covariance mismatch" note.
_ZERO_COVARIANCE = [0.0] * 36


def _as_floats(value, n, name):
    """Coerce a torch tensor / numpy array / sequence into a list of n floats."""
    # Avoid importing torch or numpy: both expose tolist() / flatten-able shapes.
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):  # tensor / ndarray
        value = value.reshape(-1).tolist()
    else:
        value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(value)}")
    return [float(v) for v in value]


def quat_wxyz_to_ros(quat):
    """Reorder a repo ``(w, x, y, z)`` quaternion into a ROS Quaternion dict."""
    w, x, y, z = _as_floats(quat, 4, "quat")
    return {"x": x, "y": y, "z": z, "w": w}


def _vec3(vec):
    x, y, z = vec
    return {"x": x, "y": y, "z": z}


def rotate_world_to_body(quat, vec):
    """Rotate a world-frame vector into the body frame of ``quat``.

    Equivalent to ``R(q).T @ vec``, i.e. ``q* (x) (0, vec) (x) q`` with the
    repo's ``(w, x, y, z)`` layout. Mirrors
    ``torch_quaternion.rotate_vec_quat(inverse_unit_quat(q), vec)``; the unit
    test asserts agreement with that reference implementation.
    """
    w, x, y, z = _as_floats(quat, 4, "quat")
    vx, vy, vz = _as_floats(vec, 3, "vec")

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("cannot rotate by a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Rows of R(q); the transpose is applied by dotting columns below.
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1

    return [
        r00 * vx + r10 * vy + r20 * vz,
        r01 * vx + r11 * vy + r21 * vz,
        r02 * vx + r12 * vy + r22 * vz,
    ]


def ros_time_from_seconds(seconds):
    """Split float seconds into a ROS ``{secs, nsecs}`` stamp."""
    secs = int(math.floor(seconds))
    nsecs = int(round((seconds - secs) * 1e9))
    if nsecs >= 1_000_000_000:  # rounding carried into the next second
        secs += 1
        nsecs -= 1_000_000_000
    return {"secs": secs, "nsecs": nsecs}


def build_odometry_msg(rod_name, stamp_seconds, pos, quat, linvel, angvel,
                       frame_id=DEFAULT_FRAME_ID, twist_frame="body",
                       position_scale=DEFAULT_POSITION_SCALE):
    """Build a ``nav_msgs/Odometry`` message dict for one rod.

    Args:
        rod_name: Rod name, used as ``child_frame_id`` (e.g. ``"rod_01"``).
        stamp_seconds: Header stamp in float seconds.
        pos: World-frame position, 3 elements, in simulator units.
        quat: Orientation as repo-order ``(w, x, y, z)``, 4 elements.
        linvel: World-frame linear velocity, 3 elements, in simulator units.
        angvel: World-frame angular velocity, 3 elements, in rad/s.
        frame_id: Fixed world frame for ``header.frame_id``.
        twist_frame: ``"body"`` (ROS convention, rotates twist by the inverse of
            ``quat``) or ``"world"`` (publish world-frame velocities as-is).
        position_scale: Simulator-units-to-meters factor applied to ``pos`` and,
            since it is a length per unit time, to ``linvel``. ``angvel`` is in
            rad/s and is never scaled; orientation is scale-invariant. Pass
            ``1.0`` to publish raw simulator units.

    Returns:
        A dict matching the ``nav_msgs/Odometry`` layout rosbridge expects.
    """
    if twist_frame not in ("body", "world"):
        raise ValueError(f"twist_frame must be 'body' or 'world', got {twist_frame!r}")

    scale = float(position_scale)
    pos = [v * scale for v in _as_floats(pos, 3, "pos")]
    linvel = [v * scale for v in _as_floats(linvel, 3, "linvel")]
    angvel = _as_floats(angvel, 3, "angvel")

    if twist_frame == "body":
        linvel = rotate_world_to_body(quat, linvel)
        angvel = rotate_world_to_body(quat, angvel)

    return {
        "header": {
            "stamp": ros_time_from_seconds(stamp_seconds),
            "frame_id": frame_id,
        },
        "child_frame_id": rod_name,
        "pose": {
            "pose": {
                "position": _vec3(pos),
                "orientation": quat_wxyz_to_ros(quat),
            },
            "covariance": list(_ZERO_COVARIANCE),
        },
        "twist": {
            "twist": {
                "linear": _vec3(linvel),
                "angular": _vec3(angvel),
            },
            "covariance": list(_ZERO_COVARIANCE),
        },
    }


def _flat_len(value):
    if hasattr(value, "numel"):  # torch.Tensor
        return int(value.numel())
    if hasattr(value, "size") and not callable(value.size):  # numpy.ndarray
        return int(value.size)
    return len(list(value))


def split_rod_states(state):
    """Split a flat EKF state into per-rod ``(pos, quat, linvel, angvel)`` tuples."""
    flat = _as_floats(state, _flat_len(state), "state")
    if len(flat) % STATE_DIM_PER_ROD != 0:
        raise ValueError(
            f"state length {len(flat)} is not a multiple of {STATE_DIM_PER_ROD}"
        )
    rods = []
    for i in range(len(flat) // STATE_DIM_PER_ROD):
        b = flat[i * STATE_DIM_PER_ROD:(i + 1) * STATE_DIM_PER_ROD]
        rods.append((b[0:3], b[3:7], b[7:10], b[10:13]))
    return rods


class RolloutStateFileWriter:
    """Writes the whitespace-separated rollout format the ROS side already reads.

    The `interface` package's own `sim_data_publisher.py` (in the companion
    catkin workspace) parses one line per timestep as 39 floats:

        # 0:3   3:7  7:10 10:13 13:16  16:20  20:23  23:26  26:29  29:33  33:36 36:39
        # PosA quatA  VpA  VqA  PosB   quatB   VpB    VqB    PosC  quatC   VpC   VqC

    which is exactly this repo's EKF state layout -- 3 rods x 13, with red,
    green, blue as rod 0, 1, 2 -- and its `create_transform` reads the
    quaternion as `(w, x, y, z)`, the same order used here. So a line is just
    the flattened state, with no conversion and no leading timestamp column (the
    reader expects PosA at index 0).

    Positions are written in raw simulator units: the ROS side applies its own
    `data_scale_factor` to convert to meters, as it already does for
    `rollout_states.txt` / `rollout_states_ekf.txt`. For this robot the ROS
    launch must set `data_scale_factor:=0.11017` (= 0.325 / 2.95); see
    `DEFAULT_POSITION_SCALE`.

    This is the same format `eval.py` already emits after a rollout; this writer
    only exists to stream it incrementally (one line as each frame is produced)
    so it can run inside `CompositeSink` next to the live publisher, at full
    float64 precision.

    Implements the same `publish_state(time, state)` interface as
    `RodStatePublisher`, so it can be handed to `run_ekf_rollout(publisher=...)`
    directly, or combined with a live publisher via `CompositeSink`.

    Args:
        path: Output file path; parent directories are created.
        float_fmt: Per-value format. The default round-trips float64 exactly.
        expected_n_rods: Fail loudly if the state does not hold this many rods,
            since the ROS reader hard-codes 3 (39 columns). Pass None to allow
            any rod count.
        flush_every: Flush after this many lines (0 disables explicit flushing).
    """

    def __init__(self, path, float_fmt="%.17g", expected_n_rods=3, flush_every=0):
        self.path = path
        self.float_fmt = float_fmt
        self.expected_n_rods = expected_n_rods
        self.flush_every = flush_every
        self._file = None
        self._lines_written = 0

    def open(self):
        """Open the output file for writing. Idempotent."""
        if self._file is None:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._file = open(self.path, "w")
            self._lines_written = 0
        return self

    def close(self):
        """Close the output file. Idempotent."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def lines_written(self):
        return self._lines_written

    def publish_state(self, time, state):
        """Append one timestep. `time` is accepted but unused -- the format has
        no timestamp column."""
        del time  # the reader expects PosA at column 0
        if self._file is None:
            raise RuntimeError(f"{type(self).__name__} is not open; call open() first")

        rods = split_rod_states(state)
        if self.expected_n_rods is not None and len(rods) != self.expected_n_rods:
            raise ValueError(
                f"state holds {len(rods)} rods but the ROS reader expects "
                f"{self.expected_n_rods} ({self.expected_n_rods * STATE_DIM_PER_ROD} "
                f"columns); pass expected_n_rods=None to override"
            )

        values = [v for rod in rods for block in rod for v in block]
        self._file.write(" ".join(self.float_fmt % v for v in values) + "\n")
        self._lines_written += 1
        if self.flush_every and self._lines_written % self.flush_every == 0:
            self._file.flush()
        return self._lines_written


class CompositeSink:
    """Fans `publish_state` out to several sinks.

    Lets a rollout write the ROS-readable file and stream live at the same time::

        sinks = CompositeSink(RolloutStateFileWriter("rollout_ekf.txt"),
                              RodStatePublisher(rod_names=...))
        with sinks:
            run_ekf_rollout(..., publisher=sinks)
    """

    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def publish_state(self, time, state):
        return [sink.publish_state(time, state) for sink in self.sinks]

    def open(self):
        for sink in self.sinks:
            # RodStatePublisher exposes connect(); RolloutStateFileWriter open().
            starter = getattr(sink, "open", None) or getattr(sink, "connect", None)
            if starter is not None:
                starter()
        return self

    def close(self):
        for sink in self.sinks:
            closer = getattr(sink, "close", None)
            if closer is not None:
                closer()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def rod_names_from_simulator(simulator):
    """Read ordered rod names from a simulator's robot config.

    The EKF state is laid out in ``simulator.robot.rods`` order (an
    ``OrderedDict`` keyed by rod name -- here ``rods`` is a property aliasing
    ``rigid_bodies``), so this keeps topic names aligned with state blocks
    (e.g. ``rod_01``, ``rod_23``, ``rod_45`` for the 3-bar config).
    """
    return list(simulator.robot.rods.keys())


class RodStatePublisher:
    """Publishes per-rod EKF estimates as ``nav_msgs/Odometry`` over rosbridge.

    Args:
        url: rosbridge websocket URL. Defaults to the ``ROSBRIDGE_URL``
            environment variable, else ``ws://localhost:9090``.
        rod_names: Ordered rod names matching the EKF state layout. If omitted,
            names are derived lazily as ``rod_0 ... rod_{n-1}`` from the first
            published state; prefer passing ``rod_names_from_simulator(sim)``.
        frame_id: Fixed world frame for ``header.frame_id``.
        topic_namespace: Topic prefix; topics are ``<ns>/<rod_name>/odom``.
        stamp_source: ``"wall"`` publishes wall-clock time (safe default -- works
            without ``/clock`` or ``use_sim_time``); ``"sim"`` publishes the
            rollout's simulated time, which requires the ROS side to run with
            ``use_sim_time`` and a ``/clock`` source to display sensibly.
        twist_frame: ``"body"`` (ROS convention) or ``"world"``. See module docs.
        position_scale: Simulator-units-to-meters factor for ``pose.position``
            and ``twist.linear`` (default ``0.325/2.95 ~= 0.11017``, so rods
            publish at their true 0.325 m length instead of ~10x oversized).
            ``twist.angular`` is rad/s and is never scaled. Pass ``1.0`` for raw
            simulator units.
        queue_size: Per-topic rosbridge queue size.
        connect_timeout: Seconds to wait for the websocket handshake.
    """

    def __init__(self, url=None, rod_names=None, frame_id=DEFAULT_FRAME_ID,
                 topic_namespace=DEFAULT_TOPIC_NAMESPACE, stamp_source="wall",
                 twist_frame="body", position_scale=DEFAULT_POSITION_SCALE,
                 queue_size=10, connect_timeout=10.0):
        if stamp_source not in ("wall", "sim"):
            raise ValueError(
                f"stamp_source must be 'wall' or 'sim', got {stamp_source!r}"
            )
        if twist_frame not in ("body", "world"):
            raise ValueError(
                f"twist_frame must be 'body' or 'world', got {twist_frame!r}"
            )

        self.url = url or os.environ.get("ROSBRIDGE_URL", DEFAULT_ROSBRIDGE_URL)
        self.rod_names = list(rod_names) if rod_names is not None else None
        self.frame_id = frame_id
        self.topic_namespace = topic_namespace.rstrip("/")
        self.stamp_source = stamp_source
        self.twist_frame = twist_frame
        self.position_scale = float(position_scale)
        self.queue_size = queue_size
        self.connect_timeout = connect_timeout

        self._ros = None
        self._topics = {}

    # -- connection lifecycle ------------------------------------------------

    def connect(self):
        """Open the rosbridge websocket. Idempotent."""
        if self._ros is not None:
            return self

        import roslibpy  # lazy: keeps this module importable without roslibpy

        url = self.url
        if "://" not in url:
            url = "ws://" + url
        scheme, _, hostport = url.partition("://")
        host, _, port = hostport.partition(":")
        ros = roslibpy.Ros(
            host=host,
            port=int(port) if port else 9090,
            is_secure=(scheme == "wss"),
        )
        ros.run(timeout=self.connect_timeout)
        if not ros.is_connected:
            raise ConnectionError(f"could not connect to rosbridge at {self.url}")
        self._ros = ros
        return self

    def close(self):
        """Unadvertise topics and close the websocket. Idempotent."""
        for topic in self._topics.values():
            try:
                topic.unadvertise()
            except Exception:  # noqa: BLE001 - best effort during teardown
                pass
        self._topics.clear()
        if self._ros is not None:
            try:
                self._ros.terminate()
            finally:
                self._ros = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- publishing ----------------------------------------------------------

    def topic_name(self, rod_name):
        return f"{self.topic_namespace}/{rod_name}/odom"

    def _topic(self, rod_name):
        if rod_name not in self._topics:
            if self._ros is None:
                raise RuntimeError("not connected; call connect() first")
            import roslibpy

            topic = roslibpy.Topic(
                self._ros,
                self.topic_name(rod_name),
                ODOMETRY_MSG_TYPE,
                queue_size=self.queue_size,
            )
            topic.advertise()
            self._topics[rod_name] = topic
        return self._topics[rod_name]

    def _stamp(self, sim_time):
        return sim_time if self.stamp_source == "sim" else _time.time()

    def publish_rod_state(self, rod_name, time, pos, quat, linvel, angvel):
        """Publish one rod's state. ``time`` is the rollout's simulated time."""
        msg = build_odometry_msg(
            rod_name,
            self._stamp(time),
            pos,
            quat,
            linvel,
            angvel,
            frame_id=self.frame_id,
            twist_frame=self.twist_frame,
            position_scale=self.position_scale,
        )
        import roslibpy

        self._topic(rod_name).publish(roslibpy.Message(msg))
        return msg

    def publish_state(self, time, state):
        """Publish every rod in one flat EKF state vector.

        This is the hook ``run_ekf_rollout`` / ``OnlineEKF`` calls once per
        timestep.
        """
        rods = split_rod_states(state)
        if self.rod_names is None:
            self.rod_names = [f"rod_{i}" for i in range(len(rods))]
        if len(self.rod_names) != len(rods):
            raise ValueError(
                f"have {len(self.rod_names)} rod names but state holds "
                f"{len(rods)} rods"
            )
        return [
            self.publish_rod_state(name, time, pos, quat, linvel, angvel)
            for name, (pos, quat, linvel, angvel) in zip(self.rod_names, rods)
        ]
