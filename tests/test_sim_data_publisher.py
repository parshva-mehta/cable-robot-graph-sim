"""Unit tests for sim_data_publisher (no ROS or roslibpy required).

Covers the two convention conversions the design doc flags as error-prone --
quaternion reordering and the world->body twist rotation -- plus the Odometry
field mapping, the position/velocity scaling (re-derived for THIS robot: 2.95
sim units -> 0.325 m), and the roslibpy publish path with a mocked client.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim_data_publisher as sdp

SQRT_HALF = math.sqrt(0.5)
# 90 degrees about +z, in the repo's (w, x, y, z) order.
QUAT_Z90 = [SQRT_HALF, 0.0, 0.0, SQRT_HALF]
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]

SCALE = sdp.DEFAULT_POSITION_SCALE  # 0.325 / 2.95 for this robot


# -- fake roslibpy -----------------------------------------------------------

class _FakeTopic:
    # Parameter names mirror roslibpy.Topic.__init__ (message_type, queue_size).
    def __init__(self, ros, name, message_type, queue_size=100):
        self.ros = ros
        self.name = name
        self.msg_type = message_type
        self.queue_size = queue_size
        self.advertised = False
        self.published = []

    def advertise(self):
        self.advertised = True

    def unadvertise(self):
        self.advertised = False

    def publish(self, msg):
        assert self.advertised, "published before advertise()"
        self.published.append(msg)


class _FakeRos:
    def __init__(self, host, port, is_secure=False):
        self.host = host
        self.port = port
        self.is_secure = is_secure
        self.is_connected = False
        self.terminated = False

    def run(self, timeout=None):
        self.is_connected = True

    def terminate(self):
        self.is_connected = False
        self.terminated = True


class _FakeRoslibpy:
    """Stands in for the roslibpy module in sys.modules."""

    def __init__(self):
        self.instances = []
        self.topics = []

    def Ros(self, host, port, is_secure=False):  # noqa: N802 - mirrors roslibpy
        ros = _FakeRos(host, port, is_secure)
        self.instances.append(ros)
        return ros

    def Topic(self, ros, name, message_type, queue_size=100):  # noqa: N802
        topic = _FakeTopic(ros, name, message_type, queue_size)
        self.topics.append(topic)
        return topic

    @staticmethod
    def Message(msg):  # noqa: N802 - roslibpy.Message is a dict subclass
        return msg


@pytest.fixture
def fake_roslibpy(monkeypatch):
    fake = _FakeRoslibpy()
    monkeypatch.setitem(sys.modules, "roslibpy", fake)
    return fake


# -- quaternion reorder ------------------------------------------------------

def test_quat_reorder_wxyz_to_xyzw():
    assert sdp.quat_wxyz_to_ros([1.0, 2.0, 3.0, 4.0]) == {
        "w": 1.0, "x": 2.0, "y": 3.0, "z": 4.0,
    }


def test_quat_reorder_rejects_wrong_length():
    with pytest.raises(ValueError, match="must have 4 elements"):
        sdp.quat_wxyz_to_ros([1.0, 0.0, 0.0])


# -- world -> body twist rotation -------------------------------------------

def test_rotate_world_to_body_identity_is_noop():
    out = sdp.rotate_world_to_body(IDENTITY_QUAT, [1.0, 2.0, 3.0])
    assert out == pytest.approx([1.0, 2.0, 3.0])


def test_rotate_world_to_body_z90():
    """R(q) maps body +x to world +y, so world +y maps back to body +x."""
    assert sdp.rotate_world_to_body(QUAT_Z90, [0.0, 1.0, 0.0]) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1e-12
    )
    assert sdp.rotate_world_to_body(QUAT_Z90, [1.0, 0.0, 0.0]) == pytest.approx(
        [0.0, -1.0, 0.0], abs=1e-12
    )
    # Rotation axis is unchanged by its own rotation.
    assert sdp.rotate_world_to_body(QUAT_Z90, [0.0, 0.0, 5.0]) == pytest.approx(
        [0.0, 0.0, 5.0], abs=1e-12
    )


def test_rotate_world_to_body_preserves_norm():
    quat = [0.5, 0.5, 0.5, 0.5]
    vec = [1.0, -2.0, 3.5]
    out = sdp.rotate_world_to_body(quat, vec)
    assert math.fsum(v * v for v in out) == pytest.approx(
        math.fsum(v * v for v in vec)
    )


def test_rotate_world_to_body_normalizes_input_quat():
    """A non-unit quaternion must not scale the vector."""
    scaled = [2 * q for q in QUAT_Z90]
    assert sdp.rotate_world_to_body(scaled, [0.0, 1.0, 0.0]) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1e-12
    )


def test_rotate_world_to_body_rejects_zero_quat():
    with pytest.raises(ValueError, match="zero quaternion"):
        sdp.rotate_world_to_body([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


def test_rotate_world_to_body_matches_torch_quaternion():
    """Agreement with the repo's own reference implementation, where torch exists."""
    torch = pytest.importorskip("torch")
    from utilities import torch_quaternion

    quat = [0.5, 0.5, 0.5, 0.5]
    vec = [1.0, -2.0, 3.5]
    q_t = torch.tensor(quat, dtype=torch.float64).reshape(1, 4, 1)
    v_t = torch.tensor(vec, dtype=torch.float64).reshape(1, 3, 1)
    expected = torch_quaternion.rotate_vec_quat(
        torch_quaternion.inverse_unit_quat(q_t), v_t
    ).flatten().tolist()

    assert sdp.rotate_world_to_body(quat, vec) == pytest.approx(expected, abs=1e-12)


# -- ROS time ----------------------------------------------------------------

def test_ros_time_from_seconds():
    assert sdp.ros_time_from_seconds(0.0) == {"secs": 0, "nsecs": 0}
    assert sdp.ros_time_from_seconds(1.5) == {"secs": 1, "nsecs": 500_000_000}
    assert sdp.ros_time_from_seconds(2.25) == {"secs": 2, "nsecs": 250_000_000}


def test_ros_time_rounding_carries_into_next_second():
    stamp = sdp.ros_time_from_seconds(1.9999999999)
    assert stamp == {"secs": 2, "nsecs": 0}


# -- Odometry field mapping --------------------------------------------------

def test_build_odometry_msg_field_mapping():
    msg = sdp.build_odometry_msg(
        "rod_01",
        stamp_seconds=1.5,
        pos=[1.0, 2.0, 3.0],
        quat=IDENTITY_QUAT,
        linvel=[0.1, 0.2, 0.3],
        angvel=[0.4, 0.5, 0.6],
        frame_id="world",
        twist_frame="world",
        position_scale=1.0,
    )

    assert msg["header"] == {
        "stamp": {"secs": 1, "nsecs": 500_000_000},
        "frame_id": "world",
    }
    assert msg["child_frame_id"] == "rod_01"
    assert msg["pose"]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert msg["pose"]["pose"]["orientation"] == {
        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
    }
    assert msg["twist"]["twist"]["linear"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert msg["twist"]["twist"]["angular"] == {"x": 0.4, "y": 0.5, "z": 0.6}
    # v1 leaves covariance zeroed; see the design doc's covariance note.
    assert msg["pose"]["covariance"] == [0.0] * 36
    assert msg["twist"]["covariance"] == [0.0] * 36


def test_build_odometry_msg_body_twist_rotates_but_pose_does_not():
    msg = sdp.build_odometry_msg(
        "rod_01",
        stamp_seconds=0.0,
        pos=[1.0, 2.0, 3.0],
        quat=QUAT_Z90,
        linvel=[0.0, 1.0, 0.0],
        angvel=[0.0, 0.0, 2.0],
        twist_frame="body",
        position_scale=1.0,
    )
    # Position stays world-frame.
    assert msg["pose"]["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    # Orientation is a plain reorder of the (w, x, y, z) input.
    assert msg["pose"]["pose"]["orientation"]["z"] == pytest.approx(SQRT_HALF)
    assert msg["pose"]["pose"]["orientation"]["w"] == pytest.approx(SQRT_HALF)
    # World +y linear velocity becomes body +x.
    lin = msg["twist"]["twist"]["linear"]
    assert (lin["x"], lin["y"], lin["z"]) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    ang = msg["twist"]["twist"]["angular"]
    assert (ang["x"], ang["y"], ang["z"]) == pytest.approx((0.0, 0.0, 2.0), abs=1e-12)


# -- position scaling (RE-DERIVED for this robot) ----------------------------

def test_default_position_scale_matches_ros_geometry():
    """This robot's rods are 2.95 sim units; the ROS side hardcodes 0.325 m
    endcaps. The factor is 0.325/2.95, NOT the reference repo's 0.1."""
    assert sdp.DEFAULT_POSITION_SCALE == pytest.approx(0.325 / 2.95)
    assert sdp.DEFAULT_POSITION_SCALE == pytest.approx(0.1101694915, abs=1e-9)
    # Guard against silently reverting to the reference repo's value.
    assert abs(sdp.DEFAULT_POSITION_SCALE - 0.1) > 1e-3


def test_odometry_scales_position_and_linvel_but_not_angvel():
    msg = sdp.build_odometry_msg(
        "rod_01",
        stamp_seconds=0.0,
        pos=[1.0, 2.0, 3.0],
        quat=IDENTITY_QUAT,
        linvel=[10.0, 20.0, 30.0],
        angvel=[0.4, 0.5, 0.6],
        twist_frame="world",
    )
    pos = msg["pose"]["pose"]["position"]
    assert (pos["x"], pos["y"], pos["z"]) == pytest.approx(
        (1.0 * SCALE, 2.0 * SCALE, 3.0 * SCALE)
    )
    # Linear velocity is a length per unit time, so it scales with position.
    lin = msg["twist"]["twist"]["linear"]
    assert (lin["x"], lin["y"], lin["z"]) == pytest.approx(
        (10.0 * SCALE, 20.0 * SCALE, 30.0 * SCALE)
    )
    # Angular velocity is rad/s -- scale-invariant.
    ang = msg["twist"]["twist"]["angular"]
    assert (ang["x"], ang["y"], ang["z"]) == pytest.approx((0.4, 0.5, 0.6))


def test_odometry_scaling_leaves_orientation_untouched():
    msg = sdp.build_odometry_msg(
        "rod_01", 0.0, [1.0, 2.0, 3.0], QUAT_Z90, [0.0] * 3, [0.0] * 3,
    )
    orient = msg["pose"]["pose"]["orientation"]
    assert orient["z"] == pytest.approx(SQRT_HALF)
    assert orient["w"] == pytest.approx(SQRT_HALF)


def test_scaled_rod_length_matches_ros_side_geometry():
    """A 2.95-unit rod must publish as 0.325 m apart, the length the ROS side
    hardcodes as its endcap offsets (+/-0.325/2). This is the whole point of the
    re-derived scale factor."""
    half = 2.95 / 2.0
    endpoints = []
    for sign in (+1, -1):
        msg = sdp.build_odometry_msg(
            "rod_01", 0.0, [0.0, 0.0, sign * half], IDENTITY_QUAT,
            [0.0] * 3, [0.0] * 3,
        )
        endpoints.append(msg["pose"]["pose"]["position"]["z"])
    assert abs(endpoints[0] - endpoints[1]) == pytest.approx(0.325)


def test_position_scale_one_publishes_raw_units():
    msg = sdp.build_odometry_msg(
        "rod_01", 0.0, [1.0, 2.0, 3.0], IDENTITY_QUAT, [4.0, 5.0, 6.0],
        [0.0] * 3, twist_frame="world", position_scale=1.0,
    )
    pos = msg["pose"]["pose"]["position"]
    assert (pos["x"], pos["y"], pos["z"]) == (1.0, 2.0, 3.0)
    lin = msg["twist"]["twist"]["linear"]
    assert (lin["x"], lin["y"], lin["z"]) == (4.0, 5.0, 6.0)


def test_scaling_commutes_with_body_frame_rotation():
    """Scaling then rotating must equal rotating then scaling (R is orthonormal)."""
    scaled = sdp.build_odometry_msg(
        "rod_01", 0.0, [0.0] * 3, QUAT_Z90, [0.0, 10.0, 0.0], [0.0] * 3,
        twist_frame="body", position_scale=0.1,
    )["twist"]["twist"]["linear"]
    raw = sdp.build_odometry_msg(
        "rod_01", 0.0, [0.0] * 3, QUAT_Z90, [0.0, 10.0, 0.0], [0.0] * 3,
        twist_frame="body", position_scale=1.0,
    )["twist"]["twist"]["linear"]
    assert scaled["x"] == pytest.approx(raw["x"] * 0.1)
    assert scaled["y"] == pytest.approx(raw["y"] * 0.1)
    assert scaled["z"] == pytest.approx(raw["z"] * 0.1)


def test_publisher_applies_position_scale(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090",
                               rod_names=["rod_01"]) as pub:
        pub.publish_state(0.0, _synthetic_state(1))
    pos = fake_roslibpy.topics[0].published[0]["pose"]["pose"]["position"]
    # rod 0 position block is [0, 1, 2] in the synthetic state.
    assert (pos["x"], pos["y"], pos["z"]) == pytest.approx(
        (0.0, 1.0 * SCALE, 2.0 * SCALE)
    )


def test_file_writer_is_not_scaled(tmp_path):
    """The ROS reader applies data_scale_factor itself; pre-scaling would
    double-convert."""
    path = tmp_path / "out.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        writer.publish_state(0.0, _synthetic_state(3))
    values = [float(t) for t in path.read_text().split()]
    assert values[0:3] == [0.0, 1.0, 2.0]
    assert values[13:16] == [100.0, 101.0, 102.0]


def test_build_odometry_msg_rejects_bad_twist_frame():
    with pytest.raises(ValueError, match="twist_frame"):
        sdp.build_odometry_msg(
            "rod_01", 0.0, [0.0] * 3, IDENTITY_QUAT, [0.0] * 3, [0.0] * 3,
            twist_frame="inertial",
        )


# -- state splitting ---------------------------------------------------------

def _synthetic_state(n_rods):
    """Flat state where rod r's block is r*100 + [0..12]."""
    return [r * 100 + i for r in range(n_rods) for i in range(13)]


def test_split_rod_states_slices_13_blocks():
    rods = sdp.split_rod_states(_synthetic_state(3))
    assert len(rods) == 3
    pos, quat, linvel, angvel = rods[1]
    assert pos == [100.0, 101.0, 102.0]
    assert quat == [103.0, 104.0, 105.0, 106.0]
    assert linvel == [107.0, 108.0, 109.0]
    assert angvel == [110.0, 111.0, 112.0]


def test_split_rod_states_rejects_ragged_length():
    with pytest.raises(ValueError, match="not a multiple of 13"):
        sdp.split_rod_states([0.0] * 20)


def test_split_rod_states_accepts_numpy_column_shape():
    """run_ekf_rollout passes state shaped (1, 13*n_rods, 1)."""
    np = pytest.importorskip("numpy")
    state = np.array(_synthetic_state(3), dtype=np.float64).reshape(1, 39, 1)
    rods = sdp.split_rod_states(state)
    assert len(rods) == 3
    assert rods[2][0] == [200.0, 201.0, 202.0]
    assert rods[0][3] == [10.0, 11.0, 12.0]


def test_split_rod_states_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    state = torch.tensor(_synthetic_state(2), dtype=torch.float64).reshape(1, 26, 1)
    rods = sdp.split_rod_states(state)
    assert len(rods) == 2
    assert rods[1][0] == [100.0, 101.0, 102.0]


# -- publisher over mocked roslibpy -----------------------------------------

def test_publisher_advertises_one_topic_per_rod(fake_roslibpy):
    pub = sdp.RodStatePublisher(
        url="ws://localhost:9090", rod_names=["rod_01", "rod_23", "rod_45"]
    )
    with pub:
        pub.publish_state(0.0, _synthetic_state(3))
        pub.publish_state(0.01, _synthetic_state(3))

    names = [t.name for t in fake_roslibpy.topics]
    assert names == [
        "/tensegrity/rod_01/odom",
        "/tensegrity/rod_23/odom",
        "/tensegrity/rod_45/odom",
    ]
    assert all(t.msg_type == "nav_msgs/Odometry" for t in fake_roslibpy.topics)
    # Topics are advertised once and reused across timesteps.
    assert [len(t.published) for t in fake_roslibpy.topics] == [2, 2, 2]


def test_publisher_parses_url_and_terminates(fake_roslibpy):
    with sdp.RodStatePublisher(url="wss://ros-noetic:9091", rod_names=["rod_01"]):
        pass
    ros = fake_roslibpy.instances[0]
    assert (ros.host, ros.port, ros.is_secure) == ("ros-noetic", 9091, True)
    assert ros.terminated


def test_publisher_defaults_url_from_env(monkeypatch):
    monkeypatch.setenv("ROSBRIDGE_URL", "ws://ros-noetic:9090")
    assert sdp.RodStatePublisher().url == "ws://ros-noetic:9090"


def test_publisher_default_url_without_env(monkeypatch):
    monkeypatch.delenv("ROSBRIDGE_URL", raising=False)
    assert sdp.RodStatePublisher().url == "ws://localhost:9090"


def test_publisher_derives_rod_names_when_unset(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090") as pub:
        pub.publish_state(0.0, _synthetic_state(2))
    assert pub.rod_names == ["rod_0", "rod_1"]


def test_publisher_rejects_rod_name_count_mismatch(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090",
                               rod_names=["rod_01"]) as pub:
        with pytest.raises(ValueError, match="have 1 rod names"):
            pub.publish_state(0.0, _synthetic_state(3))


def test_publisher_sim_stamp_uses_rollout_time(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"],
                               stamp_source="sim") as pub:
        pub.publish_state(2.5, _synthetic_state(1))
    msg = fake_roslibpy.topics[0].published[0]
    assert msg["header"]["stamp"] == {"secs": 2, "nsecs": 500_000_000}


def test_publisher_wall_stamp_ignores_rollout_time(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"],
                               stamp_source="wall") as pub:
        pub.publish_state(2.5, _synthetic_state(1))
    msg = fake_roslibpy.topics[0].published[0]
    # Wall clock is far past the rollout's 2.5s of simulated time.
    assert msg["header"]["stamp"]["secs"] > 1_600_000_000


def test_publisher_rejects_bad_stamp_source():
    with pytest.raises(ValueError, match="stamp_source"):
        sdp.RodStatePublisher(stamp_source="clock")


def test_publisher_custom_namespace_and_frame(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"],
                               topic_namespace="/est/", frame_id="map") as pub:
        pub.publish_state(0.0, _synthetic_state(1))
    assert fake_roslibpy.topics[0].name == "/est/rod_01/odom"
    assert fake_roslibpy.topics[0].published[0]["header"]["frame_id"] == "map"


def test_publish_before_connect_raises():
    pub = sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"])
    with pytest.raises(RuntimeError, match="not connected"):
        pub.publish_state(0.0, _synthetic_state(1))


def test_close_is_idempotent_and_unadvertises(fake_roslibpy):
    pub = sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"])
    pub.connect()
    pub.publish_state(0.0, _synthetic_state(1))
    pub.close()
    pub.close()
    assert not fake_roslibpy.topics[0].advertised


def test_connect_is_idempotent(fake_roslibpy):
    pub = sdp.RodStatePublisher(url="ws://localhost:9090", rod_names=["rod_01"])
    pub.connect()
    pub.connect()
    assert len(fake_roslibpy.instances) == 1
    pub.close()


# -- rollout state file writer ----------------------------------------------

def _parse_ros_side(line):
    """Reimplements interface/sim_data_publisher.py's get_values() slicing.

    That node parses each line as:
      0:3 PosA  3:7 quatA  7:10 VpA 10:13 VqA
     13:16 PosB 16:20 quatB 20:23 VpB 23:26 VqB
     26:29 PosC 29:33 quatC 33:36 VpC 36:39 VqC
    and its create_transform() reads the quaternion as (w, x, y, z).
    """
    v = [float(t) for t in line.split()]
    assert len(v) == 39, f"expected 39 columns, got {len(v)}"
    return {
        "red": {"pos": v[0:3], "quat": v[3:7]},
        "green": {"pos": v[13:16], "quat": v[16:20]},
        "blue": {"pos": v[26:29], "quat": v[29:33]},
    }


def test_file_writer_layout_matches_ros_reader(tmp_path):
    """Rod 0/1/2 must land where the ROS node reads red/green/blue."""
    path = tmp_path / "rollout_ekf.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        writer.publish_state(0.0, _synthetic_state(3))

    lines = path.read_text().splitlines()
    assert len(lines) == 1

    parsed = _parse_ros_side(lines[0])
    # rod 0 -> red, rod 1 -> green, rod 2 -> blue (id_red=0, id_green=1, id_blue=2)
    assert parsed["red"]["pos"] == [0.0, 1.0, 2.0]
    assert parsed["red"]["quat"] == [3.0, 4.0, 5.0, 6.0]
    assert parsed["green"]["pos"] == [100.0, 101.0, 102.0]
    assert parsed["green"]["quat"] == [103.0, 104.0, 105.0, 106.0]
    assert parsed["blue"]["pos"] == [200.0, 201.0, 202.0]
    assert parsed["blue"]["quat"] == [203.0, 204.0, 205.0, 206.0]


def test_file_writer_has_no_timestamp_column(tmp_path):
    """The reader expects PosA at column 0, so `time` must not be written."""
    path = tmp_path / "out.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        writer.publish_state(12.5, _synthetic_state(3))
    assert len(path.read_text().split()) == 39
    assert path.read_text().split()[0] == "0"


def test_file_writer_one_line_per_timestep(tmp_path):
    path = tmp_path / "out.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        for _ in range(5):
            writer.publish_state(0.0, _synthetic_state(3))
        assert writer.lines_written == 5
    lines = path.read_text().splitlines()
    assert len(lines) == 5
    assert all(len(line.split()) == 39 for line in lines)


def test_file_writer_round_trips_float64_exactly(tmp_path):
    path = tmp_path / "out.txt"
    state = [0.1 + i * 1e-16 for i in range(39)]
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        writer.publish_state(0.0, state)
    recovered = [float(t) for t in path.read_text().split()]
    assert recovered == state


def test_file_writer_rejects_wrong_rod_count(tmp_path):
    path = tmp_path / "out.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        with pytest.raises(ValueError, match="ROS reader expects 3"):
            writer.publish_state(0.0, _synthetic_state(6))


def test_file_writer_allows_other_rod_counts_when_opted_out(tmp_path):
    path = tmp_path / "out.txt"
    with sdp.RolloutStateFileWriter(str(path), expected_n_rods=None) as writer:
        writer.publish_state(0.0, _synthetic_state(6))
    assert len(path.read_text().split()) == 78


def test_file_writer_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deeper" / "out.txt"
    with sdp.RolloutStateFileWriter(str(path)) as writer:
        writer.publish_state(0.0, _synthetic_state(3))
    assert path.exists()


def test_file_writer_requires_open(tmp_path):
    writer = sdp.RolloutStateFileWriter(str(tmp_path / "out.txt"))
    with pytest.raises(RuntimeError, match="not open"):
        writer.publish_state(0.0, _synthetic_state(3))


def test_file_writer_close_is_idempotent(tmp_path):
    writer = sdp.RolloutStateFileWriter(str(tmp_path / "out.txt"))
    writer.open()
    writer.close()
    writer.close()


# -- composite sink ----------------------------------------------------------

def test_composite_sink_fans_out(tmp_path, fake_roslibpy):
    path = tmp_path / "out.txt"
    writer = sdp.RolloutStateFileWriter(str(path))
    pub = sdp.RodStatePublisher(url="ws://localhost:9090",
                               rod_names=["rod_01", "rod_23", "rod_45"])
    # CompositeSink.open() must start both: writer.open() and pub.connect().
    with sdp.CompositeSink(writer, pub) as sinks:
        sinks.publish_state(0.0, _synthetic_state(3))

    assert len(path.read_text().splitlines()) == 1
    assert [len(t.published) for t in fake_roslibpy.topics] == [1, 1, 1]
    assert fake_roslibpy.instances[0].terminated


def test_composite_sink_closes_all(tmp_path):
    class _Sink:
        def __init__(self):
            self.closed = False
            self.states = []

        def publish_state(self, time, state):
            self.states.append((time, state))

        def close(self):
            self.closed = True

    a, b = _Sink(), _Sink()
    with sdp.CompositeSink(a, b) as sinks:
        sinks.publish_state(1.0, _synthetic_state(3))
    assert len(a.states) == len(b.states) == 1
    assert a.closed and b.closed


def test_rod_names_from_simulator():
    class _Robot:
        # `rods` is a property aliasing `rigid_bodies` in robots/tensegrity.py.
        rods = {"rod_01": object(), "rod_23": object(), "rod_45": object()}

    class _Sim:
        robot = _Robot()

    assert sdp.rod_names_from_simulator(_Sim()) == ["rod_01", "rod_23", "rod_45"]
