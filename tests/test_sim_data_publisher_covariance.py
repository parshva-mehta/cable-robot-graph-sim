"""Unit tests for the covariance path added to sim_data_publisher.

Covers the quaternion -> 3D small-angle (so(3)) reduction that fills the
Odometry pose/twist covariances (the projection the v1 bridge left zeroed), the
covariance plumbing through the publisher hook, and the Float64MultiArray
side-channel that ships a full matrix (covariance now, Jacobian once plumbed).

Pure-Python + numpy; no ROS, torch, or gtsam required (roslibpy is mocked).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim_data_publisher as sdp

SQRT_HALF = math.sqrt(0.5)
QUAT_Z90 = [SQRT_HALF, 0.0, 0.0, SQRT_HALF]   # 90 deg about +z, (w, x, y, z)
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]
SCALE = sdp.DEFAULT_POSITION_SCALE


# -- fake roslibpy (mirrors test_sim_data_publisher) -------------------------

class _FakeTopic:
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
    def __init__(self):
        self.instances = []
        self.topics = []

    def Ros(self, host, port, is_secure=False):  # noqa: N802
        ros = _FakeRos(host, port, is_secure)
        self.instances.append(ros)
        return ros

    def Topic(self, ros, name, message_type, queue_size=100):  # noqa: N802
        topic = _FakeTopic(ros, name, message_type, queue_size)
        self.topics.append(topic)
        return topic

    @staticmethod
    def Message(msg):  # noqa: N802
        return msg


@pytest.fixture
def fake_roslibpy(monkeypatch):
    fake = _FakeRoslibpy()
    monkeypatch.setitem(sys.modules, "roslibpy", fake)
    return fake


def _rod_cov(pos_var=0.0, quat_vec_var=0.0, quat_w_var=0.0,
             linvel_var=0.0, angvel_var=0.0, pos_quat=0.0):
    """A 13x13 diagonal-ish covariance with per-block variances.

    ``pos_quat`` puts a cross term between every pos axis and the matching quat
    vector axis (indices 0<->4, 1<->5, 2<->6) so the pos-theta block is testable.
    """
    P = np.zeros((13, 13))
    for i in range(3):
        P[i, i] = pos_var
    P[3, 3] = quat_w_var
    for i in (4, 5, 6):
        P[i, i] = quat_vec_var
    for i in range(7, 10):
        P[i, i] = linvel_var
    for i in range(10, 13):
        P[i, i] = angvel_var
    if pos_quat:
        for pi, qi in ((0, 4), (1, 5), (2, 6)):
            P[pi, qi] = P[qi, pi] = pos_quat
    return P


# -- tangent basis -----------------------------------------------------------

def test_quat_tangent_basis_is_orthonormal_and_tangent():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    E = sdp._quat_tangent_basis(q)
    assert E.shape == (4, 3)
    assert np.allclose(E.T @ E, np.eye(3), atol=1e-12)   # orthonormal columns
    assert np.allclose(E.T @ q, np.zeros(3), atol=1e-12)  # tangent to q on S^3


def test_quat_tangent_basis_normalizes_input():
    E_unit = sdp._quat_tangent_basis(QUAT_Z90)
    E_scaled = sdp._quat_tangent_basis([2 * v for v in QUAT_Z90])
    assert np.allclose(E_unit, E_scaled, atol=1e-12)


def test_quat_tangent_basis_matches_linearization_where_torch_exists():
    pytest.importorskip("torch")  # linearization.py imports torch
    from linearization import _build_quat_E_matrix

    q = [0.5, -0.5, 0.5, 0.5]
    assert np.allclose(sdp._quat_tangent_basis(q), _build_quat_E_matrix(q), atol=1e-12)


def test_quat_tangent_basis_rejects_zero_quat():
    with pytest.raises(ValueError, match="zero quaternion"):
        sdp._quat_tangent_basis([0.0, 0.0, 0.0, 0.0])


# -- rod covariance projection ----------------------------------------------

def test_identity_quat_reduces_quat_variance_by_factor_four():
    """For q = identity, the vector-part quaternion variance v maps to a
    small-angle variance 4v (dtheta = 2 E.T dq)."""
    v = 0.01
    pose, twist = sdp.rod_covariance_to_ros(
        _rod_cov(pos_var=0.04, quat_vec_var=v, quat_w_var=0.5),
        IDENTITY_QUAT, position_scale=1.0, twist_frame="body",
    )
    pose = np.array(pose).reshape(6, 6)
    assert pose[0, 0] == pytest.approx(0.04)          # pos x variance passes through
    assert pose[3, 3] == pytest.approx(4 * v)         # small-angle x variance
    assert pose[4, 4] == pytest.approx(4 * v)
    assert pose[5, 5] == pytest.approx(4 * v)
    # The quaternion scalar (w) variance does not enter the tangent covariance.
    assert not np.isclose(pose[3, 3], 0.5)


def test_pose_and_twist_covariances_are_symmetric():
    P = _rod_cov(pos_var=0.02, quat_vec_var=0.01, linvel_var=0.03,
                 angvel_var=0.05, pos_quat=0.004)
    pose, twist = sdp.rod_covariance_to_ros(P, QUAT_Z90, position_scale=0.3)
    pose = np.array(pose).reshape(6, 6)
    twist = np.array(twist).reshape(6, 6)
    assert np.allclose(pose, pose.T, atol=1e-12)
    assert np.allclose(twist, twist.T, atol=1e-12)


def test_twist_block_is_the_state_twist_block_for_identity_quat():
    P = _rod_cov(linvel_var=0.09, angvel_var=0.16)
    _, twist = sdp.rod_covariance_to_ros(
        P, IDENTITY_QUAT, position_scale=1.0, twist_frame="world",
    )
    twist = np.array(twist).reshape(6, 6)
    assert np.allclose(np.diag(twist), [0.09, 0.09, 0.09, 0.16, 0.16, 0.16])


def test_pos_theta_cross_term_uses_two_E():
    """pos-theta block = 2 * Cov(pos, quat) @ E; for identity quat the cross var
    c on (pos_i, quat_vec_i) yields 2c on the (pos_i, theta_i) entry."""
    c = 0.004
    pose, _ = sdp.rod_covariance_to_ros(
        _rod_cov(pos_var=0.02, quat_vec_var=0.01, pos_quat=c),
        IDENTITY_QUAT, position_scale=1.0, twist_frame="body",
    )
    pose = np.array(pose).reshape(6, 6)
    assert pose[0, 3] == pytest.approx(2 * c)
    assert pose[1, 4] == pytest.approx(2 * c)
    assert pose[2, 5] == pytest.approx(2 * c)


def test_scaling_hits_length_dims_only():
    s = 0.11017
    P = _rod_cov(pos_var=1.0, quat_vec_var=1.0, linvel_var=1.0, angvel_var=1.0)
    pose, twist = sdp.rod_covariance_to_ros(
        P, IDENTITY_QUAT, position_scale=s, twist_frame="world",
    )
    pose = np.array(pose).reshape(6, 6)
    twist = np.array(twist).reshape(6, 6)
    assert pose[0, 0] == pytest.approx(s * s)   # position variance ~ s^2
    assert pose[3, 3] == pytest.approx(4.0)     # small-angle variance unscaled
    assert twist[0, 0] == pytest.approx(s * s)  # linear-velocity variance ~ s^2
    assert twist[3, 3] == pytest.approx(1.0)    # angular-rate variance unscaled


def test_body_frame_rotates_world_variance_into_body_axes():
    """World +y linear-velocity variance appears on body +x after a 90 deg z
    rotation (mirrors rotate_world_to_body: world +y -> body +x)."""
    P = _rod_cov(linvel_var=0.0)
    P[8, 8] = 0.5   # variance on world-y linear velocity (index 7+1)
    _, twist = sdp.rod_covariance_to_ros(
        P, QUAT_Z90, position_scale=1.0, twist_frame="body",
    )
    twist = np.array(twist).reshape(6, 6)
    assert twist[0, 0] == pytest.approx(0.5, abs=1e-12)   # now on body vx
    assert twist[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_world_frame_leaves_twist_unrotated():
    P = _rod_cov()
    P[8, 8] = 0.5
    _, twist = sdp.rod_covariance_to_ros(
        P, QUAT_Z90, position_scale=1.0, twist_frame="world",
    )
    twist = np.array(twist).reshape(6, 6)
    assert twist[1, 1] == pytest.approx(0.5, abs=1e-12)   # stays on world vy
    assert twist[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_rod_covariance_rejects_bad_twist_frame():
    with pytest.raises(ValueError, match="twist_frame"):
        sdp.rod_covariance_to_ros(_rod_cov(), IDENTITY_QUAT, twist_frame="inertial")


# -- split_rod_covariances ---------------------------------------------------

def test_split_rod_covariances_takes_diagonal_blocks():
    P = np.arange((3 * 13) ** 2, dtype=float).reshape(39, 39)
    blocks = sdp.split_rod_covariances(P, 3)
    assert len(blocks) == 3
    assert blocks[0].shape == (13, 13)
    assert np.array_equal(blocks[1], P[13:26, 13:26])
    # Cross-rod covariance is dropped.
    assert np.array_equal(blocks[2], P[26:39, 26:39])


# -- build_odometry_msg with covariance --------------------------------------

def test_build_odometry_msg_zeroes_covariance_by_default():
    msg = sdp.build_odometry_msg(
        "rod_01", 0.0, [0.0] * 3, IDENTITY_QUAT, [0.0] * 3, [0.0] * 3,
    )
    assert msg["pose"]["covariance"] == [0.0] * 36
    assert msg["twist"]["covariance"] == [0.0] * 36


def test_build_odometry_msg_fills_covariance_when_given():
    P = _rod_cov(pos_var=0.04, quat_vec_var=0.01, linvel_var=0.09, angvel_var=0.16)
    msg = sdp.build_odometry_msg(
        "rod_01", 0.0, [0.0] * 3, IDENTITY_QUAT, [0.0] * 3, [0.0] * 3,
        twist_frame="world", position_scale=1.0, rod_covariance=P,
    )
    pose = np.array(msg["pose"]["covariance"]).reshape(6, 6)
    twist = np.array(msg["twist"]["covariance"]).reshape(6, 6)
    assert pose[0, 0] == pytest.approx(0.04)
    assert pose[3, 3] == pytest.approx(0.04)   # 4 * 0.01
    assert twist[0, 0] == pytest.approx(0.09)
    assert twist[3, 3] == pytest.approx(0.16)


# -- covariance through the publisher hook -----------------------------------

def _synthetic_state(n_rods):
    return [r * 100 + i for r in range(n_rods) for i in range(13)]


def test_publish_state_without_covariance_keeps_zeros(fake_roslibpy):
    with sdp.RodStatePublisher(url="ws://localhost:9090",
                               rod_names=["rod_01"]) as pub:
        pub.publish_state(0.0, _synthetic_state(1))
    msg = fake_roslibpy.topics[0].published[0]
    assert msg["pose"]["covariance"] == [0.0] * 36
    assert msg["twist"]["covariance"] == [0.0] * 36


def test_publish_state_projects_per_rod_covariance(fake_roslibpy):
    cov = np.zeros((39, 39))
    # Give rod 1 a distinctive angular-velocity variance block.
    cov[13 + 10, 13 + 10] = 0.25
    with sdp.RodStatePublisher(
        url="ws://localhost:9090", rod_names=["rod_01", "rod_23", "rod_45"],
        twist_frame="world", position_scale=1.0,
    ) as pub:
        pub.publish_state(0.0, _synthetic_state(3), covariance=cov)

    rod1_twist = np.array(
        fake_roslibpy.topics[1].published[0]["twist"]["covariance"]
    ).reshape(6, 6)
    assert rod1_twist[3, 3] == pytest.approx(0.25)   # angular x variance
    # Rods 0 and 2 have zero covariance blocks.
    rod0_twist = fake_roslibpy.topics[0].published[0]["twist"]["covariance"]
    assert rod0_twist == [0.0] * 36


# -- Float64MultiArray side-channel ------------------------------------------

def test_build_float64_multiarray_encodes_shape_and_data():
    M = np.arange(6.0).reshape(2, 3)
    msg = sdp.build_float64_multiarray_msg(M, label="jacobian")
    assert msg["data"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    dims = msg["layout"]["dim"]
    assert dims[0]["label"] == "jacobian"
    assert (dims[0]["size"], dims[1]["size"]) == (2, 3)
    assert dims[0]["stride"] == 6 and dims[1]["stride"] == 3


def test_build_float64_multiarray_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2D"):
        sdp.build_float64_multiarray_msg(np.arange(4.0))


def test_matrix_stream_publisher_publishes_full_covariance(fake_roslibpy):
    cov = np.eye(39) * 0.7
    with sdp.MatrixStreamPublisher(url="ws://localhost:9090") as pub:
        pub.publish_state(0.0, _synthetic_state(3), covariance=cov)
    topic = fake_roslibpy.topics[0]
    assert topic.name == "/tensegrity/ekf/covariance"
    assert topic.msg_type == "std_msgs/Float64MultiArray"
    msg = topic.published[0]
    assert msg["layout"]["dim"][0]["size"] == 39
    assert np.array(msg["data"]).reshape(39, 39)[0, 0] == pytest.approx(0.7)


def test_matrix_stream_publisher_is_noop_without_covariance(fake_roslibpy):
    with sdp.MatrixStreamPublisher(url="ws://localhost:9090") as pub:
        assert pub.publish_state(0.0, _synthetic_state(3)) is None
    assert fake_roslibpy.topics[0].published == []


# -- CompositeSink forwarding ------------------------------------------------

def test_composite_sink_forwards_covariance_and_falls_back():
    seen = {}

    class _CovSink:
        def publish_state(self, time, state, covariance=None):
            seen["cov"] = covariance

    class _LegacySink:  # original 2-arg contract
        def __init__(self):
            self.calls = 0

        def publish_state(self, time, state):
            self.calls += 1

    cov_sink, legacy = _CovSink(), _LegacySink()
    sinks = sdp.CompositeSink(cov_sink, legacy)
    P = np.eye(39)
    sinks.publish_state(0.0, _synthetic_state(3), covariance=P)
    assert seen["cov"] is P          # covariance-aware sink got it
    assert legacy.calls == 1         # legacy sink still called (no TypeError)
