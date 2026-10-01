"""The EKF -> publisher hooks fire on every frame, for BOTH hook points.

This repo has two hook points, unlike the reference repo:
  * ``run_ekf_rollout`` -- batch rollout, and
  * ``OnlineEKF``       -- streaming wrapper.

Both must call ``publisher.publish_state(time, state)`` once for the initial
state and once per timestep, and must be unchanged when ``publisher=None``.

These tests drive the REAL EKF math (gtsam predict/update + the quaternion
tangent-space linearization in ``linearization.py``) against a lightweight stub
simulator (identity dynamics), so no trained ``.pt`` model or dataset is needed.
Skipped where torch/gtsam are unavailable.
"""

import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

pytest.importorskip("torch")
pytest.importorskip("gtsam")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ekf_gtsam import OnlineEKF, run_ekf_rollout  # noqa: E402
from e2e_check import build_stub_simulator, synthetic_data  # noqa: E402

CONFIG = str(_REPO / "simulators" / "configs" / "3_bar_gnn_sim_config.json")


class RecordingSink:
    """Duck-typed publish_state sink that records every (time, flat_state).

    Uses the original two-argument signature on purpose, so the EKF's
    ``_publish_state`` fallback (kwarg -> positional) stays covered."""

    def __init__(self):
        self.calls = []

    def publish_state(self, time, state):
        flat = state.detach().cpu().numpy().reshape(-1) if hasattr(state, "detach") \
            else np.asarray(state).reshape(-1)
        self.calls.append((time, flat))


class CovRecordingSink:
    """publish_state sink that records the covariance and Jacobian the EKF passes."""

    def __init__(self):
        self.covs = []
        self.jacs = []

    def publish_state(self, time, state, covariance=None, jacobian=None):
        self.covs.append(covariance)
        self.jacs.append(jacobian)


@pytest.fixture
def sim():
    return build_stub_simulator(CONFIG)


def _start_state(gt0, n_rods):
    vals = []
    for r in range(n_rods):
        vals += (gt0["pos"][r * 3:(r + 1) * 3]
                 + gt0["quat"][r * 4:(r + 1) * 4]
                 + gt0["linvel"][r * 3:(r + 1) * 3]
                 + gt0["angvel"][r * 3:(r + 1) * 3])
    return torch.tensor(vals, dtype=torch.float32).reshape(1, -1, 1)


def _z(gt_t, n_rods):
    pos = np.array(gt_t["pos"], np.float64).reshape(-1, 3)
    quat = np.array(gt_t["quat"], np.float64).reshape(-1, 4)
    lv = np.array(gt_t["linvel"], np.float64).reshape(-1, 3)
    av = np.array(gt_t["angvel"], np.float64).reshape(-1, 3)
    return np.hstack([pos, quat, lv, av]).reshape(-1)


# -- run_ekf_rollout ---------------------------------------------------------

def test_run_ekf_rollout_publishes_every_frame(sim):
    n_steps = 4
    gt, extra = synthetic_data(CONFIG, sim, n_steps)
    sink = RecordingSink()
    frames = run_ekf_rollout(sim, gt, extra, dt=0.01,
                             use_finite_diff=True, publisher=sink)
    # Initial state + one per step.
    assert len(frames) == n_steps + 1
    assert len(sink.calls) == len(frames)
    # First publish is the initial state at t=0.
    assert sink.calls[0][0] == pytest.approx(0.0)
    # Times advance by dt.
    assert sink.calls[-1][0] == pytest.approx(n_steps * 0.01)
    # Each published state is 39-dim (3 rods x 13).
    assert all(c[1].size == 39 for c in sink.calls)


def test_run_ekf_rollout_passes_covariance(sim):
    gt, extra = synthetic_data(CONFIG, sim, 3)
    sink = CovRecordingSink()
    run_ekf_rollout(sim, gt, extra, dt=0.01, use_finite_diff=True, publisher=sink)
    # A covariance is recorded for every frame; whenever present it is the
    # filter's own 36x36 (3 rods x 12) exp-map tangent covariance.
    assert len(sink.covs) == 4
    present = [c for c in sink.covs if c is not None]
    assert present, "expected at least one non-None covariance"
    assert all(np.asarray(c).shape == (36, 36) for c in present)


def test_run_ekf_rollout_passes_tangent_jacobian(sim):
    gt, extra = synthetic_data(CONFIG, sim, 3)
    sink = CovRecordingSink()
    run_ekf_rollout(sim, gt, extra, dt=0.01, use_finite_diff=True, publisher=sink)
    present = [j for j in sink.jacs if j is not None]
    assert present, "expected at least one tangent Jacobian"
    # Minimal tangent form: 12 per rod (quaternion 4 -> small-angle 3).
    assert all(np.asarray(j).shape == (36, 36) for j in present)
    assert all(np.all(np.isfinite(j)) for j in present)


def test_run_ekf_rollout_unchanged_when_publisher_none(sim):
    n_steps = 3
    gt, extra = synthetic_data(CONFIG, sim, n_steps)
    frames = run_ekf_rollout(sim, gt, extra, dt=0.01,
                             use_finite_diff=True, publisher=None)
    assert len(frames) == n_steps + 1  # no crash, no behavior change


def test_run_ekf_rollout_published_state_matches_frame(sim):
    gt, extra = synthetic_data(CONFIG, sim, 3)
    sink = RecordingSink()
    frames = run_ekf_rollout(sim, gt, extra, dt=0.01,
                             use_finite_diff=True, publisher=sink)
    # Frames carry the 36-D exp-map state; the sink is published the 39-D
    # ambient quat state. They must describe the same pose.
    from linearization_exp import quat_state_to_exp_state
    for frame, (_, published) in zip(frames, sink.calls):
        published_exp = quat_state_to_exp_state(
            torch.as_tensor(published, dtype=torch.float32).reshape(1, -1, 1)
        ).detach().cpu().numpy().reshape(-1)
        assert published_exp == pytest.approx(
            frame["state"].detach().cpu().numpy().reshape(-1), abs=1e-5
        )


# -- OnlineEKF ---------------------------------------------------------------

def test_online_ekf_publishes_every_frame(sim):
    n_steps = 4
    gt, extra = synthetic_data(CONFIG, sim, n_steps)
    n_rods = len(sim.robot.rods)
    sink = RecordingSink()

    ekf = OnlineEKF(sim, dt=0.01, n_rods=n_rods, use_finite_diff=True,
                    publisher=sink)
    ekf.initialize(_start_state(gt[0], n_rods),
                   rest_lengths=extra[0]["rest_lengths"],
                   motor_speeds=extra[0]["motor_speeds"])
    # initialize publishes the initial state at t=0.
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == pytest.approx(0.0)

    for k, ex in enumerate(extra):
        have_meas = k + 1 < len(gt)
        z = _z(gt[k + 1], n_rods) if have_meas else None
        ekf.step(z_t=z, u_t=ex["controls"], have_measurement=have_meas)

    # 1 (init) + n_steps.
    assert len(sink.calls) == n_steps + 1
    assert sink.calls[-1][0] == pytest.approx(n_steps * 0.01)
    assert all(c[1].size == 39 for c in sink.calls)


def test_online_ekf_passes_covariance(sim):
    gt, extra = synthetic_data(CONFIG, sim, 3)
    n_rods = len(sim.robot.rods)
    sink = CovRecordingSink()
    ekf = OnlineEKF(sim, dt=0.01, n_rods=n_rods, use_finite_diff=True,
                    publisher=sink)
    ekf.initialize(_start_state(gt[0], n_rods),
                   rest_lengths=extra[0]["rest_lengths"],
                   motor_speeds=extra[0]["motor_speeds"])
    for k, ex in enumerate(extra):
        have_meas = k + 1 < len(gt)
        z = _z(gt[k + 1], n_rods) if have_meas else None
        ekf.step(z_t=z, u_t=ex["controls"], have_measurement=have_meas)
    present = [c for c in sink.covs if c is not None]
    assert present, "expected at least one non-None covariance"
    assert all(np.asarray(c).shape == (36, 36) for c in present)


def test_quat2exp_canonicalizes_antipodal_quaternion():
    """q and -q are the same rotation and must map to the same exp-map vector.

    The GTSAM/exp-map filter relies on ``quat2exp`` folding the double cover at
    the conversion boundary; without it, antipodal-but-identical measurements
    differ by ~2*pi and produce a spurious innovation.
    """
    from utilities.torch_quaternion import quat2exp

    # identity, a 5-degree rotation, and one near 180 degrees (the hard case)
    for angle_deg in (0.0, 5.0, 179.0):
        half = math.radians(angle_deg) / 2.0
        q = torch.tensor([[[math.cos(half)], [math.sin(half)], [0.0], [0.0]]],
                         dtype=torch.float64)
        e_pos = quat2exp(q)
        e_neg = quat2exp(-q)
        assert e_pos.shape == (1, 3, 1)
        np.testing.assert_allclose(
            e_neg.numpy(), e_pos.numpy(), atol=1e-12,
            err_msg=f"double cover not folded at {angle_deg} deg")
        # the canonical exp-map norm stays within [0, pi]
        assert float(e_pos.norm()) <= math.pi + 1e-9


def test_double_cover_measurement_does_not_flip_estimate(sim):
    """An antipodal-but-identical orientation measurement must not drag the
    filtered quaternion toward -q. Without the double-cover fold in
    ``quat2exp`` this produces a large spurious innovation."""
    gt, extra = synthetic_data(CONFIG, sim, 1)
    n_rods = len(sim.robot.rods)
    start = _start_state(gt[0], n_rods)

    ekf = OnlineEKF(sim, dt=0.01, n_rods=n_rods, use_finite_diff=True)
    ekf.initialize(start, rest_lengths=extra[0]["rest_lengths"],
                   motor_speeds=extra[0]["motor_speeds"])

    # Measurement = the start state, but every rod's quaternion negated (same
    # physical orientation, opposite hemisphere).
    z = start.detach().cpu().numpy().reshape(-1).astype(float).copy()
    for r in range(n_rods):
        z[13 * r + 3: 13 * r + 7] *= -1.0
    # step() returns the 36-D exp-map state; convert back to quats to compare.
    from linearization_exp import exp_state_to_quat_state
    out_exp = ekf.step(z_t=z, u_t=extra[0]["controls"])
    out = exp_state_to_quat_state(out_exp).detach().cpu().numpy().reshape(-1)

    for r in range(n_rods):
        q_pred = start.detach().cpu().numpy().reshape(-1)[13 * r + 3: 13 * r + 7]
        q_out = out[13 * r + 3: 13 * r + 7]
        # Stayed in the predicted hemisphere (did not flip toward -q)...
        assert float(np.dot(q_out, q_pred)) > 0.9
        # ...and barely moved (the antipodal measurement carried no real error).
        assert float(np.linalg.norm(q_out - q_pred)) < 0.1


def test_online_ekf_unchanged_when_publisher_none(sim):
    gt, extra = synthetic_data(CONFIG, sim, 2)
    n_rods = len(sim.robot.rods)
    ekf = OnlineEKF(sim, dt=0.01, n_rods=n_rods, use_finite_diff=True)
    ekf.initialize(_start_state(gt[0], n_rods),
                   rest_lengths=extra[0]["rest_lengths"],
                   motor_speeds=extra[0]["motor_speeds"])
    out = ekf.step(z_t=_z(gt[1], n_rods), u_t=extra[0]["controls"])
    assert out.shape == (1, 36, 1)  # no crash, returns an exp-map state
