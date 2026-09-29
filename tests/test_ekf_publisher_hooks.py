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

from ekf import OnlineEKF, run_ekf_rollout  # noqa: E402
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
    # A covariance is recorded for every frame; whenever present it is the full
    # 39x39 (3 rods x 13) EKF covariance.
    assert len(sink.covs) == 4
    present = [c for c in sink.covs if c is not None]
    assert present, "expected at least one non-None covariance"
    assert all(np.asarray(c).shape == (39, 39) for c in present)


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
    for frame, (_, published) in zip(frames, sink.calls):
        assert published == pytest.approx(
            frame["state"].detach().cpu().numpy().reshape(-1)
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
    assert all(np.asarray(c).shape == (39, 39) for c in present)


def test_hemisphere_align_flips_antipodal_quaternion():
    from ekf import _hemisphere_align_measurement

    n_rods = 2
    # Predicted state: identity quats. Measurement (full state, 13/rod): rod 0 is
    # antipodal (-identity, same rotation), rod 1 already aligned.
    mean_pred = np.zeros(13 * n_rods)
    for r in range(n_rods):
        mean_pred[13 * r + 3] = 1.0  # w = 1
    z = np.zeros(13 * n_rods)
    z[3] = -1.0            # rod 0 quat = -identity (antipodal)
    z[13 + 3] = 1.0        # rod 1 quat = +identity (aligned)
    out = _hemisphere_align_measurement(z, mean_pred, n_rods)
    assert out[3] == pytest.approx(1.0)          # rod 0 flipped to +identity
    assert out[13 + 3] == pytest.approx(1.0)     # rod 1 untouched
    # Pose-only layout (7/rod): quaternion still at offset 3.
    zp = np.zeros(7 * n_rods)
    zp[3] = -1.0
    outp = _hemisphere_align_measurement(zp, mean_pred, n_rods)
    assert outp[3] == pytest.approx(1.0)


def test_double_cover_measurement_does_not_flip_estimate(sim):
    """An antipodal-but-identical orientation measurement must not drag the
    filtered quaternion toward -q. Before the hemisphere-alignment fix this
    produced a large spurious innovation."""
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
    out = ekf.step(z_t=z, u_t=extra[0]["controls"]).detach().cpu().numpy().reshape(-1)

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
    assert out.shape == (1, 39, 1)  # no crash, returns a state
