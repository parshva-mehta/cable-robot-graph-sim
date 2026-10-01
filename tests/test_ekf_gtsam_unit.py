"""Component-level unit tests for the GTSAM-based MEKF (`ekf_gtsam`).

These mirror the position-prediction fixes ported from the manual `ekf.py`
(see tests/test_ekf_unit.py) but target the GTSAM-only `OnlineEKF`:

  1. `test_measurement_canonicalization_ignores_axial_spin`
       Pins down that `ekf_gtsam` resolves `_pose_quat_to_exp` to the
       CANONICALIZED helper (principal-axis frame, axial spin discarded).
       A regression here puts the rotation innovation in the wrong frame and
       diverges the filter.

  2. `test_pose_flat_from_measurement_deinterleaves_full_state`
  3. `test_inject_fd_velocities_full_state_layout`
       Lock in the de-interleave fix in `OnlineEKF._pose_flat_from_measurement`.
       The old `[:7*n_rods]` slice silently read the wrong elements for rod >= 1
       of a 13-per-rod [pos, quat, linvel, angvel] full-state measurement,
       corrupting the finite-difference velocities for two of the three rods.

The tests are model-free: `OnlineEKF.__init__` only touches the simulator to
discover dtype/device, so a parameter-less stub triggers the CPU/float fallback.
They require `gtsam` (imported by `ekf_gtsam`) but no torch_geometric/GNN.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from ekf_gtsam import OnlineEKF, _pose_quat_to_exp, _fd_inject_velocities

_N_RODS = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _StubSim:
    """Parameter-free stand-in for the simulator.

    `OnlineEKF.__init__` probes `next(simulator.parameters())` only to discover
    dtype/device; an empty iterator triggers its `StopIteration` fallback
    (DEFAULT_DTYPE on CPU).  The noise/observation builders it calls are pure, so
    the wrapper constructs fully without a GNN.
    """

    def parameters(self):
        return iter(())


def _identity_quat_state(n_rods: int = _N_RODS) -> np.ndarray:
    """Flat (13*n_rods,) quat state: zero pos, identity quat [1,0,0,0], zero vel."""
    x = np.zeros(13 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        x[13 * r + 3] = 1.0  # qw = 1
    return x


def _hamilton(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for [w,x,y,z] quaternions."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


# ---------------------------------------------------------------------------
# 1. Measurement canonicalization discards axial spin (shared helper, via ekf_gtsam)
# ---------------------------------------------------------------------------

def test_measurement_canonicalization_ignores_axial_spin():
    """Two quats with the same principal axis but different axial spin map to the
    same exp-map measurement.

    Confirms `ekf_gtsam._pose_quat_to_exp` is the GNN-convention canonicalizing
    helper (compute_prin_axis → compute_quat_btwn_z_and_vec).  A regression here
    silently corrupts every innovation — it was a root cause of the NEES=521 bug.
    """
    n_rods = 1
    dev = torch.device("cpu")
    beta = 0.7
    q_base = np.array([np.cos(beta / 2), np.sin(beta / 2), 0.0, 0.0])  # rot about x
    pose_no_spin = np.array([0.1, 0.2, 0.3, *q_base], dtype=np.float64)
    exp_no = _pose_quat_to_exp(pose_no_spin, n_rods, torch.float64, dev)

    for alpha in [0.5, 1.3, 2.5]:
        # Rotation about the BODY z-axis leaves the principal (body-z) axis fixed.
        q_axial = np.array([np.cos(alpha / 2), 0.0, 0.0, np.sin(alpha / 2)])
        q_spun = _hamilton(q_base, q_axial)
        q_spun /= np.linalg.norm(q_spun)
        pose_spin = np.array([0.1, 0.2, 0.3, *q_spun], dtype=np.float64)
        exp_sp = _pose_quat_to_exp(pose_spin, n_rods, torch.float64, dev)

        assert np.allclose(exp_no[0:3], exp_sp[0:3], atol=1e-12), "position must be unchanged"
        assert np.allclose(exp_no[3:6], exp_sp[3:6], atol=1e-8), (
            f"axial spin α={alpha} changed exp_rot: {exp_no[3:6]} vs {exp_sp[3:6]}"
        )


# ---------------------------------------------------------------------------
# 2. De-interleave: full-state measurement → stride-7 pose
# ---------------------------------------------------------------------------

def test_pose_flat_from_measurement_deinterleaves_full_state():
    """`_pose_flat_from_measurement` extracts the correct [pos, quat] for every rod.

    A 13-per-rod [pos, quat, linvel, angvel] measurement is de-interleaved to a
    stride-7 [pos, quat] array.  The old `[:7*n_rods]` slice read velocity bytes
    of rod r-1 as the pose of rod r for r >= 1; this pins the per-rod offsets.
    """
    n = _N_RODS
    ekf = OnlineEKF(_StubSim(), dt=0.01, n_rods=n)

    # Distinct, non-overlapping values per field so a mis-offset is detectable.
    z_full = np.zeros(13 * n, dtype=np.float64)
    expected = np.zeros(7 * n, dtype=np.float64)
    for r in range(n):
        pos    = np.array([r + 0.1, r + 0.2, r + 0.3])
        quat   = np.array([0.5, 0.5, 0.5, 0.5]) + r          # arbitrary marker
        linvel = np.array([100.0 + r, 200.0 + r, 300.0 + r])  # must NOT leak in
        angvel = np.array([400.0 + r, 500.0 + r, 600.0 + r])
        z_full[13 * r:13 * r + 3]   = pos
        z_full[13 * r + 3:13 * r + 7] = quat
        z_full[13 * r + 7:13 * r + 10] = linvel
        z_full[13 * r + 10:13 * r + 13] = angvel
        expected[7 * r:7 * r + 3]     = pos
        expected[7 * r + 3:7 * r + 7] = quat

    out = ekf._pose_flat_from_measurement(z_full)
    assert out.shape == (7 * n,)
    assert np.array_equal(out, expected), f"de-interleave mismatch:\n{out}\nvs\n{expected}"
    # Guard: no velocity component leaked into the pose array.
    assert out.max() < 10.0, "velocity bytes leaked into the pose array (slice bug)"


def test_pose_flat_from_measurement_passthrough_pose_only():
    """A pose-only (7-per-rod) measurement is returned unchanged (copy)."""
    n = _N_RODS
    ekf = OnlineEKF(_StubSim(), dt=0.01, n_rods=n)
    z_pose = np.arange(7 * n, dtype=np.float64)
    out = ekf._pose_flat_from_measurement(z_pose)
    assert np.array_equal(out, z_pose)
    assert out is not z_pose, "must return a copy, not a view"


# ---------------------------------------------------------------------------
# 3. FD velocity injection through the full-state layout
# ---------------------------------------------------------------------------

def test_inject_fd_velocities_full_state_layout():
    """Full-state measurements yield the analytically expected linvel/angvel.

    Each rod has a distinct, nonzero motion so the old `[:7*n_rods]` slice (which
    mislabeled rod >= 1) would produce wrong velocities:
        rod 0: pure translation        → linvel [1, 2, 3],  angvel 0
        rod 1: rotation φ about world z → angvel [0, 0, φ/dt]
        rod 2: rotation ψ about world x + translation → linvel [0, -1, 0]
    """
    n = _N_RODS
    dt = 0.01
    ekf = OnlineEKF(_StubSim(), dt=dt, n_rods=n)

    def _full_from_pose(pose7):
        """Embed a stride-7 pose into a 13-per-rod full-state vector (zero vels)."""
        out = np.zeros(13 * n, dtype=np.float64)
        for r in range(n):
            out[13 * r:13 * r + 7] = pose7[7 * r:7 * r + 7]
        return out

    # Build stride-7 prev/curr poses, then embed into the interleaved layout.
    z_prev7 = np.zeros(7 * n, dtype=np.float64)
    z_curr7 = np.zeros(7 * n, dtype=np.float64)
    for r in range(n):
        z_prev7[7 * r + 3] = 1.0  # identity quat
        z_curr7[7 * r + 3] = 1.0

    # rod 0: pure translation → linvel [1, 2, 3]
    z_curr7[0:3] = [0.01, 0.02, 0.03]
    # rod 1: rotation φ about z
    phi = 0.1
    z_curr7[7 * 1 + 3:7 * 1 + 7] = [np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)]
    # rod 2: rotation ψ about x + translation → linvel [0, -1, 0]
    psi = 0.2
    z_curr7[7 * 2:7 * 2 + 3] = [0.0, -0.01, 0.0]
    z_curr7[7 * 2 + 3:7 * 2 + 7] = [np.cos(psi / 2), np.sin(psi / 2), 0.0, 0.0]

    z_prev_full = _full_from_pose(z_prev7)
    z_curr_full = _full_from_pose(z_curr7)

    x_in = _identity_quat_state(n)
    ekf._prev_z_quat = z_prev_full.copy()
    x_out = ekf._inject_fd_velocities(x_in, z_curr_full)

    # rod 0
    assert np.allclose(x_out[7:10], [1.0, 2.0, 3.0], atol=1e-9)
    assert np.allclose(x_out[10:13], [0.0, 0.0, 0.0], atol=1e-9)
    # rod 1  (would be corrupted by the [:7n] slice bug)
    assert np.allclose(x_out[13 + 7:13 + 10], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(x_out[13 + 10:13 + 13], [0.0, 0.0, phi / dt], atol=1e-7), (
        f"rod1 angvel = {x_out[13 + 10:13 + 13]}, expected [0,0,{phi / dt}]"
    )
    # rod 2  (would be corrupted by the [:7n] slice bug)
    assert np.allclose(x_out[26 + 7:26 + 10], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(x_out[26 + 10:26 + 13], [psi / dt, 0.0, 0.0], atol=1e-7), (
        f"rod2 angvel = {x_out[26 + 10:26 + 13]}, expected [{psi / dt},0,0]"
    )


def test_inject_fd_velocities_noop_without_prev():
    """With no previous measurement, the state is returned unchanged."""
    n = _N_RODS
    ekf = OnlineEKF(_StubSim(), dt=0.01, n_rods=n)
    x_in = _identity_quat_state(n)
    x_in[7:10] = [9.0, 9.0, 9.0]  # sentinel velocities
    out = ekf._inject_fd_velocities(x_in, np.zeros(13 * n))
    assert np.array_equal(out, x_in)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
