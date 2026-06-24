"""Unit tests for each MEKF component.

Tests are split into two groups:

Algebraic / unit (no model required) — fast, always run, and now exercise the
REAL `_ekf_step` code path (via a parameter-free `_StubSim` + the
`x_pred_quat_np` hook that skips the GNN call), not a re-implementation of the
Kalman equations inside the test:
  test_retraction_zero_delta            zero delta leaves state unchanged
  test_retraction_known_rotation        known rotation delta produces expected quat
  test_make_pd                          symmetrizes + clamps eigenvalues, no-op on PD
  test_predict_covariance_grows         _ekf_step predict: P_pred = F P Fᵀ + Q (grows)
  test_update_covariance_shrinks        _ekf_step update: Joseph form reduces trace(P)
  test_update_innovation_zero           z = H @ x_pred  →  posterior mean == prediction
  test_innovation_gate                  absurd measurement is rejected, mean unchanged
  test_fd_velocity_injection            known consecutive poses → known FD linvel/angvel
  test_measurement_canonicalization_ignores_axial_spin
                                        axial spin must not change the exp-map measurement
  test_exp_quat_roundtrip               quat → exp → quat is identity (qw ≥ 0)

Model-dependent (skip if model/data absent) — require simulator assets:
  run_predict_only_matches_gnn          EKF with no measurements replicates GNN rollout
  run_update_corrects_toward_truth      EKF RMSE < predict-only AND near the noise floor;
                                        covariance stays symmetric PSD throughout
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import numpy as np
import torch
import pytest

# `ekf` and `linearization_exp` import cleanly without torch_geometric, so the
# model-free tests below can run in any environment with numpy + torch.  The
# simulator import is deferred into `_load_assets` (model-dependent tests only).
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
)
from ekf import (
    _apply_exp_correction,
    _make_pd,
    _ekf_step,
    _ensure_ctrl_for_step,
    _structured_Q_sigmas,
    _structured_R_sigmas,
    _fd_inject_velocities,
    _pose_quat_to_exp,
)
from utilities.misc_utils import DEFAULT_DTYPE

_DEFAULT_MODEL = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/models/best_rollout_model.pt"
)
_DEFAULT_DATA = (
    "/Users/parshvamehta/PRACSYS/cablegraphrobot/tensegrity/data_sets/"
    "3bar_new_platform_high_friction/dataset_0/traj_6"
)

_N_RODS = 3
_EXP_DIM = EXP_STATE_DIM  # 36


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity_quat_state(n_rods: int = _N_RODS) -> np.ndarray:
    """Flat (13*n_rods,) quat state: zero pos, identity quat [1,0,0,0], zero vel."""
    x = np.zeros(13 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        x[13 * r + 3] = 1.0  # qw = 1
    return x


class _StubSim:
    """Parameter-free stand-in for the simulator.

    `_ekf_step` probes `next(simulator.parameters())` only to discover dtype /
    device; an empty iterator triggers its `StopIteration` fallback (DEFAULT_DTYPE
    on CPU).  When `x_pred_quat_np` is supplied to `_ekf_step`, the GNN step and
    the LSTM sync are both skipped, so the real predict / update covariance code
    runs with no model — letting these tests check production arithmetic instead
    of a re-derivation of it.
    """

    def parameters(self):
        return iter(())


def _sample_quat_state(seed: int, n_rods: int = _N_RODS) -> np.ndarray:
    """Deterministic non-trivial (13*n_rods,) quat state with normalized qw≥0 quats."""
    rng = np.random.default_rng(seed)
    x = np.zeros(13 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        x[13 * r: 13 * r + 3] = rng.normal(0.0, 0.1, 3)
        q = rng.normal(0.0, 1.0, 4)
        if q[0] < 0:
            q = -q
        q /= np.linalg.norm(q)
        x[13 * r + 3: 13 * r + 7] = q
        x[13 * r + 7: 13 * r + 13] = rng.normal(0.0, 0.1, 6)
    return x


def _quat_to_exp_np(x_quat_np: np.ndarray) -> np.ndarray:
    """Exp-map of a quat state using the SAME conversion `_ekf_step` uses internally.

    Uses DEFAULT_DTYPE (float32) so a measurement built as `H @ this` produces a
    bit-exact zero innovation inside `_ekf_step`.
    """
    t = torch.tensor(x_quat_np, dtype=DEFAULT_DTYPE).reshape(1, -1, 1)
    return quat_state_to_exp_state(t)[0, :, 0].cpu().numpy().astype(np.float64)


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
# 1. Retraction correctness
# ---------------------------------------------------------------------------

def test_retraction_zero_delta():
    """Zero correction must leave every component of the state unchanged."""
    x = _identity_quat_state()
    delta = np.zeros(_EXP_DIM, dtype=np.float64)
    result = _apply_exp_correction(x, delta, _N_RODS, torch.float32)
    assert np.allclose(result, x, atol=1e-12), (
        f"max diff = {np.max(np.abs(result - x)):.2e}"
    )


def test_retraction_known_rotation():
    """Applying a π/2 rotation around z to identity quat gives (√2/2, 0, 0, √2/2).

    The exp-map vector [0, 0, π/2] encodes a rotation of angle π/2 around z.
    exp2quat([0,0,π/2]) = [cos(π/4), 0, 0, sin(π/4)] = [√2/2, 0, 0, √2/2].
    Left-multiplying identity quat by this delta gives the delta itself.
    Rods 1 and 2 must be unchanged.
    """
    x = _identity_quat_state()
    delta = np.zeros(_EXP_DIM, dtype=np.float64)
    angle = np.pi / 2.0
    delta[EXP_BLOCK_SIZE * 0 + 3: EXP_BLOCK_SIZE * 0 + 6] = [0.0, 0.0, angle]

    result = _apply_exp_correction(x, delta, _N_RODS, torch.float32)

    s2 = np.sqrt(2.0) / 2.0
    expected_quat = np.array([s2, 0.0, 0.0, s2], dtype=np.float64)
    got_quat = result[3:7]
    assert np.allclose(got_quat, expected_quat, atol=1e-10), (
        f"expected {expected_quat}, got {got_quat}"
    )
    # Other rods unchanged
    for r in range(1, _N_RODS):
        qb = 13 * r
        assert np.allclose(result[qb: qb + 13], x[qb: qb + 13], atol=1e-12), (
            f"rod {r} should be unchanged"
        )


# ---------------------------------------------------------------------------
# 2. _make_pd: symmetrize + eigenvalue clamp
# ---------------------------------------------------------------------------

def test_make_pd():
    """_make_pd symmetrizes, clamps negative eigenvalues, and is a no-op on PD input."""
    # Symmetrizes an asymmetric matrix.
    M = np.array([[1.0, 2.0], [0.0, 1.0]])
    R = _make_pd(M, min_eig=1e-9)
    assert np.allclose(R, R.T), "result must be symmetric"

    # Clamps a negative eigenvalue up to min_eig.
    A = np.diag([1.0, -0.5, 2.0])
    R = _make_pd(A, min_eig=1e-6)
    assert np.linalg.eigvalsh(R).min() >= 1e-6 - 1e-12, (
        f"min eigenvalue {np.linalg.eigvalsh(R).min():.2e} not clamped to 1e-6"
    )

    # Near no-op on an already-PD matrix.
    P = np.diag([0.3, 1.2, 0.7])
    R = _make_pd(P, min_eig=1e-9)
    assert np.allclose(R, P, atol=1e-12), "should not perturb an already-PD matrix"


# ---------------------------------------------------------------------------
# 3. Predict step: covariance must grow and F must participate (real _ekf_step)
# ---------------------------------------------------------------------------

def test_predict_covariance_grows():
    """`_ekf_step` predict path returns P_pred = make_pd(F P Fᵀ + Q), which grows.

    Driven through the REAL `_ekf_step` (have_measurement=False, x_pred supplied)
    so a regression in the predict-covariance arithmetic would fail here.
    """
    sim = _StubSim()
    x = _sample_quat_state(seed=1)
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, _N_RODS, base_sigma=0.01)
    Q = np.diag(Q_sigmas ** 2)
    H_dummy = np.zeros((6 * _N_RODS, _EXP_DIM))   # unused (no measurement)
    R_dummy = np.ones(6 * _N_RODS)

    # Case 1: F = I — iterate; trace must grow and match make_pd(P + Q) each step.
    P = np.eye(_EXP_DIM) * 0.01
    F = np.eye(_EXP_DIM)
    prev_trace = np.trace(P)
    for step in range(5):
        _, P_pred = _ekf_step(
            x, P, sim, None, F, H_dummy, None,
            Q_sigmas, R_dummy, _N_RODS,
            have_measurement=False, x_pred_quat_np=x,
        )
        assert np.trace(P_pred) > prev_trace, (
            f"step {step}: trace should grow ({prev_trace:.6f} → {np.trace(P_pred):.6f})"
        )
        assert np.allclose(P_pred, _make_pd(F @ P @ F.T + Q)), (
            f"step {step}: P_pred does not match make_pd(F P Fᵀ + Q)"
        )
        prev_trace = np.trace(P_pred)
        P = P_pred

    # Case 2: non-identity F (spectral radius 1.1) must actually scale F P Fᵀ.
    P = np.eye(_EXP_DIM) * 0.01
    F = np.eye(_EXP_DIM) * 1.1
    _, P_pred = _ekf_step(
        x, P, sim, None, F, H_dummy, None,
        Q_sigmas, R_dummy, _N_RODS,
        have_measurement=False, x_pred_quat_np=x,
    )
    assert np.allclose(P_pred, _make_pd(F @ P @ F.T + Q)), (
        "non-identity F not applied as F P Fᵀ"
    )
    assert np.trace(P_pred) > np.trace(_make_pd(P + Q)), (
        "F with spectral radius 1.1 should inflate P more than F = I"
    )


# ---------------------------------------------------------------------------
# 4. Update step: covariance shrinks; perfect measurement leaves the mean fixed
# ---------------------------------------------------------------------------

def test_update_covariance_shrinks():
    """`_ekf_step` Joseph-form update with full H reduces trace(P) and keeps P PSD."""
    sim = _StubSim()
    x_pred = _sample_quat_state(seed=2)
    x_pred_exp = _quat_to_exp_np(x_pred)

    H = np.eye(_EXP_DIM)
    z = H @ x_pred_exp                       # zero innovation → no inflation / gating
    R_sigmas = np.ones(_EXP_DIM) * 0.1
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, _N_RODS, base_sigma=1e-3)
    Q = np.diag(Q_sigmas ** 2)
    F = np.eye(_EXP_DIM)
    P = np.eye(_EXP_DIM)

    _, P_post = _ekf_step(
        x_pred, P, sim, None, F, H, z,
        Q_sigmas, R_sigmas, _N_RODS,
        have_measurement=True, x_pred_quat_np=x_pred,
    )

    P_pred = _make_pd(F @ P @ F.T + Q)
    assert np.trace(P_post) < np.trace(P_pred), (
        f"trace: {np.trace(P_pred):.4f} → {np.trace(P_post):.4f} (should shrink)"
    )
    assert np.allclose(P_post, P_post.T, atol=1e-10), "P_post must be symmetric"
    min_eig = np.linalg.eigvalsh(P_post).min()
    assert min_eig >= 0, f"P_post not PSD; min eigenvalue = {min_eig:.2e}"


def test_update_innovation_zero():
    """When z = H @ x_pred the posterior mean equals the prediction (no correction)."""
    sim = _StubSim()
    x_pred = _sample_quat_state(seed=3)
    x_pred_exp = _quat_to_exp_np(x_pred)

    H = np.eye(_EXP_DIM)
    z = H @ x_pred_exp                       # perfect measurement → innovation vanishes
    R_sigmas = np.ones(_EXP_DIM) * 0.1
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, _N_RODS, base_sigma=0.5)
    F = np.eye(_EXP_DIM)
    P = np.eye(_EXP_DIM) * 0.5

    x_post, _ = _ekf_step(
        x_pred, P, sim, None, F, H, z,
        Q_sigmas, R_sigmas, _N_RODS,
        have_measurement=True, x_pred_quat_np=x_pred,
    )
    assert np.allclose(x_post, x_pred, atol=1e-12), (
        f"posterior mean should equal prediction; max diff = "
        f"{np.max(np.abs(x_post - x_pred)):.2e}"
    )


# ---------------------------------------------------------------------------
# 5. Innovation gate rejects an absurd measurement
# ---------------------------------------------------------------------------

def test_innovation_gate():
    """A measurement far outside the gate is rejected: posterior mean == prediction."""
    sim = _StubSim()
    x_pred = _sample_quat_state(seed=4)
    x_pred_exp = _quat_to_exp_np(x_pred)

    H = np.eye(_EXP_DIM)
    z = x_pred_exp + 1000.0                  # absurd measurement, way outside the gate
    R_sigmas = np.ones(_EXP_DIM) * 0.1
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, _N_RODS, base_sigma=1e-3)
    F = np.eye(_EXP_DIM)
    P = np.eye(_EXP_DIM)

    diag = {}
    x_post, _ = _ekf_step(
        x_pred, P, sim, None, F, H, z,
        Q_sigmas, R_sigmas, _N_RODS,
        have_measurement=True,
        innovation_gate_sigma=1.0,
        diagnostics=diag, x_pred_quat_np=x_pred,
    )
    assert diag.get("gated") is True, "absurd measurement should trip the gate"
    assert np.allclose(x_post, x_pred, atol=1e-12), (
        "gated step must leave the mean at the prediction"
    )


# ---------------------------------------------------------------------------
# 6. Finite-difference velocity injection from consecutive poses
# ---------------------------------------------------------------------------

def test_fd_velocity_injection():
    """Known consecutive poses produce the analytically expected linvel/angvel.

    rod 0: pure translation (Δp/dt, zero angvel)
    rod 1: rotation φ about world z  → angvel = [0, 0, φ/dt]
    rod 2: rotation ψ about world x + translation → angvel = [ψ/dt, 0, 0]
    """
    n_rods = 3
    dt = 0.01
    stride = 7  # [x y z qw qx qy qz]

    z_prev = np.zeros(stride * n_rods, dtype=np.float64)
    z_curr = np.zeros(stride * n_rods, dtype=np.float64)
    for r in range(n_rods):
        z_prev[stride * r + 3] = 1.0  # identity quat
        z_curr[stride * r + 3] = 1.0

    # rod 0: pure translation → linvel [1, 2, 3]
    z_curr[0:3] = [0.01, 0.02, 0.03]

    # rod 1: rotation φ about z
    phi = 0.1
    z_curr[stride * 1 + 3: stride * 1 + 7] = [np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)]

    # rod 2: rotation ψ about x + translation → linvel [0, -1, 0]
    psi = 0.2
    z_curr[stride * 2: stride * 2 + 3] = [0.0, -0.01, 0.0]
    z_curr[stride * 2 + 3: stride * 2 + 7] = [np.cos(psi / 2), np.sin(psi / 2), 0.0, 0.0]

    x_in = _identity_quat_state(n_rods)
    x_out = _fd_inject_velocities(x_in, z_curr, z_prev, dt, n_rods)

    # rod 0
    assert np.allclose(x_out[7:10], [1.0, 2.0, 3.0], atol=1e-9)
    assert np.allclose(x_out[10:13], [0.0, 0.0, 0.0], atol=1e-9)
    # rod 1
    assert np.allclose(x_out[13 + 7: 13 + 10], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(x_out[13 + 10: 13 + 13], [0.0, 0.0, phi / dt], atol=1e-7), (
        f"rod1 angvel = {x_out[13 + 10:13 + 13]}, expected [0,0,{phi / dt}]"
    )
    # rod 2
    assert np.allclose(x_out[26 + 7: 26 + 10], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(x_out[26 + 10: 26 + 13], [psi / dt, 0.0, 0.0], atol=1e-7), (
        f"rod2 angvel = {x_out[26 + 10:26 + 13]}, expected [{psi / dt},0,0]"
    )


# ---------------------------------------------------------------------------
# 7. Measurement canonicalization discards axial spin
# ---------------------------------------------------------------------------

def test_measurement_canonicalization_ignores_axial_spin():
    """Two quats with the same principal axis but different axial spin map to the
    same exp-map measurement.

    This pins down the GNN-convention canonicalization in `_pose_quat_to_exp`
    (compute_prin_axis → compute_quat_btwn_z_and_vec).  A regression here silently
    corrupts every innovation — it was a root cause of the earlier NEES=521 bug.
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
# 8. exp ↔ quat round-trip identity
# ---------------------------------------------------------------------------

def test_exp_quat_roundtrip():
    """quat → exp → quat reproduces the original state (qw ≥ 0) to float64 precision.

    The whole filter rests on these conversions; a convention bug here breaks
    every innovation and retraction.
    """
    x = _sample_quat_state(seed=11)
    t = torch.tensor(x, dtype=torch.float64).reshape(1, -1, 1)
    exp = quat_state_to_exp_state(t)
    back = exp_state_to_quat_state(exp)[0, :, 0].cpu().numpy().astype(np.float64)
    assert np.allclose(back, x, atol=1e-10), (
        f"round-trip max diff = {np.max(np.abs(back - x)):.2e}"
    )


# ---------------------------------------------------------------------------
# Model-dependent helpers
# ---------------------------------------------------------------------------

_ASSETS_PRESENT = Path(_DEFAULT_MODEL).exists() and Path(_DEFAULT_DATA).exists()
_SKIP = pytest.mark.skipif(not _ASSETS_PRESENT, reason="model/data not found")
_DEV = torch.device("cpu")


def _load_assets(model_path, data_dir, device):
    from simulators.tensegrity_gnn_simulator import load_simulator  # lazy: needs torch_geometric
    sim = load_simulator(
        model_path, map_location=torch.device("cpu"), cache_batch_sizes=[1]
    )
    sim = sim.to(device)
    sim.eval()
    p = Path(data_dir)
    with open(p / "processed_data.json") as f:
        gt_data = json.load(f)
    with open(p / "extra_state_data.json") as f:
        extra_data = json.load(f)
    return sim, gt_data, extra_data


def _make_start_state(gt_data, n_rods, device):
    d0 = gt_data[0]
    vals = []
    for r in range(n_rods):
        vals.extend(
            d0["pos"][r * 3: (r + 1) * 3]
            + d0["quat"][r * 4: (r + 1) * 4]
            + d0["linvel"][r * 3: (r + 1) * 3]
            + d0["angvel"][r * 3: (r + 1) * 3]
        )
    return torch.tensor(vals, dtype=DEFAULT_DTYPE).reshape(1, -1, 1).to(device)


def _reset_sim(sim, extra_data, device):
    dtype = DEFAULT_DTYPE
    cables = list(sim.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            extra_data[0]["rest_lengths"][i], dtype=dtype
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            extra_data[0]["motor_speeds"][i], dtype=dtype, device=device
        ).reshape(1, 1, 1)
    sim.ctrls_hist = None
    sim.node_hidden_state = None


def _assert_cov_well_formed(P, tag=""):
    """Covariance must stay symmetric and (numerically) PSD."""
    assert np.allclose(P, P.T, atol=1e-8), f"{tag}: P not symmetric"
    min_eig = np.linalg.eigvalsh(0.5 * (P + P.T)).min()
    assert min_eig >= -1e-8, f"{tag}: P not PSD; min eigenvalue = {min_eig:.2e}"


# ---------------------------------------------------------------------------
# 9. Predict-only matches GNN rollout
# ---------------------------------------------------------------------------

def run_predict_only_matches_gnn():
    """EKF with no measurements must replicate the GNN rollout to near machine precision.

    Verifies: direct-GNN mean propagation, context save/restore, and LSTM sync.
    If F or the LSTM sync path is broken the states will diverge.  Also asserts the
    covariance stays symmetric PSD at every step.
    """
    print("\n=== run_predict_only_matches_gnn ===")
    sim, gt_data, extra_data = _load_assets(_DEFAULT_MODEL, _DEFAULT_DATA, _DEV)
    n_rods = len(sim.robot.rigid_bodies)
    quat_dim = 13 * n_rods
    start = _make_start_state(gt_data, n_rods, _DEV)
    dtype = DEFAULT_DTYPE
    n_steps = 10
    dataset_idx = 9
    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=_DEV)}

    # GNN-only rollout — ground truth for this test
    _reset_sim(sim, extra_data, _DEV)
    gnn_states = []
    x_t = start.clone()
    with torch.no_grad():
        for k in range(n_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            ns, _ = sim.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
            x_np = ns[0, :quat_dim, 0].detach().cpu().numpy().astype(np.float64)
            gnn_states.append(x_np)
            x_t = torch.tensor(x_np, dtype=dtype, device=_DEV).reshape(1, quat_dim, 1)

    # EKF predict-only — must produce the same sequence
    _reset_sim(sim, extra_data, _DEV)
    P = np.eye(_EXP_DIM) * 0.01
    F = np.eye(_EXP_DIM)
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, n_rods, base_sigma=1e-4)
    R_sigmas = np.ones(6 * n_rods)  # unused (no measurement)
    H_dummy = np.zeros((6 * n_rods, _EXP_DIM))

    x_ekf = start.detach().cpu().numpy().reshape(-1)[:quat_dim].astype(np.float64)
    ekf_states = []
    with torch.no_grad():
        for k in range(n_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            x_ekf, P = _ekf_step(
                x_ekf, P, sim, ctrl, F, H_dummy, None,
                Q_sigmas, R_sigmas, n_rods,
                have_measurement=False,
                dataset_idx_val=dataset_idx,
            )
            _assert_cov_well_formed(P, tag=f"predict step {k}")
            ekf_states.append(x_ekf.copy())

    max_err = max(np.max(np.abs(ekf_states[k] - gnn_states[k])) for k in range(n_steps))
    print(f"  max deviation over {n_steps} steps: {max_err:.2e}")
    assert max_err < 1e-6, (
        f"EKF predict-only differs from GNN rollout by {max_err:.2e} (expect < 1e-6)"
    )


# ---------------------------------------------------------------------------
# 10. Update step corrects toward truth (and reaches the noise floor)
# ---------------------------------------------------------------------------

def run_update_corrects_toward_truth():
    """EKF with noisy measurements must (a) beat predict-only and (b) reach the floor.

    Setup:
      True trajectory  — GNN rollout from GT x0.
      Perturbed init   — GT x0 + Gaussian position noise (σ = 0.05 m).
      Measurements     — true pose corrupted by small noise (σ_pos = 0.001 m,
                         σ_rot = 0.005 rad in exp-map space), pose-only.
      Process noise    — large (σ_Q = 1.0), so the filter heavily trusts measurements.

    Beating predict-only alone is a weak bar (predict-only diverges exponentially),
    so we additionally require the EKF position RMSE to sit near the measurement
    noise floor — proof the filter actually tracks truth.  Covariance is checked
    symmetric PSD at every step.
    """
    print("\n=== run_update_corrects_toward_truth ===")
    sim, gt_data, extra_data = _load_assets(_DEFAULT_MODEL, _DEFAULT_DATA, _DEV)
    n_rods = len(sim.robot.rigid_bodies)
    quat_dim = 13 * n_rods
    start = _make_start_state(gt_data, n_rods, _DEV)
    dtype = DEFAULT_DTYPE
    n_steps = 20
    eval_start = 10
    dataset_idx = 9
    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=_DEV)}
    rng = np.random.default_rng(42)

    # True trajectory from GT x0
    _reset_sim(sim, extra_data, _DEV)
    true_states = []  # (quat_dim,) each, indexed k = 0..n_steps
    x_t = start.clone()
    true_states.append(x_t.detach().cpu().numpy().reshape(-1)[:quat_dim].astype(np.float64))
    with torch.no_grad():
        for k in range(n_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            ns, _ = sim.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
            x_np = ns[0, :quat_dim, 0].detach().cpu().numpy().astype(np.float64)
            true_states.append(x_np)
            x_t = torch.tensor(x_np, dtype=dtype, device=_DEV).reshape(1, quat_dim, 1)

    # Noisy measurements at each step (pose-only: pos + exp_rot per rod)
    meas_noise_pos = 0.001
    meas_noise_rot = 0.005

    def _make_meas(x_quat_np):
        x_t_ = torch.tensor(x_quat_np, dtype=dtype).reshape(1, quat_dim, 1)
        x_exp = quat_state_to_exp_state(x_t_).squeeze().cpu().numpy().astype(np.float64)
        z = np.zeros(6 * n_rods, dtype=np.float64)
        for r in range(n_rods):
            z[6 * r: 6 * r + 3] = (
                x_exp[EXP_BLOCK_SIZE * r: EXP_BLOCK_SIZE * r + 3]
                + rng.normal(0, meas_noise_pos, 3)
            )
            z[6 * r + 3: 6 * r + 6] = (
                x_exp[EXP_BLOCK_SIZE * r + 3: EXP_BLOCK_SIZE * r + 6]
                + rng.normal(0, meas_noise_rot, 3)
            )
        return z

    # Measurements are at steps 1..n_steps (after the first GNN step)
    measurements = [_make_meas(true_states[k + 1]) for k in range(n_steps)]

    # Perturbed initial state: add position noise to GT x0
    x0_perturbed = true_states[0].copy()
    for r in range(n_rods):
        x0_perturbed[13 * r: 13 * r + 3] += rng.normal(0, 0.05, 3)

    # H: pose-only observation (pos + exp_rot per rod)
    H = np.zeros((6 * n_rods, _EXP_DIM), dtype=np.float64)
    for r in range(n_rods):
        for i in range(6):
            H[6 * r + i, EXP_BLOCK_SIZE * r + i] = 1.0

    R_sigmas = np.empty(6 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        R_sigmas[6 * r: 6 * r + 3] = meas_noise_pos
        R_sigmas[6 * r + 3: 6 * r + 6] = meas_noise_rot

    # Large process noise so the gain is high and measurements dominate
    Q_sigmas = _structured_Q_sigmas(_EXP_DIM, n_rods, base_sigma=1.0)
    F = np.eye(_EXP_DIM, dtype=np.float64)
    P0 = np.eye(_EXP_DIM, dtype=np.float64)

    def _run(with_measurements: bool):
        _reset_sim(sim, extra_data, _DEV)
        x = x0_perturbed.copy()
        P = P0.copy()
        states = []
        with torch.no_grad():
            for k in range(n_steps):
                ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
                x, P = _ekf_step(
                    x, P, sim, ctrl, F, H,
                    measurements[k] if with_measurements else None,
                    Q_sigmas, R_sigmas, n_rods,
                    have_measurement=with_measurements,
                    dataset_idx_val=dataset_idx,
                )
                _assert_cov_well_formed(P, tag=f"update step {k}")
                states.append(x.copy())
        return states

    ekf_states = _run(with_measurements=True)
    pred_states = _run(with_measurements=False)

    # Position RMSE over evaluation window (steps eval_start..n_steps)
    pos_col = np.array([EXP_BLOCK_SIZE * r + j for r in range(n_rods) for j in range(3)])

    def _pos_rmse(states):
        errs = []
        for k in range(eval_start, n_steps):
            x_exp = (
                quat_state_to_exp_state(
                    torch.tensor(states[k], dtype=dtype).reshape(1, quat_dim, 1)
                ).squeeze().cpu().numpy().astype(np.float64)
            )
            x_true_exp = (
                quat_state_to_exp_state(
                    torch.tensor(true_states[k + 1], dtype=dtype).reshape(1, quat_dim, 1)
                ).squeeze().cpu().numpy().astype(np.float64)
            )
            errs.append(np.sqrt(np.mean((x_exp[pos_col] - x_true_exp[pos_col]) ** 2)))
        return float(np.mean(errs))

    ekf_rmse = _pos_rmse(ekf_states)
    pred_rmse = _pos_rmse(pred_states)
    print(f"  EKF RMSE = {ekf_rmse:.4f} m,  predict-only RMSE = {pred_rmse:.4f} m")

    # (a) Must beat the diverging predict-only baseline.
    assert ekf_rmse < pred_rmse, (
        f"EKF RMSE ({ekf_rmse:.4f} m) should be < predict-only RMSE ({pred_rmse:.4f} m)"
    )
    # (b) Must actually track truth: high-gain filter should reach the noise floor.
    #     Floor = 20× the per-axis position measurement σ (generous, but a real
    #     ceiling — pred-only diverges to >> this).  Tracking to ~mm proves the
    #     update is correcting, not merely "less bad than garbage".
    rmse_floor = 20.0 * meas_noise_pos  # 0.02 m
    assert ekf_rmse < rmse_floor, (
        f"EKF RMSE ({ekf_rmse:.4f} m) should reach the noise floor (< {rmse_floor:.3f} m)"
    )


# ---------------------------------------------------------------------------
# pytest entry points (model-dependent)
# ---------------------------------------------------------------------------

@_SKIP
def test_pytest_predict_only_matches_gnn():
    run_predict_only_matches_gnn()


@_SKIP
def test_pytest_update_corrects_toward_truth():
    run_update_corrects_toward_truth()


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["all", "algebraic", "model"], default="all")
    args = parser.parse_args()

    algebraic = [
        ("retraction_zero_delta",     test_retraction_zero_delta),
        ("retraction_known_rotation", test_retraction_known_rotation),
        ("make_pd",                   test_make_pd),
        ("predict_covariance_grows",  test_predict_covariance_grows),
        ("update_covariance_shrinks", test_update_covariance_shrinks),
        ("update_innovation_zero",    test_update_innovation_zero),
        ("innovation_gate",           test_innovation_gate),
        ("fd_velocity_injection",     test_fd_velocity_injection),
        ("measurement_canonicalization_ignores_axial_spin",
         test_measurement_canonicalization_ignores_axial_spin),
        ("exp_quat_roundtrip",        test_exp_quat_roundtrip),
    ]
    model_dep = [
        ("predict_only_matches_gnn",     run_predict_only_matches_gnn),
        ("update_corrects_toward_truth", run_update_corrects_toward_truth),
    ]

    to_run = []
    if args.test in ("all", "algebraic"):
        to_run.extend(algebraic)
    if args.test in ("all", "model"):
        if not _ASSETS_PRESENT:
            print("WARNING: model/data not found — skipping model-dependent tests")
        else:
            to_run.extend(model_dep)

    results = {}
    for name, fn in to_run:
        try:
            fn()
            results[name] = True
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            results[name] = False
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            results[name] = False

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    sys.exit(0 if all(results.values()) else 1)
