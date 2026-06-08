"""Multiplicative Extended Kalman Filter (MEKF) for tensegrity simulation.

State representation
--------------------
Mean        : 39D quat space  (3 pos + 4 quat + 3 linvel + 3 angvel per rod)
              Advanced by direct GNN step — zero exp-map round-trip error.
Covariance  : 36D exp-map space  (3 pos + 3 exp_rot + 3 linvel + 3 angvel)
              Jacobian is 36×36 and naturally full-rank.

Measurement update
------------------
Innovation is computed in exp-map space (H @ x_pred_exp).
The Kalman correction delta_exp is mapped back to quat via the MEKF retraction:
  quat_post = exp2quat(quat2exp(quat_pred) + delta_rot)
  pos_post  = pos_pred  + delta_pos
  vel_post  = vel_pred  + delta_vel

LSTM sync
---------
After each step the LSTM hidden state is advanced from x_k (the INPUT state
at time k), matching exactly what the GNN rollout does.

Public API
----------
  run_ekf_rollout(...)  →  list of frame dicts  {'time', 'pose', 'state'}
  OnlineEKF             →  streaming step-by-step wrapper
"""

import numpy as np
import torch
import tqdm

from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
)
from utilities.misc_utils import DEFAULT_DTYPE
from utilities.torch_quaternion import (
    quat2exp,
    compute_prin_axis,
    compute_quat_btwn_z_and_vec,
)  # used in _pose_quat_to_exp


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _make_pd(P: np.ndarray, min_eig: float = 1e-9) -> np.ndarray:
    """Return a symmetric positive-definite version of P via eigenvalue clamping."""
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, min_eig)
    return (eigvecs * eigvals) @ eigvecs.T


def _control_to_numpy_vector(ctrl):
    """Convert control input to a flat float64 numpy vector."""
    if ctrl is None:
        return None
    if isinstance(ctrl, torch.Tensor):
        return ctrl.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if isinstance(ctrl, np.ndarray):
        return ctrl.reshape(-1).astype(np.float64)
    if isinstance(ctrl, (list, tuple)):
        vals = []
        for c in ctrl:
            if isinstance(c, torch.Tensor):
                vals.extend(c.detach().cpu().numpy().reshape(-1).astype(np.float64).tolist())
            else:
                vals.append(float(c))
        return np.asarray(vals, dtype=np.float64)
    return np.asarray([float(ctrl)], dtype=np.float64)


def _ensure_ctrl_for_step(ctrl, simulator):
    """Convert per-step control to (1, num_cables, 1) tensor on simulator device."""
    if ctrl is None:
        return None
    dtype = getattr(simulator, 'dtype', DEFAULT_DTYPE)
    device = getattr(simulator, 'device', 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if isinstance(ctrl, torch.Tensor):
        t = ctrl.to(device=device, dtype=dtype)
        if t.dim() == 1:
            t = t.reshape(1, -1, 1)
        return t
    if isinstance(ctrl, np.ndarray):
        return torch.from_numpy(ctrl).to(device=device, dtype=dtype).reshape(1, -1, 1)
    if isinstance(ctrl, (list, tuple)):
        return torch.tensor(
            [float(c) for c in ctrl], dtype=dtype, device=device
        ).reshape(1, -1, 1)
    return torch.tensor([float(ctrl)], dtype=dtype, device=device).reshape(1, 1, 1)


# ---------------------------------------------------------------------------
# Measurement conversion  (sensor quat → exp-map measurement vector)
# ---------------------------------------------------------------------------

def _pose_quat_to_exp(pos_quat_np: np.ndarray,
                      n_rods: int,
                      dtype: torch.dtype,
                      device: torch.device) -> np.ndarray:
    """Convert a flat pos+quat measurement to pos+exp_rot.

    The GNN's quaternion output uses compute_quat_btwn_z_and_vec(principal_axis),
    which is the minimal rotation from the z-axis to the rod's principal axis.
    This discards axial spin (rotation around the rod's long axis), which is
    unobservable from the GNN dynamics.  Measurement quaternions (from physics
    simulation) may include axial spin, so we canonicalize them to the same
    GNN convention before computing the innovation.

    Args:
        pos_quat_np: (7*n_rods,)  [x y z qw qx qy qz] per rod
    Returns:
        (6*n_rods,)  [x y z ex ey ez] per rod  (in GNN canonical orientation)
    """
    out = np.empty(6 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        pos  = pos_quat_np[7 * r     : 7 * r + 3]
        quat = pos_quat_np[7 * r + 3 : 7 * r + 7]
        quat_t = torch.tensor(quat, dtype=dtype, device=device).reshape(1, 4, 1)
        # Canonicalize: extract principal axis, then recover GNN-form quaternion.
        prin_axis   = compute_prin_axis(quat_t)         # (1, 3, 1)
        q_canonical = compute_quat_btwn_z_and_vec(
            prin_axis.squeeze(-1)                        # (1, 3)
        ).reshape(1, 4, 1)
        exp_rot = quat2exp(q_canonical)[0, :, 0].cpu().numpy()
        out[6 * r     : 6 * r + 3] = pos
        out[6 * r + 3 : 6 * r + 6] = exp_rot
    return out


def _full_quat_state_to_exp_np(state_13n: np.ndarray,
                                n_rods: int,
                                dtype: torch.dtype,
                                device: torch.device) -> np.ndarray:
    """Convert a full 13*n_rods quat state vector to 12*n_rods exp state."""
    t = torch.tensor(state_13n, dtype=dtype, device=device).reshape(1, -1, 1)
    return quat_state_to_exp_state(t)[0, :, 0].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def _structured_Q_sigmas(state_dim, n_rods, base_sigma,
                          exp_inflation=1.5, vel_inflation=2.0):
    """Diagonal process-noise sigmas for exp-map covariance."""
    sigmas = np.full(state_dim, base_sigma, dtype=np.float64)
    for r in range(n_rods):
        base = EXP_BLOCK_SIZE * r
        sigmas[base + 3 : base + 6]  *= exp_inflation
        sigmas[base + 6 : base + 12] *= vel_inflation
    return sigmas


def _structured_R_sigmas(meas_dim, n_rods, pos_sigma, exp_sigma=None):
    """Diagonal measurement-noise sigmas for exp-map measurements."""
    if exp_sigma is None:
        exp_sigma = 5.0 * pos_sigma
    sigmas = np.empty(meas_dim, dtype=np.float64)
    if meas_dim == 6 * n_rods:
        for r in range(n_rods):
            sigmas[6 * r     : 6 * r + 3] = pos_sigma
            sigmas[6 * r + 3 : 6 * r + 6] = exp_sigma
    else:
        for r in range(n_rods):
            sigmas[12 * r     : 12 * r + 3]  = pos_sigma
            sigmas[12 * r + 3 : 12 * r + 6]  = exp_sigma
            sigmas[12 * r + 6 : 12 * r + 12] = pos_sigma
    return sigmas


# ---------------------------------------------------------------------------
# MEKF retraction: apply exp-map correction to a quat-space state
# ---------------------------------------------------------------------------

def _apply_exp_correction(x_quat_np: np.ndarray,
                           delta_exp: np.ndarray,
                           n_rods: int,
                           dtype: torch.dtype) -> np.ndarray:
    """Retract a 36D exp-map correction onto a 39D quat-space state.

    All arithmetic in float64 to avoid float32 quantization errors.

    Rotation uses left-multiplicative retraction:
        q_post = exp2quat(delta_rot) ⊗ q_pred  (Hamilton product)

    When ||delta_rot|| < _RETRACT_SKIP the rotation quaternion is left
    unchanged — the correction is unobservable at float32 precision and any
    renormalization would introduce a larger error (~6e-8) than the correction.
    """
    # If the total correction is negligible (below float32 precision), skip.
    _RETRACT_SKIP = 1e-7
    if np.linalg.norm(delta_exp) < _RETRACT_SKIP:
        return x_quat_np.copy()

    x_post = x_quat_np.copy()
    for r in range(n_rods):
        qb = 13 * r
        eb = EXP_BLOCK_SIZE * r
        # Position and velocity — additive in both spaces
        x_post[qb     : qb + 3]  += delta_exp[eb     : eb + 3]
        x_post[qb + 7 : qb + 13] += delta_exp[eb + 6 : eb + 12]
        # Rotation — float64 left-multiplicative retraction
        dv  = delta_exp[eb + 3 : eb + 6].astype(np.float64)
        th  = np.linalg.norm(dv)
        if th < 1e-10:
            qd = np.array([1.0, 0.5 * dv[0], 0.5 * dv[1], 0.5 * dv[2]], dtype=np.float64)
        else:
            ht  = 0.5 * th
            qd  = np.empty(4, dtype=np.float64)
            qd[0]  = np.cos(ht)
            qd[1:] = (np.sin(ht) / th) * dv
        # Hamilton product qd ⊗ q_pred (float64)
        q   = x_quat_np[qb + 3 : qb + 7].astype(np.float64)
        qw, qx, qy, qz = q
        dw, dx, dy, dz = qd
        rw = dw*qw - dx*qx - dy*qy - dz*qz
        rx = dw*qx + dx*qw + dy*qz - dz*qy
        ry = dw*qy - dx*qz + dy*qw + dz*qx
        rz = dw*qz + dx*qy - dy*qx + dz*qw
        n  = np.sqrt(rw*rw + rx*rx + ry*ry + rz*rz)
        x_post[qb + 3 : qb + 7] = np.array([rw, rx, ry, rz]) / n
    return x_post


# ---------------------------------------------------------------------------
# Core MEKF step
# ---------------------------------------------------------------------------

def _fd_inject_velocities(x_quat_np: np.ndarray,
                          z_curr_flat: np.ndarray,
                          z_prev_flat: np.ndarray,
                          dt: float,
                          n_rods: int) -> np.ndarray:
    """Replace velocity in a quat state with finite-difference from consecutive pose measurements.

    z_curr_flat / z_prev_flat: flat [x y z qw qx qy qz] * n_rods arrays.
    """
    x_out       = x_quat_np.copy()
    pose_stride = 7   # [x y z qw qx qy qz]

    for r in range(n_rods):
        pos_curr  = z_curr_flat[pose_stride * r     : pose_stride * r + 3]
        pos_prev  = z_prev_flat[pose_stride * r     : pose_stride * r + 3]
        linvel_fd = (pos_curr - pos_prev) / dt

        q_curr = z_curr_flat[pose_stride * r + 3 : pose_stride * r + 7]
        q_prev = z_prev_flat[pose_stride * r + 3 : pose_stride * r + 7]
        if np.dot(q_curr, q_prev) < 0:
            q_prev = -q_prev
        q_prev_conj = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
        w0, x0, y0, z0 = q_curr
        w1, x1, y1, z1 = q_prev_conj
        q_rel = np.array([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
        ])
        vec_norm  = np.linalg.norm(q_rel[1:])
        angle     = 2.0 * np.arctan2(vec_norm, q_rel[0])
        if abs(angle - 2.0 * np.pi) < abs(angle):
            angle -= 2.0 * np.pi
        axis      = q_rel[1:] / np.sin(angle / 2.0) if vec_norm > 1e-10 else np.zeros(3)
        angvel_fd = angle * axis / dt

        qb = 13 * r
        x_out[qb + 7  : qb + 10] = linvel_fd
        x_out[qb + 10 : qb + 13] = angvel_fd

    return x_out


def _clamp_velocities_quat(x_quat_np: np.ndarray, n_rods: int,
                            max_linvel: float = 3.0,
                            max_angvel: float = 25.0) -> np.ndarray:
    """Clamp linvel and angvel in a quat-space state to physical bounds."""
    x_out = x_quat_np.copy()
    for r in range(n_rods):
        qb     = 13 * r
        linvel = x_out[qb + 7  : qb + 10]
        angvel = x_out[qb + 10 : qb + 13]
        lv     = np.linalg.norm(linvel)
        av     = np.linalg.norm(angvel)
        if lv > max_linvel:
            x_out[qb + 7  : qb + 10] = linvel * (max_linvel / lv)
        if av > max_angvel:
            x_out[qb + 10 : qb + 13] = angvel * (max_angvel / av)
    return x_out


def _ekf_step(x_quat_np: np.ndarray,
              P_exp: np.ndarray,
              simulator,
              ctrl,
              F_exp: np.ndarray,
              H_np: np.ndarray,
              z_exp,
              Q_sigmas: np.ndarray,
              R_sigmas: np.ndarray,
              n_rods: int,
              have_measurement: bool,
              innovation_gate_sigma: float = np.inf,
              dataset_idx_val: int = 9,
              diagnostics: dict | None = None,
              x_pred_quat_np: np.ndarray | None = None):
    """One MEKF predict-and-update step.

    Mean is kept in quat space; covariance in 36D exp-map space.
    Predict mean uses a direct GNN step — no exp-map round-trip error.

    Args:
        x_quat_np:   (39,) current quat-space state.
        P_exp:       (36, 36) current covariance.
        simulator:   TensegrityGNNSimulator.
        ctrl:        (1, num_cables, 1) control tensor or None.
        F_exp:       (36, 36) pre-computed state-transition Jacobian.
        H_np:        (meas_dim, 36) observation matrix.
        z_exp:       (meas_dim,) measurement in exp-map space, or None.
        Q_sigmas:    (36,) process-noise standard deviations.
        R_sigmas:    (meas_dim,) measurement-noise standard deviations.
        n_rods:      Number of rods.
        have_measurement: Whether to run the update step.
        innovation_gate_sigma: Gate threshold; np.inf = no gating.
        dataset_idx_val: Passed to the graph processor.
        diagnostics: Optional dict populated with per-step diagnostics.
        x_pred_quat_np: Pre-computed GNN prediction (39,). When provided,
            the internal GNN call and LSTM sync are skipped — the caller
            owns the GNN cadence. Required when batching num_out_steps steps
            per GNN call to match sim.run's LSTM progression.

    Returns:
        x_post_quat: (39,) posterior quat-space state.
        P_post:      (36, 36) posterior covariance.
    """
    try:
        ref   = next(simulator.parameters())
        dtype = ref.dtype
        dev   = ref.device
    except StopIteration:
        dtype = DEFAULT_DTYPE
        dev   = torch.device('cpu')

    quat_dim  = x_quat_np.size           # 39
    state_dim = P_exp.shape[0]           # 36
    s2g = {'dataset_idx': torch.tensor([[dataset_idx_val]],
                                        dtype=torch.long, device=dev)}

    x_t = torch.tensor(x_quat_np, dtype=dtype, device=dev).reshape(1, quat_dim, 1)

    # ---- Predict mean: direct GNN step (no exp-map conversion) --------------
    if x_pred_quat_np is not None:
        # Caller owns the GNN call; use the pre-computed prediction directly.
        x_pred_quat = x_pred_quat_np
    else:
        # Save LSTM + cable context at time k so we can restore before the sync.
        ctx_pre = _save_model_ctx(simulator)
        with torch.no_grad():
            ns, _ = simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
        x_pred_quat = ns[0, :quat_dim, 0].detach().cpu().numpy().astype(np.float64)
        # Restore context (the LSTM sync at the end will re-advance from x_k)
        _restore_model_ctx(simulator, ctx_pre)

    # Sanitize F
    if not np.all(np.isfinite(F_exp)):
        F_exp = np.eye(state_dim, dtype=np.float64)

    # ---- Predict covariance --------------------------------------------------
    Q_sigmas_safe = np.maximum(Q_sigmas, 1e-9)

    # Adaptive Q: inflate when current state is far from the measurement
    if have_measurement and z_exp is not None:
        x_curr_exp = quat_state_to_exp_state(x_t)[0, :state_dim, 0] \
                         .detach().cpu().numpy().astype(np.float64)
        innov_prior = np.linalg.norm(z_exp.reshape(-1) - H_np @ x_curr_exp)
        r_rms       = np.sqrt(np.mean(np.maximum(R_sigmas, 1e-9)**2))
        threshold   = 3.0 * r_rms * np.sqrt(float(z_exp.size))
        if innov_prior > threshold:
            inflate       = min((innov_prior / threshold)**2, 10.0)
            Q_sigmas_safe = Q_sigmas_safe * np.sqrt(inflate)

    Q  = np.diag(Q_sigmas_safe**2)
    P_pred = _make_pd(F_exp @ P_exp @ F_exp.T + Q)

    if not have_measurement:
        if x_pred_quat_np is None:
            with torch.no_grad():
                simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
        return x_pred_quat, P_pred

    # ---- Innovation in exp-map space -----------------------------------------
    x_pred_exp = quat_state_to_exp_state(
        torch.tensor(x_pred_quat, dtype=dtype, device=dev).reshape(1, quat_dim, 1)
    )[0, :state_dim, 0].detach().cpu().numpy().astype(np.float64)

    innovation = z_exp.reshape(-1) - H_np @ x_pred_exp

    if diagnostics is not None:
        obs_dim    = z_exp.size
        pos_stride = 6 if obs_dim == 6 * n_rods else EXP_BLOCK_SIZE
        diagnostics['pos_innovation_norm'] = float(np.mean([
            np.linalg.norm(innovation[pos_stride * r : pos_stride * r + 3])
            for r in range(n_rods)
        ]))

    # Innovation gate
    if (np.isfinite(innovation_gate_sigma) and
            np.linalg.norm(innovation) > innovation_gate_sigma * np.sqrt(innovation.size)):
        if x_pred_quat_np is None:
            with torch.no_grad():
                simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
        if diagnostics is not None:
            diagnostics['pos_correction_norm'] = 0.0
            diagnostics['gated'] = True
        return x_pred_quat, P_pred

    # ---- Kalman update -------------------------------------------------------
    R       = np.diag(np.maximum(R_sigmas, 1e-9)**2)
    S       = H_np @ P_pred @ H_np.T + R
    try:
        K = np.linalg.solve(S.T, (P_pred @ H_np.T).T).T
    except np.linalg.LinAlgError:
        K = np.zeros((state_dim, z_exp.size), dtype=np.float64)

    delta_exp   = K @ innovation
    x_post_quat = _apply_exp_correction(x_pred_quat, delta_exp, n_rods, dtype)

    # Joseph form for numerical stability: P_post = (I-KH)P(I-KH)^T + KRK^T
    IKH    = np.eye(state_dim, dtype=np.float64) - K @ H_np
    P_post = _make_pd(IKH @ P_pred @ IKH.T + K @ R @ K.T)

    if diagnostics is not None:
        x_post_exp = quat_state_to_exp_state(
            torch.tensor(x_post_quat, dtype=dtype, device=dev).reshape(1, quat_dim, 1)
        )[0, :state_dim, 0].detach().cpu().numpy().astype(np.float64)
        diagnostics['pos_correction_norm'] = float(np.mean([
            np.linalg.norm(
                x_post_exp[EXP_BLOCK_SIZE * r : EXP_BLOCK_SIZE * r + 3]
                - x_pred_exp[EXP_BLOCK_SIZE * r : EXP_BLOCK_SIZE * r + 3]
            )
            for r in range(n_rods)
        ]))
        diagnostics['gated'] = False

    # LSTM sync from x_k — skip when caller owns the GNN cadence.
    if x_pred_quat_np is None:
        with torch.no_grad():
            simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)

    return x_post_quat, P_post


# ---------------------------------------------------------------------------
# Online (streaming) EKF wrapper
# ---------------------------------------------------------------------------

class OnlineEKF:
    """Streaming MEKF wrapper.

    initialize() accepts a quat-format state (1, 39, 1).
    step() accepts measurements in quat format and returns filtered exp-map state.
    """

    def __init__(self, simulator, dt, n_rods,
                 process_noise_scale=1e-4, measurement_noise_scale=1e-3,
                 observe_pose_only=False, use_finite_diff=True,
                 innovation_gate_sigma=5.0,
                 exp_inflation=1.5, vel_inflation=0.5,
                 dataset_idx_val=9,
                 ema_alpha=0.35,
                 max_linvel=3.0, max_angvel=25.0,
                 jacobian_update_interval=10,
                 max_spectral_radius=None,
                 control_jacobian_mode="simulator",
                 require_control_jacobian=False):
        self.simulator               = simulator
        self.dt                      = dt
        self.n_rods                  = n_rods
        self.state_dim               = EXP_BLOCK_SIZE * n_rods       # 36
        self.quat_dim                = 13 * n_rods                   # 39
        self.process_noise_scale     = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.observe_pose_only       = observe_pose_only
        self.use_finite_diff         = use_finite_diff
        self.innovation_gate_sigma   = innovation_gate_sigma
        self.dataset_idx_val         = dataset_idx_val
        self.max_spectral_radius     = max_spectral_radius
        self.control_jacobian_mode   = control_jacobian_mode
        self.require_control_jacobian = require_control_jacobian

        try:
            ref         = next(simulator.parameters())
            self.dtype  = ref.dtype
            self.device = ref.device
        except StopIteration:
            self.dtype  = DEFAULT_DTYPE
            self.device = torch.device('cpu')

        base_Q_sigma  = np.sqrt(float(process_noise_scale))
        self.Q_sigmas = _structured_Q_sigmas(
            self.state_dim, n_rods, base_Q_sigma, exp_inflation, vel_inflation
        )

        pos_sigma = np.sqrt(float(measurement_noise_scale))
        if observe_pose_only:
            meas_dim  = 6 * n_rods
            self.H_np = np.zeros((meas_dim, self.state_dim), dtype=np.float64)
            for r in range(n_rods):
                for i in range(6):
                    self.H_np[6 * r + i, EXP_BLOCK_SIZE * r + i] = 1.0
        else:
            meas_dim  = self.state_dim
            self.H_np = np.eye(self.state_dim, dtype=np.float64)

        self.R_sigmas = _structured_R_sigmas(meas_dim, n_rods, pos_sigma)

        self.ema_alpha    = ema_alpha
        self.max_linvel   = max_linvel
        self.max_angvel   = max_angvel
        self._ema_state   = None
        self._prev_z_quat = None

        self.jacobian_update_interval = jacobian_update_interval
        self._step_count  = 0
        self._cached_F    = None

        # State: quat mean + exp covariance
        self.x_quat_np = None
        self.P_exp     = None

    def initialize(self, start_state: torch.Tensor,
                   rest_lengths=None, motor_speeds=None):
        """Initialize MEKF state from a quat-format state tensor (1, 39, 1)."""
        start_state = start_state.to(device=self.device, dtype=self.dtype)
        if start_state.dim() == 2:
            start_state = start_state.unsqueeze(-1)

        if rest_lengths is not None and motor_speeds is not None:
            cables = list(self.simulator.robot.actuated_cables.values())
            for i, c in enumerate(cables):
                c.actuation_length = c._rest_length - rest_lengths[i]
                c.motor.motor_state.omega_t = torch.tensor(
                    motor_speeds[i], dtype=self.dtype, device=self.device
                ).reshape(1, 1, 1)

        self.x_quat_np = (start_state.detach().cpu().numpy()
                          .reshape(-1).astype(np.float64))
        self.P_exp     = float(self.measurement_noise_scale) * np.eye(
            self.state_dim, dtype=np.float64
        )
        self._ema_state   = None
        self._prev_z_quat = None
        self._cached_F    = None
        self._step_count  = 0

    def step(self, z_t: np.ndarray = None, u_t=None,
             have_measurement=True) -> torch.Tensor:
        """Run one MEKF predict+update step.

        Args:
            z_t:  Measurement in quat format.
                  pose-only: (7*n_rods,) [x y z qw qx qy qz] per rod
                  full:      (13*n_rods,) quat state
            u_t:  Per-step control.
            have_measurement: If False, skip update.

        Returns:
            Filtered state in exp-map format (1, 36, 1).
        """
        if have_measurement and z_t is None:
            raise ValueError("z_t must be provided when have_measurement=True")

        ctrl_step = _ensure_ctrl_for_step(u_t, self.simulator)

        # Convert measurement quat → exp
        z_exp = None
        if have_measurement:
            z_exp = self._convert_measurement(z_t)

        # Recompute Jacobian periodically
        state_exp_t = quat_state_to_exp_state(
            torch.tensor(self.x_quat_np, dtype=self.dtype, device=self.device)
            .reshape(1, self.quat_dim, 1)
        )
        if (self._cached_F is None
                or self._step_count % self.jacobian_update_interval == 0):
            _, self._cached_F = linearize_dynamics_exp(
                self.simulator, state_exp_t,
                sample_index=self.dataset_idx_val,
                use_finite_diff=self.use_finite_diff,
                ctrls=ctrl_step,
                max_spectral_radius=self.max_spectral_radius,
                verbose=False,
            )

        with torch.no_grad():
            self.x_quat_np, self.P_exp = _ekf_step(
                self.x_quat_np, self.P_exp,
                self.simulator, ctrl_step,
                self._cached_F,
                self.H_np, z_exp,
                self.Q_sigmas, self.R_sigmas, self.n_rods,
                have_measurement=have_measurement,
                innovation_gate_sigma=self.innovation_gate_sigma,
                dataset_idx_val=self.dataset_idx_val,
            )

        self._step_count += 1

        if have_measurement and z_t is not None:
            self.x_quat_np = self._inject_fd_velocities(self.x_quat_np, z_t)
            self._prev_z_quat = np.asarray(z_t, dtype=np.float64).reshape(-1).copy()
        self.x_quat_np = self._clamp_velocities_quat(self.x_quat_np)

        # Convert quat state → exp for output
        state_exp_out = quat_state_to_exp_state(
            torch.tensor(self.x_quat_np, dtype=self.dtype, device=self.device)
            .reshape(1, self.quat_dim, 1)
        )
        if self._ema_state is None:
            self._ema_state = state_exp_out.clone()
        else:
            self._ema_state = (self.ema_alpha * state_exp_out
                               + (1.0 - self.ema_alpha) * self._ema_state)
        return self._ema_state

    def _inject_fd_velocities(self, x_quat_np: np.ndarray,
                               z_t: np.ndarray) -> np.ndarray:
        """Replace EKF velocity estimates with finite-difference values from measurements."""
        if self._prev_z_quat is None:
            return x_quat_np
        z_curr = np.asarray(z_t, dtype=np.float64).reshape(-1)[:7 * self.n_rods]
        z_prev = self._prev_z_quat.reshape(-1)[:7 * self.n_rods]
        return _fd_inject_velocities(x_quat_np, z_curr, z_prev, self.dt, self.n_rods)

    def _clamp_velocities_quat(self, x_quat_np: np.ndarray) -> np.ndarray:
        return _clamp_velocities_quat(x_quat_np, self.n_rods,
                                       self.max_linvel, self.max_angvel)

    def _convert_measurement(self, z_quat: np.ndarray) -> np.ndarray:
        """Convert a raw pos+quat measurement to pos+exp_rot."""
        n = self.n_rods
        if self.observe_pose_only:
            return _pose_quat_to_exp(z_quat, n, self.dtype, self.device)
        else:
            return _full_quat_state_to_exp_np(z_quat, n, self.dtype, self.device)


# ---------------------------------------------------------------------------
# Batch rollout
# ---------------------------------------------------------------------------

def run_ekf_rollout(simulator,
                    gt_data,
                    extra_gt_data,
                    dt,
                    process_noise_scale=1e-4,
                    measurement_noise_scale=1e-3,
                    observe_pose_only=False,
                    start_state=None,
                    use_finite_diff=True,
                    exp_inflation=1.5,
                    vel_inflation=0.5,
                    innovation_gate_sigma=5.0,
                    control_jacobian_mode="simulator",
                    require_control_jacobian=False,
                    dataset_idx_val=9,
                    max_spectral_radius=None,
                    jacobian_update_interval=10,
                    log_diagnostics=False,
                    verbose=False):
    """Run a MEKF rollout over ground-truth data.

    Returns:
        frames: List of dicts {'time', 'pose', 'state'}.
            'state' is a torch tensor (1, 36, 1) in exp-map format.
            'pose'  is flattened (pos, quat) per rod.
    """
    dtype  = getattr(simulator, 'dtype', DEFAULT_DTYPE)
    device = getattr(simulator, 'device', 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # Initialize cable and motor state
    init_rest_lengths = extra_gt_data[0]['rest_lengths']
    init_motor_speeds = extra_gt_data[0]['motor_speeds']
    cables = list(simulator.robot.actuated_cables.values())
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - torch.tensor(
            init_rest_lengths[i], dtype=dtype
        ).reshape(1, 1, 1).to(device)
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i], dtype=dtype, device=device
        ).reshape(1, 1, 1)
    simulator.ctrls_hist        = None
    simulator.node_hidden_state = None

    n_rods    = len(simulator.robot.rigid_bodies)
    state_dim = EXP_BLOCK_SIZE * n_rods   # 36
    quat_dim  = 13 * n_rods               # 39

    # Build initial quat state
    if start_state is None:
        d0 = gt_data[0]
        state_vals = []
        for r in range(n_rods):
            state_vals.extend(
                d0['pos'][r * 3:(r + 1) * 3]
                + d0['quat'][r * 4:(r + 1) * 4]
                + d0['linvel'][r * 3:(r + 1) * 3]
                + d0['angvel'][r * 3:(r + 1) * 3]
            )
        start_state = torch.tensor(
            state_vals, dtype=dtype
        ).reshape(1, -1, 1).to(device)
    else:
        start_state = start_state.to(device=device, dtype=dtype)
        if start_state.dim() == 2:
            start_state = start_state.unsqueeze(-1)

    # State: quat mean + exp-map covariance
    x_quat_np = start_state.detach().cpu().numpy().reshape(-1).astype(np.float64)
    P_exp     = float(measurement_noise_scale) * np.eye(state_dim, dtype=np.float64)

    Q_sigmas = _structured_Q_sigmas(
        state_dim, n_rods, np.sqrt(float(process_noise_scale)),
        exp_inflation, vel_inflation
    )
    R_sigmas = _structured_R_sigmas(
        6 * n_rods if observe_pose_only else state_dim,
        n_rods, np.sqrt(float(measurement_noise_scale))
    )

    if observe_pose_only:
        H_np = np.zeros((6 * n_rods, state_dim), dtype=np.float64)
        for r in range(n_rods):
            for i in range(6):
                H_np[6 * r + i, EXP_BLOCK_SIZE * r + i] = 1.0
    else:
        H_np = np.eye(state_dim, dtype=np.float64)

    frames   = []
    time     = 0.0
    cached_F = None
    _diag_innov: list[float] = []
    _diag_corr:  list[float] = []

    # First frame
    start_exp = quat_state_to_exp_state(start_state)
    pose      = _exp_state_to_pose_np(start_exp, n_rods, dtype, device)
    frames.append({"time": time, "pose": pose,
                   "state": start_exp.detach().clone()})

    def _gt_pose_flat(d):
        pos  = np.array(d['pos'],  dtype=np.float64).reshape(n_rods, 3)
        quat = np.array(d['quat'], dtype=np.float64).reshape(n_rods, 4)
        return np.hstack([pos, quat]).reshape(-1)

    prev_pose_flat = _gt_pose_flat(gt_data[0])

    num_out_steps = getattr(simulator, 'num_out_steps', 1)
    n_extra       = len(extra_gt_data)
    s2g = {'dataset_idx': torch.tensor([[dataset_idx_val]], dtype=torch.long, device=device)}

    with torch.no_grad():
        batch_start = 0
        for _ in tqdm.tqdm(range(0, n_extra, num_out_steps)):
            batch_end = min(batch_start + num_out_steps, n_extra)
            batch     = extra_gt_data[batch_start:batch_end]

            # Build multi-step control with the actual future controls for this batch.
            ctrl_batch = torch.cat(
                [_ensure_ctrl_for_step(b['controls'], simulator) for b in batch],
                dim=-1,  # (1, num_cables, actual_n)
            )

            # One GNN call for the whole batch — matches sim.run's LSTM cadence.
            x_t_batch = torch.tensor(
                x_quat_np, dtype=dtype, device=device
            ).reshape(1, quat_dim, 1)
            ctx_batch = _save_model_ctx(simulator)
            ns_batch, _ = simulator.step(
                x_t_batch, ctrls=ctrl_batch, state_to_graph_kwargs=s2g
            )
            # ns_batch: (1, quat_dim, num_out_steps)
            _restore_model_ctx(simulator, ctx_batch)

            for i, extra in enumerate(batch):
                k = batch_start + i
                x_pred_k = ns_batch[0, :quat_dim, i].cpu().numpy().astype(np.float64)

                have_measurement = k + 1 < len(gt_data)

                state_exp_t = quat_state_to_exp_state(
                    torch.tensor(x_quat_np, dtype=dtype, device=device)
                    .reshape(1, quat_dim, 1)
                )
                ctrl_step = _ensure_ctrl_for_step(extra['controls'], simulator)

                if cached_F is None or k % jacobian_update_interval == 0:
                    _, cached_F = linearize_dynamics_exp(
                        simulator, state_exp_t,
                        sample_index=dataset_idx_val,
                        use_finite_diff=use_finite_diff,
                        ctrls=ctrl_step,
                        max_spectral_radius=max_spectral_radius,
                        verbose=verbose,
                    )

                z_exp = None
                if have_measurement:
                    gt   = gt_data[k + 1]
                    pos  = np.array(gt['pos'],  dtype=np.float64)
                    quat = np.array(gt['quat'], dtype=np.float64)
                    if observe_pose_only:
                        z_quat = np.hstack([
                            pos.reshape(n_rods, 3),
                            quat.reshape(n_rods, 4)
                        ]).reshape(-1)
                        z_exp = _pose_quat_to_exp(z_quat, n_rods, dtype, device)
                    else:
                        lv = np.array(gt['linvel'], dtype=np.float64)
                        av = np.array(gt['angvel'], dtype=np.float64)
                        z_quat_full = np.hstack([
                            pos.reshape(n_rods, 3),
                            quat.reshape(n_rods, 4),
                            lv.reshape(n_rods, 3),
                            av.reshape(n_rods, 3),
                        ]).reshape(-1)
                        z_exp = _full_quat_state_to_exp_np(
                            z_quat_full, n_rods, dtype, device
                        )

                step_diag = {} if log_diagnostics else None
                x_quat_np, P_exp = _ekf_step(
                    x_quat_np, P_exp,
                    simulator, ctrl_step,
                    cached_F,
                    H_np, z_exp,
                    Q_sigmas, R_sigmas, n_rods,
                    have_measurement=have_measurement,
                    innovation_gate_sigma=innovation_gate_sigma,
                    dataset_idx_val=dataset_idx_val,
                    diagnostics=step_diag,
                    x_pred_quat_np=x_pred_k,
                )
                if log_diagnostics and step_diag and 'pos_innovation_norm' in step_diag:
                    _diag_innov.append(step_diag['pos_innovation_norm'])
                    _diag_corr.append(step_diag.get('pos_correction_norm', 0.0))

                if have_measurement:
                    curr_pose_flat = _gt_pose_flat(gt_data[k + 1])
                    x_quat_np = _fd_inject_velocities(
                        x_quat_np, curr_pose_flat, prev_pose_flat, dt, n_rods
                    )
                    x_quat_np = _clamp_velocities_quat(x_quat_np, n_rods)
                    prev_pose_flat = curr_pose_flat

                state_exp_out = quat_state_to_exp_state(
                    torch.tensor(x_quat_np, dtype=dtype, device=device)
                    .reshape(1, quat_dim, 1)
                )
                time += dt
                pose = _exp_state_to_pose_np(state_exp_out, n_rods, dtype, device)
                frames.append({"time": time, "pose": pose,
                               "state": state_exp_out.detach().clone()})

            # Advance LSTM once for the whole batch from the batch-start state —
            # matches exactly the cadence sim.run uses.
            simulator.step(x_t_batch, ctrls=ctrl_batch, state_to_graph_kwargs=s2g)

            batch_start = batch_end

    if log_diagnostics and _diag_innov:
        mean_innov = float(np.mean(_diag_innov))
        mean_corr  = float(np.mean(_diag_corr))
        ratio      = mean_corr / max(mean_innov, 1e-10)
        print(f"\n[EKF diagnostics — {len(_diag_innov)} steps]")
        print(f"  Mean pos innovation  ||z - H·x_pred||: {mean_innov:.4f} m")
        print(f"  Mean pos correction  ||x_post - x_pred||[pos]: {mean_corr:.4f} m")
        print(f"  Correction/innovation ratio: {ratio:.4f}")

    return frames


# ---------------------------------------------------------------------------
# Internal helper: convert exp state → flattened (pos, quat) for output
# ---------------------------------------------------------------------------

def _exp_state_to_pose_np(state_exp_t: torch.Tensor,
                           n_rods: int,
                           dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
    """Return flattened (pos, quat) per rod from an exp-map state tensor."""
    quat_state = exp_state_to_quat_state(state_exp_t)  # (1, 39, 1)
    return quat_state.reshape(-1, 13, 1)[:, :7].flatten()
