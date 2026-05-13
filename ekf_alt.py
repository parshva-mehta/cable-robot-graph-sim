"""Extended Kalman Filter using exp-map state representation.

Drop-in alternative to ekf.py.  The key differences are:

  State space  : 36D exp-map  (3 pos + 3 exp_rot + 3 linvel + 3 angvel per rod)
                 instead of 39D quaternion ambient space.
  Jacobian     : 36×36 from linearize_dynamics_exp — naturally full-rank,
                 no T-projection or ε·qqᵀ regularisation needed.
  Measurements : converted from pos+quat sensor data to pos+exp_rot before
                 the GTSAM update step (linear H matrix remains valid).
  Renormalize  : not needed — exp_rot lives in unconstrained R³.

Public API mirrors ekf.py:
  run_ekf_rollout(...)  →  list of frame dicts  {'time', 'pose', 'state'}
  OnlineEKF             →  streaming step-by-step wrapper
"""

import numpy as np
import torch
import tqdm

import gtsam

from ekf import _control_to_numpy_vector, _ensure_ctrl_for_step, _reinit_state_jitter, _make_pd
from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
    compute_nominal_step_exp,
)
from utilities.misc_utils import DEFAULT_DTYPE


# ---------------------------------------------------------------------------
# Measurement conversion  (sensor quat → exp-map measurement vector)
# ---------------------------------------------------------------------------

def _pose_quat_to_exp(pos_quat_np: np.ndarray,
                      n_rods: int,
                      dtype: torch.dtype,
                      device: torch.device) -> np.ndarray:
    """Convert a flat pos+quat measurement to pos+exp_rot.

    Args:
        pos_quat_np: (7*n_rods,)  [x y z qw qx qy qz] per rod
        n_rods:      number of rods

    Returns:
        (6*n_rods,)  [x y z ex ey ez] per rod
    """
    out = np.empty(6 * n_rods, dtype=np.float64)
    for r in range(n_rods):
        pos  = pos_quat_np[7 * r     : 7 * r + 3]
        quat = pos_quat_np[7 * r + 3 : 7 * r + 7]

        quat_t  = torch.tensor(quat,  dtype=dtype, device=device).reshape(1, 4, 1)
        exp_rot = torch.tensor([0., 0., 0.], dtype=dtype, device=device)

        from utilities.torch_quaternion import quat2exp
        exp_rot = quat2exp(quat_t)[0, :, 0].cpu().numpy()

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
# Noise helpers  (updated for 12D exp-map block)
# ---------------------------------------------------------------------------

def _structured_Q_sigmas(state_dim, n_rods, base_sigma,
                          exp_inflation=1.5, vel_inflation=2.0):
    """Diagonal process-noise sigmas for exp-map state.

    Per-rod layout: pos(3) | exp_rot(3) | linvel(3) | angvel(3)
    """
    sigmas = np.full(state_dim, base_sigma, dtype=np.float64)
    for r in range(n_rods):
        base = EXP_BLOCK_SIZE * r
        sigmas[base + 3 : base + 6]  *= exp_inflation   # exp_rot
        sigmas[base + 6 : base + 12] *= vel_inflation   # linvel + angvel
    return sigmas


def _structured_R_sigmas(meas_dim, n_rods, pos_sigma, exp_sigma=None):
    """Diagonal measurement-noise sigmas for exp-map measurements.

    meas_dim == 6*n_rods  → pose-only  (pos + exp_rot per rod)
    meas_dim == 12*n_rods → full state
    """
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
# Core EKF step  (exp-map version)
# ---------------------------------------------------------------------------

# Maximum covariance eigenvalue passed to GTSAM.  GTSAM stores the Gaussian in
# information form (Λ = P⁻¹).  If P has eigenvalues > MAX_COV_EIG, then Λ has
# eigenvalues < 1/MAX_COV_EIG ≈ 0, which triggers IndeterminantLinearSystem
# even on a brand-new KalmanFilter object.
_MAX_COV_EIG = 1e8


def _make_pd_bounded(P: np.ndarray,
                     min_eig: float = 1e-6,
                     max_eig: float = _MAX_COV_EIG) -> np.ndarray:
    """Symmetric positive-definite projection with eigenvalue clamping on both ends.

    If P contains NaN/Inf (e.g., from a corrupted GTSAM covariance read),
    falls back to a scaled identity rather than propagating NaN through eigh.
    """
    n = P.shape[0]
    if not np.all(np.isfinite(P)):
        return np.eye(n, dtype=np.float64) * min_eig
    P_sym  = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    if not np.all(np.isfinite(eigvals)):
        return np.eye(n, dtype=np.float64) * min_eig
    eigvals = np.clip(eigvals, min_eig, max_eig)
    return (eigvecs * eigvals) @ eigvecs.T


def _sanitize_mean(mean: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return mean if finite, otherwise fallback."""
    if not np.all(np.isfinite(mean)):
        return fallback.copy()
    return mean


def _fresh_kf_init(kf, state_dim, mean_np, P_np):
    """Call kf.init() with sanitized inputs; recreate KF object if still corrupted.

    Two layers of defence:
      1. mean and P are sanitized before the call (finite mean, bounded-PD cov).
      2. If the KF object's internal Bayes-tree is corrupted, replace it with a
         fresh gtsam.KalmanFilter and retry once.

    Returns:
        kf          : original or freshly-created KalmanFilter
        state_gtsam : new GTSAM state
    """
    mean_safe = mean_np.reshape(state_dim, 1)
    if not np.all(np.isfinite(mean_safe)):
        mean_safe = np.zeros((state_dim, 1), dtype=np.float64)
    P_safe = _make_pd_bounded(P_np)
    try:
        return kf, kf.init(mean_safe, P_safe)
    except RuntimeError:
        kf = gtsam.KalmanFilter(state_dim)
        return kf, kf.init(mean_safe, P_safe)


def _ekf_step_gtsam(kf, state_gtsam, simulator, state_exp_t, dt, ctrl,
                    H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                    use_finite_diff, innovation_gate_sigma=np.inf,
                    control_jacobian_mode="identity",
                    require_control_jacobian=False,
                    dataset_idx_val=9,
                    cached_F=None,
                    diagnostics=None):
    """One EKF predict-and-update step using GTSAM (exp-map state space).

    Differences from ekf.py:
      - Jacobian is 36×36 from linearize_dynamics_exp (no rank fix needed).
      - Nominal next state computed via step_exp (exp → quat → model → exp).
      - No quaternion renormalization after update.
      - LSTM sync: hidden state is advanced from the posterior (or predict mean
        for no-measurement steps) so GNN dynamics stay consistent over time.

    Args:
        kf:            GTSAM KalmanFilter (initialized with state_dim=36).
        state_gtsam:   Current GTSAM state (36D mean).
        simulator:     TensegrityGNNSimulator.
        state_exp_t:   Current exp-map state tensor (1, 36, 1).
        dt:            Timestep (informational).
        ctrl:          (1, num_cables, 1) control tensor or None.
        H_np:          Observation matrix (meas_dim, 36).
        z_np:          Measurement vector in exp-map space.
        Q_sigmas:      Diagonal process-noise sigmas (36,).
        R_sigmas:      Diagonal measurement-noise sigmas.
        n_rods:        Number of rods.
        have_measurement: Whether to run the update step.
        use_finite_diff:  Use finite differences for Jacobian.
        innovation_gate_sigma: Reject update when innovation norm too large.
        control_jacobian_mode: "identity" (control Jacobian not used).
        require_control_jacobian: Raise if simulator Jacobian unavailable.
        dataset_idx_val: dataset_idx forwarded to graph processor.

    Returns:
        mean_np:     (36,) filtered exp-map state.
        state_gtsam: Updated GTSAM state.
        kf:          KalmanFilter instance (may be a fresh object if the
                     original became corrupted; callers must use this going forward).
    """
    state_dim = EXP_STATE_DIM

    dev_sim   = state_exp_t.device
    dtype_sim = state_exp_t.dtype
    s2g_kwargs = {
        'dataset_idx': torch.tensor([[dataset_idx_val]],
                                    dtype=torch.long, device=dev_sim)
    }

    # Guard: state_gtsam.mean() can fail if the KF object is already corrupted
    # from a previous step.  Fall back to the current state_exp_t mean.
    try:
        x_mean = np.array(state_gtsam.mean()).reshape(-1).astype(np.float64)
    except RuntimeError:
        x_mean = state_exp_t.detach().cpu().numpy().reshape(-1).astype(np.float64)

    # Save simulator context (LSTM + cable state) before linearization so we
    # can re-advance the LSTM from the posterior after the update.
    ctx_pre = _save_model_ctx(simulator)

    # --- Nominal next state + Jacobian ---------------------------------------
    # When a cached F is provided, skip the expensive Jacobian computation and
    # run only a single cheap GNN forward pass for the predict mean.
    if cached_F is not None:
        next_state_0_np = compute_nominal_step_exp(
            simulator, state_exp_t,
            sample_index=dataset_idx_val,
            ctrls=ctrl,
        )
        F_np = cached_F
        _restore_model_ctx(simulator, ctx_pre)
    else:
        next_state_0_np, F_np = linearize_dynamics_exp(
            simulator, state_exp_t,
            sample_index=dataset_idx_val,
            use_finite_diff=use_finite_diff,
            ctrls=ctrl,
        )

    if not np.all(np.isfinite(F_np)):
        F_np = np.eye(state_dim, dtype=np.float64)

    Q_sigmas_safe = np.maximum(Q_sigmas, 1e-6)

    # --- Adaptive Q: inflate process noise when measurement is far from
    # current estimate.  Signals model mismatch (e.g. stale LSTM history) and
    # increases Kalman gain so measurements correct the estimate faster.
    if have_measurement and z_np is not None:
        prior_innov_norm = np.linalg.norm(z_np.reshape(-1) - H_np @ x_mean)
        r_rms = np.sqrt(np.mean(np.maximum(R_sigmas, 1e-6)**2))
        threshold = 3.0 * r_rms * np.sqrt(float(z_np.size))
        if prior_innov_norm > threshold:
            inflate = min((prior_innov_norm / threshold)**2, 10.0)
            Q_sigmas_safe = Q_sigmas_safe * np.sqrt(inflate)

    F_cont = np.ascontiguousarray(F_np, dtype=np.float64)

    # Control Jacobian (exp-map state space — falls back to identity)
    J_u = None
    if control_jacobian_mode == "simulator":
        # NOTE: simulator.compute_control_jacobian (if present) returns a
        # Jacobian in quat space; using it here without conversion would be
        # incorrect, so we leave J_u as None and absorb the control effect
        # entirely into the affine term b_aff.
        pass
    if J_u is not None and require_control_jacobian:
        raise RuntimeError(
            "control_jacobian_mode='simulator' is not supported in ekf_alt.py "
            "because the simulator Jacobian is in quat space. "
            "Use control_jacobian_mode='identity' instead."
        )

    B_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
    b_aff  = (next_state_0_np - F_np @ x_mean).reshape(state_dim, 1)
    u_eff  = b_aff

    # Guard: NaN/Inf in u_eff (from corrupted next_state) causes a C++ segfault
    # inside gtsam — fall back to a zero affine term so predict() stays safe.
    if not np.all(np.isfinite(u_eff)):
        u_eff = np.zeros_like(u_eff)

    P_reset_scale = float(np.mean(Q_sigmas_safe ** 2))
    model_q = gtsam.noiseModel.Diagonal.Sigmas(Q_sigmas_safe)

    # Three-level predict fallback.  GTSAM's square-root Bayes-tree can become
    # permanently ill-conditioned; once corrupted, even kf.init() raises.
    # _fresh_kf_init() creates a brand-new KalmanFilter object when needed.
    try:
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
        state_pred = _reinit_state_jitter(kf, state_pred, state_dim)
    except RuntimeError:
        try:
            F_id = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
            u_id = np.zeros((state_dim, 1), dtype=np.float64)
            state_pred = kf.predict(state_gtsam, F_id, B_cont, u_id, model_q)
            state_pred = _reinit_state_jitter(kf, state_pred, state_dim)
        except RuntimeError:
            # Both predict attempts failed: reset with a fresh KF object.
            P_fresh = np.eye(state_dim, dtype=np.float64) * P_reset_scale
            kf, state_pred = _fresh_kf_init(kf, state_dim, x_mean, P_fresh)

    # Extract predicted mean — may still raise on a corrupted KF object.
    try:
        mean_pred_raw = np.array(state_pred.mean()).reshape(-1).copy()
        mean_pred_raw = _sanitize_mean(mean_pred_raw, x_mean)
    except RuntimeError:
        mean_pred_raw = x_mean.copy()
        P_fresh = np.eye(state_dim, dtype=np.float64) * P_reset_scale
        kf, state_pred = _fresh_kf_init(kf, state_dim, mean_pred_raw, P_fresh)

    if not have_measurement:
        # LSTM sync: linearize_dynamics_exp restored ctx, so LSTM is still at
        # ctx_pre.  Advance it from the predict mean (= posterior without update).
        _restore_model_ctx(simulator, ctx_pre)
        pred_quat = exp_state_to_quat_state(
            torch.tensor(mean_pred_raw, dtype=dtype_sim, device=dev_sim).reshape(1, state_dim, 1)
        )
        with torch.no_grad():
            simulator.step(pred_quat, ctrls=ctrl, state_to_graph_kwargs=s2g_kwargs)
        return mean_pred_raw, state_pred, kf

    mean_pred = _sanitize_mean(mean_pred_raw, x_mean)

    innovation = z_np.reshape(-1) - (H_np @ mean_pred)

    # Diagnostic: per-rod position innovation norm (measurement vs predicted pos).
    if diagnostics is not None:
        obs_dim = z_np.size
        pos_stride = 6 if obs_dim == 6 * n_rods else EXP_BLOCK_SIZE
        diagnostics['pos_innovation_norm'] = float(np.mean([
            np.linalg.norm(innovation[pos_stride * r : pos_stride * r + 3])
            for r in range(n_rods)
        ]))

    if (np.isfinite(innovation_gate_sigma) and
            np.linalg.norm(innovation) > innovation_gate_sigma * np.sqrt(innovation.size)):
        # Gate fired: advance LSTM from predicted mean (rejecting the measurement).
        _restore_model_ctx(simulator, ctx_pre)
        gated_quat = exp_state_to_quat_state(
            torch.tensor(mean_pred, dtype=dtype_sim, device=dev_sim).reshape(1, state_dim, 1)
        )
        with torch.no_grad():
            simulator.step(gated_quat, ctrls=ctrl, state_to_graph_kwargs=s2g_kwargs)
        if diagnostics is not None:
            diagnostics['pos_correction_norm'] = 0.0
            diagnostics['gated'] = True
        return mean_pred.copy(), state_pred, kf

    meas_dim      = z_np.size
    z_col         = np.asarray(z_np, dtype=np.float64).reshape(meas_dim, 1)
    R_sigmas_safe = np.maximum(R_sigmas, 1e-6)
    model_r       = gtsam.noiseModel.Diagonal.Sigmas(R_sigmas_safe)
    try:
        state_post = kf.update(state_pred,
                               np.ascontiguousarray(H_np, dtype=np.float64),
                               z_col, model_r)
        state_post = _reinit_state_jitter(kf, state_post, state_dim)
        mean_np = _sanitize_mean(
            np.array(state_post.mean()).reshape(-1).copy(), mean_pred
        )
    except RuntimeError:
        mean_np    = mean_pred.copy()
        state_post = state_pred

    # Diagnostic: per-rod position correction norm (posterior vs predicted pos).
    if diagnostics is not None:
        diagnostics['pos_correction_norm'] = float(np.mean([
            np.linalg.norm(
                mean_np[EXP_BLOCK_SIZE * r : EXP_BLOCK_SIZE * r + 3]
                - mean_pred[EXP_BLOCK_SIZE * r : EXP_BLOCK_SIZE * r + 3]
            )
            for r in range(n_rods)
        ]))
        diagnostics['gated'] = False

    # LSTM sync: advance hidden state from the posterior so the next predict
    # step uses correct GNN temporal context.  linearize_dynamics_exp already
    # restored ctx; restore again to be safe, then advance from posterior.
    _restore_model_ctx(simulator, ctx_pre)
    posterior_quat = exp_state_to_quat_state(
        torch.tensor(mean_np, dtype=dtype_sim, device=dev_sim).reshape(1, state_dim, 1)
    )
    with torch.no_grad():
        simulator.step(posterior_quat, ctrls=ctrl, state_to_graph_kwargs=s2g_kwargs)

    return mean_np, state_post, kf


# ---------------------------------------------------------------------------
# Online (streaming) EKF wrapper
# ---------------------------------------------------------------------------

class OnlineEKF:
    """Streaming EKF wrapper using exp-map state representation.

    API is identical to ekf.OnlineEKF; state tensors are in exp-map format
    (1, 36, 1) rather than quat format (1, 39, 1).

    initialize() accepts a quat-format state and converts automatically.
    step() expects z_t as pos+quat per rod and converts to pos+exp_rot internally.
    """

    def __init__(self, simulator, dt, n_rods,
                 process_noise_scale=1e-4, measurement_noise_scale=1e-3,
                 observe_pose_only=False, use_finite_diff=False,
                 innovation_gate_sigma=5.0,
                 exp_inflation=1.5, vel_inflation=0.5,
                 dataset_idx_val=9,
                 ema_alpha=0.35,
                 max_linvel=3.0, max_angvel=25.0,
                 jacobian_update_interval=10):
        self.simulator             = simulator
        self.dt                    = dt
        self.n_rods                = n_rods
        self.state_dim             = EXP_BLOCK_SIZE * n_rods          # 36
        self.process_noise_scale   = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.observe_pose_only     = observe_pose_only
        self.use_finite_diff       = use_finite_diff
        self.innovation_gate_sigma = innovation_gate_sigma
        self.dataset_idx_val       = dataset_idx_val

        try:
            ref   = next(simulator.parameters())
            self.dtype  = ref.dtype
            self.device = ref.device
        except StopIteration:
            self.dtype  = DEFAULT_DTYPE
            self.device = torch.device('cpu')

        base_Q_sigma = np.sqrt(float(process_noise_scale))
        self.Q_sigmas = _structured_Q_sigmas(
            self.state_dim, n_rods, base_Q_sigma, exp_inflation, vel_inflation
        )

        pos_sigma = np.sqrt(float(measurement_noise_scale))
        if observe_pose_only:
            meas_dim   = 6 * n_rods      # pos(3) + exp_rot(3) per rod
            self.H_np  = np.zeros((meas_dim, self.state_dim), dtype=np.float64)
            for r in range(n_rods):
                # pos at [12r : 12r+3], exp_rot at [12r+3 : 12r+6]
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
        self._prev_z_quat = None  # last raw quat measurement for FD velocity

        self.jacobian_update_interval = jacobian_update_interval
        self._step_count  = 0
        self._cached_F    = None   # reused between Jacobian recomputations

        self.kf           = gtsam.KalmanFilter(self.state_dim)
        self.state_gtsam  = None
        self.state_exp_t  = None

    def initialize(self, start_state: torch.Tensor,
                   rest_lengths=None, motor_speeds=None):
        """Initialize EKF state.

        Args:
            start_state: Initial state tensor in *quat* format
                         (1, 39, 1) or (1, 13*n_rods, 1).
            rest_lengths, motor_speeds: Optional cable/motor initialization.
        """
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

        # Convert initial quat state → exp state
        self.state_exp_t = quat_state_to_exp_state(start_state)    # (1, 36, 1)

        x0_np = self.state_exp_t.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        P0_np = float(self.measurement_noise_scale) * np.eye(
            self.state_dim, dtype=np.float64
        )
        self.state_gtsam  = self.kf.init(x0_np, P0_np)
        self._ema_state   = None
        self._prev_z_quat = None

    def step(self, z_t: np.ndarray = None, u_t=None,
             have_measurement=True) -> torch.Tensor:
        """Run one EKF predict+update step.

        Args:
            z_t:              Measurement in *quat* format:
                              - pose-only: (7*n_rods,) [x y z qw qx qy qz] per rod
                              - full state: (13*n_rods,) quat state
            u_t:              Per-step control input.
            have_measurement: If False, skip the update step.

        Returns:
            Filtered state tensor in *exp-map* format (1, 36, 1).
        """
        if have_measurement and z_t is None:
            raise ValueError("z_t must be provided when have_measurement=True")

        ctrl_step = _ensure_ctrl_for_step(u_t, self.simulator)

        # Convert measurement from quat → exp-map
        z_exp = None
        if have_measurement:
            z_exp = self._convert_measurement(z_t)

        # Recompute Jacobian every jacobian_update_interval steps.
        # linearize_dynamics_exp restores simulator context internally, so it is
        # safe to call before _ekf_step_gtsam.
        if (self._cached_F is None
                or self._step_count % self.jacobian_update_interval == 0):
            _, self._cached_F = linearize_dynamics_exp(
                self.simulator, self.state_exp_t,
                sample_index=self.dataset_idx_val,
                use_finite_diff=self.use_finite_diff,
                ctrls=ctrl_step,
            )

        # Always pass the cached F — _ekf_step_gtsam uses compute_nominal_step_exp
        # (one cheap GNN forward pass) for the predict mean instead of re-running
        # the full linearization.
        with torch.no_grad():
            mean_np, self.state_gtsam, self.kf = _ekf_step_gtsam(
                self.kf, self.state_gtsam, self.simulator,
                self.state_exp_t, self.dt, ctrl_step,
                self.H_np, z_exp,
                self.Q_sigmas, self.R_sigmas, self.n_rods,
                have_measurement=have_measurement,
                use_finite_diff=self.use_finite_diff,
                innovation_gate_sigma=self.innovation_gate_sigma,
                dataset_idx_val=self.dataset_idx_val,
                cached_F=self._cached_F,
            )

        self._step_count += 1

        if have_measurement and z_t is not None:
            mean_np = self._inject_fd_velocities(mean_np, z_t)
            self._prev_z_quat = np.asarray(z_t, dtype=np.float64).reshape(-1).copy()
        mean_np = self._clamp_velocities(mean_np)
        self.state_exp_t = torch.tensor(
            mean_np, dtype=self.dtype, device=self.device
        ).view(1, self.state_dim, 1)
        if self._ema_state is None:
            self._ema_state = self.state_exp_t.clone()
        else:
            self._ema_state = (self.ema_alpha * self.state_exp_t
                               + (1.0 - self.ema_alpha) * self._ema_state)
        return self._ema_state

    def _inject_fd_velocities(self, mean_np: np.ndarray,
                              z_t: np.ndarray) -> np.ndarray:
        """Inject finite-difference velocity estimates into the exp-map state.

        Velocities are not observed in pose-only mode, so their Kalman gain is
        zero and they stay frozen (typically at zero after a cold LSTM start).
        We overwrite them with FD estimates from consecutive quat measurements.

        State layout per rod (EXP_BLOCK_SIZE=12):
          pos(3) | exp_rot(3) | linvel(3) | angvel(3)
        Measurement z_t is in raw quat format: [x y z qw qx qy qz] per rod.
        """
        if self._prev_z_quat is None:
            return mean_np

        mean_np  = mean_np.copy()
        z_curr   = np.asarray(z_t, dtype=np.float64).reshape(-1)
        z_prev   = self._prev_z_quat.reshape(-1)
        dt       = self.dt
        pose_per_rod = z_curr.size // self.n_rods  # 7 (pos+quat)

        for r in range(self.n_rods):
            pos_curr = z_curr[pose_per_rod * r     : pose_per_rod * r + 3]
            pos_prev = z_prev[pose_per_rod * r     : pose_per_rod * r + 3]
            linvel_fd = (pos_curr - pos_prev) / dt

            # Axis-angle angular velocity matching compute_ang_vel_quat convention.
            # q = [w, x, y, z]  (w-first)
            q_curr = z_curr[pose_per_rod * r + 3 : pose_per_rod * r + 7]
            q_prev = z_prev[pose_per_rod * r + 3 : pose_per_rod * r + 7]
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
            vec_norm = np.linalg.norm(q_rel[1:])
            angle = 2.0 * np.arctan2(vec_norm, q_rel[0])
            if abs(angle - 2.0 * np.pi) < abs(angle):
                angle -= 2.0 * np.pi
            axis = q_rel[1:] / np.sin(angle / 2.0) if vec_norm > 1e-10 else np.zeros(3)
            angvel_fd = angle * axis / dt

            base = EXP_BLOCK_SIZE * r
            mean_np[base + 6 : base + 9]  = linvel_fd
            mean_np[base + 9 : base + 12] = angvel_fd

        return mean_np

    def _clamp_velocities(self, mean_np: np.ndarray) -> np.ndarray:
        """Clamp linvel and angvel in exp-map state to physical bounds."""
        mean_np = mean_np.copy()
        for r in range(self.n_rods):
            base   = EXP_BLOCK_SIZE * r
            linvel = mean_np[base + 6 : base + 9]
            angvel = mean_np[base + 9 : base + 12]
            lv_norm = np.linalg.norm(linvel)
            av_norm = np.linalg.norm(angvel)
            if lv_norm > self.max_linvel:
                mean_np[base + 6 : base + 9] = linvel * (self.max_linvel / lv_norm)
            if av_norm > self.max_angvel:
                mean_np[base + 9 : base + 12] = angvel * (self.max_angvel / av_norm)
        return mean_np

    def _convert_measurement(self, z_quat: np.ndarray) -> np.ndarray:
        """Convert a raw pos+quat measurement to pos+exp_rot."""
        n = self.n_rods
        if self.observe_pose_only:
            # Input: (7*n,) pos+quat  →  output: (6*n,) pos+exp_rot
            return _pose_quat_to_exp(z_quat, n, self.dtype, self.device)
        else:
            # Input: (13*n,) full quat state  →  output: (12*n,) full exp state
            return _full_quat_state_to_exp_np(z_quat, n, self.dtype, self.device)


# ---------------------------------------------------------------------------
# Batch rollout  (mirrors ekf.run_ekf_rollout)
# ---------------------------------------------------------------------------

def run_ekf_rollout(simulator,
                    gt_data,
                    extra_gt_data,
                    dt,
                    process_noise_scale=1e-4,
                    measurement_noise_scale=1e-3,
                    observe_pose_only=False,
                    start_state=None,
                    use_finite_diff=False,
                    exp_inflation=1.5,
                    vel_inflation=0.5,
                    innovation_gate_sigma=5.0,
                    dataset_idx_val=9,
                    jacobian_update_interval=10,
                    log_diagnostics=False):
    """Run an EKF rollout over ground-truth data (exp-map state space).

    API matches ekf.run_ekf_rollout; 'pose' in returned frames is still
    flattened (pos, quat) per rod so downstream metrics code is unchanged.

    Args:
        simulator:               TensegrityGNNSimulator.
        gt_data:                 List of dicts with 'pos', 'quat', 'linvel', 'angvel'.
        extra_gt_data:           List of dicts with 'controls', 'rest_lengths',
                                 'motor_speeds'.
        dt:                      Time step between frames.
        process_noise_scale:     Scale for Q.
        measurement_noise_scale: Scale for R and P0.
        observe_pose_only:       If True, z is 7*n_rods (pos+quat) per step.
        start_state:             Optional initial quat-format state tensor.
        use_finite_diff:         Passed to linearize_dynamics_exp.
        exp_inflation:           Q sigma multiplier for exp_rot block.
        vel_inflation:           Q sigma multiplier for velocity block.
        innovation_gate_sigma:   Gate threshold (default: no gating).
        dataset_idx_val:         dataset_idx for graph processor.

    Returns:
        frames: List of dicts {'time', 'pose', 'state'}.
            'state' is a torch tensor (1, 36, 1) in *exp-map* format.
            'pose'  is flattened (pos, quat) per rod (converted from exp for
            compatibility with existing evaluation code).
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

    simulator.ctrls_hist       = None
    simulator.node_hidden_state = None

    num_rods = len(simulator.robot.rigid_bodies)

    # Build initial quat state from gt_data[0] if not provided
    if start_state is None:
        d0 = gt_data[0]
        state_vals = []
        for r in range(num_rods):
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

    # Convert initial quat state → exp state
    start_exp = quat_state_to_exp_state(start_state)   # (1, 36, 1)
    n_rods    = num_rods
    state_dim = EXP_BLOCK_SIZE * n_rods

    meas_dim  = 6 * n_rods if observe_pose_only else state_dim
    Q_sigmas  = _structured_Q_sigmas(
        state_dim, n_rods, np.sqrt(float(process_noise_scale)),
        exp_inflation, vel_inflation
    )
    R_sigmas  = _structured_R_sigmas(
        meas_dim, n_rods, np.sqrt(float(measurement_noise_scale))
    )

    # H matrix: pose-only maps exp state → [pos, exp_rot] per rod
    if observe_pose_only:
        H_np = np.zeros((meas_dim, state_dim), dtype=np.float64)
        for r in range(n_rods):
            for i in range(6):
                H_np[6 * r + i, EXP_BLOCK_SIZE * r + i] = 1.0
    else:
        H_np = np.eye(state_dim, dtype=np.float64)

    x0_np = start_exp.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
    P0_np = float(measurement_noise_scale) * np.eye(state_dim, dtype=np.float64)
    kf    = gtsam.KalmanFilter(state_dim)
    state_gtsam = kf.init(x0_np, P0_np)

    frames = []
    time   = 0.0
    cached_F = None
    _diag_innov: list[float] = []
    _diag_corr:  list[float] = []

    # First frame — convert exp back to quat for pose output
    pose = _exp_state_to_pose_np(start_exp, n_rods, dtype, device)
    frames.append({"time": time, "pose": pose,
                   "state": start_exp.detach().clone()})

    state_exp_t = start_exp.clone()

    with torch.no_grad():
        for k, extra in enumerate(tqdm.tqdm(extra_gt_data)):
            have_measurement = k + 1 < len(gt_data)

            # Rebuild state_exp_t from current GTSAM mean
            state_exp_t = torch.from_numpy(
                np.array(state_gtsam.mean()).reshape(-1)
            ).to(device=device, dtype=dtype).reshape(1, state_dim, 1)

            ctrl_step = _ensure_ctrl_for_step(extra['controls'], simulator)

            # Recompute Jacobian every jacobian_update_interval steps.
            # linearize_dynamics_exp restores simulator context internally.
            if cached_F is None or k % jacobian_update_interval == 0:
                _, cached_F = linearize_dynamics_exp(
                    simulator, state_exp_t,
                    sample_index=dataset_idx_val,
                    use_finite_diff=use_finite_diff,
                    ctrls=ctrl_step,
                )

            # Build measurement in exp-map space
            z_exp = None
            if have_measurement:
                gt = gt_data[k + 1]
                pos  = np.array(gt['pos'],  dtype=np.float64)
                quat = np.array(gt['quat'], dtype=np.float64)
                if observe_pose_only:
                    z_quat = np.hstack([
                        pos.reshape(n_rods, 3),
                        quat.reshape(n_rods, 4)
                    ]).reshape(-1)                               # (7*n_rods,)
                    z_exp = _pose_quat_to_exp(z_quat, n_rods, dtype, device)
                else:
                    lv = np.array(gt['linvel'], dtype=np.float64)
                    av = np.array(gt['angvel'], dtype=np.float64)
                    z_quat_full = np.hstack([
                        pos.reshape(n_rods, 3),
                        quat.reshape(n_rods, 4),
                        lv.reshape(n_rods, 3),
                        av.reshape(n_rods, 3),
                    ]).reshape(-1)                               # (13*n_rods,)
                    z_exp = _full_quat_state_to_exp_np(
                        z_quat_full, n_rods, dtype, device
                    )

            step_diag = {} if log_diagnostics else None
            mean_np, state_gtsam, kf = _ekf_step_gtsam(
                kf, state_gtsam, simulator, state_exp_t, dt, ctrl_step,
                H_np, z_exp, Q_sigmas, R_sigmas, n_rods, have_measurement,
                use_finite_diff=use_finite_diff,
                innovation_gate_sigma=innovation_gate_sigma,
                dataset_idx_val=dataset_idx_val,
                cached_F=cached_F,
                diagnostics=step_diag,
            )
            if log_diagnostics and step_diag and 'pos_innovation_norm' in step_diag:
                _diag_innov.append(step_diag['pos_innovation_norm'])
                _diag_corr.append(step_diag.get('pos_correction_norm', 0.0))

            state_for_frame = torch.from_numpy(mean_np).to(
                device=device, dtype=dtype
            ).reshape(1, state_dim, 1)

            time += dt
            pose = _exp_state_to_pose_np(state_for_frame, n_rods, dtype, device)
            frames.append({"time": time, "pose": pose,
                           "state": state_for_frame.detach().clone()})

    if log_diagnostics and _diag_innov:
        mean_innov = float(np.mean(_diag_innov))
        mean_corr  = float(np.mean(_diag_corr))
        ratio      = mean_corr / max(mean_innov, 1e-10)
        print(f"\n[EKF position diagnostics — {len(_diag_innov)} steps with measurements]")
        print(f"  Mean pos innovation  ||z_pos - H·x_pred||: {mean_innov:.4f} m")
        print(f"  Mean pos correction  ||x_post - x_pred||[pos]: {mean_corr:.4f} m")
        print(f"  Correction/innovation ratio: {ratio:.4f}")
        print(f"  (ratio≈0 → Kalman gain near zero for position; ratio≈1 → strong correction)")

    return frames


# ---------------------------------------------------------------------------
# Internal helper: convert exp state → flattened (pos, quat) pose for output
# ---------------------------------------------------------------------------

def _exp_state_to_pose_np(state_exp_t: torch.Tensor,
                           n_rods: int,
                           dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
    """Return flattened (pos, quat) per rod from an exp-map state tensor.

    Used to produce pose output compatible with the existing evaluation code
    which expects the same (pos, quat) layout as ekf.py frames.
    """
    quat_state = exp_state_to_quat_state(state_exp_t)  # (1, 39, 1)
    # Extract [pos(3), quat(4)] = first 7 dims per rod
    return quat_state.reshape(-1, 13, 1)[:, :7].flatten()
