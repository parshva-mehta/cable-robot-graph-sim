"""GTSAM-based MEKF for tensegrity simulation.

Identical state representation and public API to ekf.py, but delegates the
Kalman predict-update linear algebra to a GTSAM GaussianFactorGraph.

State representation
--------------------
Same as ekf.py:
  Mean       : 39D quat space  (3 pos + 4 quat + 3 linvel + 3 angvel per rod)
  Covariance : 36D exp-map space  (3 pos + 3 exp_rot + 3 linvel + 3 angvel)

GTSAM role
----------
Each Kalman update step builds a two-factor GaussianFactorGraph:
  Factor 1 (prediction prior): ||x - x_pred_exp||²_{P_pred}
  Factor 2 (measurement):      ||H·x - z||²_R
GTSAM's QR/Cholesky solver produces the MAP (= Kalman posterior mean).
The posterior covariance is recovered from the factor graph's information
matrix:  P_post = ( P_pred⁻¹ + Hᵀ R⁻¹ H )⁻¹.

All helpers (conversions, retraction, noise builders, LSTM sync) are
re-used verbatim from ekf.py.

Public API
----------
  run_ekf_rollout(...)  →  list of frame dicts  {'time', 'pose', 'state'}
  OnlineEKF             →  streaming step-by-step wrapper
"""

import gtsam
import numpy as np
import torch
import tqdm

from ekf import (
    _make_pd,
    _control_to_numpy_vector,
    _ensure_ctrl_for_step,
    _pose_quat_to_exp,
    _full_quat_state_to_exp_np,
    _structured_Q_sigmas,
    _structured_R_sigmas,
    _apply_exp_correction,
    _fd_inject_velocities,
    _clamp_velocities_quat,
    _exp_state_to_pose_np,
)
from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
)
from utilities.misc_utils import DEFAULT_DTYPE


# ---------------------------------------------------------------------------
# GTSAM Gaussian factor graph update  (replaces manual Kalman gain)
# ---------------------------------------------------------------------------

def _gtsam_update(x_pred_exp: np.ndarray,
                  P_pred: np.ndarray,
                  H_np: np.ndarray,
                  z_exp: np.ndarray,
                  R_sigmas: np.ndarray,
                  state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Kalman update via GTSAM GaussianFactorGraph.

    Builds the two-factor linear-Gaussian graph and solves for the MAP.
    Mathematically equivalent to the Joseph-form Kalman update in ekf.py
    but uses GTSAM's numerically robust QR/Cholesky elimination.

    Args:
        x_pred_exp : (state_dim,)  predicted state in exp-map space.
        P_pred     : (state_dim, state_dim) predicted covariance.
        H_np       : (meas_dim, state_dim) observation matrix.
        z_exp      : (meas_dim,) measurement in exp-map space.
        R_sigmas   : (meas_dim,) measurement noise standard deviations.
        state_dim  : state dimension (36 for a 3-bar tensegrity).

    Returns:
        delta_exp : (state_dim,) correction vector in exp-map space.
        P_post    : (state_dim, state_dim) posterior covariance.
    """
    key = gtsam.symbol('x', 0)

    P_pred_pd = _make_pd(P_pred)
    R_sigmas_safe = np.maximum(R_sigmas, 1e-9).astype(np.float64)

    gfg = gtsam.GaussianFactorGraph()

    # Factor 1: prediction prior   ||I·x - x_pred_exp||²_{P_pred}
    # This gtsam build's JacobianFactor only accepts a Diagonal noise model, so
    # whiten the full-covariance prior by hand.  With W such that Wᵀ W = P_pred⁻¹,
    #   ||W·x - W·x_pred||²_unit = (x - x_pred)ᵀ P_pred⁻¹ (x - x_pred),
    # which is the same Gaussian prior and contributes Wᵀ W = P_pred⁻¹ to the
    # posterior information matrix recovered below.
    #
    # Whiten from the eigendecomposition rather than cholesky(inv(P_pred)): a
    # blown-up P_pred (unclamped Jacobian) can be conditioned past float64
    # precision, so an explicit inverse loses positive-definiteness.  With
    # P_pred = V diag(λ) Vᵀ (λ floored > 0), W = diag(1/√λ) Vᵀ is exact and stable.
    _evals, _evecs = np.linalg.eigh(P_pred_pd)
    _evals = np.maximum(_evals, 1e-12)
    W_prior = (_evecs / np.sqrt(_evals)).T.astype(np.float64)
    gfg.add(gtsam.JacobianFactor(
        key,
        W_prior,
        W_prior @ x_pred_exp.astype(np.float64),
        gtsam.noiseModel.Diagonal.Sigmas(np.ones(state_dim, dtype=np.float64)),
    ))

    # Factor 2: measurement   ||H·x - z||²_R
    meas_noise = gtsam.noiseModel.Diagonal.Sigmas(R_sigmas_safe)
    gfg.add(gtsam.JacobianFactor(
        key,
        H_np.astype(np.float64),
        z_exp.reshape(-1).astype(np.float64),
        meas_noise,
    ))

    # MAP solve  (GTSAM uses QR / Cholesky on the normal equations)
    solution = gfg.optimize()
    x_post_exp = solution.at(key).astype(np.float64)

    # Posterior covariance: P_post = (Omega_post)^{-1}
    # where Omega_post = sum of factor Hessians = P_pred^{-1} + H^T R^{-1} H
    try:
        omega_post, _ = gfg.hessian()
        P_post = _make_pd(np.linalg.inv(omega_post.astype(np.float64)))
    except Exception:
        # Fallback: compute information matrix directly without GTSAM hessian()
        R_inv_diag = 1.0 / R_sigmas_safe ** 2
        omega_post = np.linalg.inv(P_pred_pd) + H_np.T @ np.diag(R_inv_diag) @ H_np
        P_post = _make_pd(np.linalg.inv(omega_post))

    delta_exp = x_post_exp - x_pred_exp
    return delta_exp, P_post


# ---------------------------------------------------------------------------
# Core MEKF step  (mirrors ekf._ekf_step; Kalman block replaced by GTSAM)
# ---------------------------------------------------------------------------

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
              dataset_idx_val: int = 0,
              diagnostics: dict | None = None,
              x_pred_quat_np: np.ndarray | None = None):
    """One GTSAM-MEKF predict-and-update step.

    Identical contract to ekf._ekf_step.  The only difference is that
    the Kalman gain + Joseph-form block is replaced by _gtsam_update().
    """
    try:
        ref   = next(simulator.parameters())
        dtype = ref.dtype
        dev   = ref.device
    except StopIteration:
        dtype = DEFAULT_DTYPE
        dev   = torch.device('cpu')

    quat_dim  = x_quat_np.size
    state_dim = P_exp.shape[0]
    s2g = {'dataset_idx': torch.tensor([[dataset_idx_val]],
                                        dtype=torch.long, device=dev)}

    x_t = torch.tensor(x_quat_np, dtype=dtype, device=dev).reshape(1, quat_dim, 1)

    # ---- Predict mean: direct GNN step (no exp-map round-trip error) -----
    if x_pred_quat_np is not None:
        x_pred_quat = x_pred_quat_np
    else:
        ctx_pre = _save_model_ctx(simulator)
        with torch.no_grad():
            ns, _ = simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
        x_pred_quat = ns[0, :quat_dim, 0].detach().cpu().numpy().astype(np.float64)
        _restore_model_ctx(simulator, ctx_pre)

    # Sanitize F
    if not np.all(np.isfinite(F_exp)):
        F_exp = np.eye(state_dim, dtype=np.float64)

    # ---- Predict covariance -----------------------------------------------
    Q_sigmas_safe = np.maximum(Q_sigmas, 1e-9)

    # Adaptive Q inflation when state is far from measurement
    if have_measurement and z_exp is not None:
        x_curr_exp = quat_state_to_exp_state(x_t)[0, :state_dim, 0] \
                         .detach().cpu().numpy().astype(np.float64)
        innov_prior = np.linalg.norm(z_exp.reshape(-1) - H_np @ x_curr_exp)
        r_rms       = np.sqrt(np.mean(np.maximum(R_sigmas, 1e-9)**2))
        threshold   = 3.0 * r_rms * np.sqrt(float(z_exp.size))
        if innov_prior > threshold:
            inflate       = min((innov_prior / threshold)**2, 10.0)
            Q_sigmas_safe = Q_sigmas_safe * np.sqrt(inflate)

    Q      = np.diag(Q_sigmas_safe**2)
    P_pred = _make_pd(F_exp @ P_exp @ F_exp.T + Q)

    if not have_measurement:
        if x_pred_quat_np is None:
            with torch.no_grad():
                simulator.step(x_t, ctrls=ctrl, state_to_graph_kwargs=s2g)
        return x_pred_quat, P_pred

    # ---- Innovation for diagnostics / gating (exp-map space) --------------
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

    # ---- Kalman update via GTSAM GaussianFactorGraph ---------------------
    delta_exp, P_post = _gtsam_update(
        x_pred_exp, P_pred, H_np, z_exp, R_sigmas, state_dim
    )
    x_post_quat = _apply_exp_correction(x_pred_quat, delta_exp, n_rods, dtype)

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
# Online (streaming) EKF wrapper  (mirrors ekf.OnlineEKF)
# ---------------------------------------------------------------------------

class OnlineEKF:
    """Streaming GTSAM-MEKF wrapper.

    Identical interface to ekf.OnlineEKF.  Uses _ekf_step (GTSAM version).
    """

    def __init__(self, simulator, dt, n_rods,
                 process_noise_scale=1e-4, measurement_noise_scale=1e-3,
                 observe_pose_only=False, use_finite_diff=True,
                 innovation_gate_sigma=5.0,
                 exp_inflation=1.5, vel_inflation=0.5,
                 dataset_idx_val=0,
                 ema_alpha=0.35,
                 max_linvel=3.0, max_angvel=25.0,
                 jacobian_update_interval=10,
                 max_spectral_radius=None,
                 control_jacobian_mode="simulator",
                 require_control_jacobian=False):
        self.simulator               = simulator
        self.dt                      = dt
        self.n_rods                  = n_rods
        self.state_dim               = EXP_BLOCK_SIZE * n_rods
        self.quat_dim                = 13 * n_rods
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

        self.x_quat_np = None
        self.P_exp     = None

    def initialize(self, start_state: torch.Tensor,
                   rest_lengths=None, motor_speeds=None):
        """Initialize GTSAM-MEKF state from a quat-format tensor (1, 39, 1)."""
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
        """Run one GTSAM-MEKF predict+update step.

        Same interface as ekf.OnlineEKF.step().
        """
        if have_measurement and z_t is None:
            raise ValueError("z_t must be provided when have_measurement=True")

        ctrl_step = _ensure_ctrl_for_step(u_t, self.simulator)

        z_exp = None
        if have_measurement:
            z_exp = self._convert_measurement(z_t)

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

    def _pose_flat_from_measurement(self, z: np.ndarray) -> np.ndarray:
        """Return a (7*n_rods,) [pos quat] array from a measurement.

        Handles both layouts: pose-only (7 per rod, contiguous) and full-state
        (13 per rod, interleaved pos/quat/linvel/angvel).  `_fd_inject_velocities`
        requires a stride-7 pose array, so a plain `[:7*n_rods]` slice silently
        reads the wrong elements for rod >= 1 in the full-state case.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        n = self.n_rods
        if z.size == 13 * n:                       # full-state: extract pos+quat
            out = np.empty(7 * n, dtype=np.float64)
            for r in range(n):
                out[7 * r:7 * r + 7] = z[13 * r:13 * r + 7]
            return out
        return z[:7 * n].copy()                    # pose-only: already stride-7

    def _inject_fd_velocities(self, x_quat_np, z_t):
        if self._prev_z_quat is None:
            return x_quat_np
        z_curr = self._pose_flat_from_measurement(z_t)
        z_prev = self._pose_flat_from_measurement(self._prev_z_quat)
        return _fd_inject_velocities(x_quat_np, z_curr, z_prev, self.dt, self.n_rods)

    def _clamp_velocities_quat(self, x_quat_np):
        return _clamp_velocities_quat(x_quat_np, self.n_rods,
                                       self.max_linvel, self.max_angvel)

    def _convert_measurement(self, z_quat):
        n = self.n_rods
        if self.observe_pose_only:
            return _pose_quat_to_exp(z_quat, n, self.dtype, self.device)
        else:
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
                    use_finite_diff=True,
                    exp_inflation=1.5,
                    vel_inflation=0.5,
                    innovation_gate_sigma=5.0,
                    control_jacobian_mode="simulator",
                    require_control_jacobian=False,
                    dataset_idx_val=0,
                    max_spectral_radius=None,
                    jacobian_update_interval=10,
                    log_diagnostics=False,
                    verbose=False):
    """Run a GTSAM-MEKF rollout over ground-truth data.

    Identical interface to ekf.run_ekf_rollout.  Uses _gtsam_update internally.

    Returns:
        frames: list of dicts {'time', 'pose', 'state'} in exp-map format.
    """
    dtype  = getattr(simulator, 'dtype', DEFAULT_DTYPE)
    device = getattr(simulator, 'device', 'cpu')
    if not isinstance(device, torch.device):
        device = torch.device(device)

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
    state_dim = EXP_BLOCK_SIZE * n_rods
    quat_dim  = 13 * n_rods

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

            ctrl_batch = torch.cat(
                [_ensure_ctrl_for_step(b['controls'], simulator) for b in batch],
                dim=-1,
            )

            x_t_batch = torch.tensor(
                x_quat_np, dtype=dtype, device=device
            ).reshape(1, quat_dim, 1)
            ctx_batch = _save_model_ctx(simulator)
            ns_batch, _ = simulator.step(
                x_t_batch, ctrls=ctrl_batch, state_to_graph_kwargs=s2g
            )
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

            simulator.step(x_t_batch, ctrls=ctrl_batch, state_to_graph_kwargs=s2g)

            batch_start = batch_end

    if log_diagnostics and _diag_innov:
        mean_innov = float(np.mean(_diag_innov))
        mean_corr  = float(np.mean(_diag_corr))
        ratio      = mean_corr / max(mean_innov, 1e-10)
        print(f"\n[GTSAM-EKF diagnostics — {len(_diag_innov)} steps]")
        print(f"  Mean pos innovation  ||z - H·x_pred||: {mean_innov:.4f} m")
        print(f"  Mean pos correction  ||x_post - x_pred||[pos]: {mean_corr:.4f} m")
        print(f"  Correction/innovation ratio: {ratio:.4f}")

    return frames
