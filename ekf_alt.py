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
# import _numpy_kf as gtsam  # Fallback disabled: enforce GTSAM backend.

from ekf import _control_to_numpy_vector, _ensure_ctrl_for_step, _reinit_state_jitter
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    exp_state_to_quat_state,
    linearize_dynamics_exp,
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

def _ekf_step_gtsam(kf, state_gtsam, simulator, state_exp_t, dt, ctrl,
                    H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                    use_finite_diff, innovation_gate_sigma=np.inf,
                    control_jacobian_mode="identity",
                    require_control_jacobian=False,
                    dataset_idx_val=9):
    """One EKF predict-and-update step using GTSAM (exp-map state space).

    Differences from ekf.py:
      - Jacobian is 36×36 from linearize_dynamics_exp (no rank fix needed).
      - Nominal next state computed via step_exp (exp → quat → model → exp).
      - No quaternion renormalization after update.

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
    """
    state_dim = EXP_STATE_DIM
    x_mean = np.array(state_gtsam.mean()).reshape(-1).astype(np.float64)

    # --- 36×36 Jacobian + nominal next state (actual controls) ---------------
    next_state_0_np, F_np = linearize_dynamics_exp(
        simulator, state_exp_t,
        sample_index=dataset_idx_val,
        use_finite_diff=use_finite_diff,
        ctrls=ctrl,
    )

    if not np.all(np.isfinite(F_np)):
        F_np = np.eye(state_dim, dtype=np.float64)

    Q_sigmas_safe = np.maximum(Q_sigmas, 1e-6)
    F_cont = np.ascontiguousarray(F_np, dtype=np.float64)
    u_np   = _control_to_numpy_vector(ctrl)

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

    model_q = gtsam.noiseModel.Diagonal.Sigmas(Q_sigmas_safe)
    try:
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
        state_pred = _reinit_state_jitter(kf, state_pred, state_dim)
    except RuntimeError:
        F_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        u_eff  = np.zeros((state_dim, 1), dtype=np.float64)
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
        state_pred = _reinit_state_jitter(kf, state_pred, state_dim)

    if not have_measurement:
        return np.array(state_pred.mean()).reshape(-1), state_pred

    mean_pred = np.array(state_pred.mean()).reshape(-1).copy()
    try:
        P_pred = np.asarray(state_pred.covariance(), dtype=np.float64)
    except RuntimeError:
        P_pred = np.eye(state_dim, dtype=np.float64) * float(np.mean(Q_sigmas_safe ** 2))
    P_sym      = 0.5 * (P_pred + P_pred.T) + 1e-8 * np.eye(state_dim, dtype=np.float64)
    state_pred = kf.init(mean_pred.reshape(state_dim, 1), P_sym)

    innovation = z_np.reshape(-1) - (H_np @ mean_pred)
    if (np.isfinite(innovation_gate_sigma) and
            np.linalg.norm(innovation) > innovation_gate_sigma * np.sqrt(innovation.size)):
        return mean_pred.copy(), state_pred

    meas_dim     = z_np.size
    z_col        = np.asarray(z_np, dtype=np.float64).reshape(meas_dim, 1)
    R_sigmas_safe = np.maximum(R_sigmas, 1e-6)
    model_r      = gtsam.noiseModel.Diagonal.Sigmas(R_sigmas_safe)
    state_post   = kf.update(state_pred,
                             np.ascontiguousarray(H_np, dtype=np.float64),
                             z_col, model_r)
    state_post = _reinit_state_jitter(kf, state_post, state_dim)
    return np.array(state_post.mean()).reshape(-1).copy(), state_post


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
                 innovation_gate_sigma=np.inf,
                 exp_inflation=1.5, vel_inflation=2.0,
                 dataset_idx_val=9):
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
        self.state_gtsam = self.kf.init(x0_np, P0_np)

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

        with torch.no_grad():
            mean_np, self.state_gtsam = _ekf_step_gtsam(
                self.kf, self.state_gtsam, self.simulator,
                self.state_exp_t, self.dt, ctrl_step,
                self.H_np, z_exp,
                self.Q_sigmas, self.R_sigmas, self.n_rods,
                have_measurement=have_measurement,
                use_finite_diff=self.use_finite_diff,
                innovation_gate_sigma=self.innovation_gate_sigma,
                dataset_idx_val=self.dataset_idx_val,
            )

        self.state_exp_t = torch.tensor(
            mean_np, dtype=self.dtype, device=self.device
        ).view(1, self.state_dim, 1)
        return self.state_exp_t

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
                    vel_inflation=2.0,
                    innovation_gate_sigma=np.inf,
                    dataset_idx_val=9):
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

            mean_np, state_gtsam = _ekf_step_gtsam(
                kf, state_gtsam, simulator, state_exp_t, dt, ctrl_step,
                H_np, z_exp, Q_sigmas, R_sigmas, n_rods, have_measurement,
                use_finite_diff=use_finite_diff,
                innovation_gate_sigma=innovation_gate_sigma,
                dataset_idx_val=dataset_idx_val,
            )

            state_for_frame = torch.from_numpy(mean_np).to(
                device=device, dtype=dtype
            ).reshape(1, state_dim, 1)

            time += dt
            pose = _exp_state_to_pose_np(state_for_frame, n_rods, dtype, device)
            frames.append({"time": time, "pose": pose,
                           "state": state_for_frame.detach().clone()})

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
