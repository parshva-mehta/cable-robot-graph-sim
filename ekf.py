"""Extended Kalman Filter for tensegrity simulation.

Uses GTSAM for predict/update steps. Linearization is delegated to
linearization.py (linearize_dynamics). State is per-rod:
  13 dims each: 3 pos, 4 quat, 3 linvel, 3 angvel.

Adapted from ekf_old.py for this repository's simulator interface:
  - simulator.step(curr_state, ctrls, state_to_graph_kwargs)
  - State initialized from pos/quat/linvel/angvel (not end_pts)
  - Controls are (1, num_cables, 1) tensors; dataset_idx in state_to_graph_kwargs
"""

import numpy as np
import torch
import tqdm

import gtsam
# import _numpy_kf as gtsam  # Fallback disabled: enforce GTSAM backend.

from linearization import (
    _build_quat_E_matrix,       # noqa: F401  (re-exported for callers)
    _build_tangent_projections,  # noqa: F401
    _fix_jacobian_quaternion_rank,
    linearize_dynamics,
)
from utilities.misc_utils import DEFAULT_DTYPE


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _renormalize_quats_numpy(mean: np.ndarray, n_rods: int) -> None:
    """In-place renormalize quaternion blocks in the state vector.

    State layout: 13 dims per rod; quat occupies indices [13*r+3 : 13*r+7].
    """
    for r in range(n_rods):
        q = mean[13 * r + 3 : 13 * r + 7]
        n = np.linalg.norm(q)
        if n > 1e-10:
            mean[13 * r + 3 : 13 * r + 7] = q / n


# ---------------------------------------------------------------------------
# GTSAM helpers
# ---------------------------------------------------------------------------

def _reinit_state_jitter(kf, state, state_dim, jitter=1e-8):
    """Symmetrize covariance, add jitter, and re-init the GTSAM filter state."""
    mean_col = np.asarray(state.mean(), dtype=np.float64).reshape(state_dim, 1)
    P = np.asarray(state.covariance(), dtype=np.float64)
    P_sym = 0.5 * (P + P.T) + jitter * np.eye(state_dim, dtype=np.float64)
    return kf.init(mean_col, P_sym)


# ---------------------------------------------------------------------------
# Control helpers
# ---------------------------------------------------------------------------

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
    """Convert per-step control to (1, num_cables, 1) tensor on simulator device.

    Accepts a list of num_cables scalar values (as stored in extra_data[k]['controls']),
    a numpy array, or a torch tensor.
    """
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


def _get_simulator_control_jacobian(simulator, state_torch, dt, ctrl, sample_index=0):
    """Query simulator for control Jacobian d f / d u, if available."""
    candidates = [simulator, getattr(simulator, "gnn_sim", None)]
    for cand in candidates:
        if cand is None or not hasattr(cand, "compute_control_jacobian"):
            continue
        fn = cand.compute_control_jacobian
        try:
            J_u = fn(curr_state=state_torch, dt=dt,
                     control_signals=ctrl, sample_index=sample_index)
        except TypeError:
            try:
                J_u = fn(state_torch, dt, ctrl, sample_index)
            except TypeError:
                J_u = fn(state_torch, dt, ctrl)
        if J_u is None:
            continue
        if torch.is_tensor(J_u):
            J_u = J_u.detach().cpu().numpy()
        J_u = np.asarray(J_u, dtype=np.float64)
        if J_u.ndim != 2:
            raise ValueError(
                f"compute_control_jacobian must return 2D, got shape {J_u.shape}"
            )
        return J_u
    return None


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def _structured_Q_sigmas(state_dim, n_rods, base_sigma,
                          quat_inflation=2.0, vel_inflation=2.0):
    """Diagonal process-noise sigmas with per-block scaling."""
    sigmas = np.full(state_dim, base_sigma, dtype=np.float64)
    for r in range(n_rods):
        base = 13 * r
        sigmas[base + 3 : base + 7]  *= quat_inflation
        sigmas[base + 7 : base + 13] *= vel_inflation
    return sigmas


def _structured_R_sigmas(meas_dim, n_rods, pos_sigma, quat_sigma=None):
    """Diagonal measurement-noise sigmas.

    meas_dim == 7*n_rods  → pose-only (pos + quat per rod)
    meas_dim == 13*n_rods → full state
    """
    if quat_sigma is None:
        quat_sigma = 5.0 * pos_sigma
    sigmas = np.empty(meas_dim, dtype=np.float64)
    if meas_dim == 7 * n_rods:
        for r in range(n_rods):
            sigmas[7 * r     : 7 * r + 3] = pos_sigma
            sigmas[7 * r + 3 : 7 * r + 7] = quat_sigma
    else:
        for r in range(n_rods):
            sigmas[13 * r     : 13 * r + 3]  = pos_sigma
            sigmas[13 * r + 3 : 13 * r + 7]  = quat_sigma
            sigmas[13 * r + 7 : 13 * r + 13] = pos_sigma
    return sigmas


# ---------------------------------------------------------------------------
# Core EKF step
# ---------------------------------------------------------------------------

def _ekf_step_gtsam(kf, state_gtsam, simulator, state_torch, dt, ctrl,
                    H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                    use_finite_diff, innovation_gate_sigma=np.inf,
                    control_jacobian_mode="identity",
                    require_control_jacobian=False,
                    dataset_idx_val=9):
    """One EKF predict-and-update step using GTSAM.

    Linearization (Jacobian F and nominal next state) comes from linearize_dynamics.
    The nominal next state is recomputed with the actual controls so that the
    affine term b = f(x; u) - F @ x correctly captures the control effect.

    Args:
        kf: GTSAM KalmanFilter.
        state_gtsam: Current GTSAM state.
        simulator: TensegrityGNNSimulator.
        state_torch: Current state (1, state_dim, 1).
        dt: Time step (unused by linearize_dynamics but kept for API parity).
        ctrl: Per-step controls as (1, num_cables, 1) tensor, or None.
        H_np: Observation matrix (meas_dim, state_dim).
        z_np: Measurement vector (used only when have_measurement=True).
        Q_sigmas: Diagonal process-noise sigmas.
        R_sigmas: Diagonal measurement-noise sigmas.
        n_rods: Number of rods.
        have_measurement: Whether to run the update step.
        use_finite_diff: Passed to linearize_dynamics.
        innovation_gate_sigma: Reject update when innovation norm too large.
        control_jacobian_mode: "identity" or "simulator".
        require_control_jacobian: Raise if simulator Jacobian unavailable.
        dataset_idx_val: Integer dataset index forwarded to the graph processor.

    Returns:
        mean_for_output: State mean (quats renormalized).
        state_gtsam: New GTSAM state.
    """
    state_dim = state_torch.numel()
    x_mean = np.array(state_gtsam.mean()).reshape(-1).astype(np.float64)

    # --- Jacobian F from linearize_dynamics (uses zero controls internally) ---
    _, F_np = linearize_dynamics(
        simulator, state_torch,
        sample_index=dataset_idx_val,
        use_finite_diff=use_finite_diff,
    )

    if not np.all(np.isfinite(F_np)):
        F_np = np.eye(state_dim, dtype=np.float64)
    else:
        F_np = _fix_jacobian_quaternion_rank(F_np, x_mean, n_rods)

    # --- Nominal next state with actual controls ----------------------------
    device = state_torch.device
    dtype = state_torch.dtype
    s2g_kwargs = {'dataset_idx': torch.tensor([[dataset_idx_val]],
                                               dtype=torch.long, device=device)}
    with torch.no_grad():
        step_out = simulator.step(
            state_torch, ctrls=ctrl, state_to_graph_kwargs=s2g_kwargs
        )
    full_next = (step_out[0] if isinstance(step_out, tuple) else step_out).detach()
    next_state_0_np = full_next[0, :state_dim, 0].cpu().numpy().astype(np.float64)

    Q_sigmas_safe = np.maximum(Q_sigmas, 1e-6)
    F_cont = np.ascontiguousarray(F_np, dtype=np.float64)
    u_np = _control_to_numpy_vector(ctrl)

    J_u = None
    if control_jacobian_mode == "simulator":
        J_u = _get_simulator_control_jacobian(
            simulator, state_torch, dt=dt, ctrl=ctrl, sample_index=0
        )
        if J_u is None and require_control_jacobian:
            raise RuntimeError(
                "control_jacobian_mode='simulator' requested but "
                "simulator.compute_control_jacobian is unavailable."
            )

    if J_u is not None and u_np is not None and J_u.shape[1] == u_np.size:
        J_u = np.ascontiguousarray(J_u, dtype=np.float64)
        b_aff = (next_state_0_np - F_np @ x_mean - J_u @ u_np).reshape(state_dim, 1)
        B_cont = np.hstack([J_u, np.eye(state_dim, dtype=np.float64)])
        u_eff = np.concatenate([u_np.reshape(-1), b_aff.reshape(-1)]).reshape(-1, 1)
    else:
        B_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        b_aff = (next_state_0_np - F_np @ x_mean).reshape(state_dim, 1)
        u_eff = b_aff

    # Guard: NaN/Inf in u_eff (from corrupted next_state) causes a C++ segfault
    # inside gtsam — fall back to a zero affine term so predict() stays safe.
    if not np.all(np.isfinite(u_eff)):
        u_eff = np.zeros_like(u_eff)

    model_q = gtsam.noiseModel.Diagonal.Sigmas(Q_sigmas_safe)
    try:
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
        state_pred = _reinit_state_jitter(kf, state_pred, state_dim)
    except RuntimeError:
        F_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        B_cont = np.ascontiguousarray(np.eye(state_dim, dtype=np.float64))
        u_eff  = np.zeros((state_dim, 1), dtype=np.float64)
        state_pred = kf.predict(state_gtsam, F_cont, B_cont, u_eff, model_q)
        state_pred = _reinit_state_jitter(kf, state_pred, state_dim)

    if not have_measurement:
        mean_np = np.array(state_pred.mean()).reshape(-1)
        return mean_np, state_pred

    mean_pred = np.array(state_pred.mean()).reshape(-1).copy()
    try:
        P_pred = np.asarray(state_pred.covariance(), dtype=np.float64)
    except RuntimeError:
        P_pred = np.eye(state_dim, dtype=np.float64) * float(np.mean(Q_sigmas_safe**2))
    P_sym = 0.5 * (P_pred + P_pred.T) + 1e-8 * np.eye(state_dim, dtype=np.float64)
    state_pred = kf.init(mean_pred.reshape(state_dim, 1), P_sym)

    innovation = z_np.reshape(-1) - (H_np @ mean_pred)
    if (np.isfinite(innovation_gate_sigma) and
            np.linalg.norm(innovation) > innovation_gate_sigma * np.sqrt(innovation.size)):
        mean_for_output = mean_pred.copy()
        _renormalize_quats_numpy(mean_for_output, n_rods)
        return mean_for_output, state_pred

    meas_dim = z_np.size
    z_col = np.asarray(z_np, dtype=np.float64).reshape(meas_dim, 1)
    R_sigmas_safe = np.maximum(R_sigmas, 1e-6)
    model_r = gtsam.noiseModel.Diagonal.Sigmas(R_sigmas_safe)
    state_post = kf.update(state_pred,
                           np.ascontiguousarray(H_np, dtype=np.float64),
                           z_col, model_r)
    state_post = _reinit_state_jitter(kf, state_post, state_dim)
    mean_np = np.array(state_post.mean()).reshape(-1).copy()
    _renormalize_quats_numpy(mean_np, n_rods)
    return mean_np, state_post


# ---------------------------------------------------------------------------
# Online (streaming) EKF wrapper
# ---------------------------------------------------------------------------

class OnlineEKF:
    """Streaming EKF wrapper for step-by-step filtering in a timer loop."""

    def __init__(self, simulator, dt, n_rods,
                 process_noise_scale=1e-4, measurement_noise_scale=1e-3,
                 observe_pose_only=False, use_finite_diff=False,
                 innovation_gate_sigma=np.inf,
                 Q_quat_inflation=2.0, Q_vel_inflation=2.0,
                 control_jacobian_mode="identity",
                 require_control_jacobian=False,
                 dataset_idx_val=9):
        self.simulator = simulator
        self.dt = dt
        self.n_rods = n_rods
        self.state_dim = 13 * n_rods
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale
        self.observe_pose_only = observe_pose_only
        self.use_finite_diff = use_finite_diff
        self.innovation_gate_sigma = innovation_gate_sigma
        self.control_jacobian_mode = control_jacobian_mode
        self.require_control_jacobian = require_control_jacobian
        self.dataset_idx_val = dataset_idx_val

        base_Q_sigma = np.sqrt(float(process_noise_scale))
        self.Q_sigmas = _structured_Q_sigmas(
            self.state_dim, n_rods, base_Q_sigma, Q_quat_inflation, Q_vel_inflation
        )
        pos_sigma = np.sqrt(float(measurement_noise_scale))
        if observe_pose_only:
            meas_dim = 7 * n_rods
            self.H_np = np.zeros((meas_dim, self.state_dim), dtype=np.float64)
            for i in range(meas_dim):
                self.H_np[i, i] = 1.0
        else:
            meas_dim = self.state_dim
            self.H_np = np.eye(self.state_dim, dtype=np.float64)
        self.R_sigmas = _structured_R_sigmas(meas_dim, n_rods, pos_sigma)

        self.kf = gtsam.KalmanFilter(self.state_dim)
        self.state_gtsam = None
        self.state_torch = None

    def initialize(self, start_state: torch.Tensor,
                   rest_lengths=None, motor_speeds=None):
        """Initialize EKF state and optionally configure simulator actuators.

        Args:
            start_state: Initial state tensor, shape (1, state_dim, 1) or compatible.
            rest_lengths: List of cable rest lengths.
            motor_speeds: List of motor speeds.
        """
        dtype = getattr(self.simulator, 'dtype', DEFAULT_DTYPE)
        device = getattr(self.simulator, 'device', 'cpu')
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if rest_lengths is not None and motor_speeds is not None:
            cables = list(self.simulator.robot.actuated_cables.values())
            for i, c in enumerate(cables):
                c.actuation_length = c._rest_length - rest_lengths[i]
                c.motor.motor_state.omega_t = torch.tensor(
                    motor_speeds[i], dtype=dtype, device=device
                ).reshape(1, 1, 1)

        start_state = start_state.to(device=device, dtype=dtype)
        if start_state.dim() == 2:
            start_state = start_state.unsqueeze(-1)

        x0_np = start_state.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        P0_np = float(self.measurement_noise_scale) * np.eye(
            self.state_dim, dtype=np.float64
        )
        self.state_gtsam = self.kf.init(x0_np, P0_np)
        self.state_torch = start_state.clone()

    def step(self, z_t: np.ndarray = None, u_t=None,
             have_measurement=True) -> torch.Tensor:
        """Run one EKF predict+update step.

        Args:
            z_t: Measurement vector (state_dim or pose_dim numpy array).
            u_t: Per-step control input (list of scalars, array, or tensor).
            have_measurement: If False, skip the update step.

        Returns:
            Filtered state tensor, shape (1, state_dim, 1).
        """
        if have_measurement and z_t is None:
            raise ValueError("z_t must be provided when have_measurement=True")
        dtype = getattr(self.simulator, 'dtype', DEFAULT_DTYPE)
        device = getattr(self.simulator, 'device', 'cpu')
        if not isinstance(device, torch.device):
            device = torch.device(device)

        ctrl_step = _ensure_ctrl_for_step(u_t, self.simulator)
        with torch.no_grad():
            mean_np, self.state_gtsam = _ekf_step_gtsam(
                self.kf, self.state_gtsam, self.simulator,
                self.state_torch, self.dt, ctrl_step, self.H_np, z_t,
                self.Q_sigmas, self.R_sigmas, self.n_rods,
                have_measurement=have_measurement,
                use_finite_diff=self.use_finite_diff,
                innovation_gate_sigma=self.innovation_gate_sigma,
                control_jacobian_mode=self.control_jacobian_mode,
                require_control_jacobian=self.require_control_jacobian,
                dataset_idx_val=self.dataset_idx_val,
            )
        self.state_torch = torch.tensor(
            mean_np, dtype=dtype, device=device
        ).view(1, self.state_dim, 1)
        return self.state_torch


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
                    use_finite_diff=False,
                    Q_quat_inflation=2.0,
                    Q_vel_inflation=2.0,
                    innovation_gate_sigma=np.inf,
                    control_jacobian_mode="simulator",
                    dataset_idx_val=9):
    """Run an EKF rollout over ground-truth data with predict/update steps.

    Initializes from start_state or from gt_data[0] (pos, quat, linvel, angvel).
    For each timestep: predicts with linearized GNN dynamics, updates with
    the next gt frame as measurement.

    Args:
        simulator: TensegrityGNNSimulator.
        gt_data: List of dicts with 'pos', 'quat', 'linvel', 'angvel' as flat
            lists.  pos: 3*n_rods values, quat: 4*n_rods, etc.
        extra_gt_data: List of dicts with 'controls', 'rest_lengths',
            'motor_speeds'.  'controls': list of num_cables values per step.
        dt: Time step between frames.
        process_noise_scale: Scale for Q (sqrt applied).
        measurement_noise_scale: Scale for R and initial P0.
        observe_pose_only: If True, z is 7*n_rods (pos+quat); else full state.
        start_state: Optional initial state tensor; if None, built from gt_data[0].
        use_finite_diff: Passed to linearize_dynamics.
        Q_quat_inflation: Multiplier for quat block in Q (default 2.0).
        Q_vel_inflation: Multiplier for velocity block in Q (default 2.0).
        innovation_gate_sigma: Reject update when innovation exceeds this ×
            sqrt(meas_dim) (default np.inf = no gating).
        control_jacobian_mode: "identity" or "simulator".
        dataset_idx_val: Integer dataset index for the graph processor
            (matches the value used in eval.py; default 9).

    Returns:
        frames: List of dicts with keys 'time', 'pose', 'state'.
            'state' is a torch tensor (1, state_dim, 1).
            'pose' is flattened (pos, quat) per rod.
    """
    dtype = getattr(simulator, 'dtype', DEFAULT_DTYPE)
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

    simulator.ctrls_hist = None
    simulator.node_hidden_state = None

    num_rods = len(simulator.robot.rigid_bodies)

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

    state_dim = start_state.shape[1]
    n_rods = state_dim // 13
    meas_dim = 7 * n_rods if observe_pose_only else state_dim

    Q_sigmas = _structured_Q_sigmas(
        state_dim, n_rods, np.sqrt(float(process_noise_scale)),
        Q_quat_inflation, Q_vel_inflation
    )
    R_sigmas = _structured_R_sigmas(
        meas_dim, n_rods, np.sqrt(float(measurement_noise_scale))
    )

    x0_np = start_state.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
    P0_np = float(measurement_noise_scale) * np.eye(state_dim, dtype=np.float64)
    kf = gtsam.KalmanFilter(state_dim)
    state_gtsam = kf.init(x0_np, P0_np)

    if observe_pose_only:
        H_np = np.zeros((meas_dim, state_dim), dtype=np.float64)
        for i in range(meas_dim):
            H_np[i, i] = 1.0
    else:
        H_np = np.eye(state_dim, dtype=np.float64)

    frames = []
    time = 0.0
    state_for_frame = start_state
    pose = state_for_frame.reshape(-1, 13, 1)[:, :7].flatten()
    frames.append({"time": time, "pose": pose,
                   "state": state_for_frame.detach().clone()})

    with torch.no_grad():
        for k, extra in enumerate(tqdm.tqdm(extra_gt_data)):
            have_measurement = k + 1 < len(gt_data)

            state_torch = torch.from_numpy(
                np.array(state_gtsam.mean()).reshape(-1)
            ).to(device=device, dtype=dtype).reshape(1, -1, 1)

            ctrl_step = _ensure_ctrl_for_step(extra['controls'], simulator)

            z_np = None
            if have_measurement:
                gt = gt_data[k + 1]
                pos  = np.array(gt['pos'],  dtype=np.float64).reshape(-1, 3)
                quat = np.array(gt['quat'], dtype=np.float64).reshape(-1, 4)
                z_np = np.hstack([pos, quat]).reshape(-1)
                if not observe_pose_only:
                    lv = np.array(gt['linvel'], dtype=np.float64).reshape(-1, 3)
                    av = np.array(gt['angvel'], dtype=np.float64).reshape(-1, 3)
                    z_np = np.hstack([pos, quat, lv, av]).reshape(-1)

            mean_np, state_gtsam = _ekf_step_gtsam(
                kf, state_gtsam, simulator, state_torch, dt, ctrl_step,
                H_np, z_np, Q_sigmas, R_sigmas, n_rods, have_measurement,
                use_finite_diff=use_finite_diff,
                innovation_gate_sigma=innovation_gate_sigma,
                control_jacobian_mode=control_jacobian_mode,
                dataset_idx_val=dataset_idx_val,
            )

            state_for_frame = torch.from_numpy(mean_np).to(
                device=device, dtype=dtype
            ).reshape(1, -1, 1)
            time += dt
            pose = state_for_frame.reshape(-1, 13, 1)[:, :7].flatten()
            frames.append({"time": time, "pose": pose,
                           "state": state_for_frame.detach().clone()})

    return frames
