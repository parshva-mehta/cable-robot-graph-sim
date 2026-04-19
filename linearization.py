"""
Jacobian / linearization utilities for TensegrityGNNSimulator.

State layout per rod (BLOCK_SIZE = 13):
  indices 0:3   — pos     (x, y, z)
  indices 3:7   — quat    (w, x, y, z)
  indices 7:10  — linvel  (vx, vy, vz)
  indices 10:13 — angvel  (wx, wy, wz)

Total state dim = N_BODIES * BLOCK_SIZE = 3 * 13 = 39.
Tangent dim     = N_BODIES * 12         = 3 * 12 = 36
  (quaternion 4D → 3D tangent reduces each body by 1).
"""

import numpy as np
import torch

N_BODIES = 3
BLOCK_SIZE = 13
STATE_DIM = N_BODIES * BLOCK_SIZE       # 39
QUAT_OFFSET = 3                         # offset of quat within a body block
QUAT_SIZE = 4
TANGENT_BLOCK_SIZE = 12                 # 3+3+3+3 (quat 4→3 tangent)
TANGENT_DIM = N_BODIES * TANGENT_BLOCK_SIZE  # 36


# ---------------------------------------------------------------------------
# Core quaternion tangent-space helpers
# ---------------------------------------------------------------------------

def _build_quat_E_matrix(q: np.ndarray) -> np.ndarray:
    """
    Return the 4×3 right-perturbation tangent basis matrix E for a unit
    quaternion q = [w, x, y, z].

    E is the Jacobian of the right-composition map at δ = 0:
        q ⊗ [1, δ₁, δ₂, δ₃]

    Properties:
      E.T @ q  == 0         (columns tangent to q on S³)
      E.T @ E  == I_3       (orthonormal — required so T_out @ T_in == I)
      E @ E.T  == I_4 - q q.T
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    w, x, y, z = q
    # No 0.5 factor: keeps columns orthonormal (E.T @ E == I_3).
    # The 0.5 arises in quaternion kinematics (q̇ = 0.5 E ω) but is NOT
    # included here because the projection T_out @ T_in must equal I_tangent.
    E = np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w],
    ], dtype=np.float64)
    return E


def _build_tangent_projections(
    state_mean: np.ndarray,
    n_bodies: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build projection matrices between the ambient state space and the
    quaternion-constrained tangent space.

    Returns
    -------
    T_out : (tangent_dim × ambient_dim) float64 ndarray
        Projects ambient → tangent.  For each body block:
          pos / linvel / angvel rows  →  identity
          quat row block              →  E.T  (3×4)
    T_in : (ambient_dim × tangent_dim) float64 ndarray
        Lifts tangent → ambient.  Corresponds to:
          pos / linvel / angvel rows  →  identity
          quat row block              →  E    (4×3)
    """
    state_mean = np.asarray(state_mean, dtype=np.float64).flatten()
    ambient_dim = n_bodies * BLOCK_SIZE
    tangent_dim = n_bodies * TANGENT_BLOCK_SIZE

    T_out = np.zeros((tangent_dim, ambient_dim), dtype=np.float64)
    T_in  = np.zeros((ambient_dim, tangent_dim), dtype=np.float64)

    for i in range(n_bodies):
        a = i * BLOCK_SIZE          # base index in ambient
        t = i * TANGENT_BLOCK_SIZE  # base index in tangent

        q = state_mean[a + QUAT_OFFSET: a + QUAT_OFFSET + QUAT_SIZE]
        norm = np.linalg.norm(q)
        q = q / norm if norm > 1e-8 else q
        E = _build_quat_E_matrix(q)  # (4, 3)

        # pos (3×3 identity)
        T_out[t:t+3,     a:a+3]     = np.eye(3)
        T_in [a:a+3,     t:t+3]     = np.eye(3)

        # quat  E.T (3×4) / E (4×3)
        T_out[t+3:t+6,   a+3:a+7]  = E.T
        T_in [a+3:a+7,   t+3:t+6]  = E

        # linvel (3×3 identity)
        T_out[t+6:t+9,   a+7:a+10] = np.eye(3)
        T_in [a+7:a+10,  t+6:t+9]  = np.eye(3)

        # angvel (3×3 identity)
        T_out[t+9:t+12,  a+10:a+13] = np.eye(3)
        T_in [a+10:a+13, t+9:t+12]  = np.eye(3)

    return T_out, T_in


# ---------------------------------------------------------------------------
# Jacobian rank-fix
# ---------------------------------------------------------------------------

def _fix_jacobian_quaternion_rank(
    F_np: np.ndarray,
    state_mean: np.ndarray,
    n_bodies: int,
    eps: float = 1.0,
    max_spectral_radius: float = 1.0,
) -> np.ndarray:
    """
    Repair rank deficiency in a state-transition Jacobian caused by the
    unit-quaternion constraint.

    Steps
    -----
    a. Project to tangent space:   F_reduced = T_out @ F @ T_in
    b. Clamp spectral radius if    max(|eig(F_reduced)|) > max_spectral_radius
    c. Lift back:                  F_fixed = T_in @ F_reduced @ T_out
    d. Add eps * outer(q, q) to each body's 4×4 quaternion diagonal block
       to restore full rank in the ambient representation.

    Parameters
    ----------
    F_np               : (state_dim, state_dim) Jacobian in ambient space
    state_mean         : (state_dim,) linearisation point
    n_bodies           : number of rigid bodies (rods)
    eps                : weight for the outer(q,q) regulariser
    max_spectral_radius: spectral radius clamp threshold

    Returns
    -------
    F_fixed : (state_dim, state_dim) float64 ndarray
    """
    state_mean = np.asarray(state_mean, dtype=np.float64).flatten()
    F_np = np.asarray(F_np, dtype=np.float64)

    T_out, T_in = _build_tangent_projections(state_mean, n_bodies)

    # a. Project
    F_reduced = T_out @ F_np @ T_in          # (tangent_dim, tangent_dim)

    # b. Clamp spectral radius
    eigenvalues = np.linalg.eigvals(F_reduced)
    sr = float(np.max(np.abs(eigenvalues)))
    if sr > max_spectral_radius:
        F_reduced = F_reduced * (max_spectral_radius / sr)

    # c. Lift back
    F_fixed = T_in @ F_reduced @ T_out       # (ambient_dim, ambient_dim)

    # d. Regularise each body's quaternion block
    for i in range(n_bodies):
        qs = i * BLOCK_SIZE + QUAT_OFFSET
        qe = qs + QUAT_SIZE
        q  = state_mean[qs:qe]
        norm = np.linalg.norm(q)
        q = q / norm if norm > 1e-8 else q
        F_fixed[qs:qe, qs:qe] += eps * np.outer(q, q)

    return F_fixed


# ---------------------------------------------------------------------------
# Internal helpers for linearize_dynamics
# ---------------------------------------------------------------------------

def _build_zero_ctrls(model, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build a zero-control tensor with the batch / cable dimensions the model
    expects.  step() pads and prepends ctrls_hist, so we pass shape
    (1, num_actuated_cables, 1).
    """
    num_cables = len(model.robot.actuated_cables)
    return torch.zeros(1, num_cables, 1, dtype=dtype, device=device)


def _save_model_ctx(model) -> dict:
    """Snapshot the mutable per-step context carried by the simulator."""
    act_cables = list(model.robot.actuated_cables.values())
    return {
        'ctrls_hist': (
            model.ctrls_hist.clone() if model.ctrls_hist is not None else None
        ),
        'node_hidden_state': (
            model.node_hidden_state.clone()
            if model.node_hidden_state is not None
            else None
        ),
        # actuation_length is mutated by set_rest_length() inside _process_gnn
        # when use_cable_decoder=True; must be restored between perturbations.
        'actuation_lengths': [
            c.actuation_length.clone() if c.actuation_length is not None else None
            for c in act_cables
        ],
    }


def _restore_model_ctx(model, ctx: dict) -> None:
    """Restore the mutable per-step context from a snapshot."""
    model.ctrls_hist = (
        ctx['ctrls_hist'].clone() if ctx['ctrls_hist'] is not None else None
    )
    model.node_hidden_state = (
        ctx['node_hidden_state'].clone()
        if ctx['node_hidden_state'] is not None
        else None
    )
    act_cables = list(model.robot.actuated_cables.values())
    for cable, saved_len in zip(act_cables, ctx['actuation_lengths']):
        cable.actuation_length = (
            saved_len.clone() if saved_len is not None else None
        )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def linearize_dynamics(
    model,
    state,
    dt=None,
    sample_index: int = 0,
    use_finite_diff: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearize the one-step dynamics map at *state*.

    Returns
    -------
    next_state : (state_dim,) float64 ndarray
        One-step prediction from the GNN at the given state.
    J          : (state_dim, state_dim) float64 ndarray
        Jacobian  ∂next_state / ∂state  at the given state.

    Parameters
    ----------
    model          : TensegrityGNNSimulator  (should be in eval() mode)
    state          : (state_dim,) numpy array or 1-D / (1,state_dim,1) Tensor
    dt             : unused; kept for API consistency (model carries its own dt)
    sample_index   : dataset_idx forwarded to the graph data processor
    use_finite_diff: if True, central finite differences; otherwise torch.func.jacrev
    """
    # ---- Normalise input ------------------------------------------------
    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        state_np = np.asarray(state, dtype=np.float64).flatten()

    state_dim = len(state_np)

    try:
        ref = next(model.parameters())
        dtype, dev = ref.dtype, ref.device
    except StopIteration:
        dtype, dev = torch.float32, torch.device('cpu')

    # Snapshot mutable simulator context so we can restore it after each call
    ctx = _save_model_ctx(model)

    dataset_idx = torch.tensor([[sample_index]], dtype=torch.long, device=dev)
    s2g_kwargs  = {'dataset_idx': dataset_idx}
    ctrls_t     = _build_zero_ctrls(model, dtype, dev)

    # ---- Thin wrapper for a no-grad forward pass ------------------------
    def _fwd_np(x_np: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x_np, dtype=dtype, device=dev).reshape(1, state_dim, 1)
        with torch.no_grad():
            ns, _ = model.step(x_t, ctrls=ctrls_t, state_to_graph_kwargs=s2g_kwargs)
        return ns[0, :state_dim, 0].detach().cpu().numpy().astype(np.float64)

    # =====================================================================
    if not use_finite_diff:
        # ---- Autodiff via torch.func.jacrev -----------------------------
        # step_fn is called ONCE (forward pass); jacrev does N backward VJPs.
        def step_fn(x_flat: torch.Tensor) -> torch.Tensor:
            _restore_model_ctx(model, ctx)
            x = x_flat.reshape(1, state_dim, 1)
            ns, _ = model.step(x, ctrls=ctrls_t, state_to_graph_kwargs=s2g_kwargs)
            return ns[0, :state_dim, 0]   # (state_dim,)

        state_in = torch.tensor(state_np, dtype=dtype, device=dev)
        J_t = torch.func.jacrev(step_fn)(state_in)   # (state_dim, state_dim)
        J_np = J_t.detach().cpu().numpy().astype(np.float64)

        # Clean forward pass for next_state
        _restore_model_ctx(model, ctx)
        next_state_np = _fwd_np(state_np)

    else:
        # ---- Central finite differences ---------------------------------
        eps_pos  = 1e-4
        eps_quat = 1e-6
        eps_vel  = 1e-4

        J_np = np.zeros((state_dim, state_dim), dtype=np.float64)

        _restore_model_ctx(model, ctx)
        next_state_np = _fwd_np(state_np)

        for j in range(state_dim):
            body_j = j // BLOCK_SIZE
            offset = j % BLOCK_SIZE

            if QUAT_OFFSET <= offset < QUAT_OFFSET + QUAT_SIZE:
                eps = eps_quat
            elif offset < 3:
                eps = eps_pos
            else:
                eps = eps_vel

            def _perturbed(sign: float) -> np.ndarray:
                s = state_np.copy()
                s[j] += sign * eps
                # Renormalise the body's quaternion after perturbation
                qs = body_j * BLOCK_SIZE + QUAT_OFFSET
                qe = qs + QUAT_SIZE
                qn = np.linalg.norm(s[qs:qe])
                if qn > 1e-8:
                    s[qs:qe] /= qn
                return s

            _restore_model_ctx(model, ctx)
            ns_fwd = _fwd_np(_perturbed(+1.0))

            _restore_model_ctx(model, ctx)
            ns_bwd = _fwd_np(_perturbed(-1.0))

            J_np[:, j] = (ns_fwd - ns_bwd) / (2.0 * eps)

    # Restore simulator to pre-linearization state so the caller (e.g.
    # _ekf_step_gtsam) can call simulator.step() with actual controls
    # without seeing corrupted ctrls_hist / node_hidden_state / cable
    # actuation_lengths left over from the last forward pass above.
    _restore_model_ctx(model, ctx)

    return next_state_np, J_np
