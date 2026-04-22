"""
Exp-map linearization utilities (shared module).

State layout per rod (EXP_BLOCK_SIZE = 12):
  indices 0:3   — pos      (x, y, z)
  indices 3:6   — exp_rot  (axis * angle, 3D)
  indices 6:9   — linvel   (vx, vy, vz)
  indices 9:12  — angvel   (wx, wy, wz)

Total exp-map state dim = N_BODIES * EXP_BLOCK_SIZE = 3 * 12 = 36.

Because exp_rot lives in unconstrained R³, the Jacobian is naturally full-rank
(36×36) — no tangent-space projection or ε·qqᵀ regularisation needed.
"""

import numpy as np
import torch

from utilities.torch_quaternion import quat2exp, exp2quat
from linearization import (
    N_BODIES,
    BLOCK_SIZE,
    STATE_DIM,
    _save_model_ctx,
    _restore_model_ctx,
    _build_zero_ctrls,
)

EXP_BLOCK_SIZE = 12
EXP_STATE_DIM  = N_BODIES * EXP_BLOCK_SIZE   # 36


# ---------------------------------------------------------------------------
# State-space conversions
# ---------------------------------------------------------------------------

def quat_state_to_exp_state(state_quat: torch.Tensor) -> torch.Tensor:
    """Convert ambient quat state (39D) to exp-map state (36D).

    Args:
        state_quat: (batch, 39, 1)
    Returns:
        (batch, 36, 1)
    """
    squeeze = (state_quat.dim() == 2)
    if squeeze:
        state_quat = state_quat.unsqueeze(-1)

    batch = state_quat.shape[0]
    s = state_quat.reshape(batch, N_BODIES, BLOCK_SIZE, 1)

    pos     = s[:, :, 0:3,  :]
    quat    = s[:, :, 3:7,  :]
    vel     = s[:, :, 7:13, :]

    quat_flat = quat.reshape(batch * N_BODIES, 4, 1)
    exp_rot   = quat2exp(quat_flat).reshape(batch, N_BODIES, 3, 1)

    exp_state = torch.cat([pos, exp_rot, vel], dim=2).reshape(batch, EXP_STATE_DIM, 1)
    return exp_state.squeeze(-1) if squeeze else exp_state


def exp_state_to_quat_state(state_exp: torch.Tensor) -> torch.Tensor:
    """Convert exp-map state (36D) to ambient quat state (39D).

    Args:
        state_exp: (batch, 36, 1)
    Returns:
        (batch, 39, 1)
    """
    squeeze = (state_exp.dim() == 2)
    if squeeze:
        state_exp = state_exp.unsqueeze(-1)

    batch = state_exp.shape[0]
    s = state_exp.reshape(batch, N_BODIES, EXP_BLOCK_SIZE, 1)

    pos     = s[:, :, 0:3,  :]
    exp_rot = s[:, :, 3:6,  :]
    vel     = s[:, :, 6:12, :]

    quat = exp2quat(exp_rot.reshape(batch * N_BODIES, 3, 1)).reshape(batch, N_BODIES, 4, 1)

    quat_state = torch.cat([pos, quat, vel], dim=2).reshape(batch, STATE_DIM, 1)
    return quat_state.squeeze(-1) if squeeze else quat_state


# ---------------------------------------------------------------------------
# Exp-map wrapped simulator step
# ---------------------------------------------------------------------------

def step_exp(model,
             state_exp: torch.Tensor,
             ctrls: torch.Tensor,
             s2g_kwargs: dict) -> torch.Tensor:
    """One-step dynamics in exp-map state space.

    Converts exp → quat before model.step, then converts first output quat → exp.

    Args:
        model:      TensegrityGNNSimulator
        state_exp:  (1, 36, 1)
        ctrls:      (1, num_cables, T) or None
        s2g_kwargs: state-to-graph keyword arguments

    Returns:
        next_state_exp: (1, 36, 1)
    """
    state_quat = exp_state_to_quat_state(state_exp)
    next_quat, _ = model.step(state_quat, ctrls=ctrls, state_to_graph_kwargs=s2g_kwargs)
    return quat_state_to_exp_state(next_quat[:, :STATE_DIM, 0:1])


# ---------------------------------------------------------------------------
# Jacobian in exp-map space
# ---------------------------------------------------------------------------

def linearize_dynamics_exp(
    model,
    state_exp,
    sample_index: int = 0,
    use_finite_diff: bool = False,
    ctrls=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the exp-map-wrapped one-step dynamics at *state_exp*.

    Returns
    -------
    next_state_exp : (36,) float64 ndarray  — computed with *ctrls* (or zero if None)
    J_exp          : (36, 36) float64 ndarray — naturally full-rank
    """
    if isinstance(state_exp, torch.Tensor):
        state_exp_np = state_exp.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        state_exp_np = np.asarray(state_exp, dtype=np.float64).flatten()

    try:
        ref   = next(model.parameters())
        dtype, dev = ref.dtype, ref.device
    except StopIteration:
        dtype, dev = torch.float32, torch.device('cpu')

    ctx         = _save_model_ctx(model)
    dataset_idx = torch.tensor([[sample_index]], dtype=torch.long, device=dev)
    s2g_kwargs  = {'dataset_idx': dataset_idx}
    ctrls_t     = _build_zero_ctrls(model, dtype, dev)
    nominal_ctrls = ctrls if ctrls is not None else ctrls_t

    def _fwd_np(x_np: np.ndarray, c=ctrls_t) -> np.ndarray:
        x_t = torch.tensor(x_np, dtype=dtype, device=dev).reshape(1, EXP_STATE_DIM, 1)
        with torch.no_grad():
            ns = step_exp(model, x_t, c, s2g_kwargs)
        return ns[0, :EXP_STATE_DIM, 0].detach().cpu().numpy().astype(np.float64)

    if not use_finite_diff:
        def step_fn(x_flat: torch.Tensor) -> torch.Tensor:
            _restore_model_ctx(model, ctx)
            x = x_flat.reshape(1, EXP_STATE_DIM, 1)
            ns = step_exp(model, x, ctrls_t, s2g_kwargs)
            return ns[0, :EXP_STATE_DIM, 0]

        state_in = torch.tensor(state_exp_np, dtype=dtype, device=dev)
        J_np     = torch.func.jacrev(step_fn)(state_in).detach().cpu().numpy().astype(np.float64)

        _restore_model_ctx(model, ctx)
        next_state_np = _fwd_np(state_exp_np, nominal_ctrls)

    else:
        eps_pos = 1e-4
        eps_exp = 1e-5
        eps_vel = 1e-4
        J_np = np.zeros((EXP_STATE_DIM, EXP_STATE_DIM), dtype=np.float64)

        _restore_model_ctx(model, ctx)
        next_state_np = _fwd_np(state_exp_np, nominal_ctrls)

        for j in range(EXP_STATE_DIM):
            off = j % EXP_BLOCK_SIZE
            eps = eps_exp if 3 <= off < 6 else (eps_pos if off < 3 else eps_vel)

            sf = state_exp_np.copy(); sf[j] += eps
            sb = state_exp_np.copy(); sb[j] -= eps

            _restore_model_ctx(model, ctx)
            nsf = _fwd_np(sf)
            _restore_model_ctx(model, ctx)
            nsb = _fwd_np(sb)

            J_np[:, j] = (nsf - nsb) / (2.0 * eps)

    # Restore simulator to pre-linearization state so the caller can
    # call simulator.step() with actual controls without seeing corrupted
    # state left over from the last forward pass above.
    _restore_model_ctx(model, ctx)

    return next_state_np, J_np
