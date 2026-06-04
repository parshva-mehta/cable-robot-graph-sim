"""Phase 0 diagnostic: localize the Jacobian-quality failure by block.

Hypothesis (from CLAUDE.md analysis):
  - The dynamics are non-smooth in velocity & rotation input directions
    (1/dt node-vel extraction, contact-tangent v/|v| kink, acos angvel).
  - So F@δ carries first-order information for POSITION input directions but
    not for ROTATION / VELOCITY input directions.
  - The failing test perturbs the full 36-D state, so it fails by construction.

This script:
  1. Warms up the LSTM (same as test_jacobian_quality).
  2. Computes F via central finite differences.
  3. Prints the 4x4 block-Frobenius map of F (out-block x in-block, summed
     over rods) to expose which entries are huge.
  4. Runs the F@δ ≈ f(x+δ)-f(x) check for perturbation directions restricted
     to each INPUT block (pos / rot / linvel / angvel) and reports the
     resulting OUTPUT-block rel-error breakdown.

Run:
    python tests/phase0_jacobian_diagnostic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE, EXP_STATE_DIM,
    quat_state_to_exp_state, linearize_dynamics_exp, step_exp,
)
from utilities.misc_utils import DEFAULT_DTYPE
from ekf import _ensure_ctrl_for_step

from tests.test_jacobian_ekf import (
    _DEFAULT_MODEL, _DEFAULT_DATA,
    _load_assets, _make_start_state, _reset_sim,
)

BLOCKS = ["pos", "rot", "linvel", "angvel"]   # offsets 0:3, 3:6, 6:9, 9:12


def _block_indices(n_rods):
    """Return dict block_name -> np.array of global indices across all rods."""
    out = {b: [] for b in BLOCKS}
    for r in range(n_rods):
        base = r * EXP_BLOCK_SIZE
        out["pos"]    += list(range(base + 0, base + 3))
        out["rot"]    += list(range(base + 3, base + 6))
        out["linvel"] += list(range(base + 6, base + 9))
        out["angvel"] += list(range(base + 9, base + 12))
    return {k: np.array(v) for k, v in out.items()}


def main():
    device = torch.device("cpu")
    dtype = DEFAULT_DTYPE
    dataset_idx = 9
    warm_up_steps = 5

    sim, gt_data, extra_data = _load_assets(_DEFAULT_MODEL, _DEFAULT_DATA, device)
    n_rods = len(sim.robot.rigid_bodies)
    bidx = _block_indices(n_rods)
    start = _make_start_state(gt_data, n_rods, device)
    _reset_sim(sim, extra_data, device)

    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=device)}

    # Warm up LSTM (identical to test_jacobian_quality).
    x_quat = start.clone()
    with torch.no_grad():
        for k in range(warm_up_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            ns, _ = sim.step(x_quat, ctrls=ctrl, state_to_graph_kwargs=s2g)
            x_quat = ns[..., 0:1]

    ctrl = _ensure_ctrl_for_step(extra_data[warm_up_steps]["controls"], sim)
    state_exp = quat_state_to_exp_state(x_quat)

    ctx = _save_model_ctx(sim)
    f_x_np, F = linearize_dynamics_exp(
        sim, state_exp, sample_index=dataset_idx, use_finite_diff=True,
        ctrls=ctrl, max_spectral_radius=1.0, verbose=False,
    )
    x_exp_np = state_exp.squeeze().cpu().numpy().astype(np.float64)

    def gnn_exp(x_np):
        _restore_model_ctx(sim, ctx)
        x_t = torch.tensor(x_np, dtype=dtype, device=device).reshape(1, EXP_STATE_DIM, 1)
        with torch.no_grad():
            ns = step_exp(sim, x_t, ctrl, s2g)
        return ns[0, :EXP_STATE_DIM, 0].cpu().numpy().astype(np.float64)

    # ---- (3) Block-Frobenius map of F: out-block (row) x in-block (col) ----
    print("\n=== F block-Frobenius map  (rows = OUTPUT block, cols = INPUT block) ===")
    print(f"{'':>8s}" + "".join(f"{c:>12s}" for c in BLOCKS))
    for ob in BLOCKS:
        row = []
        for ib in BLOCKS:
            sub = F[np.ix_(bidx[ob], bidx[ib])]
            row.append(np.linalg.norm(sub))
        print(f"{ob:>8s}" + "".join(f"{v:12.3e}" for v in row))
    print(f"\nF overall: |max|={np.abs(F).max():.3e}  frob={np.linalg.norm(F):.3e}  "
          f"spectral_radius={np.max(np.abs(np.linalg.eigvals(F))):.3f}")

    # ---- (4) Per-INPUT-block perturbation, OUTPUT-block rel-error breakdown ----
    eps_list = [1e-2, 1e-3, 1e-4]
    rng = np.random.default_rng(0)

    # Build perturbation directions: "full" plus one per input block.
    dirs = {}
    full = rng.standard_normal(EXP_STATE_DIM); full /= np.linalg.norm(full)
    dirs["full"] = full
    for ib in BLOCKS:
        d = np.zeros(EXP_STATE_DIM)
        d[bidx[ib]] = rng.standard_normal(len(bidx[ib]))
        d /= np.linalg.norm(d)
        dirs[f"in:{ib}"] = d

    for dname, ddir in dirs.items():
        print(f"\n=== perturbation direction = {dname} ===")
        for eps in eps_list:
            delta = eps * ddir
            actual = gnn_exp(x_exp_np + delta) - f_x_np
            pred = F @ delta
            resid = actual - pred
            overall = np.linalg.norm(resid) / max(np.linalg.norm(actual), 1e-15)
            # output-block breakdown
            parts = []
            for ob in BLOCKS:
                a = np.linalg.norm(actual[bidx[ob]])
                rr = np.linalg.norm(resid[bidx[ob]]) / max(a, 1e-15)
                parts.append(f"{ob}={rr:5.2f}(|Δ|={a:.1e})")
            print(f"  eps={eps:.0e}  overall_rel={overall:5.2f} | " + "  ".join(parts))


if __name__ == "__main__":
    main()
