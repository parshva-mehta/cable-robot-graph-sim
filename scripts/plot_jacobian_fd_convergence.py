"""Visual proof that the EKF Jacobian F is the true first-order linearization.

For the dynamics f and a perturbation δ = eps * d (‖d‖ = 1):

    f(x + δ) = f(x) + F δ + O(‖δ‖²)

so the linearization residual

    r(eps) = ‖ (f(x + δ) - f(x)) - F δ ‖

must scale as O(eps²): on a log-log plot of r vs eps it traces a line of
*slope 2* until eps shrinks enough that floating-point round-off in f
dominates (the residual floors out / turns up).  A wrong F gives slope 1
(or a flat, non-vanishing residual) instead.

This is the cleanest single check that the Jacobian *arithmetic* is correct,
and it reuses the exact same F that the EKF uses in P = F P Fᵀ + Q.

Usage:
    conda run -n cable_robot_gnn python scripts/plot_jacobian_fd_convergence.py
    [--model_path ...] [--data_dir ...] [--out fd_convergence.png]
    [--n_dirs 8] [--warm_up_steps 5]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities.misc_utils import DEFAULT_DTYPE
from linearization import _save_model_ctx, _restore_model_ctx
from linearization_exp import (
    EXP_BLOCK_SIZE,
    EXP_STATE_DIM,
    quat_state_to_exp_state,
    linearize_dynamics_exp,
    step_exp,
)
from ekf import _ensure_ctrl_for_step

# Reuse the same asset loading / warm-up helpers the Jacobian tests use, so the
# linearization point here is identical to the one the test suite validates.
from tests.test_jacobian_ekf import (
    _load_assets,
    _make_start_state,
    _reset_sim,
    _DEFAULT_MODEL,
    _DEFAULT_DATA,
)


def compute_convergence(model_path, data_dir, device,
                        warm_up_steps=5, n_dirs=8,
                        eps_list=None, dataset_idx=9, seed=0):
    """Return (eps_list, pose_residual, full_residual, pose_ref, full_ref).

    *_residual arrays are the mean linearization residual over n_dirs random
    unit directions at each eps; *_ref are the mean ‖f(x+δ)-f(x)‖ used to
    normalize into a relative error.  pose_* restricts to pos+exp_rot rows.
    """
    if eps_list is None:
        eps_list = np.logspace(-1, -6, 11)

    sim, gt_data, extra_data = _load_assets(model_path, data_dir, device)
    n_rods = len(sim.robot.rigid_bodies)
    start = _make_start_state(gt_data, n_rods, device)
    _reset_sim(sim, extra_data, device)

    dtype = DEFAULT_DTYPE
    s2g = {"dataset_idx": torch.tensor([[dataset_idx]], dtype=torch.long, device=device)}

    pose_idx = np.array([
        r * EXP_BLOCK_SIZE + j
        for r in range(n_rods)
        for j in range(6)
    ])

    # Warm up the LSTM so the linearization point is a realistic mid-rollout state.
    x_quat = start.clone()
    with torch.no_grad():
        for k in range(warm_up_steps):
            ctrl = _ensure_ctrl_for_step(extra_data[k]["controls"], sim)
            ns, _ = sim.step(x_quat, ctrls=ctrl, state_to_graph_kwargs=s2g)
            x_quat = ns[..., 0:1]

    ctrl = _ensure_ctrl_for_step(extra_data[warm_up_steps]["controls"], sim)
    state_exp = quat_state_to_exp_state(x_quat)

    # F and the nominal next state f(x); no SR clamping (true local linearization).
    ctx = _save_model_ctx(sim)
    f_x_np, F = linearize_dynamics_exp(
        sim, state_exp,
        sample_index=dataset_idx,
        use_finite_diff=True,
        ctrls=ctrl,
        verbose=False,
    )
    x_exp_np = state_exp.squeeze().cpu().numpy().astype(np.float64)

    def _gnn_exp(x_np):
        _restore_model_ctx(sim, ctx)
        x_t = torch.tensor(x_np, dtype=dtype, device=device).reshape(1, EXP_STATE_DIM, 1)
        with torch.no_grad():
            ns = step_exp(sim, x_t, ctrl, s2g)
        return ns[0, :EXP_STATE_DIM, 0].detach().cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, EXP_STATE_DIM))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    eps_arr = np.asarray(eps_list, dtype=np.float64)
    pose_res = np.zeros(len(eps_arr))
    full_res = np.zeros(len(eps_arr))
    pose_ref = np.zeros(len(eps_arr))
    full_ref = np.zeros(len(eps_arr))

    for i, eps in enumerate(eps_arr):
        for d in dirs:
            delta = eps * d
            actual = _gnn_exp(x_exp_np + delta) - f_x_np
            resid = actual - F @ delta
            pose_res[i] += np.linalg.norm(resid[pose_idx])
            full_res[i] += np.linalg.norm(resid)
            pose_ref[i] += np.linalg.norm(actual[pose_idx])
            full_ref[i] += np.linalg.norm(actual)
        pose_res[i] /= n_dirs
        full_res[i] /= n_dirs
        pose_ref[i] /= n_dirs
        full_ref[i] /= n_dirs

    return eps_arr, pose_res, full_res, pose_ref, full_ref


def make_plot(eps, pose_res, full_res, pose_ref, full_ref, out_path):
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: absolute residual vs eps, with slope-1 and slope-2 references ---
    ax_abs.loglog(eps, pose_res, "o-", color="C0", label="pose rows (pos+exp_rot)")
    ax_abs.loglog(eps, full_res, "s--", color="C1", alpha=0.7, label="full state")

    # Anchor reference lines at the largest eps where round-off is negligible.
    anchor = pose_res[0] / (eps[0] ** 2)
    ax_abs.loglog(eps, anchor * eps ** 2, "k:", alpha=0.6,
                  label="slope 2 (correct F)")
    anchor1 = pose_res[0] / eps[0]
    ax_abs.loglog(eps, anchor1 * eps, color="0.6", ls=":", alpha=0.6,
                  label="slope 1 (wrong F)")

    ax_abs.set_xlabel("perturbation size  eps  =  ‖δ‖")
    ax_abs.set_ylabel("linearization residual  ‖(f(x+δ)-f(x)) - Fδ‖")
    ax_abs.set_title("Jacobian FD convergence (absolute)")
    ax_abs.grid(True, which="both", alpha=0.3)
    ax_abs.legend(fontsize=8)

    # --- Right: relative error vs eps ---
    pose_rel = pose_res / np.maximum(pose_ref, 1e-300)
    full_rel = full_res / np.maximum(full_ref, 1e-300)
    ax_rel.loglog(eps, pose_rel, "o-", color="C0", label="pose rows")
    ax_rel.loglog(eps, full_rel, "s--", color="C1", alpha=0.7, label="full state")
    ax_rel.axhline(0.05, color="r", ls=":", alpha=0.6, label="5% test threshold")
    ax_rel.set_xlabel("perturbation size  eps")
    ax_rel.set_ylabel("relative residual  ‖resid‖ / ‖f(x+δ)-f(x)‖")
    ax_rel.set_title("Jacobian FD convergence (relative)")
    ax_rel.grid(True, which="both", alpha=0.3)
    ax_rel.legend(fontsize=8)

    fig.suptitle(
        "EKF Jacobian validity: residual ∝ eps²  ⇒  F is the true first-order "
        "linearization", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved plot → {out_path}")


def _empirical_slope(eps, res, floor_factor=3.0):
    """Log-log slope of the *converging* region only.

    Below some eps the residual stops shrinking — it hits the floating-point
    round-off floor of f and flattens out.  Including those points would pull
    the fitted slope toward 0, so we fit only points whose residual is still
    well above the floor (> floor_factor × min residual).

    Returns (slope, floor_index).  A correct first-order F yields a clearly
    positive slope (ideally 2; ~1.3–1.7 in practice because F is itself an FD
    approximation and the GNN has mild non-smoothness).  A wrong F gives slope
    ≈ 1 *and* a relative residual that never drops below O(1)."""
    res = np.asarray(res)
    floor = res.min()
    floor_i = int(np.argmin(res))
    converging = res > floor_factor * floor
    if converging.sum() < 2:
        converging[:2] = True
    log_e = np.log(eps[converging])
    log_r = np.log(np.maximum(res[converging], 1e-300))
    sl = np.polyfit(log_e, log_r, 1)[0]
    return sl, floor_i


def write_txt(eps, pose_res, full_res, pose_ref, full_ref, out_path):
    """Write the sweep as a plain whitespace-delimited matrix, eval.py-style.

    One row per eps sample, loadable with `np.loadtxt(path)`.  Columns:
        0 eps          1 pose_resid   2 full_resid
        3 pose_rel     4 full_rel     5 slope1_ref   6 slope2_ref
    pose_resid is the EKF-critical curve; on a log-log plot it should fall
    between slope1_ref (wrong F) and slope2_ref (correct F) until the
    round-off floor.  pose_rel dipping below 0.05 is the relative-error proof.
    Reference lines are anchored at the largest eps (matches the PNG).
    """
    pose_rel = pose_res / np.maximum(pose_ref, 1e-300)
    full_rel = full_res / np.maximum(full_ref, 1e-300)
    s2 = (pose_res[0] / eps[0] ** 2) * eps ** 2   # slope-2 reference
    s1 = (pose_res[0] / eps[0]) * eps             # slope-1 reference

    cols = ["eps", "pose_resid", "full_resid",
            "pose_rel", "full_rel", "slope1_ref", "slope2_ref"]
    data = np.column_stack([eps, pose_res, full_res, pose_rel, full_rel, s1, s2])
    with open(out_path, "w") as f:
        f.write("# Jacobian FD convergence  (one row per eps sample)\n")
        f.write("# " + " ".join(f"{c}[{i}]" for i, c in enumerate(cols)) + "\n")
        for row in data:
            f.write(" ".join(f"{v:.8e}" for v in row) + "\n")
    print(f"Saved TXT   → {out_path}  ({len(eps)} rows, {len(cols)} cols, "
          f"np.loadtxt-ready)")


def write_mcap(eps, pose_res, full_res, pose_ref, full_ref, out_path):
    """Write the convergence sweep as an MCAP file for Foxglove.

    One JSON message per eps on topic /jacobian_fd.  In Foxglove, drag the
    file in, add a Plot panel, set the panel's **X Axis → "msg path"** with
    path `/jacobian_fd.eps` (or `.log_eps`), then add Y series:
        /jacobian_fd.pose_resid   /jacobian_fd.full_resid
        /jacobian_fd.slope2_ref   /jacobian_fd.slope1_ref
    Enable the Y-axis log-scale option, OR plot the `log_*` fields against
    `log_eps` on a linear panel — then the convergence ORDER is literally the
    slope of the line (correct F → 2, wrong F → 1).
    """
    from mcap.writer import Writer

    pose_rel = pose_res / np.maximum(pose_ref, 1e-300)
    full_rel = full_res / np.maximum(full_ref, 1e-300)
    # Reference lines anchored at the largest eps (same as the matplotlib plot).
    s2 = (pose_res[0] / eps[0] ** 2) * eps ** 2
    s1 = (pose_res[0] / eps[0]) * eps

    def _log10(v):
        return float(np.log10(max(float(v), 1e-300)))

    schema = {
        "type": "object",
        "properties": {
            k: {"type": "number"} for k in (
                "eps", "log_eps",
                "pose_resid", "log_pose_resid",
                "full_resid", "log_full_resid",
                "pose_rel", "full_rel",
                "slope1_ref", "slope2_ref",
                "log_slope1_ref", "log_slope2_ref",
                "rel_threshold",
            )
        },
    }

    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start()
        sid = writer.register_schema(
            name="JacobianFD", encoding="jsonschema",
            data=json.dumps(schema).encode(),
        )
        cid = writer.register_channel(
            topic="/jacobian_fd", message_encoding="json", schema_id=sid,
        )
        # eps decreases along the sweep; emit in decreasing-eps order so the
        # message timeline runs from coarse to fine perturbation.
        for i in range(len(eps)):
            t = int(i * 1e9)  # 1 s per sample; arbitrary but monotonic
            msg = {
                "eps": float(eps[i]),
                "log_eps": _log10(eps[i]),
                "pose_resid": float(pose_res[i]),
                "log_pose_resid": _log10(pose_res[i]),
                "full_resid": float(full_res[i]),
                "log_full_resid": _log10(full_res[i]),
                "pose_rel": float(pose_rel[i]),
                "full_rel": float(full_rel[i]),
                "slope1_ref": float(s1[i]),
                "slope2_ref": float(s2[i]),
                "log_slope1_ref": _log10(s1[i]),
                "log_slope2_ref": _log10(s2[i]),
                "rel_threshold": 0.05,
            }
            writer.add_message(
                cid, log_time=t, publish_time=t,
                data=json.dumps(msg).encode(),
            )
        writer.finish()
    print(f"Saved MCAP  → {out_path}  (topic /jacobian_fd, {len(eps)} msgs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default=_DEFAULT_MODEL)
    ap.add_argument("--data_dir", default=_DEFAULT_DATA)
    ap.add_argument("--out", default="jacobian_fd_convergence.png")
    ap.add_argument("--txt", default="jacobian_fd_convergence.txt",
                    help="eval.py-style whitespace matrix (np.loadtxt-ready)")
    ap.add_argument("--mcap", default="",
                    help="optional Foxglove MCAP output (e.g. out.mcap)")
    ap.add_argument("--n_dirs", type=int, default=8)
    ap.add_argument("--warm_up_steps", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    eps, pose_res, full_res, pose_ref, full_ref = compute_convergence(
        args.model_path, args.data_dir, device,
        warm_up_steps=args.warm_up_steps, n_dirs=args.n_dirs,
    )

    print("\n eps        pose_resid    pose_rel     full_resid")
    for e, pr, pf, fr in zip(eps, pose_res, pose_res / np.maximum(pose_ref, 1e-300), full_res):
        print(f" {e:.2e}   {pr:.4e}    {pf:.4e}   {fr:.4e}")

    slope, floor_i = _empirical_slope(eps, pose_res)
    pose_rel = pose_res / np.maximum(pose_ref, 1e-300)
    min_rel = pose_rel.min()
    print(f"\n empirical pose-row slope in convergence region: {slope:.2f}  "
          f"(ideal 2.0; healthy >1.2 since F is itself FD-computed)")
    print(f" min relative pose residual: {min_rel:.2e}  (wrong F stays ~O(1))")
    print(f" round-off floor reached at eps = {eps[floor_i]:.2e}")
    # Discriminator vs a wrong F: convergence must be genuine (slope clearly >1)
    # AND the relative residual must actually drop small (not plateau near 1).
    ok = slope > 1.2 and min_rel < 0.05
    verdict = ("GO — residual converges (slope %.2f, rel→%.1e): F is a valid "
               "first-order Jacobian" % (slope, min_rel)) if ok else \
        ("NO-GO — slope %.2f / min_rel %.1e: F may be wrong" % (slope, min_rel))
    print(f" verdict: {verdict}")

    make_plot(eps, pose_res, full_res, pose_ref, full_ref, args.out)
    if args.txt:
        write_txt(eps, pose_res, full_res, pose_ref, full_ref, args.txt)
    if args.mcap:
        write_mcap(eps, pose_res, full_res, pose_ref, full_ref, args.mcap)


if __name__ == "__main__":
    main()
