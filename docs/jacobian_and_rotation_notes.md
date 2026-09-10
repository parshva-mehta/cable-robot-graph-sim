# Jacobian, rotation representations, and rollout horizon — audit notes

Companion to `ekf_ros_integration_design.md`. Covers three linked questions:

1. How the Jacobian is actually computed, and which form is trustworthy to send.
2. The rotational discontinuity inherent in <5-variable rotation representations,
   and one concrete latent bug it causes here.
3. What "increase the inference rollout horizon (predict_n)" means in this code
   and how it interacts with the factor-graph (EKF).

Nothing in this file could be run end-to-end in the audit environment (no trained
`.pt`, no dataset, and torch/gtsam are not installed), so items marked **[unrun]**
are code-level analysis to be confirmed with a model via
`diagnostics/diag_jacobians.py` and `diagnostics/diag_exp_vs_quat.py`.

---

## 1. How the Jacobian is calculated

`linearization.linearize_dynamics` (`linearization.py:232`):

- Wraps `simulator.step` and takes `∂(next_state)/∂(state)` with
  `torch.func.jacrev` (or central finite differences). Crucially it differentiates
  **one** predicted step — `ns[0, :state_dim, 0]` (`linearization.py:293`,
  `:283`) — even though the model emits `n_fwd_pred_steps` steps per call
  (see §3). So `F` is the single-step map `∂x_{t+1}/∂x_t`.
- Result is the ambient **39×39** Jacobian over `[pos, quat, linvel, angvel] × 3`
  rods.

Then `_fix_jacobian_quaternion_rank` (`linearization.py:116`) "repairs" it:

1. Project to the 36-D tangent: `F_red = T_out @ F @ T_in`, where the quaternion
   rows/cols are mapped through the orthonormal 4×3 basis `E` (`E.T E = I`,
   `E.T q = 0`).
2. **Clamp the spectral radius** of `F_red` to ≤ 1.
3. Lift back: `F_fixed = T_in @ F_red @ T_out`.
4. Add `ε·qqᵀ` (ε=1.0) to each rod's 4×4 quaternion diagonal block to make the
   39×39 matrix full-rank again.

### Why the ambient 39×39 `F_fixed` is **not** the object to send

- **It is rank-deficient by construction.** The unit-norm constraint removes one
  DOF per rod, so the true ambient Jacobian has rank ≤ 36 for 3 rods
  (`diag_exp_vs_quat.py` reports this as `rank_raw`). The full-rank appearance of
  `F_fixed` comes entirely from the `ε·qqᵀ` term, which injects **fictitious
  variance along the quaternion radial (norm) direction** — an unphysical
  direction. It exists only to keep GTSAM's matrix inverses well-posed.
- **The spectral-radius clamp biases it.** Clamping to ≤1 is an EKF-stability
  heuristic; after it, `F_fixed ≠ ∂f/∂x`. Sending it as "the Jacobian" would ship
  a stabilized surrogate, not the model's true sensitivity.

**Recommendation.** For ROS, send *uncertainty*, not the raw ambient Jacobian —
which is what Task C now does: the EKF covariance projected to the 6×6 small-angle
tangent (`rod_covariance_to_ros`). If a downstream consumer genuinely needs a
Jacobian matrix, send one of the **continuous, full-rank 36×36** forms over the
`Float64MultiArray` channel (`MatrixStreamPublisher`):

- the tangent Jacobian `T_out @ F @ T_in` (before the `ε·qqᵀ` hack, ideally before
  the spectral clamp), or
- the exp-map Jacobian from `linearization_exp.py`, which is naturally 36×36 and
  full-rank (no projection, no `ε·qqᵀ`).

Do **not** send the 39×39 `F_fixed`.

---

## 2. Rotational discontinuity in <5-variable representations

Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*
(CVPR 2019): there is **no continuous, injective map from SO(3) into ℝ^d for
d < 5**. Every representation used in this repo is below that bound, so each has a
discontinuity somewhere:

| Representation | dim | Discontinuity | Where it bites here |
|---|---|---|---|
| Quaternion | 4 | Double cover: `q` and `−q` are the same rotation | EKF measurement update (see bug below); `rank_raw < 39` |
| Exp-map / axis-angle | 3 | Singularity at ‖θ‖ = π (and removable one at 0) | `quat2exp/exp2quat` blow up near θ=π; tracked as `θ_max/π` in `diag_exp_vs_quat.py` |

For a **local EKF linearization** the discontinuity is tolerable as long as
trajectories stay away from the singular set (quaternion hemisphere boundary /
θ≈π). The 36×36 tangent or exp-map Jacobian sidesteps the quaternion *rank*
problem, but note it is still a <5-D parameterization and therefore still globally
discontinuous — fine locally, not fine for a network that *regresses* absolute
rotations (that is the case Zhou et al. argue needs 6-D). We do not regress
absolute rotations here, so 6-D is not warranted; the recommendation is scoped to
the filter/linearization.

### Concrete latent bug: double-cover in the EKF update **[unrun]**

`run_ekf_rollout` builds the measurement `z` directly from ground-truth
quaternions with `H = I` (`ekf.py:587-593`, `:563`) and forms the innovation as
`z − H·x̂` (`ekf.py:323`). There is **no hemisphere alignment** anywhere in
`ekf.py` / `ekf_alt.py` / `linearization.py` (verified by grep). So when a
measurement quaternion and the predicted quaternion lie in **opposite
hemispheres** (`q_meas ≈ −q_pred`) — the *same physical orientation* — the
per-rod quaternion innovation is ≈ `2·q`, a large spurious correction, even
though the orientation error is zero.

- With the default `innovation_gate_sigma = np.inf` (no gating), that spurious
  innovation is applied in full.
- Fix: before the update, flip the sign of each measurement quaternion block to
  match the predicted state, e.g. `if dot(q_meas, q_pred) < 0: q_meas = −q_meas`.
  This is the standard double-cover guard and belongs in `_ekf_step_gtsam` just
  before the innovation is formed (and in any pose-only path).

This is intentionally **not** bundled into the Task C change — it alters filter
behavior and deserves its own change plus a regression test against an antipodal
measurement.

---

## 3. Inference rollout horizon ("predict_n") and factor-graph compatibility

### Two different "horizons" — don't conflate them

- **Per-call prediction depth** = `n_fwd_pred_steps` (aka `num_out_steps`),
  currently **4** in `simulators/configs/3_bar_gnn_sim_config.json:17`. The GNN
  decoder outputs `n_out * n_fwd_pred_steps` accelerations
  (`tensegrity_gnn_simulator.py:107`), so this is **architectural** — you cannot
  raise it at inference beyond what the decoder was trained to emit; it requires
  retraining.
- **Total rollout length** = the `num_steps` argument to `run()`
  (`tensegrity_gnn_simulator.py:214`). `run()` already chains
  `⌈num_steps / num_out_steps⌉` multi-step blocks autoregressively, so this
  horizon is **already free** — raising it needs no code change; only compounding
  error grows with horizon (expected for open-loop rollout).
- Distinct again from **`num_steps_fwd`**, the training-loss horizon
  (`train_sim_data.py:26` ramps 4→4→8→8→16). Architecture (`n_fwd_pred_steps`) ≠
  loss horizon (`num_steps_fwd`).

So "increase the inference rollout horizon" is a no-op for open-loop eval/MPPI
(just pass more control steps). The real question is the factor graph.

### Factor-graph (EKF) compatibility — the actual incompatibility

The EKF/factor-graph currently does **one predict per measurement**, and its
linearization uses only step 0 of the model's 4-step output (§1). Making the
filter advance `n > 1` steps between measurements would require all of:

1. **The n-step Jacobian** `F_n = ∂x_{t+n}/∂x_t`, either by chaining single-step
   Jacobians `F_n = F_{t+n-1} ··· F_t` (needs the intermediate states) or by
   `jacrev` through the n-step map. `linearize_dynamics` does neither today.
2. **Process-noise accumulation** `Q_n ≈ Σ_k F^k Q F^{kᵀ}`; a single-step `Q`
   over an n-step jump makes the filter overconfident.
3. **Recurrent-state handling.** With an LSTM/GRU processor the next state depends
   on the carried `node_hidden_state` (`tensegrity_gnn_simulator.py:210`), which
   is **not** part of the 39-D EKF state. `∂x_{t+n}/∂x_t` alone then omits the
   hidden-state pathway, and the Markov assumption a factor graph relies on is
   violated. A correct multi-step factor would have to fold the hidden state into
   the estimated state (or use single-step factors that re-derive it).

**Recommendation.** Keep the two concerns orthogonal:

- Open-loop rollout (eval/MPPI): raise the horizon freely via `run()`.
- Filter/factor-graph: keep **single-step factors between consecutive states**.
  That keeps `F` valid, `Q` simple, and the recurrent state consistent. Pursue a
  true multi-step predict only if there is a measurement-rate vs. sim-rate
  mismatch that forces it — and then implement items 1–3 above deliberately.
  Multi-step *prediction accuracy* of the learned model is best validated with
  `diag_exp_vs_quat.py` (`pred_diff`, `θ_max/π`) on a real checkpoint. **[unrun]**

### Aside: a latent crash in the MPPI compile path

`tensegrity_mppi_planner.py:91` calls `self.sim.run_compile()`, but the simulator
defines the method as `torch_compile()` (`tensegrity_gnn_simulator.py:81`) — there
is no `run_compile`. Any `TensegrityMPPIPlanner(..., torch_compile=True)` raises
`AttributeError` immediately. Out of scope for these tasks, but worth fixing
(rename the call to `torch_compile()`).
