# Reviewer-response experiments

Reproducibility artifacts for the numbers folded into `paper/paper.tex` during
the v0.6 reviewer response (2026-07-14). Each script emits a JSON blob between
`###RESULTS_JSON_START###` / `###RESULTS_JSON_END###`; the committed
`*_result.json` files are the exact outputs the paper cites.

## `todo_experiments.py` → `todo_experiments_result.json`

Three self-contained computations, one per author-TODO:

- **Scaling law** (paper §5.4): re-runs the manifold library (5 Wolfram rules ×
  4 seeds = 20 manifolds) to collect the (emergent-dim `d`, angle-ρ) cloud, then
  OLS slope **−0.157 ± 0.028**, **R² = 0.64**, bootstrap slope 95% CI
  [−0.185, −0.132], Spearman −0.881 [−0.97, −0.67]. Reproduces the published
  central values to the digit (deterministic seeds). Contains the reviewer's
  outlier: `d=1.97, ρ=0.387`.
- **Small-world two-factor screen** (paper §5.3): clustering `C`, path length
  `L`, Humphries σ and Telesford ω vs degree-matched ER + ring-lattice baselines,
  plus the three-estimator dimension spread, for each substrate. Clean manifolds
  ω<0 / spread≤0.51; contaminated substrates ω≥0 / spread≥0.77.
- **Headline bootstrap CIs** (paper §5.2 Methods): 2-torus core table with pair
  counts (11,175) and 95% pair-resampling CIs (≤±0.005); m=10 angle-ρ =
  0.914±0.009 over 5 draws.

**Dependency:** imports the frozen substrate builders and estimators from the
`wolfram-observer-bridge` working tree (`rung0_validate.py`, `library.py`,
`crosssubstrate.py`). That tree is not a git repo; run this script from within
it. Pure NumPy/SciPy/NetworkX, CPU-only.

## `mobse_cross.py` → `mobse_cross_result.json`

MoBSE cross-program experiment (paper §5.8): the angular-observer diagnostic
applied to a **moral** embedding space — 2,400 BGE-M3 (1024-d) ethics-text items
from the Atlas `ethics_chunks` table, stratified across six traditions, kNN
graph. Self-contained (no local imports); pulls precomputed pgvector embeddings
via `psycopg2`, CPU-only. Result: **angle-ρ = 0.893** (shuffle control 0.233,
isotropic 0.311) with dim-spread 2.6, ρ_mag 0.56, ω +0.15 — substantially
manifold-like but not a pure low-d Riemannian substrate. Run on Atlas under
`/home/claude/env`.

## `reviewer2_experiments.py` → `reviewer2_result.json`

Second reviewer round.

- **Truncation bound + metric distortion** (paper §3 Proposition 1, §5.5): on the
  2-torus, numerically verifies the spectral-gap truncation bound
  `0 ≤ ‖Ψ∞‖²−‖Ψᵐ‖² ≤ 1/λ_{m+1}` (radius) and `≤ 2/λ_{m+1}` (pairwise) — holds at
  every `m` with realized error 3.5–17% of the bound. Confirms the radial
  degeneracy transfer (full normalized radius CV 0.09 ≈ constant; truncated
  radius vs 1/√kᵢ ρ 0.79–0.87). Reports strict metric distortion: empirical
  bi-Lipschitz κ (angle 2.64 < full 3.09 < random 8.33 ≪ magnitude 125) and
  Procrustes disparity vs canonical R⁴ torus (angle 0.07 vs random 0.995).
- **Nonlinear scaling** (paper §5.4): logistic `ρ=[1+e^{0.82(d−3.66)}]⁻¹`
  respects 0<ρ<1 at R²=0.62 vs linear 0.64; exponential forms fail (A>1 or
  negative asymptote).
- **ω sensitivity** (paper §5.3): king-lattice rewire sweep. angle-ρ collapses at
  1% rewiring (0.81→0.47) while ω crosses 0 only near 5% — so the ω<0 gate is a
  conservative destruction flag and angle-ρ is the finer detector.

Dependency: same wolfram-observer-bridge builders as `todo_experiments.py`, plus
it reloads `todo_experiments_result.json` for the 20 scaling points. CPU-only.

## `normalization_check.py` → `normalization_check_result.json`

Tests the Normalization Lemma (paper §3.3, Lemma 1) on the 2-torus. Verifies the
exact identity `‖û−v̂‖² = (‖u−v‖²−(a−b)²)/(ab)` to machine precision
(max abs err 8.9e-16) and the deterministic sufficient condition `Λ < 1/L`: the
radius Lipschitz constant Λ=0.092 sits just under the map's lower stretch A=0.092,
with **0 of 9,200 local pairs violating** it (median Λ 5× smaller). Local angular
bi-Lipschitz constant 2.2. This is the empirical support for hypothesis (H2) of
the conditional Theorem 1. CPU-only, reuses the torus builder.

## `ablation_experiments.py` → `ablation_result.json`

Weighting ablation (referee comment 1), paper §5.5.1. On the 2-torus and 3-torus,
scores angle-ρ under per-mode weightings: plain `v_k`, commute `v_k/√λ_k`,
diffusion `v_k·e^{−λt}`, and degree-corrected `÷√k_i` / `×√k_i`.
- **Commute weighting is essential**: plain eigenmaps collapse with mode count
  (torus angle-ρ 0.55→0.06 as m 10→20) while commute stays flat (0.91→0.89).
  "Flat in m" is a property of the weighting, and it selects commute/diffusion.
- **Degree correction is invisible to the angle**: `deg_div` = `deg_mul` =
  `commute` = 0.889 identically, since a per-node radial factor cancels under
  row-normalization — direct confirmation the degree lives in the radius.
- **euclid/Isomap** at dim=2 = 0.62 but dim-matched (dim=m) = 0.97: the dim-2
  baseline was unfair; Isomap edges the angle when given the geodesics as input.
- Random-mode control averaged over 10 draws: 0.08±0.16 (max 0.40).
CPU-only, reuses torus builder.

## `mobse_decompose.py` → `mobse_decompose_result.json`

§5.8 within/between-tradition decomposition + hubness (referee comment 7, Atlas).
Confirms the confound: all-pairs angle-ρ = 0.90, but between-tradition = 0.85 and
**within-tradition = 0.67** — the headline was partly cluster separation (which
row-normalization enhances). Magnitude tracks kNN in-degree at 0.58 (hubness), so
the "informative radius" is largely degree, per Corollary 1.

## `rung4_scaledm.py` → `rung4_scaledm_result.json`

Scaled-m continuum refinement (review-2's decisive test): is §5.7's dissociation a
fixed-observer-budget artifact? With m spectral-gap-matched (τ fixed, m grows with
N to ~47 at N=10⁴), the dissociation **inverts**: rgg3 fixed +1.0/scaled −1.0;
wolf2 fixed −1.0/scaled −0.5. So the dissociation is budget-relative — a property
of the fixed-budget observer, not an absolute substrate fact. Manifolds keep
geometry in a fixed low-mode band; emergent graphs spread it across scales.
Reuses rung4_pod builders. Local, N≤10⁴, one seed (direction, not a law).

## `bakeoff.py` → `bakeoff_result.json`

Baseline bake-off (paper §5.5.2): the angle vs named spectral distances on the
same graph/anchors, scored by Spearman ρ against unweighted geodesics. On the
2-torus (n=2000) / 3-torus (n=3000): **angle 0.888 / 0.830** beats full commute
(0.601 / 0.360), biharmonic (0.735 / 0.515), diffusion at best t (0.771 / 0.631),
and the **amplified commute distance** (0.628 / 0.364) — von Luxburg-Radl-Hein's
own degeneracy correction, which barely improves on raw commute because
*subtracting* 1/kᵢ+1/kⱼ is not *dividing it out*. Isomap wins (0.971 / 0.952)
only because it is fed the geodesics as input — an oracle ceiling, not a
competitor. Amplified-commute + resistance use the UNNORMALIZED Laplacian
pseudoinverse (formulas verbatim from vonLuxburg-Radl-Hein NIPS 2010 §4). Reuses
`rung0_validate.torus_graph`. CPU-only, dense eigh.

## `density_torus.py` → `density_torus_result.json`

Non-uniform-density torus (paper §5.6, referee's "most valuable new experiment").
Samples the flat 2-torus with density p(x)∝1+a·cos2πx, fixed-radius RGG, sweeps
contrast a (degree ratio up to 45× at a=0.9), scoring against the TRUE flat-torus
geodesics. Confirms **radius = density** (ρ(radius, 1/√p) = 0.53–0.67) and shows
the raw angle is **not** sampling-invariant (angle-ρ_true 0.93→0.79 as contrast
grows, since the graph Laplacian → density-weighted operator, not
Laplace-Beltrami). The **Coifman-Lafon α=1 normalization** W↦D⁻¹WD⁻¹ restores it
to 0.92–0.96, flat across the whole range — the same normalization that fixes the
angle isolates the density into the radius. Reuses `_norm_laplacian_eigs`.
CPU-only, dense eigh.
