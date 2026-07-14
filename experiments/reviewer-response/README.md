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
