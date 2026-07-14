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
