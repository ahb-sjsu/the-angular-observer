# Reviewer-response experiments

Reproducibility artifacts for the numbers folded into `paper/paper.tex`
(corresponds to paper **v0.7**). Each script emits a JSON blob between
`###RESULTS_JSON_START###` / `###RESULTS_JSON_END###`; the committed
`*_result.json` files are the exact outputs the paper cites.

**Section and theorem references are keyed to LaTeX labels**, not raw numbers,
because the paper's numbering has drifted across revisions (and the theorem-like
environments now share one counter). The current resolution, for convenience:

| label | number | name |
|---|---|---|
| `thm:radial` | Corollary 1 | Radial Degeneracy (§3) |
| `prop:trunc` | Proposition 2 | Truncation is spectral-gap bounded (§3) |
| `conj:angle` | Conjecture 3 | Angular Preservation (§3) |
| `lem:norm` | Lemma 4 | Normalization preserves bi-Lipschitz (§3.1) |
| `prop:angdist` | Proposition 5 | Exact angular distortion (§3.1) |
| `thm:cond` | Theorem 6 | Conditional local form of Conjecture 3 (§3.1) |
| `sec:methods` | §4 | Methods (headline CIs / uncertainty live here) |
| `sec:univ` | §5.3 | Universality + two-factor small-world screen |
| `sec:law` | §5.4 | The scaling law |
| `sec:verify` | §5.5 | Verifying Corollary 1 and Conjecture 3 |
| `sec:ablation` | §5.6 | Which weighting carries the angle? |
| `sec:bakeoff` | §5.7 | Bake-off against named spectral distances |
| `sec:density` | §5.8 | Radius is density, angle is geometry |
| `sec:stability` | §5.9 | Stability across substrate evolution |
| `sec:continuum` | §5.10 | Continuum survival, and a dissociation |
| `sec:moral` | §5.11 | Cross-domain probe: the moral embedding |

**Dependency (builders are in this repo).** The substrate builders and
estimators these scripts import---`rung0_validate.py`, `library.py`,
`crosssubstrate.py`, `arity3.py`, `rung1b_rewriter.py`---live in the **repository
root** and are versioned here. Run with the repo root on `PYTHONPATH`, e.g.
`PYTHONPATH=. python experiments/reviewer-response/todo_experiments.py`
(`clustered_bootstrap.py` and `figures.py` insert the repo root themselves). Pure
NumPy/SciPy/NetworkX, CPU-only unless noted.

## `todo_experiments.py` → `todo_experiments_result.json`

Three self-contained computations, one per author-TODO:

- **Scaling law** (`sec:law`, §5.4): re-runs the manifold library (5 Wolfram
  rules × 4 seeds = 20 manifolds) to collect the (emergent-dim `d`, angle-ρ)
  cloud, then OLS slope **−0.157**, **R² = 0.64**, Spearman −0.881. Reproduces the
  published central values to the digit (deterministic seeds); contains the
  outlier `d=1.97, ρ=0.387`. NOTE the naive point-bootstrap slope CI here
  ([−0.185, −0.132], SE 0.028) is **superseded** as the headline by the
  rule-clustered CI [−0.343, −0.094] (SE 0.061) from `clustered_bootstrap.py`
  below; the paper §5.4 and Fig. 3 report the clustered version.
- **Small-world two-factor screen** (`sec:univ`, §5.3): clustering `C`, path
  length `L`, Humphries σ and Telesford ω vs degree-matched ER + ring-lattice
  baselines, plus the three-estimator dimension spread, per substrate. Clean
  manifolds ω<0 / spread≤0.51; contaminated substrates ω≥0 / spread≥0.77.
- **Headline bootstrap CIs** (`sec:methods`, §4): 2-torus core table with pair
  counts (11,175) and 95% pair-resampling CIs (≤±0.005); m=10 angle-ρ =
  0.914±0.009 over 5 draws.

## `reviewer2_experiments.py` → `reviewer2_result.json`

Second reviewer round.

- **Truncation bound + metric distortion** (`prop:trunc` = Proposition 2;
  `sec:verify` §5.5): on the 2-torus, numerically verifies the spectral-gap
  truncation bound `0 ≤ ‖Ψ∞‖²−‖Ψᵐ‖² ≤ 1/λ_{m+1}` (radius) and `≤ 2/λ_{m+1}`
  (pairwise) — holds at every `m`, realized error 3.5–17% of the bound. Radial
  degeneracy transfer: full **normalized** radius CV **0.095** (≈ constant), and
  the **truncated** radius tracks `1/√kᵢ` at **ρ = 0.79–0.87** across m=5–40
  (this is the number the paper's Remark and §5.5 now print; the full
  normalized-radius correlation is 0.706). Metric distortion: empirical
  bi-Lipschitz κ (angle 2.64 < full 3.09 < random 8.33 ≪ magnitude 125) and
  Procrustes disparity vs the canonical R⁴ Clifford torus (angle 0.07 vs random
  0.995).
- **Nonlinear scaling** (`sec:law`, §5.4): logistic `ρ=[1+e^{0.82(d−3.66)}]⁻¹`
  respects 0<ρ<1 at R²=0.62 vs linear 0.64; exponential forms fail (A>1 or
  negative asymptote).
- **ω sensitivity** (`sec:univ`, §5.3): king-lattice rewire sweep. angle-ρ
  collapses at 1% rewiring (0.81→0.47) while ω crosses 0 only near 5% — so the
  ω<0 gate is a conservative destruction flag and angle-ρ is the finer detector.

Reloads `todo_experiments_result.json` for the 20 scaling points.

## `normalization_check.py` → `normalization_check_result.json`

Tests the Normalization Lemma (`lem:norm` = Lemma 4, §3.1) on the 2-torus.
Verifies the exact identity `‖û−v̂‖² = (‖u−v‖²−(a−b)²)/(ab)` to machine precision
(max abs err 8.9e-16) and the deterministic sufficient condition `Λ < 1/L`: the
radius Lipschitz constant **Λ=0.0918** sits just under the map's lower stretch
**A=0.0920** (the third digit matters — the margin is razor-thin), with **0 of
9,200 local pairs violating** it (median Λ 5× smaller). Local angular
bi-Lipschitz constant 2.2. Empirical support for hypothesis (H2) of the
conditional theorem (`thm:cond` = Theorem 6, §3.1). CPU-only, reuses the torus
builder.

## `ablation_experiments.py` → `ablation_result.json`

Weighting ablation (`sec:ablation`, §5.6). On the 2-torus and 3-torus, scores
angle-ρ under per-mode weightings: plain `v_k`, commute `v_k/√λ_k`, diffusion
`v_k·e^{−λt}`, and degree-corrected `÷√k_i` / `×√k_i`.
- **Commute weighting is essential**: plain eigenmaps collapse with mode count
  (torus angle-ρ 0.55→0.06 as m 10→20) while commute stays flat (0.91→0.89).
- **Degree correction is invisible to the angle**: `deg_div` = `deg_mul` =
  `commute` = 0.889 identically — a per-node radial factor cancels under
  row-normalization, confirming the degree lives in the radius.
- **Isomap₍d=2₎** on the torus = **0.62 here** vs the **0.67** printed in the
  §5.3 universality table: different run (different anchor set / graph draw at the
  same dim=2), not a contradiction. Dim-matched (dim=m) it reaches 0.97 — the
  dim-2 baseline is unfair, Isomap edges the angle only when given the geodesics.
- Random-mode control over 10 draws: 0.08±0.16 (max 0.40).

## `bakeoff.py` → `bakeoff_result.json`

Baseline bake-off (`sec:bakeoff`, §5.7): the angle vs named spectral distances on
the same graph/anchors, Spearman ρ vs unweighted geodesics. On the 2-torus
(n=2000) / 3-torus (n=3000): **angle 0.888 / 0.830** beats full commute
(0.601 / 0.360), biharmonic (0.735 / 0.515), diffusion at best t (0.771 / 0.631),
and the **amplified commute distance** (0.628 / 0.364) — von Luxburg-Radl-Hein's
own degeneracy correction, which barely improves on raw commute because
*subtracting* 1/kᵢ+1/kⱼ is not *dividing it out*. Isomap wins (0.971 / 0.952)
only as an oracle fed the geodesics. Amplified-commute + resistance use the
UNNORMALIZED Laplacian pseudoinverse (formulas verbatim from vonLuxburg-Radl-Hein
NIPS 2010 §4). Reuses `rung0_validate.torus_graph`. CPU-only, dense eigh.

## `clustered_bootstrap.py` → `clustered_bootstrap_result.json`

Rule-clustered bootstrap for the scaling-law slope (`sec:law`, §5.4; referee delta
comment 3). The 20 scaling manifolds are 5 rules × 4 seeds, so the points are
clustered, not independent; a naive point bootstrap understates the slope
uncertainty. Recomputes the OLS slope (−0.157) and its 95% CI two ways: **point**
(resample 20 points i.i.d.) → [−0.185, −0.132] (the old figure band), **cluster**
(two-stage: resample the 5 rules, then the 4 seeds within each drawn rule) → the
honest **[−0.343, −0.094]** (SE 0.061), still excluding zero, four times wider.
Fig. 3 draws the clustered band; §5.4 reports it. Reuses library.LIB / measure.
Deterministic (fixed integer seeds).

## `density_torus.py` → `density_torus_result.json`

Non-uniform-density torus (`sec:density`, §5.8, referee's "most valuable new
experiment"). Samples the flat 2-torus with density p(x)∝1+a·cos2πx, fixed-radius
RGG, sweeps contrast a (degree ratio up to 45× at a=0.9), scoring against the TRUE
flat-torus geodesics. Confirms **radius = density** (ρ(radius, 1/√p) = 0.53–0.67)
and shows the raw angle is **not** sampling-invariant (angle-ρ_true 0.93→0.79 as
contrast grows, since the graph Laplacian → density-weighted operator, not
Laplace-Beltrami). The **Coifman-Lafon α=1 normalization** W↦D⁻¹WD⁻¹ restores it
to 0.92–0.96, flat across the whole range — the same normalization that fixes the
angle isolates the density into the radius. Reuses `_norm_laplacian_eigs`.
CPU-only, dense eigh.

## Atlas-run artifacts (not locally reproducible without the cluster)

Two experiments were run on the Atlas host and their result JSONs are committed
here, but the input data is **not** yet vendored (see the reproducibility
punch-list below):

- **`mobse_cross.py` → `mobse_cross_result.json`** — cross-domain probe
  (`sec:moral`, §5.11). Angular diagnostic on a moral/ethical embedding space:
  2,400 BGE-M3 (1024-d) items, six traditions, kNN graph. **angle-ρ = 0.893**
  (shuffle 0.233, isotropic 0.311); dim-spread 2.6, ρ_mag 0.56, **ω +0.15** —
  which places it *between* the clean manifolds and the destroyed substrates on
  the §5.3 screen (now cited in §5.11). The script pulls precomputed embeddings
  from an internal pgvector table via `psycopg2`, so it is not externally
  runnable as-is; the ~10 MB embedding export is the intended reproducible path.
- **`mobse_decompose.py` → `mobse_decompose_result.json`** — within/between
  decomposition + hubness (`sec:moral`, §5.11). All-pairs angle-ρ 0.90 is partly
  cluster separation: between-tradition 0.85 vs **within-tradition 0.67**;
  magnitude tracks kNN in-degree at 0.58 (hubness).
- **`rung4_scaledm.py` → `rung4_scaledm_result.json`** — scaled-budget continuum
  refinement (`sec:continuum`, §5.10). With m spectral-gap-matched (τ fixed, m
  grows to ~47 at N=10⁴), the fixed-budget dissociation **inverts**: rgg3 fixed
  +1.0 / scaled −1.0; wolf2 fixed −1.0 / scaled −0.5. Budget-relative, not an
  absolute substrate fact. Depends on `rung4_pod` builders (**not yet in the
  repo**). Local, N≤10⁴, one seed (direction, not a law).

## Reproducibility punch-list (still open)

- Vendor the moral embeddings (~10 MB float32 export) and make the pgvector pull
  an optional refresh path, removing the internal-infra dependency from §5.11.
- Add `rung4_pod.py` and the fixed-budget §5.10 continuum runs.
- Add the `PREREG_RUNG3/4` documents (with dated amendments) and the §5.9
  stability harness (151 rules, Amendment-2 capture hook).
