# Pre-registration — Rung 4 (large-N continuum survival)

Registered 2026-07-14, BEFORE any rung-4 experiment was run. Bars below are
frozen; a FAIL is reported as a FAIL, and no salvage analysis runs without a
new dated amendment (same discipline as `PREREG_RUNG3.md`).

## Why this rung

The rung-3 ladder is complete: the angular basis preserves geodesics on five
substrate types (rung 2), is dynamically stable across a rule's own evolution
(rung 3a / 3a-XL, 100% universality, cross-arity), carries no dynamical
curvature (rung 3b null, doubly confirmed), and degrades gracefully off-manifold
(rung 3c). Every one of those results is at **finite N** (≈2k–4k nodes, capped
by dense `eigh`).

The theorem's honest ledger (`theorem.md` §4) marks the load-bearing claim as a
**conjecture**: *"angular low-mode distance ≍ geodesic, uniform in m and N — no
uniform bi-Lipschitz bound proven; empirically ρ≈0.92, flat."* "Uniform in N"
has never been tested past the dense-eigensolver ceiling. Rung 4 asks the one
big empirical question left:

> **Does the angle-only low-mode coarse-graining's geodesic preservation SURVIVE
> as N grows toward the continuum — flat in N — or is it a finite-size effect
> that decays as the graph refines?**

This is empirical support for (or against) the conjecture, NOT the analytic
proof; the proof that "the angle lives" remains a parallel open track and is out
of scope here.

## Substrates (registered)

Three substrates with a well-defined continuum limit, so "refine N at fixed
geometry" is meaningful:

1. **RGG-2 — random geometric graph on the flat 2-torus** (d=2). Points uniform
   on [0,1)², periodic; connect within radius r(N) chosen so mean degree is held
   ≈ constant (k̄ ≈ 12) as N grows — this is the honest "refine the same manifold"
   knob. Canonical, geometry exactly known.
2. **RGG-3 — random geometric graph on the flat 3-torus** (d=3). Same
   construction in [0,1)³, k̄ ≈ 16. This is where the radial degeneracy is
   cleanest (von Luxburg d≥2; Theorem holds sharply at d=3) — the strongest test.
3. **WOLF-2 — the R_3D Wolfram-model rule** (emergent ≈2D, the rung-3a manifold
   rule `{{1,1,2},{3,4,1}}→{{4,4,3},{5,4,5},{5,2,1}}`). Emergent geometry, no
   latent coordinates — tests the substrate-agnostic claim at scale, not just
   coordinate-backed RGGs.

## N ladder (registered)

N ∈ {2 000, 5 000, 10 000, 20 000, 50 000}, plus **100 000 if the eigensolver
and controls remain valid at 50k**. Measurement is on the largest connected
component; if an N cannot reach the target component size it is reported
unresolved at that scale (not dropped silently). Seeds: 4 per (substrate, N).

## Method (registered, matches prior rungs)

Per (substrate, N, seed):
1. Build graph, take largest connected component, adjacency A.
2. Symmetric-normalized Laplacian L = I − D^{-1/2} A D^{-1/2}. Take the
   **m = 10 lowest non-trivial modes** via **sparse `scipy.sparse.linalg.eigsh`
   (shift-invert, sigma=0, which='LM')** — or LOBPCG with a Jacobi preconditioner
   if shift-invert factorization exceeds memory. (Dense `eigh` is not used; that
   ceiling is the whole reason for this rung.) Scale each mode by 1/√λ.
3. **Angular embedding** U = row-normalize that m-column matrix to the unit
   sphere (Ng–Jordan–Weiss / PolarQuant). This is the object under test.
4. Draw **n_anchor = 300** nodes (raised from 150 for large-N stability; frozen).
   Geodesics G = unweighted `shortest_path` between anchors. Angular distances
   d = Euclidean between anchor rows of U.
5. **angle_ρ** = Spearman(d, G) over the C(300,2) anchor pairs.
6. **Controls, same anchors, every N:**
   - **random_ρ** = Spearman using m RANDOM Laplacian modes (must stay ≈0).
   - **magnitude_ρ** = Spearman using the radius ‖scaled-mode row‖ only (must
     stay low — this is the von-Luxburg-degenerate coordinate).

Report per cell: N_lcc, angle_ρ, random_ρ, magnitude_ρ, mean degree, and the
eigensolver residual ‖Lx − λx‖ (a solver-health check).

## Pre-registered bars (frozen)

**Validity gate (must hold at every N, else that cell is instrument-invalid,
NOT a result):**
- random_ρ ≤ 0.15 AND magnitude_ρ ≤ 0.30, AND max eigensolver residual ≤ 1e-6.
  If a cell breaks this, it is an eigensolver artifact at scale → re-tool, not
  salvage; the cell is excluded and the failure reported.

**PRIMARY — continuum survival (the falsifiable claim):** on a clean substrate
the angular basis SURVIVES refinement iff
- angle_ρ(N_max) ≥ 0.90 × angle_ρ(N = 2 000)   *(no collapse)*, **and**
- angle_ρ(N_max) ≥ 0.75                          *(absolute floor)*.

**Universality:** the observer principle survives to the continuum iff PRIMARY
holds on **≥ 2 of the 3 substrates**. RGG-3 (d=3, cleanest) is expected to be the
strongest; WOLF-2 the hardest.

**FLATNESS (distribution-free, secondary):** Spearman(angle_ρ, log N) per
substrate. A value ≥ −0.3 supports "flat/uniform in N"; a clearly negative
trend (≤ −0.6) with angle_ρ crossing below the floor is **decay**.

## Interpretation, fixed in advance

- **PASS (survival on ≥2 substrates):** the "uniform in N" clause of the
  conjecture gains continuum-scale empirical support; the observer principle is
  not a finite-size artifact. Joins the paper as rung 4. Still not a proof —
  the analytic bi-Lipschitz bound remains open.
- **DECAY (angle_ρ falls below floor as N grows):** the observer principle is a
  finite-N effect. This is a real boundary, reported as an honest negative like
  the rung-3b curvature null — it bounds the claim, it is not massaged away.
- **INSTRUMENT FAILURE (validity gate breaks at scale before any geometric
  conclusion):** report as a sparse-eigensolver instrument-failure, re-tool the
  solver under a new amendment; no geometric claim either way.

## Amendment 1 (2026-07-14, registered BEFORE any rung-4 run)

Building the harness exposed that the original control — "random_ρ = m modes
drawn from the WHOLE spectrum" — needs the full dense eigendecomposition, which
is exactly what does not exist at N ≥ 50 000 (the reason this rung uses sparse
`eigsh`). Registered fix, before any run:

- Compute the **lowest K = 64** non-trivial modes in one sparse solve.
- **angle_ρ** uses the lowest m = 10 (unchanged).
- **higher_ρ** (replaces random_ρ) uses the **top m of that lowest-64 band**
  (modes 54–63): non-lowest modes that, if the low subspace is special, must
  carry no geodesic signal. Same bar: **higher_ρ ≤ 0.15** at every N.
- **magnitude_ρ** (radius-only) unchanged, bar ≤ 0.30.

Rationale: the claim under test is that the *lowest* modes are geometrically
special; a non-lowest computed band is a faithful, sparse-computable control for
that, and avoids fabricating a whole-spectrum draw we cannot afford. All primary
survival bars are unchanged.

## Compute (registered)

- N ≤ 20 000: single Atlas run or an NRP CPU Indexed Job (one pod per
  (substrate, N, seed)); sparse `eigsh` on CPU is minutes-scale here.
- N ≥ 50 000: **Atlas GV100** — GPU sparse eigensolve (cupy / LOBPCG) where CPU
  shift-invert memory runs out; the graphs exceed the ~5k dense-`eigh` ceiling
  by design. Thermal + `reference_nrp_cluster_policy` rules apply; check before
  any pod submission.
- Deterministic seeding per (substrate, N, seed); results emitted between
  `###RESULTS_JSON_START###`/`###RESULTS_JSON_END###` markers and log-aggregated,
  same as rung-2 / rung-3a-XL.
