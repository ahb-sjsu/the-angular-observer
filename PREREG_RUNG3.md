# Pre-registration — Rung 3 (a/b/c)

Registered 2026-07-13, BEFORE any rung-3 experiment was run. Bars below are
frozen; results will be reported against them verbatim (pass or fail).

Context: the published core (the-angular-observer) is that the angle-only
low-mode Laplacian embedding is the observer's coarse-graining basis —
angle_rho ~0.9 on emergent manifolds, dimension-degradation law, von Luxburg
mechanism, two honest negatives (no hyperbolic rescue; no dynamical/Einstein
curvature: raw −0.98 vs update density deflates to partial ρ = −0.14 after
controlling degree).

## Rung 3a — time-stability of the angular basis under the graph's own dynamics

**Claim under test:** the observer's angular basis is stable under hypergraph
evolution: for nodes present at both times, the *relative angular geometry*
at step T predicts the relative angular geometry at step T+Δ, while the
graph exponentially grows.

**Protocol.** Rule R_3D (Wolfram Apr-2020, borrowed; our one reliable clean-2D
manifold: angle_rho 0.822 ± 0.008, spread 0.05), collapsed seed. Snapshots at
hyperedge counts ~[500, 1000, 2000, 4000]. Anchors = up to 150 nodes present
in the largest component of EVERY snapshot (fixed by node ID; the rewriter
never renames surviving nodes). Per snapshot: symmetric-normalized Laplacian,
m=20 low modes, embedding rows split into angle (row-normalized) and
magnitude (row norm); random-mode control = 20 modes drawn uniformly from the
nontrivial spectrum. All distance comparisons are Spearman on anchor-pair
distances (rank geometry), which is invariant to eigenbasis rotation, so no
cross-time alignment step is needed or used. Rewriter seeds 1–4; headline
numbers reported mean ± std.

**Measures.**
- fidelity(s, basis) = |Spearman(d_basis, hop-geodesic)| within snapshot s
  (common anchors; full-anchor angle_rho also reported for continuity).
- stability(s→s′, basis) = Spearman(d_basis at s, d_basis at s′) on the same
  anchor pairs.
- churn baseline = Spearman(geodesics at s, geodesics at s′): the substrate's
  own geometric persistence; nothing can honestly beat it.

**Pre-registered bars.**
- H1 (fidelity persists through growth): angle fidelity ≥ 0.75 at every
  snapshot with N ≥ 500; magnitude and random-mode fidelity ≤ 0.30.
- H2 (consecutive stability): angle stability ≥ 0.70 for every consecutive
  snapshot pair AND ≥ 0.9 × churn baseline for that pair.
- H3 (long-lever stability, first→last): angle stability ≥ 0.60 AND
  ≥ 0.85 × churn.
- Validity gate: if churn < 0.5 anywhere, the substrate itself is
  geometrically unstable in this regime — report as such, no observer claim
  either way.
- Stated in advance: magnitude stability may well be HIGH (degree persists on
  surviving nodes) while magnitude fidelity stays ~0 (rung 1c: 0.03). High
  magnitude stability is therefore NOT evidence of geometry; the
  geometry-carrying-AND-stable cell must be angle alone. The anchor set is
  biased toward old/core nodes by construction (they must exist early);
  new-region geometry is probed by per-snapshot fidelity only. Disclosed.

**Interpretation if PASS:** "the angular basis is dynamically stable under
the universe's own update rule" joins the paper as rung 3a. If FAIL: report
the failing bar; no salvage analysis without a new pre-registration.

### Rung 3a-XL — is dynamical stability universal across the manifold-rule library? (NRP protocol, registered before launch)

Rung 3a establishes stability on ONE rule (R_3D). The XL question is whether
stability is a LAW across independently discovered manifold rules, with a
dimension dependence mirroring the fidelity law (library.py: fidelity slope
−0.157/dim). Protocol: the validated rung-2 CPU-pod recipe (Indexed Job,
cpu=1/mem=2Gi, python:3.12-slim + numpy/scipy, code via ConfigMap, results as
stdout JSON between markers; kubectl driven from Atlas). Each shard draws
rules deterministically from the rung-2 rule space (`sweep.worker`,
SEED_BASE+i), and for every QUALIFYING rule (angle_rho ≥ 0.80 AND spread ≤
0.30 — the library's manifold-grade definition) runs the 3a protocol at
snapshots [300, 600, 1200, 2400] edges, rewriter seeds {1, 2}, m=20, ≤120
common anchors. Magnitude/random-mode controls are NOT re-run at scale (they
were validated per-substrate locally in 3a); each rule reports angle
fidelity per snapshot, consecutive + long-lever angle stability, and its own
churn baseline.

Pre-registered analysis and bars:
- **Universality bar:** stability is "universal across the library" iff
  ≥ 80% of qualifying rules (with churn ≥ 0.5, the same validity gate) have
  mean consecutive angle stability ≥ 0.9 × their own churn.
- **Law:** regression of long-lever angle stability on emergent dimension
  (ball+spec mean), slope and Spearman reported alongside the fidelity law;
  no bar, this is measurement.
- Rules failing the churn gate are reported as substrate-unstable, excluded
  from the universality denominator, and counted (if > 30% of qualifying
  rules fail the gate, that fact is itself the headline and the universality
  claim is not made).

## Rung 3b — angle-basis Ollivier–Ricci curvature (does the honest negative survive a basis change?)

**Two claims, falsification order, run static first.**

- 3b-static (instrument validation): OR curvature with the ANGULAR distance
  as ground metric (transport cost between neighbor distributions measured in
  angular distance, m=20 low modes) must reproduce the sign pattern of the
  validated hop-metric implementation: ~0 on line and 2D grid, negative on a
  binary tree, positive on the torus. Bar: same sign classification on all
  four controls (|mean κ| < 0.05 counts as zero). HARD STOP: if static fails,
  3b-dynamical is not run and 3b reports an instrument failure.
- 3b-dynamical (the re-test of the paper's §6 null): on R_3D snapshots,
  per-edge angle-basis κ vs update activity; report raw Spearman AND partial
  Spearman controlling degree, seeds 1–4. Pre-registered EXPECTATION: the
  null survives (the angle basis deletes degree by construction; the prior
  partial ρ was −0.14). Bar to claim ANY dynamical curvature signal:
  |partial ρ| ≥ 0.30 with p < 1e-3, same sign, in ≥ 3 of 4 seeds. Anything
  less: "the honest negative survives the basis change" goes in the paper.

### Amendment 1 (2026-07-13, after 3b-static iteration 1, BEFORE any 3b-dynamical run)

3b-static iteration 1 (one-step measures, m=20 angular ground metric) FAILED
the hard stop: grid mean κ = −0.061 and RGG-torus −0.064 (both misclassified
negative; line and tree correct). Diagnosis: a unit-hop transport problem
sits below the resolution of a 20-mode angular metric (mode wavelengths span
the whole graph; adjacent nodes are angularly near-coincident), so flat
controls pick up a small uniform negative interpolation bias. The dynamical
test was NOT run (hard stop honored).

Iteration 2, registered here before running: **coarse Ollivier curvature at
the observer's scale** — measures uniform on hop-radius-b balls, node pairs
at hop distance h, frozen at (b, h) = (3, 6); the identical coarse
instrument is also run in the hop metric as a comparability column. Revised
static bars: line ~0, grid ~0, tree negative, **RGG-sphere positive** (the
honest constant-positive-curvature control at scale). RGG-torus is kept as a
reported CONTEXT row with expectation → 0 at coarse scale: its one-step
positivity is the ball-overlap discreteness artifact (so labeled in
curvature.py from the start), and a scale-aware instrument is supposed to
see past it. |mean κ| < 0.05 still counts as zero. Hard stop unchanged.

3b-dynamical adaptation (κ is now a two-ball property): activity and degree
proxies become REGIONAL means over B(x,b) ∪ B(y,b). Bar unchanged:
|partial ρ| ≥ 0.30, p < 1e-3, same sign, ≥ 3 of 4 seeds to claim a signal;
expectation unchanged (null survives). No further static iterations without
another amendment; a second static failure ends 3b as an instrument-failure
report.

## Rung 3c — multiway/branchial substrate transfer (exploratory, classification pre-registered)

**Question:** does angle-only low-mode coarse-graining preserve geodesic rank
on (a) a multiway states graph and (b) a branchial slice, as it does on
Riemannian substrates — or do these join the Lorentzian causal set as a
correct failure?

**Protocol.** String rewrite multiway systems (e.g. {"A"→"AB", "BB"→"A"}
class), states-graph to depth giving N ≥ 300; branchial graph = same-step
states linked iff they share a one-step ancestor, largest slice with N ≥ 300.
Same angle_rho measurement, same dimension estimators + spread, plus the
random-mode and magnitude controls.

**Pre-registered classification (library.py conventions):** transfer =
angle_rho ≥ 0.80 with spread ≤ 0.30; partial = 0.5–0.8; correct failure
< 0.5 (reported as a negative, like the causal set — an informative boundary
of the observer principle, not a defeat). NO quantum-phase interpretation at
any outcome; the angular coordinate here is a real unit direction of an
eigenvector row, not a U(1) amplitude phase, and the Born-rule observer
discards phase and keeps magnitude — the opposite compression. Any language
connecting 3c to quantum phase requires a separate, dedicated experiment
design with interference-like predictions, not this test.
