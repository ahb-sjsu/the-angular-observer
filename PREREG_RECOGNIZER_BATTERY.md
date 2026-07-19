# Pre-registration — Recognizer held-out battery (Paper I.5)

Registered 2026-07-19, BEFORE any held-out instance was scored. Bars below are
frozen; a FAIL is reported as a FAIL, no salvage without a dated amendment (same
discipline as `PREREG_RUNG3.md` / `PREREG_RUNG4.md`).

## Why this battery

The angular manifold recognizer (`experiments/manifold-recovery/recover.py`)
identifies which closed manifold an emergent geometry resembles from its
low-spectrum multiplet signature (λ_k/λ_1, k=1..8) against templates. To date it
carried four templates (flat 2-torus, sphere S², Neumann square patch, interval).
This battery **adds three** — cylinder (S¹×[0,1]), 3-torus T³, and a genus-2
surface — freezes them on a calibration instance, and asks the held-out question:

> **Do disjoint held-out instances of each new manifold recover their OWN frozen
> signature template (not a neighbour's), and does the recognizer read T³'s
> dimension strictly above the two 2-D surfaces — dimension separating before
> shape does?**

This is the paper's "dimension emerges before shape" headline extended to a
richer template library, under registration.

## Frozen templates (calibrated on seed 0)

Low-spectrum ratios λ_k/λ_1, k=1..8 (in `manifold_battery.FROZEN_TEMPLATES`):
- **cylinder**: `[1.0, 3.21, 3.49, 3.8, 4.34, 4.75, 6.88, 7.44]`
- **T³**:      `[1.0, 1.08, 1.14, 1.19, 1.37, 1.51, 1.8, 2.01]`  (near-flat multiplet; theory `[1,1,1,1,1,1,2,2]`)
- **genus-2**: `[1.0, 2.5, 4.39, 5.38, 8.68, 9.7, 11.33, 12.11]` (rising, multiplet-free — symmetry broken)

Scored against the full 7-template set (these 3 + the 4 originals).

## Frozen generators (geometric random graphs, n≈2000)

- **cylinder**: points on `[0,1)×[0,1]`, periodic in x only (cKDTree
  `boxsize=[1.0, height+10]`), mean degree 14.
- **T³**: `rung0_validate.torus_graph(n, 3)`, 3-D periodic box, mean degree 14.
- **genus-2**: box points Newton-projected (8 steps) onto the implicit lemniscate
  tube `((x²+y²)²−(x²−y²))² + z² = ε²`, ε=0.25, mean degree 26, oversample 6;
  geometric graph by 3-D chordal distance at the median-`k`-NN radius.

Frozen recognizer code (`recover.py` + `manifold_battery.py`) binds to
`sha256:eee6eeda856c87846082894dfcdff53c7692d2c83c1184f756ede44c9ab91d8f`.

## Method (registered)

Per (manifold, held-out seed): build the graph (largest component), run
`recover.measure` for the low-spectrum ratios + dimension + ecc-spread + angle-ρ,
then `extended_signature` picks the best-matching template among all 7 by
log-RMS distance. Held-out seeds **{1, 2, 3, 4}** (disjoint from calibration seed 0).

## Pre-registered bars (frozen)

- **PRIMARY — recognition**: across the 3 manifolds × 4 held-out seeds (n=12), the
  best-matching template equals the true manifold on **≥ 10/12**.
- **SECONDARY — dimension before shape**: median recovered dimension of T³ is
  strictly greater than both cylinder and genus-2 (dimension separates the 3-D
  manifold from the 2-D ones before the shape signature is read).

## Honest scope, fixed in advance

- The genus-2 instance is an **embedded, non-constant-curvature** surface (a tube
  around the figure-8 lemniscate), so it is geometrically **inhomogeneous** and
  reads a broad eccentricity spread (~0.48), unlike a closed homogeneous manifold
  (torus/sphere ecc ~0.10). Closedness/homogeneity is therefore **not** a sealed
  bar for genus-2; the sealed claims are signature recognition and the dimension
  ordering. The broad ecc is reported, not massaged.
- Cylinder is a manifold **with boundary**; its broad ecc (~0.53) is expected and
  reported, not a failure.

## Interpretation, fixed in advance

- **PASS** (recognition ≥10/12 AND dim ordering): the recognizer's template
  library extends to cylinder, T³, and genus-2 with held-out stability; dimension
  separates before shape. Joins Paper I.5 as the recognition battery.
- **FAIL** (either bar): reported as a FAIL. If recognition misses, the confusable
  pair is named (a real limit of the low-spectrum signature); if the dimension
  ordering fails, "dimension before shape" is bounded, not restated.

## Compute

CPU, dense `eigh` (n≤2000), pure NumPy/SciPy. Deterministic per seed. Results to
`experiments/manifold-recovery/battery_result.json`.

## Outcome — CONFIRMED, 2026-07-19

Held-out seeds {1,2,3,4}, scored vs the frozen templates (sealed `e8d9bd2` before
any held-out run):
- **PRIMARY recognition 12/12** — every instance best-matched its own template:
  cylinder→cylinder ×4, T³→T³ ×4, genus-2→genus-2 ×4 (best-scores 0.03–0.23; no
  confusions).
- **SECONDARY dimension before shape** — median recovered dimension T³ **2.80** >
  cylinder **1.81**, genus-2 **1.74**; T³ also reads closed (ecc ~0.11) while the
  two 2-D surfaces read bounded/inhomogeneous (ecc ~0.5), as pre-stated.

Both frozen bars PASS → **CONFIRMED**. The recognizer's template library extends to
cylinder, T³, and genus-2 with held-out stability; the "dimension emerges before
shape" headline holds on the enlarged library.
`experiments/manifold-recovery/battery_result.json`.
