# Green-rank numerics: master-inequality validation + counterexample hunt

**Status: `[exploratory]`.** These two scripts numerically probe the results of
`paper/simods/green-rank-positivity.tex` (the *Master inequality* and Green-rank
positivity, i.e. Conjecture SM2.9). They **support** the master inequality and **find no
counterexample**; they do **not** prove Green-rank positivity, which the note leaves open.
Nothing here is a sealed confirmatory claim — no umbrella statement may cite these rows
(PROTOCOL Rule 1.1).

## What is being measured

The **Green-rank coefficient** `rho_G(M) = rho_S(d_M(X,Y), -G_M(X,Y))`, the Spearman rank
correlation between geodesic distance and the (negated) zero-mean Green kernel over i.i.d.
uniform point pairs. Green-rank positivity is the statement `rho_G(M) > 0`; on two-point
homogeneous spaces `rho_G = 1` exactly. The *Master inequality* is
`rho_G >= 1 - 3 mu_Psi(2||R||_inf)`, where `||R||_inf` is the non-radiality of `-G` against
its best monotone-radial fit (here estimated by isotonic regression) and `mu_Psi` the
near-tie fraction of the fitted profile.

## `rho_g_torus_sweep.py` — deformed flat tori (homogeneous family)

Uses the **verified** truncated-kernel limit (supplement.tex SM2.7,
`rho_S(d_M,-G_Lambda) -> rho_G`) with exact plane-wave sums — no Ewald summation, no
manifold discretization. Sweeps aspect ratio `a` (unit-area torus `aZ x (1/a)Z`), 5 draws.

**Findings** (`rho_g_torus_sweep_result.json`):
- `rho_G` is well-converged in the cutoff `Lambda` (3 cutoffs agree to 3-4 digits).
- `rho_G` stays `>= 0.98` and **rises to 1.000 as the torus thins** (a thin rectangular
  torus degenerates toward the circle, where `G` is exactly monotone). **No counterexample.**
- The **master inequality holds in every draw, every aspect** (`master_holds=True`).

## `rho_g_dumbbell.py` — surfaces of revolution (sphere control + thin-neck dumbbells)

The *inhomogeneous* geometries the torus cannot reach. Discrete, intrinsic pipeline (no R^3
embedding): sample uniformly on `du^2 + f(u)^2 dtheta^2`, build a symmetric kNN graph with
Gaussian weights, take the **Green kernel = pseudoinverse of the graph Laplacian** and the
**geodesic = Dijkstra** on the weighted graph. 5 draws.

**Control (load-bearing):** the round sphere is two-point homogeneous, so `rho_G` must be
`~1`. Measured `0.9912 +/- 0.0016` -> the ~1% gap is the discretization floor, and the
method is **trustworthy** (`method_trustworthy=True`). Dumbbell numbers are read against
this floor.

**Findings** (`rho_g_dumbbell_result.json`):
- Closing the neck drives `rho_G` monotonically **0.98 -> ~0.83**, while the non-radiality
  `||R||` **climbs 0.055 -> 0.17** — the master-inequality mechanism made visible:
  inhomogeneity grows `||R||`, which lowers the `rho_G` floor.
- `rho_G` stays clearly **positive and plateaus** (~0.826) as the neck pinches. **No
  counterexample** — positive evidence that Green-rank positivity survives strong
  inhomogeneity in this family, opposite to the homogeneous torus (which rises to 1).

## `rho_g_hunt.py` — round 2: multi-neck, asymmetric, and Berger spheres

Extends the hunt to the geometries most likely to break positivity, on the same generic
pipeline (a local-distance matrix -> graph-Laplacian Green kernel + Dijkstra geodesic).

- **2-D inhomogeneous** (multi-neck chains, asymmetric lobes): `rho_G` drops to **0.79–0.85**,
  plateauing like the single neck; lowest overall is the asymmetric lobe at **0.79**.
- **Berger spheres** (3-D, homogeneous but NOT two-point homogeneous — round `S^3` squashed
  by `eps^2` along the Hopf fiber): **round-`S^3` control `eps=1` gives `rho_G=0.9819`**
  (pipeline validated in 3-D), and `rho_G` then stays **0.97–0.99 across every squash**,
  `||R||~0.03`, *rising* toward 1 as `eps->0`. Clean reason: a Berger sphere interpolates two
  two-point homogeneous spaces (round `S^3` at `eps=1`, base `S^2` in the collapse), both with
  `rho_G=1`, so it stays near-radial — anisotropy alone does not decouple `G` from distance.

**Headline** (`rho_g_hunt_result.json`): the two geometries the SIMODS note itself flags as
counterexample candidates — **thin necks of revolution AND Berger spheres — both fail to
produce a counterexample.** Across three independent families (tori, surfaces of revolution,
Berger) `rho_G` never drops below **0.79**; the master-inequality target `rho_G <= 0`
(`mu_Psi(2||R||) >= 1/3`) is nowhere approached. Accumulating evidence *for* Green-rank
positivity — still not a proof.

## `rho_g_handle.py` — round 3: topological shortcuts (thin handles)

A short thin handle connecting two distant sphere regions is the one construction that
*decouples* `G` from geodesic distance: cross-handle pairs get a short geodesic (through the
handle) but high resistance (thin handle) => a rank reversal. Embedded R^3 point-cloud
pipeline; **plain-sphere control `rho_G=0.9916`** (validated).

**Finding** (`rho_g_handle_result.json`): the handle **barely moves `rho_G`** — it stays
~0.97 down to a very thin handle (87 points). The reversal is real (`||R||` spikes to 0.24)
but affects only the near-pole cross-handle pairs, a vanishing fraction of all `C(N,2)`, so
the global rank correlation is unmoved. **No counterexample.** Secondary observation: the
sup-norm master bound goes vacuously negative here while true `rho_G` stays ~0.97 — the
inequality is loose precisely when non-radiality is concentrated on *rare* pairs.

## Cumulative status of the hunt (4 families, no counterexample)

| family | mechanism | lowest `rho_G` | breaks positivity? |
|---|---|---|---|
| deformed flat tori | homogeneous | ~0.98 (→1 thinning) | no |
| surfaces of revolution (necks) | macroscopic bottleneck | **0.79** (asymmetric) | no |
| Berger spheres | homogeneous anisotropy | 0.97 (→1 collapse) | no |
| sphere + thin handle | topological shortcut | 0.97 | no |

Across all four, `rho_G` never drops below **0.79**; the master-inequality target for a
counterexample (`rho_G <= 0`, `mu_Psi(2||R||) >= 1/3`) is nowhere approached. The
mechanisms that move `rho_G` most are macroscopic bottlenecks, which **plateau ~0.79** —
suggestive (not proof) that `rho_G` may be bounded below by a universal constant for
surfaces, a stronger statement than mere positivity and a candidate theorem direction.

## Scope / caveats (kept honest)

- Rank-level results over sampled pairs; the dumbbell uses a **discrete graph-Laplacian
  approximation** of Laplace-Beltrami — the sphere control quantifies its ~1% floor.
- Necks (single, multi, asymmetric) and Berger spheres are now covered (rho_g_hunt.py) and
  clear. The remaining live candidate is **topological shortcuts** — high-genus / handle-body
  surfaces, where two points can be geodesically far yet connected by a short handle (low
  resistance => `G` less negative), the one mechanism that genuinely decouples `G` from
  geodesic distance. The master inequality gives the exact target throughout: any
  counterexample must have `mu_Psi(2||R||_inf) >= 1/3` for *every* monotone profile
  (equivalently `rho_G <= 0`).
- Reproducible: fixed seeds (`BASE_SEED=20260720`), deterministic per-config offsets.

## Reproduce

```
python rho_g_torus_sweep.py      # writes rho_g_torus_sweep_result.json
python rho_g_dumbbell.py         # writes rho_g_dumbbell_result.json  (needs scipy)
```
