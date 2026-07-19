"""GO-B held-out manifold-recognition battery for Paper I.5 (the angular recognizer).

Extends recover.py's low-spectrum multiplet recognizer with three new closed-manifold
templates -- cylinder (S^1 x [0,1]), 3-torus T^3, and a genus-2 surface -- FROZEN on a
calibration instance, then confirmed on disjoint HELD-OUT instances (fresh seeds).

Registration-first: --calibrate prints the raw low-spectrum ratios that BECOME the frozen
templates (and the recognition threshold); after sealing, --confirm scores held-out
instances against the frozen templates and checks best-match == ground truth.

Generators (geometric random graphs, same style as rung0_validate torus/sphere):
  cylinder : points on [0,1) x [0,1], periodic in x only (S^1 x interval)
  torus3   : torus_graph(n, 3) -- 3D periodic box = flat T^3
  genus2   : points Newton-projected onto the implicit lemniscate tube
             ((x^2+y^2)^2 - (x^2-y^2))^2 + z^2 = eps^2  (a smooth genus-2 surface)

Run:  PYTHONPATH=. python experiments/manifold-recovery/manifold_battery.py --calibrate
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rung0_validate import _edges_to_csr, _norm_laplacian_eigs, torus_graph
from rung1b_rewriter import largest_component
from recover import measure  # full recognizer battery (dim, angle-rho, stress, ecc, signature)

HERE = os.path.dirname(os.path.abspath(__file__))

# FROZEN templates (low-spectrum ratios lambda_k/lambda_1, k=1..8), calibrated on seed 0.
# Sealed by GO-P-2026-041 before any held-out instance is scored.
FROZEN_TEMPLATES = {
    "cylinder": [1.0, 3.21, 3.49, 3.8, 4.34, 4.75, 6.88, 7.44],
    "T^3":      [1.0, 1.08, 1.14, 1.19, 1.37, 1.51, 1.8, 2.01],
    "genus-2":  [1.0, 2.5, 4.39, 5.38, 8.68, 9.7, 11.33, 12.11],
}
HELDOUT_SEEDS = [1, 2, 3, 4]      # disjoint from the calibration seed (0)


def cylinder_graph(n, rng, mean_deg=14, height=1.0):
    """S^1 x [0,height]: periodic in x (circumference 1), bounded in y."""
    pts = np.column_stack([rng.random(n), rng.random(n) * height])
    r = np.sqrt(mean_deg * height / (n * np.pi))
    tree = cKDTree(pts, boxsize=[1.0, height + 10.0])  # y-box >> extent => y non-periodic
    return largest_component(_edges_to_csr(n, tree.query_pairs(r)))


def torus3_graph(n, rng, mean_deg=14):
    """Flat T^3 via 3D periodic box (rung0 torus_graph, dim=3), reseeded per instance."""
    import rung0_validate as r0
    r0.RNG = rng
    A, _, _ = torus_graph(n, 3, mean_deg=mean_deg)
    return largest_component(A)


def _g2_f_grad(P, eps):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    s = x * x + y * y
    u = s * s - (x * x - y * y)                     # lemniscate level fn
    f = u * u + z * z - eps * eps
    du_dx = 2 * x * (2 * s - 1.0)
    du_dy = 2 * y * (2 * s + 1.0)
    g = np.column_stack([2 * u * du_dx, 2 * u * du_dy, 2 * z])
    return f, g


def genus2_graph(n, rng, mean_deg=26, eps=0.25, oversample=6):
    """Genus-2 = tube around the figure-8 lemniscate. Sample box points, Newton-project
    onto the implicit surface f=0, geometric-graph by 3D chordal distance."""
    m = n * oversample
    P = np.column_stack([rng.uniform(-1.35, 1.35, m), rng.uniform(-0.85, 0.85, m),
                         rng.uniform(-eps - 0.15, eps + 0.15, m)])
    for _ in range(8):                              # Newton projection onto f=0
        f, g = _g2_f_grad(P, eps)
        P = P - (f / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
    f, _ = _g2_f_grad(P, eps)
    keep = (np.abs(f) < 1e-4) & (np.abs(P[:, 0]) < 1.4) & (np.abs(P[:, 1]) < 0.9)
    P = P[keep]
    if len(P) > n:                                   # thin to n by a random subsample
        P = P[rng.choice(len(P), n, replace=False)]
    tree = cKDTree(P)
    # radius for target mean degree on a 2D surface embedded in R^3: tune by kth-NN scale
    d, _ = tree.query(P, k=min(mean_deg + 1, len(P)))
    r = float(np.median(d[:, -1]))
    return largest_component(_edges_to_csr(len(P), tree.query_pairs(r))), len(P)


def ratios(A):
    w, _ = _norm_laplacian_eigs(A)
    return [round(float(x), 3) for x in (w[1:9] / max(w[1], 1e-12))]


def gen(name, seed):
    rng = np.random.default_rng(seed)
    if name == "cylinder":
        return cylinder_graph(2000, rng)
    if name == "T^3":
        return torus3_graph(2000, rng)
    if name == "genus-2":
        A, npts = genus2_graph(2000, rng)
        return A
    raise ValueError(name)


def extended_signature(ratios):
    """Score a graph's low-spectrum ratios against the FULL template set (4 original from
    recover.TEMPLATES + the 3 FROZEN new manifolds); return (best_name, best_score, all_scores)."""
    from recover import TEMPLATES as ORIG
    allt = {**ORIG, **FROZEN_TEMPLATES}
    lam = np.array(ratios, float)
    scores = {name: float(np.sqrt(((np.log(lam) - np.log(np.array(t, float))) ** 2).mean()))
              for name, t in allt.items()}
    best = min(scores, key=scores.get)
    return best, scores[best], {k: round(v, 3) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    names = ["cylinder", "T^3", "genus-2"]

    if args.calibrate:
        print("=== CALIBRATION (seed 0): raw low-spectrum ratios lambda_k/lambda_1, k=1..8 ===")
        for nm in names:
            A = gen(nm, 0); rng = np.random.default_rng(0)
            res, _, _ = measure(nm, A, rng)
            best, sc, _ = extended_signature(res["spectrum_ratios"])
            print(f"{nm:9s} n={res['n']:4d} degCV={res['deg_cv']:.2f} d={res['dim']:.2f}"
                  f"(+-{res['dim_spread']:.2f}) ecc={res['ecc_spread']:.2f} angle={res['angle_rho']:.2f} "
                  f"ratios={res['spectrum_ratios']}  -> {best} ({sc:.3f})")
        return

    if args.confirm:
        print("=== HELD-OUT CONFIRM: seeds", HELDOUT_SEEDS, "scored vs FROZEN templates ===")
        rows, correct = [], 0
        for nm in names:
            for sd in HELDOUT_SEEDS:
                A = gen(nm, sd); rng = np.random.default_rng(100 + sd)
                res, _, _ = measure(nm, A, rng)
                best, sc, allsc = extended_signature(res["spectrum_ratios"])
                hit = best == nm
                correct += hit
                rows.append(dict(manifold=nm, seed=sd, n=res["n"], dim=res["dim"],
                                 ecc_spread=res["ecc_spread"], deg_cv=res["deg_cv"],
                                 angle_rho=res["angle_rho"], best_match=best,
                                 best_score=round(sc, 3), correct=hit,
                                 own_score=round(allsc[nm], 3)))
                print(f"{nm:9s} s={sd} d={res['dim']:.2f} ecc={res['ecc_spread']:.2f} "
                      f"angle={res['angle_rho']:.2f} -> {best:14s} ({sc:.3f}) "
                      f"{'OK' if hit else 'MISS (own=%.3f)' % allsc[nm]}")
        ntot = len(names) * len(HELDOUT_SEEDS)
        # dimension ordering: T^3 median dim strictly above cylinder & genus-2
        dim_by = {nm: np.median([r["dim"] for r in rows if r["manifold"] == nm]) for nm in names}
        dim_order = dim_by["T^3"] > max(dim_by["cylinder"], dim_by["genus-2"])
        checks = {"signature recognition >= 10/12": correct >= 10,
                  "T^3 dimension highest (>cyl,>genus2)": bool(dim_order)}
        print(f"\nrecognition {correct}/{ntot}; median dims {({k: round(v,2) for k,v in dim_by.items()})}")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        verdict = "CONFIRMED" if all(checks.values()) else "NOT CONFIRMED"
        print(f"\nVERDICT: {verdict}")
        out = dict(id="GO-P-2026-041", frozen_templates=FROZEN_TEMPLATES,
                   heldout_seeds=HELDOUT_SEEDS, rows=rows, recognition=f"{correct}/{ntot}",
                   dim_medians={k: round(v, 2) for k, v in dim_by.items()},
                   checks=checks, verdict=verdict)
        with open(os.path.join(HERE, "battery_result.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("wrote battery_result.json")
        return


if __name__ == "__main__":
    main()
