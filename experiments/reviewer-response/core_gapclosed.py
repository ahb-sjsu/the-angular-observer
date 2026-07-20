"""Core torus table at GAP-CLOSED cutoffs m in {4,8,12,20} (multiplet-closed on
T^2), 5 independent draws each — review comment 19. Same angle/full/magnitude
measurement as todo_experiments.headline_bootstrap."""

import sys, json, numpy as np, statistics as st

sys.path.insert(0, "../..")
sys.path.insert(0, ".")
import rung0_validate as rv
from rung0_validate import torus_graph, _norm_laplacian_eigs
from todo_experiments import _pair_dist, _rho, _anchor_geodesics

MS = [4, 8, 12, 20]
acc = {m: {"angle": [], "full": [], "magnitude": []} for m in MS}
for s in range(5):
    rv.RNG = np.random.default_rng(5000 + s)  # same draws as ablation/kappa
    A = torus_graph(2000, 2)[0]
    w, V = _norm_laplacian_eigs(A)
    anc, iu, g, ok = _anchor_geodesics(A, 150, 100 + s)
    gok = g[ok]
    for m in MS:
        Yl = (V[:, 1 : 1 + m] / np.sqrt(np.maximum(w[1 : 1 + m], 1e-9)))[anc]
        Y = Yl / np.maximum(np.linalg.norm(Yl, axis=1, keepdims=True), 1e-12)
        rad = np.linalg.norm(Yl, axis=1, keepdims=True)
        acc[m]["angle"].append(_rho(_pair_dist(Y, iu)[ok], gok))
        acc[m]["full"].append(_rho(_pair_dist(Yl, iu)[ok], gok))
        acc[m]["magnitude"].append(_rho(_pair_dist(rad, iu)[ok], gok))
res = {}
print("=== core table, gap-closed cutoffs, 5 draws (mean +/- sd) ===")
print(
    f"{'m':>3} | {'full':>13} | {'angle':>13} | {'magnitude':>13}   (gap-closed on T^2)"
)
for m in MS:
    r = {
        k: dict(mean=round(st.mean(v), 3), sd=round(st.pstdev(v), 3), draws=v)
        for k, v in acc[m].items()
    }
    res[f"m{m}"] = r
    print(
        f"{m:>3} | {r['full']['mean']:.3f}+/-{r['full']['sd']:.3f} | "
        f"{r['angle']['mean']:.3f}+/-{r['angle']['sd']:.3f} | {r['magnitude']['mean']:.3f}+/-{r['magnitude']['sd']:.3f}"
    )
json.dump(res, open("core_gapclosed_result.json", "w"), indent=2)
print(
    "\nheadline (gap-closed m=20): angle =",
    f"{res['m20']['angle']['mean']:.3f} +/- {res['m20']['angle']['sd']:.3f}",
    res["m20"]["angle"]["draws"],
)
