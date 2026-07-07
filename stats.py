"""Uncertainty quantification for the paper's headline numbers (review-2 #6):
bootstrap 95% CIs over node-pair resampling, variation across graph draws, and
the small-world inter-estimator spread (review-2 moderate issue #3)."""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from rung0_validate import (
    torus_graph,
    _norm_laplacian_eigs,
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
)
from crosssubstrate import king_lattice_graph

RNG = np.random.default_rng(0)


def _dg(A, w, V, modes, kind, anchors):
    D = shortest_path(A, unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), 1)
    g = D[iu]
    Y = (V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9)))[anchors]
    if kind == "magnitude":
        r = np.linalg.norm(Y, axis=1)
        d = np.abs(r[:, None] - r[None, :])[iu]
    else:  # angle (used for both low-mode and random-mode)
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    return d, g


def boot_ci(d, g, b=2000):
    idx = np.arange(len(d))
    st = np.empty(b)
    for i in range(b):
        s = RNG.choice(idx, len(d), replace=True)
        st[i] = spearmanr(d[s], g[s]).statistic
    return np.percentile(np.abs(st), [2.5, 97.5])


print("=== headline uncertainty: 2-torus n=2000, n_anchor=150 ===")
A, _, _ = torus_graph(2000, 2)
w, V = _norm_laplacian_eigs(A)
anchors = RNG.choice(A.shape[0], 150, replace=False)
npairs = 150 * 149 // 2
low = np.arange(1, 21)
rand = RNG.choice(np.arange(1, len(w)), 20, replace=False)
for label, modes, kind in [
    ("angle", low, "angle"),
    ("magnitude", low, "magnitude"),
    ("random", rand, "angle"),
]:
    d, g = _dg(A, w, V, modes, kind, anchors)
    pt = abs(spearmanr(d, g).statistic)
    lo, hi = boot_ci(d, g)
    print(f"  {label:9s} rho={pt:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  n_pairs={npairs}")

print("\n=== across 8 independent torus draws (angle, low 20 modes) ===")
vals = []
for _ in range(8):
    Aa, _, _ = torus_graph(2000, 2)
    wa, Va = _norm_laplacian_eigs(Aa)
    anc = RNG.choice(Aa.shape[0], 150, replace=False)
    d, g = _dg(Aa, wa, Va, np.arange(1, 21), "angle", anc)
    vals.append(abs(spearmanr(d, g).statistic))
vals = np.array(vals)
print(f"  angle_rho over 8 draws: {vals.mean():.3f} +/- {vals.std():.3f}")

print("\n=== small-world king-lattice estimator spread (review-2 #3) ===")
for p in (0.0, 0.03):
    Ak = king_lattice_graph(side=45, rewire_p=p)
    wk, Vk = _norm_laplacian_eigs(Ak)
    db, ds, de = dim_ball_growth(Ak), dim_spectral(wk), dim_effective_rank(Ak, wk, Vk)
    tag = "clean p=0" if p == 0 else "small-world p=0.03"
    print(
        f"  {tag:20s} N={Ak.shape[0]} ball={db:.2f} spec={ds:.2f} eff={de:.2f} "
        f"spread={np.std([db, ds, de]):.2f}"
    )
