"""Composite #2 — a LIBRARY of independent clean emergent manifolds.

Pools the rules that produced clean geometry (Gorard/Wolfram borrowed #1 +
evolved #2), grows each across several rewriter seeds (independent manifold
instances), and measures the observer metric with the full control battery.

Two composite deliverables stronger than any single rule:
  (A) angle-rho DISTRIBUTION over independent clean ~2D manifolds  -> mu +/- sigma
  (B) the angle-rho-vs-EMERGENT-DIMENSION law, spanning d ~ 1 .. 3.3
      (quantifies the degradation #1 spotted, across independent families)
Controls per manifold: random_rho (random modes, must ~0) and euclid_rho (2D MDS).
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from rung1b_rewriter import rewrite, largest_component
from rung0_validate import (
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
    _norm_laplacian_eigs,
)
from arity3 import hyper_to_csr

N_CAP, SEEDS, M = 2000, (1, 2, 3, 4), 20

# all arity-3; seed chosen per rule to the one that actually grows it
S_COLLAPSE = [(0, 0, 0), (0, 0, 0)]
S_TET = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
LIB = {
    "R_2D  (borrowed)": dict(
        lhs=[("1", "2", "2"), ("3", "1", "4")],
        rhs=[("2", "5", "2"), ("2", "3", "5"), ("4", "5", "5")],
        seed=S_COLLAPSE,
    ),
    "R_frac(borrowed)": dict(
        lhs=[("1", "2", "3")],
        rhs=[("1", "4", "6"), ("2", "5", "4"), ("3", "6", "5")],
        seed=[(0, 1, 2)],
    ),
    "evolved-2D (#2)": dict(
        lhs=[("a", "b", "c")],
        rhs=[("b", "x", "c"), ("x", "x", "b"), ("c", "a", "y")],
        seed=S_TET,
    ),
    "R_3D  (borrowed)": dict(
        lhs=[("1", "1", "2"), ("3", "4", "1")],
        rhs=[("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")],
        seed=S_COLLAPSE,
    ),
    "R_SR  (borrowed)": dict(
        lhs=[("v1", "v2", "v3"), ("v2", "v4", "v5")],
        rhs=[("v5", "v6", "v1"), ("v6", "v4", "v2"), ("v4", "v5", "v3")],
        seed=[(1, 2, 3), (2, 4, 5), (4, 6, 7)],
    ),
}


def crop(A, target=N_CAP):
    n = A.shape[0]
    if n <= target:
        return A
    d = shortest_path(
        A, unweighted=True, indices=[np.random.default_rng(0).integers(n)]
    )[0]
    keep = np.sort(np.argsort(d)[:target])
    return largest_component(A[keep][:, keep])


def _rho(A, w, V, modes, rng, angle=True, n_anchor=150):
    anc = rng.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = D[iu]
    Y = (V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9)))[anc]
    if angle:
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    return abs(spearmanr(d, g).statistic)


def euclid_rho(A, rng, n_anchor=150):
    anc = rng.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    n = len(anc)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    X = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))
    d = np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1)
    return abs(spearmanr(d, D[iu]).statistic)


def measure(name, R):
    rows = []
    for s in SEEDS:
        tris, nid = rewrite(
            R["lhs"], R["rhs"], R["seed"], max_edges=3000, max_gen=500, seed=s
        )
        A = largest_component(hyper_to_csr(tris, nid))
        if A.shape[0] < 200:
            continue
        A = crop(A)
        w, V = _norm_laplacian_eigs(A)
        rng = np.random.default_rng(100 + s)
        dim = np.mean([dim_ball_growth(A), dim_spectral(w)])
        spread = np.std(
            [dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)]
        )
        low = np.arange(1, 1 + M)
        rand = rng.choice(np.arange(1, len(w)), size=M, replace=False)
        rows.append(
            dict(
                dim=dim,
                spread=spread,
                angle=_rho(A, w, V, low, rng, True),
                random=_rho(A, w, V, rand, rng, True),
                euclid=euclid_rho(A, rng),
            )
        )
    return rows


if __name__ == "__main__":
    print("MANIFOLD LIBRARY — observer metric across independent clean manifolds\n")
    print(
        f"{'rule':18s} {'n':>2} {'dim':>10} {'spread':>7} {'angle_rho':>16} "
        f"{'random':>7} {'euclid':>7}"
    )
    allpts = []
    for name, R in LIB.items():
        rows = measure(name, R)
        if not rows:
            print(f"{name:18s}  (stalled)")
            continue
        d = np.array([r["dim"] for r in rows])
        a = np.array([r["angle"] for r in rows])
        rr = np.array([r["random"] for r in rows])
        e = np.array([r["euclid"] for r in rows])
        sp = np.array([r["spread"] for r in rows])
        for di, ai in zip(d, a):
            allpts.append((di, ai))
        print(
            f"{name:18s} {len(rows):2d} {d.mean():5.2f}+/-{d.std():4.2f} "
            f"{sp.mean():7.2f} {a.mean():7.3f}+/-{a.std():4.3f}    "
            f"{rr.mean():7.3f} {e.mean():7.3f}"
        )

    pts = np.array(allpts)
    # (A) 2D distribution: manifolds with measured dim in [1.8, 2.3]
    m2d = pts[(pts[:, 0] >= 1.8) & (pts[:, 0] <= 2.3)]
    print(
        f"\n(A) angle_rho over independent ~2D manifolds (dim in [1.8,2.3], "
        f"n={len(m2d)}): {m2d[:,1].mean():.3f} +/- {m2d[:,1].std():.3f}"
    )
    # (B) degradation law: angle_rho vs dimension
    if len(pts) >= 3:
        slope, inter = np.polyfit(pts[:, 0], pts[:, 1], 1)
        r = spearmanr(pts[:, 0], pts[:, 1]).statistic
        print(
            f"(B) angle_rho vs emergent dim (n={len(pts)} manifolds, d~"
            f"{pts[:,0].min():.1f}..{pts[:,0].max():.1f}): "
            f"slope={slope:+.3f}/dim, Spearman={r:+.3f}"
        )
        print(
            f"    => observer fidelity {'FALLS' if slope<0 else 'RISES'} "
            f"with emergent dimension (the composite law)."
        )
