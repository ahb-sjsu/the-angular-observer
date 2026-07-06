"""The mathematical trick (from geometric-methods Ch.3): measure emergent
Wolfram-rewrite geometry in HYPERBOLIC space, not Euclidean.

Diagnosis (Ch.3 Thm 3.1): a polynomial-growth (Euclidean, Vol~r^d) space cannot
hold exponentially-branching structure without huge distortion -> that IS our
"tangle" (eff_rank 11, ball 3.4, spread 3.8). Wolfram rewriting grows
exponentially => the emergent graph is tree-like => hyperbolic. Thm 3.2: any
tree embeds in 2D hyperbolic at distortion 1+eps.

Two decisive tests per graph:
  (1) GROWTH LAW: is log N(r) more linear in r (hyperbolic, Vol~e^{(d-1)r})
      or in log r (Euclidean, Vol~r^d)?  Compare R^2.
  (2) OBSERVER PRESERVATION: does a 2D POINCARE embedding reproduce graph
      geodesics better than a 2D EUCLIDEAN (classical-MDS) embedding?
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from rung0_validate import torus_graph, RNG
from arity3 import (
    rewrite,
    hyper_to_csr,
    largest_component,
    CANONICAL,
    SEED3,
    gen_rule3,
    score_rule3,
)


def tree_graph(n=2000, k=2):
    """Balanced k-ary tree — a KNOWN hyperbolic graph (positive control)."""
    ii, jj = [], []
    for i in range(1, n):
        p = (i - 1) // k
        ii += [i, p]
        jj += [p, i]
    A = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    A.data[:] = 1.0
    return A


def gromov_delta(A, n_sample=120):
    """Sampled 4-point delta-hyperbolicity, normalized by diameter.
    ~0 => tree-like/hyperbolic; large => not."""
    s = RNG.choice(A.shape[0], size=min(n_sample, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=s)[:, s]
    diam = D[np.isfinite(D)].max()
    deltas = []
    idx = np.arange(len(s))
    for _ in range(4000):
        w, x, y, z = RNG.choice(idx, 4, replace=False)
        d1 = D[w, x] + D[y, z]
        d2 = D[w, y] + D[x, z]
        d3 = D[w, z] + D[x, y]
        a, b, _c = sorted([d1, d2, d3])
        deltas.append((b - a) / 2)
    return np.mean(deltas) / diam


# ---- Poincare ball (numpy port of geometric-methods Ch.3 PoincareBall) ----
EPS = 1e-5


def _mobius_add(x, y, c):
    xy = (x * y).sum(-1, keepdims=True)
    xx = (x * x).sum(-1, keepdims=True)
    yy = (y * y).sum(-1, keepdims=True)
    num = (1 + 2 * c * xy + c * yy) * x + (1 - c * xx) * y
    den = (1 + 2 * c * xy + c * c * xx * yy).clip(EPS)
    return num / den


def poincare_dist(x, y, c=1.0):
    d = _mobius_add(-x, y, c)
    dn = np.linalg.norm(d, axis=-1).clip(EPS)
    arg = (np.sqrt(c) * dn).clip(max=1 - EPS)
    return (2 / np.sqrt(c)) * np.arctanh(arg)


def spectral_init(D, dim=2, c=1.0, max_norm=0.9):  # Ch.3 §3.3.2
    sigma = np.median(D[D > 0])
    K = np.exp(-(D**2) / (2 * sigma**2))
    w, V = np.linalg.eigh(K)
    coords = V[:, -dim:][:, ::-1] * np.sqrt(np.maximum(w[-dim:][::-1], 0.0))
    m = np.max(np.linalg.norm(coords, axis=1))
    if m > 1e-8:
        coords *= (max_norm / np.sqrt(c)) / m
    return coords


def classical_mds(D, dim=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    return V[:, -dim:][:, ::-1] * np.sqrt(np.maximum(w[-dim:][::-1], 0.0))


# ------------------------------------- growth law + observer preservation ----
def growth_fit(A, n_src=40):
    src = RNG.choice(A.shape[0], size=min(n_src, A.shape[0]), replace=False)
    Dm = shortest_path(A, unweighted=True, indices=src)
    rmax = int(np.nanmax(Dm[np.isfinite(Dm)]))
    rs = np.arange(1, rmax + 1)
    N = np.array([np.mean([np.sum(row <= r) for row in Dm]) for r in rs])
    lo, hi = 1, np.searchsorted(N, 0.6 * A.shape[0])
    hi = max(hi, lo + 3)
    r, y = rs[lo:hi], np.log(N[lo:hi])

    def r2(x):
        p = np.polyfit(x, y, 1)
        res = y - np.polyval(p, x)
        return 1 - res.var() / y.var(), p[0]

    eucl_r2, _ = r2(np.log(r))  # log N vs log r  (Euclidean r^d)
    hyp_r2, hyp_slope = r2(r)  # log N vs r      (hyperbolic e^{(d-1)r})
    return eucl_r2, hyp_r2, hyp_slope + 1  # slope+1 ~ hyperbolic dim


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def preservation(A, n_anchor=200):
    anc = RNG.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = D[iu]
    Xe = classical_mds(D, 2)
    de = np.linalg.norm(Xe[iu[0]] - Xe[iu[1]], axis=1)
    Xh = spectral_init(D, 2)
    dh = poincare_dist(Xh[iu[0]], Xh[iu[1]])
    return abs(spearman(de, g)), abs(spearman(dh, g))


def analyze(name, A):
    er2, hr2, hdim = growth_fit(A)
    pe, ph = preservation(A)
    delta = gromov_delta(A)
    print(
        f"{name:>18} N={A.shape[0]:5d} | delta/diam={delta:.3f} | growth R^2 "
        f"eucl={er2:.3f} hyp={hr2:.3f} | geodesic pres  "
        f"EUCL={pe:.3f}  HYP={ph:.3f}"
    )


def find_tangle():
    """Grab a genuine high-eff_rank random arity-3 tangle (not the 2D-refine rule)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        r = gen_rule3(rng)
        if not r:
            continue
        m = score_rule3(r[0], r[1], rng, max_edges=1500)
        if m.get("valid") and m["eff"] > 4.0 and m["spread"] > 0.8:
            tris, nid = rewrite(r[0], r[1], SEED3, max_edges=1500, seed=1)
            return largest_component(hyper_to_csr(tris, nid)), m
    return None, None


if __name__ == "__main__":
    print("Hyperbolic re-analysis (Ch.3 trick) — with positive/negative controls\n")
    analyze("binary-tree(+ctrl)", tree_graph(2000, 2))  # KNOWN hyperbolic
    analyze("2-torus(-ctrl)", torus_graph(2000, 2)[0])  # KNOWN Euclidean
    tris, nid = rewrite(*CANONICAL, SEED3, max_edges=2000, max_gen=300, seed=1)
    analyze("arity3-canonical", largest_component(hyper_to_csr(tris, nid)))
    A_tan, m = find_tangle()
    if A_tan is not None:
        print(f"  (random tangle rule: eff={m['eff']:.1f} spread={m['spread']:.2f})")
        analyze("arity3-random-tangle", A_tan)
    print(
        "\nTest is VALID only if the tree shows low delta + HYP>>EUCL. "
        "Then whether tangles are hyperbolic is decided by THEIR delta."
    )
