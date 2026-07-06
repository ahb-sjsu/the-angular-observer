"""
Rung 0 — Estimator validation against ground truth.

Claim under test (Wolfram observer <-> compressibility bridge):
    Your "effective-rank-in-a-learned-eigenbasis" estimator recovers the SAME
    emergent dimension that Wolfram/Gorard's ball-growth estimator does.

Before trusting either on emergent-dimension hypergraphs, we check that all
three estimators recover a KNOWN dimension on random geometric graphs sampled
from manifolds of ground-truth dimension d:

    flat 2-torus  -> d = 2     (periodic [0,1)^2)
    flat 3-torus  -> d = 3     (periodic [0,1)^3)
    2-sphere S^2  -> d = 2     (embedded in R^3)

Estimators:
    (1) ball-growth      N(r) ~ r^d               [Wolfram / Gorard]
    (2) spectral dim     P(t) = <exp(-L t)> ~ t^-(d_s/2)   [Laplacian heat kernel]
    (3) effective rank   local PCA participation ratio in the Laplacian
                         eigenbasis                [the compressibility estimator]

No GPU, no cluster. If (3) recovers d and tracks (1), the bridge is worth
scaling to real hypergraphs (Rung 1) on Atlas.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- graph builders
def _edges_to_csr(n, pairs):
    if not pairs:
        raise ValueError("no edges — radius too small")
    i = np.array([a for a, b in pairs] + [b for a, b in pairs])
    j = np.array([b for a, b in pairs] + [a for a, b in pairs])
    data = np.ones(len(i))
    A = csr_matrix((data, (i, j)), shape=(n, n))
    A.data[:] = 1.0  # dedupe -> binary
    return A


def torus_graph(n, dim, mean_deg=14):
    pts = RNG.random((n, dim))
    # radius for target mean degree: deg = n * V_dim(r), V_2=pi r^2, V_3=4/3 pi r^3
    if dim == 2:
        r = np.sqrt(mean_deg / (n * np.pi))
    else:
        r = (3 * mean_deg / (4 * np.pi * n)) ** (1 / 3)
    tree = cKDTree(pts, boxsize=1.0)          # periodic box -> flat torus
    pairs = tree.query_pairs(r)
    return _edges_to_csr(n, pairs), pts, r


def sphere_graph(n, mean_deg=14):
    v = RNG.normal(size=(n, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True)   # uniform on S^2
    theta = np.sqrt(4 * mean_deg / n)                    # geodesic cap radius
    r = 2 * np.sin(theta / 2)                            # chordal threshold
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r)
    return _edges_to_csr(n, pairs), pts, r


# ------------------------------------------------------------------- estimators
def dim_ball_growth(A, n_src=60):
    """N(r) ~ r^d  (r = graph hop distance).  Wolfram / Gorard ball growth."""
    n = A.shape[0]
    src = RNG.choice(n, size=min(n_src, n), replace=False)
    D = shortest_path(A, method="D", unweighted=True, indices=src)  # (n_src, n)
    maxhop = int(np.nanmax(D[np.isfinite(D)]))
    rs = np.arange(1, maxhop + 1)
    counts = np.array([[np.sum(row <= r) for r in rs] for row in D], float)
    Nr = counts.mean(0)
    # fit in the intermediate regime: drop first hop (lattice noise) and the
    # saturating tail (finite-size cutoff)
    lo, hi = 1, np.searchsorted(Nr, 0.6 * n)
    hi = max(hi, lo + 3)
    x, y = np.log(rs[lo:hi]), np.log(Nr[lo:hi])
    d = np.polyfit(x, y, 1)[0]
    return d


def _norm_laplacian_eigs(A):
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Ad = A.toarray() * dinv[:, None] * dinv[None, :]
    L = np.eye(A.shape[0]) - Ad
    L = 0.5 * (L + L.T)
    w, V = eigh(L)                       # full spectrum, symmetric normalized L
    w = np.clip(w, 0, None)
    return w, V


def dim_spectral(w):
    """P(t) = mean(exp(-w t)) ~ t^-(d_s/2) in the intermediate-t regime."""
    ts = np.logspace(0.3, 2.2, 40)                 # intermediate diffusion times
    P = np.array([np.mean(np.exp(-w * t)) for t in ts])
    x, y = np.log(ts), np.log(P)
    # fit the middle 60% where power-law scaling lives
    a, b = int(0.2 * len(ts)), int(0.8 * len(ts))
    slope = np.polyfit(x[a:b], y[a:b], 1)[0]
    return -2 * slope


def dim_effective_rank(A, w, V, m_modes=40, n_probe=200, k_hop=2):
    """Local PCA participation ratio in the Laplacian eigenbasis.

    Laplacian eigenmaps: coords = bottom m non-trivial eigenvectors (skip the
    lambda=0 constant mode). For each probe node take its k-hop neighborhood,
    PCA the coords, and count effective dimensions via the participation ratio
    pr = (sum s_i)^2 / sum s_i^2  of the PCA variances.  This is the
    'effective rank in a learned eigenbasis' compressibility estimator.
    """
    n = A.shape[0]
    coords = V[:, 1:1 + m_modes]                    # skip constant mode
    D = shortest_path(A, method="D", unweighted=True,
                      indices=RNG.choice(n, size=n_probe, replace=False))
    prs = []
    for row in D:
        nb = np.where(row <= k_hop)[0]
        if len(nb) < 8:
            continue
        X = coords[nb]
        X = X - X.mean(0)
        s = np.linalg.svd(X, compute_uv=False) ** 2   # PCA variances
        s = s[s > 1e-12]
        pr = (s.sum() ** 2) / (s ** 2).sum()          # participation ratio
        prs.append(pr)
    return float(np.mean(prs))


# ------------------------------------------------------------------------- main
def run(name, A, pts, r, d_true):
    w, V = _norm_laplacian_eigs(A)
    d_ball = dim_ball_growth(A)
    d_spec = dim_spectral(w)
    d_eff = dim_effective_rank(A, w, V)
    deg = np.asarray(A.sum(1)).ravel().mean()
    print(f"{name:>10} | N={A.shape[0]:5d} <deg>={deg:4.1f} | "
          f"d_true={d_true} | ball={d_ball:5.2f}  spectral={d_spec:5.2f}  "
          f"eff_rank={d_eff:5.2f}")
    return d_ball, d_spec, d_eff


if __name__ == "__main__":
    print("Rung 0 — estimator validation on ground-truth manifolds\n")
    A, p, r = torus_graph(2000, 2);  run("2-torus", A, p, r, 2)
    A, p, r = sphere_graph(2000);    run("2-sphere", A, p, r, 2)
    A, p, r = torus_graph(3000, 3);  run("3-torus", A, p, r, 3)
    print("\nPASS criterion: eff_rank tracks ball-growth and both land near "
          "d_true (within ~0.3).")
