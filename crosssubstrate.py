"""
Task #4 — Is angle-only low-eigenvalue Laplacian coarse-graining a UNIVERSAL
observer basis, or a Wolfram-hypergraph-specific trick?

We take the SAME observer metric that scored ~0.93 on a clean torus and ~0.40 on
a small-world tangle (search_harness.angle_rho: geodesic preservation of the
unit-normalized, angle-only, lowest-m Laplacian eigenmode embedding) and apply it
to THREE non-Wolfram emergent-geometry substrates:

  (1) CAUSAL-SET SPRINKLING  — Poisson points in a 1+1D Minkowski diamond,
      each linked to its k nearest causal-PAST neighbours (respects light cones).
      Genuinely Lorentzian emergent geometry, intended intrinsic dim = 2.
  (2) KING-MOVE LATTICE      — 8-neighbour 2D grid (a CA / cellular substrate
      with genuine flat 2D geometry), intended dim = 2. Optional small-world
      rewiring probes robustness.
  (3) EMBEDDING TRAJECTORY   — a smooth 2D manifold (swiss roll) carried into a
      higher-d ambient space with noise, kNN graph. This is the geometry-series
      analogue: embedding a corpus line-by-line onto a low-d manifold. Dim = 2.

For each substrate we report, on the SAME anchor set / geodesics:
  angle_rho   = the observer metric (lowest-m modes, angle-only)          claim
  random_rho  = identical pipeline but m RANDOMLY-chosen modes            control (~0)
  euclid_rho  = classical-MDS 2D Euclidean embedding of the geodesics     baseline (Isomap)
plus ball-growth and spectral dimension (verified BEFORE interpreting rho).

HONESTY: random_rho MUST fail (~0) for the result to mean anything. A substrate
that does not show the effect is reported plainly.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr

from rung0_validate import _norm_laplacian_eigs, dim_ball_growth, dim_spectral, RNG
from search_harness import angle_rho as _angle_rho_ref   # cross-check reference


# --------------------------------------------------------------------- utilities
def _sym_binary_csr(n, pairs):
    """Undirected, deduped, binary CSR adjacency from a list of (i,j) pairs."""
    pairs = [(a, b) for a, b in pairs if a != b]
    if not pairs:
        raise ValueError("no edges")
    i = np.array([a for a, b in pairs] + [b for a, b in pairs])
    j = np.array([b for a, b in pairs] + [a for a, b in pairs])
    A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def _largest_cc(A):
    ncc, lbl = connected_components(A, directed=False)
    if ncc == 1:
        return A
    keep = np.where(lbl == np.bincount(lbl).argmax())[0]
    return A[keep][:, keep]


# ------------------------------------------------------------- observer + controls
def observer_rhos(A, w, V, m=20, n_anchor=150, rng=None):
    """
    Compute angle_rho (claim), random_rho (control), euclid_rho (baseline) on the
    SAME anchors / geodesics so they are directly comparable.

    angle_rho  : rows of V[:,1:1+m]/sqrt(w) L2-normalised (angle only) -> pdist vs geodesic
    random_rho : identical but m modes drawn uniformly from the WHOLE spectrum
    euclid_rho : classical MDS of the geodesic matrix to 2D -> Euclid pdist vs geodesic
    """
    if rng is None:
        rng = RNG
    n = A.shape[0]
    anchors = rng.choice(n, size=min(n_anchor, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    g = G[iu]
    if np.ptp(g) < 1e-9 or not np.all(np.isfinite(g)):
        return dict(angle=0.0, random=0.0, euclid=0.0)

    def _rho_from_modes(idx, angle=True):
        Y = V[:, idx] / np.sqrt(np.maximum(w[idx], 1e-9))
        Y = Y[anchors]
        if angle:
            Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
        if np.ptp(d) < 1e-9:
            return 0.0
        r = spearmanr(d, g).statistic
        return abs(r) if np.isfinite(r) else 0.0

    low_idx = np.arange(1, 1 + m)                                   # lowest m (skip 0)
    rand_idx = rng.choice(np.arange(1, n), size=m, replace=False)   # random m modes

    # classical MDS (Isomap) of the geodesic distances -> 2D Euclidean
    k = len(anchors)
    J = np.eye(k) - np.ones((k, k)) / k
    B = -0.5 * J @ (G ** 2) @ J
    B = 0.5 * (B + B.T)
    ew, ev = np.linalg.eigh(B)
    top = ew.argsort()[::-1][:2]
    coords = ev[:, top] * np.sqrt(np.maximum(ew[top], 0.0))
    de = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))[iu]
    erho = abs(spearmanr(de, g).statistic) if np.ptp(de) > 1e-9 else 0.0

    return dict(angle=_rho_from_modes(low_idx, True),
                random=_rho_from_modes(rand_idx, True),
                euclid=erho if np.isfinite(erho) else 0.0)


# ----------------------------------------------------------- substrate 1: causal set
def causal_set_graph(n=2000, k=6, seed=1):
    """
    Sprinkle n Poisson points in a 1+1D Minkowski diamond (light-cone coords
    u,v in the unit square). p precedes q iff u_p<u_q AND v_p<v_q (both future
    light-cone coords increase). Link each point to its k nearest causal-PAST
    neighbours (nearest in Minkowski proper time). Undirected graph for BFS.
    Emergent 2D Lorentzian geometry.
    """
    rng = np.random.default_rng(seed)
    uv = rng.random((n, 2))                      # light-cone coords (u, v)
    t = uv.sum(1)                                # t = u + v
    x = uv[:, 0] - uv[:, 1]                       # x = u - v
    order = np.argsort(t)                         # process in time order
    uv, t, x = uv[order], t[order], x[order]
    pairs = []
    for q in range(n):
        # causal past of q: earlier points with both light-cone coords smaller
        past = np.where((uv[:q, 0] < uv[q, 0]) & (uv[:q, 1] < uv[q, 1]))[0]
        if len(past) == 0:
            continue
        # proper time^2 = dt^2 - dx^2 (timelike, positive on the past set)
        dt = t[q] - t[past]
        dx = x[q] - x[past]
        tau2 = dt * dt - dx * dx
        nn = past[np.argsort(tau2)[:k]]          # k nearest in proper time
        pairs.extend((int(q), int(p)) for p in nn)
    return _largest_cc(_sym_binary_csr(n, pairs))


# --------------------------------------------------------- substrate 2: king lattice
def king_lattice_graph(side=45, rewire_p=0.0, seed=2):
    """
    8-neighbour (king-move) 2D grid graph on side*side cells. Genuine flat 2D
    geometry. rewire_p randomly reconnects that fraction of edges (small-world)
    to probe robustness of the emergent geometry.
    """
    rng = np.random.default_rng(seed)
    n = side * side
    idx = lambda r, c: r * side + c
    pairs = []
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(side):
        for c in range(side):
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    a, b = idx(r, c), idx(nr, nc)
                    if a < b:
                        pairs.append((a, b))
    if rewire_p > 0:
        pairs = [p for p in pairs]
        for i in range(len(pairs)):
            if rng.random() < rewire_p:
                a = pairs[i][0]
                b = int(rng.integers(n))
                if a != b:
                    pairs[i] = (a, b)
    return _largest_cc(_sym_binary_csr(n, pairs))


# ----------------------------------------------- substrate 3: embedding trajectory
def embedding_trajectory_graph(n=2000, ambient=12, k=10, noise=0.03, seed=3):
    """
    A smooth 2D manifold (swiss roll) carried into a higher-d ambient space via a
    random orthogonal map, plus isotropic noise; kNN graph over the ambient
    points. Intrinsic dim = 2. Analogue of embedding a text corpus line-by-line
    onto a low-d semantic manifold and connecting nearest lines.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)     # swiss-roll angle (intrinsic 1)
    v = rng.uniform(0.0, 20.0, n)                    # height (intrinsic 2)
    roll = np.stack([u * np.cos(u), v, u * np.sin(u)], axis=1)   # 2D manifold in R^3
    roll = (roll - roll.mean(0)) / roll.std(0)
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, 3)))           # R^3 -> R^ambient
    X = roll @ Q.T + noise * rng.normal(size=(n, ambient))
    tree = cKDTree(X)
    _, nb = tree.query(X, k=k + 1)                                # incl. self
    pairs = [(i, int(j)) for i in range(n) for j in nb[i, 1:]]
    return _largest_cc(_sym_binary_csr(n, pairs))


# ----------------------------------------------------------------------------- run
def evaluate(name, A, intended_dim, rng):
    w, V = _norm_laplacian_eigs(A)
    d_ball = dim_ball_growth(A)
    d_spec = dim_spectral(w)
    rhos = observer_rhos(A, w, V, rng=rng)
    ref = _angle_rho_ref(A, w, V, rng=rng)          # cross-check vs shared harness
    deg = float(np.asarray(A.sum(1)).ravel().mean())
    print(f"{name:<22} N={A.shape[0]:5d} <deg>={deg:5.1f} | "
          f"dim_intended={intended_dim}  ball={d_ball:4.2f}  spec={d_spec:4.2f} | "
          f"angle_rho={rhos['angle']:.3f}  random_rho={rhos['random']:.3f}  "
          f"euclid_rho={rhos['euclid']:.3f}   (ref angle={ref:.3f})")
    return dict(name=name, N=A.shape[0], intended=intended_dim,
                ball=d_ball, spec=d_spec, **rhos)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    print("Task #4 — angle-only Laplacian observer basis on non-Wolfram substrates")
    print("reference: clean torus angle_rho~0.93, tangle~0.40; random_rho SHOULD be ~0\n")

    rows = []
    print("--- baseline sanity: clean 2-torus (Wolfram-adjacent ground truth) ---")
    from rung0_validate import torus_graph
    At, _, _ = torus_graph(2000, 2)
    rows.append(evaluate("0. torus-2D (control)", At, 2, rng))

    print("\n--- substrate 1: causal-set sprinkling (1+1D Minkowski) ---")
    A1 = causal_set_graph(n=2000, k=6)
    rows.append(evaluate("1. causal-set 1+1D", A1, 2, rng))

    print("\n--- substrate 2: king-move lattice (cellular / grid) ---")
    A2 = king_lattice_graph(side=45, rewire_p=0.0)
    rows.append(evaluate("2. king-lattice 2D", A2, 2, rng))
    A2b = king_lattice_graph(side=45, rewire_p=0.03)
    rows.append(evaluate("2b king-lattice sw p=.03", A2b, 2, rng))

    print("\n--- substrate 3: embedding trajectory (swiss roll in R^12) ---")
    A3 = embedding_trajectory_graph(n=2000, ambient=12, k=10)
    rows.append(evaluate("3. embed-traj 2D", A3, 2, rng))

    print("\n================================ SUMMARY ================================")
    print(f"{'substrate':<24}{'intend':>7}{'ball':>6}{'spec':>6}"
          f"{'angle':>7}{'random':>8}{'euclid':>8}")
    for r in rows:
        print(f"{r['name']:<24}{r['intended']:>7}{r['ball']:>6.2f}{r['spec']:>6.2f}"
              f"{r['angle']:>7.3f}{r['random']:>8.3f}{r['euclid']:>8.3f}")
    print("\nverdict rule: observer principle holds on a substrate iff "
          "angle_rho is high (>~0.7) AND random_rho ~0.")
