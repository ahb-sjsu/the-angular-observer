"""
Known Wolfram-model / Gorard hypergraph rewriting rules, translated into our
rewrite() format, grown to a few thousand hyperedges, clique-expanded, and
measured with the three emergent-dimension estimators + the angle-only observer
metric (angle_rho).

SOURCES / CITATIONS
-------------------
[R_SR]  SetReplace canonical rule (VERIFIED from the repo README front-page demo,
        https://github.com/maxitg/SetReplace README):
          {{v1,v2,v3},{v2,v4,v5}} -> Module[{v6}, {{v5,v6,v1},{v6,v4,v2},{v4,v5,v3}}]
        initial condition {{1,2,3},{2,4,5},{4,6,7}}.
        This is THE most-shown Wolfram-model rule. README makes no explicit
        dimension claim; commonly grows an ~2-2.5D surface-like structure.

[R_2D], [R_3D], [R_frac]  From Stephen Wolfram's April 2020 announcement
        "Finally We May Have a Path to the Fundamental Theory of Physics"
        (writings.stephenwolfram.com/2020/04/...). Node labels below were read
        back by an automated fetch of that page and were NOT independently
        cross-verified string-for-string, so treat the LABELS as approximate;
        we rely on the MEASURED dimension, not the page's claim.
          R_2D   {{1,2,2},{3,1,4}} -> {{2,5,2},{2,3,5},{4,5,5}}   claim: -> 2D
          R_3D   {{1,1,2},{3,4,1}} -> {{4,4,3},{5,4,5},{5,2,1}}   claim: -> 3D
          R_frac {{1,2,3}} -> {{1,4,6},{2,5,4},{3,6,5}}           claim: ~1.58D (Sierpinski)

We do NOT assert a dimension a priori. Torus controls (rung0): 2-torus ->
ball1.90/spec1.95/eff2.11 (spread~0.09); 3-torus -> ball2.73/spec2.95/eff3.15.
A "clean manifold" means the three estimators AGREE (low spread) near an integer.
"""
import sys
import numpy as np
from scipy.sparse.csgraph import shortest_path
from rung1b_rewriter import rewrite, largest_component
from rung0_validate import (dim_ball_growth, dim_spectral, dim_effective_rank,
                            _norm_laplacian_eigs, RNG)
from arity3 import hyper_to_csr
from search_harness import angle_rho

N_CAP = 2200          # crop below this before the dense O(N^3) eigendecomposition


def crop_to_ball(A, target=N_CAP):
    """Keep a connected BFS ball of ~target nodes (a local manifold patch).

    Dimension is a LOCAL property, so measuring on a geodesic ball is faithful
    and keeps the dense Laplacian solve tractable on exponentially-growing rules.
    """
    n = A.shape[0]
    if n <= target:
        return A
    src = RNG.integers(n)
    d = shortest_path(A, method="D", unweighted=True, indices=[src])[0]
    order = np.argsort(d)                       # nearest-first by hop distance
    keep = np.sort(order[:target])
    return largest_component(A[keep][:, keep])

# ---- rules: (lhs, rhs, seed_candidates, claimed_dim, source) ----------------
S_COLLAPSE = [(0, 0, 0), (0, 0, 0)]           # Wolfram's {{1,1,1},{1,1,1}} style init
S_TET = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

RULES = {
    "R_SR (SetReplace canonical)": dict(
        lhs=[("v1", "v2", "v3"), ("v2", "v4", "v5")],
        rhs=[("v5", "v6", "v1"), ("v6", "v4", "v2"), ("v4", "v5", "v3")],
        seeds={"README": [(1, 2, 3), (2, 4, 5), (4, 6, 7)], "tet": S_TET},
        claim="~2-2.5D (no explicit claim)", src="[R_SR] SetReplace README (verified)"),
    "R_2D (announcement)": dict(
        lhs=[("1", "2", "2"), ("3", "1", "4")],
        rhs=[("2", "5", "2"), ("2", "3", "5"), ("4", "5", "5")],
        seeds={"collapse": S_COLLAPSE, "tet": S_TET},
        claim="2D", src="[R_2D] Wolfram Apr-2020 (labels unverified)"),
    "R_3D (announcement)": dict(
        lhs=[("1", "1", "2"), ("3", "4", "1")],
        rhs=[("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")],
        seeds={"collapse": S_COLLAPSE, "tet": S_TET},
        claim="3D", src="[R_3D] Wolfram Apr-2020 (labels unverified)"),
    "R_frac (Sierpinski)": dict(
        lhs=[("1", "2", "3")],
        rhs=[("1", "4", "6"), ("2", "5", "4"), ("3", "6", "5")],
        seeds={"tri": [(0, 1, 2)], "tet": S_TET},
        claim="~1.58D", src="[R_frac] Wolfram Apr-2020 (labels unverified)"),
}


def grow_and_measure(lhs, rhs, seed, max_edges=3000, max_gen=500):
    tris, nid = rewrite(lhs, rhs, seed, max_edges=max_edges, max_gen=max_gen, seed=1)
    A = largest_component(hyper_to_csr(tris, nid))
    n_full = A.shape[0]
    if n_full < 200:
        return dict(ok=False, N=n_full, tris=len(tris))
    A = crop_to_ball(A)                          # local patch, keeps eigh tractable
    n = A.shape[0]
    w, V = _norm_laplacian_eigs(A)
    db, ds, de = dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)
    spread = float(np.std([db, ds, de]))
    deg = float(np.asarray(A.sum(1)).ravel().mean())
    arho = angle_rho(A, w, V, rng=RNG)
    return dict(ok=True, N=n, N_full=n_full, tris=len(tris), deg=deg,
                ball=db, spec=ds, eff=de, spread=spread, angle_rho=arho)


if __name__ == "__main__":
    print("Known Wolfram/Gorard rules -> emergent dimension + observer metric")
    print("controls (rung0): 2-torus ball1.90/spec1.95/eff2.11 spread~0.09 | "
          "3-torus ball2.73/spec2.95/eff3.15\n")
    print(f"{'rule':30s} {'seed':9s} {'Nful':>6} {'N':>5} {'tris':>5} {'deg':>5} "
          f"{'ball':>5} {'spec':>5} {'eff':>5} {'sprd':>5} {'a_rho':>6}  claim")
    for name, R in RULES.items():
        for sname, seed in R["seeds"].items():
            try:
                m = grow_and_measure(R["lhs"], R["rhs"], seed)
            except Exception as e:
                print(f"{name:30s} {sname:9s}  ERROR {type(e).__name__}: {e}")
                sys.stdout.flush()
                continue
            if not m["ok"]:
                print(f"{name:30s} {sname:9s} {'':>6} {m['N']:>5} {m['tris']:>5}  "
                      f"(stalled, <200 nodes)")
                sys.stdout.flush()
                continue
            print(f"{name:30s} {sname:9s} {m['N_full']:>6} {m['N']:>5} {m['tris']:>5} "
                  f"{m['deg']:>5.1f} {m['ball']:>5.2f} {m['spec']:>5.2f} {m['eff']:>5.2f} "
                  f"{m['spread']:>5.2f} {m['angle_rho']:>6.3f}  {R['claim']}")
            sys.stdout.flush()
        print(f"    src: {R['src']}")
        sys.stdout.flush()
    print("\nangle_rho reference: clean 2-torus ~0.93, tangle ~0.40.")
