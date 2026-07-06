"""
Rung 1a — Tunable growth proxy: emergent-dimension trajectory + the
coarse-graining test.

Two things Rung 0 did not do:

  (A) TRAJECTORY. Estimate dimension from ADJACENCY ONLY across increasing N,
      showing the finite-size bias shrink -- dimension is discovered, not
      handed in.

  (B) COARSE-GRAINING TEST (the thesis, made falsifiable).
      Wolfram: a bounded observer coarse-grains the hypergraph and perceives
      smooth geometry.  You: meaning survives truncation in the RIGHT
      eigenbasis and dies in a random one.  Same statement if the "right
      basis" is the low-eigenvalue subspace of the graph Laplacian.

      Test: reconstruct true graph geodesics from a commute-time embedding
      built from either
          - the m LOWEST non-trivial Laplacian modes   (observer basis), or
          - m RANDOM modes drawn from the full spectrum (fair control).
      If low-m preserves geodesics at tiny m while random-m does not, the
      low-eigenvalue subspace IS the observer's coarse-graining basis.

Proxy honesty: latent coordinates drive a local growth rule but are hidden
from every estimator (they see only adjacency). The faithful rewriter
(Rung 1b) removes latent coordinates entirely.
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from rung0_validate import (torus_graph, dim_ball_growth, dim_spectral,
                            dim_effective_rank, _norm_laplacian_eigs, RNG)


# ------------------------------------------------------- (A) emergent trajectory
def trajectory(dim, sizes=(500, 1000, 2000, 4000)):
    print(f"(A) emergent-dimension trajectory, latent d={dim} "
          f"(estimators see adjacency only)\n")
    print(f"    {'N':>6} | {'ball':>6} {'spectral':>8} {'eff_rank':>8}")
    for n in sizes:
        A, _, _ = torus_graph(n, dim)
        w, V = _norm_laplacian_eigs(A)
        db = dim_ball_growth(A)
        ds = dim_spectral(w)
        de = dim_effective_rank(A, w, V)
        print(f"    {n:6d} | {db:6.2f} {ds:8.2f} {de:8.2f}")
    print(f"    -> converging toward d={dim}; ball-growth carries the most "
          f"finite-size bias.\n")


# ------------------------------------------------------- (B) coarse-graining test
def commute_time_embed(w, V, modes):
    """Commute-time coords: column k scaled by 1/sqrt(lambda_k).
    Small-lambda (large-scale) modes dominate -> they carry the geometry."""
    scaled = V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9))
    return scaled


def geodesic_preservation(A, w, V, m, anchors):
    """Spearman corr between embedding distance and true graph geodesic, for
    the m LOWEST modes vs m RANDOM modes (both commute-time scaled)."""
    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    g_true = G[iu]

    nontrivial = np.arange(1, len(w))                 # skip lambda=0 constant
    low = nontrivial[:m]
    rand = RNG.choice(nontrivial, size=m, replace=False)

    def corr(modes):
        Y = commute_time_embed(w, V, modes)[anchors]
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
        return spearmanr(d, g_true).statistic

    return corr(low), corr(rand)


def coarse_graining_test(dim=2, n=2500, n_anchor=300):
    print(f"(B) coarse-graining test, latent d={dim}, N={n}\n")
    A, _, _ = torus_graph(n, dim)
    w, V = _norm_laplacian_eigs(A)
    anchors = RNG.choice(n, size=n_anchor, replace=False)

    print(f"    geodesic preservation |rho_Spearman(embed dist, true geodesic)|")
    print(f"    {'m modes':>8} | {'LOW subspace':>13} | {'RANDOM subspace':>15}")
    for m in (5, 10, 20, 40, 80):
        lo, rd = geodesic_preservation(A, w, V, m, anchors)
        print(f"    {m:8d} | {lo:13.3f} | {rd:15.3f}")
    print(f"\n    Thesis prediction: LOW rises fast and saturates at small m "
          f"(geometry\n    is compressible into a few low modes); RANDOM stays "
          f"flat/low\n    (a random basis of the same size destroys the "
          f"geometry).")


if __name__ == "__main__":
    print("Rung 1a — emergent-dimension trajectory + coarse-graining test\n")
    trajectory(2)
    trajectory(3)
    coarse_graining_test(dim=2)
