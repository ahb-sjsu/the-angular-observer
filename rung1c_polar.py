"""
Rung 1c — The PolarQuant probe.

PolarQuant's principle: split a vector into magnitude (radius) + direction
(angle); the DIRECTION is the scale-invariant, transferable, meaning-bearing
part. Ng-Jordan-Weiss spectral clustering independently does the same move
(row-normalize the eigen-embedding to the unit sphere).

Sharp prediction for the observer bridge: in the low-mode Laplacian embedding,
the ANGLE should carry the graph geometry and the RADIUS should be throwaway.

Test on the clean d=2 torus (where geometry is real), comparing geodesic
preservation of three decompositions of the SAME low-mode embedding:
    full      = magnitude x direction   (commute-time scaled)
    angle     = direction only          (rows normalized to unit sphere)  <- PolarQuant
    magnitude = radius only             (||row||, a 1-D scalar)
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from rung0_validate import torus_graph, _norm_laplacian_eigs, RNG
from rung1_proxy import commute_time_embed


def preservation(A, w, V, m, anchors):
    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    g = G[iu]
    modes = np.arange(1, 1 + m)
    Y = commute_time_embed(w, V, modes)[anchors]          # full: mag x dir

    r = np.linalg.norm(Y, axis=1, keepdims=True)
    ang = Y / np.maximum(r, 1e-12)                         # direction only

    def corr(Z):
        d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[iu]
        return abs(spearmanr(d, g).statistic)

    d_full = corr(Y)
    d_ang = corr(ang)
    d_mag = abs(spearmanr(abs(r[:, 0][:, None] - r[:, 0][None, :])[iu], g).statistic)
    return d_full, d_ang, d_mag


if __name__ == "__main__":
    print("Rung 1c — PolarQuant probe on the clean d=2 torus\n")
    A, _, _ = torus_graph(2500, 2)
    w, V = _norm_laplacian_eigs(A)
    anchors = RNG.choice(A.shape[0], size=300, replace=False)
    print(f"    geodesic preservation |rho|")
    print(f"    {'m':>4} | {'full (mag x dir)':>16} | {'ANGLE only':>11} | "
          f"{'magnitude only':>14}")
    for m in (5, 10, 20, 40):
        f, a, mg = preservation(A, w, V, m, anchors)
        print(f"    {m:4d} | {f:16.3f} | {a:11.3f} | {mg:14.3f}")
    print("\n    PolarQuant prediction: ANGLE ~ full, magnitude ~ throwaway.")
