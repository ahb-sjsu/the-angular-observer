"""Baseline bake-off (referee comment 6): the angle vs named competitors.

All distances are computed on the SAME graph and scored by Spearman rank
correlation with unweighted graph geodesics, on the same anchor set. Spectral
distances use the symmetric-normalized Laplacian L = I - D^-1/2 A D^-1/2 with
eigenpairs (lam_k, v_k), k>=1 (trivial mode dropped).

  angle (ours)     unit-normalized lowest-m commute embedding, chordal distance
  commute/resist   full: sum_k (v_k(i)-v_k(j))^2 / lam_k              (degenerate)
  biharmonic       sum_k (v_k(i)-v_k(j))^2 / lam_k^2   (Lipman-Rustamov-Funkhouser)
  diffusion_t      sum_k e^{-2 lam_k t} (v_k(i)-v_k(j))^2, tuned t   (Coifman-Lafon)
  amplified_commute  vLRH degeneracy correction  [formula filled from primary source]
  isomap_dim_m     classical MDS of geodesics to dim m  (given the geodesics)

JSON between markers.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from rung0_validate import torus_graph, _norm_laplacian_eigs

M = 20


def amplified_commute(A, deg, Vt, lam, anc, iu, ok, g):
    """Amplified commute distance (von Luxburg-Radl-Hein, NIPS 2010, sec 4).

    C_amp(i,j) = R_ij - 1/d_i - 1/d_j + 2 w_ij/(d_i d_j)   (no self-loops)
    with R_ij the raw effective resistance from the UNNORMALIZED Laplacian
    L = D - W. This subtracts the degenerate 1/d_i + 1/d_j term (the whole
    reason full commute collapses) and adds the Euclidean-correction term.
    Distance scored = sqrt(max(C_amp, 0)) on the anchor pairs.
    """
    n = A.shape[0]
    L = np.diag(deg) - A.toarray()
    ev, U = np.linalg.eigh(L)                 # unnormalized Laplacian
    inv = np.where(ev > 1e-9, 1.0 / ev, 0.0)  # drop trivial null mode
    emb = U * np.sqrt(inv)                     # resistance embedding: R_ij = ||emb_i-emb_j||^2
    Ea = emb[anc]
    R = ((Ea[iu[0]] - Ea[iu[1]]) ** 2).sum(1)
    di = deg[anc][iu[0]]; dj = deg[anc][iu[1]]
    Ad = A.toarray()
    wij = Ad[anc][:, anc][iu]
    camp = R - 1.0 / di - 1.0 / dj + 2.0 * wij / (di * dj)
    dist = np.sqrt(np.maximum(camp, 0.0))[ok]
    return round(_rho(dist, g), 3)


def _rho(d, g):
    if np.ptp(d) < 1e-12:
        return 0.0
    r = spearmanr(d, g).statistic
    return abs(r) if np.isfinite(r) else 0.0


def bakeoff(name, A, seed=7):
    n = A.shape[0]
    w, V = _norm_laplacian_eigs(A)
    lam = w[1:]; Vt = V[:, 1:]                      # nontrivial modes
    deg = np.asarray(A.sum(1)).ravel()
    rng = np.random.default_rng(seed)
    anc = rng.choice(n, size=min(250, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]; ok = np.isfinite(g); g = g[ok]

    def pair(coords):
        Y = coords[anc]
        return np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(1))[ok]

    res = {"substrate": name, "n": int(n)}

    # angle (ours): unit-normalized lowest-M commute embedding
    sc = Vt[:, :M] / np.sqrt(np.maximum(lam[:M], 1e-9))
    ang = sc / np.maximum(np.linalg.norm(sc, axis=1, keepdims=True), 1e-12)
    res["angle_ours"] = round(_rho(pair(ang), g), 3)

    # full commute / resistance distance (all modes)
    commute = Vt / np.sqrt(np.maximum(lam, 1e-9))
    res["commute_full"] = round(_rho(pair(commute), g), 3)

    # biharmonic: 1/lam^2 weighting (all modes)
    bih = Vt / np.maximum(lam, 1e-9)
    res["biharmonic"] = round(_rho(pair(bih), g), 3)

    # diffusion distance at several t (choose t relative to the spectral gap)
    diff = {}
    for scale in (0.5, 1.0, 2.0, 4.0):
        t = scale / lam[0]
        coords = Vt * np.exp(-lam * t)
        diff[f"t={scale}/lam1"] = round(_rho(pair(coords), g), 3)
    res["diffusion_distance"] = diff

    # Isomap (classical MDS of geodesics) at dim=M -- given the geodesics
    k = len(anc)
    J = np.eye(k) - np.ones((k, k)) / k
    B = -0.5 * J @ (G ** 2) @ J; B = 0.5 * (B + B.T)
    ew, ev = np.linalg.eigh(B)
    top = ew.argsort()[::-1][:M]
    X = ev[:, top] * np.sqrt(np.maximum(ew[top], 0.0))
    de = np.sqrt(((X[iu[0]] - X[iu[1]]) ** 2).sum(1))[ok]
    res["isomap_dim_m_given_geodesics"] = round(_rho(de, g), 3)

    # --- amplified commute distance (filled from primary source) ---
    try:
        res["amplified_commute"] = amplified_commute(A, deg, Vt, lam, anc, iu, ok, g)
    except NameError:
        res["amplified_commute"] = "pending-formula"
    return res


if __name__ == "__main__":
    out = {}
    out["torus_d2"] = bakeoff("2-torus", torus_graph(2000, 2)[0])
    out["torus_d3"] = bakeoff("3-torus", torus_graph(3000, 3)[0])
    with open("bakeoff_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("###RESULTS_JSON_START###"); print(json.dumps(out)); print("###RESULTS_JSON_END###")
    for kk, vv in out.items():
        print(f"  {kk}: {vv}", file=sys.stderr)
