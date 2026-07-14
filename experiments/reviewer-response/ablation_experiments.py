"""Weighting ablation (referee comment 1) + baseline fixes.

Which angle carries the geometry? The direction depends on the per-mode weight:
  plain      w_k = 1                 (Laplacian eigenmaps)
  commute    w_k = 1/sqrt(lambda_k)  (ours; global point signature)
  diffusion  w_k = exp(-lambda_k t)  (diffusion maps, two t)
  deg_div    commute then / sqrt(k_i) (diffusion-map degree convention)
  deg_mul    commute then * sqrt(k_i) (opposite correction)
We row-normalize each and score angle-rho vs graph geodesics, on the 2-torus
(d=2) and a 3-torus RGG (d=3). Also: euclid/Isomap matched to dimension m (not
fixed 2), and the random-mode control averaged over draws with a spread.

Local, CPU-only, reuses the frozen builders. JSON between markers.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from rung0_validate import torus_graph, _norm_laplacian_eigs


def _rho(d, g):
    if np.ptp(d) < 1e-12:
        return 0.0
    r = spearmanr(d, g).statistic
    return abs(r) if np.isfinite(r) else 0.0


def _angle_rho(coords, anc, iu, g):
    Y = coords[anc]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    d = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(1))
    return _rho(d, g)


def _bilip(coords, anc, iu, g, r0=8):
    Y = coords[anc]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    d = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(1))
    loc = g <= r0
    if loc.sum() < 50:
        return None
    ratio = d[loc] / g[loc]
    ratio = ratio[ratio > 0]
    return round(float(np.percentile(ratio, 97.5) / np.percentile(ratio, 2.5)), 2)


def substrate(name, A, seed=7):
    n = A.shape[0]
    w, V = _norm_laplacian_eigs(A)
    deg = np.asarray(A.sum(1)).ravel()
    rng = np.random.default_rng(seed)
    anc = rng.choice(n, size=200, replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]

    out = {"substrate": name, "n": int(n)}
    for m in (10, 20):
        lam = w[1:1 + m]
        Vt = V[:, 1:1 + m]
        sk = np.sqrt(np.maximum(deg, 1e-12))[:, None]
        weightings = {
            "plain":      Vt,
            "commute":    Vt / np.sqrt(np.maximum(lam, 1e-12)),
            "diffusion_t1": Vt * np.exp(-lam * (1.0 / lam[-1])),
            "diffusion_t3": Vt * np.exp(-lam * (3.0 / lam[-1])),
            "deg_div":    (Vt / np.sqrt(np.maximum(lam, 1e-12))) / sk,
            "deg_mul":    (Vt / np.sqrt(np.maximum(lam, 1e-12))) * sk,
        }
        rec = {}
        for wname, coords in weightings.items():
            rec[wname] = dict(angle_rho=round(_angle_rho(coords, anc, iu, g), 3),
                              bilip=_bilip(coords, anc, iu, g))
        out[f"m{m}"] = rec

    # euclid / Isomap: classical MDS of geodesics to dim 2 vs dim m
    k = len(anc)
    J = np.eye(k) - np.ones((k, k)) / k
    B = -0.5 * J @ (G ** 2) @ J
    B = 0.5 * (B + B.T)
    ew, ev = np.linalg.eigh(B)
    order = ew.argsort()[::-1]
    euc = {}
    for dim in (2, 10, 20):
        top = order[:dim]
        X = ev[:, top] * np.sqrt(np.maximum(ew[top], 0.0))
        de = np.sqrt(((X[iu[0]] - X[iu[1]]) ** 2).sum(1))
        euc[f"isomap_dim{dim}"] = round(_rho(de, g), 3)
    out["euclid_isomap"] = euc

    # random-mode control averaged over draws
    rr = []
    for s in range(10):
        r2 = np.random.default_rng(1000 + s)
        ridx = r2.choice(np.arange(1, V.shape[1]), size=20, replace=False)
        coords = V[:, ridx] / np.sqrt(np.maximum(w[ridx], 1e-12))
        rr.append(_angle_rho(coords, anc, iu, g))
    out["random_mode_control"] = dict(mean=round(float(np.mean(rr)), 3),
                                      std=round(float(np.std(rr)), 3),
                                      max=round(float(np.max(rr)), 3), n_draws=10)
    print(f"[{name}] done", file=sys.stderr)
    return out


if __name__ == "__main__":
    res = {}
    res["torus_d2"] = substrate("2-torus", torus_graph(2000, 2)[0])
    res["torus_d3"] = substrate("3-torus", torus_graph(3000, 3)[0])
    with open("ablation_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("###RESULTS_JSON_START###"); print(json.dumps(res)); print("###RESULTS_JSON_END###")
