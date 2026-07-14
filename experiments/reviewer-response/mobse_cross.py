"""MoBSE cross-program experiment: is the moral embedding space a manifold?

The reviewer's suggested cross-experiment for "Keep the Angle": take the same
angle-only low-mode Laplacian diagnostic that scores ~0.93 on a clean Riemannian
substrate and ~0 on a tangle, and point it at a *moral* embedding space.

Substrate: BGE-M3 (1024-d) embeddings of moral/ethical text chunks, pulled from
the Atlas `ethics_chunks` table (pgvector, precomputed). Stratified across
traditions so the geometry is not one corpus's stylistic manifold. A symmetric
kNN graph on the unit-normalized embeddings is the moral substrate; its
unweighted geodesics are the ground-truth metric.

We report the full diagnostic battery -- angle / higher-mode / magnitude /
random-mode rho, the three dimension estimators and their spread, and the
two-factor small-world screen (C, L, sigma, omega) -- against two destroyed
controls (per-feature shuffle; isotropic Gaussian). The question is not "does
moral space embed at rho=0.93" but *where on the clean-manifold-to-tangle axis
the moral embedding space actually falls*, measured by the same instrument.

CPU-only (embeddings are precomputed); no GPU, no model load. Run under the
Atlas venv. Emits one JSON blob between the markers.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import networkx as nx
import psycopg2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

N_TOTAL = int(os.environ.get("MOBSE_N", "2400"))
KNN = 10
M_MODES = 20
N_ANCHOR = 200
TRADITIONS = ["jewish", "greco_roman", "hindu", "american_advice", "buddhist", "vedic"]
RNG = np.random.default_rng(20260714)


# ------------------------------------------------------------------ data pull
def fetch_embeddings(n_total=N_TOTAL):
    per = n_total // len(TRADITIONS)
    conn = psycopg2.connect(dbname="atlas", user="claude")
    cur = conn.cursor()
    vecs, meta = [], []
    for trad in TRADITIONS:
        cur.execute(
            "select embedding::text, tradition from ethics_chunks "
            "where embedding is not null and length(content) > 200 and tradition = %s "
            "order by md5(id::text) limit %s", (trad, per))
        for txt, tr in cur.fetchall():
            v = np.fromstring(txt.strip()[1:-1], sep=",")
            if v.size == 1024 and np.all(np.isfinite(v)):
                vecs.append(v)
                meta.append(tr)
    cur.close(); conn.close()
    X = np.asarray(vecs, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X, meta


# ------------------------------------------------------------------ graph
def knn_graph(X, k=KNN):
    tree = cKDTree(X)
    _, nb = tree.query(X, k=k + 1)              # incl. self
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = nb[:, 1:].reshape(-1)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.setdiag(0); A.eliminate_zeros()
    ncc, lab = connected_components(A, directed=False)
    if ncc > 1:
        keep = np.flatnonzero(lab == np.bincount(lab).argmax())
        A = A[keep][:, keep]
    return A


# ------------------------------------------------------------------ estimators
def norm_lap_eigs(A, k_band=64):
    deg = np.asarray(A.sum(1)).ravel()
    dis = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(dis)
    Nadj = (D @ A @ D).tocsr()
    k = min(k_band + 1, A.shape[0] - 2)
    ncv = min(A.shape[0] - 1, max(2 * k + 1, 60))
    mu, V = eigsh(Nadj, k=k, which="LA", ncv=ncv, maxiter=5000, tol=0)
    order = np.argsort(mu)[::-1]
    mu, V = mu[order], V[:, order]
    lam = 1.0 - mu
    return lam, V


def dim_ball(A, n_src=60):
    n = A.shape[0]
    src = RNG.choice(n, size=min(n_src, n), replace=False)
    D = shortest_path(A, method="D", unweighted=True, indices=src)
    maxhop = int(np.nanmax(D[np.isfinite(D)]))
    rs = np.arange(1, maxhop + 1)
    Nr = np.array([[np.sum(row <= r) for r in rs] for row in D], float).mean(0)
    lo, hi = 1, np.searchsorted(Nr, 0.6 * n)
    hi = max(hi, lo + 3)
    return float(np.polyfit(np.log(rs[lo:hi]), np.log(Nr[lo:hi]), 1)[0])


def dim_spectral(lam):
    ts = np.logspace(0.3, 2.2, 40)
    P = np.array([np.mean(np.exp(-lam * t)) for t in ts])
    a, b = int(0.2 * len(ts)), int(0.8 * len(ts))
    return float(-2 * np.polyfit(np.log(ts[a:b]), np.log(P[a:b]), 1)[0])


def dim_effrank(A, lam, V, m_modes=40, n_probe=200, k_hop=2):
    n = A.shape[0]
    coords = V[:, 1:1 + m_modes]
    D = shortest_path(A, method="D", unweighted=True,
                      indices=RNG.choice(n, size=min(n_probe, n), replace=False))
    prs = []
    for row in D:
        nb = np.where(row <= k_hop)[0]
        if len(nb) < 8:
            continue
        Xl = coords[nb] - coords[nb].mean(0)
        s = np.linalg.svd(Xl, compute_uv=False) ** 2
        s = s[s > 1e-12]
        prs.append((s.sum() ** 2) / (s ** 2).sum())
    return float(np.mean(prs)) if prs else float("nan")


# ------------------------------------------------------------------ rho battery
def _rho(d, g):
    if np.ptp(d) < 1e-12:
        return 0.0
    r = spearmanr(d, g).statistic
    return abs(r) if np.isfinite(r) else 0.0


def rho_battery(A, lam, V, m=M_MODES, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    anc = rng.choice(n, size=min(N_ANCHOR, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]
    ok = np.isfinite(g)
    scaled = V / np.sqrt(np.maximum(lam, 1e-9))

    def pd(Y):
        Ya = Y[anc]
        return np.sqrt(((Ya[iu[0]] - Ya[iu[1]]) ** 2).sum(-1))[ok]

    low = scaled[:, 1:1 + m]
    lown = low / np.maximum(np.linalg.norm(low, axis=1, keepdims=True), 1e-12)
    hi = scaled[:, -m:]
    hin = hi / np.maximum(np.linalg.norm(hi, axis=1, keepdims=True), 1e-12)
    rad = np.linalg.norm(low, axis=1, keepdims=True)
    ridx = rng.choice(np.arange(1, V.shape[1]), size=m, replace=False)
    rnd = scaled[:, ridx]
    rndn = rnd / np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
    g_ok = g[ok]
    return dict(n_pairs=int(ok.sum()),
                angle=round(_rho(pd(lown), g_ok), 3),
                higher=round(_rho(pd(hin), g_ok), 3),
                magnitude=round(_rho(pd(rad), g_ok), 3),
                random=round(_rho(pd(rndn), g_ok), 3))


def small_world(A, seed=0):
    G = nx.from_scipy_sparse_array(A)
    G = nx.convert_node_labels_to_integers(
        G.subgraph(max(nx.connected_components(G), key=len)).copy())
    n, m = G.number_of_nodes(), G.number_of_edges()
    kbar = 2.0 * m / n

    def sampL(H, src=150, s=0):
        rng = np.random.default_rng(s)
        nodes = list(H.nodes())
        sr = rng.choice(len(nodes), size=min(src, len(nodes)), replace=False)
        tot = cnt = 0
        for i in sr:
            for _, dist in nx.single_source_shortest_path_length(H, nodes[i]).items():
                if dist > 0:
                    tot += dist; cnt += 1
        return tot / max(cnt, 1)

    C = nx.average_clustering(G)
    L = sampL(G, s=seed)
    Cr, Lr = [], []
    for j in range(3):
        R = nx.gnm_random_graph(n, m, seed=seed + j)
        R = nx.convert_node_labels_to_integers(
            R.subgraph(max(nx.connected_components(R), key=len)).copy())
        Cr.append(nx.average_clustering(R)); Lr.append(sampL(R, 100, seed + j))
    C_rand, L_rand = float(np.mean(Cr)), float(np.mean(Lr))
    k = max(2, int(round(kbar)))
    Lat = nx.watts_strogatz_graph(n, k if k % 2 == 0 else k + 1, 0.0, seed=seed)
    C_latt = nx.average_clustering(Lat)
    sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L > 0 else float("nan")
    omega = (L_rand / L) - (C / C_latt) if C_latt > 0 and L > 0 else float("nan")
    return dict(kbar=round(kbar, 2), C=round(C, 4), L=round(L, 3),
                C_rand=round(C_rand, 4), L_rand=round(L_rand, 3),
                sigma=round(sigma, 3), omega=round(omega, 3))


def full_measure(name, X, seed=0):
    A = knn_graph(X)
    lam, V = norm_lap_eigs(A)
    dims = [dim_ball(A), dim_spectral(lam), dim_effrank(A, lam, V)]
    rho = rho_battery(A, lam, V, seed=seed)
    sw = small_world(A, seed=seed)
    return dict(name=name, n=int(A.shape[0]),
                dim_ball=round(dims[0], 3), dim_spectral=round(dims[1], 3),
                dim_effrank=round(dims[2], 3),
                dim_mean=round(float(np.mean(dims)), 3),
                dim_spread=round(float(np.ptp(dims)), 3), **rho, **sw)


if __name__ == "__main__":
    X, meta = fetch_embeddings()
    from collections import Counter
    print(f"pulled {X.shape[0]} moral items, {X.shape[1]}-d; "
          f"traditions={dict(Counter(meta))}", file=sys.stderr)

    result = {"n_items": int(X.shape[0]), "ambient_dim": int(X.shape[1]),
              "traditions": dict(Counter(meta)), "knn": KNN}

    result["moral"] = full_measure("moral-embedding", X, seed=1)
    print("[moral] done", file=sys.stderr)

    # control 1: per-feature shuffle (destroys cross-feature manifold, keeps marginals)
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        Xs[:, j] = Xs[RNG.permutation(Xs.shape[0]), j]
    Xs = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    result["shuffle_control"] = full_measure("shuffle", Xs, seed=2)
    print("[shuffle] done", file=sys.stderr)

    # control 2: isotropic Gaussian in ambient dim (pure tangle)
    Xg = RNG.normal(size=X.shape)
    Xg = Xg / np.maximum(np.linalg.norm(Xg, axis=1, keepdims=True), 1e-12)
    result["isotropic_control"] = full_measure("isotropic", Xg, seed=3)
    print("[isotropic] done", file=sys.stderr)

    print("###RESULTS_JSON_START###")
    print(json.dumps(result))
    print("###RESULTS_JSON_END###")
