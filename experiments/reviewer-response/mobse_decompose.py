"""MoBSE §5.8 follow-up: within- vs between-tradition decomposition + hubness.

Referee comment 7: row-normalization is exactly the operation that enhances
between-cluster separation, so an all-pairs angle-rho on a tradition-stratified
corpus can be inflated by between-cluster pairs (large geodesic AND large angular
distance). The within-cluster number is the one that speaks to manifold structure.
We decompose angle-rho into within-tradition and between-tradition pairs. We also
test the hubness explanation for the informative magnitude: does r_i = ||Psi_i||
track the node's kNN in-degree (hubs) after symmetrization?

CPU-only, pgvector embeddings. Emits JSON between markers.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import psycopg2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

N_TOTAL = int(os.environ.get("MOBSE_N", "2400"))
KNN, M_MODES, N_ANCHOR = 10, 20, 300
TRADITIONS = ["jewish", "greco_roman", "hindu", "american_advice", "buddhist", "vedic"]


def fetch():
    per = N_TOTAL // len(TRADITIONS)
    conn = psycopg2.connect(dbname="atlas", user="claude"); cur = conn.cursor()
    vecs, meta = [], []
    for trad in TRADITIONS:
        cur.execute("select embedding::text from ethics_chunks where embedding is not null "
                    "and length(content) > 200 and tradition = %s order by md5(id::text) limit %s",
                    (trad, per))
        for (txt,) in cur.fetchall():
            v = np.fromstring(txt.strip()[1:-1], sep=",")
            if v.size == 1024 and np.all(np.isfinite(v)):
                vecs.append(v); meta.append(trad)
    cur.close(); conn.close()
    X = np.asarray(vecs, float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12), np.array(meta)


def knn(X, k=KNN):
    tree = cKDTree(X)
    _, nb = tree.query(X, k=k + 1)
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k); cols = nb[:, 1:].reshape(-1)
    indeg = np.bincount(cols, minlength=n)                 # kNN in-degree (hubness)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr(); A.data[:] = 1.0; A.setdiag(0); A.eliminate_zeros()
    ncc, lab = connected_components(A, directed=False)
    keep = np.arange(n)
    if ncc > 1:
        keep = np.flatnonzero(lab == np.bincount(lab).argmax())
        A = A[keep][:, keep]
    return A, keep, indeg


def low_modes(A, m=M_MODES, kb=64):
    deg = np.asarray(A.sum(1)).ravel()
    D = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    Nadj = (D @ A @ D).tocsr()
    k = min(kb + 1, A.shape[0] - 2)
    mu, V = eigsh(Nadj, k=k, which="LA", ncv=min(A.shape[0] - 1, max(2 * k + 1, 60)), tol=0)
    order = np.argsort(mu)[::-1]; mu, V = mu[order], V[:, order]
    lam = 1.0 - mu
    return lam[1:1 + m], V[:, 1:1 + m], deg


def main():
    X, meta = fetch()
    A, keep, indeg = knn(X)
    meta = meta[keep]; indeg = indeg[keep]
    lam, V, deg = low_modes(A)
    scaled = V / np.sqrt(np.maximum(lam, 1e-9))
    ang = scaled / np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1e-12)
    radius = np.linalg.norm(scaled, axis=1)
    n = A.shape[0]

    rng = np.random.default_rng(7)
    anc = rng.choice(n, size=min(N_ANCHOR, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]; ok = np.isfinite(g)
    Ya = ang[anc]
    d = np.sqrt(((Ya[iu[0]] - Ya[iu[1]]) ** 2).sum(1))
    ma = meta[anc]
    same = (ma[iu[0]] == ma[iu[1]])

    def rho(mask):
        mm = mask & ok
        if mm.sum() < 50 or np.ptp(d[mm]) < 1e-12 or np.ptp(g[mm]) < 1e-12:
            return None, int(mm.sum())
        r = spearmanr(d[mm], g[mm]).statistic
        return (round(abs(r), 3) if np.isfinite(r) else None), int(mm.sum())

    all_rho, all_n = rho(np.ones_like(ok))
    win_rho, win_n = rho(same)
    bet_rho, bet_n = rho(~same)

    # hubness: does the magnitude track kNN in-degree / degree?
    out = {
        "n_items": int(n), "knn": KNN, "m_modes": M_MODES,
        "angle_rho_all": all_rho, "n_pairs_all": all_n,
        "angle_rho_within_tradition": win_rho, "n_pairs_within": win_n,
        "angle_rho_between_tradition": bet_rho, "n_pairs_between": bet_n,
        "hubness": {
            "rho_magnitude_vs_indegree": round(abs(spearmanr(radius, indeg).statistic), 3),
            "rho_magnitude_vs_degree": round(abs(spearmanr(radius, deg).statistic), 3),
            "rho_magnitude_vs_invsqrt_deg": round(abs(spearmanr(radius, 1/np.sqrt(deg)).statistic), 3),
        },
        "per_tradition_counts": {t: int((meta == t).sum()) for t in TRADITIONS},
    }
    print("###RESULTS_JSON_START###"); print(json.dumps(out)); print("###RESULTS_JSON_END###")
    for k, v in out.items():
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
