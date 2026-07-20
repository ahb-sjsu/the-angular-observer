"""(A2) tangential-noncollapse margin on the swiss roll (non-homogeneous), the
test review comment 4/C asks for. Same local-pair measurement as
normalization_check.py (torus: kap_ang~2.2, frac_viol=0/9200)."""

import sys, json, numpy as np, statistics as st

sys.path.insert(0, "../..")
sys.path.insert(0, ".")
from scipy.sparse.csgraph import shortest_path
from rung0_validate import _norm_laplacian_eigs
from crosssubstrate import embedding_trajectory_graph


def a2(A, m=20, r0=8, anchor_seed=7):
    w, V = _norm_laplacian_eigs(A)
    scaled = V[:, 1 : 1 + m] / np.sqrt(np.maximum(w[1 : 1 + m], 1e-12))
    rng = np.random.default_rng(anchor_seed)
    anc = rng.choice(A.shape[0], size=min(250, A.shape[0]), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]
    U = scaled[anc]
    a = np.linalg.norm(U, axis=1)
    du = np.sqrt(((U[iu[0]] - U[iu[1]]) ** 2).sum(1))
    da = np.abs(a[iu[0]] - a[iu[1]])
    hatU = U / np.maximum(a[:, None], 1e-12)
    dhat = np.sqrt(((hatU[iu[0]] - hatU[iu[1]]) ** 2).sum(1))
    loc = np.isfinite(g) & (g <= r0) & (du > 0)
    gl = g[loc]
    rF = du[loc] / gl
    rrad = da[loc] / gl
    ah = dhat[loc] / gl
    A_lo = np.percentile(rF, 2.5)
    Lam = np.percentile(rrad, 97.5)
    Lam_med = np.percentile(rrad, 50)
    return dict(
        n_local=int(loc.sum()),
        kap_ang=float(np.percentile(ah, 97.5) / np.percentile(ah, 2.5)),
        frac_viol=float(np.mean(rrad >= rF)),
        n_degenerate=int(np.sum(rrad >= rF)),
        A_lo=float(A_lo),
        Lam=float(Lam),
        Lam_med=float(Lam_med),
        Lam_med_over_A=float(Lam_med / A_lo),
    )


acc = {
    k: [] for k in ["kap_ang", "frac_viol", "n_degenerate", "n_local", "Lam_med_over_A"]
}
for s in range(5):
    A = embedding_trajectory_graph(n=2000, ambient=12, k=10, noise=0.03, seed=s)
    o = a2(A, anchor_seed=7)
    for k in acc:
        acc[k].append(o[k])
    print(
        f"draw seed={s}: N_local={o['n_local']:5d}  kap_ang={o['kap_ang']:.3f}  "
        f"degenerate={o['n_degenerate']}/{o['n_local']}  Lam_med/A={o['Lam_med_over_A']:.2f}",
        file=sys.stderr,
    )
res = {
    k: dict(
        mean=round(st.mean(v), 3),
        sd=round(st.pstdev(v), 3),
        min=round(min(v), 3),
        max=round(max(v), 3),
        draws=v,
    )
    for k, v in acc.items()
}
json.dump(res, open("a2_swissroll_result.json", "w"), indent=2)
print("=== swiss-roll (A2), 5 draws ===")
print(
    f"  local angular bi-Lipschitz kappa : {res['kap_ang']['mean']:.2f} +/- {res['kap_ang']['sd']:.2f}  (min {res['kap_ang']['min']:.2f}, max {res['kap_ang']['max']:.2f})"
)
tot_deg = sum(acc["n_degenerate"])
tot_loc = sum(acc["n_local"])
print(
    f"  radially-degenerate local pairs  : {tot_deg}/{tot_loc} total  (per-draw frac {res['frac_viol']['mean']:.4f})"
)
print(
    f"  median Lambda / lower stretch A   : {res['Lam_med_over_A']['mean']:.2f} +/- {res['Lam_med_over_A']['sd']:.2f}"
)
