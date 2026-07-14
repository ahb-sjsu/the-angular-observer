"""Numerical test of the Normalization Lemma hypotheses on the 2-torus.

Identity (exact):  ||hat u - hat v||^2 = (||u-v||^2 - (a-b)^2) / (a b),
with u=Psi^m_x, a=||u||.  The lemma: if the unnormalized map F is L-bi-Lipschitz
to geodesic distance and its radius is Lambda-Lipschitz with Lambda < 1/L, the
angular map is bi-Lipschitz.  We measure, on LOCAL pairs (small geodesic g, where
JMS charts live):
  A     = robust lower stretch of F        (min ratio ||u-v||/g)
  B     = robust upper stretch of F        (max ratio ||u-v||/g)
  Lam   = robust radius Lipschitz constant (max |a-b|/g)
and check whether Lam < A pointwise-robustly (the deterministic sufficient
condition) or only on average (which would mean the global claim is rank-level).
Also verifies the identity to machine precision.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from rung0_validate import torus_graph, _norm_laplacian_eigs

def run(n=2000, m=20, r0=8, seed=7):
    A, pts, r = torus_graph(n, 2)
    w, V = _norm_laplacian_eigs(A)
    scaled = V[:, 1:1 + m] / np.sqrt(np.maximum(w[1:1 + m], 1e-12))
    rng = np.random.default_rng(seed)
    anc = rng.choice(A.shape[0], size=250, replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]
    U = scaled[anc]
    a = np.linalg.norm(U, axis=1)
    du = np.sqrt(((U[iu[0]] - U[iu[1]]) ** 2).sum(1))          # ||u-v||
    da = np.abs(a[iu[0]] - a[iu[1]])                            # |a-b|
    ab = a[iu[0]] * a[iu[1]]
    hatU = U / np.maximum(a[:, None], 1e-12)
    dhat = np.sqrt(((hatU[iu[0]] - hatU[iu[1]]) ** 2).sum(1))   # ||hat u - hat v||

    # identity check
    rhs = np.sqrt(np.maximum((du ** 2 - da ** 2) / np.maximum(ab, 1e-12), 0.0))
    id_err = float(np.max(np.abs(dhat - rhs)))

    # local pairs
    loc = g <= r0
    gl = g[loc]
    ratio_F = du[loc] / gl
    ratio_rad = da[loc] / gl
    A_lo = float(np.percentile(ratio_F, 2.5))
    B_hi = float(np.percentile(ratio_F, 97.5))
    Lam = float(np.percentile(ratio_rad, 97.5))
    Lam_med = float(np.percentile(ratio_rad, 50))
    frac_viol = float(np.mean(ratio_rad >= ratio_F))   # fraction of local pairs where |da|/g >= ||u-v||/g

    # angular bi-Lipschitz on local pairs, robust
    ah = dhat[loc] / gl
    kap_ang = float(np.percentile(ah, 97.5) / np.percentile(ah, 2.5))

    return dict(
        n=int(A.shape[0]), m=m, r0=r0, n_local_pairs=int(loc.sum()),
        identity_max_abs_err=id_err,
        A_lower_stretch_F=round(A_lo, 4), B_upper_stretch_F=round(B_hi, 4),
        Lambda_radius_lip=round(Lam, 4), Lambda_median=round(Lam_med, 4),
        deterministic_condition_Lambda_lt_A=bool(Lam < A_lo),
        frac_local_pairs_violating=round(frac_viol, 4),
        angular_local_bilip_robust=round(kap_ang, 3),
        rho_radius_vs_invsqrtk=round(abs(spearmanr(
            a, 1.0 / np.sqrt(np.asarray(A.sum(1)).ravel()[anc])).statistic), 3),
    )

if __name__ == "__main__":
    out = run()
    print("###RESULTS_JSON_START###")
    print(json.dumps(out))
    print("###RESULTS_JSON_END###")
    for k, v in out.items():
        print(f"  {k}: {v}", file=sys.stderr)
