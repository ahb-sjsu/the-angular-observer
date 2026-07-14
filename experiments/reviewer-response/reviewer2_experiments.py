"""Second reviewer round for "Keep the Angle".

D. TRUNCATION BOUND + METRIC DISTORTION on the 2-torus (reviewer major #1, #2).
   Verifies the spectral-gap truncation bound
     0 <= D2_inf(i,j) - D2_m(i,j) <= 2/lambda_{m+1}   (pairwise)
     0 <= r2_inf(i)   - r2_m(i)   <= 1/lambda_{m+1}    (radius)
   between the FULL normalized commute-style embedding (all n-1 modes of L_sym)
   and the m-truncated Psi the paper uses; checks the radial-degeneracy transfer
   (full normalized radius ~ constant; truncated radius vs 1/sqrt(k_i)); and
   reports strict metric distortion -- empirical bi-Lipschitz constants and
   Procrustes disparity vs the canonical R^4 torus -- for angle / full /
   magnitude / random embeddings.

E. NONLINEAR SCALING LAW (reviewer major #3). Refits the 20 (d, angle-rho)
   points with linear, exponential, offset-exponential, and logistic models and
   compares R^2 and boundedness (rho <= 1).

F. OMEGA-THRESHOLD SENSITIVITY (reviewer minor, small-world gate). Sweeps the
   king-lattice rewire probability and reports omega / dim-spread / angle-rho so
   the gate's separation margin (and its transition zone) is explicit.

Local, CPU-only; reuses the frozen builders/estimators. JSON between markers.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

from rung0_validate import torus_graph, _norm_laplacian_eigs
from crosssubstrate import king_lattice_graph
from rung0_validate import dim_ball_growth, dim_spectral, dim_effective_rank

S, E = "###RESULTS_JSON_START###", "###RESULTS_JSON_END###"


# =========================================================== D. TRUNCATION + METRIC
def truncation_and_distortion(n=2000, seed=7):
    A, pts, r = torus_graph(n, 2)                # pts in [0,1)^2 (known geometry)
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    # full symmetric-normalized Laplacian spectrum (dense: this is the whole point)
    w, V = _norm_laplacian_eigs(A)               # w ascending, w[0]~0 trivial
    lam = w[1:]                                  # nontrivial eigenvalues
    Vt = V[:, 1:]                                # nontrivial eigenvectors (n x n-1)
    scaled = Vt / np.sqrt(np.maximum(lam, 1e-12))   # phi_k(i) = v_k(i)/sqrt(lam_k)

    rng = np.random.default_rng(seed)
    n_anchor = 200
    anc = rng.choice(n, size=n_anchor, replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(n_anchor, 1)
    g = G[iu]

    out = {"n": int(n), "n_anchor": int(n_anchor), "n_pairs": int(len(g))}

    # ---- truncation bound verification across m ----
    r2_inf = (scaled ** 2).sum(1)                        # full radius^2 per node
    bound_checks = []
    for m in [5, 10, 20, 40]:
        gap = float(lam[m])                              # lambda_{m+1} (0-indexed lam[m])
        # radius truncation error
        r2_m = (scaled[:, :m] ** 2).sum(1)
        rad_err = r2_inf - r2_m
        rad_ok = bool(np.all(rad_err <= 1.0 / gap + 1e-9)) and bool(np.all(rad_err >= -1e-9))
        rad_tight = float(np.max(rad_err) * gap)         # <=1 if bound holds; ratio to bound
        # pairwise distance truncation error on anchors
        Yinf = scaled[anc]
        Ym = scaled[anc][:, :m]
        d2_inf = ((Yinf[iu[0]] - Yinf[iu[1]]) ** 2).sum(1)
        d2_m = ((Ym[iu[0]] - Ym[iu[1]]) ** 2).sum(1)
        pair_err = d2_inf - d2_m
        pair_ok = bool(np.all(pair_err <= 2.0 / gap + 1e-9)) and bool(np.all(pair_err >= -1e-9))
        pair_tight = float(np.max(pair_err) * gap / 2.0)
        bound_checks.append(dict(m=m, lambda_gap=round(gap, 5),
                                 radius_bound_holds=rad_ok,
                                 radius_max_over_bound=round(rad_tight, 3),
                                 pair_bound_holds=pair_ok,
                                 pair_max_over_bound=round(pair_tight, 3)))
    out["truncation_bound"] = bound_checks

    # ---- radial degeneracy transfer ----
    inv_sqrt_k = 1.0 / np.sqrt(deg)
    out["degeneracy_transfer"] = dict(
        full_radius_mean=round(float(r2_inf.mean()), 4),
        full_radius_cv=round(float(r2_inf.std() / r2_inf.mean()), 4),  # ~0 => constant
        rho_full_radius_vs_invk=round(abs(spearmanr(np.sqrt(r2_inf), inv_sqrt_k).statistic), 3),
        rho_trunc_radius_vs_invk={
            str(m): round(abs(spearmanr(np.sqrt((scaled[:, :m] ** 2).sum(1)), inv_sqrt_k).statistic), 3)
            for m in [5, 10, 20, 40]})

    # ---- metric distortion: empirical bi-Lipschitz + Procrustes ----
    def bilip(dvec):
        ratio = dvec / np.maximum(g, 1e-9)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if len(ratio) < 10:
            return None
        kappa = float(np.max(ratio) / np.min(ratio))
        kappa_r = float(np.percentile(ratio, 97.5) / np.percentile(ratio, 2.5))
        return dict(kappa_full=round(kappa, 2), kappa_robust=round(kappa_r, 2))

    m = 20
    Ym = scaled[anc][:, :m]
    ang = Ym / np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
    d_angle = np.sqrt(((ang[iu[0]] - ang[iu[1]]) ** 2).sum(1))
    d_full = np.sqrt(((Ym[iu[0]] - Ym[iu[1]]) ** 2).sum(1))
    rad = np.linalg.norm(Ym, axis=1, keepdims=True)
    d_mag = np.abs(rad[iu[0]] - rad[iu[1]]).ravel()
    ridx = rng.choice(np.arange(scaled.shape[1]), size=m, replace=False)
    Yr = scaled[anc][:, ridx]
    Yr = Yr / np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
    d_rand = np.sqrt(((Yr[iu[0]] - Yr[iu[1]]) ** 2).sum(1))
    out["bilipschitz"] = {k: bilip(v) for k, v in
                          dict(angle=d_angle, full=d_full, magnitude=d_mag, random=d_rand).items()}
    out["bilip_note"] = ("kappa = (max d/g)/(min d/g); kappa_robust uses 2.5/97.5 pct. "
                         "Lower = tighter bi-Lipschitz. Rank fidelity for reference:")
    out["rank_rho"] = {k: round(abs(spearmanr(v, g).statistic), 3) for k, v in
                       dict(angle=d_angle, full=d_full, magnitude=d_mag, random=d_rand).items()}

    # Procrustes vs canonical flat-torus R^4 coords on the anchors
    P = pts[anc]
    canon = np.column_stack([np.cos(2 * np.pi * P[:, 0]), np.sin(2 * np.pi * P[:, 0]),
                             np.cos(2 * np.pi * P[:, 1]), np.sin(2 * np.pi * P[:, 1])])

    def pca4(Y):
        Yc = Y - Y.mean(0)
        U, s, Vt_ = np.linalg.svd(Yc, full_matrices=False)
        return (U[:, :4] * s[:4])

    proc = {}
    for name, Y in dict(angle=ang, full=Ym, random=Yr).items():
        try:
            _, _, disp = procrustes(canon, pca4(Y))
            proc[name] = round(float(disp), 4)
        except Exception as exc:  # noqa: BLE001
            proc[name] = f"err:{exc}"
    out["procrustes_disparity_vs_torus_R4"] = proc
    print("[D] truncation+distortion done", file=sys.stderr)
    return out


# =========================================================== E. NONLINEAR SCALING
def nonlinear_scaling(points):
    d = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)

    def r2(yhat):
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    models = {}
    # linear
    b, a = np.polyfit(d, y, 1)  # slope, intercept via polyfit returns [slope, intercept]
    models["linear"] = dict(form="a + b d", a=round(float(a), 3), b=round(float(b), 3),
                            r2=round(r2(a + b * d), 3), at_d0=round(float(a), 3),
                            bounded_le1=bool(a <= 1.0))
    # exponential A exp(-k d)
    try:
        (A, k), _ = curve_fit(lambda x, A, k: A * np.exp(-k * x), d, y,
                              p0=[1.0, 0.1], maxfev=20000)
        models["exponential"] = dict(form="A exp(-k d)", A=round(float(A), 3),
                                     k=round(float(k), 3), r2=round(r2(A * np.exp(-k * d)), 3),
                                     at_d0=round(float(A), 3), bounded_le1=bool(A <= 1.0),
                                     limit_dinf="0")
    except Exception as exc:  # noqa: BLE001
        models["exponential"] = f"err:{exc}"
    # offset exponential c + A exp(-k d)
    try:
        (c, A, k), _ = curve_fit(lambda x, c, A, k: c + A * np.exp(-k * x), d, y,
                                 p0=[0.3, 0.7, 0.3], maxfev=40000)
        models["offset_exp"] = dict(form="c + A exp(-k d)", c=round(float(c), 3),
                                    A=round(float(A), 3), k=round(float(k), 3),
                                    r2=round(r2(c + A * np.exp(-k * d)), 3),
                                    at_d0=round(float(c + A), 3), bounded_le1=bool(c + A <= 1.0),
                                    limit_dinf=round(float(c), 3))
    except Exception as exc:  # noqa: BLE001
        models["offset_exp"] = f"err:{exc}"
    # logistic 1/(1+exp(k(d-d0))) -- strictly in (0,1)
    try:
        (k, d0), _ = curve_fit(lambda x, k, d0: 1.0 / (1.0 + np.exp(k * (x - d0))), d, y,
                               p0=[1.0, 3.0], maxfev=40000)
        yl = 1.0 / (1.0 + np.exp(k * (d - d0)))
        models["logistic"] = dict(form="1/(1+exp(k(d-d0)))", k=round(float(k), 3),
                                  d0=round(float(d0), 3), r2=round(r2(yl), 3),
                                  at_d0=round(float(1 / (1 + np.exp(k * (0 - d0)))), 3),
                                  bounded_le1=True, limit_dinf="0")
    except Exception as exc:  # noqa: BLE001
        models["logistic"] = f"err:{exc}"
    print("[E] nonlinear scaling done", file=sys.stderr)
    return dict(n_points=len(points), models=models)


# =========================================================== F. OMEGA SENSITIVITY
def omega_sensitivity():
    import networkx as nx

    def screen(A, seed):
        G = nx.convert_node_labels_to_integers(
            nx.from_scipy_sparse_array(A).subgraph(
                max(nx.connected_components(nx.from_scipy_sparse_array(A)), key=len)).copy())
        n, mm = G.number_of_nodes(), G.number_of_edges()
        kbar = 2.0 * mm / n

        def sampL(H, s=0, src=120):
            rng = np.random.default_rng(s)
            nodes = list(H.nodes()); sr = rng.choice(len(nodes), size=min(src, len(nodes)), replace=False)
            t = c = 0
            for i in sr:
                for _, dd in nx.single_source_shortest_path_length(H, nodes[i]).items():
                    if dd > 0:
                        t += dd; c += 1
            return t / max(c, 1)
        C = nx.average_clustering(G); L = sampL(G, seed)
        R = nx.convert_node_labels_to_integers(nx.gnm_random_graph(n, mm, seed=seed))
        R = R.subgraph(max(nx.connected_components(R), key=len)).copy()
        C_rand, L_rand = nx.average_clustering(R), sampL(nx.convert_node_labels_to_integers(R), seed)
        k = max(2, int(round(kbar)))
        C_latt = nx.average_clustering(nx.watts_strogatz_graph(n, k if k % 2 == 0 else k + 1, 0.0, seed=seed))
        return (L_rand / L) - (C / C_latt)

    def angle_rho(A, seed):
        w, V = _norm_laplacian_eigs(A)
        rng = np.random.default_rng(seed); n = A.shape[0]
        anc = rng.choice(n, size=min(150, n), replace=False)
        G = shortest_path(A, unweighted=True, indices=anc)[:, anc]
        iu = np.triu_indices(len(anc), 1); g = G[iu]
        Y = (V[:, 1:21] / np.sqrt(np.maximum(w[1:21], 1e-9)))[anc]
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        d = np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(1))
        return abs(spearmanr(d, g).statistic)

    def dspread(A):
        w, V = _norm_laplacian_eigs(A)
        return float(np.ptp([dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)]))

    rows = []
    for p in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]:
        om, ar, ds = [], [], []
        for s in range(3):
            A = king_lattice_graph(side=45, rewire_p=p, seed=2 + s)
            om.append(screen(A, 10 + s)); ar.append(angle_rho(A, 20 + s)); ds.append(dspread(A))
        rows.append(dict(rewire_p=p, omega=round(float(np.mean(om)), 3),
                         omega_std=round(float(np.std(om)), 3),
                         angle_rho=round(float(np.mean(ar)), 3),
                         dim_spread=round(float(np.mean(ds)), 3)))
        print(f"[F] p={p} done", file=sys.stderr)
    # gate margin: max omega among rho>=0.7 (clean) vs min omega among rho<0.6 (contaminated)
    clean = [r["omega"] for r in rows if r["angle_rho"] >= 0.7]
    cont = [r["omega"] for r in rows if r["angle_rho"] < 0.6]
    margin = dict(max_omega_clean=max(clean) if clean else None,
                  min_omega_contaminated=min(cont) if cont else None)
    return dict(sweep=rows, gate_margin=margin)


if __name__ == "__main__":
    import os
    result = {}
    result["torus_truncation_distortion"] = truncation_and_distortion()
    # reload the 20 scaling points from the round-1 artifact
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "todo_experiments_result.json")) as f:
        pts = json.load(f)["scaling_law"]["points"]
    result["nonlinear_scaling"] = nonlinear_scaling(pts)
    result["omega_sensitivity"] = omega_sensitivity()
    with open(os.path.join(here, "reviewer2_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(S); print(json.dumps(result)); print(E)
