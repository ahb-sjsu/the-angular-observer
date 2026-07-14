"""Reviewer-response experiments for "Keep the Angle" (the-angular-observer).

Three self-contained computations, each answering an author-TODO left in the
paper draft. Reuses the frozen substrate builders and estimators from rungs 0/1
so nothing here changes a published number by re-defining the pipeline.

  A. SCALING LAW (paper 5.4, TODOs on slope SE / R^2 / band)
     Re-run the manifold library (library.LIB, 5 rules x seeds) to collect the
     (emergent-dim d, angle-rho) cloud, then OLS slope with standard error, R^2,
     a bootstrap 95% CI on the slope, and the Spearman monotonicity CI.

  B. SMALL-WORLD TWO-FACTOR SCREEN (paper 5.3, reviewer's requested gate)
     For each universality-table substrate: the three dimension estimators
     (ball / spectral / effective-rank) and their spread, angle-rho, and a
     two-factor small-world screen -- clustering C and characteristic path
     length L against a degree-matched random (ER) and ring-lattice baseline,
     giving Humphries sigma and Telesford omega. Flags substrates whose low
     angle-rho is small-world contamination rather than clean geometry.

  C. HEADLINE BOOTSTRAP CIs (paper 5.2, TODO: pair counts + CIs)
     The 2-torus core table (angle vs magnitude vs full, over mode-count m) with
     the exact number of sampled node pairs and a 95% bootstrap CI over
     pair-resampling for every angle-rho, plus variation over independent draws.

Emits one JSON blob between the markers for the paper-folding step.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from rung0_validate import (torus_graph, _norm_laplacian_eigs, dim_ball_growth,
                            dim_spectral, dim_effective_rank)
from crosssubstrate import (causal_set_graph, king_lattice_graph,
                            embedding_trajectory_graph)

S, E = "###RESULTS_JSON_START###", "###RESULTS_JSON_END###"
BOOT = 2000
RNG = np.random.default_rng(20260714)


def _pct(a, lo=2.5, hi=97.5):
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


# ------------------------------------------------------------------- geodesics
def _anchor_geodesics(A, n_anchor, seed):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    anc = rng.choice(n, size=min(n_anchor, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]
    ok = np.isfinite(g)
    return anc, iu, g, ok


def _angle_embed(w, V, anc, m):
    Y = (V[:, 1:1 + m] / np.sqrt(np.maximum(w[1:1 + m], 1e-9)))[anc]
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def _pair_dist(Y, iu):
    return np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(-1))


def _rho(d, g):
    if np.ptp(d) < 1e-12:
        return 0.0
    r = spearmanr(d, g).statistic
    return abs(r) if np.isfinite(r) else 0.0


def _boot_rho_ci(d, g, boot=BOOT, seed=0):
    """95% CI for |Spearman(d,g)| by resampling the pair set with replacement."""
    rng = np.random.default_rng(seed)
    npairs = len(g)
    stats = np.empty(boot)
    for b in range(boot):
        idx = rng.integers(0, npairs, npairs)
        stats[b] = _rho(d[idx], g[idx])
    return _pct(stats)


# =============================================================== A. SCALING LAW
def scaling_law():
    from library import LIB, measure
    pts = []
    for name, R in LIB.items():
        for r in measure(name, R):
            pts.append((float(r["dim"]), float(r["angle"]), name))
    d = np.array([p[0] for p in pts])
    a = np.array([p[1] for p in pts])
    n = len(pts)

    # OLS with slope standard error and R^2
    X = np.vstack([np.ones(n), d]).T
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    inter, slope = beta
    resid = a - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(cov[1, 1]))
    se_inter = float(np.sqrt(cov[0, 0]))
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot

    # bootstrap CI on the slope (resample manifolds)
    bs = np.empty(BOOT)
    for b in range(BOOT):
        idx = RNG.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(X[idx], a[idx], rcond=None)
        bs[b] = bb[1]
    slope_ci = _pct(bs)

    # Spearman monotonicity + its bootstrap CI
    sp = spearmanr(d, a).statistic
    sps = np.empty(BOOT)
    for b in range(BOOT):
        idx = RNG.integers(0, n, n)
        if np.ptp(d[idx]) < 1e-9:
            sps[b] = np.nan
            continue
        sps[b] = spearmanr(d[idx], a[idx]).statistic
    sp_ci = _pct(sps[np.isfinite(sps)])

    return dict(
        n_manifolds=n, d_min=float(d.min()), d_max=float(d.max()),
        pairs_per_point=int(150 * 149 / 2),
        slope=float(slope), slope_se=se_slope, slope_ci95=slope_ci,
        intercept=float(inter), intercept_se=se_inter,
        r2=float(r2), spearman=float(sp), spearman_ci95=sp_ci,
        points=[[round(x, 3), round(y, 3)] for x, y, _ in pts],
    )


# ===================================================== B. SMALL-WORLD SCREEN
def _sample_L(G, sources=200, seed=0):
    """Characteristic path length from BFS off a sampled source set."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    src = rng.choice(len(nodes), size=min(sources, len(nodes)), replace=False)
    tot, cnt = 0.0, 0
    for s in src:
        for _, dist in nx.single_source_shortest_path_length(G, nodes[s]).items():
            if dist > 0:
                tot += dist
                cnt += 1
    return tot / max(cnt, 1)


def _small_world(A, seed=0):
    """Two-factor screen: C and L vs degree-matched ER and ring-lattice."""
    G = nx.from_scipy_sparse_array(A)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    kbar = 2.0 * m / n
    C = nx.average_clustering(G)
    L = _sample_L(G, seed=seed)

    # degree-matched Erdos-Renyi baseline (mean over a few draws)
    Cr, Lr = [], []
    for j in range(3):
        R = nx.gnm_random_graph(n, m, seed=seed + j)
        R = R.subgraph(max(nx.connected_components(R), key=len)).copy()
        R = nx.convert_node_labels_to_integers(R)
        Cr.append(nx.average_clustering(R))
        Lr.append(_sample_L(R, sources=120, seed=seed + j))
    C_rand, L_rand = float(np.mean(Cr)), float(np.mean(Lr))

    # ring-lattice baseline for omega's clustering term
    k = max(2, int(round(kbar)))
    Lat = nx.watts_strogatz_graph(n, k if k % 2 == 0 else k + 1, 0.0, seed=seed)
    C_latt = nx.average_clustering(Lat)

    sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L > 0 else float("nan")
    omega = (L_rand / L) - (C / C_latt) if C_latt > 0 and L > 0 else float("nan")
    return dict(n=n, kbar=round(kbar, 2), C=round(C, 4), L=round(L, 3),
                C_rand=round(C_rand, 4), L_rand=round(L_rand, 3),
                C_latt=round(C_latt, 4), sigma=round(sigma, 3),
                omega=round(omega, 3))


def small_world_screen():
    builders = [
        ("2-torus",              lambda: torus_graph(2000, 2)[0]),
        ("king-lattice p=0",     lambda: king_lattice_graph(side=45, rewire_p=0.0)),
        ("king-lattice p=0.03",  lambda: king_lattice_graph(side=45, rewire_p=0.03)),
        ("king-lattice p=0.10",  lambda: king_lattice_graph(side=45, rewire_p=0.10)),
        ("king-lattice p=0.30",  lambda: king_lattice_graph(side=45, rewire_p=0.30)),
        ("causal-set 1+1D",      lambda: causal_set_graph(n=2000, k=6)),
        ("swiss-roll R12",       lambda: embedding_trajectory_graph(n=2000, ambient=12, k=10)),
    ]
    out = []
    for i, (name, build) in enumerate(builders):
        try:
            A = build()
            w, V = _norm_laplacian_eigs(A)
            dims = [float(dim_ball_growth(A)), float(dim_spectral(w)),
                    float(dim_effective_rank(A, w, V))]
            anc, iu, g, ok = _anchor_geodesics(A, 150, 1 + i)
            Y = _angle_embed(w, V, anc, 20)
            d = _pair_dist(Y, iu)
            angle = _rho(d[ok], g[ok])
            sw = _small_world(A, seed=1 + i)
            rec = dict(substrate=name,
                       dim_ball=round(dims[0], 3), dim_spectral=round(dims[1], 3),
                       dim_effrank=round(dims[2], 3),
                       dim_mean=round(float(np.mean(dims)), 3),
                       dim_spread=round(float(np.ptp(dims)), 3),
                       dim_std=round(float(np.std(dims)), 3),
                       angle_rho=round(angle, 3), **sw)
        except Exception as exc:  # noqa: BLE001
            rec = dict(substrate=name, error=str(exc))
        out.append(rec)
        print(f"[B] {name:<22} done", file=sys.stderr)
    return out


# ===================================================== C. HEADLINE BOOTSTRAP
def headline_bootstrap():
    ms = [5, 10, 20, 40]
    # primary draw
    A = torus_graph(2000, 2)[0]
    w, V = _norm_laplacian_eigs(A)
    anc, iu, g, ok = _anchor_geodesics(A, 150, 42)
    g_ok = g[ok]
    n_pairs = int(ok.sum())
    rows = []
    for m in ms:
        Y = _angle_embed(w, V, anc, m)
        dA = _pair_dist(Y, iu)[ok]
        # magnitude-only (radius of low-mode rows) and full (mag x dir)
        Yl = (V[:, 1:1 + m] / np.sqrt(np.maximum(w[1:1 + m], 1e-9)))[anc]
        rad = np.linalg.norm(Yl, axis=1, keepdims=True)
        dMag = _pair_dist(rad, iu)[ok]
        dFull = _pair_dist(Yl, iu)[ok]
        rows.append(dict(
            m=m, n_pairs=n_pairs,
            angle=round(_rho(dA, g_ok), 3),
            angle_ci95=[round(x, 3) for x in _boot_rho_ci(dA, g_ok, seed=m)],
            magnitude=round(_rho(dMag, g_ok), 3),
            full=round(_rho(dFull, g_ok), 3),
        ))
        print(f"[C] m={m} done", file=sys.stderr)

    # variation over independent graph draws at m=10
    draws = []
    for s in range(5):
        Ad = torus_graph(2000, 2)[0]
        wd, Vd = _norm_laplacian_eigs(Ad)
        ancd, iud, gd, okd = _anchor_geodesics(Ad, 150, 100 + s)
        Yd = _angle_embed(wd, Vd, ancd, 10)
        dd = _pair_dist(Yd, iud)[okd]
        draws.append(_rho(dd, gd[okd]))
    return dict(n_pairs=n_pairs, table=rows,
                m10_over_draws=dict(mean=round(float(np.mean(draws)), 3),
                                    std=round(float(np.std(draws)), 3),
                                    n_draws=len(draws),
                                    values=[round(x, 3) for x in draws]))


if __name__ == "__main__":
    result = {}
    print(">> A. scaling law", file=sys.stderr)
    result["scaling_law"] = scaling_law()
    print(">> B. small-world screen", file=sys.stderr)
    result["small_world"] = small_world_screen()
    print(">> C. headline bootstrap", file=sys.stderr)
    result["headline"] = headline_bootstrap()
    with open("todo_experiments_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(S)
    print(json.dumps(result))
    print(E)
