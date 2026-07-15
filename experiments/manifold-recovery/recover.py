"""Manifold recovery from Wolfram-model hypergraphs via the angular basis.

Question (beyond angle-rho rank fidelity, which the paper already measures):
when we row-normalize the commute-weighted low-mode embedding of an *emergent*
hypergraph, does the angular geometry recover a smooth, RECOGNIZABLE manifold
-- and which one -- or something unexpected?

Battery per substrate (controls with known ground truth + the 5 library rules):
  1. density irregularity: degree CV (the vL-R-H failure driver);
  2. intrinsic dimension: ball-growth / spectral / effective-rank + spread;
  3. fidelity: angle-rho vs hop geodesics; magnitude-rho; random-mode control;
     alpha=1 density-normalized angle-rho (Coifman-Lafon);
  4. SHAPE of the angular geometry:
     - flat-stress: residual stress embedding the angular distances in R^2/R^3
       (a flat patch embeds with low stress; curved/closed geometry cannot);
     - sphere fit: best-fit sphere to the 3D MDS of angular distances,
       relative RMS radial residual (S^2 should fit; a plane/torus should not);
     - homogeneity: anchor-eccentricity spread from hop geodesics
       (closed homogeneous manifold -> narrow; bounded patch -> broad);
     - low-spectrum multiplet signature: lambda_k/lambda_1, k=1..8, scored
       against templates (flat 2-torus, S^2, Neumann square patch, interval).
  5. figures: 2D MDS layout of the ANGULAR distances per substrate, shaded by
     local degree (shape + density in one look), plus the spectral signatures.

Run from repo root:  PYTHONPATH=. python experiments/manifold-recovery/recover.py
Emits JSON between ###RESULTS_JSON_START###/###RESULTS_JSON_END### and PNGs
next to this file. Pure NumPy/SciPy/Matplotlib, CPU, dense eigh (n<=2000).
"""

import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from arity3 import hyper_to_csr
from library import LIB, crop
from rung0_validate import (
    _norm_laplacian_eigs,
    dim_ball_growth,
    dim_effective_rank,
    dim_spectral,
    sphere_graph,
    torus_graph,
)
from rung1b_rewriter import largest_component, rewrite

M = 20            # angular mode budget (paper default)
N_ANCHOR = 150    # geodesic anchor set (paper default)
HERE = os.path.dirname(os.path.abspath(__file__))

# low-spectrum templates: lambda_k / lambda_1 for k = 1..8
TEMPLATES = {
    "flat 2-torus":  [1, 1, 1, 1, 2, 2, 2, 2],          # mu ~ j^2+k^2: 1x4, 2x4
    "sphere S^2":    [1, 1, 1, 3, 3, 3, 3, 3],          # mu = l(l+1): 2x3, 6x5
    "square patch":  [1, 1, 2, 4, 4, 5, 5, 8],          # Neumann square, mu ~ j^2+k^2
    "interval (1D)": [1, 4, 9, 16, 25, 36, 49, 64],     # mu ~ k^2
}


def angular_rows(w, V, modes):
    Y = V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9))
    r = np.linalg.norm(Y, axis=1)
    return Y / np.maximum(r[:, None], 1e-12), r


def alpha1_eigs(A):
    q = np.asarray(A.sum(1)).ravel()
    Dq = 1.0 / np.maximum(q, 1e-12)
    W1 = A.multiply(Dq[:, None]).multiply(Dq[None, :]).tocsr()
    return _norm_laplacian_eigs(W1)


def pair_dists(X, iu):
    return np.sqrt(((X[iu[0]] - X[iu[1]]) ** 2).sum(-1))


def classical_mds(D, k):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def stress(D, X, iu):
    d = pair_dists(X, iu)
    g = D[iu]
    return float(np.sqrt(((d - g) ** 2).sum() / (g ** 2).sum()))


def sphere_fit_residual(X3):
    # algebraic least-squares sphere: |x|^2 = 2 c.x + (R^2 - |c|^2)
    A_ = np.hstack([2 * X3, np.ones((len(X3), 1))])
    b = (X3 ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A_, b, rcond=None)
    c, r2 = sol[:3], sol[3] + (sol[:3] ** 2).sum()
    R = np.sqrt(max(r2, 1e-12))
    return float(np.sqrt(((np.linalg.norm(X3 - c, axis=1) - R) ** 2).mean()) / R)


def signature(w):
    lam = w[1:9] / max(w[1], 1e-12)
    scores = {
        name: float(np.sqrt(((np.log(lam) - np.log(np.array(t, float))) ** 2).mean()))
        for name, t in TEMPLATES.items()
    }
    best = min(scores, key=scores.get)
    return [round(float(x), 2) for x in lam], scores, best


def build_rule(name, R):
    for s in (1, 2, 3, 4):
        tris, nid = rewrite(R["lhs"], R["rhs"], R["seed"],
                            max_edges=3000, max_gen=500, seed=s)
        A = largest_component(hyper_to_csr(tris, nid))
        if A.shape[0] >= 200:
            return crop(A), s
    return None, None


def measure(name, A, rng):
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    w, V = _norm_laplacian_eigs(A)

    anc = rng.choice(n, size=min(N_ANCHOR, n), replace=False)
    Dg = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    ok = np.isfinite(Dg[iu])
    g = Dg[iu][ok]

    low = np.arange(1, 1 + M)
    U, r = angular_rows(w, V, low)
    Dang = np.sqrt(((U[anc][:, None, :] - U[anc][None, :, :]) ** 2).sum(-1))
    d_ang = Dang[iu][ok]

    rand = rng.choice(np.arange(1, len(w)), size=M, replace=False)
    Ur, _ = angular_rows(w, V, rand)
    d_rand = pair_dists(Ur[anc], iu)[ok]
    d_mag = np.abs(r[anc][iu[0]] - r[anc][iu[1]])[ok]

    w1, V1 = alpha1_eigs(A)
    U1, _ = angular_rows(w1, V1, low)
    d_a1 = pair_dists(U1[anc], iu)[ok]

    X2 = classical_mds(Dang, 2)
    X3 = classical_mds(Dang, 3)
    ecc = Dg[np.isfinite(Dg)].reshape(len(anc), -1).max(1) if np.isfinite(Dg).all() \
        else np.array([row[np.isfinite(row)].max() for row in Dg])
    lam, scores, best = signature(w)

    dims = [dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)]
    res = dict(
        substrate=name, n=int(n),
        mean_deg=round(float(deg.mean()), 2),
        deg_cv=round(float(deg.std() / deg.mean()), 3),
        dim=round(float(np.mean(dims[:2])), 2),
        dim_spread=round(float(np.std(dims)), 2),
        angle_rho=round(abs(spearmanr(d_ang, g).statistic), 3),
        angle_rho_alpha1=round(abs(spearmanr(d_a1, g).statistic), 3),
        magnitude_rho=round(abs(spearmanr(d_mag, g).statistic), 3),
        random_rho=round(abs(spearmanr(d_rand, g).statistic), 3),
        flat_stress_2d=round(stress(Dang, X2, iu), 3),
        flat_stress_3d=round(stress(Dang, X3, iu), 3),
        sphere_residual=round(sphere_fit_residual(X3), 3),
        ecc_spread=round(float((ecc.max() - ecc.min()) / ecc.mean()), 3),
        spectrum_ratios=lam,
        signature_scores={k: round(v, 3) for k, v in scores.items()},
        signature_best=best,
    )
    return res, X2, deg[anc]


def main():
    rng = np.random.default_rng(7)
    graphs = []
    A, _, _ = torus_graph(2000, 2)
    graphs.append(("torus T^2 (control)", largest_component(A)))
    A, _, _ = sphere_graph(2000)
    graphs.append(("sphere S^2 (control)", largest_component(A)))
    for name, R in LIB.items():
        A, s = build_rule(name, R)
        if A is None:
            print(f"{name}: stalled at all seeds, skipped")
            continue
        graphs.append((f"{name.strip()} [s={s}]", A))

    results, layouts = [], []
    for name, A in graphs:
        res, X2, deg_anc = measure(name, A, rng)
        results.append(res)
        layouts.append((name, X2, deg_anc, res))
        print(f"{name:26s} n={res['n']:4d} degCV={res['deg_cv']:.2f} "
              f"d={res['dim']:.2f}(±{res['dim_spread']:.2f}) "
              f"angle={res['angle_rho']:.2f} a1={res['angle_rho_alpha1']:.2f} "
              f"mag={res['magnitude_rho']:.2f} rnd={res['random_rho']:.2f} "
              f"stress2d={res['flat_stress_2d']:.2f} sph={res['sphere_residual']:.2f} "
              f"ecc±={res['ecc_spread']:.2f} sig={res['signature_best']}")

    make_figures(layouts, results)
    with open(os.path.join(HERE, "recover_result.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("###RESULTS_JSON_START###")
    print(json.dumps(results, indent=1))
    print("###RESULTS_JSON_END###")


def make_figures(layouts, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(layouts)
    ncol = min(4, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[k:]:
        ax.axis("off")
    for ax, (name, X2, deg_anc, res) in zip(axes, layouts):
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=deg_anc, cmap="Blues", s=14,
                        vmin=0, edgecolors="#1f3a5f", linewidths=0.3)
        ax.set_title(name, fontsize=9, color="#333333")
        ax.text(0.02, 0.02,
                f"angle ρ={res['angle_rho']:.2f}  d={res['dim']:.1f}\n"
                f"stress₂={res['flat_stress_2d']:.2f}  "
                f"sph res={res['sphere_residual']:.2f}",
                transform=ax.transAxes, fontsize=7, color="#555555", va="bottom")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color("#dddddd")
    cb = fig.colorbar(sc, ax=axes.tolist(), shrink=0.5, pad=0.01)
    cb.set_label("node degree (sampling density)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("2D MDS of ANGULAR distances — what the angular observer sees",
                 fontsize=11)
    fig.savefig(os.path.join(HERE, "angular_layouts.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ks = np.arange(1, 9)
    for i, res in enumerate(results):
        ax.plot(ks + i * 0.04 - 0.14, res["spectrum_ratios"], "o", ms=4.5,
                color=plt.cm.Blues(0.35 + 0.6 * i / max(len(results) - 1, 1)),
                markeredgecolor="#1f3a5f", markeredgewidth=0.3)
        ax.annotate(res["substrate"].split("[")[0].strip(),
                    (8.15, res["spectrum_ratios"][-1]), fontsize=6.5,
                    color="#555555", va="center")
    for name, t, style in (("flat 2-torus", TEMPLATES["flat 2-torus"], "-"),
                           ("sphere S²", TEMPLATES["sphere S^2"], "--")):
        ax.plot(ks, t, style, color="#bbbbbb", lw=1.2, zorder=0)
        ax.annotate(name, (5.1, t[5]), fontsize=7, color="#999999")
    ax.set_xlabel("mode k", fontsize=9)
    ax.set_ylabel(r"$\lambda_k/\lambda_1$", fontsize=9)
    ax.set_ylim(0, 9)
    ax.set_title("Low-spectrum multiplet signatures vs closed-manifold templates",
                 fontsize=10)
    ax.grid(color="#eeeeee", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#dddddd")
    fig.savefig(os.path.join(HERE, "spectral_signatures.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
