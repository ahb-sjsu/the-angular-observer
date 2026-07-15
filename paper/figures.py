"""Regenerate the paper's figures from the code (PDF for LaTeX, PNG for markdown).

    python paper/figures.py

Fig 1  the core result: angle / full / magnitude vs mode count on a 2-torus.
Fig 2  the scaling law: angle-rho vs emergent dimension across manifolds.
Fig 3  the basis control: low-mode vs random-mode preservation vs mode count.
"""

import os
import sys
import json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from rung0_validate import torus_graph, _norm_laplacian_eigs  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", name)


def _pres(A, w, V, modes, kind, n_anchor=150):
    anc = RNG.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = D[iu]
    Y = (V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9)))[anc]
    if kind == "angle":
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    elif kind == "full":
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    else:  # magnitude
        r = np.linalg.norm(Y, axis=1)
        d = np.abs(r[:, None] - r[None, :])[iu]
    return abs(spearmanr(d, g).statistic)


def fig1_core():
    A, _, _ = torus_graph(2000, 2)
    w, V = _norm_laplacian_eigs(A)
    ms = [5, 10, 20, 30, 40, 60, 80]
    curves = {
        k: [_pres(A, w, V, np.arange(1, 1 + m), k) for m in ms]
        for k in ("angle", "full", "magnitude")
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(
        ms,
        curves["angle"],
        "o-",
        color="#0088aa",
        lw=2,
        label="angle only (normalized)",
    )
    ax.plot(
        ms, curves["full"], "s--", color="#cc7700", label="full (magnitude x direction)"
    )
    ax.plot(ms, curves["magnitude"], "^:", color="#aa0000", label="magnitude only")
    ax.set_xlabel("number of low modes $m$")
    ax.set_ylabel(r"geodesic preservation $\rho$")
    ax.set_title("Angle carries the geometry; magnitude is throwaway")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig1_core")


def fig2_scaling():
    """The scaling law over the *exact* 20 library manifolds the paper cites.

    Plots the committed reviewer-response data (5 rules x 4 seeds) so the figure
    is guaranteed consistent with section 5.4: slope -0.157, R^2 0.64, Spearman
    -0.881. The shaded band is the RULE-CLUSTERED 95% slope CI [-0.343, -0.094]
    (5 rules are the independent unit, not 20 points), drawn as the envelope of
    the CI-bound slopes pivoting about the data centroid.
    """
    src = os.path.join(
        ROOT, "experiments", "reviewer-response", "todo_experiments_result.json"
    )
    sl = json.load(open(src))["scaling_law"]
    cb = json.load(
        open(
            os.path.join(
                ROOT,
                "experiments",
                "reviewer-response",
                "clustered_bootstrap_result.json",
            )
        )
    )
    P = np.array(sl["points"])
    d, r = P[:, 0], P[:, 1]
    slope, inter = sl["slope"], sl["intercept"]
    r2, sp = sl["r2"], sl["spearman"]
    lo_s, hi_s = cb["cluster_ci95"]  # rule-clustered, [-0.343, -0.094]
    dbar, rbar = d.mean(), r.mean()  # centroid the slope band pivots through

    # flag the one shortcut-contaminated outlier the text calls out (d~1.97, rho 0.39)
    resid = r - (slope * d + inter)
    out = int(np.argmin(resid))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    keep = np.ones(len(d), bool)
    keep[out] = False
    ax.scatter(
        d[keep],
        r[keep],
        c="#0088aa",
        zorder=3,
        label="20 manifolds (5 rules x 4 seeds)",
    )
    ax.scatter(
        d[out],
        r[out],
        c="#aa0000",
        marker="D",
        zorder=4,
        label="shortcut-contaminated\n(small-world screen)",
    )
    xs = np.linspace(d.min(), d.max(), 50)
    ax.plot(
        xs,
        slope * xs + inter,
        "--",
        color="#aa0000",
        label=f"fit: slope ${slope:+.3f}$\n$R^2={r2:.2f}$, Spearman ${sp:+.2f}$",
    )
    ax.fill_between(
        xs,
        lo_s * (xs - dbar) + rbar,
        hi_s * (xs - dbar) + rbar,
        color="#aa0000",
        alpha=0.15,
        label=f"95% rule-clustered\nslope CI [{lo_s:.2f}, {hi_s:.2f}]",
    )
    ax.set_xlabel("emergent dimension (ball/spectral mean)")
    ax.set_ylabel(r"angle-only $\rho$")
    ax.set_title("Angular fidelity falls with emergent dimension")
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, "fig2_scaling")
    print(
        f"fig2 from {os.path.basename(src)}: n={len(d)} slope={slope:+.3f} R2={r2:.2f}"
    )


def fig3_control(n_draws=25):
    A, _, _ = torus_graph(2000, 2)
    w, V = _norm_laplacian_eigs(A)
    ms = [5, 10, 20, 40, 80]
    low = [_pres(A, w, V, np.arange(1, 1 + m), "angle") for m in ms]
    # Random-mode control: average over many draws so a single lucky draw cannot
    # produce a misleading spike. Report mean and the full min-max envelope.
    rnd = np.array(
        [
            [
                _pres(
                    A, w, V, RNG.choice(np.arange(1, len(w)), m, replace=False), "angle"
                )
                for _ in range(n_draws)
            ]
            for m in ms
        ]
    )
    rnd_mean, rnd_lo, rnd_hi = rnd.mean(1), rnd.min(1), rnd.max(1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ms, low, "o-", color="#0088aa", lw=2, label="lowest $m$ modes")
    ax.plot(
        ms,
        rnd_mean,
        "x--",
        color="#aa0000",
        label=f"random $m$ modes (mean of {n_draws})",
    )
    ax.fill_between(
        ms, rnd_lo, rnd_hi, color="#aa0000", alpha=0.15, label="random min--max"
    )
    ax.set_xlabel("number of modes $m$")
    ax.set_ylabel(r"angle-only $\rho$")
    ax.set_title("Geometry lives in the low-eigenvalue subspace")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig3_control")


if __name__ == "__main__":
    fig1_core()
    fig3_control()
    fig2_scaling()
    print("figures ->", OUT)
