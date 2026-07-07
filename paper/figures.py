"""Regenerate the paper's figures from the code (PDF for LaTeX, PNG for markdown).

    python paper/figures.py

Fig 1  the core result: angle / full / magnitude vs mode count on a 2-torus.
Fig 2  the scaling law: angle-rho vs emergent dimension across manifolds.
Fig 3  the basis control: low-mode vs random-mode preservation vs mode count.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from rung0_validate import (
    torus_graph,
    _norm_laplacian_eigs,
    dim_ball_growth,
)  # noqa: E402
from rung1b_rewriter import rewrite, largest_component  # noqa: E402
from arity3 import hyper_to_csr  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

RULES = {  # (lhs, rhs, seed) — all arity-3
    "R_frac": (
        [("1", "2", "3")],
        [("1", "4", "6"), ("2", "5", "4"), ("3", "6", "5")],
        [(0, 1, 2)],
    ),
    "R_3D": (
        [("1", "1", "2"), ("3", "4", "1")],
        [("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")],
        [(0, 0, 0), (0, 0, 0)],
    ),
    "R_SR": (
        [("v1", "v2", "v3"), ("v2", "v4", "v5")],
        [("v5", "v6", "v1"), ("v6", "v4", "v2"), ("v4", "v5", "v3")],
        [(1, 2, 3), (2, 4, 5), (4, 6, 7)],
    ),
}


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


def ring(n=2000, k=3):
    ii, jj = [], []
    for i in range(n):
        for s in range(1, k + 1):
            j = (i + s) % n
            ii += [i, j]
            jj += [j, i]
    A = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    A.data[:] = 1.0
    return A


def _crop(A, cap=1600):
    n = A.shape[0]
    if n <= cap:
        return A
    d = shortest_path(A, unweighted=True, indices=[int(RNG.integers(n))])[0]
    keep = np.sort(np.argsort(d)[:cap])
    return largest_component(A[keep][:, keep])


def grow(lhs, rhs, seed, cap=1600):
    tris, nid = rewrite(lhs, rhs, seed, max_edges=2200, max_gen=400, seed=1)
    return _crop(largest_component(hyper_to_csr(tris, nid)), cap)


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
        ms, curves["angle"], "o-", color="#0088aa", lw=2, label="angle only (observer)"
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
    pts = []
    graphs = [
        ("ring", ring()),
        ("2-torus", torus_graph(2000, 2)[0]),
        ("3-torus", torus_graph(3000, 3)[0]),
    ]
    for name, (lhs, rhs, seed) in RULES.items():
        graphs.append((name, grow(lhs, rhs, seed)))
    for name, A in graphs:
        w, V = _norm_laplacian_eigs(A)
        dim = dim_ball_growth(A)
        rho = _pres(A, w, V, np.arange(1, 21), "angle")
        pts.append((dim, rho, name))
    d = np.array([p[0] for p in pts])
    r = np.array([p[1] for p in pts])
    slope, inter = np.polyfit(d, r, 1)
    sp = spearmanr(d, r).statistic
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(d, r, c="#0088aa", zorder=3)
    for di, ri, nm in pts:
        ax.annotate(nm, (di, ri), fontsize=7, xytext=(4, 4), textcoords="offset points")
    xs = np.linspace(d.min(), d.max(), 50)
    ax.plot(
        xs,
        slope * xs + inter,
        "--",
        color="#aa0000",
        label=f"fit: slope={slope:+.3f}/dim\nSpearman={sp:+.2f}",
    )
    ax.set_xlabel("emergent dimension (ball-growth)")
    ax.set_ylabel(r"angle-only $\rho$")
    ax.set_title("Observer fidelity falls with emergent dimension")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, "fig2_scaling")


def fig3_control():
    A, _, _ = torus_graph(2000, 2)
    w, V = _norm_laplacian_eigs(A)
    ms = [5, 10, 20, 40, 80]
    low = [_pres(A, w, V, np.arange(1, 1 + m), "angle") for m in ms]
    rnd = [
        _pres(A, w, V, RNG.choice(np.arange(1, len(w)), m, replace=False), "angle")
        for m in ms
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ms, low, "o-", color="#0088aa", lw=2, label="lowest $m$ modes")
    ax.plot(ms, rnd, "x--", color="#aa0000", label="random $m$ modes (control)")
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
