"""Regenerate fig2 as the 20 library manifolds (5 rules x 4 seeds) with per-point
pair-bootstrap 95% CIs and a regression confidence band, so the figure shows the
exact data behind the section 5.4 fit (slope -0.157 +/- 0.028, R^2 = 0.64)."""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from scipy.stats import spearmanr, t as tdist  # noqa: E402

from library import LIB, crop, SEEDS, M  # noqa: E402
from rung1b_rewriter import rewrite, largest_component  # noqa: E402
from arity3 import hyper_to_csr  # noqa: E402
from rung0_validate import (  # noqa: E402
    _norm_laplacian_eigs,
    dim_ball_growth,
    dim_spectral,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")


def angle_pairs(A, w, V, rng, n_anchor=150):
    anc = rng.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = D[iu]
    Y = (V[:, np.arange(1, 1 + M)] / np.sqrt(np.maximum(w[1 : 1 + M], 1e-9)))[anc]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    return d, g


def boot_ci(d, g, rng, b=1000):
    idx = np.arange(len(d))
    st = np.empty(b)
    for i in range(b):
        s = rng.choice(idx, len(d), replace=True)
        st[i] = spearmanr(d[s], g[s]).statistic
    return np.percentile(np.abs(st), [2.5, 97.5])


pts = []  # (dim, rho, lo, hi)
for ri, (name, R) in enumerate(LIB.items()):
    for s in SEEDS:
        tris, nid = rewrite(
            R["lhs"], R["rhs"], R["seed"], max_edges=3000, max_gen=500, seed=s
        )
        A = largest_component(hyper_to_csr(tris, nid))
        if A.shape[0] < 200:
            continue
        A = crop(A)
        w, V = _norm_laplacian_eigs(A)
        rng = np.random.default_rng(1000 + ri * 10 + s)
        dim = float(np.mean([dim_ball_growth(A), dim_spectral(w)]))
        d, g = angle_pairs(A, w, V, rng)
        rho = abs(spearmanr(d, g).statistic)
        lo, hi = boot_ci(d, g, rng)
        pts.append((dim, rho, lo, hi))
        print(
            f"{name:16s} seed{s}: dim={dim:.2f} rho={rho:.3f} CI[{lo:.3f},{hi:.3f}]",
            flush=True,
        )

P = np.array(pts)
x, y, lo, hi = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
(slope, inter), cov = np.polyfit(x, y, 1, cov=True)
se = float(np.sqrt(cov[0, 0]))
yhat = slope * x + inter
r2 = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())

# regression 95% mean-response confidence band
n = len(x)
xbar = x.mean()
sxx = ((x - xbar) ** 2).sum()
sres = np.sqrt(((y - yhat) ** 2).sum() / (n - 2))
tv = tdist.ppf(0.975, n - 2)
xg = np.linspace(x.min(), x.max(), 120)
yg = slope * xg + inter
band = tv * sres * np.sqrt(1.0 / n + (xg - xbar) ** 2 / sxx)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.errorbar(
    x,
    y,
    yerr=[y - lo, hi - y],
    fmt="o",
    color="#0088aa",
    ms=4,
    capsize=2,
    lw=1,
    label="manifolds (95% CI, pair bootstrap)",
)
ax.plot(
    xg,
    yg,
    "--",
    color="#aa0000",
    label=f"fit: slope ${slope:+.3f}\\pm{se:.3f}$, $R^2={r2:.2f}$",
)
ax.fill_between(
    xg, yg - band, yg + band, color="#aa0000", alpha=0.15, label="95% confidence band"
)
ax.set_xlabel("emergent dimension (ball/spectral mean)")
ax.set_ylabel(r"angle-only $\rho$")
ax.set_title("Angular fidelity vs emergent dimension (20 manifolds)")
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=7, loc="upper right")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig2_scaling.{ext}"), bbox_inches="tight", dpi=150)
print(
    f"\nfit: angle_rho ~ {inter:+.3f} {slope:+.3f}*dim  SE={se:.3f}  R2={r2:.3f}  n={n}"
)
print("wrote fig2_scaling.{pdf,png}")
