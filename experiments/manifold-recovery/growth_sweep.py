"""Growth sweep: watch emergent dimension arrive under the angular observer.

recover.py showed that at a 3000-edge growth budget R_2D reads as an INTERVAL
(1D filament, spectrum ratios ~ k^2) while R_3D reads as a bounded 2D patch.
Question: is "R_2D is a curve" a fact about the rule or about the growth
stage? Sweep the rewrite budget and track, per stage:
  - n (largest component, cropped at 2000), measured dimension + spread,
  - angle-rho fidelity,
  - the multiplet-signature scores (interval vs square patch vs closed
    templates) -- a signature CROSSING is dimension emerging in real time.

Run from repo root: PYTHONPATH=. python experiments/manifold-recovery/growth_sweep.py
Deterministic (fixed seeds). CPU-only, dense eigh (n<=2000).
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
)
from rung1b_rewriter import largest_component, rewrite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import TEMPLATES, angular_rows, signature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M, N_ANCHOR = 20, 150
BUDGETS = [750, 1500, 3000, 6000, 12000]
RULES = ["R_2D  (borrowed)", "R_3D  (borrowed)", "evolved-2D (#2)"]


def stage(A, rng):
    n = A.shape[0]
    w, V = _norm_laplacian_eigs(A)
    anc = rng.choice(n, size=min(N_ANCHOR, n), replace=False)
    Dg = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    ok = np.isfinite(Dg[iu])
    U, _ = angular_rows(w, V, np.arange(1, 1 + min(M, len(w) - 1)))
    d_ang = np.sqrt(((U[anc][iu[0]] - U[anc][iu[1]]) ** 2).sum(-1))[ok]
    ecc = np.array([row[np.isfinite(row)].max() for row in Dg])
    dims = [dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)]
    lam, scores, best = signature(w)
    return dict(
        n=int(n),
        dim=round(float(np.mean(dims[:2])), 2),
        dim_spread=round(float(np.std(dims)), 2),
        angle_rho=round(abs(spearmanr(d_ang, Dg[iu][ok]).statistic), 3),
        ecc_spread=round(float((ecc.max() - ecc.min()) / ecc.mean()), 3),
        spectrum_ratios=lam,
        signature_scores={k: round(v, 3) for k, v in scores.items()},
        signature_best=best,
    )


def main():
    out = {}
    for rule in RULES:
        R = LIB[rule]
        rows = []
        for B in BUDGETS:
            # max_gen must scale with the budget: these rules add ~1 node/gen,
            # so a fixed cap silently freezes growth above ~max_gen edges.
            tris, nid = rewrite(R["lhs"], R["rhs"], R["seed"],
                                max_edges=B, max_gen=2 * B, seed=1)
            A = largest_component(hyper_to_csr(tris, nid))
            if A.shape[0] < 60:
                print(f"{rule} @ {B}: component too small ({A.shape[0]}), skip",
                      flush=True)
                continue
            A = crop(A)
            rng = np.random.default_rng(7)
            row = dict(budget=B, **stage(A, rng))
            rows.append(row)
            print(f"{rule:18s} B={B:5d} n={row['n']:4d} "
                  f"d={row['dim']:.2f}(±{row['dim_spread']:.2f}) "
                  f"angle={row['angle_rho']:.2f} ecc±={row['ecc_spread']:.2f} "
                  f"sig={row['signature_best']:14s} "
                  f"int={row['signature_scores']['interval (1D)']:.2f} "
                  f"patch={row['signature_scores']['square patch']:.2f}",
                  flush=True)
        out[rule.strip()] = rows

    with open(os.path.join(HERE, "growth_sweep_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    make_figure(out)
    print("###RESULTS_JSON_START###")
    print(json.dumps(out, indent=1))
    print("###RESULTS_JSON_END###")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(out), figsize=(4.0 * len(out), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    shades = {"interval (1D)": 0.85, "square patch": 0.55,
              "flat 2-torus": 0.3, "sphere S^2": 0.3}
    styles = {"interval (1D)": "-", "square patch": "-",
              "flat 2-torus": ":", "sphere S^2": ":"}
    for ax, (rule, rows) in zip(axes, out.items()):
        B = [r["budget"] for r in rows]
        for name in TEMPLATES:
            y = [r["signature_scores"][name] for r in rows]
            ax.plot(B, y, styles[name], color=plt.cm.Blues(shades[name]),
                    lw=1.8 if styles[name] == "-" else 1.1, marker="o", ms=3.5,
                    markeredgecolor="#1f3a5f", markeredgewidth=0.3)
            ax.annotate(name, (B[-1] * 1.03, y[-1]), fontsize=6.5,
                        color="#555555", va="center")
        for r in rows:
            ax.annotate(f"d={r['dim']:.1f}", (r["budget"], -0.16), fontsize=6.5,
                        color="#888888", ha="center", annotation_clip=False)
        ax.set_xscale("log")
        ax.set_title(rule, fontsize=9, color="#333333")
        ax.set_xlabel("rewrite budget (edges)", fontsize=8)
        ax.grid(color="#eeeeee", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#dddddd")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("signature distance (lower = closer)", fontsize=8)
    fig.suptitle("Which manifold template fits, as the substrate grows",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "growth_signatures.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
