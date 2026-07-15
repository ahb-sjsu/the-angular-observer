"""Multi-seed growth sweep: replicate growth_sweep.py across rewriter seeds.

The growth axis is the only meaningful replication axis for R_3D (whose
growth is deterministic in the seed, seed_sweep.py) and the axis on which
"dimension emerges before shape" was claimed from a single seed. Here:
3 rules x seeds 1-4 x budgets 750-12,000, the stage battery of
growth_sweep.stage at each point.

Partitioned for wall-clock: `run` computes a subset and writes a partial
JSON; `merge` combines partials, prints per-(rule,budget) across-seed stats,
and draws the banded figure (mean line + min-max band per template; a
zero-width band is a visual determinism check).

  PYTHONPATH=. python experiments/manifold-recovery/multiseed_growth.py run RULE_IDX SEEDS OUT
      e.g. run 0 1,2 part_r0_s12     (RULE_IDX indexes growth_sweep.RULES)
  PYTHONPATH=. python experiments/manifold-recovery/multiseed_growth.py merge

Deterministic. CPU-only, dense eigh (n<=2000).
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arity3 import hyper_to_csr                      # noqa: E402
from library import LIB, crop                        # noqa: E402
from rung1b_rewriter import largest_component, rewrite  # noqa: E402
from growth_sweep import BUDGETS, RULES, stage       # noqa: E402
from recover import TEMPLATES                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(rule_idx, seeds, out):
    rule = RULES[rule_idx]
    R = LIB[rule]
    rows = []
    for s in seeds:
        for B in BUDGETS:
            tris, nid = rewrite(R["lhs"], R["rhs"], R["seed"],
                                max_edges=B, max_gen=2 * B, seed=s)
            A = largest_component(hyper_to_csr(tris, nid))
            if A.shape[0] < 60:
                print(f"{rule} s={s} B={B}: too small, skip", flush=True)
                continue
            A = crop(A)
            rng = np.random.default_rng(7)
            row = dict(rule=rule.strip(), seed=s, budget=B, **stage(A, rng))
            rows.append(row)
            print(f"{rule:18s} s={s} B={B:5d} n={row['n']:4d} "
                  f"d={row['dim']:.2f} angle={row['angle_rho']:.2f} "
                  f"sig={row['signature_best']}", flush=True)
    path = os.path.join(HERE, f"growth_multiseed_{out}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {path}")


def merge():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "growth_multiseed_part_*.json"))):
        rows += json.load(open(p))
    by = {}
    for r in rows:
        by.setdefault(r["rule"], {}).setdefault(r["budget"], []).append(r)

    summary = {}
    for rule, budgets in by.items():
        summary[rule] = []
        for B in sorted(budgets):
            rs = budgets[B]
            sigs = [r["signature_best"] for r in rs]
            d = [r["dim"] for r in rs]
            a = [r["angle_rho"] for r in rs]
            det = len({tuple(r["spectrum_ratios"]) for r in rs}) == 1 and len(rs) > 1
            summary[rule].append(dict(
                budget=B, n_seeds=len(rs),
                dim_mean=round(float(np.mean(d)), 2),
                dim_sd=round(float(np.std(d)), 2),
                angle_mean=round(float(np.mean(a)), 2),
                angle_sd=round(float(np.std(a)), 2),
                verdicts=sigs,
                stable=len(set(sigs)) == 1,
                deterministic=det,
                scores={t: [round(float(x), 3) for x in
                            (min(v), np.mean(v), max(v))]
                        for t in TEMPLATES
                        for v in [[r["signature_scores"][t] for r in rs]]},
            ))
            print(f"{rule:18s} B={B:5d} d={np.mean(d):.2f}±{np.std(d):.2f} "
                  f"angle={np.mean(a):.2f}±{np.std(a):.2f} "
                  f"verdicts={sigs}"
                  f"{' DETERMINISTIC' if det else ''}", flush=True)

    with open(os.path.join(HERE, "growth_multiseed_result.json"), "w") as f:
        json.dump(dict(rows=rows, summary=summary), f, indent=1)
    make_figure(summary)
    print("merged", len(rows), "rows")


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shades = {"interval (1D)": 0.85, "square patch": 0.55,
              "flat 2-torus": 0.3, "sphere S^2": 0.3}
    styles = {"interval (1D)": "-", "square patch": "-",
              "flat 2-torus": ":", "sphere S^2": ":"}
    fig, axes = plt.subplots(1, len(summary), figsize=(4.0 * len(summary), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (rule, rows) in zip(axes, summary.items()):
        B = [r["budget"] for r in rows]
        for name in TEMPLATES:
            lo = [r["scores"][name][0] for r in rows]
            mu = [r["scores"][name][1] for r in rows]
            hi = [r["scores"][name][2] for r in rows]
            c = plt.cm.Blues(shades[name])
            ax.fill_between(B, lo, hi, color=c, alpha=0.25, lw=0)
            ax.plot(B, mu, styles[name], color=c,
                    lw=1.8 if styles[name] == "-" else 1.1,
                    marker="o", ms=3, markeredgecolor="#1f3a5f",
                    markeredgewidth=0.3)
            ax.annotate(name, (B[-1] * 1.03, mu[-1]), fontsize=6.5,
                        color="#555555", va="center")
        for r in rows:
            ax.annotate(f"d={r['dim_mean']:.1f}", (r["budget"], -0.16),
                        fontsize=6.5, color="#888888", ha="center",
                        annotation_clip=False)
        ax.set_xscale("log")
        ax.set_title(f"{rule}  ({rows[0]['n_seeds']} seeds)", fontsize=9,
                     color="#333333")
        ax.set_xlabel("rewrite budget (edges)", fontsize=8)
        ax.grid(color="#eeeeee", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#dddddd")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("signature distance (lower = closer)", fontsize=8)
    fig.suptitle("Template fit vs growth, across rewriter seeds "
                 "(band = min-max over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "growth_multiseed.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")],
            sys.argv[4])
    else:
        merge()
