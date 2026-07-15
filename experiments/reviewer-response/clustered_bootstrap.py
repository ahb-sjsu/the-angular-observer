"""Rule-clustered bootstrap for the scaling-law slope (referee delta comment 3).

The 20 scaling manifolds are 5 rules x 4 seeds, so the points are CLUSTERED by
rule, not independent. A naive pair/point bootstrap that resamples 20 exchangeable
points understates the slope's uncertainty (effective n at the rule level is 5).
We recompute the OLS slope of angle-rho vs emergent dimension and its 95% CI two
ways for comparison:

  point   : resample the 20 points i.i.d. with replacement (the old CI)
  cluster : two-stage -- resample the 5 RULES with replacement, then the 4 seeds
            within each drawn rule (the honest, clustered CI)

Emits JSON between markers. Deterministic (fixed integer seeds; no wall-clock).
Reuses library.LIB / measure from the-angular-observer root.
"""

from __future__ import annotations
import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
from library import LIB, SEEDS, measure

B = 5000


def collect():
    """Return per-rule arrays of (dim, angle) over the 4 seeds, in LIB order."""
    rules = []
    for name, R in LIB.items():
        rows = measure(name, R)  # one row per seed
        d = np.array([r["dim"] for r in rows], float)
        a = np.array([r["angle"] for r in rows], float)
        rules.append((name, d, a))
    return rules


def slope(d, a):
    return float(np.polyfit(d, a, 1)[0])


def point_boot(d, a, rng, b=B):
    n = len(d)
    out = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        out[i] = slope(d[idx], a[idx])
    return out


def cluster_boot(rules, rng, b=B):
    """Two-stage: resample rules w/ replacement, then seeds within each rule."""
    R = len(rules)
    out = np.empty(b)
    for i in range(b):
        ds, as_ = [], []
        for rj in rng.integers(0, R, R):  # resample rules
            _, d, a = rules[rj]
            k = len(d)
            si = rng.integers(0, k, k)  # resample seeds within the rule
            ds.append(d[si])
            as_.append(a[si])
        dd = np.concatenate(ds)
        aa = np.concatenate(as_)
        out[i] = slope(dd, aa) if np.ptp(dd) > 1e-9 else np.nan
    return out[np.isfinite(out)]


def main():
    rules = collect()
    d = np.concatenate([r[1] for r in rules])
    a = np.concatenate([r[2] for r in rules])
    s0 = slope(d, a)
    rng = np.random.default_rng(20260714)
    pb = point_boot(d, a, rng)
    cb = cluster_boot(rules, rng)
    res = {
        "n_rules": len(rules),
        "n_seeds_per_rule": len(SEEDS),
        "n_points": int(len(d)),
        "slope": round(s0, 4),
        "point_ci95": [
            round(float(np.percentile(pb, 2.5)), 4),
            round(float(np.percentile(pb, 97.5)), 4),
        ],
        "point_se": round(float(pb.std(ddof=1)), 4),
        "cluster_ci95": [
            round(float(np.percentile(cb, 2.5)), 4),
            round(float(np.percentile(cb, 97.5)), 4),
        ],
        "cluster_se": round(float(cb.std(ddof=1)), 4),
        "rule_names": [r[0] for r in rules],
    }
    print("###RESULTS_JSON_START###")
    print(json.dumps(res))
    print("###RESULTS_JSON_END###")
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "clustered_bootstrap_result.json",
        ),
        "w",
    ) as f:
        json.dump(res, f, indent=2)
    print(
        f"slope {s0:+.4f} | point CI {res['point_ci95']} | "
        f"cluster CI {res['cluster_ci95']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
