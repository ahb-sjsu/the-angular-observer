"""Seed sweep: firm up the per-rule recognizer verdicts of recover.py.

recover.py ran one seed per rule; the library shows seed-to-seed dimension
variation, so each verdict needs a stability check. Here: all 5 rules x
rewriter seeds 1-4, full recognition battery per instance, at the full
3000-edge budget (generation cap scaled with the budget -- see the gotcha
fixed in growth_sweep.py: these rules add ~1 node/generation, so the
library's fixed max_gen=500 silently capped growth at ~500 nodes).

Run from repo root: PYTHONPATH=. python experiments/manifold-recovery/seed_sweep.py
Deterministic. CPU-only, dense eigh (n<=2000).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from arity3 import hyper_to_csr
from library import LIB, crop
from rung1b_rewriter import largest_component, rewrite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import measure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET, SEEDS = 3000, (1, 2, 3, 4)


def main():
    out = {}
    for name, R in LIB.items():
        rows = []
        for s in SEEDS:
            tris, nid = rewrite(R["lhs"], R["rhs"], R["seed"],
                                max_edges=BUDGET, max_gen=2 * BUDGET, seed=s)
            A = largest_component(hyper_to_csr(tris, nid))
            if A.shape[0] < 200:
                print(f"{name} s={s}: component too small ({A.shape[0]}), skip",
                      flush=True)
                continue
            A = crop(A)
            rng = np.random.default_rng(7)
            res, _, _ = measure(f"{name.strip()} [s={s}]", A, rng)
            res["seed"] = s
            rows.append(res)
            print(f"{name:18s} s={s} n={res['n']:4d} degCV={res['deg_cv']:.2f} "
                  f"d={res['dim']:.2f}(±{res['dim_spread']:.2f}) "
                  f"angle={res['angle_rho']:.2f} a1={res['angle_rho_alpha1']:.2f} "
                  f"mag={res['magnitude_rho']:.2f} ecc±={res['ecc_spread']:.2f} "
                  f"sig={res['signature_best']}", flush=True)
        out[name.strip()] = rows
        if rows:
            sigs = [r["signature_best"] for r in rows]
            d = [r["dim"] for r in rows]
            a = [r["angle_rho"] for r in rows]
            da1 = [r["angle_rho_alpha1"] - r["angle_rho"] for r in rows]
            print(f"  -> {name.strip()}: d={np.mean(d):.2f}±{np.std(d):.2f}  "
                  f"angle={np.mean(a):.2f}±{np.std(a):.2f}  "
                  f"alpha1 delta={np.mean(da1):+.3f}  "
                  f"verdicts={sigs} "
                  f"({'STABLE' if len(set(sigs)) == 1 else 'MIXED'})", flush=True)

    with open(os.path.join(HERE, "seed_sweep_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("###RESULTS_JSON_START###")
    print(json.dumps(out, indent=1))
    print("###RESULTS_JSON_END###")


if __name__ == "__main__":
    main()
