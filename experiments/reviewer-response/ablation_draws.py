import sys, numpy as np, statistics as st

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/reviewer-response")
import rung0_validate as rv
from ablation_experiments import substrate


def collect(dim, n, seeds):
    acc = {
        ("plain", 10): [],
        ("plain", 20): [],
        ("commute", 10): [],
        ("commute", 20): [],
    }
    for s in seeds:
        rv.RNG = np.random.default_rng(5000 + s)
        A = rv.torus_graph(n, dim)[0]
        out = substrate(f"{dim}-torus", A, seed=s)
        for w in ("plain", "commute"):
            for m in (10, 20):
                acc[(w, m)].append(out[f"m{m}"][w]["angle_rho"])
    return acc


for dim, n in [(2, 2000), (3, 3000)]:
    acc = collect(dim, n, range(5))
    print(f"=== {dim}-torus (5 draws) ===")
    for w in ("plain", "commute"):
        for m in (10, 20):
            v = acc[(w, m)]
            print(
                f"  {w:8s} m={m}: mean={st.mean(v):.3f} sd={st.pstdev(v):.3f} min={min(v):.3f} max={max(v):.3f}  {['%.3f'%x for x in v]}"
            )
