import sys, json, numpy as np, statistics as st

sys.path.insert(0, "../..")
sys.path.insert(0, ".")
import rung0_validate as rv
from reviewer2_experiments import truncation_and_distortion

keys_bilip = ["angle", "full", "magnitude", "random"]
acc = {f"kappa_{k}": [] for k in keys_bilip}
acc.update({f"proc_{k}": [] for k in ["angle", "random"]})
for s in range(5):
    rv.RNG = np.random.default_rng(5000 + s)  # same draws as the ablation
    out = truncation_and_distortion(n=2000, seed=s)
    for k in keys_bilip:
        acc[f"kappa_{k}"].append(out["bilipschitz"][k]["kappa_robust"])
    for k in ["angle", "random"]:
        acc[f"proc_{k}"].append(out["procrustes_disparity_vs_torus_R4"][k])
res = {}
for k, v in acc.items():
    res[k] = dict(
        mean=round(st.mean(v), 3),
        sd=round(st.pstdev(v), 3),
        min=round(min(v), 3),
        max=round(max(v), 3),
        draws=v,
        n_draws=5,
    )
    print(f"{k:14s}: mean={res[k]['mean']:.3f} sd={res[k]['sd']:.3f}  {v}")
json.dump(res, open("kappa_draws_result.json", "w"), indent=2)
print("saved kappa_draws_result.json")
