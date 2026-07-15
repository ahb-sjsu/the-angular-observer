"""Scaled-m continuum refinement (review-2's decisive test).

The §5.7 dissociation held m=10 fixed while N grew, so the Wolfram substrate's
decay could be a fixed-observer-budget artifact: as N grows the mode budget
relative to the substrate shrinks. Here we rerun the refinement with m
SPECTRAL-GAP-MATCHED --- for each substrate we fix the eigenvalue cutoff tau to
the m=10 gap at the smallest N, then take m(N) = #{k : lambda_k <= tau}, so the
observer resolves the SAME spectral scale at every N. If wolf2 survives with
scaled m, the dissociation is a statement about fixed-budget observers; if it
still decays, the anomaly is real.

Reuses rung4_pod builders. Local, CPU. JSON between markers.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from rung4_pod import build, low_band, N_ANCHOR

K_BAND = 160
NS = [2000, 5000, 10000]


def angle_rho_at(scaled, lam, A, m, seed):
    n = A.shape[0]
    rng = np.random.default_rng(1000 + seed)
    anc = rng.choice(n, size=min(N_ANCHOR, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    g = G[iu]; ok = np.isfinite(g)
    low = scaled[:, :m]
    lown = low / np.maximum(np.linalg.norm(low, axis=1, keepdims=True), 1e-12)
    Ya = lown[anc]
    d = np.sqrt(((Ya[iu[0]] - Ya[iu[1]]) ** 2).sum(1))[ok]
    r = spearmanr(d, g[ok]).statistic
    return abs(r) if np.isfinite(r) else 0.0


def run_substrate(name, seed=1):
    tau = None
    rows = []
    for N in NS:
        try:
            A = build(name, N, seed)
            lam, V, _ = low_band(A, K_BAND)
            scaled = V / np.sqrt(np.maximum(lam, 1e-9))
            if tau is None:
                tau = float(lam[9])                       # m=10 gap at smallest N
            m_scaled = int(np.searchsorted(lam, tau) + 1)
            m_scaled = max(10, min(m_scaled, scaled.shape[1]))
            rows.append(dict(
                substrate=name, n_target=N, n_lcc=int(A.shape[0]),
                tau=round(tau, 5), m_fixed=10, m_scaled=m_scaled,
                angle_fixed=round(angle_rho_at(scaled, lam, A, 10, seed), 3),
                angle_scaled=round(angle_rho_at(scaled, lam, A, m_scaled, seed), 3),
            ))
            print(f"[{name} N={N}] m_scaled={m_scaled} "
                  f"fixed={rows[-1]['angle_fixed']} scaled={rows[-1]['angle_scaled']}",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(substrate=name, n_target=N, error=str(exc)))
            print(f"[{name} N={N}] ERR {exc}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    res = {}
    for sub in ["rgg3", "wolf2"]:
        res[sub] = run_substrate(sub, seed=1)
    # trend: Spearman of angle vs log N, fixed vs scaled
    for sub, rows in list(res.items()):
        good = [r for r in rows if isinstance(r, dict) and "angle_fixed" in r]
        if len(good) >= 3:
            ln = np.log([r["n_lcc"] for r in good])
            sf = spearmanr(ln, [r["angle_fixed"] for r in good]).statistic
            ss = spearmanr(ln, [r["angle_scaled"] for r in good]).statistic
            res[sub + "_trend"] = dict(spearman_fixed_vs_logN=round(float(sf), 3),
                                       spearman_scaled_vs_logN=round(float(ss), 3))
    with open("rung4_scaledm_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("###RESULTS_JSON_START###"); print(json.dumps(res)); print("###RESULTS_JSON_END###")
