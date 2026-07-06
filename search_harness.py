"""
Rung 2 harness — rule-space search for MANIFOLD-producing hyperedge rules.

Rung 1b showed generic rules make small-world tangles (estimators disagree,
angle-rho ~0.4). To test the observer principle on genuinely emergent geometry
we need rules whose graphs ARE manifolds. This searches rule-space and scores
each rule by two validated signals:

  spread     = std(ball, spectral, eff_rank)     cheap manifold filter (low=good)
  angle_rho  = geodesic preservation of the unit-normalized (PolarQuant) low-mode
               embedding                          direct manifold quality (high=good)

Composite: manifold_score = angle_rho - 0.25*spread, gated to emergent
dimension in [1.5, 4.5] and a graph that actually grew.

Designed to run locally on a small batch first; on NRP the same score_rule()
fans out over thousands of rules (one pod per seed-block, A10x8), max_edges
raised, GPU sparse eigensolves swapped in past ~5k nodes.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from rung1b_rewriter import rewrite, edges_to_csr, largest_component
from rung0_validate import (
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
    _norm_laplacian_eigs,
)

SEED_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # K4 seed
EXIST = ["a", "b", "c", "d"]
NEW = ["x", "y", "z"]


# --------------------------------------------------------------- rule generation
def gen_rule(rng):
    """Random arity-2 rule. Returns (lhs, rhs) or None if malformed."""
    n_lhs = rng.integers(1, 3)  # 1 or 2 LHS edges
    pool = EXIST[: rng.integers(2, 5)]  # 2-4 existing vars
    lhs = [
        tuple(str(t) for t in rng.choice(pool, 2, replace=True)) for _ in range(n_lhs)
    ]
    if n_lhs == 2 and not (set(lhs[0]) & set(lhs[1])):
        return None  # 2-edge LHS must share a var
    if len({t for e in lhs for t in e}) < 2:
        return None

    n_new = rng.integers(1, 3)  # 1-2 new nodes -> growth
    new = NEW[:n_new]
    n_rhs = rng.integers(2, 5)
    tokens = list(set(t for e in lhs for t in e)) + new
    rhs = [
        tuple(str(t) for t in rng.choice(tokens, 2, replace=True)) for _ in range(n_rhs)
    ]

    # every new var must connect to an existing var (no floating components)
    for v in new:
        if not any(v in e and any(t not in new for t in e) for e in rhs):
            return None
    if not any(t in new for e in rhs for t in e):
        return None  # RHS must actually add nodes
    return lhs, rhs


# ------------------------------------------------------------------- scoring
def angle_rho(A, w, V, m=20, n_anchor=150, rng=None):
    """PolarQuant angle-only geodesic preservation (manifold quality)."""
    n = A.shape[0]
    anchors = rng.choice(n, size=min(n_anchor, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    Y = V[:, 1 : 1 + m] / np.sqrt(np.maximum(w[1 : 1 + m], 1e-9))
    Y = Y[anchors]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)  # angle
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    if np.ptp(d) < 1e-9 or np.ptp(G[iu]) < 1e-9:  # degenerate -> no geometry
        return 0.0
    r = spearmanr(d, G[iu]).statistic
    return abs(r) if np.isfinite(r) else 0.0


def score_rule(lhs, rhs, rng, max_edges=1500):
    try:
        edges, nid = rewrite(
            lhs, rhs, SEED_EDGES, max_edges=max_edges, max_gen=300, seed=1
        )
        A = largest_component(edges_to_csr(edges, nid))
        n = A.shape[0]
        if n < 300:
            return {"valid": False, "reason": f"stalled N={n}"}
        w, V = _norm_laplacian_eigs(A)
        db, ds, de = dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)
        spread = float(np.std([db, ds, de]))
        dim_mean = float(np.mean([db, ds]))  # eff_rank noisy; use ball+spec
        arho = angle_rho(A, w, V, rng=rng)
        gated = 1.5 <= dim_mean <= 4.5
        score = (arho - 0.25 * spread) if gated else -1.0
        return {
            "valid": True,
            "N": n,
            "ball": db,
            "spec": ds,
            "eff": de,
            "spread": spread,
            "dim": dim_mean,
            "angle_rho": arho,
            "score": score,
        }
    except Exception as e:
        return {"valid": False, "reason": type(e).__name__}


# ------------------------------------------------------------------- search
def search(n_rules=40, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    tried = 0
    while len(results) < n_rules and tried < n_rules * 6:
        tried += 1
        r = gen_rule(rng)
        if r is None:
            continue
        lhs, rhs = r
        m = score_rule(lhs, rhs, rng)
        if m.get("valid"):
            m["lhs"], m["rhs"] = lhs, rhs
            results.append(m)

    results.sort(
        key=lambda d: d["score"] if np.isfinite(d["score"]) else -1e9, reverse=True
    )
    print(f"scored {len(results)} valid rules (of {tried} drawn)\n")
    print(
        f"  {'score':>6} {'angle_rho':>9} {'spread':>7} {'dim':>5} "
        f"{'N':>5}   rule (lhs -> rhs)"
    )
    for m in results[:10]:
        rule = f"{list(m['lhs'])} -> {list(m['rhs'])}"
        print(
            f"  {m['score']:6.3f} {m['angle_rho']:9.3f} {m['spread']:7.2f} "
            f"{m['dim']:5.2f} {m['N']:5d}   {rule}"
        )
    best = results[0] if results else None
    if best:
        print(
            f"\nbest manifold rule: angle_rho={best['angle_rho']:.3f} "
            f"(torus reference ~0.93, tangle ~0.40), dim={best['dim']:.2f}"
        )
    return results


if __name__ == "__main__":
    print("Rung 2 harness — local manifold-rule search\n")
    search(n_rules=40, seed=0)
