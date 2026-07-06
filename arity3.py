"""Arity-3 (triangle) hyperedge rewriter — the class that grows 2D+ manifolds.

State is a collection of ordered TRIPLES. rung1b.rewrite/_unify are already
arity-agnostic, so we reuse them; the new pieces are a triangle seed, a
clique-expansion to a graph (each triple {a,b,c} -> edges ab,bc,ca) for the
Laplacian/geometry estimators, and arity-3 rule generation.

Validation: the canonical subdivision rule {a,b,c} -> {a,b,x},{b,c,x},{c,a,x}
(barycentric-style triangle refinement) should emit emergent dim ~= 2.
"""

import numpy as np
from scipy.sparse import csr_matrix
from rung1b_rewriter import rewrite, largest_component
from rung0_validate import (
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
    _norm_laplacian_eigs,
    RNG,
)
from search_harness import angle_rho

# closed simplicial surface seed (tetrahedron faces = 2-sphere topology)
SEED3 = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
EXIST = ["a", "b", "c", "d", "e"]
NEW = ["x", "y", "z"]

CANONICAL = ([("a", "b", "c")], [("a", "b", "x"), ("b", "c", "x"), ("c", "a", "x")])


def hyper_to_csr(triples, n):
    """Clique-expand triples to a graph adjacency (edges = triangle sides)."""
    ii, jj = [], []
    for t in triples:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            ii += [a, b]
            jj += [b, a]
    A = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    A.data[:] = 1.0
    return A


def gen_rule3(rng):
    n_lhs = rng.integers(1, 3)
    pool = EXIST[: rng.integers(3, 6)]
    lhs = [
        tuple(str(t) for t in rng.choice(pool, 3, replace=True)) for _ in range(n_lhs)
    ]
    if n_lhs == 2 and not (set(lhs[0]) & set(lhs[1])):
        return None
    if len({t for e in lhs for t in e}) < 3:
        return None
    new = NEW[: rng.integers(1, 3)]
    tokens = list(set(t for e in lhs for t in e)) + new
    rhs = [
        tuple(str(t) for t in rng.choice(tokens, 3, replace=True))
        for _ in range(rng.integers(2, 5))
    ]
    for v in new:
        if not any(v in e and any(t not in new for t in e) for e in rhs):
            return None
    if not any(t in new for e in rhs for t in e):
        return None
    return lhs, rhs


def score_rule3(lhs, rhs, rng, max_edges=1500):
    try:
        tris, nid = rewrite(lhs, rhs, SEED3, max_edges=max_edges, max_gen=300, seed=1)
        A = largest_component(hyper_to_csr(tris, nid))
        n = A.shape[0]
        if n < 300:
            return {"valid": False, "reason": f"stalled N={n}"}
        w, V = _norm_laplacian_eigs(A)
        db, ds, de = dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)
        spread = float(np.std([db, ds, de]))
        dim = float(np.mean([db, ds]))
        arho = angle_rho(A, w, V, rng=rng)
        gated = 1.5 <= dim <= 4.5
        return {
            "valid": True,
            "N": n,
            "tris": len(tris),
            "ball": db,
            "spec": ds,
            "eff": de,
            "spread": spread,
            "dim": dim,
            "angle_rho": arho,
            "score": (arho - 0.25 * spread) if gated else -1.0,
        }
    except Exception as e:
        return {"valid": False, "reason": type(e).__name__}


if __name__ == "__main__":
    print("arity-3 rewriter — validating the canonical subdivision rule\n")
    m = score_rule3(*CANONICAL, RNG, max_edges=2000)
    print(f"  canonical {CANONICAL[0]} -> {CANONICAL[1]}")
    print(
        f"  N={m['N']} tris={m.get('tris')} | ball={m['ball']:.2f} "
        f"spectral={m['spec']:.2f} eff_rank={m['eff']:.2f} | "
        f"dim={m['dim']:.2f} spread={m['spread']:.2f} angle_rho={m['angle_rho']:.2f}"
    )
    print("  (expect dim ~2 for a refined triangular surface)\n")

    print("mini local search over 25 arity-3 rules:")
    rng = np.random.default_rng(1)
    res = []
    tried = 0
    while len(res) < 25 and tried < 150:
        tried += 1
        r = gen_rule3(rng)
        if r is None:
            continue
        m = score_rule3(r[0], r[1], rng)
        if m.get("valid"):
            m["lhs"], m["rhs"] = r
            res.append(m)
    res.sort(key=lambda d: d["score"], reverse=True)
    print(f"  {'score':>6} {'a_rho':>6} {'spread':>7} {'dim':>5} {'N':>5}  rule")
    for m in res[:8]:
        print(
            f"  {m['score']:6.3f} {m['angle_rho']:6.3f} {m['spread']:7.2f} "
            f"{m['dim']:5.2f} {m['N']:5d}  {m['lhs']} -> {m['rhs']}"
        )
    hi = [m for m in res if m["dim"] >= 2.0 and m["spread"] <= 0.4]
    print(f"\n  rules with dim>=2.0 & spread<=0.4: {len(hi)} / {len(res)}")
