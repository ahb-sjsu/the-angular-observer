"""
Rung 1b — Faithful hyperedge rewriter (no latent coordinates).

Removes the proxy's crutch: here dimension is TRULY emergent from combinatorial
rewriting, exactly as in the Wolfram model. State is a collection of ordered
edges; a rule rewrites a small matched sub-pattern into a replacement, minting
fresh nodes for RHS-only variables. We then run the SAME three estimators and
the SAME low-vs-random coarse-graining test on the resulting graph.

We do NOT assert a rule's dimension a priori -- we measure whatever the
estimators discover. Many Wolfram rules yield non-manifold (small-world /
tree-like) growth; that is a real feature, and we report it honestly.

Matcher supports LHS of 1-2 edges with variable unification, standard
generational (non-overlapping) updating.
"""

import numpy as np
from scipy.sparse import csr_matrix
from rung0_validate import (
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
    _norm_laplacian_eigs,
    RNG,
)
from rung1_proxy import geodesic_preservation


def _unify(edge, template, binding):
    """Unify one edge (tuple of ints) against a template (tuple of var names)."""
    if len(edge) != len(template):
        return None
    b = dict(binding)
    for tok, node in zip(template, edge):
        if tok in b:
            if b[tok] != node:
                return None
        else:
            b[tok] = node
    return b


def rewrite(lhs, rhs, init_edges, max_edges=3000, max_gen=400, seed=1):
    rng = np.random.default_rng(seed)
    edges = [tuple(e) for e in init_edges]
    next_id = max(n for e in edges for n in e) + 1

    for _ in range(max_gen):
        if len(edges) >= max_edges:
            break
        E = len(edges)
        order = rng.permutation(E)
        used, matches = set(), []

        for i in order:
            if i in used:
                continue
            b1 = _unify(edges[i], lhs[0], {})
            if b1 is None:
                continue
            if len(lhs) == 1:
                matches.append(([i], b1))
                used.add(i)
                continue
            for j in order:  # second edge, sharing vars
                if j == i or j in used:
                    continue
                b2 = _unify(edges[j], lhs[1], b1)
                if b2 is not None:
                    matches.append(([i, j], b2))
                    used.add(i)
                    used.add(j)
                    break

        if not matches:
            break

        new_edges = [e for k, e in enumerate(edges) if k not in used]
        for _idx, b in matches:
            local = {}
            for tmpl in rhs:
                e = []
                for tok in tmpl:
                    if tok in b:
                        e.append(b[tok])
                    else:
                        if tok not in local:
                            local[tok] = next_id
                            next_id += 1
                        e.append(local[tok])
                new_edges.append(tuple(e))
        edges = new_edges

    return edges, next_id


def edges_to_csr(edges, n):
    ii = [e[0] for e in edges] + [e[1] for e in edges]
    jj = [e[1] for e in edges] + [e[0] for e in edges]
    A = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    A.data[:] = 1.0
    return A


def largest_component(A):
    from scipy.sparse.csgraph import connected_components

    k, lab = connected_components(A, directed=False)
    if k == 1:
        return A
    keep = np.where(lab == np.bincount(lab).argmax())[0]
    return A[keep][:, keep]


RULES = {
    # two edges sharing a node -> mesh with a bridging new node (loop-forming)
    "mesh": (
        [("a", "b"), ("a", "c")],
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("a", "d")],
    ),
    # single-edge subdivision with a shortcut (tree-ish / small-world)
    "subdiv": (
        [("a", "b")],
        [("a", "c"), ("c", "b"), ("a", "b")],
    ),
}


def study(name, lhs, rhs):
    edges, nid = rewrite(lhs, rhs, [(0, 1), (0, 2), (1, 2)], max_edges=3000)
    A = largest_component(edges_to_csr(edges, nid))
    n = A.shape[0]
    w, V = _norm_laplacian_eigs(A)
    db, ds, de = dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)
    deg = np.asarray(A.sum(1)).ravel().mean()
    print(
        f"rule={name:7s} | N={n:5d} E={len(edges):5d} <deg>={deg:4.1f} | "
        f"emergent d: ball={db:5.2f} spectral={ds:5.2f} eff_rank={de:5.2f}"
    )

    if n >= 400:  # coarse-graining test if big enough
        anchors = RNG.choice(n, size=min(250, n), replace=False)
        print("           coarse-graining  rho(low)/rho(random):", end="")
        for m in (5, 20, 80):
            lo, rd = geodesic_preservation(A, w, V, m, anchors)
            print(f"  m={m}:{lo:.2f}/{rd:.2f}", end="")
        print()


if __name__ == "__main__":
    print("Rung 1b — faithful hyperedge rewriter, emergent dimension\n")
    for name, (lhs, rhs) in RULES.items():
        study(name, lhs, rhs)
    print(
        "\n(dimension is measured, not assumed; small-world rules give "
        "high/ill-defined d by design)"
    )
