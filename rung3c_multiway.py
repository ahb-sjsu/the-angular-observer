"""
Rung 3c — multiway / branchial substrate transfer (PREREG_RUNG3.md).

Question: does angle-only low-mode coarse-graining preserve geodesic rank on
(a) a string-rewrite MULTIWAY states graph and (b) a BRANCHIAL slice, the way
it does on Riemannian substrates — or do these join the Lorentzian causal set
as a correct failure?

Pre-registered classification (library.py conventions):
  transfer          angle_rho >= 0.80 and estimator spread <= 0.30
  partial           0.50 <= angle_rho < 0.80
  correct failure   angle_rho < 0.50   (reported as a negative — an
                    informative boundary of the observer principle)

Explicitly NOT claimed at any outcome: any quantum-phase reading. The angular
coordinate here is a real unit direction of an eigenvector row, not a U(1)
amplitude phase; the Born-rule observer keeps magnitude and discards phase —
the opposite compression.

Substrates: two canonical-style string rewrite systems evolved as multiway
systems (every rule applied at every position of every state). States graph =
distinct strings, undirected edge per one-step derivation. Branchial slice at
step t = states first produced at step t, linked iff they share a parent.
"""

import json
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from curvature import largest_component_keep
from rung0_validate import (
    RNG,
    _norm_laplacian_eigs,
    dim_ball_growth,
    dim_effective_rank,
    dim_spectral,
)
from search_harness import angle_rho

MAX_LEN = 14  # drop strings longer than this (state-space cap)
MAX_STATES = 3500
SYSTEMS = {
    "MW1 {A->AB, B->A} from A": ({"A": "AB", "B": "A"}, "A"),
    "MW2 {A->AB, BB->A} from A": ({"A": "AB", "BB": "A"}, "A"),
}


def successors(s, rules):
    out = set()
    for lhs, rhs in rules.items():
        start = 0
        while True:
            i = s.find(lhs, start)
            if i < 0:
                break
            t = s[:i] + rhs + s[i + len(lhs) :]
            if len(t) <= MAX_LEN:
                out.add(t)
            start = i + 1
    return out


def multiway(rules, init):
    """Breadth-first multiway evolution. Returns (states graph edges as id
    pairs, id map, per-generation first-reach layers, parent map)."""
    ids = {init: 0}
    layers = [[init]]
    parents = defaultdict(set)
    edges = set()
    frontier = [init]
    while frontier and len(ids) < MAX_STATES:
        nxt = []
        for s in frontier:
            for t in successors(s, rules):
                if t not in ids:
                    if len(ids) >= MAX_STATES:
                        continue
                    ids[t] = len(ids)
                    nxt.append(t)
                edges.add((min(ids[s], ids[t]), max(ids[s], ids[t])))
                parents[t].add(s)
        if nxt:
            layers.append(nxt)
        frontier = nxt
    return edges, ids, layers, parents


def pairs_to_csr(pairs, n):
    pairs = [(a, b) for a, b in pairs if a != b]
    i = np.array([a for a, b in pairs] + [b for a, b in pairs])
    j = np.array([b for a, b in pairs] + [a for a, b in pairs])
    A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1.0
    return A


def branchial_slice(layers, parents, ids):
    """Largest branchial slice with >= 300 states: same-layer states linked
    iff they share at least one parent state."""
    best = None
    for layer in layers[1:]:
        if len(layer) < 300:
            continue
        lid = {s: k for k, s in enumerate(layer)}
        by_parent = defaultdict(list)
        for s in layer:
            for p in parents[s]:
                by_parent[p].append(lid[s])
        pairs = set()
        for sibs in by_parent.values():
            sibs = sorted(sibs)
            for a in range(len(sibs)):
                for b in range(a + 1, len(sibs)):
                    pairs.add((sibs[a], sibs[b]))
        if pairs and (best is None or len(layer) > best[1]):
            best = (pairs_to_csr(pairs, len(layer)), len(layer))
    return None if best is None else best[0]


def measure(name, A):
    A, _ = largest_component_keep(A)
    n = A.shape[0]
    if n < 300:
        print(f"  {name:<42} N={n} too small — skipped")
        return None
    w, V = _norm_laplacian_eigs(A)
    db, ds, de = dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)
    spread = float(np.std([db, ds, de]))
    arho = angle_rho(A, w, V, rng=RNG)
    # magnitude control: same anchors protocol, radius-only distances
    m = 20
    anchors = RNG.choice(n, size=min(150, n), replace=False)
    from scipy.sparse.csgraph import shortest_path
    from scipy.stats import spearmanr

    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    Y = V[:, 1 : 1 + m] / np.sqrt(np.maximum(w[1 : 1 + m], 1e-9))
    r = np.linalg.norm(Y[anchors], axis=1)
    mag_rho = abs(spearmanr(np.abs(r[:, None] - r[None, :])[iu], G[iu]).statistic)
    cls = (
        "transfer"
        if arho >= 0.80 and spread <= 0.30
        else "partial" if arho >= 0.50 else "correct failure"
    )
    print(
        f"  {name:<42} N={n:>5} dims ball/spec/eff "
        f"{db:4.1f}/{ds:4.1f}/{de:4.1f} spread {spread:4.2f}  "
        f"angle_rho {arho:.3f}  mag {mag_rho:.3f}  -> {cls}"
    )
    return dict(
        name=name,
        N=n,
        ball=db,
        spec=ds,
        eff=de,
        spread=spread,
        angle_rho=float(arho),
        mag_rho=float(mag_rho),
        verdict=cls,
    )


if __name__ == "__main__":
    print("Rung 3c — multiway/branchial substrate transfer (PREREG_RUNG3.md)")
    print(
        f"caps: MAX_LEN={MAX_LEN}, MAX_STATES={MAX_STATES}; classification "
        "transfer>=0.80&spread<=0.30 | partial 0.5-0.8 | failure <0.5\n"
    )
    out = []
    for name, (rules, init) in SYSTEMS.items():
        edges, ids, layers, parents = multiway(rules, init)
        print(
            f"{name}: {len(ids)} states, {len(edges)} derivation edges, "
            f"{len(layers)} generations, layer sizes "
            f"{[len(x) for x in layers][:12]}..."
        )
        r = measure("states graph", pairs_to_csr(edges, len(ids)))
        if r:
            r["substrate"] = name + " / states"
            out.append(r)
        B = branchial_slice(layers, parents, ids)
        if B is None:
            print(f"  {'branchial slice':<42} no layer with >=300 states")
        else:
            r = measure("branchial slice (largest layer)", B)
            if r:
                r["substrate"] = name + " / branchial"
                out.append(r)
        print()
    with open("rung3c_result.json", "w") as f:
        json.dump(
            dict(prereg="PREREG_RUNG3.md#rung-3c", results=out),
            f,
            indent=1,
            default=float,
        )
    print("wrote rung3c_result.json")
