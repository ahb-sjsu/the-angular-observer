"""
Task #5 — Discrete Ollivier-Ricci (OR) curvature on the emergent Wolfram graph,
and its correlation with local rewrite-update density.

The moonshot claim (Wolfram): emergent spatial curvature is *dynamical* — it
responds to local update activity the way the Einstein tensor responds to
energy. Operationally: edges / regions that get hit by many rewrite events
should have systematically different Ollivier-Ricci curvature than quiet ones.

This file:
  (1) implements OR curvature via an explicit earth-mover LP (scipy.linprog),
      with the lazy-random-walk measure (idleness alpha),
  (2) SANITY-CHECKS the implementation on controls with known curvature sign
      (line ~ 0, square grid ~ 0, triangular lattice / RGG-torus > 0, tree < 0),
  (3) re-implements the rewrite loop LOCALLY (does NOT edit rung1b_rewriter) so
      it can log, per generation, which node-ids are consumed/created, giving a
      real per-node rewrite-event "activity" count and a birth generation,
  (4) correlates per-edge OR curvature vs per-edge update density on the
      emergent graph grown by the canonical arity-3 subdivision rule.

Honest by construction: whatever the correlation is, it gets printed with its
p-value, and the controls are printed first so the OR numbers can be trusted (or
distrusted).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.optimize import linprog
from scipy.stats import pearsonr, spearmanr

from arity3 import hyper_to_csr, SEED3, CANONICAL, gen_rule3
from rung1b_rewriter import _unify
from rung0_validate import torus_graph


# =====================================================================
#  OLLIVIER-RICCI CURVATURE
# =====================================================================
def _lazy_measure(A_indptr, A_indices, x, alpha):
    """Lazy-random-walk probability measure m_x on node x.

    mass alpha stays at x, mass (1-alpha) spread uniformly over neighbors.
    alpha=0 -> classic Ollivier one-step walk; alpha=0.5 -> lazy.
    Returns (nodes ndarray, mass ndarray).
    """
    nb = A_indices[A_indptr[x] : A_indptr[x + 1]]
    deg = len(nb)
    if deg == 0:
        return np.array([x]), np.array([1.0])
    nodes = np.concatenate(([x], nb))
    mass = np.empty(len(nodes))
    mass[0] = alpha
    mass[1:] = (1.0 - alpha) / deg
    return nodes, mass


def _w1(supp_a, mass_a, supp_b, mass_b, D):
    """1-Wasserstein (earth-mover) distance between two discrete measures.

    D is the ground-metric matrix with rows indexed by supp_a, cols by supp_b.
    Solved as a transportation LP with scipy HiGHS.
    """
    p, q = len(supp_a), len(supp_b)
    if p == 1 and q == 1:
        return D[0, 0]
    c = D.ravel()
    # equality constraints: p row-marginals + q col-marginals over p*q vars
    # row i:  sum_j pi[i,j] = mass_a[i]
    # col j:  sum_i pi[i,j] = mass_b[j]
    rows, cols, vals = [], [], []
    for i in range(p):
        for j in range(q):
            rows.append(i)
            cols.append(i * q + j)
            vals.append(1.0)
            rows.append(p + j)
            cols.append(i * q + j)
            vals.append(1.0)
    A_eq = csr_matrix((vals, (rows, cols)), shape=(p + q, p * q))
    b_eq = np.concatenate([mass_a, mass_b])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    return res.fun


def ollivier_ricci_edges(A, edges, alpha=0.5, seed=0, max_edges=None):
    """Ollivier-Ricci curvature kappa(x,y)=1 - W1(m_x,m_y)/d(x,y) for a list of
    edges (x,y) with d(x,y)=1 (graph-adjacent). Ground metric = global unweighted
    shortest-path distance. Returns (edges_used, kappa)."""
    A = A.tocsr()
    indptr, indices = A.indptr, A.indices
    rng = np.random.default_rng(seed)
    edges = list(edges)
    if max_edges is not None and len(edges) > max_edges:
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in idx]

    used, kappas = [], []
    for x, y in edges:
        sa, ma = _lazy_measure(indptr, indices, x, alpha)
        sb, mb = _lazy_measure(indptr, indices, y, alpha)
        # ground metric between the two supports: BFS from supp_a, read cols supp_b
        Drows = shortest_path(A, method="D", unweighted=True, indices=sa)
        D = Drows[:, sb]
        if not np.all(np.isfinite(D)):
            continue
        w1 = _w1(sa, ma, sb, mb, D)
        if not np.isfinite(w1):
            continue
        used.append((x, y))
        kappas.append(1.0 - w1)  # d(x,y)=1
    return used, np.array(kappas)


def edge_list(A):
    """Undirected edge list (i<j) from a symmetric CSR adjacency."""
    Ac = A.tocoo()
    m = Ac.row < Ac.col
    return list(zip(Ac.row[m].tolist(), Ac.col[m].tolist()))


# =====================================================================
#  CONTROL GRAPHS
# =====================================================================
def _csr_from_pairs(n, pairs):
    i = np.array([a for a, b in pairs] + [b for a, b in pairs])
    j = np.array([b for a, b in pairs] + [a for a, b in pairs])
    A = csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1.0
    return A


def path_graph(n):
    return _csr_from_pairs(n, [(i, i + 1) for i in range(n - 1)])


def square_grid(L):
    idx = lambda r, c: r * L + c
    pairs = []
    for r in range(L):
        for c in range(L):
            if c + 1 < L:
                pairs.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < L:
                pairs.append((idx(r, c), idx(r + 1, c)))
    return _csr_from_pairs(L * L, pairs)


def triangular_lattice(L):
    """Triangular lattice with periodic wrap: each node has 6 neighbors, and
    unit triangles exist (so neighborhoods overlap -> positive curvature)."""
    idx = lambda r, c: (r % L) * L + (c % L)
    pairs = set()
    for r in range(L):
        for c in range(L):
            for dr, dc in ((0, 1), (1, 0), (1, -1)):  # 3 of 6 dirs, undirected
                a, b = idx(r, c), idx(r + dr, c + dc)
                pairs.add((min(a, b), max(a, b)))
    return _csr_from_pairs(L * L, list(pairs))


def balanced_tree(depth, branch=2):
    pairs, nxt = [], 1
    frontier = [0]
    for _ in range(depth):
        new = []
        for p in frontier:
            for _ in range(branch):
                pairs.append((p, nxt))
                new.append(nxt)
                nxt += 1
        frontier = new
    return _csr_from_pairs(nxt, pairs)


# =====================================================================
#  LOCAL rewrite loop WITH per-node update-density instrumentation
#  (a copy of rung1b_rewriter.rewrite; rung1b is NOT modified)
# =====================================================================
def rewrite_instrumented(lhs, rhs, init_edges, max_edges=3000, max_gen=400, seed=1):
    """Same semantics as rung1b_rewriter.rewrite, but returns, in addition to
    (edges, next_id):
      activity  : dict node-id -> number of rewrite events that TOUCHED it
                  (touched = appeared in a consumed/matched edge, or was minted)
      birth_gen : dict node-id -> generation at which the node first appeared
    """
    rng = np.random.default_rng(seed)
    edges = [tuple(e) for e in init_edges]
    next_id = max(n for e in edges for n in e) + 1

    activity = {n: 0 for e in edges for n in e}
    birth_gen = {n: 0 for e in edges for n in e}

    for gen in range(1, max_gen + 1):
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
            for j in order:
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
        for idxs, b in matches:
            # nodes touched on the LHS (consumed edges) -> activity++
            touched = {n for k in idxs for n in edges[k]}
            local = {}
            for tmpl in rhs:
                e = []
                for tok in tmpl:
                    if tok in b:
                        e.append(b[tok])
                    else:
                        if tok not in local:
                            local[tok] = next_id
                            activity[next_id] = 0
                            birth_gen[next_id] = gen
                            next_id += 1
                        e.append(local[tok])
                new_edges.append(tuple(e))
            touched |= set(local.values())  # minted nodes also touched
            for n in touched:
                activity[n] = activity.get(n, 0) + 1
        edges = new_edges

    return edges, next_id, activity, birth_gen


def largest_component_keep(A):
    """Largest connected component + the original node indices it keeps."""
    k, lab = connected_components(A, directed=False)
    if k == 1:
        return A, np.arange(A.shape[0])
    keep = np.where(lab == np.bincount(lab).argmax())[0]
    return A[keep][:, keep], keep


# =====================================================================
#  DRIVERS
# =====================================================================
def run_controls(alpha, n_sample, seed=0):
    print(
        f"\n=== OR-curvature CONTROLS (lazy alpha={alpha}, "
        f"<= {n_sample} edges each) ==="
    )
    print(
        f"  {'graph':<22}{'N':>6}{'edges':>8}{'mean kappa':>12}"
        f"{'median':>9}{'std':>8}   expected"
    )
    controls = [
        ("path / line", path_graph(400), "~0  (1D flat)"),
        ("square grid", square_grid(30), "~0  (2D flat tiling)"),
        ("triangular lattice", triangular_lattice(24), "~0  (2D flat tiling!)"),
        ("RGG torus (d=2)", torus_graph(1500, 2)[0], ">0  (ball-overlap bias)"),
        ("balanced tree b=2", balanced_tree(11, 2), "<0  (mean; leaves=0)"),
    ]
    for name, A, exp in controls:
        Alc, _ = largest_component_keep(A)
        es = edge_list(Alc)
        _, k = ollivier_ricci_edges(Alc, es, alpha=alpha, seed=seed, max_edges=n_sample)
        print(
            f"  {name:<22}{Alc.shape[0]:>6}{len(k):>8}{k.mean():>12.4f}"
            f"{np.median(k):>9.4f}{k.std():>8.4f}   {exp}"
        )


def run_emergent(rule, rule_name, alpha, n_sample, max_edges=2500, seed=1):
    print(f"\n=== EMERGENT GRAPH — rule '{rule_name}' ===")
    lhs, rhs = rule
    tris, nid, activity, birth = rewrite_instrumented(
        lhs, rhs, SEED3, max_edges=max_edges, max_gen=400, seed=seed
    )
    A = hyper_to_csr(tris, nid)
    Alc, keep = largest_component_keep(A)
    n = Alc.shape[0]
    # map local graph index -> original node id, then to activity / birth
    act_arr = np.array([activity.get(int(o), 0) for o in keep], float)
    birth_arr = np.array([birth.get(int(o), 0) for o in keep], float)
    deg_arr = np.asarray(Alc.sum(1)).ravel()
    max_gen = birth_arr.max() if birth_arr.max() > 0 else 1.0
    print(
        f"  triples={len(tris)}  nodes(nid)={nid}  largest comp N={n}  "
        f"<deg>={deg_arr.mean():.2f}  max birth-gen={int(max_gen)}"
    )
    print(
        f"  node activity: min={act_arr.min():.0f} med={np.median(act_arr):.0f}"
        f" max={act_arr.max():.0f}"
    )

    es = edge_list(Alc)
    used, kappa = ollivier_ricci_edges(
        Alc, es, alpha=alpha, seed=seed, max_edges=n_sample
    )
    print(
        f"  OR curvature on {len(used)} sampled edges: "
        f"mean={kappa.mean():.4f} median={np.median(kappa):.4f} "
        f"std={kappa.std():.4f} [{kappa.min():.3f},{kappa.max():.3f}]"
    )

    ex = np.array([e[0] for e in used])
    ey = np.array([e[1] for e in used])
    # per-edge update-density proxies
    proxies = {
        "mean activity (event count)": 0.5 * (act_arr[ex] + act_arr[ey]),
        "mean birth-gen (recency)": 0.5 * (birth_arr[ex] + birth_arr[ey]),
        "mean degree": 0.5 * (deg_arr[ex] + deg_arr[ey]),
    }
    print("\n  correlation of per-edge OR curvature vs update-density proxy:")
    print(f"    {'proxy':<30}{'Pearson r':>11}{'p':>10}" f"{'Spearman r':>12}{'p':>10}")
    results = {}
    for pname, pv in proxies.items():
        if np.std(pv) < 1e-12:
            print(f"    {pname:<30}{'(constant proxy)':>33}")
            continue
        pr, pp = pearsonr(kappa, pv)
        sr, sp = spearmanr(kappa, pv)
        results[pname] = (pr, pp, sr, sp)
        print(f"    {pname:<30}{pr:>11.4f}{pp:>10.2e}{sr:>12.4f}{sp:>10.2e}")

    # ---- the honest deflationary test --------------------------------------
    # activity / birth-gen / degree are near-collinear in a growth rule. Does
    # rewrite-ACTIVITY predict curvature *beyond* what node DEGREE already does?
    # If the partial correlation (activity vs curvature, controlling for degree)
    # collapses to ~0, the "dynamical curvature" is just the trivial
    # degree<->curvature fact, not an Einstein-like independent-density law.
    act = proxies["mean activity (event count)"]
    deg = proxies["mean degree"]
    rank = lambda v: spearmanr(v, v)[0] * 0 + np.argsort(np.argsort(v))  # ranks

    def partial_spearman(a, b, c):
        ra, rb, rc = rank(a).astype(float), rank(b).astype(float), rank(c).astype(float)
        res_a = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
        res_b = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
        if np.std(res_a) < 1e-9 or np.std(res_b) < 1e-9:
            return np.nan, np.nan
        return pearsonr(res_a, res_b)

    corr_ad, _ = spearmanr(act, deg)
    pr_partial, pp_partial = partial_spearman(kappa, act, deg)
    print("\n  DEFLATION CHECK:")
    print(
        f"    activity vs degree Spearman = {corr_ad:+.4f}  "
        f"(if ~1, 'activity' IS degree here)"
    )
    if not np.isfinite(pr_partial):
        print(
            "    partial Spearman(curvature, activity | degree) = UNDEFINED "
            "-- activity is a perfect monotone function of degree (collinear)"
        )
    else:
        print(
            f"    partial Spearman(curvature, activity | degree) = "
            f"{pr_partial:+.4f}  p={pp_partial:.2e}"
        )
    print(
        "    -> if ~0 / undefined, curvature<->activity is just "
        "curvature<->degree, NOT independent dynamical curvature"
    )
    return results


if __name__ == "__main__":
    ALPHA = 0.0  # uniform 1-step measure (classic Ollivier; cleanest signs)
    N_SAMPLE = 300

    print("Task #5 — Ollivier-Ricci curvature vs rewrite-update density")
    print("=" * 68)

    run_controls(ALPHA, N_SAMPLE)

    # emergent graph via the canonical 2D-surface subdivision rule
    run_emergent(
        CANONICAL,
        "canonical subdivision {abc}->{abx}{bcx}{cax}",
        ALPHA,
        N_SAMPLE,
        max_edges=2500,
        seed=1,
    )

    # a second, randomly generated arity-3 growth rule for cross-check
    rng = np.random.default_rng(3)
    r2 = None
    for _ in range(300):
        cand = gen_rule3(rng)
        if cand is None:
            continue
        try:
            t, nn, _, _ = rewrite_instrumented(
                cand[0], cand[1], SEED3, max_edges=1500, max_gen=120, seed=1
            )
            Alc, _ = largest_component_keep(hyper_to_csr(t, nn))
            if Alc.shape[0] >= 600:  # needs a big CONNECTED manifold
                r2 = cand
                break
        except Exception:
            continue
    if r2 is not None:
        run_emergent(
            r2,
            f"random gen_rule3 {r2[0]}->{r2[1]}",
            ALPHA,
            N_SAMPLE,
            max_edges=2000,
            seed=1,
        )
    else:
        print("\n(no fast-growing random rule found for cross-check)")

    print("\n" + "=" * 68)
    print("Interpretation: controls fix the sign/scale of OR curvature; the")
    print("emergent-graph correlations above are the actual evidence for (or")
    print("against) dynamical / Einstein-like curvature. Read the p-values.")
