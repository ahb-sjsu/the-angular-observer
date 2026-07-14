"""
Rung 3a — time-stability of the angular observer basis under the graph's OWN
dynamics (pre-registered in PREREG_RUNG3.md; bars frozen before this ran).

The published core says: at a fixed time, the angle of the low-mode Laplacian
embedding carries the graph's geometry and the radius is degree noise. An
observer, however, persists THROUGH time while the hypergraph updates under
it. This rung asks the missing question: is the angular basis STABLE — do the
relative angular distances among surviving nodes at step T still rank-predict
the angular distances at step T+Delta, while the graph exponentially grows?

Design (all pre-registered):
  rule       R_3D (Wolfram Apr-2020 announcement, borrowed), collapsed seed —
             our one reliable clean-2D emergent manifold (angle_rho 0.822).
  snapshots  captured as hyperedge count crosses ~[500, 1000, 2000, 4000].
  anchors    up to 150 node IDs present in the largest component of EVERY
             snapshot (the rewriter never renames surviving nodes).
  bases      angle (row-normalized m=20 low modes) / magnitude (row norm) /
             random-mode control (20 modes uniform over the spectrum).
  measures   fidelity  = |Spearman(basis distances, hop geodesics)| per snap
             stability = Spearman(basis distances at s, at s') on same pairs
             churn     = Spearman(geodesics at s, geodesics at s') — the
                         substrate's own persistence; the honest ceiling.
  seeds      rewriter seeds 1-4, headline mean +/- std.

Rank comparisons on pairwise distances are invariant to any orthogonal mixing
of the eigenbasis, so NO cross-time alignment (Procrustes etc.) is needed —
degenerate-eigenspace rotation and sign flips cancel out by construction.
"""

import json

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr

from arity3 import hyper_to_csr
from rung0_validate import _norm_laplacian_eigs

# ---- R_3D rule (gorard_rules.py), collapsed seed ----------------------------
LHS = [("1", "1", "2"), ("3", "4", "1")]
RHS = [("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")]
SEED_COLLAPSE = [(0, 0, 0), (0, 0, 0)]

# R_3D from the collapsed seed grows LINEARLY (~1 rewrite event/generation:
# each event consumes the single (a,a,b)-form edge and mints exactly one new
# one, so the causal chain is sequential). Thresholds sized to that rate.
SNAP_EDGES = [400, 800, 1600, 3200]  # hyperedge-count snapshot thresholds
M_MODES = 20
N_ANCHOR = 150
REWRITE_SEEDS = [1, 2, 3, 4]


def _unify(edge, template, binding):
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


def rewrite_snapshots(lhs, rhs, init_edges, thresholds, max_gen=3600, seed=1):
    """rung1b_rewriter.rewrite with a snapshot hook.

    Returns [(edges_copy, next_id_at_snap), ...], one per crossed threshold.
    Node IDs are stable across generations: surviving nodes keep their integer
    ID and fresh RHS nodes mint strictly increasing IDs — which is what makes
    cross-time identity (and hence this whole rung) well-defined.
    """
    rng = np.random.default_rng(seed)
    edges = [tuple(e) for e in init_edges]
    next_id = max(n for e in edges for n in e) + 1
    snaps, t_idx = [], 0

    for _ in range(max_gen):
        if t_idx >= len(thresholds):
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
        while t_idx < len(thresholds) and len(edges) >= thresholds[t_idx]:
            snaps.append((list(edges), next_id))
            t_idx += 1
    return snaps


def largest_component_ids(A):
    """Largest component + the ORIGINAL node IDs of its rows (identity map)."""
    k, lab = connected_components(A, directed=False)
    keep = (
        np.arange(A.shape[0])
        if k == 1
        else np.where(lab == np.bincount(lab).argmax())[0]
    )
    return A[keep][:, keep], keep


def snapshot_geometry(tris, nid, rng):
    """One snapshot -> (ids, geodesic fn, per-basis anchor-distance fns)."""
    A, ids = largest_component_ids(hyper_to_csr(tris, nid))
    w, V = _norm_laplacian_eigs(A)
    Y = V[:, 1 : 1 + M_MODES] / np.sqrt(np.maximum(w[1 : 1 + M_MODES], 1e-9))
    r = np.linalg.norm(Y, axis=1)
    U = Y / np.maximum(r[:, None], 1e-12)  # angle
    lo, hi = 1, len(w)
    rand_idx = rng.choice(np.arange(lo, hi), size=M_MODES, replace=False)
    Yr = V[:, rand_idx] / np.sqrt(np.maximum(w[rand_idx], 1e-9))
    Ur = Yr / np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
    return dict(A=A, ids=ids, U=U, r=r, Ur=Ur)


def pair_dists(X, rows, iu):
    P = X[rows]
    if P.ndim == 1:  # magnitude: |ri-rj|
        return np.abs(P[:, None] - P[None, :])[iu]
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))[iu]


def run_seed(seed, rng):
    snaps = rewrite_snapshots(LHS, RHS, SEED_COLLAPSE, SNAP_EDGES, seed=seed)
    if len(snaps) < len(SNAP_EDGES):
        return None
    geo = [snapshot_geometry(tris, nid, rng) for tris, nid in snaps]

    common = set(geo[0]["ids"])
    for g in geo[1:]:
        common &= set(g["ids"])
    common = np.array(sorted(common))
    anchors = (
        common
        if len(common) <= N_ANCHOR
        else np.sort(rng.choice(common, size=N_ANCHOR, replace=False))
    )
    iu = np.triu_indices(len(anchors), k=1)

    per_snap, D = [], []
    for g in geo:
        pos = {v: i for i, v in enumerate(g["ids"])}
        rows = np.array([pos[v] for v in anchors])
        G = shortest_path(g["A"], method="D", unweighted=True, indices=rows)[:, rows][
            iu
        ]
        d = {
            b: pair_dists(g[k], rows, iu)
            for b, k in (("angle", "U"), ("mag", "r"), ("rand", "Ur"))
        }
        fid = {b: abs(spearmanr(v, G).statistic) for b, v in d.items()}
        d["geo"] = G
        D.append(d)
        per_snap.append(dict(N=g["A"].shape[0], fid=fid))

    stab = []
    pairs = [(i, i + 1) for i in range(len(geo) - 1)] + [(0, len(geo) - 1)]
    for i, j in pairs:
        s = {
            b: spearmanr(D[i][b], D[j][b]).statistic
            for b in ("angle", "mag", "rand", "geo")
        }
        stab.append(dict(pair=(i, j), **s))
    return dict(seed=seed, n_anchor=len(anchors), per_snap=per_snap, stab=stab)


if __name__ == "__main__":
    print("Rung 3a — angular-basis time-stability under R_3D evolution")
    print(
        f"snapshots at hyperedges >= {SNAP_EDGES}, m={M_MODES}, "
        f"anchors<={N_ANCHOR}, rewriter seeds {REWRITE_SEEDS}"
    )
    print(
        "bars (PREREG_RUNG3.md): H1 angle fid>=0.75 every snap (ctrl<=0.30) | "
        "H2 consec stab>=0.70 & >=0.9*churn | H3 first->last >=0.60 & >=0.85*churn\n"
    )

    rng = np.random.default_rng(7)
    runs = []
    for s in REWRITE_SEEDS:
        r = run_seed(s, rng)
        if r is None:
            print(f"seed {s}: rule stalled before final snapshot — excluded")
            continue
        runs.append(r)
        Ns = [p["N"] for p in r["per_snap"]]
        fids = " ".join(f"{p['fid']['angle']:.2f}" for p in r["per_snap"])
        print(
            f"seed {s}: N per snap {Ns}, common anchors {r['n_anchor']}, "
            f"angle fidelity per snap [{fids}]"
        )
        for st in r["stab"]:
            i, j = st["pair"]
            print(
                f"    T{i}->T{j}:  angle {st['angle']:+.3f}  "
                f"mag {st['mag']:+.3f}  rand {st['rand']:+.3f}  "
                f"churn(geo) {st['geo']:+.3f}"
            )

    if not runs:
        raise SystemExit("no seed completed — nothing to score")

    # ---- score against the pre-registered bars ------------------------------
    def agg(get):
        vals = np.array([get(r) for r in runs], float)
        return vals.mean(0), vals.std(0)

    fid_mean, fid_std = agg(lambda r: [p["fid"]["angle"] for p in r["per_snap"]])
    ctl_mean, _ = agg(
        lambda r: [max(p["fid"]["mag"], p["fid"]["rand"]) for p in r["per_snap"]]
    )
    h1 = bool((fid_mean >= 0.75).all() and (ctl_mean <= 0.30).all())

    n_consec = len(SNAP_EDGES) - 1
    st_mean, st_std = agg(lambda r: [s["angle"] for s in r["stab"]])
    ch_mean, _ = agg(lambda r: [s["geo"] for s in r["stab"]])
    h2 = bool(
        (st_mean[:n_consec] >= 0.70).all()
        and (st_mean[:n_consec] >= 0.9 * ch_mean[:n_consec]).all()
    )
    h3 = bool(st_mean[-1] >= 0.60 and st_mean[-1] >= 0.85 * ch_mean[-1])
    gate = bool((ch_mean >= 0.5).all())

    print("\n---- pre-registered scoring ----")
    print(f"validity gate (churn >= 0.5 everywhere): {'OK' if gate else 'FAILED'}")
    print(
        f"H1 fidelity persists   : {'PASS' if h1 else 'FAIL'}   "
        f"angle {np.round(fid_mean, 3).tolist()} +/- {np.round(fid_std, 3).tolist()}, "
        f"max ctrl {np.round(ctl_mean, 3).tolist()}"
    )
    print(
        f"H2 consecutive stability: {'PASS' if h2 else 'FAIL'}   "
        f"angle {np.round(st_mean[:n_consec], 3).tolist()} "
        f"vs 0.9*churn {np.round(0.9 * ch_mean[:n_consec], 3).tolist()}"
    )
    print(
        f"H3 first->last stability: {'PASS' if h3 else 'FAIL'}   "
        f"angle {st_mean[-1]:.3f} +/- {st_std[-1]:.3f} vs bar 0.60 "
        f"and 0.85*churn={0.85 * ch_mean[-1]:.3f}"
    )

    out = dict(
        prereg="PREREG_RUNG3.md#rung-3a",
        rule="R_3D collapsed seed",
        snap_edges=SNAP_EDGES,
        m=M_MODES,
        seeds=[r["seed"] for r in runs],
        gate_churn_ok=gate,
        H1=h1,
        H2=h2,
        H3=h3,
        fid_angle_mean=fid_mean.tolist(),
        fid_angle_std=fid_std.tolist(),
        ctrl_fid_max_mean=ctl_mean.tolist(),
        stab_angle_mean=st_mean.tolist(),
        stab_angle_std=st_std.tolist(),
        churn_mean=ch_mean.tolist(),
        runs=runs,
    )
    with open("rung3a_result.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote rung3a_result.json")
