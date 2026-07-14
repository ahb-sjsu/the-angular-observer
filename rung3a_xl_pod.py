"""NRP shard worker for Rung 3a-XL — dynamical stability across the
manifold-rule library (PREREG_RUNG3.md, rung 3a-XL section; registered before
launch).

Each shard draws rules deterministically from the rung-2 rule space
(sweep.worker, SEED_BASE+i). For every QUALIFYING rule (angle_rho >= 0.80 and
spread <= 0.30) it runs the rung-3a stability protocol: snapshots at
[300, 600, 1200, 2400] edges, rewriter seeds {1, 2}, m=20 low modes, up to
120 anchors common to every snapshot's largest component. Reports per rule:
angle fidelity per snapshot, consecutive + long-lever angle stability, and
the rule's own geodesic churn baseline. Controls (magnitude, random modes)
were validated locally in rung 3a and are not re-run at scale.

Indexed Job recipe identical to sweep_pod.py: JOB_COMPLETION_INDEX picks the
slice; results printed as JSON between markers for log aggregation (no PVC).
"""

import json
import math
import os

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr

from sweep import worker  # pins BLAS at import; draw+score
from rung1b_rewriter import _unify, edges_to_csr
from rung0_validate import _norm_laplacian_eigs
from search_harness import SEED_EDGES

SNAP_EDGES = [300, 600, 1200, 2400]
M_MODES = 20
N_ANCHOR = 120
REWRITE_SEEDS = [1, 2]

IDX = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
DRAWS_TOTAL = int(os.environ.get("DRAWS_TOTAL", "6000"))


def qualifies(m):
    return m.get("angle_rho", 0.0) >= 0.80 and m.get("spread", 9.9) <= 0.30


def rewrite_snapshots(lhs, rhs, init_edges, thresholds, max_gen=400, seed=1):
    """rung1b rewrite loop with an edge-count snapshot hook (arity-agnostic)."""
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
    k, lab = connected_components(A, directed=False)
    keep = (
        np.arange(A.shape[0])
        if k == 1
        else np.where(lab == np.bincount(lab).argmax())[0]
    )
    return A[keep][:, keep], keep


def snapshot_geometry(edges, nid):
    A, ids = largest_component_ids(edges_to_csr(edges, nid))
    w, V = _norm_laplacian_eigs(A)
    Y = V[:, 1 : 1 + M_MODES] / np.sqrt(np.maximum(w[1 : 1 + M_MODES], 1e-9))
    U = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return dict(A=A, ids=ids, U=U)


def stability_for_rule(lhs, rhs, rng):
    """Rung-3a measures for one rule; None if it cannot reach all snapshots."""
    per_seed = []
    for seed in REWRITE_SEEDS:
        snaps = rewrite_snapshots(lhs, rhs, SEED_EDGES, SNAP_EDGES, seed=seed)
        if len(snaps) < len(SNAP_EDGES):
            continue
        geo = [snapshot_geometry(e, n) for e, n in snaps]
        common = set(geo[0]["ids"])
        for g in geo[1:]:
            common &= set(g["ids"])
        common = np.array(sorted(common))
        if len(common) < 30:
            continue
        anchors = (
            common
            if len(common) <= N_ANCHOR
            else np.sort(rng.choice(common, size=N_ANCHOR, replace=False))
        )
        iu = np.triu_indices(len(anchors), k=1)
        dists, fids = [], []
        for g in geo:
            pos = {v: i for i, v in enumerate(g["ids"])}
            rows = np.array([pos[v] for v in anchors])
            G = shortest_path(g["A"], method="D", unweighted=True, indices=rows)[
                :, rows
            ][iu]
            P = g["U"][rows]
            d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))[iu]
            fids.append(abs(spearmanr(d, G).statistic))
            dists.append((d, G))
        consec, churn = [], []
        for i in range(len(geo) - 1):
            consec.append(spearmanr(dists[i][0], dists[i + 1][0]).statistic)
            churn.append(spearmanr(dists[i][1], dists[i + 1][1]).statistic)
        long_a = spearmanr(dists[0][0], dists[-1][0]).statistic
        long_g = spearmanr(dists[0][1], dists[-1][1]).statistic
        per_seed.append(
            dict(
                fid=fids,
                consec=consec,
                churn=churn,
                long_angle=float(long_a),
                long_churn=float(long_g),
            )
        )
    if not per_seed:
        return None
    mean = lambda k: np.mean([s[k] for s in per_seed], axis=0)
    return dict(
        fid=mean("fid").tolist(),
        consec=mean("consec").tolist(),
        churn=mean("churn").tolist(),
        long_angle=float(np.mean([s["long_angle"] for s in per_seed])),
        long_churn=float(np.mean([s["long_churn"] for s in per_seed])),
        n_seeds=len(per_seed),
    )


if __name__ == "__main__":
    per = math.ceil(DRAWS_TOTAL / N_SHARDS)
    start, end = IDX * per, min((IDX + 1) * per, DRAWS_TOTAL)
    print(f"3a-XL shard {IDX}/{N_SHARDS}: draws [{start},{end})", flush=True)

    rng = np.random.default_rng(7000 + IDX)
    out, n_scored, n_qual = [], 0, 0
    for i in range(start, end):
        m = worker(i)
        if m is None:
            continue
        n_scored += 1
        if not qualifies(m):
            continue
        n_qual += 1
        st = stability_for_rule(
            [tuple(e) for e in m["lhs"]], [tuple(e) for e in m["rhs"]], rng
        )
        rec = dict(
            draw=i,
            lhs=m["lhs"],
            rhs=m["rhs"],
            dim=m["dim"],
            angle_rho=m["angle_rho"],
            spread=m["spread"],
            stability=st,
        )
        out.append(rec)
        if st is None:
            msg = "stalled before final snapshot"
        else:
            msg = (
                f"consec {np.mean(st['consec']):.2f} "
                f"vs churn {np.mean(st['churn']):.2f}"
            )
        print(
            f"  qual draw {i}: dim {m['dim']:.2f} rho {m['angle_rho']:.2f} "
            f"-> {msg}",
            flush=True,
        )

    print(
        f"shard {IDX}: {n_scored} scored, {n_qual} qualifying, "
        f"{sum(1 for r in out if r['stability'])} measured",
        flush=True,
    )
    print("###RESULTS_JSON_START###", flush=True)
    print(json.dumps(out), flush=True)
    print("###RESULTS_JSON_END###", flush=True)
