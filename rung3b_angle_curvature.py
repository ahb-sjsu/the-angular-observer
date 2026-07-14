"""
Rung 3b — coarse Ollivier-Ricci curvature in the ANGULAR ground metric: does
the paper's honest negative (no dynamical / Einstein-like curvature) survive
a basis change? Pre-registered in PREREG_RUNG3.md (incl. Amendment 1).

Iteration history (honest record):
  it-1  one-step measures, unit-hop pairs -> FAILED static (grid −0.061,
        torus −0.064): a unit-hop transport problem is below the resolution
        of the m=20 angular metric. Hard stop honored; dynamical not run.
  it-2  (this file) coarse curvature at the OBSERVER'S scale: measures
        uniform on hop-radius-b balls, pairs at hop distance h, (b,h)=(3,6),
        frozen in Amendment 1 before running. Sphere replaces torus as the
        positive control (torus kept as context, expected -> 0 at scale:
        its one-step positivity is the ball-overlap discreteness artifact).

kappa(x,y) = 1 - W1(mu_x, mu_y) / d(x,y): mu uniform on B(., b); ground
metric and d(x,y) both either hop distance or angular chord distance
d_angle(i,j) = ||u_i - u_j|| between row-normalized m=20 low-mode embedding
rows. kappa is invariant to global metric rescaling, so the two columns are
comparable dimensionless numbers.

Why the re-test is NOT redundant with the published partial correlation: the
partial Spearman removed degree STATISTICALLY from hop-metric curvature; the
angular metric removes it GEOMETRICALLY, before the transport problem is
posed — a different instrument, not a different control. Why the expected
outcome is still the null: if the only curvature-dynamics coupling in these
graphs was degree, an instrument blind to degree sees none.
"""

import json

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr

from arity3 import hyper_to_csr
from curvature import (
    _w1,
    path_graph,
    square_grid,
    balanced_tree,
    rewrite_instrumented,
    largest_component_keep,
)
from rung0_validate import _norm_laplacian_eigs, torus_graph, sphere_graph

M_MODES = 20
BALL_R = 3  # measure = uniform on B(x, BALL_R)      (Amendment 1)
PAIR_H = 6  # curvature measured between hop-H pairs (Amendment 1)
N_PAIRS = 200
REWRITE_SEEDS = [1, 2, 3, 4]

R3D_LHS = [("1", "1", "2"), ("3", "4", "1")]
R3D_RHS = [("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")]
SEED_COLLAPSE = [(0, 0, 0), (0, 0, 0)]


def angle_embedding(A):
    w, V = _norm_laplacian_eigs(A)
    Y = V[:, 1 : 1 + M_MODES] / np.sqrt(np.maximum(w[1 : 1 + M_MODES], 1e-9))
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def coarse_or_pairs(A, U=None, n_pairs=N_PAIRS, b=BALL_R, h=PAIR_H, seed=0):
    """Coarse OR curvature at scale (b, h) in hop metric (U=None) or angular
    metric (U = unit embedding rows). Returns (pairs, kappas, supports)."""
    A = A.tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    pairs, kappas, supports = [], [], []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 8:
        tries += 1
        x = int(rng.integers(n))
        dx = shortest_path(A, method="D", unweighted=True, indices=[x])[0]
        at_h = np.where(dx == h)[0]
        if len(at_h) == 0:
            continue
        y = int(rng.choice(at_h))
        sa = np.where(dx <= b)[0]
        dy = shortest_path(A, method="D", unweighted=True, indices=[y])[0]
        sb = np.where(dy <= b)[0]
        ma = np.full(len(sa), 1.0 / len(sa))
        mb = np.full(len(sb), 1.0 / len(sb))
        if U is None:  # hop ground metric
            D = shortest_path(A, method="D", unweighted=True, indices=sa)[:, sb]
            dxy = float(h)
            if not np.all(np.isfinite(D)):
                continue
        else:  # angular ground metric
            D = np.linalg.norm(U[sa][:, None, :] - U[sb][None, :, :], axis=-1)
            dxy = float(np.linalg.norm(U[x] - U[y]))
            if dxy < 1e-9:
                continue
        w1 = _w1(sa, ma, sb, mb, D)
        if not np.isfinite(w1):
            continue
        pairs.append((x, y))
        kappas.append(1.0 - w1 / dxy)
        supports.append(np.union1d(sa, sb))
    return pairs, np.array(kappas), supports


def _rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def partial_spearman(a, b, c):
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    res_a = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
    res_b = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    if np.std(res_a) < 1e-9 or np.std(res_b) < 1e-9:
        return np.nan, np.nan
    return pearsonr(res_a, res_b)


# ---------------------------------------------------------------- 3b-static
def run_static():
    print(
        f"=== 3b-static it-2: coarse OR at observer scale "
        f"(b={BALL_R}, h={PAIR_H}) ==="
    )
    print(f"  {'graph':<20}{'N':>6}{'hop mean k':>12}{'angle mean k':>14}   bar")
    bars = [
        ("path / line", path_graph(400), "zero", True),
        ("square grid", square_grid(30), "zero", True),
        ("balanced tree b=2", balanced_tree(11, 2), "negative", True),
        ("RGG sphere (d=2)", sphere_graph(2000)[0], "positive", True),
        ("RGG torus (d=2)", torus_graph(1500, 2)[0], "zero", False),  # context
    ]
    verdicts, ctx = {}, {}
    for name, A, bar, scored in bars:
        Alc, _ = largest_component_keep(A)
        U = angle_embedding(Alc)
        _, kh, _ = coarse_or_pairs(Alc, U=None, seed=0)
        _, ka, _ = coarse_or_pairs(Alc, U=U, seed=0)
        cls = (
            "zero"
            if abs(ka.mean()) < 0.05
            else ("positive" if ka.mean() > 0 else "negative")
        )
        tag = "OK" if cls == bar else "MISMATCH"
        (verdicts if scored else ctx)[name] = (cls == bar, float(ka.mean()))
        print(
            f"  {name:<20}{Alc.shape[0]:>6}{kh.mean():>12.4f}"
            f"{ka.mean():>14.4f}   {bar:<9} -> {cls} "
            f"{tag}{'' if scored else ' (context row, unscored)'}"
        )
    ok = all(v[0] for v in verdicts.values())
    print(f"  3b-static it-2: {'PASS' if ok else 'FAIL (ends 3b per Amendment 1)'}")
    return ok, verdicts, ctx


# ------------------------------------------------------------- 3b-dynamical
def run_dynamical_seed(seed):
    tris, nid, activity, birth = rewrite_instrumented(
        R3D_LHS, R3D_RHS, SEED_COLLAPSE, max_edges=1600, max_gen=1800, seed=seed
    )
    A, keep = largest_component_keep(hyper_to_csr(tris, nid))
    U = angle_embedding(A)
    act = np.array([activity.get(int(o), 0) for o in keep], float)
    deg = np.asarray(A.sum(1)).ravel()

    pairs, ka, supports = coarse_or_pairs(A, U=U, seed=seed)
    r_act = np.array([act[s].mean() for s in supports])  # regional density
    r_deg = np.array([deg[s].mean() for s in supports])

    raw_r, raw_p = spearmanr(ka, r_act)
    deg_r, _ = spearmanr(ka, r_deg)
    par_r, par_p = partial_spearman(ka, r_act, r_deg)
    return dict(
        seed=seed,
        N=A.shape[0],
        n_pairs=len(pairs),
        kappa_mean=float(ka.mean()),
        kappa_vs_deg=float(deg_r),
        raw=float(raw_r),
        raw_p=float(raw_p),
        partial=(float(par_r) if np.isfinite(par_r) else None),
        partial_p=(float(par_p) if np.isfinite(par_p) else None),
    )


if __name__ == "__main__":
    print(
        "Rung 3b — does the no-dynamical-curvature null survive the angle "
        "basis?  (PREREG_RUNG3.md + Amendment 1)\n"
    )
    static_ok, verdicts, ctx = run_static()
    result = dict(
        prereg="PREREG_RUNG3.md#rung-3b+amendment-1",
        scale=dict(b=BALL_R, h=PAIR_H),
        static_pass=static_ok,
        static={k: v[1] for k, v in verdicts.items()},
        static_context={k: v[1] for k, v in ctx.items()},
    )

    if not static_ok:
        print(
            "\ninstrument failed static it-2 — 3b ends as an instrument-"
            "failure report (Amendment 1); dynamical NOT run"
        )
    else:
        print(
            f"\n=== 3b-dynamical: coarse angle-kappa vs REGIONAL update "
            f"activity on R_3D (seeds {REWRITE_SEEDS}) ==="
        )
        print(
            f"  {'seed':>4}{'N':>6}{'pairs':>7}{'k mean':>9}{'k~deg':>8}"
            f"{'raw rho':>9}{'p':>10}{'partial|deg':>12}{'p':>10}"
        )
        runs = []
        for s in REWRITE_SEEDS:
            r = run_dynamical_seed(s)
            runs.append(r)
            pr = "undef" if r["partial"] is None else f"{r['partial']:+.3f}"
            pp = "" if r["partial_p"] is None else f"{r['partial_p']:.1e}"
            print(
                f"  {r['seed']:>4}{r['N']:>6}{r['n_pairs']:>7}"
                f"{r['kappa_mean']:>9.3f}{r['kappa_vs_deg']:>+8.2f}"
                f"{r['raw']:>+9.3f}{r['raw_p']:>10.1e}{pr:>12}{pp:>10}"
            )

        sig = [
            r
            for r in runs
            if r["partial"] is not None
            and abs(r["partial"]) >= 0.30
            and r["partial_p"] is not None
            and r["partial_p"] < 1e-3
        ]
        same_sign = len({np.sign(r["partial"]) for r in sig}) <= 1
        claim = len(sig) >= 3 and same_sign
        print(
            f"\n  bar for a dynamical-curvature claim: |partial|>=0.30, "
            f"p<1e-3, same sign, >=3/4 seeds -> "
            f"{len(sig)}/4 qualify{'' if same_sign else ' (signs differ)'}"
        )
        print(
            f"  VERDICT: {'DYNAMICAL SIGNAL (claim earned)' if claim else 'NULL SURVIVES the basis change'}"
        )
        result.update(dynamical=runs, n_qualifying=len(sig), claim=claim)

    with open("rung3b_result.json", "w") as f:
        json.dump(result, f, indent=1, default=float)
    print("\nwrote rung3b_result.json")
