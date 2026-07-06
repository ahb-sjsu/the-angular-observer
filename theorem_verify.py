"""
theorem_verify.py — numerical defense of the Keep-the-Angle Theorem (theorem.md).

For each graph family and size N we build the normalized-Laplacian eigen-embedding
and test, against true graph geodesics (Spearman rho):

  (a) RADIUS-only        -> should be ~0            (radius is geometry-free)
  (b) ANGLE-only         -> high, FLAT in N         (Ng-Weiss row-normalization)
  (c) FULL commute dist  -> DECAYS as N grows       (von Luxburg degeneracy)
  (d) direct von Luxburg: Spearman( R(i,j), 1/d_i + 1/d_j ) -> rises toward 1

Two "full" objects are reported:
  Phi  = project embedding  u_k/sqrt(lam_k)               (low m modes; the code's object)
  R    = exact effective resistance  sum_k (u_k(i)/sqrt(d_i lam_k) - ...)^2  (ALL modes)
R is literally the quantity von Luxburg-Radl-Hein (2014) prove degenerates.

Families: flat 2-torus, flat 3-torus, and an emergent hyperedge-rewrite graph
(rung1b). No GPU; dense eigh up to N=4000.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.sparse.csgraph import shortest_path
from rung0_validate import torus_graph, _norm_laplacian_eigs
from rung1b_rewriter import rewrite, edges_to_csr, largest_component

RNG = np.random.default_rng(11)
M_MODES = 20  # low modes used for Phi (angle / radius / full-Phi)
TORUS_SIZES = (500, 1000, 2000, 4000)

# emergent rewrite graph: a genuine ~1.5D manifold rule found by the rung2 search
# (angle_rho ~0.9, spread ~0.1, ball~spec~1.5). Generational updating grows it
# geometrically, so N is quantized to {513, 2049, 8193} -- 16x span, still shows
# the flat-vs-decay signature. max_edges values below select those three sizes.
_REWRITE_LHS = [("c", "a")]
_REWRITE_RHS = [("c", "x"), ("y", "a"), ("x", "a"), ("x", "x")]
REWRITE_ME = (400, 1000, 5000)  # -> N = 513, 2049, 8193


# ----------------------------------------------------------------- graph builders
def build_torus(dim):
    def f(target_N):
        A, _, _ = torus_graph(target_N, dim)
        return largest_component(A)

    return f


def build_rewrite(max_edges):
    edges, nid = rewrite(
        _REWRITE_LHS,
        _REWRITE_RHS,
        [(0, 1), (0, 2), (1, 2)],
        max_edges=max_edges,
        max_gen=800,
        seed=1,
    )
    return largest_component(edges_to_csr(edges, nid))


# ------------------------------------------------------------------------ metrics
def _pairwise_sq(Z, iu):
    """Squared Euclidean distances between rows of Z, upper-triangle indices iu."""
    return (((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))[iu]


def _sp(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def measure(A, m=M_MODES, n_anchor=250):
    n = A.shape[0]
    d = np.asarray(A.sum(1)).ravel()
    w, V = _norm_laplacian_eigs(A)

    anchors = RNG.choice(n, size=min(n_anchor, n), replace=False)
    G = shortest_path(A, method="D", unweighted=True, indices=anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), k=1)
    geo = G[iu]

    # project embedding Phi = u_k / sqrt(lam_k), low m nontrivial modes
    lo = slice(1, 1 + m)
    Phi = (V[:, lo] / np.sqrt(np.maximum(w[lo], 1e-9)))[anchors]
    r = np.linalg.norm(Phi, axis=1, keepdims=True)
    ang = Phi / np.maximum(r, 1e-12)  # unit sphere (angle)

    rho_full = _sp(_pairwise_sq(Phi, iu), geo)  # (c) full Phi (low m)
    rho_ang = _sp(_pairwise_sq(ang, iu), geo)  # (b) angle only
    rr = r[:, 0]
    rho_rad = _sp(np.abs(rr[:, None] - rr[None, :])[iu], geo)  # (a) radius only

    # exact effective resistance over ALL nontrivial modes: Psi_i = u_k(i)/sqrt(d_i lam_k)
    Psi_all = (V[:, 1:] / np.sqrt(np.maximum(w[1:], 1e-9))) / np.sqrt(d)[:, None]
    Psi = Psi_all[anchors]
    Rij = _pairwise_sq(Psi, iu)  # = R(i,j)
    rho_R = _sp(Rij, geo)  # full commute (von Luxburg object)

    # (d) direct von Luxburg: R(i,j) vs 1/d_i + 1/d_j
    da = d[anchors]
    deg_term = (1.0 / da[:, None] + 1.0 / da[None, :])[iu]
    vl = _sp(Rij, deg_term)

    # radius vs pure degree noise 1/sqrt(d): should be high (radius is degree-free of geometry)
    rho_rad_deg = _sp(np.linalg.norm(Psi, axis=1), 1.0 / np.sqrt(da))
    return dict(
        N=n,
        deg=float(d.mean()),
        rho_rad=rho_rad,
        rho_ang=rho_ang,
        rho_full=rho_full,
        rho_R=rho_R,
        vl=vl,
        rho_rad_deg=rho_rad_deg,
    )


# --------------------------------------------------------------------------- runs
def run_family(name, builder, specs):
    print(f"\n### {name}")
    print(f"  m={M_MODES} low modes for Phi/angle/radius; R uses all modes\n")
    print(
        f"  {'N':>5} {'<deg>':>5} | {'(a)radius':>9} {'(b)ANGLE':>8} "
        f"{'(c)fullPhi':>10} {'R_allmodes':>10} | {'(d)VL:R~1/d':>11} {'|r|~1/sqrt(d)':>13}"
    )
    rows = []
    for tN in specs:
        A = builder(tN)
        r = measure(A)
        rows.append(r)
        print(
            f"  {r['N']:5d} {r['deg']:5.1f} | {r['rho_rad']:9.3f} {r['rho_ang']:8.3f} "
            f"{r['rho_full']:10.3f} {r['rho_R']:10.3f} | {r['vl']:11.3f} {r['rho_rad_deg']:13.3f}"
        )
    a = rows[0]
    b = rows[-1]
    print(
        f"  -> ANGLE  {a['rho_ang']:.3f} -> {b['rho_ang']:.3f}  (flat?)"
        f"   fullPhi {a['rho_full']:.3f} -> {b['rho_full']:.3f}"
        f"   R {a['rho_R']:.3f} -> {b['rho_R']:.3f}"
        f"   VL {a['vl']:.3f} -> {b['vl']:.3f} (->1?)"
    )
    return rows


def m_sweep(name, A):
    print(
        f"\n### m-sweep (fixed graph, {name}, N={A.shape[0]}): full-Phi should decay in m, angle flat"
    )
    print(f"  {'m':>4} | {'ANGLE':>6} {'fullPhi':>8} {'radius':>6}")
    for m in (5, 10, 20, 40, 80):
        r = measure(A, m=m)
        print(
            f"  {m:4d} | {r['rho_ang']:6.3f} {r['rho_full']:8.3f} {r['rho_rad']:6.3f}"
        )


if __name__ == "__main__":
    print("=" * 78)
    print("Keep-the-Angle Theorem — numerical verification")
    print("=" * 78)

    fams = [
        ("2-torus (d=2 manifold)", build_torus(2), TORUS_SIZES),
        ("3-torus (d=3 manifold)", build_torus(3), TORUS_SIZES),
        (
            "emergent rewrite graph (rung1b, ~1.5D manifold rule)",
            build_rewrite,
            REWRITE_ME,
        ),
    ]

    all_rows = {}
    for name, builder, specs in fams:
        all_rows[name] = run_family(name, builder, specs)

    # one m-sweep on the largest 2-torus to show full-vs-m decay
    A2 = build_torus(2)(4000)
    m_sweep("2-torus", A2)

    print("\n" + "=" * 78)
    print(
        "READING: (a) radius ~ 0  (b) ANGLE high & flat in N  (c) fullPhi & R decay in N"
    )
    print("         (d) VL:R~1/d rises toward 1 = von Luxburg degeneracy confirmed")
    print("=" * 78)
