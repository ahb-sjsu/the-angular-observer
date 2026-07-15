"""Non-uniform-density torus (referee's 'most valuable new experiment').

Thesis under test: radius = density, angle = geometry. If the low-mode commute
radius is the von Luxburg density coordinate (Corollary 1), then under NON-uniform
sampling the radius/degree should track the density profile while the angular
fidelity to the true (flat-torus) metric stays put.

We sample the flat 2-torus with density p(x) proportional to 1 + a*cos(2*pi*x)
(varying along x, periodic), build a fixed-radius RGG, and sweep the contrast a.
Per a we report:
  rho(degree, local density)         -- does degree track density?
  rho(radius, 1/sqrt(degree))        -- Corollary 1 (radius = local density coord)
  rho(radius, 1/sqrt(local density)) -- radius tracks the density profile directly
  angle_rho_graph                    -- angle vs unweighted graph geodesics
  angle_rho_true                     -- angle vs TRUE flat-torus geodesic distance
  magnitude_rho_true, full_rho_true  -- controls vs the true metric

If angle_rho_true holds while radius tracks density, the decomposition is
confirmed: the density lands in the radius, the geometry stays in the angle.

Local, CPU, dense eigh. JSON between markers.
"""
from __future__ import annotations
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr
from rung0_validate import _norm_laplacian_eigs

N, MEAN_DEG, M = 2500, 14, 10


def sample_density_torus(n, a, seed):
    rng = np.random.default_rng(seed)
    xs = []
    while len(xs) < n:
        x = rng.random(2 * n)
        keep = rng.random(2 * n) < (1 + a * np.cos(2 * np.pi * x)) / (1 + a)
        xs.extend(x[keep].tolist())
    x = np.array(xs[:n])
    y = rng.random(n)
    return np.column_stack([x, y])


def rgg(pts, r):
    n = len(pts)
    tree = cKDTree(pts, boxsize=1.0)
    pairs = tree.query_pairs(r, output_type="ndarray")
    A = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T); A.data[:] = 1.0
    return A


def largest_cc(A, pts):
    ncc, lab = connected_components(A, directed=False)
    if ncc == 1:
        return A, pts
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    return A[keep][:, keep], pts[keep]


def torus_geo(P, iu):
    dx = np.abs(P[iu[0], 0] - P[iu[1], 0]); dx = np.minimum(dx, 1 - dx)
    dy = np.abs(P[iu[0], 1] - P[iu[1], 1]); dy = np.minimum(dy, 1 - dy)
    return np.sqrt(dx ** 2 + dy ** 2)


def _rho(d, g):
    if np.ptp(d) < 1e-12 or np.ptp(g) < 1e-12:
        return 0.0
    r = spearmanr(d, g).statistic
    return abs(r) if np.isfinite(r) else 0.0


def alpha1_angle(A, anc, iu, ok, gt):
    """Coifman-Lafon alpha=1 density-normalized angle (recovers Laplace-Beltrami)."""
    q = np.asarray(A.sum(1)).ravel()
    Dq = 1.0 / np.maximum(q, 1e-12)
    W1 = A.multiply(Dq[:, None]).multiply(Dq[None, :]).tocsr()   # A_ij/(q_i q_j)
    w1, V1 = _norm_laplacian_eigs(W1)
    sc = V1[:, 1:1 + M] / np.sqrt(np.maximum(w1[1:1 + M], 1e-9))
    an = sc / np.maximum(np.linalg.norm(sc, axis=1, keepdims=True), 1e-12)
    Ya = an[anc]
    d = np.sqrt(((Ya[iu[0]] - Ya[iu[1]]) ** 2).sum(1))
    return round(_rho(d, gt), 3)


def run(a, seed=3):
    r = np.sqrt(MEAN_DEG / (N * np.pi))               # radius for target mean degree
    pts = sample_density_torus(N, a, seed)
    A, pts = largest_cc(rgg(pts, r), pts)
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    dens = 1 + a * np.cos(2 * np.pi * pts[:, 0])       # local density (up to const)
    w, V = _norm_laplacian_eigs(A)
    scaled = V[:, 1:1 + M] / np.sqrt(np.maximum(w[1:1 + M], 1e-9))
    radius = np.linalg.norm(scaled, axis=1)
    ang = scaled / np.maximum(radius[:, None], 1e-12)

    rng = np.random.default_rng(50 + seed)
    anc = rng.choice(n, size=min(250, n), replace=False)
    Gg = shortest_path(A, method="D", unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    gg = Gg[iu]; ok = np.isfinite(gg)
    gt = torus_geo(pts[anc], iu)

    Ya = ang[anc]
    da = np.sqrt(((Ya[iu[0]] - Ya[iu[1]]) ** 2).sum(1))
    Yl = scaled[anc]
    dmag = np.abs(radius[anc][iu[0]] - radius[anc][iu[1]])
    dfull = np.sqrt(((Yl[iu[0]] - Yl[iu[1]]) ** 2).sum(1))

    return dict(
        a=a, n=int(n), mean_deg=round(float(deg.mean()), 1),
        deg_contrast=round(float(deg.max() / max(deg.min(), 1)), 2),
        rho_deg_density=round(_rho(deg, dens), 3),
        rho_radius_invsqrt_deg=round(_rho(radius, 1 / np.sqrt(deg)), 3),
        rho_radius_invsqrt_density=round(_rho(radius, 1 / np.sqrt(dens)), 3),
        angle_rho_graph=round(_rho(da[ok], gg[ok]), 3),
        angle_rho_true=round(_rho(da, gt), 3),
        angle_rho_true_alpha1=alpha1_angle(A, anc, iu, ok, gt),
        magnitude_rho_true=round(_rho(dmag, gt), 3),
        full_rho_true=round(_rho(dfull, gt), 3),
    )


if __name__ == "__main__":
    rows = []
    for a in [0.0, 0.3, 0.6, 0.9]:
        rows.append(run(a))
        print(f"[a={a}] {rows[-1]}", file=sys.stderr)
    with open("density_torus_result.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("###RESULTS_JSON_START###"); print(json.dumps(rows)); print("###RESULTS_JSON_END###")
