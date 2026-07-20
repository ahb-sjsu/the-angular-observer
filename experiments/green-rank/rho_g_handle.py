"""
rho_g_handle.py  --  EXPLORATORY. Green-rank positivity hunt, round 3: TOPOLOGICAL SHORTCUTS.

The one mechanism that genuinely decouples the Green kernel from geodesic distance: a short,
THIN handle connecting two otherwise-distant regions of a surface. Pairs across the handle
get a SHORT geodesic (through the handle) but HIGH effective resistance (the handle is thin)
=> very negative G => a rank REVERSAL (looks geodesically close, electrically far). If enough
pairs reverse, rho_G drops -- potentially <= 0, a genuine counterexample to Conjecture SM2.9.

Geometry: unit-area sphere (radius R) with two polar caps (half-angle delta) removed and
replaced by a straight thin cylindrical handle along the z-axis connecting the two cap holes
(handle radius r = R sin(delta), length 2 R cos(delta)). Thinner handle = smaller delta.
Sphere geodesic pole-to-pole = pi R; handle path = 2 R cos(delta) < pi R => the handle is a
geodesic shortcut; thin => electrical bottleneck.

Embedded point-cloud pipeline (Euclidean R^3 local distances; the graph Laplacian of a fine
embedded sample approximates Laplace-Beltrami). CONTROL: plain sphere (no handle, no caps)
must give rho_G ~ 1. Reported first. See README.md for scope.
"""
import json
import numpy as np

try:
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

BASE_SEED = 20260720
N_DRAWS = 3


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def pava(y):
    vals, wts, cnts = [], [], []
    for yi in y.astype(float):
        vals.append(yi); wts.append(1.0); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnts.append(c1 + c2)
    out = np.empty_like(y, dtype=float); i = 0
    for v, c in zip(vals, cnts):
        out[i:i + c] = v; i += c
    return out


def rho_G_cloud(P, rng, k=14):
    """Embedded R^3 point cloud P (N x 3) -> rho_G via Euclidean-kNN graph Laplacian."""
    N = P.shape[0]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    eps = np.median(D[np.arange(N)[:, None], idx])
    W = np.zeros((N, N))
    for i in range(N):
        for j in idx[i]:
            wv = np.exp(-D[i, j] ** 2 / (2 * eps * eps))
            W[i, j] = wv; W[j, i] = wv
    Lap = np.diag(W.sum(1)) - W
    G = np.linalg.pinv(Lap, rcond=1e-10)
    Wd = np.where(W > 0, D, 0.0)
    geo = shortest_path(csr_matrix(Wd), method='D', directed=False) if HAVE_SCIPY else D
    iu = np.triu_indices(N, 1)
    g = geo[iu]; Gp = G[iu]
    fin = np.isfinite(g)
    g, Gp = g[fin], Gp[fin]
    rho = spearman(g, -Gp)
    order = np.argsort(g)
    psi = pava((-Gp)[order]); R_inf = float(np.max(np.abs((-Gp)[order] - psi)))
    Msamp = 200000
    ii = rng.integers(0, len(psi), Msamp); jj = rng.integers(0, len(psi), Msamp)
    mu = float(np.mean(np.abs(psi[ii] - psi[jj]) <= 2 * R_inf))
    return rho, bool(not np.isinf(geo).any()), R_inf, 1 - 3 * mu


def sphere_points(n, R, rng, cap=None):
    P = rng.standard_normal((int(n * 1.4), 3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    if cap is not None:                          # remove polar caps (|cos polar angle| > cos delta)
        keep = np.abs(P[:, 2]) < np.cos(cap)
        P = P[keep]
    return R * P[:n]


def handle_points(n, R, delta, rng):
    r = R * np.sin(delta); zmax = R * np.cos(delta)
    z = rng.uniform(-zmax, zmax, n); a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def build(delta, R, Nsph, rng):
    sph = sphere_points(Nsph, R, rng, cap=delta)
    # handle point count ~ matches sphere point density (area ratio)
    area_sph = 4 * np.pi * R ** 2
    area_handle = 2 * np.pi * (R * np.sin(delta)) * (2 * R * np.cos(delta))
    Nh = max(30, int(Nsph * area_handle / area_sph))
    hnd = handle_points(Nh, R, delta, rng)
    return np.vstack([sph, hnd]), len(sph), Nh


def sweep(delta, R, Nsph, offset):
    rows = []
    for d in range(N_DRAWS):
        rng = np.random.default_rng(BASE_SEED + 100 * offset + d)
        if delta is None:                        # plain-sphere control
            P = sphere_points(Nsph, R, rng, cap=None); nh = 0
        else:
            P, nsp, nh = build(delta, R, Nsph, rng)
        rho, conn, R_inf, bound = rho_G_cloud(P, rng)
        rows.append((rho, conn, R_inf, bound, nh))
    rg = np.array([x[0] for x in rows])
    return dict(delta=delta, n_draws=N_DRAWS, n_handle=int(np.mean([x[4] for x in rows])),
                rho_G_mean=round(float(rg.mean()), 4), rho_G_std=round(float(rg.std()), 4),
                R_inf_mean=round(float(np.mean([x[2] for x in rows])), 4),
                master_bound_mean=round(float(np.mean([x[3] for x in rows])), 4),
                connected_all=all(x[1] for x in rows))


def run():
    R = 1.0 / np.sqrt(4 * np.pi)          # unit area
    Nsph = 1100
    control = sweep(None, R, Nsph, 0)
    handles = [sweep(delta, R, Nsph, i + 1) for i, delta in enumerate([0.6, 0.4, 0.25, 0.15, 0.08])]
    lowest = min([h["rho_G_mean"] for h in handles])
    return dict(
        experiment="green-rank hunt: sphere with a thin topological handle (shortcut)",
        class_="exploratory", seed=BASE_SEED, n_draws=N_DRAWS, scipy=HAVE_SCIPY,
        control_plain_sphere=dict(rho_G=control["rho_G_mean"], std=control["rho_G_std"],
                                  trustworthy=bool(control["rho_G_mean"] > 0.95)),
        handles=handles,
        findings=dict(lowest_rho_G_handle=lowest, counterexample_found=bool(lowest <= 0.0),
                      note="thinner handle (smaller delta) = stronger geodesic/resistance "
                           "decoupling; counterexample requires rho_G <= 0."))


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
    with open(__file__.replace(".py", "_result.json"), "w") as f:
        json.dump(res, f, indent=2)
