"""
rho_g_hunt.py  --  EXPLORATORY. Green-rank positivity counterexample hunt, round 2.

Extends rho_g_dumbbell.py past the single symmetric neck to the geometries most likely to
break Green-rank positivity (Conjecture SM2.9):
  (A) 2-D INHOMOGENEOUS surfaces of revolution -- multi-neck chains, asymmetric lobes.
  (B) 3-D BERGER SPHERES -- homogeneous but NOT two-point homogeneous: the round S^3 metric
      squashed by eps^2 along the Hopf fiber. G then depends on fiber alignment, not just
      distance -- the decoupling that could push rho_G below zero. eps=1 is round S^3 (a
      two-point homogeneous CONTROL: rho_G must be ~1).

Same generic discrete pipeline as the dumbbell (validated there by the S^2 control): a
local intrinsic distance matrix -> symmetric kNN graph, Gaussian weights -> Green kernel =
pinv(graph Laplacian) -> geodesic = Dijkstra -> rho_G = Spearman(d_geo, -G). Any geometry
is just a recipe for the local distance matrix.

The master inequality gives the exact target: a counterexample needs
mu_Psi(2||R||_inf) >= 1/3 for every monotone profile, i.e. rho_G <= 0.
"""
import json
import numpy as np

try:
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz"))
BASE_SEED = 20260720
N_DRAWS = 3


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def pava(y):
    vals, cnts = [], []
    wts = []
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


def rho_G_from_localdist(Dloc, rng, k=12):
    """Generic core: intrinsic local-distance matrix -> rho_G + master-inequality diagnostic."""
    N = Dloc.shape[0]
    idx = np.argsort(Dloc, axis=1)[:, 1:k + 1]
    eps = np.median(Dloc[np.arange(N)[:, None], idx])
    W = np.zeros_like(Dloc)
    for i in range(N):
        for j in idx[i]:
            wv = np.exp(-Dloc[i, j] ** 2 / (2 * eps * eps))
            W[i, j] = wv; W[j, i] = wv
    Lap = np.diag(W.sum(1)) - W
    G = np.linalg.pinv(Lap, rcond=1e-10)
    Wd = np.where(W > 0, Dloc, 0.0)
    geo = shortest_path(csr_matrix(Wd), method='D', directed=False) if HAVE_SCIPY else Dloc
    iu = np.triu_indices(N, 1)
    g = geo[iu]; Gp = G[iu]
    fin = np.isfinite(g)
    g, Gp = g[fin], Gp[fin]
    rho = spearman(g, -Gp)
    order = np.argsort(g)
    psi = pava((-Gp)[order]); R_inf = float(np.max(np.abs((-Gp)[order] - psi)))
    M = 200000
    ii = rng.integers(0, len(psi), M); jj = rng.integers(0, len(psi), M)
    mu = float(np.mean(np.abs(psi[ii] - psi[jj]) <= 2 * R_inf))
    return rho, bool(not np.isinf(geo).any()), R_inf, 1 - 3 * mu


# ---------- (A) 2-D surfaces of revolution ----------
def surf_localdist(profile, N, rng):
    u = np.linspace(1e-6, profile["L"] - 1e-6, 400)
    f = np.maximum(profile["f"](u), 1e-4)
    f = f / (2 * np.pi * TRAPZ(f, u))
    cdf = np.cumsum(f); cdf /= cdf[-1]
    us = np.interp(rng.uniform(0, 1, N), cdf, u)
    ths = rng.uniform(0, 2 * np.pi, N)
    fs = np.interp(us, u, f)
    du = us[:, None] - us[None, :]
    dth = ths[:, None] - ths[None, :]
    dth = dth - 2 * np.pi * np.round(dth / (2 * np.pi))
    fmid = 0.5 * (fs[:, None] + fs[None, :])
    return np.sqrt(du * du + (fmid * dth) ** 2)


def prof_multineck(depth, L=1.0, w=0.08):
    return {"L": L, "f": lambda u: np.sin(np.pi * u / L) *
            (1 - depth * (np.exp(-((u - L / 3) / w) ** 2) + np.exp(-((u - 2 * L / 3) / w) ** 2)))}


def prof_asym(depth, L=1.0, w=0.10):
    # skewed base (bigger left lobe) + off-center neck
    return {"L": L, "f": lambda u: np.sin(np.pi * u / L) * (1 + 0.6 * np.cos(np.pi * u / L)) *
            (1 - depth * np.exp(-((u - 0.38 * L) / w) ** 2))}


# ---------- (B) 3-D Berger spheres ----------
def berger_localdist(eps, N, rng):
    Q = rng.standard_normal((N, 4)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)  # uniform S^3
    Tq = Q[:, [1, 0, 3, 2]] * np.array([-1.0, 1.0, -1.0, 1.0])                       # Hopf fiber dir
    cosM = np.clip(Q @ Q.T, -1.0, 1.0)
    roundd = np.arccos(cosM)
    sinM = np.sqrt(np.maximum(1 - cosM ** 2, 1e-18))
    fiberM = Tq @ Q.T                      # [i,j] = T_i . Q_j
    fib = fiberM / sinM
    np.fill_diagonal(fib, 0.0)
    bergerd = roundd * np.sqrt(np.maximum(1 - (1 - eps ** 2) * fib ** 2, 0.0))
    return 0.5 * (bergerd + bergerd.T)     # symmetrize


def sweep(kind, param, N, offset):
    rhos, conns, Rs, bounds = [], [], [], []
    for d in range(N_DRAWS):
        rng = np.random.default_rng(BASE_SEED + 100 * offset + d)
        if kind == "multineck":
            Dloc = surf_localdist(prof_multineck(param), N, rng)
        elif kind == "asym":
            Dloc = surf_localdist(prof_asym(param), N, rng)
        elif kind == "berger":
            Dloc = berger_localdist(param, N, rng)
        rho, conn, R_inf, bound = rho_G_from_localdist(Dloc, rng)
        rhos.append(rho); conns.append(conn); Rs.append(R_inf); bounds.append(bound)
    rg = np.array(rhos)
    return dict(kind=kind, param=param, n_draws=N_DRAWS,
                rho_G_mean=round(float(rg.mean()), 4), rho_G_std=round(float(rg.std()), 4),
                R_inf_mean=round(float(np.mean(Rs)), 4),
                master_bound_mean=round(float(np.mean(bounds)), 4),
                connected_all=all(conns))


def run():
    N2, N3 = 1000, 1200
    results = dict(experiment="green-rank counterexample hunt: multi-neck, asymmetric, Berger",
                   class_="exploratory", seed=BASE_SEED, n_draws=N_DRAWS, scipy=HAVE_SCIPY)
    off = 1
    results["multineck"] = [sweep("multineck", d, N2, off + i) for i, d in enumerate([0.0, 0.6, 0.85, 0.95])]
    off = 20
    results["asymmetric"] = [sweep("asym", d, N2, off + i) for i, d in enumerate([0.0, 0.6, 0.85, 0.95])]
    off = 40
    berger = [sweep("berger", e, N3, off + i) for i, e in enumerate([1.0, 0.7, 0.4, 0.2, 0.1, 0.05])]
    results["berger"] = berger
    ctrl = berger[0]  # eps=1 == round S^3, two-point homogeneous
    results["berger_control_round_S3"] = dict(rho_G=ctrl["rho_G_mean"],
                                              trustworthy=bool(ctrl["rho_G_mean"] > 0.95))
    lowest = min([r["rho_G_mean"] for grp in ["multineck", "asymmetric", "berger"] for r in results[grp]])
    results["findings"] = dict(lowest_rho_G=lowest, counterexample_found=bool(lowest <= 0.0),
                               note="counterexample requires rho_G <= 0 (mu_Psi(2||R||)>=1/3).")
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
    with open(__file__.replace(".py", "_result.json"), "w") as f:
        json.dump(res, f, indent=2)
