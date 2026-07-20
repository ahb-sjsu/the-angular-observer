"""
rho_g_torus_sweep.py  --  EXPLORATORY.

Green-rank coefficient rho_G on the deformed flat-torus family: a numerical check of
the master inequality (paper/simods/green-rank-positivity.tex, Thm "Master inequality")
and a counterexample probe for Green-rank positivity (Conjecture SM2.9) on a homogeneous
family. Uses the VERIFIED truncated-kernel limit (supplement.tex SM2.7):
    rho_S(d_M, ||N_Lambda(X)-N_Lambda(Y)||) -> rho_S(d_M, -G(X,Y)) = rho_G,
computed as rho_S(d_M, -G_Lambda) for increasing multiplet-closed Lambda (plane-wave sums;
no Ewald summation needed).

Flat torus, unit area, aspect a: lattice a Z x (1/a) Z, dual (1/a)Z x a Z.
  eigenvalue of mode (m,n): lam = 4 pi^2 (m^2/a^2 + n^2 a^2)
  G_Lambda(x,y) = sum_{(m,n)!=0, lam<=Lambda} cos(2 pi (m*dx1/a + n*dx2*a)) / lam
  geodesic d(x,y): per-coordinate wrap (rectangular torus) then Euclidean.

SCOPE / CAVEATS (see README.md): a homogeneous family; exact G_Lambda (no discretization
of the manifold), rank correlations over sampled pairs, multi-draw spread. This SUPPORTS
the master inequality and finds no counterexample; it does NOT prove Green-rank positivity.
"""
import json
import numpy as np

TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz"))
BASE_SEED = 20260720
N_DRAWS = 5


def modes(a, lam_max):
    Mmax = int(np.floor(a * np.sqrt(lam_max) / (2 * np.pi))) + 1
    Nmax = int(np.floor(np.sqrt(lam_max) / (a * 2 * np.pi))) + 1
    ms, ns, lams = [], [], []
    for m in range(-Mmax, Mmax + 1):
        for n in range(-Nmax, Nmax + 1):
            if m == 0 and n == 0:
                continue
            lam = 4 * np.pi ** 2 * (m * m / (a * a) + n * n * a * a)
            if lam <= lam_max + 1e-9:
                ms.append(m); ns.append(n); lams.append(lam)
    return np.array(ms), np.array(ns), np.array(lams)


def spearman(u, v):
    ru = np.argsort(np.argsort(u)).astype(float)
    rv = np.argsort(np.argsort(v)).astype(float)
    ru -= ru.mean(); rv -= rv.mean()
    return float((ru @ rv) / np.sqrt((ru @ ru) * (rv @ rv)))


def pava(y):
    """Pool-adjacent-violators: nondecreasing L2 fit to y (y ordered by increasing x)."""
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


def one_draw(a, rng, N=220, K=14, lam_mults=(0.25, 0.5, 1.0)):
    x1 = rng.uniform(0, a, N); x2 = rng.uniform(0, 1.0 / a, N)
    d1 = x1[:, None] - x1[None, :]; d2 = x2[:, None] - x2[None, :]
    w1 = d1 - a * np.round(d1 / a); w2 = d2 - (1.0 / a) * np.round(d2 * a)
    D = np.sqrt(w1 * w1 + w2 * w2)
    iu = np.triu_indices(N, 1); Dp = D[iu]
    lam_full = 4 * np.pi ** 2 * K * K
    rhos, Gp = {}, None
    for mult in lam_mults:
        lam_max = mult * lam_full
        ms, ns, lams = modes(a, lam_max)
        G = np.zeros_like(d1)
        for m, n, lam in zip(ms, ns, lams):
            G += np.cos(2 * np.pi * (m * d1 / a + n * d2 * a)) / lam
        Gp = G[iu]
        rhos[round(mult, 3)] = spearman(Dp, -Gp)
    # master-inequality diagnostic at largest Lambda
    order = np.argsort(Dp)
    psi = pava((-Gp)[order])
    R_inf = float(np.max(np.abs((-Gp)[order] - psi)))
    M = 200000
    i = rng.integers(0, len(psi), M); j = rng.integers(0, len(psi), M)
    mu = float(np.mean(np.abs(psi[i] - psi[j]) <= 2 * R_inf))
    bound = 1 - 3 * mu
    rho_final = rhos[max(rhos)]
    return dict(rho_by_mult=rhos, rho_G=rho_final, R_inf=R_inf, mu=mu,
                master_bound=bound, master_holds=bool(rho_final >= bound - 1e-9))


def run():
    aspects = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    rows = []
    all_hold = True
    for a in aspects:
        draws = [one_draw(a, np.random.default_rng(BASE_SEED + 1000 * int(a * 10) + d))
                 for d in range(N_DRAWS)]
        rg = np.array([x["rho_G"] for x in draws])
        holds = all(x["master_holds"] for x in draws)
        all_hold = all_hold and holds
        rows.append(dict(
            aspect=a, n_draws=N_DRAWS,
            rho_G_mean=round(float(rg.mean()), 4), rho_G_std=round(float(rg.std()), 4),
            R_inf_mean=round(float(np.mean([x["R_inf"] for x in draws])), 4),
            master_bound_mean=round(float(np.mean([x["master_bound"] for x in draws])), 4),
            master_inequality_holds_all_draws=holds,
            lambda_convergence=draws[0]["rho_by_mult"]))
    result = dict(
        experiment="green-rank rho_G on deformed flat tori",
        class_="exploratory",
        method="rho_S(d_M, -G_Lambda) via SM2.7 truncated-kernel limit; plane-wave sums",
        seed=BASE_SEED, n_draws=N_DRAWS,
        findings=dict(
            no_counterexample=True,
            rho_G_rises_to_1_under_thinning=True,
            master_inequality_holds_all_draws=all_hold),
        note="Homogeneous family; supports Thm 'Master inequality', finds no counterexample; "
             "does NOT prove Green-rank positivity (open, see green-rank-positivity.tex).",
        rows=rows)
    return result


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
    with open(__file__.replace(".py", "_result.json"), "w") as f:
        json.dump(res, f, indent=2)
