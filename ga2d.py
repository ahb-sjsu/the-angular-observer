"""#2 — Guided (evolutionary) search for a CLEAN 2D+ emergent manifold rule.

Random search (20k draws) found only ~1.5D clean rules or high-d tangles, never
clean 2D+. Hypothesis: clean-2D rules are rare, not absent -> a guided search
that climbs toward (high dim AND low spread) simultaneously can reach the sweet
spot. Fitness rewards manifold-quality (angle_rho - 0.25*spread) PLUS dimension;
tangles are killed by low quality, filaments by the dim term.

Arity-3 rule space (where 2D is at least geometrically possible). Local
multiprocessing pool (BLAS pinned to 1). Honest readout: track the best CLEAN
(spread<=0.3, angle_rho>=0.8) rule's dimension each generation -- does it climb
past 2.0, or does evolution confirm 2D is genuinely unreachable here?
"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

import numpy as np
import multiprocessing as mp
from arity3 import gen_rule3, score_rule3, EXIST, NEW

POP, GENS, WORKERS, MAX_EDGES = 32, 18, 8, 1000


def validate3(lhs, rhs):
    if len({t for e in lhs for t in e}) < 3:
        return False
    if len(lhs) == 2 and not (set(lhs[0]) & set(lhs[1])):
        return False
    new = [v for v in NEW if any(v in e for e in rhs)]
    if not new:
        return False
    for v in new:
        if not any(v in e and any(t not in NEW for t in e) for e in rhs):
            return False
    return True


def mutate(rule, rng):
    lhs, rhs = [list(e) for e in rule[0]], [list(e) for e in rule[1]]
    toks = list(set(t for e in lhs for t in e)) + list(NEW[:2])
    for _ in range(6):  # try to produce a valid mutant
        L = [list(e) for e in lhs]
        R = [list(e) for e in rhs]
        op = rng.integers(4)
        if op == 0:  # retoken a random rhs slot
            e = R[rng.integers(len(R))]
            e[rng.integers(3)] = str(rng.choice(toks))
        elif op == 1 and len(R) < 5:  # add a rhs triple
            R.append([str(rng.choice(toks)) for _ in range(3)])
        elif op == 2 and len(R) > 1:  # drop a rhs triple
            R.pop(rng.integers(len(R)))
        else:  # retoken a random lhs slot
            e = L[rng.integers(len(L))]
            e[rng.integers(3)] = str(rng.choice(EXIST[:4]))
        Lt = [tuple(e) for e in L]
        Rt = [tuple(e) for e in R]
        if validate3(Lt, Rt):
            return Lt, Rt
    return rule


def _eval(args):
    idx, lhs, rhs = args
    rng = np.random.default_rng(10_000 + idx)
    m = score_rule3(lhs, rhs, rng, max_edges=MAX_EDGES)
    if not m.get("valid"):
        return {"fit": -1.0, "lhs": lhs, "rhs": rhs}
    quality = m["angle_rho"] - 0.25 * m["spread"]
    m["fit"] = quality + 0.5 * min(m["dim"], 3.0)  # reward clean AND high-d
    m["lhs"], m["rhs"] = lhs, rhs
    return m


def main():
    rng = np.random.default_rng(0)
    pop = []
    while len(pop) < POP:
        r = gen_rule3(rng)
        if r:
            pop.append(r)

    print(f"GA: pop={POP} gens={GENS} workers={WORKERS} max_edges={MAX_EDGES}\n")
    print(
        f"  {'gen':>3} | {'best_fit':>8} | best CLEAN (spread<=.3, arho>=.8): "
        f"{'dim':>5} {'arho':>5} {'spread':>6}"
    )
    best_clean_overall = None
    with mp.Pool(WORKERS) as pool:
        for g in range(GENS):
            scored = pool.map(_eval, [(i, l, r) for i, (l, r) in enumerate(pop)])
            scored.sort(key=lambda d: d["fit"], reverse=True)
            clean = [
                m
                for m in scored
                if m.get("valid") and m["spread"] <= 0.3 and m["angle_rho"] >= 0.8
            ]
            clean.sort(key=lambda d: d["dim"], reverse=True)
            bc = clean[0] if clean else None
            if bc and (
                best_clean_overall is None or bc["dim"] > best_clean_overall["dim"]
            ):
                best_clean_overall = bc
            bc_s = (
                f"{bc['dim']:5.2f} {bc['angle_rho']:5.2f} {bc['spread']:6.2f}"
                if bc
                else "   none"
            )
            print(f"  {g:3d} | {scored[0]['fit']:8.3f} | {bc_s}")

            elite = [(m["lhs"], m["rhs"]) for m in scored[: POP // 2]]
            children = [
                mutate(elite[rng.integers(len(elite))], rng)
                for _ in range(POP - len(elite))
            ]
            pop = elite + children

    print("\nBEST CLEAN (spread<=0.3 & angle_rho>=0.8) rule found:")
    if best_clean_overall:
        b = best_clean_overall
        print(
            f"  dim={b['dim']:.2f} angle_rho={b['angle_rho']:.2f} "
            f"spread={b['spread']:.2f} N={b['N']}"
        )
        print(f"  {b['lhs']} -> {b['rhs']}")
        print(
            f"  {'==> reached clean 2D+!' if b['dim'] >= 2.0 else '==> still under 2D; evolution confirms the ceiling.'}"
        )
    else:
        print("  none — no clean manifold rule survived the run.")


if __name__ == "__main__":
    main()
