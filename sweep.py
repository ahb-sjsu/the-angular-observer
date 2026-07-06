"""Full manifold-rule sweep, driven on Atlas CPUs.

Embarrassingly parallel over rule draws. Bounded pool (thermal cap), each worker
single-threaded BLAS (no oversubscription), per-rule SIGALRM timeout so a
pathological rewrite can't hang a worker. Writes ranked results to JSON.
"""

import os

# pin BLAS to 1 thread per process BEFORE numpy is imported (via search_harness)
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

import json
import signal
import argparse
import multiprocessing as mp
import numpy as np
from search_harness import gen_rule, score_rule

MAX_EDGES = 2000
TIMEOUT_S = 20
SEED_BASE = 100_000


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def worker(i):
    """Draw + score one rule deterministically from index i."""
    rng = np.random.default_rng(SEED_BASE + i)
    r = gen_rule(rng)
    if r is None:
        return None
    lhs, rhs = r
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        m = score_rule(lhs, rhs, rng, max_edges=MAX_EDGES)
    except _Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
    if not m.get("valid"):
        return None
    m["lhs"] = [list(e) for e in lhs]
    m["rhs"] = [list(e) for e in rhs]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="sweep_results.json")
    a = ap.parse_args()

    print(
        f"sweep: draws={a.draws} workers={a.workers} "
        f"max_edges={MAX_EDGES} timeout={TIMEOUT_S}s",
        flush=True,
    )
    results, done = [], 0
    with mp.Pool(a.workers) as pool:
        for m in pool.imap_unordered(worker, range(a.draws), chunksize=8):
            done += 1
            if m is not None:
                results.append(m)
            if done % 1000 == 0:
                best = max((r["score"] for r in results), default=float("nan"))
                print(
                    f"  {done}/{a.draws} drawn | {len(results)} scored | "
                    f"best_score={best:.3f}",
                    flush=True,
                )

    results.sort(
        key=lambda d: d["score"] if np.isfinite(d["score"]) else -1e9, reverse=True
    )
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDONE: {len(results)} scored rules -> {a.out}\n", flush=True)
    print(f"  {'score':>6} {'angle_rho':>9} {'spread':>7} {'dim':>5} {'N':>5}   rule")
    for m in results[:20]:
        print(
            f"  {m['score']:6.3f} {m['angle_rho']:9.3f} {m['spread']:7.2f} "
            f"{m['dim']:5.2f} {m['N']:5d}   {m['lhs']} -> {m['rhs']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
