"""Arity-3 sweep worker logic (draw + score one rule with timeout)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import signal
import numpy as np
from arity3 import gen_rule3, score_rule3

MAX_EDGES = 1200          # hyperedges; clique-expansion keeps graph tractable for eigh
TIMEOUT_S = 25
SEED_BASE = 200_000       # distinct from arity-2 sweep -> independent draws


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def worker3(i):
    rng = np.random.default_rng(SEED_BASE + i)
    r = gen_rule3(rng)
    if r is None:
        return None
    lhs, rhs = r
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT_S)
    try:
        m = score_rule3(lhs, rhs, rng, max_edges=MAX_EDGES)
    except (_Timeout, Exception):
        return None
    finally:
        signal.alarm(0)
    if not m.get("valid"):
        return None
    m["lhs"] = [list(e) for e in lhs]
    m["rhs"] = [list(e) for e in rhs]
    return m
