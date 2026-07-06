"""
Lean confirmation of the standout case: rule R_3D (announcement labels
{{1,1,2},{3,4,1}}->{{4,4,3},{5,4,5},{5,2,1}}, labels unverified) produced a
strikingly clean ~2D patch (ball2.06/spec2.03/eff2.13, spread0.04) at N=501 in
gorard_rules.py. Because our generational non-overlapping updater fires ~1 match
per generation and the 2-edge matcher is O(E^2)/gen, we grow it only as far as
time allows and check whether the clean ~2D signature is STABLE with N (real
emergent manifold) rather than a 501-node finite-size fluke. We also test a
non-degenerate direct-instance seed for seed robustness.
"""
import sys
import numpy as np
from rung1b_rewriter import rewrite, largest_component
from rung0_validate import (dim_ball_growth, dim_spectral, dim_effective_rank,
                            _norm_laplacian_eigs, RNG)
from arity3 import hyper_to_csr
from search_harness import angle_rho
from gorard_rules import crop_to_ball

LHS = [("1", "1", "2"), ("3", "4", "1")]
RHS = [("4", "4", "3"), ("5", "4", "5"), ("5", "2", "1")]
CONFIGS = [("collapse", [(0, 0, 0), (0, 0, 0)], 800),
           ("collapse", [(0, 0, 0), (0, 0, 0)], 1600),
           ("instance", [(0, 0, 1), (2, 3, 0)], 1600)]


def measure(A):
    A = crop_to_ball(A)
    w, V = _norm_laplacian_eigs(A)
    return (A.shape[0], dim_ball_growth(A), dim_spectral(w),
            dim_effective_rank(A, w, V), angle_rho(A, w, V, rng=RNG))


if __name__ == "__main__":
    print("R_3D larger-N confirmation (labels unverified; MEASURED matters)")
    print("controls: 2-torus 1.90/1.95/2.11 (spread~0.09) | 3-torus 2.73/2.95/3.15")
    print("earlier: collapse gen=500 -> N=501 ball2.06 spec2.03 eff2.13 "
          "spread0.04 a_rho0.813\n")
    for sname, seed, mg in CONFIGS:
        tris, nid = rewrite(LHS, RHS, seed, max_edges=2200, max_gen=mg, seed=1)
        A = largest_component(hyper_to_csr(tris, nid))
        if A.shape[0] < 120:
            print(f"  {sname:9s} gen={mg:5d}: N={A.shape[0]} (too small)")
            sys.stdout.flush(); continue
        n, db, ds, de, arho = measure(A)
        spread = float(np.std([db, ds, de]))
        print(f"  {sname:9s} gen={mg:5d}: N={n:5d} tris={len(tris):5d} | "
              f"ball={db:5.2f} spec={ds:5.2f} eff={de:5.2f} spread={spread:4.2f} "
              f"angle_rho={arho:5.3f}")
        sys.stdout.flush()
    print("\nStable ball~spec~eff near 2 across N => clean emergent 2D manifold.")
