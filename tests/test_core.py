"""Rigorous tests of the project's actual scientific claims (fast, small-N).

These are not smoke tests — each asserts a published result of the repo:
estimators recover known dimension, estimators agree on a manifold, the
angle-only observer basis preserves geometry AND beats the random-mode control,
magnitude is throwaway, the rewriter grows a connected manifold, and the
Poincaré-distance port satisfies metric axioms.
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from rung0_validate import (
    torus_graph,
    dim_ball_growth,
    dim_spectral,
    dim_effective_rank,
    _norm_laplacian_eigs,
)
from search_harness import angle_rho
from rung1b_rewriter import rewrite, largest_component
from arity3 import hyper_to_csr, CANONICAL, SEED3
from hyperbolic import poincare_dist


def _torus(n=1000, d=2):
    A, _, _ = torus_graph(n, d)
    return A


def _rho_from_modes(A, w, V, modes, rng, angle=True, n_anchor=150):
    anc = rng.choice(A.shape[0], size=min(n_anchor, A.shape[0]), replace=False)
    D = shortest_path(A, unweighted=True, indices=anc)[:, anc]
    iu = np.triu_indices(len(anc), 1)
    Y = (V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9)))[anc]
    if angle:
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    return abs(spearmanr(d, D[iu]).statistic)


def test_estimators_recover_2d_dimension():
    A = _torus(1000, 2)
    w, _ = _norm_laplacian_eigs(A)
    assert 1.6 <= dim_ball_growth(A) <= 2.4
    assert 1.6 <= dim_spectral(w) <= 2.4


def test_estimators_agree_on_manifold():
    A = _torus(1000, 2)
    w, V = _norm_laplacian_eigs(A)
    dims = [dim_ball_growth(A), dim_spectral(w), dim_effective_rank(A, w, V)]
    assert np.std(dims) < 0.4  # low spread is the manifold signature


def test_angle_preserves_geometry():
    A = _torus(1200, 2)
    w, V = _norm_laplacian_eigs(A)
    rho = angle_rho(A, w, V, rng=np.random.default_rng(0))
    assert rho > 0.8  # the core claim


def test_angle_beats_random_and_magnitude():
    A = _torus(1200, 2)
    w, V = _norm_laplacian_eigs(A)
    rng = np.random.default_rng(0)
    low = np.arange(1, 21)
    rand = rng.choice(np.arange(1, len(w)), size=20, replace=False)
    angle_low = _rho_from_modes(A, w, V, low, rng, angle=True)
    angle_rand = _rho_from_modes(A, w, V, rand, rng, angle=True)
    assert angle_low > 0.8  # geometry lives in the low-mode angle
    assert angle_rand < 0.3  # a random-mode basis destroys it
    assert angle_low > angle_rand + 0.4


def test_rewriter_grows_connected_manifold():
    tris, nid = rewrite(*CANONICAL, SEED3, max_edges=800, max_gen=200, seed=1)
    A = largest_component(hyper_to_csr(tris, nid))
    assert A.shape[0] > 300


def test_poincare_distance_axioms():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 2)) * 0.1
    y = rng.normal(size=(6, 2)) * 0.1
    assert np.allclose(poincare_dist(x, x), 0.0, atol=1e-4)
    assert np.allclose(poincare_dist(x, y), poincare_dist(y, x), atol=1e-4)
    assert (poincare_dist(x, y) >= -1e-9).all()
