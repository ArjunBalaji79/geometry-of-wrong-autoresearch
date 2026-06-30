"""Tests for weighted-graph support via the optional `weights` argument."""
import numpy as np
import pytest

import ricci_numpy as rn

try:
    import ricci_numpy.fast as rnf
    HAS_FAST = True
except ImportError:
    HAS_FAST = False


def test_uniform_weights_match_unweighted():
    """Floyd–Warshall with all weights = 1 must agree with unweighted BFS."""
    A = rn.adjacency([(0,1),(1,2),(0,2),(2,3),(3,4),(3,5),(4,5)])
    W = A.astype(float)  # weight 1 on every edge
    k_uw = rn.all_curvatures(A)
    k_w = rn.all_curvatures(A, weights=W)
    assert np.allclose(k_uw, k_w, atol=1e-12)


def test_weights_change_curvature():
    """Non-uniform weights should genuinely change the curvature values."""
    A = rn.adjacency([(0,1),(1,2),(2,3),(3,0)])  # 4-cycle
    k_uniform = rn.all_curvatures(A)
    W = A.astype(float)
    W[0, 1] = W[1, 0] = 10.0  # one heavy edge
    k_weighted = rn.all_curvatures(A, weights=W)
    assert not np.allclose(k_uniform, k_weighted), \
        "heavy edge should perturb curvature"


def test_shortest_paths_weighted_simple():
    """Triangle inequality: 1+1 path beats a direct edge of weight 3."""
    A = np.array([[0,1,1],[1,0,1],[1,1,0]])
    W = np.array([[0,1,3],[1,0,1],[3,1,0]], dtype=float)
    D = rn.shortest_paths(A, weights=W)
    assert D[0, 2] == 2.0, f"expected 2.0 via node 1, got {D[0, 2]}"


def test_disconnected_weighted_gives_inf():
    A = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    W = A.astype(float)
    D = rn.shortest_paths(A, weights=W)
    assert np.isinf(D[0, 2]) and np.isinf(D[0, 3])
    assert D[0, 1] == 1.0 and D[2, 3] == 1.0


@pytest.mark.skipif(not HAS_FAST, reason="numba not installed")
def test_fast_weighted_matches_core():
    A = rn.adjacency([(0,1),(1,2),(0,2),(2,3),(3,4),(3,5),(4,5)])
    W = A.astype(float)
    W[2, 3] = W[3, 2] = 5.0  # heavy bridge
    k_core = rn.all_curvatures(A, weights=W)
    k_fast = rnf.all_curvatures(A, weights=W)
    assert np.allclose(k_core, k_fast, atol=1e-12)
