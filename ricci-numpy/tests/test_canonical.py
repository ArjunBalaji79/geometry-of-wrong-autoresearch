"""Validate against analytical Ollivier-Ricci values on canonical graphs."""
import numpy as np
import pytest

import ricci_numpy as rn

try:
    import ricci_numpy.fast as rnf
    HAS_FAST = True
except ImportError:
    HAS_FAST = False


CANONICAL = {
    "P6_path":     ([(i, i+1) for i in range(5)], [0.5, 0, 0, 0, 0.5]),
    "K4_complete": ([(i, j) for i in range(4) for j in range(i+1, 4)], [2/3]*6),
    "bridge_2tri": ([(0,1),(1,2),(0,2),(2,3),(3,4),(3,5),(4,5)],
                    [0.75, 5/12, 5/12, -1/3, 5/12, 5/12, 0.75]),
}


@pytest.mark.parametrize("name", list(CANONICAL))
def test_core_matches_analytical(name):
    edges, expected = CANONICAL[name]
    A = rn.adjacency(edges)
    k = rn.all_curvatures(A)
    assert np.allclose(k, expected, atol=1e-12), \
        f"{name}: got {k}, expected {expected}"


@pytest.mark.skipif(not HAS_FAST, reason="numba not installed")
@pytest.mark.parametrize("name", list(CANONICAL))
def test_fast_matches_analytical(name):
    edges, expected = CANONICAL[name]
    A = rnf.adjacency(edges)
    k = rnf.all_curvatures(A)
    assert np.allclose(k, expected, atol=1e-12), \
        f"{name}: got {k}, expected {expected}"


@pytest.mark.skipif(not HAS_FAST, reason="numba not installed")
@pytest.mark.parametrize("name", list(CANONICAL))
def test_fast_matches_core(name):
    edges, _ = CANONICAL[name]
    A = rn.adjacency(edges)
    k_core = rn.all_curvatures(A)
    k_fast = rnf.all_curvatures(A)
    assert np.allclose(k_core, k_fast, atol=1e-12)


def test_features_smoke():
    A = rn.adjacency([(i, i+1) for i in range(5)])
    k = rn.all_curvatures(A)
    f = rn.curvature_features(k)
    assert f["n_edges"] == 5
    assert f["min_kappa"] == pytest.approx(0.0)
    assert f["max_kappa"] == pytest.approx(0.5)
    assert f["mean_kappa"] == pytest.approx(0.2)


def test_empty_graph():
    A = np.zeros((3, 3), dtype=int)
    k = rn.all_curvatures(A)
    assert len(k) == 0
    f = rn.curvature_features(k)
    assert f["n_edges"] == 0
