"""Edge cases: disconnected components, isolated nodes, empty graphs."""
import numpy as np
import pytest

import ricci_numpy as rn


def test_two_disconnected_triangles_match_single():
    """In two disconnected triangles, each edge's κ must equal the κ of an
    edge in an isolated triangle (locality of Ollivier–Ricci)."""
    edges_single = [(0,1),(1,2),(0,2)]
    A_single = rn.adjacency(edges_single)
    k_single = rn.all_curvatures(A_single)

    edges_double = [(0,1),(1,2),(0,2),  (3,4),(4,5),(3,5)]
    A_double = rn.adjacency(edges_double)
    k_double = rn.all_curvatures(A_double)

    assert len(k_double) == 6
    np.testing.assert_allclose(k_double[:3], k_single, atol=1e-12)
    np.testing.assert_allclose(k_double[3:], k_single, atol=1e-12)


def test_isolated_node_does_not_break_other_edges():
    """An isolated node alongside a connected component must not perturb
    the connected component's curvatures."""
    A_iso = np.zeros((4, 4), dtype=int)
    A_iso[0, 1] = A_iso[1, 0] = 1
    A_iso[1, 2] = A_iso[2, 1] = 1
    A_iso[0, 2] = A_iso[2, 0] = 1
    # node 3 has no edges
    k_with_iso = rn.all_curvatures(A_iso)

    A_no_iso = rn.adjacency([(0,1),(1,2),(0,2)])
    k_no_iso = rn.all_curvatures(A_no_iso)

    np.testing.assert_allclose(k_with_iso, k_no_iso, atol=1e-12)


def test_empty_edge_set_returns_empty():
    A = np.zeros((5, 5), dtype=int)
    k = rn.all_curvatures(A)
    assert isinstance(k, np.ndarray) and k.size == 0
    feat = rn.curvature_features(k)
    assert feat["n_edges"] == 0
    assert feat["mean_kappa"] == 0.0


def test_single_edge_graph():
    """Two nodes, one edge: both random walks are concentrated and W_1 = 1, κ = 0."""
    A = np.array([[0, 1], [1, 0]])
    k = rn.all_curvatures(A)
    assert k.shape == (1,)
    # m_0 = (alpha, 1-alpha) on (0, 1); m_1 = (1-alpha, alpha) on (0, 1)
    # W_1 = |alpha - (1-alpha)| = |2*alpha - 1| = 0 (with alpha=0.5)
    # κ = 1 - 0/1 = 1.0
    assert np.isclose(k[0], 1.0, atol=1e-12)


def test_self_loops_are_ignored():
    """adjacency() drops self-loops; verify a self-loop in the edge list
    does not perturb curvatures."""
    A_no_loop = rn.adjacency([(0,1),(1,2),(0,2)])
    A_with_loop = rn.adjacency([(0,1),(1,2),(0,2),(0,0)])
    np.testing.assert_array_equal(A_no_loop, A_with_loop)
    np.testing.assert_allclose(rn.all_curvatures(A_no_loop),
                               rn.all_curvatures(A_with_loop), atol=1e-12)


def test_two_components_different_topology():
    """Triangle + path P_3, disconnected. Each component's edges match
    the values for that component in isolation, and disconnection causes
    no leakage between components."""
    A_joint = rn.adjacency([(0,1),(1,2),(0,2),  (3,4),(4,5)])
    k_joint = rn.all_curvatures(A_joint)
    assert k_joint.shape == (5,)

    # K_3 in isolation
    A_tri = rn.adjacency([(0,1),(1,2),(0,2)])
    k_tri = rn.all_curvatures(A_tri)
    # P_3 in isolation
    A_path = rn.adjacency([(0,1),(1,2)])
    k_path = rn.all_curvatures(A_path)

    # Triangle edges (first 3 in the joint graph) match isolated triangle
    np.testing.assert_allclose(k_joint[:3], k_tri, atol=1e-12)
    # Path edges (last 2) match isolated path
    np.testing.assert_allclose(k_joint[3:], k_path, atol=1e-12)
    # Sanity: triangle has positive curvature, path has 0 (interior) or higher
    # at endpoints. With α=0.5, K_3 edges → κ=0.75; P_3 edges (both endpoints) → κ=0.5.
    assert np.allclose(k_tri, [0.75]*3, atol=1e-12)
    assert np.allclose(k_path, [0.5]*2, atol=1e-12)
