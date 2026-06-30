"""Fuzz test: random Erdős–Rényi graphs vs. scipy.optimize.linprog reference.

Each random graph is computed with both ricci_numpy.core (and .fast, when
available) and a scipy-LP reference implementation; results must agree to
1e-10 per edge."""
import numpy as np
import pytest

import ricci_numpy as rn

try:
    import ricci_numpy.fast as rnf
    HAS_FAST = True
except ImportError:
    HAS_FAST = False

scipy = pytest.importorskip("scipy")
nx = pytest.importorskip("networkx")
from scipy.optimize import linprog  # noqa: E402


def _scipy_emd(p, q, C):
    K, L = len(p), len(q)
    A = np.zeros((K + L, K * L))
    for i in range(K):
        for j in range(L):
            A[i, i * L + j] = 1
    for j in range(L):
        for i in range(K):
            A[K + j, i * L + j] = 1
    res = linprog(C.flatten(), A_eq=A[:-1], b_eq=np.concatenate([p, q])[:-1],
                  bounds=(0, None), method="highs")
    return float(res.fun)


def _scipy_curvatures(A, alpha=0.5):
    G = nx.from_numpy_array(A)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    def m(node):
        nbrs = list(G.neighbors(node))
        if not nbrs:
            return {node: 1.0}
        d = {node: alpha}; w = (1 - alpha) / len(nbrs)
        for nb in nbrs:
            d[nb] = w
        return d
    kappas = []
    edges = []
    for u, v in G.edges():
        m_u, m_v = m(u), m(v)
        sup = sorted(set(m_u) | set(m_v))
        idx = {s: i for i, s in enumerate(sup)}
        p = np.zeros(len(sup)); q = np.zeros(len(sup))
        for k, val in m_u.items():
            p[idx[k]] = val
        for k, val in m_v.items():
            q[idx[k]] = val
        C = np.array([[dist[a].get(b, 1e9) for b in sup] for a in sup],
                     dtype=float)
        W = _scipy_emd(p, q, C)
        d_uv = dist[u].get(v, 1e9)
        kappas.append(1 - W / d_uv if d_uv > 0 else 0.0)
        edges.append(tuple(sorted((int(u), int(v)))))
    pairs = sorted(zip(edges, kappas))
    return np.array([k for _, k in pairs])


def _aligned(A, k):
    edges = [tuple(sorted((int(u), int(v))))
             for u, v in np.argwhere(np.triu(A, 1) > 0)]
    pairs = sorted(zip(edges, k.tolist()))
    return np.array([v for _, v in pairs])


def _random_connected_graph(n, p, rng):
    A = (rng.random((n, n)) < p).astype(int)
    A = np.triu(A, 1); A = A + A.T
    for i in range(n - 1):  # chain backbone keeps it connected
        A[i, i + 1] = A[i + 1, i] = 1
    return A


@pytest.mark.parametrize("seed", list(range(10)))
def test_core_matches_scipy_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 18))
    p = float(rng.uniform(0.2, 0.5))
    A = _random_connected_graph(n, p, rng)
    k_ours = _aligned(A, rn.all_curvatures(A))
    k_ref = _scipy_curvatures(A)
    assert np.allclose(k_ours, k_ref, atol=1e-10), \
        f"seed={seed}: max diff {np.max(np.abs(k_ours - k_ref))}"


@pytest.mark.skipif(not HAS_FAST, reason="numba not installed")
@pytest.mark.parametrize("seed", list(range(10)))
def test_fast_matches_scipy_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 18))
    p = float(rng.uniform(0.2, 0.5))
    A = _random_connected_graph(n, p, rng)
    k_fast = _aligned(A, rnf.all_curvatures(A))
    k_ref = _scipy_curvatures(A)
    assert np.allclose(k_fast, k_ref, atol=1e-10), \
        f"seed={seed}: max diff {np.max(np.abs(k_fast - k_ref))}"


def test_dense_graph_matches_scipy():
    """Single larger denser graph — the regime where Sinkhorn fails but exact LP doesn't."""
    rng = np.random.default_rng(42)
    A = _random_connected_graph(20, 0.5, rng)
    k_ours = _aligned(A, rn.all_curvatures(A))
    k_ref = _scipy_curvatures(A)
    assert np.allclose(k_ours, k_ref, atol=1e-10)
