"""
Ollivier-Ricci curvature on graphs, pure numpy/scipy.

Deliberately does NOT depend on GraphRicciCurvature, which crashes on some
environments (observed on GCP deep-learning VMs as of April 2026). Uses
scipy.optimize.linprog for the Wasserstein linear program.

Reference: Ollivier (2009), Ricci curvature of Markov chains on metric spaces.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from scipy.optimize import linprog

__all__ = ["random_walk_measure", "ollivier_ricci_edge", "all_curvatures",
           "curvature_features"]


def random_walk_measure(G: nx.Graph, node, alpha: float = 0.5) -> dict:
    """
    Lazy random-walk probability measure centered at `node`.

    Mass `alpha` stays at the node; mass `(1 - alpha)` is distributed uniformly
    over neighbors. Returns a dict mapping node_id -> probability.
    """
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return {node: 1.0}
    m = {node: alpha}
    w = (1.0 - alpha) / len(nbrs)
    for nb in nbrs:
        m[nb] = w
    return m


def _emd_linprog(p: np.ndarray, q: np.ndarray, C: np.ndarray) -> float:
    """
    Earth Mover's Distance (1-Wasserstein) via LP.

    min <T, C> subject to T 1 = p, T^T 1 = q, T >= 0.

    Uses the 'highs' method which is the scipy default and much faster than
    the old simplex. One redundant row is dropped for numerical stability.
    """
    m, n = len(p), len(q)
    c = C.flatten()
    A_eq = np.zeros((m + n, m * n))
    for i in range(m):
        for j in range(n):
            A_eq[i, i * n + j] = 1.0
    for j in range(n):
        for i in range(m):
            A_eq[m + j, i * n + j] = 1.0
    b_eq = np.concatenate([p, q])
    res = linprog(c, A_eq=A_eq[:-1], b_eq=b_eq[:-1], bounds=(0, None), method='highs')
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(res.fun)


def ollivier_ricci_edge(G: nx.Graph, u, v, dist: dict, alpha: float = 0.5) -> float:
    """
    Ollivier-Ricci curvature for a single edge (u, v).

    kappa(u, v) = 1 - W_1(m_u, m_v) / d(u, v)

    `dist` must be a precomputed all-pairs shortest-path dict:
    dist[i][j] = length of shortest path from i to j.
    """
    m_u = random_walk_measure(G, u, alpha)
    m_v = random_walk_measure(G, v, alpha)

    support = list(set(m_u.keys()) | set(m_v.keys()))
    idx = {n: i for i, n in enumerate(support)}
    k = len(support)

    p = np.zeros(k)
    q = np.zeros(k)
    for n, pr in m_u.items():
        p[idx[n]] = pr
    for n, pr in m_v.items():
        q[idx[n]] = pr

    C = np.zeros((k, k))
    for i, ni in enumerate(support):
        row_dist = dist[ni]
        for j, nj in enumerate(support):
            C[i, j] = row_dist.get(nj, 1e9) if ni != nj else 0.0

    W1 = _emd_linprog(p, q, C)
    d_uv = dist[u].get(v, 1e9)
    if d_uv == 0:
        return 0.0
    return 1.0 - W1 / d_uv


def all_curvatures(G: nx.Graph, alpha: float = 0.5) -> np.ndarray:
    """
    Compute Ollivier-Ricci curvature for every edge of G.

    Returns an array of length |E(G)| in the order given by G.edges().
    """
    if G.number_of_edges() == 0:
        return np.array([])
    dist = dict(nx.all_pairs_shortest_path_length(G))
    return np.array([
        ollivier_ricci_edge(G, u, v, dist, alpha)
        for u, v in G.edges()
    ])


def curvature_features(kappas: np.ndarray, threshold_neg: float = -0.05) -> dict:
    """
    Summary features from a graph's edge curvature distribution.

    frac_negative is the single strongest discriminator observed in the pilot
    (Cohen's d = +2.97 on synthetic topological regimes).
    """
    if len(kappas) == 0:
        return {"mean_kappa": 0.0, "std_kappa": 0.0,
                "frac_negative": 0.0, "min_kappa": 0.0,
                "max_kappa": 0.0, "n_edges": 0}
    return {
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "frac_negative": float(np.mean(kappas < threshold_neg)),
        "min_kappa": float(np.min(kappas)),
        "max_kappa": float(np.max(kappas)),
        "n_edges": int(len(kappas)),
    }
