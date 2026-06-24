"""
Cosine-similarity threshold graph construction.

Builds an undirected graph on sentence embeddings: edge (i, j) exists iff
cosine similarity > eps (default 0.30, matched per encoder elsewhere).
If the resulting graph is disconnected, sequential chain edges (i, i+1)
are added to restore connectivity without perturbing the thresholded
structure elsewhere.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

EPS_DEFAULT = 0.3


def cosine_threshold_graph(embeddings: np.ndarray, eps: float = EPS_DEFAULT,
                            ensure_connected: bool = True) -> nx.Graph:
    """
    Build an undirected graph on sentence embeddings.

    Edge (i, j) exists iff cosine similarity > eps. Embeddings are assumed to
    be unit-normalized (MiniLM-L6-v2 output); we normalize defensively anyway.

    If ensure_connected is True, the sequential chain of edges (i, i+1) is
    added wherever the cosine graph is disconnected.
    """
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms

    sim = X @ X.T
    G = nx.Graph()
    G.add_nodes_from(range(n))
    ij = np.triu_indices(n, k=1)
    mask = sim[ij] > eps
    edges = list(zip(ij[0][mask].tolist(), ij[1][mask].tolist()))
    G.add_edges_from(edges)

    if ensure_connected and not nx.is_connected(G):
        for i in range(n - 1):
            if not G.has_edge(i, i + 1):
                G.add_edge(i, i + 1)

    return G


def knn_graph(embeddings: np.ndarray, k: int = 3) -> nx.Graph:
    """k-NN graph. Used in ablations only."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        top = np.argpartition(-sim[i], k)[:k]
        for j in top:
            if i != j:
                G.add_edge(i, int(j))
    return G
