"""
Five Laplacian-spectrum features computed from a sentence-graph and its
embedding signal:
  - spectral_entropy: entropy of the normalized eigenvalue distribution
  - fiedler_value:    algebraic connectivity (lambda_2)
  - grs:              graph Rayleigh smoothness Tr(X^T L X) / Tr(X^T X)
  - shs:              unweighted high-frequency signal-energy fraction
  - hfer:             eigenvalue-weighted high-frequency energy ratio
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.linalg import eigh


def _laplacian(G: nx.Graph) -> np.ndarray:
    """Combinatorial (unnormalized) graph Laplacian L = D - A."""
    return nx.laplacian_matrix(G).toarray().astype(np.float64)


def _eigendecompose(G: nx.Graph):
    """Eigenvalues and eigenvectors of L, sorted ascending."""
    L = _laplacian(G)
    evals, evecs = eigh(L)
    return evals, evecs


def spectral_entropy(G: nx.Graph) -> float:
    """H = -sum_k p_k log p_k where p_k = lambda_k / sum_j lambda_j."""
    evals, _ = _eigendecompose(G)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    if total == 0:
        return 0.0
    p = evals / total
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def fiedler_value(G: nx.Graph) -> float:
    """Algebraic connectivity lambda_2."""
    evals, _ = _eigendecompose(G)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


def graph_rayleigh_smoothness(G: nx.Graph, X: np.ndarray) -> float:
    """
    GRS = Tr(X^T L X) / Tr(X^T X).

    X is the node-embedding matrix used to construct G. Measures how smooth
    the embeddings are with respect to the graph structure (Dirichlet energy).
    """
    L = _laplacian(G)
    num = float(np.trace(X.T @ L @ X))
    den = float(np.trace(X.T @ X))
    if den == 0:
        return 0.0
    return num / den


def spectral_high_frequency_score(G: nx.Graph) -> float:
    """
    SHS: unweighted fraction of eigenvalues in the upper half of the spectrum.

    Specifically, fraction of lambda_k > lambda_max / 2.
    """
    evals, _ = _eigendecompose(G)
    if len(evals) == 0:
        return 0.0
    lam_max = float(np.max(evals))
    if lam_max == 0:
        return 0.0
    return float(np.mean(evals > lam_max / 2))


def high_frequency_energy_ratio(G: nx.Graph) -> float:
    """
    HFER: eigenvalue-weighted high-frequency energy fraction.

    sum(lambda_k for lambda_k > lambda_max/2) / sum(lambda_k).
    """
    evals, _ = _eigendecompose(G)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    if total == 0:
        return 0.0
    lam_max = float(np.max(evals))
    high = evals[evals > lam_max / 2].sum()
    return float(high / total)


def all_spectral_features(G: nx.Graph, X: np.ndarray = None) -> dict:
    """Compute all five spectral features for a graph."""
    return {
        "spectral_entropy": spectral_entropy(G),
        "fiedler_value": fiedler_value(G),
        "grs": graph_rayleigh_smoothness(G, X) if X is not None else None,
        "shs": spectral_high_frequency_score(G),
        "hfer": high_frequency_energy_ratio(G),
    }
