"""ricci-numpy — Ollivier–Ricci curvature on graphs in pure numpy.

Default (pure-numpy, ~250 lines, zero deps beyond numpy):
    from ricci_numpy import all_curvatures, curvature_features

Fast path (drop-in, ~10-200x speedup via numba JIT):
    from ricci_numpy.fast import all_curvatures
"""
from .core import (
    adjacency,
    shortest_paths,
    lazy_walk,
    wasserstein,
    edge_curvature,
    all_curvatures,
    curvature_features,
)

__version__ = "0.1.0"
__all__ = [
    "adjacency", "shortest_paths", "lazy_walk", "wasserstein",
    "edge_curvature", "all_curvatures", "curvature_features",
]
