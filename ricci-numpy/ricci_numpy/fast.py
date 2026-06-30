"""Optional numba-JIT fast path. Requires `pip install ricci-numpy[fast]`.

Same algorithm as core.py (network simplex transportation LP), but the hot
inner functions are decorated with @numba.njit and rewritten as array-only
so they compile to native code. Matches the pure-numpy version to machine
precision; typically 10-200x faster on small dense graphs.
"""
try:
    import numba  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ricci_numpy.fast requires numba. "
        "Install with: pip install 'ricci-numpy[fast]'"
    ) from e

from ._numba import (
    adjacency,
    shortest_paths,
    lazy_walk,
    wasserstein,
    edge_curvature,
    all_curvatures,
    curvature_features,
)

__all__ = [
    "adjacency", "shortest_paths", "lazy_walk", "wasserstein",
    "edge_curvature", "all_curvatures", "curvature_features",
]
