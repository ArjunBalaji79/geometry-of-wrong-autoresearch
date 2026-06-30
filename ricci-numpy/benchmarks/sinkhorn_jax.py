"""
ricci_jax.py — Ollivier–Ricci curvature, JIT-compiled and vmapped on GPU via JAX.

Same strategy as ricci_torch: pad each edge's (p, q, C) to a common support
size K, stack along an edge axis E, run one log-domain Sinkhorn batched over
E via vmap + jit. The whole inner loop becomes a single fused GPU kernel
(XLA). Sinkhorn is approximate; with annealed eps down to 1e-5 it matches
the exact LP to ~1e-7 on real graphs.

Public API mirrors ricci_pure.
"""
from __future__ import annotations
from functools import partial
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def adjacency(edges, n=None):
    E = np.asarray(list(edges), dtype=int)
    if n is None:
        n = int(E.max()) + 1
    A = np.zeros((n, n), dtype=int)
    A[E[:, 0], E[:, 1]] = A[E[:, 1], E[:, 0]] = 1
    np.fill_diagonal(A, 0)
    return A


def shortest_paths(A):
    n = A.shape[0]
    Ab = (A > 0).astype(int)
    D = np.full((n, n), n, dtype=int)
    np.fill_diagonal(D, 0)
    reached = np.eye(n, dtype=bool)
    frontier = Ab.astype(bool)
    for k in range(1, n):
        new = frontier & ~reached
        if not new.any():
            break
        D[new] = k
        reached |= new
        frontier = (frontier.astype(int) @ Ab) > 0
    return D


def lazy_walk(A, alpha=0.5):
    deg = A.sum(axis=1, keepdims=True).astype(float)
    M = (1.0 - alpha) * A / np.where(deg == 0, 1.0, deg)
    np.fill_diagonal(M, alpha)
    iso = np.flatnonzero(deg.ravel() == 0)
    M[iso] = 0.0; M[iso, iso] = 1.0
    return M


def _build_edge_arrays(A, alpha):
    D = shortest_paths(A)
    M = lazy_walk(A, alpha)
    edges = np.argwhere(np.triu(A, 1) > 0)
    supports = [np.flatnonzero((M[u] > 0) | (M[v] > 0)) for u, v in edges]
    K = max(len(s) for s in supports)
    E = len(edges)
    P = np.zeros((E, K)); Q = np.zeros((E, K))
    Cpad = np.full((E, K, K), 1e9)
    d_uv = np.zeros(E)
    for e, ((u, v), sup) in enumerate(zip(edges, supports)):
        k = len(sup)
        P[e, :k] = M[u, sup]; Q[e, :k] = M[v, sup]
        Cpad[e, :k, :k] = D[np.ix_(sup, sup)]
        d_uv[e] = D[u, v]
    return P, Q, Cpad, d_uv


def _sinkhorn_one(p, q, C, schedule, n_iter):
    """Log-domain Sinkhorn on a single (K,)/(K,K) problem. Returns W_1 scalar."""
    log_p = jnp.log(jnp.maximum(p, 1e-30))
    log_q = jnp.log(jnp.maximum(q, 1e-30))
    f = jnp.zeros_like(p); g = jnp.zeros_like(q)
    def stage(carry, eps):
        f, g = carry
        def step(fg, _):
            f, g = fg
            f = eps * (log_p - jax.scipy.special.logsumexp((g[None, :] - C) / eps, axis=1))
            g = eps * (log_q - jax.scipy.special.logsumexp((f[:, None] - C) / eps, axis=0))
            return (f, g), None
        (f, g), _ = jax.lax.scan(step, (f, g), None, length=n_iter)
        return (f, g), None
    (f, g), _ = jax.lax.scan(stage, (f, g), schedule)
    T = jnp.exp((f[:, None] + g[None, :] - C) / schedule[-1])
    return jnp.sum(T * C)


@partial(jax.jit, static_argnames=("n_iter",))
def _sinkhorn_batched(P, Q, C, schedule, n_iter):
    return jax.vmap(_sinkhorn_one, in_axes=(0, 0, 0, None, None))(P, Q, C, schedule, n_iter)


_SCHEDULE = jnp.array([0.5, 0.1, 0.02, 0.005, 1e-3, 1e-4, 1e-5])


def all_curvatures(A, alpha=0.5, schedule=_SCHEDULE, n_iter=300):
    if A.shape[0] == 0 or np.sum(np.triu(A, 1)) == 0:
        return np.array([])
    P, Q, C, d_uv = _build_edge_arrays(A, alpha)
    W = _sinkhorn_batched(jnp.asarray(P), jnp.asarray(Q), jnp.asarray(C),
                          jnp.asarray(schedule), n_iter)
    kappa = 1.0 - np.asarray(W) / np.maximum(d_uv, 1e-30)
    return np.asarray(kappa)


def curvature_features(kappas, threshold_neg=-0.05):
    k = np.asarray(kappas, float)
    if k.size == 0:
        return {"mean_kappa": 0.0, "std_kappa": 0.0, "frac_negative": 0.0,
                "min_kappa": 0.0, "max_kappa": 0.0, "n_edges": 0}
    return {"mean_kappa": float(k.mean()), "std_kappa": float(k.std()),
            "frac_negative": float((k < threshold_neg).mean()),
            "min_kappa": float(k.min()), "max_kappa": float(k.max()),
            "n_edges": int(k.size)}


_CANONICAL = {
    "P6_path":     ([(i, i+1) for i in range(5)], [0.5, 0, 0, 0, 0.5]),
    "K4_complete": ([(i, j) for i in range(4) for j in range(i+1, 4)], [2/3]*6),
    "bridge_2tri": ([(0,1),(1,2),(0,2),(2,3),(3,4),(3,5),(4,5)],
                    [0.75, 5/12, 5/12, -1/3, 5/12, 5/12, 0.75]),
}


def validate(tol=1e-5):
    ok = True
    for name, (edges, expected) in _CANONICAL.items():
        k = all_curvatures(adjacency(edges))
        err = float(np.max(np.abs(k - np.array(expected))))
        print(f"  [{'PASS' if err < tol else 'FAIL'}] {name:14s} "
              f"max|err|={err:.2e}  got={np.round(k, 6).tolist()}")
        ok &= err < tol
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
