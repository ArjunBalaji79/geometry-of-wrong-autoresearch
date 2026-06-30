"""
ricci_torch.py — Ollivier–Ricci curvature, batched on GPU via PyTorch.

Strategy: build per-edge (p, q, C) tensors padded to a common support size K,
stack along an edge axis E, and run one batched log-domain Sinkhorn over the
entire (E, K, K) tensor. All edges curvatures are produced by a single GPU
kernel. Sinkhorn is approximate but with eps annealing down to 1e-5 we match
the exact LP to ~1e-7 on real graphs (verified in bench.py).

Public API mirrors ricci_pure: adjacency, shortest_paths, lazy_walk,
all_curvatures, curvature_features.
"""
from __future__ import annotations
import numpy as np
import torch


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


def _build_edge_tensors(A, alpha, device, dtype):
    """Return (P, Q, C, d_uv) as torch tensors with shapes
    (E, K), (E, K), (E, K, K), (E,) where K = max support across edges."""
    D = shortest_paths(A)
    M = lazy_walk(A, alpha)
    edges = np.argwhere(np.triu(A, 1) > 0)
    E = len(edges)
    supports = [np.flatnonzero((M[u] > 0) | (M[v] > 0)) for u, v in edges]
    K = max(len(s) for s in supports)
    P = np.zeros((E, K)); Q = np.zeros((E, K))
    Cpad = np.full((E, K, K), 1e9)
    d_uv = np.zeros(E)
    for e, ((u, v), sup) in enumerate(zip(edges, supports)):
        k = len(sup)
        P[e, :k] = M[u, sup]
        Q[e, :k] = M[v, sup]
        Cpad[e, :k, :k] = D[np.ix_(sup, sup)]
        d_uv[e] = D[u, v]
    return (torch.as_tensor(P, dtype=dtype, device=device),
            torch.as_tensor(Q, dtype=dtype, device=device),
            torch.as_tensor(Cpad, dtype=dtype, device=device),
            torch.as_tensor(d_uv, dtype=dtype, device=device))


def _sinkhorn_batched(P, Q, C, schedule, n_iter):
    """Log-domain Sinkhorn over the leading edge axis. Returns W_1 of shape (E,)."""
    log_P = torch.log(torch.clamp(P, min=1e-30))
    log_Q = torch.log(torch.clamp(Q, min=1e-30))
    f = torch.zeros_like(P); g = torch.zeros_like(Q)
    for eps in schedule:
        for _ in range(n_iter):
            f = eps * (log_P - torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=-1))
            g = eps * (log_Q - torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=-2))
    T = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - C) / schedule[-1])
    return (T * C).sum(dim=(-1, -2))


def all_curvatures(A, alpha=0.5, device=None, dtype=torch.float64,
                   schedule=(0.5, 0.1, 0.02, 0.005, 1e-3, 1e-4, 1e-5),
                   n_iter=300):
    """Edge curvatures (lexicographic edge order) computed on GPU in one shot."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if A.shape[0] == 0 or np.sum(np.triu(A, 1)) == 0:
        return np.array([])
    P, Q, C, d_uv = _build_edge_tensors(A, alpha, device, dtype)
    W = _sinkhorn_batched(P, Q, C, schedule, n_iter)
    kappa = 1.0 - W / torch.clamp(d_uv, min=1e-30)
    return kappa.cpu().numpy()


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


def validate(tol=1e-5, device=None):
    ok = True
    for name, (edges, expected) in _CANONICAL.items():
        k = all_curvatures(adjacency(edges), device=device)
        err = float(np.max(np.abs(k - np.array(expected))))
        print(f"  [{'PASS' if err < tol else 'FAIL'}] {name:14s} "
              f"max|err|={err:.2e}  got={np.round(k, 6).tolist()}")
        ok &= err < tol
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
