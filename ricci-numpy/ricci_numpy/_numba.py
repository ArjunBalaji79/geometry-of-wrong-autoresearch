"""
ricci_numba.py — Ollivier–Ricci curvature, exact LP, CPU-JITed via numba.

Same network simplex as ricci_pure.py but the hot inner loop (`_transport_lp`
and helpers `_solve_duals`, `_find_cycle`, `_is_forest`) is rewritten in
array-only form so numba's nopython mode can compile it to machine code.
Exact agreement with scipy LP to machine precision; ~5-50x faster than the
pure-numpy version. CPU only.

Public API mirrors ricci_pure.
"""
from __future__ import annotations
import numpy as np
from numba import njit


def adjacency(edges, n=None):
    E = np.asarray(list(edges), dtype=int)
    if n is None:
        n = int(E.max()) + 1
    A = np.zeros((n, n), dtype=int)
    A[E[:, 0], E[:, 1]] = A[E[:, 1], E[:, 0]] = 1
    np.fill_diagonal(A, 0)
    return A


def shortest_paths(A, weights=None):
    """Unweighted BFS (default) or weighted Floyd–Warshall (if weights given)."""
    n = A.shape[0]
    if weights is None:
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
    W = np.where((A > 0) & (np.asarray(weights) > 0),
                 np.asarray(weights, dtype=float), np.inf)
    np.fill_diagonal(W, 0.0)
    D = W.copy()
    for k in range(n):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])
    return D


def lazy_walk(A, alpha=0.5):
    deg = A.sum(axis=1, keepdims=True).astype(float)
    M = (1.0 - alpha) * A / np.where(deg == 0, 1.0, deg)
    np.fill_diagonal(M, alpha)
    iso = np.flatnonzero(deg.ravel() == 0)
    M[iso] = 0.0; M[iso, iso] = 1.0
    return M


@njit(cache=True)
def _uf_find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@njit(cache=True)
def _is_forest(basis):
    m, n = basis.shape
    par = np.arange(m + n)
    for i in range(m):
        for j in range(n):
            if basis[i, j]:
                a = _uf_find(par, i)
                b = _uf_find(par, m + j)
                if a == b:
                    return False
                par[a] = b
    return True


@njit(cache=True)
def _solve_duals(C, basis):
    m, n = basis.shape
    u = np.full(m, np.nan); v = np.full(n, np.nan)
    u[0] = 0.0
    for _ in range(m + n + 1):
        changed = False
        for i in range(m):
            for j in range(n):
                if basis[i, j]:
                    if not np.isnan(u[i]) and np.isnan(v[j]):
                        v[j] = C[i, j] - u[i]; changed = True
                    elif not np.isnan(v[j]) and np.isnan(u[i]):
                        u[i] = C[i, j] - v[j]; changed = True
        if not changed:
            break
    for i in range(m):
        if np.isnan(u[i]): u[i] = 0.0
    for j in range(n):
        if np.isnan(v[j]): v[j] = 0.0
    return u, v


@njit(cache=True)
def _find_cycle(basis, i_in, j_in):
    """BFS from row i_in to col j_in in bipartite basis graph.
    Returns (cycle_i, cycle_j) — flattened cell list, alternating + and -."""
    m, n = basis.shape
    parent_row = np.full(m, -1, dtype=np.int64)
    parent_col = np.full(n, -1, dtype=np.int64)
    parent_row[i_in] = -2  # marker for source
    qk = np.empty(m + n + 1, dtype=np.int64)
    qi = np.empty(m + n + 1, dtype=np.int64)
    qk[0] = 0; qi[0] = i_in
    head, tail = 0, 1
    found = False
    while head < tail and not found:
        kind = qk[head]; idx = qi[head]; head += 1
        if kind == 0:  # row
            for jj in range(n):
                if basis[idx, jj] and parent_col[jj] == -1:
                    parent_col[jj] = idx
                    if jj == j_in:
                        found = True; break
                    qk[tail] = 1; qi[tail] = jj; tail += 1
        else:  # col
            for ii in range(m):
                if basis[ii, idx] and parent_row[ii] == -1:
                    parent_row[ii] = idx
                    qk[tail] = 0; qi[tail] = ii; tail += 1
    # Backtrack: col j_in -> row -> col -> ... -> row i_in
    buf_i = np.empty(2 * (m + n) + 4, dtype=np.int64)
    buf_j = np.empty(2 * (m + n) + 4, dtype=np.int64)
    buf_i[0] = i_in; buf_j[0] = j_in
    L = 1
    cur_col = j_in
    while True:
        r = parent_col[cur_col]
        buf_i[L] = r; buf_j[L] = cur_col; L += 1
        if r == i_in:
            break
        c = parent_row[r]
        buf_i[L] = r; buf_j[L] = c; L += 1
        cur_col = c
    return buf_i[:L].copy(), buf_j[:L].copy()


@njit(cache=True)
def _transport_lp(p, q, C, tol=1e-10, max_iter=5000):
    m, n = len(p), len(q)
    T = np.zeros((m, n))
    basis = np.zeros((m, n), dtype=np.bool_)
    # Northwest corner
    pp = p.copy(); qq = q.copy()
    i = 0; j = 0
    while True:
        amt = min(pp[i], qq[j])
        T[i, j] = amt
        basis[i, j] = True
        pp[i] -= amt; qq[j] -= amt
        if i == m - 1 and j == n - 1: break
        if pp[i] <= tol and i < m - 1: i += 1
        else: j += 1
    # Degeneracy fill
    target = m + n - 1
    while basis.sum() < target:
        added = False
        for ii in range(m):
            for jj in range(n):
                if not basis[ii, jj]:
                    basis[ii, jj] = True
                    if _is_forest(basis):
                        added = True; break
                    basis[ii, jj] = False
            if added: break
        if not added: break
    # MODI
    for _ in range(max_iter):
        u, v = _solve_duals(C, basis)
        # reduced costs (set basic to +inf so they don't get picked)
        i_in = -1; j_in = -1; best = -tol
        for i in range(m):
            for j in range(n):
                if not basis[i, j]:
                    rc = C[i, j] - u[i] - v[j]
                    if rc < best:
                        best = rc; i_in = i; j_in = j
        if i_in < 0:
            break
        ci, cj = _find_cycle(basis, i_in, j_in)
        # leave: min T over minus-sign cells (odd indices)
        amt = np.inf; lk = -1
        for k in range(1, len(ci), 2):
            if T[ci[k], cj[k]] < amt:
                amt = T[ci[k], cj[k]]; lk = k
        for k in range(len(ci)):
            if k % 2 == 0:
                T[ci[k], cj[k]] += amt
            else:
                T[ci[k], cj[k]] -= amt
        basis[i_in, j_in] = True
        basis[ci[lk], cj[lk]] = False
    cost = 0.0
    for i in range(m):
        for j in range(n):
            cost += T[i, j] * C[i, j]
    return cost


def wasserstein(p, q, C):
    return _transport_lp(np.asarray(p, float), np.asarray(q, float),
                         np.asarray(C, float))


def edge_curvature(u, v, D, M):
    sup = np.flatnonzero((M[u] > 0) | (M[v] > 0))
    p, q = M[u, sup], M[v, sup]
    C = D[np.ix_(sup, sup)].astype(float)
    d_uv = float(D[u, v])
    if d_uv == 0 or not np.isfinite(d_uv) or d_uv >= D.shape[0]:
        return 0.0
    return 1.0 - wasserstein(p, q, C) / d_uv


def all_curvatures(A, alpha=0.5, weights=None):
    D = shortest_paths(A, weights=weights)
    M = lazy_walk(A, alpha)
    return np.array([edge_curvature(u, v, D, M)
                     for u, v in np.argwhere(np.triu(A, 1) > 0)])


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


def validate(tol=1e-9):
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
