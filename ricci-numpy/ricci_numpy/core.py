"""
ricci.py — Ollivier–Ricci curvature on graphs in pure numpy.

For an edge (u, v) with shortest-path distance d(u, v),
    kappa(u, v) = 1 - W_1(m_u, m_v) / d(u, v)
where m_x is the lazy random walk at x and W_1 is the 1-Wasserstein distance
under shortest-path cost. W_1 is solved *exactly* by a compact transportation
network simplex (MODI). No external graph or LP library — only numpy.

References: Ollivier (2009); Dantzig (1951) on the transportation simplex.
"""
from __future__ import annotations
import numpy as np

__all__ = ["adjacency", "shortest_paths", "lazy_walk", "wasserstein",
           "edge_curvature", "all_curvatures", "curvature_features"]


# --- graph helpers -----------------------------------------------------------

def adjacency(edges, n=None):
    """Symmetric 0/1 adjacency from an iterable of (u, v) pairs."""
    E = np.asarray(list(edges), dtype=int)
    if n is None:
        n = int(E.max()) + 1
    A = np.zeros((n, n), dtype=int)
    A[E[:, 0], E[:, 1]] = A[E[:, 1], E[:, 0]] = 1
    np.fill_diagonal(A, 0)
    return A


def shortest_paths(A, weights=None):
    """All-pairs shortest paths.

    If ``weights`` is None, uses unweighted BFS via boolean matrix powers
    on the 0/1 adjacency A; unreachable pairs get distance n (=infinity
    sentinel). If ``weights`` is provided (same shape as A, with 0 or
    np.inf for non-edges and positive weights elsewhere), uses
    Floyd–Warshall over the float weight matrix; unreachable pairs get
    np.inf."""
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
    """Row i = m_i: mass alpha on i, (1-alpha)/deg on each neighbour.
    Isolated nodes keep all mass on themselves."""
    deg = A.sum(axis=1, keepdims=True).astype(float)
    M = (1.0 - alpha) * A / np.where(deg == 0, 1.0, deg)
    np.fill_diagonal(M, alpha)
    iso = np.flatnonzero(deg.ravel() == 0)
    M[iso] = 0.0
    M[iso, iso] = 1.0
    return M


# --- exact transportation LP (network simplex / MODI) ------------------------

def _northwest_corner(p, q, tol):
    """Initial basic feasible solution by the northwest-corner rule."""
    m, n = len(p), len(q)
    T = np.zeros((m, n))
    basis = np.zeros((m, n), dtype=bool)
    pp, qq = p.astype(float).copy(), q.astype(float).copy()
    i = j = 0
    while True:
        amt = min(pp[i], qq[j])
        T[i, j] = amt
        basis[i, j] = True
        pp[i] -= amt
        qq[j] -= amt
        if i == m - 1 and j == n - 1:
            break
        if pp[i] <= tol and i < m - 1:
            i += 1
        else:
            j += 1
    return T, basis


def _is_forest(basis):
    """Union-find: bipartite basis cells form a forest iff every added edge
    joins two different components."""
    m, n = basis.shape
    par = list(range(m + n))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in np.argwhere(basis):
        a, b = find(int(i)), find(m + int(j))
        if a == b:
            return False
        par[a] = b
    return True


def _solve_duals(C, basis, m, n):
    """Solve u_i + v_j = C_ij for basic cells (set u_0 = 0)."""
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    cells = np.argwhere(basis)
    for _ in range(m + n + 1):
        changed = False
        for i, j in cells:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = C[i, j] - u[i]; changed = True
            elif not np.isnan(v[j]) and np.isnan(u[i]):
                u[i] = C[i, j] - v[j]; changed = True
        if not changed:
            break
    return np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)


def _find_cycle(basis, i_in, j_in):
    """BFS in bipartite basis from row i_in to col j_in; close cycle with
    the entering cell. Returns [(i_in, j_in), ...] alternating + and -."""
    m, n = basis.shape
    row_nbrs = [list(np.flatnonzero(basis[i])) for i in range(m)]
    col_nbrs = [list(np.flatnonzero(basis[:, j])) for j in range(n)]
    pre = {("r", i_in): None}
    stack = [("r", i_in)]
    while stack:
        kind, idx = stack.pop()
        nxt = "c" if kind == "r" else "r"
        nbrs = row_nbrs[idx] if kind == "r" else col_nbrs[idx]
        for nb in nbrs:
            if (nxt, nb) in pre:
                continue
            pre[(nxt, nb)] = (kind, idx)
            if nxt == "c" and nb == j_in:
                chain = [("c", j_in)]
                while pre[chain[-1]] is not None:
                    chain.append(pre[chain[-1]])
                cells = [(i_in, j_in)]
                for k in range(len(chain) - 1, 0, -1):
                    (k2, v2), (k1, v1) = chain[k], chain[k - 1]
                    cells.append((v2, v1) if k2 == "r" else (v1, v2))
                return cells
            stack.append((nxt, nb))
    raise RuntimeError("basis disconnected; cycle not found")


def _transport_lp(p, q, C, tol=1e-10, max_iter=5000):
    """Exact transportation LP via network simplex. Returns min sum(T*C)."""
    p, q, C = np.asarray(p, float), np.asarray(q, float), np.asarray(C, float)
    m, n = len(p), len(q)
    T, basis = _northwest_corner(p, q, tol)
    # Repair degeneracy: pad basis up to m+n-1 cells while keeping it a forest.
    while basis.sum() < m + n - 1:
        added = False
        for i, j in zip(*np.where(~basis)):
            basis[i, j] = True
            if _is_forest(basis):
                added = True
                break
            basis[i, j] = False
        if not added:
            break
    for _ in range(max_iter):
        u, v = _solve_duals(C, basis, m, n)
        rc = C - u[:, None] - v[None, :]
        rc_open = np.where(basis, np.inf, rc)
        flat = int(np.argmin(rc_open))
        i_in, j_in = divmod(flat, n)
        if rc_open[i_in, j_in] >= -tol:
            return float(np.sum(T * C))
        cycle = _find_cycle(basis, i_in, j_in)
        minus = cycle[1::2]
        amts = np.array([T[i, j] for i, j in minus])
        k_leave = int(np.argmin(amts))
        amt = float(amts[k_leave])
        for k, (i, j) in enumerate(cycle):
            T[i, j] += amt if k % 2 == 0 else -amt
        basis[i_in, j_in] = True
        basis[minus[k_leave][0], minus[k_leave][1]] = False
    return float(np.sum(T * C))


def wasserstein(p, q, C):
    """Exact 1-Wasserstein W_1(p, q) under cost matrix C."""
    return _transport_lp(p, q, C)


# --- curvature ---------------------------------------------------------------

def edge_curvature(u, v, D, M):
    """Ollivier–Ricci curvature on edge (u, v) given distances D, walks M.
    Returns 0.0 if u and v are disconnected (d_uv is the +inf sentinel)."""
    sup = np.flatnonzero((M[u] > 0) | (M[v] > 0))
    p, q = M[u, sup], M[v, sup]
    C = D[np.ix_(sup, sup)].astype(float)
    d_uv = float(D[u, v])
    if d_uv == 0 or not np.isfinite(d_uv) or d_uv >= D.shape[0]:
        return 0.0
    return 1.0 - wasserstein(p, q, C) / d_uv


def all_curvatures(A, alpha=0.5, weights=None):
    """Edge curvatures in lexicographic edge order (np.argwhere(triu(A, 1))).

    If ``weights`` is None, treats A as 0/1 adjacency (unweighted). If
    ``weights`` is provided, uses it for shortest-path distances (via
    Floyd–Warshall); edge presence is still determined by A > 0."""
    D = shortest_paths(A, weights=weights)
    M = lazy_walk(A, alpha)
    return np.array([edge_curvature(u, v, D, M)
                     for u, v in np.argwhere(np.triu(A, 1) > 0)])


def curvature_features(kappas, threshold_neg=-0.05):
    """Summary statistics on a graph's edge curvature distribution."""
    k = np.asarray(kappas, float)
    if k.size == 0:
        return {"mean_kappa": 0.0, "std_kappa": 0.0, "frac_negative": 0.0,
                "min_kappa": 0.0, "max_kappa": 0.0, "n_edges": 0}
    return {"mean_kappa": float(k.mean()), "std_kappa": float(k.std()),
            "frac_negative": float((k < threshold_neg).mean()),
            "min_kappa": float(k.min()), "max_kappa": float(k.max()),
            "n_edges": int(k.size)}


# --- validation against published analytical values --------------------------

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
        status = "PASS" if err < tol else "FAIL"
        print(f"  [{status}] {name:14s} max|err|={err:.2e}  "
              f"got={np.round(k, 6).tolist()}")
        ok &= err < tol
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
