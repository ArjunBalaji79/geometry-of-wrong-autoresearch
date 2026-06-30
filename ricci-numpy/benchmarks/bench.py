"""Benchmark ricci-numpy variants vs. reference implementations.

Reference implementations:
  - scipy_lp:     GraphRicciCurvature-style (networkx + scipy.optimize.linprog)
  - sinkhorn_torch: GPU Sinkhorn (this repo, sinkhorn_torch.py)
  - sinkhorn_jax:   GPU Sinkhorn (this repo, sinkhorn_jax.py)

Compares per-edge agreement with scipy_lp and wall-clock runtime on a sample
of real cosine-threshold sentence graphs (or random graphs if not available).

Usage:
  python benchmarks/bench.py                  # canned synthetic graphs
  python benchmarks/bench.py --traces PATH    # cached MiniLM embeddings (*.npy)
"""
import argparse, sys, time, random, importlib
from pathlib import Path
import numpy as np

import ricci_numpy as rn

try:
    import ricci_numpy.fast as rnf
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import networkx as nx
    from scipy.optimize import linprog  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _scipy_ricci(G, alpha=0.5):
    """Reference: lazy-walk OR-curvature via scipy.optimize.linprog."""
    from scipy.optimize import linprog
    n = G.number_of_nodes()
    dist = dict(nx.all_pairs_shortest_path_length(G))
    def m(node):
        nbrs = list(G.neighbors(node))
        if not nbrs: return {node: 1.0}
        d = {node: alpha}
        w = (1 - alpha) / len(nbrs)
        for nb in nbrs: d[nb] = w
        return d
    def emd(p, q, C):
        K, L = len(p), len(q)
        c = C.flatten()
        A = np.zeros((K + L, K * L))
        for i in range(K):
            for j in range(L): A[i, i * L + j] = 1
        for j in range(L):
            for i in range(K): A[K + j, i * L + j] = 1
        b = np.concatenate([p, q])
        return linprog(c, A_eq=A[:-1], b_eq=b[:-1], bounds=(0, None),
                       method='highs').fun
    kappas = []
    for u, v in G.edges():
        m_u, m_v = m(u), m(v)
        sup = sorted(set(m_u) | set(m_v))
        idx = {s: i for i, s in enumerate(sup)}
        p = np.zeros(len(sup)); q = np.zeros(len(sup))
        for k, val in m_u.items(): p[idx[k]] = val
        for k, val in m_v.items(): q[idx[k]] = val
        C = np.array([[dist[a].get(b, 1e9) for b in sup] for a in sup], dtype=float)
        W = emd(p, q, C)
        d_uv = dist[u].get(v, 1e9)
        kappas.append(1 - W / d_uv if d_uv > 0 else 0)
    return np.array(kappas), list(G.edges())


def make_random_graphs(n_graphs=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_graphs):
        n = int(rng.integers(6, 30))
        p = float(rng.uniform(0.2, 0.5))
        A = (rng.random((n, n)) < p).astype(int)
        A = np.triu(A, 1); A = A + A.T
        # ensure connected: chain backbone
        for i in range(n - 1): A[i, i+1] = A[i+1, i] = 1
        out.append(A)
    return out


def graphs_from_npy(folder, n_graphs=20, seed=0):
    rng = random.Random(seed)
    files = list(Path(folder).glob("*.npy"))
    rng.shuffle(files)
    out = []
    for fp in files:
        emb = np.load(fp)
        if not (6 <= emb.shape[0] <= 40): continue
        # cosine threshold graph with chain backbone
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = emb / norms
        sim = X @ X.T
        n = X.shape[0]
        A = (np.triu(sim, 1) > 0.3).astype(int)
        A = A + A.T
        for i in range(n - 1): A[i, i+1] = A[i+1, i] = 1
        if int(np.triu(A, 1).sum()) < 3: continue
        out.append(A)
        if len(out) >= n_graphs: break
    return out


def align(kappas, edges):
    pairs = sorted(zip([tuple(sorted(e)) for e in edges], list(kappas)))
    return np.array([v for _, v in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=None,
                    help="Folder of cached *.npy MiniLM embeddings to build graphs from.")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    graphs = (graphs_from_npy(args.traces, args.n) if args.traces
              else make_random_graphs(args.n))
    print(f"benchmarking on {len(graphs)} graphs")

    impls = [("numpy_lp", rn.all_curvatures)]
    if HAS_NUMBA:
        # warm up JIT
        _ = rnf.all_curvatures(graphs[0])
        impls.append(("numba_lp", rnf.all_curvatures))

    times = {name: [] for name, _ in impls}
    diffs = {name: [] for name, _ in impls}
    ref_times = []

    for gi, A in enumerate(graphs, 1):
        n = A.shape[0]; E = int(np.triu(A, 1).sum())
        # reference (scipy LP)
        ref = None
        if HAS_SCIPY:
            G = nx.from_numpy_array(A)
            t0 = time.time()
            k_ref, e_ref = _scipy_ricci(G)
            ref_times.append(time.time() - t0)
            ref = align(k_ref, e_ref)
        out = f"  g{gi:02d} n={n:3d} E={E:4d}  "
        if HAS_SCIPY: out += f"scipy={ref_times[-1]:6.2f}s  "
        for name, fn in impls:
            t0 = time.time()
            k = fn(A)
            times[name].append(time.time() - t0)
            edges = [(int(u), int(v)) for u, v in np.argwhere(np.triu(A, 1) > 0)]
            ka = align(k, edges)
            d = float(np.max(np.abs(ka - ref))) if ref is not None else float("nan")
            diffs[name].append(d)
            out += f"{name}={times[name][-1]:6.2f}s/Δ={d:.1e}  "
        print(out)

    print()
    print(f"{'impl':<12s} {'total_s':>10s} {'mean_s':>10s} {'max_|Δκ|':>14s}")
    print("-" * 50)
    if HAS_SCIPY:
        print(f"{'scipy_lp':<12s} {sum(ref_times):10.2f} {np.mean(ref_times):10.3f} "
              f"{0.0:14.3e}  (reference)")
    for name, _ in impls:
        ts = np.array(times[name]); ds = np.array(diffs[name])
        print(f"{name:<12s} {ts.sum():10.2f} {ts.mean():10.3f} {np.nanmax(ds):14.3e}")


if __name__ == "__main__":
    main()
