# ricci-numpy

**Ollivier–Ricci curvature on graphs in pure numpy.** No networkx. No scipy.
Just a 250-line implementation of the discrete Ricci curvature with an
embedded transportation network simplex solving the 1-Wasserstein step
exactly.

```python
import ricci_numpy as rn

A = rn.adjacency([(0,1),(1,2),(0,2),(2,3),(3,4),(3,5),(4,5)])  # two triangles + bridge
kappas = rn.all_curvatures(A)
# array([ 0.75,  0.417,  0.417, -0.333,  0.417,  0.417,  0.75])
```

For each edge `(u, v)`:

> κ(u, v) = 1 − W₁(m_u, m_v) / d(u, v)

where `m_x` is the lazy random walk measure at `x`, `W₁` is the 1-Wasserstein
distance under graph shortest-path cost, and `d(u, v)` is the
shortest-path distance. The `W₁` step is solved by a compact network simplex
(MODI method); no LP library is invoked.

## Why this exists

The reference graph-curvature library, [`GraphRicciCurvature`][grc], depends
on `networkx`, `cvxpy`, `POT`, and `numba`, and has been observed to bus-error
on some compute environments (notably GCP deep-learning VMs as of 2026). For
reproducibility purposes a smaller, self-contained, dependency-free
implementation is useful as a reference, even if not the fastest.

This package provides two implementations:

| Module | Solver | Dependencies | Lines | Use case |
|---|---|---|---|---|
| `ricci_numpy` | Network simplex (pure numpy) | numpy | ~250 | reference, reproducibility |
| `ricci_numpy.fast` | Network simplex (@njit) | numpy + numba | ~280 | production |

Both implementations produce **bit-exact agreement** to scipy's HiGHS LP
solver on the graphs we have tested (max edge curvature error 2.55 × 10⁻¹⁵
across 30 real graphs — see [Benchmark](#benchmark) below).

[grc]: https://github.com/saibalmars/GraphRicciCurvature

## Install

```bash
pip install ricci-numpy           # core (numpy only)
pip install ricci-numpy[fast]     # + numba JIT fast path
pip install ricci-numpy[dev]      # + scipy/networkx for the benchmark suite
```

## API

```python
import ricci_numpy as rn

A = rn.adjacency([(0, 1), (1, 2), (2, 0)])        # 0/1 adjacency from edge list
D = rn.shortest_paths(A)                          # all-pairs BFS (matrix powers)
M = rn.lazy_walk(A, alpha=0.5)                    # row i = m_i

kappas = rn.all_curvatures(A, alpha=0.5)          # κ for each edge (lex order)
W = rn.wasserstein(p, q, C)                       # exact 1-Wasserstein (LP)
features = rn.curvature_features(kappas)          # mean, std, frac_negative, ...
```

The fast path is a drop-in replacement:

```python
import ricci_numpy.fast as rn   # same API, ~10-200x faster (numba JIT)
```

## Validation on canonical graphs

The package ships with a built-in validation suite over Ollivier's canonical
test cases. All three pass to machine precision (≤ 2.78 × 10⁻¹⁶).

| Graph | Expected | `ricci_numpy` | `ricci_numpy.fast` |
|---|---|---|---|
| Path P₆ (interior edges) | κ = 0 | 0.0 | 0.0 |
| Path P₆ (end edges) | κ = 0.5 | 0.5 | 0.5 |
| Complete K₄ (all edges) | κ = 2/3 | 0.6666667 | 0.6666667 |
| Two triangles + bridge edge | κ = −1/3 | −0.3333333 | −0.3333333 |

```bash
python -m ricci_numpy.core      # validate pure-numpy version
python -m ricci_numpy._numba    # validate numba version
pytest tests/                   # full test suite (43 tests: canonical,
                                # random-graph fuzz vs scipy LP, weighted,
                                # disconnected, isolated nodes, edge cases)
```

## Benchmark

Head-to-head on 30 real cosine-threshold sentence graphs (5–32 nodes,
10–364 edges) from a chain-of-thought trace dataset. Reference is scipy's
HiGHS LP via `scipy.optimize.linprog`.

| Implementation | Solver | Total time | Mean / graph | Max \|Δκ\| vs scipy |
|---|---|---:|---:|---:|
| `scipy_lp`         | HiGHS LP (C)              |   8.47 s | 0.282 s | 0 (reference) |
| **`ricci_numpy.fast`** | **Network simplex (@njit)** | **0.46 s** | **0.015 s** | **2.55 × 10⁻¹⁵** |
| `ricci_numpy`      | Network simplex (numpy)   |  85.87 s | 2.862 s | 2.33 × 10⁻¹⁵ |
| `sinkhorn_torch` ⚠️ | log-Sinkhorn (PyTorch, A100 GPU) | 18.01 s | 0.600 s | 3.77 × 10⁻² |
| `sinkhorn_jax` ⚠️   | log-Sinkhorn (JAX, A100 GPU)     | 12.66 s | 0.422 s | 3.77 × 10⁻² |

Hardware: A100-SXM4-40GB GPU + Skylake CPU. See [`benchmarks/bench.py`](benchmarks/bench.py)
to reproduce.

### Three things this benchmark shows

1. **A JIT-compiled network simplex (`ricci_numpy.fast`) is 18× faster than
   scipy's HiGHS** on small dense graphs, at machine precision. The win is
   from removing Python/LP-wrapper overhead — for small problems the math is
   tiny and call overhead dominates.

2. **GPU is the wrong tool for graphs of this size.** Both `sinkhorn_torch`
   and `sinkhorn_jax` run ~0.5 s per graph regardless of graph size — that
   floor is pure GPU launch overhead. On the largest graph in the sample
   (364 edges) the GPU versions are still slower than scipy on CPU.

3. **Sinkhorn hits a precision wall at ~10⁻²** on dense graphs, even with
   aggressive ε annealing down to 10⁻⁵ in log-domain. The bias term
   `O(ε · H(T*))` does not vanish because `H(T*)` grows with support size.
   **Sinkhorn cannot reproduce exact LP results for downstream features that
   use a fixed κ threshold** (e.g. `frac_negative` at κ < −0.05). This is
   a property of the algorithm, not the implementation.

### Reproducing the GPU benchmark

The two Sinkhorn implementations live in [`benchmarks/`](benchmarks/) for
reference. They are NOT exposed as a package API — by design, since they fail
the precision test for this use case.

```bash
pip install ricci-numpy[dev] torch "jax[cuda12]"
python benchmarks/bench.py --traces /path/to/cached/embeddings/
```

## Weighted graphs

Pass a `weights` matrix (same shape as A, zero off-edges, positive on edges)
to use Floyd–Warshall shortest paths instead of unweighted BFS:

```python
W = A.astype(float)
W[2, 3] = W[3, 2] = 5.0   # heavy edge
kappas = rn.all_curvatures(A, weights=W)
```

## When `ricci_numpy` is *not* the right choice

- **You have very large sparse graphs (n > a few thousand).** The matrix-power
  BFS in `shortest_paths` is O(n³); Floyd–Warshall in `shortest_paths(A, weights=W)`
  is also O(n³). Use a sparse-graph library.
- **You need to scale to GPU.** Use [`GraphRicciCurvature`][grc] or roll your
  own with batched Sinkhorn — but see the precision wall above.
- **You need other curvature notions** (Forman, Bochner-Lichnerowicz, etc.).
  Out of scope.

## Implementation notes

`ricci_numpy/core.py` is one file, < 250 lines, organized as:

```
adjacency, shortest_paths, lazy_walk           # graph primitives
_northwest_corner, _is_forest, _solve_duals    # LP helpers
_find_cycle, _transport_lp                     # network simplex
wasserstein                                    # public W_1 wrapper
edge_curvature, all_curvatures                 # curvature
curvature_features                             # summary stats
validate                                       # built-in test suite
```

The LP uses the classical [Modified Distribution (MODI) method][modi]:
northwest-corner initial basis → solve duals on the basis tree → find the
most-negative reduced cost → identify the cycle in basis ∪ {entering cell}
via BFS in the bipartite basis graph → pivot. Degeneracy is handled by
padding the basis with zero-flow cells that preserve the forest property.

[modi]: https://en.wikipedia.org/wiki/Stepping-stone_method

`ricci_numpy/_numba.py` is the same algorithm with array-only data structures
and `@numba.njit(cache=True)` decorators on the hot loops.

## Citation

If you use this in published work:

```bibtex
@software{ricci_numpy_2026,
  author  = {Balaji, Arjun},
  title   = {ricci-numpy: Ollivier-Ricci curvature on graphs in pure numpy},
  url     = {https://github.com/ArjunBalaji79/ricci-numpy},
  version = {0.1.0},
  year    = {2026}
}
```

Underlying mathematics:

- Y. Ollivier, *Ricci curvature of Markov chains on metric spaces*, Journal
  of Functional Analysis 256 (2009), 810-864.
- G. B. Dantzig, *Application of the simplex method to a transportation
  problem*, in *Activity Analysis of Production and Allocation* (1951).

## License

MIT. See [LICENSE](LICENSE).
