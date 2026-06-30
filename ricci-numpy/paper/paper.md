---
title: "ricci-numpy: Exact Ollivier–Ricci curvature on graphs in pure NumPy"
tags:
  - Python
  - graph theory
  - discrete curvature
  - optimal transport
  - Wasserstein distance
authors:
  - name: Arjun Balaji
    orcid: 0009-0005-1790-0034
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 24 May 2026
bibliography: paper.bib
---

# Summary

`ricci-numpy` is a small, self-contained Python package for computing
Ollivier–Ricci curvature [@ollivier2009] on weighted-by-edge-count graphs.
For each edge $(u, v)$ in a graph $G$ with shortest-path distance $d(u, v)$,
the curvature is defined as

$$ \kappa(u, v) = 1 - \frac{W_1(m_u, m_v)}{d(u, v)} $$

where $m_x$ is the lazy random walk measure at vertex $x$ and $W_1$ is the
1-Wasserstein (earth mover's) distance under shortest-path cost. The
$W_1$ step — a transportation linear program — is solved *exactly* by a
compact implementation of the classical network simplex (MODI / Modified
Distribution method) [@dantzig1951]. The package depends only on NumPy
[@harris2020array]; an optional Numba JIT path [@lam2015numba] provides a
drop-in fast variant for production use.

The package contains two interchangeable implementations of identical
algorithmic content:

| Module | Lines | Dependencies | Typical runtime per edge |
|---|---|---|---|
| `ricci_numpy.core` | ~250 | numpy | reference (slow) |
| `ricci_numpy.fast` | ~280 | numpy + numba | ~100–1000× faster |

Both are validated against published analytical values for the path graph
$P_6$, the complete graph $K_4$, and a two-triangles-plus-bridge graph to
machine precision (max absolute error $2.78 \times 10^{-16}$). On a
sample of 30 real cosine-threshold sentence-embedding graphs, both produce
bit-exact agreement with `scipy.optimize.linprog`'s HiGHS LP solver
[@huangfu2018parallelizing] (max edge-curvature error $2.55 \times 10^{-15}$).

# Statement of need

The most widely used Python implementation of graph Ricci curvature,
`GraphRicciCurvature` [@ni2019community], depends on a stack including
`networkx`, `cvxpy`, `POT`, and `numba`. In some compute environments
this stack is fragile: we have observed segmentation faults
on Google Cloud Platform deep-learning VMs (CUDA 12 / CUDA 13 builds)
that prevent the package from running at all. Reproducibility of
curvature-based results — in graph machine learning [@topping2022understanding;
@nguyen2023revisiting], in network analysis [@sandhu2015graph], and in
recent work on chain-of-thought reasoning quality — depends on having a
reference implementation that is simple enough to be trusted and small
enough to be audited.

`ricci-numpy` is that reference: one file, ~250 lines, depending only on
NumPy. It is intended for two audiences. First, researchers who want to
understand or audit the computation — the entire algorithm, including the
transportation LP solver, fits in a single readable file. Second,
practitioners who need a dependency-light implementation that produces
results bit-identical to an industrial LP solver and runs on systems where
heavier stacks fail to build or import.

A common alternative for the $W_1$ step is entropic regularization via
Sinkhorn iterations [@cuturi2013sinkhorn], which is GPU-friendly and the
basis of most modern optimal transport libraries. We deliberately use an
exact LP for two reasons. First, Sinkhorn introduces a regularization bias
of $O(\varepsilon \cdot H(T^*))$ where $H(T^*)$ is the entropy of the
optimal transport plan; on dense small graphs (such as those arising from
local neighborhoods of high-degree vertices) this bias does not vanish in
practice even with aggressive $\varepsilon$ annealing in the log domain.
Downstream features defined by a fixed curvature threshold — for example,
the fraction of edges with $\kappa < -0.05$, commonly used as a measure of
graph bottleneck-ness — can flip under this bias. Second, on the small
graphs typical of local neighborhood computations, a compiled exact LP
solver outperforms even a vectorized GPU Sinkhorn because GPU launch
overhead dominates the math.

We document these trade-offs empirically in the package's benchmark
suite. On 30 real graphs (5–32 nodes, 10–364 edges), the Numba-accelerated
exact LP variant runs in 0.46 s total (~18× faster than `scipy.linprog`),
while a vectorized log-domain Sinkhorn batched over edges and run on an
NVIDIA A100 GPU reaches a maximum per-edge curvature error of
$3.77 \times 10^{-2}$ against the LP reference — large enough to flip
threshold-based downstream features.

# Key features

- **One file, ~250 lines, NumPy-only** core implementation
  (`ricci_numpy/core.py`).
- **Exact** 1-Wasserstein via network simplex; bit-exact agreement with
  scipy's HiGHS LP solver on tested graphs.
- **Optional Numba fast path** (`ricci_numpy.fast`) as a drop-in
  replacement; ~10–200× faster on small dense graphs with identical numerics.
- **Built-in validation** against canonical analytical examples
  (`python -m ricci_numpy.core`).
- **Reproducible benchmark suite** (`benchmarks/bench.py`) that compares
  against `scipy.optimize.linprog` and, optionally, GPU Sinkhorn variants
  in PyTorch and JAX (provided in `benchmarks/` for reference).

# Acknowledgements

The implementation strategy draws on the classical transportation simplex
literature and the Python optimal transport community
[@flamary2021pot]. Empirical comparisons against `GraphRicciCurvature` were
performed on cosine-threshold graphs derived from MiniLM sentence
embeddings of large language model chain-of-thought traces.

# References
