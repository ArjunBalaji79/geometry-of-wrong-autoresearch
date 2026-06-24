"""
Verify the numba fast path (ricci_numpy.fast) is bit-exact to the pure-numpy
reference (ricci_numpy) on REAL trace graphs from this corpus, not just the
canonical validation set. We use the fast path for the bulk D1/D2 run only
because it is the identical exact network-simplex algorithm; this script is the
evidence for that claim. Writes results/00_curvature_backend_check.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "ricci-numpy"))
sys.path.insert(0, str(ROOT / "src"))

import ricci_numpy as rn
import ricci_numpy.fast as rnf
from features import split_sentences, embed
from geom.graph import cosine_threshold_graph

def main():
    import glob
    files = sorted(glob.glob(str(ROOT / "data/traces/*.json")))
    rng = np.random.default_rng(0)
    max_diff = 0.0
    n_graphs = 0
    n_edges = 0
    # sample a spread of traces across all 18 files
    for f in files:
        d = json.load(open(f))
        idxs = rng.choice(len(d), size=min(4, len(d)), replace=False)
        for i in idxs:
            sents = split_sentences(d[int(i)]["cot_trace"])
            if len(sents) < 3:
                continue
            X = embed(sents)
            G = cosine_threshold_graph(X)
            if G.number_of_edges() == 0:
                continue
            nodes = sorted(G.nodes())
            A = (nx.to_numpy_array(G, nodelist=nodes) > 0).astype(int)
            np.fill_diagonal(A, 0)
            k_ref = rn.all_curvatures(A, alpha=0.5)
            k_fast = rnf.all_curvatures(A, alpha=0.5)
            if k_ref.size:
                max_diff = max(max_diff, float(np.max(np.abs(k_ref - k_fast))))
                n_edges += int(k_ref.size)
            n_graphs += 1
    out = {"n_graphs": n_graphs, "n_edges_checked": n_edges,
           "max_abs_diff_fast_vs_reference": max_diff,
           "bit_exact": max_diff == 0.0}
    Path(ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/00_curvature_backend_check.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
