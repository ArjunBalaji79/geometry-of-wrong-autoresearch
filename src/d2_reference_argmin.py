"""
D2 robustness: recompute the CURVATURE argmin edge for the entire D2 subset using
the pure-numpy REFERENCE solver (ricci_numpy.core), not the fast path. The
backend recheck found 86/88 argmin edges identical and 2 differing at machine-
epsilon curvature ties; this script recomputes the headline hit-rate under the
canonical exact solver to confirm the conclusion is solver-invariant.

Writes results/23_d2_reference_solver.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "ricci-numpy"))
import ricci_numpy as rn  # pure-numpy reference (core)
from prontoqa_logic import gold_answer, model_answer_bool
from labels import prontoqa_broken_sentence, primary_correct
from features import embed
from geom.graph import cosine_threshold_graph
from run_d2 import hit, expected_random_hit, trans_energy_edge

TOL = 1

def main():
    feats = [json.loads(l) for l in open(ROOT / "results/features.jsonl")
             if '"benchmark": "prontoqa"' in l]
    hc, hm, hl, er = [], [], [], []
    n = 0
    for f in feats:
        if primary_correct(f) is not False:
            continue
        if model_answer_bool(f.get("final_answer")) is None:
            continue
        gold, info = gold_answer(f["question"])
        b, _ = prontoqa_broken_sentence(f["sentences"], info.get("entity"),
                                        info.get("derived", set()))
        if b is None:
            continue
        edges = [tuple(e) for e in f["edges"]]
        if len(edges) < 2 or b >= f["n_sentences"]:
            continue
        # reference-solver curvature on the SAME graph
        X = embed(f["sentences"]); G = cosine_threshold_graph(X)
        A = (nx.to_numpy_array(G, nodelist=sorted(G.nodes())) > 0).astype(int)
        np.fill_diagonal(A, 0)
        k_ref = rn.all_curvatures(A, alpha=0.5)
        edges_ref = [tuple(int(x) for x in e) for e in np.argwhere(np.triu(A, 1) > 0)]
        cur_e = edges_ref[int(np.argmin(k_ref))]
        mte_e = trans_energy_edge(f, edges)
        last_e = max(edges, key=lambda e: max(e))
        hc.append(hit(cur_e, b)); hm.append(hit(mte_e, b)); hl.append(hit(last_e, b))
        er.append(expected_random_hit(edges, b))
        n += 1
    hc, hm, hl, er = map(lambda x: np.array(x, float), (hc, hm, hl, er))
    rng = np.random.default_rng(0)
    def paired(a, b_):
        d = [a[i].mean() - b_[i].mean() for i in
             (rng.integers(0, len(a), len(a)) for _ in range(2000))]
        d = np.array([a[idx].mean() - b_[idx].mean()
                      for idx in (rng.integers(0, len(a), len(a)) for _ in range(2000))])
        return {"delta": float(a.mean() - b_.mean()),
                "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}
    out = {"solver": "ricci_numpy (pure-numpy reference, core)", "n_subset": n,
           "hit_rates_within1": {"curvature": float(hc.mean()),
                                  "max_transition_energy": float(hm.mean()),
                                  "last_edge": float(hl.mean()),
                                  "random_expectation": float(np.nanmean(er))},
           "BAR_curvature_minus_mte": paired(hc, hm),
           "curvature_minus_random": paired(hc, er),
           "curvature_minus_last": paired(hc, hl)}
    json.dump(out, open(ROOT / "results/23_d2_reference_solver.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
