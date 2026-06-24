"""
EXECUTE D2 (PrOntoQA): does the argmin Ollivier-Ricci edge localize the broken
deduction step?

Subset: PrOntoQA engine-incorrect traces with an identifiable invalid inference
(mode M3) whose broken sentence maps to a node and whose graph has >=2 edges.
M1 (no answer) and M2 (wrong-by-omission, no present broken step) are excluded.

Methods (one edge per trace): CURVATURE (argmin exact kappa), MAX-TRANSITION-
ENERGY (max 1-cos over connected pairs -- the BAR), RANDOM (analytic expectation),
LAST-EDGE (incident to highest sentence index).

Hit: edge (u,v) hits broken step b iff min(|u-b|,|v-b|) <= 1 (within one step).

Writes results/20_d2_localization.json and results/22_d2_backend_argmin_check.json.
The alignment audit sample is written for manual review (21).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "ricci-numpy"))
from prontoqa_logic import gold_answer, model_answer_bool
from labels import prontoqa_broken_sentence, primary_correct
from features import embed
import ricci_numpy as rn  # reference solver for the backend recheck

TOL = 1   # "within one step"


def hit(edge, b, tol=TOL):
    u, v = edge
    return int(min(abs(u - b), abs(v - b)) <= tol)


def expected_random_hit(edges, b, tol=TOL):
    if not edges:
        return np.nan
    return float(np.mean([hit(e, b, tol) for e in edges]))


def build_subset():
    feats = [json.loads(l) for l in open(ROOT / "results/features.jsonl")
             if '"benchmark": "prontoqa"' in l]
    subset = []
    for f in feats:
        if primary_correct(f) is not False:
            continue
        if model_answer_bool(f.get("final_answer")) is None:
            continue
        gold, info = gold_answer(f["question"])
        b, invprop = prontoqa_broken_sentence(f["sentences"], info.get("entity"),
                                              info.get("derived", set()))
        if b is None:
            continue
        edges = [tuple(e) for e in f["edges"]]
        if len(edges) < 2 or b >= f["n_sentences"]:
            continue
        subset.append({"f": f, "b": b, "invprop": invprop, "edges": edges,
                       "kappas": np.array(f["kappas"])})
    return subset


def trans_energy_edge(f, edges):
    """Edge maximizing 1 - cos(x_i,x_j) (lowest cosine among connected pairs)."""
    X = embed(f["sentences"])             # cached -> fast
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1
    Xn = X / norms
    best, best_e = -1, None
    for (u, v) in edges:
        te = 1.0 - float(Xn[u] @ Xn[v])
        if te > best:
            best, best_e = te, (u, v)
    return best_e


def main():
    subset = build_subset()
    n = len(subset)
    print("D2 subset size:", n)
    rng = np.random.default_rng(0)

    rows = []
    for s in subset:
        f, b, edges, kappas = s["f"], s["b"], s["edges"], s["kappas"]
        # CURVATURE: argmin kappa (edges & kappas share order from features.py)
        cur_e = edges[int(np.argmin(kappas))]
        # MAX TRANSITION ENERGY
        mte_e = trans_energy_edge(f, edges)
        # LAST edge: incident to highest sentence index
        last_e = max(edges, key=lambda e: max(e))
        rows.append({
            "model": f["model"], "problem_idx": f["problem_idx"], "b": b,
            "n_sent": f["n_sentences"], "n_edges": f["n_edges"],
            "min_kappa": float(kappas.min()),
            "hit_curv": hit(cur_e, b), "hit_mte": hit(mte_e, b),
            "hit_last": hit(last_e, b),
            "exp_random": expected_random_hit(edges, b),
            "hit_curv_strict": hit(cur_e, b, 0), "hit_mte_strict": hit(mte_e, b, 0),
            "curv_edge": list(cur_e), "mte_edge": list(mte_e),
        })

    arr = lambda k: np.array([r[k] for r in rows], dtype=float)
    hc, hm, hl, er = arr("hit_curv"), arr("hit_mte"), arr("hit_last"), arr("exp_random")

    def paired_ci(a, b_, n_boot=2000):
        d = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(a), len(a))
            d.append(a[idx].mean() - b_[idx].mean())
        d = np.array(d)
        return {"delta": float(a.mean() - b_.mean()),
                "ci_lo": float(np.percentile(d, 2.5)),
                "ci_hi": float(np.percentile(d, 97.5)),
                "frac_gt0": float(np.mean(d > 0))}

    out = {
        "n_subset": n,
        "hit_rates_within1": {
            "curvature": float(hc.mean()), "max_transition_energy": float(hm.mean()),
            "last_edge": float(hl.mean()), "random_expectation": float(np.nanmean(er)),
        },
        "hit_rates_strict": {
            "curvature": float(arr("hit_curv_strict").mean()),
            "max_transition_energy": float(arr("hit_mte_strict").mean()),
        },
        "BAR_curvature_minus_mte": paired_ci(hc, hm),
        "curvature_minus_random": paired_ci(hc, er),
        "curvature_minus_last": paired_ci(hc, hl),
        "per_model": {},
    }
    for m in sorted(set(r["model"] for r in rows)):
        mi = [i for i, r in enumerate(rows) if r["model"] == m]
        out["per_model"][m] = {"n": len(mi),
                               "curv": float(hc[mi].mean()), "mte": float(hm[mi].mean())}
    json.dump(out, open(ROOT / "results/20_d2_localization.json", "w"), indent=2)
    json.dump(rows, open(ROOT / "results/20_d2_per_trace.json", "w"), indent=2)
    print(json.dumps(out["hit_rates_within1"], indent=2))
    print("BAR (curv-mte):", out["BAR_curvature_minus_mte"])

    # alignment audit sample (n=30) for manual review
    audit_idx = rng.choice(n, size=min(30, n), replace=False)
    audit = []
    for i in sorted(audit_idx.tolist()):
        s = subset[i]; f = s["f"]
        audit.append({"model": f["model"], "problem_idx": f["problem_idx"],
                      "broken_sentence_idx": s["b"], "invalid_prop": s["invprop"],
                      "broken_sentence_text": f["sentences"][s["b"]],
                      "prev_sentence": f["sentences"][s["b"] - 1] if s["b"] > 0 else None})
    json.dump(audit, open(ROOT / "results/21_d2_alignment_audit_sample.json", "w"), indent=2)

    # backend recheck: reference solver argmin == fast argmin, on graphs <=400 edges
    checked = matched = skipped = 0
    import networkx as nx
    from features import split_sentences
    from geom.graph import cosine_threshold_graph
    for s in subset:
        f = s["f"]
        if f["n_edges"] > 400:
            skipped += 1; continue
        X = embed(f["sentences"]); G = cosine_threshold_graph(X)
        A = (nx.to_numpy_array(G, nodelist=sorted(G.nodes())) > 0).astype(int)
        np.fill_diagonal(A, 0)
        k_ref = rn.all_curvatures(A, alpha=0.5)
        edges_ref = [tuple(int(x) for x in e) for e in np.argwhere(np.triu(A, 1) > 0)]
        argmin_ref = edges_ref[int(np.argmin(k_ref))]
        argmin_fast = s["edges"][int(np.argmin(s["kappas"]))]
        checked += 1; matched += int(tuple(argmin_ref) == tuple(argmin_fast))
    bk = {"n_checked": checked, "n_argmin_identical": matched, "n_skipped_large": skipped,
          "all_identical": checked == matched}
    json.dump(bk, open(ROOT / "results/22_d2_backend_argmin_check.json", "w"), indent=2)
    print("backend argmin check:", bk)


if __name__ == "__main__":
    main()
