"""
Apply correctness re-derivation + failure-mode taxonomy to every extracted trace.
Writes results/labels.jsonl (one row per trace, joinable to features.jsonl by
(benchmark, model, problem_idx)) and results/02_label_summary.json with:
  - recomputed correctness vs released-flag agreement (per benchmark/model),
  - failure-mode distribution per benchmark,
  - parse-failure counts.
"""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from labels import (correctness_and_mode, primary_correct, failure_mode,
                    MODE_NAMES, TAU_REL)

def main():
    feats = [json.loads(l) for l in open(ROOT / "results/features.jsonl")]
    fout = open(ROOT / "results/labels.jsonl", "w")
    agree = defaultdict(lambda: [0, 0])          # recompute vs released flag
    mode_dist = defaultdict(Counter)             # benchmark -> mode counts (primary)
    corr_counts = defaultdict(lambda: Counter()) # primary correctness counts
    unrec = defaultdict(int)
    for rec in feats:
        b = rec["benchmark"]
        # --- primary labels (policy: flag for gsm8k/folio, engine for prontoqa)
        pc = primary_correct(rec)
        mode = None if pc is not False else failure_mode(rec)
        # --- independent recompute, for the corruption audit only
        out = correctness_and_mode(rec)
        row = {
            "benchmark": b, "model": rec["model"], "problem_idx": rec["problem_idx"],
            "released_correct": bool(rec["correct"]),
            "primary_correct": pc, "mode": mode,
            "recomputed_correct": out["is_correct"],
        }
        fout.write(json.dumps(row) + "\n")
        if out["is_correct"] is not None:
            for k in (f"{b}/{rec['model']}", b + "__ALL"):
                agree[k][1] += 1
                agree[k][0] += int(bool(out["is_correct"]) == bool(rec["correct"]))
        if pc is None:
            unrec[b] += 1
            continue
        corr_counts[b][bool(pc)] += 1
        if mode is not None:
            mode_dist[b][MODE_NAMES[mode]] += 1

    fout.close()
    summary = {
        "tau_rel": TAU_REL,
        "policy": "correctness: released flag for gsm8k/folio, engine for prontoqa",
        "recomputed_vs_released_agreement": {
            k: {"agree": v[0], "total": v[1], "rate": round(v[0] / v[1], 4)}
            for k, v in sorted(agree.items())},
        "primary_correct_counts": {b: dict(c) for b, c in corr_counts.items()},
        "mode_distribution": {b: dict(c) for b, c in mode_dist.items()},
        "unrecoverable_prontoqa": dict(unrec),
    }
    json.dump(summary, open(ROOT / "results/02_label_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
