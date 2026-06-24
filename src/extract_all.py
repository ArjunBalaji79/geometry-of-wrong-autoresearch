"""
Extract the 9-dim geometric signature + per-edge curvature + surface features
for every trace in data/traces/, regenerating everything from the raw JSONs.

Corpus definition (from EXCLUSIONS.md, the repo's documented rule):
  - 6 models x 3 benchmarks x 200 problems = 3600 raw traces.
  - Apply the 15-trace generation-truncation exclusion (model-blind rule).
  - extract_full() additionally drops traces with <3 sentences or no graph edges
    (no geometry possible). Empty / no-output traces fall out here too.

Outputs:
  - results/features.jsonl : one JSON object per surviving trace with the full
    record (signature, spectral, ricci, surface, edges, kappas, sentences, meta).
  - results/01_extract_summary.json : per-(model,benchmark) counts + drop reasons.

Run with cwd = repo root.  Uses ricci-numpy fast path (bit-exact; see
results/00_curvature_backend_check.json).
"""
import json, sys, time, glob
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import extract_full  # noqa: E402

# 15-trace truncation exclusion (EXCLUSIONS.md), keyed (model_file_tag, benchmark, problem_idx)
TRUNCATION_EXCLUSIONS = {
    ("gemini_2_5_flash", "folio", i) for i in (134, 142, 158, 173, 187)
} | {
    ("gemini_2_5_flash", "prontoqa", i) for i in (15, 60, 110, 120)
} | {
    ("gpt_oss_120b", "folio", 99),
    ("gpt_oss_120b", "prontoqa", 168),
} | {
    ("llama_3_1_8b", "prontoqa", i) for i in (59, 78, 101, 198)
}


def parse_fname(path):
    """data/traces/<benchmark>_<modeltag>.json -> (benchmark, modeltag)."""
    stem = Path(path).stem
    bench = stem.split("_", 1)[0]
    modeltag = stem.split("_", 1)[1]
    return bench, modeltag


def main():
    files = sorted(glob.glob(str(ROOT / "data/traces/*.json")))
    out_path = ROOT / "results/features.jsonl"
    fout = open(out_path, "w")
    summary = defaultdict(lambda: defaultdict(int))
    t0 = time.time()
    n_written = 0
    for f in files:
        bench, modeltag = parse_fname(f)
        d = json.load(open(f))
        for r in d:
            pidx = int(r["problem_idx"])
            key = f"{bench}/{modeltag}"
            summary[key]["raw"] += 1
            if (modeltag, bench, pidx) in TRUNCATION_EXCLUSIONS:
                summary[key]["excl_truncation"] += 1
                continue
            rec, status = extract_full(r["cot_trace"])
            if status != "ok":
                summary[key][f"drop_{status}"] += 1
                continue
            obj = {
                "benchmark": bench,
                "model": modeltag,
                "model_string": r.get("model"),
                "problem_idx": pidx,
                "correct": bool(r["correct"]),
                "ground_truth": r.get("ground_truth"),
                "final_answer": r.get("final_answer"),
                "question": r.get("question"),
                "n_sentences": rec["n_sentences"],
                "n_edges": rec["n_edges"],
                "signature": rec["signature"],
                "spectral": rec["spectral"],
                "ricci": rec["ricci"],
                "surface": rec["surface"],
                "edges": rec["edges"],
                "kappas": rec["kappas"],
                "sentences": rec["sentences"],
            }
            fout.write(json.dumps(obj) + "\n")
            n_written += 1
            summary[key]["kept"] += 1
        print(f"  {bench}/{modeltag}: kept {summary[f'{bench}/{modeltag}']['kept']}"
              f"  ({time.time()-t0:.0f}s elapsed, {n_written} total)", flush=True)
    fout.close()

    totals = defaultdict(int)
    for key, c in summary.items():
        for k, v in c.items():
            totals[k] += v
    out = {"elapsed_sec": time.time() - t0,
           "n_written": n_written,
           "totals": dict(totals),
           "per_cell": {k: dict(v) for k, v in sorted(summary.items())}}
    json.dump(out, open(ROOT / "results/01_extract_summary.json", "w"), indent=2)
    print("\nTOTALS:", json.dumps(dict(totals), indent=2))
    print("kept:", n_written)


if __name__ == "__main__":
    main()
