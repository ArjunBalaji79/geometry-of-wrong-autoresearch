"""
six_model_rerun.py -- 6-model re-run of orthogonality / D1 / D2 with the
EXCLUSIONS.md truncation exclusion, reported at 4 corpus configs.

Configs:
  4m_incl  original 4 models, truncations INCLUDED  (contaminated baseline)
  4m_excl  original 4 models, truncations EXCLUDED  (decontaminated 4-model)
  6m_excl  6 models, truncations EXCLUDED           (clean target corpus)
  6m_incl  6 models, truncations INCLUDED           (with/without dual-report)

- 4 original models: features loaded from the locked rerun_phase1 JSONL.
- 2 new models (gpt-4o-mini, gemini-2.5-flash): features extracted fresh via
  rerun_common.extract_features (MiniLM, eps=0.3, sequential fallback, 9 feats).
- ANCHOR: 4m_incl orthogonality must reproduce results/rerun_phase3_stage1
  -> else STOP.

seed=0. Existing trace data only. Writes SIX_MODEL_RERUN_<ts>.json.
"""
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerun_common as rc            # noqa: E402
import stage1_lock_floor as s1       # noqa: E402
import exp_d1_gate as d1             # noqa: E402
import exp_d2_fingerprint_gate as d2  # noqa: E402

ROOT = Path(".")
RESULTS = ROOT / "results"

ORIG4 = ["claude_sonnet", "gpt_oss_120b", "llama_3_1_8b", "mistral_7b"]
NEW2 = ["gpt_4o_mini", "gemini_2_5_flash"]
ALLOW6 = ORIG4 + NEW2
DOMAINS = ["folio", "gsm8k", "prontoqa"]

# excluded trace_ids (EXCLUSIONS.md) -- "<domain>_<slug>__<problem_idx>"
EXCLUDE = set()
for _i in (134, 142, 158, 173, 187):
    EXCLUDE.add("folio_gemini_2_5_flash__%d" % _i)
for _i in (15, 60, 110, 120):
    EXCLUDE.add("prontoqa_gemini_2_5_flash__%d" % _i)
EXCLUDE.add("folio_gpt_oss_120b__99")
EXCLUDE.add("prontoqa_gpt_oss_120b__168")
for _i in (59, 78, 101, 198):
    EXCLUDE.add("prontoqa_llama_3_1_8b__%d" % _i)
assert len(EXCLUDE) == 15, len(EXCLUDE)


def latest(pat):
    h = sorted(glob.glob(str(RESULTS / pat)))
    if not h:
        sys.exit("missing: " + pat)
    return h[-1]


def slug_of(trace_id):
    """'gsm8k_claude_sonnet__0' -> 'claude_sonnet'."""
    stem = trace_id.rsplit("__", 1)[0]
    return stem.split("_", 1)[1]


# --------------------------------------------------------------------------
# feature extraction (new models only)
# --------------------------------------------------------------------------
def extract_new_model(slug):
    recs, skipped = [], []
    for dom in DOMAINS:
        path = "data/traces/%s_%s.json" % (dom, slug)
        stem = "%s_%s" % (dom, slug)
        data = json.load(open(path))
        for i, tr in enumerate(data):
            idx = tr.get("problem_idx", i)
            tid = "%s__%s" % (stem, idx)
            feat, status = rc.extract_features(tr.get("cot_trace") or "")
            if status != "ok":
                skipped.append({"trace_id": tid, "status": status})
                continue
            recs.append({
                "trace_id": tid, "model": tr["model"], "domain": dom,
                "correct": bool(tr["correct"]),
                "n_sentences": feat["n_sentences"], "n_edges": feat["n_edges"],
                "spectral": feat["spectral"], "ricci": feat["ricci"],
            })
        print("  [extract] %-26s extracted=%d" % (stem, sum(
            1 for r in recs if r["domain"] == dom)))
    return recs, skipped


def feature_sanity(recs_by_slug):
    """NaN/inf, degenerate-graph, range checks per model."""
    rep = {}
    for slug, recs in recs_by_slug.items():
        if not recs:
            rep[slug] = {"n": 0}
            continue
        nsent = np.array([r["n_sentences"] for r in recs], float)
        nedge = np.array([r["n_edges"] for r in recs], float)
        dens = nedge / np.maximum(nsent * (nsent - 1) / 2.0, 1.0)
        d = {"n": len(recs),
             "n_sentences": {"min": float(nsent.min()), "max": float(nsent.max()),
                             "mean": float(nsent.mean())},
             "edge_density": {"min": float(dens.min()), "max": float(dens.max()),
                              "mean": float(dens.mean())},
             "degenerate_chain": int(np.sum(nedge == nsent - 1)),
             "near_complete": int(np.sum(dens > 0.99)),
             "features": {}}
        for f in d2.GEOM9:
            v = np.array([(r["spectral"].get(f) if f in d2.SPEC
                           else r["ricci"].get(f)) for r in recs], float)
            d["features"][f] = {"min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                                "mean": float(np.nanmean(v)),
                                "n_nonfinite": int(np.sum(~np.isfinite(v)))}
        rep[slug] = d
    return rep


# --------------------------------------------------------------------------
# orthogonality
# --------------------------------------------------------------------------
def disagreement_stats(orth):
    rates = [r["disagreement_rate"] for r in orth]
    return {"n_conditions": len(orth),
            "min_pct": min(rates) * 100.0, "max_pct": max(rates) * 100.0,
            "mean_pct": float(np.mean(rates) * 100.0),
            "median_pct": float(np.median(rates) * 100.0)}


# --------------------------------------------------------------------------
# D1 -- gate logic verbatim from exp_d1_gate.run_gate
# --------------------------------------------------------------------------
def run_gate(S, R, label):
    print("    [d1] gate %s  S=%d R=%d" % (label, len(S), len(R)))
    g = {"n_S": len(S), "n_R": len(R)}
    Ln = {"S": np.array([r["n_sentences"] for r in S], float),
          "R": np.array([r["n_sentences"] for r in R], float)}
    g["3d_length"] = {}
    for fld in ("n_sentences", "n_edges"):
        s = [r[fld] for r in S]
        rr = [r[fld] for r in R]
        dd, p = d1.cliffs_delta(s, rr)
        g["3d_length"][fld] = {"S_median": float(np.median(s)),
                               "R_median": float(np.median(rr)),
                               "cliffs_delta": dd, "mw_p": p}
    g["3a_text"] = {}
    for prop in d1.TEXT_PROPS:
        sc = np.array([d1.text_counts(r["cot_trace"])[prop] for r in S], float)
        rcc = np.array([d1.text_counts(r["cot_trace"])[prop] for r in R], float)
        s_rate = sc / np.maximum(Ln["S"], 1)
        r_rate = rcc / np.maximum(Ln["R"], 1)
        d_raw, p_raw = d1.cliffs_delta(sc, rcc)
        d_rate, p_rate = d1.cliffs_delta(s_rate, r_rate)
        allc = np.concatenate([sc, rcc])
        allL = np.concatenate([Ln["S"], Ln["R"]])
        res = d1.residualize(allc, allL)
        d_res, p_res = d1.cliffs_delta(res[:len(sc)], res[len(sc):])
        g["3a_text"][prop] = {
            "raw": {"S_median": float(np.median(sc)), "R_median": float(np.median(rcc)),
                    "cliffs_delta": d_raw, "mw_p": p_raw},
            "per_sentence_rate": {"S_median": float(np.median(s_rate)),
                                  "R_median": float(np.median(r_rate)),
                                  "cliffs_delta": d_rate, "mw_p": p_rate},
            "length_residualized": {"cliffs_delta": d_res, "mw_p": p_res},
        }
    for axis, key in (("3b_domain", "domain"), ("3c_model", "model")):
        levels = sorted(set(r[key] for r in S + R))
        tab = [[sum(1 for r in S if r[key] == lv) for lv in levels],
               [sum(1 for r in R if r[key] == lv) for lv in levels]]
        try:
            chi2, p, dof, _ = chi2_contingency(tab)
        except ValueError:
            chi2, p = float("nan"), float("nan")
        g[axis] = {"levels": levels,
                   "S_counts": dict(zip(levels, tab[0])),
                   "R_counts": dict(zip(levels, tab[1])),
                   "chi2": float(chi2), "p": float(p)}
    return g


def run_d1(recs):
    for r in recs:
        r["y"] = 0 if r["correct"] else 1
        r["spectral_entropy"] = r["spectral"]["spectral_entropy"]
        r["frac_negative"] = r["ricci"]["frac_negative"]
        r["_grp"] = None
    conds = {}
    for r in recs:
        conds.setdefault((r["model"], r["domain"]), []).append(r)
    recon = []
    for (model, domain), rows in sorted(conds.items()):
        y = np.array([r["y"] for r in rows])
        if len(set(y.tolist())) < 2 or len(rows) < 20:
            continue
        sp = np.array([r["spectral_entropy"] for r in rows])
        rn = np.array([r["frac_negative"] for r in rows])
        sp_pred = d1.median_thresholded_preds(sp, y)
        rn_pred = d1.median_thresholded_preds(rn, y)
        sp_right = sp_pred == y
        rn_right = rn_pred == y
        for r, sr, rr in zip(rows, sp_right, rn_right):
            r["_grp"] = ("S" if (sr and not rr) else
                         "R" if (rr and not sr) else None)
        recon.append({"model": model, "domain": domain, "n": len(rows),
                      "only_sp": int(np.sum(sp_right & ~rn_right)),
                      "only_ri": int(np.sum(~sp_right & rn_right))})
    S_all = [r for r in recs if r["_grp"] == "S"]
    R_all = [r for r in recs if r["_grp"] == "R"]
    S_err = [r for r in S_all if r["y"] == 1]
    R_err = [r for r in R_all if r["y"] == 1]
    g = run_gate(S_err, R_err, "incorrect-only (PRIMARY)")
    g2 = run_gate(S_all, R_all, "all-traces (secondary)")
    NONTRIVIAL = 0.20
    survivors = []
    for prop, dd in g["3a_text"].items():
        rate, res = dd["per_sentence_rate"], dd["length_residualized"]
        if (abs(rate["cliffs_delta"]) >= NONTRIVIAL and rate["mw_p"] < 0.05 and
                abs(res["cliffs_delta"]) >= NONTRIVIAL and res["mw_p"] < 0.05):
            survivors.append("3a:%s (rate d=%.2f, residual d=%.2f)" % (
                prop, rate["cliffs_delta"], res["cliffs_delta"]))
    for axis in ("3b_domain", "3c_model"):
        if g[axis]["p"] < 0.05:
            survivors.append("%s (chi2 p=%.1e)" % (axis, g[axis]["p"]))
    verdict = "ALIVE" if survivors else "DEAD"
    if not survivors:
        borderline = any(abs(dd["per_sentence_rate"]["cliffs_delta"]) >= 0.10
                         for dd in g["3a_text"].values()) \
            or g["3b_domain"]["p"] < 0.2 or g["3c_model"]["p"] < 0.2
        if borderline:
            verdict = "WEAK"
    return {"step1_reconstruction": recon,
            "step2_pool": {"all_traces": {"S": len(S_all), "R": len(R_all)},
                           "incorrect_only": {"S": len(S_err), "R": len(R_err)}},
            "gate_incorrect_only_primary": g,
            "gate_all_traces_secondary": g2,
            "verdict": verdict, "surviving_axes": survivors}


# --------------------------------------------------------------------------
# D2 -- replicates exp_d2_fingerprint_gate.main (fresh rng per config)
# --------------------------------------------------------------------------
def run_d2(recs):
    rng = np.random.default_rng(d2.SEED)
    models = sorted(set(r["model"] for r in recs))
    m2i = {m: i for i, m in enumerate(models)}
    y = np.array([m2i[r["model"]] for r in recs])
    groups = np.array(["%s|%s" % (r["domain"], r["trace_id"].rsplit("__", 1)[-1])
                       for r in recs])
    feat = {f: np.array([(r["spectral"].get(f) if f in d2.SPEC else
                          r["ricci"].get(f) if f in d2.RICCI else r[f])
                         for r in recs], float)
            for f in d2.GEOM9 + ["n_sentences", "n_edges"]}
    cc = Counter(y.tolist())

    def X_of(keys):
        return np.column_stack([feat[k] for k in keys])

    size_corr = {f: float(abs(np.corrcoef(feat[f], feat["n_edges"])[0, 1]))
                 for f in d2.GEOM9}
    size_feats = [f for f in d2.GEOM9 if size_corr[f] >= 0.5]
    shape_feats = [f for f in d2.GEOM9 if size_corr[f] < 0.5]
    featuresets = {
        "A_length_only": ["n_sentences", "n_edges"],
        "B_geometry9": d2.GEOM9,
        "C_geometry9_plus_length": d2.GEOM9 + ["n_sentences", "n_edges"],
        "B_shape_subset": shape_feats,
        "B_size_subset": size_feats,
    }
    out = {"n_traces": len(recs), "n_classes": len(models), "models": models,
           "chance": 1.0 / len(models),
           "class_counts": {models[i]: cc.get(i, 0) for i in range(len(models))},
           "size_correlation_with_n_edges": size_corr,
           "shape_subset": shape_feats, "size_subset": size_feats,
           "results": {}, "oof": {}}
    for fs, keys in featuresets.items():
        if not keys:
            continue
        X = X_of(keys)
        out["results"][fs] = {}
        for algo in ("gbt", "rf", "lr"):
            oof = d2.cv_oof_pred(X, y, groups, d2.FACTORIES[algo])
            mt = d2.metrics_with_ci(y, oof, groups, rng)
            pr, rc_, f1c, sup = precision_recall_fscore_support(
                y, oof, zero_division=0)
            mt["per_class"] = {models[i]: {"precision": float(pr[i]),
                                           "recall": float(rc_[i]),
                                           "f1": float(f1c[i]),
                                           "support": int(sup[i])}
                               for i in range(len(models))}
            out["results"][fs][algo] = mt
            if algo == "gbt":
                out["oof"][fs] = oof.tolist()
            print("    [d2] %-26s %-3s acc=%.3f ci=%s" % (
                fs, algo, mt["accuracy"],
                [round(x, 3) for x in mt["accuracy_ci"]]))
    oA = np.array(out["oof"]["A_length_only"])
    oB = np.array(out["oof"]["B_geometry9"])
    oC = np.array(out["oof"]["C_geometry9_plus_length"])
    oBsh = np.array(out["oof"]["B_shape_subset"])
    dBA = d2.paired_delta_acc(y, oB, oA, groups, rng)
    dCA = d2.paired_delta_acc(y, oC, oA, groups, rng)
    dShA = d2.paired_delta_acc(y, oBsh, oA, groups, rng)
    alive = bool(dBA["ci_excludes_zero"] and dBA["delta_accuracy"] > 0
                 and dCA["ci_excludes_zero"] and dCA["delta_accuracy"] > 0)
    out["verdict"] = {"B_minus_A": dBA, "C_minus_A": dCA, "B_shape_minus_A": dShA,
                      "direction2": "ALIVE" if alive else "DEAD"}
    return out


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("SIX-MODEL RE-RUN  %s" % ts)
    print("=" * 70)

    # ---- locked 4-model features ----
    locked_path = latest("rerun_phase1_features_*.jsonl")
    locked = [json.loads(l) for l in open(locked_path)]
    for r in locked:
        r["_slug"] = slug_of(r["trace_id"])
    locked_slugs = sorted(set(r["_slug"] for r in locked))
    print("[load] locked phase1: %s  (%d traces; slugs=%s)" % (
        Path(locked_path).name, len(locked), locked_slugs))
    if set(locked_slugs) != set(ORIG4):
        sys.exit("STOP: locked phase1 slugs %s != ORIG4 %s" % (locked_slugs, ORIG4))

    # ---- fresh features for the 2 new models ----
    new_by_slug, new_recs, skipped_all = {}, [], []
    for slug in NEW2:
        print("[extract] %s" % slug)
        recs, skipped = extract_new_model(slug)
        for r in recs:
            r["_slug"] = slug
        new_by_slug[slug] = recs
        new_recs += recs
        skipped_all += skipped
        print("  -> %s: %d extracted, %d skipped" % (slug, len(recs), len(skipped)))

    # ---- feature sanity ----
    orig_by_slug = {s: [r for r in locked if r["_slug"] == s] for s in ORIG4}
    sanity_new = feature_sanity(new_by_slug)
    sanity_orig = feature_sanity(orig_by_slug)

    # ---- cot_trace for all 6 models (D1 text analysis) ----
    text = {}
    for slug in ALLOW6:
        for dom in DOMAINS:
            stem = "%s_%s" % (dom, slug)
            for i, tr in enumerate(json.load(open("data/traces/%s.json" % stem))):
                idx = tr.get("problem_idx", i)
                text["%s__%s" % (stem, idx)] = tr.get("cot_trace") or ""

    all_recs = locked + new_recs
    for r in all_recs:
        r["cot_trace"] = text.get(r["trace_id"], "")
        r["_excluded"] = r["trace_id"] in EXCLUDE
    n_excl_present = sum(1 for r in all_recs if r["_excluded"])
    miss_text = sum(1 for r in all_recs if not r["cot_trace"])
    print("[corpus] %d feature records; %d/15 excluded trace_ids present in "
          "feature space; %d missing raw text" % (
              len(all_recs), n_excl_present, miss_text))

    # ---- configs ----
    def make_config(slugs, exclude_trunc):
        rs = [r for r in all_recs if r["_slug"] in slugs]
        if exclude_trunc:
            rs = [r for r in rs if not r["_excluded"]]
        return [dict(r) for r in rs]

    configs = {
        "4m_incl": make_config(ORIG4, False),
        "4m_excl": make_config(ORIG4, True),
        "6m_excl": make_config(ALLOW6, True),
        "6m_incl": make_config(ALLOW6, False),
    }
    for k, v in configs.items():
        print("[config] %-8s n=%d" % (k, len(v)))

    # ---- ANCHOR: 4m_incl orthogonality must reproduce rerun_phase3 Table 1 ----
    p3 = json.load(open(latest("rerun_phase3_stage1_*.json")))
    ref = {(o["model"], o["domain"]): o for o in p3["orthogonality"]}
    anchor_orth = s1.orthogonality_per_condition(configs["4m_incl"])
    anchor_disc = []
    for o in anchor_orth:
        k = (o["model"], o["domain"])
        if k not in ref:
            anchor_disc.append("%s not in phase3 ref" % str(k))
            continue
        for fld in ("n", "both_right", "only_sp", "only_ri", "both_wrong"):
            if o[fld] != ref[k][fld]:
                anchor_disc.append("%s.%s rerun=%s ref=%s" % (
                    k, fld, o[fld], ref[k][fld]))
    print("[anchor] 4m_incl orthogonality vs rerun_phase3 Table 1: %s" % (
        "MATCH" if not anchor_disc else "MISMATCH"))
    for x in anchor_disc[:10]:
        print("   ", x)
    if anchor_disc:
        sys.exit("ANCHOR FAIL: 4m_incl does not reproduce Table 1. STOP.")

    # ---- run all 4 configs ----
    rerun = {}
    for name, recs in configs.items():
        print("\n" + "-" * 60)
        print("CONFIG %s  (n=%d)" % (name, len(recs)))
        print("-" * 60)
        orth = s1.orthogonality_per_condition(recs)
        dstat = disagreement_stats(orth)
        print("  [orth] %d conditions  disagreement %.1f%%-%.1f%% (mean %.1f%%)" % (
            dstat["n_conditions"], dstat["min_pct"], dstat["max_pct"],
            dstat["mean_pct"]))
        d1res = run_d1(recs)
        print("  [d1] verdict=%s survivors=%s" % (
            d1res["verdict"], d1res["surviving_axes"] or "NONE"))
        d2res = run_d2(recs)
        v = d2res["verdict"]
        print("  [d2] verdict=%s A=%.3f B=%.3f C=%.3f  B-A=%+.3f%s" % (
            v["direction2"],
            d2res["results"]["A_length_only"]["gbt"]["accuracy"],
            d2res["results"]["B_geometry9"]["gbt"]["accuracy"],
            d2res["results"]["C_geometry9_plus_length"]["gbt"]["accuracy"],
            v["B_minus_A"]["delta_accuracy"], v["B_minus_A"]["ci"]))
        rerun[name] = {"n_traces": len(recs), "orthogonality": orth,
                       "disagreement_stats": dstat, "d1": d1res, "d2": d2res}

    # ---- anchor 2: 4m_incl vs existing gate results (informational) ----
    anchor2 = {}
    try:
        ed2 = json.load(open(latest("exp_d2_fingerprint_gate_*.json")))
        prev_B = ed2["results"]["B_geometry9"]["gbt"]["accuracy"]
        cur_B = rerun["4m_incl"]["d2"]["results"]["B_geometry9"]["gbt"]["accuracy"]
        anchor2["d2_B_geometry9_gbt"] = {"existing": prev_B, "rerun_4m_incl": cur_B,
                                         "abs_diff": abs(prev_B - cur_B)}
        print("\n[anchor2] D2 B_geometry9 gbt acc: existing=%.6f  4m_incl=%.6f  "
              "diff=%.2e" % (prev_B, cur_B, abs(prev_B - cur_B)))
    except SystemExit:
        print("[anchor2] no existing exp_d2 result to compare")
    try:
        ed1 = json.load(open(latest("exp_d1_gate_*.json")))
        anchor2["d1_verdict"] = {"existing": ed1.get("verdict", {}).get("direction1"),
                                 "rerun_4m_incl": rerun["4m_incl"]["d1"]["verdict"]}
        print("[anchor2] D1 verdict: existing=%s  4m_incl=%s" % (
            anchor2["d1_verdict"]["existing"], anchor2["d1_verdict"]["rerun_4m_incl"]))
    except SystemExit:
        print("[anchor2] no existing exp_d1 result to compare")

    # ---- write ----
    out = {
        "meta": {
            "timestamp": ts, "seed": d2.SEED,
            "allowlist_6": ALLOW6, "orig4": ORIG4, "new2": NEW2,
            "locked_phase1_source": Path(locked_path).name,
            "n_excluded_truncated_total": len(EXCLUDE),
            "n_excluded_present_in_feature_space": n_excl_present,
            "excluded_trace_ids": sorted(EXCLUDE),
            "config_sizes": {k: len(v) for k, v in configs.items()},
        },
        "feature_sanity_new": sanity_new,
        "feature_sanity_orig4": sanity_orig,
        "new_model_skipped": skipped_all,
        "anchor_table1": "MATCH",
        "anchor2_vs_existing_gates": anchor2,
        "configs": rerun,
    }
    op = ROOT / ("SIX_MODEL_RERUN_%s.json" % ts)
    json.dump(out, open(op, "w"), indent=2)

    # ---- comparison summary ----
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print("\nORTHOGONALITY disagreement band:")
    for k in ("4m_incl", "4m_excl", "6m_excl", "6m_incl"):
        s = rerun[k]["disagreement_stats"]
        print("  %-8s %2d cond  %.1f%% - %.1f%%  (mean %.1f%%)" % (
            k, s["n_conditions"], s["min_pct"], s["max_pct"], s["mean_pct"]))
    print("\nD1 (incorrect-only primary):")
    for k in ("4m_incl", "4m_excl", "6m_excl", "6m_incl"):
        g = rerun[k]["d1"]["gate_incorrect_only_primary"]
        nt = g["3a_text"]["numeric_tokens"]
        print("  %-8s verdict=%-5s  S=%d R=%d  numeric_tokens rate_d=%+.2f "
              "residual_d=%+.2f  domain_chi2_p=%.1e" % (
                  k, rerun[k]["d1"]["verdict"], g["n_S"], g["n_R"],
                  nt["per_sentence_rate"]["cliffs_delta"],
                  nt["length_residualized"]["cliffs_delta"],
                  g["3b_domain"]["p"]))
    print("\nD2 (GBT model-identity):")
    for k in ("4m_incl", "4m_excl", "6m_excl", "6m_incl"):
        r = rerun[k]["d2"]
        print("  %-8s %d-way chance=%.3f  A=%.3f B=%.3f C=%.3f  "
              "B-A=%+.3f%s  verdict=%s" % (
                  k, r["n_classes"], r["chance"],
                  r["results"]["A_length_only"]["gbt"]["accuracy"],
                  r["results"]["B_geometry9"]["gbt"]["accuracy"],
                  r["results"]["C_geometry9_plus_length"]["gbt"]["accuracy"],
                  r["verdict"]["B_minus_A"]["delta_accuracy"],
                  r["verdict"]["B_minus_A"]["ci"],
                  r["verdict"]["direction2"]))
    print("\n[done] %.0fs -> %s" % (time.time() - t0, op))
    print("SIXMODEL_OK %s" % op)


if __name__ == "__main__":
    main()
