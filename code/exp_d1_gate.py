"""
exp_d1_gate.py -- Direction 1 gate.

Are the only-spectral-catches (S) and only-Ricci-catches (R) error groups
structurally DISTINCT on axes that are (i) NOT the defining geometric features
(circular) and (ii) NOT trace length (confounded)?

STEP 1: reconstruct the orthogonality groups (spectral_entropy +
        frac_negative thresholded at per-condition
        median, direction-flipped). Verify all-traces only_sp/only_ri reproduce
        Table 1. If not -> STOP.
STEP 2: pool across the 12 in-corpus conditions. S/R built two ways:
        all-traces (the Table-1 object) and incorrect-only (the "kinds of
        error" object). The gate runs on both; incorrect-only is primary.
STEP 3: test S vs R on non-geometric text properties (raw + per-sentence rate
        + length-residualized), domain composition, model composition, and
        length itself.

BINDING RULE: Direction 1 is ALIVE iff S and R differ on >=1 non-geometric,
non-length axis (3a rate/residual, 3b, or 3c) with non-trivial effect size
that survives length control. Defining-feature differences are circular and
do not count; raw-length differences are confounded and do not count.

seed=0. Existing data only.
"""
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score

RESULTS = Path("results")
SEED = 0


def latest(pat):
    h = sorted(glob.glob(str(RESULTS / pat)))
    if not h:
        sys.exit("missing: " + pat)
    return h[-1]


# ---- non-geometric text properties (regex on raw cot_trace) ----
RE = {
    "numeric_tokens": re.compile(r"\d+"),
    "logical_connectives": re.compile(
        r"\b(therefore|thus|because|if|then|so|hence|implies)\b", re.I),
    "negations": re.compile(r"\b(not|no|never)\b|n't", re.I),
    "hedging": re.compile(
        r"\b(maybe|possibly|perhaps|might|could|uncertain|unclear)\b", re.I),
    "contradiction_markers": re.compile(
        r"\b(but|however|although|contradiction)\b", re.I),
    "self_correction": re.compile(
        r"\?|\b(wait|actually)\b|let me reconsider", re.I),
}
TEXT_PROPS = list(RE.keys())


def text_counts(txt):
    return {k: len(RE[k].findall(txt or "")) for k in TEXT_PROPS}


def median_thresholded_preds(values, y):
    """stage1_lock_floor.median_thresholded_preds, verbatim."""
    pred = (values > np.median(values)).astype(int)
    try:
        if roc_auc_score(y, values) < 0.5:
            pred = 1 - pred
    except ValueError:
        pass
    return pred


def cliffs_delta(a, b):
    """Cliff's delta via Mann-Whitney U; returns (delta, p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return 0.0, 1.0
    delta = 2.0 * U / (len(a) * len(b)) - 1.0
    return float(delta), float(p)


def residualize(values, length):
    """Residuals of values regressed on length (linear) -- length control."""
    v, L = np.asarray(values, float), np.asarray(length, float)
    A = np.column_stack([L, np.ones_like(L)])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def main():
    t0 = time.time()
    p1 = [json.loads(l) for l in open(latest("rerun_phase1_features_*.jsonl"))]
    # raw cot_trace by trace_id
    text = {}
    for tf in glob.glob("data/traces/*.json"):
        stem = Path(tf).stem
        if "gemini" in stem:
            continue
        for i, r in enumerate(json.load(open(tf))):
            text["%s__%s" % (stem, r.get("problem_idx", i))] = r.get("cot_trace") or ""
    for r in p1:
        r["cot_trace"] = text.get(r["trace_id"], "")
        r["y"] = 0 if r["correct"] else 1
        r["spectral_entropy"] = r["spectral"]["spectral_entropy"]
        r["frac_negative"] = r["ricci"]["frac_negative"]
    miss = sum(1 for r in p1 if not r["cot_trace"])
    print("[d1] %d in-corpus traces; %d missing raw text" % (len(p1), miss))

    # ---- STEP 1: reconstruct only_sp / only_ri per (model,domain) ----
    conds = {}
    for r in p1:
        conds.setdefault((r["model"], r["domain"]), []).append(r)
    recon = []
    for (model, domain), rows in sorted(conds.items()):
        y = np.array([r["y"] for r in rows])
        if len(set(y.tolist())) < 2 or len(rows) < 20:
            continue
        sp = np.array([r["spectral_entropy"] for r in rows])
        rn = np.array([r["frac_negative"] for r in rows])
        sp_pred = median_thresholded_preds(sp, y)
        rn_pred = median_thresholded_preds(rn, y)
        sp_right = sp_pred == y
        rn_right = rn_pred == y
        for r, sr, rr in zip(rows, sp_right, rn_right):
            r["_grp"] = ("S" if (sr and not rr) else
                         "R" if (rr and not sr) else None)
        only_sp = int(np.sum(sp_right & ~rn_right))
        only_ri = int(np.sum(~sp_right & rn_right))
        recon.append({"model": model, "domain": domain, "n": len(rows),
                      "only_sp": only_sp, "only_ri": only_ri})

    # verify against rerun_phase3 orthogonality (the Table-1 object)
    p3 = json.load(open(latest("rerun_phase3_stage1_*.json")))
    ref = {(o["model"], o["domain"]): o for o in p3["orthogonality"]}
    disc = []
    for c in recon:
        k = (c["model"], c["domain"])
        if k not in ref:
            disc.append("%s: not in ref" % str(k)); continue
        if c["only_sp"] != ref[k]["only_sp"] or c["only_ri"] != ref[k]["only_ri"]:
            disc.append("%s: recon %d/%d vs Table1 %d/%d" % (
                k, c["only_sp"], c["only_ri"], ref[k]["only_sp"], ref[k]["only_ri"]))
    print("[d1] STEP 1 reconstruction vs Table 1: %s" % (
        "MATCH (all 12)" if not disc else "MISMATCH"))
    for d in disc:
        print("   ", d)
    if disc:
        print("STOP: orthogonality reconstruction does not reproduce Table 1.")
        sys.exit(1)

    # ---- STEP 2: pool ----
    S_all = [r for r in p1 if r.get("_grp") == "S"]
    R_all = [r for r in p1 if r.get("_grp") == "R"]
    S_err = [r for r in S_all if r["y"] == 1]
    R_err = [r for r in R_all if r["y"] == 1]
    print("[d1] STEP 2 pooled: all-traces S=%d R=%d | incorrect-only S=%d R=%d" % (
        len(S_all), len(R_all), len(S_err), len(R_err)))

    out = {"meta": {"n_traces": len(p1), "seed": SEED},
           "step1_reconstruction": {"per_condition": recon,
                                    "reproduces_table1": True},
           "step2_pool": {"all_traces": {"S": len(S_all), "R": len(R_all)},
                          "incorrect_only": {"S": len(S_err), "R": len(R_err)}},
           "step3_gate": {}}

    def run_gate(S, R, label):
        print("\n[d1] STEP 3 gate -- %s (S=%d, R=%d)" % (label, len(S), len(R)))
        g = {"n_S": len(S), "n_R": len(R)}
        Ln = {"S": np.array([r["n_sentences"] for r in S], float),
              "R": np.array([r["n_sentences"] for r in R], float)}
        # 3d length
        g["3d_length"] = {}
        for fld in ("n_sentences", "n_edges"):
            s = [r[fld] for r in S]
            rr = [r[fld] for r in R]
            d, p = cliffs_delta(s, rr)
            g["3d_length"][fld] = {"S_median": float(np.median(s)),
                                   "R_median": float(np.median(rr)),
                                   "cliffs_delta": d, "mw_p": p}
        # 3a text properties: raw count, per-sentence rate, length-residualized
        g["3a_text"] = {}
        for prop in TEXT_PROPS:
            sc = np.array([text_counts(r["cot_trace"])[prop] for r in S], float)
            rc = np.array([text_counts(r["cot_trace"])[prop] for r in R], float)
            s_rate = sc / np.maximum(Ln["S"], 1)
            r_rate = rc / np.maximum(Ln["R"], 1)
            d_raw, p_raw = cliffs_delta(sc, rc)
            d_rate, p_rate = cliffs_delta(s_rate, r_rate)
            # residualize raw count on length (pooled fit over S+R)
            allc = np.concatenate([sc, rc])
            allL = np.concatenate([Ln["S"], Ln["R"]])
            res = residualize(allc, allL)
            d_res, p_res = cliffs_delta(res[:len(sc)], res[len(sc):])
            g["3a_text"][prop] = {
                "raw": {"S_median": float(np.median(sc)), "R_median": float(np.median(rc)),
                        "cliffs_delta": d_raw, "mw_p": p_raw},
                "per_sentence_rate": {"S_median": float(np.median(s_rate)),
                                      "R_median": float(np.median(r_rate)),
                                      "cliffs_delta": d_rate, "mw_p": p_rate},
                "length_residualized": {"cliffs_delta": d_res, "mw_p": p_res},
            }
        # 3b domain, 3c model -- chi-squared
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
                       "S_frac": {lv: tab[0][i] / max(len(S), 1) for i, lv in enumerate(levels)},
                       "R_frac": {lv: tab[1][i] / max(len(R), 1) for i, lv in enumerate(levels)},
                       "chi2": float(chi2), "p": float(p)}
        return g

    out["step3_gate"]["incorrect_only_primary"] = run_gate(S_err, R_err, "incorrect-only (PRIMARY)")
    out["step3_gate"]["all_traces_secondary"] = run_gate(S_all, R_all, "all-traces (secondary)")

    # ---- verdict on the primary (incorrect-only) ----
    g = out["step3_gate"]["incorrect_only_primary"]
    NONTRIVIAL = 0.20   # |Cliff's delta| threshold for a non-trivial effect
    survivors = []
    for prop, d in g["3a_text"].items():
        rate, res = d["per_sentence_rate"], d["length_residualized"]
        if (abs(rate["cliffs_delta"]) >= NONTRIVIAL and rate["mw_p"] < 0.05 and
                abs(res["cliffs_delta"]) >= NONTRIVIAL and res["mw_p"] < 0.05):
            survivors.append("3a:%s (rate delta=%.2f, residual delta=%.2f)" % (
                prop, rate["cliffs_delta"], res["cliffs_delta"]))
    for axis in ("3b_domain", "3c_model"):
        if g[axis]["p"] < 0.05:
            survivors.append("%s (chi2 p=%.1e)" % (axis, g[axis]["p"]))
    verdict = "ALIVE" if survivors else "DEAD"
    # WEAK: a length-confounded or borderline signal but nothing clean
    if not survivors:
        borderline = any(
            abs(d["per_sentence_rate"]["cliffs_delta"]) >= 0.10
            for d in g["3a_text"].values()) or g["3b_domain"]["p"] < 0.2 \
            or g["3c_model"]["p"] < 0.2
        if borderline:
            verdict = "WEAK"
    out["verdict"] = {"direction1": verdict, "surviving_axes": survivors,
                      "nontrivial_threshold_cliffs_delta": NONTRIVIAL,
                      "rule": "ALIVE iff a non-geometric non-length axis differs "
                              "(3a per-sentence-rate AND length-residualized both "
                              "|delta|>=0.20 p<0.05, or 3b/3c chi2 p<0.05)."}

    op = RESULTS / ("exp_d1_gate_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(out, open(op, "w"), indent=2)
    print("\nVERDICT: Direction 1 is %s" % verdict)
    print("  surviving non-geometric/non-length axes: %s" % (survivors or "NONE"))
    print("[d1] done (%.0fs) -> %s" % (time.time() - t0, op))
    print("D1_OK %s" % op)


if __name__ == "__main__":
    main()
