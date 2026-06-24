"""
exp_reviewer_holdout_permutation.py -- reviewer comments 1 + 2.

C1: held-out confirmatory split for D1.
    The full-data D1 picks per-condition median thresholds on
    spectral_entropy + frac_negative using the same traces it then tests
    on (numeric-rate / domain / model). The construction and the test
    share information. Fix: per-condition stratified 50/50 split; pick
    thresholds + direction-flip on half A only; assign S/R on half B
    using those frozen rules; pool held-out B's S/R across the 18
    conditions; run the same tests. 50 random splits, plus A/B swap.

C2: permutation-based exact p-values for the chi-square + Cliff's-delta
    tests on the FULL 6m_excl incorrect-only S/R pool (n_S=345, n_R=265).
    Shuffle the S/R label across the 610 rows N=100,000 times, recompute
    chi^2(domain), chi^2(model), and Cliff's delta on numeric-token
    per-sentence rate; report exact perm p side-by-side with the
    asymptotic chi^2 p (Mann-Whitney p for delta).

Corpus: 6m_excl, exactly as six_model_rerun.make_config builds it.
  - 4 orig models: features from rerun_phase1_features_*.jsonl (on disk).
  - 2 new models (gpt_4o_mini, gemini_2_5_flash): features extracted
    fresh via rerun_common.extract_features (MiniLM, eps=0.3) on first
    run; embeddings cached under data/embeddings_cache/.
  - 15 truncation exclusions per EXCLUSIONS.md.

Anchor: with the same seed=0 reconstruction, full-data incorrect-only
pool must produce n_S=345, n_R=265 to match SMR.configs.6m_excl.d1. If
not, STOP.

seed=0; CPU-only; no SBERT pass after first run thanks to the cache.
Writes exp_reviewer_holdout_permutation_<ts>.json and a short
REVIEWER_RESPONSE_NUMBERS.md alongside.
"""
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rerun_common as rc          # noqa: E402
import exp_d1_gate as d1           # noqa: E402

ROOT = Path(".")
RESULTS = ROOT / "results"

ORIG4 = ["claude_sonnet", "gpt_oss_120b", "llama_3_1_8b", "mistral_7b"]
NEW2 = ["gpt_4o_mini", "gemini_2_5_flash"]
ALLOW6 = ORIG4 + NEW2
DOMAINS = ["folio", "gsm8k", "prontoqa"]

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

SEED = 0
N_SPLITS = 50
N_PERM = 100_000

ANCHOR_NS = 345
ANCHOR_NR = 265


# ---------------------------------------------------------------- loading
def slug_of(trace_id):
    stem = trace_id.rsplit("__", 1)[0]
    return stem.split("_", 1)[1]


def latest(pat):
    h = sorted(glob.glob(str(RESULTS / pat)))
    if not h:
        sys.exit("missing: " + pat)
    return h[-1]


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
    return recs, skipped


def load_corpus_6m_excl():
    locked = [json.loads(l) for l in open(latest("rerun_phase1_features_*.jsonl"))]
    for r in locked:
        r["_slug"] = slug_of(r["trace_id"])
    locked_slugs = sorted(set(r["_slug"] for r in locked))
    if set(locked_slugs) != set(ORIG4):
        sys.exit("STOP: locked phase1 slugs %s != ORIG4 %s"
                 % (locked_slugs, ORIG4))
    print("[load] locked phase1: %d traces (4 orig models)" % len(locked))

    new_recs, all_skipped = [], []
    for slug in NEW2:
        t = time.time()
        recs, skipped = extract_new_model(slug)
        for r in recs:
            r["_slug"] = slug
        new_recs += recs
        all_skipped += skipped
        print("  [extract] %s: %d extracted, %d skipped  (%.1fs)"
              % (slug, len(recs), len(skipped), time.time() - t))

    text = {}
    for slug in ALLOW6:
        for dom in DOMAINS:
            stem = "%s_%s" % (dom, slug)
            for i, tr in enumerate(json.load(open(
                    "data/traces/%s.json" % stem))):
                idx = tr.get("problem_idx", i)
                text["%s__%s" % (stem, idx)] = tr.get("cot_trace") or ""

    all_recs = locked + new_recs
    n_excl = 0
    for r in all_recs:
        r["cot_trace"] = text.get(r["trace_id"], "")
        if r["trace_id"] in EXCLUDE:
            n_excl += 1
    assert n_excl == 15, "expected 15 EXCLUDE hits, got %d" % n_excl

    recs_6m_excl = [r for r in all_recs if r["trace_id"] not in EXCLUDE]
    for r in recs_6m_excl:
        r["y"] = 0 if r["correct"] else 1
        r["spectral_entropy"] = r["spectral"]["spectral_entropy"]
        r["frac_negative"] = r["ricci"]["frac_negative"]
    print("[corpus] 6m_excl: %d traces" % len(recs_6m_excl))
    return recs_6m_excl, all_skipped


# ---------------------------------------------- group-by condition helpers
def group_by_condition(recs):
    conds = {}
    for r in recs:
        conds.setdefault((r["model"], r["domain"]), []).append(r)
    keep = {}
    for k, rows in sorted(conds.items()):
        y = np.array([r["y"] for r in rows])
        if len(set(y.tolist())) < 2 or len(rows) < 20:
            continue
        keep[k] = rows
    return keep


# -------------------------------------------- full-data anchor (Step 1+2)
def full_data_S_R(conds):
    """Apply the SMR-style per-condition median-threshold reconstruction
    to the WHOLE condition (no split). Returns the pooled incorrect-only
    S and R lists."""
    S_all, R_all = [], []
    for k, rows in conds.items():
        y = np.array([r["y"] for r in rows])
        sp = np.array([r["spectral_entropy"] for r in rows])
        rn = np.array([r["frac_negative"] for r in rows])
        sp_pred = d1.median_thresholded_preds(sp, y)
        rn_pred = d1.median_thresholded_preds(rn, y)
        sp_right = sp_pred == y
        rn_right = rn_pred == y
        for r, sr, rr in zip(rows, sp_right, rn_right):
            if sr and not rr:
                S_all.append(r)
            elif rr and not sr:
                R_all.append(r)
    S_err = [r for r in S_all if r["y"] == 1]
    R_err = [r for r in R_all if r["y"] == 1]
    return S_err, R_err


# ---------------------------------------------- gate (subset of d1.run_gate)
def gate_numeric_domain_model(S, R):
    """Run only the four headline tests the reviewer is asking about:
    numeric-rate delta, length-residualized delta, domain chi^2(2),
    model chi^2(5). Returns a flat dict."""
    if not S or not R:
        return {"n_S": len(S), "n_R": len(R), "viable": False}
    Ln_S = np.array([r["n_sentences"] for r in S], float)
    Ln_R = np.array([r["n_sentences"] for r in R], float)
    sc = np.array([d1.text_counts(r["cot_trace"])["numeric_tokens"]
                   for r in S], float)
    rc_ = np.array([d1.text_counts(r["cot_trace"])["numeric_tokens"]
                    for r in R], float)
    s_rate = sc / np.maximum(Ln_S, 1)
    r_rate = rc_ / np.maximum(Ln_R, 1)
    d_rate, p_rate = d1.cliffs_delta(s_rate, r_rate)
    allc = np.concatenate([sc, rc_])
    allL = np.concatenate([Ln_S, Ln_R])
    res = d1.residualize(allc, allL)
    d_res, p_res = d1.cliffs_delta(res[:len(sc)], res[len(sc):])

    out = {"n_S": len(S), "n_R": len(R), "viable": True,
           "numeric_rate": {"S_median": float(np.median(s_rate)),
                            "R_median": float(np.median(r_rate)),
                            "cliffs_delta": d_rate, "mw_p": p_rate},
           "numeric_residual": {"cliffs_delta": d_res, "mw_p": p_res}}
    for axis, key in (("domain", "domain"), ("model", "model")):
        levels = sorted(set(r[key] for r in S + R))
        tab = [[sum(1 for r in S if r[key] == lv) for lv in levels],
               [sum(1 for r in R if r[key] == lv) for lv in levels]]
        try:
            chi2, p, dof, _ = chi2_contingency(tab)
        except ValueError:
            chi2, p, dof = float("nan"), float("nan"), len(levels) - 1
        n_S, n_R = max(len(S), 1), max(len(R), 1)
        out[axis] = {"levels": levels,
                     "S_counts": dict(zip(levels, tab[0])),
                     "R_counts": dict(zip(levels, tab[1])),
                     "S_frac": {lv: tab[0][i] / n_S
                                for i, lv in enumerate(levels)},
                     "R_frac": {lv: tab[1][i] / n_R
                                for i, lv in enumerate(levels)},
                     "chi2": float(chi2), "p": float(p), "dof": int(dof)}
    if "gsm8k" in out["domain"]["levels"] and "prontoqa" in out["domain"]["levels"]:
        s_gsm = out["domain"]["S_frac"].get("gsm8k", 0.0)
        s_pro = out["domain"]["S_frac"].get("prontoqa", 0.0)
        r_gsm = out["domain"]["R_frac"].get("gsm8k", 0.0)
        r_pro = out["domain"]["R_frac"].get("prontoqa", 0.0)
        out["direction_check"] = {
            "S_gsm_minus_S_pro": s_gsm - s_pro,
            "R_pro_minus_R_gsm": r_pro - r_gsm,
            "direction_held": bool((s_gsm > s_pro) and (r_pro > r_gsm)),
        }
    return out


# -------------------------------------------- C1: per-condition split
def stratified_half(rows, rng):
    y = np.array([r["y"] for r in rows])
    idx_correct = np.where(y == 0)[0]
    idx_wrong = np.where(y == 1)[0]
    rng.shuffle(idx_correct)
    rng.shuffle(idx_wrong)
    nA_c = len(idx_correct) // 2
    nA_w = len(idx_wrong) // 2
    a = np.concatenate([idx_correct[:nA_c], idx_wrong[:nA_w]])
    b = np.concatenate([idx_correct[nA_c:], idx_wrong[nA_w:]])
    return a.tolist(), b.tolist()


def thresh_and_dir(values, y):
    """Return (threshold, flip) such that
       pred = (values > threshold).astype(int); if flip: pred = 1 - pred
    matches median_thresholded_preds exactly when fit and applied on the
    same data."""
    thr = float(np.median(values))
    flip = False
    try:
        if roc_auc_score(y, values) < 0.5:
            flip = True
    except ValueError:
        pass
    return thr, flip


def predict_with(thr, flip, values):
    pred = (values > thr).astype(int)
    if flip:
        pred = 1 - pred
    return pred


def held_out_one_split(conds, split_seed):
    """Returns (gate_dict_for_B_evaluated_on_thresholds_from_A,
                gate_dict_for_A_evaluated_on_thresholds_from_B,
                pool_sizes_dict)."""
    rng = np.random.default_rng(split_seed)
    SA, RA, SB, RB = [], [], [], []  # S/R pools for the held-out side
    # SA, RA are evaluated on A using B-fit thresholds (A is held-out for "swap")
    # SB, RB are evaluated on B using A-fit thresholds (B is held-out for "main")
    skipped = 0
    for k in sorted(conds.keys()):
        rows = conds[k]
        idx_A, idx_B = stratified_half(rows, rng)
        rows_A = [rows[i] for i in idx_A]
        rows_B = [rows[i] for i in idx_B]
        # need both classes in each half for direction-flip
        yA = np.array([r["y"] for r in rows_A])
        yB = np.array([r["y"] for r in rows_B])
        if len(set(yA.tolist())) < 2 or len(set(yB.tolist())) < 2:
            skipped += 1
            continue

        # ---- Main: fit on A, score B ----
        spA = np.array([r["spectral_entropy"] for r in rows_A])
        rnA = np.array([r["frac_negative"] for r in rows_A])
        thr_spA, flip_spA = thresh_and_dir(spA, yA)
        thr_rnA, flip_rnA = thresh_and_dir(rnA, yA)
        spB = np.array([r["spectral_entropy"] for r in rows_B])
        rnB = np.array([r["frac_negative"] for r in rows_B])
        sp_pred_B = predict_with(thr_spA, flip_spA, spB)
        rn_pred_B = predict_with(thr_rnA, flip_rnA, rnB)
        sp_right_B = sp_pred_B == yB
        rn_right_B = rn_pred_B == yB
        for r, sr, rr in zip(rows_B, sp_right_B, rn_right_B):
            if sr and not rr:
                SB.append(r)
            elif rr and not sr:
                RB.append(r)

        # ---- Swap: fit on B, score A ----
        thr_spB, flip_spB = thresh_and_dir(spB, yB)
        thr_rnB, flip_rnB = thresh_and_dir(rnB, yB)
        sp_pred_A = predict_with(thr_spB, flip_spB, spA)
        rn_pred_A = predict_with(thr_rnB, flip_rnB, rnA)
        sp_right_A = sp_pred_A == yA
        rn_right_A = rn_pred_A == yA
        for r, sr, rr in zip(rows_A, sp_right_A, rn_right_A):
            if sr and not rr:
                SA.append(r)
            elif rr and not sr:
                RA.append(r)

    SB_err = [r for r in SB if r["y"] == 1]
    RB_err = [r for r in RB if r["y"] == 1]
    SA_err = [r for r in SA if r["y"] == 1]
    RA_err = [r for r in RA if r["y"] == 1]
    gate_B = gate_numeric_domain_model(SB_err, RB_err)
    gate_A = gate_numeric_domain_model(SA_err, RA_err)
    pools = {"main_B": {"S": len(SB_err), "R": len(RB_err)},
             "swap_A": {"S": len(SA_err), "R": len(RA_err)},
             "skipped_conditions": skipped}
    return gate_B, gate_A, pools


def summarize_splits(gates, key, subkey=None):
    vals = []
    for g in gates:
        if not g.get("viable"):
            continue
        v = g[key] if subkey is None else g[key][subkey]
        if isinstance(v, dict):
            v = v.get("cliffs_delta") if "cliffs_delta" in v else v.get("p")
        vals.append(v)
    vals = np.array(vals, float)
    return {"n_splits_viable": int(len(vals)),
            "median": float(np.median(vals)) if len(vals) else float("nan"),
            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "pct_05": float(np.percentile(vals, 5)) if len(vals) else float("nan"),
            "pct_95": float(np.percentile(vals, 95)) if len(vals) else float("nan")}


def frac_sig(gates, key, subkey, alpha=0.05):
    vals = []
    for g in gates:
        if not g.get("viable"):
            continue
        v = g[key]
        if subkey:
            v = v[subkey]
        p = v["p"] if "p" in v else v["mw_p"]
        vals.append(float(p) < alpha)
    return float(np.mean(vals)) if vals else float("nan")


def frac_direction(gates):
    vals = []
    for g in gates:
        if not g.get("viable"):
            continue
        dc = g.get("direction_check")
        if dc:
            vals.append(dc["direction_held"])
    return float(np.mean(vals)) if vals else float("nan")


def frac_positive_delta(gates, key):
    vals = []
    for g in gates:
        if not g.get("viable"):
            continue
        vals.append(g[key]["cliffs_delta"] > 0)
    return float(np.mean(vals)) if vals else float("nan")


# -------------------------------------------- C2: permutation tests
def permutation_chi2(group01, col_onehot, n_perm, rng):
    """
    group01: (N,) int, 1 if S, 0 if R. Sums to nS.
    col_onehot: (N, K) 0/1.
    Returns (observed_chi2, perm_p, asymptotic_chi2, asymptotic_p)
    using the same Pearson chi^2 with continuity correction off, matching
    scipy.stats.chi2_contingency(correction=False).
    """
    N = len(group01)
    nS = int(group01.sum())
    nR = N - nS
    col_tots = col_onehot.sum(axis=0).astype(float)
    K = col_onehot.shape[1]
    if K < 2 or nS == 0 or nR == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    exp_S = nS * col_tots / N
    exp_R = nR * col_tots / N

    def chi2(s_per_col):
        r_per_col = col_tots - s_per_col
        with np.errstate(divide="ignore", invalid="ignore"):
            term_S = np.where(exp_S > 0, (s_per_col - exp_S) ** 2 / exp_S, 0.0)
            term_R = np.where(exp_R > 0, (r_per_col - exp_R) ** 2 / exp_R, 0.0)
        return float(term_S.sum() + term_R.sum())

    obs_S_per_col = (group01[:, None] * col_onehot).sum(axis=0).astype(float)
    obs_chi2 = chi2(obs_S_per_col)

    # asymptotic via scipy (no Yates correction; default in chi2_contingency
    # for tables larger than 2x2)
    obs_R_per_col = col_tots - obs_S_per_col
    tab = np.vstack([obs_S_per_col, obs_R_per_col])
    asymp_chi2, asymp_p, _, _ = chi2_contingency(tab, correction=False)

    # Permutation. Shuffle the group label vector and recompute.
    extreme = 0
    g = group01.copy()
    for _ in range(n_perm):
        rng.shuffle(g)
        s_per_col = (g[:, None] * col_onehot).sum(axis=0).astype(float)
        if chi2(s_per_col) >= obs_chi2 - 1e-12:
            extreme += 1
    perm_p = (extreme + 1) / (n_perm + 1)  # Phipson-Smyth lower bound
    return float(obs_chi2), float(perm_p), float(asymp_chi2), float(asymp_p)


def permutation_cliffs_delta(group01, values, n_perm, rng):
    """
    group01: 1 if S, 0 if R.
    Test statistic: Cliff's delta on values (S vs R), where
        delta = 2 U_S / (nS * nR) - 1
    U_S computed from rank sums (handles ties by average rank).
    Permutation: two-sided on |delta|.
    Also returns Mann-Whitney p (asymptotic, two-sided) on the observed
    split for the side-by-side display.
    """
    N = len(group01)
    nS = int(group01.sum())
    nR = N - nS
    ranks = rankdata(values, method="average")
    rank_sum_S_obs = float((ranks * group01).sum())
    U_S_obs = rank_sum_S_obs - nS * (nS + 1) / 2.0
    delta_obs = 2.0 * U_S_obs / (nS * nR) - 1.0

    s_vals = values[group01 == 1]
    r_vals = values[group01 == 0]
    try:
        _, mw_p_obs = mannwhitneyu(s_vals, r_vals, alternative="two-sided")
    except ValueError:
        mw_p_obs = float("nan")

    abs_obs = abs(delta_obs)
    extreme = 0
    g = group01.copy()
    nSnR = nS * nR
    for _ in range(n_perm):
        rng.shuffle(g)
        rsS = float((ranks * g).sum())
        U_S = rsS - nS * (nS + 1) / 2.0
        delta = 2.0 * U_S / nSnR - 1.0
        if abs(delta) >= abs_obs - 1e-15:
            extreme += 1
    perm_p = (extreme + 1) / (n_perm + 1)
    return (float(delta_obs), float(perm_p),
            float(mw_p_obs))


def run_permutations(S_err, R_err, n_perm, rng):
    rows = list(S_err) + list(R_err)
    group01 = np.array([1] * len(S_err) + [0] * len(R_err), dtype=np.int64)

    domains = sorted(set(r["domain"] for r in rows))
    dom_idx = {d: i for i, d in enumerate(domains)}
    dom_oh = np.zeros((len(rows), len(domains)), dtype=np.int64)
    for i, r in enumerate(rows):
        dom_oh[i, dom_idx[r["domain"]]] = 1

    models = sorted(set(r["model"] for r in rows))
    mod_idx = {m: i for i, m in enumerate(models)}
    mod_oh = np.zeros((len(rows), len(models)), dtype=np.int64)
    for i, r in enumerate(rows):
        mod_oh[i, mod_idx[r["model"]]] = 1

    # numeric per-sentence rate
    nsent = np.array([r["n_sentences"] for r in rows], float)
    ncount = np.array([d1.text_counts(r["cot_trace"])["numeric_tokens"]
                       for r in rows], float)
    nrate = ncount / np.maximum(nsent, 1)

    print("[c2] domain perm test ...")
    t = time.time()
    dom_obs, dom_pp, dom_asymp_chi2, dom_asymp_p = permutation_chi2(
        group01, dom_oh, n_perm, rng)
    print("    chi2_obs=%.3f  perm_p=%.2e  asymp_p=%.2e  (%.1fs)"
          % (dom_obs, dom_pp, dom_asymp_p, time.time() - t))

    print("[c2] model perm test ...")
    t = time.time()
    mod_obs, mod_pp, mod_asymp_chi2, mod_asymp_p = permutation_chi2(
        group01, mod_oh, n_perm, rng)
    print("    chi2_obs=%.3f  perm_p=%.2e  asymp_p=%.2e  (%.1fs)"
          % (mod_obs, mod_pp, mod_asymp_p, time.time() - t))

    print("[c2] numeric-rate Cliff's delta perm test ...")
    t = time.time()
    delta_obs, delta_pp, mw_p_obs = permutation_cliffs_delta(
        group01, nrate, n_perm, rng)
    print("    delta_obs=%+.4f  perm_p=%.2e  mw_p=%.2e  (%.1fs)"
          % (delta_obs, delta_pp, mw_p_obs, time.time() - t))

    return {
        "n_pool": len(rows), "n_S": len(S_err), "n_R": len(R_err),
        "n_perm": n_perm,
        "domain": {"levels": domains, "chi2_observed": dom_obs,
                   "perm_p": dom_pp,
                   "asymptotic_chi2": dom_asymp_chi2,
                   "asymptotic_p": dom_asymp_p},
        "model": {"levels": models, "chi2_observed": mod_obs,
                  "perm_p": mod_pp,
                  "asymptotic_chi2": mod_asymp_chi2,
                  "asymptotic_p": mod_asymp_p},
        "numeric_rate_delta": {"delta_observed": delta_obs,
                               "perm_p": delta_pp,
                               "mw_p_observed": mw_p_obs},
        "note_perm_p": ("perm_p = (extreme + 1) / (n_perm + 1); "
                        "lower-bounded at 1/(N+1) = %.2e" % (1.0 / (n_perm + 1))),
    }


# -------------------------------------------- markdown report writer
def fmt_p(p):
    if p != p:
        return "nan"
    if p == 0:
        return "0"
    if p < 1e-4:
        return "%.2e" % p
    return "%.4f" % p


def write_response_md(out_path, full, c1, c2):
    L = []
    A = L.append
    A("# REVIEWER_RESPONSE_NUMBERS\n")
    A("Generated by `code/exp_reviewer_holdout_permutation.py`.\n")
    A("\nCorpus: 6m_excl (3493 traces, 6 models, 15 truncations excluded).\n")
    A("Source for full-data anchor: `SIX_MODEL_RERUN_20260522_194904.json` "
      "(`configs.6m_excl.d1.gate_incorrect_only_primary`).\n")

    A("\n## Anchor check (full-data 6m_excl, this script)\n")
    A("| field | this script | SMR anchor | match |\n|---|---|---|---|\n")
    A("| n_S | %d | %d | %s |\n" % (full["n_S"], ANCHOR_NS,
                                     "YES" if full["n_S"] == ANCHOR_NS else "NO"))
    A("| n_R | %d | %d | %s |\n" % (full["n_R"], ANCHOR_NR,
                                     "YES" if full["n_R"] == ANCHOR_NR else "NO"))
    A("| numeric-rate delta | %+.4f (p=%s) | +0.2892 (p=8.97e-10) |\n"
      % (full["numeric_rate"]["cliffs_delta"],
         fmt_p(full["numeric_rate"]["mw_p"])))
    A("| numeric-residual delta | %+.4f (p=%s) | +0.3368 (p=9.65e-13) |\n"
      % (full["numeric_residual"]["cliffs_delta"],
         fmt_p(full["numeric_residual"]["mw_p"])))
    A("| domain chi^2(2) | %.3f (p=%s) | 50.069 (p=1.34e-11) |\n"
      % (full["domain"]["chi2"], fmt_p(full["domain"]["p"])))
    A("| model chi^2(5) | %.3f (p=%s) | 51.122 (p=8.17e-10) |\n"
      % (full["model"]["chi2"], fmt_p(full["model"]["p"])))

    A("\n## Comment 1: held-out confirmatory split\n")
    A("Per-condition stratified 50/50 split on each of the 18 (model,domain) "
      "conditions; spectral_entropy + frac_negative median thresholds + "
      "direction-flip fit on half A; S/R then assigned on held-out half B "
      "using those frozen rules; pooled across conditions; same tests as the "
      "full-data D1 gate. Repeated for 50 random splits (seed 0..49). "
      "A/B swap reported symmetrically.\n")

    A("\n### Headline single split (seed=0, main: fit-on-A, test-on-B)\n")
    g0 = c1["per_split"][0]["gate_main_B"]
    if not g0.get("viable"):
        A("Split unproducible at seed=0 (no S or R).\n")
    else:
        A("| field | value |\n|---|---|\n")
        A("| n_S (held-out B) | %d |\n" % g0["n_S"])
        A("| n_R (held-out B) | %d |\n" % g0["n_R"])
        A("| numeric-rate delta | %+.4f (p=%s) |\n"
          % (g0["numeric_rate"]["cliffs_delta"],
             fmt_p(g0["numeric_rate"]["mw_p"])))
        A("| numeric-residual delta | %+.4f (p=%s) |\n"
          % (g0["numeric_residual"]["cliffs_delta"],
             fmt_p(g0["numeric_residual"]["mw_p"])))
        A("| domain chi^2(2) | %.3f (p=%s) |\n"
          % (g0["domain"]["chi2"], fmt_p(g0["domain"]["p"])))
        A("| model chi^2 | %.3f (p=%s) |\n"
          % (g0["model"]["chi2"], fmt_p(g0["model"]["p"])))
        if "direction_check" in g0:
            dc = g0["direction_check"]
            A("| direction held (S>P in GSM8K, R>S in PrOntoQA) | %s |\n"
              % ("YES" if dc["direction_held"] else "NO"))
            A("| S_gsm8k - S_prontoqa | %+.3f |\n" % dc["S_gsm_minus_S_pro"])
            A("| R_prontoqa - R_gsm8k | %+.3f |\n" % dc["R_pro_minus_R_gsm"])

    A("\n### Distribution over 50 splits\n")
    A("Both the main (fit-A, test-B) and swap (fit-B, test-A) directions.\n")
    A("\n| metric | direction | median | 5th-95th pctile | "
      "fraction with p<0.05 | fraction with delta>0 / chi^2 p<0.05 |\n"
      "|---|---|---|---|---|---|\n")
    for side in ("main", "swap"):
        gates = c1["per_split_gates_" + side]
        d_rate = summarize_splits(gates, "numeric_rate", "cliffs_delta")
        d_res = summarize_splits(gates, "numeric_residual", "cliffs_delta")
        d_dom = summarize_splits(gates, "domain", "chi2")
        d_mod = summarize_splits(gates, "model", "chi2")
        A("| numeric-rate delta | %s | %+.4f | [%+.4f, %+.4f] | %.2f | %.2f |\n"
          % (side, d_rate["median"], d_rate["pct_05"], d_rate["pct_95"],
             frac_sig(gates, "numeric_rate", None),
             frac_positive_delta(gates, "numeric_rate")))
        A("| numeric-residual delta | %s | %+.4f | [%+.4f, %+.4f] | %.2f | %.2f |\n"
          % (side, d_res["median"], d_res["pct_05"], d_res["pct_95"],
             frac_sig(gates, "numeric_residual", None),
             frac_positive_delta(gates, "numeric_residual")))
        A("| domain chi^2 | %s | %.3f | [%.3f, %.3f] | %.2f | -- |\n"
          % (side, d_dom["median"], d_dom["pct_05"], d_dom["pct_95"],
             frac_sig(gates, "domain", None)))
        A("| model chi^2 | %s | %.3f | [%.3f, %.3f] | %.2f | -- |\n"
          % (side, d_mod["median"], d_mod["pct_05"], d_mod["pct_95"],
             frac_sig(gates, "model", None)))
        A("| domain direction (S>P GSM8K, R>S PrOntoQA) | %s | -- | -- | -- | "
          "%.2f held |\n" % (side, frac_direction(gates)))

    A("\n### Held-out pool sizes (across the 50 splits)\n")
    A("| direction | median n_S | median n_R |\n|---|---|---|\n")
    for side, key in (("main (test-B)", "main_B"), ("swap (test-A)", "swap_A")):
        nS = np.array([p["pools"][key]["S"] for p in c1["per_split"]], int)
        nR = np.array([p["pools"][key]["R"] for p in c1["per_split"]], int)
        A("| %s | %d | %d |\n" % (side, int(np.median(nS)), int(np.median(nR))))

    A("\n## Comment 2: permutation-based exact p-values\n")
    A("Full 6m_excl incorrect-only pool (n_S=%d, n_R=%d). %d permutations of "
      "the S/R label across the %d-row pool.\n"
      % (c2["n_S"], c2["n_R"], c2["n_perm"], c2["n_pool"]))
    A("\n| test | observed statistic | permutation p | asymptotic p |\n"
      "|---|---|---|---|\n")
    A("| domain chi^2(%d) | %.3f | %s | %s |\n"
      % (len(c2["domain"]["levels"]) - 1,
         c2["domain"]["chi2_observed"],
         fmt_p(c2["domain"]["perm_p"]),
         fmt_p(c2["domain"]["asymptotic_p"])))
    A("| model chi^2(%d) | %.3f | %s | %s |\n"
      % (len(c2["model"]["levels"]) - 1,
         c2["model"]["chi2_observed"],
         fmt_p(c2["model"]["perm_p"]),
         fmt_p(c2["model"]["asymptotic_p"])))
    A("| numeric-rate delta | %+.4f | %s | %s (MW) |\n"
      % (c2["numeric_rate_delta"]["delta_observed"],
         fmt_p(c2["numeric_rate_delta"]["perm_p"]),
         fmt_p(c2["numeric_rate_delta"]["mw_p_observed"])))
    A("\nNote: permutation p = (extreme + 1) / (n_perm + 1); lower-bounded at "
      "%.2e at this n_perm.\n" % (1.0 / (c2["n_perm"] + 1)))

    A("\n## Honest interpretation rules (locked in script)\n")
    A("- SUCCESS = held-out numeric-rate delta stays positive and the domain "
      "direction (S more GSM8K, R more PrOntoQA) is preserved across the "
      "large majority of splits. Held-out effect sizes are smaller and p's "
      "larger than full-data; that is expected and fine. The point is "
      "direction + survival on data not used to construct the groups.\n")
    A("- PARTIAL/FAIL = direction flips or effect vanishes. Reported as-is, "
      "no spin.\n")
    out_path.write_text("".join(L))


# -------------------------------------------- main
def main():
    t0 = time.time()
    recs = load_corpus_6m_excl()[0]

    conds = group_by_condition(recs)
    print("[conds] %d viable (model,domain) conditions" % len(conds))

    print("\n=== full-data anchor (Step 1+2 reconstruction) ===")
    S_full, R_full = full_data_S_R(conds)
    print("[anchor] full-data incorrect-only: n_S=%d n_R=%d  (SMR: %d / %d)"
          % (len(S_full), len(R_full), ANCHOR_NS, ANCHOR_NR))
    if len(S_full) != ANCHOR_NS or len(R_full) != ANCHOR_NR:
        sys.exit("STOP: full-data anchor mismatch -- pipeline is wrong.")

    full = gate_numeric_domain_model(S_full, R_full)
    print("[anchor] numeric-rate delta=%+.4f (p=%.2e); residual delta=%+.4f (p=%.2e)"
          % (full["numeric_rate"]["cliffs_delta"], full["numeric_rate"]["mw_p"],
             full["numeric_residual"]["cliffs_delta"], full["numeric_residual"]["mw_p"]))
    print("[anchor] domain chi2=%.3f p=%.2e; model chi2=%.3f p=%.2e"
          % (full["domain"]["chi2"], full["domain"]["p"],
             full["model"]["chi2"], full["model"]["p"]))

    print("\n=== C1: held-out splits (%d, A/B swap) ===" % N_SPLITS)
    per_split = []
    gates_main, gates_swap = [], []
    for s in range(N_SPLITS):
        gate_B, gate_A, pools = held_out_one_split(conds, split_seed=s)
        per_split.append({"seed": s,
                          "gate_main_B": gate_B,
                          "gate_swap_A": gate_A,
                          "pools": pools})
        gates_main.append(gate_B)
        gates_swap.append(gate_A)
        if s in (0, 1, 24, 49) or s == N_SPLITS - 1:
            print("  [c1] seed=%d  main n_S/n_R=%d/%d  delta_rate=%+.3f  "
                  "dom_p=%.2e  swap n_S/n_R=%d/%d"
                  % (s, gate_B.get("n_S", 0), gate_B.get("n_R", 0),
                     gate_B.get("numeric_rate", {}).get("cliffs_delta", float("nan")),
                     gate_B.get("domain", {}).get("p", float("nan")),
                     gate_A.get("n_S", 0), gate_A.get("n_R", 0)))

    c1 = {"n_splits": N_SPLITS,
          "per_split": per_split,
          "per_split_gates_main": gates_main,
          "per_split_gates_swap": gates_swap}

    print("\n=== C2: permutation tests (n_perm=%d) ===" % N_PERM)
    rng = np.random.default_rng(SEED)
    c2 = run_permutations(S_full, R_full, N_PERM, rng)

    out = {"meta": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seed": SEED, "n_splits": N_SPLITS, "n_perm": N_PERM,
                    "corpus": "6m_excl", "n_traces": len(recs),
                    "n_conditions": len(conds),
                    "anchor_match": True},
           "full_data_anchor": full,
           "c1_heldout": c1,
           "c2_permutation": c2}

    ts = time.strftime("%Y%m%d_%H%M%S")
    op = RESULTS / ("exp_reviewer_holdout_permutation_%s.json" % ts)
    json.dump(out, open(op, "w"), indent=2)
    md = ROOT / "REVIEWER_RESPONSE_NUMBERS.md"
    write_response_md(md, full, c1, c2)
    print("\n[done] %.1fs  ->  %s\n            %s"
          % (time.time() - t0, op, md))


if __name__ == "__main__":
    main()
