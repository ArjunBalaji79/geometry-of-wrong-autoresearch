"""
exp_d2_fingerprint_gate.py -- Direction 2 quick gate.

Does model identity predict from the 9 geometric features BEYOND a trace-length
baseline? Decisive gate before any Direction-2 build.

Task: 4-class model identity on the 2308 in-corpus traces (Claude Sonnet 4,
gpt-oss 120B, Llama 3.1 8B, Mistral 7B). Gemini held out. Chance ~0.25.

Three feature sets, each under GBT (primary) + RF + LR:
  (A) length-only:  [n_sentences, n_edges]      -- the assassin
  (B) geometry-9:   5 spectral + 4 Ricci
  (C) geometry-9 + length: all 11

Plus, for the shape-vs-size decomposition: B restricted to size-correlated
features and to shape (length-independent) features, split empirically by each
feature's |correlation| with n_edges.

CV: StratifiedGroupKFold(5), group = (domain, problem_idx) to block
problem-content leakage, stratified on model. seed=0. Classifier
hyperparameters identical to stage1_lock_floor.py. Bootstrap CIs resample
groups (1000x). Existing data only.

BINDING RULE: Direction 2 is ALIVE only if (B) and (C) each meaningfully
exceed (A) -- accuracy gap with non/barely-overlapping bootstrap CIs. If
(B) <= (A), the fingerprint is length: Direction 2 is DEAD.
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_SPLITS = 5
N_BOOT = 1000
RESULTS = Path("results")
SPEC = ["spectral_entropy", "fiedler_value", "grs", "shs", "hfer"]
RICCI = ["mean_kappa", "std_kappa", "frac_negative", "min_kappa"]
GEOM9 = SPEC + RICCI


def latest(pat):
    h = sorted(glob.glob(str(RESULTS / pat)))
    if not h:
        sys.exit("missing input: " + pat)
    return h[-1]


def mk_gbt():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.05, random_state=0)


def mk_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=5, random_state=0, n_jobs=-1)


def mk_lr():
    return Pipeline([("s", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, C=1.0))])


FACTORIES = {"gbt": mk_gbt, "rf": mk_rf, "lr": mk_lr}


def cv_oof_pred(X, y, groups, factory):
    """StratifiedGroupKFold OOF predicted class labels."""
    oof = np.full(len(y), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in sgkf.split(X, y, groups):
        clf = factory()
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof


def metrics_with_ci(y, oof, groups, rng):
    """Accuracy + macro-F1 on pooled OOF; group-resampled bootstrap CIs."""
    acc = float(accuracy_score(y, oof))
    mf1 = float(f1_score(y, oof, average="macro"))
    uniq = np.array(sorted(set(groups)))
    g2idx = {g: np.where(groups == g)[0] for g in uniq}
    accs, f1s = [], []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2idx[g] for g in samp])
        accs.append(accuracy_score(y[idx], oof[idx]))
        f1s.append(f1_score(y[idx], oof[idx], average="macro"))
    return {
        "accuracy": acc, "macro_f1": mf1,
        "accuracy_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def paired_delta_acc(y, oof_b, oof_a, groups, rng):
    """accuracy(B) - accuracy(A), paired group bootstrap."""
    d = float(accuracy_score(y, oof_b) - accuracy_score(y, oof_a))
    uniq = np.array(sorted(set(groups)))
    g2idx = {g: np.where(groups == g)[0] for g in uniq}
    boots = []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2idx[g] for g in samp])
        boots.append(accuracy_score(y[idx], oof_b[idx]) - accuracy_score(y[idx], oof_a[idx]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"delta_accuracy": d, "ci": [lo, hi], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    recs = [json.loads(l) for l in open(latest("rerun_phase1_features_*.jsonl"))]
    print("[d2] %d in-corpus traces" % len(recs))

    models = sorted(set(r["model"] for r in recs))
    m2i = {m: i for i, m in enumerate(models)}
    y = np.array([m2i[r["model"]] for r in recs])
    groups = np.array(["%s|%s" % (r["domain"], r["trace_id"].rsplit("__", 1)[-1])
                       for r in recs])
    feat = {f: np.array([(r["spectral"].get(f) if f in SPEC else
                          r["ricci"].get(f) if f in RICCI else r[f])
                         for r in recs], float)
            for f in GEOM9 + ["n_sentences", "n_edges"]}

    from collections import Counter
    cc = Counter(y.tolist())
    print("[d2] classes:", {models[i]: cc[i] for i in range(len(models))})
    print("[d2] majority-class baseline accuracy: %.3f" % (max(cc.values()) / len(y)))

    def X_of(keys):
        return np.column_stack([feat[k] for k in keys])

    # size-correlation of each geometric feature with n_edges -> shape/size split
    size_corr = {f: float(abs(np.corrcoef(feat[f], feat["n_edges"])[0, 1])) for f in GEOM9}
    size_feats = [f for f in GEOM9 if size_corr[f] >= 0.5]
    shape_feats = [f for f in GEOM9 if size_corr[f] < 0.5]

    featuresets = {
        "A_length_only": ["n_sentences", "n_edges"],
        "B_geometry9": GEOM9,
        "C_geometry9_plus_length": GEOM9 + ["n_sentences", "n_edges"],
        "B_shape_subset": shape_feats,
        "B_size_subset": size_feats,
    }

    out = {"meta": {"n_traces": len(recs), "n_classes": len(models),
                    "models": models, "chance": 1.0 / len(models),
                    "majority_baseline": max(cc.values()) / len(y),
                    "seed": SEED, "cv": "StratifiedGroupKFold(5), group=(domain,problem)",
                    "size_correlation_with_n_edges": size_corr,
                    "shape_subset": shape_feats, "size_subset": size_feats},
           "results": {}, "oof": {}}

    for fs, keys in featuresets.items():
        X = X_of(keys)
        out["results"][fs] = {}
        for algo in ("gbt", "rf", "lr"):
            oof = cv_oof_pred(X, y, groups, FACTORIES[algo])
            mt = metrics_with_ci(y, oof, groups, rng)
            # per-class precision/recall/f1
            pr, rc, f1c, sup = precision_recall_fscore_support(y, oof, zero_division=0)
            mt["per_class"] = {models[i]: {"precision": float(pr[i]),
                                           "recall": float(rc[i]),
                                           "f1": float(f1c[i]), "support": int(sup[i])}
                               for i in range(len(models))}
            out["results"][fs][algo] = mt
            if algo == "gbt":
                out["oof"][fs] = oof.tolist()
            print("  %-26s %-3s acc=%.3f %s  macroF1=%.3f" % (
                fs, algo, mt["accuracy"], mt["accuracy_ci"], mt["macro_f1"]))

    # ---- verdict: GBT primary, B vs A and C vs A ----
    oA = np.array(out["oof"]["A_length_only"])
    oB = np.array(out["oof"]["B_geometry9"])
    oC = np.array(out["oof"]["C_geometry9_plus_length"])
    oBsh = np.array(out["oof"]["B_shape_subset"])
    dBA = paired_delta_acc(y, oB, oA, groups, rng)
    dCA = paired_delta_acc(y, oC, oA, groups, rng)
    dShA = paired_delta_acc(y, oBsh, oA, groups, rng)
    accA = out["results"]["A_length_only"]["gbt"]["accuracy"]
    accB = out["results"]["B_geometry9"]["gbt"]["accuracy"]
    accC = out["results"]["C_geometry9_plus_length"]["gbt"]["accuracy"]
    alive = bool(dBA["ci_excludes_zero"] and dBA["delta_accuracy"] > 0
                 and dCA["ci_excludes_zero"] and dCA["delta_accuracy"] > 0)
    out["verdict"] = {
        "B_minus_A": dBA, "C_minus_A": dCA, "B_shape_minus_A": dShA,
        "direction2": "ALIVE" if alive else "DEAD",
        "rule": "ALIVE iff B>A and C>A with delta-CI excluding zero (GBT).",
    }

    # ---- interpretability (always computed; meaningful only if B beats A) ----
    Xb = X_of(GEOM9)
    gbt = mk_gbt().fit(Xb, y)
    imp = {GEOM9[i]: float(gbt.feature_importances_[i]) for i in range(len(GEOM9))}
    # per-model standardized feature profile
    Z = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-12)
    profile = {models[c]: {GEOM9[i]: float(Z[y == c, i].mean()) for i in range(len(GEOM9))}
               for c in range(len(models))}
    out["interpretability"] = {"gbt_feature_importance_B": imp,
                               "per_model_standardized_profile": profile}

    op = RESULTS / ("exp_d2_fingerprint_gate_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(out, open(op, "w"), indent=2)
    print()
    print("VERDICT: Direction 2 is %s" % out["verdict"]["direction2"])
    print("  A length-only GBT acc = %.3f %s" % (accA, out["results"]["A_length_only"]["gbt"]["accuracy_ci"]))
    print("  B geometry-9  GBT acc = %.3f %s" % (accB, out["results"]["B_geometry9"]["gbt"]["accuracy_ci"]))
    print("  C geom9+len   GBT acc = %.3f %s" % (accC, out["results"]["C_geometry9_plus_length"]["gbt"]["accuracy_ci"]))
    print("  B-A delta = %+.3f CI%s (excl 0: %s)" % (
        dBA["delta_accuracy"], dBA["ci"], dBA["ci_excludes_zero"]))
    print("  C-A delta = %+.3f CI%s (excl 0: %s)" % (
        dCA["delta_accuracy"], dCA["ci"], dCA["ci_excludes_zero"]))
    print("  B-shape-only vs A delta = %+.3f CI%s" % (dShA["delta_accuracy"], dShA["ci"]))
    print("[d2] done (%.0fs) -> %s" % (time.time() - t0, op))
    print("D2_OK %s" % op)


if __name__ == "__main__":
    main()
