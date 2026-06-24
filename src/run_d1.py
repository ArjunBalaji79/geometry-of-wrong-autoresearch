"""
EXECUTE D1. Writes:
  results/10_d1_indist.json   in-distribution grouped-CV, both tasks, all conditions
  results/11_d1_lodo.json     leave-one-dataset-out, both tasks, all conditions
  results/12_d1_composition_shift.json
  results/13_d1_permutation.json
  results/14_d1_robustness_prontoqa_flag.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import d1_common as D

CONDITIONS = ["GEOM9", "SURFACE", "GEOM9_resid_SURF", "GEOM9+SURF"]
BENCHES = ["gsm8k", "folio", "prontoqa"]


# ---------------------------------------------------------------------------
def oof_binary(X9, Xs, y, grp, condition, kind="logistic", k=5):
    """Out-of-fold P(correct) via grouped CV."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X9, y, grp):
        p, classes = D.fit_predict(condition, X9[tr], Xs[tr], y[tr],
                                   X9[te], Xs[te], kind=kind, proba=True)
        pos = list(classes).index(1)
        oof[te] = p[:, pos]
    return oof


def oof_mode(X9, Xs, y, grp, condition, kind="logistic", k=5):
    oof = np.full(len(y), -1)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X9, y, grp):
        pred, _ = D.fit_predict(condition, X9[tr], Xs[tr], y[tr],
                                X9[te], Xs[te], kind=kind, proba=False)
        oof[te] = pred
    return oof


def run_indist(rows, kind="logistic"):
    X9, Xs, bench, grp = D.arrays(rows)
    y_corr = np.array([1 if r["primary_correct"] else 0 for r in rows])
    out = {"kind": kind, "binary": {}, "mode": {}}

    # binary
    for c in CONDITIONS:
        oof = oof_binary(X9, Xs, y_corr, grp, c, kind)
        m = D.bootstrap_metric(D.binary_auc, y_corr, oof, grp)
        out["binary"][c] = {"auc": D.binary_auc(y_corr, oof), **m}

    # mode (incorrect only)
    inc = np.array([i for i, r in enumerate(rows) if not r["primary_correct"]])
    Xm9, Xms, gm = X9[inc], Xs[inc], grp[inc]
    ym = np.array([rows[i]["mode"] for i in inc])
    for c in CONDITIONS:
        oof = oof_mode(Xm9, Xms, ym, gm, c, kind)
        f1 = D.mode_macro_f1(ym, oof)
        ba = D.mode_bal_acc(ym, oof)
        mf = D.bootstrap_metric(D.mode_macro_f1, ym, oof, gm)
        mb = D.bootstrap_metric(D.mode_bal_acc, ym, oof, gm)
        out["mode"][c] = {"macro_f1": f1, "macro_f1_ci": [mf["ci_lo"], mf["ci_hi"]],
                          "bal_acc": ba, "bal_acc_ci": [mb["ci_lo"], mb["ci_hi"]]}
    out["mode_n_incorrect"] = int(len(inc))
    out["mode_class_counts"] = {int(k2): int((ym == k2).sum()) for k2 in (1, 2, 3)}
    # paired delta resid-geom vs surface (in-dist)
    oof_rg = oof_mode(Xm9, Xms, ym, gm, "GEOM9_resid_SURF", kind)
    oof_sf = oof_mode(Xm9, Xms, ym, gm, "SURFACE", kind)
    out["mode_delta_residgeom_minus_surface"] = D.paired_bootstrap_delta(
        D.mode_macro_f1, ym, oof_rg, oof_sf, gm)
    return out


def run_lodo(rows, kind="logistic"):
    X9, Xs, bench, grp = D.arrays(rows)
    y_corr = np.array([1 if r["primary_correct"] else 0 for r in rows])
    modes = np.array([r["mode"] if r["mode"] is not None else -1 for r in rows])
    out = {"kind": kind, "folds": {}}
    for held in BENCHES:
        te = bench == held
        tr = ~te
        fold = {"held_out": held, "n_test": int(te.sum()), "binary": {}, "mode": {}}
        # binary
        for c in CONDITIONS:
            p, classes = D.fit_predict(c, X9[tr], Xs[tr], y_corr[tr],
                                       X9[te], Xs[te], kind=kind, proba=True)
            pos = list(classes).index(1)
            proba = p[:, pos]
            mm = D.bootstrap_metric(D.binary_auc, y_corr[te], proba, grp[te])
            fold["binary"][c] = {"auc": D.binary_auc(y_corr[te], proba),
                                 "ci": [mm["ci_lo"], mm["ci_hi"]]}
        # mode: train incorrect of train benches, test incorrect of held
        tr_inc = tr & (modes > 0)
        te_inc = te & (modes > 0)
        ym_tr, ym_te = modes[tr_inc], modes[te_inc]
        gm_te = grp[te_inc]
        preds = {}
        for c in CONDITIONS:
            pred, _ = D.fit_predict(c, X9[tr_inc], Xs[tr_inc], ym_tr,
                                    X9[te_inc], Xs[te_inc], kind=kind, proba=False)
            preds[c] = pred
            f1 = D.mode_macro_f1(ym_te, pred)
            ba = D.mode_bal_acc(ym_te, pred)
            mf = D.bootstrap_metric(D.mode_macro_f1, ym_te, pred, gm_te)
            fold["mode"][c] = {"macro_f1": f1, "macro_f1_ci": [mf["ci_lo"], mf["ci_hi"]],
                               "bal_acc": ba}
        # headline paired deltas on the held-out test set
        fold["mode_delta_residgeom_minus_surface"] = D.paired_bootstrap_delta(
            D.mode_macro_f1, ym_te, preds["GEOM9_resid_SURF"], preds["SURFACE"], gm_te)
        fold["mode_delta_geom9_minus_surface"] = D.paired_bootstrap_delta(
            D.mode_macro_f1, ym_te, preds["GEOM9"], preds["SURFACE"], gm_te)
        # chance macro-f1 for reference: majority + stratified
        fold["mode_test_class_counts"] = {int(k2): int((ym_te == k2).sum()) for k2 in (1, 2, 3)}
        fold["n_test_incorrect"] = int(te_inc.sum())
        out["folds"][held] = fold
    return out


def run_permutation(rows, n_perm=200, kind="logistic"):
    """Label-permutation null for in-distribution mode macro-F1 (GEOM9_resid_SURF)."""
    X9, Xs, bench, grp = D.arrays(rows)
    inc = np.array([i for i, r in enumerate(rows) if not r["primary_correct"]])
    Xm9, Xms, gm = X9[inc], Xs[inc], grp[inc]
    ym = np.array([rows[i]["mode"] for i in inc])
    real = D.mode_macro_f1(ym, oof_mode(Xm9, Xms, ym, gm, "GEOM9_resid_SURF", kind))
    rng = np.random.default_rng(D.SEED)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(ym)
        null.append(D.mode_macro_f1(yp, oof_mode(Xm9, Xms, yp, gm, "GEOM9_resid_SURF", kind)))
    null = np.array(null)
    # also binary permutation
    y_corr = np.array([1 if r["primary_correct"] else 0 for r in rows])
    real_b = D.binary_auc(y_corr, oof_binary(X9, Xs, y_corr, grp, "GEOM9", kind))
    nb = []
    for _ in range(n_perm):
        yp = rng.permutation(y_corr)
        nb.append(D.binary_auc(yp, oof_binary(X9, Xs, yp, grp, "GEOM9", kind)))
    nb = np.array(nb)
    return {
        "mode_residgeom_macro_f1_real": real,
        "mode_null_mean": float(null.mean()), "mode_null_p95": float(np.percentile(null, 95)),
        "mode_p_value": float((np.sum(null >= real) + 1) / (n_perm + 1)),
        "binary_geom9_auc_real": real_b,
        "binary_null_mean": float(nb.mean()), "binary_null_p95": float(np.percentile(nb, 95)),
        "binary_p_value": float((np.sum(nb >= real_b) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def run_composition_shift(rows, lodo):
    """Relate binary OOD drop to train/test mode-mixture JS divergence."""
    def js(p, q):
        p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        def kl(a, b): return float(np.sum(a * np.log(a / b)))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    modes_by_bench = {b: np.array([0, 0, 0]) for b in BENCHES}
    for r in rows:
        if r["mode"] in (1, 2, 3):
            modes_by_bench[r["benchmark"]][r["mode"] - 1] += 1

    # in-dist binary AUC (GEOM9) as reference
    indist = run_indist(rows)["binary"]["GEOM9"]["auc"]
    pts = []
    for held in BENCHES:
        train_mix = sum((modes_by_bench[b] for b in BENCHES if b != held), np.array([0, 0, 0]))
        test_mix = modes_by_bench[held]
        drop = indist - lodo["folds"][held]["binary"]["GEOM9"]["auc"]
        pts.append({"held_out": held, "binary_ood_drop": drop,
                    "mode_mix_JS_div": js(train_mix, test_mix),
                    "train_mix": train_mix.tolist(), "test_mix": test_mix.tolist()})
    xs = np.array([p["mode_mix_JS_div"] for p in pts])
    ys = np.array([p["binary_ood_drop"] for p in pts])
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(pts) > 1 else None
    return {"indist_binary_auc_geom9": indist, "points": pts,
            "pearson_r_n3": corr,
            "note": "n=3 LODO folds; descriptive only."}


def run_robustness_prontoqa_flag(rows, kind="logistic"):
    """Re-run LODO mode/binary using the RELEASED flag for PrOntoQA (corrupted),
    to confirm the headline is not an artifact of the engine relabeling."""
    import copy
    from labels import failure_mode
    rows2 = []
    feats = {}
    for l in open(ROOT / "results/features.jsonl"):
        r = json.loads(l); feats[(r["benchmark"], r["model"], r["problem_idx"])] = r
    for r in rows:
        r2 = dict(r)
        if r["benchmark"] == "prontoqa":
            r2["primary_correct"] = bool(r["released_correct"])
            if not r2["primary_correct"]:
                fr = feats[(r["benchmark"], r["model"], r["problem_idx"])]
                r2["mode"] = failure_mode(fr)
            else:
                r2["mode"] = None
        rows2.append(r2)
    return run_lodo(rows2, kind)


def main():
    rows = D.load()
    print("loaded", len(rows), "traces")
    indist = run_indist(rows)
    json.dump(indist, open(ROOT / "results/10_d1_indist.json", "w"), indent=2)
    print("in-dist done")
    lodo = run_lodo(rows)
    json.dump(lodo, open(ROOT / "results/11_d1_lodo.json", "w"), indent=2)
    print("lodo done")
    comp = run_composition_shift(rows, lodo)
    json.dump(comp, open(ROOT / "results/12_d1_composition_shift.json", "w"), indent=2)
    print("composition done")
    perm = run_permutation(rows, n_perm=200)
    json.dump(perm, open(ROOT / "results/13_d1_permutation.json", "w"), indent=2)
    print("permutation done")
    rob = run_robustness_prontoqa_flag(rows)
    json.dump(rob, open(ROOT / "results/14_d1_robustness_prontoqa_flag.json", "w"), indent=2)
    print("robustness done")
    # secondary: GBT in-dist + lodo for robustness
    gbt_indist = run_indist(rows, kind="hgb")
    gbt_lodo = run_lodo(rows, kind="hgb")
    json.dump({"indist": gbt_indist, "lodo": gbt_lodo},
              open(ROOT / "results/15_d1_gbt.json", "w"), indent=2)
    print("gbt done")


if __name__ == "__main__":
    main()
