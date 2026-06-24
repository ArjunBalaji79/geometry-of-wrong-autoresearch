"""
D1 shared machinery: data loading, surface residualization, classifiers,
grouped CV, and group-aware paired bootstrap.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import SIG_KEYS, SURFACE_KEYS

SEED = 0


def load():
    """Join features.jsonl + labels.jsonl on (benchmark,model,problem_idx)."""
    feats = {}
    for l in open(ROOT / "results/features.jsonl"):
        r = json.loads(l)
        feats[(r["benchmark"], r["model"], r["problem_idx"])] = r
    rows = []
    for l in open(ROOT / "results/labels.jsonl"):
        lab = json.loads(l)
        k = (lab["benchmark"], lab["model"], lab["problem_idx"])
        if k not in feats:
            continue
        f = feats[k]
        rows.append({
            "benchmark": lab["benchmark"], "model": lab["model"],
            "problem_idx": lab["problem_idx"],
            "primary_correct": lab["primary_correct"],
            "released_correct": lab["released_correct"],
            "mode": lab["mode"],
            "X9": np.array([f["signature"][k2] for k2 in SIG_KEYS], dtype=float),
            "Xs": np.array([f["surface"][k2] for k2 in SURFACE_KEYS], dtype=float),
        })
    return rows


def arrays(rows):
    X9 = np.vstack([r["X9"] for r in rows])
    Xs = np.vstack([r["Xs"] for r in rows])
    bench = np.array([r["benchmark"] for r in rows])
    grp = np.array([f"{r['benchmark']}::{r['problem_idx']}" for r in rows])
    return X9, Xs, bench, grp


def residualize_fit(X9_tr, Xs_tr):
    """Fit OLS(geom_col ~ surface) on TRAIN; return a transform closure."""
    reg = LinearRegression().fit(Xs_tr, X9_tr)   # multi-output
    def transform(X9, Xs):
        return X9 - reg.predict(Xs)
    return transform


def make_clf(kind="logistic", multiclass=False):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    return HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                          learning_rate=0.1, max_iter=200)


def _design(condition, X9_tr, Xs_tr, X9_te, Xs_te):
    """Return (Xtr, Xte) for a feature condition, residualizing on train only."""
    if condition == "GEOM9":
        return X9_tr, X9_te
    if condition == "SURFACE":
        return Xs_tr, Xs_te
    if condition == "GEOM9_resid_SURF":
        tf = residualize_fit(X9_tr, Xs_tr)
        return tf(X9_tr, Xs_tr), tf(X9_te, Xs_te)
    if condition == "GEOM9+SURF":
        return np.hstack([X9_tr, Xs_tr]), np.hstack([X9_te, Xs_te])
    raise ValueError(condition)


def fit_predict(condition, X9_tr, Xs_tr, y_tr, X9_te, Xs_te,
                kind="logistic", proba=True):
    Xtr, Xte = _design(condition, X9_tr, Xs_tr, X9_te, Xs_te)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = make_clf(kind, multiclass=(len(np.unique(y_tr)) > 2))
    clf.fit(Xtr, y_tr)
    if proba:
        p = clf.predict_proba(Xte)
        classes = clf.classes_
        return p, classes
    return clf.predict(Xte), None


# --- metrics ----------------------------------------------------------------
def binary_auc(y_true, proba_pos):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, proba_pos)


def mode_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[1, 2, 3])


def mode_bal_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


# --- group-aware bootstrap --------------------------------------------------
def group_bootstrap_indices(groups, n_boot, rng):
    """Yield resampled sample-index arrays by resampling unique groups w/ replacement."""
    uniq = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gi[g] for g in chosen])
        yield idx


def paired_bootstrap_delta(metric_fn, y_true, predA, predB, groups,
                           n_boot=2000, seed=SEED):
    """95% CI of metric(A)-metric(B), resampling groups. predX are arrays aligned to y_true."""
    rng = np.random.default_rng(seed)
    deltas = []
    for idx in group_bootstrap_indices(groups, n_boot, rng):
        try:
            a = metric_fn(y_true[idx], predA[idx])
            b = metric_fn(y_true[idx], predB[idx])
        except Exception:
            continue
        if np.isnan(a) or np.isnan(b):
            continue
        deltas.append(a - b)
    deltas = np.array(deltas)
    return {"delta_mean": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(deltas, 2.5)),
            "ci_hi": float(np.percentile(deltas, 97.5)),
            "frac_gt0": float(np.mean(deltas > 0)), "n_boot": len(deltas)}


def bootstrap_metric(metric_fn, y_true, pred, groups, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for idx in group_bootstrap_indices(groups, n_boot, rng):
        try:
            v = metric_fn(y_true[idx], pred[idx])
        except Exception:
            continue
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    return {"mean": float(np.mean(vals)),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)), "n_boot": len(vals)}
