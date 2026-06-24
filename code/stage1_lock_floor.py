"""
Helper module used as a library by code/six_model_rerun.py. Provides
`orthogonality_per_condition` and the median-thresholded-predictor
helpers that the six-model rerun re-uses without modification.

Not intended to be invoked as __main__ in this release. The standalone
entry point assumes a per-trace JSON layout that this release does not
ship; the dependent functions called by six_model_rerun.py read their
inputs from results/rerun_phase1_features_*.jsonl instead.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
SPECTRAL_FEATURES = ["spectral_entropy", "fiedler_value", "grs", "shs", "hfer"]
RICCI_FEATURES = ["mean_kappa", "std_kappa", "frac_negative", "min_kappa"]
ALL_FEATURES = [("spectral", f) for f in SPECTRAL_FEATURES] \
             + [("ricci", f) for f in RICCI_FEATURES]


def median_thresholded_preds(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Binary prediction at the median, flipped if it anti-correlates with y."""
    pred = (values > np.median(values)).astype(int)
    try:
        auc = roc_auc_score(y, values)
        if auc < 0.5:
            pred = 1 - pred
    except ValueError:
        pass
    return pred


def orthogonality_per_condition(items):
    """
    For each (model, domain) condition, compute:
      - n_total, n_both_right, n_only_sp, n_only_ri, n_both_wrong
      - disagreement_rate = (n_only_sp + n_only_ri) / n_total
    Returns a list of dicts, one per condition.
    """
    pairs = {}
    for r in items:
        key = (r["model"], r["domain"])
        pairs.setdefault(key, []).append(r)

    rows = []
    for (model, domain), cond_items in sorted(pairs.items()):
        if len(cond_items) < 20:
            continue
        y = np.array([0 if it["correct"] else 1 for it in cond_items])
        if len(np.unique(y)) < 2:
            continue
        sp = np.array([it["spectral"]["spectral_entropy"] for it in cond_items])
        rn = np.array([it["ricci"]["frac_negative"] for it in cond_items])

        sp_pred = median_thresholded_preds(sp, y)
        rn_pred = median_thresholded_preds(rn, y)

        sp_right = (sp_pred == y)
        rn_right = (rn_pred == y)

        both_right = int(np.sum(sp_right & rn_right))
        only_sp = int(np.sum(sp_right & ~rn_right))
        only_ri = int(np.sum(~sp_right & rn_right))
        both_wrong = int(np.sum(~sp_right & ~rn_right))
        n = int(len(y))
        disagreement = (only_sp + only_ri) / n

        rows.append({
            "model": model,
            "domain": domain,
            "n": n,
            "both_right": both_right,
            "only_sp": only_sp,
            "only_ri": only_ri,
            "both_wrong": both_wrong,
            "disagreement_rate": float(disagreement),
        })
    return rows


def plot_orthogonality_heatmap(rows, save_path):
    """
    Two-panel heatmap: left shows disagreement rate per condition,
    right shows the breakdown (only_sp vs only_ri) as a diverging color.
    """
    models = sorted(set(r["model"] for r in rows))
    domains = sorted(set(r["domain"] for r in rows))

    disagreement_matrix = np.full((len(models), len(domains)), np.nan)
    asymmetry_matrix = np.full((len(models), len(domains)), np.nan)
    label_matrix = np.empty((len(models), len(domains)), dtype=object)

    for r in rows:
        i = models.index(r["model"])
        j = domains.index(r["domain"])
        disagreement_matrix[i, j] = r["disagreement_rate"]
        denom = max(r["only_sp"] + r["only_ri"], 1)
        asymmetry_matrix[i, j] = (r["only_ri"] - r["only_sp"]) / denom
        label_matrix[i, j] = f"{r['only_sp']} / {r['only_ri']}"

    display_models = []
    for m in models:
        if "claude-sonnet-4" in m:
            display_models.append("Claude Sonnet 4")
        elif "gpt-oss" in m:
            display_models.append("gpt-oss 120B")
        elif "llama" in m.lower():
            display_models.append("Llama 3.1 8B")
        elif "mistral" in m.lower():
            display_models.append("Mistral 7B")
        else:
            display_models.append(m.split("/")[-1][:20])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im = ax.imshow(disagreement_matrix, cmap="YlOrRd", vmin=0.3, vmax=0.6, aspect="auto")
    for i in range(len(models)):
        for j in range(len(domains)):
            v = disagreement_matrix[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.48 else "black"
                ax.text(j, i, f"{v*100:.0f}%",
                        ha="center", va="center", color=color, fontsize=11,
                        fontweight="semibold")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(display_models)
    ax.set_title("Spectral–Ricci disagreement rate\n(fraction of traces where methods disagree)",
                  fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="disagreement fraction")

    ax = axes[1]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im2 = ax.imshow(asymmetry_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(len(models)):
        for j in range(len(domains)):
            if label_matrix[i, j] is not None:
                ax.text(j, i, label_matrix[i, j],
                        ha="center", va="center", color="black", fontsize=10)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.upper() for d in domains])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(display_models)
    ax.set_title("Disagreement asymmetry\n(only-spectral-right / only-Ricci-right count)",
                  fontsize=11)
    cb = plt.colorbar(im2, ax=ax, fraction=0.045, pad=0.04,
                       label="← spectral catches more  |  Ricci catches more →")

    fig.suptitle("Spectral and Ricci features detect orthogonal failure modes",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path.with_suffix('.png')}")
    print(f"  saved: {save_path.with_suffix('.pdf')}")


def single_feature_cv_auc(values, y, n_splits=5, seed=0):
    """5-fold CV AUC for a single feature. Direction chosen per-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(values.reshape(-1, 1), y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        auc = roc_auc_score(y[te], values[te])
        auc = max(auc, 1 - auc)
        aucs.append(auc)
    return aucs


def classifier_cv_auc(factory, X, y, n_splits=5, seed=0):
    """5-fold CV AUC for a classifier built by factory()."""
    if len(np.unique(y)) < 2 or X.shape[0] < n_splits * 2:
        return []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = factory()
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return aucs


def bootstrap_ci(fold_aucs, n_boot=1000, seed=0):
    """Fold-level bootstrap 95% CI."""
    if len(fold_aucs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(fold_aucs, size=len(fold_aucs), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def fusion_auc_analysis(items):
    """
    For each (model, domain) condition, compute CV AUC for:
      - best single spectral feature
      - best single Ricci feature
      - oracle = max of the two
      - LR on all 9 features
      - RF on all 9 features
      - GBT on all 9 features
    Returns per-condition dict with AUC + CI for each method.
    """
    pairs = {}
    for r in items:
        key = (r["model"], r["domain"])
        pairs.setdefault(key, []).append(r)

    lr_fac = lambda: Pipeline([("s", StandardScaler()),
                                ("lr", LogisticRegression(max_iter=1000, C=1.0))])
    rf_fac = lambda: RandomForestClassifier(n_estimators=200, max_depth=6,
                                             min_samples_leaf=5, random_state=0,
                                             n_jobs=-1)
    gbt_fac = lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                  learning_rate=0.05, random_state=0)

    rows = []
    for (model, domain), cond_items in sorted(pairs.items()):
        if len(cond_items) < 20:
            continue
        y = np.array([0 if it["correct"] else 1 for it in cond_items])
        if len(np.unique(y)) < 2:
            continue

        sp_best_aucs, sp_best_name = [], None
        for f in SPECTRAL_FEATURES:
            vals = np.array([it["spectral"].get(f) for it in cond_items], dtype=float)
            if np.any(np.isnan(vals)): continue
            aucs = single_feature_cv_auc(vals, y)
            if aucs and np.mean(aucs) > (np.mean(sp_best_aucs) if sp_best_aucs else 0):
                sp_best_aucs = aucs
                sp_best_name = f

        ri_best_aucs, ri_best_name = [], None
        for f in RICCI_FEATURES:
            vals = np.array([it["ricci"].get(f) for it in cond_items], dtype=float)
            if np.any(np.isnan(vals)): continue
            aucs = single_feature_cv_auc(vals, y)
            if aucs and np.mean(aucs) > (np.mean(ri_best_aucs) if ri_best_aucs else 0):
                ri_best_aucs = aucs
                ri_best_name = f

        X = []
        for it in cond_items:
            row = [it["spectral"].get(fn) for fn in SPECTRAL_FEATURES] + \
                  [it["ricci"].get(fn) for fn in RICCI_FEATURES]
            if any(v is None for v in row):
                continue
            X.append(row)
        X = np.array(X)
        if X.shape[0] < len(cond_items):
            y = y[:X.shape[0]]

        lr_aucs = classifier_cv_auc(lr_fac, X, y)
        rf_aucs = classifier_cv_auc(rf_fac, X, y)
        gbt_aucs = classifier_cv_auc(gbt_fac, X, y)

        def summarize(aucs):
            if not aucs:
                return {"mean": float("nan"), "ci_low": float("nan"),
                        "ci_high": float("nan"), "n_folds": 0}
            lo, hi = bootstrap_ci(aucs)
            return {"mean": float(np.mean(aucs)), "ci_low": lo, "ci_high": hi,
                    "n_folds": len(aucs)}

        sp_best_mean = np.mean(sp_best_aucs) if sp_best_aucs else 0
        ri_best_mean = np.mean(ri_best_aucs) if ri_best_aucs else 0
        oracle_aucs = sp_best_aucs if sp_best_mean >= ri_best_mean else ri_best_aucs

        rows.append({
            "model": model,
            "domain": domain,
            "n": int(X.shape[0]),
            "sp_best": {"name": sp_best_name, **summarize(sp_best_aucs)},
            "ri_best": {"name": ri_best_name, **summarize(ri_best_aucs)},
            "oracle_single": summarize(oracle_aucs),
            "lr": summarize(lr_aucs),
            "rf": summarize(rf_aucs),
            "gbt": summarize(gbt_aucs),
        })
    return rows


def plot_fusion_auc(rows, save_path):
    """Bar chart: per condition, best single vs LR vs RF vs GBT."""
    display = []
    for r in rows:
        m = r["model"]
        if "claude-sonnet-4" in m:
            m_short = "Claude Sonnet 4"
        elif "gpt-oss" in m:
            m_short = "gpt-oss 120B"
        elif "llama" in m.lower():
            m_short = "Llama 3.1 8B"
        elif "mistral" in m.lower():
            m_short = "Mistral 7B"
        else:
            m_short = m.split("/")[-1][:20]
        display.append(f"{m_short}\n{r['domain'].upper()}")

    x = np.arange(len(rows))
    width = 0.2

    best_single_means = [r["oracle_single"]["mean"] for r in rows]
    best_single_cilo = [r["oracle_single"]["ci_low"] for r in rows]
    best_single_cihi = [r["oracle_single"]["ci_high"] for r in rows]
    lr_means = [r["lr"]["mean"] for r in rows]
    lr_cilo = [r["lr"]["ci_low"] for r in rows]
    lr_cihi = [r["lr"]["ci_high"] for r in rows]
    rf_means = [r["rf"]["mean"] for r in rows]
    rf_cilo = [r["rf"]["ci_low"] for r in rows]
    rf_cihi = [r["rf"]["ci_high"] for r in rows]
    gbt_means = [r["gbt"]["mean"] for r in rows]
    gbt_cilo = [r["gbt"]["ci_low"] for r in rows]
    gbt_cihi = [r["gbt"]["ci_high"] for r in rows]

    def yerr(means, los, his):
        return np.array([[m - l for m, l in zip(means, los)],
                         [h - m for m, h in zip(means, his)]])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, best_single_means, width,
           yerr=yerr(best_single_means, best_single_cilo, best_single_cihi),
           label="Best single feature (oracle)", color="#6b7280", alpha=0.85,
           capsize=3)
    ax.bar(x - 0.5*width, lr_means, width,
           yerr=yerr(lr_means, lr_cilo, lr_cihi),
           label="Logistic regression (9 features)", color="#3b82f6", alpha=0.85,
           capsize=3)
    ax.bar(x + 0.5*width, rf_means, width,
           yerr=yerr(rf_means, rf_cilo, rf_cihi),
           label="Random forest (9 features)", color="#10b981", alpha=0.85,
           capsize=3)
    ax.bar(x + 1.5*width, gbt_means, width,
           yerr=yerr(gbt_means, gbt_cilo, gbt_cihi),
           label="Gradient boosted trees (9 features)", color="#ef4444", alpha=0.85,
           capsize=3)

    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("5-fold CV AUC", fontsize=11)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("CoT correctness verification: non-linear fusion vs single features\n"
                 "(5-fold CV AUC with 95% bootstrap CI)", fontsize=12)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(save_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path.with_suffix('.png')}")
    print(f"  saved: {save_path.with_suffix('.pdf')}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = latest_per_trace_file()
    print(f"Reading: {path.name}\n")
    with open(path) as f:
        items = json.load(f)
    print(f"Total traces: {len(items)}\n")

    print("=" * 72)
    print("PART 1: Orthogonality analysis")
    print("=" * 72)
    orth_rows = orthogonality_per_condition(items)
    rates = [r["disagreement_rate"] for r in orth_rows]
    print(f"Conditions analyzed: {len(orth_rows)}")
    print(f"Disagreement rate range: {min(rates)*100:.1f}% to {max(rates)*100:.1f}%")
    print(f"Mean disagreement rate: {np.mean(rates)*100:.1f}%")
    print(f"Median disagreement rate: {np.median(rates)*100:.1f}%")
    print()
    print(f"{'model':<40} {'domain':<10} {'n':>4} {'disagr.':>9} {'only_sp':>8} {'only_ri':>8}")
    for r in orth_rows:
        short_m = r["model"].split("/")[-1][:38]
        print(f"{short_m:<40} {r['domain']:<10} {r['n']:>4} "
              f"{r['disagreement_rate']*100:>8.1f}% {r['only_sp']:>8} {r['only_ri']:>8}")
    print()

    stamp = time.strftime("%Y%m%d_%H%M%S")

    heatmap_path = RESULTS_DIR / f"stage1_orthogonality_heatmap_{stamp}"
    print(f"Generating orthogonality heatmap...")
    plot_orthogonality_heatmap(orth_rows, heatmap_path)
    print()

    print("=" * 72)
    print("PART 2: Fusion AUC with confidence intervals")
    print("=" * 72)
    print("Computing 5-fold CV AUC for: best single, LR, RF, GBT per condition...")
    print("(bootstrap 95% CI on the 5 fold-level AUCs per method)")
    print()
    fusion_rows = fusion_auc_analysis(items)

    print(f"{'condition':<42} {'oracle':>15} {'LR':>15} {'RF':>15} {'GBT':>15}")
    for r in fusion_rows:
        short_m = r["model"].split("/")[-1][:20]
        cond = f"{short_m}__{r['domain']}"
        def fmt(d):
            return f"{d['mean']:.3f} [{d['ci_low']:.3f},{d['ci_high']:.3f}]"
        print(f"{cond:<42} {fmt(r['oracle_single']):>15} {fmt(r['lr']):>15} "
              f"{fmt(r['rf']):>15} {fmt(r['gbt']):>15}")

    best = max(fusion_rows, key=lambda r: r["gbt"]["mean"])
    print()
    print(f"Best GBT AUC: {best['gbt']['mean']:.3f} "
          f"[{best['gbt']['ci_low']:.3f}, {best['gbt']['ci_high']:.3f}] "
          f"on {best['model'].split('/')[-1]} x {best['domain']}")
    print(f"  vs best single ({best['oracle_single']['name'] if 'name' in best['oracle_single'] else 'N/A'}): "
          f"{best['oracle_single']['mean']:.3f} "
          f"[{best['oracle_single']['ci_low']:.3f}, {best['oracle_single']['ci_high']:.3f}]")
    lift = best["gbt"]["mean"] - best["oracle_single"]["mean"]
    print(f"  lift: +{lift:.3f} AUC points")

    rf_wins = sum(1 for r in fusion_rows if r["rf"]["mean"] > r["oracle_single"]["mean"])
    gbt_wins = sum(1 for r in fusion_rows if r["gbt"]["mean"] > r["oracle_single"]["mean"])
    print(f"\nWins over oracle-best-single: RF {rf_wins}/{len(fusion_rows)}, "
          f"GBT {gbt_wins}/{len(fusion_rows)}")
    print()

    fusion_path = RESULTS_DIR / f"stage1_fusion_auc_{stamp}"
    print(f"Generating fusion AUC comparison figure...")
    plot_fusion_auc(fusion_rows, fusion_path)
    print()

    combined = {
        "source_per_trace": str(path),
        "orthogonality": orth_rows,
        "fusion_auc": fusion_rows,
        "headline_numbers": {
            "disagreement_range_pct": [min(rates)*100, max(rates)*100],
            "mean_disagreement_pct": float(np.mean(rates) * 100),
            "best_condition": {
                "model": best["model"],
                "domain": best["domain"],
                "gbt_auc": best["gbt"]["mean"],
                "gbt_ci": [best["gbt"]["ci_low"], best["gbt"]["ci_high"]],
                "oracle_single_auc": best["oracle_single"]["mean"],
                "lift": lift,
            },
            "rf_wins_over_oracle": rf_wins,
            "gbt_wins_over_oracle": gbt_wins,
            "total_conditions": len(fusion_rows),
        }
    }
    combined_path = RESULTS_DIR / f"stage1_numbers_{stamp}.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"All numbers saved to: {combined_path}")
    print()

    print("=" * 72)
    print(f"  - orthogonality across {len(orth_rows)} conditions, "
          f"mean disagreement {np.mean(rates)*100:.1f}%")
    print(f"  - fusion AUC across {len(fusion_rows)} conditions, "
          f"best {best['gbt']['mean']:.3f} "
          f"[{best['gbt']['ci_low']:.3f}, {best['gbt']['ci_high']:.3f}]")
    print(f"  - lift over best single feature: +{lift:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
