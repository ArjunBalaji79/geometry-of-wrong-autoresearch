# PLAN — D1 and D2 (pre-registered before any result)

Committed before running EXECUTE. Success criteria are stated here and judged,
unchanged, in EXECUTE / SELF-CHECK / paper. Every metric dumps to `results/`.

## Shared setup

- **Inputs**: `results/features.jsonl` (9-dim signature + per-edge curvature +
  surface features per trace) and `results/labels.jsonl` (primary correctness +
  failure mode), both regenerated from raw traces in SURVEY.
- **Signature (9-dim)**: `spectral_entropy, fiedler_value, grs, shs, hfer,
  mean_kappa, std_kappa, frac_negative, min_kappa`. Curvature is exact
  (ricci-numpy).
- **Surface features (control)**: `log_n_tokens, repetition_rate,
  bigram_repetition`. These are NEVER part of the 9-dim signature and were not
  used to define the failure modes.
- **Grouping**: a "problem" = `(benchmark, problem_idx)`. All CV is grouped so a
  problem never appears in both train and test (`sklearn GroupKFold`, k=5, for
  in-distribution; benchmark identity for LODO).
- **Correctness policy (from SURVEY)**: released flag for GSM8K/FOLIO, engine for
  PrOntoQA. Robustness: D1 binary/mode re-run with the released flag on PrOntoQA
  too, to show the conclusion is not manufactured by the relabeling.
- **Classifier**: primary = L2-regularized logistic regression on standardized
  features (binary) / multinomial logistic (3-class mode). Secondary robustness =
  `HistGradientBoostingClassifier`. Same family for binary and mode so the
  contrast is fair. `seed = 0`.
- **Bootstrap**: paired, ≥1,000 resamples over test instances (resampled by
  problem-group to respect grouping), 95% percentile CI on every headline delta.
- **Permutation null**: labels shuffled within the training pool, model retrained,
  metric must collapse to chance. ≥200 permutations for the null distribution.

## D1 — failure MODE vs failure PRESENCE

### Tasks
- **Binary** task: predict `correct` vs `incorrect` from the 9-dim signature.
- **Mode** task: 3-class M1/M2/M3 among incorrect traces.

### Feature conditions (applied identically to BOTH tasks — symmetric control)
1. **GEOM9** — the 9 signature features (raw).
2. **SURFACE** — the 3 surface features only (the length+repetition baseline).
3. **GEOM9⊥SURF** — the 9 features residualized on the 3 surface features:
   per feature, fit OLS(geom ~ surface) on the TRAIN fold, replace with residuals
   on train and test. This is the clean geometric signal with length/repetition
   removed. (Required-control residualization, applied to both tasks.)
4. **GEOM9+SURF** — concatenation (reported for completeness).

### Metrics
- Binary: ROC-AUC (primary), balanced accuracy.
- Mode: macro-F1 (primary), balanced accuracy. Chance balanced-acc = 1/3.

### Evaluations
- **In-distribution**: 5-fold grouped CV pooled across all benchmarks; per-task,
  per-condition metric + bootstrap CI + permutation null.
- **Leave-one-dataset-out (LODO, the CRUX)**: for each held-out benchmark, train
  on the other two, test on the held-out one. Report binary AUC and mode macro-F1
  for every condition.

### Composition-shift check
For each LODO fold compute (a) the binary OOD drop = in-distribution binary AUC −
LODO binary AUC, and (b) the train↔test **mode-mixture divergence** (Jensen-
Shannon divergence between train and test M1/M2/M3 distributions). Report the
relationship across the 3 folds (n=3 — reported descriptively, plus a finer
per-test-model breakdown for more points). Tests whether OOD binary degradation
tracks mode-composition shift.

### Pre-registered success criteria
- **SC-D1a (sanity)**: in-distribution, both tasks beat their permutation null and
  chance — binary AUC CI excludes 0.5; mode balanced-acc CI excludes 1/3.
- **SC-D1b (CRUX, the headline)**: in LODO, mode prediction under **GEOM9⊥SURF**
  beats BOTH (i) chance (1/3) and (ii) the **SURFACE** baseline, with the paired
  bootstrap CI on (GEOM9⊥SURF − SURFACE) macro-F1 **excluding 0**, in **≥2 of 3**
  held-out benchmarks; AND binary prediction does **not** transfer (LODO binary
  AUC CI overlaps/sits near 0.5). The D1 hypothesis is SUPPORTED only if mode
  transfers above the surface baseline where binary does not. If GEOM9⊥SURF mode
  fails to beat SURFACE, we report the **null** (surface confound not separable).
- **SC-D1c (control integrity)**: residualization and the surface baseline are
  applied to binary and mode identically; reported side by side regardless of
  which way the result cuts.

## D2 — negative curvature localizes the broken step (PrOntoQA)

### Subset
PrOntoQA traces that are (a) engine-incorrect and (b) have an identifiable
invalid deduction = at least one asserted entity-property not derivable under the
gold ontology (mode M3), and (c) whose broken sentence maps to a graph node and
whose graph has ≥2 edges. M1 (no answer) and M2 (wrong by omission — no present
broken step) are **excluded with disclosure**: you cannot localize a step that is
absent. Subset size reported.

### Broken-step gold + sentence-to-step alignment
- Engine (`src/prontoqa_logic.py`) gives the gold derivable-property set.
- Broken step = the FIRST trace sentence that asserts an entity-property not in
  the gold set (the first invalid inference). Graph nodes ARE sentences, so the
  "step" is a sentence index `b`.
- **Alignment error bound (required)**: manually audit a random sample (n≈30) of
  located broken sentences — does the located sentence actually contain the first
  invalid deduction? Report the agreement rate as the empirical alignment
  accuracy; it bounds D2 (and applies equally to all methods, so the paired delta
  is robust to it).

### Methods (each selects ONE edge per trace)
- **CURVATURE**: argmin-κ edge (most negative exact Ollivier-Ricci). The
  argmin edge is re-confirmed under the pure-numpy reference solver on this subset
  (backend-independence of the headline).
- **MAX-TRANSITION-ENERGY** (the BAR): graph edge maximizing semantic
  discontinuity = `1 − cos(x_i, x_j)` (lowest cosine among connected pairs).
- **RANDOM**: uniform random edge (null; expectation computed analytically per
  trace and averaged).
- **LAST-EDGE**: the edge incident to the highest sentence index.

### Hit criterion
A method's edge `(u,v)` **hits** the broken step `b` if
`min(|u−b|, |v−b|) ≤ 1` (on the broken sentence or an immediate neighbor —
"within one step"). Hit-rate = fraction of subset traces hit. Also report the
strict (`=0`) variant.

### Pre-registered success criteria
- **SC-D2 (BAR, headline)**: CURVATURE hit-rate − MAX-TRANSITION-ENERGY hit-rate
  > 0 with a paired bootstrap 95% CI **excluding 0**. If the CI includes 0 →
  **tie**: curvature merely redescribes semantic discontinuity — reported plainly
  as the outcome.
- **SC-D2b (context)**: CURVATURE beats RANDOM and LAST-EDGE (paired bootstrap CI
  on each delta excluding 0).
- **SC-D2c (exactness)**: argmin edge identical under reference vs fast solver on
  the subset (else re-run with reference).

## Outputs
- D1: `results/10_d1_indist.json`, `results/11_d1_lodo.json`,
  `results/12_d1_composition_shift.json`, `results/13_d1_permutation.json`,
  `results/14_d1_robustness_prontoqa_flag.json`.
- D2: `results/20_d2_localization.json`, `results/21_d2_alignment_audit.json`,
  `results/22_d2_backend_argmin_check.json`.
- Tables echoed into `paper.md`; SELF-CHECK in `results/99_selfcheck.md`.
