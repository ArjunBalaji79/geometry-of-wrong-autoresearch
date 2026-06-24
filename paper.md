# Black-box geometric signatures of chain-of-thought failure: two pre-registered negative results

*Autonomous first-draft artifact. Both headline hypotheses fail their
pre-registered success criteria; this is reported as the primary result.*

## Abstract

We test two claims that a black-box, text-only geometric signature of a
chain-of-thought (CoT) trace — 5 spectral + 4 exact Ollivier-Ricci features over
a sentence-level cosine-proximity graph — carries information about *how* and
*whether* reasoning fails, across GSM8K, FOLIO, and PrOntoQA (6 models, 3,493
temperature-0 traces).

**D1 (failure mode is a more domain-general target than failure presence): not
supported.** In-distribution, the signature predicts both the failure mode
(3-class macro-F1 0.49 vs a 0.26 permutation null, p=0.005) and the binary
correct/incorrect label (AUC 0.556 vs 0.495 null, p=0.005) above chance. But the
*required* symmetric control kills the headline: once length and repetition are
residualized out, the geometric signal for mode does **not** beat a
length+repetition surface baseline either in-distribution (Δmacro-F1 = +0.007,
95% CI [−0.034, +0.047]) or leave-one-dataset-out (0 of 3 held-out benchmarks
beat surface with a CI excluding 0). Binary prediction does not transfer either
(LODO AUC 0.51/0.51/0.43). The out-of-distribution binary drop is not predicted
by train/test mode-mixture divergence (n=3 folds, descriptive).

**D2 (negative curvature localizes the broken step): not supported.** On 96
PrOntoQA traces with a structurally identifiable broken deduction (alignment
accuracy ≈93% by manual audit), the most-negatively-curved edge hits the broken
step (within one sentence) only 19% of the time — **below the 36% expectation of
a random edge** (Δ = −0.18, 95% CI [−0.25, −0.10]). It ties the
max-transition-energy baseline that the pre-registered BAR required it to beat
(Δ = +0.04, 95% CI [−0.04, +0.13]); per pre-registration a tie means curvature at
most redescribes semantic discontinuity — and here neither localizes.

We also document a **data-integrity finding**: the released PrOntoQA labels are
corrupted (the `ground_truth` field is a constant; the `correct` flag marks many
logically-correct traces wrong), which we correct with a validated forward-
chaining engine. All curvature is exact (ricci-numpy); no approximate solver is
used anywhere.

## 1. Introduction

A black-box signature that flags *where* or *why* a reasoning trace breaks —
without logits or activations — would be useful and surprising. Two specific
hypotheses are tested here, exactly as posed in the task:

- **D1**: a 9-dim geometric signature predicts the *type* of reasoning failure in
  a way that transfers across benchmarks even where the binary correct/incorrect
  label does not.
- **D2**: the most negatively curved Ollivier-Ricci edge in a trace's semantic
  graph coincides with the step where the deduction breaks.

Our contribution is a rigorous, pre-registered (PLAN.md) test of both, with the
controls that distinguish a real geometric signal from surface confounds and
chance. Both hypotheses fail those controls. We report the nulls in full, per the
operating rules.

## 2. Method

**Corpus.** 18 trace files (6 models × {GSM8K, FOLIO, PrOntoQA}, 200 problems
each) = 3,600 raw traces; after the repository's documented 15-trace generation-
truncation exclusion (EXCLUSIONS.md) and a ≥3-sentence/≥1-edge filter, **3,493
traces** remain — matching the released count, regenerated from raw
(`results/01_extract_summary.json`).

**Signature (9-dim).** Per trace we split into sentences, embed with MiniLM-L6-v2,
build the cosine ε-proximity graph (ε=0.3; sequential edges added if
disconnected) using the repository's `geom/` pipeline, and compute 5 spectral
features (spectral entropy, Fiedler value, graph Rayleigh smoothness, high-
frequency score, high-frequency energy ratio) and 4 Ollivier-Ricci features (mean,
std, fraction-negative at κ<−0.05, min) (`src/features.py`).

**Exact curvature.** All curvature uses `ricci-numpy`'s exact network-simplex
solver. Its numba path is the same algorithm as the pure-numpy reference and is
bit-exact to 1.3×10⁻¹⁵ over 8,068 corpus edges
(`results/00_curvature_backend_check.json`); the reference solver (48 s/graph,
~46 h corpus-wide) is infeasible at scale so we use the fast path for the bulk and
re-verify the D2 argmin under the reference solver (§4.2). No Sinkhorn/POT
approximation is used.

**Correctness labels (data-integrity correction).** The PrOntoQA `ground_truth`
field is the constant `"True"`, and its `correct` flag is corrupted: on traces it
marks correct, the model answer matches an independent forward-chaining engine at
a uniform 88–94% across all six models, but on traces it marks *incorrect* the
agreement collapses asymmetrically (claude 8%, gpt-oss 13%) — the signature of
label corruption, not engine error (`results/00_prontoqa_label_audit.json`). For
claude, 92% of PrOntoQA "incorrect" traces are in fact logically correct. We
therefore use a validated forward-chaining engine
(`src/prontoqa_logic.py`) as the PrOntoQA correctness gold; GSM8K and FOLIO flags
are reliable (FOLIO 100%, GSM8K ~97%+ where parseable) and used as released
(`results/02_label_summary.json`).

**Failure-mode taxonomy (geometry-free).** For incorrect traces we assign one of
three *mechanism* classes by a uniform scheme instantiated per domain with
objective signals only (ground truth + answer parsing + the PrOntoQA engine) —
never the geometric features that the classifier uses, which is how we avoid
circularity:

- **M1 abstain / under-commit** (no committed in-space answer; FOLIO "Unknown"
  when gold is definite),
- **M2 local slip** (committed wrong but engaged: GSM8K relative error ≤0.5;
  FOLIO True↔False flip; PrOntoQA wrong but every asserted property is valid),
- **M3 misplan / hallucination** (GSM8K gross error; FOLIO over-commits on an
  Unknown; PrOntoQA asserts a non-derivable property).

Counts (incorrect traces): FOLIO 144/84/198, GSM8K 6/231/98, PrOntoQA 162/147/81
(M1/M2/M3) (`results/02_label_summary.json`).

**Classifiers & CV.** L2 logistic (binary) / multinomial logistic (3-class mode),
standardized; HistGradientBoosting as a robustness check. CV is grouped so a
problem `(benchmark, idx)` never spans folds; LODO trains on two benchmarks and
tests on the third. Headline deltas use a group-aware paired bootstrap (2,000
resamples); a label-permutation null (200) must collapse to chance.

**Surface control (symmetric, required).** Surface features = {log token length,
repetition rate, bigram repetition}. We run, for **both** tasks identically: GEOM9
(raw), SURFACE (only), GEOM9⊥SURF (geometry residualized on surface, fit on
train), GEOM9+SURF. The mode hypothesis counts only if GEOM9⊥SURF beats SURFACE.

## 3. D1 results

### 3.1 In-distribution (grouped 5-fold)

| condition | binary AUC | mode macro-F1 |
|---|---|---|
| GEOM9 | 0.556 | 0.490 |
| SURFACE | 0.515 | 0.468 |
| GEOM9⊥SURF | 0.629 | 0.475 |
| GEOM9+SURF | 0.634 | 0.513 |

(`results/10_d1_indist.json`.) Both tasks beat their permutation null
(mode 0.475 vs 0.258, p=0.005; binary 0.556 vs 0.495, p=0.005;
`results/13_d1_permutation.json`) — **SC-D1a passes**. But the clean test of
geometry *beyond surface* for mode is a tie: GEOM9⊥SURF − SURFACE = **+0.007, 95%
CI [−0.034, +0.047]**. The mode signal in GEOM9 is largely surface-explainable.

### 3.2 Leave-one-dataset-out (the CRUX)

| held-out | binary AUC (GEOM9) | mode F1 GEOM9⊥SURF | mode F1 SURFACE | Δ(resid−surf) 95% CI |
|---|---|---|---|---|
| gsm8k | 0.505 | 0.201 | 0.196 | +0.005 [−0.041, +0.055] |
| folio | 0.512 | 0.220 | 0.217 | +0.003 [−0.055, +0.060] |
| prontoqa | 0.427 | 0.345 | 0.306 | +0.038 [−0.019, +0.094] |

(`results/11_d1_lodo.json`.) **SC-D1b fails**: residualized geometry beats the
surface baseline for mode in **0 of 3** folds with a CI excluding 0; and binary
prediction does not transfer (AUC ≈ 0.5, below it for PrOntoQA). The hypothesis —
mode transfers above surface where binary does not — is not supported. The same
pattern holds under HistGradientBoosting (gsm8k Δ = −0.115, CI [−0.161, −0.068];
folio/prontoqa ties; `results/15_d1_gbt.json`) and under the corrupted PrOntoQA
released flag (all folds tie; `results/14_d1_robustness_prontoqa_flag.json`), so
the null is not an artifact of our relabeling.

### 3.3 Composition-shift check

Across the 3 LODO folds, the binary OOD AUC drop does not track train/test
mode-mixture Jensen-Shannon divergence (Pearson r = −0.63, n=3; the largest drop,
PrOntoQA at 0.129, has the *smallest* divergence, 0.034). With n=3 this is
descriptive only and provides no support (`results/12_d1_composition_shift.json`).

## 4. D2 results

### 4.1 Localization (PrOntoQA, 96 traces)

Subset: engine-incorrect PrOntoQA traces with an identifiable invalid inference
(M3) whose broken sentence maps to a node (graph ≥2 edges). The broken step is the
first sentence affirmatively asserting an entity-property not derivable under the
gold ontology. Sentence-to-step alignment accuracy is **≈93% (28/30, manual
audit)**; the 2 residual errors are a plural rule-clause and an "either/or"
hedge, which perturb all methods' targets symmetrically
(`results/21b_alignment_audit_manual.json`).

| method | hit-rate (within 1 step) |
|---|---|
| **curvature (argmin κ)** | **0.188** |
| max-transition-energy | 0.146 |
| last edge | 0.156 |
| random edge (expectation) | **0.364** |

(`results/20_d2_localization.json`.) Pre-registered comparisons:

- **BAR — curvature vs max-transition-energy: Δ = +0.042, 95% CI [−0.042,
  +0.125]** → CI includes 0 → **tie**. SC-D2 not met. Per pre-registration, a tie
  means curvature at most redescribes semantic discontinuity.
- curvature vs random: **Δ = −0.177, 95% CI [−0.251, −0.096]** → curvature is
  *significantly worse than chance* at localizing the broken step.
- curvature vs last edge: Δ = +0.031, CI [−0.042, +0.104] → tie.

Mechanistically, the argmin-curvature edge is a structural bridge between
dissimilar sentence clusters (e.g. problem-statement ↔ reasoning), whereas the
broken inference sits *inside* a dense, self-similar reasoning cluster whose edges
are positively curved — so the argmin systematically points away from it.

### 4.2 Solver-exactness robustness

The two exact solvers agree on the argmin edge in 86/88 checked graphs (≤400
edges); the 2 disagreements are edges whose curvatures tie to within ~10⁻¹⁵ (the
argmin is then arbitrary for either exact solver)
(`results/22_d2_backend_argmin_check.json`). Recomputing the headline with the
pure-numpy reference solver on the 54 subset graphs it can handle (≤150 edges; the
dense remainder is infeasible for the O(n³) reference but is covered by the
argmin-identity check above) leaves the conclusion unchanged: curvature 0.204 vs
MTE 0.204 (BAR Δ = 0.0, CI [−0.11, +0.11]) and well below random 0.476 (Δ =
−0.27, CI [−0.38, −0.16]) (`results/23_d2_reference_solver.json`).

## 5. Limitations

- **Taxonomy degrees of freedom.** The 3-class mechanism taxonomy maps one scheme
  onto three domains via objective per-domain rules (pre-registered); the GSM8K
  slip/misplan split uses a relative-error threshold τ=0.5, and M1 is thin in
  GSM8K (n=6). Modes are objective/engine-derived, not human gold; GSM8K/FOLIO
  modes inherit answer-parser noise. The D1 result is a null that such noise
  reinforces rather than manufactures.
- **D2 alignment is judgment-heavy.** The broken-step detector was hardened in
  three audited stages (181→140→96 traces); this moved the BAR *against* the
  hypothesis (curv>MTE became a tie), i.e. it was driven by alignment accuracy,
  not by the outcome. D2 is scoped to PrOntoQA M3 traces; M1/M2 (no present broken
  step) are excluded by construction.
- **Pipeline artifact.** `geom/graph.py` adds sequential chain edges when the
  cosine graph is disconnected; such edges can be the argmin/last edge. This is
  inherited from the released feature definition.
- **Over-squashing is analogy, not mechanism.** We make no claim that negative
  curvature is causally tied to reasoning failure; the data do not support even a
  descriptive localization.
- **Scope.** 3 benchmarks, 6 models, temperature-0 English, one embedding model,
  ε=0.3. The composition-shift test has n=3 folds.

## 6. Conclusion

Under symmetric surface controls, grouped CV, and exact curvature, neither a
geometric *mode* signal that survives a length+repetition baseline and transfers
across benchmarks (D1), nor a curvature-based localization of the broken step that
beats chance (D2), is present in this corpus. The honest headline is two negative
results — plus a corrected PrOntoQA labeling that future black-box analyses of
this release should adopt.

*All numbers above are reproducible from `results/` (regenerate via
`src/extract_all.py` → `src/build_labels.py` → `src/run_d1.py` / `src/run_d2.py`
/ `src/d2_reference_argmin.py`).*
