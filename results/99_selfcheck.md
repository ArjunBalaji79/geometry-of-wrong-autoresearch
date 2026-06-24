# SELF-CHECK — claim ↔ artifact audit

Phase 4 of AUTORESEARCH.md. Every claim the paper makes, the artifact that backs
it, and an honest flag on weak/skipped controls and approximations. Written
before paper.md.

## Claim → artifact table

| # | Claim | Backing artifact | Status |
|---|---|---|---|
| 1 | Corpus = 3,493 traces after the documented filters, regenerated from raw | `results/01_extract_summary.json` (kept=3493; 15 truncation + 92 <3-sent) | matches README count |
| 2 | 9-dim signature extracted via geom graph+spectral + ricci-numpy curvature | `src/features.py`, `results/features.jsonl` | OK |
| 3 | Curvature is exact; fast path bit-exact to reference (≤1.3e-15) on real graphs | `results/00_curvature_backend_check.json` | OK (machine-eps) |
| 4 | PrOntoQA `ground_truth` constant; `correct` flag corrupted in the incorrect direction | `results/00_prontoqa_label_audit.json` | OK; drives label policy |
| 5 | GSM8K/FOLIO flags reliable (FOLIO 100%, GSM8K ~97%+ where parseable) | `results/02_label_summary.json` (`recomputed_vs_released_agreement`) | OK |
| 6 | Failure-mode taxonomy defined geometry-free | `src/labels.py` (uses only gold + answer parsing + engine) | OK |
| 7 | D1 in-dist: both tasks beat permutation null & chance (SC-D1a) | `results/13_d1_permutation.json` (mode p=.005, binary p=.005) | PASS |
| 8 | D1 CRUX: residualized geometry does NOT beat surface for mode in any LODO fold | `results/11_d1_lodo.json` (deltas all CI-include-0) | SC-D1b FAIL → null |
| 9 | D1: binary does not transfer LODO (AUC≈0.5) | `results/11_d1_lodo.json` (0.505/0.512/0.427) | OK |
| 10 | D1 null holds under GBT and under the PrOntoQA released-flag relabeling | `results/15_d1_gbt.json`, `results/14_d1_robustness_prontoqa_flag.json` | OK |
| 11 | Composition-shift: OOD binary drop not predicted by mode-mix divergence (n=3) | `results/12_d1_composition_shift.json` (r=−0.63, n=3) | null, descriptive |
| 12 | D2 subset = 96 PrOntoQA M3 traces with a localizable broken step | `results/20_d2_localization.json` (`n_subset`) | OK |
| 13 | D2 alignment accuracy ≈93% (28/30), residual error named | `results/21b_alignment_audit_manual.json` | OK, bounded |
| 14 | D2 BAR: curvature ties max-transition-energy (CI includes 0) | `results/20_d2_localization.json` (`BAR_curvature_minus_mte`) | SC-D2 NOT met → tie |
| 15 | D2: curvature significantly BELOW random | `results/20_d2_localization.json` (`curvature_minus_random` CI<0) | OK (key negative) |
| 16 | D2: argmin edge identical under reference vs fast solver in 86/88 graphs; 2 differ at machine-epsilon curvature ties | `results/22_d2_backend_argmin_check.json` | SC-D2c partial; see #17 |
| 17 | D2 conclusion solver-invariant: pure-numpy reference argmin on 54/96 graphs (≤150 edges) — curv 0.204 = MTE 0.204 (BAR Δ=0.0), ≪ random 0.476 | `results/23_d2_reference_solver.json` | confirms tie + below-random |

## Honest flags

- **Weak control / small n**: the composition-shift check has only 3 LODO folds
  (n=3). Reported descriptively; no inferential weight. NOT dropped (required).
- **Taxonomy researcher-degrees-of-freedom**: the mode taxonomy maps one
  mechanism scheme onto three domains via domain-specific objective rules; the
  GSM8K M2/M3 split uses a pre-registered relative-error threshold τ=0.5. M1
  (abstain) is thin in GSM8K (n=6). These are judgment calls, pre-registered in
  PLAN.md, and the D1 conclusion (geometry ≯ surface OOD) is a NULL that label
  noise would only reinforce, not manufacture.
- **No human gold for modes**: modes are objective/engine-derived, not
  human-annotated. The PrOntoQA engine IS validated against the released labels
  (88–94% on flag-correct traces). GSM8K/FOLIO modes inherit answer-parser noise.
- **D2 alignment is the load-bearing judgment**: the broken-step detector was
  hardened in three stages (181→140→96) driven solely by audited alignment
  accuracy. This MOVED THE BAR AGAINST the hypothesis (curv>MTE at 140 became a
  tie at 96) — documented to show it is not outcome-tuned. Residual ~7% alignment
  error perturbs all methods' hit-rates symmetrically.
- **D2 excludes M1/M2**: traces with no present broken step (omission / no answer)
  are excluded — you cannot localize an absent step. Disclosed; subset size given.
- **Approximation use**: NONE for curvature. The fast path is the same exact
  network-simplex algorithm as the reference (bit-exact to machine epsilon,
  verified on corpus graphs). No Sinkhorn/POT/GraphRicciCurvature anywhere.
- **Solver-tie sensitivity (honest)**: the QUESTION warned the argmin edge is
  sensitive to solver error. The two exact solvers agree on the argmin in 86/88
  D2 graphs; the 2 disagreements are edges whose curvatures tie to within ~1e-15
  (the argmin is then arbitrary for *either* exact solver — not an approximation
  error). The D2 headline is recomputed with the pure-numpy REFERENCE solver on
  the full subset (`results/23_d2_reference_solver.json`); the conclusion
  (curvature ties MTE, falls below random) is unchanged.
- **`ensure_connected` chain edges**: `geom/graph.py` adds sequential (i,i+1)
  edges when the cosine graph is disconnected. These artificial edges can be the
  argmin/last edge in D2; this is inherited from the provided pipeline and noted
  as a limitation (it is part of the released feature definition).

## Bottom line
Both headline hypotheses (D1 mode-transfer beyond surface; D2 curvature
localization) FAIL their pre-registered success criteria. The in-distribution
mode signal is real but not separable from surface confounds and does not
transfer; the argmin-curvature edge localizes the broken step below chance. These
are reported as the primary results.
