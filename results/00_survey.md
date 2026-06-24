# SURVEY — repo map, smoke test, corpus, data-integrity audit

Phase 1 of AUTORESEARCH.md. All numbers here are reproducible from committed
artifacts in `results/` and code in `src/`.

## 1. Repo map

- **Traces**: `data/traces/<benchmark>_<model>.json`, 18 files
  (6 models × 3 benchmarks), 200 problems each → 3,600 raw records. Schema per
  record: `question, cot_trace, final_answer, model, temperature, backend,
  ground_truth, correct, problem_idx` (README.md). All `temperature = 0.0`.
- **Feature pipeline** (reused): `geom/graph.py` (cosine ε-proximity graph,
  ε = 0.3, MiniLM embeddings, chain edges added if disconnected),
  `geom/spectral.py` (5 spectral features), and the shared
  `split_sentences`/`embed` logic mirrored in `code/rerun_common.py`.
- **Exact curvature** (reused): `ricci-numpy/` — pure-numpy network-simplex
  Ollivier-Ricci. We use this for ALL curvature.

The 9-dim signature = 5 spectral (`spectral_entropy, fiedler_value, grs, shs,
hfer`) + 4 Ollivier-Ricci (`mean_kappa, std_kappa, frac_negative, min_kappa`),
fixed order in `src/features.py:SIG_KEYS`.

We did **not** reuse any end-to-end analysis script in `code/`; D1/D2 are
implemented fresh in `src/`. `results_prior/` and `*_RERUN*.json` were not read.

## 2. Smoke test (10 traces)

`gsm8k_claude_sonnet[:10]` extracted cleanly (10/10 ok, 0 failures); 9-dim
signature populated with plausible values (e.g. spectral entropy 2.2–3.1,
frac_negative 0–0.012). Confirms load → split → embed → graph → spectral +
ricci-numpy curvature works before scaling.

## 3. Curvature backend (exactness is load-bearing — QUESTION.md)

`ricci-numpy` ships a pure-numpy reference (`ricci_numpy`) and a numba JIT path
(`ricci_numpy.fast`) that is the **same exact network-simplex algorithm**.

- Verified bit-exactness on **real corpus graphs**, not just canonical cases
  (`src/verify_curvature_backends.py` → `results/00_curvature_backend_check.json`):
  max |Δκ| = **1.33 × 10⁻¹⁵** over 69 graphs / 8,068 edges (machine epsilon from
  float summation order). This is 13 orders of magnitude below the Sinkhorn/POT
  error (≈3.8 × 10⁻²) the task forbids, and far below both the `frac_negative`
  threshold (κ < −0.05) and any real argmin gap.
- Reference solver timing: **48 s/graph** average (one 1,770-edge graph dominates)
  → ≈46 h corpus-wide, infeasible. Fast path: **0.09 s/graph** → ≈5 min corpus-wide.

**Decision**: use the fast path for the bulk 9-dim extraction. For D2's headline
argmin we additionally re-confirm the argmin edge under the reference solver on
the (small) PrOntoQA subset, so the headline is provably backend-independent. We
never use an approximate (Sinkhorn/POT/GraphRicciCurvature) solver.

## 4. Corpus after filtering

`src/extract_all.py` → `results/features.jsonl`, `results/01_extract_summary.json`.

- 3,600 raw − 15 generation-truncation exclusions (`EXCLUSIONS.md`, model-blind
  rule) − 92 traces with <3 sentences / no graph → **3,493 kept**.
- This **matches the README's documented count (3,493) exactly**, regenerated
  from raw traces.
- Per-cell kept counts in `results/01_extract_summary.json` (e.g. gpt_oss_120b
  loses the most to the 3-sentence filter: 31 FOLIO, 56 PrOntoQA — consistent
  with its 12% empty-output rate noted in EXCLUSIONS.md).

## 5. DATA-INTEGRITY AUDIT (consequential — changes the label policy)

### 5a. PrOntoQA `ground_truth` is broken; the `correct` flag is corrupted

- The JSON `ground_truth` field is the **constant string `"True"`** for all 200
  PrOntoQA problems in every model file (`gt_unique = 1`). It is unusable.
- We built an independent **forward-chaining proof engine** from the question
  text (`src/prontoqa_logic.py`): parse the ontology (universal + disjunctive
  rules), forward-chain the entity's properties under closed-world assumption,
  and answer the True/False query. This engine is the **structural ground truth**.
- Validation (`results/00_prontoqa_label_audit.json`): on traces the dataset
  marks **correct**, the model's answer matches the engine gold at a uniform
  **88–94%** across all six models — strong triangulation that the engine is
  right. But on traces marked **incorrect**, the model's answer *disagrees* with
  the engine gold at wildly different rates:

  | model | flag=correct: model==engine | flag=incorrect: model≠engine |
  |---|---|---|
  | claude_sonnet | 133/146 = 0.91 | **4/52 = 0.08** |
  | gpt_oss_120b | 21/24 = 0.88 | **2/15 = 0.13** |
  | llama_3_1_8b | 71/78 = 0.91 | 39/72 = 0.54 |
  | gemini_2_5_flash | 172/188 = 0.91 | 5/9 = 0.56 |
  | gpt_4o_mini | 159/175 = 0.91 | 21/25 = 0.84 |
  | mistral_7b | 76/81 = 0.94 | 97/110 = 0.88 |

  The asymmetry (high, uniform agreement on flag=correct; collapsing agreement on
  flag=incorrect) is the signature of **label corruption in the incorrect
  direction**, not engine error (which would be symmetric). For claude, **92% of
  its PrOntoQA "incorrect" traces are in fact logically correct.**

### 5b. GSM8K / FOLIO flags are reliable

- FOLIO `correct` matches an independent 3-way {True,False,Unknown} recompute at
  **100%** (all 6 models). GSM8K matches a numeric recompute at 100% (gemini,
  gpt-4o-mini), 97% (mistral); claude/llama ≈86% gaps are answer-parser noise
  (units, multiple numbers), and gpt-oss 55% is its empty-output problem — not
  label corruption. (`results/02_label_summary.json`)

### 5c. Correctness policy adopted (and documented for all downstream use)

- **GSM8K, FOLIO**: use the released `correct` flag (validated reliable).
- **PrOntoQA**: use the engine-derived label (released flag corrupted).

This is the single most important methodological consequence of the survey: a
naive analysis on the released PrOntoQA flag would let label noise — not geometry
— drive the D1 binary-vs-mode contrast. We avoid that. The corruption is also
re-examined as a robustness check in EXECUTE.

## 6. Failure-mode taxonomy (defined here, used in D1)

Mechanism-based, domain-general, 3-class, defined **without any geometric
feature** (`src/labels.py`):

- **M1 ABSTAIN/UNDER-COMMIT** — no committed in-space answer (GSM8K: no number;
  PrOntoQA: no True/False; FOLIO: predicts Unknown when gold is definite).
- **M2 LOCAL SLIP** — committed wrong but engaged (GSM8K: rel-err ≤ 0.5;
  FOLIO: True↔False flip; PrOntoQA: wrong answer but every asserted property is
  valid under the ontology, engine-checked).
- **M3 MISPLAN/HALLUCINATION** — committed but fundamentally off (GSM8K: rel-err
  > 0.5; FOLIO: over-commits to True/False when gold is Unknown; PrOntoQA:
  asserts ≥1 property not derivable under the ontology).

Mode distribution over incorrect traces (`results/02_label_summary.json`):

| benchmark | M1 | M2 | M3 | total incorrect |
|---|---|---|---|---|
| folio | 144 | 84 | 198 | 426 |
| gsm8k | 6 | 231 | 98 | 335 |
| prontoqa | 162 | 47 | 181 | 390 |

All three modes appear in all three benchmarks (M1 is thin in GSM8K — math
models rarely abstain; flagged as a caveat for GSM8K-as-test-fold in LODO).

## 7. Transition to PLAN

Survey complete. Feature matrix (`results/features.jsonl`) and labels
(`results/labels.jsonl`) are built and regenerable. Next: PLAN.md with exact
baselines, controls, and pre-registered success criteria, committed before any
D1/D2 result.
