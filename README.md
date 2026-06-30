# Geometry of (Wrong) Reasoning Traces — Autoresearch

This is a **lossfunk autoresearch submission**: an autonomous agent (Claude Code)
was handed a research question and a corpus, and ran the whole study end to end —
survey, plan, code, experiments, self-check, paper — with no human in the loop.

The question: can a **black-box, text-only geometric signature** of a
chain-of-thought trace (5 spectral + 4 exact Ollivier–Ricci features over a
sentence-similarity graph) tell you *how* a model's reasoning fails (D1) or *where*
the deduction breaks (D2)? The honest answer the agent reached, under its own
pre-registered controls, is **two negative results** — both headline hypotheses
fail once you control for trace length and repetition. It also caught a real
data-integrity bug in the PrOntoQA labels along the way. Full writeup in
[paper.md](paper.md).

---

## How to read this repo

The submission is the trail of an autonomous run, so the files make the most sense
in the order the agent produced them:

| Order | File | What it is |
|---|---|---|
| 1 | [AUTORESEARCH.md](AUTORESEARCH.md) | The harness prompt. The operating rules the agent had to follow (use exact curvature, grouped CV, report nulls in full, etc.). |
| 2 | [QUESTION.md](QUESTION.md) | The research task. Defines claims **D1** (failure *mode* vs failure *presence*) and **D2** (curvature localizes the broken step) and the non-negotiable rigor requirements. |
| 3 | [results/00_survey.md](results/00_survey.md) | SURVEY phase: the agent maps the repo, confirms it can load traces and extract the 9-dim signature, and flags the PrOntoQA label bug. |
| 4 | [PLAN.md](PLAN.md) | PLAN phase: the concrete experiment list with **success criteria pre-registered before any result was seen**. |
| 5 | [results/](results/) | EXECUTE phase: every raw metric, as JSON (see the table below). |
| 6 | [results/99_selfcheck.md](results/99_selfcheck.md) | SELF-CHECK phase: the agent audits its own claims against the artifacts that back them before writing the paper. |
| 7 | [paper.md](paper.md) | WRITE phase: the final paper-style writeup (abstract, method, D1/D2 results, limitations). |

Two extra docs sit outside the agent's run:

- [predictions.md](predictions.md) — the **human's** predictions, written *before*
  the run and scored *after*. The meta-point of the submission: what the agent got
  right and where human judgment still mattered.
- [AUDIT_PRONTOQA.md](AUDIT_PRONTOQA.md) — a standalone, read-only diagnosis of the
  corrupted PrOntoQA labels (`ground_truth` is a constant; the `correct` flag is
  unreliable), and confirmation that this repo grades PrOntoQA independently.
- [EXCLUSIONS.md](EXCLUSIONS.md) — the 15-trace, model-blind generation-truncation
  exclusion rule applied at load time.

---

## Repository layout

```
.
├── README.md               ← this file
├── requirements.txt        ← pinned Python deps
│
├── AUTORESEARCH.md         ← agent harness prompt
├── QUESTION.md             ← the research task (D1, D2)
├── PLAN.md                 ← pre-registered plan + success criteria
├── paper.md                ← final writeup (two negative results)
├── predictions.md          ← human pre-run predictions, scored
├── AUDIT_PRONTOQA.md       ← PrOntoQA label-bug diagnosis
├── EXCLUSIONS.md           ← 15-trace truncation exclusion rule
│
├── data/                   ← the corpus (inputs to everything)
├── src/                    ← the autoresearch pipeline (what the agent wrote + ran)
├── geom/                   ← shared geometric-feature library (spectral + graph)
├── ricci-numpy/            ← bundled exact Ollivier–Ricci curvature solver
├── results/                ← all outputs of the autoresearch run
│
├── code/                   ← the prior provided pipeline (pre-autoresearch experiments)
└── results_prior/          ← outputs of that prior pipeline
```

### `data/` — the corpus

- `data/traces/` — **18 trace files**, `<benchmark>_<model>.json`, for 6 models ×
  {GSM8K, FOLIO, PrOntoQA} × 200 problems = 3,600 raw traces (3,493 after the
  truncation exclusion + a ≥3-sentence / ≥1-edge filter). Each record has
  `question, cot_trace, final_answer, model, ground_truth, correct, problem_idx`,
  etc. All generated at temperature 0. **These JSONs are the input to every
  analysis script** — the upstream generation pipeline (paid API keys + GPU) is not
  bundled.
- `data/embeddings_cache/` — MiniLM sentence-embedding cache (`*.npy`, one per
  trace). Regenerable; gitignored.

| Paper label | `model` string in the trace records |
|---|---|
| Claude Sonnet 4 | `claude-sonnet-4-20250514` |
| gpt-oss 120B | `gpt-oss-120b` |
| Llama 3.1 8B | `llama3.1-8b` |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` |
| GPT-4o-mini | `gpt-4o-mini` |
| Gemini 2.5 Flash | `gemini-2.5-flash` |

### `src/` — the autoresearch pipeline (the reproduction path)

The code the agent wrote and ran. This is what reproduces `paper.md`.

| File | What it does |
|---|---|
| `src/features.py` | Per-trace feature extraction: sentence split → MiniLM embed → cosine ε-graph (via `geom/`) → 5 spectral + 4 **exact** Ollivier–Ricci features (via `ricci-numpy/`) + surface features. Deliberately does **not** use `geom/ricci.py`. |
| `src/extract_all.py` | Runs `features.py` over `data/traces/`, applies the exclusion rule, writes `results/features.jsonl` + `results/01_extract_summary.json`. |
| `src/prontoqa_logic.py` | Forward-chaining engine for PrOntoQA: parses each question's ontology and derives the gold proof chain. The structural ground truth for D2 and the validated correctness gold for PrOntoQA. |
| `src/labels.py` | Re-derives correctness (using `prontoqa_logic.py` for PrOntoQA) and assigns the geometry-free 3-class failure-mode taxonomy (M1 abstain / M2 local slip / M3 misplan). |
| `src/build_labels.py` | Runs `labels.py` over the extracted features, writes `results/labels.jsonl` + `results/02_label_summary.json`. |
| `src/d1_common.py` | D1 shared machinery: data join, surface residualization, classifiers, grouped CV, group-aware paired bootstrap. |
| `src/run_d1.py` | **D1 experiments**: in-distribution + leave-one-dataset-out, both tasks, all feature conditions, permutation null, composition-shift check. Writes `results/10–15_*`. |
| `src/run_d2.py` | **D2 experiments**: argmin-curvature edge vs random / max-transition-energy / last-edge localization on PrOntoQA. Writes `results/20–22_*`. |
| `src/d2_reference_argmin.py` | D2 robustness: recomputes the curvature argmin under the pure-numpy *reference* solver to confirm the result is solver-invariant. Writes `results/23_*`. |
| `src/verify_curvature_backends.py` | Verifies the fast (numba) curvature path is bit-exact to the reference solver on real corpus graphs. Writes `results/00_curvature_backend_check.json`. |

### `geom/` — shared geometric-feature library

The original repo's feature utilities, reused by `src/`.

- `geom/graph.py` — cosine ε-proximity graph construction (ε = 0.3; adds sequential
  edges if the graph is disconnected).
- `geom/spectral.py` — the 5 Laplacian-spectrum features.
- `geom/ricci.py` — the original scipy-`linprog` Ollivier–Ricci implementation.
  **Not used by the autoresearch curvature path** — `ricci-numpy/` is used instead
  for exactness.

### `ricci-numpy/` — bundled exact curvature solver

A self-contained, dependency-free exact Ollivier–Ricci curvature library (embedded
network-simplex Wasserstein solver, bit-exact to a reference LP solver). The task
requires *all* curvature to come from here, not from Sinkhorn/POT approximations.
See [ricci-numpy/README.md](ricci-numpy/README.md). It has its own tests,
benchmarks, and standalone repo at <https://github.com/ArjunBalaji79/ricci-numpy>.

### `results/` — outputs of the autoresearch run

| File | Produced by | Contents |
|---|---|---|
| `00_survey.md` | SURVEY phase | repo map + smoke test + label-bug flag |
| `00_curvature_backend_check.json` | `verify_curvature_backends.py` | fast vs reference solver agreement |
| `00_prontoqa_label_audit.json` | label audit | evidence the released PrOntoQA labels are corrupted |
| `01_extract_summary.json` | `extract_all.py` | per-(model,benchmark) trace counts + drop reasons |
| `02_label_summary.json` | `build_labels.py` | correctness agreement + failure-mode distribution |
| `features.jsonl` | `extract_all.py` | the 9-dim signature + per-edge curvature + surface features per trace (~22 MB) |
| `labels.jsonl` | `build_labels.py` | correctness + failure mode per trace |
| `10_d1_indist.json` | `run_d1.py` | D1 in-distribution, both tasks, all conditions |
| `11_d1_lodo.json` | `run_d1.py` | D1 leave-one-dataset-out (the CRUX) |
| `12_d1_composition_shift.json` | `run_d1.py` | OOD drop vs mode-mixture divergence |
| `13_d1_permutation.json` | `run_d1.py` | label-permutation null |
| `14_d1_robustness_prontoqa_flag.json` | `run_d1.py` | D1 re-run with the released PrOntoQA flag (robustness) |
| `15_d1_gbt.json` | `run_d1.py` | D1 under HistGradientBoosting (robustness) |
| `20_d2_localization.json` | `run_d2.py` | D2 hit-rates: curvature vs baselines |
| `20_d2_per_trace.json` | `run_d2.py` | per-trace D2 selections |
| `21_d2_alignment_audit_sample.json` | `run_d2.py` | sentence-to-step alignment sample for manual audit |
| `21b_alignment_audit_manual.json` | manual audit | the audited alignment accuracy (~93%) |
| `22_d2_backend_argmin_check.json` | `run_d2.py` | argmin-edge agreement across solvers |
| `23_d2_reference_solver.json` | `d2_reference_argmin.py` | D2 headline recomputed under the reference solver |
| `99_selfcheck.md` | SELF-CHECK phase | claim-by-claim audit before writing the paper |
| `*.log` | each script | run logs (gitignored) |

### `code/` + `results_prior/` — the prior provided pipeline

The pre-autoresearch experiment set that shipped with the corpus (a different
framing: spectral-vs-Ricci orthogonality and 6-model fingerprinting). It is **not**
on the autoresearch reproduction path — `src/` does not import it — but it's kept
for provenance.

- `code/rerun_common.py` — the original `split_sentences` + MiniLM + 9-feature
  extraction pipeline.
- `code/six_model_rerun.py` — the master prior script (4 corpus configs).
- `code/stage1_lock_floor.py` — helper library for the rerun.
- `code/exp_d1_gate.py`, `code/exp_d2_fingerprint_gate.py`,
  `code/exp_reviewer_holdout_permutation.py` — the prior D1/D2/holdout experiments.
- `results_prior/` — their outputs (`SIX_MODEL_RERUN_*.json`,
  `rerun_phase1_features_*.jsonl`, `rerun_phase3_stage1_*.json`).

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 recommended. Pinned versions the run was verified under:

```
numpy==2.4.3  scipy==1.17.0  scikit-learn==1.8.0
sentence-transformers==5.2.2  networkx==3.6  xgboost==3.2.0
```

The MiniLM encoder (`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB) downloads
automatically on first feature extraction.

`ricci-numpy/` is vendored directly into this repo (its source lives here, not as a
submodule), so a plain `git clone` gets everything and `import ricci_numpy` works
with no extra steps. It is also maintained as a standalone package at
<https://github.com/ArjunBalaji79/ricci-numpy>.

## Reproduce

Run from the repo root, in order:

```bash
python src/extract_all.py          # → results/features.jsonl, 01_extract_summary.json
python src/build_labels.py         # → results/labels.jsonl, 02_label_summary.json
python src/run_d1.py               # → results/10–15_*
python src/run_d2.py               # → results/20–22_*
python src/d2_reference_argmin.py  # → results/23_* (D2 solver-invariance)
```

Every numeric claim in `paper.md` traces to one of these `results/` files.

## Reproducibility notes

- `seed = 0` everywhere seeds matter; all CV is grouped so a problem never spans
  train/test folds.
- All curvature is **exact** (`ricci-numpy`); the fast path is bit-exact to the
  reference solver to ~1e-15 (`results/00_curvature_backend_check.json`).
- PrOntoQA correctness is re-derived from the forward-chaining engine, not the
  corrupted released flag (see `AUDIT_PRONTOQA.md`). GSM8K/FOLIO use the released
  flags.
