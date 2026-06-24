# Geometry of Reasoning Traces — Release

Code + data release for the paper *On the Geometry of Reasoning Traces:
Spectral and Ollivier-Ricci Signatures of Chain-of-Thought*.

This release reproduces every headline number in the paper from the
released chain-of-thought traces.

## What this repository contains

```
.
├── README.md
├── requirements.txt
├── EXCLUSIONS.md                                # 15 truncated trace IDs + the model-blind rule
├── SIX_MODEL_RERUN_20260522_194904.json         # master output (orthogonality, D1, D2, all configs)
├── geom/                                        # 9-feature pipeline
│   ├── graph.py                                 #   cosine epsilon-proximity graph
│   ├── spectral.py                              #   5 spectral features
│   └── ricci.py                                 #   4 Ollivier-Ricci features (130-line implementation)
├── code/
│   ├── rerun_common.py                          # split_sentences + MiniLM + 9-feature extraction
│   ├── stage1_lock_floor.py                     # helpers reused by SMR
│   ├── exp_d1_gate.py                           # D1: modality split (S vs R on numeric / domain / model)
│   ├── exp_d2_fingerprint_gate.py               # D2: 6-way model identity (length-only vs geometry-9)
│   ├── six_model_rerun.py                       # the master script (4m_incl / 4m_excl / 6m_excl / 6m_incl)
│   └── exp_reviewer_holdout_permutation.py      # held-out confirmatory split + permutation p-values
├── data/
│   └── traces/                                  # 18 raw CoT trace JSONs (6 models x 3 domains)
└── results/
    ├── rerun_phase1_features_20260518_004610.jsonl   # per-trace 9-feature outputs (4 original models)
    └── rerun_phase3_stage1_20260518_013157.json      # anchor used by exp_d1_gate.py Step-1 reconstruction check
```

## Data

`data/traces/` contains 18 files named `<benchmark>_<model>.json` — the
6 paper models on all 3 benchmarks. Each record has fields
`question, cot_trace, final_answer, model, temperature, backend,
ground_truth, correct, problem_idx`. All traces were generated at
`temperature = 0.0` (greedy decoding).

### Models (verbatim `model` strings as they appear in each trace record)

| Paper label | `model` string |
|---|---|
| Claude Sonnet 4 | `claude-sonnet-4-20250514` |
| gpt-oss 120B | `gpt-oss-120b` |
| Llama 3.1 8B | `llama3.1-8b` |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` |
| GPT-4o-mini | `gpt-4o-mini` |
| Gemini 2.5 Flash | `gemini-2.5-flash` |

Inference backend, decoding hyperparameters, and prompt templates per
model are listed in the paper's implementation-details appendix.

### Benchmarks

GSM8K, FOLIO, and PrOntoQA. 200 problems each, identical problem indices
across all models. The 6-model x 3-domain corpus is therefore 3,600
traces before filtering, 3,493 after the truncation exclusion +
3-sentence filter (see `EXCLUSIONS.md`).

### Generation pipeline (upstream; not bundled)

The trace JSONs in `data/traces/` are the **inputs** to every analysis
script in this release. The pipeline that produced them is not
included in this release: it requires multiple paid API keys plus a
GPU inference account, and the released traces are deterministic at
temperature 0.0. System prompts, user-prompt templates, decoding
hyperparameters, and answer-extraction regexes are documented in the
paper's implementation-details appendix.

### Generation dates

Trace generation: February 2026 (4 base models) and May 2026 (2
additional models). The feature extraction and the analyses in
`results/` were performed 2026-05-18 through 2026-05-25.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 recommended. The pinned versions are the ones the release
was verified under:

```
numpy==2.4.3
scipy==1.17.0
scikit-learn==1.8.0
sentence-transformers==5.2.2
networkx==3.6
xgboost==3.2.0
```

The MiniLM sentence encoder
(`sentence-transformers/all-MiniLM-L6-v2`) downloads automatically on
first feature extraction (~80MB).

## Reproducing the paper's headline numbers

All scripts run with `cwd = repo root`.

### 1. Orthogonality table (Table 1) + D1 modality split + D2 fingerprint

```bash
python code/six_model_rerun.py
```

Writes `SIX_MODEL_RERUN_<timestamp>.json` at the repo root with all four
configs (4m_incl, 4m_excl, 6m_excl, 6m_incl). The pre-computed master
output `SIX_MODEL_RERUN_20260522_194904.json` is included so the
analysis scripts can be inspected without a full re-run. Re-running
this script re-extracts features for gpt-4o-mini and gemini-2.5-flash;
the 4 other models read from `results/rerun_phase1_features_*.jsonl`.

Expected anchors (6m_excl, incorrect-only):
- n_S = 345, n_R = 265
- numeric per-sentence-rate Cliff's delta = +0.2892 (p = 8.97e-10)
- length-residualized delta = +0.3368 (p = 9.65e-13)
- domain chi^2(2) = 50.069 (p = 1.34e-11)
- model chi^2(5) = 51.122 (p = 8.17e-10)
- D2 6-way model identity (GBT): A length-only = 0.349, B geometry-9 = 0.439,
  B - A = +0.090, 95% CI [0.072, 0.108]
- D2 shape-only - length-only = +0.070, CI [0.052, 0.088]

### 2. D1 alone (faster; uses the locked 4-model features)

```bash
python code/exp_d1_gate.py
```

### 3. Held-out confirmatory split + permutation p-values

```bash
python code/exp_reviewer_holdout_permutation.py
```

Per-condition 50/50 stratified split, S/R built on half A and tested on
half B; 50 random splits with A/B swap; and 100,000-permutation exact
p-values on the full incorrect-only S/R pool. Writes
`results/exp_reviewer_holdout_permutation_<timestamp>.json` and
`REVIEWER_RESPONSE_NUMBERS.md` at the repo root.

Expected:
- Full-data anchor (must match): n_S = 345, n_R = 265
- Held-out (50 splits, main direction): median rate delta = +0.27
  (5th-95th [+0.13, +0.39]); 50/50 splits with delta > 0; 49/50 splits
  preserve the domain direction (S more GSM8K, R more PrOntoQA)
- Permutation: 0 of 100,000 relabelings extreme; permutation p <= 1e-5
  for domain, model, and the numeric-rate delta

## Notes on reproducibility

- All scripts use `seed = 0` where seeds matter.
- Feature extraction is deterministic given the MiniLM model weights and
  the cosine-similarity threshold eps = 0.3.
- The 4-model feature outputs in `results/rerun_phase1_features_*.jsonl`
  were produced by the same `geom/` pipeline that the scripts in `code/`
  re-run end-to-end. Re-extracting from `data/traces/` reproduces them
  bit-exactly (validated by the anchor check in
  `code/exp_d1_gate.py`).

## Truncation exclusion

The 15-trace uniform model-blind exclusion rule and the list of trace
IDs are in `EXCLUSIONS.md`. The rule excludes traces that are
generation-truncated, detected by truncation finish-reason where logged
or spot-confirmed mid-step truncation where the finish-reason was not
recorded.

## Citing

Citation block will be added on de-anonymization.
