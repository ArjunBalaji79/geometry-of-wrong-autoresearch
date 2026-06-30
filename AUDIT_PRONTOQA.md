# PrOntoQA label-bug audit (read-only diagnosis)

**Question.** A shared CoT dump has corrupted PrOntoQA labels: `ground_truth` is
the constant `"True"` for all 1,200 PrOntoQA traces, and `correct` is not a real
grade but equals `(final_answer == "True")`. Does **this repository's** analysis
inherit that bug, or does it grade PrOntoQA through an independent path?

**Verdict (one line).** **NOT inherited** — this repo grades PrOntoQA
independently from the `question` via a forward-chaining engine and never reads
the dump's `correct`/`ground_truth` for PrOntoQA (engine-vs-dump-flag disagreement
17.9%, or 11.0% after correcting the engine's own bug; far from the near-zero that
inheritance would produce). **Caveat:** the independent oracle is itself imperfect
(~8% of PrOntoQA gold labels are wrong due to a parser bug), so the repo's
PrOntoQA labels are *independent but noisy*, not verified gold.

---

## 1. Location of grading

The single code path that decides PrOntoQA correctness for everything in
`results/` is `primary_correct()` in **`src/labels.py:177-185`**:

```python
def primary_correct(rec):
    if rec["benchmark"] == "prontoqa":
        gold, _ = gold_answer(rec["question"])          # <-- engine, from question text
        pred = model_answer_bool(rec.get("final_answer"))
        if gold is None or pred is None:
            return False if gold is not None else None
        return pred == gold                              # <-- compare to engine gold
    return bool(rec["correct"])                          # gsm8k/folio: dump flag (valid there)
```

`gold_answer()` is the forward-chaining engine in
**`src/prontoqa_logic.py:139-162`**: it parses the ontology out of the `question`,
forward-chains the entity's properties under closed-world assumption, and answers
the True/False/negation/disjunction query. It explicitly does **not** read the
JSON `ground_truth` (docstring line 7: "...not ... the (unreliable) `ground_truth`
JSON field").

Consumers of this label (grep over `src/`):
`build_labels.py:28` (writes `primary_correct` into `results/labels.jsonl`) →
`d1_common.py:37` (loads it) → `run_d1.py:47,57,80,126,137` (binary + mode tasks)
and `run_d2.py:48` / `d2_reference_argmin.py:36` (D2 subset).
The dump's `correct`/`ground_truth` are carried into `results/features.jsonl`
(`extract_all.py:74-76`) only as audit columns; for PrOntoQA they are used as a
*result* in exactly one place, the deliberately-labeled robustness check
`run_d1.py:187-200` → `results/14_d1_robustness_prontoqa_flag.json`.

The prior pipeline in `code/` (`six_model_rerun.py`, `exp_*`) is **not** used to
produce `results/`; it does not grade PrOntoQA correctness at all (it does an S/R
"modality" split, not correct/incorrect).

## 2. The oracle

Type **(b): recomputes correctness from the `question` via an in-repo
forward-chaining engine** (`src/prontoqa_logic.py`). Not (a) dump flag, not (c)
upstream dataset files (the original PrOntoQA gold is **not present** in this
repo), not (d).

Trace source: this repo loads traces from **`data/traces/*.json`** — i.e. the
**same corrupted dump** (see §3A). So the corrupted *fields* are physically
present in this repo's inputs; the analysis simply declines to use them for
PrOntoQA correctness.

## 3. Decisive test

All numbers below are reproducible read-only from `data/traces/` + `src/`.

**(A) The dump corruption is present in this repo.** Over all 1,200 PrOntoQA
traces, `ground_truth` takes exactly one value: `{"True": 1200}`. A constant gold
column is impossible for a True/False task.

**(B) The dump `correct` flag is a broken grader (engine-independent proof).**
`correct == (final_answer == "True")` for **92.7% (1112/1200)** of PrOntoQA
traces — i.e. the flag is essentially "did the model say True", not a real grade.
The engine's own gold is genuinely **mixed** (≈90% True / 10% False per model;
e.g. claude 178/20), confirming the constant column is wrong.

**(C) The repo's engine grade disagrees with the dump flag → independent.**
On the 975 traces with a parseable model answer:

|                     | dump flag = True | dump flag = False |
|---------------------|------------------|-------------------|
| **engine = correct**   | 632              | **115**           |
| **engine = incorrect** | 60               | 168               |

Overall disagreement **17.9% (175/975)**. The 115 cell = traces the dump marks
*incorrect* that are logically *correct* — the corruption. Per model
(engine-correct vs dump-flag disagreement): claude 30.8%, llama 26.7%, gpt-oss
41.0%, gpt-4o-mini 10.0%, mistral 9.4%, gemini 10.2%. This is **nowhere near
zero**, so the repo's labels are not the dump flag.

**(D) After correcting the engine's own bug (see §4), disagreement is 11.0%**
(107/975) — still clearly independent. The ~7pp difference between (C) and (D) is
the engine's own error, not dump corruption.

## 4. Skepticism about the oracle — TWO BUGS FOUND

Spot-checking FALSE-gold items (step 4) exposed that the oracle is **not** itself
correct:

- **Bug 1 (material): the first context rule is silently dropped.**
  `parse_context` (src/prontoqa_logic.py) does not strip the `"Context:"` prefix,
  so for sentence 0 the anchored `re.match("everything that is ...")` fails on
  `"Context: Everything that is ..."` and the rule is lost. Example
  (claude/prontoqa, "Wren is a zumpus"): the rule *"Everything that is a wumpus, a
  lorpus, and a tumpus is a numpus, a zumpus, and a rompus"* is rule 0 and is
  dropped; Wren is given as wumpus+lorpus+tumpus, so zumpus **is** entailed, but
  the engine returns gold=False. Verified by reparse: parsed-rules list omits that
  rule entirely.
- **Bug 2 (latent): conjunctive antecedents are fired disjunctively.**
  `forward_chain` fires a rule if *any* antecedent prop is present
  (`hit = ante & derived`), so *"wumpus **and** lorpus **and** tumpus → ..."* is
  treated as *"wumpus **or** ..."*. This over-derives in principle.

**Quantification** (corrected engine in scratchpad; strip `"Context:"` + respect
and/or antecedents): **8.0% of gold labels flip (96/1200; exactly 16/200 per
model — identical across models, confirming gold is question-determined).**
Attribution: **Bug 1 accounts for all 8.0%**; Bug 2 flips **0/200** on this
corpus (PrOntoQA proof chains supply the full conjunctive antecedent, so any/all
firing coincide) — it is latent, not active here.

The original upstream PrOntoQA gold is not in the repo, so the oracle cannot be
cross-checked against it; the corrected engine is internally consistent on the
manual spot-checks (FALSE-gold derivations, negation and disjunction queries, and
counterfactual predicate queries all resolve correctly).

**Net:** the oracle is materially better than the dump flag (which is just
`answer=="True"`) but carries ~8% error. The repo's `results/00_prontoqa_label_audit.json`
and `results/00_survey.md` attribute the full ~15-18% engine-vs-flag gap to dump
corruption; ~7pp of it is in fact this engine bug. The qualitative corruption
finding still holds (proof (B) is engine-independent).

## 5. Blast radius

Everything PrOntoQA-correctness-dependent (would move if labels were corrected):

- **D2 — entirely PrOntoQA** (subset = engine-incorrect M3 traces): the ~8%
  oracle error and any relabeling shift subset membership and the broken-step
  set. Files: `results/20_d2_localization.json`, `20_d2_per_trace.json`,
  `21_*`, `22_*`, `23_*`. Headline D2 numbers (curvature hit 0.188 vs random
  0.364; BAR tie) sit on this subset.
- **D1 PrOntoQA portions**: the PrOntoQA rows/folds of
  `results/10_d1_indist.json`, `11_d1_lodo.json` (prontoqa LODO binary AUC 0.427
  and mode F1 0.345/0.306), `12_d1_composition_shift.json` (PrOntoQA mode mixture
  + OOD drop 0.129), `13_d1_permutation.json`, `15_d1_gbt.json`.
- **Label tables**: `results/02_label_summary.json` (PrOntoQA correct counts
  745/390 and mode dist 162/147/81), `labels.jsonl`, and the corruption-magnitude
  numbers in `results/00_prontoqa_label_audit.json` / `00_survey.md`.
- **Not affected**: GSM8K and FOLIO correctness (separate sources, valid).

**Risk assessment.** The D1 and D2 headlines are *null* results (geometry does not
beat the surface baseline; curvature localizes below chance). Label noise of ~8%
reinforces nulls rather than creating them, and the D2 below-random gap (−0.18 to
−0.27) is far larger than any 8%-subset perturbation. So no headline conclusion
flips. What is overstated is the *corruption magnitude* (by ~7pp) and the precise
PrOntoQA D1/D2 point values carry ~8% oracle noise.

## 6. Verdict

**This repository does NOT inherit the constant-`ground_truth` bug.** PrOntoQA
correctness is decided solely by `src/labels.py:primary_correct → gold_answer`,
which recomputes the answer from the `question` by forward chaining and never
consults the dump's `correct`/`ground_truth` (those are used only for GSM8K/FOLIO,
where they are valid, and for one explicitly-labeled robustness file,
`results/14_d1_robustness_prontoqa_flag.json`). The independence is confirmed
three ways: the code path; a 17.9% engine-vs-dump-flag disagreement (11.0% after
fixing the engine's own bug) — not the near-zero that inheritance implies; and an
engine-independent test showing the dump flag is itself ≈`(final_answer=="True")`
at 92.7%, i.e. the broken grader the repo refuses to use. **However**, the
independent oracle has a real parser bug (it drops each problem's first rule
because it does not strip the `"Context:"` prefix), which flips ~8% of PrOntoQA
gold labels and means roughly 7 of the ~18 percentage-point disagreement the repo
attributes to dump corruption is actually the oracle's own error. At-risk numbers
are all of D2 (PrOntoQA-only) and the PrOntoQA portions of D1, plus the
corruption-magnitude claims; because those headlines are null results, the oracle
noise does not manufacture a positive finding, but the exact PrOntoQA point values
and the stated corruption rate should be treated as ±~8% / overstated-by-~7pp
respectively. Recommended (out of scope for this read-only audit): strip the
`"Context:"` prefix in `parse_context`, add `and`/`or` antecedent handling, and
re-run `build_labels.py` + D1/D2.

---
*Method: read-only. All diagnostics ran against `data/traces/` and `src/`; the
corrected-engine comparison ran in the session scratchpad and modified no repo
file. No file other than this report was changed.*
