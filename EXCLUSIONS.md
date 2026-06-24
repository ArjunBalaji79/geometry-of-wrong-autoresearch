# EXCLUSIONS — 15-trace truncation exclusion

**Applies to:** the 6-model geometric analyses (orthogonality table, D1
modality split, D2 model-identity fingerprint).
**Nature:** pipeline-level only. No file in `data/traces/` is deleted
or modified; the rule applies at load time.

## Rule

A trace is excluded iff it is **generation-truncated**, detected by:

- **finish-reason where logged** — the trace carries a truncation
  finish-reason (`MAX_TOKENS`, `length`, or equivalent) in its
  `_finish_reason` field; or
- **spot-confirmed mid-reasoning truncation where not logged** — the
  trace has no finish-reason field, but manual inspection confirms it
  ends mid-sentence / mid-step with no valid final answer.

### Strict variant rejected

The strict rule "exclude iff a truncation finish-reason is present" was
rejected as **logging-blind, not model-blind**. Four of the six models
(`claude_sonnet`, `gpt_oss_120b`, `llama_3_1_8b`, `mistral_7b`) were
generated without a `_finish_reason` field. Under the strict rule,
truncated `gemini_2_5_flash` traces would be excluded while
equally-truncated `gpt_oss_120b` and `llama_3_1_8b` traces would be
kept, solely because of a logging difference. The uniform rule above
asks the same "is this trace truncated?" question of every model, using
the best signal available for each.

## 6-model allowlist

Released corpus = exactly these 6 models:

  **claude_sonnet, gpt_oss_120b, llama_3_1_8b, mistral_7b, gpt_4o_mini, gemini_2_5_flash**

## Excluded traces (15 total)

| Model | Benchmark | problem_idx | Signal |
|---|---|---|---|
| gemini_2_5_flash | folio | 134, 142, 158, 173, 187 | `MAX_TOKENS` finish-reason |
| gemini_2_5_flash | prontoqa | 15, 60, 110, 120 | `MAX_TOKENS` finish-reason |
| gpt_oss_120b | folio | 99 | spot-confirmed truncation |
| gpt_oss_120b | prontoqa | 168 | spot-confirmed truncation |
| llama_3_1_8b | prontoqa | 59, 78, 101, 198 | spot-confirmed truncation |

### Spot-check evidence (the 6 non-logged truncations)

- `gpt_oss_120b/folio/99` — ends `"...2. Derive consequences from the chain of implications:"`; `final_answer = "2."`.
- `gpt_oss_120b/prontoqa/168` — ends `"...4. Rule:"`; `final_answer = "4."`.
- `llama_3_1_8b/prontoqa/59` — repetition loop, ends `"...Rex is a grimpus, a dumpus,"`; `final_answer = "4"`.
- `llama_3_1_8b/prontoqa/78` — repetition loop, ends `"...Therefore,"`; `final_answer = "6"`.
- `llama_3_1_8b/prontoqa/101` — ends `"...everything that is a numpus, a shumpus,"`; `final_answer = "4"`.
- `llama_3_1_8b/prontoqa/198` — ends `"...it follows that Fae is a rompus. Therefore,"`; `final_answer = "1"`.

All 6 are logic-domain, all marked `correct = False`, all carry a
garbage fragment as `final_answer`.

## Empty / no-output traces — disclosure only, not excluded by this rule

Empty traces produce no chain-of-thought graph, so they are already
outside the geometric feature analysis (no graph -> no 9 features).
They are not generation-truncated; they are total generation failures
(blank output or no extractable answer).

| Model | Benchmark | empty / no-output |
|---|---|---|
| gpt_oss_120b | gsm8k | 5 |
| gpt_oss_120b | folio | 25 |
| gpt_oss_120b | prontoqa | 42 |
| **gpt_oss_120b** | **total** | **72 / 600 (12%)** |
| mistral_7b | folio | 5 |

## Reporting configurations

D1 and D2 are reported at four configurations to isolate the effect of
the exclusion versus the effect of the additional models:

1. **4 models, truncations included** — original 4-model baseline.
2. **4 models, truncations excluded** — same 4 models, exclusion applied.
3. **6 models, truncations excluded** — clean target corpus.
4. **6 models, truncations included** — dual-report.

Comparing (1) -> (2) isolates the effect of the exclusion alone;
comparing (2) -> (3) isolates the effect of adding `gpt_4o_mini` and
`gemini_2_5_flash`.
