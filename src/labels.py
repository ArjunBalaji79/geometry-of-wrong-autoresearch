"""
Correctness re-derivation and failure-mode taxonomy.

WHY re-derive correctness:
  SURVEY found the released `correct` field is corrupted for PrOntoQA (the JSON
  `ground_truth` is the constant "True"; ~92% of claude's PrOntoQA `correct=False`
  traces are in fact logically correct -- see results/00_survey.md). GSM8K and
  FOLIO carry real `ground_truth`. We therefore recompute a VALIDATED correctness
  label per domain and use it as primary, reporting agreement with the released
  flag.

Correctness (model answer vs reliable gold):
  - GSM8K   : numeric match of extracted answer to ground_truth (real numbers).
  - FOLIO   : 3-way {True,False,Unknown} match to ground_truth (real labels).
  - PrOntoQA: match to the forward-chaining engine gold (src/prontoqa_logic.py),
              since the JSON ground_truth is unusable.

FAILURE-MODE TAXONOMY (defined for INCORRECT traces only).
  The taxonomy is MECHANISM-based and *domain-general*, with a uniform 3-class
  structure instantiated per domain via OBJECTIVE signals (ground truth + answer
  parsing + the PrOntoQA engine). It NEVER references the 9 geometric features
  (spectral / Ollivier-Ricci) that the classifier uses -- this is how we avoid
  circularity. The only non-answer signals used are the benchmark gold and (for
  the abstain class) answer parseability; trace length / repetition are NOT used
  to define modes (they are the residualization control, kept separate).

  M1  ABSTAIN / UNDER-COMMIT : the model fails to commit to a definite in-space
        answer when a definite one exists.
        GSM8K: no parseable number.   PrOntoQA: no parseable True/False.
        FOLIO: predicts "Unknown" while gold is True/False.
  M2  LOCAL SLIP (MISCOMPUTE): commits to a definite wrong answer, but the
        approach is engaged / internally valid.
        GSM8K: wrong number within relative error <= TAU_REL (ballpark slip).
        FOLIO: True<->False flip (both definite).
        PrOntoQA: wrong answer but every property the trace asserts about the
                  entity is valid under the gold ontology (chain incomplete /
                  final-mapping error), engine-checked.
  M3  GLOBAL MISPLAN / HALLUCINATION: commits but is fundamentally off.
        GSM8K: wrong number with relative error > TAU_REL (gross).
        FOLIO: over-commit -- predicts a definite True/False while gold is Unknown
               (hallucinated entailment).
        PrOntoQA: the trace asserts at least one property NOT derivable under the
                  gold ontology (invalid inference / hallucinated rule).

TAU_REL is pre-registered at 0.5 and its sensitivity is reported.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prontoqa_logic import gold_answer, model_answer_bool

TAU_REL = 0.5
MODE_NAMES = {1: "M1_abstain", 2: "M2_local_slip", 3: "M3_misplan"}


# --- answer extraction ------------------------------------------------------
def gsm8k_number(s):
    """Extract the model's numeric answer (prefer explicit answer markers)."""
    t = str(s).replace(",", "")
    for pat in [r"####\s*(-?\d+\.?\d*)",
                r"answer is\s*\$?\s*(-?\d+\.?\d*)",
                r"=\s*\$?\s*(-?\d+\.?\d*)\s*$"]:
        m = re.search(pat, t, flags=re.I | re.M)
        if m:
            return m.group(1)
    nums = re.findall(r"-?\d+\.?\d*", t)
    return nums[-1] if nums else None


def folio_answer(s):
    """Map a FOLIO final answer to {True,False,Unknown} or None."""
    t = str(s).strip().lower()
    head = re.sub(r"^[^a-z]*", "", t)
    if head.startswith("unknown") or "uncertain" in head[:20] or "cannot" in head[:20]:
        return "Unknown"
    if head.startswith("true"):
        return "True"
    if head.startswith("false"):
        return "False"
    # fallback by unique mention
    has = {k: (k.lower() in t) for k in ["true", "false", "unknown"]}
    present = [k for k, v in has.items() if v]
    if len(present) == 1:
        return present[0].capitalize()
    return None


# --- PrOntoQA asserted-property extraction (for M2 vs M3) --------------------
def prontoqa_asserted_props(cot_text, entity):
    """Positive kind-properties the trace asserts about the entity."""
    props = set()
    if not entity:
        return props
    for m in re.finditer(re.escape(entity) + r"\s+(?:is|must be|is also)\s+([^.\n]+)",
                         cot_text, flags=re.I):
        frag = m.group(1)
        if re.search(r"\bnot\b", frag):
            continue
        for pm in re.finditer(r"[a-z]+pus", frag.lower()):
            w = pm.group(0)
            props.add(w[:-2] if w.endswith("puses") else w)
    return props


# --- correctness source policy ----------------------------------------------
# SURVEY validation (results/02_label_summary.json): the released `correct` flag
# matches an independent recompute at 100% on FOLIO and ~97-100% on GSM8K for the
# models whose answers parse cleanly (gaps are answer-parser noise, not label
# corruption), but is badly corrupted on PrOntoQA (constant ground_truth field).
# Policy: trust the released flag for GSM8K/FOLIO; use the engine for PrOntoQA.
def primary_correct(rec):
    """The correctness label used as primary throughout D1."""
    if rec["benchmark"] == "prontoqa":
        gold, _ = gold_answer(rec["question"])
        pred = model_answer_bool(rec.get("final_answer"))
        if gold is None or pred is None:
            return False if gold is not None else None  # unparseable answer = incorrect
        return pred == gold
    return bool(rec["correct"])  # gsm8k / folio: released flag (validated reliable)


def failure_mode(rec):
    """
    Assign a failure mode to a trace KNOWN to be incorrect (per primary_correct).
    Always returns 1/2/3 (never None) so every incorrect trace is typed.
    Uses only ground truth + answer parsing + the PrOntoQA engine -- no geometry.
    """
    bench = rec["benchmark"]
    fa = rec.get("final_answer")
    if bench == "gsm8k":
        gold = gsm8k_number(rec.get("ground_truth"))
        pred = gsm8k_number(fa)
        if pred is None:
            return 1
        if gold is None:
            return 2
        try:
            rel = abs(float(gold) - float(pred)) / max(abs(float(gold)), 1.0)
        except ValueError:
            return 2
        return 2 if rel <= TAU_REL else 3
    if bench == "folio":
        gold = str(rec.get("ground_truth")).strip().capitalize()
        pred = folio_answer(fa)
        if pred is None:
            return 1
        if pred == "Unknown" and gold in ("True", "False"):
            return 1
        if gold == "Unknown" and pred in ("True", "False"):
            return 3
        return 2  # True<->False flip (or pred==gold edge: treat as local slip)
    if bench == "prontoqa":
        gold, info = gold_answer(rec["question"])
        pred = model_answer_bool(fa)
        if pred is None:
            return 1
        cot = " ".join(rec.get("sentences", []))
        asserted = prontoqa_asserted_props(cot, info.get("entity"))
        invalid = asserted - set(info.get("derived", set()))
        return 3 if invalid else 2
    return 2


# --- legacy combined recompute (used for the SURVEY label-corruption audit) --
def correctness_and_mode(rec):
    """
    rec is a features.jsonl object (has benchmark, final_answer, ground_truth,
    question, sentences/cot via 'sentences', correct[released]).
    Returns dict: is_correct(bool|None), mode(int|None), parse_ok(bool), extra.
    is_correct None / mode None when unrecoverable.
    """
    bench = rec["benchmark"]
    fa = rec.get("final_answer")
    gt = rec.get("ground_truth")

    if bench == "gsm8k":
        gold = gsm8k_number(gt)
        pred = gsm8k_number(fa)
        if gold is None:
            return {"is_correct": None, "mode": None, "parse_ok": False}
        if pred is None:
            return {"is_correct": False, "mode": 1, "parse_ok": False}
        try:
            g, p = float(gold), float(pred)
            corr = abs(g - p) < 1e-6
            rel = abs(g - p) / max(abs(g), 1.0)
        except ValueError:
            corr = (gold == pred); rel = 0.0 if corr else 1e9
        if corr:
            return {"is_correct": True, "mode": None, "parse_ok": True}
        mode = 2 if rel <= TAU_REL else 3
        return {"is_correct": False, "mode": mode, "parse_ok": True, "rel_err": rel}

    if bench == "folio":
        gold = str(gt).strip().capitalize() if gt is not None else None
        pred = folio_answer(fa)
        if gold not in ("True", "False", "Unknown"):
            return {"is_correct": None, "mode": None, "parse_ok": False}
        if pred is None:
            return {"is_correct": False, "mode": 1, "parse_ok": False}
        if pred == gold:
            return {"is_correct": True, "mode": None, "parse_ok": True}
        if pred == "Unknown":            # under-commit
            mode = 1
        elif gold == "Unknown":          # over-commit (hallucinated entailment)
            mode = 3
        else:                             # True<->False flip
            mode = 2
        return {"is_correct": False, "mode": mode, "parse_ok": True}

    if bench == "prontoqa":
        gold, info = gold_answer(rec["question"])
        pred = model_answer_bool(fa)
        if gold is None:
            return {"is_correct": None, "mode": None, "parse_ok": False}
        if pred is None:
            return {"is_correct": False, "mode": 1, "parse_ok": False}
        if pred == gold:
            return {"is_correct": True, "mode": None, "parse_ok": True}
        # committed wrong: M2 vs M3 via engine
        cot = " ".join(rec.get("sentences", []))
        asserted = prontoqa_asserted_props(cot, info.get("entity"))
        derived = info.get("derived", set())
        invalid = asserted - set(derived)
        mode = 3 if invalid else 2
        return {"is_correct": False, "mode": mode, "parse_ok": True,
                "n_invalid_props": len(invalid)}

    return {"is_correct": None, "mode": None, "parse_ok": False}
