"""
PrOntoQA structural ground truth: parse the synthetic ontology in each question,
forward-chain the entity's properties, and recover the gold proof chain.

This is the STRUCTURAL ANCHOR for D2 (and a validity check for PrOntoQA labels).
It depends only on the question text and closed-world forward chaining -- NOT on
any geometric feature, embedding, or the (unreliable) `ground_truth` JSON field.

Rule grammar observed in data/traces/prontoqa_*.json:
  - "Wumpuses are grimpuses."                     plural subj -> consequent(s)
  - "Lempuses are lorpuses, impuses, and rompuses."  conjunction of consequents
  - "Each rompus is a sterpus." / "Every lorpus is a tumpus."  singular universal
  - "Everything that is an impus or a vumpus or a rompus is a wumpus, a lorpus, and a yumpus."
        disjunctive antecedent -> conjunctive consequent
  - facts: "Polly is a rompus.",  "Sally is a gorpus, a yumpus, and a shumpus.",
           "Sally is a zumpus and Sally is not a sterpus and Sally is a tumpus."

Queries: "Polly is a lorpus.",  "Stella is a gorpus, a zumpus, or an impus.",
         "Sally is not a jompus."   (answer True/False; closed-world for negation)
"""
from __future__ import annotations
import re

PROP = r"[a-z]+pus"  # all PrOntoQA kind names end in 'pus'


def singularize(w):
    w = w.strip().lower()
    if w.endswith("puses"):
        return w[:-2]   # wumpuses -> wumpus
    return w


def _props(text):
    """All kind-tokens (singularized) appearing in a fragment."""
    return [singularize(m.group(0)) for m in re.finditer(PROP + r"e?s?", text)]


def parse_context(context):
    """
    Return (rules, facts_pos, facts_neg, entity).
      rules: list of (antecedent_set, consequent_set) meaning
             (any prop in antecedent) -> (all props in consequent).
             Plain universals "X are Y" become ({X}, {Y...}).
      facts_pos / facts_neg: sets of props asserted (not) true of the entity.
      entity: the subject name (Polly/Stella/Sally/...).
    """
    rules = []
    facts_pos, facts_neg = set(), set()
    entity = None
    # context sentences end with '.'
    sents = [s.strip() for s in re.split(r"(?<=\.)\s+", context.strip()) if s.strip()]
    for s in sents:
        sl = s.rstrip(".")
        low = sl.lower()

        # disjunctive-antecedent rule
        m = re.match(r"everything that is (.+?) is (.+)$", low)
        if m:
            ante = set(_props(m.group(1)))
            cons = set(_props(m.group(2)))
            if ante and cons:
                rules.append((ante, cons))
            continue

        # singular universal: "Every/Each X is a Y..."
        m = re.match(r"(?:every|each) (.+?) is (.+)$", low)
        if m:
            ante = set(_props(m.group(1)))
            cons = set(_props(m.group(2)))
            if ante and cons:
                rules.append((ante, cons))
            continue

        # entity fact: "<Name> is ..."  (Name is a capitalized non-kind token)
        m = re.match(r"([A-Z][a-z]+) (is|are) (.+)$", sl)
        if m and not re.match(r"(everything|every|each)$", m.group(1).lower()):
            name = m.group(1)
            body = m.group(3)
            # plural-subject rule disguised? handled below; this is a singular entity fact
            # split on 'and'/','; track negation per clause
            # normalize "X is not a P and X is a Q" by splitting on ' and '
            clauses = re.split(r"\s+and\s+", body)
            for cl in clauses:
                cl2 = re.sub(r"^%s\s+(is|are)\s+" % re.escape(name), "", cl.strip(), flags=re.I)
                neg = bool(re.search(r"\bnot\b", cl2))
                for p in _props(cl2):
                    (facts_neg if neg else facts_pos).add(p)
            if facts_pos or facts_neg:
                entity = name
            continue

        # plural-subject universal: "Wumpuses are grimpuses, ..."
        m = re.match(r"(.+?) are (.+)$", low)
        if m:
            ante = set(_props(m.group(1)))
            cons = set(_props(m.group(2)))
            if ante and cons:
                rules.append((ante, cons))
            continue

    return rules, facts_pos, facts_neg, entity


def forward_chain(rules, facts_pos):
    """
    Closed-world forward chaining. Returns (derived_set, provenance) where
    provenance[p] = (rule_index, triggering_prop) for each newly derived p
    (facts have provenance None). Deterministic order over rules.
    """
    derived = set(facts_pos)
    prov = {p: None for p in facts_pos}
    changed = True
    while changed:
        changed = False
        for ri, (ante, cons) in enumerate(rules):
            hit = ante & derived
            if hit:
                trigger = sorted(hit)[0]
                for c in cons:
                    if c not in derived:
                        derived.add(c)
                        prov[c] = (ri, trigger)
                        changed = True
    return derived, prov


def parse_query(question):
    """Return (query_props, is_disjunction, is_negation) from the 'Question:' line."""
    m = re.search(r"Question:\s*(.+?)\s*Is the Question", question, flags=re.S)
    q = m.group(1).strip() if m else ""
    low = q.lower()
    is_neg = bool(re.search(r"\bnot\b", low))
    is_disj = " or " in low
    props = _props(low)
    return props, is_disj, is_neg


def gold_answer(question):
    """
    Compute the gold True/False answer by forward chaining + closed-world query.
    Returns (answer_bool, info) where info carries derived set, query props, etc.
    answer is None if the query could not be parsed.
    """
    ctx = question.split("Question:")[0]
    rules, fp, fn, entity = parse_context(ctx)
    derived, prov = forward_chain(rules, fp)
    qprops, is_disj, is_neg = parse_query(question)
    if not qprops:
        return None, {"reason": "no_query_props"}
    if is_disj:
        base = any(p in derived for p in qprops)
    else:
        base = all(p in derived for p in qprops)
    ans = (not base) if is_neg else base
    return ans, {
        "entity": entity, "derived": derived, "prov": prov,
        "query_props": qprops, "is_disj": is_disj, "is_neg": is_neg,
        "rules": rules, "facts_pos": fp, "facts_neg": fn,
    }


# --- model-answer extraction (for validation) -------------------------------
def model_answer_bool(final_answer):
    """Extract the model's True/False from final_answer. None if unclear."""
    t = str(final_answer).strip().lower()
    # look for explicit option or leading token
    if re.search(r"\b0\)\.*\s*true|answer is 0", t):
        return True
    if re.search(r"\b1\)\.*\s*false|answer is 1", t):
        return False
    # leading word
    head = re.sub(r"^[^a-z]*", "", t)
    if head.startswith("true"):
        return True
    if head.startswith("false"):
        return False
    if "true" in t and "false" not in t:
        return True
    if "false" in t and "true" not in t:
        return False
    return None


if __name__ == "__main__":
    import json, sys
    d = json.load(open("data/traces/prontoqa_claude_sonnet.json"))
    agree = tot = unparsed = 0
    for r in d:
        gold, info = gold_answer(r["question"])
        if gold is None:
            unparsed += 1
            continue
        ma = model_answer_bool(r["final_answer"])
        if ma is None:
            continue
        recovered_gold = ma if r["correct"] else (not ma)
        tot += 1
        agree += int(gold == recovered_gold)
    print(f"engine vs recovered-gold agreement: {agree}/{tot} = {agree/tot:.3f}  unparsed_q={unparsed}")
