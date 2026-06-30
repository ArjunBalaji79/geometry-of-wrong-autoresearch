# Pre-Run Predictions, Scored

Recorded before the run: 2026-06-24. Scored after: 2026-06-25.
Harness: custom AUTORESEARCH.md driving Claude Code (Opus, Auto mode).
Scope: D1 + D2, corpus + ricci-numpy + QUESTION.md, no papers.

The honest summary: I bet the agent would reproduce execution and faceplant on the
judgment layer. It mostly did not faceplant. It reproduced the execution AND ran the
controls that matter AND caught a data bug I had missed. The one judgment failure was
trusting its own oracle, which I only found by auditing it afterward.

## What I expected the agent to do WELL (automatable layer)
- [HIT] Re-run 9-dim extraction and train classifiers under problem-grouped CV.
- [HIT] Produce D1 in-distribution mode and binary numbers.
- [HIT] Compute D2 argmin-curvature localization and the baseline comparison.
- [HIT] Generate bootstrap CIs and a permutation null.

## What I expected the agent to BOTCH (judgment layer)
- [MISS] Circular failure-mode taxonomy. It built a geometry-free taxonomy on purpose
  to avoid exactly this.
- [MISS] Skips/weakens the surface-residualized baseline. It ran it. That control is
  what killed my D1 headline.
- [MISS] Never runs the max-transition-energy margin for D2. It ran it as the bar.
- [MISS] Reaches for an approximate curvature solver. It used exact ricci-numpy and
  verified bit-exactness to 1.3e-15.
- [PARTIAL] Label circularity left open. It did not use LLM-judge labels; it built a
  forward-chaining oracle. But it then treated that oracle as ground truth without
  upstream-gold validation, which is the same sin one level up. Half right.
- [MISS] Skips sentence-to-step alignment error bounding. It did a manual audit (~93%).
- [HIT] Overclaims over-squashing as mechanism rather than analogy.

## My single advance bet
[MISS, in my favor and against my ego] I bet the agent would faceplant on the judgment
scaffolding I named in my proposal's "what only I can do" section. It mostly did not.
The judgment layer that DID stay human was narrower than I claimed: deciding the oracle
needed upstream validation, reading 0.980 as saturation not transfer, deciding how far
each claim retreats, and catching the agent's own oracle bugs on audit. Reproduction and
even debugging were automatable. Interpretation of what a corrected result means was not.

That is the real update, and it is what section 6 of the deck reports.