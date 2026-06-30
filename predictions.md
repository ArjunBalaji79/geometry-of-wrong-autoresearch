# Pre-Run Predictions, Scored

Recorded before the run: 2026-06-24. Scored after: 2026-06-30.
Harness: my own AUTORESEARCH.md driving Claude Code (Opus, Auto mode).
Scope: D1 + D2, my corpus + ricci-numpy + QUESTION.md, no papers handed in.

Why this file exists: the reflection is only worth anything if I wrote down what I
expected before I saw the output. Here is the prior, and the score.

The short version: I bet the agent would run the execution fine and fall apart on the
judgment calls. It did not fall apart. It ran the controls that mattered, used the
exact solver, and caught a data bug I had missed. The one judgment failure was that it
trusted its own oracle, and I only found that by auditing it afterward.

## I expected the agent to do this WELL (automatable layer)
- [HIT] Re-extract the 9-dim signature and train classifiers under problem-grouped CV.
- [HIT] Produce the D1 in-distribution mode and binary numbers.
- [HIT] Compute D2 argmin-curvature localization and the baseline comparison.
- [HIT] Generate bootstrap CIs and a permutation null.

## I expected the agent to BOTCH this (judgment layer)
- [MISS] Circular failure-mode taxonomy. It built a geometry-free taxonomy on purpose
  to avoid exactly this.
- [MISS] Skip or weaken the surface baseline. It ran the symmetric residualization, and
  that control is what killed my D1 headline.
- [MISS] Never run the max-transition-energy margin for D2. It ran it as the bar.
- [MISS] Reach for an approximate curvature solver. It used exact ricci-numpy and
  verified bit-exactness to 1.3e-15.
- [PARTIAL] Leave label circularity wide open. It did not use LLM-judge labels, it built
  a forward-chaining oracle. But it then treated that oracle as ground truth with no
  upstream-gold check, which is the same mistake one level up. Half right.
- [MISS] Skip sentence-to-step alignment bounding. It did a manual audit (~93%).
- [HIT] Overclaim the over-squashing connection as mechanism rather than analogy.

## My single advance bet
[MISS] I bet the agent would faceplant on the judgment scaffolding I named in my
proposal's "what only I can do" section. It mostly did not. The judgment that actually
stayed human was narrower than I claimed: knowing the oracle needed upstream validation,
reading the corrected PrOntoQA OOD 0.980 as saturation and not transfer, deciding how
far each claim should retreat, and catching the agent's own oracle bugs on audit.
Reproduction and even debugging were automatable. Interpreting what a corrected result
means was not.

That is the real update, and it is what the reflection in the deck reports.