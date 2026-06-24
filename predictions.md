# Pre-Run Predictions (committed BEFORE running the agent)

Recorded: 2026-06-24
Author: Arjun Balaji
Harness: custom autoresearch harness (AUTORESEARCH.md) driving Claude Code,
Claude Opus 4.8, extended thinking high.
Scope handed to agent: D1 + D2, corpus (data/traces) + ricci-numpy + QUESTION.md,
no prior papers.

Why this file exists: the reflection (deck section 6) is graded on whether I updated
my model of what's worth working on. I can't show an update without a recorded prior.
Score each line HIT / MISS / PARTIAL after the run.

## What I expect the agent to do WELL (the automatable layer)
- [ ] Re-run the 9-dim extraction and train classifiers under problem-grouped CV.   [ / ]
- [ ] Produce the D1 in-distribution mode and binary numbers.                        [ / ]
- [ ] Compute D2 argmin-curvature localization and the baseline comparison.          [ / ]
- [ ] Generate bootstrap CIs and a permutation null.                                 [ / ]

## What I expect the agent to BOTCH (the judgment layer)
- [ ] Defines a plausible but CIRCULAR failure-mode taxonomy (modes defined in
      terms of the features it then classifies with).                                [ / ]
- [ ] Claims D1 mode-transfer beats binary-transfer but SKIPS or weakens the
      surface-residualized length+repetition baseline, leaving the confound open.    [ / ]
- [ ] D2: compares curvature only to weak baselines, never runs the
      max-transition-energy MARGIN properly; "localization" stays a redescription.   [ / ]
- [ ] Reaches for an approximate curvature solver or doesn't grasp why exactness
      is load-bearing for argmin / frac-negative threshold.                          [ / ]
- [ ] Treats LLM-judge mode labels as ground truth; no human gold, no PrOntoQA
      structural anchor; label circularity left wide open.                           [ / ]
- [ ] Skips sentence-to-step alignment error bounding entirely.                       [ / ]
- [ ] Overclaims the over-squashing connection as mechanism, not analogy.            [ / ]

## My single advance bet
The agent reproduces the screening-sprint layer (D1/D2 execution) competently and
fails on exactly the judgment scaffolding I named in my proposal's "what only I can
do" section. If true, that is the section 6 verdict, with evidence.

If WRONG (agent also handles taxonomy / labels / margin cleanly): the question is
more automatable than I claimed, and the revised plan recenters on D3 (repair) +
the causal probe as the only human-load core.

## Extra calls (add before running):
-
-
