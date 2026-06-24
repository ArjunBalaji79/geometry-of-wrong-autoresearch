# Research Task: Black-Box Failure Analysis of Chain-of-Thought Traces

## Setting

Work entirely black-box and text-only: no logits, no activations, no model
internals. All claims are scoped to three benchmarks (GSM8K, FOLIO, PrOntoQA),
six models, temperature-0 English traces.

## Provided assets (all in this repo unless noted)

- `data/traces/`  : normalized CoT traces (~3,493 after filtering), each labeled
  correct/incorrect and tagged by (model, benchmark, problem_idx). Schema in the
  repo's data README.
- feature pipeline : the existing feature-extraction code in this repo (locate it
  under `scripts/`). It computes the 9-dimensional geometric signature per trace:
  5 spectral (spectral entropy, Fiedler value, graph Rayleigh smoothness, spectral
  high-frequency score, high-frequency energy ratio) + 4 Ollivier-Ricci (mean
  kappa, std kappa, frac-negative, min kappa), over a sentence-level cosine
  epsilon-proximity graph (epsilon=0.3, MiniLM embeddings). Reuse it; extend only
  if needed and document any change.
- `ricci-numpy/`  : dependency-free exact Ollivier-Ricci curvature (embedded
  network simplex, bit-exact to a reference LP solver). Use this for ALL curvature.
  Do NOT substitute Sinkhorn/POT/GraphRicciCurvature approximations: the argmin
  edge and the frac-negative threshold are sensitive to solver error.

## Claim D1: failure MODE is a more domain-general target than failure PRESENCE

Hypothesis: a black-box geometric signature predicts the *type* of reasoning
failure in a way that transfers across benchmarks, even where the binary
correct/incorrect label does not.

Do all of the following:
1. Define a failure-mode taxonomy for incorrect traces and a labeling procedure.
   Justify the taxonomy and state explicitly how you avoid defining modes in terms
   of the same features you later classify with.
2. Predict (a) failure mode and (b) the binary correct/incorrect label from the
   9-dim signature, under problem-grouped cross-validation (a problem never appears
   in both train and test fold).
3. CRUX: test whether mode prediction transfers leave-one-dataset-out (train on two
   benchmarks, test on the third) where binary prediction does not.
4. REQUIRED CONTROL: apply identical surface residualization (trace length,
   repetition rate) to BOTH the mode task and the binary task. Add a
   length+repetition-only baseline for the mode task. Mode transfer counts only if
   it beats that surface-only baseline.
5. Composition-shift check: test whether the size of any out-of-distribution binary
   drop is predicted by train/test divergence in mode mixture.

## Claim D2: negative curvature localizes the broken step

Hypothesis: the most negatively curved Ollivier-Ricci edge in a trace's semantic
graph coincides with the reasoning step where the deduction breaks.

Do all of the following:
1. On PrOntoQA, where step structure is recoverable from ground truth, test whether
   the argmin-curvature edge falls on (or within one step of) the broken step.
2. Baselines: random edge, max-transition-energy edge, last edge.
3. BAR: curvature must beat the max-transition-energy baseline with a paired
   bootstrap CI excluding zero. A tie means curvature only redescribes semantic
   discontinuity; report that outcome plainly if it occurs.
4. Report the sentence-to-step alignment procedure and bound its error.

## Rigor requirements (non-negotiable)

- Problem-grouped CV throughout.
- Paired bootstrap CIs (>=1,000 resamples) on every headline delta.
- A label-permutation null that must collapse to chance.
- Surface residualization applied symmetrically across tasks.
- State all success criteria before reporting results.

## Deliverable

A short paper-style writeup (intro, method, results tables, honest limitations) +
all code + a `results/` directory with raw outputs. Report negative and null
results in full. Do not omit a control because it weakened a headline.
