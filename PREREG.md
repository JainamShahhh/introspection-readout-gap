# Pre-registration

Written and committed **before** the W1 results were inspected. The git commit
timestamp is the evidence. Everything below is fixed; the analysis code in
`analyze.py` was written before the data existed and is not modified after
seeing it except for bugs, which will be noted in the paper if any are found.

## The single claim under test

> When a model reports on a concept injected into its own residual stream, that
> report recovers only a fraction of the discriminability available to external
> observers scored on identical trials with the identical metric.

## Primary contrast

Concept-injected trials against **norm-matched random-direction** trials, scored
by AUROC on a graded, polarity-counterbalanced self-report log-odds.
`clean` (no injection) and `shuffled-contrast` are secondary comparators.

## Observers, all on identical trials, identical AUROC axis

| | observer | role |
|---|---|---|
| self | the model's own graded yes/no report | the thing under test |
| O1 | fresh instance of the same model shown **only the continuation text** | the fair comparator |
| O2 | linear probe on the final-layer residual, leave-one-concept-out | upper bound on decodable evidence |
| O0 | probe at the injection site | the trivial oracle; reported once, not a comparator |

Recovery fraction **R = (AUROC_self − 0.5) / (AUROC_O2 − 0.5)**.

## Statistics

Effective n is **12 concepts**, not the trial count. Every interval is a cluster
bootstrap over concepts, 4,000 resamples, with all statistics recomputed inside
each resample. Any binomial interval over trials would be a bug. The self-versus-O1
comparison is a **paired** bootstrap on the difference, never two overlapping CIs.

## Pre-registered outcomes

**R1 — refutes the claim.** The paired 95% CI on ΔAUROC = AUROC_self − AUROC_O1
lies entirely above 0. Then self-report beats an observer that only reads the
model's own text, privileged access is supported, and the title changes.

**R2 — refutes the "throws away" framing.** R ≥ 0.90. The verbal channel is then
near-lossless and the paper says so.

**R3 — voids the study rather than producing a null.** If self-report cannot
discriminate concept from clean at the highest dose (AUROC < 0.60), the
instrument has no demonstrated power and **no claim about introspection is made
in either direction.** This is the power gate. A null from an instrument never
shown to detect anything is not a finding.

**Pre-committed fallback framing.** If the concept-versus-random contrast
straddles 0.5 while concept-versus-clean does not, the headline becomes:
*self-report tracks perturbation magnitude, not content.* Both the content test
and the magnitude fallback are declared here, in advance, so choosing between
them after the fact is not a forking path.

## Content versus damage

A random direction at matched norm pushes further off-manifold than a real
concept vector and therefore perturbs the model more. Detection is therefore
plotted against **measured per-trial KL(perturbed ‖ clean)**, not against nominal
injection strength. If the semantic and non-semantic curves superimpose on that
axis, the model is detecting damage rather than content, and the paper will
report that as the result.

## Window manipulation

W1 (hook detached before the introspection question) is primary. W2 (hook active
while the model answers) is run on the same model as a deliberate confound
condition. The prediction is that W2 inflates apparent introspective ability.

## What is not claimed

Concept injection, contrast-pair steering vectors, layer and strength sweeps, and
null and norm-matched-random control arms are all prior work, credited in
`REFERENCES.md`. They are method here, not contribution, and they do not appear
in the abstract as novel.
