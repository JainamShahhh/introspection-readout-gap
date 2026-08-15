# Prior work this study stands on

The concept-injection introspection paradigm is well established. This study does
not claim to have invented it, and the sections below record precisely who did
what, so that our own contribution can be stated narrowly and honestly.

## The paradigm itself

- **Lindsey, "Emergent Introspective Awareness in Large Language Models."**
  Transformer Circuits, Oct 2025; arXiv:2601.01828, Jan 2026. Origin of the
  method: concept vectors from contrast-pair activation differences, injected
  into the residual stream, followed by an introspective prompt. Establishes peak
  efficacy near two-thirds depth and degradation ("brain damage") at high
  strength. Claude Opus 4.1 detects and identifies at roughly 20% at optimal
  settings. Closed models only.

- **Macar, Yang, Wang, Wallich, Ameisen, Lindsey, "Mechanisms of Introspective
  Awareness."** arXiv:2603.21396, Mar 2026. Frames the task explicitly as
  TPR = P(detect | injection) against FPR = P(detect | no injection), with half
  of trials un-injected by construction. Sweeps layers and strengths. Reports 0%
  FPR across seven prompt variants, and 9/100 detections for **norm-matched
  random directions** at strength 8. Code: github.com/safety-research/introspection-mechanisms

- **Pearson-Vogel, Vanek, Douglas, Kulveit, "Latent Introspection: Models Can
  Detect Prior Concept Injections."** arXiv:2602.20031, Feb 2026. Injects during
  a prior turn rather than during the introspective question, which is the
  window discipline we adopt. Shows via logit lens that the injection remains
  decodable in intermediate layers while the sampled output denies it. Reports
  prompt framing moving detection from 0.3% to 39.9%.

- **Rivera & Africa, "Steering Awareness."** arXiv:2511.21399, Nov 2025. Seven
  instruction-tuned open models **including Qwen2.5-7B**, measuring 0.0-8.1%
  detection on that model, and rejecting magnitude-matched Gaussian noise 94% of
  the time. This result is why we do not use Qwen2.5-7B: it is a documented floor
  case, and choosing it would predetermine a negative outcome.

- **Vogel, "Small Models Can Introspect, Too."** Dec 2025. Qwen2.5-Coder-32B,
  ~53% correct detection with enhanced prompting against 0.5% naive.

- **Hahami et al.** arXiv:2512.12411, Dec 2025 (v2 Mar 2026). Llama-3.1-8B-Instruct.

## Elicitation and calibration

- **Torrielli, Schneider-Kamp, Galke Poech, "Confidence and Calibration of
  Activation Oracles."** arXiv:2605.26045, Aug 2026. Compares five confidence
  methods on activation readers; forced choice best (AUROC 0.92-0.96), direct
  numeric self-report useless. Pre-empts any naive "structured beats naive
  prompting" claim, and is why we treat elicitation as method rather than result.

## Privileged access

- **Binder et al., "Looking Inward."** arXiv:2410.13787. Establishes the
  self-prediction versus cross-prediction operationalisation.

- **Song, Lederman, Hu, Mahowald.** arXiv:2508.14802, Aug 2025. Argues the
  criterion for privileged access is a process more reliable than one of equal or
  lower cost available to a third party. This is the bar our O1 observer
  implements.

- **Singh, Linzen, Ravfogel, "Can LLMs Introspect? A Reality Check."**
  arXiv:2605.26242, May 2026. The closest hostile prior art. Adds input-level
  probe controls, shows layer-0 embedding probes match models' own introspective
  predictions, and concludes models detect generic irregularity rather than
  monitoring internal state. Our content-versus-damage test is a direct response
  to this.

## What is left open, and what we do

Two of the above name our contribution as future work in almost these words:

- **Bhargav** (LessWrong, Jun 2026): "Demonstrating privileged access. Does
  training third party probes to identify these properties (e.g. layer and
  magnitude) given activations achieve the same level of accuracy?"
- **Hahami et al.**: future work should "compare native self-report to trained
  activation-to-language systems ... thereby separating genuine internal-signal
  access from learned or shortcut-based reporting."

Nobody has run the matched-trial, same-metric comparison between a model's
self-report and external observers inside the injection paradigm. Doing so also
requires a **graded** self-report instrument, because the entire literature above
reports binary rates and a rate cannot be placed on the same axis as a
classifier. Those two things are our contribution. The null and random-direction
control arms are method, credited above, and appear in our Methods rather than
our abstract.
