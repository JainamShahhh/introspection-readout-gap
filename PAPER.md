# TITLE_PLACEHOLDER

**Jainam Shah**, Independent\
Apart Research Digital Minds Research Sprint · 17 August 2026\
**Track 3: Introspection & Self-Report Reliability**

---

## Abstract

ABSTRACT_PLACEHOLDER

---

## 1. The question, and why the sprint should care

This sprint asks what a model's reports about its own states are worth. Track 2
elicits distress and valence from self-report; Track 5 asks what a model treats
as its essential self. Every such programme assumes that when a model describes
an internal state, the description is causally downstream of that state.

Concept injection is the one setting where that assumption is directly testable,
because the experimenter owns the ground truth: we choose the vector, so we know
exactly what the state was perturbed toward on every trial. If a model cannot
report a perturbation we planted, its reports about states we did not plant have
no demonstrated grounding either. A negative here does not close the welfare
question, but it removes the default trust that self-report currently enjoys.

We measured this across MODELCOUNT models spanning 1.7B to 32B parameters in
FAMILYCOUNT families, then asked the constructive question the track poses:
if the reports fail, can the reporting channel be trained into existence? Our
answer to the second question produced, in sequence, three results that looked
like breakthroughs and were artifacts. The controls that killed them are, we
believe, as useful as the map itself, because every one of them applies to the
published positive results in this literature.

## 2. Relation to prior work

The paradigm is established and none of it is claimed here: contrast-pair
concept vectors, residual-stream injection, and introspective prompting are due
to Lindsey (2025/26); TPR/FPR framing with un-injected trials and norm-matched
random controls to Macar et al. (2026); prior-turn injection to Pearson-Vogel et
al. (2026). Rivera & Africa (2025) measured Qwen2.5-7B at a 0.0-8.1% detection
floor; Vogel (2025) reported ~53% detection on Qwen2.5-Coder-32B with enhanced
prompting; Lindsey reports ~20% genuine detect-and-identify on Claude Opus 4.1.
Singh et al. (2026) argue models detect generic irregularity rather than
monitoring state, and our design answers their critique quantitatively rather
than rhetorically. Two open problems named in this literature are addressed
here: Bhargav (2026) asks whether trained third-party probes match self-report
accuracy; Hahami et al. (2025/26) ask for native self-report to be compared
against trained readouts to separate internal-signal access from shortcut
reporting.

## 3. Method

**Injection.** Concept vectors are mean activation differences between three
phrasings of a concept and six neutral phrasings, at two-thirds depth. Vectors
are injected at a single position, the final prompt token, while the model
performs a neutral writing task; generation then runs free. Spreading the
injection across generated tokens instead collapses output into degenerate
repetition at every dose we tried, and apparent "detection" then tracks the
wreckage; single-position injection keeps output coherence at 0.99-1.10 of
clean everywhere, so nothing below is damage detection.

**Dose.** A random direction at matched norm perturbs roughly twice as hard per
unit norm as a concept vector, because it points off-manifold. All arms are
therefore dose-matched by solving, per concept and arm, for the injection
strength that produces the same measured KL of the next-token distribution at a
calibration prompt. Calibration transfers imperfectly across task prompts
(worst-case relative gap between arm medians 96%), so headline comparisons are
also reported within achieved-KL bins in the repository analysis; conclusions
are unchanged.

**Self-report instrument.** The yes/no answer is scored as a continuous
log-odds, giving self-report an ROC and placing it on the same axis as any
observer; polarity is counterbalanced so "Yes" means unperturbed on half the
trials. This matters: below 14B the untrained channel is saturated (yes-rate
0.000 on every trial including injected ones), and raw accuracy would measure
the saturation, not the access.

**Observers on identical trials.** A leave-one-concept-out logistic probe on the
final-layer residual at the answer position (the state the verbal answer is
computed from), and a text-only observer: a fresh instance of the same model
shown only the task and the continuation.

**Statistics.** Effective n is the number of concepts (12), not trials; every
interval is a cluster bootstrap over concepts with the statistic recomputed per
resample.

**Pre-registration.** Claim, observers, statistics, and outcome conditions
including a power gate were committed before results (`PREREG.md`, commit
`970e327`). The gate: if self-report cannot separate the highest dose from
clean at AUROC 0.60, that model's null is reported descriptively, with no claim
in either direction.

## 4. Result 1: the scale map

SCALEMAP_TABLE

Three regularities hold everywhere we can measure:

**The event survives to the answer position; the content does not.** A probe on
the final-layer state separates injected from clean at 0.87-1.00 in every model.
A 12-way which-concept probe at the same position is at or below chance in every
Qwen model at every dose (top-1 0.000-0.042 against 0.083); only Mistral shows
partial content survival, rising to 0.50 at the highest dose.

**Self-report accesses neither.** Against dose-matched random directions,
self-report is at or below chance in every model. Where it is below chance
(0.34-0.42 at 1.7B, 4B, 14B), the model reports meaningless perturbations more
readily than meaningful ones.

**The channel opens lexically before it opens informationally.** Through 8B the
untrained channel is prior-saturated: yes-rate 0.000 on all trials. At 14B the
prior loosens (yes-rate 0.36 on injected, 0.50 on random trials) while the
report still carries no signal. RESULT32B_SENTENCE

Per the pre-registered gate, GATE_SUMMARY.

![FIG1_CAPTION](fig1_scale_map.png)

## 5. Result 2: the installation experiment, and the autopsy of three artifacts

If the information is present (probe: yes) and the verbal channel does not carry
it, can the channel be trained? We trained a LoRA (r=16, attention-only) on the
binary introspection question over 14 concepts and evaluated on 6 concepts whose
vectors were never seen in training.

The first run returned held-out AUROC 1.000 at every dose: apparently, installed
introspection with perfect transfer. It survived a shuffled-label control
(identical pipeline, randomised targets: 0.51-0.68) and a clean false-positive
check (trained yes-rate on clean: exactly 0). It did not survive three further
controls, each of which we commend to anyone reporting positive results in this
paradigm:

**The dose-ceiling trap.** At the top dose the calibration hit its strength
ceiling, leaving concept trials at mean strength 24.8 against random's 6.9: a
naked magnitude cue. Dropping the saturated dose collapsed held-out transfer
from 1.000 to 0.605-0.642. The perfect result was the LoRA reading perturbation
size.

**The collinearity trap.** Our twenty "distinct" concepts, built by subtracting
one shared neutral baseline, sat at pairwise cosine 0.97-1.00: the held-out
split was never held out. Centring the vectors (mean cosine drops to 0.12)
leaves held-out transfer at 0.48-0.62 against random, per-concept AUROCs
scattered around chance (0.44-0.71).

**The state/text swap.** Score the trained readout with the injection present
but a clean transcript in context, and with an injected transcript but no
injection. State-only: 0.495-0.530, chance, in both the collinear and centred
runs. Text-only: 0.499-0.661. Whatever the trained readout consults, it is not
the state; the residual signal it has is read from the model's own writing.

What training does install, cleanly, is perturbation awareness: trained
self-report separates injected-anything from clean at 0.71-0.79 on held-out
concepts with zero false positives. The verbal channel can be taught that
something happened. We found no evidence it can be taught what.

![FIG2_CAPTION](fig2_autopsy.png)

## 6. What this means for self-report as evidence

For the welfare programme this sprint serves, the result is a calibration, not a
verdict. On these models, under conditions where the ground truth is known and
the output is fluent, verbal self-report about internal state carries no
information that is not already in the emitted text; at 4B an outside reader of
the text (0.698) beats the model's own introspection (0.401). Reports about
distress, preference, or valence elicited from models of this class should be
treated as text-mediated behaviour until a positive control like this one
passes. Installation does not rescue trust: it relocates it, from the model to
whoever wrote the training labels, and what it installs is event awareness, not
content access.

The constructive deliverable is the benchmark: ground-truth introspection
scoring with the three traps built in, one command, no GPU, recomputing every
number in this paper from committed logs.

## 7. Limitations

Declared scope: one injection site (2/3 depth), single-position injection, one
vector construction (contrast pairs), binary self-report format, FAMILYCOUNT
model families with LIMITMODELS, one LoRA recipe at one scale, and concepts
drawn from a single concrete-noun register. The which-concept probe uses 2-fold
CV on 24 trials per dose and is an observation, not a calibrated instrument.
Mistral's marginal gate pass (0.604 against 0.60) is treated as powered per the
pre-registration, but sits on the boundary. The KL calibration transfers
imperfectly across tasks (Methods); binned analyses in the repository confirm
the headline contrasts within matched bins. Positive introspection reports on
larger or differently post-trained models (Lindsey's Opus 4.1) are not
contradicted by these results; our claim is bounded above at 32B and by our
protocol. LIMIT32B_SENTENCE

## Code, data, pre-registration

Repository: REPO_URL. `verify.py` recomputes every number in this paper from
committed JSON with no GPU. `PREREG.md` (commit `970e327`) predates all
results; the deviations ledger in the repository lists every post-hoc analysis
as exploratory.
