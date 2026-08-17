# Deviations ledger

`PREREG.md` (commit `970e327`) was written before any result was inspected.
Everything below is a change made after seeing data, recorded so a reader can
audit which claims are confirmatory and which are exploratory.

1. **Fresh-prompt introspection replaced with in-context introspection.** The
   pre-registered design asked the introspection question as a fresh prompt with
   the hook detached. That made the forward pass byte-identical across trials
   (self-report variance exactly zero); the instrument was degenerate, not the
   model silent. Redesigned so the perturbed episode stays in context. Recorded
   as instrument failure; no claim salvaged from the first run.

2. **Span injection replaced with single-position injection.** Steering all
   generated tokens collapsed output into repetition loops at every dose, and
   apparent detection tracked the damage. Single-position injection preserves
   coherence (0.99-1.10 of clean) and is used for all headline claims. Span
   results are retained in the repository as the damage-confound demonstration.

3. **Norm-matched doses replaced with measured-KL-matched doses**, after
   observing an ~18x KL gap between arms at equal norm.

4. **KL targets changed across iterations** (0.1/0.5/2.0 → 0.5/2.0/8.0 →
   0.02-3.0 ladder; installation at 1.0/4.0/12.0) as the coherent operating
   range was established. The installation's 12.0 target saturated the strength
   ceiling and is excluded from all headline numbers (Result 2, dose-ceiling
   trap); it appears only in the autopsy.

5. **The pre-registered power gate voided 8B and 14B diagnostics as
   confirmatory** (self-vs-clean 0.479 and 0.521 against the 0.60 gate); they
   are reported descriptively. Mistral passes at 0.604. The gate as written
   contemplated one model; applying it per-model is our (conservative)
   interpretation.

6. **The installation experiment and its controls (Result 2) are entirely
   post-registration.** The shuffled-label, dose-ceiling, collinearity, and
   state/text-swap controls were designed after the first 1.000 result, each to
   attack it. They are exploratory by the prereg's standard and are the reason
   the 1.000 is not in the abstract.

7. **The 14B and 32B rungs were added in the final 12 hours** to turn a
   single-scale null into a scale map. Same harness, unchanged; no analysis
   choices were altered after seeing their results.

8. **Bug fixes after data**: an in-place hook broke autograd during LoRA
   training (rerun after an out-of-place rewrite, results unchanged in kind);
   a NameError in the swap-test scorer crashed one run before saving (rerun);
   `sentencepiece` missing from the image for Mistral (rerun).

9. **The pre-registered W2 window condition (hook active during the answer) was
   never run.** All diagnostic ladders are W1 only. Disclosed rather than run:
   W2 steers the computation performing the introspection, so a positive there
   would be uninterpretable anyway; the prereg listed it as a contrast, and we
   record its absence instead of quietly dropping it.

10. **The primed-elicitation arm (ladder6) and the 14B/32B rungs are
    post-registration additions**, responding to the literature's
    enhanced-prompting positives (Vogel; Pearson-Vogel et al.) and to scale
    coverage. Same harness and scoring throughout; no analysis choices changed
    after their results were seen.

11. **The Qwen2.5-32B opening failed its seed replication.** Run 1 (seed 0)
    gave self-vs-random 0.622 [0.556, 0.677] at the low dose; the pre-committed
    replication (seed 1, identical harness) returned 0.45-0.57 at all doses.
    The paper reports the cell and its non-replication together and claims no
    opening anywhere.
