# Qwen2.5-32B: both seeds

Two runs of the identical harness (seeds 0 and 1), reported pooled in the paper.

| kt | seed 0 | seed 1 | pooled |
|---|---|---|---|
| 1.0 | 0.622 | 0.447 | 0.534 [0.491, 0.566] |
| all doses | 0.583 | 0.494 | 0.542 [0.502, 0.580] |

Raw logs: `ladder5_q25_32b.json`, `ladder5_q25rep.json`. The pooled estimate is
the paper's reported value; the per-seed split is the basis of the seed trap in
Section 5.
