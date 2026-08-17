# Seed replication of the Qwen2.5-32B opening

Run before the submission deadline, as committed in the paper. Original run
(seed 0) showed self-vs-random 0.622 [0.556, 0.677] at kt=1.0, the map's only
above-chance cell. Replication (seed 1, identical harness, `ladder5_q25rep.json`):

| kt | self vs random |
|---|---|
| 0.02 | 0.538 |
| 0.1 | 0.458 |
| 0.3 | 0.567 |
| 1.0 | **0.447** |
| 3.0 | 0.461 |

The opening does not survive reseeding. The lineage hypothesis it suggested is
withdrawn, and the cell joins the paper's catalogue of artifact classes: a seed
fluctuation, caught by the replication we ran on ourselves.
