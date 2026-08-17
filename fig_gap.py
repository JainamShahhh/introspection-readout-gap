"""The readout gap across scale.

Coherence does not vary across this dose range (0.99-1.10 of clean everywhere),
which removes the damage confound entirely and makes a
detection-versus-coherence plot degenerate. That is a better outcome than the
figure it kills: every point below is a fluency-preserving injection, so the gap
between what a probe reads and what the model reports cannot be explained by
output degradation.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("summary.json"))
MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
sing = {m: sorted([d for d in D if d["span"] == 1 and d["model"] == m],
                  key=lambda d: d["kt"]) for m in MODELS}

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
for ax, m in zip(axes, MODELS):
    rows = sing[m]
    if not rows:
        ax.set_visible(False); continue
    x = [d["kt"] for d in rows]
    ax.plot(x, [d["probe"] for d in rows], "s-", color="#4d4d4d", lw=2, ms=7,
            label="linear probe on the same state")
    ax.plot(x, [d["self"] for d in rows], "o-", color="#b2182b", lw=2.4, ms=8,
            label="the model's own report")
    ax.fill_between(x, [d["self_lo"] for d in rows], [d["self_hi"] for d in rows],
                    color="#b2182b", alpha=0.18)
    ax.plot(x, [d["o1"] for d in rows], "^--", color="#ef8a62", lw=1.5, ms=6,
            label="stranger reading only the output")
    ax.axhline(0.5, ls=":", c="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("injected dose, KL (nats)\nmatched across arms", fontsize=9)
    ax.set_title(m, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.22, 1.05)
axes[0].set_ylabel("AUROC\nconcept vs matched random")
axes[0].legend(fontsize=7.6, loc="lower left")
fig.suptitle("The information is in the state the answer is computed from. The model does not report it.\n"
             "Every point preserves fluency, so this is not damage detection. No trend across a 4.7x parameter range.",
             fontsize=10.5)
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig("fig1_readout_gap.png", dpi=190)
print("wrote fig1_readout_gap.png")

fl = [d for d in D if d["span"] == 1]
s = [d["self"] for d in fl]; p = [d["probe"] for d in fl]; o = [d["o1"] for d in fl]
print(f"single-position cells: {len(fl)}")
print(f"  self  mean {np.mean(s):.3f}  range {min(s):.3f}-{max(s):.3f}")
print(f"  probe mean {np.mean(p):.3f}  range {min(p):.3f}-{max(p):.3f}")
print(f"  O1    mean {np.mean(o):.3f}")
print(f"  gap (probe - self) = {np.mean(p)-np.mean(s):.3f}")
for m in MODELS:
    r = sing[m]
    if r:
        print(f"  {m:12s} self {np.mean([d['self'] for d in r]):.3f}  "
              f"probe {np.mean([d['probe'] for d in r]):.3f}  "
              f"O1 {np.mean([d['o1'] for d in r]):.3f}  "
              f"coh {np.mean([d['coh'] for d in r]):.2f}")
