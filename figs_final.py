"""Final figures. Reads only committed JSON; no GPU."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("r1_tables.json"))

ORDER = [("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B"), ("Qwen3-8B", "8B"),
         ("Qwen3-14B", "14B"), ("Qwen3-32B", "32B"),
         ("Qwen2.5-32B", "Q2.5-32B"), ("Mistral-7B", "Mistral-7B")]


def find(key):
    for k in R:
        if k.startswith(key):
            return R[k]
    return None


# ---------------- F1: the scale map ----------------
rows = []
for key, label in ORDER:
    M = find(key)
    if M is None:
        continue
    kts = sorted([k for k in M if k.startswith("kt")], key=lambda s: float(s[2:]))
    # summary per model: mean over doses of each channel
    sr = [M[k]["self_vs_random"][0] for k in kts]
    sc = [M[k]["self_vs_clean"][0] for k in kts]
    pc = [M[f"probe_vs_clean"][k] for k in kts if k in M["probe_vs_clean"]]
    wc = [v["top1"] for v in M["probe_which_concept"].values()]
    rows.append(dict(label=label, self_random=np.mean(sr),
                     self_random_lo=np.mean([M[k]["self_vs_random"][1] for k in kts]),
                     self_random_hi=np.mean([M[k]["self_vs_random"][2] for k in kts]),
                     probe_clean=np.mean(pc), which=np.mean(wc) if wc else np.nan,
                     gate=M["power_gate"]["self_vs_clean"],
                     yes_concept=M["yesrate_concept"]))

fig, ax = plt.subplots(figsize=(8.8, 4.1))
x = np.arange(len(rows))
ax.bar(x - 0.22, [r["probe_clean"] for r in rows], 0.4, color="#4d4d4d",
       edgecolor="k", lw=0.4, label="probe: injection event vs clean")
ax.bar(x + 0.22, [r["self_random"] for r in rows], 0.4, color="#b2182b",
       edgecolor="k", lw=0.4, label="self-report: concept vs matched random",
       yerr=[[max(0, r["self_random"] - r["self_random_lo"]) for r in rows],
             [max(0, r["self_random_hi"] - r["self_random"]) for r in rows]],
       capsize=3)
for i, r in enumerate(rows):
    ax.annotate(f"yes-rate\n{r['yes_concept']:.2f}", (x[i] + 0.22, 0.06),
                ha="center", fontsize=6.6, color="white")
ax.axhline(0.5, ls="--", c="k", lw=1)
ax.text(len(rows) - 0.4, 0.51, "chance", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.06)
ax.set_title("The scale map: the injection event is decodable everywhere; "
             "the model's report carries no signal anywhere\n"
             "(fluency preserved in all cells; untrained yes-rate on injected "
             "trials printed inside the report bars)", fontsize=9.6)
ax.legend(fontsize=8.2, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig1_scale_map.png", dpi=190)
print("wrote fig1_scale_map.png")

# ---------------- F2: the installation autopsy ----------------
runs = {
    "as first run\n(saturated dose in)": json.load(open("readout3_swap.json")),
    "dose fixed": None,     # same file: dose-matched doses are the kept ones
    "vectors centred": json.load(open("readout4_centred.json")),
}
first = json.load(open("readout_install.json"))
swap3 = json.load(open("readout3_swap.json"))
cent = json.load(open("readout4_centred.json"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

# panel 1: the collapse of the 1.000
stages = [
    ("original run\n(dose ceiling +\ncollinear vectors)",
     np.mean([v["vs_random"] for v in first["after"]["held_out"].values()])),
    ("saturated dose\nremoved",
     np.mean([v["vs_random"] for v in swap3["after"]["held_out"].values()])),
    ("vectors centred\n(real held-out)",
     np.mean([v["vs_random"] for v in cent["after"]["held_out"].values()])),
]
cols = ["#b2182b", "#ef8a62", "#fddbc7"]
xx = np.arange(3)
a1.bar(xx, [s[1] for s in stages], 0.62, color=cols, edgecolor="k", lw=0.5)
for i, s in enumerate(stages):
    a1.annotate(f"{s[1]:.3f}", (i, s[1] + 0.02), ha="center", fontsize=10,
                fontweight="bold")
a1.axhline(0.5, ls="--", c="k", lw=1)
a1.set_xticks(xx); a1.set_xticklabels([s[0] for s in stages], fontsize=8)
a1.set_ylabel("held-out AUROC vs matched random")
a1.set_ylim(0, 1.1)
a1.set_title("'Installed introspection' under\nsuccessive integrity controls", fontsize=9.5)
a1.spines[["top", "right"]].set_visible(False)

# panel 2: the swap test
sw = cent["swap"]
labels = ["state only\nKL=1", "state only\nKL=4", "text only\nKL=1", "text only\nKL=4"]
vals = [sw["state_kt1.0"]["auroc"], sw["state_kt4.0"]["auroc"],
        sw["text_kt1.0"]["auroc"], sw["text_kt4.0"]["auroc"]]
cols2 = ["#2166ac", "#2166ac", "#ef8a62", "#ef8a62"]
a2.bar(np.arange(4), vals, 0.62, color=cols2, edgecolor="k", lw=0.5)
for i, v in enumerate(vals):
    a2.annotate(f"{v:.3f}", (i, v + 0.02), ha="center", fontsize=10)
a2.axhline(0.5, ls="--", c="k", lw=1)
a2.set_xticks(np.arange(4)); a2.set_xticklabels(labels, fontsize=8)
a2.set_ylim(0, 1.1)
a2.set_title("What does the trained readout consult?\nIts state (blue) or its transcript (orange)?",
             fontsize=9.5)
a2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("fig2_autopsy.png", dpi=190)
print("wrote fig2_autopsy.png")

# summary numbers for the paper
print("\nstage values:", [round(s[1], 3) for s in stages])
print("swap:", {k: round(v["auroc"], 3) for k, v in sw.items()})
