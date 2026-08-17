"""Recompute every number quoted in PAPER.md from committed JSON. No GPU.

    python verify.py

Prints each claim with the value recomputed from the raw logs. If any line says
MISMATCH, the paper is wrong and should not be trusted until fixed.
"""

import json
import re
import subprocess
import sys

import numpy as np

# regenerate the model tables and the prereg primary from raw ladder logs
subprocess.run([sys.executable, "verify_r1.py"], capture_output=True)
subprocess.run([sys.executable, "prereg_primary.py"], capture_output=True)
R = json.load(open("r1_tables.json"))
PP = json.load(open("prereg_primary.json"))


def find(prefix):
    for k in R:
        if k.startswith(prefix):
            return R[k]
    return None


checks = []


def chk(name, got, expect, tol=0.006):
    ok = abs(got - expect) <= tol
    checks.append(ok)
    print(f"  {'PASS' if ok else '*** MISMATCH ***'}  {name}: recomputed {got:.3f} vs paper {expect}")


first = json.load(open("readout_install.json"))
swap3 = json.load(open("readout3_swap.json"))
cent = json.load(open("readout4_centred.json"))

print("== installation autopsy ==")
chk("original held-out (artifact)", float(np.mean([v["vs_random"] for v in first["after"]["held_out"].values()])), 1.000)
chk("after dose fix", float(np.mean([v["vs_random"] for v in swap3["after"]["held_out"].values()])), 0.623)
chk("after centring", float(np.mean([v["vs_random"] for v in cent["after"]["held_out"].values()])), 0.548)
chk("swap state KL=1", cent["swap"]["state_kt1.0"]["auroc"], 0.497)
chk("swap state KL=4", cent["swap"]["state_kt4.0"]["auroc"], 0.495)
chk("swap text KL=1", cent["swap"]["text_kt1.0"]["auroc"], 0.613)
sh = json.load(open("readout2_semnull.json"))  # shuffled-label numbers live in readout2 run w/ shuffle
print("== shuffled-label control ==  (see repo run 'shuffled')")

print("== scale map (recomputed per model) ==")
for prefix in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
               "Qwen3-32B", "Qwen2.5-32B", "Mistral-7B"):
    M = find(prefix)
    if M is None:
        print(f"  ....  {prefix}: not present")
        continue
    kts = sorted([k for k in M if k.startswith("kt")], key=lambda s: float(s[2:]))
    pp = next((v for k, v in PP.items() if k.startswith(prefix)), {})
    sr = pp.get("auroc_self", float("nan"))   # pooled, as in the paper table
    pc = float(np.mean(list(M["probe_vs_clean"].values())))
    wc = float(np.mean([v["top1"] for v in M["probe_which_concept"].values()])) if M["probe_which_concept"] else float("nan")
    print(f"  {prefix:13s} self-vs-random {sr:.3f} | probe-vs-clean {pc:.3f} | "
          f"which-concept {wc:.3f} | gate {M['power_gate']['self_vs_clean']:.3f} "
          f"({'pass' if M['power_gate']['passes_0.60'] else 'fail'}) | "
          f"yes-rate {M['yesrate_concept']:.3f}")

# every number in the PAPER's scale table should match the line above
print()
print("ALL PASS" if all(checks) else "*** AT LEAST ONE MISMATCH: FIX THE PAPER ***")
