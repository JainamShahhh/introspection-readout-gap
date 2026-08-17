"""The pre-registered primary analysis, and the KL-binned robustness check.

Two analyses the paper references and a reviewer would look for first:

1. PREREG PRIMARY: recovery fraction R = (AUROC_self - 0.5) / (AUROC_O2 - 0.5)
   and the PAIRED cluster bootstrap of dAUROC = AUROC_self - AUROC_O1, per
   model. Outcome R1 in PREREG.md turns on the paired difference, not on two
   overlapping intervals.

2. KL-BINNED: dose calibration was done on one task prompt and transfers
   imperfectly to the others, so the headline concept-vs-random contrast is
   recomputed inside achieved-KL bins, where both arms have trials in the same
   bin. Direction of the imbalance: concept trials are perturbed HARDER than
   random ones off-calibration, which biases against the null we report.
"""

import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(0)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    a = np.concatenate([pos, neg])
    o = a.argsort(kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(c)); np.add.at(s, inv, r); r = (s / c)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


out = {}
for path in sorted(glob.glob("ladder5_*.json")) + sorted(glob.glob("ladder6_*.json")):
    j = json.load(open(path))
    if j["config"].get("span", 1) != 1:
        continue
    rows, cs = j["rows"], j["concepts"]
    model = j["model"].split("/")[-1] + (" primed" if "ladder6" in path else "")
    M = {}

    # ---------- paired self vs O1, and R against the probe ----------
    def contrast(cc, field):
        p = [r[field] for r in rows if r["arm"] == "concept" and r["concept"] in cc]
        n = [r[field] for r in rows if r["arm"] == "random" and r["concept"] in cc]
        return auroc(p, n)

    def probe_auroc(cc):
        sel = [r for r in rows if r["arm"] in ("concept", "random")
               and r["concept"] in set(cc) and "act" in r]
        X = np.array([r["act"] for r in sel], np.float32)
        y = np.array([r["arm"] == "concept" for r in sel], int)
        grp = np.array([r["concept"] for r in sel])
        oof = np.full(len(y), np.nan)
        for c in set(cc):
            te = grp == c; tr = ~te
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            oof[te] = LogisticRegression(max_iter=2000, C=0.05).fit(
                X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof)
        return auroc(oof[ok & (y == 1)], oof[ok & (y == 0)])

    self_a = contrast(cs, "self")
    o1_a = contrast(cs, "o1")
    o2_a = probe_auroc(cs)
    M["auroc_self"] = round(self_a, 3)
    M["auroc_o1"] = round(o1_a, 3)
    M["auroc_probe"] = round(o2_a, 3)
    M["recovery_R"] = round((self_a - 0.5) / (o2_a - 0.5), 3) if o2_a > 0.52 else None

    # paired bootstrap on the DIFFERENCE, resampling concepts
    diffs = []
    for _ in range(3000):
        cc = list(RNG.choice(cs, len(cs), replace=True))
        d = contrast(cc, "self") - contrast(cc, "o1")
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    M["delta_self_minus_o1"] = [round(self_a - o1_a, 3), round(float(lo), 3),
                                round(float(hi), 3)]
    M["prereg_R1_privileged_access"] = bool(lo > 0)

    # ---------- KL-binned concept vs random ----------
    ck = [(r["kl"], r["self"]) for r in rows if r["arm"] == "concept"]
    rk = [(r["kl"], r["self"]) for r in rows if r["arm"] == "random"]
    allk = np.array([k for k, _ in ck + rk])
    allk = allk[allk > 1e-6]
    edges = np.quantile(allk, [0, .25, .5, .75, 1.0])
    bins = []
    for i in range(4):
        a0, a1 = edges[i], edges[i + 1] + (1e-9 if i == 3 else 0)
        p = [s for k, s in ck if a0 <= k < a1]
        n = [s for k, s in rk if a0 <= k < a1]
        if len(p) >= 5 and len(n) >= 5:
            bins.append({"kl_range": [round(float(a0), 3), round(float(a1), 3)],
                         "auroc": round(auroc(p, n), 3),
                         "n_pos": len(p), "n_neg": len(n),
                         "med_kl_pos": round(float(np.median([k for k, _ in ck
                                                              if a0 <= k < a1])), 3),
                         "med_kl_neg": round(float(np.median([k for k, _ in rk
                                                              if a0 <= k < a1])), 3)})
    M["kl_binned_self_vs_random"] = bins
    binned = [b["auroc"] for b in bins]
    M["kl_binned_max"] = max(binned) if binned else None

    out[model] = M
    print(f"{model:22s} self {self_a:.3f} | O1 {o1_a:.3f} | probe {o2_a:.3f} | "
          f"R {M['recovery_R']} | d(self-O1) {M['delta_self_minus_o1']} | "
          f"binned max {M['kl_binned_max']}")

json.dump(out, open("prereg_primary.json", "w"), indent=1)
print("\nwrote prereg_primary.json")
