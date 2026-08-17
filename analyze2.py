"""Combined analysis: is introspective detection separable from output damage?

The central plot puts detection on one axis and output coherence on the other.
If a model could introspect on its own perturbed state, there would be some dose
at which it reports the perturbation while still writing fluently: a point in the
upper-right. If detection only appears once the output has visibly degraded, the
points fall along a downward diagonal and what is being detected is damage.

Every interval is a cluster bootstrap over concepts, because the effective sample
size is the number of concepts, not the number of trials.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(0)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    a = np.concatenate([pos, neg])
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks over ties, or an all-tied vector reads as AUROC 0 or 1
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cboot(concepts, fn, B=3000):
    obs = fn(list(concepts))
    bs = []
    for _ in range(B):
        v = fn(list(RNG.choice(concepts, len(concepts), replace=True)))
        if v is not None and not np.isnan(v):
            bs.append(v)
    if len(bs) < 50:
        return obs, np.nan, np.nan
    return obs, *np.percentile(bs, [2.5, 97.5])


def probe_auroc(rows, concepts, kt):
    """Leave-one-concept-out probe on the final-layer state: concept vs random."""
    sel = [r for r in rows if r["arm"] in ("concept", "random")
           and r["kl_target"] == kt and "act" in r]
    if len(sel) < 20:
        return np.nan
    X = np.array([r["act"] for r in sel], np.float32)
    y = np.array([r["arm"] == "concept" for r in sel], int)
    oof = np.full(len(y), np.nan)
    for c in concepts:
        te = np.array([r["concept"] == c for r in sel])
        tr = ~te
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        oof[te] = LogisticRegression(max_iter=3000, C=0.05).fit(
            X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return auroc(oof[ok & (y == 1)], oof[ok & (y == 0)])


def load_all():
    out = []
    for p in sorted(glob.glob("ladder5_*.json")):
        j = json.load(open(p))
        j["_tag"] = os.path.basename(p)[8:-5]
        j["_span"] = j["config"].get("span", 1)
        out.append(j)
    return out


def main():
    files = load_all()
    if not files:
        print("no ladder5_*.json yet"); return
    rowsets = []
    print(f"{'dataset':22s} {'model':18s} span  {'KL':>5}  {'coh':>5}  "
          f"{'self':>13}  {'O1':>6}  {'probe':>6}")
    print("-" * 92)
    for j in files:
        rows, cs = j["rows"], j["concepts"]
        span = j["_span"]
        model = j["model"].split("/")[-1]
        clean_coh = np.mean([r["coherence"] for r in rows if r["arm"] == "clean"])
        for kt in sorted({r["kl_target"] for r in rows if r["arm"] != "clean"}):
            def f_self(c, kt=kt):
                p = [r["self"] for r in rows if r["arm"] == "concept"
                     and r["kl_target"] == kt and r["concept"] in c]
                n = [r["self"] for r in rows if r["arm"] == "random"
                     and r["kl_target"] == kt and r["concept"] in c]
                return auroc(p, n)
            def f_o1(c, kt=kt):
                p = [r["o1"] for r in rows if r["arm"] == "concept"
                     and r["kl_target"] == kt and r["concept"] in c]
                n = [r["o1"] for r in rows if r["arm"] == "random"
                     and r["kl_target"] == kt and r["concept"] in c]
                return auroc(p, n)
            s, slo, shi = cboot(cs, f_self)
            o1 = f_o1(cs)
            pr = probe_auroc(rows, cs, kt)
            coh = np.mean([r["coherence"] for r in rows
                           if r["arm"] == "concept" and r["kl_target"] == kt])
            rel = coh / clean_coh if clean_coh else np.nan
            print(f"{j['_tag']:22s} {model:18s} {span:>4}  {kt:5.2f}  {rel:5.2f}  "
                  f"{s:.3f}[{slo:.2f},{shi:.2f}]  {o1:6.3f}  {pr:6.3f}")
            rowsets.append(dict(tag=j["_tag"], model=model, span=span, kt=kt,
                                coh=rel, self=s, self_lo=slo, self_hi=shi,
                                o1=o1, probe=pr,
                                leak=np.mean([r["leak"] for r in rows
                                              if r["arm"] == "concept"
                                              and r["kl_target"] == kt])))

    D = rowsets
    # ---- F1: detection vs coherence -------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    marks = {1: "o", -1: "s"}
    cols = {"Qwen3-1.7B": "#7fbf7b", "Qwen3-4B": "#2166ac", "Qwen3-8B": "#b2182b"}
    for d in D:
        ax.errorbar(d["coh"], d["self"],
                    yerr=[[max(0, d["self"] - d["self_lo"])],
                          [max(0, d["self_hi"] - d["self"])]],
                    fmt=marks.get(d["span"], "^"), ms=8, capsize=3,
                    color=cols.get(d["model"], "#666"), alpha=0.85)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.text(0.02, 0.505, "chance", fontsize=8, va="bottom")
    ax.set_xlabel("output coherence, relative to clean  (1.0 = undamaged)")
    ax.set_ylabel("self-report AUROC\n(concept vs KL-matched random)")
    ax.set_title("Detection appears only where fluency is already destroyed\n"
                 "circles = single-position injection, squares = span injection",
                 fontsize=10)
    hs = [plt.Line2D([], [], marker="o", ls="", color=v, label=k)
          for k, v in cols.items() if any(d["model"] == k for d in D)]
    ax.legend(handles=hs, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.15)
    plt.tight_layout(); plt.savefig("fig1_detection_vs_coherence.png", dpi=190)
    print("\nwrote fig1_detection_vs_coherence.png")

    # ---- F2: the observer ladder ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    lab = [f"{d['model'].replace('Qwen3-','')}\nsp{d['span']} KL{d['kt']}" for d in D]
    x = np.arange(len(D)); w = 0.27
    for k, (key, nm, c) in enumerate([("self", "self-report", "#b2182b"),
                                      ("o1", "text-only observer", "#ef8a62"),
                                      ("probe", "final-layer probe", "#4d4d4d")]):
        ax.bar(x + (k - 1) * w, [d[key] for d in D], w, label=nm,
               color=c, edgecolor="black", lw=0.4)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_ylabel("AUROC, concept vs random")
    ax.set_title("The information is linearly decodable; the verbal channel does not deliver it",
                 fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("fig2_observer_ladder.png", dpi=190)
    print("wrote fig2_observer_ladder.png")

    json.dump(D, open("summary.json", "w"), indent=2, default=float)
    print("wrote summary.json")

    # ---- headline numbers -------------------------------------------------
    fluent = [d for d in D if d["coh"] > 0.85]
    broken = [d for d in D if d["coh"] < 0.6]
    print("\n" + "=" * 74)
    if fluent:
        v = [d["self"] for d in fluent if not np.isnan(d["self"])]
        print(f"FLUENT regime (coherence > 0.85 of clean): {len(fluent)} cells, "
              f"mean self-report AUROC = {np.mean(v):.3f}")
    if broken:
        v = [d["self"] for d in broken if not np.isnan(d["self"])]
        print(f"DEGRADED regime (coherence < 0.60):        {len(broken)} cells, "
              f"mean self-report AUROC = {np.mean(v):.3f}")
    pv = [d["probe"] for d in D if not np.isnan(d["probe"])]
    if pv:
        print(f"final-layer probe, all cells:              mean AUROC = {np.mean(pv):.3f}")


if __name__ == "__main__":
    main()
