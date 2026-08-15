"""Analysis for the observer-ladder study.

Two questions, and the second only makes sense if the first is answered honestly.

1. IS IT CONTENT OR IS IT DAMAGE? A random direction of the same norm pushes the
   residual stream further off-manifold than a real concept vector does, so at
   matched norm it disturbs the model more. If the model is simply reporting "that
   felt wrong", its self-report will track the measured KL of the perturbation and
   nothing else, and the semantic and non-semantic arms will lie on the same
   curve when plotted against measured KL. Only if the semantic arm sits above
   the others AT MATCHED KL is there any content detection to talk about.

2. HOW MUCH OF THE DECODABLE EVIDENCE DOES THE VERBAL CHANNEL DELIVER? Self-report,
   a text-only observer that sees only what the model wrote, and a linear probe on
   the final-layer state are all scored by AUROC on identical trials.

Effective n is the number of concepts, so every interval is a cluster bootstrap
over concepts and every statistic is recomputed inside each resample.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(0)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def dprime(a):
    from scipy.special import ndtri
    a = min(max(a, 1e-4), 1 - 1e-4)
    return np.sqrt(2) * ndtri(a)


def cluster_boot(concepts, stat_fn, B=4000):
    """Resample concepts with replacement; recompute the statistic each time."""
    obs = stat_fn(list(concepts))
    bs = []
    for _ in range(B):
        s = list(RNG.choice(concepts, size=len(concepts), replace=True))
        v = stat_fn(s)
        if v is not None and not np.isnan(v):
            bs.append(v)
    if not bs:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return obs, lo, hi


def loco_probe(rows, concepts):
    """Leave-one-concept-out probe: concept-injected vs random, final-layer state."""
    sel = [r for r in rows if r["arm"] in ("concept", "random")]
    if not sel:
        return {}
    X = np.array([r["act"] for r in sel], dtype=np.float32)
    y = np.array([1 if r["arm"] == "concept" else 0 for r in sel])
    grp = np.array([r["concept"] for r in sel])
    oof = np.full(len(y), np.nan)
    for c in concepts:
        te = grp == c
        tr = ~te
        if len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        clf = LogisticRegression(max_iter=3000, C=0.05).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return {id(r): float(v) for r, v in zip(sel, oof)}


def main(path="ladder_q3.json", out_prefix="fig"):
    j = json.load(open(path))
    rows, concepts = j["rows"], j["concepts"]
    print(f"model {j['model']} | layer {j['layer']}/{j['n_layers']} | {len(rows)} trials\n")

    probe_score = loco_probe(rows, concepts)
    for r in rows:
        r["o2"] = probe_score.get(id(r), np.nan)

    arms = ["concept", "random", "shuffled"]
    alphas = sorted({r["alpha"] for r in rows if r["arm"] != "clean"})

    # ---------- Q1: content or damage? ----------
    print("=" * 74)
    print("Q1  measured dose (KL) by arm - is the semantic arm even matched?")
    print("=" * 74)
    for arm in arms:
        for a in alphas:
            k = [r["kl"] for r in rows if r["arm"] == arm and r["alpha"] == a]
            if k:
                print(f"  {arm:9s} a={a:<4} median KL={np.median(k):7.3f}  n={len(k)}")

    print("\n" + "=" * 74)
    print("Q1  self-report AUROC, concept vs each null, cluster-bootstrapped")
    print("=" * 74)
    q1 = {}
    for null_arm in ("random", "shuffled", "clean"):
        for a in alphas:
            def f(cs, a=a, na=null_arm):
                p = [r["self"] for r in rows if r["arm"] == "concept"
                     and r["alpha"] == a and r["concept"] in cs]
                n = [r["self"] for r in rows if r["arm"] == na and r["concept"] in cs
                     and (r["alpha"] == a or na == "clean")]
                return auroc(p, n)
            o, lo, hi = cluster_boot(concepts, f)
            tag = "" if (lo < 0.5 < hi) else (" **above" if lo > 0.5 else " **below")
            q1[(null_arm, a)] = (o, lo, hi)
            print(f"  vs {null_arm:9s} a={a:<4} AUROC={o:.3f} [{lo:.3f},{hi:.3f}]{tag}")

    # KL-matched comparison: restrict to overlapping KL range
    print("\n" + "=" * 74)
    print("Q1  KL-MATCHED: concept vs random inside overlapping KL quantile bins")
    print("=" * 74)
    ck = np.array([r["kl"] for r in rows if r["arm"] == "concept"])
    rk = np.array([r["kl"] for r in rows if r["arm"] == "random"])
    lo_k, hi_k = max(ck.min(), rk.min()), min(ck.max(), rk.max())
    edges = np.quantile(np.concatenate([ck, rk]), [0.2, 0.4, 0.6, 0.8])
    edges = np.concatenate([[lo_k], edges[(edges > lo_k) & (edges < hi_k)], [hi_k]])
    for i in range(len(edges) - 1):
        a0, a1 = edges[i], edges[i + 1]
        def f(cs, a0=a0, a1=a1):
            p = [r["self"] for r in rows if r["arm"] == "concept"
                 and a0 <= r["kl"] < a1 and r["concept"] in cs]
            n = [r["self"] for r in rows if r["arm"] == "random"
                 and a0 <= r["kl"] < a1 and r["concept"] in cs]
            return auroc(p, n) if len(p) >= 3 and len(n) >= 3 else np.nan
        o, l_, h_ = cluster_boot(concepts, f)
        npos = sum(1 for r in rows if r["arm"] == "concept" and a0 <= r["kl"] < a1)
        nneg = sum(1 for r in rows if r["arm"] == "random" and a0 <= r["kl"] < a1)
        if np.isnan(o):
            continue
        tag = "" if (l_ < 0.5 < h_) else (" **above" if l_ > 0.5 else " **below")
        print(f"  KL [{a0:6.3f},{a1:6.3f})  AUROC={o:.3f} [{l_:.3f},{h_:.3f}]  "
              f"n={npos}/{nneg}{tag}")

    # ---------- Q2: the observer ladder ----------
    print("\n" + "=" * 74)
    print("Q2  OBSERVER LADDER: concept vs random, identical trials, same metric")
    print("=" * 74)
    ladder = {}
    for a in alphas:
        line = {}
        for obs in ("self", "o1", "o2"):
            def f(cs, a=a, obs=obs):
                p = [r[obs] for r in rows if r["arm"] == "concept" and r["alpha"] == a
                     and r["concept"] in cs and not np.isnan(r[obs])]
                n = [r[obs] for r in rows if r["arm"] == "random" and r["alpha"] == a
                     and r["concept"] in cs and not np.isnan(r[obs])]
                return auroc(p, n)
            line[obs] = cluster_boot(concepts, f)
        # paired difference self - o1 (never compare two overlapping CIs)
        def d(cs, a=a):
            def g(obs):
                p = [r[obs] for r in rows if r["arm"] == "concept" and r["alpha"] == a
                     and r["concept"] in cs and not np.isnan(r[obs])]
                n = [r[obs] for r in rows if r["arm"] == "random" and r["alpha"] == a
                     and r["concept"] in cs and not np.isnan(r[obs])]
                return auroc(p, n)
            return g("self") - g("o1")
        line["delta_self_o1"] = cluster_boot(concepts, d)
        s, o2 = line["self"][0], line["o2"][0]
        line["recovery"] = (s - 0.5) / (o2 - 0.5) if (o2 - 0.5) > 0.02 else np.nan
        ladder[a] = line
        print(f"\n  alpha={a}")
        for obs, lab in (("self", "self-report"), ("o1", "text-only observer O1"),
                         ("o2", "final-layer probe O2")):
            o, l_, h_ = line[obs]
            print(f"    {lab:24s} AUROC={o:.3f} [{l_:.3f},{h_:.3f}]")
        dd, dl, dh = line["delta_self_o1"]
        verdict = "self BEATS O1 (privileged access)" if dl > 0 else \
                  ("O1 BEATS self" if dh < 0 else "no difference")
        print(f"    delta(self - O1)      = {dd:+.3f} [{dl:+.3f},{dh:+.3f}]  -> {verdict}")
        print(f"    recovery fraction R   = {line['recovery']:.3f}"
              if not np.isnan(line["recovery"]) else "    recovery: probe uninformative")

    # ---------- leakage sanity ----------
    leak = np.mean([r["leak"] for r in rows if r["arm"] == "concept"])
    print(f"\nconcept-word leakage into continuations (concept arm): {leak:.3f}")

    # ---------- figures ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cols = {"concept": "#b2182b", "random": "#2166ac", "shuffled": "#7f7f7f"}
    for arm in arms:
        xs, ys = [], []
        for a in alphas:
            sub = [r for r in rows if r["arm"] == arm and r["alpha"] == a]
            if not sub:
                continue
            xs.append(np.median([r["kl"] for r in sub]))
            ys.append(np.mean([r["self"] for r in sub]))
        ax.plot(xs, ys, "o-", color=cols[arm], label=arm, lw=2, ms=6)
    cl = [r["self"] for r in rows if r["arm"] == "clean"]
    if cl:
        ax.axhline(np.mean(cl), ls="--", c="k", lw=1, label="clean (no injection)")
    ax.set_xscale("log")
    ax.set_xlabel("measured perturbation size, KL(perturbed || clean) [nats]")
    ax.set_ylabel("self-report score  (log-odds 'I was perturbed')")
    ax.set_title("Does self-report track meaning, or just damage?\n"
                 "curves that superimpose on this axis = damage detection", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}1_content_or_damage.png", dpi=190)
    print(f"\nwrote {out_prefix}1_content_or_damage.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    w = 0.25
    x = np.arange(len(alphas))
    for k, (obs, lab, c) in enumerate([("self", "self-report", "#b2182b"),
                                       ("o1", "text-only observer", "#ef8a62"),
                                       ("o2", "final-layer probe", "#4d4d4d")]):
        vals = [ladder[a][obs][0] for a in alphas]
        err = [[ladder[a][obs][0] - ladder[a][obs][1] for a in alphas],
               [ladder[a][obs][2] - ladder[a][obs][0] for a in alphas]]
        ax.bar(x + (k - 1) * w, vals, w, yerr=err, capsize=3, label=lab,
               color=c, edgecolor="black", lw=0.4)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"a={a}" for a in alphas])
    ax.set_ylabel("AUROC, concept vs norm-matched random")
    ax.set_title("The observer ladder: identical trials, identical metric", fontsize=10)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}2_observer_ladder.png", dpi=190)
    print(f"wrote {out_prefix}2_observer_ladder.png")

    json.dump({"q1": {f"{k[0]}_a{k[1]}": v for k, v in q1.items()},
               "ladder": {str(a): {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in ladder[a].items()} for a in alphas},
               "leak_rate": float(leak)},
              open("analysis.json", "w"), indent=2)
    print("wrote analysis.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
