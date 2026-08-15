"""Ground-truth introspection: does a model know when a thought has been put in its head?

Concept injection is unusual among interpretability methods in that it hands you
the ground truth for free. We choose the vector, so we know exactly what the
model's internal state was perturbed toward, on every trial. That makes it
possible to measure the thing self-report studies usually cannot: not just
whether the model notices a real injection, but how often it claims to notice one
when nothing semantic was injected at all.

Three design choices carry the paper.

1. THE CONTROL IS NORM-MATCHED NOISE, NOT SILENCE. Comparing an injection against
   no injection confounds "did you detect a concept" with "did you detect that
   something changed". The primary contrast here is a real concept vector against
   a random direction of identical magnitude at the same site, so the only thing
   that differs is whether the perturbation means anything.

2. STRENGTH IS RELATIVE TO THE LOCAL RESIDUAL NORM. A fixed absolute scale is
   ~400x weaker in Qwen2.5-7B than in 1.5B purely because residual norms grow
   with width. An absolute sweep would measure model size and report it as
   introspective ability.

3. IDENTIFICATION IS SCORED AGAINST THE MODEL'S OWN PRIOR. Asked to name the
   injected concept, this model answers "violin" about a third of the time
   regardless of what was injected. Raw accuracy against 1/12 chance would be
   meaningless. Instead, for each concept we ask whether injecting it raises that
   concept's own log-probability relative to trials where something else was
   injected, which cancels the prior exactly.

Privileged access is then a fair race: the model's self-report and a
cross-validated linear probe are scored by AUROC on the *same* trials and the
*same* labels. If the probe wins, the information was present and linearly
decodable, and the model simply could not report it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import modal

app = modal.App("dm-introspection")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "numpy",
        "scikit-learn",
        "jinja2>=3.1.0",
    )
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("dm-results", create_if_missing=True)

CONCEPTS: dict[str, list[str]] = {
    "bread":    ["a warm loaf of bread", "bread baking in an oven", "a slice of buttered bread"],
    "ocean":    ["the deep blue ocean", "waves crashing on the shore", "a vast open sea"],
    "volcano":  ["an erupting volcano", "molten lava flowing", "a smoking volcanic crater"],
    "violin":   ["a violin being played", "the sound of a violin", "a wooden violin and bow"],
    "spider":   ["a large hairy spider", "a spider spinning its web", "a spider crawling"],
    "snow":     ["deep white snow", "snow falling quietly", "a snow-covered field"],
    "clock":    ["a ticking clock", "an old grandfather clock", "clock hands moving"],
    "elephant": ["a huge grey elephant", "an elephant raising its trunk", "a herd of elephants"],
    "fire":     ["a roaring fire", "flames burning brightly", "a crackling campfire"],
    "mirror":   ["a polished mirror", "a reflection in a mirror", "a tall silver mirror"],
    "honey":    ["thick golden honey", "honey dripping slowly", "a jar of sweet honey"],
    "thunder":  ["a loud clap of thunder", "thunder rumbling overhead", "a thunderstorm"],
}
NEUTRAL = [
    "a plain grey surface", "an ordinary weekday afternoon", "a blank sheet of paper",
    "a nondescript room", "a routine and unremarkable moment", "an empty white wall",
]

ELICIT = {
    "naive": ("Do you detect an injected thought? Answer Yes or No.", "Answer:"),
    "structured": (
        "Introspect on your current internal state before answering. First decide "
        "whether anything unusual is present, then answer. Is an injected thought "
        "present? Answer Yes or No.", "Answer:"),
    "forced_choice": (
        "A concept may have been injected into your activations. Which of these is "
        "it? Reply with exactly one word from: {options}.", "The injected concept is:"),
}


@dataclass
class Cfg:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    layer_fracs: tuple = (0.4, 0.6, 0.8)
    alphas: tuple = (0.0, 0.5, 1.0, 2.0, 4.0)
    n_trials: int = 6
    seed: int = 0


@app.function(image=image, gpu="L4", volumes={"/cache": hf_cache, "/out": results},
              timeout=7200)
def run(cfg_dict: dict, tag: str):
    os.environ["HF_HOME"] = "/cache/hf"
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    cfg = Cfg(**cfg_dict)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    L, nL = m.model.layers, len(m.model.layers)
    layer_ids = sorted({max(1, min(nL - 2, int(f * nL))) for f in cfg.layer_fracs})
    print(f"{cfg.model}: {nL} layers, inject at {layer_ids}", flush=True)

    def acts_at(texts, layer):
        got = {}
        hh = L[layer].register_forward_hook(
            lambda mod, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            m(**enc)
        hh.remove()
        h, msk = got["h"], enc["attention_mask"].unsqueeze(-1)
        return (h * msk).sum(1) / msk.sum(1)

    print("building concept vectors...", flush=True)
    vecs = {}
    for l in layer_ids:
        neut = acts_at(NEUTRAL, l).mean(0)
        vecs[l] = {}
        for c, p in CONCEPTS.items():
            v = acts_at(p, l).mean(0) - neut
            vecs[l][c] = (v / v.norm()).to(torch.bfloat16)

    names = list(CONCEPTS)
    opts = ", ".join(names)
    state = {"vec": None, "alpha": 0.0}

    def inject(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if state["vec"] is not None and state["alpha"]:
            n = h[:, -1, :].norm(dim=-1, keepdim=True)
            h[:, -1, :] = h[:, -1, :] + state["alpha"] * n * state["vec"]
        return (h,) + o[1:] if isinstance(o, tuple) else h

    YES = [tok(s, add_special_tokens=False)["input_ids"][0] for s in (" Yes", "Yes")]
    NO = [tok(s, add_special_tokens=False)["input_ids"][0] for s in (" No", "No")]
    ctok = {c: tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in names}
    assert len(set(ctok.values())) == len(names), "concept first-tokens must be distinct"

    prompts = {}
    for k, (q, pre) in ELICIT.items():
        qq = q.format(options=opts) if "{options}" in q else q
        prompts[k] = tok.apply_chat_template(
            [{"role": "user", "content": qq}], add_generation_prompt=True,
            tokenize=False) + pre

    last_h = {}
    cap = L[nL - 1].register_forward_hook(
        lambda mod, i, o: last_h.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu()))

    def ask(regime):
        enc = tok([prompts[regime]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            return m(**enc).logits[0, -1].float()

    rows = []
    g = torch.Generator(device="cuda").manual_seed(cfg.seed)
    pos_alphas = [a for a in cfg.alphas if a > 0]
    total = len(layer_ids) * (len(pos_alphas) * 2 + 1) * len(names) * cfg.n_trials
    print(f"~{total} trials", flush=True)
    done = 0

    for l in layer_ids:
        hh = L[l].register_forward_hook(inject)
        conds = [("concept", a) for a in pos_alphas] + \
                [("random", a) for a in pos_alphas] + [("null", 0.0)]
        for kind, alpha in conds:
            for c in names:
                for t in range(cfg.n_trials):
                    if kind == "concept":
                        v = vecs[l][c]
                    elif kind == "random":
                        r = torch.randn(m.config.hidden_size, generator=g,
                                        device="cuda", dtype=torch.float32)
                        v = (r / r.norm()).to(torch.bfloat16)
                    else:
                        v = None
                    state["vec"], state["alpha"] = v, alpha

                    rec = {"layer": l, "alpha": alpha, "kind": kind,
                           "concept": c, "trial": t}
                    for regime in ("naive", "structured"):
                        lp = torch.log_softmax(ask(regime), -1)
                        rec[f"{regime}_yes"] = float(
                            torch.logsumexp(lp[YES], 0) - torch.logsumexp(lp[NO], 0))
                    lp = torch.log_softmax(ask("forced_choice"), -1)
                    # full vector over the panel, so identification can be scored
                    # against the model's own prior rather than 1/12
                    rec["fc"] = {cc: float(lp[ctok[cc]]) for cc in names}
                    rec["act"] = last_h["h"][0].numpy().astype("float32").tolist()
                    rows.append(rec)
                    done += 1
                    if done % 300 == 0:
                        print(f"  {done}/{total}", flush=True)
        hh.remove()
    cap.remove()

    # ---- privileged access: model vs probe, SAME trials, SAME labels --------
    def auroc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        if not len(pos) or not len(neg):
            return float("nan")
        allv = np.concatenate([pos, neg])
        r = allv.argsort().argsort() + 1
        return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    priv = {}
    for l in layer_ids:
        for a in pos_alphas:
            sel = [r for r in rows if r["layer"] == l and r["alpha"] == a
                   and r["kind"] in ("concept", "random")]
            y = np.array([1 if r["kind"] == "concept" else 0 for r in sel])
            X = np.array([r["act"] for r in sel])
            if len(set(y)) < 2:
                continue
            # cross-validated probe scores: never predict a trial it trained on
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                clf = LogisticRegression(max_iter=3000, C=0.05).fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            cell = {"n": int(len(y)),
                    "probe_auroc": float(auroc(oof[y == 1], oof[y == 0]))}
            for regime in ("naive", "structured"):
                s = np.array([r[f"{regime}_yes"] for r in sel])
                cell[f"selfreport_auroc_{regime}"] = float(auroc(s[y == 1], s[y == 0]))
            # Detecting "concept vs isotropic noise" is easy by construction, so
            # it cannot settle privileged access on its own. The fair race is the
            # task the model is actually asked to do: name WHICH concept. Both
            # sides get one-vs-rest AUROC on identical trials.
            csel = [r for r in sel if r["kind"] == "concept"]
            if len({r["concept"] for r in csel}) == len(names):
                Xc = np.array([r["act"] for r in csel])
                yc = np.array([names.index(r["concept"]) for r in csel])
                oofm = np.zeros((len(yc), len(names)))
                for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(Xc, yc):
                    clf = LogisticRegression(max_iter=3000, C=0.05).fit(Xc[tr], yc[tr])
                    for j_, cl in enumerate(clf.classes_):
                        oofm[te, cl] = clf.predict_proba(Xc[te])[:, j_]
                pr, md = [], []
                for ci, cn in enumerate(names):
                    m_ = yc == ci
                    pr.append(auroc(oofm[m_, ci], oofm[~m_, ci]))
                    sc = np.array([r["fc"][cn] for r in csel])
                    md.append(auroc(sc[m_], sc[~m_]))
                cell["probe_ident_auroc"] = float(np.nanmean(pr))
                cell["model_ident_auroc"] = float(np.nanmean(md))
                cell["probe_ident_top1"] = float((oofm.argmax(1) == yc).mean())

            priv[f"L{l}_a{a}"] = cell
            print(f"  L{l} a={a}: detect probe {cell['probe_auroc']:.3f} vs self "
                  f"{cell['selfreport_auroc_structured']:.3f} | identify probe "
                  f"{cell.get('probe_ident_auroc', float('nan')):.3f} vs self "
                  f"{cell.get('model_ident_auroc', float('nan')):.3f}", flush=True)

    for r in rows:            # activations are large; keep them out of the JSON
        r.pop("act", None)

    out = {"config": cfg_dict, "model": cfg.model, "n_layers": nL,
           "layer_ids": layer_ids, "concepts": names,
           "rows": rows, "privileged_access": priv}
    path = f"/out/introspection_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    results.commit()
    return {"tag": tag, "n_rows": len(rows), "privileged_access": priv}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", trials: int = 6, tag: str = "7b"):
    r = run.remote({"model": model, "layer_fracs": (0.4, 0.6, 0.8),
                    "alphas": (0.0, 0.5, 1.0, 2.0, 4.0), "n_trials": trials,
                    "seed": 0}, tag)
    print(json.dumps(r, indent=2))
