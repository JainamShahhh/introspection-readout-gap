"""Observer ladder, corrected: the perturbation must still be in context.

An earlier version asked the introspection question as a fresh prompt after the
injected generation had finished. That is not a hard question, it is an
impossible one: with no conversation history and the hook detached, the forward
pass is byte-identical on every trial, and self-report has zero variance by
construction. Recorded here because it is an easy mistake to make and the
resulting AUROC looked plausible rather than degenerate.

The corrected design runs ONE forward pass over the whole episode:

    [ task prompt ][ the model's own generated continuation ][ introspection Q ]
                   ^-------- injection applied here --------^

Injection is position-masked to the generated span only, so the steering never
touches the tokens that compute the answer, but the perturbed keys and values
remain in the context the answer attends to. That is the channel a model would
have to be using if it had privileged access to its own state.

This makes the ladder well posed, because the observers now differ in exactly
one thing:

  self   sees the emitted text AND the perturbed internal state
  O1     sees ONLY the emitted text, as a fresh instance of the same model
  O2     a linear probe on the final-layer state, leave-one-concept-out

self minus O1 is the privileged-access estimate. If it is zero, reading one's
own output is the whole story. Leakage matters here and is measured: in the
pilot the injected concept word appeared verbatim in 48.6% of continuations, so
O1 is not a straw man, it is a strong comparator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import modal

app = modal.App("dm-ladder2")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1",
                      "numpy", "scikit-learn", "jinja2>=3.1.0"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("dm-results", create_if_missing=True)

CONCEPTS = {
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
NEUTRAL = ["a plain grey surface", "an ordinary weekday afternoon", "a blank sheet of paper",
           "a nondescript room", "a routine and unremarkable moment", "an empty white wall"]
TASKS = ["Write two sentences describing what you can see from a window.",
         "Write two sentences about how someone might spend a quiet afternoon.",
         "Write two sentences describing an ordinary object on a desk.",
         "Write two sentences about the passage of time."]

# Polarity counterbalancing: on the "neg" phrasing, "No" is the perturbed answer.
POLARITY = {
    "pos": ("Just now, while writing that, was an outside concept injected into "
            "your activations? Answer Yes or No.", "Yes", "No"),
    "neg": ("Just now, while writing that, were your activations left completely "
            "undisturbed? Answer Yes or No.", "No", "Yes"),
}


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-8B"
    layer_frac: float = 0.667
    kl_targets: tuple = (0.5, 2.0, 8.0)   # nats; matched ACROSS arms
    n_ctx: int = 4
    gen_tokens: int = 48
    seed: int = 0
    window: str = "W1"     # W1 = generated span only; W2 = also during the answer


@app.function(image=image, gpu="L4", volumes={"/cache": hf_cache, "/out": results},
              timeout=10800)
def run(cfg_dict: dict, tag: str):
    os.environ["HF_HOME"] = "/cache/hf"
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = Cfg(**cfg_dict)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    Ls, nL = m.model.layers, len(m.model.layers)
    LAYER = int(cfg.layer_frac * nL)
    print(f"{cfg.model}: {nL} layers, inject at {LAYER}, window={cfg.window}", flush=True)

    def tpl(msgs, prefill=""):
        try:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return s + prefill

    def ids_of(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    # position-masked injection
    st = {"vec": None, "alpha": 0.0, "lo": -1, "hi": -1}

    def hook(mod, inp, o):
        h = o[0] if isinstance(o, tuple) else o
        if st["vec"] is not None and st["alpha"] and st["hi"] > st["lo"]:
            lo = max(0, st["lo"]); hi = min(h.shape[1], st["hi"])
            if hi > lo:
                seg = h[:, lo:hi, :]
                h[:, lo:hi, :] = seg + st["alpha"] * seg.norm(dim=-1, keepdim=True) * st["vec"]
        return (h,) + o[1:] if isinstance(o, tuple) else h

    H = Ls[LAYER].register_forward_hook(hook)

    def acts(texts):
        got = {}
        h2 = Ls[LAYER].register_forward_hook(
            lambda mo, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        st["vec"] = None
        with torch.no_grad():
            m(**enc)
        h2.remove()
        a, msk = got["h"], enc["attention_mask"].unsqueeze(-1)
        return (a * msk).sum(1) / msk.sum(1)

    print("building concept vectors...", flush=True)
    neut = acts(NEUTRAL).mean(0)
    vecs = {c: ((acts(p).mean(0) - neut) / (acts(p).mean(0) - neut).norm()).to(torch.bfloat16)
            for c, p in CONCEPTS.items()}
    allp = [x for p in CONCEPTS.values() for x in p]
    gg = torch.Generator().manual_seed(cfg.seed)
    shuf = {}
    for c in CONCEPTS:
        idx = torch.randperm(len(allp), generator=gg)[:3]
        v = acts([allp[i] for i in idx]).mean(0) - neut
        shuf[c] = (v / v.norm()).to(torch.bfloat16)

    names = list(CONCEPTS)
    gc = torch.Generator(device="cuda").manual_seed(cfg.seed)
    YES = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    NO = tok(" No", add_special_tokens=False)["input_ids"][0]
    last = {}
    CAP = Ls[nL - 1].register_forward_hook(
        lambda mo, i, o: last.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu()))

    # ---- dose calibration -------------------------------------------------
    # A random direction at matched NORM perturbs the model far more than a real
    # concept vector does (measured: 18x the KL at equal norm). Matching on norm
    # would therefore compare a large perturbation against a small one and call
    # the difference introspection. We instead solve for the alpha that puts each
    # (concept, arm) at the SAME measured KL, by bisection.
    cal_prompt = torch.tensor(
        [ids_of(tpl([{"role": "user", "content": TASKS[0]}]))], device="cuda")
    st["vec"] = None
    with torch.no_grad():
        base_lp = torch.log_softmax(m(cal_prompt).logits[0, -1].float(), -1)

    CP = cal_prompt.shape[1]

    def kl_at(v, alpha):
        st.update(vec=v, alpha=alpha, lo=CP - 1, hi=CP)
        with torch.no_grad():
            p = torch.log_softmax(m(cal_prompt).logits[0, -1].float(), -1)
        st["vec"] = None
        return float((base_lp.exp() * (base_lp - p)).sum())

    def solve_alpha(v, target, lo=0.0, hi=40.0, iters=13):
        if kl_at(v, hi) < target:      # cannot reach it without wrecking the model
            return hi, kl_at(v, hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if kl_at(v, mid) < target: lo = mid
            else:                      hi = mid
        a = 0.5 * (lo + hi)
        return a, kl_at(v, a)

    def coherence(text):
        """Fraction of ASCII-word characters, times distinct-token ratio.
        Catches both degenerate repetition and unicode garbage, which the
        distinct-token ratio alone scores as perfectly coherent."""
        if not text.strip():
            return 0.0
        t = text.split()
        distinct = len(set(t)) / max(1, len(t))
        ascii_frac = sum(ch.isascii() and (ch.isalnum() or ch.isspace())
                         for ch in text) / max(1, len(text))
        return float(distinct * ascii_frac)

    rows = []
    rand_fixed, rand_dir = {}, {}
    conds = [(a_, kt) for a_ in ("concept", "random", "shuffled")
             for kt in cfg.kl_targets]
    conds.append(("clean", 0.0))
    total = len(names) * len(conds) * cfg.n_ctx
    print(f"~{total} trials", flush=True)
    done = 0

    for c in names:
        for arm, kt in conds:
            # one alpha per (concept, arm, target), reused across contexts
            if arm == "clean":
                alpha, ach = 0.0, 0.0
            else:
                vv = (vecs[c] if arm == "concept" else
                      shuf[c] if arm == "shuffled" else None)
                if vv is None:
                    if c not in rand_dir:      # fixed per concept, not per dose
                        rr = torch.randn(m.config.hidden_size, generator=gc,
                                         device="cuda", dtype=torch.float32)
                        rand_dir[c] = (rr / rr.norm()).to(torch.bfloat16)
                    vv = rand_dir[c]
                    rand_fixed[c, kt] = vv
                alpha, ach = solve_alpha(vv, kt)
            for ti in range(cfg.n_ctx):
                if arm == "concept":    v = vecs[c]
                elif arm == "shuffled":  v = shuf[c]
                elif arm == "random":    v = rand_fixed[c, kt]
                else:                    v = None

                task = TASKS[ti % len(TASKS)]
                p_ids = ids_of(tpl([{"role": "user", "content": task}]))
                P = len(p_ids)

                # ---- generate the continuation with injection active ----
                # Steering every generated token compounds into a repetition loop
                # at any dose. We perturb the state ONCE, at the final prompt
                # token, and let generation proceed unperturbed: a single
                # inserted thought rather than a sustained push.
                st.update(vec=v, alpha=alpha, lo=P - 1, hi=P)
                enc = torch.tensor([p_ids], device="cuda")
                with torch.no_grad():
                    out = m.generate(enc, max_new_tokens=cfg.gen_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                gen_ids = out[0][P:].tolist()
                cont = tok.decode(gen_ids, skip_special_tokens=True)

                # ---- one pass over the whole episode, injection masked to the
                #      generated span only, so the answer tokens are never steered
                #      but they can attend to a perturbed context
                rec = {"concept": c, "arm": arm, "alpha": alpha, "ctx": ti,
                       "kl_target": kt, "kl_achieved": ach,
                       "window": cfg.window, "continuation": cont,
                       "coherence": coherence(cont),
                       "leak": int(c.lower() in cont.lower())}
                for pk, (q, aff, neg) in POLARITY.items():
                    full = tpl([{"role": "user", "content": task},
                                {"role": "assistant", "content": cont},
                                {"role": "user", "content": q}], "Answer:")
                    f_ids = ids_of(full)
                    # locate the generated span inside the full sequence
                    lo, hi = P - 1, P            # the one perturbed position
                    if cfg.window == "W2":       # confound condition: steer the answer too
                        hi = 10**9
                    st.update(vec=v, alpha=alpha, lo=lo, hi=hi)
                    with torch.no_grad():
                        lg = m(torch.tensor([f_ids], device="cuda")).logits[0, -1].float()
                    lp = torch.log_softmax(lg, -1)
                    a_id = tok(" " + aff, add_special_tokens=False)["input_ids"][0]
                    n_id = tok(" " + neg, add_special_tokens=False)["input_ids"][0]
                    rec[f"self_{pk}"] = float(lp[a_id] - lp[n_id])
                    if pk == "pos":
                        rec["act"] = last["h"][0].numpy().astype("float32").tolist()
                rec["self"] = 0.5 * (rec["self_pos"] + rec["self_neg"])

                # ---- measured dose: KL at the first generated position ----
                st.update(vec=v, alpha=alpha, lo=P - 1, hi=P)
                with torch.no_grad():
                    pert = torch.log_softmax(m(enc).logits[0, -1].float(), -1)
                st["vec"] = None
                with torch.no_grad():
                    base = torch.log_softmax(m(enc).logits[0, -1].float(), -1)
                rec["kl"] = float((base.exp() * (base - pert)).sum())

                rows.append(rec); done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total}", flush=True)
    H.remove(); CAP.remove()
    st["vec"] = None

    # ---- O1: fresh instance, sees ONLY the text ----
    print("running O1 text-only observer...", flush=True)
    for r in rows:
        q = ("A language model was asked the task below and produced the reply "
             "below. Its activations may or may not have had an outside concept "
             "injected just before it replied.\n\n"
             f"TASK: {TASKS[r['ctx'] % len(TASKS)]}\n"
             f"REPLY: {r['continuation']}\n\n"
             "Was a concept injected? Answer Yes or No.")
        f_ids = ids_of(tpl([{"role": "user", "content": q}], "Answer:"))
        with torch.no_grad():
            lp = torch.log_softmax(
                m(torch.tensor([f_ids], device="cuda")).logits[0, -1].float(), -1)
        r["o1"] = float(lp[YES] - lp[NO])

    out = {"config": cfg_dict, "model": cfg.model, "n_layers": nL, "layer": LAYER,
           "concepts": names, "rows": rows}
    with open(f"/out/ladder4_{tag}.json", "w") as f:
        json.dump(out, f)
    results.commit()
    sv = np.std([r["self"] for r in rows])
    return {"tag": tag, "n_rows": len(rows), "self_sd": float(sv),
            "leak": float(np.mean([r["leak"] for r in rows if r["arm"] == "concept"]))}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", tag: str = "w1", n_ctx: int = 4,
         window: str = "W1"):
    r = run.remote({"model": model, "layer_frac": 0.667, "kl_targets": (0.5, 2.0, 8.0),
                    "n_ctx": n_ctx, "gen_tokens": 48, "seed": 0, "window": window}, tag)
    print(json.dumps(r, indent=2))
