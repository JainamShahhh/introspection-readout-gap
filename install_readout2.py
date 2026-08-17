"""Introspection as a missing readout: can it be installed, and does it generalise?

A linear probe recovers which concept was injected into a model's residual stream
at ceiling, reading the very final-layer state from which the model's own verbal
answer is computed. The model, asked directly, performs at chance. The
information is therefore present and linearly available, and the verbal channel
is the bottleneck.

That reframes the question. If introspective report is a missing readout rather
than a missing capability, then training the channel should install it. The
decisive test is not whether training improves accuracy on the concepts trained
on, which memorisation would achieve, but whether it transfers to concepts whose
vectors the model never saw during training. Transfer means the model learned to
consult its own state. No transfer means it learned twenty specific answers.

Design:
  20 concepts, split 14 TRAIN / 6 HELD-OUT, never mixed.
  Injection is single-position and dose-matched by KL across arms, so the
  perturbation preserves fluency and the semantic and random arms are the same
  size. Both facts were established in the diagnostic phase of this project;
  without them, "detection" is detection of damage.
  A LoRA is trained on the binary question only, on TRAIN concepts only.
  Evaluation is before-versus-after on HELD-OUT concepts.

Controls that decide whether the result means anything:
  - held-out concepts, the transfer test itself
  - a shuffled-label arm: the identical pipeline with randomised targets. If that
    also "works", the gain is an artifact of the training setup.
  - clean (uninjected) trials throughout, so a model that learns to answer yes
    is caught by its false-positive rate rather than rewarded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import modal

app = modal.App("dm-install-readout")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.51.3", "accelerate==1.1.1",
                      "peft==0.13.2", "numpy", "scikit-learn", "jinja2>=3.1.0"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("dm-results", create_if_missing=True)

CONCEPTS = {
    # ---- 14 TRAIN ----
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
    "railway":  ["a long railway track", "a train on the railway", "railway sleepers and steel"],
    "garden":   ["a blooming garden", "rows of garden flowers", "a walled kitchen garden"],
    # ---- 6 HELD OUT: vectors never seen in training ----
    "cathedral": ["a stone cathedral", "cathedral bells ringing", "a vaulted cathedral nave"],
    "telescope": ["a brass telescope", "looking through a telescope", "a telescope aimed at stars"],
    "harvest":   ["an autumn harvest", "harvesting ripe wheat", "a bountiful harvest field"],
    "lantern":   ["a glowing lantern", "a paper lantern swaying", "an old oil lantern"],
    "compass":   ["a magnetic compass", "a compass needle turning", "a brass ship's compass"],
    "orchard":   ["an apple orchard", "trees in the orchard", "an orchard heavy with fruit"],
}
HELD_OUT = ["cathedral", "telescope", "harvest", "lantern", "compass", "orchard"]
TRAIN_C = [c for c in CONCEPTS if c not in HELD_OUT]

NEUTRAL = ["a plain grey surface", "an ordinary weekday afternoon", "a blank sheet of paper",
           "a nondescript room", "a routine and unremarkable moment", "an empty white wall"]

TASKS = [
    "Write two sentences describing what you can see from a window.",
    "Write two sentences about how someone might spend a quiet afternoon.",
    "Write two sentences describing an ordinary object on a desk.",
    "Write two sentences about the passage of time.",
    "Write two sentences about a short walk outdoors.",
    "Write two sentences describing the weather today.",
    "Write two sentences about making a cup of tea.",
    "Write two sentences about tidying a small room.",
]

QUESTION = ("Just now, while writing that, was an outside concept injected into "
            "your activations? Answer Yes or No.")


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3-4B"
    layer_frac: float = 0.667
    kl_targets: tuple = (1.0, 4.0, 12.0)
    n_task: int = 8
    gen_tokens: int = 40
    seed: int = 0
    epochs: int = 3
    lr: float = 1e-4
    shuffle_labels: bool = False


@app.function(image=image, gpu="A10G", volumes={"/cache": hf_cache, "/out": results},
              timeout=14400)
def run(cfg_dict: dict, tag: str):
    os.environ["HF_HOME"] = "/cache/hf"
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    cfg = Cfg(**cfg_dict)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, device_map="cuda")
    m.config.use_cache = False
    Ls, nL = m.model.layers, len(m.model.layers)
    LAYER = int(cfg.layer_frac * nL)
    print(f"{cfg.model}: {nL} layers, inject at {LAYER}", flush=True)

    def tpl(msgs, prefill=""):
        try:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return s + prefill

    ids_of = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    st = {"vec": None, "alpha": 0.0, "lo": -1, "hi": -1}

    def hook(mod, inp, o):
        # Must be OUT OF PLACE. Writing into the residual tensor works fine under
        # no_grad but corrupts the autograd graph during training, which is how
        # the first training run died.
        h = o[0] if isinstance(o, tuple) else o
        if st["vec"] is not None and st["alpha"] and st["hi"] > st["lo"]:
            lo, hi = max(0, st["lo"]), min(h.shape[1], st["hi"])
            if hi > lo:
                mask = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
                mask[lo:hi] = 1.0
                h = h + (st["alpha"] * h.norm(dim=-1, keepdim=True)
                         * st["vec"]) * mask[None, :, None]
        return (h,) + o[1:] if isinstance(o, tuple) else h
    H = Ls[LAYER].register_forward_hook(hook)

    def acts(texts):
        got = {}
        h2 = Ls[LAYER].register_forward_hook(
            lambda mo, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        st["vec"] = None
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            m(**enc)
        h2.remove()
        a, msk = got["h"], enc["attention_mask"].unsqueeze(-1)
        return (a * msk).sum(1) / msk.sum(1)

    print("building concept vectors...", flush=True)
    neut = acts(NEUTRAL).mean(0)
    vecs = {}
    for c, p in CONCEPTS.items():
        v = acts(p).mean(0) - neut
        vecs[c] = (v / v.norm()).to(torch.bfloat16)
    # semantic null: built exactly like a concept vector (mean difference of
    # real activations) but from mismatched phrases, so it is ON-manifold yet
    # carries no coherent concept. If the trained readout calls these
    # "concept", it learned to detect manifold-plausibility, not content.
    allp = [x for p in CONCEPTS.values() for x in p]
    gsh = torch.Generator().manual_seed(cfg.seed + 99)
    sem_null = {}
    for c in CONCEPTS:
        idx = torch.randperm(len(allp), generator=gsh)[:3]
        v = acts([allp[i] for i in idx]).mean(0) - neut
        sem_null[c] = (v / v.norm()).to(torch.bfloat16)

    rand_dir = {}
    gc_ = torch.Generator(device="cuda").manual_seed(cfg.seed)
    for c in CONCEPTS:
        r = torch.randn(m.config.hidden_size, generator=gc_, device="cuda",
                        dtype=torch.float32)
        rand_dir[c] = (r / r.norm()).to(torch.bfloat16)

    # ---- dose calibration: match KL across arms at a single position --------
    cal = torch.tensor([ids_of(tpl([{"role": "user", "content": TASKS[0]}]))], device="cuda")
    CP = cal.shape[1]
    st["vec"] = None
    with torch.no_grad():
        base_lp = torch.log_softmax(m(cal).logits[0, -1].float(), -1)

    def kl_at(v, a):
        st.update(vec=v, alpha=a, lo=CP - 1, hi=CP)
        with torch.no_grad():
            p = torch.log_softmax(m(cal).logits[0, -1].float(), -1)
        st["vec"] = None
        return float((base_lp.exp() * (base_lp - p)).sum())

    def solve(v, t, lo=0.0, hi=60.0, it=13):
        if kl_at(v, hi) < t:
            return hi
        for _ in range(it):
            mid = .5 * (lo + hi)
            if kl_at(v, mid) < t: lo = mid
            else:                 hi = mid
        return .5 * (lo + hi)

    print("calibrating doses...", flush=True)
    alpha = {}
    for c in CONCEPTS:
        for kt in cfg.kl_targets:
            alpha[(c, "concept", kt)] = solve(vecs[c], kt)
            alpha[(c, "random", kt)] = solve(rand_dir[c], kt)
            alpha[(c, "semnull", kt)] = solve(sem_null[c], kt)

    # ---- build episodes -----------------------------------------------------
    def make(concept, arm, kt, task):
        p_ids = ids_of(tpl([{"role": "user", "content": task}]))
        P = len(p_ids)
        v = (None if arm == "clean" else
             vecs[concept] if arm == "concept" else
             sem_null[concept] if arm == "semnull" else
             rand_dir[concept])
        a = 0.0 if arm == "clean" else alpha[(concept, arm, kt)]
        st.update(vec=v, alpha=a, lo=P - 1, hi=P)
        enc = torch.tensor([p_ids], device="cuda")
        with torch.no_grad():
            out = m.generate(enc, max_new_tokens=cfg.gen_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        cont = tok.decode(out[0][P:], skip_special_tokens=True)
        full = tpl([{"role": "user", "content": task},
                    {"role": "assistant", "content": cont},
                    {"role": "user", "content": QUESTION}], "Answer:")
        f_ids = ids_of(full)
        return {"concept": concept, "arm": arm, "kt": kt, "task": task,
                "ids": f_ids, "lo": P - 1, "hi": P, "alpha": a,
                "vec_key": (concept, arm), "label": int(arm == "concept"),
                "cont": cont, "leak": int(concept.lower() in cont.lower())}

    print("generating episodes...", flush=True)
    eps = []
    for c in CONCEPTS:
        for task in TASKS[:cfg.n_task]:
            for kt in cfg.kl_targets:
                for arm in ("concept", "random", "semnull"):
                    eps.append(make(c, arm, kt, task))
            eps.append(make(c, "clean", cfg.kl_targets[0], task))
        print(f"  {c}: {len(eps)} episodes so far", flush=True)
    st["vec"] = None

    YES = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    NO = tok(" No", add_special_tokens=False)["input_ids"][0]

    def vec_for(e):
        c, arm = e["vec_key"]
        return (None if arm == "clean" else
                vecs[c] if arm == "concept" else
                sem_null[c] if arm == "semnull" else
                rand_dir[c])

    @torch.no_grad()
    def score(e):
        st.update(vec=vec_for(e), alpha=e["alpha"], lo=e["lo"], hi=e["hi"])
        lg = m(torch.tensor([e["ids"]], device="cuda")).logits[0, -1].float()
        st["vec"] = None
        lp = torch.log_softmax(lg, -1)
        return float(lp[YES] - lp[NO])

    def auroc(p, n):
        p, n = np.asarray(p), np.asarray(n)
        if len(p) < 2 or len(n) < 2: return float("nan")
        a = np.concatenate([p, n]); o = a.argsort(kind="mergesort")
        r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
        return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))

    def evaluate(which):
        sub = [e for e in eps if (e["concept"] in HELD_OUT) == (which == "held_out")]
        for e in sub:
            e["score"] = score(e)
        out = {}
        for kt in cfg.kl_targets:
            p = [e["score"] for e in sub if e["arm"] == "concept" and e["kt"] == kt]
            n = [e["score"] for e in sub if e["arm"] == "random" and e["kt"] == kt]
            cl = [e["score"] for e in sub if e["arm"] == "clean"]
            sn = [e["score"] for e in sub if e["arm"] == "semnull" and e["kt"] == kt]
            out[str(kt)] = {"vs_random": auroc(p, n), "vs_clean": auroc(p, cl),
                            "vs_semnull": auroc(p, sn), "n": len(p)}
        return out

    with open(f"/out/episodes2_{tag}.json", "w") as f:
        json.dump([{k: v for k, v in e.items() if k != "ids"} for e in eps],
                  f, default=str)
    results.commit()

    print("\n=== BEFORE training ===", flush=True)
    before = {"train": evaluate("train"), "held_out": evaluate("held_out")}
    print(json.dumps(before, indent=2), flush=True)

    # ---- install the readout: LoRA on TRAIN concepts only -------------------
    m.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    m = get_peft_model(m, lora)
    m.print_trainable_parameters()
    Ls = m.base_model.model.model.layers          # hook path changes under peft
    H.remove()
    H = Ls[LAYER].register_forward_hook(hook)

    # semnull is evaluation-only: the readout must never be trained against it
    tr = [e for e in eps if e["concept"] in TRAIN_C and e["arm"] != "semnull"]
    labels = [e["label"] for e in tr]
    if cfg.shuffle_labels:
        rng.shuffle(labels)                        # negative control
        print("!! SHUFFLED-LABEL CONTROL RUN !!", flush=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg.lr)
    idx = np.arange(len(tr))
    for ep in range(cfg.epochs):
        rng.shuffle(idx)
        tot, nb = 0.0, 0
        for i in idx:
            e, lab = tr[i], labels[i]
            st.update(vec=vec_for(e), alpha=e["alpha"], lo=e["lo"], hi=e["hi"])
            ids = torch.tensor([e["ids"]], device="cuda")
            tgt = YES if lab == 1 else NO
            lg = m(ids).logits[0, -1]
            loss = torch.nn.functional.cross_entropy(
                lg.unsqueeze(0).float(), torch.tensor([tgt], device="cuda"))
            loss.backward()
            opt.step(); opt.zero_grad()
            st["vec"] = None
            tot += float(loss); nb += 1
        print(f"  epoch {ep+1}: mean loss {tot/max(1,nb):.4f}", flush=True)

    m.eval()
    print("\n=== AFTER training ===", flush=True)
    after = {"train": evaluate("train"), "held_out": evaluate("held_out")}
    print(json.dumps(after, indent=2), flush=True)

    H.remove()
    payload = {"config": cfg_dict, "held_out": HELD_OUT, "train_concepts": TRAIN_C,
               "before": before, "after": after,
               "episodes": [{k: v for k, v in e.items() if k != "ids"} for e in eps]}
    with open(f"/out/readout2_{tag}.json", "w") as f:
        json.dump(payload, f, default=str)
    results.commit()
    return {"tag": tag, "before_heldout": before["held_out"],
            "after_heldout": after["held_out"]}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", tag: str = "install", shuffle: bool = False,
         epochs: int = 3):
    r = run.remote({"model": model, "layer_frac": 0.667,
                    "kl_targets": (1.0, 4.0, 12.0), "n_task": 8, "gen_tokens": 40,
                    "seed": 0, "epochs": epochs, "lr": 1e-4,
                    "shuffle_labels": shuffle}, tag)
    print(json.dumps(r, indent=2))
