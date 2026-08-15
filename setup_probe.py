"""Boot check and weight prefetch for the introspection experiment.

Two jobs, done in one container boot so we pay the cold start once:
  1. Confirm the GPU, torch, and transformers stack actually work together.
  2. Pull Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct into a persistent Modal
     Volume, so every later run starts from cached weights instead of a 15 GB
     download. Download, not compute, is what makes these runs expensive.

It also exercises the one primitive the whole experiment depends on: adding a
vector to the residual stream via a forward hook, and confirming that doing so
actually changes the output. If that does not work, nothing downstream does.
"""

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
        "jinja2>=3.1.0",          # older jinja2 breaks apply_chat_template
    )
)

# Persisted across runs: weights land here once and stay.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]


@app.function(
    image=image,
    gpu="L4",
    volumes={"/cache": hf_cache},
    timeout=3600,
)
def boot_and_prefetch():
    import os, time
    os.environ["HF_HOME"] = "/cache/hf"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = {}
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} "
          f"| {torch.cuda.get_device_name(0)}", flush=True)
    out["gpu"] = torch.cuda.get_device_name(0)
    out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)

    for repo in MODELS:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo)
        m = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        load_s = time.time() - t0
        print(f"{repo}: loaded in {load_s:.0f}s", flush=True)

        # --- the primitive: does a residual-stream injection change the output? ---
        layers = m.model.layers
        d = m.config.hidden_size
        mid = len(layers) // 2
        torch.manual_seed(0)
        vec = torch.randn(d, dtype=torch.bfloat16, device="cuda")
        vec = vec / vec.norm()

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "Say the single word: apple"}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            clean = m(**ids).logits[0, -1].float()

        scale = 40.0
        def hook(mod, inp, o):
            h = o[0] if isinstance(o, tuple) else o
            h[:, -1, :] = h[:, -1, :] + scale * vec
            return (h,) + o[1:] if isinstance(o, tuple) else h

        hh = layers[mid].register_forward_hook(hook)
        with torch.no_grad():
            steered = m(**ids).logits[0, -1].float()
        hh.remove()

        delta = (steered - clean).abs().max().item()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(steered, -1), torch.softmax(clean, -1),
            reduction="sum").item()
        print(f"  injection at L{mid}: max logit delta {delta:.3f}, KL {kl:.4f}",
              flush=True)

        out[repo] = {
            "load_s": round(load_s),
            "n_layers": len(layers),
            "hidden": d,
            "inject_max_logit_delta": round(delta, 3),
            "inject_kl": round(kl, 4),
            "hook_works": delta > 1e-3,
        }

        del m, tok
        torch.cuda.empty_cache()

    hf_cache.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    r = boot_and_prefetch.remote()
    print(json.dumps(r, indent=2))
