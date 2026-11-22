"""
profile_experts.py — record which parts of a model fire for a set of prompts.

Two units can be measured:
  experts  (MoE models)   which experts the router picks, per layer
  neurons  (dense models) which MLP neurons are among the most active, per layer
           (a neuron "fires" for a token if it is in the top --neuron-frac of
           that token's |activation| in that layer)

For every prompt we run generation and count, per layer, how often each unit
fired. Output is a .npz file with:
    counts   int32 [num_prompts, num_layers, units_per_layer]
    tokens   int32 [num_prompts]            tokens counted per prompt
    meta     json string (model id, unit, param counts for RAM estimates)

--skip-template ignores the chat-template tokens (system prompt etc.) that are
identical for every prompt, so only your text and the model's answer count.

Usage:
    python profile_experts.py --model ibm-granite/granite-3.1-1b-a400m-instruct \
        --prompts prompts/personal.jsonl --out results/personal.npz
    python profile_experts.py --model Qwen/Qwen2.5-1.5B-Instruct --unit neurons \
        --prompts prompts/personal.jsonl --out results_neurons/personal.npz
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------- prompts

def load_prompts(path):
    """Read prompts from .jsonl ({"text": ...} per line) or .txt (one per line)."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith(".jsonl"):
                obj = json.loads(line)
                text = obj.get("text") or obj.get("prompt") or ""
            else:
                text = line
            text = text.strip()
            if text:
                prompts.append(text)
    if not prompts:
        sys.exit(f"No prompts found in {path}")
    return prompts


# ---------------------------------------------------------------- routers

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def find_routers(model):
    """Find router modules: a child named gate/router of a module that has `.experts`."""
    routers = []
    for name, module in model.named_modules():
        if not hasattr(module, "experts"):
            continue
        for child_name in ("gate", "router"):
            child = getattr(module, child_name, None)
            if isinstance(child, torch.nn.Module):
                m = LAYER_RE.search(name + ".")
                if m is None:
                    continue
                routers.append((int(m.group(1)), f"{name}.{child_name}", child))
                break
    routers.sort(key=lambda r: r[0])
    return routers


def num_experts_of(config):
    for key in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        v = getattr(config, key, None)
        if isinstance(v, int) and v > 0:
            return v
    sys.exit("Could not read number of experts from the model config.")


def top_k_of(config):
    for key in ("num_experts_per_tok", "moe_top_k", "top_k"):
        v = getattr(config, key, None)
        if isinstance(v, int) and v > 0:
            return v
    sys.exit("Could not read top-k from the model config.")


def extract_indices(output, num_experts, top_k):
    """Turn a router's output into a [tokens, top_k] tensor of expert ids.

    Handles both styles:
      - transformers >= 5: router returns (logits, weights, indices)
      - plain linear gate: returns logits [tokens, num_experts] -> we take top-k
    """
    if isinstance(output, (tuple, list)):
        for t in output:
            if torch.is_tensor(t) and not t.is_floating_point() and t.shape[-1] == top_k:
                return t.reshape(-1, top_k)
        for t in output:
            if torch.is_tensor(t) and t.is_floating_point() and t.shape[-1] == num_experts:
                return torch.topk(t.reshape(-1, num_experts), top_k, dim=-1).indices
        return None
    if torch.is_tensor(output) and output.shape[-1] == num_experts:
        return torch.topk(output.reshape(-1, num_experts).float(), top_k, dim=-1).indices
    return None


def find_mlp_down(model):
    """Dense MLP output projections: layers.{i}.mlp.down_proj (input = per-neuron activations)."""
    found = []
    for name, module in model.named_modules():
        if name.endswith(".mlp.down_proj") and isinstance(module, torch.nn.Linear):
            m = LAYER_RE.search(name + ".")
            if m:
                found.append((int(m.group(1)), name, module))
    found.sort(key=lambda r: r[0])
    return found


def template_prefix_len(tok, text):
    """Number of leading tokens that the chat template adds before the user's text."""
    a = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
    b = tok.apply_chat_template([{"role": "user", "content": "x" + text}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# ---------------------------------------------------------------- params (for RAM estimates)

def count_params(model_id, trust_remote_code, unit):
    """Split parameters into the part that scales with the core size and the fixed rest."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    scalable, other = 0, 0
    for name, p in meta_model.named_parameters():
        if unit == "experts":
            hit = ".experts." in name or name.endswith(".experts")
        else:
            hit = ".mlp." in name and any(k in name for k in ("gate_proj", "up_proj", "down_proj"))
        if hit:
            scalable += p.numel()
        else:
            other += p.numel()
    del meta_model
    # key names kept for compatibility with older result files
    return {"expert_params": int(scalable), "non_expert_params": int(other)}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id or local path")
    ap.add_argument("--unit", default="auto", choices=["auto", "experts", "neurons"],
                    help="auto = experts for MoE models, neurons otherwise")
    ap.add_argument("--neuron-frac", type=float, default=0.1,
                    help="neurons mode: share of a layer's neurons counted as firing for each token")
    ap.add_argument("--trace-out", default="",
                    help="experts mode: also save the per-token expert choices (for cache simulation)")
    ap.add_argument("--skip-template", action="store_true",
                    help="do not count the chat-template tokens that are the same for every prompt")
    ap.add_argument("--prompts", required=True, help=".jsonl or .txt file of prompts")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="tokens to generate per prompt (0 = prompt only, fastest)")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N prompts")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16 | float32")
    ap.add_argument("--gpu-mem", default="3.5GiB",
                    help="max GPU memory for weights when the model does not fit (default fits a 4GB card)")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    has_cuda = torch.cuda.is_available()
    device = ("cuda" if has_cuda else "cpu") if args.device == "auto" else args.device
    if args.dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
    else:
        dtype = getattr(torch, args.dtype)

    print(f"[load] {args.model}  device={device}  dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    load_kwargs = dict(dtype=dtype, trust_remote_code=args.trust_remote_code)
    if device == "cuda":
        # Fill the GPU first, spill the rest to CPU RAM (needed on small cards).
        load_kwargs.update(device_map="auto", max_memory={0: args.gpu_mem, "cpu": "64GiB"})
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    cfg = model.config
    routers = find_routers(model)
    unit = args.unit if args.unit != "auto" else ("experts" if routers else "neurons")
    misses = {"n": 0}
    # During prefill the hooks see the whole prompt; skip its first `skip` rows (template).
    span = {"prompt_len": -1, "skip": 0}

    def keep_rows(x):
        if span["skip"] and x.shape[0] == span["prompt_len"]:
            return x[span["skip"]:]
        return x

    if unit == "experts":
        if not routers:
            sys.exit("No MoE routers found. Use --unit neurons for a dense model.")
        E, K = num_experts_of(cfg), top_k_of(cfg)
        layer_ids = [r[0] for r in routers]
        L = len(routers)
        print(f"[model] {L} MoE layers, {E} experts per layer, top-{K}", flush=True)
        current = np.zeros((L, E), dtype=np.int64)
        trace_buf = [[] for _ in range(L)]

        def make_hook(slot):
            def hook(_module, _inputs, output):
                idx = extract_indices(output, E, K)
                if idx is None:
                    misses["n"] += 1
                    return
                rows = keep_rows(idx).detach().to("cpu").numpy()
                current[slot] += np.bincount(rows.reshape(-1), minlength=E)[:E]
                if args.trace_out:
                    trace_buf[slot].append(rows.reshape(-1, K).astype(np.uint16))
            return hook

        handles = [r[2].register_forward_hook(make_hook(i)) for i, r in enumerate(routers)]
    else:
        downs = find_mlp_down(model)
        if not downs:
            sys.exit("No dense MLP layers (mlp.down_proj) found for neuron profiling.")
        layer_ids = [d[0] for d in downs]
        L = len(downs)
        E = downs[0][2].in_features
        K = max(1, int(round(E * args.neuron_frac)))
        print(f"[model] {L} MLP layers, {E} neurons per layer, counting the top {K} "
              f"({args.neuron_frac:.0%}) per token", flush=True)
        gpu_counts = [None] * L

        def make_hook(slot):
            def hook(_module, inputs):
                h = keep_rows(inputs[0].reshape(-1, E))
                if h.shape[0] == 0:
                    return
                idx = torch.topk(h.abs().float(), K, dim=-1).indices.reshape(-1)
                c = torch.bincount(idx, minlength=E)
                gpu_counts[slot] = c if gpu_counts[slot] is None else gpu_counts[slot] + c
            return hook

        handles = [d[2].register_forward_pre_hook(make_hook(i)) for i, d in enumerate(downs)]
        current = np.zeros((L, E), dtype=np.int64)

    first_device = next(model.parameters()).device
    traces, offsets = [], [0]
    counts = np.zeros((len(prompts), L, E), dtype=np.int32)
    tokens = np.zeros(len(prompts), dtype=np.int32)
    t0 = time.time()

    for i, text in enumerate(prompts):
        if tok.chat_template and not args.no_chat_template:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)["input_ids"]
        else:
            ids = tok(text, return_tensors="pt")["input_ids"]
        ids = ids[:, -args.max_prompt_tokens:].to(first_device)
        span["prompt_len"] = ids.shape[1]
        span["skip"] = 0
        if args.skip_template and tok.chat_template and not args.no_chat_template:
            span["skip"] = min(template_prefix_len(tok, text), ids.shape[1] - 1)

        current[:] = 0
        if unit == "experts" and args.trace_out:
            for b in trace_buf:
                b.clear()
        if unit == "neurons":
            gpu_counts[:] = [None] * L
        with torch.no_grad():
            if args.max_new_tokens > 0:
                out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
                n_tok = out.shape[1] - 1  # the last sampled token is never fed back in
            else:
                model(ids)
                n_tok = ids.shape[1]
        if unit == "neurons":
            for j, c in enumerate(gpu_counts):
                if c is not None:
                    current[j] = c.to("cpu").numpy()
        counts[i] = current
        tokens[i] = n_tok - span["skip"]
        if unit == "experts" and args.trace_out:
            per_layer = [np.concatenate(b) if b else np.zeros((0, K), np.uint16) for b in trace_buf]
            t = min(len(x) for x in per_layer)
            traces.append(np.stack([x[:t] for x in per_layer], axis=1))   # [tokens, L, K]
            offsets.append(offsets[-1] + t)

        if (i + 1) % 5 == 0 or i + 1 == len(prompts):
            rate = (i + 1) / (time.time() - t0)
            eta = (len(prompts) - i - 1) / max(rate, 1e-9)
            print(f"  {i + 1}/{len(prompts)} prompts  ({rate:.2f}/s, ~{eta / 60:.1f} min left)", flush=True)

    for h in handles:
        h.remove()
    if misses["n"]:
        print(f"[warn] {misses['n']} router calls could not be parsed", flush=True)

    # Sanity check: every token should pick exactly K experts in every layer.
    per_tok = counts.sum(axis=2) / np.maximum(tokens[:, None], 1)
    print(f"[check] avg {unit} per token per layer = {per_tok.mean():.2f} (expected ~{K})", flush=True)

    meta = {
        "model": args.model, "unit": unit, "num_experts": E, "top_k": K, "num_layers": L,
        "layer_ids": layer_ids, "prompts_file": os.path.abspath(args.prompts),
        "max_new_tokens": args.max_new_tokens, "num_prompts": len(prompts),
        "skip_template": bool(args.skip_template),
        "neuron_frac": args.neuron_frac if unit == "neurons" else None,
    }
    try:
        meta.update(count_params(args.model, args.trust_remote_code, unit))
    except Exception as e:  # RAM estimate is optional
        print(f"[warn] could not count parameters: {e}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, counts=counts, tokens=tokens, meta=json.dumps(meta))
    if unit == "experts" and args.trace_out:
        np.savez_compressed(args.trace_out, trace=np.concatenate(traces) if traces else np.zeros((0, L, K), np.uint16),
                            offsets=np.array(offsets, dtype=np.int64))
        print(f"[trace] saved {args.trace_out}  ({offsets[-1]} tokens)", flush=True)
    print(f"[done] saved {args.out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
