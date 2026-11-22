"""
engine.py — run an MoE model with its experts on SSD and only a hot core in RAM.

    split   one-time: cut a Hugging Face MoE model into
              skeleton.safetensors   everything except the experts (always in RAM)
              experts/L{i}.*.npy     one file per layer, memory-mapped from disk
    load    build the model without ever allocating the experts in RAM, then
            serve each expert from: hot core (RAM) -> LRU cache (RAM) -> SSD

CLI:
    python -m pcore.engine split --model <hf id or path> --out store/
    python -m pcore.engine check --model <hf id or path> --store store/   (outputs must match)
"""

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def _expert_modules(model):
    """Modules holding stacked expert weights (transformers >= 5 layout)."""
    found = []
    for name, m in model.named_modules():
        if isinstance(getattr(m, "gate_up_proj", None), nn.Parameter) and \
                isinstance(getattr(m, "down_proj", None), nn.Parameter):
            for flag, want in (("is_concatenated", True), ("is_transposed", False), ("has_bias", False)):
                if getattr(m, flag, want) != want:
                    raise SystemExit(f"{name}: unsupported expert layout ({flag}={getattr(m, flag)})")
            found.append((name, m))
    if not found:
        raise SystemExit("No expert modules found — is this an MoE model supported by transformers >= 5?")
    return found


# ---------------------------------------------------------------- split

def split(model_id, out, dtype="float16", trust_remote_code=False):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(os.path.join(out, "experts"), exist_ok=True)
    print(f"[split] loading {model_id} (needs the full model in RAM once)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32 if dtype == "float32" else torch.float16,
                                                 trust_remote_code=trust_remote_code)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(out)
    model.config.save_pretrained(out)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out)

    np_dtype = np.float32 if dtype == "float32" else np.float16
    layers, skip = [], set()
    for i, (name, m) in enumerate(_expert_modules(model)):
        gu = m.gate_up_proj.detach().float().numpy().astype(np_dtype)
        dn = m.down_proj.detach().float().numpy().astype(np_dtype)
        np.save(os.path.join(out, "experts", f"L{i}.gate_up.npy"), gu)
        np.save(os.path.join(out, "experts", f"L{i}.down.npy"), dn)
        layers.append({"module": name, "num_experts": int(gu.shape[0]),
                       "bytes_per_expert": int(gu[0].nbytes + dn[0].nbytes)})
        skip.update({f"{name}.gate_up_proj", f"{name}.down_proj"})

    seen, skeleton = {}, {}
    for key, t in model.state_dict().items():
        if key in skip:
            continue
        ptr = t.data_ptr()
        if ptr in seen:          # tied weights (e.g. lm_head = embeddings): store once
            continue
        seen[ptr] = key
        skeleton[key] = t.contiguous()
    save_file(skeleton, os.path.join(out, "skeleton.safetensors"))

    meta = {"source": model_id, "dtype": dtype, "layers": layers,
            "skeleton_bytes": int(sum(t.numel() * t.element_size() for t in skeleton.values()))}
    with open(os.path.join(out, "pcore_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    exp = sum(l["bytes_per_expert"] * l["num_experts"] for l in layers)
    print(f"[split] skeleton {meta['skeleton_bytes'] / 1e9:.2f} GB (RAM)  +  experts {exp / 1e9:.2f} GB (SSD)  -> {out}")
    return meta


# ---------------------------------------------------------------- store

class ExpertStore:
    """Serves expert weights: hot core (pinned RAM) -> LRU cache -> memory-mapped SSD."""

    def __init__(self, path, device="cpu", dtype=torch.float32, cache_experts=64):
        with open(os.path.join(path, "pcore_meta.json")) as f:
            self.meta = json.load(f)
        self.path, self.device, self.dtype = path, device, dtype
        self.L = len(self.meta["layers"])
        self.E = self.meta["layers"][0]["num_experts"]
        self.bytes_per_expert = self.meta["layers"][0]["bytes_per_expert"]
        self.maps = [(np.load(os.path.join(path, "experts", f"L{i}.gate_up.npy"), mmap_mode="r"),
                      np.load(os.path.join(path, "experts", f"L{i}.down.npy"), mmap_mode="r"))
                     for i in range(self.L)]
        self.hot = {}
        self.cache = OrderedDict()
        self.cache_experts = cache_experts
        self.core = [[] for _ in range(self.L)]
        self.usage = np.zeros((self.L, self.E), dtype=np.int64)   # activations since last drain
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"hot": 0, "cache": 0, "ssd": 0, "ssd_loads": 0, "ssd_seconds": 0.0}

    def _read(self, layer, e):
        gu, dn = self.maps[layer]
        return (torch.from_numpy(np.array(gu[e])).to(self.device, self.dtype),
                torch.from_numpy(np.array(dn[e])).to(self.device, self.dtype))

    def set_core(self, core):
        """Pin a new hot core in RAM (loads new members, frees removed ones)."""
        want = {(l, e) for l, ids in enumerate(core) for e in ids}
        for key in list(self.hot):
            if key not in want:
                del self.hot[key]
        for key in want:
            if key not in self.hot:
                self.hot[key] = self.cache.pop(key, None) or self._read(*key)
        self.core = [sorted(ids) for ids in core]

    def get(self, layer, e, n_tokens=1):
        key = (layer, e)
        w = self.hot.get(key)
        if w is not None:
            self.stats["hot"] += n_tokens
            return w
        w = self.cache.get(key)
        if w is not None:
            self.cache.move_to_end(key)
            self.stats["cache"] += n_tokens
            return w
        t0 = time.perf_counter()
        w = self._read(layer, e)
        self.stats["ssd_seconds"] += time.perf_counter() - t0
        self.stats["ssd_loads"] += 1
        self.stats["ssd"] += n_tokens
        if self.cache_experts > 0:
            self.cache[key] = w
            while len(self.cache) > self.cache_experts:
                self.cache.popitem(last=False)
        return w

    def drain_usage(self):
        u, self.usage = self.usage, np.zeros_like(self.usage)
        return u

    def summary(self):
        s = self.stats
        total = s["hot"] + s["cache"] + s["ssd"]
        bpe = self.bytes_per_expert * (torch.finfo(self.dtype).bits / 8) / \
            (np.dtype(self.maps[0][0].dtype).itemsize)
        return {
            "hot_experts": len(self.hot), "cached_experts": len(self.cache),
            "total_experts": self.L * self.E,
            "hot_ram_bytes": int(len(self.hot) * bpe), "cache_ram_bytes": int(len(self.cache) * bpe),
            "all_experts_bytes": int(self.L * self.E * bpe),
            "skeleton_bytes": int(self.meta["skeleton_bytes"]),
            "hit_rate_hot": s["hot"] / total if total else None,
            "hit_rate_ram": (s["hot"] + s["cache"]) / total if total else None,
            "ssd_loads": s["ssd_loads"], "ssd_seconds": s["ssd_seconds"], "activations": total,
        }


class DiskExperts(nn.Module):
    """Drop-in replacement for a transformers Experts module, backed by an ExpertStore."""

    def __init__(self, store, layer, act_fn, num_experts):
        super().__init__()
        self.store, self.layer, self.act_fn, self.num_experts = store, layer, act_fn, num_experts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        flat = top_k_index.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_experts)[: self.num_experts].cpu().numpy()
        self.store.usage[self.layer] += counts
        for e in np.nonzero(counts)[0]:
            token_idx, pos = torch.where(top_k_index == int(e))
            gate_up, down = self.store.get(self.layer, int(e), int(counts[e]))
            x = hidden_states[token_idx]
            gate, up = nn.functional.linear(x, gate_up.to(x.dtype)).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, down.to(x.dtype))
            h = h * top_k_weights[token_idx, pos, None]
            out.index_add_(0, token_idx, h.to(out.dtype))
        return out


# ---------------------------------------------------------------- load

def load(path, device="auto", dtype="auto", core=None, cache_experts=64):
    """Returns (model, tokenizer, store). The experts are never fully loaded into RAM."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    store = ExpertStore(path, device=device, dtype=dtype, cache_experts=cache_experts)

    for i, layer in enumerate(store.meta["layers"]):
        parent_name, _, attr = layer["module"].rpartition(".")
        parent = model.get_submodule(parent_name)
        old = getattr(parent, attr)
        setattr(parent, attr, DiskExperts(store, i, old.act_fn, layer["num_experts"]))

    state = load_file(os.path.join(path, "skeleton.safetensors"))
    state = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta or unexpected:
        raise SystemExit(f"store does not match the model: missing={still_meta[:3]} unexpected={unexpected[:3]}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if core is not None:
        store.set_core(core)
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok, store


# ---------------------------------------------------------------- cli

def _check(model_id, path):
    """Engine output must equal the original model's output."""
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    model, tok, store = load(path, device="cpu", dtype="float32", cache_experts=8)
    store.set_core([list(range(store.E // 4)) for _ in range(store.L)])
    ids = tok("Explain how a mixture-of-experts transformer routes tokens.", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        a, b = ref(ids).logits, model(ids).logits
    err = (a - b).abs().max().item()
    print(f"[check] max |logit difference| = {err:.2e}   stats = {store.summary()}")
    tol = 1e-4 if store.meta["dtype"] == "float32" else 5e-2
    if err > tol:
        raise SystemExit("[check] FAILED: engine output differs from the original model")
    print("[check] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.add_argument("--trust-remote-code", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--model", required=True)
    c.add_argument("--store", required=True)
    args = ap.parse_args()
    if args.cmd == "split":
        split(args.model, args.out, args.dtype, args.trust_remote_code)
    else:
        _check(args.model, args.store)


if __name__ == "__main__":
    main()
