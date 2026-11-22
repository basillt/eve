"""
lora.py — a small LoRA implementation (no extra dependencies).

The base model is never modified. All personal learning lives in an adapter
file of a few MB; removing it gives back the original model exactly.
"""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.enabled = True
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        delta = (x.to(torch.float32) @ self.A.t()) @ self.B.t()
        return out + (delta * self.scale).to(out.dtype)


def attach(model, rank=8, alpha=16, targets=("q_proj", "v_proj")):
    """Wrap the target Linear layers. Returns the list of LoRA modules."""
    existing = modules(model)
    if existing:
        return existing
    todo = [(name, m) for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    if not todo:
        raise SystemExit(f"no layers named {targets} found for LoRA")
    for name, m in todo:
        parent_name, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent_name), attr, LoRALinear(m, rank, alpha))
    return modules(model)


def modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def parameters(model):
    return [p for m in modules(model) for p in (m.A, m.B)]


def set_enabled(model, on):
    for m in modules(model):
        m.enabled = on


def reset(model):
    """Back to a fresh (no-op) adapter."""
    for m in modules(model):
        nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
        nn.init.zeros_(m.B)


def state(model):
    return {f"{name}.{k}": getattr(m, k).detach().cpu().clone()
            for name, m in model.named_modules() if isinstance(m, LoRALinear) for k in ("A", "B")}


def load_state(model, sd):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.A.copy_(sd[f"{name}.A"])
                m.B.copy_(sd[f"{name}.B"])


def save(model, path):
    from safetensors.torch import save_file
    save_file(state(model), path)


def load(model, path):
    from safetensors.torch import load_file
    load_state(model, load_file(path))
