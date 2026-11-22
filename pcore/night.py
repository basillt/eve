"""
night.py — what the assistant does while it "sleeps".

  1. core    re-rank the experts by your (decayed) usage and re-pin the hot core
  2. learn   train the LoRA adapter on answers you approved or corrected,
             mixed with replay of older examples so it does not forget
  3. exam    general exam (must not get worse) + personal exam (must get better)
  4. commit  keep the new adapter only if it passed; otherwise roll back

CLI:
    python -m pcore.night --store store/ --data data/
    python -m pcore.night --store store/ --data data/ --rollback
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from . import core as corelib
from . import lora
from .db import DB


@dataclass
class NightConfig:
    core_frac: float = 0.25        # share of experts per layer kept in RAM
    usage_decay: float = 0.9       # per-day decay of old usage
    stickiness: float = 0.1        # bonus for experts already in the core (avoids thrashing)
    rank: int = 8
    alpha: int = 16
    lr: float = 1e-4
    epochs: int = 3
    max_steps: int = 300
    max_len: int = 512
    min_examples: int = 4          # do not train on less than this
    replay_ratio: float = 1.0      # old examples replayed per new example
    exam_holdout: float = 0.2      # share of new examples moved to the personal exam (never trained on)
    general_tolerance: float = 0.02   # general exam may get at most 2% worse vs. last night
    max_drift: float = 0.10           # ... and at most 10% worse vs. the bare base model, ever
    seed: int = 0


def read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- encoding / loss

def encode(tok, q, a, max_len):
    if tok.chat_template:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)["input_ids"][0]
    else:
        prompt = tok(q + "\n", return_tensors="pt")["input_ids"][0]
    answer = tok(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if tok.eos_token_id is not None:
        answer = torch.cat([answer, torch.tensor([tok.eos_token_id])])
    answer = answer[: max(8, max_len // 2)]
    prompt = prompt[-(max_len - len(answer)):]
    ids = torch.cat([prompt, answer])
    labels = ids.clone()
    labels[: len(prompt)] = -100
    return ids, labels


def pair_loss(model, tok, pair, max_len):
    ids, labels = encode(tok, pair["q"], pair["a"], max_len)
    dev = next(model.parameters()).device
    logits = model(ids[None].to(dev)).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(logits, labels[1:].to(dev), ignore_index=-100)


@torch.no_grad()
def exam(model, tok, pairs, max_len=512):
    """Mean loss on reference answers (lower is better). None if there is no exam."""
    if not pairs:
        return None
    return float(np.mean([pair_loss(model, tok, p, max_len).item() for p in pairs]))


def train(model, tok, pairs, cfg, log=print):
    params = lora.parameters(model)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    rng = random.Random(cfg.seed)
    steps = min(cfg.max_steps, cfg.epochs * len(pairs))
    order, losses = [], []
    for step in range(steps):
        if not order:
            order = pairs[:]
            rng.shuffle(order)
        loss = pair_loss(model, tok, order.pop(), cfg.max_len)
        if not torch.isfinite(loss):
            log("[night] loss is not finite — stopping training")
            return False
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 25 == 0 or step + 1 == steps:
            log(f"[night] step {step + 1}/{steps}  loss {np.mean(losses[-25:]):.3f}")
    for p in params:
        p.requires_grad_(False)
    return True


# ---------------------------------------------------------------- adapters

def adapter_dir(db):
    d = os.path.join(db.dir, "adapters")
    os.makedirs(d, exist_ok=True)
    return d


def load_current_adapter(model, db, cfg=NightConfig()):
    """Attach LoRA and load the adapter the DB says is current (if any)."""
    lora.attach(model, cfg.rank, cfg.alpha)
    hist = db.get("adapters", [])
    if hist and os.path.exists(os.path.join(adapter_dir(db), hist[-1])):
        lora.load(model, os.path.join(adapter_dir(db), hist[-1]))
        return hist[-1]
    lora.reset(model)
    return None


def rollback(model, db):
    """Drop the newest adapter and go back to the one before it."""
    hist = db.get("adapters", [])
    if not hist:
        return None
    dropped = hist.pop()
    db.set("adapters", hist)
    # examples learned that night become "new" again, so a later night can retry them
    db._x("UPDATE messages SET trained_night=NULL WHERE trained_night=?", (dropped.replace(".safetensors", ""),))
    if model is not None:
        load_current_adapter(model, db)
    return dropped


# ---------------------------------------------------------------- the night

def run_night(model, tok, store, db, cfg=NightConfig(), exam_file=None, replay_file=None, log=print):
    night = time.strftime("%Y-%m-%d_%H%M%S")
    report = {"night": night, "config": asdict(cfg)}

    # ---- 1. core --------------------------------------------------------
    db.add_usage(store.drain_usage())
    shape = (store.L, store.E)
    usage, recent = db.usage(shape, cfg.usage_decay), db.usage(shape, day=time.strftime("%Y-%m-%d"))
    old_core = db.get("core")
    if usage.sum() > 0:
        new_core = corelib.build_core(usage, frac=cfg.core_frac, previous=old_core, stickiness=cfg.stickiness)
        report["core"] = {
            "experts_per_layer": len(new_core[0]), "of": store.E,
            "swapped": corelib.diff(old_core, new_core),
            "coverage_before": corelib.coverage(old_core, recent) if old_core else None,
            "coverage_after": corelib.coverage(new_core, recent),
            "coverage_random": len(new_core[0]) / store.E,
        }
        store.set_core(new_core)
        db.set("core", new_core)
    else:
        report["core"] = {"skipped": "no usage recorded yet"}
    s = store.summary()
    report["ram"] = {"hot_core_bytes": s["hot_ram_bytes"], "skeleton_bytes": s["skeleton_bytes"],
                     "all_experts_bytes": s["all_experts_bytes"]}

    # ---- 2. learn -------------------------------------------------------
    load_current_adapter(model, db, cfg)
    exam_ids = set(db.get("personal_exam", []))
    all_new = db.examples(new_only=True)
    new = [e for e in all_new if e["id"] not in exam_ids]
    old = [e for e in db.examples(new_only=False) if e["id"] not in exam_ids]
    rng = random.Random(cfg.seed)

    if len(new) < cfg.min_examples:
        report["learning"] = {"skipped": f"only {len(new)} new approved/corrected answers (need {cfg.min_examples})"}
        db.add_report(night, report)
        log(f"[night] nothing to learn tonight: {report['learning']['skipped']}")
        return report

    n_hold = max(1, int(round(len(new) * cfg.exam_holdout)))
    held = rng.sample(new, n_hold)
    held_ids = {e["id"] for e in held}
    new_train = [e for e in new if e["id"] not in held_ids]
    exam_ids |= held_ids
    personal_exam = [e for e in db.examples() if e["id"] in exam_ids][-50:]
    general_exam = read_jsonl(exam_file)
    replay = rng.sample(old, min(len(old), int(len(new_train) * cfg.replay_ratio)))
    general_replay = read_jsonl(replay_file)
    replay += rng.sample(general_replay, min(len(general_replay), max(1, len(new_train) // 2))) if general_replay else []

    # ---- 3. exam before / train / exam after ---------------------------
    lora.set_enabled(model, False)
    base_general = exam(model, tok, general_exam, cfg.max_len)
    lora.set_enabled(model, True)
    before = {"general": exam(model, tok, general_exam, cfg.max_len),
              "personal": exam(model, tok, personal_exam, cfg.max_len)}
    previous_state = lora.state(model)

    log(f"[night] training on {len(new_train)} new + {len(replay)} replay examples")
    ok = train(model, tok, new_train + replay, cfg, log)
    after = {"general": exam(model, tok, general_exam, cfg.max_len),
             "personal": exam(model, tok, personal_exam, cfg.max_len)} if ok else before

    reasons = []
    if not ok:
        reasons.append("training diverged")
    if ok and not after["personal"] < before["personal"]:
        reasons.append("no improvement on the personal exam")
    if ok and general_exam:
        if after["general"] > before["general"] * (1 + cfg.general_tolerance):
            reasons.append("general exam got worse than last night")
        if after["general"] > base_general * (1 + cfg.max_drift):
            reasons.append("drifted too far from the base model")
    accepted = not reasons

    # ---- 4. commit or roll back ----------------------------------------
    if accepted:
        name = f"{night}.safetensors"
        lora.save(model, os.path.join(adapter_dir(db), name))
        db.set("adapters", db.get("adapters", []) + [name])
        db.set("personal_exam", sorted(exam_ids))
        db.mark_trained([e["id"] for e in new_train], night)
    else:
        lora.load_state(model, previous_state)

    report["learning"] = {
        "accepted": accepted, "reasons": reasons,
        "new_examples": len(new_train), "corrections": sum(e["corrected"] for e in new_train),
        "replayed": len(replay), "exam_size": {"general": len(general_exam), "personal": len(personal_exam)},
        "base_general": base_general, "before": before, "after": after,
        "samples": [e["q"][:120] for e in new_train[:3]],
    }
    db.add_report(night, report)
    log(f"[night] {'ACCEPTED' if accepted else 'ROLLED BACK: ' + '; '.join(reasons)}  "
        f"personal {before['personal']:.3f} -> {after['personal']:.3f}"
        + (f"  general {before['general']:.3f} -> {after['general']:.3f}" if general_exam else ""))
    return report


def main():
    from . import engine
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--exam", default="exams/general_exam.jsonl")
    ap.add_argument("--replay", default="exams/general_replay.jsonl")
    ap.add_argument("--rollback", action="store_true", help="undo the last accepted night")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="float32 is the safest for training")
    for f, t in (("core_frac", float), ("lr", float), ("epochs", int), ("max_steps", int), ("rank", int),
                 ("min_examples", int)):
        ap.add_argument("--" + f.replace("_", "-"), type=t, default=getattr(NightConfig, f))
    args = ap.parse_args()
    db = DB(args.data)
    if args.rollback:
        print(f"[night] rolled back: {rollback(None, db)}")
        return
    cfg = NightConfig(core_frac=args.core_frac, lr=args.lr, epochs=args.epochs, max_steps=args.max_steps,
                      rank=args.rank, min_examples=args.min_examples)
    model, tok, store = engine.load(args.store, device=args.device, dtype=args.dtype, core=db.get("core"))
    report = run_night(model, tok, store, db, cfg, args.exam, args.replay)
    print(json.dumps({k: report[k] for k in ("core", "ram", "learning")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
