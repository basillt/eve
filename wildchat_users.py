"""
wildchat_users.py — build per-user prompt sets from WildChat-1M (real, public, anonymized).

WildChat (allenai/WildChat-1M, ODC-BY) has a hashed IP per conversation, which
lets us group conversations of the same (anonymous) user. For every user we keep
the FIRST user message of each conversation, in time order. Users with too few
or suspiciously many conversations (shared IPs, bots) are skipped.

    python wildchat_users.py --users 40 --per-user 50 --out prompts/wildchat_users.jsonl

Output lines: {"text": ..., "user": "u07", "time": "2023-..."}  (grouped by user, time-sorted)
"""

import argparse
import json
import random
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--per-user", type=int, default=50, help="prompts kept per user (time order)")
    ap.add_argument("--min-prompts", type=int, default=50, help="skip users with fewer distinct prompts")
    ap.add_argument("--max-convs", type=int, default=600, help="skip users with more conversations (shared IPs / bots)")
    ap.add_argument("--language", default="English")
    ap.add_argument("--scan", type=int, default=0, help="only scan the first N conversations (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prompts/wildchat_users.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    per_user = defaultdict(list)
    n = 0
    for row in ds:
        n += 1
        if args.scan and n > args.scan:
            break
        if n % 100000 == 0:
            print(f"  scanned {n} conversations, {len(per_user)} users so far", flush=True)
        if args.language and row.get("language") != args.language:
            continue
        ip = row.get("hashed_ip")
        conv = row.get("conversation") or []
        first = next((t for t in conv if t.get("role") == "user"), None)
        if not ip or not first:
            continue
        text = re.sub(r"\s+", " ", (first.get("content") or "")).strip()
        if not (15 <= len(text) <= 1500):
            continue
        per_user[ip].append((str(row.get("timestamp")), text))

    print(f"[scan] {n} conversations, {len(per_user)} users")
    eligible = []
    for ip, items in per_user.items():
        if len(items) > args.max_convs:
            continue
        items.sort()
        seen, uniq = set(), []
        for ts, text in items:
            key = text.lower()[:200]
            if key in seen:
                continue
            seen.add(key)
            uniq.append((ts, text))
        if len(uniq) >= args.min_prompts:
            eligible.append((ip, uniq))
    print(f"[users] {len(eligible)} users have >= {args.min_prompts} distinct prompts")
    if len(eligible) < args.users:
        print(f"[warn] only {len(eligible)} eligible users")
    random.Random(args.seed).shuffle(eligible)
    chosen = eligible[: args.users]

    with open(args.out, "w", encoding="utf-8") as f:
        for i, (_ip, uniq) in enumerate(chosen):
            # evenly spaced over the user's history, kept in time order
            step = max(1, len(uniq) // args.per_user)
            picked = uniq[::step][: args.per_user]
            for ts, text in picked:
                f.write(json.dumps({"text": text, "user": f"u{i:02d}", "time": ts}, ensure_ascii=False) + "\n")
    print(f"[done] {len(chosen)} users x up to {args.per_user} prompts -> {args.out}")


if __name__ == "__main__":
    main()
