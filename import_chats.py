"""
import_chats.py — turn a ChatGPT or Claude data export into a prompts file.

Only YOUR messages are kept (not the assistant's replies). Everything stays on
your machine.

  ChatGPT: Settings → Data controls → Export data → unzip → conversations.json
  Claude:  Settings → Privacy → Export data → unzip → conversations.json

Usage:
    python import_chats.py path/to/conversations.json --out prompts/personal.jsonl
"""

import argparse
import json
import random


def chatgpt_messages(data):
    for conv in data:
        mapping = conv.get("mapping") or {}
        nodes = [n for n in mapping.values() if n and n.get("message")]
        nodes.sort(key=lambda n: n["message"].get("create_time") or 0)
        for n in nodes:
            msg = n["message"]
            if (msg.get("author") or {}).get("role") != "user":
                continue
            parts = (msg.get("content") or {}).get("parts") or []
            text = "\n".join(p for p in parts if isinstance(p, str))
            yield text, msg.get("create_time") or 0


def claude_messages(data):
    for conv in data:
        for msg in conv.get("chat_messages") or []:
            if msg.get("sender") != "human":
                continue
            text = msg.get("text") or ""
            if not text and isinstance(msg.get("content"), list):
                text = "\n".join(c.get("text", "") for c in msg["content"] if isinstance(c, dict))
            yield text, msg.get("created_at") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="conversations.json from ChatGPT or Claude")
    ap.add_argument("--out", default="prompts/personal.jsonl")
    ap.add_argument("--min-chars", type=int, default=15, help="drop very short messages like 'ok' or 'thanks'")
    ap.add_argument("--max-chars", type=int, default=2000, help="cut very long pastes")
    ap.add_argument("--max", type=int, default=0, help="keep at most N prompts (random sample)")
    args = ap.parse_args()

    with open(args.export, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("conversations") or [data]

    is_chatgpt = any("mapping" in c for c in data[:20])
    source = "ChatGPT" if is_chatgpt else "Claude"
    messages = list(chatgpt_messages(data) if is_chatgpt else claude_messages(data))

    seen, prompts = set(), []
    for text, ts in messages:
        text = (text or "").strip()
        if len(text) < args.min_chars:
            continue
        text = text[: args.max_chars]
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append({"text": text, "time": ts})

    # Keep chronological order: lets you later build the core from the past
    # and test it on the future, which is the honest version of the experiment.
    prompts.sort(key=lambda p: str(p["time"]))
    if args.max and len(prompts) > args.max:
        keep = sorted(random.Random(0).sample(range(len(prompts)), args.max))
        prompts = [prompts[i] for i in keep]

    with open(args.out, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{source} export: {len(messages)} of your messages → {len(prompts)} prompts saved to {args.out}")


if __name__ == "__main__":
    main()
