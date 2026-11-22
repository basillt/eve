"""
reflect.py — the night on any device: read the day's chats, keep what is worth remembering.

Works with any model behind a `complete(messages) -> str` function, so it needs no
GPU training. Only things the user said about themself are kept, never passwords or
numbers, and every saved line is visible and deletable in the app's memory screen.
"""

import json
import re
import time

PROMPT = """You read messages a user wrote to their assistant today. Extract facts worth remembering
for future conversations: things the user stated about themself (name, preferences, projects,
people in their life, plans, how they like answers).

Rules:
- Only facts the user clearly stated. No guesses, no facts about the world.
- Never include passwords, codes, card/ID/account numbers, addresses or health details.
- Each fact is one short sentence in the language the user wrote in.
- At most {k} facts. If nothing is worth keeping, return [].
- Answer with a JSON array of strings only.

Already remembered (do not repeat):
{known}

Today's messages:
{msgs}"""

SECRET = re.compile(r"\d{6,}|password|كلمة السر|كلمة المرور|סיסמה", re.IGNORECASE)


def parse_list(text):
    """First JSON array of strings in the model's answer (tolerates code fences and chatter)."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, str):
            it = " ".join(it.split())
            if 3 <= len(it) <= 200 and not SECRET.search(it):
                out.append(it)
    return out


def run(complete, db, max_chars=6000, k=6, log=print):
    night = time.strftime("%Y-%m-%d_%H%M%S")
    since = float(db.get("reflect_since", 0) or 0)
    msgs = db.user_messages_since(since)
    report = {"night": night, "learning": {"mode": "memory", "messages_read": len(msgs), "new_memories": []}}
    if msgs:
        text, used = "", 0
        for m in reversed(msgs):                           # newest first, until the budget is used
            line = "- " + " ".join(m["content"].split())[:500] + "\n"
            if used + len(line) > max_chars:
                break
            text, used = line + text, used + len(line)
        known = "\n".join("- " + m["text"] for m in db.memories()[:40]) or "(nothing yet)"
        answer = complete([{"role": "user", "content": PROMPT.format(k=k, known=known, msgs=text)}],
                          max_tokens=500, temperature=0.2)
        existing = {m["text"].strip().lower() for m in db.memories()}
        for fact in parse_list(answer)[:k]:
            if fact.lower() in existing:
                continue
            db.remember(fact, source="night")
            existing.add(fact.lower())
            report["learning"]["new_memories"].append(fact)
        db.set("reflect_since", msgs[-1]["created"])
    log(f"[night] read {len(msgs)} messages, saved {len(report['learning']['new_memories'])} memories")
    db.add_report(night, report)
    return report
