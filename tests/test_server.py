"""API test against the real engine on the tiny random model."""
import json, sys, tempfile, time
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from pcore.db import DB
from pcore.server import RealBackend, MockBackend, create_app
from pcore import persona

# persona rights rule
for bad in ({"id": "x", "name": "X", "prompt": "p"},
            {"id": "x", "name": "X", "prompt": "p", "rights": {"type": "licensed", "holder": "Studio"}}):
    try:
        persona.validate(bad); raise SystemExit("should have been refused")
    except persona.RightsError:
        pass
f = persona.EmotionFilter(); out = "".join(f.feed(p) for p in ["[hap", "py] hel", "lo"]) + f.flush()
assert (out, f.emotion) == ("hello", "happy"), (out, f.emotion)
f = persona.EmotionFilter(); out = "".join(f.feed(p) for p in ["[1] list", " item"]) + f.flush()
assert (out, f.emotion) == ("[1] list item", "neutral"), out
f = persona.EmotionFilter(); out = "".join(f.feed(p) for p in ["hi"]) + f.flush()
assert out == "hi"

import shutil
pdir = tempfile.mkdtemp(); shutil.copy("personas/eve.json", pdir)
db = DB(tempfile.mkdtemp())
backend = MockBackend() if sys.argv[1] == "mock" else RealBackend(sys.argv[1], db, "cpu", "float32", 16)
c = TestClient(create_app(backend, db, pdir, default_max_tokens=12,
                          night_overrides={"lr": 5e-3, "min_examples": 2, "epochs": 2}))

r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "remember: my kernel uses GRUB"}]})
assert r.status_code == 200, r.text
body = r.json(); mid = body["pcore"]["message_id"]
assert body["choices"][0]["message"]["role"] == "assistant" and mid

ids = [mid]
with c.stream("POST", "/v1/chat/completions", json={"stream": True, "user": "c2",
              "messages": [{"role": "user", "content": "How does my kernel boot with GRUB?"}]}) as s:
    lines = [l for l in s.iter_lines() if l.startswith("data: ")]
assert lines[-1] == "data: [DONE]"
last = json.loads(lines[-2][6:]); assert last["pcore"]["memories_used"] == 1, last
ids.append(last["pcore"]["message_id"])
for i in range(4):
    r = c.post("/v1/chat/completions", json={"user": f"k{i}", "messages": [{"role": "user", "content": f"Explain paging step {i}"}]})
    ids.append(r.json()["pcore"]["message_id"])

assert c.post("/api/feedback", json={"message_id": ids[0], "action": "approve"}).status_code == 200
for i in ids[1:]:
    assert c.post("/api/feedback", json={"message_id": i, "action": "correct", "correction": "Paging maps virtual pages to physical frames."}).status_code == 200
assert c.post("/api/feedback", json={"message_id": 99999, "action": "approve"}).status_code == 404
assert c.post("/api/feedback", json={"message_id": ids[0], "action": "correct"}).status_code == 400

st = c.get("/api/state").json()
assert st["counts"]["examples_waiting"] == 6 and st["counts"]["memories"] == 1 and st["persona"]["id"] == "eve"
print("brain:", {k: st["brain"][k] for k in ("hot_experts", "total_experts", "hit_rate_ram")})

assert c.post("/api/night").status_code == 200
assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503
for _ in range(600):
    if not c.get("/api/state").json()["sleeping"]: break
    time.sleep(0.2)
st = c.get("/api/state").json()
assert st["night_error"] is None, st["night_error"]
assert st["morning"] is not None
print("morning:", json.dumps({k: st["morning"][k] for k in ("core", "learning") if k in st["morning"]}, ensure_ascii=False)[:400])
c.post("/api/morning/seen"); assert c.get("/api/state").json()["morning"] is None
m = c.get("/api/memories").json(); assert c.delete(f"/api/memories/{m[0]['id']}").status_code == 200
assert c.get("/").status_code == 200
assert c.get("/web/brain.js").status_code == 200 and c.get("/web/vendor/three.module.min.js").status_code == 200

# characters designed in the app
av = {"skin": "#c08463", "hair": {"style": "spiky", "color": "#141114"}, "outfit": {"style": "hoodie", "color": "#2f4a7a"}}
assert c.post("/api/personas", json={"name": "", "avatar": av}).status_code == 400
assert c.post("/api/personas", json={"name": "x", "avatar": {"blob": "a" * 5000}}).status_code == 400
r = c.post("/api/personas", json={"name": "رعد", "tagline": "صاحبي", "personality": "هادي ومضحك", "avatar": av})
assert r.status_code == 200, r.text
pid = r.json()["id"]
st = c.get("/api/state").json()
assert st["persona"]["id"] == pid and st["persona"]["avatar"]["hair"]["style"] == "spiky"
assert st["persona"]["rights"]["type"] == "original" and st["persona"]["created_by"] == "user"
assert c.put(f"/api/personas/{pid}", json={"name": "رعد 2", "avatar": av}).status_code == 200
assert c.get("/api/state").json()["persona"]["name"] == "رعد 2"
assert c.put("/api/personas/eve", json={"name": "x", "avatar": av}).status_code == 403
r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}); assert r.status_code == 200
assert c.delete(f"/api/personas/{pid}").status_code == 200
assert c.get("/api/state").json()["persona"]["id"] == "eve"
assert c.post("/api/wipe").status_code == 200
st = c.get("/api/state").json(); assert st["counts"] == {"memories": 0, "examples_waiting": 0, "examples_learned": 0, "adapters": 0}, st["counts"]
print("SERVER OK")
