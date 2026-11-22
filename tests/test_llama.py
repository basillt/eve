"""
End-to-end test of the llama.cpp path: real llama-server + any small GGUF.

    python tests/test_llama.py path/to/llama-server path/to/model.gguf
"""

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from pcore import reflect  # noqa: E402
from pcore.app import seed_personas  # noqa: E402
from pcore.db import DB  # noqa: E402
from pcore.llama import LlamaBackend, LlamaServer  # noqa: E402
from pcore.server import create_app  # noqa: E402


def main():
    exe, model = sys.argv[1], sys.argv[2]
    tmp = tempfile.mkdtemp()
    try:
        server = LlamaServer(exe, model, ctx=1024, log_path=os.path.join(tmp, "llama.log"), threads=2).start(60)
        backend = LlamaBackend(server, "test-model", info={"flavor": "cpu", "ram_gb": 8.0, "vram_gb": 0, "gpu": None})
        db = DB(os.path.join(tmp, "data"))
        app = create_app(backend, db, seed_personas(tmp), default_max_tokens=24)
        c = TestClient(app)

        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello there"}]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["pcore"]["message_id"]
        print("non-stream ok:", repr(body["choices"][0]["message"]["content"][:60]))

        with c.stream("POST", "/v1/chat/completions",
                      json={"stream": True, "messages": [{"role": "user", "content": "remember: I like football"}]}) as r:
            lines = [l for l in r.iter_lines() if l]
        assert lines[-1] == "data: [DONE]", lines[-3:]
        last = json.loads(lines[-2][5:])
        assert last["choices"][0]["finish_reason"] == "stop" and "pcore" in last
        assert any("football" in m["text"] for m in db.memories()), "explicit memory not saved"
        print("stream ok:", len(lines), "events")

        s = c.get("/api/state").json()
        assert s["backend"] == "llama" and s["brain"]["engine"] == "llama.cpp" and s["brain"]["model_bytes"] > 0
        print("state ok:", {k: s["brain"][k] for k in ("engine", "flavor", "model_bytes", "alive")})

        # the night: a random model will not produce valid JSON, which must be handled
        assert c.post("/api/night").status_code == 200
        for _ in range(240):
            time.sleep(0.25)
            s = c.get("/api/state").json()
            if not s["sleeping"]:
                break
        assert not s["sleeping"] and s["night_error"] is None, s["night_error"]
        rep = s["last_report"]
        assert rep["learning"]["mode"] == "memory" and rep["learning"]["messages_read"] == 2, rep
        print("night ok:", rep["learning"])

        # the night with a model that does answer properly
        fake = lambda msgs, **kw: 'Sure:\n```json\n["The user likes football", "My card is 4111111111111111", "Studies engineering"]\n```'
        db.add_message("default", "user", "I study engineering")
        rep = reflect.run(fake, db, log=lambda *a: None)
        assert rep["learning"]["new_memories"] == ["The user likes football", "Studies engineering"], rep
        assert not any("4111" in m["text"] for m in db.memories())
        rep = reflect.run(fake, db, log=lambda *a: None)
        assert rep["learning"]["messages_read"] == 0 and rep["learning"]["new_memories"] == []
        print("reflection ok: filters secrets, no repeats, reads only new messages")

        assert c.post("/api/rollback").status_code == 404
        assert c.post("/api/wipe").status_code == 200 and db.memories() == []
        server.stop()
        assert not server.alive()
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
