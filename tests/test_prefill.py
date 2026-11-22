"""Small models sometimes answer with the mood tag only: the server then lets them continue from it."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from pcore.db import DB
from pcore.server import MockBackend, create_app
class TagOnly(MockBackend):
    calls = []
    def stream(self, messages, max_tokens, temperature):
        self.calls.append(messages[-1])
        if messages[-1]["role"] == "assistant":
            yield "Hello, I am Eve."
        else:
            yield "[happy]"
be = TagOnly(); d = tempfile.mkdtemp()
c = TestClient(create_app(be, DB(d), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "personas")))
j = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).json()
assert j["choices"][0]["message"]["content"] == "Hello, I am Eve." and j["pcore"]["emotion"] == "happy", j
assert be.calls[-1] == {"role": "assistant", "content": "[happy] "}
print("PREFILL OK")
