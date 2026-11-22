"""
llama.py — run the model with llama.cpp's llama-server, and talk to it.

llama-server is started as a child process bound to 127.0.0.1 on a free port. It
loads a 4-bit GGUF, memory-maps it (the OS keeps the hot parts in RAM) and fits as
many layers on the GPU as there is room for (--fit). Our experiments showed this
plain setup (uniform 4-bit + request-level caching) beats static per-user tricks,
so the app builds on it instead of on its own engine.
"""

import json
import os
import platform
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import reflect


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, exe, model, ctx=8192, port=None, log_path=None, threads=None, extra=()):
        self.exe, self.model, self.ctx = exe, model, ctx
        self.port = port or free_port()
        self.log_path = log_path
        self.threads = threads
        self.extra = list(extra)
        self.proc = None
        self.url = f"http://127.0.0.1:{self.port}"

    def command(self):
        cmd = [self.exe, "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(self.ctx), "--jinja", "--reasoning", "off", "--no-webui", "-np", "1"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        return cmd + self.extra

    def start(self, timeout=300):
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        flags = 0x08000000 if platform.system() == "Windows" else 0      # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags,
                                     cwd=os.path.dirname(self.exe))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}\n{self.log_tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"llama-server did not become ready in {timeout}s\n{self.log_tail()}")

    def log_tail(self, n=25):
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, "rb") as f:
            lines = f.read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------ http
    def post(self, path, body, stream=False, timeout=600):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def get(self, path, timeout=10):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))


class LlamaBackend:
    """Backend for pcore.server, powered by a running LlamaServer."""
    name = "llama"

    def __init__(self, server, model_label, info=None):
        self.server = server
        self.model_id = model_label
        self.info = info or {}
        self.lock = threading.Lock()

    def _body(self, messages, max_tokens, temperature, stream):
        return {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": 0.9, "stream": stream}

    def stream(self, messages, max_tokens, temperature):
        with self.server.post("/v1/chat/completions", self._body(messages, max_tokens, temperature, True)) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    raise RuntimeError(obj["error"].get("message", str(obj["error"])))
                for ch in obj.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        yield piece

    def complete(self, messages, max_tokens=400, temperature=0.2):
        with self.server.post("/v1/chat/completions", self._body(messages, max_tokens, temperature, False)) as r:
            obj = json.loads(r.read().decode("utf-8"))
        return obj["choices"][0]["message"].get("content") or ""

    def brain(self):
        b = {"engine": "llama.cpp", "model": self.model_id, "ctx": self.server.ctx, "alive": self.server.alive()}
        b.update(self.info)
        try:
            b["model_bytes"] = os.path.getsize(self.server.model)
        except OSError:
            pass
        return b

    def flush_usage(self, db):
        pass

    def run_night(self, db, cfg_overrides):
        return reflect.run(self.complete, db)

    def rollback(self, db):
        return None
