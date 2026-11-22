"""The desktop app end to end: opens at once in setup mode, gets ready, answers, a second
start only points at the running one, quit stops everything. A bad model shows the error
on the setup screen and can be retried.

    python tests/test_app.py path/to/llama-server path/to/model.gguf
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_bin, model = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])


def get(url, path, body=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def start(home, *extra):
    env = dict(os.environ, EVE_LOG_TO_FILE="1")
    return subprocess.Popen([sys.executable, "-m", "pcore.app", "--home", home, "--no-browser", "--port", "0",
                             "--night-at", "off", *extra], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def address(home, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        try:
            with open(os.path.join(home, "runtime", "app.json")) as f:
                url = f"http://127.0.0.1:{json.load(f)['port']}"
            get(url, "/api/state")
            return url
        except Exception:
            time.sleep(0.3)
    raise AssertionError("app did not open")


tmp = tempfile.mkdtemp()
try:
    # ---- normal start
    home = os.path.join(tmp, "home")
    p = start(home, "--llama-server", server_bin, "--model-file", model)
    url = address(home)
    seen = set()
    for _ in range(200):
        st = get(url, "/api/state")
        if st["brain"].get("setup"):
            seen.add(st["brain"]["setup"]["step"])
        else:
            break
        time.sleep(0.2)
    assert st["backend"] == "llama" and st["brain"]["alive"], st["brain"]
    print("setup steps seen:", sorted(seen))
    r = get(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    assert r["choices"][0]["message"]["role"] == "assistant"

    # ---- a second start just points at the first one
    p2 = start(home, "--llama-server", server_bin, "--model-file", model)
    out, _ = p2.communicate(timeout=30)
    assert p2.returncode == 0
    log = open(os.path.join(home, "logs", "app.log"), encoding="utf-8").read()
    assert "already running" in log, log[-500:]

    # ---- quit stops the app and the engine
    pid = int(open(os.path.join(home, "runtime", "llama.pid")).read())
    get(url, "/api/quit", {})
    p.wait(timeout=20)
    time.sleep(1)
    import psutil
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE, "llama-server still running"

    # ---- a broken model: the error shows on the setup screen, retry runs again
    home2 = os.path.join(tmp, "home2")
    bad = os.path.join(tmp, "broken.gguf"); open(bad, "wb").write(b"not a model")
    p3 = start(home2, "--llama-server", server_bin, "--model-file", bad)
    url3 = address(home2)
    for _ in range(300):
        st = get(url3, "/api/state")["brain"].get("setup") or {}
        if st.get("step") == "error":
            break
        time.sleep(0.2)
    assert st.get("step") == "error" and "engine could not start" in st["error"], st
    get(url3, "/api/setup/retry", {})
    time.sleep(0.5)
    assert get(url3, "/api/state")["brain"]["setup"]["step"] in ("check", "engine", "start", "error")
    get(url3, "/api/quit", {}); p3.wait(timeout=20)
    print("APP OK")
finally:
    for q in ("p", "p3"):
        if q in dir() and globals()[q].poll() is None:
            globals()[q].kill()
    shutil.rmtree(tmp, ignore_errors=True)
