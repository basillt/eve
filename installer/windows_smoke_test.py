"""Install Eve on a real Windows machine and use it like a person would. Writes a report.

    python installer/windows_smoke_test.py dist/Eve-Setup-0.1.0.exe report-dir

Runs the installer silently, starts Eve the way the desktop shortcut does (pythonw, no console),
waits through the first-start setup (engine + a real Qwen model download), chats in English and
Arabic, reads a file, runs a night of learning, takes a screenshot, quits, and uninstalls.
Exit code 0 only if every step passed. Made for the GitHub Actions Windows runner.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
setup_exe, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
LOCAL = os.environ["LOCALAPPDATA"]
INST = os.path.join(LOCAL, "Programs", "Eve")
HOME = os.path.join(LOCAL, "Eve")
URL = "http://127.0.0.1:8000"
results, notes = [], []


def step(name):
    def wrap(fn):
        def run(*a, **k):
            t = time.time()
            try:
                value = fn(*a, **k)
                results.append((name, True, f"{time.time() - t:.0f}s"))
                return value
            except Exception as e:
                results.append((name, False, f"{type(e).__name__}: {e}"))
                notes.append(f"### {name} failed\n```\n{traceback.format_exc()}\n```")
                return None
        return run
    return wrap


def api(path, body=None, raw=None, ctype="application/json", timeout=600):
    data = raw if raw is not None else (json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None)
    req = urllib.request.Request(URL + path, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@step("install silently")
def install():
    subprocess.run([setup_exe, "/S"], check=True, timeout=600)
    for p in [os.path.join(INST, "python", "pythonw.exe"), os.path.join(INST, "app", "eve.pyw"),
              os.path.join(INST, "Uninstall.exe")]:
        assert os.path.exists(p), f"missing {p}"
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop", "Eve.lnk")
    menu = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows", "Start Menu", "Programs", "Eve", "Eve.lnk")
    assert os.path.exists(desktop), "no desktop shortcut"
    assert os.path.exists(menu), "no Start-menu shortcut"


@step("start like the desktop shortcut (pythonw)")
def start():
    subprocess.Popen([os.path.join(INST, "python", "pythonw.exe"), os.path.join(INST, "app", "eve.pyw"),
                      "--no-browser", "--night-at", "off"], cwd=os.path.join(INST, "app"))   # model chosen for this machine, like for users
    for _ in range(120):
        try:
            return api("/api/state", timeout=5)
        except Exception:
            time.sleep(1)
    raise TimeoutError("the app did not open within 2 minutes")


@step("first-start setup (engine + model download)")
def wait_ready():
    seen, last = [], None
    t0 = time.time()
    while time.time() - t0 < 40 * 60:
        st = api("/api/state", timeout=10)
        s = st["brain"].get("setup")
        if not s:
            notes.append("Setup steps seen: " + " → ".join(seen))
            notes.append(f"Engine: `{json.dumps(st['brain'], ensure_ascii=False)}`")
            return st
        key = s["step"] + (f" ({s.get('flavor')})" if s.get("flavor") else "")
        if key != last:
            seen.append(f"{key} @{time.time() - t0:.0f}s"); last = key
        if s["step"] == "error":
            raise RuntimeError(s["error"])
        time.sleep(2)
    raise TimeoutError("setup took longer than 40 minutes")


def chat(text, **extra):
    t = time.time()
    r = api("/v1/chat/completions", {"messages": [{"role": "user", "content": text}], "max_tokens": 200, **extra})
    answer = r["choices"][0]["message"]["content"].strip()
    notes.append(f"**Q:** {text}\n\n**A** ({time.time() - t:.0f}s, mood {r['pcore']['emotion']}"
                 f"{', from ' + ', '.join(s['name'] for s in r['pcore']['sources']) if r['pcore'].get('sources') else ''}):"
                 f" {answer}")
    assert answer, "empty answer"
    return r


@step("chat in English")
def chat_en():
    chat("In one or two sentences, what can you help me with?")


@step("chat in Arabic")
def chat_ar():
    r = chat("عرّفي عن حالك بجملة وحدة.")
    text = r["choices"][0]["message"]["content"]
    if sum("\u0621" <= ch <= "\u064a" for ch in text) < 10:
        notes.append("⚠️ the Arabic answer has few Arabic letters")


@step("read a file and answer from it")
def files():
    doc = api("/api/files?name=notes.txt", raw="Project notes.\nThe secret code word for the launch is PINEAPPLE.\n".encode(),
              ctype="application/octet-stream")
    r = chat("What is the secret code word in my notes?", docs=[doc["id"]])
    assert [s["name"] for s in r["pcore"]["sources"]] == ["notes.txt"], r["pcore"]
    if "PINEAPPLE" not in r["choices"][0]["message"]["content"].upper():
        notes.append("⚠️ the model did not repeat the code word, but the file was used")


@step("night of learning")
def night():
    chat("Remember that I study electrical engineering and I like to work early in the morning.")
    api("/api/night", {})
    t0 = time.time()
    while time.time() - t0 < 15 * 60:
        st = api("/api/state", timeout=10)
        if not st["sleeping"] and st.get("last_report"):
            notes.append(f"Night report: `{json.dumps(st['last_report'].get('learning'), ensure_ascii=False)}`")
            assert not st.get("night_error"), st["night_error"]
            if not (st["last_report"].get("learning") or {}).get("new_memories"):
                notes.append("⚠️ the night saved no memories")
            return
        time.sleep(3)
    raise TimeoutError("the night did not finish in 15 minutes")


@step("screenshot (Edge, headless)")
def screenshot():
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    shot = os.path.join(out, "eve-on-windows.png")
    subprocess.run([edge, "--headless=new", "--no-first-run", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--window-size=1440,900", "--virtual-time-budget=6000", f"--screenshot={shot}", URL + "/"],
                   timeout=120, check=False)
    assert os.path.exists(shot), "no screenshot"


@step("second click opens the same Eve")
def second():
    p = subprocess.run([os.path.join(INST, "python", "python.exe"), os.path.join(INST, "app", "eve.pyw"), "--no-browser"],
                       timeout=60, capture_output=True, text=True)
    assert "already running" in p.stdout, "a second Eve started: " + p.stdout[-300:]


@step("quit")
def quit_():
    subprocess.run([os.path.join(INST, "python", "python.exe"), os.path.join(INST, "app", "eve.pyw"), "--quit"], timeout=60)
    time.sleep(3)
    try:
        api("/api/state", timeout=3)
        raise AssertionError("still running after quit")
    except urllib.error.URLError:
        pass
    tasks = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    assert "llama-server" not in tasks, "llama-server still running"


@step("uninstall silently (keeps your data)")
def uninstall():
    subprocess.run([os.path.join(INST, "Uninstall.exe"), "/S", f"_?={INST}"], check=True, timeout=300)
    assert not os.path.exists(os.path.join(INST, "python")), "program files left behind"
    assert not os.path.exists(os.path.join(os.environ["USERPROFILE"], "Desktop", "Eve.lnk")), "desktop shortcut left"
    assert os.path.exists(os.path.join(HOME, "data")), "user data was deleted"


def report():
    for name in ("app.log", "llama-server.log"):
        p = os.path.join(HOME, "logs", name)
        if os.path.exists(p):
            shutil.copy(p, out)
    ok = all(r[1] for r in results)
    lines = [f"# Eve on Windows: {'all steps passed' if ok else 'some steps failed'}", "",
             f"Commit `{os.environ.get('GITHUB_SHA', '?')[:7]}` on `{os.environ.get('RUNNER_OS', 'Windows')}`, "
             f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
             "| Step | Result |", "|---|---|"]
    lines += [f"| {n} | {'✅ ' + d if passed else '❌ ' + d} |" for n, passed, d in results]
    lines += ["", "## Details", ""] + [n + "\n" for n in notes]
    with open(os.path.join(out, "windows-test-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return ok


install()
if start():
    wait_ready()
    if results[-1][1]:
        chat_en(); chat_ar(); files(); night(); screenshot(); second()
    quit_()
uninstall()
sys.exit(0 if report() else 1)
