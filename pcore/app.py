"""
app.py — the app for normal users: one command, everything else is automatic.

    python -m pcore.app

First start: looks at the machine, downloads the llama.cpp engine that fits it
(CUDA / Vulkan / CPU / Metal) and a model sized to its memory, then opens the app
in the browser. Later starts work offline. Everything lives in one folder:

    Windows  %LOCALAPPDATA%\\Eve      macOS  ~/Library/Application Support/Eve
    Linux    ~/.local/share/eve

Options: --model tiny|small|medium|large|xl   --model-file my.gguf   --llama-server path
         --home folder   --port 8000   --no-browser   --night-at 03:00
"""

import argparse
import atexit
import json
import os
import shutil
import platform
import signal
import subprocess
import socket
import sys
import threading
import webbrowser

from fastapi import HTTPException
from . import hardware, runtime
from .db import DB
from .llama import LlamaBackend, LlamaServer, free_port

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(home):
    try:
        with open(os.path.join(home, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(home, cfg):
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)


def seed_personas(home):
    """Built-in characters are copied once into the user's folder (where new ones are saved too)."""
    folder = os.path.join(home, "personas")
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(ROOT, "personas")
    for f in os.listdir(src) if os.path.isdir(src) else []:
        if f.endswith(".json") and not os.path.exists(os.path.join(folder, f)):
            shutil.copy(os.path.join(src, f), folder)
    return folder


def _chromium():
    """Path to a Chromium browser for app-mode (its own window, no tabs or address bar)."""
    sysname = platform.system()
    if sysname == "Windows":
        import glob
        roots = [os.environ.get("PROGRAMFILES", r"C:\\Program Files"),
                 os.environ.get("PROGRAMFILES(X86)", r"C:\\Program Files (x86)"),
                 os.environ.get("LOCALAPPDATA", "")]
        for root in roots:
            for rel in (r"Microsoft\\Edge\\Application\\msedge.exe", r"Google\\Chrome\\Application\\chrome.exe"):
                hit = glob.glob(os.path.join(root, rel))
                if hit:
                    return hit[0]
        return None
    if sysname == "Darwin":
        for app in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                    "/Applications/Chromium.app/Contents/MacOS/Chromium"):
            if os.path.exists(app):
                return app
        return None
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"):
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def open_window(url, home):
    """Open Eve in its own window if a Chromium browser is around; else fall back to a browser tab."""
    exe = _chromium()
    if not exe:
        webbrowser.open(url)
        return
    profile = os.path.join(home, "runtime", "window")       # a private profile, separate from the user's browser
    try:
        flags = 0x08000000 if platform.system() == "Windows" else 0
        subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1200,840",
                          "--no-first-run", "--no-default-browser-check"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         creationflags=flags, start_new_session=(platform.system() != "Windows"))
    except Exception:
        webbrowser.open(url)


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def kill_leftover(pid_file):
    """A llama-server left behind by a crashed earlier run would hold RAM: stop it."""
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        import psutil
        p = psutil.Process(pid)
        if "llama-server" in p.name():
            p.terminate()
    except Exception:
        pass


def start_engine(home, hw, model_path, cfg, args, progress_factory=runtime.console_progress, on_try=None, mmproj=None):
    """Start llama-server with the first build that works on this machine."""
    if args.llama_server:
        order = [("custom", args.llama_server)]
    else:
        names = runtime.flavors(hw)
        if cfg.get("flavor") in names:                    # what worked last time goes first
            names.remove(cfg["flavor"])
            names.insert(0, cfg["flavor"])
        order = [(n, None) for n in names]
    logs = os.path.join(home, "logs")
    os.makedirs(logs, exist_ok=True)
    errors = []
    for flavor, exe in order:
        try:
            if on_try:
                on_try(flavor, "engine")
            exe = exe or runtime.ensure_llama(home, hw, flavor, progress_factory)
            if on_try:
                on_try(flavor, "start")
            print(f"  starting the engine ({flavor})…")
            server = LlamaServer(exe, model_path, ctx=args.ctx, log_path=os.path.join(logs, "llama-server.log"), mmproj=mmproj)
            server.start()
            cfg["flavor"] = flavor
            return server, flavor
        except Exception as e:
            errors.append(f"{flavor}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            print(f"  {flavor} did not work, trying the next one")
    raise RuntimeError("the engine could not start:\n  " + "\n  ".join(errors) +
                       f"\n  details: {os.path.join(logs, 'llama-server.log')}")


class LateBackend:
    """Stands in for the real backend while Eve gets ready, so the app can open at once and
    show the setup progress. When the engine is up, `ready()` hands everything to it."""

    def __init__(self, setup):
        self.setup, self.target = setup, None
        self.name, self.model_id, self.vision = "setup", "…", False

    def ready(self, backend):
        self.target = backend
        self.name, self.model_id, self.vision = backend.name, backend.model_id, backend.vision

    def stream(self, messages, max_tokens, temperature):
        if not self.target:
            raise RuntimeError("Eve is still getting ready")
        return self.target.stream(messages, max_tokens, temperature)

    def complete(self, *a, **k):
        return self.target.complete(*a, **k)

    def brain(self):
        return self.target.brain() if self.target else {"setup": dict(self.setup)}

    def flush_usage(self, db):
        if self.target:
            self.target.flush_usage(db)

    def run_night(self, db, cfg):
        if not self.target:
            raise RuntimeError("Eve is still getting ready")
        return self.target.run_night(db, cfg)

    def rollback(self, db):
        return self.target.rollback(db) if self.target else None


def log_to_file(home):
    """With no console (the desktop shortcut runs pythonw) print() has nowhere to go: use a log file."""
    if sys.stdout is not None and sys.stderr is not None and not os.environ.get("EVE_LOG_TO_FILE"):
        return None
    os.makedirs(os.path.join(home, "logs"), exist_ok=True)
    path = os.path.join(home, "logs", "app.log")
    f = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f
    return path


def running_instance(home):
    """If Eve is already running, its address (so a second click just opens the browser)."""
    try:
        with open(os.path.join(home, "runtime", "app.json"), encoding="utf-8") as f:
            info = json.load(f)
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{int(info['port'])}/api/state", timeout=2) as r:
            if r.status == 200 and "backend" in json.loads(r.read().decode("utf-8")):
                return f"http://127.0.0.1:{int(info['port'])}"
    except Exception:
        return None
    return None


def main(argv=None):
    import time
    import uvicorn
    from .server import create_app, schedule_night

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=runtime.default_home())
    ap.add_argument("--model", default="", help="model size (default: chosen for this machine)")
    ap.add_argument("--model-file", default="", help="use your own .gguf file")
    ap.add_argument("--llama-server", default="", help="use your own llama-server binary")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--night-at", default="03:00", help="HH:MM, or 'off'")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--quit", action="store_true", help="stop a running Eve (used by the installer)")
    args = ap.parse_args(argv)

    home = os.path.abspath(args.home)
    os.makedirs(os.path.join(home, "runtime"), exist_ok=True)
    pkgs = runtime.pkgs_dir(home)                     # on-demand packages (OCR) live here
    if os.path.isdir(pkgs) and pkgs not in sys.path:
        sys.path.insert(0, pkgs)
    log_to_file(home)
    url = running_instance(home)
    if args.quit:
        if url:
            import urllib.request
            try:
                urllib.request.urlopen(urllib.request.Request(url + "/api/quit", data=b"{}", method="POST",
                                       headers={"Content-Type": "application/json"}), timeout=5)
                time.sleep(2)
            except Exception:
                pass
        return
    if url:
        print(f"  Eve is already running: {url}")
        if not args.no_browser:
            open_window(url, home)
        return

    cfg = load_config(home)
    setup = {"step": "check", "flavor": None, "model": None, "size_gb": None, "done": 0, "total": 0, "error": None,
             "logs": os.path.join(home, "logs")}
    late = LateBackend(setup)
    db = DB(os.path.join(home, "data"))
    app = create_app(late, db, seed_personas(home), args.max_tokens)
    servers = []

    def shutdown(*_):
        for s in servers:
            s.stop()
    atexit.register(shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: (shutdown(), sys.exit(0)))

    def progress(label):
        def cb(done, total):
            setup["done"], setup["total"] = done, total or 0
        return cb

    def prepare():
        try:
            setup.update(step="check", error=None, done=0, total=0)
            hw = hardware.describe()
            gpu = ", ".join(f"{g['name']} {g['vram_gb']:.0f}GB" for g in hw["nvidia"]) or "no NVIDIA GPU"
            print(f"\n  Eve — {hw['os']}, {hw['ram_gb']:.0f} GB RAM, {gpu}\n  folder: {home}")
            mmproj, is_vision = None, False
            if args.model_file:
                model_path, label = os.path.abspath(args.model_file), os.path.basename(args.model_file)
            else:
                model_id = args.model or cfg.get("model") or runtime.pick_model(hw)
                info = runtime.model_info(model_id)
                is_vision = bool(info.get("vision"))
                setup.update(step="model", model=info["name"], size_gb=info.get("size_gb"), done=0, total=0, vision=is_vision)
                model_path = runtime.ensure_model(home, model_id, progress)
                if is_vision:
                    try:
                        with open(os.path.join(home, "models", model_id, "source.json"), encoding="utf-8") as f:
                            repo = json.load(f)["repo"]
                    except (OSError, ValueError, KeyError):
                        repo = info["repos"][0]
                    setup.update(step="vision", done=0, total=0)
                    mmproj = runtime.ensure_mmproj(home, model_id, repo, progress)
                cfg["model"] = model_id
                label = info["name"]
            setup["model"] = label
            pid_file = os.path.join(home, "runtime", "llama.pid")
            kill_leftover(pid_file)

            def on_try(flavor, step):
                setup.update(step=step, flavor=flavor, done=0, total=0)
            server, flavor = start_engine(home, hw, model_path, cfg, args, progress, on_try, mmproj=mmproj)
            servers.append(server)
            with open(pid_file, "w") as f:
                f.write(str(server.proc.pid))
            save_config(home, cfg)
            late.ready(LlamaBackend(server, label, vision=is_vision, info={
                "flavor": flavor, "ram_gb": hw["ram_gb"], "vram_gb": hw["vram_gb"],
                "gpu": hw["nvidia"][0]["name"] if hw["nvidia"] else None}))
            setup["step"] = "ready"
            if args.night_at and args.night_at != "off":
                schedule_night(app, args.night_at)
            print("  ready")
        except BaseException as e:                       # includes SystemExit from the download helpers
            setup.update(step="error", error=str(e) or type(e).__name__)
            print("  setup failed:", setup["error"])

    @app.post("/api/setup/retry")
    def retry():
        if setup["step"] == "error":
            threading.Thread(target=prepare, daemon=True).start()
        return {"ok": True}

    switching = threading.Lock()

    @app.post("/api/vision")
    def set_vision(on: dict):
        want = bool(on.get("on"))
        if args.model_file:
            raise HTTPException(400, "a model file was given on the command line; vision can't be switched")
        if not switching.acquire(blocking=False):
            raise HTTPException(409, "already switching")
        try:
            hw = hardware.describe()
            is_vision_now = bool(runtime.model_info(cfg.get("model") or runtime.pick_model(hw)).get("vision"))
            if want == is_vision_now and late.target is not None:
                return {"ok": True, "vision": want}
            if want:
                if not runtime.model_info(cfg.get("model", "")).get("vision"):
                    cfg["text_model"] = cfg.get("model") or runtime.pick_model(hw)
                cfg["model"] = runtime.pick_vision(hw)
            else:
                cfg["model"] = cfg.get("text_model") or runtime.pick_model(hw)
            save_config(home, cfg)
            for sv in servers:
                sv.stop()
            servers.clear()
            late.target = None
            late.name, late.model_id = "setup", "…"
            setup.update(step="check", error=None, done=0, total=0, flavor=None, model=None, vision=want)
            threading.Thread(target=prepare, daemon=True).start()
            return {"ok": True, "vision": want}
        finally:
            switching.release()

    @app.post("/api/quit")
    def quit_app():
        def bye():
            time.sleep(0.4)
            shutdown()
            os._exit(0)
        threading.Thread(target=bye, daemon=True).start()
        return {"ok": True}

    port = args.port if args.port and port_free(args.port) else free_port()
    url = f"http://127.0.0.1:{port}"
    with open(os.path.join(home, "runtime", "app.json"), "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "port": port}, f)
    threading.Thread(target=prepare, daemon=True).start()
    print(f"\n  open: {url}\n  (Eve keeps running in the background; quit it from Settings)\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: open_window(url, home)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
