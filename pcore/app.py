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
import signal
import socket
import sys
import threading
import webbrowser

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


def start_engine(home, hw, model_path, cfg, args):
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
            exe = exe or runtime.ensure_llama(home, hw, flavor)
            print(f"  starting the engine ({flavor})…")
            server = LlamaServer(exe, model_path, ctx=args.ctx, log_path=os.path.join(logs, "llama-server.log"))
            server.start()
            cfg["flavor"] = flavor
            return server, flavor
        except Exception as e:
            errors.append(f"{flavor}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            print(f"  {flavor} did not work, trying the next one")
    raise SystemExit("the engine could not start:\n  " + "\n  ".join(errors) +
                     f"\n  details: {os.path.join(logs, 'llama-server.log')}")


def main(argv=None):
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
    args = ap.parse_args(argv)

    home = os.path.abspath(args.home)
    os.makedirs(home, exist_ok=True)
    cfg = load_config(home)
    hw = hardware.describe()
    gpu = ", ".join(f"{g['name']} {g['vram_gb']:.0f}GB" for g in hw["nvidia"]) or "no NVIDIA GPU"
    print(f"\n  Eve — {hw['os']}, {hw['ram_gb']:.0f} GB RAM, {gpu}")
    print(f"  folder: {home}")

    if args.model_file:
        model_path, label = os.path.abspath(args.model_file), os.path.basename(args.model_file)
    else:
        model_id = args.model or cfg.get("model") or runtime.pick_model(hw)
        info = runtime.model_info(model_id)
        model_path = runtime.ensure_model(home, model_id)
        cfg["model"] = model_id
        label = info["name"]

    pid_file = os.path.join(home, "runtime", "llama.pid")
    kill_leftover(pid_file)
    server, flavor = start_engine(home, hw, model_path, cfg, args)
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(server.proc.pid))
    save_config(home, cfg)

    def shutdown(*_):
        server.stop()
    atexit.register(shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: (shutdown(), sys.exit(0)))

    db = DB(os.path.join(home, "data"))
    backend = LlamaBackend(server, label, info={
        "flavor": flavor, "ram_gb": hw["ram_gb"], "vram_gb": hw["vram_gb"],
        "gpu": hw["nvidia"][0]["name"] if hw["nvidia"] else None})
    app = create_app(backend, db, seed_personas(home), args.max_tokens)
    if args.night_at and args.night_at != "off":
        schedule_night(app, args.night_at)

    port = args.port if port_free(args.port) else free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  ready: {url}\n  (keep this window open; close it to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
