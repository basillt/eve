"""Build the Windows installer (dist/Eve-Setup-<version>.exe) from Linux, macOS or Windows.

    python installer/build_windows.py [--version 0.1.0]

Needs: makensis (NSIS 3), pip, and internet for the first build. Steps:
  1. a private Python for Windows (python-build-standalone, checked by SHA-256)
  2. the app's packages as Windows wheels, installed into that Python
  3. Eve itself (pcore/, web/, personas/)
  4. makensis installer/eve.nsi
The installer needs no admin rights: it installs into %LOCALAPPDATA%\\Programs\\Eve and
adds a desktop and Start-menu shortcut. Eve's data stays in %LOCALAPPDATA%\\Eve.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
          "cpython-3.12.16%2B20261110-x86_64-pc-windows-msvc-install_only_stripped.tar.gz")
PY_SHA256 = "355ee61b8207b91574491b1c6dc9b0ebb4408fc56bfafe498f3ef439314b52dc"
TRIM = ["Scripts", "include", "libs", "Lib/idlelib", "Lib/tkinter",   # keep pip + ensurepip: on-demand OCR installs with them
        "Lib/turtledemo", "Lib/test", "Lib/lib2to3", "tcl", "DLLs/_tkinter.pyd", "Lib/site-packages/bin"]
APP = ["pcore", "web", "personas", "LICENSE", "requirements-app.txt"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def python_runtime(build):
    cache = os.path.join(HERE, "cache"); os.makedirs(cache, exist_ok=True)
    tgz = os.path.join(cache, "python-win64.tar.gz")
    if not os.path.exists(tgz) or sha256(tgz) != PY_SHA256:
        print("downloading Python for Windows…")
        urllib.request.urlretrieve(PY_URL, tgz)
    if sha256(tgz) != PY_SHA256:
        raise SystemExit("the Python download does not match its SHA-256")
    with tarfile.open(tgz) as t:
        t.extractall(build, filter="data")
    py = os.path.join(build, "python")
    site = os.path.join(py, "Lib", "site-packages")
    print("installing the app's packages (Windows wheels)…")
    common = [sys.executable, "-m", "pip", "install", "-q", "--target", site, "--platform", "win_amd64",
              "--python-version", "3.12", "--implementation", "cp", "--only-binary=:all:", "--no-compile"]
    subprocess.check_call(common + ["-r", os.path.join(ROOT, "requirements-app.txt")])
    subprocess.check_call(common + ["--no-deps", "colorama"])      # click needs it on Windows
    for t in TRIM:
        p = os.path.join(py, t)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    for root, dirs, _ in os.walk(py):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d)); dirs.remove(d)


def app_files(build):
    dst = os.path.join(build, "app")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "_brain_test.html")
    for name in APP:
        src = os.path.join(ROOT, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, name), ignore=ignore)
        else:
            os.makedirs(dst, exist_ok=True); shutil.copy(src, dst)
    shutil.copy(os.path.join(HERE, "eve.pyw"), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0")
    args = ap.parse_args()
    if not shutil.which("makensis"):
        raise SystemExit("makensis not found: install NSIS 3 (Ubuntu: apt install nsis)")
    build = os.path.join(HERE, "build")
    shutil.rmtree(build, ignore_errors=True); os.makedirs(build)
    python_runtime(build)
    app_files(build)
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    print("building the installer…")
    subprocess.check_call(["makensis", "-V2", f"-DVERSION={args.version}", os.path.join(HERE, "eve.nsi")])
    out = os.path.join(ROOT, "dist", f"Eve-Setup-{args.version}.exe")
    print(f"done: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
