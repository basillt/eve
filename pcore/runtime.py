"""
runtime.py — everything the app downloads on first start, standard library only.

  * llama.cpp's llama-server (the inference engine), the right build for this machine
  * one GGUF model, sized to this machine's memory

Nothing else is ever downloaded, and nothing is uploaded.
"""

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

LLAMA_TAG = "b12600"          # pinned llama.cpp release; falls back to the latest one if missing
UA = {"User-Agent": "eve/0.2"}

# Model sizes, smallest first. All Apache-2.0 (Qwen3.5 / Qwen3.6). size_gb is the Q4_K_M download.
MODELS = [
    {"id": "tiny", "label": "tiny (0.8B)", "name": "Qwen3.5 0.8B", "repos": ["unsloth/Qwen3.5-0.8B-GGUF", "bartowski/Qwen_Qwen3.5-0.8B-GGUF"],
     "size_gb": 0.6, "license": "Apache-2.0"},
    {"id": "small", "label": "small (2B)", "name": "Qwen3.5 2B", "repos": ["unsloth/Qwen3.5-2B-GGUF", "bartowski/Qwen_Qwen3.5-2B-GGUF"],
     "size_gb": 1.3, "license": "Apache-2.0"},
    {"id": "medium", "label": "medium (4B)", "name": "Qwen3.5 4B", "repos": ["unsloth/Qwen3.5-4B-GGUF", "bartowski/Qwen_Qwen3.5-4B-GGUF"],
     "size_gb": 2.8, "license": "Apache-2.0"},
    {"id": "large", "label": "large (9B)", "name": "Qwen3.5 9B", "repos": ["unsloth/Qwen3.5-9B-GGUF", "bartowski/Qwen_Qwen3.5-9B-GGUF"],
     "size_gb": 5.7, "license": "Apache-2.0"},
    {"id": "xl", "label": "extra large (35B MoE, 3B active)", "name": "Qwen3.6 35B-A3B", "repos": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
     "size_gb": 21.0, "license": "Apache-2.0"},
    # vision models (see images). They also chat, so turning vision on just switches Eve to one of these.
    {"id": "vision", "label": "vision (3B, sees images)", "name": "Qwen2.5-VL 3B", "vision": True,
     "repos": ["ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", "unsloth/Qwen2.5-VL-3B-Instruct-GGUF", "bartowski/Qwen2.5-VL-3B-Instruct-GGUF"],
     "size_gb": 3.2, "license": "Apache-2.0"},
    {"id": "vision-large", "label": "vision (7B, sees images)", "name": "Qwen2.5-VL 7B", "vision": True,
     "repos": ["ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "bartowski/Qwen2.5-VL-7B-Instruct-GGUF"],
     "size_gb": 6.5, "license": "Apache-2.0"},
]


def pkgs_dir(home):
    """Where on-demand Python packages (the OCR engine) are installed; added to sys.path at startup."""
    return os.path.join(home, "pkgs")


def ensure_ocr(home, log=print):
    """Install the image-text reader (rapidocr-onnxruntime) on first use. Returns True if ready.
    Kept out of the installer so the base download stays tiny; needs internet the first time only."""
    from . import docs as docslib
    if docslib.ocr_available():
        return True
    target = pkgs_dir(home)
    os.makedirs(target, exist_ok=True)
    if target not in sys.path:
        sys.path.insert(0, target)
    log("  installing the image-text reader (one-time)…")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                        "--target", target, "rapidocr-onnxruntime"],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        log(f"  image-text reader install failed: {e}")
        return False
    docslib.reset_ocr()
    return docslib.ocr_available()


def vision_models():
    return [m for m in MODELS if m.get("vision")]


def pick_vision(hw):
    """The vision model that fits this machine (7B if it has the memory, else 3B)."""
    usable = hw["ram_gb"] + hw.get("vram_gb", 0)
    return "vision-large" if usable >= 14 else "vision"
QUANT = "Q4_K_M"


def pick_model(hw):
    """Largest model this machine runs comfortably (RAM + VRAM are both usable by llama.cpp)."""
    ram, vram = hw.get("ram_gb", 0), hw.get("vram_gb", 0)
    if ram >= 40:
        return "xl"
    if ram >= 24 or (ram >= 14 and vram >= 6):
        return "large"
    if ram >= 10 or (ram >= 7 and vram >= 4):
        return "medium"
    if ram >= 6:
        return "small"
    return "tiny"


def model_info(model_id):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise SystemExit(f"unknown model size '{model_id}' (choose: {', '.join(m['id'] for m in MODELS)})")


def default_home():
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "Eve")
    if system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/Eve")
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "eve")


# ---------------------------------------------------------------- downloads

def _get_json(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def console_progress(label):
    state = {"t": 0.0}

    def cb(done, total):
        now = time.time()
        if now - state["t"] < 0.5 and done != total:
            return
        state["t"] = now
        if total:
            pct = 100 * done / total
            bar = "#" * int(pct / 4)
            sys.stdout.write(f"\r  {label}: [{bar:<25}] {pct:5.1f}%  {done / 1e9:.2f}/{total / 1e9:.2f} GB")
        else:
            sys.stdout.write(f"\r  {label}: {done / 1e6:.0f} MB")
        sys.stdout.flush()
        if total and done >= total:
            sys.stdout.write("\n")
    return cb


def download(url, dest, progress=None, chunk=1 << 20):
    """Download to dest (resumes a previous .part file). Returns dest."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:          # server ignored the range: start over
            have = 0
        total = r.headers.get("Content-Length")
        total = int(total) + have if total else None
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress:
                    progress(done, total)
    if total and os.path.getsize(part) != total:
        raise IOError(f"download incomplete: {url}")
    os.replace(part, dest)
    return dest


# ---------------------------------------------------------------- models

def resolve_gguf(repo, quant=QUANT):
    """(filename, size) of the main GGUF for `quant` in a Hugging Face repo."""
    info = _get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    files = [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", [])]
    q = quant.lower()
    cands = [(f, n) for f, n in files if f.lower().endswith(".gguf") and q in f.lower()
             and "mmproj" not in f.lower() and "/" not in f]
    if not cands:
        raise FileNotFoundError(f"no {quant} GGUF in {repo}")
    exact = [c for c in cands if c[0].lower().endswith(f"-{q}.gguf") or c[0].lower().endswith(f".{q}.gguf")]
    return sorted(exact or cands, key=lambda c: len(c[0]))[0]


def resolve_mmproj(repo, quant=QUANT):
    """(filename, size) of the vision projector (mmproj) GGUF in a repo, if any."""
    info = _get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    cands = [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", [])
             if "mmproj" in s["rfilename"].lower() and s["rfilename"].lower().endswith(".gguf") and "/" not in s["rfilename"]]
    if not cands:
        raise FileNotFoundError(f"no mmproj in {repo}")
    # prefer an f16 projector (small and accurate); else the smallest
    best = [c for c in cands if "f16" in c[0].lower()] or cands
    return sorted(best, key=lambda c: c[1] or len(c[0]))[0]


def ensure_mmproj(home, model_id, repo, progress_factory=console_progress):
    """Path to the mmproj file for a vision model, downloading it next to the model."""
    folder = os.path.join(home, "models", model_id)
    have = [f for f in os.listdir(folder) if "mmproj" in f.lower() and f.endswith(".gguf")] if os.path.isdir(folder) else []
    if have:
        return os.path.join(folder, have[0])
    name, size = resolve_mmproj(repo)
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    print(f"  vision projector: {repo}/{name} ({size / 1e9:.2f} GB)")
    return download(url, os.path.join(folder, name), progress_factory("vision") if progress_factory else None)


def ensure_model(home, model_id, progress_factory=console_progress):
    """Path to the GGUF for this model size, downloading it on first use."""
    m = model_info(model_id)
    folder = os.path.join(home, "models", model_id)
    if os.path.isdir(folder):
        have = [f for f in os.listdir(folder) if f.endswith(".gguf") and "mmproj" not in f.lower()]
        if have:
            return os.path.join(folder, have[0])
    errors = []
    for repo in m["repos"]:
        try:
            name, size = resolve_gguf(repo)
        except Exception as e:
            errors.append(f"{repo}: {e}")
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        print(f"  model: {repo}/{name} ({size / 1e9:.1f} GB, {m['license']})")
        dest = os.path.join(folder, name)
        download(url, dest, progress_factory("model") if progress_factory else None)
        with open(os.path.join(folder, "source.json"), "w") as f:
            json.dump({"repo": repo, "file": name, "license": m["license"], "url": url}, f, indent=1)
        return dest
    raise SystemExit("could not download the model:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------- llama.cpp

def flavors(hw):
    """llama.cpp builds to try on this machine, best first."""
    os_, arch = hw["os"], hw["arch"]
    if os_ == "Darwin":
        return ["metal"]
    out = []
    if os_ == "Windows" and hw.get("nvidia") and arch == "x64":     # on Linux, Vulkan covers NVIDIA too
        from .hardware import cuda_driver_ok
        if any(cuda_driver_ok(g["driver"]) for g in hw["nvidia"]):
            out.append("cuda")
    out += ["vulkan", "cpu"]
    return out


def asset_names(hw, flavor, tag):
    """Release files for one flavor. The first one holds llama-server."""
    os_, arch = hw["os"], hw["arch"]
    if os_ == "Windows":
        if flavor == "cuda":
            return [f"llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
        return [f"llama-{tag}-bin-win-{flavor}-{arch}.zip"]
    if os_ == "Darwin":
        return [f"llama-{tag}-bin-macos-{arch}.tar.gz"]
    if flavor == "vulkan":
        return [f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"]
    return [f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"]


def _extract(archive, folder):
    os.makedirs(folder, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                target = os.path.realpath(os.path.join(folder, member))
                if not target.startswith(os.path.realpath(folder)):
                    raise IOError(f"unsafe path in {archive}: {member}")
            z.extractall(folder)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(folder, filter="data")
            except TypeError:                    # Python < 3.12
                t.extractall(folder)


def find_server(folder):
    name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for root, _, files in os.walk(folder):
        if name in files:
            path = os.path.join(root, name)
            if platform.system() != "Windows":
                os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return path
    return None


def latest_tag():
    return _get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")["tag_name"]


def ensure_llama(home, hw, flavor, progress_factory=console_progress):
    """Path to llama-server for this flavor, downloading the release on first use."""
    folder = os.path.join(home, "runtime", "llama", flavor)
    exe = find_server(folder) if os.path.isdir(folder) else None
    if exe:
        return exe
    tmp = os.path.join(home, "runtime", "downloads")
    for tag in (LLAMA_TAG, None):
        try:
            tag = tag or latest_tag()
            names = asset_names(hw, flavor, tag)
            for i, name in enumerate(names):
                url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
                path = download(url, os.path.join(tmp, name), progress_factory(f"engine ({flavor})") if progress_factory else None)
                _extract(path, folder)
            break
        except urllib.error.HTTPError as e:
            if e.code != 404 or tag != LLAMA_TAG:
                raise
            shutil.rmtree(folder, ignore_errors=True)
    exe = find_server(folder)
    if not exe:
        raise FileNotFoundError(f"llama-server not found in the {flavor} build")
    if platform.system() != "Windows":       # shared libraries sit next to the binary
        for root, _, files in os.walk(folder):
            for f in files:
                if ".so" in f or f.endswith(".dylib"):
                    p = os.path.join(root, f)
                    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
    return exe
