"""
hardware.py — what is this machine? (RAM, NVIDIA/other GPU, OS) — standard library only.

Used once at first start to pick a model size and the right llama.cpp build.
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess


def ram_gb():
    """Total physical memory in GB (0 if unknown)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Windows":
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1e9
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) / 1e9
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return 0.0


def nvidia_gpus():
    """[(name, vram_gb, driver_version)] for NVIDIA GPUs, via nvidia-smi (empty if none)."""
    exe = shutil.which("nvidia-smi")
    if not exe and platform.system() == "Windows":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append((parts[0], float(parts[1]) / 1024, parts[2]))
            except ValueError:
                pass
    return gpus


def cuda_driver_ok(driver_version, needed="528.33"):
    """CUDA 12.4 builds need a reasonably recent driver (Windows 528.33+, Linux 525.60+)."""
    def nums(v):
        return [int(x) for x in re.findall(r"\d+", v)[:2]] or [0]
    if platform.system() != "Windows":
        needed = "525.60"
    return nums(driver_version) >= nums(needed)


def describe():
    gpus = nvidia_gpus()
    return {
        "os": platform.system(),                  # Windows / Linux / Darwin
        "arch": "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64",
        "ram_gb": round(ram_gb(), 1),
        "nvidia": [{"name": n, "vram_gb": round(v, 1), "driver": d} for n, v, d in gpus],
        "vram_gb": round(max((v for _, v, _ in gpus), default=0.0), 1),
        "cpu_threads": os.cpu_count() or 4,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
