"""System monitor: CPU, RAM, GPU and the processes that run the models."""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

import psutil

_cache = {"t": 0, "data": None}
_procs = {}
_lock = threading.Lock()
_gpu_name = None


def _gpu():
    """GPU usage from IOKit (no admin rights needed). Returns (name, percent) or (name, None)."""
    global _gpu_name
    if _gpu_name is None:
        try:
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"Chipset Model: (.+)", out)
            _gpu_name = m.group(1).strip() if m else "GPU"
        except Exception:
            _gpu_name = "GPU"
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out) or re.search(r'"Device Utilization % at cur p-state"=(\d+)', out)
        return _gpu_name, int(m.group(1)) if m else None
    except Exception:
        return _gpu_name, None


def _ollama_models():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=0.5) as r:
            models = json.load(r).get("models", [])
        return [{"name": m["name"], "size": m.get("size", 0), "vram": m.get("size_vram", 0),
                 "context": m.get("context_length")} for m in models]
    except Exception:
        return None


def _process_groups():
    me = os.getpid()
    groups = {"Ollama": [0.0, 0], "Neural Chat": [0.0, 0]}
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower()
        group = "Ollama" if ("ollama" in name or "llama" in name) else "Neural Chat" if p.pid == me else None
        if not group:
            continue
        proc = _procs.setdefault(p.pid, p)
        try:
            groups[group][0] += proc.cpu_percent(None)
            groups[group][1] += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            _procs.pop(p.pid, None)
    return {k: {"cpu": round(v[0], 1), "ram": v[1]} for k, v in groups.items()}


def snapshot():
    with _lock:
        if time.time() - _cache["t"] < 0.8 and _cache["data"]:
            return _cache["data"]
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        gpu_name, gpu = _gpu()
        data = {
            "cpu": psutil.cpu_percent(None),
            "cores": psutil.cpu_percent(None, percpu=True),
            "cpu_count": psutil.cpu_count(),
            "ram_total": vm.total, "ram_used": vm.total - vm.available, "swap_used": sw.used,
            "gpu_name": gpu_name, "gpu": gpu,
            "procs": _process_groups(),
            "ollama": _ollama_models(),
        }
        _cache.update(t=time.time(), data=data)
        return data
