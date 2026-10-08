"""N11 measurement probe (mp4), receipts only: not part of the recipe or the engine.

Loaded by Python's site module as `usercustomize` when the container runs with PYTHONPATH=/cache/tf/n11probe (run.sh
EXTRA_ENV; /cache/tf is run.sh's TF_CACHE/TF_SHA mount). It wraps, without changing what they compute:
  DeepSeekV41Engine.__init__ / ._lanes / ._warm / ._warm_lanes   (engine.py)
  Graphs.warm / LaneGraphs.warm                                   (graphs.py; their return value is the graph count)
and prints one `[n11probe]` line before and after each: torch.cuda.memory_reserved, memory_allocated, mem_get_info,
the host's MemAvailable and the elapsed seconds (after torch.cuda.synchronize). Any probe failure is printed and
ignored, so the engine runs as without it.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import socket
import sys
import time

GIB = 2**30


def _snap(cuda: bool = True) -> dict:
    import torch

    avail = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    if not cuda:            # before the engine's own CUDA setup: the host only
        return {"t": time.perf_counter(), "memavail": avail}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"t": time.perf_counter(), "reserved": torch.cuda.memory_reserved(),
            "allocated": torch.cuda.memory_allocated(), "dev_free": free, "dev_total": total, "memavail": avail}


def _emit(event: str, what: str, before: dict | None, after: dict, extra: dict | None = None) -> None:
    row = {"event": event, "what": what, "host": socket.gethostname(), "pid": os.getpid(),
           **{k: round(v / GIB, 3) for k, v in after.items() if k not in ("t",) and v is not None}}
    if before is not None:
        row["seconds"] = round(after["t"] - before["t"], 3)
        for k in ("reserved", "allocated", "dev_free", "memavail"):
            if before.get(k) is not None and after.get(k) is not None:
                row[f"d_{k}"] = round((after[k] - before[k]) / GIB, 3)
    if extra:
        row.update(extra)
    print("[n11probe] " + json.dumps(row), flush=True)


def _wrap(cls, name: str, what: str) -> None:
    fn = getattr(cls, name)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            before = _snap(cuda=not what.endswith("__init__"))
            _emit("before", what, None, before)
        except Exception as exc:  # noqa: BLE001
            print(f"[n11probe] {what} before failed: {exc!r}", flush=True)
            before = None
        out = fn(*args, **kwargs)
        try:
            extra = {"returned": out} if isinstance(out, int) else None
            if what.endswith("__init__"):
                self = args[0]
                extra = {"slots": getattr(self, "slots", None), "concurrent": getattr(self, "concurrent", None)}
            _emit("after", what, before, _snap(), extra)
        except Exception as exc:  # noqa: BLE001
            print(f"[n11probe] {what} after failed: {exc!r}", flush=True)
        return out

    setattr(cls, name, wrapped)


def _patch_graphs(mod) -> None:
    _wrap(mod.Graphs, "warm", "Graphs.warm")
    _wrap(mod.LaneGraphs, "warm", "LaneGraphs.warm")


def _patch_engine(mod) -> None:
    for name in ("__init__", "_lanes", "_warm", "_warm_lanes"):
        _wrap(mod.DeepSeekV41Engine, name, f"DeepSeekV41Engine.{name}")


PREFIX = "tensorfold.families.deepseek_v41.cuda."
TARGETS = {PREFIX + "graphs": _patch_graphs, PREFIX + "engine": _patch_engine}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module, run=run, name=name):
            run(module)
            try:
                TARGETS[name](module)
                print(f"[n11probe] patched {name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[n11probe] patch {name} failed: {exc!r}", flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
