"""How much of a C time is the harness and the program start, measured on one pinned performance core (idle machine).

For C exact on target A at 216 sites (one shell), 50 sequential runs each, measures:
  wall_affinity    the t10 harness (Popen, psutil affinity call, wait)
  wall_inherit     Popen and wait only (affinity inherited from the pinned parent)
  lifetime         the process's own lifetime from GetProcessTimes (creation to exit, 100 ns resolution)
  cpu              its user + kernel CPU time
and the same lifetime for a call that only prints the usage and exits (program start alone).

    python t10_overhead.py
"""
import ctypes
import os
import shutil
import subprocess
import tempfile
import time
from ctypes import wintypes

import numpy as np
import psutil

import t10_latest as T

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def times(handle):
    c, e, k, u = (wintypes.FILETIME() for _ in range(4))
    assert k32.GetProcessTimes(wintypes.HANDLE(int(handle)), ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
    f = lambda t: (t.dwHighDateTime << 32 | t.dwLowDateTime) * 1e-7  # noqa: E731
    return f(e) - f(c), f(k) + f(u)


def one(mode, args):
    d = tempfile.mkdtemp(prefix="t10o_", dir=T.TMP)
    cmd = [T.EXE, *args, f"--output-dir={os.path.join(d, 'output')}"] if args else [T.EXE]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if mode == "affinity":
        psutil.Process(p.pid).cpu_affinity([0])
    p.wait()
    wall = time.perf_counter() - t0
    life, cpu = times(p._handle)
    shutil.rmtree(d, ignore_errors=True)
    return wall, life, cpu


def main():
    psutil.Process().cpu_affinity([0])
    target = T.rescale(T.TARGETS["A"], 216)
    args = ["6x6x6", "111", "6", "100000000", "1", T.spec_of(target), "0", "SC", "1.0",
            "--search=exact", "--min-giant-fraction=1", "--seed=1"]
    for mode in ("affinity", "inherit"):
        r = np.array([one(mode, args) for _ in range(50)])
        print(f"exact A 216, {mode:8s}: wall median {np.median(r[:, 0]) * 1e3:.1f} ms, lifetime {np.median(r[:, 1]) * 1e3:.1f} ms,"
              f" cpu {np.median(r[:, 2]) * 1e3:.1f} ms  (wall p10-p90 {np.percentile(r[:, 0], 10) * 1e3:.1f}-{np.percentile(r[:, 0], 90) * 1e3:.1f})")
    r = np.array([one("inherit", None) for _ in range(50)])
    print(f"usage only (program start): wall {np.median(r[:, 0]) * 1e3:.1f} ms, lifetime {np.median(r[:, 1]) * 1e3:.1f} ms")
    py = []
    for s in range(1, 21):
        G, w = T.run_py("exact", 6, 1, target, s)
        py.append(w)
    print(f"python exact A 216 in-process: median {np.median(py) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
