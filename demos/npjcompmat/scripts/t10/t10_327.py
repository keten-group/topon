"""The supplementary benchmark table, rerun with the latest topon: one network per coarse-grained ensemble target,
every method.

The 327 targets are the degree counts of the deposited networks (companion data/mechanics folder names, SC 6^3,
one shell, max functionality 6). Each method gets one attempt per target with seed 1 and the 120 s cap of Table 1
(C_pruning, python_pruning, C_exact, python_exact as defined in t10_latest.py), on the same 8 pinned performance
cores. Every returned network is checked (exact counts, connected active sites).

Writes T10_OUT/t10_327.csv and prints the summary (targets reached, median and 90th percentile time per network).
Folders and cores as in t10_latest.py.

    python t10_327.py      (run after t10_latest.py, never at the same time)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import t10_latest as T

MECH = os.path.join(T.COMPANION, "data", "mechanics")
OUTF = os.path.join(T.OUT, "t10_327.csv")


def targets():
    names = sorted(d for d in os.listdir(MECH) if os.path.isdir(os.path.join(MECH, d)) and d.count("_") == 6)
    t = [[int(x) for x in n.split("_")] for n in names]
    assert len(t) == 327, len(t)
    for c in t:
        assert sum(c) == 216
    return names, t


def one(job):
    method, name, counts = job
    search = "strict" if method.endswith("pruning") else "exact"
    G, wall = (T.run_c if method.startswith("C_") else T.run_py)(search, 6, 1, np.asarray(counts), 1)
    ok, h = T.check_and_hash(G, counts) if G is not None else (False, "")
    return dict(method=method, target=name, success=bool(G is not None and ok), returned=G is not None,
                wall_s=round(wall, 6), hash=h, core=T._core)


def main():
    import multiprocessing as mp
    names, counts = targets()
    done = set()
    if os.path.exists(OUTF):
        d0 = pd.read_csv(OUTF)
        done = set(zip(d0.method, d0.target))
    jobs = [(m, n, c) for m in T.METHODS for n, c in zip(names, counts) if (m, n) not in done]
    q = mp.Manager().Queue()
    for c in T.CORES:
        q.put(c)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(T.CORES), initializer=T.pin, initargs=(q,)) as ex:
        futures = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            pd.DataFrame([f.result()]).to_csv(OUTF, mode="a", header=not os.path.exists(OUTF), index=False)
            if i % 100 == 0 or i == len(futures):
                print(f"{i}/{len(futures)} in {time.time() - t0:6.0f} s", flush=True)
    df = pd.read_csv(OUTF)
    for m, x in df.groupby("method"):
        ok = x[x.success]
        print(f"{m:15s} reached {len(ok)}/{len(x)}  median {ok.wall_s.median():.4g} s  "
              f"p90 {ok.wall_s.quantile(0.9):.4g} s  max {ok.wall_s.max():.4g} s")


if __name__ == "__main__":
    main()
