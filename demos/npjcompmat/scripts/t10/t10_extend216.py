"""Complete the 216-site block of Table 1: the cases stopped after 10 failed attempts get seeds 11-100 as well.

216 sites is the lattice of the coarse-grained ensemble, so every case there gets the full 100 attempts (same
harness, cap, checks and 8 pinned performance cores as t10_latest.py). Larger lattices keep the early stop.
Appends to T10_OUT/t10_timing.csv and rewrites T10_OUT/t10_summary.csv (folders and cores as in t10_latest.py).

    python t10_extend216.py      (after t10_latest.py and t10_327.py, never at the same time as another timing run)
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import t10_latest as T


def main():
    df = pd.read_csv(T.TIMING)
    done = set(zip(df.method, df["size"], df.shells, df.target, df.seed))
    first = df[df.seed <= T.EARLY].groupby(["method", "size", "shells", "target"]).success.sum()
    stopped = [c for c, s in first.items() if s == 0 and c[1] == 6]
    print("216-site cases to complete:", stopped, flush=True)
    jobs = [(*c, seed) for c in stopped for seed in range(T.EARLY + 1, T.N_SEEDS + 1)]
    # interleave the cases so each one is spread over all cores
    jobs.sort(key=lambda j: (j[-1], j[:-1]))
    q = mp.Manager().Queue()
    for c in T.CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(T.CORES), initializer=T.pin, initargs=(q,)) as ex:
        T.run_jobs(jobs, ex, done)
    df = pd.read_csv(T.TIMING).drop_duplicates(["method", "size", "shells", "target", "seed"], keep="last")
    summ = T.summarize(df)
    summ.to_csv(os.path.join(T.OUT, "t10_summary.csv"), index=False)
    print(summ[summ.sites == 216].to_string())


if __name__ == "__main__":
    main()
