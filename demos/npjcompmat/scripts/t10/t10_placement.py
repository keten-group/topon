"""Recheck of the earlier strand placement timing (Appendix B text) with the latest topon (54f2474, same source as
t10_latest.py).

Same cases as before: graphs from the exact search (target A rescaled to the lattice), then place() draws every strand
at the build state (build density 0.1) with the three routes, three repeats each. One case per process, each pinned to
one performance core (the first three of T10_CORES). The MD relaxation is not timed. Writes T10_OUT/t10_placement.csv.

    python t10_placement.py      (after t10_extend216.py, never at the same time as a timing run)
"""
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import t10_latest as T
from topon.conformation.placement.chains import place  # noqa: E402  (from TOPON_SRC via t10_latest)

CASES = [(6, 30), (10, 30), (14, 20)]      # (lattice edge, strand DP)
ROUTES = ["straight", "meander", "walk"]
BUILD_DENSITY = 0.1


def case(job):
    size, dp = job
    target = T.rescale(T.TARGETS["A"], size ** 3)
    G, gen_s = T.run_py("exact", size, 1, target, 1)
    assert G is not None and T.check_and_hash(G, target)[0]
    rows = []
    for route in ROUTES:
        for rep in range(3):
            t0 = time.perf_counter()
            pl = place(G, dp=dp, placement=route, build_density=BUILD_DENSITY, rng=np.random.default_rng(rep))
            wall = time.perf_counter() - t0
            rows.append(dict(sites=size ** 3, dp=dp, edges=G.number_of_edges(), route=route, rep=rep, graph_s=gen_s,
                             place_s=wall, beads=sum(len(s.path) for s in pl.strands), core=T._core))
            print(rows[-1], flush=True)
    return rows


def main():
    q = mp.Manager().Queue()
    for c in T.CORES[:len(CASES)]:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(CASES), initializer=T.pin, initargs=(q,)) as ex:
        rows = [r for rs in ex.map(case, CASES) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_placement.csv"), index=False)
    print(df.groupby(["sites", "dp", "route"]).agg(edges=("edges", "first"), beads=("beads", "first"),
                                                   place_s=("place_s", "median"), max_s=("place_s", "max")).to_string())


if __name__ == "__main__":
    main()
