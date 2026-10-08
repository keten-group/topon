"""Recheck of the degeneracy rows of Table G1 (one P(f), many graphs; data/derived/generator_benchmark/tables.md) with
the latest topon (54f2474, same harness as t10_latest.py).

The 10 targets of that check (5th to 95th percentiles of the deposited lambda_2, TARGETS below). For each, 20 networks
from C exact matching and 20 from C pruning (seeds 1-20, 60 s cap per attempt). Within-target standard deviation of
each metric over the ensemble standard deviation of the 327 deposited networks, averaged over targets with at least 5
networks; every network checked (exact counts, connected) and hashed. Not a timing run: 4 workers on efficiency cores
24-27 (T10_CHECK_CORES to change).

Writes T10_OUT/t10_degeneracy.csv and T10_OUT/t10_degeneracy_summary.json. Reads t0_deposited_graphs.csv as
t10_bias.py does.

    python t10_degeneracy.py
"""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import t10_latest as T
from t10_bias import METRICS, graph_metrics

CORES = T.cpu_list("T10_CHECK_CORES", [24, 25, 26, 27])
SEEDS = range(1, 21)
TARGETS = ["11_20_28_65_21_25_46", "14_33_21_34_50_23_41", "14_19_23_45_46_34_35", "11_30_32_26_49_28_40",
           "14_43_15_28_34_33_49", "14_33_11_35_55_22_46", "15_34_11_33_51_49_23", "14_31_34_25_41_14_57",
           "10_15_33_29_78_22_29", "10_0_26_75_43_53_9"]


def one(job):
    method, key, seed = job
    counts = [int(x) for x in key.split("_")]
    T.CAP = 60.0
    G, wall = T.run_c("strict" if method == "C_pruning" else "exact", 6, 1, np.asarray(counts), seed)
    ok, h = T.check_and_hash(G, counts) if G is not None else (False, "")
    row = dict(method=method, key=key, seed=seed, success=bool(ok), wall_s=wall, hash=h)
    if ok:
        row.update(graph_metrics(G))
    return row


def main():
    keys = TARGETS
    jobs = [(m, k, s) for s in SEEDS for k in keys for m in ("C_exact", "C_pruning")]
    q = mp.Manager().Queue()
    for c in CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(CORES), initializer=T.pin, initargs=(q,)) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_degeneracy.csv"), index=False)
    ens = pd.read_csv(T.DEPOSITED)[METRICS].std(ddof=1)
    out = {}
    for m, x in df[df.success].groupby("method"):
        n = x.groupby("key").size()
        keep = n[n >= 5].index
        sd = x[x.key.isin(keep)].groupby("key")[METRICS].std(ddof=1)
        out[m] = dict(networks_per_target=n.to_dict(), all_distinct=bool(x.groupby("key").hash.nunique().eq(n).all()),
                      within_over_ensemble_sd={k: float(sd[k].mean() / ens[k]) for k in METRICS})
    json.dump(out, open(os.path.join(T.OUT, "t10_degeneracy_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
