"""Second arm of t10_bias.py: Python exact matching (seed 1, 120 s cap, same harness) on the 327 ensemble targets,
compared with the deposited graphs, with C exact and with C pruning from T10_OUT/t10_bias.csv.
Not a timing run: 4 workers on efficiency cores (logical CPUs 28-31, T10_CHECK_CORES to change).

Writes T10_OUT/t10_bias_python_exact.csv and T10_OUT/t10_bias_py_summary.json. Reads t0_deposited_graphs.csv
as t10_bias.py does.

    python t10_bias_py.py      (after t10_bias.py)
"""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import t10_327
import t10_latest as T
from t10_bias import E_CORES, METRICS, graph_metrics


def one(job):
    name, counts = job
    G, wall = T.run_py("exact", 6, 1, np.asarray(counts), 1)
    ok = G is not None and T.check_and_hash(G, counts)[0]
    row = dict(method="python_exact", key=name, success=bool(ok), wall_s=wall)
    if ok:
        row.update(graph_metrics(G))
    return row


def main():
    names, counts = t10_327.targets()
    q = mp.Manager().Queue()
    for c in E_CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(E_CORES), initializer=T.pin, initargs=(q,)) as ex:
        rows = list(ex.map(one, list(zip(names, counts))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_bias_python_exact.csv"), index=False)
    D0 = pd.read_csv(T.DEPOSITED).set_index("key")
    ens = D0[METRICS].std(ddof=1)
    b = pd.read_csv(os.path.join(T.OUT, "t10_bias.csv"))
    X = {m: x[x.success].set_index("key") for m, x in pd.concat([b, df]).groupby("method")}
    X["deposited"] = D0
    out = dict(python_exact=int(len(X["python_exact"])))
    for a, c in (("python_exact", "deposited"), ("python_exact", "C_pruning"), ("python_exact", "C_exact")):
        k = X[a].index.intersection(X[c].index)
        res = dict(n=int(len(k)))
        for m in METRICS:
            d = X[a].loc[k, m].astype(float) - X[c].loc[k, m].astype(float)
            res[m] = dict(mean_diff=float(d.mean()), over_ensemble_sd=float(d.mean() / ens[m]) if ens[m] else 0.0,
                          wilcoxon_p=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0)
        out[f"{a}_minus_{c}"] = res
    json.dump(out, open(os.path.join(T.OUT, "t10_bias_py_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
