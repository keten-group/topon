"""Do the two searches of the latest code draw the same graphs for one P(f)? (recheck of the sampler comparison of
Table G1, data/derived/generator_benchmark/tables.md, with topon 54f2474)

For every target of the coarse-grained ensemble (327, SC 6^3, one shell), one network from C exact matching and one
from C pruning (seed 1, 120 s cap, same binary and harness as t10_latest.py), compared pairwise with the deposited
network of that target. Metrics on the active subgraph with the manuscript definitions (graph_metrics below).
Differences are given in units of the ensemble standard deviation (deposited 327) with a Wilcoxon signed-rank p.
Not a timing run: 4 workers on efficiency cores (logical CPUs 28-31, T10_CHECK_CORES to change).

Writes T10_OUT/t10_bias.csv and T10_OUT/t10_bias_summary.json. Reads the same metrics of the deposited networks
(key, lambda_2, num_bridges, max_betweenness, cycle_rank) from t0_deposited_graphs.csv in
data/derived/generator_benchmark/, an earlier audit of data/mechanics/.

    python t10_bias.py
"""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import t10_327
import t10_latest as T

METRICS = ["lambda_2", "num_bridges", "max_betweenness", "cycle_rank"]
E_CORES = T.cpu_list("T10_CHECK_CORES", [28, 29, 30, 31])


def graph_metrics(G):
    A = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    n, e = A.number_of_nodes(), A.number_of_edges()
    comps = nx.number_connected_components(A)
    adj = nx.to_numpy_array(A, nodelist=list(A.nodes()))
    ev = np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj)
    return dict(lambda_2=float(ev[1]), cycle_rank=e - n + comps, num_bridges=sum(1 for _ in nx.bridges(A)),
                max_betweenness=float(max(nx.betweenness_centrality(A).values())))


def one(job):
    method, name, counts = job
    G, wall = T.run_c("strict" if method == "C_pruning" else "exact", 6, 1, np.asarray(counts), 1)
    ok = G is not None and T.check_and_hash(G, counts)[0]
    row = dict(method=method, key=name, success=bool(ok), wall_s=wall)
    if ok:
        row.update(graph_metrics(G))
    return row


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    names, counts = t10_327.targets()
    jobs = [(m, n, c) for m in ("C_exact", "C_pruning") for n, c in zip(names, counts)]
    q = mp.Manager().Queue()
    for c in E_CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(E_CORES), initializer=T.pin, initargs=(q,)) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_bias.csv"), index=False)
    D0 = pd.read_csv(T.DEPOSITED).set_index("key")
    ens = D0[METRICS].std(ddof=1)
    X = {m: x[x.success].set_index("key") for m, x in df.groupby("method")}
    X["deposited"] = D0
    out = {m: int(len(X[m])) for m in ("C_exact", "C_pruning")}
    for a, b in (("C_exact", "deposited"), ("C_pruning", "deposited"), ("C_exact", "C_pruning")):
        k = X[a].index.intersection(X[b].index)
        res = dict(n=int(len(k)))
        for m in METRICS:
            d = X[a].loc[k, m].astype(float) - X[b].loc[k, m].astype(float)
            res[m] = dict(mean_diff=float(d.mean()), over_ensemble_sd=float(d.mean() / ens[m]) if ens[m] else 0.0,
                          wilcoxon_p=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0)
        out[f"{a}_minus_{b}"] = res
    json.dump(out, open(os.path.join(T.OUT, "t10_bias_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
