"""Recheck of the earlier bias check of the fallback with the latest topon (54f2474): how much more uniform are the
networks of the balanced deal (fallback) than those of the regular random deal, for the typical target T on
nearest-neighbor SC lattices of 216 and 1,728 sites (the Appendix B sentence "algebraic connectivity 6 to 13% higher").

degree_matching.sculpt_exact(base, need, rng(seed), max_f=6, min_giant_fraction=1.0, fallback=False/True), one attempt
per seed, seeds 1-125 (the first 100 accepted networks are compared). lambda_2 of the active subgraph. Not a timing run: 8 workers on efficiency cores 24-31 (T10_CHECK_CORES to change).

Writes T10_OUT/t10_fallback_bias.csv and prints the mean lambda_2 per mode and the relative difference.

    python t10_fallback_bias.py
"""
import hashlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import t10_latest as T
from topon.topology import degree_matching as dm  # noqa: E402  (TOPON_SRC via t10_latest)

CORES = T.cpu_list("T10_CHECK_CORES", [24, 25, 26, 27, 28, 29, 30, 31])
CASES = [(6, "T"), (12, "T")]
_BASE = {}


def scaffold(size):
    if size not in _BASE:
        cfg = T.py_config("exact", size, 1, T.rescale(T.TARGETS["T"], size ** 3), 1)
        _BASE[size] = T.PythonTopologyGenerator(cfg)._create_lattice((size,) * 3, "SC")
    return _BASE[size]


def one(args):
    size, tname, mode, seed = args
    base = scaffold(size)
    need = {d: int(c) for d, c in enumerate(T.rescale(T.TARGETS[tname], size ** 3))}
    edges, _, rec = dm.sculpt_exact(base, need, np.random.default_rng(seed), max_f=6, min_giant_fraction=1.0,
                                    fallback=(mode == "fallback"))
    row = dict(size=size, target=tname, mode=mode, seed=seed, success=edges is not None)
    if edges is not None:
        G = nx.Graph()
        G.add_nodes_from(base.nodes())
        G.add_edges_from(edges)
        assert T.check_and_hash(G, [need[d] for d in range(7)])[0]
        A = G.subgraph([n for n, d in G.degree() if d > 0])
        adj = nx.to_numpy_array(A)
        row["lambda_2"] = float(np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj)[1])
        row["hash"] = hashlib.md5(repr(sorted(tuple(sorted(e)) for e in edges)).encode()).hexdigest()
    return row


def main():
    q = mp.Manager().Queue()
    for c in CORES:
        q.put(c)
    jobs = [(s, t, m, seed) for s, t in CASES for m in ("random", "fallback") for seed in range(1, 126)]
    with ProcessPoolExecutor(max_workers=len(CORES), initializer=T.pin, initargs=(q,)) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_fallback_bias.csv"), index=False)
    for (s, t), g in df.groupby(["size", "target"]):
        acc = {m: g[(g["mode"] == m) & g.success].sort_values("seed").head(100) for m in ("random", "fallback")}
        a, b = acc["random"].lambda_2, acc["fallback"].lambda_2
        same = len(set(acc["random"].hash) & set(acc["fallback"].hash))
        print(f"{s ** 3} sites {t}: random {len(a)} (accept {g[g['mode'] == 'random'].success.mean():.2f}) "
              f"lambda_2 {a.mean():.4f} +- {a.std():.4f}; fallback {len(b)} lambda_2 {b.mean():.4f} +- {b.std():.4f}; "
              f"fallback/random {b.mean() / a.mean() - 1:+.1%}; KS p {ks_2samp(a, b).pvalue:.2g}; identical graphs {same}")


if __name__ == "__main__":
    main()
