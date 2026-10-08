"""Table 2 (Appendix C) with the latest topon: smallest-ring statistics versus neighbor range, at the reference P(f).

The original table came from a prototype exact solver of the bond/create validation (its cubic sweep,
degree_matching.sculpt_exact, three graphs per row). Here the graphs come from the topon package itself
(54f2474, same source as t10_latest.py): PythonTopologyGenerator, exact search, SC lattice with neighbour_shells k,
max functionality 4, the package's default connectivity acceptance, at the effective degree counts of each reference
on the matched cubic cell (N20: 14^3, 0:43,1:217,2:356,3:153,4:1975; N100: 9^3, 0:157,1:73,2:33,3:59,4:407), as in
the original sweep. Ten seeds per row.

Ring statistics exactly as in that sweep: simple graph, giant component, degree-1 nodes
removed repeatedly (elastic core), shortest cycle through each core edge (bridges excluded), mean of the finite
values, and the Jensen-Shannon divergence (log2, ring sizes 3-15) to the reference distribution (the "cycle"
distribution of the reference in data/derived/bond_create/four_distributions.json, the same counts the original
sweep used). Not a timing run: workers on efficiency cores 24-31 (T10_CHECK_CORES to change).

Writes T10_OUT/t10_shells.csv and T10_OUT/t10_shells_summary.csv, and prints the table rows.

    python t10_shells.py
"""
import collections
import contextlib
import io
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

import t10_latest as T
from topon.config.schema import GeneratorConfig  # noqa: E402  (TOPON_SRC via t10_latest)
from topon.topology.generator_python import PythonTopologyGenerator  # noqa: E402

REF = os.path.join(T.COMPANION, "data", "derived", "bond_create", "four_distributions.json")
SPECS = {"N20": (14, "0:43,1:217,2:356,3:153,4:1975"), "N100": (9, "0:157,1:73,2:33,3:59,4:407")}
SHELLS = [1, 2, 3, 4, 5, 6, 8]
SEEDS = list(range(1, 11))
CORES = T.cpu_list("T10_CHECK_CORES", [24, 25, 26, 27, 28, 29, 30, 31])


def reference(spec):
    """Smallest-ring distribution {ring size: count} of the reaction-cured reference network."""
    return {int(a): b for a, b in json.load(open(REF))[spec]["reference"]["cycle"].items()}


def core_ring_sizes(G):
    S = nx.Graph()
    S.add_edges_from((u, v) for u, v in G.edges() if u != v)
    C = S.subgraph(max(nx.connected_components(S), key=len)).copy()
    while True:
        leaves = [n for n, d in C.degree() if d <= 1]
        if not leaves:
            break
        C.remove_nodes_from(leaves)
    out = []
    for u, v in list(C.edges()):
        C.remove_edge(u, v)
        try:
            out.append(nx.shortest_path_length(C, u, v) + 1)
        except nx.NetworkXNoPath:
            out.append(np.inf)
        C.add_edge(u, v)
    return np.array(out, float)


def js(p, q, kmin=3, kmax=16):
    P = np.array([p.get(k, 0) for k in range(kmin, kmax)], float)
    Q = np.array([q.get(k, 0) for k in range(kmin, kmax)], float)
    P /= max(P.sum(), 1)
    Q /= max(Q.sum(), 1)
    M = 0.5 * (P + Q)
    kl = lambda a, b: float((a[a > 0] * np.log2(a[a > 0] / b[a > 0])).sum())  # noqa: E731
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def one(job):
    spec, shells, seed = job
    n, dd = SPECS[spec]
    kw = dict(lattice_size=f"{n}x{n}x{n}", lattice_type="SC", periodicity="111", max_functionality=4,
              degree_distribution=dd, search="exact", seed=seed)
    if shells > 1:
        kw["neighbour_shells"] = shells
    G = None
    for k in range(20):            # new seeds if a call gives up, as in t10_latest.run_py
        kw["seed"] = seed * 100003 + k
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                gs = PythonTopologyGenerator(GeneratorConfig(**kw)).generate(trials=10**9, max_saves=1, time_limit=300)
            except Exception:  # noqa: BLE001
                gs = []
        if gs:
            G = nx.Graph(gs[0])
            break
    if G is None:
        return dict(spec=spec, shells=shells, seed=seed, success=False)
    want = {int(a): int(b) for a, b in (x.split(":") for x in dd.split(","))}
    got = collections.Counter(d for _, d in G.degree())
    assert all(got.get(d, 0) == c for d, c in want.items() if d > 0), (got, want)
    sc = core_ring_sizes(G)
    fin = sc[np.isfinite(sc)]
    dist = dict(collections.Counter(fin.astype(int)))
    ref = reference(spec)
    return dict(spec=spec, shells=shells, seed=seed, success=True, calls=k + 1, mean_ring=float(fin.mean()),
                js=js(dist, ref), frac_odd=float(sum(v for r, v in dist.items() if r % 2) / len(fin)),
                core_edges=len(sc), dist=json.dumps({int(a): int(b) for a, b in sorted(dist.items())}))


def main():
    jobs = [(s, k, seed) for s in SPECS for k in SHELLS for seed in SEEDS]
    q = mp.Manager().Queue()
    for c in CORES:
        q.put(c)
    with ProcessPoolExecutor(max_workers=len(CORES), initializer=T.pin, initargs=(q,)) as ex:
        rows = list(ex.map(one, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(T.OUT, "t10_shells.csv"), index=False)
    ok = df[df.success]
    g = ok.groupby(["spec", "shells"]).agg(n=("seed", "count"), mean_ring=("mean_ring", "mean"),
                                           mean_ring_sd=("mean_ring", "std"), js=("js", "mean"), js_sd=("js", "std"),
                                           frac_odd=("frac_odd", "mean")).reset_index()
    g.to_csv(os.path.join(T.OUT, "t10_shells_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for spec in SPECS:
        ref = reference(spec)
        print(spec, "reference mean ring", round(sum(k * v for k, v in ref.items()) / sum(ref.values()), 3))


if __name__ == "__main__":
    main()
