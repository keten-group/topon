"""Generate a batch of 25 polymer-network topologies and export each
in three formats (.nodes/.edges, GraphML, NPZ), plus a single CSV of
per-graph properties.

This is the kind of dataset you'd feed into a graph-neural-network
training pipeline or a structure-property survey — "give me N graphs
with the same lattice + degree distribution, with their summary stats
in one place."

Graph i is generated with ``topology.generator.seed = SEED_BASE + i``, so
the 25 realisations are reproducible (the same graphs that seeding the
global streams with SEED_BASE + i used to give). Tune the dataset by
editing the knobs at the top — everything below them is mechanical.

Outputs (under output/ next to this script):
    summary.csv                  one row per graph, columns below
    graph_000/
        record.json              the outcome: success or failure, seed, pid
        network.nodes            x/y/z coordinates, one line per node
        network.edges            u v pairs, one line per edge
        network.graphml          dual-graph form for GNN libraries
        network.npz              dual-graph dense arrays for PyTorch
    graph_001/ ...

CSV columns: seed, n_nodes, n_edges, avg_degree, min_degree, max_degree,
n_chain_ends (degree-1 nodes), n_interior (degree>=2 nodes).

Resuming. Every graph leaves a ``record.json``, a failed one included, so
a second run skips every graph that already has a record, whatever its
outcome, and only builds the ones still missing. A failure is recorded
because re-running it gives the same failure (the seed is fixed); pass
``--retry-failed`` to try the failures again, e.g. after raising the
trial budget. ``summary.csv`` is rebuilt from all the records each time.

One writer per output folder. A run claims the folder in
``output/.writer.json`` (pid, host, start time) and refuses to start
while another live process holds it, so two copies started by mistake do
not build the same graphs twice. A claim left by a run that has exited is
taken over.

Usage:
    python demos/workflows/batch_polymer_topology/run.py [--retry-failed]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np

from topon.config.schema import GeneratorConfig
from topon.topology.generator_python import PythonTopologyGenerator
from topon.utils.processes import other_live_writer, stop_hint, this_process
from topon.writers.graphml_writer import write_graphml
from topon.writers.npz_writer import write_npz


# ----- knobs you'd typically change ----------------------------------------
N_GRAPHS = 25
OUTPUT_ROOT = Path(__file__).parent / "output"
LATTICE_SIZE = "5x5x5"
LATTICE_TYPE = "SC"
PERIODICITY = "111"
MAX_FUNCTIONALITY = 4
# "0:13,1:25" = 13 degree-0 sites (vacancies) and 25 degree-1 sites (chain caps),
# the rest spread over degree-2..max_functionality. See
# topon.topology.generator_python::PythonTopologyGenerator for the format.
DEGREE_DISTRIBUTION = "0:13,1:25"
SEED_BASE = 42
DP_DEFAULT = 50         # nominal degree of polymerization, written into the NPZ/GraphML
MAX_TRIALS = 1_000_000  # generator retries per graph
TRIALS_PER_GRAPH = 20   # sculpting trials before a graph is recorded as failed

RECORD = "record.json"
CLAIM = ".writer.json"

# ----- helpers --------------------------------------------------------------

def write_nodes_edges(G: nx.Graph, nodes_path: Path, edges_path: Path) -> None:
    """Persist the raw lattice form so other topon configs can `load` this graph."""
    with nodes_path.open("w") as f:
        for n, attrs in G.nodes(data=True):
            f.write(f"{n} {attrs.get('x', 0)} {attrs.get('y', 0)} {attrs.get('z', 0)}\n")
    with edges_path.open("w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")


def stats(G: nx.MultiGraph) -> dict:
    degs = [d for _, d in G.degree()]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": round(float(np.mean(degs)), 4),
        "min_degree": int(min(degs)),
        "max_degree": int(max(degs)),
        "n_chain_ends": sum(1 for d in degs if d == 1),
        "n_interior": sum(1 for d in degs if d >= 2),
    }


def read_record(sub: Path) -> dict | None:
    """The graph's recorded outcome, or None if it has not been built."""
    try:
        return json.loads((sub / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def should_build(record: dict | None, retry_failed: bool) -> bool:
    """The skip rule: build what has no record, and failures only on request."""
    if record is None:
        return True
    return retry_failed and not record.get("success", False)


def claim(root: Path) -> None:
    """Take the folder for this run, or stop if a live run holds it."""
    holder = None
    try:
        holder = json.loads((root / CLAIM).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    other = other_live_writer(holder)
    if other:
        raise SystemExit(
            f"{root} is being written by process {other['pid']} (started "
            f"{other.get('started', '?')}). One writer per output folder: wait "
            f"for it, or stop it ({stop_hint(other['pid'])}).")
    (root / CLAIM).write_text(json.dumps(this_process()), encoding="utf-8")


def release(root: Path) -> None:
    try:
        (root / CLAIM).unlink()
    except OSError:
        pass


def build_one(i: int, sub: Path) -> dict:
    """Generate graph ``i``, write its files, and return its record."""
    seed = SEED_BASE + i
    cfg = GeneratorConfig(
        lattice_size=LATTICE_SIZE,
        lattice_type=LATTICE_TYPE,
        periodicity=PERIODICITY,
        max_functionality=MAX_FUNCTIONALITY,
        degree_distribution=DEGREE_DISTRIBUTION,
        max_trials=MAX_TRIALS,
        max_saves=1,
        seed=seed,
    )
    record = {"graph_id": sub.name, "seed": seed, **this_process()}
    try:
        graphs = PythonTopologyGenerator(cfg).generate(trials=TRIALS_PER_GRAPH)
    except ValueError as exc:            # a request the generator refuses
        return {**record, "success": False, "error": str(exc)}
    if not graphs:
        return {**record, "success": False,
                "error": f"no graph in {TRIALS_PER_GRAPH} trials"}
    G = graphs[0]
    if not isinstance(G, nx.MultiGraph):
        G = nx.MultiGraph(G)
    write_nodes_edges(G, sub / "network.nodes", sub / "network.edges")
    write_graphml(G, str(sub / "network.graphml"), dp=DP_DEFAULT)
    write_npz(G, str(sub / "network.npz"), dp=DP_DEFAULT)
    return {**record, "success": True, "stats": stats(G)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--retry-failed", action="store_true",
                        help="build again the graphs whose record says they failed")
    args = parser.parse_args(argv)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    claim(OUTPUT_ROOT)
    try:
        print(f"--- batch_polymer_topology: {N_GRAPHS} graphs, "
              f"{LATTICE_TYPE} {LATTICE_SIZE} ---")
        skipped = 0
        rows: list[dict] = []
        for i in range(N_GRAPHS):
            sub = OUTPUT_ROOT / f"graph_{i:03d}"
            record = read_record(sub)
            if should_build(record, args.retry_failed):
                sub.mkdir(exist_ok=True)
                record = build_one(i, sub)
                (sub / RECORD).write_text(json.dumps(record, indent=1),
                                          encoding="utf-8")
                if record["success"]:
                    s = record["stats"]
                    print(f"  {sub.name}: {s['n_nodes']} nodes, {s['n_edges']} "
                          f"edges, avg_deg={s['avg_degree']}")
                else:
                    print(f"  {sub.name}: failed ({record['error']})")
            else:
                skipped += 1
            if record.get("success"):
                rows.append({"graph_id": sub.name, "seed": record["seed"],
                             **record["stats"]})
        if skipped:
            print(f"  skipped {skipped} graphs with a record already "
                  f"(--retry-failed builds the failed ones again)")
    finally:
        release(OUTPUT_ROOT)

    if not rows:
        print("No graphs were produced. Check generator config.")
        return

    csv_path = OUTPUT_ROOT / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} graphs and {csv_path}")


if __name__ == "__main__":
    main()
