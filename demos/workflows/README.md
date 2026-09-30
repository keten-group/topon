# Workflows

Standalone Python scripts that drive topon without a config file, for when a single demo config is not enough (a loop,
a sweep, a CSV of results). Each script has a block of settings at the top. Copy it, change the settings and run it.

| Workflow | What it does | Output |
|---|---|---|
| [`batch_polymer_topology/run.py`](batch_polymer_topology/run.py) | Generates up to 25 5x5x5 lattice networks (graph i with `topology.generator.seed` set to `SEED_BASE + i`), exports each as `.nodes`/`.edges`, GraphML and NPZ, and writes one CSV of per-graph properties. A second run skips every graph that already has a `record.json`, a failed one included, unless it is given `--retry-failed`, and only one run at a time may write the folder | `output/graph_NNN/{record.json, network.nodes, network.edges, network.graphml, network.npz}` and `output/summary.csv` |

The GraphML and NPZ files hold the network as a dual graph (strands as nodes), in a form that graph-learning libraries
read directly.
