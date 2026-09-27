# Workflows

Standalone Python scripts that drive topon without a config file, for when a single demo config is not enough (a loop,
a sweep, a CSV of results). Each script has a block of settings at the top. Copy it, change the settings and run it.

| Workflow | What it does | Output |
|---|---|---|
| [`batch_polymer_topology/run.py`](batch_polymer_topology/run.py) | Generates up to 25 seeded 5x5x5 lattice networks (a seed that finds no network within its trial budget is skipped), exports each as `.nodes`/`.edges`, GraphML and NPZ, and writes one CSV of per-graph properties | `output/graph_NNN/{network.nodes, network.edges, network.graphml, network.npz}` and `output/summary.csv` |

The GraphML and NPZ files hold the network as a dual graph (strands as nodes), in a form that graph-learning libraries
read directly.
