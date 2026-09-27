# Showcase network

A small network in the format the generators write and `topology.source = "load"` reads. Use it to see what a
generated topology looks like, or as a ready topology for any demo.

## `network_5x5x5/`

A 5x5x5 simple cubic network found by the topology generator on trial 3 (125 sites, of which 4 are vacancies, and 210
strands).

| File | Content |
|---|---|
| `network.nodes` | one line per site, with the site ID, its x, y, z coordinates and its degree |
| `network.edges` | one line per strand, with the IDs of the two sites it joins |
| `generation.log` | the successful edge removals of that trial |

To build on it, use this topology section (paths relative to the repository root):

```json
"topology": {
    "source": "load",
    "existing_files": {
        "nodes_file": "demos/showcase/network_5x5x5/network.nodes",
        "edges_file": "demos/showcase/network_5x5x5/network.edges"
    }
}
```

`demos/templates/minimal.json` does this. The networks of the paper (6x6x6) are in
[`demos/npjcompmat/data/mechanics/`](../npjcompmat/data/mechanics/).
