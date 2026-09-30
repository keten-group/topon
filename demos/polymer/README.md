# Polymer-network demos

Each demo is one `config.json`. Run it from the repository root with

```bash
topon generate demos/polymer/<atomistic|coarse_grained>/<demo>/config.json --output ./runs
```

Every config in the two tables below generates a 5x5x5 simple cubic network with the Python generator (125 sites,
about a fifth of them dangling ends, junction functionality up to 4), then builds the chemistry, places the strands and
writes the LAMMPS files. The two resolutions share the same features, so pick whichever your LAMMPS workflow expects.

## Atomistic (DREIDING, PDMS)

| Demo | What it shows |
|---|---|
| [`atomistic/basic/`](atomistic/basic/) | A plain network with one node type and one edge type. |
| [`atomistic/entanglement/`](atomistic/entanglement/) | Five entanglements between neighbouring strands. |
| [`atomistic/copolymer/`](atomistic/copolymer/) | Block copolymer strands (PDMS and phenyl siloxane). |
| [`atomistic/graft/`](atomistic/graft/) | PDMS side chains grafted onto the backbone. |
| [`atomistic/defect/`](atomistic/defect/) | Five secondary loops (parallel strands between two junctions). |
| [`atomistic/combined/`](atomistic/combined/) | Entanglements and grafts together. |

## Coarse-grained (Kremer-Grest)

The same six demos with `chemistry.model_type = "coarse_grained"` and the Kremer-Grest simulation options
(`include_angles`, `pair_style`). In the coarse-grained model the monomer names only label bead types. The copolymer,
graft and combined configs also declare those labels in `chemistry.monomers`, which the coarse-grained model does not
need (the SMILES there are not used).

| Demo | What it shows |
|---|---|
| [`coarse_grained/basic/`](coarse_grained/basic/) | A bead-spring network with attractive LJ (`pair_style: attractive`). |
| [`coarse_grained/entanglement/`](coarse_grained/entanglement/) | Five entanglements. |
| [`coarse_grained/copolymer/`](coarse_grained/copolymer/) | Block copolymer strands (bead types A and B). |
| [`coarse_grained/graft/`](coarse_grained/graft/) | Grafted side chains (bead type B). |
| [`coarse_grained/defect/`](coarse_grained/defect/) | Five secondary loops. |
| [`coarse_grained/combined/`](coarse_grained/combined/) | Entanglements and grafts together. |

## Atomistic with CHARMM

[`atomistic/charmm_peg/`](atomistic/charmm_peg/) builds a tetra-PEG network on a 2x2x2 diamond lattice and writes it
with CHARMM parameters (`chemistry.force_field = "charmm"`, the bundled C35r ether force field and a stream file for
the junction) instead of DREIDING. Its README explains the residue matching and the three LAMMPS stages.

## Expected output

Each atomistic demo (the six DREIDING ones and the CHARMM one) has an `expected_output/` folder with the small text
files of one build relaxed through its three LAMMPS stages. These are the coefficient includes, the groups, the stage
scripts, the LAMMPS logs, the run manifest and the `topon track` page, with a README that gives the numbers of every
stage and the commands that made them. The data files are not included. For these runs the DREIDING demos load the
showcase network instead of generating one, so they share one graph.

## Where each feature lives in the config

| Feature | Config section | Notes |
|---|---|---|
| Entanglements | `assignment.entanglements` | a count or a distribution of entangled strand pairs |
| Copolymers | `assignment.copolymer.per_edge_type` | block, random, alternating or gradient sequences |
| Grafts | `assignment.grafts.per_edge_type` | graft density, side-chain DP and monomer per edge type |
| Defects | `assignment.defects.secondary_loops` | parallel strands with a valence cap |

The two defect demos give the degree counts in full (`"0:0,1:24,2:30,3:40,4:31"`) and use the exact search
(`"search": "exact"`), which places the secondary loops as forced double edges while it builds the network. The full
schema is in Appendix A of [`docs/USAGE.md`](../../docs/USAGE.md).

## Using another topology

To build on an existing network instead, set `topology.source` to `"load"` and point `topology.existing_files` at a
`.nodes` and `.edges` pair, for example the network written by
[`../topology/end_linking/python/run.py`](../topology/end_linking/python/run.py), the showcase network in
[`../showcase/`](../showcase/), or one of the paper's networks in `demos/npjcompmat/data/mechanics/`.
