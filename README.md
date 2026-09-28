<p align="center">
  <img src="assets/topon_header.gif" alt="The word topon spelled by a topon-generated polymer network relaxing under Kremer-Grest MD" width="660">
</p>

# topon

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19672939.svg)](https://doi.org/10.5281/zenodo.19672939)

topon generates polymer networks with a prescribed topology and writes them as LAMMPS input files.

The network is built as a graph first (i.e., which junctions connect, how long the strands are and what sits on them),
and the chemistry is mapped onto it afterwards. The same graph can therefore be written as a coarse-grained or an
all-atom system.

```
Topology → Analysis → Assignment → Chemistry → Conformation → Output
```

## Network generation

topon starts from a lattice of junctions and removes edges until the degree distribution matches the requested one.

<p align="center">
  <img src="assets/anim/sculpt_arc.gif" alt="A cubic lattice being pruned edge by edge while a histogram converges on the target degree distribution" width="760">
</p>

<sub>Nodes are colored by their current degree, and the histogram approaches the requested distribution (dashed). The
animation replays the edge removals recorded by the generator on a 6x6x6 lattice.</sub>

There are two searches, both implemented in Python and in C. Pruning (`strict`) removes edges one at a time. Exact
matching (`exact`) finds a subgraph of the lattice with exactly the requested number of sites of each degree, so it
needs a count for every degree. The pipeline uses the Python generator unless `topology.generator.exe_path` points to
the compiled C generator (`topon/topology/csrc/generator.c`), which also runs on its own.

<p align="center">
  <img src="assets/anim/lattices_arc.gif" alt="Simple cubic, body-centred cubic and face-centred cubic lattices each pruned down to mean degree 4" width="880">
</p>

<sub>SC, BCC and FCC lattices (6, 8 and 12 neighbors) pruned to a mean degree of 4.</sub>

Diamond (4 neighbors) and `MIX` (SC, BCC and FCC sites overlaid at chosen fractions) are also available. A neighbor
cutoff lets strands connect sites beyond the nearest neighbors, and each axis can be periodic or open (e.g., a slab
with a free surface).

## Output

In the generated network the strands run straight between their junctions. topon therefore also writes the LAMMPS
input scripts that relax it into an equilibrated melt with the same connectivity.

<table>
<tr>
<td width="50%"><img src="assets/anim/cg_arc.gif" alt="Coarse-grained network relaxing from the lattice into an equilibrated melt"></td>
<td width="50%"><img src="assets/anim/atom_arc.gif" alt="All-atom DREIDING PDMS network relaxing from the lattice into a melt"></td>
</tr>
<tr>
<td>Coarse-grained Kremer-Grest network relaxing from the lattice.</td>
<td>All-atom PDMS network with DREIDING (silicon in gold, oxygen in red).</td>
</tr>
</table>

## Strand features

- Copolymer sequences (block, random, alternating or gradient), set per strand.
- Side-chain grafts of a chosen length and monomer, with a density set per edge type.
- Entanglements, drawn as pairs of neighboring strands wound around each other a set number of times, so that they
  stay interlocked during relaxation.
- Defects such as secondary loops (two strands between the same pair of junctions), placed without exceeding the
  maximum functionality of a junction.
- POSS chain caps with the Si₈O₁₂ cage.

<p align="center">
  <img src="assets/anim/copoly_arc.gif" alt="A block copolymer network relaxing, red and blue halves preserved" width="440">
  <img src="assets/anim/graft_arc.gif" alt="A densely grafted network relaxing, side chains in teal" width="440">
</p>

<sub>Left, a block copolymer network (A in red, B in blue). Right, side chains (teal) on a blue backbone.</sub>

<p align="center">
  <img src="assets/anim/ent_arc.gif" alt="A single entanglement shown in the full network and in a close-up, relaxing from lattice to melt" width="760">
</p>

<sub>Two entangled strands (gold and violet) in the network and in a close-up, relaxing from the lattice.</sub>

## Installation

```bash
git clone https://github.com/keten-group/topon.git
cd topon
pip install -e .
```

LAMMPS is needed only to run the generated inputs. The C generator is optional. Build it with
`gcc -O2 -o generator.exe generator.c -lm` in `topon/topology/csrc/` and set `topology.generator.exe_path` to the
binary.

## Getting started

```bash
topon init --output my_run.json        # starter config
topon generate my_run.json --output ./runs
```

Running `topon` without arguments opens an interactive session. `topon doctor` checks a config before a run,
`topon inspect` summarizes a finished run and `topon recipes` lists worked examples. Ready-made configs are in
[`demos/`](demos/).

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Command line, Python API, recipes and config reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The six stages and how they fit together |
| [demos/README.md](demos/README.md) | Demo configs and scripts |
| [demos/npjcompmat/](demos/npjcompmat/README.md) | Data and notebooks behind the figures of the paper |
| [CHANGELOG.md](CHANGELOG.md) | Changes between releases |

## Citation

If you use topon, please cite it with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite
this repository"). The DOI above resolves to the latest release on Zenodo.

## License

MIT. See [LICENSE](LICENSE).
