<p align="center">
  <img src="assets/topon_header.gif" alt="The word topon spelled by a topon-generated polymer network relaxing under Kremer-Grest MD" width="660">
</p>

# topon

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19672939.svg)](https://doi.org/10.5281/zenodo.19672939)

topon builds polymer networks you can drop straight into LAMMPS.

You describe the network you want (how the junctions connect, how long the strands are, what chemistry sits on them)
and topon turns it into a simulation box. Connectivity comes first and chemistry is mapped onto it afterwards, so the
same graph can come out as a coarse-grained melt or as an all-atom system.

```
Topology → Analysis → Assignment → Chemistry → Conformation → Output
```

## How a network gets built

topon starts from a lattice of junctions and prunes it. A bare lattice gives every junction the same functionality,
which no real network has. The generator removes edges until the degree distribution matches the one you asked for.

<p align="center">
  <img src="assets/anim/sculpt_arc.gif" alt="A cubic lattice being pruned edge by edge, nodes recolouring as they lose connectivity, while a histogram converges on the target degree distribution" width="760">
</p>

<sub>On the left the nodes are coloured by their current degree, deep blue for 6-connected and warmer as they lose
edges. On the right the degree histogram fills in against the dashed target. The animation replays the edge-removal
history the generator recorded on a 6x6x6 lattice.</sub>

There are two searches, and both exist in the Python generator and in the C generator. The pruning search (`strict`)
removes edges one at a time. The exact search (`exact`) solves for a subgraph of the lattice with exactly the requested
number of junctions of each degree, so it needs a count for every degree. The pipeline uses the Python generator,
which needs no compiler, unless `topology.generator.exe_path` points at a compiled C generator
(`topon/topology/csrc/generator.c`), which is also a standalone program for long searches.

It works from any of the lattices, and gets to the same place from each.

<p align="center">
  <img src="assets/anim/lattices_arc.gif" alt="Simple cubic, body-centred cubic and face-centred cubic lattices each pruned down to mean degree 4" width="880">
</p>

<sub>SC, BCC and FCC start 6-, 8- and 12-coordinate and are all pruned down to mean degree 4.</sub>

Diamond is the fourth lattice. It is 4-coordinate by construction, so a tetrafunctional network needs no pruning at
all. A fifth, `MIX`, overlays SC, BCC and FCC sites at fractions you choose, which spreads the junction spacing and
with it the strand end-to-end distances. A neighbour cutoff lets strands reach beyond the nearest lattice neighbours,
and any axis can be made open instead of periodic, giving a slab with a free surface.

## What comes out

The as-built network is deliberately unphysical. Strands are strung taut between junctions, so their bonds start far
too short. MD fixes that. The chains coil out to their natural length and the lattice becomes a melt, carrying the
connectivity you asked for with it. topon writes the LAMMPS data file and the input scripts for this relaxation.

<table>
<tr>
<td width="50%"><img src="assets/anim/cg_arc.gif" alt="Coarse-grained network relaxing from a taut lattice into an equilibrated melt"></td>
<td width="50%"><img src="assets/anim/atom_arc.gif" alt="All-atom DREIDING PDMS network relaxing from lattice to melt"></td>
</tr>
<tr>
<td><b>Coarse-grained.</b> A Kremer-Grest network coiling from the lattice into a melt.</td>
<td><b>All-atom.</b> The same construction in DREIDING PDMS. Gold silicon and red oxygen trace the siloxane
backbone.</td>
</tr>
</table>

## Things you can put on the strands

**Copolymer sequences.** Block, random, alternating or gradient, set per strand at build time and carried into the
melt.

<p align="center">
  <img src="assets/anim/copoly_arc.gif" alt="A block copolymer network melting, red and blue halves preserved" width="440">
  <img src="assets/anim/graft_arc.gif" alt="A densely grafted network relaxing, side chains in teal" width="440">
</p>

<sub>On the left a block copolymer. Every strand is half A (red) and half B (blue), and the pattern survives the melt
because the sequence belongs to the chain, not to the geometry. On the right dense side chains (teal) branch off a
blue backbone.</sub>

**Grafts.** Side chains of a chosen length and monomer, attached at a density you set per edge type.

**Entanglements.** Ask for a number of entanglements and topon draws pairs of neighbouring strands as spirals that
wind around each other a set number of times, so that they stay interlocked when the network relaxes.

<p align="center">
  <img src="assets/anim/ent_arc.gif" alt="A single entanglement shown in the full network and zoomed in a box, both relaxing from lattice to melt" width="760">
</p>

<sub>The two entangled chains are gold and violet, everything else grey. As the network melts they open into
interlocked coils, still hooked.</sub>

**Defects.** Secondary loops (parallel strands between two junctions that are already connected) come with valence
protection, so a junction never ends up over-coordinated. **POSS chain caps** come with the real Si₈O₁₂ cage
chemistry.

## Installation

```bash
git clone https://github.com/keten-group/topon.git
cd topon
pip install -e .
```

LAMMPS is only needed to *run* what topon writes, not to generate it. The C generator is optional. Build it with
`gcc -O2 -o generator.exe generator.c -lm` in `topon/topology/csrc/` and point `topology.generator.exe_path` at the
binary.

## Getting started

```bash
topon init --output my_run.json        # starter config
topon generate my_run.json --output ./runs
```

`topon` on its own opens an interactive session. `topon doctor` checks a config before you run it, `topon inspect`
summarises what came out, and `topon recipes` lists worked examples. Ready-made configs are in [`demos/`](demos/).

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | CLI reference, Python API, recipes and the config schema |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the six stages fit together |
| [demos/README.md](demos/README.md) | The demo configs and scripts |
| [demos/npjcompmat/](demos/npjcompmat/README.md) | Data and notebooks behind the figures of the paper |
| [CHANGELOG.md](CHANGELOG.md) | Changes between releases |

## Citation

If you use topon, please cite it. The citation metadata are in [`CITATION.cff`](CITATION.cff), which GitHub also
offers under "Cite this repository". The DOI above always resolves to the latest release on Zenodo.

## License

MIT. See [LICENSE](LICENSE).
