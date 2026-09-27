# Changelog

## 0.2.0

### Topology

- Two searches. `topology.generator.search = "strict"` prunes the lattice edge by edge, as in 0.1.0. `"exact"` solves
  for a subgraph with exactly the requested number of sites of each degree, and needs a count for every degree from 0
  to `max_functionality`. Left unset, topon uses `exact` when the degree distribution gives every degree and `strict`
  otherwise.
- The exact search accepts a network whose largest component holds at least `min_giant_fraction` of the active sites
  (default 0.99). Set it to 1 to require a fully connected network.
- Both searches run in the Python generator and in the C generator. The C source ships in `topon/topology/csrc/`.
  Build it with gcc and set `topology.generator.exe_path` to use it in the pipeline, or run it on its own
  (`--search=strict|exact`).
- The C generator shuffles whole edges (it used to mix endpoints between edges) and draws every random number from
  xoshiro256** instead of `rand()`. It prints its seed, and `TOPON_SEED` sets it, so a run can be replayed.
- SC, BCC, FCC, Diamond and `MIX` (SC, BCC and FCC sites overlaid at chosen fractions) in both generators. The Python
  generator of 0.1.0 had SC only.
- `neighbour_cutoff` (or `neighbour_shells`) lets strands join sites beyond the nearest lattice neighbours.
- `periodicity` is set per axis (e.g., `"110"` for a slab with a free surface). Open axes are not wrapped, so
  molecules at a free surface stay whole.
- The generators write the periodic cell into the `.nodes` file (a `# BOX` line) and the loader reads it back.
- A degree distribution that the lattice cannot reach raises an error before the search starts.

### Defects

- Defects are a stage of their own, run after the topology. `assignment.defects` takes `primary_loops` (self-loops),
  `secondary_loops` (two strands between the same pair of junctions), `triangles`, `four_cycles` and `sol_chains`.
  With the exact search the secondary loops are placed as forced double edges, so the degree counts stay exact.
- In 0.1.0 `primary_loops` asked for parallel strands. That request is now `secondary_loops`. The old form
  (`primary_loops` with `target`) still works in this release and prints a `FutureWarning`.

### Entanglements

- An entangled pair is now drawn by default as two strands that wind around each other a set number of times
  (`method: "waypoint"`). The winding count was checked with primitive-path analysis (Z1+). The Gaussian kink of 0.1.0
  is still available as `method: "kink"`.
- Entangled pairs can be weighted by neighbour shell and placed with a spatial bias.

### Conformation and relaxation

- `simulation.protocol` selects the coarse-grained relaxation. The new default, `pushoff`, writes five LAMMPS scripts
  that never let one chain pass through another, so a prescribed entanglement survives the relaxation.
  `hardcore_min` and `soft_push` write the older three-script decks. `simulation.rho_final` sets the density the
  compression stage goes to.
- `topon.conformation.place` (Python API) draws a coarse-grained build straight from the graph with a chosen strand
  shape (`straight`, `meander` or `walk`) and checks every strand's bond lengths and self-overlap before writing it.
  `topon generate` does not call it.

### Chemistry

- Atomistic chains of O-terminated monomers (e.g., PDMS) no longer get a peroxide O-O bond at the chain tail.
- NumPy 2 is supported.

### Command line

- New commands `doctor` (checks a config for known problems), `inspect` (summarises a finished run), `recipes` and
  `shell`. Running `topon` on its own opens the interactive shell.
- `init` copies a ready demo config (`--preset atomistic_pdms`, `cg_kg` or `poss`) or asks for the main settings
  (`--interactive`). The `--full` flag of 0.1.0 is gone.
- `generate --export-graphml` and `--export-npz` also write the network as a graph for graph-learning tools.
- Every run writes `manifest.json` with the search used, the requested and achieved degree counts, the seed and the
  defects placed.
- Unknown keys under `topology.generator` (and in a few other sections) are an error, and `topon doctor` warns about
  ignored keys in the rest.
- The `gui` placeholder is removed.

### Packaging

- A regular `pip install .` now includes the DREIDING parameter files and the C source.

### Demos and documentation

- `demos/` holds runnable configs for atomistic and coarse-grained networks (basic, entanglement, copolymer, graft,
  defect and combined), a POSS demo, templates, and scripts for topology generation and batch export. Every config
  runs as it is with `topon generate`.
- `docs/USAGE.md` and `docs/ARCHITECTURE.md` replace `cli.md`, `config_reference.md`, `simbox.md` and
  `cg_ensemble_execution.md`.
- `CITATION.cff` gives the citation metadata.

### Paper companion

- `demos/npjcompmat/` regenerates the data figures of the paper (Figs. 2 to 9) and a set of supporting checks. Next to
  the data of 0.1.0 it now holds derived data and raw simulation output (the stress-strain files of every pull over
  four velocity seeds, the MSD files of three cooling histories, and the LAMMPS inputs and generator settings of every
  coarse-grained network).

## 0.1.0

First public release, with the data and figure notebook of the paper.
