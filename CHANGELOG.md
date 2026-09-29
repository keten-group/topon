# Changelog

## 0.3.2

### Paper companion

- The paper companion now includes the as-built and equilibrated atomistic systems (LAMMPS data and DREIDING parameter
  files) of the six glass-transition systems. `demos/npjcompmat/data/raw/atomistic_systems.tar.xz` holds the data,
  coefficient and group files and the crosslink node and strand lists as built, and `atomistic_equilibrated.tar.xz`
  the equilibrated data files the cooling runs start from. Unpacked next to `tg_cooling.tar.xz`, they give the layout
  the cooling inputs read, and `data/raw/README.md` says how to run them.

The package code is unchanged.

## 0.3.1

### Paper companion

- The paper companion now includes the threshold and cross-validation checks reported in the paper.
  `demos/npjcompmat/scripts/checks/threshold_cv/` repeats the quadrant contrasts of Fig. 7 with the quadrants split at
  other percentiles and predicts the UTS and toughness of held-out networks from the 11 descriptors with
  cross-validated ridge regression. Its README gives the commands and the results, and `out/` holds the reference
  outputs.
- The companion's `requirements.txt` lists networkx, which `generate_checks.ipynb` imports.
- The generator benchmark tables and the software record of the companion no longer name development builds.

The package code is unchanged.

## 0.3.0

### CHARMM for polymer networks

- `chemistry.force_field = "charmm"` writes an atomistic network with CHARMM parameters instead of DREIDING. Each
  repeat unit, node molecule and bridge atom names its residue in RTF files you supply (`charmm_residue`, and
  optionally `charmm_atom_names`), and its atoms are matched to that residue by name or by graph. Every term comes from
  the RTF, PRM and stream files in `chemistry.charmm.files`, read in order.
- A residue, atom or parameter the files do not define stops the chemistry stage with the complete list, and
  `topon generate` ends with a message that names what to add. No term is filled in by default.
- The atom charges of each residue must add up to the charge of its RTF residue, and the network's charge must be an
  integer. POSS nodes, grafts, sol chains and `model_type: "coarse_grained"` are refused with CHARMM.
- The three LAMMPS stages use CHARMM styles (`lj/charmmfsw/coul/long` with arithmetic mixing and PPPM,
  `dihedral_style charmmfsw` with 1-4 weights, Urey-Bradley angles). The soft stage and the LJ ramp read their own
  includes (`.soft` and `.lj`).
- The CHARMM C35r ether force field is bundled (MIT), and a config names its files as `bundled:NAME`.
  `demos/polymer/atomistic/charmm_peg/` builds a PEG network with it.
- DREIDING remains the default, and its output is unchanged.

### Protein networks

- New command `topon protein` builds a crosslinked protein network from an amino-acid sequence (a repeat block or a
  whole chain) in CHARMM36m (all-atom) or Martini 3 (coarse-grained). The chains are grown on a cubic lattice (the
  bond-fluctuation model) and crosslinked through tyrosines (dityrosine) or cysteines (disulfide) up to the gel point.
  It writes the LAMMPS data file, the coefficient includes, the groups, three relaxation scripts and a summary.
- CHARMM36m builds place the atoms from the RTF internal coordinates and apply each crosslink as its RTF patch (`DITY`
  or `DISU`). Hydrated builds add TIP3P water, held rigid in the stage-3 MD, and NaCl. Every term, including the 1-4
  terms, NBFIX and the CMAP grids, comes from the bundled CHARMM36m files (MIT), and `--charmm-files` takes other
  files instead.
- Martini 3 builds use the Martini3-IDP bonded terms. The bundled polyply topologies cover the resilin reference, and
  any other sequence is run through polyply (the optional `martini` extra). Tryptophan is refused, because Martini 3
  represents it with a virtual site.
- A build that does not reach the gel point stops unless `--allow-no-gel` is given, and `--seed` makes a build
  reproducible.
- `python -m topon.protein_network check-bonds` reports the bonds that stay stretched or threaded through a ring after
  the relaxation. Ring threading is detected, not prevented.
- `demos/protein/` holds a CHARMM36m and a Martini 3 resilin network. `THIRD_PARTY_LICENSES.md` lists every bundled
  force-field file with its source and license.

### Fixes

- A strict topology request with defects no longer fails with `KeyError` when the degree distribution does not list
  every degree.
- Coarse-grained copolymer and graft configs no longer need their bead labels declared in `chemistry.monomers`.
- `topon init --preset` works from a regular install, because the presets now ship in `topon/presets/`.
- The molecule packer of `topon simbox` no longer divides by zero at its default `min_dist` of 0.

### Packaging

- Optional extras `martini` (polyply, and cgsmiles, which polyply 1.8 needs but does not declare) and `validate`
  (OpenMM).
- The CHARMM and Martini data folders and the presets are package data.

## 0.2.1

### Topology

- The exact search gains a fallback for targets that its random assignment of degrees cannot complete. On lattices
  whose bonds all join two sublattices (SC, BCC and Diamond with nearest neighbours), every network carries the same
  degree total on both sublattices. After 6 attempts in a row end with unfilled degree units, the search assigns the
  target degrees so that the two totals agree and then moves targets within a sublattice until every site is complete.
  The fallback runs in the Python and in the C generator.
- Networks of targets that the random assignment already reaches are unchanged for the same seed.
- The Python search makes 6 fallback attempts after its 6 regular ones. In C the fallback continues until a network
  is found or `max_trials` runs out, and a pipeline run that leaves `max_trials` at its default now gives the C route
  12 attempts per network.

### Paper companion

- `demos/npjcompmat/data/derived/generator_benchmark/` holds the generation times behind Table 1 (Appendix B) and a
  run of the exact search on every target of the coarse-grained ensemble, with a README of their columns.
  `scripts/make_timing_table.py` rebuilds Table 1 from them.
- The panel titles of Fig. 7 now read "Signatures of Toughening" and "Signatures of Strengthening".

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
