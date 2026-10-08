# Changelog

## 0.4.5

### Atomistic networks

- The atomistic placement draws from a stream keyed on the study name, so a config with every seed pinned writes the
  same files on every run, and the backbone settle runs up to 1,500 rounds instead of stopping at 400. The two strands
  of a secondary loop are drawn on opposite sides of their chord (`conformation.parallel_strands`).
- Copolymer sequences reach the atomistic route. Each repeat is built of its own monomer, grafts follow the sequence
  (any repeat with a methyl on its head atom can carry one), and CHARMM types each repeat by its own residue. The last
  repeat of a strand bonds to the next junction through its backbone atom, not through the last atom of its SMILES.
- Rings are placed whole. A ring that hangs from one atom (a phenyl) is a flat polygon turned so that the fewest bonds
  pass through it, and a ring the backbone runs through (a para-phenylene) is settled whole and flat. The DREIDING
  hard-backbone deck holds the aromatic ring atoms hard with the backbone.
- POSS cages are placed whole at their junctions, at the force field's bond lengths, and the strands are settled clear
  of them. A network with POSS nodes now takes the settled placement (`atomistic_placement: null` still gives the
  historic one). The cage shapes are stored with topon (`topon/data/poss_templates.json`), so a POSS build writes the
  same files whatever RDKit is installed.
- A designed strand that bonds to a cage is led out of it round the cage instead of starting at its center, the
  settle keeps the methyl positions along each backbone out of the cage cores, and a strand drawn between the arms
  of a cap on a neighboring site is turned off them.
- A backbone settle that stops at its round cap now always warns, with its bonds against r0 and its angle error
  (before 0.4.5 only a build with rings in its backbones did).
- `atomistic_placement: "coil"` winds each strand round its chord at `atomistic_coil_radius`, and an
  `entanglement.target_Z` on the atomistic route is met by searching that radius with Z1+ on the drawn network.
  `topon.simulation.protocols.z_target.relax_to_target` builds and relaxes again until the relaxed network reads the
  target.
- The atomistic gates read every designed pair's winding as the linking number of two network cycles
  (`topon.analysis.windings`), which changes only when a bond passes through another.
- The hard-backbone deck is shorter. The ramp runs 2,500 steps and stage 3's minimization stops at 1,000 iterations,
  with the same entanglement, density and passages on the test networks.

### DREIDING

- Every dihedral carries LAMMPS's sign of d (before 0.4.5 every torsional minimum sat where DREIDING puts a maximum),
  and unlike pairs mix geometrically with the tail correction. Simbox and `create_lammps_data_file` put LJ sigma at
  R0 / 2^(1/6), as the pipeline's writer did already (the parameter file's R0 is the position of the minimum, and
  sigma = R0 made every atom 12 % too large).
- A planar center carries DREIDING's three inversion terms at K/3 each under `improper_style umbrella`, and an
  aromatic C, N or O (and an aryl ether O) takes the resonant type (`C_R`, `N_R`, `O_R`).
- An atom with no DREIDING type stops the chemistry stage with `UntypedAtomError`, and a molecule that cannot be
  charged stops it with `ChargeError`. Both used to be written with invented types or zero charges. The `Na` row of
  the parameter files carried the mass of scandium and now carries that of sodium.
- `topon simbox` and `topon chain` also write `settings_x6.in`, DREIDING's exponential-6 as `pair_style buck`.

### simbox

- Every type id is the one DREIDING typing gives the box (the universal type map is gone), and an epoxy-amine box gets
  the types its cure creates and its `fix bond/react` templates written beside it.
- The PDMS chain ends carry two methyls (they carried one and an H), and every conformer is checked for a bond through
  a ring before it is used (the AM0270 cage used to be embedded tangled).

### Measuring and matching networks

- `topon fit` reads a network crosslinked along its chains (a data file, a strand graph with its chains or a crosslink
  build's `crosslinked_melt.npz`) and writes a config for the crosslink generator, or with `--route lattice` for the
  lattice route. `topon generate --verify` checks it, and with `--verify-replicate` it holds the builds against
  replicates of the reference.
- `topon generate --verify --relaxed` takes a run directory, checks that the relaxed file is a build of the config in
  the reference's state, and gives Z1+ per bridge, loop and dangling strand with an acceptance.
- `topon fit` writes into its configs the build options its knob was measured with and `loop_shape: "compact"`.

### Coarse-grained builds

- `conformation.loop_shape: "compact"` draws a primary loop as a compact closed walk about the size of a relaxed loop,
  where the default ring is an open polygon that strands pass through.
- `topon.conformation.place` draws the two strands of a secondary loop on opposite sides of their chord (`"together"`
  gives the drawing of 0.4.0), builds dangling chains at the DP the pipeline builds, and places the sol chains.
- Designed pairs whose chords cross are wound apart instead of through one point, and other strands are kept out of
  each braid, on both routes.

### Topology

- The strict Python search checks connectivity locally, so a trial on a large lattice takes a fraction of the time it
  took. Every seed builds the same graph as before.
- `crosslink_chains` takes three site rules from Python (end beads as sites, pendant side beads and crosslinked
  neighbors).

### Demos and documentation

- The atomistic demos were rebuilt on the new placement and relaxed again on the new deck and DREIDING parameters, and
  the copolymer demo now has its phenyl blocks.
- The POSS demo ships the settings, groups, stage scripts and manifest of its build. LAMMPS has not been run on it.
- The paper companion holds the rerun generator benchmark behind the paper's Tables 1 and 2.

### Known limits

- Backbones with two rings in one repeat unit (a biphenylene, a diphenyl ether) do not settle.
- The CHARMM hard-backbone deck does not hold ring atoms hard.
- Fitted DP-20 networks come out more entangled than the `fix bond/create` references (Z1+ per bridge about 0.22
  against 0.18).

Because of these changes, an atomistic build of a given config, a simbox box and a `place()` build of a graph with
secondary loops give different files than in 0.4.0.

## 0.4.0

### Measuring and matching networks

- `topon analyze` measures a network with a set of connectivity descriptors (the shortest cycle through each strand
  and the fraction of odd cycles, clustering, assortativity, betweenness, effective resistance, the spectrum, the
  elastically active core, cycle rank, and the chords and their orientation). It reads strand graphs, LAMMPS data files
  in the end-linked convention of `fix bond/create`, and the atomistic data files of a topon run. `--compare` gives the
  divergence of the cycle spectra, KS statistics and a composite against a reference, and `--z1` runs Z1+ when it is
  installed, with Z per strand class and the partner graph. The capacity counts it printed before are still available
  from Python (`topon.analysis.report.analyze_graph`).
- New command `topon fit` reads an existing network (an end-linked data file, an NPZ dual graph or a strand graph),
  measures it and writes a config that regenerates it. It picks the cubic cell and the neighbor cutoff by a short
  sweep through the pipeline's own first three stages, and it flags what it cannot match. `topon generate --verify`
  regenerates a config on several seeds and compares it with the reference.
- `topon inspect` also reads the network off the most relaxed data file of a run (strands, loops, chemical and
  effective P(f), and Z1+ when it is installed).
- The NPZ reader checks `schema_version` and upgrades the older eight-column files, which it used to misread.
- `analysis.z1plus` in a config says where Z1+ is installed. Z1+ is not shipped with topon (its license does not allow
  redistribution), and on Windows it runs inside WSL.

### Topology

- `topology.generator.seed` pins the generated graph on the Python and the C route, and the C generator takes
  `--seed` and `--output-dir` flags. A seed gives the graph that seeding the global random streams with it gave before.
- The exact search closes attempts that end short on scaffolds with odd cycles (SC or BCC beyond the first shell, FCC,
  `MIX`) with a second walk search, so they land instead of retrying (`odd_walks`, on by default, and C
  `--odd-walks`). An attempt that reached its target is unchanged, but a seed whose earlier attempt ended short now
  gives a different graph. `odd_walks: false` gives the graphs of 0.3.
- The exact search reads its no-slack stop from the scaffold's own coordination instead of `max_functionality`, so a
  request on a scaffold richer than `max_functionality` keeps its retries and the fallback (it used to stop after one
  attempt with a wrong message). Its repair also stops as soon as the one site left short cannot be reached, which
  saves time and changes no graph.
- A strict request that names every degree with an odd degree sum is refused before the first trial, and a request
  whose sites do not fit the lattice is refused with both numbers. `count_sites` gives the sites of a lattice (a `MIX`
  draw included) and `rescale_degree_counts` carries a P(f) to another site count. `topon doctor` checks the site
  count, the degree-sum parity and Diamond with dangling ends.
- Networks crosslinked along their chains. `topology.source: "crosslink"` (`topology.crosslinking`) grows the chains
  as a self-avoiding melt on a cubic lattice and crosslinks reactive beads that touch (every n-th bead, a list, or the
  crosslink residues of an amino-acid sequence) to an exact count, so every strand's DP and chain follow from where the
  crosslinks fell. `topology.generator.architecture: "random_crosslinked"` instead covers the strands of a sculpted
  graph with chains of a given length (`assignment.chains`). `topon.analysis.crosslinked` reads such networks back.

### Atomistic networks

- The strands of an atomistic network are drawn at the force field's bond lengths (`conformation.atomistic_placement`,
  a meander by default), and the backbones are settled before the first stage, so that no two backbone bonds start
  closer than 1.5 Å and every bond and angle starts at its equilibrium, with no bond moved through another.
- On the pipeline route the relaxation uses the hard-backbone stages by default (`simulation.atomistic_protocol:
  "hard_backbone"`, DREIDING and CHARMM). They keep backbone pairs hard through the first two stages under a
  thermostat, run 5,000 steps each, and cap the stage-3 minimization. `atomistic_placement: null` gives the build and
  stages of 0.3.
- A crossing detector (`topon.analysis.crossings`) reads each stage's backbone dump and finds every pair of backbone
  bonds that passed through each other. `topon.simulation.protocols.atomistic` gates the three stages on it and on the
  backbone bond lengths, and reports Z1+ over several seeds.
- Stage 4 writes a strand record into the run manifest, through which `topon analyze` reads any DREIDING or CHARMM data
  file of the run into strands (with one Z1+ point per repeat unit).
- New command `topon track` writes a self-contained HTML page for one or more atomistic runs, with the network, the
  Z1+ kinks and the numbers of every stage.
- The chemistry stage refuses a node type that `chemistry.node_type_map` does not have (it used to build a bare Si),
  and caps the free valences of a bare Si junction with methyls instead of hydrogens (MeSi(O-)3 at a trifunctional
  junction). The DREIDING and CHARMM epsilon ramps scale each pair's own depth.
- Because of these changes, an atomistic build of a given config gives different files than in 0.3.

### Coarse-grained builds

- `conformation.junction_jitter` and `conformation.settle_clearance` (both off by default) break the exact chord
  crossings of a lattice with several neighbor shells and part the bonds of different strands, with every move checked
  for passages. `guard_report()` counts the close chord triples (`chord_triples`) and the beads lying on another
  strand's bond (`bead_bond`).
- Designed pairs deliver their windings. A braided strand is redrawn from its chord, and its winding is measured before
  it is reported.
- A junction shell seats all of a junction's chains or none of them, with a radius no smaller than the bond.
- The relaxation gates judge persistence over every stage, stage 1 included, and describe each long bond by its
  nearest bead of another molecule.

### Reproducibility

- Stage 5 draws its noise from a stream keyed on the study name, so a config whose seeds are all pinned writes the same
  relaxed data file on every run. The relaxed data file of every build therefore changes by about the size of the
  noise (`conformation.noise_magnitude`).
- The run manifest names the process writing the run (its pid and a hash of the machine name), and a second process
  writing the same directory is warned about.

### Protein networks

- `topon protein` crosslinks a residue-level melt by default (`--crosslink-method melt`), with one lattice site per
  residue and no lattice-parity rule. The same command therefore builds a different network, and
  `--crosslink-method adjacent` builds what 0.3 built. `--contact-radius` sets how close two crosslink residues must be.
  The demos in `demos/protein/` set `adjacent` and are unchanged.

### Demos and documentation

- The atomistic demos ship the text output of a full relaxation on the new defaults (stage scripts, LAMMPS logs, run
  manifest and `topon track` page) with a README of the numbers of every stage.
- The batch workflow seeds each graph with `topology.generator.seed` and resumes a stopped run.
- `docs/USAGE.md` covers the new commands and config keys, and has a section on LAMMPS on Windows, Z1+ under WSL and
  long runs.

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
