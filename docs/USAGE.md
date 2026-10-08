# topon usage guide

This guide covers running topon end-to-end. It documents the CLI flags, the sub-system APIs, recipes for common systems and, in Appendix A, the JSON config schema.

[ARCHITECTURE.md](ARCHITECTURE.md) describes the package layout and design.

---

## 1. Install

```bash
pip install -e .
```

The runtime dependencies (installed from `pyproject.toml`) are `numpy`, `networkx`, `pandas`, `rdkit`, `pydantic`, `click`, `plotly` and `scipy`. LAMMPS (`lmp`) needs to be on `PATH` only to *run* the generated systems. Generation does not need it.

Two optional extras exist. `martini` installs polyply, which `topon protein --model martini` needs for any sequence other than the bundled resilin reference, together with cgsmiles, which polyply 1.8 needs but does not declare (`pip install -e ".[martini]"`). `validate` installs OpenMM, for comparing the CHARMM energies of the output with an independent CHARMM implementation.

Z1+ (primitive-path analysis) is used by `topon analyze --z1`, `topon inspect`, `topon fit` and `topon track` when it is installed. It is not part of topon, since its license does not allow redistribution. §3.6 says how to point topon at it.

---

## 2. Quick start

### 2.1 Interactive shell (recommended)

Running `topon` (or `python -m topon`) with no arguments in a terminal starts the `topon>` shell, where every sub-command is available directly.

```text
$ python -m topon
   +================ ... banner ... ================+
   Type `help` for the command list, or `exit` to leave.

topon> init --preset cg_kg --output my_run.json
Wrote my_run.json (preset: cg_kg, copied from config.json)

topon> doctor my_run.json
[ok]    schema_gap_extras: ...
Summary: 0 error / 0 warn / 1 ok

topon> generate my_run.json
... runs the 6-stage pipeline ...

topon> inspect output_cg
... atom counts, box, next LAMMPS commands ...

topon> exit
bye.
```

The shell has the built-ins `help`, `help <cmd>` and `exit | quit | q | Ctrl-D`. Arrow keys recall history if `readline` is installed. `help <cmd>` shows the same click `--help` page as one-shot mode.

The shell starts automatically when stdin is a TTY. `topon shell` starts it from a non-TTY context, and `topon --no-shell` prints the banner and exits.

### 2.2 One-shot mode

The same commands can be run one at a time.

```bash
# 1. Generate a starter config
topon init --output my_run.json

# 2. Validate it
topon validate my_run.json

# 3. Run the full six-stage pipeline
topon generate my_run.json --output ./runs
```

The output goes to `./runs/<study.name>/{topology, 02_Chemistry, 03_Conformation, 04_Simulation}/`. `02_Chemistry/` holds `system.data`, and `04_Simulation/` holds the LAMMPS `.in` scripts.

To run the simulation (optional), start LAMMPS on the first script.

```bash
cd ./runs/<study.name>/04_Simulation/
lmp -in minimize_1_serial.in
```

Ready-to-use configs are in `demos/templates/` (`minimal.json`, `full.json`) and in the demo folders `demos/polymer/`, `demos/poss/` and `demos/protein/`. [`demos/README.md`](../demos/README.md) lists them all.

---

## 3. CLI reference

The package installs a `topon` console script that dispatches to several sub-commands. All sub-commands accept `--help`.

```
topon [--version] [--help] <command> [options]
```

### 3.1 `topon generate` (run the full pipeline)

```bash
topon generate CONFIG_PATH [--output DIR] [--dry-run] [--export-graphml] [--export-npz]
               [--verify REFERENCE [--verify-seeds N] [--verify-only] [--relaxed DATA] [--junction-type T]
                [--verify-replicate REPLICATE ...]]
```

| Argument / Option | Description |
|---|---|
| `CONFIG_PATH` | Path to the JSON config file (required) |
| `--output`, `-o` | Override `study.output_dir` from config |
| `--dry-run` | Validate the config and exit without running |
| `--export-graphml` | Also write the graph (chains and entanglement edges) as `<study.name>.graphml` in the study folder. Same as `output.export_graphml` |
| `--export-npz` | Also write the graph as `<study.name>.npz` for graph-learning pipelines. Same as `output.export_npz` |
| `--verify REFERENCE` | After the build, regenerate the graph and compare it with a reference network (a LAMMPS data file, an NPZ dual graph or a strand graph), and write `verify.json` into the run directory (§3.6a) |
| `--verify-seeds N` | Number of seeds `--verify` regenerates, counting up from `topology.generator.seed` (or `topology.crosslinking.seed` for a crosslinked melt, default 1) |
| `--verify-only` | Build nothing and only regenerate and verify the graphs (stages 1 to 3, in a scratch directory) |
| `--relaxed DATA` | With `--verify`, a relaxed end-linked data file of this build, or the run directory it is in (the last MD checkpoint is taken), for Z1+ per strand class, the per-bridge histogram and the partners against the reference (§3.6a) |
| `--junction-type T` | With `--verify`, the junction atom type of a reference data file that is not typed 1 end, 2 interior, 3 junction |
| `--verify-replicate REPLICATE` | With `--verify` of a network crosslinked along its chains, a replicate of the reference (the same process at another seed), repeated for each. The report says per build which measures sit inside the replicates' scatter and gives a Mann-Whitney test per measure (§3.6a) |

The command runs the six-stage pipeline (Topology → Analysis → Assignment → Chemistry → Conformation → Output) and writes the LAMMPS data files and input scripts to `output_dir/study_name/`. The atomistic placement and the conformation noise draw from streams keyed on the study name, not on `--output`, so two runs of a pinned config into different directories write the same files. Give each replicate a `study.name` of its own (see `topology.generator.seed` in Appendix A).

```bash
topon generate demos/templates/full.json
topon generate demos/templates/full.json --output ./my_run
topon generate demos/templates/full.json --dry-run
topon generate ref_config.json --verify ref.data --verify-seeds 3
```

### 3.2 `topon validate`

```bash
topon validate CONFIG_PATH
```

The command prints `Configuration is valid!` or lists every validation error. It is a quick check before a long cluster run.

### 3.3 `topon init` (starter config that runs as-is)

```bash
topon init                              # fastest: write atomistic_pdms preset
topon init --preset cg_kg               # different preset
topon init --interactive                # prompt-driven walk through 6 knobs
```

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` | `config.json` | Path for the new config file |
| `--preset` | `atomistic_pdms` | One of `atomistic_pdms`, `cg_kg`, `poss`. Each copies a demo `config.json` (`demos/polymer/atomistic/basic/`, `demos/polymer/coarse_grained/basic/`, `demos/poss/`). The copies ship inside the package as `topon/presets/<name>.json`, so the presets also work from a regular install. |
| `--interactive`, `-i` | off | Prompt for the 5-6 settings that usually vary (study name, output dir, model type, lattice type+size, max functionality, DP, density) and write the result. |

The non-interactive default copies `demos/polymer/atomistic/basic/config.json`. Every preset-produced file passes `topon validate` as written.

### 3.3a `topon doctor` (semantic lint)

```bash
topon doctor my_run.json           # informational + warns
topon doctor my_run.json --strict  # warns also exit 1
```

`validate` checks the Pydantic schema. `doctor` runs a set of semantic rules that catch known problems and common mistakes.

| Rule | Level | Catches |
|---|---|---|
| `lattice_size_format` | error | `"lattice_size": 5` instead of `"5x5x5"` |
| `neighbour_cutoff_vs_box` | warn | `neighbour_cutoff` above a third of a periodic axis, where three candidate edges can close a cycle around the box |
| `site_count` | error / warn | The sites `degree_distribution` asks for do not fit the lattice (an error on SC, BCC, FCC, Diamond and a seeded `MIX`), or a `MIX` whose site count no seed fixes may not fit (a warning) |
| `degree_sum_parity` | error / warn | A `degree_distribution` that names every degree has an odd degree sum, which no graph has (an error). A partial one with an odd sum over the named degrees runs in Python, but the C binary refuses it (a warning, only with `exe_path`) |
| `diamond_dangling_ends` | warn | Degree-1 sites on Diamond at the default cutoff (four candidates per site) while most active sites are asked for degree 4, so nothing can take up the missing bonds. The exact search stops after one attempt on such a request |
| `deprecated_mix_cutoff` | warn | `mix_cutoff` still present (it loads as `neighbour_cutoff`) |
| `unknown_config_keys` | warn | A nested section carries a key the schema does not define, so it is ignored without notice |
| `unknown_node_type` | error / warn | `assignment.node_types.degree.mapping` references a type that is not in `chemistry.node_type_map`. The atomistic chemistry stage refuses such a build (an error), while a coarse-grained junction is one bead whatever its molecule (a warning) |
| `poss_at_internal_junction` | warn | POSS mapped to degree >= 2 junctions, which gives a bond longer than half the periodic box at LAMMPS stage 1 |
| `atomistic_graft_non_pdms` | warn | Graft density set on an atomistic monomer with no methyl on its head atom (its repeats are built ungrafted), read over the copolymer composition too, or a side chain that is not PDMS |
| `dp_below_kuhn` | warn | DP < 5 (conformation/entanglement edge cases) |
| `defects_endcap_safe` | ok | Reminder that loop defects skip degree-1 chain caps |
| `entanglement_target_below_floor` | warn | Bead-spring route. `conformation.entanglement.target_Z` is below the lowest value that route has reached at this DP. The warning names the floor and, for the walk, the meander's floor at that DP and whether it is lower by more than their seed spread. Each controller round is a full relaxation protocol, so this is worth catching before it starts |
| `conformation_build_knob` | error | Bead-spring route. A `target_Z` with neither `coil_ratio` nor `build_density`, and nothing measured for that DP and placement to seed the controller from |
| `atomistic_target_placement` | error | Atomistic route. A `target_Z` with `atomistic_placement` set to something other than `"coil"`, whose radius is what a target turns |
| `schema_gap_extras` | ok | Config has `simulation`/`execution` (not Pydantic-validated, the CLI handles them through `load_config_full`) |

To add a rule, write `check_<name>(cfg, raw) -> list[Issue]` in `topon/diagnostics/rules.py` and append it to `RULE_REGISTRY`.

### 3.3b `topon inspect <run_dir>` (post-run summary)

```bash
topon inspect runs/my_study [--no-z1] [--z1-exe EXE] [--config CONFIG]
```

The command summarizes a finished run, so there is no need to read the `system.data` headers by hand. `RUN_DIR` is the study folder or its parent. The command reads each stage directory (`02_Chemistry/`, `03_Conformation/`, `04_Simulation/` in the Pipeline layout) or a flat folder with every file at the top level, and prints the following.


- atom count, atom-type count, box dimensions
- what the topology stage was asked for against what it produced, from the run manifest
- per-stage status (which files landed, what they say)
- the loops and P(f) requested against achieved, when the assignment stage recorded them
- the network read off the most relaxed data file, when that file is in the end-linked convention or is an atomistic file with a strand record (strands by class, primary and secondary loops, chemical and effective P(f), and Z1+ when it is installed, see §3.6). `--no-z1` skips Z1+, and `--z1-exe` or `--config` say where it is
- the next LAMMPS commands to run

The data file read for the network is the furthest checkpoint in `04_Simulation/`, or
`03_Conformation/system_relaxed.data` when no stage has run. The summary names it, because
Z depends on the state it was read in.

`Pipeline` writes the run manifest (`manifest.json`) into the run directory as
it goes. It records what each stage was asked for, what it produced and how
long it took. For stage 1 it records the lattice, the search, the requested
and achieved degree counts, the giant-component fraction and the seed.
`topon inspect` prints it as follows.

```
  Topology (generate, python generator, exact search):
    lattice   : 9x9x9 SC, cutoff 1, max f 4
    graph     : 572 nodes, 972 edges, 0.22 s
    degree    :       0      1      2      3      4
      requested     157     73     33     59    407
      achieved      157     73     33     59    407
      match     : exact
    sculpt    : 0.20 s in the search; 1 attempt(s); giant 1.000; 1 component(s); 444 augmentations; 372 repairs
```

A strict run records the same two rows but lists only the degrees its target
named (the others read `-`), plus any `e:N` budget. The manifest is advisory.
Nothing reads it back to make a decision, so a missing or half-written
manifest only loses detail in the summary. In Python, read it with
`topon.core.manifest.read_manifest(run_dir)`. Its `run` entry names the
process writing the directory (`pid`, `host` as a short hash of the machine
name, and `started`), so that a second process started on the same directory
warns that it is not alone (§8).

### 3.3c `topon recipes` (common use cases)

```bash
topon recipes
```

The command prints a short table that maps common tasks to commands for all sub-systems (polymer networks through Pipeline, protein networks, simbox, single-chain, the batch topology demo, `inspect` and `analyze`). The rows are defined in `topon/cli.py:recipes()`.

### 3.4 `topon simbox` (pack a crosslink box)

simbox is an independent sub-system. It builds Epoxy-PDMS, Amino-PDMS and AM0270-POSS molecules, packs them at a target density, and writes DREIDING LAMMPS data and input scripts.

```bash
topon simbox [--output DIR] [--n-epoxy N] [--n-amino N] [--n-poss N]
             [--density FLOAT] [--seed INT]
```

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` | `simbox_output` | Output directory |
| `--n-epoxy` | `50` | Number of Epoxy-PDMS molecules |
| `--n-amino` | `25` | Number of Amino-PDMS molecules |
| `--n-poss` | `10` | Number of AM0270-POSS molecules |
| `--density` | `0.85` | Target packing density (g/cm³) |
| `--seed` | `42` | Random seed for reproducible packing |

```bash
# Default ~85-molecule system
topon simbox

# Production: 600 epoxy + 300 amino, no POSS
topon simbox --output pdms_box --n-epoxy 600 --n-amino 300 --n-poss 0

# 50% POSS
topon simbox --output poss50 --n-epoxy 60 --n-amino 15 --n-poss 15
```

The output is a single flat directory.

```
simbox_output/
├── system.data           # LAMMPS data file (atom_style full)
├── ff_coeffs.in          # Force-field coefficients
├── settings.in           # pair_coeff / bond_coeff / angle_coeff / dihedral_coeff
├── settings_x6.in        # DREIDING's exponential-6 as pair_style buck (§4.1, no script includes it)
├── groups.txt            # group definitions by reactive-group type
├── 1_minimize.in         # Stage 1: soft push-off + CG minimisation
├── 2_nvt.in              # Stage 2: NVT thermalisation
├── 3_npt.in              # Stage 3: NPT density equilibration
├── 4b_crosslink.in       # Stage 4: crosslink template (fix bond/react or bond/create)
├── pre_react_primary.mol, post_react_primary.mol, rxn_map_primary.txt
└── pre_react_secondary.mol, post_react_secondary.mol, rxn_map_secondary.txt
                          # fix bond/react templates in the box's own type ids,
                          # written when the box holds epoxide and amine
```

Then run LAMMPS.

```bash
cd simbox_output && lmp -in 1_minimize.in
```

See §4.1 for the simbox Python API.

The files use DREIDING as follows since 0.4.5. LJ sigma is R0 / 2^(1/6) (the parameter file's R0 is the position of the minimum), unlike pairs mix geometrically with the tail correction (`pair_modify mix geometric tail yes`), and every dihedral has LAMMPS's sign of d (the file's negated, so an sp3-sp3 torsion has its minimum staggered). The Si-O-Si angle is still DREIDING's generic O_3 angle, 104.51 degrees, against 143 to 148 in siloxanes.

### 3.5 `topon chain` (single chain in solvent)

The command builds a single polymer chain in solvent and writes DREIDING LAMMPS files.

```bash
topon chain --chain-smiles SMILES --dp N [options]
```

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` | `chain_output` | Output directory |
| `--chain-smiles` | *(required)* | SMILES for the polymer repeat unit |
| `--dp` | *(required)* | Degree of polymerization |
| `--solvent-smiles` | `None` (toluene fallback) | Single-solvent SMILES, ignored if `--solvent-mixture` is set. If both are unset, `run_workflow` falls back to toluene. |
| `--n-solvent` | auto | Number of solvent molecules, calculated from the density if omitted |
| `--solvent-mixture` | `None` | Multi-solvent JSON (e.g., `'[{"smiles":"...","weight_fraction":0.5}, ...]'`) |
| `--graft-density` | `0.0` | Graft attachment probability per backbone unit (0-1) |
| `--graft-smiles` | `None` | SMILES for graft repeat unit (required if `--graft-density > 0`). The side chain is built of PDMS whatever this says, and another SMILES gets a warning |
| `--graft-dp` | `5` | Repeat units per side chain |
| `--density` | `0.85` | Target packing density (g/cm³) |
| `--seed` | `42` | Random seed |

```bash
# PDMS in toluene
topon chain --chain-smiles "[Si](C)(C)O" --dp 20 \
            --solvent-smiles "Cc1ccccc1" --n-solvent 200

# Fluorinated PDMS in THF
topon chain --chain-smiles "[Si](C)(CCC(F)(F)F)O" --dp 15 \
            --solvent-smiles "C1CCOC1" --n-solvent 100 --output fpdms_in_thf

# With grafts
topon chain --chain-smiles "[Si](C)(C)O" --dp 30 \
            --graft-density 0.1 --graft-smiles "[Si](C)(C)O" --graft-dp 5 \
            --solvent-smiles "Cc1ccccc1" --n-solvent 150
```

The chain is `--chain-smiles` written `--dp` times between two trimethylsilyl end caps (with `--graft-density`, a grafted repeat carries a PDMS side chain in place of the last methyl on its first atom). The head cap bonds to the first atom of the SMILES. The tail cap bonds to the atom the next repeat would bond to, which for a repeat unit that ends in a side group, such as polystyrene `CC(c1ccccc1)` or poly(methyl acrylate) `CC(C(=O)OC)`, is the backbone carbon before the branch (before 0.4.5 it bonded to the last atom of the SMILES, a phenyl or ester carbon). A bridge O goes between a cap and the chain where the element `topon chain` guesses for that end of the repeat is Si, as the cap's is, which for PDMS puts one O between the head cap and the first Si. The guess reads the SMILES text, not the molecule, so write a siloxane Si first (PDMS written from its O, `O[Si](C)(C)`, gets an O-O bond at the head and a Si-Si bond at the tail).

The output directory contains the files below.

```
chain_output/
├── system.data       # LAMMPS data (chain + solvent, DREIDING)
├── ff_coeffs.in
├── settings.in       # pair coefficients (re-applied after soft push-off)
├── settings_x6.in    # DREIDING's exponential-6 as pair_style buck (§4.1)
├── groups.txt
├── 1_minimize.in
├── 2_nvt.in
└── 3_npt.in
```

### 3.6 `topon analyze` (connectivity descriptors, Z1+ and comparison)

The command measures one network. It gives the connectivity descriptors that tell network topologies apart, the chord statistics and, when Z1+ is installed, the primitive-path numbers. With `--compare` it also says how far the network is from a reference.

```bash
topon analyze PATH [--compare OTHER] [--z1] [--json OUT] [--fast]
              [--format text|json] [--nodes NODES] [--seed N] [--strands MANIFEST]
              [--z1-exe EXE] [--z1-distro NAME] [--config CONFIG]
```

| Argument / Option | Description |
|---|---|
| `PATH` | A strand graph (`.gpickle`, `.nodes` with its `.edges`, `.edges` with `--nodes`, `.graphml`), a LAMMPS data file in the end-linked convention (below), or an atomistic data file of a topon run (read through the run's strand record, below) |
| `--strands MANIFEST` | The run manifest (or run directory) of an atomistic data file that is not inside its run directory |
| `--compare OTHER` | Reference to compare against, a graph, an end-linked data file or a report written by `--json` |
| `--z1` | Run Z1+ on `PATH`, which must then be a data file |
| `--json OUT` | Also write the report to `OUT`, and the distributions beside it as `OUT` with `.npz` |
| `--fast` | Skip the cycle spectrum, the effective resistance and the edge betweenness (most of the time on a large network) |
| `--format`, `-f` | `text` (default) or `json` on stdout |
| `--seed` | Seed of the sampled betweenness and path lengths (default 0) |
| `--z1-exe`, `--z1-distro`, `--config` | Where Z1+ is (see *Z1+* below) |

```bash
topon analyze network.gpickle
topon analyze 04_Simulation/stage5_final_quench.data --z1 --json final.json
topon analyze network.gpickle --compare final.json
topon analyze run/03_Conformation/system_relaxed.data --z1     # atomistic
```

**Input.** A graph is read in the strand convention. Nodes are junctions and the free ends of dangling strands, edges are strands, a primary loop is a self-loop and a secondary loop a parallel edge. A node's `kind` attribute (`junction` or `end`) says which it is. Graphs from topon's generators carry none, and there a degree-1 node is a dangling end and a degree-0 node an empty lattice site, which is dropped.

A data file must be in the end-linked convention of `fix bond/create` datasets (type 1 chain end, 2 interior, 3 junction, one molecule per chain and one per junction), which topon writes with `output.lammps_convention: "endlinked"` or `topon.writers.write_endlinked`. A file typed another way is read from Python with `read_endlinked(path, junction_type=T)` (or with `topon fit --junction-type T`). The atoms of type `T` are then the junctions, and each connected run of the other atoms is a chain whose ends come from its bonds. Each chain is a bridge (its two ends on two different junctions), a loop (both on one), dangling (one end bonded) or free (sol). Free chains are counted but are not in the graph.

A network crosslinked along its chains, whose junctions are chain beads, is read from Python with `topon.analysis.crosslinked.read_crosslinked(path, crosslink_bond_types=...)`. Any bead with three or more bonds is a junction bead, junction beads bonded to each other form one junction, each run of the other beads is a strand with its `dp`, and each strand carries its chain and its place along it. `from_bfm_snapshot` reads a snapshot of the bond-fluctuation generator of `topon.protein_network.bfm` in the same way. The result is a strand graph like any other, so `describe`, `compare` and `topon fit` (from a saved `.gpickle`) take it.

**Atomistic data files.** A DREIDING or CHARMM data file puts every atom in one molecule and types atoms by chemistry, so it cannot be read into strands on its own. Stage 4 writes a strand record into the run's `manifest.json` (section `strands`, in LAMMPS atom ids, which LAMMPS keeps through every stage), and `topon analyze` reads any data file of the run with it. The manifest is looked for in the file's folder and the two folders above it, and `--strands` names it otherwise. The record holds, per strand, its class, its DP, the junction atom at each bonded end, its backbone in order from the bonded end (bridge atoms included), one atom per repeat unit, a dangling strand's free end, and its heavy atoms and hydrogens as id ranges. Per node it holds its atoms, junction caps included. The strand graph then describes the same network as the coarse-grained build of the same config (every descriptor that does not sample is equal, and chords come out in Å), and the report gives the backbone bonds against their equilibrium lengths in place of the bead statistics. For Z1+ a strand is one point per repeat unit (the Si of each Si-O pair in PDMS) with the junction Si at each bonded end, so a DP-N bridge exports N + 2 points, as a bead-spring bridge of N beads does. `topon.analysis.atomistic` has the reader (`read_atomistic`) and `measure_atomistic`.

**What is measured, and on what.** Counts and degrees are taken on the whole network. Everything else is taken on the *core*, the largest connected component with self-loops removed, parallel strands merged, and degree-1 nodes pruned until none are left (the elastically active backbone). `lambda2` and the path lengths grow or shrink with the number of junctions, so they compare only between networks of matched size. Everything in the composite is size-free.

The counts and degrees are listed below.

| Descriptor | Definition |
|---|---|
| `n_junctions`, `n_end_nodes` | Junction nodes, and free ends of dangling strands (one per dangling strand) |
| `n_vacancies` | Lattice sites with no strand, left out of everything else |
| `n_bridging_chains`, `n_dangling_chains`, `n_sol_chains` | Strands between two junctions, strands with a free end, and chains bonded to nothing (from the data file, or `G.graph["sol_chains"]`) |
| `n_primary_loops` | Strands that leave a junction and return to it (self-loops) |
| `n_secondary_loops` | Extra strands on a junction pair that is already joined (a pair with three strands counts two) |
| `deg_chem_*` | Chemical functionality of each junction, a primary loop counting twice. The histogram (`_dist`), mean, standard deviation, Shannon entropy of the histogram (natural log), skewness, and Sarle's bimodality coefficient (g^2 + 1) / (k + 3(n-1)^2 / ((n-2)(n-3))) with g the skewness and k the excess kurtosis (above 5/9 suggests two modes) |
| `deg_eff_*` | The same for the effective functionality, primary loops left out |
| `n_components`, `giant_frac_junctions` | Connected components, and the fraction of junctions in the largest |
| `cycle_rank`, `cycle_rank_per_junction` | Independent cycles of the largest component, E - N + 1 (edges minus nodes plus one), and per junction |
| `core_nodes`, `core_edges`, `core_deg_*` | Size and degree histogram of the core |
| `frac_active_junctions` | Core nodes over junctions, the fraction that is elastically active |
| `core_cycle_rank_per_node` | E - N + 1 of the core, per core node |

The structure of the core is described by the descriptors below.

| Descriptor | Definition |
|---|---|
| `edge_shortest_cycle_*` | For each strand, the length of the shortest cycle through it (one plus the shortest path between its two junctions with the strand removed). `_dist` is the histogram, `_mean` the mean over strands on a cycle |
| `frac_odd_cycles` | Fraction of strands on a cycle whose shortest cycle is odd. It is 0 on any bipartite scaffold (SC, BCC and Diamond at their first shell) |
| `frac_edges_in_cycle_le4`, `_le6` | Fraction of all core strands whose shortest cycle has at most 4 (or 6) strands |
| `transitivity_core` | Three times the triangles over the connected triples |
| `avg_clustering_core` | Mean over nodes of the fraction of neighbor pairs that are joined |
| `square_clustering_core` | Mean over nodes of the square clustering coefficient (the fraction of possible squares through a node that exist, as NetworkX defines it) |
| `assortativity_core` | Pearson correlation of the degrees at the two ends of a strand |
| `lambda2_full`, `lambda2_core` | Algebraic connectivity (the second-smallest eigenvalue of the graph Laplacian) of the largest component and of the core. Size-dependent |
| `graph_energy_core_per_node` | Sum of the absolute adjacency eigenvalues over the node count (cores up to 6,000 nodes) |
| `spectral_radius_core` | Largest adjacency eigenvalue |
| `avg_path_core`, `diameter_core_est` | Mean shortest-path length in strands from 200 sampled nodes to every other, and the largest distance seen from them (a lower bound on the diameter). Size-dependent |
| `*_betweenness_core`, `betweenness_gini_core` | Node betweenness (the normalized fraction of shortest paths through a node, estimated from 400 sampled sources), as maximum, mean and Gini coefficient. The Gini coefficient is 0 when every node carries the same load and approaches 1 when one carries all of it |
| `edge_betweenness_gini_core`, `_cv_core` | The same for strands, as a Gini coefficient and a coefficient of variation |
| `mean_eigvec_cent_core`, `eigvec_cent_cv_core` | Mean and coefficient of variation of eigenvector centrality |
| `edge_eff_resistance_mean`, `_cv` | Effective resistance between the two junctions of each strand when every strand is a unit resistor, from the pseudo-inverse of the Laplacian (cores up to 4,000 nodes). Low and uniform means many parallel load paths |
| `kirchhoff_per_node` | The Kirchhoff index (the sum of the resistance over all node pairs) over the node count, which is the trace of the Laplacian pseudo-inverse |
| `n_bridges_core`, `n_articulation_core` | Strands, and nodes, whose removal splits the core |
| `mean_k_core`, `max_k_core` | Mean and largest k-core number |

When the graph carries positions and a cell (a data file always does, in sigma, and a topon graph in lattice units), two geometric descriptors are added.

| Descriptor | Definition |
|---|---|
| `chord_mean`, `_sd`, `_cv`, `_p5`, `_p50`, `_p95`, `_max` | Minimum-image distance between the two junctions of each bridge |
| `orientation_eigs` | Eigenvalues of the mean of u u^T over the unit chord vectors u (1/3 each for an isotropic network) |

The distributions behind the scalars (effective and core degrees, path lengths, betweenness, eigenvector centrality, shortest cycle per strand, effective resistance, edge betweenness, chords) go into the `.npz` that `--json` writes.

**Comparison.** `--compare` (`topon.analysis.compare(a, b)`) reports the Jensen-Shannon divergence of the two shortest-cycle spectra (base 2, lengths 3 to 15, 0 for identical and 1 for disjoint), the Kolmogorov-Smirnov statistic of every distribution both carry, and a composite. The composite is the mean of eleven terms, namely the cycle divergence, the KS statistic of the core betweenness, and the relative deviation |a - b| / |b| of the odd-cycle fraction and of eight size-free scalars (transitivity, square clustering, assortativity, betweenness Gini, resistance CV, graph energy, mean shortest cycle, fraction on cycles of at most 4). It is 0 for identical networks and grows the further apart they are. The second network is the reference, so the measure is not symmetric. A term that the reference has at zero cannot be scaled this way and is left out and named (e.g., comparing against an SC lattice drops transitivity and the odd fraction). `--fast` leaves out the cycle and resistance terms too, so compare composites only over the same terms. Given a population (the descriptors of a whole sweep), `compare(a, b, population=...)` divides each scalar by max(|b|, its standard deviation over the population) instead.

**Z1+.** Z1+ (Kröger's primitive-path analysis) is not part of topon, since its license does not allow redistribution. Install it yourself and name the binary with `analysis.z1plus.executable` in a config (read with `--config`), or with `--z1-exe`. The default is `~/z1/Z1+`. Z1+ ships for Linux, so on Windows it runs inside WSL (`analysis.z1plus.wsl` is `auto`, `always` or `never`, and `wsl_distro` names the distribution), and the path is a Linux path. When Z1+ cannot be reached, `--z1` reports why and the rest of the report is unchanged. Every chain is exported junction to junction (with the junction bead at each bonded end and the two end beads moved by 1e-3 sigma, so that chains sharing a junction do not share a coordinate, which Z1+ cannot take), unwrapped bead by bead, in molecule order, and Z1+ runs with `-SP+`. The report gives Z per chain over the whole system and per class (bridge, loop, dangling, free), the entanglement length from the classical Kuhn, modified Kuhn and coil estimators, and the chain-chain partner graph (pairs of chains with at least one entanglement between them, and partners per chain). Z1+ counts the kinks of the shortest paths between the current junction positions, so it changes through equilibration and compression even where nothing crosses. Compare two measurements only at the same density and temperature. From Python, `topon.analysis.z1plus.measure_checkpoint(path)` measures a checkpoint, and `run_z1(chains, box)` runs Z1+ on any set of unwrapped chains.

`topon.analysis.report.analyze_graph()`, which gave the capacity counts (triangle-closing edges, parallel strands, entanglement pairs) that this command printed before 0.4.0, is still there for Python callers.

### 3.6a `topon fit` (a config that regenerates an existing network)

The command reads a network, measures it and writes a topon config whose generated networks match it, with a report of what was measured, what was chosen and what could not be matched. The check is `topon generate CONFIG --verify REFERENCE`.

```bash
topon fit REFERENCE [--out CONFIG] [--junction-type T]
          [--lattice SC|BCC|FCC|Diamond|MIX] [--mix 0.9,0.05,0.05]
          [--sweep-cutoffs 1.74,2.01] [--seeds 2]
          [--seed 1] [--max-functionality F] [--dp N] [--density RHO]
          [--no-z1] [--z1-exe EXE] [--z1-distro NAME] [--name NAME]
          [--no-control] [--quiet]
          [--crosslinked] [--route crosslink|lattice] [--crosslink-bond-type T]
          [--sequence SEQ [--repeats N] [--crosslink-residue Y]]
          [--reactive-every K [--reactive-start S]] [--packing P]
          [--contact-radius R]
topon generate CONFIG --verify REFERENCE [--verify-seeds 3] [--relaxed DATA]
                      [--verify-replicate REPLICATE ...]
```

| Argument / Option | Description |
|---|---|
| `REFERENCE` | An end-linked LAMMPS data file (the `fix bond/create` convention, or topon's `endlinked` output), an NPZ dual graph (schema 1, or topon's own schema 2, see *The NPZ dual graph* in Appendix A), or a strand graph (`.gpickle`, `.graphml`, `.nodes`/`.edges`). Or a network crosslinked along its chains, as a data file, a strand graph with its chains or a `crosslinked_melt.npz` (see *Networks crosslinked along their chains* below) |
| `--out`, `-o` | The config to write (default `<reference stem>_config.json`). The report goes beside it as `<stem>.fit.json` |
| `--junction-type T` | For a data file typed other than 1 end, 2 interior, 3 junction. The atoms of type `T` are the junctions, every connected run of the other atoms is a chain, and a chain's ends are found from its bonds. A junction bonded to the middle of a chain is refused (that is a crosslink along a chain, not an end-linked network) |
| `--lattice`, `--mix` | Lattice of the fitted cell, SC by default, or `MIX` with its SC,BCC,FCC fractions (the two go together). The cutoff candidates are simple-cubic shell radii in cell units. On BCC, FCC and Diamond they are ranges without a shell meaning, and the report says so |
| `--sweep-cutoffs` | Cutoffs to sweep in place of the ones the rule of thumb gives |
| `--seeds`, `--seed` | Builds per candidate cutoff, and the first seed. The config is pinned to `--seed`, so its graph is the sweep's first build of the chosen cutoff |
| `--max-functionality` | Junction ceiling, by default the reference's highest degree. One below the reference's highest effective degree is refused, and one below its highest chemical degree is flagged, since the loops that would pass it go unplaced |
| `--dp`, `--density` | For a graph file, which carries neither (the fallbacks are the schema's DP 25 and a bead density of 0.85, both flagged) |
| `--no-z1` | Skip Z1+ on the reference, so no entanglement target is written |
| `--no-control` | Skip the nearest-neighbor control row of the sweep |
| `--crosslinked` | Read `REFERENCE` as a network crosslinked along its chains. Otherwise it is decided from the file |
| `--route` | For a crosslinked reference, the generator the config is for, `crosslink` (default, `topology.source: "crosslink"`) or `lattice` (`architecture: "random_crosslinked"`) |
| `--crosslink-bond-type T` | For a crosslinked data file, a bond type of its crosslinks (repeat for several), so that a chain with a crosslink within itself can be walked. Read off the bonds when not given |
| `--sequence`, `--repeats`, `--crosslink-residue` | For a crosslinked reference with one chain length, the chain as a sequence (one bead per residue, the crosslink residues reactive), checked against where its crosslinks sit and written in place of the period read off them |
| `--reactive-every`, `--reactive-start` | For a crosslinked reference, the period of the reactive beads, checked and written in place of the one read off them |
| `--packing`, `--contact-radius` | For a crosslinked reference, the generator's lattice packing and contact radius, in place of the ones read or swept |

**What is measured.** The strand classes (bridge, primary loop, dangling, sol), the effective and chemical P(f), the secondary loops with the effective degrees of the junction pairs they join, the primary loops by the effective degree of the junction that carries them, DP (mean, polydispersity and histogram), bead density, the descriptors of §3.6 and, with coordinates, the junction-junction separation of the bridges (mean, CV, 95th percentile) and the strand end-to-end distance. From a data file with Z1+ installed it adds Z1+ per strand class. The sculpt target is the P(f) in topon's site convention (i.e., degree 1 counts the dangling-chain ends and the junctions with one other strand).

**The cell and the cutoff.** The cell is the smallest cube whose lattice holds the active sites (for a `MIX`, whose site count is a draw, with three standard deviations to spare), and the junction-only cell is reported beside it. The cutoff comes from a rule of thumb, cutoff ≈ p95(junction separation) / site spacing, with the site spacing the reference box over the cell's edge count. The candidates are the outermost three simple-cubic shells whose radius is within the rule, and a short sweep decides between them. Every candidate is built through the pipeline's own stages 1 to 3 on each seed (`Pipeline.run_graph_stages`, so it is exactly the graph `topon generate` builds from that config and seed), described, and scored with the composite of `compare`. The composite that chooses divides each term by max(|reference|, its spread over the sweep's graphs), because a reference value near zero would otherwise turn seed noise into the ranking. The spread is taken over this sweep's graphs (the candidates and the control), so `--no-control` or other `--sweep-cutoffs` give other numbers. The report and `--verify` also give the composite over the reference value alone, the default of `compare`. A nearest-neighbor row (cutoff 1.0) is scored as a control and never chosen. Without coordinates (topon's own NPZ, or a graph without positions) there is no rule, and the sweep covers the shells 2, 3, 4, 6 and 8 whose radius is within a third of the box. The sweep seeds the global random streams for each build (and restores them), so a polydisperse DP draws the same way every time.

**What the config holds.** `topology.generator` holds the cell, the chosen cutoff, every degree count, `search: "exact"` and the seed. `assignment.dp_distribution` holds the mean and PDI (the Schulz-Zimm form the schema holds, so a polydisperse reference is matched in its first two moments and flagged, because `topon generate` draws those DPs from the global stream that no config seed pins), with `endlinked_dangling` for a data file or NPZ. `assignment.defects` holds the primary loops placed by effective degree, the secondary loops with the reference's endpoint degrees and the sol chains. `chemistry` is coarse-grained at the reference density, `simulation` uses the push-off protocol compressing to that density, and `output.lammps_convention` is `"endlinked"`. When Z1+ measured the reference, `conformation` holds the placement route, its build knob and `entanglement.target_Z` and `target_hist` per bridge. The route is the random walk when the target is at or above the lowest final-state Z the walk has reached at that DP, and the meander below it. The knob is where the controller would start (`seed_actuator`) on the calibration rows of this DP, or of the nearest DP when this one has none (flagged, since Z per strand grows steeply with DP). Below the route's floor the knob is the one that measured lowest (coil ratio 1.402 for DP 20). The block also carries the build options the knob was measured with, and says so in a note. The DP-20 rows are `place()` builds with the junctions jittered (see *Chords that cross* in Appendix A), so a DP-20 meander config gets `junction_jitter` 0.15 and `settle_clearance` 1.0 and a walk the jitter alone (the same coil built without them pinches strands at their junctions and ends at a higher Z). Since 0.4.5 the block also asks for compact primary loops (`loop_shape: "compact"`), whether or not the graph has any, while the schema's default stays `"ring"`. Compact loops brought the relaxed fits closer to the references, and a graph without primary loops builds the same under both. A config with no Z1+ target gets no `conformation` block. Every one of these is a final-state number that only MD can check.

**What is flagged.** A degree above `max_functionality` (refused), Diamond at its canonical cutoff with dangling ends and half or more of the sites four-fold (the `diamond_dangling_ends` doctor rule, refused when no candidate is left), an entanglement target below the route's floor, calibration rows that were not built with `place()`, loops on junctions the lattice route cannot hold (a junction with loops only is a vacancy to the sculptor, and one with a single other strand reads as a dangling end, so their loops go to other junctions and the chemical P(f) moves), the bead count against the reference's, a choice inside the seed scatter, and every warning `topon doctor` gives on the written config.

**NPZ dual graphs.** A dual graph records which crosslinkers a chain reaches, not how many of its ends bond each one. In a schema-1 file a primary loop and a dangling chain both reach one crosslinker and cannot be told apart, so `topon fit` reads them all as dangling, fits no primary loops and says how many chains that covers. Such files carry crosslinker coordinates in sigma and get the rule of thumb. topon's own NPZ (schema 2, coordinates NaN) is read the way its writer writes it. Every graph node is a crosslinker row, so a row with one chain is a dangling-chain end site and a chain reaching one crosslinker can only be a primary loop, and the file measures like the data file of the same build except for the sol chains the writer leaves out. Its box is the lattice cell, so it has no density (pass `--density`) and gets the wider sweep. No NPZ has a Z target.

**The verification.** `topon generate CONFIG --verify REFERENCE` runs the build as usual, then regenerates the graph on `--verify-seeds` seeds (from the config's own) and measures each the way the reference is measured. It reports requested against achieved P(f) (exact or not, per seed), the defects asked for against those placed, the bead count, the chemical P(f) against the reference's, the composite with its per-descriptor table, and the reach of the bridges as built (lattice positions scaled to the box the density gives, labeled as the built state). When one term makes up much of the composite it is named, with the composite without it. The graph regenerated for the config's own seed is compared with the one the pipeline just built, edges and DPs, which checks that the build is deterministic. The config is taken as the pipeline took it (validated, legacy keys renamed, defaults filled), and a loaded topology has no P(f) to verify. Nothing in this part reads `conformation.loop_shape`, since the graphs come from stages 1 to 3. The entanglement part needs a relaxed system, and `topon generate` runs no MD. With `--relaxed DATA` it adds Z1+ per bridge, per primary loop and per dangling strand with KS p-values against the reference (a class on one side only is listed with its counts), the per-bridge histogram P(Z) beside the reference's and against `target_hist`, the partner pairs with the mean partners per strand and per bridge, the reach and end-to-end distance after relaxation, and the acceptance (Z per bridge within 10 % of the reference's and a per-bridge KS p above 0.05, `topon.inverse.verify.Z_ACCEPT`).

`DATA` is a data file or a run directory (a `topon generate` study folder or its `04_Simulation` folder), from which the furthest MD checkpoint is taken. A directory with no MD checkpoint is refused before anything is built, and `03_Conformation/system_relaxed.data` is never taken from one. The file is checked before its Z1+ is read, and every failure is a flag in the report.

- The network. Relaxation moves beads and never bonds, so a relaxed build's strand graph is the config's. Strands per class and the effective P(f) say whether it is a build of this config at all ("not this config's network", and then no verdict is given), a Weisfeiler-Lehman hash of the strand graph says whether it is the graph of the config's seed ("another seed"), and the sol chains, DP per class and bead count say whether its strands are the ones the pipeline writes.
- The state. The first line of a LAMMPS `write_data` file with a timestep above zero ("not a relaxed state" otherwise, since the Z1+ of a build is not the final-state number the reference carries), the units on that line against the reference's, and the density and the temperature (from the Velocities section) within 2 % and 10 % of the reference's. A file with no Velocities section is flagged as unchecked on temperature rather than passed.

**Building and relaxing a fitted config.** Only the bead-spring route that `topon.conformation.place` draws honors the fitted `conformation` block. `topon generate` draws every strand on its chord whatever `placement` says, and the push-off keeps the entanglement state a build starts with, so a relaxed `topon generate` build of a fitted config reads as the same network at a much lower Z. Build the fitted knob with `place` at the config's seed (the graph `--verify` regenerates, which the hash check confirms), relax it with the deck `topon generate` writes for the config, and pass the last checkpoint to `--relaxed`. `place` builds the same strands as the pipeline (dangling chains at their DP and the sol chains), so the network check reads such a build as the pipeline's.

#### Networks crosslinked along their chains

A randomly crosslinked or vulcanized melt, or a protein network crosslinked at fixed residues, has no crosslinker at a chain end. Two chain beads are bonded instead. `topon fit` reads such a reference with `topon.analysis.crosslinked` (a junction is a crosslink, a strand the run of chain beads between two) and writes a config for the crosslink generator (`topology.source: "crosslink"`, see `topology.crosslinking` in Appendix A), or with `--route lattice` for the lattice route (`architecture: "random_crosslinked"`).

**The reference.** A LAMMPS data file (`atom_style full`, one molecule per chain) is read this way when the end-linked reader refuses it and half or more of its beads with three or more bonds have two bonds within their own molecule, or always with `--crosslinked`. A chain carrying a crosslink within itself has no single backbone path unless the crosslinks' bond type is known. When chains come back without an order, a bond type whose bonds nearly all join two beads of three or more bonds is read as the crosslinks' type, the file is read again, and a note says so. `--crosslink-bond-type` names it outright. A strand graph (`.gpickle`) is read this way when it carries `G.graph["chains"]` and `G.graph["architecture"] == "random_crosslinked"`, as the crosslinked reader and the crosslink generator write it, with its cell in `G.graph["box"]` (lattice units unless `G.graph["units"]` is `"sigma"`). topon's own data file of a crosslink build carries no chains (a crosslink is built as one bead its strands share), so one of its builds is fitted from the `topology/crosslinked_melt.npz` it writes beside the graph. No Z1+ target is read, since these networks are fitted on their connectivity and their chains.

**What is measured** (the report's `reference.crosslinked`). The chains (count, length, sol), the crosslinks (count, within one chain or between two, fused junctions), each crosslinked bead's position along its chain, the gaps of the crosslinks within a chain and their parity, passes per chain, the strand DP of each class, the descriptors and, for a lattice reference, its packing.

**What the config holds.**

| Key | From |
|---|---|
| `chains` | One type per chain length with its count, in the order the reference lists them. Molecules of one or two beads (solvent, ions) are left out and flagged. The reactive beads are read off the positions. The greatest common divisor of the spacings along each chain is the period, written as `reactive_every` from `reactive_start`. The period's slots are held against a uniform draw, and when the reference leaves some of them empty far more often than such a draw would, the positions seen are written as a `reactive` list (or, for the beads next to the chain ends, as `min_dangling_dp: 1`). A chain length that caught no crosslink where the others' rate predicts several is written with an empty list (a diluent). `--sequence` writes the sequence instead, after checking every crosslinked bead is on a crosslink residue |
| `crosslinks` | The count, which the generator meets exactly |
| `contact_radius` | 1.0 (face contacts) when nearly all the crosslinks within a chain close across an odd number of bonds (on the cubic lattice two beads of one chain touch face to face across odd gaps only), 1.5 (the generator's default) when half or more are even, and otherwise both are built and scored |
| `min_gap` | The generator's 6, or the reference's smallest gap within a chain when that is smaller (3 at least). A reference with no crosslink within a chain gets a gap past its longest chain when the builds would make several |
| `packing` | A lattice reference's beads over its sites, so the generator picks the same lattice. Without a lattice (a data file in sigma, a graph with no cell), a range of packings is built and scored |
| `min_dangling_dp` | Only when the beads next to the chain ends never crosslink where a uniform draw would have (above) |
| `seed` | `--seed` (the config's `crosslinking.seed`) |
| `chemistry.target_density` | The data file's bead density, or 0.85, flagged |

A crosslinked reference refuses `--sweep-cutoffs` on the crosslink route and `--junction-type` with `--crosslinked`, and an end-linked one refuses every crosslinked option.

**How the settings are chosen.** Every candidate setting is built through the pipeline's stages 1 to 3 (`Pipeline.run_graph_stages`) on `--seeds` seeds and scored by the mean of the descriptor composite, the deviations of lambda2 and of the mean path of the core from the reference's, the KS statistics of passes per chain and of the bridge, dangling and loop DPs, and the deviation of the loop count. The lowest mean wins. When the best two are closer than their seeds' spread, the best three are built on 16 seeds each and chosen between again, and a choice still inside the scatter is flagged. With everything read off the reference there is one candidate, and its builds are the report's first check.

**The lattice route** (`--route lattice`) writes the SC cell for the active sites, the exact P(f) with the loops and sol of the end-linked fit, a connectivity floor of 0.95, and `assignment.chains` with the chain length, the period and the chord floor. The cutoff candidates are the simple-cubic shells around cutoff ≈ p75(junction separation) / site spacing, one in and one out (the p95 rule of the end-linked references overshoots on these networks). It builds no nearest-neighbor control row, since on these P(f) the exact search takes minutes to sculpt one shell or gives up.

**The verification.** For a config that builds a crosslinked network, `topon generate CONFIG --verify REFERENCE` regenerates `--verify-seeds` seeds (from `crosslinking.seed`, or `generator.seed` on the lattice route) and reports the chains and crosslinks requested against built (or the P(f) on the lattice route), the determinism check, the composite, lambda2, the path and the four chain KS statistics, the counts (crosslinks, within a chain, loops, secondary loops, bridges, dangling, sol, junctions outside the largest piece) and the reach as built, in the reference's units. The generator puts a junction at its two beads' midpoint, which reads 3 to 6 % low against a reference that puts it on one of them. With `--verify-replicate` (repeated, networks of the reference's own process at other seeds) it gives two readings per measure. The scatter is the furthest a replicate sits from the reference, and each build is inside it or not. A build of the reference's own process falls outside with a chance of 1 in N + 1 for N replicates, so a measure is flagged only when more builds fall outside than that explains (binomial tail below 0.05). The other reading is a Mann-Whitney test of the builds against the replicates, reported raw and Holm-corrected over the measures, with the verdict taken on the corrected p. It needs 4 builds and 4 replicates at least and says so below that. Verify on seeds the fit did not choose its settings on (e.g., with `crosslinking.seed` set past the seeds of the fit). `--relaxed` is refused for these networks.

```bash
topon fit N100_seed1.gpickle --out n100_config.json
topon generate n100_config.json --verify N100_seed1.gpickle --verify-only \
    --verify-seeds 8 --verify-replicate N100_seed2.gpickle --verify-replicate N100_seed3.gpickle
topon fit resilin64_seed1.gpickle --sequence GGRPSDSYGAPGGGN --repeats 12
```

From Python, `topon.inverse.fit(path, ...)` returns the config, the report and the reference measurement, `topon.inverse.verify(config, reference, seeds=...)` the verification report, and `topon.inverse.measure` and `read_reference` the measurement alone.

### 3.7 `topon protein` (protein network from a sequence)

```bash
topon protein [--config CONFIG_PATH] [--sequence SEQ] [--repeats N] [--chains N] [--model charmm|martini]
              [--output DIR] [options]
```

The sequence comes from `--sequence` or from the config file. The command builds a crosslinked protein network from an amino-acid sequence, in CHARMM36m (all-atom) or Martini 3 (coarse-grained). It grows the chains as a lattice melt with one site per residue, crosslinks the crosslink residues that touch, builds the network in the chosen model and writes the data file, the coefficient includes, the groups and three relaxation scripts. It does not run LAMMPS. §4.3 lists every option and the output.

The default crosslinking changed in 0.4.0. Earlier versions laid each chain on the node lattice of the bond-fluctuation model (one node for every few residues) and crosslinked nodes on neighboring sites. The residue-level melt (`--crosslink-method melt`, now the default) has no lattice-parity rule and merges no residues onto one site, so the same command gives a different network, and its crosslink count can differ a good deal (e.g., the elastin-like example of §5.5 makes 11 crosslinks instead of 21). `--crosslink-method adjacent` (or `"crosslink_method": "adjacent"` in a config, as in the demos of `demos/protein/`) builds exactly what earlier versions built.

```bash
topon protein --sequence GGRPSDSYGAPGGGN --repeats 18 --chains 8 \
              --model martini --seed 42 --output runs/resilin_martini
topon protein --sequence GGRPSDSYGAPGGGN --repeats 12 --chains 8 \
              --model charmm --water-content 35 --seed 3 --output runs/resilin_charmm
topon protein --config demos/protein/charmm/config.json --output runs/charmm
```

`python -m topon.protein_network build ...` is the same command.

### 3.8 `topon track` (a page for atomistic relaxation runs)

```bash
topon track RUN_DIR... [-o PAGE] [--seeds N] [--no-energies] [--lmp LMP] [--omp N]
            [--title TITLE] [--label LABEL ...] [--z1-exe EXE] [--config CONFIG]
```

The command writes one self-contained HTML page (`relaxation_tracker.html` by default) for one or more study folders of the atomistic route. For every checkpoint it shows the network as Z1+ reads it, and through the stages it plots Z per bridge, the backbone passages, the energy under the full force field, temperature, density and the longest backbone bond. The energies need LAMMPS (`--lmp`, one zero-step run per checkpoint) and the Z columns need Z1+. Appendix A (*Watching a relaxation*) describes the page.

```bash
topon track output/pdms_dp30
topon track runs/z1 runs/z2 --label "Z = 1" --label "Z = 2" -o z.html --omp 4
```

### Global options

| Option | Description |
|---|---|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |
| `--no-shell` | With no sub-command, print the banner and exit instead of starting the interactive shell |

---

## 4. Sub-systems

### 4.1 simbox (molecule packing)

`topon.simbox` packs individual molecules into a periodic simulation box and writes LAMMPS input scripts for crosslinking studies. It is independent of the polymer-network pipeline.

The packing workflow has five steps.

```
MoleculeLibrary       →  Molecule objects (RDKit mol + reactive-site annotations)
       ↓
BoxPacker.pack()      →  PackedBox (placed molecules with 3D coordinates)
       ↓
assemble(packed)      →  AssembledSystem (merged RDKit mol, reactive-site registry)
       ↓
write_lammps(system)  →  system.data, settings.in, groups.txt, ff_coeffs.in
       ↓
write_inputs(system)  →  1_minimize.in, 2_nvt.in, 3_npt.in, 4b_crosslink.in
```

The quickest path is `topon simbox` (§3.4) or `topon.simbox.workflow.run_workflow()`.

`Molecule` (`topon/simbox/molecule.py`) is an RDKit Mol with explicit hydrogens, an ETKDGv3 + MMFF-optimized 3D conformer, and reactive sites detected by SMARTS.

| Site | SMARTS |
|---|---|
| `epoxide` | `[C]1[O][C]1` |
| `primary_amine` | `[NX3;H2;!$([NH2]C=O)]` |
| `secondary_amine` | `[NX3;H1]([#6])[#6]` |

```python
from topon.simbox.molecule import Molecule
mol = Molecule.from_smiles("EpoxyPDMS", "C1OC1COCCC[Si](C)(C)O...")
mol = Molecule.from_pdb("MyMol", "path/to/file.pdb")
mol = Molecule.from_mol("MyMol", rdkit_mol_object)
```

Every conformer simbox embeds (`from_smiles` and the library builders) goes through `embed_conformer`, which embeds from ETKDG's own starting coordinates and checks the result with `conformer_defects`. On a defect it discards the conformer and tries the next seed (42, 43, and so on, ten at most), and a molecule with no sound conformer raises `RuntimeError` naming the defects. A defect is a bond passing through a ring of the same molecule (rings of up to 12 atoms), which no minimizer can take back out, or a bond more than 15 % off its MMFF94 r0 (UFF's when MMFF lacks a parameter). Before 0.4.5 the AM0270 cage was embedded from random coordinates, which at seed 42 put one isooctyl Si-C bond through the opposite face of the cage, in every POSS box. `from_pdb` and `from_mol` keep the coordinates they are given, and `conformer_defects(mol.mol)` checks them (one readable line per defect).

`MoleculeLibrary` (`topon/simbox/library.py`) holds pre-built siloxane molecules.

```python
from topon.simbox.library import MoleculeLibrary
lib = MoleculeLibrary()
epoxy  = lib.epoxy_pdms(n_dms=2)    # Glycidoxypropyl-PDMS, C20H46O7Si4, 510.9 g/mol
amino  = lib.amino_pdms(n_dms=8)    # Aminopropyl-PDMS, C26H76N2O9Si10, 841.8 g/mol
poss   = lib.am0270_poss()           # AminopropylIsooctyl POSS, ~1267 g/mol
custom = lib.custom("C1OC1", name="MyEpoxide")
```

The library structures are as follows.
- Epoxy-PDMS is `Epoxide-CH₂-O-CH₂CH₂CH₂-Si(Me)₂-[O-Si(Me)₂]ₙ-O-Si(Me)₂-CH₂CH₂CH₂-O-CH₂-Epoxide` (77 atoms at n = 2).
- Amino-PDMS is `H₂N-CH₂CH₂CH₂-Si(Me)₂-[O-Si(Me)₂]ₙ-O-Si(Me)₂-CH₂CH₂CH₂-NH₂` (123 atoms at n = 8).
- Both chain ends carry two methyls since 0.4.5. Before, each terminal Si had one methyl and RDKit filled its fourth valence with an H.
- AM0270 POSS is a Si₈O₁₂ cube cage with `-CH₂CH₂CH₂-NH₂` on corner 0 and 2,4,4-trimethylpentyl (isooctyl, inert) on corners 1-7.

`BoxPacker` detects overlaps in O(N) with grid-based spatial hashing.

```python
from topon.simbox.packer import BoxPacker

packer = BoxPacker(
    density=0.85,        # g/cm³
    min_dist=2.0,        # Å
    seed=42,
    max_attempts=1000,   # placement attempts per molecule
    growth_factor=1.05,  # box expansion when packing fails
)
packed = packer.pack([(epoxy, 100), (amino, 50), (poss, 10)])
```

The packer computes the initial box from the total mass and the target density and shuffles the insertion order. Each molecule gets a random rotation (Shoemake quaternion) and a random translation, with an overlap check under the minimum image. If a molecule is not placed within `max_attempts`, the box grows by `growth_factor` and the packer retries that molecule (up to 20 rounds).

`AssembledSystem` is the merged Mol with global bookkeeping.

```python
from topon.simbox.system import assemble
system = assemble(packed)
# system.mol               merged RDKit Mol
# system.box_lengths       ndarray([Lx, Ly, Lz]) in Å
# system.molecule_ids      per-atom LAMMPS molecule ID (1-based)
# system.species_names     per-molecule species name
# system.reactive_sites    list of ReactiveSiteEntry (global atom index + group name)
```

The writers produce the data file and the input scripts.

```python
from topon.simbox.writer import write_lammps
from topon.simbox.inputs import write_inputs

files = write_lammps(system, output_dir="output/simbox")
files.update(write_inputs(system, output_dir="output/simbox", temperature=300.0, pressure=1.0))

# for an epoxy-amine box, the types its cure creates and the bond/react templates
from topon.simbox.workflow import prepare_bond_react
prepare_bond_react(system, "output/simbox", files)
```

`SimBox.write`, `write_lammps` and `write_inputs` write the box alone. `run_workflow`, `topon simbox` and `reactive_crosslink.run` also call `prepare_bond_react`, and only then does the box list the types its cure creates and carry its templates.

Stage 1 (soft push-off and minimization) has two phases. Phase A uses `pair_style soft` with a prefactor ramped from 0 to 60 and a brief NVT run to resolve overlaps. Phase B switches to the `lj/cut` DREIDING potentials and runs a conjugate-gradient minimization.

Impropers are DREIDING's inversion, `improper_style umbrella` in all three stage scripts. A planar center (sp2, three neighbors) carries three terms, each neighbor out of plane in one of them, each with a third of the parameter file's K, so the data file's Improper Coeffs read `K omega0`. Before 0.4.5 they were cvff `K -1 1`, one term of full K per center. The epoxy, amine and POSS molecules have no planar center.

`settings_x6.in` holds DREIDING's own van der Waals term, the exponential-6 of the parameter file, as `pair_style buck` coefficients (A = 6 D0 exp(zeta)/(zeta - 6), rho = R0/zeta, C = zeta D0 R0^6/(zeta - 6), with the minimum at R0 and depth D0). `buck` does not mix, so every pair of types is written, with R0 and D0 combined geometrically and zeta arithmetically, and the file ends with `pair_modify tail yes`. `topon.forcefield.dreiding.x6_buck_coefficients(type_i, type_j, params)` gives the coefficients for any pair. No stage script includes the file, and `settings.in` stays the LJ 12-6 substitute, whose r^-12 wall is much harder. Switch to it only on a dense state with no overlaps, since the -C/r^6 term of `buck` has only a finite wall in front of it and two overlapping atoms fuse. `system.data` holds LJ's two Pair Coeffs values, so read the data file under `lj/cut` first. Simbox writes every charge as 0 and its stage scripts run `lj/cut 12.0` with no Coulomb term, so the switch is to plain `buck` (with charges of your own, to `buck/coul/long 12.0` and a `kspace_style`).

```
pair_style      lj/cut 12.0
read_data       system.data
# ... relax and densify under lj/cut (the stage scripts) ...
pair_style      buck 12.0
include         settings_x6.in
```

Stage 4b is a crosslink template that the user must configure. Option A uses `fix bond/react` (template-based reactions with molecule pre and post files). Option B uses `fix bond/create` (simple distance-based bond formation).

`run_workflow` does all of this in one call and is the main entry point.

```python
from topon.simbox.workflow import run_workflow

files = run_workflow(
    output_dir="output/simbox_run",
    n_epoxy=600, n_amino=300, n_poss=0,
    density=0.85, seed=42,
)
```

**Type ids and `fix bond/react`.** Every type keeps the id DREIDING typing gives it, in order of first appearance, so the order follows the first molecule the packer places (N_3 is atom type 4 when that molecule holds an N and 5 when it does not), and every coefficient line names its types. A dihedral type is a name and a K, so the epoxide ring's torsions carry 0.125 and 0.5 beside the chain's 0.111111 and 0.333333. For a box holding epoxide and amine sites, `run_workflow` then calls `prepare_bond_react(system, output_dir, files)`. It appends the types the cure creates (the O-H bond, the C-O-H and C-N-C angles and the torsions about the new bonds) to `system.data` and `ff_coeffs.in` (`react_templates.add_reaction_types`), and writes the four `fix bond/react` molecule files and two maps beside them in the box's own ids (`react_templates.write_epoxy_amine_templates`), which `4b_crosslink.in` names. A box without both an epoxide and an amine is left as written. `topon.workflows.reactive_crosslink.run` does the same.

Before 0.4.5 `run_workflow` patched `topon.forcefield.dreiding` at write time with a `UniversalTypeMapper` that forced fixed ids so that hand-written templates fitted every composition. It matched dihedral types by atom types only, so the epoxide ring's torsions were written with the chain's K, and in some compositions the data file listed two K under one id, which LAMMPS refuses to read. The templates now come from each box's own types, so fixed ids have no use.

### 4.2 singlechain (solubility utility)

`topon.singlechain` builds a single polymer chain in a solvent box for solubility studies. Use the CLI (§3.5) or the Python entry point below.

```python
from topon.singlechain.workflow import run_workflow as chain_workflow

chain_workflow(
    output_dir="chain_output",
    chain_smiles="[Si](C)(C)O",
    dp=20,
    solvent_smiles="Cc1ccccc1",
    n_solvent=200,
    density=0.85,
    seed=42,
)
```

The chain's coordinates come from `embedder`, a Python-only argument (the CLI always takes the default). `embedder="linear"`, the default, lays the backbone out straight and puts branch atoms at offsets from their parents that can land them almost on other atoms, which stage 1's soft push-off has to undo. `embedder="etkdg"` calls `topon.chemistry.embed.embed_with_etkdg`, ETKDGv3 from random starting coordinates at `seed` followed by 200 MMFF94 iterations (UFF when MMFF cannot type the chain), with no check of the result. It is slower, and the time grows steeply with chain size (minutes for a few hundred heavy atoms).

### 4.3 Protein networks (`topon protein`)

`topon protein` (or `build_protein_network` in Python) builds a protein network from a one-letter sequence. The sequence is either a repeat block (with `--repeats`) or a whole chain (`--repeats 1`). The network is built in one of two models.

- `--model charmm` builds CHARMM36m all-atom chains from the bundled RTF and PRM files and applies each crosslink as its RTF patch (`DITY` for dityrosine, `DISU` for a disulfide). A hydrated build adds TIP3P water and NaCl, and `fix shake` holds the water rigid in the stage-3 MD. The CMAP grids are written from the parameter file. `topon/protein_network/charmm/data/README.md` lists the bundled files and their sources. Every term is looked up by `topon.forcefield.charmm`, and a missing one stops the build.
- `--model martini` builds Martini 3 chains with the Martini3-IDP bonded terms. The chain topology is the bundled polyply ITP for the resilin reference, and for any other sequence it is generated with `polyply gen_params -lib martini3` (polyply is the optional `martini` extra). Tryptophan is refused, because Martini 3 represents it with a virtual site, which LAMMPS lacks.

By default (`--crosslink-method melt`) every residue is a site of a self-avoiding melt on a cubic lattice, grown one chain after another at the target packing by `topon.topology.chain_crosslinking`. Every pair of crosslink residues that touch (face or edge neighbors, within `--contact-radius`) is then crosslinked in random order, one bond at a time, and the builders place every residue where the melt put it. The other methods use the node lattice of the bond-fluctuation model (`bfm.py`), where each chain is a self-avoiding walk of nodes that stand for a few residues each, Monte Carlo moves equilibrate the walks, and crosslink residues on neighboring nodes react. Either way the build keeps a snapshot at the gel point, where one cluster first holds every chain, and a few snapshots beyond it. Crosslinks are dityrosine (`--crosslink-residue Y`) or disulfide (`C`) bonds and nothing else.

On the node lattice the crosslink residue sets the layout. A block with one crosslink residue away from its ends puts one crosslink node on the lattice per block. Any other sequence (two sites per block, a site at a block end, a chain without repeats) gets one lattice node per crosslink residue, with the chain ends as the other anchors. The melt needs no layout, and it has no lattice-parity rule (with one crosslinkable node in two, the node lattice can never crosslink a chain to itself). The build is deterministic for a given `--seed` (lattice, Monte Carlo, placement, water and ions).

| Flag | Default | Meaning |
|---|---|---|
| `--sequence` | required | One-letter sequence (the block, or the whole chain) |
| `--repeats` | `1` | Copies of the sequence per chain |
| `--chains` | `8` | Number of chains |
| `--model` | `martini` | `charmm` (CHARMM36m all-atom) or `martini` (Martini 3) |
| `--crosslink-residue` | `Y` | `Y` (dityrosine) or `C` (disulfide) |
| `--crosslink-method` | `melt` | `melt` (a residue-level melt, crosslink residues joined where they touch), or on the node lattice `adjacent` (the default before 0.4.0), `winding_safe` (no crosslink that closes a cycle around the periodic box), `distance`, or `none` (an uncrosslinked melt, to crosslink during the simulation, e.g., with `fix bond/react`) |
| `--snapshot` | `gel_point` | Snapshot to build (`gel_point`, `post_gel_N` or an index). If it was not reached, the build stops and says how many clusters the chains form. |
| `--allow-no-gel` | off | Build the last snapshot instead of stopping when the one asked for was not reached. The summary then notes it. |
| `--seed` | `42` | Seed of every random step |
| `--output` | `protein_network` | Output directory |
| `--water-content` | `0` | Water as a weight percent of the system |
| `--salt-conc` | `0.15` | NaCl in the water (mol/L). Counter-ions are always added to neutralize the protein. |
| `--target-density` | `0.85` | Initial density used to size the box (g/cm³) |
| `--equil-steps` | `20000` | Monte Carlo steps on the node lattice (not used by `melt`) |
| `--target-packing` | `0.45` | Packing fraction of the lattice |
| `--segs-per-block`, `--residues-per-segment` | `2`, auto | Lattice steps per block, and residues per lattice step for the anchor layout (node lattice only) |
| `--n-extra-snapshots`, `--snapshot-delta-conv` | `2`, `0.05` | Snapshots kept past the gel point, and their spacing in conversion |
| `--min-intrachain-sep` | `2` | Smallest gap between two crosslink sites of one chain that may react |
| `--contact-radius` | `1.5` | `melt` only. Crosslink residues this close (in lattice units, about one residue step) may crosslink. 1.5 takes the face and edge neighbors and 1.8 adds the corners, and a larger radius makes more crosslinks and gels more often |
| `--no-physical-backbone` | off | CHARMM only. Place atoms by jitter instead of from the RTF internal coordinates. |
| `--xpro-cis-fraction` | `0` | CHARMM only. Fraction of X-Pro peptide bonds seeded cis. |
| `--charmm-files` | bundled | CHARMM only. RTF, PRM, stream and CMAP files to use instead of the bundled CHARMM36m. They are the whole force field, and nothing bundled is read with them. |
| `--martini-itp` | polyply | Martini only. A chain ITP to use instead of polyply's. |
| `--water-bead` | `W` | Martini only. Water bead type (`W`, `SW` or `TW`). |
| `--config` | none | JSON file with any of the settings (the keys of `ProteinNetworkSettings`). Flags override it. |
| `--quiet` | off | Print less |

The output folder holds the data file, the coefficient includes, `protein_network.in.groups`, `relaxation/protein_network_stage{1,2,3}.in`, the lattice snapshots (`protein_network_topology.json`) and `protein_network_summary.json`. The summary records the inputs, the snapshot used (with the number of clusters the chains form, the largest one and the lattice packing), the atom and term counts, the net charge and the files. CHARMM adds `charmm36m.cmap` and two more includes (`.in.settings.soft` for the soft stage and `.in.settings.lj` for the LJ ramp). Martini adds `protein_network_chain.itp`, the chain topology used.

Short chains gel less often than long ones. On the node lattice, with one crosslink site per repeat, 8 chains of 12 repeats gel at the default seed where 8 chains of 8 do not, and the lattice edge (at least 9 sites, odd) keeps small systems below the target packing.

On the resilin reference the melt makes about as many crosslinks as the node lattice and gels about as often, and it also closes loops within a chain. A Martini build from the melt starts with two to three times more bonds threaded through rings, which the soft stage has to clear (a CHARMM build starts with fewer).

The three stages run in order from `relaxation/`.

```bash
cd runs/resilin_charmm/relaxation
lmp -in protein_network_stage1.in    # soft overlap removal
lmp -in protein_network_stage2.in    # LJ epsilon ramp under nve/limit
lmp -in protein_network_stage3.in    # minimization and MD -> ../system_equilibrated.data
```

Bonds can be threaded through rings. The soft stage can leave a strand through a ring (a proline or tyrosine ring in CHARMM, a three-bead ring in Martini), and the full force field then holds the bond and the ring stretched. topon detects this but does not prevent it. The summary counts the bonds that already pass through a ring as built (`threaded_bonds_as_built`, most of which the soft stage clears). After the run, `check-bonds` lists the bonds longer than 1.25 r0 and those that pass through a ring, with atom names from the as-built file. It writes `system_equilibrated_bond_check.json` and adds the counts to the summary.

```bash
python -m topon.protein_network check-bonds runs/resilin_charmm/system_equilibrated.data
```

In the CHARMM resilin demo (8 chains of 12 repeats) 32 of 16,901 bonds stayed stretched, all at six places where a bond went through a ring.

The same build from Python is below.

```python
from topon.protein_network.network import build_protein_network

summary = build_protein_network(sequence="GGRPSDSYGAPGGGN", repeats=12, chains=8,
                                model="charmm", water_content=35.0, seed=3,
                                output_dir="runs/resilin_charmm")
print(summary["counts"]["n_atoms"], summary["counts"]["total_charge"])
```

The demos in `demos/protein/{charmm,martini}/` run this entry point on a `config.json`. Their configs set `"crosslink_method": "adjacent"`, so they build what they built before 0.4.0.

#### CHARMM36m

The terms come from the shared CHARMM reader `topon.forcefield.charmm`. It matches parameters as CHARMM does (exact types before `X` wildcards, and the four improper patterns in order), writes the 1-4 LJ columns and the NBFIX pairs (e.g., Arg-Asp `NC2`-`OC`), and sets each dihedral's 1-4 weight so that every 1-4 pair counts once (0.5 in aromatic rings, 0 in the proline ring and on extra Fourier terms). It keeps the peptide-plane impropers that name the neighboring residue and stops on any term the files lack instead of writing a default. The `fix cmap` grid file is written from the CMAP section of the PRM, so it is CHARMM36m too. During development, LAMMPS single-point energies of these files were compared term by term with OpenMM's CHARMM implementation reading the same RTF and PRM, and they agreed to 4e-10 relative or better in the bonded and CMAP terms and to 5e-6 in the nonbonded terms.

The styles are the CHARMM-GUI set for LAMMPS. Stage 3 runs `lj/charmmfsw/coul/long 10 12` with `pair_modify mix arithmetic`, PPPM, `dihedral_style charmmfsw`, `angle_style charmm` (Urey-Bradley), `improper_style harmonic`, `special_bonds charmm` and `fix cmap`. Stage 1 runs `pair_style soft` with the `.in.settings.soft` include, which sets the 1-4 weights to 0 and uses `dihedral_style charmm`, because LAMMPS accepts a 1-4 weight only with an `lj/charmm*` pair style. Stage 2 ramps epsilon with `fix adapt` under `lj/cut/coul/long` and the `.in.settings.lj` include, which keeps CHARMM's arithmetic mixing and NBFIX (the `lj/charmm*` styles do not support `fix adapt`), so stage 3 starts without a jump in the mixing rule.

`pair_style soft 1.0` alone would give stage 1 a ghost cutoff of 3 Å, shorter than the longest as-built bond (about 7 Å for a crosslink). A bond across the box edge or a domain boundary would then act on the wrong image of its partner, even on one rank, so stage 1 sets `comm_modify cutoff 14` as stage 2 does.

By default the atoms are placed from the internal-coordinate tables of the RTF (real bond lengths and angles, planar impropers, real rotamers and L chirality), and the backbone is coiled to about 3.8 Å between consecutive CA atoms. The stage scripts hold omega trans (apart from the `--xpro-cis-fraction` share of X-Pro bonds) and the CA chirality at L with `fix restrain`, and release both before the dynamics of each stage. With `--no-physical-backbone` every residue's atoms are dropped at its lattice site with a small random jitter instead. Minimization then leaves cis and trans to chance behind the omega barrier (about 12 % of the non-proline peptide bonds came out cis in a test, against under 0.1 % in real proteins) and about half of the CA atoms D.

A crosslink that closes a cycle around the periodic box cannot be made short by any image flags. The builder removes such a reaction before patching, so its two residues stay unpatched, and the summary counts them. Water and ions go on free grid sites at least 2.4 Å from the protein.

#### Martini 3

The port of the GROMACS force field to LAMMPS makes these approximations. None of them changes the topology.

1. GROMACS reaction-field electrostatics (`epsilon_r = 15`) becomes `pair_style lj/cut/coul/cut` with `dielectric 15.0`, which drops the reaction-field correction term.
2. The restricted-bending angle of the Martini 3 IDP backbone (GROMACS `funct=10`) becomes `angle_style cosine/squared`, which drops the `1/sin²θ` factor. This is adequate for disordered chains and should be reviewed for folded ones.
3. Each term of a multi-term proper dihedral (GROMACS `funct=9`) becomes its own `dihedral_style charmm` coefficient set on the same atoms, which is exact.
4. Constraints become very stiff harmonic bonds (`K = 1e6 kJ/mol/nm²`, the reference's `FLEXIBLE` setting).
5. The scripts use `special_bonds lj 0.0 1.0 1.0 coul 0.0 1.0 1.0`, which excludes bonded pairs only (the reference's `nrexcl=1`). The extra ring exclusions of the ITP are not written.

Virtual sites (and so Martini 3 tryptophan and folded-protein models with Go contacts) and elastic networks are not supported. The stage-3 script minimizes and runs a short NVT at 310 K. The bundled Martini files, their sources and the papers to cite are in `topon/protein_network/data/README.md`.

`python -m topon.protein_network` has the sub-commands `build` (the same as `topon protein`) and `check-bonds`. Its `generate`, `sweep` and `topology` sub-commands and `python -m topon.protein_network.charmm.build_systems` are older entry points that start from a repeat block or a lattice topology file. `topon protein` covers what they do.

### 4.4 CHARMM for polymer networks (`chemistry.force_field = "charmm"`)

The atomistic pipeline writes DREIDING by default, and its DREIDING output does not change with this option. With `force_field: "charmm"` it writes CHARMM instead, from RTF, PRM and stream files you supply. Every repeat unit, node molecule and bridge atom names the RTF residue it is. Its atoms take their types and charges from that residue, and every bonded and nonbonded term comes from the parameter files. Nothing is substituted. A residue, atom or parameter the files do not define stops the chemistry stage with the complete list, which is the list to fill (e.g., with parameters by analogy, as CGenFF does).

topon does not type new molecules. CGenFF atom typing needs the CGenFF program, so run it (or write the residue by hand) and pass its stream file.

```json
"chemistry": {
  "model_type": "atomistic",
  "force_field": "charmm",
  "charmm": {
    "files": ["bundled:top_all35_ethers.rtf", "bundled:par_all35_ethers.prm",
              "peg_junction.str"],
    "pair_style": "lj/charmmfsw/coul/long"
  },
  "node_type_map": {"A": {"molecule": "C", "charmm_residue": "PEJ"}},
  "edge_type_map": {"A": {"monomer": "PEG"}},
  "monomers": {
    "PEG": {"smiles": "COC", "chain_head": "C", "chain_tail": "C",
            "charmm_residue": "PEGM", "charmm_atom_names": ["C1", "O1", "C2"]}
  },
  "connection": {"auto_bridge": false}
}
```

The chain is the monomer SMILES repeated `dp` times, so each repeat unit is a known set of heavy atoms, and each node molecule and bridge atom is another set. With `charmm_atom_names` (the unit's heavy atoms in SMILES order) the names are taken as given and checked against the RTF (element, hydrogen count and bonds). Without it the unit is matched to the residue by graph isomorphism on element, hydrogen count and, for residues that link through `+`/`-` atoms, the number of bonds leaving the unit. If two matches would give an atom a different type or charge, topon asks for the names instead of choosing. Hydrogens follow their heavy atom. RTF impropers are kept, including those that name the previous or next unit of the chain. At a strand end such a `+`/`-` atom is looked up in the junction (or bridge) residue the unit is bonded to, and the build stops if that residue has no atom of that name. The atom charges of each residue must add up to the charge on its RESI line, and the charge of the whole network must be an integer.

Relative file paths are read from the config's folder, and `bundled:NAME` is `topon/chemistry/charmm/data/NAME`. That folder holds the C35r ether force field (`top_all35_ethers.rtf` and `par_all35_ethers.prm`, unchanged from the MacKerell lab's CHARMM36 release, MIT). Its `PEGM` residue is the PEG repeat unit. Cite Vorobyov et al., J. Chem. Theory Comput. 3, 1120 (2007) and Lee et al., Biophys. J. 95, 1590 (2008) when you use it.

`02_Chemistry/system.data` has the DREIDING layout, so the conformation stage is unchanged, with RTF charges and CHARMM types. Three includes sit beside it (`system.in.settings`, `.soft` and `.lj`). `04_Simulation/` holds the same three stages as the DREIDING route with CHARMM styles (`bond harmonic`, `angle charmm`, `dihedral charmmfsw` with 1-4 weights, `improper harmonic`, `special_bonds charmm`, and `lj/charmmfsw/coul/long 10 12` with arithmetic mixing and PPPM in stage 3). Stages 1 and 2 differ from stage 3 for the reasons given in §4.3. The run manifest records the files read, the net charge of every residue kind and how many terms matched a wildcard.

POSS nodes, grafts and sol chains are refused with `force_field: "charmm"`, and so are residues with NOANG/NODIH or CMAP lines and `model_type: "coarse_grained"`. RTF patches other than whole residues are not applied to polymer units. Every refusal, and every missing parameter, ends `topon generate` with a message that names what to add.

```bash
cd demos/polymer/atomistic/charmm_peg
topon generate config.json
```

---

## 5. Recipes

Each recipe gives a config and the command. Most settings live in the JSON config, and Appendix A has the full schema.

### 5.1 CG network with entanglements

```json
{
  "study": { "name": "cg_entangled", "output_dir": "./runs" },
  "topology": {
    "source": "generate",
    "generator": {
      "lattice_size": "6x6x6",
      "lattice_type": "SC",
      "max_functionality": 4,
      "degree_distribution": "0:13,1:25"
    }
  },
  "assignment": {
    "node_types": { "method": "degree", "degree": { "mapping": {"1": "end", "2": "A", "3": "A", "4": "A"} } },
    "edge_types": { "method": "uniform", "uniform": { "type": "A" } },
    "dp_distribution": { "default": { "mean": 25, "pdi": 1.0 } },
    "entanglements": {
      "enabled": true, "target": 5, "target_type": "count"
    }
  },
  "chemistry": {
    "model_type": "coarse_grained",
    "node_type_map": {
      "end": { "molecule": "Si", "is_end_cap": true },
      "A":   { "molecule": "Si", "is_end_cap": false }
    },
    "edge_type_map": { "A": { "monomer": "PDMS" } }
  }
}
```

```bash
topon generate cg_entangled.json --output ./runs
```

### 5.2 Atomistic network with POSS chain caps

POSS goes on the degree-1 chain ends. POSS at an internal junction is not supported, and `topon doctor` reports it.
The config loads the topology that `demos/topology/end_linking/python/run.py` writes, so run that script
first from the repository root.

```json
{
  "study": { "name": "atomistic_poss", "output_dir": "./runs" },
  "topology": { "source": "load",
    "existing_files": { "nodes_file": "demos/topology/end_linking/python/output/network.nodes",
                        "edges_file": "demos/topology/end_linking/python/output/network.edges" } },
  "assignment": {
    "node_types": { "method": "degree",
      "degree": { "mapping": {"1": "POSS", "2": "A", "3": "A", "4": "A"} } },
    "edge_types": { "method": "uniform", "uniform": { "type": "A" } },
    "dp_distribution": { "default": { "mean": 10, "pdi": 1.0 } }
  },
  "chemistry": {
    "model_type": "atomistic", "target_density": 1.1,
    "node_type_map": {
      "A":    { "molecule": "Si",          "is_end_cap": false },
      "POSS": { "molecule": "POSS_AM0270", "is_end_cap": true }
    },
    "edge_type_map": { "A": { "monomer": "PDMS" } }
  }
}
```

```bash
topon generate atomistic_poss.json --output ./runs
```

`POSS_AM0270` has one attachment atom, its propyl arm's end carbon, which every strand of a junction would share, so POSS stays on the chain ends (see `chemistry.node_type_map` in Appendix A for what the cap is). Since 0.4.5 each cage is placed whole by the settled placement. `demos/poss/` is this recipe with a generated topology.

### 5.3 Atomistic network with grafts and entanglements

Add a `grafts` block to `assignment` in the entanglement recipe (5.1).

```json
"grafts": {
  "enabled": true,
  "per_edge_type": {
    "A": { "graft_density": 0.05, "side_chain_monomer": "PDMS", "side_chain_dp": 5 }
  }
}
```

A complete config with combined features ships as `demos/polymer/atomistic/combined/config.json` (and `demos/polymer/coarse_grained/combined/config.json` for CG).

### 5.4 simbox crosslink workflow

```bash
topon simbox --output runs/simbox_crosslink \
             --n-epoxy 600 --n-amino 300 --n-poss 0 --seed 42
```

```bash
cd runs/simbox_crosslink
lmp -in 1_minimize.in
lmp -in 2_nvt.in
lmp -in 3_npt.in
# Edit 4b_crosslink.in to choose Option A (fix bond/react, whose templates are
# written beside system.data) or B (fix bond/create), then:
lmp -in 4b_crosslink.in
```

### 5.5 Protein network from any sequence, both models

```bash
# An elastin-like block with Lys, Glu and two Tyr per block (Martini 3 via polyply)
topon protein --sequence GVGVPGKGVPGYGVPGEGYG --repeats 10 --chains 8 \
              --model martini --water-content 30 --seed 7 --output runs/elp_martini

# The same network, CHARMM36m all-atom
topon protein --sequence GVGVPGKGVPGYGVPGEGYG --repeats 10 --chains 8 \
              --model charmm --water-content 30 --seed 7 --output runs/elp_charmm

# Cys-Cys disulfide crosslinks instead of dityrosine
topon protein --sequence GSGCGAPGSG --repeats 12 --chains 8 \
              --model charmm --crosslink-residue C --seed 42 --output runs/disulfide
```

The Martini build of a sequence other than the resilin reference needs polyply (the `martini` extra).

### 5.6 PEG network with CHARMM parameters

```bash
cd demos/polymer/atomistic/charmm_peg
topon generate config.json          # C35r PEG strands on a diamond lattice
cd output_charmm_peg/charmm_peg/04_Simulation
lmp -in minimize_1_serial.in
lmp -in minimize_2_parallel.in
lmp -in minimize_3_parallel.in
```

Remove `peg_junction.str` from `chemistry.charmm.files` to see the list of what C35r does not define at the junction.

---

## 6. Python API (alternative to CLI)

The CLI is a thin wrapper around the Python API. The equivalent Python calls are below.

```python
# topon generate equivalent: full pipeline
from topon.config import load_config_full
from topon.pipeline import Pipeline

config, raw = load_config_full("demos/templates/full.json")
pipe = Pipeline(config, raw_config=raw)   # raw carries simulation / execution / experimental
pipe.run()

# topon validate equivalent
from topon.config import load_config, validate_config
errors = validate_config(load_config("config.json"))

# topon fit and topon generate --verify equivalents
from topon.inverse import fit, verify, write_fit
result = fit("ref.data", seeds=2)            # .config, .report, .measurement
write_fit(result, "ref_config.json")         # and ref_config.fit.json beside it
report = verify(result.config, "ref.data", seeds=[1, 2, 3],
                measured=result.measurement)
report = verify(result.config, "ref.data", seeds=[1],     # and a relaxed build
                relaxed="runs/fit/04_Simulation")         # report["relaxed"]

# a network crosslinked along its chains, without the pipeline
from topon.topology.chain_crosslinking import ChainType, crosslink_chains
melt = crosslink_chains([ChainType.every(200, 100, every=3)], crosslinks=600, seed=1)
G = melt.graph                    # strand graph with dp, chain and chain_index per strand
melt.positions, melt.crosslinks   # the melt, and the crosslinks in the order made
G_half = melt.graph_at(300)       # the network after its first 300 crosslinks
resilin = ChainType.from_sequence("GGRPSDSYGAPGGGN", count=8, repeats=12)

# topon simbox equivalent
from topon.simbox.workflow import run_workflow
run_workflow("simbox_output", n_epoxy=600, n_amino=300, n_poss=0,
             density=0.85, seed=42)

# topon chain equivalent
from topon.singlechain.workflow import run_workflow as chain_workflow
chain_workflow("chain_output", chain_smiles="[Si](C)(C)O", dp=20,
               solvent_smiles="Cc1ccccc1", n_solvent=200,
               density=0.85, seed=42)

# topon protein equivalent
from topon.protein_network.network import build_protein_network
build_protein_network(sequence="GGRPSDSYGAPGGGN", repeats=12, chains=8,
                      model="charmm", seed=3, output_dir="runs/resilin_charmm")
```

To call individual stages, see ARCHITECTURE.md §2, where each stage names the module that drives it.

---

## 7. Demo scripts

`demos/` holds runnable Python scripts that drive topon without the CLI. They are not part of the package API. To use one, copy it, change the settings at the top and run it with `python <path>`. [`demos/README.md`](../demos/README.md) describes the whole folder.

| Script | Purpose |
|---|---|
| `demos/run_via_api.py` | Runs a demo config through `topon.pipeline.Pipeline` directly, as a starting point for scripting the pipeline |
| `demos/topology/end_linking/python/run.py` | Generates a 6x6x6 SC topology with the pure-Python generator |
| `demos/topology/end_linking/c/run.py` | The same topology through the compiled C generator (set `TOPON_GENERATOR_EXE` to the binary) |
| `demos/workflows/batch_polymer_topology/run.py` | Generates 25 seeded lattice graphs, exports each as `.nodes`/`.edges`, GraphML and NPZ, and writes one CSV of per-graph properties. A second run resumes (§8) |
| `demos/protein/charmm/run.py` | Builds the CHARMM36m resilin network of its `config.json`, dry and at 35 wt% water |
| `demos/protein/martini/run.py` | Builds the Martini 3 resilin network of its `config.json` |

---

## 8. Windows, LAMMPS and long runs

### LAMMPS on Windows runs OpenMP only

The Windows LAMMPS installer (`LAMMPS 64-bit <version>`) is a serial build with the OPENMP package. It ships `libgomp` and no MPI library, so run it threaded, with the suffix and the thread count on the command line.

```powershell
lmp -sf omp -pk omp 8 -in minimize_2_parallel.in
```

`mpiexec -np 8 lmp -in ...` does not run in parallel. It starts eight independent serial copies of the whole input, each writing the same log and data files, which looks like a slow run with scrambled output. The same goes for `use_mpi=True` or `n_procs` in `topon.simulation` (`SimulationRunner`, `protocols.StagedRun`), which prepend `mpirun -np` and are meant for a cluster build. On Windows use `StagedRun(..., omp=8)`, which adds `-sf omp -pk omp 8` and sets `OMP_NUM_THREADS`. Under `-sf omp` a quartic bond needs the `suffix off` and `suffix on` wrapper that the generated stage-6 script already carries (see *Deformation runs* in Appendix A), because `bond_style quartic/omp` leaves the subtracted LJ term out of the energy and the pressure.

### Z1+ runs under WSL

Z1+ is a Linux binary, so on Windows topon runs it inside WSL (as `wsl.exe -e bash -lc "~/z1/Z1+ ..."`). Three things follow.

- The `.Z1` file and any scratch directory must be on a drive WSL can see. Fixed drives are mounted under `/mnt/<letter>`, and `wsl.exe -e wslpath -a <path>` translates a Windows path. A network share, a removable drive or a path WSL has not mounted comes back empty, and the run fails with "could not map path into WSL". Copy the file to a local fixed drive first.
- Z1+ writes its output (`Z1+summary.dat`, `Z_values.dat` and others) into its working directory, so every concurrent run needs its own directory, or the second overwrites the first. `topon.analysis.z1plus` makes one per call.
- Text coming back from `wsl.exe` can carry NUL bytes (UTF-16 from the launcher), so strip `\x00` before parsing it.

### One writer per output directory

Two processes writing one run directory overwrite each other's stage files and manifest, and two sweep drivers sharing an output folder rerun each other's cases. Give every concurrent run its own study name or `output_dir`, and every concurrent C generator run its own `--output-dir`.

The run manifest records the writer. When a pipeline run starts, it writes `pid`, `host` (a short hash of the machine name, so a shared manifest does not carry the name) and `started` under `run` in `manifest.json`. If the entry already there names another process on this machine that is still running, the run prints a warning with that pid and how to stop it. It warns rather than stops, since Windows reuses process ids and a false alarm must not block a build.

### Resuming a sweep

A sweep driver should record every case it finishes, a failure as much as a success, and skip both when it is started again. With a fixed seed a failed case fails the same way every time, so rerunning it only costs time, and two copies of a driver that each rerun what they see as unfinished end up building everything twice. The batch workflow (`demos/workflows/batch_polymer_topology/run.py`) works this way. Each graph leaves `record.json` (outcome, seed, pid), a rerun builds only the graphs without one, `--retry-failed` builds the failures again (e.g., after raising the trial budget), and `summary.csv` is rebuilt from the records. The script also holds its output folder in `output/.writer.json` while it runs and refuses to start while another live process holds it.

```bash
python demos/workflows/batch_polymer_topology/run.py                 # resumes
python demos/workflows/batch_polymer_topology/run.py --retry-failed  # and redoes failures
```

### Listing and stopping topon processes on Windows

`wmic` is deprecated, and grepping its output (or that of `tasklist`) from Git Bash matches nothing, because it is UTF-16. PowerShell returns objects, so filter those instead.

```powershell
# every python, generator and LAMMPS process, with its command line
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^(python|generator|lmp)' } |
  Select-Object ProcessId, CreationDate, CommandLine | Format-List

# anything whose command line mentions topon (topon generate, a script
# under a topon folder, and the shells that started them)
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'topon' } |
  Select-Object ProcessId, Name, CreationDate

Stop-Process -Id 12345          # the pid from the list, or from manifest.json
```

The Microsoft Store Python runs as `python3.12.exe`, which the `^python` pattern covers. `Get-Process -Id <pid>` shows the one process a manifest names.

---

## Appendix A. JSON config schema

`topon` is configured by a single JSON file. Use `topon init` to generate a starter and edit it as needed.

The top-level sections are listed below.

```json
{
  "study":        { ... },
  "topology":     { ... },
  "assignment":   { ... },
  "chemistry":    { ... },
  "conformation": { ... },
  "output":       { ... },
  "analysis":     { ... }
}
```

`conformation` is part of the validated schema. `load_config_full` also
returns a copy of it in the raw-extras dict, where `Pipeline`,
`topon.workflows.cg_network` and `topon.workflows.atomistic_network` read
it. `simulation`, `execution` and `experimental` are raw-only.

### `study`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | `"my_network"` | Study name, used as the sub-directory under `output_dir` |
| `output_dir` | string | `"./output"` | Root output directory |

### `topology`

| Key | Type | Default | Description |
|---|---|---|---|
| `source` | `"generate"` \| `"load"` \| `"crosslink"` | `"load"` | Generate a new topology, load an existing one, or grow a melt and crosslink it along its chains |
| `generator` | object | - | Settings for the C / Python generator (when `source="generate"`) |
| `existing_files` | object | - | File paths (when `source="load"`) |
| `crosslinking` | object | - | Chains, target and melt (when `source="crosslink"`, see below) |

#### `topology.generator`

| Key | Type | Default | Description |
|---|---|---|---|
| `exe_path` | string \| null | `null` | Path to `generator.exe`. `null` uses the Python generator |
| `lattice_size` | string | `"6x6x6"` | Lattice dimensions, e.g. `"8x8x8"` |
| `lattice_type` | `"SC"` \| `"BCC"` \| `"FCC"` \| `"Diamond"` \| `"MIX"` | `"SC"` | Lattice type. `MIX` overlays SC/BCC/FCC (see below) |
| `mix_fractions` | object | `{"SC":1,"BCC":0,"FCC":0}` | Sublattice fractions for `MIX`, which must sum to 1 |
| `neighbour_cutoff` | float | `1.0` | Candidate-edge range for every lattice type, in cell units. `1.0` is the canonical nearest-neighbor lattice, and larger values admit further shells (see below) |
| `neighbour_shells` | int \| null | `null` | SC only. Number of neighbor shells, an alternative to `neighbour_cutoff` (`2` -> 1.42, `3` -> 1.74, `4` -> 2.01, `5` -> 2.24, `6` -> 2.45, `8` -> 3.01) |
| `mix_cutoff` | float | - | **Deprecated** alias of `neighbour_cutoff`. It loads with a warning (at its default 1.0 it yields to `neighbour_shells`) |
| `periodicity` | string | `"111"` | Periodicity per axis (`1`=periodic, `0`=open), see below |
| `max_functionality` | int | `6` | Maximum crosslink degree per node |
| `max_trials` | int | `1000000` | Trials before giving up (strict search only) |
| `max_saves` | int | `1` | Number of networks to save |
| `degree_distribution` | string | `"0:0,1:0"` | Target degree distribution |
| `search` | `"strict"` \| `"exact"` \| null | `null` | Which sculptor runs. `null` picks `exact` when the degree distribution pins every degree and `strict` otherwise (see below) |
| `min_giant_fraction` | float | `0.99` | Exact search only. Smallest allowed fraction of active sites in the largest component |
| `odd_walks` | bool \| null | `null` (on) | Exact search only. Close attempts that end short on a scaffold with odd cycles by a second walk search (see below). `false` gives the graphs of earlier versions |
| `seed` | int \| null | `null` | Pins the generated graph (0 to 2^32 - 1). `null` draws from the global random streams (see below) |
| `architecture` | `"end_linked"` \| `"random_crosslinked"` | `"end_linked"` | What an edge is. With `random_crosslinked` the junctions are crosslinks along longer chains, degree-1 sites are chain ends, and stage 3 covers the strands with chains of `assignment.chains.dp` beads (see below). Junction degrees above 2 must then be even |

**`seed` pins stage 1.** With `"seed": 42` the Python generator draws
from its own streams seeded with 42, so two runs of the config build the
same graph with the same edge order, and the global streams are left
alone. The draws are the ones that seeding the global streams by hand
just before generating gave, so a graph pinned that way is the graph
`"seed": 42` builds.

```python
import random, numpy as np
random.seed(42); np.random.seed(42)          # what "seed": 42 replaces
```

On the C route (`exe_path` set) the binary gets a `TOPON_SEED` drawn from
a stream seeded with the same number, and the strict C search is seeded
too (without a seed it takes its seed from the clock). The seed covers
only the topology. Where a config uses them, the DP draws (a PDI above 1),
random edge and node types, copolymer sequences, entanglements and graft
positions still draw from the global streams, and `assignment.defects.seed`
pins the defects, so such a config still needs the global streams seeded
for a byte-identical LAMMPS file, which
`topon.workflows.cg_network.run(seed=...)` does. The conformation noise
needs no seed. It comes from a stream keyed on the study name, so the same
study writes the same `system_relaxed.data` whenever everything before it
is pinned. Since 0.4.5 the atomistic placement (`straight`, `walk`,
`meander`, `coil` and the side-chain offsets of the historic placement)
draws from streams keyed on the study name too. An atomistic build on the
default route that uses none of the global draws above, with
`generator.seed` set (and `defects.seed` with defects), therefore writes
the same files at every stage on every run with no global stream seeded,
and leaves both global streams where it found them. The same config under
the same study name is the same build, whatever `--output` directory it is
written to, so replicates need study names of their own. The run manifest
records the seed as `generator_seed`.

`topology.generator` refuses unknown keys, as do the per-class blocks under
`assignment.defects`, the whole `conformation` section and `analysis`.
Other sections drop an unrecognized key without notice. The refusal matters
because `"seed": 7` under `topology.generator` validated and did nothing
before `seed` was a field, so a config that looked pinned was not. In the
other sections, `topon doctor` reports ignored keys under the
`unknown_config_keys` rule instead of refusing to load, because a config may
carry parameters for other tools under `topology`.

In the degree distribution, `"d:N"` requires N nodes of degree d, `"e:N"` requires N edges in total, and omitted degrees are unconstrained (e.g., `"0:15,1:30,e:371"`).

**The degree sum must be even.** Every edge adds 2 to it, so a request
that names every degree from 0 to `max_functionality` with an odd sum
describes no graph, and both searches refuse it before the first trial.
A partial request such as `"0:13,1:25"` is not checked in Python, because
the degrees it leaves out take up the parity. The strict search of the C
binary refuses any odd sum over the named degrees, so that request runs in
Python but not with `exe_path` set.

**Degree counts for another cell.** Absolute counts belong to one site
count. To carry a P(f) to a different lattice (a smaller cell, or a `MIX`
draw), rescale it.

```python
from topon.topology.degree_matching import (
    rescale_degree_counts, format_degree_distribution)

ref = {0: 8, 1: 217, 2: 356, 3: 153, 4: 1975}      # 2709 sites
counts = rescale_degree_counts(ref, n_sites=216, vacancy_fraction=0.02)
format_degree_distribution(counts)                 # "0:4,1:18,2:27,3:12,4:155"
```

The vacancies come first (`round(vacancy_fraction * n_sites)`, or the
reference's own share when the fraction is left out), the active sites are
shared out by the largest-remainder rule, and an odd degree sum is fixed by
moving one site to the adjacent degree that keeps the counts closest to the
exact shares. The result sums to `n_sites`, so it is a full
`degree_distribution` for either search.

##### Which sculptor (`search`)

The candidate-edge lattice is a superset, and a sculptor picks the network
from it. The two sculptors suit different targets.

| | `strict` | `exact` |
|---|---|---|
| how | removes candidate edges one at a time until the graph matches | assigns the requested degrees to sites, then completes the degree sequence with augmenting paths |
| needs | any subset of targets (`d:N`, `e:N`, or nothing) | a count for every degree from 0 to `max_functionality` |
| samples | `max_trials` random trials | up to 6 seeds per network, then 6 fallback attempts when all six ended short. In C (`exe_path` set) an explicit `max_trials` bounds the attempts instead |
| move history | yes, `G.graph["move_history"]` | no |
| good at | loose targets, an edge budget, a ceiling well above the mean degree | near-complete tetrafunctional targets, dangling-end sites, high vacancy fractions |
| bad at | exactly those (see below) | a scaffold with no spare candidate edge (Diamond at `max_functionality: 4`) |

If `search` is unset, it resolves to `exact` when the degree distribution
names every degree from 0 to `max_functionality` and to `strict` otherwise.
An explicit value always takes precedence. A fully specified degree
distribution already fixes the edge count, so an `e:N` term next to one is
accepted when it agrees and refused when it does not.

The strict sculptor cannot reach a P(f) that is 73 % tetrafunctional with
8 % dangling ends on any scaffold, in Python or C, within 3 to 10 minutes.
It prunes sites to the ceiling greedily and overshoots the edge budget, and
when a tenth of the sites must be chain ends, 99.7 % of its trials fail.
The exact search reaches such targets in 0.1 to 2 seconds, as the examples
below show.

| target | scaffold | result |
|---|---|---|
| `0:43,1:217,2:356,3:153,4:1975` (DP 20) | SC 14³ | exact, giant 1.000, 0.5 s |
| the same | 90/5/5 MIX 14³ at cutoff 2.01 | exact, giant 1.000, 0.1 s |
| `0:157,1:73,2:33,3:59,4:407` (DP 100, 21 % vacancies) | SC 9³ | exact, giant 0.995 to 1.000, 0.2 s |
| `0:10,1:0,2:26,3:75,4:43,5:53,6:9` (a network of the paper dataset in `demos/npjcompmat/data/mechanics/`) | SC 6³ | exact, 0.03 s (strict also reaches it) |

An example generator block for the DP-20 target is below.

```json
"generator": { "lattice_type": "SC", "lattice_size": "14x14x14",
               "neighbour_cutoff": 1.74, "max_functionality": 4,
               "degree_distribution": "0:43,1:217,2:356,3:153,4:1975" }
```

The exact search has the following properties.

- The degree sum must be even. Every edge contributes 2, so no graph has an
  odd degree sum, and the search refuses such a request at once. Moving one
  site between two odd degrees fixes it.
- Degree 0 is what is left over. The active sites are placed first and the
  remaining scaffold sites become vacancies, so `0:43` on a 2744-site cell
  follows from the cell size and is not a separate constraint. On a larger
  cell the active counts are still exact and the run prints how many sites
  were left empty. Vacancies are dropped before chemistry.
- A dangling end always attaches to a junction. No edge joins two degree-1
  sites, since that would be a free chain outside the network.
- A sculpting animation needs `strict`. The exact search keeps no
  edge-removal history, so its graphs carry no `move_history` for an
  animation to replay. Set `search: "strict"` when an animation is needed.
- A scaffold with no spare edges fails with a message. On Diamond at
  `max_functionality: 4` a junction at the ceiling must bond to all four
  neighbors, so with 73 % of the sites at the ceiling every vacancy and
  dangling end removes capacity that cannot be recovered. When most of a
  request sits at the scaffold's own coordination like this, every seed
  gives the same forced assignment. The run then stops after one attempt
  and names the coordination, the residual and the remedy. A target with
  only a few sites at that coordination keeps its retries, and a failure to
  connect always retries. On Diamond the remedy is `neighbour_cutoff: 0.71`,
  which admits the second shell (z = 16) and reaches the same target in
  0.01 s. The default 1.0 means the canonical lattice and is not a range,
  so here the wider setting is the smaller number.
- A target that the random deal cannot place gets a fallback. SC, BCC and
  Diamond at their first shell are bipartite (every edge adds one degree to
  each sublattice), so the targets on the two sublattices must have equal
  sums, and a random deal rarely balances them. A target with many dangling
  ends and many sites at the lattice's own coordination (e.g., 44 and 54 of
  216 sites on SC) then ends short on every attempt. After 6 attempts in a
  row end with unfilled degree units, the search deals the targets balanced
  across the sublattices. It then repairs within one sublattice by moving
  demand out of the region a failed augmenting search reached, and keeps a
  move only if it loses no edge. Connectivity failures do not count toward
  the switch. The attempts before the switch are unchanged, so a target the
  random deal reaches gives the same graph for the same seed.
- Odd walks close what the marked walk misses. The augmenting walk marks a
  site the first time it reaches it and never returns to its start. On a
  scaffold with odd cycles (SC or BCC beyond the first shell, FCC, `MIX`)
  that loses walks that exist, and on a small dense cell many attempts end
  short (about half of them on SC 4x4x4 at three shells, each then retried).
  When an attempt's repair ends short on such a scaffold, the odd walks
  search again over (site, next move) states, draw no random number, and
  close what they find, so the attempt lands. An attempt that reached its
  target is not affected, but a seed whose earlier attempt ended short now
  lands on that attempt, with a different graph than earlier versions gave.
  `odd_walks: false` (C `--odd-walks=off`) gives the earlier graph.

Both searches exist in both generators. With `exe_path` set, an exact
request runs the C port in `topon/topology/csrc/` (`--search=exact`,
passed by `run_generator`), with the same steps, constants and refusals,
and with `min_giant_fraction` passed through. The C binary treats one trial
as one attempt. At its default, `max_trials` becomes the Python budget of 6
random and 6 fallback attempts per network, so an infeasible request gives
up within seconds on either route. Set `max_trials` explicitly to let the C
search retry longer. The pipeline passes the binary a seed drawn from the
global NumPy stream, so `np.random.seed(n)` fixes the C route as it fixes
the Python one. The manifest records this seed with the requested and
achieved counts. The one exception is an exact request that forces double
edges or reserves capacity for triangles and four-cycles (the
`assignment.defects` keys below). Only the Python search accepts those, so
the pipeline keeps such a request on Python and reports this.

**Random-crosslinked networks** (`architecture: "random_crosslinked"`). In a
network crosslinked along its chains (vulcanized, randomly crosslinked, or a
protein network) a junction is a crosslink point that chains pass through,
not a crosslinker at chain ends. A crosslink between two chain beads is a
junction of degree 4 (two chains passing), a junction of degree 2 carries a
primary loop (a crosslink within one chain with nothing between its beads),
and every chain end is a degree-1 site. The sculpt is the same exact search
on the same shell lattice, so the `degree_distribution` holds the crosslinks
at 4 (and 2 for the loop carriers) and the chain ends at 1, and
`assignment.defects` adds the loops. What changes is stage 3, which covers
the strands with chains (`topon.assignment.chains`, set in
`assignment.chains`). The strands meeting at each junction are paired at
random, each pair a chain passing through, and following the pairs from a
chain end leads to another chain end. Rings left over are merged into
chains, tails are exchanged at junctions until the number of junctions per
chain follows the binomial distribution of random crosslinking, and each
chain's crosslinked beads are drawn uniformly from its reactive beads, so
the strands get the beads between them. Every edge then carries `dp`,
`chain` and `chain_index`, and `G.graph["chains"]` lists each chain's
strands and junctions in order. The coarse-grained builder still builds a
junction as one bead the chains share (as the bond-fluctuation model merges
two crosslinked beads onto one site) and each strand as its own molecule.
The chain record is in the graph and in the run manifest (`chains`). A
strand drawn this way can be shorter in contour than its lattice chord,
since the lattice has no two junctions closer than one site spacing, and
`assignment.chains.chord_floor` is there for that. To grow the chains and
crosslink them where they touch instead, use `topology.source: "crosslink"`
(see `topology.crosslinking` below).

`PythonTopologyGenerator.generate` takes a `double_pairs={(a, b): count}`
argument (exact search only). It places parallel edges between sites of
target degree `a` and `b` before the fill and holds them fixed, so
secondary loops land with the right endpoint degrees. For example,
`{(4,4):115,(3,4):6,(2,4):7,(2,3):1}` gives 129 parallel pairs at exactly
the target P(f). topon's own injector (`inject_secondary_loops` in
`assignment/defects.py`) picks eligible pairs at random on an
already-sculpted graph and shifts P(f) (79 too few f = 2 and 134 too many
f = 3 for the same DP-20 target). The result is a `MultiGraph`, and the
second edge of each pair carries `is_secondary_loop=True`.

##### Diamond (`lattice_type: "Diamond"`)

Diamond is two interpenetrating FCC sublattices offset by ¼ along the body
diagonal. It has 8 sites per cubic cell, and every site is exactly
4-coordinated. A `max_functionality: 4` network therefore needs no pruning,
which makes Diamond a clean backbone for a tetrafunctional network and much
faster to generate than SC or FCC sculpted down to 4.

```json
"generator": { "lattice_type": "Diamond", "lattice_size": "6x6x6",
               "max_functionality": 4, "degree_distribution": "" }
```

An `NxNxN` Diamond has `8N³` sites and `16N³` bonds at a nearest-neighbor
distance of `√3/4 ≈ 0.433` cells. Both generators build it identically.

##### Neighbor shells (`neighbour_cutoff`)

Every lattice type takes `neighbour_cutoff`, the candidate-edge range in
cell units (i.e., the simple-cubic site spacing). At the default `1.0` each
pure lattice is its canonical nearest-neighbor graph (6 neighbors on SC, 8
on BCC, 12 on FCC, 4 on Diamond). Any other value connects every pair of
sites within that distance under the minimum image, as `MIX` does. The
sculptor then picks the final network from this larger candidate set, so
crosslinkers several site spacings apart can end up bonded, as they are in
a reaction-generated (end-linked) network.

```json
"generator": { "lattice_type": "SC", "lattice_size": "14x14x14",
               "neighbour_cutoff": 1.74, "max_functionality": 4 }
```

On SC the shells sit at `sqrt(n)` cell units for n = 1, 2, 3, 4, 5, 6, 8,
9, ..., so `neighbour_shells` can set the range instead of the cutoff.

| `neighbour_shells` | `neighbour_cutoff` | neighbors per SC site |
|---|---|---|
| 1 | 1.0 (default) | 6 |
| 2 | 1.42 | 18 |
| 3 | 1.74 | 26 |
| 4 | 2.01 | 32 |
| 5 | 2.24 | 56 |
| 6 | 2.45 | 80 |
| 8 | 3.01 | 122 |

`neighbour_shells` is SC-only. On BCC, FCC, Diamond and MIX it is accepted
only together with an explicit `neighbour_cutoff`, which then takes
precedence. If both are given on SC, the cutoff must admit exactly that many
shells. The graph records the shells a lattice actually produced as
`G.graph["shell_distances"]` (the sorted distinct edge lengths), next to
`G.graph["neighbour_cutoff"]`.

To choose a cutoff, take about the 95th percentile of the strand
junction-to-junction separation divided by the site spacing. This rule
reproduced the connectivity of two reference networks end-linked in LAMMPS
(`fix bond/create`) with DP-20 and DP-100 strands.

| reference | site spacing | 95th-percentile reach | SC shells that match | z | composite (SC) | best cell in the sweep |
|---|---|---|---|---|---|---|
| DP 20 strands, bead density 0.31 | 4.95 sigma (14^3 cell) | 2.1 spacings | 3 to 4 (cutoff 1.74 to 2.01) | 26 to 32 | 0.13 | 0.09, a 90/5/5 mixture at 4 shells (z 39) |
| DP 100 strands, bead density 0.31 | 7.7 sigma (9^3 cell) | 3.3 spacings | 8 (cutoff 3.01); 6 (2.45) is close | 122; 80 | 0.10; 0.15 | 0.08, BCC sites at 5 shells (z 64) |

At cutoff 1.0, every nearest-neighbor scaffold (SC, BCC, FCC and their
mixtures) is far from the same references (composite 0.4 to 1.05).
Bipartite lattices have no odd cycles, and mixtures carry too many triangles
and four-cycles. Adding shells spreads each site's candidate partners over a
larger neighborhood, which lowers clustering and lengthens cycles.

Three cautions apply.

- Keep the cutoff under a third of every periodic axis (`lattice_size` of
  at least `3 * cutoff` cells). Beyond that, three candidate edges can
  close a cycle around the box, so the candidate set has triangles that
  come from the box and not from the lattice. Beyond half the box, a pair
  can be within range through two images at once, which a simple graph
  cannot represent. Both `topon doctor` (`neighbour_cutoff_vs_box`) and the
  generator warn about this.
- Edge lengths vary. DP is assigned independently of edge length, so at
  three shells the same DP is built over lengths from 1.0 to 1.73 site
  spacings. Check for FENE strain on the long edges.
- The neighbor search compares every pair up to 4 000 sites and uses a cell
  list above that (e.g., 22^3 SC at cutoff 3.01 has 650 000 candidate edges
  and builds in about a second). The sculptor's cost grows with the
  candidate coordination.

A shell exactly at 1.0 (corner-corner on BCC and FCC, the fourth Diamond
shell) is excluded by the default and included by any larger value (e.g.,
`1.01`). `MIX` uses the distance search at every cutoff, so `MIX` at
fractions `{"SC": 1}` has exactly the candidate-edge set of `SC` at the same
cutoff (same ids, positions and edges). At the default cutoff the two insert
those edges in a different order, so a seeded sculpt gives a different
draw. A cutoff below a pure lattice's nearest-neighbor distance (0.866 on
BCC, 0.707 on FCC, 0.433 on Diamond) is refused, since no site would have a
neighbor.

The C generator takes the same range as an optional ninth argument (e.g.,
`generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 SC 1.74`). For `MIX` the
range is part of its own argument. Runs with `topology.generator.exe_path`
pass it through automatically.

**Running the C binary by hand.** `--seed=N` fixes its whole random stream
(the same stream as `TOPON_SEED=N`, which the flag overrides), and
`--output-dir=DIR` says where the `.nodes` and `.edges` files go (default
`output` in the working directory, created if missing). Without a seed the
binary seeds from the clock and the process id and prints the seed it used.
The file names carry only the lattice size and the trial number, so
concurrent runs must write into separate directories, or they overwrite each
other's networks. Two runs with the same seed and arguments write identical
files. `--odd-walks=on|off` switches the odd walks of the exact search (on by
default). The full command line is in `topon/topology/csrc/README.md`.

```bash
generator.exe 8x8x8 111 4 1000 1 "0:20,1:40" 0 SC --seed=7 --output-dir=runs/seed7
```

##### Boundaries (`periodicity`)

`periodicity` takes one digit per axis, `1` for periodic and `0` for open.
An open axis has no wrap-around bonds, so the lattice has a free surface
there and the sites on it lose coordination. The set of sites is the same
either way.

```json
"generator": { "lattice_size": "6x6x6", "periodicity": "110" }
```

This builds a slab, periodic in x and y and open in z. On a 4x4x4 SC
lattice the bond count drops from 192 to 176, 160 and 144 as one, two and
three axes are opened, and the surface sites drop from degree 6 to 5.

Open boundaries interact with `degree_distribution`. Corner and edge sites
on a free surface have very low coordination, and the centered lattices
lose the most.

| lattice (4x4x4) | min degree, `"111"` | `"110"` | `"000"` |
|---|---|---|---|
| SC | 6 | 5 | 3 |
| BCC | 8 | 4 | 1 (2 such sites) |
| FCC | 12 | 8 | 3 |
| Diamond | 4 | 2 | 1 (22 such sites) |

The usual `"0:0,1:0"` (no isolated nodes, no dangling ends) therefore
cannot be satisfied on a fully open BCC or Diamond lattice. The only way to
clear a degree-1 site is to cut its last bond, which makes it degree 0, and
that is also forbidden. Both generators report failure in this case. On SC
the minimum stays at 3, so the same request works. On open lattices, drop
the `1:0` term or leave `degree_distribution` empty and let
`max_functionality` set the degrees.

`max_functionality` still applies on top, so a partially open lattice
reaches the ceiling with less pruning than a closed one.

In the data file, coordinates are wrapped into the box only on periodic
axes. On an open axis the atoms stay where they were placed and the box
grows to contain them, so a junction on the free surface stays next to the
chains bonded to it. The `.nodes` file records the boundaries in a
`# PERIODICITY 100` header (written only when an axis is open), and the
conformation stage reads it.

An open axis also gets 12 Å of vacuum between the outermost atom and the
box face, equal to the pair cutoff of the generated scripts. LAMMPS
*deletes* atoms that leave a non-periodic (`f`) face, and the starting
geometry of stage 1 is strained enough that surface atoms move several Å
in the first few dozen steps. With only 1 Å of clearance a bonded atom can
be lost. The `open_axis_pad` argument of
`ConformationManager.apply_displacements` changes the padding (it is not a
config key), for a run that needs more or one where the extra volume
matters.

The generated LAMMPS scripts do not read the periodicity and always write
`boundary p p p`. Set the boundary to match by hand (e.g., `boundary p f f`).
The data file is already correct for it. On a Diamond `100` network both
`p f f` and `p p p` complete the stage-1 minimization with no bond crossing
an open face.

##### Mixed lattices (`lattice_type: "MIX"`)

The three cubic lattices share the cell corner. BCC adds one body center
and FCC adds three face centers. `MIX` puts the corner in every cell, the
body center with probability `mix_fractions.BCC`, and each face center with
probability `mix_fractions.FCC`. The `SC` entry is the remainder and places
no site of its own, so the three fractions sum to 1. The expected site
count is `Nx*Ny*Nz * (1 + f_bcc + 3*f_fcc)`.

```json
"generator": {
  "lattice_size": "6x6x6",
  "lattice_type": "MIX",
  "mix_fractions": {"SC": 0.2, "BCC": 0.4, "FCC": 0.4},
  "max_functionality": 4
}
```

Mixing gives more neighbor distances. A pure SC lattice has a single edge
length, while the mixture above has four (0.5, 0.707, 0.866 and 1.0 cell
units), which smooths the distribution of strand end-to-end distances.
Three points apply.

- `MIX` at `{"SC": 1}` reproduces `SC` exactly, down to node ids, at every
  cutoff. The other two pure settings do *not* reproduce their lattices at
  the default cutoff. `MIX` connects by distance cutoff instead of a fixed
  neighbor pattern, so at `{"BCC": 1}` the 1.0 cutoff also admits the
  corner-corner shell and every node has 14 neighbors instead of BCC's 8
  (18 instead of 12 at `{"FCC": 1}`). Use `lattice_type: "BCC"` or `"FCC"`
  for the canonical coordination. `"BCC"` with `neighbour_cutoff: 1.01`
  gives the same 14.
- Edge lengths vary. A body center and a face center can be 0.5 cells
  apart, half the SC spacing. DP is assigned independently of edge length,
  so strands of the same DP are built at bond lengths that differ by up to
  2x. Check for FENE strain on the long edges.
- The fractions are a weak control. Many SC/BCC/FCC splits fit a given
  target about equally well. Site jitter and a Gaussian-weighted edge rule
  (neither is a topon option) change the strand statistics much more. As a
  starting point, use SC-heavy fractions for short strands and shift toward
  BCC and FCC as strands get longer, then check the result by measurement
  instead of fine-tuning the percentages.

On `MIX`, lowering `neighbour_cutoff` below 1.0 drops the corner-corner
shell and disconnects the corner sublattice (present in every cell) from
itself, which is why 1.0 is the default. Raising it admits further shells
on the mixed point set, as on the pure lattices (see *Neighbor shells*
above).

**The site count is a draw.** A 5x5x5 mixture at 0.5/0.3/0.2 averages
237.5 sites with a spread of about 9, so a `degree_distribution` with
absolute counts fits one draw and not the next. Work the counts out for
the lattice that is actually built. `count_sites(config, seed)` in
`topon.topology.generator_python` returns the site count for a seed (the
same draw the generator makes, so `topology.generator.seed` with that
number builds exactly that many sites), and `rescale_degree_counts`
carries a P(f) to it.

```python
from topon.config.schema import GeneratorConfig
from topon.topology.generator_python import count_sites
from topon.topology.degree_matching import (
    rescale_degree_counts, format_degree_distribution)

cfg = GeneratorConfig(lattice_type="MIX", lattice_size="5x5x5",
                      mix_fractions={"SC": 0.5, "BCC": 0.3, "FCC": 0.2}, seed=7)
n = count_sites(cfg)                                   # this seed's sites
spec = format_degree_distribution(
    rescale_degree_counts({0: 5, 1: 10, 2: 20, 3: 25, 4: 40}, n))
```

Seed 7 draws 247 sites there and seed 8 draws 219. A request that does not
fit is refused before the first trial with both numbers (the counts that seed 7
gives, run with `"seed": 8`, are refused as 235 active sites on a lattice of
219). The strict search needs the counts of a
request that names every degree to add up to the lattice exactly, and those
of a partial one not to exceed it. The exact search takes any vacancy count
and only needs the active sites to fit. `topon doctor` runs the same check
(`site_count`) and warns about a `MIX` that no seed fixes. The C binary draws
its `MIX` sites from its own stream, so `count_sites` describes the lattice
of the Python generator only.

#### `topology.existing_files`

Give either `gpickle_file` or both `nodes_file` and `edges_file`.

| Key | Type | Default | Description |
|---|---|---|---|
| `nodes_file` | string \| null | `null` | Path to `.nodes` file |
| `edges_file` | string \| null | `null` | Path to `.edges` file |
| `gpickle_file` | string \| null | `null` | Path to NetworkX `.gpickle` file |

##### `.nodes` file format

Each line holds whitespace-separated `NodeID X Y Z Degree`, and `#` starts a
comment. An optional `# BOX Lx Ly Lz` header records the periodic cell in
lattice units.

```
# BOX 6 6 6
# NodeID X Y Z Degree
0 0.000000 0.000000 0.000000 3
1 1.000000 0.000000 0.000000 3
```

The header is optional, but **write it for any lattice whose sites are not
integer-spaced.** Without it topon estimates the cell as `max - min + 1`
over the coordinates. This is exact for SC but too large for BCC, FCC and
Diamond, whose basis sites sit at fractional offsets (e.g., 4.5 for a
4x4x4 BCC or FCC). The estimate drives every minimum-image calculation, so
about a third of BCC edges (a quarter of FCC) are then built at twice their
true bond length. Files that topon generates always have the header, so
this matters only for hand-written or external files.

#### `topology.crosslinking`

This block is read when `topology.source` is `"crosslink"`. It builds a
network crosslinked along its chains (a randomly crosslinked or vulcanized
polymer, or a protein whose crosslinkable residues the sequence places) the
way the chemistry does. The chains are grown one after another as
self-avoiding walks on a periodic cubic lattice (one bead per site, with
`packing` of the sites filled), and reactive beads that touch are then
crosslinked in random order until the target is met
(`topon.topology.chain_crosslinking`). A crosslink joins two beads, so every
junction has functionality 4, a primary loop is a crosslink within one chain
with no other between its beads, and every strand's `dp` is the number of
beads strictly between its two crosslinked beads (one less than their
distance along the chain). The chains and the DP are decided here, in stage
1, so stage 3 keeps them. It assigns types, entanglements, grafts and
copolymers as for any graph, but draws no `dp_distribution` and runs no
chain cover. `assignment.defects` and `topology.generator.architecture:
"random_crosslinked"` are refused with this source, since the melt makes its
own loops and sol.

| Key | Type | Default | Description |
|---|---|---|---|
| `chains` | list | required | Chain types, each with `count` and its reactive beads from exactly one of `reactive_every` (with `reactive_start`, default 1), `reactive` (bead indices) or `sequence` (with `repeats` and `crosslink_residue`, default `"Y"`, one bead per residue). `dp` is the beads per chain unless a sequence gives it. `site_type` labels the reactive beads for `pairs` (a sequence labels them with its residue) |
| `crosslinks` / `conversion` / `per_chain` | int / float / float | one required | The target, as a crosslink count, as the fraction of the reactive beads that react (every reactive bead the chains carry, including those left out next to a chain end), or as crosslinked beads per chain (a crosslink counts on both chains). It is met exactly, or the run stops and says how many it could make |
| `packing` | float | `0.4` | Fraction of the lattice sites the melt fills (at most 0.6). This is a physical parameter, since a fuller lattice spreads the crosslinks more evenly |
| `persistence` | float \| null | `null` (0.2) | Weight of a straight step against a quarter of the rest for each turn (0.2 weighs the five open directions alike) |
| `c_inf` | float \| null | `null` | The chains' characteristic ratio in lattice bonds, in place of `persistence`, which is then solved for on a small melt at the same packing |
| `contact_radius` | float | `1.5` | Reactive beads this close (in lattice units) may crosslink. 1.5 takes the face and edge neighbors, 1.0 the face neighbors only (the rule of the bond-fluctuation generator, see below) |
| `max_radius` | float | `3.0` | How far the candidate shells go out when the contacts cannot make the count (at most 6) |
| `min_gap` | int | `6` | Fewest bonds between two beads of one chain that crosslink each other (at least 3, so a primary loop keeps two beads) |
| `pairs` | list \| null | `null` | Site types that may pair (e.g., `[["A", "B"]]`). `null` lets any pair react |
| `keep_windings` | bool | `false` | Refuse a crosslink that makes a strand run across half the box, which the builder would draw the short way round. Off, such strands are made and counted (`strands_rewound`) |
| `min_dangling_dp` | int \| null | `null` (0 coarse-grained, 1 atomistic) | Fewest beads a dangling strand keeps between its crosslink and the chain's end bead. Reactive beads closer to a chain end are left out, and 0 lets a crosslink sit on the bead next to a chain end. The atomistic route needs at least 1 |
| `seed` | int \| null | `null` | Pins the melt and the crosslinks. `null` takes a seed from the global NumPy stream |

```json
"topology": {"source": "crosslink", "crosslinking": {
  "chains": [{"count": 200, "dp": 100, "reactive_every": 3}],
  "crosslinks": 600, "seed": 1}},
"chemistry": {"model_type": "coarse_grained", "target_density": 0.85}
```

`topon fit` writes this block from a network crosslinked along its chains
(the chains, the reactive beads, the count, the contact rule, `min_gap` and
the packing, §3.6a).

A protein chain type is `{"count": 8, "sequence": "GGRPSDSYGAPGGGN",
"repeats": 12}`, which makes 180 beads per chain with the tyrosines reactive
(a tyrosine at a chain end is skipped). Two reactive types that only pair
with each other are two chain types with their own `site_type` and
`"pairs": [["A", "B"]]`.

Three rules keep the build sound. Every strand's chord is at most its
contour at the scale of the coarse-grained build (`dp + 1` bonds of 0.97
sigma in the box that `chemistry.target_density` gives, with one bead per
crosslink, since a crosslink is built as one bead), so no strand starts
stretched. No two crosslinks sit on neighboring beads of a chain (they
would form one junction of functionality 6). And a chain's end bead never
crosslinks.

On the coarse-grained route a crosslink may sit on the bead next to a chain
end, and the builder then bonds the end bead straight to the junction (a
dangling strand of no beads). The atomistic route joins a strand to its
junction through a monomer, so there `min_dangling_dp` stays 1, the beads
next to a chain end are left out, and the run reports how many. Sol chains
of several lengths are built at their lengths.

The graph keeps its melt. `topology/crosslinked_melt.npz` holds every
chain's beads on the lattice (`positions`, `lengths`), the crosslinks as
`(chain, bead, chain, bead)` rows in the order they were made, and the
`box`, and the `topology` section of the manifest records the counts, the
measured `c_inf`, the rejections by rule and the timing. From Python,
`crosslink_chains(..., positions=..., lattice=L)` crosslinks a melt made
elsewhere on the same lattice (each chain's beads as unwrapped integer
coordinates) instead of growing one.

**Three site rules for the Python API.** `crosslink_chains` takes three
opt-in rules that `topology.crosslinking` does not. `ChainType(...,
allow_ends=True)` lets the end beads be sites (an end crosslinked to an
interior bead is a junction of 3, and two ends crosslinked only join their
chains). `ChainType(..., pendants=(k, ...))` hangs one side bead from each
of those beads, the side bead being the site (`grow_melt(...,
pendants=...)` places each beside its anchor as the chain grows and
returns them in `record["pendants"]`, and a given melt passes them as
`pendant_positions`). `allow_neighbours=True` crosslinks a bead whose chain
neighbor is crosslinked already, as `fix bond/create` does (the two fuse
into one junction, of 6 for two neighboring crosslinks and more for a
longer run). With any of them in use the graph is read from the bead
system by `topon.analysis.crosslinked.from_beads`, side beads reduced to
their anchors, pendant sites need `build_density=None` (the builder builds
no side bead), and the record adds `allow_neighbours`, `end_sites` and
`pendant_sites`. With the rules off every build is what it was.

**The contact rule and the parity of the lattice.** A walk on the cubic
lattice changes sublattice at every step, so with face contacts
(`contact_radius: 1.0`) two beads of one chain touch only across an odd
number of bonds, and beads on one sublattice never touch each other (a
sequence with one reactive residue in two would crosslink only between
chains of opposite sublattice). Face and edge contacts, the default, have no
such rule and give about twice the primary loops. Use 1.0 to reproduce a
bond-fluctuation network, whose loops follow the parity. Against such
references the default makes about twice the primary loops, which at low
crosslink density lengthens the paths through the network and leaves more
small pieces.

Growing the chains first matters. A pairing of crosslinks with no
positions gives a network that is far too uniform, and the same pairing on
phantom walks (chains that overlap freely) gives one that is too stringy,
because an ideal-gas density clusters the crosslinks. Grown with excluded
volume at the packing and growth step of bond-fluctuation references and
crosslinked by their contact rule, the builds show no significant
difference from nine such networks in the composite of §3.6, `lambda2`,
the mean path, the loops or the chain statistics (the mean chord aside,
since a crosslink sits at its two beads' midpoint).


### `assignment`

This section sets the graph attributes (types, DP, defects, entanglements, grafts, copolymers) written before chemistry is built.

#### `assignment.node_types`

`method` ∈ `"degree"` / `"positional"` / `"random"` / `"explicit"`.

```json
"node_types": { "method": "degree", "degree": {"mapping": {"1": "end", "2": "A", "3": "A", "4": "A"}} }
"node_types": { "method": "positional", "positional": {"dimension": "z", "num_layers": 2, "layer_types": ["A","B"]} }
"node_types": { "method": "random", "random": {"type_ratios": {"A": 70, "B": 30}} }
"node_types": { "method": "explicit", "explicit": {"0": "POSS", "1": "Si"} }
```

#### `assignment.edge_types`

`method` ∈ `"uniform"` / `"random"` / `"composite"`.

```json
"edge_types": { "method": "uniform", "uniform": {"type": "A"} }
"edge_types": { "method": "random", "random": {"type_ratios": {"A": 60, "B": 40}} }
"edge_types": { "method": "composite", "composite": {"dimension": "z", "num_layers": 3, "layer_types": ["A","B","A"]} }
```

#### `assignment.dp_distribution`

```json
"dp_distribution": {
  "default": { "mean": 25, "pdi": 1.0 },
  "per_edge_type": {
    "A": { "mean": 20, "pdi": 1.2 },
    "B": { "mean": 40, "pdi": 1.5 }
  }
}
```

`pdi` is the polydispersity index of a Schulz-Zimm distribution, and `1.0` is monodisperse.

`endlinked_dangling` (default `false`) spends one bead of every dangling
strand on its free end site. In the end-linked convention a dangling chain
has DP beads and the last one *is* the free end, while topon models that
end as a separate degree-1 site. Turn it on, together with
`output.lammps_convention: "endlinked"`, when the bead count must match an
end-linked dataset (e.g., 209 dangling strands in the DP-20 reference give
102 500 beads instead of 102 709).

#### `assignment.defects`

The defects stage runs after the topology is sculpted and before chemistry
is built. It adds four classes of network defect, plus sol chains, and
records the requested and achieved count for each.

```json
"defects": {
  "primary_loops":   { "count": 345, "placement": "by_effective_degree", "dp": null },
  "secondary_loops": { "count": 129, "endpoint_degrees": {"4,4": 115, "3,4": 6, "2,4": 7, "2,3": 1} },
  "triangles":       { "count": 0 },
  "four_cycles":     { "count": 0 },
  "sol_chains":      { "count": 11, "dp": null },
  "seed": 42
}
```

`count` is absolute at 1 or above and a fraction of the strands below 1, so
`345` and `0.069` are the same request on a 5000-strand network. Giving a
count switches that class on. `enabled: true` with no count does nothing.

| Class | What it is | How it is placed |
|---|---|---|
| `primary_loops` | A strand that returns to the junction it left (a self-loop edge, `cls="loop"`). | `"by_effective_degree"` (default) fills junctions in ascending effective degree. An effective-degree-0 junction takes two loops, and every junction from effective degree 1 up to `f_target - 2` takes one, at random within each degree. This matches the end-linked reference and reproduces its chemical P(f) exactly. `f_target` is the network's own functionality (the highest effective degree any junction carries), not `max_functionality`, which is only the valence ceiling. `"random"` takes any junction with spare chemical valence. `dp` overrides the loop's length, and `null` inherits it from the bridges on that junction. |
| `secondary_loops` | Two strands between the same junction pair (a parallel edge). | With `topology.generator.search: "exact"` they are forced into the sculpt as double edges, which is the only way to place them with the degree counts staying exact. The stage then verifies and records them (`source: "sculpt"`). Otherwise it falls back to endpoint-targeted injection, which raises both endpoints by one and so shifts P(f) by `2 x count`. `endpoint_degrees: "auto"` lets the sculpt pair like with like from the highest degree down. |
| `triangles`, `four_cycles` | One added edge closing a three- or four-cycle. | Both raise two junctions by one degree. With `topology.generator.search: "exact"` the stage reserves that capacity in the sculpt target (`2 x count` junctions sculpted one degree below `max_functionality`) and raises exactly those back, so the delivered P(f) is the requested one and `degree_shift` is 0. On any other topology the edges are added to the finished graph and `degree_shift` reports the deviation. `collateral_cycles` counts the cycles of the *other* class that the added edges also closed. |
| `sol_chains` | Chains bonded to no junction. | Not part of the junction graph, but part of the bead budget. Coarse-grained only. |

`seed` makes the placement reproducible. Without it the stage draws from
the global RNG state.

A junction has two functionalities. Its *effective* functionality counts
the strands that lead away from it, which is what elasticity sees. Its
*chemical* functionality adds two per primary loop, which is what the
crosslinker valence sees. `topon inspect` prints both.

Loops and sol chains are part of the bead budget. The box is sized from the
bead count at the target density, so leaving them out shrinks the box. In
the DP-20 end-linked reference, 345 primary loops and 11 sol chains hold 7 %
of the beads. A build without them has 7.3 % more bridges per unit volume,
which alone raises the tensile peak by 7 %.

A primary loop is a self-loop and a secondary loop is a parallel pair, as in
the polymer-network literature. In earlier versions the `primary_loops` key
and the `inject_primary_loops` function both meant parallel edges. A config
that still sets `primary_loops.target` (the old field) gets the old behavior
with a `FutureWarning`. Use `secondary_loops` for parallel strands and
`primary_loops.count` for self-loops.


#### `assignment.entanglements`

```json
"entanglements": {
  "enabled": true, "target": 5, "target_type": "count"
}
```

Distribution mode sets an average per chain instead.

```json
"entanglements": {
  "enabled": true,
  "avg_crosslinks_per_chain": 2.0
}
```

##### `method` (how an entangled pair is realized in 3D)

| `method` | What it draws |
|---|---|
| `"waypoint"` (default) | The pair together. Both chains are splines that spiral about their contact in antiphase, so the pair carries exactly `entanglement_count` windings by construction (checked with primitive-path analysis). |
| `"kink"` | The legacy Gaussian bump aimed at the partner's midpoint. Each chain is drawn alone, so the windings the pair carries after relaxation are statistical. Kept to reproduce systems built with it before `waypoint` existed. |

**Chords that cross.** The waypoint braid gives each chain a radius of 0.45
of the gap between the two chords at the site. On a three-shell SC lattice
two face diagonals of one face, or two body diagonals of one cell, cross
at their midpoints, and since their midpoints coincide the selection takes
such pairs first. Before 0.4.5 a crossing pair was drawn with both chains
through one point and no winding. A pair whose gap at the site is under
5 % of the shorter chord (`waypoints.CROSSING`) is now wound as though the
chords were half the shorter chord apart (`CROSSING_GAP`), the chains
parted along the normal to both chords.

**Other strands kept out of the braid.** A braid's radius is also capped at
half the distance from its axis to the nearest other strand's chord beside
its span (`realize.CHORD_SHARE`, at least 5 % of the shorter chord), since a
strand whose chord passes inside the radius would run between the two arms
and share the winding. On the atomistic route the placement then turns
every other strand with a backbone atom inside a braid's cylinder (its
radius plus 1.5 Å) about its own chord to the first of 36 angles that
clears it (`topon.conformation.atomistic.clear_braids`). Turning keeps its
bonds, angles and junctions and draws nothing from the random stream. The
manifest's `placement.braids` counts the strands found inside, turned
clear and left inside.

`kink_params` applies only to `method: "kink"`.

| `kink_params` key | Default | Description |
|---|---|---|
| `overshoot` | `0.2` | How far the kink extends past the midpoint (0-1) |
| `z_amp` | `0.5` | Out-of-plane amplitude of the Gaussian kink |
| `sigma` | `0.15` | Width of the Gaussian kink |

##### Choosing pairs on a conformation instead of on crosslink distance

By default, candidate pairs are ranked by the distance between their
crosslinks, which is a property of the network and not of the chains. Two
chains can be nearest neighbors by crosslink and never come near each
other, and a kink placed there aims one chain at a partner that is not
nearby.

`select_entanglements` accepts a `chain_paths` argument (a mapping of
`frozenset((u, v))` to that chain's bead path) and ranks candidates by how
much of the two chains lie alongside each other. The assignment stage does
not draw the conformation. The caller supplies one, in the same units as
the node positions. The steps are to draw a provisional conformation with
no entanglements, rank and select pairs on it, and then draw the final
conformation with the kinks.

`topon/conformation/paths.py` provides `bridging_walk` for the first pass,
a random walk of fixed bond length that ends exactly on its far junction. A
straight chain is not suitable, since it lies on its chord and says nothing
about which chains meet.

The table compares the two rankings on a 354-chain network (281 candidates,
0.20 entanglements per chain).

| ranking | median proximity of chosen pairs | chosen pairs whose chains never touch |
|---|---|---|
| crosslink distance (the default) | 39 | 8 of 33 |
| on a conformation | **156** | **0 of 35** |

The median over all candidates is 50, so the default ranking does slightly
worse than a random choice.

##### Drawing a path around what is already there

`topon/conformation/paths.py` also has the functions for placing a chain in
an occupied box. `Pipeline` does not use them, so call them from Python
directly.

| name | does |
|---|---|
| `Clearance(points, box, radius)` | the beads already present, as a minimum-image nearest-neighbor query, with methods `near`, `worst` and `ok` |
| `bridging_walk(..., avoid=)` | the same fixed-bond random walk, keeping each step clear of `avoid` |
| `loop_around(target, i, radius, n_pts, phase, avoid, span)` | waypoints encircling a strand. `span` is turns, so 0.5 is a hook and 2.0 is two turns |
| `taut_leg(start, end, n_bonds, bond, avoid, placed)` | a deterministic leg with exact bonds that lands on its end and avoids `avoid`, its own earlier beads, and `placed` |
| `route_through(start, end, waypoints, n_bonds, bond, avoid)` | visits every waypoint in order using `taut_leg` for each leg |

Route with clearance whenever the box already holds beads. A path drawn
without it lands on existing beads (e.g., a closest pair at 0.195 σ in a
relaxed melt at density 0.85, where the WCA energy is of order 10⁵ kT). The
next minimization then drags chains through each other and changes the
topology just built. With `Clearance` the tightest contact of a routed path
in the same melt was 0.822 σ.

A chain usually has much more contour than its route needs (e.g., 77 σ for
a route of about 21 σ). A random walk uses up the extra length by
wandering, and the wandering can cross the target again, so the
entanglement count no longer follows the design (the same pair, site and
winding drawn with three seeds gave 4, 7 and 0). `route_through` places the
extra contour deterministically, so a requested count is repeatable.

`walk_through` (random legs) and `route_through` (deterministic legs) take
the same arguments and give the same guarantees on bonds and junctions. Use
the first for a melt-like conformation and the second for a specific
topology.

##### Choosing which neighbor shell to entangle

`shell_weights` biases the draw toward particular neighbor shells. Shells
are numbered from 1, closest first, and are read from the lattice. The
closest approach between two strands takes a few discrete values (e.g.,
0.20, 0.35, 0.41 and 0.50 lattice units on a mixed SC/BCC/FCC network), and
these bands define the first, second and later shells.

```json
"entanglements": {
  "enabled": true,
  "avg_crosslinks_per_chain": 2.0,
  "shell_weights": { "1": 0.7, "2": 0.3 }
}
```

Naming shells restricts the draw to those shells and weights them in
proportion. Without the key (the default) the draw uses every shell
equally. The shell weight multiplies `placement_bias_kind` instead of
replacing it, so spatial and shell biases combine.

Only the first shell reliably gives an entanglement. The table shows how
many requested pairs were realized, each checked by primitive-path analysis
after the full three-stage protocol.

| shell | gap | realized as asked |
|---|---|---|
| 1 | 12.3 σ | 5 of 7 |
| 2 | 21.4 σ | 2 of 16 |
| 3 | 24.7 σ | 0 of 16 |

What matters is the gap between the pair divided by the chain's chord, 0.29
in the first shell and 0.50 in the second. A chain must spend contour to
reach its partner, and beyond about a third of a chord it runs out. The
ratio is scale-free, so it is a property of the lattice. It is 0.29, 0.50
and 0.58 for every SC/BCC/FCC mixture at any fractions and any box size,
and larger for the pure lattices (FCC 0.71, BCC 0.82). Weighting the outer
shells up is allowed but does not give more entanglements there.

#### `assignment.grafts`

```json
"grafts": {
  "enabled": true,
  "per_edge_type": {
    "A": { "graft_density": 0.05, "side_chain_monomer": "PDMS", "side_chain_dp": 5 }
  }
}
```

On the atomistic route a graft replaces a methyl on the head atom of its
repeat (the Si of a siloxane) with the O its side chain hangs from, and the
side chain is PDMS (`Si(C)(C)O` x `side_chain_dp`) whatever
`side_chain_monomer` names. Another monomer there gets a build warning and a
`topon doctor` warning. A repeat whose monomer has no methyl on its head
atom stays ungrafted, with a warning. Before 0.4.5 only PDMS backbones were
grafted on this route.

#### `assignment.copolymer`

```json
"copolymer": {
  "enabled": true,
  "per_edge_type": {
    "A": {
      "arrangement": "block",
      "composition": [
        { "monomer": "A", "fraction": 0.5 },
        { "monomer": "B", "fraction": 0.5 }
      ]
    }
  }
}
```

`arrangement` ∈ `"block"` / `"alternating"` / `"random"` / `"gradient"`.

Each strand of the edge type gets one monomer name per repeat
(`monomer_sequence`). On the coarse-grained route a name is a bead type. On
the atomistic route it names an entry of `chemistry.monomers`, and since
0.4.5 each repeat is built of its own monomer, in order from the strand's
first junction (before, every atomistic strand was built of its edge type's
monomer and the sequence was dropped). With grafts on the same edge type,
each grafted repeat carries its side chain in place of a methyl on its head
atom, whatever its monomer, and a monomer with no such methyl is left
ungrafted, with a warning. With `force_field: "charmm"` each repeat is
typed by its own monomer's `charmm_residue`.

> **`gradient` is broken.** It ignores the requested composition and always
> gives a 50:50 split (e.g., a request for A=0.1 still gives A=0.50). For two
> monomers at equal fractions and even DP it is byte-identical to `block`.

#### `assignment.chains`

This block is read only when `topology.generator.architecture` is
`"random_crosslinked"`. The DP of every strand comes from here and
overwrites `dp_distribution`.

| Key | Type | Default | Description |
|---|---|---|---|
| `dp` | int | required | Beads per chain, its two end beads and its crosslinked beads included |
| `reactive_every` | int | `1` | Beads that may carry a crosslink, every n-th from bead 1 (`1` is every interior bead) |
| `passes` | list of float \| null | `null` | Target probability that a chain passes k junctions, for k = 1, 2, .... `null` is the binomial distribution of random crosslinking with the graph's mean |
| `chord_floor` | bool | `true` | Coarse-grained builds only. Every strand gets at least the beads that span its chord at the 0.97 sigma design bond, in the box that `chemistry.target_density` gives. Off, the crosslinked beads are uniform along each chain |
| `seed` | int \| null | `null` | Pins the cover and the split of the beads. `null` takes a seed from the global NumPy stream |

```json
"topology": {"source": "generate", "generator": {
  "lattice_type": "SC", "lattice_size": "10x10x10", "neighbour_cutoff": 1.42,
  "max_functionality": 4, "search": "exact", "min_giant_fraction": 0.95,
  "degree_distribution": "0:0,1:400,2:59,3:0,4:541",
  "architecture": "random_crosslinked"}},
"assignment": {
  "defects": {"primary_loops": {"count": 59, "placement": "by_effective_degree"},
              "secondary_loops": {"count": 73, "endpoint_degrees": {"4,4": 68, "2,4": 5}}},
  "chains": {"dp": 100, "reactive_every": 3, "seed": 1}}
```

A degree-2 junction left without a loop is a chain passing a site where
nothing crosslinks it, so the loop count should match the degree-2 count.
A primary loop gets at least two beads and a dangling strand at least one
(the coarse-grained builder bonds neither shorter one to its junction), and
the sol chains of `assignment.defects.sol_chains` are chains of `dp` beads
unless their own `dp` is set. The chain cover refuses a junction of odd
degree, a component with no chain end, and a mean number of junctions per
chain above what `dp` beads can carry. With the chord floor on, the run
prints how many chains could not meet it (their beads are then split
without it) and how many strands are still shorter than their chord. Both
numbers are in the `chains` section of the manifest, with the histogram of
passes before and after the tail exchanges.


### `chemistry`

| Key | Type | Default | Description |
|---|---|---|---|
| `model_type` | `"coarse_grained"` \| `"atomistic"` | `"coarse_grained"` | Force-field resolution |
| `force_field` | `"dreiding"` \| `"charmm"` | `"dreiding"` | Atomistic force field (§4.4) |
| `charmm.files` | list of str | `[]` | RTF, PRM and `.str` files, read in order (`bundled:NAME` for the bundled C35r ethers) |
| `charmm.bridge_residue` | str | none | RTF residue of the `auto_bridge` atoms |
| `charmm.pair_style` | `"lj/charmmfsw/coul/long"` \| `"lj/charmm/coul/long"` | `"lj/charmmfsw/coul/long"` | Stage-3 pair style (with `dihedral_style charmmfsw` or `charmm`) |
| `target_density` | float | `0.9` | Target density, in g/cm³ on the atomistic route and beads per sigma³ on the coarse-grained one (the build box is sized from it) |

With `force_field: "charmm"`, a node type and a monomer each take `charmm_residue` (the RTF `RESI` name) and, optionally, `charmm_atom_names` (the RTF names of the heavy atoms in SMILES order).

#### `chemistry.node_type_map`

```json
"node_type_map": {
  "end": { "molecule": "[Si](C)(C)C", "is_end_cap": true },
  "A":   { "molecule": "Si",          "is_end_cap": false },
  "B":   { "molecule": "POSS",        "is_end_cap": false }
}
```

The built-in molecule names are `"Si"`, `"POSS"` (Si₈O₁₂ cage) and `"POSS_AM0270"` (an end cap built on AM0270 aminopropyl POSS). Any SMILES string is also accepted.

`POSS_AM0270` is the Si₈O₁₂ cage with seven isooctyl arms and, at corner 0, a propyl arm whose end carbon bonds to the strand, 204 atoms with hydrogens. The real AM0270 has `-CH₂CH₂CH₂-NH₂` there, and a cure bonds its N to an opened epoxide, POSS-(CH₂)₃-NH-CH₂-CH(OH)-CH₂-O-(CH₂)₃-[PDMS]. The pipeline's PDMS strands have no such linker, so the arm is tethered straight to the strand's end atom, and that bond stands in for the N and the linker. It goes to the head Si when the cap is the strand's first end in the graph, a carbosilane (`C_3-Si3`), and to the tail O when it is the last, an alkoxysilane (`C_3-O_3`). `topon simbox` builds the amine (its AM0270 is 207 atoms, the N typed `N_3`) and the epoxy-amine reaction.

On the atomistic route every node type the graph uses must be in the map, and a molecule must be an element symbol, a built-in name or a SMILES that RDKit parses. Anything else stops the chemistry stage with the unmatched types and the keys the map has (before 0.4.0 such a node became a bare Si without notice). A coarse-grained junction is one bead whatever its molecule, so the coarse-grained route does not check. A junction built as a bare Si with fewer strands than its four bonds (an effective degree 2 or 3 site, or a crosslinker that carries one strand) gets methyls on the rest, so a trifunctional junction is MeSi(O-)3 and a crosslinker with one strand is Me3Si-O-, as on the workflow route. Before 0.4.0 the pipeline filled those valences with hydrogen, giving a Si-H that PDMS does not have.

**DREIDING atom types.** Every atom is typed by its element and the hybridization RDKit perceives, as the element with the hybridization suffix (`Si3`, `Al3`), the same with an underscore (`C_3`, `O_2`, `N_1`), the element alone (`Cl`, `Br`, `Ca`) or the element and an underscore (`H_`, `F_`, `I_`), whichever the parameter file has first. Since 0.4.5 an aromatic C, N or O (as RDKit perceives aromaticity) takes the resonant type instead, `C_R`, `N_R` or `O_R`. Before, a phenyl ring was typed `C_2`, so every ring bond took the C=C bond (r0 1.33 Å) and every ring torsion the double bond's. Only aromatic atoms are resonant, so an amide, ester or conjugated chain keeps its `_2` and `_3` types. The one exception is an O bonded to an aromatic atom by single bonds only and to no acyl carbon (an aryl ether or phenol O, which RDKit perceives as sp2), which is `O_R` (`topon.forcefield.dreiding._aryl_oxygen`). The parameter file has no `S_R`, so an aromatic S is an error (below). A planar center (sp2, three neighbors) carries DREIDING's three inversion terms at K/3 each under `improper_style umbrella`, which every DREIDING stage script declares. Before 0.4.5 the pipeline's writer wrote one term `K -1 0` under cvff, a constant with no force, so no sp2 center on the pipeline route was held planar. The pipeline's writer reads `topon/utils/DreidingX6parameters.txt`, and simbox and `create_lammps_data_file` read `topon/forcefield/DreidingX6parameters.txt`. The two copies are identical.

An atom with none of these types stops the chemistry stage with `UntypedAtomError`, which lists each such atom by LAMMPS id with its element, hybridization and bond count, and `topon generate` prints it after `DREIDING:` and exits 1. Three things lead there, a molecule that reached the writer unsanitized (on the pipeline route, when Sanitize fails, and the error then names that failure too), a Si with five or six bonds (RDKit perceives it as SP3D, and DREIDING has only `Si3`), and an element or hybridization the file lacks (e.g., Li, Mg, an sp2 Al or the aromatic S of thiophene). Before 0.4.5 such an atom was written as an invented type (`Si_`, `O_`) with a made-up mass, a placeholder pair term and the generic bonded terms, without a warning on most routes. Simbox types its molecules by the same rule (`topon.forcefield.dreiding.assign_atom_types`). Each type's mass is its element's atomic weight, plus its hydrogens' for the united-atom types (`C_34` and the like). Before 0.4.5 the `Na` row of both parameter files carried the element and mass of scandium.

**Charges.** The pipeline charges a DREIDING build with Gasteiger charges (`topon.forcefield.dreiding.gasteiger_charges`), after Sanitize and AddHs and after typing, and spreads a net charge over 1e-6 e evenly across the atoms. Since 0.4.5 a failure there stops the chemistry stage before any file is written, with `ChargeError` naming the step. That is Sanitize, AddHs or Gasteiger raising, or Gasteiger charges that come back NaN, which is how RDKit reports an element it has no Gasteiger parameters for (among those DREIDING types, e.g., Na, Ca, Ti, Fe, Zn, Ge and Sn). The error lists each such atom by LAMMPS id, element, hybridization and bond count, and `topon generate` prints it after `DREIDING:` and exits 1. Before 0.4.5 the stage wrote the heavy-atom molecule uncharged and without hydrogens when a step raised, and set NaN charges to 0, each with only a warning. This is the pipeline's route only. Simbox and `topon chain` compute no charges and write every atom's as 0 by design, and simbox's stage scripts use `lj/cut` with no k-space.

#### `chemistry.edge_type_map`

```json
"edge_type_map": { "A": { "monomer": "PDMS" }, "B": { "monomer": "FPDMS" } }
```

#### `chemistry.monomers`

The built-in monomers are listed below.

| Name | SMILES | Description |
|---|---|---|
| `PDMS` | `[Si](C)(C)O` | Polydimethylsiloxane |
| `FPDMS` | `[Si](C)(CCC(F)(F)F)O` | Fluorinated PDMS |
| `Phenyl` | `[Si](C)(c1ccccc1)O` | Phenyl-PDMS |

Custom monomers are added in the same block.

```json
"monomers": {
  "MyMonomer": {
    "smiles": "[Si](CC)(CC)O",
    "chain_head": "Si",
    "chain_tail": "O"
  }
}
```

A strand is its repeat units' SMILES written one after the other, so the SMILES decides which atoms join. The `u` junction (or end cap) bonds to the first atom of the first repeat, and the `v` junction to the last repeat's tail, the atom the next repeat would bond to if one followed (the last atom written outside a branch). For PDMS that is the O, and for polystyrene `CC(c1ccccc1)` or poly(methyl acrylate) `CC(C(=O)OC)` it is the backbone CH before the branch, not the phenyl or ester carbon the SMILES ends on (before 0.4.5 the `v` end bonded to the last atom of the SMILES). Write a repeat unit backbone first, with side groups as branches. `chain_head` and `chain_tail` do not choose these atoms. They are the elements `auto_bridge` compares with the junction's, `chain_head` at the `u` end and `chain_tail` at the `v` end, and their defaults are `"Si"` and `"O"`, the PDMS pair, so a carbon monomer that leaves them out gets a bridge O between a Si junction and its first carbon.

#### `chemistry.connection`

```json
"connection": { "auto_bridge": true, "default_bridge_atom": "O" }
```

With `auto_bridge`, a bridge atom (`default_bridge_atom`) goes between a strand end and its junction when the strand end's element (`chain_head` at its `u` end, `chain_tail` at its `v` end) is the junction's (e.g., both Si). Set it to `false` to always use direct bonds.

### `conformation`

This section configures stage 5. It holds three sets of keys, for the data-file route, the bead-spring route and the atomistic placement.

The first three keys drive `ConformationManager`, which rewrites a data file that already has coordinates (the atomistic and legacy CG route).

| Key | Type | Default | Description |
|---|---|---|---|
| `overlap_cutoff` | float | `0.01` | Separation below which two atoms are pushed apart, in the data file's own units |
| `overlap_max_iters` | int | `10` | Passes of the overlap resolver before it gives up |
| `noise_magnitude` | float | `1e-4` | Uniform jitter on every atom, to break lattice degeneracy. It is drawn from a stream keyed on the study name, so it is the same on every run |

The other keys drive `topon.conformation.place`, which draws a bead-spring build from the graph itself.

| Key | Type | Default | Description |
|---|---|---|---|
| `placement` | `"straight"` \| `"meander"` \| `"walk"` | `"meander"` | Chain shape at build |
| `coil_ratio` | float \| null | `null` | Strand contour over chord at build, mean over mean |
| `build_density` | float \| null | `null` | Bead density the paths are drawn at |
| `bond` | float | `0.97` | Design bond length, in sigma |
| `meander_waves` | float | `6.0` | Waves the meander spends its slack in, before the self-fold gate halves it |
| `min_bond` | float | `0.85` | Shortest bond a placed strand may carry |
| `min_self_separation` | float \| null | `null` | Closest a bead may come to a non-adjacent bead of its own chain. `null` uses the route's own floor (1.0 sigma for a drawn path, 0.05 for a walk). It is also the floor a sol chain is grown to |
| `path_jitter` | float | `0.02` | Gaussian jitter on the interior beads of a straight path |
| `junction_shell_spacing` | float \| null | `null` | Seat the first bead of every chain leaving a junction on a spread shell at least this far apart. All or nothing per junction, and most junctions decline (read `guard_report()["junction_shells"]` for what was seated) |
| `junction_jitter` | float | `0.0` | Move every junction by a Gaussian offset per axis before the strands are drawn, as a fraction of the site spacing. It breaks the exact chord crossings of a lattice, and `0` leaves every junction on its site |
| `settle_clearance` | float \| null | `null` | Once the strands are drawn, part the bonds of different strands to this bond-to-bond distance (in sigma), holding the junctions and never moving one bond through another. `null` is off |
| `parallel_strands` | `"opposite"` \| `"together"` | `"opposite"` | On the meander route, how bridges that share both junctions (secondary loops) are drawn, on opposite sides of their chord or, with `"together"`, as any other strand (the drawing of 0.4.0, byte for byte). The atomistic meander placement reads it too. See *Secondary loops* below |
| `loop_shape` | `"ring"` \| `"compact"` | `"ring"` | How a primary loop is drawn (`placement` does not apply to a strand with no chord). `"ring"` is a regular polygon through the junction, and `"compact"` a closed self-avoiding walk from the junction about the size of a relaxed loop. A graph with no primary loop builds the same either way. `topon fit` writes `"compact"`. See *Primary loops* below |
| `entanglement` | object | see below | Target, distribution, designed pairs and the controller |

The last keys are for the atomistic route (`Pipeline` with `chemistry.model_type: "atomistic"`, DREIDING or CHARMM).

| Key | Type | Default | Description |
|---|---|---|---|
| `atomistic_placement` | `"straight"` \| `"meander"` \| `"walk"` \| `"coil"` \| null | `"meander"` | Draw each strand's backbone as a chain at the force field's bond lengths and settle it, from a stream keyed on the study name, so a pinned config places its atoms the same way on every run. `null` keeps the placement of earlier versions (and with it the `soft_push` stages). Since 0.4.5 a network with POSS nodes takes it too, each cage placed whole (see *POSS cages* below). `"coil"` winds each strand round its chord, and an `entanglement.target_Z` uses it |
| `atomistic_coil_radius` | float \| null | `null` | With `"coil"`, how far each strand winds from its chord (in Å). It is searched when `entanglement.target_Z` is set |
| `atomistic_clearance` | float | `1.5` | With `atomistic_placement`, settle every backbone to its bond lengths and angles with no two backbone bonds closer than this (in Å), moving nothing through anything. `0` skips it |

The placement of earlier versions spaces every heavy atom of a strand, methyls included, evenly on the straight chord and puts each hydrogen within about 0.3 Å of its carbon, so the soft first stage of the relaxation is what makes a molecule of the build (on a 4x4x4 three-shell DP-10 PDMS cell the Si-O bonds come out between 0.07 and 5.2 Å, against r0 = 1.587 Å). With `atomistic_placement` set, the backbone of every strand (junction to junction through the bridge O, Si-O-Si for PDMS) is drawn with the bead-spring path routines at the strand's mean equilibrium bond length, and every other atom is set from its placed neighbor at its bond length in tetrahedral directions. The same cell then has every Si-O bond at 0.96 to 1.02 of r0 and every Si-C and C-H bond at r0. `meander_waves`, `min_bond`, `min_self_separation` and `path_jitter` keep their meaning, read in units of the backbone bond over 0.97, so a shape means the same on both routes. Here `straight` is the planar zigzag between the junctions at bond length, and it is meant for taut strands. On a slack strand the zigzag keeps every bond at r0 by closing its angles, so use `meander` or `walk` for a coiled network. A primary loop is a ring through its junction (`closed_meander`), and an entangled pair keeps the path its method draws.

A network drawn this way is not yet ready for the first stage, for two reasons. Strands are drawn one at a time, so two of them can pass through the same point, and a path drawn in bead-spring units has nearly straight angles where DREIDING PDMS wants 104.5 and 109.5 degrees. Left like that, the first stage pushes pairs of backbone bonds through each other. So between drawing the backbones and placing everything else, the pipeline settles them (`topon.conformation.atomistic.settle_backbones`). Each round pushes every pair of bonds closer than `atomistic_clearance` apart along the line between their closest points, moving a short Gaussian stretch of each strand so that its bonds barely change, and then pulls every bond and every 1-3 distance towards its equilibrium (from the force field's bond angle, reached over the first 40 rounds), with no atom moving more than 0.2 Å per round. Last, the round is read as a trajectory with the geometry of the crossing detector (`topon.conformation.segments`), and the atoms of any two bonds that passed through each other are put back. Two bonds drawn exactly touching have no side to keep and are parted by a hair on a random side first, which is the only choice the pass makes. That parting is read as a trajectory too, so a push that carries any other bond through one is put back, and a contact it leaves with no side at all is parted again with all its pairs on one side. Junctions and end caps are held. The pass stops when no pair is closer than the clearance and every bond and 1-3 distance is within 2 % of its target, or after 1,500 rounds (`topon.conformation.atomistic.SETTLE_ROUNDS`, 400 before 0.4.5). A pass stopped by the cap keeps whatever pairs are short at that round, and the pipeline warns. Since 0.4.5 it also warns when the cap stops a settle with none short, with its rounds, its backbone bonds against r0 and its largest angle error, and for a POSS build the backbone atoms a cage still holds too close (`topon.conformation.atomistic.settle_warning`). One warning at most is given. The manifest's `placement.settle` records the pairs closer than the clearance before and after, the angle error before and after, the rounds, the largest shift, the moves put back, the touching pairs parted first and the atoms that parting put back, and the passages over the whole pass read again as a trajectory, which must be 0. On a DP-30 PDMS cell of 36,248 atoms the pass takes about 40 s, takes 2,434 pairs closer than 1.5 Å to none and the mean angle error from 62 degrees to 0.14, and leaves the bonds between 0.975 and 1.018 of r0, with no passage.

The chord guard reads the chord against the backbone's extended length, the farthest its ends can be at its equilibrium bond lengths and angles (0.79 of the contour for DREIDING PDMS at any DP), since an atomistic backbone cannot reach its contour. At 0.97 of it the strand is drawn straight, and a chord longer than the contour is warned about, as no relaxation can build it without stretching every bond. The run manifest's `placement` section records the routines used, how many strands were taut or over their contour and the backbone bond ratios (with the earlier placement, the chord ratios only). For PDMS on a 4x4x4 SC cell at three shells and 0.97 g/cm³, the body-diagonal strands are taut up to DP 7, and chords pass the contour at DP 4.

**A ring the backbone runs through.** A strand's backbone is the shortest path from each repeat's head to its tail, so in a monomer such as `[Si](C)(C)c1ccc(cc1)O` it runs along one side of the para-phenylene (ipso, ortho, meta and para carbon) and the other two ring carbons are off it. Since 0.4.5 the settle takes every such ring whole (`topon.conformation.atomistic.ring_windows`). Its flat shape is the polygon on a circle with every side at its own bond's r0 (a regular hexagon for a phenylene), with the backbone atom on each side of the stretch on its exterior bisector. The shape is fitted to the stretch as drawn, by its two outside atoms when the ring turns the backbone by more than 15 degrees (`RING_TURN`, e.g., a meta-phenylene). Laying the shapes is a move of its own before the rounds, and the backbone bonds it carries through each other are counted apart from the rounds' passages and warned about. The off-path atoms join the settle with their bonds and 1-3 distances, and each ring with its two outside atoms is pulled onto its best plane every sweep. A build with such rings settles at 24 sweeps a round and up to 3,000 rounds (`RING_SWEEPS`, `RING_SETTLE_ROUNDS`), and every other build settles as before. Hydrogens and methyls are then placed as before, except that a group with a bond through the face of one of these rings is drawn again, up to 24 times. Without the settle (`atomistic_clearance: 0`) each ring is closed on its drawn stretch by the flat ring fitted to it (`close_path_rings`).

The off-path ring atoms are not read for clearance or passages, so the settle keeps neither other strands nor other rings off a ring, and does not act on a bond through its face. Such bonds are counted. `placement.rings` records the stretches, whether the settle held them flat, the rings closed without the settle and `bonds_through` (the bonds of the placed build through a ring's face, `backbone` and `other`), and `placement.settle` adds the ring terms and the ring laying (`ring_lay`, `ring_threads`). The pipeline warns when any bond passes through a ring.

Two backbone bonds joined through a third (one strand's last bond and another's first, across their junction) sit that bond's length apart at their equilibrium angles. A joining bond shorter than `atomistic_clearance` (an aryl ether O on a junction Si, say) would leave such a pair short at equilibrium, so the pair is read against the joining bond's length less 2 %. A pair joined through a bond of the clearance or longer (Si-O 1.587 Å, C-C 1.53 Å) is read as before.

What is not covered. A saturated ring in a backbone (a 1,4-cyclohexylene) is taken whole and flat too, with 120-degree angles its `C_3` atoms do not have, and the first stage's angle terms bring it back to a chair. Two rings in one repeat unit (a biphenylene, or a diphenyl ether as in PEEK and PPO) are each in the settle, but the settle does not converge on them. A fused ring system in a backbone (a naphthylene) keeps the tree walk. The guard that draws a taut strand straight reads a ring stretch as the zigzag of its bonds, so it gives a para-phenylene repeat about 10 % more reach than the flat ring has.

**A ring that hangs from one atom.** A phenyl on a backbone Si, as in methylphenylsiloxane, is placed whole by the walk that sets every atom off the backbone, as a flat regular polygon off the bond it hangs from (rings of up to 8 atoms, `RING_MAX`). Since 0.4.5 each such ring is tried at 12 turns of 30 degrees about its bond (`PENDANT_TURNS`), and on the direction of each other atom its parent has to place (the methyl on the same Si), and placed where the fewest bonds thread it and then the fewest backbone atoms sit within 1.5 Å of one of its atoms (`PENDANT_DEEP`). A bond threads it when it passes through its face, or when one of its own bonds (or the in-plane bond each of its carbons will have to its hydrogen) passes through another ring placed before it. Hydrogens and methyls placed after it are drawn again off its face. The placement record's `pendant_rings` gives the rings placed, turned and moved, those left threaded or crowding the backbone, and `bonds_through`, and the pipeline warns when any bond passes through a pendant ring. The rest of the crowding is cleared by the hard-backbone first stage, which keeps aromatic ring atoms hard on the DREIDING deck (`atomistic_protocol`, under `simulation` below). The CHARMM hard-backbone deck takes its hard types from the backbone alone, so a CHARMM build with such rings has the placement above and the first stage of 0.4.0.

**POSS cages.** Since 0.4.5 a network with POSS nodes (`POSS_AM0270`, or the bare `POSS` cage) is placed the same way, each cage whole, as a rigid body held through the placement. Its shape is the node's own fragment of the network molecule (cage, arms and hydrogens, with a stub atom in place of each strand atom it bonds to), embedded once per kind with the simbox's checked routine (`topon.simbox.molecule.embed_conformer`) and then minimized again with every bond held at the force field's r0 (`topon.chemistry.node_bodies`). The shapes of the AM0270 fragments the POSS demo builds, and of the bare cages of topon's own tests, are stored with topon (`topon/data/poss_templates.json`, as RDKit 2025.09.6 embedded them). A fragment found there by its signature (its atoms, bonds, r0 and stubs) takes its shape from the file, so the POSS demo builds the same files whatever RDKit is installed, and any other fragment is embedded by the RDKit installed (`cages.templates_stored` in the record counts the stored ones). Other RDKit versions embed the same fragments differently (e.g., RDKit 2026.3 gives a bare cage as its mirror image), which is why the shapes are stored. They can be regenerated by embedding the fragments again with the store switched off (`topon.chemistry.node_bodies.STORE` set to None). The template is refused if a bond passes through a ring or ends 15 % off the r0 it was held at. The cage's center goes on its junction, turned so its stubs point at the strands they bond to. Two or more stubs that span a plane fix the turn, and one stub (an AM0270 cap) fixes only an axis, about which the cage takes the one of 36 turns that keeps it furthest from the other strands' chords, the other junctions and the other cages (`topon.conformation.atomistic.place_bodies`). Each strand bonded to a cage starts at its atom bonded to the cage, held where the template puts it. A strand that runs within 1.5 Å of a cage's core or through one of its faces is turned about its chord until it is clear (`clear_bodies`), and one whose chord runs through a core is drawn round it (`cage_detour`). The settle keeps every backbone bond 1.5 Å from every bond of every cage and backbone atoms out of the core's sphere, and puts back any move that takes a backbone bond through a cage bond or face, and the methyls and hydrogens set off the backbone are drawn again where one would go through a cage face. The cage's atoms go out in `system_nodes.displace`, so stage 5's overlap pass holds them as it holds the junctions. The placement record's `cages` section gives the templates, the strands turned out or drawn round a cage, `bonds_through_faces` and `bond_ratio_all` (every bond against its r0, as placed). The pipeline warns when a bond crosses a face or one is 15 % off r0. Cages closer together than their arms reach (an AM0270 reaches 9 Å from its center) cannot all be placed clear, and such a build warns. On the hard-backbone deck an AM0270 cap's attachment carbon is a strand end, so `C_3` is one of the backbone types the deck holds hard from the first step, and with it the methyls and the isooctyl arms. `atomistic_placement: null` still gives the historic placement.

Three more rules keep strands clear of cages. A designed pair's path is drawn from its junction, and a cage's junction is its center, so a designed strand bonded to a cage would start inside the cage with bonds through a face. Its stretch inside the cage's keep-out sphere is cut out instead, and the strand runs from its stub round the cage to the first of its own points outside, on an arc 1 Å outside the sphere, every atom at the spacing it was drawn with (`lead_out_of_body`, counted in `cages.led_out`). The settle keeps the side positions of each backbone atom (where the walk will put its methyls, `SIDE_BOND` out on the two tetrahedral positions off its two backbone bonds) `SIDE_CLEARANCE` (1.1 Å, a C-H) outside every cage's core, since two backbone bonds leave a methyl no other direction (`settle.bodies.sides_inside_after`). And a strand is turned off the arms of every cage it is not bonded to, a backbone bond within `ARM_CLEARANCE` (1.5 Å) of an arm bond counting as inside (`arm_margin`, `cages.near_arms`), since the settle cannot move an arm and a strand drawn between the isooctyl arms of a cap on a neighboring site stays caught there. The turn is chosen for the cores first and for the arms among the turns that tie.

#### A Z target on the atomistic route

Z1+ at the build is set by how far the strands stray from their chords. The meander's wave count moves it in jumps, because `meander_waves` is a request that the fold gate halves strand by strand (the placement record says what was drawn, `placement.waves_drawn`). `atomistic_placement: "coil"` winds each strand round its chord at `atomistic_coil_radius` Å instead (`topon.conformation.paths.coil_chain`), as a helix eased in over the first and last 15 % of the path, so the strand leaves each junction along its chord, with as many turns as the contour needs and its handedness and phase drawn from the seed. A radius wider than a strand's slack is narrowed for that strand. Z at the build rises with the radius without a step, and settling keeps it.

With `conformation.entanglement.target_Z` on the atomistic route the pipeline places coils and searches the radius on the drawn network (`topon.conformation.atomistic.coil_radius_for`). Each try draws every strand from a fresh copy of the placement's own stream at that radius, so the reading moves only with the radius, and reads Z1+ per bridge on the points the gates read (one atom per repeat unit, between the junctions) over two seeds. It reads 1.5 Å first, widens from 6 Å until the target is bracketed, and interpolates until a reading is within 3 %. The network is then settled once at that radius and read again over four seeds, and if settling moved it by more than 5 % (or `controller.tolerance`, if that is tighter) the search runs once more, aimed off by what settling added. The manifest's `placement.z_target` records the target, the radius, every reading and the settled Z. It needs Z1+. A target under what the thinnest coil (1.5 Å) reads, or past what the widest (40 Å) reads, stops there and warns. Naming another placement beside a target is refused (`topon doctor`, `atomistic_target_placement`), and the bead-spring calibration table does not apply to this route. A strand short of slack cannot stray far, so the range of targets grows with DP.

The hard-backbone deck keeps every entanglement of a coil build (no backbone passage at any stage) but not every kink, since Z1+ is not a topological count and the MD unwinds a tight helix. `topon.simulation.protocols.z_target.relax_to_target(config, raw_config)` meets the target in the relaxed network instead. Round k builds the study under `<output_dir>/<name>_z<k>`, keeping its name (which keys the placement's own stream) and the global seed, so it is the same network with only the radius changed, relaxes it with the deck and its gates, and reads Z1+ per bridge after NVT, at the chemistry's own density (the NPT reading is recorded as well). Round 2 is aimed at the target over the share of Z round 1 kept, and from round 3 the radius is interpolated on relaxed Z against radius between the two rounds that bracket the target (`next_radius`), and the network is built at that radius directly. The loop stops within 5 % of the target, or within `controller.tolerance` when the config sets it. The loop's record goes in the last round's manifest as `z_target_relaxed`. The runner is an argument (the default runs `AtomisticRun` at four threads), so the loop can be driven by another runner.

`coil_ratio` and `build_density` set the same thing, and setting both is an error. The box scales as `rho^(-1/3)` and the contour does not change, so `coil_ratio = contour / chord` scales as `rho^(1/3)`. On a given graph at a given DP, fixing one fixes the other. `place` takes exactly one, and the controller converts the one given into the one it adjusts.

#### Choosing the chain shape

The table gives Z1+ per bridge for each placement on the DP-20 end-linked
reference graph, built with `place()` and the junctions jittered (junction
jitter 0.15 and a 1-sigma settle for the meander, the jitter alone for the
walk, see *Chords that cross* below), relaxed with the default push-off,
compressed to rho 0.3075 and quenched to T 0.4. The build state is stage 3,
equilibrated at the build box.

| placement | knob | build state | final |
|---|---|---|---|
| reference (`fix bond/create`) | -- | -- | **0.178** |
| `walk` | build density 0.145 | 0.2950 | 0.3228 |
| `walk` | build density 0.095 | 0.2408 | 0.2891 |
| `walk` | build density 0.035 | 0.2220 | 0.2365 |
| `meander` | coil 1.51 (density 0.050) | 0.1986 | 0.2370 |
| `meander` | coil 1.402 (density 0.040) | 0.2017 ± 0.0048 | 0.2356 ± 0.0036 |

Coil 1.402 is the mean of three velocity seeds, and the other rows are one
seed. Neither route reaches the reference's 0.178, and their floors
coincide, since the walk at 0.035 ends where the meander at coil 1.402
does. At one build density the meander is the lower. The walk can be built
further down (coil 1.341), where the meander's settle does not converge,
and between coil 1.402 and 1.51 the meander's Z does not move beyond the
seed spread. Fewer meander waves (`meander_waves`) is the setting left
untried. `CALIBRATION` holds these rows and the earlier ones they replaced,
which carry `superseded_by` and never steer the controller.

`straight` is the chord with a small jitter. It is the cheapest and the
least entangled shape, and `meander` falls back to it when a chord is
within 3 % of its contour and there is no slack to spend. On its own it
passes the gate only near that limit. A straight strand's bonds are
`chord / n_bonds`, so at coil ratio 1.51 the bonds are 0.64 sigma and every
strand fails `min_bond`. Use it when the chords are nearly as long as the
contour, or to study a network with no coil. It is not suitable for
building a coiled network.

#### The strand gate

`place` checks three things and reports the result in `Placement.guard_report()`.

- Every bond is at or below `bond`. A longer bond means the strand's chord does not fit in its DP, which is a problem with the graph and is reported as `overstretched`.
- Every bond is at or above `min_bond`.
- No bead is within `min_self_separation` of a non-adjacent bead of its own chain.

`guard_report()` also carries `bead_bond`, which is reported and never gated. It is the closest a bead comes to a bond of another strand, and how many beads are inside 0.4 sigma of one. Every other reading is between beads (`self_contact` within a strand, `separate_coincident` across the build), and none of them can see a bead lying on a bond, which at a 0.97 bond sits 0.48 sigma from each end. On a DP-20 reference build the closest bead-to-bond distance across chains is still 0.0000 after every bead pair has been pushed a full WCA core apart. The count does not predict the stretched bonds that stop the push-off protocol, and the push-off resolves nearly all of them, so it is there only so that a clean report cannot be read as saying no bead sits on a bond. `guard_report()["chord_triples"]`, also never gated, counts the places where three strands start crowded together (see *Chords that cross* below).

`place` reports failures instead of raising them, so a sweep over
densities completes without an exception. `guard_report()` gives the
extremes over the build, how many strands failed and why, the realized
contour as a fraction of the design contour, and what the coincidence pass
found. A clean build is common but not guaranteed. On the DP-20 reference
graph, 0 of 4644 strands fail at rho 0.05 and 43 fail at the compressed
density rho 0.3075 (coil 2.77). These 43 are marginal self-contacts on
strands whose slack cannot be spent without the path touching itself, even
at the minimum wave count.

Placement takes time. On the 95 364-bead DP-20 reference graph it takes
145 s for the meander at rho 0.05, 415 s at rho 0.3075 (more slack and more
wave-halving) and 114 s for the walk. About half of this is the coincidence
pass.

The gate matters most for the meander. At its six-wave default
`meander_to_length` puts about three beads per wave at DP 20, and
resampling at equal arc length gives 0.17-sigma bonds with beads packed
between their own second neighbors. The push-off then separates those
beads *through* the bond between them, and that threaded bond lets two
strands cross later. `meander_chain` meets the gate by halving its wave
count and opening the fold (`topon.conformation.paths.unfold`), and it
reports any strand it still cannot fix.

On a lattice, chords cross, and a meander leaves beads on its chord near
the ends (where the wave envelope `sin^2(pi t)` has barely opened), so two
beads can coincide. On the DP-20 reference graph at rho 0.05, 44 bead pairs
sit within 0.05 sigma, most of them on strands whose chords intersect.
`separate_coincident` clears them all to just above the floor in under
eight rounds. `junction_shell_spacing` prevents the problem at the source
by seating the chains leaving one crosslink on a spread shell, so the beads
near the ends are off the chord.

The seat radius is `max(bond, spacing / min_chord)`. The floor matters
because `spacing / min_chord` alone is below a 0.97 bond at most
functionalities (0.577 at f = 3, 0.951 at f = 12), so without it the seat
would buy its spread by shortening the bond that gives it. At the bond
length the spread is already 1.680 sigma at f = 3 and 1.020 at f = 12, at or
above the 1.0 asked for (f = 11 is the one case where the radius does real
work, since `spread_points` relaxes numerically and is not monotone in `n`).

A junction seats all of its chains or none of them. The spread a shell
gives is the spread of every chain meeting at one node, so keeping only the
seats whose own strand passed the gate would leave moved chains beside
unmoved ones, which is more crowded than not seating at all. `place`
therefore gates strands but declines whole junctions, iterating to a fixed
point, so the build it returns is never more crowded than the one it started
from. The two ends of a chain are independent, so a chain may keep its seat
at one junction and lose it at the other. The seat is absorbed by the whole
contour and not by the few beads behind it (with the relaxation that
`separate_coincident` uses on its own moved strands), and the junction and
its seat stay exactly where they were put.

Most junctions decline, and at DP 20 all of them do. With spacing 1.0 at
build density 0.05, a 2x2x2 SC cell at DP 60 seated 1 of its 8 junctions
and a 3x3x3 cell 4 of 27, while none were seated at DP 20 or DP 100. The seat is
largest where the shell is worth most (chains that leave a junction in
nearly the same direction), so the cases with most to gain are the ones
least able to pay for it. Read `guard_report()["junction_shells"]`
(`seated_junctions`, `declined_junctions`, `seated_strands`) rather than
assuming the option did anything.

#### Chords that cross (`junction_jitter` and `settle_clearance`)

On a lattice with several neighbor shells many chords cross exactly. Every
pair of sites symmetric about one point has its chord's midpoint there, so
several chords can pass through one point, and a meander with a whole number
of waves has a node at its middle, so the strands pass through it too. Three
strands that start at one point can jam there in the push-off, and which of
the close chord triples jam depends on the velocity seed.
`guard_report()["chord_triples"]` counts them, as the pairs and triples of
bridge chords pairwise within 0.5, 1.0 and 1.5 sigma (chords that share a
junction are not a pair). It is always reported and never gated.

`junction_jitter` moves every junction by a Gaussian offset per axis, that
fraction of the site spacing, before the strands are drawn. It breaks the
exact crossings and leaves random near-triples (0.10 to 0.15 takes the
triples within 0.5 sigma of a DP-20 reference graph from 128 to between 2
and 6). The graph does not change, but the chords, the reach and the coil
ratio do. An offset that would take a strand past 0.97 of its contour (or
past its lattice chord, if that was longer), or bring two junctions within 1
sigma, is halved, and `guard_report()["junction_jitter"]` gives the reach
before and after and the junctions held back. `Placement.coil_ratio` stays
the lattice value.

`settle_clearance` parts the bonds of different strands to that distance
once everything is drawn (`topon.conformation.placement.settle_strands`,
the bead-spring counterpart of the atomistic `settle_backbones`). Junctions
are held, bonds stay within 0.03 sigma of their drawn length and inside the
gate's band, every strand keeps its self-contact floor, and every round is
read back with the geometry of the crossing detector, so that any bond that
passed through another is put back. Pairs within two bonds of a junction
that both strands share are exempt (their angle sets how close they are). A
pass that does not converge is kept only if no strand that passed the gate
fails it afterwards, and otherwise the build goes back to what was drawn
(`kept` and `gate_broken` in `guard_report()["settle"]`). Called as
`place(..., settle_clearance=c)` it runs last. After designed braids, call
`settle_placement(pl, c, rng)` instead.

Use the two together. Without jitter, or with too little (0.02), the settle
does not converge on a graph where up to five strands pass through one
point, and it puts the build back as drawn. With 0.05 to 0.15 it converges
in one to two minutes on a DP-20 reference graph of 4,644 strands, and no
two bonds of different strands are then closer than 1 sigma outside the
exempt pairs. Both keys are off by default, and the default build is
unchanged. On a `walk` the settle does not converge, since a random walk
leaves far more close bond pairs than a meander, so build walks with the
jitter alone.

#### Secondary loops (`parallel_strands`)

A secondary loop is two strands between the same two junctions. Drawn as
every other strand is, the two are one meander turned about the chord by
two random angles, so they meet wherever the wave crosses the chord and
lie within a fraction of a sigma along most of it, and the settle cannot
part them. Since 0.4.5 `place` draws the strands of a chord they share on
opposite sides of it on the `meander` route (`meander_chain(...,
side=...)`), each on a half-sine bow that leaves its junctions at 20
degrees (`paths.SIDE_ANGLE`) and is at least 1 sigma high
(`paths.SIDE_BOW_MIN`), lower where the contour cannot carry it, with the
wave laid across the bow. Three or more strands on one chord are spread
evenly about it and not held apart. The side comes from the one number the
meander takes from the random stream (its turn about the chord), so every
other strand is drawn as before, and a graph with no shared chord builds
byte for byte as before. A chord within 3 % of its strands' contour is
still drawn straight, and the `walk` and `straight` routes are unchanged.
`guard_report()["parallel"]` gives the shared chords, the strands drawn
apart and left on the chord, the bow heights and the smallest angle.
`conformation.parallel_strands: "together"` (`place(...,
parallel_strands=...)`) draws every strand as 0.4.0 did.

The atomistic placement draws its secondary loops the same way
(`Pipeline` passes `conformation.parallel_strands` to
`topon.conformation.atomistic.place_network`, and both routes take the
shared chords and their sides from `topon.conformation.paths.shared_chords`
and `chord_side`), with the bow and the wave in units of the strand's mean
r0 over 0.97 (1.64 Å for PDMS). `placement.parallel` in the manifest counts
the shared chords, the strands drawn apart and left on the chord, and their
bows.

#### Primary loops (`loop_shape`)

A primary loop leaves its junction and comes back to it, so it has no
chord and `placement` does not apply to it. `"ring"` (the default) is
`closed_meander`, a regular polygon of DP + 1 sides through the junction,
whose radius grows with the DP (3.25 sigma at DP 20, 15.6 at DP 100), so
strands drawn through such an open ring stay through it and read high Z
after relaxation. `"compact"` is `closed_walk`, a closed self-avoiding walk
from the junction. Every bond is at the design length, each step is drawn
from the cone that can still close and weighted by the Gaussian chance of
returning in the bonds left, and no bead comes within 1 sigma of another
but its bonded neighbors (or the route's floor, if higher). The floor
swells it to about the size of a relaxed loop in the `fix bond/create`
references.

| DP | `"ring"`, radius | `"compact"`, radius of gyration | reference loops, relaxed |
|---|---|---|---|
| 20 | 3.25 | 1.85 ± 0.17 | 1.96 ± 0.17 |
| 100 | 15.6 | 4.29 ± 0.60 | 4.52 ± 0.44 |

A compact loop is grown after every bridge and dangling strand. Its first
bond leaves along the emptiest direction away from the strands at its
junction, and the walk keeps 1 sigma from the beads of those strands where
it can. Strands from elsewhere are not kept off, and they thread a loop as
they would a ring in a melt. The direction of the first bond takes from the
placement's stream the two draws a ring takes, and the walk is grown on a
stream spawned from it, so every bridge and dangling strand is drawn
exactly as beside a ring. `guard_report()["loops"]` gives the shape, count,
floor, radius of gyration and closest self-contact of the loops.
`loop_shape` stays `"ring"` by default, and `topon fit` writes `"compact"`
into the configs it fits. The pipeline's own loops (`topon generate`) and
the atomistic route's are rings either way.

#### `conformation.entanglement`

```json
"entanglement": {
  "target_Z": 0.18,
  "target_hist": [0.83, 0.16, 0.01],
  "shells": {"1": 0.5, "2": 0.5},
  "pairs": [[12, 57, 1], [12, 88, 2]],
  "controller": {"max_rounds": 4, "tolerance": 0.15}
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `target_Z` | float \| null | `null` | Mean Z1+ per strand **at the final state**, or null for no target. On the atomistic route, Z1+ per bridge met by the coil radius on the build, and in the relaxed network with `relax_to_target` (see *A Z target on the atomistic route*) |
| `close_on` | `"final"` \| `"build"` | `"final"` | Which state the controller compares against. `"build"` is cheaper to reach and is right only when the build state is itself what is being matched. It does not predict the final state, since Z jumps in stage 1 and stays flat afterwards |
| `target_hist` | list \| null | `null` | Per-strand Z distribution to compare against, as fractions. Reported as a KS p-value, and nothing is tuned to it |
| `shells` | object | `{}` | Neighbor-shell mix the designed pairs are drawn from, numbered from 1 |
| `pairs` | list | `[]` | Named pairs as `[chain_a, chain_b, windings]` |
| `controller.max_rounds` | int | `4` | Build-relax-measure rounds before the controller gives up |
| `controller.tolerance` | float | `0.15` | Relative `abs(Z - target) / target` accepted as converged |

All models in this subtree refuse unknown keys, from the `conformation`
section itself down to its nested blocks. A typo such as `"target_z": 0.18` (lower-case
z) is therefore an error. If it were ignored, `target_Z` would stay null and
the controller would spend a full LAMMPS round with no target.
`topon doctor`'s `unknown_config_keys` rule skips the conformation subtree,
so the schema is the only check here.

The self-contact floor depends on the route. A meander or a jittered chord
that puts a bead within 1 sigma of a non-adjacent bead of its own chain has
folded. A freely jointed walk does this all the time (e.g., every DP-100
walk on the DP-100 reference graph), and what it must avoid is a hard
overlap (e.g., a closest pair at 0.003 sigma, where WCA is of order 1e9
kT). `place` therefore uses 1.0 sigma for a drawn path and 0.05 for a walk
unless `min_self_separation` is set. `separate_coincident` clears any pair
inside 0.05 sigma *anywhere* in the build and then puts the bonds back. It
treats junctions as fixed obstacles, because a junction's position is
shared by every strand that meets there.

`target_Z` is a final-state number. Z1+ counts the kinks of the shortest paths between the junctions at their current positions, so it changes by 10-20 % through equilibration and compression even with no bond crossings and no free-ended chains (e.g., on the DP-100 reference it was 1.16 after push-off, 1.00 after constant-volume equilibration and 1.11 after compression, with every bond under 1.2 sigma). The controller starts from the build state but closes on the final state. The atomistic route meets the target on the build instead, since its build is settled and its deck keeps every entanglement, and `relax_to_target` corrects for what the relaxation does not keep with further relaxed rounds.

Chain ids in `pairs` index `topon.conformation.strand_plans`, which follows the graph's edge order (the order in which a `Placement` keeps its strands). The sol chains come after the edges, so every other strand keeps its index, and a request naming one is refused (a sol chain has no chord to braid about). `topon.conformation.entanglement.requests_from_config` also accepts an edge key `[u, v, k]` in place of an index.

`shells` sets the shell mix for the pairs the conformation stage routes. The assignment stage chooses *which* pairs (`assignment.entanglements.shell_weights` and `select_by_shells`), and a driver calls it and passes the result in.

#### The controller

```python
from topon.conformation.entanglement import controller

manifest = controller(graph, config.conformation, runner, dp=20)
```

`runner` is a callable supplied by the caller. It receives a `RoundPlan` (the round number, the placement, which setting was changed and to what, the DP and the seed) and must return a `RoundResult` or a dict with at least `z_final`. The runner builds the system, runs the relaxation protocol and measures Z1+. Keeping these out of the module keeps the conformation stage free of LAMMPS and lets the loop be tested against a known response.

A driver for a real protocol writes the build and the five-stage push-off with `topon.writers`, runs and gates it with `topon.simulation.protocols.StagedRun`, and measures each checkpoint with `topon.analysis.z1plus.measure_checkpoint` through an installed Z1+ (§3.6).

The controller starts from the shipped calibration (`topon.conformation.entanglement.CALIBRATION`), which holds the measured (actuator, Z) pairs with the state, protocol and builder of each. These rules decide which rows seed a route.

- Rows at the requested DP come first, whatever protocol measured them. Only a DP with none borrows the nearest DP's rows, and the borrowed level is a starting guess.
- At a DP, the crossing-free push-off comes before the minimizer's rows, which are kept and labeled because that protocol stretched bonds and added crossings of its own. `"limit"` and `"pushoff"` name the same deck and are read as one.
- A row a later measurement replaced carries `superseded_by` and never steers.
- A slope is lent only when Z rises with the knob between two rows by more than the seed spread either carries (`z_sd`). Otherwise the rows fix a level, the seed is the row nearest the target, and a loop started from one knob stops after its first round and asks for a second knob rather than guess a direction.

Over the measured range Z follows a power law in the actuator (exponent 0.22 for the DP-20 walk and 0.30 for the DP-100 walk). The DP-20 meander has no measurable slope between coil 1.402 and 1.51, so it lends none.

A `target_Z` below the lowest value measured for a route gives a warning that names the floor and, for the walk, the meander's floor at the same DP, as the route to switch to when it is lower by more than the seed spread either floor carries and as no lower otherwise (the DP-20 case).

```
target_Z 0.18 is below the lowest final-state Z the walk route has reached at
DP 20 (0.2365, at build_density 0.035). ... The meander route's floor is
0.2356 at DP 20 (coil_ratio 1.402), no lower within the seed spread.
```

#### Designed pairs, and when they are refused

A named pair comes back carrying the winding it was asked for, checked on the paths as drawn. On SC 6x6x6 at DP 60, with one request per strand, 8 of 8 pairs were delivered at coil ratio 2.84 with linking numbers of 1.02 to 1.03, and 4 of 8 at coil ratio 3.85. The same machinery with no braid asked for gives a linking number of 0.00 on the same strands.

A braided strand is redrawn from its chord, not added onto the meander the placement gave it. Adding it does not work, because the meander flattens its wave across the braided stretch and the free run on either side then has to carry the whole contour in less room, and folds. The strand therefore loses the shape the placement drew, which is a fair price for a handful of designed pairs in a build of thousands, and the report says so. A winding that falls short is reported as such, and a strand that already fails the gate is not braided.

There are three limits, all refusals rather than surprises. Above a coil ratio of about 3.5 the free run's wave takes the braid apart, so the pair is drawn but does not wind. Only one braid fits on one stretch of chord, since two blending into each other collapse the winding of both. And the budgets below apply.

Each winding costs contour. The two partners leave their chords, wind around each other and come back, and a strand has only `n_bonds * bond` of contour. `route_designed_pairs` measures the routed path before drawing anything and refuses a request that does not fit, naming the smallest DP that would carry it.

```
pair (0, 514) x2: refused, the contour budget does not allow it: routing 2
winding(s) needs 33.2 sigma of path and DP 20 carries 20.4. The routed path is
33.2 sigma and these strands carry 20.4; DP 34 is the smallest that would carry
it at this geometry
```

Two other limits can apply, and a higher DP does not fix them. The braid needs `windings * pitch + 2 * ramp` of the shared axis, which two short chords may not have. That refusal suggests a lower build density, so that the chords get longer, or a partner in a closer shell. The last limit is clearance. A braid squeezed into a chord that is too short brings its two arms within a fraction of sigma, and the push-off would push them through each other, so this is also refused.

The minimum DP in a refusal holds only *at that geometry*. Rebuilding at a higher DP and the same build density puts more beads in the box, so the box and every chord grow as `DP^(1/3)` (e.g., a pair that needed DP 24 in the DP-20 box needed DP 25 once rebuilt at DP 24). Rebuilding at the same *box* size instead (i.e., raising the density with the DP) reaches the named DP in one step.

**Reading a delivered winding.** `route_designed_pairs` checks each pair with the far-closed linking number of the two paths as drawn. That reading is not an invariant, since it closes each strand through legs to a distant point, and a strand that moves through the partner's legs changes it with no passage. The reading that holds is the linking of two network cycles, one through each strand, against the same cycles with the braid untwisted (`topon.analysis.windings`, see *Did a designed pair get its winding?* under `simulation`).

```python
from topon.analysis.windings import designed_windings, strands_from_placement
from topon.conformation.entanglement import route_designed_pairs, unwound_paths

report = route_designed_pairs(placement, requests)
pairs = {(o.request.chain_a, o.request.chain_b): o.request.windings
         for o in report.accepted}
for w in designed_windings(strands_from_placement(placement), pairs,
                           placement.box, reference=unwound_paths(placement, report)):
    print(w.pair, w.requested, w.value, w.delivered)   # value is None when cycle pairs disagree
```

`unwound_paths(placement, report)` takes each braid's phase rotation out in place, so the reference differs from the build only inside the braids, and `method="redraw"` draws the strand again from its chord at zero turns.

### `output`

| Key | Default | Description |
|---|---|---|
| `lammps_data` | `true` | Write LAMMPS data file (not read by `Pipeline`, see below) |
| `lammps_convention` | `"topon"` | Coarse-grained data-file convention (see below) |
| `lammps_inputs` | `true` | Write LAMMPS input scripts (not read by `Pipeline`, see below) |
| `visualization` | `true` | Write HTML visualization (not read by `Pipeline`, see below) |
| `analysis_report` | `true` | Write analysis report (not read by `Pipeline`, see below) |
| `save_attributed_graph` | `true` | Save attributed graph as `.gpickle` (not read by `Pipeline`, see below) |
| `export_graphml` | `false` | Write the graph (chains and entanglement edges) as `<study.name>.graphml` in the study folder. `topon generate --export-graphml` sets it |
| `export_npz` | `false` | Write the graph as `<study.name>.npz` in the study folder. `topon generate --export-npz` sets it |

`lammps_data`, `lammps_inputs`, `visualization`, `analysis_report` and
`save_attributed_graph` are accepted, but `Pipeline` does not read them. It
always writes the data file and the input scripts, and it writes no HTML
visualization, analysis-report file or `.gpickle`.

#### The NPZ dual graph (`export_npz`)

The NPZ file holds one network as a dual graph, with a node per strand (`type` 0) and per crosslinker (`type` 1), strand rows first. `edge_index` holds 0-based rows in both directions, and `edge_type` is 0 for a chemical edge (strand to crosslinker) and 1 for an entanglement (strand to strand). topon writes schema 2 and stamps it (`schema_version`), with ten feature columns (`type, length (DP), contour_length, rg, COMX, COMY, COMZ, chem_degree, phys_degree, frac_ext`). The conformation columns are NaN until a LAMMPS run fills them.

`topon.topology.loader.read_npz(path)` returns the arrays in schema 2, and `load_npz(path)` rebuilds the topon graph (junctions as nodes, strands as edges). Both check the schema first. A stamp other than 1 or 2, or a feature width that does not match it, is refused. A file with no stamp and eight columns is schema 1, the layout of earlier `fix bond/create` datasets, and is upgraded on the way in.

| Schema 1 column | Schema 2 |
|---|---|
| `type`, `rg`, `COMX/Y/Z`, `chem_degree` | copied |
| `n_interior` (18 for a DP-20 strand) | `length` = n_interior + 2 on strand rows (the DP counts the two chain ends) |
| `ree2` (mean square end-to-end distance) | kept as a separate `ree2` array, with `contour_length` and `frac_ext` NaN |
| (none) | `phys_degree`, counted from the `edge_type` 1 edges |

The `n_polymer` and `n_crosslinker` values of a schema-1 file are not the row counts, so both are recounted from `type`. `load_npz` skips the strands that do not join two crosslinkers (dangling and sol chains), since the graph has no node for a free chain end, and keeps one entanglement partner per strand.

#### `output.lammps_convention`

| Value | Atom types | Molecules |
|---|---|---|
| `"topon"` (default) | The distinct `bead_type` properties, numbered in sorted order | One for the whole network |
| `"endlinked"` | 1 = chain-end bead, 2 = chain interior, 3 = junction | One per chain and one per junction. A dangling chain's free end belongs to its chain, and a primary loop is a ring whose two ends bond to the same junction |

The end-linked convention is the one `fix bond/create` datasets use, so the
same parsers read topon output and reference output. Use it with
`assignment.dp_distribution.endlinked_dangling` when the bead count must
also match. Atomistic output ignores the key.

For a bead-spring build that skips the chemistry stage,
`topon.writers.write_endlinked` writes the same convention directly from a
`topon.conformation.place` placement, with image flags that reconstruct the
coordinates it was given. It writes the chains the pipeline writes for the
same graph, since `place` reads an edge's `dp` as the chemistry builder does
(the beads between the strand's two nodes, so a dangling chain is `dp` plus
its free end) and places the sol chains of `G.graph["sol_chains"]` as free
walks drawn after every other strand, which the writer puts last, one
molecule each. A graph with no edge `dp` takes the `dp` argument as the
chain's DP, the free end among its beads.

### `analysis`

This section is read by `topon analyze --config`, `topon inspect --config` and `topon track --config`, and the pipeline ignores it. Its one block, `analysis.z1plus`, says where Z1+ is (§3.6). Z1+ is not distributed with topon.

```json
"analysis": {
  "z1plus": {"executable": "~/z1/Z1+", "wsl": "auto", "wsl_distro": null}
}
```

| Key | Default | Description |
|---|---|---|
| `z1plus.executable` | `"~/z1/Z1+"` | Path to the Z1+ binary, a Linux path inside WSL when it runs there. A leading `~/` is the home directory of the account it runs under |
| `z1plus.wsl` | `"auto"` | Run Z1+ inside WSL. `auto` does so on Windows and not elsewhere, and the other values are `always` and `never` |
| `z1plus.wsl_distro` | `null` | WSL distribution to run in. `null` is the default one |
| `z1plus.timeout` | `7200` | Seconds one Z1+ run may take |

Unknown keys are refused, so a misspelled `executable` fails to load rather than pointing at the default.

### `simulation` (raw section, not schema-validated)

This section controls the LAMMPS scripts written by stage 6.

| Key | Default | Description |
|---|---|---|
| `protocol` | `"pushoff"` | Coarse-grained relaxation protocol (see below) |
| `atomistic_protocol` | `"hard_backbone"` on a settled build, else `"soft_push"` | Atomistic relaxation (DREIDING or CHARMM). `"hard_backbone"` never lets the backbone go soft, and `"soft_push"` is the earlier deck, kept selectable and prone to crossings (see *Atomistic stages and gates* below). The default is the one `Pipeline` sets, while the workflow route and a direct `LammpsInputGenerator` keep `"soft_push"` |
| `backbone_dump_every` | `10` on a settled build, else `0` | Atomistic stages only. Dump the backbone atom types every N steps (minimizer iterations included) into `traj_stage<k>.lammpstrj` with unwrapped coordinates, for the crossing detector. 0 writes no dump |
| `rho_final` | `null` | Number density (beads per sigma³) stage 4 compresses to. `null` makes stage 4 a settle. This is not `chemistry.target_density`, which sizes the build box |
| `final_bond_style` | `"fene"` | `"quartic"` adds stage 6, for deformation runs |
| `pair_style` | `"attractive"` | `lj/cut 2.5`, or `"repulsive"` for WCA. Ignored by `pushoff`, which pins WCA |
| `include_angles` | `true` | Write angles in the data file |
| `remove_cg_angles` | `true` | Drop them before the run (Kremer-Grest chains are flexible) |

#### `simulation.protocol`

| Value | Stage 1 | Scripts | Keeps a prescribed entanglement? |
|---|---|---|---|
| `"pushoff"` (default) | FENE + WCA, `nve/limit` 0.02, no minimizer | 5 | Yes |
| `"hardcore_min"` | WCA, conjugate-gradient minimization | 3 | Yes for the push, no for the minimizer |
| `"soft_push"` | `pair_style soft` ramped 0 to 30, minimization | 3 | No |

`pushoff` is the default because a minimizer resolves an overlap by
whatever move lowers the energy. With a harmonic bond the cheapest move is
often to stretch a bond and let the overlapping bead pass through it. The
threaded strands then cross, and Z per bridge drifts upward in every later
stage (from 0.19 to 0.24 on the DP-20 reference build). Under the push-off
the same build keeps Z constant within the Z1+ noise.

The other two protocols are for reproducing earlier runs and are prone to
chain crossings.

The five `pushoff` stages and the file each one writes are listed below.

| script | what it does | writes |
|---|---|---|
| `minimize_1_serial.in` | push-off, `nve/limit` 0.02, dt 0.002, 30k steps | `stage1_min.data` |
| `minimize_2_parallel.in` | cap to 0.05 for 20k, then free NVE 20k | `stage2_pushoff.data` |
| `minimize_3_parallel.in` | equilibrate at the build density, damp 10, 200k | `stage3_build_equil.data` |
| `deform_4_parallel.in` | affine compression to `rho_final`, then settle | `stage4_final_T1.data` |
| `quench_5_parallel.in` | T 1.0 → 0.4 over 50k, settle 20k | `stage5_final_quench.data` |

Stages 1-3 keep their old script names, although nothing in this protocol
minimizes, so a caller that runs `minimize_1_serial.in` still works. The
data-file names follow the end-linked dataset convention, so the same
analysis scripts read a topon run and a reference run. Use
`LammpsInputGenerator.stages("cg")` to get the list instead of
hard-coding it.

Step counts are set under `experimental.cg.pushoff`, with one block per stage.

```json
{"experimental": {"cg": {"pushoff": {
  "seed": 12345,
  "stage1": {"timestep": 0.002, "limit": 0.02, "steps": 30000, "tdamp": 1.0},
  "stage2": {"timestep": 0.005, "limit": 0.05, "steps": 20000, "free_steps": 20000},
  "stage3": {"steps": 200000, "tdamp": 10.0},
  "stage4": {"deform_steps": 150000, "settle_steps": 100000},
  "stage5": {"ramp_steps": 50000, "settle_steps": 20000, "tdamp": 10.0}
}}}}
```

`experimental.cg.dynamics.run_steps` does **not** apply to `pushoff`. Its
equilibration length is `stage3.steps`, which defaults to 200 000 instead
of that key's 10 000.

#### Acceptance gates

`topon.simulation.protocols` reads every checkpoint and applies two gates.

```python
from topon.simulation.protocols import StagedRun, measure_stages

run = StagedRun(sim_dir=sim, stages=gen.stages("cg"), omp=8)
report = run.run()            # gates after every stage; stops at the first failure
print(report.render())

measure_stages(sim, gen.stages("cg"))    # or gate a run that already happened
```

- No bond may exceed 1.2 σ after stage 2 or at any later stage, and every
  bond is checked. `mode="instant"` (the default) fails on any such bond.
  `mode="persistent"` fails only on a bond that is long at more than one
  stage, which separates a threaded bond from a thermal excursion. Both
  counts are always reported. Persistence is judged over every stage
  measured, stage 1 included (stage 1 itself cannot fail a run), and each
  long bond (up to 8 per stage) is described by the nearest bead of another
  molecule, its index along its own strand and its distance from the bond's
  midpoint, which tells a threaded bond from one pinched at a junction.
- Z1+ must not change between stage 3 and stage 5. This gate applies only
  to a run that did not compress and has a Z1+ measurement. Under
  compression Z1+ moves by 10-20 % with no crossing anywhere, because it
  counts the kinks of the shortest paths between the *current* junction
  positions and part of it slides off as the junctions move. Z1+ is
  therefore always reported per stage but gated only where it must hold.
  `gate_z=True` / `False` overrides this.

  The ±0.01 tolerance is absolute and suits a build of about 4 427 bridges,
  where the counting noise on the mean Z is ≈ 0.007. On a smaller cell it
  is tighter than the noise (≈ 0.04 at 192 bridges), so pass a
  `z_tolerance=` scaled to the build. Otherwise a small system can fail the
  gate without any crossing.

Temperature is read from each checkpoint's velocities, because Z1+ and the
chain statistics both depend on it. A comparison across a temperature
difference would measure that difference. Z1+ is not included in this
repository (its license forbids redistribution), so pass its results in
with `z_by_stage=`.

#### Atomistic stages and gates

On the `Pipeline` route (`topon generate`, `Pipeline`) an atomistic build is placed with the settled meander (`conformation.atomistic_placement`) and relaxed on the hard-backbone stages (`simulation.atomistic_protocol: "hard_backbone"`), with its backbone dumped every 10 steps, so every run can be checked for passages. Through stages 1 and 2 the backbone-backbone pairs (Si3 and O_3 for PDMS, taken from the strand record) keep a fixed soft core (60 kcal/mol out to 3 Å) in stage 1 and are at full depth from the first step of the ramp. Only pairs with a methyl C or an H are soft and ramped (each to its own depth, `fix adapt ... scale yes`), and both stages are capped under a 300 K Langevin thermostat. Since 0.4.5 the DREIDING deck holds the resonant types (an aromatic ring's `C_R`, `N_R` and `O_R`) hard with the backbone in both stages, so no backbone bond passes through a ring and no ring through another (a methyl's or a hydrogen's bond still can), and a system with no such atom writes the deck it did. The backbone dump keeps the backbone's types. A ring of atoms typed as a chain's (a cyclohexyl's `C_3`) is not made hard, since its type is a methyl's too. Stage 3 is the earlier one, with its minimization capped. The parameters sit under `experimental.atomistic.hard_backbone` (`core` 60 kcal/mol, `core_cutoff` 3 Å, `light_cutoff` 1 Å, `light_max` 30, `temperature` 300 K, `tdamp` 100 fs, `stage1.steps` 5 000, `stage1.limit` 0.05 Å, `stage2.steps` 2 500, `stage2.limit` 0.1 Å). The ramp and stage 3's minimization were halved in 0.4.5, which kept Z, density and passages on the test networks for about 40 % less LAMMPS time. `experimental.atomistic.dynamics.run_steps` overrides `stage2.steps` when it is given, and `experimental.atomistic.stage3.minimize` sets stage 3's minimization as `"etol ftol maxiter maxeval"` (default `"1.0e-6 1.0e-8 1000 10000"` on the hard-backbone stages, `"1.0e-8 1.0e-10 10000000 100000000"` for DREIDING and `"1.0e-6 1.0e-8 100000 1000000"` for CHARMM on the earlier ones). On the CHARMM route the same two stages are written in CHARMM styles, stage 1 with the `.soft` settings (bonded terms, 1-4 weights 0) and stage 2 as `lj/cut/coul/long` with the `.lj` settings and arithmetic mixing. The hard-backbone stages expect `conformation.atomistic_placement`, and the pipeline warns when they meet the earlier placement, whose backbone atoms start a third of a bond apart.

`atomistic_placement: null` gives the build and stages of earlier versions (`soft_push`), and the workflow route (`topon.workflows.atomistic_network`) and a direct `LammpsInputGenerator` keep them too. The earlier stages are prone to strands passing through each other. On a DP-30 PDMS network the `soft_push` stages let hundreds of backbone bonds pass through each other in stage 1, even from a settled build, while the hard-backbone stages on the settled build let none through at any stage. Both ramps now scale each pair's own depth (before 0.4.0 the DREIDING ramp ran every pair's well from 0.001 to 1 kcal/mol whatever its DREIDING depth).

The dumps are large (about 200 MB a stage on a 36,248-atom network). `AtomisticRun` gzips each one once it has read it (`compress_dumps=False` keeps them as LAMMPS wrote them), and the crossing detector reads either.

`topon.simulation.protocols.atomistic` gates the three atomistic stages, in atomistic units and through the run's strand record.

```python
from topon.simulation.protocols.atomistic import AtomisticRun, measure_run

report = AtomisticRun(run_dir, omp=4).run()   # the three stages, gating after each
print(report.render())
measure_run(run_dir)                          # or gate the checkpoints already there
```

- No backbone bond more than 15 % over its r0 at two stages, junction to junction along every strand, from stage 1 on (`mode="persistent"`, the atomistic default, while `mode="instant"` fails on one). A single checkpoint can be hot, so one long bond at one stage does not fail the run. The build is reported and not gated, and is not counted towards persistence.
- No backbone bond passes through another, at any stage, stage 1 included. This is read from each stage's backbone dump with the crossing detector below (`simulation.backbone_dump_every`). One passage fails the stage, and the report gives the count by kind, the step of the first, its two bonds and their strands. `AtomisticRun` reads each stage's dump as soon as the stage ends and, with `stop_on_fail`, stops there. A designed entanglement (`assignment.entanglements`, whose partner strand the strand record names) is kept if no bond of either of its strands passed through the other, and the report says so pair by pair.
- Every designed pair's network-cycle linking holds, read from the checkpoint files themselves with no dump needed. On the first checkpoint read each pair gets up to four pairs of disjoint cycles, one through each of its strands (`topon.analysis.windings`), and every later checkpoint is read on the same cycles. The linking of two disjoint closed curves changes only when a bond of one passes through the other, so a change fails the run and names the pair, the two readings and the checkpoint. A pair with no two disjoint cycles that close in space (possible on a very small cell) is noted and not read. `report.designed_linking` holds the readings.
- Backbone Z1+ per bridge is measured at every checkpoint over `z1_seeds` seeds of the exporter's junction jitter (4 by default) and reported as their mean and spread. A partner pair counts as new or lost only when it is found at every seed of one checkpoint and at none of the other. When any stage is dumped, Z1+ is not gated, since it moves as chains settle with nothing crossing and so cannot certify a state. Without dumps the earlier Z gate runs instead (Z per bridge held from the end of the epsilon ramp at every checkpoint at the same density, with a tolerance of 1.5 / n_bridges and the bead-spring 0.01 as the floor). A gated checkpoint with no backbone bond read against an r0 fails rather than passing empty.

The checkpoints are `stage0_build` (`03_Conformation/system_relaxed.data`), `stage1_soft`, `stage2_ramp`, `stage3_min`, `stage3_nvt` and `stage3_npt`. Density is in g/cm³, and temperature is in K from the velocities with real masses. Z1+ runs when it is installed and is skipped otherwise.

A single Z1+ run cannot be read pair by pair. The exporter moves each junction by 1e-3 Å so that Z1+ does not crash on shared points, and another seed for that move changes 10 to 15 of 55 partner pairs on an identical DP-30 configuration. Compare Z1+ between checkpoints averaged over several seeds (`topon.analysis.z1plus.measure_seeds`, which the gates use), and the pairs present in every seed.

#### Watching a relaxation (`topon track`)

```bash
topon track output/pdms_dp30
topon track runs/z1 runs/z2 --label "Z = 1" --label "Z = 2" -o z.html --omp 4
```

The command writes one self-contained HTML page (`relaxation_tracker.html` by default) for one or more atomistic study folders. For each run and each checkpoint (the build, stage 1, the ramp, and the minimized, NVT and NPT states of stage 3) it shows the network as Z1+ reads it. Every strand is drawn as a curve from junction through one point per repeat unit to junction, with the crosslinks, the Z1+ kinks and the primitive paths, in a view that rotates and zooms and names a strand's entanglement partners when it is hovered. Beside it, through the stages, are Z per bridge over `--seeds` Z1+ seeds (8 by default, and 0 leaves Z1+ out) with its spread, how many of the build's robust pairs are still seen, the backbone passages of each stage dump, the energy density under the full force field with its bonded, van der Waals and Coulomb parts, temperature, density and the longest backbone bond. Below, Z over its build value is plotted for every run in the page, with the LAMMPS time of each. The energy is every checkpoint evaluated again with a zero-step LAMMPS run under the run's own stage-3 styles and settings (`--lmp`, `--omp`, and `--no-energies` skips it), so stages 1 and 2 are not read under their soft or ramped potentials. The build's energy is very high, since only its backbone is settled, so the charts start after it. A run that stopped early shows the checkpoints it wrote. The page loads nothing from anywhere (its fonts are local font stacks) and nothing is uploaded.

#### Did a strand pass through another? The crossing detector

Z1+ cannot say whether an entanglement state survived, because it moves as chains settle with nothing crossing. `topon.analysis.crossings` watches for the one event that changes topology, two backbone bonds passing through each other between two frames of a trajectory. Every atom is taken to move in a straight line between frames, and a passage is a moment at which the four atoms of the two bonds are coplanar and the two segments meet. Bonds that share an atom are never a pair, and a passage is classed `self` (one strand), `shared` (two strands that end on the same junction, which can wind round each other there) or `apart`.

```python
from topon.analysis.crossings import crossings_of_run

for stage, rep in crossings_of_run(run_dir).items():   # traj_stage1..3 in 04_Simulation
    print(stage, rep.transitions, rep.max_step, rep.counts())
```

It needs `simulation.backbone_dump_every` (10 keeps an atom within about 1 Å between frames in every stage). Coordinates are unwrapped per atom and each bond is taken in its shortest image, since topon's data files carry no image flags. As a check, a 3x3x3 DP-6 network run with no pair interaction at all at 2000 K for 20 ps gives 489 passages, and the same network through either relaxation deck gives none.

#### Did a designed pair get its winding? The network-cycle reading

A designed pair is two open strands wound about each other, each held at two junctions, and no reading of two open strands is a topological invariant, since any closure that is not made of strands can be crossed by a strand without a passage. `topon.analysis.windings` therefore closes each strand through the network. For a pair (A, B) it takes a cycle of strands through A and one through B that share no junction and close in space (their bond vectors, in the minimum image, sum to zero, so a cycle that winds round the periodic box is skipped), and computes their linking number exactly, from the solid angle of every pair of segments. It changes exactly when a bond of one cycle passes through a bond of the other, by one up or down, so junctions may move and the rest of the network may go anywhere. A build settled with no passage, or a relaxation in which the crossing detector finds none, keeps it.

The two cycles link through their other strands too, so the delivered winding is read against a reference, the same build with the pair's winding taken out (`delivered = Lk - Lk_ref`, both integers, the reference computed once on the build). Up to four cycle pairs are read per designed pair, the shortest first and then pairs that leave the designed strands' junctions by strands not used yet. The braid adds its winding to all of them, and a third strand running through the braid changes only some, so `PairWinding.value` is the delivered count when every cycle pair agrees and None when they do not (`delivered` has them all). Each pair is read against a reference with only its own two strands unwound. The build is read after the settle has parted the bonds that were drawn touching, since a cycle bond drawn exactly on another has no linking number until one side is chosen. The sign of a delivered count says how the two strands are labeled (antiparallel pairs read -1), not the hand.

On the atomistic route the reference is the pair drawn by its waypoint construction at zero turns over the same span.

```python
from topon.analysis.atomistic import load_strand_record
from topon.analysis.crossings import designed_pairs
from topon.analysis.windings import (designed_windings, record_reference,
                                     strands_from_record)
from topon.conformation.entanglement.realize import entangled_backbone_paths

record, _ = load_strand_record(run_dir / "manifest.json")
pairs = designed_pairs(record)                       # {(k, l): windings}
unwound = entangled_backbone_paths(pipe.graph, pipe.dims,
                                   pipe._builder.edge_backbone_path, windings=0)
ref = record_reference(record, parted_pos, box,      # the build after the settle's parting, in Å
                       {e: np.asarray(p) * scale for e, p in unwound.items()})
for w in designed_windings(strands_from_record(record, parted_pos, box), pairs,
                           box, reference=ref):
    print(w.pair, w.requested, w.value)
```

#### Deformation runs

For a quartic bond in the final force field, set
`simulation.final_bond_style: "quartic"`. The switch happens only at the
end (20k steps at the target temperature after the FENE relaxation),
because the quartic bond breaks without warning above 1.5 σ, while FENE
stops with an error. When the OpenMP package is active, the generated stage
wraps the style in `suffix off` / `suffix on`. `bond_style quartic/omp`
computes correct forces but leaves the subtracted bonded-LJ term out of
`E_pair` and the virial (e.g., pressure 4.79 instead of 0.05 at ρ = 0.30).
Any stress read from a run with the OpenMP package must still be checked
against a serial `run 0`.

`fix deform ... erate` resets its reference box at every `run` command, so
a deformation split across several runs applies the same factor once per
run. Stage 4 drives the box to an absolute `final` size in a single `run`
and avoids this. topon does not generate a *tensile* stage. A hand-written
one that uses `erate` must account for this (e.g., by running the whole
deformation in a single `run`).
