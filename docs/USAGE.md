# topon usage guide

This guide covers running topon end-to-end. It documents the CLI flags, the sub-system APIs, recipes for common systems and, in Appendix A, the JSON config schema.

[ARCHITECTURE.md](ARCHITECTURE.md) describes the package layout and design.

---

## 1. Install

```bash
pip install -e .
```

The runtime dependencies (installed from `pyproject.toml`) are `numpy`, `networkx`, `pandas`, `rdkit`, `pydantic`, `click`, `plotly` and `scipy`. LAMMPS (`lmp`) needs to be on `PATH` only to *run* the generated systems. Generation does not need it.

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

Ready-to-use configs are in `demos/templates/` (`minimal.json`, `full.json`) and in the demo folders `demos/polymer/` and `demos/poss/`. [`demos/README.md`](../demos/README.md) lists them all.

---

## 3. CLI reference

The package installs a `topon` console script that dispatches to several sub-commands. All sub-commands accept `--help`.

```
topon [--version] [--help] <command> [options]
```

### 3.1 `topon generate` (run the full pipeline)

```bash
topon generate CONFIG_PATH [--output DIR] [--dry-run] [--export-graphml] [--export-npz]
```

| Argument / Option | Description |
|---|---|
| `CONFIG_PATH` | Path to the JSON config file (required) |
| `--output`, `-o` | Override `study.output_dir` from config |
| `--dry-run` | Validate the config and exit without running |
| `--export-graphml` | Also write the graph (chains and entanglement edges) as `<study.name>.graphml` in the study folder. Same as `output.export_graphml` |
| `--export-npz` | Also write the graph as `<study.name>.npz` for graph-learning pipelines. Same as `output.export_npz` |

The command runs the six-stage pipeline (Topology → Analysis → Assignment → Chemistry → Conformation → Output) and writes the LAMMPS data files and input scripts to `output_dir/study_name/`.

```bash
topon generate demos/templates/full.json
topon generate demos/templates/full.json --output ./my_run
topon generate demos/templates/full.json --dry-run
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
| `--preset` | `atomistic_pdms` | One of `atomistic_pdms`, `cg_kg`, `poss`. Each copies a demo `config.json` from the source tree (`demos/polymer/atomistic/basic/`, `demos/polymer/coarse_grained/basic/`, `demos/poss/`), which `topon init` finds by walking up from the installed package, so the presets need the editable install of §1. |
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
| `deprecated_mix_cutoff` | warn | `mix_cutoff` still present (it loads as `neighbour_cutoff`) |
| `unknown_config_keys` | warn | A nested section carries a key the schema does not define, so it is ignored without notice |
| `unknown_node_type` | warn | `assignment.node_types.degree.mapping` references a type that is not in `chemistry.node_type_map` (it would fall through to Si without notice) |
| `poss_at_internal_junction` | warn | POSS mapped to degree >= 2 junctions, which gives a bond longer than half the periodic box at LAMMPS stage 1 |
| `atomistic_graft_non_pdms` | warn | Graft density set on a non-PDMS atomistic monomer (the build skips those grafts with only a `RuntimeWarning`) |
| `dp_below_kuhn` | warn | DP < 5 (conformation/entanglement edge cases) |
| `defects_endcap_safe` | ok | Reminder that loop defects skip degree-1 chain caps |
| `entanglement_target_below_floor` | warn | `conformation.entanglement.target_Z` is below the lowest value that route has reached at this DP. The warning names the floor and the route that goes lower. Each controller round is a full relaxation protocol, so this is worth catching before it starts |
| `conformation_build_knob` | error | A `target_Z` with neither `coil_ratio` nor `build_density`, and nothing measured for that DP and placement to seed the controller from |
| `schema_gap_extras` | ok | Config has `simulation`/`execution` (not Pydantic-validated, the CLI handles them through `load_config_full`) |

To add a rule, write `check_<name>(cfg, raw) -> list[Issue]` in `topon/diagnostics/rules.py` and append it to `RULE_REGISTRY`.

### 3.3b `topon inspect <run_dir>` (post-run summary)

```bash
topon inspect runs/my_study
```

The command summarizes a finished run, so there is no need to read the `system.data` headers by hand. `RUN_DIR` is the study folder or its parent. The command reads each stage directory (`02_Chemistry/`, `03_Conformation/`, `04_Simulation/` in the Pipeline layout) or a flat folder with every file at the top level, and prints the following.


- atom count, atom-type count, box dimensions
- what the topology stage was asked for against what it produced, from the run manifest
- per-stage status (which files landed, what they say)
- the next LAMMPS commands to run

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
`topon.core.manifest.read_manifest(run_dir)`.

### 3.3c `topon recipes` (common use cases)

```bash
topon recipes
```

The command prints a short table that maps common tasks to commands for all sub-systems (polymer networks through Pipeline, simbox, single-chain, the batch topology demo, `inspect` and `analyze`). The rows are defined in `topon/cli.py:recipes()`.

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
├── groups.txt            # group definitions by reactive-group type
├── 1_minimize.in         # Stage 1: soft push-off + CG minimisation
├── 2_nvt.in              # Stage 2: NVT thermalisation
├── 3_npt.in              # Stage 3: NPT density equilibration
└── 4b_crosslink.in       # Stage 4: crosslink template (fix bond/react or bond/create)
```

Then run LAMMPS.

```bash
cd simbox_output && lmp -in 1_minimize.in
```

See §4.1 for the simbox Python API.

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
| `--graft-smiles` | `None` | SMILES for graft repeat unit (required if `--graft-density > 0`) |
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

The output directory contains the files below.

```
chain_output/
├── system.data       # LAMMPS data (chain + solvent, DREIDING)
├── ff_coeffs.in
├── settings.in       # pair coefficients (re-applied after soft push-off)
├── groups.txt
├── 1_minimize.in
├── 2_nvt.in
└── 3_npt.in
```

### 3.6 `topon analyze` (graph statistics)

The command analyzes a topology graph file and prints statistics.

```bash
topon analyze GRAPH_PATH [--format text|json] [--nodes NODES_PATH]
```

| Argument / Option | Description |
|---|---|
| `GRAPH_PATH` | Path to `.gpickle`, `.nodes`, or `.edges` file |
| `--format`, `-f` | Output format, `text` (default) or `json` |
| `--nodes` | Companion `.nodes` file for a `.edges` `GRAPH_PATH`. Without it, the `.nodes` file with the same stem is used |

```bash
topon analyze network.gpickle
topon analyze network.nodes
topon analyze network.edges --nodes network.nodes
topon analyze network.gpickle --format json
```

The CLI calls `topon.analysis.report.analyze_graph()` and prints the degree distribution, connectivity and topology statistics. The same function can be imported in Python.

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

`MoleculeLibrary` (`topon/simbox/library.py`) holds pre-built siloxane molecules.

```python
from topon.simbox.library import MoleculeLibrary
lib = MoleculeLibrary()
epoxy  = lib.epoxy_pdms(n_dms=2)    # Glycidoxypropyl-PDMS, ~500 g/mol
amino  = lib.amino_pdms(n_dms=8)    # Aminopropyl-PDMS, ~850 g/mol
poss   = lib.am0270_poss()           # AminopropylIsooctyl POSS, ~1267 g/mol
custom = lib.custom("C1OC1", name="MyEpoxide")
```

The library structures are as follows.
- Epoxy-PDMS is `Epoxide-CH₂-O-CH₂CH₂CH₂-Si(Me)-[O-Si(Me)₂]ₙ-O-Si(Me)-CH₂CH₂CH₂-O-CH₂-Epoxide`.
- Amino-PDMS is `H₂N-CH₂CH₂CH₂-Si(Me)-[O-Si(Me)₂]ₙ-O-Si(Me)-CH₂CH₂CH₂-NH₂`.
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

write_lammps(system, output_dir="output/simbox")
write_inputs(system, output_dir="output/simbox", temperature=300.0, pressure=1.0)
```

Stage 1 (soft push-off and minimization) has two phases. Phase A uses `pair_style soft` with a prefactor ramped from 0 to 60 and a brief NVT run to resolve overlaps. Phase B switches to the `lj/cut` DREIDING potentials and runs a conjugate-gradient minimization.

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

`run_workflow` also activates `UniversalTypeMapper`, a context manager that patches `topon.forcefield.dreiding` at write time. It keeps the DREIDING type IDs the same across all compositions, so predefined `fix bond/react` templates stay compatible.

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
# Edit 4b_crosslink.in to choose Option A (fix bond/react) or B (fix bond/create), then:
lmp -in 4b_crosslink.in
```

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

# topon simbox equivalent
from topon.simbox.workflow import run_workflow
run_workflow("simbox_output", n_epoxy=600, n_amino=300, n_poss=0,
             density=0.85, seed=42)

# topon chain equivalent
from topon.singlechain.workflow import run_workflow as chain_workflow
chain_workflow("chain_output", chain_smiles="[Si](C)(C)O", dp=20,
               solvent_smiles="Cc1ccccc1", n_solvent=200,
               density=0.85, seed=42)
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
| `demos/workflows/batch_polymer_topology/run.py` | Generates 25 seeded lattice graphs, exports each as `.nodes`/`.edges`, GraphML and NPZ, and writes one CSV of per-graph properties |

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
  "output":       { ... }
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
| `source` | `"generate"` \| `"load"` | `"load"` | Generate a new topology or load an existing one |
| `generator` | object | - | Settings for the C / Python generator (when `source="generate"`) |
| `existing_files` | object | - | File paths (when `source="load"`) |

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

`topology.generator` refuses unknown keys, as do the per-class blocks under
`assignment.defects` and the whole `conformation` section. Other sections
drop an unrecognized key without notice. There is no `seed` key under
`topology.generator`, so `"seed": 7` there is refused. To make a generated
graph reproducible, seed the global random streams before generating.

```python
import random, numpy as np
random.seed(42); np.random.seed(42)          # then build
```

`topon.workflows.cg_network.run(seed=...)` does this itself. `Pipeline(config)`
does not, so a direct API build needs the two lines above. In the other
sections, `topon doctor` reports ignored keys under the `unknown_config_keys`
rule instead of refusing to load, because a config may carry parameters for
other tools under `topology`.

In the degree distribution, `"d:N"` requires N nodes of degree d, `"e:N"` requires N edges in total, and omitted degrees are unconstrained (e.g., `"0:15,1:30,e:371"`).

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
> **`gradient` is broken.** It ignores the requested composition and always
> gives a 50:50 split (e.g., a request for A=0.1 still gives A=0.50). For two
> monomers at equal fractions and even DP it is byte-identical to `block`.

### `chemistry`

| Key | Type | Default | Description |
|---|---|---|---|
| `model_type` | `"coarse_grained"` \| `"atomistic"` | `"coarse_grained"` | Force-field resolution |
| `target_density` | float | `0.9` | Target density, in g/cm³ on the atomistic route and beads per sigma³ on the coarse-grained one (the build box is sized from it) |

#### `chemistry.node_type_map`

```json
"node_type_map": {
  "end": { "molecule": "[Si](C)(C)C", "is_end_cap": true },
  "A":   { "molecule": "Si",          "is_end_cap": false },
  "B":   { "molecule": "POSS",        "is_end_cap": false }
}
```

The built-in molecule names are `"Si"`, `"POSS"` (Si₈O₁₂ cage) and `"POSS_AM0270"` (AM0270 aminopropyl POSS). Any SMILES string is also accepted.

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

#### `chemistry.connection`

```json
"connection": { "auto_bridge": true, "default_bridge_atom": "O" }
```

With `auto_bridge`, a bridge atom is inserted automatically when the chain head and the node atom are the same element (e.g., both Si). Set it to `false` to always use direct bonds.

### `conformation`

This section configures stage 5. It holds two unrelated sets of keys, and a config uses one or the other.

The first three keys drive `ConformationManager`, which rewrites a data file that already has coordinates (the atomistic and legacy CG route).

| Key | Type | Default | Description |
|---|---|---|---|
| `overlap_cutoff` | float | `0.01` | Separation below which two atoms are pushed apart, in the data file's own units |
| `overlap_max_iters` | int | `10` | Passes of the overlap resolver before it gives up |
| `noise_magnitude` | float | `1e-4` | Uniform jitter on every atom, to break lattice degeneracy |

The other keys drive `topon.conformation.place`, which draws a bead-spring build from the graph itself.

| Key | Type | Default | Description |
|---|---|---|---|
| `placement` | `"straight"` \| `"meander"` \| `"walk"` | `"meander"` | Chain shape at build |
| `coil_ratio` | float \| null | `null` | Strand contour over chord at build, mean over mean |
| `build_density` | float \| null | `null` | Bead density the paths are drawn at |
| `bond` | float | `0.97` | Design bond length, in sigma |
| `meander_waves` | float | `6.0` | Waves the meander spends its slack in, before the self-fold gate halves it |
| `min_bond` | float | `0.85` | Shortest bond a placed strand may carry |
| `min_self_separation` | float \| null | `null` | Closest a bead may come to a non-adjacent bead of its own chain. `null` uses the route's own floor (1.0 sigma for a drawn path, 0.05 for a walk) |
| `path_jitter` | float | `0.02` | Gaussian jitter on the interior beads of a straight path |
| `junction_shell_spacing` | float \| null | `null` | Seat the first bead of every chain leaving a junction on a spread shell at least this far apart |
| `entanglement` | object | see below | Target, distribution, designed pairs and the controller |

`coil_ratio` and `build_density` set the same thing, and setting both is an error. The box scales as `rho^(-1/3)` and the contour does not change, so `coil_ratio = contour / chord` scales as `rho^(1/3)`. On a given graph at a given DP, fixing one fixes the other. `place` takes exactly one, and the controller converts the one given into the one it adjusts.

#### Choosing the chain shape

The chain shape matters more than the build density. The table gives the
mean Z1+ per strand for each placement on the DP-20 end-linked reference
graph.

| placement | build density | Z1+ per DP-20 strand, build state | after compression to rho 0.3075 |
|---|---|---|---|
| reference (`fix bond/create`) | -- | -- | **0.178** |
| `walk` | 0.145 | 0.300 | 0.335 |
| `walk` | 0.095 | 0.250 | 0.288 |
| `walk` | 0.035 | 0.232 | 0.262 |
| `meander` | 0.050 | **0.189** (KS p = 1.0 vs reference) | 0.239 |
| `meander` | 0.040 | 0.192 | 0.224 |

The last column mixes two relaxation protocols. The `walk` rows used the
older minimizer protocol, which adds crossings of its own, so part of their
excess Z comes from the protocol. The `meander` rows used the crossing-free
protocol, re-quenched to T = 0.4.
`CALIBRATION` labels every row with its protocol (the crossing-free rows as
`limit`), and only the crossing-free rows steer the controller.

A random walk bottoms out near Z = 0.23 per DP-20 strand at any build
density, because coiled chains collapse during the push-off at fixed
volume and trap crossings. The meander at the same state reproduces the
reference per-strand distribution. The shape sets the range of Z, and the
density moves Z within that range.

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
| `target_Z` | float \| null | `null` | Mean Z1+ per strand **at the final state**, or null for no target |
| `close_on` | `"final"` \| `"build"` | `"final"` | Which state the controller compares against. `"build"` is cheaper to reach and is right only when the build state is itself what is being matched |
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

`target_Z` is a final-state number. Z1+ counts the kinks of the shortest paths between the junctions at their current positions, so it changes by 10-20 % through equilibration and compression even with no bond crossings and no free-ended chains (e.g., on the DP-100 reference it was 1.16 after push-off, 1.00 after constant-volume equilibration and 1.11 after compression, with every bond under 1.2 sigma). The controller starts from the build state but closes on the final state.

Chain ids in `pairs` index `topon.conformation.strand_plans`, which follows the graph's edge order (the order in which a `Placement` keeps its strands). `topon.conformation.entanglement.requests_from_config` also accepts an edge key `[u, v, k]` in place of an index.

`shells` sets the shell mix for the pairs the conformation stage routes. The assignment stage chooses *which* pairs (`assignment.entanglements.shell_weights` and `select_by_shells`), and a driver calls it and passes the result in.

#### The controller

```python
from topon.conformation.entanglement import controller

manifest = controller(graph, config.conformation, runner, dp=20)
```

`runner` is a callable supplied by the caller. It receives a `RoundPlan` (the round number, the placement, which setting was changed and to what, the DP and the seed) and must return a `RoundResult` or a dict with at least `z_final`. The runner builds the system, runs the relaxation protocol and measures Z1+. Keeping these out of the module keeps the conformation stage free of LAMMPS and lets the loop be tested against a known response.

A driver for a real protocol writes the build and the five-stage push-off with `topon.writers`, runs and gates it with `topon.simulation.protocols.StagedRun`, and measures Z1+ with its own primitive-path code. Z1+ cannot be redistributed, so such a driver does not ship with topon.

The controller starts from the shipped calibration (`topon.conformation.entanglement.CALIBRATION`), which holds the measured (actuator, Z) pairs with the state and protocol of each. Only final-state points from the crossing-free protocol steer the controller. Points from the minimizer protocol are kept and labeled but do not steer, because that protocol stretches bonds and adds crossings of its own. Over the measured range Z follows a power law in the actuator (exponent 0.87 for the DP-20 meander, 0.30 for the DP-100 walk), so two rounds fix the curve for the graph at hand.

A `target_Z` below the lowest value measured for a route gives a warning that names the floor and, for the walk, the meander route that goes lower.

```
target_Z 0.18 is below the lowest final-state Z the walk route has reached at
DP 20 (0.262, at build_density 0.035). ... Switch conformation.placement to
'meander' to go lower; it is the shape, not the density, that sets the floor.
```

#### Designed pairs, and when they are refused

Designed pairs are only partly working. The contour budget check is complete, but the routing does not yet pass the gate on a strand already drawn as a meander. Adding a braid to such a strand leaves beads two to five places apart at 0.5 to 1.0 sigma where the braid meets the meander, which is the threaded-bond problem the gate catches. With `strict` (the default) the strand keeps the path the placement drew, and the request is refused with the reading that failed. For delivered windings, use the pipeline route (`assignment.entanglements` with `method: "waypoint"`), which builds the braided strand *from its chord*.

Each winding costs contour. The two partners leave their chords, wind around each other and come back, and a strand has only `n_bonds * bond` of contour. `route_designed_pairs` measures the routed path before drawing anything and refuses a request that does not fit, naming the smallest DP that would carry it.

```
pair (0, 514) x2: refused, the contour budget does not allow it: routing 2
winding(s) needs 33.2 sigma of path and DP 20 carries 20.4. The routed path is
33.2 sigma and these strands carry 20.4; DP 34 is the smallest that would carry
it at this geometry
```

Two other limits can apply, and a higher DP does not fix them. The braid needs `windings * pitch + 2 * ramp` of the shared axis, which two short chords may not have. That refusal suggests a lower build density, so that the chords get longer, or a partner in a closer shell. The last limit is clearance. A braid squeezed into a chord that is too short brings its two arms within a fraction of sigma, and the push-off would push them through each other, so this is also refused.

The minimum DP in a refusal holds only *at that geometry*. Rebuilding at a higher DP and the same build density puts more beads in the box, so the box and every chord grow as `DP^(1/3)` (e.g., a pair that needed DP 24 in the DP-20 box needed DP 25 once rebuilt at DP 24). Rebuilding at the same *box* size instead (i.e., raising the density with the DP) reaches the named DP in one step.

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
coordinates it was given.

### `simulation` (raw section, not schema-validated)

This section controls the LAMMPS scripts written by stage 6.

| Key | Default | Description |
|---|---|---|
| `protocol` | `"pushoff"` | Coarse-grained relaxation protocol (see below) |
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
  counts are always reported.
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
