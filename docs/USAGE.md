# topon usage guide

This is the reference for **running topon end-to-end**: CLI flags, sub-system APIs, recipes for the most common system types, and the JSON config schema (appendix).

For the package layout and design rationale, see [ARCHITECTURE.md](ARCHITECTURE.md) first.

---

## 1. Install

```bash
pip install -e .
```

Runtime dependencies (installed from `pyproject.toml`): `numpy`, `networkx`, `pandas`, `rdkit`, `pydantic`, `click`, `plotly`, `scipy`. LAMMPS (`lmp`) must be on `PATH` only if you want to *run* the generated systems; generation itself does not need it.

---

## 2. Quick start

### 2.1 Interactive shell (recommended)

Run `topon` (or `python -m topon`) with no arguments on a real terminal. You land in the `topon>` REPL, where every subcommand is one token away:

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

Shell built-ins: `help`, `help <cmd>`, `exit | quit | q | Ctrl-D`. Arrow keys for history if `readline` is installed. `help <cmd>` shows the same click `--help` page you'd see in one-shot mode.

The shell auto-launches when stdin is a TTY. To force it from a non-TTY context use `topon shell`. To skip it and just print the banner, use `topon --no-shell`.

### 2.2 One-shot mode

Same commands, but each one re-invokes the CLI:

```bash
# 1. Generate a starter config
topon init --output my_run.json

# 2. Validate it
topon validate my_run.json

# 3. Run the full six-stage pipeline
topon generate my_run.json --output ./runs
```

The output lands in `./runs/<study.name>/{topology, 02_Chemistry, 03_Conformation, 04_Simulation}/`. `02_Chemistry/` holds `system.data`, and `04_Simulation/` holds the LAMMPS `.in` scripts.

To run the simulation (optional):

```bash
cd ./runs/<study.name>/04_Simulation/
lmp -in minimize_1_serial.in
```

For ready-to-use config files, see `demos/templates/` (`minimal.json`, `full.json`) and the demo configs under `demos/polymer/` and `demos/poss/`. [`demos/README.md`](../demos/README.md) lists them all.

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
| `--export-graphml` | Also write the graph (chains and entanglement edges) as `<study.name>.graphml` in the study folder; same as `output.export_graphml` |
| `--export-npz` | Also write the graph as `<study.name>.npz` for graph-learning pipelines; same as `output.export_npz` |

Runs the six-stage pipeline (Topology → Analysis → Assignment → Chemistry → Conformation → Output). LAMMPS data files and input scripts are written to `output_dir/study_name/`.

```bash
topon generate demos/templates/full.json
topon generate demos/templates/full.json --output ./my_run
topon generate demos/templates/full.json --dry-run
```

### 3.2 `topon validate`

```bash
topon validate CONFIG_PATH
```

Prints `Configuration is valid!` or lists every validation error. Cheap pre-flight check before submitting a long run to HPC.

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
| `--interactive`, `-i` | off | Prompt for the 5-6 knobs that actually vary (study name, output dir, model type, lattice type+size, max functionality, DP, density) and write the result. |

The non-interactive default copies `demos/polymer/atomistic/basic/config.json`. Every preset-produced file passes `topon validate` immediately.

### 3.3a `topon doctor` (semantic lint)

```bash
topon doctor my_run.json           # informational + warns
topon doctor my_run.json --strict  # warns also exit 1
```

Where `validate` is a Pydantic schema check, `doctor` runs a small registry of semantic rules for known issues and common mistakes. Current rules:

| Rule | Level | Catches |
|---|---|---|
| `lattice_size_format` | error | `"lattice_size": 5` instead of `"5x5x5"` |
| `neighbour_cutoff_vs_box` | warn | `neighbour_cutoff` above a third of a periodic axis: three candidate edges can close a cycle around the box |
| `deprecated_mix_cutoff` | warn | `mix_cutoff` still present; it loads as `neighbour_cutoff` |
| `unknown_config_keys` | warn | A nested section carries a key the schema does not define, so it is ignored silently |
| `unknown_node_type` | warn | `assignment.node_types.degree.mapping` references a type that's not in `chemistry.node_type_map` (would silently fall through to Si) |
| `poss_at_internal_junction` | warn | POSS mapped to degree >= 2 junctions, which gives a bond longer than half the periodic box at LAMMPS stage 1 |
| `atomistic_graft_non_pdms` | warn | Graft density set on a non-PDMS atomistic monomer (the build skips those grafts with only a `RuntimeWarning`) |
| `dp_below_kuhn` | warn | DP < 5 (conformation/entanglement edge cases) |
| `defects_endcap_safe` | ok | Reminder that loop defects skip degree-1 chain caps |
| `entanglement_target_below_floor` | warn | `conformation.entanglement.target_Z` is below the lowest value that route has reached at this DP; names the floor and the route that goes lower. A round of the controller is a full relaxation protocol, so this is worth catching before it starts |
| `conformation_build_knob` | error | A `target_Z` with neither `coil_ratio` nor `build_density`, and nothing measured for that DP and placement to seed the controller from |
| `schema_gap_extras` | ok | Config has `simulation`/`execution` (not Pydantic-validated; the CLI handles them via `load_config_full`) |

Adding a new rule: write `check_<name>(cfg, raw) -> list[Issue]` in `topon/diagnostics/rules.py` and append to `RULE_REGISTRY`.

### 3.3b `topon inspect <run_dir>` (post-run summary)

```bash
topon inspect runs/my_study
```

Replaces hand-grepping `system.data` headers after a long pipeline. `RUN_DIR` is the study folder or its parent. Parses each stage directory (Pipeline layout: `02_Chemistry/`, `03_Conformation/`, `04_Simulation/`) or a flat folder with every file at the top level, and prints:

- atom count, atom-type count, box dimensions
- what the topology stage was asked for against what it produced, from the run manifest
- per-stage status (which files landed, what they say)
- the next LAMMPS commands to run

**The run manifest.** `Pipeline` writes `manifest.json` into the run
directory as it goes, with what each stage was asked for, what it produced and
how long it took. Stage 1 records the lattice, the search, the degree
counts requested against the ones achieved, the giant-component fraction
and the seed, which is the precision / convergence / success-rate evidence
behind a topology claim. `topon inspect` renders it:

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

A strict run records the same two rows, listing only the degrees its
target named (the others read `-`) plus any `e:N` budget. The manifest is
advisory. Nothing reads it back to make a decision, so a missing or
half-written one costs an inspection detail and nothing else. Read it from
Python with `topon.core.manifest.read_manifest(run_dir)`.

### 3.3c `topon recipes` (common use cases)

```bash
topon recipes
```

Prints a "I want X -> run Y" cheatsheet covering all sub-systems (polymer networks via Pipeline, simbox, single-chain, the batch topology demo, `inspect` and `analyze`). Edit `topon/cli.py:recipes()` to add rows.

### 3.4 `topon simbox` (pack a crosslink box)

Independent sub-system. Builds Epoxy-PDMS / Amino-PDMS / AM0270-POSS molecules, packs them at target density, and writes DREIDING LAMMPS data + input scripts.

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

Output (single flat directory):

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

Then run LAMMPS:

```bash
cd simbox_output && lmp -in 1_minimize.in
```

See §4.1 for the simbox Python API.

### 3.5 `topon chain` (single chain in solvent)

Build a single polymer chain in solvent and emit DREIDING LAMMPS files.

```bash
topon chain --chain-smiles SMILES --dp N [options]
```

| Option | Default | Description |
|---|---|---|
| `--output`, `-o` | `chain_output` | Output directory |
| `--chain-smiles` | *(required)* | SMILES for the polymer repeat unit |
| `--dp` | *(required)* | Degree of polymerization |
| `--solvent-smiles` | `None` (toluene fallback) | Single-solvent SMILES; ignored if `--solvent-mixture` set. If both are unset, `run_workflow` falls back to toluene. |
| `--n-solvent` | auto | Solvent molecules; auto-calculated from density if omitted |
| `--solvent-mixture` | `None` | Multi-solvent JSON: `'[{"smiles":"...","weight_fraction":0.5}, ...]'` |
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

Output:

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

Analyze a topology graph file and print statistics.

```bash
topon analyze GRAPH_PATH [--format text|json] [--nodes NODES_PATH]
```

| Argument / Option | Description |
|---|---|
| `GRAPH_PATH` | Path to `.gpickle`, `.nodes`, or `.edges` file |
| `--format`, `-f` | Output format: `text` (default) or `json` |
| `--nodes` | Companion `.nodes` file for a `.edges` `GRAPH_PATH`; without it the `.nodes` file with the same stem is used |

```bash
topon analyze network.gpickle
topon analyze network.nodes
topon analyze network.edges --nodes network.nodes
topon analyze network.gpickle --format json
```

The CLI dispatches to `topon.analysis.report.analyze_graph()` and prints degree distribution, connectivity, and topology statistics. Available as both CLI and Python import.

### Global options

| Option | Description |
|---|---|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |
| `--no-shell` | With no sub-command, print the banner and exit instead of starting the interactive shell |

---

## 4. Sub-systems

### 4.1 simbox (molecule packing)

`topon.simbox` packs individual molecules into a periodic simulation box and emits LAMMPS input scripts for crosslinking studies. Independent of the polymer-network pipeline.

**Workflow:**

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

Quickest path: `topon simbox` (§3.4) or `topon.simbox.workflow.run_workflow()`.

**`Molecule`** (`topon/simbox/molecule.py`) is an RDKit Mol with explicit H, an ETKDGv3 + MMFF-optimised 3D conformer, and reactive-site annotations auto-detected via SMARTS:

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

**`MoleculeLibrary`** (`topon/simbox/library.py`) holds pre-built siloxane molecules:

```python
from topon.simbox.library import MoleculeLibrary
lib = MoleculeLibrary()
epoxy  = lib.epoxy_pdms(n_dms=2)    # Glycidoxypropyl-PDMS, ~500 g/mol
amino  = lib.amino_pdms(n_dms=8)    # Aminopropyl-PDMS, ~850 g/mol
poss   = lib.am0270_poss()           # AminopropylIsooctyl POSS, ~1267 g/mol
custom = lib.custom("C1OC1", name="MyEpoxide")
```

Structures:
- **Epoxy-PDMS**: `Epoxide-CH₂-O-CH₂CH₂CH₂-Si(Me)-[O-Si(Me)₂]ₙ-O-Si(Me)-CH₂CH₂CH₂-O-CH₂-Epoxide`
- **Amino-PDMS**: `H₂N-CH₂CH₂CH₂-Si(Me)-[O-Si(Me)₂]ₙ-O-Si(Me)-CH₂CH₂CH₂-NH₂`
- **AM0270 POSS**: Si₈O₁₂ cube cage, corner 0 with `-CH₂CH₂CH₂-NH₂`, corners 1-7 with 2,4,4-trimethylpentyl (isooctyl, inert)

**`BoxPacker`** uses grid-based spatial hashing for O(N) overlap detection:

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

Algorithm: compute initial box from total mass and target density → shuffle insertion order → for each molecule random rotation (Shoemake quaternion) + random translation, with min-image overlap check → if `max_attempts` exceeded, grow box by `growth_factor` and retry (up to 20 rounds).

**`AssembledSystem`** is the merged Mol with global bookkeeping:

```python
from topon.simbox.system import assemble
system = assemble(packed)
# system.mol               merged RDKit Mol
# system.box_lengths       ndarray([Lx, Ly, Lz]) in Å
# system.molecule_ids      per-atom LAMMPS molecule ID (1-based)
# system.species_names     per-molecule species name
# system.reactive_sites    list of ReactiveSiteEntry (global atom index + group name)
```

**Writers:**

```python
from topon.simbox.writer import write_lammps
from topon.simbox.inputs import write_inputs

write_lammps(system, output_dir="output/simbox")
write_inputs(system, output_dir="output/simbox", temperature=300.0, pressure=1.0)
```

Stage 1 (soft push-off + minimisation):
- Phase A: `pair_style soft` with ramped prefactor (0→60) + brief NVT to resolve overlaps
- Phase B: switch to `lj/cut` DREIDING potentials + conjugate-gradient minimisation

Stage 4b (crosslink template, user must configure):
- **Option A** (`fix bond/react`): template-based reactions with molecule pre/post files
- **Option B** (`fix bond/create`): simple distance-based bond formation

**One-call workflow** (canonical entry point):

```python
from topon.simbox.workflow import run_workflow

files = run_workflow(
    output_dir="output/simbox_run",
    n_epoxy=600, n_amino=300, n_poss=0,
    density=0.85, seed=42,
)
```

`run_workflow` also activates `UniversalTypeMapper`, a context manager that patches `topon.forcefield.dreiding` at write time to enforce stable DREIDING type IDs across all compositions, keeping pre-defined `fix bond/react` templates compatible.

### 4.2 singlechain (solubility utility)

`topon.singlechain` builds a single polymer chain in a solvent box for solubility studies. Use the CLI (§3.5) or the Python entry point:

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

Each recipe shows a config + the command. Most knobs live in the JSON config, and Appendix A has the full schema.

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
The topology below is the one `demos/topology/end_linking/python/run.py` writes (run it first, from the repository
root).

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

Combine the entanglement recipe (5.1) with a `grafts` block in `assignment`:

```json
"grafts": {
  "enabled": true,
  "per_edge_type": {
    "A": { "graft_density": 0.05, "side_chain_monomer": "PDMS", "side_chain_dp": 5 }
  }
}
```

A complete combined-features config ships as `demos/polymer/atomistic/combined/config.json` (and `demos/polymer/coarse_grained/combined/config.json` for CG).

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

The CLI is a thin wrapper. Equivalent Python:

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

For lower-level entry points (each stage individually), see ARCHITECTURE.md §2 (the per-stage `Module` line tells you which import path drives that stage).

---

## 7. Demo scripts

`demos/` contains runnable Python scripts that drive topon without the CLI. They are *not* part of the package API. Copy one, change the knobs at the top and run it with `python <path>`. [`demos/README.md`](../demos/README.md) describes the whole folder.

| Script | Purpose |
|---|---|
| `demos/run_via_api.py` | Runs a demo config through `topon.pipeline.Pipeline` directly, as a starting point for scripting the pipeline |
| `demos/topology/end_linking/python/run.py` | Generates a 6x6x6 SC topology with the pure-Python generator |
| `demos/topology/end_linking/c/run.py` | The same topology through the compiled C generator (set `TOPON_GENERATOR_EXE` to the binary) |
| `demos/workflows/batch_polymer_topology/run.py` | Generates 25 seeded lattice graphs, exports each as `.nodes`/`.edges`, GraphML and NPZ, and writes one CSV of per-graph properties |

---

## Appendix A. JSON config schema

`topon` is configured by a single JSON file. Use `topon init` to generate a starter and edit it as needed.

Top-level sections:

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

`conformation` is part of the validated schema. `load_config_full` also hands
a copy of it back in the raw-extras dict, so callers that have always read it
from there (`Pipeline`, `topon.workflows.cg_network`,
`topon.workflows.atomistic_network`) keep working unchanged; `simulation`,
`execution` and `experimental` are still raw-only.

### `study`

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | `"my_network"` | Study name; used as sub-directory under `output_dir` |
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
| `exe_path` | string \| null | `null` | Path to `generator.exe`; `null` → use Python generator |
| `lattice_size` | string | `"6x6x6"` | Lattice dimensions, e.g. `"8x8x8"` |
| `lattice_type` | `"SC"` \| `"BCC"` \| `"FCC"` \| `"Diamond"` \| `"MIX"` | `"SC"` | Lattice type; `MIX` overlays SC/BCC/FCC (see below) |
| `mix_fractions` | object | `{"SC":1,"BCC":0,"FCC":0}` | Sublattice fractions for `MIX`; must sum to 1 |
| `neighbour_cutoff` | float | `1.0` | Candidate-edge range for every lattice type, in cell units; `1.0` is the canonical nearest-neighbour lattice, larger values admit further shells (see below) |
| `neighbour_shells` | int \| null | `null` | SC only: number of neighbour shells, an alternative to `neighbour_cutoff` (`2` -> 1.42, `3` -> 1.74, `4` -> 2.01, `5` -> 2.24, `6` -> 2.45, `8` -> 3.01) |
| `mix_cutoff` | float | - | **Deprecated** alias of `neighbour_cutoff`; loads with a warning (at its default 1.0 it yields to `neighbour_shells`) |
| `periodicity` | string | `"111"` | Periodicity per axis (`1`=periodic, `0`=open); see below |
| `max_functionality` | int | `6` | Maximum crosslink degree per node |
| `max_trials` | int | `1000000` | Trials before giving up (strict search only) |
| `max_saves` | int | `1` | Number of networks to save |
| `degree_distribution` | string | `"0:0,1:0"` | Target degree distribution |
| `search` | `"strict"` \| `"exact"` \| null | `null` | Which sculptor runs; `null` picks `exact` when the degree distribution pins every degree, `strict` otherwise (see below) |
| `min_giant_fraction` | float | `0.99` | Exact search only: smallest fraction of active sites the largest component may hold |

**`topology.generator` refuses unknown keys.** So do the per-class blocks
under `assignment.defects` and the whole `conformation` section; everywhere
else an unrecognised key is dropped in silence. That mattered because `"seed": 7` under `topology.generator`
validated cleanly and did nothing, so a config that looked pinned was
not. There is no `seed` key here; to make a generated graph reproducible,
seed the global streams before generating:

```python
import random, numpy as np
random.seed(42); np.random.seed(42)          # then build
```

`topon.workflows.cg_network.run(seed=...)` does this for you;
`Pipeline(config)` does not, so a direct API build needs the two lines
above. For the other sections, `topon doctor` reports ignored keys under
the `unknown_config_keys` rule rather than refusing to load, because a
config may carry other tools' parameters under `topology`.

Degree distribution format: `"d:N"` requires N nodes of degree d, `"e:N"` requires N edges total, and omitted degrees are unconstrained. Example: `"0:15,1:30,e:371"`.

##### Which sculptor (`search`)

The candidate-edge lattice is a superset; a sculptor picks the network out
of it. There are two, and they fail in opposite places.

| | `strict` | `exact` |
|---|---|---|
| how | removes candidate edges one at a time until the graph matches | assigns the requested degrees to sites, then completes the degree sequence with augmenting paths |
| needs | any subset of targets (`d:N`, `e:N`, or nothing) | a count for every degree from 0 to `max_functionality` |
| samples | `max_trials` random trials | up to 6 seeds per network, then 6 fallback attempts when all six ended short; in C (`exe_path` set) an explicit `max_trials` bounds the attempts instead |
| move history | yes, `G.graph["move_history"]` | no |
| good at | loose targets, an edge budget, a ceiling well above the mean degree | near-complete tetrafunctional targets, dangling-end sites, high vacancy fractions |
| bad at | exactly those (see below) | a scaffold with no spare candidate edge (Diamond at `max_functionality: 4`) |

Left unset, `search` resolves to `exact` when the degree distribution names
every degree from 0 to `max_functionality` and to `strict` otherwise, so
every config written before this key existed behaves as it did. An
explicit value always wins. A fully specified degree distribution already
fixes the edge count, so an `e:N` term next to one is accepted when it
agrees and refused when it does not.

**Why the second search exists.** Measured against the two end-linked
reference networks, the strict sculptor cannot reach a P(f) that is 73 % tetrafunctional with 8 %
dangling ends on any scaffold at all, in Python or C, inside a 3 to 10
minute budget. Diamond is the one scaffold that reaches a junction-only
version of the spec (0.6 s for DP 100, 20 s for DP 20), and once
dangling-end sites are in the target it never finishes either. Stage 3 prunes every site to the ceiling greedily and
overshoots the edge budget, stage 4 can then only remove more, and the
strict degree-1 stage kills 99.7 % of trials when a tenth of the sites
must be chain ends. The exact search reaches the same targets in 0.1 to
2 seconds:

| target | scaffold | result |
|---|---|---|
| `0:43,1:217,2:356,3:153,4:1975` (DP 20) | SC 14³ | exact, giant 1.000, 0.5 s |
| the same | 90/5/5 MIX 14³ at cutoff 2.01 | exact, giant 1.000, 0.1 s |
| `0:157,1:73,2:33,3:59,4:407` (DP 100, 21 % vacancies) | SC 9³ | exact, giant 0.995 to 1.000, 0.2 s |
| `0:10,1:0,2:26,3:75,4:43,5:53,6:9` (a network of the paper dataset in `demos/npjcompmat/data/mechanics/`) | SC 6³ | exact, 0.03 s (strict also reaches it) |

```json
"generator": { "lattice_type": "SC", "lattice_size": "14x14x14",
               "neighbour_cutoff": 1.74, "max_functionality": 4,
               "degree_distribution": "0:43,1:217,2:356,3:153,4:1975" }
```

Six things to know about the exact search:

- **The degree sum has to be even.** Every edge contributes 2, so an odd
  sum belongs to no graph at all and is refused outright rather than
  searched for. Move one site between two odd degrees to fix it.
- **Degree 0 is the leftover.** The active sites are placed first and
  whatever the scaffold has over becomes vacancies, so `0:43` on a
  2744-site cell is a statement about the cell, not a constraint. On a
  larger cell the active counts still land exactly and the run prints how
  many sites were left empty. Vacancies are dropped before chemistry, as
  they always were.
- **A dangling end attaches to a junction.** No edge ever joins two
  degree-1 sites, which would be a free chain rather than part of the
  network.
- **The animation needs `strict`.** The exact search has no edge-removal
  history, so a graph it produces carries no `move_history` and a
  sculpting animation cannot replay it. Pin `search: "strict"` when you
  want one.
- **A scaffold with no slack fails, and says so.** Diamond has four
  candidate partners per site, so at `max_functionality: 4` a junction at
  the ceiling must bond to every neighbour it has; with 73 % of the sites
  at the ceiling, each vacancy and each dangling end takes capacity
  nothing can give back. When most of a request is pinned to the
  scaffold's own coordination like that, a fresh seed draws the same
  forced assignment, so the run stops after one attempt with a message
  naming the coordination, the residual and the remedy. A target that
  puts only a handful of sites there keeps its retries, and a failure to
  connect (rather than to fill) always retries. The remedy on Diamond is
  `neighbour_cutoff: 0.71`, which admits its second shell (z = 16) and
  lands the same target in 0.01 s; note that the default 1.0 is the
  canonical-lattice sentinel rather than a range, so the wider setting is
  here the smaller number.
- **A target the random deal cannot place gets a fallback.** SC, BCC and
  Diamond at their first shell are bipartite, so every edge adds one to
  each sublattice and the targets on the two must sum to the same number.
  A random deal rarely balances, and a target with many dangling ends and
  many sites at the lattice's own coordination (44 and 54 of 216 on SC)
  then ends short on every attempt. After 6 attempts in a row end with
  unfilled degree units, the search deals the targets balanced across the
  sublattices and repairs by moving demand out of the region a failed
  augmenting search reached, within one sublattice, keeping a move only
  if it loses no edge. Connectivity failures do not count toward the
  switch, and the attempts before it draw what they always drew, so any
  target the random deal reaches comes out as before for the same seed.

Both searches exist in both generators. With `exe_path` set, an exact
request runs the C port in `topon/topology/csrc/` (`--search=exact`,
passed by `run_generator`), with the same steps, constants and refusals
and `min_giant_fraction` passed through. The C binary treats one trial as
one attempt. Left at its default, `max_trials` becomes the Python budget
of 6 random and 6 fallback attempts per network, so an infeasible request
gives up after seconds on either route; set it explicitly to let the C
search retry longer. The pipeline hands the binary a seed drawn from the global NumPy
stream, so `np.random.seed(n)` pins the C route as it pins the Python
one, and the manifest records it with the requested and achieved counts.
The one exception is an exact request that forces double edges or
reserves capacity for triangles and four-cycles (the `assignment.defects`
keys below). Only the Python search takes those, so the pipeline keeps it
on Python and says so.

**Forced parallel strands.** `PythonTopologyGenerator.generate` takes a
`double_pairs={(a, b): count}` argument (exact search only) that places
parallel edges between sites of target degree `a` and `b` before the fill
and holds them fixed. It reproduces the reference's secondary loops with
their endpoint statistics, which random injection does not. N20's
`{(4,4):115,(3,4):6,(2,4):7,(2,3):1}` gives 129 parallel pairs at exactly
the reference P(f), where topon's own injector (`inject_secondary_loops`
in `assignment/defects.py`, used on an already-sculpted graph) picking
eligible pairs at random leaves 79 too few f = 2 and 134 too many f = 3.
The result is a `MultiGraph` and the second
edge of each pair carries `is_secondary_loop=True`.

##### Diamond (`lattice_type: "Diamond"`)

Two interpenetrating FCC sublattices offset by ¼ along the body diagonal:
8 sites per cubic cell, **every site exactly 4-coordinated by
construction**. A `max_functionality: 4` network therefore needs no
pruning at all, which makes it the cleanest backbone for a tetrafunctional
network and much faster to generate than sculpting SC or FCC down to 4.

```json
"generator": { "lattice_type": "Diamond", "lattice_size": "6x6x6",
               "max_functionality": 4, "degree_distribution": "" }
```

An `NxNxN` Diamond has `8N³` sites and `16N³` bonds at a nearest-neighbour
distance of `√3/4 ≈ 0.433` cells. Both generators build it identically.

##### Neighbour shells (`neighbour_cutoff`)

Every lattice type takes `neighbour_cutoff`, the candidate-edge range in
cell units (the simple-cubic site spacing). At the default `1.0` each pure
lattice is its canonical nearest-neighbour graph: 6 neighbours on SC, 8 on
BCC, 12 on FCC, 4 on Diamond. Any other value connects every pair of sites
within that distance under the minimum image, the way `MIX` has always
built its edges, so the candidate-edge set becomes a lattice plus a range.
The sculptor then picks the final network from that larger set, which is
what lets crosslinkers several site spacings apart end up bonded, as they
are in a reaction-generated (end-linked) network.

```json
"generator": { "lattice_type": "SC", "lattice_size": "14x14x14",
               "neighbour_cutoff": 1.74, "max_functionality": 4 }
```

On SC the shells sit at `sqrt(n)` cell units for n = 1, 2, 3, 4, 5, 6, 8,
9, ..., so `neighbour_shells` can name the range instead of the cutoff:

| `neighbour_shells` | `neighbour_cutoff` | neighbours per SC site |
|---|---|---|
| 1 | 1.0 (default) | 6 |
| 2 | 1.42 | 18 |
| 3 | 1.74 | 26 |
| 4 | 2.01 | 32 |
| 5 | 2.24 | 56 |
| 6 | 2.45 | 80 |
| 8 | 3.01 | 122 |

`neighbour_shells` is SC-only; on BCC, FCC, Diamond and MIX it is accepted
only next to an explicit `neighbour_cutoff`, which then governs. Given
both on SC, the cutoff must admit exactly that many shells. The shells a
lattice actually produced are recorded on the graph as
`G.graph["shell_distances"]` (sorted distinct edge lengths) next to
`G.graph["neighbour_cutoff"]`.

**Which cutoff.** Measured against two end-linked reference networks,
the cutoff that reproduces the reference connectivity is about the 95th percentile of the
strand's junction-to-junction separation divided by the site spacing.

| reference | site spacing | 95th-percentile reach | SC shells that match | z | composite (SC) | best cell in the sweep |
|---|---|---|---|---|---|---|
| DP 20 strands, bead density 0.31 | 4.95 sigma (14^3 cell) | 2.1 spacings | 3 to 4 (cutoff 1.74 to 2.01) | 26 to 32 | 0.13 | 0.09, a 90/5/5 mixture at 4 shells (z 39) |
| DP 100 strands, bead density 0.31 | 7.7 sigma (9^3 cell) | 3.3 spacings | 8 (cutoff 3.01); 6 (2.45) is close | 122; 80 | 0.10; 0.15 | 0.08, BCC sites at 5 shells (z 64) |

Every nearest-neighbour scaffold, SC, BCC, FCC and every mixture of them
at cutoff 1.0, is far off the same references (composite 0.4 to 1.05),
because bipartite lattices have no odd cycles and mixtures pile up
triangles and four-cycles. Adding shells spreads each site's candidate partners over a
larger neighbourhood, which lowers clustering and lengthens cycles.

Three things to watch:

- **The box.** Keep the cutoff under a third of every periodic axis
  (`lattice_size` of at least `3 * cutoff` cells). Past that three
  candidate edges can close a cycle around the box, so the candidate set
  carries triangles that come from the box, not the lattice; past half
  the box a pair can be within range through two images at once, which a
  simple graph cannot even represent. `topon doctor` warns
  (`neighbour_cutoff_vs_box`), and so does the generator.
- **Bond lengths spread.** DP is assigned independently of edge length,
  so at three shells the same DP is built at lengths from 1.0 to 1.73
  site spacings. Watch for FENE strain on the long edges.
- **Cost.** The neighbour search compares every pair up to 4 000 sites
  and switches to a cell list above that (22^3 SC at cutoff 3.01, 650 000
  candidate edges, builds in about a second). The sculptor's own cost
  grows with the candidate coordination.

A shell that sits exactly at 1.0 (corner-corner on BCC and FCC, the
fourth Diamond shell) is excluded by the default and included by any
value above it, e.g. `1.01`, which is how the sweep's "BCC sites, 2
shells" cells were built. `MIX` keeps its distance-search semantics at
every cutoff, so `MIX` at fractions `{"SC": 1}` and any cutoff has exactly
the candidate-edge set of `SC` at that cutoff (same ids, positions and
edges; at the default the two insert those edges in a different order, so
a seeded sculpt is a different draw). A cutoff
below a pure lattice's nearest-neighbour
distance (0.866 on BCC, 0.707 on FCC, 0.433 on Diamond) is refused, since
no site would have a neighbour.

The C generator takes the same range as an optional ninth argument, e.g.
`generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 SC 1.74`; `MIX` keeps it
inside its own argument. `topology.generator.exe_path` runs pass it
through automatically.

##### Boundaries (`periodicity`)

One digit per axis, `1` periodic and `0` open. An open axis omits its
wrap-around bonds, so the lattice grows a **free surface** there and the
sites on it lose coordination. The site set is unchanged either way.

```json
"generator": { "lattice_size": "6x6x6", "periodicity": "110" }
```

That builds a slab, periodic in x and y and open in z. On a 4x4x4 SC lattice
the bond count goes 192 → 176 → 160 → 144 as you open one, two and three
axes, and the surface sites drop from degree 6 to 5.

**Open boundaries interact with `degree_distribution`.** Corner and edge
sites on a free surface have very low coordination, and the centred
lattices lose the most:

| lattice (4x4x4) | min degree, `"111"` | `"110"` | `"000"` |
|---|---|---|---|
| SC | 6 | 5 | 3 |
| BCC | 8 | 4 | 1 (2 such sites) |
| FCC | 12 | 8 | 3 |
| Diamond | 4 | 2 | 1 (22 such sites) |

So the usual `"0:0,1:0"` (no isolated nodes, no dangling ends) is
**unsatisfiable** on a fully open BCC or Diamond. The only way to clear a
degree-1 site is to cut its last bond, which makes it degree 0, and that
is forbidden too. Both generators decline rather than claim success. On
SC the minimum stays at 3, so the same request is fine. Either drop the
`1:0` term on open lattices, or leave `degree_distribution` empty and let
`max_functionality` do the work.

`max_functionality` still applies on top, so a partially open lattice
reaches the ceiling with less pruning than a closed one.

**What an open axis does to the data file.** Coordinates are wrapped into
the box only on periodic axes. An open axis keeps its atoms where they
were placed and the box grows to contain them, so a junction on the free
surface stays next to the chains bonded to it instead of being split
across the cell. The `.nodes` file records the boundaries in a
`# PERIODICITY 100` header (written only when an axis is open), and the
conformation stage reads it.

An open axis also gets **12 Å of vacuum** between the outermost atom and
the box face, matching the pair cutoff the generated scripts use. That is
not cosmetic. LAMMPS *deletes* atoms that leave a non-periodic (`f`) face,
and the geometry handed to stage 1 is strained enough that surface atoms
move several Å in the first few dozen steps. With only 1 Å of clearance a
bonded atom was lost at step 49. Override it with the `open_axis_pad`
argument of `ConformationManager.apply_displacements` (it is not a config
key) if a run needs more, or less when the extra volume matters.

The generated LAMMPS scripts still say `boundary p p p`, because they are
not periodicity-aware. Set `boundary p f f` yourself to match; the data
file is already correct for it. Verified on a Diamond `100` network, where `p f f` and `p p p` both
complete stage-1 minimization, with no bond crossing an open face.

##### Mixed lattices (`lattice_type: "MIX"`)

All three cubic lattices share the cell corner and each adds sites on top
of it: BCC one body centre, FCC three face centres. `MIX` puts the corner
in every cell, the body centre with probability `mix_fractions.BCC`, and
each face centre with probability `mix_fractions.FCC`. The `SC` entry is
the remainder and places no site of its own, which is what makes the
three a partition summing to 1. Expected site count is
`Nx*Ny*Nz * (1 + f_bcc + 3*f_fcc)`.

```json
"generator": {
  "lattice_size": "6x6x6",
  "lattice_type": "MIX",
  "mix_fractions": {"SC": 0.2, "BCC": 0.4, "FCC": 0.4},
  "max_functionality": 4
}
```

The point of mixing is more neighbour distances. A pure SC lattice offers
a single edge length; the mixture above offers four (0.5, 0.707, 0.866,
1.0 cell units), which smooths the distribution of strand end-to-end
distances. Three things to know before using it:

- **`MIX` at `{"SC": 1}` reproduces `SC` exactly**, down to node ids, at
  every cutoff. That is *not* true at the other two corners at the default
  cutoff. `MIX` connects by distance cutoff rather than by a fixed
  neighbour pattern, so at `{"BCC": 1}` the 1.0 cutoff also admits the
  corner-corner shell and every node carries 14 neighbours instead of
  BCC's 8 (18 instead of 12 at `{"FCC": 1}`). Use `lattice_type: "BCC"`
  or `"FCC"` when you want the canonical coordination; `"BCC"` with
  `neighbour_cutoff: 1.01` gives the same 14.
- **Bond lengths spread.** A body centre and a face centre can land 0.5
  cells apart, half the SC spacing. DP is assigned independently of edge
  length, so strands of the same DP get built at bond lengths differing by
  up to 2x. Watch for FENE strain on the long edges.
- **The split is a coarse dial.** The SC/BCC/FCC percentages are a
  weak, ill-conditioned knob, and many splits fit a given target
  comparably well. Site jitter and a Gaussian-weighted edge rule (neither
  is a topon option) move the strand statistics much more. Treat the fractions as
  SC-heavy for short strands shifting toward BCC/FCC as strand length
  grows, and verify by measurement rather than by tuning percentages.

Lowering `neighbour_cutoff` below 1.0 drops the corner-corner shell, which
disconnects the always-present corner sublattice from itself. 1.0 is the
default for that reason. Raising it admits further shells on the mixed
point set, exactly as on the pure lattices (see *Neighbour shells* above).

#### `topology.existing_files`

Provide either `gpickle_file` OR both `nodes_file` + `edges_file`.

| Key | Type | Default | Description |
|---|---|---|---|
| `nodes_file` | string \| null | `null` | Path to `.nodes` file |
| `edges_file` | string \| null | `null` | Path to `.edges` file |
| `gpickle_file` | string \| null | `null` | Path to NetworkX `.gpickle` file |

##### `.nodes` file format

Whitespace-separated `NodeID X Y Z Degree`, with `#` starting a comment.
An optional `# BOX Lx Ly Lz` header records the periodic cell in lattice
units:

```
# BOX 6 6 6
# NodeID X Y Z Degree
0 0.000000 0.000000 0.000000 3
1 1.000000 0.000000 0.000000 3
```

The header is optional and files without it load exactly as before, but
**write it for any lattice whose sites are not integer-spaced.** Without
it topon estimates the cell as `max - min + 1` over the coordinates,
which is exact for SC but overshoots BCC, FCC and Diamond because their
basis sites sit at fractional offsets and never reach the cell edge. A
4x4x4 BCC or FCC is estimated at 4.5, and since that value drives every
minimum-image calculation, about a third of BCC edges (a quarter of FCC)
get built at twice their true bond length. Anything topon generates
records the header for you; the caveat applies to hand-written or
externally-produced files.

### `assignment`

Controls how graph attributes (types, DP, defects, entanglements, grafts, copolymers) are written before chemistry is built.

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

`pdi` = polydispersity index (Schulz-Zimm distribution). `1.0` = monodisperse.

`endlinked_dangling` (default `false`) spends one bead of every dangling
strand on its free end site. In the end-linked convention a dangling chain
is DP beads, the last of which *is* the free end, and topon models that end
as a degree-1 site of its own. Turn it on when the bead count has to match
an end-linked dataset (209 dangling strands on the DP-20 reference: 102 500
beads instead of 102 709), and with `output.lammps_convention:
"endlinked"`.

#### `assignment.defects`

The defects stage runs after the topology is sculpted and before chemistry
is built. It injects four classes of network defect, plus sol, and records
requested against achieved for each of them.

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
count switches that class on; `enabled: true` with no count does nothing.

| Class | What it is | How it is placed |
|---|---|---|
| `primary_loops` | A strand that returns to the junction it left (a self-loop edge, `cls="loop"`). | `"by_effective_degree"` (default) fills in ascending effective degree: an effective-degree-0 junction takes two loops, every junction from effective degree 1 up to `f_target - 2` takes one, at random within each degree. That is what the reference does, and it reproduces its chemical P(f) exactly. `f_target` is the network's own functionality (the highest effective degree any junction carries), not `max_functionality`, which is only the valence ceiling. `"random"` takes any junction with spare chemical valence. `dp` overrides the loop's length; `null` inherits it from the bridges on that junction. |
| `secondary_loops` | Two strands between the same junction pair (a parallel edge). | With `topology.generator.search: "exact"` they are forced into the sculpt as double edges, which is the only way to place them with the degree counts staying exact; the stage then verifies and records them (`source: "sculpt"`). Otherwise it falls back to endpoint-targeted injection, which raises both endpoints by one and so shifts P(f) by `2 x count`. `endpoint_degrees: "auto"` lets the sculpt pair like with like from the highest degree down. |
| `triangles`, `four_cycles` | One added edge closing a three- or four-cycle. | Both raise two junctions by one degree. With `topology.generator.search: "exact"` the stage reserves that capacity in the sculpt target (`2 x count` junctions sculpted one degree below `max_functionality`) and raises exactly those back, so the delivered P(f) is the requested one and `degree_shift` is 0. On any other topology the edges are added to the finished graph and `degree_shift` reports the deviation. `collateral_cycles` counts the cycles of the *other* class the added edges closed as well. |
| `sol_chains` | Chains bonded to no junction. | Not part of the junction graph, but part of the bead budget. Coarse-grained only. |

`seed` makes the placement reproducible; leave it out to draw from the
global RNG state.

**Two functionalities.** A junction's *effective* functionality counts the
strands that reach somewhere else, which is what elasticity sees; its
*chemical* functionality adds two per primary loop, which is what the
crosslinker's valence sees. `topon inspect` prints both.

**Loops and sol are part of the bead budget.** The box is sized from the
bead count at the target density, so leaving them out shrinks it. On the
DP-20 end-linked reference the 345 primary loops and 11 sol chains hold
7 % of the beads, and a build without them has 7.3 % more bridges per unit
volume, which on its own raises the tensile peak by 7 %.

**Naming.** A primary loop is a self-loop and a secondary loop is a
parallel pair, which is the polymer-network usage. In earlier versions
topon's `primary_loops` key and `inject_primary_loops` function both meant
parallel edges. A config that still sets `primary_loops.target` (the old
field) gets the old behaviour with a `FutureWarning`; use
`secondary_loops` for parallel strands and `primary_loops.count` for
self-loops.


#### `assignment.entanglements`

```json
"entanglements": {
  "enabled": true, "target": 5, "target_type": "count"
}
```

Or distribution mode (average per chain):

```json
"entanglements": {
  "enabled": true,
  "avg_crosslinks_per_chain": 2.0
}
```

##### `method` (how an entangled pair is realised in 3D)

| `method` | What it draws |
|---|---|
| `"waypoint"` (default) | The pair together: both chains are splines spiralling about their contact in antiphase, so the pair carries exactly `entanglement_count` windings by construction. Verified with primitive-path analysis. |
| `"kink"` | The legacy Gaussian bump aimed at the partner's midpoint. Each chain is drawn alone; what the pair carries after relaxation is statistical. Kept for reproducing systems built with it before `waypoint` existed. |

`kink_params` applies only to `method: "kink"`:

| `kink_params` key | Default | Description |
|---|---|---|
| `overshoot` | `0.2` | How far the kink extends past the midpoint (0-1) |
| `z_amp` | `0.5` | Out-of-plane amplitude of the Gaussian kink |
| `sigma` | `0.15` | Width of the Gaussian kink |

##### Choosing pairs on a conformation instead of on crosslink distance

By default candidates are ranked by the distance between their crosslinks,
which is a property of the network rather than of the chains. Two chains can
be nearest neighbours by crosslink and never come near each other, and a kink
placed there aims one chain at a partner that is not present.

`select_entanglements` accepts a `chain_paths` argument (a mapping of
`frozenset((u, v))` to that chain's bead path) and ranks candidates by how
much of the two chains actually lies alongside. The assignment stage does not
draw the conformation itself; the caller supplies one, in the same units as
the node positions. The sequence is to draw a provisional conformation with
no entanglements, rank on it, select, then draw the final one with the kinks.

`topon/conformation/paths.py` provides `bridging_walk` for that first pass,
a random walk of fixed bond length that closes exactly on its far junction.
A straight chain will not do, since it lies on its chord and so cannot say
anything about which chains meet.

Measured on a 354-chain network, 281 candidates, 0.20 entanglements per chain:

| ranking | median proximity of chosen pairs | chosen pairs whose chains never touch |
|---|---|---|
| crosslink distance (the default) | 39 | 8 of 33 |
| on a conformation | **156** | **0 of 35** |

The pool median is 50, so the default ranking is slightly worse than choosing
at random.

##### Drawing a path around what is already there

`topon/conformation/paths.py` also carries the pieces used to place a chain
into an occupied box. None of it is wired into `Pipeline`; call it from
Python directly.

| name | does |
|---|---|
| `Clearance(points, box, radius)` | the beads already present, as a minimum-image nearest-neighbour query. `near`, `worst`, `ok` |
| `bridging_walk(..., avoid=)` | the same fixed-bond random walk, keeping each step clear of `avoid` |
| `loop_around(target, i, radius, n_pts, phase, avoid, span)` | waypoints encircling a strand. `span` is turns, so 0.5 is a hook and 2.0 is two turns |
| `taut_leg(start, end, n_bonds, bond, avoid, placed)` | a deterministic leg: exact bonds, lands on its end, avoids `avoid`, its own earlier beads, and `placed` |
| `route_through(start, end, waypoints, n_bonds, bond, avoid)` | visits every waypoint in order using `taut_leg` for each leg |

Two things about this matter more than they look.

**Draw around, not through.** A path placed with no regard for the beads
already there lands on top of them. Measured on a relaxed melt at density
0.85, routing one chain took the closest pair in the system from 0.502 σ to
0.195 and put 153 beads inside 0.5 σ where there had been none. At 0.195 σ the
WCA energy is of order 10⁵ kT, so the next minimisation does not relax that
contact, it shoves, hard enough to drag chains through each other, which
rewrites whatever topology was just built. `Clearance` is what avoids making
the overlap in the first place; through a real relaxed melt it takes the
tightest contact a routed path makes from 0.081 σ to 0.822.

**Spend the slack deliberately.** A chain carries far more contour than its
route needs, 77 σ for a route of about 21 in one measured case. A random walk
disposes of the rest by wandering, and the wandering crosses the target again
on its own account, so the entanglement count stops being a property of the
design. The same pair, same site, same winding, drawn on three seeds, measured
4, 7 and 0. `route_through` spends it deterministically instead, which is what
makes a requested count repeatable.

`walk_through` (random legs) and `route_through` (deterministic legs) take the
same arguments and give the same guarantees about bonds and junctions. Use the
first when a melt-like conformation is wanted and the second when a specific
topology is.

##### Choosing which neighbour shell to entangle

`shell_weights` biases the draw toward particular neighbour shells. Shells are
numbered from 1, closest first, and are read off the lattice rather than
assumed. The closest approach between two strands takes a handful of discrete
values (0.20, 0.35, 0.41, 0.50 lattice units on a mixed SC/BCC/FCC network),
and those bands are what "first neighbour" and "second neighbour" mean.

```json
"entanglements": {
  "enabled": true,
  "avg_crosslinks_per_chain": 2.0,
  "shell_weights": { "1": 0.7, "2": 0.3 }
}
```

Naming a shell restricts the draw to the shells named and weights them in
proportion. Omitting the key entirely, which is the default, draws from every
shell equally and is the behaviour of every earlier version. It multiplies
into `placement_bias_kind` rather than replacing it, so spatial and shell
biasing compose.

**Only the first shell reliably produces an entanglement.** Measured with a
primitive-path analysis, each pair checked on its own after the full
three-stage protocol:

| shell | gap | realised as asked |
|---|---|---|
| 1 | 12.3 σ | 5 of 7 |
| 2 | 21.4 σ | 2 of 16 |
| 3 | 24.7 σ | 0 of 16 |

The reason is the pair's gap divided by the chain's chord, 0.29 in the first
shell and 0.50 in the second. A chain has to spend contour reaching its partner,
and past about a third of a chord it runs out. That ratio is scale-free and
so a property of the lattice. It is 0.29 / 0.50 / 0.58 for every SC/BCC/FCC
mixture whatever the fractions, unchanged by box size, and worse for the pure
lattices (FCC 0.71, BCC 0.82). Weighting the outer shells up is allowed and
will not give you more entanglements there.

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
> **`gradient` is broken.** It ignores the requested composition and emits a
> hard 50:50 split (ask for A=0.1 and you still get A=0.50). For two monomers
> at equal fractions and even DP it is byte-identical to `block`.

### `chemistry`

| Key | Type | Default | Description |
|---|---|---|---|
| `model_type` | `"coarse_grained"` \| `"atomistic"` | `"coarse_grained"` | Force-field resolution |
| `target_density` | float | `0.9` | Target density: g/cm³ on the atomistic route, beads per sigma³ on the coarse-grained one (the build box is sized from it) |

#### `chemistry.node_type_map`

```json
"node_type_map": {
  "end": { "molecule": "[Si](C)(C)C", "is_end_cap": true },
  "A":   { "molecule": "Si",          "is_end_cap": false },
  "B":   { "molecule": "POSS",        "is_end_cap": false }
}
```

Built-in molecule names: `"Si"`, `"POSS"` (Si₈O₁₂ cage), `"POSS_AM0270"` (AM0270 aminopropyl POSS). Any SMILES string is also accepted.

#### `chemistry.edge_type_map`

```json
"edge_type_map": { "A": { "monomer": "PDMS" }, "B": { "monomer": "FPDMS" } }
```

#### `chemistry.monomers`

Built-in defaults:

| Name | SMILES | Description |
|---|---|---|
| `PDMS` | `[Si](C)(C)O` | Polydimethylsiloxane |
| `FPDMS` | `[Si](C)(CCC(F)(F)F)O` | Fluorinated PDMS |
| `Phenyl` | `[Si](C)(c1ccccc1)O` | Phenyl-PDMS |

Add custom monomers:

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

`auto_bridge`: when the chain head and node atom are the same element (e.g. both Si), automatically inserts a bridge atom. Set `false` to always use direct bonds.

### `conformation`

Stage 5. Two unrelated sets of keys live here, and a config uses one or the other.

The first three drive `ConformationManager`, which rewrites a data file that already has coordinates (the atomistic and legacy CG route):

| Key | Type | Default | Description |
|---|---|---|---|
| `overlap_cutoff` | float | `0.01` | Separation below which two atoms are pushed apart, in the data file's own units |
| `overlap_max_iters` | int | `10` | Passes of the overlap resolver before it gives up |
| `noise_magnitude` | float | `1e-4` | Uniform jitter on every atom, to break lattice degeneracy |

The rest drive `topon.conformation.place`, which draws a bead-spring build from the graph itself:

| Key | Type | Default | Description |
|---|---|---|---|
| `placement` | `"straight"` \| `"meander"` \| `"walk"` | `"meander"` | Chain shape at build |
| `coil_ratio` | float \| null | `null` | Strand contour over chord at build, mean over mean |
| `build_density` | float \| null | `null` | Bead density the paths are drawn at |
| `bond` | float | `0.97` | Design bond length, in sigma |
| `meander_waves` | float | `6.0` | Waves the meander spends its slack in, before the self-fold gate halves it |
| `min_bond` | float | `0.85` | Shortest bond a placed strand may carry |
| `min_self_separation` | float \| null | `null` | Closest a bead may come to a non-adjacent bead of its own chain. `null` uses the route's own floor: 1.0 sigma for a drawn path, 0.05 for a walk |
| `path_jitter` | float | `0.02` | Gaussian jitter on the interior beads of a straight path |
| `junction_shell_spacing` | float \| null | `null` | Seat the first bead of every chain leaving a junction on a spread shell at least this far apart |
| `entanglement` | object | see below | Target, distribution, designed pairs and the controller |

`coil_ratio` and `build_density` are **one knob**, and setting both is an error. The box scales as `rho^(-1/3)` and the contour does not move at all, so `coil_ratio = contour / chord` scales as `rho^(1/3)`, and fixing either fixes the other on a given graph at a given DP. `place` takes exactly one; the controller converts whichever you give into the one it turns.

#### Which shape, and why it matters more than the density

Measured on the N20 reference graph (the DP-20 end-linked reference network):

| placement | build density | Z1+ per DP-20 strand, build state | after compression to rho 0.3075 |
|---|---|---|---|
| reference (`fix bond/create`) | -- | -- | **0.178** |
| `walk` | 0.145 | 0.300 | 0.335 |
| `walk` | 0.095 | 0.250 | 0.288 |
| `walk` | 0.035 | 0.232 | 0.262 |
| `meander` | 0.050 | **0.189** (KS p = 1.0 vs reference) | 0.239 |
| `meander` | 0.040 | 0.192 | 0.224 |

The compressed column is not one protocol. The three `walk` rows were run with the minimiser stage 1, which stretched 85 bonds to 1.70 sigma and left 57 threaded, so part of their excess is the minimiser's own; the two `meander` rows are the crossing-free protocol (labelled `limit` in `CALIBRATION`), re-quenched to T = 0.4. `CALIBRATION` carries that label on every row and only the crossing-free ones steer the controller.

A random walk floors near Z = 0.23 per DP-20 strand whatever the build density, because coiled chains collapse during the push-off at fixed volume and trap the crossings. The meander at the same state reproduces the reference's per-strand distribution exactly. Density moves Z over the range shape leaves it, not the other way round.

`straight` is the chord with a jitter. It is the cheapest and the least entangled, and it is what `meander` falls back to when a chord is within 3 % of its contour and there is no slack to wave. On its own it only clears the gate near that limit. A straight strand's bonds are `chord / n_bonds`, so a build at coil ratio 1.51 gives 0.64 sigma and every strand fails `min_bond`. Use it where the chords nearly are the contour, or to see what a network with no coil at all does; it is not a build route for a coiled network.

#### The gate every strand passes before it is written

`place` checks three things and reports what it saw in `Placement.guard_report()`:

- every bond at or below `bond` (a longer one is a strand whose chord does not fit in its DP, which is a graph problem, reported as `overstretched`);
- every bond at or above `min_bond`;
- no bead within `min_self_separation` of a non-adjacent bead of its own chain.

Failures are reported, not raised, so a sweep across densities gets its answer instead of an exception. `guard_report()` gives the extremes over the build, how many strands failed and why, the realised contour as a fraction of the design one, and what the coincidence pass found. A clean build is normal but not guaranteed. The N20 reference graph at rho 0.05 comes back with 0 of 4644 strands failing, and the same graph at the compressed density rho 0.3075 (coil 2.77) with 43, all marginal self-contacts on strands whose slack cannot be spent without the path touching itself even at the minimum wave count.

Placing is not free. On the 95 364-bead N20 graph it takes 145 s for the meander at rho 0.05, 415 s at rho 0.3075 (more slack, more wave-halving) and 114 s for the walk. Roughly half of that is the coincidence pass.

This is not cosmetic. `meander_to_length` at its six-wave default is about three beads per wave at DP 20, and resampling that at equal arc length gives bonds of 0.17 sigma with beads jammed between their own second neighbours. The push-off then separates them *through* the bond that lies between, and a threaded bond is what lets two strands cross later (57 of them took N20 from Z 0.19 to 0.24). `meander_chain` meets the gate by halving its wave count and by opening the fold (`topon.conformation.paths.unfold`); what it cannot meet is reported rather than written silently.

**On a lattice, chords cross, and a meander leaves beads on its chord near the
ends.** Those two facts together put beads at exactly 0.0 sigma from each other.
Measured on the N20 graph at rho 0.05, 44 bead pairs sit inside 0.05 sigma, 40 of
them on strands whose chords intersect (median chord-to-chord distance 0.0000),
and 27 of the 44 sit two places from a junction, where the wave envelope
`sin^2(pi t)` has barely opened. `separate_coincident` clears them all to just
above the floor in under eight rounds. `junction_shell_spacing` attacks the same
thing from the other side by seating the chains leaving one crosslink on a
spread shell, so the near-end beads are not on the chord to begin with.

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
| `target_hist` | list \| null | `null` | Per-strand Z distribution to compare against, as fractions. Reported as a KS p-value; nothing is tuned to it |
| `shells` | object | `{}` | Neighbour-shell mix the designed pairs are drawn from, numbered from 1 |
| `pairs` | list | `[]` | Named pairs as `[chain_a, chain_b, windings]` |
| `controller.max_rounds` | int | `4` | Build-relax-measure rounds before the controller gives up |
| `controller.tolerance` | float | `0.15` | Relative `abs(Z - target) / target` accepted as converged |

Every model in this subtree refuses unknown keys, not just the `conformation` section itself. Pydantic's default is to ignore them, and `"target_z": 0.18` with a lower-case z was dropped silently, leaving `target_Z` at null. The controller then ran one round with no target and reported it as unconverged. That is an hour of LAMMPS spent on a typo with nothing in the output pointing at it. `topon doctor`'s `unknown_config_keys` rule skips the conformation subtree, so the schema is the only thing standing here.

**A random walk is not failed for coming back near itself.** The self-contact
floor belongs to the route. A meander or a jittered chord that puts a bead
within 1 sigma of a non-adjacent bead of its own chain has folded, while a
freely jointed walk does that constantly (all 972 DP-100 walks on the N100
reference graph do). What a walk must not have is a hard overlap, and the
closest pair in that build sat at 0.003 sigma, where WCA is of order 1e9 kT.
So `place` uses 1.0 sigma for a drawn path and 0.05 for a walk unless
`min_self_separation` says otherwise, and `separate_coincident` clears anything
inside 0.05 sigma *anywhere* in the build (junctions included as fixed
obstacles, because a junction's position is shared by every strand that meets
there) and puts the bonds back afterwards.

**`target_Z` is a final-state number.** Z1+ counts the kinks of the shortest paths between the junctions *as they currently sit*, so it moves by 10-20 % through equilibration and compression even with zero bond crossings and no free-ended chains at all. N100 went 1.16 after push-off, 1.00 after equilibration at constant volume and 1.11 after compression, with every bond under 1.2 sigma throughout. The build state is where the controller starts, not what it closes on.

Chain ids in `pairs` index `topon.conformation.strand_plans`, which is the graph's edge order and the order a `Placement` keeps its strands in. `topon.conformation.entanglement.requests_from_config` also accepts an edge key `[u, v, k]` in place of an index.

`shells` says what the conformation stage should be handed to route; choosing *which* pairs is the assignment stage's job (`assignment.entanglements.shell_weights` and `select_by_shells`), and a driver calls that and passes the result in.

#### The controller

```python
from topon.conformation.entanglement import controller

manifest = controller(graph, config.conformation, runner, dp=20)
```

`runner` is a callable the caller supplies. It is given a `RoundPlan` (the round number, the placement, which knob was turned and to what, the DP and the seed) and must return a `RoundResult`, or a dict carrying at least `z_final`. Building the system, running the relaxation protocol and measuring Z1+ are deliberately *not* this module's business, which keeps the conformation stage free of LAMMPS and makes the loop testable against a known response.

To wire it to a real protocol, a driver uses `topon.writers` to write the build and the five-stage push-off and `topon.simulation.protocols.StagedRun` to run and gate it, and measures Z1+ with its own primitive-path code. Z1+ is not redistributable, so that driver does not ship with topon.

The starting point comes from the shipped calibration (`topon.conformation.entanglement.CALIBRATION`), which holds every measured (actuator, Z) pair from the crossing-free runs together with the state and protocol each was measured under. Only final-state points measured with the crossing-free protocol steer; the minimiser's points are kept and labelled, because that protocol stretched 85 bonds to 1.70 sigma and added crossings of its own. Over the measured range Z is a power law in the actuator (exponent 0.87 for the DP-20 meander, 0.30 for the DP-100 walk), so two rounds fix the curve for the graph in hand.

A `target_Z` below the lowest value a route has been seen to reach returns a warning naming the floor and, for the walk, the meander route that goes lower:

```
target_Z 0.18 is below the lowest final-state Z the walk route has reached at
DP 20 (0.262, at build_density 0.035). ... Switch conformation.placement to
'meander' to go lower; it is the shape, not the density, that sets the floor.
```

#### Designed pairs, and when they are refused

**What this delivers today.** The budget half is complete and measured. The routing half does not yet clear the gate on a strand that was already drawn as a meander. Composing a braid onto one and waving the remaining free run back out to the contour leaves beads two to five places apart at 0.5 to 1.0 sigma at the seam where the braid meets the meander, across every geometry tried. That is the threaded-bond pathology the gate exists to catch, so with `strict` (the default) the strand goes back to the path the placement drew and the request is refused with the reading that failed. For delivered windings today, use the pipeline's own route (`assignment.entanglements` with `method: "waypoint"`), which builds the braided strand *from its chord* rather than composing onto a meander.

A winding is paid for in contour. The two partners leave their chords, go round each other and come back, and a strand has only `n_bonds * bond` of contour to spend. `route_designed_pairs` measures the routed path before drawing anything and refuses what does not fit, naming the smallest DP that would carry it:

```
pair (0, 514) x2: refused, the contour budget does not allow it: routing 2
winding(s) needs 33.2 sigma of path and DP 20 carries 20.4. The routed path is
33.2 sigma and these strands carry 20.4; DP 34 is the smallest that would carry
it at this geometry
```

A second budget can bind instead, and it is not DP's to fix. The braid needs `windings * pitch + 2 * ramp` of the shared axis, and two short chords cannot spare it. That refusal says to lower the build density so the chords lengthen, or to pick a partner in a closer shell. A third is the clearance. A braid squeezed into a chord too short for it brings its two arms within a fraction of sigma, which the push-off resolves by pushing them through each other, so that is refused too.

The minimum DP a refusal names holds *at that geometry*. Rebuilding at a higher DP and the same build density puts more beads in the box, so the box grows as `DP^(1/3)` and every chord with it. Measured on the N20 graph, a pair needing DP 24 at DP 20's box needed DP 25 once rebuilt at DP 24. Rebuild at the same *box* instead (raise the density with the DP) and it is reached in one step.

### `output`

| Key | Default | Description |
|---|---|---|
| `lammps_data` | `true` | Write LAMMPS data file (not read by `Pipeline`; see below) |
| `lammps_convention` | `"topon"` | Coarse-grained data-file convention (see below) |
| `lammps_inputs` | `true` | Write LAMMPS input scripts (not read by `Pipeline`; see below) |
| `visualization` | `true` | Write HTML visualization (not read by `Pipeline`; see below) |
| `analysis_report` | `true` | Write analysis report (not read by `Pipeline`; see below) |
| `save_attributed_graph` | `true` | Save attributed graph as `.gpickle` (not read by `Pipeline`; see below) |
| `export_graphml` | `false` | Write the graph (chains and entanglement edges) as `<study.name>.graphml` in the study folder; `topon generate --export-graphml` sets it |
| `export_npz` | `false` | Write the graph as `<study.name>.npz` in the study folder; `topon generate --export-npz` sets it |

`lammps_data`, `lammps_inputs`, `visualization`, `analysis_report` and
`save_attributed_graph` are accepted, but `Pipeline` does not read them. It
always writes the data file and the input scripts, and it writes no HTML
visualization, analysis-report file or `.gpickle`.

#### `output.lammps_convention`

| Value | Atom types | Molecules |
|---|---|---|
| `"topon"` (default) | The distinct `bead_type` properties, numbered in sorted order | One for the whole network |
| `"endlinked"` | 1 = chain-end bead, 2 = chain interior, 3 = junction | One per chain and one per junction; a dangling chain's free end belongs to its chain, and a primary loop is a ring whose two ends bond to the same junction |

The end-linked convention is the one `fix bond/create` datasets use, so the
same parsers read topon output and reference output alike. Pair it with
`assignment.dp_distribution.endlinked_dangling` when the bead count has to
match as well. Atomistic output ignores the key.

For a bead-spring build that never goes through the chemistry stage,
`topon.writers.write_endlinked` writes the same convention straight from a
`topon.conformation.place` placement, with image flags that reconstruct the
coordinates they were written from.

### `simulation` (raw section, not schema-validated)

Controls the LAMMPS scripts stage 6 writes.

| Key | Default | Description |
|---|---|---|
| `protocol` | `"pushoff"` | Coarse-grained relaxation protocol (see below) |
| `rho_final` | `null` | Number density (beads per sigma³) stage 4 compresses to; `null` makes stage 4 a settle. It is not `chemistry.target_density`, which sizes the build box |
| `final_bond_style` | `"fene"` | `"quartic"` adds stage 6, for deformation runs |
| `pair_style` | `"attractive"` | `lj/cut 2.5`, or `"repulsive"` for WCA. Ignored by `pushoff`, which pins WCA |
| `include_angles` | `true` | Write angles in the data file |
| `remove_cg_angles` | `true` | Drop them before the run (Kremer-Grest chains are flexible) |

#### `simulation.protocol`

| Value | Stage 1 | Scripts | Keeps a prescribed entanglement? |
|---|---|---|---|
| `"pushoff"` (default) | FENE + WCA, `nve/limit` 0.02, no minimiser | 5 | Yes |
| `"hardcore_min"` | WCA, conjugate-gradient minimisation | 3 | The push, yes; the minimiser, no |
| `"soft_push"` | `pair_style soft` ramped 0 to 30, minimisation | 3 | No |

`pushoff` is the default because a minimiser resolves an overlap by whatever
move lowers the energy, and with a harmonic bond the cheapest move is often to
stretch a bond and let the overlapping bead through it. Measured on the N20
build, 85 bonds reached 1.70 σ during stage 1 and 57 were still threaded at
1.3-1.4 σ in every later stage, and Z per bridge drifted 0.19 → 0.24 as those
strands crossed. Under the push-off the same build ends with 3 of 93 128 and Z
flat to the Z1+ noise level.

The other two are for reproducing earlier runs and are crossing-prone.

The five stages, and what each leaves behind:

| script | what it does | writes |
|---|---|---|
| `minimize_1_serial.in` | push-off, `nve/limit` 0.02, dt 0.002, 30k steps | `stage1_min.data` |
| `minimize_2_parallel.in` | cap to 0.05 for 20k, then free NVE 20k | `stage2_pushoff.data` |
| `minimize_3_parallel.in` | equilibrate at the build density, damp 10, 200k | `stage3_build_equil.data` |
| `deform_4_parallel.in` | affine compression to `rho_final`, then settle | `stage4_final_T1.data` |
| `quench_5_parallel.in` | T 1.0 → 0.4 over 50k, settle 20k | `stage5_final_quench.data` |

The names of stages 1-3 are historic (nothing in this protocol minimises)
and were kept so a caller that has always run `minimize_1_serial.in` still
does. The data-file names are the interchange convention of the end-linked
validation scripts, so those scripts read a topon run and a reference run the
same way. Ask `LammpsInputGenerator.stages("cg")` for
the list rather than hard-coding it.

Step counts live under `experimental.cg.pushoff`, one block per stage:

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
equilibration length is `stage3.steps`, which defaults to the protocol's
200 000 rather than that key's 10 000.

#### Acceptance gates

`topon.simulation.protocols` reads every checkpoint and applies two gates.

```python
from topon.simulation.protocols import StagedRun, measure_stages

run = StagedRun(sim_dir=sim, stages=gen.stages("cg"), omp=8)
report = run.run()            # gates after every stage; stops at the first failure
print(report.render())

measure_stages(sim, gen.stages("cg"))    # or gate a run that already happened
```

- **Zero bonds above 1.2 σ** after stage 2 and at every later stage, scanning
  every bond. `mode="instant"` (the default) fails on the count, which is the
  criterion as written; `mode="persistent"` fails only on a bond long at more
  than one stage, which is what a threaded bond is and what a thermal
  excursion is not. Both counts are always reported.
- **Z1+ unchanged between stage 3 and stage 5**, applied only to a run that did
  not compress and for which a Z1+ measurement exists. Under compression Z1+
  moves 10-20 % with no crossing anywhere (it counts the kinks of the shortest
  paths between the *current* junction positions, so part of it slides off as
  junctions move), so it is reported per stage always and gated only where it
  must hold. `gate_z=True` / `False` overrides.

  The ±0.01 tolerance is absolute and calibrated on the reference-scale build
  (4 427 bridges, where counting noise on the mean Z is ≈ 0.007). On a smaller
  cell it is tighter than the noise (192 bridges gives ≈ 0.04), so pass
  `z_tolerance=` scaled to the build rather than reading a small system's
  failure as a crossing.

Temperature is read from each checkpoint's velocities, because Z1+ and the
chain statistics both depend on it and a comparison across a temperature gap is
measuring the gap. Z1+ itself is not in this repository (its licence forbids
redistribution); pass the numbers in with `z_by_stage=`.

#### Deformation runs

When the final force field is the quartic bond, set
`simulation.final_bond_style: "quartic"`. It converts only at the end (20k
steps at the target temperature after the FENE relaxation), because the quartic
bond breaks silently above 1.5 σ (64 broke in a smoke run and split chains)
while FENE errors out instead. When the OpenMP package is active the generated
stage wraps the style in `suffix off` / `suffix on`, because
`bond_style quartic/omp` computes correct forces
but leaves the subtracted bonded-LJ term out of `E_pair` and the virial
(pressure 4.79 instead of 0.05 at ρ = 0.30), so any stress read from a run with
the OpenMP package still has to be checked against a serial `run 0`.

`fix deform ... erate` re-bases its reference box at every `run` command, so a
deformation split across several runs multiplies the box by the same factor
once per chunk. Stage 4 drives the box to an absolute `final` size in a single
`run`, so it is not exposed to that. topon does not generate a *tensile*
stage, and one written by hand with `erate` must follow the same rule.
