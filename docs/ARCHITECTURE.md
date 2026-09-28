# topon architecture

topon is a Python toolkit that builds polymer networks for LAMMPS molecular dynamics. It decides the topology as a graph first and then maps that graph to chemistry and coordinates. The same graph can produce a coarse-grained Kremer-Grest system or an atomistic DREIDING system with no change to the topology code.

This document describes the package layout for contributors. [USAGE.md](USAGE.md) explains how to run topon.

---

## 1. The topon family

The package contains two independent sub-systems with related jobs.

| | **core topon** (polymer networks) | **simbox** (molecule packing) |
|---|---|---|
| Implementation path | `topon/{topology,assignment,chemistry,conformation,writers}/` | `topon/simbox/` |
| Entry point | `topon` CLI / `topon.pipeline.Pipeline` | `topon simbox` CLI / `topon.simbox` API |
| Resolution | atomistic (DREIDING) or coarse-grained (Kremer-Grest) | atomistic (DREIDING) |
| Force field | DREIDING / Kremer-Grest | DREIDING |
| LAMMPS data | `atom_style full`, wrap-only (7-column atom rows, no image flags) | `atom_style full`, wrap-only |
| Pipeline | the six stages below | independent packing flow |
| Crosslinks | Y-merge (CG) / chemistry-defined (atomistic) | reaction templates (epoxy/amine, etc.) |
| Topology shape | lattice graph (SC / BCC / FCC / Diamond / MIX, configurable functionality) | molecule library + grid packing |

A third, smaller utility (`topon/singlechain/`) handles single-chain solubility calculations. The main pipeline does not use it.

---

## 2. The six-stage pipeline

The `Pipeline` class in [`topon/pipeline.py`](../topon/pipeline.py) runs the core topon pipeline. Its `run()` method calls six stages in order.

```
config.json
    │
    ▼
load_config_full() ───► ToponConfig (Pydantic) ──► Pipeline.run()
                                                       │
   Stage 1: Topology ──────────────────────────────────┤
   Stage 2: Analysis ──────────────────────────────────┤
   Stage 3: Assignment ────────────────────────────────┤
   Stage 4: Chemistry ─────────────────────────────────┤
   Stage 5: Conformation ──────────────────────────────┤
   Stage 6: Output ────────────────────────────────────┘
                                                       │
                                                       ▼
                                          <output_dir>/<name>/
                                            ├── topology/
                                            ├── 02_Chemistry/
                                            ├── 03_Conformation/
                                            └── 04_Simulation/
```

### Stage 1. Topology
`Pipeline._run_topology_stage` runs this stage from `topon/topology/`.

This stage generates or loads a NetworkX `MultiGraph`. Nodes are network junctions and edges are polymer chains. There are two sources.
- `source="generate"` runs the pure-Python generator (`topon.topology.generator_python.PythonTopologyGenerator`) in process. When `topology.generator.exe_path` is set, it calls the C generator (`generator.exe`) through `topon.topology.generator.run_generator` instead and then loads the `.nodes`/`.edges` files it writes.
- `source="load"` reads existing `.gpickle` or `.nodes`+`.edges` files through `topon.topology.loader.load_graph`.

#### Lattices and range

The lattice types are `SC`, `BCC`, `FCC` and `Diamond`, each with a fixed
neighbor pattern at the default range, and `MIX`, which overlays SC, BCC
and FCC basis sites in one cubic cell at set fractions. The candidate
edges are defined by a lattice and a range. `topology.generator.neighbour_cutoff`
sets the range in cell units for every lattice type. At the default 1.0
the pure lattices keep their canonical nearest-neighbor pattern. At any
other value every pair of sites within the range (under the minimum
image) is a candidate edge, which is the search `MIX` always uses.
Crosslinkers several site spacings apart then become candidate partners
for the sculptor. This lets a sculpted graph match a reaction-generated
network (three to four SC shells for DP-20 strands, six to eight for
DP-100). The graph records the cutoff and the shell distances it produced
(`G.graph["neighbour_cutoff"]`, `G.graph["shell_distances"]`). The shell
table and the key resolver live in `topology/shells.py`. `periodicity`
opens individual axes and gives a free surface on that face. An open axis
is never bonded across, at any range. Both generators support all five
lattices, per-axis boundaries and the range, and they build identical
lattices for every combination.

#### Strict and exact searches

The lattice gives a superset of candidate edges, and a sculptor picks the
network from it. `topology.generator.search` selects the sculptor.
`generator_python.run_single_trial` is the strict sculptor. It removes
candidate edges one at a time, samples `max_trials` trials and records
every removal in `G.graph["move_history"]`, which a sculpting animation
replays.
`topology/degree_matching.py` is the exact search. It assigns the
requested degree to every site and completes the degree sequence by
augmenting paths (i.e., a degree-constrained subgraph), so it samples no
trials and keeps no move history. If `search` is unset, it resolves to
`exact` when the degree distribution pins every degree from 0 to
`max_functionality` and to `strict` otherwise.

The two searches fail in different cases. The strict sculptor struggles
with fully pinned requests. A near-complete tetrafunctional target with
dangling-end sites fails on every scaffold in either generator, while the
exact search finds it in under two seconds. The exact search
needs a scaffold with spare candidate edges. It refuses a request that
pins most of its sites to the scaffold's own coordination (e.g., Diamond
at `max_functionality: 4` with 73 % of the sites at the ceiling) and
reports this instead of retrying. Both searches return the same kind of
graph. Every site is a node, vacancies stay at degree 0, the builder's
`box`, `periodicity` and shell attributes are kept, and edges follow the
scaffold's own order. Both searches exist in both generators. The C
generator runs the exact search with `--search=exact` (a port with the
same steps and constants). When `exe_path` is set, the pipeline sends
exact requests to C. The exception is a request that forces double edges
or reserves defect capacity, which only the Python search accepts.

#### C and Python generators

The C source in [`topon/topology/csrc/`](../topon/topology/csrc/) is a
standalone searcher. It runs without Python and is meant for long
exhaustive searches. The pure-Python `generator_python.py` is the pipeline
default and generates likely networks quickly in process, with no
compiler. The two are independent programs. Nothing in `csrc/` is called
from Python, and it should not get a Python binding. Only the shared parts
(lattice construction and the `.nodes`/`.edges` format) must stay in step.
The two agree in distribution. Individual draws differ because the C
generator uses its own random stream (seeded from the clock unless
`TOPON_SEED` is set).

On SC at `max_func=4`, the time to first success is 0.01 s in Python and
0.04 s in C at 6³ (process startup dominates), 1.5 s and 0.11 s at 12³,
and 6.7 s in C at 24³ (13824 nodes), where Python would run for hours.

The stage produces `self.graph` (the annotated `MultiGraph`) and `self.dims` (the box size as an `np.ndarray`), plus its section of the run manifest.

#### Run manifest

`Pipeline` writes `manifest.json` into the run directory through
`topon/core/manifest.py`, with one section per stage. The stage 1 section
records the lattice, the search, the requested and achieved degree counts,
the giant-component fraction, the seed and the timing. `topon inspect`
prints it next to the stage-output summary. The manifest is advisory. No
stage reads it back, so a missing or half-written manifest does not change
a run.

#### Periodic cell

Generators write the exact repeat distance into `G.graph["box"]`, and
`.nodes` files carry it in a `# BOX Lx Ly Lz` header. `infer_dims_from_graph`
returns that value when it is present. For older graphs without it, the
function estimates the cell from the coordinate extent (`max - min + 1`).
The estimate is exact for SC, whose sites are integer-spaced, but too
large for any lattice with fractional basis sites (BCC and FCC body and
face sites sit at +0.5 and Diamond sites at quarter-cell offsets, so a
4x4x4 BCC or FCC gives 4.5). Every minimum-image calculation uses
`self.dims`, so an oversized box breaks the `bond < box/2` rule of Design
Principle 3. It sends about a third of BCC edges (a quarter of FCC) to the
wrong periodic image, where they are built at twice their true bond
length. Any new lattice with non-integer sites must record its cell for
this reason.

### Stage 2. Analysis
`Pipeline._run_analysis_stage` runs this stage through `topon/assignment/manager.py:analyze`.

This stage computes graph statistics (degree distribution, connectivity, and the defect and entanglement *capacity* of the topology) before any annotation is written. The result is stored in `self.analysis_report`. The stage does not modify the graph.

It is separate from the `topon/analysis/` package, which provides `analyze_graph()` for the `topon analyze` sub-command. `Pipeline` does not import from `topon.analysis`.

### Stage 3. Assignment
`Pipeline._run_assignment_stage` runs this stage from `topon/assignment/`.

This stage annotates the graph in place. `AssignmentManager` (`assignment/manager.py`) calls the module-level assigners listed below.

| Concern | API | Lives in |
|---|---|---|
| Orchestration | `AssignmentManager.run()` | `assignment/manager.py` |
| Node types (degree-based, positional, random, explicit) | `assign_node_types()` | `assignment/node_types.py` |
| Edge types (uniform, random, composite) | `assign_edge_types()` | `assignment/edge_types.py` |
| Degree-of-polymerization distribution (Schulz-Zimm PDI) | `assign_dp()` | `assignment/dp_distribution.py` |
| Copolymer sequences (Block, Random, Alternating, Gradient) | `generate_monomer_sequence()` | `chemistry/sequences.py` |
| Defects (primary loops, secondary loops, triangles, four-cycles, sol) | `apply_defects()` | `assignment/defects.py` |
| Entanglements (which chain pairs to entangle) | `select_entanglements()` | `assignment/entanglements.py` |
| Re-attribute existing graph (round-trip workflows) | `GraphAttributor` (class) | `assignment/attributor.py` |

After this stage the graph carries every annotation the later stages need.

### Stage 4. Chemistry
`Pipeline._run_chemistry_stage` runs this stage from `topon/chemistry/`.

`ChemistryBuilder` (`chemistry/builder.py`) builds an RDKit `Mol` with 3D coordinates from the graph. It is a single class for both resolutions, with no separate CG and atomistic classes. The supporting modules are listed below.

| Module | Role |
|---|---|
| `chemistry/builder.py` | The `ChemistryBuilder` class, with most of the stage-4 logic |
| `chemistry/sequences.py` | Shared monomer sequence helpers used by the builder |
| `chemistry/embed.py` | ETKDGv3 per-chain conformer embedding |
| `chemistry/{dreiding,kg}/` | Stub sub-namespaces reserved for force-field-specific chemistry (`kg/` has only a docstring, and `dreiding/` is a 1-line stub). The CG or atomistic switch happens inside `builder.py` through `config.chemistry.model_type`. |

This stage also writes the first LAMMPS files to `<output_dir>/02_Chemistry/`.

- `system.data` holds atom positions and connectivity (written by `CGWriter` or `DreidingWriter` in `topon/writers/`).
- `system.in.settings` holds force-field coefficients. `DreidingWriter` writes it on the atomistic route, and the coarse-grained route leaves it as a stub.
- `system.groups` defines the `nodes` and `beads` LAMMPS groups.
- `system_nodes.displace`, `system_backbone.displace` and `system_grafts.displace` (plus `system_pendant.displace` and `system_hydrogens.displace` on the atomistic route) are the displacement files for stage 5.

### Stage 5. Conformation
`Pipeline._run_conformation_stage` runs this stage from `topon/conformation/`.

Stage 5 has two entry points, one for each kind of build.

`ConformationManager` (`conformation/manager.py`) is the pipeline's path for a system that already has a data file. It reads the chemistry-stage output, applies the displacement files, adds a small uniform noise to break degeneracy, and resolves overlaps iteratively.

`conformation/placement/chains.py::place` builds a bead-spring system directly from the graph. `Pipeline` does not call it (`Pipeline._run_conformation_stage` constructs only `ConformationManager`), so it is used through the API as `topon.conformation.place`. It takes a graph, a DP and one of three chain shapes. It sizes the build box from the bead count and the build density (or, equivalently, a coil ratio), draws every strand and checks each strand against a gate. It runs no dynamics and writes no file. It returns the coordinates and the gate readings that show whether they are usable. `conformation/packing/` is still a 1-line stub.

The gate checks three things per strand (every bond at or below the design length, every bond at or above `min_bond`, and no bead within `min_sep` of a non-adjacent bead of its own chain). A path can have the right length and end on both junctions and still be unusable. For example, a six-wave meander at DP 20 resamples to 0.17-sigma bonds with beads packed between their own second neighbors. The push-off then separates those beads *through* the bond between them, and that threaded bond lets strands cross later.

Three more modules are opt-in and not on the default pipeline path. The entanglement workflows use them.

- `conformation/entanglement/` builds chain paths that wind around each other a set number of times. `waypoints.py` draws a chain through points the caller chooses (`Site(at, turns)`) and is the current approach. `braid.py` and `allocation.py` are an earlier search-based construction that picks its own positions. They are kept because the waypoint path has no equivalent yet for their budgeting and obstruction checks. `designed.py` puts a braid on a chain that is already placed. It refuses a request whose windings do not fit in the strands' contour and names the minimum DP. `control.py` closes a loop on the measured Z1+ of the relaxed system. It is exported as `topon.conformation.entanglement.controller` and takes a caller-supplied `runner`, so the stage does not depend on LAMMPS.
- `conformation/junction_shell.py` spreads the chains leaving a crosslink so their first beads do not overlap. The shell radius grows with functionality.
- `conformation/paths.py` is plain geometry (two junctions and a bead count in, a path out). `bridging_walk` is melt-like and random. `route_through` is deterministic, for a prescribed topology that must be repeatable. Both keep every bond length exact and end on both junctions. `Clearance` lets either of them draw around the beads already in the box, so the following minimization does not resolve an overlap by pushing two chains through each other. Two routines handle strands with no second endpoint, and the pipeline's chemistry stage calls both. `closed_meander` draws a primary loop as a regular polygon through its junction, and `free_walk` draws a sol chain.

The conformation defaults are in `Pipeline._DEFAULT_CONFORMATION`.

```python
{"overlap_cutoff": 0.01, "overlap_max_iters": 10, "noise_magnitude": 1e-4}
```

The stage writes `<output_dir>/03_Conformation/system_relaxed.data`.

The simulation box comes from stage 1, not from the coordinates. Callers
pass `lattice_box=dims` to `apply_displacements`, which gives a box of
`dims * scale`. Only when it is omitted does the manager estimate
`(max node coord + 1) * scale` from the `.displace` files. The box must
match the cell that stage 4 used to route chains across the periodic
boundary. If the two differ, a chain that wraps under one period lands in
a box of another size and its closing bond stays stretched across the
system. The two estimates coincide for SC. Passing the box also makes
the written box exactly `volume^(1/3)`, so the target density is reached
on every lattice.

### Stage 6. Output
`Pipeline._run_output_stage` runs this stage through `topon/writers/` (`LammpsInputGenerator`).

This stage writes the LAMMPS input scripts. Their content depends on
`simulation.protocol`.

- `pushoff` (the coarse-grained default) has five stages (capped-displacement push-off under FENE + WCA, uncapped push-off, equilibration at the build density, affine compression, quench) and no minimizer.
- `hardcore_min` and `soft_push` are the older three-stage decks with a minimizer. They are kept to reproduce earlier runs. The atomistic route always uses the `soft_push` layout, because its bonded terms are stiff and a capped push-off would take much longer than a minimizer to resolve the same overlaps.

The scripts go to `<output_dir>/04_Simulation/*.in`. The two protocol
families differ in which scripts they write and what each one leaves
behind, so use `LammpsInputGenerator.stages(model_type)` to get the list
instead of hard-coding the names.

`Pipeline.run()` does not run the simulation. `topon/simulation/` does, with `SimulationRunner` for a plain sequence or `protocols.StagedRun` to run the stages and apply the acceptance gates after each one.

---

## 3. Module map

The table lists every directory under `topon/`. Modules above the dashed row belong to the main pipeline, and the sub-systems below it are independent.

| Directory | Files / Subdirs | Role |
|---|---|---|
| `topology/` | 7 files (`shells.py` holds the neighbor-shell table and cutoff resolver, `degree_matching.py` the exact degree-sequence search) + `csrc/` (C generator) + `network/` (a thin loader wrapper), `sequence/`, `simple/` (stubs) | Graph generation and loading. Stage 1. |
| `analysis/` | 3 files (`report.py` is `topon analyze`, `run_summary.py` is `topon inspect`) | Read-only graph statistics for the `topon analyze` CLI, and the post-run summary. Not used by `Pipeline` stage 2, which calls `AssignmentManager.analyze()`. |
| `assignment/` | 8 files | Graph annotation (node/edge types, DP, defects, entanglements, copolymers). Stage 3. |
| `chemistry/` | 4 files (`builder.py` is most of stage 4), and `dreiding/` and `kg/` are stubs | RDKit Mol construction with 3D coords. Stage 4. |
| `conformation/` | `manager.py` (data-file route) and `placement/chains.py` (bead-spring route, `place`) are stage 5. `entanglement/` (waypoint and braid construction, designed pairs, the Z controller) and `junction_shell.py` are opt-in and off the default path. `packing/` is a stub | Chain placement, overlap resolution, and designed entanglement geometry. Stage 5. |
| `writers/` | 8 files (`lammps_endlinked.py` writes the `fix bond/create` convention straight from a placement) | LAMMPS data and input-script writers. Stage 4 + Stage 6. |
| `forcefield/` | 4 files | DREIDING parameter parser and Kremer-Grest parameters. Read by chemistry/writers. |
| `config/` | 4 files | Pydantic `ToponConfig` schema, `load_config()` and `load_config_full()`. |
| `diagnostics/` | 2 files (`rules.py` is the rule registry) | The semantic checks behind `topon doctor`. |
| `core/` | 3 files (`manifest.py` is the run manifest read/written across stages) | Shared types, protocols and run-level artifacts. |
| `utils/` | 4 files | Shared helpers (e.g., `write_lammps_displacement_file`). |
| `pipeline.py` | - | The `Pipeline` orchestrator class. |
| `cli.py`, `shell.py`, `__main__.py` | - | CLI dispatch and the interactive `topon>` shell. |
| `workflows/` | 4 files | High-level workflow helpers (`cg_network`, `atomistic_network`, `reactive_crosslink`). |
| - | - | - |
| `simbox/` | 8 files | Independent molecule packer + crosslink-template emitter. DREIDING-only. |
| `singlechain/` | 3 files | Single-chain solubility utility. |
| `simulation/` | `runner.py` + `protocols/` (`gates.py`, `staged.py`) | LAMMPS subprocess runner and the acceptance gates that check a relaxation (bond histogram per checkpoint, Z1+ held where it must hold, temperature from the velocities). |

---

## 4. Design principles

Every change should follow these rules.

1. **Graph-first separation.** Topology is decided as a NetworkX graph before chemistry is built. The same graph can produce a CG system or an atomistic system by switching the chemistry builder. Topology code never sees atom types, coordinates or force-field details.

2. **Six-stage pipeline, one-way data flow.** Each stage has a single responsibility. Downstream stages do not reach back into upstream state. The stage descriptions in §2 and the module-boundary table below define these boundaries.

3. **LAMMPS data files are wrap-only.** The core topon and simbox writers (`topon/writers/`, `topon/simbox/writer.py`) write 7-column Atoms rows (no `ix iy iz`). LAMMPS handles periodic boundaries through its neighbor and ghost-atom lists under the minimum image. This is correct as long as every bond is shorter than `box/2`, which holds for KG and DREIDING networks because their chains rarely wrap differently from each other. The one exception is `write_endlinked` (`topon/writers/lammps_endlinked.py`), which writes image flags so the coordinates it was given can be reconstructed.

4. **Configuration through Pydantic, no globals in stage code.** Every stage module reads from a `ToponConfig` (or its raw-dict supplements). Code inside `topon/` does not use `os.environ` lookups, module-level constants that change behavior, or hard-coded paths. The config is loaded once, by the CLI or by the code that builds the `Pipeline`, and passed down.

5. **Module boundaries.** Each module has a fixed scope.

| Module | Does | Does NOT |
|---|---|---|
| `topology/` | Generate connectivity graph | Assign types, coordinates, force field |
| `assignment/` | Assign node/edge types, DP, defects, entanglements | Generate chemistry or coordinates |
| `chemistry/` | Build RDKit molecular structure from graph | Generate placement coordinates |
| `conformation/` | Place atoms in 3D, resolve overlaps | Assign force-field types |
| `writers/` | Format and write LAMMPS files | Computation of any kind |
| `analysis/` | Compute graph statistics | Modify the graph |
| `simbox/` | Independent molecule packing sub-system | Interact with the main pipeline |

---

## 5. Configuration model

The package validates the JSON configuration with Pydantic (`topon/config/`). Three sections have no schema yet (`simulation`, `execution`, `experimental`). They are passed as a raw dict next to the validated config (see `Pipeline.__init__`). `conformation` is validated like any other section. `load_config_full` also returns a copy of it in the raw dict, because `Pipeline` and both workflow modules read it from there.

The top-level `ToponConfig` sections, in the order the pipeline uses them, are listed below.

| Section | Used by | Notes |
|---|---|---|
| `study` | all stages | `study.name`, `study.output_dir` |
| `topology` | Stage 1 | `source` (`"generate"` / `"load"`), `lattice_size`, `degree_distribution`, etc. |
| `assignment` | Stage 3 | sub-objects for entanglements, defects, grafts, copolymer sequences |
| `chemistry` | Stage 4 | `model_type` (`"coarse_grained"` / `"atomistic"`), `target_density` |
| `conformation` | Stage 5 | `overlap_cutoff`, `overlap_max_iters`, `noise_magnitude` for the data-file route, which is the one `Pipeline` runs. The bead-spring keys (`placement`, `coil_ratio` / `build_density`, `junction_shell_spacing`, `entanglement`) are validated here but read only by `topon.conformation.place` and the controller. `Pipeline` does not call these, so `topon generate` validates the keys and ignores them. Also passed through raw |
| `simulation` (raw) | Stage 4 (angles) and Stage 6 | relaxation protocol, LAMMPS pair_style, angle handling |
| `execution` (raw) | the `topon.workflows` runners, not `Pipeline` | LAMMPS subprocess settings (`auto_run`, `executable`, `n_procs`) |
| `experimental` (raw) | Stage 6 | feature-flagged extras |

`GeneratorConfig`, the per-class blocks under `assignment.defects` and the
whole `conformation` section refuse unknown keys. The other nested
sections ignore them, because a config may carry parameters for other
tools (e.g., under `topology`). For those sections `topon doctor` warns
through its `unknown_config_keys` rule.

[USAGE.md](USAGE.md) has the full key reference and example configs.

---

## 6. CLI surface

The `topon` CLI (`topon/cli.py`) maps each sub-command to a backend.

| Sub-command | Backend |
|---|---|
| `topon generate` | `load_config_full()` + `Pipeline.run()` |
| `topon validate` | `load_config_full()` + `validate_config()` |
| `topon doctor` | `topon.diagnostics` rule registry |
| `topon init` | writes a starter config (a copy of a demo config, or one built from prompts) |
| `topon inspect` | `topon.analysis.run_summary` |
| `topon analyze` | `topon.analysis.report.analyze_graph()` |
| `topon simbox` | `topon.simbox` API |
| `topon chain` | `topon.singlechain` |
| `topon recipes` | prints a table of common use cases |
| `topon shell` | the interactive `topon>` shell (`topon/shell.py`) |

[USAGE.md](USAGE.md) lists every flag.

---

## 7. Where to go next

| Question | Doc |
|---|---|
| How do I run topon end-to-end? | [USAGE.md](USAGE.md) |
| Which CLI flag does X? | [USAGE.md](USAGE.md) |
| What's the JSON config schema? | [USAGE.md](USAGE.md) (config-reference appendix) |
| Which demo shows X? | [`demos/README.md`](../demos/README.md) |
| What's changed across versions? | [CHANGELOG.md](../CHANGELOG.md) |
