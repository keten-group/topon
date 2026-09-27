# topon architecture

A graph-first Python toolkit for building polymer networks ready to drive LAMMPS molecular dynamics. Topology is decided as a graph first, then mapped into chemistry and coordinates; the same graph can produce a coarse-grained Kremer-Grest system or a fully atomistic DREIDING system without changing the topology code.

This document is the onboarding doc for new contributors. Read it first, then go to [USAGE.md](USAGE.md) for how to run things.

---

## 1. The topon family

Two peer sub-systems share the same Python package. They do related but distinct jobs.

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

A third, smaller utility, `topon/singlechain/`, handles single-chain solubility computations and is not used by the main pipeline.

---

## 2. The six-stage pipeline

The `Pipeline` class in [`topon/pipeline.py`](../topon/pipeline.py) is the single orchestrator for **core topon**. Its `run()` method calls six stages in order.

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
**Module:** `topon/topology/` &nbsp;**Code:** `Pipeline._run_topology_stage`

Generates or loads a NetworkX `MultiGraph`. Nodes are network junctions; edges are polymer chains. Two paths:
- `source="generate"` runs the pure-Python generator (`topon.topology.generator_python.PythonTopologyGenerator`) in process. When `topology.generator.exe_path` is set it calls the C generator (`generator.exe`) via `topon.topology.generator.run_generator` instead, then loads the `.nodes`/`.edges` files it writes.
- `source="load"` reads existing `.gpickle` or `.nodes`+`.edges` files via `topon.topology.loader.load_graph`.

Lattices: `SC`, `BCC`, `FCC` and `Diamond` (fixed neighbour patterns at
the default range), plus `MIX`, which overlays SC/BCC/FCC basis sites in
one cubic cell at configurable fractions. **The candidate-edge set is a
lattice plus a range.** `topology.generator.neighbour_cutoff`, in
cell units, is a parameter of every type. At the default 1.0 the pure
lattices keep their canonical nearest-neighbour pattern; at any other
value the edges are every pair of sites within the range under the
minimum image, the search `MIX` has always used, so crosslinkers several
site spacings apart become candidate partners for the sculptor. That is
what makes a sculpted graph match a reaction-generated network (three to
four SC shells for DP-20 strands, six to eight for DP-100). The graph
records the cutoff and the shell distances it produced
(`G.graph["neighbour_cutoff"]`, `G.graph["shell_distances"]`); the shell
table and the key resolver live in `topology/shells.py`. `periodicity`
opens individual axes, giving a free surface on that face; an open axis
is never bonded across, at any range. Both generators support all five
lattices, per-axis boundaries and the range, and build identical lattices
for every combination.

**Two searches, two failure modes.** The lattice is a superset of
candidate edges; a sculptor picks the network out of it, and
`topology.generator.search` says which one.
`generator_python.run_single_trial` is the **strict** sculptor. It removes
candidate edges one at a time, samples `max_trials` trials and records
every removal in `G.graph["move_history"]`, which is what a sculpting
animation replays.
`topology/degree_matching.py` is the **exact** search. It assigns the
requested degree to every site and completes the degree sequence by
augmenting paths (a degree-constrained subgraph), so it does not sample
trials and has no move history. Left unset, `search` resolves to `exact`
when the degree distribution pins every degree from 0 to
`max_functionality` and to `strict` otherwise. A fully pinned request is
the case the strict sculptor cannot reach. A near-complete tetrafunctional
target with dangling-end sites fails on every scaffold in either generator,
while the exact search lands it in under two seconds. They fail in opposite
places. The exact search needs a scaffold with spare candidate edges, so
a request that pins most of its sites to the scaffold's own coordination
(Diamond at `max_functionality: 4`, where 73 % of the reference P(f) sits
at the ceiling) is its one refusal, and it says so rather than retrying. Both return the same graph, with every
site a node, vacancies kept at degree 0, the builder's `box`,
`periodicity` and shell attributes intact, edges in the scaffold's own
order. Both searches exist in both generators. The C searcher runs the
exact search behind `--search=exact` (a port with the same steps and
constants), and the pipeline sends an exact request there when `exe_path`
is set, except one that forces double edges or reserves defect capacity,
which only the Python search takes.

**Two generators, two jobs.** The C source in
[`topon/topology/csrc/`](../topon/topology/csrc/) is the standalone
searcher. It runs on its own, without Python, and is the tool for long
exhaustive searches. The pure-Python `generator_python.py` is the pipeline
default and exists for quick in-process generation of likely networks with
no compiler. They are independent programs, not a library and a wrapper;
nothing in `csrc/` is called from Python and it should not grow a Python
binding. Only the shared surface (lattice construction, the
`.nodes`/`.edges` format) has to stay in step. They agree on
distributions, not individual draws, since the C one draws from its own
stream (seeded from the clock unless `TOPON_SEED` is set).

The split is earned. Measured on SC at `max_func=4` as time to first
success, at 6³ Python takes 0.01 s against the C's 0.04 s (process
startup dominates), at 12³ Python takes 1.5 s against 0.11 s, and at 24³
(13824 nodes) the C finishes in 6.7 s where Python would run for hours.

Produces: `self.graph` (annotated `MultiGraph`) and `self.dims` (box size as `np.ndarray`), plus the stage's section of the run manifest.

**The run manifest.** `Pipeline` writes `manifest.json` into the run
directory, one section per stage, through `topon/core/manifest.py`. Stage
1's section records the lattice, the search, the degree counts requested
against the ones achieved, the giant-component fraction, the seed and the
timing; `topon inspect` renders it next to the stage-output summary. It is
advisory. No stage reads it back to make a decision, so a missing or
half-written manifest changes nothing about a run.

**The periodic cell is recorded, not inferred.** Generators write the exact
repeat distance into `G.graph["box"]`, and `.nodes` files carry it in a
`# BOX Lx Ly Lz` header. `infer_dims_from_graph` returns that value when it
is present and only falls back to estimating the cell from the coordinate
extent (`max - min + 1`) for graphs written before this existed. The estimate
is exact for SC, whose sites are integer-spaced, but overshoots any lattice
with fractional basis sites. BCC and FCC body/face sites sit at +0.5 and
Diamond sites at quarter-cell offsets, so a 4x4x4 BCC or FCC reported 4.5.
Because `self.dims` is the box every minimum-image calculation uses, that
overshoot violated the `bond < box/2` invariant in Design Principle 3 and
sent roughly a third of BCC edges (a quarter of FCC) to the wrong periodic
replica, where they were built at twice their true bond length. Any new
lattice with non-integer sites must record its cell for the same reason.

### Stage 2. Analysis
**Module:** `topon/assignment/manager.py:analyze` &nbsp;**Code:** `Pipeline._run_analysis_stage`

Computes graph statistics (degree distribution, connectivity, defect/entanglement *capacity* of the topology) before any annotations are written. The analysis result is held in `self.analysis_report`. Read-only; does not modify the graph.

(Distinct from the `topon/analysis/` package, which exposes `analyze_graph()` for the standalone `topon analyze` CLI sub-command. `Pipeline` does not import from `topon.analysis`.)

### Stage 3. Assignment
**Module:** `topon/assignment/` &nbsp;**Code:** `Pipeline._run_assignment_stage`

Annotates the graph in place. `AssignmentManager` (`assignment/manager.py`) orchestrates several module-level assigners:

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

After this stage the graph carries every annotation downstream stages need.

### Stage 4. Chemistry
**Module:** `topon/chemistry/` &nbsp;**Code:** `Pipeline._run_chemistry_stage`

`ChemistryBuilder` (in `chemistry/builder.py`, one unified entry point, *not* split into CG/atomistic classes) builds an RDKit `Mol` with 3D coordinates derived from the graph. Supporting modules:

| Module | Role |
|---|---|
| `chemistry/builder.py` | The `ChemistryBuilder` class; bulk of stage-4 logic |
| `chemistry/sequences.py` | Shared monomer sequence helpers used by the builder |
| `chemistry/embed.py` | ETKDGv3 per-chain conformer embedding |
| `chemistry/{dreiding,kg}/` | **Stub sub-namespaces** reserved for force-field-specific chemistry (`kg/` has only a docstring; `dreiding/` is a 1-line stub). The CG vs atomistic switch happens inside `builder.py` via `config.chemistry.model_type`. |

This stage also writes the first set of LAMMPS-relevant outputs to `<output_dir>/02_Chemistry/`:

- `system.data`: atom positions and connectivity (via `CGWriter` or `DreidingWriter` in `topon/writers/`)
- `system.in.settings`: force-field coefficients, written by `DreidingWriter` on the atomistic route and left as a stub on the coarse-grained one
- `system.groups`: `nodes` and `beads` LAMMPS group definitions
- `system_nodes.displace`, `system_backbone.displace`, `system_grafts.displace` (plus `system_pendant.displace` and `system_hydrogens.displace` on the atomistic route): displacement files for stage 5

### Stage 5. Conformation
**Module:** `topon/conformation/` &nbsp;**Code:** `Pipeline._run_conformation_stage`

Stage 5 has two entry points, for the two kinds of build.

`ConformationManager` (`conformation/manager.py`) is the pipeline's path for a system that already has a data file. It reads the chemistry-stage output, applies the displacement files, adds a small uniform noise to break degeneracy, and resolves overlaps iteratively.

`conformation/placement/chains.py::place` is the path for a bead-spring build drawn from the graph itself. It is **not** on the `Pipeline` path. `Pipeline._run_conformation_stage` constructs only `ConformationManager`, and `place` is reached by direct API use (`topon.conformation.place`). It takes a graph, a DP and one of three shapes, sizes the build box from the bead count and the build density (or, equivalently, from a coil ratio), draws every strand and reports the gate every strand had to pass. It runs no dynamics and writes no file. It returns coordinates and the measurements that say whether they are fit to hand on. `conformation/packing/` remains a 1-line stub.

The gate is three readings per strand (every bond at or below the design length, every bond at or above `min_bond`, no bead within `min_sep` of a non-adjacent bead of its own chain). It exists because a path can be the right length, land on both junctions and still be unusable. A six-wave meander at DP 20 resamples to 0.17-sigma bonds with beads jammed between their own second neighbours, and the push-off then separates them *through* the bond between, which is the threaded bond that lets strands cross later.

Three further modules sit beside these and are **not** on the default pipeline path. They are used by the entanglement workflows and are opt-in:

- `conformation/entanglement/` builds chain paths that wind around each other a prescribed number of times. `waypoints.py` draws a chain through points the caller chooses (`Site(at, turns)`) and is the current approach; `braid.py` and `allocation.py` are an earlier search-based construction that picks its own positions, kept because its budgeting and obstruction checks have no equivalent yet in the waypoint path. `designed.py` puts a braid on a chain that is *already placed* and refuses a request whose windings do not fit in the strands' contour, naming the minimum DP. `control.py` closes a loop on the measured Z1+ of the relaxed system; it is exported as `topon.conformation.entanglement.controller` and takes a caller-supplied `runner` so the stage stays free of LAMMPS.
- `conformation/junction_shell.py` spreads the chains leaving a crosslink so their first beads do not overlap, with the shell radius growing with functionality.
- `conformation/paths.py` is plain geometry (two junctions and a bead count in, a path out). `bridging_walk` is melt-like and random; `route_through` is deterministic, for when a prescribed topology has to be repeatable. Both keep every bond exact and land on both junctions. `Clearance` lets any of them be drawn around the beads already in the box, which is what stops the following minimisation from resolving an overlap by pushing two chains through each other. Two routines are for the strands with no second endpoint to interpolate to, and the pipeline's chemistry stage does call these two. `closed_meander` draws a primary loop as a regular polygon through its junction, and `free_walk` draws a sol chain.

Conformation defaults (`Pipeline._DEFAULT_CONFORMATION`):

```python
{"overlap_cutoff": 0.01, "overlap_max_iters": 10, "noise_magnitude": 1e-4}
```

Output: `<output_dir>/03_Conformation/system_relaxed.data`.

**The simulation box comes from stage 1, not from the coordinates.** Callers
pass `lattice_box=dims` into `apply_displacements`, giving a box of
`dims * scale`; only when it is omitted does the manager fall back to
estimating `(max node coord + 1) * scale` from the `.displace` files. This
has to match the cell stage 4 used to route chains across the periodic
boundary. When the two disagree, a chain that wraps under one period lands
in a box of another and its closing bond is left stretched across the
system. The two estimates happen to coincide for SC, which is why the
fallback survived so long, and passing the box also makes the written box
exactly `volume^(1/3)`, so the target density is hit on every lattice.

### Stage 6. Output
**Module:** `topon/writers/` (`LammpsInputGenerator`) &nbsp;**Code:** `Pipeline._run_output_stage`

Writes the LAMMPS input scripts that drive the simulation. What it writes
depends on `simulation.protocol`:

- **`pushoff`** (the coarse-grained default) has five stages (capped-displacement push-off under FENE + WCA, uncapped push-off, equilibration at the build density, affine compression, quench) and no minimiser at any stage.
- **`hardcore_min`** and **`soft_push`** are the two older three-stage decks, minimiser and all, kept so earlier runs can be reproduced. The atomistic route always uses `soft_push`'s shape, because its bonded terms are stiff and a capped push-off would take far longer than a minimiser to resolve the same overlaps.

Output: `<output_dir>/04_Simulation/*.in`. Ask
`LammpsInputGenerator.stages(model_type)` which scripts were written and what
each one leaves behind, rather than naming them, because the two protocol
families differ in both.

The simulation itself is *not* run by `Pipeline.run()`. That is the job of `topon/simulation/` (`SimulationRunner` for a plain sequence, or `protocols.StagedRun` to run the stages and apply the acceptance gates as it goes).

---

## 3. Module map

Concise tour of every directory under `topon/`. Every module above the dashed line is part of the main pipeline; sub-systems below are independent.

| Directory | Files / Subdirs | Role |
|---|---|---|
| `topology/` | 7 files (`shells.py` holds the neighbour-shell table and cutoff resolver; `degree_matching.py` the exact degree-sequence search) + `csrc/` (C generator) + `network/` (a thin loader wrapper), `sequence/`, `simple/` (stubs) | Graph generation and loading. Stage 1. |
| `analysis/` | 3 files (`report.py` is `topon analyze`, `run_summary.py` is `topon inspect`) | Read-only graph statistics for the `topon analyze` CLI, and the post-run summary. **Not used by `Pipeline` stage 2**, which calls `AssignmentManager.analyze()`. |
| `assignment/` | 8 files | Graph annotation: node/edge types, DP, defects, entanglements, copolymers. Stage 3. |
| `chemistry/` | 4 files (`builder.py` is most of stage 4); `dreiding/`/`kg/` are stubs | RDKit Mol construction with 3D coords. Stage 4. |
| `conformation/` | `manager.py` (data-file route) and `placement/chains.py` (bead-spring route, `place`) are stage 5; `entanglement/` (waypoint and braid construction, designed pairs, the Z controller) and `junction_shell.py` are opt-in and off the default path; `packing/` is a stub | Chain placement, overlap resolution, and designed entanglement geometry. Stage 5. |
| `writers/` | 8 files (`lammps_endlinked.py` writes the `fix bond/create` convention straight from a placement) | LAMMPS data and input-script writers. Stage 4 + Stage 6. |
| `forcefield/` | 4 files | DREIDING parameter parser; Kremer-Grest parameters. Read by chemistry/writers. |
| `config/` | 4 files | Pydantic `ToponConfig` schema; `load_config()` / `load_config_full()`. |
| `diagnostics/` | 2 files (`rules.py` is the rule registry) | The semantic checks behind `topon doctor`. |
| `core/` | 3 files (`manifest.py` is the run manifest read/written across stages) | Shared types, protocols and run-level artifacts. |
| `utils/` | 4 files | Shared helpers (e.g. `write_lammps_displacement_file`). |
| `pipeline.py` | - | The `Pipeline` orchestrator class. |
| `cli.py`, `shell.py`, `__main__.py` | - | CLI dispatch and the interactive `topon>` shell. |
| `workflows/` | 4 files | High-level workflow helpers (`cg_network`, `atomistic_network`, `reactive_crosslink`). |
| - | - | - |
| `simbox/` | 8 files | Independent molecule packer + crosslink-template emitter. DREIDING-only. |
| `singlechain/` | 3 files | Single-chain solubility utility. |
| `simulation/` | `runner.py` + `protocols/` (`gates.py`, `staged.py`) | LAMMPS subprocess runner, and the acceptance gates that read a relaxation back: bond histogram per checkpoint, Z1+ held where it must hold, temperature from the velocities. |

---

## 4. Design principles

These are the rules every change is expected to follow.

1. **Graph-first separation.** Topology is decided as a NetworkX graph before chemistry is built. The same graph can produce a CG system or an atomistic system by switching the chemistry builder. *Topology never sees atom types, coordinates, or force-field details.*

2. **Six-stage pipeline, one-way data flow.** Each stage has a single responsibility. Downstream stages do not reach back into upstream state. The stage descriptions in §2 and the module-boundary table below are authoritative.

3. **LAMMPS data files are wrap-only.** The core topon and simbox writers (`topon/writers/`, `topon/simbox/writer.py`) emit 7-column Atoms rows (no `ix iy iz`) and rely on LAMMPS's neighbor / ghost-atom system to handle PBC via min-image, which is correct as long as every bond is shorter than `box/2` (true for KG / DREIDING networks where chains rarely wrap differently from each other). The one exception is `write_endlinked` (`topon/writers/lammps_endlinked.py`), which writes image flags so the coordinates it was given can be reconstructed.

4. **Configuration via Pydantic, no globals in stage code.** Every stage module reads from a `ToponConfig` (or its raw-dict supplements). No `os.environ` lookups, no module-level constants that change behaviour, no hard-coded paths inside `topon/`. Config is loaded once, by the CLI or by whoever builds the `Pipeline`, and handed down.

5. **Module boundaries:**

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

The package uses **Pydantic** (`topon/config/`) to validate JSON configuration. Some sections that have not yet been migrated to schema (`simulation`, `execution`, `experimental`) are passed through as a raw dict alongside the validated config (see `Pipeline.__init__`). `conformation` is validated like any other section, and `load_config_full` still hands a copy of it back in the raw dict, because `Pipeline` and both workflow modules have always read it from there.

Top-level `ToponConfig` sections, in order of pipeline use:

| Section | Used by | Notes |
|---|---|---|
| `study` | all stages | `study.name`, `study.output_dir` |
| `topology` | Stage 1 | `source` (`"generate"` / `"load"`), `lattice_size`, `degree_distribution`, etc. |
| `assignment` | Stage 3 | sub-objects for entanglements, defects, grafts, copolymer sequences |
| `chemistry` | Stage 4 | `model_type` (`"coarse_grained"` / `"atomistic"`), `target_density` |
| `conformation` | Stage 5 | `overlap_cutoff`, `overlap_max_iters`, `noise_magnitude` for the data-file route, which is the one `Pipeline` runs. The bead-spring keys (`placement`, `coil_ratio` / `build_density`, `junction_shell_spacing`, `entanglement`) are validated here but read by `topon.conformation.place` and the controller, which `Pipeline` does not call, so `topon generate` validates them and ignores them. Also passed through raw |
| `simulation` (raw) | Stage 4 (angles) and Stage 6 | relaxation protocol, LAMMPS pair_style, angle handling |
| `execution` (raw) | the `topon.workflows` runners, not `Pipeline` | LAMMPS subprocess settings (`auto_run`, `executable`, `n_procs`) |
| `experimental` (raw) | Stage 6 | feature-flagged extras |

**Unknown keys.** `GeneratorConfig`, the per-class blocks under
`assignment.defects` and the whole `conformation` section refuse keys
they do not define. The other nested sections ignore them, because a
config may carry other tools' parameters (under `topology`, for example),
and `topon doctor`'s `unknown_config_keys` warning covers them instead.

For the full key-by-key reference and example configs, see [USAGE.md](USAGE.md).

---

## 6. CLI surface

The `topon` CLI (`topon/cli.py`) dispatches to:

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

Full flag-by-flag reference: [USAGE.md](USAGE.md).

---

## 7. Where to go next

| Question | Doc |
|---|---|
| How do I run topon end-to-end? | [USAGE.md](USAGE.md) |
| Which CLI flag does X? | [USAGE.md](USAGE.md) |
| What's the JSON config schema? | [USAGE.md](USAGE.md) (config-reference appendix) |
| Which demo shows X? | [`demos/README.md`](../demos/README.md) |
| What's changed across versions? | [CHANGELOG.md](../CHANGELOG.md) |
