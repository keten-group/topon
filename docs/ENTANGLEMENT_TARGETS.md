# Reaching a target entanglement

This note is for anyone who asks topon for a number of entanglements per strand and wants the relaxed network to
carry it. It explains what the number means, why the build sets it, which range can be reached, how to run the loop
that meets a target, and the usual reasons a target is missed. It applies to topon 0.4.5 and later, and the counts
by partner (section 1) to 0.4.7 and later. The reference
for every key is `docs/USAGE.md` (`conformation`, `conformation.entanglement` and `simulation.protocol`).

## 1. The number being targeted

Entanglement is measured with Z1+ (Kröger's primitive-path analysis, run with `-SP+`), as the number of kinks per
strand, Z. Two averages are in use, and a target has to say which one it means.

| Average | Used by |
|---|---|
| Z per bridge (strands with a junction at both ends) | `conformation.entanglement.target_Z` through the controller, the shipped calibration, `topon fit` |
| Z per chain over the whole system (bridges, loops, dangling and free chains) | `topon analyze --z1` and `topon inspect` (which also give it per class, and split by partner under the total), the entanglement-range figure of the paper |

A target also has to name its count, which says which kinks are counted. Z1+ is given each strand on its own, with
its own copy of every junction it ends on, so a junction with four strands is four points. The exporter moves each
copy by a Gaussian of 1e-3 sigma, because Z1+ stops when two chains share a coordinate, and Z1+ holds the copies
fixed. A strand's shortest path can then bend round the end copy of a sibling (a strand on the same junction). In
the network that point is the strand's own junction, not an obstacle. The kink is an artifact of the export, and
how many such kinks there are is set by the jitter, not by the configuration. Since 0.4.7 every kink is classed by
the strand it is pinned on (its partner).

| Count | Kinks counted |
|---|---|
| `total` | every kink Z1+ reports. The default, and every number before 0.4.7 |
| `junction_end` | on a sibling, within 1 sigma of the junction the two share. The export artifact |
| `sibling` | on a sibling, farther out (two arms of one junction twisted round each other) |
| `nonsibling` | on a strand that shares no junction with the kinked one |

On a periodic 5x5x5 SC graph with 20-bead strands drawn as random walks at bead density 0.85 (375 bridges, as
built), Z per bridge as `total` reads 1.66, 1.74, 2.00 and 2.11 at an export jitter of 1e-4, 1e-3, 1e-2 and 0.1, as
`junction_end` 0.65, 0.73, 1.03 and 1.11, and as `nonsibling` 0.448, 0.435, 0.469 and 0.435.

An entanglement level, in this guide, is the `nonsibling` count per bridge at a stated state and export jitter,
reported with the `total` and `sibling` counts beside it. The non-sibling kinks are the part a relaxation in which
nothing crosses keeps. Sibling kinks are two arms of one junction held round each other, which partly come undone
when the network is expanded back to its build box, and junction-end kinks are the export's.

`total` stays topon's default count, so a `target_Z` that names no count, the shipped calibration, `topon fit` and
every number before 0.4.7 read as before, at the same jitter, and compare with each other like for like. To set or
read a level, name the count (`count: "nonsibling"`, below). Only the junction-end count follows the jitter, so a
reading from another export (another jitter, another tool) compares with topon's on the other two. On a POSS network the strands that end on one cage
are siblings, though each ends on its own cage atom. On an atomistic network the 1 sigma cut is 1 A, since it is in
the file's length unit, and it is not validated there. The exported points of an atomistic strand are about 3 A
apart, so the cut is a third of the first segment where on a bead-spring strand it is about one bond, and the
junction-end and sibling classes of the two kinds are not cut at the same place.

`topon analyze --z1` and `topon inspect` print the three on the line after the total, and again under each class.
From a record, `bridge_z(record, count="nonsibling")` gives it (`topon.analysis.z1plus`, with `"junction_end"`,
`"sibling"` and `"total"` beside it), `measure_seeds(system, count="nonsibling")` averages it over the export's
seeds, and a report of `topon analyze --z1 --json` carries it under `z1.by_class.<class>.by_partner`. A record
written before 0.4.7 has no split, and asking it for a partner count is an error rather than a total.

A target names its count with `conformation.entanglement.count`, `total` unless it says otherwise.

```json
"conformation": {"entanglement": {"target_Z": 2.0, "count": "nonsibling"}}
```

The bead-spring controller then closes on that count. It hands the count to the runner in each `RoundPlan`, and the
runner returns `z_final` in it (section 5.4). On the atomistic route the coil-radius search, `relax_to_target` and its
deck read it, and `topon generate --verify --relaxed` checks the target in it. The relaxation gates stay on the total,
since their tolerances were set against it. The shipped calibration and the floors are totals too. A target in
another count is seeded from the table with a warning, so its first round reads below the target and the loop has
further to go, and no floor is checked against it. Built with no MD, a DP-30 PDMS network on a 3x3x3 cell aimed at
2 non-sibling kinks per bridge met it at coil radius 12.78 A, reading 1.958 (a total of 2.65), and placed 60-bead
walks of a 4x4x4 cell closed on 1.94 non-sibling in three rounds of the bead-spring loop (USAGE section 3.6, "A target
in a count").

Z1+ is not a topological count. It reads the shortest paths between the junctions where they sit now, so it moves by
10 to 20 % between stages even when no strand crosses another (on the DP-100 reference, 1.16 after the push-off,
1.00 after equilibration at constant volume and 1.11 after compression, with every bond under 1.2 sigma). A target is
therefore a target at one state. The default is the final state (compressed to `simulation.rho_final`, equilibrated
and quenched to T 0.4), and two readings can be compared only at the same density and temperature.

## 2. What sets the count

Three measurements made while validating the paper decide how control works.

1. **The build sets Z.** On one graph at one DP and one final density, strands built at a higher density (or with
   more wander) end with more entanglements. A DP-30 network with the published P(f), built at bead density 0.20,
   0.40 and 0.60 and compressed to 0.85, ended at Z 0.59, 0.99 and 1.85 as totals (0.18, 0.49 and 1.20 in the
   non-sibling count), and the same network built at 0.85 at a total of 3.14.
2. **Compression keeps it.** Built dilute and compressed affinely, a network keeps its Z within 5 to 6 %
   (retention 1.05 to 1.06 over a threefold range of build density, for builds that start entangled). For the three
   builds of point 1 the ratio of final to build Z is 1.12, 1.06 and 1.05 in the total and 1.25, 1.09 and 1.08 in the
   non-sibling count. A build that starts unentangled gains the floor that packing alone gives (section 3).
   Excluded volume forbids a strand to pass through another, so MD can neither add entanglements to a relaxed
   network nor remove them.
3. **Any crossing breaks control.** A stage that lets strands pass through each other changes Z by an amount
   nobody chose. `pair_style soft` does this by design, and a minimizer does it by stretching a bond until a bead
   slips through. On the published dilute ensemble a soft stage raised Z from nearly zero to 1.6 to 1.9, and the
   hard potential that followed brought it back below 0.1. On the DP-20 reference build the minimizer deck moved Z per
   bridge from 0.19 to 0.24.

So the knob is the build (chain shape and build density), and the reading is Z1+ after a relaxation in which no
strand crosses another. `simulation.protocol: "pushoff"` (FENE and WCA from the first step, `nve/limit`, no
minimizer, five stages) is that relaxation and is the default. `"hardcore_min"` and `"soft_push"` are kept to
reproduce earlier runs and should not be used for a target.

## 3. The range that can be reached

**Ceiling.** Strands can only be packed so densely, so each DP has a largest Z. On six networks from the paper's
ensemble (SC 6x6x6, spanning its degree distributions), strands built as random walks, relaxed with WCA from the
first step and measured over the whole system at the build density, a request came back as follows (mean and
standard deviation over the six networks). These are totals, whole-system averages measured before the kinks were
split by partner, and the table has not been measured in the non-sibling count.

| requested Z | DP 20 | DP 40 | DP 80 | DP 120 |
|---|---|---|---|---|
| 0 | 0.008 | 0.036 | 0.036 | 0.013 |
| 1 | 0.98 ± 0.05 | 0.89 ± 0.09 | 1.01 ± 0.13 | 1.20 ± 0.18 |
| 2 | 1.93 ± 0.09 | 1.91 ± 0.18 | 2.00 ± 0.18 | 1.93 ± 0.20 |
| 4 | 2.10 (ceiling) | 4.00 (ceiling) | 3.74 ± 0.26 | 4.06 ± 0.17 |
| 8 | 2.10 (ceiling) | 4.00 (ceiling) | 7.53 (ceiling) | 7.52 ± 0.24 |

The ceiling is close to DP/11 (2.1, 4.0, 7.5 and about 10.8). A request above it returns the ceiling. Requests in
the middle of the range land within a few percent. Requests near the floor at long DP are the least reliable
(1 at DP 120 came back as 1.20), because there Z hardly changes with the build. The data of this table are in
`demos/npjcompmat/data/derived/entanglement/`.

**Floor.** Straight strands give Z = 0, but only when the box is sized from the strand contour, so that every
straight strand fits at its bond length (`(DP + 1) * bond`). Beads spaced along a chord that is too short sit inside
each other's cores, the strands explode when WCA switches on, and they pass through each other on the way out (this
once produced a false floor that rose with DP). Sized from the contour, straight strands read 0.000 as built and
0.01 to 0.04 relaxed at every DP, at their build density. Packed to melt density the total reads more, with nothing
crossing, because Z1+ counts the kinks that packed neighbors force on the shortest paths. Most of those are
junction-end and sibling kinks, which an expand-back to the build box removes, so in the non-sibling count a
topologically unentangled network reads about 0 at the final state, and level 0 can be reached there. The expand-back
test stays the check for a crossing (a crossed network stays high when expanded).

The shaped routes of `topon.conformation.place` have a floor above zero. At DP 20
on the 4-shell reference graph both `walk` and `meander` end near 0.236 per bridge (the `fix bond/create` reference
they were measured against has 0.178, which neither reaches). `topon doctor` warns when a target is below the floor
measured for its route and DP (`entanglement_target_below_floor`).


## 4. Which mechanism to use

| Goal | Use |
|---|---|
| A mean Z on a bead-spring network | `conformation.entanglement.target_Z` and the controller (section 5) |
| Windings between named strand pairs | `conformation.entanglement.pairs`, or `assignment.entanglements` for pairs chosen by the assignment stage |
| A Z on an atomistic network | `atomistic_placement: "coil"` with `target_Z`, then `relax_to_target` (section 6) |

Designed pairs and Z1+ targets answer different questions. A designed pair carries the windings it was asked for,
checked by a linking number, but the network's Z1+ also counts every entanglement the placement makes between all
other strands. `assignment.entanglements` with `"target": 5, "target_type": "count"` asks for five designed pairs,
not for Z = 5. For a count, use `target_Z`.

## 5. The bead-spring recipe

### 5.1 Before starting

- Install Z1+. It is not redistributable and does not ship with topon. Point `analysis.z1plus.executable` (or
  `--z1-exe`) at the binary, which runs under WSL on Windows, and check it with `topon analyze <data file> --z1` on
  any data file in the end-linked convention.
- Use LAMMPS with OpenMP threads. On Windows `mpiexec -np 8 lmp` starts eight serial copies of the same run, not a
  parallel run.
- Budget the time. Every controller round is a full relaxation, about 35 minutes at 8 threads for a 95,000-bead
  build, and a target usually takes two to four rounds.

### 5.2 The graph

Generate the graph with the exact degree distribution (`search: "exact"`, counts that add up to the number of
lattice sites), and give every edge its DP (or pass `dp=` to the controller). The DP and the neighbor range matter
more than anything else for the range of Z (section 3).

### 5.3 The conformation block

```json
"conformation": {
  "placement": "walk",
  "build_density": 0.095,
  "junction_jitter": 0.15,
  "loop_shape": "compact",
  "entanglement": {
    "target_Z": 0.29,
    "close_on": "final",
    "controller": {"max_rounds": 4, "tolerance": 0.05}
  }
},
"simulation": {"protocol": "pushoff", "rho_final": 0.3075}
```

- **Route.** `walk` (random walks, knob `build_density`) for targets at or above the walk's floor, `meander`
  (knob `coil_ratio`) below it. At DP 20 the two floors coincide.
- **One knob.** `coil_ratio` and `build_density` are the same setting in two units (the box scales as
  `rho^(-1/3)` and the contour does not), so give one of them, never both. With neither, the controller starts from
  the shipped calibration, and `topon doctor` refuses a target with no knob and no calibration row to start from
  (`conformation_build_knob`).
- **Crossing chords.** On lattices whose chords cross at exact points (several neighbor shells, mixed lattices),
  strands drawn through one point jam there in the push-off and end with a higher Z. Set `junction_jitter: 0.15`,
  and for the meander also `settle_clearance: 1.0`. The DP-20 calibration rows were measured with these settings.
- **Primary loops.** Set `loop_shape: "compact"` whenever the graph has primary loops. The default `"ring"` draws
  each loop as an open polygon, which raised Z per bridge by about 12 % on the DP-100 reference (about 6 % with
  `"compact"`). A graph without primary loops builds the same either way.
- **Dense walks.** A plain walk drawn dense lands beads on other strands and on itself, and the hard push-off, which
  lets nothing cross, can pull those contacts into a few bonds over 1.2 sigma. Set `walk_clearance: 0.85` there. The
  clear walk redraws every strand so that each step keeps clear of everything drawn so far and of the strand itself
  (*Dense walks* in `docs/USAGE.md`). The calibration rows were measured without it, so a controller seeded from them
  starts from a build of another kind.
- **Count.** A level is set in the non-sibling count (`entanglement.count: "nonsibling"`, section 1). The shipped
  calibration rows (DP 20 and DP 100, final density near 0.30) are totals, so the controller's first move in another
  count is off until the rows are measured again in it.
- **Tolerance.** The default is 0.15 (15 %). A tighter tolerance costs rounds.
- **Spelling.** The `conformation` subtree refuses unknown keys, so `target_z` (lower-case z) is an error rather
  than a silently empty target.

Run `topon doctor <config>` before any LAMMPS run.

### 5.4 Build and relax through the controller

`topon generate` does not meet a bead-spring Z target. It draws every strand on its chord whatever `placement` says,
and the push-off keeps the low Z such a build starts with, so a relaxed `topon generate` build reads at that low Z
whatever the target asks for. The target is met by the controller, which repeats build, relax, measure and move the
knob.

```python
from topon.conformation.entanglement import controller

manifest = controller(graph, config.conformation, runner, dp=20, bounds=(0.02, 0.85))
```

`runner` is a function the caller supplies. It receives a `RoundPlan` (`round`, `placement`, `actuator`, `value`,
`coil_ratio` or `build_density`, `dp`, `seed`, and `count`, the config's `entanglement.count`) and returns a
`RoundResult` or a dict with at least `z_final` (and `z_build` when the loop closes on the build state), both in
`plan.count`. One round looks like this.

```python
import numpy as np
from topon.conformation import place
from topon.analysis.z1plus import measure_checkpoint, bridge_z

def runner(plan):
    pl = place(graph, plan.dp, plan.placement,
               coil_ratio=plan.coil_ratio, build_density=plan.build_density,
               rng=np.random.default_rng(plan.seed),
               junction_jitter=0.15, loop_shape="compact")
    # Write pl with topon.writers.write_endlinked, write the five push-off stages with
    # topon.writers.LammpsInputGenerator, and run and gate them with
    # topon.simulation.protocols.StagedRun (it stops at any bond over 1.2 sigma from stage 2 on).
    # run_dir is the folder the five stages ran in.
    build, _ = measure_checkpoint(run_dir / "stage3_build_equil.data")
    final, _ = measure_checkpoint(run_dir / "stage5_final_quench.data")
    return {"z_final": bridge_z(final, plan.count), "z_build": bridge_z(build, plan.count)}
```

The development sources carry a complete runner of this kind, which writes, runs, gates and measures every
stage. It is not part of the release.

`bounds` clamps the knob (the walk's build density in
the example above, since the meander's knob is the coil ratio). Give it, since the first move is often an
extrapolation off a short table and an unclamped power law can ask for a build density of 0.002. The returned manifest lists every round (knob, Z at every
stage, largest bond at every stage), whether the loop converged, and its warnings.

### 5.5 Check the result

- No bond over 1.2 sigma from stage 2 on. A run that failed this gate has threaded strands, and its Z means
  nothing.
- Report Z per bridge at stage 5, at `rho_final` and T 0.4, in the count the target names, with the total and the
  sibling count beside it and the export jitter stated, and repeat with a second velocity seed. The seed spread at
  DP 20 is about 0.004.
- Anything done to the network afterwards (further compression, a deformation run) needs a new reading.

## 6. The atomistic route

The bead-spring calibration does not apply here. With `atomistic_placement: "coil"` each strand winds round its chord
at a radius, and `target_Z` makes the pipeline search that radius against Z1+ per bridge on the drawn network,
settle it, and read it again. The hard-backbone deck keeps every entanglement of such a build, but MD unwinds part of
a tight helix (coils kept 74 % of Z at a target of 1 and 87 % at 2 after NVT, the meander 94 to 99 %), so a target on
the relaxed network needs `topon.simulation.protocols.z_target.relax_to_target`, which builds again at a corrected
radius until the relaxed reading is within 5 % (or `controller.tolerance`). With `entanglement.count` set, the search,
the settled reading and the relaxed readings are all in that count (section 1), and the deck's gates stay on the total.

## 7. Why a target is missed

| What you see | Cause | What to do |
|---|---|---|
| A dense walk comes out of the hard push-off with a bond over 1.2 sigma | beads drawn on other strands or on their own strand, which a push-off that lets nothing cross pulls into a stretched bond | `walk_clearance: 0.85` (5.3) |
| A relaxed `topon generate` build reads far below the target | the pipeline draws strands on their chords | build with `place` through the controller (5.4) |
| Z rises or jumps between stages, or bonds pass 1.2 sigma | a deck that lets strands cross (`soft_push`, a minimizer), or strands jammed where chords cross | `protocol: "pushoff"`, `junction_jitter: 0.15` (and `settle_clearance: 1.0` for the meander) |
| The target cannot be reached from below | it is under the route's floor (`topon doctor` warns), or under the floor that packing alone gives an unentangled network at the final density | switch to the meander if its floor is lower, or build straight strands in a box sized from the contour, which is the lowest a network can read at that state (check it by expanding back to the build box) |
| More is asked for and the same comes back | the ceiling (about DP/11 at melt density) | longer strands |
| The first round is far off and the loop is slow | calibration exists only for DP 20 (4-shell mixed lattice) and DP 100 (8-shell SC), both at final density near 0.30, and another DP borrows the nearest one | expect more rounds, give `bounds`, or start from a knob already measured on this system |
| The loop stops after one round and asks for a second knob | the rows at this DP fix a level but no slope (e.g., the DP-20 meander between coil 1.402 and 1.51) | measure a second knob by hand, or switch to the walk |
| The controller converged but your number disagrees | a different average (whole system against per bridge) or a different state (build against final, another density or temperature) | compare the same average at the same state |
| Z1+ changes when the export jitter or a coordinate rescale changes, or topon and another tool read the same file differently | the junction-end kinks, whose number follows the export's jitter (section 1) | read the non-sibling count (`bridge_z(record, count="nonsibling")`) |
| Z overshoots on a graph with primary loops | loops drawn as open rings | `loop_shape: "compact"` |
| Two networks with the same settings land 20 % apart | near the entanglement onset (low Z, long DP) Z depends on the topology | close the loop on each network, not on a shared table |
| Z1+ stops with `CRASHED. contact mk`, or strands come out joined through junctions | chains sharing a coordinate, or junctions found by degree instead of by atom type | measure through `topon analyze --z1` or `measure_checkpoint`, which move the shared ends by 1e-3 sigma and read the end-linked atom types |
| An atomistic target is met at the build and missed after relaxation | the radius search reads the drawn network | `relax_to_target` (section 6) |

## 8. How the paper's entanglement range was measured

The table in section 3 was measured before the controller existed, with scripts that are not part of topon. Strands
were drawn with `topon.conformation.paths.bridging_walk`, and the build density for each request was read off a
measured map of Z against build density at each DP, capped at 0.85. The zero request used straight strands in a box
sized from the contour. Every system was relaxed with WCA from the first step through three short stages (the
predecessor of today's push-off) and measured with Z1+ over the whole system at its build density. The same three
facts (section 2) are what the controller and its calibration rest on.

## Checklist

1. Z1+ installed and checked, LAMMPS with OpenMP.
2. Exact graph, DP on the edges, and a target inside the range for that DP (section 3).
3. `placement`, one knob, `junction_jitter` on crossing lattices, `loop_shape: "compact"`, `target_Z` per bridge
   with its count named (`entanglement.count`, `total` unless said otherwise), `protocol: "pushoff"`, `rho_final`.
4. `topon doctor` clean.
5. The controller with a runner, not `topon generate`, and `bounds` given.
6. Gates passed, Z per bridge read at the final state in the target's count with the total and sibling counts
   beside it, a second seed.
