# C topology generator

`generator.c` is the standalone searcher. It runs on its own, without
Python, and is the tool for long exhaustive searches over many trials.

The pure-Python [`generator_python.py`](../generator_python.py) is a
separate program with a different job: quick in-process generation of
likely networks, no compiler needed, and it is the pipeline default.

**These are two independent tools, not a library and a wrapper.** Nothing
here is called from Python and nothing here should grow a Python binding.
The division is deliberate: long searches belong to C because it is
faster at them, everyday generation belongs to Python because it is
immediate. What they share is the lattice construction and the
`.nodes`/`.edges` format, and only that shared surface has to stay in
step.

## Feature status

The two now cover the same ground:

| capability | C | Python |
|---|---|---|
| SC / BCC / FCC lattices | yes | yes |
| Diamond lattice | yes | yes (`generator_python_diamond.py`) |
| `MIX` overlay with fractions + cutoff | yes | yes |
| neighbour cutoff on SC / BCC / FCC / Diamond (candidate-edge range) | yes (ninth argument) | yes |
| per-axis periodicity | yes | yes |
| `# BOX` header recording the periodic cell | yes | yes (reader and writer) |
| per-degree targets `d:N` | yes | yes |
| total edge target `e:N` | yes | yes |
| pruning search (`strict`) | yes (default) | yes |
| exact degree matching (`exact`) | yes (`--search=exact`) | yes (`degree_matching.py`) |
| forced double edges for the exact search (`double_pairs`) | no | yes |
| move history for the sculpting animation | no | yes (strict only) |

Both searches exist in both programs. The exact search is a port of
[`degree_matching.py`](../degree_matching.py) with the same steps and
constants (random target permutation, greedy fill with the dangling-end
rule, 6 rounds of alternating-path augmentation, up to 20 000 target
swaps and vacancy moves, acceptance at a giant component of
`min_giant_fraction`, default 0.99). It draws from a different stream, so
the two agree on distributions rather than on individual networks. The
forced double edges that the defects stage asks for have no command-line
channel here, and the pipeline keeps such a request on Python.

Verified by `tests/workflows/compare_generators.py` (in the development
repository, not in this release), which sweeps both
across 48 configurations (lattices, sizes including non-cubic, open
axes, mixtures, distribution modes, and 8 exact-search cases on SC at one
and three shells, BCC, FCC, a mixture and Diamond) and compares site
counts, mean degree, edge-length shells and the recorded box, and for the
exact search the degree counts of every network on both sides. On
2026-09-25, **43 agree** (17 of them edge for edge) and the other 5 ask
for something neither can supply, which both refuse (two of them were the
exact search's known refusals then, target B on SC at one shell, which
the fallback below now reaches, and Diamond at `max_func` 4 with dangling
ends). With the fallback (2026-09-26) **44 agree** and 4 are refused by
both, target B on SC at one shell now reaching its counts on every run on
both sides. A subset is then built through
the pipeline to a LAMMPS stage-1 minimize, which all six complete:
SC/BCC/FCC pruned to `max_func=4`, a 0.2/0.4/0.4 mixture, a non-cubic
3x4x5, and an `e:200` edge-count target.

The exact search was also compared with the Python one statistically (the
streams differ, so no single graph is expected to match) on SC 6^3 at
one, two and three shells for the three generator-benchmark targets (100
accepted graphs a side, in the paper's generator benchmark): per-attempt
success rates agree, and algebraic
connectivity, maximum betweenness, bridge count and cycle rank agree in
mean, spread and a two-sample KS test (all 32 p-values at or above
0.05).

## Build

```bash
gcc -O2 -o generator.exe generator.c -lm
```

The binary is gitignored. Point topon at it with
`topology.generator.exe_path`, or leave that `null` to use the Python
generator.

## Command line

```
generator.exe <dims> <periodicity> <max_func> <max_trials> <max_saves> "<degree_dist>" <logging> <lattice_type> [neighbour_cutoff] [--search=strict|exact] [--min-giant-fraction=F] [--odd-walks=on|off] [--seed=N] [--output-dir=DIR]
```

| argument | meaning |
|---|---|
| `dims` | `NxxNyxNz`, e.g. `8x6x8` |
| `periodicity` | one digit per axis, `1` periodic and `0` open, e.g. `111` |
| `max_func` | maximum crosslink degree |
| `max_trials` | trials before giving up (for `exact`, attempts) |
| `max_saves` | networks to write |
| `degree_dist` | `"d:N,..."` per-degree counts, or `"e:N"` for a total edge count |
| `logging` | `0` or `1` |
| `lattice_type` | `SC`, `BCC`, `FCC`, `Diamond`, or `MIX:<sc>,<bcc>,<fcc>[,<cutoff>]` |
| `neighbour_cutoff` | optional; the candidate-edge range for `SC`/`BCC`/`FCC`/`Diamond` in cell units (default `1.0`, the canonical lattice). `MIX` carries it inside its own argument; giving both is refused. |
| `--search=` | optional flag, anywhere in the line. `strict` (the default) prunes the lattice edge by edge; `exact` runs the exact degree matching |
| `--min-giant-fraction=` | optional flag for `exact`; the share of active sites the largest component must hold (default `0.99`, the Python default; `1` demands a fully connected active subgraph) |
| `--odd-walks=` | optional flag for `exact`; `on` (the default) searches the augmenting walk again when an attempt ends short on a scaffold with odd cycles, `off` gives the networks of earlier versions (the same switch as `topology.generator.odd_walks`) |
| `--seed=` | optional flag; a non-negative integer that fixes the whole random stream. It wins over the `TOPON_SEED` environment variable, which gives the same stream for the same number. With neither, the seed comes from the clock and the pid and is printed, so any run can be replayed |
| `--output-dir=` | optional flag; where the `.nodes`/`.edges` files go (default `output` in the working directory). Missing directories are created |

The flags are named rather than positional, so every eight- or
nine-argument call keeps its meaning and gets the strict search.

**Seeds and output directories.** Two runs with the same seed, lattice
and request write identical files. The file names carry only the lattice
size and the trial number (`network_N6x6x6_trial0.nodes`), so two runs
writing into one directory overwrite each other's networks, and a script
that collects the newest file can pick up the other run's. Give every
concurrent run its own `--output-dir` (the pipeline does, one run
directory each). The seed is printed on every run, `--seed` or not.

```bash
# the same network twice, in two directories
./generator.exe 8x8x8 111 4 1000 1 "0:20,1:40" 0 SC --seed=7 --output-dir=runs/a
./generator.exe 8x8x8 111 4 1000 1 "0:20,1:40" 0 SC --seed=7 --output-dir=runs/b
```

`--search=exact` needs a count for every degree from 0 to `max_func` (an
`e:N` term, if given, must agree with them). Degree 0 is the leftover
sites, so a request whose counts do not add up to the site count still
runs, with a note. One trial is one attempt, so `max_trials` bounds the
attempts and a large value keeps retrying until an attempt lands or the
run is killed. The one early stop is Python's: when most of the request
sits at the scaffold's own coordination (Diamond at `max_func` 4 with
dangling ends) a shortfall ends the run, since a fresh draw meets the
same forced assignment. A run that finds nothing exits with status 1 and
prints the Python search's failure message (the request, the best
attempt's shortfall and what to change); a run that writes some networks
but fewer than `max_saves` exits 0 with a warning. Requests that cannot
be met at all are refused before any attempt, by the same checks the
Python generator makes (in its words for the exact-search ones): a
degree left unspecified, an `e:N` that contradicts the counts, too many
active sites, an odd degree sum, or a degree above the lattice's
coordination or `max_func`.

Besides the random stream, two fixed orders differ from Python. A
dangling end that the repair step hands a partner takes the first
eligible neighbour in the scaffold's neighbour order, which the two
lattice builders lay down differently, and after a vacancy move the
former partners are drained in edge-list order here and in
set-iteration order there. About 16 000 attempts compared on SC and FCC
showed no effect of either.

**The fallback.** SC, BCC and Diamond at their first shell are
bipartite (every edge joins the two sublattices), so a degree sequence
is realisable only if the targets on the two sublattices sum to the
same number. A random deal almost never balances, and the repair swaps targets
across the sublattices at random. Once 6 attempts in a row have ended
with unfilled degree units, the attempts for that network switch to a
balanced deal (targets swapped across the sublattices until the sums
match) and a residual-driven repair (a target from the region a failed
augmenting search reached, on the stuck site's sublattice, is swapped with
a lower one outside it on the same sublattice, and the swap is kept only
if no edge was lost). Attempts that fill every degree but fail on
connectivity do not count toward the switch, and the random-deal attempts
before it draw exactly what they drew before, so a target the random deal
reaches gives the same network for the same seed. The Python search does
the same (6 random attempts, then 6 fallback ones), and its module
docstring is the specification. The manuscript's hardest target (44 dangling ends
and 54 six-fold sites per 216) on SC at one shell, which the random deal
reached 3 times in 100 at 216 sites and never on larger cells, lands on
the first fallback attempt at 216 to 1 728 sites.

The mix fractions ride inside the `lattice_type` argument rather than
taking a ninth position, so scripts written against the original
eight-argument CLI keep working unchanged. The optional ninth argument is
the neighbour cutoff: at any value other than `1.0` the pure lattice's
sites keep their numbering and the edges are rebuilt as every pair within
the range under the minimum image (open axes are never bonded across).
Below the lattice's nearest-neighbour distance the run is refused.

`MIX`, and every pure lattice at a non-default cutoff, builds its edges by
an all-pairs neighbour search, which is O(N²) in the site count. That is
unnoticeable at the cell counts topon normally uses (a 6x6x6 mixture is a
few thousand sites) and still cheap at 20x20x20: measured 0.7 s for SC at
cutoff 3.01 (8000 sites, 488 000 edges) and 1.4 s for a 0.2/0.4/0.4
mixture, whole run. The pure lattices at the default cutoff are
unaffected, since they enumerate a fixed neighbour pattern instead. (The
Python generator switches to a cell list above 4 000 sites; this file
does not.)

```bash
./generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 SC
./generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 MIX:0.2,0.4,0.4

# Diamond is 4-coordinated already, so max_func=4 needs no pruning
./generator.exe 6x6x6 111 4 10 1 "" 0 Diamond

# Three simple-cubic shells (z = 26) as the candidate-edge set
./generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 SC 1.74

# A slab: periodic in x and y, open in z
./generator.exe 6x6x6 110 4 1000 1 "0:0,1:0" 0 SC

# Exact degree matching, the manuscript's example target on two SC shells
./generator.exe 6x6x6 111 6 100 1 "0:10,1:0,2:26,3:75,4:43,5:53,6:9" 0 SC 1.42 --search=exact
```

From Python, `topon.topology.generator.run_generator` passes
`--search=exact` whenever the config resolves to the exact search by the
Python generator's own rule (`search: "exact"`, or `search` unset with
every degree pinned), and the pipeline sends such a request here when
`exe_path` is set. Two things keep that route in line with the Python
one. A config that leaves `max_trials` at its default (a million, sized
for the strict sculptor) gets the Python budget of 6 random and 6
fallback attempts per network instead, while an explicit `max_trials` is
passed as given. And the
pipeline hands the binary a `TOPON_SEED` drawn from the global NumPy
stream, as the Python exact search draws its own, so `np.random.seed(n)`
pins either route; the seed lands in the run manifest with the requested
and achieved counts. With `topology.generator.seed` set the draw comes
from a stream seeded with it instead (the same number `np.random.seed`
would have given), and the strict search is then seeded as well, where
unpinned it still seeds from the clock. The seed goes through the environment rather than
`--seed` so that a binary built before the flag (the paper's release
binaries) still takes it, and the binary runs in the run's own
`topology/` directory, so it writes `topology/output/`.

## Output

Writes `<output-dir>/network_N<dims>_trial<n>.{nodes,edges}`, with
`output` in the working directory as the default. The `.nodes` file
opens with a `# BOX Lx Ly Lz` header recording the true periodic cell,
which `topon.topology.loader` reads back. That header matters: without it
the loader falls back to estimating the cell from the coordinate extent,
which is exact only for SC and overshoots any lattice with fractional
basis sites.

## What has to stay in step

Only the shared surface: lattice construction (the basis sites, their
numbering and the neighbour search behind the cutoff), and the `.nodes` /
`.edges` format. Changes there land in both this file and
`generator_python.py`. The sculpting search itself is each program's own
business.

`tests/unit/topology/test_c_generator.py` (development repository) compiles this source and checks
site counts, coordinates, the `# BOX` header, and that the C sculpts the
same configurations Python does. It skips when no compiler is on PATH.

The two do not produce identical draws: this one draws from its own
64-bit stream seeded per process, so parity is over distributions, not
individual networks.

## Seeding

Every draw (the sculptor's shuffles, the `MIX` site draws and the exact
search) comes from xoshiro256**, seeded by passing a 64-bit seed through
splitmix64, with unbiased bounded draws. It replaced `rand()`: MinGW's
`rand()` gives 15 bits from one shared 2^32 cycle, so every seed was only
a starting point on the same sequence, and long searches from different
seeds could fall into step and return identical networks. A 15-bit draw
also cannot index past 32 767.

The seed mixes the clock with the pid and is printed on an `INFO:` line.
`time(NULL)` alone advances only once a second, so a script looping this
executable to collect N networks used to get **byte-identical output
from every run that started in the same second** (three back-to-back MIX
runs produced the same file).

Set `TOPON_SEED` (any unsigned 64-bit integer) for a reproducible run:

```bash
TOPON_SEED=42 ./generator.exe 6x6x6 111 4 1000 1 "0:0,1:0" 0 MIX:0.2,0.4,0.4
```

Without it every run differs, which is what you want when collecting a
population of networks. The Python generator gets the same control by
seeding `random` before calling `generate`.

A `TOPON_SEED` used with a build from before the switch (2026-09-25) does
not reproduce that build's network under this one, since the stream it
seeds is a different generator.

### Known divergences

All predate the vendoring and are deliberately left alone:

- **The degree-2 guard.** In the sculpting stages it is gated on
  `is_sc_lattice`, computed as `strcmp(lattice_type, "SC") == 0`. Two
  consequences: the guard is off for BCC and FCC where Python applies it
  unconditionally, and it is also off for `MIX:1,0,0` even though that
  builds the identical simple-cubic lattice. Measured on 5x5x5 pruning to
  `max_func=4`, `SC` averages 221 edges against `MIX:1,0,0`'s 216. Both
  succeed; the networks are just drawn from slightly different
  distributions.
- **Unreachable targets** (closed). Python's fail-fast guard rejects a
  per-degree target above the *lattice's* coordination and above
  `max_func`; this file used to check only `max_func`. It now checks both,
  in Python's order, for either search.
- **No wall-clock limit.** Python's `generate` takes a `time_limit`; this
  file has no equivalent, so a structurally-impossible request looks like
  a hang rather than a failure. It is not an infinite loop — each trial
  does terminate — but stage 4's systematic search on a doomed graph
  scales badly: with `"0:0,1:0"` on a fully open Diamond, 2x2x2 is
  instant, 3x3x3 exceeds 90 s for three trials, and a single 4x4x4 trial
  did not finish in 300 s. Prefer a small lattice when probing whether a
  distribution is satisfiable at all.

## Provenance

Vendored 2026-08-05 from `generator_serial_debug11.c`, md5
`e7631f4bbcb963d50c382721de3b3c18`, dated 2025-11-03. That is the version
the shipped `generator.exe` was built from and the one
`generator_python.py` was ported from.

### The variant that was NOT taken

A later `generator_serial_debug11.c` exists (md5 `83d7f9d37c72`,
2026-02-27) under `experiments/pruning_research/pruning_algorithm_math*`,
replicated to five archive locations. Editor history confirms it is the
newest source by timestamp, and it was vendored first on that basis. It
is wrong for this slot.

It replaces the per-degree count check in `is_move_safe` with a
cumulative one (`N_leq_v + increase > T_leq_v`). Measured across six
standard SC configurations it sculpts **1/6** where this version and the
Python port both do **6/6**, failing every case where `max_func` sits
below the lattice coordination, which is the ordinary use. Treat it as an
open experiment rather than a newer release. If the cumulative rule turns
out to be the correct one, the Python port has to move with it and the
sculpting failures need fixing first.

`test_c_sculpts_the_configs_python_sculpts` pins this: it fails on the
`83d7f9d3` variant and passes on this one.
