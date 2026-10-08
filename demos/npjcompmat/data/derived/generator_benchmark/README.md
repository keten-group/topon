# `generator_benchmark/`

Data of the generator benchmark in Appendices B and C of the paper. The folder holds two benchmarks. The t10 files are
the current one, rerun on 6 Oct 2026 with the code of topon 0.4.5. Table 1 (Appendix B) and Table 2 (Appendix C) of
the paper and its supplementary benchmark table come from them, and the other generator numbers of the Appendix B
text were rechecked with the same code. The t8 files are an earlier benchmark on topon 0.2.1, which the earlier
version of Table 1 used.

| File | Content |
|---|---|
| `T10_README.md` | notes of the t10 rerun (code, machine, scripts and the results quoted in the paper) |
| `t10_provenance.json` | code and build of the t10 runs (commit, compiler and build line, MD5 of the generator binary, Python version, CPU, workers) |
| `t10_timing.csv` | Table 1, one row per attempt |
| `t10_summary.csv` | Table 1, one row per search, lattice size, neighbour range and target |
| `t10_327.csv` | the supplementary benchmark table, one network per target of the coarse-grained ensemble with each search |
| `t10_327_log.txt` | printed output of that run, with the targets reached and the median and 90th percentile time per network of each search |
| `t10_bias.csv`, `t10_bias_python_exact.csv` | graph metrics of one network per ensemble target from exact matching (C and Python) and from pruning (C) |
| `t10_bias_summary.json`, `t10_bias_py_summary.json` | paired differences of these metrics between the searches and from the deposited networks |
| `t10_degeneracy.csv`, `t10_degeneracy_summary.json` | 20 networks per target for 10 targets of the ensemble, and their spread relative to the ensemble |
| `t10_fallback_bias.csv` | algebraic connectivity of networks from the fallback of the exact search (the balanced deal) and from the regular deal |
| `t10_fallback_bias_log.txt` | printed comparison of the two (the "6 to 13% higher" of Appendix B) |
| `t10_shells.csv`, `t10_shells_summary.csv` | Table 2, smallest-ring statistics for 1 to 8 neighbour shells, per graph and per table row |
| `t10_shells_log.txt` | printed Table 2 rows and the mean ring size of each reference |
| `t10_placement.csv` | strand placement times quoted in the Appendix B text |
| `t8_final_summary.csv` | generation times of both searches in C and Python. The earlier version of Table 1 took its pruning columns from this file. Its exact-matching rows are from the exact search before the fallback of 0.2.1, and that table did not use them. |
| `t8_exact_v2w10_summary.csv` | generation times of the exact search with the fallback, in C and Python. The earlier version of Table 1 took its exact-matching columns from this file. |
| `t8_exact_v2_327.csv` | the exact search with the fallback on every target of the coarse-grained ensemble, one network per target in C and in Python |
| `tables.md` | benchmark tables G1 (degree-distribution control) and G2 (infeasible requests) of earlier runs |

`scripts/t10/t10_tables.py` builds the current Table 1 from `t10_summary.csv` and writes it to
`tables_checks/table_timing_t10.tex`. `scripts/make_timing_table.py` builds the earlier version of Table 1 from the two
t8 summary files and writes it to `tables_checks/table_timing.tex`.

## Targets

The targets are counts of sites of degree 0 to 6 on 216 sites. They are rescaled to each lattice (scaled by the number
of sites over 216 and rounded, with degree 0 taking the remainder and one site moved from degree 2 to degree 3 when
the degree sum is odd).

- A `10,0,26,75,43,53,9`, the documented example without dangling ends
- T `14,23,23,20,75,41,20`, a typical target of the ensemble
- B `10,44,16,35,44,13,54`, the target of the ensemble that needed the most trials

## t10, the current benchmark

The t10 runs used the main branch of the development repository at commit 54f2474. Between that commit and the source
of topon 0.4.5 only files of the atomistic route changed, so everything the t10 scripts call is the code of 0.4.5.
`T10_README.md` lists the scripts and the results quoted in the paper.

### Table 1

Table 1 covers periodic SC lattices of 216, 512, 1,000 and 1,728 sites (6³ to 12³) with one, two or three neighbour
shells (6, 18 or 26 candidate partners per site), max f 6 and the targets A, T and B, for pruning and exact matching,
each in C and in Python. Each case got up to 100 attempts (seeds 1 to 100), each limited to 120 s. An attempt
succeeded when it returned a network with exactly the requested number of sites of each degree and all active sites
in one connected component. The scripts check this themselves and hash every network. A case whose first 10 attempts
all failed was stopped there. On the 216-site lattice of the coarse-grained ensemble the stopped cases were then
completed to 100 attempts (`t10_extend216.py`), so only the larger lattices keep the early stop. A cell of Table 1
gives the median time per network over the successful attempts in seconds to three significant figures, with the
number of networks found out of 100 in parentheses when it is below 100, and a dash when no attempt succeeded.

C runs call the compiled `generator.c` with `--search=strict` (pruning) or `--search=exact` (exact matching),
`--min-giant-fraction=1` and the seed in `--seed`, and their times include starting the program and writing the
network files. Python runs call `PythonTopologyGenerator` in process with `search` "strict" or "exact" (and
`min_giant_fraction` 1 for the exact search). A Python exact call that gives up is called again with new seeds until a
network is found or the 120 s are used. Other settings are the defaults of the code. All timing runs used 8 workers,
each pinned to one performance core of an Intel Core i9-14900 (Windows 11).

### The supplementary benchmark table and the other checks

`t10_327.py` gives the supplementary benchmark table. Every target of the coarse-grained ensemble (the folder names in `data/mechanics/`, SC 6³
with one neighbour shell and max f 6) got one attempt with each search in each language, with seed 1 and the cap,
checks and cores of Table 1. The other scripts recheck the remaining generator numbers of Appendices B and C. Those
that are not timed ran 4 or 8 at a time on the efficiency cores.

- `t10_bias.py` and `t10_bias_py.py` draw one network per ensemble target with C exact matching, C pruning and Python
  exact matching (seed 1, 120 s cap). They compare the algebraic connectivity, bridges, maximum betweenness and cycle
  rank of the active subgraph between the searches and with the deposited network of the same target. This rechecks
  the sampler comparison of Table G1.
- `t10_degeneracy.py` draws 20 networks (seeds 1 to 20, 60 s cap) for each of 10 ensemble targets that span the
  deposited algebraic connectivity, with C exact matching and with C pruning. This rechecks the degeneracy rows of
  Table G1.
- `t10_fallback_bias.py` compares the algebraic connectivity of networks from the fallback of the exact search and
  from the regular deal, for target T on 216 and 1,728 sites (seeds 1 to 125, the first 100 accepted networks of
  each kind).
- `t10_shells.py` makes Table 2. For each reaction-cured reference of Appendix C (DP 20 and DP 100) and each number
  of neighbour shells, it draws ten graphs with the exact search of the package at the degree counts of the reference
  on its matched cubic cell. It gives the smallest ring through each edge of the elastic core, the mean ring size and
  the Jensen-Shannon divergence to the ring distribution of the reference
  (`data/derived/bond_create/four_distributions.json`).
- `t10_placement.py` times strand placement with the straight, meander and walk routes (three repeats each) on
  networks of 216, 1,000 and 2,744 sites from the exact search.
- `t10_overhead.py` measures how much of a C time is the harness (about 2 ms) and the program start (about 10 ms). It
  prints its result and writes no file.

### Columns

`t10_timing.csv` (rows in the order the attempts finished)

| Column | Meaning |
|---|---|
| `method` | `C_pruning`, `python_pruning`, `C_exact` or `python_exact` |
| `size`, `sites` | lattice edge (6, 8, 10 or 12) and number of sites (216 to 1728) |
| `shells`, `partners` | neighbour shells (1, 2 or 3) and candidate partners per site (6, 18 or 26) |
| `target` | A, T or B |
| `seed` | 1 to 100 |
| `success` | a connected network with exactly the target counts was returned within the cap |
| `returned` | a network was returned within the cap |
| `wall_s` | wall time of the attempt (s) |
| `hash` | MD5 of the sorted edge list, empty when no network was returned |
| `core` | logical CPU the attempt ran on |

`t10_summary.csv`

| Column | Meaning |
|---|---|
| `method`, `sites`, `partners`, `target` | as in `t10_timing.csv` |
| `attempts` | attempts made (100, or 10 for a case stopped after 10 failures) |
| `successes` | attempts that succeeded |
| `median_s`, `q25_s`, `q75_s`, `p90_s` | median, quartiles and 90th percentile of the wall time per network over the successful attempts (s), empty when none succeeded |
| `distinct` | distinct networks (edge sets) among the successful attempts |

`t10_327.csv` has the columns `method`, `success`, `returned`, `wall_s`, `hash` and `core` of `t10_timing.csv`, and
`target`, the counts of sites of degree 0 to 6 (the folder name in `data/mechanics/`).

`t10_bias.csv`, `t10_bias_python_exact.csv` and `t10_degeneracy.csv`

| Column | Meaning |
|---|---|
| `method` | `C_exact`, `C_pruning` or `python_exact` |
| `key` | the target (the folder name in `data/mechanics/`) |
| `seed` | 1 to 20 (`t10_degeneracy.csv` only, the other two use seed 1) |
| `success` | a connected network with exactly the target counts was found |
| `wall_s` | wall time (s) |
| `hash` | MD5 of the sorted edge list (`t10_degeneracy.csv` only) |
| `lambda_2`, `num_bridges`, `max_betweenness`, `cycle_rank` | algebraic connectivity, number of bridges, maximum betweenness centrality and cycle rank of the active subgraph, empty when no network was found |

`t10_bias_summary.json` and `t10_bias_py_summary.json` give the networks found with each search and, for each
comparison (e.g., `C_exact_minus_deposited`), the number of paired targets `n`. For each metric they give the mean
difference (`mean_diff`), the same in units of the standard deviation of the deposited ensemble (`over_ensemble_sd`)
and the Wilcoxon signed-rank p (`wilcoxon_p`). `t10_degeneracy_summary.json` gives for each search the networks found
per target, whether they are all distinct, and the mean standard deviation within a target over the ensemble standard
deviation (`within_over_ensemble_sd`, over the targets with at least 5 networks).

`t10_fallback_bias.csv`

| Column | Meaning |
|---|---|
| `size` | lattice edge (6 or 12, i.e., 216 or 1,728 sites) |
| `target` | T |
| `mode` | `random` (the regular deal) or `fallback` (the balanced deal) |
| `seed` | 1 to 125 |
| `success` | a connected network with exactly the target counts was found in this attempt |
| `lambda_2` | algebraic connectivity of the active subgraph |
| `hash` | MD5 of the sorted edge list |

`t10_shells.csv` and `t10_shells_summary.csv`

| Column | Meaning |
|---|---|
| `spec` | `N20` or `N100`, the DP 20 and DP 100 references (cells of 14³ and 9³ sites) |
| `shells` | neighbour shells (1 to 6, and 8) |
| `seed` | 1 to 10 (`t10_shells.csv` only) |
| `success`, `calls` | a graph was found, and the calls of the exact search it took |
| `mean_ring` | mean smallest ring size over the edges of the elastic core that lie on a ring |
| `js` | Jensen-Shannon divergence (log2, ring sizes 3 to 15) between the ring distribution and that of the reference |
| `frac_odd` | fraction of these rings that are odd |
| `core_edges` | edges of the elastic core |
| `dist` | ring sizes and their counts (JSON) |
| `n`, `mean_ring_sd`, `js_sd` | graphs per row and standard deviations over them (`t10_shells_summary.csv` only, whose `mean_ring`, `js` and `frac_odd` are means over the graphs) |

`t10_placement.csv`

| Column | Meaning |
|---|---|
| `sites`, `dp`, `edges` | sites of the network (216, 1,000 or 2,744), strand DP and number of strands |
| `route` | `straight`, `meander` or `walk` |
| `rep` | repeat (0 to 2, also the seed of the placement) |
| `graph_s` | time to generate the graph with the Python exact search (s) |
| `place_s` | time to place all strands (s) |
| `beads` | beads placed |
| `core` | logical CPU the case ran on |

The `_log.txt` files are the printed output of the runs.

### Rerunning t10

The scripts are in `scripts/t10/` and import each other. Run them as `python scripts/t10/<script>.py` from any
folder. They need the topon source and the compiled C generator in one folder, named by the environment variable
`TOPON_BENCH`.

```
$TOPON_BENCH/src/            source tree of topon 0.4.5 (the scripts import topon from here and check that they do)
$TOPON_BENCH/generator.exe   C generator built from src/topon/topology/csrc/generator.c
```

The runs of the paper built the generator with

```
gcc -O2 -o generator.exe generator.c -lm
```

(gcc 15.2.0 of MinGW-Builds, x86_64-win32-seh, with the MD5 of the binary in `t10_provenance.json`) and used Python
3.12.10 with networkx, numpy, pandas, psutil and scipy.

| Variable | Default | Use |
|---|---|---|
| `TOPON_BENCH` | none (required) | folder with `src/` and `generator.exe` |
| `T10_TMP` | `$TOPON_BENCH/tmp` | scratch folder of the C runs |
| `T10_OUT` | this folder | folder the scripts read their inputs from and write their outputs to |
| `T10_CORES` | `0,2,4,6,8,10,12,14` | logical CPUs of the timing workers, one worker per CPU (the performance cores of the i9-14900) |
| `T10_CHECK_CORES` | `28,29,30,31` (`t10_bias.py`, `t10_bias_py.py`), `24,25,26,27` (`t10_degeneracy.py`), `24` to `31` (`t10_fallback_bias.py`, `t10_shells.py`) | logical CPUs of the checks that are not timed |

| Script | Writes | When |
|---|---|---|
| `t10_latest.py` | `t10_timing.csv`, `t10_summary.csv` | first (seeds 1 to 10 in every case, then 11 to 100 where one of them succeeded) |
| `t10_327.py` | `t10_327.csv` (printed summary in `t10_327_log.txt`) | after `t10_latest.py` |
| `t10_extend216.py` | adds to `t10_timing.csv`, rewrites `t10_summary.csv` | after `t10_327.py` |
| `t10_placement.py` | `t10_placement.csv` | after `t10_extend216.py` |
| `t10_overhead.py` | nothing (prints) | on an idle machine |
| `t10_bias.py` | `t10_bias.csv`, `t10_bias_summary.json` | any time |
| `t10_bias_py.py` | `t10_bias_python_exact.csv`, `t10_bias_py_summary.json` | after `t10_bias.py` |
| `t10_degeneracy.py` | `t10_degeneracy.csv`, `t10_degeneracy_summary.json` | any time |
| `t10_fallback_bias.py` | `t10_fallback_bias.csv` (printed comparison in `t10_fallback_bias_log.txt`) | any time |
| `t10_shells.py` | `t10_shells.csv`, `t10_shells_summary.csv` (printed rows in `t10_shells_log.txt`) | any time |
| `t10_tables.py` | `tables_checks/table_timing_t10.tex` | after the timing runs |
| `t10_shells_table.py` | nothing (prints the Table 2 rows) | after `t10_shells.py` |

The four timing scripts ran one at a time on the performance cores, and the checks that are not timed ran on the
efficiency cores. The times depend on the machine, on the compiler and on running 8 attempts at a time on pinned
cores, so a rerun elsewhere gives other times, and cases near the 120 s cap can also give other success counts.
`t10_latest.py` and `t10_327.py` append every attempt to their file as it finishes and skip the attempts already
recorded there, so an interrupted run resumes. In this folder every attempt is already recorded, so a new run needs
`T10_OUT` set to another folder.

`t10_bias.py`, `t10_bias_py.py` and `t10_degeneracy.py` also read `t0_deposited_graphs.csv` from this folder
(whatever `T10_OUT` is). It holds the four metrics of the deposited networks of `data/mechanics/` (columns `key`,
`lambda_2`, `num_bridges`, `max_betweenness`, `cycle_rank`, with the network checks and the same metrics as
`data/csv/mechanics.csv` gives them in the `csv_` columns) from an earlier audit. The values agree with
`data/csv/mechanics.csv` (`folder_name`, `lambda_2`, `num_bridges`, `max_betweenness`, and `cyclomatic_idx` for
`cycle_rank`) to within 3e-14. `t10_overhead.py` reads process times through the Windows API and runs on Windows
only. The CPU pinning (psutil) works on Windows and Linux.

## t8, the earlier benchmark

### Timing summaries

Both summary files have one row per search, lattice size, neighbour range and target. Every case had 100 attempts
(seeds 1 to 100). An attempt succeeded when it returned, within 120 s, a network with exactly the requested number of
sites of each degree and all active sites in one connected component.

| Column | Meaning |
|---|---|
| `method` | `C_pruning`, `python_pruning`, `C_exact` or `python_exact` |
| `sites` | 216, 512, 1000 or 1728 (periodic SC lattices of 6³ to 12³ sites) |
| `partners` | candidate partners per site, 6, 18 or 26 (one, two or three neighbour shells) |
| `target` | degree distribution A, T or B (above) |
| `attempts` | attempts made (100) |
| `successes` | attempts that succeeded |
| `median_s`, `p90_s` | median and 90th percentile of the wall time per network over the successful attempts (s), empty when none succeeded |
| `distinct` | distinct networks (edge sets) among the successful attempts |

C runs are the compiled `generator.c` (`--search=strict`, or `--search=exact --min-giant-fraction=1`, with the seed in
`TOPON_SEED`), and their times include starting the program and writing the network files. Python runs call the
generator in process, with `min_giant_fraction` 1 for the exact search. The exact search is the code of topon 0.2.1.
All runs were made on one workstation (Intel i9-14900, Windows 11), mostly 10 attempts at a time.

### `t8_exact_v2_327.csv`

One row per target of the coarse-grained ensemble and search, on the periodic SC lattice of 216 sites with one
neighbour shell and max f 6. Each target got one network with seed 1, a 120 s cap and a connected network required.

| Column | Meaning |
|---|---|
| `method` | `C_exact` or `python_exact` |
| `target` | counts of sites of degree 0 to 6 (the folder name in `data/mechanics/`) |
| `deposited_trial` | trial index of the deposited network (in its file names in `data/mechanics/<target>/`) |
| `n_dangling`, `n_f6` | sites of degree 1 and of degree 6 in the target |
| `success` | a connected network with exactly the target counts was found |
| `returned` | a network was returned within the cap |
| `wall_s` | wall time (s) |
| `hash` | MD5 of the sorted edge list |
| `attempt` | the attempt that found the network, counted from 0 (attempts 6 and later used the fallback) |
| `fallback` | the network came from the fallback |
| `calls` | calls to the Python search (empty for C) |
