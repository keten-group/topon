# `generator_benchmark/`

Data of the generator benchmark in Appendix B of the paper.

| File | Content |
|---|---|
| `t8_final_summary.csv` | generation times of both searches in C and Python. Table 1 takes its pruning columns from this file. Its exact-matching rows are from the exact search before the fallback of 0.2.1, and the table does not use them. |
| `t8_exact_v2w10_summary.csv` | generation times of the exact search with the fallback, in C and Python. Table 1 takes its exact-matching columns from this file. |
| `t8_exact_v2_327.csv` | the exact search with the fallback on every target of the coarse-grained ensemble, one network per target in C and in Python |
| `tables.md` | benchmark tables G1 (degree-distribution control) and G2 (infeasible requests) |

`scripts/make_timing_table.py` builds Table 1 from the two summary files and writes it to
`tables_checks/table_timing.tex`.

## Timing summaries

Both summary files have one row per search, lattice size, neighbour range and target. Every case had 100 attempts
(seeds 1 to 100). An attempt succeeded when it returned, within 120 s, a network with exactly the requested number of
sites of each degree and all active sites in one connected component.

| Column | Meaning |
|---|---|
| `method` | `C_pruning`, `python_pruning`, `C_exact` or `python_exact` |
| `sites` | 216, 512, 1000 or 1728 (periodic SC lattices of 6³ to 12³ sites) |
| `partners` | candidate partners per site, 6, 18 or 26 (one, two or three neighbour shells) |
| `target` | degree distribution A, T or B (below) |
| `attempts` | attempts made (100) |
| `successes` | attempts that succeeded |
| `median_s`, `p90_s` | median and 90th percentile of the wall time per network over the successful attempts (s), empty when none succeeded |
| `distinct` | distinct networks (edge sets) among the successful attempts |

The targets are counts of sites of degree 0 to 6 on 216 sites. They are rescaled to each lattice (scaled by the number
of sites over 216 and rounded, with degree 0 taking the remainder and one site moved from degree 2 to degree 3 when
the degree sum is odd).

- A `10,0,26,75,43,53,9`, the documented example without dangling ends
- T `14,23,23,20,75,41,20`, a typical target of the ensemble
- B `10,44,16,35,44,13,54`, the target of the ensemble that needed the most trials

C runs are the compiled `generator.c` (`--search=strict`, or `--search=exact --min-giant-fraction=1`, with the seed in
`TOPON_SEED`), and their times include starting the program and writing the network files. Python runs call the
generator in process, with `min_giant_fraction` 1 for the exact search. The exact search is the code of this release
(0.2.1). All runs were made on one workstation (Intel i9-14900, Windows 11), mostly 10 attempts at a time.

## `t8_exact_v2_327.csv`

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
