# Appendix tables with the latest topon (6 Oct 2026)

Tables 1 and 2 of the paper and its supplementary benchmark table, regenerated with `main` of the development repository at 54f2474 (the code frozen for the v0.4.5 release). Every Appendix B number that depends on the generator was rechecked with the same code.

## Code and machine

- Source: `git archive` of 54f2474 in `$TOPON_BENCH/src`, imported from there (the scripts assert the path).
- C generator: `generator.exe` built from the same source with `gcc -O2 -o generator.exe generator.c -lm` (gcc 15.2.0, md5 589fbd0bf28408f358d412dd5e950b85). `t10_provenance.json` in this folder records the build.
- Machine: Intel Core i9-14900. Timing runs use 8 workers, each pinned to one performance core (logical CPUs 0, 2, ..., 14). The checks that are not timed run on efficiency cores (logical CPUs 24 to 31).

## Scripts (`scripts/t10/`) and outputs (this folder)

| Script | What it does | Outputs |
|---|---|---|
| `t10_latest.py` | Table 1 grid: SC 6^3 to 12^3, 1 to 3 shells, targets A/T/B, C and Python, pruning and exact matching, 100 seeds, 120 s cap, connected network required, every network checked and hashed. Cases with no success in seeds 1-10 stop there. | `t10_timing.csv` (one row per attempt), `t10_summary.csv` |
| `t10_extend216.py` | Seeds 11-100 for the 216-site cases stopped after 10 failures, so every 216-site case has 100 attempts | appends to `t10_timing.csv`, rewrites `t10_summary.csv` |
| `t10_tables.py` | Table 1 body in seconds, three significant digits, count when below 100 | `tables_checks/table_timing_t10.tex` |
| `t10_327.py` | Supplementary benchmark table: one attempt per ensemble target (327), all four methods, same cap and checks | `t10_327.csv`, `t10_327_log.txt` |
| `t10_overhead.py` | Harness and program-start overhead of the C timings on an idle core | printed (harness ~2 ms, program start ~10 ms) |
| `t10_bias.py`, `t10_bias_py.py` | lambda_2, bridges and betweenness of exact-matching and pruning networks against the deposited networks, paired over the 327 targets | `t10_bias*.csv`, `t10_bias*_summary.json` |
| `t10_degeneracy.py` | 20 networks per target for 10 targets, spread within one P(f) relative to the ensemble | `t10_degeneracy.csv`, `t10_degeneracy_summary.json` |
| `t10_fallback_bias.py` | lambda_2 of the balanced deal against the regular deal, target T, 216 and 1,728 sites | `t10_fallback_bias.csv`, `t10_fallback_bias_log.txt` |
| `t10_shells.py`, `t10_shells_table.py` | Table 2: smallest-ring statistics at the reference P(f) for 1 to 8 neighbor shells, 10 graphs per row, from the package's exact search | `t10_shells.csv`, `t10_shells_summary.csv` |
| `t10_placement.py` | Strand placement time for straight, meander and walk routes (Appendix B text) | `t10_placement.csv` |

## Results used in the paper

- Table 1: exact matching finds networks in all 36 cases (C median at most 1.85 s). Pruning finds networks in 19 cases in C and 18 in Python. At 216 sites every case has 100 attempts; the completed cells agree with the earlier runs (C pruning, target B, 6 partners: 37.9 s, 18 of 100, before 38 s, 17; target T, 26 partners: 91.8 s, 4, before 89 s, 4). Python is up to 19 times slower than C for pruning with one shell and about as fast with more shells; for exact matching the gap is 9 to 43 times at 1,728 sites. All networks found are distinct.
- Supplementary benchmark table: C pruning 325/327 (median 0.0435 s), Python pruning 319/327 (0.156 s), C exact 327/327 (0.0498 s), Python exact 327/327 (0.485 s), one attempt of at most 120 s per target.
- Exact matching draws lambda_2 about 0.7 ensemble standard deviations higher than pruning (C +0.66 against the deposited networks, Python +0.65; C and Python exact agree, p 0.8). Current pruning matches the deposited networks (-0.10 SD, p 0.17).
- One P(f), many graphs: the spread of lambda_2 and maximum betweenness within a target is 0.64 to 0.79 of the ensemble spread.
- Balanced deal: lambda_2 +6.5% (216 sites) and +13.4% (1,728 sites) for target T.
- Table 2: same conclusions as before (three to four shells for DP 20, six to eight for DP 100).
