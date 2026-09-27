**Table G1. Degree-distribution control: success, runtime and error.** SC 6x6x6 periodic (216 sites, z = 6), max f 6, unless stated. Error = L1 distance between achieved and requested site counts over f = 0..6; every returned graph was checked independently. Times are wall clock per graph (median / p90 / max) on one core of an i9-14900 (Windows 11); the paper-era binary ran in WSL2.

| Method | Targets | Success | Time per graph | P(f) error | Notes |
|---|---|---|---|---|---|
| Paper-era strict sculptor (C binary, 2025-11-03), deposited graphs | 327 manuscript | 327/327 | trial index median 16, p90 7,980, max 575,056 | 0 (327/327) | 32 targets needed >10^4 trials; all active subgraphs connected; wall time not recorded |
| Exact search (V52), one call (<=6 seeds) | 327 manuscript | 231/327 (71%) | 106 ms / 3.1 s / 10.3 s | 0 | a failed call costs 8.1 s (median) |
| Exact search, up to 6 calls (<=36 seeds) | 327 manuscript | 259/327 (79%) | 131 ms / 7.5 s / 58.7 s (cumulative) | 0 | by number of dangling ends: f1 0-20: 149/149, f1 21-30: 70/87, f1 31-40: 38/77, f1 >40: 2/14 |
| Exact search, up to 6 calls | 221 unused candidates | 87/221 (39%; first call 71) | 198 ms / 12.3 s / 39.5 s | 0 | paper-era binary (60 s) reaches 1/134 of the rest; their feasibility is unknown |
| Strict sculptor, Python (package), 60 s cap | 30 manuscript, stratified by deposited trial index k | 26/30 | 817 ms / 16.9 s / 45.8 s | 0 | k<=16: 10/10, 16<k<=1e4: 10/10, k>1e4: 6/10 |
| Strict sculptor, paper-era C binary, 60 s cap | 30 manuscript, stratified by deposited trial index k | 17/30 | 1.3 s / 5.7 s / 22.8 s | 0 | k<=16: 6/10, 16<k<=1e4: 8/10, k>1e4: 3/10 |
| Strict sculptor, current repo C exe (V47 build), 60 s cap | 30 manuscript, stratified by deposited trial index k | 17/30 | 938 ms / 9.6 s / 30.1 s | 0 | k<=16: 5/10, 16<k<=1e4: 8/10, k>1e4: 4/10 |
| Strict fallback, Python, 60 s | the 68 manuscript targets the exact search missed | 56/68 | 3.5 s / 35.6 s / 57.3 s | 0 | median 502 trials to success |
| Strict fallback, paper-era binary, 60 s | the 68 manuscript targets the exact search missed | 46/68 | 5.0 s / 32.3 s / 52.9 s | 0 | median 515 trials to success |
| Exact, then strict fallback | 327 manuscript | 319/327 | - | 0 | 8 not reached within these caps; all are feasible (deposited graphs exist) and all had k > 10^4 |
| Degeneracy: one P(f), many graphs (exact search, 5 seeds) | 10 manuscript (spanning lambda_2) | 40 graphs (8 targets x 5) | - | 0 (40/40 identical P(f)) | all pairwise non-isomorphic; within-target SD / ensemble SD: lambda_2 0.85, max betweenness 0.70, bridges 0.06, cycle rank 0 |
| Degeneracy: one P(f), many graphs (C strict (V47 exe), up to 5 saves in 60 s) | 10 manuscript (spanning lambda_2) | 37 graphs (7 targets x 5) | - | 0 (37/37 identical P(f)) | all pairwise non-isomorphic; within-target SD / ensemble SD: lambda_2 0.73, max betweenness 0.88, bridges 0.11, cycle rank 0 |
| Sampler comparison (paired, same P(f)) | 96 manuscript | - | - | 0 | exact - deposited: lambda_2 +0.017 (p 9e-04), max betweenness -0.0025 (p 2e-04); Python strict - deposited: -0.001 (p 0.44), +0.0010 (p 0.23) |
| Baseline: uniform random bond pruning to the target edge count | 327 x 1000 draws | 0/327,000 exact | < 1 ms | mean L1 125 sites (TV 0.29); best draw 24 | controls the mean degree only |
| Baseline: connectivity-preserving random pruning | 327 x 20 draws | 0/6,540 exact | ~0.1 s | mean L1 126 (TV 0.29); best draw 28 | mean f0 1.1 vs 12.2 requested, f6 8.2 vs 30.5 |
| Portability: SC 6^3 (z 6), N = 216 | targets A, B rescaled | exact A 3/3, B 0/3 (median 21 ms); strict A 1/1, B 0/1 (median 277 ms) | - | 0 | strict: Python, 60 s cap |
| Portability: SC 6^3, 3 shells (z 26), N = 216 | targets A, B rescaled | exact A 3/3, B 3/3 (median 11 ms); strict A 0/1, B 0/1 | - | 0 | strict: Python, 60 s cap |
| Portability: BCC 5^3 (z 8), N = 250 | targets A, B rescaled | exact A 3/3, B 3/3 (median 36 ms); strict A 1/1, B 0/1 (median 351 ms) | - | 0 | strict: Python, 60 s cap |
| Portability: FCC 4^3 (z 12), N = 256 | targets A, B rescaled | exact A 3/3, B 3/3 (median 6 ms); strict A 1/1, B 0/1 (median 1.3 s) | - | 0 | strict: Python, 60 s cap |
| Portability: MIX 80/10/10 6^3, N = 294-304 | targets A, B rescaled | exact A 3/3, B 3/3 (median 44 ms); strict A 1/1, B 0/1 (median 2.3 s) | - | 0 | strict: Python, 60 s cap |

A = 10_0_26_75_43_53_9 (documented example, no dangling ends); B = 10_44_16_35_44_13_54 (largest deposited trial index, 575,056). Unused candidates: the 221 rows of network_candidates_SC_6x6x6_v2.txt that are not among the 327.

**Table G2. Infeasible or ill-posed requests** (base target A on SC 6x6x6 unless stated; outcome and time to answer; strict runs capped at 60 s).

| Request | Class | Exact search | Strict, Python | Strict, C (V47 exe) |
|---|---|---|---|---|
| odd_degree_sum (`0:10,1:0,2:25,3:76,4:43,5:53,6:9`; SC 6x6x6, max f 6) | arithmetic | refused (1 ms) | no refusal, no graph (60.0 s) | refused (16 ms) |
| sites_sum_211_lt_216 (`0:5,1:0,2:26,3:75,4:43,5:53,6:9`; SC 6x6x6, max f 6) | site count | graph returned, f0 differs (24 ms) | no refusal, no graph (60.1 s) | no refusal, no graph (60.0 s) |
| sites_sum_221_gt_216 (`0:15,1:0,2:26,3:75,4:43,5:53,6:9`; SC 6x6x6, max f 6) | site count | graph returned, f0 differs (21 ms) | no refusal, no graph (60.0 s) | refused (11 ms) |
| active_226_gt_216 (`0:10,1:0,2:26,3:75,4:63,5:53,6:9`; SC 6x6x6, max f 6) | site count | refused (1 ms) | no refusal, no graph (60.1 s) | refused (10 ms) |
| degree7_on_SC_z6 (`0:10,1:0,2:26,3:75,4:43,5:48,6:9,7:5`; SC 6x6x6, max f 7) | above lattice max | refused (1 ms) | refused (1 ms) | no refusal, no graph (60.0 s) |
| manuscript_on_Diamond_z4 (`0:10,1:0,2:26,3:75,4:43,5:53,6:9`; Diamond 3x3x3, max f 6) | above lattice max | refused (9 ms) | refused (1 ms) | no refusal, no graph (60.0 s) |
| degree5_6_above_maxf4 (`0:10,1:0,2:26,3:75,4:43,5:53,6:9`; SC 6x6x6, max f 4) | above max_functionality | refused (1 ms) | refused (1 ms) | refused (11 ms) |
| all_active_at_z6_with_10_vacancies (`0:10,1:0,2:0,3:0,4:0,5:0,6:206`; SC 6x6x6, max f 6) | scaffold-unreachable | searched, gave up with error (3 ms) | no refusal, no graph (60.0 s) | no refusal, no graph (60.0 s) |
| three_f2_needs_triangle (`0:213,1:0,2:3,3:0,4:0,5:0,6:0`; SC 6x6x6, max f 6) | scaffold-unreachable | searched, gave up with error (3 ms) | no refusal, no graph (60.0 s) | no refusal, no graph (60.0 s) |
| two_f1_only (`0:214,1:2,2:0,3:0,4:0,5:0,6:0`; SC 6x6x6, max f 6) | scaffold-unreachable | searched, gave up with error (9.6 s) | no refusal, no graph (60.0 s) | no refusal, no graph (60.0 s) |
