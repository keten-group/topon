# Threshold and cross-validation checks

These scripts test the quadrant analysis of Figs. 6 and 7 in two ways. The first is whether the significant contrasts
depend on splitting the networks at the median. The second is how well the 11 graph descriptors predict the UTS and
toughness of networks they were not fitted to. They read only files in `data/derived/` and run no simulation.

## Headline numbers

- The 31 significant contrasts of Fig. 7 (of 55) keep their sign when the quadrants are split at the 40th or 60th
  percentile instead of the median. 31 and 28 of them stay significant, and the three that drop have q between 0.051
  and 0.058.
- Ridge regression cross-validated over held-out networks predicts UTS with R^2 = 0.68 and toughness with R^2 = 0.36
  from the 11 descriptors. The dangling-end fraction alone gives 0.32 and 0.20.
- The variance inflation factors of maximum betweenness centrality and mean eigenvector centrality are 1.9 and 2.5.

## Running

Run the scripts from the companion folder (`demos/npjcompmat/`) in the environment of `requirements.txt`.

```
python scripts/checks/threshold_cv/s0_reproduce.py
python scripts/checks/threshold_cv/s1_thresholds.py
python scripts/checks/threshold_cv/s2_cv_ridge.py
python scripts/checks/threshold_cv/s3_collinearity.py
python scripts/checks/threshold_cv/s4_ceiling_check.py
```

The paths are relative to the scripts, so they also run from any other folder. Run `s0_reproduce.py` first, because
`s2_cv_ridge.py` reads the reliabilities it writes. `s2_cv_ridge.py` takes one to two minutes (most of it in the
permutation test), `s3_collinearity.py` up to half a minute (the bootstrap) and the others a few seconds. Each script
prints its results and writes them to `out/` next to it. The scripts need numpy, scipy and pandas. If scikit-learn is installed, `s2_cv_ridge.py` also compares its
ridge code with scikit-learn's `RidgeCV` on one fold and prints the difference (below 1e-12).

## Scripts

| Script | What it does | Writes to `out/` |
|---|---|---|
| `common.py` | loads the data and holds the shared definitions (quadrants, Cohen's d, Welch test, Benjamini-Hochberg) | |
| `ridge_cv.py` | ridge regression with the penalty chosen inside each training set, and repeated K-fold cross-validation over networks | |
| `s0_reproduce.py` | reproduces the 55 cells and 44 correlations of the published analysis and the reliabilities of the network means | `s0_reproduce.txt`, `reliability.csv` |
| `s1_thresholds.py` | repeats the 55 cells with other thresholds and checks the toughening cells at fixed UTS | `thresholds_*.csv`, `s1_thresholds.txt` |
| `s2_cv_ridge.py` | cross-validated ridge regression of UTS and toughness from several predictor sets | `cv_*.csv`, `s2_cv_ridge.txt` |
| `s3_collinearity.py` | variance inflation factors, the correlation matrix, and partial correlations at fixed dangling-end fraction | `collinearity_vif.csv`, `collinearity_corr_matrix.csv`, `partial_corr_dangling.csv`, `s3_collinearity.txt` |
| `s4_ceiling_check.py` | which reliability of the network means bounds the cross-validated R^2 | `ceiling_check.csv`, `s4_ceiling_check.txt` |

## Data and definitions

| Input in `data/derived/` | Use |
|---|---|
| `mechanics/mechanics_pulls_4seeds.csv` | UTS and toughness of the 3,924 pulls (327 networks x 3 axes x 4 seeds) |
| `mechanics/mechanics_12pull.csv` | graph descriptors, degree counts, `frac_dangling` and `cyclomatic_idx` of each network |
| `figure_stats/figure6_stats.csv`, `figure6_continuous.csv` | reference values of the published analysis |
| `cg_structure/task1_within_network_axis_correlations.csv` | printed by `s4_ceiling_check.py` for comparison |

The definitions are those of the statistics cell of `generate_checks.ipynb`. Network UTS and toughness are the means of
the 12 pulls. The quadrants split both properties at the median (Q1 weak and brittle, Q2 weak and tough, Q3 strong and
brittle, Q4 strong and tough). The 11 descriptors are lambda_2, lambda_2,core, the mean shortest path length L, the cycle
rank of the active graph (E - N_active + 1), maximum betweenness centrality, bridges, degree bimodality D_bi, the
degree SD sigma_k, degree entropy H, mean eigenvector centrality and assortativity. Cohen's d uses the pooled SD, the
tests are Welch t tests, and the Benjamini-Hochberg correction runs over the 55 cells. The dangling-end fraction D is
the number of f = 1 sites divided by 216.

## Results

### Reproduction (`s0_reproduce.py`)

The quadrants hold 128/35/35/129 networks, and the 55 cells (d, p and q) and the 44 correlations agree exactly with
the tables in `data/derived/figure_stats/`. 31 of the 55 cells are significant. At fixed dangling-end fraction the
partial correlations of lambda_2 and lambda_2,core with UTS are 0.556 and 0.565, and UTS and toughness correlate with
r = 0.78. The reliability of the 12-pull network means is 0.93 for UTS and 0.80 for toughness with the three cube axes
fixed (i.e., new velocity seeds), and 0.50 and 0.21 with the axes treated as a random sample of loading directions
(as in `tables_checks/table_variance.tex`).

### Thresholds (`s1_thresholds.py`)

"Kept" counts the 31 significant cells of the median split that stay significant with the same sign. The group sizes
are from/to.

| Split | Q1->Q2 | Q3->Q4 | Q1->Q3 | Q2->Q4 | Q1->Q4 | Cells tested | Significant | Kept of 31 | Lost | Sign changes | New |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Median (published) | 128/35 | 35/129 | 128/35 | 35/129 | 128/129 | 55 | 31 | 31 | 0 | 0 | 0 |
| 40th percentile | 93/38 | 38/158 | 93/38 | 38/158 | 93/158 | 55 | 35 | 31 | 0 | 0 | 4 |
| 60th percentile | 163/33 | 33/98 | 163/33 | 33/98 | 163/98 | 55 | 31 | 28 | 3 | 0 | 3 |
| Terciles of both, middle dropped | 79/1 | 2/75 | 79/2 | 1/75 | 79/75 | 11 | 10 | 10 of 10 tested | 0 | 0 | 0 |
| Terciles within halves | 54/54 | 55/55 | 54/54 | 55/55 | 79/75 | 55 | 40 | 31 | 0 | 0 | 9 |

The plain tercile split leaves one and two networks in Q2 and Q3 (UTS and toughness correlate), so only Q1->Q4 can be
tested (a cell needs at least 5 networks per group, and the correction then runs over the tested cells). The
within-halves split cuts each transition into thirds along its own property inside the relevant half of the other one
(e.g., Q3->Q4 compares the bottom and top thirds of toughness among the networks above the UTS median). The three cells
lost at the 60th percentile lie just above the threshold (lambda_2,core Q1->Q3 and Q1->Q4 with q = 0.058 and 0.057,
and max betweenness Q2->Q4 with q = 0.051). The mixed splits (UTS at the 40th and toughness at the 60th percentile,
and the reverse) keep 29 and 28 of the 31.

A common percentile from 30 to 70 in steps of 5 keeps 30, 30, 31, 31, 31, 28, 28, 25 and 27 of the 31 cells, and
none changes sign. The smaller quadrants never hold fewer than 28 networks. The eigenvector centrality and lambda_2
cells and max betweenness Q1->Q3 and Q1->Q4 are significant at every percentile. The cells that drop are mostly
lambda_2,core, sigma_k Q3->Q4, assortativity Q2->Q4 and degree entropy Q1->Q3.

Q4 is also stronger than Q3 (mean UTS 4.03 vs 3.85, Welch p = 2e-5). Among the 164 networks above the UTS median,
bridges and D_bi correlate with toughness (r = -0.35 and -0.37), but at fixed UTS their partial correlations are -0.08
(p = 0.33) and -0.16 (p = 0.045). The Q3->Q4 differences in bridges and bimodality therefore mostly follow the UTS
difference between the two quadrants.

### Cross-validated ridge regression (`s2_cv_ridge.py`)

The unit is the network. The cross-validation uses 10 folds repeated 20 times, with the same 200 splits for every
model. Inside each training set the predictors are standardized and the ridge penalty is chosen from 91 values by
leave-one-out error, so the test networks never enter the fit. R^2 is the pooled out-of-fold R^2 of each repeat, and the
table gives its mean and range over the 20 repeats and the mean and SD of the 200 per-fold values. The last column
divides R^2 by the reliability of the network means with the axes fixed (0.93 and 0.80).

| Response | Predictors | CV R^2 | Range | Per-fold R^2 | R^2 / reliability |
|---|---|---|---|---|---|
| UTS | D alone | 0.323 | 0.318-0.328 | 0.29 +- 0.14 | 0.35 |
| UTS | 11 descriptors (G) | 0.678 | 0.672-0.682 | 0.65 +- 0.12 | 0.73 |
| UTS | D + G | 0.677 | 0.672-0.682 | 0.65 +- 0.12 | 0.73 |
| UTS | degree fractions f = 0..6 (Pf) | 0.502 | 0.492-0.513 | 0.47 +- 0.14 | 0.54 |
| UTS | Pf + G | 0.675 | 0.668-0.680 | 0.65 +- 0.12 | 0.72 |
| toughness | D alone | 0.198 | 0.193-0.203 | 0.16 +- 0.13 | 0.25 |
| toughness | G | 0.357 | 0.348-0.367 | 0.32 +- 0.15 | 0.44 |
| toughness | D + G | 0.356 | 0.348-0.367 | 0.32 +- 0.15 | 0.44 |
| toughness | Pf | 0.220 | 0.208-0.228 | 0.18 +- 0.13 | 0.27 |
| toughness | Pf + G | 0.351 | 0.343-0.364 | 0.31 +- 0.15 | 0.44 |
| toughness | UTS alone | 0.607 | 0.603-0.609 | 0.58 +- 0.12 | |
| toughness | UTS + G | 0.615 | 0.608-0.621 | 0.59 +- 0.11 | |
| toughness residual after UTS | G | 0.003 | -0.015-0.023 | -0.04 +- 0.08 | |

The descriptors add 0.354 to the R^2 of D for UTS and 0.158 for toughness, and D adds nothing to the descriptors. They
add 0.173 (UTS) and 0.132 (toughness) to the full degree composition Pf. All 20 repeats agree in the sign of these
gains. At fixed UTS the descriptors add 0.008 to the R^2 of toughness (permutation p = 0.005 with 200 permutations of
the descriptor rows), i.e., about 1% of the toughness variance beyond UTS.

Added one at a time to D, the largest gains for UTS come from lambda_2,core (+0.21), lambda_2 (+0.21), L (+0.20),
eigenvector centrality (+0.15), sigma_k (+0.13), bridges (+0.11) and max betweenness (+0.09). Dropped one at a time
from D + G, only L (-0.045) and sigma_k (-0.016) cost more than 0.01, because the descriptors share most of their
information.

### Which reliability bounds the prediction (`s4_ceiling_check.py`)

The CV R^2 for UTS (0.68) lies above the reliability with random axes (0.50), and the same holds for toughness (0.36 vs
0.21). After the descriptor prediction is removed, the residuals of a network along its three axes are negatively
correlated (mean r = -0.14 for UTS), so the covariance between axes understates what network structure explains. The
reliabilities with the axes fixed (0.93 and 0.80) are the bound for predicting the 12-pull means, and the descriptors
reach 73% and 44% of them.

### Collinearity and partial correlations (`s3_collinearity.py`)

The condition number of the correlation matrix is 140 for the 11 descriptors and 1,905 with D added. The median
pairwise |r| among the descriptors is 0.37, and 4 of the 55 pairs have |r| > 0.8. The partial correlations use linear
adjustment, with p from the t distribution and 95% bootstrap intervals over networks (5,000 resamples).

| Descriptor | VIF (11) | VIF (11 + D) | r with D | Partial r with UTS at fixed D [95% CI] | Partial r with toughness at fixed D [95% CI] |
|---|---|---|---|---|---|
| lambda_2 | 3.9 | 4.4 | -0.55 | +0.556 [+0.47, +0.63] | +0.360 [+0.26, +0.45] |
| lambda_2,core | 2.9 | 3.0 | +0.38 | +0.565 [+0.48, +0.64] | +0.362 [+0.27, +0.45] |
| L | 6.2 | 6.9 | +0.46 | -0.542 [-0.62, -0.46] | -0.346 [-0.44, -0.25] |
| cycle rank | 2.6 | 2.6 | +0.07 | +0.044 [-0.06, +0.14] | -0.019 [-0.13, +0.09] |
| max betweenness | 1.9 | 1.9 | +0.15 | -0.368 [-0.47, -0.26] | -0.217 [-0.32, -0.11] |
| bridges | 11.7 | 188.6 | +0.99 | -0.416 [-0.50, -0.33] | -0.279 [-0.37, -0.19] |
| D_bi | 5.7 | 6.0 | +0.84 | -0.160 [-0.29, -0.03] | -0.172 [-0.27, -0.07] |
| sigma_k | 19.3 | 19.3 | +0.84 | -0.439 [-0.53, -0.34] | -0.205 [-0.31, -0.09] |
| H | 6.3 | 6.4 | +0.66 | -0.324 [-0.42, -0.21] | -0.107 [-0.21, -0.01] |
| eigenvector | 2.5 | 2.5 | -0.52 | +0.483 [+0.39, +0.57] | +0.251 [+0.14, +0.36] |
| assortativity | 1.6 | 1.6 | +0.32 | -0.314 [-0.41, -0.21] | -0.234 [-0.33, -0.13] |

The large VIF of bridges comes from their correlation with D (r = 0.993). Bridges still correlate with UTS at fixed D
(partial r = -0.42). `partial_corr_dangling.csv` also gives the partial correlations with toughness at fixed D and UTS
together, and the Benjamini-Hochberg q values over the 22 tests with UTS and toughness.

## Reproducibility

The splits, the permutations and the bootstrap use fixed seeds, so a rerun gives the same numbers. With the versions in
`requirements.txt` the scripts write the files in `out/` byte for byte. Other versions of numpy, scipy or pandas change
only the last digits. With numpy 2.5.3, scipy 1.18.1 and pandas 3.0.6 the largest difference in any number was 3e-14,
and the text files differed only in the reproduction differences that `s0_reproduce.py` prints (below 1e-15 instead
of 0).
