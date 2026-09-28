# `data/derived/`: derived data read by the notebooks

Everything the two notebooks need beyond `data/dataset.pkl` and `data/csv/`. All files here are derived data, and no
raw simulation output is stored in this folder (the raw data are in `data/raw/`). `MANIFEST.csv` lists every file with
its source and SHA-256. Text files are stored with LF line endings, and the checksums are those of the stored files.

CG quantities are in Lennard-Jones units (length σ, energy ε, time τ). `folder_name` identifies one of the CG
networks (the case ID of `data/csv/mechanics.csv`, with the graph files in `data/mechanics/<folder_name>/`). Quadrants
come from a median split of the 12-pull network means of UTS and toughness (Q1 weak/brittle, Q2 weak/tough, Q3
strong/brittle, Q4 strong/tough, with 128/35/35/129 networks). Read CSV files with `float_precision='round_trip'` to get
the stored floats exactly (pandas' default parser can be off in the last bit).

## mechanics/

| File | Rows | Columns | Units, notes | Provenance |
|---|---|---|---|---|
| `mechanics_pulls_4seeds.csv` | 3,924 (327 networks × 3 axes × 4 seeds) | `folder_name`; `axis` (x, y, z); `seed` (1-4); `velocity_seed` (4928459 original, 7391052, 2648317, 9184735); `uts` peak true stress −P_aa (ε/σ³); `sab` strain at break (engineering strain of the first post-peak point with stress < 5% of UTS, or the last point if there is none); `toughness` trapezoid area under true stress vs engineering strain up to `sab` (ε/σ³, not a work per volume) | one row per uniaxial pull | raw `stress_strain_npt_{x,y,z}.dat` of the four velocity seeds (`data/raw/cg_stress_strain_seed{1,2,3,4}.tar.xz`); metrics by `scripts/raw_data_analyses/cg_mechanics/three_seed_analysis.py` and `four_seed_analysis.py`, which write `four_seed_cube.npz`; this file is that cube, bit for bit |
| `mechanics_12pull.csv` | 327 | the 38 columns of `data/csv/mechanics.csv` with `uts_mean`, `toughness_mean`, `strain_break_mean` and their `*_std` (population SD) over the 12 pulls, `num_directions` = 12; plus `cycle_rank` (**older convention**, E − 216 + 1, vacancies counted as nodes; the analyses use `cyclomatic_idx` = E − N_active + 1), `deg_bimodality`, `Quadrant` | graph descriptors unchanged from v0.1.0; `graph_file` keeps the relative path of v0.1.0 | network table of the analyses, copied verbatim |

## figure_stats/ (reference values for the checks in both notebooks)

| File | Content |
|---|---|
| `figure6_stats.csv` | 55 cells of Fig. 7a (11 metrics × 5 transitions): n, Cohen's d, 95% network-bootstrap CI (2,000 resamples, seed 0), Welch p, Benjamini-Hochberg q, Holm p, flags |
| `figure6_bars.csv` | Fig. 7b,c bars: Δ\|d\| with bootstrap CI |
| `figure6_continuous.csv` | Pearson and partial correlations of each metric with UTS and toughness |
| `figure5b_degree_bands_k12.csv` | Fig. 6b: pooled degree distribution per quadrant with 95% network-bootstrap band |

Copies of the statistics tables of the analyses. `generate_checks.ipynb` recomputes all four and checks them against
these files.

## tg/ (atomistic Tg, manuscript Fig. 5 and the Tg checks)

| File | Columns | Units, notes | Provenance |
|---|---|---|---|
| `tg_msd_histories.csv` (756 rows) | `system` (PDMS, PMTFPS), `dp` (10, 30, 100), `history` (original, replicate2, replicate3), `Temp` (K), `n_windows`, `MSD_0ps` … `MSD_10ps` (Å²) | mean over the 10 ps windows (at least 10 per temperature, and a temperature run twice keeps all windows) of `MSD_all` at lag l ps, the estimator of `make_figure5_replicates.py`; 42 temperatures (100-305 K) per system and history | raw `msd.dat` of the three cooling histories (`data/raw/tg_cooling.tar.xz`). The `original` rows agree with the deposited `data/csv/tg_msd.csv` to ≤ 8.9e-16 Å² (averaging order) |
| `oo_bond_check.csv` (6 rows) | `system`, `atoms`, `bonds`, `OO_bonds_system_data`, `OO_bonds_ready2deform_data` | text cells (FPDMS = PMTFPS) | record of the replicate Tg simulations, table "O-O (peroxide) bond check" |

## bond_create/ (Fig. 9, Appendix C)

| File | Content | Provenance |
|---|---|---|
| `four_distributions.json` | for `N20`, `N100` × `reference` (fix bond/create) / `topon`: `cycle` (shortest cycle through each bridging strand, {length: count}), `Z_bridge_hist` (Z1+ kinks per bridging strand), `summary` (`Zmean_bridge`, `partners_bridge`, …) and further descriptors not plotted | bond/create comparison (`four_distributions.py`, not included since it needs the topon development tree and Z1+) |
| `N20_reference_stress_strain_high.txt`, `N100_reference_…`, `N20_topon_…`, `N100_topon_…` | LAMMPS `fix print` output with step, engineering strain and stress −p_zz (ε/σ³); the figure smooths over 100 rows | deformation runs of the bond/create references and of the topon builds (same deformation protocol) |
| `reach_distributions.json` | for `N20`, `N100` × `reference` / `topon_built` / `topon_relaxed`: reach of every bridging strand in units of the mean junction spacing a = (V / N_junctions)^(1/3), and a | `scripts/raw_data_analyses/reach_distributions.py`; reach and a rounded to 4 decimals |

## entanglement/ (Fig. 8, Appendix B)

| File | Content | Provenance |
|---|---|---|
| `target_vs_realized.json` | one record per network and request: `net`, `dp` (20, 40, 80, 120), `target` (requested Z per strand), `saturated` (request clamped at the ceiling), `Zmean` (realised Z1+ entanglements per strand), `frac_zero`, `Ne_CK`, `Zmax`, `secs` | entanglement analyses (`target_vs_realized.py`, not included since it needs the topon development tree and Z1+) |
| `figure_completion.json` | the same, with `kind` = `zero` (unentangled reference) or `ceiling` (DP 120), `rho`, `bond_built` | the same (`figure_completion.py`) |

## stress_definition/ (true versus nominal stress, and the nominal-stress column of the cell table)

| File | Content | Provenance |
|---|---|---|
| `network_true_vs_nominal.csv` | per network: UTS and toughness under true and nominal stress for seed 1 (3 pulls) and all 12 pulls, and the quadrant (1-4) under each definition | `scripts/raw_data_analyses/stress_definition/s2_analysis.py` |
| `fig6_cells_true_vs_nominal.csv` | the 55 Fig. 7 cells for four quadrant definitions (`set`): d, p, BH q, significance under true and nominal stress | the same |
| `area_universality_seed1.csv` | per strain: V/V0 and A/A0 statistics over the 981 seed-1 pulls (all, and pulls not yet broken) | the same |
| `fig6_cells_summary_12pull.csv` | robustness class of each cell (reference for the check in `generate_checks.ipynb`, which regenerates it) | `s3_figure.py` |
| `box_curves_seed1_examples.npz` | 6 arrays `<net>|<axis>` (800 × 7) with step, engineering strain, true stress, A/A0, V/V0, density and bond count, for the two example networks of panels a, b of the stress figure | 6 of the 981 arrays of `box_curves_seed1.npz` (`s1_extract_box.py`, from the thermo output of the seed-1 pulls, which is not deposited) |

## cg_structure/ (composition and quadrant tables, stationarity and orientation figures)

| File | Content | Provenance |
|---|---|---|
| `task2_counts_per_network.csv` | per network: site counts by degree, strands, 2-core strands and junctions, mean degree over active sites, beads, volume ⟨V⟩ over the last production block (σ³), densities (σ⁻³), quadrant, 12-pull means | `scripts/raw_data_analyses/cg_structure/task2_counts.py` (LAMMPS data files and thermo output, not deposited) |
| `task2_quadrant_table.csv` | per-quadrant mean/SD/min/max, Kruskal-Wallis p, Spearman ρ with UTS and toughness (reference, regenerated in `generate_checks.ipynb`) | the same |
| `network_2core_audit.csv` | `E` (strands), `N_active` (sites with f ≥ 1), `core_nodes`, `core_edges` (2-core), `cascade` (sites removed by the 2-core peeling beyond the f = 1 sites) | 2-core decomposition of the deposited graphs |
| `task1_orientation_per_network.csv` | per network and build stage (gen, min2, min3, minsoft, minfin, minreal, nvt, first, eq): strand cubic invariant C4, order parameter S, directional memory ⟨P2(u·e0)⟩, z scores against isotropic and rotation nulls, bond statistics, structure factors, box lengths | `task1_orientation.py` (strand vectors from the LAMMPS data files of every stage, via `extract_per_network.py`) |
| `task1_per_axis.csv` | per network and pull axis: Q_aa, box length, 4-seed UTS and toughness, strands crossing the weakest lattice plane (scaffold and 2-core) | `task1_orientation.py` |
| `task1_mechanics_correlations.csv`, `task1_within_network_axis_correlations.csv` | correlations quoted in panels e, f of the orientation figure | `task1_orientation.py`, `task1_followup.py` |
| `strand_memory_eq.npz` | `fmin`, `fmax` (degrees of the two end junctions), `P2` = P2(u·e0) of each of the 123,216 equilibrated strands | columns of the strand-level cache (8.4 MB, not deposited) |
| `strand_vectors_first_eq.npz` | `first__<net>`, `eq__<net>` (strand end-to-end vectors at the MD start and after equilibration, σ, float32), `axis__<net>` (lattice direction of each strand in the generator output) | keys of the strand-vector cache (`extract_per_network.py`, not deposited) |
| `stationarity_per_network.csv` | per network: block means of potential energy per bead, density and box anisotropy max(L)/min(L) − 1 over the five production blocks; drift t statistics (OLS on 20 sub-block means) of PE, density and pressure; block-5 means of P and of the normal-stress differences with standard errors; snapshot shear stresses P_xy, P_xz, P_yz of the pull-start state; thermal SD of (P_xx − P_yy)/2 | recomputed with the code of `task3_stationarity.py` from the thermo cache (43 MB, parsed from the Slurm outputs, not deposited); equal to that script's per-network table |
| `stationarity_scalars.json` | `anneal_end_aniso_mean` (dotted line of panel c of the stationarity figure) | the same |

## generator_benchmark/, software/

| File | Content | Provenance |
|---|---|---|
| `generator_benchmark/tables.md` | benchmark tables G1 (degree-distribution control) and G2 (infeasible requests) | runs of the topon generator (`make_tables.py`) |
| `generator_benchmark/t8_final_summary.csv`, `t8_exact_v2w10_summary.csv`, `t8_exact_v2_327.csv` | generation times behind Table 1 (Appendix B) and the exact search on every target of the ensemble (columns in `generator_benchmark/README.md`) | timing runs of the topon generators in C and Python |
| `software/software_seeds.csv` | rows of the software table (`section`, `item`, `value`, `source`; LaTeX cell text) | static record |

The analysis scripts that produced the files above from raw simulation output are in `scripts/raw_data_analyses/` (see
its README for the raw inputs they read).
