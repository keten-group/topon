# Raw-data analyses

These scripts turned the raw simulation output into the derived files in `data/derived/`. They are kept as a record of
how those files were made. The notebooks do not run them. No script here runs a simulation.

## Folders

`paths.py` defines the folders that all scripts use. Two environment variables change them.

| Variable | Default | Content |
|---|---|---|
| `NPJ_RAW_ROOT` | `data/raw/` of this companion | raw simulation output |
| `NPJ_ANALYSIS_OUT` | `scripts/raw_data_analyses/output/` | intermediate and output files |

Each script writes into a subfolder of `NPJ_ANALYSIS_OUT` named after its own folder (`cg_mechanics/`, `cg_structure/`,
`stress_definition/`, `tg/`). `reach_distributions.py` writes into `bond_create/`. The subfolders are created when
needed. Files that one script writes and another reads (e.g., `four_seed_metrics.csv`, `box_curves_seed1.npz`, the
`cache/` of `cg_structure/`) are passed through these subfolders. The deposited companion data (`data/dataset.pkl`,
`data/csv/`, `data/mechanics/`, `data/derived/`) are found relative to the scripts.

## Raw inputs in `data/raw/`

The stress-strain files of all pulls and the MSD files of all cooling histories are deposited. Unpack them in
`data/raw/` with `tar` or with Python's `tarfile`:

```
cd data/raw
tar -xJf cg_stress_strain_seed1.tar.xz
tar -xJf cg_stress_strain_seed2.tar.xz
tar -xJf cg_stress_strain_seed3.tar.xz
tar -xJf cg_stress_strain_seed4.tar.xz
tar -xJf tg_cooling.tar.xz
```

This gives the layout the scripts read:

| Path below `NPJ_RAW_ROOT` | Content |
|---|---|
| `seed<k>/<folder_name>/stress_strain_npt_{x,y,z}.dat` | stress-strain file of every pull, velocity seeds k = 1 to 4 |
| `<system>/DP<n>/<history>/msd.dat` | MSD file of each system (PDMS, PMTFPS), chain length (DP 10, 30, 100) and cooling history (`original`, `replicate2`, `replicate3`) |

`MANIFEST_raw.csv` gives the original path of every archived file. No script here reads `cg_lammps_inputs.tar.xz`.

With the deposit alone, `cg_mechanics/three_seed_analysis.py`, `cg_mechanics/four_seed_analysis.py` (up to its
descriptor section) and the two `tg/` scripts run. The other scripts need files that are not deposited.

## Inputs that are not deposited

Some scripts read files of the original simulation trees. They look for them below `NPJ_RAW_ROOT`.

| Path below `NPJ_RAW_ROOT` | Content | Read by |
|---|---|---|
| `crosslinker/sc_6x6x6/<folder_name>/minimize_equilibrate/` | LAMMPS data files of every build stage (`nodes.data` to `equilibrated_network.data`), `slurm-*.out` and `log.lammps` of the anneal and production runs | `cg_structure/extract_per_network.py`, `cg_structure/task1_followup.py` |
| `crosslinker/sc_6x6x6/<folder_name>/tensile_test_attractive_xyz/` | `slurm-*.out` and `log.lammps` of the seed-1 pulls | `stress_definition/s1_extract_box.py` |
| `bond_create_validation/` | bond/create reference data files, Topon builds (`data/runs/N{20,100}_v3b/`) and the parser `scripts/refnet.py` | `reach_distributions.py` |

A few tables come from earlier analyses that are not part of this folder. They are not deposited. The scripts read them
from `NPJ_ANALYSIS_OUT/cg_mechanics/`.

| File | Read by | Use |
|---|---|---|
| `maxflow_loadpaths.csv`, `D_core_all.csv`, `simple_slice_metrics.csv`, `tearing_metrics.csv` | `four_seed_analysis.py` | graph descriptors for the printed descriptor comparison (read after `four_seed_cube.npz` and `four_seed_metrics.csv` are written) |
| `figure6_SI_stats_k12.csv` | `s2_analysis.py` | reproduction check of the 12-pull effect sizes (read before `network_true_vs_nominal.csv` and `fig6_cells_true_vs_nominal.csv` are written) |
| `per_direction_metrics_v2.csv` | `s2_analysis.py` | check of an earlier nominal-stress estimate (read after all tables are written) |

## Scripts

Within each folder, run the scripts in the order of its table. `cg_structure/task1_orientation.py` also needs
`block_matching_seed1.csv` from `stress_definition/s1_extract_box.py` and `four_seed_metrics.csv` from
`cg_mechanics/four_seed_analysis.py`. `stress_definition/s2_analysis.py` also needs `four_seed_metrics.csv`. The last
column names the file in `data/derived/` that came from each script. These outputs were copied or reduced into
`data/derived/`, and `data/derived/MANIFEST.csv` records the source of each file.

### `cg_mechanics/` (metrics of the 3,924 coarse-grained pulls)

| Script | Reads | Writes | File in `data/derived/` |
|---|---|---|---|
| `three_seed_analysis.py` | `seed{1,2,3}/` stress-strain files (deposited), `data/csv/mechanics.csv` | `three_seed_metrics.csv`, `three_seed_per_pull_metrics.csv`, `three_seed_network_summary.csv`, `three_seed_cube.npz` | none (input of `four_seed_analysis.py`) |
| `four_seed_analysis.py` | `seed4/` stress-strain files (deposited), the `three_seed_*` files, `data/dataset.pkl`, four descriptor tables (not deposited) | `four_seed_metrics.csv`, `four_seed_cube.npz`, `four_seed_network_summary.csv` | `mechanics/mechanics_pulls_4seeds.csv` (the values of `four_seed_cube.npz`, one row per pull) |

### `stress_definition/` (true and nominal stress)

| Script | Reads | Writes | File in `data/derived/` |
|---|---|---|---|
| `s1_extract_box.py` | thermo output of the 981 seed-1 pulls (not deposited), `seed1/` stress-strain files (deposited), `data/csv/mechanics.csv` | `block_matching_seed1.csv`, `box_curves_seed1.npz` | `stress_definition/box_curves_seed1_examples.npz` (6 of the 981 arrays) |
| `s2_analysis.py` | outputs of `s1_extract_box.py`, `seed{2,3,4}/` stress-strain files (deposited), `four_seed_metrics.csv`, `data/csv/mechanics.csv`, `data/derived/mechanics/mechanics_12pull.csv`, `data/dataset.pkl`, two earlier tables (not deposited) | `s2_analysis_output.txt`, `area_universality_seed1.csv`, `calibration_*.csv`, `per_pull_true_vs_nominal.csv`, `network_true_vs_nominal.csv`, `fig6_cells_true_vs_nominal.csv` | `stress_definition/network_true_vs_nominal.csv`, `fig6_cells_true_vs_nominal.csv`, `area_universality_seed1.csv` |
| `s3_figure.py` | outputs of `s1_extract_box.py` and `s2_analysis.py` | `fig6_cells_summary_12pull.csv`, `true_vs_nominal_comparison.png` (the true versus nominal stress comparison) | `stress_definition/fig6_cells_summary_12pull.csv` |

### `cg_structure/` (structure and equilibration of the 327 networks)

`common.py` holds the shared paths and parsers. `thermo_parse.py` reads the thermo output. Outputs go to `cache/`,
`data/`, `tables/` and `figures/` in `NPJ_ANALYSIS_OUT/cg_structure/`.

| Script | Reads | Writes | File in `data/derived/` |
|---|---|---|---|
| `extract_per_network.py` | LAMMPS data files, `slurm-*.out` and `log.lammps` of every network (not deposited) | `cache/per_network_raw.csv`, `cache/strand_vectors.npz`, `cache/thermo.npz` | `cg_structure/strand_vectors_first_eq.npz` (keys of `strand_vectors.npz`) |
| `task2_counts.py` | the cache, the graphs `data/mechanics/<folder_name>/*.gpickle`, `mechanics_12pull.csv` | `data/task2_counts_per_network.csv`, `data/scaffold_per_axis_planes.csv`, `tables/task2_quadrant_table.csv`, `tables/task2_quadrant_table.md`, `tables/task2_crosschecks.txt` | `cg_structure/task2_counts_per_network.csv`, `task2_quadrant_table.csv` |
| `task1_orientation.py` | the cache, `scaffold_per_axis_planes.csv`, `mechanics_12pull.csv`, `four_seed_metrics.csv`, `block_matching_seed1.csv` | `data/task1_orientation_per_network.csv`, `data/task1_per_axis.csv`, `tables/task1_*.csv`, `tables/task1_extra.txt` | `cg_structure/task1_orientation_per_network.csv`, `task1_per_axis.csv`, `task1_mechanics_correlations.csv`, `task1_within_network_axis_correlations.csv` |
| `task1_followup.py` | outputs of the scripts above, `mechanics_12pull.csv`, `equilibrated_network.data` of every network (not deposited) | `cache/strand_level_memory_eq.csv`, `tables/task1_followup.txt` | `cg_structure/strand_memory_eq.npz` (columns `fmin`, `fmax`, `P2`) |
| `task1_figure.py` | outputs of the scripts above | `figures/task1_orientation.png` (the strand orientation check) | none |
| `task3_stationarity.py` | `cache/per_network_raw.csv`, `cache/thermo.npz`, `mechanics_12pull.csv` | `data/task3_stationarity_per_network.csv`, `tables/task3_*`, `cache/task3_acf.npz`, `figures/task3_*.png` (the stationarity check) | `cg_structure/stationarity_per_network.csv` and `stationarity_scalars.json` (recomputed from the cache with the code of this script) |

### `tg/` (atomistic glass transition)

| Script | Reads | Writes | File in `data/derived/` |
|---|---|---|---|
| `make_figure5_replicates.py` | `msd.dat` of the three cooling histories (deposited), `data/npj_style_v1.py` | `Figure_4_Tg_replicates.pdf` and `.png` (manuscript Fig. 5, file `Figure_4_Tg`) | `tg/tg_msd_histories.csv` (made with the estimator `load_lag` of this script) |
| `make_lag_rationale_figure.py` | `msd.dat` of DP 100 (deposited), `data/npj_style_v1.py` | `Tg_lag_rationale.pdf` and `.png` (the check of the 4 ps MSD lag) | none |

`TG_LAG=<ps>` selects another lag for `make_figure5_replicates.py`. The file name then ends in `_<ps>ps`.

### `reach_distributions.py`

| Script | Reads | Writes | File in `data/derived/` |
|---|---|---|---|
| `reach_distributions.py` | `bond_create_validation/` (not deposited) | `bond_create/reach_distributions.json` | `bond_create/reach_distributions.json` (`a`, `n_junctions` and the reach values of this file, rounded to four decimals) |

## Not included

- `four_distributions.py`, which made the cycle and Z1+ distributions of Fig. 9 (`data/derived/bond_create/four_distributions.json`). It depends on the Topon development tree and on Z1+.
- The LAMMPS deformation runs behind the stress-strain files of Fig. 9.
- `target_vs_realized.py` and `figure_completion.py`, which made Fig. 8 (`data/derived/entanglement/`).
- `refnet.py`, the LAMMPS data-file parser that `reach_distributions.py` imports from `bond_create_validation/scripts/`.

The other files in `data/derived/` were not made by these scripts. `data/derived/README.md` describes them.

## Packing `data/raw/`

`scripts/build_raw_data.py` packed `data/raw/` from the full simulation folders. It also reads `NPJ_RAW_ROOT`, which
must then name the folder that holds `crosslinker/` and `glass_transition_atomistic/`. These folders are not
deposited, so the script stops with a message when they are missing. It records each source path relative to that
folder, prefixed with `Studies/`, in `MANIFEST_raw.csv`.
