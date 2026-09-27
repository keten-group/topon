# `data/raw/`: raw simulation output

The files behind the derived data in `data/derived/`, as written by LAMMPS and by the generator. `MANIFEST_raw.csv`
lists every archived file with its size, SHA-256 and original path. `scripts/build_raw_data.py` assembled this folder
and checked, before writing, that the peak stress of every archived stress-strain file equals the `uts` column of
`data/derived/mechanics/mechanics_pulls_4seeds.csv` (all 3,924 pulls, exact). Unpack with `tar -xJf <file>` or
Python's `tarfile`. Trajectories, restart files and logs are not included.

## Coarse-grained networks (Lennard-Jones units)

| File | Content |
|---|---|
| `cg_stress_strain_seed{1,2,3,4}.tar.xz` | `seed<k>/<folder_name>/stress_strain_npt_{x,y,z}.dat` for the four velocity seeds (327 × 3 files each). Columns `TimeStep v_strain v_stress`, i.e., engineering strain along the pulling axis and true stress −P_aa (ε/σ³), from `fix ave/time 1000 10 10000` (mean of 10 samples, one row per 10,000 steps, 800 rows). One file has 787 rows (seed 4, `14_28_22_46_20_58_28`, y), which the analysis uses as it is. |
| `cg_velocity_seeds.csv` | Velocity seed of each replicate (4928459, 7391052, 2648317, 9184735). Replicates 2 to 4 ran the seed-1 inputs with only this number changed. |
| `cg_lammps_inputs.tar.xz` | Per network, `minimize_equilibrate/` (minimization and the staged equilibration `001_Anneal.in` to `006_Ext_05.in`) and `tensile_test_attractive_xyz/` (`tensile_{x,y,z}_parallel_p0_attractive.lmp`). The inputs are identical across networks except `minimize_1_serial.lmp`, which names the network's data file. `minimize_equilibrate/polymer.coeff` is written by LAMMPS (`write_coeff`) during the minimization, so ten networks carry the larger FENE range used there. The equilibration itself sets R0 = 1.5 σ for every network. |
| `cg_generation_settings.csv` | Generator call of every network: target counts `n_f0` to `n_f6` (site level, 216 sites), lattice, size, periodicity, maximum functionality, `max_trials`, and `trial_ordinal`, the trial at which the saved graph was found. The C generator was seeded from the clock, so no seed was recorded. The graphs are in `data/mechanics/<folder_name>/`. |

`folder_name` is the target `n_f0_n_f1_..._n_f6` and is the case ID used throughout `data/`.

## Atomistic glass transition (PDMS and PMTFPS, DP 10, 30 and 100)

| File | Content |
|---|---|
| `tg_cooling.tar.xz` | `<system>/DP<n>/<history>/msd.dat` and the per-temperature inputs `simulation_<T>K.lmp` for the three cooling histories (`original`, `replicate2`, `replicate3`), plus `simulation_template.lmp`, which regenerates the original inputs. `msd.dat` columns: `Temp` (K), `Time` (fs), `MSD_all`, `MSD_allcm`, `MSD_node`, `MSD_nodecm` (Å²), `Density` (g cm⁻³) and `CohesiveEnergy`, one row per ps. PMTFPS is called FPDMS in the original folders. PMTFPS DP 10 at 280 K was run twice, and its `msd.dat` holds both runs (all 20 windows are used). |
