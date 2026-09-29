# `data/raw/`: raw simulation output

The files behind the derived data in `data/derived/`, as written by LAMMPS and by the generator, and the atomistic
systems the cooling runs start from. `MANIFEST_raw.csv` lists every archived file with its size, SHA-256 and original
path. `scripts/build_raw_data.py` assembled this folder and checked, before writing, that the peak stress of every
archived stress-strain file equals the `uts` column of `data/derived/mechanics/mechanics_pulls_4seeds.csv` (all 3,924
pulls, exact). Unpack with `tar -xJf <file>` or Python's `tarfile`. Trajectories, restart files and logs are not
included.

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
| `atomistic_systems.tar.xz` | `<system>/DP<n>/02_Chemistry/`, the six systems as built by topon, before the atoms are placed. `system.data` holds the atoms with their types and charges (all coordinates are still zero), and the bonds, angles and dihedrals. `system.in.settings` holds the DREIDING pair, bond, angle and dihedral coefficients, and `system.groups` the group `nodes` (see the note below). `system_node_info.txt` lists every crosslink node with its degree and atom ID, and `system_edge_info.txt` every strand with its two nodes and the IDs of its atoms. |
| `atomistic_equilibrated.tar.xz` | `<system>/DP<n>/05_ExtendedSampling/ready2deform.data`, the six systems at the end of the equilibration described in the paper (an anneal from 1000 K to 400 K, then 25 ns at 300 K and 1 atm), with coordinates, image flags and velocities. The first step (305 K) of every cooling history starts from this file, and the replicate histories draw new velocities (their seeds are in their inputs). |

The folder names are those the cooling inputs refer to. Seen from `<system>/DP<n>/<history>/`, the first step
(`simulation_305K.lmp`) reads `../05_ExtendedSampling/ready2deform.data` and includes `../02_Chemistry/system.in.settings`
and `../02_Chemistry/system.groups`, and each later step reads the restart file of the step before. Unpacked in one
folder, the three archives give this layout, and a cooling history runs from its own folder (here PDMS DP 10, original
history).

```
tar -xJf tg_cooling.tar.xz
tar -xJf atomistic_systems.tar.xz
tar -xJf atomistic_equilibrated.tar.xz
cd PDMS/DP10/original
lmp -in simulation_305K.lmp
lmp -in simulation_300K.lmp
```

and so on down to 100 K. The inputs append to `msd.dat`, so rerun them in a copy of the history folder to keep the
deposited file.

topon wrote the files of `02_Chemistry/` (the first line of `system.data` reads "LAMMPS data file (Dreiding)"). The
minimization and equilibration that led to `ready2deform.data` ran with LAMMPS 22 Jul 2025 (Update 1), and its
`write_data` wrote the file (see its first line). The `02_Chemistry/` files have Windows line endings, which LAMMPS
reads without change. The `*.displace` files of the build are not included, since no deposited input reads them.

The atom IDs in `system.groups` are one lower than those in `system_node_info.txt`, because the builder wrote
zero-based IDs at the time. The group `nodes` therefore holds 114 of the 121 node atoms and six carbon atoms. It enters
only the `MSD_node` and `MSD_nodecm` columns of `msd.dat`, and the analysis uses `MSD_all`.
