# CHARMM36m resilin network

Eight resilin chains of twelve repeats (`GGRPSDSYGAPGGGN`, 180 residues each) on the BFM lattice, crosslinked
through their tyrosines and built all-atom with CHARMM36m. Each crosslink is the `DITY` patch (a CE2-CE2 bond, the two
ring carbons retyped and both HE2 removed, see `topon/protein_network/charmm/data/README.md`). Every term comes from
the bundled CHARMM36m RTF/PRM through `topon.forcefield.charmm`, which writes the 1-4 terms, NBFIX and the CMAP grids
of the same file and stops on any term it cannot find.

## Run

From the repository root,

```bash
topon protein --config demos/protein/charmm/config.json --output runs/charmm_demo/w0
# or, dry and 35 wt% water in one go
python demos/protein/charmm/run.py --output runs/charmm_demo --water 0,35
```

Then, in each `relaxation/` folder,

```bash
lmp -in protein_network_stage1.in    # pair soft, removes overlaps
lmp -in protein_network_stage2.in    # CHARMM LJ epsilon ramp under nve/limit
lmp -in protein_network_stage3.in    # full CHARMM36m: minimize, NVT, NPT -> ../system_equilibrated.data
```

## What the build holds

The chains gel at 22.9 % conversion (11 dityrosine reactions), and the summary records one cluster holding all 8
chains. None of the crosslinks winds around the box, so all 11 are built. The dry network has 16,610 atoms and a net
charge of 0 (the RTF charges sum to zero in exact decimal arithmetic). At 8 repeats per chain this seed does not gel,
and the build would stop and say so (see `--allow-no-gel`).

## Files in `expected_output/`

| File | What |
|---|---|
| `protein_network.in.settings` | CHARMM36m coefficients with the 1-4 columns, NBFIX and the dihedral 1-4 weights |
| `protein_network.in.settings.soft` | the bonded terms with weights 0, for the `pair_style soft` stages |
| `protein_network.in.settings.lj` | the LJ pair coefficients for the stage-2 ramp |
| `protein_network.in.groups` | `protein`, `water`, `ions` and one group per chain |
| `relaxation/protein_network_stage{1,2,3}.in` | the three stages |
| `relaxation/protein_network.in.omega*` | omega and CA-chirality restraints of the internal-coordinate build, released before each output |
| `relaxation/stage{1,2,3}.log` | LAMMPS logs of the three stages (LAMMPS 2 Apr 2025, 4 OpenMP threads) |
| `protein_network_summary.json`, `protein_network_topology.json` | inputs, counts, charges, and the BFM snapshots |

The build also writes `protein_network.data` (atoms in 10 columns with image flags, bonds, angles, dihedrals with one
row per Fourier term, impropers and CMAP, about 3.8 MB) and `charmm36m.cmap` (the `fix cmap` grids, written from the
PRM). They are not kept here.

The atoms are placed from the RTF internal coordinates (trans peptide bonds, L chirality, real side-chain geometry),
which is the default of `topon protein`. The three stages take about 32 minutes on 4 OpenMP threads, 11 of them in
stage 2. Stage 3 minimizes from 46,820 to -10,035 kcal/mol, and its NPT run ends at 300 K and 1.00 g/cm³ (from 0.85).
In the relaxed network 32 of the 16,901 bonds are longer than 1.25 times their length
(`python -m topon.protein_network check-bonds system_equilibrated.data`). All of them sit at six places where a bond
passed through a ring (three prolines and three tyrosines) during the soft stage and now holds it open. A
crossing-free relaxation is not part of this demo.
