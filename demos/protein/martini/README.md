# Martini 3 resilin network

Eight resilin chains of eighteen repeats (`GGRPSDSYGAPGGGN`, 270 residues each) on the BFM lattice, crosslinked
through their tyrosines and built with Martini 3 and the Martini3-IDP bonded terms. The chain topology is the bundled
polyply ITP of the resilin reference, so polyply is not needed here. For any other sequence `topon protein` runs
`polyply gen_params -lib martini3`.

## Run

From the repository root,

```bash
topon protein --config demos/protein/martini/config.json --output runs/martini_demo
# or
python demos/protein/martini/run.py --output runs/martini_demo
```

Then, in `runs/martini_demo/relaxation/`,

```bash
lmp -in protein_network_stage1.in    # soft overlap removal (hierarchical)
lmp -in protein_network_stage2.in    # LJ epsilon ramp
lmp -in protein_network_stage3.in    # minimization and a short NVT at 310 K
```

## What the build holds

The network gels at 12.5 % conversion (9 dityrosine reactions). One of them closes a cycle around the periodic box and
is dropped by the writer, which leaves 8 crosslink bonds (Tyr SC4-SC4). The network has 4,032 beads and a net charge
of 0, in a box sized for 0.85 g/cm³. The three stages take one to two minutes on 4 OpenMP threads, and stage 3
minimizes from 1.7e5 to -7,215 kcal/mol and ends its NVT run near 310 K.

## Files in `expected_output/`

| File | What |
|---|---|
| `protein_network.in.settings` | Martini 3 pair coefficients (every pair explicit) and bonded coefficients |
| `protein_network.in.groups` | protein, water and per-class groups |
| `relaxation/protein_network_stage{1,2,3}.in` | the three stages |
| `relaxation/stage{1,2,3}.log` | LAMMPS logs of the three stages (LAMMPS 2 Apr 2025, 4 OpenMP threads) |
| `protein_network_summary.json`, `protein_network_topology.json` | inputs, counts, charges, and the BFM snapshots |

The build also writes `protein_network.data` (beads in 10 columns with image flags, bonds, angles, dihedrals and
impropers) and `protein_network_chain.itp` (the chain topology used, here the bundled `nat_pro.itp`). They are not
kept here.

The port from GROMACS to LAMMPS approximates reaction-field electrostatics and the restricted-bending angle (see
`docs/USAGE.md`).
