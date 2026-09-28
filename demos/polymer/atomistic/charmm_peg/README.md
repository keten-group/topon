# PEG network with CHARMM parameters

A tetra-PEG network on a diamond lattice (2x2x2 cells, 64 tetrafunctional junctions, 128 strands), built through the
ordinary pipeline with `chemistry.force_field = "charmm"` instead of DREIDING.

- The strands are PEG, four `PEGM` units each (`-CH2-O-CH2-`), the PEG residue of the CHARMM C35r ether force field
  (bundled, MIT, cite Vorobyov et al. 2007 and Lee et al. 2008, see `topon/chemistry/charmm/data/README.md`). The
  monomer SMILES `COC` repeated four times is the same chain, so each unit maps onto `PEGM` by name
  (`charmm_atom_names: ["C1", "O1", "C2"]`) or by graph.
- The junction is a quaternary carbon bonded to four strand ends (a pentaerythritol-type core), residue `PEJ`, typed
  `CC30A` with charge 0 as the quaternary carbon of the CHARMM alkanes. It and a methyl end cap (`PEGC`, the atoms of
  C35r's `MET1` patch) are defined in `peg_junction.str`.
- C35r has every term of the strands and every term at the junction but three (one angle and two torsions around a
  quaternary carbon bonded to an ether CH2). `peg_junction.str` supplies them by analogy, each copied from the C35r
  term named beside it. They are not fitted parameters.

## Run

```bash
cd demos/polymer/atomistic/charmm_peg
topon generate config.json
cd output_charmm_peg/charmm_peg/04_Simulation
lmp -in minimize_1_serial.in       # pair soft, bonded terms with 1-4 weights 0
lmp -in minimize_2_parallel.in     # CHARMM LJ epsilon ramp (lj/cut/coul/long, arithmetic mixing)
lmp -in minimize_3_parallel.in     # full CHARMM: lj/charmmfsw/coul/long + PPPM, minimize, NVT, NPT
```

Take `peg_junction.str` out of `chemistry.charmm.files` and the chemistry stage stops with the list of what is missing
(the `PEJ` residue first, and with only its RTF part, exactly the three terms above). topon never writes a default
parameter.

## What the build holds

3,648 atoms in 576 residues (64 `PEJ`, 512 `PEGM`). Every `PEGM` unit and the network are neutral term by term, no
term matched a wildcard, and there are 14 dihedral types (the C35r O-C-C-O and C-O-C-C torsions carry two and three
Fourier terms, only the first of which holds the 1-4 weight). The run manifest lists the files read and the charge of
each residue kind.

`expected_output/` has the three coefficient includes and the groups file of the chemistry stage, the three stage
scripts, and the LAMMPS logs of the three stages (LAMMPS 2 Apr 2025, 4 OpenMP threads, about 4 minutes in all). Stage
3 minimizes the energy from 12,022 to 3,869 kcal/mol and ends near 307 K. The conformation stage adds noise from the
global random stream, so the relaxed data file is not kept and these numbers change a little from run to run (another
run went from 10,473 to 2,187 kcal/mol and ended near 301 K).
