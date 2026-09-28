# Bundled CHARMM36m files

`top_all36m_prot_C2L.rtf` and `par_all36m_prot_C2L.prm` hold, in one topology
and one parameter file,

- the CHARMM36m protein force field (`top_all36_prot.rtf`,
  `par_all36m_prot.prm`), including its CMAP grids,
- TIP3P water and the ions of `toppar_water_ions.str`,

from the MacKerell lab's official CHARMM36 repository
(https://github.com/mackerell-lab/charmm36-force-field, `toppar_c36_feb26.tgz`),
which distributes them under the MIT License reproduced in `LICENSE.toppar`.
Every atom type, charge, parameter and CMAP grid of these parts agrees with
that release.

Added for topon, and nothing else:

| Block | Source |
|---|---|
| `PRES DITY` (RTF) | Written for topon. Two TYR joined CE2-CE2, each CE2 retyped `CG2R67` and given the charge of the HE2 it loses (0.00), so both sides keep the CHARMM36m TYR charges. |
| `MASS` and `NONBONDED` of `CG2R67` | Copied from `par_all36_cgenff.prm` (CGenFF v5.0, same release, MIT). |
| 2 bonds, 5 angles and 10 dihedrals with `CG2R67` | Copied from `par_all36_cgenff.prm` with the CGenFF types written as their CHARMM36m equivalents (`CG2R61` as `CA`, `HGR61` as `HP`, `OG311` as `OH1`, `HGP1` as `H`). Each line names its CGenFF source. |
| dihedral `CG2R67 CA CA CT2` (3.1, n 2, 180) | Analogy, marked in the file. CGenFF has no entry with `CG321` there, so it takes the value of `CG2R61 CG2R61 CG2R61 CG321` (and of `CG2R61 CG2R61 CG2R61 CG2R67`). Not fitted. |

The disulfide (`DISU`) and terminal patches are CHARMM36m's own.

Dityrosine terms produced by the CGenFF program for a model compound are not
bundled, because their redistribution terms are not stated. Users with
access to the CGenFF program can make their own stream file and pass it after the bundled
files, as in `topon protein --charmm-files <data>/top_all36m_prot_C2L.rtf
<data>/par_all36m_prot_C2L.prm my_dity.str` with `<data>` this folder. A
later file replaces a patch or term of the same name.

The fix cmap file LAMMPS reads is written from the CMAP section of the PRM
(`topon.forcefield.charmm.write_lammps_cmap`), so it is the same force field.

Please cite, when using them,

- J. Huang, S. Rauscher, G. Nawrocki, T. Ran, M. Feig, B. L. de Groot,
  H. Grubmuller and A. D. MacKerell Jr., CHARMM36m: an improved force field
  for folded and intrinsically disordered proteins, Nat. Methods 14, 71-73
  (2017).
- R. B. Best, X. Zhu, J. Shim, P. E. M. Lopes, J. Mittal, M. Feig and
  A. D. MacKerell Jr., Optimization of the additive CHARMM all-atom protein
  force field targeting improved sampling of the backbone phi, psi and
  side-chain chi1 and chi2 dihedral angles, J. Chem. Theory Comput. 8,
  3257-3273 (2012).
- K. Vanommeslaeghe et al., CHARMM general force field: a force field for
  drug-like molecules compatible with the CHARMM all-atom additive biological
  force fields, J. Comput. Chem. 31, 671-690 (2010) (the CG2R67 terms).
