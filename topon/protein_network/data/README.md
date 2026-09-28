# Bundled Martini 3 files

These files come from the Martini 3 force field and from polyply, both
distributed under the Apache License 2.0 (reproduced in `LICENSE.apache-2.0`).

| File | Source | Changes |
|---|---|---|
| `martini_v3_protein.itp` | `martini_v3.0.0.itp` of the Martini 3.0.0 release (https://github.com/marrink-lab/martini-forcefields, `martini_forcefields/regular/v3.0.0/gmx_files/`) | Pruned to the 33 bead types a protein network can use (the `[ atomtypes ]` and `[ nonbond_params ]` rows of those types). No value changed. |
| `martini_v3_water.itp` | `martini_v3.0.0_solvents_v1.itp`, same release | None. |
| `martini_v3_ions.itp` | `martini_v3.0.0_ions_v1.itp`, same release | The residue name of NA and of CL is ION (noted at the top of the file). |
| `nat_pro.itp`, `high_pro.itp`, `no_pro.itp` | `polyply gen_params -lib martini3` (polyply, University of Groningen) for the resilin repeat `GGRPSDSYGAPGGGN` and its high-proline (`GPRPSDSYGAPGPGN`) and no-proline (`GGRGSDSYGAGGGGN`) variants (270 residues per chain) | The command on line 1 no longer names local paths. |

The chain topologies use the Martini3-IDP bonded terms of polyply's library.
For any other sequence `topon protein` runs polyply itself (`pip install
topon[martini]`).

Please cite, when using them,

- P. C. T. Souza et al., Martini 3: a general purpose force field for
  coarse-grained molecular dynamics, Nat. Methods 18, 382-388 (2021),
  doi:10.1038/s41592-021-01098-3.
- F. Grunewald, R. Alessandri, P. C. Kroon, L. Monticelli, P. C. T. Souza and
  S. J. Marrink, Nat. Commun. (2022), doi:10.1038/s41467-021-27627-4
  (polyply).
- L. Wang, C. Brasnett, L. Borges-Araujo, P. C. T. Souza and S. J. Marrink,
  Nat. Commun. (2025), doi:10.1038/s41467-025-58199-2 (Martini3-IDP).

The polyply ITP headers list the further papers polyply asks to cite.
