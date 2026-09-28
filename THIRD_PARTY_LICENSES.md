# Third-party data bundled with topon

topon ships force-field files written by others. Each folder below carries
the license of its files and a README with their source, what was changed,
and what to cite.

| Files | Source | License |
|---|---|---|
| `topon/chemistry/charmm/data/top_all35_ethers.rtf`, `par_all35_ethers.prm` | CHARMM C35r ether force field, `toppar_c36_feb26.tgz` of https://github.com/mackerell-lab/charmm36-force-field, unchanged | MIT (`topon/chemistry/charmm/data/LICENSE.toppar`) |
| `topon/protein_network/charmm/data/top_all36m_prot_C2L.rtf`, `par_all36m_prot_C2L.prm` | CHARMM36m protein force field and TIP3P water and ions from the same release, plus the dityrosine terms copied from its `par_all36_cgenff.prm` and a DITY patch written for topon | MIT (`topon/protein_network/charmm/data/LICENSE.toppar`) |
| `topon/protein_network/data/martini_v3_*.itp` | Martini 3.0.0 force field, https://github.com/marrink-lab/martini-forcefields (pruned and marked where changed) | Apache-2.0 (`topon/protein_network/data/LICENSE.apache-2.0`) |
| `topon/protein_network/data/{nat,high,no}_pro.itp` | Output of polyply (University of Groningen) for the resilin sequences | Apache-2.0 (same file) |
| `topon/forcefield/DreidingX6parameters.txt`, `topon/utils/DreidingX6parameters.txt` | DREIDING parameters (Mayo, Olafson and Goddard, J. Phys. Chem. 94, 8897, 1990) in Cerius2 format | Not recorded |

The PEG junction and end-cap stream of the CHARMM demo
(`demos/polymer/atomistic/charmm_peg/peg_junction.str`) is written
for topon, with its analogy parameters named in the file.
