# Protein-network demos

Protein networks built from an amino-acid sequence with `topon protein` (the Python function is
`topon.protein_network.network.build_protein_network`). Both demos are the resilin repeat `GGRPSDSYGAPGGGN`
crosslinked through its tyrosines (dityrosine), in the two models topon writes.

- [`charmm/`](charmm/) is CHARMM36m all-atom (8 chains of 12 repeats, dry, and a 35 wt% water build from `run.py`).
- [`martini/`](martini/) is Martini 3 with the Martini3-IDP bonded terms (8 chains of 18 repeats, dry).

Each folder has a `config.json` that `topon protein --config` reads, a `run.py`, and an `expected_output/` with the
small text files of the build and the LAMMPS logs of its three stages. Any other sequence, crosslink residue (`Y` or
`C`) or crosslink method works the same way (see [`docs/USAGE.md`](../../docs/USAGE.md)).
