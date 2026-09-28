"""MARTINI 3 protein-network generator for topon.

Sequence-driven coarse-grained polymer-network builder, peer to topon.simbox
and topon.singlechain. Has the same two-stage shape as the CHARMM atomistic
builder in `charmm/`: BFM lattice topology -> JSON -> sequence + MARTINI FF + water -> LAMMPS.
"""
