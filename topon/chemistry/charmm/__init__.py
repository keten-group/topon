"""CHARMM chemistry for polymer networks (``chemistry.force_field = "charmm"``).

The atomistic route builds the network as one RDKit molecule. With CHARMM,
every atom takes its type and charge from the RTF residue its monomer or
node molecule names (:func:`type_network`), and every bonded and nonbonded
term comes from the parameter files through the shared CHARMM reader
(:mod:`topon.forcefield.charmm`), which never substitutes a missing term.

Bundled data: the C35r ether force field (``data/``, MIT, see its README).
"""
from __future__ import annotations

from topon.forcefield.charmm import CharmmTerms, MissingCharmmParameters, parameterize

from .assign import (
    CharmmTyping,
    CharmmTypingError,
    ResidueInstance,
    bundled_data_dir,
    load_parameters,
    resolve_files,
    type_network,
)


def build_terms(mol_h, typing: CharmmTyping, ps) -> CharmmTerms:
    """Every CHARMM term of the typed network (raises on any missing one)."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_h.GetBonds()]
    return parameterize(ps, typing.atom_type, bonds, typing.impropers,
                        context="the polymer network")


__all__ = [
    "CharmmTerms",
    "CharmmTyping",
    "CharmmTypingError",
    "MissingCharmmParameters",
    "ResidueInstance",
    "build_terms",
    "bundled_data_dir",
    "load_parameters",
    "resolve_files",
    "type_network",
]
