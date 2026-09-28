"""CHARMM36m force field for the atomistic protein builder.

A thin view over :class:`topon.forcefield.charmm.CharmmParameterSet`, the one
CHARMM reader in topon. It keeps the dictionary layout the builder was
written against (``residues[name]["atoms"]``, ``bonds_prm`` and so on) and the
``lookup_*`` methods, which now follow CHARMM's own matching rules.

TIP3P and ion masses are guaranteed even when an RTF leaves them out, since
the solvation step places those types whatever the files say.
"""
from __future__ import annotations

from topon.forcefield.charmm import CharmmParameterSet


def _residue_dict(res) -> dict:
    return {
        "atoms": dict(res.atoms),
        "bonds": list(res.bonds),
        "impropers": list(res.impropers),
        "cmaps": list(res.cmaps),
        "deletes": list(res.deletes),
        "ics": list(res.ics),
        "charge": res.charge,
    }


class CHARMMForceField:
    """CHARMM RTF + PRM files, read once, looked up the CHARMM way.

    ``CHARMMForceField(prm_path, rtf_path, *extra)`` keeps the historic
    argument order; any further files (e.g. a stream file with extra
    residues or parameters) are read after them, in order.
    """

    def __init__(self, prm_path, rtf_path, *extra_paths):
        self.params = CharmmParameterSet.from_files(rtf_path, prm_path, *extra_paths)
        ps = self.params
        for t, m in (("OT", 15.9994), ("HT", 1.0080), ("SOD", 22.9898), ("CLA", 35.4500)):
            ps.masses.setdefault(t, m)
        self.masses = ps.masses
        self.bonds_prm = ps.bonds
        self.angles_prm = ps.angles
        self.dihedrals_prm = ps.dihedrals
        self.impropers_prm = ps.impropers
        self.vdw_prm = {t: v[:2] for t, v in ps.nonbonded.items()}
        self.residues = {n: _residue_dict(r) for n, r in ps.residues.items()}
        self.patches = {n: _residue_dict(r) for n, r in ps.patches.items()}

    def lookup_bond(self, t1, t2):
        return self.params.lookup_bond(t1, t2)

    def lookup_angle(self, t1, t2, t3):
        return self.params.lookup_angle(t1, t2, t3)

    def lookup_dihedral(self, t1, t2, t3, t4):
        """Terms ``[(K, n, delta), ...]``: exact types first, then ``X B C X``."""
        return self.params.lookup_dihedral(t1, t2, t3, t4)

    def lookup_improper(self, t1, t2, t3, t4):
        """``(K, psi0)``: ``A B C D``, ``A X X D``, ``X B C D``, ``X X C D``."""
        return self.params.lookup_improper(t1, t2, t3, t4)
