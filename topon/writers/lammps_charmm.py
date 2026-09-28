"""LAMMPS data and coefficient files for a CHARMM-typed polymer network.

The CHARMM counterpart of :class:`topon.writers.lammps_atomistic.DreidingWriter`.
It formats what :mod:`topon.chemistry.charmm` computed and computes nothing
itself. The data file has the layout the conformation stage expects from the
DREIDING writer (one molecule, atoms in RDKit order, zero coordinates, a
+-1000 A box); the coordinates come from the displacement files.

Three coefficient includes sit beside the data file:

``<base>.in.settings``       CHARMM pair (with 1-4 columns and NBFIX),
                             bonds, angles with Urey-Bradley, dihedrals with
                             their 1-4 weights, impropers
``<base>.in.settings.soft``  the bonded terms with 1-4 weights 0 and no pair
                             coefficients, for ``pair_style soft`` stages
``<base>.in.settings.lj``    the LJ pair coefficients for the
                             ``lj/cut/coul/long`` epsilon ramp
"""
from __future__ import annotations

import os

from topon.forcefield.charmm import CharmmTerms, write_lj_pair_coeffs, write_settings


class CharmmWriter:
    """``positions`` (atom index -> xyz) and ``box`` ((lo, hi) per axis) are
    optional; without them the atoms sit at the origin in a +-1000 A box and
    the conformation stage places them, as for DREIDING."""

    def __init__(self, mol, typing, terms: CharmmTerms, output_file: str,
                 positions=None, box=None):
        self.mol = mol
        self.typing = typing
        self.terms = terms
        self.output_file = output_file
        self.positions = positions
        self.box = box or ((-1000.0, 1000.0),) * 3
        stem = os.path.splitext(output_file)[0]
        self.settings_file = f"{stem}.in.settings"
        self.soft_settings_file = f"{stem}.in.settings.soft"
        self.lj_settings_file = f"{stem}.in.settings.lj"

    def write(self) -> None:
        print(f"Writing LAMMPS Data File (CHARMM): {self.output_file}")
        self._write_data()
        write_settings(self.settings_file, self.terms, title="CHARMM")
        write_settings(self.soft_settings_file, self.terms, soft=True, title="CHARMM")
        write_lj_pair_coeffs(self.lj_settings_file, self.terms)
        print(f"Writing LAMMPS Settings Files: {os.path.basename(self.settings_file)} "
              f"(+ .soft, .lj)")

    def _write_data(self) -> None:
        t, ty = self.terms, self.typing
        n = self.mol.GetNumAtoms()
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("LAMMPS data file (CHARMM, topon)\n\n")
            f.write(f"{n} atoms\n{len(t.bonds)} bonds\n{len(t.angles)} angles\n"
                    f"{len(t.dihedrals)} dihedrals\n{len(t.impropers)} impropers\n\n")
            f.write(f"{len(t.atom_types)} atom types\n{len(t.bond_types)} bond types\n"
                    f"{len(t.angle_types)} angle types\n"
                    f"{len(t.dihedral_types)} dihedral types\n"
                    f"{len(t.improper_types)} improper types\n\n")
            for (lo, hi), ax in zip(self.box, "xyz"):
                f.write(f"{lo} {hi} {ax}lo {ax}hi\n")
            f.write("\nMasses\n\n")
            for atype, tid in sorted(t.atom_types.items(), key=lambda x: x[1]):
                f.write(f"{tid} {t.masses[tid]:.4f} # {atype}\n")
            f.write("\nAtoms # full\n\n")
            for idx in range(n):
                res = ty.residues[ty.residue_of[idx]].resname
                if self.positions is None:
                    xyz = "0.0 0.0 0.0"
                else:
                    x, y, z = self.positions[idx]
                    xyz = f"{x:.6f} {y:.6f} {z:.6f}"
                f.write(f"{idx + 1} 1 {t.atom_types[ty.atom_type[idx]]} "
                        f"{ty.charge[idx]:.6f} {xyz} # {ty.atom_type[idx]} "
                        f"{res}:{ty.atom_name[idx]}\n")
            for title, rows in (("Bonds", t.bonds), ("Angles", t.angles),
                                ("Dihedrals", t.dihedrals), ("Impropers", t.impropers)):
                if not rows:
                    continue
                f.write(f"\n{title}\n\n")
                for k, r in enumerate(rows, 1):
                    f.write(f"{k} {r[0]} " + " ".join(str(a + 1) for a in r[1:]) + "\n")
            f.write("\n")
