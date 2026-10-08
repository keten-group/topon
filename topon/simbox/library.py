"""
Pre-defined molecule library for simbox.

Provides ``MoleculeLibrary`` with builder methods for the three target
chemistries used in epoxy-PDMS / amino-PDMS / POSS crosslink experiments.
Each builder constructs the molecule programmatically with RDKit's RWMol
so that every atom is precisely controlled and reactive sites are tagged
automatically.
"""

from __future__ import annotations

from typing import Optional

from topon.simbox.molecule import Molecule, embed_conformer


class MoleculeLibrary:
    """Factory for pre-defined siloxane / POSS molecules.

    Usage::

        lib = MoleculeLibrary()
        epoxy = lib.epoxy_pdms(n_dms=2)      # average MW 510.9 g/mol
        amino = lib.amino_pdms(n_dms=8)      # average MW 841.8 g/mol
        poss  = lib.am0270_poss()             # ~1267 g/mol
    """

    # ------------------------------------------------------------------
    # Public builders
    # ------------------------------------------------------------------
    def epoxy_pdms(self, n_dms: int = 2) -> Molecule:
        """Glycidoxypropyl-terminated PDMS.

        Structure::

            Epoxide-CH2-O-CH2CH2CH2-Si(Me)2-[O-Si(Me)2]_n-O-Si(Me)2-CH2CH2CH2-O-CH2-Epoxide

        Parameters
        ----------
        n_dms : int
            Number of internal dimethylsiloxane repeat units (default 2
            gives C20H46O7Si4, 77 atoms, average MW 510.9 g/mol; the
            collaborators' epoxy-PDMS is about 500).
        """
        mol = self._build_functional_pdms(
            n_dms=n_dms,
            end_group_builder=self._attach_glycidoxypropyl,
        )
        return Molecule.from_mol(f"Epoxy-PDMS-n{n_dms}", mol)

    def amino_pdms(self, n_dms: int = 8) -> Molecule:
        """Aminopropyl-terminated PDMS.

        Structure::

            H2N-CH2CH2CH2-Si(Me)2-[O-Si(Me)2]_n-O-Si(Me)2-CH2CH2CH2-NH2

        Parameters
        ----------
        n_dms : int
            Number of internal dimethylsiloxane repeat units (default 8
            gives C26H76N2O9Si10, 123 atoms, average MW 841.8 g/mol; the
            collaborators' amino-PDMS is about 850).
        """
        mol = self._build_functional_pdms(
            n_dms=n_dms,
            end_group_builder=self._attach_aminopropyl,
        )
        return Molecule.from_mol(f"Amino-PDMS-n{n_dms}", mol)

    def am0270_poss(self) -> Molecule:
        """AminopropylIsooctyl POSS (AM0270, MW ~1267 g/mol).

        Structure:
        - Si8O12 cage (cube topology)
        - Corner 0: aminopropyl arm  (-CH2CH2CH2-NH2)
        - Corners 1-7: isooctyl arms (2,4,4-trimethylpentyl)

        The terminal amine nitrogen is tagged as the single reactive site.
        """
        mol = self._build_am0270()
        return Molecule.from_mol("AM0270-POSS", mol)

    def custom(
        self,
        smiles: str,
        name: str = "Custom",
        reactive_smarts: Optional[dict[str, str]] = None,
    ) -> Molecule:
        """Escape hatch – build any molecule from a SMILES string."""
        return Molecule.from_smiles(name, smiles, reactive_smarts=reactive_smarts)

    # ------------------------------------------------------------------
    # Internal: functional PDMS chain builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_functional_pdms(n_dms: int, end_group_builder) -> "Chem.Mol":
        """Build a PDMS backbone and attach end groups.

        The backbone has ``n_dms + 2`` silicon atoms: two terminal Si, each
        carrying two methyls and the functional group, and ``n_dms`` internal
        Si, each carrying two methyls. Adjacent Si atoms are bridged by
        oxygen, so every Si has four heavy-atom neighbours and no H.

        Before 0.4.5 a terminal Si carried one methyl, and RDKit's ``AddHs``
        filled its fourth valence with an H: every chain end was
        -SiH(Me)- where aminopropyl- and glycidoxypropyl-terminated PDMS
        end in -Si(Me)2-: each molecule had an H in place of a methyl at
        both ends, 6 atoms and 28.05 g/mol (two CH2) short.

        Parameters
        ----------
        n_dms : int
            Internal DMS repeat units.
        end_group_builder : callable
            ``f(rwmol, si_atom_idx)`` – attaches the functional group.
        """
        from rdkit import Chem

        rwmol = Chem.RWMol()
        n_si = n_dms + 2
        si_indices: list[int] = []

        # --- Si-O backbone ---
        for i in range(n_si):
            si_idx = rwmol.AddAtom(Chem.Atom(14))  # Si
            si_indices.append(si_idx)

            if i > 0:
                o_idx = rwmol.AddAtom(Chem.Atom(8))  # bridging O
                rwmol.AddBond(si_indices[i - 1], o_idx, Chem.BondType.SINGLE)
                rwmol.AddBond(o_idx, si_idx, Chem.BondType.SINGLE)

        # --- Methyl substituents ---
        # Two on every Si: an internal Si's other two neighbours are bridging
        # O, a terminal Si's are one bridging O and the end group.
        for si_idx in si_indices:
            for _ in range(2):
                c = rwmol.AddAtom(Chem.Atom(6))
                rwmol.AddBond(si_idx, c, Chem.BondType.SINGLE)

        # --- End groups ---
        end_group_builder(rwmol, si_indices[0])
        end_group_builder(rwmol, si_indices[-1])

        # --- Finalise ---
        mol = rwmol.GetMol()

        # Sanitize before adding Hs (required for RWMol-built molecules)
        Chem.SanitizeMol(mol)

        mol = Chem.AddHs(mol)
        embed_conformer(mol, f"functional PDMS (n_dms={n_dms})", max_iters=500)

        return mol

    # ------------------------------------------------------------------
    # End-group attachment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _attach_glycidoxypropyl(rwmol, si_idx: int) -> None:
        """Attach ``-CH2CH2CH2-O-CH2-CH(-O-)CH2`` (glycidoxypropyl) to *si_idx*.

        The glycidoxypropyl group is a propyl linker connected through an
        ether oxygen to a glycidyl (epoxide / oxirane) ring::

              O
             / \\
        Si-CH2CH2CH2-O-CH2-CH--CH2
        """
        from rdkit import Chem

        # Propyl linker: 3 carbons
        c1 = rwmol.AddAtom(Chem.Atom(6))
        c2 = rwmol.AddAtom(Chem.Atom(6))
        c3 = rwmol.AddAtom(Chem.Atom(6))
        rwmol.AddBond(si_idx, c1, Chem.BondType.SINGLE)
        rwmol.AddBond(c1, c2, Chem.BondType.SINGLE)
        rwmol.AddBond(c2, c3, Chem.BondType.SINGLE)

        # Ether oxygen
        o_ether = rwmol.AddAtom(Chem.Atom(8))
        rwmol.AddBond(c3, o_ether, Chem.BondType.SINGLE)

        # Glycidyl (epoxide) group: CH2-CH-CH2 with O bridging CH and CH2
        c4 = rwmol.AddAtom(Chem.Atom(6))   # -CH2-
        c5 = rwmol.AddAtom(Chem.Atom(6))   # -CH< (part of ring)
        c6 = rwmol.AddAtom(Chem.Atom(6))   # -CH2 (part of ring)
        o_ring = rwmol.AddAtom(Chem.Atom(8))  # epoxide oxygen

        rwmol.AddBond(o_ether, c4, Chem.BondType.SINGLE)
        rwmol.AddBond(c4, c5, Chem.BondType.SINGLE)
        rwmol.AddBond(c5, c6, Chem.BondType.SINGLE)
        # Close the epoxide ring
        rwmol.AddBond(c5, o_ring, Chem.BondType.SINGLE)
        rwmol.AddBond(c6, o_ring, Chem.BondType.SINGLE)

    @staticmethod
    def _attach_aminopropyl(rwmol, si_idx: int) -> None:
        """Attach ``-CH2CH2CH2-NH2`` (aminopropyl) to *si_idx*."""
        from rdkit import Chem

        c1 = rwmol.AddAtom(Chem.Atom(6))
        c2 = rwmol.AddAtom(Chem.Atom(6))
        c3 = rwmol.AddAtom(Chem.Atom(6))
        n = rwmol.AddAtom(Chem.Atom(7))  # nitrogen

        rwmol.AddBond(si_idx, c1, Chem.BondType.SINGLE)
        rwmol.AddBond(c1, c2, Chem.BondType.SINGLE)
        rwmol.AddBond(c2, c3, Chem.BondType.SINGLE)
        rwmol.AddBond(c3, n, Chem.BondType.SINGLE)

    # ------------------------------------------------------------------
    # AM0270 POSS builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_am0270() -> "Chem.Mol":
        """Build AM0270 POSS (AminopropylIsooctyl POSS) as a standalone molecule.

        Adapted from ``topon.chemistry.builder.ChemistryBuilder._place_poss_am0270``
        but without any graph dependency.

        Topology
        --------
        Si8O12 cage (cube edges → oxygen bridges) with:
        - Corner 0: aminopropyl   (-CH2CH2CH2-NH2)
        - Corners 1-7: isooctyl   (2,4,4-trimethylpentyl  CC(C)CC(C)(C)C)
        """
        from rdkit import Chem

        rwmol = Chem.RWMol()

        # ------ 1. Si8O12 cage (cube topology) ------
        corner_si = [rwmol.AddAtom(Chem.Atom(14)) for _ in range(8)]

        # 12 edges of a cube → 12 bridging oxygens
        cube_edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]
        for a, b in cube_edges:
            o = rwmol.AddAtom(Chem.Atom(8))
            rwmol.AddBond(corner_si[a], o, Chem.BondType.SINGLE)
            rwmol.AddBond(o, corner_si[b], Chem.BondType.SINGLE)

        # ------ 2. Corner 0 → aminopropyl ------
        # -CH2CH2CH2-NH2
        prev = corner_si[0]
        for _ in range(3):
            c = rwmol.AddAtom(Chem.Atom(6))
            rwmol.AddBond(prev, c, Chem.BondType.SINGLE)
            prev = c
        n = rwmol.AddAtom(Chem.Atom(7))
        rwmol.AddBond(prev, n, Chem.BondType.SINGLE)

        # ------ 3. Corners 1-7 → isooctyl ------
        isooctyl_smiles = "CC(C)CC(C)(C)C"  # 2,4,4-trimethylpentyl
        isooctyl_template = Chem.MolFromSmiles(isooctyl_smiles)
        isooctyl_template = Chem.RemoveHs(isooctyl_template)

        for corner_idx in range(1, 8):
            iso_map: dict[int, int] = {}
            for atom in isooctyl_template.GetAtoms():
                new_idx = rwmol.AddAtom(atom)
                iso_map[atom.GetIdx()] = new_idx
            for bond in isooctyl_template.GetBonds():
                rwmol.AddBond(
                    iso_map[bond.GetBeginAtomIdx()],
                    iso_map[bond.GetEndAtomIdx()],
                    bond.GetBondType(),
                )
            # Connect first C of isooctyl to corner Si
            rwmol.AddBond(corner_si[corner_idx], iso_map[0], Chem.BondType.SINGLE)

        # ------ 4. Finalise ------
        mol = rwmol.GetMol()

        # Sanitize before adding Hs (required for RWMol-built molecules)
        Chem.SanitizeMol(mol)

        mol = Chem.AddHs(mol)
        # Not from random starting coordinates: those tangle the cage in 18
        # of seeds 0-29, and seed 42 put corner 1 into the cage with its Si-C
        # bond out through face 0-2-6-4 at 2.43 A, in every POSS simbox built
        # before 0.4.5. MMFF cannot undo it.
        embed_conformer(mol, "AM0270 POSS", max_iters=1000)

        return mol
