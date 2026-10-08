"""
Molecule definition and 3D conformer generation for simbox.

Provides the Molecule class with factory methods for creating molecules
from SMILES strings, PDB files, or existing RDKit mol objects.  Reactive
sites (epoxides, amines, etc.) are auto-detected via SMARTS patterns so
that downstream packing and writing stages know which atoms participate
in crosslinking reactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# SMARTS patterns for reactive-site auto-detection
# ---------------------------------------------------------------------------
REACTIVE_SMARTS: dict[str, str] = {
    "epoxide": "[C]1[O][C]1",                  # oxirane ring
    "primary_amine": "[NX3;H2;!$([NH2]C=O)]",  # -NH2 (not amide)
    "secondary_amine": "[NX3;H1]([#6])[#6]",   # >NH between two carbons
}

# ---------------------------------------------------------------------------
# Conformer embedding and its check
# ---------------------------------------------------------------------------
#: Seeds tried, from the requested one upward, before a molecule is refused.
EMBED_ATTEMPTS = 10

#: Largest ring the bond-through-ring reading looks at. A T8 POSS face is an
#: 8-ring (Si4O4) and a T12 face a 10-ring.
MAX_CHECKED_RING = 12

#: A bond this far (as a fraction) from the r0 of the force field that
#: cleaned the conformer is refused. Sound MMFF conformers stay within 6.4 %
#: (142 untangled AM0270 conformers over seeds; the two library PDMS within
#: 3.2 %, and within 4.6 % since their chain ends carry two methyls);
#: every tangled cage has a bond 22 % out or more.
MAX_BOND_STRAIN = 0.15


def _segment_crosses_triangle(p0, p1, a, b, c) -> bool:
    """Moller-Trumbore: does the segment p0-p1 cross the triangle abc."""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = e1 @ h
    if abs(det) < 1e-12:
        return False
    s = p0 - a
    u = (s @ h) / det
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, e1)
    v = (d @ q) / det
    if v < 0.0 or u + v > 1.0:
        return False
    t = (e2 @ q) / det
    return 0.0 <= t <= 1.0


def segment_through_ring(p0, p1, ring) -> bool:
    """Does the segment p0-p1 pass through the ring whose atom positions are *ring* (in ring order).

    The ring is read as the fan of triangles from its centroid, which for a
    ring of up to a dozen atoms is the surface it spans.
    """
    ring = np.asarray(ring, dtype=float)
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    centre = ring.mean(axis=0)
    reach = np.linalg.norm(ring - centre, axis=1).max()
    if np.linalg.norm(0.5 * (p0 + p1) - centre) > reach + np.linalg.norm(p1 - p0):
        return False
    n = len(ring)
    return any(_segment_crosses_triangle(p0, p1, centre, ring[k], ring[(k + 1) % n])
               for k in range(n))


def _bond_r0_table(mol) -> dict:
    """{(i, j): r0} for every bond: MMFF94 when it types the whole molecule, else UFF."""
    from rdkit.Chem import ChemicalForceFields
    from rdkit.Chem import rdForceFieldHelpers

    props = None
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    table = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        params = props.GetMMFFBondStretchParams(mol, i, j) if props is not None else None
        if params:
            table[(i, j)] = params[2]
            continue
        params = ChemicalForceFields.GetUFFBondStretchParams(mol, i, j)
        if params:
            table[(i, j)] = params[1]
    return table


def conformer_defects(mol, conf_id: int = -1,
                      max_strain: float = MAX_BOND_STRAIN) -> list[str]:
    """What is wrong with a conformer that should stop it being packed.

    Returns one readable line per defect, empty when there is none. Two
    readings:

    - A bond passing through a ring of the same molecule, neither of its
      atoms in the ring (:func:`segment_through_ring`), for every ring of
      the symmetrised SSSR up to ``MAX_CHECKED_RING`` atoms. No minimiser
      takes a bond back out through a ring. Embedding the AM0270 POSS from
      random starting coordinates tangled the cage in 18 of seeds 0-29 and
      at seed 42 (corner 1 pushed into the cage, its Si-C bond out through
      the opposite face), and MMFF converges inside the tangle.
    - A bond more than *max_strain* off its r0 in MMFF94, or in UFF when
      MMFF lacks a parameter: the strain a tangle leaves, or a cleanup that
      stopped far from converged.
    """
    from rdkit import Chem

    X = mol.GetConformer(conf_id).GetPositions()
    sym = lambda k: f"{mol.GetAtomWithIdx(k).GetSymbol()}{k}"
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    defects: list[str] = []

    for ring in Chem.GetSymmSSSR(mol):
        ring = list(ring)
        if len(ring) > MAX_CHECKED_RING:
            continue
        members = set(ring)
        for i, j in bonds:
            if i in members or j in members:
                continue
            if segment_through_ring(X[i], X[j], X[ring]):
                defects.append(
                    f"bond {sym(i)}-{sym(j)} passes through the "
                    f"{len(ring)}-ring {'-'.join(sym(a) for a in ring)}")

    for (i, j), r0 in _bond_r0_table(mol).items():
        r = float(np.linalg.norm(X[i] - X[j]))
        if abs(r / r0 - 1.0) > max_strain:
            defects.append(f"bond {sym(i)}-{sym(j)} is {r:.3f} A against "
                           f"r0 {r0:.3f} A ({100 * (r / r0 - 1):+.0f} %)")
    return defects


def embed_conformer(mol, name: str, seed: int = 42, max_iters: int = 500) -> int:
    """Give *mol* (explicit H, no conformer) a checked 3D conformer, in place.

    Each attempt embeds with ETKDGv3 from its own starting coordinates,
    falling back to random ones only when that embedding fails, and
    optimises with MMFF (UFF if MMFF raises; a molecule MMFF cannot type
    is left as embedded, since ``MMFFOptimizeMolecule`` then returns
    without raising, as it always has here). A conformer
    :func:`conformer_defects` reports is discarded and the next seed tried,
    up to ``EMBED_ATTEMPTS`` seeds, so a molecule whose seed-*seed*
    conformer is sound gets exactly the conformer this call has always
    given it. Returns the seed used. Raises ``RuntimeError`` at once when
    the embedding itself fails, as before, and naming the last defects
    when no seed gives a sound conformer.
    """
    from rdkit.Chem import AllChem

    last: list[str] = []
    for attempt in range(EMBED_ATTEMPTS):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + attempt
        if AllChem.EmbedMolecule(mol, params) == -1:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) == -1:
                raise RuntimeError(f"Failed to generate 3D conformer for '{name}'")
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        last = conformer_defects(mol)
        if not last:
            if attempt:
                print(f"[simbox] {name}: conformer from seed {seed + attempt} "
                      f"(seeds {seed}-{seed + attempt - 1} gave defects)")
            return seed + attempt
        mol.RemoveAllConformers()
    raise RuntimeError(
        f"Failed to generate 3D conformer for '{name}': seeds {seed}-"
        f"{seed + EMBED_ATTEMPTS - 1} all gave defects; the last: "
        f"{'; '.join(last[:3])}")


@dataclass
class Molecule:
    """
    A molecule with a 3D conformer and reactive-site annotations.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"Epoxy-PDMS"``).
    mol : rdkit.Chem.rdchem.Mol
        RDKit Mol with explicit H and an embedded 3D conformer.
    mw : float
        Exact molecular weight (g mol-1).
    smiles : str | None
        Canonical SMILES, if the molecule was built from one.
    reactive_sites : dict[str, list[int]]
        Mapping from reactive-group name to a list of heavy-atom indices
        that belong to that group (e.g. ``{"epoxide": [12, 14]}``).
    """

    name: str
    mol: object                        # RDKit Mol (lazy import avoids top-level dep)
    mw: float
    smiles: Optional[str] = None
    reactive_sites: dict[str, list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_smiles(
        cls,
        name: str,
        smiles: str,
        reactive_smarts: Optional[dict[str, str]] = None,
    ) -> Molecule:
        """Build a *Molecule* from a SMILES string.

        The method adds explicit H, embeds a 3D conformer (ETKDGv3),
        and optimises geometry with MMFF (falling back to UFF), through
        :func:`embed_conformer`, which refuses a tangled conformer.

        Parameters
        ----------
        name : str
            Identifier for this molecule.
        smiles : str
            Valid SMILES string.
        reactive_smarts : dict, optional
            Extra / override SMARTS patterns for reactive-site detection.
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        embed_conformer(mol, name, max_iters=500)

        mw = Descriptors.ExactMolWt(mol)
        sites = cls._detect_reactive_sites(mol, reactive_smarts)

        return cls(
            name=name, mol=mol, mw=mw,
            smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
            reactive_sites=sites,
        )

    @classmethod
    def from_pdb(
        cls,
        name: str,
        pdb_path: str,
        reactive_sites: Optional[dict[str, list[int]]] = None,
        reactive_smarts: Optional[dict[str, str]] = None,
    ) -> Molecule:
        """Build a *Molecule* from a PDB file.

        Parameters
        ----------
        name : str
            Identifier for this molecule.
        pdb_path : str
            Path to a .pdb file.
        reactive_sites : dict, optional
            Manually specified reactive sites (skip auto-detection).
        reactive_smarts : dict, optional
            Extra SMARTS patterns for auto-detection.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse PDB file: {pdb_path}")

        # Add explicit H if none are present
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)

        mw = Descriptors.ExactMolWt(mol)
        sites = reactive_sites or cls._detect_reactive_sites(mol, reactive_smarts)

        return cls(name=name, mol=mol, mw=mw, reactive_sites=sites)

    @classmethod
    def from_mol(
        cls,
        name: str,
        rdkit_mol,
        reactive_sites: Optional[dict[str, list[int]]] = None,
        reactive_smarts: Optional[dict[str, str]] = None,
    ) -> Molecule:
        """Wrap an existing RDKit Mol object.

        The molecule *must* already have explicit H and a 3D conformer.
        """
        from rdkit.Chem import Descriptors

        mw = Descriptors.ExactMolWt(rdkit_mol)
        sites = reactive_sites or cls._detect_reactive_sites(
            rdkit_mol, reactive_smarts
        )
        return cls(name=name, mol=rdkit_mol, mw=mw, reactive_sites=sites)

    # ------------------------------------------------------------------
    # Reactive-site detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_reactive_sites(
        mol, custom_smarts: Optional[dict[str, str]] = None
    ) -> dict[str, list[int]]:
        """Return reactive sites found via SMARTS sub-structure matching."""
        from rdkit import Chem

        patterns: dict[str, str] = dict(REACTIVE_SMARTS)
        if custom_smarts:
            patterns.update(custom_smarts)

        sites: dict[str, list[int]] = {}
        for group_name, smarts in patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            matches = mol.GetSubstructMatches(pat)
            if matches:
                atom_indices = sorted(set(idx for match in matches for idx in match))
                sites[group_name] = atom_indices
        return sites

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def get_coordinates(self) -> np.ndarray:
        """Return atom positions as an (N, 3) array (Angstrom)."""
        conf = self.mol.GetConformer()
        return np.array(
            [list(conf.GetAtomPosition(i)) for i in range(self.mol.GetNumAtoms())]
        )

    def get_centroid(self) -> np.ndarray:
        """Centroid of the molecule (Angstrom)."""
        return self.get_coordinates().mean(axis=0)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def num_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def num_bonds(self) -> int:
        return self.mol.GetNumBonds()

    def __repr__(self) -> str:
        sites = list(self.reactive_sites.keys())
        return (
            f"Molecule(name='{self.name}', atoms={self.num_atoms}, "
            f"mw={self.mw:.1f}, reactive_sites={sites})"
        )
