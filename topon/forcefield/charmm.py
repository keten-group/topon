"""CHARMM topology (RTF) and parameter (PRM) files, read with CHARMM's rules.

One reader for every CHARMM route in topon: the atomistic protein builder
(``topon.protein_network.charmm``) and the polymer-network route
(``chemistry.force_field = "charmm"``). It reads ``.rtf``, ``.prm`` and ``.str``
stream files (the format CGenFF writes), any number of them, in order.

Two parts:

* :class:`CharmmParameterSet` holds what the files define and looks terms up
  the way CHARMM does. Bonds and angles match forward or reversed, never with
  wildcards. Proper dihedrals match all four types first and ``X B C X``
  second. Impropers match ``A B C D``, then ``A X X D``, then ``X B C D``,
  then ``X X C D``, each forward before reversed. A wildcard is used only when
  the files define it and no exact term exists, which is CHARMM's precedence.
* :func:`parameterize` takes typed atoms, bonds and impropers, generates the
  angles and dihedrals, looks every term up and returns LAMMPS type tables.
  Nothing is ever substituted: any term the files do not define raises
  :class:`MissingCharmmParameters` with the full list.

The 1-4 interactions follow CHARMM (``nbxmod 5``, ``e14fac 1.0``). LAMMPS
computes them inside ``dihedral_style charmm`` / ``charmmfsw`` with the
``lj/charmm*`` pair style's 1-4 parameters, weighted by the fourth
``dihedral_coeff``. The weight is 1 for a pair reached by one dihedral, 1/n
for a pair reached by n (0.5 in six-membered rings), 0 when the pair is also
1-2 or 1-3 (four- and five-membered rings), and it rides only on the first
Fourier term of a multi-term dihedral, so every 1-4 pair is counted once.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

#: Masses of the elements CHARMM files use, for types whose MASS line gives
#: no element symbol.
_ELEMENT_MASSES = {
    "H": 1.008, "HE": 4.0026, "LI": 6.941, "B": 10.811, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180, "NA": 22.990,
    "MG": 24.305, "AL": 26.982, "SI": 28.086, "P": 30.974, "S": 32.06,
    "CL": 35.45, "K": 39.098, "AR": 39.948, "CA": 40.078, "MN": 54.938,
    "FE": 55.845, "CO": 58.933, "NI": 58.693, "CU": 63.546, "ZN": 65.38,
    "SE": 78.971, "BR": 79.904, "RB": 85.468, "CD": 112.41, "I": 126.90,
    "CS": 132.91, "BA": 137.33,
}

#: Section keywords of a parameter file, by the spellings CHARMM accepts
#: (the full word or its first four letters).
_SECTIONS_PRM = {}
for _sect, _words in {
    "ATOMS": ("ATOMS",), "BONDS": ("BONDS",), "ANGLES": ("ANGLES", "THETAS"),
    "DIHEDRALS": ("DIHEDRALS", "PHI"), "IMPROPER": ("IMPROPERS", "IMPHI"),
    "CMAP": ("CMAP",), "NONBONDED": ("NONBONDED", "NBONDED"),
    "NBFIX": ("NBFIX",), "HBOND": ("HBONDS",), "END": ("END",),
}.items():
    for _w in _words:
        for _n in range(min(4, len(_w)), len(_w) + 1):
            _SECTIONS_PRM[_w[:_n]] = _sect


class CharmmFormatError(ValueError):
    """A CHARMM file line that cannot be read as what its section says."""


class MissingCharmmParameters(KeyError):
    """Terms the parameter files do not define. Never filled in by topon."""

    def __init__(self, missing: dict[str, list[tuple[str, ...]]], context: str = ""):
        self.missing = {k: sorted(set(v)) for k, v in missing.items() if v}
        lines = []
        for kind, keys in self.missing.items():
            lines.append(f"  {kind} ({len(keys)}):")
            lines.extend(f"    {' '.join(k)}" for k in keys[:60])
            if len(keys) > 60:
                lines.append(f"    ... and {len(keys) - 60} more")
        head = "Missing CHARMM parameters"
        if context:
            head += f" for {context}"
        msg = (head + ". Add them to a parameter file (e.g. by analogy, as "
               "CGenFF does) and pass that file too; topon never substitutes "
               "a generic term.\n" + "\n".join(lines))
        super().__init__(msg)

    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0]


def element_from_mass(mass: float) -> str | None:
    """The element whose standard mass is nearest ``mass`` (within 0.6 amu)."""
    if mass <= 0.5:
        return None                      # lone pairs, Drude particles
    best = min(_ELEMENT_MASSES, key=lambda e: abs(_ELEMENT_MASSES[e] - mass))
    return best if abs(_ELEMENT_MASSES[best] - mass) < 0.6 else None


@dataclass
class RtfResidue:
    """One ``RESI`` or ``PRES`` block of a topology file.

    Atom names are upper-cased, as CHARMM reads them. ``atoms`` keeps file
    order. Names prefixed with ``+``/``-`` refer to the next/previous residue,
    and names prefixed with ``1``/``2`` in a patch refer to its two residues.
    """
    name: str
    charge: float
    is_patch: bool = False
    atoms: dict[str, tuple[str, float]] = field(default_factory=dict)
    bonds: list[tuple[str, str]] = field(default_factory=list)
    impropers: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: explicit ANGLE / DIHE entries (used by NOANG / NODIH residues such
    #: as TIP3, whose one angle is listed rather than generated)
    angles: list[tuple[str, str, str]] = field(default_factory=list)
    dihedrals: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: CMAP cross terms (eight atom names each), as the protein backbone lists them
    cmaps: list[tuple[str, ...]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    ics: list[dict] = field(default_factory=list)
    no_angles: bool = False
    no_dihedrals: bool = False
    source: str = ""

    def external_atoms(self) -> dict[str, int]:
        """Atoms bonded outside the residue, with how many such bonds.

        An atom counts once for every ``BOND X +Y`` / ``BOND X -Y`` it takes
        part in, and once for every ``+X`` / ``-X`` reference to it (the
        neighbouring residue's end of the same link, e.g. PEGM's ``C2`` is
        bonded by the next residue's ``BOND C1 -C2``).
        """
        count: dict[str, int] = defaultdict(int)
        refs: set[str] = set()
        for a, b in self.bonds:
            for x, y in ((a, b), (b, a)):
                if y[:1] in "+-" and x[:1] not in "+-":
                    count[x] += 1
                    refs.add(y[1:])
        for name in refs:
            if name in self.atoms:
                count[name] += 1
        return dict(count)


def _uc_tokens(line: str) -> list[str]:
    return line.upper().split()


def _strip(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _logical_lines(text: str) -> Iterable[str]:
    """Comment-free lines, with ``-`` continuations joined."""
    pending = ""
    for raw in text.splitlines():
        s = _strip(raw)
        if pending:
            s = pending + " " + s
            pending = ""
        if s.endswith(" -") or s == "-":
            pending = s[:-1].strip()
            continue
        if s:
            yield s
    if pending:
        yield pending


class CharmmParameterSet:
    """Everything a set of CHARMM files defines, with CHARMM lookup rules."""

    def __init__(self) -> None:
        self.masses: dict[str, float] = {}
        self.elements: dict[str, str] = {}
        self.bonds: dict[tuple[str, str], tuple[float, float]] = {}
        self.angles: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
        self.dihedrals: dict[tuple[str, str, str, str], list[tuple[float, int, float]]] = {}
        self.impropers: dict[tuple[str, str, str, str], tuple[float, float]] = {}
        #: type -> (epsilon, Rmin/2, epsilon_14, Rmin/2_14), CHARMM signs.
        self.nonbonded: dict[str, tuple[float, float, float, float]] = {}
        #: sorted type pair -> (Emin, Rmin, Emin_14, Rmin_14), CHARMM signs.
        self.nbfix: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        self.residues: dict[str, RtfResidue] = {}
        self.patches: dict[str, RtfResidue] = {}
        #: CMAP grids by their eight atom types, energies in file order.
        self.cmaps: dict[tuple[str, ...], list[float]] = {}
        self._cmap_open = None
        self.sources: list[str] = []
        self.e14fac: float = 1.0

    # ------------------------------------------------------------------ reading
    @classmethod
    def from_files(cls, *paths: str | Path) -> "CharmmParameterSet":
        """Read every file in order. Kind is taken from the extension
        (``.rtf``/``.top``/``.inp`` with RESI, ``.prm``/``.par``, ``.str``)."""
        ps = cls()
        for p in paths:
            ps.read(p)
        return ps

    def read(self, path: str | Path) -> None:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        ext = path.suffix.lower()
        if ext == ".str":
            self._read_stream(text, str(path))
        elif ext in (".rtf", ".top"):
            self._read_rtf(text, str(path))
        elif ext in (".prm", ".par"):
            self._read_prm(text, str(path))
        else:
            if re.search(r"^\s*(RESI|PRES)\s", text, flags=re.I | re.M):
                self._read_rtf(text, str(path))
            else:
                self._read_prm(text, str(path))
        self.sources.append(str(path))

    def _read_stream(self, text: str, src: str) -> None:
        """A stream file holds ``read rtf card`` and ``read param card``
        blocks, each closed by ``END``."""
        block: list[str] | None = None
        kind = ""
        for raw in text.splitlines():
            toks = _uc_tokens(_strip(raw))
            if block is None:
                if len(toks) >= 2 and toks[0] == "READ":
                    if toks[1].startswith("RTF"):
                        block, kind = [], "rtf"
                    elif toks[1].startswith("PARA"):
                        block, kind = [], "prm"
                continue
            if toks and toks[0] == "END":
                body = "\n".join(block)
                if kind == "rtf":
                    self._read_rtf(body, src)
                else:
                    self._read_prm(body, src)
                block = None
                continue
            block.append(raw)

    def _read_rtf(self, text: str, src: str) -> None:
        cur: RtfResidue | None = None
        for line in _logical_lines(text):
            if line.startswith("*"):
                continue
            toks = _uc_tokens(line)
            key = toks[0][:4]
            if key == "MASS" and len(toks) >= 4:
                t = toks[2]
                self.masses[t] = float(toks[3])
                if len(toks) >= 5 and toks[4].isalpha():
                    self.elements[t] = toks[4].upper()
                continue
            if key in ("RESI", "PRES"):
                name = toks[1]
                charge = float(toks[2]) if len(toks) > 2 and _is_float(toks[2]) else 0.0
                cur = RtfResidue(name=name, charge=charge, is_patch=(key == "PRES"),
                                 source=src,
                                 no_angles=any(t.startswith("NOAN") for t in toks[2:]),
                                 no_dihedrals=any(t.startswith("NODI") for t in toks[2:]))
                (self.patches if cur.is_patch else self.residues)[name] = cur
                continue
            if key == "END":
                cur = None
                continue
            if cur is None:
                continue
            if key == "ATOM" and len(toks) >= 4:
                cur.atoms[toks[1]] = (toks[2], float(toks[3]))
            elif key in ("BOND", "DOUB", "TRIP", "AROM"):
                names = toks[1:]
                if len(names) % 2:
                    raise CharmmFormatError(f"{src}: odd atom count on {line!r}")
                cur.bonds.extend(zip(names[0::2], names[1::2]))
            elif key in ("IMPR", "IMPH"):
                names = toks[1:]
                if len(names) % 4:
                    raise CharmmFormatError(f"{src}: impropers need 4 atoms: {line!r}")
                for i in range(0, len(names), 4):
                    cur.impropers.append(tuple(names[i:i + 4]))
            elif key in ("ANGL", "THET"):
                names = toks[1:]
                for i in range(0, len(names) - len(names) % 3, 3):
                    cur.angles.append(tuple(names[i:i + 3]))
            elif key in ("DIHE", "PHI"):
                names = toks[1:]
                for i in range(0, len(names) - len(names) % 4, 4):
                    cur.dihedrals.append(tuple(names[i:i + 4]))
            elif key == "CMAP":
                names = toks[1:]
                for i in range(0, len(names) - len(names) % 8, 8):
                    cur.cmaps.append(tuple(names[i:i + 8]))
            elif key == "DELE" and len(toks) >= 3 and toks[1].startswith("ATOM"):
                cur.deletes.extend(toks[2:])
            elif key == "IC" and len(toks) >= 10:
                a3 = toks[3]
                improper = a3.startswith("*")
                try:
                    cur.ics.append({
                        "atoms": (toks[1], toks[2], a3.lstrip("*"), toks[4]),
                        "r12": float(toks[5]), "a123": float(toks[6]),
                        "d1234": float(toks[7]), "a234": float(toks[8]),
                        "r34": float(toks[9]), "improper": improper,
                    })
                except ValueError:
                    pass

    def _read_prm(self, text: str, src: str) -> None:
        section = None
        seen_dih: set[tuple] = set()
        for line in _logical_lines(text):
            if line.startswith("*"):
                continue
            toks = _uc_tokens(line)
            head = toks[0]
            sect = _SECTIONS_PRM.get(head)
            # A header is a keyword not followed by a number (data rows start
            # with an atom type and then numbers, or four types).
            if sect is not None and (len(toks) == 1 or not _is_float(toks[1])):
                if sect == "END":
                    break
                section = sect
                if sect == "NONBONDED":
                    m = re.search(r"E14FAC\s+([-+0-9.eE]+)", line.upper())
                    if m:
                        self.e14fac = float(m.group(1))
                continue
            if head == "MASS" and len(toks) >= 4:
                self.masses[toks[2]] = float(toks[3])
                if len(toks) >= 5 and toks[4].isalpha():
                    self.elements[toks[2]] = toks[4]
                continue
            if head in ("CUTNB", "READ", "SET", "RETURN") or head.startswith("CTON"):
                continue
            try:
                self._read_prm_row(section, toks, seen_dih)
            except (ValueError, IndexError) as exc:
                raise CharmmFormatError(f"{src}: cannot read {section} line {line!r}") from exc

    def _read_prm_row(self, section, toks, seen_dih) -> None:
        if section == "BONDS" and len(toks) >= 4:
            self.bonds[_pair(toks[0], toks[1])] = (float(toks[2]), float(toks[3]))
        elif section == "ANGLES" and len(toks) >= 5:
            kub = float(toks[5]) if len(toks) > 6 else 0.0
            s0 = float(toks[6]) if len(toks) > 6 else 0.0
            key = (toks[0], toks[1], toks[2])
            # A term written in the other atom order is the same term, so a
            # later line replaces it (lookups try both orders).
            self.angles.pop(key[::-1], None)
            self.angles[key] = (float(toks[3]), float(toks[4]), kub, s0)
        elif section == "DIHEDRALS" and len(toks) >= 7:
            key = (toks[0], toks[1], toks[2], toks[3])
            k, n, d = float(toks[4]), int(toks[5]), float(toks[6])
            rev = key[::-1]
            if rev != key and rev in self.dihedrals:
                if rev in seen_dih:
                    key = rev                    # this file, other order: one entry
                else:
                    del self.dihedrals[rev]      # an earlier file's entry, replaced
            if key not in seen_dih:
                # First time this file mentions the key: it replaces what an
                # earlier file said, as a later parameter file does in CHARMM.
                self.dihedrals[key] = []
                seen_dih.add(key)
            terms = self.dihedrals[key]
            # The same multiplicity again replaces that term (last one wins).
            terms[:] = [t for t in terms if t[1] != n] + [(k, n, d)]
        elif section == "CMAP":
            # A map starts with its eight atom types and the grid size, then
            # lists grid**2 energies (psi fastest) over any number of lines.
            if len(toks) == 9 and not _is_float(toks[0]):
                n = int(toks[8])
                self._cmap_open = (tuple(toks[:8]), n)
                self.cmaps[tuple(toks[:8])] = []
            elif self._cmap_open is not None:
                key, n = self._cmap_open
                self.cmaps[key].extend(float(t) for t in toks)
                if len(self.cmaps[key]) >= n * n:
                    if len(self.cmaps[key]) > n * n:
                        raise ValueError(f"CMAP {' '.join(key)} has more than {n * n} values")
                    self._cmap_open = None
        elif section == "IMPROPER" and len(toks) >= 7:
            mult = int(float(toks[5]))
            if mult != 0:
                raise ValueError(f"improper {' '.join(toks[:4])} has multiplicity "
                                 f"{mult}; only the harmonic form (0) is supported")
            key = (toks[0], toks[1], toks[2], toks[3])
            self.impropers.pop(key[::-1], None)
            self.impropers[key] = (float(toks[4]), float(toks[6]))
        elif section == "NONBONDED" and len(toks) >= 4:
            eps, rmin2 = float(toks[2]), float(toks[3])
            if len(toks) >= 7 and _is_float(toks[5]):
                eps14, rmin2_14 = float(toks[5]), float(toks[6])
            else:
                eps14, rmin2_14 = eps, rmin2
            self.nonbonded[toks[0]] = (eps, rmin2, eps14, rmin2_14)
        elif section == "NBFIX" and len(toks) >= 4:
            emin, rmin = float(toks[2]), float(toks[3])
            if len(toks) >= 6 and _is_float(toks[5]):
                emin14, rmin14 = float(toks[4]), float(toks[5])
            else:
                emin14, rmin14 = emin, rmin
            self.nbfix[_pair(toks[0], toks[1])] = (emin, rmin, emin14, rmin14)
        # CMAP and HBOND rows are not needed here (CMAP grids are read by
        # LAMMPS `fix cmap` from their own file).

    # ------------------------------------------------------------------ lookups
    def element(self, atom_type: str) -> str | None:
        if atom_type in self.elements:
            return self.elements[atom_type]
        m = self.masses.get(atom_type)
        return None if m is None else element_from_mass(m)

    def lookup_bond(self, t1: str, t2: str) -> tuple[float, float] | None:
        return self.bonds.get(_pair(t1, t2))

    def lookup_angle(self, t1: str, t2: str, t3: str):
        return self.angles.get((t1, t2, t3)) or self.angles.get((t3, t2, t1))

    def lookup_dihedral_match(self, t1, t2, t3, t4):
        """``(terms, "exact" | "wildcard")`` or ``None``."""
        for key in ((t1, t2, t3, t4), (t4, t3, t2, t1)):
            if key in self.dihedrals:
                return self.dihedrals[key], "exact"
        for key in (("X", t2, t3, "X"), ("X", t3, t2, "X")):
            if key in self.dihedrals:
                return self.dihedrals[key], "wildcard"
        return None

    def lookup_dihedral(self, t1, t2, t3, t4):
        hit = self.lookup_dihedral_match(t1, t2, t3, t4)
        return hit[0] if hit else None

    def lookup_improper_match(self, t1, t2, t3, t4):
        """``((K, psi0), "exact" | "wildcard")`` or ``None``.

        CHARMM order: ``A B C D``, ``A X X D``, ``X B C D``, ``X X C D``,
        each tried on the atoms as listed and then reversed.
        """
        fwd, rev = (t1, t2, t3, t4), (t4, t3, t2, t1)
        patterns = (
            lambda a, b, c, d: (a, b, c, d),
            lambda a, b, c, d: (a, "X", "X", d),
            lambda a, b, c, d: ("X", b, c, d),
            lambda a, b, c, d: ("X", "X", c, d),
        )
        for i, pat in enumerate(patterns):
            for q in (fwd, rev):
                key = pat(*q)
                if key in self.impropers:
                    return self.impropers[key], ("exact" if i == 0 else "wildcard")
        return None

    def lookup_improper(self, t1, t2, t3, t4):
        hit = self.lookup_improper_match(t1, t2, t3, t4)
        return hit[0] if hit else None

    def lookup_lj(self, t: str):
        return self.nonbonded.get(t)

    def lookup_nbfix(self, t1: str, t2: str):
        return self.nbfix.get(_pair(t1, t2))


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# =============================================================================
# From typed atoms to LAMMPS type tables
# =============================================================================

@dataclass
class CharmmTerms:
    """A fully parameterised system, ready for a LAMMPS writer.

    Instances refer to atoms by the ids the caller passed. Type ids are
    1-based and deterministic (sorted keys). ``dihedrals`` holds one row per
    Fourier term: ``(type_id, i, j, k, l)``.
    """
    atom_type_of: dict[int, str]
    atom_types: dict[str, int]
    bond_types: dict[tuple, int]
    angle_types: dict[tuple, int]
    dihedral_types: dict[tuple, int]
    improper_types: dict[tuple, int]
    bond_params: dict[int, tuple[float, float]]
    angle_params: dict[int, tuple[float, float, float, float]]
    dihedral_params: dict[int, tuple[float, int, int, float]]
    improper_params: dict[int, tuple[float, float]]
    bonds: list[tuple[int, int, int]]
    angles: list[tuple[int, int, int, int]]
    dihedrals: list[tuple[int, int, int, int, int]]
    impropers: list[tuple[int, int, int, int, int]]
    #: LAMMPS pair_coeff i i: (eps, sigma, eps14, sigma14), LAMMPS signs.
    pair_params: dict[int, tuple[float, float, float, float]]
    #: explicit i<j pairs from NBFIX: (eps, sigma, eps14, sigma14).
    nbfix_params: dict[tuple[int, int], tuple[float, float, float, float]]
    masses: dict[int, float]
    wildcard_counts: dict[str, int] = field(default_factory=dict)

    def type_label(self, kind: str, tid: int) -> str:
        table = {"atom": self.atom_types, "bond": self.bond_types,
                 "angle": self.angle_types, "dihedral": self.dihedral_types,
                 "improper": self.improper_types}[kind]
        for key, v in table.items():
            if v == tid:
                if kind == "atom":
                    return key
                if kind == "dihedral":
                    return "-".join(key[0])
                return "-".join(key)
        return "?"


def rmin_half_to_sigma(rmin_half: float) -> float:
    """CHARMM Rmin/2 to the LJ sigma LAMMPS takes (sigma = Rmin / 2^(1/6))."""
    return 2.0 * abs(rmin_half) / 2.0 ** (1.0 / 6.0)


def generate_angles(bonds: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)
    out = []
    for c in sorted(adj):
        nb = sorted(adj[c])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                out.append((nb[i], c, nb[j]))
    return out


def generate_dihedrals(bonds: Iterable[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    bonds = list(bonds)
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)
    out = []
    for j, k in bonds:
        for i in sorted(adj[j]):
            if i == k:
                continue
            for l in sorted(adj[k]):
                if l == j or l == i:
                    continue
                out.append((i, j, k, l))
    return out


def one_four_weights(bonds, dihedrals) -> list[float]:
    """CHARMM 1-4 weight per dihedral (see the module docstring)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in bonds:
        adj[a].add(b)
        adj[b].add(a)
    count: dict[frozenset, int] = defaultdict(int)
    for i, _j, _k, l in dihedrals:
        count[frozenset((i, l))] += 1
    out = []
    for i, _j, _k, l in dihedrals:
        if l in adj[i] or adj[i] & adj[l]:
            out.append(0.0)           # also 1-2 or 1-3: excluded, as in CHARMM
        else:
            out.append(1.0 / count[frozenset((i, l))])
    return out


def parameterize(
    ps: CharmmParameterSet,
    atom_types: dict[int, str],
    bonds: Sequence[tuple[int, int]],
    impropers: Sequence[tuple[int, int, int, int]] = (),
    *,
    angles: Sequence[tuple[int, int, int]] | None = None,
    dihedrals: Sequence[tuple[int, int, int, int]] | None = None,
    context: str = "",
) -> CharmmTerms:
    """Look every term up and build LAMMPS type tables.

    ``atom_types`` maps atom id to CHARMM type. Angles and dihedrals are
    generated from ``bonds`` unless given (a caller passes them to leave out,
    e.g., the angles of a NOANG water). Raises
    :class:`MissingCharmmParameters` listing every undefined term.
    """
    if ps.e14fac != 1.0:
        # The dihedral weight scales the 1-4 LJ and Coulomb terms together,
        # so a scaled 1-4 Coulomb (older force fields) cannot be written.
        raise ValueError(f"E14FAC {ps.e14fac:g} in {'; '.join(Path(x).name for x in ps.sources)}; "
                         f"only 1.0 (every CHARMM36 and CGenFF file) is supported")
    missing: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    wild = defaultdict(int)
    angles = generate_angles(bonds) if angles is None else list(angles)
    dihedrals = generate_dihedrals(bonds) if dihedrals is None else list(dihedrals)
    t = atom_types

    # Atom types, masses, LJ.
    used_types = sorted(set(t[i] for i in t))
    atom_tid = {ty: n + 1 for n, ty in enumerate(used_types)}
    masses, pair_params = {}, {}
    for ty, tid in atom_tid.items():
        if ty not in ps.masses:
            missing["MASS"].append((ty,))
        else:
            masses[tid] = ps.masses[ty]
        lj = ps.lookup_lj(ty)
        if lj is None:
            missing["NONBONDED"].append((ty,))
            continue
        eps, rmin2, eps14, rmin2_14 = lj
        pair_params[tid] = (abs(eps), rmin_half_to_sigma(rmin2),
                            abs(eps14), rmin_half_to_sigma(rmin2_14))
    nbfix_params = {}
    for (a, b), (emin, rmin, emin14, rmin14) in ps.nbfix.items():
        if a in atom_tid and b in atom_tid:
            i, j = sorted((atom_tid[a], atom_tid[b]))
            s = 2.0 ** (-1.0 / 6.0)
            nbfix_params[(i, j)] = (abs(emin), abs(rmin) * s, abs(emin14), abs(rmin14) * s)

    # Bonds.
    bond_key = lambda a, b: _pair(t[a], t[b])
    bkeys = sorted({bond_key(a, b) for a, b in bonds})
    bond_tid = {k: n + 1 for n, k in enumerate(bkeys)}
    bond_params = {}
    for k, tid in bond_tid.items():
        p = ps.lookup_bond(*k)
        if p is None:
            missing["BONDS"].append(k)
        else:
            bond_params[tid] = p
    bond_rows = [(bond_tid[bond_key(a, b)], a, b) for a, b in bonds]

    # Angles (Urey-Bradley included).
    def akey(a, b, c):
        f, r = (t[a], t[b], t[c]), (t[c], t[b], t[a])
        return min(f, r)
    akeys = sorted({akey(*x) for x in angles})
    angle_tid = {k: n + 1 for n, k in enumerate(akeys)}
    angle_params = {}
    for k, tid in angle_tid.items():
        p = ps.lookup_angle(*k)
        if p is None:
            missing["ANGLES"].append(k)
        else:
            angle_params[tid] = p
    angle_rows = [(angle_tid[akey(*x)], *x) for x in angles]

    # Dihedrals: one LAMMPS type per (quad, term, 1-4 weight).
    weights = one_four_weights(bonds, dihedrals)
    dih_rows_spec = []
    dkeys: set[tuple] = set()
    for (a, b, c, d), w in zip(dihedrals, weights):
        f = (t[a], t[b], t[c], t[d])
        q = min(f, f[::-1])
        hit = ps.lookup_dihedral_match(*q)
        if hit is None:
            missing["DIHEDRALS"].append(q)
            continue
        terms, how = hit
        if how == "wildcard":
            wild["dihedral"] += 1
        for n, (k, mult, delta) in enumerate(terms):
            if abs(delta - round(delta)) > 1e-6:
                raise ValueError(f"dihedral {' '.join(q)} has a non-integer phase "
                                 f"{delta}; LAMMPS dihedral_style charmm needs integers")
            key = (q, n, (w if n == 0 else 0.0))
            dkeys.add(key)
            dih_rows_spec.append((key, a, b, c, d))
    dkeys_sorted = sorted(dkeys, key=lambda x: (x[0], x[1], -x[2]))
    dih_tid = {k: n + 1 for n, k in enumerate(dkeys_sorted)}
    dih_params = {}
    for (q, n, w), tid in dih_tid.items():
        k, mult, delta = ps.lookup_dihedral(*q)[n]
        dih_params[tid] = (k, mult, int(round(delta)), w)
    dih_rows = [(dih_tid[key], a, b, c, d) for key, a, b, c, d in dih_rows_spec]

    # Impropers, in the atom order the topology lists them.
    ikeys = sorted({(t[a], t[b], t[c], t[d]) for a, b, c, d in impropers})
    imp_tid = {k: n + 1 for n, k in enumerate(ikeys)}
    imp_params = {}
    for k, tid in imp_tid.items():
        hit = ps.lookup_improper_match(*k)
        if hit is None:
            missing["IMPROPER"].append(k)
            continue
        (kpsi, psi0), how = hit
        if how == "wildcard":
            wild["improper"] += 1
        imp_params[tid] = (kpsi, psi0)
    imp_rows = [(imp_tid[(t[a], t[b], t[c], t[d])], a, b, c, d) for a, b, c, d in impropers]

    if any(missing.values()):
        raise MissingCharmmParameters(missing, context)

    return CharmmTerms(
        atom_type_of=dict(t), atom_types=atom_tid,
        bond_types=bond_tid, angle_types=angle_tid,
        dihedral_types=dih_tid, improper_types=imp_tid,
        bond_params=bond_params, angle_params=angle_params,
        dihedral_params=dih_params, improper_params=imp_params,
        bonds=bond_rows, angles=angle_rows, dihedrals=dih_rows, impropers=imp_rows,
        pair_params=pair_params, nbfix_params=nbfix_params, masses=masses,
        wildcard_counts=dict(wild),
    )


def write_settings(path: str | Path, terms: CharmmTerms, *, soft: bool = False,
                   title: str = "CHARMM") -> None:
    """Write the LAMMPS coefficient include for ``terms``.

    ``soft=True`` writes the variant for stages that run ``pair_style soft``
    (or any pair style without 1-4 parameters): no ``pair_coeff`` lines and
    every dihedral weight 0, since LAMMPS refuses a non-zero weight without
    an ``lj/charmm*`` pair style. The bonded coefficients are otherwise the
    same.
    """
    lab = terms.type_label
    lines = [f"# {title} force-field coefficients (topon). "
             + ("Soft-stage variant: no pair coeffs, 1-4 weights 0."
                if soft else "Units real (kcal/mol, Angstrom)."), ""]
    if not soft:
        lines.append("# pair_coeff i i  epsilon sigma epsilon_14 sigma_14  (lj/charmm*)")
        for tid in sorted(terms.pair_params):
            e, s, e14, s14 = terms.pair_params[tid]
            lines.append(f"pair_coeff {tid} {tid} {e:.6f} {s:.6f} {e14:.6f} {s14:.6f} "
                         f"# {lab('atom', tid)}")
        if terms.nbfix_params:
            lines.append("# NBFIX pairs (override mixing)")
            for (i, j), (e, s, e14, s14) in sorted(terms.nbfix_params.items()):
                lines.append(f"pair_coeff {i} {j} {e:.6f} {s:.6f} {e14:.6f} {s14:.6f} "
                             f"# NBFIX {lab('atom', i)}-{lab('atom', j)}")
        lines.append("")
    lines.append("# bond_coeff  K  r0  (harmonic, E = K (r - r0)^2)")
    for tid in sorted(terms.bond_params):
        k, r0 = terms.bond_params[tid]
        lines.append(f"bond_coeff {tid} {k:.6g} {r0:.6g} # {lab('bond', tid)}")
    lines.append("")
    lines.append("# angle_coeff  K  theta0  K_ub  r_ub  (charmm)")
    for tid in sorted(terms.angle_params):
        k, th, kub, s0 = terms.angle_params[tid]
        lines.append(f"angle_coeff {tid} {k:.6g} {th:.6g} {kub:.6g} {s0:.6g} "
                     f"# {lab('angle', tid)}")
    lines.append("")
    lines.append("# dihedral_coeff  K  n  d  w  (charmm; w = 1-4 weight)")
    for tid in sorted(terms.dihedral_params):
        k, n, d, w = terms.dihedral_params[tid]
        w = 0.0 if soft else w
        lines.append(f"dihedral_coeff {tid} {k:.6g} {n} {d} {w:.6g} "
                     f"# {lab('dihedral', tid)}")
    lines.append("")
    lines.append("# improper_coeff  K  psi0  (harmonic)")
    for tid in sorted(terms.improper_params):
        k, psi0 = terms.improper_params[tid]
        lines.append(f"improper_coeff {tid} {k:.6g} {psi0:.6g} # {lab('improper', tid)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


#: The six maps LAMMPS ``fix cmap`` reads, in its order (crossterm types 1-6),
#: by the atom types of their CHARMM36/36m definitions.
LAMMPS_CMAP_ORDER = (
    ("alanine", ("C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1")),
    ("alanine before proline", ("C", "NH1", "CT1", "C", "NH1", "CT1", "C", "N")),
    ("proline", ("C", "N", "CP1", "C", "N", "CP1", "C", "NH1")),
    ("proline before proline", ("C", "N", "CP1", "C", "N", "CP1", "C", "N")),
    ("glycine", ("C", "NH1", "CT2", "C", "NH1", "CT2", "C", "NH1")),
    ("glycine before proline", ("C", "NH1", "CT2", "C", "NH1", "CT2", "C", "N")),
)


def write_lammps_cmap(ps: CharmmParameterSet, path: str | Path) -> None:
    """Write the ``fix cmap`` grid file from the parameter files' CMAP section.

    The file is then the same force field as the PRM (a separately shipped
    map can be an older version; LAMMPS's ``charmm36.cmap`` is the C36 map,
    whose alanine maps differ from CHARMM36m's by up to 1.8 kcal/mol).
    """
    lines = ["# CMAP grids written by topon from the CHARMM parameter files:",
             "#   " + "; ".join(Path(x).name for x in ps.sources), "# UNITS: real", ""]
    for k, (label, key) in enumerate(LAMMPS_CMAP_ORDER, 1):
        grid = ps.cmaps.get(key)
        if grid is None:
            raise MissingCharmmParameters({"CMAP": [key]}, "fix cmap")
        n = int(round(len(grid) ** 0.5))
        if n * n != len(grid) or n != 24:
            raise ValueError(f"CMAP {label}: {len(grid)} values, LAMMPS needs 24 x 24")
        lines.append(f"# {label} map, type {k}")
        lines.append("")
        for row in range(n):
            lines.append(f"# {-180.0 + 15.0 * row:.1f}")
            vals = grid[row * n:(row + 1) * n]
            for i in range(0, n, 5):
                lines.append(" ".join(f"{v:12.6f}" for v in vals[i:i + 5]))
            lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lj_pair_coeffs(path: str | Path, terms: CharmmTerms) -> None:
    """``pair_coeff`` lines for an ``lj/cut*`` pair style (the ramp stage).

    ``fix adapt`` cannot scale epsilon in the ``lj/charmm*`` styles, so an
    epsilon ramp runs ``lj/cut/coul/long`` with these coefficients and
    ``pair_modify mix arithmetic``, which reproduces CHARMM's mixing and its
    NBFIX pairs. The 1-4 terms are off while it runs (pair with the soft
    settings, weights 0) and come back with the full settings afterwards.
    """
    lab = terms.type_label
    lines = ["# CHARMM LJ for lj/cut* pair styles (epsilon ramp stage); use with",
             "# pair_modify mix arithmetic. No 1-4 parameters here.", ""]
    for tid in sorted(terms.pair_params):
        e, s, _, _ = terms.pair_params[tid]
        lines.append(f"pair_coeff {tid} {tid} {e:.6f} {s:.6f} # {lab('atom', tid)}")
    for (i, j), (e, s, _, _) in sorted(terms.nbfix_params.items()):
        lines.append(f"pair_coeff {i} {j} {e:.6f} {s:.6f} "
                     f"# NBFIX {lab('atom', i)}-{lab('atom', j)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "CharmmFormatError",
    "CharmmParameterSet",
    "CharmmTerms",
    "MissingCharmmParameters",
    "RtfResidue",
    "element_from_mass",
    "generate_angles",
    "generate_dihedrals",
    "one_four_weights",
    "parameterize",
    "rmin_half_to_sigma",
    "LAMMPS_CMAP_ORDER",
    "write_lammps_cmap",
    "write_lj_pair_coeffs",
    "write_settings",
]
