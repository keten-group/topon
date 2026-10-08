"""``fix bond/react`` templates generated from each system's own type order.

An epoxy-amine cure runs two reactions: an epoxide
with a primary amine (NH2 to NH, 21-atom templates, two edge atoms) and with
a secondary amine (NH to N, 24 atoms, five edge atoms). In both the epoxide
ring opens at its terminal CH2 carbon, which bonds to the N, and the N-H
hydrogen moves to the epoxide oxygen; no atom is deleted. The template
chemistry here (atoms by DREIDING type, coordinates, bonds before and after,
initiators and edge atoms) is fixed, and only the type ids change from one
system to another.

Templates must carry the type ids of the system they run in, and systems
built in different ways do not share one order (Si O C H N in some, Si O C N H
in others), so templates are generated per system and never copied between
systems: :func:`read_type_tables` reads which DREIDING types every atom,
bond, angle and dihedral type id of a data file stands for, and
:func:`write_epoxy_amine_templates` writes the four molecule files and two
map files with those ids. Angles and dihedrals are enumerated from each
template's bonds; a dihedral type is matched by its four atom types and its
K (DREIDING's V/2 over the torsions about the central bond, counted in the
template), since one system can hold several types of one name with
different K. The molecule files carry no Charges section: with one,
bond/react sets the post-reaction template's charges on every reacting atom,
and without one it leaves every charge as it is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Element of a mass in a data file, and the DREIDING type an sp3 atom of it takes.
_MASS_ELEMENT = {1.008: "H", 12.011: "C", 14.007: "N", 15.999: "O", 28.086: "Si",
                 32.06: "S", 18.998: "F", 30.974: "P"}
_SP3_TYPE = {"H": "H_", "C": "C_3", "N": "N_3", "O": "O_3", "Si": "Si3", "S": "S_3",
             "F": "F_", "P": "P_3"}

_SECTIONS = {"Masses", "Pair", "PairIJ", "Bond", "Angle", "Dihedral", "Improper", "Atoms",
             "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"}


class TemplateError(ValueError):
    """The system cannot hold these templates (a type is missing or ambiguous)."""


def _canon_bond(t1, t2):
    return tuple(sorted((t1, t2)))


def _canon_angle(t1, tc, t2):
    o = sorted((t1, t2))
    return (o[0], tc, o[1])


def _canon_dihedral(t1, t2, t3, t4):
    return min((t1, t2, t3, t4), (t4, t3, t2, t1))


@dataclass
class TypeTables:
    """What every type id of a data file stands for, by DREIDING type names.

    ``atom`` maps an atom type id to its name; ``bond``, ``angle`` and
    ``dihedral`` map a type id to ``(canonical names, coefficients)``;
    ``pair`` holds the Pair Coeffs (epsilon, sigma) when the file has them.
    ``source`` says where each table's names came from (``comments`` or
    ``instances``: the atoms that carry the type in the file).
    """

    atom: Dict[int, str] = field(default_factory=dict)
    bond: Dict[int, tuple] = field(default_factory=dict)
    angle: Dict[int, tuple] = field(default_factory=dict)
    dihedral: Dict[int, tuple] = field(default_factory=dict)
    pair: Dict[int, tuple] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)

    def atom_id(self, name: str) -> int:
        ids = [t for t, n in self.atom.items() if n == name]
        if not ids:
            raise TemplateError(f"the system has no atom type {name}")
        if len(ids) > 1:
            raise TemplateError(f"atom type {name} has several ids: {ids}")
        return ids[0]

    def _by_names(self, table, names, what):
        ids = [t for t, (nm, _c) in sorted(table.items()) if nm == names]
        if not ids:
            raise TemplateError(f"the system has no {what} type {'-'.join(names)}")
        return ids

    def bond_id(self, names) -> int:
        """The bond type of these atom types (the last listed, if several)."""
        return self._by_names(self.bond, _canon_bond(*names), "bond")[-1]

    def angle_id(self, names) -> int:
        """The angle type of these atom types (the last listed, if several)."""
        return self._by_names(self.angle, _canon_angle(*names), "angle")[-1]

    def dihedral_id(self, names, k: float, d: int, n: int, tol: float = 1e-4) -> int:
        """The dihedral type of these atom types with ``harmonic`` coefficients K d n.

        Several ids of one name and one set of coefficients are the same
        term, written once per molecule species; the last listed is taken.
        """
        canon = _canon_dihedral(*names)
        ids = self._by_names(self.dihedral, canon, "dihedral")
        hit = [t for t in ids if abs(self.dihedral[t][1][0] - k) <= tol
               and tuple(int(x) for x in self.dihedral[t][1][1:3]) == (int(d), int(n))]
        if not hit:
            have = sorted({tuple(round(x, 6) for x in self.dihedral[t][1][:3]) for t in ids})
            raise TemplateError(f"no dihedral type {'-'.join(canon)} with K d n "
                                f"{k:.6f} {d} {n} (the system has {have})")
        return hit[-1]


def _names_from_comment(line: str, n: int = 1):
    """The ``n`` type names a coefficient comment gives (``O_3-Si3`` or
    ``C_3 Si3 C_3 H_``), or None when it gives another number of them."""
    if "#" not in line:
        return None
    c = line.split("#", 1)[1].strip()
    if not c:
        return None
    names = tuple(x.strip() for x in (c.split("-") if "-" in c else c.split()))
    names = names if n > 1 else names[:1]
    return names if len(names) == n and all(names) else None


def read_type_tables(path) -> TypeTables:
    """The type tables of a LAMMPS data file (``atom_style full``).

    Atom type names come from the comments of Masses or Pair Coeffs (as
    topon's simbox writes them), or else from the masses (an sp3 DREIDING
    type per element: ``C_3``, ``N_3``, ``O_3``, ``Si3``, ``H_``). Bond,
    angle and dihedral names come from their coefficient comments, or else
    from the atoms that carry the type in the file; where both exist they
    must agree. A type that no molecule holds yet (one that only a reaction
    creates) can only be named by its comment.
    """
    masses, mass_names, pair, pair_names = {}, {}, {}, {}
    coeffs = {"Bond": {}, "Angle": {}, "Dihedral": {}}
    comments = {"Bond": {}, "Angle": {}, "Dihedral": {}}
    atype, inst = {}, {"Bonds": {}, "Angles": {}, "Dihedrals": {}}
    sec = None
    with open(path, errors="ignore") as fh:
        fh.readline()
        for line in fh:
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            if s[0].isalpha():
                w = s.split()[0]
                sec = w if w in _SECTIONS else "?"
                if sec in ("Pair", "Bond", "Angle", "Dihedral", "Improper", "PairIJ"):
                    sec = sec + "C" if "Coeffs" in s else sec
                continue
            p = s.split()
            if sec == "Masses":
                masses[int(p[0])] = float(p[1])
                nm = _names_from_comment(line)
                if nm:
                    mass_names[int(p[0])] = nm[0]
            elif sec == "PairC" and len(p) >= 3:
                pair[int(p[0])] = (float(p[1]), float(p[2]))
                nm = _names_from_comment(line)
                if nm:
                    pair_names[int(p[0])] = nm[0]
            elif sec in ("BondC", "AngleC", "DihedralC"):
                kind = sec[:-1]
                coeffs[kind][int(p[0])] = tuple(float(x) for x in p[1:])
                nm = _names_from_comment(line, {"Bond": 2, "Angle": 3, "Dihedral": 4}[kind])
                if nm:
                    comments[kind][int(p[0])] = nm
            elif sec == "Atoms" and len(p) >= 7:
                atype[int(p[0])] = int(p[2])
            elif sec in inst and len(p) >= 4:
                t = int(p[1])
                if t not in inst[sec]:
                    inst[sec][t] = [int(x) for x in p[2:]]
    tt = TypeTables(pair=pair)
    if mass_names or pair_names:
        tt.atom = {t: mass_names.get(t, pair_names.get(t)) for t in masses}
        tt.source["atom"] = "comments"
    else:
        tt.atom = {}
        for t, m in masses.items():
            el = _MASS_ELEMENT[min(_MASS_ELEMENT, key=lambda k: abs(k - m))]
            tt.atom[t] = _SP3_TYPE[el]
        tt.source["atom"] = "masses (sp3 types assumed)"
    if any(v is None for v in tt.atom.values()):
        raise TemplateError(f"{path}: some atom types are named and some are not")
    canon = {"Bond": _canon_bond, "Angle": _canon_angle, "Dihedral": _canon_dihedral}
    for kind, sec in (("Bond", "Bonds"), ("Angle", "Angles"), ("Dihedral", "Dihedrals")):
        table, used = {}, set()
        for t, c in coeffs[kind].items():
            from_comment = comments[kind].get(t)
            from_atoms = None
            if t in inst[sec]:
                from_atoms = canon[kind](*(tt.atom[atype[a]] for a in inst[sec][t]))
            if from_comment is not None:
                nm = canon[kind](*from_comment)
                if from_atoms is not None and from_atoms != nm:
                    raise TemplateError(f"{path}: {kind.lower()} type {t} is labelled "
                                        f"{'-'.join(nm)} but joins {'-'.join(from_atoms)}")
                used.add("comments")
            elif from_atoms is not None:
                nm = from_atoms
                used.add("instances")
            else:
                continue                    # unnamed and unused: no template can need it
            table[t] = (nm, c)
        setattr(tt, kind.lower(), table)
        tt.source[kind.lower()] = "+".join(sorted(used)) or "none"
    return tt


# ---------------------------------------------------------------------------
# The epoxy-amine reaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactionTemplate:
    """One reaction: atoms (DREIDING types), coordinates, bonds before and after."""

    name: str
    pre_header: str
    post_header: str
    atoms: Tuple[str, ...]
    pre_coords: Tuple[Tuple[float, float, float], ...]
    post_coords: Tuple[Tuple[float, float, float], ...]
    pre_bonds: Tuple[Tuple[int, int], ...]
    post_bonds: Tuple[Tuple[int, int], ...]
    initiators: Tuple[int, int]
    edges: Tuple[int, ...]


_PRI_ATOMS = ("C_3", "C_3", "O_3", "N_3", "H_", "H_", "H_", "H_", "H_", "C_3", "C_3", "H_",
              "H_", "O_3", "C_3", "H_", "H_", "C_3", "H_", "H_", "C_3")
_SIDE = (                     # epoxide side chain and amine side, shared by both templates
    (1.980, -0.080, 0.930), (2.670, -0.530, -0.800), (-0.100, -6.540, -0.150),
    (2.300, -0.280, -1.800), (2.550, -1.600, -0.650), (4.050, -0.150, -0.650),
    (4.650, -0.950, 0.350), (0.550, -6.100, 0.650), (0.300, -6.250, -1.130),
    (-1.550, -6.050, 0.100), (-1.450, -4.960, 0.250), (-1.950, -6.400, 1.050),
    (-2.450, -6.550, -1.100))
_SIDE_POST = (
    (1.980, -0.080, 0.930), (2.670, -0.530, -0.800), (-0.100, -2.930, -0.150),
    (2.300, -0.280, -1.800), (2.550, -1.600, -0.650), (4.050, -0.150, -0.650),
    (4.650, -0.950, 0.350), (0.550, -3.370, 0.650), (0.300, -3.220, -1.130),
    (-1.550, -3.420, 0.100), (-1.450, -4.510, 0.250), (-1.950, -3.070, 1.050),
    (-2.450, -2.920, -1.100))
_SIDE_BONDS = ((1, 7), (1, 8), (2, 9), (2, 10), (10, 12), (10, 13), (10, 14), (14, 15))
_AMINE_BONDS = ((4, 11), (11, 16), (11, 17), (11, 18), (18, 19), (18, 20), (18, 21))

#: NH2 + epoxide -> NH: the C1-O bond and one N-H break, C1-N and O-H form.
PRIMARY = ReactionTemplate(
    name="primary",
    pre_header="Pre: epoxide + primary amine (NH2)",
    post_header="Post: primary->secondary",
    atoms=_PRI_ATOMS,
    pre_coords=((0.000, 0.000, 0.000), (1.470, 0.350, 0.000), (0.730, 1.200, 0.000),
                (0.000, -8.000, 0.000), (-0.510, -8.520, 0.810), (0.510, -8.520, -0.810),
                (-0.520, -0.360, 0.890), (-0.520, -0.360, -0.890)) + _SIDE,
    post_coords=((0.000, 0.000, 0.000), (1.470, 0.350, 0.000), (0.730, 1.200, 0.000),
                 (0.000, -1.470, 0.000), (1.060, 1.850, 0.650), (0.510, -1.990, -0.810),
                 (-0.520, -0.360, 0.890), (-0.520, -0.360, -0.890)) + _SIDE_POST,
    pre_bonds=((1, 2), (1, 3), (2, 3), (1, 7), (1, 8), (2, 9), (2, 10), (10, 12), (10, 13),
               (10, 14), (14, 15), (4, 5), (4, 6), (4, 11), (11, 16), (11, 17), (11, 18),
               (18, 19), (18, 20), (18, 21)),
    post_bonds=((1, 2), (2, 3), (1, 7), (1, 8), (2, 9), (2, 10), (10, 12), (10, 13), (10, 14),
                (14, 15), (4, 6), (4, 11), (11, 16), (11, 17), (11, 18), (18, 19), (18, 20),
                (18, 21), (1, 4), (3, 5)),
    initiators=(1, 4),
    edges=(15, 21),
)

#: NH + epoxide -> N: atom 5 is the carbon of the first reaction, its three
#: other neighbours (22-24) edge atoms, so the match is unambiguous.
SECONDARY = ReactionTemplate(
    name="secondary",
    pre_header="Pre: epoxide + secondary amine (NH)",
    post_header="Post: secondary->tertiary",
    atoms=_PRI_ATOMS[:4] + ("C_3",) + _PRI_ATOMS[5:] + ("C_3", "H_", "H_"),
    pre_coords=((0.000, 0.000, 0.000), (1.470, 0.350, 0.000), (0.730, 1.200, 0.000),
                (0.000, -8.000, 0.000), (-0.510, -9.470, 0.000), (0.510, -8.520, -0.810),
                (-0.520, -0.360, 0.890), (-0.520, -0.360, -0.890)) + _SIDE
               + ((0.960, -9.820, 0.000), (-1.030, -9.110, 0.890), (-1.030, -9.110, -0.890)),
    post_coords=((0.000, 0.000, 0.000), (1.470, 0.350, 0.000), (0.730, 1.200, 0.000),
                 (0.000, -1.470, 0.000), (-0.510, -2.940, 0.000), (1.060, 1.850, 0.650),
                 (-0.520, -0.360, 0.890), (-0.520, -0.360, -0.890)) + _SIDE_POST
                + ((0.960, -3.290, 0.000), (-1.030, -2.580, 0.890), (-1.030, -2.580, -0.890)),
    pre_bonds=((1, 2), (1, 3), (2, 3), (1, 7), (1, 8), (2, 9), (2, 10), (10, 12), (10, 13),
               (10, 14), (14, 15), (4, 5), (4, 6), (4, 11), (11, 16), (11, 17), (11, 18),
               (18, 19), (18, 20), (18, 21), (5, 22), (5, 23), (5, 24)),
    post_bonds=((1, 2), (2, 3), (1, 7), (1, 8), (2, 9), (2, 10), (10, 12), (10, 13), (10, 14),
                (14, 15), (4, 5), (4, 11), (11, 16), (11, 17), (11, 18), (18, 19), (18, 20),
                (18, 21), (1, 4), (3, 6), (5, 22), (5, 23), (5, 24)),
    initiators=(1, 4),
    edges=(15, 21, 22, 23, 24),
)

EPOXY_AMINE = (PRIMARY, SECONDARY)


def _adjacency(bonds):
    adj = {}
    for a, b in bonds:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def template_angles(bonds) -> List[tuple]:
    """Every angle of a bond list, centre by centre, in sorted order."""
    adj = _adjacency(bonds)
    return [(n1, c, n2) for c in sorted(adj) for n1, n2 in combinations(sorted(adj[c]), 2)]


def template_dihedrals(bonds) -> List[tuple]:
    """Every dihedral of a bond list with the torsions about its central bond."""
    adj = _adjacency(bonds)
    out = []
    for i, j in bonds:
        ni, nj = sorted(adj[i] - {j}), sorted(adj[j] - {i})
        n = sum(1 for x in ni for y in nj if x != y)
        out += [(x, i, j, y, n) for x in ni for y in nj if x != y]
    return out


def _dihedral_coeffs(names, n_torsions, params) -> tuple:
    """``harmonic`` K d n of a template dihedral: DREIDING's V / 2 over the
    ``n_torsions`` about its central bond counted in the template, and
    LAMMPS's sign of d.

    The count is the molecule's own only where both central atoms keep
    every neighbour inside the template, which holds for every dihedral of
    the epoxy-amine templates (edge atoms are never central); a template
    cut through a central atom would get the wrong K. A term with no
    DREIDING parameters (K 0) is written with d 1, which harmonic requires.
    """
    from topon.forcefield.dreiding import find_parameter
    terms = find_parameter(_canon_dihedral(*names), params["dihedral_params"]) or []
    t = terms[0] if terms else {}
    v = t.get("v_n", 0.0)
    k = round(0.5 * v / n_torsions if n_torsions else 0.5 * v, 6)
    d = -int(t.get("d", 0)) or 1
    return k, d, int(t.get("n", 0))


def molecule_file(tt: TypeTables, rx: ReactionTemplate, which: str, params=None) -> str:
    """The text of one bond/react molecule file (``which`` is ``pre`` or ``post``)."""
    if params is None:
        from topon.forcefield.dreiding import _BUNDLED_PARAM_FILE, parse_dreiding_parameter_file
        params = parse_dreiding_parameter_file(str(_BUNDLED_PARAM_FILE))
    bonds = rx.pre_bonds if which == "pre" else rx.post_bonds
    coords = rx.pre_coords if which == "pre" else rx.post_coords
    header = rx.pre_header if which == "pre" else rx.post_header
    name = {i + 1: t for i, t in enumerate(rx.atoms)}
    angles = template_angles(bonds)
    dihedrals = template_dihedrals(bonds)
    out = [f"# {header}", "", f"{len(rx.atoms)} atoms", f"{len(bonds)} bonds",
           f"{len(angles)} angles", f"{len(dihedrals)} dihedrals", "", "Coords", ""]
    out += [f"{i + 1}    {x:.3f}   {y:.3f}   {z:.3f}" for i, (x, y, z) in enumerate(coords)]
    out += ["", "Types", ""]
    out += [f"{i} {tt.atom_id(name[i])}" for i in range(1, len(rx.atoms) + 1)]
    # no Charges section, deliberately (see the module doc)
    out += ["", "Molecules", ""] + [f"{i} 1" for i in range(1, len(rx.atoms) + 1)]
    out += ["", "Bonds", ""]
    out += [f"{k}   {tt.bond_id((name[a], name[b]))}   {a}   {b}"
            for k, (a, b) in enumerate(bonds, 1)]
    out += ["", "Angles", ""]
    out += [f"{k}  {tt.angle_id((name[a], name[c], name[b]))}   {a}   {c}   {b}"
            for k, (a, c, b) in enumerate(angles, 1)]
    out += ["", "Dihedrals", ""]
    for k, (a, b, c, d, n) in enumerate(dihedrals, 1):
        names = (name[a], name[b], name[c], name[d])
        out.append(f"{k}  {tt.dihedral_id(names, *_dihedral_coeffs(names, n, params))}   "
                   f"{a}   {b}   {c}   {d}")
    return "\n".join(out) + "\n"


def map_file(rx: ReactionTemplate) -> str:
    """The text of a reaction map: identity equivalences, initiators, edge atoms."""
    n = len(rx.atoms)
    out = ["Epoxide-amine crosslink map", "", f"{n} equivalences", f"{len(rx.edges)} edgeIDs",
           "", "InitiatorIDs", ""] + [str(i) for i in rx.initiators]
    out += ["", "EdgeIDs", ""] + [str(e) for e in rx.edges]
    out += ["", "Equivalences", ""] + [f"{i} {i}" for i in range(1, n + 1)]
    return "\n".join(out) + "\n"


def _needed(reactions, params):
    """Every (kind, names, coefficients) the templates' interactions need."""
    from topon.forcefield.dreiding import type_coefficients
    need = {}
    for rx in reactions:
        name = {i + 1: t for i, t in enumerate(rx.atoms)}
        for bonds in (rx.pre_bonds, rx.post_bonds):
            for a, b in bonds:
                nm = _canon_bond(name[a], name[b])
                need.setdefault(("bond", nm), type_coefficients("bond", nm, params))
            for a, c, b in template_angles(bonds):
                nm = _canon_angle(name[a], name[c], name[b])
                need.setdefault(("angle", nm), type_coefficients("angle", nm, params))
            for a, b, c, d, n in template_dihedrals(bonds):
                nm = _canon_dihedral(name[a], name[b], name[c], name[d])
                kdn = _dihedral_coeffs(nm, n, params)
                need.setdefault(("dihedral", nm, kdn), kdn)
    return need


def add_reaction_types(src, dst, reactions=EPOXY_AMINE) -> list:
    """Copy a data file, adding any type the reactions need that it lacks.

    A cell built for bond/react must already list the bond, angle and
    dihedral types its reactions create, since a reaction cannot add a type.
    A cell that lists every one is copied byte for byte; a cell built
    otherwise may lack some, or hold a dihedral of
    the right atom types only at another K (as the universal type map topon
    applied before 0.4.5 wrote them, merging types by name; a topon simbox
    box lacks the product-only types, which
    :func:`topon.simbox.workflow.prepare_bond_react` adds with this). Each
    missing type is appended after the last of
    its kind, with DREIDING's coefficients for ``harmonic`` styles
    (:func:`topon.forcefield.dreiding.type_coefficients`; a dihedral's K
    from the torsions about its central bond in the template), and the
    header count raised. Returns ``[(kind, id, names, coefficients)]``.
    """
    from topon.forcefield.dreiding import _BUNDLED_PARAM_FILE, parse_dreiding_parameter_file
    params = parse_dreiding_parameter_file(str(_BUNDLED_PARAM_FILE))
    tt = read_type_tables(src)
    missing = []
    for key, coeffs in _needed(reactions, params).items():
        kind, nm = key[0], key[1]
        table = getattr(tt, kind)
        if kind == "dihedral":
            k, d, n_ = key[2]
            ok = any(n == nm and abs(c[0] - k) <= 1e-4
                     and tuple(int(x) for x in c[1:3]) == (d, n_) for n, c in table.values())
        else:
            ok = any(n == nm for n, c in table.values())
        if not ok:
            missing.append((kind, nm, coeffs))
    raw = Path(src).read_bytes().decode("utf-8", errors="replace")
    if not missing:
        Path(dst).write_bytes(raw.encode("utf-8"))
        return []
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(eol)
    added = []
    for kind in ("bond", "angle", "dihedral"):
        todo = [m for m in missing if m[0] == kind]
        if not todo:
            continue
        head = next((i for i, l in enumerate(lines)
                     if re.match(rf"^\s*\d+\s+{kind}\s+types\b", l)), None)
        sec = next((i for i, l in enumerate(lines)
                    if l.strip().startswith(f"{kind.capitalize()} Coeffs")), None)
        if head is None or sec is None:
            raise TemplateError(f"{src}: no {kind} types header or {kind.capitalize()} Coeffs "
                                f"section to add {len(todo)} reaction type(s) to")
        n0 = int(lines[head].split()[0])
        # the section's last data line: the last digit-led line before the next header
        j = sec + 1
        end = sec
        while j < len(lines) and (not lines[j].strip() or lines[j].strip()[0].isdigit()):
            if lines[j].strip():
                end = j
            j += 1
        if end == sec:                      # an empty section: data after the blank line
            end = sec + 1
            if end >= len(lines) or lines[end].strip():
                lines.insert(end, "")
            if end + 1 >= len(lines) or lines[end + 1].strip():
                lines.insert(end + 1, "")
        new = []
        for k, (_kind, nm, c) in enumerate(todo, 1):
            tid = n0 + k
            if kind == "dihedral":
                text = f"{tid} {c[0]:.6f} {int(c[1])} {int(c[2])}  # {'-'.join(nm)}"
            else:
                text = f"{tid} {c[0]:g} {c[1]:g}  # {'-'.join(nm)}"
            new.append(text)
            added.append((kind, tid, nm, c))
        lines[end + 1:end + 1] = new
        lines[head] = re.sub(r"\d+", str(n0 + len(todo)), lines[head], count=1)
    Path(dst).write_bytes(eol.join(lines).encode("utf-8"))
    return added


def write_epoxy_amine_templates(system_data, out_dir, reactions=EPOXY_AMINE) -> dict:
    """The four molecule files and two maps of the epoxy-amine cure, for this system.

    Writes ``pre_react_<r>.mol``, ``post_react_<r>.mol`` and
    ``rxn_map_<r>.txt`` for ``r`` in primary and secondary into ``out_dir``,
    with the type ids of ``system_data``. Raises :class:`TemplateError`
    when the system lacks a type the templates need (an atom type, or a bond,
    angle or dihedral type a reaction creates, which the system must already
    list). Returns the files written, the atom type order and where the
    names were read from.
    """
    from topon.forcefield.dreiding import _BUNDLED_PARAM_FILE, parse_dreiding_parameter_file
    params = parse_dreiding_parameter_file(str(_BUNDLED_PARAM_FILE))
    tt = read_type_tables(system_data)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for rx in reactions:
        for which in ("pre", "post"):
            p = out / f"{which}_react_{rx.name}.mol"
            p.write_text(molecule_file(tt, rx, which, params), encoding="utf-8", newline="\n")
            files.append(p.name)
        p = out / f"rxn_map_{rx.name}.txt"
        p.write_text(map_file(rx), encoding="utf-8", newline="\n")
        files.append(p.name)
    return {"files": files, "atom_types": {int(t): n for t, n in sorted(tt.atom.items())},
            "names_from": dict(tt.source)}


def template_types(path) -> Dict[str, list]:
    """The type ids a molecule file uses, per section (for checks and reports)."""
    sec, out = None, {"Types": [], "Bonds": [], "Angles": [], "Dihedrals": []}
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^[A-Za-z]", s):
            sec = s.split()[0]
            continue
        if sec in out:
            out[sec].append(int(s.split()[1]))
    return out
