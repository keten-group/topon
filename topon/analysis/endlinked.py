"""Read a LAMMPS data file in the end-linked convention.

The convention of `fix bond/create` datasets, and of topon's own
``write_endlinked`` and ``CGWriter(convention="endlinked")``: atom type 1 is
a chain-end bead, 2 a chain interior bead, 3 a junction; every chain is one
molecule and every junction its own. A chain is classed by how its two ends
are bonded:

``bridge``
    both ends on junctions, two different ones (an elastically active strand)
``loop``
    both ends on the same junction (a primary loop)
``dangling``
    one end on a junction
``free``
    neither (sol)

The strand graph comes back in the convention :mod:`topon.analysis.descriptors`
reads: junctions are nodes with ``kind="junction"``, every dangling chain adds
one ``kind="end"`` node at its free end, chains are edges with ``cls`` and
``chain`` (the molecule id), and a primary loop is a self-loop. Free chains are
not in the graph, since nothing bonds them to it; their count is recorded in
``G.graph["sol_chains"]``.

Ported from ``refnet.py`` of the bond/create validation scripts, which the
reference numbers in docs/USAGE.md were measured with. Two differences: a
chain of one bead (both ends on the same bead) is read instead of failing,
and nothing is cached beside the input file.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

#: Atom types of the end-linked convention.
END, INTERIOR, JUNCTION = 1, 2, 3

#: Strand classes, in the order every per-class table is printed.
CLASSES = ("bridge", "loop", "dangling", "free")

_SECTIONS = ("Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals",
             "Impropers", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs",
             "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs")


class NotEndLinked(ValueError):
    """The data file is not in the end-linked convention."""


@dataclass
class Strand:
    """One chain: its beads in order and what each end is bonded to."""

    mol: int
    seq: list                          # bead ids, bonded end first
    ends: tuple                        # (junction or None, junction or None)
    cls: str                           # bridge | loop | dangling | free


@dataclass
class EndLinkedSystem:
    """A data file read down to its chains and its strand graph."""

    path: Optional[Path]
    box: np.ndarray                    # (3,) edge lengths
    lo: np.ndarray                     # (3,) lower corner
    types: dict                        # atom id -> type
    mol: dict                          # atom id -> molecule id
    pos: dict                          # atom id -> (3,) wrapped position
    bonds: list                        # (i, j) atom-id pairs
    adj: dict                          # atom id -> bonded atom ids
    junctions: list                    # junction atom ids, sorted
    strands: dict                      # molecule id -> Strand, sorted by id
    graph: nx.MultiGraph
    velocities: dict = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.types)

    @property
    def density(self) -> float:
        return float(self.n_atoms / np.prod(self.box))

    def class_counts(self) -> dict:
        c = collections.Counter(s.cls for s in self.strands.values())
        return {k: int(c.get(k, 0)) for k in CLASSES}

    def strand_path(self, strand: Strand, junctions: bool = True) -> np.ndarray:
        """The strand's beads, unwrapped bead to bead.

        With ``junctions`` the junction bead at each bonded end is included,
        which is the junction-to-junction path a Z1+ export wants.
        """
        seq = list(strand.seq)
        if junctions:
            x1, x2 = strand.ends
            seq = ([x1] if x1 is not None else []) + seq + (
                [x2] if x2 is not None else [])
        return unwrap(seq, self.pos, self.box)

    def temperature(self) -> Optional[float]:
        """Kinetic temperature from the Velocities section (unit masses).

        With the 3 degrees of freedom LAMMPS removes for the centre of mass,
        as :func:`topon.simulation.protocols.gates.read_checkpoint` does.
        None when the file carries no velocities.
        """
        if not self.velocities:
            return None
        v = np.array(list(self.velocities.values()), float)
        dof = 3 * len(v) - 3
        return float((v * v).sum() / dof) if dof > 0 else None


def unwrap(seq, pos, box) -> np.ndarray:
    """Walk a bead sequence applying the minimum image bead to bead."""
    box = np.asarray(box, float)
    q = [np.asarray(pos[seq[0]], float)]
    for a in seq[1:]:
        d = pos[a] - q[-1]
        q.append(q[-1] + d - box * np.round(d / box))
    return np.asarray(q)


def _split_sections(lines):
    """Header lines and ``{section: body lines}`` of a LAMMPS data file."""
    header, sections, current = [], {}, None
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line in _SECTIONS:
            current = line
            sections[current] = []
            continue
        if current is None:
            header.append(line)
        else:
            sections[current].append(line)
    return header, sections


def read_endlinked(path, junction_type: Optional[int] = None) -> EndLinkedSystem:
    """Read an end-linked LAMMPS data file (``atom_style full``).

    By default the file must follow the convention of the module doc. With
    ``junction_type`` it may type its atoms any way it likes: the atoms of
    that type are the junctions, every connected run of the other atoms is
    a chain, and a chain's ends are found from its bonds rather than from a
    type, so a file with one bead type for all chain beads reads the same.

    Raises:
        NotEndLinked: when the file has no junction or no chain-end atoms, or
            a chain cannot be walked end to end (branched, or ends missing),
            or (with ``junction_type``) a junction sits mid-chain.
    """
    path = Path(path)
    header, sections = _split_sections(
        path.read_text(encoding="utf-8", errors="replace").splitlines())

    lo = np.zeros(3)
    hi = np.zeros(3)
    for line in header:
        p = line.split()
        if len(p) >= 4 and p[2] in ("xlo", "ylo", "zlo"):
            i = "xyz".index(p[2][0])
            lo[i], hi[i] = float(p[0]), float(p[1])
    box = hi - lo
    if not np.all(box > 0):
        raise ValueError(f"{path.name}: no box dimensions found")

    types, mol, pos = {}, {}, {}
    for line in sections.get("Atoms", []):
        p = line.split()
        a = int(p[0])
        mol[a] = int(p[1])
        types[a] = int(p[2])
        pos[a] = np.array([float(p[4]), float(p[5]), float(p[6])])
    velocities = {}
    for line in sections.get("Velocities", []):
        p = line.split()
        velocities[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))

    bonds = []
    adj = collections.defaultdict(list)
    for line in sections.get("Bonds", []):
        p = line.split()
        i, j = int(p[2]), int(p[3])
        bonds.append((i, j))
        adj[i].append(j)
        adj[j].append(i)

    if junction_type is None:
        junctions = sorted(a for a, t in types.items() if t == JUNCTION)
        ends_by_mol = collections.defaultdict(list)
        for a, t in types.items():
            if t == END:
                ends_by_mol[mol[a]].append(a)
        if not junctions or not ends_by_mol:
            raise NotEndLinked(
                f"{path.name}: no type-3 junction or no type-1 chain-end atoms. "
                f"Only the end-linked convention (1 chain end, 2 interior, "
                f"3 junction, one molecule per chain) can be read into strands "
                f"as it stands; a topon build writes it with "
                f"output.lammps_convention \"endlinked\". For another typing, "
                f"name the junction atom type (junction_type).")
    else:
        junctions, ends_by_mol = _chains_by_bonds(path, types, mol, adj,
                                                  int(junction_type))
    jset = set(junctions)

    strands = {}
    for m in sorted(ends_by_mol):
        strands[m] = _walk(path, m, ends_by_mol[m], types, adj, jset)

    G = strand_graph(junctions, strands, pos)
    G.graph["box"] = tuple(float(x) for x in box)
    return EndLinkedSystem(path=path, box=box, lo=lo, types=types, mol=mol,
                           pos=pos, bonds=bonds, adj=dict(adj),
                           junctions=junctions, strands=strands, graph=G,
                           velocities=velocities)


def _chains_by_bonds(path, types, mol, adj, junction_type):
    """Junctions and chain ends of a file that is not typed 1 / 2 / 3.

    The junctions are the atoms of ``junction_type``; a chain is a connected
    run of the other atoms, and its ends are the atoms with fewer than two
    bonded neighbours in it. Each chain keeps its molecule id when no other
    chain shares it, which is the case for a file with one molecule per
    chain, and is numbered after the largest molecule id otherwise.

    Raises:
        NotEndLinked: no atom of that type, or a junction bonded to a bead
            in the middle of a chain, which is a mid-chain crosslink and not
            an end-linked network.
    """
    junctions = sorted(a for a, t in types.items() if t == junction_type)
    if not junctions:
        raise NotEndLinked(f"{path.name}: no atom of junction type {junction_type}")
    jset = set(junctions)
    seen, runs = set(), []
    for a in sorted(types):
        if a in jset or a in seen:
            continue
        run, stack = [], [a]
        seen.add(a)
        while stack:
            b = stack.pop()
            run.append(b)
            for n in adj.get(b, []):
                if n not in jset and n not in seen:
                    seen.add(n)
                    stack.append(n)
        runs.append(sorted(run))

    owners = collections.Counter(mol[run[0]] for run in runs)
    next_id = max(mol.values()) + 1
    ends_by_chain = {}
    for run in runs:
        members = set(run)
        ends = [b for b in run
                if sum(1 for n in adj.get(b, []) if n in members) < 2]
        for b in run:
            if b in ends:
                continue
            x = next((n for n in adj.get(b, []) if n in jset), None)
            if x is not None:
                raise NotEndLinked(
                    f"{path.name}: junction {x} is bonded to bead {b} in the "
                    f"middle of a chain. That is a mid-chain crosslink, not an "
                    f"end-linked network, and has no strand reading here.")
        m = mol[run[0]]
        if owners[m] > 1 or m in ends_by_chain:
            m, next_id = next_id, next_id + 1
        ends_by_chain[m] = ends
    return junctions, ends_by_chain


def _walk(path, m, ends, types, adj, junctions=None) -> Strand:
    """Follow one chain from one end bead to the other.

    ``junctions`` is the set of junction atoms; by default the atoms of the
    convention's junction type.
    """
    if junctions is None:
        junctions = {a for a, t in types.items() if t == JUNCTION}
    if len(ends) == 1:
        e = ends[0]
        xs = [b for b in adj.get(e, []) if b in junctions]
        seq = [e]
        x1 = xs[0] if xs else None
        x2 = xs[1] if len(xs) > 1 else None
    elif len(ends) == 2:
        e1, e2 = sorted(ends)
        seq, prev, cur = [e1], None, e1
        while cur != e2:
            nxt = [n for n in adj.get(cur, []) if n != prev
                   and n not in junctions]
            if len(nxt) != 1:
                raise NotEndLinked(
                    f"{path.name}: molecule {m} is not a linear chain "
                    f"(bead {cur} has {len(nxt)} chain neighbours)")
            prev, cur = cur, nxt[0]
            seq.append(cur)
        x1 = next((b for b in adj.get(e1, []) if b in junctions), None)
        x2 = next((b for b in adj.get(e2, []) if b in junctions), None)
    else:
        raise NotEndLinked(
            f"{path.name}: molecule {m} has {len(ends)} chain-end beads")

    if x1 is None and x2 is None:
        cls = "free"
    elif x1 is None or x2 is None:
        cls = "dangling"
        if x1 is None:                         # bonded end first
            seq = seq[::-1]
            x1, x2 = x2, x1
    elif x1 == x2:
        cls = "loop"
    else:
        cls = "bridge"
    return Strand(mol=m, seq=seq, ends=(x1, x2), cls=cls)


def strand_graph(junctions, strands, pos) -> nx.MultiGraph:
    """The strand graph in the descriptor convention (see the module doc)."""
    G = nx.MultiGraph()
    for x in junctions:
        G.add_node(x, kind="junction", pos=np.asarray(pos[x], float))
    end_id = max(max(pos), max(junctions)) + 1
    n_free = 0
    for m, s in strands.items():
        x1, x2 = s.ends
        if s.cls in ("bridge", "loop"):
            G.add_edge(x1, x2, chain=m, cls=s.cls)
        elif s.cls == "dangling":
            G.add_node(end_id, kind="end", pos=np.asarray(pos[s.seq[-1]], float))
            G.add_edge(x1, end_id, chain=m, cls="dangling")
            end_id += 1
        else:
            n_free += 1
    G.graph["sol_chains"] = {"count": n_free}
    return G


def chain_statistics(system: EndLinkedSystem) -> tuple[dict, dict]:
    """Bond lengths, strand sizes and chords of a system, as numbers.

    Returns ``(scalars, per_chain)``. Every bond is scanned (minimum image),
    not a sample: a stretched bond is wherever it is. ``chain_ree`` is the
    end-to-end distance of a chain's own beads, bridges only; ``chord`` the
    junction-to-junction distance of a bridge.
    """
    box = system.box
    pos = system.pos
    if system.bonds:
        b = np.array(system.bonds)
        ids = np.fromiter(pos.keys(), int)
        at = np.zeros((ids.max() + 1, 3))
        at[ids] = np.array(list(pos.values()))
        d = at[b[:, 1]] - at[b[:, 0]]
        d -= box * np.round(d / box)
        bl = np.linalg.norm(d, axis=1)
    else:
        bl = np.zeros(0)

    order = sorted(system.strands)
    cls, ree, chord = [], [], []
    for m in order:
        s = system.strands[m]
        inner = unwrap(s.seq, pos, box)
        cls.append(s.cls)
        ree.append(float(np.linalg.norm(inner[-1] - inner[0])))
        if s.cls == "bridge":
            q = system.strand_path(s)
            chord.append(float(np.linalg.norm(q[-1] - q[0])))
    cls = np.array(cls)
    ree = np.array(ree)
    chord = np.array(chord)
    bridge = ree[cls == "bridge"]

    out = {
        "box": [float(x) for x in box],
        "density": system.density,
        "n_atoms": system.n_atoms,
        "chain_classes": system.class_counts(),
        "bond_mean": float(bl.mean()) if len(bl) else None,
        "bond_max": float(bl.max()) if len(bl) else None,
        "bonds_over_1.2": int((bl > 1.2).sum()),
    }
    if len(bridge):
        out["chain_ree_mean"] = float(bridge.mean())
        out["chain_ree2_mean"] = float((bridge ** 2).mean())
    if len(chord):
        out["chord_mean"] = float(chord.mean())
        out["chord_sd"] = float(chord.std())
        out["chord_cv"] = float(chord.std() / chord.mean())
        out["chord_pct"] = [float(x) for x in
                            np.percentile(chord, [5, 25, 50, 75, 95])]
    t = system.temperature()
    if t is not None:
        out["temperature"] = t
    return out, {"mol": np.array(order), "cls": cls, "ree": ree,
                 "chord": chord}
