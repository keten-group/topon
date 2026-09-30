"""Read an atomistic build or checkpoint into strands, through the run manifest.

An atomistic data file (DREIDING or CHARMM) puts every atom in one molecule
and types atoms by chemistry, so unlike the end-linked convention it cannot
be read into strands on its own. The pipeline records which atoms are which
strand in the run manifest (``stages.strands``, written by stage 4 in LAMMPS
atom ids, which LAMMPS keeps through every stage), and this module reads a
data file with that record into the same
:class:`~topon.analysis.endlinked.EndLinkedSystem` the end-linked reader
returns. Everything built on that class -- the strand graph and its
descriptors, the Z1+ export and :func:`topon.analysis.z1plus.measure_system`
-- then works on an atomistic system unchanged.

What a strand is, for Z1+: one point per repeat unit (the repeat's first
backbone atom, Si of each Si-O pair in PDMS), with the junction atom at each
bonded end, so a DP-N bridge exports N + 2 points exactly as a bead-spring
bridge of N beads does. A dangling strand ends on its end cap's attachment
atom (the Si of the trimethylsilyl cap). The junctions are jittered by the
exporter, as for a bead-spring system. Taking one atom per repeat instead of
the whole backbone cuts each Si-O-Si corner: the O sits 0.97 A off the Si-Si
line at DREIDING's 104.5-degree angle (0.5 A at the experimental 143), and a
straight Si-Si segment passes within 0.43 A of the triangle's edges at most.
In a relaxed state no other backbone comes within 3 A of an Si-O bond, so
the cut changes no crossing there; in an unrelaxed build, with atoms still
overlapping, it can.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from topon.analysis.endlinked import (
    CLASSES, EndLinkedSystem, Strand, _split_sections, strand_graph, unwrap)
from topon.core.manifest import MANIFEST_NAME, read_manifest

#: LAMMPS ``real`` units: m v^2 in g/mol (A/fs)^2 to kcal/mol, and R.
MVV2E = 2390.0573
BOLTZ = 0.0019872067

#: g/mol per A^3 to g/cm^3.
DENSITY_UNIT = 1.66053907

#: A backbone bond this far over its equilibrium length is stretched.
BOND_TOLERANCE = 0.15


class NoStrandRecord(ValueError):
    """No run manifest with a strand record was found for this data file."""


@dataclass
class AtomisticSystem(EndLinkedSystem):
    """An atomistic data file read down to its strands.

    The strands' ``seq`` are the repeat-unit atoms (one per repeat), so
    :meth:`strand_path` gives the coarse path Z1+ reads. Real units:
    positions and the box in A, masses in g/mol, velocities in A/fs.
    """

    masses: dict = field(default_factory=dict)          # atom type -> g/mol
    bond_types: dict = field(default_factory=dict)      # (i, j) -> bond type
    bond_r0: dict = field(default_factory=dict)         # bond type -> A
    backbone_bonds: list = field(default_factory=list)  # (i, j) along strands
    record: dict = field(default_factory=dict)          # the manifest's strands
    run_dir: Optional[Path] = None

    @property
    def mass_density(self) -> float:
        """g/cm^3."""
        mass = sum(self.masses.get(t, 0.0) for t in self.types.values())
        return float(mass / np.prod(self.box) * DENSITY_UNIT)

    def temperature(self) -> Optional[float]:
        """Kinetic temperature in K from the Velocities section (real units)."""
        if not self.velocities:
            return None
        m = np.array([self.masses.get(self.types[a], 0.0) for a in self.velocities])
        v = np.array(list(self.velocities.values()), float)
        dof = 3 * len(v) - 3
        return (float((m[:, None] * v * v).sum() * MVV2E / (dof * BOLTZ))
                if dof > 0 else None)

    def bond_length(self, i, j) -> float:
        d = self.pos[j] - self.pos[i]
        d = d - self.box * np.round(d / self.box)
        return float(np.linalg.norm(d))

    def backbone_ratios(self) -> tuple[list, np.ndarray]:
        """Every backbone bond and its length over its equilibrium length."""
        pairs, ratio = [], []
        for i, j in self.backbone_bonds:
            t = self.bond_types.get((min(i, j), max(i, j)))
            r0 = self.bond_r0.get(t)
            if r0:
                pairs.append((min(i, j), max(i, j)))
                ratio.append(self.bond_length(i, j) / r0)
        return pairs, np.asarray(ratio)


# ---------------------------------------------------------------------------
# Finding the record
# ---------------------------------------------------------------------------

def find_manifest(data_path) -> Optional[Path]:
    """The run manifest with a strand record for this data file, if any.

    A run keeps its data files in ``02_Chemistry``, ``03_Conformation`` and
    ``04_Simulation`` below the run directory, so the file's own folder and
    the two above it are searched.
    """
    here = Path(data_path).resolve().parent
    for d in (here, here.parent, here.parent.parent):
        if (d / MANIFEST_NAME).exists():
            m = read_manifest(d)
            if m and "strands" in (m.get("stages") or {}):
                return d / MANIFEST_NAME
    return None


def load_strand_record(source) -> tuple[dict, Optional[Path]]:
    """The strand record and its run directory, from a dict, a manifest or a run."""
    if isinstance(source, dict):
        rec = source.get("stages", {}).get("strands", source)
        return rec, None
    p = Path(source)
    if p.is_dir():
        p = p / MANIFEST_NAME
    m = read_manifest(p.parent) if p.name == MANIFEST_NAME else None
    if m is None and p.exists():
        import json
        m = json.loads(p.read_text(encoding="utf-8"))
    rec = (m or {}).get("stages", {}).get("strands") or (m or {})
    if "strands" not in rec:
        raise NoStrandRecord(f"{source}: no strand record (stages.strands)")
    return rec, p.parent


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _settings_r0(run_dir) -> dict:
    """``bond_coeff`` equilibrium lengths from the run's settings file."""
    if run_dir is None:
        return {}
    f = Path(run_dir) / "02_Chemistry" / "system.in.settings"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*bond_coeff\s+(\d+)\s+(\S+)\s+(\S+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(3))
    return out


def _ids(span) -> list:
    if not span:
        return []
    if span.get("range"):
        lo, hi = span["range"]
        return list(range(int(lo), int(hi) + 1))
    return [int(i) for i in span.get("ids", [])]


def read_atomistic(path, strands=None) -> AtomisticSystem:
    """Read an atomistic data file into strands.

    Args:
        path: a data file of the run (the chemistry-stage file, the
            conformed one or any LAMMPS checkpoint written from it).
        strands: the strand record, a manifest or its run directory. None
            looks for the run manifest beside the data file.

    Raises:
        NoStrandRecord: no record was given and none was found.
    """
    path = Path(path)
    run_dir = None
    if strands is None:
        found = find_manifest(path)
        if found is None:
            raise NoStrandRecord(
                f"{path.name}: no run manifest with a strand record beside it. "
                f"An atomistic data file cannot be read into strands on its "
                f"own; pass the run's manifest.json (topon writes the record "
                f"in stage 4 since 0.4.0).")
        strands = found
    record, run_dir = load_strand_record(strands)

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

    masses = {}
    for line in sections.get("Masses", []):
        p = line.split()
        masses[int(p[0])] = float(p[1])
    types, pos = {}, {}
    for line in sections.get("Atoms", []):
        p = line.split()
        a = int(p[0])
        types[a] = int(p[2])
        pos[a] = np.array([float(p[4]), float(p[5]), float(p[6])])
    velocities = {}
    for line in sections.get("Velocities", []):
        p = line.split()
        velocities[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    bonds, bond_types, adj = [], {}, {}
    for line in sections.get("Bonds", []):
        p = line.split()
        t, i, j = int(p[1]), int(p[2]), int(p[3])
        bonds.append((i, j))
        bond_types[(min(i, j), max(i, j))] = t
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    bond_r0 = {}
    for line in sections.get("Bond Coeffs", []):
        p = line.split()
        bond_r0[int(p[0])] = float(p[2])
    if not bond_r0:
        bond_r0 = _settings_r0(run_dir)

    n_record = record.get("n_atoms")
    if n_record is not None and int(n_record) != len(types):
        raise NoStrandRecord(
            f"{path.name}: the strand record describes a build of {n_record} "
            f"atoms and the file has {len(types)}; it belongs to another run")

    # A strand ends on its node's attachment atom. A POSS cage has one per
    # strand, all of them one junction, so the graph groups them by node.
    node_of = {}
    for n in record.get("nodes", []):
        attach = n["attach"] if isinstance(n["attach"], list) else [n["attach"]]
        for a in attach:
            node_of[int(a)] = int(attach[0])

    strand_map, graph_strands, mol, backbone_bonds = {}, {}, {}, []
    for k, row in enumerate(record["strands"], start=1):
        j1, j2 = (row["junctions"] + [None, None])[:2]
        caps = list(row.get("free_ends") or [])
        seq = caps[:1] + list(row["repeat_heads"]) + caps[1:]
        if row.get("free_end") is not None:
            seq.append(int(row["free_end"]))
        strand_map[k] = Strand(mol=k, seq=seq, ends=(j1, j2), cls=row["cls"])
        graph_strands[k] = Strand(
            mol=k, seq=seq, cls=row["cls"],
            ends=tuple(None if j is None else node_of.get(j, j) for j in (j1, j2)))
        for a in _ids(row.get("heavy")) + _ids(row.get("hydrogens")):
            mol[a] = k
        chain = (([j1] if j1 is not None else caps[:1]) + list(row["backbone"])
                 + ([j2] if j2 is not None else
                    [row["free_end"]] if row.get("free_end") is not None else
                    caps[1:]))
        backbone_bonds.extend(zip(chain[:-1], chain[1:]))
    missing = sorted({a for s in strand_map.values()
                      for a in list(s.seq) + [e for e in s.ends if e is not None]
                      if a not in pos})
    if missing:
        raise NoStrandRecord(
            f"{path.name}: {len(missing)} strand atoms of the record are not in "
            f"the file (first {missing[:5]}); the record belongs to another build")
    junctions = sorted({j for s in graph_strands.values() for j in s.ends
                        if j is not None})
    G = strand_graph(junctions, graph_strands, pos)
    G.graph["box"] = tuple(float(x) for x in box)
    for a in types:
        mol.setdefault(a, 0)
    return AtomisticSystem(
        path=path, box=box, lo=lo, types=types, mol=mol, pos=pos, bonds=bonds,
        adj=adj, junctions=junctions, strands=strand_map, graph=G,
        velocities=velocities, masses=masses, bond_types=bond_types,
        bond_r0=bond_r0, backbone_bonds=backbone_bonds, record=record,
        run_dir=run_dir)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def backbone_statistics(system: AtomisticSystem,
                        tolerance: float = BOND_TOLERANCE) -> dict:
    """Backbone bonds against their equilibrium lengths, strands and chords.

    Every backbone bond (junction to junction, through the bridge atoms) is
    read against the ``r0`` of its bond type; ``stretched`` counts those
    more than ``tolerance`` over it. Chords are junction to junction, bridges
    only, in A.
    """
    _pairs, ratio = system.backbone_ratios()
    chords, points = [], []
    for s in system.strands.values():
        q = system.strand_path(s)
        points.append(len(q))
        if s.cls == "bridge":
            chords.append(float(np.linalg.norm(q[-1] - q[0])))
    chords = np.asarray(chords)
    out = {"box": [float(x) for x in system.box],
           "n_atoms": system.n_atoms,
           "mass_density": system.mass_density,
           "chain_classes": system.class_counts(),
           "points_per_strand": {int(k): int(v) for k, v in zip(
               *np.unique(points, return_counts=True))} if points else {},
           "backbone_bonds": int(len(ratio)),
           "backbone_ratio_mean": float(ratio.mean()) if len(ratio) else None,
           "backbone_ratio_min": float(ratio.min()) if len(ratio) else None,
           "backbone_ratio_max": float(ratio.max()) if len(ratio) else None,
           "backbone_stretched": int((ratio > 1.0 + tolerance).sum()),
           "tolerance": tolerance}
    if len(chords):
        out["chord_mean"] = float(chords.mean())
        out["chord_sd"] = float(chords.std())
    t = system.temperature()
    if t is not None:
        out["temperature"] = t
    return out


def export_backbone(system: AtomisticSystem, seed: int = 0):
    """The Z1+ input of an atomistic system: ``(chains, classes, strand ids)``.

    One point per repeat unit plus the junction atoms, the junctions
    jittered; see the module doc.
    """
    from topon.analysis.z1plus import export_system
    return export_system(system, seed=seed)


def measure_atomistic(path, strands=None, partners: bool = True, config=None,
                      seed: int = 0) -> tuple[dict, dict]:
    """Backbone statistics and Z1+ of one atomistic checkpoint.

    Returns ``(record, per_chain)`` in the layout of
    :func:`topon.analysis.z1plus.measure_checkpoint`; a record carries
    ``z1_error`` instead of ``z1`` when Z1+ cannot run.
    """
    from topon.analysis.z1plus import (
        Z1PlusFailed, Z1PlusUnavailable, measure_system)

    system = read_atomistic(path, strands)
    record = {"data": str(path), **backbone_statistics(system)}
    per_chain = {"mol": np.array(sorted(system.strands)),
                 "cls": np.array([system.strands[m].cls
                                  for m in sorted(system.strands)])}
    try:
        z, res = measure_system(system, partners=partners, config=config,
                                seed=seed)
        record["z1"] = z
        per_chain["Z"] = res.Z
        per_chain["Lpp"] = res.Lpp
    except (Z1PlusUnavailable, Z1PlusFailed) as exc:
        record["z1_error"] = str(exc)
    return record, per_chain


__all__ = ["AtomisticSystem", "NoStrandRecord", "BOND_TOLERANCE", "CLASSES",
           "backbone_statistics", "export_backbone", "find_manifest",
           "load_strand_record", "measure_atomistic", "read_atomistic", "unwrap"]
