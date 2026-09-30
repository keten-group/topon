"""Did two backbone bonds pass through each other? A crossing detector.

Z1+ counts kinks of the shortest paths between the current junction
positions, so it moves as chains settle even where nothing crosses (on a
DP-10 test network by a few of its 5 to 14 kinks). Whether an entanglement
state survived a relaxation is a question about topology, and topology can
only change when one strand passes through another. This module watches for
exactly that in a trajectory: two bond segments, neither sharing an atom
with the other, that pass through each other between two frames. The
geometry (every atom moving in a straight line between two frames, a
passage being a root of the coplanarity cubic at which the two segments
meet) is :mod:`topon.conformation.segments`, which the atomistic backbone
clearance uses to keep its own moves from doing the same thing.

The frames must be close enough together that an atom moves a fraction of a
bond between them; ``simulation.backbone_dump_every`` 10 does that for the
atomistic stages (the capped stages move an atom at most 0.5-1 A in 10
steps). Coordinates must be unwrapped and continuous in time (LAMMPS ``xu yu
zu``); pairs are taken under the minimum image of the earlier frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from topon.conformation.segments import passing_pairs


@dataclass
class Frame:
    step: int
    box: np.ndarray          # (3,) edge lengths
    ids: np.ndarray          # (n,) atom ids, sorted
    xyz: np.ndarray          # (n, 3) unwrapped positions


@dataclass
class Crossing:
    """One passage of two bonds through each other."""

    transition: int          # between frame transition and transition + 1
    step: int                # timestep of the later frame
    bonds: tuple             # the two bonds, as (atom, atom) LAMMPS ids
    strands: tuple           # their strands (record numbers, 1-based)
    kind: str                # "self", "shared" (strands share a junction), "apart"
    tau: float               # where between the two frames


@dataclass
class CrossingReport:
    crossings: list = field(default_factory=list)
    transitions: int = 0
    max_step: float = 0.0    # largest atom displacement between two frames, A

    def counts(self) -> dict:
        out = {"self": 0, "shared": 0, "apart": 0}
        for c in self.crossings:
            out[c.kind] += 1
        return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def dump_path(sim_dir, stage: str) -> Optional[Path]:
    """A stage's backbone dump, as LAMMPS wrote it or gzipped, if either is there."""
    for name in (f"traj_{stage}.lammpstrj", f"traj_{stage}.lammpstrj.gz"):
        path = Path(sim_dir) / name
        if path.exists():
            return path
    return None


def read_lammpstrj(path) -> Iterator[Frame]:
    """Frames of a LAMMPS text dump with ``id`` and three coordinate columns.

    A ``.gz`` file is read through gzip.
    """
    path = Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    for block in text.split("ITEM: TIMESTEP")[1:]:
        lines = block.strip().splitlines()
        step = int(lines[0].split()[0])
        n = int(lines[2].split()[0])
        lo_hi = [tuple(float(x) for x in lines[i].split()[:2]) for i in (4, 5, 6)]
        box = np.array([hi - lo for lo, hi in lo_hi])
        cols = lines[7].split()[2:]
        data = np.array(" ".join(lines[8:8 + n]).split(), float).reshape(n, len(cols))
        order = np.argsort(data[:, cols.index("id")])
        data = data[order]
        xyz_cols = [cols.index(c) for c in
                    (("xu", "yu", "zu") if "xu" in cols else ("x", "y", "z"))]
        yield Frame(step=step, box=box, ids=data[:, cols.index("id")].astype(int),
                    xyz=data[:, xyz_cols])


def backbone_bonds(record) -> tuple[np.ndarray, np.ndarray]:
    """Every backbone bond of a strand record, with its strand number.

    Junction to junction through the bridge atoms, the end-cap Si of a
    dangling strand included. Returns ``(bonds (m, 2) LAMMPS ids, strand (m,))``.
    """
    bonds, strand = [], []
    for k, row in enumerate(record["strands"], start=1):
        j1, j2 = (row["junctions"] + [None, None])[:2]
        caps = list(row.get("free_ends") or [])
        chain = (([j1] if j1 is not None else caps[:1]) + list(row["backbone"])
                 + ([j2] if j2 is not None else
                    [row["free_end"]] if row.get("free_end") is not None else
                    caps[1:]))
        for a, b in zip(chain[:-1], chain[1:]):
            bonds.append((a, b))
            strand.append(k)
    return np.array(bonds, int).reshape(-1, 2), np.array(strand, int)


def junction_sharing(record) -> set:
    """Pairs of strands (1-based, i < j) that end on the same junction."""
    by_junction: dict = {}
    for k, row in enumerate(record["strands"], start=1):
        for j in {x for x in row["junctions"] if x is not None}:
            by_junction.setdefault(j, []).append(k)
    out = set()
    for ks in by_junction.values():
        out |= {(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]}
    return out


def designed_pairs(record) -> dict:
    """The designed entanglements of a strand record, ``{(k, l): windings}``.

    ``k < l``, strand numbers 1-based in record order, from the ``partner``
    the pipeline records for a strand drawn with ``assignment.entanglements``.
    """
    out = {}
    for k, row in enumerate(record["strands"], start=1):
        partner = row.get("partner")
        if partner:
            out[(min(k, int(partner)), max(k, int(partner)))] = int(row.get("windings", 1))
    return out


# ---------------------------------------------------------------------------
# Detecting
# ---------------------------------------------------------------------------

def find_crossings(frames: Iterable[Frame], bonds: np.ndarray, strand: np.ndarray,
                   sharing: Optional[set] = None) -> CrossingReport:
    """Every passage of two backbone bonds through each other along ``frames``.

    Bonds that share an atom are never a pair. ``sharing`` (strand pairs that
    end on the same junction) classes a passage between two of them as
    ``shared`` rather than ``apart``.
    """
    sharing = sharing or set()
    report = CrossingReport()
    prev = None
    for f in frames:
        if prev is None:
            row = {int(a): i for i, a in enumerate(f.ids)}
            try:
                bi = np.array([[row[a], row[b]] for a, b in bonds])
            except KeyError as exc:
                raise ValueError(f"bond atom {exc} is not in the trajectory "
                                 f"(dump the backbone atom types)") from exc
            prev = f
            continue
        if len(f.ids) != len(prev.ids) or not np.array_equal(f.ids, prev.ids):
            raise ValueError(f"frame at step {f.step} has other atoms than the one before")
        i, j, tau, big = passing_pairs(prev.xyz, f.xyz, bi, bonds, prev.box)
        report.max_step = max(report.max_step, big)
        for a, b, t in zip(i, j, tau):
            sa, sb = int(strand[a]), int(strand[b])
            kind = ("self" if sa == sb else
                    "shared" if (min(sa, sb), max(sa, sb)) in sharing else "apart")
            report.crossings.append(Crossing(
                transition=report.transitions, step=f.step,
                bonds=(tuple(int(x) for x in bonds[a]),
                       tuple(int(x) for x in bonds[b])),
                strands=(sa, sb), kind=kind, tau=float(t)))
        report.transitions += 1
        prev = f
    return report


def crossings_of_run(run_dir, stages=("stage1", "stage2", "stage3"),
                     sim: str = "04_Simulation") -> dict:
    """Passages per stage of a run dumped with ``simulation.backbone_dump_every``.

    Reads ``traj_<stage>.lammpstrj`` (or its ``.gz``) from the run's simulation folder and the
    strand record from its manifest. Returns ``{stage: CrossingReport}``.
    """
    from topon.analysis.atomistic import load_strand_record

    run_dir = Path(run_dir)
    record, _ = load_strand_record(run_dir / "manifest.json")
    bonds, strand = backbone_bonds(record)
    sharing = junction_sharing(record)
    out = {}
    for stage in stages:
        path = dump_path(run_dir / sim, stage)
        if path is not None:
            out[stage] = find_crossings(read_lammpstrj(path), bonds, strand, sharing)
    return out
