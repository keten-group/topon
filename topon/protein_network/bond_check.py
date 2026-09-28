"""Stretched bonds, and bonds threaded through rings, in a LAMMPS data file.

A soft-core stage lets a strand pass through a ring (a proline or tyrosine
ring in CHARMM, a three-bead aromatic ring in Martini). Once the full force
field is on, the strand cannot get out again, and the bond and the ring
both stay stretched. :func:`check_bonds` finds those cases in any data file
topon writes or LAMMPS writes back, so a build can report them before the
run (``threaded`` in the as-built structure) and a relaxed network can be
checked after it (``python -m topon.protein_network check-bonds``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass
class LammpsData:
    box: np.ndarray
    pos: dict[int, np.ndarray]
    bonds: list[tuple[int, int, int]]                  # (type, i, j)
    bond_coeffs: dict[int, tuple[float, float]] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)


def read_lammps_data(path: str | Path) -> LammpsData:
    """Atoms (``full`` style), bonds, bond coefficients and atom comments."""
    box, pos, bonds, coeffs, labels = [], {}, [], {}, {}
    section = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        body, _, comment = line.partition("#")
        s = body.strip()
        if not s:
            continue
        if s.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            lo, hi = map(float, s.split()[:2])
            box.append(hi - lo)
            continue
        if s[0].isalpha():
            section = s.split()[0] + (" " + s.split()[1] if len(s.split()) > 1 and
                                      s.split()[1] == "Coeffs" else "")
            continue
        f = s.split()
        if section == "Atoms":
            i = int(f[0])
            pos[i] = np.array([float(v) for v in f[4:7]])
            if comment.strip():
                labels[i] = comment.strip()
        elif section == "Bonds":
            bonds.append((int(f[1]), int(f[2]), int(f[3])))
        elif section == "Bond Coeffs":
            coeffs[int(f[0])] = (float(f[1]), float(f[2]))
    return LammpsData(np.array(box), pos, bonds, coeffs, labels)


def read_bond_coeffs(settings: str | Path) -> dict[int, tuple[float, float]]:
    """``bond_coeff`` lines of a LAMMPS include (harmonic K r0)."""
    out = {}
    for line in Path(settings).read_text(encoding="utf-8").splitlines():
        t = line.split("#")[0].split()
        if len(t) >= 4 and t[0] == "bond_coeff":
            out[int(t[1])] = (float(t[2]), float(t[3]))
    return out


def small_rings(bonds, min_size: int = 3, max_size: int = 6) -> list[list[int]]:
    """Every ring of ``min_size`` to ``max_size`` atoms, in ring order."""
    g = nx.Graph((i, j) for _, i, j in bonds)
    return [c for c in nx.simple_cycles(g, length_bound=max_size) if len(c) >= min_size]


def _mic(d, box):
    return d - box * np.round(d / box)


def _wrap(x, box):
    """Into [0, box); a tiny negative coordinate can round to box itself."""
    x = np.mod(x, box)
    return np.where(x >= box, 0.0, x)


def _inside(pt, poly):
    """2D point in polygon (ray casting)."""
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def threaded_bonds(data: LammpsData, rings=None) -> list[tuple[tuple[int, int], tuple[int, ...]]]:
    """Bonds whose segment passes through the inside of a ring they are not part of."""
    from scipy.spatial import cKDTree

    rings = small_rings(data.bonds) if rings is None else rings
    if not rings:
        return []
    box = data.box
    geo = []
    for ring in rings:
        ref = data.pos[ring[0]]
        pts = np.array([ref + _mic(data.pos[a] - ref, box) for a in ring])
        cen = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - cen)
        poly = (pts - cen) @ vt[:2].T
        geo.append((set(ring), tuple(ring), cen, vt, poly,
                    float(np.max(np.linalg.norm(pts - cen, axis=1)))))
    cens = np.array([_wrap(g[2], box) for g in geo])
    tree = cKDTree(cens, boxsize=box)
    lengths = [np.linalg.norm(_mic(data.pos[j] - data.pos[i], box)) for _, i, j in data.bonds]
    reach = max(g[5] for g in geo) + max(lengths)
    out = []
    for (_, i, j), length in zip(data.bonds, lengths):
        a = data.pos[i]
        d = _mic(data.pos[j] - a, box)
        mid = _wrap(a + d / 2, box)
        for k in tree.query_ball_point(mid, reach / 2 + 1e-9):
            members, ring, cen, vt, poly, _ = geo[k]
            if i in members or j in members:
                continue
            c = a + _mic(cen - a, box)
            n = vt[2]
            denom = float(np.dot(n, d))
            if abs(denom) < 1e-12:
                continue
            s = float(np.dot(n, c - a)) / denom
            if not 0.0 <= s <= 1.0:
                continue
            hit = (a + s * d - c) @ vt[:2].T
            if _inside(hit, poly):
                out.append(((i, j), ring))
                break
    return out


def check_bonds(data_file: str | Path, *, reference: str | Path | None = None,
                settings: str | Path | None = None, factor: float = 1.25) -> dict:
    """Stretched and ring-threaded bonds of a data file, as a JSON-ready dict.

    Bond lengths are compared with r0 from the file's Bond Coeffs, or from
    ``settings`` (an include with ``bond_coeff`` lines). ``reference`` (the
    as-built data file topon wrote) supplies the atom labels a file written
    by LAMMPS lacks.
    """
    data = read_lammps_data(data_file)
    coeffs = data.bond_coeffs or (read_bond_coeffs(settings) if settings else {})
    labels = read_lammps_data(reference).labels if reference else data.labels

    def name(i):
        return labels.get(i, str(i))

    stretched = []
    if coeffs:
        for t, i, j in data.bonds:
            r = float(np.linalg.norm(_mic(data.pos[j] - data.pos[i], data.box)))
            r0 = coeffs[t][1]
            if r > factor * r0:
                stretched.append({"atoms": [i, j], "names": [name(i), name(j)],
                                  "r": round(r, 3), "r0": r0})
    threads = threaded_bonds(data)
    return {
        "data_file": Path(data_file).name,
        "n_bonds": len(data.bonds),
        "stretch_factor": factor,
        "n_stretched": len(stretched) if coeffs else None,
        "stretched": sorted(stretched, key=lambda b: -b["r"] / b["r0"]),
        "n_threaded": len(threads),
        "threaded": [{"bond": [name(i), name(j)], "ring": [name(a) for a in ring]}
                     for (i, j), ring in threads],
    }
