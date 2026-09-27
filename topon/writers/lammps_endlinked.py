"""The end-linked LAMMPS data file, written from a placement.

This is the convention of `fix bond/create` datasets: type 1 = chain-end
bead, 2 = chain interior, 3 = junction, one molecule per chain and one per
junction. It is what ``refnet.parse`` and the Z1+ exporter read, so a topon
build written this way is measured by exactly the tooling a reference
dataset is, and the two are directly comparable.

:class:`topon.writers.lammps_cg.CGWriter` writes the same convention from an
RDKit molecule, for the chemistry-stage route through the Pipeline. This one
writes it from a :class:`~topon.conformation.placement.chains.Placement` --
the bead-spring route, where the coordinates come from the conformation
stage rather than from a displacement file, and no RDKit structure is built
at all.

Molecule order is load-bearing, not cosmetic. The junctions take the first
molecules and the strands follow in placement order, because the Z1+
exporter walks molecules in sorted order: its chain *k* is
``placement.strands[k - 1]``, which is what lets a named-pair request be
read straight off a Z1+ partner list.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Atom types of the end-linked convention.
END, INTERIOR, JUNCTION = 1, 2, 3


def wrap_with_images(x, box):
    """Wrap coordinates into ``[0, L)`` and return the matching image flags.

    The two have to be produced together or they disagree. Wrapping with
    ``floor`` alone is not enough: for an ``x`` a little below zero and large
    in magnitude, ``x - L*floor(x/L)`` rounds up to exactly ``L``, which is
    outside the half-open cell every downstream reader assumes. Folding that
    point down to the lower face without also incrementing its image flag is
    how a build ends up one box-length out on reconstruction -- LAMMPS reads
    it as a bond spanning more than half the box and warns "Inconsistent
    image flags", and the Z1+ exporter reads the bead a box away from where
    it belongs.

    A dangling strand is where this shows up in practice: its free end is the
    one bead placed beyond its junction with nothing on the far side to pull
    it back, so it is the bead most likely to land on a face.

    What this cannot fix, and nothing can: a bridge that crosses the periodic
    boundary closes a cycle around the box, and a cycle with a nonzero
    winding has no consistent assignment of image flags -- one bond of it
    must disagree. Measured on a 4x4x4 cell carrying 192 bridges, 40 dangling
    strands and 6 loops (4 958 bonds), the bonds spanning more than half a box
    once unwrapped by their flags are:

    =================  =========  ============  ========
    junction rule      bridges    dangling      loops
    =================  =========  ============  ========
    first that touches       123             5         2
    first that starts         50             0         0
    =================  =========  ============  ========

    So the 50 that remain are exactly the intrinsic case, and the brief's ask
    -- consistent flags along a dangling chain -- is met. LAMMPS says
    "Inconsistent image flags" once for the 50 and then uses the minimum image
    for the bond force, which is correct; ``refnet.unwrap`` walks chains bead
    to bead by minimum image for the same reason. The flags are for reading,
    not for reconstruction.

    Returns ``(wrapped, image)`` with ``wrapped + image * L == x`` exactly.
    """
    x = np.asarray(x, float)
    L = np.asarray(box, float).reshape(3)
    image = np.floor(x / L).astype(int)
    wrapped = x - image * L
    over = wrapped >= L
    # np.where, not fancy indexing: a boolean mask over an (N, 3) array
    # flattens to the selected elements and will not broadcast against L.
    return np.where(over, wrapped - L, wrapped), image + over


def _bead_types(n: int) -> list[int]:
    """Type per bead a strand owns: its two ends are chain ends.

    Every strand class gets the same answer. A bridge gives both of its
    junction ends away and the beads next to them become the chain ends; a
    dangling strand's far end is its own last bead; a loop is a ring whose
    two ends bond to the same junction. All three are chains of the same
    length with the same two end beads, which is the point of the
    convention -- one parser reads all of them.
    """
    if n == 1:
        return [END]
    return [END] + [INTERIOR] * (n - 2) + [END]


def write_endlinked(path, placement, title: str = "topon end-linked build") -> dict:
    """Write ``placement`` as an end-linked data file.

    Returns a summary of what was written (atoms, bonds, junctions, and the
    beads that needed an image flag), for a manifest.

    Raises:
        ValueError: if the atom count does not match the bead count the box
            was sized from. The density on disk would then not be the density
            that was asked for, which is silent and ruins every comparison.
    """
    path = Path(path)
    L = np.asarray(placement.box, float).reshape(3)

    atoms: list[list] = []          # [id, mol, type, xyz]
    bonds: list[tuple] = []
    node_id: dict = {}

    junctions = sorted({p.plan.u for p in placement.strands}
                       | {p.plan.v for p in placement.strands
                          if p.plan.kind == "bridge"})
    # Where each junction sits, preferring a strand that *starts* there.
    #
    # A strand is drawn from its own junction to the nearest periodic image of
    # the other one, so its first point is the junction's canonical position
    # and its last may be a whole box away from where that junction is
    # written. Taking the first strand that merely *touches* the junction
    # picks a far end almost always -- 63 of 64 junctions on a 4x4x4 cell --
    # and for 20 of those the far end is in a different periodic image, so the
    # junction is written a box out. The wrapped coordinate is the same either
    # way, so this changes no geometry; what it fixes is the image flag, and
    # with it every bond along a dangling strand, whose free end has nothing
    # on the far side to pull it back into agreement.
    origin: dict = {}
    for s in placement.strands:
        origin.setdefault(s.plan.u, np.asarray(s.path[0], float))
    for s in placement.strands:
        if s.plan.kind == "bridge":
            origin.setdefault(s.plan.v, np.asarray(s.path[-1], float))

    for n in junctions:
        aid = len(atoms) + 1
        node_id[n] = aid
        atoms.append([aid, len(atoms) + 1, JUNCTION, origin[n]])

    mol = len(atoms)
    for s in placement.strands:
        mol += 1
        own = s.beads()
        ids = []
        for t, xyz in zip(_bead_types(len(own)), own):
            aid = len(atoms) + 1
            atoms.append([aid, mol, t, np.asarray(xyz, float)])
            ids.append(aid)
        if s.plan.kind == "bridge":
            chain = [node_id[s.plan.u]] + ids + [node_id[s.plan.v]]
        elif s.plan.kind == "dangling":
            chain = [node_id[s.plan.u]] + ids
        else:                                   # a loop closes on its anchor
            chain = [node_id[s.plan.u]] + ids + [node_id[s.plan.u]]
        bonds.extend(zip(chain[:-1], chain[1:]))

    if len(atoms) != placement.n_beads:
        raise ValueError(
            f"wrote {len(atoms)} atoms for a build sized at "
            f"{placement.n_beads}. The box was sized from the second number, "
            f"so the density on disk is not the density that was asked for. "
            f"Most likely the graph carries a junction no strand touches.")

    X = np.array([a[3] for a in atoms], float)
    Xw, img = wrap_with_images(X, L)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"{len(atoms)} atoms\n3 atom types\n"
                f"{len(bonds)} bonds\n1 bond types\n\n")
        for ax, Lx in zip("xyz", L):
            f.write(f"0.0 {Lx:.6f} {ax}lo {ax}hi\n")
        f.write("\nMasses\n\n1 1\n2 1\n3 1\n\nAtoms # full\n\n")
        for (i, m, t, _p), p, im in zip(atoms, Xw, img):
            f.write(f"{i} {m} {t} 0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{im[0]} {im[1]} {im[2]}\n")
        f.write("\nBonds\n\n")
        for k, (x, y) in enumerate(bonds, start=1):
            f.write(f"{k} 1 {x} {y}\n")

    return {"atoms": len(atoms), "bonds": len(bonds),
            "junctions": len(junctions),
            "imaged_beads": int(np.any(img != 0, axis=1).sum())}
