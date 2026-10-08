"""Atomistic build geometry: each strand's backbone drawn as a chain.

The default atomistic placement spaces every heavy atom of a strand (methyls
included) evenly on the straight chord and leaves the rest to the pendant
pass, so a built PDMS network has Si-O bonds anywhere from 0.07 to 5 A
(median 0.57 against r0 1.587 on a 4x4x4 three-shell DP-10 cell) and C-H
bonds of 0.3 A. The soft first stage of the relaxation does all the work
of making it a molecule. That is what this module replaces, when
``conformation.atomistic_placement`` asks for it.

Here the backbone of every strand (junction to junction through the bridge
atoms, Si-O-Si-O... for PDMS) is drawn with the same path routines as the
bead-spring build (:mod:`topon.conformation.paths`): ``straight``, ``meander``
or ``walk``, or ``coil``, which only this route has (see "A Z target" below). The routines work in units of the bead-spring design bond
(0.97), so the path is drawn in units where the strand's mean backbone bond
is 0.97 and scaled back; the shape knobs (``meander_waves``, ``min_bond``,
``min_self_separation``, ``path_jitter``) then mean the same shape on both
routes. Every backbone bond is drawn at the strand's mean equilibrium
length, which is exact for PDMS (all Si-O) and within 4 % of each bond for
PEG (C-O against C-C); spacing points along the arc by each bond's own
length would cut the corners of a zigzag or a walk. Every other atom
(methyls, hydrogens, end-cap methyls, junction caps) is placed from its
already-placed neighbour at its bond length, in tetrahedral directions away
from the bonds that atom already has, except a ring that hangs from one
atom (a phenyl), which is placed whole as a flat regular polygon at its
bond length (:func:`place_substituents`).

A ring the backbone runs through
--------------------------------
A backbone is the shortest path from a repeat's head to its tail, so in a
para-phenylene between Si and O it runs along one side of the ring (ipso,
ortho, meta, para) and the other two carbons are off it. Before 0.4.5 the
settle held only bonds and 1-3 distances on the path, so the four ring
carbons came out at any twist (on the 5x5x5 sample graph at DP 3, a
dihedral of 1 to 179 degrees, ipso to para 2.34 to 3.75 A against 2.78),
and the tree walk placed the two off-path carbons from opposite ends,
leaving the bond between them anywhere from 1.2 to 6 A. Now the settle takes
every such ring whole (:func:`ring_windows`): its off-path atoms join the
settle with their bonds and 1-3 distances, which a flat ring alone meets,
and each ring with the backbone atom on each side of it is held to one
plane. A hydrogen or methyl group that would put a bond through such a
ring's face is drawn again (up to 24 times; a ring carbon's hydrogen, set
by its two ring bonds, cannot be), and a bond through one is counted, not
moved (see USAGE). A fused ring system keeps the tree walk.

The bead-spring guard carries over with one change. A bead-spring strand
goes straight when its chord reaches 0.97 of its contour, the sum of its
bonds. An atomistic backbone cannot reach its contour at all: at its
equilibrium bond angles its two ends are at most the sum of the
next-nearest-neighbour distances apart, which for DREIDING PDMS (Si-O-Si
104.5 degrees, O-Si-O 109.5) is 0.79 of the contour. So the guard reads
the chord against that extended length (:func:`extended_length`): at 0.97
of it the strand is drawn straight, and a chord beyond the contour itself
cannot be built without stretching every bond, which is reported and warned
about. "Straight" on this route is the planar zigzag between the two
junctions at bond length (:func:`~topon.conformation.paths.zigzag`); the
bead-spring straight chord would put a slack strand's bonds at a fraction
of r0.

Strands are drawn one at a time, so two of them can be drawn through the
same point. On the DP-30 meander build (36,248 atoms, 0.97 g/cm3) 169 pairs
of backbone bonds of different strands were within 0.25 A of each other and
1,062 within 1 A, and the hard-backbone first stage then pushed 59 pairs
through each other in its first 160 steps: 57 of them had started within 1
A (median 0.09 A). Two bonds that close have no side to be pushed apart on,
so which way they go is an accident of the first few steps. The paths also
carry bead-spring bond angles, and closing them all at once in stage 1
drags strands through each other too. :func:`settle_backbones` opens every
close pair to ``clearance`` and brings every bond and angle to equilibrium
before anything else is placed, and puts back any move that takes one bond
through another. It is the atomistic
:func:`~topon.conformation.placement.chains.separate_coincident`, read on
bonds instead of beads: at Si-O 1.6 A and a hard core of 3 A, two bonds
cross with their four atoms 1.2 A apart or more, so a clearance read on
atoms would not see them.

A Z target
----------
A strand can only hold another if that one threads the loop the strand
makes with its own chord, which is what Z1+ reads as a kink. So Z at the
build is set by how far each strand strays from its chord, and ``coil``
(:func:`~topon.conformation.paths.coil_chain`) is the shape whose stray is a
number: a helix about the chord at ``radius`` A, with as many turns as the
contour needs. On the DP-30 PDMS network Z1+ per bridge rises with the radius
without a step (0.23 at 2 A, 1.49 at 6, 3.15 at 10), where the meander's wave
count moves it in jumps (its fold gate halves the waves strand by strand,
and 2.5 waves asked draws three different shapes). The settled build keeps
the coil's Z (1.487 -> 1.485 at 6 A, 3.15 -> 3.09 at 10) and the
hard-backbone deck keeps the build's, so a target is met on the build:
:func:`coil_radius_for` searches the radius against a measurement the
caller passes in, drawing the whole network at each radius from the same
seed, so the measured Z moves only with the radius. No table and no fit.

Nothing here knows about chemistry or force fields. The caller passes the
bonded neighbours of every atom, the equilibrium length of every bond and
angle, and the strands with their end positions; the pipeline gets those
from the data file it has just written.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .paths import (STRAIGHT_AT, Clearance, bond_lengths, bridging_walk, chord_side,
                    closed_meander, coil_chain, meander_chain, route_through,
                    self_contact, shared_chords, straight_chain, zigzag)
from .segments import (MEET, bond_ends, passing_pairs, segment_closest,
                       segments_through_ring, segments_through_rings, signed_volume)

#: The bead-spring design bond. Paths are drawn in units where the backbone
#: bond is this long, so every shape knob keeps its bead-spring meaning.
BEAD_BOND = 0.97

#: Fraction of the extended length at which a strand is drawn straight: the
#: bead-spring ``straight_at``, read against the longest the backbone can be
#: at its equilibrium angles rather than against its contour.
TAUT = 0.97

SHAPES = ("straight", "meander", "walk", "coil")

#: How the meander draws the strands of a secondary loop (two bridges between
#: the same two junctions): on opposite sides of their chord, or each as any
#: other strand (``conformation.parallel_strands``, as on the bead-spring
#: route).
PARALLEL_STRANDS = ("opposite", "together")

#: The closest two backbone bonds may be drawn, in A, unless they share an
#: atom or sit within two bonds of each other along one strand.
CLEARANCE = 1.5

#: How far past the clearance a short pair is pushed, as a fraction of it.
OVERSHOOT = 0.1

#: Two bonds closer than this (A) are touching: they have no side to keep.
TOUCH = 0.01

#: How far outside a designed braid's radius (A) every other backbone atom is
#: kept (:func:`clear_braids`).
BRAID_CLEARANCE = 1.5

#: Angles a strand is tried at about its chord to clear a braid.
BRAID_TURNS = 36

#: The round cap of :func:`settle_backbones`, for every build (since 0.4.5;
#: every build stopped at 400 before). On the DP-30 smoke network the
#: settle had converged by round 400 on 1 of 12 seeds with six designed
#: pairs and on none of 12 without, and the pairs it left closer than the
#: clearance were
#: whichever happened to be short at that round: near the end the count
#: goes up and down by a few as the last pairs are parted and the bonds and
#: angles pull them back (seed 7: 0 at round 350, 2 at 400, 4 at 450). Run
#: on, 24 builds of it (18 seeds without designed pairs, 6 with) converge in
#: 411 to 1137 rounds, and in 403 to 1166 with their secondary loops drawn
#: on opposite sides. A settle that converges sooner stops where it does;
#: one stopped here keeps pairs within a few hundredths of an angstrom of
#: the clearance, as the slowest of those 24 had at rounds 600 to 1000.
SETTLE_ROUNDS = 1500

#: The largest ring :func:`place_substituents` places whole, as a flat
#: regular polygon.
RING_MAX = 8

#: Sweeps per round and the round cap of :func:`settle_backbones` for a
#: build whose backbones run through rings; every other build keeps
#: 8 and :data:`SETTLE_ROUNDS`. A whole ring in the settle is a stiff block
#: of a dozen terms that the averaged corrections move a little per sweep,
#: and a strand of para-phenylenes relaxes the bonds between its rings
#: slowly: on the 5x5x5 sample graph at DP 3 four study names stopped at
#: 1500 rounds of 8 sweeps with bonds up to 8 % long, and two of them, rerun
#: at 24 sweeps, converged in 2,060 and 2,071 rounds (9 minutes each).
RING_SWEEPS = 24
RING_SETTLE_ROUNDS = 3000


#: How far outside a cage's core (the sphere through its ring atoms, A) the
#: backbone atoms of the strands not bonded to it are kept, as drawn
#: (:func:`clear_bodies`) and as settled. The bond clearance alone
#: lets a backbone atom sit over the opening of a face, about 2 A from its
#: bonds, where a tiny crowded build (a bare cage on DP-3 strands) had a
#: methyl's C-H through the face; the methyls and hydrogens are then kept
#: out by :func:`place_substituents`. 2.5 A, the reach of a methyl
#: hydrogen, kept them out too, but squeezed that build's backbone bonds to
#: 0.84 r0 against 0.94 at 1.5.
BODY_CLEARANCE = 1.5

#: How far out from a backbone atom (A) :func:`settle_backbones` reads the
#: two tetrahedral positions off its two backbone bonds, where
#: :func:`place_substituents` puts its first two side atoms (a PDMS Si's
#: methyls, DREIDING Si-C 1.697 A).
SIDE_BOND = 1.7

#: How far outside a cage's core (A) the settle keeps those two positions, a
#: C-H bond. Every face lies inside the core, so no bond of a methyl whose
#: carbon is that far out can pass through one. A backbone atom just outside
#: :data:`BODY_CLEARANCE` can point a methyl straight at the cage. On the
#: bare cage test network with a designed pair, a strand passing 4.2 A from
#: the centre had a methyl carbon 2.8 A from it and a C-H through a face,
#: and :func:`place_substituents` has nothing to turn there, since the two
#: backbone bonds fix both methyl directions.
SIDE_CLEARANCE = 1.1

#: How far (A) :func:`clear_bodies` keeps the backbone bonds of a strand, as
#: drawn, from the bonds of the arms of a cage it is not bonded to (the
#: settle's bond clearance). An AM0270 cap's arms reach 9 A from its
#: centre, past the core's keep-out sphere, and a strand drawn between them
#: on a neighbouring site is held there by bonds that do not move. On the
#: POSS demo config's own generated network (25 caps) 16 to 20 of 194
#: strands came within 1 A of another cage's arm atoms, and the settle
#: stopped at its round cap on 5 of 6 such networks, 0 to 4 pairs short and
#: angles up to 25 degrees off.
ARM_CLEARANCE = CLEARANCE

#: Turns tried about the strand direction for a cage with one strand.
BODY_TWISTS = 36

_TET = np.deg2rad(109.4712)


@dataclass
class StrandSpec:
    """One strand to draw.

    ``backbone`` is the atoms between the two end atoms, in order from the
    start. ``start`` and ``end`` are the end atoms' positions in A, with
    ``end`` already in the image nearest ``start``; a loop has ``end`` equal
    to ``start``. ``r0`` has one entry per backbone bond (``len(backbone) +
    1``) and ``theta0`` one per backbone atom, in degrees.
    """

    edge: tuple
    cls: str
    start_atom: int
    end_atom: int
    backbone: list
    start: np.ndarray
    end: np.ndarray
    r0: list
    theta0: list
    away_from: Optional[list] = None
    path: Optional[np.ndarray] = None     # interior points to use as drawn
    #: A held atom before the start (or after the end) that the settle reads
    #: the strand's first (last) angle from: ``(atom, xyz, r0, theta0)``, the
    #: bond's r0 and the angle at the start (end) atom in degrees. A strand
    #: bonded to a cage starts at the atom the cage holds on its stub, and
    #: this is the cage's attachment atom.
    lead_start: Optional[tuple] = None
    lead_end: Optional[tuple] = None
    #: Waypoints (A) taking a strand whose chord runs through a cage's core
    #: round it, and the cage atoms (A) its legs keep off
    #: (:func:`cage_detour`).
    via: Optional[list] = None
    keep_off: Optional[np.ndarray] = None


@dataclass
class PlacementReport:
    """What the placement did, per strand and in total."""

    shape: str
    strands: list = field(default_factory=list)
    settle: Optional[dict] = None
    #: What :func:`clear_braids` found and did, when there were braids.
    braids: Optional[dict] = None
    #: The strands of secondary loops drawn on opposite sides of their
    #: chord, when there were any (``parallel_strands="opposite"``).
    parallel: Optional[dict] = None
    #: What :func:`clear_bodies` found and did, when there were cages.
    cages: Optional[dict] = None
    #: Backbone positions before and after every round of the settling
    #: pass, ``(ids, [xyz, ...])``, for a caller to check that no bond went
    #: through another on the way (:mod:`topon.analysis.crossings`).
    frames: Optional[tuple] = None
    #: The rings the backbones run through and how they were closed
    #: (:func:`close_path_rings`), when there were any.
    rings: Optional[dict] = None
    #: The rings that hang from one atom (a phenyl), how they were turned
    #: and the bonds through them as placed, when there were any.
    pendant_rings: Optional[dict] = None

    def summary(self) -> dict:
        rows = self.strands
        ratio = np.array([r["chord_over_extended"] for r in rows
                          if r.get("chord_over_extended") is not None])
        routines = {}
        for r in rows:
            routines[r["routine"]] = routines.get(r["routine"], 0) + 1
        out = {
            "shape": self.shape,
            "strands": len(rows),
            "routines": routines,
            "taut": int(sum(1 for r in rows if r.get("taut"))),
            "over_contour": int(sum(1 for r in rows if r.get("over_contour"))),
            "chord_over_extended": (
                {"mean": float(ratio.mean()), "max": float(ratio.max())}
                if len(ratio) else None),
            "backbone_bond_ratio": _ratio_stats(rows),
            **_drawn_shapes(rows),
        }
        if self.braids is not None:
            out["braids"] = self.braids
        if self.parallel is not None:
            out["parallel"] = self.parallel
        if self.rings is not None:
            out["rings"] = self.rings
        if self.pendant_rings is not None:
            out["pendant_rings"] = self.pendant_rings
        if self.cages is not None:
            out["cages"] = self.cages
        out["settle"] = self.settle
        return out


def _drawn_shapes(rows) -> dict:
    """What the shape knobs came out as, strand by strand.

    The meander's wave count is a request its fold gate halves strand by
    strand (on the DP-30 PDMS network 6 waves asked drew 89 strands with 3
    and 25 with 0.75), and a coil too wide for a strand's slack is narrowed.
    """
    out = {}
    waves = [r["waves"] for r in rows if r.get("routine") == "meander" and "waves" in r]
    if waves:
        counts = {}
        for w in waves:
            counts[f"{w:g}"] = counts.get(f"{w:g}", 0) + 1
        out["waves_drawn"] = dict(sorted(counts.items(), key=lambda kv: -float(kv[0])))
    radii = np.array([r["radius"] for r in rows if r.get("routine") == "coil"])
    if len(radii):
        asked = next(r.get("radius_asked") for r in rows if r.get("routine") == "coil")
        out["coil_radius"] = {"asked": asked,
                              "min": float(radii.min()), "max": float(radii.max()),
                              "narrowed": int(sum(1 for r in rows if r.get("routine") == "coil"
                                                  and r.get("narrowed")))}
    return out


def _ratio_stats(rows) -> Optional[dict]:
    lo = [r["bond_ratio_min"] for r in rows if "bond_ratio_min" in r]
    hi = [r["bond_ratio_max"] for r in rows if "bond_ratio_max" in r]
    if not lo:
        return None
    return {"min": float(min(lo)), "max": float(max(hi))}


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def extended_length(r0, theta0) -> float:
    """The farthest the backbone's two ends can be at its equilibrium geometry.

    Atoms two bonds apart are a fixed distance apart once the two bonds and
    the angle between them are, ``sqrt(a^2 + b^2 - 2ab cos theta)``, so the
    end-to-end distance can be no more than the sum of those distances along
    the even atoms, or along the odd ones (plus the one bond each path leaves
    over at an end). This is the smaller of the two sums. For equal bonds and
    angles it is the planar zigzag exactly. For DREIDING PDMS it is the chain
    of Si-Si distances across the 104.5-degree Si-O-Si angle, 0.79 of the
    contour at any length: the Si atoms can all lie on one line, with the O
    atoms off it, while the planar all-trans chain with its two unequal angles
    curls into a ring and would read a long strand as taut. ``r0`` has one
    entry per bond, ``theta0`` one per interior atom, in degrees.
    """
    r0 = np.asarray(r0, float)
    theta = np.deg2rad(np.asarray(theta0, float))
    n = len(r0)
    if n == 0:
        return 0.0
    if n == 1:
        return float(r0[0])
    v = np.sqrt(r0[:-1] ** 2 + r0[1:] ** 2 - 2.0 * r0[:-1] * r0[1:] * np.cos(theta))
    even = v[0::2].sum() + (r0[-1] if n % 2 else 0.0)
    odd = r0[0] + v[1::2].sum() + (0.0 if n % 2 else r0[-1])
    return float(min(even, odd))


# ---------------------------------------------------------------------------
# Drawing a backbone
# ---------------------------------------------------------------------------

def draw_backbone(spec: StrandSpec, shape: str, rng, *, waves: float = 6.0,
                  min_bond: float = 0.85, min_sep: Optional[float] = None,
                  jitter: float = 0.02, shared=None,
                  radius: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """The backbone atoms' positions, in order, and what drew them.

    Returns the interior points only (``len(spec.backbone)`` of them): the
    two end atoms are junctions and stay where the graph put them.
    ``radius`` is the coil's, in A, and only ``coil`` reads it.

    ``shared`` is ``(chord, rank, count, frames)`` for a strand that shares
    its chord with another (:func:`~topon.conformation.paths.shared_chords`,
    from :func:`place_network`): drawn as a meander, it goes on its side of
    the chord (:func:`~topon.conformation.paths.chord_side`,
    ``meander_chain(..., side=...)``), the side taking the number the
    meander's turn would have taken. A taut strand goes straight, and a
    chord the meander would draw straight keeps that draw; both are
    reported with a bow of 0, left on the chord. So is a sided meander that
    falls back to a walk, which then draws after the side's number, where
    the drawing without sides drew without it (not seen on any build).
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown atomistic placement {shape!r} "
                         f"(expected one of {', '.join(SHAPES)})")
    if shape == "coil" and (radius is None or radius < 0):
        raise ValueError("the coil placement needs a radius in A "
                         "(conformation.atomistic_coil_radius, or a target_Z)")
    r0 = np.asarray(spec.r0, float)
    n = len(r0)
    contour = float(r0.sum())
    k = BEAD_BOND / float(r0.mean())          # A -> bead units
    a = np.asarray(spec.start, float)
    b = np.asarray(spec.end, float)
    info = {"edge": list(spec.edge), "cls": spec.cls, "n_bonds": n,
            "contour": contour}

    detour = None
    if spec.path is None and spec.via is not None:
        # round a cage its chord runs through, by the deterministic legs of
        # route_through, kept off the cage's atoms
        try:
            off = (None if spec.keep_off is None or not len(spec.keep_off) else
                   Clearance(np.asarray(spec.keep_off, float) * k, radius=CLEARANCE * k))
            detour = route_through(a * k, b * k, [np.asarray(w, float) * k for w in spec.via],
                                   n, BEAD_BOND, avoid=off) / k
        except ValueError:
            info["detour"] = "too short"
    if spec.path is not None:
        # An entangled pair keeps the path its method drew for it.
        interior = np.asarray(spec.path, float)
        full = np.vstack([a, interior, b])
        info["routine"] = "entangled"
    elif detour is not None:
        full = detour
        info["routine"] = "detour"
        info["chord"] = float(np.linalg.norm(b - a))
    elif spec.cls == "loop":
        if n < 3:
            raise ValueError(f"primary loop {spec.edge} has {n} backbone bonds; "
                             f"a ring needs at least 3")
        interior = closed_meander(a * k, n, bond=BEAD_BOND, rng=rng,
                                  away_from=spec.away_from) / k
        full = np.vstack([a, interior, a])
        info["routine"] = "closed_meander"
    else:
        chord = float(np.linalg.norm(b - a))
        extended = extended_length(r0, spec.theta0)
        info.update(chord=chord, extended=extended,
                    chord_over_extended=chord / extended if extended else None,
                    taut=bool(extended and chord >= TAUT * extended),
                    over_contour=bool(chord > contour))
        if info["over_contour"]:
            warnings.warn(
                f"strand {spec.edge}: chord {chord:.2f} A is longer than its "
                f"backbone contour {contour:.2f} A, so every bond is built "
                f"stretched; lower the neighbour cutoff, raise the DP or the "
                f"density", RuntimeWarning, stacklevel=2)
        use = "straight" if info["taut"] else shape
        if shared is not None:
            info["bow"] = 0.0       # left on the chord unless drawn apart below
        if use == "straight" and info["over_contour"]:
            # Nothing reaches: the chord at equal, stretched bonds.
            full = straight_chain(a * k, b * k, n, jitter=0.0) / k
        elif use == "straight":
            # The atomistic straight line is the planar zigzag at bond
            # length: a straight chord shorter than the contour would need
            # bonds shorter than r0 (0.57 r0 at DP 10 on a 20 A chord),
            # and at the taut end the zigzag is as straight as bond
            # angles near their equilibrium allow. On a slack strand it
            # pays with its angles instead (62 degrees at DP 10 on a 19 A
            # chord, 28 at DP 30 on 26 A), so it is a shape for taut
            # strands, where the guard sends them.
            full = zigzag(a * k, b * k, n, BEAD_BOND, hint=rng.normal(size=3)) / k
        elif use == "meander":
            side = None
            if (shared is not None and float(np.linalg.norm(b * k - a * k))
                    < STRAIGHT_AT * (n * BEAD_BOND)):
                # meander_chain's own straight test, on the same numbers: a
                # chord it would draw straight keeps the draw it always took
                key, rank, count, frames = shared
                side = chord_side(b - a, rank, count, frames, key, rng)
            full, drawn = meander_chain(
                a * k, b * k, n, bond=BEAD_BOND, rng=rng, waves=waves,
                min_bond=min_bond, min_sep=1.0 if min_sep is None else min_sep,
                jitter=jitter, side=side)
            full = full / k
            use = drawn["routine"]
            if use == "meander":
                info["waves"] = float(drawn["waves"])
            if side is not None and use == "meander":
                info["bow"] = float(drawn.get("bow", 0.0)) / k
        elif use == "coil":
            full, drawn = coil_chain(
                a * k, b * k, n, float(radius) * k, bond=BEAD_BOND, rng=rng,
                min_sep=1.0 if min_sep is None else min_sep)
            full = full / k
            use = drawn["routine"]
            if use == "coil":
                info.update(radius=drawn["radius"] / k, radius_asked=float(radius),
                            turns=drawn["turns"],
                            narrowed=bool(drawn["radius"] / k < float(radius) - 1e-6))
        else:
            full = bridging_walk(a * k, b * k, n, BEAD_BOND, rng) / k
        info["routine"] = use

    ratio = bond_lengths(full) / r0
    info["bond_ratio_min"] = float(ratio.min())
    info["bond_ratio_max"] = float(ratio.max())
    info["self_contact"] = float(self_contact(full, closed=spec.cls == "loop"))
    return full[1:-1], info


# ---------------------------------------------------------------------------
# A Z target: the coil radius, searched against a measurement
# ---------------------------------------------------------------------------

@dataclass
class RadiusSearch:
    """Where :func:`coil_radius_for` stopped, and every reading on the way.

    ``status`` is ``"met"`` (within the tolerance), ``"closest"`` (tries ran
    out first; the closest reading is kept), ``"floor"`` (even the smallest
    radius reads above the target) or ``"ceiling"`` (even ``r_max`` reads
    below it).
    """

    target: float
    radius: float
    z: float
    status: str
    trace: list = field(default_factory=list)     # [(radius, z), ...] in order

    def as_dict(self) -> dict:
        return {"target": self.target, "radius": round(self.radius, 4),
                "z": round(self.z, 4), "status": self.status,
                "trace": [[round(r, 4), round(z, 4)] for r, z in self.trace]}


def coil_radius_for(target: float, measure, *, lo: float = 1.5, start: float = 6.0,
                    r_max: float = 40.0, rel_tol: float = 0.03,
                    tries: int = 12) -> RadiusSearch:
    """The coil radius (A) at which ``measure`` reads ``target``.

    ``measure(radius)`` draws the network at that radius and returns its Z;
    the caller draws from the same seed every time, so the reading moves
    only with the radius, and more radius means more of every strand's
    neighbours inside its loop, so it moves one way. The search reads
    ``lo`` first, then widens from ``start`` by 1.6 until the target is
    bracketed, then interpolates inside the bracket (falling back to the
    middle when the interpolation would land at an edge) until a reading is
    within ``rel_tol`` of the target or ``tries`` readings are spent.
    """
    target = float(target)
    trace: list = []

    def read(r):
        z = float(measure(float(r)))
        trace.append((float(r), z))
        return z

    def best(status):
        r, z = min(trace, key=lambda rz: abs(rz[1] - target))
        if status is None:
            status = "met" if abs(z - target) <= rel_tol * max(target, 1e-9) else "closest"
        return RadiusSearch(target, r, z, status, list(trace))

    z_lo = read(lo)
    if z_lo >= target * (1.0 - rel_tol):
        return best("met" if z_lo <= target * (1.0 + rel_tol) else "floor")
    hi = max(float(start), 1.5 * lo)
    z_hi = read(hi)
    while z_hi < target and hi < r_max and len(trace) < tries:
        lo, z_lo = hi, z_hi
        hi = min(float(r_max), 1.6 * hi)
        z_hi = read(hi)
    if z_hi < target * (1.0 - rel_tol):
        return best("ceiling" if hi >= r_max else None)
    while len(trace) < tries:
        if abs(min(trace, key=lambda rz: abs(rz[1] - target))[1] - target) <= rel_tol * target:
            break
        if hi - lo < 0.02:
            break
        r = (lo + (target - z_lo) * (hi - lo) / (z_hi - z_lo)
             if z_hi > z_lo else 0.5 * (lo + hi))
        margin = 0.1 * (hi - lo)
        if not lo + margin <= r <= hi - margin:
            r = 0.5 * (lo + hi)
        z = read(r)
        if z < target:
            lo, z_lo = r, z
        else:
            hi, z_hi = r, z
    return best(None)


# ---------------------------------------------------------------------------
# Keeping the other strands out of the designed braids
# ---------------------------------------------------------------------------

def _tube_margin(points, braid, box) -> float:
    """How far outside ``braid``'s tube the nearest of ``points`` is (A).

    The tube is the cylinder about the braid's axis over its span,
    ``mid +- half * axis``, of the braid's radius plus
    :data:`BRAID_CLEARANCE`, with flat ends: past the span the two strands
    run back to their chords and there is no braid to be inside. (A rounded
    end took in the strands joining the partners' junctions, which pass the
    axis just beyond the span: 45 strands of a DP-10 SC build with 5 pairs.)
    Negative is inside; a strand with no point beside the span is clear.
    """
    d = np.asarray(points, float) - braid["mid"]
    if box is not None:
        d = d - box * np.round(d / box)
    u = d @ braid["axis"]
    beside = np.abs(u) <= braid["half"]
    if not beside.any():
        return float("inf")
    r = np.linalg.norm(d[beside] - np.outer(u[beside], braid["axis"]), axis=1)
    return float(r.min()) - (braid["radius"] + BRAID_CLEARANCE)


def _turn_about_chord(full, phi) -> np.ndarray:
    """The path turned by ``phi`` about the line joining its ends."""
    p = np.asarray(full, float)
    axis = p[-1] - p[0]
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or phi == 0.0:
        return p.copy()
    axis = axis / n
    c, s = np.cos(phi), np.sin(phi)
    v = p - p[0]
    return (p[0] + v * c + np.cross(axis, v) * s
            + np.outer(v @ axis, axis) * (1.0 - c))


def clear_braids(chains, movable, braids, box, avoid=None) -> dict:
    """Turn every strand that runs through a designed braid out of it.

    The designed pairs are drawn first and the other strands one at a time
    around nothing, so a meander can pass straight between the two arms of a
    braid. The pair is still wound, but the winding is then shared with
    that strand, and the network-cycle reading (:mod:`topon.analysis.windings`)
    depends on which cycles it takes. On two DP-30 builds with six designed
    pairs each, 5 of the 12 pairs had a strand within 0.2 to 0.7 A of their
    braid, and 2 of them read differently on different cycle pairs because of
    it; the settle, which puts back every passage, keeps such a strand where
    it is.

    ``chains`` are ``(ids, points, ...)`` as :func:`place_network` builds them,
    ``movable`` the indices of the strands that may be turned (drawn by the
    placement: not a designed strand, not a primary loop), and ``braids`` the
    braid sites in A (``mid``, ``axis``, ``half``, ``radius``, from
    :func:`~topon.conformation.entanglement.realize.entangled_backbone_paths`
    with ``sites``). A strand with any backbone atom inside a braid's tube
    (radius plus :data:`BRAID_CLEARANCE`) is turned about its chord to the
    first of :data:`BRAID_TURNS` angles that leaves it outside every tube,
    or, if none does, to the one that leaves it furthest out. Turning about
    the chord moves neither junction and keeps every bond, angle and
    self-contact, and it draws nothing from the random stream, so a strand
    that was clear is drawn exactly as before. ``avoid`` maps a strand to
    turns (of the :data:`BRAID_TURNS`) it must not take: for a strand drawn
    on one side of a chord it shares, the turns that would lay it on a
    strand drawn on the other side (half a turn, for a secondary loop).
    Returns the count of strands found inside, turned clear, and left inside
    with their margins (negative: how far inside).
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    report = {"braids": len(braids), "inside": 0, "turned_clear": 0, "left_inside": []}
    if not braids:
        return report
    for k in movable:
        ids, pts = chains[k][0], np.asarray(chains[k][1], float)
        worst = min(_tube_margin(pts[1:-1], b, box) for b in braids)
        if worst >= 0.0:
            continue
        report["inside"] += 1
        best = (worst, 0.0)
        skip = (avoid or {}).get(k, ())
        for j in range(1, BRAID_TURNS):
            if j in skip:
                continue
            phi = 2.0 * np.pi * j / BRAID_TURNS
            turned = _turn_about_chord(pts, phi)
            m = min(_tube_margin(turned[1:-1], b, box) for b in braids)
            if m > best[0]:
                best = (m, phi)
            if m >= 0.0:
                break
        chains[k] = (ids, _turn_about_chord(pts, best[1])) + tuple(chains[k][2:])
        if best[0] >= 0.0:
            report["turned_clear"] += 1
        else:
            report["left_inside"].append(round(best[0], 3))
    return report


# ---------------------------------------------------------------------------
# Cages placed whole
# ---------------------------------------------------------------------------

@dataclass
class RigidBody:
    """A node placed whole and held through the placement: a POSS cage.

    ``atoms`` are every atom id of the node (hydrogens included) and ``xyz``
    where they are, in A and one image; ``bonds`` its bonds between heavy
    atoms and ``faces`` its rings (ids in ring order), ``centre`` the centre
    of its core (the ring atoms) and ``core`` the core's radius. ``attach``
    maps the index of each strand bonded to it to the ends of that strand
    at the body (``0`` its first atom, ``-1`` its last). ``rotation`` is the
    turn from the template's frame.
    """

    node: int
    atoms: list
    xyz: np.ndarray
    bonds: list
    faces: list
    centre: np.ndarray
    core: float
    attach: dict = field(default_factory=dict)
    rotation: Optional[np.ndarray] = None


def _rotation_onto(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1.0 + 1e-12:
        # opposite: half a turn about any axis normal to a
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p = p / np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _rotation_about(axis, phi) -> np.ndarray:
    """Rotation by ``phi`` about unit vector ``axis``."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(phi) * kx + (1.0 - np.cos(phi)) * (kx @ kx)


def body_rotation(vectors, targets, score=None, twists: int = BODY_TWISTS) -> np.ndarray:
    """The turn that points a body's stubs at the strands they bond to.

    ``vectors`` are the stubs as seen from the body's centre in its own
    frame (where each strand atom bonded to the body sits), ``targets`` the
    directions from the junction to each strand's other end. Two or more
    stubs that span a plane fix the turn, the one that best lays each stub
    on its target (Kabsch). One stub, or stubs all along one line, fix only
    the axis: the mean stub goes onto the mean target, and the turn about it
    is the one of ``twists`` that ``score(rotation)`` reads highest (the
    first of equals; without a score, none).
    """
    R0, axis = _stub_turn(vectors, targets)
    if axis is None or score is None:
        return R0
    best, best_s = R0, -np.inf
    for j in range(int(twists)):
        R = _rotation_about(axis, 2.0 * np.pi * j / twists) @ R0
        s = float(score(R))
        if s > best_s:
            best, best_s = R, s
    return best


def _stub_turn(vectors, targets):
    """``(R, axis)``: the turn the stubs fix, and the axis they leave free.

    ``axis`` is None when two or more stubs span a plane (the Kabsch turn),
    and the mean target direction when they do not, ``R`` then laying the
    mean stub on it. No stub: the identity, with no axis.
    """
    u = np.asarray(vectors, float).reshape(-1, 3)
    t = np.asarray(targets, float).reshape(-1, 3)
    if not len(u):
        return np.eye(3), None
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    if len(u) >= 2:
        U, S, Vt = np.linalg.svd(u.T @ t)
        if S[1] > 1e-6 * S[0]:
            d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
            return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T, None
    a, b = u.mean(axis=0), t.mean(axis=0)
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        a, b = u[0], t[0]
    return _rotation_onto(a, b), b / np.linalg.norm(b)


def point_segment_distance(points, a, b, box=None) -> np.ndarray:
    """Distance from every point to the nearest of the segments ``a[k]``-``b[k]``.

    Under the minimum image of ``box`` when it is given (each segment read
    in the image nearest each point). ``points`` ``(n, 3)``, ``a`` and ``b``
    ``(m, 3)``; returns ``(n,)``, inf with no segment.
    """
    p = np.asarray(points, float).reshape(-1, 3)
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    if not len(a):
        return np.full(len(p), np.inf)
    d = p[:, None, :] - a[None, :, :]
    if box is not None:
        d = d - box * np.round(d / box)
    s = b - a
    if box is not None:
        s = s - box * np.round(s / box)
    ss = np.maximum(np.einsum("ij,ij->i", s, s), 1e-12)
    f = np.clip(np.einsum("nmj,mj->nm", d, s) / ss, 0.0, 1.0)
    return np.linalg.norm(d - f[:, :, None] * s[None, :, :], axis=2).min(axis=1)


def place_bodies(shapes, box, chords=None, chord_strands=None, points=None,
                 sweeps: int = 2) -> list:
    """Every cage set at its junction, turned so its stubs face its strands.

    ``shapes`` holds one dict per cage: ``node``, ``atoms`` (ids), ``xyz``
    (the template, its own frame), ``heavy`` (bool per atom), ``centre`` and
    ``core`` (of the template), ``bonds``, ``faces``, ``stubs`` (the
    template's position of the strand atom of each bond out), ``targets``
    (per stub, the direction from the junction to that strand's other end
    in A, or None for a loop back to the same cage), ``at`` (the junction's
    position, A) and ``attach``. The cage's centre goes on ``at``. Two or
    more stubs that span a plane fix its turn, the best fit of each stub on
    its target (Kabsch, as :func:`body_rotation`). Where the stubs leave the
    turn about their axis free (one strand, or targets on one line), it is
    the one of :data:`BODY_TWISTS` that
    keeps the cage furthest from everything else: its heavy atoms from the
    ``chords`` of the strands not bonded to it (``(a, b)`` arrays of segment
    ends, A, with ``chord_strands`` their strand indices) and from
    ``points`` (other junctions), and its atoms from every atom of the other
    cages. Cages on neighbouring sites reach into each other (an AM0270's
    arms reach 9 A from its centre, and sites can be 12.7 A apart), so the
    turns are chosen in ``sweeps`` passes, the first against the cages
    already turned and every later one against all of them. Draws nothing
    from any random stream. Returns :class:`RigidBody` objects in order.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    ca, cb = (np.zeros((0, 3)), np.zeros((0, 3))) if chords is None else (
        np.asarray(chords[0], float).reshape(-1, 3), np.asarray(chords[1], float).reshape(-1, 3))
    cs = np.asarray(chord_strands if chord_strands is not None else [], int)
    pts = np.zeros((0, 3)) if points is None else np.asarray(points, float).reshape(-1, 3)
    prep = []
    for sh in shapes:
        local = np.asarray(sh["xyz"], float) - np.asarray(sh["centre"], float)
        vec, tgt = [], []
        for stub, t in zip(sh["stubs"], sh["targets"]):
            if t is not None and np.linalg.norm(t) > 1e-9:
                vec.append(np.asarray(stub, float) - np.asarray(sh["centre"], float))
                tgt.append(np.asarray(t, float))
        mine = set(sh["attach"])
        keep = np.array([int(c) not in mine for c in cs], bool) if len(cs) else np.zeros(0, bool)
        R0, axis = _stub_turn(vec, tgt)
        prep.append({"local": local, "heavy": np.asarray(sh["heavy"], bool),
                     "at": np.asarray(sh["at"], float), "R0": R0, "axis": axis,
                     "a": ca[keep] if len(cs) else ca, "b": cb[keep] if len(cs) else cb,
                     "reach": float(np.linalg.norm(local, axis=1).max())})
    turns = [p["R0"] for p in prep]

    def placed(i, R):
        return prep[i]["at"] + prep[i]["local"] @ R.T

    def score(i, R, others):
        p = prep[i]
        P = placed(i, R)
        m = (float(point_segment_distance(P[p["heavy"]], p["a"], p["b"], box).min())
             if len(p["a"]) else np.inf)
        if len(pts):
            d = P[p["heavy"]][:, None, :] - pts[None, :, :]
            if box is not None:
                d = d - box * np.round(d / box)
            m = min(m, float(np.linalg.norm(d, axis=2).min()))
        for j in others:
            q = prep[j]
            c = q["at"] - p["at"]
            if box is not None:
                c = c - box * np.round(c / box)
            if np.linalg.norm(c) > p["reach"] + q["reach"] + 1.0:
                continue
            d = P[:, None, :] - placed(j, turns[j])[None, :, :]
            if box is not None:
                d = d - box * np.round(d / box)
            m = min(m, float(np.linalg.norm(d, axis=2).min()))
        return m

    for sweep in range(max(1, int(sweeps))):
        for i, p in enumerate(prep):
            if p["axis"] is None:
                continue
            others = [j for j in range(len(prep)) if j != i and (sweep or j < i)]
            best, best_s = turns[i], -np.inf
            for k in range(BODY_TWISTS):
                R = _rotation_about(p["axis"], 2.0 * np.pi * k / BODY_TWISTS) @ p["R0"]
                s_k = score(i, R, others)
                if s_k > best_s:
                    best, best_s = R, s_k
            turns[i] = best
    bodies: list = []
    for i, (sh, p) in enumerate(zip(shapes, prep)):
        bodies.append(RigidBody(
            node=sh["node"], atoms=list(sh["atoms"]), xyz=placed(i, turns[i]),
            bonds=list(sh["bonds"]), faces=[list(f) for f in sh["faces"]],
            centre=p["at"].copy(), core=float(sh["core"]),
            attach={k: tuple(v) for k, v in sh["attach"].items()}, rotation=turns[i]))
    return bodies


def body_margin(points, body, box, skip=(), ids=None) -> float:
    """How far outside ``body``'s core a strand's backbone keeps (A).

    The smallest distance from the core's centre to any of ``points`` (but
    the rows in ``skip``: held ends, and a strand's own atoms next to the
    cage it is bonded to), less the core's radius and
    :data:`BODY_CLEARANCE`, and far below zero when a bond of the strand
    passes through one of the body's faces. Every bond is read against each
    face but one with an atom on that face (``ids`` names the points' atoms;
    a bare cage's corner, which a strand bonds to, is on three faces).
    Negative is inside.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    pts = np.asarray(points, float)
    d = pts - body.centre
    if box is not None:
        d = d - box * np.round(d / box)
    keep = np.ones(len(pts), bool)
    keep[list(skip)] = False
    r = np.linalg.norm(d, axis=1)
    margin = float(r[keep].min()) - (body.core + BODY_CLEARANCE) if keep.any() else np.inf
    # a face read on the bonds of the strand in the body's image, ends next
    # to the attachment (which may be ring atoms of the body) left out
    q = body.centre + d
    at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
    reach = body.core + 4.0
    ids = None if ids is None else [int(a) for a in ids]
    for k in range(len(pts) - 1):
        if min(r[k], r[k + 1]) > reach:
            continue
        for ring in body.faces:
            if ids is not None and (ids[k] in ring or ids[k + 1] in ring):
                continue
            P = np.array([at[int(a)] for a in ring])
            if segments_through_ring(q[k][None], q[k + 1][None], P)[0]:
                margin = min(margin, -(body.core + BODY_CLEARANCE) - 10.0)
    return margin


def cage_detour(start, end, body, box, attached=(), bond: float = 1.6):
    """Waypoints round ``body`` for a strand whose chord runs through its core.

    Turning a strand about its chord cannot take it out of a cage its chord
    runs through: a loop between two corners of a bare cage, or a bridge
    whose chord passes a cage on a third site. Such a strand is drawn through
    points on the sphere of the core's radius plus :data:`BODY_CLEARANCE`
    and 1 A, on two great-circle arcs: from the direction of its start to the
    side its chord passes on (or, through the centre, a side normal to the
    chord), then on to the direction of its end, no two points more than 30
    degrees apart. Each arc starts and stops where a straight leg from the
    end it runs to would touch the sphere, so the legs stay outside it.
    ``attached`` holds the ends (0, -1) bonded to this cage, whose first
    ``bond`` A along the chord are not read. Returns the waypoints (A, in the
    image of ``start``), or None when the chord keeps clear.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    c = start - body.centre
    if box is not None:
        c = c - box * np.round(c / box)
    centre = start - c                      # the cage in the strand's image
    u, w = start - centre, end - centre
    chord = w - u
    L = float(np.linalg.norm(chord))
    if L < 1e-9:
        return None
    lo = bond / L if 0 in attached else 0.0
    hi = 1.0 - bond / L if -1 in attached else 1.0
    if lo >= hi:
        return None
    t = float(np.clip(-(u @ chord) / (L * L), lo, hi))
    q = u + t * chord
    keep = body.core + BODY_CLEARANCE
    if float(np.linalg.norm(q)) >= keep:
        return None
    radius = keep + 1.0
    side = q - chord * float(q @ chord) / (L * L)
    if float(np.linalg.norm(side)) < 1e-3:
        trial = np.eye(3)[int(np.argmin(np.abs(chord)))]
        side = np.cross(chord, trial)
    side = side / np.linalg.norm(side)

    def arc(a, b, skip_a, skip_b):
        """Points at ``radius`` from direction ``a`` to ``b`` (unit), the
        first ``skip_a`` and last ``skip_b`` radians left out."""
        ang = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if ang < 1e-9:
            return [radius * a]
        perp = b - a * float(a @ b)
        perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-9 else side
        a0, a1 = min(skip_a, ang), max(min(skip_a, ang), ang - skip_b)
        n = max(1, int(np.ceil((a1 - a0) / np.radians(30.0))))
        return [radius * (np.cos(x) * a + np.sin(x) * perp)
                for x in np.linspace(a0, a1, n + 1)]

    def tangent(r):
        return float(np.arccos(min(1.0, radius / r))) if r > radius else 0.0

    ru, rw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    pts = arc(u / ru, side, tangent(ru), 0.0) + arc(side, w / rw, 0.0, tangent(rw))[1:]
    return [centre + p for p in pts]


def keep_off_points(body, start, box) -> np.ndarray:
    """``body``'s atoms in the image of ``start`` (A), for a detour to keep off."""
    box = None if box is None else np.asarray(box, float).reshape(3)
    d = np.asarray(body.xyz, float) - np.asarray(start, float)
    if box is not None:
        d = d - box * np.round(d / box)
    return np.asarray(start, float) + d


def lead_out_of_body(first, path, last, body, box):
    """A designed path that starts inside a cage, led out round it.

    A designed pair's path is drawn from its junction, and the junction of a
    cage is the cage's centre, so a strand bonded to a cage has its first
    atoms drawn inside the cage, with bonds through its faces (on the bare
    cage test network, the first six atoms, as close as 1.17 A to the
    centre, and two bonds through one face). No turn about the chord takes a
    designed path out, since its shape is the design, so before 0.4.5 the
    settle had to drag the strand out through the face while putting back
    every move that threaded another bond. Whether it got out depended on
    where the cage's atoms sat to a hundredth of an angstrom. On RDKit
    2025.09.6 that build settled in 207 rounds, and with the cage moved by
    0.001 A, or embedded on RDKit 2026.3.6, it ended with bonds through the
    faces.

    ``first`` is the atom the cage's stub holds, where the strand now starts
    (A), ``path`` the strand's interior points as designed (A, one per atom)
    and ``last`` its other end atom, in the image of ``first``. The path is
    read in that image too: a designed pair is drawn in the image of its
    first strand, so the other's path can lie a whole cell from its own
    junction. The points from the start that lie inside the sphere the
    placement keeps backbones out of (the core's radius plus
    :data:`BODY_CLEARANCE`) are cut out, and the path runs instead from
    ``first`` round the cage to the first point outside it, on the
    great-circle arc between the two, then along the rest of its own points.
    The arc keeps 1 A outside the sphere, as :func:`cage_detour`'s does (a
    bare cage's corner hydrogens, which the settle does not read, sit 1.26 A
    outside its core). Every atom is laid along that line at the spacing it
    was drawn with, scaled to the new length, so the path keeps its shape
    outside the cage, its winding included. Returns the new interior points,
    as many as ``path`` and in the image of ``first``, or None when the
    path's first point is outside the sphere.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    first = np.asarray(first, float)
    last = np.asarray(last, float)
    pts = np.asarray(path, float).reshape(-1, 3)
    if not len(pts):
        return None
    if box is not None:
        pts = pts - box * np.round((pts[0] - first) / box)
    c = first - np.asarray(body.centre, float)
    if box is not None:
        c = c - box * np.round(c / box)
    centre = first - c                     # the cage in the path's image
    keep = float(body.core) + BODY_CLEARANCE
    r = np.linalg.norm(pts - centre, axis=1)
    n_in = 0
    while n_in < len(pts) and r[n_in] < keep:
        n_in += 1
    if not n_in:
        return None
    rejoin = pts[n_in] if n_in < len(pts) else last
    u, w = first - centre, rejoin - centre
    ru, rw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    a, b = u / ru, w / rw
    ang = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    arc = []
    if ang > 1e-9:
        perp = b - a * float(a @ b)
        if float(np.linalg.norm(perp)) < 1e-9:
            # the two on opposite sides: round any side, the same one each time
            perp = np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))])
        perp = perp / np.linalg.norm(perp)
        n = max(1, int(np.ceil(ang / np.radians(15.0))))
        for t in np.linspace(0.0, 1.0, n + 1)[1:-1]:
            rad = max(keep + 1.0, ru + t * (rw - ru))
            arc.append(centre + rad * (np.cos(t * ang) * a + np.sin(t * ang) * perp))
    line = np.array([first, *arc, *pts[n_in:], last])
    # the spacing the atoms were drawn with: the first bond, from the stub
    # into the cage, read as the path's mean
    drawn = np.linalg.norm(np.diff(np.vstack([pts, last]), axis=0), axis=1)
    gaps = np.concatenate([[drawn.mean()], drawn])
    at = np.cumsum(gaps)[:-1] / gaps.sum()
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = at * s[-1]
    k = np.clip(np.searchsorted(s, want, side="right") - 1, 0, len(seg) - 1)
    f = (want - s[k]) / np.maximum(seg[k], 1e-12)
    return line[k] + f[:, None] * (line[k + 1] - line[k])


def arm_bonds(body) -> tuple:
    """The bonds of ``body``'s arms as two ``(m, 3)`` arrays of ends (A), its
    heavy bonds that are not bonds of its faces (an AM0270 cap's propyl and
    isooctyl arms, from the corner Si out). A bare cage has none."""
    ring = {int(a) for f in body.faces for a in f}
    at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
    ends = [(at[int(a)], at[int(b)]) for a, b in body.bonds
            if not (int(a) in ring and int(b) in ring)]
    return (np.array([e[0] for e in ends]).reshape(-1, 3),
            np.array([e[1] for e in ends]).reshape(-1, 3))


def arm_margin(points, arms, box) -> float:
    """How far a strand's backbone bonds keep from a cage's arm bonds
    (``arms``, from :func:`arm_bonds`), less :data:`ARM_CLEARANCE` (A;
    negative is closer, inf with no arm near). The bonds at the strand's
    two ends are not read, since its end atoms are held junctions that no
    turn moves. Each pair of bonds is read in its nearest image."""
    a1, a2 = arms
    pts = np.asarray(points, float)
    if not len(a1) or len(pts) < 4:
        return np.inf
    p1, p2 = pts[1:-2], pts[2:-1]
    pm, qm = 0.5 * (p1 + p2), 0.5 * (a1 + a2)
    d = qm[None, :, :] - pm[:, None, :]
    shift = np.zeros_like(d) if box is None else -box * np.round(d / box)
    # no two bonds can be closer than their midpoints less their half lengths
    half = (0.5 * np.linalg.norm(p2 - p1, axis=1)[:, None]
            + 0.5 * np.linalg.norm(a2 - a1, axis=1)[None, :])
    i, j = np.nonzero(np.linalg.norm(d + shift, axis=2) - half < ARM_CLEARANCE)
    if not len(i):
        return np.inf
    dist = segment_closest(p1[i], p2[i], a1[j] + shift[i, j], a2[j] + shift[i, j])[2]
    return float(dist.min()) - ARM_CLEARANCE


def clear_bodies(chains, movable, bodies, box, avoid=None, braids=None) -> dict:
    """Turn every strand that runs into a cage out of it.

    The strands are drawn around nothing, so a meander can pass through the
    core of a cage placed at a junction nearby, through one of its faces,
    and no settling takes a bond back out through a ring. A strand
    in ``movable`` with a backbone atom within :data:`BODY_CLEARANCE` of a
    cage's core, or a bond through a face, is turned about its chord to the
    first of :data:`BRAID_TURNS` angles that leaves it clear of every cage
    (and of every designed braid, when ``braids`` are given), or, if none
    does, to the one that leaves it furthest out, as :func:`clear_braids`
    turns strands out of braids. A strand bonded to a cage is drawn from
    the atom the cage's stub holds, and that atom and the next are not read
    against that cage (nor any strand's two end atoms, which are held).
    A strand is also read against the arms of every cage it is
    not bonded to (:func:`arm_margin`). A backbone bond within
    :data:`ARM_CLEARANCE` of an arm bond is inside too, since the settle,
    which cannot move the arms, could stop with the strand caught between
    them. A turn is chosen for the core first (and the braids): of the turns
    that leave the strand as far into a core as the best of them, or clear
    of every core, the one furthest from the arms, so an arm that no turn
    clears cannot leave the strand in a core that a turn would. ``avoid``
    is :func:`clear_braids`'s. Returns the count of strands found inside, of
    them those found inside only by an arm (``near_arms``), turned clear,
    and left inside with their margins (A, negative).
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    report = {"cages": len(bodies), "inside": 0, "turned_clear": 0, "left_inside": []}
    if not bodies:
        return report
    arms = [arm_bonds(body) for body in bodies]
    if any(len(a[0]) for a in arms):
        report["near_arms"] = 0

    def score(k, pts):
        """``(core, overall)``: the core's margin (and the braids') clipped
        at 0, and the smallest of every margin, arms included. Compared as
        a pair, a turn deeper into a core never wins on the arms; without
        arms the pair orders turns as the core's margin alone."""
        n = len(pts)
        core = arm = np.inf
        for body, arm_k in zip(bodies, arms):
            skip = {0, n - 1}           # the strand's own ends are held
            for end in body.attach.get(k, ()):
                # drawn from the atom the cage's stub holds: that and the
                # next, which the settle exempts too
                skip |= ({0, 1} if end == 0 else {n - 1, n - 2})
            core = min(core, body_margin(pts, body, box, skip, ids=chains[k][0]))
            if k not in body.attach:
                arm = min(arm, arm_margin(pts, arm_k, box))
        if braids:
            core = min(core, min(_tube_margin(pts[1:-1], b, box) for b in braids))
        return (min(core, 0.0), min(core, arm))

    for k in movable:
        ids, pts = chains[k][0], np.asarray(chains[k][1], float)
        now = score(k, pts)
        if now[1] >= 0.0:
            continue
        report["inside"] += 1
        if "near_arms" in report and now[0] >= 0.0:
            report["near_arms"] += 1
        best, best_phi = now, 0.0
        skip = (avoid or {}).get(k, ())
        for j in range(1, BRAID_TURNS):
            if j in skip:
                continue
            phi = 2.0 * np.pi * j / BRAID_TURNS
            m = score(k, _turn_about_chord(pts, phi))
            if m > best:
                best, best_phi = m, phi
            if m[1] >= 0.0:
                break
        chains[k] = (ids, _turn_about_chord(pts, best_phi)) + tuple(chains[k][2:])
        if best[1] >= 0.0:
            report["turned_clear"] += 1
        else:
            report["left_inside"].append(round(best[1], 3))
    return report


# ---------------------------------------------------------------------------
# Settling the backbones: apart, at their bonds and angles, nothing crossed
# ---------------------------------------------------------------------------

def _angles(P, ta, tb):
    """Bond angle at every vertex ``ta + 1`` between rows ``ta`` and ``tb``, degrees."""
    u = P[ta] - P[ta + 1]
    v = P[tb] - P[ta + 1]
    c = np.einsum("ij,ij->i", u, v) / np.maximum(
        np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def settle_backbones(chains, box, clearance: float = CLEARANCE, rng=None,
                     rounds: int = SETTLE_ROUNDS, step: float = 0.2, sweeps: int = 8,
                     width: float = 1.0, ramp: int = 40, tol: float = 0.02,
                     keep_frames: bool = False, bodies=None, held_ids=None,
                     side_atoms=None):
    """Bring every backbone to its bonds and angles, clear of every other, crossing nothing.

    ``chains`` is one ``(ids, points, r0, closed, one_three)`` per strand:
    its backbone atom ids from end atom to end atom, their positions in the
    strand's own image (A), the equilibrium length of each bond, whether it
    is a loop (both ends the same junction), and the equilibrium distance
    between each atom's two backbone neighbours (from the bond angle at it;
    None keeps the distances the path was drawn with). A sixth item, when
    present, is the rings the strand runs through (:func:`ring_windows`).
    The two end atoms of every strand are junctions or end caps and are held.

    Two things are wrong with a freshly drawn network. Strands drawn one at
    a time can pass through the same point, and a path drawn in bead-spring
    units has bead-spring bond angles, nearly straight on a meander, where
    DREIDING PDMS wants 104.5 and 109.5 degrees. The first is a pair of bonds
    with no side to be pushed apart on. The second is worse: the first stage
    closes every angle at once, each strand gathers in about a fifth of its
    length, and at melt density that pulls strands through each other (the
    last passage left on the DP-30 build once the first was cured: two bonds
    that started 1.5 A apart, drawn steadily together over 200 steps by the
    chains around them contracting, their Si-O bonds held at 1.16 r0).

    So each round does two things. Every pair of bonds short of
    ``clearance`` (two bonds that share no atom and, on one strand, are
    more than two bonds apart) is pushed apart along the line from one
    bond's closest point to the other's, a little past the clearance, half
    each (all of it on a side whose atoms are held); the push moves a
    stretch of each strand about the closest point, Gaussian along the chain
    with ``width`` atoms, so the bonds inside it barely change. Then the
    bonds and 1-3 distances are pulled towards their targets, which move
    from what was drawn to the equilibrium over the first ``ramp`` rounds.
    No atom moves more than ``step`` in a round. Last, the round is read as
    a move from where every atom was to where it is now
    (:func:`~topon.conformation.segments.passing_pairs`), and the atoms of
    any two bonds that passed through each other on the way are put back,
    until none did. Two bonds drawn exactly touching have no line between
    them and open along the normal to both, the side drawn at random; the
    sides of touching pairs are the only choices the pass makes, and a
    drawn path that touches has made no choice to keep. That first parting
    is read for passages too: the pairs drawn touching may come apart
    either way, and a push that carries any other bond through one is put
    back (``touching_moves_put_back`` in the report counts the atoms). Two
    strands drawn through one point at an
    atom of each touch in four pairs, whose sides can cancel; a contact
    left with no side by that is parted again along one normal, one side
    drawn for all its pairs (``sideless_parted`` in the report is the
    number of pairs parted so). Rounds stop when no pair is short and
    every bond and 1-3 distance is within ``tol`` of its target, or after
    ``rounds`` of them.

    Two bonds joined through a third sit that bond's length apart at angles
    over 90 degrees. Along one strand such a pair is skipped (two bonds
    apart). Across a junction it is read: strand A's bond before its
    junction bond against strand B's junction bond, joined through A's
    junction bond, which guards the angle between the two strands there. A
    joining bond shorter than the clearance (an aryl ether O typed O_2
    bonded to a junction Si, 1.487 A) left that pair short at equilibrium,
    and the settle pushed it every round to the cap without clearing it.
    Such a pair is now read against the joining bond as it is, less
    ``tol``, and pushed no further than that bond's length: short only when
    the angle between the two strands closes. Pairs joined through a bond of
    the clearance or longer (Si-O 1.587, C-C 1.53) are read as they always
    were.

    A ring a strand runs through (a para-phenylene between Si and O) is
    settled whole: its flat shape is laid on the stretch as drawn, its
    off-path atoms join the settle as rows of their own with every ring bond
    and every 1-3 distance round a ring atom (a six-ring with all of these at
    their targets is a flat hexagon, and an outside atom 120 degrees from
    both its ring neighbours lies in its plane), and each sweep also pulls
    the ring and its two outside atoms onto their best plane, since distances
    alone leave a ring free to pucker. A ring that turns the backbone by
    more than :data:`RING_TURN` (:func:`_turns`) is laid with its two
    outside atoms weighing most, between them as drawn. The added atoms are
    not read for clearance or passages, and the settle does not act on a
    bond through a ring's face.
    Converged also means every such term within ``tol`` and every ring
    within ``tol`` of a bond length of its plane; the report records
    ``ring_atoms``, ``ring_terms_ratio``, ``ring_off_plane``,
    ``rings_laid_by_outside_atoms``, ``ring_threads`` (the backbone bonds
    through a ring's face as the settle starts and as it ends) and
    ``_ring_atoms`` (the added atoms' places, which :func:`place_network`
    takes out of the report). A build with rings runs :data:`RING_SWEEPS`
    sweeps a round.

    ``bodies`` are the cages placed whole (:class:`RigidBody`). They
    are held like the junctions they are: every heavy bond of a body is one
    more bond the backbones are kept ``clearance`` from and never moved
    through, with all of the push on the backbone side; a pair of a body's
    bond and a backbone bond that share an atom or are joined by one bond
    (the strand's first bonds against the arm it hangs from) is left out,
    as two bonds within two of each other along one strand are. And no
    backbone bond may come to pass through a face of a cage (a move that
    takes one in is put back like a passage), which no pair of bonds would
    read: a bond through the middle of an Si4O4 face is about 2 A from its
    bonds. The backbone atoms a cage's keep-out sphere reads are also kept
    from pointing a side atom into it. The two tetrahedral positions
    off an atom's two backbone bonds, :data:`SIDE_BOND` out, where
    :func:`place_substituents` puts its methyls, stay :data:`SIDE_CLEARANCE`
    outside the core. They are read once the ramp has brought the bonds and
    angles to their targets, since only then are they where the methyls
    will go. From that round an atom with one inside is moved out along that
    position's line from the centre, and the settle is not converged while
    one is. ``side_atoms`` (``{atom id: side atoms}``) says how many side
    atoms each backbone atom has: none is not read, one is read on the
    bisector of its two backbone bonds, where the walk puts it, and two or
    more on the two tetrahedral positions. Without it every atom is read as
    having two. ``held_ids`` are atoms held wherever they sit in a chain, besides
    the two ends: the stub atom of a strand bonded to a cage, whose chain
    starts at the cage's attachment atom. Without bodies or ``held_ids`` the
    pass is exactly as before.

    Returns ``(points, report, frames)``: the new positions per chain,
    what was found and left, and (with ``keep_frames``) ``(ids, [xyz, ...])``
    with the position of every backbone atom before the first round and
    after each, one row per atom id.
    """
    rng = np.random.default_rng() if rng is None else rng
    box = np.asarray(box, float).reshape(3)
    n_strands = len(chains)
    if bodies:
        # every heavy bond of every body, as a chain of two held atoms
        chains = list(chains)
        for body in bodies:
            at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
            for a, b in body.bonds:
                pts = np.array([at[int(a)], at[int(b)]])
                chains.append(([int(a), int(b)], pts,
                               np.array([np.linalg.norm(pts[1] - pts[0])]), False, None))
    sizes = [len(c[0]) for c in chains]
    X = np.vstack([np.asarray(c[1], float).reshape(-1, 3) for c in chains])
    ids = np.concatenate([np.asarray(c[0], int) for c in chains])
    starts = np.cumsum([0] + sizes[:-1])
    held = np.zeros(len(X), bool)
    held[starts] = True
    held[starts + np.asarray(sizes) - 1] = True
    if held_ids:
        # atoms held wherever they sit in a chain: the stub atom a strand
        # bonded to a cage starts on
        held |= np.isin(ids, np.array(sorted(held_ids), int))
    # Every strand in one image, bond by bond from its first atom. A designed
    # pair's braid comes back in the image nearest its partner's chord, so
    # without this a bond of it can span the box (47 r0 on the DP-30 build
    # with six pairs), and every pair search reaches across the whole cell.
    for s0, m in zip(starts, sizes):
        step_ = np.diff(X[s0:s0 + m], axis=0)
        step_ -= box * np.round(step_ / box)
        X[s0 + 1:s0 + m] = X[s0] + np.cumsum(step_, axis=0)
    # Every ring a strand runs through, whole (ring_windows): its flat
    # shape laid on the stretch as drawn (a held atom kept where it is), and
    # its off-path atoms added as rows of their own, held to the stretch by
    # their bonds and 1-3 distances. A six-ring with every bond at length
    # and every angle at 120 degrees can only be a flat hexagon, and an
    # outside atom 120 degrees from both its ring neighbours lies on their
    # bisector in its plane, so plain distances hold the ring flat. Without
    # the off-path atoms an outside atom drawn in line with the ring could
    # not be bent back (at 180 degrees a 1-3 distance pulls along the bond).
    # The added atoms are not read for clearance or passages, as before.
    n_rows = len(X)
    ring_ids, ring_xyz, ea, eb, e_eq, flat = [], [], [], [], [], []
    ring_cycles, ring_skip, turning = [], [], 0
    X_drawn = X.copy() if any(len(c) > 5 and c[5] for c in chains) else None
    for c, s0, m in zip(chains, starts, sizes):
        if not (len(c) > 5 and c[5]):
            continue
        row_of: dict = {}
        for k_, a in enumerate(c[0]):
            row_of.setdefault(int(a), s0 + k_)
        for ring in c[5]:
            tpl = ring["template"]
            fit = np.array([row_of[a] for a in ring["fit"]], int)
            # an outside atom bonded to a held one (the O a strand ends on, at
            # its junction) stays where it was drawn, as the held atom does:
            # laid elsewhere, its bond to the junction drags the ring and can
            # crush it (two last rings of one sample-graph build)
            pinned = held[fit] | np.array([
                (r > s0 and held[r - 1]) or (r < s0 + m - 1 and held[r + 1]) for r in fit])
            shape = np.array([tpl[a] for a in ring["fit"]])
            if _turns(shape) > RING_TURN:
                # A ring that turns the backbone (a meta-phenylene by 60
                # degrees, a 2,5-furan by 36) is laid with its two outside
                # atoms weighing most too, between them as drawn. Laid by its
                # ring atoms alone on a stretch drawn nearly straight, it
                # swung them, and on the meta-phenylene sample build 110 of
                # 210 strands ended folded back at an O (a5-O-Si 11 degrees)
                # with the O-Si bond read through a ring bond every round. A
                # ring the backbone runs straight through (para) is laid by
                # its ring atoms as before: held by its outside atoms, the
                # pyridine sample build jammed 4 strands at an O-Si bond.
                ends = np.zeros(len(fit), bool)
                ends[[0, -1]] = True
                pinned = pinned | ends
                turning += 1
            rot, shift = _laid(shape, X[fit], pinned)
            X[fit] = np.where(held[fit][:, None], X[fit],
                              np.array([rot @ tpl[a] + shift for a in ring["fit"]]))
            for a in ring["extra"]:
                row_of[a] = n_rows + len(ring_ids)
                ring_ids.append(int(a))
                ring_xyz.append(rot @ tpl[a] + shift)
            for a, b, dist in ring["terms"]:
                ea.append(row_of[a])
                eb.append(row_of[b])
                e_eq.append(float(dist))
            flat.append([row_of[a] for a in ring["fit"]] + [row_of[a] for a in ring["extra"]])
            ring_cycles.append([row_of[a] for a in ring["ring"]])
            ring_skip.append({row_of[a] for a in list(ring["fit"]) + list(ring["extra"])})
    if ring_ids:
        X = np.vstack([X, np.array(ring_xyz)])
        held = np.concatenate([held, np.zeros(len(ring_ids), bool)])
    ea, eb, e_eq = np.asarray(ea, int), np.asarray(eb, int), np.asarray(e_eq, float)
    # the atoms of each ring and its exo atoms, grouped by count, held to
    # one plane: distances alone leave a ring free to pucker (a chair 0.35 A
    # off its plane passes a 2 % tolerance on every bond and 1-3 distance)
    by_count: dict = {}
    for rows_f in flat:
        by_count.setdefault(len(rows_f), []).append(rows_f)
    flat = [np.array(v, int) for _n, v in sorted(by_count.items())]

    sa, sb, chain, along, nseg, closed, r0, t_eq = [], [], [], [], [], [], [], []
    for k, (c, s0, m) in enumerate(zip(chains, starts, sizes)):
        rows_k = np.arange(s0, s0 + m)
        sa.append(rows_k[:-1])
        sb.append(rows_k[1:])
        chain.append(np.full(m - 1, k))
        along.append(np.arange(m - 1))
        nseg.append(np.full(m - 1, m - 1))
        closed.append(np.full(m - 1, bool(c[3])))
        r0.append(np.asarray(c[2], float).reshape(m - 1))
        pts = X[s0:s0 + m]          # joined into one image above
        one_three = c[4] if len(c) > 4 else None
        t_eq.append(np.linalg.norm(pts[2:] - pts[:-2], axis=1) if one_three is None
                    else np.asarray(one_three, float).reshape(max(m - 2, 0)))
    sa, sb = np.concatenate(sa), np.concatenate(sb)
    chain, along = np.concatenate(chain), np.concatenate(along)
    nseg, closed, r0 = np.concatenate(nseg), np.concatenate(closed), np.concatenate(r0)
    ida, idb = ids[sa], ids[sb]
    ta = np.concatenate([np.arange(s0, s0 + m - 2) for s0, m in zip(starts, sizes)])
    tb = ta + 2
    t_eq = np.concatenate(t_eq)
    r_drawn = np.linalg.norm(X[sb] - X[sa], axis=1)
    t_drawn = np.linalg.norm(X[tb] - X[ta], axis=1)
    e_drawn = np.linalg.norm(X[eb] - X[ea], axis=1)
    # the angle each 1-3 target stands for, to report against
    ra, rb = r0[np.searchsorted(sa, ta)], r0[np.searchsorted(sa, ta + 1)]
    theta_eq = np.degrees(np.arccos(np.clip(
        (ra ** 2 + rb ** 2 - t_eq ** 2) / (2.0 * ra * rb), -1.0, 1.0)))
    mobile = (~held).astype(float)
    half_bond = 0.5 * float(max(np.max(r0), np.max(r_drawn))) if len(r0) else 0.0
    rows = np.stack([sa, sb], axis=1)
    pair_ids = np.stack([ida, idb], axis=1)
    held_back = 0
    # every row's chain bounds, for the stretch a push moves
    row_lo = np.concatenate([np.full(m, s0) for s0, m in zip(starts, sizes)])
    row_hi = row_lo + np.concatenate([np.full(m, m - 1) for m in sizes])
    reach = max(1, int(np.ceil(2.5 * width)))
    # The bodies' bonds: held, never a pair with each other, and not a pair
    # with a backbone bond they share an atom with or are bonded to.
    fixed = chain >= n_strands
    near_keys = np.zeros(0, np.int64)
    faces = []
    if bodies:
        bonded: dict = {}
        for a, b in pair_ids.tolist():
            bonded.setdefault(a, set()).add(b)
            bonded.setdefault(b, set()).add(a)
        rows_of: dict = {}
        for r, (a, b) in enumerate(pair_ids.tolist()):
            rows_of.setdefault(a, []).append(r)
            rows_of.setdefault(b, []).append(r)
        keys = set()
        for r in np.flatnonzero(fixed):
            a, b = pair_ids[r]
            near = {int(a), int(b)} | bonded.get(int(a), set()) | bonded.get(int(b), set())
            for x in near:
                for s in rows_of.get(x, ()):
                    if not fixed[s]:
                        keys.add((min(r, s), max(r, s)))
        near_keys = np.array(sorted(i * len(sa) + j for i, j in keys), np.int64)
        for b_index, body in enumerate(bodies):
            at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
            for ring in body.faces:
                faces.append((np.array([at[int(a)] for a in ring]),
                              frozenset(int(a) for a in ring),
                              np.asarray(body.centre, float), float(body.core), b_index))
    reach_face = 2.0 * float(np.max(r0)) if len(r0) else 0.0
    # Every backbone atom kept out of each cage's core by BODY_CLEARANCE, but
    # the first three of a strand bonded to it (its attachment atom, the atom
    # its stub holds, the one after), which its bonds and angles put closer.
    keep_out = []
    for body in bodies or ():
        free = ~held.copy()
        for k, ends in body.attach.items():
            if k >= n_strands:
                continue
            s0, m = int(starts[k]), int(sizes[k])
            for end in ends:
                free[[s0, s0 + 1, s0 + 2] if end == 0 else
                     [s0 + m - 1, s0 + m - 2, s0 + m - 3]] = False
        keep_out.append((np.asarray(body.centre, float), float(body.core) + BODY_CLEARANCE,
                         np.flatnonzero(free), float(body.core)))
    # The same rows' side positions, where place_substituents puts
    # the first two side atoms of an atom with two backbone bonds (the two
    # tetrahedral positions off them, SIDE_BOND out), kept SIDE_CLEARANCE
    # outside the core, so that no bond of a methyl can reach a face. Read on
    # the rows with a backbone neighbour on each side (not a ring's own).
    side_rows = []
    for _centre, _radius, rows_k, _core in keep_out:
        rows_k = rows_k[rows_k < n_rows]
        rows_k = rows_k[(rows_k > row_lo[rows_k]) & (rows_k < row_hi[rows_k])]
        count = (np.full(len(rows_k), 2) if side_atoms is None else
                 np.array([side_atoms.get(int(ids[r]), 0) for r in rows_k], int))
        side_rows.append((rows_k[count > 0], count[count > 0]))
    half_tet = 0.5 * _TET

    def inside(P):
        """For every cage, the rows (and their offsets) inside its keep-out sphere."""
        out = []
        for centre, radius, rows_k, _core in keep_out:
            d = P[rows_k] - centre
            d = d - box * np.round(d / box)
            r = np.linalg.norm(d, axis=1)
            hit = r < radius
            out.append((rows_k[hit], d[hit], r[hit], radius))
        return out

    def sides_in(P):
        """For every cage, the rows with a side position inside its core plus
        SIDE_CLEARANCE, that position's offset from the centre (the deeper of
        the two) and that radius."""
        out = []
        for (centre, _radius, _rows, core), (rows_k, count) in zip(keep_out, side_rows):
            radius_s = core + SIDE_CLEARANCE
            if not len(rows_k):
                out.append((rows_k, np.zeros((0, 3)), np.zeros(0), radius_s))
                continue
            d = P[rows_k] - centre
            d = d - box * np.round(d / box)
            near = np.linalg.norm(d, axis=1) < radius_s + SIDE_BOND
            rows_k, d, count = rows_k[near], d[near], count[near]
            n1 = P[rows_k - 1] - P[rows_k]
            n2 = P[rows_k + 1] - P[rows_k]
            n1 = n1 / np.maximum(np.linalg.norm(n1, axis=1), 1e-12)[:, None]
            n2 = n2 / np.maximum(np.linalg.norm(n2, axis=1), 1e-12)[:, None]
            bis = -(n1 + n2)
            nrm = np.cross(n1, n2)
            lb = np.linalg.norm(bis, axis=1)
            ln = np.linalg.norm(nrm, axis=1)
            ok = (lb > 1e-6) & (ln > 1e-6)
            rows_k, d, count = rows_k[ok], d[ok], count[ok]
            bis = bis[ok] / lb[ok][:, None]
            nrm = nrm[ok] / ln[ok][:, None]
            one = (count == 1)[:, None]
            best_d, best_r = d.copy(), np.full(len(rows_k), np.inf)
            for sign in (1.0, -1.0):
                # one side atom on the bisector, two on the tetrahedral pair
                q = d + SIDE_BOND * np.where(
                    one, bis, np.cos(half_tet) * bis + sign * np.sin(half_tet) * nrm)
                rq = np.linalg.norm(q, axis=1)
                deeper = rq < best_r
                best_d[deeper], best_r[deeper] = q[deeper], rq[deeper]
            hit = best_r < radius_s
            out.append((rows_k[hit], best_d[hit], best_r[hit], radius_s))
        return out

    def keep_out_move(P, sides=True):
        """Every backbone atom inside a cage's sphere, moved out past it, and
        every one with a side position inside the core's side clearance moved
        out along that position's line from the centre."""
        move = np.zeros_like(P)
        for rows_k, d, r, radius in inside(P) + (sides_in(P) if sides else []):
            if len(rows_k):
                u = d / np.maximum(r, 1e-9)[:, None]
                np.add.at(move, rows_k, u * ((1.0 + OVERSHOOT) * radius - r)[:, None])
        return move

    def threading(P) -> set:
        """``(row, face)`` for every backbone bond through a cage face."""
        out = set()
        if not faces:
            return out
        mob = np.flatnonzero(~fixed)
        p1, p2 = P[sa[mob]], P[sb[mob]]
        cage_near = {}
        for f, (ring, members, centre, core, b_index) in enumerate(faces):
            if b_index not in cage_near:
                # the bonds near this cage, read once for all its faces
                d = 0.5 * (p1 + p2) - centre
                sh_all = -box * np.round(d / box)
                cage_near[b_index] = (np.flatnonzero(
                    np.linalg.norm(d + sh_all, axis=1) < core + reach_face), sh_all)
            near, sh = cage_near[b_index]
            if not len(near):
                continue
            rows_n = mob[near]
            skip = np.array([int(ida[r]) in members or int(idb[r]) in members
                             for r in rows_n], bool)
            near, rows_n = near[~skip], rows_n[~skip]
            hit = segments_through_ring(p1[near] + sh[near], p2[near] + sh[near], ring)
            out |= {(int(r), f) for r in rows_n[hit]}
        return out

    # Every backbone bond by its two atom ids, for the pairs joined through
    # one bond shorter than the clearance.
    short_joint = bool(len(r0) and float(np.min(r0)) < clearance)
    span = int(ids.max()) + 1 if len(ids) else 1
    bond_key = (np.minimum(ida, idb).astype(np.int64) * span + np.maximum(ida, idb))
    order_k = np.argsort(bond_key)
    bond_key, bond_row = bond_key[order_k], order_k

    def joined(i, j):
        """The row of a backbone bond joining bond rows ``i`` and ``j`` (-1 where none)."""
        row = np.full(len(i), -1)
        for x in (ida[i], idb[i]):
            for y in (ida[j], idb[j]):
                key = np.minimum(x, y).astype(np.int64) * span + np.maximum(x, y)
                at = np.minimum(np.searchsorted(bond_key, key), len(bond_key) - 1)
                row = np.where(bond_key[at] == key, bond_row[at], row)
        return row

    def survey(P):
        p1, p2 = P[sa], P[sb]
        mid = 0.5 * (p1 + p2)
        w = mid - box * np.floor(mid / box)
        w = np.clip(w, 0.0, np.nextafter(box, 0.0))
        pairs = cKDTree(w, boxsize=box).query_pairs(
            clearance + 2.0 * half_bond, output_type="ndarray")
        if not len(pairs):
            return (np.zeros(0, int),) * 2 + (np.zeros((0, 3)),) + (np.zeros(0),) * 4
        i, j = pairs[:, 0], pairs[:, 1]
        share = (ida[i] == ida[j]) | (ida[i] == idb[j]) | (idb[i] == ida[j]) | (idb[i] == idb[j])
        gap = np.abs(along[i] - along[j])
        gap = np.where(closed[i], np.minimum(gap, nseg[i] - gap), gap)
        keep = ~share & ~((chain[i] == chain[j]) & (gap <= 2))
        if bodies:
            keep &= ~(fixed[i] & fixed[j])
            keep &= ~np.isin(i.astype(np.int64) * len(sa) + j, near_keys)
        i, j = i[keep], j[keep]
        sh = -box * np.round((mid[j] - mid[i]) / box)
        s, t, d = segment_closest(p1[i], p2[i], p1[j] + sh, p2[j] + sh)
        short = d < clearance
        # what each pair is read against and pushed to: a little past the
        # clearance, since the bonds take some of every push back and a pair
        # aimed exactly at it only creeps up to it
        want = np.full(len(i), float(clearance))
        target = (1.0 + OVERSHOOT) * want
        if short_joint and short.any():
            # read against the joining bond as it now is: at angles over 90
            # degrees the pair is exactly its length apart, so it is short
            # only when the angle between the two strands closes
            row = joined(i, j)
            low = (row >= 0) & (r0[row] < clearance)
            if low.any():
                r_now = np.linalg.norm(P[sb[row]] - P[sa[row]], axis=1)
                want = np.where(low, (1.0 - tol) * r_now, want)
                target = np.where(low, np.minimum(target, r_now), target)
                short = d < want
        return (i[short], j[short], sh[short], s[short], t[short], d[short],
                target[short])

    def relax(P, r_want, t_want, e_want=None, f=1.0):
        terms = ((sa, sb, r_want), (ta, tb, t_want))
        if len(ea):
            # the off-path ring atoms' bonds and 1-3 distances
            terms = terms + ((ea, eb, e_want),)
        for _ in range(int(sweeps)):
            corr = np.zeros_like(P)
            count = np.zeros(len(P))
            for a, b, want in terms:
                v = P[b] - P[a]
                length = np.linalg.norm(v, axis=1)
                wa, wb = mobile[a], mobile[b]
                tot = wa + wb
                ok = tot > 0
                err = np.where(ok, (length - want) / np.maximum(length, 1e-9), 0.0)
                u = err[:, None] * v / np.where(ok, tot, 1.0)[:, None]
                np.add.at(corr, a, wa[:, None] * u)
                np.add.at(corr, b, -wb[:, None] * u)
                np.add.at(count, a, wa)
                np.add.at(count, b, wb)
            for rows_f in flat:
                # each ring onto its best plane, at the ramp's strength
                off, nrm = off_plane(P, rows_f)
                w = mobile[rows_f]
                np.add.at(corr, rows_f.ravel(),
                          (-(f * w * off)[:, :, None] * nrm[:, None, :]).reshape(-1, 3))
                np.add.at(count, rows_f.ravel(), w.ravel())
            P = P + corr / np.maximum(count, 1.0)[:, None]
        return P

    def off_plane(P, rows_f):
        """Each atom's signed distance from its ring's best plane, and the normals."""
        pw = P[rows_f]
        cen = pw.mean(axis=1, keepdims=True)
        nrm = np.linalg.svd(pw - cen)[2][:, 2, :]
        return np.einsum("wmi,wi->wm", pw - cen, nrm), nrm

    def errors(P):
        rb_ = np.linalg.norm(P[sb] - P[sa], axis=1) / r0 - 1.0
        rt_ = np.linalg.norm(P[tb] - P[ta], axis=1) / np.maximum(t_eq, 1e-9) - 1.0
        e_13 = float(np.abs(rt_).max()) if len(rt_) else 0.0
        if len(ea):
            re_ = np.linalg.norm(P[eb] - P[ea], axis=1) / e_eq - 1.0
            e_13 = max(e_13, float(np.abs(re_).max()))
            # a ring is flat within tol of its shortest bond
            e_13 = max(e_13, ring_off(P) / float(np.min(e_eq)))
        return (float(np.abs(rb_).max()) if len(rb_) else 0.0, e_13)

    def ring_off(P):
        """The farthest any ring atom is from its ring's plane, A."""
        return max((float(np.abs(off_plane(P, rows_f)[0]).max()) for rows_f in flat),
                   default=0.0)

    def push(P, found, target, normals=None):
        """Moves opening every pair in ``found`` to ``target`` (an array),
        along ``normals`` where they are given."""
        i, j, sh, s, t, d = found[:6]
        move = np.zeros_like(P)
        if not len(i):
            return move
        p1, p2 = P[sa[i]], P[sb[i]]
        q1, q2 = P[sa[j]] + sh, P[sb[j]] + sh
        if normals is not None:
            n = np.array(normals, float)
        else:
            n = (p1 + s[:, None] * (p2 - p1)) - (q1 + t[:, None] * (q2 - q1))
            flat = d < 1e-9
            if flat.any():
                c = np.cross(p2[flat] - p1[flat], q2[flat] - q1[flat])
                for k in np.flatnonzero(np.linalg.norm(c, axis=1) < 1e-9):
                    c[k] = _perpendicular(p2[flat][k] - p1[flat][k], rng)
                n[flat] = c * rng.choice([-1.0, 1.0], size=(len(c), 1))
        n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        offs = np.arange(-reach, reach + 2)
        rows_i = sa[i][:, None] + offs[None, :]
        rows_j = sa[j][:, None] + offs[None, :]
        lo_i, hi_i = row_lo[sa[i]][:, None], row_hi[sa[i]][:, None]
        lo_j, hi_j = row_lo[sa[j]][:, None], row_hi[sa[j]][:, None]
        in_i = (rows_i >= lo_i) & (rows_i <= hi_i)
        in_j = (rows_j >= lo_j) & (rows_j <= hi_j)
        rows_i = np.clip(rows_i, lo_i, hi_i)
        rows_j = np.clip(rows_j, lo_j, hi_j)
        wi = in_i * mobile[rows_i] * np.exp(-0.5 * ((offs[None, :] - s[:, None]) / width) ** 2)
        wj = in_j * mobile[rows_j] * np.exp(-0.5 * ((offs[None, :] - t[:, None]) / width) ** 2)
        # how far each closest point moves per unit of weight
        di = (1 - s) * wi[:, reach] + s * wi[:, reach + 1]
        dj = (1 - t) * wj[:, reach] + t * wj[:, reach + 1]
        gap = target - d
        fi = np.where(dj > 1e-9, np.where(di > 1e-9, 0.5, 0.0), 1.0)
        fj = np.where(di > 1e-9, 1.0 - fi, np.where(dj > 1e-9, 1.0, 0.0))
        gi = np.where(di > 1e-9, gap * fi / np.maximum(di, 1e-9), 0.0)
        gj = np.where(dj > 1e-9, gap * fj / np.maximum(dj, 1e-9), 0.0)
        np.add.at(move, rows_i.ravel(),
                  ((wi * gi[:, None])[:, :, None] * n[:, None, :]).reshape(-1, 3))
        np.add.at(move, rows_j.ravel(),
                  (-(wj * gj[:, None])[:, :, None] * n[:, None, :]).reshape(-1, 3))
        return move * mobile[:, None]

    def put_back(old, new, allowed=frozenset()):
        """The atoms of any two bonds that passed through each other going
        from ``old`` to ``new``, put back, until none did.

        ``allowed`` pairs (bond rows, low first) may pass: a pair with no
        side. With bodies, a backbone bond that has come to pass through a
        cage face (one not through it as drawn) is put back too. Returns the
        positions, ``old`` itself when 50 tries do not clear them, and the
        number of atoms put back.
        """
        nonlocal faces_put_back
        count = 0
        for _ in range(50):
            pi, pj, _tau, _big = passing_pairs(old, new, rows, pair_ids, box,
                                               fixed=fixed if bodies else None)
            if allowed and len(pi):
                other = np.array([(min(a, b), max(a, b)) not in allowed
                                  for a, b in zip(pi.tolist(), pj.tolist())], bool)
                pi, pj = pi[other], pj[other]
            into = (np.array(sorted({r for r, _f in threading(new) - threaded_drawn}), int)
                    if faces else np.zeros(0, int))
            if not len(pi) and not len(into):
                return new, count
            back = np.unique(np.concatenate([sa[pi], sb[pi], sa[pj], sb[pj],
                                             sa[into], sb[into]]))
            new[back] = old[back]
            count += len(back)
            faces_put_back += len(into)
        return old, count

    def part_sideless(P, found):
        """Part again, on one side, every contact the first parting left with none.

        Two strands drawn through one point at an atom of each touch in four
        bond pairs, and the first parting draws a side for each pair. Where
        both strands run straight through the point the four normals are one
        line, so a draw of two sides each way cancels and neither atom moves.
        If the two atoms are then the same point exactly, as on two strands
        drawn on one chord (a meander with a whole number of waves puts the
        middle atom of an even backbone on its chord's midpoint), the pair
        has a signed volume of exactly zero, every move of it reads as a
        passage, and the rounds put its atoms back for good. Each such
        contact (the pairs still touching, joined through the bonds they
        share) is pushed along one normal, its first sideless pair's, on one
        side drawn for the contact. A push that carries any other bond
        through one is put back, as in the rounds. Returns the positions and
        the count of pairs pushed; it draws nothing when every contact has a
        side.
        """
        i, j, sh = found[0], found[1], found[2]
        pushed = np.zeros(len(i), bool)
        # Once is enough for two strands; three strands through one point
        # can leave a pair in the plane of the first normal, taken next.
        for _ in range(3):
            p1, p2 = P[sa[i]], P[sb[i]]
            q1, q2 = P[sa[j]] + sh, P[sb[j]] + sh
            s, t, d = segment_closest(p1, p2, q1, q2)
            # the ends as the passage test reads them, so that what is
            # sideless here is what it reads as sideless
            ea, eb = bond_ends(P, rows, box)
            vol = signed_volume(ea[i], eb[i], ea[j] + sh, eb[j] + sh)
            none = (d < MEET) & (vol == 0.0)
            if not none.any():
                break
            near = np.flatnonzero(d < TOUCH)
            bonds, inv = np.unique(np.concatenate([i[near], j[near]]), return_inverse=True)
            m = len(near)
            graph = coo_matrix((np.ones(m), (inv[:m], inv[m:])), shape=(len(bonds),) * 2)
            contact = connected_components(graph, directed=False)[1][inv[:m]]
            sideless = np.unique(contact[none[near]])
            side = rng.choice([-1.0, 1.0], size=len(sideless))
            normals = np.zeros((m, 3))
            for c, sgn in zip(sideless, side):
                k = near[(contact == c) & none[near]][0]
                v = np.cross(p2[k] - p1[k], q2[k] - q1[k])
                if np.linalg.norm(v) < 1e-9:
                    v = _perpendicular(p2[k] - p1[k], rng)
                normals[contact == c] = sgn * v / np.linalg.norm(v)
            sub = np.isin(contact, sideless)
            rows_ = near[sub]
            new = P + push(P, (i[rows_], j[rows_], sh[rows_], s[rows_], t[rows_], d[rows_]),
                           np.full(len(rows_), 5.0 * TOUCH), normals=normals[sub])
            # the contacts' own pairs part either way, having no side
            own = frozenset((min(a, b), max(a, b))
                            for a, b in zip(i[rows_].tolist(), j[rows_].tolist()))
            P, _n = put_back(P, new, own)
            pushed[rows_] = True
        return P, int(pushed.sum())

    # The backbone bonds through the face of a ring a backbone runs through,
    # counted as the settle starts and as it ends, for the record only: the
    # settle does not act on them.
    ring_groups: dict = {}
    for r_i, cyc in enumerate(ring_cycles):
        ring_groups.setdefault(len(cyc), []).append(r_i)
    ring_groups = [(np.array(v, int), np.array([ring_cycles[r] for r in v], int))
                   for _n, v in sorted(ring_groups.items())]

    def ring_threads(P) -> int:
        """How many (backbone bond, ring) pairs have the bond through the
        ring's face, a ring's own bonds and its outside atoms' left out."""
        if not ring_cycles:
            return 0
        mids = 0.5 * (P[sa] + P[sb])
        tree = cKDTree(np.clip(mids - box * np.floor(mids / box), 0.0,
                               np.nextafter(box, 0.0)), boxsize=box)
        count = 0
        for ring_i, rows_r in ring_groups:
            poly = P[rows_r[:, :1]] + _min_image(P[rows_r] - P[rows_r[:, :1]], box)
            cen = poly.mean(axis=1)
            near = tree.query_ball_point(np.clip(cen - box * np.floor(cen / box), 0.0,
                                                 np.nextafter(box, 0.0)), 4.0)
            ks, gs = [], []
            for g, cand in enumerate(near):
                skip = ring_skip[ring_i[g]]
                for k_ in cand:
                    if int(sa[k_]) not in skip and int(sb[k_]) not in skip:
                        ks.append(k_)
                        gs.append(g)
            if not ks:
                continue
            ks, gs = np.array(ks, int), np.array(gs, int)
            pa = cen[gs] + _min_image(P[sa[ks]] - cen[gs], box)
            pb = pa + _min_image(P[sb[ks]] - P[sa[ks]], box)
            count += int(segments_through_rings(pa, pb, poly[gs]).sum())
        return count

    # what passes through a cage face as drawn (nothing, once clear_bodies
    # has turned the strands out), which no move may add to
    threaded_drawn = threading(X)
    faces_put_back = 0
    # Laying the shapes is a move of its own, before the rounds and their
    # frames (whose passages the record counts): the backbone bonds it
    # carries through each other are counted here, for the record only (on
    # the sample graph 7 to 11 per para build, 181 on a 2,5-pyridine build).
    # Putting their atoms back where they were drawn was tried and jammed
    # the pyridine build (1,110 pairs short) and left ringA unconverged.
    lay_passages = 0
    if X_drawn is not None:
        X_from = X.copy()
        X_from[:n_rows] = X_drawn
        lay_passages = len(passing_pairs(X_from, X, rows, pair_ids, box,
                                         fixed=fixed if bodies else None)[0])
    ring_threads_drawn = ring_threads(X)
    before = survey(X)
    n_before = len(before[0])
    worst_before = float(before[5].min()) if n_before else None
    angle_before = np.abs(_angles(X, ta, tb) - theta_eq) if len(ta) else np.zeros(0)
    # Two bonds drawn touching have no side, and every way of parting them
    # reads as one passing through the other. They are parted first, by a
    # hair and on a side drawn at random, and the record starts after that:
    # the only choices the pass makes. Only the pairs drawn touching may
    # come apart either way: a parting that carries any other bond through
    # one is put back like any move of the rounds (as the bead-spring
    # settle does). A contact the parting leaves with no side at all is
    # parted again, all its pairs on one side drawn for it.
    touching = before[5] < TOUCH
    sideless_parted = 0
    touching_put_back = 0
    if touching.any():
        found = tuple(x[touching] for x in before)
        drawn_touching = frozenset((min(a, b), max(a, b))
                                   for a, b in zip(found[0].tolist(), found[1].tolist()))
        X, touching_put_back = put_back(
            X, X + push(X, found, np.full(int(touching.sum()), 5.0 * TOUCH)),
            drawn_touching)
        X, sideless_parted = part_sideless(X, found)
    first_row = {}
    for r, a in enumerate(ids):
        first_row.setdefault(int(a), r)
    rows_out = np.array(sorted(first_row.values()), int)
    frames = [X[rows_out].copy()] if keep_frames else None
    X0 = X.copy()
    done = 0
    for done in range(1, int(rounds) + 1):
        f = min(1.0, done / max(int(ramp), 1))
        r_want = r_drawn + f * (r0 - r_drawn)
        t_want = t_drawn + f * (t_eq - t_drawn)
        e_want = e_drawn + f * (e_eq - e_drawn)
        found = survey(X)
        e_bond, e_13 = errors(X)
        # the side positions once the angles are at their targets
        cored = (sum(len(x[0]) for x in inside(X) + (sides_in(X) if f >= 1.0 else []))
                 if keep_out else 0)
        if f >= 1.0 and not len(found[0]) and e_bond < tol and e_13 < tol and not cored:
            done -= 1
            break
        # each pair to its target (survey): a little past the clearance
        move = push(X, found, found[6])
        if cored:
            move = move + keep_out_move(X, sides=f >= 1.0)
        new = relax(X + move, r_want, t_want, e_want, f)
        shift = new - X
        size = np.linalg.norm(shift, axis=1)
        new = X + shift * np.minimum(1.0, step / np.maximum(size, 1e-12))[:, None]
        # An atom pushed by two pairs at once, or pulled by its bonds and
        # angles, moves along no pair's line and can take a bond through a
        # third one close by. Whatever does is put back where it was.
        X, n_back = put_back(X, new)
        held_back += n_back
        if keep_frames:
            frames.append(X[rows_out].copy())

    after = survey(X)
    n_after = len(after[0])
    e_bond, e_13 = errors(X)
    cored_after = sum(len(x[0]) for x in inside(X)) if keep_out else 0
    sides_after = sum(len(x[0]) for x in sides_in(X)) if keep_out else 0
    angle_after = np.abs(_angles(X, ta, tb) - theta_eq) if len(ta) else np.zeros(0)
    ratio = (np.linalg.norm(X[sb] - X[sa], axis=1) / r0)[~fixed]

    def stats(x):
        return ({"mean": round(float(x.mean()), 3), "max": round(float(x.max()), 3)}
                if len(x) else None)

    report = {
        "clearance": float(clearance),
        "pairs_below_before": int(n_before),
        "closest_before": worst_before,
        "pairs_below_after": int(n_after),
        "closest_after": float(after[5].min()) if n_after else None,
        "angle_error_before": stats(angle_before),
        "angle_error_after": stats(angle_after),
        "bond_ratio": ({"min": float(ratio.min()), "max": float(ratio.max())}
                       if len(ratio) else None),
        "converged": bool(not n_after and e_bond < tol and e_13 < tol and not cored_after
                          and not sides_after),
        "rounds": int(done),
        "largest_shift": float(np.linalg.norm(X - X0, axis=1).max()) if len(X) else 0.0,
        "moves_put_back": int(held_back),
        "touching_parted": int(touching.sum()),
        "touching_moves_put_back": int(touching_put_back),
        "sideless_parted": int(sideless_parted),
    }
    if ring_ids:
        re_ = np.linalg.norm(X[eb] - X[ea], axis=1) / e_eq
        report["ring_atoms"] = len(ring_ids)
        report["ring_terms_ratio"] = {"min": float(re_.min()), "max": float(re_.max())}
        report["ring_off_plane"] = ring_off(X)
        report["ring_threads"] = {"drawn": ring_threads_drawn, "after": ring_threads(X)}
        report["ring_lay"] = {"passages": lay_passages}
        report["rings_laid_by_outside_atoms"] = turning
        # the off-path atoms' places, for place_network to take (not recorded)
        report["_ring_atoms"] = {a: X[n_rows + k] for k, a in enumerate(ring_ids)}
    if bodies:
        report["bodies"] = {
            "cages": len(bodies), "bonds": int(fixed.sum()), "faces": len(faces),
            "through_faces_drawn": len(threaded_drawn),
            "through_faces_after": len(threading(X)),
            "face_moves_put_back": int(faces_put_back),
            "keep_out": BODY_CLEARANCE,
            "inside_keep_out_after": int(cored_after),
            "sides_inside_after": int(sides_after)}
    points = [X[s0:s0 + m] for s0, m in zip(starts[:n_strands], sizes[:n_strands])]
    out_frames = (ids[rows_out], frames) if keep_frames else None
    return points, report, out_frames


def settle_warning(settle: dict) -> Optional[str]:
    """The warning for a backbone settle that did not finish, or None.

    ``settle`` is :func:`settle_backbones`'s report. Pairs of backbone bonds
    left closer than the clearance come first, since the first stage may
    push them through each other. A settle stopped at its round cap with
    none short warns too, a build with rings in its backbones with its
    bonds, angles and rings, and since 0.4.5 every other build with
    the range of its backbone bonds against r0 and its angle error, and the
    backbone atoms a cage still holds inside its keep-out sphere or with a
    side position within a C-H of its core. Before 0.4.5 such a build said
    nothing, and on the POSS demo config's own generated networks (25 caps)
    the settle stopped at its cap on 5 of 6, two of them with no pair short,
    angles up to 25 degrees off and no warning. One warning at most.
    """
    if not settle:
        return None
    if settle["pairs_below_after"]:
        return (f"{settle['pairs_below_after']} pair(s) of backbone bonds are still "
                f"closer than {settle['clearance']} A after {settle['rounds']} rounds "
                f"(closest {settle['closest_after']:.2f} A); the first stage may "
                f"push them through each other")
    if settle["converged"]:
        return None
    if settle.get("ring_atoms"):
        # a build with backbone rings stopped at its round cap with no pair
        # short: its bonds and rings as the settle left them
        return (f"the backbone settle stopped at its cap of {settle['rounds']} rounds "
                f"before its bonds and rings were within 2 %: backbone bond / r0 "
                f"{settle['bond_ratio']}, angle error {settle['angle_error_after']} degrees, "
                f"ring terms / target {settle['ring_terms_ratio']}, rings off their "
                f"plane by up to {settle['ring_off_plane']:.3f} A")
    ratio = settle.get("bond_ratio") or {}
    angle = settle.get("angle_error_after") or {}
    text = (f"the backbone settle stopped at its cap of {settle['rounds']} rounds with no "
            f"pair of backbone bonds closer than {settle['clearance']} A, but before its "
            f"bonds and angles were within 2 %: backbone bond / r0 "
            f"{ratio.get('min', float('nan')):.3f} to {ratio.get('max', float('nan')):.3f}, "
            f"angle error up to {angle.get('max', float('nan')):.1f} degrees")
    bodies = settle.get("bodies") or {}
    if bodies.get("inside_keep_out_after"):
        text += (f", {bodies['inside_keep_out_after']} backbone atom(s) inside a cage's "
                 f"keep-out sphere")
    if bodies.get("sides_inside_after"):
        text += (f", {bodies['sides_inside_after']} with a side position within a C-H "
                 f"of a cage's core")
    return text


def one_three(r0, theta0) -> np.ndarray:
    """The distance between each atom's two chain neighbours at equilibrium.

    ``r0`` one per bond, ``theta0`` one per interior atom, in degrees.
    """
    r0 = np.asarray(r0, float)
    th = np.deg2rad(np.asarray(theta0, float))
    return np.sqrt(r0[:-1] ** 2 + r0[1:] ** 2 - 2.0 * r0[:-1] * r0[1:] * np.cos(th))


# ---------------------------------------------------------------------------
# Everything that is not backbone
# ---------------------------------------------------------------------------

def _perpendicular(v, rng) -> np.ndarray:
    v = np.asarray(v, float)
    h = rng.normal(size=3)
    h = h - v * float(h @ v) / max(float(v @ v), 1e-12)
    n = float(np.linalg.norm(h))
    if n < 1e-9:
        h = np.cross(v, [1.0, 0.0, 0.0])
        if np.linalg.norm(h) < 1e-9:
            h = np.cross(v, [0.0, 1.0, 0.0])
        n = float(np.linalg.norm(h))
    return h / n


def tetrahedral_directions(known, m: int, rng) -> list:
    """``m`` unit vectors for new bonds from an atom whose bonds ``known`` has.

    sp3 geometry: with one known bond the new ones sit at 109.47 degrees
    from it, spread 120 degrees about it (random azimuth); with two, the two
    remaining tetrahedral positions, or their bisector for one; with three,
    opposite their sum. A straight backbone through the atom (two known
    bonds at 180 degrees) puts the new ones perpendicular to it. Beyond the
    four tetrahedral positions, directions are random away from the known
    ones.
    """
    known = [np.asarray(v, float) / float(np.linalg.norm(v))
             for v in known if float(np.linalg.norm(v)) > 1e-12]
    out: list = []
    if m <= 0:
        return out
    if not known:
        first = rng.normal(size=3)
        out.append(first / np.linalg.norm(first))
        known = [out[0]]
    need = m - len(out)
    if len(known) == 1:
        a = known[0]
        p1 = _perpendicular(a, rng)
        p2 = np.cross(a, p1)
        phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
        for j in range(min(need, 3)):
            phi = phi0 + 2.0 * np.pi * j / 3.0
            out.append(np.cos(_TET) * a + np.sin(_TET) * (np.cos(phi) * p1
                                                          + np.sin(phi) * p2))
    elif len(known) == 2:
        a1, a2 = known
        bis = -(a1 + a2)
        if np.linalg.norm(bis) < 1e-6:
            bis = _perpendicular(a1, rng)
        bis = bis / np.linalg.norm(bis)
        nrm = np.cross(a1, a2)
        if np.linalg.norm(nrm) < 1e-6:
            nrm = np.cross(bis, a1)
        nrm = nrm / np.linalg.norm(nrm)
        if need == 1:
            out.append(bis)
        else:
            half = _TET / 2.0
            out.append(np.cos(half) * bis + np.sin(half) * nrm)
            out.append(np.cos(half) * bis - np.sin(half) * nrm)
    else:
        d = -np.sum(known, axis=0)
        if np.linalg.norm(d) < 1e-6:
            d = _perpendicular(known[0], rng)
        out.append(d / np.linalg.norm(d))
    taken = known + out
    for _ in range(1000):
        if len(out) >= m:
            break
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if all(float(v @ t) < 0.3 for t in taken):
            out.append(v)
            taken.append(v)
    while len(out) < m:                 # crowded beyond any sensible valence
        v = rng.normal(size=3)
        out.append(v / np.linalg.norm(v))
    return out[:m]


def free_rings(neighbours: dict, placed) -> dict:
    """The simple rings among the atoms not in ``placed``, by atom.

    A ring here is a biconnected piece of the unplaced atoms' bond graph
    that is a single cycle of 3 to :data:`RING_MAX` atoms (a phenyl, a
    cyclohexyl). A fused ring system has more bonds than atoms in its piece
    and is left out, so its atoms keep the tree walk. Returns ``{atom:
    ring}``, ``ring`` the cycle's atoms in bond order from its smallest id.
    """
    import networkx as nx

    g = nx.Graph()
    for a, nbs in neighbours.items():
        a = int(a)
        if a in placed:
            continue
        for b in nbs:
            b = int(b)
            if b not in placed and a < b:
                g.add_edge(a, b)
    rings: dict = {}
    for comp in nx.biconnected_components(g):
        if not 3 <= len(comp) <= RING_MAX:
            continue
        sub = g.subgraph(comp)
        if sub.number_of_edges() != len(comp):
            continue
        start = min(comp)
        cycle, prev, cur = [start], None, start
        while True:
            nxt = min(n for n in sub.neighbors(cur) if n != prev)
            if nxt == start:
                break
            cycle.append(nxt)
            prev, cur = cur, nxt
        for a in cycle:
            rings[a] = cycle
    return rings


def _hangs_from_one_atom(ring, entry, neighbours, placed) -> bool:
    """Whether ``ring``, just entered at ``entry``, is bonded to the placed atoms there only."""
    if sum(1 for n in neighbours.get(entry, ()) if n in placed) != 1:
        return False
    members = set(ring)
    return all(m not in placed
               and not any(n in placed and n not in members for n in neighbours.get(m, ()))
               for m in ring if m != entry)


def ring_polygon(ring, entry: int, at, outward, bond_r0: dict, rng,
                 default_r0: float = 1.5) -> dict:
    """The rest of ``ring`` as a flat regular polygon from ``entry``.

    ``entry`` sits at ``at`` and the ring points along ``outward`` (the
    direction of the bond it hangs from), so the atom opposite the entry is
    furthest from the parent, as the para carbon of a phenyl. The side is
    the mean equilibrium length of the ring's bonds, and the ring's plane
    is turned about ``outward`` at random. Returns the positions of every
    ring atom but ``entry``.
    """
    k = len(ring)
    i0 = ring.index(entry)
    order = list(ring[i0:]) + list(ring[:i0])
    side = float(np.mean([bond_r0.get((min(a, b), max(a, b)), default_r0)
                          for a, b in zip(order, order[1:] + order[:1])]))
    radius = side / (2.0 * np.sin(np.pi / k))
    a = np.asarray(outward, float)
    a = a / float(np.linalg.norm(a))
    u = _perpendicular(a, rng)
    centre = np.asarray(at, float) + radius * a
    out = {}
    for j, m in enumerate(order[1:], start=1):
        phi = 2.0 * np.pi * j / k
        out[m] = centre + radius * (-np.cos(phi) * a + np.sin(phi) * u)
    return out


def ring_template(ring, run, exo, bond_r0, default_r0: float = 1.5) -> dict:
    """A flat ring and the atoms bonded to the ends of a run through it.

    ``run`` is two or more consecutive atoms of ``ring`` in the order a
    backbone takes them, ``exo`` maps an end atom of the run to ``(atom,
    r0)``, the backbone atom bonded to it outside the ring. The ring is the
    flat polygon on a circle with every side at its own bond's equilibrium
    length: a regular hexagon for a phenylene, with a pyridine's two C-N
    sides shorter. Each exo atom sits on its end atom's exterior bisector,
    120 degrees from both ring bonds of a regular hexagon, so a
    para-phenylene's two exo atoms lie on its para axis. Returns ``{atom:
    xyz}`` in the ring's own frame, for every ring atom and every exo atom.
    """
    k = len(ring)
    i0 = ring.index(run[0])
    step = 1 if ring[(i0 + 1) % k] == run[1] else -1
    order = [ring[(i0 + step * t) % k] for t in range(k)]
    sides = np.array([bond_r0.get((min(a, b), max(a, b)), default_r0)
                      for a, b in zip(order, order[1:] + order[:1])], float)
    # the circle the sides span: their central angles add to a full turn
    lo, hi = 0.5 * float(sides.max()), float(sides.sum())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.sum(2.0 * np.arcsin(np.minimum(1.0, sides / (2.0 * mid)))) > 2.0 * np.pi:
            lo = mid
        else:
            hi = mid
    radius = 0.5 * (lo + hi)
    phi = np.concatenate([[0.0], np.cumsum(2.0 * np.arcsin(sides[:-1] / (2.0 * radius)))])
    pos = {a: radius * np.array([np.cos(f), np.sin(f), 0.0]) for a, f in zip(order, phi)}
    for end, (atom, r) in exo.items():
        t = order.index(end)
        u = pos[order[t - 1]] - pos[end]
        v = pos[order[(t + 1) % k]] - pos[end]
        out = -(u / np.linalg.norm(u) + v / np.linalg.norm(v))
        pos[atom] = pos[end] + float(r) * out / np.linalg.norm(out)
    return pos


def ring_runs(chain, rings) -> list:
    """Every stretch of ``chain`` through one ring, as ``(i, j, ring)``.

    ``rings`` is ``{atom: ring}`` (:func:`free_rings`). A stretch is rows
    ``i`` to ``j`` of the chain, two or more consecutive atoms of one ring:
    the ipso, ortho, meta and para carbons of a para-phenylene that the
    backbone (the shortest path from head to tail) runs along.
    """
    out = []
    i, n = 0, len(chain)
    while i < n - 1:
        ring = rings.get(chain[i])
        j = i
        if ring is not None:
            members = set(ring)
            while j + 1 < n and chain[j + 1] in members:
                j += 1
        if j > i:
            out.append((i, j, ring))
            i = j
        else:
            i += 1
    return out


def ring_windows(chain, r0, rings, bond_r0, default_r0: float = 1.5):
    """What the settle needs to hold each ring a backbone runs through.

    A window is a stretch (:func:`ring_runs`) and the backbone atom on each
    side of it, the exo atoms, at their places in the flat ring
    (:func:`ring_template`). ``r0`` is the chain's bond lengths, one per
    bond. Returns ``(one_three, rings_held)``: ``{k: distance}``, the 1-3
    distance across chain atom ``k + 1`` wherever a window sets it (120
    degrees at a phenylene's carbons), and one dict per stretch:
    ``template`` (``{atom: xyz}``, the flat ring and its exo atoms), ``fit``
    (the window's atoms in chain order, which the settle lays the template
    on), ``extra`` (the ring's off-path atoms, which the settle adds) and
    ``terms`` (``[(a, b, distance)]``, every ring bond and every 1-3 distance
    round a ring atom that has an off-path atom in it). Two windows may share
    atoms (two rings bonded to each other, or on one ether O); their terms
    agree there.
    """
    one3, held_rings = {}, []
    for i, j, ring in ring_runs(chain, rings):
        exo, rows = {}, list(range(i, j + 1))
        if i > 0:
            exo[chain[i]] = (chain[i - 1], r0[i - 1])
            rows.insert(0, i - 1)
        if j + 1 < len(chain):
            exo[chain[j]] = (chain[j + 1], r0[j])
            rows.append(j + 1)
        run = list(chain[i:j + 1])
        pos = ring_template(ring, run, exo, bond_r0, default_r0)
        for x, p in enumerate(rows[:-2]):
            one3[p] = float(np.linalg.norm(pos[chain[rows[x + 2]]] - pos[chain[p]]))
        on_path = set(run)
        extra = [a for a in ring if a not in on_path]
        off = set(extra)
        k = len(ring)
        out_of = {end: atom for end, (atom, _r) in exo.items()}
        terms = []
        for t in range(k):
            a, b = ring[t], ring[(t + 1) % k]
            if a in off or b in off:
                terms.append((a, b, float(np.linalg.norm(pos[b] - pos[a]))))
        for t in range(k):
            centre = ring[t]
            nbs = [ring[t - 1], ring[(t + 1) % k]] + (
                [out_of[centre]] if centre in out_of else [])
            for x in range(len(nbs)):
                for y in range(x + 1, len(nbs)):
                    a, b = nbs[x], nbs[y]
                    if a in off or b in off:
                        terms.append((a, b, float(np.linalg.norm(pos[b] - pos[a]))))
        held_rings.append({"template": pos, "fit": [chain[p] for p in rows],
                           "extra": extra, "terms": terms, "ring": list(ring)})
    return one3, held_rings


def _min_image(d, box) -> np.ndarray:
    return d - box * np.round(d / box)


def _turned(point, a, b, phi) -> np.ndarray:
    """``point`` turned by ``phi`` about the line through ``a`` and ``b``."""
    axis = np.asarray(b, float) - np.asarray(a, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    v = np.asarray(point, float) - a
    return (a + v * np.cos(phi) + np.cross(axis, v) * np.sin(phi)
            + axis * float(axis @ v) * (1.0 - np.cos(phi)))


#: How far (degrees) a ring may turn the backbone and still be laid by its
#: ring atoms alone: a para-phenylene turns it by 0, a meta-phenylene
#: by 60 and a 2,5-furan by 36 (:func:`_turns`).
RING_TURN = 15.0


def _turns(shape) -> float:
    """By how much (degrees) a ring stretch's flat shape turns the backbone:
    the angle between its two outside bonds, read in the direction of the
    chain (``shape`` is the window's atoms in chain order, the outside atoms
    first and last)."""
    a = shape[1] - shape[0]
    b = shape[-1] - shape[-2]
    return float(np.degrees(np.arccos(np.clip(
        a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


def _laid(template, points, fixed) -> tuple:
    """The rotation and shift laying ``template`` on ``points`` (Kabsch),
    the ``fixed`` points weighing most."""
    w = np.where(fixed, 1.0e3, 1.0)[:, None]
    t = np.asarray(template, float)
    p = np.asarray(points, float)
    ct, cp = (w * t).sum(axis=0) / w.sum(), (w * p).sum(axis=0) / w.sum()
    u, _s, vt = np.linalg.svd(((t - ct) * w).T @ (p - cp))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, cp - rot @ ct


def _fit(template, target):
    """The rotation and shift laying ``template`` points on ``target`` (Kabsch)."""
    a, b = np.asarray(template, float), np.asarray(target, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, cb - rot @ ca


def close_path_rings(rings, placed, neighbours, bond_r0, box,
                     default_r0: float = 1.5) -> tuple[dict, dict]:
    """The other atoms of every ring a placed backbone runs through.

    A ring whose placed atoms are one run along it (the backbone's half of
    a para-phenylene) gets the rest from its flat ring
    (:func:`ring_template`) laid on that run, or on the run and its placed
    exo atoms when the run has two atoms. Before 0.4.5 the tree walk placed
    the two off-path carbons of a para-phenylene from opposite ends, one
    from the ipso carbon and one from the para, and left the bond between
    them anywhere from 1.2 to 6 A. Returns ``(positions, report)``: the new
    atoms, and how many rings were closed and the largest distance (A) of a
    placed atom from the fitted ring, 0 for a run the settle held flat.
    """
    box = None if box is None else np.asarray(box, float)
    new, worst, closed = {}, 0.0, 0
    for ring in {tuple(r): r for r in rings.values()}.values():
        k = len(ring)
        on = [a in placed for a in ring]
        if all(on) or not any(on):
            continue
        starts = [t for t in range(k) if on[t] and not on[t - 1]]
        if len(starts) != 1:
            continue
        run = []
        t = starts[0]
        while on[t % k] and len(run) < k:
            run.append(ring[t % k])
            t += 1
        if len(run) < 2:
            continue
        members = set(ring)
        exo = {}
        for end in (run[0], run[-1]):
            out = [n for n in neighbours.get(end, ()) if n in placed and n not in members]
            if len(out) == 1:
                exo[end] = (out[0], bond_r0.get((min(end, out[0]), max(end, out[0])),
                                                default_r0))
        # three run atoms fix the ring (and an exo atom is off its plane
        # when the settle did not run); two need the exo atoms too
        fit = run if len(run) >= 3 else run + [x for x, _r in exo.values()]
        if len(fit) < 3:
            continue
        pos = ring_template(ring, run, exo, bond_r0, default_r0)
        ref = placed[run[0]]
        tgt = []
        for a in fit:
            d = placed[a] - ref
            if box is not None:
                d = d - box * np.round(d / box)
            tgt.append(ref + d)
        rot, shift = _fit([pos[a] for a in fit], tgt)
        worst = max(worst, max(float(np.linalg.norm(rot @ pos[a] + shift - x))
                               for a, x in zip(fit, tgt)))
        for a in ring:
            if a not in placed:
                new[a] = rot @ pos[a] + shift
        closed += 1
    return new, {"closed": closed, "fit_off_max": worst}


#: Draws of a substituent group's directions tried when the first would put
#: a bond through a cage face (:func:`place_substituents`).
SUBSTITUENT_TRIES = 24

#: The turns a ring hanging from one atom is tried at about the bond it
#: hangs from, from the one drawn. A flat ring turned half a turn is
#: its mirror image, so twelve steps of 30 degrees cover both faces.
PENDANT_TURNS = 12

#: A ring hanging from one atom is turned, among the turns no bond threads,
#: (and, where its parent has other atoms to place, put on the direction of
#: the one) with the fewest backbone atoms nearer than this (A) to an atom,
#: more than three bonds apart. On the copolymer demo phenyl carbons
#: were placed 0.4 to 0.6 A from a backbone Si or O of their own strand, and
#: a stage 1 that keeps ring atoms hard pushed those apart hard enough in
#: its first steps to carry a backbone bond through its neighbour.
PENDANT_DEEP = 1.5


def _face_shape(poly) -> tuple:
    """A ring's centre, the normal of its best plane, its largest vertex
    radius and how far its vertices stray from that plane: what
    :func:`_may_cross` reads."""
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    n = np.linalg.svd(poly - c)[2][2]
    return (c, n, float(np.linalg.norm(poly - c, axis=1).max()),
            float(np.abs((poly - c) @ n).max()))


def _may_cross(pa, pb, shape) -> bool:
    """False only when the segment ``pa``-``pb`` cannot meet the ring of
    ``shape`` (:func:`_face_shape`) read as the fan of triangles from its
    centre: both ends beyond the fan's spread on one side of its plane, or
    no point of the segment within its largest radius of the centre. A
    cheap test before :func:`~topon.conformation.segments.segments_through_ring`,
    which it never contradicts."""
    c, n, rmax, dev = shape
    da = float((pa - c) @ n)
    db = float((pb - c) @ n)
    m = dev + 1e-6
    if (da > m and db > m) or (da < -m and db < -m):
        return False
    d = pb - pa
    l2 = float(d @ d)
    t = 0.0 if l2 < 1e-24 else min(1.0, max(0.0, float((c - pa) @ d) / l2))
    return float(np.linalg.norm(pa + t * d - c)) <= rmax + 1e-6


class _Grid:
    """Points in cells of about ``cell`` A, under the minimum image of
    ``box`` (None: no box), for the items near a point."""

    def __init__(self, box, cell: float = 4.0):
        self.box = None if box is None else np.asarray(box, float).reshape(3)
        if self.box is None:
            self.n = None
            self.size = np.full(3, cell)
        else:
            self.n = np.maximum(1, np.floor(self.box / cell)).astype(int)
            self.size = self.box / self.n
        self.cells: dict = {}
        self._around: dict = {}

    def _key(self, x):
        x = np.asarray(x, float)
        if self.box is not None:
            x = x - self.box * np.floor(x / self.box)
        k = np.floor(x / self.size).astype(int)
        return tuple(int(v) for v in (k if self.n is None else np.mod(k, self.n)))

    def add(self, x, item):
        self.cells.setdefault(self._key(x), []).append(item)

    def near(self, x):
        k = self._key(x)
        keys = self._around.get(k)
        if keys is None:
            keys = set()
            for d0 in (-1, 0, 1):
                for d1 in (-1, 0, 1):
                    for d2 in (-1, 0, 1):
                        t = (k[0] + d0, k[1] + d1, k[2] + d2)
                        if self.n is not None:
                            t = tuple(int(v) for v in np.mod(t, self.n))
                        keys.add(t)
            keys = self._around[k] = sorted(keys)
        out = []
        for t in keys:
            out.extend(self.cells.get(t, ()))
        return out

    def image(self, x, ref):
        """``x`` in the image nearest ``ref``."""
        x = np.asarray(x, float)
        return x if self.box is None else ref + _min_image(x - ref, self.box)


class _PendantRings:
    """What a ring hanging from one atom is placed against: every
    bond placed so far, and the faces of every ring placed so far, the
    backbones' and cages' read through ``avoid``."""

    def __init__(self, neighbours, placed, box, avoid, bond_r0, default_r0):
        self.neighbours = neighbours
        self.avoid = avoid
        self.bond_r0 = bond_r0
        self.default_r0 = default_r0
        self.bonds = _Grid(box)
        self.faces = _Grid(box)
        # the atoms placed before the walk (backbones, junctions, cages)
        self.frame = _Grid(box, cell=PENDANT_DEEP + 2.5)
        for a in sorted(placed):
            self.frame.add(placed[a], a)
        self.polys: list = []
        self.shapes: list = []
        self.rings: list = []
        self.turned = 0
        self.left = 0
        self.left_bonds = 0
        self.deep_left = 0
        self.swapped = 0
        self._close: dict = {}
        for a in sorted(placed):
            for b in neighbours.get(a, ()):
                if b in placed and b < a:
                    self._add(a, b, placed)

    def placed_atom(self, a, placed):
        """Every bond of ``a``, just placed, to an atom placed before it."""
        for b in self.neighbours.get(a, ()):
            if b in placed and b != a:
                self._add(a, b, placed)

    def _add(self, a, b, placed):
        self.bonds.add(0.5 * (placed[a] + self.bonds.image(placed[b], placed[a])), (a, b))

    def crossed(self, a, b) -> bool:
        """Whether the bond ``a``-``b`` (A) passes through a pendant ring
        placed so far (not one it starts or ends on)."""
        mid = 0.5 * (np.asarray(a, float) + np.asarray(b, float))
        for k in self.faces.near(mid):
            poly = self.polys[k]
            c = self.shapes[k][0]
            pa = self.faces.image(a, c)
            pb = pa + (np.asarray(b, float) - np.asarray(a, float)
                       if self.faces.box is None else _min_image(np.asarray(b, float)
                                                                 - np.asarray(a, float),
                                                                 self.faces.box))
            if not _may_cross(pa, pb, self.shapes[k]):
                continue
            if min(np.linalg.norm(poly - pa, axis=1).min(),
                   np.linalg.norm(poly - pb, axis=1).min()) < 0.05:
                continue
            if segments_through_ring(pa[None], pb[None], poly)[0]:
                return True
        return False

    def near(self, ring, centre, placed, reach: float = 2.6) -> tuple:
        """The placed bonds that could pass through a ring centred at
        ``centre`` (A): their midpoints within ``reach`` of it (the ring's
        radius and half the longest bond, 2.6 A for a phenyl), none of the
        ring's own, each in the image nearest the centre. A ring turned
        about the bond it hangs from keeps its centre, so this serves every
        turn."""
        members = set(ring)
        pa, pb = [], []
        for a, b in self.bonds.near(centre):
            if a in members or b in members:
                continue
            x = self.bonds.image(placed[a], centre)
            y = x + (placed[b] - placed[a] if self.bonds.box is None
                     else _min_image(placed[b] - placed[a], self.bonds.box))
            if float(np.linalg.norm(0.5 * (x + y) - centre)) < reach:
                pa.append(x)
                pb.append(y)
        return np.array(pa, float).reshape(-1, 3), np.array(pb, float).reshape(-1, 3)

    def threads(self, ring, xyz, placed, near=None) -> int:
        """How many placed bonds pass through the ring at ``xyz`` (its atoms
        in ring order), and how many of its bonds through another ring:
        its ring bonds, and the bond each ring atom with one atom still to
        place will have to it, which the walk puts in the ring's plane on
        the atom's exterior bisector (a phenyl's hydrogens). ``near`` is
        :meth:`near` for the ring's centre."""
        poly = np.array(xyz)
        members = set(ring)
        pa, pb = near if near is not None else self.near(ring, poly.mean(axis=0), placed)
        count = int(segments_through_rings(
            pa, pb, np.repeat(poly[None], len(pa), axis=0)).sum()) if len(pa) else 0
        ends = [(x, y) for x, y in zip(poly, np.roll(poly, -1, axis=0))]
        k = len(ring)
        for t in range(1, k):
            m = ring[t]
            rest = [n for n in self.neighbours.get(m, ()) if n not in members and n not in placed]
            if len(rest) != 1:
                continue
            out = 2.0 * poly[t] - poly[t - 1] - poly[(t + 1) % k]
            norm = float(np.linalg.norm(out))
            if norm < 1e-9:
                continue
            r = self.bond_r0.get((min(m, rest[0]), max(m, rest[0])), self.default_r0)
            ends.append((poly[t], poly[t] + r * out / norm))
        hit = self._crossed_many(ends)
        if self.avoid is not None:
            hit = [h or self.avoid(x, y) for h, (x, y) in zip(hit, ends)]
        return count + int(sum(hit))

    def _crossed_many(self, ends) -> list:
        """:meth:`crossed` for many bonds at once."""
        pa, pb, ring_i, end_i = [], [], [], []
        for e, (a, b) in enumerate(ends):
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            for k in self.faces.near(0.5 * (a + b)):
                poly = self.polys[k]
                c = self.shapes[k][0]
                x = self.faces.image(a, c)
                y = x + (b - a if self.faces.box is None else _min_image(b - a, self.faces.box))
                if not _may_cross(x, y, self.shapes[k]):
                    continue
                if min(np.linalg.norm(poly - x, axis=1).min(),
                       np.linalg.norm(poly - y, axis=1).min()) < 0.05:
                    continue
                pa.append(x)
                pb.append(y)
                ring_i.append(k)
                end_i.append(e)
        out = [False] * len(ends)
        if not pa:
            return out
        polys = np.array([self.polys[k] for k in ring_i])
        for e, h in zip(end_i, segments_through_rings(np.array(pa), np.array(pb), polys)):
            if h:
                out[e] = True
        return out

    def frame_near(self, ring, centre, placed, reach: float = 3.0) -> np.ndarray:
        """The backbone atoms (those placed before the walk) within ``reach``
        of a ring's centre, more than three bonds from the ring, each in the
        image nearest the centre: every one that can come within
        :data:`PENDANT_DEEP` of a ring atom (``reach`` 3 A for a phenyl,
        whose atoms are 1.4 A from its centre)."""
        key = tuple(sorted(ring))
        near_bonds = self._close.get(key)
        if near_bonds is None:
            near_bonds = set(ring)
            frontier = set(ring)
            for _ in range(3):
                frontier = {n for a in frontier for n in self.neighbours.get(a, ())} - near_bonds
                near_bonds |= frontier
            self._close[key] = near_bonds
        xs = []
        for b in self.frame.near(centre):
            if b in near_bonds:
                continue
            x = self.frame.image(placed[b], centre)
            if float(np.linalg.norm(x - centre)) < reach:
                xs.append(x)
        return np.array(xs, float).reshape(-1, 3)

    def deep(self, ring, xyz, placed, frame=None) -> int:
        """How many backbone atoms (those placed before the walk) sit nearer
        than :data:`PENDANT_DEEP` to an atom of the ring at ``xyz``, more
        than three bonds apart (pairs counted). ``frame`` is
        :meth:`frame_near` for the ring's centre."""
        xyz = np.asarray(xyz, float)
        if frame is None:
            frame = self.frame_near(ring, xyz.mean(axis=0), placed)
        if not len(frame):
            return 0
        d = np.linalg.norm(xyz[:, None, :] - frame[None, :, :], axis=2)
        return int((d < PENDANT_DEEP).sum())

    def _around(self, ring, centre, radius, placed) -> tuple:
        """:meth:`near` and :meth:`frame_near` with reaches from the ring's
        radius: never under a phenyl's 2.6 and 3.0 A, so a phenyl reads as
        it did, and wide enough for an eight-ring."""
        return (self.near(ring, centre, placed, reach=max(2.6, radius + 1.3)),
                self.frame_near(ring, centre, placed,
                                reach=max(3.0, radius + PENDANT_DEEP + 0.05)))

    def score(self, ring, entry, at, outward, placed, default_r0) -> tuple:
        """The best (threads, backbone atoms too near) of the ring hanging
        from ``entry`` at ``at`` along ``outward``, over its turns."""
        order = list(ring[ring.index(entry):]) + list(ring[:ring.index(entry)])
        k = len(order)
        side = float(np.mean([self.bond_r0.get((min(a, b), max(a, b)), default_r0)
                              for a, b in zip(order, order[1:] + order[:1])]))
        radius = side / (2.0 * np.sin(np.pi / k))
        a = np.asarray(outward, float) / float(np.linalg.norm(outward))
        u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        centre = at + radius * a
        flat = [at] + [centre + radius * (-np.cos(2 * np.pi * j / k) * a
                                          + np.sin(2 * np.pi * j / k) * u)
                       for j in range(1, k)]
        best = None
        near, frame = self._around(order, centre, radius, placed)
        for turn in range(PENDANT_TURNS):
            phi = 2.0 * np.pi * turn / PENDANT_TURNS
            xyz = [at] + [_turned(x, at, at + a, phi) for x in flat[1:]]
            key = (self.threads(order, xyz, placed, near), self.deep(order, xyz, placed, frame))
            if best is None or key < best:
                best = key
            if key == (0, 0):
                break
        return best

    def assign(self, parent, todo, dirs, placed, rings, default_r0) -> list:
        """``dirs`` for the atoms ``todo`` of ``parent``, with a ring among
        them that hangs from one atom moved to the direction that suits it
        best, when the one drawn for it leaves it threaded or on a backbone
        atom (a phenyl and a methyl on one Si swap)."""
        entries = [k for k, n in enumerate(todo) if rings.get(n) is not None
                   and _hangs_from_one_atom(rings[n], n, self.neighbours, placed)]
        if len(entries) != 1 or len(todo) < 2:
            return dirs
        k0 = entries[0]
        n = todo[k0]
        r = self.bond_r0.get((min(parent, n), max(parent, n)), default_r0)
        keys = [self.score(rings[n], n, placed[parent] + r * d, d, placed, default_r0)
                for d in dirs]
        best = min(range(len(dirs)), key=lambda k: (keys[k], k != k0))
        if keys[best] < keys[k0]:
            dirs = list(dirs)
            dirs[k0], dirs[best] = dirs[best], dirs[k0]
            self.swapped += 1
        return dirs

    def choose(self, ring, entry, at, outward, drawn, placed) -> dict:
        """The ring's atoms (but ``entry``) at the first turn about
        ``outward``, from the one ``drawn``, that no bond threads; the least
        threaded when none is clear."""
        order = list(ring[ring.index(entry):]) + list(ring[:ring.index(entry)])
        at = np.asarray(at, float)
        best = None
        centre = np.mean([at] + [drawn[m] for m in order[1:]], axis=0)
        near, frame = self._around(order, centre, float(np.linalg.norm(at - centre)), placed)
        for turn in range(PENDANT_TURNS):
            phi = 2.0 * np.pi * turn / PENDANT_TURNS
            cand = {m: _turned(x, at, at + outward, phi) for m, x in drawn.items()}
            xyz = [at] + [cand[m] for m in order[1:]]
            key = (self.threads(order, xyz, placed, near),
                   self.deep(order, xyz, placed, frame))
            if best is None or key < best[0]:
                best = (key, turn, cand)
            if key == (0, 0):
                break
        (hits, deep), turn, cand = best
        self.deep_left += bool(deep)
        self.turned += bool(turn)
        self.left += bool(hits)
        self.left_bonds += hits
        self.rings.append(order)
        poly = np.array([at] + [cand[m] for m in order[1:]])
        self.polys.append(poly)
        self.shapes.append(_face_shape(poly))
        self.faces.add(poly.mean(axis=0), len(self.polys) - 1)
        return cand


def ring_face_test(polys, box):
    """``(test, crossed)`` for the flat rings ``polys`` (each ``(r, 3)``, its
    atoms in ring order, one image): ``test(a, b)`` is whether the bond from
    ``a`` to ``b`` (A) passes through any of them, for
    :func:`place_substituents`; a bond that starts or ends on a ring's
    own atom is not read against that ring. ``crossed`` is the same test, for
    the count after placement."""
    box = None if box is None else np.asarray(box, float).reshape(3)
    polys = [np.asarray(p, float) for p in polys]
    shapes = [_face_shape(p) for p in polys]
    centres = np.array([p.mean(axis=0) for p in polys])
    wrapped = centres if box is None else np.clip(
        centres - box * np.floor(centres / box), 0.0, np.nextafter(box, 0.0))
    tree = cKDTree(wrapped, boxsize=box)

    def crossed(a, b) -> bool:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        mid = 0.5 * (a + b)
        q = mid if box is None else np.clip(mid - box * np.floor(mid / box), 0.0,
                                            np.nextafter(box, 0.0))
        for k in tree.query_ball_point(q, 4.0):
            poly, c = polys[k], centres[k]
            pa = a if box is None else c + _min_image(a - c, box)
            pb = pa + (b - a if box is None else _min_image(b - a, box))
            if not _may_cross(pa, pb, shapes[k]):
                continue
            if min(np.linalg.norm(poly - pa, axis=1).min(),
                   np.linalg.norm(poly - pb, axis=1).min()) < 0.05:
                continue
            if segments_through_ring(pa[None], pb[None], poly)[0]:
                return True
        return False

    return crossed, crossed


def cage_face_test(bodies, box):
    """``test(a, b)``: whether the bond from ``a`` to ``b`` (A) passes through
    a face of any of ``bodies`` or ends inside a cage's core, for
    :func:`place_substituents`; ``faces_crossed(pairs)`` counts the bonds
    (``(a, b)`` position arrays) that pass through a face."""
    box = None if box is None else np.asarray(box, float).reshape(3)
    cages = []
    for body in bodies:
        at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
        cages.append((np.asarray(body.centre, float), float(body.core),
                      [np.array([at[int(a)] for a in ring]) for ring in body.faces]))

    def crossed(a, b):
        a = np.asarray(a, float).reshape(-1, 3)
        b = np.asarray(b, float).reshape(-1, 3)
        hit = np.zeros(len(a), bool)
        for centre, core, rings in cages:
            d = a - centre
            sh = np.zeros_like(d) if box is None else -box * np.round(d / box)
            near = np.linalg.norm(d + sh, axis=1) < core + 4.0
            if not near.any():
                continue
            pa, pb = a[near] + sh[near], b[near] + sh[near]
            for ring in rings:
                hit[np.flatnonzero(near)] |= segments_through_ring(pa, pb, ring)
        return hit

    def test(a, b) -> bool:
        if crossed(a, b)[0]:
            return True
        for centre, core, _rings in cages:
            d = np.asarray(b, float) - centre
            if box is not None:
                d = d - box * np.round(d / box)
            if np.linalg.norm(d) < core:
                return True
        return False

    return test, crossed


def faces_crossed(pairs, A, B, bodies, box) -> int:
    """How many bonds pass through a face of any of ``bodies``.

    ``pairs`` are the bonds' atom ids and ``A``, ``B`` their ends (A, each
    bond in one image). Each face is read on every bond near its cage but
    one with an atom on that face, as a check of a whole data file would
    read it: a cage's own arm counts, and so does another cage's.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    hit = np.zeros(len(pairs), bool)
    for body in bodies:
        at = {int(a): np.asarray(x, float) for a, x in zip(body.atoms, body.xyz)}
        d = 0.5 * (A + B) - body.centre
        sh = np.zeros_like(d) if box is None else -box * np.round(d / box)
        near = np.flatnonzero(np.linalg.norm(d + sh, axis=1) < body.core + 4.0)
        for ring in body.faces:
            ring_ids = np.array([int(a) for a in ring])
            ok = near[~(np.isin(pairs[near, 0], ring_ids) | np.isin(pairs[near, 1], ring_ids))]
            if len(ok):
                P = np.array([at[int(a)] for a in ring])
                hit[ok] |= segments_through_ring(A[ok] + sh[ok], B[ok] + sh[ok], P)
    return int(hit.sum())


def place_substituents(neighbours: dict, coords: dict, bond_r0: dict, box,
                       rng, default_r0: float = 1.5, avoid=None, record=None) -> dict:
    """Place every atom not yet in ``coords`` from a placed neighbour.

    Breadth first from the placed atoms (sorted, so the result depends only
    on ``rng``): each placed atom gives its unplaced neighbours tetrahedral
    directions away from the bonds it already has, at the equilibrium length
    of each bond. Bond vectors are taken under the minimum image of ``box``,
    since a strand drawn across the boundary sits in the image nearest its
    first junction. Returns the new positions only.

    A ring that hangs from one atom (the phenyl of a methylphenylsiloxane)
    is placed whole when the walk reaches it, as a flat regular polygon
    pointing away from the bond it hangs from (:func:`ring_polygon`);
    its own substituents then go off it as from any other atom, a ring
    hydrogen in the ring's plane. The tree walk alone reaches every ring
    atom from one side and leaves the bond that closes the ring at whatever
    length the two branches end at: 0.40 to 5.48 A for the C-C bonds of the
    copolymer demo's phenyls, against 1.33 A. A fused ring system, or a ring
    bonded to placed atoms at two places, keeps the tree walk here; a ring a
    placed backbone runs through is closed before this, by
    :func:`close_path_rings`.

    ``avoid(a, b)`` (:func:`cage_face_test`, :func:`ring_face_test`) says
    whether a bond from ``a`` to ``b`` would pass through a cage face or
    through the face of a ring a backbone runs through: a group whose bonds would is
    drawn again, up to :data:`SUBSTITUENT_TRIES` times, and the first draw
    clear of every face is kept (or the last, when none is: an atom whose
    bonds fix its new ones, such as a backbone Si with two, has nothing else
    to draw). Without ``avoid``, or with no bond through a face, the walk
    draws exactly as before.
    """
    box = None if box is None else np.asarray(box, float)
    placed = {int(i): np.asarray(x, float) for i, x in coords.items()}
    new: dict = {}
    rings = free_rings(neighbours, placed)
    # With rings to place whole, every bond placed and every ring's face is
    # kept, so that each ring is turned clear of them and no later group is
    # drawn through it. A build with no such ring never comes here.
    pend = None
    if rings:
        pend = _PendantRings(neighbours, placed, box, avoid, bond_r0, default_r0)
        base = avoid
        avoid = (pend.crossed if base is None
                 else (lambda a_, b_: base(a_, b_) or pend.crossed(a_, b_)))
    queue = deque(sorted(placed))
    while queue:
        p = queue.popleft()
        todo = [n for n in neighbours.get(p, ()) if n not in placed]
        if not todo:
            continue
        known = []
        for n in neighbours.get(p, ()):
            if n in placed:
                d = placed[n] - placed[p]
                if box is not None:
                    d = d - box * np.round(d / box)
                known.append(d)
        dirs = tetrahedral_directions(known, len(todo), rng)
        if avoid is not None:
            for _try in range(SUBSTITUENT_TRIES):
                if not any(avoid(placed[p], placed[p] + bond_r0.get(
                        (min(p, n), max(p, n)), default_r0) * d)
                           for n, d in zip(sorted(todo), dirs)):
                    break
                dirs = tetrahedral_directions(known, len(todo), rng)
        if pend is not None:
            dirs = pend.assign(p, sorted(todo), dirs, placed, rings, default_r0)
        for n, d in zip(sorted(todo), dirs):
            r = bond_r0.get((min(p, n), max(p, n)), default_r0)
            placed[n] = placed[p] + r * d
            new[n] = placed[n]
            queue.append(n)
            if pend is not None:
                pend.placed_atom(n, placed)
            ring = rings.get(n)
            if ring is not None and _hangs_from_one_atom(ring, n, neighbours, placed):
                drawn = ring_polygon(ring, n, placed[n], d, bond_r0, rng, default_r0)
                drawn = pend.choose(ring, n, placed[n], d, drawn, placed)
                for m, x in drawn.items():
                    placed[m] = x
                    new[m] = x
                    queue.append(m)
                for m in drawn:
                    pend.placed_atom(m, placed)
    if record is not None and pend is not None and pend.rings:
        record.update({"rings": len(pend.rings), "turned": pend.turned,
                       "left_threaded": pend.left, "threads_left": pend.left_bonds,
                       "left_on_backbone": pend.deep_left, "moved": pend.swapped,
                       "_rings": pend.rings})
    return new


def draw_network(strands, shape: str, rng, parallel_strands: str = "opposite",
                 **knobs) -> tuple[list, list, dict]:
    """Every strand's backbone as drawn, end atom to end atom, and what drew it.

    The drawing half of :func:`place_network`, in the same order and from
    the same stream, so a caller that draws from a generator seeded as
    :func:`place_network`'s is drawing exactly the network that is then
    turned out of any braids and settled (the pipeline's
    :func:`coil_radius_for` readings are these). Returns ``(paths, infos,
    shared)``, ``shared`` being the strands of secondary loops the meander
    draws on their own sides (:func:`~topon.conformation.paths.shared_chords`).
    """
    if parallel_strands not in PARALLEL_STRANDS:
        raise ValueError(f"unknown parallel_strands {parallel_strands!r}; "
                         f"expected one of {', '.join(PARALLEL_STRANDS)}")
    # Bridges between the same two junctions, which the meander would draw
    # as one shape turned twice about their chord: they meet wherever the
    # wave crosses it. An entangled pair keeps the path its method drew.
    shared = (shared_chords([(s.edge[0], s.edge[1])
                             if s.cls == "bridge" and s.path is None and s.via is None
                             and s.edge[0] != s.edge[1] else None
                             for s in strands])
              if shape == "meander" and parallel_strands == "opposite" else {})
    frames: dict = {}
    paths, infos = [], []
    for k_spec, spec in enumerate(strands):
        share = (shared[k_spec] + (frames,)) if k_spec in shared else None
        interior, info = draw_backbone(spec, shape, rng, shared=share, **knobs)
        paths.append(np.vstack([np.asarray(spec.start, float), interior,
                                np.asarray(spec.end, float)]))
        infos.append(info)
    return paths, infos, shared


def place_network(strands, neighbours: dict, anchors: dict, bond_r0: dict,
                  box, shape: str, rng, clearance: Optional[float] = CLEARANCE,
                  keep_frames: bool = False, braids=None,
                  parallel_strands: str = "opposite", bodies=None,
                  **knobs) -> tuple[dict, PlacementReport]:
    """Positions for every atom: anchors, backbones, then the rest.

    ``anchors`` are the atoms whose position the graph fixes (junction and
    end-cap attachment atoms), in A. Every backbone is drawn, turned out of
    the designed pairs' ``braids`` (their sites in A; see
    :func:`clear_braids`), then settled to its bonds and angles clear of
    every other to ``clearance`` (None or 0 skips it; see
    :func:`settle_backbones`, which may run to :data:`SETTLE_ROUNDS`) with
    every ring a backbone runs through settled whole (a ring not settled is
    closed on its stretch by :func:`close_path_rings`; both recorded in
    ``report.rings``), and everything else is placed off them. On the meander the two bridges of a
    secondary loop are drawn on opposite sides of their chord
    (``parallel_strands="opposite"``, the default, as the bead-spring route
    draws them); ``"together"`` draws each as any other strand,
    which is the drawing before. ``bodies`` are cages already placed whole
    (:class:`RigidBody`, :func:`place_bodies`): every atom of them is
    placed where it is, the strands are turned out of them
    (:func:`clear_bodies`) and settled clear of them and through none of
    their faces. Returns ``(coords, report)`` with every atom reachable
    from an anchor placed.
    """
    if parallel_strands not in PARALLEL_STRANDS:
        raise ValueError(f"unknown parallel_strands {parallel_strands!r}; "
                         f"expected one of {', '.join(PARALLEL_STRANDS)}")
    report = PlacementReport(shape=shape)
    coords = {int(i): np.asarray(x, float) for i, x in anchors.items()}
    for body in bodies or ():
        for a, x in zip(body.atoms, body.xyz):
            coords[int(a)] = np.asarray(x, float)
    paths, report.strands, shared = draw_network(
        strands, shape, rng, parallel_strands=parallel_strands, **knobs)
    chains = [([spec.start_atom] + list(spec.backbone) + [spec.end_atom], path,
               np.asarray(spec.r0, float), spec.cls == "loop",
               one_three(spec.r0, spec.theta0))
              for spec, path in zip(strands, paths)]
    # the strands drawn apart, and the turns that would put each on a strand
    # it shares its chord with (rank r of m sits 2 pi r / m from the first),
    # which clear_braids must not take
    apart = {k for k in shared if report.strands[k].get("bow", 0.0) > 0.0}
    if shared:
        bows = np.array([report.strands[k]["bow"] for k in sorted(apart)], float)
        report.parallel = {
            "chords": len({v[0] for v in shared.values()}),
            "strands": len(shared),
            "drawn_apart": len(apart),
            "left_on_chord": len(shared) - len(apart),
            "bow_mean": round(float(bows.mean()), 4) if len(bows) else None,
            "bow_min": round(float(bows.min()), 4) if len(bows) else None,
            "bow_max": round(float(bows.max()), 4) if len(bows) else None}
    if braids:
        movable = [k for k, spec in enumerate(strands)
                   if spec.path is None and spec.cls != "loop"]
        avoid = {k: {BRAID_TURNS * r // shared[k][2] for r in range(1, shared[k][2])
                     if (BRAID_TURNS * r) % shared[k][2] == 0}
                 for k in apart}
        report.braids = clear_braids(chains, movable, list(braids), box, avoid=avoid)
    if bodies:
        # (a strand drawn round a cage already keeps off it; turned about a
        # chord that runs through the cage, it could only go back in)
        movable = [k for k, spec in enumerate(strands)
                   if spec.path is None and spec.cls != "loop" and spec.via is None]
        avoid = {k: {BRAID_TURNS * r // shared[k][2] for r in range(1, shared[k][2])
                     if (BRAID_TURNS * r) % shared[k][2] == 0}
                 for k in apart}
        report.cages = clear_bodies(chains, movable, list(bodies), box, avoid=avoid,
                                    braids=list(braids) if braids else None)
    # a strand that starts on a cage's stub is settled from the cage's
    # attachment atom, which with the stub atom is held: the angle at the
    # stub atom is then one of its 1-3 targets
    held_ids = set()
    leads = [(spec.lead_start is not None, spec.lead_end is not None) for spec in strands]
    if any(a or b for a, b in leads):
        for k, (spec, (ls, le)) in enumerate(zip(strands, leads)):
            if not (ls or le):
                continue
            ids, pts = list(chains[k][0]), np.asarray(chains[k][1], float)
            r0, th = list(spec.r0), list(spec.theta0)
            if ls:
                a, x, r, t = spec.lead_start
                ids, pts = [int(a)] + ids, np.vstack([np.asarray(x, float), pts])
                r0, th = [float(r)] + r0, [float(t)] + th
                held_ids.add(int(spec.start_atom))
            if le:
                a, x, r, t = spec.lead_end
                ids, pts = ids + [int(a)], np.vstack([pts, np.asarray(x, float)])
                r0, th = r0 + [float(r)], th + [float(t)]
                held_ids.add(int(spec.end_atom))
            chains[k] = (ids, pts, np.asarray(r0, float), chains[k][3], one_three(r0, th))
    # The rings a backbone runs through (a para-phenylene between Si and O),
    # taken from the molecule with the anchors and cages out: every network
    # cycle runs through a junction, an end cap or a cage, so what is left
    # are the monomers' own rings. The settle takes each of them whole
    # (ring_windows).
    rings = free_rings(neighbours, {int(a) for a in anchors}
                       | {int(a) for body in bodies or () for a in body.atoms})
    side_atoms = None
    for k in range(len(strands)) if rings else ():
        ids = chains[k][0]
        if any(a in rings for a in ids):
            one3, held_rings = ring_windows(ids, list(chains[k][2]), rings, bond_r0)
            if not held_rings:
                continue           # a ring atom on the path, no stretch through it
            t_eq = np.array(chains[k][4], float)
            for row, dist in one3.items():
                t_eq[row] = dist
            chains[k] = tuple(chains[k][:4]) + (t_eq, held_rings)
    if bodies:
        # how many side atoms each backbone atom has, for the settle to read
        # where they will go near a cage: its neighbours off every
        # path, every cage and every ring stretch the settle holds
        taken = {int(a) for c in chains for a in c[0]}
        taken |= {int(a) for c in chains if len(c) > 5 for ring in c[5]
                  for a in ring["extra"]}
        taken |= {int(a) for body in bodies for a in body.atoms}
        side_atoms = {int(a): sum(1 for n in neighbours.get(int(a), ()) if int(n) not in taken)
                      for c in chains for a in c[0]}
    if clearance and box is not None and chains:
        slow = {"sweeps": RING_SWEEPS, "rounds": RING_SETTLE_ROUNDS} if any(
            len(c) > 5 for c in chains) else {}
        drawn, report.settle, report.frames = settle_backbones(
            chains, box, float(clearance), rng, keep_frames=keep_frames,
            bodies=list(bodies) if bodies else None, held_ids=held_ids or None,
            side_atoms=side_atoms, **slow)
        chains = [(c[0], p) + tuple(c[2:]) for c, p in zip(chains, drawn)]
    for spec, (ls, le), (_ids, pts, *_rest) in zip(strands, leads, chains):
        pts = pts[int(ls):len(pts) - int(le)]
        for idx, xyz in zip(spec.backbone, pts[1:-1]):
            coords[int(idx)] = np.asarray(xyz, float)
    if rings and any(a in rings for c in chains for a in c[0]):
        # the off-path ring atoms where the settle left them; a ring it did
        # not hold (no settle) is closed on its stretch as drawn
        settled = (report.settle or {}).pop("_ring_atoms", {})
        coords.update({int(a): np.asarray(x, float) for a, x in settled.items()})
        closed, done = close_path_rings(rings, coords, neighbours, bond_r0, box)
        report.rings = {"stretches": int(sum(len(ring_runs(c[0], rings))
                                             for c in chains[:len(strands)])),
                        "held_flat": bool(settled), "settled_atoms": len(settled),
                        **done}
        coords.update(closed)
    # The faces of the rings the backbones run through, as placed: no
    # hydrogen or methyl is drawn through one. A build with no such
    # ring has none, and places its substituents as before.
    path_atoms = {int(a) for c in chains for a in c[0]}
    faces_r = sorted(ring for ring in {tuple(r) for r in rings.values()}
                     if any(a in path_atoms for a in ring) and all(a in coords for a in ring))
    ring_test = None
    if faces_r:
        bx = None if box is None else np.asarray(box, float).reshape(3)
        ring_test, ring_crossed = ring_face_test(
            [np.array([coords[ring[0]] if bx is None else
                       coords[ring[0]] + _min_image(coords[a] - coords[ring[0]], bx)
                       for a in ring]) for ring in faces_r], box)
    pendant: dict = {}
    if bodies:
        test, _crossed = cage_face_test(bodies, box)
        avoid = test if ring_test is None else (lambda a, b: test(a, b) or ring_test(a, b))
        coords.update(place_substituents(neighbours, coords, bond_r0, box, rng,
                                         avoid=avoid, record=pendant))
        # Every bond of the build read against every face of every cage, as
        # check_poss_cages reads the written file: all but the bonds with an
        # atom on that face (a bare cage's corner, a strand bonds to).
        pairs = np.array([(a, b) for a, nbs in neighbours.items() for b in nbs
                          if a < b and a in coords and b in coords], int).reshape(-1, 2)
        A = np.array([coords[int(a)] for a in pairs[:, 0]]).reshape(-1, 3)
        B = np.array([coords[int(b)] for b in pairs[:, 1]]).reshape(-1, 3)
        if box is not None and len(A):
            bx = np.asarray(box, float).reshape(3)
            B = A + (B - A) - bx * np.round((B - A) / bx)
        report.cages["bonds_through_faces"] = faces_crossed(pairs, A, B, bodies, box)
        # and every bond against its r0, as placed
        r0s = np.array([bond_r0.get((int(min(a, b)), int(max(a, b))), np.nan)
                        for a, b in pairs])
        ratio = np.linalg.norm(B - A, axis=1) / r0s
        ratio = ratio[np.isfinite(ratio)]
        report.cages["bond_ratio_all"] = ({"min": round(float(ratio.min()), 4),
                                           "max": round(float(ratio.max()), 4)}
                                          if len(ratio) else None)
    else:
        coords.update(place_substituents(neighbours, coords, bond_r0, box, rng,
                                         avoid=ring_test, record=pendant))
    if ring_test is not None:
        # every bond of the build read against those faces, but a ring's own
        pairs = [(a, b) for a, nbs in neighbours.items() for b in nbs
                 if a < b and a in coords and b in coords]
        through = {"backbone": 0, "other": 0}
        for a, b in pairs:
            if ring_crossed(coords[a], coords[b]):
                through["backbone" if a in path_atoms and b in path_atoms else "other"] += 1
        report.rings["bonds_through"] = through
    if pendant:
        # every bond of the build read against the pendant rings' faces as
        # placed, but a ring's own and those bonded to it
        bx = None if box is None else np.asarray(box, float).reshape(3)
        polys = [np.array([coords[ring[0]] if bx is None else
                           coords[ring[0]] + _min_image(coords[a] - coords[ring[0]], bx)
                           for a in ring]) for ring in pendant.pop("_rings")]
        _t, crossed_p = ring_face_test(polys, box)
        on_path = {int(a) for c in chains for a in c[0]}
        through_p = {"backbone": 0, "other": 0}
        for a, nbs in neighbours.items():
            for b in nbs:
                if a < b and a in coords and b in coords and crossed_p(coords[a], coords[b]):
                    through_p["backbone" if a in on_path and b in on_path else "other"] += 1
        report.pendant_rings = {**pendant, "bonds_through": through_p}
    return coords, report
