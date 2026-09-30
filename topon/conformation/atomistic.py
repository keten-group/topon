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
or ``walk``. The routines work in units of the bead-spring design bond
(0.97), so the path is drawn in units where the strand's mean backbone bond
is 0.97 and scaled back; the shape knobs (``meander_waves``, ``min_bond``,
``min_self_separation``, ``path_jitter``) then mean the same shape on both
routes. Every backbone bond is drawn at the strand's mean equilibrium
length, which is exact for PDMS (all Si-O) and within 4 % of each bond for
PEG (C-O against C-C); spacing points along the arc by each bond's own
length would cut the corners of a zigzag or a walk. Every other atom
(methyls, hydrogens, end-cap methyls, junction caps) is placed from its
already-placed neighbour at its bond length, in tetrahedral directions away
from the bonds that atom already has.

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
from scipy.spatial import cKDTree

from .paths import (bond_lengths, bridging_walk, closed_meander, meander_chain,
                    self_contact, straight_chain, zigzag)
from .segments import passing_pairs, segment_closest

#: The bead-spring design bond. Paths are drawn in units where the backbone
#: bond is this long, so every shape knob keeps its bead-spring meaning.
BEAD_BOND = 0.97

#: Fraction of the extended length at which a strand is drawn straight: the
#: bead-spring ``straight_at``, read against the longest the backbone can be
#: at its equilibrium angles rather than against its contour.
TAUT = 0.97

SHAPES = ("straight", "meander", "walk")

#: The closest two backbone bonds may be drawn, in A, unless they share an
#: atom or sit within two bonds of each other along one strand.
CLEARANCE = 1.5

#: How far past the clearance a short pair is pushed, as a fraction of it.
OVERSHOOT = 0.1

#: Two bonds closer than this (A) are touching: they have no side to keep.
TOUCH = 0.01

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


@dataclass
class PlacementReport:
    """What the placement did, per strand and in total."""

    shape: str
    strands: list = field(default_factory=list)
    settle: Optional[dict] = None
    #: Backbone positions before and after every round of the settling
    #: pass, ``(ids, [xyz, ...])``, for a caller to check that no bond went
    #: through another on the way (:mod:`topon.analysis.crossings`).
    frames: Optional[tuple] = None

    def summary(self) -> dict:
        rows = self.strands
        ratio = np.array([r["chord_over_extended"] for r in rows
                          if r.get("chord_over_extended") is not None])
        routines = {}
        for r in rows:
            routines[r["routine"]] = routines.get(r["routine"], 0) + 1
        return {
            "shape": self.shape,
            "strands": len(rows),
            "routines": routines,
            "taut": int(sum(1 for r in rows if r.get("taut"))),
            "over_contour": int(sum(1 for r in rows if r.get("over_contour"))),
            "chord_over_extended": (
                {"mean": float(ratio.mean()), "max": float(ratio.max())}
                if len(ratio) else None),
            "backbone_bond_ratio": _ratio_stats(rows),
            "settle": self.settle,
        }


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
                  jitter: float = 0.02) -> tuple[np.ndarray, dict]:
    """The backbone atoms' positions, in order, and what drew them.

    Returns the interior points only (``len(spec.backbone)`` of them): the
    two end atoms are junctions and stay where the graph put them.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown atomistic placement {shape!r} "
                         f"(expected one of {', '.join(SHAPES)})")
    r0 = np.asarray(spec.r0, float)
    n = len(r0)
    contour = float(r0.sum())
    k = BEAD_BOND / float(r0.mean())          # A -> bead units
    a = np.asarray(spec.start, float)
    b = np.asarray(spec.end, float)
    info = {"edge": list(spec.edge), "cls": spec.cls, "n_bonds": n,
            "contour": contour}

    if spec.path is not None:
        # An entangled pair keeps the path its method drew for it.
        interior = np.asarray(spec.path, float)
        full = np.vstack([a, interior, b])
        info["routine"] = "entangled"
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
            full, drawn = meander_chain(
                a * k, b * k, n, bond=BEAD_BOND, rng=rng, waves=waves,
                min_bond=min_bond, min_sep=1.0 if min_sep is None else min_sep,
                jitter=jitter)
            full = full / k
            use = drawn["routine"]
        else:
            full = bridging_walk(a * k, b * k, n, BEAD_BOND, rng) / k
        info["routine"] = use

    ratio = bond_lengths(full) / r0
    info["bond_ratio_min"] = float(ratio.min())
    info["bond_ratio_max"] = float(ratio.max())
    info["self_contact"] = float(self_contact(full, closed=spec.cls == "loop"))
    return full[1:-1], info


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
                     rounds: int = 400, step: float = 0.2, sweeps: int = 8,
                     width: float = 1.0, ramp: int = 40, tol: float = 0.02,
                     keep_frames: bool = False):
    """Bring every backbone to its bonds and angles, clear of every other, crossing nothing.

    ``chains`` is one ``(ids, points, r0, closed, one_three)`` per strand:
    its backbone atom ids from end atom to end atom, their positions in the
    strand's own image (A), the equilibrium length of each bond, whether it
    is a loop (both ends the same junction), and the equilibrium distance
    between each atom's two backbone neighbours (from the bond angle at it;
    None keeps the distances the path was drawn with). The two end atoms of
    every strand are junctions or end caps and are held.

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
    them and open along the normal to both, the side drawn at random; that
    is the only choice the pass makes, and a drawn path that touches has
    made no choice to keep. Rounds stop when no pair is short and every bond
    and 1-3 distance is within ``tol`` of its target.

    Returns ``(points, report, frames)``: the new positions per chain,
    what was found and left, and (with ``keep_frames``) ``(ids, [xyz, ...])``
    with the position of every backbone atom before the first round and
    after each, one row per atom id.
    """
    rng = np.random.default_rng() if rng is None else rng
    box = np.asarray(box, float).reshape(3)
    sizes = [len(c[0]) for c in chains]
    X = np.vstack([np.asarray(c[1], float).reshape(-1, 3) for c in chains])
    ids = np.concatenate([np.asarray(c[0], int) for c in chains])
    starts = np.cumsum([0] + sizes[:-1])
    held = np.zeros(len(X), bool)
    held[starts] = True
    held[starts + np.asarray(sizes) - 1] = True
    # Every strand in one image, bond by bond from its first atom. A designed
    # pair's braid comes back in the image nearest its partner's chord, so
    # without this a bond of it can span the box (47 r0 on the DP-30 build
    # with six pairs), and every pair search reaches across the whole cell.
    for s0, m in zip(starts, sizes):
        step_ = np.diff(X[s0:s0 + m], axis=0)
        step_ -= box * np.round(step_ / box)
        X[s0 + 1:s0 + m] = X[s0] + np.cumsum(step_, axis=0)

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

    def survey(P):
        p1, p2 = P[sa], P[sb]
        mid = 0.5 * (p1 + p2)
        w = mid - box * np.floor(mid / box)
        w = np.clip(w, 0.0, np.nextafter(box, 0.0))
        pairs = cKDTree(w, boxsize=box).query_pairs(
            clearance + 2.0 * half_bond, output_type="ndarray")
        if not len(pairs):
            return (np.zeros(0, int),) * 2 + (np.zeros((0, 3)),) + (np.zeros(0),) * 3
        i, j = pairs[:, 0], pairs[:, 1]
        share = (ida[i] == ida[j]) | (ida[i] == idb[j]) | (idb[i] == ida[j]) | (idb[i] == idb[j])
        gap = np.abs(along[i] - along[j])
        gap = np.where(closed[i], np.minimum(gap, nseg[i] - gap), gap)
        keep = ~share & ~((chain[i] == chain[j]) & (gap <= 2))
        i, j = i[keep], j[keep]
        sh = -box * np.round((mid[j] - mid[i]) / box)
        s, t, d = segment_closest(p1[i], p2[i], p1[j] + sh, p2[j] + sh)
        short = d < clearance
        return i[short], j[short], sh[short], s[short], t[short], d[short]

    def relax(P, r_want, t_want):
        for _ in range(int(sweeps)):
            corr = np.zeros_like(P)
            count = np.zeros(len(P))
            for a, b, want in ((sa, sb, r_want), (ta, tb, t_want)):
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
            P = P + corr / np.maximum(count, 1.0)[:, None]
        return P

    def errors(P):
        rb_ = np.linalg.norm(P[sb] - P[sa], axis=1) / r0 - 1.0
        rt_ = np.linalg.norm(P[tb] - P[ta], axis=1) / np.maximum(t_eq, 1e-9) - 1.0
        return (float(np.abs(rb_).max()) if len(rb_) else 0.0,
                float(np.abs(rt_).max()) if len(rt_) else 0.0)

    def push(P, found, target):
        """Moves opening every pair in ``found`` to ``target`` (an array)."""
        i, j, sh, s, t, d = found
        move = np.zeros_like(P)
        if not len(i):
            return move
        p1, p2 = P[sa[i]], P[sb[i]]
        q1, q2 = P[sa[j]] + sh, P[sb[j]] + sh
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

    before = survey(X)
    n_before = len(before[0])
    worst_before = float(before[5].min()) if n_before else None
    angle_before = np.abs(_angles(X, ta, tb) - theta_eq) if len(ta) else np.zeros(0)
    # Two bonds drawn touching have no side, and every way of parting them
    # reads as one passing through the other. They are parted first, by a
    # hair and on a side drawn at random, and the record starts after that:
    # the only choice the pass makes.
    touching = before[5] < TOUCH
    if touching.any():
        found = tuple(x[touching] for x in before)
        X = X + push(X, found, np.full(int(touching.sum()), 5.0 * TOUCH))
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
        found = survey(X)
        e_bond, e_13 = errors(X)
        if f >= 1.0 and not len(found[0]) and e_bond < tol and e_13 < tol:
            done -= 1
            break
        # a little past the clearance, since the bonds take some of every
        # push back and a pair aimed exactly at it only creeps up to it
        move = push(X, found, np.full(len(found[0]), (1.0 + OVERSHOOT) * clearance))
        new = relax(X + move, r_want, t_want)
        shift = new - X
        size = np.linalg.norm(shift, axis=1)
        new = X + shift * np.minimum(1.0, step / np.maximum(size, 1e-12))[:, None]
        # An atom pushed by two pairs at once, or pulled by its bonds and
        # angles, moves along no pair's line and can take a bond through a
        # third one close by. Whatever does is put back where it was.
        for _ in range(50):
            pi, pj, _tau, _big = passing_pairs(X, new, rows, pair_ids, box)
            if not len(pi):
                break
            back = np.unique(np.concatenate([sa[pi], sb[pi], sa[pj], sb[pj]]))
            new[back] = X[back]
            held_back += len(back)
        else:
            new = X
        X = new
        if keep_frames:
            frames.append(X[rows_out].copy())

    after = survey(X)
    n_after = len(after[0])
    e_bond, e_13 = errors(X)
    angle_after = np.abs(_angles(X, ta, tb) - theta_eq) if len(ta) else np.zeros(0)
    ratio = np.linalg.norm(X[sb] - X[sa], axis=1) / r0

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
        "converged": bool(not n_after and e_bond < tol and e_13 < tol),
        "rounds": int(done),
        "largest_shift": float(np.linalg.norm(X - X0, axis=1).max()) if len(X) else 0.0,
        "moves_put_back": int(held_back),
        "touching_parted": int(touching.sum()),
    }
    points = [X[s0:s0 + m] for s0, m in zip(starts, sizes)]
    out_frames = (ids[rows_out], frames) if keep_frames else None
    return points, report, out_frames


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


def place_substituents(neighbours: dict, coords: dict, bond_r0: dict, box,
                       rng, default_r0: float = 1.5) -> dict:
    """Place every atom not yet in ``coords`` from a placed neighbour.

    Breadth first from the placed atoms (sorted, so the result depends only
    on ``rng``): each placed atom gives its unplaced neighbours tetrahedral
    directions away from the bonds it already has, at the equilibrium length
    of each bond. Bond vectors are taken under the minimum image of ``box``,
    since a strand drawn across the boundary sits in the image nearest its
    first junction. Returns the new positions only.
    """
    box = None if box is None else np.asarray(box, float)
    placed = {int(i): np.asarray(x, float) for i, x in coords.items()}
    new: dict = {}
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
        for n, d in zip(sorted(todo), tetrahedral_directions(known, len(todo), rng)):
            r = bond_r0.get((min(p, n), max(p, n)), default_r0)
            placed[n] = placed[p] + r * d
            new[n] = placed[n]
            queue.append(n)
    return new


def place_network(strands, neighbours: dict, anchors: dict, bond_r0: dict,
                  box, shape: str, rng, clearance: Optional[float] = CLEARANCE,
                  keep_frames: bool = False, **knobs) -> tuple[dict, PlacementReport]:
    """Positions for every atom: anchors, backbones, then the rest.

    ``anchors`` are the atoms whose position the graph fixes (junction and
    end-cap attachment atoms), in A. Every backbone is drawn, then settled
    to its bonds and angles clear of every other to ``clearance`` (None or 0
    skips it; see :func:`settle_backbones`), then everything else is placed
    off them. Returns ``(coords, report)`` with every atom reachable from an
    anchor placed.
    """
    report = PlacementReport(shape=shape)
    coords = {int(i): np.asarray(x, float) for i, x in anchors.items()}
    chains = []
    for spec in strands:
        interior, info = draw_backbone(spec, shape, rng, **knobs)
        ends = [np.asarray(spec.start, float), np.asarray(spec.end, float)]
        chains.append(([spec.start_atom] + list(spec.backbone) + [spec.end_atom],
                       np.vstack([ends[0], interior, ends[1]]),
                       np.asarray(spec.r0, float), spec.cls == "loop",
                       one_three(spec.r0, spec.theta0)))
        report.strands.append(info)
    if clearance and box is not None and chains:
        drawn, report.settle, report.frames = settle_backbones(
            chains, box, float(clearance), rng, keep_frames=keep_frames)
        chains = [(c[0], p) + tuple(c[2:]) for c, p in zip(chains, drawn)]
    for spec, (_ids, pts, *_rest) in zip(strands, chains):
        for idx, xyz in zip(spec.backbone, pts[1:-1]):
            coords[int(idx)] = np.asarray(xyz, float)
    coords.update(place_substituents(neighbours, coords, bond_r0, box, rng))
    return coords, report
