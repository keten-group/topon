"""Named and shell-selected pairs, wound on top of chains that are already placed.

The placement stage decides the entanglement state of the network as a
*statistic*: a shape and a build density, and Z comes out where the calibration
says it will. This module is the other kind of request -- *these two strands,
this many windings* -- and it works on the strands as placed rather than
replacing them, so the statistical state the build was tuned for survives the
designed ones being added.

What this delivers, and where
-----------------------------
A named pair comes back carrying the winding it was asked for, verified on the
paths as drawn rather than assumed from the plan. Measured on SC 6x6x6 at
DP 60, one request per strand: **8 of 8 delivered at coil 2.84 with linking
numbers 1.02 to 1.03**, and 4 of 8 at coil 3.85. The acceptance test is
half the requested windings at DP 60; both densities meet it and the lower one
saturates it.

The control that makes that claim mean anything: the same machinery with no
braid asked for gives a linking number of 0.00 on the same strands. The winding
comes from the braid, not from the meander that would have been there anyway.

Three limits, all of them refusals rather than surprises:

* **coil ratio.** Above about 3.5 the free run's own wave is tight enough to
  take the braid apart, and the pair is drawn but does not wind. Swept at
  DP 40, 60 and 120: delivered from coil 1.5 to 3.5, refused above.
* **one braid per stretch of chord.** Two blending into each other collapse the
  winding of both. Asked four partners against one strand on SC 6x6x6, every
  pair read 0.00 to 0.01; the same strands asked one at a time read 1.02. The
  second request on a stretch is refused;
  :func:`~topon.conformation.entanglement.allocation.allocate_contacts` is the
  thing that packs several along one chain properly.
* **the budgets below**, which are checked before anything is drawn.

What it can refuse, and why that is the point
---------------------------------------------
A winding is paid for in contour. The two partners have to leave their chords,
go round each other and come back, and a strand only has ``n_bonds * bond`` of
contour to spend, and at DP 20 on the reference geometry there is not enough.

The number to quote is the one measured for the pair in hand, not a constant.
An earlier estimate put designed windings out of reach below roughly DP 45,
and that figure is real but belongs to a different cell: the pilot's SC 6x6x6
at DP-30 site spacing with a ring radius of 2.0, where one ring costs
``2 pi r = 12.57`` sigma and two exceed a DP-20 contour of 19.95. On the N20
geometry with the default :class:`BraidShape` (``n_radius`` 0.9) a single
winding measures out at a minimum of DP 22 to 30 depending on the partner --
consistent with the pilot once the ring radius is accounted for, and not the
same number. So the budget is computed from the actual chord, detour and ring
radius of each pair before anything is drawn, and a request that does not fit
is refused by name with the smallest DP that would carry *it*. Quietly delivering a compressed braid instead
is the failure mode this replaces: it costs clearance, and a braid whose
partners come within a fraction of sigma is one the push-off resolves by
pushing them through each other.

Two budgets bind, and they are not the same:

* **contour** -- how long the routed path is against what the beads carry. DP
  fixes this directly, so the refusal can name a minimum DP.
* **axial room** -- how much of the shared axis the braid needs
  (``e * pitch + 2 * ramp``) against what the two chords can spare. This is
  set by the geometry, not by DP, so when it binds the message says to move
  the build density or pick a closer partner instead.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np

from topon.conformation.entanglement.allocation import (
    _interval_on,
    _overlaps,
)
from topon.conformation.entanglement.braid import (
    BraidShape,
    axial_room,
    braid_path,
    closest_approach,
    far_closed_linking,
    feasible_window,
    gap_at,
    make_contact,
    min_separation,
    plan_braid,
)

__all__ = [
    "PairRequest",
    "PairOutcome",
    "DesignReport",
    "chords_of",
    "nearest_image_of",
    "requests_from_config",
    "contour_cost",
    "route_designed_pairs",
]

#: Points used to measure a routed path's arc length. The braid is a smooth
#: curve and a 22-bead sample of it under-measures its own length by the chord
#: error of every bead, which is exactly the error the budget must not make.
_DENSE = 600

#: Least gap between two braids on one chord, as a fraction of it. The same
#: default ``allocate_contacts`` uses, and for the same reason: adjacent
#: blends that touch fight each other.
_BRAID_SEPARATION = 0.02


@dataclass(frozen=True)
class PairRequest:
    """Wind ``chain_a`` and ``chain_b`` around each other ``windings`` times.

    Chain ids index :func:`topon.conformation.strand_plans`, which is the
    graph's edge order and the order :class:`~topon.conformation.Placement`
    keeps its strands in. A request may also name the edge key ``(u, v, k)``,
    and **that is the durable form**: the topology stage guarantees a stable
    edge order for a fixed seed (verified across the strict and exact searches,
    ``test_the_edges_keep_the_scaffold_ordering``), but the exact search draws
    from ``numpy.random.default_rng``, whose stream NumPy does not promise
    across feature releases. A NumPy upgrade can therefore change which edges
    survive on an exact-search graph at the same seed, and an index that meant
    strand 57 last month can mean a different one after it. A junction pair
    does not move. Both forms are resolved by :func:`requests_from_config`.
    """

    chain_a: int
    chain_b: int
    windings: int = 1


@dataclass
class PairOutcome:
    """What became of one request."""

    request: PairRequest
    granted: int = 0
    reason: str = ""
    minimum_dp: Optional[int] = None
    contour_needed: float = 0.0
    contour_have: float = 0.0
    axial_needed: float = 0.0
    axial_have: float = 0.0
    gap: float = 0.0
    clearance: Optional[float] = None
    linking: Optional[float] = None
    """Winding measured on the two paths as drawn, by the Gauss integral over
    chord-closed loops. ``None`` until the pair has been built."""

    @property
    def refused(self) -> bool:
        return self.granted < 1

    def message(self) -> str:
        r = self.request
        head = (f"pair ({r.chain_a}, {r.chain_b}) x{r.windings}: ")
        if self.granted >= r.windings:
            return head + "delivered in full"
        if self.granted >= 1:
            return head + f"granted {self.granted} of {r.windings} ({self.reason})"
        out = head + f"refused, {self.reason}"
        if self.minimum_dp is not None:
            out += (f". The routed path is {self.contour_needed:.1f} sigma and "
                    f"these strands carry {self.contour_have:.1f}; DP "
                    f"{self.minimum_dp} is the smallest that would carry it "
                    f"at this geometry -- rebuild at that DP and the same "
                    f"build *box*, since holding the density instead grows "
                    f"the chords with the bead count and this DP becomes a "
                    f"lower bound")
        return out


@dataclass
class DesignReport:
    """Every request, and the paths that came back changed."""

    outcomes: list[PairOutcome] = field(default_factory=list)
    paths: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def accepted(self) -> list[PairOutcome]:
        return [o for o in self.outcomes if not o.refused]

    @property
    def refused(self) -> list[PairOutcome]:
        return [o for o in self.outcomes if o.refused]

    def summary(self) -> dict:
        req = sum(o.request.windings for o in self.outcomes)
        got = sum(o.granted for o in self.outcomes)
        return {
            "requested_pairs": len(self.outcomes),
            "delivered_pairs": len(self.accepted),
            "requested_windings": req,
            "granted_windings": got,
            "fraction_delivered": (got / req) if req else None,
            "refusals": [o.message() for o in self.refused],
            "partial": [o.message() for o in self.accepted
                        if o.granted < o.request.windings],
        }


# ---------------------------------------------------------------------------
# Geometry in, budget out
# ---------------------------------------------------------------------------

def chords_of(placement, indices: Optional[Iterable[int]] = None) -> dict:
    """``{strand index: (start, end)}`` for the strands that have a chord.

    A primary loop has none -- both of its ends are the same junction -- so it
    is left out, and a request naming one is refused on that ground rather than
    crashing on a degenerate chord.

    The chords come back in each strand's own image, which is where its beads
    are. Bringing a pair into one image is :func:`nearest_image_of`'s job and
    is done per pair, because there is no single image that suits every pair
    at once.
    """
    out = {}
    wanted = None if indices is None else set(int(i) for i in indices)
    for i, s in enumerate(placement.strands):
        if s.plan.kind == "loop":
            continue
        if wanted is not None and i not in wanted:
            continue
        out[i] = (np.asarray(s.path[0], float), np.asarray(s.path[-1], float))
    return out


def nearest_image_of(chord_b, chord_a, box):
    """Chord B moved to the periodic image nearest chord A, plus the shift.

    Two strands that are neighbours *across* the boundary sit a box apart in
    the coordinates each was drawn in, and a braid planned on those raw chords
    reaches across the whole system: measured on the N20 graph, the nearest
    disjoint partner of strand 0 came out 163 sigma away in a 124 sigma box,
    and the request was refused for a contour it never actually needed. This
    is the same image convention the kink and the waypoint pair have always
    used (:func:`~topon.conformation.entanglement.realize.
    entangled_backbone_paths`): take the partner in the image nearest this
    edge's own midpoint.

    Returns ``((b0, b1), delta)``; ``delta`` is what was added to B, so
    subtracting it from a contact built in A's image puts that contact back in
    B's.
    """
    a0, a1 = (np.asarray(v, float) for v in chord_a)
    b0, b1 = (np.asarray(v, float) for v in chord_b)
    if box is None:
        return (b0, b1), np.zeros(3)
    L = np.asarray(box, float).reshape(3)
    mid_a = 0.5 * (a0 + a1)
    mid_b = 0.5 * (b0 + b1)
    delta = -L * np.round((mid_b - mid_a) / L)
    return (b0 + delta, b1 + delta), delta


def requests_from_config(pairs, placement=None) -> list[PairRequest]:
    """Turn the config's ``[[a, b, windings], ...]`` into requests.

    ``pairs`` entries may name strand indices or edge keys; an edge key is
    resolved against ``placement``'s strand order when one is given.
    """
    index_of = {}
    if placement is not None:
        for i, s in enumerate(placement.strands):
            index_of[tuple(s.plan.key)] = i
            index_of[(s.plan.key[0], s.plan.key[1])] = i

    def resolve(x):
        if isinstance(x, (list, tuple)):
            key = tuple(x)
            if key in index_of:
                return index_of[key]
            raise KeyError(f"no strand with edge key {key}")
        return int(x)

    out = []
    for row in pairs:
        a, b, e = row[0], row[1], int(row[2])
        out.append(PairRequest(resolve(a), resolve(b), e))
    return out


def _path_length(p) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(p, float), axis=0),
                                axis=1).sum())


def _room_at(a0, a1, b0, b1, s_a: float, windings: int, shape: BraidShape):
    """Contact, fitted shape and axial room for one position along A."""
    _gap, s_b = gap_at(a0, a1, b0, b1, s_a)
    contact = make_contact(a0, a1, b0, b1, s_a=s_a, s_b=s_b)
    fitted = shape.fit_to_gap(contact.gap)
    la, ha = axial_room(a0, a1, contact)
    lb, hb = axial_room(b0, b1, contact)
    have = min(ha, hb, -la, -lb)                 # half-span the pair can spare
    half, e_max = plan_braid(a0, a1, b0, b1, contact, windings, fitted)
    return contact, fitted, float(have), float(half), int(e_max)


def contour_cost(a0, a1, b0, b1, windings: int,
                 shape: Optional[BraidShape] = None,
                 min_clearance: float = 1.0, samples: int = 25,
                 window_tolerance: float = 0.35) -> dict:
    """What a braid of ``windings`` turns would cost this pair.

    Returns the contour each partner's routed path would need, the axial room
    the braid wants against the room the chords can spare, the turns the room
    actually supports, and the clearance the built pair would have. Nothing is
    assumed about DP here: the answer is in sigma, and the caller compares it
    with whatever contour its beads carry.

    The braid is not sited at the bare closest approach. On a lattice that
    answer is routinely degenerate -- two strands sharing a junction come
    closest *at* the junction, where :func:`closest_approach` clamps, and a
    contact pinned to the end of a chord has no axial room on one side, so
    every such pair would read as impossible. The position is chosen the way
    :func:`~topon.conformation.entanglement.allocation.allocate_contacts`
    chooses it: slide along the stretch where the pair stays close
    (:func:`feasible_window`) and keep the position with the most room.

    The cost is *measured*, not estimated: the braid is built at 600 points and
    its arc length taken, because the closed form for an elliptical helix
    ignores the ramps, and the ramps are half the span of a one-turn braid.
    """
    shape = shape or BraidShape()
    a0, a1, b0, b1 = (np.asarray(v, float) for v in (a0, a1, b0, b1))
    s_star, _ = closest_approach(a0, a1, b0, b1)
    lo, hi = feasible_window(a0, a1, b0, b1, tolerance=window_tolerance)
    tried = sorted({float(s) for s in
                    np.concatenate([np.linspace(lo, hi, max(int(samples), 2)),
                                    [s_star, 0.5]])})

    best = None
    for s_a in tried:
        cand = _room_at(a0, a1, b0, b1, s_a, windings, shape)
        key = (cand[4], cand[2])                 # turns first, then room
        if best is None or key > best[0]:
            best = (key, cand)
    contact, fitted, have, half, e_max = best[1]

    pa = braid_path(a0, a1, contact, windings, -1, _DENSE, half, fitted)
    pb = braid_path(b0, b1, contact, windings, +1, _DENSE, half, fitted)
    clearance = float(min_separation(pa, pb))

    return {
        "gap": float(contact.gap),
        "s_a": float(contact.s_a),
        "s_b": float(contact.s_b),
        "contour_a": _path_length(pa),
        "contour_b": _path_length(pb),
        "chord_a": float(np.linalg.norm(a1 - a0)),
        "chord_b": float(np.linalg.norm(b1 - b0)),
        "axial_needed": float(fitted.span(windings)),
        "axial_have": float(2.0 * max(have, 0.0)),
        "turns_supported": int(e_max),
        "clearance": clearance,
        "clears": clearance >= min_clearance,
        "contact": contact,
        "shape": fitted,
        "half_span": float(half),
    }


def _minimum_dp(cost: dict, n_bonds_per_dp: int, bond: float) -> int:
    """Smallest DP whose contour covers the longer of the two routed paths.

    ``n_bonds_per_dp`` is 1 for a bridge (DP beads, DP + 1 bonds) and 0 for a
    dangling strand (DP beads, DP bonds); it is the offset between the DP and
    the bond count, not a multiplier.

    The answer holds *at this geometry*, which is a real qualification and not
    a hedge. Rebuilding at a higher DP and the same build density puts more
    beads in the box, so the box grows as ``DP^(1/3)`` and every chord with
    it, and the route the braid has to take gets longer too. Measured on the
    N20 graph, a pair that needed DP 24 at DP 20's box needed DP 25 once
    rebuilt at DP 24. The contour grows as ``DP`` and the chord as
    ``DP^(1/3)``, so raising DP does converge -- just a step at a time near
    the boundary. Rebuilding at the same *box* instead (raise the density with
    the DP) reaches it in one.
    """
    need = max(cost["contour_a"], cost["contour_b"])
    bonds = math.ceil(need / bond)
    return int(max(1, bonds - n_bonds_per_dp))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

#: Multipliers tried on each braid's ``n_radius``, in order: the fitted radius
#: first, because it is the one the budget priced, then wider, because a wider
#: turn is one the bond length can follow.
_RADIUS_SCALES = (1.0, 1.2, 1.5, 1.8, 0.8)


def _braided_path(chord_pair, entries, n_out: int, bond: float,
                  min_bond: float = 0.85, min_sep: float = 1.0,
                  waves: float = 6.0, min_waves: float = 0.5,
                  is_loop: bool = False, dense_per_bead: int = 6,
                  radius_scales=_RADIUS_SCALES, reach: float = 0.4):
    """Draw a strand that carries braids, from its chord.

    Not from the strand the placement already drew. Composing a braid onto a
    meander and waving what is left of the free run back out to the contour
    was the first attempt and it does not work: it left beads two to five
    places apart at 0.5 to 1.0 sigma where the braid met the meander, on every
    geometry tried. :func:`~topon.conformation.entanglement.waypoints.
    meander_to_length` zeroes its wave across a protected stretch and ramps it
    back over 1.6 times that stretch's half-width, so the free run either side
    has to carry the *whole* contour in less room than it had before the
    braid, and folds. A meander with a second meander laid over it is not a
    meander.

    Drawn from the chord there is only ever one wave, which is the
    construction :func:`~topon.conformation.entanglement.waypoints.
    entangled_pair` has used since 0.2.0 and the one the pipeline's own
    entangled edges take. Three things are borrowed from it:

    * **draw dense, then resample once.** A wave has to be resolved before
      beads are placed on it, or the resampling cuts the corners off and loses
      the length the wave just added. ``dense_per_bead`` sets how finely; six
      is what ``entangled_pair`` uses.
    * **protect in chord fractions**, ``at +- half / chord``, not in indices of
      whatever the base happened to be.
    * **the braid keeps its geometry and the free run pays for it.** Only the
      unprotected stretches are waved.

    The braid's radius is searched alongside the wave count, which is the one
    thing ``entangled_pair`` does not have to do and this does. A turn of
    radius ``r`` drawn with a bond of ``b`` puts a bead and its second
    neighbour at ``2 r sin(2 asin(b / 2 r))``, so a radius too tight for the
    bead spacing leaves that pair touching however the free run is waved:
    measured at the default 0.9, one pair inside the braid at 0.992 against a
    floor of 1.0 with everything else clear. Widening the turn costs nothing
    in partner clearance, which is set by the gap, and ``fit_to_gap``'s
    ceiling is respected. Neither knob is monotone on its own -- the same pair
    reads 0.992, 1.101 and 0.996 at radii 0.9, 1.1 and 1.3 -- so both are
    searched and the first combination that clears the gate is taken.

    What it costs: the strand no longer has the shape the placement gave it.
    For a handful of designed pairs in a build of thousands that is what it
    should cost, since the statistical entanglement state is set by the other
    strands, and it is reported in ``info["redrawn"]`` rather than hidden.

    Returns ``(path, info)``.
    """
    from topon.conformation.entanglement.waypoints import (
        meander_to_length, resample_path)
    from topon.conformation.paths import (bond_lengths, self_contact, straight,
                                          unfold, _relax_bonds)

    a0, a1 = (np.asarray(v, float) for v in chord_pair)
    chord = float(np.linalg.norm(a1 - a0))
    dense = max(int(n_out) * int(dense_per_bead), _DENSE)
    target = (n_out - 1) * bond
    axis = (a1 - a0) / max(chord, 1e-12)

    def scaled(scale):
        """Every braid radius scaled, inside the ceiling its gap allows.

        ``fit_to_gap`` capped each radius at ``reach`` of that pair's gap so a
        braid cannot reach past its partner's chord. Widening respects the
        same ceiling.
        """
        out = []
        for contact, half, side, windings, shape in entries:
            n_r = min(shape.n_radius * scale, reach * contact.gap)
            out.append((contact, half, side, windings,
                        replace(shape, n_radius=n_r)))
        return out

    def build(these):
        """The chord with every braid on it, at dense resolution."""
        base = straight(a0, a1, dense)
        protect = []
        for contact, half, side, windings, shape in these:
            arm = braid_path(a0, a1, contact, windings, side, dense, half,
                             shape)
            u = (base - contact.origin) @ contact.axis
            inside = np.abs(u) < half
            if not inside.any():
                continue
            base[inside] = arm[inside]
            at = float((contact.origin - a0) @ axis / max(chord, 1e-12))
            half_frac = min(0.45, half / max(chord, 1e-12))
            protect.append((at - half_frac, at + half_frac))
        return base, protect

    draws = 0
    best = None
    for scale in radius_scales:
        these = scaled(scale)
        w = float(waves)
        while True:
            draws += 1
            path, protect = build(these)
            length = _path_length(path)
            if length > target:
                # The braid alone is longer than the beads can carry at this
                # radius, and waving only ever adds. Next scale.
                break
            if length < target:
                path = meander_to_length(path, target, protect=protect,
                                         waves=w)

            out = resample_path(path, n_out)
            frozen = np.zeros(n_out, bool)
            for lo, hi in protect:
                lo_i = max(0, int(np.floor(lo * (n_out - 1))))
                hi_i = min(n_out - 1, int(np.ceil(hi * (n_out - 1))))
                frozen[lo_i:hi_i + 1] = True
            # The braid's own beads are held through the unfold: opening a
            # fold is the one move that would undo a winding, since the two
            # arms of a braid are beads sitting close with no bond between.
            out = unfold(out, bond, min_sep=min_sep, iters=400, smooth=False,
                         frozen=frozen)
            interior = np.zeros(n_out, bool)
            interior[1:-1] = True
            _relax_bonds(out, bond, interior, 400, 0.0, floor=min_bond)
            _relax_bonds(out, bond, interior, 400, 0.0, only_long=True)

            b_min = float(bond_lengths(out).min())
            gap = self_contact(out, closed=is_loop)
            score = (min(b_min / max(min_bond, 1e-9), 1.0),
                     min(gap / max(min_sep, 1e-9), 1.0))
            if best is None or score > best[0]:
                best = (score, out, w, scale, b_min, gap)
            if b_min >= min_bond and gap >= min_sep:
                break
            if w <= min_waves:
                break
            # A finer ladder than halving. Several geometries were measured
            # landing at 0.986 and 0.993 against a floor of 1.0, inside 1.5 %
            # of clearing, which is the signature of stepping over the wave
            # count that would have worked rather than of a shape that cannot
            # exist.
            w = max(min_waves, 0.7 * w)
        if best is not None and best[0] >= (1.0, 1.0):
            break

    if best is None:
        raise ValueError(
            "no braid radius in the search leaves the beads enough contour: "
            "every candidate drew a path longer than (n_beads - 1) * bond")
    _score, out, w, scale, b_min, gap = best
    return out, {"waves": float(w), "radius_scale": float(scale),
                 "draws": draws, "bond_min": b_min,
                 "self_contact": float(gap), "redrawn": True, "dense": dense}


def route_designed_pairs(placement, requests: Sequence[PairRequest], *,
                         shape: Optional[BraidShape] = None,
                         min_clearance: float = 1.0,
                         apply: bool = True,
                         strict: bool = True,
                         waves: float = 6.0,
                         linking_tolerance: float = 0.25) -> DesignReport:
    """Wind every request that fits, refuse the rest by name.

    ``placement`` is a :class:`~topon.conformation.Placement`; with ``apply``
    the strands it holds are replaced by their braided paths and their gate
    readings are refreshed, so the caller can check
    :meth:`Placement.guard_report` afterwards exactly as it would for a plain
    build. The routed paths are also returned in ``DesignReport.paths``.

    Every accepted braid is composed onto the strand *as placed*, so a meander
    build stays a meander build; only the stretch inside the braid's span is
    replaced.

    With ``strict`` (the default) a pair that fails either check is put back
    the way it was and refused: the strand no longer clears the placement
    gate, or the two paths as drawn do not measure the winding that was asked
    for, within ``linking_tolerance``. Turn it off to keep what was drawn and
    have the failures reported instead; nothing else changes.

    Reverting is by pair rather than by strand, because a braid is a property
    of two chains and half of one is a chain wound about a partner that is no
    longer there -- which measures as an entanglement and is not one.
    """
    shape = shape or BraidShape()
    bond = float(placement.bond)
    report = DesignReport()
    chords = chords_of(placement)
    per_chain: dict[int, list] = {}
    taken: dict = defaultdict(list)     # chord fractions each strand has spent

    for req in requests:
        out = PairOutcome(request=req)
        if req.chain_a == req.chain_b:
            out.reason = "a strand cannot be wound with itself"
            report.outcomes.append(out)
            continue
        missing = [i for i in (req.chain_a, req.chain_b) if i not in chords]
        if missing:
            known = len(placement.strands)
            out.reason = (
                f"strand {missing[0]} has no chord to braid about "
                f"(a primary loop, or out of range: the build has {known} "
                f"strands)")
            report.outcomes.append(out)
            continue

        a0, a1 = chords[req.chain_a]
        (b0, b1), delta = nearest_image_of(chords[req.chain_b],
                                           chords[req.chain_a], placement.box)
        plan_a = placement.strands[req.chain_a].plan
        plan_b = placement.strands[req.chain_b].plan
        have = min(plan_a.n_bonds, plan_b.n_bonds) * bond

        # Axial room first, because it caps the winding count and the contour
        # a request costs is the contour of the braid it will actually get.
        # Checking the contour of the full ask would refuse a request on a
        # budget it was never going to spend.
        cost = contour_cost(a0, a1, b0, b1, req.windings, shape, min_clearance)
        out.gap = round(cost["gap"], 4)
        out.axial_needed = round(cost["axial_needed"], 4)
        out.axial_have = round(cost["axial_have"], 4)

        if cost["turns_supported"] < 1:
            out.contour_needed = round(max(cost["contour_a"],
                                           cost["contour_b"]), 4)
            out.contour_have = round(have, 4)
            out.clearance = round(cost["clearance"], 4)
            out.reason = (
                f"no axial room: a braid of {req.windings} winding(s) needs "
                f"{out.axial_needed:.1f} sigma along the shared axis and these "
                f"chords can spare {out.axial_have:.1f}. DP does not fix this "
                f"-- lower the build density so the chords lengthen, or pick a "
                f"partner in a closer shell")
            report.outcomes.append(out)
            continue

        granted = min(req.windings, cost["turns_supported"])
        short_of_room = granted < req.windings
        if short_of_room:
            cost = contour_cost(a0, a1, b0, b1, granted, shape, min_clearance)
        out.contour_needed = round(max(cost["contour_a"], cost["contour_b"]), 4)
        out.contour_have = round(have, 4)
        out.clearance = round(cost["clearance"], 4)

        if out.contour_needed > have:
            offset = min(plan_a.n_bonds - plan_a.dp, plan_b.n_bonds - plan_b.dp)
            out.minimum_dp = _minimum_dp(cost, offset, bond)
            cut = ("" if not short_of_room else
                   f" (already cut from {req.windings} by the axial room, "
                   f"which spares {out.axial_have:.1f} sigma)")
            # Naming a DP is only half an instruction: at a fixed build
            # density the box grows with the bead count, so the chords grow
            # too and the DP named here is a lower bound.
            out.reason = (
                f"the contour budget does not allow it: routing "
                f"{granted} winding(s){cut} needs {out.contour_needed:.1f} "
                f"sigma of path and DP {min(plan_a.dp, plan_b.dp)} carries "
                f"{have:.1f}")
            report.outcomes.append(out)
            continue

        if not cost["clears"]:
            # The clearance is what the whole refusal mechanism is for: a
            # braid whose two arms come within a fraction of sigma is one the
            # push-off resolves by pushing them through each other, and what
            # comes out is not the winding that was asked for.
            out.reason = (
                f"the two arms would pass within {out.clearance:.2f} sigma of "
                f"each other, under the {min_clearance:g} this pair needs. "
                f"The braid is squeezed into a chord too short for it: give "
                f"the pair more room by lowering the build density, or ask "
                f"for fewer windings")
            report.outcomes.append(out)
            continue

        if short_of_room:
            out.reason = (
                f"axial room supports {granted} of {req.windings} windings "
                f"({out.axial_have:.1f} sigma available, "
                f"{shape.span(req.windings):.1f} wanted)")

        # Two braids on the same stretch of one chord blend into each other
        # and the realised winding of both collapses. `allocate_contacts` was
        # written for this and packs the intervals properly; this refuses the
        # overlap instead, because the requests here are named by the caller
        # rather than chosen. Measured before the check: four partners asked
        # against one strand on SC 6x6x6 delivered nothing at all, every pair
        # reading a linking number of 0.00 to 0.01, while the same strands
        # asked one at a time delivered eight of eight at 1.02.
        iv_a = _interval_on(a0, a1, cost["contact"], cost["half_span"])
        iv_b = _interval_on(b0, b1, cost["contact"], cost["half_span"])
        clash = None
        if _overlaps(iv_a, taken[req.chain_a], _BRAID_SEPARATION):
            clash = req.chain_a
        elif _overlaps(iv_b, taken[req.chain_b], _BRAID_SEPARATION):
            clash = req.chain_b
        if clash is not None:
            out.reason = (
                f"strand {clash} already carries a braid on this stretch of "
                f"its chord. Two braids blending into each other collapse the "
                f"winding of both, so the second is refused: ask for it on a "
                f"different partner, or use allocate_contacts, which packs "
                f"several contacts along one chain properly")
            report.outcomes.append(out)
            continue

        out.granted = int(granted)
        report.outcomes.append(out)
        taken[req.chain_a].append(iv_a)
        taken[req.chain_b].append(iv_b)
        contact = cost["contact"]
        # A is braided in the image the contact was planned in; B lives one
        # image away, so its copy of the contact is the same frame with the
        # origin shifted back. Only the origin moves -- axis, toward and
        # across are directions.
        contact_b = replace(contact, origin=contact.origin - delta)
        per_chain.setdefault(req.chain_a, []).append(
            (contact, cost["half_span"], -1, granted, cost["shape"]))
        per_chain.setdefault(req.chain_b, []).append(
            (contact_b, cost["half_span"], +1, granted, cost["shape"]))

    notes: dict = {}
    for idx, entries in per_chain.items():
        st = placement.strands[idx]
        report.paths[idx], notes[idx] = _braided_path(
            chords[idx], entries, len(st.path), bond,
            min_bond=placement.limits.min_bond,
            min_sep=placement.limits.min_sep, waves=waves,
            is_loop=st.plan.kind == "loop")

    if not apply:
        return report

    # Compose, then read the gate again. A braid is a tight helix and the
    # detour is paid for out of the free run, so the strand that comes back is
    # not the strand that went in: measured on a DP-120 pair, an accepted
    # one-winding braid left beads two to five places apart at 0.65 sigma
    # inside the turn. Writing that is exactly the threaded bond this module
    # opens by saying it exists to prevent, so with `strict` the strand goes
    # back to the path the placement drew and the request is refused with what
    # the gate read. Without it the braid is kept and the failure is only
    # reported.
    # A winding is not delivered because the geometry was planned for it. It
    # is delivered when the two paths that were actually drawn measure it.
    # Without this check the search optimises the gate and can hand back a
    # pair that clears every bond and contact and carries no winding at all:
    # measured before it was added, requests granted "in full" whose linking
    # came out at 0.33, 0.27 and 0.15 against the 1 that was asked. That is
    # the failure this module opens by saying it exists to prevent.
    unwound: dict = {}
    for i, out in enumerate(report.outcomes):
        if out.refused:
            continue
        a, b = out.request.chain_a, out.request.chain_b
        if a not in report.paths or b not in report.paths:
            continue
        pa = report.paths[a]
        (qb, _q1), delta = nearest_image_of(
            (report.paths[b][0], report.paths[b][-1]),
            (pa[0], pa[-1]), placement.box)
        lk = abs(float(far_closed_linking(pa, report.paths[b] + delta)))
        out.linking = round(lk, 3)
        if abs(lk - out.granted) > linking_tolerance:
            unwound[i] = lk

    was_clean = {idx: not placement.strands[idx].failures(placement.limits)
                 for idx in report.paths}
    before = {idx: placement.strands[idx].path for idx in report.paths}
    degraded: dict = {}
    for idx, path in report.paths.items():
        st = placement.strands[idx]
        st.path = path
        st.routine = f"{st.routine}+braid"
        st.measure()
        why = st.failures(placement.limits)
        if why:
            # Any failure, not only one the braid introduced. Gating on
            # "was clean before" looked kinder and was worse: a strand the
            # placement had already failed got braided into something worse
            # and reported as delivered in full. Measured on a DP-120 pair at
            # coil 3.9, that produced a linking number of 2.6 for a request of
            # 1, on a path whose closest self-contact was 0.937. A winding
            # cannot be delivered on a strand that cannot be drawn.
            degraded[idx] = (why, st.bond_min, st.self_contact,
                             was_clean[idx])

    if strict and (degraded or unwound):
        # Revert by pair, not by strand. A braid is a property of two chains:
        # keeping the half that happens to clear the gate leaves one chain
        # wound around a partner that is no longer there, which measures as an
        # entanglement and is not one. So a refused request takes both its
        # strands back, and because a strand may carry braids from more than
        # one request, taking it back refuses those too -- iterate until the
        # set stops growing.
        doomed = set(unwound)
        revert = set(degraded)
        for i in unwound:
            out = report.outcomes[i]
            revert |= {out.request.chain_a, out.request.chain_b}
        while True:
            grew = False
            for i, out in enumerate(report.outcomes):
                if i in doomed or out.refused:
                    continue
                if {out.request.chain_a, out.request.chain_b} & revert:
                    doomed.add(i)
                    revert |= {out.request.chain_a, out.request.chain_b}
                    grew = True
            if not grew:
                break

        for idx in sorted(revert & set(report.paths)):
            st = placement.strands[idx]
            st.path = before[idx]
            if st.routine.endswith("+braid"):
                st.routine = st.routine[:-len("+braid")]
            st.measure()
            report.paths.pop(idx, None)

        for i in sorted(doomed):
            out = report.outcomes[i]
            if i in unwound:
                lk = unwound[i]
                asked = out.granted
                out.granted = 0
                out.reason = (
                    f"the pair was drawn but does not wind: the two paths "
                    f"measure a linking number of {lk:.2f} against the "
                    f"{asked} asked for. The braid cleared every budget and "
                    f"the gate, and then the free run's own wave took it "
                    f"apart -- lower the coil ratio so the wave is gentler, "
                    f"or ask for fewer windings")
                continue
            blamed = sorted({out.request.chain_a, out.request.chain_b}
                            & set(degraded))
            out.granted = 0
            if not blamed:
                other = sorted({out.request.chain_a, out.request.chain_b}
                               & revert)
                out.reason = (
                    f"withdrawn with the braid on strand {other[0]}, which "
                    f"another request on the same strand could not clear. A "
                    f"braid is a property of the pair, so half of one is not "
                    f"worth keeping")
                continue
            idx = blamed[0]
            why, b_min, sep, was_ok = degraded[idx]
            if not was_ok:
                out.reason = (
                    f"strand {idx} does not clear the gate before the braid "
                    f"either ({', '.join(why)}), so there is nothing to wind: "
                    f"fix the placement first, most likely by lowering the "
                    f"coil ratio")
                continue
            out.reason = (
                f"the drawn path does not clear the gate on strand {idx} "
                f"({', '.join(why)}: shortest bond {b_min:.3f}, closest "
                f"self-contact {sep:.3f}). The braid fits the chords but the "
                f"free run left over cannot carry the contour without folding "
                f"-- lower the coil ratio, or ask for fewer windings")
    return report
