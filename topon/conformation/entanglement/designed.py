"""Named and shell-selected pairs, wound on top of chains that are already placed.

The placement stage decides the entanglement state of the network as a
*statistic*: a shape and a build density, and Z comes out where the calibration
says it will. This module is the other kind of request -- *these two strands,
this many windings* -- and it works on the strands as placed rather than
replacing them, so the statistical state the build was tuned for survives the
designed ones being added.

What this delivers today
------------------------
The budget half is complete and measured: a request is priced before anything
is drawn, and one that does not fit is refused by name with the minimum DP
where the contour binds and with the reason DP will not help where the axial
room binds.

The *routing* half does not yet clear the gate on a strand that was already
drawn as a meander. Composing a braid onto one and waving the remaining free
run back out to the contour leaves beads two to five places apart at 0.48 to
0.99 sigma at the seam where the braid meets the meander, across every
geometry tried (DP 60 to 300, chords 15 to 25 sigma, one winding). That is the
threaded-bond pathology the gate exists to catch, so with ``strict`` -- the
default -- the strand goes back to the path the placement drew and the request
is refused with the reading that failed. The cause is understood:
:func:`meander_to_length` zeroes its wave across a protected stretch and ramps
it back over 1.6 times the stretch's half-width, so the free run on either side
has to carry the whole contour in less room than it had before the braid, and
it folds. The fix is to build a braided strand *from its chord* the way
:func:`~topon.conformation.entanglement.waypoints.entangled_pair` and
:mod:`~topon.conformation.entanglement.realize` already do for the pipeline's
entangled edges, rather than composing onto a meander; that is the route to use
for delivered windings today.

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
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

import numpy as np

from topon.conformation.entanglement.braid import (
    BraidShape,
    axial_room,
    braid_path,
    closest_approach,
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

def _compose_on_path(path, chord_pair, entries, bond: float,
                     min_bond: float = 0.85, min_sep: float = 1.0,
                     waves: float = 6.0, min_waves: float = 0.5,
                     is_loop: bool = False):
    """Put the braids on a strand that already has a shape.

    The braid is defined on the chord, so the beads inside its axial span take
    the braid and every other bead stays on the path the placement drew. The
    contour is then restored by waving the *unprotected* stretches out again
    (:func:`meander_to_length` with the braid intervals protected), so the
    winding keeps the geometry it was given and only the free run between the
    braids takes up the slack the detour spent.

    Two things the plain placement already knows, and that this has to know
    too. The wave count is searched, not fixed: the detour leaves less free run
    to spend the contour in than the strand started with, so the waves that
    were fine before the braid fold after it, and the first sign is a bead
    sitting on a bead a few places along at the seam where the braid meets the
    meander -- measured at 0.65 sigma on a DP-120 pair. And the path is *not*
    resampled at equal arc afterwards: a braid is a tight helix, so resampling
    along the polyline cuts every corner off it, which put a 0.569 sigma bond
    inside the turn.

    The braid's own beads are held still through all of it. Unfolding pushes
    apart beads that sit close with no bond between them, which is precisely
    what the two turns of a braid do.

    Returns ``(path, info)``.
    """
    from topon.conformation.entanglement.waypoints import (
        meander_to_length, resample_path)
    from topon.conformation.paths import (bond_lengths, self_contact, unfold,
                                          _relax_bonds)

    a0, a1 = chord_pair
    n_out = len(path)
    base = resample_path(np.asarray(path, float), _DENSE)
    target = (n_out - 1) * bond

    braided = []
    protect = []
    for contact, half, side, windings, shape in entries:
        arm = braid_path(a0, a1, contact, windings, side, _DENSE, half, shape)
        u = (base - contact.origin) @ contact.axis
        inside = np.abs(u) < half
        if not inside.any():
            continue
        braided.append((inside, arm))
        idx = np.flatnonzero(inside)
        protect.append((idx[0] / (_DENSE - 1.0), idx[-1] / (_DENSE - 1.0)))

    w = float(waves)
    best = None
    draws = 0
    while True:
        draws += 1
        dense = np.array(base, copy=True)
        for inside, arm in braided:
            dense[inside] = arm[inside]
        if _path_length(dense) < target:
            dense = meander_to_length(dense, target, protect=protect, waves=w)

        out = resample_path(dense, n_out)
        frozen = np.zeros(n_out, bool)
        for lo, hi in protect:
            frozen[int(np.floor(lo * (n_out - 1))):
                   int(np.ceil(hi * (n_out - 1))) + 1] = True
        out = unfold(out, bond, min_sep=min_sep, iters=120, smooth=False,
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
            best = (score, out, w, b_min, gap)
        if (b_min >= min_bond and gap >= min_sep) or w <= min_waves:
            break
        w = max(min_waves, 0.5 * w)

    _score, out, w, b_min, gap = best
    return out, {"waves": float(w), "draws": draws, "bond_min": b_min,
                 "self_contact": float(gap)}


def route_designed_pairs(placement, requests: Sequence[PairRequest], *,
                         shape: Optional[BraidShape] = None,
                         min_clearance: float = 1.0,
                         apply: bool = True,
                         strict: bool = True,
                         waves: float = 6.0) -> DesignReport:
    """Wind every request that fits, refuse the rest by name.

    ``placement`` is a :class:`~topon.conformation.Placement`; with ``apply``
    the strands it holds are replaced by their braided paths and their gate
    readings are refreshed, so the caller can check
    :meth:`Placement.guard_report` afterwards exactly as it would for a plain
    build. The routed paths are also returned in ``DesignReport.paths``.

    Every accepted braid is composed onto the strand *as placed*, so a meander
    build stays a meander build; only the stretch inside the braid's span is
    replaced.

    With ``strict`` (the default) a composed strand that no longer clears the
    gate is put back the way it was and its request is refused with the
    reading that failed. Turn it off to keep the braid and have the failure
    reported instead; nothing else changes.
    """
    shape = shape or BraidShape()
    bond = float(placement.bond)
    report = DesignReport()
    chords = chords_of(placement)
    per_chain: dict[int, list] = {}

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

        out.granted = int(granted)
        report.outcomes.append(out)
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
        report.paths[idx], notes[idx] = _compose_on_path(
            st.path, chords[idx], entries, bond,
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
        if why and was_clean[idx]:
            degraded[idx] = (why, st.bond_min, st.self_contact)

    if strict and degraded:
        for idx in degraded:
            st = placement.strands[idx]
            st.path = before[idx]
            st.routine = st.routine[:-len("+braid")]
            st.measure()
            report.paths.pop(idx, None)
        for out in report.outcomes:
            touched = ({out.request.chain_a, out.request.chain_b}
                       & set(degraded))
            if not touched or out.refused:
                continue
            why, b_min, sep = degraded[sorted(touched)[0]]
            idx = sorted(touched)[0]
            out.granted = 0
            out.reason = (
                f"the composed path does not clear the gate on strand {idx} "
                f"({', '.join(why)}: shortest bond {b_min:.3f}, closest "
                f"self-contact {sep:.3f}). The braid fits the chords but not "
                f"the beads -- they are packed tightly enough inside the turn "
                f"that opening it would undo the winding. Lower the build "
                f"density so the braid has room, or ask for fewer windings")
    return report
