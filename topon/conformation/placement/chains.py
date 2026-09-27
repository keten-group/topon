"""Where every bead of every strand goes at build, and the gate it must pass.

The conformation stage has one actuator that decides the entanglement state of
a bead-spring network, and it is not the density: it is the shape of the chain
at build. Measured on the N20 reference graph (``REPORT.md`` section 4), random
walks floor at Z = 0.23 per DP-20 strand at *any* build density -- 0.145, 0.095
and 0.035 give 0.30, 0.25 and 0.23 -- while a meander of the same contour at
rho 0.05 reproduces the reference's per-strand distribution exactly (Z 0.189
against 0.178, KS p = 1.0, partner degree and Ne_CK inside noise). Density does
move Z, but only over the range shape leaves it.

So this module makes both explicit. :func:`place` takes a graph, a DP and one of
three shapes, sizes the build box from the bead count and the build density (or,
equivalently, from a coil ratio), draws every strand, and reports what the gate
saw. Nothing here runs dynamics and nothing here writes a file: it returns
coordinates and the measurements that say whether they are fit to hand on.

The gate, and why it is not optional
------------------------------------
A path can be the right length, land on both junctions and still be unusable.
The six-wave default of :func:`meander_to_length` at DP 20 is about three beads
per wave, and resampling that at equal arc length gives bonds of 0.17 sigma with
beads jammed between their own second neighbours. The push-off then separates
them *through* the bond that lies between, and a threaded bond is what lets two
strands cross later: 57 of them on the N20 build took Z from 0.19 to 0.24
(``REPORT.md`` 4.1). The cure is upstream of the dynamics, so every strand is
checked before it is written:

* every bond at or below the design length (a longer one is a strand whose
  chord does not fit in its DP, which is a graph problem, not a placement one);
* every bond at or above ``min_bond`` (default 0.85 sigma);
* no bead within ``min_sep`` (default 1.0 sigma) of a non-adjacent bead of its
  own chain.

:func:`~topon.conformation.paths.meander_chain` meets the last two by halving
its wave count and unfolding; what it cannot meet is reported here rather than
silently written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from topon.conformation.paths import (
    bond_lengths,
    bridging_walk,
    closed_meander,
    fold_into_box,
    meander_chain,
    self_contact,
    straight_chain,
    unfold,
)

__all__ = [
    "BOND",
    "SELF_SEPARATION",
    "separate_coincident",
    "limits_from",
    "GuardLimits",
    "StrandPlan",
    "PlacedStrand",
    "Placement",
    "PLACEMENTS",
    "strand_plans",
    "lattice_frame",
    "bead_count",
    "box_for_density",
    "coil_ratio_of",
    "density_for_coil_ratio",
    "place",
]

#: Design bond length of a Kremer-Grest build, in sigma. The reference
#: end-linked datasets sit at 0.962 +- 0.011 after equilibration and never
#: exceed 1.01, so a build drawn at 0.97 needs no bond to move to join them.
BOND = 0.97

#: The three shapes :func:`place` knows.
PLACEMENTS = ("straight", "meander", "walk")

#: How close a bead may come to a non-adjacent bead of its own chain, by
#: route. For a drawn path -- a meander or a jittered chord -- a sub-sigma
#: contact is an artifact of the drawing and 1.0 is a real gate: it is what
#: catches a wave tight enough to fold back on itself.
#:
#: For a random walk it is not a gate at all, it is the physics. A freely
#: jointed walk returns near its own path constantly; that is what makes it a
#: melt chain rather than a rod, and every one of 972 DP-100 walks on the N100
#: reference graph comes within 1 sigma of itself. What a walk genuinely must
#: not have is a hard *overlap*, where WCA at a few thousandths of sigma turns
#: the push-off into a shove large enough to drag chains through each other --
#: measured on that same build, the closest such pair sat at 0.003 sigma. So
#: the walk's floor is an overlap floor, and it is the one the validation
#: script used when it nudged near-coincident pairs apart.
SELF_SEPARATION = {"meander": 1.0, "straight": 1.0, "walk": 0.05}


@dataclass(frozen=True)
class GuardLimits:
    """What a strand has to satisfy before it is written.

    ``max_bond`` defaults to the design bond length itself. It is not a FENE
    limit (that is 1.5, and the protocol gate is 1.2): it says the placement
    never *stretches* a strand, so whatever the relaxation finds is its own
    doing. The one case that trips it honestly is a chord longer than the
    contour, where no path of that many bonds exists.
    """

    min_bond: float = 0.85
    max_bond: float = BOND
    min_sep: float = 1.0
    #: Relative slack on ``max_bond`` and ``min_sep``, for the last digits of
    #: an iterative solve rather than for real stretching or a real contact.
    #: :func:`~topon.conformation.paths.unfold` separates beads to exactly
    #: ``min_sep`` and the path is then turned about its chord, so a strand
    #: sitting on the limit comes back at ``min_sep`` minus a rounding error
    #: and would otherwise be reported as a failure.
    bond_tol: float = 1e-3
    sep_tol: float = 1e-6


@dataclass(frozen=True)
class StrandPlan:
    """One strand of the graph, before it has coordinates.

    ``key`` is the edge key ``(u, v, k)`` for a bridge or a dangling strand and
    ``(u, u, k)`` for a primary loop. ``n_bonds`` counts the bonds of the whole
    strand including the two that attach it to its junctions, so a DP-20 bridge
    has 21 and a DP-20 dangling strand 20 -- its far end *is* its twentieth
    bead, which is the end-linked convention every reference dataset uses.
    """

    key: tuple
    kind: str                      # "bridge" | "dangling" | "loop"
    u: int
    v: int
    dp: int
    n_bonds: int
    n_beads: int                   # beads this strand owns (junctions excluded)


@dataclass
class PlacedStrand:
    """A strand with coordinates, and what the gate read off them."""

    plan: StrandPlan
    path: np.ndarray               # (n_bonds + 1, 3), unwrapped, ends on nodes
    routine: str
    waves: float = 0.0
    draws: int = 0
    bond_min: float = 0.0
    bond_max: float = 0.0
    self_contact: float = float("inf")
    chord: float = 0.0

    @property
    def contour(self) -> float:
        return float(bond_lengths(self.path).sum())

    def beads(self) -> np.ndarray:
        """The beads this strand owns, junctions excluded.

        A bridge gives up both ends (they are junctions); a dangling strand
        keeps its far end, which is its own free bead; a loop keeps everything
        between its two visits to the anchor.
        """
        if self.plan.kind == "bridge":
            return self.path[1:-1]
        if self.plan.kind == "dangling":
            return self.path[1:]
        return self.path[1:-1]

    def measure(self) -> None:
        """Re-read the gate off the path as it now stands.

        Called after anything that moves beads -- the junction shells, the
        coincidence pass, a designed braid -- so the reading in
        :meth:`Placement.guard_report` is always of the coordinates that will
        be written, not of an earlier draft of them.
        """
        bl = bond_lengths(self.path)
        self.bond_min = float(bl.min()) if len(bl) else 0.0
        self.bond_max = float(bl.max()) if len(bl) else 0.0
        self.self_contact = self_contact(self.path,
                                         closed=self.plan.kind == "loop")

    def failures(self, limits: GuardLimits) -> list[str]:
        out = []
        if self.bond_max > limits.max_bond * (1.0 + limits.bond_tol):
            out.append("bond above the design length")
        if self.bond_min < limits.min_bond:
            out.append("bond below min_bond")
        if self.self_contact < limits.min_sep * (1.0 - limits.sep_tol):
            out.append("self-contact")
        return out


@dataclass
class Placement:
    """Every strand placed, with the box they were drawn in.

    ``coil_ratio`` is the contour of a strand over its chord, averaged over the
    strands that have a chord (loops do not). It is the density-free way to say
    how coiled the build is, and it is the actuator the controller turns for the
    meander route; ``build_density`` is the same knob for the walk route. The
    two are one knob seen from two sides, since the box scales as
    ``rho^(-1/3)`` and the contour does not move at all.
    """

    box: np.ndarray                # sigma, the build cell
    scale: float                   # sigma per lattice unit
    build_density: float
    coil_ratio: float
    placement: str
    dp: int
    bond: float
    strands: list[PlacedStrand] = field(default_factory=list)
    n_beads: int = 0
    limits: GuardLimits = field(default_factory=GuardLimits)
    coincidence: dict = field(default_factory=dict)

    # ---------------- reporting ----------------

    @property
    def failed(self) -> list[PlacedStrand]:
        return [s for s in self.strands if s.failures(self.limits)]

    @property
    def overstretched(self) -> list[PlacedStrand]:
        """Strands whose chord does not fit in the contour their DP gives."""
        return [s for s in self.strands
                if s.bond_max > self.limits.max_bond * (1.0 + self.limits.bond_tol)]

    def guard_report(self) -> dict:
        """What the gate saw, as plain numbers for a manifest."""
        if not self.strands:
            return {"strands": 0}
        b_min = min(s.bond_min for s in self.strands)
        b_max = max(s.bond_max for s in self.strands)
        sep = min(s.self_contact for s in self.strands)
        by_routine: dict[str, int] = {}
        for s in self.strands:
            by_routine[s.routine] = by_routine.get(s.routine, 0) + 1
        bad = self.failed
        want = sum(s.plan.n_bonds * self.bond for s in self.strands)
        have = sum(s.contour for s in self.strands)
        return {
            "strands": len(self.strands),
            "beads": self.n_beads,
            "bond_min": round(b_min, 6),
            "bond_max": round(b_max, 6),
            "self_contact_min": round(sep, 6),
            "contour_realised": round(have / want, 6) if want else None,
            "routines": by_routine,
            "limits": {"min_bond": self.limits.min_bond,
                       "max_bond": self.limits.max_bond,
                       "min_sep": self.limits.min_sep},
            "failed": len(bad),
            "overstretched": len(self.overstretched),
            "coincidence": dict(self.coincidence),
            "failed_examples": [
                {"key": list(s.plan.key), "kind": s.plan.kind,
                 "why": s.failures(self.limits),
                 "bond_min": round(s.bond_min, 4),
                 "bond_max": round(s.bond_max, 4),
                 "self_contact": round(s.self_contact, 4)}
                for s in bad[:5]],
        }

    def ok(self) -> bool:
        return not self.failed

    def coordinates(self, fold: bool = True) -> np.ndarray:
        """Every bead of every strand, in strand order.

        Folded into ``[0, L)`` unless ``fold`` is false; the unfolded form is
        what a contour or a Z1+ export wants, the folded one is what a LAMMPS
        data file wants.
        """
        if not self.strands:
            return np.zeros((0, 3))
        xyz = np.vstack([s.beads() for s in self.strands])
        return fold_into_box(xyz, self.box) if fold else xyz


# ---------------------------------------------------------------------------
# The graph, in the frame the placement works in
# ---------------------------------------------------------------------------

def _kind_of(graph, node) -> str:
    """``junction`` or ``end``, however the graph chose to say it.

    Graphs written by the validation parser carry ``kind`` on every node;
    graphs from topon's own generators do not, and there a degree-1 node is
    the free end of a dangling strand -- the same rule
    :class:`~topon.chemistry.builder.ChemistryBuilder` applies when it decides
    what is an end cap.
    """
    declared = graph.nodes[node].get("kind")
    if declared in ("junction", "end"):
        return declared
    return "end" if graph.degree(node) == 1 else "junction"


def lattice_frame(graph, dims=None):
    """Node positions and cell in one consistent set of units.

    Returns ``(pos, box)``. Graphs from topon's generators keep both in
    lattice units (``G.graph["box"]``); graphs saved by the validation scripts
    keep positions in sigma with the cell in ``box_sigma`` and the conversion
    in ``scale``, so those are divided back down. Either way what comes out is
    a frame in which the cell is the period and the placement is free to pick
    the sigma-per-unit that hits the density asked for.
    """
    if dims is not None:
        box = np.asarray(dims, float).reshape(3)
        pos = {n: np.asarray(d["pos"], float)
               for n, d in graph.nodes(data=True)}
        return pos, box

    scale = float(graph.graph.get("scale", 1.0) or 1.0)
    if graph.graph.get("box_sigma") is not None:
        box = np.asarray(graph.graph["box_sigma"], float).reshape(3) / scale
    elif graph.graph.get("box") is not None:
        box = np.asarray(graph.graph["box"], float).reshape(3)
        scale = 1.0
    else:
        # No cell recorded. The extent plus one recovers the period only for
        # integer-spaced sites, which is the same fallback (and the same
        # caveat) as ConformationManager.apply_displacements.
        pts = np.array([d["pos"] for _, d in graph.nodes(data=True)], float)
        box = pts.max(axis=0) - pts.min(axis=0) + 1.0
        scale = 1.0
    pos = {n: np.asarray(d["pos"], float) / scale
           for n, d in graph.nodes(data=True)}
    return pos, box


def strand_plans(graph, dp: int) -> list[StrandPlan]:
    """Every strand of the graph, in a fixed order.

    The order is the graph's edge order, and it is what indexes a strand
    everywhere downstream -- the ``pairs`` of a designed-entanglement request,
    the chain ids of a Z1+ export, the molecule ids of an end-linked data file.
    A per-edge ``dp`` overrides the argument, so a DP distribution placed by
    the assignment stage is honoured.
    """
    multi = graph.is_multigraph()
    plans: list[StrandPlan] = []
    edges = (graph.edges(keys=True, data=True) if multi
             else ((u, v, 0, d) for u, v, d in graph.edges(data=True)))
    for u, v, key, data in edges:
        d = int(data.get("dp", dp))
        if u == v:
            # A primary loop leaves its junction and comes back: DP beads
            # between DP + 1 bonds.
            plans.append(StrandPlan((u, v, key), "loop", u, v, d, d + 1, d))
            continue
        a, b = u, v
        if _kind_of(graph, a) == "end" and _kind_of(graph, b) == "junction":
            a, b = b, a
        if _kind_of(graph, b) == "junction":
            plans.append(StrandPlan((a, b, key), "bridge", a, b, d, d + 1, d))
        else:
            # Dangling: the degree-1 node is the strand's DP-th bead, not an
            # extra one, so it carries DP - 1 interior beads plus that node.
            plans.append(StrandPlan((a, b, key), "dangling", a, b, d, d, d))
    return plans


def bead_count(graph, dp: int, plans=None) -> int:
    """Beads the build will contain: one per junction plus every strand's."""
    plans = strand_plans(graph, dp) if plans is None else plans
    n_junction = sum(1 for n in graph if _kind_of(graph, n) == "junction")
    return n_junction + sum(p.n_beads for p in plans)


def box_for_density(graph, dp: int, build_density: float, dims=None,
                    plans=None):
    """The build cell, in sigma, that puts ``n_beads`` at that bead density.

    Returns ``(box_sigma, scale, n_beads)``. The cell's *shape* comes from the
    graph and never changes; only the sigma-per-lattice-unit moves, so every
    chord scales together and the network is the same network at every build
    density.
    """
    pos, box_lat = lattice_frame(graph, dims)
    plans = strand_plans(graph, dp) if plans is None else plans
    n_beads = bead_count(graph, dp, plans)
    volume_lat = float(np.prod(box_lat))
    if build_density <= 0.0 or volume_lat <= 0.0:
        raise ValueError(
            f"build_density must be positive and the cell non-degenerate; got "
            f"density {build_density!r} and cell {tuple(box_lat)}")
    scale = float((n_beads / build_density / volume_lat) ** (1.0 / 3.0))
    return box_lat * scale, scale, n_beads


def _chords(graph, plans, pos, box_lat) -> np.ndarray:
    """Minimum-image chord of every strand that has one, in lattice units."""
    out = []
    for p in plans:
        if p.kind == "loop":
            continue
        d = pos[p.v] - pos[p.u]
        d = d - box_lat * np.round(d / box_lat)
        out.append(float(np.linalg.norm(d)))
    return np.asarray(out, float)


def coil_ratio_of(graph, dp: int, build_density: float, bond: float = BOND,
                  dims=None, plans=None) -> float:
    """Design contour over chord at this build density, over the strands.

    The definition matters because the number is quoted so often. It is the
    *mean contour over the mean chord*, both over the strands that have a
    chord; taking the mean of the per-strand ratios instead weights the short
    chords heavily and gives a different number (1.61 against 1.51 on the N20
    graph at rho 0.05). The mean-over-mean form is the one that reproduces the
    figure quoted for the direct N20 build at rho 0.3075, coil 2.8
    (``REPORT.md`` 4.1): it comes out at 2.77 here.

    The contour is the *design* one, ``n_bonds * bond``, not what the placed
    path measures. A walk lands on it exactly; a meander comes back at 98.3 %
    of it on the N20 graph, because opening its folds costs a little length.
    Using the design value is what makes the number a property of the graph
    and the density alone -- an actuator, which is what the controller needs
    -- rather than a property of the draw. ``Placement.guard_report()``
    reports the realised contour beside it so the deficit is visible.
    """
    pos, box_lat = lattice_frame(graph, dims)
    plans = strand_plans(graph, dp) if plans is None else plans
    _box, scale, _n = box_for_density(graph, dp, build_density, dims, plans)
    chords = _chords(graph, plans, pos, box_lat)
    if not len(chords):
        return float("nan")
    contour = float(np.mean([p.n_bonds * bond for p in plans
                             if p.kind != "loop"]))
    return contour / (float(chords.mean()) * scale)


def density_for_coil_ratio(graph, dp: int, coil_ratio: float,
                           bond: float = BOND, dims=None,
                           plans=None) -> float:
    """The build density at which the strands carry that coil ratio.

    Inverse of :func:`coil_ratio_of`. The coil ratio scales as ``rho^(1/3)``
    because the contour is fixed and the chords all scale with the box, so one
    evaluation at any density fixes the whole curve.
    """
    if coil_ratio <= 0.0:
        raise ValueError(f"coil_ratio must be positive; got {coil_ratio!r}")
    probe = 0.1
    have = coil_ratio_of(graph, dp, probe, bond, dims, plans)
    if not np.isfinite(have) or have <= 0.0:
        raise ValueError(
            "this graph has no strand with a chord, so a coil ratio says "
            "nothing about it; set build_density instead")
    return float(probe * (coil_ratio / have) ** 3)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _draw(kind_of_path, c0, c1, n_bonds, bond, rng, waves, limits, jitter):
    """One strand's path and the note that says how it was drawn."""
    if kind_of_path == "walk":
        p = bridging_walk(c0, c1, n_bonds, bond, rng)
        # Separate the hard overlaps and leave the melt-like contacts alone.
        # The walk is the shape of a chain in a melt and is meant to come back
        # near itself; what it must not do is put two beads on top of each
        # other, which the cone it draws from does not prevent.
        if limits.min_sep > 0 and self_contact(p) < limits.min_sep:
            p = unfold(p, bond, min_sep=limits.min_sep, iters=200,
                       smooth=False)
        # The last bond of a walk is the one the cone did not get to choose:
        # the final bead has to sit one bond from both its neighbour and the
        # junction, and about one DP-20 walk in 400 closes that at under 0.85
        # sigma (worst seen 0.816). Lift it here, where the cause is, rather
        # than leaving it to a pass whose business is overlaps.
        if float(bond_lengths(p).min()) < limits.min_bond:
            interior = np.zeros(len(p), bool)
            interior[1:-1] = True
            from topon.conformation.paths import _relax_bonds
            _relax_bonds(p, bond, interior, 200, 0.0, floor=limits.min_bond)
            _relax_bonds(p, bond, interior, 200, 0.0, only_long=True)
        return p, {"routine": "walk", "waves": 0.0, "draws": 1}
    if kind_of_path == "straight":
        p = straight_chain(c0, c1, n_bonds, jitter=jitter, rng=rng, bond=bond)
        return p, {"routine": "straight", "waves": 0.0, "draws": 1}
    return meander_chain(c0, c1, n_bonds, bond, rng, waves=waves,
                         min_sep=limits.min_sep, min_bond=limits.min_bond,
                         jitter=jitter)


def limits_from(config, placement: str) -> GuardLimits:
    """The gate a :class:`~topon.config.schema.ConformationConfig` asks for.

    ``min_self_separation`` of ``None`` means "whatever this route's floor
    is", which is the only sensible default: one number cannot be right for
    both a meander and a walk (see :data:`SELF_SEPARATION`).
    """
    bond = float(getattr(config, "bond", BOND))
    sep = getattr(config, "min_self_separation", None)
    if sep is None:
        sep = SELF_SEPARATION.get(placement, 1.0)
    return GuardLimits(min_bond=float(getattr(config, "min_bond", 0.85)),
                       max_bond=bond, min_sep=float(sep))


def place(graph, dp: int, placement: str = "meander",
          coil_ratio: Optional[float] = None,
          build_density: Optional[float] = None,
          rng=None, *, bond: float = BOND, dims=None,
          waves: float = 6.0, jitter: float = 0.02,
          limits: Optional[GuardLimits] = None,
          coincident: float = 0.05,
          junction_shell_spacing: Optional[float] = None,
          junction_shell_blend: int = 4) -> Placement:
    """Draw every strand of ``graph`` at the build state.

    ``placement`` is one of ``straight`` (the chord with a jitter), ``meander``
    (:func:`~topon.conformation.paths.meander_chain`, which falls back to the
    chord when there is no slack to wave) or ``walk``
    (:func:`~topon.conformation.paths.bridging_walk`). Primary loops have no
    chord to interpolate along and are always drawn with
    :func:`~topon.conformation.paths.closed_meander` whatever ``placement``
    says; there is only one shape that closes on a junction with every bond
    exact.

    Give exactly one of ``coil_ratio`` and ``build_density``: they are the same
    knob, and :func:`density_for_coil_ratio` converts. Giving neither is an
    error rather than a default, because a build density picked silently is a
    build whose entanglement state was picked silently.

    ``junction_shell_spacing`` seats the beads next to each junction on a
    spread shell (:mod:`topon.conformation.junction_shell`) so the chains
    leaving one crosslink do not start on top of each other; ``None`` leaves
    them where their chords put them.

    ``coincident`` is the separation below which two beads anywhere in the
    build are pushed apart (:func:`separate_coincident`); 0 skips the pass.

    Returns a :class:`Placement`. It is *not* checked for you -- read
    ``Placement.ok()`` or ``guard_report()`` and decide. The gate is advisory
    here on purpose: a caller sweeping build densities wants the failures
    reported, not raised.
    """
    if placement not in PLACEMENTS:
        raise ValueError(
            f"unknown placement {placement!r}; expected one of "
            f"{', '.join(PLACEMENTS)}")
    if (coil_ratio is None) == (build_density is None):
        raise ValueError(
            "give exactly one of coil_ratio and build_density: they are the "
            "same knob (the box scales as rho^(-1/3) and the contour does "
            "not move), and giving neither leaves the entanglement state of "
            "the build unspecified")

    rng = np.random.default_rng() if rng is None else rng
    # The self-contact floor is a property of the route, not of the build:
    # see SELF_SEPARATION. An explicit `limits` overrides it, because a caller
    # that wants one gate for a sweep across routes should get one gate.
    limits = limits or GuardLimits(
        max_bond=bond, min_sep=SELF_SEPARATION.get(placement, 1.0))
    plans = strand_plans(graph, dp)
    pos, box_lat = lattice_frame(graph, dims)

    if build_density is None:
        build_density = density_for_coil_ratio(graph, dp, float(coil_ratio),
                                               bond, dims, plans)
    box, scale, n_beads = box_for_density(graph, dp, float(build_density),
                                          dims, plans)
    realised_coil = coil_ratio_of(graph, dp, float(build_density), bond, dims,
                                  plans)

    P = {n: p * scale for n, p in pos.items()}
    box_sigma = box_lat * scale

    placed: list[PlacedStrand] = []
    for plan in plans:
        if plan.kind == "loop":
            anchor = P[plan.u]
            away = []
            for a, b in graph.edges(plan.u):
                other = b if a == plan.u else a
                if other == plan.u:
                    continue
                d = P[other] - anchor
                away.append(d - box_sigma * np.round(d / box_sigma))
            ring = closed_meander(anchor, plan.n_bonds, bond, rng,
                                  away_from=away or None)
            path = np.vstack([anchor, ring, anchor])
            note = {"routine": "closed_meander", "waves": 0.0, "draws": 1}
            chord = 0.0
        else:
            c0 = P[plan.u]
            d = P[plan.v] - c0
            d = d - box_sigma * np.round(d / box_sigma)
            c1 = c0 + d
            chord = float(np.linalg.norm(d))
            path, note = _draw(placement, c0, c1, plan.n_bonds, bond, rng,
                               waves, limits, jitter)
        placed.append(PlacedStrand(plan=plan, path=np.asarray(path, float),
                                   routine=note["routine"],
                                   waves=float(note.get("waves", 0.0)),
                                   draws=int(note.get("draws", 1)),
                                   chord=chord))

    if junction_shell_spacing:
        _seat_on_shells(placed, float(junction_shell_spacing),
                        int(junction_shell_blend))

    coincidence = (separate_coincident(placed, box_sigma, float(coincident),
                                       rng, bond=bond,
                                       min_bond=limits.min_bond)
                   if coincident else {})

    for s in placed:
        s.measure()

    return Placement(box=box_sigma, scale=scale,
                     build_density=float(build_density),
                     coil_ratio=float(realised_coil), placement=placement,
                     dp=int(dp), bond=float(bond), strands=placed,
                     n_beads=int(n_beads), limits=limits,
                     coincidence=coincidence)


def separate_coincident(placed, box, floor: float, rng, rounds: int = 8,
                        bond: float = BOND, min_bond: float = 0.85) -> dict:
    """Break up any two beads in the build that landed on top of each other.

    Two beads at *exactly* 0.0 sigma is not a near miss, and on a lattice it is
    not rare either. Measured on the N20 meander build at rho 0.05: 44 bead
    pairs inside 0.05 sigma, of which 40 belong to strands whose **chords
    intersect** -- the chord-to-chord distance at closest approach is 0.0000
    for the median of them. A MIX cell with four neighbour shells has chords
    long enough to pass exactly through other sites and exactly across other
    chords, and a meander's wave vanishes at both ends, so the beads two to
    four places from a junction still lie on the chord (27 of the 44 sit at
    index 2 from an end, 9 at index 4). Where two chords cross near an end,
    those beads land on the crossing point.

    The kick is along the pair vector, sized to just clear the floor, and
    random only for a pair at 0.0 sigma, which has no direction to separate
    along. :func:`~topon.conformation.paths.unfold` does the same within one
    chain. Successive rounds re-read the pairs, so a kick that does not clear
    a contact is followed by another; measured, the closest pair goes from 0.0
    to just over the floor in under eight rounds.

    The kick used to be random for *every* pair, which is adequate at the
    0.05 sigma default -- almost every pair there really is at 0.0 -- and
    fails badly above about 0.5, where most pairs have a perfectly good
    direction and a random displacement the size of the floor creates about
    as many violations as it clears. Measured on the N20 meander build at
    rho 0.05 with a floor of 1.0: 944 197 kicks over eight rounds that left
    the closest pair at 0.030 sigma, *tighter* than the 0.051 of the
    untreated build, with self-contact collapsed to 0.008 and 3231 strands
    failing the gate. Directed, the same floor gives 0.485, 0.841 and 278.

    Pairs held by a bond take no part: two beads adjacent along one strand,
    and a strand's own end bead against the junction it is bonded to. At a
    floor anywhere near the design bond length every bond in the system is
    otherwise a violation -- 97 307 of them on that build -- and the pass
    spends its whole budget fighting the bond relaxation below instead of
    separating anything.

    The per-strand gate looks at one chain at a time, and two *different*
    chains can still be drawn through the same point: on the N100 reference
    build the validation script counted 69 pairs inside 0.05 sigma. Those are
    the ones that matter. WCA at 0.05 sigma is of order 1e9 kT, and the
    push-off does not relax a contact that stiff, it shoves -- hard enough to
    drag one chain through another, which is the one move that rewrites the
    topology the build just decided.

    Junctions take part as *obstacles*. A bead sitting on a crosslink is the
    same blow-up as a bead sitting on another bead, but the crosslink cannot
    be the one to move: its position is shared by every strand that meets
    there, and shifting one strand's copy of it tears the network apart. So
    the bead moves and the junction does not.

    After each round the strands that moved have their bonds relaxed back
    towards the design length -- both ways, so the nudge does not eat the
    contour -- and the two constraints are then re-read. That is the
    difference from the validation script's own nudge, which left bonds at
    1.39 sigma in the build it wrote; a 1.39 sigma bond before step zero is a
    threaded bond waiting to happen.

    Returns what it found and what it left.
    """
    from scipy.spatial import cKDTree

    from topon.conformation.paths import _relax_bonds

    L = np.asarray(box, float).reshape(3)
    if not placed:
        return {}
    sizes = [len(s.beads()) for s in placed]
    starts = np.cumsum([0] + sizes[:-1])
    n_mobile = int(sum(sizes))

    # The junctions, once each: they are obstacles, not movers.
    seen: dict = {}
    for s in placed:
        seen.setdefault(s.plan.u, np.asarray(s.path[0], float))
        if s.plan.kind != "dangling":
            seen.setdefault(s.plan.v, np.asarray(s.path[-1], float))
    anchors = (np.vstack(list(seen.values())) if seen
               else np.zeros((0, 3)))

    # Which strand owns each mobile bead and where along it that bead sits,
    # so a pair held by a bond can be told from a pair that is free to move.
    owner = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)]) \
        if sizes else np.zeros(0, int)
    along = np.concatenate([np.arange(n) for n in sizes]) \
        if sizes else np.zeros(0, int)
    anchor_row = {node: i for i, node in enumerate(seen)}
    # (strand, bead index) -> the row of the junction that bead is bonded to.
    own_anchor: dict = {}
    for k, s in enumerate(placed):
        own_anchor[(k, 0)] = n_mobile + anchor_row[s.plan.u]
        if s.plan.kind == "bridge":
            own_anchor[(k, sizes[k] - 1)] = n_mobile + anchor_row[s.plan.v]
        elif s.plan.kind == "loop":
            own_anchor[(k, sizes[k] - 1)] = n_mobile + anchor_row[s.plan.u]

    def free_pairs(pairs):
        """Drop the pairs a bond holds together; they cannot separate."""
        if not len(pairs):
            return pairs
        i, j = pairs[:, 0], pairs[:, 1]
        mobile = j < n_mobile
        bonded = np.zeros(len(pairs), bool)
        bonded[mobile] = ((owner[i[mobile]] == owner[j[mobile]])
                          & (np.abs(along[i[mobile]] - along[j[mobile]]) <= 1))
        for n, (a, b, m) in enumerate(zip(i, j, mobile)):
            if not m and own_anchor.get((int(owner[a]), int(along[a]))) == int(b):
                bonded[n] = True
        return pairs[~bonded]

    def interior_mask(strand):
        n = len(strand.path)
        m = np.zeros(n, bool)
        m[1:-1] = True
        if strand.plan.kind == "dangling":
            m[-1] = True
        return m

    def gather():
        return np.vstack([s.beads() for s in placed])

    def survey(points):
        allp = np.vstack([points, anchors]) if len(anchors) else points
        w = fold_into_box(allp, L)
        tree = cKDTree(w, boxsize=L)
        return tree, w

    found = 0
    worst_before = None
    moved: set = set()
    for _round in range(int(rounds)):
        xyz = gather()
        tree, w = survey(xyz)
        if worst_before is None:
            near = tree.query(w[:n_mobile], k=2)[0][:, 1]
            worst_before = float(near.min()) if len(near) else float("inf")
        pairs = tree.query_pairs(float(floor), output_type="ndarray")
        # A pair of two anchors is two junctions of the graph sitting close,
        # which is the topology's business and not this pass's.
        pairs = pairs[pairs[:, 0] < n_mobile] if len(pairs) else pairs
        pairs = free_pairs(pairs)
        if not len(pairs):
            break
        found += len(pairs)

        # How far short of the floor each pair is, and which way it opens.
        d = w[pairs[:, 0]] - w[pairs[:, 1]]
        d -= L * np.round(d / L)
        sep = np.linalg.norm(d, axis=1)
        # Half the shortfall each when both beads can move, slightly
        # over-relaxed so a bead several pairs pull on at once still makes
        # progress -- and the whole shortfall when the partner is a junction,
        # which will not be moving to meet it.
        share = np.where(pairs[:, 1] < n_mobile, 0.55, 1.0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            step_by_pair = ((float(floor) - sep)[:, None]
                            * (d / sep[:, None]) * share)
        # A pair at 0.0 sigma has no direction, and gets the same random kick
        # it has always had -- scaled to the floor, not to the shortfall,
        # which at the 0.05 default is the difference between clearing a
        # coincidence in one round and grinding at it for several while the
        # bond relaxation pulls the strand back into a fold.
        blind = sep < 1e-9
        if blind.any():
            step_by_pair[blind] = rng.normal(0.0, float(floor),
                                             (int(blind.sum()), 3))

        touched = set()
        for (i, j), step in zip(pairs, step_by_pair):
            xyz[i] = xyz[i] + step
            touched.add(int(np.searchsorted(starts, i, side="right") - 1))
            if j < n_mobile:
                xyz[j] = xyz[j] - step
                touched.add(int(np.searchsorted(starts, j, side="right") - 1))
        moved |= touched
        for k in sorted(touched):
            st = placed[k]
            own = xyz[starts[k]:starts[k] + sizes[k]]
            if st.plan.kind == "dangling":
                st.path[1:] = own
            else:
                st.path[1:-1] = own
            _relax_bonds(st.path, bond, interior_mask(st), 64, 0.0,
                         floor=min_bond)

    # Whatever the compromise between the two constraints, the bonds of the
    # strands this pass moved leave it inside the gate: anything above the
    # design length is shortened and anything below ``min_bond`` is
    # lengthened, and every bond in between is left where the drawing put it.
    #
    # The band is the whole point. Driving *every* bond to the design length
    # instead -- which is what an untargeted two-sided relax does -- pulls the
    # path back into the folds :func:`unfold` had just opened: measured on the
    # N20 build it took the strands failing the self-contact gate from 0 to
    # 114 at rho 0.05 and from 42 to 2204 at rho 0.3075, and left the closest
    # pair in the system at 0.044 sigma, under the floor this pass exists to
    # enforce.
    #
    # And only the strands it moved. Running it over all of them rewrites the
    # gate reading of strands this pass never touched: a `straight` build on a
    # coiled lattice has bonds of 0.64 sigma because the chord cannot carry
    # the contour, and stretching those to `min_bond` buckles the chord into a
    # self-contact -- so the report blames the fold instead of the shape,
    # which is the one thing it needed to say.
    #
    # A strand that *was* moved can still come back inside the fold gate: the
    # kick and the bond band pull against each other, and on the N20 meander
    # build one bridge in a few thousand ends around 0.92 sigma against a
    # floor of 1.0. That is a real compromise between two constraints and it
    # is reported as a failure, not smoothed over.
    for k in sorted(moved):
        st = placed[k]
        mask = interior_mask(st)
        _relax_bonds(st.path, bond, mask, 200, 0.0, floor=min_bond)
        _relax_bonds(st.path, bond, mask, 200, 0.0, only_long=True)

    xyz = gather()
    tree, w = survey(xyz)
    near = tree.query(w[:n_mobile], k=2)[0][:, 1]
    return {"floor": float(floor), "pairs_nudged": int(found),
            "closest_pair_before": (None if worst_before is None
                                    else round(worst_before, 6)),
            "closest_pair": round(float(near.min()), 6) if len(near) else None}


def _seat_on_shells(placed, spacing: float, blend: int) -> None:
    """Spread the first beads of the strands that share a junction, in place.

    Loops are left out: both of a loop's ends are the same junction, and a
    shell seat that pulls its two ends apart opens the ring it was drawn to
    close.
    """
    from topon.conformation.junction_shell import apply_junction_shells

    movable = [s for s in placed if s.plan.kind != "loop"]
    if not movable:
        return
    paths = {i: s.path for i, s in enumerate(movable)}
    ends = {i: (s.plan.u, s.plan.v) for i, s in enumerate(movable)}
    out = apply_junction_shells(paths, ends, spacing=spacing, blend=blend)
    for i, s in enumerate(movable):
        s.path = out[i]
