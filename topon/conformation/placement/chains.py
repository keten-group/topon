"""Where every bead of every strand goes at build, and the gate it must pass.

The conformation stage has one actuator that decides the entanglement state of
a bead-spring network, and it is not the density: it is the shape of the chain
at build. Measured on the N20 reference graph (``REPORT.md`` section 4), random
walks floor at Z = 0.23 per DP-20 strand at *any* build density -- 0.145, 0.095
and 0.035 give 0.30, 0.25 and 0.23 -- while a meander of the same contour at
rho 0.05 reproduces the reference's per-strand distribution exactly (Z 0.189
against 0.178, KS p = 1.0, partner degree and Ne_CK inside noise). Density does
move Z, but only over the range shape leaves it. Those are the validation
scripts' builds. Built here with the pinch fix and relaxed through the full
push-off, the two shapes' final-state floors coincide at DP 20 (walk
0.2365 at rho 0.035, meander 0.2356 at coil 1.402), both above the reference's
0.178, though at one build density the meander is still the lower.

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
    Clearance,
    STRAIGHT_AT,
    _emptiest,
    bond_lengths,
    bridging_walk,
    chord_side,
    closed_meander,
    closed_walk,
    fold_into_box,
    free_walk,
    meander_chain,
    self_contact,
    shared_chords,
    straight_chain,
    unfold,
)
from topon.conformation.placement.settle import chord_triples, settle_strands

__all__ = [
    "BOND",
    "LOOP_SEPARATION",
    "LOOP_SHAPES",
    "SELF_SEPARATION",
    "bead_bond_gaps",
    "separate_coincident",
    "limits_from",
    "GuardLimits",
    "StrandPlan",
    "PlacedStrand",
    "Placement",
    "PLACEMENTS",
    "PARALLEL_STRANDS",
    "sol_lengths",
    "strand_plans",
    "lattice_frame",
    "bead_count",
    "box_for_density",
    "coil_ratio_of",
    "density_for_coil_ratio",
    "place",
    "settle_placement",
    "site_spacing",
]

#: Design bond length of a Kremer-Grest build, in sigma. The reference
#: end-linked datasets sit at 0.962 +- 0.011 after equilibration and never
#: exceed 1.01, so a build drawn at 0.97 needs no bond to move to join them.
BOND = 0.97

#: The three shapes :func:`place` knows.
PLACEMENTS = ("straight", "meander", "walk")

#: How :func:`place` draws bridges that share both junctions on the meander
#: route: on opposite sides of their chord, or as any other strand.
PARALLEL_STRANDS = ("opposite", "together")

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

#: How a primary loop is drawn. ``ring`` is the regular polygon of
#: :func:`~topon.conformation.paths.closed_meander`, whose radius grows with
#: the DP (15.6 sigma at DP 100); ``compact`` is the closed self-avoiding walk
#: of :func:`~topon.conformation.paths.closed_walk`, about the size a ring of
#: that DP has in a melt.
LOOP_SHAPES = ("ring", "compact")

#: The least floor a ``compact`` loop is grown to, in sigma: no bead of the
#: ring within a bead diameter of another that is not its bonded neighbour,
#: the junction included. A loop has no chord to hold it open, so without
#: excluded volume it is an ideal ring (radius of gyration 2.8 sigma at
#: DP 100), well inside the 4.5 a DP-100 loop has in the N100 reference; at
#: 1.0 the walk gives 4.29 there, and 1.85 at DP 20 against the N20
#: reference's 1.96. On the walk route, whose own floor is 0.05, the loops
#: are therefore grown to 1.0; a route floor above it is kept.
LOOP_SEPARATION = 1.0


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

    ``key`` is the edge key ``(u, v, k)`` for a bridge or a dangling strand,
    ``(u, u, k)`` for a primary loop and ``("sol", i)`` for the ``i``-th sol
    chain, which has no edge and no junction (``u`` and ``v`` are ``None``).
    ``n_bonds`` counts the bonds of the whole strand including the two that
    attach it to its junctions, so a DP-20 bridge has 21 and a DP-20 dangling
    strand 20 -- its far end *is* its twentieth bead, which is the end-linked
    convention every reference dataset uses -- and a DP-20 sol chain 19.
    ``dp`` is the chain's bead count, the same number as ``n_beads``.
    """

    key: tuple
    kind: str                      # "bridge" | "dangling" | "loop" | "free"
    u: Optional[int]
    v: Optional[int]
    dp: int
    n_bonds: int
    n_beads: int                   # beads this strand owns (junctions excluded)

    @property
    def chorded(self) -> bool:
        """Whether the strand runs between two sites: a bridge or a dangling
        strand. A loop leaves and returns to one junction and a sol chain
        touches none, so neither has a chord."""
        return self.kind in ("bridge", "dangling")


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
    #: The height of the bow a strand sharing its chord with another was
    #: drawn on (:func:`_parallel_strands`), 0 if it was not drawn on a side
    #: (it went straight, or the meander fell back to a walk); ``None`` for a
    #: strand that shares its chord with none, or off the meander route.
    bow: Optional[float] = None

    @property
    def contour(self) -> float:
        return float(bond_lengths(self.path).sum())

    @property
    def own(self) -> slice:
        """Where along ``path`` the beads this strand owns sit.

        A bridge gives up both ends (they are junctions); a dangling strand
        keeps its far end, which is its own free bead; a loop keeps everything
        between its two visits to the anchor; a sol chain is all its own.
        """
        if self.plan.kind == "dangling":
            return slice(1, None)
        if self.plan.kind == "free":
            return slice(0, None)
        return slice(1, -1)

    def beads(self) -> np.ndarray:
        """The beads this strand owns, junctions excluded (see :attr:`own`)."""
        return self.path[self.own]

    def measure(self) -> None:
        """Re-read the gate off the path as it now stands.

        Called after anything that moves beads -- the junction shells, the
        coincidence pass, a designed braid -- so the reading in
        :meth:`Placement.guard_report` is always of the coordinates that will
        be written, not of an earlier draft of them. A sol chain of one bead
        has no bond, and reads as none too short and none too long.
        """
        bl = bond_lengths(self.path)
        self.bond_min = float(bl.min()) if len(bl) else float("inf")
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
    strands that have a chord (loops and sol chains do not). It is the density-free way to say
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
    junction_shells: dict = field(default_factory=dict)
    bead_bond: dict = field(default_factory=dict)
    junction_jitter: dict = field(default_factory=dict)
    settle: dict = field(default_factory=dict)
    chord_triples: dict = field(default_factory=dict)
    loop_shape: str = "ring"

    # ---------------- reporting ----------------

    def loop_report(self) -> dict:
        """The primary loops as they stand: shape, floor and size.

        The radius of gyration is over each loop's own beads, junction
        excluded, which is how a relaxed reference's loops are read
        (:mod:`topon.analysis.endlinked`, ``strand_path(junctions=False)``).
        Empty for a build with no loop.
        """
        rings = [s for s in self.strands if s.plan.kind == "loop"]
        if not rings:
            return {}
        rg = np.array([float(np.sqrt(((b - b.mean(axis=0)) ** 2)
                                     .sum(axis=1).mean()))
                       for b in (s.beads() for s in rings)])
        return {"shape": self.loop_shape, "count": len(rings),
                "floor": (round(max(self.limits.min_sep, LOOP_SEPARATION), 6)
                          if self.loop_shape == "compact" else None),
                "rg_mean": round(float(rg.mean()), 4),
                "rg_min": round(float(rg.min()), 4),
                "rg_max": round(float(rg.max()), 4),
                "self_contact_min": round(min(s.self_contact for s in rings),
                                          6)}

    def parallel_report(self) -> dict:
        """The strands that share both junctions with another, and how they
        were drawn: ``{}`` on a graph with none."""
        shared = [s for s in self.strands if s.bow is not None]
        if not shared:
            return {}
        chords = {frozenset((s.plan.u, s.plan.v)) for s in shared}
        bows = np.array([s.bow for s in shared if s.bow > 0.0], float)
        apart = [s for s in shared if s.bow > 0.0]
        # the angle each bow leaves its junctions at
        angle = np.degrees(np.arctan(np.pi * bows / np.array(
            [s.chord for s in apart], float))) if len(apart) else bows
        return {
            "chords": len(chords),
            "strands": len(shared),
            "drawn_apart": len(apart),
            "left_on_chord": len(shared) - len(apart),
            "bow_mean": round(float(bows.mean()), 4) if len(bows) else None,
            "bow_min": round(float(bows.min()), 4) if len(bows) else None,
            "bow_max": round(float(bows.max()), 4) if len(bows) else None,
            "angle_min": round(float(angle.min()), 3) if len(bows) else None,
        }

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
        # The realised contour is a reading of the drawn shapes; a sol chain
        # is grown at the design bond exactly and would only dilute it.
        drawn = [s for s in self.strands if s.plan.kind != "free"]
        want = sum(s.plan.n_bonds * self.bond for s in drawn)
        have = sum(s.contour for s in drawn)
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
            "junction_shells": dict(self.junction_shells),
            "bead_bond": dict(self.bead_bond),
            "chord_triples": dict(self.chord_triples),
            "parallel": self.parallel_report(),
            "junction_jitter": dict(self.junction_jitter),
            "settle": dict(self.settle),
            "coincidence": dict(self.coincidence),
            "loops": self.loop_report(),
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


def sol_lengths(graph, dp: int) -> list[int]:
    """The bead count of every sol chain ``G.graph["sol_chains"]`` records.

    Read as the chemistry builder reads it: ``dps`` (DP to count) when the
    chains differ in length, as a crosslinked melt records them, otherwise
    ``count`` chains of ``dp``; a count of zero or a length below one is no
    chain. A record with only a count, as :mod:`topon.analysis.endlinked`,
    :mod:`topon.analysis.crosslinked` and :mod:`topon.inverse.measure` write
    when they read a system back, takes ``dp`` here. The builder takes 25
    for such a record; the records the defects stage and
    :mod:`topon.topology.chain_crosslinking` write always carry a length, so
    the two agree on every graph built from a config.
    """
    spec = graph.graph.get("sol_chains")
    if not isinstance(spec, dict) or int(spec.get("count") or 0) <= 0:
        return []
    if spec.get("dps"):
        lengths = [int(d) for d, k in sorted(spec["dps"].items())
                   for _ in range(int(k))]
    else:
        given = spec.get("dp")
        lengths = [int(dp if given is None else given)] * int(spec["count"])
    return [d for d in lengths if d > 0]


def strand_plans(graph, dp: int) -> list[StrandPlan]:
    """Every strand of the graph, in a fixed order.

    The order is the graph's edge order, then the sol chains, and it is what
    indexes a strand everywhere downstream -- the ``pairs`` of a
    designed-entanglement request, the chain ids of a Z1+ export, the
    molecule ids of an end-linked data file. The sol chains come last so
    that the index of every other strand is what it was on a graph without
    them.

    A per-edge ``dp`` overrides the argument, so a DP distribution placed by
    the assignment stage is honoured, and it is read the way the chemistry
    builder reads it (:mod:`topon.analysis.crosslinked` states the same
    convention): the beads strictly between the strand's two nodes. A
    dangling strand's free end is a node, and a bead of its own, so the
    strand is ``dp + 1`` beads: DP under
    ``dp_distribution.endlinked_dangling``, which writes DP - 1 on that edge,
    and DP + 1 in topon's own convention, as the pipeline builds it either
    way. The argument, when an edge carries no ``dp``, is the chain's DP, and
    a dangling chain is then DP beads with the end site the last of them.
    Every graph the validation scripts saved is of that kind (none of 632
    carries an edge ``dp``).

    Sol chains (``G.graph["sol_chains"]``, :func:`sol_lengths`) are
    ``"free"`` strands of that many beads.
    """
    multi = graph.is_multigraph()
    plans: list[StrandPlan] = []
    edges = (graph.edges(keys=True, data=True) if multi
             else ((u, v, 0, d) for u, v, d in graph.edges(data=True)))
    for u, v, key, data in edges:
        given = data.get("dp")
        d = int(dp if given is None else given)
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
            # Dangling: the degree-1 node is one of the strand's beads. An
            # edge dp does not count it; a DP passed in does.
            n = d + 1 if given is not None else d
            plans.append(StrandPlan((a, b, key), "dangling", a, b, n, n, n))
    for i, n in enumerate(sol_lengths(graph, dp)):
        plans.append(StrandPlan(("sol", i), "free", None, None, n, n - 1, n))
    return plans


def bead_count(graph, dp: int, plans=None) -> int:
    """Beads the build will contain: one per junction plus every strand's.

    The sol chains are strands here, so this is the chemistry stage's bead
    budget (:func:`topon.assignment.defects.bead_budget`) for a graph that
    came through the pipeline, and the box the density is set in holds them.
    """
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
        if not p.chorded:
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
                             if p.chorded]))
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

def _draw(kind_of_path, c0, c1, n_bonds, bond, rng, waves, limits, jitter,
          side=None):
    """One strand's path and the note that says how it was drawn.

    ``side`` is passed to :func:`~topon.conformation.paths.meander_chain`
    (a strand drawn on one side of a chord it shares); the other routes do
    not take it.
    """
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
                         jitter=jitter, side=side)


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


def site_spacing(graph, box) -> float:
    """The mean distance between sites, in the units of ``box``.

    ``(V / N)^(1/3)`` over every node of the graph (junctions and the sites of
    free ends). On a full simple-cubic lattice it is the lattice constant; on
    the N20 MIX 90/5/5 graph at coil 1.51 it is 8.91 sigma against the 8.86 of
    box over sites per side; and it means the same thing on a lattice whose
    unit cell holds several sites (Diamond: half the cell) or on a graph with
    no lattice at all, where the lattice unit would not.
    """
    n = max(int(graph.number_of_nodes()), 1)
    return float(np.prod(np.asarray(box, float).reshape(3)) / n) ** (1.0 / 3.0)


def _reach_stats(reach, taut) -> dict:
    if not len(reach):
        return {}
    return {"mean": round(float(reach.mean()), 4),
            "p95": round(float(np.percentile(reach, 95)), 4),
            "max": round(float(reach.max()), 4),
            "taut": int((reach >= taut).sum()),
            "over_contour": int((reach > 1.0).sum())}


#: The closest two jittered junctions may come, in sigma, unless their sites
#: were already closer. Nothing afterwards can part them: the coincidence
#: pass leaves pairs of junctions alone and the settle holds both.
JUNCTION_FLOOR = 1.0


def _jitter_junctions(graph, plans, P, box, fraction: float, bond: float,
                      rng, taut: float = 0.97, rounds: int = 40,
                      floor: float = JUNCTION_FLOOR) -> dict:
    """Move every junction by a Gaussian offset, in place, and say what it did.

    The offset is drawn per axis with a standard deviation of ``fraction``
    times :func:`site_spacing`, one draw for every junction in graph order.
    The graph does not change; its chords do, and so do the reach (chord over
    design contour) and the coil ratio of every strand.

    A jittered chord can reach the contour, which on a dilute build the lattice
    chord did not. The meander goes straight at ``taut`` (0.97) of it and a
    chord past the contour is a strand the gate rejects, so an offset may not
    take any strand past ``max(taut * contour, its lattice chord)``: the
    offsets at both ends of such a strand are halved until none does (set to
    zero after ``rounds``, and checked again, until nothing is left over).
    Nor may it flip a chord's periodic image, which a cell smaller than twice
    its longest chord would allow, or bring two junctions within ``floor``
    of each other (or closer than their sites were). The report gives the
    reach before and after, and how many junctions were held back.
    """
    from scipy.spatial import cKDTree

    L = np.asarray(box, float).reshape(3)
    nodes = [n for n in graph if _kind_of(graph, n) == "junction"]
    spacing = site_spacing(graph, L)
    sd = float(fraction) * spacing
    offs = rng.normal(0.0, sd, (len(nodes), 3))
    row = {n: k for k, n in enumerate(nodes)}

    chorded = [p for p in plans if p.chorded]
    iu = np.array([row.get(p.u, -1) for p in chorded], int)
    iv = np.array([row.get(p.v, -1) for p in chorded], int)
    contour = np.array([p.n_bonds * bond for p in chorded], float)
    d0 = np.array([P[p.v] - P[p.u] for p in chorded], float).reshape(-1, 3)
    d0 = d0 - L * np.round(d0 / L)
    c0 = np.linalg.norm(d0, axis=1)
    cap = np.maximum(taut * contour, c0)

    def jittered(o):
        ou = np.where(iu[:, None] >= 0, o[np.maximum(iu, 0)], 0.0)
        ov = np.where(iv[:, None] >= 0, o[np.maximum(iv, 0)], 0.0)
        return d0 + ov - ou

    site = (np.array([P[n] for n in nodes], float).reshape(-1, 3)
            if nodes else np.zeros((0, 3)))

    def crowded(o):
        """Junctions a jitter brings within the floor of another."""
        if len(site) < 2 or floor <= 0.0:
            return set()
        at = fold_into_box(site + o, L)
        pairs = cKDTree(at, boxsize=L).query_pairs(floor, output_type="ndarray")
        if not len(pairs):
            return set()
        a, b = pairs[:, 0], pairs[:, 1]
        d_now = at[b] - at[a]
        d_now -= L * np.round(d_now / L)
        d_was = site[b] - site[a]
        d_was -= L * np.round(d_was / L)
        near = (np.linalg.norm(d_now, axis=1)
                < np.minimum(floor, np.linalg.norm(d_was, axis=1)))
        return set(a[near].tolist()) | set(b[near].tolist())

    def offenders(o):
        d1 = jittered(o)
        bad = ((np.linalg.norm(d1, axis=1) > cap * (1.0 + 1e-12))
               | np.any(np.round(d1 / L) != 0, axis=1))
        hit = set(iu[bad][iu[bad] >= 0].tolist()) | set(iv[bad][iv[bad] >= 0].tolist())
        return hit | crowded(o)

    held_back: set = set()
    for _ in range(int(rounds)):
        hit = offenders(offs)
        if not hit:
            break
        held_back |= hit
        offs[sorted(hit)] *= 0.5
    else:
        # The last resort, checked again: zeroing one junction can put a
        # neighbour's chord over the limit. All zero is the lattice itself,
        # which passes, so this ends.
        while True:
            hit = {k for k in offenders(offs) if np.any(offs[k] != 0.0)}
            if not hit:
                break
            held_back |= hit
            offs[sorted(hit)] = 0.0

    for n, k in row.items():
        P[n] = P[n] + offs[k]

    c1 = np.linalg.norm(jittered(offs), axis=1)
    size = np.linalg.norm(offs, axis=1)
    return {
        "fraction": float(fraction),
        "spacing": round(spacing, 6),
        "sigma": round(sd, 6),
        "junctions": len(nodes),
        "held_back": len(held_back),
        "offset_mean": round(float(size.mean()), 6) if len(size) else 0.0,
        "offset_max": round(float(size.max()), 6) if len(size) else 0.0,
        "reach_before": _reach_stats(c0 / contour, taut),
        "reach_after": _reach_stats(c1 / contour, taut),
        "coil_ratio_after": (round(float(contour.mean() / c1.mean()), 6)
                             if len(c1) else None),
    }


def _grow_loops(placed, pending, box, bond: float, floor: float, rng) -> None:
    """Grow the compact loops, in place, clear of the strands at their junction.

    ``pending`` holds ``(index, plan, anchor, hint)`` for every loop, whose
    slot in ``placed`` is empty. A loop's first bond leaves along the
    emptiest direction away from the first bonds of the strands at its
    junction (the hint, drawn where a ring would draw it, breaks the tie, as
    in :func:`~topon.conformation.paths.closed_meander`), and the walk keeps
    ``floor`` from the beads of those strands and of any loop grown at that
    junction before it, where it can
    (:func:`~topon.conformation.paths.closed_walk` with ``avoid``).

    Only those. A loop grown free of its junction's strands can curl round
    one of them, and the settle cannot part the two without passing one
    through the other: on SC 3^3 at DP 20 with the jitter (seed 6) it ran
    its 400 rounds against a loop bond jammed on a sibling bridge and put
    the build back. Strands from elsewhere are left to thread the loop as
    they would a ring of a melt chain. On the N100 fit the loops read Z1+
    0.81 per loop at build this way, 1.35 grown with nothing in view and
    0.27 kept clear of every strand (the relaxed reference's is 1.04).
    """
    L = np.asarray(box, float).reshape(3)
    beads_at: dict = {}          # junction -> beads of the strands there
    bonds_at: dict = {}          # junction -> their bonds leaving it
    for s in placed:
        if s is None:
            continue
        ends = [(s.plan.u, s.path[1] - s.path[0])]
        if s.plan.kind == "bridge":
            ends.append((s.plan.v, s.path[-2] - s.path[-1]))
        for node, leaving in ends:
            beads_at.setdefault(node, []).append(s.beads())
            bonds_at.setdefault(node, []).append(leaving)
    for i, plan, anchor, hint in pending:
        first = _emptiest(hint, bonds_at.get(plan.u) or None)
        here = beads_at.get(plan.u)
        room = Clearance(np.vstack(here), L, floor) if here else None
        ring = closed_walk(anchor, plan.n_bonds, bond, rng, min_sep=floor,
                           first=first, avoid=room)
        beads_at.setdefault(plan.u, []).append(ring)
        bonds_at.setdefault(plan.u, []).extend([ring[0] - anchor,
                                                ring[-1] - anchor])
        placed[i] = PlacedStrand(plan=plan,
                                 path=np.vstack([anchor, ring, anchor]),
                                 routine="closed_walk", draws=1, chord=0.0)


def _loop_stream(rng):
    """The stream ``compact`` loops grow on, spawned from the placement's.

    ``Generator.spawn`` derives a child from the generator's seed sequence
    and draws nothing from the generator, so the strands drawn after a loop
    see the same stream they would beside a ring. A generator whose bit
    generator has no seed sequence cannot spawn; the loops then grow on the
    placement's own stream, and the strands after the first loop move.
    """
    try:
        return rng.spawn(1)[0]
    except (AttributeError, TypeError, ValueError):
        return rng


def _parallel_strands(plans) -> dict:
    """Bridges that share both junctions with another, by plan index:
    :func:`~topon.conformation.paths.shared_chords` on the plans' bridges.

    Returns ``{index: (chord, rank, count)}``: ``chord`` the pair of
    junctions, ``rank`` the strand's place among the ``count`` strands on it
    in strand order. Each one's side comes from
    :func:`~topon.conformation.paths.chord_side`, which takes the one number
    the meander's turn would have taken, at the strand's own place in the
    strand order, so every other strand is drawn as before. Two exceptions
    shift the stream for what comes after: a side meander that falls back to
    a walk (the walk draws after this number, where the drawing before 0.4.5
    drew without it; not seen on any build), and the coincidence pass, which
    draws a direction for beads drawn exactly on top of each other, as
    before 0.4.5 the two strands of a loop of odd DP are at their middle beads
    and the strands drawn apart are not.
    """
    return shared_chords([(p.u, p.v) if p.kind == "bridge" else None
                          for p in plans])


def place(graph, dp: int, placement: str = "meander",
          coil_ratio: Optional[float] = None,
          build_density: Optional[float] = None,
          rng=None, *, bond: float = BOND, dims=None,
          waves: float = 6.0, jitter: float = 0.02,
          limits: Optional[GuardLimits] = None,
          coincident: float = 0.05,
          junction_shell_spacing: Optional[float] = None,
          junction_shell_blend: int = 4,
          junction_jitter: float = 0.0,
          settle_clearance: Optional[float] = None,
          parallel_strands: str = "opposite",
          loop_shape: str = "ring") -> Placement:
    """Draw every strand of ``graph`` at the build state.

    ``placement`` is one of ``straight`` (the chord with a jitter), ``meander``
    (:func:`~topon.conformation.paths.meander_chain`, which falls back to the
    chord when there is no slack to wave) or ``walk``
    (:func:`~topon.conformation.paths.bridging_walk`). Primary loops have no
    chord to interpolate along, so ``placement`` does not apply to them and
    ``loop_shape`` says how they are drawn: ``ring`` (the default) as the
    regular polygon of :func:`~topon.conformation.paths.closed_meander`, which
    at DP 100 is an open ring of radius 15.6 sigma, or ``compact`` as the
    closed self-avoiding walk of :func:`~topon.conformation.paths.closed_walk`,
    grown to ``max(route floor, LOOP_SEPARATION)`` (radius of gyration about
    4.3 sigma at DP 100). A compact loop takes the two draws the ring takes
    from ``rng``, where the ring takes them, for the direction of its first
    bond, and is grown after every bridge and dangling strand, clear of the
    strands at its own junction (:func:`_grow_loops`), on a stream spawned
    from ``rng`` (``Generator.spawn``). So every other strand of the build is
    drawn exactly as beside a ring, and a build with no loop is the same
    build under either shape. ``guard_report()["loops"]`` gives their size.
    Sol chains (``G.graph["sol_chains"]``) have no junction at all:
    each is a free walk from a random point of the cell, grown so no bead
    comes within the route's self-contact floor of another of its own
    (:func:`~topon.conformation.paths.free_walk` with ``min_sep``), and they
    are drawn after every other strand and kept last in ``strands``.

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

    ``junction_jitter`` moves every junction by a Gaussian offset per axis,
    that fraction of the site spacing (:func:`site_spacing`), before any
    strand is drawn, so that no three chords of a lattice cross at one point
    (:func:`_jitter_junctions`; the report is
    ``guard_report()["junction_jitter"]``). ``settle_clearance`` parts the
    bonds of different strands to that bond-to-bond distance once everything
    is drawn, holding the junctions and never passing one bond through
    another (:func:`~topon.conformation.placement.settle.settle_strands`;
    ``guard_report()["settle"]``). Both default off, and with both off the
    random stream is not touched, so a build is what it was without them.
    ``guard_report()["chord_triples"]`` counts the close chord triples the
    build carries either way.

    On the ``meander`` route the bridges that share both junctions (a
    secondary loop) are drawn on opposite sides of their chord
    (``parallel_strands="opposite"``, the default; :func:`_parallel_strands`,
    ``meander_chain(..., side=...)``); one meander turned twice about a
    chord meets itself wherever its wave crosses the chord.
    ``guard_report()["parallel"]`` says what was drawn so.
    ``parallel_strands="together"`` draws them as any other strand, the
    drawing before 0.4.5. A graph with no shared chord draws the same either
    way.

    Returns a :class:`Placement`. It is *not* checked for you -- read
    ``Placement.ok()`` or ``guard_report()`` and decide. The gate is advisory
    here on purpose: a caller sweeping build densities wants the failures
    reported, not raised.
    """
    if placement not in PLACEMENTS:
        raise ValueError(
            f"unknown placement {placement!r}; expected one of "
            f"{', '.join(PLACEMENTS)}")
    if parallel_strands not in PARALLEL_STRANDS:
        raise ValueError(
            f"unknown parallel_strands {parallel_strands!r}; expected one of "
            f"{', '.join(PARALLEL_STRANDS)}")
    if loop_shape not in LOOP_SHAPES:
        raise ValueError(
            f"unknown loop_shape {loop_shape!r}; expected one of "
            f"{', '.join(LOOP_SHAPES)}")
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

    # Off unless asked for, and then drawn before anything else, so a build
    # without it takes exactly the random stream it always took.
    jitter_report = {}
    if junction_jitter:
        if junction_jitter < 0:
            raise ValueError(
                f"junction_jitter must be at least 0; got {junction_jitter!r}")
        jitter_report = _jitter_junctions(graph, plans, P, box_sigma,
                                          float(junction_jitter), bond, rng)

    # Strands that share both junctions (secondary loops) are drawn on
    # opposite sides of their chord on the meander route; see
    # paths.meander_chain(side=...). A graph with none draws as before, and
    # so does "together", which leaves every strand to the usual draw.
    parallel = (_parallel_strands(plans)
                if placement == "meander" and parallel_strands == "opposite"
                else {})
    frames: dict = {}

    loop_floor = max(float(limits.min_sep), LOOP_SEPARATION)
    pending: list = []
    placed: list = []
    for k_plan, plan in enumerate(plans):
        if plan.kind == "free":
            continue                       # drawn below, after every other
        if plan.kind == "loop":
            anchor = P[plan.u]
            if loop_shape == "compact":
                # The ring's two draws (the hints of its direction and of its
                # plane), taken where the ring takes them; the walk is grown
                # below, once every strand it keeps clear of is drawn.
                hint = rng.normal(size=3)
                rng.normal(size=3)
                pending.append((len(placed), plan, anchor, hint))
                placed.append(None)
                continue
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
            side = None
            if (k_plan in parallel
                    and float(np.linalg.norm(c1 - c0))
                    < STRAIGHT_AT * (plan.n_bonds * bond)):
                # A chord the meander would draw straight keeps its draw,
                # which takes from the stream what it always took. The test
                # is meander_chain's own, on the same number.
                side = chord_side(d, parallel[k_plan][1], parallel[k_plan][2],
                                  frames, parallel[k_plan][0], rng)
            path, note = _draw(placement, c0, c1, plan.n_bonds, bond, rng,
                               waves, limits, jitter, side=side)
            if k_plan in parallel:
                note.setdefault("bow", 0.0)
        placed.append(PlacedStrand(plan=plan, path=np.asarray(path, float),
                                   routine=note["routine"],
                                   waves=float(note.get("waves", 0.0)),
                                   draws=int(note.get("draws", 1)),
                                   chord=chord, bow=note.get("bow")))

    if pending:
        _grow_loops(placed, pending, box_sigma, bond, loop_floor,
                    _loop_stream(rng))

    # The sol chains, last, so a build without them draws exactly what it
    # drew before. Each starts anywhere in the cell, as the pipeline drops
    # its own, and is grown self-avoiding to the route's own floor: a free
    # walk opened up afterwards misses it (paths.free_walk).
    for plan in plans:
        if plan.kind != "free":
            continue
        start = rng.random(3) * box_sigma
        path = free_walk(start, plan.n_bonds, bond, rng,
                         min_sep=limits.min_sep)
        placed.append(PlacedStrand(plan=plan, path=path,
                                   routine="free_walk", draws=1))

    junction_shells = {}
    if junction_shell_spacing:
        for st in placed:
            st.measure()
        junction_shells = _seat_on_shells(
            placed, float(junction_shell_spacing),
            int(junction_shell_blend), bond, limits)

    coincidence = (separate_coincident(placed, box_sigma, float(coincident),
                                       rng, bond=bond,
                                       min_bond=limits.min_bond)
                   if coincident else {})

    for s in placed:
        s.measure()

    pl = Placement(box=box_sigma, scale=scale,
                   build_density=float(build_density),
                   coil_ratio=float(realised_coil), placement=placement,
                   dp=int(dp), bond=float(bond), strands=placed,
                   n_beads=int(n_beads), limits=limits,
                   coincidence=coincidence,
                   junction_shells=junction_shells,
                   junction_jitter=jitter_report,
                   chord_triples=chord_triples(placed, box_sigma),
                   loop_shape=loop_shape)
    if settle_clearance:
        settle_placement(pl, float(settle_clearance), rng)
    else:
        pl.bead_bond = bead_bond_gaps(placed, box_sigma)
    return pl


def settle_placement(pl: Placement, clearance: float, rng=None,
                     **knobs) -> dict:
    """Settle a placement's strands in place and record what it did.

    :func:`~topon.conformation.placement.settle.settle_strands` with the
    placement's own bond, band and self-contact floor. ``place(...,
    settle_clearance=c)`` calls it last; a caller that moves beads after the
    placement (designed braids, :func:`~topon.conformation.entanglement.
    route_designed_pairs`) calls it after those instead, so the settle sees
    the coordinates that will be written. The report goes to
    ``pl.settle`` with the bead-to-bond reading from before it
    (``bead_bond_before``); ``pl.bead_bond`` is read again after.
    """
    before = bead_bond_gaps(pl.strands, pl.box)
    knobs.setdefault("tol", pl.limits.bond_tol)
    report, _frames = settle_strands(
        pl.strands, pl.box, float(clearance), rng, bond=pl.limits.max_bond,
        min_bond=pl.limits.min_bond, min_sep=pl.limits.min_sep, **knobs)
    report["bead_bond_before"] = before
    pl.settle = report
    pl.bead_bond = bead_bond_gaps(pl.strands, pl.box)
    return report


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

    # The junctions, once each: they are obstacles, not movers. A sol chain
    # has none, and every one of its beads moves.
    seen: dict = {}
    for s in placed:
        if s.plan.kind == "free":
            continue
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
        if s.plan.kind == "free":
            continue
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
        m = np.zeros(len(strand.path), bool)
        m[strand.own] = True
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
            st.path[st.own] = xyz[starts[k]:starts[k] + sizes[k]]
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


def bead_bond_gaps(placed, box, cutoff: float = 0.4) -> dict:
    """Closest a bead comes to a *bond* it is not part of.

    Every other reading in the gate is bead-to-bead: :func:`self_contact`
    within one strand, :func:`separate_coincident` across the build. None of
    them can see a bead lying on a bond. At a 0.97 design bond such a bead sits
    0.48 sigma from each endpoint, so it clears a 0.05 bead-to-bead floor
    comfortably and clears a 1.122 one too -- measured on the N20 build at the
    WCA floor, the cross-chain bead-to-bond minimum is still 0.0000 after
    every bead pair has been pushed 1.122 sigma apart.

    **Reported, never gated.** It is not known to cause anything. Tracing the
    persistent 1.3-1.4 sigma bonds that stop the push-off back to their placed
    builds, this count does not predict them: 757 beads inside 0.05 sigma on
    one build gave three, 18 on another gave three, and 205 on a third gave
    none. The push-off resolves nearly all of them. It is here so that a clean
    report cannot be read as saying no bead sits on a bond, which is a thing
    the gate otherwise has no way to know.

    ``min`` is the closest approach anywhere in the build and ``under_cutoff``
    counts *beads*, not bead-bond pairs: one bead wedged between two strands is
    one bead. Junctions are counted once each however many strands meet there,
    and are compared against every bond except those of the strands they
    terminate -- a junction wedged in a third strand's bond is the commonest
    form of this, so excluding junctions outright would hide exactly the case
    worth seeing.
    """
    from scipy.spatial import cKDTree

    if not placed:
        return {}
    L = np.asarray(box, float).reshape(3)

    # Beads each strand owns, then each junction once, under its node id.
    pts, owner = [], []
    for si, st in enumerate(placed):
        b = st.beads()
        pts.append(b)
        owner.append(np.full(len(b), si))
    nodes: dict = {}
    for st in placed:
        if st.plan.kind == "free":
            continue
        nodes.setdefault(st.plan.u, st.path[0])
        if st.plan.kind != "dangling":
            nodes.setdefault(st.plan.v, st.path[-1])
    node_ids = list(nodes)
    at = {n: -2 - k for k, n in enumerate(node_ids)}   # owner tag per junction
    if node_ids:
        pts.append(np.asarray([nodes[n] for n in node_ids], float))
        owner.append(np.asarray([at[n] for n in node_ids]))
    P = np.vstack(pts)
    owner = np.concatenate(owner)

    # Every bond, with the strand that owns it and the junction tags it ends
    # on. An end that is no junction (a free end, either end of a sol chain)
    # is tagged -1, which no owner carries: strands are 0 up and junctions -2
    # down. It used to be 0, the first strand's own tag, so that strand's
    # beads were never compared with a dangling strand's bonds.
    segs, seg_owner, ends_u, ends_v = [], [], [], []
    for si, st in enumerate(placed):
        p = st.path
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        n = len(p) - 1
        seg_owner.append(np.full(n, si))
        ends_u.append(np.full(n, at.get(st.plan.u, -1)))
        ends_v.append(np.full(n, at.get(st.plan.v, -1)))
    S = np.concatenate(segs, axis=0)
    seg_owner = np.concatenate(seg_owner)
    ends_u = np.concatenate(ends_u)
    ends_v = np.concatenate(ends_v)

    a = S[:, 0, :]
    d = S[:, 1, :] - a
    d -= L * np.round(d / L)
    mid = a + 0.5 * d
    ll = np.einsum("ij,ij->i", d, d)

    def wrap(X):
        Y = np.mod(X, L)
        return np.minimum(Y, L * (1.0 - 1e-12))

    reach = float(0.5 * np.sqrt(ll.max()) + max(cutoff, 1.0))
    near = cKDTree(wrap(mid), boxsize=L).query_ball_tree(
        cKDTree(wrap(P), boxsize=L), r=reach)

    best = np.full(len(P), np.inf)
    for k, cand in enumerate(near):
        if not cand:
            continue
        c = np.asarray(cand)
        keep = (owner[c] != seg_owner[k]) & (owner[c] != ends_u[k]) \
            & (owner[c] != ends_v[k])
        c = c[keep]
        if not len(c):
            continue
        q = P[c] - a[k]
        q -= L * np.round(q / L)
        t = (np.clip((q @ d[k]) / ll[k], 0.0, 1.0) if ll[k]
             else np.zeros(len(c)))
        g = np.linalg.norm(q - t[:, None] * d[k], axis=1)
        np.minimum.at(best, c, g)

    seen = best[np.isfinite(best)]
    if not len(seen):
        return {}
    return {"cutoff": float(cutoff),
            "min": round(float(seen.min()), 4),
            "under_cutoff": int((seen < cutoff).sum()),
            "beads": int(len(seen))}


def _seat_on_shells(placed, spacing: float, blend: int, bond: float,
                    limits: GuardLimits, rounds: int = 12) -> dict:
    """Spread the first beads of the strands that share a junction, in place.

    Loops are left out: both of a loop's ends are the same junction, and a
    shell seat that pulls its two ends apart opens the ring it was drawn to
    close. So are sol chains, which meet no junction.

    **A junction seats all of its chains or none of them.** The spread a shell
    delivers is the spread of every chain meeting at one node, so keeping the
    seats whose own strand happened to pass the gate and dropping the rest
    leaves the ones that moved sitting next to the ones that did not. That is
    worse than never seating: measured on SC 3 at DP 60 it took sibling first
    beads from 26 sub-sigma pairs to 45 and the worst separation from 0.887 to
    0.402. Declining by junction can only leave the build as it was, which is
    the weakest guarantee worth having and the one the per-strand version did
    not give. It is the same lesson as reverting a braid by pair rather than
    by strand (:mod:`topon.conformation.entanglement.designed`).

    Strands are gated, junctions are declined, and the two ends of one chain
    are independent -- a chain can keep a seat at ``u`` and lose one at ``v``,
    since each is blended from its own end. Dropping a junction changes the
    strands it touches, so it iterates to a fixed point; ``rounds`` caps it.

    The seat is then absorbed by the whole strand rather than by the four
    beads behind it. Without that nothing is ever seated at all: a strand drawn
    at the design bond has no sideways room, the blend stretches a bond to
    1.023-1.245 against a 0.971 ceiling, and widening the blend does not help
    (0 of 24 DP-20 strands survive at blend 2, 4, 8, 16 or 32) because the
    blend only looks as far as bead ``blend``. The room is there, further along
    the contour, and the same ``floor`` then ``only_long`` relaxation
    :func:`separate_coincident` finishes its own moved strands with reaches it.
    Beads 0, 1, -2 and -1 are held, so the junction and the seat it was given
    both stay exactly where they were put.

    Even so, most junctions decline. The seat is largest exactly where the
    shell is worth most -- pulling apart chains that leave a junction in nearly
    the same direction -- so the cases with most to gain are the ones least
    able to pay. Measured at spacing 1.0: 1 junction of 8 on SC 2 at DP 60, 4
    of 27 on SC 3 at DP 60, none at all at DP 20 or on SC 3 at DP 100. (The
    SC 2 numbers are from before 0.4.5, when both strands of each SC 2 chord
    were drawn on top of each other; on SC 3 at DP 20 all 27 junctions seat
    at seed 4.)
    """
    from topon.conformation.junction_shell import apply_junction_shells
    from topon.conformation.paths import _relax_bonds

    def settle(path, strand):
        """Let the contour absorb the seat, holding both junctions and seats."""
        m = np.zeros(len(path), bool)
        m[2:-2] = True
        if strand.plan.kind == "dangling":
            m[-1] = True
        if not m.any():
            return path
        _relax_bonds(path, bond, m, 200, 0.0, floor=limits.min_bond)
        _relax_bonds(path, bond, m, 200, 0.0, only_long=True)
        return path

    movable = [s for s in placed if s.plan.chorded]
    base = [s.path for s in movable]
    at: dict = {}
    for i, s in enumerate(movable):
        at.setdefault(s.plan.u, []).append(i)
        at.setdefault(s.plan.v, []).append(i)
    shared = {j for j, ks in at.items() if len(ks) > 1}
    report = {"spacing": float(spacing), "movable": len(movable),
              "junctions": len(shared), "seated_junctions": 0,
              "seated_strands": 0, "declined_junctions": len(shared)}
    if not movable or not shared:
        return report

    paths = {i: p for i, p in enumerate(base)}
    ends = {i: (s.plan.u, s.plan.v) for i, s in enumerate(movable)}

    def touched_by(active):
        return {i for j in active for i in at[j]}

    active = set(shared)
    for _ in range(max(1, int(rounds))):
        out = apply_junction_shells(paths, ends, spacing=spacing, blend=blend,
                                    bond=bond, only=active)
        drop = set()
        for i in touched_by(active):
            s = movable[i]
            was, s.path = s.path, settle(out[i], s)
            s.measure()
            if s.failures(limits):
                drop |= {s.plan.u, s.plan.v} & active
            s.path = was
            s.measure()
        if not drop:
            break
        active -= drop
        if not active:
            break

    # Only the strands of a surviving junction are rebuilt at all, so a
    # declined junction leaves its chains byte-identical rather than merely
    # unchanged to within the relaxation's tolerance.
    out = apply_junction_shells(paths, ends, spacing=spacing, blend=blend,
                                bond=bond, only=active)
    seated = 0
    for i in touched_by(active):
        s = movable[i]
        p = settle(out[i], s)
        if not np.allclose(p, base[i]):
            seated += 1
        s.path = p
        s.measure()
    report["seated_junctions"] = len(active)
    report["declined_junctions"] = len(shared) - len(active)
    report["seated_strands"] = seated
    return report
