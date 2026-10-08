"""Where the strands of a bead-spring build cross, and a pass that parts them.

Two things live here, both about the same defect. On a lattice with several
neighbour shells many junction-to-junction chords cross exactly: every pair of
sites symmetric about one point has its chord's midpoint there, so on the N20
MIX 90/5/5 four-shell graph up to five chords pass through one point. A
meander has a node at its middle when it carries a whole number of waves, so
the strands drawn along those chords pass through the crossing too, and three
strands that start on one point jam there in the push-off. Every pinch
cluster traced on the dilute DP-20 ``place()`` builds (5 of 5, the pinch
follow-up of 29 Sep 2026) sits on such a crossing: at the build the middle
bonds of the three strands of the gate-off run are 0.000 sigma apart, their
chords 0.000 at both midpoints, and the jam never resolves.

:func:`chord_triples` counts the places a build carries that risk: triples of
bridge chords pairwise within a distance of each other, the count the
follow-up measured (129 within 1 sigma on that build). It is a diagnostic in
:meth:`~topon.conformation.placement.chains.Placement.guard_report`.

:func:`settle_strands` is the coarse-grained counterpart of the atomistic
:func:`~topon.conformation.atomistic.settle_backbones`: it pushes bonds of
different strands apart to a bond-to-bond clearance, keeps every bond in the
placement band, keeps each drawn strand clear of itself, holds the junctions,
and puts back any move that takes one bond through another, read with the
segment geometry the crossing detector uses
(:func:`~topon.conformation.segments.passage_times`, shortest image).
Bead-to-bead floors cannot see this defect: two bonds can cross with their
four beads 1.2 sigma apart, and a bead can sit on a bond at 0.48 sigma from
both its ends.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from topon.conformation.segments import bond_ends, passage_times, segment_closest

__all__ = [
    "CHORD_RADII",
    "TOUCH",
    "chord_triples",
    "moved_passing_pairs",
    "settle_strands",
]

#: The chord distances (sigma) :func:`chord_triples` counts within.
CHORD_RADII = (0.5, 1.0, 1.5)

#: Two bonds closer than this (sigma) are touching: they have no side to keep
#: and are parted first, on the side one random stacking of the strands gives.
TOUCH = 0.01

#: How far past the clearance a short pair is pushed, as a fraction of it.
OVERSHOOT = 0.1

#: A pair's push grows as ``clearance / d`` as it closes, floored at this
#: fraction of the clearance.
BARRIER = 0.02

#: The least weight (of 1) with which a bond's closest point moves with the
#: stretch pushed about it for that side to take part in opening the pair.
MOVABLE = 0.2


# ---------------------------------------------------------------------------
# The diagnostic: how many places do three chords meet
# ---------------------------------------------------------------------------

def _fold(x, box):
    y = x - box * np.floor(x / box)
    return np.clip(y, 0.0, np.nextafter(box, 0.0))


def chord_triples(placed, box, radii=CHORD_RADII, samples: int = 9) -> dict:
    """Close pairs and triples of bridge chords, within each of ``radii``.

    A chord is the straight line between a bridge's two junctions as the
    strand was drawn (after any junction jitter). Two chords that end on a
    common junction meet there by construction and are not a pair. A triple
    is three chords pairwise within the radius, the definition the pinch
    follow-up counted with (``pinch_chords.py``); it does not ask that the
    three meet at one point, so it is an upper bound on the places three
    strands can start crowded together. Dangling strands are left out: a free
    end can back out of a jam, and every pinch traced sat on three bridges.

    Returns ``{"radii", "pairs", "triples", "bridges"}``, one count per
    radius, or ``{}`` when the build has fewer than two bridges.
    """
    L = np.asarray(box, float).reshape(3)
    radii = tuple(float(r) for r in radii)
    bridges = [s for s in placed if s.plan.kind == "bridge"]
    if len(bridges) < 2 or not radii:
        return {}
    a = np.array([s.path[0] for s in bridges], float)
    b = np.array([s.path[-1] for s in bridges], float)
    ends = np.array([(s.plan.u, s.plan.v) for s in bridges], dtype=object)
    f = np.linspace(0.0, 1.0, int(samples))
    pts = (a[:, None, :] + f[None, :, None] * (b - a)[:, None, :]).reshape(-1, 3)
    owner = np.repeat(np.arange(len(bridges)), len(f))
    step = float(np.linalg.norm(b - a, axis=1).max()) / max(len(f) - 1, 1)
    rmax = max(radii)
    cand = cKDTree(_fold(pts, L), boxsize=L).query_pairs(
        rmax + step, output_type="ndarray")
    if not len(cand):
        return {"radii": list(radii), "pairs": [0] * len(radii),
                "triples": [0] * len(radii), "bridges": len(bridges)}
    i, j = owner[cand[:, 0]], owner[cand[:, 1]]
    keep = i != j
    i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    key = np.unique(np.stack([i, j], axis=1), axis=0)
    i, j = key[:, 0], key[:, 1]
    shared = np.array([bool({ends[x][0], ends[x][1]} & {ends[y][0], ends[y][1]})
                       for x, y in zip(i, j)], bool)
    i, j = i[~shared], j[~shared]
    mid = 0.5 * (a + b)
    sh = -L * np.round((mid[j] - mid[i]) / L)
    _s, _t, d = segment_closest(a[i], b[i], a[j] + sh, b[j] + sh)

    n_pairs, n_tri = [], []
    for r in radii:
        close = d < r
        n_pairs.append(int(close.sum()))
        nbr: dict = {}
        for x, y in zip(i[close], j[close]):
            nbr.setdefault(int(x), set()).add(int(y))
            nbr.setdefault(int(y), set()).add(int(x))
        tri = 0
        for x, y in zip(i[close], j[close]):
            x, y = int(x), int(y)
            tri += sum(1 for z in nbr[x] & nbr[y] if z > y)
        n_tri.append(int(tri))
    return {"radii": list(radii), "pairs": n_pairs, "triples": n_tri,
            "bridges": len(bridges)}


# ---------------------------------------------------------------------------
# The settle
# ---------------------------------------------------------------------------

def moved_passing_pairs(before, after, rows, atom_ids, box, moved):
    """:func:`~topon.conformation.segments.passing_pairs`, for a move that
    leaves most bonds where they were.

    A bond that did not move can only be passed by one that did, so the
    candidates are the pairs with at least one bond in ``moved`` (a boolean
    per bond), found from the moved bonds outward, and the passage test is
    the same one (:func:`~topon.conformation.segments.passage_times`, each
    pair in the image nearest the first bond). On a 95,000-bead build most
    rounds of the settle move a few thousand beads. Returns ``(i, j, tau)``.
    """
    box = np.asarray(box, float).reshape(3)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    m_idx = np.flatnonzero(moved)
    if not len(m_idx):
        return empty
    pa, pb = bond_ends(before, rows, box)
    fa, fb = bond_ends(after, rows, box)
    disp = np.linalg.norm(after - before, axis=1)
    big = float(disp.max()) if len(disp) else 0.0
    mid = 0.5 * (pa + pb)
    half = 0.5 * np.linalg.norm(pb - pa, axis=1)
    reach = 2.0 * float(half.max()) + 2.0 * big + 0.5
    w = _fold(mid, box)
    near = cKDTree(w[m_idx], boxsize=box).query_ball_tree(
        cKDTree(w, boxsize=box), reach)
    counts = np.fromiter((len(x) for x in near), int, len(near))
    if not counts.sum():
        return empty
    i = np.repeat(m_idx, counts)
    j = np.concatenate([np.asarray(x, int) for x in near if len(x)])
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = np.unique(np.stack([lo, hi], axis=1)[lo != hi], axis=0)
    if not len(key):
        return empty
    i, j = key[:, 0], key[:, 1]
    ids = np.asarray(atom_ids)
    share = ((ids[i, 0] == ids[j, 0]) | (ids[i, 0] == ids[j, 1])
             | (ids[i, 1] == ids[j, 0]) | (ids[i, 1] == ids[j, 1]))
    i, j = i[~share], j[~share]
    shift = -box * np.round((mid[j] - mid[i]) / box)
    A = np.stack([pa[i], pb[i], pa[j] + shift, pb[j] + shift])
    B = np.stack([fa[i], fb[i], fa[j] + shift, fb[j] + shift])
    tau = passage_times(A, B)
    hit = ~np.isnan(tau)
    return i[hit], j[hit], tau[hit]


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


def settle_strands(placed, box, clearance: float, rng=None, *,
                   bond: float = 0.97, min_bond: float = 0.85,
                   min_sep: float = 1.0, near_junction: int = 2,
                   rounds: int = 400, step: float = 0.1, sweeps: int = 20,
                   width: float = 2.0, give: float = 0.03,
                   tol: float = 1e-3, keep_frames: bool = False):
    """Part the bonds of different strands to ``clearance``, crossing nothing.

    ``placed`` is the list of :class:`~topon.conformation.placement.chains.
    PlacedStrand` of a placement; their paths are changed in place (and
    re-measured). The junctions are held: a junction's position is shared by
    every strand that meets there. Every other bead moves, a dangling strand's
    free end included.

    What is held apart: every pair of bonds of *different* strands, except a
    pair sharing a bead (two strands' first bonds at their junction) and a
    pair whose bonds both sit within ``near_junction`` bonds of a junction the
    two strands share. That exception is geometry, not leniency: two strands
    leaving one junction at an angle ``theta`` put the second bond of one
    ``0.97 sin(theta)`` from the first bond of the other, under a sigma at any
    angle, and a junction of functionality 12 cannot spread its first beads
    past about 63 degrees. The crowding at a junction is
    :mod:`~topon.conformation.junction_shell`'s business; the pinch is where
    strands cross away from their junctions.

    Each round does what the atomistic pass does, read on beads. Every pair
    short of ``clearance`` is pushed apart along the line between the two
    bonds' closest points, a little past the clearance, half each (all of it
    on a side whose beads are held), the push moving a stretch of each strand
    about the closest point, Gaussian along the chain with ``width`` beads.
    Unlike the atomistic pass, a pair's push is weighted by ``clearance / d``
    (:data:`BARRIER` floors ``d``), so where one bead is pushed by several
    pairs the tightest decides which way it goes: at a point where four
    strands cross, even pushes open some pairs by squeezing another shut, and
    the pass then stalls with bonds dragged long (measured on a lattice of
    cube diagonals, 2 of 3 seeds unconverged at 300 rounds with bonds to 4.3
    sigma; weighted, all converge). Then every bond is pulled back to the
    length it was drawn at, give or take ``give`` (0.03 sigma) and inside the
    placement band ``[min_bond, bond]`` where its chord allows, and two beads
    of one strand at least two places apart are held ``min_sep`` apart (the
    placement's self-contact gate: 1.0 for a drawn path, 0.05 for a walk). A
    bond inside its window and a pair past the floor add nothing, so a strand
    no push reaches is left exactly as drawn. No bead moves more than ``step``
    in a round.

    Last, the round is read as a move from where every bead was to where it
    is (:func:`moved_passing_pairs`: every bond that moved against every bond
    near it, one strand's own included, in its shortest image, with the
    passage test of :mod:`~topon.conformation.segments`), and the beads of
    any two bonds that passed through each other are put back, until none
    did. Two bonds drawn exactly touching (closer than :data:`TOUCH`) have no
    side to keep; they are parted first by a hair along the normal to both
    (no bead more than ``step``, and any other pair that passes on the way
    put back), and the record starts after that. The side comes from one random
    stacking of the strands (an order, and an axis to stack along), not from
    a coin per pair: where three strands touch at one point a coin per pair
    can weave them, each over the next, and only a detour around each other
    opens a weave. It is the only choice the pass makes, and a drawn path
    that touches has made no choice to keep.

    Rounds stop when no pair is short, every bond whose strand can carry it
    is in the band (to ``tol`` relative on the long side, as the gate reads
    it) and no strand that met its self-contact floor as drawn is inside it.
    A round that moves nothing ends the pass (``stopped``), since the next
    would be the same round. Whatever the pass ends on is kept only if every
    strand that passed the placement gate as drawn still does, and otherwise
    the whole build goes back to what was drawn (``kept`` false,
    ``gate_broken`` the strands it would have failed; the ``_after`` numbers
    then describe the attempt). Measured on the lattice of cube diagonals, where every one of
    27 crossings holds four strands through one point, 7 of 8 seeds converge
    unjittered and 8 of 8 with ``junction_jitter`` 0.15. On the N20
    reference graph at coil 1.51 and 1.402 it converges in 44-144 rounds
    with jitter from 0.05 to 0.15, and not without jitter or at 0.02 (up to
    five strands through one point), where the build is put back as
    drawn.

    Returns ``(report, frames)``; ``frames`` is ``None`` unless
    ``keep_frames``, and is then ``(ids, [xyz, ...])``: one row per bead and
    junction (junction ids are ``-1 - k``), before the first round and after
    each, for reading the pass back through
    :mod:`topon.analysis.crossings`.
    """
    rng = np.random.default_rng() if rng is None else rng
    L = np.asarray(box, float).reshape(3)
    clearance = float(clearance)
    if not placed or clearance <= 0.0:
        return {}, None

    # ---- rows: every point of every path, junctions once per strand ----
    sizes = [len(s.path) for s in placed]
    starts = np.cumsum([0] + sizes[:-1])
    X = np.vstack([np.asarray(s.path, float) for s in placed])
    n_rows = len(X)
    held = np.zeros(n_rows, bool)
    jlabel: dict = {}
    ids = np.empty(n_rows, np.int64)
    next_bead = 0
    for k, (s, s0, m) in enumerate(zip(placed, starts, sizes)):
        # Held: the junctions a strand ends on. A dangling strand's far end
        # is its own bead, and a sol chain has no junction at all.
        ends = [] if s.plan.kind == "free" else [(0, s.plan.u)]
        if s.plan.kind in ("bridge", "loop"):
            ends.append((m - 1, s.plan.v))
        for r in range(m):
            ids[s0 + r] = next_bead
            next_bead += 1
        for r, node in ends:
            held[s0 + r] = True
            ids[s0 + r] = -1 - jlabel.setdefault(node, len(jlabel))
    mobile = (~held).astype(float)

    sa = np.concatenate([np.arange(s0, s0 + m - 1) for s0, m in zip(starts, sizes)])
    sb = sa + 1
    chain = np.concatenate([np.full(m - 1, k) for k, m in enumerate(sizes)])
    along = np.concatenate([np.arange(m - 1) for m in sizes])
    nseg = np.concatenate([np.full(m - 1, m - 1) for m in sizes])
    ju = np.concatenate([np.full(m - 1, ids[s0]) for s0, m in zip(starts, sizes)])
    jv = np.concatenate([np.full(m - 1, ids[s0 + m - 1] if held[s0 + m - 1] else 0)
                         for s0, m in zip(starts, sizes)])
    du, dv = along, nseg - 1 - along
    rows = np.stack([sa, sb], axis=1)
    pair_ids = np.stack([ids[sa], ids[sb]], axis=1)
    row_chain = np.concatenate([np.full(m, k) for k, m in enumerate(sizes)])
    row_along = np.concatenate([np.arange(m) for m in sizes])
    row_lo = np.concatenate([np.full(m, s0) for s0, m in zip(starts, sizes)])
    row_hi = row_lo + np.concatenate([np.full(m, m - 1) for m in sizes])
    ring = np.array([s.plan.kind == "loop" for s in placed])
    ring_n = np.array(sizes) - 1               # distinct points on a ring
    reach = max(1, int(np.ceil(2.5 * width)))
    near = int(near_junction)
    lo_band = float(min_bond)
    hi_band = float(bond)
    sep = float(min_sep)
    step = float(step)

    # Bond targets: a window of ``give`` either side of the length drawn,
    # inside the band (a hair inside its lower edge, which the gate reads
    # strictly). A bond inside its window is left alone, so a strand no push
    # reaches keeps its bonds exactly, and a pushed one can lengthen a
    # little to bend -- a strand drawn straight has no other slack -- but
    # never drifts to an edge of the band. A strand whose chord is longer
    # than its contour cannot be in the band at all; its bonds are held at
    # what was drawn.
    r_drawn = np.linalg.norm(X[sb] - X[sa], axis=1)
    fits = np.array([s.plan.kind == "loop"
                     or float(np.linalg.norm(s.path[-1] - s.path[0]))
                     <= s.plan.n_bonds * hi_band for s in placed])
    banded = fits[chain]
    lo_edge = lo_band * (1.0 + tol)
    r_lo = np.where(banded, np.clip(r_drawn - give, lo_edge, hi_band), r_drawn)
    r_hi = np.where(banded, np.clip(r_drawn + give, lo_edge, hi_band), r_drawn)
    # The self floor is pushed for on every strand; the pass is done when
    # the strands that met it as drawn still do.
    own_ok = np.array([s.self_contact >= sep * (1.0 - 1e-6) for s in placed])

    def half_bond(P):
        return 0.5 * float(np.linalg.norm(P[sb] - P[sa], axis=1).max())

    def survey(P, radius):
        """Pairs of bonds of different strands closer than ``radius``."""
        p1, p2 = P[sa], P[sb]
        mid = 0.5 * (p1 + p2)
        pairs = cKDTree(_fold(mid, L), boxsize=L).query_pairs(
            radius + 2.0 * half_bond(P), output_type="ndarray")
        empty = (np.zeros(0, int),) * 2 + (np.zeros((0, 3)),) + (np.zeros(0),) * 3
        if not len(pairs):
            return empty
        i, j = pairs[:, 0], pairs[:, 1]
        keep = chain[i] != chain[j]
        a0, a1, b0, b1 = ids[sa[i]], ids[sb[i]], ids[sa[j]], ids[sb[j]]
        keep &= ~((a0 == b0) | (a0 == b1) | (a1 == b0) | (a1 == b1))
        if near > 0:
            for Ji, di in ((ju[i], du[i]), (jv[i], dv[i])):
                for Jj, dj in ((ju[j], du[j]), (jv[j], dv[j])):
                    keep &= ~((Ji == Jj) & (Ji < 0) & (np.maximum(di, dj) < near))
        i, j = i[keep], j[keep]
        if not len(i):
            return empty
        sh = -L * np.round((mid[j] - mid[i]) / L)
        s, t, d = segment_closest(p1[i], p2[i], p1[j] + sh, p2[j] + sh)
        short = d < radius
        return i[short], j[short], sh[short], s[short], t[short], d[short]

    def self_pairs(P, only_ok):
        """Beads of one strand, two or more places apart, inside ``sep``."""
        if sep <= 0.0:
            return np.zeros((0, 2), int)
        pairs = cKDTree(_fold(P, L), boxsize=L).query_pairs(
            sep, output_type="ndarray")
        if not len(pairs):
            return np.zeros((0, 2), int)
        i, j = pairs[:, 0], pairs[:, 1]
        same = (row_chain[i] == row_chain[j]) & (ids[i] != ids[j])
        if only_ok:
            same &= own_ok[row_chain[i]]
        i, j = i[same], j[same]
        gap = np.abs(row_along[i] - row_along[j])
        k = row_chain[i]
        gap = np.where(ring[k], np.minimum(gap % ring_n[k],
                                           ring_n[k] - gap % ring_n[k]), gap)
        keep = gap >= 2
        pij = np.stack([i[keep], j[keep]], axis=1)
        if not len(pij):
            return pij
        # a strand is in one image, so the pair vector is direct
        d = np.linalg.norm(P[pij[:, 1]] - P[pij[:, 0]], axis=1)
        return pij[d < sep * (1.0 - 1e-6)]

    def push(P, found, target, weighted=True, stacked=None):
        """Moves opening every pair in ``found`` to ``target``."""
        i, j, sh, s, t, d = found
        move = np.zeros_like(P)
        if not len(i):
            return move
        p1, p2 = P[sa[i]], P[sb[i]]
        q1, q2 = P[sa[j]] + sh, P[sb[j]] + sh
        n = (p1 + s[:, None] * (p2 - p1)) - (q1 + t[:, None] * (q2 - q1))
        # A touching pair has no side of its own. It takes one from a single
        # random stacking of the strands (an order, and a direction to stack
        # along), so that where several strands touch at one point they come
        # apart as a stack and never as a weave: a random side per pair makes
        # three strands each over the next, which only a detour around each
        # other can open.
        flat = d < 1e-9 if stacked is None else (stacked | (d < 1e-9))
        if flat.any():
            c = np.cross(p2[flat] - p1[flat], q2[flat] - q1[flat])
            for k in np.flatnonzero(np.linalg.norm(c, axis=1) < 1e-9):
                c[k] = _perpendicular(p2[flat][k] - p1[flat][k], rng)
            c = c / np.linalg.norm(c, axis=1)[:, None]
            up = (stack_rank[chain[i[flat]]] - stack_rank[chain[j[flat]]]) \
                * (c @ stack_axis)
            side = np.where(np.abs(up) > 1e-6, np.sign(up),
                            rng.choice([-1.0, 1.0], size=len(c)))
            n[flat] = c * side[:, None]
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
        wi = in_i * mobile[rows_i] * np.exp(
            -0.5 * ((offs[None, :] - s[:, None]) / width) ** 2)
        wj = in_j * mobile[rows_j] * np.exp(
            -0.5 * ((offs[None, :] - t[:, None]) / width) ** 2)
        # how far each closest point moves per unit of weight
        di = (1 - s) * wi[:, reach] + s * wi[:, reach + 1]
        dj = (1 - t) * wj[:, reach] + t * wj[:, reach + 1]
        # Weighted by how far inside the clearance a pair is, so that where
        # one bead is pushed by several pairs the tightest one decides which
        # way it goes. Summed evenly, the pushes of a crossing of four
        # strands squeeze some pair shut while opening the others.
        gap = target - d
        if weighted:
            gap = gap * clearance / np.maximum(d, BARRIER * clearance)
        # A side whose closest point hardly moves with its stretch (a bond
        # whose closest point is on a held junction) leaves the pair to the
        # other side: asking it for half the gap would move the rest of its
        # stretch by the gap over that small weight.
        mi, mj = di > MOVABLE, dj > MOVABLE
        fi = np.where(mj, np.where(mi, 0.5, 0.0), 1.0)
        fj = np.where(mi, 1.0 - fi, np.where(mj, 1.0, 0.0))
        gi = np.where(mi, gap * fi / np.maximum(di, MOVABLE), 0.0)
        gj = np.where(mj, gap * fj / np.maximum(dj, MOVABLE), 0.0)
        np.add.at(move, rows_i.ravel(),
                  ((wi * gi[:, None])[:, :, None] * n[:, None, :]).reshape(-1, 3))
        np.add.at(move, rows_j.ravel(),
                  (-(wj * gj[:, None])[:, :, None] * n[:, None, :]).reshape(-1, 3))
        return move * mobile[:, None]

    def relax(P, own):
        """Bonds back to their target lengths, each strand clear of itself."""
        for _ in range(int(sweeps)):
            corr = np.zeros_like(P)
            count = np.zeros(len(P))
            v = P[sb] - P[sa]
            length = np.linalg.norm(v, axis=1)
            err = length - np.clip(length, r_lo, r_hi)
            act = err != 0.0
            groups = [(sa[act], sb[act], v[act], length[act], err[act])]
            if len(own):
                w = P[own[:, 1]] - P[own[:, 0]]
                dl = np.linalg.norm(w, axis=1)
                e = np.minimum(dl - sep * (1.0 + 1e-4), 0.0)
                on = e < 0.0
                groups.append((own[on, 0], own[on, 1], w[on], dl[on], e[on]))
            for a, b, vec, ln, er in groups:
                if not len(a):
                    continue
                wa, wb = mobile[a], mobile[b]
                tot = wa + wb
                ok = tot > 0
                u = (np.where(ok, er / np.maximum(ln, 1e-9), 0.0)[:, None]
                     * vec / np.where(ok, tot, 1.0)[:, None])
                np.add.at(corr, a, wa[:, None] * u)
                np.add.at(corr, b, -wb[:, None] * u)
                np.add.at(count, a, wa)
                np.add.at(count, b, wb)
            if not count.any():
                break
            P = P + corr / np.maximum(count, 1.0)[:, None]
        return P

    def in_band(P):
        """Every bond whose strand can carry it is inside the gate's band."""
        length = np.linalg.norm(P[sb] - P[sa], axis=1)[banded]
        return bool((length <= hi_band * (1.0 + tol)).all()
                    and (length >= lo_band).all())

    def closest(P):
        """The smallest bond-to-bond distance the pass holds apart."""
        found = survey(P, max(2.0 * clearance, 1.0))
        return round(float(found[5].min()), 6) if len(found[5]) else None

    X_drawn = X.copy()
    before = survey(X, clearance)
    n_before = len(before[0])
    closest_before = closest(X)
    def put_back_passages(old, new, allowed=frozenset()):
        """Beads of any two bonds that passed through each other, put back.

        ``allowed`` pairs (bond indices, low first) may pass: a touching pair
        has no side to keep. Returns the positions and the rows put back, or
        ``None`` when 50 tries did not clear them.
        """
        count = 0
        for _ in range(50):
            moved = (np.any(new[sa] != old[sa], axis=1)
                     | np.any(new[sb] != old[sb], axis=1))
            pi, pj, _tau = moved_passing_pairs(old, new, rows, pair_ids, L, moved)
            if allowed and len(pi):
                ok = np.array([(min(a, b), max(a, b)) in allowed
                               for a, b in zip(pi.tolist(), pj.tolist())], bool)
                pi, pj = pi[~ok], pj[~ok]
            if not len(pi):
                return new, count
            back = np.unique(np.concatenate([sa[pi], sb[pi], sa[pj], sb[pj]]))
            new[back] = old[back]
            count += len(back)
        return None, count

    # Two bonds drawn touching have no side, and every way of parting them
    # reads as one passing through the other. They are parted first, by a
    # hair and on the side one random stacking of the strands gives, and the
    # record starts after that. Where several strands meet at one point one
    # parting can leave another pair touching, so it is repeated (a few
    # times at most). Only the pairs drawn touching may come apart either
    # way: a pair the parting brings to touch had a side as drawn and keeps
    # it (it is pushed along its own closest-point line), and a parting that
    # carries any bond through one it was not drawn touching is put back like
    # any other move.
    stack_axis = rng.normal(size=3)
    stack_axis /= np.linalg.norm(stack_axis)
    stack_rank = rng.permutation(len(placed)).astype(float)
    drawn_touching = frozenset(
        (min(a, b), max(a, b)) for a, b, d in
        zip(before[0].tolist(), before[1].tolist(), before[5].tolist())
        if d < TOUCH)
    touching_parted = 0
    prepass_put_back = 0
    touch = before
    for _ in range(5):
        hit = touch[5] < TOUCH
        if not hit.any():
            break
        touching_parted += int(hit.sum())
        found = tuple(x[hit] for x in touch)
        stacked = np.array([(min(a, b), max(a, b)) in drawn_touching
                            for a, b in zip(found[0].tolist(),
                                            found[1].tolist())], bool)
        move = push(X, found, np.full(int(hit.sum()), 5.0 * TOUCH),
                    weighted=False, stacked=stacked)
        size = np.linalg.norm(move, axis=1)
        move *= np.minimum(1.0, step / np.maximum(size, 1e-12))[:, None]
        new, n_back = put_back_passages(X, X + move, drawn_touching)
        prepass_put_back += n_back
        if new is None:
            break
        X = new
        touch = survey(X, TOUCH)
    first_row: dict = {}
    for r, a in enumerate(ids):
        first_row.setdefault(int(a), r)
    rows_out = np.array(sorted(first_row.values()), int)
    frames = [X[rows_out].copy()] if keep_frames else None
    put_back = 0
    reverted = 0
    stopped = "rounds"
    done = 0
    for done in range(1, int(rounds) + 1):
        found = survey(X, clearance)
        own = self_pairs(X, only_ok=False)
        if (not len(found[0]) and not len(self_pairs(X, only_ok=True))
                and in_band(X)):
            done -= 1
            stopped = "converged"
            break
        # A little past the clearance, since the bonds take some of every
        # push back and a pair aimed exactly at it only creeps up to it. The
        # push is capped at ``step`` a bead before the bonds are pulled back,
        # so the bonds answer a small move rather than a weighted one many
        # sigma long, and the whole round is capped at ``step`` again after.
        move = push(X, found, np.full(len(found[0]),
                                      (1.0 + OVERSHOOT) * clearance))
        size = np.linalg.norm(move, axis=1)
        move *= np.minimum(1.0, step / np.maximum(size, 1e-12))[:, None]
        new = relax(X + move, own)
        shift = new - X
        size = np.linalg.norm(shift, axis=1)
        new = X + shift * np.minimum(1.0, step / np.maximum(size, 1e-12))[:, None]
        # A bead pushed by two pairs at once, or pulled by its bonds, moves
        # along no pair's line and can take a bond through a third close by.
        # Whatever does is put back where it was.
        new, n_back = put_back_passages(X, new)
        put_back += n_back
        if new is None:
            new = X
            reverted += 1
        if not np.any(new != X):
            # Nothing moved, so the next round would be this one again.
            done -= 1
            stopped = "no move"
            break
        X = new
        if keep_frames:
            frames.append(X[rows_out].copy())

    after = survey(X, clearance)
    n_after = len(after[0])
    own_after = self_pairs(X, only_ok=True)
    lengths = np.linalg.norm(X[sb] - X[sa], axis=1)
    shift = np.linalg.norm(X - X_drawn, axis=1)
    moved_strands = int(sum(1 for s0, m in zip(starts, sizes)
                            if np.any(X[s0:s0 + m] != X_drawn[s0:s0 + m])))

    # A settle that did not converge is kept only if it leaves every strand
    # that passed the placement gate as drawn still passing it. Otherwise
    # the whole build goes back to what was drawn: putting back single
    # strands could take one through a neighbour that moved, and a pass
    # asked to make the build safer may not hand on one that is worse.
    def gate_ok(s):
        return (s.bond_max <= hi_band * (1.0 + tol) and s.bond_min >= lo_band
                and s.self_contact >= sep * (1.0 - 1e-6))

    passed = [gate_ok(s) for s in placed]
    for s, s0, m in zip(placed, starts, sizes):
        if np.any(X[s0:s0 + m] != np.asarray(s.path)):
            s.path = X[s0:s0 + m].copy()
            s.measure()
    converged = bool(not n_after and not len(own_after) and in_band(X))
    broke = sum(1 for s, p in zip(placed, passed) if p and not gate_ok(s))
    kept = not broke
    if not kept:
        for s, s0, m in zip(placed, starts, sizes):
            if np.any(X_drawn[s0:s0 + m] != np.asarray(s.path)):
                s.path = X_drawn[s0:s0 + m].copy()
                s.measure()

    report = {
        "clearance": clearance,
        "near_junction": near,
        "pairs_below_before": int(n_before),
        "closest_before": closest_before,
        "touching_parted": int(touching_parted),
        "pairs_below_after": int(n_after),
        "closest_after": closest(X),
        "self_pairs_below_after": int(len(own_after)),
        "bond_min": round(float(lengths.min()), 6),
        "bond_max": round(float(lengths.max()), 6),
        "rounds": int(done),
        "stopped": stopped,
        "converged": converged,
        "kept": bool(kept),
        "gate_broken": int(broke),
        "strands_moved": moved_strands if kept else 0,
        "largest_shift": round(float(shift.max()), 6),
        "mean_shift_moved": (round(float(shift[shift > 0].mean()), 6)
                             if (shift > 0).any() else 0.0),
        "moves_put_back": int(put_back),
        "touching_moves_put_back": int(prepass_put_back),
        "rounds_reverted": int(reverted),
    }
    out_frames = (ids[rows_out], frames) if keep_frames else None
    return report, out_frames
