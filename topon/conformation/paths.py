"""Chain paths between two fixed endpoints.

Coordinate generation, so it belongs to the conformation stage. Kept apart
from ``manager.py`` because that stage reads and rewrites a LAMMPS data file,
while these are plain geometry: give them two junctions and a bead count and
they hand back a path.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["Clearance", "bridging_walk", "closed_meander", "free_walk",
           "straight", "walk_via", "walk_through", "loop_around", "zigzag",
           "taut_leg", "route_through",
           "straight_chain", "meander_chain", "unfold",
           "bond_lengths", "self_contact", "fold_into_box"]


class Clearance:
    """The beads already in the box, so a new path can be drawn around them.

    A path drawn without regard to what is already there lands beads on top of
    beads. Measured on a relaxed melt at density 0.85, routing a single chain
    took the closest pair in the system from 0.502 sigma to 0.195 and put 153
    beads inside 0.5 sigma where there had been none.

    That is not a small mistake to leave to the minimiser. At 0.195 sigma the
    WCA energy is of order 1e5 kT, so the next minimisation does not relax the
    contact, it shoves -- and the shove is large enough to drag chains through
    each other, which rewrites the topology that was just designed. It shows
    up as designed entanglements being lost and undesigned ones appearing, and
    no amount of care in choosing the winding survives it. The cure is to not
    make the overlap in the first place.

    Distances are minimum-image when a box is given.
    """

    def __init__(self, points, box=None, radius: float = 0.9):
        self.radius = float(radius)
        self.box = None if box is None else np.asarray(box, float).reshape(3)
        pts = np.asarray(points, float).reshape(-1, 3)
        self.tree = None if len(pts) == 0 else cKDTree(
            self._wrap(pts), boxsize=self.box)

    def _wrap(self, p):
        p = np.asarray(p, float).reshape(-1, 3)
        if self.box is None:
            return p
        p = p - self.box * np.floor(p / self.box)
        # cKDTree's periodic box is half-open, and it rejects the whole tree
        # over a single point sitting exactly on the upper face.
        return np.clip(p, 0.0, np.nextafter(self.box, 0.0))

    def near(self, pts) -> np.ndarray:
        """Distance from each point to the nearest bead already there."""
        q = self._wrap(pts)
        if self.tree is None:
            return np.full(len(q), np.inf)
        return np.atleast_1d(self.tree.query(q, k=1)[0])

    def worst(self, pts) -> float:
        """The tightest contact a path makes. Larger is better."""
        return float(self.near(pts).min())

    def ok(self, pts) -> bool:
        return self.worst(pts) >= self.radius


def straight(start, end, n_beads: int) -> np.ndarray:
    """A straight line, ``n_beads`` points including both ends."""
    t = np.linspace(0.0, 1.0, n_beads)[:, None]
    a = np.asarray(start, float)
    return a + t * (np.asarray(end, float) - a)


def bridging_walk(start, end, n_bonds: int, bond: float = 0.97,
                  rng=None, avoid: "Clearance | None" = None,
                  tries: int = 64) -> np.ndarray:
    """A random walk of fixed bond length that lands exactly on ``end``.

    Every step is drawn from the cone of directions that still leaves the far
    junction reachable in the bonds remaining, so the walk closes on it
    without any bond being rescaled afterwards. Returns ``n_bonds + 1``
    points.

    This is the shape a chain in a melt actually has, and the difference from
    a straight line is not cosmetic. A straight chain lies on its chord, so
    whether two chains meet is decided by where their crosslinks sit; a
    coiled one wanders, and chains whose crosslinks are far apart routinely
    run alongside each other while nearest neighbours by crosslink may never
    touch. Anything that needs to know which chains are near which -- the
    entanglement candidate ranking, for one -- needs the coiled shape.

    The walk is unbiased only in the limit of plenty of slack. As the chain
    approaches full extension the reachable cone narrows to nothing and the
    path straightens, which is correct: there is only one way to span a chord
    of ``n_bonds * bond``.

    ``avoid`` keeps the walk off beads that are already there. Each step is
    drawn from the reachable cone as before and kept if it clears; ``tries``
    sets how many draws before settling for the roomiest of them. Every draw
    comes from the same cone as before and the accept test is all that is
    added, so a walk with room everywhere has the same distribution as one
    with no obstacles at all -- though not the same sequence, since drawing
    ``tries`` at a time uses the generator differently. The last bead is a
    junction and is placed wherever the junction is, clear or not.

    Measured drawing one chain through a relaxed melt at density 0.85, the
    tightest contact the path makes: 0.081 to 0.227 sigma with no ``avoid``,
    0.822 with it at 16 draws or fewer only intermittently, 0.822 every time
    at 64. That 0.822 is the melt's own nearest-neighbour distance at the
    junction the walk is pinned to, so it is the best a path between those two
    junctions can do, and the default is the smallest number of draws that
    reaches it reliably.
    """
    rng = np.random.default_rng() if rng is None else rng
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    n_try = 1 if avoid is None else max(1, int(tries))

    pts = [start]
    for k in range(n_bonds - 1):
        d = end - pts[-1]
        r = float(np.linalg.norm(d))
        remaining = n_bonds - k - 1

        # Cosine of the widest angle from which the end is still reachable.
        cos_min = ((r * r + bond * bond - (remaining * bond) ** 2)
                   / (2.0 * r * bond) if r > 1e-12 else -1.0)
        cos_min = min(1.0, max(-1.0, cos_min))

        c = rng.uniform(cos_min, 1.0, n_try)
        s = np.sqrt(np.maximum(0.0, 1.0 - c * c))
        head = d / r if r > 1e-12 else np.array([0.0, 0.0, 1.0])

        ref = [0.0, 0.0, 1.0] if abs(head[2]) < 0.9 else [1.0, 0.0, 0.0]
        t = np.cross(head, ref)
        t /= np.linalg.norm(t)
        u = np.cross(head, t)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_try)

        cand = pts[-1] + bond * (c[:, None] * head
                                 + s[:, None] * (np.cos(phi)[:, None] * t
                                                 + np.sin(phi)[:, None] * u))
        if avoid is None:
            pts.append(cand[0])
            continue
        # First acceptable draw, not the roomiest of them: always taking the
        # roomiest would walk the chain down the middle of whatever void it
        # can find and stop looking like a melt chain. Falling back to the
        # roomiest only matters where nothing clears, which is where the cone
        # has closed and there is no choice left to make anyway.
        gap = avoid.near(cand)
        clear = np.flatnonzero(gap >= avoid.radius)
        pts.append(cand[clear[0]] if len(clear) else cand[int(gap.argmax())])
    pts.append(end)
    return np.array(pts)


def walk_via(start, end, via, n_bonds: int, bond: float = 0.97,
             rng=None, at: float = 0.5,
             avoid: "Clearance | None" = None) -> np.ndarray:
    """A bridging walk from ``start`` to ``end`` that passes through ``via``.

    Two bridging walks joined at the waypoint: ``at`` sets what fraction of
    the bonds are spent getting there. The chain still lands exactly on both
    junctions and every bond is still ``bond`` long.

    This is how a chain reaches a partner its crosslinks are nowhere near. A
    chain carries far more contour than its chord needs -- 77 sigma on a 5.4
    sigma chord at melt density -- and that slack is enough to visit a
    neighbour one or two chord-lengths away and come back. Measured with
    blind draws, partners at 1.0 to 1.7 chord-lengths were reached by 2 to 16
    of 50 attempts; routing through a point on the partner reaches them by
    construction.

    Raises when the detour does not fit: the two legs together cannot be
    shorter than the distance they have to cover.
    """
    rng = np.random.default_rng() if rng is None else rng
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    via = np.asarray(via, float)

    n1 = max(1, int(round(at * n_bonds)))
    n2 = n_bonds - n1
    if n2 < 1:
        raise ValueError("no bonds left for the second leg")

    need1 = float(np.linalg.norm(via - start))
    need2 = float(np.linalg.norm(end - via))
    if need1 > n1 * bond or need2 > n2 * bond:
        raise ValueError(
            f"waypoint out of reach: legs need {need1:.1f} and {need2:.1f} "
            f"sigma but carry {n1 * bond:.1f} and {n2 * bond:.1f}. Move the "
            f"waypoint, shift `at`, or give the chain more contour.")

    first = bridging_walk(start, via, n1, bond, rng, avoid)
    second = bridging_walk(via, end, n2, bond, rng, avoid)
    return np.vstack([first, second[1:]])


def walk_through(start, end, waypoints, n_bonds: int, bond: float = 0.97,
                 rng=None, avoid: "Clearance | None" = None) -> np.ndarray:
    """A bridging walk visiting each of ``waypoints`` in order.

    Bonds are shared between the legs in proportion to how far each has to
    travel, so no leg is left short of what it needs. Raises when the whole
    route is longer than the chain.
    """
    rng = np.random.default_rng() if rng is None else rng
    pts = [np.asarray(start, float)]
    pts += [np.asarray(w, float) for w in waypoints]
    pts.append(np.asarray(end, float))

    legs = [float(np.linalg.norm(pts[i + 1] - pts[i]))
            for i in range(len(pts) - 1)]
    total = sum(legs)
    if total > n_bonds * bond:
        raise ValueError(
            f"route is {total:.1f} sigma but the chain carries "
            f"{n_bonds * bond:.1f}")

    # One bond minimum per leg, the rest shared by distance.
    share = [max(1, int(round(n_bonds * L / total))) for L in legs]
    while sum(share) > n_bonds:
        share[int(np.argmax(share))] -= 1
    while sum(share) < n_bonds:
        share[int(np.argmin([s / max(L, 1e-9) for s, L in zip(share, legs)]))] += 1

    out = [bridging_walk(pts[0], pts[1], share[0], bond, rng, avoid)]
    for i in range(1, len(legs)):
        out.append(
            bridging_walk(pts[i], pts[i + 1], share[i], bond, rng, avoid)[1:])
    return np.vstack(out)


def loop_around(target, i: int, radius: float, n_pts: int = 6,
                phase: float = 0.0,
                avoid: "Clearance | None" = None,
                span: float = 1.0) -> np.ndarray:
    """Waypoints that encircle ``target``'s strand at bead ``i``.

    Returns ``n_pts`` points on a circle of ``radius`` about the target's
    local tangent, so a chain routed through them in order passes once around
    that strand.

    Going *around* is the thing. Routing a chain to a point *on* its intended
    partner puts the two side by side and creates no link between them:
    measured, twelve such attempts raised the routed chain's own entanglement
    count from 3 to 8 while leaving the count with the named partner at zero,
    because at melt density the arriving chain is caught by whichever
    neighbour is topologically in the way. Encircling is what cannot be
    undone by pulling the two taut.

    ``span`` is how much of a turn to make, in units of a full circle. One
    full turn is not the smallest thing that links: it puts two crossings into
    the primitive path, not one, which is why asking for a count of one and
    only ever generating full turns comes back with two every time. Values
    under one make a hook rather than a loop, and somewhere above a half turn
    is where it starts to catch.

    With ``avoid``, the ring keeps its winding but not its exact shape: each
    waypoint is placed inside its own slice of the circle, wherever in that
    slice is clear of the beads already there. The waypoints are landed on
    exactly, so a ring laid across occupied sites puts beads on top of beads
    however well the walk between them behaves.
    """
    target = np.asarray(target, float)
    i = int(np.clip(i, 1, len(target) - 2))
    tan = target[i + 1] - target[i - 1]
    n = float(np.linalg.norm(tan))
    tan = tan / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(tan @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(tan, ref)
    u /= np.linalg.norm(u)
    v = np.cross(tan, u)

    full = 2.0 * np.pi * float(span)

    def ring(ph):
        th = np.linspace(0.0, full, n_pts, endpoint=(span != 1.0)) + ph
        return target[i] + radius * (np.cos(th)[:, None] * u
                                     + np.sin(th)[:, None] * v)

    if avoid is None:
        return ring(phase)

    # Place each waypoint on its own, inside its own slice of the circle.
    #
    # Rotating the ring rigidly does not work. At melt density a point lies
    # within 0.9 sigma of 2.6 beads on average, so there is almost no free
    # volume to rotate into, and one shared angle is not enough freedom to
    # clear every point at once: measured, it left the tightest contact at
    # 0.10 sigma, no better than not trying at all. Nor can the walk between
    # the points make up for it, because at radius 1.2 sigma with eight points
    # they sit 0.92 sigma apart, one bond, so there is nothing in between to
    # move.
    #
    # A path still goes once around as long as each waypoint stays in its own
    # sector, which leaves a slice of angle and a range of radius free per
    # point. Candidates are tried nearest-to-nominal first, so the ring keeps
    # the shape it was asked for wherever it can.
    th0 = np.linspace(0.0, full, n_pts, endpoint=(span != 1.0)) + phase
    d_th = np.linspace(-1.0, 1.0, 7) * (0.6 * abs(full) / max(n_pts, 2))
    d_r = radius * np.array([1.0, 1.15, 0.87, 1.35, 0.75, 1.6])
    rr, tt = (x.ravel() for x in np.meshgrid(d_r, d_th, indexing="ij"))

    out = []
    for a in th0:
        cand = target[i] + rr[:, None] * (np.cos(a + tt)[:, None] * u
                                          + np.sin(a + tt)[:, None] * v)
        nominal = target[i] + radius * (np.cos(a) * u + np.sin(a) * v)
        cand = cand[np.argsort(np.linalg.norm(cand - nominal, axis=1))]
        gap = avoid.near(cand)
        clear = np.flatnonzero(gap >= avoid.radius)
        out.append(cand[clear[0]] if len(clear) else cand[int(gap.argmax())])
    return np.array(out)


def free_walk(start, n_bonds: int, bond: float = 0.97, rng=None) -> np.ndarray:
    """A freely jointed walk of ``n_bonds`` bonds from ``start``.

    Nothing to close on: this is the shape of a sol chain, which is bonded
    to no junction and so has one fixed end at most. Returns
    ``n_bonds + 1`` points, every bond exactly ``bond``.
    """
    rng = np.random.default_rng() if rng is None else rng
    steps = rng.normal(size=(int(n_bonds), 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True) + 1e-12
    return np.vstack([np.asarray(start, float),
                      np.asarray(start, float) + np.cumsum(steps * bond, axis=0)])


def closed_meander(anchor, n_bonds: int, bond: float = 0.97, rng=None,
                   away_from=None, jitter: float = 0.0) -> np.ndarray:
    """The path of a primary loop: out of ``anchor`` and back to it.

    A strand that returns to the junction it left has no chord, so none of
    the routines above applies: there is nothing to interpolate between.
    What it does have is a contour, ``n_bonds * bond``, and the shape that
    spends it with every bond exact and no self-contact is a regular
    polygon of ``n_bonds`` sides with the junction at one vertex.

    Returns the ``n_bonds - 1`` interior beads, in order, so the strand is
    ``anchor -> beads -> anchor`` with ``n_bonds`` bonds. A loop of DP beads
    therefore asks for ``n_bonds = DP + 1``.

    The ring's plane and the direction of its centre are random, except
    that ``away_from`` (the directions of the other strands leaving that
    junction) tilts the centre towards the emptiest side, so a loop does
    not lie on top of the strands it shares a crosslink with. ``jitter``
    adds Gaussian noise in units of ``bond``; the default 0 keeps the
    construction exactly reproducible, and the conformation stage adds its
    own noise afterwards.

    The radius follows from the contour: a polygon of ``n`` sides of length
    ``bond`` has circumradius ``bond / (2 sin(pi / n))``, so a DP-20 loop at
    bond 0.97 is a ring of radius 3.26 sigma. It is small, but every bond
    is right and no two beads of the loop come closer than ``bond``.
    """
    rng = np.random.default_rng() if rng is None else rng
    anchor = np.asarray(anchor, float)
    n_bonds = int(n_bonds)
    if n_bonds < 3:
        raise ValueError(
            f"A closed loop needs at least 3 bonds to have any area; got "
            f"{n_bonds}. A DP-{max(n_bonds - 1, 0)} primary loop cannot be "
            f"drawn; raise the strand's DP."
        )

    radius = bond / (2.0 * np.sin(np.pi / n_bonds))

    # An orthonormal frame: `out` points from the anchor to the ring's
    # centre, `side` spans the plane with it.
    out = _loop_direction(rng, away_from)
    side = _lateral(out, rng.normal(size=3))
    centre = anchor + radius * out

    angles = 2.0 * np.pi * np.arange(1, n_bonds) / n_bonds
    pts = (centre[None, :]
           - radius * np.cos(angles)[:, None] * out[None, :]
           + radius * np.sin(angles)[:, None] * side[None, :])
    if jitter:
        pts = pts + rng.normal(0.0, jitter * bond, pts.shape)
    return pts


def _loop_direction(rng, away_from) -> np.ndarray:
    """A unit vector pointing away from the strands already at a junction."""
    hint = rng.normal(size=3)
    if away_from is None or len(away_from) == 0:
        return hint / (np.linalg.norm(hint) + 1e-12)
    dirs = np.asarray(away_from, float).reshape(-1, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs[norms[:, 0] > 1e-9] / norms[norms[:, 0] > 1e-9].reshape(-1, 1)
    if len(dirs) == 0:
        return hint / (np.linalg.norm(hint) + 1e-12)
    # The emptiest direction of a fixed spread, i.e. the one whose worst
    # alignment with an existing strand is smallest.
    cand = np.vstack([_sphere(), hint / (np.linalg.norm(hint) + 1e-12)])
    worst = (cand @ dirs.T).max(axis=1)
    return cand[int(np.argmin(worst))]


def _lateral(direction, hint):
    """A unit vector perpendicular to ``direction``, as near ``hint`` as it
    can be."""
    d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    h = np.asarray(hint, float)
    h = h - d * float(h @ d)
    n = float(np.linalg.norm(h))
    if n < 1e-9:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(d @ ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        h = np.cross(d, ref)
        n = float(np.linalg.norm(h))
    return h / n


def zigzag(start, end, n_bonds: int, bond: float = 0.97, hint=None):
    """Exact bonds from ``start`` to ``end``, using up every one of them.

    A chain carries far more contour than its route needs. Measured on one
    designed pair: 77 sigma of contour for a route of about 21. A random walk
    disposes of that slack by wandering, and the wandering crosses the target
    again on its own account, so the count stops being a property of the design
    -- the same design, same site, same winding, drawn with three different
    seeds, came back 4, 7 and 0.

    A zigzag spends the same slack in a fixed local shape. It folds against one
    lateral direction and stays near the straight line between its ends, so the
    contour goes somewhere known instead of somewhere random, and the count
    becomes reproducible. Nothing here draws from a generator.

    ``hint`` is the lateral direction to fold against, and only its component
    perpendicular to the leg is used. Point it away from whatever the path is
    meant not to touch.
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    d = end - start
    dist = float(np.linalg.norm(d))
    if n_bonds < 1:
        raise ValueError("a leg needs at least one bond")
    if dist > n_bonds * bond + 1e-9:
        raise ValueError(
            f"leg is {dist:.1f} sigma but carries {n_bonds * bond:.1f}")
    if n_bonds == 1:
        return np.vstack([start, end])

    axis = d / dist if dist > 1e-12 else _lateral([0, 0, 1], [1, 0, 0])
    lat = _lateral(axis, hint if hint is not None else axis + 1.0)

    # Offsets alternate about the straight line and vanish at both ends, so
    # the leg starts and finishes exactly where it was told to.
    sign = np.array([0.0] + [1.0 if i % 2 else -1.0
                             for i in range(1, n_bonds)] + [0.0])

    def reach(h):
        off = sign * h
        step = np.sqrt(np.maximum(bond ** 2 - np.diff(off) ** 2, 0.0))
        return float(step.sum())

    # One bisection on the fold height: taller folds eat more contour, so
    # there is exactly one height that lands the leg on its far end.
    if reach(0.0) < dist:
        h = 0.0
    else:
        lo, hi = 0.0, bond
        while reach(hi) > dist and hi < 50.0 * bond:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if reach(mid) > dist:
                lo = mid
            else:
                hi = mid
        h = 0.5 * (lo + hi)

    off = sign * h
    step = np.sqrt(np.maximum(bond ** 2 - np.diff(off) ** 2, 0.0))
    total = step.sum()
    step = step * (dist / total) if total > 1e-12 else step
    along = np.concatenate([[0.0], np.cumsum(step)])
    return start + along[:, None] * axis + off[:, None] * lat


def route_through(start, end, waypoints, n_bonds: int, bond: float = 0.97,
                  avoid: "Clearance | None" = None):
    """Visit every waypoint in order, spending all the contour deterministically.

    The same job as ``walk_through`` and the same guarantees -- exact bonds,
    lands on both junctions, hits each waypoint -- but the legs come from
    ``taut_leg`` rather than a random walk, so the same design gives the same
    path every time. That is what makes a requested entanglement count
    reproducible: with random legs the leftover contour wanders back across the
    target and adds crossings of its own.

    ``avoid`` keeps the legs off the beads already in the box, and each leg
    also avoids the ones laid down before it.
    """
    pts = [np.asarray(start, float)]
    pts += [np.asarray(w, float) for w in waypoints]
    pts.append(np.asarray(end, float))

    legs = [float(np.linalg.norm(pts[i + 1] - pts[i]))
            for i in range(len(pts) - 1)]
    total = sum(legs)
    if total > n_bonds * bond:
        raise ValueError(
            f"route is {total:.1f} sigma but the chain carries "
            f"{n_bonds * bond:.1f}")

    # Each leg needs enough bonds to span itself; the rest is shared by
    # length. Reserving a spare bond per leg looks harmless and is not: a ring
    # of seventeen waypoints is eighteen legs, so it quietly costs eighteen
    # bonds of contour and turns routes that fit into routes that do not.
    need = [max(1, int(np.ceil(L / bond - 1e-9))) for L in legs]
    share = list(need)
    spare = n_bonds - sum(share)
    if spare < 0:
        raise ValueError(
            f"route needs {sum(need)} bonds but the chain has {n_bonds}")
    for i in np.argsort([-L for L in legs]):
        take = int(round(spare * legs[i] / total)) if total > 1e-12 else 0
        share[i] += take
    share[int(np.argmax(legs))] += n_bonds - sum(share)

    out, placed = [], []
    for i in range(len(legs)):
        leg = taut_leg(pts[i], pts[i + 1], share[i], bond, avoid, placed)
        placed.extend(leg[:-1])
        out.append(leg if not out else leg[1:])
    return np.vstack(out)


def _sphere(n=72):
    """A fixed, evenly spread set of unit directions.

    Deterministic by construction, which is the point: it stands in for the
    random draw so the same design gives the same path every time.
    """
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


_DIRS = _sphere()


def taut_leg(start, end, n_bonds: int, bond: float = 0.97,
             avoid: "Clearance | None" = None, placed=None):
    """A deterministic leg of exact bonds that avoids itself and its
    surroundings.

    Same guarantees as ``bridging_walk`` -- every step from the cone that keeps
    the far end reachable, so it lands there exactly with no bond rescaled --
    but the direction is chosen from a fixed set rather than drawn, so the path
    is reproducible.

    It also avoids the beads it has already laid down. A planar fold cannot:
    given far more bonds than the leg needs, its axial step goes to zero and
    every second bead lands exactly on the one two before it. That is not a
    near miss, it is a zero separation, and LAMMPS reports an infinite pair
    energy and stops. Chains carry three or four times the contour their route
    needs, so this is the normal case rather than the corner one.
    """
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    if n_bonds < 1:
        raise ValueError("a leg needs at least one bond")
    dist = float(np.linalg.norm(end - start))
    if dist > n_bonds * bond + 1e-9:
        raise ValueError(
            f"leg is {dist:.1f} sigma but carries {n_bonds * bond:.1f}")

    mine = [] if placed is None else [np.asarray(p, float) for p in placed]
    pts = [start]
    mine.append(start)
    for k in range(n_bonds - 1):
        d = end - pts[-1]
        r = float(np.linalg.norm(d))
        remaining = n_bonds - k - 1
        cos_min = ((r * r + bond * bond - (remaining * bond) ** 2)
                   / (2.0 * r * bond) if r > 1e-12 else -1.0)
        cos_min = min(1.0, max(-1.0, cos_min))
        head = d / r if r > 1e-12 else np.array([0.0, 0.0, 1.0])

        ok = _DIRS[(_DIRS @ head) >= cos_min - 1e-12]
        if not len(ok):
            ok = head[None, :]
        cand = pts[-1] + bond * ok

        gap = (avoid.near(cand) if avoid is not None
               else np.full(len(cand), np.inf))
        if len(mine) > 1:
            # Its own beads, minus the one it is stepping off.
            #
            # Minimum image where a box is known. The path is built unwrapped,
            # so a chain long enough to cross the cell and fold back meets
            # itself through the boundary at a separation that raw distances
            # do not see, and its own beads are the one set `avoid` cannot
            # cover.
            prev = np.array(mine[:-1])
            d = cand[:, None, :] - prev[None, :, :]
            if avoid is not None and avoid.box is not None:
                d -= avoid.box * np.round(d / avoid.box)
            gap = np.minimum(gap, np.linalg.norm(d, axis=2).min(axis=1))

        # Spend the slack evenly, and prefer a step that clears.
        #
        # The two obvious rules both fail. Roomiest walks the chain down the
        # middle of whatever void it can find, and a chain carrying three
        # times the contour its route needs wanders far enough to cross the
        # target again on its own account: a pair built for 2 came back 3, and
        # 0 after minimisation. Straightest is worse in a quieter way -- it
        # arrives early and has to burn every leftover bond in the small space
        # left, folding on itself to 0.13 sigma, which is the overlap that
        # makes minimisation push chains through each other.
        #
        # Closing the gap at a constant rate spreads the slack along the whole
        # leg instead, which is what an unbiased walk does by accident and
        # this does on purpose.
        want = float(np.clip(r / (bond * max(remaining, 1)), cos_min, 1.0))
        room = np.flatnonzero(gap >= (avoid.radius if avoid is not None
                                      else 0.9))
        pool = room if len(room) else np.arange(len(cand))
        pick = int(pool[np.abs((ok[pool] @ head) - want).argmin()])
        nxt = cand[pick]

        if remaining == 1:
            # The last free bead has to sit exactly one bond from the junction
            # as well as one bond from here, and a fixed set of directions
            # cannot land on that by chance: it closed at 0.841 sigma instead
            # of 0.95. The two conditions describe a circle, so put the bead on
            # the point of it nearest the direction already chosen.
            ax = end - pts[-1]
            dd = float(np.linalg.norm(ax))
            if 1e-9 < dd <= 2.0 * bond:
                ax = ax / dd
                mid = pts[-1] + 0.5 * dd * ax
                rho = np.sqrt(max(bond ** 2 - 0.25 * dd * dd, 0.0))
                v = nxt - mid
                v = v - ax * float(v @ ax)
                n = float(np.linalg.norm(v))
                v = v / n if n > 1e-9 else _lateral(ax, ax + 1.0)
                nxt = mid + rho * v

        pts.append(nxt)
        mine.append(pts[-1])
    pts.append(end)
    return np.array(pts)




def bond_lengths(path) -> np.ndarray:
    """Length of every bond along a path."""
    p = np.asarray(path, float)
    if len(p) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(p, axis=0), axis=1)


def self_contact(path, closed: bool = False) -> float:
    """Closest approach between two beads of a path that share no bond.

    Beads one place apart are held by a bond and are meant to be ``bond``
    apart, so the distance worth reporting is over pairs at least two places
    apart. Returns ``inf`` for a path too short to have such a pair.

    ``closed`` is for a ring, where the first and last points are the *same*
    bead visited twice -- a primary loop leaves its junction and comes back to
    it. Their separation is exactly zero and it is not a contact, so that one
    pair is dropped, along with the two pairs the closing bond holds.
    """
    p = np.asarray(path, float)
    n = len(p)
    if n < 3:
        return float("inf")
    i_idx, j_idx = np.triu_indices(n, k=2)
    if closed:
        # Drop (0, n-1), the repeated anchor, and (0, n-2) / (1, n-1), which
        # the closing bond holds exactly as a bond holds any other neighbour.
        keep = ~(((i_idx == 0) & (j_idx >= n - 2))
                 | ((i_idx == 1) & (j_idx == n - 1)))
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        if not len(i_idx):
            return float("inf")
    return float(np.linalg.norm(p[j_idx] - p[i_idx], axis=1).min())


def fold_into_box(points, box) -> np.ndarray:
    """Wrap coordinates into ``[0, L)`` on every axis.

    The fold is needed because ``x % L`` is not always below ``L``: for an
    ``x`` a little below zero and large in magnitude the nearest representable
    result of the modulo *is* ``L``, and a bead written at exactly the box
    length sits outside the half-open cell every downstream reader assumes.
    ``cKDTree(boxsize=L)`` rejects the whole tree over one such point, the Z1+
    exporter reads it as a bead a box away from where it belongs, and LAMMPS
    remaps it with an image flag that then disagrees with its neighbours'. So
    anything landing on the upper face is folded to the lower one.
    """
    p = np.asarray(points, float)
    L = np.asarray(box, float).reshape(3)
    q = p - L * np.floor(p / L)
    return np.where(q >= L, q - L, q)


def _relax_bonds(p, bond, interior, sweeps, tol, only_long=False,
                 floor=None):
    """Pull the bonds of ``p`` towards ``bond``, holding the ends.

    A Jacobi pass: each bond hands half its error to each of its two beads,
    and a held end simply never receives its half.

    Three modes, and which one is right depends on what else is being asked of
    the path. ``only_long`` corrects bonds above ``bond`` and leaves shorter
    ones alone, which is how a path is finished: a meander carries more
    contour than its chord needs, so shortening is always feasible while
    lengthening may not be. ``floor`` adds the other side of the gate --
    bonds *below* ``floor`` are lengthened too, and everything between
    ``floor`` and ``bond`` is left exactly where it is. That band matters:
    driving every bond to ``bond`` is what undoes the fold-opening the path
    was just given, and it cost 114 of 4644 strands their self-contact
    clearance on the N20 build when it was tried. The plain mode (neither
    flag) chases ``bond`` from both sides and belongs inside
    :func:`unfold`'s own loop, where the separation pass runs again right
    after it.
    """
    n = len(p)
    lo = np.arange(n - 1)
    hi = np.arange(1, n)
    worst = 0.0
    for _ in range(int(sweeps)):
        seg = p[1:] - p[:-1]
        length = np.linalg.norm(seg, axis=1)
        err = length - bond
        if only_long:
            act = err > tol * bond
        elif floor is not None:
            act = (err > tol * bond) | (length < floor)
        else:
            act = np.abs(err) > tol * bond
        worst = float(np.abs(err).max()) if len(err) else 0.0
        if not act.any():
            break
        unit = seg / np.maximum(length, 1e-9)[:, None]
        half = 0.5 * np.where(act, err, 0.0)[:, None] * unit
        corr = np.zeros_like(p)
        np.add.at(corr, lo, half)
        np.add.at(corr, hi, -half)
        # An interior bead is pulled by both its bonds, so the raw sum can
        # overshoot; half of it converges without ringing.
        p[interior] += 0.5 * corr[interior]
    return worst


def unfold(path, bond: float = 0.97, min_sep: float = 1.0, iters: int = 200,
           tol: float = 0.02, bond_sweeps: int = 8,
           smooth: bool = True, frozen=None) -> np.ndarray:
    """Open a path's tight turns and put its bonds back where they belong.

    A meander drawn to a fixed contour spends its slack in waves, and a wave
    tight enough to double back leaves two beads of the *same* chain almost
    touching while the bond between them is nowhere near that short. Measured
    at DP 20 with the six-wave default of :func:`meander_to_length`, bonds came
    out at 0.17 sigma and beads sat jammed between their own neighbours. The
    push-off then shoves those beads apart through the bond that separates
    them, which is a threaded bond -- the same defect the minimiser produced,
    and the one that lets strands cross later (``REPORT.md`` 4.1, 4.2).

    So the fold is opened before anything is written. Beads closer than
    ``min_sep`` to a bead two or more places along the same chain are pushed
    apart, and the bonds are relaxed back towards ``bond`` after every push;
    both ends are held, because they are junctions.

    The bond relaxation is not cosmetic either. Resampling a wavy path at
    equal arc length gives bonds that are *chords* of the arcs between the
    samples, so a six-wave meander at DP 20 -- about three beads per wave --
    comes back with bonds of 0.73 sigma from a path that is exactly the right
    length. Left alone those beads are a third closer together than the melt
    they are about to join. The last pass corrects only bonds that are too
    long, so the path leaves here with every bond at or below ``bond``.

    This is the ``_unfold`` step of ``bond_create_validation/scripts/
    build_topon_system.py``, vectorised and with the bond pass run to
    convergence: the script's pair loop is O(n^2) in Python per iteration per
    chain, which is minutes over a 95 000-bead build and milliseconds here.
    """
    p = np.array(path, float)
    n = len(p)
    if n < 4:
        return p

    # Pairs at least two places apart: the ones no bond already holds.
    i_idx, j_idx = np.triu_indices(n, k=2)
    interior = np.zeros(n, bool)
    interior[1:-1] = True
    if frozen is not None:
        # Beads that carry geometry somebody asked for -- the arms of a
        # designed braid -- and that opening a fold must not move. Unfolding
        # pushes apart beads that sit close with no bond between them, which
        # is exactly what the two turns of a braid do, so letting it touch
        # them would undo the winding it was given.
        interior = interior & ~np.asarray(frozen, bool)

    for _ in range(int(iters)):
        d = p[j_idx] - p[i_idx]
        r = np.linalg.norm(d, axis=1)
        hit = r < min_sep
        moved = bool(hit.any())
        if moved:
            ii, jj = i_idx[hit], j_idx[hit]
            dd, rr = d[hit], r[hit]
            # A coincident pair has no direction to separate along; any fixed
            # one will do, and z is as good as another.
            unit = np.where(rr[:, None] > 1e-9,
                            dd / np.maximum(rr, 1e-9)[:, None],
                            np.array([0.0, 0.0, 1.0]))
            push = 0.5 * (min_sep - rr)[:, None] * unit
            # Accumulated, not applied pair by pair: several violations on one
            # bead would otherwise be served in index order and the last would
            # undo the first.
            delta = np.zeros_like(p)
            np.add.at(delta, ii, -push)
            np.add.at(delta, jj, push)
            p[interior] += delta[interior]

        worst = _relax_bonds(p, bond, interior, bond_sweeps, tol)
        if not moved and worst <= tol * bond:
            break

    # Finish from above: every bond at or below the design length. Two steps,
    # because the gate downstream is `bond_max <= bond` and an iterative
    # solver only ever gets close. The relaxation takes the long bonds most of
    # the way; the equal-arc pass then makes it exact, since a chord is never
    # longer than the arc it spans, so spreading the beads evenly along a path
    # of total length `S` caps every bond at `S / n_bonds`. A path whose
    # length still exceeds its contour has nowhere to put the excess and keeps
    # its long bonds -- that is a strand whose chord does not fit in its DP,
    # which the placement stage reports rather than hides.
    #
    # ``smooth`` is what says whether the equal-arc pass is safe. It resamples
    # *along the polyline*, so the new points cut every corner: on a path
    # whose beads are nearly collinear -- a meander drawn from 600 dense
    # points -- that costs nothing, but on a random walk, whose polyline *is*
    # its own shape, one pass loses about 3 % of the contour and six passes
    # take a 98 sigma chain to 77 with a 0.11 sigma bond in it. So a walk
    # leaves with whatever the relaxation gave it.
    _relax_bonds(p, bond, interior, 4 * int(bond_sweeps), 0.0, only_long=True)
    return _equal_arc(p, n) if smooth else p


def _equal_arc(p, n: int) -> np.ndarray:
    """``n`` points at equal arc length along the polyline ``p``.

    The same arithmetic as
    :func:`~topon.conformation.entanglement.waypoints.resample_path`, kept
    here so this module imports nothing from the rest of the package: it is
    the bottom of the stack and everything else draws on it.
    """
    p = np.asarray(p, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(p[:1], n, axis=0)
    want = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(want, s, p[:, d]) for d in range(3)])


def straight_chain(start, end, n_bonds: int, jitter: float = 0.02,
                   rng=None, bond: float | None = None) -> np.ndarray:
    """The chord, with the interior beads jittered off it.

    ``n_bonds + 1`` points evenly spaced on the straight line between the two
    junctions, so every bond is ``chord / n_bonds`` -- at or below the bond
    length whenever the chord fits inside the contour. The jitter is Gaussian,
    in sigma, and touches only the interior: the two ends are junctions and
    belong where the graph put them. Without it a lattice of identical chords
    gives identical paths and beads of neighbouring strands line up exactly, a
    degeneracy the push-off has to break before it can do anything else.

    The jitter is applied and then taken back out of any bond it pushed past
    ``bond``, because a chord this close to the contour has almost no slack:
    at DP 20 on a chord of 20 sigma the bonds sit at 0.952 and a 0.02 sigma
    jitter is enough to put one over 1.00. Pass ``bond=None`` to skip that
    correction and keep the raw jitter.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_bonds = int(n_bonds)
    base = straight(start, end, n_bonds + 1)
    if not jitter:
        return base
    noise = rng.normal(0.0, float(jitter), (n_bonds + 1, 3))
    noise[0] = 0.0
    noise[-1] = 0.0
    if bond is None or n_bonds < 2:
        return base + noise

    # Shrink the jitter to whatever slack the strand actually has. A jittered
    # path is longer than the chord it was drawn on, and a chord already at
    # the contour has no room for that: the bonds simply come out over the
    # design length and no amount of relaxing brings them back, because the
    # ends are fixed and the straight line is already the shortest path
    # between them. Bisection on the amplitude is exact and costs nothing.
    room = n_bonds * float(bond)
    def length(a):
        return float(np.linalg.norm(np.diff(base + a * noise, axis=0),
                                    axis=1).sum())
    if length(1.0) > room:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if length(mid) > room:
                hi = mid
            else:
                lo = mid
        noise = lo * noise
    p = base + noise
    interior = np.zeros(n_bonds + 1, bool)
    interior[1:-1] = True
    _relax_bonds(p, float(bond), interior, 32, 0.0, only_long=True)
    return _equal_arc(p, n_bonds + 1)


def meander_chain(start, end, n_bonds: int, bond: float = 0.97, rng=None,
                  waves: float = 6.0, min_waves: float = 0.5,
                  min_sep: float = 1.0, min_bond: float = 0.85,
                  straight_at: float = 0.97, jitter: float = 0.02,
                  samples: int = 600, unfold_iters: int = 200):
    """A smooth meander carrying the contour the beads need.

    A chain's contour is ``n_bonds * bond`` whatever its chord is, so the path
    is drawn at that length rather than in the hope that the chord obliges:
    the chord is waved sideways by
    :func:`~topon.conformation.entanglement.waypoints.meander_to_length` until
    it measures the contour, then resampled to ``n_bonds + 1`` points at equal
    arc length and unfolded (:func:`unfold`) so the bonds come back to length.

    Why not a random walk. A walk of the same contour disposes of its slack by
    wandering, and the wandering runs through whatever is beside it: measured
    on the N20 reference graph, random-walk placement floors at Z = 0.23 per
    DP-20 strand however dilute the build, while the meander at the same state
    reproduces the reference's per-strand distribution exactly (Z 0.189 against
    0.178, KS p = 1.0). ``REPORT.md`` section 4.

    ``waves`` is how many full waves the slack is spent in, and it is tried and
    then halved down to ``min_waves`` while the path still has a bond below
    ``min_bond`` or a bead within ``min_sep`` of a non-adjacent bead of its own
    chain. Both gates matter and they pull the same way: three beads per wave
    is not enough to draw a wave with, and what comes back is a fold, not a
    coil. The roomiest draw is the one kept, scored on the bond first because a
    short bond is what threads.

    A chord within ``straight_at`` of the contour has no slack to spend and
    nothing to wave, so it is drawn as the chord with a little jitter.

    Returns ``(path, info)``; ``info`` names the routine that drew it, the
    waves kept, how many draws it took and the two gate readings.
    """
    from topon.conformation.entanglement.waypoints import meander_to_length

    rng = np.random.default_rng() if rng is None else rng
    a = np.asarray(start, float)
    b = np.asarray(end, float)
    n_bonds = int(n_bonds)
    contour = n_bonds * bond
    chord = float(np.linalg.norm(b - a))

    if chord >= straight_at * contour:
        p = straight_chain(a, b, n_bonds, jitter=jitter, rng=rng, bond=bond)
        return p, {"routine": "straight", "waves": 0.0, "draws": 0,
                   "reason": f"chord is within {straight_at:g} of the contour",
                   "bond_min": float(bond_lengths(p).min()),
                   "self_contact": self_contact(p)}

    dense = _equal_arc(np.stack([a, b]), int(samples))
    w = float(waves)
    draws = 0
    best = None
    while True:
        draws += 1
        p = _equal_arc(meander_to_length(dense, contour, waves=w),
                       n_bonds + 1)
        if not np.all(np.isfinite(p)):
            p = bridging_walk(a, b, n_bonds, bond, rng)
            return p, {"routine": "walk", "waves": w, "draws": draws,
                       "reason": "meander returned non-finite points",
                       "bond_min": float(bond_lengths(p).min()),
                       "self_contact": self_contact(p)}
        p = unfold(p, bond, min_sep=min_sep, iters=unfold_iters)
        b_min = float(bond_lengths(p).min())
        gap = self_contact(p)
        score = (min(b_min / max(min_bond, 1e-9), 1.0),
                 min(gap / max(min_sep, 1e-9), 1.0))
        if best is None or score > best[0]:
            best = (score, p, w, b_min, gap)
        if (b_min >= min_bond and gap >= min_sep) or w <= min_waves:
            break
        w = max(min_waves, 0.5 * w)

    _score, p, w, b_min, gap = best
    p = _spin_about_chord(p, rng)
    return p, {"routine": "meander", "waves": float(w), "draws": draws,
               "bond_min": b_min, "self_contact": float(gap)}


def _spin_about_chord(p, rng) -> np.ndarray:
    """Turn a path about the line joining its ends, by a random angle.

    Everything :func:`meander_to_length` decides is deterministic, down to the
    plane the wave lies in: the normal comes from parallel-transporting a fixed
    reference axis along the chord. On a lattice that is a degeneracy, because
    every strand with the same chord direction then waves into the same plane
    and rows of parallel strands bulge together instead of into each other's
    gaps. It also leaves the placement with nothing for a seed to change.

    A rotation about the chord fixes both. The two junctions lie on the axis so
    neither moves, and every bond length, every self-contact and the contour
    are rotation invariant, so the gate reads exactly the same -- only which
    way the slack points changes, which is the one thing that should depend on
    the seed.
    """
    p = np.asarray(p, float)
    axis = p[-1] - p[0]
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return p
    axis = axis / n
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    c, s = np.cos(phi), np.sin(phi)
    v = p - p[0]
    # Rodrigues, with the chord as the axis.
    return (p[0] + v * c + np.cross(axis, v) * s
            + np.outer(v @ axis, axis) * (1.0 - c))
