"""Did a designed pair get the winding it was asked for? A topological reading.

A designed entanglement is two strands wound ``e`` times about each other,
each held at its two junctions. The winding of two *open* strands is not a
topological invariant, however it is closed off. The far-closed linking number
(:func:`topon.conformation.entanglement.braid.far_closed_linking`) closes each
strand through two legs from its junctions to a point far away. Those legs are
not atoms, and a strand can move through the partner's legs without passing
through the partner. That changes the reading by one with no passage anywhere.
On a DP-30 atomistic build with six designed pairs, one pair read 1.21 as drawn
and -0.11 settled, with no passage between them. Fixing the legs in space
does not help, since a strand crosses a fixed leg as easily as a moving one.
The only closure a strand cannot move through without being seen is one made
of other strands.

So the reading here closes each strand of the pair through the network that
holds it. For a designed pair (A, B) it takes a cycle of strands C_A through
A and a cycle C_B through B that share no junction. It also takes both cycles
contractible (their bond vectors, read in the minimum image, sum to zero), so
each is a closed curve in space and not a loop around the periodic box. The
linking number Lk(C_A, C_B) of two disjoint closed curves is an integer. It
is constant under any motion that keeps them disjoint (Gauss's integral is a
continuous function on such pairs, and it takes integer values). With atoms
moving in straight lines between two frames, which is what the crossing
detector (:mod:`topon.analysis.crossings`) and the settles assume, it changes
exactly when a bond of one cycle passes through a bond of the other, by one
up or down for each such passage. Junctions may move, as they do in MD, and
the cycles' other strands may move anywhere. A run with no backbone passage
keeps every Lk(C_A, C_B), and one reading at a checkpoint says so without a
trajectory. The reading needs the two cycles: on a very small cell every
cycle through one strand of a pair can touch every cycle through the other
(two of three pairs on a 3x3x3 SC cell of 47 strands), and such a pair is
not read.

Lk(C_A, C_B) is the pair's winding plus whatever the rest of the two cycles
happen to link, which at a DP-30 build's entanglement density can be anything.
The designed winding is therefore read against a reference: the same build
with that pair's two strands drawn without their winding and every other
strand, the other designed pairs included, as it is. On the
coarse-grained route that is the braid untwisted in place
(:func:`~topon.conformation.entanglement.designed.unwound_paths`), on the
atomistic route the pair drawn by its waypoint construction at zero turns
(:func:`~topon.conformation.entanglement.realize.entangled_backbone_paths`
with ``windings=0``):

    delivered = Lk(C_A, C_B) - Lk_ref(C_A, C_B)

Both terms are integers. The reference is a fixed number, computed once on
the build (``reference_linking`` passes it to a later reading). So a run with
no passage delivers at every stage what it delivered at the build. The sign
says which way the two strands are labelled: every waypoint braid has the
same hand, and a pair whose strands run antiparallel (junction u to v) reads
-1. A request gives no sign, and a reading of magnitude ``e`` delivers ``e``. A braid is a local change to the drawing: two strands
twisted ``e`` times about each other where the unwound drawing passes side by
side. Such a change adds exactly ``e`` to the linking of any two cycles through
the two strands, unless a strand of either cycle runs through the braid
itself, where the winding is then shared with it. That is why several cycle
pairs are read. The designed winding is common to all of them, and a third
strand in the braid shifts only the cycle pairs it lies on, so the reading is
quoted when every choice agrees and flagged when they do not.
:func:`cycle_pairs` picks the cycle pairs to leave the pair's junctions by
different strands where the network has them, which is a best effort: a
strand that every candidate cycle runs through, or one through the braid far
from the junctions, shifts all of them alike, and they then agree on a
reading it has changed. Agreement is evidence, not proof.

A reading more than 0.05 from an integer is not counted: two polygons that
nearly touch (a drawn build before the settle parts its touching bonds) have
a linking number that the smallest move changes. Two limits: at a multi-atom
junction (POSS) the cycle steps between attachment atoms in a straight line,
which is not a bond and which a strand can cross unseen; and the partner
cycle's periodic image is chosen at every reading by where the two strands
come closest, which could pick another image in a box not much larger than
the strands.

Nothing here draws a strand. The caller passes each strand's points, from
junction to junction, continuous (unwrapped). :func:`strands_from_record`
builds them from an atomistic strand record and positions (a data file or a
settle frame), and :func:`strands_from_placement` from a bead-spring
``Placement``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Iterator, Optional

import numpy as np

__all__ = [
    "StrandPath",
    "PairWinding",
    "gauss_linking",
    "polygon_distance",
    "cycle_candidates",
    "cycle_pairs",
    "cycle_polygon",
    "cycle_linking",
    "find_cycles",
    "measure_cycles",
    "designed_windings",
    "strands_from_record",
    "record_reference",
    "strands_from_placement",
]

#: Rows of segment pairs handled at once by :func:`gauss_linking`.
_CHUNK = 200_000

#: How far from an integer a linking number may be and still be read.
WHOLE = 0.05


def _whole(x) -> bool:
    return abs(float(x) - round(float(x))) <= WHOLE


@dataclass(frozen=True)
class StrandPath:
    """One strand, junction to junction.

    ``points`` run from the end on node ``u`` to the end on node ``v`` and are
    continuous (each step the bond as it is, not wrapped). Only bridges, with
    a junction at each end, can be part of a cycle.
    """

    u: Hashable
    v: Hashable
    points: np.ndarray

    @property
    def displacement(self) -> np.ndarray:
        p = np.asarray(self.points, float)
        return p[-1] - p[0]


@dataclass
class PairWinding:
    """One designed pair, read on several cycle pairs.

    ``linking`` and ``reference`` hold one value per cycle pair (floats, to
    show how close to an integer each came); ``delivered`` is their rounded
    difference per cycle pair, and ``value`` the delivered winding when every
    cycle pair agrees on it (None when they do not, or when no reference was
    given).
    """

    pair: tuple
    requested: Optional[int]
    cycles: list = field(default_factory=list)
    linking: list = field(default_factory=list)
    reference: Optional[list] = None
    distance: list = field(default_factory=list)

    @property
    def delivered(self) -> Optional[list]:
        """Delivered count per cycle pair; None where a reading is not an integer."""
        if self.reference is None:
            return None
        return [int(round(a)) - int(round(b)) if _whole(a) and _whole(b) else None
                for a, b in zip(self.linking, self.reference)]

    @property
    def value(self) -> Optional[int]:
        d = self.delivered
        if not d or None in d or len(set(d)) != 1:
            return None
        return d[0]

    @property
    def agree(self) -> bool:
        d = self.delivered
        return bool(d) and len(set(d)) == 1

    def summary(self) -> dict:
        return {
            "pair": list(self.pair),
            "requested": self.requested,
            "delivered": self.value,
            "per_cycle_pair": self.delivered,
            "linking": [round(float(x), 4) for x in self.linking],
            "reference": (None if self.reference is None else
                          [round(float(x), 4) for x in self.reference]),
            "closest_approach": [round(float(x), 4) for x in self.distance],
            "cycles": [[[k for k, _f in ca], [k for k, _f in cb]]
                       for ca, cb in self.cycles],
        }


# ---------------------------------------------------------------------------
# The linking number of two closed polygons, exactly
# ---------------------------------------------------------------------------

def _unit_rows(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = n[..., 0] > 1e-300
    return np.where(n > 1e-300, v / np.where(n > 1e-300, n, 1.0), 0.0), ok


def _segment_pairs(p1, p2, p3, p4) -> np.ndarray:
    """Gauss's integral over each pair of segments, exactly.

    The integral of the linking density over segment ``p1 p2`` against
    segment ``p3 p4`` is the signed solid angle the quadrilateral of
    difference vectors subtends, over 4 pi (Klenin and Langowski, Biopolymers
    54, 307 (2000), eq. 16). Summed over every segment of one closed polygon
    against every segment of another, it is their linking number, with no
    discretisation error however coarse the polygons are.
    """
    r13, r14 = p3 - p1, p4 - p1
    r23, r24 = p3 - p2, p4 - p2
    n1, o1 = _unit_rows(np.cross(r13, r14))
    n2, o2 = _unit_rows(np.cross(r14, r24))
    n3, o3 = _unit_rows(np.cross(r24, r23))
    n4, o4 = _unit_rows(np.cross(r23, r13))

    def asin_dot(a, b):
        return np.arcsin(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))

    omega = asin_dot(n1, n2) + asin_dot(n2, n3) + asin_dot(n3, n4) + asin_dot(n4, n1)
    sign = np.sign(np.einsum("ij,ij->i", np.cross(p4 - p3, p2 - p1), r13))
    # Collinear or touching configurations subtend no solid angle: a face
    # normal of zero length means the four points are degenerate there.
    ok = o1 & o2 & o3 & o4
    return np.where(ok, omega * sign, 0.0) / (4.0 * np.pi)


def _closed(poly) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(poly, float).reshape(-1, 3)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p, np.roll(p, -1, axis=0)


def gauss_linking(loop_a, loop_b) -> float:
    """Linking number of two closed polygons, from their vertices.

    Each polygon is closed from its last vertex back to its first (a repeated
    first vertex at the end is accepted and ignored). The result is an
    integer up to rounding for any two disjoint polygons. Two polygons that
    touch have no linking number; :func:`polygon_distance` says how close they
    come.
    """
    a0, a1 = _closed(loop_a)
    b0, b1 = _closed(loop_b)
    m, n = len(a0), len(b0)
    if m < 3 or n < 3:
        return 0.0
    total = 0.0
    rows = max(1, _CHUNK // n)
    for s in range(0, m, rows):
        e = min(m, s + rows)
        k = e - s
        p1 = np.repeat(a0[s:e], n, axis=0)
        p2 = np.repeat(a1[s:e], n, axis=0)
        p3 = np.tile(b0, (k, 1))
        p4 = np.tile(b1, (k, 1))
        total += float(_segment_pairs(p1, p2, p3, p4).sum())
    return total


def polygon_distance(loop_a, loop_b) -> float:
    """Closest approach between two closed polygons."""
    from topon.conformation.segments import segment_closest

    a0, a1 = _closed(loop_a)
    b0, b1 = _closed(loop_b)
    n = len(b0)
    best = np.inf
    rows = max(1, _CHUNK // max(n, 1))
    for s in range(0, len(a0), rows):
        e = min(len(a0), s + rows)
        k = e - s
        d = segment_closest(np.repeat(a0[s:e], n, axis=0), np.repeat(a1[s:e], n, axis=0),
                            np.tile(b0, (k, 1)), np.tile(b1, (k, 1)))[2]
        best = min(best, float(d.min()))
    return best


# ---------------------------------------------------------------------------
# Cycles of the network
# ---------------------------------------------------------------------------

def _mic(d, box):
    if box is None:
        return d
    return d - box * np.round(d / box)


def _simple_graph(strands: dict, exclude: set):
    """Junctions and the bridges between them, parallel strands kept by chord."""
    import networkx as nx

    G = nx.Graph()
    for key, s in strands.items():
        if key in exclude or s.u == s.v or s.u is None or s.v is None:
            continue
        w = float(np.linalg.norm(s.displacement))
        if G.has_edge(s.u, s.v):
            G[s.u][s.v]["keys"].append((w, key))
            G[s.u][s.v]["w"] = min(G[s.u][s.v]["w"], w)
        else:
            G.add_edge(s.u, s.v, keys=[(w, key)], w=w)
    for u, v in G.edges():
        G[u][v]["keys"].sort(key=lambda x: (x[0], str(x[1])))
    return G


def cycle_candidates(strands: dict, through, box=None, avoid_nodes=(),
                     avoid_strands=(), limit: int = 200) -> Iterator[list]:
    """Contractible cycles of strands through strand ``through``, shortest first.

    A cycle is a list of ``(strand key, forward)``, starting with ``through``
    run from its ``u`` to its ``v``, then back to ``u`` through other bridges
    that avoid ``avoid_nodes`` and ``avoid_strands``. Shortest is by the sum of
    the strands' end-to-end distances. A cycle whose displacements do not sum
    to zero winds around the periodic box and is not a closed curve in space,
    so it is skipped. At most ``limit`` paths are examined.
    """
    import networkx as nx

    box = None if box is None else np.asarray(box, float).reshape(3)
    s0 = strands[through]
    if s0.u == s0.v:
        return
    avoid_nodes = set(avoid_nodes)
    if s0.u in avoid_nodes or s0.v in avoid_nodes:
        return
    G = _simple_graph(strands, set(avoid_strands) | {through})
    G.remove_nodes_from([n for n in avoid_nodes if n in G])
    if s0.u not in G or s0.v not in G:
        return
    tol = 0.25 * float(box.min()) if box is not None else np.inf
    try:
        paths = nx.shortest_simple_paths(G, s0.v, s0.u, weight="w")
        for count, nodes in enumerate(paths):
            if count >= limit:
                break
            cycle = [(through, True)]
            total = np.array(s0.displacement, float)
            for x, y in zip(nodes[:-1], nodes[1:]):
                key = G[x][y]["keys"][0][1]
                s = strands[key]
                forward = (s.u == x)
                cycle.append((key, forward))
                total = total + (s.displacement if forward else -s.displacement)
            if box is not None and float(np.linalg.norm(total)) > tol:
                continue
            yield cycle
    except nx.NetworkXNoPath:
        return


def _nodes_of(strands: dict, cycle) -> set:
    out = set()
    for key, _f in cycle:
        out |= {strands[key].u, strands[key].v}
    return out


def _ends(cycle) -> set:
    """The strands a cycle leaves its first strand's two junctions by."""
    return {cycle[1][0], cycle[-1][0]} if len(cycle) > 1 else set()


def cycle_pairs(strands: dict, a, b, box=None, n: int = 4, per_a: int = 6,
                limit: int = 200, tries: int = 12, avoid=()) -> list:
    """Up to ``n`` pairs of disjoint contractible cycles, one through ``a``, one through ``b``.

    The cycle through ``a`` avoids ``b``'s junctions, and the one through
    ``b`` every junction of the first, so the two share no strand and no
    junction. The shortest pair comes first. The rest are picked to leave the
    four junctions of the pair by strands not used yet, so that a strand
    running through the braid next to a junction is on some cycle pairs and
    not on others: shortest-first cycles tend to leave a junction by the same
    strand every time, and then all of them agree on a reading that strand
    has changed (on a DP-30 build all four shortest cycles through one
    designed strand left its junction by the strand its partner's braid
    passed 0.69 A from). At most ``tries`` cycles through ``a`` and
    ``per_a`` partners for each are considered. ``avoid`` are strands neither cycle may use.
    """
    if a not in strands or b not in strands:
        return []                 # a loop or a dangling strand has no cycle
    sa, sb = strands[a], strands[b]
    avoid = set(avoid) - {a, b}
    pool = []
    for t, ca in enumerate(cycle_candidates(strands, a, box, avoid_nodes={sb.u, sb.v},
                                            avoid_strands={b} | avoid, limit=limit)):
        if t >= tries:
            break
        for k, cb in enumerate(cycle_candidates(strands, b, box,
                                                avoid_nodes=_nodes_of(strands, ca),
                                                avoid_strands={k for k, _f in ca} | avoid,
                                                limit=limit)):
            if k >= per_a:
                break
            pool.append((ca, cb))
    out, used = [], set()
    while pool and len(out) < n:
        best = max(range(len(pool)),
                   key=lambda i: (len((_ends(pool[i][0]) | _ends(pool[i][1])) - used), -i))
        ca, cb = pool.pop(best)
        out.append((ca, cb))
        used |= _ends(ca) | _ends(cb)
    return out


def cycle_polygon(strands: dict, cycle, box=None, replace: Optional[dict] = None,
                  tol: float = 1e-6) -> np.ndarray:
    """The cycle as one closed polygon in space, its vertices in order.

    Each strand is moved by the box vector that puts its first point on the
    end of the one before, so the polygon is continuous across the boundary.
    Where two strands of the cycle end on different atoms of one node (a
    multi-atom junction), the polygon steps between them in a straight line.
    ``replace`` gives other points for some strands (a reference drawing).
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    replace = replace or {}
    out = []
    for key, forward in cycle:
        p = np.asarray(replace.get(key, strands[key].points), float)
        p = p if forward else p[::-1]
        if out:
            end = out[-1][-1]
            if box is not None:
                p = p + box * np.round((end - p[0]) / box)
            if float(np.linalg.norm(p[0] - end)) <= tol:
                p = p[1:]
        out.append(p)
    poly = np.vstack(out)
    first, last = poly[0], poly[-1]
    gap = last - first
    if box is not None:
        gap = gap - box * np.round(gap / box)
    if float(np.linalg.norm(gap)) <= tol:
        poly = poly[:-1]
    return poly


def _pair_shift(pa, pb, box):
    """The box vector that puts ``pb`` in the image nearest ``pa``.

    Read at the two strands' closest approach, which for a wound pair is a
    few bonds, far inside half a box, so the image is unambiguous.
    """
    if box is None:
        return np.zeros(3)
    d = pa[:, None, :] - pb[None, :, :]
    m = _mic(d, box)
    i, j = np.unravel_index(int(np.argmin((m * m).sum(-1))), m.shape[:2])
    return d[i, j] - m[i, j]


def cycle_linking(strands: dict, a, b, ca, cb, box=None,
                  replace: Optional[dict] = None) -> tuple[float, float]:
    """``(Lk, closest approach)`` of cycle ``ca`` (through ``a``) and ``cb`` (through ``b``).

    ``cb`` is taken in the periodic image where strand ``b`` comes closest to
    strand ``a``.
    """
    box = None if box is None else np.asarray(box, float).reshape(3)
    replace = replace or {}
    pa = cycle_polygon(strands, ca, box, replace)
    pb = cycle_polygon(strands, cb, box, replace)
    # the two strands as they sit in their cycles' frames
    sa = np.asarray(replace.get(a, strands[a].points), float)
    sb = np.asarray(replace.get(b, strands[b].points), float)
    sa = sa + (pa[0] - sa[0])
    sb = sb + (pb[0] - sb[0])
    shift = _pair_shift(sa, sb, box)
    pb = pb + shift
    return gauss_linking(pa, pb), polygon_distance(pa, pb)


def find_cycles(strands: dict, pairs: Iterable, box=None, n: int = 4) -> dict:
    """``{(a, b): [(cycle_a, cycle_b), ...]}`` for every pair.

    A pair's cycles keep off the strands of the other designed pairs where
    the network allows, and use them only when no other two disjoint cycles
    close in space. A designed strand is wound with its own partner, and a
    cycle through it carries that winding into the reading. It is in the
    build and in this pair's reference alike (each pair is read with only its
    own strands unwound) and cancels, but keeping off it keeps each reading
    to its own pair's strands.
    """
    pairs = [tuple(p) for p in pairs]
    designed = {k for p in pairs for k in p}
    out = {}
    for p in pairs:
        found = cycle_pairs(strands, p[0], p[1], box, n=n, avoid=designed - set(p))
        out[p] = found or cycle_pairs(strands, p[0], p[1], box, n=n)
    return out


def measure_cycles(strands: dict, cycles: dict, box=None,
                   replace: Optional[dict] = None) -> dict:
    """Lk and closest approach of every cycle pair, ``{(a, b): [(Lk, d), ...]}``."""
    return {pair: [cycle_linking(strands, pair[0], pair[1], ca, cb, box, replace)
                   for ca, cb in cps]
            for pair, cps in cycles.items()}


def designed_windings(strands: dict, pairs: dict, box=None,
                      reference: Optional[dict] = None, n_cycles: int = 4,
                      cycles: Optional[dict] = None,
                      reference_linking: Optional[dict] = None) -> list:
    """Every designed pair read on its network cycles.

    ``pairs`` is ``{(a, b): windings requested}`` (the requested count may be
    None). ``reference`` maps strand keys to the points of the unwound
    drawing; each pair is read against its own two strands replaced and
    every other strand as in ``strands``, so the reference is only meaningful
    on the state it was drawn for (the build). ``reference_linking``
    (``{(a, b): [Lk_ref per cycle pair]}``, a build's ``PairWinding.reference``)
    reads a later state against that fixed number instead. Without either,
    only the linking numbers are reported. ``cycles`` reuses the cycles of an
    earlier reading, which is what makes readings at two stages comparable.
    """
    cycles = cycles if cycles is not None else find_cycles(strands, pairs, box, n_cycles)
    now = measure_cycles(strands, cycles, box)
    out = []
    for pair, req in pairs.items():
        pair = tuple(pair)
        cps = cycles.get(pair, [])
        pw = PairWinding(pair=pair, requested=req, cycles=cps,
                         linking=[x for x, _d in now.get(pair, [])],
                         distance=[d for _x, d in now.get(pair, [])])
        if reference_linking is not None and pair in reference_linking:
            pw.reference = list(reference_linking[pair])
        elif reference is not None:
            own = {k: reference[k] for k in pair if k in reference}
            ref = measure_cycles(strands, {pair: cps}, box, own)[pair]
            pw.reference = [x for x, _d in ref]
        out.append(pw)
    return out


# ---------------------------------------------------------------------------
# Strands from an atomistic record or a bead-spring placement
# ---------------------------------------------------------------------------

def strands_from_record(record: dict, pos: dict, box) -> dict:
    """Every bridge of an atomistic strand record, as points.

    ``record`` is the run manifest's strand record (LAMMPS atom ids) and
    ``pos`` maps atom id to position, wrapped or not (a data file's Atoms,
    or a frame of the settle). Each strand is junction atom, backbone,
    junction atom, unwrapped bond by bond from its first junction. Nodes are
    named by their first attachment atom, so strands on a multi-atom node
    share it. Keys are strand numbers, 1-based in record order, the
    numbering :func:`topon.analysis.crossings.designed_pairs` uses.
    """
    box = np.asarray(box, float).reshape(3)
    node_of = {}
    for n in record.get("nodes", []):
        attach = n["attach"] if isinstance(n["attach"], list) else [n["attach"]]
        for a in attach:
            node_of[int(a)] = int(attach[0])
    out = {}
    for k, row in enumerate(record["strands"], start=1):
        j1, j2 = (list(row.get("junctions") or []) + [None, None])[:2]
        if j1 is None or j2 is None:
            continue
        chain = [int(j1)] + [int(a) for a in row["backbone"]] + [int(j2)]
        x = np.array([pos[a] for a in chain], float)
        step = _mic(np.diff(x, axis=0), box)
        pts = x[0] + np.vstack([np.zeros(3), np.cumsum(step, axis=0)])
        u, v = node_of.get(int(j1), int(j1)), node_of.get(int(j2), int(j2))
        if u == v:
            continue
        out[k] = StrandPath(u=u, v=v, points=pts)
    return out


def record_reference(record: dict, pos: dict, box, interiors: dict) -> dict:
    """Reference points for the strands of a record drawn some other way.

    ``interiors`` maps a strand's edge key ``(u, v, key)`` to the positions
    of its backbone atoms (junctions excluded), as the pipeline's drawing
    returns them (:func:`~topon.conformation.entanglement.realize.
    entangled_backbone_paths`, scaled to A). Each is put between the
    strand's junction atoms at ``pos`` and unwrapped, keyed by strand number
    like :func:`strands_from_record`, ready for
    :func:`designed_windings`'s ``reference``.
    """
    box = np.asarray(box, float).reshape(3)
    out = {}
    for k, row in enumerate(record["strands"], start=1):
        interior = interiors.get(tuple(row["edge"]))
        j1, j2 = (list(row.get("junctions") or []) + [None, None])[:2]
        if interior is None or j1 is None or j2 is None:
            continue
        x = np.vstack([pos[int(j1)], np.asarray(interior, float).reshape(-1, 3),
                       pos[int(j2)]])
        step = _mic(np.diff(x, axis=0), box)
        out[k] = x[0] + np.vstack([np.zeros(3), np.cumsum(step, axis=0)])
    return out


def strands_from_placement(placement) -> dict:
    """Every bridge of a bead-spring ``Placement``, keyed by strand index.

    The index is the order ``placement.strands`` keeps, the one a designed
    request (:class:`~topon.conformation.entanglement.PairRequest`) names.
    """
    out = {}
    for i, s in enumerate(placement.strands):
        if s.plan.kind != "bridge":
            continue
        out[i] = StrandPath(u=s.plan.u, v=s.plan.v,
                            points=np.asarray(s.path, float))
    return out
