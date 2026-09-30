"""Network defects, injected after sculpting and before chemistry.

Four classes of defect, with the vocabulary the polymer-network literature
uses (topon called parallel edges "primary loops" before 0.2.0; see the
deprecation note at the bottom of this module):

``primary loop``
    A strand that returns to the junction it left. Stored as a MultiGraph
    self-loop edge with ``cls="loop"``. It has no lattice representation,
    so it is added here rather than by the topology search.
``secondary loop``
    Two strands between the same junction pair, i.e. a parallel edge.
    Placed by the exact sculptor as a forced double so the degree counts
    stay exact; this module verifies and records what the sculptor placed,
    and falls back to endpoint-targeted injection on a graph that is
    already sculpted.
``triangle`` / ``four-cycle``
    Higher-order loops: one added edge that closes a three- or four-cycle.
``sol``
    Chains bonded to no junction. They are not in the junction graph at
    all, but they carry beads, so they are recorded on the graph for the
    bead budget that sizes the box at the target density.

Two functionalities per junction, kept apart throughout:

``effective``
    The strands that leave the junction and reach somewhere else, which
    is what elasticity sees. ``G.degree(n)`` minus twice the self-loops.
``chemical``
    What the crosslinker's valence sees: effective plus two per primary
    loop, which is exactly ``G.degree(n)`` on a MultiGraph, since NetworkX
    counts a self-loop twice.

Reference statistics, measured on the DP-20 `fix bond/create` network in
``bond_create_validation/``: 345 primary loops on 338 junctions (322 of
effective degree 2 with one loop, 8 of effective degree 1 with one, 7 of
effective degree 0 with two, 1 of effective degree 0 with one), 129
secondary loops with endpoint effective degrees ``{(4,4): 115, (3,4): 6,
(2,4): 7, (2,3): 1}``, and 11 sol chains; effective P(f) ``{4: 1975,
3: 153, 2: 356, 1: 8, 0: 8}`` against chemical ``{4: 2304, 3: 161,
2: 35}``.

All three are reproduced exactly by ``by_effective_degree`` placement,
which fills junctions in ascending effective degree (0 takes two loops,
1 through ``f_target - 2`` take one). The original rule was written with
classes for 0 and ``f_target - 2`` only, which leaves the eight
effective-degree-1 junctions empty and lands on ``{4: 2312, 3: 153,
2: 27, 1: 8}``; the intermediate classes are the correction, measured
against the reference on 2026-09-16.
"""

from __future__ import annotations

import random
import warnings
from collections import Counter
from typing import Iterable, Optional, Sequence

import networkx as nx


__all__ = [
    "apply_defects",
    "bead_budget",
    "chemical_degrees",
    "analyze_defect_potential",
    "count_parallel_edges",
    "count_secondary_loops",
    "count_self_loops",
    "degree_histogram",
    "effective_degree",
    "effective_degrees",
    "get_eligible_pairs",
    "is_end_site",
    "inject_four_cycles",
    "inject_secondary_loops",
    "inject_self_loops",
    "inject_triangles",
    "loop_placement_classes",
    "loop_placement_plan",
    "network_functionality",
    "plan_double_pairs",
    "reserve_capacity",
    "resolve_count",
    "verify_secondary_loops",
    # deprecated
    "analyze_primary_loop_potential",
    "count_primary_loops",
    "inject_primary_loops",
]


# ---------------------------------------------------------------------------
# Degrees
# ---------------------------------------------------------------------------

def effective_degree(G: nx.MultiGraph, n) -> int:
    """Strands leaving ``n`` that end somewhere else (self-loops excluded)."""
    return int(G.degree(n)) - 2 * int(G.number_of_edges(n, n))


def effective_degrees(G: nx.MultiGraph) -> dict:
    """``{node: effective degree}`` for every node."""
    return {n: effective_degree(G, n) for n in G.nodes()}


def chemical_degrees(G: nx.MultiGraph) -> dict:
    """``{node: chemical degree}``: effective plus two per primary loop."""
    return {n: int(d) for n, d in G.degree()}


def degree_histogram(degrees: dict, nodes: Optional[Iterable] = None) -> dict:
    """``{degree: count}`` over ``nodes`` (all of them by default)."""
    if nodes is not None:
        nodes = set(nodes)
        degrees = {n: d for n, d in degrees.items() if n in nodes}
    return dict(sorted(Counter(degrees.values()).items()))


def count_self_loops(G: nx.MultiGraph) -> int:
    """Number of primary loops (self-loop edges)."""
    return sum(1 for u, v in G.edges() if u == v)


def count_secondary_loops(G: nx.MultiGraph) -> int:
    """Number of junction pairs carrying more than one strand."""
    return sum(1 for c in _pair_multiplicity(G).values() if c > 1)


def count_parallel_edges(G: nx.MultiGraph) -> int:
    """Number of *extra* strands on multiply-connected pairs.

    A pair with three strands contributes two. This is the count the
    reference reports as its secondary-loop total (129 for N20), and it
    equals :func:`count_secondary_loops` whenever no pair carries more
    than two strands.
    """
    return sum(c - 1 for c in _pair_multiplicity(G).values() if c > 1)


def _pair_multiplicity(G: nx.MultiGraph) -> dict:
    """``{(u, v): strand count}`` for every distinct pair, self-loops out."""
    pairs: dict = {}
    for u, v in G.edges():
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _declared_type(G: nx.MultiGraph, n):
    """The node's declared type, or None when it carries none.

    ``node_type`` is topon's attribute; ``type`` is the legacy spelling and
    ``kind`` the one the end-linked reference parser writes.
    """
    attrs = G.nodes[n]
    for key in ("node_type", "type", "kind"):
        if key in attrs:
            return attrs[key]
    return None


def _node_type(G: nx.MultiGraph, n) -> str:
    declared = _declared_type(G, n)
    return "A" if declared is None else declared


def is_end_site(G: nx.MultiGraph, n) -> bool:
    """A degree-1 site: the free end of a dangling chain, not a crosslinker.

    An explicit node type decides when there is one; otherwise the
    effective degree does, which is the convention the sculptors write
    (``"end" if degree == 1 else "junction"``).
    """
    declared = _declared_type(G, n)
    if declared is not None:
        return declared in ("end", "END")
    return effective_degree(G, n) == 1


def _is_junction(G: nx.MultiGraph, n, exclude_node_types=("end",)) -> bool:
    """A site a strand may be attached to at both ends.

    End sites have one valence, so they can carry neither a loop nor a
    second strand.
    """
    if exclude_node_types and _node_type(G, n) in set(exclude_node_types):
        return False
    return not is_end_site(G, n)


def resolve_count(count, n_strands: int) -> int:
    """Turn a config ``count`` into a number of defects.

    ``count >= 1`` is absolute; ``0 < count < 1`` is a fraction of the
    strands in the graph.
    """
    if count is None:
        return 0
    count = float(count)
    if count <= 0:
        return 0
    if count < 1.0:
        return int(round(count * n_strands))
    return int(round(count))


# ---------------------------------------------------------------------------
# Primary loops (self-loops)
# ---------------------------------------------------------------------------

def _inherited_attrs(G: nx.MultiGraph, nodes, dp: Optional[int], dp_default: int) -> dict:
    """DP and edge type for a new strand, taken from the strands already
    on the junctions it joins."""
    # Bridges first: a dangling strand's DP is one short under the
    # end-linked convention (its free end is the DP-th bead), and a loop
    # that inherited that would spend one bead too few.
    neighbour_dps, fallback_dps, edge_type = [], [], None
    for node in nodes:
        for u, v, data in G.edges(node, data=True):
            if edge_type is None and data.get("edge_type"):
                edge_type = data["edge_type"]
            if not data.get("dp"):
                continue
            if u == v or data.get("cls") in ("dangling", "loop"):
                fallback_dps.append(int(data["dp"]))
            elif is_end_site(G, u) or is_end_site(G, v):
                fallback_dps.append(int(data["dp"]))
            else:
                neighbour_dps.append(int(data["dp"]))
    neighbour_dps = neighbour_dps or fallback_dps
    if dp is not None:
        value = int(dp)
    elif neighbour_dps:
        value = int(round(sum(neighbour_dps) / len(neighbour_dps)))
    else:
        value = int(dp_default)
    attrs = {"dp": value}
    if edge_type is not None:
        attrs["edge_type"] = edge_type
    return attrs


def _loop_edge_attrs(G: nx.MultiGraph, node, dp: Optional[int], dp_default: int) -> dict:
    """Attributes for a loop strand: DP and edge type from its junction."""
    return {"cls": "loop", "is_primary_loop": True,
            **_inherited_attrs(G, (node,), dp, dp_default)}


def network_functionality(G: nx.MultiGraph, fallback: int = 4) -> int:
    """The functionality the network was built for.

    The highest effective degree any junction carries, which is the
    tetrafunctionality of an f = 4 network whatever ceiling the config left
    ``max_functionality`` at. Placement classes are keyed off this, not off
    the valence ceiling: with the schema default of 6 on a tetrafunctional
    graph, ``f_target - 2`` would otherwise select the f = 4 junctions and
    push them to chemical f = 6.
    """
    degrees = [effective_degree(G, n) for n in G.nodes() if _is_junction(G, n)]
    return max(degrees) if degrees else int(fallback)


def loop_placement_plan(G: nx.MultiGraph, max_f: int = 4,
                        f_target: Optional[int] = None) -> list:
    """Which junctions take loops, in the order they take them.

    ``[(effective degree, loops each, [junctions]), ...]`` in ascending
    effective degree: an effective-degree-0 junction takes two loops, every
    junction from there up to ``f_target - 2`` takes one. A junction of
    effective degree ``f_target - 2`` carrying one loop reads as
    ``f_target`` chemically, which is the defect the reference shows.

    ``f_target`` defaults to the network's own functionality; ``max_f`` is
    the separate valence ceiling, which decides what a junction may carry.

    Ascending order is what the reference does, measured on the DP-20
    network: its 345 loops sit on 8 effective-degree-0 junctions (7 with
    two, 1 with one), 8 effective-degree-1 junctions and 322
    effective-degree-2 junctions. Filling only the 0 and ``f_target - 2``
    classes, as the original rule did, leaves those 8
    effective-degree-1 junctions empty and moves their loops to
    effective-degree-2 junctions, which shifts the chemical P(f) on eight
    junctions.
    """
    if f_target is None:
        f_target = network_functionality(G, fallback=max_f)
    by_degree: dict = {}
    for n in G.nodes():
        if not _is_junction(G, n):
            continue
        eff = effective_degree(G, n)
        if 0 <= eff <= f_target - 2:
            by_degree.setdefault(eff, []).append(n)
    return [(eff, 2 if eff == 0 else 1, by_degree[eff])
            for eff in sorted(by_degree)]


def loop_placement_classes(G: nx.MultiGraph, max_f: int = 4,
                           f_target: Optional[int] = None) -> dict:
    """The plan of :func:`loop_placement_plan` as two classes.

    ``two_loops`` are the junctions that take two (effective degree 0),
    ``one_loop`` those that take one, in ascending effective degree.
    """
    plan = loop_placement_plan(G, max_f=max_f, f_target=f_target)
    doubles = [n for eff, each, nodes in plan if each == 2 for n in nodes]
    singles = [n for eff, each, nodes in plan if each == 1 for n in nodes]
    return {"two_loops": doubles, "one_loop": singles}


def inject_self_loops(
    G: nx.MultiGraph,
    count: int,
    placement: str = "by_effective_degree",
    dp: Optional[int] = None,
    dp_default: int = 25,
    max_f: int = 4,
    rng: Optional[random.Random] = None,
    f_target: Optional[int] = None,
) -> dict:
    """Attach ``count`` primary loops to junctions, modifying ``G`` in place.

    ``max_f`` is the valence ceiling a junction may not exceed; ``f_target``
    is the functionality the placement classes are keyed off, defaulting to
    the network's own (see :func:`network_functionality`). Placement
    ``by_effective_degree`` fills in ascending effective degree, which is
    what the reference does (see :func:`loop_placement_plan`).

    Returns a record with the requested and achieved counts and how many
    loops each placement class took.
    """
    rng = rng or random
    placed = {"two_loops": 0, "one_loop": 0, "spare_valence": 0}
    if count <= 0:
        return {"requested": 0, "achieved": 0, "placement": placement, "by_class": placed}

    def attach(node) -> bool:
        if G.degree(node) + 2 > max_f:
            return False
        G.add_edge(node, node, **_loop_edge_attrs(G, node, dp, dp_default))
        return True

    remaining = int(count)
    by_degree: dict = {}
    if placement == "by_effective_degree":
        # Ascending effective degree, at random within each degree.
        for eff, each, nodes in loop_placement_plan(G, max_f=max_f,
                                                    f_target=f_target):
            if remaining <= 0:
                break
            for node in _shuffled(nodes, rng):
                if remaining <= 0:
                    break
                for _ in range(each):
                    if remaining <= 0:
                        break
                    if attach(node):
                        placed["two_loops" if each == 2 else "one_loop"] += 1
                        by_degree[eff] = by_degree.get(eff, 0) + 1
                        remaining -= 1
    elif placement != "random":
        raise ValueError(
            f"Unknown primary-loop placement {placement!r} "
            f"(expected 'by_effective_degree' or 'random')"
        )

    if remaining > 0:
        # `random` placement, or `by_effective_degree` asked for more loops
        # than its two classes can hold: any junction with spare valence,
        # one loop per junction per pass so they spread.
        key = "random" if placement == "random" else "spare_valence"
        placed.setdefault(key, 0)
        spare = [n for n in G.nodes() if _is_junction(G, n) and G.degree(n) + 2 <= max_f]
        progress = True
        while remaining > 0 and progress:
            progress = False
            for node in _shuffled(spare, rng):
                if remaining <= 0:
                    break
                if attach(node):
                    placed[key] += 1
                    remaining -= 1
                    progress = True

    achieved = int(count) - remaining
    placed["f_target"] = int(
        f_target if f_target is not None else network_functionality(G, fallback=max_f)
    )
    placed["by_effective_degree"] = dict(sorted(by_degree.items()))
    if remaining:
        warnings.warn(
            f"Requested {int(count)} primary loops, placed {achieved}; no "
            f"junction with spare chemical valence (max_functionality "
            f"{max_f}) was left.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "requested": int(count),
        "achieved": achieved,
        "placement": placement,
        "f_target": placed.pop("f_target"),
        "by_effective_degree": placed.pop("by_effective_degree"),
        "by_class": {k: v for k, v in placed.items() if v},
    }


def _shuffled(items: Sequence, rng) -> list:
    out = list(items)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Secondary loops (parallel edges)
# ---------------------------------------------------------------------------

def get_eligible_pairs(
    G: nx.MultiGraph,
    max_degree: Optional[int] = None,
    exclude_node_types: tuple = ("end",),
) -> list:
    """Junction pairs that can take a second strand.

    Conditions: the pair carries exactly one strand today; with
    ``max_degree`` set, both ends are below it; and neither end is an
    end-cap node, which has one free valence (a Si chain-cap with three
    methyls) and is the wrong place for a parallel strand anyway.
    """
    candidates = []
    excluded = set(exclude_node_types or ())
    for (u, v), count in _pair_multiplicity(G).items():
        if count != 1:
            continue
        if max_degree is not None and (G.degree(u) >= max_degree or G.degree(v) >= max_degree):
            continue
        if excluded and (_node_type(G, u) in excluded or _node_type(G, v) in excluded):
            continue
        candidates.append((u, v))
    return candidates


def plan_double_pairs(endpoint_degrees, count: Optional[int] = None) -> Optional[dict]:
    """Translate a config's ``endpoint_degrees`` into the sculptor's request.

    Returns ``{(a, b): n}`` for ``sculpt_exact(double_pairs=...)``, or
    ``None`` for ``"auto"`` (the sculptor draws pairs so the multigraph
    P(f) matches the requested one). When ``count`` disagrees with the
    histogram's total, the histogram is scaled to it, largest remainder
    first, so ``count`` stays the authority.
    """
    if endpoint_degrees is None or endpoint_degrees == "auto":
        return None
    spec: dict = {}
    for key, value in endpoint_degrees.items():
        if isinstance(key, str):
            a, b = (int(x) for x in key.replace(" ", "").split(","))
        else:
            a, b = (int(x) for x in key)
        spec[(min(a, b), max(a, b))] = spec.get((min(a, b), max(a, b)), 0) + int(value)
    total = sum(spec.values())
    if not count or count == total or total == 0:
        return spec
    scaled, remainders = {}, []
    for pair, n in spec.items():
        exact = n * count / total
        scaled[pair] = int(exact)
        remainders.append((exact - int(exact), pair))
    short = count - sum(scaled.values())
    for _, pair in sorted(remainders, reverse=True)[:short]:
        scaled[pair] += 1
    return {p: n for p, n in scaled.items() if n > 0}


def auto_double_pairs(need: dict, count: int, max_f: int = 4) -> dict:
    """Endpoint degrees for ``count`` doubles when the config says "auto".

    "Auto" means: let the sculpt decide which pairs carry the doubles, so
    long as the multigraph P(f) comes out as requested. Since the forced
    doubles are placed before the fill and the target already counts them,
    any assignment keeps P(f) exact and the only question is whether the
    sites exist. This pairs like with like from the highest degree down,
    which is where the room is and, on a near-complete tetrafunctional
    target, what the reference does anyway (115 of its 129 doubles sit
    between two f = 4 junctions).

    Raises ``ValueError`` when the target has too few sites of degree 2 or
    above to carry that many doubles.
    """
    spec, left = {}, int(count)
    for degree in range(int(max_f), 1, -1):
        if left <= 0:
            break
        room = int(need.get(degree, 0)) // 2
        take = min(room, left)
        if take:
            spec[(degree, degree)] = take
            left -= take
    if left:
        raise ValueError(
            f"{count} secondary loops asked for, room for {count - left}: the "
            f"target has too few sites of degree 2 and above to pair up. Give "
            f"endpoint_degrees explicitly, or ask for fewer."
        )
    return spec


def inject_secondary_loops(
    G: nx.MultiGraph,
    target: int,
    target_type: str = "count",
    inherit_dp: bool = True,
    max_degree: Optional[int] = None,
    endpoint_degrees=None,
    rng: Optional[random.Random] = None,
) -> int:
    """Add parallel strands to junction pairs, modifying ``G`` in place.

    This is the *fallback* path, for a graph that is already sculpted. The
    exact route is to force the doubles inside the sculpt
    (:func:`plan_double_pairs` builds that request), which keeps the degree
    counts exact; injecting afterwards raises both endpoints by one and so
    shifts P(f).

    With ``endpoint_degrees`` the injection is targeted: for a requested
    pair class ``(a, b)`` it picks single strands whose endpoints are at
    effective degree ``(a - 1, b - 1)`` today, so that they read ``(a, b)``
    once the parallel strand is there. Without it, eligible pairs are
    drawn at random, which is what distorted P(f) before 0.2.0 (on the DP-20
    reference: 79 too few f = 2, 134 too many f = 3).

    Returns the number of parallel strands added.
    """
    rng = rng or random
    if target_type == "percentage":
        num_to_inject = max(1, int(len(get_eligible_pairs(G, max_degree=max_degree)) * target / 100))
    else:
        num_to_inject = int(target)
    if num_to_inject <= 0:
        return 0

    injected = 0
    if endpoint_degrees:
        spec = plan_double_pairs(endpoint_degrees, num_to_inject) or {}
        for (a, b), wanted in sorted(spec.items(), key=lambda kv: -kv[1]):
            placed = 0
            pairs = _shuffled(get_eligible_pairs(G, max_degree=max_degree), rng)
            for u, v in pairs:
                if placed >= wanted:
                    break
                du, dv = effective_degree(G, u), effective_degree(G, v)
                if sorted((du + 1, dv + 1)) != [a, b]:
                    continue
                if max_degree is not None and (G.degree(u) >= max_degree or G.degree(v) >= max_degree):
                    continue
                _add_parallel(G, u, v, inherit_dp)
                placed += 1
                injected += 1
        if injected < num_to_inject:
            warnings.warn(
                f"Endpoint-targeted secondary-loop injection placed {injected} of "
                f"{num_to_inject}; the remaining pair classes had no eligible "
                f"strands at the required endpoint degrees.",
                RuntimeWarning,
                stacklevel=2,
            )
        return injected

    eligible = get_eligible_pairs(G, max_degree=max_degree)
    if not eligible:
        warnings.warn(
            f"No eligible pairs for secondary-loop injection (max_degree={max_degree}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    for u, v in _shuffled(eligible, rng):
        if injected >= num_to_inject:
            break
        # A node can appear in several eligible pairs; re-check the degree
        # at injection time or we silently over-valence at chemistry.
        if max_degree is not None and (G.degree(u) >= max_degree or G.degree(v) >= max_degree):
            continue
        _add_parallel(G, u, v, inherit_dp)
        injected += 1
    if injected < num_to_inject:
        warnings.warn(
            f"Secondary-loop injection placed {injected} of {num_to_inject}; "
            f"{len(eligible)} pairs were eligible at the start and the "
            f"valence ceiling {max_degree} consumed the rest.",
            RuntimeWarning,
            stacklevel=2,
        )
    return injected


def _add_parallel(G: nx.MultiGraph, u, v, inherit_dp: bool) -> None:
    existing = G.get_edge_data(u, v)
    attrs = {}
    if existing:
        attrs = dict(existing[list(existing.keys())[0]])
        if not inherit_dp:
            attrs.pop("dp", None)
    attrs["cls"] = attrs.get("cls", "bridge")
    attrs["is_secondary_loop"] = True
    G.add_edge(u, v, **attrs)


def verify_secondary_loops(G: nx.MultiGraph, requested: Optional[int] = None) -> dict:
    """Count the parallel strands in ``G`` and their endpoint degrees.

    This is the whole of the secondary-loop step when the sculptor has
    already forced the doubles: the graph is right by construction, and
    what is left is to record what it holds.
    """
    endpoint = Counter()
    for (u, v), count in _pair_multiplicity(G).items():
        if count > 1:
            du, dv = effective_degree(G, u), effective_degree(G, v)
            endpoint[(min(du, dv), max(du, dv))] += count - 1
    achieved = count_parallel_edges(G)
    record = {
        "achieved": achieved,
        "pairs": count_secondary_loops(G),
        "endpoint_degrees": {f"{a},{b}": n for (a, b), n in sorted(endpoint.items())},
    }
    if requested is not None:
        record["requested"] = int(requested)
    return record


# ---------------------------------------------------------------------------
# Higher-order loops: triangles and four-cycles
# ---------------------------------------------------------------------------

def _simple_adjacency(G: nx.MultiGraph) -> dict:
    """``{node: set(neighbours)}`` with self-loops and multiplicity dropped."""
    adj = {n: set() for n in G.nodes()}
    for u, v in G.edges():
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _new_triangles(adj: dict, x, y) -> int:
    """Triangles that appear when the edge ``x-y`` is added."""
    return len(adj[x] & adj[y])


def _new_four_cycles(adj: dict, x, y) -> int:
    """Four-cycles that appear when the edge ``x-y`` is added.

    One per path of length three from ``x`` to ``y``.
    """
    count = 0
    for a in adj[x]:
        if a == y:
            continue
        for b in adj[a]:
            if b in (x, y):
                continue
            if y in adj[b]:
                count += 1
    return count


def _inject_cycles(
    G: nx.MultiGraph,
    count: int,
    max_f: int,
    rng,
    kind: str,
    prefer_degree: Optional[int] = None,
    dp_default: int = 25,
) -> dict:
    """Shared driver for triangle and four-cycle injection.

    Adds one edge per requested cycle, between two junctions that are not
    adjacent and both have spare chemical valence. Pairs that create
    exactly one new cycle are preferred, so ``count`` requested is
    ``count`` delivered; a pair creating several is taken only when no
    single-cycle pair is left, and the record says so.

    Two passes: the first takes only endpoints at ``prefer_degree``, which
    is where :func:`reserve_capacity` parked the spare valence, so the
    delivered P(f) is the requested one. What that pass cannot place, the
    second takes from any junction below ``max_f``, the deviation route,
    where ``degree_shift`` says how far P(f) moved.
    """
    record = {"requested": int(count), "achieved": 0, "cycles_created": 0,
              "endpoints_raised": 0, "endpoints_at_reserved_degree": 0,
              "degree_shift": 0, "multi_cycle_edges": 0,
              "collateral_cycles": 0}
    if count <= 0:
        return record

    adj = _simple_adjacency(G)
    counter = _new_triangles if kind == "triangle" else _new_four_cycles
    # An edge that closes a triangle usually closes four-cycles as well, and
    # the other way round. Square clustering is part of the descriptor set
    # the reference is matched on, so the collateral is counted, not hidden.
    other = _new_four_cycles if kind == "triangle" else _new_triangles

    def candidate_pairs(pivot):
        neighbours = [n for n in adj[pivot] if _is_junction(G, n)]
        if kind == "triangle":
            return [(x, y) for i, x in enumerate(neighbours)
                    for y in neighbours[i + 1:]]
        two_step = set()
        for w in neighbours:
            two_step |= {z for z in adj[w]
                         if z != pivot and z not in adj[pivot] and _is_junction(G, z)}
        return [(x, z) for x in neighbours for z in two_step if x != z]

    def run_pass(required_degree):
        def has_room(n):
            if G.degree(n) >= max_f:
                return False
            return required_degree is None or G.degree(n) == required_degree

        junctions = [n for n in G.nodes() if _is_junction(G, n)]
        for pivot in _shuffled(junctions, rng):
            if record["cycles_created"] >= count:
                return
            pairs = candidate_pairs(pivot)
            rng.shuffle(pairs)
            best = None
            for x, y in pairs:
                if x == y or y in adj[x] or not (has_room(x) and has_room(y)):
                    continue
                made = counter(adj, x, y)
                if made == 1:
                    best = (x, y, made)
                    break
                # An edge closing several cycles at once is a last resort,
                # and never one that would overshoot the request.
                if 1 < made <= count - record["cycles_created"] and best is None:
                    best = (x, y, made)
            if best is None:
                continue
            x, y, made = best
            record["collateral_cycles"] += other(adj, x, y)
            G.add_edge(x, y, cls="bridge", defect_cycle=kind,
                       **_inherited_attrs(G, (x, y), None, dp_default))
            adj[x].add(y)
            adj[y].add(x)
            record["achieved"] += 1
            record["cycles_created"] += made
            record["endpoints_raised"] += 2
            if required_degree is not None:
                record["endpoints_at_reserved_degree"] += 2
            if made > 1:
                record["multi_cycle_edges"] += 1

    if prefer_degree is not None:
        run_pass(prefer_degree)
    if record["cycles_created"] < count:
        run_pass(None)
    # Every added edge raises two junctions. When the sculpt reserved the
    # capacity (``reserve_capacity`` parked that many junctions one class
    # below max_f, and the topology stage recorded it on the graph), the
    # endpoints taken from that class land exactly where the requested
    # P(f) wanted them and are not a deviation. Otherwise every raised
    # endpoint is one.
    reserved = bool(G.graph.get("reserved_capacity", False))
    record["reserved_capacity"] = reserved
    record["degree_shift"] = record["endpoints_raised"] - (
        record["endpoints_at_reserved_degree"] if reserved else 0
    )

    if record["cycles_created"] < count:
        warnings.warn(
            f"Requested {count} {kind}s, created {record['cycles_created']}; "
            f"the graph had no further non-adjacent pair with spare valence.",
            RuntimeWarning,
            stacklevel=3,
        )
    return record


def inject_triangles(G: nx.MultiGraph, count: int, max_f: int = 4, rng=None,
                     prefer_degree: Optional[int] = None,
                     dp_default: int = 25) -> dict:
    """Close ``count`` three-cycles, modifying ``G`` in place.

    Each added edge raises both endpoints' chemical degree by one. On a
    graph sculpted with :func:`reserve_capacity` the endpoints are at
    ``max_f - 1`` (pass ``prefer_degree=max_f - 1``) and the final P(f) is
    the requested one; otherwise the record's ``degree_shift`` is the
    deviation to report.
    """
    return _inject_cycles(G, count, max_f, rng or random, "triangle",
                          prefer_degree, dp_default)


def inject_four_cycles(G: nx.MultiGraph, count: int, max_f: int = 4, rng=None,
                       prefer_degree: Optional[int] = None,
                       dp_default: int = 25) -> dict:
    """Close ``count`` four-cycles, modifying ``G`` in place. See
    :func:`inject_triangles`."""
    return _inject_cycles(G, count, max_f, rng or random, "four_cycle",
                          prefer_degree, dp_default)


def reserve_capacity(need: dict, n_triangles: int = 0, n_four_cycles: int = 0,
                     max_f: int = 4) -> dict:
    """The sculpt target that leaves room for the higher-order defects.

    Each added edge raises two junctions by one degree, so the sculptor is
    asked for a P(f) in which ``2 * (triangles + four_cycles)`` junctions
    sit one class below ``max_f``; the injector then raises exactly those
    back, and the delivered P(f) is the one the user asked for.

    Raises ``ValueError`` when the requested P(f) has too few ``max_f``
    junctions to reserve from. The alternative is the deviation route:
    inject into the sculpted graph and report the shift.
    """
    reserved = 2 * (int(n_triangles) + int(n_four_cycles))
    target = {int(d): int(n) for d, n in need.items()}
    if reserved == 0:
        return target
    if target.get(max_f, 0) < reserved:
        raise ValueError(
            f"Cannot reserve capacity for {n_triangles} triangles and "
            f"{n_four_cycles} four-cycles: {reserved} junctions at degree "
            f"{max_f} are needed, the target has {target.get(max_f, 0)}."
        )
    target[max_f] -= reserved
    target[max_f - 1] = target.get(max_f - 1, 0) + reserved
    return target


# ---------------------------------------------------------------------------
# Bead budget
# ---------------------------------------------------------------------------

def bead_budget(
    G: nx.MultiGraph,
    dp_default: int = 25,
    sol_count: int = 0,
    sol_dp: Optional[int] = None,
) -> dict:
    """Beads per strand class, and the total that sizes the box.

    Loops and sol are part of the budget: in the DP-20 reference they hold
    7 % of the beads, and leaving them out shrinks the box at a fixed
    density until the bridge density, and with it the tensile peak, is
    7 % too high (REPORT.md 5.4).

    Every strand contributes the ``dp`` on its edge, so the end-linked
    convention (where a dangling strand's free end *is* its DP-th bead,
    leaving ``dp - 1`` beads of its own) needs nothing here: it is applied
    once, at DP assignment (``dp_distribution.endlinked_dangling``). That
    holds for a strand with one end site. A strand between two of them
    would be one bead over, since it pays for two end sites and gave up
    one bead; the sculptors never build one (a degree-1 site may only bond
    to a junction), so it is not special-cased.
    """
    junction_beads = end_beads = 0
    for n in G.nodes():
        if G.degree(n) == 0:
            continue
        if is_end_site(G, n):
            end_beads += 1
        else:
            junction_beads += 1

    beads = {"bridge": 0, "dangling": 0, "loop": 0}
    counts = {"bridge": 0, "dangling": 0, "loop": 0}
    for u, v, data in G.edges(data=True):
        dp = int(data.get("dp") or dp_default)
        if u == v:
            cls = "loop"
        elif is_end_site(G, u) or is_end_site(G, v):
            cls = "dangling"
        else:
            cls = "bridge"
        counts[cls] += 1
        beads[cls] += dp

    sol_dp = int(sol_dp or dp_default)
    sol_beads = int(sol_count) * sol_dp
    total = junction_beads + end_beads + sum(beads.values()) + sol_beads
    return {
        "junction_beads": junction_beads,
        "end_beads": end_beads,
        "chains": {**counts, "sol": int(sol_count)},
        "beads": {**beads, "sol": sol_beads},
        "total_beads": total,
    }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def apply_defects(
    G: nx.MultiGraph,
    config,
    dp_default: int = 25,
    max_f: int = 4,
    rng: Optional[random.Random] = None,
) -> dict:
    """Run the post-sculpt defects stage on ``G``, in place.

    ``config`` is a :class:`topon.config.schema.DefectsConfig`. Returns the
    record the run manifest and ``topon inspect`` report: requested against
    achieved for every class, and the chemical against the effective P(f).
    """
    if rng is None:
        rng = random.Random(config.seed) if getattr(config, "seed", None) is not None else random
    n_strands_before = G.number_of_edges()
    report: dict = {"max_functionality": int(max_f)}

    # --- secondary loops ---------------------------------------------------
    sec = config.secondary_loops
    if sec.enabled:
        requested = resolve_count(
            sec.count if sec.count is not None else sec.target, n_strands_before
        )
        present = count_parallel_edges(G)
        if present >= requested > 0:
            # The sculptor forced the doubles; verify and record only.
            record = verify_secondary_loops(G, requested)
            record["source"] = "sculpt"
        else:
            added = inject_secondary_loops(
                G,
                target=requested - present,
                target_type="count",
                max_degree=max_f,
                endpoint_degrees=(
                    None if sec.endpoint_degrees == "auto" else sec.endpoint_degrees
                ),
                rng=rng,
            )
            record = verify_secondary_loops(G, requested)
            record["source"] = "injected"
            record["injected"] = added
            record["degree_shift"] = 2 * added
        report["secondary_loops"] = record

    # --- primary loops -----------------------------------------------------
    pri = config.primary_loops
    if pri.enabled and pri.is_legacy_parallel_request:
        warnings.warn(
            "assignment.defects.primary_loops.target injects parallel edges, "
            "which are secondary loops; this is the v0.1.0 meaning of the key "
            "and is kept for one release. Use assignment.defects.secondary_loops "
            "for parallel strands, and primary_loops.count for self-loops.",
            FutureWarning,
            stacklevel=2,
        )
        added = inject_secondary_loops(
            G, target=pri.target, target_type=pri.target_type,
            max_degree=max_f, rng=rng,
        )
        record = verify_secondary_loops(G, pri.target)
        record["source"] = "injected (legacy primary_loops key)"
        record["injected"] = added
        report["secondary_loops"] = record
    elif pri.enabled:
        requested = resolve_count(pri.count, n_strands_before)
        present = count_self_loops(G)
        if present >= requested > 0:
            # Already there: a re-run of the stage, or a graph loaded with
            # its loops. Record, do not double up.
            report["primary_loops"] = {
                "requested": requested, "achieved": present,
                "placement": pri.placement, "source": "present",
                "by_class": {},
            }
        else:
            report["primary_loops"] = inject_self_loops(
                G,
                count=requested - present,
                placement=pri.placement,
                dp=pri.dp,
                dp_default=dp_default,
                max_f=max_f,
                rng=rng,
            )
            report["primary_loops"]["requested"] = requested
            report["primary_loops"]["achieved"] = count_self_loops(G)

    # --- higher-order loops ------------------------------------------------
    for key, injector in (("triangles", inject_triangles),
                          ("four_cycles", inject_four_cycles)):
        cfg = getattr(config, key)
        if not cfg.enabled:
            continue
        report[key] = injector(
            G, resolve_count(cfg.count, n_strands_before), max_f=max_f, rng=rng,
            prefer_degree=max_f - 1, dp_default=dp_default,
        )

    # --- sol ---------------------------------------------------------------
    sol = config.sol_chains
    sol_count = resolve_count(sol.count, n_strands_before) if sol.enabled else 0
    if sol_count:
        G.graph["sol_chains"] = {
            "count": sol_count,
            "dp": int(sol.dp or dp_default),
        }
        report["sol_chains"] = {"requested": sol_count, "achieved": sol_count,
                                "dp": int(sol.dp or dp_default)}

    # --- what the graph now holds -----------------------------------------
    junctions = [n for n in G.nodes() if G.degree(n) > 0 and _is_junction(G, n)]
    report["effective_degree"] = degree_histogram(effective_degrees(G), junctions)
    report["chemical_degree"] = degree_histogram(chemical_degrees(G), junctions)
    report["bead_budget"] = bead_budget(
        G, dp_default=dp_default, sol_count=sol_count,
        sol_dp=(sol.dp if sol.enabled else None),
    )
    G.graph["defects"] = report
    return report


def analyze_defect_potential(G: nx.MultiGraph, max_degree: Optional[int] = None,
                             max_f: int = 4) -> dict:
    """Capacity of the graph for each defect class, for the analysis stage."""
    eligible = get_eligible_pairs(G, max_degree=max_degree)
    classes = loop_placement_classes(G, max_f=max_f)
    return {
        "max_possible_secondary_loops": len(eligible),
        "existing_secondary_loops": count_secondary_loops(G),
        "eligible_pairs": len(eligible),
        "max_possible_primary_loops": 2 * len(classes["two_loops"]) + len(classes["one_loop"]),
        "existing_primary_loops": count_self_loops(G),
        "constraints": {"max_degree": max_degree, "max_functionality": max_f},
    }


# ---------------------------------------------------------------------------
# Deprecated names (0.2.0; kept for one release)
# ---------------------------------------------------------------------------

def inject_primary_loops(G: nx.MultiGraph, target: int, target_type: str = "count",
                         inherit_dp: bool = True, max_degree: Optional[int] = None) -> int:
    """Deprecated alias of :func:`inject_secondary_loops`.

    Despite the name this has always added *parallel edges*, which are
    secondary loops. A primary loop, a strand returning to its own
    junction, is :func:`inject_self_loops`.
    """
    warnings.warn(
        "inject_primary_loops adds parallel edges, which are secondary loops; "
        "it is renamed inject_secondary_loops. Primary (self-) loops are "
        "inject_self_loops. This alias will be removed in a later release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return inject_secondary_loops(
        G, target=target, target_type=target_type,
        inherit_dp=inherit_dp, max_degree=max_degree,
    )


def count_primary_loops(G: nx.MultiGraph) -> int:
    """Deprecated alias of :func:`count_secondary_loops` (its old meaning:
    the number of pairs carrying a parallel strand)."""
    warnings.warn(
        "count_primary_loops counts parallel-edge pairs, i.e. secondary "
        "loops; it is renamed count_secondary_loops. Self-loops are "
        "count_self_loops. This alias will be removed in a later release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return count_secondary_loops(G)


def analyze_primary_loop_potential(G: nx.MultiGraph, max_degree: Optional[int] = None) -> dict:
    """Deprecated alias of :func:`analyze_defect_potential`."""
    warnings.warn(
        "analyze_primary_loop_potential is renamed analyze_defect_potential.",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyze_defect_potential(G, max_degree=max_degree)
