"""Backbone paths for entangled edges, by either construction.

One function serves the pipeline and both canonical workflow modules, which
until now each carried their own copy of the kink loop:

    paths = entangled_backbone_paths(graph, dims, edge_atoms,
                                     method="waypoint", kink_params={...})
    # paths[(u, v, key)] -> [xyz, ...], one per chain atom, junctions excluded

Two constructions:

``waypoint`` (the default) is the prescribed winding of
:mod:`topon.conformation.entanglement.waypoints`: the two chains of a pair
are drawn together as splines that spiral about their contact in antiphase,
so the pair carries exactly ``entanglement_count`` windings by construction.
Verified with primitive-path analysis: the requested count is
delivered through the full protocol, and nothing appears on pairs that were
not asked.

``kink`` is the legacy Gaussian bump aimed at the partner's midpoint
(:func:`topon.utils.network_helpers.calculate_entangled_kink`). Each chain
is drawn alone, so what the pair carries after relaxation is statistical
rather than prescribed. Kept for comparison with systems built by 0.1.0.
"""
from __future__ import annotations

import numpy as np

from topon.conformation.entanglement.waypoints import (
    Site,
    entangled_pair,
    resample_path,
    winding_waypoints,
)
from topon.utils.network_helpers import calculate_entangled_kink


#: A braid's radius is at most this share of the distance from its axis to
#: the nearest other chord beside its span.
CHORD_SHARE = 0.5

#: ... and at least this fraction of the pair's shorter chord.
RADIUS_FLOOR = 0.05


def _chords(graph, dims):
    """Every strand's chord that has two ends: ``(edge key, start, end)``."""
    out = []
    multi = graph.is_multigraph()
    edges = graph.edges(keys=True) if multi else ((u, v, 0) for u, v in graph.edges())
    for u, v, k in edges:
        if u == v:
            continue
        p0 = np.asarray(graph.nodes[u].get("pos", (0.0,) * 3), float)
        p1 = p0 + _mic(np.asarray(graph.nodes[v].get("pos", (0.0,) * 3), float) - p0, dims)
        out.append(((u, v, k), p0, p1))
    return out


def _room_beside(chords, skip, mid, axis, half, dims) -> float:
    """How close another strand's chord comes to a braid's axis beside its span.

    The braid is a cylinder about the axis over ``mid +- half * axis``; past
    its ends the two strands run back to their chords. Each chord but the
    pair's own (``skip``) is cut to the stretch beside the span, and its
    least distance from the axis line is the room the braid has there. A
    strand whose chord passes inside the braid's radius runs between the two
    arms, and turning it about its chord cannot take it out
    (:func:`topon.conformation.atomistic.clear_braids`); one whose chord passes
    the axis only beyond the span, as the strands joining the partners'
    junctions do on a one-shell lattice, is not beside the braid at all.
    """
    rows = [(p0, p1) for key, p0, p1 in chords if key not in skip]
    if not rows:
        return float("inf")
    p0 = np.array([r[0] for r in rows], float)
    p1 = np.array([r[1] for r in rows], float)
    if dims is not None:
        d = np.asarray(dims, float)
        shift = -d * np.round((0.5 * (p0 + p1) - mid) / d)
        p0, p1 = p0 + shift, p1 + shift
    d0 = p0 - mid
    dd = p1 - p0
    u0, du = d0 @ axis, dd @ axis
    # the stretch of each chord with |u| <= half
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = np.where(np.abs(du) > 1e-12, (-half - u0) / du, -np.inf)
        tb = np.where(np.abs(du) > 1e-12, (half - u0) / du, np.inf)
    lo = np.where(np.abs(du) > 1e-12, np.minimum(ta, tb), np.where(np.abs(u0) <= half, 0.0, 2.0))
    hi = np.where(np.abs(du) > 1e-12, np.maximum(ta, tb), np.where(np.abs(u0) <= half, 1.0, -1.0))
    lo, hi = np.maximum(lo, 0.0), np.minimum(hi, 1.0)
    ok = lo <= hi
    if not ok.any():
        return float("inf")
    # least distance from the axis line over that stretch, a quadratic in t
    w0 = d0 - np.outer(u0, axis)
    dw = dd - np.outer(du, axis)
    denom = np.einsum("ij,ij->i", dw, dw)
    t = np.where(denom > 1e-12, -np.einsum("ij,ij->i", w0, dw) / np.maximum(denom, 1e-12), lo)
    t = np.clip(t, lo, hi)
    r = np.linalg.norm(w0 + t[:, None] * dw, axis=1)
    return float(r[ok].min())


def _mic(vec, dims):
    if dims is None:
        return np.asarray(vec, float)
    d = np.asarray(dims, float)
    v = np.asarray(vec, float)
    return v - d * np.round(v / d)


def _perp_of(mic):
    """A deterministic vector perpendicular to the chord, for the rare case
    of a partner whose midpoint coincides with the edge's own."""
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(mic)))] = 1.0
    p = np.cross(mic, axis)
    n = np.linalg.norm(p)
    return p / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _kink_path(pos_u, mic, n_atoms, orient_vec, count, kink_params):
    """The legacy per-edge kink, exactly as the pipeline drew it."""
    kink_dict = calculate_entangled_kink(
        start_pos=np.zeros(3),
        end_pos=mic,
        num_atoms=n_atoms + 2,          # N+2 fix
        params=kink_params,
        orientation_vec=orient_vec,
        z_phase=1.0,
        num_entanglements=count,
    )
    full = [kink_dict[k] for k in sorted(kink_dict.keys())]
    return [pos_u + np.array(pt) for pt in full[1:-1]]


def entangled_backbone_paths(graph, dims, edge_atoms, method="waypoint",
                             kink_params=None, windings=None, sites=None):
    """Interior bead positions for every edge carrying ``entangled_with``.

    ``edge_atoms`` maps ``(u, v, key)`` to that edge's chain atoms (only the
    length is used). Returns ``{edge_key: [xyz, ...]}`` with one position per
    atom, junction ends excluded, for exactly the edges that are entangled;
    the caller places every other chain however it already does.

    The partner's chord is taken in the image nearest the edge's own
    midpoint, which is the same convention the kink always used and what
    makes a pair across the periodic boundary wind rather than reach across
    the box.

    ``windings`` draws every pair at that many turns instead of its
    ``entanglement_count``, over the span its own count would have taken.
    ``windings=0`` is the reference a delivered winding is read against
    (:mod:`topon.analysis.windings`): the same pair with the same site, each
    chain passing its partner on its own side. Only the waypoint
    construction has a winding count to set.

    ``sites``, a dict, receives each waypoint pair's braid under both of its
    edge keys: ``{"mid", "axis", "half", "radius"}`` in the units of the node
    positions, ``mid`` in the image of the pair's first edge. The atomistic
    placement keeps the other strands out of it
    (:func:`topon.conformation.atomistic.clear_braids`).
    """
    if windings is not None and method != "waypoint":
        raise ValueError("windings= is for the waypoint construction; the kink "
                         "has no winding count to set")
    kink_params = kink_params or {}
    out = {}
    seen_pairs = set()
    chords = _chords(graph, dims) if method == "waypoint" else []

    multi = graph.is_multigraph()
    for edge_key, atoms in edge_atoms.items():
        u, v, key = edge_key
        data = graph[u][v][key] if multi else graph[u][v]
        partner = data.get("entangled_with")
        if partner is None or edge_key in out:
            continue

        pos_u = np.asarray(graph.nodes[u].get("pos", (0.0,) * 3), float)
        pos_v = np.asarray(graph.nodes[v].get("pos", (0.0,) * 3), float)
        mic = _mic(pos_v - pos_u, dims)
        count = int(data.get("entanglement_count", 1))

        p_u, p_v = partner[0], partner[1]
        p_key = partner[2] if len(partner) > 2 else 0
        p_pos_u = np.asarray(graph.nodes[p_u].get("pos", (0.0,) * 3), float)
        p_pos_v = np.asarray(graph.nodes[p_v].get("pos", (0.0,) * 3), float)
        p_mic = _mic(p_pos_v - p_pos_u, dims)

        my_mid = pos_u + 0.5 * mic
        delta = _mic((p_pos_u + 0.5 * p_mic) - my_mid, dims)
        b0 = my_mid + delta - 0.5 * p_mic       # partner chord, my image
        b1 = b0 + p_mic

        if method == "kink":
            orient = (delta if np.linalg.norm(delta) >= 0.01
                      else _perp_of(mic))
            out[edge_key] = _kink_path(pos_u, mic, len(atoms), orient,
                                       count, kink_params)
            continue

        # Waypoint: the pair is drawn together, once. Both edges get their
        # paths here; the partner's entry is filled so the caller's loop
        # finds it whichever edge it reaches first.
        pair_id = frozenset([edge_key, (p_u, p_v, p_key)])
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)

        p_atoms = edge_atoms.get((p_u, p_v, p_key))
        if p_atoms is None:
            # Partner edge not built (filtered dangling end): nothing to
            # wind around, fall back to the single-chain kink.
            orient = (delta if np.linalg.norm(delta) >= 0.01
                      else _perp_of(mic))
            out[edge_key] = _kink_path(pos_u, mic, len(atoms), orient,
                                       count, kink_params)
            continue

        na, nb = len(atoms) + 2, len(p_atoms) + 2
        site = Site(at=0.5, turns=count)
        # A braid is sized to the room it has. Its radius is a share of the
        # gap between the two chords, and a third strand whose chord passes
        # closer to the braid's axis than that, beside its span, runs between
        # the two arms and shares the winding; no turn of that strand about
        # its own chord takes it out.
        wa, _wb, mid, axis, half = winding_waypoints(pos_u, pos_u + mic, b0, b1, site)
        off = wa - mid
        r0 = float(np.linalg.norm(off - np.outer(off @ axis, axis), axis=1).max())
        room = _room_beside(chords, {edge_key, (p_u, p_v, p_key)}, mid, axis, half, dims)
        if r0 > CHORD_SHARE * room:
            shorter = min(float(np.linalg.norm(mic)), float(np.linalg.norm(p_mic)))
            site = Site(at=0.5, turns=count,
                        radius=min(r0, max(CHORD_SHARE * room, RADIUS_FLOOR * shorter)))
        if windings is not None:
            # The site its own count sizes, drawn at the turns asked for.
            *_rest, half = winding_waypoints(pos_u, pos_u + mic, b0, b1, site)
            chord = float(np.linalg.norm(mic))
            site = Site(at=0.5, turns=int(windings), radius=site.radius,
                        span=2.0 * half / chord if chord > 0 else None)
        pa, pb, _info = entangled_pair(
            pos_u, pos_u + mic, b0, b1, [site],
            n_beads=max(na, nb))
        if sites is not None:
            wa, _wb, mid, axis, half = winding_waypoints(
                pos_u, pos_u + mic, b0, b1, site)
            off = wa - mid
            off = off - np.outer(off @ axis, axis)
            braid = {"mid": np.asarray(mid, float), "axis": np.asarray(axis, float),
                     "half": float(half),
                     "radius": float(np.linalg.norm(off, axis=1).max())}
            sites[edge_key] = braid
            sites[(p_u, p_v, p_key)] = braid
        # Each chain at its own bead count: the pair is drawn at one density
        # and re-placed by arc length, so unequal DP costs nothing.
        pa = resample_path(pa, na)
        pb = resample_path(pb, nb)
        out[edge_key] = [np.array(q) for q in pa[1:-1]]
        out[(p_u, p_v, p_key)] = [np.array(q) for q in pb[1:-1]]

    return out
