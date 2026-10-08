"""Two straight segments: how close they come, and whether one passed through the other.

Plain geometry shared by the atomistic backbone clearance
(:func:`topon.conformation.atomistic.clear_backbones`), which must not move
a bond through another while it opens them up, and the crossing detector
(:mod:`topon.analysis.crossings`), which watches a trajectory for exactly
that. Both read a bond as the straight segment between its two atoms.

Between two configurations every point is taken to move in a straight line.
Four points ``p1 p2`` (one segment) and ``q1 q2`` (the other) are coplanar
when the signed volume ``(p2 - p1) x (q1 - p1) . (q2 - p1)`` is zero, a cubic
in the interpolation parameter. A passage is a root of that cubic in [0, 1]
at which the two segments actually meet, which is checked by their closest
distance at that moment. The cubic is sampled and each sign change bisected,
so a double root (two segments that touch and part on the same side) is not
counted, which is the right answer.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Closest distance (A) below which two coplanar segments count as meeting.
MEET = 1e-3


def segment_closest(p1, p2, q1, q2):
    """Closest points of pairs of segments, vectorised (Ericson's method).

    All four are ``(m, 3)``. Returns ``(s, t, distance)``, the closest
    points being ``p1 + s (p2 - p1)`` and ``q1 + t (q2 - q1)``.
    """
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12,
                 np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1), 0, 1), 0.0)
    t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1), 0, 1)
    s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1), 0, 1)
    dist = np.linalg.norm((p1 + d1 * s[:, None]) - (q1 + d2 * t[:, None]), axis=1)
    return s, t, dist


def signed_volume(p1, p2, q1, q2):
    return np.einsum("ij,ij->i", np.cross(p2 - p1, q1 - p1), q2 - p1)


def passage_times(A, B, samples: int = 9, iters: int = 40):
    """For each candidate pair, the tau in [0, 1] of a passage, or nan.

    ``A`` and ``B`` are ``(4, m, 3)``: p1, p2, q1, q2 before and after.
    """
    ts = np.linspace(0.0, 1.0, samples)
    vol = np.stack([signed_volume(*(A + t * (B - A))) for t in ts])   # (samples, m)
    out = np.full(vol.shape[1], np.nan)
    for k in range(samples - 1):
        lo_v, hi_v = vol[k], vol[k + 1]
        # Half-open, so a root that falls exactly on a sample is found once,
        # in the interval it leaves from.
        flip = (((lo_v <= 0) & (hi_v > 0)) | ((lo_v >= 0) & (hi_v < 0))) & np.isnan(out)
        if not flip.any():
            continue
        idx = np.flatnonzero(flip)
        lo = np.full(len(idx), ts[k])
        hi = np.full(len(idx), ts[k + 1])
        s_hi = np.sign(hi_v[idx])
        a, b = A[:, idx], B[:, idx]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            v = signed_volume(*(a + mid[None, :, None] * (b - a)))
            like_hi = np.sign(v) == s_hi
            hi = np.where(like_hi, mid, hi)
            lo = np.where(like_hi, lo, mid)
        tau = 0.5 * (lo + hi)
        at = a + tau[None, :, None] * (b - a)
        meet = segment_closest(*at)[2] < MEET
        out[idx[meet]] = tau[meet]
    return out


def bond_ends(x, rows, box):
    """Each bond's two ends, the second in the image nearest the first.

    ``rows`` is ``(m, 2)`` row indices into ``x``. topon's data files carry
    no image flags, so a bond that crosses the boundary can have its two
    atoms a box apart in unwrapped coordinates.
    """
    a = x[rows[:, 0]]
    d = x[rows[:, 1]] - a
    return a, a + d - box * np.round(d / box)


def segments_through_ring(p0, p1, ring) -> np.ndarray:
    """Which of the segments ``p0[k]``-``p1[k]`` pass through ``ring``, vectorised.

    ``ring`` is ``(r, 3)``, its atoms in ring order, read as the fan of
    triangles from its centroid, the surface it spans (as
    :func:`topon.simbox.molecule.segment_through_ring` reads it, here
    for many segments at once). ``p0`` and ``p1`` are ``(m, 3)``.
    """
    p0 = np.asarray(p0, float).reshape(-1, 3)
    p1 = np.asarray(p1, float).reshape(-1, 3)
    ring = np.asarray(ring, float)
    out = np.zeros(len(p0), bool)
    if not len(p0):
        return out
    centre = ring.mean(axis=0)
    d = p1 - p0
    for k in range(len(ring)):
        a, b = centre, ring[k]
        c = ring[(k + 1) % len(ring)]
        e1, e2 = b - a, c - a
        h = np.cross(d, e2)
        det = h @ e1
        ok = np.abs(det) >= 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = p0 - a
        u = np.einsum("ij,ij->i", s, h) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,ij->i", d, q) * inv
        t = (q @ e2) * inv
        out |= ok & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
    return out


def segments_through_rings(p0, p1, rings) -> np.ndarray:
    """Whether each segment ``p0[k]``-``p1[k]`` passes through ``rings[k]``.

    Pairwise, many rings at once: ``rings`` is ``(m, r, 3)``, each ring's
    atoms in ring order, read as :func:`segments_through_ring` reads one (the
    fan of triangles from its centroid). ``p0`` and ``p1`` are ``(m, 3)``
    (for the rings an atomistic backbone runs through).
    """
    p0 = np.asarray(p0, float).reshape(-1, 3)
    p1 = np.asarray(p1, float).reshape(-1, 3)
    rings = np.asarray(rings, float).reshape(len(p0), -1, 3)
    out = np.zeros(len(p0), bool)
    if not len(p0):
        return out
    centre = rings.mean(axis=1)
    d = p1 - p0
    for k in range(rings.shape[1]):
        e1 = rings[:, k] - centre
        e2 = rings[:, (k + 1) % rings.shape[1]] - centre
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", h, e1)
        ok = np.abs(det) >= 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = p0 - centre
        u = np.einsum("ij,ij->i", s, h) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", q, e2) * inv
        out |= ok & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= 1.0)
    return out


def passing_pairs(before, after, rows, atom_ids, box, fixed=None):
    """Pairs of bonds that pass through each other between two configurations.

    ``before`` and ``after`` are ``(n, 3)`` positions, ``rows`` the bonds as
    ``(m, 2)`` row indices and ``atom_ids`` what tells two bonds sharing an
    atom (``(m, 2)``, any labels). Bonds that share an atom are never a pair,
    and neither are two of the bonds ``fixed`` marks (``(m,)`` bool: bonds
    that do not move, such as a cage's). Returns ``(i, j, tau)``: bond
    indices and where between the two the passage is, plus the largest
    displacement.
    """
    box = np.asarray(box, float).reshape(3)
    pa, pb = bond_ends(before, rows, box)
    fa, fb = bond_ends(after, rows, box)
    disp = np.linalg.norm(after - before, axis=1)
    big = float(disp.max()) if len(disp) else 0.0
    mid = 0.5 * (pa + pb)
    half = 0.5 * np.linalg.norm(pb - pa, axis=1)
    reach = 2.0 * float(half.max()) + 2.0 * big + 0.5
    w = mid - box * np.floor(mid / box)
    w = np.clip(w, 0.0, np.nextafter(box, 0.0))
    pairs = cKDTree(w, boxsize=box).query_pairs(reach, output_type="ndarray")
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), big)
    if not len(pairs):
        return empty
    i, j = pairs[:, 0], pairs[:, 1]
    ids = np.asarray(atom_ids)
    share = ((ids[i, 0] == ids[j, 0]) | (ids[i, 0] == ids[j, 1])
             | (ids[i, 1] == ids[j, 0]) | (ids[i, 1] == ids[j, 1]))
    if fixed is not None:
        share = share | (fixed[i] & fixed[j])
    i, j = i[~share], j[~share]
    # the second bond in the image nearest the first, the same shift at both ends
    shift = -box * np.round((mid[j] - mid[i]) / box)
    A = np.stack([pa[i], pb[i], pa[j] + shift, pb[j] + shift])
    B = np.stack([fa[i], fb[i], fa[j] + shift, fb[j] + shift])
    tau = passage_times(A, B)
    hit = ~np.isnan(tau)
    return i[hit], j[hit], tau[hit], big
