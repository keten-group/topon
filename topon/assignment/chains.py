"""Chains through the junctions of a sculpted graph (random crosslinking).

An end-linked network is its graph: every edge is a chain and every junction
a crosslinker. In a randomly crosslinked one the chains are longer than the
strands, and each chain passes through the junctions its crosslinks sit on.
The graph is the same kind of object (junctions, strands between them,
dangling chain ends as degree-1 nodes), so the lattice route can sculpt it,
and what is left to decide is which strands make up which chain and how many
beads each strand gets.

The chains come from a path cover of the strands. At every junction the
strands meeting there are paired up, each pair being one chain passing
through; following the pairs from a dangling end leads through junctions to
another dangling end, and that trail is one chain. A junction of
functionality 4 is two chains passing once each, or one chain passing twice
(an intra-chain crosslink; a self-loop is one).

1. pair the strand ends at every junction at random
2. follow each trail from a dangling end; the strands no trail reached form
   closed trails, rings with no end, and each is merged into a trail that
   shares a junction with it by re-pairing there (either re-pairing merges)
3. shape how many junctions each chain passes by exchanging tails: at a
   junction two different chains pass, re-pair so the chains swap what
   follows, kept when the histogram of passes per chain comes no further
   from the target and never when a chain would pass more junctions than it
   has reactive beads. The default target is the binomial of random
   crosslinking over the chain's reactive beads, with the graph's own mean
4. give each chain its beads: as many crosslinked beads as it passes
   junctions, drawn uniformly from its reactive beads with no two on
   neighbouring beads and none next to a chain end, and each strand the
   beads between them (the chain's two end beads are its end nodes, and a
   primary loop keeps at least :data:`LOOP_MIN_DP`)

The graph's degrees and cycles, and so every descriptor of
:mod:`topon.analysis.descriptors`, are untouched; only the edges' ``dp``,
``chain`` and ``chain_index`` and ``G.graph["chains"]`` are written, in the
layout :mod:`topon.analysis.crosslinked` reads a reference into.
"""
from __future__ import annotations

import collections
import math
from typing import Optional, Sequence

import numpy as np


#: Design bond of the Kremer-Grest build, in sigma (``CGWriter``'s R0), the
#: length the chord floor spans a strand's chord with.
KG_BOND = 0.97

#: Fewest beads of a primary loop. The coarse-grained builder bonds a loop's
#: first and last bead to its junction, which for a single bead is the same
#: bond twice.
LOOP_MIN_DP = 2


class ChainAssignmentError(ValueError):
    """The strands cannot be covered by chains of the requested length."""


def reactive_positions(dp: int, every: int = 1) -> list[int]:
    """Bead indices that may carry a crosslink on a chain of ``dp`` beads.

    Interior beads only (0 and ``dp - 1`` are the chain ends), every
    ``every``-th one starting at bead 1.
    """
    if dp < 3:
        raise ValueError(f"a chain of {dp} beads has no interior bead")
    return list(range(1, dp - 1, max(1, int(every))))


def crosslink_sites(dp: int, every: int = 1) -> list[int]:
    """The reactive beads a crosslink may take: bead 1 and ``dp - 2`` excluded.

    A crosslink next to a chain end would leave the end bead as a dangling
    strand of no beads, and the coarse-grained builder bonds a dangling
    strand's end bead to its junction only through the strand's own beads,
    so the split never puts one there.
    """
    return [p for p in reactive_positions(dp, every) if 2 <= p <= dp - 3]


def max_passes(dp: int, every: int = 1) -> int:
    """Most crosslinks a chain of ``dp`` beads can carry.

    No two on neighbouring beads (they would be one junction) and none next
    to a chain end (:func:`crosslink_sites`).
    """
    k = 0
    while _slack(dp, every, [1] * (k + 2)) is not None:   # k + 1 crosslinks
        k += 1
    return k


def binomial_target(mean: float, n: int, kmax: int) -> np.ndarray:
    """P(passes = k) for k = 1..kmax: a binomial over ``n`` sites, k >= 1.

    Chains with no crosslink are sol and not in the graph, so the binomial
    is conditioned on at least one crosslink and cut at ``kmax``; its
    probability is solved for so that the conditioned mean is ``mean``.
    """
    if not 1.0 <= mean <= kmax:
        raise ValueError(f"mean passes {mean:.3f} outside 1 to {kmax}")
    k = np.arange(0, n + 1)
    logc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1)
                     - math.lgamma(n - i + 1) for i in k])

    def pmf(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        w = np.exp(logc + k * np.log(p) + (n - k) * np.log1p(-p))
        w[0] = 0.0
        w[kmax + 1:] = 0.0
        return w / w.sum()

    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (pmf(mid) * k).sum() < mean:
            lo = mid
        else:
            hi = mid
    return pmf(0.5 * (lo + hi))[1:kmax + 1]


def _other(h):
    return (h[0], 1 - h[1])


class _Cover:
    """Half-edge pairings at the junctions and the trails they make.

    A half-edge is ``(edge index, side)``, side 0 at the edge's first node
    and 1 at its second. A trail is the list of half-edges it enters its
    strands by, in order, from one chain end to the other.
    """

    def __init__(self, G, rng):
        if G.is_multigraph():
            self.edges = [(u, v, k) for u, v, k in G.edges(keys=True)]
        else:
            self.edges = [(u, v, None) for u, v in G.edges()]
        self.node_of = {}
        self.at = collections.defaultdict(list)
        for i, (u, v, _k) in enumerate(self.edges):
            self.node_of[(i, 0)] = u
            self.node_of[(i, 1)] = v
            self.at[u].append((i, 0))
            self.at[v].append((i, 1))
        self.end_half = {}
        odd = []
        for n, hs in self.at.items():
            if len(hs) == 1:
                self.end_half[n] = hs[0]
            elif len(hs) % 2:
                odd.append(n)
        if odd:
            raise ChainAssignmentError(
                f"{len(odd)} junctions have an odd number of strand ends "
                f"(e.g. node {odd[0]} with {len(self.at[odd[0]])}). A chain "
                f"passing a junction takes two, so an odd degree needs a "
                f"chain to end inside the junction, which a crosslink "
                f"between two chain beads never does.")
        self.partner = {}
        for n in sorted(self.at, key=str):
            if n in self.end_half:
                continue
            hs = list(self.at[n])
            order = rng.permutation(len(hs))
            hs = [hs[i] for i in order]
            for a, b in zip(hs[0::2], hs[1::2]):
                self.partner[a] = b
                self.partner[b] = a
        self.trails: list = []
        self.where: dict = {}          # edge index -> (trail id, position)

    # --- trails ------------------------------------------------------------

    def _walk_open(self, h0):
        path, h = [], h0
        while True:
            path.append(h)
            far = _other(h)
            if self.node_of[far] in self.end_half:
                return path
            h = self.partner[far]

    def _walk_closed(self, h0):
        path, h = [], h0
        while True:
            path.append(h)
            h = self.partner[_other(h)]
            if h == h0:
                return path

    def build(self):
        """Open trails from the chain ends; closed trails from what is left."""
        seen = set()
        opened = []
        for n in sorted(self.end_half, key=str):
            h0 = self.end_half[n]
            if h0[0] in seen:
                continue
            path = self._walk_open(h0)
            seen.update(h[0] for h in path)
            opened.append(path)
        closed = []
        for i in range(len(self.edges)):
            if i not in seen:
                path = self._walk_closed((i, 0))
                seen.update(h[0] for h in path)
                closed.append(path)
        return opened, closed

    def close_rings(self) -> int:
        """Merge every closed trail into an open one; the merges made."""
        merges = 0
        while True:
            opened, closed = self.build()
            if not closed:
                self._index(opened)
                return merges
            in_open = {h[0] for t in opened for h in t}
            done = False
            for ring in closed:
                for h in ring:
                    # the ring enters edge h[0] from the node of h; it arrived
                    # there by the half-edge paired with h
                    arrive = self.partner[h]
                    n = self.node_of[h]
                    x = next((y for y in self.at[n]
                              if y[0] in in_open and y in self.partner), None)
                    if x is None:
                        continue
                    y = self.partner[x]
                    self.partner[arrive], self.partner[x] = x, arrive
                    self.partner[h], self.partner[y] = y, h
                    merges += 1
                    done = True
                    break
                if done:
                    break
            if not done:
                # no ring meets an open trail: join two rings if any meet
                ring_of = {h[0]: r for r, ring in enumerate(closed) for h in ring}
                for r, ring in enumerate(closed):
                    for h in ring:
                        n = self.node_of[h]
                        x = next((y for y in self.at[n]
                                  if ring_of.get(y[0], r) != r), None)
                        if x is None:
                            continue
                        arrive, y = self.partner[h], self.partner[x]
                        self.partner[arrive], self.partner[x] = x, arrive
                        self.partner[h], self.partner[y] = y, h
                        merges += 1
                        done = True
                        break
                    if done:
                        break
            if not done:
                raise ChainAssignmentError(
                    f"{len(closed)} rings of strands share no junction with "
                    f"any chain end: a component of the graph has no "
                    f"dangling end, so its strands cannot be linear chains")

    def _index(self, opened):
        self.trails = [list(t) for t in opened]
        self.where = {}
        for t, path in enumerate(self.trails):
            for p, h in enumerate(path):
                self.where[h[0]] = (t, p)

    def _reindex(self, t):
        for p, h in enumerate(self.trails[t]):
            self.where[h[0]] = (t, p)

    def pass_at(self, x):
        """``(trail, i)``: the pass of half-edge ``x`` is between strands i, i+1."""
        t, p = self.where[x[0]]
        h = self.trails[t][p]
        return (t, p - 1) if h == x else (t, p)

    # --- tail exchange -----------------------------------------------------

    def exchange(self, x, y, mode):
        """Swap tails at the passes of ``x`` and ``y`` (different trails)."""
        t1, i = self.pass_at(x)
        t2, j = self.pass_at(y)
        T1, T2 = self.trails[t1], self.trails[t2]
        far_i, a = _other(T1[i]), T1[i + 1]
        far_j, b = _other(T2[j]), T2[j + 1]
        if mode == "B":
            self.partner[far_i], self.partner[b] = b, far_i
            self.partner[far_j], self.partner[a] = a, far_j
            n1 = T1[:i + 1] + T2[j + 1:]
            n2 = T2[:j + 1] + T1[i + 1:]
        else:
            self.partner[far_i], self.partner[far_j] = far_j, far_i
            self.partner[a], self.partner[b] = b, a
            n1 = T1[:i + 1] + [_other(h) for h in reversed(T2[:j + 1])]
            n2 = [_other(h) for h in reversed(T1[i + 1:])] + T2[j + 1:]
        self.trails[t1], self.trails[t2] = n1, n2
        self._reindex(t1)
        self._reindex(t2)


def _slack(dp: int, every: int, mins: Sequence[int]):
    """``(slack, first gap, gaps)`` of a split in reactive-bead units, or None.

    Positions are ``1 + every * q``. The first crosslink needs ``q`` of at
    least the first gap, each later one ``q`` beyond the previous by its
    gap, and the last may go no further than the last strand's minimum
    allows; the slack is what is left to spread. Dangling strands keep at
    least one bead (:func:`crosslink_sites`).
    """
    s = max(1, int(every))
    n_pos = len(reactive_positions(dp, s))
    g0 = -(-max(int(mins[0]), 1) // s)
    gaps = [-(-(max(int(m), 1) + 1) // s) for m in mins[1:-1]]
    qmax = min(n_pos - 1, (dp - 3 - max(int(mins[-1]), 1)) // s)
    slack = qmax - g0 - sum(gaps)
    return None if slack < 0 else (slack, g0, gaps)


def split_chain(rng, dp: int, every: int, mins: Sequence[int]) -> Optional[list[int]]:
    """Crosslinked bead positions of one chain, uniform under minimum DPs.

    ``mins`` holds the fewest beads each strand of the chain may carry, in
    chain order (the first and last are the dangling ends); there are
    ``len(mins) - 1`` crosslinks. Every strand gets at least one bead: two
    crosslinks on neighbouring beads are one junction, and a dangling strand
    of none leaves its end bead unbonded in the builder. The
    positions are drawn uniformly over every placement on the reactive
    beads that meets the minimums: in units of the reactive spacing, the
    gaps above their minimums are a composition of the slack, drawn by
    stars and bars. Returns None when no placement meets them.
    """
    s = max(1, int(every))
    k = len(mins) - 1
    if k == 0:
        return []
    got = _slack(dp, s, mins)
    if got is None:
        return None
    slack, g0, gaps = got
    bars = np.sort(rng.choice(slack + k, size=k, replace=False))
    x = [int(bars[0])] + [int(bars[i + 1] - bars[i] - 1) for i in range(k - 1)]
    q = [g0 + x[0]]
    for i in range(k - 1):
        q.append(q[-1] + gaps[i] + x[i + 1])
    return [1 + s * qi for qi in q]


def assign_chains(G, dp: int, rng, *, reactive_every: int = 1,
                  target: Optional[Sequence[float]] = None,
                  sweeps: int = 200,
                  bonds_per_unit: Optional[float] = None) -> dict:
    """Cover the strands of ``G`` with chains of ``dp`` beads and write them.

    ``G`` is a sculpted graph with its defects (self-loops and parallel
    strands are fine): degree-1 nodes are chain ends, degree-0 nodes empty
    sites and ignored, every other node a junction. ``rng`` is a
    ``numpy.random.Generator``, the only source of randomness. ``dp`` is the
    beads per chain. ``reactive_every`` spaces the beads that may carry a
    crosslink (1, every interior bead). ``target`` is P(passes = k) for
    k = 1, 2, ...; unset, the binomial of :func:`binomial_target` with the
    graph's mean passes per chain. ``sweeps`` bounds the tail exchanges at
    that many tries per chain. ``bonds_per_unit`` is the number of bond
    lengths one unit of node position spans; given, every strand gets at
    least the beads a path of such bonds needs to span its chord
    (``dp + 1 >= chord * bonds_per_unit``, minimum image in
    ``G.graph["box"]``), the positions drawn uniformly among the placements
    that meet every minimum (:func:`split_chain`). A chain whose chords do
    not all fit on ``dp`` beads is split without them and counted.

    Writes ``dp``, ``chain`` (1, 2, ...) and ``chain_index`` on every strand
    and ``G.graph["chains"]`` (per chain its ``dp``, its strands as
    ``(u, v, key)`` in order, the junctions it passes and ``passes``), and
    returns a record: chains, passes histogram before and after the
    exchanges against the target, exchanges tried and kept, rings merged.

    Raises:
        ChainAssignmentError: a junction of odd degree, a component with no
            chain end, or more passes than the chains' reactive beads carry.
    """
    sites = crosslink_sites(int(dp), reactive_every)
    kmax = max_passes(int(dp), reactive_every)
    cover = _Cover(G, rng)
    n_chains = len(cover.end_half) // 2
    if n_chains == 0:
        raise ChainAssignmentError("the graph has no chain end (degree-1 node)")
    merges = cover.close_rings()
    total = sum(len(t) - 1 for t in cover.trails)
    mean = total / n_chains
    if mean > kmax:
        raise ChainAssignmentError(
            f"{total} junction passes over {n_chains} chains is {mean:.2f} per "
            f"chain; a chain of {dp} beads with reactive_every "
            f"{reactive_every} carries at most {kmax}")
    if target is None:
        pmf = binomial_target(mean, len(sites), kmax)
    else:
        pmf = np.asarray(target, float)[:kmax]
        if (pmf < 0).any() or pmf.sum() <= 0:
            raise ChainAssignmentError(
                f"the passes target {list(target)} is not a distribution over "
                f"1 to {kmax} junctions per chain")
        pmf = pmf / pmf.sum()
    want = n_chains * np.concatenate([[0.0], pmf])       # index = passes

    def hist():
        h = np.zeros(max(kmax + 1, max(len(t) for t in cover.trails)), float)
        for t in cover.trails:
            h[len(t) - 1] += 1
        return h

    def distance(h):
        w = np.zeros(len(h))
        w[:min(len(h), len(want))] = want[:len(h)]
        # A chain over kmax cannot be built. The penalty grows with the
        # excess, so a long chain is worn down one exchange at a time.
        excess = (h[kmax + 1:] * np.arange(1, len(h) - kmax)).sum()
        return float(np.abs(h - w).sum() + 1e6 * excess)

    h = hist()
    before = h.copy()
    d = distance(h)
    tried = kept = 0
    junction_halves = [hs for n, hs in cover.at.items()
                       if n not in cover.end_half and len(hs) >= 4]
    budget = int(sweeps) * n_chains if junction_halves else 0
    for _ in range(budget):
        hs = junction_halves[int(rng.integers(len(junction_halves)))]
        x = hs[int(rng.integers(len(hs)))]
        px = cover.partner[x]
        y = hs[int(rng.integers(len(hs)))]
        if y in (x, px):
            continue
        t1, i = cover.pass_at(x)
        t2, j = cover.pass_at(y)
        if t1 == t2:
            continue
        L1, L2 = len(cover.trails[t1]), len(cover.trails[t2])
        mode = "A" if rng.random() < 0.5 else "B"
        if mode == "B":
            new = ((i + 1) + (L2 - j - 1), (j + 1) + (L1 - i - 1))
        else:
            new = ((i + 1) + (j + 1), (L1 - i - 1) + (L2 - j - 1))
        tried += 1
        h2 = h.copy()
        if max(new) - 1 >= len(h2):
            h2 = np.concatenate([h2, np.zeros(max(new) - len(h2))])
        h2[L1 - 1] -= 1
        h2[L2 - 1] -= 1
        h2[new[0] - 1] += 1
        h2[new[1] - 1] += 1
        d2 = distance(h2)
        if d2 <= d:
            cover.exchange(x, y, mode)
            h, d = h2, d2
            kept += 1
    longest = max(len(t) - 1 for t in cover.trails)
    if longest > kmax:
        raise ChainAssignmentError(
            f"a chain passes {longest} junctions after the tail exchanges and "
            f"one of {dp} beads carries at most {kmax}")

    box = G.graph.get("box")
    box = None if box is None else np.asarray(box, float).reshape(3)

    def floor_dp(u, v):
        """Fewest beads that span the strand's chord with bonds of length 1."""
        if bonds_per_unit is None or u == v:
            return 0
        pu, pv = G.nodes[u].get("pos"), G.nodes[v].get("pos")
        if pu is None or pv is None:
            return 0
        d = np.asarray(pu, float) - np.asarray(pv, float)
        if box is not None:
            d -= box * np.round(d / box)
        return max(0, int(math.ceil(np.linalg.norm(d) * bonds_per_unit - 1e-9)) - 1)

    chains = {}
    unmet = 0
    for c, path in enumerate(cover.trails, start=1):
        k = len(path) - 1
        ends = [cover.edges[hh[0]][:2] for hh in path]
        least = [LOOP_MIN_DP if u == v else 0 for u, v in ends]
        mins = [max(m, floor_dp(u, v)) for m, (u, v) in zip(least, ends)]
        picks = split_chain(rng, int(dp), reactive_every, mins)
        if picks is None:
            unmet += 1
            picks = split_chain(rng, int(dp), reactive_every, least)
        if picks is None:
            raise ChainAssignmentError(
                f"chain {c} passes {k} junctions with "
                f"{sum(1 for u, v in ends if u == v)} primary loops, which "
                f"need {LOOP_MIN_DP} beads each, more than {dp} beads can hold")
        cuts = [0] + picks + [int(dp) - 1]
        strands, junctions = [], []
        for idx, hh in enumerate(path):
            u, v, key = cover.edges[hh[0]]
            n_beads = cuts[idx + 1] - cuts[idx] - 1
            data = G.edges[u, v, key] if key is not None else G.edges[u, v]
            data["dp"] = int(n_beads)
            data["chain"] = c
            data["chain_index"] = idx
            strands.append((u, v, key))
            if idx < k:
                junctions.append(cover.node_of[_other(hh)])
        chains[c] = {"dp": int(dp), "strands": strands,
                     "junctions": junctions, "passes": k,
                     "junction_beads": k}
    G.graph["chains"] = chains
    G.graph["architecture"] = "random_crosslinked"
    after = hist()

    def as_dict(v):
        return {int(i): int(c) for i, c in enumerate(v) if c}

    short = 0
    if bonds_per_unit is not None:
        for u, v, data in G.edges(data=True):
            short += data.get("dp", 0) < floor_dp(u, v)
    return {"chains": n_chains, "passes_mean": mean, "kmax": kmax,
            "chains_chord_unmet": unmet, "strands_too_short": int(short),
            "rings_merged": merges, "exchanges_tried": tried,
            "exchanges_kept": kept, "passes_before": as_dict(before),
            "passes_after": as_dict(after),
            "passes_target": {int(i): round(float(w), 2)
                              for i, w in enumerate(want) if w >= 0.005},
            "distance_before": distance(before), "distance_after": d}
