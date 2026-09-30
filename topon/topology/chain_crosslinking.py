"""Networks crosslinked along their chains: grow a lattice melt, crosslink its contacts.

In a randomly crosslinked (vulcanised) polymer, and in a protein network whose
crosslinkable residues sit where the sequence puts them, the chains exist
before the crosslinks and the crosslinks form between reactive beads that
touch. The graph follows from where the chains are. So the generator does
what the chemistry does, in two steps:

1. **Grow the melt.** The chains are grown one after another as
   self-avoiding walks on a periodic cubic lattice, one bead per site, each
   from a random free site. Every step goes to a free neighbour site,
   straight on with weight ``persistence`` and each of the four turns with
   ``(1 - persistence) / 4`` (0.2 weighs the five open directions equally,
   the kinetic growth walk the bond-fluctuation generator places its chains
   with). A chain that walks into a dead end starts again from another free
   site, as that generator's placement does. ``packing`` is the fraction of
   the sites filled.
2. **Crosslink the contacts.** Every pair of reactive beads within
   ``contact_radius`` lattice units (minimum image, a periodic k-d tree) is a
   candidate. The candidates are offered in a uniformly random order, and a
   crosslink is made unless one of its beads has reacted already, one of its
   beads has a crosslinked chain neighbour (the two would be one junction of
   functionality 6), the pair is one chain's beads closer in sequence than
   ``min_gap``, their types may not pair, it would put a strand's chord
   beyond its contour at the build scale (or, with ``keep_windings``,
   make it run across half the box), or its junction would sit where
   another one does. It stops at the requested
   count. When the contacts run out first, the candidates of the next
   distance shell are offered the same way, then the next, up to
   ``max_radius``.

A crosslink joins two beads, and its junction is their midpoint, so every
junction has functionality 4 and two chains (or one chain twice) passing.
The strands are the runs of beads between a chain's crosslinked beads, so
their DP is the sequence distance less one, as the chemistry builder reads
it, and a primary loop is an intra-chain crosslink with no other between
its two beads. The graph is the strand graph of
:mod:`topon.analysis.crosslinked` (the same nodes, edges, ``dp``, ``chain``,
``chain_index`` and ``G.graph["chains"]`` that reading the bead system back
gives), so the descriptors, ``chain_statistics`` and the later stages take
it as it is.

**Why grow a melt rather than decide the graph first** (measured against
BFM random-crosslinked references). The large-scale connectivity comes
from where the chains are. A pairing with no positions is far too uniform
(lambda2 of the core 0.127 against 0.051 at DP 100), and the lattice
sculpt too uniform at DP 50 (0.043 against 0.023). The same pairing on
phantom walks (overlapping freely, on or off a lattice) is too stringy
(0.034 to 0.039 at DP 100, 0.011 to 0.014 at DP 50), because an ideal-gas
density clusters the crosslinks. Excluded volume is what makes it right.
Grown at the references' packing with their growth step and crosslinked
by their contact rule, the builds are held against nine BFM networks
without a Mann-Whitney difference in the composite, lambda2, the mean
path, the loops or the chain statistics (the chord mean differs, see
below). How full the lattice is matters for the same reason (at 0.29 the
network is stringier again), so ``packing`` is a physical parameter, not
a discretisation. So does the order the melt is assembled in. Grown all
together, one step per chain per round, the same chains give DP-50
networks about 15 % lower in lambda2 than grown one after another, and
two million of BFM's Monte Carlo moves on those melts do not undo it.

**Contact rule and lattice parity.** On the cubic lattice a walk changes
sublattice at every step, so two beads of one chain can touch face to face
only across an odd number of bonds, and reactive beads on one sublattice
(the BFM node lattice's one reactive node in two) can never touch each other at all. The
default ``contact_radius`` 1.5 takes the 6 face and the 12 edge neighbours,
which reach both sublattices, so no pair is forbidden by parity. 1.0 is the
bond-fluctuation generator's rule (face neighbours only), which the BFM
references were built with and which reproduces the parity of their loops
and their number. With it loops close across odd gaps only, and there are
about half as many.

**Cost.** Growing is ``O(N)``, a Python loop over the beads with six
occupancy lookups each (0.05 s for 20,000 beads, 2.7 s for a million),
plus the restarts, which the packing and the chain length set (at 0.4, 10
% of chains of 50 beads and 20 to 27 % of chains of 100 start again at
least once). The candidates are a k-d tree query, ``O(S log S)`` in the
``S`` reactive beads. Offering them is a Python loop over the candidates,
each with a bisection in its two chains' lists of crosslinked beads,
``O(C log k)``, and building the graph is linear. In all 0.12 s at 20,000
beads, 0.6 s at 100,000 and 7 s at a million.

**Exact and sampled.** Exact: the chain count, lengths and reactive beads,
the crosslink count (or it raises), each bead once, the rules above, and so
every strand's DP, its chain and its place along it. Sampled: the
conformations (the growth, so the chains' size is measured, reported as
``c_inf``, and set only through ``persistence``), and which contacts react
(a uniformly random order). With ``c_inf`` given, the persistence is solved
for first on a small melt at the same packing, which is a measurement and
carries its noise (a few per cent).

The embedding is the melt the crosslinks formed in, so ``positions`` is a
conformation of every chain as well. A melt made elsewhere (on the same
lattice) can be crosslinked instead of a grown one (``positions=``), which
is how the generator's rules were checked on BFM's own melts. The crosslinks are kept in the order
they were made, and the first ``n`` of them are the network the run had
made at that point (:meth:`CrosslinkedMelt.graph_at`), which is how a gel
point or a conversion series is read off one run. (A run asked for ``n``
crosslinks makes the same ones, except that its chord bound, set by the
final bead count, is a fraction of a per cent tighter.)
"""
from __future__ import annotations

import bisect
import collections
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

#: The six lattice steps, in pairs of opposites.
_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1]], dtype=np.int64)

#: Design bond and bead density of the coarse-grained (Kremer-Grest) build,
#: the scale a strand's chord is held to its contour at.
KG_BOND = 0.97
KG_DENSITY = 0.85

#: Persistence that weighs the five open directions equally.
UNIFORM_PERSISTENCE = 0.2

#: Default fraction of lattice sites filled (the bond-fluctuation generator's
#: 0.45 target gives 0.395 on its odd lattice).
DEFAULT_PACKING = 0.4

#: Above this the growth jams more than it grows.
MAX_PACKING = 0.6


class CrosslinkingError(ValueError):
    """The request cannot be met: say which number and why."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainType:
    """``count`` chains of ``dp`` beads with crosslinkable beads at ``reactive``.

    ``reactive`` are bead indices, 0 being a chain end, interior beads only
    (a crosslink on an end bead would give it two bonds, which is not a
    junction). ``site_types`` labels each reactive bead (one label for all,
    or one per bead) for the pairing rule of :func:`crosslink_chains`.
    """

    count: int
    dp: int
    reactive: tuple
    site_types: object = "X"
    name: str = "A"

    def __post_init__(self):
        if int(self.count) < 1:
            raise CrosslinkingError(f"chain type {self.name!r}: count {self.count} < 1")
        if int(self.dp) < 3:
            raise CrosslinkingError(f"chain type {self.name!r}: {self.dp} beads "
                                    f"leave no interior bead to crosslink")
        r = tuple(int(x) for x in self.reactive)
        if list(r) != sorted(set(r)):
            raise CrosslinkingError(f"chain type {self.name!r}: reactive beads "
                                    f"must be distinct and increasing")
        bad = [x for x in r if not 1 <= x <= int(self.dp) - 2]
        if bad:
            raise CrosslinkingError(
                f"chain type {self.name!r}: reactive beads {bad[:5]} are not "
                f"interior (1 to {int(self.dp) - 2}); a crosslink on a chain "
                f"end bead leaves it with two bonds, which is no junction")
        object.__setattr__(self, "reactive", r)
        if isinstance(self.site_types, str):
            types = (self.site_types,) * len(r)
        else:
            types = tuple(str(t) for t in self.site_types)
            if len(types) != len(r):
                raise CrosslinkingError(
                    f"chain type {self.name!r}: {len(types)} site types for "
                    f"{len(r)} reactive beads")
        object.__setattr__(self, "site_types", types)

    @classmethod
    def every(cls, count: int, dp: int, every: int = 1, start: int = 1,
              site_type: str = "X", name: str = "A") -> "ChainType":
        """Every ``every``-th bead reactive from ``start``, interior beads only."""
        if int(every) < 1 or int(start) < 1:
            raise CrosslinkingError("every and start must be at least 1")
        return cls(count=count, dp=dp,
                   reactive=tuple(range(int(start), int(dp) - 1, int(every))),
                   site_types=site_type, name=name)

    @classmethod
    def from_sequence(cls, sequence: str, count: int, repeats: int = 1,
                      crosslink_residue: str = "Y",
                      name: Optional[str] = None) -> "ChainType":
        """One bead per residue, the crosslink residue's beads reactive.

        Through :func:`topon.protein_network.sequence.plan_chain`, so the
        residues it skips (a crosslink residue at a chain end) are skipped
        here too.
        """
        from topon.protein_network.sequence import plan_chain

        plan = plan_chain(sequence, repeats=repeats,
                          crosslink_residue=crosslink_residue)
        return cls(count=count, dp=plan.n_residues,
                   reactive=tuple(sorted(plan.crosslink_residue_indices)),
                   site_types=plan.crosslink_residue,
                   name=name or plan.sequence[:12])


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CrosslinkedMelt:
    """The network and the melt it was crosslinked in.

    ``positions[c]`` are chain ``c``'s bead coordinates in lattice units,
    unwrapped along the chain (wrap them into ``box`` for a periodic frame).
    ``crosslinks`` are ``((chain, bead), (chain, bead))`` in the order they
    were made, chains counted from 0. ``graph`` is the strand graph of all
    of them; :meth:`graph_at` gives it for any prefix.
    """

    graph: nx.MultiGraph
    positions: list
    crosslinks: list
    box: np.ndarray
    chain_types: list
    record: dict = field(default_factory=dict)

    @property
    def lengths(self) -> list:
        return [len(p) for p in self.positions]

    def graph_at(self, n: int) -> nx.MultiGraph:
        """The strand graph after the first ``n`` crosslinks of this run."""
        if not 0 <= int(n) <= len(self.crosslinks):
            raise CrosslinkingError(f"n {n} outside 0 to {len(self.crosslinks)}")
        return strand_graph(self.positions, self.crosslinks[:int(n)], self.box,
                            chain_types=self.chain_types)

    def bead_system(self):
        """``(pos, mol, bonds, bond_types)`` of the melt, one bead per chain bead.

        Bead ids from 1 in chain order, molecule ids from 1 per chain, bond
        type 1 along a chain and 2 for a crosslink, positions wrapped into
        the box: what :func:`topon.analysis.crosslinked.from_beads` reads.
        """
        pos, mol, bonds, types = {}, {}, [], []
        first = {}
        nxt = 1
        for c, P in enumerate(self.positions):
            first[c] = nxt
            W = np.mod(P, self.box)
            for b in range(len(P)):
                pos[nxt] = W[b]
                mol[nxt] = c + 1
                if b:
                    bonds.append((nxt - 1, nxt))
                    types.append(1)
                nxt += 1
        for (ci, bi), (cj, bj) in self.crosslinks:
            bonds.append((first[ci] + bi, first[cj] + bj))
            types.append(2)
        return pos, mol, bonds, types


# ---------------------------------------------------------------------------
# Step 1: the melt
# ---------------------------------------------------------------------------

def lattice_size(n_beads: int, packing: float) -> int:
    """Cubic lattice edge that holds ``n_beads`` at no more than ``packing``."""
    if not 0 < packing <= MAX_PACKING:
        raise CrosslinkingError(f"packing {packing} outside (0, {MAX_PACKING}]; "
                                f"above it the growth jams")
    return int(math.ceil((n_beads / packing) ** (1.0 / 3.0) - 1e-9))


def grow_melt(lengths: Sequence[int], L: int, persistence: float, rng,
              max_tries: int = 1000) -> tuple[list, dict]:
    """Self-avoiding chains grown one after another on a periodic ``L^3`` lattice.

    Returns the chains' unwrapped coordinates (one ``(dp, 3)`` array each)
    and a record of the restarts (chains that walked into a dead end and
    began again from another free site).
    """
    lengths = [int(n) for n in lengths]
    V = int(L) ** 3
    if sum(lengths) > V:
        raise CrosslinkingError(f"{sum(lengths)} beads do not fit on {V} sites")
    p = float(persistence)
    if not 0.0 <= p < 1.0:
        raise CrosslinkingError(f"persistence {p} outside [0, 1)")
    steps = tuple(tuple(int(v) for v in row) for row in _DIRS)
    turn = (1.0 - p) / 4.0
    L2 = L * L
    occ = bytearray(V)
    positions, restarts = [], 0
    for n in lengths:
        for _ in range(int(max_tries)):
            while True:
                site = int(rng.integers(V))
                if not occ[site]:
                    break
            z, r = divmod(site, L2)
            y, x = divmod(r, L)
            occ[site] = 1
            used, moves, last = [site], [], -1
            u = rng.random(n)
            for k in range(1, n):
                opts, tot = [], 0.0
                for d in range(6):
                    dx, dy, dz = steps[d]
                    nx_, ny_, nz_ = (x + dx) % L, (y + dy) % L, (z + dz) % L
                    f = nz_ * L2 + ny_ * L + nx_
                    if occ[f]:
                        continue
                    tot += 1.0 if last < 0 else (p if d == last else turn)
                    opts.append((tot, d, f, nx_, ny_, nz_))
                if not opts or tot <= 0.0:
                    break
                t = u[k] * tot
                for o in opts:
                    if t < o[0]:
                        break
                _, last, f, x, y, z = o
                occ[f] = 1
                used.append(f)
                moves.append(last)
            if len(used) == n:
                break
            for f in used:                         # a dead end: start again
                occ[f] = 0
            restarts += 1
        else:
            raise CrosslinkingError(
                f"a chain of {n} beads found no room in {max_tries} tries at "
                f"packing {sum(lengths) / V:.3f}; lower the packing")
        z, r = divmod(used[0], L2)
        y, x = divmod(r, L)
        P = np.empty((n, 3))
        P[0] = (x, y, z)
        if n > 1:
            P[1:] = P[0] + np.cumsum(_DIRS[np.asarray(moves, dtype=np.int64)], axis=0)
        positions.append(P)
    return positions, {"restarts": restarts}


def measured_c_inf(positions: Sequence[np.ndarray], n: int = 64) -> Optional[float]:
    """``<R^2(m)> / m`` over every bead pair ``m`` bonds apart, lattice units.

    ``m`` is ``n`` or the longest chain's bonds, whichever is fewer, and the
    pairs come from every chain that long; the chains' characteristic ratio
    at that separation (a lattice bond is 1).
    """
    m = min(int(n), max(len(P) for P in positions) - 1)
    if m < 1:
        return None
    s = [((P[m:] - P[:-m]) ** 2).sum(axis=1) for P in positions if len(P) > m]
    return float(np.concatenate(s).mean() / m)


def persistence_for(c_inf: float, packing: float, dp: int, *, seed: int = 0,
                    tol: float = 0.01) -> float:
    """The persistence whose grown chains have ``c_inf`` at this packing.

    Solved by bisection on a small melt (chains of ``min(dp, 128)`` beads,
    ``dp`` the longest chain, about 6,000 beads in all, at the packing asked
    for to within one chain) grown with one fixed seed at every trial, so
    the answer is repeatable. :func:`measured_c_inf` reads it at the same
    separation. Excluded volume swells the chains, so this is not
    ``(C - 1) / (C + 1)``: at packing 0.4 the uniform walk (0.2) gives
    about 1.75 where a phantom one gives 1.5.
    """
    n = max(8, min(int(dp), 128))
    L = lattice_size(6000, packing)
    chains = max(8, int(round(packing * L ** 3 / n)))
    while chains * n > MAX_PACKING * L ** 3:
        chains -= 1

    def c_of(p):
        rng = np.random.default_rng(seed)
        P, _ = grow_melt([n] * chains, L, p, rng)
        return measured_c_inf(P, n=min(n - 1, 64))

    lo, hi = 0.0, 0.95
    c_lo, c_hi = c_of(lo), c_of(hi)
    if not c_lo <= c_inf <= c_hi:
        raise CrosslinkingError(
            f"c_inf {c_inf} is outside what chains grown at packing {packing} "
            f"reach ({c_lo:.2f} to {c_hi:.2f}); a lower packing reaches "
            f"smaller values")
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        c = c_of(mid)
        if abs(c - c_inf) <= tol * c_inf:
            return mid
        lo, hi = (mid, hi) if c < c_inf else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Step 2: the crosslinks
# ---------------------------------------------------------------------------

def contact_shells(contact_radius: float, max_radius: float) -> list:
    """Distances at which the candidates are offered, nearest first.

    The first shell is every lattice distance up to ``contact_radius``;
    each later one is the next distinct lattice distance, up to
    ``max_radius``.
    """
    top = int(math.ceil(max(contact_radius, max_radius))) + 1
    d2 = sorted({i * i + j * j + k * k for i in range(top) for j in range(top)
                 for k in range(top)} - {0})
    first = [math.sqrt(x) for x in d2 if math.sqrt(x) <= contact_radius + 1e-9]
    if not first:
        raise CrosslinkingError(f"contact_radius {contact_radius} is below the "
                                f"lattice spacing (1)")
    later = [math.sqrt(x) for x in d2
             if contact_radius + 1e-9 < math.sqrt(x) <= max_radius + 1e-9]
    return [first[-1]] + later


def _contacts(X, lo, hi, L, chain_of, bead_of, type_of, min_gap, allowed):
    """Reactive pairs with ``lo < d <= hi`` (minimum image) that may pair."""
    tree = cKDTree(X, boxsize=float(L))
    pr = tree.query_pairs(hi + 1e-6, output_type="ndarray")
    if not len(pr):
        return pr
    d = X[pr[:, 1]] - X[pr[:, 0]]
    d -= L * np.round(d / L)
    dist = np.sqrt((d * d).sum(axis=1))
    keep = dist > lo + 1e-6
    same = chain_of[pr[:, 0]] == chain_of[pr[:, 1]]
    gap = np.abs(bead_of[pr[:, 0]] - bead_of[pr[:, 1]])
    keep &= ~(same & (gap < min_gap))
    if allowed is not None:
        keep &= allowed[type_of[pr[:, 0]], type_of[pr[:, 1]]]
    pr = pr[keep]
    # a canonical order before shuffling, so the draw does not depend on
    # the tree's internal order
    pr = np.sort(pr, axis=1)
    return pr[np.lexsort((pr[:, 1], pr[:, 0]))]


class _Pairing:
    """The crosslinks made so far and the rules a new one must pass."""

    def __init__(self, positions, box_edge, bound, keep_windings):
        self.P = positions
        self.L = float(box_edge)
        self.half = 0.5 * self.L
        self.bound = bound                       # chord bound per bond, or None
        self.keep_windings = bool(keep_windings)
        self.on = collections.defaultdict(list)  # chain -> crosslinked beads
        self.jpos = {}                           # (chain, bead) -> junction, chain frame
        self.mids = set()                        # junction sites, doubled
        self.made = []
        self.rejected = collections.Counter()

    def _strands_ok(self, c, beads):
        """Every strand of chain ``c`` touching ``beads`` (tentatively in)."""
        lst = self.on[c]
        P = self.P[c]
        n = len(P)
        for b in beads:
            k = lst.index(b)
            here = self.jpos[(c, b)]
            sides = []
            if k > 0:
                p = lst[k - 1]
                sides.append((here - self.jpos[(c, p)], b - p))
            else:
                sides.append((here - P[0], b))
            if k + 1 < len(lst):
                q = lst[k + 1]
                sides.append((self.jpos[(c, q)] - here, q - b))
            else:
                sides.append((P[n - 1] - here, n - 1 - b))
            for v, bonds in sides:
                # the builder draws a strand along the minimum image of its
                # chord; one that ran more than half the box in the melt is
                # drawn the short way round
                w = v - self.L * np.round(v / self.L)
                if self.keep_windings and np.any(np.abs(v - w) > 1e-9):
                    self.rejected["winding"] += 1
                    return False
                if self.bound is not None and float(w @ w) > (bonds * self.bound) ** 2 + 1e-9:
                    self.rejected["chord"] += 1
                    return False
        return True

    def offer(self, a, b) -> bool:
        (ci, bi), (cj, bj) = a, b
        for c, x in ((ci, bi), (cj, bj)):
            lst = self.on[c]
            k = bisect.bisect_left(lst, x)
            if k < len(lst) and lst[k] == x:
                return False                       # reacted already
            if (k > 0 and lst[k - 1] == x - 1) or (k < len(lst) and lst[k] == x + 1):
                self.rejected["neighbour"] += 1
                return False
        d = np.mod(self.P[cj][bj], self.L) - np.mod(self.P[ci][bi], self.L)
        d -= self.L * np.round(d / self.L)
        oi = 0.5 * d
        mid = tuple(np.round(2 * np.mod(self.P[ci][bi] + oi, self.L)).astype(int)
                    % int(round(2 * self.L)))
        if mid in self.mids:
            self.rejected["coincident"] += 1
            return False
        # tentatively in, then every strand the two beads touch
        self.jpos[(ci, bi)] = self.P[ci][bi] + oi
        self.jpos[(cj, bj)] = self.P[cj][bj] - oi
        bisect.insort(self.on[ci], bi)
        bisect.insort(self.on[cj], bj)
        if ci == cj:
            ok = self._strands_ok(ci, (bi, bj))
        else:
            ok = self._strands_ok(ci, (bi,)) and self._strands_ok(cj, (bj,))
        if not ok:
            self.on[ci].remove(bi)
            self.on[cj].remove(bj)
            del self.jpos[(ci, bi)], self.jpos[(cj, bj)]
            return False
        self.mids.add(mid)
        self.made.append(((int(ci), int(bi)), (int(cj), int(bj))))
        return True


def crosslink_chains(chains: Sequence[ChainType], *,
                     crosslinks: Optional[int] = None,
                     conversion: Optional[float] = None,
                     per_chain: Optional[float] = None,
                     packing: float = DEFAULT_PACKING,
                     persistence: Optional[float] = None,
                     c_inf: Optional[float] = None,
                     contact_radius: float = 1.5,
                     max_radius: float = 3.0,
                     min_gap: int = 6,
                     pairs: Optional[Iterable[tuple]] = None,
                     build_density: Optional[float] = KG_DENSITY,
                     build_bond: float = KG_BOND,
                     min_dangling_dp: int = 0,
                     keep_windings: bool = False,
                     up_to: bool = False,
                     lattice: Optional[int] = None,
                     positions: Optional[Sequence[np.ndarray]] = None,
                     seed: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None) -> CrosslinkedMelt:
    """Grow the chains as a lattice melt and crosslink their contacts.

    Args:
        chains: the chain types (:class:`ChainType`).
        crosslinks, conversion, per_chain: the target, exactly one of them.
            ``conversion`` is the fraction of the reactive beads that
            react, ``per_chain`` the crosslinked beads per chain (each
            crosslink counts on both chains it joins). Rounded to a whole
            number of crosslinks, which is then met exactly or refused.
        packing: fraction of the lattice sites filled (at most 0.6).
        persistence: weight of a straight step (each turn weighs
            ``(1 - persistence) / 4``); 0.2, uniform over the open
            directions, when neither this nor ``c_inf`` is given.
        c_inf: the chains' characteristic ratio in lattice units (a bond is
            one site spacing); the persistence is solved for
            (:func:`persistence_for`).
        contact_radius: reactive beads this close (lattice units) are in
            contact. 1.5 takes face and edge neighbours, free of the
            lattice's parity; 1.0 face neighbours only, the bond-fluctuation
            generator's rule.
        max_radius: how far the candidate shells may go out when the
            contacts cannot make the count.
        min_gap: fewest bonds between two beads of one chain that may
            crosslink each other.
        pairs: site types that may pair, as ``(type, type)``; None lets any
            pair.
        build_density, build_bond: the coarse-grained build the chords are
            held to (beads per sigma^3, bond in sigma): a strand of ``dp``
            beads may have a chord of at most ``(dp + 1) * build_bond`` in
            the box that density gives, one bead less per crosslink since a
            crosslink is built as one bead. ``build_density=None`` sets no
            chord bound (the atomistic route, whose placement has its own
            guard).
        min_dangling_dp: fewest beads on a dangling strand. Reactive beads
            closer to a chain end are left out (a chain's end bead always
            is). The pipeline passes 0 on the coarse-grained route, whose
            builder bonds the end bead of a strand of no beads straight to
            its junction, and 1 on the atomistic route.
        keep_windings: refuse a crosslink that makes a strand run more than
            half the box in the melt. The builder draws every strand along
            the minimum image of its chord, so such a strand would be built
            the short way round the box, and its cycles would wind
            differently from the melt's. Off, they are made and counted
            (``strands_rewound``), which a box smaller than the chains
            (a few long chains) needs; in a box several strands wide there
            are few of them either way.
        up_to: take the target as a ceiling, making as many crosslinks as
            the contacts within ``max_radius`` allow up to it, instead of
            refusing when they fall short. With the target at half the
            reactive beads and ``max_radius`` at ``contact_radius`` it
            crosslinks every contact the rules allow, in a random order,
            which is how the bond-fluctuation generator crosslinks (a prefix
            is then the network at any lower conversion).
        lattice: the lattice edge, overriding the one ``packing`` gives.
        positions: a melt to crosslink instead of growing one, each chain's
            beads as unwrapped integer lattice coordinates in chain order
            (the chains of ``chains`` in order, their lengths matching), on
            the ``lattice`` given, which is then required. Nothing checks
            that the melt is self-avoiding.
        seed, rng: the one random stream (``rng`` wins when both are given;
            the record's ``seed`` is then None).

    Raises:
        CrosslinkingError: an impossible or unreachable request, with the
            number that decides it.
    """
    t_start = time.perf_counter()
    chains = list(chains)
    if not chains:
        raise CrosslinkingError("no chain types")
    if sum(x is not None for x in (crosslinks, conversion, per_chain)) != 1:
        raise CrosslinkingError("give exactly one of crosslinks, conversion, per_chain")
    if persistence is not None and c_inf is not None:
        raise CrosslinkingError("give persistence or c_inf, not both")
    if int(min_gap) < 2:
        raise CrosslinkingError("min_gap must be at least 2 (neighbouring beads "
                                "cannot both be crosslinked)")
    rng_given = rng is not None
    if rng is None:
        rng = np.random.default_rng(seed)

    # the chains and their reactive beads
    lengths, type_of_chain = [], []
    for t, ct in enumerate(chains):
        lengths += [int(ct.dp)] * int(ct.count)
        type_of_chain += [t] * int(ct.count)
    n_chains, n_beads = len(lengths), int(sum(lengths))
    labels = sorted({s for ct in chains for s in ct.site_types})
    label_id = {s: i for i, s in enumerate(labels)}
    if pairs is not None:
        pairs = [tuple(pr) for pr in pairs]
        unknown = sorted({x for pr in pairs for x in pr} - set(label_id))
        if unknown:
            raise CrosslinkingError(
                f"pairs names site types {unknown} that no chain type carries "
                f"(the chains carry {labels})")
    m = int(min_dangling_dp)
    n_reactive = sum(len(ct.reactive) * int(ct.count) for ct in chains)
    site_chain, site_bead, site_type = [], [], []
    excluded = 0
    for c, t in enumerate(type_of_chain):
        ct = chains[t]
        for bead, lab in zip(ct.reactive, ct.site_types):
            if bead - 1 < m or ct.dp - 2 - bead < m:
                excluded += 1
                continue
            site_chain.append(c)
            site_bead.append(bead)
            site_type.append(label_id[lab])
    site_chain = np.array(site_chain, dtype=np.int64)
    site_bead = np.array(site_bead, dtype=np.int64)
    site_type = np.array(site_type, dtype=np.int64)
    n_sites = len(site_chain)
    allowed = None
    if pairs is not None:
        allowed = np.zeros((len(labels), len(labels)), dtype=bool)
        for a, b in pairs:
            allowed[label_id[a], label_id[b]] = allowed[label_id[b], label_id[a]] = True

    if crosslinks is not None:
        target = int(crosslinks)
    elif conversion is not None:
        # of every reactive bead the chains carry, those min_dangling_dp
        # leaves out included, so the target does not move with it
        target = int(round(float(conversion) * n_reactive / 2))
    else:
        target = int(round(float(per_chain) * n_chains / 2))
    if target < 0 or 2 * target > n_sites:
        raise CrosslinkingError(
            f"{target} crosslinks need {2 * target} reactive beads and the "
            f"chains carry {n_sites}"
            + (f" ({excluded} left out by min_dangling_dp {m})" if excluded else ""))

    # step 1: the melt
    if positions is not None and lattice is None:
        raise CrosslinkingError("a given melt needs the lattice it sits on")
    L = int(lattice) if lattice is not None else lattice_size(n_beads, packing)
    packing_actual = n_beads / L ** 3
    if packing_actual > MAX_PACKING + 1e-9 and positions is None:
        raise CrosslinkingError(f"packing {packing_actual:.3f} on a {L}^3 lattice "
                                f"is above {MAX_PACKING}")
    t0 = time.perf_counter()
    if positions is not None:
        positions = [np.asarray(P, dtype=float).reshape(-1, 3) for P in positions]
        if [len(P) for P in positions] != lengths:
            raise CrosslinkingError("the given melt's chains do not match the chain "
                                    "types' counts and lengths")
        p, growth = None, {"given": True}
    else:
        if c_inf is not None:
            p = persistence_for(float(c_inf), packing_actual, max(lengths),
                                seed=int(rng.integers(2 ** 31)))
        else:
            p = UNIFORM_PERSISTENCE if persistence is None else float(persistence)
        positions, growth = grow_melt(lengths, L, p, rng)
    t_grow = time.perf_counter() - t0

    # the chord bound, per bond, in lattice units
    bound = None
    if build_density is not None:
        sigma_per_unit = ((n_beads - target) / float(build_density)) ** (1 / 3) / L
        bound = float(build_bond) / sigma_per_unit

    # step 2: the crosslinks, shell by shell
    t0 = time.perf_counter()
    X = np.mod(np.array([positions[c][b] for c, b in zip(site_chain, site_bead)],
                        dtype=float).reshape(-1, 3), L)
    pg = _Pairing(positions, L, bound, keep_windings)
    shells = contact_shells(float(contact_radius), max(float(max_radius), float(contact_radius)))
    lo, offered, used = 0.0, 0, 0
    for hi in shells:
        if len(pg.made) >= target:
            break
        used += 1
        pr = _contacts(X, lo, hi, L, site_chain, site_bead, site_type,
                       int(min_gap), allowed)
        lo = hi
        for t in rng.permutation(len(pr)):
            i, j = pr[t]
            offered += 1
            pg.offer((int(site_chain[i]), int(site_bead[i])),
                     (int(site_chain[j]), int(site_bead[j])))
            if len(pg.made) >= target:
                break
    if len(pg.made) < target and not up_to:
        raise CrosslinkingError(
            f"{len(pg.made)} of {target} crosslinks made: the reactive beads "
            f"within {shells[-1]:.2f} lattice units of each other are used up "
            f"under the rules (rejected: {dict(pg.rejected)}); raise max_radius, "
            f"or ask for fewer")
    t_pair = time.perf_counter() - t0

    t0 = time.perf_counter()
    box = np.array([L, L, L], dtype=float)
    G = strand_graph(positions, pg.made, box, chain_types=type_of_chain)
    t_graph = time.perf_counter() - t0

    counts = collections.Counter()
    for u, v, d in G.edges(data=True):
        counts[d["cls"]] += 1
    rewound = _rewound(positions, pg.jpos, G, L)
    record = {
        "chains": n_chains, "beads": n_beads,
        "chain_types": [{"name": ct.name, "count": int(ct.count), "dp": int(ct.dp),
                         "reactive": len(ct.reactive)} for ct in chains],
        "reactive_beads": n_reactive, "reactive_left_out": excluded,
        "crosslinks": len(pg.made), "conversion": 2 * len(pg.made) / max(1, n_reactive),
        "lattice": L, "packing": packing_actual, "persistence": p,
        "c_inf_requested": c_inf, "c_inf": measured_c_inf(positions),
        "contact_radius": float(contact_radius), "min_gap": int(min_gap),
        "min_dangling_dp": m, "keep_windings": bool(keep_windings),
        "strands_rewound": rewound, "shells_used": used,
        "shell_radius": shells[used - 1] if used else None,
        "candidates_offered": offered, "rejected": dict(pg.rejected),
        "chord_bound_per_bond": bound,
        "build": ({"density": float(build_density), "bond": float(build_bond)}
                  if build_density is not None else None),
        "growth": growth,
        "strands": dict(counts), "sol_chains": int(G.graph["sol_chains"]["count"]),
        "seconds": {"grow": round(t_grow, 4), "pair": round(t_pair, 4),
                    "graph": round(t_graph, 4),
                    "total": round(time.perf_counter() - t_start, 4)},
        "seed": seed if rng_given is False else None,
    }
    G.graph["crosslinking"] = record
    return CrosslinkedMelt(graph=G, positions=positions, crosslinks=list(pg.made),
                           box=box, chain_types=type_of_chain, record=record)


def _rewound(positions, jpos, G, L) -> int:
    """Strands that ran more than half the box in the melt.

    ``jpos`` holds each crosslinked bead's junction in its own chain's
    unwrapped frame; a strand's melt vector is the difference along the
    chain, and the build draws its minimum image instead.
    """
    n = 0
    for c, rec in G.graph["chains"].items():
        P = positions[c - 1]
        for e in rec["strands"]:
            lo, hi = G.edges[e]["span"]
            a = jpos.get((c - 1, lo - 1), P[0])
            b = jpos.get((c - 1, hi + 1), P[len(P) - 1])
            v = np.asarray(b) - np.asarray(a)
            n += bool(np.any(np.abs(v - L * np.round(v / L)) < np.abs(v) - 1e-9))
    return n


# ---------------------------------------------------------------------------
# The strand graph
# ---------------------------------------------------------------------------

def strand_graph(positions: Sequence[np.ndarray], crosslinks: Sequence,
                 box, chain_types: Optional[Sequence[int]] = None) -> nx.MultiGraph:
    """The strand graph of chains and the crosslinks between their beads.

    Junction ``i`` is crosslink ``i`` (functionality 4, at its two beads'
    midpoint), end nodes follow, one per chain end with a dangling strand.
    Every strand carries ``cls``, ``dp`` (beads strictly between its two
    junction beads, a free end bead excluded), ``chain`` (from 1),
    ``chain_index`` (from 0 at the chain's bead 0) and ``span``, its first
    and last bead on the chain. ``G.graph`` carries ``box``,
    ``architecture``, ``chains`` (per chain its ``dp``, strands in order,
    the junctions it passes, ``passes``, ``junction_beads``) and
    ``sol_chains``, as :func:`topon.analysis.crosslinked.from_beads` writes
    them.
    """
    box = np.asarray(box, dtype=float).reshape(3)
    L = box
    G = nx.MultiGraph()
    on = collections.defaultdict(dict)          # chain -> bead -> junction
    for x, ((ci, bi), (cj, bj)) in enumerate(crosslinks):
        for c, b in ((ci, bi), (cj, bj)):
            if not 1 <= b <= len(positions[c]) - 2:
                raise CrosslinkingError(f"crosslink {x} is on bead {b} of chain {c}, "
                                        f"which is not an interior bead")
            if b in on[c]:
                raise CrosslinkingError(f"bead {b} of chain {c} is crosslinked twice")
        a = np.mod(np.asarray(positions[ci][bi], float), L)
        b = np.mod(np.asarray(positions[cj][bj], float), L)
        d = b - a
        d -= L * np.round(d / L)
        G.add_node(x, kind="junction", pos=np.mod(a + 0.5 * d, L),
                   beads=[(int(ci), int(bi)), (int(cj), int(bj))])
        on[ci][bi] = x
        on[cj][bj] = x
    end_id = len(crosslinks)
    chains, sol = {}, []
    for c, P in enumerate(positions):
        n = len(P)
        beads = sorted(on[c]) if c in on else []
        rec = {"dp": int(n), "strands": [], "junctions": [], "passes": len(beads),
               "junction_beads": len(beads)}
        if chain_types is not None:
            rec["type"] = int(chain_types[c])
        chains[c + 1] = rec
        if not beads:
            sol.append(n)
            continue
        runs = [(None, 0, beads[0])]
        runs += [(beads[k], beads[k], beads[k + 1]) for k in range(len(beads) - 1)]
        runs += [(beads[-1], beads[-1], None)]
        for idx, (_, lo, hi) in enumerate(runs):
            if hi is not None and idx == 0:
                # dangling from the chain's first end: the end node is bead 0
                G.add_node(end_id, kind="end", pos=np.mod(np.asarray(P[0], float), L),
                           bead=(c, 0))
                u, v = on[c][hi], end_id
                end_id += 1
                attrs = {"cls": "dangling", "dp": int(hi - 1), "span": (1, int(hi) - 1)}
            elif hi is None:
                G.add_node(end_id, kind="end", pos=np.mod(np.asarray(P[n - 1], float), L),
                           bead=(c, n - 1))
                u, v = on[c][lo], end_id
                end_id += 1
                attrs = {"cls": "dangling", "dp": int(n - 2 - lo),
                         "span": (int(lo) + 1, n - 2)}
            else:
                u, v = on[c][lo], on[c][hi]
                attrs = {"cls": "loop" if u == v else "bridge", "dp": int(hi - lo - 1),
                         "span": (int(lo) + 1, int(hi) - 1)}
            k = G.add_edge(u, v, chain=c + 1, chain_index=idx, **attrs)
            rec["strands"].append((u, v, k))
            if hi is not None:
                rec["junctions"].append(on[c][hi])
    G.graph["box"] = tuple(float(x) for x in box)
    G.graph["architecture"] = "random_crosslinked"
    G.graph["contracted"] = True
    G.graph["chains"] = chains
    G.graph["chains_unordered"] = 0
    counts = collections.Counter(sol)
    G.graph["sol_chains"] = {"count": len(sol),
                             "dp": int(counts.most_common(1)[0][0]) if sol else 0}
    if len(counts) > 1:
        G.graph["sol_chains"]["dps"] = {int(k): int(v) for k, v in sorted(counts.items())}
    return G
