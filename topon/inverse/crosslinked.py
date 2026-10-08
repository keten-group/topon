"""``topon fit`` for a network crosslinked along its chains.

In a randomly crosslinked or vulcanised melt, or a protein network
crosslinked at fixed residues, no crosslinker molecule sits at a chain end:
two chain beads are bonded. Read with :mod:`topon.analysis.crosslinked` (a
junction is a crosslink, a strand a run of chain beads between crosslinks),
such a reference has chains, and its crosslinks sit at bead positions along
them. Those are the inputs of the generator of
:mod:`topon.topology.chain_crosslinking` (``topology.source: "crosslink"``),
so the fit reads them off and writes them:

``chains``
    one chain type per chain length, with its count (sol chains included).
``reactive beads``
    each crosslinked bead's position along its chain, 0 at the chain's first
    end, read from the strands in chain order (a strand's DP is the beads
    strictly between its two crosslinks). When the positions sit on one
    period (the greatest common divisor of their spacings along each chain,
    each chain read from whichever end puts it on the common residue), the
    period is written (``reactive_every`` from ``reactive_start``), which
    also names the beads of that period no crosslink reached. The slots of
    the period are then held against a uniform draw at the reference's
    crosslinks per chain and slot: as many empty slots as a uniform draw
    leaves in fewer than 1 in 1,000 cases makes the positions seen the list;
    single slots left that improbably empty are taken off the period, and
    when those are exactly the beads next to the chain ends,
    ``min_dangling_dp`` 1 is written instead. Chain lengths that caught
    none of the crosslinks the others' rate predicts are written with no
    reactive bead. Positions on no common period are written as a list as
    each chain reads. A sequence or a period given by hand is checked
    against them and written instead.
``crosslinks``
    the count, which the generator meets exactly.
``contact_radius``
    1.0, face contacts, when nine in ten or more of the crosslinks within a
    chain close across an odd number of bonds: on the cubic lattice a chain
    changes sublattice at every step, so under face contacts beads of one
    chain touch across odd gaps only (except across the box of an odd
    lattice), and under the generator's default 1.5 three quarters or more
    close across even ones. Half or more even gives 1.5. It takes
    :data:`PARITY_MIN` such crosslinks and a period that allows both
    parities; otherwise both radii are built and scored.
``min_gap``
    the generator's 6, or the reference's smallest gap within a chain when
    that is smaller (3 at least). A reference with no crosslink within a
    chain, where the builds make three or more on average, gets a gap past
    its longest chain, so none are made, and every setting is built again.
``packing``
    a lattice reference's beads over its sites (the bond-fluctuation
    references are 0.3948 on 37^3), written so that the generator picks the
    same lattice. A reference with no lattice to read (a data file in sigma,
    a graph with no cell) has :data:`PACKINGS` built and scored.
``chemistry.target_density``
    the bead density of a data file; 0.85, the Kremer-Grest melt, otherwise
    (flagged). The generator holds every strand's chord to its contour at
    that density.

Each candidate setting is built through the pipeline's own stages 1 to 3
(:func:`topon.inverse.scaffold.build_graph`) on ``seeds`` seeds and scored on
the acceptance measures: the descriptor composite, lambda2 and the mean path
of the core, and the KS statistics of passes per chain and of the bridge,
dangling and loop DPs, with the loop count. When the best two are closer
than their seeds' spread, the best :data:`TIE_TOP` are built on more seeds
(:data:`TIE_SEEDS` each) and chosen between again: on a small reference one
build's lambda2 moves by a third from seed to seed. With every setting
measured there is one candidate, and its builds are the report's first
check.

The lattice route (``route="lattice"``: a sculpted graph covered with
chains, ``topology.generator.architecture: "random_crosslinked"``) is fitted
through the end-linked machinery of :mod:`topon.inverse.scaffold`, with the
p75 of the junction separation for the rule of thumb (the p95 overshoots
on these networks), the shells around it swept, and a connectivity
floor of 0.95 for the small pieces the sculpt leaves. It builds no
nearest-neighbour control row: on these networks the exact search takes
minutes to sculpt one shell or gives up (225 and 373 s a seed on the DP-50
reference's cell), and it scored far off when it was measured (0.41 and 0.51).
"""
from __future__ import annotations

import collections
import copy
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from topon.inverse.fit import FitError, FitResult, _doctor, _flag, base_config
from topon.inverse.scaffold import (
    CONTROL_CUTOFF, build_graph, candidate_cutoffs, choose_cell,
    describe_cutoff, rule_of_thumb, shells_within, sweep, with_cutoff)

#: The generator's own smallest gap within a chain.
DEFAULT_MIN_GAP = 6
#: Crosslinks within a chain the parity reading needs before it decides.
PARITY_MIN = 8
#: Face contacts, and face and edge contacts (the generator's default).
FACE, FACE_EDGE = 1.0, 1.5
#: Fraction of even gaps within chains at or below which the contact rule
#: reads as face contacts, and at or above which as face and edge contacts.
#: Face contacts close even gaps only across the box of an odd lattice
#: (1.2 % of them with chains longer than the box); face and edge contacts
#: close three quarters or more across even gaps.
EVEN_FACE, EVEN_EDGE = 0.1, 0.5
#: When the best settings are within the seeds' scatter, the best
#: ``TIE_TOP`` are built again until they have ``TIE_SEEDS`` seeds each.
TIE_TOP, TIE_SEEDS = 3, 16
#: Chance below which a reading of the reactive beads is refused: a slot, or
#: as many slots, left without a crosslink that a uniform draw leaves this
#: rarely.
EMPTY_P = 1e-3
#: Packings built when the reference has no lattice to read one from.
PACKINGS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
#: Mean crosslinks within a chain per build past which a reference with none
#: has them forbidden (a Poisson count of 3 gives no such crosslink in 5 %).
INTRA_EXPECTED = 3.0
#: Connectivity floor of the lattice route (measured: 2 to 5 % of the active
#: sites end in small pieces).
LATTICE_FLOOR = 0.95
#: Bead density written when the reference does not say.
FALLBACK_DENSITY = 0.85
#: Chain-level distributions compared by KS.
CHAIN_KEYS = ("passes", "dp_bridge", "dp_dangling", "dp_loop")


# ---------------------------------------------------------------------------
# Chains and the crosslinks along them
# ---------------------------------------------------------------------------

@dataclass
class ChainWalk:
    """One chain: its length and the crosslinked beads along it."""

    chain: object
    dp: int
    positions: list          # crosslinked bead indices, 0 at the first end
    junctions: list          # the junction at each position
    ok: bool = True          # the positions are exact
    why: str = ""


def chain_walks(G) -> list:
    """Every chain of ``G.graph["chains"]`` with its crosslinked beads.

    Walks the chain's strands in order: its first crosslinked bead is the
    first strand's DP plus one (the strand runs from the end bead, which it
    leaves out, to the junction bead), and each next one the strand between
    plus one. A chain is exact (``ok``) when its beads add up, two end beads,
    its strands and one bead per pass; it is not when it passes a fused
    junction through neighbouring beads or a crosslink sits on its end bead
    (the strand then runs into another molecule).
    """
    out = []
    for m, c in (G.graph.get("chains") or {}).items():
        dp = int(c["dp"])
        strands = list(c.get("strands") or [])
        passes = list(c.get("junctions") or [])
        if not strands:
            ok = int(c.get("passes", 0)) == 0
            out.append(ChainWalk(m, dp, [], [], ok, "" if ok else "no strands"))
            continue
        if int(c.get("junction_beads", len(passes))) != len(passes):
            out.append(ChainWalk(m, dp, [], [], False, "fused junction"))
            continue
        if len(strands) != len(passes) + 1:
            out.append(ChainWalk(m, dp, [], [], False, "not a linear chain"))
            continue
        dps = [int(G.edges[e]["dp"]) for e in strands]
        if sum(dps) + len(passes) + 2 != dp:
            out.append(ChainWalk(m, dp, [], [], False, "beads do not add up"))
            continue
        pos, p = [], 0
        for k in range(len(passes)):
            p += dps[k] + 1
            pos.append(p)
        out.append(ChainWalk(m, dp, pos, passes))
    return out


def intra_gaps(walks) -> list:
    """Bonds between the two beads of every crosslink within one chain."""
    gaps = []
    for w in walks:
        if not w.ok:
            continue
        at = collections.defaultdict(list)
        for p, j in zip(w.positions, w.junctions):
            at[j].append(p)
        for ps in at.values():
            ps.sort()
            gaps += [b - a for a, b in zip(ps, ps[1:])]
    return gaps


def crosslink_count(G) -> dict:
    """Crosslinks, and the junctions that are not one crosslink of two chain beads.

    Every crosslinked bead carries one crosslink, so with every chain walked
    the count is half the chains' junction beads, which holds for a fused
    junction too (two chains crosslinked at two neighbouring pairs are one
    junction of degree four and two crosslinks). Otherwise a junction a
    chain passes through one bead carries two of its strands, so a junction
    of degree ``d`` holds ``d / 2 - 1``. A junction is fused when it holds
    more than two beads or more than four strands.
    """
    from topon.analysis.descriptors import node_kind

    n, fused, odd = 0, 0, 0
    for x, data in G.nodes(data=True):
        if node_kind(G, x) != "junction":
            continue
        d = G.degree(x)
        n += max(1, d // 2 - 1)
        fused += d > 4 or len(data.get("beads") or ()) > 2
        odd += d % 2
    chains = G.graph.get("chains") or {}
    if chains and not G.graph.get("chains_unordered") and all(
            "junction_beads" in c for c in chains.values()):
        n = sum(int(c["junction_beads"]) for c in chains.values()) // 2
    return {"count": int(n), "fused_junctions": int(fused),
            "odd_junctions": int(odd)}


def _gcd(values) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, int(v))
    return g


def infer_reactive(walks, every: Optional[int] = None,
                   start: Optional[int] = None) -> dict:
    """The period the crosslinked beads sit on, or the positions seen.

    ``every`` (and ``start``) given are checked instead of inferred. Returns
    ``method`` ``"pattern"`` with ``every``, ``start``, the chains read from
    their far end (``flipped``) and how many of the period's slots a
    crosslink reached, or ``"list"`` with the positions seen per chain
    length and why the period did not hold.
    """
    located = [w for w in walks if w.ok and w.positions]
    out = {"chains_located": len(located),
           "chains_not_exact": sum(1 for w in walks if not w.ok),
           "beads_located": sum(len(w.positions) for w in located)}
    if not located:
        out.update(method="none", why="no crosslinked bead could be located")
        return out
    given = every is not None
    g = int(every) if given else _gcd(b - a for w in located
                                      for a, b in zip(w.positions, w.positions[1:]))
    listed = _position_lists(located)
    if g == 0:
        out.update(method="list", lists=listed,
                   why="no chain has two crosslinked beads to read a period from")
        return out

    # every chain votes for its residue read from either end; a tie goes to
    # the residue more chains sit on read from their first end
    votes, forward = collections.Counter(), collections.Counter()
    for w in located:
        f, b = w.positions[0] % g, (w.dp - 1 - w.positions[0]) % g
        votes[f] += 1
        forward[f] += 1
        if b != f:
            votes[b] += 1
    if given and start is not None:
        r = int(start) % g
    else:
        r = min(votes, key=lambda k: (-votes[k], -forward[k], k))
    s = int(start) if (given and start is not None) else (r if r >= 1 else g)

    def on(ps):
        return all(p % g == r and p >= s for p in ps)

    flipped, misfit = 0, []
    for w in located:
        if on(w.positions):
            continue
        if on([w.dp - 1 - p for p in w.positions]):
            flipped += 1
        else:
            misfit.append(w.chain)
    out.update(every=g, start=s, flipped=flipped)
    if misfit:
        out.update(method="list", lists=listed, misfit=len(misfit),
                   why=f"{len(misfit)} chains have crosslinked beads off the "
                       f"period {g} from {s}, read from either end, so the "
                       f"positions are listed as each chain reads from its "
                       f"first end")
        return out

    # Every chain read the way it sits on the period, and the slots it hit
    oriented = {}
    for w in located:
        oriented[w.chain] = (list(w.positions) if on(w.positions)
                             else sorted(w.dp - 1 - p for p in w.positions))
    hits = collections.Counter((w.dp, p) for w in located for p in oriented[w.chain])
    chains_of = collections.Counter(w.dp for w in walks if w.ok)
    slots = {dp: list(range(s, dp - 1, g)) for dp in chains_of}
    by_length = {int(dp): {"chains": int(chains_of[dp]), "slots": len(slots[dp]),
                           "hits": int(sum(hits[(dp, p)] for p in slots[dp]))}
                 for dp in sorted(chains_of)}
    unreactive = _unreactive_lengths(by_length)
    active = [dp for dp in sorted(chains_of) if dp not in unreactive and slots[dp]]
    chain_slots = sum(chains_of[dp] * len(slots[dp]) for dp in active)
    rate = sum(by_length[dp]["hits"] for dp in active) / max(1, chain_slots)
    # A slot left empty that a uniform draw leaves empty this rarely is off
    # the reactive set on its own; the rest are held against the count a
    # uniform draw leaves empty among them.
    empty, bad, mu, var = [], [], 0.0, 0.0
    n_tested = sum(len(slots[dp]) for dp in active)
    for dp in active:
        # crosslinks a slot of this length caught, at its own rate: lengths
        # need not react alike per bead
        lam = by_length[dp]["hits"] / len(slots[dp])
        q = math.exp(-lam)                    # chance a reactive slot stays empty
        by_length[int(dp)]["per_slot"] = round(lam, 3)
        alone = q * n_tested < EMPTY_P
        for p in slots[dp]:
            if hits[(dp, p)] == 0:
                empty.append((dp, p))
                if alone:
                    bad.append((dp, p))
            if not (alone and hits[(dp, p)] == 0):
                mu += q
                var += q * (1.0 - q)
    count_p = _upper_tail(len(empty) - len(bad), mu, var)
    out.update(slots=n_tested, slots_hit=n_tested - len(empty),
               crosslinks_per_chain_slot=round(rate, 4), by_length=by_length,
               empty_slots=len(empty), empty_expected=round(mu, 2),
               empty_chance=count_p)
    if unreactive:
        out["unreactive_lengths"] = unreactive
    if given:
        out["method"] = "pattern"
        return out
    if count_p < EMPTY_P:
        # the period is not the reactive set: the positions the crosslinks
        # reached, each chain read the way it sits on the period
        seen = collections.defaultdict(set)
        for w in located:
            seen[w.dp].update(oriented[w.chain])
        out.update(method="list",
                   lists={int(dp): sorted(int(p) for p in seen.get(dp, ()))
                          for dp in sorted(chains_of)},
                   why=f"the period {g} from {s} leaves {len(empty)} of its "
                       f"{n_tested} slots without a crosslink where a uniform "
                       f"draw leaves {mu:.1f} (a chance of {count_p:.1e}); the "
                       f"reactive beads are read as the positions seen")
        return out
    ends = [(dp, p) for dp in active for p in slots[dp] if p in (1, dp - 2)]
    if bad and sorted(bad) == sorted(ends):
        # only the beads next to a chain end, all of them: min_dangling_dp 1
        out.update(method="pattern", min_dangling_dp=1,
                   why=f"no crosslink on the bead next to a chain end, where a "
                       f"uniform draw puts "
                       f"{sum(by_length[dp]['per_slot'] for dp, _ in ends):.1f}")
        return out
    if bad:
        out.update(method="list",
                   lists={int(dp): [int(p) for p in slots[dp] if (dp, p) not in bad]
                          for dp in sorted(chains_of)
                          if dp not in unreactive},
                   why=f"{len(bad)} slots of the period {g} from {s} "
                       f"({', '.join(f'bead {p} of DP {dp}' for dp, p in bad[:4])}"
                       f"{', ...' if len(bad) > 4 else ''}) carry no crosslink "
                       f"where a uniform draw would have put one in all but "
                       f"{EMPTY_P:g} of cases; the period less those slots is "
                       f"listed")
        for dp in unreactive:
            out["lists"][int(dp)] = []
        return out
    out["method"] = "pattern"
    return out


def _upper_tail(k: int, mu: float, var: float) -> float:
    """Chance of ``k`` or more empty slots, for a count of mean ``mu`` and variance ``var``.

    The empty slots are a sum of independent Bernoulli counts; the normal
    approximation with a continuity correction, exact at no variance.
    """
    from scipy import stats

    if k <= mu:
        return 1.0
    if var <= 1e-12:
        return 0.0
    return float(stats.norm.sf((k - 0.5 - mu) / math.sqrt(var)))


def _position_lists(located) -> dict:
    by = collections.defaultdict(set)
    for w in located:
        by[w.dp].update(w.positions)
    return {int(dp): sorted(int(p) for p in ps) for dp, ps in sorted(by.items())}


def lattice_packing(ref, n_beads: int) -> tuple[Optional[float], Optional[int], str]:
    """The packing of a lattice reference, its lattice edge, and why not.

    A reference in lattice units whose cell is a cube of a whole number of
    sites holds one bead per site, so its packing is its beads over its
    sites.
    """
    from topon.topology.chain_crosslinking import MAX_PACKING

    if ref.units != "lattice":
        return None, None, "the reference is not on a lattice"
    if ref.box is None or not np.all(np.isfinite(ref.box)):
        return None, None, "the reference has no cell"
    box = np.asarray(ref.box, float)
    L = float(box[0])
    if not np.allclose(box, L) or abs(L - round(L)) > 1e-6:
        return None, None, "the reference's cell is not a cube of whole sites"
    p = n_beads / L ** 3
    if not 0 < p <= MAX_PACKING:
        return None, None, (f"{n_beads} beads on {int(round(L))}^3 sites is a "
                            f"packing of {p:.3f}, not a lattice melt")
    return float(p), int(round(L)), ""


def packing_for(n_beads: int, L: int) -> float:
    """The packing, to six decimals, from which the generator picks ``L``."""
    from topon.topology.chain_crosslinking import lattice_size

    p = math.ceil(n_beads / L ** 3 * 1e6) / 1e6
    while lattice_size(n_beads, p) > L:
        p += 1e-6
    return round(p, 6)


def measure_chains(ref) -> dict:
    """The chains, the crosslinks and the positions of a crosslinked reference.

    Goes into the measurement record as ``crosslinked``. Everything the
    crosslink generator takes is here, and the numbers its output is held
    against: the strand DP of each class, passes per chain, the crosslinks
    within a chain and their gaps.
    """
    from topon.analysis.crosslinked import chain_statistics

    G = ref.graph
    chains = G.graph.get("chains") or {}
    walks = chain_walks(G)
    dps = collections.Counter(int(c["dp"]) for c in chains.values())
    xl = crosslink_count(G)
    gaps = intra_gaps(walks)
    st = chain_statistics(G)
    n_beads = int(sum(int(c["dp"]) for c in chains.values()))
    pos_hist = collections.Counter(p for w in walks if w.ok for p in w.positions)
    rec = {
        "chains": {"count": len(chains), "beads": n_beads,
                   "dp": {int(k): int(v) for k, v in sorted(dps.items())},
                   "sol": int(sum(1 for c in chains.values()
                                  if int(c.get("passes", 0)) == 0)),
                   "unordered": int(G.graph.get("chains_unordered", 0) or 0),
                   "not_exact": dict(collections.Counter(
                       w.why for w in walks if not w.ok))},
        "crosslinks": {**xl, "intra_chain": len(gaps),
                       "inter_chain": int(xl["count"] - len(gaps)),
                       "per_chain": (round(2.0 * xl["count"] / len(chains), 4)
                                     if chains else None)},
        "positions": {int(k): int(v) for k, v in sorted(pos_hist.items())},
        "intra_gaps": {"n": len(gaps), "odd": int(sum(g % 2 for g in gaps)),
                       "even": int(sum(1 - g % 2 for g in gaps)),
                       "min": int(min(gaps)) if gaps else None,
                       "hist": {int(k): int(v) for k, v in
                                sorted(collections.Counter(gaps).items())}},
        "passes": {"mean": st["means"].get("passes"),
                   "hist": {int(k): int(v) for k, v in sorted(
                       collections.Counter(st["passes"].astype(int)).items())}},
        "strand_dp": {cls: {"n": int(len(st[f"dp_{cls}"])),
                            "mean": st["means"].get(f"dp_{cls}")}
                      for cls in ("bridge", "dangling", "loop")},
        "reactive": infer_reactive(walks),
    }
    p, L, why = lattice_packing(ref, n_beads)
    rec["lattice"] = {"packing": p, "L": L, "why": why}
    return rec


# ---------------------------------------------------------------------------
# Scoring a build against the reference
# ---------------------------------------------------------------------------

def reference_numbers(meas, G) -> dict:
    """What every build and replicate is held against: ``meas`` of graph ``G``."""
    from topon.analysis.crosslinked import chain_statistics

    s = meas.descriptors.scalars
    return {"descriptors": meas.descriptors,
            "stats": chain_statistics(G),
            "lambda2": s.get("lambda2_core"), "path": s.get("avg_path_core"),
            "loops": s.get("n_primary_loops"),
            "intra": (meas.record.get("crosslinked") or {}).get(
                "crosslinks", {}).get("intra_chain")}


def _ks(a, b) -> Optional[float]:
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return None
    return float(stats.ks_2samp(a, b).statistic)


def _rel(x, ref) -> Optional[float]:
    if x is None or ref is None or not np.isfinite(x) or abs(ref) < 1e-12:
        return None
    return float(abs(x - ref) / abs(ref))


def score_graph(G, refn: dict, desc=None) -> dict:
    """The acceptance measures of one graph against the reference.

    ``composite`` is :func:`topon.analysis.descriptors.compare`'s default
    (each term over the reference value); ``lambda2`` and ``path`` the core's
    values, with ``d_lambda2`` and ``d_path`` their deviations over the
    reference's; ``ks_*`` the KS statistic of passes per chain and of each
    class's strand DP (None when either side has none). ``score`` is the mean
    of the composite, the two deviations, the KS statistics and the loop
    count's deviation (over the reference's, at most 1), which is what a
    sweep chooses by. A graph with no core scores a deviation of 1 for
    lambda2 and the path.
    """
    from topon.analysis.crosslinked import chain_statistics
    from topon.analysis.descriptors import compare, describe

    d = desc if desc is not None else describe(G, heavy=True, seed=0)
    s = d.scalars
    cmp = compare(d, refn["descriptors"])
    st = chain_statistics(G)
    walks = chain_walks(G)
    row = {"composite": cmp["composite"],
           "lambda2": s.get("lambda2_core"), "path": s.get("avg_path_core"),
           "d_lambda2": _rel(s.get("lambda2_core"), refn["lambda2"]),
           "d_path": _rel(s.get("avg_path_core"), refn["path"]),
           "loops": int(s.get("n_primary_loops", 0)),
           "secondary": int(s.get("n_secondary_loops", 0)),
           "bridges": int(s.get("n_bridging_chains", 0)),
           "dangling": int(s.get("n_dangling_chains", 0)),
           "sol": int(s.get("n_sol_chains", 0)),
           "junctions": int(s.get("n_junctions", 0)),
           "detached": float(1.0 - float(s.get("giant_frac_junctions", 1.0) or 1.0)),
           "crosslinks": crosslink_count(G)["count"],
           "intra": len(intra_gaps(walks)),
           "passes_mean": st["means"].get("passes"),
           "dp_means": {k: st["means"].get(f"dp_{k}")
                        for k in ("bridge", "dangling", "loop")}}
    for k in CHAIN_KEYS:
        row[f"ks_{k}"] = (_ks(st[k], refn["stats"][k])
                          if refn.get("stats") is not None else None)
    ref_loops = refn.get("loops") or 0
    # a build with no core has no lambda2 or path: the whole deviation
    for k in ("lambda2", "path"):
        if row[f"d_{k}"] is None and refn.get(k):
            row[f"d_{k}"] = 1.0
    terms = [row["composite"], row["d_lambda2"], row["d_path"]]
    terms += [row[f"ks_{k}"] for k in CHAIN_KEYS]
    terms.append(min(1.0, abs(row["loops"] - ref_loops) / max(1.0, ref_loops)))
    terms = [t for t in terms if t is not None and np.isfinite(t)]
    row["score"] = float(np.mean(terms)) if terms else None
    return row


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

def chain_types(xl: dict, walks, *, sequence: Optional[str] = None,
                repeats: int = 1, crosslink_residue: str = "Y",
                reactive_every: Optional[int] = None,
                reactive_start: Optional[int] = None) -> tuple[list, dict]:
    """``topology.crosslinking.chains``: one type per chain length.

    Returns the chain types, in the order the reference lists its chains,
    and the reactive-bead reading they rest on. A length the period leaves
    no reactive bead on gets an empty ``reactive`` list.

    Raises:
        FitError: a sequence whose chain length or crosslink residues do not
            fit the reference's chains, or a period that misses some of its
            crosslinked beads.
    """
    counts = {int(k): int(v) for k, v in xl["chains"]["dp"].items()}
    # molecules of one or two beads (solvent, ions) are no chain the
    # generator can grow; they are left out and flagged
    short = {dp: n for dp, n in counts.items() if dp < 3}
    counts = {dp: n for dp, n in counts.items() if dp >= 3}
    if sequence is not None:
        from topon.protein_network.sequence import plan_chain

        plan = plan_chain(sequence, repeats=repeats,
                          crosslink_residue=crosslink_residue)
        if set(counts) != {plan.n_residues}:
            raise FitError(
                f"the sequence gives chains of {plan.n_residues} residues and "
                f"the reference's chains are {sorted(counts)} beads long")
        sites = set(plan.crosslink_residue_indices)
        bad = [w.chain for w in walks if w.ok and w.positions
               and not set(w.positions) <= sites
               and not {w.dp - 1 - p for p in w.positions} <= sites]
        if bad:
            raise FitError(
                f"{len(bad)} chains have crosslinked beads on residues other "
                f"than {plan.crosslink_residue} of the sequence, read from "
                f"either end (chain {bad[0]} first)")
        info = {"method": "sequence", "sequence": sequence, "repeats": int(repeats),
                "crosslink_residue": plan.crosslink_residue,
                "reactive": len(sites),
                "positions": sorted(int(x) for x in sites)}
        if short:
            info["short_molecules"] = short
        return ([{"count": counts[plan.n_residues], "sequence": sequence,
                  "repeats": int(repeats),
                  "crosslink_residue": plan.crosslink_residue,
                  "name": f"dp{plan.n_residues}"}], info)
    if reactive_every is not None:
        info = infer_reactive(walks, every=reactive_every, start=reactive_start)
        if info["method"] != "pattern":
            raise FitError(
                f"the period {reactive_every}"
                + (f" from {reactive_start}" if reactive_start else "")
                + f" does not hold the reference's crosslinked beads: "
                + info.get("why", ""))
        info["method"] = "given"
    else:
        info = dict(xl["reactive"])
    if short:
        info["short_molecules"] = short
    # in the order the reference lists its chains, which is the order the
    # generator grows them in when the reference is one of its melts
    order = [dp for dp in dict.fromkeys(w.dp for w in walks) if dp in counts]
    order += sorted(set(counts) - set(order))
    unreactive = {int(k) for k in info.get("unreactive_lengths") or {}}
    types = []
    for dp in order:
        n = counts[dp]
        t = {"count": n, "dp": dp, "name": f"dp{dp}"}
        if info["method"] in ("pattern", "given"):
            if info["start"] > dp - 2 or dp in unreactive:
                t["reactive"] = []
            else:
                t.update(reactive_every=int(info["every"]),
                         reactive_start=int(info["start"]))
        elif info["method"] == "list":
            t["reactive"] = list(info["lists"].get(dp, []))
        else:
            raise FitError("no crosslinked bead of the reference could be "
                           "placed along its chain: " + info.get("why", ""))
        types.append(t)                  # an empty list: a length never crosslinked
    return types, info


def _unreactive_lengths(rows: dict) -> dict:
    """Chain lengths no crosslink reached where the period says it would have.

    ``rows`` maps a chain length to its chains, slots and hits. At the rate
    the lengths' slots were reached (crosslinked beads per chain and slot),
    a length whose chains would have caught :data:`INTRA_EXPECTED` or more
    and caught none is a chain of another kind (a diluent, an unreactive
    sol), written with no reactive bead and left out of the slot test.
    """
    total_slots = sum(r["chains"] * r["slots"] for r in rows.values())
    hits = sum(r["hits"] for r in rows.values())
    if not total_slots or not hits:
        return {}
    rate = hits / total_slots
    out = {}
    for dp, r in rows.items():
        expected = rate * r["chains"] * r["slots"]
        if r["hits"] == 0 and expected >= INTRA_EXPECTED:
            out[int(dp)] = round(expected, 2)
    return out


def crosslink_config(types: list, crosslinks: int, *, packing: float,
                     contact: float, min_gap: int, seed: int, density: float,
                     name: str, min_dangling_dp: Optional[int] = None) -> dict:
    """The fitted config for ``topology.source: "crosslink"``.

    ``min_dangling_dp`` is written only when the reference asks for it (no
    crosslink on the bead next to a chain end); unset it is the route's
    default, 0 on the coarse-grained route.
    """
    xl = {"chains": copy.deepcopy(types), "crosslinks": int(crosslinks),
          "packing": float(packing), "contact_radius": float(contact),
          "min_gap": int(min_gap), "seed": int(seed)}
    if min_dangling_dp is not None:
        xl["min_dangling_dp"] = int(min_dangling_dp)
    return {
        "study": {"name": name, "output_dir": "./output"},
        "topology": {"source": "crosslink", "crosslinking": xl},
        "chemistry": {"model_type": "coarse_grained",
                      "target_density": round(float(density), 6)},
        "simulation": {"protocol": "pushoff",
                       "rho_final": round(float(density), 6)},
    }


def _parity_choice(gaps: dict, info: dict) -> tuple[Optional[float], str]:
    """The contact radius the gaps within chains point at, or None to sweep.

    On the cubic lattice a chain changes sublattice at every step, so under
    face contacts two of its beads touch across an odd gap, except across
    the box of an odd lattice (1.2 % even with chains longer than the box);
    under face and edge contacts 16 to 24 % of the gaps are odd. At most
    :data:`EVEN_FACE` even reads as face contacts, at least
    :data:`EVEN_EDGE` as face and edge, anything between, fewer than
    :data:`PARITY_MIN` gaps or an even period (every gap even whatever the
    rule) is built both ways.
    """
    from scipy import stats

    n, even = gaps["n"], gaps["even"]
    period = info.get("every") if info.get("method") in ("pattern", "given") else None
    if period is not None and period % 2 == 0:
        return None, (f"the reactive period {period} is even, so every gap "
                      f"within a chain is even whatever the contact rule")
    if n < PARITY_MIN:
        return None, (f"{n} crosslinks within a chain are too few to tell the "
                      f"contact rule ({PARITY_MIN} needed)")
    frac = even / n
    if frac <= EVEN_FACE:
        chance = float(stats.binom.cdf(even, n, 0.75))
        return FACE, (f"{n - even} of the {n} crosslinks within a chain close "
                      f"across an odd number of bonds, the cubic lattice's "
                      f"parity under face contacts (under face and edge "
                      f"contacts a quarter or fewer are odd, and this many "
                      f"has a chance of {chance:.1e})")
    if frac >= EVEN_EDGE:
        return FACE_EDGE, (f"{even} of the {n} crosslinks within a chain close "
                           f"across an even number of bonds, which face "
                           f"contacts make only across the box")
    return None, (f"{even} of the {n} crosslinks within a chain close across an "
                  f"even number of bonds, between the two contact rules")


def _min_gap(gaps: dict, flags: list) -> int:
    if not gaps["n"]:
        return DEFAULT_MIN_GAP
    g = int(gaps["min"])
    if g < 3:
        n_small = sum(v for k, v in gaps["hist"].items() if int(k) < 3)
        _flag(flags, "warn", "loops the builder cannot make",
              f"{n_small} crosslinks within a chain are fewer than 3 bonds "
              f"apart; the builder needs a primary loop of two beads, so "
              f"min_gap is 3 and those are not made.")
        return 3
    return min(DEFAULT_MIN_GAP, g)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_crosslinked(ref, meas, *, route: str = "crosslink", seeds: int = 2,
                    seed: int = 1, density: Optional[float] = None,
                    packing: Optional[float] = None,
                    contact_radius: Optional[float] = None,
                    sequence: Optional[str] = None, repeats: int = 1,
                    crosslink_residue: str = "Y",
                    reactive_every: Optional[int] = None,
                    reactive_start: Optional[int] = None,
                    cutoffs: Optional[Sequence[float]] = None,
                    name: Optional[str] = None,
                    log: Optional[Callable[[str], None]] = None,
                    started: Optional[tuple] = None) -> FitResult:
    """A config that regenerates the crosslinked reference ``ref``.

    ``route`` ``"crosslink"`` writes ``topology.source: "crosslink"`` (the
    default); ``"lattice"`` the lattice route. ``packing``, ``contact_radius``,
    ``density``, a ``sequence`` or a period (``reactive_every``,
    ``reactive_start``) replace what would be read or swept. ``started`` is
    the ``(wall, cpu)`` clock of the caller, so the report's time covers the
    reading.
    """
    say = log or (lambda msg: None)
    wall0, cpu0 = started or (time.perf_counter(), time.process_time())
    flags: list = []
    rec = meas.record
    xl = rec["crosslinked"]
    if route not in ("crosslink", "lattice"):
        raise FitError(f"route {route!r}: use 'crosslink' or 'lattice'")
    if cutoffs and route != "lattice":
        raise FitError("the sweep cutoffs are the lattice route's (and the "
                       "end-linked fit's); the crosslink generator has none, "
                       "so pass --route lattice or leave them out")
    if not xl["chains"]["count"]:
        raise FitError("the reference carries no chains: a crosslinked data file "
                       "is read one molecule per chain. topon's own build writes "
                       "each strand as a molecule; fit its "
                       "topology/crosslinked_melt.npz instead")
    walks = chain_walks(ref.graph)
    types, info = chain_types(xl, walks, sequence=sequence, repeats=repeats,
                              crosslink_residue=crosslink_residue,
                              reactive_every=reactive_every,
                              reactive_start=reactive_start)
    n_xl = int(xl["crosslinks"]["count"])
    say(f"{xl['chains']['count']} chains ({_dp_text(xl['chains']['dp'])}), "
        f"{n_xl} crosslinks, {xl['crosslinks']['intra_chain']} within a chain; "
        f"reactive beads: {_reactive_text(info)}")
    _reference_flags(xl, info, flags)

    if density is None:
        density = ref.density
        if density is None:
            density = FALLBACK_DENSITY
            _flag(flags, "warn", "no density in the input",
                  f"chemistry.target_density is written at {FALLBACK_DENSITY}, "
                  f"the Kremer-Grest melt; the generator holds every strand's "
                  f"chord to its contour at that density. Pass --density.")
    name = name or f"{Path(ref.path).stem}_fit"
    seed_list = [int(seed) + i for i in range(max(1, int(seeds)))]
    refn = reference_numbers(meas, ref.graph)

    if route == "lattice":
        return _fit_lattice(ref, meas, xl, info, refn, seed=seed,
                            seed_list=seed_list, density=density,
                            cutoffs=cutoffs, name=name,
                            flags=flags, say=say, clock=(wall0, cpu0))

    # --- what is read, and what is swept ------------------------------------
    choices = {}
    lat = xl["lattice"]
    # the beads of the chains written, which the generator sizes its lattice by
    n_beads = int(sum(int(dp) * int(n) for dp, n in xl["chains"]["dp"].items()
                      if int(dp) >= 3))
    if packing is not None:
        packs = [float(packing)]
        choices["packing"] = {"value": float(packing), "source": "given"}
    elif lat["packing"] is not None:
        packs = [packing_for(n_beads, lat["L"])]
        choices["packing"] = {"value": packs[0], "source": "measured",
                              "why": f"{n_beads} beads on {lat['L']}^3 sites"}
    else:
        packs = list(PACKINGS)
        choices["packing"] = {"source": "swept", "candidates": packs,
                              "why": lat["why"]}
    if contact_radius is not None:
        contacts = [float(contact_radius)]
        choices["contact_radius"] = {"value": float(contact_radius),
                                     "source": "given"}
    else:
        c, why = _parity_choice(xl["intra_gaps"], info)
        contacts = [c] if c is not None else [FACE, FACE_EDGE]
        choices["contact_radius"] = ({"value": c, "source": "measured", "why": why}
                                     if c is not None else
                                     {"source": "swept", "candidates": contacts,
                                      "why": why})
    min_gap = _min_gap(xl["intra_gaps"], flags)
    choices["min_gap"] = {"value": min_gap, "source": "measured",
                          "why": (f"the reference's smallest gap within a chain "
                                  f"is {xl['intra_gaps']['min']}"
                                  if xl["intra_gaps"]["n"] else
                                  "no crosslink within a chain; the generator's "
                                  "default until the builds say otherwise")}

    cands = [{"packing": p, "contact_radius": c} for p, c in
             itertools.product(packs, contacts)]
    say(f"building {len(cands)} setting(s) on seeds {seed_list}: "
        + "; ".join(f"packing {c['packing']:g}, contact {c['contact_radius']:g}"
                    for c in cands))
    t_sweep0 = time.perf_counter()

    def config_for(cand, gap):
        return crosslink_config(types, n_xl, packing=cand["packing"],
                                contact=cand["contact_radius"], min_gap=gap,
                                seed=seed, density=density, name=name,
                                min_dangling_dp=info.get("min_dangling_dp"))

    def run(gap):
        rows = []
        for cand in cands:
            rows += _build_rows(config_for(cand, gap), cand, seed_list, refn,
                                gap, say)
        summary = _summarise(rows, cands, gap)
        return rows, summary, [s for s in summary if s["n_ok"] == s["n_seeds"]
                               and s["score"] is not None]

    rows, summary, ok = run(min_gap)
    ref_intra = int(xl["crosslinks"]["intra_chain"])
    made = [r["intra"] for r in rows if r.get("intra") is not None]
    if ref_intra == 0 and made and float(np.mean(made)) >= INTRA_EXPECTED:
        # None of the reference's crosslinks is within a chain and the
        # builds make them: every setting again, with them forbidden.
        forbid = max(int(k) for k in xl["chains"]["dp"])
        _flag(flags, "note", "no crosslink within a chain",
              f"the reference has none of its {n_xl} crosslinks between two "
              f"beads of one chain, and the builds at min_gap {min_gap} make "
              f"{np.mean(made):.1f} on average; min_gap is set to {forbid}, "
              f"past the longest chain, so the generator makes none either, "
              f"and every setting is built again with it.")
        choices["min_gap"] = {"value": forbid, "source": "measured",
                              "why": f"no crosslink within a chain in the "
                                     f"reference, {np.mean(made):.1f} a build at "
                                     f"min_gap {min_gap}"}
        min_gap = forbid
        rows2, summary2, ok = run(forbid)
        rows += rows2
        summary += summary2
    if not ok:
        errors = sorted({r.get("error", "") for r in rows if r.get("error")})
        raise FitError("no setting built on every seed: " + "; ".join(errors[:3]))
    chosen = dict(min(ok, key=lambda s: s["score"]))
    if choices["packing"]["source"] == "swept":
        # one step finer: the packings halfway to the best one's neighbours
        grid = sorted(packs)
        i = grid.index(chosen["packing"])
        mids = [round(0.5 * (chosen["packing"] + grid[j]), 4)
                for j in (i - 1, i + 1) if 0 <= j < len(grid)]
        finer = [{"packing": x, "contact_radius": chosen["contact_radius"]}
                 for x in mids]
        say("refining the packing: " + ", ".join(f"{x:g}" for x in mids))
        rows3 = []
        for cand in finer:
            rows3 += _build_rows(config_for(cand, min_gap), cand, seed_list,
                                 refn, min_gap, say)
        summary3 = _summarise(rows3, finer, min_gap)
        rows += rows3
        summary += summary3
        ok = ok + [s for s in summary3 if s["n_ok"] == s["n_seeds"]
                   and s["score"] is not None]
        choices["packing"]["candidates"] = packs + mids
        chosen = dict(min(ok, key=lambda s: s["score"]))

    def tie(ranked):
        gap = ranked[1]["score"] - ranked[0]["score"]
        return gap < max(ranked[0]["score_sd"], ranked[1]["score_sd"])

    settled = None
    if len(ok) > 1 and tie(sorted(ok, key=lambda s: s["score"])) \
            and len(seed_list) < TIE_SEEDS:
        # a choice the seeds' scatter decides: the best few again on more seeds
        top = sorted(ok, key=lambda s: s["score"])[:TIE_TOP]
        extra = list(range(seed_list[-1] + 1, seed_list[-1] + 1 + TIE_SEEDS
                           - len(seed_list)))
        say(f"a tie within the seeds' scatter: the best {len(top)} settings on "
            f"seeds {extra} as well")
        for cand in top:
            c = {k: cand[k] for k in ("packing", "contact_radius")}
            rows += _build_rows(config_for(c, cand["min_gap"]), c, extra, refn,
                                cand["min_gap"], say)
        again = []
        for cand in top:
            c = {k: cand[k] for k in ("packing", "contact_radius")}
            again += _summarise(rows, [c], cand["min_gap"])
        again = [s for s in again if s["n_ok"] == s["n_seeds"] and s["score"] is not None]
        if again:
            settled = {"seeds": seed_list + extra, "settings": again}
            keys = {(s["packing"], s["contact_radius"], s["min_gap"]) for s in again}
            summary = [s for s in summary if (s["packing"], s["contact_radius"],
                                              s["min_gap"]) not in keys] + again
            ok = again
            chosen = dict(min(ok, key=lambda s: s["score"]))
    t_sweep = time.perf_counter() - t_sweep0
    for key in ("packing", "contact_radius"):
        if choices[key]["source"] == "swept":
            choices[key]["value"] = chosen[key]
    if len(ok) > 1:
        ranked = sorted(ok, key=lambda s: s["score"])
        chosen["runner_up"] = {k: ranked[1][k] for k in (
            "packing", "contact_radius", "score")}
        chosen["within_scatter"] = bool(tie(ranked))
        if chosen["within_scatter"]:
            _flag(flags, "note", "choice within seed scatter",
                  f"packing {chosen['packing']:g}, contact "
                  f"{chosen['contact_radius']:g} scored {chosen['score']:.3f} "
                  f"and packing {ranked[1]['packing']:g}, contact "
                  f"{ranked[1]['contact_radius']:g} {ranked[1]['score']:.3f}"
                  + (f" over {len(settled['seeds'])} seeds" if settled else "")
                  + "; the gap is smaller than the spread between seeds.")
    if settled:
        chosen["settled_on"] = settled["seeds"]

    config = config_for({k: chosen[k] for k in ("packing", "contact_radius")},
                        chosen["min_gap"])
    for note in rec.get("notes", []):
        _flag(flags, "note", "reference", note)
    for issue in _doctor(config):
        if issue["level"] in ("warn", "error"):
            _flag(flags, issue["level"], f"doctor: {issue['rule']}",
                  issue["message"])
    report = _report(rec, "crosslink", choices, info, rows, summary, chosen,
                     flags, t_sweep, wall0, cpu0, seed_list)
    return FitResult(config=config, report=report, measurement=meas,
                     flags=flags)


def _build_rows(config, cand, seed_list, refn, min_gap, say) -> list:
    rows = []
    for s in seed_list:
        row = {**cand, "min_gap": int(min_gap), "seed": int(s)}
        t0 = time.perf_counter()
        try:
            G, _man = build_graph(config, seed=s)
        except Exception as exc:          # a setting that cannot build
            row["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0]
            rows.append(row)
            say(f"  packing {cand['packing']:g}, contact "
                f"{cand['contact_radius']:g}, seed {s}: failed, {row['error']}")
            continue
        row.update(score_graph(G, refn))
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
        say(f"  packing {cand['packing']:g}, contact {cand['contact_radius']:g}, "
            f"min_gap {min_gap}, seed {s}: score {_f(row['score'])} (composite "
            f"{_f(row['composite'])}, lambda2 {_f(row['lambda2'], '.4f')}, path "
            f"{_f(row['path'], '.2f')}, loops {row['loops']}) in "
            f"{row['seconds']:.0f} s")
    return rows


def _summarise(rows, cands, min_gap) -> list:
    out = []
    for cand in cands:
        mine = [r for r in rows if r["packing"] == cand["packing"]
                and r["contact_radius"] == cand["contact_radius"]
                and r["min_gap"] == min_gap]
        ok = [r for r in mine if r.get("score") is not None]
        sc = [r["score"] for r in ok]
        entry = {**cand, "min_gap": None if min_gap is None else int(min_gap),
                 "n_seeds": len(mine),
                 "n_ok": len(ok),
                 "score": float(np.mean(sc)) if sc else None,
                 "score_sd": float(np.std(sc, ddof=1)) if len(sc) > 1 else 0.0}
        for k in ("composite", "d_lambda2", "d_path", "loops", "intra",
                  *(f"ks_{c}" for c in CHAIN_KEYS)):
            vals = [r[k] for r in ok if r.get(k) is not None]
            entry[k if k != "intra" else "intra_mean"] = (float(np.mean(vals))
                                                          if vals else None)
        out.append(entry)
    return out


def _report(rec, route, choices, info, rows, summary, chosen, flags, t_sweep,
            wall0, cpu0, seed_list) -> dict:
    return {
        "architecture": "crosslinked",
        "route": route,
        "reference": {k: v for k, v in rec.items() if k != "descriptors"},
        "reference_descriptors": {k: rec["descriptors"].get(k) for k in (
            "edge_shortest_cycle_mean", "frac_odd_cycles", "transitivity_core",
            "lambda2_core", "avg_path_core", "n_primary_loops",
            "n_secondary_loops", "giant_frac_junctions")},
        "reactive": info,
        "choices": choices,
        "builds": {"seeds": seed_list, "rows": rows, "summary": summary},
        "chosen": chosen,
        "flags": flags,
        "seconds": {"sweep": round(t_sweep, 1),
                    "total": round(time.perf_counter() - wall0, 1),
                    "cpu": round(time.process_time() - cpu0, 1)},
    }


def _reference_flags(xl: dict, info: dict, flags: list) -> None:
    ch, cx = xl["chains"], xl["crosslinks"]
    if ch["unordered"]:
        _flag(flags, "warn", "chains without an order",
              f"{ch['unordered']} molecules have no single backbone path (an "
              f"intra-chain crosslink of the backbone's bond type, or a "
              f"branched molecule; give --crosslink-bond-type); they are not "
              f"among the chains written.")
    if ch["not_exact"]:
        _flag(flags, "warn", "crosslinks the generator cannot place",
              "chains whose crosslinked beads cannot be placed exactly: "
              + ", ".join(f"{n} ({why})" for why, n in ch["not_exact"].items())
              + ". The generator makes pairwise crosslinks between interior "
                "beads, none on neighbouring beads.")
    if cx["fused_junctions"] or cx["odd_junctions"]:
        _flag(flags, "warn", "junctions that are not one crosslink",
              f"{cx['fused_junctions']} junctions carry more than four strands "
              f"(crosslinks on neighbouring beads, fused) and "
              f"{cx['odd_junctions']} an odd number; the generator makes "
              f"every junction one crosslink of two chain beads.")
    if info.get("unreactive_lengths"):
        _flag(flags, "note", "chains with no reactive bead",
              "chains of " + ", ".join(
                  f"DP {dp} (about {exp:g} crosslinks expected at the others' "
                  f"rate)" for dp, exp in info["unreactive_lengths"].items())
              + " carry no crosslink, and are written with no reactive bead.")
    if info.get("short_molecules"):
        _flag(flags, "warn", "molecules of fewer than three beads",
              ", ".join(f"{n} of {dp} bead(s)" for dp, n in
                        sorted(info["short_molecules"].items()))
              + " (solvent, ions) are no chain the generator grows, and are "
                "left out of the chains written.")
    if info.get("min_dangling_dp"):
        _flag(flags, "note", "no crosslink next to a chain end",
              f"{info.get('why', '')}; min_dangling_dp 1 is written, which "
              f"leaves those beads out.")
    if info.get("method") == "list":
        _flag(flags, "note", "reactive beads as a list",
              f"{info.get('why', '')}. The config lists the beads a crosslink "
              f"reached, so a reactive bead no crosslink reached is not among "
              f"them.")
    elif info.get("method") == "pattern" and info.get("flipped"):
        _flag(flags, "note", "chains read from their far end",
              f"{info['flipped']} chains sit on the period from their other "
              f"end; a chain is the same read either way, so they count as "
              f"on it.")


# ---------------------------------------------------------------------------
# The lattice route
# ---------------------------------------------------------------------------

def lattice_candidates(rule: Optional[float], n: int) -> list:
    """The simple-cubic shell counts around the p75 rule: one in, one out.

    The first shell, the nearest-neighbour scaffold, is offered only when
    the rule is within it; see the module doc for why it is no control row.

    The end-linked p95 rule overshoots on these networks
    (2.56 and 2.35) and the p75 of the chord pointing at the matching range
    (1.46 and 1.45 for two and three shells).
    """
    from topon.topology.shells import sc_shell_cutoff

    if rule is None:
        return candidate_cutoffs(None, n)
    s = max(1, shells_within(rule))
    cut = [sc_shell_cutoff(k) for k in (s - 1, s, s + 1)
           if k >= 2 or (k == 1 and s == 1)]
    cut = [c for c in cut if c < n / 2.0] or [CONTROL_CUTOFF]
    return [describe_cutoff(c) for c in reversed(cut)]


def _fit_lattice(ref, meas, xl, info, refn, *, seed, seed_list, density,
                 cutoffs, name, flags, say, clock) -> FitResult:
    """``architecture: "random_crosslinked"``: a sculpt covered with chains."""
    wall0, cpu0 = clock
    rec = meas.record
    dps = {int(k): int(v) for k, v in xl["chains"]["dp"].items()}
    dp = int(round(sum(k * v for k, v in dps.items()) / sum(dps.values())))
    if len(dps) > 1:
        _flag(flags, "warn", "several chain lengths",
              f"the lattice route's cover takes one chain length and the "
              f"reference has {_dp_text(dps)}; the mean, {dp}, is written.")
    if info.get("method") in ("pattern", "given"):
        every = int(info["every"])
        if int(info["start"]) != 1:
            _flag(flags, "warn", "reactive beads from bead 1",
                  f"the cover places crosslinks every {every} beads from bead "
                  f"1, and the reference's period starts at bead "
                  f"{info['start']}.")
    else:
        every = 1
        _flag(flags, "warn", "reactive beads as every bead",
              f"the cover takes a period from bead 1 and the reference's "
              f"reactive beads are {info.get('method')}; every bead is "
              f"written reactive.")
    top_eff = max((int(d) for d, n in rec["target"].items() if n), default=1)
    top_chem = max((int(d) for d in rec["pf_chemical"]), default=top_eff)
    max_f = max(top_eff, top_chem, 4)
    cell = choose_cell(rec["n_active_sites"], "SC")
    rule = None
    if meas.chords is not None and len(meas.chords) and ref.box is not None:
        rule = rule_of_thumb(float(np.percentile(meas.chords, 75)), ref.box,
                             cell["n"])
    cands = ([describe_cutoff(c) for c in cutoffs] if cutoffs
             else lattice_candidates(rule, cell["n"]))
    config = base_config(rec, cell, lattice="SC", mix=None, max_f=max_f,
                         seed=seed, dp=float(dp), pdi=1.0, density=density,
                         name=name, endlinked=False)
    gen = config["topology"]["generator"]
    gen["architecture"] = "random_crosslinked"
    gen["min_giant_fraction"] = min(float(gen.get("min_giant_fraction", 1.0)),
                                    LATTICE_FLOOR)
    config["assignment"]["chains"] = {"dp": dp, "reactive_every": every,
                                      "chord_floor": True, "seed": int(seed)}
    say(f"lattice route: cell {cell['lattice_size']} SC, p75 rule "
        + (f"{rule:.2f}" if rule is not None else "n/a") + "; sweeping "
        + ", ".join(f"{c['cutoff']:.2f}" for c in cands))
    t0 = time.perf_counter()
    sw = sweep(config, meas.descriptors, cands, seed_list,
               control=None, log=say)
    chosen = sw["chosen"]
    if chosen is None:
        errors = sorted({r.get("error", "") for r in sw["rows"] if r.get("error")})
        raise FitError("no candidate cutoff built on every seed: "
                       + "; ".join(errors[:3]))
    config = with_cutoff(config, chosen["cutoff"])
    rows = []
    for s in seed_list:
        G, _man = build_graph(config, seed=s)
        rows.append({"seed": s, "cutoff": chosen["cutoff"],
                     **score_graph(G, refn)})
    t_sweep = time.perf_counter() - t0
    _flag(flags, "note", "the lattice route",
          "Measured on the BFM references, the cover reproduces passes per chain "
          "and the bridge and dangling DPs but not the loop DPs, and the "
          "sculpt leaves 2 to 5 % of the active sites in small pieces (a "
          "connectivity floor of 0.95 is written). The crosslink route is "
          "the default for these references.")
    for note in rec.get("notes", []):
        _flag(flags, "note", "reference", note)
    for issue in _doctor(config):
        if issue["level"] in ("warn", "error"):
            _flag(flags, issue["level"], f"doctor: {issue['rule']}",
                  issue["message"])
    choices = {"cell": cell, "rule_p75": rule, "chain_dp": dp,
               "reactive_every": every}
    summary = _summarise([{**r, "packing": None, "contact_radius": None,
                           "min_gap": None} for r in rows],
                         [{"packing": None, "contact_radius": None}], None)
    report = _report(rec, "lattice", choices, info, rows, summary,
                     {**chosen, **summary[0]}, flags, t_sweep, wall0, cpu0,
                     seed_list)
    report["sweep"] = sw
    return FitResult(config=config, report=report, measurement=meas,
                     flags=flags)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _dp_text(dps: dict) -> str:
    items = sorted((int(k), int(v)) for k, v in dps.items())
    if len(items) <= 4:
        return ", ".join(f"{v} of DP {k}" for k, v in items)
    return f"DP {items[0][0]} to {items[-1][0]} ({len(items)} lengths)"


def _reactive_text(info: dict) -> str:
    m = info.get("method")
    if m in ("pattern", "given"):
        return (f"every {info['every']} from bead {info['start']}"
                + (f" (given)" if m == "given" else
                   f", {info.get('slots_hit')} of {info.get('slots')} slots reached"))
    if m == "sequence":
        return (f"the {info['crosslink_residue']} residues of the sequence "
                f"({info['reactive']} a chain)")
    if m == "list":
        n = sum(len(v) for v in info.get("lists", {}).values())
        return f"{n} positions seen, as a list"
    return "none located"


def format_fit_crosslinked(result: FitResult) -> str:
    """The fit of a crosslinked reference for the terminal."""
    r = result.report
    ref = r["reference"]
    xl = ref["crosslinked"]
    cfg = result.config
    lines = [f"topon fit  {ref['input']}  (crosslinked along its chains, "
             f"{r['route']} route)", ""]
    ch, cx = xl["chains"], xl["crosslinks"]
    lines.append(f"  chains    : {ch['count']} ({_dp_text(ch['dp'])}), {ch['beads']} "
                 f"beads, {ch['sol']} with no crosslink")
    lines.append(f"  crosslinks: {cx['count']}, {cx['intra_chain']} within a chain "
                 f"(gaps {xl['intra_gaps']['odd']} odd, "
                 f"{xl['intra_gaps']['even']} even, smallest "
                 f"{xl['intra_gaps']['min']}), {cx['per_chain']} per chain")
    st = ref["crosslinked"]["strand_dp"]
    lines.append("  strands   : " + ", ".join(
        f"{st[c]['n']} {c} (DP {st[c]['mean']:.1f})" if st[c]["mean"] is not None
        else f"0 {c}" for c in ("bridge", "dangling", "loop"))
        + f"; passes {xl['passes']['mean']:.2f} a chain")
    lines.append(f"  reactive  : {_reactive_text(r['reactive'])}")
    d = r["reference_descriptors"]
    lines.append(f"  core      : lambda2 {_f(d.get('lambda2_core'), '.4f')}, mean "
                 f"path {_f(d.get('avg_path_core'), '.2f')}")
    lines.append("")
    if r["route"] == "crosslink":
        for key in ("packing", "contact_radius", "min_gap"):
            c = r["choices"][key]
            val = c.get("value")
            lines.append(f"  {key:<10s}: {val:g} ({c['source']})"
                         + (f": {c['why']}" if c.get("why") else ""))
        lines.append(f"  settings  : score over seeds "
                     + ",".join(str(s) for s in r["builds"]["seeds"])
                     + " (mean of composite, lambda2 and path deviations, "
                       "chain KS, loops)")
        for s in r["builds"]["summary"]:
            mark = ("<- chosen" if (s["packing"], s["contact_radius"], s["min_gap"])
                    == (r["chosen"]["packing"], r["chosen"]["contact_radius"],
                        r["chosen"]["min_gap"]) else "")
            lines.append(f"              packing {s['packing']:<8g} contact "
                         f"{s['contact_radius']:<4g} min_gap {s['min_gap']:<4d} "
                         + (f"{s['score']:.3f} +- {s['score_sd']:.3f}"
                            if s["score"] is not None else "failed")
                         + f"  {mark}")
        c = r["chosen"]
        lines.append(f"  builds    : composite {_f(c.get('composite'))}, lambda2 "
                     f"off by {_pc(c.get('d_lambda2'))}, path by "
                     f"{_pc(c.get('d_path'))}, KS passes "
                     f"{_f(c.get('ks_passes'))}, bridge DP "
                     f"{_f(c.get('ks_dp_bridge'))}, dangling DP "
                     f"{_f(c.get('ks_dp_dangling'))}, loop DP "
                     f"{_f(c.get('ks_dp_loop'))}; loops {_f(c.get('loops'), '.1f')} "
                     f"against {ref['crosslinked']['strand_dp']['loop']['n']}")
        x = cfg["topology"]["crosslinking"]
        lines.append("")
        lines.append(f"  config    : topology.source crosslink, {x['crosslinks']} "
                     f"crosslinks, packing {x['packing']}, contact "
                     f"{x['contact_radius']}, min_gap {x['min_gap']}, seed "
                     f"{x['seed']}, density {cfg['chemistry']['target_density']}")
        for t in x["chains"]:
            lines.append("              " + ", ".join(
                f"{k} {v if not isinstance(v, list) else _list_text(v)}"
                for k, v in t.items()))
    else:
        gen = cfg["topology"]["generator"]
        c = r["chosen"]
        lines.append(f"  config    : lattice route, {gen['lattice_type']} "
                     f"{gen['lattice_size']}, cutoff {gen['neighbour_cutoff']:.2f}, "
                     f"{gen['degree_distribution']}, chains of "
                     f"{cfg['assignment']['chains']['dp']} every "
                     f"{cfg['assignment']['chains']['reactive_every']}")
        lines.append(f"  builds    : composite {_f(c.get('composite'))}, lambda2 "
                     f"off by {_pc(c.get('d_lambda2'))}, path by "
                     f"{_pc(c.get('d_path'))}, KS passes {_f(c.get('ks_passes'))}, "
                     f"bridge DP {_f(c.get('ks_dp_bridge'))}, loop DP "
                     f"{_f(c.get('ks_dp_loop'))}")
    if r["flags"]:
        lines.append("")
        for f in r["flags"]:
            lines.append(f"  [{f['level']}] {f['what']}: {f['detail']}")
    s = r["seconds"]
    lines.append("")
    lines.append(f"  time      : builds {s['sweep']:.0f} s, total {s['total']:.0f} s "
                 f"(CPU {s['cpu']:.0f} s)")
    return "\n".join(lines)


def _list_text(v: list) -> str:
    return (str(v) if len(v) <= 8 else
            f"[{', '.join(str(x) for x in v[:6])}, ... {v[-1]}] ({len(v)})")


def _f(x, fmt=".3f") -> str:
    if x is None:
        return "-"
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def _pc(x) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"
