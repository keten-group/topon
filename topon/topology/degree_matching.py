"""Exact per-degree sculpting: a degree-constrained subgraph of the lattice.

The strict sculptor in :mod:`topon.topology.generator_python` reaches a
target degree distribution by removing candidate edges one at a time,
never adding one back. That works when the target leaves room above the
ceiling, and it is the only search that records a move history, so it is
what the sculpting animation replays. It cannot reach a near-complete
tetrafunctional target: on a lattice with ``z > max_functionality`` stage
3 prunes every site to the ceiling greedily and overshoots the edge
budget, stage 4 can then only remove more, and the strict degree-1 stage
fails almost every trial when a tenth of the sites have to be dangling
ends (``bond_create_validation/REPORT.md`` section 2; measured with
``scripts/sculpt_probe.py`` and ``scripts/stage4_diag.py``).

The search here is the other way round. It starts from no edges, assigns
every site a target degree drawn from the requested multiset, and builds
a subgraph of the candidate edges whose degree sequence is exactly that
multiset:

1. assign a target degree ``t_i`` to every site by a random permutation
   of the requested counts; the sites left over are vacancies (``t = 0``)
2. greedy random fill: add candidate edges while both ends have spare
   capacity, with one rule -- a degree-1 site (a dangling chain end) may
   only bond to a site with ``t >= 2``, so an end always attaches to a
   junction, as it does in an end-linked network
3. augmenting paths: for every site still below target, walk alternating
   paths (add an absent candidate edge, remove a present one, ...) to
   another deficient site and flip the path. This is exact on a bipartite
   scaffold (SC, BCC, Diamond) and a heuristic through the odd cycles of
   FCC and MIX, where a leftover shows up as a residual
4. repair a locally infeasible draw: swap targets between a deficient
   site and a saturated site of lower target (never a degree-1 site), or
   relocate a vacancy, then augment again
5. optional forced double edges, placed before the fill and fixed for the
   rest of the search, so parallel strands (secondary loops) come out
   with prescribed endpoint degrees
6. accept when every site reached its target and the giant component
   holds at least ``min_giant_fraction`` of the active sites

Ported from ``bond_create_validation/scripts/degree_matching.py``, which
the study measured at 0.3 to 2 s on 2 700 to 3 500 sites, up to 10^5
candidate edges and vacancy fractions to 28 %, on SC / BCC / FCC / MIX at
any cutoff; the smaller cells in this package's tests land in 0.02 to
0.5 s. It fails on Diamond once dangling-end sites are required: a
4-coordinated scaffold at ``max_functionality = 4`` has no spare
candidate edge, so a site at the ceiling must use every neighbour it has
and any vacancy or dangling end beside it takes away capacity nothing can
give back. With most of a request pinned there, retrying draws the same
forced assignment, so :func:`build_exact_graph` says so instead.

The exact search has no edge-removal history, so a graph it produces
carries no ``move_history``: the sculpting animation in
``assets/gallery/`` needs ``search: "strict"``.

The fallback
------------
Some targets defeat the random deal on a bipartite scaffold. SC, BCC and
Diamond at their first shell only join sites of opposite sublattices, so
every edge adds one to each side, and a degree sequence is realisable
only if the targets on the two sublattices sum to the same number. A
random deal almost never balances (the manuscript's hardest target puts
44 dangling ends and 54 six-fold junctions on 216 SC sites; its deals are
off by 29 degree units on average and balance 2.6 % of the time), and
step 4 swaps targets across the sublattices at random, so it walks the
imbalance around instead of removing it. Once :data:`FALLBACK_AFTER`
attempts in a row have ended with unfilled degree units, the search
switches to a fallback that changes the deal and the repair, and nothing
else:

- the deal is balanced: after the random permutation, targets are swapped
  between the sublattices (heavier side down by a smaller-or-equal step)
  until their sums match
- the repair is driven by the residual: the failed augmenting search from
  a deficient site marks the region that is short of partners, a target
  from that region (on the site's own sublattice) is swapped with a lower
  one outside it on the same sublattice, which keeps the balance, and the
  swap is kept only if the subgraph did not lose an edge (otherwise it is
  undone)

On a scaffold that is not bipartite the balancing is skipped and the
sublattice restriction does not apply. Attempts that fail on
connectivity with every degree filled do not count toward the switch, and
nothing in the random-deal path draws a different number, so a target
the random deal reaches gives the same graph for the same seed as before.
"""
from __future__ import annotations

import collections
import time
from typing import Mapping, Optional

import networkx as nx
import numpy as np

#: Fraction of the active sites the largest component must hold for a
#: sculpt to be accepted. The DP-20 end-linked reference network sits at
#: 0.997 and the DP-100 one at 1.0 (``REPORT.md`` section 1), so 0.99
#: admits the small sol fraction a real network has without admitting a
#: shattered graph.
DEFAULT_MIN_GIANT_FRACTION = 0.99

#: Seeds tried before :func:`build_exact_graph` gives up. Every accepted
#: case in the validation study landed on the first or second.
DEFAULT_ATTEMPTS = 6

#: Rounds of augmentation before the target-repair loop takes over, and
#: repair steps before the attempt is abandoned. Both are the values the
#: study measured with; they only bound work, they do not shape results.
MAX_AUGMENT_ROUNDS = 6
MAX_REPAIR_STEPS = 20000

#: Attempts in a row that must end with unfilled degree units before the
#: fallback deal and repair take over, and the attempts the fallback then
#: gets in :func:`build_exact_graph`. Measured on the generator benchmark
#: (SC 6^3 and 12^3, one to three shells, targets A and T, 100 seeds), the
#: random deal never ended short more than once in a row where it
#: succeeds, so six leaves its results untouched.
FALLBACK_AFTER = 6
DEFAULT_FALLBACK_ATTEMPTS = 6

#: Tries at balancing the two sublattices, per site, before the fallback
#: deal gives up and lets the attempt fail on its residual.
BALANCE_TRIES_PER_SITE = 100


class ExactSculptError(RuntimeError):
    """The exact search could not reach the requested degree counts."""


# ---------------------------------------------------------------------------
# Reading the request
# ---------------------------------------------------------------------------

def parse_degree_distribution(spec: Optional[str]) -> tuple[dict[int, int], int]:
    """Read a ``degree_distribution`` string into counts and an edge budget.

    ``"0:43,1:217,4:1975"`` gives ``({0: 43, 1: 217, 4: 1975}, -1)``;
    ``"0:15,1:30,e:371"`` gives ``({0: 15, 1: 30}, 371)``. A degree may be
    written ``d3`` as well as ``3``. Degrees the string does not mention
    are absent from the mapping, which is what "unspecified" means -- as
    distinct from ``d:0``, "none of these". ``-1`` means no edge budget.

    Lives here rather than on the generator so the pipeline can read a
    config's request before it builds anything.
    """
    counts: dict[int, int] = {}
    edge_count = -1
    if not spec:
        return counts, edge_count
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("e:"):
            edge_count = int(part.split(":")[1])
        elif ":" in part:
            degree, count = part.split(":")
            counts[int(degree.replace("d", ""))] = int(count)
    return counts, edge_count


def is_fully_specified(target_counts: Mapping[int, int], max_f: int) -> bool:
    """True when every degree from 0 to ``max_f`` carries an explicit count.

    ``target_counts`` is the generator's parsed ``degree_distribution``,
    where a degree absent from the string is missing (or carries the -2
    "unspecified" sentinel) and ``d:0`` is an explicit "none of these".
    A fully specified request pins the whole degree sequence, which is
    what the exact search needs and what the strict sculptor is worst at.
    """
    return all(target_counts.get(d, -2) >= 0 for d in range(0, max_f + 1))


def missing_degrees(target_counts: Mapping[int, int], max_f: int) -> list[int]:
    """The degrees from 0 to ``max_f`` the request left unspecified."""
    return [d for d in range(0, max_f + 1) if target_counts.get(d, -2) < 0]


def resolve_search(
    search: Optional[str],
    target_counts: Mapping[int, int],
    max_f: int,
    target_edge_count: int = -1,
) -> str:
    """Decide between the strict sculptor and the exact search.

    ``search`` is ``topology.generator.search``: ``"strict"`` or
    ``"exact"`` when the config names one, ``None`` when it does not. The
    default is ``"exact"`` exactly when the request pins every degree
    from 0 to ``max_f``, since that is the case the strict sculptor
    cannot reach and the exact search solves in seconds; anything looser
    (an ``e:N`` edge budget, a couple of ``d:N`` targets) stays strict, so
    every config written before this existed behaves as it did.

    Raises
    ------
    ValueError
        If ``search`` is not one of the two names, if ``"exact"`` is asked
        for with a request that does not pin every degree, or if an
        ``e:N`` edge budget contradicts the per-degree counts.
    """
    if search is not None and search not in ("strict", "exact"):
        raise ValueError(
            f"topology.generator.search must be 'strict' or 'exact', "
            f"got {search!r}"
        )
    full = is_fully_specified(target_counts, max_f)
    if search is None:
        search = "exact" if full else "strict"
    if search == "exact" and not full:
        missing = missing_degrees(target_counts, max_f)
        raise ValueError(
            f"search 'exact' needs a count for every degree from 0 to "
            f"max_functionality ({max_f}); degree_distribution leaves "
            f"{', '.join(str(d) for d in missing)} unspecified. Give them "
            f"explicitly (0:N for the vacancies, d:0 for a degree that "
            f"must not occur), or use search 'strict'."
        )
    if search == "exact" and target_edge_count >= 0:
        implied = sum(d * target_counts[d] for d in range(1, max_f + 1))
        if implied != 2 * target_edge_count:
            raise ValueError(
                f"degree_distribution e:{target_edge_count} contradicts its "
                f"own per-degree counts, which need {implied // 2} edges "
                f"(degree sum {implied}). Drop the e: term; the exact "
                f"search derives the edge count from the degree counts."
            )
    return search


def needs_from_targets(target_counts: Mapping[int, int], max_f: int) -> dict[int, int]:
    """The ``need`` mapping :func:`sculpt_exact` takes, from a parsed spec.

    Degree 0 is carried through for the record but never used to place
    sites: the vacancy count is whatever the scaffold has left after the
    active sites are placed, which is how a config can say ``0:43`` on a
    2744-site lattice and mean "the other 43 sites stay empty".
    """
    return {d: int(target_counts[d]) for d in range(0, max_f + 1)}


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def two_colouring(nodes, nb) -> Optional[dict]:
    """The sublattice of every site if the scaffold is bipartite, else None.

    Breadth-first from the first site, in site order. ``None`` when an edge
    joins two sites of the same colour (FCC, MIX, any second shell on SC,
    an odd periodic axis) or when the scaffold falls apart into more than
    one piece, where a single balance would not be the condition anyway.
    """
    if not nodes:
        return None
    colour = {nodes[0]: 0}
    queue = collections.deque([nodes[0]])
    while queue:
        x = queue.popleft()
        for y in nb[x]:
            if y not in colour:
                colour[y] = 1 - colour[x]
                queue.append(y)
            elif colour[y] == colour[x]:
                return None
    return colour if len(colour) == len(nodes) else None


def balance_targets(nodes, t, colour, rng) -> int:
    """Swap targets across the two sublattices until their sums match.

    Draws two sites; when they sit on opposite sublattices, the target on
    the heavier side is swapped with the other one if it is larger by no
    more than half the imbalance, so the imbalance only ever shrinks. Gives
    up after ``BALANCE_TRIES_PER_SITE`` draws per site and returns what is
    left (0 when balanced), which the attempt then fails on.
    """
    n = len(nodes)
    imbalance = sum(t[x] if colour[x] == 0 else -t[x] for x in nodes)
    for _ in range(BALANCE_TRIES_PER_SITE * n):
        if imbalance == 0:
            break
        i = nodes[rng.integers(n)]
        j = nodes[rng.integers(n)]
        if colour[i] == colour[j]:
            continue
        heavy = 0 if imbalance > 0 else 1
        a, b = (i, j) if colour[i] == heavy else (j, i)
        delta = t[a] - t[b]
        if delta <= 0 or 2 * delta > abs(imbalance):
            continue
        t[a], t[b] = t[b], t[a]
        imbalance += -2 * delta if heavy == 0 else 2 * delta
    return imbalance


def sculpt_exact(
    base,
    need: Mapping[int, int],
    rng,
    max_f: int = 4,
    double_pairs: Optional[Mapping[tuple[int, int], int]] = None,
    *,
    min_giant_fraction: float = DEFAULT_MIN_GIANT_FRACTION,
    max_rounds: int = MAX_AUGMENT_ROUNDS,
    max_repair_steps: int = MAX_REPAIR_STEPS,
    fallback: bool = False,
):
    """One attempt at a subgraph of ``base`` with exactly the degrees in ``need``.

    Args:
        base: The candidate-edge graph a lattice builder produced. Read
            only; its nodes and edges are the sites and the pairs the
            search may bond.
        need: ``{degree: count}``. Counts for degree 1 and above place
            sites; the degree-0 entry is ignored, the vacancies being
            whatever is left over.
        rng: A ``numpy.random.Generator``. The whole search is driven
            from it, so the same generator state gives the same graph.
        max_f: The highest degree the request may use.
        double_pairs: ``{(a, b): count}`` forced parallel edges between a
            site of target degree ``a`` and one of target degree ``b``,
            placed before the fill and never moved afterwards. Each one
            contributes 2 to both endpoints, so a pair of degree-4
            junctions joined by a double has 3 other partners each.
        min_giant_fraction: Reject a sculpt whose largest component holds
            less than this fraction of the active sites.
        max_rounds: Augmentation rounds before the repair loop takes over.
        max_repair_steps: Target swaps and vacancy relocations before the
            attempt is abandoned.
        fallback: Balance the deal across the sublattices of a bipartite
            scaffold and run the residual-driven repair instead of step 4
            (see the module docstring). :func:`build_exact_graph` turns it
            on after :data:`FALLBACK_AFTER` short attempts in a row.

    Returns:
        ``(edges, degrees, record)``. ``edges`` is a set of ``(u, v)``
        pairs with ``u < v`` (a forced double appears once here and again
        in ``record["doubles"]``), ``degrees`` maps every active site to
        its achieved degree, and ``record`` carries the counts requested
        and achieved, the vacancies, augmentations, repairs, residual,
        giant fraction and seconds. On failure the first two are ``None``
        and ``record["error"]`` says why.
    """
    t0 = time.time()
    nodes = list(base.nodes())
    n_sites = len(nodes)
    # Degrees above the ceiling would be dropped silently by the slice
    # below, leaving a record that claims a target the caller never asked
    # for. The generator refuses them earlier with a fuller message; this
    # is the guard for a direct caller.
    above = sorted(d for d, c in need.items() if d > max_f and c > 0)
    if above:
        raise ValueError(
            f"need asks for {', '.join(f'{d}:{need[d]}' for d in above)} but "
            f"max_f is {max_f}; raise max_f or drop those degrees"
        )
    requested = {d: int(need.get(d, 0)) for d in range(0, max_f + 1)}
    n_active = sum(c for d, c in requested.items() if d > 0)
    n_vacancies = n_sites - n_active
    if 0 not in need:
        # Nothing was asked of degree 0, so the leftover sites are the
        # answer rather than a shortfall. When a count *was* given it is
        # kept as written: the record then shows it against the vacancies
        # the scaffold actually had, which is how a cell of the wrong size
        # announces itself (a 90/5/5 MIX 14^3 draw has ~650 spare sites
        # where SC 14^3 has 43).
        requested[0] = max(n_vacancies, 0)

    def _record(**extra):
        rec = dict(
            n_sites=n_sites, n_vacancies=n_vacancies,
            base_edges=base.number_of_edges(),
            requested=dict(requested),
            min_giant_fraction=float(min_giant_fraction),
        )
        rec.update(extra)
        rec["seconds"] = time.time() - t0
        return rec

    if n_vacancies < 0:
        return None, None, _record(
            error=f"the scaffold has {n_sites} sites but the request places "
                  f"{n_active} active ones",
            residual=None, augmentations=0, target_swaps=0,
        )

    # Every edge contributes 2 to the degree sum, so an odd sum belongs to
    # no graph at all (the handshake lemma). Caught here because the search
    # would otherwise spend its whole repair budget one unit short and
    # report it as an ordinary shortfall.
    degree_sum = sum(d * c for d, c in requested.items() if d > 0)
    if degree_sum % 2:
        return None, None, _record(
            error=f"the requested counts have an odd degree sum "
                  f"({degree_sum}); every edge adds 2, so no graph has one. "
                  f"Move one site between two odd degrees, or add one site "
                  f"of odd degree",
            residual=None, augmentations=0, target_swaps=0,
        )

    # --- target degrees -------------------------------------------------
    pool = []
    for d in range(1, max_f + 1):
        pool += [d] * requested[d]
    pool += [0] * n_vacancies
    perm = rng.permutation(n_sites)
    t = {nodes[perm[i]]: pool[i] for i in range(n_sites)}
    base_nb = {n: list(base[n]) for n in nodes}
    colour = None
    imbalance = None
    if fallback:
        colour = two_colouring(nodes, base_nb)
        if colour is not None:
            imbalance = balance_targets(nodes, t, colour, rng)
    active = [n for n in nodes if t[n] > 0]

    def eligible(n):
        """Candidate partners of ``n`` under the current targets.

        Evaluated at the point of use rather than cached, because the
        repair loop moves targets around: a site that was a vacancy can
        become a junction and vice versa. The one standing rule is the
        dangling-end rule -- two degree-1 sites joined to each other
        would be a free chain, not part of the network.
        """
        return [m for m in base_nb[n]
                if t[m] > 0 and not (t[n] == 1 and t[m] == 1)]

    adj = {n: set() for n in nodes}
    deg = {n: 0 for n in nodes}
    fixed = set()          # pairs carrying a forced double edge
    # Edge count, and a log of edge changes the fallback repair can undo.
    # Neither draws a random number, so the random-deal path is unchanged.
    n_edges = [0]
    log = []
    logging = [False]

    def add(u, v):
        adj[u].add(v); adj[v].add(u); deg[u] += 1; deg[v] += 1
        n_edges[0] += 1
        if logging[0]:
            log.append((1, u, v))

    def rem(u, v):
        adj[u].discard(v); adj[v].discard(u); deg[u] -= 1; deg[v] -= 1
        n_edges[0] -= 1
        if logging[0]:
            log.append((0, u, v))

    # --- forced double edges --------------------------------------------
    # A parallel strand between the same two junctions, with the endpoint
    # degrees the reference network has. Placed first, because they are
    # the least free choice in the whole search, and held fixed after:
    # the augmenting walk below never removes one.
    n_double = 0
    if double_pairs:
        used = set()
        for (a, b), count in sorted(double_pairs.items(), key=lambda kv: -kv[1]):
            cand = [(u, v) for u in active if t[u] in (a, b) for v in base_nb[u]
                    if u < v and {t[u], t[v]} == {a, b}
                    and (a != b or t[u] == t[v] == a)
                    and u not in used and v not in used]
            rng.shuffle(cand)
            placed = 0
            for u, v in cand:
                if placed >= count:
                    break
                if u in used or v in used:
                    continue
                # One edge in `adj`, two units of degree: a simple graph
                # cannot hold the parallel edge, so the second unit is
                # added straight to the degree counters and the pair is
                # listed in the record for the caller to realise.
                add(u, v)
                deg[u] += 1; deg[v] += 1
                fixed.add((u, v) if u < v else (v, u))
                used.add(u); used.add(v)
                n_double += 1; placed += 1

    # --- greedy random fill ---------------------------------------------
    order = list(active)
    rng.shuffle(order)
    for u in order:
        cand = [v for v in eligible(u) if v not in adj[u] and deg[v] < t[v]]
        rng.shuffle(cand)
        for v in cand:
            if deg[u] >= t[u]:
                break
            add(u, v)

    # --- augmenting paths ------------------------------------------------
    def augment(s):
        """Alternating BFS from a deficient ``s``.

        Moves alternate: add an absent candidate edge, remove a present
        one, add, ... and the walk ends the moment it reaches another
        deficient site by an "add". Flipping such a path raises both end
        degrees by one and leaves every site between them untouched.
        """
        parent = {s: None}
        frontier = collections.deque([(s, "add")])
        while frontier:
            x, move = frontier.popleft()
            if move == "add":
                cand = list(eligible(x))
                rng.shuffle(cand)
                for y in cand:
                    if y in adj[x] or y in parent:
                        continue
                    parent[y] = (x, "add")
                    if deg[y] < t[y] and y != s:
                        return y, parent
                    frontier.append((y, "rm"))
            else:
                cand = list(adj[x])
                rng.shuffle(cand)
                for y in cand:
                    if y in parent or ((x, y) if x < y else (y, x)) in fixed:
                        continue
                    parent[y] = (x, "rm")
                    frontier.append((y, "add"))
        return None, parent

    n_aug = 0

    def drain(s):
        """Augment from ``s`` until it reaches its target or gets stuck.

        Returns the sites the failed search reached when ``s`` is stuck
        (the region short of partners, which the fallback repair reads),
        else None.
        """
        nonlocal n_aug
        while deg[s] < t[s]:
            end, parent = augment(s)
            if end is None:
                return parent
            y = end
            while parent[y] is not None:
                x, kind = parent[y]
                (add if kind == "add" else rem)(x, y)
                y = x
            n_aug += 1
        return None

    for _ in range(max_rounds):
        deficient = [n for n in active if deg[n] < t[n]]
        if not deficient:
            break
        rng.shuffle(deficient)
        for s in deficient:
            drain(s)

    # --- repair a locally infeasible draw --------------------------------
    # The random assignment can put a high target where the scaffold
    # cannot serve it (next to vacancies, or in a corner the dangling-end
    # rule has emptied). Two repairs keep the requested counts exact while
    # giving the augmentation a fresh start: swap the target with a
    # saturated site of lower target, or move the vacancy to the deficient
    # site and hand its target to a vacancy with enough active neighbours.
    # The fallback runs its own repair instead (residual_repair below).
    def residual_repair():
        """The fallback repair: move demand out of the region short of partners.

        A deficient site whose augmenting search fails has reached every
        site it can; that region is short of partners. A target from the
        region (on the site's own sublattice, 2 or more) is swapped with a
        lower one outside it on the same sublattice, which keeps the two
        sublattices balanced, edges above a new target are dropped at
        random, and the touched sites are drained. The swap is kept if the
        subgraph lost no edge and undone otherwise.
        """
        kept = 0
        on_fixed = {x for pair in fixed for x in pair}

        def same(a, b):
            return colour is None or colour[a] == colour[b]

        for _ in range(max_repair_steps):
            deficient = [n for n in nodes if t[n] > 0 and deg[n] < t[n]]
            if not deficient:
                break
            u = deficient[rng.integers(len(deficient))]
            if t[u] == 1:
                cands = [w for w in eligible(u) if deg[w] < t[w]]
                if cands:
                    add(u, cands[0])
                    continue
            if u in on_fixed:
                others = [n for n in deficient if n not in on_fixed]
                if not others:
                    break
                u = others[rng.integers(len(others))]
            region = drain(u)
            if region is None:
                continue
            inside = [x for x in nodes if x in region and t[x] >= 2
                      and same(x, u) and x not in on_fixed]
            src = inside[rng.integers(len(inside))] if inside else u
            pool_w = [w for w in nodes
                      if w not in region and t[w] < t[src] and same(w, src)
                      and w not in on_fixed
                      and (t[w] >= 2 or (t[w] == 0 and sum(
                          1 for m in base_nb[w] if t[m] > 0 and m != src) >= t[src]))]
            if not pool_w:
                continue
            w = pool_w[rng.integers(len(pool_w))]
            before = n_edges[0]
            log.clear()
            logging[0] = True
            t[src], t[w] = t[w], t[src]
            touched = [w, src]
            for x in (src, w):
                extra = deg[x] - t[x]
                if extra > 0:
                    partners = list(adj[x])
                    rng.shuffle(partners)
                    for m in partners[:extra]:
                        rem(x, m)
                        touched.append(m)
            for s in touched:
                drain(s)
            logging[0] = False
            if n_edges[0] < before:
                for kind, a, b in reversed(log):
                    (rem if kind == 1 else add)(a, b)
                t[src], t[w] = t[w], t[src]
            else:
                kept += 1
        return kept

    n_swap = residual_repair() if fallback else 0
    for _ in range(0 if fallback else max_repair_steps):
        deficient = [n for n in active if deg[n] < t[n]]
        if not deficient:
            break
        u = deficient[rng.integers(len(deficient))]
        if t[u] == 1:
            # A dangling end with no partner: take any neighbour with room.
            cands = [w for w in eligible(u) if deg[w] < t[w]]
            if cands:
                add(u, cands[0])
                continue
        on_fixed = {x for pair in fixed for x in pair}
        if u in on_fixed:
            others = [n for n in deficient if n not in on_fixed]
            if not others:
                break
            u = others[rng.integers(len(others))]
        pool_w = [w for w in active
                  if 2 <= t[w] < t[u] and deg[w] == t[w] and deg[u] <= t[w]
                  and w not in on_fixed]
        if not pool_w:
            pool_w = [w for w in active
                      if 2 <= t[w] < t[u] and deg[u] <= t[w] and w not in on_fixed]
        if not pool_w or rng.random() < 0.3:
            vacs = [] if any(((u, m) if u < m else (m, u)) in fixed for m in adj[u]) \
                else [w for w in nodes if t[w] == 0
                      and sum(1 for m in base_nb[w] if t[m] > 0 and m != u) >= t[u]]
            if vacs:
                w = vacs[rng.integers(len(vacs))]
                partners = list(adj[u])
                for m in partners:
                    rem(u, m)
                t[w], t[u] = t[u], 0
                active = [n for n in nodes if t[n] > 0]
                n_swap += 1
                for s in [w] + partners:
                    drain(s)
                continue
            if not pool_w:
                break
        w = pool_w[rng.integers(len(pool_w))]
        t[u], t[w] = t[w], t[u]
        n_swap += 1
        for s in (w, u):
            drain(s)

    # --- accept or reject -------------------------------------------------
    active = [n for n in nodes if t[n] > 0]
    residual = sum(t[n] - deg[n] for n in active)
    edges = {(u, v) if u < v else (v, u) for u in active for v in adj[u]}
    degrees = {n: deg[n] for n in active}
    achieved = collections.Counter(degrees.values())
    rec = _record(
        augmentations=n_aug, target_swaps=n_swap,
        doubles=sorted(fixed), n_double=n_double, residual=residual,
        counts={d: int(achieved.get(d, 0)) for d in range(0, max_f + 1)},
    )
    if fallback:
        rec["fallback"] = True
        rec["bipartite"] = colour is not None
        rec["imbalance"] = imbalance
    rec["counts"][0] = n_sites - len(active)
    rec["achieved"] = dict(rec["counts"])
    if residual != 0:
        rec["error"] = (
            f"{residual} degree unit(s) unfilled on "
            f"{sum(1 for n in active if deg[n] < t[n])} site(s)"
        )
        return None, None, rec

    H = nx.Graph()
    H.add_nodes_from(active)
    H.add_edges_from(edges)
    components = sorted(nx.connected_components(H), key=len, reverse=True)
    rec["giant_frac"] = len(components[0]) / len(active) if active else 1.0
    rec["n_components"] = len(components)
    if rec["giant_frac"] < min_giant_fraction:
        rec["error"] = (
            f"the giant component holds {rec['giant_frac']:.3f} of the active "
            f"sites, below min_giant_fraction {min_giant_fraction:g}"
        )
        return None, None, rec
    return edges, degrees, rec


# ---------------------------------------------------------------------------
# Building the graph the pipeline consumes
# ---------------------------------------------------------------------------

#: Share of the active sites that has to be pinned at the scaffold's own
#: coordination before a failed attempt counts as hopeless rather than
#: unlucky. A site asked for a degree the scaffold can only just serve has
#: no choice of partner at all, so when most sites are in that position
#: the assignment is effectively forced and a fresh permutation changes
#: nothing. Below it a re-seed is worth trying: the manuscript target puts
#: 9 of 206 sites at SC's own coordination of 6 and lands on seed 1.
_PINNED_SHARE = 0.5


def _no_slack(base, need: Mapping[int, int], max_f: int) -> bool:
    """True when most of the request is pinned to the scaffold's ceiling.

    A Diamond lattice at the default cutoff has exactly four candidates
    per site, so a ``max_functionality = 4`` request leaves no choice
    anywhere: 73 % of the sites in the reference P(f) are at the ceiling
    and each has to use every neighbour it has. Vacancies and dangling-end
    sites then remove capacity nothing can give back, and retrying draws
    the same forced assignment again.

    The test is the scaffold's own maximum coordination, not
    ``max_functionality``: SC at ``max_functionality = 6`` is equally
    pinned for the sites asked to reach 6, but a target that puts only a
    handful of sites there has room everywhere else and deserves its
    retries.
    """
    degrees = [d for _, d in base.degree()]
    if not degrees:
        return False
    ceiling = min(max(degrees), max_f)
    active = sum(c for d, c in need.items() if d > 0 and c > 0)
    pinned = sum(c for d, c in need.items() if 0 < d <= max_f and d >= ceiling)
    return bool(active) and pinned >= _PINNED_SHARE * active


def _failure_message(base, need, max_f, records, label, attempts) -> str:
    """Why the search stopped, in terms of the scaffold and the request."""
    best = min(
        (r for r in records if r.get("residual") is not None),
        key=lambda r: (r["residual"], -r.get("giant_frac", 0.0)),
        default=records[-1] if records else {},
    )
    degrees = [d for _, d in base.degree()]
    z_mean = (sum(degrees) / len(degrees)) if degrees else 0.0
    n_active = sum(c for d, c in need.items() if d > 0 and d <= max_f)
    wanted_sum = sum(d * c for d, c in need.items() if d > 0 and d <= max_f)
    lines = [
        f"the exact degree search did not reach the requested counts on "
        f"{label} after {attempts} attempt(s).",
        f"  requested : {n_active} active sites, degree sum {wanted_sum}, "
        f"on {base.number_of_nodes()} sites with "
        f"{base.number_of_edges()} candidate edges "
        f"(mean coordination {z_mean:.1f}, max {max(degrees) if degrees else 0})",
        f"  best try  : {best.get('error', 'no attempt completed')}"
        + (f", giant component {best['giant_frac']:.3f}"
           if best.get("giant_frac") is not None else ""),
    ]
    if _no_slack(base, need, max_f):
        ceiling = min(max(degrees), max_f)
        pinned = sum(c for d, c in need.items() if 0 < d <= max_f and d >= ceiling)
        lines.append(
            f"  reason    : the scaffold offers at most {max(degrees)} "
            f"candidate partners per site, and {pinned} of the {n_active} "
            f"active sites are asked for degree {ceiling}, so most of them "
            f"must bond to every neighbour they have. There is almost no "
            f"spare edge, and each vacancy"
            + (" and each dangling-end site" if need.get(1, 0) else "")
            + " takes capacity from its neighbours that nothing can give back."
        )
        lines.append(
            "  fix       : give the scaffold more candidates per site than "
            "the request needs -- a wider neighbour_cutoff, or a lattice "
            "with a higher coordination -- or drop the degree-1 sites. Note "
            "that the default cutoff of 1.0 is the canonical-lattice "
            "sentinel rather than a range, so on Diamond the wider setting "
            "is the smaller number: 0.71 cell units admits the second shell "
            "(z = 16) where 1.0 gives the canonical 4."
        )
    else:
        lines.append(
            "  fix       : give the search more room -- a larger lattice, a "
            "larger neighbour_cutoff, or fewer sites at the ceiling -- or "
            "lower min_giant_fraction if the giant component is what failed."
        )
    return "\n".join(lines)


def build_exact_graph(
    base_graph,
    need: Mapping[int, int],
    max_f: int,
    seed,
    *,
    double_pairs: Optional[Mapping[tuple[int, int], int]] = None,
    min_giant_fraction: float = DEFAULT_MIN_GIANT_FRACTION,
    attempts: int = DEFAULT_ATTEMPTS,
    fallback_attempts: int = DEFAULT_FALLBACK_ATTEMPTS,
    label: str = "the lattice",
    verbose: bool = True,
):
    """Sculpt ``base_graph`` to exactly ``need`` and return it as a graph.

    Retries with a fresh seed until one attempt is accepted, then returns
    a graph in the same form the strict sculptor returns: every site of
    the scaffold is a node, carrying whatever attributes the lattice
    builder gave it, the vacancies among them at degree 0, and the graph
    keeps the builder's ``box``, ``periodicity`` and neighbour-shell
    attributes. The sculpt record lands on ``G.graph["sculpt_record"]``
    and the search that produced it on ``G.graph["sculpt_search"]``.

    The edges are inserted in the scaffold's own edge order, so the graph
    reads like a strict-sculpted one: the candidate edges that survived,
    in the order the lattice built them.

    A graph with forced double edges is a ``MultiGraph`` -- a simple
    graph cannot hold a parallel strand -- and the second edge of each
    pair carries ``is_secondary_loop=True``, which is what a parallel
    strand is in polymer usage (0.2.0 moved the name ``is_primary_loop`` to
    the self-loops it belongs to). Without ``double_pairs`` the result is
    a plain ``Graph``, exactly what the strict sculptor hands back.

    Up to ``attempts`` attempts use the random deal. When
    ``min(FALLBACK_AFTER, attempts)`` of them in a row end with unfilled
    degree units, the rest (and ``fallback_attempts`` more) use the
    fallback deal and repair (see the module docstring). Attempts that fill
    every degree and fail on connectivity do not count toward the switch.
    The attempts before the switch draw from the same seeds as they always
    did, so a request the random deal reaches is unaffected.

    Raises:
        ExactSculptError: if no attempt reached the request. The message
            names the scaffold, the request, the best attempt's residual
            and, when the scaffold has no spare candidate edge, that
            reason and what to change.
    """
    attempts = max(1, attempts)
    fallback_attempts = max(0, fallback_attempts)
    root = np.random.SeedSequence(seed)
    # spawn() is indexed, so the first `attempts` children are the ones a
    # plain spawn(attempts) gives: the random-deal attempts are unchanged.
    children = root.spawn(attempts + fallback_attempts)
    records = []
    short = 0            # random-deal attempts in a row that ended short
    fallback = False
    for attempt in range(attempts + fallback_attempts):
        if attempt >= attempts and not fallback:
            break
        rng = np.random.default_rng(children[attempt])
        edges, degrees, rec = sculpt_exact(
            base_graph, need, rng, max_f=max_f, double_pairs=double_pairs,
            min_giant_fraction=min_giant_fraction, fallback=fallback,
        )
        rec["attempt"] = attempt
        rec["seed"] = list(root.entropy) if isinstance(root.entropy, (list, tuple)) \
            else root.entropy
        records.append(rec)
        if verbose:
            print(f"  [exact] attempt {attempt}{' (fallback)' if fallback else ''}: "
                  f"{'reached' if edges is not None else 'failed'} in "
                  f"{rec['seconds']:.2f}s "
                  f"({rec['augmentations']} augmentations, "
                  f"{rec['target_swaps']} repairs)"
                  + ("" if edges is not None else f" -- {rec['error']}"))
        if edges is not None:
            rec["attempts"] = attempt + 1
            spare = rec["n_vacancies"] - rec["requested"].get(0, 0)
            if spare and verbose:
                print(f"  [exact] note: {label} left {rec['n_vacancies']} sites "
                      f"empty where degree_distribution asked for "
                      f"{rec['requested'][0]}. Degree 0 is whatever the "
                      f"scaffold has over, so the active counts are still "
                      f"exact; resize the cell if the site density was "
                      f"meant to match.")
            return _assemble(base_graph, edges, degrees, rec, double_pairs)
        if rec.get("residual") is None:
            break      # the request does not fit the scaffold at all
        if rec["residual"] and _no_slack(base_graph, need, max_f):
            break      # retrying cannot find room that does not exist
        if not fallback:
            short = short + 1 if rec["residual"] else 0
            if short >= min(FALLBACK_AFTER, attempts) and fallback_attempts:
                fallback = True
                if verbose:
                    print(f"  [exact] {short} attempts in a row ended short; "
                          f"switching to the balanced deal and the "
                          f"residual-driven repair")
    raise ExactSculptError(
        _failure_message(base_graph, need, max_f, records, label, len(records))
    )


def _assemble(base_graph, edges, degrees, rec, double_pairs):
    """The scaffold with only the sculpted edges, plus the doubles."""
    doubles = [tuple(pair) for pair in rec.get("doubles", [])]
    G = nx.MultiGraph() if doubles else nx.Graph()
    G.add_nodes_from((n, dict(d)) for n, d in base_graph.nodes(data=True))
    G.graph.update(base_graph.graph)
    # The scaffold's own edge order, so a sculpted graph is read in the
    # order the lattice builder laid its candidates down, as it is after
    # strict sculpting.
    G.add_edges_from(
        (u, v) for u, v in base_graph.edges()
        if ((u, v) if u < v else (v, u)) in edges
    )
    for u, v in doubles:
        G.add_edge(u, v, cls="bridge", is_secondary_loop=True)
    G.graph["sculpt_search"] = "exact"
    G.graph["sculpt_record"] = rec
    return G


def counts_by_degree(G, max_f: Optional[int] = None) -> dict[int, int]:
    """Degree histogram of ``G``, vacancies included at degree 0.

    The shape the ``degree_distribution`` string speaks in, so a caller
    can compare what was asked for with what came back.
    """
    hist = collections.Counter(d for _, d in G.degree())
    top = max(hist, default=0)
    if max_f is not None:
        top = max(top, int(max_f))
    return {d: int(hist.get(d, 0)) for d in range(0, top + 1)}
