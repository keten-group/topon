
import random
import time
import math
import warnings
from collections import defaultdict, deque
from itertools import product

import networkx as nx
import numpy as np

from topon.topology.degree_matching import (
    DEFAULT_ATTEMPTS,
    DEFAULT_MIN_GIANT_FRACTION,
    build_exact_graph,
    is_fully_specified,
    needs_from_targets,
    parse_degree_distribution,
    resolve_search,
)
from topon.topology.shells import (
    DEFAULT_CUTOFF,
    SEARCH_TOLERANCE_SQ,
    resolve_neighbour_cutoff,
)

#: Above this many sites the neighbour search bins the sites into cells at
#: least one cutoff wide instead of comparing every pair. Measured on SC
#: at cutoffs 1.74 and 3.01: the all-pairs search is faster up to a few
#: thousand sites and its memory grows quadratically (about 400 MB at
#: ten thousand sites), while the cell list stays within tens of MB.
CELL_LIST_MIN_SITES = 4000


def edges_within_cutoff(points, box, wrap, cutoff, method="auto"):
    """Every pair of sites within ``cutoff`` under the minimum image.

    Args:
        points: ``(N, 3)`` site coordinates in cell units, in node order.
        box: the periodic cell ``(Lx, Ly, Lz)``.
        wrap: per-axis flags; only a periodic axis takes the minimum
            image, an open one keeps the raw separation so nothing bonds
            across the free face.
        cutoff: the range, inclusive (``d <= cutoff`` with a 1e-12
            tolerance on the squared distance, matching the C searcher).
        method: ``"pairs"`` compares every pair in blocks of 256 rows,
            ``"cells"`` bins the sites into cells at least one cutoff
            wide and compares neighbouring cells only, ``"auto"`` picks
            cells above :data:`CELL_LIST_MIN_SITES`. Both give the same
            edges; the cell list exists for large lattices, where the
            all-pairs search is quadratic in time and memory.

    Returns:
        A sorted list of ``(u, v)`` index pairs with ``u < v``. That is
        the order ``MIX`` has always inserted its edges in, so adjacency
        order, and with it the sculptor's random stream, is unchanged
        for existing seeds.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    box = np.asarray(box, dtype=float)
    wrap = np.asarray(wrap, dtype=bool)
    cutoff = float(cutoff)
    if method == "auto":
        method = "cells" if len(pts) > CELL_LIST_MIN_SITES else "pairs"
    if method == "cells":
        found = _pairs_by_cells(pts, box, wrap, cutoff)
    elif method == "pairs":
        found = _pairs_all(pts, box, wrap, cutoff)
    else:
        raise ValueError(f"unknown neighbour-search method {method!r}")
    found = [f for f in found if len(f)]
    if not found:
        return []
    # Both searches report each unordered pair once, as (u, v) with u < v;
    # np.unique sorts lexicographically, which is the required order.
    pairs = np.unique(np.concatenate(found), axis=0)
    return [tuple(p) for p in pairs.tolist()]


def _within(a, b, box, wrap, cutoff_sq):
    """Index pairs ``(i, j)`` with ``a[i]`` within the cutoff of ``b[j]``."""
    delta = a[:, None, :] - b[None, :, :]
    delta -= np.where(wrap, box * np.round(delta / box), 0.0)
    dist_sq = (delta * delta).sum(axis=-1)
    return np.nonzero((dist_sq <= cutoff_sq + SEARCH_TOLERANCE_SQ)
                      & (dist_sq > SEARCH_TOLERANCE_SQ))


def _ordered(rows, cols):
    """Keep the ``u < v`` half of the index pairs as an ``(M, 2)`` array."""
    keep = rows < cols
    return np.stack([rows[keep], cols[keep]], axis=1)


def _pairs_all(pts, box, wrap, cutoff):
    """Blocked all-pairs search: 256 rows at a time against every site.

    Blocked rather than one ``(N, N, 3)`` array so a few thousand sites
    do not blow up memory; each block's temporaries are 256 x N x 3
    doubles. Returns a list of ``(M, 2)`` index arrays.
    """
    cutoff_sq = cutoff * cutoff
    block = 256
    found = []
    for start in range(0, len(pts), block):
        rows, cols = _within(pts[start:start + block], pts, box, wrap, cutoff_sq)
        found.append(_ordered(rows + start, cols))
    return found


def _pairs_by_cells(pts, box, wrap, cutoff):
    """Cell-list search: bin the sites, compare each cell with its 27 neighbours.

    Cells are at least one cutoff wide (a hair more, so a pair exactly
    one cutoff apart can never straddle two cell boundaries), which
    means every pair within range sits in the same or an adjacent cell.
    A periodic axis with fewer than three cells wraps a cell's neighbour
    set onto itself; the neighbour cells are deduplicated, so the search
    stays correct there and merely degrades toward all-pairs. Returns a
    list of ``(M, 2)`` index arrays, each pair once as ``u < v``.
    """
    cutoff_sq = cutoff * cutoff
    n_cells = np.maximum(1, np.floor(box / cutoff * (1.0 - 1e-9)).astype(int))
    width = box / n_cells
    idx = np.clip(np.floor(pts / width).astype(int), 0, n_cells - 1)
    flat = idx[:, 0] + n_cells[0] * (idx[:, 1] + n_cells[1] * idx[:, 2])
    order = np.argsort(flat, kind="stable")
    cells, starts = np.unique(flat[order], return_index=True)
    ends = np.append(starts[1:], len(pts))
    members = {int(c): order[s:e] for c, s, e in zip(cells, starts, ends)}

    strides = (1, int(n_cells[0]), int(n_cells[0] * n_cells[1]))
    found = []
    for cell, mine in members.items():
        here = (cell % n_cells[0],
                (cell // n_cells[0]) % n_cells[1],
                cell // (n_cells[0] * n_cells[1]))
        neighbours = set()
        for step in product((-1, 0, 1), repeat=3):
            key = 0
            for axis in range(3):
                c = here[axis] + step[axis]
                if wrap[axis]:
                    c %= n_cells[axis]
                elif not 0 <= c < n_cells[axis]:
                    break
                key += int(c) * strides[axis]
            else:
                if key in members:
                    neighbours.add(key)
        candidates = np.concatenate([members[k] for k in sorted(neighbours)])
        rows, cols = _within(pts[mine], pts[candidates], box, wrap, cutoff_sq)
        found.append(_ordered(mine[rows], candidates[cols]))
    return found


def _odd_degree_sum_message(degree_sum):
    """The handshake-lemma refusal, one wording for both searches."""
    return (
        f"degree_distribution has an odd degree sum ({degree_sum}); every "
        f"edge contributes 2, so no graph has one. Move one site between two "
        f"odd degrees (e.g. one fewer at degree 1 and one more at degree 2), "
        f"or add one site of odd degree."
    )


def shell_distances(G, decimals=6):
    """The distinct edge lengths of ``G`` under the minimum image, sorted.

    Read off the graph rather than assumed from the lattice type: a MIX
    draw may miss a shell, an open axis drops the wrapped bonds, and a
    cutoff picks a different set on every lattice. Lengths are in the
    graph's own units (cell units for a freshly built lattice), rounded
    to ``decimals`` so the same shell measured across the periodic
    boundary collapses to one value. Wrapping needs the graph to carry
    ``box``; without it the raw separations are used.
    """
    if G.number_of_edges() == 0:
        return ()
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    pos = np.asarray([G.nodes[n]["pos"] for n in nodes], dtype=float)
    ends = np.asarray([(index[u], index[v]) for u, v in G.edges()])
    delta = pos[ends[:, 0]] - pos[ends[:, 1]]
    box = G.graph.get("box")
    if box is not None:
        box = np.asarray(box, dtype=float)
        wrap = np.asarray(G.graph.get("periodicity", (True, True, True)),
                          dtype=bool)
        delta -= np.where(wrap, box * np.round(delta / box), 0.0)
    lengths = np.sqrt((delta * delta).sum(axis=1))
    return tuple(float(x) for x in np.unique(np.round(lengths, decimals)))


#: Sites per cubic cell of the pure lattices.
SITES_PER_CELL = {"SC": 1, "BCC": 2, "FCC": 4, "Diamond": 8, "DIAMOND": 8}

#: Face-centre offsets, in the XY / XZ / YZ order the pure FCC builder
#: uses, so a MIX with an FCC fraction of 1.0 gives the same site set.
_FACE_OFFSETS = ((0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))


def mix_site_positions(nx_val, ny_val, nz_val, f_bcc, f_fcc, rng):
    """The sites of a MIX lattice, drawn from ``rng``.

    Every cell carries its corner; the body centre is drawn with
    probability ``f_bcc`` and each face centre with probability ``f_fcc``,
    one ``rng.random()`` per candidate, cells in the SC builder's order (z
    outer, then y, then x) so fractions (1, 0, 0) give the SC node ids.
    Shared by the builder and :func:`count_sites`, which is what makes the
    count a promise about the lattice rather than an estimate of it.
    """
    positions = []
    for k in range(nz_val):
        for j in range(ny_val):
            for i in range(nx_val):
                positions.append((float(i), float(j), float(k)))
                if f_bcc > 0.0 and rng.random() < f_bcc:
                    positions.append((i + 0.5, j + 0.5, k + 0.5))
                if f_fcc > 0.0:
                    for fx, fy, fz in _FACE_OFFSETS:
                        if rng.random() < f_fcc:
                            positions.append((i + fx, j + fy, k + fz))
    return positions


def _cells(config):
    """``(nx, ny, nz)`` from a config's ``lattice_size``, strictly."""
    size = getattr(config, "lattice_size", None)
    if size is None:
        size = getattr(config, "dimension", None)
    if isinstance(size, str):
        parts = size.lower().split("x")
        if len(parts) == 3 and all(p.strip().isdigit() for p in parts):
            return tuple(int(p) for p in parts)
    elif size is not None and len(size) == 3:
        return tuple(int(p) for p in size)
    raise ValueError(f"lattice_size {size!r} is not of the form 'NxNxN'")


def count_sites(config, seed=None):
    """How many sites the Python generator's lattice for ``config`` has.

    SC, BCC, FCC and Diamond have a fixed count per cell (1, 2, 4, 8). A
    MIX draws its body and face sites, so its count depends on the seed:
    ``seed`` is the number the generator's streams are seeded with, the
    same meaning as ``topology.generator.seed`` (or ``random.seed(seed)``
    just before generating), and defaults to the config's own seed. The
    count comes from the same draw the builder makes, so it is exact, not
    an expectation. Compute absolute degree counts from it (see
    :func:`topon.topology.degree_matching.rescale_degree_counts`).

    The C binary draws a MIX from its own stream, so this counts the
    Python generator's lattice only.

    Raises:
        ValueError: for a MIX with no seed given and none in the config,
            where the count is not determined, or a malformed size.
    """
    nx_val, ny_val, nz_val = _cells(config)
    lattice = getattr(config, "lattice_type", "SC")
    if lattice in SITES_PER_CELL:
        return SITES_PER_CELL[lattice] * nx_val * ny_val * nz_val
    if lattice != "MIX":
        raise ValueError(f"lattice_type {lattice!r} has no site rule")
    if seed is None:
        seed = getattr(config, "seed", None)
    if seed is None:
        raise ValueError(
            "a MIX lattice's site count is a random draw; give a seed "
            "(count_sites(config, seed)) or set topology.generator.seed")
    fractions = getattr(config, "mix_fractions", None) or {}
    return len(mix_site_positions(
        nx_val, ny_val, nz_val, float(fractions.get("BCC", 0.0)),
        float(fractions.get("FCC", 0.0)), random.Random(int(seed))))


def expected_mix_sites(config):
    """Mean and standard deviation of a MIX lattice's site count.

    ``N (1 + f_bcc + 3 f_fcc)`` sites on average for ``N`` cells, each
    body and face site an independent draw.
    """
    nx_val, ny_val, nz_val = _cells(config)
    n = nx_val * ny_val * nz_val
    fractions = getattr(config, "mix_fractions", None) or {}
    f_bcc = float(fractions.get("BCC", 0.0))
    f_fcc = float(fractions.get("FCC", 0.0))
    mean = n * (1.0 + f_bcc + 3.0 * f_fcc)
    sd = math.sqrt(n * (f_bcc * (1.0 - f_bcc) + 3.0 * f_fcc * (1.0 - f_fcc)))
    return mean, sd


def check_site_count(target_counts, max_func, search, n_sites, label="",
                     mix=False):
    """Refuse a request whose sites cannot fit the lattice, in plain numbers.

    ``target_counts`` is the parsed ``degree_distribution`` (missing, or
    below 0, means unspecified). What has to hold depends on the search:

    - exact: the active sites (degree 1 and up) must fit; degree 0 is the
      leftover, so any vacancy count is accepted;
    - strict, every degree from 0 to ``max_func`` named: the counts must
      add up to the lattice exactly, since every site ends with one of
      those degrees;
    - strict, some degrees named: they must not add up to more than the
      lattice.

    Without this a strict request that does not fit spends every trial
    and ends in "no graph produced". ``mix`` adds the reason a count that
    fitted one run can miss the next.

    Raises:
        ValueError: "degree_distribution asks for N sites, the lattice
            has M", with what to do about it.
    """
    counts = {d: int(n) for d, n in target_counts.items() if int(n) >= 0}
    where = f"a {label} lattice" if label else "the lattice"
    hint = (" A MIX lattice's site count is a random draw, so absolute counts "
            "have to be worked out for the seed that builds it: "
            "topon.topology.generator_python.count_sites(config, seed) gives "
            "the count and topon.topology.degree_matching.rescale_degree_counts "
            "rescales a P(f) to it." if mix else "")
    if search == "exact":
        # Worded as the C binary words it, which the two keep in step.
        asked = sum(n for d, n in counts.items() if d >= 1)
        if asked > n_sites:
            raise ValueError(
                f"degree_distribution places {asked} active sites but {where} "
                f"has only {n_sites}; enlarge lattice_size, or lower the "
                f"per-degree counts. (Degree 0 is the leftover sites, so it "
                f"does not have to be counted in.){hint}")
        return
    asked = sum(counts.values())
    if is_fully_specified(target_counts, max_func):
        if asked != n_sites:
            raise ValueError(
                f"degree_distribution asks for {asked} sites, {where} has "
                f"{n_sites}; with every degree from 0 to max_functionality "
                f"({max_func}) named, the counts have to add up to the lattice "
                f"exactly, so no trial can succeed. Adjust the degree-0 count "
                f"by {n_sites - asked:+d}, or leave a degree out.{hint}")
    elif asked > n_sites:
        raise ValueError(
            f"degree_distribution asks for {asked} sites, {where} has "
            f"{n_sites}; enlarge lattice_size or lower the counts.{hint}")


class PythonTopologyGenerator:
    """
    A Python implementation of the 'Strict Sculpting' algorithm for polymer network generation.
    Designed to exactly match the logic of the C-based generator (generator_serial_debug11.c).
    """

    def __init__(self, config):
        """
        Initialize with a topology configuration.
        Expected config attributes:
        - lattice_source: "SC" (Simple Cubic), "BCC", "FCC" (Currently only SC implemented for benchmark)
        - dimension: tuple (nx, ny, nz)
        - periodicity: bool or tuple
        - degree_distribution: str (e.g., "0:13,1:25,..." or "e:371")
        - functionality: int (max_func)
        """
        self.config = config
        self.dims = getattr(config, 'dimension', getattr(config, 'lattice_size', (6, 6, 6)))
        if isinstance(self.dims, str):
             # Parse "6x6x6" string
             try:
                 parts = self.dims.lower().split('x')
                 self.dims = (int(parts[0]), int(parts[1]), int(parts[2]))
             except:
                 print(f"Warning: Could not parse dimension string '{self.dims}', using default (6,6,6)")
                 self.dims = (6, 6, 6)
        
        # Accept either the schema's `lattice_type` (current) or the legacy
        # `lattice_source` attribute name (older callers / namedtuples).
        self.lattice_type = getattr(
            config, 'lattice_type',
            getattr(config, 'lattice_source', 'SC'),
        )
        self.max_func = getattr(config, 'max_functionality', getattr(config, 'functionality', 4))

        # Mixed-lattice fractions. Only read when lattice_type == "MIX";
        # the default reproduces a plain simple-cubic lattice.
        self.mix_fractions = dict(
            getattr(config, 'mix_fractions', None)
            or {"SC": 1.0, "BCC": 0.0, "FCC": 0.0}
        )
        # The candidate-edge range in cell units, for every lattice type.
        # `neighbour_cutoff` is the key; `mix_cutoff` is its deprecated
        # name, warned about only when it actually changes the range so
        # the many callers carrying the old default stay quiet.
        # `neighbour_shells` is folded in by the resolver.
        explicit_cutoff = getattr(config, 'neighbour_cutoff', None)
        legacy_cutoff = getattr(config, 'mix_cutoff', None)
        if (explicit_cutoff is None and legacy_cutoff is not None
                and float(legacy_cutoff) != DEFAULT_CUTOFF):
            warnings.warn(
                "mix_cutoff is deprecated; use neighbour_cutoff, which sets "
                "the candidate-edge range for every lattice type.",
                FutureWarning, stacklevel=2,
            )
        self.neighbour_cutoff = resolve_neighbour_cutoff(
            self.lattice_type,
            neighbour_cutoff=explicit_cutoff,
            neighbour_shells=getattr(config, 'neighbour_shells', None),
            mix_cutoff=legacy_cutoff,
        )
        # Kept under the old name for callers that read it back.
        self.mix_cutoff = self.neighbour_cutoff

        # Per-axis periodic boundaries, matching the C searcher's p_dims.
        # Defaults to fully periodic, which is what every builder did
        # unconditionally before this was read at all.
        self.periodicity = self._parse_periodicity(
            getattr(config, 'periodicity', '111')
        )
        
        # Parse degree distribution string
        self.target_counts = defaultdict(lambda: -2)  # -2 means not specified
        self.target_edge_count = -1
        self._parse_degree_distribution(getattr(config, 'degree_distribution', ""))

        # Which sculptor turns the candidate-edge lattice into the network.
        # `None` means the config did not say, and the rule in
        # `resolve_search` picks: the exact search when the request pins
        # every degree from 0 to max_functionality, the strict sculptor
        # otherwise. Resolved here so a bad combination fails at
        # construction rather than after a lattice has been built.
        self.search = resolve_search(
            getattr(config, 'search', None),
            self.target_counts, self.max_func, self.target_edge_count,
        )
        # `or` would turn an explicit 0 into the default, and the duck-typed
        # configs the workflow scripts pass are not bound by the schema's
        # `gt=0`, so the fallback tests for None instead.
        floor = getattr(config, 'min_giant_fraction', None)
        self.min_giant_fraction = float(
            DEFAULT_MIN_GIANT_FRACTION if floor is None else floor
        )
        # None leaves it to degree_matching.DEFAULT_ODD_WALKS (off).
        self.odd_walks = getattr(config, 'odd_walks', None)

        # The random streams every draw below comes from. With a seed they
        # are private and restarted at each generate(), see _seed_streams.
        self.seed = getattr(config, 'seed', None)
        self._seed_streams()

    def _seed_streams(self):
        """Point ``_rng`` / ``_np_rng`` at the streams this run draws from.

        With ``seed`` set: a fresh ``random.Random(seed)`` for the lattice
        draw and the strict sculptor, and a fresh
        ``np.random.RandomState(seed)`` for the exact search's seeds. Those
        are the generators behind ``random.seed(seed)`` and
        ``np.random.seed(seed)``, so seed n draws exactly what seeding the
        two global streams with n just before ``generate()`` drew, which
        is how configs pinned a graph before the field existed, and the
        global streams are left alone. Without a seed the module-level
        streams are used, as always.
        """
        if self.seed is None:
            self._rng = random
            self._np_rng = np.random
        else:
            self._rng = random.Random(int(self.seed))
            self._np_rng = np.random.RandomState(int(self.seed))

    @staticmethod
    def _wrap_hr(coord, hr_dims, periodic):
        """Wrap a high-res neighbour address, or None if it left an open face.

        Mirrors the C searcher's neighbour step: wrap the axis when it is
        periodic, otherwise keep the raw index and let the bounds check
        drop it.
        """
        out = []
        for value, extent, is_periodic in zip(coord, hr_dims, periodic):
            if is_periodic:
                out.append(value % extent)
            elif 0 <= value < extent:
                out.append(value)
            else:
                return None
        return tuple(out)

    @staticmethod
    def _parse_periodicity(value):
        """Normalise a periodicity spec to ``(px, py, pz)`` booleans.

        Accepts the C searcher's ``"111"`` / ``"110"`` digit string, a
        single bool applied to all three axes, or any 3-element iterable
        of truthy values. Anything unrecognised falls back to fully
        periodic with a warning rather than silently producing an open
        lattice, since a surprise free surface changes the physics.

        A non-periodic axis simply omits the wrap-around bonds, so the
        lattice grows a free surface there and the sites on it have
        reduced coordination.
        """
        if value is None:
            return (True, True, True)
        if isinstance(value, bool):
            return (value, value, value)
        if isinstance(value, str):
            digits = [c for c in value.strip() if c in "01"]
            if len(digits) == 3:
                return tuple(c == "1" for c in digits)
            print(f"Warning: could not parse periodicity {value!r}; "
                  f"using fully periodic '111'.")
            return (True, True, True)
        try:
            axes = tuple(bool(v) for v in value)
        except TypeError:
            print(f"Warning: could not parse periodicity {value!r}; "
                  f"using fully periodic '111'.")
            return (True, True, True)
        if len(axes) != 3:
            print(f"Warning: periodicity {value!r} does not have 3 axes; "
                  f"using fully periodic '111'.")
            return (True, True, True)
        return axes

    def _parse_degree_distribution(self, dist_str):
        """Fill ``target_counts`` / ``target_edge_count`` from the spec string.

        The parsing itself lives in ``topology.degree_matching`` so the
        pipeline can read a config's request without building a
        generator; this keeps the defaultdict, whose -2 sentinel is what
        the strict sculptor reads for "unspecified".
        """
        counts, edge_count = parse_degree_distribution(dist_str)
        for degree, count in counts.items():
            self.target_counts[degree] = count
        if edge_count >= 0:
            self.target_edge_count = edge_count

    def _lattice_label(self):
        """Human-readable "<nx>x<ny>x<nz> <TYPE>" tag for error messages."""
        label = f"{self.dims[0]}x{self.dims[1]}x{self.dims[2]} {self.lattice_type}"
        if self.lattice_type == "MIX":
            frac = ",".join(
                f"{k}:{self.mix_fractions.get(k, 0.0):g}"
                for k in ("SC", "BCC", "FCC")
            )
            label += f" ({frac})"
        if self.neighbour_cutoff != DEFAULT_CUTOFF:
            label += f" cutoff {self.neighbour_cutoff:g}"
        return label

    def _validate_targets_reachable(self, base_graph, target_counts=None):
        """Fail fast when the requested degree target can never be met.

        Sculpting only ever REMOVES edges from the freshly-built lattice, so
        that full lattice is a hard ceiling: its edge count bounds any ``e:N``
        target and its maximum node degree bounds any per-degree ``d:N`` target.
        Without this guard an over-target request makes :meth:`generate` grind
        through every trial (each one doomed) before giving up — for a large
        ``trials`` count that looks like an indefinite hang. Bounds are read
        from the actual constructed graph, not a ``3*nx*ny*nz`` formula, so
        periodic-boundary edge collapse on tiny lattices (e.g. a 2x2x2 SC has
        12 edges, not 24) is accounted for.

        ``target_counts`` overrides the one parsed from
        ``degree_distribution``: the exact search can be handed a target the
        caller worked out (``generate(need=...)``), and it is that target,
        not the config's, these bounds have to hold for.

        Raises
        ------
        ValueError
            If the target edge count exceeds the lattice's edges, or a
            per-degree target asks for more nodes than exist or for a degree
            higher than any node in the base lattice.
        """
        targets = self.target_counts if target_counts is None else target_counts
        # Name the target the numbers came from, so an override's failure does
        # not send the reader looking for a degree_distribution that says
        # something else.
        source = "degree_distribution" if target_counts is None else "degree target"
        base_edges = base_graph.number_of_edges()
        base_nodes = base_graph.number_of_nodes()
        label = self._lattice_label()

        # --- e:N  (total edge-count target) ---
        if self.target_edge_count != -1 and self.target_edge_count > base_edges:
            raise ValueError(
                f"degree_distribution e:{self.target_edge_count} exceeds the "
                f"{base_edges} edges of a {label} lattice; sculpting only "
                f"removes edges, so this target is unreachable."
            )

        # --- d:N  (per-degree targets) ---
        # target_counts is a defaultdict; iterate a snapshot of only the
        # explicitly-parsed entries. count == 0 means "forbidden" (reachable)
        # and the -2 sentinel means "unspecified"; both are skipped.
        if targets:
            max_base_degree = max((d for _, d in base_graph.degree()), default=0)
            for degree, count in list(targets.items()):
                if count <= 0:
                    continue
                if count > base_nodes:
                    raise ValueError(
                        f"{source} {degree}:{count} exceeds the "
                        f"{base_nodes} nodes of a {label} lattice; a lattice "
                        f"cannot hold more nodes of degree {degree} than it has "
                        f"nodes, so this target is unreachable."
                    )
                if degree > max_base_degree:
                    raise ValueError(
                        f"{source} {degree}:{count} requires degree-"
                        f"{degree} nodes, but the maximum degree in a {label} "
                        f"lattice is {max_base_degree}; sculpting only removes "
                        f"edges, so this target is unreachable."
                    )
                # Checked after the lattice bound, which is the more
                # fundamental reason when both apply. Stage 3 prunes every
                # node to max_func and stage 4 refuses to finish while any
                # sits above it, so a target above the ceiling can never be
                # met however rich the lattice was. Without this the run
                # burns through every trial before giving up, and the C
                # searcher used to report success on such a request
                # outright -- see topon/topology/csrc/README.md.
                if degree > self.max_func:
                    raise ValueError(
                        f"{source} {degree}:{count} requires degree-"
                        f"{degree} nodes on a {label} lattice, but "
                        f"max_functionality is {self.max_func}; sculpting "
                        f"enforces that ceiling, so no node can finish with "
                        f"degree {degree}."
                    )

        # --- the sites have to fit on the lattice that was built ---
        # The exact search places every requested site itself, so an
        # over-full request is a plain arithmetic failure; a strict request
        # that names every degree has to add up to the lattice exactly, and
        # one that names some must not exceed it. Caught here so it reads
        # as a config error rather than a run of identical failed trials.
        # A MIX lattice's count is a draw, which the message says.
        check_site_count(self.target_counts, self.max_func, self.search,
                         base_nodes, label, mix=self.lattice_type == "MIX")

        if self.search == "exact":
            # The handshake lemma: every edge adds 2 to the degree sum, so
            # an odd one belongs to no graph. Cheap to check and otherwise
            # only shows up as a stubborn residual of exactly 1.
            degree_sum = sum(
                d * self.target_counts[d] for d in range(1, self.max_func + 1)
            )
            if degree_sum % 2:
                raise ValueError(_odd_degree_sum_message(degree_sum))

        # --- strict search pinning every degree: the same handshake rule ---
        # A request that names every degree from 0 to max_functionality
        # fixes the whole degree sequence for the strict sculptor too (no
        # node can end above the ceiling), so an odd sum is as impossible
        # there. Before this the sculptor burnt every trial on it. A
        # partial request is left alone: the degrees it does not name can
        # take up the parity.
        elif is_fully_specified(self.target_counts, self.max_func):
            degree_sum = sum(
                d * self.target_counts[d] for d in range(1, self.max_func + 1)
            )
            if degree_sum % 2:
                raise ValueError(_odd_degree_sum_message(degree_sum))

    def generate(self, trials=1, max_saves=1, time_limit=None, double_pairs=None,
                 need=None):
        """
        Run multiple trials to generate a valid network.
        Returns a list of successful graphs (networkx.Graph objects).

        ``self.search`` decides which sculptor runs. The strict one tries
        ``trials`` random sculpts and keeps the ones that land on the
        target; the exact one solves the degree sequence directly and
        needs only a handful of seeds, so ``trials`` does not apply to it
        (see :meth:`_generate_exact`).

        Args:
            need: Exact search only. ``{degree: count}`` to sculpt for,
                in place of the one parsed from ``degree_distribution``.
                The caller works it out when something downstream has to
                be paid for in advance: the defects stage reserves the
                capacity its triangles and four-cycles will spend (see
                ``topon.assignment.defects.reserve_capacity``), and passes
                the reduced target here so the delivered P(f) is the
                requested one.
            double_pairs: Exact search only. ``{(a, b): count}`` forced
                parallel edges between sites of target degree ``a`` and
                ``b`` -- the reference network's secondary loops with
                their endpoint degrees. The result is then a MultiGraph.
        """
        successful_graphs = []

        # A seeded generator starts every run from its seed, so calling
        # generate() twice gives the same graph twice.
        if self.seed is not None:
            self._seed_streams()
        base_graph = self._create_lattice(self.dims, self.lattice_type)
        # Reject structurally-unreachable targets before churning through
        # trials (sculpting only removes edges — see _validate_targets_reachable).
        # The override is what gets sculpted when there is one, so it is what
        # the guards have to read; otherwise they check the config's target
        # and the real one fails later, deeper, with a worse message.
        self._validate_targets_reachable(base_graph, target_counts=need)

        if self.search == "exact":
            return self._generate_exact(
                base_graph, max_saves=max_saves, double_pairs=double_pairs,
                need=need,
            )
        if double_pairs:
            raise ValueError(
                "double_pairs is a parameter of the exact search; the strict "
                "sculptor only removes edges and cannot place a parallel one. "
                "Set topology.generator.search to 'exact'."
            )
        if need is not None:
            # `search` is resolved from degree_distribution at construction,
            # so a need passed against a partly specified config would land
            # here and be dropped without a word.
            raise ValueError(
                "need is a parameter of the exact search; the strict sculptor "
                "reads its target from degree_distribution. Set "
                "topology.generator.search to 'exact', or give every degree "
                "from 0 to max_functionality in degree_distribution so the "
                "exact search is chosen."
            )
        print(f"DEBUG: Entering generate loop (trials={trials})")
        
        start_time = time.time()
        
        for trial in range(trials):
            if time_limit and (time.time() - start_time > time_limit):
                print(f"  [Python] Time limit reached ({time_limit}s). Stopping.")
                break
                
            if trial % 100 == 0:
                 print(f"  [Python] Trial {trial}/{trials}...")
            g = self.run_single_trial(base_graph, trial)
            if g is not None:
                print(f"  [Python] Success on trial {trial}!")
                successful_graphs.append(g)
                if len(successful_graphs) >= max_saves:
                    break

        return successful_graphs

    def _generate_exact(self, base_graph, max_saves=1, double_pairs=None,
                        need=None):
        """Solve the degree sequence directly instead of sculpting for it.

        The exact search does not sample trials: it assigns the requested
        degrees to sites and completes the degree sequence with augmenting
        paths, so one seed almost always lands. ``max_trials`` therefore
        does not apply; each saved graph gets up to
        ``degree_matching.DEFAULT_ATTEMPTS`` seeds, and a scaffold with no
        spare candidate edge (Diamond at ``max_functionality = 4``) is
        reported after the first, since retrying cannot find room that
        does not exist.

        Seeds come from the NumPy stream ``_np_rng`` (the global one, or the
        generator's own when ``topology.generator.seed`` is set), so either
        the seed field or ``np.random.seed(n)`` before generating makes the
        result reproducible, and the seed lands in the record for replay.

        Returns a list of graphs in the same form the strict sculptor
        returns, minus ``move_history``: the exact search has no
        edge-removal history, so the sculpting animation in
        ``assets/gallery/`` needs ``search: "strict"``.
        """
        if need is None:
            need = needs_from_targets(self.target_counts, self.max_func)
        else:
            need = {int(d): int(n) for d, n in need.items()}
        graphs = []
        for _ in range(max(1, max_saves)):
            seed = int(self._np_rng.randint(0, 2 ** 31 - 1))
            g = build_exact_graph(
                base_graph, need, self.max_func, seed,
                double_pairs=double_pairs,
                min_giant_fraction=self.min_giant_fraction,
                attempts=DEFAULT_ATTEMPTS,
                odd_walks=self.odd_walks,
                label=self._lattice_label(),
            )
            rec = g.graph["sculpt_record"]
            print(f"  [exact] reached the target degree counts in "
                  f"{rec['seconds']:.2f}s "
                  f"({rec['n_sites'] - rec['n_vacancies']} active sites, "
                  f"{g.number_of_edges()} edges, giant "
                  f"{rec['giant_frac']:.3f})")
            graphs.append(g)
        return graphs

    def _create_lattice(self, dims, lattice_type):
        """Creates the initial full lattice: the candidate-edge set.

        A lattice plus a range. The four pure builders enumerate their
        canonical neighbour pattern, matching the C generator's
        create_*_lattice functions; when ``neighbour_cutoff`` is not the
        default 1.0 their bonds are rebuilt by the minimum-image search
        so that every pair within the range is a candidate edge. MIX
        always searches. The graph records the cutoff and the shells it
        actually produced.
        """
        nx_val, ny_val, nz_val = dims

        if lattice_type == "SC":
            g = self._create_sc_lattice(nx_val, ny_val, nz_val)
        elif lattice_type == "BCC":
            g = self._create_bcc_lattice(nx_val, ny_val, nz_val)
        elif lattice_type == "FCC":
            g = self._create_fcc_lattice(nx_val, ny_val, nz_val)
        elif lattice_type == "MIX":
            g = self._create_mixed_lattice(nx_val, ny_val, nz_val)
        elif lattice_type in ("Diamond", "DIAMOND"):
            # Delegates to the standalone module rather than inlining the
            # basis here: the Diamond logic deliberately lives in its own
            # file so it can be reviewed in isolation. This is a dispatch
            # bridge so a config can name Diamond on either generator, not
            # a merge of the two.
            from topon.topology.generator_python_diamond import (
                create_diamond_lattice,
            )
            g = create_diamond_lattice(nx_val, ny_val, nz_val, self.periodicity)
        else:
            raise NotImplementedError(
                f"Lattice type {lattice_type} not supported. "
                f"Use SC, BCC, FCC, Diamond, or MIX."
            )

        if lattice_type != "MIX" and self.neighbour_cutoff != DEFAULT_CUTOFF:
            self._rebuild_edges_within_cutoff(g)
        self._record_neighbourhood(g)
        return g

    def _rebuild_edges_within_cutoff(self, g):
        """Replace a pure lattice's canonical bonds by every pair within range.

        The sites, their ids and the recorded cell are untouched; only the
        edge set changes, built by the same search MIX uses, so SC at a
        cutoff is exactly MIX at fractions (1, 0, 0) and that cutoff. Open
        axes are honoured by the search itself.
        """
        nodes = list(g.nodes())
        positions = [g.nodes[n]["pos"] for n in nodes]
        g.clear_edges()
        g.add_edges_from(
            (nodes[u], nodes[v])
            for u, v in edges_within_cutoff(
                positions, g.graph["box"], g.graph["periodicity"],
                self.neighbour_cutoff,
            )
        )

    def _record_neighbourhood(self, g):
        """Stamp the range and the shells it produced on the graph.

        ``shell_distances`` is read off the built edges rather than
        assumed from the lattice type, so it is right for a sparse MIX
        draw, an open axis and any cutoff. A cutoff beyond a third of a
        periodic axis is flagged here as well as by ``topon doctor``,
        since workflow scripts build lattices without ever running it:
        three candidate edges can then close a cycle around the box.
        """
        g.graph["neighbour_cutoff"] = float(self.neighbour_cutoff)
        shells = shell_distances(g)
        g.graph["shell_distances"] = shells
        g.graph["neighbour_shells"] = len(shells)

        box = g.graph.get("box")
        periodic = g.graph.get("periodicity", (True, True, True))
        if box is not None:
            wrapped = "".join(
                axis for axis, length, p in zip("xyz", box, periodic)
                if p and self.neighbour_cutoff > length / 3.0 + 1e-9
            )
            if wrapped:
                print(
                    f"Warning: neighbour_cutoff {self.neighbour_cutoff:g} is "
                    f"more than a third of the periodic box along {wrapped} "
                    f"(box {tuple(box)}); three candidate edges can close a "
                    f"cycle around the box. Use at least "
                    f"{math.ceil(3.0 * self.neighbour_cutoff)} cells per "
                    f"periodic axis, or a smaller cutoff."
                )

    def _create_sc_lattice(self, nx_val, ny_val, nz_val):
        """Simple Cubic: N nodes, 6 neighbours each when fully periodic.

        Honours ``self.periodicity`` per axis, matching the C searcher's
        ``if (p_dims[0] || x < Nx - 1)`` guard: an open axis simply omits
        the wrap-around bond, leaving a free surface whose sites have
        reduced coordination.
        """
        px, py, pz = self.periodicity
        g = nx.Graph()
        g.graph["box"] = (float(nx_val), float(ny_val), float(nz_val))
        g.graph["periodicity"] = (px, py, pz)

        total_nodes = nx_val * ny_val * nz_val
        for i in range(total_nodes):
            z = i // (nx_val * ny_val)
            rem = i % (nx_val * ny_val)
            y = rem // nx_val
            x = rem % nx_val
            g.add_node(i, pos=(float(x), float(y), float(z)))

        for z in range(nz_val):
            for y in range(ny_val):
                for x in range(nx_val):
                    u = z * (nx_val * ny_val) + y * nx_val + x

                    if px or x < nx_val - 1:
                        v_x = z * (nx_val * ny_val) + y * nx_val + (x + 1) % nx_val
                        if u != v_x and not g.has_edge(u, v_x):
                            g.add_edge(u, v_x)

                    if py or y < ny_val - 1:
                        v_y = z * (nx_val * ny_val) + ((y + 1) % ny_val) * nx_val + x
                        if u != v_y and not g.has_edge(u, v_y):
                            g.add_edge(u, v_y)

                    if pz or z < nz_val - 1:
                        v_z = ((z + 1) % nz_val) * (nx_val * ny_val) + y * nx_val + x
                        if u != v_z and not g.has_edge(u, v_z):
                            g.add_edge(u, v_z)

        return g

    def _create_bcc_lattice(self, nx_val, ny_val, nz_val):
        """Body-Centered Cubic: 2*N nodes, 8 neighbors each (periodic BC).

        Mirrors C create_bcc_lattice:
        - Corner atoms at (i, j, k), high-res coords (2i, 2j, 2k)
        - Body atoms at (i+0.5, j+0.5, k+0.5), high-res coords (2i+1, 2j+1, 2k+1)
        - Neighbors via all 8 (±1, ±1, ±1) offsets in high-res space

        The periodic cell is (nx, ny, nz), not the extent of the site
        coordinates: body-centre sites sit at +0.5, so the coordinates
        only reach nx-0.5 and a max-min+1 estimate would overshoot by
        half a cell. Recording the cell explicitly keeps every downstream
        minimum-image calculation on the right periodic replica.
        """
        px, py, pz = self.periodicity
        g = nx.Graph()
        g.graph["box"] = (float(nx_val), float(ny_val), float(nz_val))
        g.graph["periodicity"] = (px, py, pz)

        hr_nx = 2 * nx_val
        hr_ny = 2 * ny_val
        hr_nz = 2 * nz_val

        # Map from high-res (hx, hy, hz) index -> node id
        coord_map = {}
        node_idx = 0

        for k in range(nz_val):
            for j in range(ny_val):
                for i in range(nx_val):
                    # Corner node
                    cx, cy, cz = 2 * i, 2 * j, 2 * k
                    g.add_node(node_idx, pos=(float(i), float(j), float(k)))
                    coord_map[(cx, cy, cz)] = node_idx
                    node_idx += 1

                    # Body-center node
                    bx, by, bz = 2 * i + 1, 2 * j + 1, 2 * k + 1
                    g.add_node(node_idx, pos=(i + 0.5, j + 0.5, k + 0.5))
                    coord_map[(bx, by, bz)] = node_idx
                    node_idx += 1

        # Connect: each node links to 8 diagonal neighbors in high-res space.
        # An open axis does not wrap, so neighbours off that face simply
        # do not exist and the surface sites lose coordination.
        for (hx, hy, hz), uid in coord_map.items():
            for dx in (-1, 1):
                for dy in (-1, 1):
                    for dz in (-1, 1):
                        nbr = self._wrap_hr(
                            (hx + dx, hy + dy, hz + dz),
                            (hr_nx, hr_ny, hr_nz), (px, py, pz),
                        )
                        if nbr is None:
                            continue
                        vid = coord_map.get(nbr)
                        if vid is not None and uid < vid:
                            g.add_edge(uid, vid)

        return g

    def _create_fcc_lattice(self, nx_val, ny_val, nz_val):
        """Face-Centered Cubic: 4*N nodes, 12 neighbors each (periodic BC).

        Mirrors C create_fcc_lattice:
        - Corner at (2i, 2j, 2k)
        - Face-XY at (2i+1, 2j+1, 2k)
        - Face-XZ at (2i+1, 2j, 2k+1)
        - Face-YZ at (2i, 2j+1, 2k+1)
        - Neighbors via 12 face-diagonal offsets: XY(±1,±1,0), XZ(±1,0,±1), YZ(0,±1,±1)

        As for BCC, the periodic cell is (nx, ny, nz) while the face-site
        coordinates only reach nx-0.5, so the cell is recorded rather than
        inferred from the coordinate extent.
        """
        px, py, pz = self.periodicity
        g = nx.Graph()
        g.graph["box"] = (float(nx_val), float(ny_val), float(nz_val))
        g.graph["periodicity"] = (px, py, pz)

        hr_nx = 2 * nx_val
        hr_ny = 2 * ny_val
        hr_nz = 2 * nz_val

        coord_map = {}
        node_idx = 0

        for k in range(nz_val):
            for j in range(ny_val):
                for i in range(nx_val):
                    # Corner
                    coord_map[(2 * i, 2 * j, 2 * k)] = node_idx
                    g.add_node(node_idx, pos=(float(i), float(j), float(k)))
                    node_idx += 1
                    # Face XY (z shared)
                    coord_map[(2 * i + 1, 2 * j + 1, 2 * k)] = node_idx
                    g.add_node(node_idx, pos=(i + 0.5, j + 0.5, float(k)))
                    node_idx += 1
                    # Face XZ (y shared)
                    coord_map[(2 * i + 1, 2 * j, 2 * k + 1)] = node_idx
                    g.add_node(node_idx, pos=(i + 0.5, float(j), k + 0.5))
                    node_idx += 1
                    # Face YZ (x shared)
                    coord_map[(2 * i, 2 * j + 1, 2 * k + 1)] = node_idx
                    g.add_node(node_idx, pos=(float(i), j + 0.5, k + 0.5))
                    node_idx += 1

        # 12 nearest-neighbor offsets in high-res space (face diagonals)
        fcc_offsets = [
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),  # XY plane
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),  # XZ plane
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),  # YZ plane
        ]

        for (hx, hy, hz), uid in coord_map.items():
            for dx, dy, dz in fcc_offsets:
                nbr = self._wrap_hr(
                    (hx + dx, hy + dy, hz + dz),
                    (hr_nx, hr_ny, hr_nz), (px, py, pz),
                )
                if nbr is None:
                    continue
                vid = coord_map.get(nbr)
                if vid is not None and uid < vid:
                    g.add_edge(uid, vid)

        return g

    def _create_mixed_lattice(self, nx_val, ny_val, nz_val):
        """Overlay of SC / BCC / FCC basis sites in one cubic cell.

        All three lattices share the cell corner, and each adds its own
        sites on top of it: BCC one body centre, FCC three face centres.
        So the corner is placed in every cell, the body centre with
        probability ``mix_fractions["BCC"]`` and each face centre with
        probability ``mix_fractions["FCC"]``. The ``"SC"`` fraction is the
        remainder, contributing no site of its own, which is what makes
        the three fractions a partition summing to 1.

        Expected site count is ``Nx*Ny*Nz * (1 + f_bcc + 3*f_fcc)``, which
        recovers the exact counts of the pure lattices: ``N`` for SC,
        ``2N`` for BCC, ``4N`` for FCC.

        Edges join every pair within ``neighbour_cutoff`` under the
        minimum image, rather than the fixed offset patterns the pure
        builders use at the default cutoff, because on a mixed point set
        there is no single neighbour shell. The 1.0 default is the
        simple-cubic nearest-neighbour distance, which keeps the
        always-present corner sublattice connected however few body and
        face sites are drawn.

        Two consequences worth knowing:

        * ``MIX`` at fractions ``(1, 0, 0)`` reproduces ``SC`` exactly,
          same node ids, positions and edges, at every cutoff. At the
          default cutoff it is **not** true of the other two corners: at
          ``(0, 1, 0)`` the cutoff also admits the
          corner-corner shell at 1.0, so nodes carry 14 neighbours rather
          than BCC's 8, and at ``(0, 0, 1)`` 18 rather than FCC's 12. Use
          ``lattice_type`` SC / BCC / FCC when the canonical coordination
          is what you want; ``MIX`` is for genuine mixtures.
        * Body and face sites can land 0.5 cells apart, closer than any
          pure lattice's nearest-neighbour distance (SC 1.0, BCC 0.866,
          FCC 0.707). Since DP is assigned independently of edge length,
          that widens the spread of bond lengths a strand of given DP is
          built at.
        """
        f_bcc = float(self.mix_fractions.get("BCC", 0.0))
        f_fcc = float(self.mix_fractions.get("FCC", 0.0))

        px, py, pz = self.periodicity
        g = nx.Graph()
        g.graph["box"] = (float(nx_val), float(ny_val), float(nz_val))
        g.graph["periodicity"] = (px, py, pz)
        g.graph["mix_fractions"] = dict(self.mix_fractions)

        positions = mix_site_positions(nx_val, ny_val, nz_val, f_bcc, f_fcc,
                                       self._rng)

        for idx, pos in enumerate(positions):
            g.add_node(idx, pos=pos)

        # Neighbour search under the minimum image, shared with the pure
        # builders at a non-default cutoff. Only periodic axes take the
        # minimum image; an open axis keeps the raw separation, so
        # nothing bonds across that face.
        g.add_edges_from(edges_within_cutoff(
            positions, (nx_val, ny_val, nz_val), (px, py, pz),
            self.neighbour_cutoff,
        ))

        return g

    def run_single_trial(self, base_graph, trial_num):
        """
        Runs the Strict Sculpting algorithm stages on a copy of the base graph.
        """
        g = base_graph.copy()
        total_nodes = g.number_of_nodes()
        
        # Track edge removal history for visualization
        move_history = []
        
        # Node Status: 0=ACTIVE, 1=IS_DEGREE_0, 2=IS_DEGREE_1
        # In Python we can use a dict or node attribute
        # Default active
        node_status = {n: "ACTIVE" for n in g.nodes()}

        # Running degree histogram: hist[d] is the number of sites of
        # degree d, kept in step with every edge removed or put back. The
        # move-safety check and stage 4's completion test read it instead
        # of recounting all N sites each time, which at 2,744 sites was
        # most of a trial. Every edge change goes through drop/restore.
        deg = g.degree
        hist = [0] * (max((d for _, d in deg), default=0) + 2)
        for _, d in deg:
            hist[d] += 1

        def drop(u, v):
            du, dv = deg[u], deg[v]
            hist[du] -= 1
            hist[du - 1] += 1
            hist[dv] -= 1
            hist[dv - 1] += 1
            g.remove_edge(u, v)

        def restore(u, v):
            g.add_edge(u, v)
            du, dv = deg[u], deg[v]
            hist[du - 1] -= 1
            hist[du] += 1
            hist[dv - 1] -= 1
            hist[dv] += 1

        # Shuffle node indices
        node_indices = list(g.nodes())
        self._rng.shuffle(node_indices)
        
        current_node_offset = 0
        
        # Targets
        n0_target = max(0, self.target_counts[0]) if self.target_counts[0] != -2 else 0
        n1_target = max(0, self.target_counts[1]) if self.target_counts[1] != -2 else 0
        
        target_degree_sum = self.target_edge_count * 2
        
        # --- Stage 1: Set d0 (Strict) ---
        for _ in range(n0_target):
            if current_node_offset >= total_nodes: break
            node_idx = node_indices[current_node_offset]
            current_node_offset += 1
            
            while g.degree[node_idx] > 0:
                neighbors = list(g.neighbors(node_idx))
                removed = False
                for neighbor in neighbors:
                    # SC Optimization in C: if neighbor degree <= 2, skip to avoid breaking chains too much
                    if g.degree[neighbor] <= 2:
                        continue
                        
                    if not self._is_move_safe(g, node_idx, neighbor, stage=1, 
                                              target_degree_sum=target_degree_sum, 
                                              current_total_degree_sum=-1, # sum not needed for stg 1
                                              degree_counts=hist):
                        continue

                    drop(node_idx, neighbor)
                    move_history.append({'stage': 1, 'edge': (node_idx, neighbor), 'reason': 'd0'})
                    removed = True
                    break
                
                if not removed:
                    return None # Failed to isolate node
            
            node_status[node_idx] = "IS_DEGREE_0"
            
        # --- Stage 2: Set d1 (Strict) ---
        for _ in range(n1_target):
            if current_node_offset >= total_nodes: break
            node_idx = node_indices[current_node_offset]
            if node_status[node_idx] != "ACTIVE": 
                # Should not happen as we iterate sequential offset, but good check
                pass
            current_node_offset += 1
            
            while g.degree[node_idx] > 1:
                neighbors = list(g.neighbors(node_idx))
                self._rng.shuffle(neighbors)
                removed = False
                for neighbor in neighbors:
                    if g.degree[neighbor] <= 2:
                        continue

                    if not self._is_move_safe(g, node_idx, neighbor, stage=2, 
                                              target_degree_sum=target_degree_sum, 
                                              current_total_degree_sum=-1,
                                              degree_counts=hist):
                        continue

                    drop(node_idx, neighbor)
                    
                    if self._is_subgraph_connected(g, node_status):
                        move_history.append({'stage': 2, 'edge': (node_idx, neighbor), 'reason': 'd1'})
                        removed = True
                        break
                    else:
                        restore(node_idx, neighbor) # Backtrack
                        
                if not removed:
                    return None # Failed to reduce to d1
            
            node_status[node_idx] = "IS_DEGREE_1"

        # --- Stage 3: Enforce Max Functionality (Strict) ---
        for i in range(total_nodes):
            node_idx = node_indices[i]
            if node_status[node_idx] != "ACTIVE": continue
            
            while g.degree[node_idx] > self.max_func:
                neighbors = list(g.neighbors(node_idx))
                self._rng.shuffle(neighbors)
                removed = False
                for neighbor in neighbors:
                    if g.degree[neighbor] <= 2:
                        continue
                        
                    if not self._is_move_safe(g, node_idx, neighbor, stage=3, 
                                              target_degree_sum=target_degree_sum, 
                                              current_total_degree_sum=-1,
                                              degree_counts=hist):
                        continue

                    drop(node_idx, neighbor)
                    if self._is_subgraph_connected(g, node_status):
                        move_history.append({'stage': 3, 'edge': (node_idx, neighbor), 'reason': 'max_func'})
                        removed = True
                        break
                    else:
                        restore(node_idx, neighbor) # Backtrack
                
                if not removed:
                    return None # Failed to enforce max func

        # --- Stage 4: Systematic Search Loop ---
        # Sites marked IS_DEGREE_0 / IS_DEGREE_1 sit at degree 0 or 1 by
        # now, and stage 4 only removes edges whose ends both have degree
        # 2 or more, so their degrees are fixed from here on: the active
        # degree-1 count is hist[1] less this constant.
        inactive_deg1 = sum(1 for n in g.nodes()
                            if node_status[n] != "ACTIVE" and deg[n] == 1)

        def count_at(d):
            return hist[d] if 0 <= d < len(hist) else 0

        while True:
            # Check current distribution
            current_degree_sum = sum(d * c for d, c in enumerate(hist))
            # Only an active site can sit above max_func (the inactive ones
            # are at 0 or 1), so the site scan runs only when the histogram
            # says some site does.
            has_high_degree = False
            if any(hist[self.max_func + 1:]):
                has_high_degree = any(
                    node_status[n] == "ACTIVE" and d > self.max_func
                    for n, d in deg)

            is_done = True
            
            if has_high_degree:
                is_done = False
            else:
                # 1. Check explicit targets
                for d, count in self.target_counts.items():
                    if count >= 0 and count_at(d) != count:
                        is_done = False
                        break
                
                # 2. Check total edge count / connectivity depending on mode
                if is_done:
                    if self.target_edge_count != -1:
                         # e:N mode
                        if current_degree_sum != target_degree_sum:
                            is_done = False
                        if not self._is_subgraph_connected(g, node_status):
                            is_done = False
                    else:
                        # Legacy mode (d0 already met, just need connectivity)
                        if self._is_subgraph_connected(g, node_status):
                             is_done = True
                        else:
                             return None # Failed connectivity check at end
            
            if is_done:
                # Attach move history to graph
                g.graph['move_history'] = move_history
                g.graph['sculpt_search'] = 'strict'
                return g
            
            # Not done, perform systematic edge removal
            edges = list(g.edges())
            self._rng.shuffle(edges)
            
            move_made = False
            
            for u, v in edges:
                u_deg = g.degree[u]
                v_deg = g.degree[v]
                
                if u_deg <= 1 or v_deg <= 1: continue
                
                # Legacy d2 check
                if u_deg == 2 or v_deg == 2:
                    if self.target_counts[1] != -1: # if d1 count is tracked
                        # the active degree-1 sites (see inactive_deg1)
                        d1_count = hist[1] - inactive_deg1
                        if d1_count >= self.target_counts[1]:
                            continue
                            
                if not self._is_move_safe(g, u, v, stage=4, 
                                          target_degree_sum=target_degree_sum, 
                                          current_total_degree_sum=current_degree_sum,
                                          degree_counts=hist):
                    continue

                drop(u, v)
                
                if self._is_subgraph_connected(g, node_status):
                    move_history.append({'stage': 4, 'edge': (u, v), 'reason': 'systematic'})
                    move_made = True
                    break # Restart loop
                else:
                    restore(u, v) # Backtrack
            
            if not move_made:
                return None # Stuck

    def _is_subgraph_connected(self, g, node_status):
        """
        Checks if the subgraph of ACTIVE nodes is connected.
        Ignores IS_DEGREE_0 nodes (they are isolated by definition).
        IS_DEGREE_1 nodes are part of the active graph usually?
        Wait, C code: `if (node_status[i] == ACTIVE) ...`
        Wait, in C `IS_DEGREE_1` nodes are *excluded* from the connectivity check loop?
        
        Let's look at C code `is_subgraph_connected`:
        `if (node_status[i] == ACTIVE) ...`
        Yes! C code ONLY checks connectivity among "ACTIVE" nodes.
        Nodes marked IS_DEGREE_0 or IS_DEGREE_1 are NOT part of the connectivity check.
        They are considered "done" and "removed" from the main component logic?
        
        Wait, `IS_DEGREE_1` nodes (dangling ends) *should* be connected to the main component.
        If we exclude them from the check, we only ensure the core is connected.
        Let's double check C code line 304: `if (node_status[node_status[i] == ACTIVE])`.
        
        In Stage 2 (Set d1), we mark nodes as `IS_DEGREE_1`.
        If they are excluded from connectivity check, that means we only care if the *remaining* network is connected.
        Dangling ends are by definition connected to *something* (degree 1), so as long as that something is in the main component, they are fine.

        Implementation note: this walks the adjacency mapping directly
        rather than building ``g.subgraph(active)`` and calling
        ``nx.is_connected``. The two give identical answers, but a
        NetworkX subgraph is a *view* that re-evaluates its node filter on
        every neighbour access, which costs about six million predicate
        calls per check on a 1000-node lattice. Since this function is
        roughly 99% of the generator's runtime, that made the whole
        generator about eight times slower than it needed to be.
        """
        # Edges count only when BOTH ends are ACTIVE, matching the C
        # searcher: `if (node_status[pCrawl->dest] == ACTIVE) unite_sets(...)`.
        # Raw {node: {neighbour: attrs}}. `_adj` is NetworkX-internal but
        # stable and ~1.6x faster than the public `adj` view, so take it
        # when present and fall back if a future release renames it.
        adj = getattr(g, "_adj", None)
        if adj is None:
            adj = g.adj
        start = None
        n_active = 0
        for n in adj:
            if node_status[n] == "ACTIVE":
                n_active += 1
                if start is None:
                    start = n
        if start is None:
            return True

        seen = {start}
        stack = [start]
        while stack:
            for nbr in adj[stack.pop()]:
                if nbr not in seen and node_status[nbr] == "ACTIVE":
                    seen.add(nbr)
                    stack.append(nbr)
        return len(seen) == n_active


    def _is_move_safe(self, g, u, v, stage, target_degree_sum, current_total_degree_sum,
                      degree_counts=None):
        """
        Equivalent to C `is_move_safe`.

        ``degree_counts`` is the caller's running degree histogram
        (``degree_counts[d]`` sites of degree ``d``). Without it the counts
        are taken by scanning every site, which is what this did on every
        call before 0.4.0 and what made a trial at 2,744 sites slow.
        """
        def count_at(d):
            if degree_counts is None:
                return sum(1 for n in g.nodes() if g.degree[n] == d)
            return degree_counts[d] if 0 <= d < len(degree_counts) else 0

        
        # --- Target Edge Count Check (Stage 4 only) ---
        if stage == 4 and target_degree_sum != -2: # -2 is check for "not set"
             # If removing this edge (degree sum - 2) drops us below target
             if current_total_degree_sum <= target_degree_sum:
                 return False

        u_new_degree = g.degree[u] - 1
        v_new_degree = g.degree[v] - 1
        
        # --- Check 1: Victim 'v' ---
        
        # 1a. Forbidden Degree (target=0)
        # In Python target_counts[d] returns -2 if not set.
        # If set to 0, it means forbidden.
        if v_new_degree >= 0 and self.target_counts[v_new_degree] == 0:
            return False
            
        # 1b. Overshooting
        if v_new_degree >= 0 and self.target_counts[v_new_degree] > 0:
            # Case A: d0/d1 (Sacred)
            if v_new_degree <= 1:
                # Count current
                current_count = count_at(v_new_degree)
                if current_count >= self.target_counts[v_new_degree]:
                    return False
            # Case B: d2+ (Only Stage 4)
            elif stage == 4:
                current_count = count_at(v_new_degree)
                if current_count >= self.target_counts[v_new_degree]:
                    return False
                    
        # --- Check 2: Actor 'u' (Only Stage 4) ---
        if stage == 4:
            # 2a. Forbidden
            if u_new_degree >= 0 and self.target_counts[u_new_degree] == 0:
                return False
                
            # 2b. Overshooting
            if u_new_degree >= 0 and self.target_counts[u_new_degree] > 0:
                 current_count = count_at(u_new_degree)
                 if current_count >= self.target_counts[u_new_degree]:
                     return False
                     
        return True
