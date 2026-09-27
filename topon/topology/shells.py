"""Neighbour shells: the candidate-edge range of a lattice.

Every lattice builder joins sites that lie within ``neighbour_cutoff``
cell units of each other, the cell edge being the simple-cubic site
spacing. At the default of 1.0 the four pure lattices (SC, BCC, FCC,
Diamond) keep their canonical nearest-neighbour pattern; any other value
builds the edges by a minimum-image distance search, the way ``MIX`` does
at every cutoff. The candidate-edge set is therefore a lattice plus a
range, and the range is what lets a sculpted graph connect crosslinkers
several site spacings apart, which is what a reaction-generated network
does (``bond_create_validation/REPORT.md`` section 3.1).

This module holds the arithmetic shared by the config schema, the Python
generator and the C wrapper: the simple-cubic shell table behind
``neighbour_shells`` and the resolver that turns the three config keys
(``neighbour_cutoff``, ``neighbour_shells`` and the deprecated
``mix_cutoff``) into one cutoff. It imports nothing from the rest of the
package so the schema can use it without a cycle.
"""
from __future__ import annotations

import math
from itertools import product

#: The cutoff every lattice builds at when none is given. It is the
#: simple-cubic nearest-neighbour distance and, for the pure lattices,
#: the switch that keeps their canonical neighbour pattern.
DEFAULT_CUTOFF = 1.0

#: The neighbour search admits a pair when ``d^2 <= cutoff^2 + this``,
#: in both generators (the C searcher hard-codes the same value). The
#: shell arithmetic below uses the same rule so that a shell count never
#: disagrees with the edges a builder produces.
SEARCH_TOLERANCE_SQ = 1e-12

#: Nearest-neighbour distance of each pure lattice in cell units. A
#: cutoff below this would leave every site isolated, so the resolver
#: rejects it. ``MIX`` has no single value: its closest contact depends
#: on which basis sites were drawn.
NEAREST_NEIGHBOUR = {
    "SC": 1.0,
    "BCC": math.sqrt(3.0) / 2.0,
    "FCC": math.sqrt(2.0) / 2.0,
    "Diamond": math.sqrt(3.0) / 4.0,
    "DIAMOND": math.sqrt(3.0) / 4.0,
}


# ---------------------------------------------------------------------------
# Simple-cubic shells
# ---------------------------------------------------------------------------

def sc_shell_radii_squared(max_sq: int) -> list[int]:
    """Every squared shell radius of the simple-cubic lattice up to ``max_sq``.

    A simple-cubic shell sits at ``sqrt(n)`` for every ``n >= 1`` that is a
    sum of three squares, so the sequence starts 1, 2, 3, 4, 5, 6, 8, 9
    (7 is missing, as is every ``4^a (8b + 7)``). Enumerated by brute
    force rather than from the number-theoretic rule so the table cannot
    drift from the lattice it describes.
    """
    max_sq = int(max_sq)
    if max_sq < 1:
        return []
    reach = math.isqrt(max_sq)
    found = {
        i * i + j * j + k * k
        for i, j, k in product(range(reach + 1), repeat=3)
    }
    return sorted(n for n in found if 1 <= n <= max_sq)


def _first_shells(n_shells: int) -> list[int]:
    """The squared radii of the first ``n_shells`` simple-cubic shells."""
    if n_shells < 1:
        raise ValueError(f"neighbour_shells must be at least 1, got {n_shells}")
    # Five in six integers are sums of three squares, so 1.3 n + 8 always
    # reaches far enough; the loop only exists for safety.
    max_sq = int(1.3 * n_shells) + 8
    while True:
        radii = sc_shell_radii_squared(max_sq)
        if len(radii) >= n_shells:
            return radii[:n_shells]
        max_sq *= 2


def sc_shell_radius(k: int) -> float:
    """Radius of the ``k``-th simple-cubic shell in cell units (1-based)."""
    return math.sqrt(_first_shells(k)[-1])


def sc_shell_cutoff(k: int) -> float:
    """The ``neighbour_cutoff`` that admits exactly ``k`` simple-cubic shells.

    Reproduces the table the shell sweep was run with
    (``bond_create_validation/scripts/cubic_sweep.py``): 1 -> 1.0,
    2 -> 1.42, 3 -> 1.74, 4 -> 2.01, 5 -> 2.24, 6 -> 2.45, 8 -> 3.01. The
    rule behind it is the shell radius rounded up to two decimals, with a
    0.01 margin on the shells whose radius is a whole number so the value
    sits visibly above the shell rather than exactly on it. The first
    shell is the exception: its 1.0 is the default cutoff and the
    canonical lattice, not a search radius.

    The neighbour search compares ``d <= cutoff`` inclusively, so the
    rounding never changes which shells are admitted; that is checked
    against the next shell before the value is returned.
    """
    radii = _first_shells(k + 1)
    this_sq, next_sq = radii[k - 1], radii[k]
    radius = math.sqrt(this_sq)
    if k == 1:
        cutoff = DEFAULT_CUTOFF
    elif math.isqrt(this_sq) ** 2 == this_sq:
        cutoff = float(math.isqrt(this_sq)) + 0.01
    else:
        cutoff = math.ceil(radius * 100.0 - 1e-9) / 100.0
    if cutoff * cutoff >= next_sq:
        raise ValueError(
            f"shell {k} (radius {radius:.4f}) is too close to shell {k + 1} "
            f"(radius {math.sqrt(next_sq):.4f}) for a two-decimal cutoff; "
            f"give neighbour_cutoff explicitly"
        )
    return cutoff


def sc_shells_within(cutoff: float) -> int:
    """How many simple-cubic shells a cutoff admits.

    Same inclusive rule as the neighbour search
    (``n <= cutoff^2 + SEARCH_TOLERANCE_SQ``), so the count matches the
    edges a builder produces even for a cutoff exactly on a shell.
    """
    cutoff = float(cutoff)
    if cutoff < 1.0:
        return 0
    limit = cutoff * cutoff + SEARCH_TOLERANCE_SQ
    return sum(1 for n in sc_shell_radii_squared(int(limit) + 1) if n <= limit)


def sc_coordination(cutoff: float) -> int:
    """Neighbours per site of a periodic simple-cubic lattice at ``cutoff``.

    Counts the integer vectors with ``0 < |v|^2 <= cutoff^2``, so it is
    exact only when the box is wider than twice the cutoff. Handy for
    reading a cutoff as a coordination number: 1.0 -> 6, 1.42 -> 18,
    1.74 -> 26, 2.01 -> 32, 2.24 -> 56, 2.45 -> 80, 3.01 -> 122.
    """
    cutoff = float(cutoff)
    limit = cutoff * cutoff + SEARCH_TOLERANCE_SQ
    reach = int(math.floor(cutoff + 1e-9))
    return sum(
        1
        for i, j, k in product(range(-reach, reach + 1), repeat=3)
        if 0 < i * i + j * j + k * k <= limit
    )


# ---------------------------------------------------------------------------
# Resolving the config keys
# ---------------------------------------------------------------------------

def resolve_neighbour_cutoff(
    lattice_type: str,
    neighbour_cutoff: float | None = None,
    neighbour_shells: int | None = None,
    mix_cutoff: float | None = None,
) -> float:
    """Turn the config keys into the one cutoff a builder uses.

    ``neighbour_cutoff`` wins when given; the deprecated ``mix_cutoff`` is
    read in its place otherwise (the caller decides whether to warn),
    except that at its default of 1.0 it counts as not given when
    ``neighbour_shells`` is present, so the many configs that carry
    ``mix_cutoff = 1.0`` can still name a shell count.
    ``neighbour_shells`` is a shell count and maps to a cutoff on SC only,
    where the shells are the table in :func:`sc_shell_cutoff`; on any
    other lattice it is accepted only alongside an explicit cutoff, since
    the same count means a different range on each lattice. Given both on
    SC, the cutoff has to admit exactly that many shells.

    Raises ``ValueError`` for a non-positive cutoff, an inconsistent
    shell count, a shell count without a cutoff on a non-SC lattice, or a
    cutoff below the lattice's nearest-neighbour distance (which would
    leave every site isolated).
    """
    cutoff = neighbour_cutoff
    if cutoff is None and mix_cutoff is not None and not (
            neighbour_shells is not None and float(mix_cutoff) == DEFAULT_CUTOFF):
        cutoff = mix_cutoff
    if cutoff is not None:
        cutoff = float(cutoff)
        if not math.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                f"neighbour_cutoff must be a positive number of cell units, "
                f"got {cutoff!r}"
            )

    if neighbour_shells is not None:
        shells = int(neighbour_shells)
        if shells != neighbour_shells or shells < 1:
            raise ValueError(
                f"neighbour_shells must be a whole number of at least 1, "
                f"got {neighbour_shells!r}"
            )
        if lattice_type == "SC":
            if cutoff is None:
                cutoff = sc_shell_cutoff(shells)
            else:
                admitted = sc_shells_within(cutoff)
                if admitted != shells:
                    raise ValueError(
                        f"neighbour_cutoff {cutoff:g} admits {admitted} SC "
                        f"shell(s) but neighbour_shells is {shells}; give one "
                        f"of the two, or a cutoff between "
                        f"{sc_shell_radius(shells):.4g} and "
                        f"{sc_shell_radius(shells + 1):.4g}"
                    )
        elif cutoff is None:
            raise ValueError(
                f"neighbour_shells maps to a cutoff on SC only (the shell "
                f"table is simple-cubic); for lattice_type {lattice_type!r} "
                f"give neighbour_cutoff in cell units instead"
            )

    if cutoff is None:
        cutoff = DEFAULT_CUTOFF

    nearest = NEAREST_NEIGHBOUR.get(lattice_type)
    if nearest is not None and cutoff < nearest - 1e-9:
        raise ValueError(
            f"neighbour_cutoff {cutoff:g} is below the {lattice_type} "
            f"nearest-neighbour distance of {nearest:.4g} cell units, so no "
            f"site would have a neighbour"
        )
    return cutoff
