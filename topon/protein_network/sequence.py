"""Sequence-to-BFM-node mapping for arbitrary residue blocks.

Taken from the sequence module of the earlier protein-network code topon grew
from, generalized to any one-letter block by inferring the crosslinker position from
the block string itself (uses the first 'Y' by default, configurable via
`crosslinker_letter`). The 3-letter mapping comes from this package's residues
module rather than being duplicated.

The mapping function returns ``{node_idx: residue_idx}`` so callers can ask
"which residue does this BFM lattice node represent?". Anchors are fixed first
(chain ends + crosslinker positions); intermediate NC nodes are linearly
interpolated between adjacent anchor pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .residues import ONE_TO_THREE, REFERENCE_BLOCK


DEFAULT_BLOCK_SEQ: str = REFERENCE_BLOCK
DEFAULT_CROSSLINKER_LETTER: str = "Y"
DEFAULT_Y_IN_BLOCK: int = REFERENCE_BLOCK.index("Y")  # 7 for resilin GGRPSDSYGAPGGGN


def get_node_residue_mapping(
    n_repeats: int,
    segs_per_block: int,
    y_offset_in_block: int = 0,
    block_seq: str | None = None,
    block_size: int | None = None,
    y_in_block: int | None = None,
    crosslinker_letter: str = DEFAULT_CROSSLINKER_LETTER,
) -> dict[int, int]:
    """Map every BFM chain node index to a residue index.

    Anchors:
      * End nodes  -> residues 0 and (n_repeats * block_size - 1)
      * Y nodes    -> the crosslinker residue within each repeat block

    Intermediate (NC) nodes are linearly interpolated between adjacent anchors.
    """
    if block_seq is not None:
        block_size = len(block_seq)
        if crosslinker_letter in block_seq:
            y_in_block = block_seq.index(crosslinker_letter)
        else:
            y_in_block = block_size // 2
    else:
        if block_size is None:
            block_seq = DEFAULT_BLOCK_SEQ
            block_size = len(DEFAULT_BLOCK_SEQ)
        if y_in_block is None:
            y_in_block = DEFAULT_Y_IN_BLOCK

    n_nodes = n_repeats * segs_per_block + 1
    n_residues = n_repeats * block_size

    mapping: dict[int, int] = {}
    mapping[0] = 0
    mapping[n_nodes - 1] = n_residues - 1
    for k in range(n_repeats):
        y_node = k * segs_per_block + y_offset_in_block + 1
        y_res = k * block_size + y_in_block
        mapping[y_node] = y_res

    sorted_nodes = sorted(mapping.keys())
    for i in range(len(sorted_nodes) - 1):
        n_start = sorted_nodes[i]
        n_end = sorted_nodes[i + 1]
        r_start = mapping[n_start]
        r_end = mapping[n_end]
        span = n_end - n_start
        for j in range(1, span):
            nc_node = n_start + j
            if nc_node not in mapping:
                t = j / span
                mapping[nc_node] = round(r_start + t * (r_end - r_start))
    return mapping


def build_full_sequence(block_seq: str, n_repeats: int) -> list[str]:
    """Tile a one-letter repeat block n_repeats times and convert to 3-letter codes."""
    full_1l = block_seq * n_repeats
    out: list[str] = []
    for aa in full_1l:
        code = ONE_TO_THREE.get(aa.upper())
        if code is None:
            raise ValueError(f"Unknown one-letter amino acid code: {aa!r}")
        out.append(code)
    return out


def get_tyr_node_indices(n_repeats: int, segs_per_block: int, y_offset_in_block: int = 0) -> set[int]:
    """Return the set of chain node indices that are crosslinker positions."""
    return {k * segs_per_block + y_offset_in_block + 1 for k in range(n_repeats)}


# ── Sequence-driven chain layout ────────────────────────────────────────────

STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class ChainPlan:
    """How one chain of a given sequence sits on the BFM lattice.

    ``n_nodes`` lattice sites per chain, ``y_positions`` the node indices that
    carry a crosslinkable residue, and ``node_to_res`` the residue each anchor
    node stands for (every other residue is interpolated between anchors).
    ``layout`` is ``"block"`` when the chain is a repeat of a block with one
    crosslink residue each (the layout topon has always used, kept so those
    builds do not change) and ``"anchors"`` otherwise.
    """
    sequence: str
    residues: tuple[str, ...]
    crosslink_residue: str
    n_nodes: int
    y_positions: tuple[int, ...]
    node_to_res: dict
    segs_per_block: int
    y_offset_in_block: int | None
    layout: str
    skipped_terminal_sites: tuple[int, ...] = ()

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def crosslink_residue_indices(self) -> tuple[int, ...]:
        return tuple(self.node_to_res[n] for n in self.y_positions)


def validate_sequence(seq: str) -> str:
    """Upper-case a one-letter sequence and refuse anything non-standard."""
    s = "".join(seq.split()).upper()
    bad = sorted({c for c in s if c not in STANDARD_RESIDUES})
    if not s:
        raise ValueError("empty sequence")
    if bad:
        raise ValueError(f"unknown one-letter residue code(s) {bad}; "
                         f"use the 20 standard amino acids {STANDARD_RESIDUES}")
    return s


def plan_chain(
    sequence: str,
    repeats: int = 1,
    crosslink_residue: str = "Y",
    segs_per_block: int = 2,
    residues_per_segment: float | None = None,
) -> ChainPlan:
    """Lay a chain of ``sequence * repeats`` on the lattice.

    Every occurrence of ``crosslink_residue`` becomes a crosslinkable lattice
    node, except at the two chain ends, which are lattice ends already (they
    are listed in ``skipped_terminal_sites``).

    A block with exactly one crosslink residue, not at either end of the
    block, keeps the layout the BFM stage has always used (``segs_per_block``
    lattice steps per block, the crosslink node at step 1 of each block), so
    existing builds are unchanged. Anything else (no crosslink residue, two
    in a block, one at a block end, a non-repeating chain) uses anchors: the
    chain ends and each crosslink residue get their own node, and the gap
    between two anchors gets ``round(gap / residues_per_segment)`` lattice
    steps (at least one). ``residues_per_segment`` defaults to
    ``len(sequence) / segs_per_block``, the same density as the block layout.
    """
    block = validate_sequence(sequence)
    x = crosslink_residue.upper()
    if len(x) != 1 or x not in STANDARD_RESIDUES:
        raise ValueError(f"crosslink residue must be one standard letter, got {crosslink_residue!r}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    full = block * repeats
    residues = tuple(ONE_TO_THREE[c] for c in full)

    in_block = [i for i, c in enumerate(block) if c == x]
    if len(in_block) == 1 and 0 < in_block[0] < len(block) - 1 and segs_per_block >= 2:
        y_off = 0 if segs_per_block <= 2 else 1
        n_nodes = repeats * segs_per_block + 1
        y_pos = tuple(k * segs_per_block + y_off + 1 for k in range(repeats))
        n2r = get_node_residue_mapping(repeats, segs_per_block, y_offset_in_block=y_off,
                                       block_seq=block, crosslinker_letter=x)
        return ChainPlan(full, residues, x, n_nodes, y_pos, n2r, segs_per_block,
                         y_off, "block")

    rps = residues_per_segment or (len(block) / segs_per_block)
    if rps <= 0:
        raise ValueError("residues_per_segment must be positive")
    n = len(full)
    if n < 2:
        raise ValueError("a chain needs at least two residues")
    sites = [i for i, c in enumerate(full) if c == x]
    skipped = tuple(i for i in sites if i in (0, n - 1))
    sites = [i for i in sites if 0 < i < n - 1]
    anchors = [0] + sites + [n - 1]
    n2r = {0: 0}
    node = 0
    y_pos = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        nseg = min(b - a, max(1, int(round((b - a) / rps))))
        for j in range(1, nseg + 1):
            node += 1
            n2r[node] = b if j == nseg else int(round(a + j * (b - a) / nseg))
        if b in sites:
            y_pos.append(node)
    return ChainPlan(full, residues, x, node + 1, tuple(y_pos), n2r, segs_per_block,
                     None, "anchors", skipped)
