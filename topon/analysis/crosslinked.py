"""Read a network whose crosslinks sit along its chains.

In a randomly crosslinked (vulcanised) network, and in the protein networks
built with the bond-fluctuation model, no crosslinker sits at a chain
end: two chain beads somewhere along their chains are bonded to each other.
Read at the bead level, every bead with three or more bonds is a junction
bead, whatever its type. Junction beads bonded to each other make one
junction, the crosslink point network theory counts, so a pairwise crosslink
between two chain beads is one junction of functionality 4 with two chains
passing through it, and two crosslinks on neighbouring beads of one chain
fuse into one junction of functionality 6. ``contract=False`` keeps every
junction bead a node of its own instead, with the bond between two of them an
edge of no beads.

A strand is a maximal run of beads with fewer than three bonds. Its classes
are those of :mod:`topon.analysis.endlinked`:

``bridge``
    both ends on junctions, two different ones
``loop``
    both ends on the same junction (a primary loop: an intra-chain
    crosslink with no other crosslink between its two beads)
``dangling``
    one end on a junction, the other a free chain end
``free``
    neither (a sol chain, or a ring with no junction)

The strand graph is in the convention :mod:`topon.analysis.descriptors`
reads: junctions are nodes with ``kind="junction"``, every dangling strand
adds one ``kind="end"`` node at its free-end bead, strands are edges. Each
edge carries ``cls``, ``dp``, ``chain`` and ``chain_index``:

``dp``
    beads of the strand strictly between its two junction beads; for a
    dangling strand, between its junction bead and its free-end bead, the
    end node being that last bead. This is the ``dp`` the chemistry
    builder reads, whose end-cap node is a bead of its own.
``chain``
    the molecule id of the strand's beads (a tuple when a run crosses
    molecules, which only a crosslink at a chain end makes)
``chain_index``
    the strand's position along its chain, 0 at the chain's first end,
    when the chain's backbone could be walked (see below)

``G.graph["chains"]`` holds, per chain, its bead count ``dp``, its strands
in chain order as ``(u, v, key)`` edges, the junctions it passes in order,
one entry per pass, and ``junction_beads``, its beads inside junctions (one
per pass, except where it passes a fused junction through neighbouring
beads). A chain passes a junction once per visit, so a junction's degree is
two per pass plus one per chain that ends inside it.
The backbone of a molecule is its intra-molecular bonds, less
``crosslink_bond_types`` when given; a molecule whose backbone is not one
simple path (an intra-chain crosslink of the same bond type, a branched
molecule) gets no chain order and is counted in
``G.graph["chains_unordered"]``.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np

from topon.analysis.endlinked import CLASSES, _split_sections, unwrap

#: Bonds a bead needs to be read as a junction bead.
JUNCTION_BONDS = 3


@dataclass
class Strand:
    """One run of beads between junctions (or chain ends)."""

    beads: list                        # bead ids, bonded end first
    ends: tuple                        # (junction or None, junction or None)
    cls: str                           # bridge | loop | dangling | free
    chain: object = None               # molecule id, tuple of ids, or None
    index: Optional[int] = None        # position along the chain
    edge: Optional[tuple] = None       # (u, v, key) in the strand graph

    @property
    def dp(self) -> int:
        """Beads between the junction beads (a free end bead excluded)."""
        n = len(self.beads)
        return n - 1 if self.cls == "dangling" else n


@dataclass
class CrosslinkedSystem:
    """A bead system read down to junctions, strands, chains and the graph."""

    path: Optional[Path]
    box: np.ndarray
    pos: dict                          # bead id -> (3,) position
    mol: dict                          # bead id -> molecule id
    bonds: list                        # (i, j) bead-id pairs
    adj: dict                          # bead id -> bonded bead ids
    junction_beads: list               # beads with JUNCTION_BONDS or more bonds
    junctions: dict                    # junction id -> its beads
    strands: list                      # Strand, in the order they were found
    chains: dict                       # molecule id -> chain record
    graph: nx.MultiGraph
    contract: bool = True
    notes: list = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.pos)

    def class_counts(self) -> dict:
        c = collections.Counter(s.cls for s in self.strands)
        return {k: int(c.get(k, 0)) for k in CLASSES}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def read_crosslinked(path, crosslink_bond_types: Optional[Iterable[int]] = None,
                     contract: bool = True) -> CrosslinkedSystem:
    """Read a LAMMPS data file (``atom_style full``) with mid-chain junctions.

    Atom types are not read: a junction is found from its bonds. One
    molecule per chain is what gives the strands their ``chain``; the bond
    types ``fix bond/create`` gave the crosslinks (``crosslink_bond_types``)
    are what lets a chain carrying an intra-chain crosslink be walked.
    """
    path = Path(path)
    header, sections = _split_sections(
        path.read_text(encoding="utf-8", errors="replace").splitlines())
    lo = np.zeros(3)
    hi = np.zeros(3)
    for line in header:
        p = line.split()
        if len(p) >= 4 and p[2] in ("xlo", "ylo", "zlo"):
            i = "xyz".index(p[2][0])
            lo[i], hi[i] = float(p[0]), float(p[1])
    box = hi - lo
    if not np.all(box > 0):
        raise ValueError(f"{path.name}: no box dimensions found")
    mol, pos = {}, {}
    for line in sections.get("Atoms", []):
        p = line.split()
        a = int(p[0])
        mol[a] = int(p[1])
        pos[a] = np.array([float(p[4]), float(p[5]), float(p[6])])
    bonds, btypes = [], []
    for line in sections.get("Bonds", []):
        p = line.split()
        btypes.append(int(p[1]))
        bonds.append((int(p[2]), int(p[3])))
    return from_beads(pos, mol, bonds, box, bond_types=btypes,
                      crosslink_bond_types=crosslink_bond_types,
                      contract=contract, path=path)


def from_bfm_snapshot(snapshot: dict, n_reactions: Optional[int] = None,
                      contract: bool = True) -> CrosslinkedSystem:
    """A BFM snapshot of :mod:`topon.protein_network.bfm` as a bead system.

    Each chain node is a bead at its lattice site (lattice units, the box
    the lattice), consecutive nodes are bonded, and each reaction bonds its
    two crosslinker nodes. ``n_reactions`` keeps the first that many
    reactions: the crosslinking loop applies its shuffled candidates in the
    order the snapshot lists them, so a prefix is the network the loop had
    built at that point, at any conversion below the snapshot's. The
    snapshot's chains already carry every merge, so in a prefix the beads of
    a later reaction sit on their partner's site, one lattice step from where
    they were; the strand graph does not read those positions.
    """
    Nx, Ny, Nz = (int(snapshot[k]) for k in ("Nx", "Ny", "Nz"))
    chains = snapshot["chains"]
    reactions = snapshot["reactions"]
    if n_reactions is not None:
        if not 0 <= n_reactions <= len(reactions):
            raise ValueError(f"n_reactions {n_reactions} outside 0 to "
                             f"{len(reactions)}")
        reactions = reactions[:n_reactions]
    pos, mol, bid = {}, {}, {}
    nxt = 1
    for ci, chain in enumerate(chains):
        for ni, flat in enumerate(chain):
            z = flat // (Nx * Ny)
            rem = flat % (Nx * Ny)
            pos[nxt] = np.array([rem % Nx, rem // Nx, z], float)
            mol[nxt] = ci + 1
            bid[(ci, ni)] = nxt
            nxt += 1
    bonds, btypes = [], []
    for ci, chain in enumerate(chains):
        for ni in range(1, len(chain)):
            bonds.append((bid[(ci, ni - 1)], bid[(ci, ni)]))
            btypes.append(1)
    for (ci1, ni1), (ci2, ni2) in reactions:
        a, b = bid[(ci1, ni1)], bid[(ci2, ni2)]
        bonds.append((a, b))
        btypes.append(2)
        pos[b] = pos[a].copy()        # the loop merges the pair onto a's site
    sysm = from_beads(pos, mol, bonds, np.array([Nx, Ny, Nz], float),
                      bond_types=btypes, crosslink_bond_types=(2,),
                      contract=contract)
    sysm.graph.graph["source"] = "bfm"
    sysm.graph.graph["n_reactions"] = len(reactions)
    return sysm


def from_beads(pos: dict, mol: dict, bonds: list, box,
               bond_types: Optional[list] = None,
               crosslink_bond_types: Optional[Iterable[int]] = None,
               contract: bool = True,
               path: Optional[Path] = None) -> CrosslinkedSystem:
    """Junctions, strands, chains and the strand graph of a bead system."""
    box = np.asarray(box, float).reshape(3)
    adj = collections.defaultdict(list)
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    jb = sorted(a for a in pos if len(adj.get(a, ())) >= JUNCTION_BONDS)
    jset = set(jb)

    # Junctions: bonded junction beads fused, or each bead its own.
    node_of = {}
    junctions = {}
    for a in jb:
        if a in node_of:
            continue
        if contract:
            members, stack = [], [a]
            node_of[a] = a
            while stack:
                b = stack.pop()
                members.append(b)
                for n in adj[b]:
                    if n in jset and n not in node_of:
                        node_of[n] = a
                        stack.append(n)
        else:
            members = [a]
            node_of[a] = a
        junctions[a] = sorted(members)

    # Strands: runs of beads below the junction bond count.
    run_of = {}
    strands = []
    for a in sorted(pos):
        if a in jset or a in run_of:
            continue
        run = _walk_run(a, adj, jset)
        s = _classify(run, adj, jset, node_of, mol)
        for b in run:
            run_of[b] = len(strands)
        strands.append(s)
    pseudo = {}                           # junction-junction bond -> strand
    if not contract:
        for i, j in bonds:
            if i in jset and j in jset:
                key = (min(i, j), max(i, j))
                pseudo[key] = len(strands)
                # a backbone bond gets its chain from the backbone walk; a
                # crosslink bond belongs to no chain
                strands.append(Strand(beads=[], ends=(node_of[i], node_of[j]),
                                      cls="bridge", chain=None))

    notes = []
    chains, unordered = _order_chains(pos, mol, bonds, bond_types,
                                      crosslink_bond_types, adj, jset,
                                      node_of, run_of, pseudo, strands)
    if unordered:
        notes.append(f"{unordered} molecules have no single backbone path "
                     f"(give crosslink_bond_types for intra-chain "
                     f"crosslinks); their strands carry no chain_index")

    G = _strand_graph(junctions, strands, pos, box)
    for m, rec in chains.items():
        rec["strands"] = [strands[i].edge for i in rec.pop("strand_ids")
                          if strands[i].edge is not None]
    G.graph["box"] = tuple(float(x) for x in box)
    G.graph["architecture"] = "random_crosslinked"
    G.graph["contracted"] = bool(contract)
    G.graph["chains"] = chains
    G.graph["chains_unordered"] = unordered
    return CrosslinkedSystem(path=path, box=box, pos=pos, mol=mol,
                             bonds=list(bonds), adj=dict(adj),
                             junction_beads=jb, junctions=junctions,
                             strands=strands, chains=chains, graph=G,
                             contract=contract, notes=notes)


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _walk_run(a, adj, jset) -> list:
    """The run of non-junction beads through ``a``, end to end.

    A bead below the junction bond count has at most two neighbours, so
    the run is a path, or a ring, which comes back as ``a`` and then the
    beads around it.
    """
    def extend(start):
        out, prev, cur = [], a, start
        while cur != a:
            out.append(cur)
            nxt = [n for n in adj.get(cur, ()) if n not in jset and n != prev]
            if not nxt:
                return out, False
            prev, cur = cur, nxt[0]
        return out, True

    nbrs = [n for n in adj.get(a, ()) if n not in jset]
    if not nbrs:
        return [a]
    left, ring = extend(nbrs[0])
    if ring:
        return [a] + left
    right = extend(nbrs[1])[0] if len(nbrs) > 1 else []
    return left[::-1] + [a] + right


def _classify(run, adj, jset, node_of, mol) -> Strand:
    """A run's two ends and its class; the bonded end first."""
    first, last = run[0], run[-1]
    if len(run) == 1:
        xs = [node_of[n] for n in adj.get(first, ()) if n in jset]
        x1 = xs[0] if xs else None
        x2 = xs[1] if len(xs) > 1 else None
    else:
        x1 = next((node_of[n] for n in adj.get(first, ()) if n in jset), None)
        x2 = next((node_of[n] for n in adj.get(last, ()) if n in jset), None)
    ring = (len(run) > 2 and x1 is None and x2 is None
            and run[-1] in adj.get(run[0], ()))
    if x1 is None and x2 is None:
        cls = "free"
    elif x1 is None or x2 is None:
        cls = "dangling"
        if x1 is None:
            run = run[::-1]
            x1, x2 = x2, x1
    elif x1 == x2:
        cls = "loop"
    else:
        cls = "bridge"
    mols = sorted({mol.get(b) for b in run})
    chain = mols[0] if len(mols) == 1 else tuple(mols)
    s = Strand(beads=list(run), ends=(x1, x2), cls=cls, chain=chain)
    if ring:
        s.cls = "free"
    return s


def _order_chains(pos, mol, bonds, bond_types, crosslink_bond_types, adj,
                  jset, node_of, run_of, pseudo, strands):
    """Walk each molecule's backbone; give its strands their order."""
    xl = set(int(t) for t in (crosslink_bond_types or ()))
    back = collections.defaultdict(list)
    for k, (i, j) in enumerate(bonds):
        if mol.get(i) != mol.get(j):
            continue
        if xl and bond_types is not None and int(bond_types[k]) in xl:
            continue
        back[i].append(j)
        back[j].append(i)
    by_mol = collections.defaultdict(list)
    for a in pos:
        by_mol[mol[a]].append(a)

    chains, unordered = {}, 0
    for m in sorted(by_mol):
        beads = by_mol[m]
        seq = _backbone_path(beads, back)
        if seq is None:
            unordered += 1
            continue
        order, passes = [], []
        for k, b in enumerate(seq):
            if b in jset:
                node = node_of[b]
                prev = seq[k - 1] if k else None
                if prev is not None and prev in jset:
                    if node_of[prev] != node:
                        r = pseudo[(min(prev, b), max(prev, b))]
                        strands[r].chain = m
                        order.append(r)
                        passes.append(node)
                else:
                    passes.append(node)
            else:
                r = run_of[b]
                if not order or order[-1] != r:
                    order.append(r)
        # A strand is listed once per visit; a loop starts and ends at one
        # junction and is walked once.
        seen, uniq = set(), []
        for r in order:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        for idx, r in enumerate(uniq):
            if strands[r].index is None:
                strands[r].index = idx
        chains[m] = {"dp": len(beads), "strand_ids": uniq,
                     "junctions": passes, "passes": len(passes),
                     "junction_beads": sum(1 for b in seq if b in jset)}
    return chains, unordered


def _backbone_path(beads, back) -> Optional[list]:
    """The molecule's beads as one simple path, or None."""
    if len(beads) == 1:
        return list(beads)
    ends = [b for b in beads if len(back.get(b, ())) == 1]
    if len(ends) != 2 or any(len(back.get(b, ())) > 2 for b in beads):
        return None
    start = min(ends)
    seq, prev, cur = [start], None, start
    while True:
        nxt = [n for n in back.get(cur, ()) if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        seq.append(cur)
    return seq if len(seq) == len(beads) else None


def _junction_position(members, pos, box) -> np.ndarray:
    ref = np.asarray(pos[members[0]], float)
    if len(members) == 1:
        return ref
    d = np.array([np.asarray(pos[b], float) - ref for b in members])
    d -= box * np.round(d / box)
    return ref + d.mean(axis=0)


def _strand_graph(junctions, strands, pos, box) -> nx.MultiGraph:
    G = nx.MultiGraph()
    for x, members in junctions.items():
        G.add_node(x, kind="junction", beads=list(members),
                   pos=_junction_position(members, pos, box))
    end_id = max(list(pos) + [0]) + 1
    n_free = 0
    for s in strands:
        x1, x2 = s.ends
        attrs = {"cls": s.cls, "dp": s.dp, "chain": s.chain,
                 "chain_index": s.index}
        if s.cls in ("bridge", "loop"):
            k = G.add_edge(x1, x2, **attrs)
            s.edge = (x1, x2, k)
        elif s.cls == "dangling":
            G.add_node(end_id, kind="end",
                       pos=np.asarray(pos[s.beads[-1]], float))
            k = G.add_edge(x1, end_id, **attrs)
            s.edge = (x1, end_id, k)
            end_id += 1
        else:
            n_free += 1
    G.graph["sol_chains"] = {"count": n_free}
    return G


# ---------------------------------------------------------------------------
# Chain-level numbers
# ---------------------------------------------------------------------------

def chain_statistics(G) -> dict:
    """Passes per chain and strand DP by class, from a strand graph.

    Reads ``G.graph["chains"]`` and the edges' ``dp``: what a crosslinked
    reference and a chain-assigned build are compared on beside the graph
    descriptors. A strand's class is read from the graph (a self-loop, an
    edge to an end node, or a bridge), since a generated graph's edges carry
    no ``cls``. Returns arrays (``passes``, ``dp_bridge``, ``dp_dangling``,
    ``dp_loop``, ``chain_dp``) and their means.
    """
    from topon.analysis.descriptors import node_kind

    chains = G.graph.get("chains") or {}
    passes = np.array([c["passes"] for c in chains.values()], float)
    chain_dp = np.array([c["dp"] for c in chains.values()], float)
    kind = {n: node_kind(G, n) for n in G.nodes()}
    by = collections.defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get("dp") is None:
            continue
        cls = ("loop" if u == v else
               "dangling" if "end" in (kind[u], kind[v]) else "bridge")
        by[cls].append(int(d["dp"]))
    out = {"passes": passes, "chain_dp": chain_dp}
    for cls in ("bridge", "dangling", "loop"):
        out[f"dp_{cls}"] = np.array(by.get(cls, []), float)
    means = {k: (float(v.mean()) if len(v) else None) for k, v in out.items()}
    out["means"] = means
    return out
