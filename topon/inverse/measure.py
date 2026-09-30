"""Step 1 of ``topon fit``: read a reference network and measure it.

A reference comes in one of three forms, and each carries a different amount
of information:

``data``
    a LAMMPS data file of an end-linked network (``fix bond/create`` output,
    or topon's own ``endlinked`` convention), read with
    :func:`topon.analysis.endlinked.read_endlinked`. Everything is there:
    strands and their classes, bead counts, coordinates, and with Z1+
    installed the primitive-path numbers.
``npz``
    a dual graph (chains and crosslinkers as nodes), the collaborator's
    schema or topon's own. Connectivity and DP are there. The dual graph
    records a chain's crosslinkers without saying how many of its ends bond
    each one, so a primary loop (both ends on one crosslinker) and a
    dangling chain (one end bonded) read the same; both are read as
    dangling and the record says how many there are. Crosslinker positions
    are used when the file has them (the collaborator's does, topon's own
    writes NaN), and there is no Z target in any NPZ.
``graph``
    a strand graph (``.gpickle``, ``.graphml``, ``.nodes``/``.edges``) in the
    convention of :mod:`topon.analysis.descriptors`. DP comes from the edges'
    ``dp`` when they carry one, and density is not known.

:func:`measure` turns a :class:`Reference` into the numbers a fitted config
has to reproduce: strand classes, effective and chemical P(f), the loops with
the degrees of the junctions they sit on, the sculpt target in topon's site
convention, DP, density, the connectivity descriptors of
:mod:`topon.analysis.descriptors`, the junction
separations and, from a data file, Z1+ per strand class.

What topon's lattice route cannot hold is counted here too, because it is a
property of the reference: a junction whose only strands are primary loops
(effective degree 0) is a vacancy to the sculptor, and one with a single
other strand (effective degree 1) is indistinguishable from a dangling-chain
end, so the loops those junctions carry land elsewhere and the chemical P(f)
moves on that many junctions (8 and 8 of 2,500 on the DP-20 reference).
"""
from __future__ import annotations

import collections
import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from topon.analysis.descriptors import (
    Descriptors, describe, graph_box, network_view, spatial)

DATA_SUFFIXES = (".data", ".lmp", ".lammps")
NPZ_SUFFIXES = (".npz",)
GRAPH_SUFFIXES = (".gpickle", ".graphml", ".nodes", ".edges")


@dataclass
class Reference:
    """A reference network read into the strand-graph convention."""

    path: Path
    source: str                       # "data" | "npz" | "graph"
    graph: nx.MultiGraph
    box: Optional[np.ndarray]         # in the units of the node positions
    units: str                        # "sigma" or "lattice"
    system: Optional[object] = None   # EndLinkedSystem, for a data file
    chain_dp: dict = field(default_factory=dict)   # class -> [DP, ...]
    n_atoms: Optional[int] = None
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def has_positions(self) -> bool:
        """Every junction has a finite position and the cell is known."""
        if self.box is None or not np.all(np.isfinite(self.box)):
            return False
        for _, d in self.graph.nodes(data=True):
            if d.get("kind") == "end":
                continue
            p = d.get("pos")
            if p is None or not np.all(np.isfinite(np.asarray(p, float))):
                return False
        return True

    @property
    def density(self) -> Optional[float]:
        """Beads per volume, when the bead count and the cell are known."""
        if self.n_atoms is None or self.box is None or self.units != "sigma":
            return None
        return float(self.n_atoms / np.prod(self.box))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_reference(path, junction_type: Optional[int] = None,
                   nodes: Optional[str] = None) -> Reference:
    """Read ``path`` as a data file, an NPZ dual graph or a strand graph.

    ``junction_type`` names the junction atom type of a data file that does
    not follow the end-linked typing (see
    :func:`topon.analysis.endlinked.read_endlinked`); ``nodes`` is the
    companion ``.nodes`` of a bare ``.edges`` file.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in DATA_SUFFIXES:
        return _from_data(p, junction_type)
    if junction_type is not None:
        raise ValueError(f"a junction type applies to a LAMMPS data file, not "
                         f"to {p.name}")
    if suffix in NPZ_SUFFIXES:
        return read_dual_npz(p)
    if suffix in GRAPH_SUFFIXES:
        return _from_graph(p, nodes)
    raise ValueError(
        f"cannot read {p.name} as a reference: use a LAMMPS data file "
        f"({', '.join(DATA_SUFFIXES)}), an NPZ dual graph (.npz) or a strand "
        f"graph ({', '.join(GRAPH_SUFFIXES)})")


def _from_data(path: Path, junction_type: Optional[int]) -> Reference:
    from topon.analysis.endlinked import read_endlinked

    system = read_endlinked(path, junction_type=junction_type)
    chain_dp = collections.defaultdict(list)
    for s in system.strands.values():
        chain_dp[s.cls].append(len(s.seq))
    return Reference(path=path, source="data", graph=system.graph,
                     box=np.asarray(system.box, float), units="sigma",
                     system=system, chain_dp=dict(chain_dp),
                     n_atoms=system.n_atoms)


def read_dual_npz(path) -> Reference:
    """An NPZ dual graph as a strand graph.

    Every chain row becomes a strand by the distinct crosslinker rows its
    chemical edges reach, and the file is read one of two ways.

    A file with crosslinker coordinates (the collaborator's, schema 1,
    upgraded by :func:`topon.topology.loader.read_npz`) is a reaction
    network: crosslinker rows are junctions at their positions, the box is
    in sigma, and a chain reaching two crosslinkers is a bridge, one a
    dangling chain, none a sol chain. A primary loop also reaches one
    crosslinker, so it reads as dangling; the count is in
    ``Reference.extra["one_crosslinker_chains"]`` and a note says why.

    A schema-2 file with no coordinates is what topon's own writer makes
    (the reading :func:`~topon.topology.loader.load_npz` takes for NaN
    coordinates too): it writes every graph node as a crosslinker row,
    dangling-chain end sites included, and the lattice cell as the box. So
    a crosslinker row with one chain is an end site, a chain reaching one
    crosslinker can only be a primary loop, a dangling chain's DP is its
    length plus its end-site bead, and there is no bead box and no density.
    topon's writer records no sol chains.

    Raises:
        ValueError: a chain row joined to more than two crosslinkers, which
            is not an end-linked strand.
    """
    from topon.topology.loader import read_npz
    from topon.writers.npz_writer import _FEATURE_COLUMNS

    path = Path(path)
    a = read_npz(path)
    col = {name: i for i, name in enumerate(_FEATURE_COLUMNS)}
    nf = np.asarray(a["node_features"], float)
    ids = np.asarray(a["node_ids"]).astype(int)
    kind = nf[:, col["type"]]
    ei = np.asarray(a["edge_index"]).astype(int)
    et = np.asarray(a["edge_type"]).astype(int)

    xl_rows = np.where(kind == 1)[0]
    chain_rows = np.where(kind == 0)[0]
    reach = collections.defaultdict(set)
    chains_of = collections.Counter()
    for k in np.where(et == 0)[0]:
        s, t = ei[0, k], ei[1, k]          # chain row s, crosslinker row t
        if kind[s] == 1 and kind[t] == 0:
            s, t = t, s
        elif not (kind[s] == 0 and kind[t] == 1):
            continue
        if t not in reach[s]:
            reach[s].add(t)
            chains_of[t] += 1
    com = nf[xl_rows][:, [col["COMX"], col["COMY"], col["COMZ"]]]
    upgraded = "upgraded_from" in a
    written_by_topon = (not upgraded and len(xl_rows) > 0
                        and not np.isfinite(com).any())
    ends = ({int(r) for r in xl_rows if chains_of[r] == 1}
            if written_by_topon else set())

    G = nx.MultiGraph()
    for r in xl_rows:
        xyz = nf[r, [col["COMX"], col["COMY"], col["COMZ"]]]
        G.add_node(int(ids[r]), kind="end" if int(r) in ends else "junction",
                   pos=np.asarray(xyz, float))
    end_id = int(ids.max()) + 1
    chain_dp = collections.defaultdict(list)
    n_sol = one = 0
    for r in chain_rows:
        xs = sorted(reach.get(r, ()))
        dp = int(round(nf[r, col["length"]]))
        cid = int(ids[r])
        if len(xs) > 2:
            raise ValueError(
                f"{path.name}: chain {cid} is joined to {len(xs)} crosslinkers; "
                f"an end-linked strand reaches at most two")
        if not xs:
            n_sol += 1
            chain_dp["free"].append(dp)
        elif len(xs) == 1 and written_by_topon:
            one += 1
            G.add_edge(int(ids[xs[0]]), int(ids[xs[0]]), chain=cid,
                       cls="loop", dp=dp)
            chain_dp["loop"].append(dp)
        elif len(xs) == 1:
            one += 1
            G.add_node(end_id, kind="end", pos=np.full(3, np.nan))
            G.add_edge(int(ids[xs[0]]), end_id, chain=cid, cls="dangling",
                       dp=dp)
            chain_dp["dangling"].append(dp)
            end_id += 1
        else:
            free_end = [x for x in xs if int(x) in ends]
            cls = "dangling" if len(free_end) == 1 else "bridge"
            G.add_edge(int(ids[xs[0]]), int(ids[xs[1]]), chain=cid, cls=cls,
                       dp=dp)
            # The end site is the dangling chain's last bead.
            chain_dp[cls].append(dp + 1 if cls == "dangling" else dp)
    G.graph["sol_chains"] = {"count": n_sol}

    box = None
    b = np.asarray(a.get("box", np.full(6, np.nan)), float).ravel()
    if b.size == 6 and np.all(np.isfinite(b)):
        box = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        G.graph["box"] = tuple(float(x) for x in box)
    n_atoms = int(sum(sum(v) for v in chain_dp.values())
                  + len(xl_rows) - len(ends))
    n_ent = int((et == 1).sum() // 2)
    ref = Reference(path=path, source="npz", graph=G, box=box,
                    units="lattice" if written_by_topon else "sigma",
                    chain_dp=dict(chain_dp),
                    n_atoms=None if written_by_topon else n_atoms,
                    extra={"one_crosslinker_chains": one,
                           "entanglement_pairs": n_ent,
                           "written_by_topon": written_by_topon,
                           "schema_upgraded_from": int(a["upgraded_from"])
                           if upgraded else None})
    if written_by_topon:
        ref.notes.append(
            f"Read as topon's own NPZ (schema 2, no coordinates): its "
            f"{len(ends)} one-chain crosslinker rows are dangling-chain end "
            f"sites and its {one} one-crosslinker chains primary loops. Its "
            f"box is the lattice cell, so there is no bead density (pass "
            f"--density), and topon's writer records no sol chains.")
    else:
        ref.notes.append(
            f"{one} chains reach one crosslinker. An NPZ dual graph cannot "
            f"tell a primary loop (both ends on that crosslinker) from a "
            f"dangling chain (one end bonded), so all of them are read as "
            f"dangling and no primary loops are fitted.")
    ref.notes.append(
        "An NPZ carries no primitive-path data, so there is no Z1+ target"
        + (f" ({n_ent} entanglement edges are recorded, as partner pairs, "
           f"not as Z per strand)." if n_ent else "."))
    if not ref.has_positions:
        ref.notes.append("The NPZ has no crosslinker coordinates, so there "
                         "are no spatial targets and the cutoff comes from "
                         "the sweep alone.")
    return ref


def _from_graph(path: Path, nodes: Optional[str]) -> Reference:
    from topon.analysis.analyze import load_network

    with contextlib.redirect_stdout(sys.stderr):
        G, box, _system = load_network(path, nodes)
    V = network_view(G)
    chain_dp = collections.defaultdict(list)
    kind = {n: d["kind"] for n, d in V.nodes(data=True)}
    missing = 0
    for u, v, d in V.edges(data=True):
        cls = ("loop" if u == v else
               "dangling" if "end" in (kind[u], kind[v]) else "bridge")
        if d.get("dp") is None:
            missing += 1
            continue
        chain_dp[cls].append(int(d["dp"]))
    units = "sigma" if G.graph.get("box_sigma") is not None else "lattice"
    ref = Reference(path=path, source="graph", graph=V,
                    box=None if box is None else np.asarray(box, float),
                    units=units, chain_dp=dict(chain_dp))
    if missing:
        ref.notes.append(f"{missing} strands carry no dp; DP comes from the "
                         f"strands that do, or from --dp.")
    ref.notes.append("A graph file carries no bead count, so the density "
                     "comes from --density.")
    return ref


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    """What :func:`measure` found, as plain numbers and as distributions."""

    record: dict
    descriptors: Descriptors
    z_per_strand: dict = field(default_factory=dict)   # class -> Z array
    chords: Optional[np.ndarray] = None


def _hist(values) -> dict:
    c = collections.Counter(int(v) for v in values)
    return {int(k): int(c[k]) for k in sorted(c)}


def effective_degrees(V) -> dict:
    """Strands leaving each junction for somewhere else (loops left out)."""
    return {n: sum(1 for u, v in V.edges(n) if u != v)
            for n, d in V.nodes(data=True) if d.get("kind") == "junction"}


def secondary_endpoint_degrees(V, eff=None) -> dict:
    """Parallel strands keyed ``"a,b"`` by their junctions' effective degrees.

    The layout of ``assignment.defects.secondary_loops.endpoint_degrees``:
    a pair of junctions joined by ``c`` strands contributes ``c - 1`` under
    the key of its two effective degrees, smaller first, the parallel
    strands counted in those degrees. On the DP-20 reference this gives
    ``{"4,4": 115, "3,4": 6, "2,4": 7, "2,3": 1}``.
    """
    eff = effective_degrees(V) if eff is None else eff
    mult = collections.Counter(
        (u, v) if str(u) <= str(v) else (v, u)
        for u, v in V.edges() if u != v)
    out = collections.Counter()
    for (u, v), c in mult.items():
        if c > 1:
            a, b = sorted((eff.get(u, 0), eff.get(v, 0)))
            out[f"{a},{b}"] += c - 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def sculpt_target(V, eff=None) -> dict:
    """The reference's P(f) in topon's site convention.

    Degree 1 counts the dangling-chain ends and the junctions left with one
    other strand; degrees 2 and up are the junctions' effective degrees;
    degree 0 is the junctions with no strand but loops, which the sculptor
    leaves empty. The vacancies of a cell are added by the scaffold step.
    """
    eff = effective_degrees(V) if eff is None else eff
    hist = collections.Counter(eff.values())
    n_end = sum(1 for _, d in V.nodes(data=True) if d.get("kind") == "end")
    top = max(list(hist) + [1])
    out = {d: int(hist.get(d, 0)) for d in range(0, top + 1)}
    out[1] = out.get(1, 0) + n_end
    return out


def dp_summary(chain_dp: dict) -> Optional[dict]:
    """Mean, polydispersity and histogram of the chain DPs."""
    every = [dp for v in chain_dp.values() for dp in v]
    if not every:
        return None
    x = np.asarray(every, float)
    out = {"mean": float(x.mean()),
           "pdi": float((x * x).mean() / x.mean() ** 2),
           "min": int(x.min()), "max": int(x.max()),
           "hist": _hist(x), "n_chains": int(len(x)),
           "by_class": {cls: {"n": len(v), "mean": float(np.mean(v))}
                        for cls, v in sorted(chain_dp.items()) if v}}
    return out


def measure(ref: Reference, heavy: bool = True, z1: Optional[bool] = None,
            z1_config=None, seed: int = 0) -> Measurement:
    """Everything a fitted config is built from, for one reference.

    ``z1`` None runs Z1+ on a data file when it is installed, True requires
    it, False skips it; a graph or NPZ never has it. Descriptors use
    ``seed`` for their sampled parts, so a reference always measures the
    same.
    """
    V = network_view(ref.graph)
    eff = effective_degrees(V)
    desc = describe(ref.graph, heavy=heavy, seed=seed)
    s = desc.scalars

    loops_on = collections.Counter()
    loops_by_eff = collections.Counter()
    for n in eff:
        k = sum(1 for u, v in V.edges(n) if u == v)
        if k:
            loops_on[n] = k
            loops_by_eff[eff[n]] += k
    n_primary = int(sum(loops_on.values()))
    chains = {"bridge": int(s.get("n_bridging_chains", 0)),
              "loop": n_primary,
              "dangling": int(s.get("n_dangling_chains", 0)),
              "free": int(s.get("n_sol_chains", 0))}

    target = sculpt_target(V, eff)
    eff0 = [n for n, e in eff.items() if e == 0]
    eff1 = [n for n, e in eff.items() if e == 1]
    rec = {
        "input": str(ref.path),
        "source": ref.source,
        "units": ref.units,
        "box": None if ref.box is None else [float(x) for x in ref.box],
        "n_atoms": ref.n_atoms,
        "density": ref.density,
        "chains": chains,
        "junctions": len(eff),
        "pf_effective": _hist(eff.values()),
        "pf_chemical": dict(s.get("deg_chem_dist", {})),
        "max_functionality": int(max(s.get("deg_chem_dist", {1: 0}))),
        "target": target,
        "n_active_sites": int(sum(v for d, v in target.items() if d >= 1)),
        "n_junction_sites": int(sum(1 for e in eff.values() if e >= 1)),
        "secondary_loops": {
            "count": int(s.get("n_secondary_loops", 0)),
            "endpoint_degrees": secondary_endpoint_degrees(V, eff)},
        "primary_loops": {
            "count": n_primary,
            "by_effective_degree": {int(k): int(v)
                                    for k, v in sorted(loops_by_eff.items())}},
        "sol_chains": chains["free"],
        "not_representable": {
            "loop_only_junctions": len(eff0),
            "loops_on_them": int(sum(loops_on[n] for n in eff0)),
            "one_strand_junctions": len(eff1),
            "loops_on_those": int(sum(loops_on[n] for n in eff1))},
        "giant_fraction": s.get("giant_frac_junctions"),
        "dp": dp_summary(ref.chain_dp),
        "descriptors": s,
        "notes": list(ref.notes),
    }
    if ref.extra:
        rec["extra"] = dict(ref.extra)

    chords = None
    if ref.has_positions:
        try:
            sp, chords = spatial(ref.graph, ref.box)
            sp["units"] = ref.units
            rec["spatial"] = sp
        except ValueError:
            pass
    if ref.system is not None:
        from topon.analysis.endlinked import chain_statistics
        stats = chain_statistics(ref.system)[0]
        rec.setdefault("spatial", {})
        for key in ("chain_ree_mean", "temperature", "bond_mean", "bond_max"):
            if stats.get(key) is not None:
                rec["spatial"][key] = stats[key]

    zmap: dict = {}
    if ref.system is not None and z1 is not False:
        from topon.analysis.z1plus import (
            Z1PlusFailed, Z1PlusUnavailable, measure_system, why_unavailable)
        reason = why_unavailable(z1_config)
        if reason is None:
            try:
                zsum, res = measure_system(ref.system, config=z1_config)
                rec["z1"] = zsum
                for cls in set(res.classes.tolist()):
                    zmap[cls] = res.Z[res.classes == cls]
            except (Z1PlusUnavailable, Z1PlusFailed) as exc:
                if z1:
                    raise
                rec["z1_note"] = f"Z1+ failed: {exc}"
        elif z1:
            raise RuntimeError(f"Z1+ was asked for and cannot run: {reason}")
        else:
            rec["z1_note"] = f"Z1+ not run: {reason}"
    elif ref.system is None and ref.source != "npz":     # an NPZ says so in its notes
        rec["z1_note"] = ("no Z1+ target: a graph file has no chain paths "
                          "to measure")
    return Measurement(record=rec, descriptors=desc, z_per_strand=zmap,
                       chords=chords)


def z_target(meas: Measurement, cls: str = "bridge") -> Optional[dict]:
    """``target_Z`` and ``target_hist`` from a reference's Z1+ per strand.

    Per bridge, which is what the entanglement controller closes on
    (:func:`topon.analysis.z1plus.bridge_z`). The histogram is rounded to
    four decimals and its largest bin absorbs the rounding, so it sums to 1
    the way ``conformation.entanglement.target_hist`` requires and no bin
    goes below zero.
    """
    z = meas.z_per_strand.get(cls)
    if z is None or not len(z):
        return None
    counts = np.bincount(np.asarray(z, int))
    frac = [round(float(c) / len(z), 4) for c in counts]
    top = int(np.argmax(counts))
    frac[top] = round(frac[top] + 1.0 - sum(frac), 4)
    return {"target_Z": round(float(np.mean(z)), 4), "target_hist": frac,
            "n": int(len(z)), "class": cls}
