"""`topon analyze`: descriptors, Z1+ and a comparison, for one network.

The input is a strand graph (``.gpickle``, ``.nodes``/``.edges``,
``.graphml``) or a LAMMPS data file in the end-linked convention, from which
the strand graph is read (see :mod:`topon.analysis.endlinked`). A data file
also gives the chain statistics and, with Z1+ installed, the primitive-path
numbers. :func:`analyze_network` returns everything as one dict and
:func:`format_report` renders it for the terminal.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from topon.analysis.descriptors import (
    Descriptors, compare, describe, graph_box, spatial, to_jsonable)

GRAPH_SUFFIXES = (".gpickle", ".nodes", ".edges", ".graphml")
DATA_SUFFIXES = (".data", ".lmp", ".lammps")


def load_network(path, nodes: Optional[str] = None, strands=None):
    """Read a graph, an end-linked data file or an atomistic one.

    Returns ``(G, box, system)``; ``system`` is the
    :class:`~topon.analysis.endlinked.EndLinkedSystem` for a data file and
    None for a graph. An atomistic data file is read through its run's strand
    record (:mod:`topon.analysis.atomistic`): ``strands`` names the manifest,
    and without it one is looked for beside the file. The graph loader's
    progress lines go to stderr, so a JSON report on stdout stays parseable.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in DATA_SUFFIXES:
        from topon.analysis.atomistic import (
            NoStrandRecord, find_manifest, read_atomistic)
        if strands is not None:
            system = read_atomistic(p, strands)
            return system.graph, system.box, system
        if find_manifest(p) is not None:
            try:
                system = read_atomistic(p)
                return system.graph, system.box, system
            except NoStrandRecord:
                pass        # a record left by another run; try the file's own types
        from topon.analysis.endlinked import read_endlinked
        system = read_endlinked(p)
        return system.graph, system.box, system

    from topon.topology.loader import load_graph, load_graphml
    with contextlib.redirect_stdout(sys.stderr):
        if suffix == ".gpickle":
            G, dims = load_graph(gpickle_path=p)
        elif suffix == ".graphml":
            G, dims = load_graphml(p)
        elif suffix == ".nodes":
            edges = p.with_suffix(".edges")
            if not edges.exists():
                raise FileNotFoundError(f"companion .edges file not found: {edges}")
            G, dims = load_graph(nodes_path=p, edges_path=edges)
        elif suffix == ".edges":
            nodes_path = Path(nodes) if nodes else p.with_suffix(".nodes")
            if not nodes_path.exists():
                raise FileNotFoundError(
                    "provide --nodes <path> for a .edges file")
            G, dims = load_graph(nodes_path=nodes_path, edges_path=p)
        else:
            raise ValueError(
                f"unsupported file type {p.suffix!r}: use a graph "
                f"({', '.join(GRAPH_SUFFIXES)}) or an end-linked LAMMPS data "
                f"file ({', '.join(DATA_SUFFIXES)})")
    box = graph_box(G)
    if box is None and dims is not None:
        box = np.asarray(dims, float).reshape(3)
    return G, box, None


def load_reference(path, nodes: Optional[str] = None, heavy: bool = True,
                   seed: int = 0) -> Descriptors:
    """Descriptors to compare against: a network, or a saved report.

    A ``.json`` is read as the report ``topon analyze --json`` wrote (its
    ``descriptors`` block, or the whole file when it is a bare descriptor
    set such as ``Descriptors.save`` writes), with the distributions from the
    ``.npz`` beside it. Anything else is loaded and described.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        d = Descriptors.load(p)
        if isinstance(d.scalars.get("descriptors"), dict):
            d = Descriptors(d.scalars["descriptors"], d.dists)
        return d
    G, _box, _system = load_network(p, nodes)
    return describe(G, heavy=heavy, seed=seed)


def analyze_network(path, nodes: Optional[str] = None, heavy: bool = True,
                    seed: int = 0, z1: bool = False, z1_config=None,
                    compare_to=None, strands=None) -> tuple[dict, dict]:
    """Everything `topon analyze` reports on one network.

    Returns ``(report, dists)``: the report as plain numbers (``descriptors``,
    ``spatial`` when the cell and positions are known, ``chains`` and ``z1``
    for a data file, ``backbone`` in place of ``chains`` for an atomistic one,
    ``compare`` when a reference is given) and the per-node / per-edge
    distributions for an npz. ``strands`` is the run manifest of an atomistic
    data file (see :func:`load_network`).

    Raises:
        ValueError: ``z1`` on a graph, which has no coordinates to measure.
    """
    from topon.analysis.atomistic import AtomisticSystem, backbone_statistics

    G, box, system = load_network(path, nodes, strands)
    atomistic = isinstance(system, AtomisticSystem)
    if z1 and system is None:
        raise ValueError(
            "Z1+ measures chain paths, which a graph file does not have: pass "
            "the LAMMPS data file (end-linked convention) instead")

    desc = describe(G, heavy=heavy, seed=seed)
    report = {"input": str(path),
              "source": ("atomistic data" if atomistic else
                         "data" if system is not None else "graph"),
              "descriptors": desc.scalars}
    dists = dict(desc.dists)

    has_pos = all("pos" in d for _, d in G.nodes(data=True))
    if box is not None and has_pos:
        try:
            sp, chords = spatial(G, box)
            sp["units"] = ("A" if atomistic else
                           "sigma" if system is not None or
                           G.graph.get("box_sigma") is not None else "lattice")
            report["spatial"] = sp
            dists["chord"] = chords
        except ValueError:
            pass

    if atomistic:
        report["backbone"] = backbone_statistics(system)
    elif system is not None:
        from topon.analysis.endlinked import chain_statistics
        report["chains"] = chain_statistics(system)[0]

    if z1:
        from topon.analysis.z1plus import (
            Z1PlusFailed, Z1PlusUnavailable, measure_system)
        try:
            z, res = measure_system(system, config=z1_config)
            report["z1"] = z
            dists["z1_Z"] = res.Z
            dists["z1_Lpp"] = res.Lpp
        except (Z1PlusUnavailable, Z1PlusFailed) as exc:
            report["z1_error"] = str(exc)

    if compare_to is not None:
        ref = load_reference(compare_to, heavy=heavy, seed=seed)
        report["compare"] = {"reference": str(compare_to),
                             **compare(desc, ref)}
    return report, dists


def write_report(report: dict, dists: dict, json_path) -> tuple[Path, Path]:
    """The report as JSON and its distributions beside it as ``.npz``."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=1, default=to_jsonable),
                         encoding="utf-8")
    npz = json_path.with_suffix(".npz")
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in dists.items()})
    return json_path, npz


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _f(x, fmt=".3f") -> str:
    if x is None:
        return "-"
    try:
        if not np.isfinite(x):
            return "n/a"
    except TypeError:
        return str(x)
    return format(x, fmt)


def format_pf(hist: dict) -> str:
    """A degree histogram as ``f=4: 2304, f=3: 161``, highest f first."""
    items = sorted(((int(k), v) for k, v in hist.items()), reverse=True)
    return ", ".join(f"f={f}: {n}" for f, n in items)


def format_z1(z: dict, indent: str = "  ") -> list:
    """Lines for a Z1+ summary (see :func:`topon.analysis.z1plus.summarise`)."""
    lines = [f"{indent}Z1+       : Z per chain {_f(z.get('Zmean'))} over "
             f"{z.get('n_chains')} chains ({_f(100 * z['frac_zero'], '.0f')} % "
             f"with none), Ne {_f(z.get('Ne_CK'), '.1f')} (classical Kuhn)"
             if z.get("frac_zero") is not None else
             f"{indent}Z1+       : Z per chain {_f(z.get('Zmean'))}"]
    for cls, rec in (z.get("by_class") or {}).items():
        lines.append(f"{indent}  {cls:<9s}: {rec['n']:>6d} chains, Z "
                     f"{_f(rec['Zmean'])}, {_f(100 * rec['frac_zero'], '.0f')} "
                     f"% with none")
    if z.get("partner_pairs") is not None:
        lines.append(f"{indent}  partners : {z['partner_pairs']} entangled "
                     f"chain pairs, {_f(z.get('partner_degree_mean'))} "
                     f"partners per chain")
    return lines


def format_report(report: dict) -> str:
    """Render :func:`analyze_network`'s report for the terminal."""
    d = report["descriptors"]
    lines = [f"topon analyze  {report['input']}", ""]
    lines.append(
        f"  network   : {d.get('n_junctions')} junctions, "
        f"{d.get('n_end_nodes')} dangling ends, {d.get('n_edges_total')} "
        f"strands ({d.get('n_bridging_chains')} bridging, "
        f"{d.get('n_dangling_chains')} dangling)")
    lines.append(
        f"  loops     : {d.get('n_primary_loops')} primary (self-loops), "
        f"{d.get('n_secondary_loops')} secondary (parallel strands), "
        f"{d.get('n_sol_chains', 0)} sol chains")
    if d.get("deg_chem_dist"):
        lines.append(f"  P(f) chem : {format_pf(d['deg_chem_dist'])}")
        lines.append(f"  P(f) eff  : {format_pf(d['deg_eff_dist'])}")
    if d.get("giant_frac_junctions") is not None:
        lines.append(
            f"  connected : giant component {_f(100 * d['giant_frac_junctions'], '.1f')} "
            f"% of junctions, cycle rank {_f(d.get('cycle_rank_per_junction'))} "
            f"per junction")
    if d.get("core_nodes") is not None:
        lines.append(
            f"  core      : {d['core_nodes']} nodes "
            f"({_f(100 * d['frac_active_junctions'], '.1f')} % of junctions active), "
            f"mean degree {_f(d.get('core_deg_mean'), '.2f')}, cycle rank "
            f"{_f(d.get('core_cycle_rank_per_node'))} per node")
    if d.get("edge_shortest_cycle_mean") is not None:
        lines.append(
            f"  cycles    : shortest through a strand {_f(d['edge_shortest_cycle_mean'], '.2f')} "
            f"on average, odd {_f(d.get('frac_odd_cycles'))}, <=4 "
            f"{_f(d.get('frac_edges_in_cycle_le4'))}, <=6 "
            f"{_f(d.get('frac_edges_in_cycle_le6'))}")
    if d.get("transitivity_core") is not None:
        lines.append(
            f"  cluster   : transitivity {_f(d['transitivity_core'])}, average "
            f"{_f(d.get('avg_clustering_core'))}, square "
            f"{_f(d.get('square_clustering_core'))}, assortativity "
            f"{_f(d.get('assortativity_core'))}")
    if d.get("lambda2_core") is not None:
        lines.append(
            f"  spectrum  : lambda2 {_f(d.get('lambda2_full'))} full, "
            f"{_f(d['lambda2_core'])} core; energy {_f(d.get('graph_energy_core_per_node'))} "
            f"per node; spectral radius {_f(d.get('spectral_radius_core'))}")
    if d.get("avg_path_core") is not None:
        lines.append(
            f"  paths     : mean {_f(d['avg_path_core'], '.2f')}, diameter at "
            f"least {d.get('diameter_core_est')} (sampled)")
    if d.get("betweenness_gini_core") is not None:
        lines.append(
            f"  load      : betweenness Gini {_f(d['betweenness_gini_core'])} "
            f"(edges {_f(d.get('edge_betweenness_gini_core'))}), eigenvector "
            f"centrality CV {_f(d.get('eigvec_cent_cv_core'))}")
    if d.get("edge_eff_resistance_mean") is not None:
        lines.append(
            f"  resistance: per strand {_f(d['edge_eff_resistance_mean'])} "
            f"(CV {_f(d.get('edge_eff_resistance_cv'))}), Kirchhoff "
            f"{_f(d.get('kirchhoff_per_node'), '.1f')} per node")
    if d.get("n_bridges_core") is not None:
        lines.append(
            f"  robust    : {d['n_bridges_core']} bridge edge(s), "
            f"{d.get('n_articulation_core')} articulation point(s), k-core "
            f"up to {d.get('max_k_core')}")
    sp = report.get("spatial")
    if sp:
        lines.append(
            f"  chords    : {_f(sp['chord_mean'], '.2f')} +- {_f(sp['chord_sd'], '.2f')} "
            f"{sp.get('units', '')} (CV {_f(sp['chord_cv'])}), 5/50/95 % "
            f"{_f(sp['chord_p5'], '.2f')} / {_f(sp['chord_p50'], '.2f')} / "
            f"{_f(sp['chord_p95'], '.2f')}; orientation "
            + " ".join(_f(x) for x in sp["orientation_eigs"]))
    ch = report.get("chains")
    if ch:
        lines.append(
            f"  beads     : {ch['n_atoms']} at density {_f(ch['density'], '.4f')}, "
            f"bonds {_f(ch.get('bond_mean'))} mean, {_f(ch.get('bond_max'))} "
            f"max, {ch.get('bonds_over_1.2')} over 1.2"
            + (f", T {_f(ch['temperature'])}" if ch.get("temperature") else ""))
    bb = report.get("backbone")
    if bb:
        lines.append(
            f"  atoms     : {bb['n_atoms']} at {_f(bb['mass_density'], '.4f')} "
            f"g/cm3; {bb['backbone_bonds']} backbone bonds at "
            f"{_f(bb.get('backbone_ratio_min'))}-{_f(bb.get('backbone_ratio_max'))} "
            f"of r0, {bb['backbone_stretched']} over +{bb['tolerance']:.0%}"
            + (f", T {_f(bb['temperature'], '.1f')} K" if bb.get("temperature") else ""))
    if report.get("z1"):
        lines.extend(format_z1(report["z1"]))
    elif report.get("z1_error"):
        lines.append(f"  Z1+       : not measured. {report['z1_error']}")
    c = report.get("compare")
    if c:
        lines.extend(["", f"  against {c['reference']}:"])
        lines.append(f"    composite {_f(c.get('composite'))} over "
                     f"{c.get('n_terms')} terms (0 is identical)")
        if c.get("cycle_js") is not None:
            lines.append(f"    cycle spectrum JS divergence {_f(c['cycle_js'])}")
        if c.get("ks"):
            lines.append("    KS " + ", ".join(
                f"{k} {_f(v)}" for k, v in sorted(c["ks"].items())))
        if c.get("deviations"):
            lines.append("    deviation " + ", ".join(
                f"{k} {_f(v, '.2f')}" for k, v in c["deviations"].items()))
        if c.get("omitted"):
            lines.append(f"    left out: {', '.join(c['omitted'])}")
    return "\n".join(lines)
