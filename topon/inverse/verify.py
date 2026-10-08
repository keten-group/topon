"""``topon generate --verify``: regenerate a config and hold it against a reference.

Each seed is built through the pipeline's stages 1 to 3 (the graph the
chemistry stage receives, defects included) and measured the way
:func:`topon.inverse.measure.measure` measures the reference, so the two
sides of every row come from the same code. The report has four parts:

requested against achieved
    the ``degree_distribution`` against the sculpt's degree counts (exact,
    or it says where not), the defects asked for against those placed, DP
    and bead count, and the chemical P(f) against the reference's, which
    differs by the loops the lattice route cannot place.
connectivity
    :func:`topon.analysis.descriptors.compare` of every seed against the
    reference: the composite (each term over the reference value, the
    number the acceptance thresholds are written in), the cycle-spectrum
    divergence, the betweenness KS and the per-term deviations.
spatial, as built
    junction-junction separation of the bridges on the lattice scaled to the
    box the density gives. The graph is fixed through MD and junctions move
    a fraction of a spacing, so this is most of the relaxed value, but it is
    the built state and is labelled so.
entanglement, relaxed
    only when a relaxed build is given (``relaxed``, a data file or a run
    directory): Z1+ per strand class against the reference with KS
    p-values, the per-bridge histogram beside the reference's and against
    ``conformation.entanglement.target_hist``, partners, the chain
    statistics after relaxation, and whether it meets the acceptance
    (:data:`Z_ACCEPT`). The file is checked first: that it is the network
    this config builds at its seed, that LAMMPS wrote it after a run, and
    that its units, density and temperature are the reference's. Producing
    it needs MD, which ``topon generate`` does not run; the deck it writes
    does, and so does the controller's driver in the development
    repository, which builds at the fitted knob.
"""
from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from topon.inverse.measure import Measurement, Reference, measure, read_reference
from topon.inverse.scaffold import build_graph

#: Relative gap between the relaxed file's density or temperature and the
#: reference's past which the Z comparison is flagged: Z1+ is a state
#: property and compares only between matched states (REPORT.md 4.4).
STATE_TOLERANCE = {"density": 0.02, "temperature": 0.10}

#: When a relaxed build's entanglement matches the reference's:
#: Z1+ per bridge within this fraction of the reference's, and the per-bridge
#: Z of the two not told apart by a two-sample KS test at this level.
Z_ACCEPT = {"z_bridge_gap": 0.10, "hist_p": 0.05}

#: Strand classes whose Z1+ is compared, in the order they are printed.
Z_CLASSES = ("bridge", "loop", "dangling")


def _requested(config: dict) -> dict:
    """What the config asks for. A loaded topology asks for no P(f)."""
    from topon.topology.degree_matching import parse_degree_distribution

    counts = {}
    if config.get("topology", {}).get("source", "generate") == "generate":
        gen = config["topology"].get("generator") or {}
        counts, _ = parse_degree_distribution(gen.get("degree_distribution"))
    defects = config.get("assignment", {}).get("defects", {}) or {}
    dp = config.get("assignment", {}).get("dp_distribution", {}).get("default", {})
    return {
        "degree_distribution": {int(d): int(n) for d, n in sorted(counts.items())},
        "primary_loops": int((defects.get("primary_loops") or {}).get("count") or 0),
        "secondary_loops": int((defects.get("secondary_loops") or {}).get("count") or 0),
        "sol_chains": int((defects.get("sol_chains") or {}).get("count") or 0),
        "dp": dp.get("mean", 25),
        "density": config.get("chemistry", {}).get("target_density"),
    }


def generated_reference(G, config: dict, label: str) -> Reference:
    """A generated graph as a :class:`Reference` in sigma units.

    On the coarse-grained route the positions and the cell are scaled from
    lattice units the way the chemistry stage sizes its box: one isotropic
    factor that puts the bead count at ``chemistry.target_density``. A
    chain's DP counts its beads with the free end of a dangling strand
    among them, as the reference's data file does (the end site is a bead
    in both of topon's conventions). On the atomistic route the density is
    a mass density and a strand is not its beads, so the graph stays in
    lattice units.
    """
    from topon.assignment.defects import is_end_site
    from topon.inverse.scaffold import total_beads
    from topon.topology.loader import infer_dims_from_graph

    chem = config.get("chemistry", {})
    cg = chem.get("model_type", "coarse_grained") == "coarse_grained"
    dp_mean = (config.get("assignment", {}).get("dp_distribution", {})
               .get("default", {}).get("mean", 25))
    beads = total_beads(G, dp_mean) if cg else None

    H = G.copy()
    kind = {n: ("end" if is_end_site(G, n) else "junction") for n in H.nodes()}
    chain_dp = collections.defaultdict(list)
    for u, v, d in H.edges(data=True):
        dp = int(d.get("dp") or 0)
        if u == v:
            chain_dp["loop"].append(dp)
        elif "end" in (kind[u], kind[v]):
            chain_dp["dangling"].append(dp + 1)
        else:
            chain_dp["bridge"].append(dp)
    sol = H.graph.get("sol_chains") or {}
    if isinstance(sol, dict) and sol.get("count"):
        chain_dp["free"] = [int(sol.get("dp") or dp_mean)] * int(sol["count"])
    for n, d in H.nodes(data=True):
        d["kind"] = kind[n]

    box = None
    dims = infer_dims_from_graph(G)
    if beads and dims is not None and chem.get("target_density"):
        box_lat = np.asarray(dims, float)
        volume = beads / float(chem["target_density"])
        scale = (volume / float(np.prod(box_lat))) ** (1.0 / 3.0)
        box = box_lat * scale
        for _n, d in H.nodes(data=True):
            if d.get("pos") is not None:
                d["pos"] = np.asarray(d["pos"], float) * scale
        H.graph["box"] = tuple(float(x) for x in box)
        H.graph.pop("box_sigma", None)
    return Reference(path=Path(label), source="generated", graph=H, box=box,
                     units="sigma" if box is not None else "lattice",
                     chain_dp=dict(chain_dp), n_atoms=beads)


def _pf_gap(a: dict, b: dict) -> dict:
    keys = sorted({int(k) for k in a} | {int(k) for k in b})
    return {k: int(a.get(k, a.get(str(k), 0))) - int(b.get(k, b.get(str(k), 0)))
            for k in keys}


def _ks(x, y) -> dict:
    from scipy import stats

    x, y = np.asarray(x), np.asarray(y)
    if not len(x) or not len(y):
        return {"p": None, "statistic": None}
    r = stats.ks_2samp(x, y)
    return {"p": float(r.pvalue), "statistic": float(r.statistic),
            "mean": float(x.mean()), "reference_mean": float(y.mean()),
            "n": int(len(x)), "reference_n": int(len(y))}


def verify(config: dict, reference, *, seeds: Sequence[int],
           built: Optional[dict] = None, relaxed=None,
           junction_type: Optional[int] = None, z1_config=None,
           measured: Optional[Measurement] = None,
           log: Optional[Callable[[str], None]] = None,
           replicates: Optional[Sequence] = None) -> dict:
    """Regenerate ``config`` on ``seeds`` and compare it with ``reference``.

    ``built`` maps a seed to a graph already built (the one ``topon
    generate`` just made from the config's own seed); it is regenerated
    anyway and the two are compared, which is the determinism check.
    ``relaxed`` is an end-linked data file of the relaxed build, or the run
    directory it is in, for the Z1+ part; it is held against the first of
    ``seeds``. ``measured`` is the reference's measurement when the caller
    already has it (with Z1+ per strand if ``relaxed`` is given), to save
    measuring it again. Returns the report as plain numbers.

    A config that builds a network crosslinked along its chains (the
    crosslink generator or the lattice route) is verified by
    :func:`topon.inverse.crosslinked_verify.verify_crosslinked`, which
    also takes ``replicates`` of the reference.
    """
    from topon.analysis.descriptors import compare
    from topon.inverse.crosslinked_verify import (
        is_crosslinked_config, verify_crosslinked)

    if is_crosslinked_config(config) or (
            measured is not None
            and measured.record.get("architecture") == "crosslinked"):
        return verify_crosslinked(config, reference, seeds=seeds, built=built,
                                  replicates=replicates, measured=measured,
                                  relaxed=relaxed, log=log)
    if replicates:
        raise ValueError("replicates are held against a build crosslinked "
                         "along its chains; this config is end-linked")
    say = log or (lambda msg: None)
    t0 = time.perf_counter()
    if relaxed is not None:
        resolve_relaxed(relaxed)       # a run with no MD fails before the builds
    if measured is None:
        ref = read_reference(reference, junction_type=junction_type)
        want_z1 = relaxed is not None and ref.source == "data"
        measured = measure(ref, z1=want_z1 or False, z1_config=z1_config)
    ref_meas = measured
    rrec = ref_meas.record
    requested = _requested(config)
    say(f"reference measured in {time.perf_counter() - t0:.0f} s")

    per_seed = []
    for s in seeds:
        t1 = time.perf_counter()
        G, man = build_graph(config, seed=s)
        same = None
        if built and s in built:
            same = _same_graph(built[s], G)
        gen = generated_reference(G, config, f"seed {s}")
        gm = measure(gen, z1=False)
        g = gm.record
        cmp = compare(gm.descriptors, ref_meas.descriptors)
        sculpt = (man.get("sculpt") or {})
        achieved = {int(k): int(v) for k, v in (sculpt.get("achieved") or {}).items()}
        req = requested["degree_distribution"]
        exact = (all(achieved.get(d, 0) == n for d, n in req.items() if d >= 1)
                 if req else None)
        row = {
            "seed": int(s),
            "same_as_built": same,
            "pf_requested": req,
            "pf_achieved": achieved,
            "pf_exact": exact,
            "pf_effective": g["pf_effective"],
            "pf_chemical": g["pf_chemical"],
            "chains": g["chains"],
            "secondary_loops": g["secondary_loops"],
            "primary_loops": g["primary_loops"],
            "beads": g["n_atoms"],
            "box": g["box"],
            "dp": g["dp"],
            "giant_fraction": g["giant_fraction"],
            "composite": cmp["composite"],
            "cycle_js": cmp.get("cycle_js"),
            "ks_betweenness": cmp["ks"].get("betweenness_core"),
            "deviations": cmp["deviations"],
            "omitted": cmp["omitted"],
            "descriptors": {k: g["descriptors"].get(k) for k in _SHOWN},
            "spatial_built": g.get("spatial"),
            "seconds": round(time.perf_counter() - t1, 1),
        }
        per_seed.append(row)
        pf = {True: "exact", False: "NOT exact", None: "not requested"}[exact]
        say(f"  seed {s}: P(f) {pf}, composite {cmp['composite']:.3f} "
            f"({row['seconds']:.0f} s)")

    gen = config.get("topology", {}).get("generator") or {}
    report = {
        "config": {"lattice": gen.get("lattice_type"),
                   "lattice_size": gen.get("lattice_size"),
                   "neighbour_cutoff": gen.get("neighbour_cutoff"),
                   **requested},
        "reference": {k: rrec.get(k) for k in (
            "input", "source", "chains", "pf_effective", "pf_chemical",
            "secondary_loops", "primary_loops", "sol_chains", "n_atoms",
            "box", "density", "dp", "giant_fraction", "spatial", "notes")},
        "reference_descriptors": {k: rrec["descriptors"].get(k) for k in _SHOWN},
        "seeds": per_seed,
        "summary": _summary(per_seed, rrec, requested),
    }
    if relaxed is not None:
        report["relaxed"] = _relaxed(relaxed, config, ref_meas, z1_config, say,
                                     built=per_seed[0] if per_seed else None)
    report["flags"] = _flags(report, rrec, config)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


#: Descriptors listed per seed next to the reference's.
_SHOWN = ("edge_shortest_cycle_mean", "frac_odd_cycles",
          "frac_edges_in_cycle_le4", "transitivity_core",
          "square_clustering_core", "assortativity_core",
          "betweenness_gini_core", "edge_eff_resistance_cv", "lambda2_core",
          "avg_path_core")


def _same_graph(a, b) -> bool:
    """Same nodes, and the same edges with the same DP in the same order.

    The DP is part of it: a polydisperse config draws its DPs from the
    global stream, which the pipeline does not seed, so its graph can
    repeat while its strands do not.
    """
    if sorted(a.nodes()) != sorted(b.nodes()):
        return False
    ea = [(min(u, v), max(u, v), d.get("dp")) for u, v, d in a.edges(data=True)]
    eb = [(min(u, v), max(u, v), d.get("dp")) for u, v, d in b.edges(data=True)]
    return ea == eb


def _summary(rows, rrec, requested) -> dict:
    comp = [r["composite"] for r in rows if r["composite"] is not None]
    first = rows[0] if rows else {}
    chem_gap = _pf_gap(first.get("pf_chemical", {}), rrec["pf_chemical"])
    out = {
        "n_seeds": len(rows),
        "pf_exact_all": (None if any(r["pf_exact"] is None for r in rows)
                         else all(r["pf_exact"] for r in rows)),
        "composite_mean": float(np.mean(comp)) if comp else None,
        "composite_sd": float(np.std(comp, ddof=1)) if len(comp) > 1 else 0.0,
        "composite_range": [float(min(comp)), float(max(comp))] if comp else None,
        "secondary_loops": {"requested": requested["secondary_loops"],
                            "achieved": [r["secondary_loops"]["count"] for r in rows],
                            "reference": rrec["secondary_loops"]["count"]},
        "primary_loops": {"requested": requested["primary_loops"],
                          "achieved": [r["primary_loops"]["count"] for r in rows],
                          "reference": rrec["primary_loops"]["count"]},
        "sol_chains": {"requested": requested["sol_chains"],
                       "achieved": [r["chains"]["free"] for r in rows],
                       "reference": rrec["sol_chains"]},
        "chemical_pf_gap": chem_gap,
        "beads": {"achieved": [r["beads"] for r in rows],
                  "reference": rrec.get("n_atoms")},
        "same_as_built": [r["same_as_built"] for r in rows
                          if r["same_as_built"] is not None],
    }
    out["largest_term"] = _largest_term(rows)
    sp = [r["spatial_built"] for r in rows if r.get("spatial_built")]
    if sp:
        out["spatial_built"] = {k: float(np.mean([x[k] for x in sp]))
                                for k in ("chord_mean", "chord_sd", "chord_cv",
                                          "chord_p95")}
    return out


def _largest_term(rows) -> Optional[dict]:
    """The composite's largest term over the seeds, and the composite without it.

    Each term is divided by the reference value, so a reference value near
    zero makes its term the whole composite (the DP-100 assortativity is
    -0.006, and its term is 93 % of the composite at eight shells). Naming
    it, and giving the composite without it, says whether the rest agrees.
    """
    parts = []
    for r in rows:
        if r["composite"] is None:
            continue
        p = dict(r["deviations"])
        if r.get("cycle_js") is not None:
            p["cycle_js"] = r["cycle_js"]
        if r.get("ks_betweenness") is not None:
            p["ks_betweenness"] = r["ks_betweenness"]
        parts.append(p)
    if not parts:
        return None
    keys = sorted(set.intersection(*(set(p) for p in parts)))
    mean = {k: float(np.mean([p[k] for p in parts])) for k in keys}
    top = max(mean, key=mean.get)
    total = float(np.mean([np.mean(list(p.values())) for p in parts]))
    without = [np.mean([v for k, v in p.items() if k != top]) for p in parts]
    return {"term": top, "deviation": mean[top],
            "share": mean[top] / len(keys) / total if total > 0 else None,
            "composite_without": float(np.mean(without)),
            "composite_without_sd": (float(np.std(without, ddof=1))
                                     if len(without) > 1 else 0.0)}


def resolve_relaxed(path) -> Path:
    """The relaxed data file ``path`` names: the file, or the one in a run.

    A directory is either kind of run: a ``topon generate`` study folder
    (``<output_dir>/<name>``, its checkpoints in ``04_Simulation``), that
    ``04_Simulation`` folder, or a controller round
    (``<validation>/data/runs/<case>_r<n>``, flat). The furthest MD
    checkpoint in it is taken: on the push-off the stage-5 quench, or stage
    6 when the deck converts to the quartic bond. The conformation stage's
    ``system_relaxed.data`` has only had its overlaps removed and is never
    taken from a directory.

    Raises:
        ValueError: a directory with no MD checkpoint in it.
    """
    from topon.analysis.run_summary import network_file

    p = Path(path)
    if not p.is_dir():
        return p
    for flat in (False, True):
        f = network_file(p, flat=flat)
        if f is not None and f.name != "system_relaxed.data":
            return f
    raise ValueError(
        f"{p}: no MD checkpoint here or in its 04_Simulation folder (the "
        f"push-off writes stage1_min.data to stage5_final_quench.data). Run "
        f"the relaxation first, or name the data file.")


def data_header(path) -> dict:
    """What the first line of a LAMMPS data file says about where it came from.

    ``write_data`` heads its file ``LAMMPS data file via write_data, version
    ..., timestep = N, units = U``; topon's writers and other generators
    write a first line of their own. Returns ``write_data`` (bool),
    ``timestep`` and ``units`` (None when the line does not say) and the
    line itself.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
    step = re.search(r"timestep\s*=\s*(\d+)", first)
    units = re.search(r"units\s*=\s*(\w+)", first)
    return {"write_data": "write_data" in first,
            "timestep": int(step.group(1)) if step else None,
            "units": units.group(1) if units else None, "line": first}


def _after_md(header: dict) -> bool:
    return bool(header.get("write_data")) and (header.get("timestep") or 0) > 0


def _graph_hash(G) -> str:
    """A hash of a strand graph that two isomorphic graphs share.

    Junctions and free ends are the nodes, labelled by kind and by the
    primary loops they carry; the strands between two nodes are one edge
    labelled by how many there are; the vacancies of a generated cell are
    left out (:func:`topon.analysis.descriptors.network_view`). Weisfeiler-
    Lehman, so equal hashes are not a proof of isomorphism, but a relaxed
    build and the graph it was built from always agree, and two seeds of
    one config have not been seen to.
    """
    import networkx as nx

    from topon.analysis.descriptors import network_view

    V = network_view(G)
    loops = collections.Counter(u for u, v in V.edges() if u == v)
    H = nx.Graph()
    for n, d in V.nodes(data=True):
        H.add_node(n, label=f"{d.get('kind')}:{loops.get(n, 0)}")
    mult = collections.Counter(frozenset((u, v))
                               for u, v in V.edges() if u != v)
    for pair, c in mult.items():
        u, v = tuple(pair)
        H.add_edge(u, v, mult=str(c))
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="label",
                                           edge_attr="mult", iterations=4)


def _network_check(rec: dict, built: dict,
                   same_graph: Optional[bool] = None) -> dict:
    """The relaxed file's network against the build of the config's seed.

    Relaxation moves beads and never bonds, so the strand graph of a relaxed
    build is the one the config built, exactly. ``same_network`` asks
    whether it is a build of this config: junctions by effective degree,
    strands per class and the DP of bridges and loops, which the config
    fixes whatever the seed. ``same_graph`` (given by the caller, from
    :func:`_graph_hash`) whether it is the graph of this seed.
    ``same_strands`` whether its other strands are the pipeline's: sol
    chains, dangling DP and the bead count. A file of another config fails
    the first unless that config differs only in what these do not see (a
    cutoff that gives the same counts, for instance), another seed the
    second, a build that dropped or shortened free-ended strands the third.
    """
    net, strands = [], []
    for cls in ("bridge", "loop", "dangling"):
        a, b = rec["chains"].get(cls, 0), built["chains"].get(cls, 0)
        if a != b:
            net.append(f"{a} {cls} strands against {b}")
    pa = {int(k): int(v) for k, v in rec["pf_effective"].items()}
    pb = {int(k): int(v) for k, v in built["pf_effective"].items()}
    if pa != pb:
        net.append(f"effective P(f) {_pf(pa)} against {_pf(pb)}")
    a, b = rec["chains"].get("free", 0), built["chains"].get("free", 0)
    if a != b:
        strands.append(f"{a} sol chains against {b}")
    rdp = (rec.get("dp") or {}).get("by_class") or {}
    bdp = (built.get("dp") or {}).get("by_class") or {}
    for cls in [c for c in Z_CLASSES + ("free",) if c in rdp and c in bdp]:
        if abs(rdp[cls]["mean"] - bdp[cls]["mean"]) > 1e-9:
            (net if cls in ("bridge", "loop") else strands).append(
                f"{cls} DP {rdp[cls]['mean']:.4g} against "
                f"{bdp[cls]['mean']:.4g}")
    if rec.get("n_atoms") != built.get("beads"):
        strands.append(f"{rec.get('n_atoms')} beads against "
                       f"{built.get('beads')}")
    return {"seed": built["seed"], "same_network": not net,
            "same_graph": same_graph,
            "same_strands": not strands, "differences": net + strands,
            "built": {"chains": built["chains"], "beads": built.get("beads")}}


def _z_against(z, zr) -> dict:
    """Z1+ of one strand class against the reference's, with a KS p-value.

    A class on one side only (a build with no loops, a reference with no sol
    chains) is kept with its counts and no p-value rather than dropped.
    """
    n, rn = (0 if z is None else len(z)), (0 if zr is None else len(zr))
    if n and rn:
        return _ks(z, zr)
    return {"p": None, "statistic": None,
            "mean": float(np.mean(z)) if n else None,
            "reference_mean": float(np.mean(zr)) if rn else None,
            "n": n, "reference_n": rn}


def _histogram(z, zr) -> Optional[dict]:
    """Per-bridge Z as fractions [P(Z=0), P(Z=1), ...], relaxed and reference."""
    if z is None or zr is None or not len(z) or not len(zr):
        return None
    z, zr = np.asarray(z, int), np.asarray(zr, int)
    top = int(max(z.max(), zr.max())) + 1
    return {side: [round(float(c) / len(x), 4)
                   for c in np.bincount(x, minlength=top)]
            for side, x in (("relaxed", z), ("reference", zr))}


def _partners(z1: dict, rz1: dict) -> dict:
    """Chain-chain entanglement partners, relaxed and reference.

    ``pairs`` counts the distinct chain pairs Z1+ names at a kink, which
    grows with the number of chains; ``per_strand`` and ``per_bridge`` are
    the mean number of distinct partners, which compare across networks of
    different size.
    """
    def per_bridge(z):
        h = np.asarray(z.get("partner_degree_hist_bridge") or [], float)
        if not h.sum():
            return None
        return float((np.arange(len(h)) * h).sum() / h.sum())

    return {"pairs": {"measured": z1.get("partner_pairs"),
                      "reference": rz1.get("partner_pairs")},
            "per_strand": {"measured": z1.get("partner_degree_mean"),
                           "reference": rz1.get("partner_degree_mean")},
            "per_bridge": {"measured": per_bridge(z1),
                           "reference": per_bridge(rz1)}}


def _acceptance(zb: Optional[dict], this_build: bool = True,
                state_ok: bool = True) -> Optional[dict]:
    """Z per bridge within :data:`Z_ACCEPT` of the reference's, and the
    per-bridge KS p above it.

    ``met`` needs both, on a build of this config (``this_build``) at the
    reference's state (``state_ok``, :func:`_state`); otherwise the two
    numbers are kept and the verdict is withheld (``met`` False,
    ``withheld`` saying why).
    """
    if not zb or zb.get("mean") is None or not zb.get("reference_mean"):
        return None
    gap = (zb["mean"] - zb["reference_mean"]) / zb["reference_mean"]
    p = zb.get("p")
    within = abs(gap) <= Z_ACCEPT["z_bridge_gap"]
    hist_ok = p is not None and p > Z_ACCEPT["hist_p"]
    withheld = ([] if this_build else ["another network"]) + (
        [] if state_ok else ["not at the reference's state"])
    return {"z_bridge": zb["mean"], "reference": zb["reference_mean"],
            "relative_gap": float(gap), "z_within": bool(within),
            "hist_p": p, "hist_ok": bool(hist_ok),
            "this_build": bool(this_build), "state_ok": bool(state_ok),
            "withheld": withheld,
            "met": bool(within and hist_ok and not withheld),
            "tolerance": Z_ACCEPT["z_bridge_gap"], "p_min": Z_ACCEPT["hist_p"]}


def _state(rel: dict, rrec: dict) -> dict:
    """Whether the relaxed file is at the reference's state, check by check.

    ``after_md``: LAMMPS wrote it after a run. ``units``, ``density`` and
    ``temperature`` each give both values and ``ok``, which is None when a
    side does not say (no units on its first line, no bead density, no
    Velocities section or velocities that are all zero), so a quantity that
    was not checked is never read as matched. ``ok`` overall is False when
    the file is not after MD or any check failed.
    """
    hdr, rhdr = rel.get("header") or {}, rel.get("reference_header") or {}
    out: dict = {"after_md": _after_md(hdr)}
    u, ru = hdr.get("units"), rhdr.get("units")
    out["units"] = {"relaxed": u, "reference": ru,
                    "ok": (u == ru) if u and ru else None}
    for key, tol in STATE_TOLERANCE.items():
        a = rel.get(key)
        b = (rrec.get(key) if key == "density"
             else (rrec.get("spatial") or {}).get(key))
        ok = None if a is None or not b else bool(abs(a - b) / b <= tol)
        out[key] = {"relaxed": a, "reference": b, "tolerance": tol, "ok": ok}
    out["ok"] = bool(out["after_md"] and all(
        out[k]["ok"] is not False for k in ("units", "density", "temperature")))
    return out


def _relaxed(path, config, ref_meas, z1_config, say, built=None) -> dict:
    """Z1+ and chain statistics of a relaxed build against the reference.

    ``path`` is the data file or a run directory (:func:`resolve_relaxed`).
    ``built`` is the verification's row for the config's own seed: the
    relaxed file's network is checked against it, because a file from
    another config or seed would otherwise be compared with the reference as
    if it were this build.
    """
    from topon.conformation.entanglement import hist_ks

    given = Path(path)
    path = resolve_relaxed(given)
    rel = read_reference(path)
    if rel.source != "data":
        raise ValueError(f"{path}: the relaxed build has to be an end-linked "
                         f"LAMMPS data file")
    # The graph is the build's; its descriptors are the seed row's already.
    m = measure(rel, heavy=False, z1=True, z1_config=z1_config)
    rec = m.record
    rrec = ref_meas.record
    out = {"input": str(path), "density": rec.get("density"),
           "temperature": (rec.get("spatial") or {}).get("temperature"),
           "spatial": rec.get("spatial"), "chains": rec["chains"],
           "beads": rec.get("n_atoms"), "header": data_header(path)}
    if given != path:
        out["given"] = str(given)
    if rrec.get("source") == "data":
        out["reference_header"] = data_header(rrec["input"])
    if built is not None:
        # read as the seed row reads it, node kinds from the end sites
        G, _ = build_graph(config, seed=built["seed"])
        gen = generated_reference(G, config, f"seed {built['seed']}")
        out["network"] = _network_check(
            rec, built,
            same_graph=_graph_hash(gen.graph) == _graph_hash(rel.graph))
    out["state"] = _state(out, rrec)
    rz = ref_meas.z_per_strand
    out["z_by_class"] = {cls: _z_against(m.z_per_strand.get(cls), rz.get(cls))
                         for cls in Z_CLASSES
                         if cls in m.z_per_strand or cls in rz}
    zb = m.z_per_strand.get("bridge")
    hist = _histogram(zb, rz.get("bridge"))
    if hist:
        out["histogram"] = {**hist, "p": out["z_by_class"]["bridge"]["p"]}
    ent = (config.get("conformation") or {}).get("entanglement") or {}
    if zb is not None and ent.get("target_hist"):
        out["target_hist"] = hist_ks(zb, ent["target_hist"])
    if zb is not None and ent.get("target_Z"):
        tz = float(ent["target_Z"])
        out["target_Z"] = {"target": tz, "measured": float(np.mean(zb)),
                           "relative_gap": float((np.mean(zb) - tz) / tz)}
    out["partners"] = _partners(rec.get("z1") or {}, rrec.get("z1") or {})
    out["acceptance"] = _acceptance(
        out["z_by_class"].get("bridge"),
        this_build=bool((out.get("network") or {}).get("same_network", True)),
        state_ok=out["state"]["ok"])
    acc = out["acceptance"]
    say("  relaxed: Z per bridge "
        + (f"{acc['z_bridge']:.3f} against the reference's "
           f"{acc['reference']:.3f}" if acc else "n/a"))
    return out


def _flags(report, rrec, config) -> list:
    flags = []
    s = report["summary"]
    if s["pf_exact_all"] is False:
        bad = [r["seed"] for r in report["seeds"] if r["pf_exact"] is False]
        flags.append({"level": "error", "what": "P(f) not exact",
                      "detail": f"seeds {bad} missed the requested degree counts"})
    if s["same_as_built"] and not all(s["same_as_built"]):
        flags.append({"level": "error", "what": "not deterministic",
                      "detail": "the regenerated graph differs from the one "
                                "the pipeline built from the same seed"})
    for key in ("secondary_loops", "primary_loops", "sol_chains"):
        e = s[key]
        if any(a != e["requested"] for a in e["achieved"]):
            flags.append({"level": "warn", "what": f"{key} short",
                          "detail": f"requested {e['requested']}, placed "
                                    f"{e['achieved']}"})
    gap = {k: v for k, v in s["chemical_pf_gap"].items() if v}
    if gap:
        flags.append({"level": "note", "what": "chemical P(f)",
                      "detail": "differs from the reference by "
                                + ", ".join(f"f={k}: {v:+d}" for k, v in sorted(gap.items(), reverse=True))
                                + " (the loops the lattice route cannot place "
                                  "where the reference has them)"})
    for note in rrec.get("notes", []):
        flags.append({"level": "note", "what": "reference", "detail": note})
    rel = report.get("relaxed")
    if rel is None:
        target = ((config.get("conformation") or {}).get("entanglement") or {}).get("target_Z")
        flags.append({"level": "note", "what": "entanglement not verified",
                      "detail": "Z1+ needs a relaxed system, which needs MD; "
                                + (f"the target is {target} per bridge at the "
                                   f"final state. " if target else "")
                                + "Pass --relaxed <final data file> to "
                                  "measure one."})
    else:
        flags += _relaxed_flags(rel, rrec)
    return flags


def _relaxed_flags(rel: dict, rrec: dict) -> list:
    """Why the relaxed comparison may not be read as it stands, and its verdict."""
    flags = []

    def flag(level, what, detail):
        flags.append({"level": level, "what": what, "detail": detail})

    name = Path(rel["input"]).name
    hdr = rel.get("header") or {}
    st = rel.get("state") or _state(rel, rrec)
    if not st["after_md"]:
        flag("warn", "not a relaxed state",
             f"{name} was not written by LAMMPS after a run (its first line "
             f"is {hdr.get('line', '')!r}), so its Z1+ is that of a build; "
             f"the reference's is a final-state number. Pass the last "
             f"checkpoint of the relaxation, or its run directory.")
    u = st["units"]
    if u["ok"] is False:
        flag("warn", "unmatched units",
             f"{name} is in {u['relaxed']} units and the reference in "
             f"{u['reference']}; densities, lengths and temperatures do not "
             f"compare")
    elif u["ok"] is None:
        flag("note", "units not checked",
             ("the relaxed file's" if not u["relaxed"] else "the reference's")
             + " first line does not name its units")
    for key in STATE_TOLERANCE:
        c = st[key]
        if c["ok"] is False:
            flag("warn", f"unmatched {key}",
                 f"relaxed {c['relaxed']:.4g} against the reference's "
                 f"{c['reference']:.4g}; Z1+ compares only at matched "
                 f"density and temperature")
        elif c["ok"] is None:
            side = "the relaxed file" if c["relaxed"] is None else "the reference"
            if key != "temperature":
                why = f"{side} has no bead density"
            elif c["relaxed"] is None or c["reference"] is None:
                why = f"{side} has no Velocities section"
            else:
                why = "the reference's velocities are all zero"
            flag("note", f"{key} not checked",
                 f"{why}, so the state the two Z1+ numbers were measured at "
                 f"is matched on {key} by assumption only")
    net = rel.get("network")
    if net and not net["same_network"]:
        flag("warn", "not this config's network",
             f"the relaxed file is not a build of this config (against seed "
             f"{net['seed']}: " + "; ".join(net["differences"])
             + "). It was built from another config, and its Z1+ is not "
               "this fit's.")
    else:
        if net and net.get("same_graph") is False:
            flag("note", "another seed",
                 f"the relaxed file has this config's strand counts, P(f) "
                 f"and DP but not seed {net['seed']}'s graph: a build of "
                 f"another seed, or of another config that gives the same "
                 f"counts")
        if net and not net["same_strands"]:
            flag("note", "strands differ from the pipeline's",
                 "the network is this config's, but "
                 + "; ".join(net["differences"]))
    zbc = rel.get("z_by_class") or {}
    if zbc and not any(k.get("reference_n") for k in zbc.values()):
        flag("note", "no reference Z1+",
             "the reference carries no Z1+ per strand (it is not a data "
             "file, or it was measured without Z1+), so the relaxed build's "
             "Z is listed and not compared")
    else:
        for cls, k in zbc.items():
            if k.get("p") is None and bool(k.get("n")) != bool(k.get("reference_n")):
                flag("note", f"no {cls} Z to compare",
                     f"{k.get('n', 0)} {cls} strands in the relaxed build "
                     f"and {k.get('reference_n', 0)} in the reference")
    acc = rel.get("acceptance")
    if acc:
        detail = (f"Z per bridge {acc['z_bridge']:.3f} against "
                  f"{acc['reference']:.3f} ({100 * acc['relative_gap']:+.1f} %, "
                  f"{'within' if acc['z_within'] else 'outside'} "
                  f"{100 * acc['tolerance']:.0f} %); per-bridge histogram KS "
                  f"p {_f(acc['hist_p'])} "
                  f"({'above' if acc['hist_ok'] else 'not above'} "
                  f"{acc['p_min']})")
        if acc.get("withheld"):
            flag("warn", "entanglement verdict withheld",
                 f"the relaxed file is {' and '.join(acc['withheld'])}, so "
                 f"these numbers are not a verdict on this config: {detail}")
        else:
            flag("note" if acc["met"] else "warn",
                 "entanglement " + ("matched" if acc["met"] else "not matched"),
                 detail)
    return flags


def write_verify(report: dict, path) -> Path:
    from topon.analysis.descriptors import to_jsonable

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=to_jsonable) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _pf(hist: dict) -> str:
    return ", ".join(f"{int(k)}:{v}" for k, v in sorted(
        ((int(k), v) for k, v in hist.items()), reverse=True))


def _f(x, fmt=".3f") -> str:
    if x is None:
        return "-"
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def format_verify(report: dict) -> str:
    """The verification for the terminal."""
    if report.get("architecture") == "crosslinked":
        from topon.inverse.crosslinked_verify import format_verify_crosslinked
        return format_verify_crosslinked(report)
    c, ref, s = report["config"], report["reference"], report["summary"]
    rows = report["seeds"]
    lines = [f"topon generate --verify  {ref['input']}", ""]
    lines.append(f"  config    : {c['lattice']} {c['lattice_size']}, cutoff "
                 f"{c['neighbour_cutoff']}, seeds "
                 + ", ".join(str(r["seed"]) for r in rows))
    req = {d: n for d, n in c["degree_distribution"].items() if d >= 1}
    if s["pf_exact_all"] is None:
        lines.append("  P(f)      : none requested (the topology is loaded, "
                     "not generated)")
    else:
        lines.append(f"  P(f)      : requested {_pf(req)}; "
                     + ("achieved exactly on every seed" if s["pf_exact_all"]
                        else "NOT achieved on every seed"))
    lines.append(f"  chemical  : {_pf(rows[0]['pf_chemical'])} against the "
                 f"reference's {_pf(ref['pf_chemical'])}")
    for key, label in (("primary_loops", "primary"), ("secondary_loops", "secondary"),
                       ("sol_chains", "sol")):
        e = s[key]
        lines.append(f"  {label:<10s}: requested {e['requested']}, placed "
                     f"{', '.join(str(a) for a in e['achieved'])}, reference "
                     f"{e['reference']}")
    b = s["beads"]
    lines.append(f"  beads     : {', '.join(str(x) for x in b['achieved'])} "
                 f"against {b['reference']}")
    if s["same_as_built"]:
        lines.append("  determinism: the regenerated graph "
                     + ("equals" if all(s["same_as_built"]) else "DIFFERS FROM")
                     + " the one the pipeline built")
    lines.append("")
    lines.append(f"  composite : {_f(s['composite_mean'])} +- "
                 f"{_f(s['composite_sd'])} over {s['n_seeds']} seed(s) "
                 f"(each term over the reference value; 0 is identical)")
    top = s.get("largest_term")
    if top and top.get("share") and top["share"] > 0.3:
        lines.append(f"              {top['term']} is {100 * top['share']:.0f} % "
                     f"of it (deviation {top['deviation']:.2f}); without it "
                     f"{top['composite_without']:.3f} +- "
                     f"{top['composite_without_sd']:.3f}")
    lines.append("  descriptor               reference   " + "  ".join(
        f"seed {r['seed']:<4d}" for r in rows))
    for k in _SHOWN:
        vals = "  ".join(f"{_f(r['descriptors'].get(k), '9.4f')}" for r in rows)
        lines.append(f"  {k:<24s} {_f(report['reference_descriptors'].get(k), '9.4f')}   {vals}")
    lines.append(f"  {'cycle JS divergence':<24s} {'':9s}   " + "  ".join(
        f"{_f(r['cycle_js'], '9.4f')}" for r in rows))
    lines.append(f"  {'betweenness KS':<24s} {'':9s}   " + "  ".join(
        f"{_f(r['ks_betweenness'], '9.4f')}" for r in rows))
    sp, rsp = s.get("spatial_built"), ref.get("spatial") or {}
    if sp and rsp.get("chord_mean"):
        lines.append("")
        lines.append(f"  reach     : built {sp['chord_mean']:.2f} +- "
                     f"{sp['chord_sd']:.2f} (p95 {sp['chord_p95']:.2f}) against "
                     f"the reference's {rsp['chord_mean']:.2f} +- "
                     f"{rsp['chord_sd']:.2f} (p95 {rsp['chord_p95']:.2f}) sigma; "
                     f"built state, before relaxation")
    rel = report.get("relaxed")
    if rel:
        lines += _format_relaxed(rel, rsp)
    if report["flags"]:
        lines.append("")
        for f in report["flags"]:
            lines.append(f"  [{f['level']}] {f['what']}: {f['detail']}")
    lines.append("")
    lines.append(f"  time      : {report['seconds']:.0f} s")
    return "\n".join(lines)


def _format_relaxed(rel: dict, rsp: dict) -> list:
    """The relaxed block of :func:`format_verify`.

    ``rsp`` is the reference's spatial record.
    """
    hdr = rel.get("header") or {}
    lines = ["", f"  relaxed   : {rel['input']}",
             f"              density {_f(rel.get('density'), '.4f')}, T "
             f"{_f(rel.get('temperature'))}, {rel.get('beads')} beads"
             + (f", timestep {hdr['timestep']}" if hdr.get("timestep") else "")
             + (f", {hdr['units']} units" if hdr.get("units") else "")]
    net = rel.get("network")
    if net:
        lines.append(f"    network   : "
                     + (f"NOT a build of this config (seed {net['seed']} "
                        f"compared)" if not net["same_network"] else
                        f"the build of seed {net['seed']}"
                        if net.get("same_graph") is not False else
                        f"a build of this config, not seed {net['seed']}'s graph")
                     + ("" if net["same_strands"] else
                        f" ({'; '.join(net['differences'])})"))
    for cls, k in rel["z_by_class"].items():
        if k.get("p") is None:
            lines.append(f"    Z {cls:<9s}: {_f(k.get('mean'))} over "
                         f"{k.get('n', 0)} against {_f(k.get('reference_mean'))}"
                         f" over {k.get('reference_n', 0)}, not compared")
        else:
            lines.append(f"    Z {cls:<9s}: {_f(k['mean'])} against "
                         f"{_f(k['reference_mean'])} ({k['n']} and "
                         f"{k['reference_n']} strands), KS p {_f(k['p'])}")
    h = rel.get("histogram")
    if h:
        lines.append("    P(Z)      :  Z  " + " ".join(
            f"{i:>6d}" for i in range(len(h["relaxed"]))))
        for side in ("relaxed", "reference"):
            lines.append(f"      {side:<10s}   " + " ".join(
                f"{x:6.3f}" for x in h[side]))
        lines.append(f"              KS p {_f(h['p'])} per bridge"
                     + (f"; {_f(rel['target_hist']['p'])} against target_hist"
                        if rel.get("target_hist") else ""))
    pp = rel.get("partners") or {}
    if (pp.get("pairs") or {}).get("measured") is not None:
        lines.append(
            f"    partners  : {pp['pairs']['measured']} pairs, "
            f"{_f(pp['per_strand']['measured'])} per strand, "
            f"{_f(pp['per_bridge']['measured'])} per bridge; reference "
            f"{pp['pairs']['reference']}, {_f(pp['per_strand']['reference'])}, "
            f"{_f(pp['per_bridge']['reference'])}")
    tz = rel.get("target_Z")
    if tz:
        lines.append(f"    target Z  : {tz['measured']:.3f} per bridge against "
                     f"the config's target_Z {tz['target']:.3f} "
                     f"({100 * tz['relative_gap']:+.1f} %)")
    rs = rel.get("spatial") or {}
    if rs.get("chord_mean") is not None and rsp.get("chord_mean"):
        lines.append(f"    reach     : {rs['chord_mean']:.2f} +- "
                     f"{rs['chord_sd']:.2f} (p95 {rs['chord_p95']:.2f}) "
                     f"against {rsp['chord_mean']:.2f} +- "
                     f"{rsp['chord_sd']:.2f} (p95 {rsp['chord_p95']:.2f}) "
                     f"sigma, after relaxation")
    if rs.get("chain_ree_mean") and rsp.get("chain_ree_mean"):
        lines.append(f"    Ree       : {rs['chain_ree_mean']:.2f} against "
                     f"{rsp['chain_ree_mean']:.2f} sigma (bridges)")
    acc = rel.get("acceptance")
    if acc:
        lines.append(
            f"    accept    : Z per bridge {100 * acc['relative_gap']:+.1f} % "
            f"(within {100 * acc['tolerance']:.0f} %: "
            f"{'yes' if acc['z_within'] else 'no'}), KS p {_f(acc['hist_p'])} "
            f"(above {acc['p_min']}: {'yes' if acc['hist_ok'] else 'no'}): "
            + (f"withheld ({', '.join(acc['withheld'])})"
               if acc.get("withheld") else "met" if acc["met"] else "NOT met"))
    return lines
