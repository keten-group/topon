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
    only when a relaxed data file is given (``relaxed``): Z1+ per strand
    class against the reference with KS p-values, the per-bridge histogram
    against ``conformation.entanglement.target_hist``, partners, and the
    chain statistics after relaxation. Producing that file needs MD, which
    ``topon generate`` does not run; ``tests/workflows/
    run_conformation_controller.py`` does, at the fitted build knob.
"""
from __future__ import annotations

import collections
import json
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
           log: Optional[Callable[[str], None]] = None) -> dict:
    """Regenerate ``config`` on ``seeds`` and compare it with ``reference``.

    ``built`` maps a seed to a graph already built (the one ``topon
    generate`` just made from the config's own seed); it is regenerated
    anyway and the two are compared, which is the determinism check.
    ``relaxed`` is an end-linked data file of the relaxed build, for the
    Z1+ part. ``measured`` is the reference's measurement when the caller
    already has it (with Z1+ per strand if ``relaxed`` is given), to save
    measuring it again. Returns the report as plain numbers.
    """
    from topon.analysis.descriptors import compare

    say = log or (lambda msg: None)
    t0 = time.perf_counter()
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
        report["relaxed"] = _relaxed(relaxed, config, ref_meas, z1_config, say)
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


def _relaxed(path, config, ref_meas, z1_config, say) -> dict:
    """Z1+ and chain statistics of a relaxed build against the reference."""
    from topon.conformation.entanglement import hist_ks

    rel = read_reference(path)
    if rel.source != "data":
        raise ValueError(f"{path}: the relaxed build has to be an end-linked "
                         f"LAMMPS data file")
    m = measure(rel, z1=True, z1_config=z1_config)
    rec = m.record
    out = {"input": str(path), "density": rec.get("density"),
           "temperature": (rec.get("spatial") or {}).get("temperature"),
           "spatial": rec.get("spatial"), "chains": rec["chains"]}
    rz = ref_meas.z_per_strand
    out["z_by_class"] = {cls: _ks(m.z_per_strand[cls], rz[cls])
                         for cls in ("bridge", "dangling", "loop")
                         if cls in m.z_per_strand and cls in rz}
    ent = (config.get("conformation") or {}).get("entanglement") or {}
    zb = m.z_per_strand.get("bridge")
    if zb is not None and ent.get("target_hist"):
        out["target_hist"] = hist_ks(zb, ent["target_hist"])
    if zb is not None and ent.get("target_Z"):
        tz = float(ent["target_Z"])
        out["target_Z"] = {"target": tz, "measured": float(np.mean(zb)),
                           "relative_gap": float((np.mean(zb) - tz) / tz)}
    z1 = rec.get("z1") or {}
    rz1 = ref_meas.record.get("z1") or {}
    out["partners_per_strand"] = {"measured": z1.get("partner_degree_mean"),
                                  "reference": rz1.get("partner_degree_mean")}
    say(f"  relaxed: Z per bridge "
        + (f"{out['target_Z']['measured']:.3f} against {out['target_Z']['target']:.3f}"
           if "target_Z" in out else "n/a"))
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
        for key, tol in STATE_TOLERANCE.items():
            a, b = rel.get(key), (rrec.get(key) if key == "density"
                                  else (rrec.get("spatial") or {}).get(key))
            if a and b and abs(a - b) / b > tol:
                flags.append({"level": "warn", "what": f"unmatched {key}",
                              "detail": f"relaxed {a:.4g} against the "
                                        f"reference's {b:.4g}; Z1+ compares "
                                        f"only at matched density and "
                                        f"temperature"})
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
        lines.append("")
        lines.append(f"  relaxed   : {rel['input']} (density "
                     f"{_f(rel.get('density'), '.4f')}, T {_f(rel.get('temperature'))})")
        for cls, k in rel["z_by_class"].items():
            lines.append(f"    Z {cls:<9s}: {_f(k['mean'])} against "
                         f"{_f(k['reference_mean'])}, KS p {_f(k['p'])}")
        if rel.get("target_hist"):
            lines.append(f"    histogram : KS p {_f(rel['target_hist']['p'])} "
                         f"against target_hist")
        pp = rel.get("partners_per_strand") or {}
        if pp.get("measured") is not None:
            lines.append(f"    partners  : {_f(pp['measured'])} per strand "
                         f"against {_f(pp['reference'])}")
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
    if report["flags"]:
        lines.append("")
        for f in report["flags"]:
            lines.append(f"  [{f['level']}] {f['what']}: {f['detail']}")
    lines.append("")
    lines.append(f"  time      : {report['seconds']:.0f} s")
    return "\n".join(lines)
