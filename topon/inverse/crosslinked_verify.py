"""``topon generate --verify`` for a network crosslinked along its chains.

Each seed is built through the pipeline's stages 1 to 3 and held against
the reference on the measures :mod:`topon.inverse.crosslinked` fits by:

requested against achieved
    the chains (count and length) and the crosslink count for the crosslink
    generator, the P(f) for the lattice route
connectivity
    the descriptor composite (each term over the reference value), lambda2
    and the mean path of the core
chains
    the KS statistic of passes per chain and of the bridge, dangling and
    loop DPs, and the counts: crosslinks within a chain, loops, secondary
    loops, dangling strands, sol chains, junctions outside the largest piece

With replicates of the reference (the same process at other seeds) each of
those measures gets two readings. The scatter is the furthest a replicate
sits from the reference (by the composite and the KS statistics, or by the
distance of lambda2 and the path from the reference's), and each build is
inside it or not. A build of the reference's own process falls outside that
with a chance of 1 in N + 1 for N replicates, so a measure is flagged only
when more builds fall outside than that chance explains (binomial tail
below 0.05). The second reading is a Mann-Whitney test of the builds
against the replicates per measure, with at least :data:`MW_MIN` a side
(below that the test cannot reach 0.05), the verdict taken after a Holm
correction over the measures.

No MD and no Z1+: a crosslinked reference has no strand classes to read Z
per strand by, so ``relaxed`` is refused.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from topon.inverse.crosslinked import CHAIN_KEYS, reference_numbers, score_graph
from topon.inverse.measure import Measurement, measure, read_reference
from topon.inverse.scaffold import build_graph

#: The measures a build is held inside the replicates' scatter on.
MEASURES = ("composite", "lambda2", "path") + tuple(f"ks_{k}" for k in CHAIN_KEYS)
#: Fewest builds and replicates a side for a Mann-Whitney test (three a side
#: cannot reach a two-sided p of 0.05).
MW_MIN = 4
#: Chance below which more builds outside the scatter than a build of the
#: reference's own process gives is flagged.
OUTSIDE_P = 0.05
#: Counts listed per build beside the reference's.
COUNTS = ("crosslinks", "intra", "loops", "secondary", "bridges", "dangling",
          "sol", "junctions", "detached")


def is_crosslinked_config(config: dict) -> bool:
    """A config that builds a network crosslinked along its chains."""
    topo = config.get("topology") or {}
    return (topo.get("source") == "crosslink"
            or (topo.get("generator") or {}).get("architecture")
            == "random_crosslinked")


def _requested(config: dict) -> dict:
    topo = config.get("topology") or {}
    if topo.get("source") == "crosslink":
        x = topo.get("crosslinking") or {}
        return {"route": "crosslink",
                "chains": [{"count": c.get("count"), "dp": c.get("dp"),
                            "sequence": c.get("sequence")}
                           for c in x.get("chains") or []],
                "crosslinks": x.get("crosslinks"),
                "conversion": x.get("conversion"),
                "per_chain": x.get("per_chain"),
                "packing": x.get("packing"),
                "contact_radius": x.get("contact_radius"),
                "min_gap": x.get("min_gap")}
    from topon.topology.degree_matching import parse_degree_distribution

    gen = topo.get("generator") or {}
    counts, _ = parse_degree_distribution(gen.get("degree_distribution"))
    chains = (config.get("assignment") or {}).get("chains") or {}
    return {"route": "lattice", "lattice": gen.get("lattice_type"),
            "lattice_size": gen.get("lattice_size"),
            "neighbour_cutoff": gen.get("neighbour_cutoff"),
            "degree_distribution": {int(d): int(n) for d, n in sorted(counts.items())},
            "chain_dp": chains.get("dp"),
            "reactive_every": chains.get("reactive_every")}


def _achieved(G, man: dict, requested: dict) -> dict:
    if requested["route"] == "crosslink":
        rec = man.get("crosslinking") or G.graph.get("crosslinking") or {}
        lengths = sorted(int(c["dp"]) for c in (G.graph.get("chains") or {}).values())
        want = sorted(int(c["dp"]) for c in _expanded(requested["chains"], G))
        return {"chains": len(lengths), "crosslinks": rec.get("crosslinks"),
                "lengths_exact": lengths == want if want else None,
                "crosslinks_exact": (rec.get("crosslinks") == requested["crosslinks"]
                                     if requested.get("crosslinks") is not None
                                     else None),
                "lattice": rec.get("lattice"), "packing": rec.get("packing"),
                "c_inf": rec.get("c_inf"),
                "strands_rewound": rec.get("strands_rewound")}
    sculpt = man.get("sculpt") or {}
    achieved = {int(k): int(v) for k, v in (sculpt.get("achieved") or {}).items()}
    req = requested["degree_distribution"]
    return {"pf_achieved": achieved,
            "pf_exact": (all(achieved.get(d, 0) == n for d, n in req.items() if d >= 1)
                         if req else None),
            "chains": len(G.graph.get("chains") or {})}


def _expanded(chains: list, G) -> list:
    """One entry per chain of the requested types (a sequence's length from G)."""
    out = []
    for c in chains:
        if c.get("dp") is None:
            return []                    # a sequence: its length is the plan's
        out += [{"dp": c["dp"]}] * int(c.get("count") or 0)
    return out


def scatter(rows: list, refn: dict) -> dict:
    """The furthest a replicate sits from the reference, per measure."""
    out = {}
    for k in MEASURES:
        if k in ("lambda2", "path"):
            vals = [abs(r[k] - refn[k]) for r in rows
                    if r.get(k) is not None and refn.get(k) is not None]
        else:
            vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(max(vals)) if vals else None
    return out


def inside(row: dict, sc: dict, refn: dict) -> dict:
    """Which measures of a build sit inside the replicates' scatter.

    None where either side has nothing to compare (no loops, say).
    """
    out = {}
    for k in MEASURES:
        lim, x = sc.get(k), row.get(k)
        if lim is None or x is None:
            out[k] = None
        elif k in ("lambda2", "path"):
            out[k] = bool(abs(x - refn[k]) <= lim + 1e-12)
        else:
            out[k] = bool(x <= lim + 1e-12)
    return out


def verify_crosslinked(config: dict, reference, *, seeds: Sequence[int],
                       built: Optional[dict] = None,
                       replicates: Optional[Sequence] = None,
                       measured: Optional[Measurement] = None, relaxed=None,
                       log: Optional[Callable[[str], None]] = None) -> dict:
    """Regenerate a crosslinked ``config`` on ``seeds`` against ``reference``.

    ``replicates`` are more networks of the reference's own process (paths
    read as the reference is); with any, the report gives their scatter and
    says per build and measure whether it sits inside. ``built`` and
    ``measured`` are as for :func:`topon.inverse.verify.verify`.

    Raises:
        ValueError: a reference that does not read crosslinked, or
            ``relaxed`` (no Z1+ for these networks).
    """
    from topon.inverse.verify import _same_graph

    say = log or (lambda msg: None)
    if relaxed is not None:
        raise ValueError("--relaxed reads Z1+ per strand class of an end-linked "
                         "network; a network crosslinked along its chains has "
                         "no such target")
    t0 = time.perf_counter()
    ref = read_reference(reference, crosslinked=True)
    ref_meas = measured if measured is not None else measure(ref, z1=False)
    rrec = ref_meas.record
    refn = reference_numbers(ref_meas, ref.graph)
    requested = _requested(config)
    say(f"reference measured in {time.perf_counter() - t0:.0f} s")

    reps = []
    for path in replicates or ():
        t1 = time.perf_counter()
        rr = read_reference(path, crosslinked=True)
        rm = measure(rr, z1=False)
        row = {"input": str(path), **score_graph(rr.graph, refn,
                                                 desc=rm.descriptors)}
        row["seconds"] = round(time.perf_counter() - t1, 1)
        reps.append(row)
        say(f"  replicate {Path(path).name}: composite {row['composite']:.3f}")
    sc = scatter(reps, refn) if reps else None

    density = (config.get("chemistry") or {}).get("target_density")
    rows = []
    for s in seeds:
        t1 = time.perf_counter()
        G, man = build_graph(config, seed=s)
        row = {"seed": int(s),
               "same_as_built": (_same_graph(built[s], G)
                                 if built and s in built else None),
               **_achieved(G, man, requested), **score_graph(G, refn),
               "reach": reach(G, rrec.get("units"), density, requested["route"])}
        if sc is not None:
            row["inside"] = inside(row, sc, refn)
        row["seconds"] = round(time.perf_counter() - t1, 1)
        rows.append(row)
        say(f"  seed {s}: composite {_f(row['composite'])}, lambda2 "
            f"{_f(row['lambda2'], '.4f')}, path {_f(row['path'], '.2f')} "
            f"({row['seconds']:.0f} s)")

    report = {
        "architecture": "crosslinked",
        "config": requested,
        "reference": {k: rrec.get(k) for k in (
            "input", "source", "units", "box", "n_atoms", "density", "chains",
            "crosslinked", "giant_fraction", "spatial", "notes")},
        "reference_numbers": {"composite": 0.0, "lambda2": refn["lambda2"],
                              "path": refn["path"], "loops": refn["loops"],
                              "intra": refn["intra"]},
        "reference_counts": _counts(rrec),
        "replicates": reps,
        "scatter": sc,
        "seeds": rows,
        "summary": _summary(rows, reps, sc, refn),
    }
    report["flags"] = _flags(report, rrec)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


def reach(G, units: Optional[str], density, route: str) -> Optional[dict]:
    """Junction separation of the bridges as built, in the reference's units.

    The crosslink generator's graph is in lattice units, which a lattice
    reference shares; for a reference in sigma the positions are scaled to
    the box ``density`` gives the beads the builder makes (one per crosslink
    fewer, since a crosslink is built as one bead). The lattice route's cell
    is no lattice a reference shares, so it is compared in sigma only. The
    junction sits at its two beads' midpoint, which was found to put the mean
    3 to 6 % below a reference that puts it on one of them.
    """
    from topon.analysis.descriptors import spatial
    from topon.inverse.crosslinked import crosslink_count

    box = G.graph.get("box")
    if box is None or units not in ("lattice", "sigma"):
        return None
    if units == "lattice" and route != "crosslink":
        return None
    try:
        sp, _chords = spatial(G)
    except ValueError:
        return None
    scale = 1.0
    if units == "sigma":
        if not density:
            return None
        beads = (sum(int(c["dp"]) for c in (G.graph.get("chains") or {}).values())
                 - crosslink_count(G)["count"])
        scale = (beads / float(density)) ** (1.0 / 3.0) / float(np.mean(box))
    return {k: float(sp[k]) * scale for k in ("chord_mean", "chord_p95")
            if sp.get(k) is not None}


def _counts(rrec: dict) -> dict:
    s = rrec["descriptors"]
    xl = rrec.get("crosslinked") or {}
    return {"crosslinks": (xl.get("crosslinks") or {}).get("count"),
            "intra": (xl.get("crosslinks") or {}).get("intra_chain"),
            "loops": s.get("n_primary_loops"),
            "secondary": s.get("n_secondary_loops"),
            "bridges": s.get("n_bridging_chains"),
            "dangling": s.get("n_dangling_chains"),
            "sol": s.get("n_sol_chains"),
            "junctions": s.get("n_junctions"),
            "detached": 1.0 - float(s.get("giant_frac_junctions", 1.0) or 1.0)}


def _mean_sd(vals) -> Optional[list]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return [float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0]


def mann_whitney(rows, reps, refn) -> dict:
    """Two-sided Mann-Whitney p of the builds against the replicates, per measure.

    The crosslink generator's own check: the composite and the KS statistics are distances
    from the reference, so the builds' are held against the replicates';
    lambda2 and the path are values, held against the replicates' and the
    reference's together. None with fewer than :data:`MW_MIN` a side.
    """
    from scipy import stats

    out = {}
    for k in MEASURES:
        a = [r[k] for r in rows if r.get(k) is not None]
        b = [r[k] for r in reps if r.get(k) is not None]
        if k in ("lambda2", "path") and refn.get(k) is not None:
            b.append(refn[k])
        if len(a) < MW_MIN or len(b) < MW_MIN:
            out[k] = None
            continue
        out[k] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return out


def _summary(rows, reps, sc, refn) -> dict:
    out = {"n_seeds": len(rows),
           "same_as_built": [r["same_as_built"] for r in rows
                             if r["same_as_built"] is not None],
           "measures": {k: _mean_sd(r.get(k) for r in rows) for k in MEASURES},
           "counts": {k: _mean_sd(r.get(k) for r in rows) for k in COUNTS}}
    reached = [r["reach"] for r in rows if r.get("reach")]
    if reached:
        out["reach"] = {k: _mean_sd(x.get(k) for x in reached)
                        for k in ("chord_mean", "chord_p95")}
    if reps:
        out["replicate_measures"] = {k: _mean_sd(r.get(k) for r in reps)
                                     for k in MEASURES}
        out["replicate_counts"] = {k: _mean_sd(r.get(k) for r in reps)
                                   for k in COUNTS}
        out["mann_whitney"] = mann_whitney(rows, reps, refn)
        out["mann_whitney_holm"] = holm(out["mann_whitney"])
        ps = [p for p in out["mann_whitney_holm"].values() if p is not None]
        out["no_difference"] = (min(ps) >= 0.05) if ps else None
    if sc is not None:
        from scipy import stats

        chance = 1.0 / (len(reps) + 1)
        out["inside"], out["outside_chance"] = {}, {}
        for k in MEASURES:
            v = [r["inside"][k] for r in rows if r["inside"][k] is not None]
            out["inside"][k] = [int(sum(v)), len(v)] if v else None
            if v:
                n_out = len(v) - int(sum(v))
                out["outside_chance"][k] = float(stats.binom.sf(n_out - 1, len(v), chance))
        out["outside_expected"] = round(len(rows) * chance, 2)
        judged = [v for v in out["inside"].values() if v]
        out["all_inside"] = bool(judged) and all(a == n for a, n in judged)
        out["outside_by_chance"] = all(p >= OUTSIDE_P
                                       for p in out["outside_chance"].values())
    return out


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values of a family of tests, None where a test is None."""
    keys = sorted((k for k, p in pvalues.items() if p is not None),
                  key=lambda k: pvalues[k])
    m = len(keys)
    out, running = {k: None for k in pvalues}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = float(running)
    return out


def _flags(report, rrec) -> list:
    flags = []
    rows = report["seeds"]
    req = report["config"]
    if req["route"] == "crosslink":
        bad = [r["seed"] for r in rows if r.get("crosslinks_exact") is False
               or r.get("lengths_exact") is False]
        if bad:
            flags.append({"level": "error", "what": "chains or crosslinks not exact",
                          "detail": f"seeds {bad} did not build the requested "
                                    f"chains and crosslink count"})
    elif any(r.get("pf_exact") is False for r in rows):
        flags.append({"level": "error", "what": "P(f) not exact",
                      "detail": "seeds " + str([r["seed"] for r in rows
                                                if r.get("pf_exact") is False])
                                + " missed the requested degree counts"})
    same = report["summary"]["same_as_built"]
    if same and not all(same):
        flags.append({"level": "error", "what": "not deterministic",
                      "detail": "the regenerated graph differs from the one the "
                                "pipeline built from the same seed"})
    s = report["summary"]
    ins = s.get("inside")
    if ins:
        out = {k: v for k, v in ins.items()
               if v and (s["outside_chance"].get(k) or 1.0) < OUTSIDE_P}
        if out:
            flags.append({"level": "warn",
                          "what": "outside the replicates' scatter",
                          "detail": ", ".join(
                              f"{k} {v[1] - v[0]} of {v[1]} builds (chance "
                              f"{s['outside_chance'][k]:.3f})"
                              for k, v in out.items())
                          + f", where a build of the reference's own process "
                            f"falls outside in 1 of {len(report['replicates']) + 1}"})
        few = min(len(rows), len(report["replicates"]))
        if few < MW_MIN:
            flags.append({"level": "note", "what": "too few for a Mann-Whitney test",
                          "detail": f"{len(rows)} builds and "
                                    f"{len(report['replicates'])} replicates; the "
                                    f"test needs {MW_MIN} a side to reach 0.05"})
        mw = s.get("mann_whitney_holm") or {}
        diff = {k: p for k, p in mw.items() if p is not None and p < 0.05}
        if diff:
            flags.append({"level": "warn", "what": "a Mann-Whitney difference",
                          "detail": ", ".join(f"{k} (Holm p {p:.3f})"
                                              for k, p in diff.items())})
    elif not report["replicates"]:
        flags.append({"level": "note", "what": "no replicates",
                      "detail": "pass replicates of the reference (--verify-replicate, "
                                "the same process at other seeds) to hold the "
                                "builds against its own scatter"})
    ref_loops = report["reference_counts"].get("loops")
    if not ref_loops and any(r.get("loops") for r in rows):
        flags.append({"level": "note", "what": "loops",
                      "detail": "the builds have primary loops and the reference "
                                "none, so the loop-DP KS cannot be read"})
    for note in rrec.get("notes", []):
        flags.append({"level": "note", "what": "reference", "detail": note})
    return flags


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _f(x, fmt=".3f") -> str:
    if x is None:
        return "-"
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def format_verify_crosslinked(report: dict) -> str:
    """The verification of a crosslinked build for the terminal."""
    c, ref, s = report["config"], report["reference"], report["summary"]
    rows, reps, sc = report["seeds"], report["replicates"], report["scatter"]
    lines = [f"topon generate --verify  {ref['input']}  (crosslinked along its "
             f"chains)", ""]
    if c["route"] == "crosslink":
        lines.append(f"  config    : crosslink generator, {c['crosslinks']} "
                     f"crosslinks, packing {c['packing']}, contact "
                     f"{c['contact_radius']}, min_gap {c['min_gap']}; seeds "
                     + ", ".join(str(r["seed"]) for r in rows))
        exact = all(r.get("crosslinks_exact") is not False
                    and r.get("lengths_exact") is not False for r in rows)
        lines.append("  requested : chains and crosslinks "
                     + ("built exactly on every seed" if exact
                        else "NOT built exactly on every seed"))
    else:
        lines.append(f"  config    : lattice route, {c['lattice']} "
                     f"{c['lattice_size']}, cutoff {c['neighbour_cutoff']}, "
                     f"chains of {c['chain_dp']}; seeds "
                     + ", ".join(str(r["seed"]) for r in rows))
        lines.append("  P(f)      : "
                     + ("achieved exactly on every seed"
                        if all(r.get("pf_exact") for r in rows)
                        else "NOT achieved on every seed"))
    if s["same_as_built"]:
        lines.append("  determinism: the regenerated graph "
                     + ("equals" if all(s["same_as_built"]) else "DIFFERS FROM")
                     + " the one the pipeline built")
    rn = report["reference_numbers"]
    lines.append("")
    head = f"  {'measure':<16s} {'reference':>10s} {'builds':>17s}"
    if reps:
        head += (f" {'replicates':>17s} {'scatter':>9s} {'inside':>7s} "
                 f"{'MW p':>6s} {'Holm':>6s}")
    lines.append(head)
    for k in MEASURES:
        b = s["measures"].get(k)
        refv = rn.get(k) if k in ("lambda2", "path", "composite") else None
        line = (f"  {k:<16s} {_f(refv, '10.4f') if refv is not None else '':>10s} "
                f"{(_f(b[0], '.4f') + ' +- ' + _f(b[1], '.4f')) if b else '-':>17s}")
        if reps:
            rv = s["replicate_measures"].get(k)
            ins = s["inside"].get(k)
            line += (f" {(_f(rv[0], '.4f') + ' +- ' + _f(rv[1], '.4f')) if rv else '-':>17s}"
                     f" {_f(sc.get(k), '9.4f')} "
                     f"{(str(ins[0]) + '/' + str(ins[1])) if ins else '-':>7s} "
                     f"{_f(s['mann_whitney'].get(k), '6.2f'):>6s} "
                     f"{_f(s['mann_whitney_holm'].get(k), '6.2f'):>6s}")
        lines.append(line)
    lines.append("")
    rc = report["reference_counts"]
    lines.append(f"  {'count':<16s} {'reference':>10s} {'builds':>17s}"
                 + (f" {'replicates':>17s}" if reps else ""))
    for k in COUNTS:
        b = s["counts"].get(k)
        fmt = ".3f" if k == "detached" else ".1f"
        line = (f"  {k:<16s} {_f(rc.get(k), '10' + fmt)} "
                f"{(_f(b[0], fmt) + ' +- ' + _f(b[1], fmt)) if b else '-':>17s}")
        if reps:
            rv = s["replicate_counts"].get(k)
            line += f" {(_f(rv[0], fmt) + ' +- ' + _f(rv[1], fmt)) if rv else '-':>17s}"
        lines.append(line)
    rsp = ref.get("spatial") or {}
    if s.get("reach") and rsp.get("chord_mean") is not None:
        b = s["reach"]
        lines.append("")
        lines.append(f"  reach     : built {_f(b['chord_mean'][0], '.2f')} +- "
                     f"{_f(b['chord_mean'][1], '.2f')} (p95 "
                     f"{_f(b['chord_p95'][0], '.2f')}) against the reference's "
                     f"{_f(rsp['chord_mean'], '.2f')} (p95 "
                     f"{_f(rsp.get('chord_p95'), '.2f')}) {ref.get('units')} "
                     f"units; built state, a junction at its beads' midpoint")
    if s.get("all_inside") is not None:
        lines.append("")
        lines.append("  verdict   : " + (
            "every build inside the replicates' scatter on every measure"
            if s["all_inside"] else
            f"builds outside the replicates' scatter no more often than a build "
            f"of the reference's own process would be ({s['outside_expected']:g} "
            f"a measure expected)" if s["outside_by_chance"] else
            "more builds outside the replicates' scatter than chance explains on "
            "some measure (see the flags)"))
        if s.get("no_difference") is not None:
            lines.append("              " + (
                "no Mann-Whitney difference from the replicates on any measure "
                "(Holm-adjusted p >= 0.05)" if s["no_difference"] else
                "a Mann-Whitney difference from the replicates on some measure "
                "(Holm-adjusted p < 0.05)"))
    if report["flags"]:
        lines.append("")
        for f in report["flags"]:
            lines.append(f"  [{f['level']}] {f['what']}: {f['detail']}")
    lines.append("")
    lines.append(f"  time      : {report['seconds']:.0f} s")
    return "\n".join(lines)


__all__ = ["format_verify_crosslinked", "is_crosslinked_config",
           "verify_crosslinked"]
