"""Post-run summary for a `topon generate` output directory.

Used by `topon inspect <run_dir>`. Parses each stage's outputs and
prints a one-screen status report: atom counts, box, what artifacts
landed, what the next LAMMPS commands are.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from topon.core.manifest import read_manifest


_DISPLACE_KINDS = (
    "system_nodes.displace",
    "system_backbone.displace",
    "system_beads.displace",     # legacy pre-Option-C
    "system_grafts.displace",
    "system_pendant.displace",
    "system_hydrogens.displace",
)


@dataclass
class StageReport:
    name: str
    present: bool
    files: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class RunSummary:
    root: Path
    chemistry: StageReport
    conformation: StageReport
    simulation: StageReport
    atom_count: Optional[int] = None
    n_atom_types: Optional[int] = None
    box: Optional[tuple[float, float, float]] = None
    #: Stage 1's section of ``manifest.json``, when the run wrote one.
    topology: Optional[dict] = None
    #: Stage 3's defects section of the same manifest.
    defects: Optional[dict] = None
    #: What the furthest end-linked data file says (see :func:`measure_network`).
    network: Optional[dict] = None


def _parse_system_data(path: Path) -> dict:
    """Extract atom_count, n_atom_types, box from a LAMMPS data file header."""
    info: dict = {"atom_count": None, "n_atom_types": None, "box": None}
    if not path.exists():
        return info
    box_lo = [None, None, None]
    box_hi = [None, None, None]
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i > 80:
                break
            s = line.strip()
            m = re.match(r"^(\d+)\s+atoms\s*$", s)
            if m:
                info["atom_count"] = int(m.group(1))
                continue
            m = re.match(r"^(\d+)\s+atom\s+types\s*$", s)
            if m:
                info["n_atom_types"] = int(m.group(1))
                continue
            m = re.match(r"^([-\d.eE+]+)\s+([-\d.eE+]+)\s+([xyz])lo\s+\3hi\s*$", s)
            if m:
                idx = {"x": 0, "y": 1, "z": 2}[m.group(3)]
                box_lo[idx] = float(m.group(1))
                box_hi[idx] = float(m.group(2))
    if all(lo is not None and hi is not None for lo, hi in zip(box_lo, box_hi)):
        info["box"] = (
            box_hi[0] - box_lo[0],
            box_hi[1] - box_lo[1],
            box_hi[2] - box_lo[2],
        )
    return info


def _chemistry_summary(d: Path) -> StageReport:
    files = sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []
    if not files:
        return StageReport("02_Chemistry", present=False)
    data = _parse_system_data(d / "system.data")
    displace_present = [n for n in _DISPLACE_KINDS if n in files]
    bits = []
    if data["atom_count"]:
        bits.append(f"{data['atom_count']} atoms")
    if data["n_atom_types"]:
        bits.append(f"{data['n_atom_types']} atom types")
    if data["box"]:
        lx, ly, lz = data["box"]
        bits.append(f"box {lx:.1f} x {ly:.1f} x {lz:.1f} A")
    bits.append(f"{len(displace_present)} displace file(s): "
                f"{', '.join(d.replace('system_', '').replace('.displace', '') for d in displace_present)}")
    return StageReport(
        name="02_Chemistry",
        present=True,
        files=files,
        summary="; ".join(bits),
    )


def _conformation_summary(d: Path) -> StageReport:
    files = sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []
    if not files:
        return StageReport("03_Conformation", present=False)
    bits = []
    for name, label in (
        ("system_conformed.data", "conformed"),
        ("system_relaxed.data", "relaxed"),
    ):
        if name in files:
            data = _parse_system_data(d / name)
            if data["atom_count"]:
                bits.append(f"{label} ({data['atom_count']} atoms)")
            else:
                bits.append(label)
    return StageReport(
        name="03_Conformation",
        present=True,
        files=files,
        summary="; ".join(bits) if bits else f"{len(files)} file(s)",
    )


#: The last checkpoint of each protocol that says a run got that far, newest
#: stage first. Both families are listed because a run directory may have been
#: produced by either: the push-off (the coarse-grained default since 0.2.0) writes
#: the stage-named files the end-linked validation scripts read, and the two
#: minimiser decks write the historic names.
_FURTHEST_STAGE = (
    ("stage6_quartic.data", "stage 6 done (converted to the quartic bond)"),
    ("stage5_final_quench.data", "stage 5 done (quenched)"),
    ("stage4_final_T1.data", "stage 4 done (compressed and settled)"),
    ("stage3_build_equil.data", "stage 3 done (equilibrated at the build density)"),
    ("stage2_pushoff.data", "stage 2 done (push-off finished)"),
    ("stage1_min.data", "stage 1 done (push-off)"),
    ("system_equilibrated.data", "equilibrated.data present"),
    ("system_minimized_final.data", "minimized_final.data present (stage 3 minimize done)"),
    ("system_ramped.data", "ramped.data present (stage 2 done)"),
    ("system_after_soft.data", "after_soft.data present (stage 1 done)"),
)


def _furthest_stage(files) -> str | None:
    """How far a run in this directory got, or None if nothing says."""
    have = set(files)
    for name, said in _FURTHEST_STAGE:
        if name in have:
            return said
    return None


#: Script names in the order they must be run. Sorting alphabetically used to
#: give the same answer, because the three minimiser stages were numbered
#: inside a common prefix; with `deform_4` and `quench_5` beside them it puts
#: the compression first, and following that list runs the protocol backwards.
_RUN_ORDER = ("minimize_1_serial.in", "minimize_2_parallel.in",
              "minimize_3_parallel.in", "deform_4_parallel.in",
              "quench_5_parallel.in", "convert_6_parallel.in")


def _in_run_order(names) -> list:
    """Known scripts first, in protocol order; anything else after, sorted."""
    have = set(names)
    known = [n for n in _RUN_ORDER if n in have]
    return known + sorted(have - set(known))


def _simulation_summary(d: Path) -> StageReport:
    files = sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []
    if not files:
        return StageReport("04_Simulation", present=False)
    scripts = [f for f in files if f.endswith(".in")]
    logs = [f for f in files if f.endswith(".lammps") or f.startswith("log.")]
    data_files = [f for f in files if f.endswith(".data")]
    bits = [
        f"{len(scripts)} LAMMPS script(s)",
        f"{len(logs)} log(s)",
        f"{len(data_files)} stage output data file(s)",
    ]
    furthest = _furthest_stage(files)
    if furthest:
        bits.append(furthest)
    return StageReport(
        name="04_Simulation",
        present=True,
        files=files,
        summary="; ".join(bits),
    )


#: Data files a network is read from, the most relaxed first: every protocol
#: checkpoint, then the conformation stage's output.
_NETWORK_FILES = tuple(name for name, _said in _FURTHEST_STAGE) + (
    "system_relaxed.data",)


def network_file(root: Path, flat: bool = False) -> Optional[Path]:
    """The most relaxed data file of a run, or None."""
    dirs = [root] if flat else [root / "04_Simulation", root / "03_Conformation"]
    for name in _NETWORK_FILES:
        for d in dirs:
            if (d / name).exists():
                return d / name
    return None


def measure_network(path: Path, z1: bool = True, z1_config=None) -> dict:
    """Strands, loops and P(f) read from a data file, and Z1+ when it runs.

    An end-linked file (``output.lammps_convention: "endlinked"``, or a
    `fix bond/create` dataset) is read into strands by its atom types, and an
    atomistic file of a run through the strand record in the run's manifest
    (:mod:`topon.analysis.atomistic`); any other file comes back with a
    ``note`` saying so. Z1+ is skipped with a ``z1_note`` when it is not
    installed.
    """
    from topon.analysis.atomistic import NoStrandRecord, find_manifest, read_atomistic
    from topon.analysis.descriptors import strand_counts
    from topon.analysis.endlinked import NotEndLinked, read_endlinked

    path = Path(path)
    out: dict = {"file": str(path)}
    note = ("not in the end-linked convention, so its strands cannot be read "
            "(a coarse-grained build written with output.lammps_convention "
            "\"endlinked\" can be, and an atomistic one with its run manifest)")
    system = None
    if find_manifest(path) is not None:
        try:
            system = read_atomistic(path)
        except NoStrandRecord as exc:
            # a record another run of the study left; read the file as it is
            out["record_note"] = str(exc)
    atomistic = system is not None
    if not atomistic:
        # An atomistic file has far more than three atom types; say so from
        # the header rather than parse a large file to find out.
        if (_parse_system_data(path)["n_atom_types"] or 0) > 3:
            out["note"] = note
            return out
        try:
            system = read_endlinked(path)
        except NotEndLinked:
            out["note"] = note
            return out
    out["classes"] = system.class_counts()
    if atomistic:
        out["density"] = system.mass_density
        out["units"] = "real"
    else:
        out["density"] = system.density
    t = system.temperature()
    if t is not None:
        out["temperature"] = t
    out.update(strand_counts(system.graph))
    if z1:
        from topon.analysis.z1plus import (
            Z1PlusFailed, Z1PlusUnavailable, measure_system)
        try:
            out["z1"] = measure_system(system, config=z1_config)[0]
        except (Z1PlusUnavailable, Z1PlusFailed) as exc:
            out["z1_note"] = str(exc)
    return out


def summarise(run_dir: Path, network: bool = False, z1: bool = False,
              z1_config=None) -> RunSummary:
    """Read a run directory back.

    With ``network`` the most relaxed data file is also read into its strands
    (:func:`measure_network`), and with ``z1`` measured with Z1+ as well;
    `topon inspect` asks for both. Off by default, since a large data file
    takes seconds to parse and Z1+ is an external program.
    """
    run_dir = Path(run_dir)

    # Layout A: study root containing 02_Chemistry/ + 03_Conformation/ + 04_Simulation/
    # Layout B: parent of A — find the nested study dir
    # Layout C: flat folder (expected_output/-style): every artifact at top level
    if (run_dir / "02_Chemistry").exists():
        root = run_dir
        flat = False
    else:
        nested = list(run_dir.glob("*/02_Chemistry"))
        if nested:
            root = nested[0].parent
            flat = False
        else:
            root = run_dir
            flat = True

    if not flat:
        chem = _chemistry_summary(root / "02_Chemistry")
        conf = _conformation_summary(root / "03_Conformation")
        sim = _simulation_summary(root / "04_Simulation")
        info = _parse_system_data(root / "02_Chemistry" / "system.data")
    else:
        # Flat: split files by name pattern into the three stage buckets.
        chem = _chemistry_summary(root)
        chem.name = "02_Chemistry (flat)"
        # No separate conformation/simulation buckets; report what's there.
        files = sorted(p.name for p in root.iterdir() if p.is_file())
        scripts = [f for f in files if f.endswith(".in")]
        logs = [f for f in files if f.endswith(".lammps") or f.startswith("log.")]
        stage_data = [f for f in files if f.endswith(".data")
                      and f not in ("system.data", "system_conformed.data",
                                    "system_relaxed.data")]
        sim_bits = []
        if scripts:
            sim_bits.append(f"{len(scripts)} LAMMPS script(s)")
        if logs:
            sim_bits.append(f"{len(logs)} log(s)")
        if stage_data:
            sim_bits.append(f"{len(stage_data)} stage data file(s)")
        furthest = _furthest_stage(files)
        if furthest:
            sim_bits.append(furthest)
        sim = StageReport("04_Simulation (flat)", present=bool(scripts or stage_data),
                          files=scripts + logs + stage_data,
                          summary="; ".join(sim_bits) if sim_bits else "(no LAMMPS files)")
        conf_present = any(
            n in files for n in ("system_conformed.data", "system_relaxed.data")
        )
        conf = StageReport(
            "03_Conformation (flat)",
            present=conf_present,
            files=[n for n in ("system_conformed.data", "system_relaxed.data") if n in files],
            summary="conformed+relaxed present" if conf_present else "(none)",
        )
        info = _parse_system_data(root / "system.data")

    # The run manifest sits next to the stage folders. Look in the study
    # root first, then in the directory the user pointed at, so
    # `topon inspect` finds it whether they named the study or its parent.
    manifest = read_manifest(root) or read_manifest(run_dir) or {}

    net = None
    f = network_file(root, flat) if network else None
    if f is not None:
        net = measure_network(f, z1=z1, z1_config=z1_config)

    return RunSummary(
        root=root,
        chemistry=chem,
        conformation=conf,
        simulation=sim,
        atom_count=info.get("atom_count"),
        n_atom_types=info.get("n_atom_types"),
        box=info.get("box"),
        topology=(manifest.get("stages") or {}).get("topology"),
        defects=(manifest.get("stages") or {}).get("defects"),
        network=net,
    )


def format_network(net: dict) -> list[str]:
    """Render :func:`measure_network`'s record."""
    from topon.analysis.analyze import format_pf, format_z1

    lines = ["", f"  Network (from {Path(net['file']).name}):"]
    if net.get("note"):
        lines.append(f"    {net['note']}")
        return lines
    cls = net.get("classes", {})
    real = net.get("units") == "real"
    state = (f"density {net['density']:.4f}{' g/cm3' if real else ''}"
             if net.get("density") else "")
    if net.get("temperature") is not None:
        state += (f", T {net['temperature']:.1f} K" if real
                  else f", T {net['temperature']:.3f}")
    lines.append(
        f"    strands   : {cls.get('bridge', 0)} bridge, {cls.get('dangling', 0)} "
        f"dangling, {cls.get('loop', 0)} primary loop, {cls.get('free', 0)} sol"
        + (f"; {state}" if state else ""))
    lines.append(f"    loops     : {net.get('n_primary_loops', 0)} primary, "
                 f"{net.get('n_secondary_loops', 0)} secondary (parallel strands)")
    if net.get("deg_chem_dist"):
        lines.append(f"    P(f) chem : {format_pf(net['deg_chem_dist'])}")
        lines.append(f"    P(f) eff  : {format_pf(net['deg_eff_dist'])}")
    if net.get("z1"):
        lines.extend(format_z1(net["z1"], indent="    "))
    elif net.get("z1_note"):
        lines.append(f"    Z1+       : not measured. {net['z1_note']}")
    return lines


def format_topology(entry: dict) -> list[str]:
    """Render stage 1's manifest section: what was asked, what came back.

    The degree table is the point of it. An exact sculpt is expected to
    match row for row, so a difference is the thing to look at; a strict
    sculpt lists only the degrees the target named. Degree 0 is the
    vacancies, which is why an ``achieved`` count can exceed a requested
    one on a lattice with more sites than the target needs.
    """
    lines = [""]
    bits = [entry.get("source", "?")]
    if entry.get("generator"):
        bits.append(f"{entry['generator']} generator")
    if entry.get("search"):
        bits.append(f"{entry['search']} search")
    lines.append(f"  Topology ({', '.join(bits)}):")

    lattice = []
    if entry.get("lattice_type"):
        lattice.append(f"{entry.get('lattice_size', '?')} {entry['lattice_type']}")
    if entry.get("neighbour_cutoff") is not None:
        lattice.append(f"cutoff {entry['neighbour_cutoff']:g}")
    if entry.get("max_functionality") is not None:
        lattice.append(f"max f {entry['max_functionality']}")
    if lattice:
        lines.append(f"    lattice   : {', '.join(lattice)}")
    lines.append(
        f"    graph     : {entry.get('nodes', '?')} nodes, "
        f"{entry.get('edges', '?')} edges"
        + (f", {entry['seconds']:.2f} s" if entry.get("seconds") is not None else "")
    )

    sculpt = entry.get("sculpt") or {}
    requested = {int(k): int(v) for k, v in (sculpt.get("requested") or {}).items()}
    achieved = {int(k): int(v) for k, v in (sculpt.get("achieved") or {}).items()}
    if requested or achieved:
        degrees = sorted(set(requested) | set(achieved))
        lines.append("    degree    : " + "".join(f"{d:>7d}" for d in degrees))
        lines.append("      requested " + "".join(
            f"{requested[d]:>7d}" if d in requested else f"{'-':>7s}"
            for d in degrees))
        lines.append("      achieved  " + "".join(
            f"{achieved.get(d, 0):>7d}" for d in degrees))
        if requested:
            off = [d for d in degrees
                   if d in requested and achieved.get(d, 0) != requested[d]]
            partial = len(requested) < len(degrees)
            if off:
                lines.append(
                    "      match     : off by "
                    + ", ".join(f"{achieved.get(d, 0) - requested[d]:+d} at "
                                f"degree {d}" for d in off))
            else:
                lines.append(
                    "      match     : every requested degree met"
                    if partial else "      match     : exact")

    stats = []
    if sculpt.get("seconds") is not None:
        stats.append(f"{sculpt['seconds']:.2f} s in the search")
    if sculpt.get("attempts"):
        stats.append(f"{sculpt['attempts']} attempt(s)")
    if sculpt.get("giant_frac") is not None:
        stats.append(f"giant {sculpt['giant_frac']:.3f}")
    if sculpt.get("n_components") is not None:
        stats.append(f"{sculpt['n_components']} component(s)")
    if sculpt.get("augmentations") is not None:
        stats.append(f"{sculpt['augmentations']} augmentations")
    if sculpt.get("target_swaps") is not None:
        stats.append(f"{sculpt['target_swaps']} repairs")
    if sculpt.get("n_double"):
        stats.append(f"{sculpt['n_double']} forced double(s)")
    if stats:
        lines.append(f"    sculpt    : {'; '.join(stats)}")
    return lines


def _fmt_pf(hist: dict) -> str:
    """A degree histogram as ``f=4: 2304, f=3: 161``, highest f first."""
    items = sorted(((int(k), v) for k, v in hist.items()), reverse=True)
    return ", ".join(f"f={f}: {n}" for f, n in items)


def format_summary(s: RunSummary) -> str:
    """Render a RunSummary as multi-line text."""
    lines = [
        f"topon inspect  {s.root}",
        "",
        f"  atoms       : {s.atom_count if s.atom_count is not None else '?'}",
        f"  atom types  : {s.n_atom_types if s.n_atom_types is not None else '?'}",
    ]
    if s.box:
        lx, ly, lz = s.box
        lines.append(f"  box (A)     : {lx:.2f} x {ly:.2f} x {lz:.2f}")
    if s.topology:
        lines.extend(format_topology(s.topology))
    lines.extend([
        "",
        "  Stage status:",
    ])
    for stage in (s.chemistry, s.conformation, s.simulation):
        if stage.present:
            lines.append(f"    [ok]   {stage.name:<18s}  {stage.summary}")
        else:
            lines.append(f"    [miss] {stage.name:<18s}  (no files)")

    if s.defects:
        lines.extend(["", "  Defects (requested -> achieved):"])
        for name in ("primary_loops", "secondary_loops", "triangles",
                     "four_cycles", "sol_chains"):
            rec = s.defects.get(name)
            if not rec:
                continue
            achieved = rec.get("achieved", "?")
            requested = rec.get("requested", achieved)
            extra = []
            if rec.get("source"):
                extra.append(str(rec["source"]))
            if rec.get("placement"):
                extra.append(str(rec["placement"]))
            if rec.get("cycles_created") is not None:
                extra.append(f"{rec['cycles_created']} cycles")
            if rec.get("degree_shift"):
                extra.append(f"P(f) shifted on {rec['degree_shift']} junctions")
            note = f"   ({', '.join(extra)})" if extra else ""
            lines.append(f"    {name:<16s} {requested} -> {achieved}{note}")
        eff = s.defects.get("effective_degree")
        chem = s.defects.get("chemical_degree")
        if eff:
            lines.append(f"    effective P(f) : {_fmt_pf(eff)}")
        if chem:
            lines.append(f"    chemical  P(f) : {_fmt_pf(chem)}")
        budget = s.defects.get("bead_budget") or {}
        if budget.get("total_beads") is not None:
            chains = budget.get("chains", {})
            lines.append(
                f"    bead budget    : {budget['total_beads']} beads over "
                + ", ".join(f"{v} {k}" for k, v in chains.items() if v)
            )

    if s.network:
        lines.extend(format_network(s.network))

    # Suggest the next LAMMPS command (try nested first, then flat root)
    candidates = [s.root / "04_Simulation", s.root]
    for sim_dir in candidates:
        if not sim_dir.exists():
            continue
        scripts = _in_run_order(p.name for p in sim_dir.iterdir()
                                if p.suffix == ".in")
        if scripts:
            lines.extend([
                "",
                "  To run LAMMPS:",
                f"    cd {sim_dir}",
            ])
            for sc in scripts:
                lines.append(f"    lmp -in {sc}")
            break
    return "\n".join(lines)
