"""``topon fit``: from a reference network to a config that regenerates it.

Three steps, each in its own module: :mod:`~topon.inverse.measure` reads the
reference and measures it, :mod:`~topon.inverse.scaffold` picks the cell and
sweeps the neighbour cutoff, and this module writes the config and says what
it could not match.

What goes into the config, and where it comes from:

``topology.generator``
    the cell from the active-site count, the cutoff the sweep chose, the
    reference's P(f) in topon's site convention as exact counts, the exact
    search, the seed.
``assignment``
    DP as mean and polydispersity of the measured chains (the Schulz-Zimm
    form the schema holds; a monodisperse reference is exactly that), the
    end-linked dangling convention, and the defects stage: primary loops by
    effective degree, secondary loops with the reference's endpoint
    degrees, sol chains.
``chemistry``
    coarse-grained at the reference's bead density.
``conformation``
    the placement route and its build knob from the Z1+ calibration of
    :mod:`topon.conformation.entanglement.control`, primary loops drawn
    compact (:data:`FIT_LOOP_SHAPE`), and the reference's Z1+ per bridge as
    ``entanglement.target_Z`` and ``target_hist``. Every one of these is a
    final-state number; nothing here can be checked without MD, and the
    report says so.
``simulation``
    the push-off protocol, compressing to the reference density.

The reference measurement, the sweep table and the flags go to a report
beside the config (``<config>.fit.json``), not into the config.

A reference crosslinked along its chains is fitted by
:mod:`topon.inverse.crosslinked` instead, to the crosslink generator
(``topology.source: "crosslink"``) or, asked for, to the lattice route.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from topon.inverse.measure import Measurement, measure, read_reference, z_target
from topon.inverse.scaffold import (
    CONTROL_CUTOFF, candidate_cutoffs, choose_cell, describe_cutoff,
    rule_of_thumb, shells_within, sweep, with_cutoff)

#: Bead density written when the reference does not say (a graph file):
#: the Kremer-Grest melt.
FALLBACK_DENSITY = 0.85
#: DP written when neither the reference nor --dp says: the schema default.
FALLBACK_DP = 25

#: ``conformation.loop_shape`` of every conformation block a fit writes
#: (since 0.4.5). The schema's default stays ``"ring"``; a fitted config draws its
#: primary loops as compact closed walks, which brought the fit MD closer to
#: the references: Z1+ per bridge on the N100 fit 1.399 against the
#: reference's 1.319 (rings 1.476; both arms with junction_jitter 0.15 added
#: by hand, which this config does not carry), and on the N20 fit 0.136 per
#: loop against 0.078 (rings 0.191) and 0.215 per bridge against 0.178 (rings
#: 0.226). The
#: calibration graphs carry no loops, so no row a knob rests on was drawn
#: either way, and a graph without primary loops builds the same under either
#: shape.
FIT_LOOP_SHAPE = "compact"


class FitError(ValueError):
    """The reference cannot be fitted as asked."""


@dataclass
class FitResult:
    """A fitted config, the report that explains it, and the measurement."""

    config: dict
    report: dict
    measurement: Measurement
    flags: list = field(default_factory=list)


def _flag(flags: list, level: str, what: str, detail: str) -> None:
    flags.append({"level": level, "what": what, "detail": detail})


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

def degree_distribution(target: dict, n_sites: int, max_f: int) -> str:
    """The sculpt target as a ``degree_distribution`` naming every degree.

    Degree 0 is the cell's vacancies, whatever the active sites leave over;
    the reference's own degree-0 count (junctions with loops only) is not a
    site of the lattice route and is dropped here, and reported instead.
    """
    active = {int(d): int(n) for d, n in target.items() if int(d) >= 1}
    vacancies = int(n_sites) - sum(active.values())
    parts = [f"0:{vacancies}"]
    parts += [f"{d}:{active.get(d, 0)}" for d in range(1, max_f + 1)]
    return ",".join(parts)


def _n_sites(lattice: str, n: int, mix: Optional[dict], seed: int) -> int:
    from types import SimpleNamespace

    from topon.topology.generator_python import count_sites

    cfg = SimpleNamespace(lattice_type=lattice, lattice_size=f"{n}x{n}x{n}",
                          mix_fractions=mix or {}, seed=seed)
    return int(count_sites(cfg, seed=seed))


def base_config(rec: dict, cell: dict, *, lattice: str, mix: Optional[dict],
                max_f: int, seed: int, dp: float, pdi: float, density: float,
                name: str, endlinked: bool) -> dict:
    """The fitted config with the cutoff still at its placeholder.

    ``rec`` is the measurement record. The cutoff is set by the sweep; the
    conformation block is added once the route is chosen.
    """
    n_sites = _n_sites(lattice, cell["n"], mix, seed)
    gen = {"lattice_type": lattice, "lattice_size": cell["lattice_size"],
           "neighbour_cutoff": CONTROL_CUTOFF, "periodicity": "111",
           "max_functionality": int(max_f),
           "degree_distribution": degree_distribution(rec["target"], n_sites,
                                                      max_f),
           "search": "exact", "seed": int(seed)}
    if lattice == "MIX":
        gen["mix_fractions"] = dict(mix)
    giant = rec.get("giant_fraction")
    if giant is not None and giant < 0.99:
        gen["min_giant_fraction"] = round(max(0.5, float(giant) - 0.005), 3)

    dp_block = {"default": {"mean": _clean(dp), "pdi": round(float(pdi), 4)}}
    if endlinked:
        dp_block["endlinked_dangling"] = True
    defects: dict = {}
    pri = rec["primary_loops"]["count"]
    sec = rec["secondary_loops"]
    sol = rec["sol_chains"]
    if pri:
        defects["primary_loops"] = {"count": int(pri),
                                    "placement": "by_effective_degree"}
    if sec["count"]:
        defects["secondary_loops"] = {"count": int(sec["count"]),
                                      "endpoint_degrees": dict(sec["endpoint_degrees"])}
    if sol:
        sol_dp = (rec.get("dp") or {}).get("by_class", {}).get("free", {}).get("mean")
        defects["sol_chains"] = {"count": int(sol),
                                 "dp": int(round(sol_dp if sol_dp else dp))}
    if defects:
        defects["seed"] = int(seed)
    assignment = {"dp_distribution": dp_block}
    if defects:
        assignment["defects"] = defects

    return {
        "study": {"name": name, "output_dir": "./output"},
        "topology": {"source": "generate", "generator": gen},
        "assignment": assignment,
        "chemistry": {"model_type": "coarse_grained",
                      "target_density": round(float(density), 6)},
        "output": {"lammps_convention": "endlinked"},
        "simulation": {"protocol": "pushoff",
                       "rho_final": round(float(density), 6)},
    }


def _clean(x: float):
    """An integer when the value is one, so a DP of 20 is written ``20``."""
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round(float(x), 4)


def choose_conformation(dp: float, target: Optional[dict], flags: list) -> tuple[dict, dict]:
    """Placement route, build knob and entanglement target.

    The route is the random walk when the target is at or above the lowest
    final-state Z the walk has reached at this DP, and the meander below it.
    The knob is where the controller would start (:func:`~topon.conformation.
    entanglement.control.seed_actuator`), except for a target under the
    chosen route's floor, where the knob is the one that measured lowest
    (coil ratio 1.402 for the DP-20 reference) and the flag says the target
    is below what has been reached. The block carries the build options the
    knob was measured with (``junction_jitter``, ``settle_clearance``),
    since the same knob built without them is a build the calibration did not
    measure, and ``loop_shape`` :data:`FIT_LOOP_SHAPE`, whether or not
    the graph has primary loops. Without a target there is no build to
    describe and the block is empty.

    Returns ``(conformation block, info)``.
    """
    from topon.conformation.entanglement import (
        FLOORS, actuator_name, calibration_for, floor_warning, seed_actuator)
    from topon.conformation.entanglement.control import (
        CALIBRATION, REMEASURED, CalibrationPoint, build_options)

    if target is None:
        return {}, {"why": "no Z1+ target: the conformation block is left at "
                           "the schema defaults"}
    dp_i = int(round(dp))
    t = float(target["target_Z"])
    below_walk = floor_warning(dp_i, "walk", t)
    placement = "meander" if below_walk else "walk"
    knob = actuator_name(placement)
    below = floor_warning(dp_i, placement, t)
    info: dict = {"placement": placement, "knob": knob, "target_Z": t}
    if below:
        keys = [k for k in FLOORS if k[1] == placement]
        key = min(keys, key=lambda k: abs(k[0] - dp_i))
        x = float(FLOORS[key]["actuator"])
        info.update(start="lowest measured", floor=FLOORS[key]["z_final"],
                    floor_dp=key[0])
        _flag(flags, "warn", "entanglement target below the floor", below)
    else:
        # This DP's own rows first, whatever protocol measured them: the
        # controller's seed prefers the crossing-free protocol at the
        # nearest DP, which for a DP-20 walk means the DP-100 rows and a
        # build density two orders of magnitude off.
        own = [c for c in CALIBRATION if c.dp == dp_i
               and c.placement == placement and c.state == "final"]
        x, note = seed_actuator(dp_i, placement, t, table=own or CALIBRATION)
        info.update(start="calibration", exponent=note.get("exponent"),
                    span=note.get("span"), dp_used=note.get("dp_used"),
                    protocol_used=note.get("protocol_used"),
                    extrapolating=note.get("extrapolating"))
        if note.get("dp_used") != dp_i:
            _flag(flags, "warn", "calibration from another DP",
                  f"nothing is measured at DP {dp_i} for the {placement}; the "
                  f"knob comes from the DP-{note.get('dp_used')} rows, whose "
                  f"level is not this DP's (Z per strand grows steeply with "
                  f"DP). Treat {knob} as a starting guess for the controller.")
        elif note.get("protocol_used") == "hardcore_min":
            _flag(flags, "note", "minimiser-protocol calibration",
                  f"the DP-{dp_i} {placement} rows were measured under the "
                  f"minimiser protocol (hardcore_min), which threaded bonds "
                  f"and added crossings of its own.")
    info["actuator"] = round(x, 4)

    # Which rows the knob rests on, and who drew their coordinates: the
    # controller steers place() builds, and a row the validation scripts
    # built is a measurement of a different placement.
    if below:
        rows = (calibration_for(dp_i, placement, "final", "limit")
                or calibration_for(dp_i, placement, "final", "hardcore_min"))
    else:
        rows = [CalibrationPoint(**r) for r in note.get("rows", [])]
    info["rows"] = [{"dp": c.dp, "actuator": c.actuator, "z": c.z,
                     "protocol": c.protocol, "builder": c.builder,
                     "junction_jitter": c.junction_jitter,
                     "settle_clearance": c.settle_clearance,
                     "source": c.source} for c in rows]
    if rows and all(c.builder == "script" for c in rows):
        detail = (f"every DP-{rows[0].dp} {placement} row the knob rests on ("
                  + ", ".join(f"{knob} {c.actuator:g} -> Z {c.z:g}"
                              for c in rows)
                  + ") was built by the validation scripts (builder "
                    "\"script\"), not by place(), which is what the "
                    "controller steers")
        again = REMEASURED.get((rows[0].dp, placement))
        if again:
            detail += "; " + again
        _flag(flags, "warn", "calibration rows are script-built", detail + ".")

    # A knob means what it measured only with the build that measured it.
    # The DP-20 place() rows carry the pinch fix, and the same knob
    # without it is a different build: the meander at coil 1.402 ends
    # near 0.260 unfixed, where it pinches, against 0.2356 with the fix.
    opts = build_options(rows, x)
    if opts:
        info["build_options"] = opts
        _flag(flags, "note", "build options from the calibration",
              f"the {knob} was measured on place() builds with "
              + " and ".join(f"{k} {v:g}" for k, v in opts.items())
              + ", so the config carries them: without them the same "
                f"{knob} builds a network the calibration did not measure.")

    info["loop_shape"] = FIT_LOOP_SHAPE
    block = {"placement": placement, knob: round(x, 4), **opts,
             "loop_shape": FIT_LOOP_SHAPE,
             "entanglement": {"target_Z": t,
                              "target_hist": list(target["target_hist"]),
                              "close_on": "final"}}
    return block, info


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit(path, *, junction_type: Optional[int] = None,
        nodes: Optional[str] = None, lattice: str = "SC",
        mix: Optional[dict] = None, cutoffs: Optional[Sequence[float]] = None,
        seeds: int = 2, seed: int = 1, max_functionality: Optional[int] = None,
        dp: Optional[float] = None, density: Optional[float] = None,
        z1: Optional[bool] = None, z1_config=None, name: Optional[str] = None,
        control: bool = True,
        log: Optional[Callable[[str], None]] = None,
        crosslinked: Optional[bool] = None, route: Optional[str] = None,
        crosslink_bond_types: Optional[Sequence[int]] = None,
        sequence: Optional[str] = None, repeats: int = 1,
        crosslink_residue: str = "Y", reactive_every: Optional[int] = None,
        reactive_start: Optional[int] = None,
        packing: Optional[float] = None,
        contact_radius: Optional[float] = None) -> FitResult:
    """Measure ``path`` and write a config that regenerates it.

    ``seeds`` builds per candidate cutoff, starting at ``seed``; the config
    is pinned to ``seed``, so its graph is the sweep's first build of the
    chosen cutoff. ``cutoffs`` replaces the candidates the rule of thumb
    gives. ``dp``, ``density`` and ``max_functionality`` override what the
    reference says (a graph file says nothing about the first two).

    A reference crosslinked along its chains (``crosslinked`` None decides
    from the file) goes to :func:`topon.inverse.crosslinked.fit_crosslinked`
    with ``route`` (``"crosslink"``, the default, or ``"lattice"``), the
    reactive beads (``sequence``, ``repeats``, ``crosslink_residue``, or
    ``reactive_every`` from ``reactive_start``), ``packing`` and
    ``contact_radius`` when given; ``crosslink_bond_types`` are its data
    file's crosslink bond types.

    Raises:
        FitError: a degree above ``max_functionality``, or no candidate
            cutoff that builds on every seed; for a crosslinked reference,
            what :func:`~topon.inverse.crosslinked.fit_crosslinked` refuses;
            for an end-linked one, an option only a crosslinked one takes.
    """
    say = log or (lambda msg: None)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    flags: list = []
    if lattice == "MIX" and not mix:
        raise FitError("a MIX lattice needs its fractions (--mix SC,BCC,FCC)")
    if lattice not in ("SC", "MIX"):
        _flag(flags, "note", f"{lattice} lattice",
              "the cutoff candidates are simple-cubic shell radii in cell "
              "units, which is what the validation swept; on this lattice "
              "they are ranges without a shell meaning.")

    ref = read_reference(path, junction_type=junction_type, nodes=nodes,
                         crosslinked=crosslinked,
                         crosslink_bond_types=crosslink_bond_types)
    say(f"read {Path(path).name} ({ref.source}"
        + (", crosslinked along its chains)" if ref.architecture == "crosslinked"
           else ")"))
    if ref.architecture == "crosslinked":
        from topon.inverse.crosslinked import fit_crosslinked

        if lattice != "SC" or mix:
            raise FitError("a crosslinked reference is fitted on a simple-cubic "
                           "lattice (the crosslink generator's, or the lattice "
                           "route's SC cell)")
        meas = measure(ref, z1=False)
        say(f"measured in {time.perf_counter() - wall0:.0f} s")
        return fit_crosslinked(
            ref, meas, route=route or "crosslink", seeds=seeds, seed=seed,
            density=density, packing=packing, contact_radius=contact_radius,
            sequence=sequence, repeats=repeats,
            crosslink_residue=crosslink_residue, reactive_every=reactive_every,
            reactive_start=reactive_start, cutoffs=cutoffs, name=name,
            log=log, started=(wall0, cpu0))
    given = [k for k, v in (("route", route), ("sequence", sequence),
                            ("reactive_every", reactive_every),
                            ("reactive_start", reactive_start),
                            ("crosslink_bond_types", crosslink_bond_types or None),
                            ("repeats", None if repeats == 1 else repeats),
                            ("crosslink_residue",
                             None if crosslink_residue == "Y" else crosslink_residue),
                            ("packing", packing),
                            ("contact_radius", contact_radius)) if v is not None]
    if given:
        raise FitError(f"{', '.join(given)} apply to a network crosslinked along "
                       f"its chains, and {Path(path).name} reads end-linked")
    meas = measure(ref, z1=z1, z1_config=z1_config)
    rec = meas.record
    t_measure = time.perf_counter() - wall0
    say(f"measured in {t_measure:.0f} s: {rec['junctions']} junctions, "
        f"{rec['n_active_sites']} active sites")

    # --- functionality ----------------------------------------------------
    top_eff = max((int(d) for d, n in rec["target"].items() if n), default=1)
    top_chem = max((int(d) for d in rec["pf_chemical"]), default=top_eff)
    max_f = int(max_functionality) if max_functionality else max(top_eff, top_chem)
    if top_eff > max_f:
        raise FitError(
            f"the reference has junctions of effective degree {top_eff} and "
            f"max_functionality is {max_f}; no sculpt can place them. Raise "
            f"--max-functionality to at least {top_eff}.")
    if top_chem > max_f:
        _flag(flags, "warn", "chemical degree above max_functionality",
              f"junctions reach chemical degree {top_chem} with their loops "
              f"and the ceiling is {max_f}; the defects stage will not put a "
              f"loop where it would pass the ceiling, so some loops go "
              f"unplaced.")

    # --- DP and density ---------------------------------------------------
    dps = rec.get("dp")
    if dp is None:
        if dps:
            dp, pdi = dps["mean"], dps["pdi"]
        else:
            dp, pdi = FALLBACK_DP, 1.0
            _flag(flags, "warn", "no DP in the input",
                  f"dp_distribution is written at {FALLBACK_DP}; pass --dp.")
    else:
        pdi = dps["pdi"] if dps else 1.0
    if dps and dps["pdi"] > 1.0 + 1e-6:
        _flag(flags, "warn", "polydisperse DP",
              f"the schema holds a Schulz-Zimm mean and PDI, so the measured "
              f"histogram ({dps['min']} to {dps['max']}) is matched in its "
              f"first two moments only. The DP draw comes from the global "
              f"random stream, which no config seed pins "
              f"(topology.generator.seed pins stage 1, defects.seed the "
              f"defects), so topon generate repeats the graph but not its DPs; "
              f"the sweep and --verify seed that stream themselves.")
    if density is None:
        density = rec.get("density")
        if density is None:
            density = FALLBACK_DENSITY
            _flag(flags, "warn", "no density in the input",
                  f"chemistry.target_density is written at {FALLBACK_DENSITY}; "
                  f"pass --density.")

    # --- cell and cutoff --------------------------------------------------
    cell = choose_cell(rec["n_active_sites"], lattice, mix)
    alt = choose_cell(rec["n_junction_sites"], lattice, mix)
    rule = None
    sp = rec.get("spatial") or {}
    if sp.get("chord_p95") is not None and ref.box is not None:
        rule = rule_of_thumb(sp["chord_p95"], ref.box, cell["n"])
    if cutoffs:
        cands = [describe_cutoff(c) for c in cutoffs]
    else:
        cands = candidate_cutoffs(rule, cell["n"])
    if rule is None:
        _flag(flags, "note", "no rule of thumb",
              "the reference has no junction coordinates, so the cutoff "
              "comes from a wider sweep alone.")

    name = name or f"{Path(path).stem}_fit"
    config = base_config(rec, cell, lattice=lattice, mix=mix, max_f=max_f,
                         seed=seed, dp=dp, pdi=pdi, density=density,
                         name=name, endlinked=ref.source in ("data", "npz"))
    control_cutoff = CONTROL_CUTOFF if control else None
    if lattice == "Diamond":
        cands, refusal = _diamond_candidates(config, cands)
        if refusal:
            _flag(flags, "warn", "Diamond at the canonical cutoff", refusal)
            if control_cutoff is not None and _diamond_candidates(
                    config, [describe_cutoff(control_cutoff)])[1]:
                control_cutoff = None
        if not cands:
            raise FitError(refusal)
    seed_list = [int(seed) + i for i in range(max(1, int(seeds)))]
    say(f"cell {cell['lattice_size']} {lattice} ({cell['vacancy_fraction']:.1%} "
        f"vacant); rule of thumb "
        + (f"{rule:.2f}" if rule is not None else "n/a")
        + "; sweeping " + ", ".join(f"{c['cutoff']:.2f}" for c in cands)
        + (f" and the {control_cutoff:.1f} control" if control_cutoff else "")
        + f" on seeds {seed_list}")
    t_sweep0 = time.perf_counter()
    sw = sweep(config, meas.descriptors, cands, seed_list,
               control=control_cutoff, log=say)
    t_sweep = time.perf_counter() - t_sweep0
    chosen = sw["chosen"]
    if chosen is None:
        errors = sorted({r.get("error", "") for r in sw["rows"] if r.get("error")})
        raise FitError("no candidate cutoff built on every seed: "
                       + "; ".join(errors[:3]))
    config = with_cutoff(config, chosen["cutoff"])
    if chosen.get("within_scatter"):
        ru = chosen["runner_up"]
        _flag(flags, "note", "cutoff choice within seed scatter",
              f"{chosen['cutoff']:.2f} ({chosen['shells']} shells) scored "
              f"{chosen['composite']:.3f} and {ru['cutoff']:.2f} "
              f"({ru['shells']} shells) {ru['composite']:.3f}; the gap is "
              f"smaller than the spread between seeds.")

    # --- conformation -----------------------------------------------------
    target = z_target(meas)
    conf, conf_info = choose_conformation(dp, target, flags)
    if conf:
        config["conformation"] = conf

    chosen_rows = [r for r in sw["rows"] if r["cutoff"] == chosen["cutoff"]
                   and r["seed"] == int(seed)]
    build = chosen_rows[0] if chosen_rows else {}

    # --- what cannot be matched -------------------------------------------
    nr = rec["not_representable"]
    lost = nr["loops_on_them"] + nr["loops_on_those"]
    if lost:
        detail = (f"{nr['loop_only_junctions']} junctions carry only loops "
                  f"({nr['loops_on_them']} loops) and "
                  f"{nr['one_strand_junctions']} have one other strand "
                  f"({nr['loops_on_those']} loops). The sculptor leaves the "
                  f"first empty and reads the second as dangling-chain ends, "
                  f"so those {lost} loops go to other junctions")
        if build.get("pf_chemical"):
            detail += (f": the build's chemical P(f) is "
                       f"{_pf(build['pf_chemical'])} against the reference's "
                       f"{_pf(rec['pf_chemical'])}")
        _flag(flags, "warn", "loops on junctions the lattice cannot hold",
              detail + ".")
    if build.get("beads") and rec.get("n_atoms") and build["beads"] != rec["n_atoms"]:
        _flag(flags, "note", "bead count",
              f"the build has {build['beads']} beads and the reference "
              f"{rec['n_atoms']}; at the same density its box is "
              f"{(build['beads'] / rec['n_atoms']) ** (1 / 3):.4f} of the "
              f"reference's.")
    for note in rec.get("notes", []):
        _flag(flags, "note", "reference", note)
    if rec.get("z1_note"):
        _flag(flags, "note", "Z1+", rec["z1_note"])

    # --- the doctor, on the config as written ------------------------------
    doctor = _doctor(config)
    for issue in doctor:
        if issue["rule"] == "entanglement_target_below_floor":
            continue                 # flagged above, with the route choice
        if issue["level"] in ("warn", "error"):
            _flag(flags, issue["level"], f"doctor: {issue['rule']}",
                  issue["message"])

    report = {
        "reference": {k: v for k, v in rec.items() if k != "descriptors"},
        "reference_descriptors": {k: rec["descriptors"].get(k) for k in (
            "edge_shortest_cycle_mean", "frac_odd_cycles",
            "frac_edges_in_cycle_le4", "transitivity_core",
            "square_clustering_core", "lambda2_core", "avg_path_core")},
        "cell": cell,
        "junction_only_cell": alt,
        "rule_of_thumb": None if rule is None else {
            "p95_separation": sp.get("chord_p95"),
            "site_spacing": float(sum(ref.box) / 3.0 / cell["n"]),
            "cutoff": rule, "shells": shells_within(rule)},
        "sweep": sw,
        "chosen": chosen,
        "build": {k: build.get(k) for k in ("beads", "pf_chemical",
                                            "pf_effective", "sculpt")},
        "conformation": conf_info,
        "doctor": doctor,
        "flags": flags,
        "seconds": {"measure": round(t_measure, 1),
                    "sweep": round(t_sweep, 1),
                    "total": round(time.perf_counter() - wall0, 1),
                    "cpu": round(time.process_time() - cpu0, 1)},
    }
    return FitResult(config=config, report=report, measurement=meas,
                     flags=flags)


def _diamond_candidates(config: dict, cands: list) -> tuple[list, Optional[str]]:
    """The candidates the Diamond doctor rule does not refuse, and why it did.

    Diamond at the canonical cutoff gives every site four candidate
    partners, so a P(f) with dangling ends and half or more of its active
    sites four-fold has no network on it (the doctor rule
    ``diamond_dangling_ends``, the exact search's own no-slack test). A
    wider range gives the scaffold spare candidates and is kept.
    """
    from topon.config.schema import ToponConfig
    from topon.diagnostics.rules import check_diamond_dangling_ends

    kept, why = [], None
    for c in cands:
        trial = with_cutoff(config, c["cutoff"])
        known = {k: v for k, v in trial.items() if k in ToponConfig.model_fields}
        issues = check_diamond_dangling_ends(ToponConfig.model_validate(known), trial)
        if issues:
            why = f"{issues[0].message} {issues[0].fix or ''}".strip()
        else:
            kept.append(c)
    return kept, why


def _doctor(config: dict) -> list:
    """``topon doctor`` on the config as it will be written."""
    from topon.config.schema import ToponConfig
    from topon.diagnostics import run_all_rules

    known = {k: v for k, v in config.items() if k in ToponConfig.model_fields}
    cfg = ToponConfig.model_validate(known)
    return [{"rule": i.rule, "level": i.level, "message": i.message}
            for i in run_all_rules(cfg, config)]


def write_fit(result: FitResult, out) -> tuple[Path, Path]:
    """The config at ``out`` and the report beside it as ``<stem>.fit.json``."""
    from topon.analysis.descriptors import to_jsonable

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.config, indent=2) + "\n", encoding="utf-8")
    rep = out.with_name(out.stem + ".fit.json")
    rep.write_text(json.dumps(result.report, indent=1, default=to_jsonable)
                   + "\n", encoding="utf-8")
    return out, rep


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _pf(hist: dict) -> str:
    return ", ".join(f"{int(k)}:{v}" for k, v in sorted(
        ((int(k), v) for k, v in hist.items()), reverse=True))


def format_fit(result: FitResult) -> str:
    """The fit for the terminal: what was measured, chosen and flagged."""
    r = result.report
    if r.get("architecture") == "crosslinked":
        from topon.inverse.crosslinked import format_fit_crosslinked
        return format_fit_crosslinked(result)
    ref = r["reference"]
    cfg = result.config
    gen = cfg["topology"]["generator"]
    lines = [f"topon fit  {ref['input']}", ""]
    ch = ref["chains"]
    lines.append(f"  reference : {ref['junctions']} junctions; strands "
                 f"{ch['bridge']} bridge, {ch['loop']} primary loop, "
                 f"{ch['dangling']} dangling, {ch['free']} sol; "
                 f"{ref['secondary_loops']['count']} secondary loops")
    lines.append(f"  P(f)      : effective {_pf(ref['pf_effective'])}; "
                 f"chemical {_pf(ref['pf_chemical'])}")
    if ref.get("dp"):
        lines.append(f"  DP        : mean {ref['dp']['mean']:.4g}, PDI "
                     f"{ref['dp']['pdi']:.4f}"
                     + (f"; {ref['n_atoms']} beads at density "
                        f"{ref['density']:.4f}" if ref.get("density") else ""))
    sp = ref.get("spatial") or {}
    if sp.get("chord_p95") is not None:
        lines.append(f"  reach     : junction separation {sp['chord_mean']:.2f} "
                     f"+- {sp['chord_sd']:.2f} {sp.get('units', '')}, p95 "
                     f"{sp['chord_p95']:.2f}")
    z = ref.get("z1") or {}
    zb = (z.get("by_class") or {}).get("bridge")
    if zb:
        lines.append(f"  Z1+       : {zb['Zmean']:.3f} per bridge, "
                     f"{100 * zb['frac_zero']:.0f} % with none")
    c = r["cell"]
    lines.append("")
    lines.append(f"  cell      : {c['lattice_size']} {c['lattice_type']}, "
                 f"{c['n_active']} active sites, {c['vacancy_fraction']:.1%} "
                 f"vacant (junction sites only: {r['junction_only_cell']['lattice_size']})")
    rule = r["rule_of_thumb"]
    if rule:
        lines.append(f"  rule      : p95 {rule['p95_separation']:.2f} / spacing "
                     f"{rule['site_spacing']:.2f} = {rule['cutoff']:.2f} "
                     f"({rule['shells']} SC shells within)")
    lines.append("  sweep     : cutoff  shells    z   composite          over ref "
                 "(seeds " + ",".join(str(s) for s in r["sweep"]["seeds"]) + ")")
    for s in r["sweep"]["summary"]:
        mark = ("control" if s["control"] else
                "<- chosen" if r["chosen"] and s["cutoff"] == r["chosen"]["cutoff"]
                else "")
        comp = ("failed" if s["composite"] is None else
                f"{s['composite']:.3f} +- {s['composite_sd']:.3f}")
        shipped = ("" if s.get("composite_shipped") is None
                   else f"{s['composite_shipped']:.3f}")
        lines.append(f"              {s['cutoff']:5.2f}  {s['shells']:5d}  "
                     f"{s['z']:4d}   {comp:<17s}  {shipped:<8s} {mark}")
    lines.append("")
    lines.append(f"  config    : {gen['lattice_type']} {gen['lattice_size']}, "
                 f"cutoff {gen['neighbour_cutoff']:.2f}, "
                 f"{gen['degree_distribution']}, exact, seed {gen['seed']}")
    d = cfg["assignment"].get("defects", {})
    lines.append("              DP {} (PDI {}), density {}; loops: {} primary, "
                 "{} secondary, {} sol".format(
                     cfg["assignment"]["dp_distribution"]["default"]["mean"],
                     cfg["assignment"]["dp_distribution"]["default"]["pdi"],
                     cfg["chemistry"]["target_density"],
                     d.get("primary_loops", {}).get("count", 0),
                     d.get("secondary_loops", {}).get("count", 0),
                     d.get("sol_chains", {}).get("count", 0)))
    conf = cfg.get("conformation")
    if conf:
        knob = r["conformation"]["knob"]
        built = "".join(f", {k} {conf[k]:g}" for k in
                        ("junction_jitter", "settle_clearance") if conf.get(k))
        if conf.get("loop_shape"):
            built += f", {conf['loop_shape']} loops"
        lines.append(f"              {conf['placement']} at {knob} "
                     f"{conf[knob]}{built}, target Z "
                     f"{conf['entanglement']['target_Z']}"
                     f" per bridge (final state, needs MD to check)")
    b = r.get("build") or {}
    if b.get("beads") and ref.get("n_atoms"):
        lines.append(f"              {b['beads']} beads against the "
                     f"reference's {ref['n_atoms']}")
    if r["flags"]:
        lines.append("")
        for f in r["flags"]:
            lines.append(f"  [{f['level']}] {f['what']}: {f['detail']}")
    s = r["seconds"]
    lines.append("")
    lines.append(f"  time      : measure {s['measure']:.0f} s, sweep "
                 f"{s['sweep']:.0f} s, total {s['total']:.0f} s "
                 f"(CPU {s['cpu']:.0f} s)")
    return "\n".join(lines)
