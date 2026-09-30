"""Close a loop on the measured entanglement state of a built network.

The knob is the chain's shape at build; the reading is Z1+ at the final state.
Between them sits a whole relaxation protocol, so the loop is: place -> relax ->
compress -> settle -> measure -> move the knob -> place again. This module owns
the knob-moving half. It never runs dynamics: the caller passes a ``runner``
that does, which is what keeps the conformation stage free of LAMMPS and makes
the controller testable against a known response.

Why the reading has to be the final state
-----------------------------------------
Z1+ is not a topological invariant. It counts the kinks of the shortest path
between the junctions *as they currently sit*, so contacts slide off as the
junctions move and the box changes. Measured on N100 with zero bonds above
1.2 sigma at every stage and with every free-ended chain removed from the
input, Z per bridge went 1.16 after push-off, 1.00 after equilibration at
constant volume, 1.11 after compression (``REPORT.md`` 4.3). Nothing crossed.
So a build-state number is a starting point and a calibration entry that is
not a final-state number is not a calibration entry.

That is also why the acceptance test for this loop is reproducibility across
seeds at the final state, not constancy of Z between stages. Constancy is the
wrong gate; the right gate is the protocol's own, zero bonds above 1.2 sigma at
every stage.

The response curve
------------------
Over the range that has been measured, Z is a power law in the actuator:
``Z = A * x^b`` with x the coil ratio (meander) or the build density (walk).
The two exponents that come out of the crossing-free runs are 0.87 for the
DP-20 meander and 0.30 for the DP-100 walk, so the curve is shallow and the
controller's first move is usually its biggest. Two measured points fix A and
b; with one, the exponent comes from the shipped table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "CalibrationPoint",
    "CALIBRATION",
    "FLOORS",
    "REMEASURED",
    "RoundPlan",
    "RoundResult",
    "actuator_name",
    "calibration_for",
    "seed_actuator",
    "solve_actuator",
    "floor_warning",
    "hist_ks",
    "controller",
]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationPoint:
    """One measured (actuator, Z) pair, with the state it was measured in.

    ``state`` is ``"final"`` (compressed to the chemistry's density,
    equilibrated and quenched) or ``"build"`` (equilibrated at the build box,
    i.e. stage 3 -- *not* the push-off, which reads 14 % higher on the N100
    walk: 1.139 against 1.004). Only ``final`` points steer by default;
    ``build`` points are kept because they are what the placement can be
    checked against with a shorter protocol, and a caller that wants them has
    to say ``close_on: "build"``.

    ``protocol`` is ``"limit"`` for the crossing-free push-off and
    ``"hardcore_min"`` for the minimiser that preceded it. The minimiser
    stretched 85 bonds to 1.70 sigma and left 57 threaded, which added
    crossings of its own (N20 went 0.19 -> 0.24 through the protocol), so its
    final-state points describe the minimiser as much as the placement and are
    not used to steer.

    ``"pushoff"`` is the protocol's own name for the same deck ``"limit"``
    names here, and
    the duplication is deliberate rather than tidy. Renaming the older rows
    would fold them in with the one row measured under ``pushoff``, and
    :func:`seed_actuator` prefers ``"limit"`` over ``"hardcore_min"``: at DP 20
    the walk route would go from a three-point power-law fit to a single row,
    which fixes a level and not a slope. That is a worse seed, so the label
    stays split until the other four DP-20 rows are re-measured and the whole
    set can move together. Renaming then is a one-line change here and nothing
    outside this module reads the field.

    ``builder`` is what drew the coordinates: ``"script"`` for
    ``bond_create_validation/scripts/``, ``"place"`` for
    :func:`topon.conformation.place`. Every row below is ``"script"``, which
    is worth saying out loud, because the controller steers ``place()`` builds
    with a table measured on somebody else's placement. The two are not the
    same placement: on the same graph and density the validation script and
    ``place()`` put the placed-state Z at 0.0526 and 0.0287 respectively. That
    difference washes out -- both jump in stage 1 and stay flat, to 0.186 and
    0.215 -- which is why the table is keyed on post-protocol states and not
    on the build. It is still a gap between what was measured and what is
    being steered, and a row that does not say which side it came from hides
    it. Re-measuring the DP-20 rows with ``place()`` is open and needs MD.
    """

    dp: int
    placement: str
    actuator: float
    z: float
    state: str = "final"
    protocol: str = "limit"
    graph: str = ""
    source: str = ""
    builder: str = "script"


#: What has actually been measured, and where it came from. Every entry is a
#: run in ``bond_create_validation/data/runs/``; the coil ratios are computed
#: from each run's own graph and build density with
#: :func:`topon.conformation.placement.coil_ratio_of`, so they are in the same
#: definition the controller uses (contour over chord, mean over mean).
#:
#: The coil ratio recorded beside each row is *contour over mean chord*, the
#: mean taken over strands. That definition is settled rather than chosen: the
#: validation session recomputed it from ``runs/N20_mix90_4sh_r050_meander/
#: build.data`` and got mean chord 13.46 sigma against a contour of 20.37 at
#: rho 0.05, so 1.51, and 2.77 for the same graph at rho 0.3075, which is the
#: report's "2.8". The competing readings give 1.61 (mean of the per-strand
#: ratios) and 1.45 (contour over the median chord), and neither reproduces the
#: 2.8.
#:
#: All but one row is ``builder="script"``: the coordinates were drawn by
#: ``bond_create_validation/scripts/``, not by :func:`topon.conformation.place`,
#: which is what the controller actually steers. The exception is the DP-20
#: walk pair at rho 0.035, measured with ``place()`` on 2026-09-21 and labelled
#: ``"pushoff"``; the four DP-20 rows above it stayed script-built because their
#: re-measurement stopped at stage 2 on the push-off's own bond gate.
#:
#: A second ``place()``-built point exists and is deliberately not a row:
#: DP 100 walk, rho_build 0.0894, final-state Z1+ 1.3081 against the reference
#: 1.32, KS p = 1.0000 on the per-strand histogram. Adding it would move the
#: DP-100 fit the controller extrapolates on, and confirming that did no harm
#: means re-running the controller, which needs MD.
#:
#: An earlier figure of 1.9 at rho 0.05 came from the DP-30 6x6x6 pilot cell
#: and was never recomputed for the N20 one, where the value is 1.5. The table
#: stays keyed on the build density, which is unambiguous and is what every
#: run actually recorded.
CALIBRATION: tuple[CalibrationPoint, ...] = (
    # --- DP 20, N20 MIX 90/5/5 4-shell graph, final box rho 0.3075 ---
    # Crossing-free protocol (--stage1 limit), re-quenched to T = 0.4.
    CalibrationPoint(20, "meander", 1.402, 0.224, "final", "limit",
                     "N20_MIX90_4sh", "REPORT.md 4.4, runs/N20_v3b"),
    CalibrationPoint(20, "meander", 1.510, 0.239, "final", "limit",
                     "N20_MIX90_4sh", "REPORT.md 4.4, runs/N20_v3"),
    # Build state, same runs: the placement on its own, before compression.
    CalibrationPoint(20, "meander", 1.402, 0.192, "build", "limit",
                     "N20_MIX90_4sh",
                     "data/measure_N20_v3b_stage3_build.json"),
    CalibrationPoint(20, "meander", 1.510, 0.192, "build", "limit",
                     "N20_MIX90_4sh",
                     "data/measure_N20_v3_stage3_build.json"),
    # Random walk, minimiser protocol. Build-state column of REPORT.md 4; these
    # are the three the specification names, and they are what the floor rests on.
    CalibrationPoint(20, "walk", 0.145, 0.300, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r145"),
    CalibrationPoint(20, "walk", 0.095, 0.250, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r095"),
    CalibrationPoint(20, "walk", 0.035, 0.232, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r035"),
    CalibrationPoint(20, "walk", 0.145, 0.335, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r145"),
    CalibrationPoint(20, "walk", 0.095, 0.288, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r095"),
    CalibrationPoint(20, "walk", 0.035, 0.262, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r035"),
    CalibrationPoint(20, "meander", 1.274, 0.225, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_..._r030_meander"),

    # --- the one DP-20 row measured with place() -------------------------
    # Re-measuring the five DP-20 rows with place() was authorised and run on
    # 2026-09-21. Four of the five stopped at stage 2 on the push-off's own
    # bond gate, with 3 to 6 persistent 1.3-1.4 sigma bonds that the placed
    # build did not have; those four stay script-built above and provisional.
    # This one passed every gate with zero threaded bonds, and it lands on the
    # script's number: final 0.2598 against 0.262, build 0.2388 against 0.232.
    # Labelled "pushoff" so it does not steer on its own -- see
    # CalibrationPoint for why that is deliberate.
    CalibrationPoint(20, "walk", 0.035, 0.2598, "final", "pushoff",
                     "N20_MIX90_4sh",
                     "tests/output/v54_2/recal_w0035/controller.json",
                     builder="place"),
    CalibrationPoint(20, "walk", 0.035, 0.2388, "build", "pushoff",
                     "N20_MIX90_4sh",
                     "tests/output/v54_2/recal_w0035/controller.json",
                     builder="place"),

    # --- DP 100, N100 SC 8-shell graph, final box rho 0.3015 ---
    CalibrationPoint(100, "walk", 0.060, 1.17, "final", "limit",
                     "N100_SC_8sh", "REPORT.md 4.4, runs/N100_v3"),
    CalibrationPoint(100, "walk", 0.085, 1.30, "final", "limit",
                     "N100_SC_8sh", "REPORT.md 4.4, runs/N100_v3b"),
    CalibrationPoint(100, "walk", 0.060, 1.00, "build", "limit",
                     "N100_SC_8sh", "data/measure_N100_v3_stage3_build.json"),
    # The specification's "DP 100 walk 0.05 -> 1.19 preliminary": the minimiser
    # protocol, and its 108 bonds above 1.2 sigma say how preliminary.
    CalibrationPoint(100, "walk", 0.050, 1.19, "final", "hardcore_min",
                     "N100_SC_8sh", "REPORT.md 4.1, runs/N100_sc8sh_r050"),
)


#: The lowest final-state Z each route has been seen to reach, and the
#: actuator it took. A target below one of these is not refused -- no one has
#: proved the floor is a floor -- but it is named, with the route that goes
#: lower where there is one.
#:
#: The random-walk floor is the measured one: lowering the build density from
#: 0.145 to 0.035 moved DP-20 Z only from 0.30 to 0.23 at the build state and
#: 0.335 to 0.262 at the final state, because coiled chains collapse during the
#: push-off at fixed volume and trap the crossings. Density alone does not get
#: a random walk to the reference's 0.178; shape does.
FLOORS: dict[tuple[int, str], dict] = {
    (20, "walk"): {"z_final": 0.262, "z_build": 0.232, "actuator": 0.035,
                   "note": "random-walk chains collapse during the push-off "
                           "at fixed volume and trap crossings; the meander "
                           "route reaches the reference distribution exactly "
                           "at the same state (REPORT.md 4)."},
    (20, "meander"): {"z_final": 0.224, "z_build": 0.192, "actuator": 1.402,
                      "note": "lowest measured over the build densities tried "
                              "(rho 0.03-0.05); REPORT.md 4.5 puts the DP-20 "
                              "excess on the meander's kinks per strand, so "
                              "fewer waves (conformation.meander_waves) is "
                              "the untried lever below this."},
}


#: Routes whose rows were re-measured with :func:`topon.conformation.place`
#: and did not come back, and what stopped them. The rows of such a route
#: describe the validation scripts' placement only, and anything seeded from
#: them (``topon fit``) says so.
REMEASURED: dict[tuple[int, str], str] = {
    (20, "meander"): ("the place()-built meander builds at coil 1.40 and 1.51 "
                      "stopped at stage 2 on the push-off bond gate, each "
                      "with 3 persistent bonds at 1.3-1.4 sigma, so "
                      "neither DP-20 meander row has a place() counterpart"),
}


def actuator_name(placement: str) -> str:
    """Which knob the controller turns for this placement.

    The meander is steered by its coil ratio and the walk by its build
    density. They are the same physical quantity seen from two sides -- the box
    scales as ``rho^(-1/3)`` and the contour does not move -- but they are
    reported the way each route's evidence was recorded, so a calibration entry
    means what it says.
    """
    return "build_density" if placement == "walk" else "coil_ratio"


def calibration_for(dp: int, placement: str, state: str = "final",
                    protocol: str = "limit",
                    table: Sequence[CalibrationPoint] = CALIBRATION
                    ) -> list[CalibrationPoint]:
    """Entries for this route, nearest DP first, then by actuator.

    An exact DP match is used alone when there is one. Otherwise the nearest DP
    is used and the caller is expected to say so: Z per strand is strongly
    DP-dependent (0.18 at DP 20 against 1.32 at DP 100 on the same reference
    family), so a table row from another DP fixes the *slope* and not the
    level.
    """
    rows = [c for c in table
            if c.placement == placement and c.state == state
            and c.protocol == protocol]
    if not rows:
        return []
    exact = [c for c in rows if c.dp == dp]
    if exact:
        return sorted(exact, key=lambda c: c.actuator)
    nearest = min({c.dp for c in rows}, key=lambda d: abs(d - dp))
    return sorted([c for c in rows if c.dp == nearest],
                  key=lambda c: c.actuator)


def _power_law(points: Sequence[tuple[float, float]]):
    """``(A, b)`` of ``z = A x^b`` through the two points furthest apart in x.

    The widest pair gives the most stable exponent at this sample size:
    intermediate points are noise, and the shipped table is two rows wide in
    every case anyway.

    Returns ``None`` when the points cannot fix an exponent -- one point, two
    at the same actuator, or two whose Z is the same (which the DP-20 meander
    build-state rows are, 0.192 at two coil ratios).
    """
    usable = [(x, z) for x, z in points if x > 0 and z > 0]
    if len(usable) < 2:
        return None
    usable = sorted(usable)
    lo, hi = usable[0], usable[-1]
    if abs(math.log(hi[0]) - math.log(lo[0])) < 1e-9:
        return None
    b = (math.log(hi[1]) - math.log(lo[1])) / (math.log(hi[0]) - math.log(lo[0]))
    if abs(b) < 1e-6:
        return None
    A = lo[1] / (lo[0] ** b)
    return A, b


def seed_actuator(dp: int, placement: str, target_z: float,
                  table: Sequence[CalibrationPoint] = CALIBRATION,
                  state: str = "final") -> tuple[float, dict]:
    """Where to start, from the shipped table alone.

    Returns ``(actuator, note)``. The note records which rows were used, the
    exponent they gave and whether the answer is an extrapolation past the
    measured span -- which it usually is, because the shipped span is two
    points wide.
    """
    rows = calibration_for(dp, placement, state, "limit", table)
    if not rows:
        rows = calibration_for(dp, placement, state, "hardcore_min", table)
    if not rows:
        other = "build" if state == "final" else "final"
        rows = calibration_for(dp, placement, other, "limit", table)
    if not rows:
        raise ValueError(
            f"nothing measured for DP {dp} {placement}: give an explicit "
            f"coil_ratio or build_density to start from")

    pts = [(c.actuator, c.z) for c in rows]
    fit = _power_law(pts)
    note = {"rows": [asdict(c) for c in rows],
            "dp_used": rows[0].dp,
            "state_used": rows[0].state,
            "protocol_used": rows[0].protocol}
    if fit is None:
        # One row: keep its actuator and say the level is all that is known.
        note["exponent"] = None
        note["why"] = ("a single calibration row fixes a level, not a slope, "
                       "so the first round repeats it and the second round "
                       "gets the slope from the measurement")
        return float(rows[0].actuator), note

    A, b = fit
    x = float((target_z / A) ** (1.0 / b))
    lo = min(p[0] for p in pts)
    hi = max(p[0] for p in pts)
    note["exponent"] = round(b, 4)
    note["span"] = [lo, hi]
    note["extrapolating"] = not (lo <= x <= hi)
    return x, note


def solve_actuator(measured: Sequence[tuple[float, float]], target_z: float,
                   fallback_exponent: Optional[float] = None,
                   bounds: Optional[tuple[float, float]] = None
                   ) -> tuple[float, dict]:
    """The next actuator value, from what this graph has actually measured.

    ``measured`` is the rounds so far as ``(actuator, z)``. Two or more give a
    power law through the widest pair; one uses ``fallback_exponent`` (the
    table's) anchored on that point, which is a secant step with a borrowed
    slope. ``bounds`` clamps the answer, because the arithmetic will happily
    ask for a build density of 0.002 when the measurement it is extrapolating
    from is flat.
    """
    note: dict = {"points": [[float(x), float(z)] for x, z in measured]}
    fit = _power_law(measured)
    if fit is not None:
        A, b = fit
        note["exponent"] = round(b, 4)
        note["from"] = "measured"
    elif measured and fallback_exponent:
        x0, z0 = measured[-1]
        b = float(fallback_exponent)
        A = z0 / (x0 ** b)
        note["exponent"] = round(b, 4)
        note["from"] = "table exponent, anchored on the last round"
    else:
        raise ValueError(
            "cannot solve for the next actuator: one measured point and no "
            "exponent to borrow")

    x = float((target_z / A) ** (1.0 / b))
    if bounds is not None:
        lo, hi = bounds
        if not (lo <= x <= hi):
            note["clamped_from"] = x
            x = float(min(max(x, lo), hi))
    return x, note


def floor_warning(dp: int, placement: str, target_z: float,
                  floors: dict = FLOORS) -> Optional[str]:
    """A sentence naming the floor, when the target is under it.

    Returns ``None`` when the target is reachable by everything measured.
    """
    entry = floors.get((dp, placement))
    if entry is None:
        nearest = [k for k in floors if k[1] == placement]
        if not nearest:
            return None
        key = min(nearest, key=lambda k: abs(k[0] - dp))
        entry = dict(floors[key])
        entry["note"] = (f"measured at DP {key[0]}, not DP {dp}: "
                         + entry["note"])
    if target_z >= entry["z_final"]:
        return None
    msg = (f"target_Z {target_z:g} is below the lowest final-state Z the "
           f"{placement} route has reached at DP {dp} ({entry['z_final']:g}, "
           f"at {actuator_name(placement)} {entry['actuator']:g}). "
           f"{entry['note']}")
    if placement == "walk":
        msg += (" Switch conformation.placement to 'meander' to go lower; "
                "it is the shape, not the density, that sets the floor.")
    return msg


# ---------------------------------------------------------------------------
# The per-strand distribution
# ---------------------------------------------------------------------------

def hist_ks(z_per_strand, target_hist) -> dict:
    """KS test of the measured per-strand Z against a requested histogram.

    The histogram is expanded into a sample of the same size and compared with
    a two-sample KS, which is what the validation scripts do against the
    reference's own per-strand values (``compare_entanglement.py``). Z is a
    small integer, so the sample is almost all ties and the p-value is
    conservative; it is reported and never tuned to.
    """
    z = np.asarray(z_per_strand, int).ravel()
    h = np.asarray(target_hist, float).ravel()
    if z.size == 0 or h.size == 0 or h.sum() <= 0:
        return {"p": None, "statistic": None,
                "why": "nothing to compare"}
    from scipy import stats

    h = h / h.sum()
    counts = np.round(h * z.size).astype(int)
    counts[counts < 0] = 0
    if counts.sum() == 0:
        return {"p": None, "statistic": None, "why": "empty target sample"}
    synthetic = np.repeat(np.arange(len(counts)), counts)
    res = stats.ks_2samp(z, synthetic)
    measured = np.bincount(z, minlength=len(counts)) / z.size
    return {"p": float(res.pvalue), "statistic": float(res.statistic),
            "measured_hist": [round(float(x), 4) for x in measured],
            "target_hist": [round(float(x), 4) for x in h],
            "n": int(z.size)}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundPlan:
    """What one round asks the runner to build and measure."""

    round: int
    placement: str
    actuator: str                  # "coil_ratio" or "build_density"
    value: float
    coil_ratio: Optional[float] = None
    build_density: Optional[float] = None
    dp: int = 0
    seed: int = 1
    why: dict = field(default_factory=dict)


@dataclass
class RoundResult:
    """What the runner measured. Only ``z_final`` steers the loop.

    ``z_by_stage`` and ``bond_max_by_stage`` are recorded rather than acted on:
    the protocol's gate is the bond histogram, and Z is expected to move
    between stages even when nothing crosses.
    """

    z_final: float
    z_build: Optional[float] = None
    z_per_strand: Optional[Sequence[int]] = None
    z_by_stage: dict = field(default_factory=dict)
    bond_max_by_stage: dict = field(default_factory=dict)
    density_by_stage: dict = field(default_factory=dict)
    coil_ratio: Optional[float] = None
    build_density: Optional[float] = None
    guard: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _as_result(value) -> RoundResult:
    if isinstance(value, RoundResult):
        return value
    if isinstance(value, dict):
        known = {f for f in RoundResult.__dataclass_fields__}
        extra = {k: v for k, v in value.items() if k not in known}
        kept = {k: v for k, v in value.items() if k in known}
        out = RoundResult(**kept)
        out.extra = {**out.extra, **extra}
        return out
    raise TypeError(
        f"runner returned {type(value).__name__}; expected a RoundResult or a "
        f"dict with at least a 'z_final' key")


def _convert_knob(graph, dp: int, want: str, have: float, config) -> float:
    """Turn a build density into a coil ratio, or the other way about."""
    from topon.conformation.placement import (coil_ratio_of,
                                              density_for_coil_ratio)

    bond = float(getattr(config, "bond", 0.97))
    if want == "coil_ratio":
        return float(coil_ratio_of(graph, dp, float(have), bond))
    return float(density_for_coil_ratio(graph, dp, float(have), bond))


def _dp_of(graph, dp: Optional[int]) -> int:
    if dp is not None:
        return int(dp)
    vals = [int(d["dp"]) for *_e, d in graph.edges(data=True) if "dp" in d]
    if not vals:
        raise ValueError(
            "no DP given and the graph's edges carry none; pass dp=... ")
    return int(round(float(np.median(vals))))


def controller(graph, config, runner: Callable[[RoundPlan], object], *,
               dp: Optional[int] = None, seed: int = 1,
               bounds: Optional[tuple[float, float]] = None,
               table: Sequence[CalibrationPoint] = CALIBRATION,
               log: Optional[Callable[[str], None]] = print) -> dict:
    """Drive the build until the measured Z hits the target.

    ``config`` is a :class:`~topon.config.schema.ConformationConfig` (or
    anything with the same attributes). ``runner`` is called once per round
    with a :class:`RoundPlan` and must return a :class:`RoundResult`, or a dict
    carrying at least ``z_final``: it is the half that builds the system, runs
    the relaxation protocol and measures Z1+, which is deliberately not this
    module's business.

    The reading the loop closes on is ``config.entanglement.close_on``, and the
    default is the final state for the reason at the top of this module.
    ``bounds`` clamps the actuator, which matters more than it sounds: the
    first step is usually an extrapolation off the end of a two-point table,
    and an unclamped power law will cheerfully ask for a build density of
    0.002.

    Returns the manifest: every round's request and reading, the convergence
    verdict, the KS p-value against ``target_hist`` when one was asked for, and
    the densities at build and at the end. With no ``target_Z`` the loop runs
    exactly one round and reports it, which is how a single build is measured
    without a target.
    """
    dp = _dp_of(graph, dp)
    ent = config.entanglement
    placement = config.placement
    knob = actuator_name(placement)
    ctl = ent.controller
    state = getattr(ent, "close_on", "final")
    say = (lambda *_a, **_k: None) if log is None else log

    manifest: dict = {
        "stage": "conformation",
        "placement": placement,
        "dp": dp,
        "actuator": knob,
        "target_Z": ent.target_Z,
        "close_on": state,
        "tolerance": ctl.tolerance,
        "max_rounds": ctl.max_rounds,
        "seed": seed,
        "rounds": [],
        "warnings": [],
    }

    # Where to start: whatever the config says, else the shipped table.
    #
    # The config may name the other knob -- a meander build given a build
    # density, say. That is not a mistake and it is not ignored: the two are
    # one knob (the box scales as rho^(-1/3), the contour does not move), so
    # it is converted here and the conversion is recorded.
    start = config.coil_ratio if knob == "coil_ratio" else config.build_density
    converted = None
    if start is None:
        other = config.build_density if knob == "coil_ratio" else config.coil_ratio
        if other is not None:
            start = _convert_knob(graph, dp, knob, other, config)
            converted = {"given": ("build_density" if knob == "coil_ratio"
                                   else "coil_ratio"),
                         "value": float(other), "as": knob,
                         "converted_to": float(start)}
            say(f"  config gave {converted['given']} {other:g}; on this graph "
                f"at DP {dp} that is {knob} {start:.4g}")
    if start is not None:
        note = {"from": "config"}
        if converted:
            note["converted"] = converted
    elif ent.target_Z is not None:
        start, note = seed_actuator(dp, placement, float(ent.target_Z), table,
                                    state=state)
        say(f"  seed {knob} {start:.4g} from the calibration table "
            f"(exponent {note.get('exponent')})")
    else:
        raise ValueError(
            f"nothing to start from: set conformation.{knob} or "
            f"conformation.entanglement.target_Z")
    manifest["seed_note"] = note

    if ent.target_Z is not None and state == "final":
        warn = floor_warning(dp, placement, float(ent.target_Z))
        if warn:
            manifest["warnings"].append(warn)
            say(f"  [WARN] {warn}")

    table_rows = (calibration_for(dp, placement, state, "limit", table)
                  or calibration_for(dp, placement, state, "hardcore_min",
                                     table))
    table_fit = _power_law([(c.actuator, c.z) for c in table_rows])
    table_exponent = None if table_fit is None else table_fit[1]

    measured: list[tuple[float, float]] = []
    rounds = 1 if ent.target_Z is None else int(ctl.max_rounds)
    converged = False
    x = float(start)

    for r in range(1, rounds + 1):
        plan = RoundPlan(
            round=r, placement=placement, actuator=knob, value=float(x),
            coil_ratio=float(x) if knob == "coil_ratio" else None,
            build_density=float(x) if knob == "build_density" else None,
            dp=dp, seed=seed, why=dict(note),
        )
        # The runner is told which knob was turned, not both: handing it a
        # coil ratio *and* a density invites it to honour the wrong one, and
        # `place` refuses to take both for exactly that reason.
        say(f"  round {r}: {knob} {x:.5g}")
        result = _as_result(runner(plan))
        reading = result.z_final if state == "final" else result.z_build
        if reading is None:
            raise ValueError(
                f"the controller is closing on the {state} state and the "
                f"runner returned no z_{state}")
        reading = float(reading)
        measured.append((float(x), reading))

        entry = {
            "round": r,
            knob: float(x),
            "closed_on": reading,
            "z_final": float(result.z_final),
            "z_build": result.z_build,
            "z_by_stage": dict(result.z_by_stage),
            "bond_max_by_stage": dict(result.bond_max_by_stage),
            "density_by_stage": dict(result.density_by_stage),
            "coil_ratio": result.coil_ratio,
            "build_density": result.build_density,
            "guard": dict(result.guard),
        }
        if result.extra:
            entry["extra"] = dict(result.extra)
        if ent.target_hist and result.z_per_strand is not None:
            entry["hist_ks"] = hist_ks(result.z_per_strand, ent.target_hist)
        manifest["rounds"].append(entry)

        if ent.target_Z is None:
            break

        err = abs(reading - ent.target_Z) / max(ent.target_Z, 1e-12)
        entry["relative_error"] = round(float(err), 4)
        say(f"    Z ({state}) {reading:.4f} vs target {ent.target_Z:g} "
            f"({err * 100:.1f} %)")
        if err <= ctl.tolerance:
            converged = True
            break
        if r == rounds:
            break

        try:
            x, note = solve_actuator(measured, float(ent.target_Z),
                                     fallback_exponent=table_exponent,
                                     bounds=bounds)
        except ValueError as exc:
            # One point and no slope to borrow. That is a table gap, not a
            # failure of the build, so the loop stops and says which: guessing
            # a direction here would spend an hour of LAMMPS on a coin toss.
            msg = (f"stopped after round {r}: {exc}. Add a second "
                   f"{knob} by hand, or run with a wider max_rounds from a "
                   f"different starting point.")
            manifest["warnings"].append(msg)
            say(f"  [WARN] {msg}")
            break
        if note.get("clamped_from") is not None:
            msg = (f"round {r + 1}: the fit asked for {knob} "
                   f"{note['clamped_from']:.4g}, clamped into "
                   f"{bounds}")
            manifest["warnings"].append(msg)
            say(f"  [WARN] {msg}")

    manifest["converged"] = converged
    manifest["rounds_used"] = len(manifest["rounds"])
    if manifest["rounds"]:
        last = manifest["rounds"][-1]
        manifest["z_final"] = last["z_final"]
        manifest[knob] = last[knob]
        manifest["build_density"] = last.get("build_density")
        manifest["final_density"] = (last.get("density_by_stage") or {}).get(
            "quench")
    if ent.target_Z is not None and not converged:
        msg = (f"did not reach |Z - {ent.target_Z:g}| / {ent.target_Z:g} < "
               f"{ctl.tolerance:g} in {manifest['rounds_used']} rounds; "
               f"closest was {min((abs(z - ent.target_Z) for _x, z in measured), default=float('nan')):.4f}")
        manifest["warnings"].append(msg)
        say(f"  [WARN] {msg}")
    return manifest
