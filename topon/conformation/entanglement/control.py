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
The exponents in the shipped table are 0.22 for the DP-20 walk (place() with
the junction jitter) and 0.30 for the DP-100 walk, so the curve is
shallow and the controller's first move is usually its biggest. The DP-20
meander with the pinch fix shows no slope at all over the coils it can be built
at (final Z 0.2356 at 1.402, 0.2370 at 1.51, inside the seed spread), and
rows that cannot tell a slope from noise fix a level only. Two measured
points fix A and b; with one, the exponent comes from the shipped table.
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
    "steering_rows",
    "build_options",
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
    final-state points describe the minimiser as much as the placement. They
    steer only where the requested DP has nothing measured on the
    crossing-free deck (:func:`steering_rows`).

    ``"pushoff"`` is the protocol's own name for the same deck ``"limit"``
    names here. Each row keeps the label it was recorded under, and the
    lookup reads the two as one deck: :func:`calibration_for` asked for either
    returns both. Before 0.4.5 they were kept apart on purpose, because a single
    ``place()``-built row would otherwise have outranked the three-point
    ``hardcore_min`` fit of the DP-20 walk; the DP-20 rows have since been
    re-measured as a set, which is what that split was waiting for.

    ``builder`` is what drew the coordinates: ``"script"`` for
    ``bond_create_validation/scripts/``, ``"place"`` for
    :func:`topon.conformation.place`, which is what the controller steers.
    The two are not the same placement: on the same graph and density the
    validation script and ``place()`` put the placed-state Z at 0.0526 and
    0.0287 respectively, and after the push-off at 0.186 and 0.215. The table
    is keyed on post-protocol states for that reason, and a row says which
    side it came from so the gap stays visible.

    ``superseded_by`` is set on a row a later measurement replaced, naming
    what replaced it. Such a row is kept as the record of what was measured
    and never steers: :func:`calibration_for` leaves it out unless asked.
    ``note`` carries what the row needs to be read correctly, such as the
    build settings it was measured with. ``z`` is a mean over velocity seeds
    when the row says so, and ``z_sd`` is the standard deviation over them
    (0 for a single seed): the seeding draws no slope between two rows that
    differ by no more than it (:func:`_table_slope`).

    ``junction_jitter`` and ``settle_clearance`` are the ``conformation`` keys
    a ``place()`` row was built with (the pinch fix), 0 and ``None`` for a
    row built without them or by the scripts. They are part of what the knob
    means: the meander at coil 1.402 ends at 0.2356 with the fix and near
    0.260 without it, where the build pinches. :func:`build_options` reads
    them back for a config seeded from the row.
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
    note: str = ""
    superseded_by: str = ""
    z_sd: float = 0.0
    junction_jitter: float = 0.0
    settle_clearance: Optional[float] = None


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
#: The DP-20 rows that steer are ``builder="place"``: the five cases of
#: the validation scripts re-measured with :func:`topon.conformation.place`,
#: which is what the controller steers, with the pinch fix on. The rows
#: they replace are kept and carry ``superseded_by``; they never steer. The
#: DP-100 rows are still script-built.
#:
#: A second ``place()``-built point exists and is deliberately not a row:
#: DP 100 walk, rho_build 0.0894, final-state Z1+ 1.3081 against the reference
#: 1.32, KS p = 1.0000 on the per-strand histogram. Adding it would move the
#: DP-100 fit the controller extrapolates on, and confirming that did no harm
#: means re-running the controller, which needs MD.
#:
#: Since 0.4.5 ``place()`` builds differently from a graph whose edges carry ``dp``
#: or that records sol chains: a dangling chain is one bead longer (DP beads
#: under ``endlinked_dangling``, as the pipeline writes it) and the sol chains
#: are placed. That is every graph built from a config through the
#: pipeline's stages 1 to 3 (:func:`topon.inverse.scaffold.build_graph`, the
#: graph ``topon generate --verify`` regenerates). On the N100 fit it is
#: 100 499 beads where it was 100 226: on the walk route, steered by build
#: density, a cell 0.09 % longer on a side; on the meander, steered by coil
#: ratio, the cell follows the contours and the density moves instead. The
#: rows here are unaffected: each names a graph of the validation scripts, and
#: none of the 632 graphs under ``bond_create_validation/data`` carries an edge
#: ``dp`` or a sol record, so ``place()`` builds them bead for bead as before
#: (checked on six graphs, three of them the scripts'). The graph file of the
#: unlisted DP-100 point above is not recorded, so it is not known which side
#: of the change it sits on. A row measured on a config's graph from here on
#: is a build with its dangling and sol chains whole.
#:
#: An earlier figure of 1.9 at rho 0.05 came from the DP-30 6x6x6 pilot cell
#: and was never recomputed for the N20 one, where the value is 1.5. The table
#: stays keyed on the build density, which is unambiguous and is what every
#: run actually recorded.
CALIBRATION: tuple[CalibrationPoint, ...] = (
    # --- DP 20, N20 MIX 90/5/5 4-shell graph, final box rho 0.3075 ---
    # Re-measured with place() and the pinch fix (1 Oct 2026): the
    # graph data/sweep_cubic/N20__MIX_90-5-5_4_shells__matched__s1.gpickle,
    # placement seed 1, the default five-stage push-off compressed to 0.3075
    # and quenched to 0.4, through the controller's driver of the development
    # repository at 8 OpenMP threads. Meander: junction jitter 0.15 and a 1-sigma settle.
    # Walk: jitter 0.15 alone, since the settle does not converge on a random
    # walk and a settle-1.0 walk build is byte for byte the jitter-only one.
    # Final is stage 5, build is stage 3 (equilibrated at the build box).
    # Six runs passed the bond gate as written (no bond over 1.2 sigma from
    # stage 2 on). The coil-1.51 run failed it on one bond at 1.209 sigma at
    # stage 3 only and passes it on persistent bonds (its note).
    CalibrationPoint(20, "meander", 1.402, 0.2356, "final", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_m1402_v{1001,12345,777}_r1",
                     builder="place", z_sd=0.0036,
                     junction_jitter=0.15, settle_clearance=1.0,
                     note="mean of velocity seeds 1001 / 12345 / 777 "
                          "(0.2345 / 0.2327 / 0.2397); junction_jitter 0.15, "
                          "settle_clearance 1.0; no bond over 1.2 sigma at "
                          "any stage of any seed"),
    CalibrationPoint(20, "meander", 1.402, 0.2017, "build", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_m1402_v{1001,12345,777}_r1",
                     builder="place", z_sd=0.0048,
                     junction_jitter=0.15, settle_clearance=1.0,
                     note="stage 3, mean of velocity seeds 1001 / 12345 / 777 "
                          "(0.2033 / 0.1963 / 0.2056); junction_jitter 0.15, "
                          "settle_clearance 1.0"),
    CalibrationPoint(20, "meander", 1.510, 0.2370, "final", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_m1510_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     settle_clearance=1.0,
                     note="velocity seed 1001; junction_jitter 0.15, "
                          "settle_clearance 1.0; one bond at 1.209 sigma at "
                          "stage 3 only (1547-64544, no foreign bead within "
                          "1.14 sigma), thermal by the gates' persistence rule, so "
                          "stages 4-5 were run on from the stage-3 restart"),
    CalibrationPoint(20, "meander", 1.510, 0.1986, "build", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_m1510_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     settle_clearance=1.0,
                     note="stage 3, velocity seed 1001; junction_jitter 0.15, "
                          "settle_clearance 1.0"),
    CalibrationPoint(20, "walk", 0.035, 0.2365, "final", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0035_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="velocity seed 1001; junction_jitter 0.15, no settle; "
                          "24 dangling strands drawn with bonds to 1.52 sigma "
                          "(their lattice chord is longer than their contour "
                          "at this box, as in the 2026-09-21 build), none over "
                          "1.2 sigma from stage 1 on"),
    CalibrationPoint(20, "walk", 0.035, 0.2220, "build", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0035_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="stage 3, velocity seed 1001; junction_jitter 0.15, "
                          "no settle"),
    CalibrationPoint(20, "walk", 0.095, 0.2891, "final", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0095_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="velocity seed 1001; junction_jitter 0.15, no settle"),
    CalibrationPoint(20, "walk", 0.095, 0.2408, "build", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0095_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="stage 3, velocity seed 1001; junction_jitter 0.15, "
                          "no settle"),
    CalibrationPoint(20, "walk", 0.145, 0.3228, "final", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0145_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="velocity seed 1001; junction_jitter 0.15, no settle"),
    CalibrationPoint(20, "walk", 0.145, 0.2950, "build", "pushoff",
                     "N20_MIX90_4sh", "data/runs/v88_w0145_v1001_r1",
                     builder="place", junction_jitter=0.15,
                     note="stage 3, velocity seed 1001; junction_jitter 0.15, "
                          "no settle"),

    # --- the DP-20 rows the place() rows replaced, kept as the record -----
    # Script-built: the crossing-free protocol (--stage1 limit), re-quenched
    # to T = 0.4.
    CalibrationPoint(20, "meander", 1.402, 0.224, "final", "limit",
                     "N20_MIX90_4sh", "REPORT.md 4.4, runs/N20_v3b",
                     superseded_by="place() row at coil 1.402"),
    CalibrationPoint(20, "meander", 1.510, 0.239, "final", "limit",
                     "N20_MIX90_4sh", "REPORT.md 4.4, runs/N20_v3",
                     superseded_by="place() row at coil 1.51"),
    # Build state, same runs: the placement on its own, before compression.
    CalibrationPoint(20, "meander", 1.402, 0.192, "build", "limit",
                     "N20_MIX90_4sh",
                     "data/measure_N20_v3b_stage3_build.json",
                     superseded_by="place() row at coil 1.402"),
    CalibrationPoint(20, "meander", 1.510, 0.192, "build", "limit",
                     "N20_MIX90_4sh",
                     "data/measure_N20_v3_stage3_build.json",
                     superseded_by="place() row at coil 1.51"),
    # Random walk, minimiser protocol. Build-state column of REPORT.md 4.
    CalibrationPoint(20, "walk", 0.145, 0.300, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r145",
                     superseded_by="place() row at rho 0.145"),
    CalibrationPoint(20, "walk", 0.095, 0.250, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r095",
                     superseded_by="place() row at rho 0.095"),
    CalibrationPoint(20, "walk", 0.035, 0.232, "build", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r035",
                     superseded_by="place() row at rho 0.035"),
    CalibrationPoint(20, "walk", 0.145, 0.335, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r145",
                     superseded_by="place() row at rho 0.145"),
    CalibrationPoint(20, "walk", 0.095, 0.288, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r095",
                     superseded_by="place() row at rho 0.095"),
    CalibrationPoint(20, "walk", 0.035, 0.262, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_mix90_4sh_r035",
                     superseded_by="place() row at rho 0.035"),
    # place() without the pinch fix, 2026-09-21: the one of five that passed
    # the bond gate then (final 0.2598 against the script's 0.262).
    CalibrationPoint(20, "walk", 0.035, 0.2598, "final", "pushoff",
                     "N20_MIX90_4sh",
                     "tests/output/v54_2/recal_w0035/controller.json",
                     builder="place",
                     superseded_by="place() row at rho 0.035, the same "
                                   "case with junction jitter 0.15"),
    CalibrationPoint(20, "walk", 0.035, 0.2388, "build", "pushoff",
                     "N20_MIX90_4sh",
                     "tests/output/v54_2/recal_w0035/controller.json",
                     builder="place",
                     superseded_by="place() row at rho 0.035, the same "
                                   "case with junction jitter 0.15"),
    # Not replaced: place() cannot build this coil on this graph (below
    # about 1.37 the longest chords outgrow their contour; 1.35 fails the
    # gate on 23 strands). Minimiser protocol, so it seeds nothing while the
    # place() meander rows exist.
    CalibrationPoint(20, "meander", 1.274, 0.225, "final", "hardcore_min",
                     "N20_MIX90_4sh", "REPORT.md 4, runs/N20_..._r030_meander",
                     note="script-built; no place() counterpart"),

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
#: At DP 20 both floors are ``place()`` builds with the pinch fix, and the
#: floors coincide: the walk at build density 0.035 ends at 0.2365 and the
#: meander at coil 1.402 at 0.2356 (three seeds, sd 0.0036). The routes do not:
#: at one build density the meander is the lower (the walk's power law gives
#: 0.2435 at 0.040, where the meander gives 0.2356), but the walk is built
#: further down, at coil 1.341. Against ``place()`` without the fix both drop
#: (walk 0.2598 to 0.2365 at 0.035; meander 0.260 to 0.2356 at 1.402, that one
#: a gate-off run with a pinch in it); against the validation scripts' rows the
#: walk is lower and the meander higher (0.224), which is the change of builder,
#: not the fix. Neither route gets to the reference's 0.178. ``z_sd`` is the
#: seed spread of the floor's row (0 for one seed); :func:`floor_warning`
#: compares two floors against it.
FLOORS: dict[tuple[int, str], dict] = {
    (20, "walk"): {"z_final": 0.2365, "z_build": 0.2220, "actuator": 0.035,
                   "z_sd": 0.0,
                   "note": "place() with junction jitter 0.15, one velocity "
                           "seed. Build density 0.035 is coil 1.341 on "
                           "this graph, where 30 lattice chords are longer than "
                           "their contour (24 strands are drawn overstretched "
                           "after the jitter, and the push-off pulls them in); "
                           "the meander's settle does not converge there."},
    (20, "meander"): {"z_final": 0.2356, "z_build": 0.2017, "actuator": 1.402,
                      "z_sd": 0.0036,
                      "note": "place() with junction jitter 0.15 and a 1-sigma "
                              "settle, mean of three velocity seeds. "
                              "At coil 1.37 and below (build density 0.037) 30 "
                              "lattice chords of this graph are longer than "
                              "their contour (none at 1.38) and the settle does "
                              "not converge (coil 1.35: 23 strands fail the "
                              "gate), and between "
                              "1.402 and 1.51 Z does not move beyond the seed "
                              "spread (0.2370 at 1.51). REPORT.md 4.5 puts the "
                              "DP-20 excess on the meander's kinks per strand, "
                              "so fewer waves (conformation.meander_waves) is "
                              "the untried lever below this."},
}


#: Routes whose rows were re-measured with :func:`topon.conformation.place`
#: and did not come back, and what stopped them. The rows of such a route
#: describe the validation scripts' placement only, and anything seeded from
#: them (``topon fit``) says so. Empty since 0.4.5: the DP-20 meander builds
#: that stopped on the bond gate on 2026-09-21 pass it with the pinch fix, and
#: every DP-20 route now has place() rows.
REMEASURED: dict[tuple[int, str], str] = {}


def actuator_name(placement: str) -> str:
    """Which knob the controller turns for this placement.

    The meander is steered by its coil ratio and the walk by its build
    density. They are the same physical quantity seen from two sides -- the box
    scales as ``rho^(-1/3)`` and the contour does not move -- but they are
    reported the way each route's evidence was recorded, so a calibration entry
    means what it says.
    """
    return "build_density" if placement == "walk" else "coil_ratio"


#: Protocol labels that name one deck. The crossing-free push-off was first
#: labelled ``"limit"`` and then ``"pushoff"``; rows keep the label they were
#: recorded under and are looked up as one.
_DECK = {"limit": "pushoff", "pushoff": "pushoff"}


def _deck(protocol: str) -> str:
    return _DECK.get(protocol, protocol)


def calibration_for(dp: int, placement: str, state: str = "final",
                    protocol: str = "limit",
                    table: Sequence[CalibrationPoint] = CALIBRATION,
                    include_superseded: bool = False
                    ) -> list[CalibrationPoint]:
    """Entries for this route, nearest DP first, then by actuator.

    An exact DP match is used alone when there is one. Otherwise the nearest DP
    is used and the caller is expected to say so: Z per strand is strongly
    DP-dependent (0.18 at DP 20 against 1.32 at DP 100 on the same reference
    family), so a table row from another DP fixes the *slope* and not the
    level.

    ``"limit"`` and ``"pushoff"`` are one deck, so asking for either returns
    both. A row with ``superseded_by`` set is left out unless
    ``include_superseded``: it is the record of a measurement that was
    replaced, not something to steer on.
    """
    rows = [c for c in table
            if c.placement == placement and c.state == state
            and _deck(c.protocol) == _deck(protocol)
            and (include_superseded or not c.superseded_by)]
    if not rows:
        return []
    exact = [c for c in rows if c.dp == dp]
    if exact:
        return sorted(exact, key=lambda c: c.actuator)
    nearest = min({c.dp for c in rows}, key=lambda d: abs(d - dp))
    return sorted([c for c in rows if c.dp == nearest],
                  key=lambda c: c.actuator)


#: The order in which protocols seed the controller: the crossing-free deck,
#: then the minimiser's rows where nothing else was measured.
_SEED_ORDER = ("pushoff", "hardcore_min")


def steering_rows(dp: int, placement: str, state: str = "final",
                  table: Sequence[CalibrationPoint] = CALIBRATION
                  ) -> list[CalibrationPoint]:
    """The rows the controller seeds from: this DP first, then the nearest.

    Rows measured at the requested DP win over rows of another DP whatever
    protocol measured them, crossing-free deck first; only a DP with none of
    its own borrows the nearest DP's, in the same protocol order. Superseded
    rows never steer.

    Borrowing is a starting guess and no more. Z per strand grows steeply with
    DP, so another DP's rows fix a slope, not a level, and the nearest DP
    changes as rows are added: since 0.4.5 a DP-50 walk borrows the DP-20 walk
    rows (30 away) rather than DP 100's (50 away), which is what it borrowed
    while DP 20 had only minimiser rows.
    """
    for protocol in _SEED_ORDER:
        rows = [c for c in calibration_for(dp, placement, state, protocol,
                                           table) if c.dp == dp]
        if rows:
            return rows
    for protocol in _SEED_ORDER:
        rows = calibration_for(dp, placement, state, protocol, table)
        if rows:
            return rows
    return []


def _power_law(points: Sequence[tuple[float, float]]):
    """``(A, b)`` of ``z = A x^b`` through the two points furthest apart in x.

    The widest pair gives the most stable exponent at this sample size:
    intermediate points are noise, and most routes in the shipped table are
    two or three rows wide.

    Returns ``None`` when the points cannot fix an exponent -- one point, two
    at the same actuator, or two whose Z is the same. Whether a calibration
    slope is worth lending is :func:`_table_slope`'s question.
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


def build_options(rows: Sequence[CalibrationPoint],
                  actuator: Optional[float] = None) -> dict:
    """The ``conformation`` keys these rows were built with.

    A knob read off a calibration row means what it measured only with the
    build that row had: the DP-20 ``place()`` rows carry the pinch fix
    (junction jitter, and for the meander a settle), and a build at the same
    knob without it pinches and ends elsewhere. Returns
    ``{"junction_jitter": ..., "settle_clearance": ...}`` with only the keys
    that were on, from the rows at ``actuator`` when there are any, else from
    all of them, taking the row nearest ``actuator`` when they disagree.
    Empty for script-built rows and ``place()`` rows built without the fix.
    """
    rows = [c for c in rows if c.builder == "place"]
    if not rows:
        return {}
    if actuator is not None:
        at = [c for c in rows if abs(c.actuator - actuator) < 1e-9]
        rows = at or sorted(rows, key=lambda c: abs(c.actuator - actuator))[:1]
    first = rows[0]
    out: dict = {}
    if first.junction_jitter:
        out["junction_jitter"] = float(first.junction_jitter)
    if first.settle_clearance:
        out["settle_clearance"] = float(first.settle_clearance)
    return out


def _table_slope(rows: Sequence[CalibrationPoint]):
    """:func:`_power_law` over calibration rows, kept only when it is a slope.

    Two conditions, both about what the rows can tell apart. Z has to rise
    with the knob: no route has been seen to fall with it beyond its noise,
    and following a negative exponent sends the seed the wrong way. And the
    two rows the exponent is drawn through have to differ by more than the
    seed spread either carries (``z_sd``): a difference inside it is noise,
    and its exponent extrapolates to nonsense. A single-seed row carries
    ``z_sd`` 0 and is taken at its word, so two single-seed rows closer than
    their unmeasured spread still lend a slope. The DP-20 meander with
    the pinch fix is the case for both: final Z 0.2356 (sd 0.0036 over three
    seeds) at coil 1.402 and 0.2370 at 1.51, exponent 0.08, and at the build
    state 0.2017 (sd 0.0048) against 0.1986, exponent -0.21.

    Measured rounds (:func:`solve_actuator`) are not filtered here; the table
    only decides where to start and what slope to lend.
    """
    fit = _power_law([(c.actuator, c.z) for c in rows])
    if fit is None or fit[1] <= 0:
        return None
    usable = sorted((c for c in rows if c.actuator > 0 and c.z > 0),
                    key=lambda c: c.actuator)
    lo, hi = usable[0], usable[-1]
    if abs(hi.z - lo.z) <= max(lo.z_sd, hi.z_sd):
        return None
    return fit


def seed_actuator(dp: int, placement: str, target_z: float,
                  table: Sequence[CalibrationPoint] = CALIBRATION,
                  state: str = "final") -> tuple[float, dict]:
    """Where to start, from the shipped table alone.

    Returns ``(actuator, note)``. The note records which rows were used, the
    exponent they gave and whether the answer is an extrapolation past the
    measured span -- which it usually is, because the shipped span is two
    points wide.
    """
    rows = steering_rows(dp, placement, state, table)
    if not rows:
        other = "build" if state == "final" else "final"
        rows = calibration_for(dp, placement, other, "pushoff", table)
    if not rows:
        raise ValueError(
            f"nothing measured for DP {dp} {placement}: give an explicit "
            f"coil_ratio or build_density to start from")

    pts = [(c.actuator, c.z) for c in rows]
    fit = _table_slope(rows)
    note = {"rows": [asdict(c) for c in rows],
            "dp_used": rows[0].dp,
            "state_used": rows[0].state,
            "protocol_used": rows[0].protocol}
    if fit is None and _power_law(pts) is not None:
        # Several rows, and no slope between them that their seed spread
        # can tell from noise (see _table_slope).
        note["exponent"] = None
        note["why"] = ("the rows do not rise with the knob by more than their "
                       "seed spread, so they fix a level and not a slope: the "
                       "first round starts from the row nearest the target, "
                       "and with no slope to lend the loop stops after it "
                       "unless a second knob is measured")
        nearest = min(rows, key=lambda c: abs(c.z - target_z))
        return float(nearest.actuator), note
    if fit is None:
        # One row: keep its actuator and say the level is all that is known.
        note["exponent"] = None
        note["why"] = ("a single calibration row fixes a level, not a slope: "
                       "the first round repeats it, and with no slope to lend "
                       "the loop stops after it unless a second knob is "
                       "measured")
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

    Returns ``None`` when the target is reachable by everything measured. For
    the walk it also names the meander's floor at the DP the walk's floor was
    measured at: as the route to switch to when it is lower by more than the
    seed spread either floor carries (``z_sd``), and as no lower when it is
    not, which is the DP-20 case since 0.4.5 (0.2356 against 0.2365).
    """
    src = dp
    entry = floors.get((dp, placement))
    if entry is None:
        nearest = [k for k in floors if k[1] == placement]
        if not nearest:
            return None
        key = min(nearest, key=lambda k: abs(k[0] - dp))
        src = key[0]
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
        meander = floors.get((src, "meander"))
        if meander is None:
            msg += (" Nothing is measured for the meander route "
                    "(conformation.placement 'meander') here; it is the "
                    "other one to try.")
        else:
            spread = max(meander.get("z_sd", 0.0), entry.get("z_sd", 0.0))
            where = (f"{meander['z_final']:g} at DP {src} (coil_ratio "
                     f"{meander['actuator']:g})")
            if meander["z_final"] < entry["z_final"] - spread:
                msg += (f" The meander route has reached {where}; switch "
                        f"conformation.placement to 'meander' for that.")
            else:
                msg += (f" The meander route's floor is {where}, no lower "
                        f"within the seed spread.")
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

    table_rows = steering_rows(dp, placement, state, table)
    table_fit = _table_slope(table_rows) if table_rows else None
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
            msg = (f"stopped after round {r}: {exc}. Measure a second "
                   f"{knob} by hand; the two readings fix the slope "
                   f"(solve_actuator).")
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
