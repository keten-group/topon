"""Entanglement conformation: routing chains so they actually interlock.

:mod:`~topon.conformation.entanglement.braid` builds a *waypoint braid* --
both partners follow anti-phase ellipses about one shared axis, which makes
the winding count a prescribed input rather than an emergent property of a
tuned bulge.

:mod:`~topon.conformation.entanglement.designed` puts those braids on chains
that are already placed, and refuses a request whose windings do not fit in
the strands' contour with the minimum DP that would carry it.
:mod:`~topon.conformation.entanglement.control` closes a loop on the *measured*
Z1+ of the relaxed system: ``controller`` is exported here, so the entry point
is ``topon.conformation.entanglement.controller(graph, config, runner)``.
"""

from topon.conformation.entanglement.allocation import (
    AllocatedContact,
    Allocation,
    ContactRequest,
    Rejection,
    allocate_contacts,
    compose_chain_path,
)
from topon.conformation.entanglement.control import (
    CALIBRATION,
    FLOORS,
    CalibrationPoint,
    RoundPlan,
    RoundResult,
    actuator_name,
    calibration_for,
    controller,
    floor_warning,
    hist_ks,
    seed_actuator,
    solve_actuator,
)
from topon.conformation.entanglement.designed import (
    DesignReport,
    PairOutcome,
    PairRequest,
    chords_of,
    contour_cost,
    nearest_image_of,
    requests_from_config,
    route_designed_pairs,
)
from topon.conformation.entanglement.braid import (
    BraidShape,
    Contact,
    braid_pair,
    braid_path,
    chord_closed_linking,
    far_closed_linking,
    closest_approach,
    feasible_window,
    gap_at,
    linking_number,
    make_contact,
    min_separation,
    plan_braid,
)

__all__ = [
    "CALIBRATION",
    "FLOORS",
    "CalibrationPoint",
    "RoundPlan",
    "RoundResult",
    "actuator_name",
    "calibration_for",
    "controller",
    "floor_warning",
    "hist_ks",
    "seed_actuator",
    "solve_actuator",
    "DesignReport",
    "PairOutcome",
    "PairRequest",
    "chords_of",
    "contour_cost",
    "nearest_image_of",
    "requests_from_config",
    "route_designed_pairs",
    "AllocatedContact",
    "Allocation",
    "BraidShape",
    "Contact",
    "ContactRequest",
    "Rejection",
    "allocate_contacts",
    "compose_chain_path",
    "braid_pair",
    "braid_path",
    "chord_closed_linking",
    "far_closed_linking",
    "closest_approach",
    "feasible_window",
    "gap_at",
    "linking_number",
    "make_contact",
    "min_separation",
    "plan_braid",
]
