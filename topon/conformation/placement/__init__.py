"""Chain placement: where every bead sits at the build state.

:func:`~topon.conformation.placement.chains.place` is the stage-5 entry point
for a bead-spring build. It draws each strand of a graph with one of three
shapes, sizes the build box from the bead count, and reports the gate every
strand has to pass before it is written.
"""

from topon.conformation.placement.chains import (
    BOND,
    PLACEMENTS,
    SELF_SEPARATION,
    GuardLimits,
    PlacedStrand,
    Placement,
    StrandPlan,
    bead_count,
    box_for_density,
    coil_ratio_of,
    density_for_coil_ratio,
    lattice_frame,
    limits_from,
    place,
    separate_coincident,
    strand_plans,
)

__all__ = [
    "BOND",
    "PLACEMENTS",
    "SELF_SEPARATION",
    "GuardLimits",
    "PlacedStrand",
    "Placement",
    "StrandPlan",
    "bead_count",
    "box_for_density",
    "coil_ratio_of",
    "density_for_coil_ratio",
    "lattice_frame",
    "limits_from",
    "place",
    "separate_coincident",
    "strand_plans",
]
