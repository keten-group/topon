"""
Topon Conformation Module

Coordinate generation and manipulation.

Two entry points, for the two kinds of build:

* :class:`ConformationManager` is the pipeline's stage 5 for a system that
  already has a data file -- it applies the displacement files, breaks the
  degeneracy and resolves overlaps.
* :func:`place` draws a bead-spring build from the graph itself: chain shape,
  build box and the gate every strand has to pass before it is written.
"""

from topon.conformation.manager import ConformationManager
from topon.conformation.placement import (
    BOND,
    GuardLimits,
    PlacedStrand,
    Placement,
    coil_ratio_of,
    density_for_coil_ratio,
    limits_from,
    place,
    strand_plans,
)

__all__ = [
    "ConformationManager",
    "BOND",
    "GuardLimits",
    "PlacedStrand",
    "Placement",
    "coil_ratio_of",
    "density_for_coil_ratio",
    "limits_from",
    "place",
    "strand_plans",
]
