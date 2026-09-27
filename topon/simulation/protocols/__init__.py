"""Relaxation protocols: the gates that say whether one held.

The protocol *text* is written by
:class:`topon.writers.lammps_inputs.LammpsInputGenerator`, which is where
generated LAMMPS files belong. What lives here is everything that reads a
run back: the per-stage checkpoints, the acceptance gates, and the staged
runner that applies them as the run goes.
"""

from .gates import (BOND_GATE, Checkpoint, GateReport, RHO_SAME, Z_TOLERANCE,
                    check, read_checkpoint, z_hold)
from .staged import STAGE_TAGS, GateFailure, StagedRun, measure_stages

__all__ = [
    "BOND_GATE",
    "Checkpoint",
    "GateFailure",
    "GateReport",
    "RHO_SAME",
    "STAGE_TAGS",
    "StagedRun",
    "Z_TOLERANCE",
    "check",
    "measure_stages",
    "read_checkpoint",
    "z_hold",
]
