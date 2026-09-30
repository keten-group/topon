"""Inverse design: from an existing network to a config that regenerates it.

``topon fit`` and ``topon generate --verify``. The steps, one module each:

- :mod:`~topon.inverse.measure` reads the reference (an end-linked LAMMPS
  data file, an NPZ dual graph or a strand graph) and measures what a
  config has to reproduce
- :mod:`~topon.inverse.scaffold` picks the cubic cell and sweeps the
  neighbour cutoff through the pipeline's own stages 1 to 3
- :mod:`~topon.inverse.fit` writes the config and flags what it cannot match
- :mod:`~topon.inverse.verify` regenerates a config and holds it against the
  reference

Nothing here runs dynamics. The entanglement target a fit writes is a
final-state number, checked only on a relaxed build (``--relaxed``).
"""

from topon.inverse.fit import FitError, FitResult, fit, format_fit, write_fit
from topon.inverse.measure import Measurement, Reference, measure, read_reference
from topon.inverse.verify import format_verify, verify, write_verify

__all__ = [
    "FitError",
    "FitResult",
    "Measurement",
    "Reference",
    "fit",
    "format_fit",
    "format_verify",
    "measure",
    "read_reference",
    "verify",
    "write_fit",
    "write_verify",
]
