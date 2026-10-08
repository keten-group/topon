"""An atomistic Z target met in the relaxed network, not only in the build.

``conformation.entanglement.target_Z`` on the atomistic route is met on the
build by the coil radius (:meth:`topon.pipeline.Pipeline._place_for_target`):
the radius is searched on the drawn network until Z1+ reads the target.
The relaxation keeps every entanglement (no backbone passage at any stage),
but Z1+ is not a topological count and part of a coil's kinks slide off as
the helix unwinds: on DP-30 PDMS the Z = 1 build read 0.958 and 0.711 after
the deck's NVT (74 %), the Z = 2 build 2.022 and 1.765 (87 %), where the
meander keeps 94-99 %.

So the loop measures what this network keeps and builds again, the same
network with the same draws each time: the global streams are seeded from
``seed`` and the study keeps its name, which keys the placement's own
stream, so only the radius moves. Round ``k`` is written to
``<output_dir>/<name>_z<k>/<name>``. Every round is relaxed with the deck and read after the NVT stage,
at the chemistry's own density (the deck's 1000 NPT steps move the density
by 5-7 %, and Z1+ with it; that reading is recorded as well).

- Round 1 builds at the target: the build search finds the radius.
- Round 2 builds at ``target / kept``, ``kept`` being round 1's relaxed Z
  over its build Z; the build search finds the radius again.
- From round 3 the relaxed readings themselves say where to go: the radius
  is interpolated between the two rounds that bracket the target (or the
  last two) on relaxed Z against radius, and the network is built at that
  radius directly. What a network keeps changes with the radius (74 % at
  4.46 A and 78 % at 5.23 A on the DP-30 target 1), so a fixed ``kept``
  cannot land closer than that change, and the build search's own
  overshoot no longer enters.

The loop stops when the relaxed Z is within the tolerance of the target:
``controller.tolerance`` when the config sets it, else 5 %, since Z1+ over
four seeds of a DP-30 atomistic network scatters by 2-3 % (the bead-spring
default of 15 % is the scatter of a 95 000-bead build). Nothing is looked
up: every correction is measured on the network being built.

The runner is passed in, so the loop is testable without LAMMPS: it takes
the run directory and returns the readings. The default runs the atomistic
deck with its gates (:class:`~topon.simulation.protocols.atomistic.AtomisticRun`).
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = ["TargetRound", "RelaxedTarget", "relax_to_target", "deck_runner",
           "next_radius", "TOLERANCE"]

#: The checkpoint the loop closes on, and the one it records beside it.
CLOSE_ON = "stage3_nvt"
ALSO = "stage3_npt"

#: The loop's band when the config does not set ``controller.tolerance``.
TOLERANCE = 0.05

#: The thinnest coil the build search tries, A.
R_MIN = 1.5


@dataclass
class TargetRound:
    """One build and its relaxation."""

    how: str                        # "aim" (build searched) or "radius" (built at it)
    aim: Optional[float]            # the build's target_Z, for an "aim" round
    run_dir: str
    radius: Optional[float]         # A, the coil radius the build used
    z_build: Optional[float]        # Z1+ per bridge of the settled build
    z_relaxed: Optional[float]      # after the NVT stage
    z_npt: Optional[float] = None   # after the NPT stage
    passed: Optional[bool] = None   # the deck's own gates
    status: Optional[str] = None    # the build search's, for an "aim" round


@dataclass
class RelaxedTarget:
    target: float
    tolerance: float
    met: bool
    rounds: list = field(default_factory=list)

    @property
    def run_dir(self) -> Optional[str]:
        """The last round's run directory: the network that was kept."""
        return self.rounds[-1].run_dir if self.rounds else None

    @property
    def z_relaxed(self) -> Optional[float]:
        return self.rounds[-1].z_relaxed if self.rounds else None

    def as_dict(self) -> dict:
        return {"target": self.target, "tolerance": self.tolerance, "met": self.met,
                "run_dir": self.run_dir, "rounds": [asdict(r) for r in self.rounds]}


def deck_runner(omp: int = 4, executable: str = "lmp", z1_seeds: int = 4,
                verbose: bool = True) -> Callable:
    """Run the atomistic deck with its gates; returns ``{tag: Z, "passed": bool}``."""
    from topon.simulation.protocols.atomistic import AtomisticRun

    def run(run_dir) -> dict:
        r = AtomisticRun(run_dir, executable=executable, omp=omp, z1=True,
                         z1_seeds=z1_seeds, stop_on_fail=False)
        report = r.run(verbose=verbose)
        out = {tag: cp.z_bridge for tag, cp in r.checkpoints.items()}
        out["passed"] = bool(report.passed)
        return out

    return run


def next_radius(rounds, target: float) -> Optional[float]:
    """The radius the relaxed readings point at, or None if they cannot say.

    Relaxed Z against radius, interpolated (or extrapolated, at most half the
    span beyond the rounds) between the two rounds that bracket the target,
    or the last two when none do. None when fewer than two rounds have a
    radius and a reading, or when the two chosen do not rise with the radius.
    """
    pts = [(r.radius, r.z_relaxed) for r in rounds
           if r.radius is not None and r.z_relaxed is not None]
    if len(pts) < 2:
        return None
    below = [p for p in pts if p[1] < target]
    above = [p for p in pts if p[1] >= target]
    if below and above:
        a = max(below, key=lambda p: p[1])
        b = min(above, key=lambda p: p[1])
    else:
        a, b = pts[-2], pts[-1]
    (ra, za), (rb, zb) = sorted([a, b])
    if rb - ra < 1e-6 or zb <= za:
        return None
    r = ra + (target - za) * (rb - ra) / (zb - za)
    span = rb - ra
    return float(max(R_MIN, min(rb + 0.5 * span, max(ra - 0.5 * span, r))))


def relax_to_target(config: dict, raw_config: Optional[dict] = None, *,
                    runner: Optional[Callable] = None, seed: int = 7,
                    max_rounds: Optional[int] = None,
                    tolerance: Optional[float] = None,
                    verbose: bool = True) -> RelaxedTarget:
    """Build, relax and build again until the relaxed network reads ``target_Z``.

    ``config`` is the study's config as a dict, with
    ``conformation.entanglement.target_Z`` set and an atomistic chemistry.
    Round ``k`` is built under ``<output_dir>/<name>_z<k>`` with the
    study's own name, from ``seed``. ``max_rounds`` defaults to the controller's
    ``max_rounds``, ``tolerance`` to ``controller.tolerance`` when the config
    sets it and :data:`TOLERANCE` otherwise; ``runner(run_dir)`` returns
    ``{checkpoint tag: Z per bridge, "passed": bool}`` and defaults to
    :func:`deck_runner`.
    """
    from topon.config.schema import ToponConfig
    from topon.core.manifest import read_manifest, record_stage
    from topon.pipeline import Pipeline

    base = ToponConfig(**config)
    if base.chemistry.model_type != "atomistic":
        raise ValueError("relax_to_target is the atomistic route's; the bead-spring "
                         "route has conformation.entanglement.controller")
    ent = base.conformation.entanglement
    if ent.target_Z is None:
        raise ValueError("conformation.entanglement.target_Z is not set")
    target = float(ent.target_Z)
    if tolerance is None:
        tolerance = (float(ent.controller.tolerance)
                     if "tolerance" in ent.controller.model_fields_set else TOLERANCE)
    rounds_max = int(max_rounds or ent.controller.max_rounds)
    runner = runner or deck_runner(verbose=verbose)
    result = RelaxedTarget(target=target, tolerance=float(tolerance), met=False)

    for k in range(1, rounds_max + 1):
        radius = next_radius(result.rounds, target) if k >= 3 else None
        if k == 1:
            aim = target
        elif radius is None:
            last = result.rounds[-1]
            aim = target / max(last.z_relaxed / last.z_build, 1e-3)
        cfg = copy.deepcopy(config)
        conf = cfg.setdefault("conformation", {})
        ent_cfg = conf.setdefault("entanglement", {})
        if radius is None:
            ent_cfg["target_Z"] = aim
        else:
            ent_cfg.pop("target_Z", None)
            conf.update(atomistic_placement="coil", atomistic_coil_radius=radius)
        # the study keeps its name, which keys the placement stream; the
        # round gets a folder of its own
        cfg["study"] = {**cfg.get("study", {}), "name": base.study.name,
                        "output_dir": str(Path(base.study.output_dir)
                                          / f"{base.study.name}_z{k}")}
        random.seed(seed)
        np.random.seed(seed)
        pipe = (Pipeline(ToponConfig(**cfg), raw_config=copy.deepcopy(raw_config))
                if raw_config else Pipeline(ToponConfig(**cfg)))
        pipe.run()
        placement = read_manifest(pipe.output_dir)["stages"]["placement"]
        placed = placement.get("z_target") or {}
        readings = runner(pipe.output_dir)
        rnd = TargetRound(
            how="aim" if radius is None else "radius",
            aim=aim if radius is None else None, run_dir=str(pipe.output_dir),
            radius=placed.get("radius", radius),
            z_build=placed.get("z_settled", readings.get("stage0_build")),
            z_relaxed=readings.get(CLOSE_ON), z_npt=readings.get(ALSO),
            passed=readings.get("passed"),
            status=(placed.get("rounds") or [{}])[-1].get("status"))
        result.rounds.append(rnd)
        if verbose:
            how = (f"build aimed at {aim:.3f}" if radius is None
                   else f"built at the radius the readings point at")
            print(f"  round {k}: {how} (radius {rnd.radius:.2f} A), build "
                  f"{rnd.z_build}, relaxed {rnd.z_relaxed} (NPT {rnd.z_npt})", flush=True)
        if rnd.z_relaxed is None or not rnd.z_build:
            break
        if abs(rnd.z_relaxed - target) <= tolerance * target:
            result.met = True
            break
        if (rnd.status == "ceiling" and rnd.z_relaxed < target) or (
                rnd.status == "floor" and rnd.z_relaxed > target):
            break           # the build cannot go further that way

    try:
        record_stage(Path(result.run_dir), "z_target_relaxed", result.as_dict(),
                     study=base.study.name)
    except Exception as exc:                       # the manifest is a courtesy
        if verbose:
            print(f"  (could not record the loop in the manifest: {exc})")
    if verbose:
        print(f"  target_Z {target:g}: {'met' if result.met else 'not met'} within "
              f"{tolerance:.0%} after {len(result.rounds)} round(s), relaxed "
              f"{result.z_relaxed} in {result.run_dir}", flush=True)
    return result
