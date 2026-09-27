"""Run a relaxation protocol stage by stage, gating as it goes.

The gate is checked after every stage that writes a checkpoint, not once at
the end, because a run that has already threaded a bond has nothing left to
prove: the entanglement state it was built to carry is gone, and the
remaining stages are hours of machine time spent on a system that will be
thrown away. ``stop_on_fail`` (the default) stops there and says which stage
and how many bonds.

The stage list is not written here. It comes from the generator that wrote
the scripts (:meth:`LammpsInputGenerator.stages`), so a protocol that gains a
stage does not need this file edited too.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .gates import (Z_TOLERANCE, Checkpoint, GateReport, check,
                    read_checkpoint)

#: Stage tags, in run order. The tags name the *state*, not the script, so a
#: measurement is labelled the same whichever protocol produced it.
STAGE_TAGS = ("stage1_min", "stage2_pushoff", "stage3_build",
              "stage4_final", "stage5_quench", "stage6_quartic")


@dataclass
class StageResult:
    tag: str
    script: str
    data_file: Path
    seconds: float = 0.0
    returncode: int = 0
    checkpoint: Checkpoint | None = None


@dataclass
class StagedRun:
    """A protocol run: the scripts, what each left behind, and the verdict."""

    sim_dir: Path
    stages: list = field(default_factory=list)   # [(script, data file)]
    executable: str = "lmp"
    n_procs: int = 1
    use_mpi: bool = False
    omp: int = 0
    stop_on_fail: bool = True
    results: list = field(default_factory=list)
    report: GateReport | None = None

    def __post_init__(self):
        self.sim_dir = Path(self.sim_dir)

    # ---------------- running ----------------

    def _command(self, script, log_file):
        cmd = []
        if self.use_mpi:
            cmd += ["mpirun", "-np", str(self.n_procs)]
        cmd.append(self.executable)
        if self.omp:
            cmd += ["-sf", "omp", "-pk", "omp", str(self.omp)]
        cmd += ["-in", script, "-log", log_file]
        return cmd

    def run(self, z_by_stage=None, gate_z=None, mode="instant",
            verbose=True) -> GateReport:
        """Run every stage in order, gating after each checkpoint.

        Args:
            z_by_stage: ``{tag: Z per bridge}`` from an external Z1+ pass,
                when there is one. Z1+ is not in this repository (its licence
                forbids redistribution), so the measurement is always someone
                else's to supply; the gate only decides what to do with it.
            gate_z: force the Z gate on or off. ``None`` decides from the
                densities.
            mode: ``"instant"`` (the acceptance criterion as written) fails
                on any bond over the limit; ``"persistent"`` fails only on a
                bond over it at more than one stage. See :mod:`.gates`.
            verbose: print a line per stage as it finishes.

        Returns:
            The :class:`GateReport`. Also on ``self.report``.
        """
        import os

        env = dict(os.environ, OMP_NUM_THREADS=str(self.omp or 1))
        measured: dict = {}
        self.results = []

        for (script, data_name), tag in zip(self.stages, STAGE_TAGS):
            path = self.sim_dir / script
            if not path.exists():
                raise FileNotFoundError(f"protocol script missing: {path}")
            out = StageResult(tag=tag, script=script,
                              data_file=self.sim_dir / data_name)
            # A stale checkpoint from an earlier, longer run would be read as
            # this run's result: the files are named for the stage, not the
            # run.
            out.data_file.unlink(missing_ok=True)

            t0 = time.time()
            proc = subprocess.run(self._command(script, f"log.{script}.txt"),
                                  cwd=str(self.sim_dir), capture_output=True,
                                  text=True, env=env)
            out.seconds = time.time() - t0
            out.returncode = proc.returncode
            (self.sim_dir / f"out.{script}.txt").write_text(
                proc.stdout + proc.stderr, encoding="utf-8")
            self.results.append(out)

            if proc.returncode != 0 or not out.data_file.exists():
                err = [l for l in (proc.stdout + proc.stderr).splitlines()
                       if l.startswith("ERROR") or "Last input" in l]
                raise RuntimeError(
                    f"{script} failed (exit {proc.returncode}) after "
                    f"{out.seconds:.0f} s: " + " | ".join(err[-3:] or ["no ERROR line"]))

            out.checkpoint = read_checkpoint(out.data_file)
            measured[tag] = out.checkpoint
            if verbose:
                s = out.checkpoint.summary()
                T = "-" if s["temperature"] is None else f"{s['temperature']:.3f}"
                print(f"  {tag:<16} {out.seconds:6.0f} s  rho {s['density']:.4f}"
                      f"  T {T}  bond max {s['bond_max']:.3f}"
                      f"  >1.2 sigma: {s['stretched']}", flush=True)

            if self.stop_on_fail:
                running = check(measured, z_by_stage=z_by_stage,
                                gate_z=False, mode=mode)
                if not running.passed:
                    self.report = check(measured, z_by_stage=z_by_stage,
                                        gate_z=gate_z, mode=mode)
                    raise GateFailure(self.report)

        self.report = check(measured, z_by_stage=z_by_stage,
                            gate_z=gate_z, mode=mode)
        return self.report


class GateFailure(RuntimeError):
    """A gate rejected the run. Carries the report that says why."""

    def __init__(self, report: GateReport):
        self.report = report
        super().__init__("\n" + report.render())


def measure_stages(sim_dir, stages, z_by_stage=None, gate_z=None,
                   mode="instant", z_tolerance=Z_TOLERANCE) -> GateReport:
    """Gate a protocol that has already been run.

    The same gates over checkpoints someone else produced -- a run on a
    cluster, or an older run being re-checked. Stages whose data file is
    missing are skipped rather than failed, so a partial run still reports on
    what it did reach.
    """
    sim_dir = Path(sim_dir)
    measured = {}
    for (_script, data_name), tag in zip(stages, STAGE_TAGS):
        f = sim_dir / data_name
        if f.exists():
            measured[tag] = read_checkpoint(f)
    if not measured:
        raise FileNotFoundError(
            f"no protocol checkpoints in {sim_dir}: expected "
            f"{', '.join(d for _s, d in stages)}")
    return check(measured, z_by_stage=z_by_stage, gate_z=gate_z, mode=mode,
                 z_tolerance=z_tolerance)
