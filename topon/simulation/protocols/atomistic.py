"""Acceptance gates for the atomistic relaxation (DREIDING or CHARMM).

The two gates of the bead-spring protocol (:mod:`.gates`), in atomistic
units, over the checkpoints the three atomistic stages write.

**Is any backbone bond stretched?** Every bond along every strand's backbone,
junction to junction, is read against the equilibrium length of its bond
type. A stiff harmonic Si-O bond (350 kcal/mol/A^2) fluctuates by about 2 %
of its length at 300 K, so one held 15 % over at two stages is held open by
something: a strand driven through it, or two strands pinched at a
junction, the same two readings as on the bead-spring route. The gate is
*persistent* by default: it fails a bond long at more than one stage and
reports the per-stage counts beside it. A single checkpoint can be hot.
The DREIDING ramp (stage 2) runs ``nve/limit`` with no thermostat and ended
the smoke network at 964 K, where the backbone bonds read 1.032 +- 0.035 r0
and 3 of 2,640 sat at 1.15-1.17, none long at another stage and none with a
foreign atom near it. The build itself (``stage0_build``) is reported and
never gated, nor counted as evidence of persistence: the historic placement
puts atoms at a third of a bond length and the first stage is what makes a
molecule of them.

**Did any strand pass through another?** Topology changes only when one
backbone bond passes through another, so that is what is gated, stage 1
included: every stage dumps its backbone every
``simulation.backbone_dump_every`` steps and :mod:`topon.analysis.crossings`
reads the dump for passages. A single passage fails the stage, and the
report says where (step, bonds, strands, and whether the two strands share
a junction). A designed entanglement (``assignment.entanglements``) is kept
if no bond of one of its two strands passed through the other, which the
report says pair by pair.

Z1+ on the backbone, one point per repeat unit
(:mod:`topon.analysis.atomistic`), is measured at every checkpoint and
reported, and gated only for a run without dumps. It cannot certify a
state: on the DP-30 hard-backbone run (28 Sep 2026) it moved 0.763 to 0.781
to 0.719 per bridge through stage 3 with no passage anywhere, and on the
smoke network by a few of its 5 to 14 kinks where no backbone could cross.
Without dumps the historic Z gate runs instead: from the end of the epsilon
ramp (stage 2) Z per bridge must hold at every checkpoint at the same
density, and a failure there says look, not that something crossed.

Z1+ is not part of topon; without it Z is not measured and the other gates
still run.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from topon.analysis.atomistic import BOND_TOLERANCE, load_strand_record, read_atomistic

from .gates import RHO_SAME, Z_TOLERANCE, GateReport

#: Checkpoints in run order: ``(tag, folder, data file)``. ``conf`` is
#: ``03_Conformation`` and ``sim`` is ``04_Simulation`` of the run.
CHECKPOINTS = (
    ("stage0_build", "conf", "system_relaxed.data"),
    ("stage1_soft", "sim", "system_after_soft.data"),
    ("stage2_ramp", "sim", "system_ramped.data"),
    ("stage3_min", "sim", "system_minimized_final.data"),
    ("stage3_nvt", "sim", "after_nvt_real.data"),
    ("stage3_npt", "sim", "system_equilibrated.data"),
)

#: Which script writes which checkpoints.
SCRIPTS = (
    ("minimize_1_serial.in", ("stage1_soft",)),
    ("minimize_2_parallel.in", ("stage2_ramp",)),
    ("minimize_3_parallel.in", ("stage3_min", "stage3_nvt", "stage3_npt")),
)

#: First stage the bond gate fails a run at. Stage 1 ends in a minimisation
#: under the full bonded terms, so a backbone bond still stretched after it
#: is held open.
GATE_FROM = "stage1_soft"

#: First stage with excluded volume fully on; without dumps Z must hold from here.
Z_FROM = "stage2_ramp"

#: The backbone dump each script writes, and the checkpoint its passages
#: are reported under (stage 3's one dump covers its minimisation, NVT and NPT).
DUMPS = (
    ("stage1", "minimize_1_serial.in", "stage1_soft"),
    ("stage2", "minimize_2_parallel.in", "stage2_ramp"),
    ("stage3", "minimize_3_parallel.in", "stage3_npt"),
)


@dataclass
class AtomisticCheckpoint:
    """One checkpoint read down to what the gates need."""

    tag: str
    path: Path
    n_atoms: int
    density: float                       # g/cm^3
    temperature: Optional[float]         # K
    pairs: list                          # backbone bonds, (lo, hi) atom ids
    ratio: np.ndarray                    # length / r0, per backbone bond
    z_bridge: Optional[float] = None
    z1: Optional[dict] = None
    #: Z per bridge's spread over the Z1+ seeds (``z_bridge`` is their mean).
    z_bridge_sd: Optional[float] = None
    #: Z1+ partner pairs (strand numbers, 1-based in record order) found at
    #: every seed, those found at any seed, and the pairs of strands that
    #: share a junction. Pairs come and go with no crossing (Z1+ reads the
    #: shortest paths between the junctions where they sit), so these are
    #: notes; the passages are the gate.
    partner_pairs: Optional[set] = None
    partner_pairs_seen: Optional[set] = None
    share_junction: Optional[set] = None

    def stretched_pairs(self, tolerance: float = BOND_TOLERANCE) -> set:
        return {p for p, r in zip(self.pairs, self.ratio) if r > 1.0 + tolerance}

    def summary(self, tolerance: float = BOND_TOLERANCE) -> dict:
        r = self.ratio
        return {"file": self.path.name, "atoms": self.n_atoms,
                "density": round(self.density, 6),
                "bond_max": round(float(r.max()), 6) if len(r) else None,
                "bond_min": round(float(r.min()), 6) if len(r) else None,
                "stretched": len(self.stretched_pairs(tolerance)),
                "temperature": (None if self.temperature is None
                                else round(self.temperature, 2))}


def read_checkpoint(path, strands, tag: str = "", z1: bool = False,
                    z1_config=None, z1_seeds: int = 4) -> AtomisticCheckpoint:
    """Read one atomistic checkpoint, with Z1+ on its backbone if asked.

    Z1+ runs at ``z1_seeds`` seeds of the export's junction jitter: Z per
    bridge is their mean (``z_bridge_sd`` the spread), and the partner pairs
    are those found at every seed. One run swaps 10-15 of 55 pairs on an
    identical DP-30 configuration.
    """
    system = read_atomistic(path, strands)
    pairs, ratio = system.backbone_ratios()
    cp = AtomisticCheckpoint(tag=tag, path=Path(path), n_atoms=system.n_atoms,
                             density=system.mass_density,
                             temperature=system.temperature(),
                             pairs=pairs, ratio=ratio)
    if z1:
        from topon.analysis.z1plus import Z1PlusFailed, Z1PlusUnavailable, measure_seeds
        try:
            m = measure_seeds(system, seeds=z1_seeds, config=z1_config)
            z, res = m.first
            cp.z1 = z
            cp.z_bridge = m.z_bridge if m.z_bridge is not None else m.z_all
            cp.z_bridge_sd = m.z_bridge_sd
            if res.partners is not None:
                cp.partner_pairs = m.robust
                cp.partner_pairs_seen = m.seen
                cp.share_junction = _sharing_pairs(system)
        except (Z1PlusUnavailable, Z1PlusFailed):
            pass
    return cp


def _sharing_pairs(system) -> set:
    """Pairs of strands (1-based, in export order) that share a junction."""
    by_junction: dict = {}
    for k in sorted(system.strands):
        for j in {e for e in system.strands[k].ends if e is not None}:
            by_junction.setdefault(j, []).append(k)
    out = set()
    for ks in by_junction.values():
        out |= {(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]}
    return out


@dataclass
class AtomisticGateReport(GateReport):
    """The verdict, in atomistic units."""

    tolerance: float = BOND_TOLERANCE
    #: Per dumped stage: passages by kind, frames read, largest step.
    crossings: dict = field(default_factory=dict)
    #: Designed pairs ``"k-l"`` -> the stages at which their strands passed
    #: through each other (empty: the winding was kept).
    designed: dict = field(default_factory=dict)
    #: Z per bridge's spread over the Z1+ seeds, per checkpoint.
    z_sd_by_stage: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = super().summary()
        out["crossings"] = self.crossings
        out["designed"] = self.designed
        out["z_sd_by_stage"] = self.z_sd_by_stage
        return out

    def render(self) -> str:
        head = (f"{'stage':<16}{'g/cm3':>8}{'T (K)':>9}{'bond/r0':>10}"
                f"{'>+' + format(self.tolerance, '.0%'):>7}{'Z per bridge':>16}")
        rows = [head, "-" * len(head)]
        for tag, s in self.stages.items():
            z = self.z_by_stage.get(tag)
            sd = self.z_sd_by_stage.get(tag)
            t = s["temperature"]
            ztext = ("-" if z is None else f"{z:.3f}" if sd is None
                     else f"{z:.3f} +- {sd:.3f}")
            rows.append(
                f"{tag:<16}{s['density']:>8.4f}"
                f"{(float('nan') if t is None else t):>9.1f}"
                f"{(float('nan') if s['bond_max'] is None else s['bond_max']):>10.3f}"
                f"{s['stretched']:>7d}"
                f"{ztext:>16}")
        rows.append("")
        for tag, c in self.crossings.items():
            rows.append(f"backbone passages in {tag}: {c['passages']} "
                        f"({c['apart']} apart, {c['shared']} sharing a junction, "
                        f"{c['self']} within a strand) over {c['transitions']} "
                        f"frame intervals, largest step {c['max_step']:.2f} A")
        if self.designed:
            kept = sum(1 for v in self.designed.values() if not v)
            rows.append(f"designed pairs that kept their winding: {kept} of "
                        f"{len(self.designed)}")
        rows.append(f"backbone bonds long at more than one stage: "
                    f"{len(self.persistent)}")
        for line in self.notes:
            rows.append(f"  note: {line}")
        for line in self.details:
            rows.append(f"  {line}")
        rows.append("GATES PASSED" if self.passed else
                    "GATES FAILED:\n  " + "\n  ".join(self.failures))
        return "\n".join(rows)


def check(stages: dict, tolerance: float = BOND_TOLERANCE,
          gate_from: str = GATE_FROM, z_from: str = Z_FROM,
          z_tolerance: Optional[float] = None, mode: str = "persistent",
          rho_same: float = RHO_SAME, crossings: Optional[dict] = None,
          designed: Optional[dict] = None) -> AtomisticGateReport:
    """The gates over ``{tag: AtomisticCheckpoint}`` in run order.

    ``crossings`` is ``{dump tag: CrossingReport}`` for the stages that were
    dumped (:data:`DUMPS`); any passage in any of them fails the run, and
    with any stage dumped Z1+ is reported and not gated. ``designed`` is
    ``{(k, l): windings}`` (:func:`~topon.analysis.crossings.designed_pairs`).

    ``z_tolerance`` None is ``1.5 / n_bridges`` with the bead-spring 0.01 as
    a floor. A crossing adds (or removes) a kink on each of the two strands,
    which moves the mean per bridge by 2 / n_bridges, so on a small cell the
    default fails on a single crossing; from about 150 bridges up the floor
    takes over, the bead-spring gate's allowance. It is a first choice, not a
    calibration: how far Z1+ slides through a minimisation and 1 ps of NVT
    without any crossing is what the first atomistic runs will measure.

    Only checkpoints of known tag order are gated: the bond gate applies to
    every tag from ``gate_from`` on in :data:`CHECKPOINTS` order, whichever
    of them are present, and a gated checkpoint with no backbone bond read
    against an r0 fails, since passing it would pass nothing. The build
    (``stage0_build``) is neither gated nor evidence of persistence: its
    bonds are wherever the placement left them.
    """
    if mode not in ("instant", "persistent"):
        raise ValueError(f"unknown bond-gate mode {mode!r}")
    report = AtomisticGateReport(
        stages={t: cp.summary(tolerance) for t, cp in stages.items()},
        z_by_stage={t: cp.z_bridge for t, cp in stages.items()
                    if cp.z_bridge is not None},
        z_sd_by_stage={t: cp.z_bridge_sd for t, cp in stages.items()
                       if getattr(cp, "z_bridge_sd", None) is not None},
        mode=mode, tolerance=tolerance)
    tags = list(stages)
    order = {t: i for i, (t, _f, _n) in enumerate(CHECKPOINTS)}
    start = order.get(gate_from, len(CHECKPOINTS))
    gated = [t for t in tags if order.get(t, -1) >= start]

    seen: dict = {}
    for t in tags:
        if t == "stage0_build":
            continue
        for p in stages[t].stretched_pairs(tolerance):
            seen.setdefault(p, []).append(t)
    report.persistent = {p: ts for p, ts in seen.items() if len(ts) > 1}

    for t in gated:
        if not len(stages[t].ratio):
            report.failures.append(
                f"{t}: no backbone bond could be read against its r0 (no Bond "
                f"Coeffs in the file and no settings file in the run), so the "
                f"bond gate checked nothing")
            continue
        long = stages[t].stretched_pairs(tolerance)
        if not long:
            continue
        persistent = [p for p in long if p in report.persistent]
        line = (f"{t}: {len(long)} backbone bond(s) more than "
                f"{tolerance:.0%} over r0 (longest "
                f"{stages[t].ratio.max():.3f} r0), {len(persistent)} of them "
                f"long at another stage too")
        if mode == "instant" or persistent:
            report.failures.append(line)
        else:
            report.notes.append(line)
        for p in sorted(long)[:8]:
            r = dict(zip(stages[t].pairs, stages[t].ratio))[p]
            report.details.append(f"{t}: bond {p[0]}-{p[1]} at {r:.3f} r0")

    measured = bool(crossings)
    for tag, rep_ in (crossings or {}).items():
        counts = rep_.counts()
        report.crossings[tag] = {"passages": len(rep_.crossings), **counts,
                                 "transitions": rep_.transitions,
                                 "max_step": round(rep_.max_step, 4)}
        if not rep_.crossings:
            continue
        first = rep_.crossings[0]
        (a, b), (c, d) = first.bonds
        report.failures.append(
            f"{tag}: {len(rep_.crossings)} backbone passage(s) ({counts['apart']} "
            f"between strands that share no junction, {counts['shared']} between "
            f"strands that share one, {counts['self']} within a strand), the "
            f"first at step {first.step}: bond {a}-{b} of strand "
            f"{first.strands[0]} through bond {c}-{d} of strand {first.strands[1]}")
        for x in rep_.crossings[:8]:
            (a, b), (c, d) = x.bonds
            report.details.append(
                f"{tag}: step {x.step}, bond {a}-{b} (strand {x.strands[0]}) "
                f"through {c}-{d} (strand {x.strands[1]}), {x.kind}")
    if designed:
        for (k, l) in sorted(designed):
            report.designed[f"{k}-{l}"] = [
                tag for tag, rep_ in (crossings or {}).items()
                if any(set(x.strands) == {k, l} for x in rep_.crossings)]
        lost = {p: t for p, t in report.designed.items() if t}
        if not measured:
            report.notes.append(
                f"{len(designed)} designed pair(s), not checked: no stage was "
                f"dumped (simulation.backbone_dump_every)")
        elif lost:
            report.failures.append(
                "designed pair(s) whose strands passed through each other: "
                + ", ".join(f"{p} in {', '.join(t)}" for p, t in lost.items()))
        else:
            report.notes.append(
                f"all {len(designed)} designed pair(s) kept their winding: no "
                f"passage between the two strands of any")

    if z_from in stages and stages[z_from].z_bridge is not None:
        ref = stages[z_from]
        n_bridges = ((ref.z1 or {}).get("by_class") or {}).get("bridge", {}).get("n", 0)
        tol = (z_tolerance if z_tolerance is not None else
               max(Z_TOLERANCE, 1.5 / max(n_bridges, 1)))
        held = [ref.tag]
        for t in tags[tags.index(z_from) + 1:]:
            cp = stages[t]
            if cp.z_bridge is None:
                continue
            if abs(cp.density - ref.density) > rho_same * max(cp.density, ref.density):
                report.notes.append(
                    f"{t}: Z {cp.z_bridge:.3f} at {cp.density:.4f} g/cm3, not "
                    f"gated (density moved from {ref.density:.4f})")
                continue
            held.append(t)
            if cp.partner_pairs is not None and ref.partner_pairs is not None:
                # robust pairs of one checkpoint against every pair the
                # other had at any seed, so seed noise is not read as change
                ref_seen = ref.partner_pairs_seen or ref.partner_pairs
                cp_seen = cp.partner_pairs_seen or cp.partner_pairs
                new = cp.partner_pairs - ref_seen
                apart = new - (cp.share_junction or set())
                report.notes.append(
                    f"{t}: {len(new)} partner pair(s) new since {z_from} "
                    f"({len(apart)} between strands that share no junction), "
                    f"{len(ref.partner_pairs - cp_seen)} lost")
            if abs(cp.z_bridge - ref.z_bridge) > tol:
                if measured:
                    report.notes.append(
                        f"Z per bridge moved {ref.z_bridge:.3f} -> {cp.z_bridge:.3f} "
                        f"between {z_from} and {t}; not gated, the passages are")
                else:
                    report.failures.append(
                        f"Z per bridge moved {ref.z_bridge:.3f} -> {cp.z_bridge:.3f} "
                        f"between {z_from} and {t} at the same density (tolerance "
                        f"{tol:.3f}): a crossing moves it, and so does Z1+ sliding "
                        f"with none; the partner-pair note says which")
        report.z_gated = len(held) > 1 and not measured
    return report


@dataclass
class AtomisticRun:
    """Run the three atomistic stages of one build, gating after each.

    ``run_dir`` is the study directory (``<output_dir>/<study>``) with
    ``02_Chemistry``, ``03_Conformation``, ``04_Simulation`` and the
    manifest's strand record. ``omp`` threads per LAMMPS run (0 is serial).
    """

    run_dir: Path
    executable: str = "lmp"
    omp: int = 0
    z1: bool = True
    z1_config: Optional[dict] = None
    #: Z1+ seeds per checkpoint (see :func:`read_checkpoint`).
    z1_seeds: int = 4
    stop_on_fail: bool = True
    gate_from: str = GATE_FROM
    z_tolerance: Optional[float] = None
    mode: str = "persistent"
    sim: str = "04_Simulation"
    #: Gzip each stage's backbone dump once it has been read (about a fifth
    #: of the 200 MB a DP-30 stage writes at every 10 steps).
    compress_dumps: bool = True
    checkpoints: dict = field(default_factory=dict)
    crossings: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    report: Optional[AtomisticGateReport] = None

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.sim_dir = self.run_dir / self.sim
        self.dirs = {"conf": self.run_dir / "03_Conformation",
                     "sim": self.sim_dir}
        self.strands = self.run_dir / "manifest.json"
        self._record = None

    def record(self):
        if self._record is None:
            self._record, _ = load_strand_record(self.strands)
        return self._record

    def _read_dump(self, tag):
        """The passages in one stage's backbone dump, if it was dumped."""
        from topon.analysis.crossings import (
            backbone_bonds, dump_path, find_crossings, junction_sharing, read_lammpstrj)

        path = dump_path(self.sim_dir, tag)
        if path is None:
            return
        bonds, strand = backbone_bonds(self.record())
        self.crossings[tag] = find_crossings(
            read_lammpstrj(path), bonds, strand, junction_sharing(self.record()))
        if self.compress_dumps and path.suffix != ".gz":
            import gzip
            import shutil
            with open(path, "rb") as src, gzip.open(f"{path}.gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()

    def _read(self, tag):
        folder, name = next((f, n) for t, f, n in CHECKPOINTS if t == tag)
        path = self.dirs[folder] / name
        if not path.exists():
            raise FileNotFoundError(f"{tag}: {path} was not written")
        self.checkpoints[tag] = read_checkpoint(
            path, self.strands, tag=tag, z1=self.z1, z1_config=self.z1_config,
            z1_seeds=self.z1_seeds)

    def _check(self) -> AtomisticGateReport:
        from topon.analysis.crossings import designed_pairs

        return check(self.checkpoints, gate_from=self.gate_from,
                     z_tolerance=self.z_tolerance, mode=self.mode,
                     crossings=self.crossings,
                     designed=designed_pairs(self.record()))

    def run(self, verbose: bool = True) -> AtomisticGateReport:
        env = dict(os.environ, OMP_NUM_THREADS=str(self.omp or 1))
        self._read("stage0_build")
        for script, tags in SCRIPTS:
            for tag in tags:
                folder, name = next((f, n) for t, f, n in CHECKPOINTS if t == tag)
                (self.dirs[folder] / name).unlink(missing_ok=True)
            dump = next(d for d, sc, _t in DUMPS if sc == script)
            # the stage opens its dump fresh, but a stale one from an earlier
            # run must not be read if this one dies before it does
            (self.sim_dir / f"traj_{dump}.lammpstrj").unlink(missing_ok=True)
            (self.sim_dir / f"traj_{dump}.lammpstrj.gz").unlink(missing_ok=True)
            cmd = [self.executable]
            if self.omp:
                cmd += ["-sf", "omp", "-pk", "omp", str(self.omp)]
            cmd += ["-in", script, "-log", f"log.{script}.txt"]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(self.sim_dir), capture_output=True,
                                  text=True, env=env)
            self.seconds[script] = time.time() - t0
            (self.sim_dir / f"out.{script}.txt").write_text(
                proc.stdout + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                err = [l for l in (proc.stdout + proc.stderr).splitlines()
                       if l.startswith("ERROR") or "Last input" in l]
                raise RuntimeError(f"{script} failed (exit {proc.returncode}) after "
                                   f"{self.seconds[script]:.0f} s: "
                                   + " | ".join(err[-3:] or ["no ERROR line"]))
            self._read_dump(dump)
            if verbose and dump in self.crossings:
                c = self.crossings[dump]
                print(f"  {dump:<14} backbone passages {len(c.crossings)} "
                      f"{c.counts()} over {c.transitions} frame intervals",
                      flush=True)
            for tag in tags:
                self._read(tag)
                if verbose:
                    s = self.checkpoints[tag].summary()
                    z = self.checkpoints[tag].z_bridge
                    print(f"  {tag:<14} {self.seconds[script]:6.0f} s  "
                          f"{s['density']:.4f} g/cm3  T {s['temperature']}  "
                          f"backbone max {s['bond_max']:.3f} r0  "
                          f"over +15 %: {s['stretched']}  Z "
                          f"{'-' if z is None else format(z, '.3f')}", flush=True)
            if self.stop_on_fail:
                report = self._check()
                if not report.passed:
                    self.report = report
                    return report
        self.report = self._check()
        return self.report


def measure_run(run_dir, z1: bool = True, z1_config=None,
                z_tolerance: Optional[float] = None,
                gate_from: str = GATE_FROM, mode: str = "persistent",
                sim: str = "04_Simulation", z1_seeds: int = 4) -> AtomisticGateReport:
    """Gate an atomistic run that has already been run (any checkpoints present).

    ``sim`` names the folder the stage checkpoints are in, for a run whose
    stages were written somewhere other than ``04_Simulation``.
    """
    run = AtomisticRun(run_dir, z1=z1, z1_config=z1_config,
                       z_tolerance=z_tolerance, gate_from=gate_from, mode=mode,
                       sim=sim, compress_dumps=False, z1_seeds=z1_seeds)
    for tag, folder, name in CHECKPOINTS:
        if (run.dirs[folder] / name).exists():
            run._read(tag)
    for dump, _script, _tag in DUMPS:
        run._read_dump(dump)
    if not run.checkpoints:
        raise FileNotFoundError(f"no atomistic checkpoints under {run_dir}")
    return run._check()
