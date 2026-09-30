"""Acceptance gates for the relaxation protocol.

Two questions are asked of every checkpoint a protocol writes.

**Is any bond stretched?** A bond above 1.2 sigma is the mark of a threaded
bond: a bead pushed into the gap a stretched bond opened, which is how a
strand crosses another one later. Measured on the N20 build under the old
minimiser stage 1: 85 bonds reached 1.70 sigma and 57 stayed at 1.3-1.4
through every later stage, and Z per bridge drifted 0.19 to 0.24 as those
strands crossed. Under the push-off the same build ends with 3 of 93 128.
So the gate is zero above 1.2 sigma after stage 2 and at every later stage,
and it scans every bond -- the first 50 000 is not a sample of anything,
threaded bonds are wherever the tight turns were.

Two readings of that count, because at T = 1 a healthy FENE melt puts the
occasional bond over 1.2 sigma by thermal fluctuation alone. What separates
the two is *persistence*: a threaded bond is the same bond long at every
stage, while a thermal excursion is a different bond each time. The
reference's own runs show both -- 57 bonds sat at 1.3-1.4 sigma from stage 2
to the end under the minimiser, and the 3 that survive the push-off appeared
in the first stage and never left (REPORT.md 4.2) -- and on the one
4032-bond build where a single bond did cross 1.2 sigma here it read
0.96 / 0.99 / 1.21 / 0.94 / 1.01 across the five stages, with no long bond
shared between any two of them.

``mode="instant"`` (the default) fails on the count, which is the acceptance
criterion as written; ``mode="persistent"`` fails only on a bond long at
more than one stage. Both counts are always reported, so a failure says
which kind it is.

``persistent`` is the weaker reading and can be fooled. It cannot see a bond
that is long only at the last checkpoint, or one that straddles the limit and
crosses it once. Prefer the default and read the persistence count beside it.

Persistence is judged over *every* stage measured, stage 1 included, even
though stage 1 cannot fail a run: the runner stops at stage 2, so counting
only gated stages would report "persistence unknown" at the one moment the
answer is wanted. A bond long at stages 1 and 2 was there before the gate
began.

A long bond is also *described*, not only counted. It is either threaded -- a
bead of another strand driven through the gap -- or pinched, two strands
crossing at a junction with a bead wedged off the axis; the length is the
same for both and the neighbour is what separates them. On the DP-20
re-measurement under the push-off every stopped bond turned out to be a mutual pinch at a
junction, 0.60-0.65 sigma off the axis. Z1+ jumped in stage 1, as it does on
every build, and then held (0.221 -> 0.212), so nothing crossed after the
push-off. The pinches form during the push-off and nothing measured at
placement predicted them, so the cause is still open; the gate was right to
stop, and the report line now makes the pinch-or-thread distinction in one
look.

**Did the entanglement state hold?** Z1+ is a state property, not a
topological invariant: it counts the kinks of the shortest paths between the
*current* junction positions, so part of it slides off as junctions move and
the box changes, with no crossing anywhere. Measured with zero threaded
bonds, Z per bridge still moved 1.16 -> 1.00 -> 1.11 on N100 through
equilibration and compression. So Z is *reported* per stage and only *gated*
where the acceptance test says it has to hold: between stage 3 and stage 5 of
a run that does not compress. :func:`z_hold` decides that from the densities
rather than from an opinion.

Those two states are at the same density but *not* at the same temperature --
stage 5 is the quench, 1.0 down to 0.4 -- so this is narrower than the
general rule that two states may only be compared at matched rho and T. It is
empirical: on the N20 direct build at rho 0.31, Z per bridge reads 0.253 /
0.256 / 0.254 / 0.266 / 0.255 across the five stages (REPORT.md 4.2), so with
the box held the quench does not move it and anything that does is a
crossing.

Temperature comes from the velocities in the same file, because Z1+ and the
chain statistics both depend on it and a comparison across a temperature gap
is measuring the gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: A bond longer than this is threaded. Well under the FENE divergence at
#: 1.5 sigma, and well above the 0.97 the chains are built at, so a bond has
#: to have been opened by something to be here.
BOND_GATE = 1.2

#: How far Z1+ may move between two states that are at the same density and
#: temperature.
#:
#: This is an *absolute* number calibrated on the reference-scale build: 4 427
#: bridges at Z ~ 0.25, where counting noise on the mean is about
#: sqrt(Z / n_chains) = 0.007. On a smaller system it is tighter than the noise
#: -- 192 bridges gives 0.04 -- so a small cell will trip the Z gate on
#: statistics alone. Pass ``z_tolerance=`` scaled to the build when gating one.
Z_TOLERANCE = 0.01

#: Densities within this relative distance count as the same state, so a
#: stage-4 deformation to the box the build already has is read as the
#: no-compression case it is.
RHO_SAME = 1e-3


def _min_image(d, box):
    """The shortest vector equal to ``d`` under the periodic boundary."""
    box = np.asarray(box, float)
    return d - box * np.round(d / box)


@dataclass
class Checkpoint:
    """What one data file says, in the units the gates need."""

    path: Path
    box: np.ndarray                     # (3,) edge lengths, sigma
    n_atoms: int
    bond_lengths: np.ndarray            # minimum-image, every bond
    temperature: float | None = None    # from the Velocities section
    #: ``(lo atom, hi atom)`` per entry of ``bond_lengths``. A bond's
    #: identity is its atom pair, not its position in the Bonds section:
    #: nothing promises LAMMPS writes them in the same order twice, and
    #: persistence is a statement about the same bond.
    bond_pairs: list = field(default_factory=list)
    #: Atom id -> its wrapped position and its molecule, for describing a long
    #: bond rather than only counting it. A pinch and a thread have the same
    #: bond length and a different neighbour.
    positions: dict = field(default_factory=dict)
    molecule: dict = field(default_factory=dict)

    @property
    def density(self) -> float:
        return float(self.n_atoms / np.prod(self.box))

    @property
    def bond_max(self) -> float:
        return float(self.bond_lengths.max()) if len(self.bond_lengths) else 0.0

    def stretched(self, limit: float = BOND_GATE) -> int:
        return int((self.bond_lengths > limit).sum())

    def stretched_pairs(self, limit: float = BOND_GATE) -> set:
        """Which bonds are over ``limit``, by atom pair."""
        if len(self.bond_pairs) != len(self.bond_lengths):
            return set()
        return {p for p, v in zip(self.bond_pairs, self.bond_lengths)
                if v > limit}

    def describe(self, pair) -> str:
        """What is sitting next to this bond, and where on its own strand.

        A bond over the limit is either threaded -- a bead of another strand
        driven through the gap -- or pinched, two strands crossing at a
        junction with a bead wedged off the axis. The length is the same for
        both; the neighbour is what tells them apart. Measured on the DP-20
        re-measurement, every stopped bond was a mutual pinch at a junction
        with the wedged bead 0.60-0.65 sigma off the axis, and in five of
        nine the partner was the junction bead itself or bead 0 or 1 of its
        strand.
        """
        a, b = pair
        xa, xb = self.positions.get(a), self.positions.get(b)
        if xa is None or xb is None or not self.molecule:
            return ""
        home = {self.molecule.get(a), self.molecule.get(b)}
        mid = xa + 0.5 * _min_image(xb - xa, self.box)
        best, best_d = None, float("inf")
        for i, x in self.positions.items():
            if self.molecule.get(i) in home:
                continue
            d = float(np.linalg.norm(_min_image(x - mid, self.box)))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return ""
        m = self.molecule.get(best)
        ids = sorted(i for i, mm in self.molecule.items() if mm == m)
        where = ids.index(best) if best in ids else -1
        return (f"nearest foreign bead {where} of {len(ids)} on molecule {m}, "
                f"{best_d:.2f} sigma off the bond midpoint")

    def summary(self) -> dict:
        return {"file": self.path.name, "atoms": self.n_atoms,
                "density": round(self.density, 6),
                "bond_max": round(self.bond_max, 6),
                "bond_mean": round(float(self.bond_lengths.mean()), 6)
                             if len(self.bond_lengths) else None,
                "stretched": self.stretched(),
                "temperature": (None if self.temperature is None
                                else round(self.temperature, 4))}


def read_checkpoint(path) -> Checkpoint:
    """Read a LAMMPS data file down to what the gates need.

    Bond lengths are minimum-image, not unwrapped: a build whose image flags
    are missing or wrong would otherwise report bonds a whole box long, and
    the gate would be measuring the flags instead of the bonds.

    Temperature is the kinetic one from the ``Velocities`` section, with the
    3 degrees of freedom LAMMPS subtracts for the centre of mass. It is None
    when the file has no velocities (a build, before any dynamics).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    lo = np.zeros(3)
    hi = np.zeros(3)
    n_atoms = 0
    sections: dict[str, list[str]] = {}
    current = None
    known = ("Masses", "Atoms", "Bonds", "Velocities", "Angles", "Dihedrals",
             "Impropers", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
             "Dihedral Coeffs", "Improper Coeffs")

    # Column of x in an Atoms line, by the atom style named in the section's
    # own comment. topon writes "full" everywhere; the others are here so a
    # data file from somewhere else is read rather than silently mis-parsed.
    xcol = {"full": 4, "charge": 3, "molecular": 3, "bond": 3, "angle": 3,
            "atomic": 2}
    atom_x = 4
    # Styles whose second column is the molecule id. `charge` has x in the
    # same column as `molecular` but no molecule, so the column is no guide.
    with_mol = {"full", "molecular", "bond", "angle"}
    has_mol = True

    for raw in lines:
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line == "Atoms":
            comment = raw.split("#", 1)[1].split() if "#" in raw else []
            style = comment[0] if comment else "full"
            atom_x = xcol.get(style, 4)
            has_mol = style in with_mol or style not in xcol
        if line in known:
            # A section header is the section name alone on the line; any
            # "# full" style qualifier was stripped with the comment above.
            current = line
            sections[current] = []
            continue
        if current is None:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "atoms":
                n_atoms = int(parts[0])
            elif len(parts) >= 4 and parts[2] in ("xlo", "ylo", "zlo"):
                i = "xyz".index(parts[2][0])
                lo[i], hi[i] = float(parts[0]), float(parts[1])
        else:
            sections[current].append(line)

    box = hi - lo
    if not np.all(box > 0):
        raise ValueError(f"{path.name}: no box dimensions found")

    pos: dict[int, np.ndarray] = {}
    mol: dict[int, int] = {}
    for line in sections.get("Atoms", []):
        p = line.split()
        # atom_style full: id mol type q x y z [ix iy iz]
        pos[int(p[0])] = np.array([float(p[atom_x]), float(p[atom_x + 1]),
                                   float(p[atom_x + 2])])
        if has_mol:
            mol[int(p[0])] = int(p[1])

    bl, pairs = [], []
    for line in sections.get("Bonds", []):
        p = line.split()
        i, j = int(p[2]), int(p[3])
        a, b = pos.get(i), pos.get(j)
        if a is None or b is None:
            continue
        d = a - b
        d -= box * np.round(d / box)
        bl.append(float(np.linalg.norm(d)))
        pairs.append((min(i, j), max(i, j)))

    temperature = None
    vel = sections.get("Velocities", [])
    if vel:
        v = np.array([[float(x) for x in line.split()[1:4]] for line in vel])
        # Unit masses: every bead of a Kremer-Grest melt has mass 1.
        dof = max(3 * len(v) - 3, 1)
        temperature = float((v ** 2).sum() / dof)

    return Checkpoint(path=path, box=box, n_atoms=n_atoms or len(pos),
                      bond_lengths=np.array(bl), temperature=temperature,
                      bond_pairs=pairs, positions=pos, molecule=mol)


@dataclass
class GateReport:
    """The verdict, and everything it was read from."""

    stages: dict = field(default_factory=dict)      # tag -> checkpoint summary
    failures: list = field(default_factory=list)    # human-readable, in order
    notes: list = field(default_factory=list)       # seen, but not a failure
    #: One line per long bond saying what sits next to it -- a pinch and a
    #: thread have the same length and a different neighbour.
    details: list = field(default_factory=list)
    z_by_stage: dict = field(default_factory=dict)
    z_gated: bool = False
    mode: str = "instant"
    #: Over-limit bonds seen at more than one gated stage, ``{pair: [tags]}``.
    #: These are the threaded ones; a bond in the list once is thermal.
    persistent: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> dict:
        return {"passed": self.passed, "stages": self.stages,
                "failures": self.failures, "notes": self.notes,
                "details": self.details,
                "z_by_stage": self.z_by_stage, "z_gated": self.z_gated,
                "mode": self.mode,
                "persistent_bonds": len(self.persistent),
                "persistent": {f"{a}-{b}": tags
                               for (a, b), tags in self.persistent.items()}}

    def render(self) -> str:
        head = (f"{'stage':<22}{'rho':>9}{'T':>8}{'bond max':>11}"
                f"{'>1.2':>7}{'Z':>9}")
        rows = [head, "-" * len(head)]
        for tag, s in self.stages.items():
            z = self.z_by_stage.get(tag)
            rows.append(
                f"{tag:<22}{s['density']:>9.4f}"
                f"{(float('nan') if s['temperature'] is None else s['temperature']):>8.3f}"
                f"{s['bond_max']:>11.3f}{s['stretched']:>7d}"
                f"{(float('nan') if z is None else z):>9.3f}")
        rows.append("")
        rows.append(f"threaded bonds (long at more than one stage): "
                    f"{len(self.persistent)}")
        for line in self.notes:
            rows.append(f"  note: {line}")
        for line in self.details:
            rows.append(f"  {line}")
        rows.append("GATES PASSED" if self.passed else
                    "GATES FAILED:\n  " + "\n  ".join(self.failures))
        return "\n".join(rows)


def z_hold(stages: dict, tol: float = RHO_SAME) -> bool:
    """Should the Z gate apply to this run?

    Only when stage 3 and stage 5 are at the same density, i.e. a stage 4 that
    compressed nothing. Under compression Z1+ moves 10-20 % without a single
    crossing (REPORT.md 4.3), so gating on it there would fail correct runs
    and tell nobody anything.

    Density only, not temperature: stage 5 is the quench and is always colder
    than stage 3. The acceptance test holds Z across it anyway, because with
    the box held the quench is measured not to move it.
    """
    a = stages.get("stage3_build")
    b = stages.get("stage5_quench")
    if not a or not b:
        return False
    ra, rb = a["density"], b["density"]
    return abs(ra - rb) <= tol * max(ra, rb)


def check(stages, z_by_stage=None, bond_limit=BOND_GATE,
          z_tolerance=Z_TOLERANCE, gate_z=None, mode="instant",
          gate_from="stage2_pushoff", describe_limit=8) -> GateReport:
    """Run both gates over the per-stage checkpoints.

    Args:
        stages: ordered ``{tag: Checkpoint}`` -- the run's checkpoints, in
            run order.
        z_by_stage: ``{tag: Z per bridge}`` where Z1+ was measured. Reported
            either way; gated only where :func:`z_hold` says it must hold.
        bond_limit: sigma above which a bond counts as threaded.
        z_tolerance: how far Z may move between stage 3 and stage 5 of a
            run that did not compress. The default is calibrated at the
            reference's scale; see :data:`Z_TOLERANCE`.
        gate_z: force the Z gate on or off. ``None`` decides from the
            densities, which is the specification's rule.
        mode: ``"instant"`` fails on any bond over the limit at any gated
            stage -- the acceptance criterion as written. ``"persistent"``
            fails only on a bond over the limit at more than one stage,
            which is what a threaded bond looks like and what a thermal
            excursion does not. Both counts are reported either way.
        gate_from: the first stage the bond gate applies to. Stage 1 is
            excluded from *failing* a run -- it is still resolving the
            build's overlaps, and a bond briefly above 1.2 sigma there is the
            push-off doing its job -- but it still counts as evidence for
            whether a bond is persistent.
        describe_limit: how many long bonds per stage to describe in the
            report. Describing one is a scan over every bead in the
            checkpoint, so this is capped rather than unbounded.

    Returns:
        A :class:`GateReport`, which is falsy on ``.passed`` if anything failed.
    """
    if mode not in ("instant", "persistent"):
        raise ValueError(f"unknown bond-gate mode {mode!r} "
                         f"(expected 'instant' or 'persistent')")
    summaries = {tag: cp.summary() for tag, cp in stages.items()}
    report = GateReport(stages=summaries, z_by_stage=dict(z_by_stage or {}),
                        mode=mode)

    tags = list(stages)
    # Nothing is gated until `gate_from` has actually been reached. The
    # fallback has to be "gate nothing", not "gate everything from the start":
    # the runner checks after *every* stage, so after stage 1 the only tag
    # present is stage1_min, and treating a missing gate_from as position 0
    # gates the one stage that must not be gated. Measured on the N20 build:
    # 6 bonds sit above the limit after stage 1 of the dilute build, which the
    # push-off is still resolving (the reference reports 3 for its own
    # placement, REPORT.md 4.2). Gating there throws away a correct build.
    gated = tags[tags.index(gate_from):] if gate_from in tags else []

    # Which stages each over-limit bond appears at, by atom pair. The
    # evidence spans *every* stage measured, stage 1 included: a bond long at
    # stages 1 and 2 was there before the gate began and is persistent by the
    # time the gate first looks, where counting only gated stages would
    # report it as "persistence unknown" at exactly the moment the runner
    # stops. Stage 1 still cannot fail a run -- only `gated` does that.
    seen_at: dict = {}
    for tag in tags:
        for pair in stages[tag].stretched_pairs(bond_limit):
            seen_at.setdefault(pair, []).append(tag)
    report.persistent = {p: t for p, t in seen_at.items() if len(t) > 1}

    for tag in gated:
        n = stages[tag].stretched(bond_limit)
        if not n:
            continue
        here = [p for p in stages[tag].stretched_pairs(bond_limit)
                if p in report.persistent]
        if here:
            kind = (f"{len(here)} of them long at another stage too, "
                    f"i.e. threaded")
        elif len(gated) > 1:
            kind = ("none of them long at any other stage, i.e. thermal, "
                    "not threaded")
        else:
            # One gated stage is not enough to tell the two apart, and saying
            # "thermal" here would be a claim the data cannot support.
            kind = "only one gated stage so far, so persistence is unknown"
        line = (f"{tag}: {n} bond(s) above {bond_limit} sigma "
                f"(longest {stages[tag].bond_max:.3f}) -- {kind}")
        if mode == "instant" or here:
            report.failures.append(line)
        else:
            report.notes.append(line)
        # What is sitting next to each of them. A pinch and a thread have the
        # same bond length and a different neighbour, and this is the line
        # that told the two apart in one look on the DP-20 re-measurement.
        cp = stages[tag]
        for pair in sorted(cp.stretched_pairs(bond_limit))[:describe_limit]:
            what = cp.describe(pair)
            if what:
                length = dict(zip(cp.bond_pairs, cp.bond_lengths)).get(pair)
                report.details.append(
                    f"{tag}: bond {pair[0]}-{pair[1]} at "
                    f"{length:.3f} sigma, {what}")

    a = report.z_by_stage.get("stage3_build")
    b = report.z_by_stage.get("stage5_quench")
    if gate_z is None:
        # Auto: gate only where Z has to hold *and* someone measured it. Z1+
        # is not in this repository, so a run with no Z1+ pass is the normal
        # case, not a failure -- demanding a number the caller never had
        # would turn the bond gate's verdict into noise.
        report.z_gated = z_hold(summaries) and a is not None and b is not None
    else:
        report.z_gated = bool(gate_z)
    if report.z_gated:
        if a is None or b is None:
            report.failures.append(
                "Z gate asked for but Z1+ is missing for stage3_build or "
                "stage5_quench")
        elif abs(a - b) > z_tolerance:
            report.failures.append(
                f"Z moved {a:.3f} -> {b:.3f} between stage 3 and stage 5 "
                f"(tolerance {z_tolerance}) with no compression between "
                f"them, so strands crossed")
    return report
