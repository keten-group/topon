"""A tracker page for atomistic relaxation runs (``topon track``).

One HTML file, self-contained, for one or more runs of the atomistic route.
For each run and each of its checkpoints (the build, stage 1, the ramp, and
the minimised, NVT and NPT states of stage 3) it holds:

* the network as the Z1+ export sees it: every strand from junction through
  one point per repeat unit to junction, drawn as a curve, the crosslinks,
  and Z1+'s primitive paths with their kinks (seed 0);
* Z per bridge over several seeds of the export's junction jitter, its
  spread, and how many of the build's robust partner pairs are still seen;
* the gate numbers from the run itself: backbone passages per stage dump,
  temperature, density, the longest backbone bond;
* with ``energies``, the potential energy of every checkpoint under the full
  force field (a zero-step LAMMPS run with the run's own stage-3 styles and
  settings), so the staged potentials of stages 1 and 2 are not what is
  read, as energy density with its bonded, van der Waals and Coulomb parts;
* the LAMMPS time of each script, from its log.

The page draws everything in the browser (no library): a rotatable view of
the network per checkpoint and a column of charts through the stages, and
one chart of Z over its build value for every run in the file.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: The checkpoints the page steps through, in run order: (tag, folder, file).
STAGES = (
    ("stage0_build", "03_Conformation", "system_relaxed.data"),
    ("stage1_soft", "04_Simulation", "system_after_soft.data"),
    ("stage2_ramp", "04_Simulation", "system_ramped.data"),
    ("stage3_min", "04_Simulation", "system_minimized_final.data"),
    ("stage3_nvt", "04_Simulation", "after_nvt_real.data"),
    ("stage3_npt", "04_Simulation", "system_equilibrated.data"),
)
#: Which stage dump holds each checkpoint's passages.
DUMP_OF = {"stage1_soft": "stage1", "stage2_ramp": "stage2", "stage3_min": "stage3",
           "stage3_nvt": "stage3", "stage3_npt": "stage3"}
#: kcal/mol per cubic angstrom in MJ/m^3.
KCAL_PER_A3 = 4184.0 / 6.02214076e23 / 1e-30 / 1e6
TEMPLATE = Path(__file__).with_name("tracker_template.html")

_BEFORE = re.compile(r"^(units|atom_style|boundary|bond_style|angle_style|dihedral_style|"
                     r"improper_style|special_bonds|pair_style)\b")
_AFTER = re.compile(r"^(neigh_modify|pair_style|pair_modify|kspace_style|include|special_bonds)\b")


# ---------------------------------------------------------------------------
# Energies and times from LAMMPS
# ---------------------------------------------------------------------------

def energy_deck(stage3: str, data: str) -> str:
    """A zero-step run of ``data`` under the full force field of ``stage3``.

    ``stage3`` is the run's ``minimize_3_parallel.in``: its styles before
    ``read_data`` and its pair style, settings and kspace after it are kept,
    everything from the minimiser on is not.
    """
    before, after, seen = [], [], False
    for line in stage3.splitlines():
        s = line.strip()
        if s.startswith("read_data"):
            seen = True
            continue
        if not seen:
            if _BEFORE.match(s):
                before.append(s)
        elif s.startswith("min_style") or s.startswith("minimize"):
            break
        elif _AFTER.match(s):
            after.append(s)
    return "\n".join(before + [f"read_data       {data} nocoeff"] + after + [
        "thermo_style    custom step pe ebond eangle edihed eimp evdwl ecoul elong vol",
        "thermo_modify   norm no",
        "run             0",
    ]) + "\n"


def checkpoint_energies(run_dir, lmp: str = "lmp", omp: int = 1,
                        sim: str = "04_Simulation") -> dict:
    """``{tag: {pe, ebond, ..., vol}}`` in kcal/mol and A^3, per checkpoint.

    Runs LAMMPS once per checkpoint, zero steps. A checkpoint that fails to
    evaluate is left out.
    """
    sim_dir = Path(run_dir) / sim
    stage3 = sim_dir / "minimize_3_parallel.in"
    if not stage3.exists():
        return {}
    text = stage3.read_text()
    names = ["step", "pe", "ebond", "eangle", "edihed", "eimp", "evdwl", "ecoul", "elong", "vol"]
    out = {}
    for tag, folder, name in STAGES:
        data = name if folder == "04_Simulation" else f"../{folder}/{name}"
        if not (sim_dir / data).exists():
            continue
        (sim_dir / "energy_eval.in").write_text(energy_deck(text, data))
        cmd = [lmp] + (["-sf", "omp", "-pk", "omp", str(omp)] if omp and omp > 1 else [])
        cmd += ["-in", "energy_eval.in", "-log", "log.energy_eval.txt"]
        try:
            p = subprocess.run(cmd, cwd=sim_dir, capture_output=True, text=True,
                               env=dict(os.environ, OMP_NUM_THREADS=str(max(1, omp or 1))))
        except OSError:
            return out
        lines = (p.stdout or "").splitlines()
        i = next((k for k, line in enumerate(lines) if line.split()[:2] == ["Step", "PotEng"]), None)
        if p.returncode or i is None:
            continue
        out[tag] = dict(zip(names, (float(x) for x in lines[i + 1].split())))
    return out


def lammps_seconds(run_dir, sim: str = "04_Simulation") -> dict:
    """Seconds of LAMMPS per stage script, summed from ``Loop time`` in its log."""
    out = {}
    for log in sorted((Path(run_dir) / sim).glob("log.minimize_*.in.txt")):
        loops = re.findall(r"^Loop time of ([0-9.eE+-]+)", log.read_text(errors="replace"), re.M)
        if loops:
            out[log.name[len("log."):-len(".txt")]] = round(sum(float(x) for x in loops))
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def align_paths(chains: Sequence[np.ndarray], paths: Sequence[np.ndarray], box) -> list:
    """Each primitive path moved into its chain's frame.

    Z1+ writes a chain's primitive path shifted by a whole box vector so that
    it starts inside the cell; the shift that puts the path's first point on
    the chain's first point puts every point of it, and its kinks, back.
    """
    box = np.asarray(box, float)
    out = []
    for c, p in zip(chains, paths):
        p = np.array(p, float)
        d = np.asarray(c[0], float) - p[0, :3]
        p[:, :3] += box * np.round(d / box)
        out.append(p)
    return out


def _q(x) -> list:
    """Coordinates in 0.1 A integers, flat."""
    return np.round(np.asarray(x, float) * 10).astype(int).ravel().tolist()


def track_run(run_dir, label: Optional[str] = None, seeds: int = 8, z1_config=None,
              energies: bool = True, lmp: str = "lmp", omp: int = 1,
              sim: str = "04_Simulation") -> dict:
    """Everything the tracker page shows for one run, as plain data."""
    from topon.analysis.atomistic import read_atomistic
    from topon.analysis.crossings import designed_pairs
    from topon.analysis.z1plus import (
        Z1PlusFailed, Z1PlusUnavailable, export_system, measure_seeds, z1plus_available)
    from topon.simulation.protocols.atomistic import measure_run

    run_dir = Path(run_dir)
    gates = measure_run(run_dir, z1=False, sim=sim)
    record = json.loads((run_dir / "manifest.json").read_text())["stages"]["strands"]
    energy = checkpoint_energies(run_dir, lmp=lmp, omp=omp, sim=sim) if energies else {}
    use_z1 = seeds > 0 and z1plus_available(z1_config)
    stages, base, classes, atoms = [], None, None, None
    for tag, folder, name in STAGES:
        path = run_dir / (sim if folder == "04_Simulation" else folder) / name
        if not path.exists():
            stages.append({"tag": tag, "missing": True})
            continue
        system = read_atomistic(path, run_dir / "manifest.json")
        chains, cls, _mols = export_system(system, seed=0)
        classes = [str(c) for c in cls] if classes is None else classes
        atoms = system.n_atoms if atoms is None else atoms
        entry = {"tag": tag, "box": [round(float(b), 2) for b in system.box],
                 "chains": [_q(c) for c in chains], "pp": [], "kinks": [],
                 "zchain": [0] * len(chains), "z_mean": None, "z_sd": None,
                 "build_pairs_kept": None, "build_pairs": None}
        if use_z1:
            try:
                m = measure_seeds(system, seeds=seeds, config=z1_config)
                _z, res = m.first
                paths = align_paths(chains, res.paths or [], system.box)
                entry.update(
                    z_mean=None if m.z_bridge is None else round(m.z_bridge, 4),
                    z_sd=None if m.z_bridge_sd is None else round(m.z_bridge_sd, 4),
                    zchain=res.Z.tolist(),
                    pp=[_q(p[:, :3]) for p in paths],
                    kinks=[[c, int(row[4])] + _q(row[:3])
                           for c, p in enumerate(paths, start=1) for row in p if row[3] == 1])
                base = m.robust if base is None else base
                entry.update(build_pairs=len(base), build_pairs_kept=len(base & m.seen))
            except (Z1PlusUnavailable, Z1PlusFailed):
                pass
        st = gates.stages.get(tag, {})
        entry.update(temperature=st.get("temperature"), density=st.get("density"),
                     bond_max=st.get("bond_max"), stretched=st.get("stretched"))
        dump = DUMP_OF.get(tag)
        entry["passages"] = (gates.crossings.get(dump, {}).get("passages") if dump else None)
        e = energy.get(tag)
        entry["energy"] = None if not e else {
            "pe_per_atom": round(e["pe"] / system.n_atoms, 4),
            "density_MJ_m3": round(e["pe"] / e["vol"] * KCAL_PER_A3, 2),
            "bonded_MJ_m3": round((e["ebond"] + e["eangle"] + e["edihed"] + e["eimp"])
                                  / e["vol"] * KCAL_PER_A3, 2),
            "vdw_MJ_m3": round(e["evdwl"] / e["vol"] * KCAL_PER_A3, 2),
            "coul_MJ_m3": round((e["ecoul"] + e["elong"]) / e["vol"] * KCAL_PER_A3, 2)}
        stages.append(entry)
    if classes is None:
        raise FileNotFoundError(f"no atomistic checkpoint under {run_dir}")
    return {"key": run_dir.name, "label": label or run_dir.name, "atoms": int(atoms),
            "classes": classes,
            "designed": [[k, l] for (k, l) in sorted(designed_pairs(record))],
            "passed": bool(gates.passed), "crossings": gates.crossings,
            "seconds": lammps_seconds(run_dir, sim), "seeds": int(seeds) if use_z1 else 0,
            "stages": stages}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def render_tracker(runs: Sequence[dict], title: str = "topon Relaxation Tracker",
                   fragment: bool = False) -> str:
    """The page for ``runs`` (from :func:`track_run`).

    ``fragment`` leaves out the document skeleton, for a host that adds its own.
    """
    template = TEMPLATE.read_text(encoding="utf-8")
    head, body = template.split("<!-- body -->", 1)
    data = json.dumps(list(runs), separators=(",", ":")).replace("</", "<\\/")
    seeds = max((r.get("seeds") or 0) for r in runs) if runs else 0
    head = head.replace("__TITLE__", title)
    body = (body.replace("__TITLE__", title).replace("__SEEDS__", str(seeds or "no"))
            .replace("__DATA__", data))
    if fragment:
        return head + body
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, "
            "viewport-fit=cover\">\n" + head + "</head>\n<body>\n" + body + "</body>\n</html>\n")


def write_tracker(runs: Sequence[dict], path, title: str = "topon Relaxation Tracker",
                  fragment: bool = False) -> Path:
    path = Path(path)
    path.write_text(render_tracker(runs, title=title, fragment=fragment), encoding="utf-8")
    return path
