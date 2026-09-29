"""Assemble data/raw/: the raw simulation output behind the derived data in data/derived/.

It ran once on the authors' machine. The notebooks do not need it. It reads the full simulation folders, which are
not deposited, from NPJ_RAW_ROOT, the folder that holds crosslinker/ and glass_transition_atomistic/. MANIFEST_raw.csv
records each source path relative to that folder as Studies/crosslinker/... or Studies/glass_transition_atomistic/...

data/raw/cg_stress_strain_seed{1,2,3,4}.tar.xz  stress-strain files of all 3,924 coarse-grained pulls
data/raw/cg_lammps_inputs.tar.xz                LAMMPS inputs of the equilibration and the tensile tests, per network
data/raw/cg_generation_settings.csv             generator call of every network (target counts, lattice, trials)
data/raw/cg_velocity_seeds.csv                  velocity seed of each tensile replicate
data/raw/tg_cooling.tar.xz                      MSD files and LAMMPS inputs of the three cooling histories
data/raw/atomistic_systems.tar.xz               as-built LAMMPS files of the six atomistic systems (02_Chemistry/)
data/raw/atomistic_equilibrated.tar.xz          equilibrated start of their cooling runs (05_ExtendedSampling/)
data/raw/MANIFEST_raw.csv                       SHA-256 of every archived file

The stress-strain files are checked against data/derived/mechanics/mechanics_pulls_4seeds.csv (peak stress of every
pull) before anything is written.

    NPJ_RAW_ROOT=/path/to/Studies python scripts/build_raw_data.py
    NPJ_RAW_ROOT=/path/to/Studies python scripts/build_raw_data.py atomistic

The second form needs only glass_transition_atomistic/. It rewrites the two atomistic archives and their rows of
MANIFEST_raw.csv and keeps the other rows.
"""
import csv
import hashlib
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RAW = PKG / "data" / "raw"
# folder that holds the full simulation trees crosslinker/ and glass_transition_atomistic/ (not deposited)
RAW_ROOT = Path(os.environ.get("NPJ_RAW_ROOT", "NPJ_RAW_ROOT-not-set"))
CG = RAW_ROOT / "crosslinker"
TG = RAW_ROOT / "glass_transition_atomistic"
SEED_DIRS = {1: None, 2: CG / "sc_6x6x6_replicate2", 3: CG / "sc_6x6x6_replicate3", 4: CG / "sc_6x6x6_replicate4"}
VELOCITY = {1: 4928459, 2: 7391052, 3: 2648317, 4: 9184735}
AXES = "xyz"
MANIFEST = []
# atomistic systems (folder name in the simulation tree, name in the deposit), FPDMS is PMTFPS
TG_SYSTEMS = (("PDMS", "PDMS"), ("FPDMS", "PMTFPS"))
TG_DPS = ("10", "30", "100")
# as-built files of 02_Chemistry/ (the *.displace files are left out, since no deposited input reads them)
CHEM_FILES = ("system.data", "system.in.settings", "system.groups", "system_node_info.txt", "system_edge_info.txt")
ATOMISTIC = ("atomistic_systems.tar.xz", "atomistic_equilibrated.tar.xz")


def ss_path(seed, net, ax):
    if seed == 1:
        return CG / "sc_6x6x6" / net / "tensile_test_attractive_xyz" / f"stress_strain_npt_{ax}.dat"
    return SEED_DIRS[seed] / net / f"stress_strain_npt_{ax}.dat"


def add(tar, src, arcname, archive):
    data = Path(src).read_bytes()
    ti = tarfile.TarInfo(arcname)
    ti.size = len(data)
    ti.mtime = int(Path(src).stat().st_mtime)
    ti.mode = 0o644
    tar.addfile(ti, io.BytesIO(data))
    rel = Path(src).as_posix().replace(CG.as_posix(), "Studies/crosslinker").replace(TG.as_posix(), "Studies/glass_transition_atomistic")
    MANIFEST.append((archive, arcname, len(data), hashlib.sha256(data).hexdigest(), rel))


def peak(path):
    a = np.loadtxt(path, comments="#")
    return a[:, 2].max(), len(a)


def data_counts(path):
    """Atom, bond, angle and dihedral counts from the header of a LAMMPS data file."""
    counts = {}
    with open(path, encoding="ascii") as fh:
        for line in fh:
            t = line.split()
            if len(t) == 2 and t[1] in ("atoms", "bonds", "angles", "dihedrals"):
                counts[t[1]] = int(t[0])
            if t and t[0] in ("Masses", "Atoms"):
                break
    return counts


def pack_atomistic():
    """The six atomistic systems, in the folders the cooling inputs of tg_cooling.tar.xz read from
    <system>/DP<n>/<history>/ (../02_Chemistry/ and ../05_ExtendedSampling/). The files hold no paths or user names and
    are archived byte for byte."""
    for sysname, _ in TG_SYSTEMS:
        for dp in TG_DPS:
            d = TG / sysname / f"DP_{dp}"
            built = data_counts(d / "02_Chemistry" / "system.data")
            equil = data_counts(d / "05_ExtendedSampling" / "ready2deform.data")
            assert built == equil and len(built) == 4, (d, built, equil)
            first = (d / "06_TgSimulation" / "simulation_305K.lmp").read_text()
            for ref in ("../05_ExtendedSampling/ready2deform.data", "../02_Chemistry/system.in.settings",
                        "../02_Chemistry/system.groups"):
                assert ref in first, (d, ref)
    for name, sub, files in ((ATOMISTIC[0], "02_Chemistry", CHEM_FILES),
                             (ATOMISTIC[1], "05_ExtendedSampling", ("ready2deform.data",))):
        with tarfile.open(RAW / name, "w:xz", preset=9) as tar:
            for sysname, label in TG_SYSTEMS:
                for dp in TG_DPS:
                    for fn in files:
                        add(tar, TG / sysname / f"DP_{dp}" / sub / fn, f"{label}/DP{dp}/{sub}/{fn}", name)


def write_manifest(keep_other_rows=False):
    """Write MANIFEST_raw.csv. With keep_other_rows, the rows of archives not rebuilt in this run are kept."""
    rows = []
    if keep_other_rows:
        rebuilt = {m[0] for m in MANIFEST}
        with open(RAW / "MANIFEST_raw.csv", newline="", encoding="utf-8") as fh:
            rows = [r for r in list(csv.reader(fh))[1:] if r[0] not in rebuilt]
    with open(RAW / "MANIFEST_raw.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["archive", "path_in_archive", "bytes", "sha256", "source"])
        w.writerows(rows + MANIFEST)


def main():
    if sys.argv[1:] == ["atomistic"]:
        if not TG.is_dir():
            raise SystemExit(f"Set NPJ_RAW_ROOT to the folder that holds glass_transition_atomistic/. Now: {RAW_ROOT}")
        pack_atomistic()
        write_manifest(keep_other_rows=True)
        for name in ATOMISTIC:
            print(f"{name:40s} {(RAW / name).stat().st_size / 1e6:7.2f} MB")
        print("files archived:", len(MANIFEST))
        return
    if not (CG.is_dir() and TG.is_dir()):
        raise SystemExit(f"Set NPJ_RAW_ROOT to the folder that holds crosslinker/ and glass_transition_atomistic/ "
                         f"(the full simulation folders, which are not deposited). Now: {RAW_ROOT}")
    RAW.mkdir(parents=True, exist_ok=True)
    pulls = pd.read_csv(PKG / "data" / "derived" / "mechanics" / "mechanics_pulls_4seeds.csv", float_precision="round_trip")
    nets = sorted(pulls.folder_name.unique())
    assert len(nets) == 327 and len(pulls) == 3924

    # 1. stress-strain, checked against the per-pull table first
    worst, rows_hist = 0.0, {}
    for r in pulls.itertuples():
        p = ss_path(r.seed, r.folder_name, r.axis)
        u, n = peak(p)
        worst = max(worst, abs(u - r.uts))
        rows_hist[n] = rows_hist.get(n, 0) + 1
    print(f"peak stress vs mechanics_pulls_4seeds.csv: max |diff| = {worst:.3g}; rows per file {rows_hist}")
    assert worst < 1e-9, "raw files do not reproduce the per-pull table"
    for seed in (1, 2, 3, 4):
        name = f"cg_stress_strain_seed{seed}.tar.xz"
        with tarfile.open(RAW / name, "w:xz", preset=9) as tar:
            for net in nets:
                for ax in AXES:
                    add(tar, ss_path(seed, net, ax), f"seed{seed}/{net}/stress_strain_npt_{ax}.dat", name)

    # 2. LAMMPS inputs (text only; cluster job scripts, restarts, logs and trajectories are left out)
    name = "cg_lammps_inputs.tar.xz"
    with tarfile.open(RAW / name, "w:xz", preset=9) as tar:
        for net in nets:
            base = CG / "sc_6x6x6" / net
            for sub, pats in (("minimize_equilibrate", ("*.in", "*.lmp", "polymer.coeff")),
                              ("tensile_test_attractive_xyz", ("tensile_*_parallel_p0_attractive.lmp", "polymer.coeff"))):
                files = sorted({f for pat in pats for f in (base / sub).glob(pat)})
                assert files, (net, sub)
                for f in files:
                    add(tar, f, f"{net}/{sub}/{f.name}", name)

    # 3. generation settings, read from the generator call in each network's run script
    gen = []
    for net in nets:
        txt = (CG / "sc_6x6x6" / net / "run.sh").read_text(encoding="utf-8", errors="replace")
        m = re.search(r'generator\.exe"?\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"+([0-9:,]+)[\s"]+(\d)\s+(\w+)', txt)
        assert m, net
        dims, per, maxf, trials, saves, spec, _, lat = m.groups()
        counts = dict(kv.split(":") for kv in spec.split(","))
        assert "_".join(counts[str(d)] for d in range(7)) == net, (net, spec)
        g = sorted((PKG / "data" / "mechanics" / net).glob("*.edges"))
        trial = int(re.search(r"trial(\d+)", g[0].name).group(1)) if g else None
        gen.append(dict(folder_name=net, **{f"n_f{d}": int(counts[str(d)]) for d in range(7)}, lattice=lat, dims=dims,
                        periodicity=per, max_functionality=int(maxf), max_trials=int(trials), max_saves=int(saves),
                        search="strict (C generator)", trial_ordinal=trial,
                        graph_files=f"data/mechanics/{net}/" if g else ""))
    pd.DataFrame(gen).to_csv(RAW / "cg_generation_settings.csv", index=False)

    # 4. velocity seeds of the tensile replicates; replicates 2-4 used the seed-1 input with only this number changed
    with open(RAW / "cg_velocity_seeds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "velocity_seed", "stress_strain_archive"])
        for s, v in VELOCITY.items():
            w.writerow([s, v, f"cg_stress_strain_seed{s}.tar.xz"])
    tens = (CG / "sc_6x6x6" / nets[0] / "tensile_test_attractive_xyz" / "tensile_x_parallel_p0_attractive.lmp").read_text()
    assert tens.count(str(VELOCITY[1])) == 1

    # 5. Tg cooling sweeps: MSD files of the three histories and the per-temperature inputs
    name = "tg_cooling.tar.xz"
    with tarfile.open(RAW / name, "w:xz", preset=9) as tar:
        add(tar, TG / "simulation_template.lmp", "simulation_template.lmp", name)
        for sysname, label in (("PDMS", "PDMS"), ("FPDMS", "PMTFPS")):
            for dp in ("10", "30", "100"):
                for hist, folder in (("original", "06_TgSimulation"), ("replicate2", "06_TgSimulation_replicate2"),
                                     ("replicate3", "06_TgSimulation_replicate3")):
                    d = TG / sysname / f"DP_{dp}" / folder
                    msd = d / "msd.dat" if hist == "original" else TG / f"{hist}_data" / f"{sysname}_DP{dp}_msd.dat"
                    add(tar, msd, f"{label}/DP{dp}/{hist}/msd.dat", name)
                    lmps = sorted(d.glob("simulation_*K.lmp"))
                    assert len(lmps) >= 40, (d, len(lmps))
                    for f in lmps:
                        add(tar, f, f"{label}/DP{dp}/{hist}/{f.name}", name)

    # 6. atomistic systems, as built and equilibrated (the files the first cooling step reads)
    pack_atomistic()

    write_manifest()
    for f in sorted(RAW.iterdir()):
        print(f"{f.name:40s} {f.stat().st_size / 1e6:7.2f} MB")
    print("files archived:", len(MANIFEST))


if __name__ == "__main__":
    main()
