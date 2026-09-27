"""Parse LAMMPS thermo output of the anneal + five production blocks for one network.

log.lammps in minimize_equilibrate/ only holds the LAST block (006_Ext_05), so the
full history is read from the slurm-*.out files. Each run chain ends at a fixed
timestep: anneal 4.1e6, Ext_k 4.1e6 + k*2e6. Some networks were run twice; for
each end step the file with the highest job id is used and the chain is verified
(i) Ext_05 rows == log.lammps rows, (ii) first row of block k+1 == last row of block k
(restart continuity, compared on PE and box).

Inputs (not deposited): slurm-*.out and log.lammps in RAW_ROOT/crosslinker/sc_6x6x6/<net>/minimize_equilibrate/
(see common.py).
"""
import os
import re
import glob
import numpy as np
from common import me_path

COLS = ["step", "temp", "pe", "ke", "etotal", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"]
END_STEPS = {"anneal": 4_100_000}
for k in range(1, 6):
    END_STEPS[f"ext{k}"] = 4_100_000 + 2_000_000 * k


def parse_thermo(path):
    """Return list of runs; each run is an (n, len(COLS)) array. Only runs whose
    header matches COLS are kept."""
    runs = []
    cur = None
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            s = ln.split()
            if not s:
                continue
            if s[0] == "Step":
                cur = [] if [c.lower() for c in s] == [c for c in COLS] or \
                    [c.lower() for c in s] == ["step", "temp", "poteng", "kineng", "toteng", "press", "pxx", "pyy", "pzz", "lx", "ly", "lz", "density"] else None
                if cur is not None:
                    runs.append(cur)
                continue
            if cur is not None:
                if s[0] == "Loop" or s[0].startswith("WARNING") or s[0] == "ERROR":
                    cur = None
                    continue
                if len(s) == len(COLS):
                    try:
                        cur.append([float(v) for v in s])
                    except ValueError:
                        cur = None
    return [np.array(r) for r in runs if len(r)]


def load_chain(net):
    d = os.path.dirname(me_path(net, "x"))
    files = glob.glob(os.path.join(d, "slurm-*.out"))
    cand = {}
    for f in files:
        jid = int(re.findall(r"slurm-(\d+)\.out", f)[0])
        runs = parse_thermo(f)
        if not runs:
            continue
        last = runs[-1][-1, 0]
        for key, es in END_STEPS.items():
            if int(last) == es:
                if key not in cand or jid > cand[key][0]:
                    cand[key] = (jid, runs)
    missing = [k for k in END_STEPS if k not in cand]
    if missing:
        raise RuntimeError(f"{net}: missing blocks {missing}")
    chain = {k: cand[k][1] for k in END_STEPS}
    jobs = {k: cand[k][0] for k in END_STEPS}
    # verification against log.lammps (holds Ext_05 only)
    logruns = parse_thermo(me_path(net, "log.lammps"))
    ok_log = len(logruns) == 1 and logruns[0].shape == chain["ext5"][0].shape and \
        np.allclose(logruns[0], chain["ext5"][0], rtol=0, atol=0)
    # continuity across restarts: compare PE and box of boundary rows
    cont = []
    order = ["anneal"] + [f"ext{k}" for k in range(1, 6)]
    for a, b in zip(order[:-1], order[1:]):
        ra = chain[a][-1][-1]; rb = chain[b][0][0]
        cont.append(float(np.max(np.abs(ra[[0, 2, 9, 10, 11]] - rb[[0, 2, 9, 10, 11]]) /
                                  np.maximum(1e-12, np.abs(ra[[0, 2, 9, 10, 11]])))))
    # anneal: concatenate its 5 stages, dropping duplicated boundary rows
    ann = [chain["anneal"][0]]
    for r in chain["anneal"][1:]:
        ann.append(r[1:] if r[0, 0] == ann[-1][-1, 0] else r)
    ann = np.vstack(ann)
    prod = [chain[f"ext{k}"][0] for k in range(1, 6)]
    return dict(anneal=ann, anneal_stages=chain["anneal"], prod=prod, jobs=jobs,
                ok_log=ok_log, cont_maxrel=max(cont), n_slurm=len(files))
