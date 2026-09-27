"""Shared paths and parsers for the CG structure checks (read-only on raw data).

Raw data (READ-ONLY, not deposited):
  RAW_ROOT/crosslinker/sc_6x6x6/<net>/minimize_equilibrate/*.data, slurm-*.out, log.lammps
Graphs (deposited):
  data/mechanics/<net>/*.gpickle
Mechanics (12 pulls / network):
  data/derived/mechanics/mechanics_12pull.csv                     (network means + Quadrant)
  NPJ_ANALYSIS_OUT/cg_mechanics/four_seed_metrics.csv             (per pull, from four_seed_analysis.py)
Outputs go to NPJ_ANALYSIS_OUT/cg_structure/ (cache/, data/, tables/, figures/). See ../paths.py.
"""
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # raw_data_analyses/, for paths.py
from paths import ANALYSIS_OUT, CG, DATA, out_dir

RAW = str(CG / "sc_6x6x6")
OUT = str(out_dir("cg_structure"))
for _sub in ("cache", "data", "tables", "figures"):
    os.makedirs(os.path.join(OUT, _sub), exist_ok=True)
GRAPHS = str(DATA / "mechanics")
MECH12 = str(DATA / "derived" / "mechanics" / "mechanics_12pull.csv")
PULLS = str(ANALYSIS_OUT / "cg_mechanics" / "four_seed_metrics.csv")

JUNCTION_TYPE = 3   # crosslink / junction beads (206 per network = 216 sites - vacancies)
STRAND_TYPE = 1     # strand (backbone) beads


def networks():
    """The 327 network folders that have an equilibrated data file."""
    out = []
    for d in sorted(os.listdir(RAW)):
        p = os.path.join(RAW, d, "minimize_equilibrate", "equilibrated_network.data")
        if os.path.isfile(p):
            out.append(d)
    return out


def me_path(net, fname):
    return os.path.join(RAW, net, "minimize_equilibrate", fname)


def read_lammps_data(path, want_vel=False):
    """Parse a LAMMPS data file (atom_style full, orthogonal box).

    Returns dict with box lo/hi (3,), ids, mol, type, x (N,3), img (N,3) and
    bonds (M,3: type, i, j as ATOM IDS) and optionally v (N,3) aligned with ids.
    Arrays are sorted by atom id.
    """
    with open(path, "r") as fh:
        lines = fh.read().split("\n")
    lo = np.zeros(3); hi = np.zeros(3)
    tilt = None
    sec = {}
    for k, ln in enumerate(lines[:200]):
        s = ln.split()
        if len(s) >= 4 and s[2] == "xlo":
            lo[0], hi[0] = float(s[0]), float(s[1])
        elif len(s) >= 4 and s[2] == "ylo":
            lo[1], hi[1] = float(s[0]), float(s[1])
        elif len(s) >= 4 and s[2] == "zlo":
            lo[2], hi[2] = float(s[0]), float(s[1])
        elif len(s) >= 6 and s[3] == "xy":
            tilt = (float(s[0]), float(s[1]), float(s[2]))
    # section headers
    names = ("Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers", "Masses",
             "Pair Coeffs", "Bond Coeffs", "Angle Coeffs")
    for k, ln in enumerate(lines):
        t = ln.strip()
        for n in names:
            if t == n or t.startswith(n + " #") or t.startswith(n + "  #"):
                sec[n] = k
    order = sorted(sec.items(), key=lambda kv: kv[1])

    def block(name):
        k0 = sec[name]
        # data starts after one blank line; ends at next section header
        nxt = [v for n, v in order if v > k0]
        k1 = nxt[0] if nxt else len(lines)
        rows = [l for l in lines[k0 + 1:k1] if l.strip()]
        return rows

    at = np.array([r.split()[:10] for r in block("Atoms")], dtype=float)
    idx = np.argsort(at[:, 0])
    at = at[idx]
    res = dict(lo=lo, hi=hi, L=hi - lo, tilt=tilt,
               ids=at[:, 0].astype(np.int64), mol=at[:, 1].astype(np.int64),
               type=at[:, 2].astype(np.int64), x=at[:, 4:7].copy())
    res["img"] = at[:, 7:10].astype(np.int64) if at.shape[1] >= 10 else np.zeros((len(at), 3), np.int64)
    b = np.array([r.split()[:4] for r in block("Bonds")], dtype=np.int64)
    res["bonds"] = b[:, 1:4]
    if want_vel and "Velocities" in sec:
        v = np.array([r.split()[:4] for r in block("Velocities")], dtype=float)
        v = v[np.argsort(v[:, 0])]
        assert np.array_equal(v[:, 0].astype(np.int64), res["ids"])
        res["v"] = v[:, 1:4]
    return res


def min_image(d, L):
    return d - L * np.round(d / L)


def strands_from_bonds(dat):
    """Trace junction-to-junction strands through strand beads.

    Returns list of (jA_index, jB_index, bead_path_indices) using 0-based
    indices into the id-sorted arrays. Each strand appears once.
    """
    ids = dat["ids"]; typ = dat["type"]
    n = len(ids)
    pos = {int(a): i for i, a in enumerate(ids)} if ids[-1] != n or ids[0] != 1 else None
    def ix(a):
        return pos[int(a)] if pos is not None else int(a) - 1
    nbr = [[] for _ in range(n)]
    for _, a, b in dat["bonds"]:
        i, j = ix(a), ix(b)
        nbr[i].append(j); nbr[j].append(i)
    is_j = typ == JUNCTION_TYPE
    seen_first_bond = set()
    strands = []
    for s in np.where(is_j)[0]:
        for nb in nbr[s]:
            key = (min(s, nb), max(s, nb))
            if key in seen_first_bond:
                continue
            path = [s]
            prev, cur = s, nb
            while not is_j[cur]:
                path.append(cur)
                nx = [q for q in nbr[cur] if q != prev]
                if len(nx) != 1:
                    raise RuntimeError(f"strand bead {cur} has {len(nbr[cur])} bonds")
                prev, cur = cur, nx[0]
            path.append(cur)
            seen_first_bond.add(key)
            seen_first_bond.add((min(cur, prev), max(cur, prev)))
            strands.append((s, cur, np.array(path)))
    return strands, nbr


def strand_vectors(dat, strands):
    """End-to-end vectors by summing minimum-image bond vectors along each strand."""
    x = dat["x"]; L = dat["L"]
    R = np.zeros((len(strands), 3))
    maxb = 0.0
    for k, (_, _, p) in enumerate(strands):
        d = min_image(np.diff(x[p], axis=0), L)
        maxb = max(maxb, np.sqrt((d ** 2).sum(1)).max())
        R[k] = d.sum(0)
    return R, maxb


def bond_vectors(dat):
    ids = dat["ids"]
    x = dat["x"]; L = dat["L"]
    i = dat["bonds"][:, 1] - 1; j = dat["bonds"][:, 2] - 1
    assert ids[0] == 1 and ids[-1] == len(ids)
    return min_image(x[j] - x[i], L)
