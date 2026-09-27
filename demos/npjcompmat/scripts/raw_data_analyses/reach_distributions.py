"""Reach of load-bearing strands in units of the mean junction spacing, a = (V / N_junctions)^(1/3).

reach = |r_junction2 - r_junction1| / a   (distance taken along the strand, periodic images unwrapped)

States: bond/create reference (final data file), Topon as built on the lattice (build.data, before any
relaxation), Topon after relaxation at the reference state (stage5_final_quench.data).
Inputs (not deposited): the bond/create validation tree RAW_ROOT/bond_create_validation/ with the reference data
files N20_new_mol.data and N100_new_mol.data, the Topon runs data/runs/N{20,100}_v3b/, and the parser
scripts/refnet.py. Writes reach_distributions.json to NPJ_ANALYSIS_OUT/bond_create/ (see paths.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))     # raw_data_analyses/, for paths.py
from paths import RAW_ROOT, out_dir

ROOT = RAW_ROOT / "bond_create_validation"          # bond/create validation tree (not deposited)
sys.path.insert(0, str(ROOT / "scripts"))           # refnet.py of that tree
import refnet
CASES = {
    "N20": {"reference": ROOT / "N20_new_mol.data",
            "topon_built": ROOT / "data/runs/N20_v3b/build.data",
            "topon_relaxed": ROOT / "data/runs/N20_v3b/stage5_final_quench.data"},
    "N100": {"reference": ROOT / "N100_new_mol.data",
             "topon_built": ROOT / "data/runs/N100_v3b/build.data",
             "topon_relaxed": ROOT / "data/runs/N100_v3b/stage5_final_quench.data"},
}


def reach(path):
    net = refnet.parse(path)
    L, pos, J = net["L"], net["pos"], net["junctions"]
    V = float(np.prod(L)) if np.ndim(L) else float(L) ** 3
    a = (V / len(J)) ** (1.0 / 3.0)
    r = []
    for c in net["chains"].values():
        if c["cls"] != "bridge":
            continue
        x1, x2 = c["ends"]
        q = refnet.unwrap([x1] + c["seq"] + [x2], pos, L)
        r.append(np.linalg.norm(q[-1] - q[0]) / a)
    r = np.array(r)
    return dict(a=a, n_junctions=len(J), reach=r.tolist(), mean=float(r.mean()), sd=float(r.std()),
                median=float(np.median(r)), p95=float(np.percentile(r, 95)))


out = {}
for case, states in CASES.items():
    out[case] = {}
    for tag, p in states.items():
        res = reach(p)
        out[case][tag] = res
        print(f"{case:5s} {tag:14s} a = {res['a']:6.3f}  N_j = {res['n_junctions']:5d}  reach mean {res['mean']:.3f} "
              f"sd {res['sd']:.3f} median {res['median']:.3f} p95 {res['p95']:.3f}", flush=True)
json.dump(out, open(out_dir("bond_create") / "reach_distributions.json", "w"))
