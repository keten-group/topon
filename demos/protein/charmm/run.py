"""CHARMM36m protein-network demo: resilin, 8 chains x 12 repeats.

Builds the network of `config.json` through the sequence entry point
(`topon.protein_network.network.build_protein_network`, the same as
`topon protein --config config.json`), once dry and once at 35 wt% water:

    <output>/w0/    protein_network.data, .in.settings(.soft, .lj), .in.groups,
                    charmm36m.cmap, relaxation/protein_network_stage{1,2,3}.in,
                    protein_network_summary.json, protein_network_topology.json
    <output>/w35/   the same with TIP3P water and NaCl

LAMMPS is not run here; see README.md for the three stage commands.
`expected_output/` holds the small text files of the dry build and the LAMMPS
logs of its stages.

Usage::

    python demos/protein/charmm/run.py [--output runs/charmm_demo] [--water 0,35]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from topon.protein_network.network import ProteinNetworkSettings, build_protein_network

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="runs/charmm_demo", help="output root")
    ap.add_argument("--water", default="0,35", help="water contents, wt%%")
    args = ap.parse_args()

    for wc in (float(w) for w in args.water.split(",")):
        s = ProteinNetworkSettings.from_json(HERE / "config.json")
        s.water_content = wc
        s.output_dir = str(Path(args.output) / f"w{int(wc)}")
        summary = build_protein_network(s, verbose=False)
        c = summary["counts"]
        print(f"w{int(wc)}: {c['n_atoms']} atoms, {c['n_crosslinks']} crosslinks, "
              f"snapshot {summary['snapshot']['label']!r}, total charge "
              f"{c['total_charge']:+.6f} e -> {s.output_dir}")
    print("\nTo relax, in each <output>/wXX/relaxation/:")
    print("  lmp -in protein_network_stage1.in   # serial, soft overlap removal")
    print("  lmp -in protein_network_stage2.in   # LJ epsilon ramp")
    print("  lmp -in protein_network_stage3.in   # minimisation, NVT, NPT")


if __name__ == "__main__":
    main()
