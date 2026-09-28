"""Martini 3 protein-network demo: resilin, 8 chains x 18 repeats, dry.

Builds the network of `config.json` through the sequence entry point
(`topon.protein_network.network.build_protein_network`, the same as
`topon protein --config config.json`). The resilin chain is the vendored
polyply topology, so polyply is not needed; any other sequence is generated
with polyply.

Outputs (under --output, default runs/martini_demo):
    protein_network.data, .in.settings, .in.groups
    protein_network_chain.itp            the chain topology used
    protein_network_topology.json        BFM snapshots
    protein_network_summary.json         inputs, snapshot, counts, charges
    relaxation/protein_network_stage{1,2,3}.in

`expected_output/` holds the small text files of this build and the LAMMPS logs
of its stages.

Usage::

    python demos/protein/martini/run.py [--output runs/martini_demo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from topon.protein_network.network import ProteinNetworkSettings, build_protein_network

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="runs/martini_demo", help="output directory")
    args = ap.parse_args()
    s = ProteinNetworkSettings.from_json(HERE / "config.json")
    s.output_dir = args.output
    summary = build_protein_network(s, verbose=False)
    c = summary["counts"]
    print(f"{c['n_beads']} beads, {c['n_crosslinks_written']} crosslinks "
          f"({c['n_crosslinks_dropped']} winding crosslink dropped), snapshot "
          f"{summary['snapshot']['label']!r}, total charge {c['total_charge']:+.4f}")
    print(f"\nTo relax, in {Path(args.output) / 'relaxation'}:")
    print("  lmp -in protein_network_stage1.in   # ~3 s soft overlap removal")
    print("  lmp -in protein_network_stage2.in   # ~40 s LJ epsilon ramp")
    print("  lmp -in protein_network_stage3.in   # ~10 s CG min + brief NVT")


if __name__ == "__main__":
    main()
