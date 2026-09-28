#!/usr/bin/env python
"""Build CHARMM36m atomistic LAMMPS systems from a BFM protein-network topology.

Taken from the build_systems script of the earlier protein-network code topon
grew from. Two integration changes from that CLI:

1. Default PRM/RTF/CMAP files now resolve to the bundled
   `topon/protein_network/charmm/data/` directory rather than an
   absolute path.
2. Topology JSON is interchangeable with `topon.protein_network.bfm`
   output, so `python -m topon.protein_network topology` produces the same
   schema, so a generate step is no longer required if
   you already have a topology file.

Output layout per water content::

    <output_dir>/
        w0/   protein_network.data
              protein_network.in.settings
              protein_network_stage1/2/3.in
        w35/  ...
        w55/  ...

`topon protein --model charmm` builds the same systems from a sequence in one
step. See docs/USAGE.md for the flags of both.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .charmm_ff import CHARMMForceField
from .topology_io import load_topology, get_snapshot, list_snapshots
from .workflow import DEFAULT_CMAP, DEFAULT_PRM, DEFAULT_RTF, write_charmm_system
from ..sequence import build_full_sequence, get_node_residue_mapping

_DEFAULT_PRM = DEFAULT_PRM
_DEFAULT_RTF = DEFAULT_RTF
_DEFAULT_CMAP = DEFAULT_CMAP


def parse_args():
    p = argparse.ArgumentParser(
        description="Build CHARMM36m atomistic LAMMPS systems from a BFM topology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--topology", "-t", required=True,
                   help="Path to topology JSON file (from topon.protein_network.bfm)")
    p.add_argument("--snapshot", default="gel_point",
                   help="Snapshot label or integer index to use")
    p.add_argument("--block_seq", default="GGRPSDSYGAPGGGN",
                   help="One-letter repeat-block sequence")
    p.add_argument("--n_repeats", type=int, default=None,
                   help="Override number of repeats (default: from topology)")
    p.add_argument("--charmm_prm", default=str(_DEFAULT_PRM),
                   help="CHARMM PRM file path")
    p.add_argument("--charmm_rtf", default=str(_DEFAULT_RTF),
                   help="CHARMM RTF file path")
    p.add_argument("--charmm_cmap", default=_DEFAULT_CMAP,
                   help="CHARMM CMAP grid file for fix cmap (copied next to the .data "
                        "files). Default: written from the PRM's CMAP section.")
    p.add_argument("--water_contents", default="0,35,55,65,75",
                   help="Comma-separated weight-percent water values")
    p.add_argument("--salt_conc", type=float, default=0.15,
                   help="NaCl background concentration in mol/L")
    p.add_argument("--lattice_scale", type=float, default=None,
                   help="Override lattice scale in A/BFM unit (default: auto)")
    p.add_argument("--target_density", type=float, default=0.85,
                   help="Target initial density in g/cm^3 for box sizing")
    p.add_argument("--output", "-o", default="output",
                   help="Root output directory")
    p.add_argument("--prefix", default="protein_network",
                   help="File name prefix for LAMMPS output files")
    p.add_argument("--no-image-flags", dest="image_flags", action="store_false",
                   help="Emit legacy 7-column Atoms (no ix iy iz) and do NOT "
                        "drop winding-cycle crosslinks. Keeps the exact "
                        "topology (all crosslinks) but is single-rank only "
                        "(not MPI-safe). Default: emit 10-column image flags "
                        "and drop winding crosslinks (MPI-safe).")
    p.set_defaults(image_flags=True)
    p.add_argument("--physical-backbone", dest="physical_backbone",
                   action="store_true",
                   help="Seed physically correct backbone geometry: trans "
                        "peptide bonds (omega ~180), L-chirality, and coiled "
                        "residue placement at ~3.8 A CA-CA so minimisation "
                        "needs no violent expansion (which otherwise scrambles "
                        "cis/trans + chirality). Default off = legacy jitter "
                        "placement (unchanged byte-for-byte).")
    p.add_argument("--xpro-cis-fraction", dest="xpro_cis_fraction", type=float,
                   default=0.0,
                   help="With --physical-backbone, fraction of X-Pro peptide "
                        "bonds to seed cis (physiological ~0.1-0.2; default 0 = "
                        "all trans).")
    p.set_defaults(physical_backbone=False)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for water and ion placement.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  topon.protein_network.charmm — Atomistic system builder")
    print("=" * 60)

    print(f"\n[1] Loading topology: {args.topology}")
    topo = load_topology(args.topology)
    list_snapshots(topo)

    snap_label = args.snapshot
    try:
        snap_label = int(snap_label)
    except ValueError:
        pass
    snapshot = get_snapshot(topo, snap_label)
    print(f"\n    Using snapshot: '{snapshot['label']}'  conv={snapshot['conv']:.4f}")

    cfg = topo["config"]
    n_repeats = args.n_repeats or cfg["n_repeats"]
    segs_per_block = cfg["segs_per_block"]
    y_offset = cfg["y_offset_in_block"]

    print(f"\n[2] Parsing CHARMM36m force field ...")
    ff = CHARMMForceField(args.charmm_prm, args.charmm_rtf)
    print(f"    {len(ff.masses)} atom types | {len(ff.bonds_prm)} bonds | "
          f"{len(ff.angles_prm)} angles | {len(ff.dihedrals_prm)} dihedrals")
    for patch in ["NTER", "GLYP", "CTER", "DITY"]:
        tag = "[OK]" if patch in ff.patches else "[MISSING]"
        print(f"    {tag} patch {patch}")

    print(f"\n[3] Building sequence ({n_repeats}x '{args.block_seq}') ...")
    full_seq = build_full_sequence(args.block_seq, n_repeats)
    node_to_res = get_node_residue_mapping(
        n_repeats, segs_per_block,
        y_offset_in_block=y_offset,
        block_seq=args.block_seq,
    )
    print(f"    {len(full_seq)} residues | {len(node_to_res)} node mappings")

    water_contents = [float(w) for w in args.water_contents.split(",")]
    for wc in water_contents:
        out_dir = Path(args.output) / f"w{int(wc)}"
        print(f"\n{'-' * 55}")
        print(f"  Water content: {wc:.0f} wt%  ->  {out_dir}")
        print(f"{'-' * 55}")
        write_charmm_system(
            ff, snapshot, full_seq, node_to_res, out_dir,
            prefix=args.prefix,
            water_content_pct=wc,
            salt_conc_M=args.salt_conc,
            lattice_scale=args.lattice_scale,
            target_density=args.target_density,
            physical_backbone=args.physical_backbone,
            xpro_cis_fraction=args.xpro_cis_fraction,
            image_flags=args.image_flags,
            cmap_file=args.charmm_cmap,
            seed=args.seed,
        )

    print(f"\n{'=' * 60}")
    print("  All systems built successfully.")
    print(f"  Output root: {Path(args.output).resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main())
