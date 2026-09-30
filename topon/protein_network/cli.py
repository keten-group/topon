"""argparse CLI for the MARTINI protein-network generator.

Run via ``python -m topon.protein_network`` (which dispatches to this module).

Examples
--------
* Single run, dry::

    python -m topon.protein_network generate --block-seq GGRPSDSYGAPGGGN \\
        --n-repeats 6 --n-chains 4 --output runs/resilin_dry --seed 42

* Sweep multiple water contents (one wXX/ folder each)::

    python -m topon.protein_network sweep --block-seq GGRPSDSYGAPGGGN \\
        --n-repeats 6 --n-chains 4 --water-densities 0,4,8,10 --output runs/resilin

* Two-stage: just generate the BFM topology JSON for inspection::

    python -m topon.protein_network topology --n-chains 16 --n-repeats 12 \\
        --output topo.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import bfm, topology_io, workflow


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--block-seq", default="GGRPSDSYGAPGGGN",
                   help="One-letter repeat block (default: resilin GGRPSDSYGAPGGGN).")
    p.add_argument("--n-repeats", type=int, default=6, help="Number of repeats per chain.")
    p.add_argument("--n-chains", type=int, default=4, help="Number of chains.")
    p.add_argument("--segs-per-block", type=int, default=2, help="BFM segments per repeat.")
    p.add_argument("--equil-steps", type=int, default=5_000,
                   help="Monte Carlo equilibration steps (0 = skip).")
    p.add_argument("--target-packing", type=float, default=0.45)
    p.add_argument("--min-intrachain-sep", type=int, default=2)
    p.add_argument("--lattice-scale-ang", type=float, default=None,
                   help="Angstroms per BFM lattice unit (default: auto-scaled "
                        "so BB-BB equilibrium length = MARTINI 3.6 A).")
    p.add_argument("--sc-jitter-ang", type=float, default=1.5,
                   help="Sidechain bead random offset magnitude (A).")
    p.add_argument("--snapshot-label", default="gel_point",
                   help="Which BFM snapshot to build chemistry from.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--hierarchical-stage1", action="store_true",
                   help="Use core-topon-style progressive freeze/unfreeze in stage 1 "
                        "(safer for the BFM-derived topology; mirrors "
                        "topon/writers/lammps_inputs.py:write_serial_soft_minimization).")


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = workflow.run_protein_network(
        block_seq=args.block_seq,
        n_repeats=args.n_repeats,
        n_chains=args.n_chains,
        output_dir=args.output,
        snapshot_label=args.snapshot_label,
        segs_per_block=args.segs_per_block,
        equil_steps=args.equil_steps,
        target_packing=args.target_packing,
        min_intrachain_sep=args.min_intrachain_sep,
        lattice_scale_ang=args.lattice_scale_ang,
        sc_jitter_ang=args.sc_jitter_ang,
        water_density_w_per_nm3=args.water_density,
        water_exclusion_ang=args.water_exclusion,
        water_bead_type=getattr(args, "water_bead", "W"),
        n_na_ions=getattr(args, "n_na_ions", 0),
        n_cl_ions=getattr(args, "n_cl_ions", 0),
        seed=args.seed,
        hierarchical_stage1=getattr(args, "hierarchical_stage1", False),
        verbose=not args.quiet,
    )
    print("Files written:")
    for kind, p in paths.items():
        print(f"  [{kind}] {p}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    base_out = Path(args.output)
    base_out.mkdir(parents=True, exist_ok=True)
    densities = [float(x) for x in args.water_densities.split(",")]
    for d in densities:
        # Subfolder name encodes density in W/nm^3 (integer if whole, else decimal).
        label = f"w{int(d)}" if d == int(d) else f"w{d:g}".replace(".", "p")
        sub = base_out / label
        print(f"[sweep] water density {d:.2f} W/nm^3 -> {sub}")
        workflow.run_protein_network(
            block_seq=args.block_seq,
            n_repeats=args.n_repeats,
            n_chains=args.n_chains,
            output_dir=sub,
            snapshot_label=args.snapshot_label,
            segs_per_block=args.segs_per_block,
            equil_steps=args.equil_steps,
            target_packing=args.target_packing,
            min_intrachain_sep=args.min_intrachain_sep,
            lattice_scale_ang=args.lattice_scale_ang,
            sc_jitter_ang=args.sc_jitter_ang,
            water_density_w_per_nm3=d,
            water_exclusion_ang=args.water_exclusion,
            water_bead_type=getattr(args, "water_bead", "W"),
            n_na_ions=getattr(args, "n_na_ions", 0),
            n_cl_ions=getattr(args, "n_cl_ions", 0),
            seed=args.seed,
            hierarchical_stage1=getattr(args, "hierarchical_stage1", False),
            verbose=not args.quiet,
        )
    return 0


def _cmd_topology(args: argparse.Namespace) -> int:
    topo = bfm.generate_topology(
        n_chains=args.n_chains,
        n_repeats=args.n_repeats,
        segs_per_block=args.segs_per_block,
        equil_steps=args.equil_steps,
        target_packing=args.target_packing,
        min_intrachain_sep=args.min_intrachain_sep,
        seed=args.seed,
        hierarchical_stage1=getattr(args, "hierarchical_stage1", False),
        verbose=not args.quiet,
    )
    topology_io.save_topology(topo, str(args.output))
    return 0


def add_build_args(b: argparse.ArgumentParser) -> None:
    """Flags of ``build`` (and of ``topon protein``), one per settings field."""
    from .network import CROSSLINK_METHODS, MODELS
    b.add_argument("--config", default=None,
                   help="JSON file with any of the settings below (flags override it).")
    b.add_argument("--sequence", default=None,
                   help="One-letter sequence: the repeat block, or the whole chain.")
    b.add_argument("--model", choices=MODELS, default=None,
                   help="charmm (CHARMM36m all-atom) or martini (Martini 3). Default martini.")
    b.add_argument("--repeats", type=int, default=None, help="Repeats of the block per chain (1).")
    b.add_argument("--chains", type=int, default=None, help="Number of chains (8).")
    b.add_argument("--crosslink-residue", default=None,
                   help="Y (dityrosine, default) or C (disulfide).")
    b.add_argument("--crosslink-method", choices=CROSSLINK_METHODS, default=None,
                   help="melt (default: a residue-level melt, crosslink residues "
                        "joined where they touch), or the BFM node lattice: adjacent, "
                        "winding_safe, distance, none (uncrosslinked, for in-situ "
                        "crosslinking).")
    b.add_argument("--snapshot", default=None,
                   help="Snapshot to build: gel_point (default), post_gel_N, or an index.")
    b.add_argument("--allow-no-gel", action="store_true", default=None,
                   help="Build the last snapshot when the gel point (or the snapshot asked "
                        "for) was not reached, instead of stopping.")
    b.add_argument("--seed", type=int, default=None, help="Seed for every random step (42).")
    b.add_argument("--output", default=None, help="Output directory.")
    b.add_argument("--water-content", type=float, default=None,
                   help="Water, weight percent of the system (0 = dry).")
    b.add_argument("--salt-conc", type=float, default=None,
                   help="NaCl in the water, mol/L (0.15); counter-ions are always added.")
    b.add_argument("--target-density", type=float, default=None,
                   help="Initial density used to size the box, g/cm^3 (0.85).")
    b.add_argument("--equil-steps", type=int, default=None, help="BFM Monte Carlo steps (20000).")
    b.add_argument("--target-packing", type=float, default=None, help="BFM lattice packing (0.45).")
    b.add_argument("--segs-per-block", type=int, default=None, help="Lattice steps per block (2).")
    b.add_argument("--residues-per-segment", type=float, default=None,
                   help="Residues per lattice step for non-repeating layouts.")
    b.add_argument("--n-extra-snapshots", type=int, default=None)
    b.add_argument("--snapshot-delta-conv", type=float, default=None)
    b.add_argument("--min-intrachain-sep", type=int, default=None)
    b.add_argument("--contact-radius", type=float, default=None,
                   help="melt: crosslink residues this close may crosslink, in lattice "
                        "units of about one residue step (1.5, face and edge "
                        "neighbours; 1.8 adds the corners).")
    b.add_argument("--no-physical-backbone", dest="physical_backbone", action="store_const",
                   const=False, default=None,
                   help="CHARMM: place atoms by jitter instead of internal coordinates.")
    b.add_argument("--xpro-cis-fraction", type=float, default=None,
                   help="CHARMM: fraction of X-Pro peptide bonds seeded cis (0).")
    b.add_argument("--charmm-files", nargs="+", default=None,
                   help="CHARMM: RTF, PRM, stream and CMAP files, the whole force field "
                        "(default: bundled CHARMM36m).")
    b.add_argument("--martini-itp", default=None,
                   help="MARTINI: chain ITP to use instead of polyply's.")
    b.add_argument("--water-bead", choices=["W", "SW", "TW"], default=None)
    b.add_argument("--quiet", action="store_true")


def _cmd_build(args: argparse.Namespace) -> int:
    from dataclasses import fields
    from .network import ProteinNetworkSettings, build_protein_network
    base = (ProteinNetworkSettings.from_json(args.config).__dict__ if args.config else {})
    flag = {"output": "output_dir"}
    for f in fields(ProteinNetworkSettings):
        v = getattr(args, {v: k for k, v in flag.items()}.get(f.name, f.name), None)
        if v is not None:
            base[f.name] = v
    if not base.get("sequence"):
        raise SystemExit("give --sequence (or a config with 'sequence')")
    from topon.forcefield.charmm import MissingCharmmParameters
    from .martini_topology import MartiniTopologyError
    try:
        summary = build_protein_network(ProteinNetworkSettings(**base), verbose=not args.quiet)
    except (ValueError, MissingCharmmParameters, MartiniTopologyError, FileNotFoundError) as e:
        # bad settings, a sequence a model cannot take, a network that did
        # not gel: the message says what to change
        print(f"error: {e}", file=sys.stderr)
        return 1
    c, snap = summary["counts"], summary["snapshot"]
    n = c.get("n_atoms", c.get("n_beads"))
    print(f"[protein] {summary['settings']['model']}: {n} particles, total charge "
          f"{c['total_charge']:+.4f}, snapshot {snap['label']!r}, {snap['n_clusters']} "
          f"cluster(s), the largest holding {snap['largest_cluster_chains']} of "
          f"{snap['n_chains']} chains")
    for k, v in summary["files"].items():
        print(f"  [{k}] {v}")
    return 0


def _cmd_check_bonds(args: argparse.Namespace) -> int:
    import json

    from .bond_check import check_bonds
    data = Path(args.data)
    ref = Path(args.reference) if args.reference else None
    if ref is None:
        # a relaxed file sits in the build folder or its relaxation/ folder
        for d in (data.parent, data.parent.parent):
            if (d / "protein_network.data").exists() and d / "protein_network.data" != data:
                ref = d / "protein_network.data"
                break
    settings = args.settings
    if settings is None and ref is not None and (ref.parent / "protein_network.in.settings").exists():
        settings = ref.parent / "protein_network.in.settings"
    rep = check_bonds(data, reference=ref, settings=settings, factor=args.factor)
    if rep["n_stretched"] is not None:
        print(f"{rep['n_stretched']} of {rep['n_bonds']} bonds are longer than "
              f"{args.factor} r0")
    print(f"{rep['n_threaded']} bond(s) pass through a ring")
    for t in rep["threaded"]:
        print(f"  {' - '.join(t['bond'])} through {', '.join(t['ring'])}")
    out = Path(args.json) if args.json else data.with_name(data.stem + "_bond_check.json")
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  [report] {out}")
    summary = (ref.parent if ref else data.parent) / "protein_network_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text(encoding="utf-8"))
        s.setdefault("bond_checks", {})[data.name] = {
            "n_bonds": rep["n_bonds"], "n_stretched": rep["n_stretched"],
            "stretch_factor": args.factor, "n_threaded": rep["n_threaded"]}
        summary.write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(f"  [summary] {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="topon.protein_network",
        description="Protein-network generator (sequence -> LAMMPS), CHARMM36m or Martini 3.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bld = sub.add_parser("build", help="Sequence -> network -> LAMMPS, with --model charmm|martini.")
    add_build_args(bld)
    bld.set_defaults(func=_cmd_build)

    chk = sub.add_parser("check-bonds",
                         help="Stretched bonds and bonds threaded through rings in a data file.")
    chk.add_argument("data", help="LAMMPS data file (e.g., system_equilibrated.data).")
    chk.add_argument("--reference", default=None,
                     help="As-built protein_network.data, for atom names (found if nearby).")
    chk.add_argument("--settings", default=None,
                     help="Include with bond_coeff lines, if the data file has no Bond Coeffs.")
    chk.add_argument("--factor", type=float, default=1.25,
                     help="A bond is stretched above this multiple of r0 (1.25).")
    chk.add_argument("--json", default=None, help="Report file (default <data>_bond_check.json).")
    chk.set_defaults(func=_cmd_check_bonds)

    g = sub.add_parser("generate", help="Run topology + chemistry + LAMMPS write.")
    _add_common_args(g)
    g.add_argument("--output", required=True, help="Output directory.")
    g.add_argument("--water-density", type=float, default=0.0,
                   help="Water beads per nm^3 (0 = dry; ~10 for bulk MARTINI water = W bead default).")
    g.add_argument("--water-exclusion", type=float, default=4.0,
                   help="Min protein-water distance (A).")
    g.add_argument("--water-bead", default="W", choices=["W", "SW", "TW"],
                   help="MARTINI water bead type: W = 4 H2O/bead (default, bulk water), "
                        "SW = 3 H2O/bead (small, for confined water), "
                        "TW = 2 H2O/bead (tiny, for very tight pockets).")
    g.add_argument("--n-na-ions", type=int, default=0, help="NA+ ions to pack.")
    g.add_argument("--n-cl-ions", type=int, default=0, help="CL- ions to pack.")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("sweep", help="Run a water-content sweep into wXX/ subdirs.")
    _add_common_args(s)
    s.add_argument("--output", required=True, help="Base output directory.")
    s.add_argument("--water-densities", default="0,4,8",
                   help="Comma-separated water-bead densities to sweep (default: 0,4,8).")
    s.add_argument("--water-exclusion", type=float, default=4.0)
    s.add_argument("--water-bead", default="W", choices=["W", "SW", "TW"],
                   help="MARTINI water bead type per sweep point (W=4 H2O, SW=3, TW=2).")
    s.add_argument("--n-na-ions", type=int, default=0, help="NA+ ions to pack per density.")
    s.add_argument("--n-cl-ions", type=int, default=0, help="CL- ions to pack per density.")
    s.set_defaults(func=_cmd_sweep)

    t = sub.add_parser("topology", help="Generate just the BFM topology JSON.")
    _add_common_args(t)
    t.add_argument("--output", required=True, help="Output JSON path.")
    t.set_defaults(func=_cmd_topology)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
