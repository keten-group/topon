"""
topon.workflows.reactive_crosslink
====================================
Canonical workflow: reactive crosslink simulation box (Epoxy-PDMS / Amino-PDMS / POSS).

Four-stage pipeline
-------------------
1. Molecules  — build molecular definitions from MoleculeLibrary
2. Box        — create SimBox, add molecule counts, compute box size from density
3. Packing    — random placement of all molecules (packmol-style, seed-controlled)
4. Output     — write LAMMPS data + input scripts, then make the box ready for
                ``fix bond/react`` (``topon.simbox.workflow.prepare_bond_react``)

Type ids
--------
Every type id is the one DREIDING typing gives the box. A box with epoxide
and amine also lists the bond, angle and dihedral types the cure creates,
and its bond/react templates are written beside it in its own ids.
The universal type map this module used to apply (fixed ids for
hand-written templates) matched dihedrals by atom types only and wrote the
epoxide ring's torsions with the chain's K; see ``topon.simbox.workflow``.

Usage (Python)
--------------
    from topon.workflows.reactive_crosslink import run

    run(
        output_dir="output/crosslink_run",
        n_epoxy=50,
        n_amino=25,
        n_poss=0,
        density=0.85,
        seed=42,
    )

Usage (CLI)
-----------
    python -m topon.workflows.reactive_crosslink \\
        --output output/crosslink_run \\
        --n_epoxy 50 --n_amino 25 --n_poss 0 \\
        --density 0.85 --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from topon.simbox import SimBox, MoleculeLibrary
from topon.simbox.workflow import prepare_bond_react


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    output_dir: str | Path,
    n_epoxy: int = 50,
    n_amino: int = 25,
    n_poss: int = 0,
    density: float = 0.85,
    seed: int = 42,
) -> Path:
    """
    Run the full reactive crosslink workflow.

    Parameters
    ----------
    output_dir : output directory for LAMMPS files
    n_epoxy    : number of Epoxy-PDMS molecules (bifunctional epoxide)
    n_amino    : number of Amino-PDMS molecules (bifunctional amine)
    n_poss     : number of AM0270-POSS molecules (monofunctional amine)
    density    : target density in g/cm3
    seed       : random seed for packing

    Returns
    -------
    Path to output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print("[Stage 1] Building molecules...")

    lib = MoleculeLibrary()
    epoxy = lib.epoxy_pdms(n_dms=2)    # bifunctional epoxide, 77 atoms w/ H
    amino = lib.amino_pdms(n_dms=8)    # bifunctional amine,   123 atoms w/ H
    poss  = lib.am0270_poss()          # monofunctional amine, ~207 atoms w/ H

    print(f"  {epoxy}")
    print(f"  {amino}")
    print(f"  {poss}")

    print(f"[Stage 2] Creating box (density={density} g/cm3)...")
    box = SimBox(density=density, temperature=300.0, pressure=1.0)
    if n_epoxy > 0:
        box.add(epoxy, count=n_epoxy)
    if n_amino > 0:
        box.add(amino, count=n_amino)
    if n_poss > 0:
        box.add(poss,  count=n_poss)
    print(box.summary())

    print(f"[Stage 3] Packing {n_epoxy + n_amino + n_poss} molecules (seed={seed})...")
    box.pack(seed=seed)

    print(f"[Stage 4] Writing LAMMPS files to {output_dir}...")
    files = box.write(str(output_dir), forcefield="dreiding")
    prepare_bond_react(box.system, output_dir, files)

    elapsed = time.time() - t0
    system = box.system
    print(f"Reactive crosslink workflow complete -> {output_dir} ({elapsed:.1f}s)")
    print(f"  Atoms: {system.mol.GetNumAtoms()}, Molecules: {system.num_molecules}, "
          f"Reactive sites: {len(system.reactive_sites)}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reactive crosslink simulation box (Epoxy-PDMS / Amino-PDMS / POSS)"
    )
    p.add_argument("--output",   required=True, help="Output directory")
    p.add_argument("--n_epoxy",  type=int,   default=50)
    p.add_argument("--n_amino",  type=int,   default=25)
    p.add_argument("--n_poss",   type=int,   default=0)
    p.add_argument("--density",  type=float, default=0.85)
    p.add_argument("--seed",     type=int,   default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        output_dir=args.output,
        n_epoxy=args.n_epoxy,
        n_amino=args.n_amino,
        n_poss=args.n_poss,
        density=args.density,
        seed=args.seed,
    )
