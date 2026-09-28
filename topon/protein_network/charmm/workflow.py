"""One CHARMM36m protein-network system, from snapshot to LAMMPS files.

``write_charmm_system`` is the step both the older CLI
(``python -m topon.protein_network.charmm.build_systems``) and the sequence
entry point (``topon protein --model charmm``) run for each water content:
place the atoms, apply the crosslink patches, solvate, assign image flags,
look up every CHARMM term and write the data, settings, groups and the three
relaxation scripts.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from topon.forcefield.charmm import write_lammps_cmap

from .builder import (
    add_water_and_ions,
    build_protein_system,
    compute_lattice_scale,
    find_cmap_crossterms,
    protein_mass,
)
from .lammps_writer import (
    find_chirality_impropers,
    find_omega_dihedrals,
    parameterize_system,
    write_lammps_data,
    write_lammps_groups,
    write_lammps_input,
    write_lammps_lj_settings,
    write_lammps_settings,
)

_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_PRM = _DATA / "par_all36m_prot_C2L.prm"
DEFAULT_RTF = _DATA / "top_all36m_prot_C2L.rtf"
#: The CMAP grid file is written from the PRM's CMAP section (see
#: topon.forcefield.charmm.write_lammps_cmap) unless a file is passed.
DEFAULT_CMAP = "from-prm"
CMAP_NAME = "charmm36m.cmap"


def _image_flags(atoms, bonds, xlinks, box):
    """Priority-MST image flags; ``keep[k]`` is False for a bond no flags fix."""
    from topon.protein_network.lammps_writer import _kruskal_image_flags_and_drop
    xl = {frozenset(p) for p in xlinks}
    wrapped = {a.idx: tuple(float(v) for v in (a.pos % box)) for a in atoms}
    all_b = [(i, j, 1, 0.0, 0.0, frozenset((i, j)) in xl) for (i, j) in bonds]
    flags, keep = _kruskal_image_flags_and_drop(wrapped, all_b, *map(float, box))
    real = [bonds[k] for k, kp in enumerate(keep) if not kp and not all_b[k][5]]
    if real:
        raise RuntimeError(f"image-flag pass would drop {len(real)} non-crosslink "
                           f"bond(s), first {real[:3]}; only winding crosslinks may drop")
    return flags, keep


def dry_protein_mass(ff, snapshot, full_seq, node_to_res, crosslink_patch=None) -> float:
    """Mass of the dry network in Da (independent of the lattice scale)."""
    atoms, *_ = build_protein_system(ff, snapshot, full_seq, node_to_res,
                                     lattice_scale=10.0, crosslink_patch=crosslink_patch)
    return protein_mass(atoms, ff)


def write_charmm_system(
    ff,
    snapshot: dict,
    full_seq: list[str],
    node_to_res: dict[int, int],
    out_dir: str | Path,
    *,
    prefix: str = "protein_network",
    water_content_pct: float = 0.0,
    salt_conc_M: float = 0.15,
    lattice_scale: float | None = None,
    target_density: float = 0.85,
    physical_backbone: bool = False,
    xpro_cis_fraction: float = 0.0,
    image_flags: bool = True,
    cmap_file: str | Path | None = DEFAULT_CMAP,
    crosslink_patch: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Build one system and write its LAMMPS files into ``out_dir``.

    Returns a summary with the file paths, atom and term counts, the number of
    crosslinks kept, the net charge and the number of wildcard-matched terms.
    Raises :class:`topon.forcefield.charmm.MissingCharmmParameters` if any
    term is undefined, and ``ValueError`` if the protein charge is not an
    integer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_dir / prefix)

    if lattice_scale is None:
        mass = dry_protein_mass(ff, snapshot, full_seq, node_to_res, crosslink_patch)
        lattice_scale = compute_lattice_scale(snapshot["Nx"], mass, water_content_pct,
                                              target_density=target_density)
    if verbose:
        print(f"  Lattice scale: {lattice_scale:.3f} A/unit | box ~= "
              f"{snapshot['Nx'] * lattice_scale:.1f}^3 A")

    def build(snap):
        return build_protein_system(
            ff, snap, full_seq, node_to_res, lattice_scale=lattice_scale,
            physical_backbone=physical_backbone, xpro_cis_fraction=xpro_cis_fraction,
            crosslink_patch=crosslink_patch,
        )

    atoms, bonds, impropers, box, xlinks = build(snapshot)
    n_dropped = 0
    if image_flags:
        # Image flags come from a priority MST over the bonds (crosslinks
        # last), so every tree bond is minimum-image. A crosslink closing a
        # cycle that winds around the box cannot be made short by any flags;
        # its reaction is removed and the system rebuilt, so the two residues
        # stay unpatched rather than keeping a patch without its bond.
        flags, keep = _image_flags(atoms, bonds, xlinks, box)
        dropped = {frozenset(bonds[k]) for k, kp in enumerate(keep) if not kp}
        if dropped:
            reactions = snapshot.get("reactions", [])
            if len(reactions) != len(xlinks):
                raise RuntimeError("expected one crosslink bond per reaction")
            kept = [r for r, b in zip(reactions, xlinks) if frozenset(b) not in dropped]
            n_dropped = len(reactions) - len(kept)
            snapshot = {**snapshot, "reactions": kept}
            atoms, bonds, impropers, box, xlinks = build(snapshot)
            if verbose:
                print(f"  [image-flags] removed {n_dropped} winding-cycle crosslink(s) "
                      f"of {len(reactions)} (cannot be made minimum-image)")
    prot_charge = sum(a.charge for a in atoms)
    n_prot_atoms = len(atoms)
    n_w, n_cat, n_an = add_water_and_ions(
        atoms, bonds, box, water_content_pct=water_content_pct,
        salt_conc_M=salt_conc_M, ff=ff, seed=seed, verbose=verbose,
    )
    if image_flags:
        flags, keep = _image_flags(atoms, bonds, xlinks, box)
        if not all(keep):
            raise RuntimeError("a bond still cannot be made minimum-image after "
                               "removing the winding crosslinks")
    else:
        flags = None

    crossterms = find_cmap_crossterms(atoms)
    terms = parameterize_system(ff, atoms, bonds, impropers)

    cmap_name = None
    if cmap_file == DEFAULT_CMAP:
        # The grids of the parameter files themselves, so the CMAP is the
        # same force field as every other term (CHARMM36m here).
        write_lammps_cmap(ff.params, out_dir / CMAP_NAME)
        cmap_name = CMAP_NAME
    elif cmap_file is not None:
        if not os.path.exists(cmap_file):
            raise FileNotFoundError(f"CMAP file not found: {cmap_file}")
        cmap_dst = out_dir / os.path.basename(cmap_file)
        shutil.copy2(cmap_file, cmap_dst)
        cmap_name = cmap_dst.name

    data_file = f"{base}.data"
    settings_file = f"{base}.in.settings"
    soft_file = f"{base}.in.settings.soft"
    lj_file = f"{base}.in.settings.lj"
    groups_file = f"{base}.in.groups"
    old_to_new = write_lammps_data(data_file, atoms, terms, box, crossterms=crossterms,
                                   image_flags=flags)
    write_lammps_settings(settings_file, terms)
    write_lammps_settings(soft_file, terms, soft=True)
    write_lammps_lj_settings(lj_file, terms)
    write_lammps_groups(groups_file, atoms, terms)
    omega_quads = chir_quads = None
    if physical_backbone:
        # The restraints name atoms by their ids in the data file, which
        # renumbers 1..N after the patches delete atoms (DITY drops two HE2),
        # so every quad goes through the same map as the data file.
        m = old_to_new
        omega_quads = [(m[a], m[b], m[c], m[d], xpro)
                       for a, b, c, d, xpro in find_omega_dihedrals(atoms, bonds)]
        chir_quads = [tuple(m[x] for x in q) for q in find_chirality_impropers(atoms, bonds)]
    write_lammps_input(
        base, os.path.basename(data_file), os.path.basename(settings_file), box,
        groups_file=os.path.basename(groups_file), cmap_file=cmap_name,
        omega_quads=omega_quads, xpro_cis_fraction=xpro_cis_fraction,
        chirality_quads=chir_quads, soft_settings_file=os.path.basename(soft_file),
        lj_settings_file=os.path.basename(lj_file),
        velocity_seed=12345 + seed,
        water_shake=((terms.bond_types[("HT", "OT")], terms.angle_types[("HT", "OT", "HT")])
                     if n_w else None),
    )
    total_charge = sum(a.charge for a in atoms)
    relax = out_dir / "relaxation"
    summary = {
        "data": Path(data_file), "settings": Path(settings_file),
        "settings_soft": Path(soft_file), "settings_lj": Path(lj_file),
        "groups": Path(groups_file),
        "stage1": relax / f"{prefix}_stage1.in",
        "stage2": relax / f"{prefix}_stage2.in",
        "stage3": relax / f"{prefix}_stage3.in",
        "n_atoms": len(atoms), "n_protein_atoms": n_prot_atoms,
        "n_bonds": len(terms.bonds), "n_angles": len(terms.angles),
        "n_dihedrals": len(terms.dihedrals), "n_impropers": len(terms.impropers),
        "n_crossterms": len(crossterms),
        "n_crosslinks": len(xlinks), "n_crosslinks_dropped": n_dropped,
        "n_waters": n_w, "n_cations": n_cat, "n_anions": n_an,
        # float sums of RTF charges, rounded past their last digit (+ 0.0
        # so a neutral system does not read -0.0)
        "protein_charge": round(prot_charge, 9) + 0.0,
        "total_charge": round(total_charge, 9) + 0.0,
        "wildcard_terms": dict(terms.wildcard_counts),
        "box": tuple(float(x) for x in box),
    }
    if verbose:
        print(f"  [OK] {len(atoms)} atoms | final charge = {total_charge:.4f} e | "
              f"{data_file}")
    return summary
