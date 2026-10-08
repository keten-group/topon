"""
SimBox crosslink workflow, the packaged entry point.

``run_workflow`` builds the molecules, packs the box and writes the LAMMPS
files; ``prepare_bond_react`` then makes an epoxy-amine box ready for
``fix bond/react``: it adds the bond, angle and dihedral types the cure
creates and writes the reaction templates in the box's own type ids.

Every type id is the one DREIDING typing gives the box. Before 0.4.5
a ``UniversalTypeMapper`` patched ``topon.forcefield.dreiding`` at write time
to force fixed ids (Si3 1, O_3 2, C_3 3, N_3 4, H_ 5, and fixed bond, angle
and dihedral ids), so that hand-written templates fitted every composition.
It matched dihedral types by their atom types only, so two types of one
name and different K shared an id: the epoxide ring's torsions (K 0.125
about the ring C-C bond, 0.5 about the ring C-O bond) were written with the
chain's 0.111111 and 0.333333, and this writer listed both K under the one
id, which LAMMPS refuses to read. Templates are now generated from each
box's own types (:mod:`topon.simbox.react_templates`), so fixed ids
have no use.

This module is the canonical implementation. The regression driver of the
development repository uses its ``prepare_bond_react``.
"""

from __future__ import annotations

import time
from pathlib import Path

# rdkit and dreiding are imported lazily (inside SimBox.write and
# prepare_bond_react) to keep the CLI's startup free of them.
from topon.simbox import SimBox, MoleculeLibrary

#: Reactive-site groups (``topon.simbox.molecule``) of an epoxy-amine box.
EPOXIDE_GROUP = "epoxide"
AMINE_GROUPS = ("primary_amine", "secondary_amine")


# ---------------------------------------------------------------------------
# prepare_bond_react
# ---------------------------------------------------------------------------

def prepare_bond_react(system, output_dir: str | Path, files: dict,
                       verbose: bool = True) -> dict:
    """Make a written epoxy-amine box ready for ``fix bond/react``.

    A reaction cannot add a type, so the data file must already list every
    bond, angle and dihedral type the cure creates: the O-H bond, the C-O-H
    and C-N-C angles and the torsions through the new C-N and O-H bonds,
    which no molecule of an uncured box holds.
    :func:`~topon.simbox.react_templates.add_reaction_types` appends them to
    ``system.data`` (DREIDING's coefficients, a dihedral's K from the
    torsions about its central bond in the template), and they are appended
    to ``ff_coeffs.in`` as the same commands. The four molecule templates
    and two maps of the primary- and secondary-amine reactions are then
    written beside them in this box's type ids
    (:func:`~topon.simbox.react_templates.write_epoxy_amine_templates`).

    *system* is the box's ``AssembledSystem`` and *files* what
    ``SimBox.write`` returned. A box without both an epoxide and an amine is
    left as written. Returns ``{"types_added": [(kind, id, names,
    coefficients)], "templates": [file names]}``, both empty for such a box;
    *files* gains each template's path under its file stem.
    """
    groups = {entry.group_name for entry in system.reactive_sites}
    if EPOXIDE_GROUP not in groups or not groups.intersection(AMINE_GROUPS):
        if verbose and (EPOXIDE_GROUP in groups or groups.intersection(AMINE_GROUPS)):
            print(f"[bond/react] the box holds {', '.join(sorted(groups))} but not both an "
                  f"epoxide and an amine ({EPOXIDE_GROUP}; {' or '.join(AMINE_GROUPS)}): "
                  f"no reaction types, no templates")
        return {"types_added": [], "templates": []}

    from topon.simbox.react_templates import add_reaction_types, write_epoxy_amine_templates

    out = Path(output_dir)
    data = Path(files["data"])
    added = add_reaction_types(data, data)
    if added and "ff_coeffs" in files:
        _append_ff_coeffs(Path(files["ff_coeffs"]), added)
    written = write_epoxy_amine_templates(data, out)
    for name in written["files"]:
        files[Path(name).stem] = str(out / name)

    if verbose:
        for kind, tid, names, _c in added:
            print(f"[bond/react] added {kind} type {tid} {'-'.join(names)} "
                  f"(a reaction creates it)")
        print(f"[bond/react] templates in this box's type ids: "
              f"{', '.join(written['files'])}")
    return {"types_added": added, "templates": written["files"]}


def _append_ff_coeffs(path: Path, added) -> None:
    """The types a reaction creates, as ``*_coeff`` commands after the box's own."""
    lines = ["# --- Types a reaction creates (fix bond/react; also in system.data) ---"]
    for kind, tid, names, c in added:
        label = "-".join(names)
        if kind == "dihedral":
            lines.append(f"dihedral_coeff {tid} {c[0]:.6f} {int(c[1])} {int(c[2])}  # {label}")
        else:
            lines.append(f"{kind}_coeff {tid} {c[0]:g} {c[1]:g}  # {label}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n\n")


# ---------------------------------------------------------------------------
# run_workflow
# ---------------------------------------------------------------------------

def run_workflow(
    output_dir: str | Path,
    n_epoxy: int = 50,
    n_amino: int = 25,
    n_poss: int = 10,
    density: float = 0.85,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full SimBox crosslink workflow: build molecules, pack box, write LAMMPS files.

    Parameters
    ----------
    output_dir : path
        Directory where all output files will be written.
    n_epoxy : int
        Number of Epoxy-PDMS molecules.
    n_amino : int
        Number of Amino-PDMS molecules.
    n_poss : int
        Number of AM0270-POSS molecules.
    density : float
        Target packing density in g/cm³.
    seed : int
        Random seed for reproducible packing.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    dict
        Mapping of file labels to absolute path strings (keys: ``data``,
        ``settings``, ``groups``, ``ff_coeffs``, ``settings_x6``, ``minimize``,
        ``nvt``, ``npt``, ``crosslink``, and for a box with epoxide and amine
        the templates of :func:`prepare_bond_react`: ``pre_react_primary``,
        ``post_react_primary``, ``rxn_map_primary`` and the same three for
        ``secondary``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if verbose:
        print("=" * 60)
        print("SimBox Crosslink Workflow")
        print("=" * 60)

    lib = MoleculeLibrary()
    epoxy = lib.epoxy_pdms(n_dms=2)
    amino = lib.amino_pdms(n_dms=8)
    poss = lib.am0270_poss()

    if verbose:
        print(f"\n[1/4] Molecules: {epoxy}  |  {amino}  |  {poss}")

    box = SimBox(density=density, temperature=300.0, pressure=1.0)
    if n_epoxy > 0:
        box.add(epoxy, count=n_epoxy)
    if n_amino > 0:
        box.add(amino, count=n_amino)
    if n_poss > 0:
        box.add(poss, count=n_poss)

    if verbose:
        print(f"[2/4] {box.summary()}")
        print(f"[3/4] Packing {n_epoxy + n_amino + n_poss} molecules (seed={seed})...")

    box.pack(seed=seed)

    if verbose:
        print(f"[4/4] Writing LAMMPS files to {output_dir}...")

    files = box.write(str(output_dir), forcefield="dreiding")
    prepare_bond_react(box.system, output_dir, files, verbose=verbose)

    if verbose:
        elapsed = time.time() - t0
        system = box.system
        print(f"\n{'=' * 60}")
        print("WORKFLOW COMPLETE")
        print(f"  Time:      {elapsed:.1f} s")
        print(f"  Atoms:     {system.mol.GetNumAtoms()}")
        print(f"  Molecules: {system.num_molecules}")
        bl = system.box_lengths
        print(f"  Box:       {bl[0]:.2f} x {bl[1]:.2f} x {bl[2]:.2f} Å")
        print(f"  Next: cd {output_dir} && lmp -in 1_minimize.in")
        print("=" * 60)

    return files
