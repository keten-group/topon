"""CHARMM36m atomistic protein-network builder.

Taken from the earlier protein-network code topon grew from. The BFM topology
stage lives in `topon.protein_network.bfm`, and this subpackage adds the
*atomistic* chemistry stage.

Public entry points: `topon protein --model charmm`,
`topon.protein_network.charmm.build_systems` (CLI), or
`build_protein_system` + `add_water_and_ions` + the LAMMPS writers below
if you're driving it from Python.

The bundled CHARMM36m RTF/PRM files live in `data/` (their sources are in
`data/README.md`). Pass `--charmm_prm/--charmm_rtf/--charmm_cmap` to
`build_systems`, or `--charmm-files` to `topon protein`, to use your own. The files are read by the shared CHARMM reader
(`topon.forcefield.charmm`), and `write_charmm_system` writes one system.
"""
from .charmm_ff import CHARMMForceField
from .builder import (
    CROSSLINK_PATCHES,
    Atom,
    build_protein_system,
    add_water_and_ions,
    compute_lattice_scale,
    find_cmap_crossterms,
    protein_mass,
)
from .lammps_writer import (
    find_angles,
    find_dihedrals,
    parameterize_system,
    write_lammps_data,
    write_lammps_settings,
    write_lammps_groups,
    write_lammps_input,
)
from .workflow import write_charmm_system
from .topology_io import (
    save_topology,
    load_topology,
    get_snapshot,
    list_snapshots,
)

__all__ = [
    "CHARMMForceField",
    "CROSSLINK_PATCHES",
    "Atom",
    "build_protein_system",
    "add_water_and_ions",
    "compute_lattice_scale",
    "find_cmap_crossterms",
    "protein_mass",
    "find_angles",
    "find_dihedrals",
    "parameterize_system",
    "write_lammps_data",
    "write_lammps_settings",
    "write_lammps_groups",
    "write_lammps_input",
    "write_charmm_system",
    "save_topology",
    "load_topology",
    "get_snapshot",
    "list_snapshots",
]
