"""LAMMPS files for the CHARMM36m protein network.

Every coefficient comes from :func:`topon.forcefield.charmm.parameterize`, so
the terms follow CHARMM's matching rules, carry the 1-4 LJ parameters, NBFIX
and the dihedral 1-4 weights, and a term the files do not define stops the
build (there is no ``(DEFAULT)`` fallback any more).

Public API
----------
find_angles / find_dihedrals        topology generation from bonds
find_omega_dihedrals                peptide omega quads (physical builds)
find_chirality_impropers            N-C-CA-CB quads (physical builds)
parameterize_system(ff, atoms, bonds, impropers) -> CharmmTerms
write_lammps_data(...)              -> old_to_new id map
write_lammps_settings(...)          full or soft-stage coefficient include
write_lammps_groups(...)            protein / water / ions / per-chain groups
write_lammps_input(...)             the three relaxation scripts

Styles (the CHARMM-GUI LAMMPS set): ``lj/charmmfsw/coul/long 10 12`` with
``pair_modify mix arithmetic`` and PPPM, ``bond harmonic``, ``angle charmm``,
``dihedral charmmfsw``, ``improper harmonic``, ``special_bonds charmm`` and
``fix cmap``. Stages that run ``pair_style soft`` include the soft settings
(no pair coefficients, 1-4 weights 0), because LAMMPS refuses a non-zero
dihedral weight without an ``lj/charmm*`` pair style.
"""
from __future__ import annotations

import os
from collections import defaultdict

from topon.forcefield.charmm import (
    generate_angles,
    generate_dihedrals,
    parameterize,
    write_lj_pair_coeffs,
    write_settings,
)

PAIR_STYLE = "lj/charmmfsw/coul/long 10.0 12.0"
DIHEDRAL_STYLE = "charmmfsw"


# ── Topology traversal ────────────────────────────────────────────────────────

def find_angles(bonds, atom_idx_set=None):
    """All 1-2-3 angle triplets from the bond list."""
    return generate_angles(bonds)


def find_dihedrals(bonds, atom_idx_set=None):
    """All 1-2-3-4 dihedral quadruplets from the bond list."""
    return generate_dihedrals(bonds)


def find_omega_dihedrals(atoms, bonds):
    """Peptide-bond omega dihedrals CA_i-C_i-N_{i+1}-CA_{i+1}, one per peptide
    bond, as (ca_i, c_i, n_j, ca_j, is_xpro) global atom-id quads. `is_xpro`
    is True when the C-terminal residue of the bond (the one contributing N) is
    proline -- those are the bonds allowed a physiological cis population.
    """
    name = {a.idx: a.name for a in atoms}
    resname = {a.idx: a.res_name for a in atoms}
    nbr = defaultdict(list)
    for i, j in bonds:
        nbr[i].append(j)
        nbr[j].append(i)

    def ca_of(x):
        for y in nbr[x]:
            if name.get(y) == "CA":
                return y
        return None

    quads = []
    for i, j in bonds:
        ni, nj = name.get(i), name.get(j)
        if {ni, nj} != {"C", "N"}:        # the peptide C--N bond
            continue
        c = i if ni == "C" else j
        n = j if ni == "C" else i
        cai, caj = ca_of(c), ca_of(n)
        if cai and caj:
            quads.append((cai, c, n, caj, resname.get(n) == "PRO"))
    return quads


def find_chirality_impropers(atoms, bonds):
    """CA-stereocentre impropers N-C-CA-CB, one per non-glycine residue. This
    dihedral is +121 deg for L residues and -121 for D. CHARMM has no
    CA-chirality improper, so physical builds restrain it through the soft
    stages and release it before each output."""
    by_res = defaultdict(dict)
    for a in atoms:
        by_res[(a.chain_id, a.res_id)][a.name] = a.idx
    quads = []
    for res in by_res.values():
        if all(k in res for k in ("N", "C", "CA", "CB")):
            quads.append((res["N"], res["C"], res["CA"], res["CB"]))
    return quads


# ── Parameters ───────────────────────────────────────────────────────────────

def parameterize_system(ff, atoms, bonds, impropers):
    """Look up every term of the built system; raise on any missing one."""
    types = {a.idx: a.atype for a in atoms}
    return parameterize(ff.params, types, bonds, impropers,
                        context="the CHARMM protein network")


# ── LAMMPS data file ──────────────────────────────────────────────────────────

def write_lammps_data(filename, atoms, terms, box, crossterms=None, image_flags=None):
    """Write a LAMMPS ``atom_style full`` data file for ``terms``.

    Atom ids are renumbered 1..N in list order. ``image_flags``
    (``atom.idx -> (ix, iy, iz)``) switches the Atoms rows to 10 columns,
    which percolated networks need under MPI. Returns the id map.
    """
    old_to_new = {a.idx: n for n, a in enumerate(atoms, start=1)}
    m = old_to_new.get
    rows_b = [(t, m(i), m(j)) for t, i, j in terms.bonds]
    rows_a = [(t, m(i), m(j), m(k)) for t, i, j, k in terms.angles]
    rows_d = [(t, m(i), m(j), m(k), m(l)) for t, i, j, k, l in terms.dihedrals]
    rows_i = [(t, m(i), m(j), m(k), m(l)) for t, i, j, k, l in terms.impropers]
    for rows in (rows_b, rows_a, rows_d, rows_i):
        if any(None in r for r in rows):
            raise ValueError("a bonded term refers to an atom that is not written")
    cmap_rows = []
    for (ct, c_prev, n_i, ca_i, c_i, n_next) in crossterms or []:
        ids = [m(x) for x in (c_prev, n_i, ca_i, c_i, n_next)]
        if None not in ids:
            cmap_rows.append((ct, *ids))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("LAMMPS protein network, CHARMM36m all-atom (topon)\n\n")
        f.write(f"{len(atoms)} atoms\n{len(rows_b)} bonds\n{len(rows_a)} angles\n"
                f"{len(rows_d)} dihedrals\n{len(rows_i)} impropers\n")
        if cmap_rows:
            f.write(f"{len(cmap_rows)} crossterms\n")
        f.write(f"\n{len(terms.atom_types)} atom types\n{len(terms.bond_types)} bond types\n"
                f"{len(terms.angle_types)} angle types\n"
                f"{len(terms.dihedral_types)} dihedral types\n"
                f"{len(terms.improper_types)} improper types\n\n")
        f.write(f"0.0 {box[0]:.4f} xlo xhi\n0.0 {box[1]:.4f} ylo yhi\n"
                f"0.0 {box[2]:.4f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for atype, tid in sorted(terms.atom_types.items(), key=lambda x: x[1]):
            f.write(f"{tid} {terms.masses[tid]:.5f} # {atype}\n")
        f.write("\nAtoms # full\n\n")
        for a in atoms:
            tid = terms.atom_types[a.atype]
            pos = a.pos % box
            core = (f"{old_to_new[a.idx]} {a.mol_id} {tid} {a.charge:.6f} "
                    f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
            if image_flags is not None:
                ix, iy, iz = image_flags.get(a.idx, (0, 0, 0))
                core += f" {ix} {iy} {iz}"
            f.write(f"{core} # {a.res_name} {a.name}\n")
        for title, rows in (("Bonds", rows_b), ("Angles", rows_a),
                            ("Dihedrals", rows_d), ("Impropers", rows_i)):
            f.write(f"\n{title}\n\n")
            for n, r in enumerate(rows, 1):
                f.write(f"{n} " + " ".join(str(x) for x in r) + "\n")
        if cmap_rows:
            f.write("\nCMAP\n\n")
            for n, r in enumerate(cmap_rows, 1):
                # 5 backbone atoms: C(i-1) N(i) CA(i) C(i) N(i+1)
                f.write(f"{n} " + " ".join(str(x) for x in r) + "\n")
    return old_to_new


def write_lammps_settings(filename, terms, soft=False):
    """Coefficient include (full, or the soft-stage variant)."""
    write_settings(filename, terms, soft=soft, title="CHARMM36m")


def write_lammps_lj_settings(filename, terms):
    """LJ pair coefficients for the ``lj/cut/coul/long`` ramp stage."""
    write_lj_pair_coeffs(filename, terms)


# ── LAMMPS groups include file ────────────────────────────────────────────────

_WATER_TYPES = ("OT", "HT")
_ION_TYPES = ("SOD", "CLA", "POT", "CAL", "ZN", "FE3P")


def write_lammps_groups(filename, atoms, terms):
    """Groups ``water``, ``ions``, ``protein`` and ``chainNN``, plus the water
    bond and angle type ids for ``fix shake``."""
    tmap = terms.atom_types
    water_tids = sorted(tmap[t] for t in _WATER_TYPES if t in tmap)
    ion_tids = sorted(tmap[t] for t in _ION_TYPES if t in tmap)
    skip = set(water_tids) | set(ion_tids)
    protein_mol_ids = sorted({a.mol_id for a in atoms if tmap[a.atype] not in skip})
    water_bond_tid = terms.bond_types.get(("HT", "OT"))
    water_angle_tid = terms.angle_types.get(("HT", "OT", "HT"))

    with open(filename, "w", encoding="ascii") as f:
        f.write("# Group and SHAKE-type definitions (topon)\n")
        f.write("# Include this file after read_data in every LAMMPS input script.\n\n")
        if water_tids:
            f.write(f"group           water   type {' '.join(map(str, water_tids))}"
                    f"  # OT HT (TIP3P)\n")
        else:
            f.write("# (no water molecules in this system)\n")
        if ion_tids:
            names = [t for t in _ION_TYPES if t in tmap]
            f.write(f"group           ions    type {' '.join(map(str, ion_tids))}"
                    f"  # {' '.join(names)}\n")
        else:
            f.write("# (no ions in this system)\n")
        sub = (["water"] if water_tids else []) + (["ions"] if ion_tids else [])
        if sub:
            f.write(f"group           protein subtract all {' '.join(sub)}\n\n")
        else:
            f.write("group           protein union all\n\n")
        f.write("# --- Per-chain groups ---\n")
        for mid in protein_mol_ids:
            f.write(f"group           chain{mid:02d}  molecule {mid}\n")
        f.write("\n# --- SHAKE type IDs for water (fix shake water ...) ---\n")
        if water_bond_tid is not None:
            f.write(f"variable        water_bond_type  equal {water_bond_tid}  # OT-HT bond\n")
        if water_angle_tid is not None:
            f.write(f"variable        water_angle_type equal {water_angle_tid}  # HT-OT-HT angle\n")
        if water_bond_tid is not None:
            f.write("# SHAKE usage: fix s water shake 1e-4 100 0"
                    " b ${water_bond_type} a ${water_angle_type}\n")
    return os.path.basename(filename)


# ── LAMMPS 3-stage input scripts ──────────────────────────────────────────────

def _style_header(soft=False):
    # charmmfsw reads the force-switch cutoffs from the pair style even when
    # every 1-4 weight is 0, so stages under `pair_style soft` run the plain
    # charmm dihedral (same torsion energy; only the 1-4 treatment differs).
    return ("units           real\n"
            "atom_style      full\n"
            "boundary        p p p\n"
            "bond_style      harmonic\n"
            "angle_style     charmm\n"
            f"dihedral_style  {'charmm' if soft else DIHEDRAL_STYLE}\n"
            "improper_style  harmonic\n"
            "special_bonds   charmm\n")


def _charmm_pair_block(settings, switch_dihedral=False):
    return (f"pair_style      {PAIR_STYLE}\n"
            "pair_modify     mix arithmetic\n"
            "kspace_style    pppm 1.0e-4\n"
            + (f"dihedral_style  {DIHEDRAL_STYLE}\n" if switch_dihedral else "")
            + f"include         {settings}\n")


def write_lammps_input(base_name, data_file, settings_file, box,
                       groups_file=None, cmap_file=None, omega_quads=None,
                       xpro_cis_fraction=0.0, chirality_quads=None,
                       soft_settings_file=None, lj_settings_file=None,
                       velocity_seed=12345, ramp_steps=20000, water_shake=None):
    """Write the three relaxation scripts under ``relaxation/``.

    stage1  soft overlap removal (serial; ``pair_style soft``)
    stage2  soft pre-minimisation, then CHARMM LJ (arithmetic mixing, NBFIX,
            no 1-4) with epsilon ramped from 0.001 to 1 under ``nve/limit``
    stage3  tight minimisation, NVT and NPT at 300 K; writes
            ``../system_equilibrated.data``

    ``soft_settings_file`` is the include written by
    ``write_lammps_settings(..., soft=True)`` (bonded terms, 1-4 weights 0),
    read by stages 1 and 2. ``lj_settings_file`` holds the LJ pair
    coefficients for the ``lj/cut/coul/long`` ramp of stage 2.
    ``water_shake`` is the (O-H bond, H-O-H angle) type pair of a solvated
    system; TIP3P is then held rigid with ``fix shake`` in the MD of stage
    3, as CHARMM runs it. The stage-2 ramp keeps it flexible, since its
    ``nve/limit`` integrator does not combine with SHAKE.
    """
    out_dir = os.path.dirname(os.path.abspath(base_name))
    relax_dir = os.path.join(out_dir, "relaxation")
    os.makedirs(relax_dir, exist_ok=True)
    stage_prefix = os.path.join(relax_dir, os.path.basename(base_name))

    d = "../" + os.path.basename(data_file)
    s = "../" + os.path.basename(settings_file)
    ss = "../" + os.path.basename(soft_settings_file or (settings_file + ".soft"))
    ls = "../" + os.path.basename(lj_settings_file or (settings_file + ".lj"))
    grp_line = f"include         ../{os.path.basename(groups_file)}\n" if groups_file else ""
    shake = ""
    if water_shake is not None:
        b, a = water_shake
        shake = ("# Rigid TIP3P (O-H bonds and H-O-H angle), as CHARMM runs it\n"
                 f"fix             water_shake water shake 1.0e-4 100 0 b {b} a {a}\n")

    if cmap_file:
        cm = "../" + os.path.basename(cmap_file)
        cmap_pre = f"fix             cmap all cmap {cm}\nfix_modify      cmap energy yes\n"
        cmap_rdat = " fix cmap crossterm CMAP"
    else:
        cmap_pre = cmap_rdat = ""

    # Optional omega / chirality restraints (physical_backbone builds only).
    # LAMMPS `fix restrain dihedral` has its minimum at target+180, so target
    # 0 -> 180 deg (trans) and 180 -> 0 deg (cis). Chirality impropers
    # N-C-CA-CB are held at L (121 deg -> target -59).
    omega_fix = omega_unfix = ""
    if omega_quads:
        import random as _random
        rng = _random.Random(20260714)
        # Exactly round(n_xpro * fraction) X-Pro bonds are seeded cis, so the
        # held count is the requested ratio rather than a binomial draw.
        xpro_idx = [i for i, q in enumerate(omega_quads) if len(q) > 4 and q[4]]
        n_want = int(round(len(xpro_idx) * xpro_cis_fraction)) if xpro_cis_fraction > 0 else 0
        cis_set = set(rng.sample(xpro_idx, n_want)) if n_want else set()
        specs = [(q[0], q[1], q[2], q[3], 180.0 if i in cis_set else 0.0)
                 for i, q in enumerate(omega_quads)]
        n_chir = 0
        for q in chirality_quads or []:
            specs.append((q[0], q[1], q[2], q[3], -59.0))
            n_chir += 1
        omega_base = os.path.basename(base_name) + ".in.omega"
        chunk, fix_names, blocks = 1200, [], []
        for gi in range(0, len(specs), chunk):
            fn = f"omega{gi // chunk}"
            fix_names.append(fn)
            grp = specs[gi:gi + chunk]
            lines = [f"fix {fn} all restrain &"]
            for k, (a, b, c, dd, tgt) in enumerate(grp):
                cont = " &" if k < len(grp) - 1 else ""
                lines.append(f"  dihedral {a} {b} {c} {dd} 80.0 80.0 {tgt}{cont}")
            blocks.append("\n".join(lines))
        with open(os.path.join(relax_dir, omega_base), "w", encoding="utf-8") as of:
            of.write(f"# Backbone restraints (K=80 kcal/mol/rad^2). LAMMPS restrain "
                     f"min is at target+180, so omega 0=>trans, 180=>cis; "
                     f"chirality target -59 => 121 deg = L.\n"
                     f"# omega: {len(omega_quads) - len(cis_set)} trans + {len(cis_set)} "
                     f"cis (X-Pro); chirality: {n_chir} L. Released before each output.\n"
                     + "\n".join(blocks) + "\n")
        omega_fix = ("\n# Restrain peptide omega and CA chirality while overlaps are "
                     f"removed (physical backbone).\ninclude         {omega_base}\n")
        omega_unfix = "".join(f"unfix           {n}\n" for n in fix_names)
        unfix_base = os.path.basename(base_name) + ".in.omega_unfix"
        with open(os.path.join(relax_dir, unfix_base), "w", encoding="utf-8") as uf:
            uf.write("# Release the omega/chirality restraints (see .in.omega).\n"
                     + omega_unfix)

    header = _style_header()
    soft_header = _style_header(soft=True)
    thermo = ("thermo_style    custom step pe ke etotal evdwl ecoul epair ebond "
              "eangle edihed eimp press vol temp\n")

    with open(f"{stage_prefix}_stage1.in", "w", encoding="utf-8") as f:
        f.write(f"""\
# Stage 1: soft overlap removal (serial), CHARMM36m protein network (topon)
# pair_style soft only; the soft settings carry the bonded terms with
# 1-4 weights 0 (the CHARMM 1-4 terms need an lj/charmm pair style).
# Run from this relaxation/ directory.

{soft_header}pair_style      soft 1.0

{cmap_pre}read_data       {d}{cmap_rdat}
include         {ss}
pair_coeff      * * 0.0
{grp_line}neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
# Ghosts must reach past the longest as-built bond (the soft pair alone
# gives 3 A), or a bond across the box edge acts on the far image.
comm_modify     mode single cutoff 14.0
{omega_fix}
variable        prefactor equal ramp(0,60)
thermo          100
{thermo}
# Stage A: ramped soft push
fix             soft_push all adapt 1 pair soft a * * v_prefactor
min_style       cg
minimize        1.0e-4 1.0e-6 1000 10000
unfix           soft_push
write_data      min_stage_A.data

# Stage B: nve/limit dynamics under the soft potential
reset_timestep  0
timestep        1.0
fix             soft_push all adapt 1 pair soft a * * v_prefactor
fix             nve_limit all nve/limit 0.1
run             1000
unfix           nve_limit
unfix           soft_push

# Stage C: final soft minimisation
fix             soft_push all adapt 1 pair soft a * * v_prefactor
minimize        1.0e-4 1.0e-6 1000 10000
unfix           soft_push
{omega_unfix}
write_data      system_after_soft.data nocoeff
write_restart   1.restart
""")

    with open(f"{stage_prefix}_stage2.in", "w", encoding="utf-8") as f:
        f.write(f"""\
# Stage 2: LJ epsilon ramped 0.001 -> 1 under nve/limit (parallel-safe)
# fix adapt cannot scale epsilon in the lj/charmm* styles, so the ramp runs
# lj/cut/coul/long with CHARMM's LJ, arithmetic mixing (as CHARMM mixes) and
# the NBFIX pairs; the 1-4 terms stay off here and stage 3 turns the full
# CHARMM36m set on. Run from this relaxation/ directory.

{soft_header}pair_style      soft 1.0

{cmap_pre}read_data       system_after_soft.data{cmap_rdat}
include         {ss}
pair_coeff      * * 1.0
{grp_line}{omega_fix}
neigh_modify    one 10000
comm_modify     mode single cutoff 14.0

# Soft pre-minimisation
min_style       cg
minimize        1.0e-4 1.0e-6 1000 10000

# CHARMM LJ (no 1-4) + Coulomb with PPPM
pair_style      lj/cut/coul/long 12.0
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-4
include         {ls}

variable        scale equal ramp(0.001,1.0)
timestep        1.0
thermo          1000
{thermo}
fix             1 all adapt 1 pair lj/cut/coul/long epsilon * * v_scale scale yes
fix             fxnve all nve/limit 0.1
run             {ramp_steps}
unfix           fxnve
unfix           1
{omega_unfix}
write_data      system_ramped.data nocoeff
write_restart   2.restart
""")

    with open(f"{stage_prefix}_stage3.in", "w", encoding="utf-8") as f:
        f.write(f"""\
# Stage 3: tight minimisation, NVT and NPT at 300 K, CHARMM36m
# Run from this relaxation/ directory.
# Output: ../system_equilibrated.data

{header}pair_style      {PAIR_STYLE}

{cmap_pre}read_data       system_ramped.data{cmap_rdat}
{grp_line}
neigh_modify    one 10000
{_charmm_pair_block(s)}{omega_fix}
thermo          100
{thermo}
# Tight minimisation
min_style       cg
minimize        1.0e-6 1.0e-8 100000 1000000
write_data      system_minimized_final.data
{shake}
# Short NVT (10 000 steps x 1 fs)
reset_timestep  0
variable        T equal 300
velocity        all create ${{T}} {velocity_seed}
timestep        1.0
fix             1 all nvt temp ${{T}} ${{T}} 100.0
run             10000
unfix           1
write_data      after_nvt.data

# Short NPT (10 000 steps x 1 fs)
fix             1 all npt temp ${{T}} ${{T}} 100.0 iso 1.0 1.0 1000.0
run             10000
unfix           1
{omega_unfix}
write_data      ../system_equilibrated.data
print "All stages complete."
""")
