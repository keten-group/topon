"""
Build an atomistic protein system from a BFM snapshot.

Key functions
-------------
build_protein_system(ff, snapshot, full_sequence, node_to_res, lattice_scale)
    Returns dry atoms, bonds, improper_defs, box, crosslink_bonds.

add_water_and_ions(atoms, bonds, box, water_content_pct, salt_conc_M)
    Solvates the system in-place.  Handles charge neutralisation automatically.

compute_lattice_scale(Nx, protein_mass_Da, water_content_pct, target_density)
    Returns the Å/BFM-unit scale for a given water content and target density.
"""

import numpy as np

from .charmm_ff import CHARMMForceField


# ── Atom data class ───────────────────────────────────────────────────────────

class Atom:
    __slots__ = ["idx", "name", "atype", "charge", "res_name",
                 "res_id", "chain_id", "pos", "mol_id"]

    def __init__(self, idx, name, atype, charge, res_name,
                 res_id, chain_id, pos, mol_id):
        self.idx = idx
        self.name = name
        self.atype = atype
        self.charge = charge
        self.res_name = res_name
        self.res_id = res_id
        self.chain_id = chain_id
        self.pos = np.asarray(pos, dtype=float)
        self.mol_id = mol_id


# ── Box / density helpers ─────────────────────────────────────────────────────

def compute_lattice_scale(Nx, protein_mass_Da, water_content_pct=0.0,
                          target_density=0.85):
    """
    Compute lattice_scale (Å per BFM unit) such that the simulation box
    accommodates protein + water at the specified target density.

    Parameters
    ----------
    Nx : int           Lattice edge length (cubic).
    protein_mass_Da : float  Total dry protein mass in Daltons.
    water_content_pct : float  Weight percent water (0–100).
    target_density : float  g/cm³.  Default 0.85 (slightly loose initial box).

    Returns
    -------
    float : lattice_scale in Å/BFM unit
    """
    total_mass = protein_mass_Da
    if water_content_pct > 0.0:
        water_mass = protein_mass_Da * water_content_pct / (100.0 - water_content_pct)
        total_mass += water_mass

    # V [Å³] = M [Da] × (1.66054 Å³/Da / target_density)
    V = total_mass * 1.66054 / target_density
    L_box = V ** (1.0 / 3.0)
    return L_box / Nx


def protein_mass(atoms, ff):
    """Protein mass in Da from the CHARMM masses of its atom types."""
    return float(sum(ff.masses[a.atype] for a in atoms))


def _estimate_protein_mass(atoms):
    """Rough protein mass estimation from atom types (Da).

    Kept for callers that have no force field at hand; the builders use
    :func:`protein_mass`, which reads the masses from the files.
    """
    mass = 0.0
    for a in atoms:
        t = a.atype
        if t.startswith("C"):
            mass += 12.0
        elif t.startswith("O"):
            mass += 16.0
        elif t.startswith("N"):
            mass += 14.0
        elif t.startswith("H"):
            mass += 1.0
        elif t.startswith("S"):
            mass += 32.0
        else:
            mass += 10.0
    return mass


# ── Main system builder ────────────────────────────────────────────────────────



_TARGET_CACA_ANG = 3.8   # real peptide CA-CA spacing; coil segments to ~this


def _perp_axis(v):
    """A unit vector perpendicular to v (fixed lab-axis Gram-Schmidt)."""
    nv = float(np.linalg.norm(v))
    vh = v / nv if nv > 1e-9 else np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0]) if abs(vh[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e = ref - np.dot(ref, vh) * vh
    n = float(np.linalg.norm(e))
    return e / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _coil_positions(p_start, diff, n_pts, target):
    """`n_pts` residue positions from p_start to p_start+diff.

    If the straight-line spacing (segment length / (n_pts-1)) would compress
    residues below `target`, the INTERIOR points zig-zag off the axis so their
    spacing is ~target -- decompressing the chain so minimisation needs no
    violent expansion (which is what scrambled omega/chirality). The two
    ENDPOINTS stay exactly on the nodes (envelope sin(pi k/(n-1)) = 0 there), so
    crosslinker Y residues -- always at lattice nodes -- are never moved and the
    a-priori crosslink geometry is preserved. If the straight span is already
    long enough, returns the plain linear interpolation.
    """
    if n_pts <= 1:
        return [p_start.copy()]
    L = float(np.linalg.norm(diff))
    axial = L / (n_pts - 1)
    if axial >= target or L < 1e-6:
        return [p_start + (k / (n_pts - 1)) * diff for k in range(n_pts)]
    e1 = _perp_axis(diff)
    amp = 0.5 * float(np.sqrt(max(target * target - axial * axial, 0.0)))
    out = []
    for k in range(n_pts):
        ax = p_start + (k / (n_pts - 1)) * diff
        env = float(np.sin(np.pi * k / (n_pts - 1)))    # 0 at both endpoints
        out.append(ax + ((-1) ** k) * amp * env * e1)
    return out


def _nerf(A, B, C, r, theta_deg, phi_deg):
    """Place a 4th atom from three placed ones (Natural Extension Reference
    Frame): bond C-D = r, angle B-C-D = theta, dihedral A-B-C-D = phi."""
    th = np.radians(theta_deg); ph = np.radians(phi_deg)
    bc = C - B; bc /= (np.linalg.norm(bc) + 1e-12)
    n = np.cross(B - A, bc); nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    m = np.cross(n, bc)
    return C + (-r * np.cos(th)) * bc + (r * np.sin(th) * np.cos(ph)) * m \
             + (r * np.sin(th) * np.sin(ph)) * n


def _build_physical_positions(residue_positions, full_sequence, n_residues,
                              ff, box, xpro_cis_fraction, bb_rng):
    """Physically correct per-residue atom positions from CHARMM internal
    coordinates. Two passes: (1) place backbone N/CA/C along the (coiled) CA
    trace at real bond lengths + ~111 deg N-CA-C, with a parallel-transported
    perpendicular so peptide omega starts TRANS (X-Pro bonds flipped cis at
    `xpro_cis_fraction`); (2) NeRF-build every remaining atom of each residue
    from its RTF IC table -> correct chirality (L), planar impropers, ideal
    sidechain geometry. Returns {(res_idx, atom_name): np.ndarray}.
    """
    def mi(v):
        return v - box * np.round(v / box)

    def tangent(i):
        ca = residue_positions
        if 0 < i < n_residues - 1 and (i - 1) in ca and (i + 1) in ca:
            t = mi(ca[i + 1] - ca[i - 1])
        elif (i + 1) in ca:
            t = mi(ca[i + 1] - ca[i])
        elif (i - 1) in ca:
            t = mi(ca[i] - ca[i - 1])
        else:
            t = np.array([1.0, 0.0, 0.0])
        n = float(np.linalg.norm(t))
        return t / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    # parallel-transported perpendicular per residue
    perps = {}
    pv = None
    for i in range(n_residues):
        if i not in residue_positions:
            continue
        t = tangent(i)
        if pv is None:
            ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            p = ref - np.dot(ref, t) * t
        else:
            p = pv - np.dot(pv, t) * t
            if float(np.linalg.norm(p)) < 1e-3:
                ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                p = ref - np.dot(ref, t) * t
        p /= (float(np.linalg.norm(p)) + 1e-12)
        perps[i] = p
        pv = p

    # N and C sit on the SAME perpendicular side of the tangent so the
    # N-CA-C angle is ~111 deg (opposite sides would make them collinear and
    # leave chirality undefined). This makes the peptide omega start ~cis; the
    # writer's stage-1 omega restraint then drives every bond to trans (and a
    # ~5% X-Pro subset to cis). Chirality is fixed geometrically below.
    beta = np.radians(34.5)   # half of (180 - 111)
    cb, sb = np.cos(beta), np.sin(beta)
    bb = {}
    for i in range(n_residues):
        if i not in residue_positions:
            continue
        ca = residue_positions[i]; t = tangent(i); p = perps[i]
        bb[(i, "CA")] = ca
        bb[(i, "C")] = ca + 1.52 * (cb * t + sb * p)
        bb[(i, "N")] = ca + 1.46 * (-cb * t + sb * p)

    phys = {}
    for i in range(n_residues):
        if (i, "N") not in bb:
            continue
        rn = full_sequence[i]
        if rn == "HIS":
            rn = "HSD"
        tmpl = ff.residues.get(rn)
        if not tmpl:
            continue
        known = {"N": bb[(i, "N")], "CA": bb[(i, "CA")], "C": bb[(i, "C")]}
        if (i - 1, "C") in bb:
            known["-C"] = bb[(i - 1, "C")]
        if (i + 1, "N") in bb:
            known["+N"] = bb[(i + 1, "N")]
        ics = tmpl.get("ics", [])
        for _ in range(25):
            progressed = False
            for ic in ics:
                a1, a2, a3, a4 = ic["atoms"]
                if a4 in known or a4.startswith(("+", "-")):
                    continue
                if a1 in known and a2 in known and a3 in known:
                    known[a4] = _nerf(known[a1], known[a2], known[a3],
                                      ic["r34"], ic["a234"], ic["d1234"])
                    progressed = True
            if not progressed:
                break

        # Enforce L-chirality deterministically. The lattice trace has hairpins
        # where the backbone frame flips, so IC-built CB can land on the D side.
        # Reflecting the sidechain (everything except the backbone N/CA/C/O/HN)
        # across the N-CA-C plane preserves every bond length and angle but
        # flips the CA stereocentre D -> L. (Peptide-plane atoms O/HN are kept.)
        if "CB" in known and "N" in known and "CA" in known and "C" in known:
            ca = known["CA"]
            vN = known["N"] - ca; vC = known["C"] - ca
            nrm = np.cross(vN, vC); ln = float(np.linalg.norm(nrm))
            if ln > 1e-9:
                nrm = nrm / ln
                if np.dot(nrm, known["CB"] - ca) < 0.0:   # D -> reflect sidechain
                    for nm in list(known):
                        if nm in ("N", "CA", "C", "O", "HN") or nm.startswith(("+", "-")):
                            continue
                        known[nm] = known[nm] - 2.0 * np.dot(known[nm] - ca, nrm) * nrm

        for name, pos in known.items():
            if not name.startswith(("+", "-")):
                phys[(i, name)] = pos
    return phys


#: Crosslinking residue (3-letter) -> the RTF patch that joins two of them.
CROSSLINK_PATCHES = {"TYR": "DITY", "CYS": "DISU"}


def build_protein_system(ff, snapshot, full_sequence, node_to_res,
                         lattice_scale=15.0, physical_backbone=False,
                         xpro_cis_fraction=0.0, backbone_seed=20260714,
                         crosslink_patch=None):
    """
    Build the full atomistic (dry) protein system from a BFM topology snapshot.

    Parameters
    ----------
    ff : CHARMMForceField
    snapshot : dict
        Single snapshot from a topology file.
        Must have: chains, Nx, Ny, Nz, crosslinker_positions, reactions.
    full_sequence : list of str
        3-letter residue names for the whole chain
        (length = n_residues = len(node_to_res possible residues)).
    node_to_res : dict
        chain_node_index → residue_index mapping (from sequence.get_node_residue_mapping).
    lattice_scale : float
        Å per BFM lattice unit.  Default 15.0.
    crosslink_patch : str or None
        RTF patch applied to each crosslinked residue pair (``DITY`` for
        dityrosine, ``DISU`` for a disulfide). ``None`` picks it from the
        residue at the first reaction (see ``CROSSLINK_PATCHES``).

    Every crosslink recorded in ``snapshot["reactions"]`` becomes one patch
    application: the patch's ``1``/``2`` atoms take its types and charges,
    its deleted atoms go, and its bond is added. A residue can be patched
    once; a second reaction on it is an error.

    Returns
    -------
    atoms, bonds, improper_defs, box, crosslink_bonds
    """
    chains = snapshot["chains"]
    Nx, Ny, Nz = snapshot["Nx"], snapshot["Ny"], snapshot["Nz"]
    n_residues = len(full_sequence)

    box = np.array([Nx, Ny, Nz], dtype=float) * lattice_scale

    def lattice_pos(flat_idx):
        z = flat_idx // (Nx * Ny)
        rem = flat_idx % (Nx * Ny)
        y = rem // Nx
        x = rem % Nx
        return np.array([x, y, z], dtype=float) * lattice_scale

    atoms = []
    bonds = []
    improper_defs = []
    crosslink_bonds = []

    atom_counter = 0
    global_res_counter = 0
    # Deterministic RNG for the optional X-Pro cis seeding (0.0 -> all trans).
    bb_rng = np.random.default_rng(backbone_seed)

    # (chain_id, residue_idx, atom_name) -> global atom idx, for patching
    residue_atoms = {}

    for chain_id, chain_nodes in enumerate(chains):
        n_chain_nodes = len(chain_nodes)

        # Build lattice positions for each node
        node_positions = {ni: lattice_pos(fi) for ni, fi in enumerate(chain_nodes)}

        # Interpolate residue coordinates along the chain
        residue_positions = {}
        sorted_nodes = sorted(node_to_res.keys())
        for seg_i in range(len(sorted_nodes) - 1):
            ni_start = sorted_nodes[seg_i]
            ni_end = sorted_nodes[seg_i + 1]

            if ni_start >= n_chain_nodes or ni_end >= n_chain_nodes:
                continue

            r_start = node_to_res[ni_start]
            r_end = node_to_res[ni_end]
            p_start = node_positions[ni_start]
            p_end = node_positions[ni_end]

            # Minimum image displacement
            diff = p_end - p_start
            diff -= box * np.round(diff / box)

            n_seg_res = r_end - r_start + 1
            if physical_backbone:
                seg_pts = _coil_positions(p_start, diff, n_seg_res, _TARGET_CACA_ANG)
                for k, r in enumerate(range(r_start, r_end + 1)):
                    if r not in residue_positions:
                        residue_positions[r] = seg_pts[k]
            else:
                for k, r in enumerate(range(r_start, r_end + 1)):
                    if r not in residue_positions:
                        frac = k / max(n_seg_res - 1, 1)
                        residue_positions[r] = p_start + frac * diff

        # Physically correct positions (IC/NeRF build) for the whole chain,
        # computed once so every residue's sidechain can reference its
        # neighbours' backbone (-C / +N). Falls back to per-atom jitter for any
        # atom without an IC (e.g. terminal-patch atoms).
        phys_pos = (_build_physical_positions(
                        residue_positions, full_sequence, n_residues, ff, box,
                        xpro_cis_fraction, bb_rng)
                    if physical_backbone else None)

        # Instantiate atoms residue by residue
        chain_atom_ids = {}   # (res_idx, atom_name) → global_atom_idx
        pending_impr = []     # (res_idx, improper names), resolved per chain
        prev_c_idx = None

        for res_idx in range(n_residues):
            res_name = full_sequence[res_idx]
            # Histidine: CHARMM RTF defines protonation-state-specific
            # residues (HSD/HSE/HSP), never a bare "HIS". Map the
            # one-letter 'H' (emitted as "HIS") to the neutral HSD
            # tautomer (the CHARMM default). Without this, HIS misses
            # the lookup below.
            if res_name == "HIS":
                res_name = "HSD"
            res_tmpl = ff.residues.get(res_name)
            if not res_tmpl:
                # Silently skipping a residue corrupts the chain — the
                # peptide-bond logic then fuses the neighbours across the
                # gap. Fail loudly so an unmapped residue can never
                # masquerade as a valid build.
                raise ValueError(
                    f"No CHARMM RTF template for residue {res_name!r} "
                    f"(chain {chain_id}, position {res_idx}). Map it to a "
                    f"supported residue/protonation variant before building."
                )

            global_res_counter += 1
            center = residue_positions.get(res_idx, np.zeros(3))

            atom_list = dict(res_tmpl["atoms"])
            bond_list = list(res_tmpl["bonds"])
            impr_list = list(res_tmpl.get("impropers", []))
            deletes = []

            # Terminal patches. The N-terminus needs a residue-specific
            # patch: GLYP for glycine (two HA), PROP for proline (secondary
            # amine in a ring -> keeps CA=CP1, N=NP, CD=CP3 and adds only
            # two N-H), and the generic NTER (NH3+) otherwise. Applying the
            # generic NTER to a proline mis-types CA as CT1 and N as NH3,
            # which produces CHARMM-nonexistent angles/dihedrals (CT1-CP2,
            # NH3-CT1, ...) that fall through to the writer's K=0 DEFAULT and
            # also corrupts the residue's net charge (non-integer total).
            if res_idx == 0:
                if res_name == "GLY":
                    patch_name = "GLYP"
                elif res_name == "PRO":
                    patch_name = "PROP"
                else:
                    patch_name = "NTER"
                _apply_patch(ff, patch_name, atom_list, bond_list, impr_list, deletes)

            if res_idx == n_residues - 1:
                _apply_patch(ff, "CTER", atom_list, bond_list, impr_list, deletes)

            for d in deletes:
                atom_list.pop(d, None)

            # physical_backbone: seed backbone geometry so peptide bonds start
            # TRANS and CB sits on the L-chirality side (the default random
            # jitter falls ~50/50 into cis/trans and D/L, which the omega/
            # chirality barriers then freeze). Only backbone/CB atoms are placed
            # deterministically; sidechain atoms keep the jitter. When
            # physical_backbone is False the placement is exactly the original
            # (CA at anchor, everything else jittered) -- byte-for-byte.
            for atom_name, (atype, charge) in atom_list.items():
                if atom_name.startswith("+") or atom_name.startswith("-"):
                    continue
                atom_counter += 1
                if phys_pos is not None and (res_idx, atom_name) in phys_pos:
                    pos = phys_pos[(res_idx, atom_name)]     # IC-built physical position
                else:
                    offset = np.zeros(3)
                    if atom_name != "CA":
                        rng = np.random.default_rng(seed=atom_counter)
                        offset = rng.uniform(-0.3, 0.3, 3)
                    pos = center + offset

                a = Atom(
                    idx=atom_counter,
                    name=atom_name,
                    atype=atype,
                    charge=charge,
                    res_name=res_name,
                    res_id=global_res_counter,
                    chain_id=chain_id,
                    pos=pos,
                    mol_id=chain_id + 1,
                )
                atoms.append(a)
                chain_atom_ids[(res_idx, atom_name)] = atom_counter

            # Intra-residue bonds
            for a1, a2 in bond_list:
                if a1.startswith("+") or a1.startswith("-"):
                    continue
                if a2.startswith("+") or a2.startswith("-"):
                    continue
                id1 = chain_atom_ids.get((res_idx, a1))
                id2 = chain_atom_ids.get((res_idx, a2))
                if id1 and id2:
                    bonds.append((id1, id2))

            # Impropers. Those naming the previous or next residue (the
            # peptide-plane pair N -C CA HN and C CA +N O) are resolved once
            # the whole chain exists; an improper whose atoms are not all
            # present (a terminus, a deleted HN) is not generated, as in
            # CHARMM's patching.
            for quad in impr_list:
                pending_impr.append((res_idx, tuple(quad)))

            # Peptide bond to previous residue
            if res_idx > 0 and prev_c_idx is not None:
                n_idx = chain_atom_ids.get((res_idx, "N"))
                if n_idx:
                    bonds.append((prev_c_idx, n_idx))

            prev_c_idx = chain_atom_ids.get((res_idx, "C"))

        for res_idx, quad in pending_impr:
            ids = []
            for n in quad:
                r = res_idx - 1 if n.startswith("-") else res_idx + 1 if n.startswith("+") else res_idx
                ids.append(chain_atom_ids.get((r, n.lstrip("+-"))))
            if all(ids):
                improper_defs.append(tuple(ids))
        for (r, n), idx in chain_atom_ids.items():
            residue_atoms[(chain_id, r, n)] = idx

    # ── Apply crosslinks ──────────────────────────────────────────────────────
    by_idx = {a.idx: a for a in atoms}
    reactions = snapshot.get("reactions", [])
    patched = set()
    atoms_to_remove = set()
    for rxn in reactions:
        (ci1, ni1), (ci2, ni2) = rxn[0], rxn[1]
        sides = []
        for ci, ni in ((ci1, ni1), (ci2, ni2)):
            r = node_to_res.get(ni)
            if r is None:
                raise ValueError(f"reaction node {ni} of chain {ci} has no residue")
            if (ci, r) in patched:
                raise ValueError(f"residue {r} of chain {ci} is in two crosslinks")
            sides.append((ci, r))
        resnames = {full_sequence[r] for _, r in sides}
        patch_name = crosslink_patch
        if patch_name is None:
            if len(resnames) != 1 or next(iter(resnames)) not in CROSSLINK_PATCHES:
                raise ValueError(f"no crosslink patch for residues {sorted(resnames)}; "
                                 f"known: {CROSSLINK_PATCHES}")
            patch_name = CROSSLINK_PATCHES[next(iter(resnames))]
        new_bonds, dead = _apply_pair_patch(ff, patch_name, sides, residue_atoms, by_idx)
        bonds.extend(new_bonds)
        crosslink_bonds.extend(new_bonds)
        atoms_to_remove |= dead
        patched.update(sides)

    if atoms_to_remove:
        atoms = [a for a in atoms if a.idx not in atoms_to_remove]
        bonds = [(a, b) for a, b in bonds
                 if a not in atoms_to_remove and b not in atoms_to_remove]
        improper_defs = [q for q in improper_defs if not (set(q) & atoms_to_remove)]

    return atoms, bonds, improper_defs, box, crosslink_bonds


def _apply_pair_patch(ff, patch_name, sides, residue_atoms, by_idx):
    """Apply a two-residue RTF patch (``1X``/``2X`` atom names) in place.

    Retypes and recharges the patch's atoms, returns the bonds it adds and
    the ids of the atoms it deletes. Any patch atom missing from its residue
    is an error, so a patch can never apply half-way.
    """
    patch = ff.patches.get(patch_name)
    if not patch:
        raise KeyError(f"patch {patch_name!r} is not in the RTF")
    skipped = [k for k in ("impropers", "cmaps") if patch.get(k)]
    if skipped:
        # this step applies atoms, deletes and bonds only; anything else in
        # the patch would be lost without a word
        raise ValueError(f"patch {patch_name} lists {' and '.join(skipped)}, which a "
                         f"two-residue crosslink patch does not apply")

    def atom_id(tagged):
        side, name = tagged[:1], tagged[1:]
        if side not in "12" or not name:
            raise ValueError(f"patch {patch_name}: atom {tagged!r} is not 1X or 2X")
        ci, r = sides[int(side) - 1]
        idx = residue_atoms.get((ci, r, name))
        if idx is None:
            raise ValueError(f"patch {patch_name}: residue {r} of chain {ci} has no "
                             f"atom {name}")
        return idx

    for tagged, (atype, charge) in patch["atoms"].items():
        a = by_idx[atom_id(tagged)]
        a.atype, a.charge = atype, charge
    dead = {atom_id(t) for t in patch["deletes"]}
    new_bonds = [(atom_id(a), atom_id(b)) for a, b in patch["bonds"]]
    return new_bonds, dead


def _apply_patch(ff, patch_name, atom_list, bond_list, impr_list, deletes):
    patch = ff.patches.get(patch_name, {})
    if not patch:
        return
    if patch.get("cmaps"):
        # CMAP terms come from the backbone (find_cmap_crossterms), not patches
        raise ValueError(f"patch {patch_name} lists CMAP terms, which terminal patches "
                         f"do not apply")
    for d in patch.get("deletes", []):
        deletes.append(d)
    for aname, (atype, charge) in patch.get("atoms", {}).items():
        atom_list[aname] = (atype, charge)
    bond_list.extend(patch.get("bonds", []))
    impr_list.extend(patch.get("impropers", []))


# ── Solvation ─────────────────────────────────────────────────────────────────

def add_water_and_ions(atoms, bonds, box, water_content_pct=35.0,
                       salt_conc_M=0.15, cation_type="SOD", anion_type="CLA",
                       ff=None, seed=0, min_dist=2.4, verbose=True):
    """
    Add TIP3P water and NaCl ions to the system (in-place).

    Water amount   -> weight percent relative to the protein mass.
    NaCl (background) -> ``salt_conc_M`` in the water volume.
    Neutralisation -> extra cations or anions to zero the net charge.

    Molecules go on free sites of a cubic grid that stay at least
    ``min_dist`` from every protein atom (minimum image), chosen and oriented
    with ``numpy.random.default_rng(seed)``, so a build is reproducible and
    starts without water inside the protein. With ``ff`` the protein mass
    comes from the CHARMM masses; without it, from a rough element estimate.

    Returns ``(n_waters, n_cations, n_anions)``.
    """
    from scipy.spatial import cKDTree

    box = np.asarray(box, dtype=float)
    rng = np.random.default_rng(seed)
    mass = protein_mass(atoms, ff) if ff is not None else _estimate_protein_mass(atoms)
    raw_charge = sum(a.charge for a in atoms)
    net_charge = int(round(raw_charge))
    # A correct build (complete residues, correct patches) sums to an integer.
    if abs(raw_charge - net_charge) > 1e-3:
        raise ValueError(
            f"Protein net charge {raw_charge:+.4f} e is not an integer; an atom "
            f"type or patch is wrong upstream. Refusing to neutralise by rounding."
        )

    num_waters = 0
    if water_content_pct > 0.0:
        water_mass = mass * water_content_pct / (100.0 - water_content_pct)
        num_waters = int(water_mass / 18.015)

    # Background NaCl: n = C [mol/L] x N_A x V_water [L], V from 1 g/cm^3.
    n_nacl = 0
    if num_waters > 0 and salt_conc_M > 0.0:
        v_water_l = num_waters * 18.015 / (6.022e23 * 1000.0)
        n_nacl = int(round(salt_conc_M * 6.022e23 * v_water_l))
    n_cations = n_nacl + max(0, -net_charge)
    n_anions = n_nacl + max(0, net_charge)
    n_place = num_waters + n_cations + n_anions

    if verbose:
        print("\n  [Solvation]")
        print(f"    Protein mass : {mass:.0f} Da | net charge = {net_charge:+d} e")
        print(f"    Water        : {num_waters} molecules ({water_content_pct:.0f} wt%)")
        print(f"    NaCl (bg)    : {n_nacl} pairs at {salt_conc_M:.3f} M")
        print(f"    Cations ({cation_type}) : {n_cations}")
        print(f"    Anions  ({anion_type})  : {n_anions}")
    if n_place == 0:
        return 0, 0, 0

    # Free grid sites: spacing shrinks until there are enough of them.
    prot = np.array([a.pos for a in atoms], dtype=float) % box
    prot[prot >= box] = 0.0          # x % L can round up to exactly L
    tree = cKDTree(prot, boxsize=box) if len(prot) else None
    spacing = 3.1
    for _ in range(12):
        n_ax = np.maximum(np.floor(box / spacing).astype(int), 1)
        g = [(np.arange(n) + 0.5) * (L / n) for n, L in zip(n_ax, box)]
        sites = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
        if tree is not None:
            d, _ = tree.query(sites, k=1)
            sites = sites[d >= min_dist]
        if len(sites) >= n_place:
            break
        spacing *= 0.92
    else:
        raise ValueError(
            f"only {len(sites)} free sites for {n_place} waters and ions; lower "
            f"the target density or the water content"
        )
    chosen = sites[rng.choice(len(sites), size=n_place, replace=False)]

    start_idx = max(a.idx for a in atoms) + 1 if atoms else 1
    max_mol = max(a.mol_id for a in atoms) if atoms else 0
    k = 0
    for count, typ, q in ((n_cations, cation_type, 1.0), (n_anions, anion_type, -1.0)):
        for _ in range(count):
            max_mol += 1
            atoms.append(Atom(start_idx, typ, typ, q, typ, max_mol, max_mol,
                              chosen[k], max_mol))
            start_idx += 1
            k += 1

    # TIP3P (O-H 0.9572 A, H-O-H 104.52 deg), randomly oriented.
    th = np.radians(104.52)
    local = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0],
                      [0.9572 * np.cos(th), 0.9572 * np.sin(th), 0.0]])
    for _ in range(num_waters):
        q_ = rng.normal(size=4)
        w, x, y, z = q_ / np.linalg.norm(q_)
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        xyz = chosen[k] + local @ rot.T
        k += 1
        max_mol += 1
        iO, iH1, iH2 = start_idx, start_idx + 1, start_idx + 2
        atoms.append(Atom(iO, "OH2", "OT", -0.834, "TIP3", max_mol, max_mol, xyz[0], max_mol))
        atoms.append(Atom(iH1, "H1", "HT", 0.417, "TIP3", max_mol, max_mol, xyz[1], max_mol))
        atoms.append(Atom(iH2, "H2", "HT", 0.417, "TIP3", max_mol, max_mol, xyz[2], max_mol))
        bonds.append((iO, iH1))
        bonds.append((iO, iH2))
        start_idx += 3
    return num_waters, n_cations, n_anions


# ── CMAP backbone crossterms ──────────────────────────────────────────────────

def find_cmap_crossterms(atoms):
    """
    Identify backbone CMAP crossterms for all non-terminal protein residues.

    For each internal residue i in a chain, the crossterm spans five backbone
    atoms: C(i-1) - N(i) - CA(i) - C(i) - N(i+1), defining the phi/psi pair.

    CMAP type (matching charmm36m.cmap / charmm36.cmap):
        1 = regular residue
        2 = regular residue before PRO
        3 = PRO
        4 = PRO before PRO
        5 = GLY
        6 = GLY before PRO

    Water, ion, and other non-protein atoms (identified by atype) are skipped.

    Returns
    -------
    list of (cmap_type, c_prev_idx, n_i_idx, ca_i_idx, c_i_idx, n_next_idx)
        All indices are original atom .idx values (pre-renumbering).
    """
    _SKIP_ATYPES = {"OT", "HT", "SOD", "CLA", "CAL", "ZN", "FE3P"}

    from collections import defaultdict

    # Group atoms by (chain_id, res_id) — protein only
    # chain_res[chain_id][res_id] = {atom_name: atom_idx}
    # chain_resname[chain_id][res_id] = res_name
    chain_res     = defaultdict(lambda: defaultdict(dict))
    chain_resname = defaultdict(dict)

    for a in atoms:
        if a.atype in _SKIP_ATYPES:
            continue
        chain_res[a.chain_id][a.res_id][a.name] = a.idx
        chain_resname[a.chain_id][a.res_id] = a.res_name

    crossterms = []

    for chain_id in sorted(chain_res.keys()):
        res_ids = sorted(chain_res[chain_id].keys())
        n_res   = len(res_ids)

        # Skip terminal residues (need i-1 and i+1)
        for pos in range(1, n_res - 1):
            prev_rid = res_ids[pos - 1]
            curr_rid = res_ids[pos]
            next_rid = res_ids[pos + 1]

            c_prev = chain_res[chain_id][prev_rid].get("C")
            n_i    = chain_res[chain_id][curr_rid].get("N")
            ca_i   = chain_res[chain_id][curr_rid].get("CA")
            c_i    = chain_res[chain_id][curr_rid].get("C")
            n_next = chain_res[chain_id][next_rid].get("N")

            if None in (c_prev, n_i, ca_i, c_i, n_next):
                continue

            curr_name = chain_resname[chain_id][curr_rid]
            next_name = chain_resname[chain_id][next_rid]

            before_pro = (next_name == "PRO")
            if curr_name == "GLY":
                cmap_type = 6 if before_pro else 5
            elif curr_name == "PRO":
                cmap_type = 4 if before_pro else 3
            else:
                cmap_type = 2 if before_pro else 1

            crossterms.append((cmap_type, c_prev, n_i, ca_i, c_i, n_next))

    return crossterms
