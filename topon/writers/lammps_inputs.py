import copy
import os
import json

from .lammps_endlinked import JUNCTION as ENDLINKED_JUNCTION

#: Coarse-grained relaxation protocols, selected with ``simulation.protocol``.
#:
#: ``pushoff`` (default)
#:     FENE + WCA from the first step with a capped displacement. No
#:     minimiser, no soft potential, no harmonic bond, so nothing can push a
#:     bead through a bond and the entanglement state the build carries is
#:     the one the run ends with.
#: ``hardcore_min``
#:     The older hard-core minimiser: WCA throughout, conjugate-gradient
#:     minimisation at stage 1, FENE from stage 3. Crossing-prone -- its
#:     stage 1 stretched 85 bonds to 1.70 sigma on the N20 build and left 57
#:     threaded at 1.3-1.4 sigma, which let strands cross later. Kept so
#:     earlier runs can be reproduced.
#: ``soft_push``
#:     The historic generated protocol: ``pair_style soft`` ramped 0 to 30,
#:     then an epsilon ramp to the real potential. Softer still -- a soft
#:     core has finite energy at zero separation, so chains pass straight
#:     through each other. Every CG reference under ``tests/output/`` was
#:     written with it.
CG_PROTOCOLS = ("pushoff", "hardcore_min", "soft_push")

#: ``(script, data file)`` for each stage of the push-off protocol, in run
#: order. The data-file names are the interchange convention of the
#: end-linked validation scripts (``refnet.parse``, ``measure_system.py``),
#: so a topon run and a reference run are measured by the same tooling.
#: ``stage1_min.data`` is historic: stage 1 no longer minimises anything.
PUSHOFF_STAGES = (
    ("minimize_1_serial.in", "stage1_min.data"),
    ("minimize_2_parallel.in", "stage2_pushoff.data"),
    ("minimize_3_parallel.in", "stage3_build_equil.data"),
    ("deform_4_parallel.in", "stage4_final_T1.data"),
    ("quench_5_parallel.in", "stage5_final_quench.data"),
)

#: ``(script, data file)`` for each stage of the two minimiser protocols.
MINIMISER_STAGES = (
    ("minimize_1_serial.in", "system_after_soft.data"),
    ("minimize_2_parallel.in", "system_ramped.data"),
    ("minimize_3_parallel.in", "system_equilibrated.data"),
)

#: Push-off parameters. Every number here is from the protocol's
#: specification and was measured on the N20/N100 builds: the two capped stages
#: are what keep the bond histogram under 1.2 sigma while the overlaps
#: resolve. Override any of them under ``experimental.cg.pushoff``.
PUSHOFF_DEFAULTS = {
    "seed": 12345,
    "temperature": 1.0,
    "quench_temperature": 0.4,
    # FENE: K, R0, epsilon, sigma. R0 = 1.5 is also the bond length at which
    # the potential diverges, which is why FENE errors out on a stretched
    # bond where the reference's quartic bond would break it silently.
    "bond": [30.0, 1.5, 1.0, 1.0],
    "wca_cutoff": 1.122462,
    "neighbor_skin": 1.0,
    # Ghost-atom communication has to reach a stretched FENE bond (up to 1.5
    # sigma) plus the neighbour skin, or a bond partner goes missing across a
    # processor boundary. 3.0 covers it with room to spare.
    "comm_cutoff": 3.0,
    "thermo_freq": 1000,
    "stage1": {"timestep": 0.002, "limit": 0.02, "steps": 30000, "tdamp": 1.0},
    "stage2": {"timestep": 0.005, "limit": 0.05, "steps": 20000,
               "free_steps": 20000, "tdamp": 1.0},
    "stage3": {"steps": 200000, "tdamp": 10.0},
    "stage4": {"deform_steps": 150000, "settle_steps": 100000},
    "stage5": {"ramp_steps": 50000, "settle_steps": 20000, "tdamp": 10.0},
    # Stage 6 exists only when simulation.final_bond_style is "quartic".
    "stage6": {"steps": 20000},
}


#: Atomistic relaxation protocols, selected with ``simulation.atomistic_protocol``.
#:
#: ``soft_push`` (default)
#:     The historic DREIDING deck. Stage 1 runs ``pair_style soft`` with a 1 A
#:     cutoff for every pair and minimises; stage 2 ramps every Lennard-Jones
#:     well from 0.001 of its depth under ``nve/limit`` with no thermostat.
#:     Backbones can pass through each other until the ramp ends: on a DP-10
#:     test network 4 of the build's 6 Z1+ partner pairs were gone after
#:     stage 1, and the ramp ended at 964 K.
#: ``hard_backbone``
#:     The backbone never goes soft, the atomistic counterpart of the
#:     bead-spring push-off. Stage 1 gives every pair of backbone atom types
#:     a fixed soft core (``core`` kcal/mol out to ``core_cutoff`` A, about
#:     200 kcal/mol in the way of a backbone atom passing through another
#:     strand's bond) while pairs with a light atom (methyl C, H) ramp as in
#:     the historic deck; stage 2 holds the backbone pairs at full depth and
#:     ramps only the light ones. Both stages are capped displacement under a
#:     Langevin thermostat, with no minimiser; stage 3 is the historic one. On
#:     the smoke network it kept 4 of the 6 build pairs through stage 1 and
#:     held the ramp at 300 K. The backbone types come from the pipeline (the
#:     types of the atoms on the strand record's backbones and junctions).
ATOMISTIC_PROTOCOLS = ("soft_push", "hard_backbone")

#: ``hard_backbone`` parameters; override any of them under
#: ``experimental.atomistic.hard_backbone``. Stage 2's length is the deck's
#: ``experimental.atomistic.dynamics.run_steps``, as for the historic ramp.
HARD_BACKBONE_DEFAULTS = {
    "core": 60.0,           # kcal/mol, soft-core A of backbone-backbone pairs
    "core_cutoff": 3.0,     # A
    "light_cutoff": 1.0,    # A, soft cutoff of every pair with a light atom
    "light_max": 30.0,      # kcal/mol, where the light ramp ends
    "temperature": 300.0,   # K
    "tdamp": 100.0,         # fs
    # Stage lengths: stage 1's pair energy has levelled
    # by 4,000 steps and the ramp is the epsilon ramp itself; at half the
    # historic lengths and a capped stage-3 minimisation the DP-30 runs kept
    # the same Z, energy and temperatures with no passage, in 7-8 minutes
    # against 17.5. Halved again for the ramp and the minimisation (since
    # 0.4.5): on the same two DP-30 builds Z, density and passages
    # came out as at the 0.4.0 lengths (Z after NVT 0.998 against 1.000, 1.943
    # against 1.912; no passage), with the pair energy after NVT about 1 %
    # higher, for about 40 % less LAMMPS time.
    "stage1": {"steps": 5000, "limit": 0.05, "velocity_seed": 4928459,
               "langevin_seed": 48279},
    "stage2": {"steps": 2500, "limit": 0.1, "langevin_seed": 7811},
    "stage3": {"minimize": "1.0e-6 1.0e-8 1000 10000"},
}

#: Stage 3's minimisation on the historic decks ("etol ftol maxiter maxeval").
HISTORIC_MINIMIZE = {"dreiding": "1.0e-8 1.0e-10 10000000 100000000",
                     "charmm": "1.0e-6 1.0e-8 100000 1000000"}


def light_pair_terms(pair, param, light, n_types, var):
    """``fix adapt`` terms that reach every type pair with a light type in it.

    LAMMPS stores pair coefficients for ``i <= j``, and a ``pair`` term with
    type ranges sets only those; so each light type ``t`` needs ``1*t t``
    (partners at or below it) and ``t t+1*n`` (above it).
    """
    terms = []
    for t in sorted(light):
        terms.append(f"pair {pair} {param} 1*{t} {t} v_{var}")
        if t < n_types:
            terms.append(f"pair {pair} {param} {t} {t + 1}*{n_types} v_{var}")
    return " ".join(terms)


def _merge(base, override):
    """Deep-merge ``override`` into a copy of ``base`` (dicts only)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class LammpsInputGenerator:
    """Writes the LAMMPS scripts that relax a build.

    Args:
        output_dir: the parent of the study directory.
        study_name: appended to ``output_dir`` to give the study root.
        config: the ``simulation`` block -- ``protocol``, ``pair_style``,
            ``include_angles``, ``remove_cg_angles``, ``rho_final``,
            ``final_bond_style``, and ``atomistic_protocol`` for the
            atomistic route.
        experimental: the ``experimental`` block; step counts for the
            push-off live under ``cg.pushoff``, the hard-backbone parameters
            under ``atomistic.hard_backbone``.
        flat: put the scripts, the data file and the checkpoints in one
            directory instead of the ``02_Chemistry`` / ``03_Conformation`` /
            ``04_Simulation`` layout. For a bead-spring build that never went
            through the chemistry stage, where those directories would each
            hold one file.
    """

    def __init__(self, output_dir, study_name, config=None, experimental=None,
                 flat=False):
        self.root_dir = os.path.join(output_dir, study_name)
        self.config = config or {}
        self.experimental = experimental or {}
        self.flat = flat
        if flat:
            self.sim_dir = self.chem_dir = self.conf_dir = self.root_dir
        else:
            self.sim_dir = os.path.join(self.root_dir, "04_Simulation")
            self.chem_dir = os.path.join(self.root_dir, "02_Chemistry")
            self.conf_dir = os.path.join(self.root_dir, "03_Conformation")

        self.protocol = self.config.get('protocol', 'pushoff')
        if self.protocol not in CG_PROTOCOLS:
            raise ValueError(
                f"Unknown CG relaxation protocol {self.protocol!r} "
                f"(expected one of {', '.join(CG_PROTOCOLS)})"
            )
        self.atomistic_protocol = self.config.get('atomistic_protocol', 'soft_push')
        if self.atomistic_protocol not in ATOMISTIC_PROTOCOLS:
            raise ValueError(
                f"Unknown atomistic relaxation protocol {self.atomistic_protocol!r} "
                f"(expected one of {', '.join(ATOMISTIC_PROTOCOLS)})"
            )
        # Atomistic stages only: dump the backbone atom types every N steps
        # (minimiser iterations included), for topon.analysis.crossings.
        self.backbone_dump_every = int(self.config.get('backbone_dump_every', 0) or 0)

        # Determine if in test mode
        self.test_mode = self.experimental.get('test_mode', False)

        if not os.path.exists(self.sim_dir):
            os.makedirs(self.sim_dir)

    # ------------------------------------------------------------------
    # Protocol parameters
    # ------------------------------------------------------------------

    def _pushoff(self):
        """Push-off parameters: the defaults under ``experimental.cg.pushoff``."""
        return _merge(PUSHOFF_DEFAULTS,
                      self.experimental.get('cg', {}).get('pushoff', {}))

    def _dump_open(self, tag, backbone_types, append=False):
        """Commands that dump the backbone atoms every N steps, or nothing.

        ``simulation.backbone_dump_every`` (atomistic only) writes
        ``traj_<tag>.lammpstrj`` with unwrapped coordinates, which stay
        continuous in time for every atom, so
        :mod:`topon.analysis.crossings` can follow each backbone bond from one
        frame to the next and see two of them pass through each other.
        """
        n = self.backbone_dump_every
        if not n:
            return ""
        if not backbone_types:
            raise ValueError("simulation.backbone_dump_every needs the backbone "
                             "atom types; the pipeline passes them")
        types = " ".join(str(int(t)) for t in sorted(set(backbone_types)))
        # LAMMPS will not reset the timestep under an active dump, so a stage
        # that resets it closes the dump first and reopens it here, appending.
        group = "" if append else f"group           bbdump type {types}\n"
        more = " append yes" if append else ""
        return (f"{group}"
                f"dump            bbdump bbdump custom {n} traj_{tag}.lammpstrj id xu yu zu\n"
                f"dump_modify     bbdump sort id format float %.4f{more}\n\n")

    def _dump_close(self):
        return "undump          bbdump\n" if self.backbone_dump_every else ""

    def _hard_backbone(self):
        """Hard-backbone parameters, under ``experimental.atomistic.hard_backbone``."""
        return _merge(HARD_BACKBONE_DEFAULTS,
                      self.experimental.get('atomistic', {}).get('hard_backbone', {}))

    def _hard_ramp_steps(self, s2) -> int:
        """The hard-backbone ramp's length: ``experimental.atomistic.dynamics.run_steps``
        when it is given (the historic knob for the ramp), else the deck's own."""
        given = self.experimental.get('atomistic', {}).get('dynamics', {}).get('run_steps')
        return int(given if given is not None else s2["steps"])

    def _stage3_minimize(self, force_field: str) -> str:
        """Stage 3's minimisation, "etol ftol maxiter maxeval".

        ``experimental.atomistic.stage3.minimize`` when given; else capped on
        the hard-backbone deck and the historic setting on the historic one.
        """
        given = self.experimental.get("atomistic", {}).get("stage3", {}).get("minimize")
        if given:
            return str(given)
        if self.atomistic_protocol == "hard_backbone":
            return self._hard_backbone()["stage3"]["minimize"]
        return HISTORIC_MINIMIZE[force_field]

    def stages(self, model_type="cg"):
        """``(script, data file)`` per stage, in run order.

        The one place that knows which scripts a protocol writes and what
        each leaves behind. Runners and gates ask this rather than hard-coding
        file names, so adding a stage does not mean editing every caller.
        """
        if model_type != 'cg':
            return list(MINIMISER_STAGES)
        if self.protocol != 'pushoff':
            return list(MINIMISER_STAGES)
        stages = list(PUSHOFF_STAGES)
        if self.config.get('final_bond_style', 'fene') == 'quartic':
            stages.append(("convert_6_parallel.in", "stage6_quartic.data"))
        return stages

    def _get_cg_param(self, *keys, default=None):
        """Get CG parameter from experimental config."""
        d = self.experimental.get('cg', {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default if k == keys[-1] else {})
            else:
                return default
        return d
    
    def _get_run_steps(self, model_type='cg'):
        """Get run steps from experimental config."""
        return self.experimental.get(model_type, {}).get('dynamics', {}).get('run_steps', 10000)

    def write_serial_soft_minimization(self, input_data="system_relaxed.data", groups_file="system.groups", settings_file="system.in.settings", model_type="atomistic", force_field="dreiding", backbone_types=None, n_atom_types=None, ring_types=None):
        """
        Stage 1 of the relaxation protocol.

        For ``model_type='cg'`` the stage depends on ``simulation.protocol``:
        the push-off writes capped-displacement dynamics under FENE + WCA
        (nothing is minimised, despite the script's historic name), while
        ``hardcore_min`` and ``soft_push`` minimise. The atomistic route
        minimises by default (``simulation.atomistic_protocol: "soft_push"``);
        ``"hard_backbone"`` writes a capped push-off in which the backbone
        never goes soft, and needs ``backbone_types`` (the LAMMPS atom types
        of backbone atoms) and ``n_atom_types``.
        """
        if model_type == 'cg' and self.protocol != 'soft_push':
            return self._write_cg_stage1(input_data, groups_file, settings_file)
        if model_type != 'cg' and self.atomistic_protocol == 'hard_backbone':
            self._check_hard_backbone(force_field, backbone_types, n_atom_types)
            if force_field == 'charmm':
                return self._write_charmm_hard_stage1(
                    input_data, groups_file, settings_file, backbone_types, n_atom_types)
            return self._write_hard_backbone_stage1(
                input_data, groups_file, settings_file, backbone_types, n_atom_types,
                ring_types)
        if model_type != 'cg' and force_field == 'charmm':
            return self._write_charmm_stage1(input_data, groups_file, settings_file,
                                             backbone_types)

        script_path = os.path.join(self.sim_dir, "minimize_1_serial.in")

        data_path = os.path.relpath(os.path.join(self.conf_dir, input_data), self.sim_dir).replace("\\", "/")
        groups_path = os.path.relpath(os.path.join(self.chem_dir, groups_file), self.sim_dir).replace("\\", "/")
        settings_path = os.path.relpath(os.path.join(self.chem_dir, settings_file), self.sim_dir).replace("\\", "/")
        
        with open(script_path, 'w') as f:
            f.write(f"# LAMMPS Stage 1: Serial Soft Minimization ({model_type.upper()})\n\n")
            
            if model_type == 'cg':
                f.write("units           lj\n")
                f.write("atom_style      full\n")
                f.write("boundary        p p p\n")
                f.write("comm_modify     mode single cutoff 5.0\n")
                # User requested Harmonic start
                f.write("bond_style      harmonic\n")
                if self.config.get('include_angles', True):
                    f.write("angle_style     harmonic\n")
                # Pair style: repulsive (2^(1/6)) or attractive (2.5)
                pair_style = self.config.get('pair_style', 'attractive')
                pair_cutoff = 1.122462 if pair_style == 'repulsive' else 2.5
                f.write(f"pair_style      lj/cut {pair_cutoff}\n")
                f.write("special_bonds   lj 0.0 1.0 1.0\n\n")
            else:
                f.write("units           real\n")
                f.write("atom_style      full\n")
                f.write("boundary        p p p\n")
                f.write("bond_style      harmonic\n")
                f.write("angle_style     harmonic\n")
                f.write("dihedral_style  harmonic\n")
                f.write("improper_style  umbrella\n")
                f.write("special_bonds   dreiding\n")
                f.write("pair_style      lj/cut/coul/long 12.0\n")
                f.write("kspace_style    pppm 1.0e-4\n\n")
            
            f.write(f"read_data       {data_path}\n")
            f.write(f"include         {settings_path}\n")
            f.write(f"include         {groups_path}\n\n")
            
            if model_type == 'cg':
                f.write("group           beads subtract all nodes\n\n")

            f.write("neighbor        2.0 bin\n")
            f.write("neigh_modify    every 1 delay 0 check yes\n\n")
            if model_type == 'atomistic':
                f.write(self._dump_open("stage1", backbone_types))

            f.write("# --- Switch to Soft Potential ---\n")
            if model_type == 'atomistic': f.write("kspace_style    none\n")
            f.write("pair_style      soft 1.0\n")
            f.write("pair_coeff      * * 0.0\n")
            f.write("variable        prefactor equal ramp(0,30)\n\n")
            
            if model_type == 'cg':
                f.write("# --- Stage A: Freeze All (Beads & Nodes) ---\n")
                f.write("fix             freeze_beads beads setforce 0 0 0\n")
                f.write("fix             freeze_nodes nodes setforce 0 0 0\n")
                f.write("fix             soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("min_style       cg\n")
                f.write("minimize        1.0e-4 1.0e-6 1000 10000\n")
                f.write("unfix           soft_push\n")
                f.write("unfix           freeze_beads\n")
                f.write("write_data      min_stage_A.data\n\n")

                f.write("# --- Stage B: Relax Beads (Nodes Fixed) ---\n")
                f.write("fix             soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize        1.0e-4 1.0e-6 1000 10000\n")
                f.write("unfix           soft_push\n")
                f.write("unfix           freeze_nodes\n")
                f.write("write_data      min_stage_B.data\n\n")

                f.write("# --- Stage C: Relax All ---\n")
                f.write("fix             soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize        1.0e-4 1.0e-6 10000 100000\n")
                f.write("unfix           soft_push\n")
                f.write("write_data      system_after_soft.data\n")
                f.write("write_restart   1.restart\n")
            else:
                f.write("# --- Freeze All Groups ---\n")
                f.write("if \"$(is_defined(group,si_atoms))\" then \"fix freeze_si si_atoms setforce 0 0 0\"\n")
                f.write("if \"$(is_defined(group,c_atoms))\"  then \"fix freeze_c  c_atoms  setforce 0 0 0\"\n")
                f.write("if \"$(is_defined(group,o_atoms))\"  then \"fix freeze_o  o_atoms  setforce 0 0 0\"\n")
                f.write("if \"$(is_defined(group,h_atoms))\"  then \"fix freeze_h  h_atoms  setforce 0 0 0\"\n")
                f.write("if \"$(is_defined(group,nodes))\"    then \"fix freeze_nodes nodes setforce 0 0 0\"\n\n")
                
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("min_style cg\nminimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
                
                f.write("if \"$(is_defined(fix,freeze_h))\" then \"unfix freeze_h\"\n")
                f.write("if \"$(is_defined(fix,freeze_c))\" then \"unfix freeze_c\"\n")
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
                
                f.write("if \"$(is_defined(fix,freeze_o))\" then \"unfix freeze_o\"\n")
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
                
                f.write("if \"$(is_defined(fix,freeze_si))\" then \"unfix freeze_si\"\n")
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
                
                f.write("if \"$(is_defined(fix,freeze_nodes))\" then \"unfix freeze_nodes\"\n")
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
                
                f.write(self._dump_close())
                f.write("reset_timestep 0\ntimestep 1.0\n")
                f.write(self._dump_open("stage1", backbone_types, append=True))
                f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
                f.write("fix nve_limit all nve/limit 0.1\nrun 1000\nunfix nve_limit\nunfix soft_push\n")
                f.write("minimize 1.0e-4 1.0e-6 1000 10000\n")
                f.write(self._dump_close())
                f.write("write_data system_after_soft.data\n")
                f.write("write_restart 1.restart\n")

        return script_path

    # ==================================================================
    # CG stage 1: push-off (default) and the hard-core minimiser
    # ==================================================================

    def _cg_header(self, input_data, groups_file, settings_file, wca_cutoff,
                   neighbor_skin, comm_cutoff):
        """Everything from ``units`` down to the neighbour settings.

        The bond and angle styles named here are the ones in force while the
        data file is *parsed*, not the ones the run uses. A CG data file
        carries a ``Bond Coeffs`` section in whatever style the chemistry
        stage wrote (harmonic), so ``read_data`` needs that style to read it;
        each protocol then sets its own style and coefficients before the
        first force is ever computed. A data file with no coefficient
        sections -- the end-linked convention writes none -- is read the same
        way.

        ``neighbor`` and ``comm_modify`` come before anything that triggers a
        system init (``delete_bonds`` does), because a ghost cutoff too small
        for a stretched bond is how a run loses a bond partner across a
        processor boundary.

        ``groups_file=None`` names the groups from the end-linked atom types
        instead of including a file: that convention *is* the group
        definition (type 3 = junction), so a bead-spring build that never went
        through the chemistry stage needs no groups file at all. The same
        goes for ``settings_file=None``.
        """
        def rel(base, name):
            return os.path.relpath(os.path.join(base, name),
                                   self.sim_dir).replace("\\", "/")

        lines = ["units           lj",
                 "atom_style      full",
                 "boundary        p p p",
                 "bond_style      harmonic"]
        if self.config.get('include_angles', True):
            lines.append("angle_style     harmonic")
        lines += [f"pair_style      lj/cut {wca_cutoff}",
                  "special_bonds   lj 0.0 1.0 1.0",
                  "",
                  f"read_data       {rel(self.conf_dir, input_data)}"]
        if settings_file:
            lines.append(f"include         {rel(self.chem_dir, settings_file)}")
        if groups_file:
            lines.append(f"include         {rel(self.chem_dir, groups_file)}")
        lines.append("")
        if not groups_file:
            lines.append(f"group           nodes type {ENDLINKED_JUNCTION}")
        lines += ["group           beads subtract all nodes",
                  "",
                  f"neighbor        {neighbor_skin} bin",
                  "neigh_modify    every 1 delay 0 check yes",
                  f"comm_modify     mode single cutoff {comm_cutoff}",
                  "", ""]
        return "\n".join(lines)

    def _write_cg_stage1(self, input_data, groups_file, settings_file):
        if self.protocol == 'pushoff':
            return self._write_cg_pushoff_stage1(input_data, groups_file,
                                                 settings_file)
        return self._write_cg_hardcore_stage1(input_data, groups_file,
                                              settings_file)

    def _write_cg_pushoff_stage1(self, input_data, groups_file, settings_file):
        """Stage 1: resolve the build's overlaps without a minimiser.

        A conjugate-gradient minimiser resolves an overlap by whatever move
        lowers the energy, and with a harmonic bond the cheapest move is often
        to stretch a bond and let the overlapping bead through it. Measured on
        the N20 build: 85 bonds stretched to 1.70 sigma, 57 still threaded at
        1.3-1.4 sigma in every later stage, and Z per bridge drifting 0.19 to
        0.24 as those threaded strands crossed. Capped dynamics cannot do it:
        FENE diverges at 1.5 sigma, so a bond never opens far enough, and
        ``nve/limit 0.02`` means no bead travels more than 0.02 sigma in a
        step however large the overlap force is.
        """
        p = self._pushoff()
        s1 = p["stage1"]
        K, R0, eps, sig = p["bond"]
        script_path = os.path.join(self.sim_dir, "minimize_1_serial.in")

        body = f"""# LAMMPS Stage 1: FENE + WCA push-off (CG, protocol "pushoff")
#
# No minimiser, no soft potential, no harmonic bond: the force field of the
# final state is in force from the first step and the only thing that changes
# is how far a bead may move per step. That is what keeps the network's
# entanglement state the one the build placed.
#
# The script name is historic. Nothing here minimises.

{self._cg_header(input_data, groups_file, settings_file, p["wca_cutoff"],
                 p["neighbor_skin"], p["comm_cutoff"])}\
# --- Kremer-Grest force field (replaces whatever the data file carried) ---
# The pair style is pinned to WCA, so simulation.pair_style does not apply:
# a relaxation that has to preserve a topology cannot run an attractive tail
# that pulls chains into one another.
bond_style      fene
bond_coeff      1 {K} {R0} {eps} {sig}
special_bonds   fene
pair_style      lj/cut {p["wca_cutoff"]}
pair_coeff      * * 1.0 1.0 {p["wca_cutoff"]}
"""
        if self.config.get('include_angles', True) and self.config.get('remove_cg_angles', True):
            body += """# Kremer-Grest chains are fully flexible; the stiff angles the chemistry
# stage writes are dropped here rather than at stage 3, so no stage of this
# protocol runs a force field the final state does not have.
#
# Guarded, because the same script has to read a data file that never had
# angles: the end-linked writer emits none, and `delete_bonds all angle 1*1`
# on a file with no angle types is "Numeric index 1 is out of bounds (0-0)".
# `angle_style none` then drops the style itself, not just its terms, which
# is what lets a later stage switch to the quartic bond -- that style refuses
# to run while any 3- or 4-body style is defined, even with zero angles left.
if "$(extract_setting(nangletypes)) > 0" then "delete_bonds all angle 1*1 remove"
angle_style     none
"""
        body += f"""
thermo          {p["thermo_freq"]}
thermo_style    custom step temp pe press density

# --- Capped push-off ---
velocity        all create {p["temperature"]} {p["seed"]} rot yes dist gaussian
timestep        {s1["timestep"]}
fix             lim all nve/limit {s1["limit"]}
fix             lang all langevin {p["temperature"]} {p["temperature"]} {s1["tdamp"]} {p["seed"]}
run             {s1["steps"]}
unfix           lim
unfix           lang

write_data      {PUSHOFF_STAGES[0][1]}
write_restart   1.restart
"""
        with open(script_path, 'w') as f:
            f.write(body)
        return script_path

    def _write_cg_hardcore_stage1(self, input_data, groups_file, settings_file):
        """Stage 1 of ``hardcore_min``: WCA throughout, minimiser kept.

        The protocol ``tests/workflows/lammps_hardcore/`` held before the push-off.
        A hard core stops chains passing through one another during the push,
        which the soft potential does not, but the minimiser still threads
        bonds -- so this is for reproducing earlier runs, not for new ones.
        """
        p = self._pushoff()
        m = self.experimental.get('cg', {}).get('minimize', {})
        script_path = os.path.join(self.sim_dir, "minimize_1_serial.in")

        body = f"""# LAMMPS Stage 1: hard-core minimisation (CG, protocol "hardcore_min")
#
# WCA from the first step -- no soft potential, so nothing passes through
# anything during the push. The conjugate-gradient minimiser is kept, which
# is what makes this protocol crossing-prone: it resolves an overlap by
# stretching a bond and threading a bead through it (85 bonds to 1.70 sigma
# on the N20 build). Use "pushoff" for new work.

{self._cg_header(input_data, groups_file, settings_file, p["wca_cutoff"],
                 2.0, 5.0)}\
# --- Stage A: Freeze All (Beads & Nodes) ---
fix             freeze_beads beads setforce 0 0 0
fix             freeze_nodes nodes setforce 0 0 0
min_style       cg
minimize        {m.get('etol', 1.0e-4)} {m.get('ftol', 1.0e-6)} {m.get('maxiter', 1000)} {m.get('maxeval', 10000)}
unfix           freeze_beads
write_data      min_stage_A.data

# --- Stage B: Relax Beads (Nodes Fixed) ---
minimize        {m.get('etol', 1.0e-4)} {m.get('ftol', 1.0e-6)} {m.get('maxiter', 1000)} {m.get('maxeval', 10000)}
unfix           freeze_nodes
write_data      min_stage_B.data

# --- Stage C: Relax All ---
minimize        {m.get('etol', 1.0e-4)} {m.get('ftol', 1.0e-6)} {m.get('final_maxiter', 10000)} {m.get('final_maxeval', 100000)}
write_data      {MINIMISER_STAGES[0][1]}
write_restart   1.restart
"""
        with open(script_path, 'w') as f:
            f.write(body)
        return script_path

    # ==================================================================
    # CG stages 2-6: the push-off tail
    # ==================================================================

    def _write_cg_pushoff_tail(self):
        """Stages 2 to 5 (and 6 when the run ends under the quartic bond).

        One script per stage, chained through restart files, because every
        stage has to leave a data file behind for the gate to read: the
        acceptance gate scans the bond histogram after stage 2 and at every
        later stage, and Z1+ is reported per stage so states are only ever
        compared at the same density and temperature.
        """
        p = self._pushoff()
        self._write_cg_pushoff_stage2(p)
        self._write_cg_pushoff_stage3(p)
        self._write_cg_pushoff_stage4(p)
        self._write_cg_pushoff_stage5(p)
        n = 5
        if self.config.get('final_bond_style', 'fene') == 'quartic':
            self._write_cg_pushoff_stage6(p)
            n = 6
        print(f"Generated CG push-off protocol (stages 2-{n}).")

    def _cg_restart_header(self, restart, p, title, note=""):
        return f"""# LAMMPS {title} (CG, protocol "pushoff")
{note}
read_restart    {restart}

neighbor        {p["neighbor_skin"]} bin
neigh_modify    every 1 delay 0 check yes
comm_modify     mode single cutoff {p["comm_cutoff"]}

thermo          {p["thermo_freq"]}
thermo_style    custom step temp pe press density
"""

    def _write_cg_pushoff_stage2(self, p):
        """Stage 2: loosen the cap, then take it off.

        The cap is raised rather than removed in one move because the last
        overlaps are the deep ones. By the end of this stage the dynamics are
        plain NVE under the Langevin thermostat, which is what the bond gate
        is checked against: zero bonds above 1.2 sigma from here on.
        """
        s2 = p["stage2"]
        script_path = os.path.join(self.sim_dir, "minimize_2_parallel.in")
        with open(script_path, 'w') as f:
            f.write(self._cg_restart_header(
                "1.restart", p, "Stage 2: uncapped push-off",
                "#\n# Raise the displacement cap, then drop it. Nothing is minimised.\n"
            ) + f"""
timestep        {s2["timestep"]}
fix             lim all nve/limit {s2["limit"]}
fix             lang all langevin {p["temperature"]} {p["temperature"]} {s2["tdamp"]} {p["seed"]}
run             {s2["steps"]}
unfix           lim

fix             nve all nve
run             {s2["free_steps"]}
unfix           lang

write_data      {PUSHOFF_STAGES[1][1]}
write_restart   2.restart
""")
        return script_path

    def _write_cg_pushoff_stage3(self, p):
        """Stage 3: equilibrate at the density the chains were built at."""
        s3 = p["stage3"]
        script_path = os.path.join(self.sim_dir, "minimize_3_parallel.in")
        with open(script_path, 'w') as f:
            f.write(self._cg_restart_header(
                "2.restart", p, "Stage 3: equilibration at the build density",
                "#\n# A weaker thermostat (damp 10) so the chains relax rather than being\n"
                "# dragged. If a minimiser is wanted for a final polish it belongs\n"
                "# after this stage, never before it, and the bond histogram has to be\n"
                "# re-checked afterwards.\n"
            ) + f"""
reset_timestep  0
fix             nve all nve
fix             lang all langevin {p["temperature"]} {p["temperature"]} {s3["tdamp"]} {p["seed"]}
run             {s3["steps"]}

write_data      {PUSHOFF_STAGES[2][1]}
write_restart   3.restart
""")
        return script_path

    def _write_cg_pushoff_stage4(self, p):
        """Stage 4: affine compression to the target density, then settle.

        The target box is computed inside LAMMPS from the atom count and the
        current box, so the generator never has to know how many beads the
        chemistry stage produced. With no ``simulation.rho_final`` the scale
        factor is 1 and the stage is a pure settle -- which is the
        no-compression case of the acceptance test, not a skipped stage.

        ``rho_final`` is an LJ *number* density, beads per sigma cubed, and is
        deliberately not called ``target_density``: ``chemistry.target_density``
        is a mass density in g/cm^3 and is what sizes the build box. Two keys
        with one name and different units is a compression that silently does
        not happen.

        One ``run`` for the deformation, always. ``fix deform`` re-bases its
        reference box at every ``run`` command, so a deformation split across
        several runs multiplies the box by the same factor once per chunk.
        """
        s4 = p["stage4"]
        rho = self.config.get('rho_final')
        script_path = os.path.join(self.sim_dir, "deform_4_parallel.in")

        if rho:
            scale = (f"variable        rho_target equal {rho}\n"
                     "variable        sfac equal (atoms/v_rho_target/(lx*ly*lz))^(1.0/3.0)\n")
        else:
            scale = ("# No simulation.rho_final: deform to the box the build already\n"
                     "# has, so this stage settles the equilibrated state and nothing else.\n"
                     "variable        sfac equal 1.0\n")

        with open(script_path, 'w') as f:
            f.write(self._cg_restart_header(
                "3.restart", p, "Stage 4: affine compression and settle",
                "#\n# remap x carries the atoms with the box, so the compression is affine\n"
                "# and no chain is left outside it.\n"
            ) + f"""
{scale}
fix             nve all nve
fix             lang all langevin {p["temperature"]} {p["temperature"]} {p["stage3"]["tdamp"]} {p["seed"]}
fix             def all deform 1 x final $(xlo*v_sfac) $(xhi*v_sfac) y final $(ylo*v_sfac) $(yhi*v_sfac) z final $(zlo*v_sfac) $(zhi*v_sfac) units box remap x
run             {s4["deform_steps"]}
unfix           def
run             {s4["settle_steps"]}

write_data      {PUSHOFF_STAGES[3][1]}
write_restart   4.restart
""")
        return script_path

    def _write_cg_pushoff_stage5(self, p):
        """Stage 5: quench to the comparison temperature.

        A 50k-step ramp at damp 10, not the 10k-step damp-100 ramp of the
        reference input: that one leaves the system at T = 0.83 while the
        reference *data files* sit at 0.42, and Z1+ and the chain statistics
        both depend on temperature, so a comparison across that gap is
        measuring the gap.
        """
        s5 = p["stage5"]
        T, Tq = p["temperature"], p["quench_temperature"]
        script_path = os.path.join(self.sim_dir, "quench_5_parallel.in")
        with open(script_path, 'w') as f:
            f.write(self._cg_restart_header(
                "4.restart", p, "Stage 5: quench",
                f"#\n# T {T} -> {Tq}, then settle at {Tq}.\n"
            ) + f"""
fix             nve all nve
fix             lang all langevin {T} {Tq} {s5["tdamp"]} {p["seed"]}
run             {s5["ramp_steps"]}
unfix           lang
fix             lang all langevin {Tq} {Tq} {s5["tdamp"]} {p["seed"]}
run             {s5["settle_steps"]}

write_data      {PUSHOFF_STAGES[4][1]}
write_restart   5.restart
print "=== PROTOCOL DONE ==="
""")
        return script_path

    def _write_cg_pushoff_stage6(self, p):
        """Stage 6: convert to the quartic bond, for deformation runs only.

        The quartic bond breaks silently above 1.5 sigma (64 bonds broke in a
        smoke run and split chains), so it is never the bond a relaxation runs
        under -- it goes on at the end, on a state whose bonds are already
        near 0.97. It also costs nothing in entanglement: re-quenching the
        same state under quartic gave Z 0.223 against FENE's 0.224.

        ``bond_style quartic/omp`` computes correct forces but leaves the
        subtracted bonded-LJ term out of E_pair and the virial (pressure 4.79
        instead of 0.05 at rho 0.30), so the style is pinned to the serial
        version with suffix off / suffix on. Any stress read from a run with
        the OpenMP package still has to be checked against a serial ``run 0``.

        Both the suffix wrapper and the angle style are guarded. ``suffix on``
        with no suffix ever defined is an error ("May only enable suffixes
        after defining one"), so a serial run may not carry the wrapper
        unconditionally; and the quartic style refuses to initialise while any
        3- or 4-body style is defined, whatever the angle count, so
        ``angle_style none`` has to be in force by the time it is set.
        """
        s6 = p["stage6"]
        Tq = p["quench_temperature"]
        script_path = os.path.join(self.sim_dir, "convert_6_parallel.in")
        with open(script_path, 'w') as f:
            f.write(self._cg_restart_header(
                "5.restart", p, "Stage 6: convert to the quartic bond",
                "#\n# For deformation / bond-breaking runs. The relaxation above ran under\n"
                "# FENE; only the final state is converted.\n"
            ) + f"""
angle_style     none
if "$(is_active(package,omp))" then "suffix off"
bond_style      quartic
bond_coeff      1 2351.0 0.0 -0.7425 1.5 94.745
if "$(is_active(package,omp))" then "suffix on"
special_bonds   lj 1 1 1

fix             nve all nve
fix             lang all langevin {Tq} {Tq} {p["stage5"]["tdamp"]} {p["seed"]}
run             {s6["steps"]}

write_data      stage6_quartic.data
write_restart   6.restart
print "=== PROTOCOL DONE ==="
""")
        return script_path

    def write_parallel_production(self, settings_file="system.in.settings", model_type="atomistic", force_field="dreiding", charmm_pair_style="lj/charmmfsw/coul/long", backbone_types=None, n_atom_types=None, ring_types=None):
        """
        Parent function that generates the complete parallel minimization pipeline:
        1. minimize_2_parallel.in (Stage 2: Ramp)
        2. minimize_3_parallel.in (Stage 3: Tight Min + Equilibration)
        """
        if model_type != 'cg' and self.atomistic_protocol == 'hard_backbone':
            self._check_hard_backbone(force_field, backbone_types, n_atom_types)
            if force_field == 'charmm':
                self._write_charmm_hard_stage2(settings_file, backbone_types, n_atom_types)
                self._write_charmm_stage3(settings_file, charmm_pair_style, backbone_types)
                print("Generated parallel scripts (CHARMM hard-backbone stage 2, stage 3).")
                return
            self._write_hard_backbone_stage2(settings_file, backbone_types, n_atom_types,
                                             ring_types)
            self._write_stage3_equilibration(settings_file, model_type, backbone_types)
            print("Generated parallel scripts (hard-backbone stage 2, stage 3).")
            return
        if model_type != 'cg' and force_field == 'charmm':
            self._write_charmm_stage2(settings_file, backbone_types)
            self._write_charmm_stage3(settings_file, charmm_pair_style, backbone_types)
            print("Generated parallel minimization scripts (CHARMM stages 2 & 3).")
            return
        if model_type == 'cg':
            if self.protocol == 'pushoff':
                self._write_cg_pushoff_tail()
                return
            self._write_cg_minimization_equil(settings_file)
            print(f"Generated parallel minimization scripts (CG Stages 2 & 3).")
            return

        self._write_stage2_ramp(settings_file, model_type, backbone_types)
        self._write_stage3_equilibration(settings_file, model_type, backbone_types)
        print(f"Generated parallel minimization scripts (Stages 2 & 3).")

    # ==================================================================
    # Atomistic, protocol "hard_backbone": the backbone never goes soft
    # ==================================================================

    def _check_hard_backbone(self, force_field, backbone_types, n_atom_types):
        if not backbone_types or not n_atom_types:
            raise ValueError(
                "simulation.atomistic_protocol 'hard_backbone' needs the backbone "
                "atom types; the pipeline passes them from the strand record")

    def _write_hard_backbone_stage1(self, input_data, groups_file, settings_file,
                                    backbone_types, n_atom_types, ring_types=None):
        """Stage 1: capped push-off with a fixed soft core on backbone pairs.

        ``ring_types`` (an aromatic ring's atom types, C_R and the like) join
        the hard pairs, so that no ring passes through a bond or another
        ring either; the backbone dump keeps the backbone's types.
        """
        p = self._hard_backbone()
        s1 = p["stage1"]
        backbone = sorted({int(t) for t in backbone_types})
        rings = sorted({int(t) for t in ring_types or ()} - set(backbone))
        hard = sorted(set(backbone) | set(rings))
        light = [t for t in range(1, int(n_atom_types) + 1) if t not in hard]
        rel = lambda d, f: os.path.relpath(os.path.join(d, f), self.sim_dir).replace("\\", "/")
        core = "\n".join(f"pair_coeff      {a} {b} {p['core']} {p['core_cutoff']}"
                         for i, a in enumerate(hard) for b in hard[i:])
        adapt = (f"fix             light all adapt 1 "
                 f"{light_pair_terms('soft', 'a', light, n_atom_types, 'prefactor')} "
                 f"reset no\n" if light else "")
        unadapt = "unfix           light\n" if light else ""
        if rings:
            head = f"""# Pairs of backbone and aromatic ring atoms (atom types {' '.join(map(str, hard))},
# the rings' {' '.join(map(str, rings))}) keep a soft core of {p['core']} kcal/mol out to
# {p['core_cutoff']} A from the first step, so no backbone atom passes through another
# strand's bond and no ring through a bond or another ring; every pair
# with a light atom (types {' '.join(map(str, light)) or 'none'}) ramps"""
        else:
            head = f"""# Backbone-backbone pairs (atom types {' '.join(map(str, hard))}) keep a soft
# core of {p['core']} kcal/mol out to {p['core_cutoff']} A from the first
# step, so no backbone atom passes through another strand's bond; every pair
# with a light atom (types {' '.join(map(str, light)) or 'none'}) ramps"""
        body = f"""# LAMMPS Stage 1: hard-backbone push-off (ATOMISTIC, protocol "hard_backbone")
#
{head}
# 0 -> {p['light_max']} kcal/mol at {p['light_cutoff']} A, as in the historic
# deck. Capped displacement under a Langevin thermostat and no minimiser: a
# minimiser resolves an overlap by whatever lowers the energy, which on the
# bead-spring route was a bond stretched and a bead let through it.

units           real
atom_style      full
boundary        p p p
bond_style      harmonic
angle_style     harmonic
dihedral_style  harmonic
improper_style  umbrella
special_bonds   dreiding
pair_style      lj/cut/coul/long 12.0
kspace_style    pppm 1.0e-4

read_data       {rel(self.conf_dir, input_data)}
include         {rel(self.chem_dir, settings_file)}
include         {rel(self.chem_dir, groups_file)}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
comm_modify     mode single cutoff 8.0

{self._dump_open('stage1', backbone)}kspace_style    none
pair_style      soft {p['core_cutoff']}
pair_coeff      * * 0.0 {p['light_cutoff']}
{core}
variable        prefactor equal ramp(0,{p['light_max']})
{adapt}velocity        all create {p['temperature']} {s1['velocity_seed']} dist gaussian
fix             lang all langevin {p['temperature']} {p['temperature']} {p['tdamp']} {s1['langevin_seed']}
fix             cap all nve/limit {s1['limit']}
timestep        1.0
thermo          1000
run             {s1['steps']}
unfix           cap
unfix           lang
{unadapt}{self._dump_close()}write_data      system_after_soft.data
write_restart   1.restart
"""
        path = os.path.join(self.sim_dir, "minimize_1_serial.in")
        with open(path, "w") as f:
            f.write(body)
        return path

    def _write_hard_backbone_stage2(self, settings_file, backbone_types, n_atom_types,
                                    ring_types=None):
        """Stage 2: the epsilon ramp for light pairs, backbone at full depth
        (and the ``ring_types``, as in stage 1)."""
        p = self._hard_backbone()
        s2 = p["stage2"]
        backbone = sorted({int(t) for t in backbone_types})
        rings = sorted({int(t) for t in ring_types or ()} - set(backbone))
        hard = sorted(set(backbone) | set(rings))
        light = [t for t in range(1, int(n_atom_types) + 1) if t not in hard]
        settings = os.path.relpath(os.path.join(self.chem_dir, settings_file),
                                   self.sim_dir).replace("\\", "/")
        # scale yes: each pair ramps to its own depth. Without it fix adapt
        # sets epsilon to the variable itself, 1 kcal/mol at the end for every
        # light pair (0.0957 for C_3), which is what the first DP-30 run had.
        adapt = (f"fix             ramp all adapt 1 "
                 f"{light_pair_terms('lj/cut/coul/long', 'epsilon', light, n_atom_types, 'scale')} "
                 f"scale yes reset no\n" if light else "")
        unadapt = "unfix           ramp\n" if light else ""
        run_steps = self._hard_ramp_steps(s2)
        kept = (f"backbone and aromatic ring pairs (types {' '.join(map(str, hard))}, the\n"
                f"# rings' {' '.join(map(str, rings))}) are at full depth from the first step."
                if rings else
                f"backbone-backbone pairs (types {' '.join(map(str, hard))}) are at full depth\n"
                f"# from the first step.")
        body = f"""# LAMMPS Stage 2: epsilon ramp with the backbone at full depth (protocol "hard_backbone")
#
# Pairs with a light atom ramp from 0.001 to 1 of their Lennard-Jones depth;
# {kept} Capped under a Langevin thermostat: the historic ramp
# runs nve/limit alone and ended a small test network at 964 K.

units           real
atom_style      full
boundary        p p p
bond_style      harmonic
angle_style     harmonic
dihedral_style  harmonic
improper_style  umbrella
pair_style      soft {p['core_cutoff']}

read_data       system_after_soft.data

neigh_modify    one 10000
comm_modify     mode single cutoff 12.0

{self._dump_open('stage2', backbone)}pair_style      lj/cut/coul/long 10.0 10.0
kspace_style    pppm 1.0e-4
include         {settings}
special_bonds   lj/coul 0.0 0.0 1.0

variable        scale equal ramp(0.001,1.0)
{adapt}fix             lang all langevin {p['temperature']} {p['temperature']} {p['tdamp']} {s2['langevin_seed']}
fix             cap all nve/limit {s2['limit']}
timestep        1.0
thermo          1000
# RUNTIME: {run_steps} steps
run             {run_steps}
unfix           cap
unfix           lang
{unadapt}{self._dump_close()}write_data      system_ramped.data
"""
        with open(os.path.join(self.sim_dir, "minimize_2_parallel.in"), "w") as f:
            f.write(body)

    def _write_stage2_ramp(self, settings_file, model_type, backbone_types=None):
        """
        Stage 2: Parallel Ramp.
        METHODOLOGY: Set 1 (Slow 200k Step Ramp + Extended Cutoffs)
        Inputs: system_after_soft.data
        Outputs: system_ramped.data
        """
        script_path = os.path.join(self.sim_dir, "minimize_2_parallel.in")
        settings_path = os.path.relpath(os.path.join(self.chem_dir, settings_file), self.sim_dir).replace("\\", "/")
        
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 2: Parallel Ramp\n")
            f.write("# METHODOLOGY: Set 1 (Slow 200k Step Ramp + Extended Cutoffs)\n\n")
            
            f.write("units           real\n")
            f.write("atom_style      full\n")
            f.write("boundary        p p p\n")
            f.write("bond_style      harmonic\n")
            f.write("angle_style     harmonic\n")
            f.write("dihedral_style  harmonic\n")
            f.write("improper_style  umbrella\n")
            f.write("pair_style      soft 1.0\n\n")
            
            f.write("# --- 1. Load Soft State ---\n")
            f.write("read_data       system_after_soft.data\n\n")
            
            f.write("# --- 2. CRITICAL SAFETY (From Set 1) ---\n")
            f.write("# Prevents \"Bond atoms missing\" and \"Neighbor list overflow\"\n")
            f.write("neigh_modify    one 10000\n")
            f.write("comm_modify     mode single cutoff 12.0\n\n")
            
            f.write(self._dump_open("stage2", backbone_types))
            f.write("# --- 3. Soft Pre-Minimization ---\n")
            f.write("pair_style      soft 1.0\n")
            f.write("pair_coeff      * * 1.0\n")
            f.write("min_style       cg\n")
            f.write("minimize        1.0e-4 1.0e-6 1000 10000\n\n")
            
            f.write("# --- 4. Switch to Real Potential ---\n")
            f.write("pair_style      lj/cut/coul/long 10.0 10.0\n")
            f.write("kspace_style    pppm 1.0e-4\n")
            f.write(f"include         {settings_path}\n\n")

            f.write("# Enforce Set 1 Special Bonds\n")
            f.write("special_bonds   lj/coul 0.0 0.0 1.0\n\n")

            f.write("# --- 5. The Ramp (Set 1 Logic) ---\n")
            f.write("# Linearly scale epsilon/charges from 0.001 to 1.0\n")
            f.write("variable        scale equal \"ramp(0.001, 1.0)\"\n")
            f.write("timestep        1.0\n\n")

            # scale yes: every pair ramps to its own DREIDING depth. Without
            # it fix adapt set epsilon to the variable itself, 0.001 to 1
            # kcal/mol for every pair (H_ is 0.0152, Si3 0.31) before 0.4.0.
            f.write("fix             1 all adapt 1 pair lj/cut/coul/long epsilon * * v_scale scale yes\n")
            f.write("fix             fxnve all nve/limit 0.1\n")
            f.write("thermo          1000\n\n")
            
            # RUNTIME from experimental config
            run_steps = self._get_run_steps('atomistic')
            f.write(f"# RUNTIME: {run_steps} steps\n")
            f.write(f"run             {run_steps}\n\n")
            
            f.write("unfix           fxnve\n")
            f.write("unfix           1\n")
            f.write("kspace_modify   compute yes\n\n")
            f.write(self._dump_close())

            f.write("write_data      system_ramped.data\n")

    def _write_stage3_equilibration(self, settings_file, model_type, backbone_types=None):
        """
        Stage 3: Parallel Equilibration.
        METHODOLOGY: Set 1 (Tight Min -> NVT -> NPT)
        Inputs: system_ramped.data
        Outputs: system_equilibrated.data
        """
        script_path = os.path.join(self.sim_dir, "minimize_3_parallel.in")
        settings_path = os.path.relpath(os.path.join(self.chem_dir, settings_file), self.sim_dir).replace("\\", "/")
        
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 3: Parallel Equilibration\n")
            f.write("# METHODOLOGY: Set 1 (Tight Min -> NVT -> NPT)\n\n")
            
            f.write("units           real\n")
            f.write("atom_style      full\n")
            f.write("boundary        p p p\n")
            f.write("bond_style      harmonic\n")
            f.write("angle_style     harmonic\n")
            f.write("dihedral_style  harmonic\n")
            f.write("improper_style  umbrella\n")
            f.write("pair_style      lj/cut/coul/long 10.0 10.0\n\n")
            
            f.write("# --- 1. Load Ramped State ---\n")
            f.write("read_data       system_ramped.data\n\n")
            
            f.write("# --- 2. Safety Settings ---\n")
            f.write("# Keep these even in stage 3 to prevent random crashes\n")
            f.write("neigh_modify    one 10000\n\n")
            f.write(self._dump_open("stage3", backbone_types))
            
            f.write("# --- 3. Define Potential ---\n")
            f.write("pair_style      lj/cut/coul/long 10.0 10.0\n")
            f.write("kspace_style    pppm 1.0e-4\n")
            f.write(f"include         {settings_path}\n")
            f.write("special_bonds   lj/coul 0.0 0.0 1.0\n\n")
            
            if self.atomistic_protocol == "hard_backbone":
                f.write("# --- 4. Minimization, capped (hard-backbone deck) ---\n")
            else:
                f.write("# --- 4. Tight Minimization (Set 1 Logic) ---\n")
                f.write("# High precision 1e-8/1e-10 tolerances\n")
            f.write("min_style       cg\n")
            f.write(f"minimize        {self._stage3_minimize('dreiding')}\n\n")
            
            f.write("write_data      system_minimized_final.data\n\n")
            
            f.write("# --- 5. Equilibration Loop (Set 1 Logic) ---\n")
            f.write(self._dump_close())
            f.write("reset_timestep  0\n")
            f.write(self._dump_open("stage3", backbone_types, append=True))
            f.write("variable        temp equal 300\n")
            f.write("velocity        all create ${temp} 12345\n\n")
            
            f.write("# NVT (1000 steps)\n")
            f.write("fix             1 all nvt temp ${temp} ${temp} 100.0\n")
            f.write("run             1000\n")
            f.write("unfix           1\n")
            f.write("write_data      after_nvt_real.data\n\n")
            
            f.write("# NPT (1000 steps)\n")
            f.write("fix             1 all npt temp ${temp} ${temp} 100.0 iso 1.0 1.0 1000.0\n")
            f.write("run             1000\n")
            f.write("unfix           1\n\n")
            
            f.write(self._dump_close())
            f.write("write_data      system_equilibrated.data\n")
            f.write("print \"All Minimization Stages Complete.\"\n")

    # ==================================================================
    # Atomistic CHARMM (chemistry.force_field = "charmm")
    # ==================================================================
    #
    # The same three stages and file names as the DREIDING route, with CHARMM
    # styles. The 1-4 terms live in `dihedral_style charmm(fsw)`, which LAMMPS
    # accepts only with an lj/charmm* pair style; so the soft stages include
    # `<settings>.soft` (1-4 weights 0, plain charmm dihedrals), the epsilon
    # ramp runs lj/cut/coul/long with `<settings>.lj` and CHARMM's arithmetic
    # mixing (fix adapt cannot scale the lj/charmm* styles), and stage 3 runs
    # the full CHARMM set.

    def _charmm_paths(self, settings_file, groups_file="system.groups",
                      input_data="system_relaxed.data"):
        def rel(d, f):
            return os.path.relpath(os.path.join(d, f), self.sim_dir).replace("\\", "/")
        return {
            "data": rel(self.conf_dir, input_data),
            "groups": rel(self.chem_dir, groups_file),
            "full": rel(self.chem_dir, settings_file),
            "soft": rel(self.chem_dir, settings_file + ".soft"),
            "lj": rel(self.chem_dir, settings_file + ".lj"),
        }

    @staticmethod
    def _charmm_header(dihedral_style):
        return ("units           real\n"
                "atom_style      full\n"
                "boundary        p p p\n"
                "bond_style      harmonic\n"
                "angle_style     charmm\n"
                f"dihedral_style  {dihedral_style}\n"
                "improper_style  harmonic\n"
                "special_bonds   charmm\n")

    def _write_charmm_stage1(self, input_data, groups_file, settings_file,
                             backbone_types=None):
        p = self._charmm_paths(settings_file, groups_file, input_data)
        script_path = os.path.join(self.sim_dir, "minimize_1_serial.in")
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 1: Serial Soft Minimization (ATOMISTIC, CHARMM)\n")
            f.write("# pair_style soft only; the .soft settings carry the bonded terms\n")
            f.write("# with 1-4 weights 0 (CHARMM 1-4 terms need an lj/charmm pair style).\n\n")
            f.write(self._charmm_header("charmm"))
            f.write("pair_style      soft 1.0\n\n")
            f.write(f"read_data       {p['data']}\n")
            f.write(f"include         {p['soft']}\n")
            f.write("pair_coeff      * * 0.0\n")
            f.write(f"include         {p['groups']}\n\n")
            f.write("neighbor        2.0 bin\n")
            f.write("neigh_modify    every 1 delay 0 check yes\n")
            # ghosts past any bond, not just the 3 A of pair soft + skin
            f.write("comm_modify     mode single cutoff 12.0\n")
            f.write("variable        prefactor equal ramp(0,30)\n")
            f.write("thermo          100\n\n")
            f.write(self._dump_open("stage1", backbone_types))
            f.write("# --- Junctions held, everything else pushed apart ---\n")
            f.write('if "$(is_defined(group,nodes))" then "fix freeze_nodes nodes setforce 0 0 0"\n')
            f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
            f.write("min_style cg\nminimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n")
            f.write('if "$(is_defined(fix,freeze_nodes))" then "unfix freeze_nodes"\n\n')
            f.write("# --- All atoms ---\n")
            f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
            f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n\n")
            f.write(self._dump_close())
            f.write("reset_timestep 0\ntimestep 1.0\n")
            f.write(self._dump_open("stage1", backbone_types, append=True))
            f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
            f.write("fix nve_limit all nve/limit 0.1\nrun 1000\nunfix nve_limit\nunfix soft_push\n")
            f.write("fix soft_push all adapt 1 pair soft a * * v_prefactor\n")
            f.write("minimize 1.0e-4 1.0e-6 1000 10000\nunfix soft_push\n")
            f.write(self._dump_close())
            f.write("write_data system_after_soft.data nocoeff\n")
            f.write("write_restart 1.restart\n")
        return script_path

    def _write_charmm_stage2(self, settings_file, backbone_types=None):
        p = self._charmm_paths(settings_file)
        run_steps = self._get_run_steps('atomistic')
        script_path = os.path.join(self.sim_dir, "minimize_2_parallel.in")
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 2: Parallel Ramp (CHARMM)\n")
            f.write("# CHARMM LJ (no 1-4) with arithmetic mixing and NBFIX, epsilon ramped\n")
            f.write("# 0.001 -> 1 under nve/limit; stage 3 turns the full CHARMM set on.\n\n")
            f.write(self._charmm_header("charmm"))
            f.write("pair_style      soft 1.0\n\n")
            f.write("read_data       system_after_soft.data\n")
            f.write(f"include         {p['soft']}\n")
            f.write("pair_coeff      * * 1.0\n\n")
            f.write("neigh_modify    one 10000\n")
            f.write("comm_modify     mode single cutoff 12.0\n\n")
            f.write(self._dump_open("stage2", backbone_types))
            f.write("min_style       cg\n")
            f.write("minimize        1.0e-4 1.0e-6 1000 10000\n\n")
            f.write("pair_style      lj/cut/coul/long 12.0\n")
            f.write("pair_modify     mix arithmetic\n")
            f.write("kspace_style    pppm 1.0e-4\n")
            f.write(f"include         {p['lj']}\n\n")
            f.write('variable        scale equal "ramp(0.001, 1.0)"\n')
            f.write("timestep        1.0\n")
            f.write("fix             1 all adapt 1 pair lj/cut/coul/long epsilon * * v_scale scale yes\n")
            f.write("fix             fxnve all nve/limit 0.1\n")
            f.write("thermo          1000\n")
            f.write(f"run             {run_steps}\n")
            f.write("unfix           fxnve\n")
            f.write("unfix           1\n\n")
            f.write(self._dump_close())
            f.write("write_data      system_ramped.data nocoeff\n")

    def _write_charmm_stage3(self, settings_file, pair_style, backbone_types=None):
        p = self._charmm_paths(settings_file)
        dihedral = "charmmfsw" if "charmmfsw" in pair_style else "charmm"
        script_path = os.path.join(self.sim_dir, "minimize_3_parallel.in")
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 3: Parallel Equilibration (CHARMM)\n")
            f.write("# Full CHARMM: 1-4 terms in the dihedrals, arithmetic mixing, NBFIX, PPPM.\n\n")
            f.write(self._charmm_header(dihedral))
            f.write(f"pair_style      {pair_style} 10.0 12.0\n\n")
            f.write("read_data       system_ramped.data\n\n")
            f.write("neigh_modify    one 10000\n")
            f.write("pair_modify     mix arithmetic\n")
            f.write("kspace_style    pppm 1.0e-4\n")
            f.write(f"include         {p['full']}\n\n")
            f.write("thermo          100\n")
            f.write("thermo_style    custom step pe ke etotal evdwl ecoul epair ebond "
                    "eangle edihed eimp press vol temp\n\n")
            f.write(self._dump_open("stage3", backbone_types))
            f.write("min_style       cg\n")
            f.write(f"minimize        {self._stage3_minimize('charmm')}\n")
            f.write("write_data      system_minimized_final.data\n\n")
            f.write(self._dump_close())
            f.write("reset_timestep  0\n")
            f.write(self._dump_open("stage3", backbone_types, append=True))
            f.write("variable        temp equal 300\n")
            f.write("velocity        all create ${temp} 12345\n")
            f.write("timestep        1.0\n\n")
            f.write("# NVT (1000 steps)\n")
            f.write("fix             1 all nvt temp ${temp} ${temp} 100.0\n")
            f.write("run             1000\n")
            f.write("unfix           1\n")
            f.write("write_data      after_nvt_real.data\n\n")
            f.write("# NPT (1000 steps)\n")
            f.write("fix             1 all npt temp ${temp} ${temp} 100.0 iso 1.0 1.0 1000.0\n")
            f.write("run             1000\n")
            f.write("unfix           1\n\n")
            f.write(self._dump_close())
            f.write("write_data      system_equilibrated.data\n")
            f.write('print "All Minimization Stages Complete."\n')

    def _write_charmm_hard_stage1(self, input_data, groups_file, settings_file,
                                  backbone_types, n_atom_types):
        """Stage 1 of "hard_backbone" in CHARMM styles (the DREIDING one's twin)."""
        p = self._hard_backbone()
        s1 = p["stage1"]
        c = self._charmm_paths(settings_file, groups_file, input_data)
        hard = sorted({int(t) for t in backbone_types})
        light = [t for t in range(1, int(n_atom_types) + 1) if t not in hard]
        core = "\n".join(f"pair_coeff      {a} {b} {p['core']} {p['core_cutoff']}"
                         for i, a in enumerate(hard) for b in hard[i:])
        adapt = (f"fix             light all adapt 1 "
                 f"{light_pair_terms('soft', 'a', light, n_atom_types, 'prefactor')} "
                 f"reset no\n" if light else "")
        unadapt = "unfix           light\n" if light else ""
        body = f"""# LAMMPS Stage 1: hard-backbone push-off (ATOMISTIC, CHARMM, protocol "hard_backbone")
#
# Backbone-backbone pairs (atom types {' '.join(map(str, hard))}) keep a soft
# core of {p['core']} kcal/mol out to {p['core_cutoff']} A from the first
# step; every pair with a light atom ramps 0 -> {p['light_max']} kcal/mol at
# {p['light_cutoff']} A. The .soft settings carry the bonded terms with 1-4
# weights 0. Capped displacement under a Langevin thermostat, no minimiser.

{self._charmm_header("charmm")}pair_style      soft {p['core_cutoff']}

read_data       {c['data']}
include         {c['soft']}
pair_coeff      * * 0.0 {p['light_cutoff']}
{core}
include         {c['groups']}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
comm_modify     mode single cutoff 12.0

{self._dump_open('stage1', hard)}variable        prefactor equal ramp(0,{p['light_max']})
{adapt}velocity        all create {p['temperature']} {s1['velocity_seed']} dist gaussian
fix             lang all langevin {p['temperature']} {p['temperature']} {p['tdamp']} {s1['langevin_seed']}
fix             cap all nve/limit {s1['limit']}
timestep        1.0
thermo          1000
run             {s1['steps']}
unfix           cap
unfix           lang
{unadapt}{self._dump_close()}write_data      system_after_soft.data nocoeff
write_restart   1.restart
"""
        path = os.path.join(self.sim_dir, "minimize_1_serial.in")
        with open(path, "w") as f:
            f.write(body)
        return path

    def _write_charmm_hard_stage2(self, settings_file, backbone_types, n_atom_types):
        """Stage 2 of "hard_backbone" in CHARMM styles: light pairs ramp, no minimiser."""
        p = self._hard_backbone()
        s2 = p["stage2"]
        c = self._charmm_paths(settings_file)
        hard = sorted({int(t) for t in backbone_types})
        light = [t for t in range(1, int(n_atom_types) + 1) if t not in hard]
        # scale yes: each pair ramps to its own depth. Under arithmetic
        # mixing a mixed light-backbone pair is re-mixed from its two ends,
        # so it ramps as the square root of the scale; set pairs (NBFIX)
        # ramp as the scale.
        adapt = (f"fix             ramp all adapt 1 "
                 f"{light_pair_terms('lj/cut/coul/long', 'epsilon', light, n_atom_types, 'scale')} "
                 f"scale yes reset no\n" if light else "")
        unadapt = "unfix           ramp\n" if light else ""
        run_steps = self._hard_ramp_steps(s2)
        body = f"""# LAMMPS Stage 2: epsilon ramp with the backbone at full depth (CHARMM, protocol "hard_backbone")
#
# CHARMM LJ without 1-4 terms (lj/cut/coul/long, arithmetic mixing, NBFIX);
# pairs with a light atom ramp from 0.001 to 1 of their depth, backbone
# pairs (types {' '.join(map(str, hard))}) are at full depth from the first
# step. No soft pre-minimisation: the historic CHARMM ramp opens with one.

{self._charmm_header("charmm")}pair_style      lj/cut/coul/long 12.0
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-4

read_data       system_after_soft.data
include         {c['soft']}
include         {c['lj']}

neigh_modify    one 10000
comm_modify     mode single cutoff 12.0

{self._dump_open('stage2', hard)}variable        scale equal ramp(0.001,1.0)
{adapt}fix             lang all langevin {p['temperature']} {p['temperature']} {p['tdamp']} {s2['langevin_seed']}
fix             cap all nve/limit {s2['limit']}
timestep        1.0
thermo          1000
# RUNTIME: {run_steps} steps
run             {run_steps}
unfix           cap
unfix           lang
{unadapt}{self._dump_close()}write_data      system_ramped.data nocoeff
"""
        with open(os.path.join(self.sim_dir, "minimize_2_parallel.in"), "w") as f:
            f.write(body)

    def write_equilibration_sequence(self, settings_file="system.in.settings", model_type="atomistic"):
        """
        Generates a 10-step Equilibration Sequence (Atomistic).
        1-4: Annealing (1000K -> 300K)
        5-6: 300K Equilibration (1M steps)
        7-8: 373K Equilibration (1M steps)
        9-10: 800K Equilibration (1M steps)
        """
        if model_type == 'cg':
            self._write_cg_minimization_equil(settings_file)
            print(f"Generated CG minimization scripts (Stages 2 & 3).")
            return

    def _write_cg_minimization_equil(self, settings_file):
        """
        Generates Stage 2 & 3 scripts for CG model using Reference Logic (Harmonic Ramp).
        Stage 2: Harmonic Ramp (minimize_2_cg.in) - Switch to Harmonic for stability
        Stage 3: Equilibration (minimize_3_cg.in) - Switch back to FENE

        Under ``hardcore_min`` the two soft steps -- the soft pre-minimisation
        and the epsilon ramp -- are left out and WCA runs throughout. A soft
        core has finite energy at zero separation, so during the ramp two
        beads may sit on top of one another at bounded cost and chains pass
        through each other; that is exactly the move a prescribed
        entanglement cannot survive.
        """
        hardcore = self.protocol == 'hardcore_min'

        # --- Stage 2: Harmonic Ramp Minimization ---
        script_path = os.path.join(self.sim_dir, "minimize_2_parallel.in")

        with open(script_path, 'w') as f:
            if hardcore:
                f.write("# LAMMPS Stage 2: CG hard core, no ramp (protocol \"hardcore_min\")\n\n")
            else:
                f.write("# LAMMPS Stage 2: CG Harmonic Ramp (Reference Logic)\n\n")

            # Read Restart from Stage 1 (Preserves state)
            f.write("read_restart    1.restart\n\n")

            f.write("neighbor        2.0 bin\n")
            f.write("neigh_modify    every 1 delay 0 check yes\n")
            f.write("comm_modify     mode single cutoff 5.0\n\n")

            f.write("# SWITCH TO HARMONIC for Robust Minimization\n")
            f.write("bond_style      harmonic\n")
            f.write("bond_coeff      1 466.1 0.97\n")
            if self.config.get('include_angles', True):
                f.write("angle_style     harmonic\n")
                f.write("angle_coeff     1 466.1 180.0\n\n") # Generic stiff angle

            if not hardcore:
                f.write("# Soft to Real Potential Ramp\n")
                f.write("pair_style      soft 1.0\n")
                f.write("pair_coeff      * * 1.0\n")
            f.write("min_style       cg\n")
            f.write("minimize        1e-4 1e-6 1000 10000\n\n")

            # Switch to Real LJ
            pair_style = self.config.get('pair_style', 'attractive')
            pair_cutoff = 1.122462 if (hardcore or pair_style == 'repulsive') else 2.5
            f.write(f"pair_style      lj/cut {pair_cutoff}\n")
            f.write(f"pair_coeff      * * 1.0 1.0 {pair_cutoff}\n")

            # Ramp parameters from config
            cg_ramp = self.experimental.get('cg', {}).get('ramp', {})
            scale_min = cg_ramp.get('epsilon_scale_start', 0.001)
            scale_max = cg_ramp.get('epsilon_scale_end', 1.0)
            nve_limit = cg_ramp.get('nve_limit', 0.1)
            ramp_steps = cg_ramp.get('ramp_steps', 20000)

            if not hardcore:
                f.write(f"variable        scale equal \"ramp({scale_min}, {scale_max})\"\n")
                f.write(f"fix             1 all adapt 1 pair lj/cut epsilon * * v_scale\n")
            f.write(f"fix             fxnve all nve/limit {nve_limit}\n")
            f.write("thermo          1000\n")
            f.write(f"run             {ramp_steps}\n")
            f.write("unfix           fxnve\n")
            if not hardcore:
                f.write("unfix           1\n")
            f.write("\n")

            f.write("write_restart   2.restart\n")
            f.write("write_data      system_ramped.data\n")

        # --- Stage 3: FENE Equilibration ---
        script_path = os.path.join(self.sim_dir, "minimize_3_parallel.in")
        
        with open(script_path, 'w') as f:
            f.write("# LAMMPS Stage 3: CG Equilibration (FENE Restore)\n\n")
            
            f.write("read_restart    2.restart\n\n")
            
            f.write("neighbor        2.0 bin\n")
            f.write("neigh_modify    every 1 delay 0 check yes\n\n")

            f.write("# --- Phase A: Harmonic Pre-Minimization ---\n")
            f.write("bond_style      harmonic\n")
            f.write("bond_coeff      1 466.1 0.97\n")
            if self.config.get('include_angles', True):
                f.write("angle_style     harmonic\n")
                f.write("angle_coeff     1 466.1 180.0\n")
            pair_style = self.config.get('pair_style', 'attractive')
            pair_cutoff = 1.122462 if (hardcore or pair_style == 'repulsive') else 2.5
            f.write(f"pair_style      lj/cut {pair_cutoff}\n")
            f.write(f"pair_coeff      * * 1.0 1.0 {pair_cutoff}\n")
            f.write("min_style       cg\n")
            f.write("minimize        1.0e-4 1.0e-6 1000 10000\n\n")

            # --- Phase B: Switch to FENE ---
            f.write("# --- Phase B: Switch to FENE ---\n")
            
            # Check config for angle removal (Default: True)
            # Only remove if they were included to begin with
            include_angles = self.config.get('include_angles', True)
            remove_angles = self.config.get('remove_cg_angles', True)
            
            if include_angles and remove_angles:
                f.write("delete_bonds    all angle 1-1 remove\n")
            
            f.write("bond_style      fene\n")
            f.write("special_bonds   fene\n")
            f.write("bond_coeff      1 30.0 1.5 1.0 1.0\n")
            f.write("minimize        1.0e-4 1.0e-6 1000 10000\n\n")
            
            f.write("# --- Phase C: Dynamics ---\n")
            # Get parameters from experimental config
            cg_dyn = self.experimental.get('cg', {}).get('dynamics', {})
            timestep = cg_dyn.get('timestep', 0.005)
            temp = cg_dyn.get('temperature', 1.0)
            tdamp = cg_dyn.get('tdamp', 1.0)
            thermo = cg_dyn.get('thermo_freq', 1000)
            run_steps = self._get_run_steps('cg')
            
            f.write("reset_timestep  0\n")
            f.write(f"timestep        {timestep}\n")
            f.write(f"velocity        all create {temp} 12345\n\n")
            f.write(f"fix             1 all nvt temp {temp} {temp} {tdamp}\n")
            f.write(f"thermo          {thermo}\n")
            f.write(f"run             {run_steps}\n")
            f.write("unfix           1\n\n")
            f.write("write_data      system_equilibrated.data\n")


        print(f"Generated parallel minimization scripts (CG Stages 2 & 3).")