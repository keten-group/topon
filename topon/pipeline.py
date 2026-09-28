"""
Main pipeline orchestrator for Topon.

Coordinates all stages of the polymer network generation process:
  1. Topology  — generate via C executable or load from .nodes/.edges/.gpickle
  2. Analysis  — graph statistics, defect/entanglement capacity
  3. Assignment — node/edge types, DP, defects, entanglements
  4. Chemistry  — build RDKit molecular structure
  5. Conformation — place atoms, resolve overlaps
  6. Output    — write LAMMPS data + input scripts

Usage::

    from topon.config import load_config_full
    from topon.pipeline import Pipeline

    config, raw = load_config_full("demos/polymer/coarse_grained/basic/config.json")
    pipe = Pipeline(config, raw_config=raw)
    pipe.run()

NOTE: The topology stage with ``source="generate"`` uses the C subprocess
generator when ``topology.generator.exe_path`` is set (faster), and the
pure-Python ``PythonTopologyGenerator`` otherwise (no compiler required).
Both run either search (strict or exact); an exact request that forces
double edges or reserves capacity for the defects stage stays on Python.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from topon.config.schema import ToponConfig


def _stable_rng(*parts) -> "np.random.Generator":
    """A generator seeded from ``parts``, the same in every process.

    ``hash()`` of anything containing a string is salted by
    ``PYTHONHASHSEED``, so a seed derived from it changes run to run even
    with ``random.seed`` and ``np.random.seed`` both set. This hashes the
    parts explicitly instead, which keeps a loop's ring and a sol chain's
    position reproducible and independent of the order the graph is walked
    in.
    """
    import hashlib

    key = "|".join(repr(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _local_perp_unit(backbone_xyz, k: int, fallback_unit, rand_vec) -> np.ndarray:
    """Return a unit vector for a graft to stick out perpendicular to the
    backbone at index ``k``, biased *outward* on kinked sections.

    Algorithm:
      1. Tangent T = central difference (P[k+1] - P[k-1]); endpoint cases
         use one-sided differences. Falls back to the per-edge chord unit
         when the backbone has fewer than 2 atoms.
      2. Curvature direction K = P[k+1] - 2*P[k] + P[k-1] (points *into*
         the bend, i.e. toward the centre of curvature).
      3. If K is large enough (the chain is genuinely curving — entangled
         kinks always trip this), the graft direction is the **outward**
         normal -K, projected to be perpendicular to T. This places grafts
         on the convex side of the bend so they never dive into the chain.
      4. If K is tiny (a straight backbone), fall back to the per-edge
         random ``rand_vec`` projected perpendicular to T (same answer as
         the pre-2026-05-11 chord-perpendicular behaviour on straight
         chains).
    """
    n = len(backbone_xyz)
    if n < 2:
        local_unit = fallback_unit
        curv = None
    else:
        if k <= 0:
            tangent = np.asarray(backbone_xyz[1]) - np.asarray(backbone_xyz[0])
            curv = None  # no second-derivative at endpoint
        elif k >= n - 1:
            tangent = np.asarray(backbone_xyz[-1]) - np.asarray(backbone_xyz[-2])
            curv = None
        else:
            tangent = np.asarray(backbone_xyz[k + 1]) - np.asarray(backbone_xyz[k - 1])
            curv = (
                np.asarray(backbone_xyz[k + 1])
                - 2.0 * np.asarray(backbone_xyz[k])
                + np.asarray(backbone_xyz[k - 1])
            )
        tn = float(np.linalg.norm(tangent))
        local_unit = tangent / tn if tn > 1e-9 else fallback_unit

    # Outward normal from curvature, if the chain bends enough at this point.
    if curv is not None:
        cn = float(np.linalg.norm(curv))
        if cn > 1e-3:
            outward = -curv / cn
            # Project out the tangent component so the graft is strictly perp.
            outward = outward - np.dot(outward, local_unit) * local_unit
            on = float(np.linalg.norm(outward))
            if on > 1e-6:
                return outward / on

    # Straight (or near-straight) backbone: random perp from rand_vec.
    perp = np.cross(local_unit, rand_vec)
    pn = float(np.linalg.norm(perp))
    if pn < 1e-6:
        perp = np.cross(local_unit, np.array([1.0, 0.0, 0.0]))
        pn = float(np.linalg.norm(perp))
        if pn < 1e-6:
            perp = np.cross(local_unit, np.array([0.0, 1.0, 0.0]))
            pn = float(np.linalg.norm(perp))
    return perp / (pn + 1e-12)


class Pipeline:
    """
    Main pipeline for polymer network generation.

    Parameters
    ----------
    config : ToponConfig
        Validated config object from :func:`topon.config.load_config_full`.
    raw_config : dict, optional
        Raw JSON dict for sections not yet covered by the Pydantic schema
        (keys: ``simulation``, ``execution``, ``experimental``). A copy of
        ``conformation`` is passed here too, though it is a schema section
        since 0.2.0, because this class and both workflow modules have always
        read it from here.

    Reproducibility
    ---------------
    ``Pipeline`` does not seed the random streams. The topology generators draw
    from the global ones, documented at
    ``topon/topology/generator_python.py:564``, so a build is reproducible only
    if the caller sets ``random.seed(n)`` and ``np.random.seed(n)`` first.
    ``topon.workflows.cg_network.run(seed=...)`` does; a direct
    ``Pipeline(config).run()`` does not, and two runs of the same config then
    give different graphs -- measured on SC 5x5x5 at neighbour_cutoff 1.5, 261
    edges and then 279. With the global streams seeded the graph and its edge
    order are identical, which is what makes a chain index mean the same strand
    twice (``topon.conformation.strand_plans``, and the ``pairs`` of a
    conformation config).
    """

    # Overlap-resolver defaults. These live in ConformationConfig now, and
    # this dict is the floor under a caller that passes neither a schema
    # section nor a raw one.
    _DEFAULT_CONFORMATION = {
        "overlap_cutoff": 0.01,
        "overlap_max_iters": 10,
        "noise_magnitude": 1e-4,
    }

    def __init__(self, config: ToponConfig, raw_config: Optional[dict] = None):
        self.config = config
        self.raw_config = raw_config or {}

        self.graph = None
        self.dims: Optional[np.ndarray] = None
        self.analysis_report: Optional[dict] = None
        self.chemical_space = None
        self._builder = None
        self._assignment_manager = None
        #: Stage 1's section of the run manifest, filled as it runs.
        self._topology_manifest: dict = {}

        self.output_dir = Path(config.study.output_dir) / config.study.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the complete pipeline end-to-end."""
        print(f"=== Topon Pipeline: {self.config.study.name} ===")
        print(f"Output directory: {self.output_dir}")
        print()

        self._run_topology_stage()
        self._run_analysis_stage()
        self._run_assignment_stage()
        self._run_chemistry_stage()
        self._run_conformation_stage()
        self._run_output_stage()

        print()
        print("=== Pipeline Complete ===")

    def run_from_graph(
        self,
        graph,
        dims,
    ) -> None:
        """Run only chemistry + conformation + output stages.

        Use when the graph is already fully prepared (e.g. loaded from a
        graphml / npz dual-graph file via
        :func:`topon.topology.loader.load_graphml` /
        :func:`topon.topology.loader.load_npz`) and stages 1-3 (topology,
        analysis, assignment) should be skipped.

        For deterministic output across different sources of the same
        topology (e.g. graphml-load vs npz-load), seed both
        :mod:`random` and :mod:`numpy.random` before calling this. The
        chemistry stage uses ``np.random.randn`` for graft-perp
        directions, and the conformation stage uses noise from
        ``numpy.random`` -- without seeding, byte-equivalence cannot be
        guaranteed.

        Args:
            graph: NetworkX MultiGraph with crosslink nodes (``pos``) and
                chain edges (``dp``, ``entangled_with``,
                ``entanglement_count``). Edge / node ``type`` attributes
                are optional and default to ``"A"``.
            dims: Box dimensions array ``[Lx, Ly, Lz]``.
        """
        self.graph = graph
        self.dims = (
            np.asarray(dims, dtype=float)
            if not isinstance(dims, np.ndarray) else dims.astype(float)
        )
        print(f"=== Topon Pipeline (rebuild from graph): "
              f"{self.config.study.name} ===")
        print(f"Output directory: {self.output_dir}")
        print(f"  Skipping stages 1-3 (graph supplied directly).")
        print(f"  Nodes: {self.graph.number_of_nodes()}, "
              f"Edges: {self.graph.number_of_edges()}")
        print()

        self._run_chemistry_stage()
        self._run_conformation_stage()
        self._run_output_stage()

        print()
        print("=== Pipeline Complete (rebuild) ===")

    # ------------------------------------------------------------------
    # Stage 1: Topology
    # ------------------------------------------------------------------

    def _run_topology_stage(self) -> None:
        print("--- Stage 1: Topology ---")
        self._topology_manifest = {"source": self.config.topology.source}
        started = time.time()
        if self.config.topology.source == "generate":
            self._generate_topology()
        else:
            self._load_existing_topology()
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        self._record_topology_manifest(time.time() - started)
        print()

    def _record_topology_manifest(self, seconds: float) -> None:
        """Write stage 1's section of the run manifest.

        What was asked of the topology and what came back: the lattice,
        the search that sculpted it, the degree counts requested against
        the ones achieved, and the timing. ``topon inspect`` renders it.
        Writing the manifest must never take a run down, so a filesystem
        that refuses the file is reported and stepped over.
        """
        from topon.core.manifest import record_stage

        entry = dict(self._topology_manifest)
        entry["seconds"] = round(float(seconds), 3)
        entry["nodes"] = self.graph.number_of_nodes()
        entry["edges"] = self.graph.number_of_edges()
        if self.dims is not None:
            entry["box"] = [float(x) for x in np.asarray(self.dims).ravel()]
        try:
            record_stage(
                self.output_dir, "topology", entry,
                study=self.config.study.name,
            )
        except Exception as exc:
            # Deliberately broad: the manifest is a record of the run, not
            # part of one, so a read-only directory or a value json cannot
            # encode must cost an inspection detail and nothing more.
            print(f"  (could not write the run manifest: {exc})")

    def _generate_topology(self) -> None:
        import networkx as nx

        from topon.topology.generator import run_generator
        from topon.topology.generator_python import PythonTopologyGenerator
        from topon.topology.loader import (
            infer_dims_from_graph,
            load_graph,
            remove_vacancies,
        )

        gen_cfg = self.config.topology.generator
        topology_dir = self.output_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)

        self._topology_manifest.update(
            lattice_type=gen_cfg.lattice_type,
            lattice_size=gen_cfg.lattice_size,
            neighbour_cutoff=gen_cfg.neighbour_cutoff,
            periodicity=gen_cfg.periodicity,
            max_functionality=gen_cfg.max_functionality,
            degree_distribution=gen_cfg.degree_distribution,
        )
        if gen_cfg.lattice_type == "MIX":
            self._topology_manifest["mix_fractions"] = dict(gen_cfg.mix_fractions)

        # Which sculptor runs is decided before the generator is chosen.
        # Both searches exist in both generators, so a configured
        # `exe_path` takes the C binary for either, with one exception: an
        # exact request that pays for the defects stage in advance (forced
        # double edges for secondary loops, capacity reserved for
        # triangles and four-cycles) stays on Python, because only the
        # Python exact search takes those and the binary has no channel
        # for them.
        search = self._resolve_search(gen_cfg)
        self._topology_manifest["search"] = search

        use_c = bool(gen_cfg.exe_path)
        double_pairs = reserved = None
        if not use_c or search == "exact":
            double_pairs, reserved = self._sculpt_request(gen_cfg, search)
        if use_c and (double_pairs or reserved):
            print("  exe_path is set, but this exact request forces double "
                  "edges or reserves capacity for the defects stage, which "
                  "only the Python exact search does; using the Python "
                  "generator for this run.")
            use_c = False

        if use_c:
            # C subprocess path: writes <topology_dir>/output/*.nodes + *.edges,
            # then re-load via the standard loader.
            self._topology_manifest["generator"] = "c"
            # The Python exact search draws its seed from the global NumPy
            # stream, so np.random.seed(n) pins it; hand the binary a seed
            # drawn the same way so the C route is just as reproducible.
            seed = (int(np.random.randint(0, 2 ** 31 - 1))
                    if search == "exact" else None)
            nodes_path, edges_path = run_generator(
                gen_cfg, topology_dir, exe_path=gen_cfg.exe_path, seed=seed
            )
            self._record_c_sculpt(nodes_path, gen_cfg, search, seed)
            self.graph, self.dims = load_graph(
                nodes_path=str(nodes_path),
                edges_path=str(edges_path),
            )
        else:
            # Pure-Python path: in-memory graph, no file round-trip.
            self._topology_manifest["generator"] = "python"
            gen = PythonTopologyGenerator(gen_cfg)
            graphs = gen.generate(
                trials=gen_cfg.max_trials,
                max_saves=gen_cfg.max_saves,
                double_pairs=double_pairs,
                need=reserved,
            )
            if not graphs:
                raise RuntimeError(
                    f"PythonTopologyGenerator produced no graphs after "
                    f"{gen_cfg.max_trials} trials (constraints may be too "
                    f"strict for lattice {gen_cfg.lattice_size} "
                    f"with degree_distribution={gen_cfg.degree_distribution!r})."
                )
            G = graphs[0]
            self._record_sculpt(G, gen)
            if not isinstance(G, nx.MultiGraph):
                G = nx.MultiGraph(G)
            remove_vacancies(G)
            if getattr(self, "_reserved_capacity", False):
                # The defects stage reads this to know that the junctions it
                # raises were sculpted one degree low on purpose, so raising
                # them is not a deviation from the requested P(f).
                G.graph["reserved_capacity"] = True
            self.graph = G
            self.dims = infer_dims_from_graph(G)

    def _sculpt_request(self, gen_cfg, search: str):
        """What the defects stage needs the sculpt to do for it.

        Returns ``(double_pairs, need)``. Secondary loops are forced into
        the sculpt as double edges, which is the only way to place them
        with the degree counts staying exact; triangles and four-cycles
        each raise two junctions by one, so the sculpt target is reduced
        by that much and the injector raises exactly those back. Both are
        exact-search only: the strict sculptor has neither facility, and a
        config that asks for them with ``search="strict"`` is told so.
        """
        from topon.assignment import defects as defect_rules
        from topon.topology.degree_matching import needs_from_targets, parse_degree_distribution

        cfg = self.config.assignment.defects
        sec, tri, quad = cfg.secondary_loops, cfg.triangles, cfg.four_cycles
        if not (sec.enabled or tri.enabled or quad.enabled):
            return None, None

        # The search is checked first: only an exact request pins every
        # degree, and building the table from a partial distribution (the
        # strict sculptor's usual input) raised KeyError on the first degree
        # it did not list.
        targets, _ = parse_degree_distribution(gen_cfg.degree_distribution)
        need = (needs_from_targets(targets, gen_cfg.max_functionality)
                if search == "exact" and targets else {})
        if not need:
            print("  [note] secondary loops and higher-order defects are placed "
                  "by the exact search; with this topology they are left to the "
                  "defects stage, which injects them afterwards and reports the "
                  "P(f) deviation.")
            return None, None

        n_strands = sum(d * n for d, n in need.items()) // 2
        double_pairs = None
        if sec.enabled:
            count = defect_rules.resolve_count(
                sec.count if sec.count is not None else sec.target, n_strands)
            if count:
                double_pairs = (
                    defect_rules.auto_double_pairs(need, count,
                                                   gen_cfg.max_functionality)
                    if sec.endpoint_degrees == "auto"
                    else defect_rules.plan_double_pairs(sec.endpoint_degrees, count)
                )
        reserved = None
        n_tri = defect_rules.resolve_count(tri.count, n_strands) if tri.enabled else 0
        n_quad = defect_rules.resolve_count(quad.count, n_strands) if quad.enabled else 0
        if n_tri or n_quad:
            reserved = defect_rules.reserve_capacity(
                need, n_triangles=n_tri, n_four_cycles=n_quad,
                max_f=gen_cfg.max_functionality,
            )
            self._reserved_capacity = True
            print(f"  Reserved capacity for {n_tri} triangles and {n_quad} "
                  f"four-cycles: {2 * (n_tri + n_quad)} junctions sculpted one "
                  f"degree below {gen_cfg.max_functionality}")
        if double_pairs:
            print(f"  Forcing {sum(double_pairs.values())} double edges into the "
                  f"sculpt: {dict(sorted(double_pairs.items()))}")
        return double_pairs, reserved

    @staticmethod
    def _resolve_search(gen_cfg) -> str:
        """Which sculptor this generator config asks for.

        Reads the same rule the Python generator uses, before a lattice
        is built, so the pipeline can pick the generator and the manifest
        can name the search whichever path runs. The C wrapper resolves
        with the same function, so the binary is told the same search.
        """
        from topon.topology.generator import resolve_config_search

        return resolve_config_search(gen_cfg)

    def _record_sculpt(self, G, gen) -> None:
        """Copy the generator's degree bookkeeping into the manifest entry.

        Requested against achieved counts either way: the exact search
        hands over a full record (timing, seeds, augmentations, giant
        fraction), and for the strict sculptor the same two rows are read
        off the target spec and the graph it returned.
        """
        from topon.topology.degree_matching import counts_by_degree

        achieved = counts_by_degree(G, gen.max_func)
        record = G.graph.get("sculpt_record")
        if record is not None:
            entry = {k: v for k, v in record.items() if k != "doubles"}
            entry["achieved"] = achieved
            self._topology_manifest["sculpt"] = entry
            return
        requested = {
            d: int(gen.target_counts[d])
            for d in range(0, gen.max_func + 1)
            if gen.target_counts.get(d, -2) >= 0
        }
        self._topology_manifest["sculpt"] = {
            "requested": requested,
            "achieved": achieved,
            "n_sites": G.number_of_nodes(),
            "n_vacancies": achieved.get(0, 0),
            "target_edges": (gen.target_edge_count
                             if gen.target_edge_count >= 0 else None),
        }

    def _record_c_sculpt(self, nodes_path, gen_cfg, search: str, seed) -> None:
        """The same requested-against-achieved rows for a C run.

        Read off the ``.nodes`` file the binary wrote, whose degree column
        still holds the vacancies the loader is about to drop. The C run
        hands back no search record, so the rows are the counts, the site
        numbers and, on the exact route, the seed that replays it.
        """
        from topon.topology.degree_matching import parse_degree_distribution

        max_f = int(gen_cfg.max_functionality)
        counts, edge_count = parse_degree_distribution(gen_cfg.degree_distribution)
        degrees = np.atleast_1d(np.loadtxt(nodes_path, comments="#", usecols=4, dtype=int))
        hist = np.bincount(degrees, minlength=max_f + 1) if degrees.size else np.zeros(max_f + 1, int)
        achieved = {d: int(n) for d, n in enumerate(hist)}
        entry = {
            "requested": {d: int(n) for d, n in sorted(counts.items())
                          if 0 <= d <= max_f and n >= 0},
            "achieved": achieved,
            "n_sites": int(degrees.size),
            "n_vacancies": achieved.get(0, 0),
            "target_edges": edge_count if edge_count >= 0 else None,
        }
        if seed is not None:
            entry["seed"] = int(seed)
        self._topology_manifest["sculpt"] = entry

    def _load_existing_topology(self) -> None:
        from topon.topology.loader import load_graph

        files = self.config.topology.existing_files
        if files.gpickle_file:
            self.graph, self.dims = load_graph(gpickle_path=files.gpickle_file)
        elif files.nodes_file and files.edges_file:
            self.graph, self.dims = load_graph(
                nodes_path=files.nodes_file,
                edges_path=files.edges_file,
            )
        else:
            raise ValueError(
                "No topology files specified. Provide gpickle_file or "
                "both nodes_file and edges_file."
            )

    # ------------------------------------------------------------------
    # Stage 2: Analysis
    # ------------------------------------------------------------------

    def _junction_valence_ceiling(self) -> int:
        """How many strands one junction may carry.

        The topology's own ceiling, except on the atomistic route, where a
        single-atom Si junction has four bonds: RDKit rejects degree-5 and
        degree-6 Si and Gasteiger then emits NaN on the over-valent atom
        and its neighbours, leaving a net charge PPPM cannot tune. This is
        the guard the defect injector carried hard-coded before 0.2.0.
        """
        ceiling = int(self.config.topology.generator.max_functionality)
        if self.config.chemistry.model_type == "atomistic":
            # Four is the single-atom Si junction, which is what the default
            # node_type_map builds. A POSS cage carries eight, so this
            # under-caps a POSS network; deriving it from
            # chemistry.node_type_map would be the proper fix.
            return min(ceiling, 4)
        return ceiling

    def _run_analysis_stage(self) -> None:
        print("--- Stage 2: Analysis ---")
        from topon.assignment.manager import AssignmentManager

        self._assignment_manager = AssignmentManager(
            self.graph, self.dims, self.config.assignment,
            max_functionality=self._junction_valence_ceiling(),
        )
        self.analysis_report = self._assignment_manager.analyze()
        print()

    # ------------------------------------------------------------------
    # Stage 3: Assignment
    # ------------------------------------------------------------------

    def _run_assignment_stage(self) -> None:
        print("--- Stage 3: Assignment ---")
        self._assignment_manager.run()
        print()

    # ------------------------------------------------------------------
    # Stage 4: Chemistry
    # ------------------------------------------------------------------

    def _lattice_bond(self, node=None) -> float:
        """Bond length in lattice units, as the other strands here use it.

        A primary loop has no chord to take its scale from, so it borrows
        the one the strands leaving the same junction were drawn with:
        chord over bonds. Falls back to the graph's median, then to a
        fraction of the cell, so a junction whose only strand is the loop
        still gets a sensible ring.
        """
        import numpy as _np

        def bonds_of(edges):
            out = []
            for u, v, data in edges:
                if u == v:
                    continue
                pu = self.graph.nodes[u].get("pos")
                pv = self.graph.nodes[v].get("pos")
                if pu is None or pv is None:
                    continue
                vec = _np.asarray(pv, float) - _np.asarray(pu, float)
                if self.dims is not None:
                    vec = vec - self.dims * _np.round(vec / self.dims)
                n_bonds = int(data.get("dp", 25)) + 1
                out.append(float(_np.linalg.norm(vec)) / max(n_bonds, 1))
            return out

        if node is not None:
            local = bonds_of(self.graph.edges(node, data=True))
            if local:
                return float(_np.median(local))
        if getattr(self, "_median_bond", None) is None:
            allb = bonds_of(self.graph.edges(data=True))
            self._median_bond = float(_np.median(allb)) if allb else None
        if self._median_bond:
            return self._median_bond
        cell = float(min(self.dims)) if self.dims is not None else 1.0
        return cell / 26.0

    def _loop_backbone_xyz(self, node, n_beads: int):
        """Where a primary loop's beads sit: a ring closed on its junction."""
        import numpy as _np

        from topon.conformation.paths import closed_meander

        pos = _np.asarray(self.graph.nodes[node].get("pos", (0.0, 0.0, 0.0)), float)
        away = []
        for u, v, data in self.graph.edges(node, data=True):
            if u == v:
                continue
            other = v if u == node else u
            po = self.graph.nodes[other].get("pos")
            if po is None:
                continue
            vec = _np.asarray(po, float) - pos
            if self.dims is not None:
                vec = vec - self.dims * _np.round(vec / self.dims)
            away.append(vec)
        return closed_meander(
            pos, n_bonds=n_beads + 1, bond=self._lattice_bond(node),
            rng=_stable_rng("loop", self.config.study.name, node, n_beads),
            away_from=away or None,
        )

    def _sol_backbone_coords(self) -> dict:
        """Where the sol chains sit: free walks dropped anywhere in the cell."""
        import numpy as _np

        from topon.conformation.paths import free_walk

        coords: dict = {}
        chains = getattr(self._builder, "sol_atom_map", None) or []
        if not chains:
            return coords
        rng = _stable_rng("sol", self.config.study.name, len(chains))
        box = _np.asarray(self.dims, float) if self.dims is not None else _np.ones(3)
        bond = self._lattice_bond()
        for atoms in chains:
            start = rng.random(3) * box
            path = free_walk(start, n_bonds=max(len(atoms) - 1, 1), bond=bond, rng=rng)
            for idx, xyz in zip(atoms, path):
                coords[idx] = tuple(xyz)
        print(f"  Placed {len(chains)} sol chains")
        return coords

    def _run_chemistry_stage(self) -> None:
        print("--- Stage 4: Chemistry ---")
        from topon.chemistry.builder import ChemistryBuilder
        from topon.writers import CGWriter, DreidingWriter
        from topon.utils import write_lammps_displacement_file
        from topon.utils.network_helpers import (
            generate_approximate_side_chain_coords,
        )
        from topon.conformation.entanglement.realize import (
            entangled_backbone_paths,
        )

        self._builder = ChemistryBuilder(
            self.graph, self.dims, self.config.chemistry
        )
        self.chemical_space = self._builder.build()

        chem_dir = self.output_dir / "02_Chemistry"
        chem_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(chem_dir / "system.data")

        model = self.config.chemistry.model_type
        density = self.config.chemistry.target_density

        if model == "coarse_grained":
            # Honour raw_config's simulation.include_angles flag (default True
            # to preserve historic behaviour). Mirrors the topon.workflows.
            # cg_network call pattern and the schema doc-string in
            # topon/chemistry/kg/__init__.py.
            sim_cfg_for_writer = self.raw_config.get("simulation", {})
            include_angles = sim_cfg_for_writer.get("include_angles", True)
            writer = CGWriter(
                self.chemical_space, data_path,
                include_angles=include_angles,
                convention=self.config.output.lammps_convention,
            )
            writer.write()
            # Count-based volume for CG (matches the cg_network workflow).
            n_atoms = self.chemical_space.GetNumAtoms()
            vol = n_atoms / density
        elif self.config.chemistry.force_field == "charmm":
            mol_h, vol = self._write_charmm_chemistry(data_path, density)
        else:
            # Atomistic: mirror topon.workflows.atomistic_network's tail —
            # the canonical path that produces healthy LAMMPS stage 2/3
            # output. ChemistryBuilder.build() returns
            # a heavy-atom-only RWMol; we Sanitize -> AddHs -> Gasteiger so
            # the data file (a) has H_ atom-type rows DREIDING needs, and
            # (b) is charge-neutral so PPPM auto-gewald doesn't crash. AddHs
            # preserves heavy-atom indices, so _builder.node_map and
            # edge_atom_map remain valid. Sanitize can fail on demos that
            # produce over-valent atoms (e.g. defect demo's degree-6 Si);
            # fall back to writing the heavy-atom mol uncharged in that
            # case — system.data is then DREIDING-incomplete (no H_) but
            # the chemistry stage still completes for inspection.
            from rdkit import Chem
            from rdkit.Chem import AllChem
            try:
                try:
                    Chem.SanitizeMol(self.chemical_space)
                except Chem.AtomValenceException:
                    # Defect demos can produce degree-6 Si junctions that
                    # exceed RDKit's permitted valence (max 6 by table, but
                    # the strict check trips on Si@6 with no charge). Skip
                    # just the valence-property check; the rest of sanitize
                    # (kekulize, ring-find, etc.) still runs. AddHs then
                    # assigns 0 implicit H to those Si atoms.
                    Chem.SanitizeMol(
                        self.chemical_space,
                        sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        ),
                    )
                mol_h = Chem.AddHs(self.chemical_space)
                AllChem.ComputeGasteigerCharges(mol_h)
                # Over-valent atoms (defect's degree-6 Si) make Gasteiger
                # emit NaN for those atoms and their neighbours. Scrub to
                # 0 so the LAMMPS data file is numeric; the residual net
                # charge logged below is usually within PPPM's tolerance.
                import math
                nan_count = 0
                for atom in mol_h.GetAtoms():
                    if atom.HasProp("_GasteigerCharge"):
                        q = atom.GetDoubleProp("_GasteigerCharge")
                        if math.isnan(q) or math.isinf(q):
                            atom.SetDoubleProp("_GasteigerCharge", 0.0)
                            nan_count += 1
                if nan_count:
                    print(f"  [WARN] Gasteiger NaN/Inf on {nan_count} atoms "
                          f"(over-valent neighbours); zeroed.")

                # Background charge neutralization: redistribute residual net
                # charge uniformly across all atoms with a valid Gasteiger
                # value. Two things make this load-bearing:
                #   (1) The defect demo has over-valent Si (degree 5-6 from
                #       parallel-edge defects). NaN-zeroing those atoms can
                #       leave several e residual; PPPM then prints "System
                #       is not charge neutral" and runs slowly.
                #   (2) Gasteiger output itself has a tiny non-zero residual
                #       (~1e-12 e) from finite-precision iteration on every
                #       molecule. Cheap to scrub here.
                valid_atoms = [
                    a for a in mol_h.GetAtoms() if a.HasProp("_GasteigerCharge")
                ]
                total_q = sum(
                    a.GetDoubleProp("_GasteigerCharge") for a in valid_atoms
                )
                if valid_atoms and abs(total_q) > 1e-6:
                    delta = -total_q / len(valid_atoms)
                    for a in valid_atoms:
                        a.SetDoubleProp(
                            "_GasteigerCharge",
                            a.GetDoubleProp("_GasteigerCharge") + delta,
                        )
                    print(f"  Charge-neutralized: spread {-total_q:+.4f} e "
                          f"across {len(valid_atoms)} atoms "
                          f"(delta = {delta:+.2e} e/atom)")

                self.chemical_space = mol_h
                # Mass-based volume (matches the canonical workflow).
                mass = sum(a.GetMass() for a in mol_h.GetAtoms())
                vol = (mass / density) * 1.66054  # A^3 / Da at g/cm^3
                writer = DreidingWriter(mol_h, data_path, use_charges=True)
            except Exception as exc:
                print(f"  [WARN] AddHs/Gasteiger skipped ({exc}); "
                      f"writing heavy-atom data file uncharged.")
                mol_h = self.chemical_space
                n_atoms = mol_h.GetNumAtoms()
                vol = n_atoms / density
                writer = DreidingWriter(mol_h, data_path, use_charges=False)
            writer.write()

        scale = (vol / float(np.prod(self.dims))) ** (1.0 / 3.0)
        sx = sy = sz = scale

        # Nodes displacement (both CG and atomistic). POSS node_map values
        # can be lists (cage); preserve the isinstance branch.
        node_coords: dict[int, tuple] = {}
        for node, atom_ref in self._builder.node_map.items():
            pos = self.graph.nodes[node].get("pos", (0.0, 0.0, 0.0))
            primary_idx = atom_ref[0] if isinstance(atom_ref, (list, tuple)) else atom_ref
            node_coords[primary_idx] = tuple(pos)

        write_lammps_displacement_file(
            node_coords, sx, sy, sz,
            str(chem_dir / "system_nodes.displace"), "nodes"
        )

        # Entangled edges' paths, both branches, computed once. The method
        # is the config's: "waypoint" (default) draws each pair together
        # with a prescribed winding count; "kink" is the legacy Gaussian
        # bump. Edges not in this dict are linear, as they always were.
        ent_cfg = self.config.assignment.entanglements
        ent_paths = entangled_backbone_paths(
            self.graph, self.dims, self._builder.edge_atom_map,
            method=ent_cfg.method,
            kink_params=ent_cfg.kink_params.model_dump(),
        )

        if model == "coarse_grained":
            # CG: same backbone + grafts loop as atomistic (entanglement-aware
            # winding for backbone, perpendicular placement with 3-way length
            # cap for grafts). CG doesn't have pendant/H passes — graft beads
            # come directly from `_builder.graft_atom_map`.
            backbone_coords: dict[int, tuple] = {}
            graft_coords: dict[int, tuple] = {}
            graft_atom_map = getattr(self._builder, "graft_atom_map", {}) or {}
            ext_factor = 0.5

            for (u, v, key), atoms in self._builder.edge_atom_map.items():
                data = self.graph[u][v][key]
                pos_u = np.array(self.graph.nodes[u].get("pos", (0.0, 0.0, 0.0)))
                pos_v = np.array(self.graph.nodes[v].get("pos", (0.0, 0.0, 0.0)))
                vec = pos_v - pos_u
                mic = vec - self.dims * np.round(vec / self.dims)
                edge_len = float(np.linalg.norm(mic))
                unit_vec = mic / (edge_len + 1e-9)
                rand_vec = np.random.randn(3)
                perp = np.cross(unit_vec, rand_vec)
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(unit_vec, np.array([1.0, 0.0, 0.0]))
                perp_unit = perp / (np.linalg.norm(perp) + 1e-9)

                backbone_xyz = ent_paths.get((u, v, key))
                if backbone_xyz is None and u == v:
                    # A primary loop: no chord, so a ring closed on its own
                    # junction, sized by the bonds of the strands beside it.
                    backbone_xyz = self._loop_backbone_xyz(u, len(atoms))
                if backbone_xyz is None:
                    backbone_xyz = []
                    for j in range(len(atoms)):
                        frac = (j + 1) / (len(atoms) + 1)
                        backbone_xyz.append(pos_u + frac * mic)

                for j, a_idx in enumerate(atoms):
                    backbone_coords[a_idx] = tuple(backbone_xyz[j])

                # Grafts (CG): perp to the local backbone *tangent* at the
                # anchor (kinked backbones twist; chord-perp grafts would
                # otherwise dive back into the chain on entangled edges).
                edge_grafts = graft_atom_map.get((u, v, key))
                if edge_grafts:
                    lattice_spacing = (
                        float(min(self.dims)) if self.dims is not None else edge_len
                    )
                    for frac, g_atoms in edge_grafts:
                        k_float = frac * (len(atoms) + 1) - 1
                        k = max(0, min(int(round(k_float)), len(backbone_xyz) - 1))
                        anchor_pos = backbone_xyz[k]
                        local_perp = _local_perp_unit(
                            backbone_xyz, k, unit_vec, rand_vec
                        )
                        graft_dp_eff = max(len(g_atoms), 1)
                        backbone_dp = max(len(atoms), 1)
                        eff_factor = min(
                            ext_factor,
                            graft_dp_eff / backbone_dp,
                            0.5 * lattice_spacing / max(edge_len, 1e-9),
                        )
                        graft_vec = local_perp * (edge_len * eff_factor)
                        for m, g_idx in enumerate(g_atoms):
                            g_frac = (m + 1) / len(g_atoms)
                            graft_coords[g_idx] = tuple(anchor_pos + g_frac * graft_vec)

            backbone_coords.update(self._sol_backbone_coords())

            write_lammps_displacement_file(
                backbone_coords, sx, sy, sz,
                str(chem_dir / "system_backbone.displace"), "backbone"
            )
            write_lammps_displacement_file(
                graft_coords, sx, sy, sz,
                str(chem_dir / "system_grafts.displace"), "grafts"
            )
        else:
            # Atomistic: backbone + grafts + pendant + hydrogens. Backbone
            # path consults `entangled_with` on the (u, v, key) edge data
            # for kinked-chain placement (N+2 fix). graft_atom_map is
            # currently only populated by ChemistryBuilder._build_chain_cg
            # — for atomistic it's empty, so system_grafts.displace will be
            # empty and graft side-chain atoms instead get coords from the
            # pendant pass (neighbor propagation through mol_h).
            backbone_coords: dict[int, tuple] = {}
            graft_coords: dict[int, tuple] = {}
            graft_atom_map = getattr(self._builder, "graft_atom_map", {}) or {}
            ext_factor = 0.5  # canonical workflow default

            for (u, v, key), atoms in self._builder.edge_atom_map.items():
                data = self.graph[u][v][key]
                pos_u = np.array(self.graph.nodes[u].get("pos", (0.0, 0.0, 0.0)))
                pos_v = np.array(self.graph.nodes[v].get("pos", (0.0, 0.0, 0.0)))
                vec = pos_v - pos_u
                mic = vec - self.dims * np.round(vec / self.dims)
                edge_len = float(np.linalg.norm(mic))
                unit_vec = mic / (edge_len + 1e-9)
                rand_vec = np.random.randn(3)
                perp = np.cross(unit_vec, rand_vec)
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(unit_vec, np.array([1.0, 0.0, 0.0]))
                perp_unit = perp / (np.linalg.norm(perp) + 1e-9)

                backbone_xyz = ent_paths.get((u, v, key))
                if backbone_xyz is None and u == v:
                    # A primary loop: no chord, so a ring closed on its own
                    # junction, sized by the bonds of the strands beside it.
                    backbone_xyz = self._loop_backbone_xyz(u, len(atoms))
                if backbone_xyz is None:
                    backbone_xyz = []
                    for j in range(len(atoms)):
                        frac = (j + 1) / (len(atoms) + 1)
                        backbone_xyz.append(pos_u + frac * mic)

                for j, a_idx in enumerate(atoms):
                    backbone_coords[a_idx] = tuple(backbone_xyz[j])

                # Grafts: place perpendicular to the local backbone *tangent*
                # at the anchor (per-anchor finite difference, not per-edge
                # chord). On a kinked entangled chain the local tangent
                # curves away from the chord, so chord-perp grafts would
                # otherwise dive back into the chain. Length is the minimum
                # of three competing constraints:
                #   (a) extension_factor (default 0.5) — half the edge len
                #   (b) graft_dp / backbone_dp           — chain-length scaling
                #   (c) 0.5 * lattice_spacing / edge_len — never past half
                #       a lattice cell into neighbouring cells
                edge_grafts = graft_atom_map.get((u, v, key))
                if edge_grafts:
                    lattice_spacing = float(min(self.dims)) if self.dims is not None else edge_len
                    for frac, g_atoms in edge_grafts:
                        k_float = frac * (len(atoms) + 1) - 1
                        k = max(0, min(int(round(k_float)), len(backbone_xyz) - 1))
                        anchor_pos = backbone_xyz[k]
                        local_perp = _local_perp_unit(
                            backbone_xyz, k, unit_vec, rand_vec
                        )
                        graft_dp_eff = max(len(g_atoms), 1)
                        backbone_dp = max(len(atoms), 1)
                        eff_factor = min(
                            ext_factor,
                            graft_dp_eff / backbone_dp,
                            0.5 * lattice_spacing / max(edge_len, 1e-9),
                        )
                        graft_vec = local_perp * (edge_len * eff_factor)
                        for m, g_idx in enumerate(g_atoms):
                            g_frac = (m + 1) / len(g_atoms)
                            graft_coords[g_idx] = tuple(anchor_pos + g_frac * graft_vec)

            write_lammps_displacement_file(
                backbone_coords, sx, sy, sz,
                str(chem_dir / "system_backbone.displace"), "backbone"
            )
            write_lammps_displacement_file(
                graft_coords, sx, sy, sz,
                str(chem_dir / "system_grafts.displace"), "grafts"
            )

            # Pendant heavy + hydrogens via neighbor propagation through mol_h.
            known = {**node_coords, **backbone_coords, **graft_coords}
            side_coords = generate_approximate_side_chain_coords(mol_h, known)
            h_coords = {k: v for k, v in side_coords.items()
                        if mol_h.GetAtomWithIdx(k).GetSymbol() == "H"}
            p_coords = {k: v for k, v in side_coords.items()
                        if mol_h.GetAtomWithIdx(k).GetSymbol() != "H"}

            write_lammps_displacement_file(
                p_coords, sx, sy, sz,
                str(chem_dir / "system_pendant.displace"), "pendant"
            )
            write_lammps_displacement_file(
                h_coords, sx, sy, sz,
                str(chem_dir / "system_hydrogens.displace"), "hydrogens"
            )

        # Groups: nodes (junction atoms) vs beads (everything else).
        node_atom_ids = []
        for atom_ref in self._builder.node_map.values():
            if isinstance(atom_ref, (list, tuple)):
                node_atom_ids.extend(int(i) + 1 for i in atom_ref)
            else:
                node_atom_ids.append(int(atom_ref) + 1)
        node_atom_ids.sort()

        with open(chem_dir / "system.groups", "w") as fh:
            fh.write("# LAMMPS group definitions\n")
            fh.write(f"group nodes id {' '.join(str(x) for x in node_atom_ids)}\n")
            fh.write("group beads subtract all nodes\n")

        self._record_defects()

        settings_path = chem_dir / "system.in.settings"
        if not settings_path.exists():
            with open(settings_path, "w") as fh:
                fh.write("# Force field settings (auto-generated stub)\n")
        print()

    def _write_charmm_chemistry(self, data_path: str, density: float):
        """Type the network from its RTF residues and write the CHARMM files.

        Charges come from the RTF (no Gasteiger, no redistribution: a
        network of neutral residues is neutral term by term), every bonded
        and nonbonded term from the parameter files, and the run manifest
        records per-residue charges and the files read. A missing residue,
        atom or parameter stops the stage with the full list.
        """
        from rdkit import Chem
        from topon.chemistry.charmm import build_terms, load_parameters, type_network
        from topon.core.manifest import record_stage
        from topon.writers.lammps_charmm import CharmmWriter

        chem = self.config.chemistry
        if chem.charmm is None:
            raise ValueError("chemistry.force_field is 'charmm' but chemistry.charmm "
                             "(the RTF/PRM files) is missing")
        ps = load_parameters(chem.charmm.files)
        Chem.SanitizeMol(self.chemical_space)
        mol_h = Chem.AddHs(self.chemical_space)
        typing = type_network(mol_h, self._builder, chem, ps)
        terms = build_terms(mol_h, typing, ps)
        CharmmWriter(mol_h, typing, terms, data_path).write()

        # float sum of the RTF charges (type_network has checked it is an
        # integer); rounded past their last digit, and + 0.0 so a neutral
        # network does not print as -0.000000
        total = round(sum(typing.charge.values()), 9) + 0.0
        # distinct net charges per residue name (+ 0.0 turns -0.0 into 0.0)
        per_res = {name: sorted({round(q, 6) + 0.0 for q in qs})
                   for name, qs in typing.residue_charges().items()}
        print(f"  CHARMM: {len(typing.residues)} residues {per_res}, net charge "
              f"{total:+.6f} e, {len(terms.dihedral_types)} dihedral types, "
              f"wildcard terms {terms.wildcard_counts or 'none'}")
        record_stage(self.output_dir, "chemistry", {
            "force_field": "charmm",
            "files": [Path(p).name for p in ps.sources],
            "net_charge": total,
            "residue_charges": per_res,
            "n_residues": len(typing.residues),
            "wildcard_terms": dict(terms.wildcard_counts),
            "pair_style": chem.charmm.pair_style,
        })
        mass = sum(a.GetMass() for a in mol_h.GetAtoms())
        return mol_h, (mass / density) * 1.66054

    def _record_defects(self) -> None:
        """Put the defects stage's record in the run manifest.

        Requested against achieved for every defect class, the chemical
        against the effective P(f), and the bead budget that sized the
        box. ``topon inspect`` renders it beside stage 1's sculpt record.
        As with that one, a manifest that cannot be written costs an
        inspection detail and nothing more.
        """
        from topon.core.manifest import record_stage

        entry = self.graph.graph.get("defects") if self.graph is not None else None
        if not entry:
            return
        try:
            record_stage(
                self.output_dir, "defects", dict(entry),
                study=self.config.study.name,
            )
        except Exception as exc:
            print(f"  (could not write the defects manifest: {exc})")

    # ------------------------------------------------------------------
    # Stage 5: Conformation
    # ------------------------------------------------------------------

    def _graph_periodicity(self):
        """Per-axis boundaries the topology was built with.

        Recorded by the generators and carried in ``.nodes`` files by a
        ``# PERIODICITY`` header. Returns None when unknown, which every
        consumer reads as fully periodic -- the behaviour before open
        boundaries were supported.
        """
        from topon.topology.loader import graph_periodicity

        return graph_periodicity(self.graph)

    def _conformation_params(self) -> dict:
        """What the overlap resolver runs with, from three layers.

        Narrowest last: the hard defaults, then the validated `conformation`
        section, then the raw one. The raw section wins because a caller that
        builds a Pipeline by hand passes its overrides there and has done
        since before the section was in the schema.

        Only the three overlap-resolver keys are read. The bead-spring keys in
        the same section belong to `topon.conformation.place`, which this
        stage does not call.
        """
        schema_conf = self.config.conformation
        return {
            **self._DEFAULT_CONFORMATION,
            "overlap_cutoff": schema_conf.overlap_cutoff,
            "overlap_max_iters": schema_conf.overlap_max_iters,
            "noise_magnitude": schema_conf.noise_magnitude,
            **self.raw_config.get("conformation", {}),
        }

    def _run_conformation_stage(self) -> None:
        print("--- Stage 5: Conformation ---")
        from topon.conformation import ConformationManager

        conf_params = self._conformation_params()
        cm = ConformationManager(
            str(self.config.study.output_dir),
            self.config.study.name,
        )
        # Hand down the same cell stage 4 routed the chains with, so a
        # chain that wraps the boundary lands in a box of the same period,
        # and the boundary conditions so open axes are not wrapped at all.
        periodicity = self._graph_periodicity()
        conformed, roles = cm.apply_displacements(
            "system.data",
            lattice_box=None if self.dims is None else tuple(self.dims),
            periodicity=periodicity,
        )
        noisy = cm.apply_noise(conformed, magnitude=conf_params["noise_magnitude"])
        cm.resolve_overlaps(
            noisy,
            roles,
            cutoff=conf_params["overlap_cutoff"],
            max_iters=conf_params["overlap_max_iters"],
            periodicity=periodicity,
        )
        print()

    # ------------------------------------------------------------------
    # Stage 6: Output
    # ------------------------------------------------------------------

    def _run_output_stage(self) -> None:
        print("--- Stage 6: Output ---")
        from topon.writers import LammpsInputGenerator

        # Optional graph-format exports (GraphML, NPZ).
        if self.config.output.export_graphml:
            from topon.writers.graphml_writer import write_graphml
            graphml_path = self.output_dir / f"{self.config.study.name}.graphml"
            mean_dp = int(self.config.assignment.dp_distribution.default.mean)
            write_graphml(
                self.graph,
                str(graphml_path),
                dp=mean_dp,
                dims=self.dims,
            )
            print(f"  GraphML written to: {graphml_path}")
        if self.config.output.export_npz:
            try:
                from topon.writers.npz_writer import write_npz
            except ImportError:
                print(
                    "  [skip] NPZ export requested but topon.writers.npz_writer "
                    "is not available."
                )
            else:
                npz_path = self.output_dir / f"{self.config.study.name}.npz"
                write_npz(self.graph, str(npz_path), dims=self.dims)
                print(f"  NPZ written to: {npz_path}")

        sim_cfg = self.raw_config.get("simulation", {})
        experimental = self.raw_config.get("experimental", {})
        # LammpsInputGenerator branches on "cg" vs "atomistic" literals
        # (see topon/writers/lammps_inputs.py); the schema's chemistry.model_type
        # uses "coarse_grained" / "atomistic". Map at the call site rather than
        # touching every comparison in the writer.
        model = "cg" if self.config.chemistry.model_type == "coarse_grained" else "atomistic"

        # Pass the BASE output_dir (not self.output_dir which already includes
        # study.name) — LammpsInputGenerator re-appends study.name internally.
        # Matches the ConformationManager call pattern at line 259-263.
        gen = LammpsInputGenerator(
            str(self.config.study.output_dir),
            self.config.study.name,
            config=sim_cfg,
            experimental=experimental,
        )
        ff = self.config.chemistry.force_field
        charmm_style = {}
        if ff == "charmm" and self.config.chemistry.charmm is not None:
            charmm_style = {"charmm_pair_style": self.config.chemistry.charmm.pair_style}
        gen.write_serial_soft_minimization(
            settings_file="system.in.settings",
            model_type=model,
            force_field=ff,
        )
        gen.write_parallel_production(
            settings_file="system.in.settings",
            model_type=model,
            force_field=ff,
            **charmm_style,
        )
        print(f"  LAMMPS scripts written to: {self.output_dir / '04_Simulation'}")
        print()
