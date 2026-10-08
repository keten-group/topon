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
    ``Pipeline`` does not seed the global random streams.
    ``topology.generator.seed`` pins stage 1: the generator then draws from
    its own streams seeded with it (the graph ``random.seed(n)`` and
    ``np.random.seed(n)`` before generating gave), and the C route gets a
    seed drawn from the same number. Without it the generators draw from
    the global streams, so a graph is reproducible only if the caller sets
    ``random.seed(n)`` and ``np.random.seed(n)`` first, which
    ``topon.workflows.cg_network.run(seed=...)`` does. Unpinned, two runs of
    the same config give different graphs -- measured on SC 5x5x5 at
    neighbour_cutoff 1.5, 261 edges and then 279. With the seed the graph
    and its edge order are identical, which is what makes a chain index mean
    the same strand twice (``topon.conformation.strand_plans``, and the
    ``pairs`` of a conformation config). The DP draws (PDI above 1), random
    types, copolymer sequences, entanglements and grafts still draw from
    the global streams. The named atomistic placements, the historic
    placement's side-chain offsets, and the conformation stage's noise and
    overlap pushes do not; they come from streams keyed on the study name,
    so a study whose graph and assignment are pinned builds the same
    displacement and relaxed files on every run.
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
        self._claim_run_directory()
        print()

        self._run_topology_stage()
        self._run_analysis_stage()
        self._run_assignment_stage()
        self._run_chemistry_stage()
        self._run_conformation_stage()
        self._run_output_stage()

        print()
        print("=== Pipeline Complete ===")

    def run_graph_stages(self):
        """Stages 1 to 3 only: topology, analysis and assignment.

        Returns the graph the chemistry stage would be handed, with its DP,
        types and defects in place. Nothing is written but what those three
        stages write themselves (the run manifest's topology section).
        ``topon fit`` sweeps its candidate cutoffs through this and
        ``topon generate --verify`` regenerates its seeds with it, so both
        measure the graph ``run`` builds from rather than a copy of the
        logic.
        """
        self._run_topology_stage()
        self._run_analysis_stage()
        self._run_assignment_stage()
        return self.graph

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
        directions on the coarse-grained and historic atomistic routes.
        Without seeding, byte-equivalence cannot be guaranteed there. The
        named atomistic placements, the side-chain offsets of the
        historic one, and the conformation stage's noise come from
        streams keyed on the study name and need no seed.

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
        self._claim_run_directory()
        print(f"  Skipping stages 1-3 (graph supplied directly).")
        print(f"  Nodes: {self.graph.number_of_nodes()}, "
              f"Edges: {self.graph.number_of_edges()}")
        print()

        self._run_chemistry_stage()
        self._run_conformation_stage()
        self._run_output_stage()

        print()
        print("=== Pipeline Complete (rebuild) ===")

    def _claim_run_directory(self) -> None:
        """Record this process as the run directory's writer, after a look.

        One writer per run directory: two processes on one directory
        overwrite each other's stage files and manifest, which is how the
        end-linked validation sweep lost an hour. The manifest's ``run``
        entry names the writer (pid, host, start time). A live writer
        other than this process is reported with the way to stop it; the
        run goes on, because a pid can be reused and a false alarm must
        not block a build. Like the rest of the manifest this is
        advisory, so a directory that refuses the file costs nothing.
        """
        from topon.core.manifest import read_manifest, record_run
        from topon.utils.processes import other_live_writer, stop_hint, this_process

        try:
            previous = (read_manifest(self.output_dir) or {}).get("run")
            other = other_live_writer(previous)
            if other:
                print(f"  WARNING: process {other['pid']} (started "
                      f"{other.get('started', '?')}) is still writing this run "
                      f"directory. Two writers overwrite each other's files; "
                      f"stop one, or give each run its own study name or "
                      f"output_dir. {stop_hint(other['pid'])}.")
            record_run(self.output_dir, this_process(),
                       study=self.config.study.name)
        except Exception as exc:
            print(f"  (could not record the run in the manifest: {exc})")

    # ------------------------------------------------------------------
    # Stage 1: Topology
    # ------------------------------------------------------------------

    def _run_topology_stage(self) -> None:
        print("--- Stage 1: Topology ---")
        self._topology_manifest = {"source": self.config.topology.source}
        started = time.time()
        if self.config.topology.source == "generate":
            self._generate_topology()
        elif self.config.topology.source == "crosslink":
            self._crosslink_topology()
        else:
            self._load_existing_topology()
        arch = self.config.topology.generator.architecture
        if arch != "end_linked":
            # Only a non-default architecture is recorded, so an end-linked
            # graph and its manifest are exactly what they were.
            self.graph.graph["architecture"] = arch
            self._topology_manifest["architecture"] = arch
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
        if gen_cfg.seed is not None:
            self._topology_manifest["generator_seed"] = int(gen_cfg.seed)

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
            # With topology.generator.seed the draw comes from a stream
            # seeded with it, which is what np.random.seed(seed) gave, and
            # then the strict search is seeded too (unseeded, the strict C
            # route has always taken its seed from the clock).
            if gen_cfg.seed is not None:
                seed = int(np.random.RandomState(int(gen_cfg.seed))
                           .randint(0, 2 ** 31 - 1))
            elif search == "exact":
                seed = int(np.random.randint(0, 2 ** 31 - 1))
            else:
                seed = None
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

    def _crosslink_topology(self) -> None:
        """Stage 1 for ``topology.source = "crosslink"``: a crosslinked melt.

        Grows the chains of ``topology.crosslinking`` on a lattice and
        crosslinks the reactive beads that touch
        (:mod:`topon.topology.chain_crosslinking`). The graph comes back with
        every strand's DP and chain, which stage 3 then leaves alone. On the
        coarse-grained route every chord is held to its contour in the box
        ``chemistry.target_density`` gives, and a crosslink may sit next to
        a chain end (the builder bonds the end bead straight to the
        junction). On the atomistic route a strand joins its junction
        through a monomer, so reactive beads next to a chain end are left
        out (``min_dangling_dp``). Sol chains of several lengths are built
        at their lengths. The melt (every chain's beads on the lattice, the
        crosslinks in the order made) is written beside the graph as
        ``topology/crosslinked_melt.npz``.
        """
        from topon.topology.chain_crosslinking import ChainType, crosslink_chains

        cfg = self.config.topology.crosslinking
        chem = self.config.chemistry
        types = []
        for i, c in enumerate(cfg.chains):
            name = c.name or f"type{i + 1}"
            if c.sequence is not None:
                types.append(ChainType.from_sequence(
                    c.sequence, c.count, repeats=c.repeats,
                    crosslink_residue=c.crosslink_residue, name=name))
            elif c.reactive is not None:
                types.append(ChainType(count=c.count, dp=c.dp, reactive=tuple(c.reactive),
                                       site_types=c.site_type, name=name))
            else:
                types.append(ChainType.every(c.count, c.dp, every=c.reactive_every,
                                             start=c.reactive_start,
                                             site_type=c.site_type, name=name))
        seed = (int(cfg.seed) if cfg.seed is not None
                else int(np.random.randint(0, 2 ** 31 - 1)))
        cg = chem.model_type == "coarse_grained"
        min_dangling = cfg.min_dangling_dp
        if min_dangling is None:
            min_dangling = 0 if cg else 1
        elif min_dangling < 1 and not cg:
            raise ValueError(
                "topology.crosslinking.min_dangling_dp 0 needs the coarse-grained "
                "route: the atomistic builder joins a strand to its junction "
                "through a monomer, so a dangling strand keeps at least one")
        melt = crosslink_chains(
            types, crosslinks=cfg.crosslinks, conversion=cfg.conversion,
            per_chain=cfg.per_chain, packing=cfg.packing,
            persistence=cfg.persistence, c_inf=cfg.c_inf,
            contact_radius=cfg.contact_radius, max_radius=cfg.max_radius,
            min_gap=cfg.min_gap, pairs=cfg.pairs,
            build_density=float(chem.target_density) if cg else None,
            min_dangling_dp=min_dangling, keep_windings=cfg.keep_windings, seed=seed)
        rec = melt.record
        topology_dir = self.output_dir / "topology"
        topology_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            topology_dir / "crosslinked_melt.npz",
            positions=np.concatenate(melt.positions).astype(np.float32),
            lengths=np.array(melt.lengths, dtype=np.int64),
            crosslinks=np.array(melt.crosslinks, dtype=np.int64).reshape(-1, 4),
            box=melt.box)
        self._topology_manifest.update(generator="crosslink", crosslinking=rec,
                                       architecture="random_crosslinked")
        s = rec["strands"]
        print(f"  Crosslinked melt: {rec['chains']} chains, {rec['beads']} beads on a "
              f"{rec['lattice']}^3 lattice (packing {rec['packing']:.3f}, c_inf "
              f"{rec['c_inf']:.2f})")
        print(f"    {rec['crosslinks']} crosslinks (conversion {rec['conversion']:.3f}, "
              f"seed {seed}): {s.get('bridge', 0)} bridges, {s.get('dangling', 0)} "
              f"dangling, {s.get('loop', 0)} primary loops, {rec['sol_chains']} sol "
              f"chains, in {rec['seconds']['total']:.2f} s")
        if rec["reactive_left_out"]:
            print(f"    {rec['reactive_left_out']} reactive beads closer to a chain end "
                  f"than min_dangling_dp {min_dangling} allows left out")
        self.graph = melt.graph
        self.dims = np.asarray(melt.box, dtype=float)

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

        chem = self.config.chemistry
        self._assignment_manager = AssignmentManager(
            self.graph, self.dims, self.config.assignment,
            max_functionality=self._junction_valence_ceiling(),
            bead_density=(chem.target_density
                          if chem.model_type == "coarse_grained" else None),
            architecture=self.config.topology.generator.architecture,
            chains_decided=self.config.topology.source == "crosslink",
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
            # edge_atom_map remain valid. A failure of any of the three
            # stops the stage before a file is written. Before 0.4.5
            # the heavy-atom mol went to the writer uncharged and without
            # H_ when one of them raised, and NaN Gasteiger charges were set
            # to 0, each with only a [WARN]. When a step fails with atoms
            # left untyped (after a failed Sanitize, the atoms the builder
            # made one at a time, a Si junction, a POSS cage, the PDMS repeat
            # atoms, have no hybridisation and so no DREIDING type), the
            # error is UntypedAtomError naming them, with the failure
            # chained; otherwise it is ChargeError, naming the step and what
            # it raised, or for NaN charges the atoms Gasteiger has no
            # parameters for. Before 0.4.5 an atom
            # with no type was written as an invented one (Si_, O_), as in
            # an earlier POSS demo output, which came from a pipeline that
            # did not sanitize.
            from rdkit import Chem
            from topon.forcefield.dreiding import (
                ChargeError, UntypedAtomError, dreiding_types,
                gasteiger_charges, parse_dreiding_parameter_file)
            atom_types = parse_dreiding_parameter_file(DreidingWriter(
                self.chemical_space, data_path).param_file)["atom_types"]
            step = "Sanitize"
            try:
                try:
                    Chem.SanitizeMol(self.chemical_space)
                except Chem.AtomValenceException:
                    # A Si junction with five or six strands (from a loaded
                    # topology; the defect injector caps junctions at four
                    # on this route) fails RDKit's valence check.
                    # Skip just that check; the rest of sanitize (kekulize,
                    # ring-find, hybridisation) still runs. RDKit perceives
                    # such a Si as SP3D, which DREIDING has no type for, so
                    # the typing below stops with UntypedAtomError; an
                    # over-valent atom whose hybridisation has a type goes on.
                    Chem.SanitizeMol(
                        self.chemical_space,
                        sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        ),
                    )
                step = "AddHs"
                mol_h = Chem.AddHs(self.chemical_space)
            except Exception as exc:
                cause = f"{type(exc).__name__}: {exc}"
                # An atom DREIDING has no type for is named first whichever
                # step gave up, so the error does not hang on where a given
                # RDKit version stops (that error, with this failure chained).
                try:
                    dreiding_types(self.chemical_space, atom_types)
                except UntypedAtomError as err:
                    doing = {"Sanitize": "sanitize",
                             "AddHs": "add the hydrogens of"}[step]
                    raise UntypedAtomError(err.atoms, note=(
                        f"The chemistry stage could not {doing} this "
                        f"molecule: {step} failed with {cause}")) from exc
                raise ChargeError(step, cause) from exc
            # Typed before it is charged: an atom DREIDING has no type for
            # (the SP3D Si above) has no Gasteiger parameters either, and
            # the typing error is the one that says what to fix.
            dreiding_types(mol_h, atom_types)
            gasteiger_charges(mol_h)

            # Background charge neutralization: redistribute a residual net
            # charge uniformly across all atoms. Gasteiger keeps the
            # molecule's net formal charge, and leaves a tiny residual
            # (~1e-12 e) from finite-precision iteration, which the 1e-6
            # threshold leaves alone. (Before 0.4.5 this also spread the
            # charge left by NaN charges set to 0, several e for the
            # degree 5-6 Si of an earlier defect demo.)
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
            writer.write()
            self._geometry_from_dreiding(writer)

        if model != "coarse_grained":
            self._record_strands(mol_h)
        else:
            # An atomistic run of the same study earlier would otherwise leave
            # its strand record describing a data file that is gone.
            from topon.core.manifest import drop_stage
            try:
                drop_stage(self.output_dir, "strands", "placement")
            except Exception as exc:
                print(f"  (could not update the run manifest: {exc})")

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
        elif self._atomistic_placement() is not None:
            # Every backbone drawn as a chain at the force field's bond
            # lengths, everything else at bond length off it.
            self._place_atomistic(mol_h, scale, chem_dir)
        else:
            self._record_chords(mol_h, scale)
            # Atomistic: backbone + grafts + pendant + hydrogens. Backbone
            # path consults `entangled_with` on the (u, v, key) edge data
            # for kinked-chain placement (N+2 fix). graft_atom_map holds
            # the side-chain heavy atoms of grafted strands
            # (ChemistryBuilder._build_chain_per_repeat), placed along a
            # direction drawn from the global stream; every other atom off
            # the backbone gets coords from the pendant pass (neighbor
            # propagation through mol_h).
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
            # The offsets come from a stream of their own, keyed on the study
            # name, as stage 5's noise is. From NumPy's global stream they
            # were unseeded unless the caller seeded it.
            known = {**node_coords, **backbone_coords, **graft_coords}
            side_coords = generate_approximate_side_chain_coords(
                mol_h, known, _stable_rng("sidechain", self.config.study.name))
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
        self._bond_r0 = {(min(a, b), max(a, b)): float(terms.bond_params[t][1])
                         for t, a, b in terms.bonds if t in terms.bond_params}
        self._angle_theta0 = {}
        for t, a, b, c in terms.angles:
            if t in terms.angle_params:
                th = float(terms.angle_params[t][1])
                self._angle_theta0[(a, b, c)] = self._angle_theta0[(c, b, a)] = th
        # the backbone's LAMMPS atom types, for the hard-backbone stages and
        # the backbone dumps, as on the DREIDING route
        self._backbone_types = sorted({terms.atom_types[typing.atom_type[i]]
                                       for i in self._backbone_atoms()})
        self._n_atom_types = len(terms.atom_types)

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

    # ------------------------------------------------------------------
    # Stage 4, atomistic: strands, chords and backbone placement
    # ------------------------------------------------------------------

    def _geometry_from_dreiding(self, writer) -> None:
        """Equilibrium length of every bond and angle, from the file just written.

        Keyed by 0-based atom indices of the molecule the writer wrote, which
        are the pipeline's own. The placement draws at these lengths and the
        chord guard reads the extended length from them.
        """
        r0_of = {tid: float(sig[3]) for sig, tid in writer.bond_types.items()}
        self._bond_r0 = {(min(i, j) - 1, max(i, j) - 1): r0_of[t]
                         for _bid, t, i, j in writer.bond_data}
        th_of = {tid: float(sig[4]) for sig, tid in writer.angle_types.items()}
        self._angle_theta0 = {}
        for _aid, t, a, b, c in writer.angle_data:
            self._angle_theta0[(a - 1, b - 1, c - 1)] = th_of[t]
            self._angle_theta0[(c - 1, b - 1, a - 1)] = th_of[t]
        # The LAMMPS atom types the backbones and junctions are made of (Si3
        # and O_3 for PDMS), for simulation.atomistic_protocol "hard_backbone",
        # which keeps every pair of them hard from the first step.
        self._backbone_types = sorted({
            writer.atom_types_dict[writer.atom_dreiding_types[i + 1]]
            for i in self._backbone_atoms()})
        # The resonant types (an aromatic ring's C_R, N_R, O_R), which the
        # hard-backbone deck keeps hard as well, so that no ring passes
        # through a bond or another ring in stage 1.
        self._ring_types = sorted(
            {int(tid) for name, tid in writer.atom_types_dict.items()
             if str(name).endswith("_R")} - set(self._backbone_types))
        self._n_atom_types = len(writer.atom_types_dict)

    def _backbone_atoms(self) -> set:
        """RDKit indices of every backbone and junction atom."""
        ids = set()
        for edge, path in self._builder.edge_backbone_path.items():
            ids |= set(path)
            ids |= {a for a in self._builder.edge_ends.get(edge, ()) if a is not None}
        return ids

    def _record_strands(self, mol) -> None:
        """Which atoms are which strand, in the run manifest.

        The atomistic data file puts every atom in molecule 1, so without
        this a strand cannot be told from its neighbours after the fact. The
        record is the builder's strand table with LAMMPS atom ids (RDKit
        index + 1), which LAMMPS keeps through every stage; the Z1+ export of
        the backbone and the atomistic gates read it back
        (:mod:`topon.analysis.atomistic`).
        """
        from topon.core.manifest import record_stage

        table = self._builder.strand_table(mol)

        def lammps(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return [lammps(i) for i in x]
            return int(x) + 1

        strands = []
        for row in table["strands"]:
            out = dict(row)
            for key in ("junctions", "backbone", "repeat_heads"):
                out[key] = lammps(row[key])
            for key in ("free_end", "free_ends"):
                if key in row:
                    out[key] = lammps(row[key])
            for key in ("heavy", "hydrogens"):
                span = dict(row[key])
                if span.get("range") is not None:
                    span["range"] = lammps(span["range"])
                if "ids" in span:
                    span["ids"] = lammps(span["ids"])
                out[key] = span
            strands.append(out)
        # A designed entanglement (assignment.entanglements) names its
        # partner edge; the record names the partner strand, so the gates can
        # say whether the pair kept its winding.
        index = {}
        for n, row in enumerate(strands, start=1):
            u, v, key = row["edge"]
            index[(min(u, v), max(u, v), key)] = n
        multi = self.graph.is_multigraph()
        for row in strands:
            u, v, key = row["edge"]
            try:
                data = self.graph[u][v][key] if multi else self.graph[u][v]
            except KeyError:
                continue
            partner = data.get("entangled_with")
            if partner is None:
                continue
            pu, pv = partner[0], partner[1]
            pk = partner[2] if len(partner) > 2 else 0
            k = index.get((min(pu, pv), max(pu, pv), pk))
            if k is not None:
                row["partner"] = k
                row["windings"] = int(data.get("entanglement_count", 1))
        nodes = [{**n, "attach": lammps(n["attach"]), "atoms": lammps(n["atoms"]),
                  "hydrogens": lammps(n["hydrogens"])} for n in table["nodes"]]
        classes = {}
        for row in strands:
            classes[row["cls"]] = classes.get(row["cls"], 0) + 1
        entry = {"ids": "LAMMPS atom ids of 02_Chemistry/system.data",
                 "n_atoms": int(mol.GetNumAtoms()),
                 "classes": classes, "strands": strands, "nodes": nodes}
        try:
            record_stage(self.output_dir, "strands", entry,
                         study=self.config.study.name)
        except Exception as exc:
            print(f"  (could not write the strand record: {exc})")

    def _atomistic_strand_specs(self, mol, scale: float):
        """One :class:`~topon.conformation.atomistic.StrandSpec` per strand, in A."""
        from topon.conformation.atomistic import StrandSpec

        b = self._builder
        box = np.asarray(self.dims, float) * scale
        r0 = getattr(self, "_bond_r0", {}) or {}
        th = getattr(self, "_angle_theta0", {}) or {}
        table = {tuple(r["edge"]): r for r in b.strand_table(mol)["strands"]}

        def pos(node):
            return np.asarray(self.graph.nodes[node].get("pos", (0.0, 0.0, 0.0)),
                              float) * scale

        specs = []
        for edge, backbone in b.edge_backbone_path.items():
            u, v, _key = edge
            su, sv = b.edge_ends[edge]
            if su is None or sv is None:
                continue
            chain = [su] + list(backbone) + [sv]
            r0s = [r0.get((min(x, y), max(x, y)), 1.5)
                   for x, y in zip(chain[:-1], chain[1:])]
            th0 = [th.get((chain[i - 1], chain[i], chain[i + 1]), 109.4712)
                   for i in range(1, len(chain) - 1)]
            start = pos(u)
            vec = pos(v) - start
            vec = vec - box * np.round(vec / box)
            away = None
            if u == v:
                away = []
                for x, y in self.graph.edges(u):
                    if x == y:
                        continue
                    d = pos(y if x == u else x) - start
                    away.append(d - box * np.round(d / box))
            specs.append(StrandSpec(
                edge=edge, cls=table.get(edge, {}).get("cls", "bridge"),
                start_atom=su, end_atom=sv, backbone=list(backbone),
                start=start, end=start + vec, r0=r0s, theta0=th0,
                away_from=away or None))
        return specs

    def _record_chords(self, mol, scale: float) -> None:
        """Every chord against its backbone, for the historic placement.

        That placement draws nothing, so this is the whole guard it gets: the
        manifest says how close to its extended length each strand's chord
        sits, and a chord past the backbone's contour is warned about, since
        no relaxation can build it without stretching every bond.
        """
        import warnings as _warnings

        from topon.conformation.atomistic import TAUT, extended_length
        from topon.core.manifest import record_stage

        ratios, over = [], 0
        for spec in self._atomistic_strand_specs(mol, scale):
            if spec.cls == "loop":
                continue
            chord = float(np.linalg.norm(spec.end - spec.start))
            ext = extended_length(spec.r0, spec.theta0)
            ratios.append(chord / ext if ext else 0.0)
            over += chord > float(np.sum(spec.r0))
        if not ratios:
            return
        ratios = np.asarray(ratios)
        entry = {"shape": "chord (historic)", "strands": int(len(ratios)),
                 "taut": int((ratios >= TAUT).sum()), "over_contour": int(over),
                 "chord_over_extended": {"mean": float(ratios.mean()),
                                         "max": float(ratios.max())}}
        if over:
            _warnings.warn(
                f"{over} strand chord(s) are longer than their backbone contour, "
                f"so their bonds cannot relax to length; lower the neighbour "
                f"cutoff, raise the DP or the density", RuntimeWarning, stacklevel=2)
        try:
            record_stage(self.output_dir, "placement", entry,
                         study=self.config.study.name)
        except Exception as exc:
            print(f"  (could not write the placement record: {exc})")

    def _atomistic_placement(self):
        """The placement this build uses: the configured one, or the historic.

        ``conformation.atomistic_placement`` defaults to ``meander``; null
        asks for the historic placement. A network with POSS nodes takes the
        configured placement like any other since 0.4.5, which places each
        cage whole (:meth:`_place_atomistic`); before, it fell back to the
        historic placement unless one was named, and a named one was refused.
        A ``conformation.entanglement.target_Z`` is met by the coil's radius,
        so it makes the placement ``coil``; naming another one with it is
        refused.
        """
        if hasattr(self, "_placement_used"):
            return self._placement_used
        conf = self.config.conformation
        shape = conf.atomistic_placement
        named = "atomistic_placement" in conf.model_fields_set
        if conf.entanglement.target_Z is not None:
            if named and shape != "coil":
                raise ValueError(
                    f"conformation.entanglement.target_Z on the atomistic route is "
                    f"met by the coil radius, and atomistic_placement is "
                    f"{shape!r}; set it to 'coil' or leave it out")
            shape = "coil"
        self._placement_used = shape
        return shape

    def _place_cages(self, mol, scale: float, specs, box):
        """Every POSS node as a rigid body, and the strands' ends on its arms.

        The node's shape is embedded once per kind
        (:func:`topon.chemistry.node_bodies.node_templates`: the simbox's checked
        embedding of the node's own fragment, then held at the force field's
        r0), its centre set on the junction and its stubs turned towards the
        strands they bond to (:func:`topon.conformation.atomistic.place_bodies`).
        Each strand bonded to a cage then starts (or ends) at its atom
        bonded to the cage, set where the template has it (the stub) and
        held there, with the cage's attachment atom as its lead, so the
        settle reads the angle at it and the attachment atom's own hydrogens,
        placed with the cage, stay clear of it; a strand too short for that
        starts on the attachment atom where the cage put it. A designed
        path loses its point for each end moved, and since it was drawn from
        the junction, the cage's centre, it is led out of the cage round it
        along its own curve
        (:func:`~topon.conformation.atomistic.lead_out_of_body`). A strand whose chord runs
        through a cage's core is given a detour round it
        (:func:`~topon.conformation.atomistic.cage_detour`). Returns the
        bodies, the specs with their ends moved, the stub atoms' positions,
        and what the templates came out as. A network without POSS nodes
        returns no bodies and the specs as they were.
        """
        from topon.chemistry.node_bodies import node_templates
        from topon.conformation.atomistic import (cage_detour, keep_off_points,
                                                  lead_out_of_body, place_bodies)

        templates = node_templates(mol, self._builder, getattr(self, "_bond_r0", {}) or {})
        if not templates:
            return [], specs, {}, None
        index = {}
        for k, s in enumerate(specs):
            if s.backbone:
                index[(s.start_atom, s.backbone[0])] = (k, 0)
                index[(s.end_atom, s.backbone[-1])] = (k, -1)
        shapes = []
        cage_nodes = set()
        for t in templates:
            heavy = np.array([mol.GetAtomWithIdx(int(a)).GetAtomicNum() > 1
                              for a in t.atoms], bool)
            stubs, stub_ids, targets, attach = [], [], [], {}
            for a, s, x in t.stubs:
                hit = index.get((int(a), int(s)))
                if hit is None:
                    continue
                k, end = hit
                spec = specs[k]
                attach.setdefault(k, set()).add(end)
                stubs.append(x)
                stub_ids.append((int(a), int(s)))
                if spec.edge[0] == spec.edge[1]:
                    targets.append(None)        # a loop back to this cage
                else:
                    targets.append((spec.end - spec.start) if end == 0
                                   else (spec.start - spec.end))
            at = np.asarray(self.graph.nodes[t.node].get("pos", (0.0, 0.0, 0.0)),
                            float) * scale
            shapes.append({"node": t.node, "atoms": t.atoms, "xyz": t.xyz, "heavy": heavy,
                           "centre": t.centre, "core": t.core, "bonds": t.bonds,
                           "faces": t.faces, "stubs": stubs, "stub_ids": stub_ids,
                           "targets": targets,
                           "at": at, "attach": {k: sorted(v) for k, v in attach.items()}})
            cage_nodes.add(t.node)
        chords = (np.array([s.start for s in specs]), np.array([s.end for s in specs]))
        points = [np.asarray(self.graph.nodes[n].get("pos", (0.0, 0.0, 0.0)), float) * scale
                  for n in self._builder.node_map if n not in cage_nodes]
        bodies = place_bodies(shapes, box, chords=chords,
                              chord_strands=list(range(len(specs))), points=points)
        # where each cage holds the strand atom bonded to it: its stub
        stub_at, lead, stub_body = {}, {}, {}
        for body, sh in zip(bodies, shapes):
            at = {int(a): x for a, x in zip(body.atoms, body.xyz)}
            for (a, s_atom), x in zip(sh["stub_ids"], sh["stubs"]):
                stub_at[s_atom] = body.centre + body.rotation @ (
                    np.asarray(x, float) - np.asarray(sh["centre"], float))
                lead[s_atom] = (a, at[a])
                stub_body[s_atom] = body
        # where every cage atom is, for an end left on its attachment atom
        cage_at = {int(a): x for body in bodies for a, x in zip(body.atoms, body.xyz)}
        moved, held, detours = [], {}, 0
        led, designed = 0, []
        for k, s in enumerate(specs):
            start_atom, end_atom = s.start_atom, s.end_atom
            backbone, r0, th = list(s.backbone), list(s.r0), list(s.theta0)
            start, end = s.start, s.end
            path = None if s.path is None else np.asarray(s.path, float)
            lead_start = lead_end = None
            if backbone and int(backbone[0]) in stub_at and len(backbone) >= 2:
                a, xa = lead[int(backbone[0])]
                lead_start = (a, xa, r0[0], th[0])
                start_atom, start = backbone[0], stub_at[int(backbone[0])]
                held[int(start_atom)] = start
                backbone, r0, th = backbone[1:], r0[1:], th[1:]
                if path is not None:      # a designed path, one point per atom
                    path = path[1:]
            elif int(start_atom) in cage_at:
                # a strand too short to start on its stub starts on the
                # attachment atom, where the cage put it (not the junction,
                # which is the cage's centre)
                start = cage_at[int(start_atom)]
            if backbone and int(backbone[-1]) in stub_at and len(backbone) >= 2:
                a, xa = lead[int(backbone[-1])]
                lead_end = (a, xa, r0[-1], th[-1])
                end_atom, end = backbone[-1], stub_at[int(backbone[-1])]
                held[int(end_atom)] = end
                backbone, r0, th = backbone[:-1], r0[:-1], th[:-1]
                if path is not None:
                    path = path[:-1]
            elif int(end_atom) in cage_at:
                end = cage_at[int(end_atom)]
            vec = end - start
            vec = vec - box * np.round(vec / box)
            if path is not None and len(path) and (lead_start or lead_end):
                # a designed path is drawn from its junction, the cage's
                # centre, so it is led out of the cage round it, along its
                # own curve
                # in the strand's own image: the pair is drawn in the image
                # of its first strand, a whole cell away for the other
                drawn = path - box * np.round((path[0] - start) / box)
                out_s = out_e = None
                if lead_start:
                    out_s = lead_out_of_body(start, drawn, start + vec,
                                             stub_body[int(start_atom)], box)
                led_path = drawn if out_s is None else out_s
                if lead_end:
                    out_e = lead_out_of_body(start + vec, led_path[::-1], start,
                                             stub_body[int(end_atom)], box)
                    if out_e is not None:
                        led_path = out_e[::-1]
                if out_s is not None or out_e is not None:
                    led += 1
                    path = led_path
                designed.append(([start_atom] + list(backbone) + [end_atom],
                                 np.vstack([start, drawn, start + vec]),
                                 np.vstack([start, led_path, start + vec])))
            elif path is not None:
                chain = np.vstack([start, path, start + vec])
                designed.append(([start_atom] + list(backbone) + [end_atom], chain, chain))
            cls = s.cls
            if cls == "loop" and (lead_start or lead_end or start_atom != end_atom):
                # a loop on a cage leaves it twice, from two stubs or two
                # corners, so its two ends are apart: drawn as a strand
                # between them
                cls = "bridge"
            spec = type(s)(edge=s.edge, cls=cls, start_atom=start_atom,
                           end_atom=end_atom, backbone=backbone,
                           start=np.asarray(start, float),
                           end=np.asarray(start, float) + vec,
                           r0=r0, theta0=th, away_from=s.away_from,
                           path=path, lead_start=lead_start, lead_end=lead_end)
            if path is None and cls != "loop":
                # a chord through a cage's core (a loop between two corners
                # of a bare cage, or a bridge past a cage on a third site) is
                # drawn round the cage: no turn about the chord takes it out
                for body in bodies:
                    via = cage_detour(spec.start, spec.end, body, box,
                                      attached=body.attach.get(k, ()))
                    if via is not None:
                        spec.via = via
                        spec.keep_off = keep_off_points(body, spec.start, box)
                        detours += 1
                        break
            moved.append(spec)
        record = {"cages": len(bodies),
                  "detoured": detours,
                  "templates": len({t.signature for t in templates}),
                  # of them, how many the stored shapes gave
                  "templates_stored": len({t.signature for t in templates
                                           if t.source == "stored"}),
                  "seeds": sorted({t.seed for t in templates}),
                  "atoms_per_cage": sorted({len(t.atoms) for t in templates}),
                  "template_bond_strain_max": round(max(t.max_strain for t in templates), 5),
                  "core_radius": round(max(t.core for t in templates), 4)}
        if led:
            # the designed paths led out of their cages, and whether leading
            # them out took a bond through a bond of any designed path, theirs
            # or another's (which would change a designed winding)
            from topon.conformation.segments import passing_pairs

            ids = [np.asarray(c[0], int) for c in designed]
            sizes = np.array([len(x) for x in ids])
            starts_ = np.concatenate([[0], np.cumsum(sizes)[:-1]])
            rows = np.concatenate([np.stack([np.arange(s0, s0 + m - 1),
                                             np.arange(s0 + 1, s0 + m)], axis=1)
                                   for s0, m in zip(starts_, sizes)])
            flat = np.concatenate(ids)
            before = np.vstack([c[1] for c in designed])
            after = np.vstack([c[2] for c in designed])
            passed = passing_pairs(before, after, rows, flat[rows], box)[0]
            record["led_out"] = led
            record["led_out_passages"] = int(len(passed))
        return bodies, moved, held, record

    def _place_atomistic(self, mol, scale: float, chem_dir) -> None:
        """Backbones as chains at bond length, and every other atom off them.

        ``conformation.atomistic_placement`` picks the shape. The result goes
        out through the same displacement files as the historic placement (in
        lattice units, since the files carry the scale), so the conformation
        stage reads it unchanged. A POSS node is placed whole
        (:meth:`_place_cages`), and its atoms go in the nodes file,
        written again with them, so stage 5's overlap pass holds the cage
        as it holds the junctions.
        """
        from topon.conformation.atomistic import place_network
        from topon.conformation.entanglement.realize import entangled_backbone_paths
        from topon.core.manifest import record_stage
        from topon.utils import write_lammps_displacement_file

        b = self._builder
        conf = self.config.conformation
        shape = self._atomistic_placement()
        box = np.asarray(self.dims, float) * scale
        specs = self._atomistic_strand_specs(mol, scale)

        ent_cfg = self.config.assignment.entanglements
        sites = {}
        drawn = entangled_backbone_paths(
            self.graph, self.dims, {s.edge: s.backbone for s in specs},
            method=ent_cfg.method, kink_params=ent_cfg.kink_params.model_dump(),
            sites=sites)
        for s in specs:
            if s.edge in drawn:
                s.path = np.asarray(drawn[s.edge], float) * scale
        # Each designed braid in A, once per pair, for the placement to keep
        # the other strands out of.
        braids = list({id(v): {"mid": v["mid"] * scale, "axis": v["axis"],
                               "half": v["half"] * scale,
                               "radius": v["radius"] * scale}
                       for v in sites.values()}.values())
        # POSS nodes, placed whole before any strand is drawn
        bodies, specs, stub_at, cage_record = self._place_cages(mol, scale, specs, box)
        cage_nodes = {body.node for body in bodies}
        # every cage atom, and the strand atoms its stubs hold
        body_atoms = {int(a): x for body in bodies for a, x in zip(body.atoms, body.xyz)}
        body_atoms.update(stub_at)

        anchors = {}
        for node, ref in b.node_map.items():
            if node in cage_nodes:
                continue
            anchors[int(ref)] = (np.asarray(self.graph.nodes[node].get(
                "pos", (0.0, 0.0, 0.0)), float) * scale)
        anchors.update(stub_at)
        neighbours = {a.GetIdx(): [n.GetIdx() for n in a.GetNeighbors()]
                      for a in mol.GetAtoms()}
        # From a stream of its own, keyed on the study name, as stage 5's
        # noise and the historic side chains are. It used to take its seed
        # from NumPy's global stream, which `topon generate` never seeds, so
        # a config with every seed pinned placed different backbones on
        # every run. A fresh generator from the same key draws the same
        # network, which is what the coil-radius search needs.
        study = self.config.study.name

        def stream():
            return _stable_rng("placement", study)

        knobs = dict(waves=conf.meander_waves, min_bond=conf.min_bond,
                     min_sep=conf.min_self_separation, jitter=conf.path_jitter,
                     radius=conf.atomistic_coil_radius,
                     parallel_strands=conf.parallel_strands)
        target = conf.entanglement.target_Z

        def place(radius):
            return place_network(
                specs, neighbours, anchors, getattr(self, "_bond_r0", {}), box,
                shape, stream(), clearance=conf.atomistic_clearance,
                keep_frames=True, braids=braids, bodies=bodies or None,
                **{**knobs, "radius": radius})

        if target is None:
            coords, report = place(knobs["radius"])
            z_target = None
        else:
            coords, report, z_target = self._place_for_target(
                specs, box, shape, stream, knobs, float(target), place, mol)
        if report.frames is not None:
            # The settling pass read as a trajectory: no backbone bond may
            # have gone through another on the way.
            from topon.analysis.crossings import Frame, find_crossings

            ids, steps = report.frames
            order = np.argsort(ids)
            frames = [Frame(step=k, box=box, ids=ids[order] + 1, xyz=x[order])
                      for k, x in enumerate(steps)]
            bonds, strand = [], []
            for k, s in enumerate(specs, start=1):
                chain = (([s.lead_start[0]] if s.lead_start else []) + [s.start_atom]
                         + list(s.backbone) + [s.end_atom]
                         + ([s.lead_end[0]] if s.lead_end else []))
                bonds += [(a + 1, b + 1) for a, b in zip(chain[:-1], chain[1:])]
                strand += [k] * (len(chain) - 1)
            # and every strand against the cages' bonds, which do not move
            for k, body in enumerate(bodies, start=len(specs) + 1):
                bonds += [(int(a) + 1, int(c) + 1) for a, c in body.bonds]
                strand += [k] * len(body.bonds)
            passed = find_crossings(frames, np.array(bonds, int), np.array(strand, int))
            report.settle["passages"] = len(passed.crossings)
            report.frames = None

        backbone_ids = {int(i) for s in specs for i in s.backbone}
        files = {"backbone": {}, "grafts": {}, "pendant": {}, "hydrogens": {}}
        for idx, xyz in coords.items():
            if idx in anchors or idx in body_atoms:
                continue
            key = ("backbone" if idx in backbone_ids else
                   "hydrogens" if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1
                   else "pendant")
            files[key][idx] = tuple(np.asarray(xyz, float) / scale)
        missing = mol.GetNumAtoms() - len(coords)
        for key, name in (("backbone", "system_backbone.displace"),
                          ("grafts", "system_grafts.displace"),
                          ("pendant", "system_pendant.displace"),
                          ("hydrogens", "system_hydrogens.displace")):
            write_lammps_displacement_file(files[key], scale, scale, scale,
                                           str(chem_dir / name), key)
        if bodies:
            # The nodes file again, every cage atom in it where the cage was
            # placed: stage 4 wrote the attachment atom on the junction
            nodes = {}
            for node, ref in b.node_map.items():
                if node in cage_nodes:
                    continue
                primary = ref[0] if isinstance(ref, (list, tuple)) else ref
                nodes[int(primary)] = tuple(self.graph.nodes[node].get("pos", (0.0, 0.0, 0.0)))
            for idx, xyz in body_atoms.items():
                nodes[idx] = tuple(np.asarray(xyz, float) / scale)
            write_lammps_displacement_file(nodes, scale, scale, scale,
                                           str(chem_dir / "system_nodes.displace"), "nodes")
        summary = report.summary()
        summary["unplaced_atoms"] = int(missing)
        if z_target is not None:
            summary["z_target"] = z_target
        if cage_record is not None:
            summary.setdefault("cages", {}).update(cage_record)
        worst = sorted((r for r in report.strands
                        if r.get("chord_over_extended") is not None),
                       key=lambda r: -r["chord_over_extended"])[:5]
        summary["tautest"] = [{k: r[k] for k in ("edge", "cls", "chord", "extended",
                                                 "contour", "routine")}
                              for r in worst]
        print(f"  Atomistic placement '{shape}': {summary['strands']} backbones "
              f"{summary['routines']}, {summary['taut']} taut, "
              f"{summary['over_contour']} over their contour, backbone bond / r0 "
              f"{summary['backbone_bond_ratio']}")
        par = summary.get("parallel")
        if par:
            print(f"  Secondary loops: {par['drawn_apart']} of {par['strands']} strands on "
                  f"{par['chords']} shared chords drawn on their own side")
        br = summary.get("braids")
        if br:
            print(f"  Designed braids: {br['braids']}; strands turned out of them "
                  f"{br['turned_clear']} of {br['inside']} found inside, "
                  f"{len(br['left_inside'])} left inside")
        rg = summary.get("rings")
        if rg:
            through = rg.get("bonds_through") or {}
            print(f"  Rings in the backbones: {rg['stretches']} stretches"
                  f"{' settled whole' if rg['held_flat'] else ' (not settled)'}, "
                  f"{rg['closed']} closed on their drawn stretch (largest fit offset "
                  f"{rg['fit_off_max']:.3f} A); bonds through a ring as placed: "
                  f"{through.get('backbone', 0)} backbone, {through.get('other', 0)} other")
            if through.get("backbone") or through.get("other"):
                # counted, not moved: see USAGE on rings in a backbone
                import warnings as _warnings
                _warnings.warn(
                    f"{through.get('backbone', 0)} backbone bond(s) and "
                    f"{through.get('other', 0)} other bond(s) pass through the face of "
                    f"a ring a backbone runs through, as placed; the first stage "
                    f"starts with them there", RuntimeWarning, stacklevel=2)
        pr = summary.get("pendant_rings")
        if pr:
            through_p = pr.get("bonds_through") or {}
            print(f"  Pendant rings: {pr['rings']} placed whole, {pr['turned']} turned off a "
                  f"bond through them, {pr['left_threaded']} with none clear; bonds through "
                  f"a pendant ring as placed: {through_p.get('backbone', 0)} backbone, "
                  f"{through_p.get('other', 0)} other")
            if through_p.get("backbone") or through_p.get("other"):
                # counted, not moved: see USAGE on pendant rings
                import warnings as _warnings
                _warnings.warn(
                    f"{through_p.get('backbone', 0)} backbone bond(s) and "
                    f"{through_p.get('other', 0)} other bond(s) pass through the face of "
                    f"a pendant ring, as placed; the first stage starts with them there",
                    RuntimeWarning, stacklevel=2)
        cg = summary.get("cages")
        if cg:
            through = cg.get("bonds_through_faces")
            print(f"  POSS cages placed whole: {cg['cages']} ({cg['templates']} "
                  f"template(s), {cg.get('templates_stored', 0)} stored, bonds within "
                  f"{100 * cg['template_bond_strain_max']:.2f} % "
                  f"of r0); strands turned out of them {cg.get('turned_clear', 0)} of "
                  f"{cg.get('inside', 0)} found inside"
                  + (f" ({cg['near_arms']} by another cage's arms)" if cg.get("near_arms") else "")
                  + f", {len(cg.get('left_inside', []))} "
                  f"left inside, {cg.get('detoured', 0)} drawn round one"
                  + (f", {cg['led_out']} designed led out of one "
                     f"({cg['led_out_passages']} passages)" if cg.get("led_out") else "")
                  + f"; bonds through a cage face {through}; every bond / r0 "
                  f"{cg.get('bond_ratio_all')}")
            if cg.get("led_out_passages"):
                import warnings as _warnings
                _warnings.warn(
                    f"leading {cg['led_out']} designed path(s) out of their cages took "
                    f"{cg['led_out_passages']} pair(s) of their bonds through each "
                    f"other; a designed winding may have changed", RuntimeWarning,
                    stacklevel=2)
            if through:
                import warnings as _warnings
                _warnings.warn(
                    f"{through} bond(s) pass through a POSS cage face as placed; no "
                    f"relaxation takes a bond back out through a ring",
                    RuntimeWarning, stacklevel=2)
            span = cg.get("bond_ratio_all")
            if span and (span["min"] < 0.85 or span["max"] > 1.15):
                import warnings as _warnings
                _warnings.warn(
                    f"a bond of the POSS build is placed at {span['min']:.3f} to "
                    f"{span['max']:.3f} of its r0, more than 15 % off",
                    RuntimeWarning, stacklevel=2)
        c = summary.get("settle")
        if c:
            print(f"  Backbones settled in {c['rounds']} rounds: bond pairs closer "
                  f"than {c['clearance']} A {c['pairs_below_before']} -> "
                  f"{c['pairs_below_after']}, angle error (deg) "
                  f"{c['angle_error_before']} -> {c['angle_error_after']}, largest "
                  f"shift {c['largest_shift']:.2f} A, {c.get('passages')} passages "
                  f"on the way")
            rt = c.get("ring_threads")
            if rt:
                lay = (c.get("ring_lay") or {}).get("passages", 0)
                print(f"  Ring shapes laid on their stretches: {lay} passage(s) of backbone "
                      f"bonds; backbone bonds through a ring's face: {rt['drawn']} as the "
                      f"settle starts, {rt['after']} settled")
                if lay:
                    import warnings as _warnings
                    _warnings.warn(
                        f"laying the ring shapes on their stretches carried {lay} pair(s) of "
                        f"backbone bonds through each other, before the settle's rounds (whose "
                        f"passages are counted apart)", RuntimeWarning, stacklevel=2)
            # pairs left short, or a settle stopped at its round cap with
            # none short (every build since 0.4.5)
            from topon.conformation.atomistic import settle_warning

            message = settle_warning(c)
            if message:
                import warnings as _warnings
                _warnings.warn(message, RuntimeWarning, stacklevel=2)
        try:
            record_stage(self.output_dir, "placement", summary,
                         study=self.config.study.name)
        except Exception as exc:
            print(f"  (could not write the placement record: {exc})")

    def _z_meter(self, specs, box, mol, seeds: int):
        """Z1+ per bridge of backbones given as one path per strand, end to end.

        Read on the points the gates read (one atom per repeat unit, between
        the two junctions; :func:`topon.analysis.z1plus.export_system`), with
        the junctions jittered as the export does, averaged over ``seeds``.
        Returns ``measure(paths) -> float``.
        """
        from topon.analysis.z1plus import (Z1PlusFailed, jitter_ends, run_z1,
                                           why_unavailable)

        why = why_unavailable()
        if why:
            raise RuntimeError(
                f"conformation.entanglement.target_Z on the atomistic route is "
                f"met by measuring Z1+ on the build, and Z1+ is not available: "
                f"{why}")
        heads = {tuple(r["edge"]): r["repeat_heads"]
                 for r in self._builder.strand_table(mol)["strands"]}
        rows, cls = [], []
        for spec in specs:
            chain = [spec.start_atom] + list(spec.backbone) + [spec.end_atom]
            where = {a: i for i, a in enumerate(chain)}
            # in chain order: a dangling strand's table lists them from its
            # other end
            idx = sorted(where[h] for h in heads.get(spec.edge, []) if h in where)
            idx = [0] + [i for i in idx if 0 < i < len(chain) - 1] + [len(chain) - 1]
            rows.append(idx)
            cls.append(spec.cls)
        bridge = np.array([c == "bridge" for c in cls])

        def measure(paths) -> float:
            chains = [np.asarray(p, float)[r] for p, r in zip(paths, rows)]
            zs, k, why = [], 0, None
            while len(zs) < seeds and k < 3 * seeds:
                try:
                    z = run_z1(jitter_ends(chains, k), box, partners=False).Z
                    zs.append(float(np.mean(z[bridge])) if bridge.any() else float(np.mean(z)))
                except Z1PlusFailed as exc:
                    why = exc
                k += 1
            if not zs:
                raise RuntimeError(f"Z1+ failed on every seed of the drawn network: {why}")
            return float(np.mean(zs))

        return measure

    def _place_for_target(self, specs, box, shape, stream, knobs, target, place, mol):
        """Meet ``target_Z`` with the coil radius, on the build.

        The radius is searched on the drawn network (every strand drawn from
        a fresh ``stream()``, the placement's own, at each try, so the
        reading moves only with the radius), the network is settled once at the radius found and read
        again, and if settling moved it more than 5 % (or the controller's
        tolerance, if tighter) the search runs once more, aimed off by what
        settling added. Z1+ is read over 2 seeds while searching and 4 on the
        settled build. ``met`` is read against the controller's tolerance.
        """
        from topon.conformation.atomistic import coil_radius_for, draw_network

        ctl = self.config.conformation.entanglement.controller
        search_meter = self._z_meter(specs, box, mol, seeds=2)
        check_meter = self._z_meter(specs, box, mol, seeds=4)

        def drawn_z(radius):
            paths, _infos, _shared = draw_network(specs, shape, stream(),
                                                  **{**knobs, "radius": radius})
            return search_meter(paths)

        def settled_z(coords):
            return check_meter([np.vstack([s.start] + [coords[int(i)] for i in s.backbone]
                                          + [s.end]) for s in specs])

        close = min(float(ctl.tolerance), 0.05)
        aim, rounds = target, []
        for _attempt in range(2):
            found = coil_radius_for(aim, drawn_z)
            coords, report = place(found.radius)
            z = settled_z(coords)
            rounds.append({"aim": round(aim, 4), **found.as_dict(),
                           "z_settled": round(z, 4)})
            print(f"  Z target {target:g}: coil radius {found.radius:.2f} A "
                  f"({found.status}, {len(found.trace)} draws), Z1+ per bridge "
                  f"{found.z:.3f} drawn, {z:.3f} settled")
            if (abs(z - target) <= close * target
                    or found.status in ("floor", "ceiling")):
                break
            aim = max(1e-3, aim - (z - found.z))
        final = rounds[-1]
        met = abs(final["z_settled"] - target) <= ctl.tolerance * target
        if not met:
            import warnings as _warnings
            _warnings.warn(
                f"target_Z {target:g}: the settled build reads "
                f"{final['z_settled']:.3f} per bridge at coil radius "
                f"{final['radius']:.2f} A ({final['status']}), outside the "
                f"tolerance {ctl.tolerance:g}", RuntimeWarning, stacklevel=2)
        return coords, report, {"target": target, "radius": final["radius"],
                                "z_settled": final["z_settled"], "met": bool(met),
                                "tolerance": ctl.tolerance, "rounds": rounds}

    def _record_defects(self) -> None:
        """Put the defects stage's record in the run manifest.

        Requested against achieved for every defect class, the chemical
        against the effective P(f), and the bead budget that sized the
        box. ``topon inspect`` renders it beside stage 1's sculpt record.
        As with that one, a manifest that cannot be written costs an
        inspection detail and nothing more. A random-crosslinked network's
        chain cover goes in beside it, as ``chains``, and a rerun of the
        study as an end-linked one drops that section again.
        """
        from topon.core.manifest import drop_stage, record_stage

        if self.graph is None:
            return
        covered = (self.config.topology.generator.architecture
                   == "random_crosslinked")
        if not covered:
            try:
                drop_stage(self.output_dir, "chains")
            except Exception as exc:
                print(f"  (could not update the run manifest: {exc})")
        for section, key in (("defects", "defects"), ("chains", "chain_assignment")):
            entry = self.graph.graph.get(key)
            if not entry or (section == "chains" and not covered):
                continue
            try:
                record_stage(
                    self.output_dir, section, dict(entry),
                    study=self.config.study.name,
                )
            except Exception as exc:
                print(f"  (could not write the {section} manifest: {exc})")

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
        # The noise and the push given to two atoms that coincide come from
        # streams of their own, keyed on the study name. From the global
        # stream they were unseeded unless the caller seeded it, so a config
        # with every seed pinned wrote a different relaxed file on every run.
        study = self.config.study.name
        noisy = cm.apply_noise(
            conformed,
            magnitude=conf_params["noise_magnitude"],
            rng=_stable_rng("noise", study),
        )
        cm.resolve_overlaps(
            noisy,
            roles,
            cutoff=conf_params["overlap_cutoff"],
            max_iters=conf_params["overlap_max_iters"],
            periodicity=periodicity,
            rng=_stable_rng("overlap", study),
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

        sim_cfg = dict(self.raw_config.get("simulation", {}))
        experimental = self.raw_config.get("experimental", {})
        # LammpsInputGenerator branches on "cg" vs "atomistic" literals
        # (see topon/writers/lammps_inputs.py); the schema's chemistry.model_type
        # uses "coarse_grained" / "atomistic". Map at the call site rather than
        # touching every comparison in the writer.
        model = "cg" if self.config.chemistry.model_type == "coarse_grained" else "atomistic"
        placed = getattr(self, "_placement_used", None)
        if model == "atomistic":
            # A settled build relaxes on the hard-backbone deck with its
            # backbone dumped for the crossing detector, unless asked
            # otherwise; the historic placement keeps the historic deck. The
            # writer's own defaults (soft_push, no dump) are the workflow
            # route's and a direct caller's.
            sim_cfg.setdefault("atomistic_protocol",
                               "soft_push" if placed is None else "hard_backbone")
            if placed is not None:
                sim_cfg.setdefault("backbone_dump_every", 10)

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
        # The hard-backbone deck keeps the backbone atom types hard, so it
        # needs to know which they are; stage 4 recorded them.
        backbone = {"backbone_types": getattr(self, "_backbone_types", None),
                    "n_atom_types": getattr(self, "_n_atom_types", None)}
        if getattr(self, "_ring_types", None):
            backbone["ring_types"] = self._ring_types
        if (model == "atomistic" and gen.atomistic_protocol == "hard_backbone"
                and placed is None):
            import warnings as _warnings
            _warnings.warn(
                "simulation.atomistic_protocol 'hard_backbone' with the historic "
                "placement: backbone atoms start a third of a bond apart along "
                "their chords, and the hard core pushes them apart from the first "
                "step; set conformation.atomistic_placement to start them at "
                "bond length", RuntimeWarning, stacklevel=2)
        gen.write_serial_soft_minimization(
            settings_file="system.in.settings",
            model_type=model,
            force_field=ff,
            **backbone,
        )
        gen.write_parallel_production(
            settings_file="system.in.settings",
            model_type=model,
            force_field=ff,
            **charmm_style,
            **backbone,
        )
        print(f"  LAMMPS scripts written to: {self.output_dir / '04_Simulation'}")
        print()
