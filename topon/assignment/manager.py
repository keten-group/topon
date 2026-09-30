"""
Assignment Manager for Topon.

Orchestrates all graph attribute assignment operations:
- Node types
- Edge types
- DP distribution
- Defects
- Entanglements
- Grafts
- Copolymers
"""

from typing import Optional
import networkx as nx
import numpy as np

from topon.config.schema import AssignmentConfig
from topon.assignment import node_types, edge_types, dp_distribution


class AssignmentManager:
    """
    Manages assignment of attributes to graph nodes and edges.
    
    Assignment order:
    1. Node types (based on degree/position/random)
    2. Edge types (uniform/random/composite)
    3. DP distribution (per edge type)
    4. Defects (after types assigned)
    5. Entanglements (after defects)
    6. Grafts (per edge type)
    7. Copolymers (per edge type)
    """
    
    def __init__(self, G: nx.MultiGraph, dims: Optional[np.ndarray],
                 config: AssignmentConfig, max_functionality: Optional[int] = None,
                 bead_density: Optional[float] = None,
                 architecture: str = "end_linked",
                 chains_decided: bool = False):
        """
        Initialize the assignment manager.
        
        Args:
            G: NetworkX MultiGraph with node positions.
            dims: Box dimensions for periodic boundary calculations.
            config: Assignment configuration.
            max_functionality: Chemical valence ceiling for a junction, from
                ``topology.generator.max_functionality``. Defect placement
                needs it to know how much valence is spare. Defaults to the
                highest degree the graph already carries (at least 4).
            bead_density: Beads per sigma^3 of a coarse-grained build
                (``chemistry.target_density``), which with the bead count
                sets the length of a cell unit. The chain cover of a
                random-crosslinked network reads it for its chord floor;
                None (atomistic, or a direct caller) leaves the floor off.
            architecture: ``topology.generator.architecture``. The chains
                are covered only for ``"random_crosslinked"``, whatever tag
                a loaded graph carries.
            chains_decided: the graph came from a crosslinked melt
                (``topology.source = "crosslink"``), whose strands already
                carry their DP and chain, so neither the DP draw nor the
                chain cover runs.
        """
        self.G = G
        self.dims = dims
        self.config = config
        self.max_functionality = max_functionality
        self.bead_density = bead_density
        self.architecture = architecture
        self.chains_decided = bool(chains_decided)
        self.defect_report: dict = {}
        self.chain_report: dict = {}
        
        # Analysis results (populated by analyze())
        self.analysis = {}
    
    def analyze(self) -> dict:
        """
        Analyze the graph to determine max possible modifications.
        
        Returns:
            Dict with analysis results.
        """
        print("Analyzing graph...")
        
        # Degree distribution
        degrees = [d for _, d in self.G.degree()]
        degree_counts = {}
        for d in range(max(degrees) + 1):
            degree_counts[d] = degrees.count(d)
        
        # Defect capacity, per class
        from topon.assignment import defects
        defect_analysis = defects.analyze_defect_potential(
            self.G, max_f=self._max_f()
        )

        self.analysis = {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "degree_distribution": degree_counts,
            "max_primary_loops": defect_analysis["max_possible_primary_loops"],
            "existing_primary_loops": defect_analysis["existing_primary_loops"],
            "max_secondary_loops": defect_analysis["max_possible_secondary_loops"],
            "existing_secondary_loops": defect_analysis["existing_secondary_loops"],
            "max_entanglements": None,  # spatial estimate not yet implemented; falls back to len(candidates)
        }
        
        self._print_analysis()
        return self.analysis
    
    def _print_analysis(self) -> None:
        """Print analysis report."""
        print(f"  Nodes: {self.analysis['num_nodes']}")
        print(f"  Edges: {self.analysis['num_edges']}")
        print(f"  Degree distribution:")
        for d, count in sorted(self.analysis['degree_distribution'].items()):
            if count > 0:
                print(f"    d={d}: {count}")
    
    def run(self) -> nx.MultiGraph:
        """
        Run all assignment operations.
        
        Returns:
            The modified graph with all attributes assigned.
        """
        print("Running assignment...")
        
        # 1. Assign node types
        self._assign_node_types()
        
        # 2. Assign edge types
        self._assign_edge_types()
        
        # 3. Assign DP distribution (a crosslinked melt has its DP already)
        if self.chains_decided:
            print("  DP and chains from the crosslinked melt (stage 1), kept")
        else:
            self._assign_dp()

        # 4. Apply defects (if enabled)
        if self._defects_requested():
            self._apply_defects()

        # 4b. Chains through the junctions, for a random-crosslinked network
        if self.architecture == "random_crosslinked" and not self.chains_decided:
            self._assign_chains()

        # 5. Select entanglements (if enabled)
        if self.config.entanglements.enabled:
            self._select_entanglements()
        
        # 6. Assign grafts (if enabled)
        if self.config.grafts.enabled:
            self._assign_grafts()
        
        # 7. Assign copolymers (if enabled)
        if self.config.copolymer.enabled:
            self._assign_copolymers()
        
        return self.G
    
    def _assign_node_types(self) -> None:
        """Assign node types based on configuration."""
        print("  Assigning node types...")
        node_types.assign_node_types(self.G, self.config.node_types)
    
    def _assign_edge_types(self) -> None:
        """Assign edge types based on configuration."""
        print("  Assigning edge types...")
        edge_types.assign_edge_types(self.G, self.config.edge_types, self.dims)
    
    def _assign_dp(self) -> None:
        """Assign DP values to edges."""
        print("  Assigning DP values...")
        dp_distribution.assign_dp(self.G, self.config.dp_distribution)
    
    def _max_f(self) -> int:
        """The chemical valence ceiling defect placement works against."""
        if self.max_functionality:
            return int(self.max_functionality)
        degrees = [int(d) for _, d in self.G.degree()]
        return max(4, max(degrees) if degrees else 4)

    def _defects_requested(self) -> bool:
        cfg = self.config.defects
        return any(
            getattr(cfg, name).enabled
            for name in ("primary_loops", "secondary_loops", "triangles",
                         "four_cycles", "sol_chains")
        )

    def _apply_defects(self) -> None:
        """Run the post-sculpt defects stage.

        Primary loops (self-loops), secondary loops (parallel strands),
        triangles, four-cycles and sol, with the requested-vs-achieved
        record left on the graph as ``G.graph["defects"]`` for the manifest
        and ``topon inspect``.
        """
        print("  Applying defects...")
        from topon.assignment import defects

        self.defect_report = defects.apply_defects(
            self.G,
            self.config.defects,
            dp_default=int(self.config.dp_distribution.default.mean),
            # The ceiling every defect placement works against. Pipeline
            # passes the junction's valence ceiling, which on the atomistic
            # route is the crosslinker's four bonds rather than the
            # topology's max_functionality.
            max_f=self._max_f(),
        )
        for name in ("primary_loops", "secondary_loops", "triangles",
                     "four_cycles", "sol_chains"):
            record = self.defect_report.get(name)
            if not record:
                continue
            achieved = record.get("achieved", 0)
            requested = record.get("requested", achieved)
            print(f"    {name}: {achieved} of {requested} requested")
        print(f"    effective P(f): {self.defect_report['effective_degree']}")
        print(f"    chemical  P(f): {self.defect_report['chemical_degree']}")
        print(f"    beads (incl. loops and sol): "
              f"{self.defect_report['bead_budget']['total_beads']}")

    def _bonds_per_unit(self) -> Optional[float]:
        """Design bonds per cell unit of a coarse-grained build, or None.

        The chemistry stage sizes the box from the bead count at
        ``bead_density``. The count is fixed before the beads are split:
        every chain has ``assignment.chains.dp`` beads, a junction is built
        as one bead for all the chains passing it, and sol chains add theirs.
        Graft beads, placed after this, are not counted.
        """
        if self.bead_density is None or self.dims is None:
            return None
        from topon.assignment.chains import KG_BOND

        G = self.G
        ends = sum(1 for n in G if G.degree(n) == 1)
        junctions = [n for n in G if G.degree(n) >= 2]
        passes = sum(G.degree(n) for n in junctions) // 2
        sol = G.graph.get("sol_chains") or {}
        if isinstance(sol, dict) and sol.get("dps"):
            sol_beads = sum(int(d) * int(k) for d, k in sol["dps"].items())
        else:
            sol_beads = (int(sol.get("count", 0)) * int(sol.get("dp", 0))
                         if isinstance(sol, dict) else 0)
        beads = (ends // 2) * int(self.config.chains.dp) - passes \
            + len(junctions) + sol_beads
        cell = float(np.prod(np.asarray(self.dims, float)))
        scale = (beads / float(self.bead_density) / cell) ** (1.0 / 3.0)
        return scale / KG_BOND

    def _assign_chains(self) -> None:
        """Cover a random-crosslinked graph's strands with chains.

        Every strand's ``dp`` becomes the beads between its crosslinks on
        its chain (:mod:`topon.assignment.chains`), replacing the per-strand
        draw of step 3. Sol chains are chains of the same length unless
        ``defects.sol_chains.dp`` says otherwise, and the bead budget the
        defects stage recorded is counted again with them.
        """
        print("  Assigning chains through the junctions...")
        from topon.assignment import chains, defects

        cfg = self.config.chains
        sol = self.G.graph.get("sol_chains")
        if isinstance(sol, dict) and self.config.defects.sol_chains.dp is None:
            sol["dp"] = int(cfg.dp)
            rec_sol = (self.G.graph.get("defects") or {}).get("sol_chains")
            if isinstance(rec_sol, dict):
                rec_sol["dp"] = int(cfg.dp)
        seed = (int(cfg.seed) if cfg.seed is not None
                else int(np.random.randint(0, 2 ** 31 - 1)))
        bpu = self._bonds_per_unit() if cfg.chord_floor else None
        rec = chains.assign_chains(
            self.G, int(cfg.dp), np.random.default_rng(seed),
            reactive_every=cfg.reactive_every, target=cfg.passes,
            bonds_per_unit=bpu)
        rec["seed"] = seed
        rec["dp"] = int(cfg.dp)
        rec["bonds_per_unit"] = None if bpu is None else round(bpu, 6)
        self.chain_report = rec
        self.G.graph["chain_assignment"] = rec
        record = self.G.graph.get("defects")
        if record and "bead_budget" in record:
            sol = self.G.graph.get("sol_chains") or {}
            record["bead_budget"] = defects.bead_budget(
                self.G, dp_default=int(cfg.dp),
                sol_count=int(sol.get("count", 0)) if isinstance(sol, dict) else 0,
                sol_dp=sol.get("dp") if isinstance(sol, dict) else None)
        print(f"    {rec['chains']} chains of {cfg.dp} beads, "
              f"{rec['passes_mean']:.2f} junctions passed per chain "
              f"({rec['rings_merged']} rings merged, "
              f"{rec['exchanges_kept']} tail exchanges)")
        if bpu is not None:
            print(f"    chord floor at {bpu:.3f} bonds per cell unit: "
                  f"{rec['chains_chord_unmet']} chains could not meet it, "
                  f"{rec['strands_too_short']} strands shorter than their chord")

    def _select_entanglements(self) -> None:
        """Select entanglement pairs."""
        print("  Selecting entanglements...")
        from topon.assignment import entanglements
        self.entangled_pairs = entanglements.select_entanglements(
            self.G, 
            self.config.entanglements, 
            self.dims,
            self.analysis.get("max_entanglements")
        )
    
    def _assign_grafts(self) -> None:
        """Assign graft side-chain information to edges.

        For each edge whose ``edge_type`` has a graft config, randomly
        selects backbone positions for side-chain attachment based on
        ``graft_density`` and writes three attributes to the edge:

        * ``graft_positions`` — list of backbone bead indices (0-based)
        * ``graft_dp``        — number of beads per side chain
        * ``graft_monomer``   — monomer/bead-type name for side-chain beads
        """
        import random

        graft_cfg = self.config.grafts.per_edge_type
        if not graft_cfg:
            print("  Assigning grafts... (no per_edge_type config, skipping)")
            return

        print("  Assigning grafts...")
        total = 0
        for u, v, key, data in self.G.edges(keys=True, data=True):
            edge_type = data.get("edge_type", "A")
            conf = graft_cfg.get(edge_type)
            if conf is None:
                continue
            dp = data.get("dp", 1)
            density = conf.graft_density
            positions = [k for k in range(dp) if random.random() < density]
            self.G[u][v][key]["graft_positions"] = positions
            self.G[u][v][key]["graft_dp"] = conf.side_chain_dp
            self.G[u][v][key]["graft_monomer"] = conf.side_chain_monomer
            total += len(positions)

        print(f"    Grafts assigned: {total} side chains across "
              f"{self.G.number_of_edges()} edges")

    def _assign_copolymers(self) -> None:
        """Assign per-position monomer sequences to edges.

        For each edge whose ``edge_type`` has a copolymer config, generates
        a monomer sequence of length ``dp`` using
        :func:`topon.chemistry.sequences.generate_monomer_sequence` and
        writes it as the ``monomer_sequence`` edge attribute.
        """
        from topon.chemistry.sequences import generate_monomer_sequence

        cop_cfg = self.config.copolymer.per_edge_type
        if not cop_cfg:
            print("  Assigning copolymers... (no per_edge_type config, skipping)")
            return

        print("  Assigning copolymers...")
        assigned = 0
        for u, v, key, data in self.G.edges(keys=True, data=True):
            edge_type = data.get("edge_type", "A")
            conf = cop_cfg.get(edge_type)
            if conf is None:
                continue
            dp = data.get("dp", 1)
            seq_cfg = {
                "arrangement": conf.arrangement,
                "composition": [
                    {"monomer": c.monomer, "fraction": c.fraction}
                    for c in conf.composition
                ],
            }
            self.G[u][v][key]["monomer_sequence"] = generate_monomer_sequence(
                dp, seq_cfg, default_monomer=edge_type
            )
            assigned += 1

        print(f"    Copolymer sequences assigned to {assigned} edges")
