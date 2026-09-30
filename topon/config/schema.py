"""
Pydantic schema definitions for Topon configuration.

This module defines all configuration models using Pydantic for validation.
The design follows the principle: Assignment = Abstract Types, Chemistry = Concrete Molecules.
"""

import warnings
from typing import Optional, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# STUDY CONFIG
# =============================================================================

class StudyConfig(BaseModel):
    """Study-level configuration."""
    name: str = Field(default="my_network", description="Name of the study")
    output_dir: str = Field(default="./output", description="Output directory path")


# =============================================================================
# TOPOLOGY CONFIG
# =============================================================================

class GeneratorConfig(BaseModel):
    """C generator configuration.

    The one nested section that refuses unknown keys. Pydantic's default
    is to ignore them, so ``"seed": 7`` under ``topology.generator``
    validated cleanly and did nothing before there was a seed field (0.2.0
    refused it; 0.4.0 made it real), and a user who wrote it reasonably
    believed they had pinned the graph. The failure was silent and
    surfaced much later as a run that would not reproduce. Any other
    misspelt key would do the same, which is why the refusal stays.

    Scoped to this model on purpose. Every other nested section still
    ignores unknown keys, which ``topon doctor`` warns about instead
    (``unknown_config_keys``), because forbidding one level up would
    reject configs that carry other tools' parameters under
    ``topology``.
    """

    model_config = ConfigDict(extra="forbid")

    exe_path: Optional[str] = Field(default=None, description="Path to generator executable")
    lattice_size: str = Field(default="6x6x6", description="Lattice dimensions (e.g., '6x6x6')")
    lattice_type: Literal["SC", "BCC", "FCC", "Diamond", "MIX"] = Field(
        default="SC",
        description=(
            "Lattice type. SC/BCC/FCC are the canonical single lattices. "
            "Diamond is 4-coordinated by construction, so a max_func=4 "
            "network needs no pruning. MIX overlays SC/BCC/FCC basis "
            "sites in one cubic cell using `mix_fractions`. Every type "
            "takes `neighbour_cutoff`, the candidate-edge range."
        ),
    )
    mix_fractions: dict[str, float] = Field(
        default={"SC": 1.0, "BCC": 0.0, "FCC": 0.0},
        description=(
            "Sublattice fractions for lattice_type='MIX'. Must sum to 1. "
            "Every cell carries the corner site all three lattices share; "
            "BCC contributes its body-centre site with probability "
            "mix_fractions['BCC'] and FCC each of its three face sites with "
            "probability mix_fractions['FCC']. The SC entry is the remainder "
            "(corner only) and so has no site of its own."
        ),
    )
    neighbour_cutoff: float = Field(
        default=1.0, gt=0,
        description=(
            "Candidate-edge range for every lattice type, in cell units "
            "(the simple-cubic site spacing). The default 1.0 is the "
            "canonical nearest-neighbour lattice; any other value admits "
            "every pair of sites within that distance under the minimum "
            "image, so crosslinkers several site spacings apart become "
            "candidate partners, which is what end-linked networks need. "
            "Rule of thumb: the 95th percentile of the strand's junction-"
            "to-junction separation divided by the site spacing (2.1 for "
            "DP 20, 3.3 for DP 100 at the reference density). Keep it "
            "under a third of the box. mix_cutoff is the deprecated name."
        ),
    )
    neighbour_shells: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "Number of simple-cubic neighbour shells to admit, an "
            "alternative to neighbour_cutoff on SC (1 -> 1.0, 2 -> 1.42, "
            "3 -> 1.74, 4 -> 2.01, 5 -> 2.24, 6 -> 2.45, 8 -> 3.01). On any "
            "other lattice it is accepted only next to an explicit "
            "neighbour_cutoff, which then governs."
        ),
    )
    periodicity: str = Field(default="111", description="Periodicity string (e.g., '111' for all periodic)")
    max_functionality: int = Field(default=6, ge=1, description="Maximum node functionality")
    max_trials: int = Field(default=1000000, ge=1, description="Maximum trials for generator")
    max_saves: int = Field(default=1, ge=1, description="Number of networks to save")
    degree_distribution: str = Field(
        default="0:0,1:0",
        description="Target degree distribution (e.g., '0:15,1:30,3:43')"
    )
    search: Optional[Literal["strict", "exact"]] = Field(
        default=None,
        description=(
            "Which sculptor turns the candidate-edge lattice into the "
            "network. 'strict' removes edges one at a time and records a "
            "move history (what the sculpting animation replays). 'exact' "
            "solves for a subgraph with exactly the requested degree "
            "counts (degree-constrained subgraph by augmenting paths); it "
            "needs a count for every degree from 0 to max_functionality "
            "and has no move history. Left unset, 'exact' is used when the "
            "degree_distribution pins every degree, 'strict' otherwise."
        ),
    )
    min_giant_fraction: float = Field(
        default=0.99, gt=0, le=1,
        description=(
            "Smallest fraction of the active sites the largest connected "
            "component may hold for an exact sculpt to be accepted. The "
            "DP-20 end-linked reference network sits at 0.997 and the "
            "DP-100 one at 1.0, so the default 0.99 admits a small sol "
            "fraction without admitting a shattered graph. Unused by the "
            "strict sculptor, which enforces full connectivity of the "
            "active subgraph."
        ),
    )
    odd_walks: Optional[bool] = Field(
        default=None,
        description=(
            "Exact search only. When an attempt's repair ends short on a "
            "scaffold with an odd cycle (FCC, MIX, SC or BCC beyond the "
            "first shell), search the augmenting walk again over (site, "
            "next move) states and close what the marked walk left open. "
            "Unset means on. A seed whose earlier attempt ended short on "
            "such a scaffold then lands on that attempt, with a different "
            "graph than with the walks off, as before 0.4.0. False gives that graph."
        ),
    )
    architecture: Literal["end_linked", "random_crosslinked"] = Field(
        default="end_linked",
        description=(
            "What an edge of the sculpted graph is. 'end_linked': every "
            "strand is a chain and every junction a crosslinker, the "
            "historic route. 'random_crosslinked': crosslinks sit along "
            "longer chains, so a junction is a crosslink point that two "
            "chains pass through (degree 4 for a crosslink between two "
            "chain beads) and a degree-1 site is a chain end; stage 3 then "
            "covers the strands with chains of assignment.chains.dp beads "
            "(topon.assignment.chains) and gives each strand the beads "
            "between its crosslinks. Degrees above 2 must be even, since a "
            "chain passing a junction takes two of its strands."
        ),
    )
    seed: Optional[int] = Field(
        default=None, ge=0, lt=2 ** 32,
        description=(
            "Pins the generated graph. The Python generator draws from its "
            "own streams seeded with this number, so seed n gives the graph "
            "that random.seed(n) and np.random.seed(n) just before "
            "generating gave, without touching the global streams; the C "
            "route gets a TOPON_SEED drawn the same way. Only stage 1 is "
            "pinned: the later stages still draw from the global streams "
            "(assignment.defects.seed pins the defects). null keeps the old "
            "behaviour, the generator drawing from the global streams."
        ),
    )

    @field_validator("mix_fractions")
    @classmethod
    def _check_mix_fractions(cls, v: dict[str, float]) -> dict[str, float]:
        """Reject fraction sets that cannot describe a lattice.

        Checked here rather than in the generator so a bad config fails at
        load time, before a long generation run starts.
        """
        allowed = {"SC", "BCC", "FCC"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(
                f"mix_fractions has unknown key(s) {sorted(unknown)}; "
                f"allowed keys are {sorted(allowed)}"
            )
        negative = {k: x for k, x in v.items() if x < 0}
        if negative:
            raise ValueError(f"mix_fractions must be non-negative, got {negative}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"mix_fractions must sum to 1, got {total:g} from {v}. "
                f"The fractions partition the crosslinker population, so a "
                f"sum below 1 would silently thin the lattice and a sum "
                f"above 1 would over-fill it."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _accept_mix_cutoff_alias(cls, data):
        """Read the deprecated ``mix_cutoff`` key as ``neighbour_cutoff``.

        ``mix_cutoff`` was the MIX-only cutoff before 0.2.0; the range is now
        a parameter of every lattice under the new name. The old key keeps
        working, with a warning, so existing configs load unchanged. A
        config carrying both with different values is refused rather than
        silently picking one.
        """
        if not isinstance(data, dict) or "mix_cutoff" not in data:
            return data
        data = dict(data)
        legacy = data.pop("mix_cutoff")
        if legacy is None:
            return data
        warnings.warn(
            "topology.generator.mix_cutoff is deprecated; use "
            "neighbour_cutoff, which sets the candidate-edge range for "
            "every lattice type.",
            FutureWarning, stacklevel=2,
        )
        current = data.get("neighbour_cutoff")
        if current is not None and float(current) != float(legacy):
            raise ValueError(
                f"mix_cutoff ({legacy}) and neighbour_cutoff ({current}) "
                f"disagree; mix_cutoff is a deprecated alias of "
                f"neighbour_cutoff, so give one value under the new name"
            )
        if data.get("neighbour_shells") is not None and float(legacy) == 1.0:
            return data      # the old default yields to a shell count
        data["neighbour_cutoff"] = legacy
        return data

    @model_validator(mode="after")
    def _random_crosslinked_degrees(self):
        """Refuse an odd junction degree for a random-crosslinked network.

        A chain passing a junction takes two of its strands, so a junction
        of odd degree needs a chain to end inside it, which a crosslink
        between two chain beads never gives.
        """
        if self.architecture != "random_crosslinked":
            return self
        from topon.topology.degree_matching import parse_degree_distribution

        counts, _ = parse_degree_distribution(self.degree_distribution)
        odd = sorted(d for d, n in counts.items() if d >= 3 and d % 2 and n > 0)
        if odd:
            raise ValueError(
                f"architecture 'random_crosslinked' needs even junction "
                f"degrees (two strands per chain passing), and "
                f"degree_distribution asks for degree(s) {odd}. Degree 1 is "
                f"a chain end, 2 a junction that carries a primary loop, "
                f"4 a crosslink between two chain beads.")
        return self

    @model_validator(mode="after")
    def _resolve_neighbour_cutoff(self):
        """Fold ``neighbour_shells`` into ``neighbour_cutoff`` and check both.

        After this ``neighbour_cutoff`` is the effective range for every
        consumer: both generators, the C wrapper and ``topon doctor``. The
        arithmetic lives in ``topon.topology.shells``; imported here
        rather than at module level so the schema stays a leaf module.
        """
        from topon.topology.shells import resolve_neighbour_cutoff

        explicit = "neighbour_cutoff" in self.model_fields_set
        self.neighbour_cutoff = resolve_neighbour_cutoff(
            self.lattice_type,
            neighbour_cutoff=self.neighbour_cutoff if explicit else None,
            neighbour_shells=self.neighbour_shells,
        )
        return self

    @property
    def mix_cutoff(self) -> float:
        """Deprecated alias of ``neighbour_cutoff``; reads and writes it."""
        return self.neighbour_cutoff

    @mix_cutoff.setter
    def mix_cutoff(self, value: float) -> None:
        warnings.warn(
            "mix_cutoff is deprecated; set neighbour_cutoff instead.",
            FutureWarning, stacklevel=2,
        )
        self.neighbour_cutoff = float(value)


class ExistingFilesConfig(BaseModel):
    """Configuration for loading existing topology files."""
    nodes_file: Optional[str] = Field(default=None, description="Path to .nodes file")
    edges_file: Optional[str] = Field(default=None, description="Path to .edges file")
    gpickle_file: Optional[str] = Field(default=None, description="Path to .gpickle file")


class CrosslinkChainConfig(BaseModel):
    """One chain type of a melt crosslinked along its chains.

    The reactive beads come from exactly one of ``reactive_every`` (every
    so many beads from ``reactive_start``), ``reactive`` (the bead indices)
    or ``sequence`` (one bead per residue, ``crosslink_residue`` reactive,
    through ``topon.protein_network.sequence.plan_chain``).
    """

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, description="Chains of this type.")
    dp: Optional[int] = Field(
        default=None, ge=3,
        description="Beads per chain, ends and crosslinked beads included "
                    "(from the sequence when one is given).")
    reactive_every: Optional[int] = Field(
        default=None, ge=1,
        description="Every so many beads may crosslink, from reactive_start.")
    reactive_start: int = Field(default=1, ge=1,
                                description="First reactive bead (0 is a chain end).")
    reactive: Optional[list[int]] = Field(
        default=None, description="The reactive bead indices, interior beads only.")
    sequence: Optional[str] = Field(
        default=None, description="One-letter residue sequence of one repeat.")
    repeats: int = Field(default=1, ge=1, description="Repeats of the sequence per chain.")
    crosslink_residue: str = Field(default="Y", description="The residue that crosslinks.")
    site_type: str = Field(
        default="X",
        description="Label of this type's reactive beads, for crosslinking.pairs "
                    "(a sequence labels them with its crosslink residue).")
    name: Optional[str] = Field(default=None, description="A name for the record.")

    @model_validator(mode="after")
    def _one_source_of_sites(self):
        given = [k for k in ("reactive_every", "reactive", "sequence")
                 if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError(
                f"a crosslinking chain type needs exactly one of reactive_every, "
                f"reactive or sequence, got {given or 'none'}")
        if self.sequence is None and self.dp is None:
            raise ValueError("a crosslinking chain type needs dp (or a sequence)")
        if self.sequence is not None and self.dp is not None:
            raise ValueError("give dp or a sequence, not both: the sequence sets the length")
        return self


class CrosslinkingConfig(BaseModel):
    """A melt grown on a lattice and crosslinked where reactive beads touch.

    Read when ``topology.source`` is ``"crosslink"``
    (:mod:`topon.topology.chain_crosslinking`). The chains, every strand's
    DP and the chain it belongs to are decided here, in stage 1, so stage 3
    leaves them as they are.
    """

    model_config = ConfigDict(extra="forbid")

    chains: list[CrosslinkChainConfig] = Field(
        default_factory=list, description="The chain types.")
    crosslinks: Optional[int] = Field(
        default=None, ge=0, description="Crosslinks to make (exactly one target).")
    conversion: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Fraction of the reactive beads that react (exactly one target).")
    per_chain: Optional[float] = Field(
        default=None, ge=0,
        description="Crosslinked beads per chain; a crosslink counts on both "
                    "chains it joins (exactly one target).")
    packing: float = Field(
        default=0.4, gt=0, le=0.6,
        description="Fraction of the lattice sites the melt fills. It sets how "
                    "evenly the crosslinks spread (a physical parameter, not a "
                    "resolution): the bond-fluctuation references were 0.395.")
    persistence: Optional[float] = Field(
        default=None, ge=0, lt=1,
        description="Weight of a straight step against 1/4 of the rest for each "
                    "turn. Unset (and no c_inf), 0.2, the five open directions "
                    "alike.")
    c_inf: Optional[float] = Field(
        default=None, gt=1,
        description="The chains' characteristic ratio in lattice bonds; the "
                    "persistence is solved for. Give this or persistence.")
    contact_radius: float = Field(
        default=1.5, ge=1,
        description="Reactive beads this close (lattice units) may crosslink. "
                    "1.5 takes face and edge neighbours, free of the cubic "
                    "lattice's parity; 1.0 the face neighbours only (the "
                    "bond-fluctuation generator's rule).")
    max_radius: float = Field(
        default=3.0, ge=1, le=6,
        description="How far the contact shells may go out when the contacts "
                    "cannot make the count.")
    min_gap: int = Field(
        default=6, ge=3,
        description="Fewest bonds between two beads of one chain that crosslink "
                    "each other (3 or more: the builder needs a primary loop of "
                    "two beads).")
    pairs: Optional[list[tuple[str, str]]] = Field(
        default=None,
        description="Site types that may pair, e.g. [['A', 'B']]; unset, any pair.")
    keep_windings: bool = Field(
        default=False,
        description="Refuse a crosslink that makes a strand run across half the "
                    "box in the melt, which the builder would draw the short way "
                    "round. Off, such strands are made and counted "
                    "(strands_rewound); a box smaller than the chains needs off.")
    min_dangling_dp: Optional[int] = Field(
        default=None, ge=0,
        description="Fewest beads a dangling strand keeps between its crosslink "
                    "and the chain's end bead; reactive beads closer to a chain "
                    "end are left out. Unset, 0 on the coarse-grained route (a "
                    "crosslink on the bead next to a chain end bonds the end bead "
                    "straight to the junction) and 1 on the atomistic route, "
                    "whose strands join their junctions through a monomer.")
    seed: Optional[int] = Field(
        default=None, ge=0, lt=2 ** 32,
        description="Pins the melt and the crosslinks. null draws one from "
                    "NumPy's global stream.")

    @model_validator(mode="after")
    def _one_target(self):
        given = [k for k in ("crosslinks", "conversion", "per_chain")
                 if getattr(self, k) is not None]
        if self.chains and len(given) != 1:
            raise ValueError(
                f"topology.crosslinking needs exactly one of crosslinks, "
                f"conversion or per_chain, got {given or 'none'}")
        if self.persistence is not None and self.c_inf is not None:
            raise ValueError("topology.crosslinking: give persistence or c_inf, not both")
        if self.max_radius < self.contact_radius:
            raise ValueError("topology.crosslinking.max_radius is below contact_radius")
        return self


class TopologyConfig(BaseModel):
    """Topology generation/loading configuration."""
    source: Literal["generate", "load", "crosslink"] = Field(
        default="load",
        description=(
            "Whether to generate new topology or load existing. 'crosslink' "
            "grows the chains of topology.crosslinking as a lattice melt and "
            "crosslinks the reactive beads that touch (a network crosslinked "
            "along its chains, with each strand's DP and chain decided here)."
        ),
    )
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    existing_files: ExistingFilesConfig = Field(default_factory=ExistingFilesConfig)
    crosslinking: CrosslinkingConfig = Field(default_factory=CrosslinkingConfig)

    @model_validator(mode="after")
    def _crosslink_has_chains(self):
        if self.source == "crosslink" and not self.crosslinking.chains:
            raise ValueError("topology.source 'crosslink' needs topology.crosslinking.chains")
        return self


# =============================================================================
# ASSIGNMENT CONFIG
# =============================================================================

class DegreeNodeTypeConfig(BaseModel):
    """Degree-based node type assignment."""
    mapping: dict[str, str] = Field(
        default={"1": "end", "2": "A", "3": "A", "4": "A", "5": "A", "6": "A"},
        description="Map degree -> node type"
    )


class PositionalConfig(BaseModel):
    """Positional (layer-based) assignment config."""
    dimension: Literal["x", "y", "z"] = Field(default="z")
    num_layers: int = Field(default=2, ge=1)
    layer_types: list[str] = Field(default=["A", "B"])


class RandomTypeConfig(BaseModel):
    """Random type assignment config."""
    type_ratios: dict[str, float] = Field(
        default={"A": 100},
        description="Type ratios (will be normalized)"
    )


class NodeTypesConfig(BaseModel):
    """Node type assignment configuration."""
    method: Literal["degree", "positional", "random", "explicit"] = Field(default="degree")
    degree: DegreeNodeTypeConfig = Field(default_factory=DegreeNodeTypeConfig)
    positional: PositionalConfig = Field(default_factory=PositionalConfig)
    random: RandomTypeConfig = Field(default_factory=RandomTypeConfig)
    explicit: dict[int, str] = Field(default_factory=dict, description="Per-node ID type assignment")


class UniformEdgeConfig(BaseModel):
    """Uniform edge type config."""
    type: str = Field(default="A")


class CompositeEdgeConfig(BaseModel):
    """Composite/lamellar edge type config."""
    dimension: Literal["x", "y", "z"] = Field(default="z")
    num_layers: int = Field(default=2, ge=1)
    layer_types: list[str] = Field(default=["A", "B"])


class EdgeTypesConfig(BaseModel):
    """Edge type assignment configuration."""
    method: Literal["uniform", "random", "composite"] = Field(default="uniform")
    uniform: UniformEdgeConfig = Field(default_factory=UniformEdgeConfig)
    random: RandomTypeConfig = Field(default_factory=RandomTypeConfig)
    composite: CompositeEdgeConfig = Field(default_factory=CompositeEdgeConfig)


class DPConfig(BaseModel):
    """DP distribution for a type."""
    mean: float = Field(default=25.0, gt=0)
    pdi: float = Field(default=1.0, ge=1.0, description="Polydispersity index (1.0 = monodisperse)")


class DPDistributionConfig(BaseModel):
    """DP distribution configuration."""
    default: DPConfig = Field(default_factory=DPConfig)
    per_edge_type: dict[str, DPConfig] = Field(default_factory=dict)
    endlinked_dangling: bool = Field(
        default=False,
        description=(
            "End-linked convention for dangling strands: the free end site "
            "is the strand's DP-th bead, so the strand itself carries dp - 1 "
            "beads and every chain in the system has exactly DP beads. This "
            "is what `fix bond/create` datasets and `refnet.parse` assume; "
            "leave it off for topon's own convention, where a dangling "
            "strand has dp beads plus its end-cap node."
        ),
    )


class TargetConfig(BaseModel):
    """Target specification for defects/entanglements."""
    enabled: bool = Field(default=False)
    target: int = Field(default=0, ge=0, description="Target count or percentage value")
    target_type: Literal["count", "percentage"] = Field(default="count")


class _DefectCountMixin(BaseModel):
    """Shared `count` handling for the defect classes.

    ``count`` is the number of defects of that class. A value below 1 is
    read as a fraction of the strands in the graph, so
    ``{"count": 0.069}`` and ``{"count": 345}`` say the same thing on a
    5000-strand network. Giving a count switches the class on.
    """

    model_config = {"from_attributes": True, "extra": "forbid"}

    enabled: bool = Field(default=False)
    count: Optional[float] = Field(
        default=None, ge=0,
        description=(
            "Defect count. >= 1 is an absolute number; 0 < count < 1 is a "
            "fraction of the strands in the graph."
        ),
    )

    @model_validator(mode="after")
    def _count_enables(self):
        if self.count is not None and self.count > 0 and not self.enabled:
            object.__setattr__(self, "enabled", True)
        return self


class PrimaryLoopsConfig(_DefectCountMixin):
    """Primary loops: a strand that returns to the junction it left.

    Stored as MultiGraph self-loop edges (``cls="loop"``). A junction's
    *chemical* functionality is its effective functionality plus two per
    loop it carries, which is what the crosslinker's valence sees.

    ``placement``:

    * ``by_effective_degree``: the reference's statistics, one loop on a
      junction whose effective degree is ``max_functionality - 2``, two on
      an effective-degree-0 junction, at random within each class.
    * ``random``: any junction with spare chemical valence.

    ``dp`` overrides the loop strand's length; ``None`` inherits the DP of
    the other strands on that junction (or the DP distribution's mean when
    the junction has none).

    .. deprecated:: 0.2.0
       ``target`` / ``target_type`` are the 0.1.0 fields, when this key
       injected *parallel edges* (secondary loops). A config that sets them
       still gets that old behaviour, with a warning. Use
       ``secondary_loops`` for parallel edges and ``count`` here for
       self-loops.
    """

    placement: Literal["by_effective_degree", "random"] = Field(
        default="by_effective_degree"
    )
    dp: Optional[int] = Field(default=None, gt=0)
    target: int = Field(default=0, ge=0, description="Deprecated: see class doc")
    target_type: Literal["count", "percentage"] = Field(default="count")

    @property
    def is_legacy_parallel_request(self) -> bool:
        """True when this block is a 0.1.0 parallel-edge request."""
        return self.enabled and self.count is None and self.target > 0


class SecondaryLoopsConfig(_DefectCountMixin):
    """Secondary loops: two strands between the same junction pair.

    Placed by the exact sculptor as forced double edges so the degree
    counts stay exact (``topology.generator.search="exact"``); the defects
    stage verifies and records them. On a graph that is already sculpted
    the stage falls back to endpoint-targeted injection.

    ``endpoint_degrees`` is the effective-degree histogram of the pairs the
    doubles sit between, keyed ``"a,b"``, for example
    ``{"4,4": 115, "3,4": 6, "2,4": 7, "2,3": 1}``. ``"auto"`` lets the
    sculptor draw pairs so the multigraph P(f) equals the requested P(f).
    """

    endpoint_degrees: Union[dict[str, int], Literal["auto"]] = Field(
        default="auto",
        description='Endpoint effective-degree histogram keyed "a,b", or "auto"',
    )
    target: int = Field(default=0, ge=0, description="Deprecated alias of count")
    target_type: Literal["count", "percentage"] = Field(default="count")


class CycleDefectConfig(_DefectCountMixin):
    """Higher-order loops: three-cycles (triangles) or four-cycles.

    Both change the degrees of the junctions they touch, so they are
    requested from the exact sculptor with reserved capacity (the sculpt
    target is the requested P(f) minus the defect contribution). Injected
    into an already-sculpted graph instead, they shift P(f); the stage
    reports that deviation rather than hiding it.
    """


class SolChainsConfig(_DefectCountMixin):
    """Sol: chains bonded to no junction at all.

    They are not part of the junction graph (they have no endpoints in
    it), but they carry beads, so they belong in the bead budget that sizes
    the box at the target density. In the DP-20 reference, the 345 primary
    loops and 11 sol chains together hold 7 % of the beads.
    """

    dp: Optional[int] = Field(default=None, gt=0)


class DefectsConfig(BaseModel):
    """Defect injection configuration (the post-sculpt defects stage)."""
    primary_loops: PrimaryLoopsConfig = Field(default_factory=PrimaryLoopsConfig)
    secondary_loops: SecondaryLoopsConfig = Field(default_factory=SecondaryLoopsConfig)
    triangles: CycleDefectConfig = Field(default_factory=CycleDefectConfig)
    four_cycles: CycleDefectConfig = Field(default_factory=CycleDefectConfig)
    sol_chains: SolChainsConfig = Field(default_factory=SolChainsConfig)
    seed: Optional[int] = Field(
        default=None,
        description="Seed for defect placement; None draws from global RNG state",
    )


class KinkParams(BaseModel):
    """Parameters for entanglement kink geometry."""
    overshoot: float = Field(default=0.2, ge=0, le=1)
    z_amp: float = Field(default=0.5, ge=0)
    sigma: float = Field(default=0.15, gt=0)


class EntanglementsConfig(BaseModel):
    """Entanglement configuration."""
    enabled: bool = Field(default=False)
    target: int = Field(default=0, ge=0)
    target_type: Literal["count", "percentage"] = Field(default="count")

    # How an entangled pair is realised in 3D.
    #
    #   * waypoint — the pair is drawn together: both chains are splines
    #     that spiral about their contact in antiphase, so the pair carries
    #     exactly `entanglement_count` windings by construction. Verified
    #     with primitive-path analysis; the default.
    #   * kink — the legacy Gaussian bump aimed at the partner's midpoint.
    #     Each chain is drawn alone, so what survives relaxation is
    #     statistical rather than prescribed. Kept for reproducing 0.1.0
    #     systems.
    method: Literal["waypoint", "kink"] = Field(
        default="waypoint",
        description="Entanglement realisation: 'waypoint' (prescribed "
                    "winding, default) or 'kink' (legacy Gaussian bump).",
    )

    # Distribution mode: specify average crosslinks per chain
    # Formula: total_draws = avg_crosslinks_per_chain * 0.5 * num_chains
    avg_crosslinks_per_chain: Optional[float] = Field(
        default=None, ge=0,
        description="Average crosslinks per chain. If set, uses distribution mode with replacement."
    )

    kink_params: KinkParams = Field(default_factory=KinkParams)

    # Spatial placement bias. Default "uniform" gives the legacy
    # behaviour (homogeneous random selection from crossing
    # candidates). Other kinds reweight the draw pool by a spatial
    # function of each candidate's midpoint center.
    #
    # See :func:`topon.assignment.entanglements.compute_bias_weights`
    # for the supported kinds and their params keys:
    #   * region      — uniform inside a sphere, low outside.
    #                    params: center (fractional [0..1] x/y/z),
    #                            radius (fraction of min(dims)),
    #                            strength (in/out density ratio).
    #   * anti_region — depleted inside a sphere, normal outside.
    #                    params: center, radius, strength.
    #   * gradient    — power-law gradient along an axis.
    #                    params: axis ("x"/"y"/"z"), strength (exponent).
    #   * clusters    — gaussian peaks at multiple centers.
    #                    params: centers (list of fractional xyz),
    #                            sigma (fraction of min(dims)),
    #                            strength.
    placement_bias_kind: Literal[
        "uniform", "region", "anti_region", "gradient", "clusters"
    ] = Field(
        default="uniform",
        description="Spatial bias applied to the entanglement-candidate draw.",
    )
    placement_bias_params: dict = Field(
        default_factory=dict,
        description="Parameters consumed by the placement-bias function.",
    )

    # Neighbour-shell weighting. Empty is the legacy behaviour: every
    # candidate is equally eligible whatever shell it sits in. Naming
    # shells restricts the draw to those named and weights them in
    # proportion, e.g. {"1": 0.6, "2": 0.4}. Shells are numbered from 1,
    # closest first, and are read off the lattice rather than assumed.
    #
    # Only the first shell reliably delivers. Measured with a
    # primitive-path analysis, each pair checked alone after the full
    # protocol: band 1 gave 5 of 7, band 2 gave 2 of 16, band 3 gave 0 of
    # 16. That follows from the pair's gap over the chain's chord, 0.29 in
    # the first shell and 0.50 in the second, which is fixed by the lattice
    # and does not move with the mix fractions or the box size.
    shell_weights: dict = Field(
        default_factory=dict,
        description=(
            "Relative weight per neighbour shell, e.g. {'1': 0.6, '2': 0.4}. "
            "Empty draws from all shells equally. Only the first shell "
            "reliably realises an entanglement."
        ),
    )


class GraftConfig(BaseModel):
    """Graft configuration for a single edge type."""
    graft_density: float = Field(default=0.5, ge=0, le=1)
    side_chain_monomer: str = Field(default="PDMS")
    side_chain_dp: int = Field(default=5, ge=1)
    cap_atom: str | None = Field(
        default=None,
        description=(
            "Atom symbol for the unreacted side-chain end. None (default) "
            "means: cap with the same chemistry as the backbone monomer "
            "(e.g. trimethylsilyl for PDMS = add one extra methyl to the "
            "terminal Si). Set to 'H' to leave the terminal atom hydrogen-"
            "capped (RDKit default valence fill). Currently only used by "
            "the atomistic generator; CG ignores this field."
        ),
    )


class GraftsConfig(BaseModel):
    """Grafts configuration."""
    enabled: bool = Field(default=False)
    per_edge_type: dict[str, GraftConfig] = Field(default_factory=dict)


class CopolymerComposition(BaseModel):
    """Single monomer in copolymer composition."""
    monomer: str
    fraction: float = Field(ge=0, le=1)


class CopolymerTypeConfig(BaseModel):
    """Copolymer config for a single edge type."""
    arrangement: Literal["block", "alternating", "random", "gradient"] = Field(default="block")
    composition: list[CopolymerComposition] = Field(default_factory=list)


class CopolymerConfig(BaseModel):
    """Copolymer configuration."""
    enabled: bool = Field(default=False)
    per_edge_type: dict[str, CopolymerTypeConfig] = Field(default_factory=dict)


class ChainsConfig(BaseModel):
    """Chains through the junctions of a random-crosslinked network.

    Read only when ``topology.generator.architecture`` is
    ``"random_crosslinked"`` (see :mod:`topon.assignment.chains`). Every
    chain has ``dp`` beads; the strands between its crosslinks get the beads
    between them, so the per-strand DP comes from here and
    ``dp_distribution`` is overwritten.
    """

    model_config = ConfigDict(extra="forbid")

    dp: Optional[int] = Field(
        default=None, ge=3,
        description="Beads per chain, ends and crosslinked beads included.")
    reactive_every: int = Field(
        default=1, ge=1,
        description=(
            "Spacing of the beads that may carry a crosslink, from bead 1: "
            "1 is every interior bead, 3 every third one."))
    passes: Optional[list[float]] = Field(
        default=None,
        description=(
            "Target P(a chain passes k junctions) for k = 1, 2, ... . Unset, "
            "the binomial of random crosslinking over the reactive beads "
            "with the graph's own mean."))
    chord_floor: bool = Field(
        default=True,
        description=(
            "Coarse-grained builds only: give every strand at least the "
            "beads that span its chord at the design bond (0.97 sigma), at "
            "the box chemistry.target_density gives, so no strand starts "
            "stretched. Off, the crosslinked beads are uniform along each "
            "chain whatever the chords."))
    seed: Optional[int] = Field(
        default=None, ge=0, lt=2 ** 32,
        description=(
            "Pins the cover and the bead split. null draws the generator's "
            "seed from NumPy's global stream."))


class AssignmentConfig(BaseModel):
    """Complete assignment configuration."""
    node_types: NodeTypesConfig = Field(default_factory=NodeTypesConfig)
    edge_types: EdgeTypesConfig = Field(default_factory=EdgeTypesConfig)
    dp_distribution: DPDistributionConfig = Field(default_factory=DPDistributionConfig)
    defects: DefectsConfig = Field(default_factory=DefectsConfig)
    entanglements: EntanglementsConfig = Field(default_factory=EntanglementsConfig)
    grafts: GraftsConfig = Field(default_factory=GraftsConfig)
    copolymer: CopolymerConfig = Field(default_factory=CopolymerConfig)
    chains: ChainsConfig = Field(default_factory=ChainsConfig)


# =============================================================================
# CHEMISTRY CONFIG
# =============================================================================

class NodeMoleculeConfig(BaseModel):
    """Configuration for a node type's molecule."""
    molecule: str = Field(description="SMILES or molecule name (e.g., 'Si', 'POSS')")
    is_end_cap: bool = Field(default=False, description="Whether this is an end-cap molecule")
    charmm_residue: Optional[str] = Field(
        default=None,
        description="force_field 'charmm': the RTF residue (RESI) this node molecule is.",
    )
    charmm_atom_names: Optional[list[str]] = Field(
        default=None,
        description=("force_field 'charmm': RTF names of the molecule's heavy atoms in "
                     "SMILES order; without it the atoms are matched by graph."),
    )


class MonomerConfig(BaseModel):
    """Monomer definition."""
    smiles: str = Field(description="SMILES string for the repeating unit")
    chain_head: str = Field(default="Si", description="Atom type at chain head")
    chain_tail: str = Field(default="O", description="Atom type at chain tail")
    charmm_residue: Optional[str] = Field(
        default=None,
        description="force_field 'charmm': the RTF residue (RESI) one repeat unit is.",
    )
    charmm_atom_names: Optional[list[str]] = Field(
        default=None,
        description=("force_field 'charmm': RTF names of the repeat unit's heavy atoms in "
                     "SMILES order; without it the atoms are matched by graph."),
    )


class CharmmConfig(BaseModel):
    """CHARMM files and styles for ``chemistry.force_field = "charmm"``.

    topon types atoms from the RTF residues named on the monomers and node
    molecules and takes every term from the parameter files. It never
    substitutes a parameter; a term the files do not define stops the build
    with the full list. CGenFF typing is not done here: run the CGenFF
    program on a new monomer and pass its stream file.
    """

    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(
        default_factory=list,
        description=("RTF, PRM and stream (.str) files, read in order. A name written "
                     "'bundled:NAME' refers to topon/chemistry/charmm/data/NAME."),
    )
    bridge_residue: Optional[str] = Field(
        default=None,
        description="RTF residue for auto_bridge atoms (none: a bridge atom is an error).",
    )
    pair_style: Literal["lj/charmmfsw/coul/long", "lj/charmm/coul/long"] = Field(
        default="lj/charmmfsw/coul/long",
        description=("Production pair style (10-12 A switching, PPPM). The force-switched "
                     "one pairs with dihedral_style charmmfsw, the other with charmm."),
    )


class EdgeChemistryConfig(BaseModel):
    """Chemistry configuration for an edge type."""
    monomer: str = Field(description="Monomer name from monomers library")


class ConnectionConfig(BaseModel):
    """Chain-node connection configuration."""
    auto_bridge: bool = Field(
        default=True,
        description="Automatically insert bridge atom when needed"
    )
    default_bridge_atom: str = Field(
        default="O",
        description="Default bridge atom when auto_bridge is True"
    )


class ChemistryConfig(BaseModel):
    """Chemistry configuration."""
    model_type: Literal["atomistic", "coarse_grained"] = Field(default="coarse_grained")
    force_field: Literal["dreiding", "charmm"] = Field(
        default="dreiding",
        description=("Atomistic force field. 'dreiding' (default) types atoms by element "
                     "and hybridisation; 'charmm' takes types, charges and every term from "
                     "the RTF/PRM files in `charmm`."),
    )
    charmm: Optional[CharmmConfig] = Field(
        default=None, description="CHARMM files and styles (force_field 'charmm').")
    target_density: float = Field(default=0.9, gt=0, description="Target density in g/cm³")
    
    node_type_map: dict[str, NodeMoleculeConfig] = Field(
        default={
            "end": NodeMoleculeConfig(molecule="[Si](C)(C)C", is_end_cap=True),
            "A": NodeMoleculeConfig(molecule="Si"),
        },
        description="Map node types to molecules"
    )
    
    edge_type_map: dict[str, EdgeChemistryConfig] = Field(
        default={"A": EdgeChemistryConfig(monomer="PDMS")},
        description="Map edge types to chemistry"
    )
    
    monomers: dict[str, MonomerConfig] = Field(
        default={
            "PDMS": MonomerConfig(smiles="[Si](C)(C)O", chain_head="Si", chain_tail="O"),
            "FPDMS": MonomerConfig(smiles="[Si](C)(CCC(F)(F)F)O", chain_head="Si", chain_tail="O"),
            "Phenyl": MonomerConfig(smiles="[Si](C)(c1ccccc1)O", chain_head="Si", chain_tail="O"),
        },
        description="Monomer library"
    )

    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)

    @model_validator(mode="after")
    def _charmm_is_atomistic(self):
        """CHARMM types atoms, so it has no meaning for a bead model."""
        if self.force_field == "charmm" and self.model_type != "atomistic":
            raise ValueError("chemistry.force_field 'charmm' needs model_type 'atomistic' "
                             "(the coarse-grained route is Kremer-Grest)")
        return self


# =============================================================================
# CONFORMATION CONFIG
# =============================================================================

class EntanglementControllerConfig(BaseModel):
    """How hard the closed loop tries to hit ``target_Z``.

    ``tolerance`` is relative: 0.15 accepts a measured Z within 15 % of the
    target, which is roughly the spread Z1+ shows between seeds on a 95 000-bead
    build and is therefore the tightest band worth asking for.
    """

    max_rounds: int = Field(
        default=4, ge=1, le=12,
        description="Build-relax-measure rounds before the controller gives up",
    )
    tolerance: float = Field(
        default=0.15, gt=0, le=1.0,
        description="Relative |Z - target| / target accepted as converged",
    )

    model_config = {"extra": "forbid"}


class EntanglementTargetConfig(BaseModel):
    """What the entanglement state of the built network should be.

    Every number here is a *final-state* number: measured after the relaxation
    protocol has compressed the build to the chemistry's density, equilibrated
    and quenched. That is not a convention, it is a measurement. Z1+ counts the
    kinks of the shortest paths between the junctions as they currently sit, so
    it moves by 10-20 % through equilibration and compression even with zero
    bond crossings and no free-ended chains at all (``REPORT.md`` 4.3: N100 went
    1.16 -> 1.00 -> 1.11 with every bond under 1.2 sigma throughout). The build
    state is a starting point for the controller and nothing more.
    """

    target_Z: Optional[float] = Field(
        default=None, ge=0,
        description=(
            "Mean Z1+ per strand at the final state, or null for no target. "
            "The controller turns coil_ratio (meander) or build_density (walk) "
            "until the measured value lands inside the tolerance."
        ),
    )
    close_on: Literal["final", "build"] = Field(
        default="final",
        description=(
            "Which measurement the controller closes on. 'final' is the "
            "state after compression, equilibration and quench, and is the "
            "only state a target means anything in: Z1+ counts the kinks of "
            "the shortest paths between the junctions as they currently sit, "
            "so it moves 10-20 % through the protocol with zero crossings. "
            "'build' closes on the build box instead, which is cheaper to "
            "reach and is the right choice only when the build state is "
            "itself the thing being matched."
        ),
    )
    target_hist: Optional[list[float]] = Field(
        default=None,
        description=(
            "Per-strand Z distribution to compare against, as fractions "
            "[P(Z=0), P(Z=1), ...]. Reported as a KS p-value against the "
            "measured per-strand values. Diagnostic only: nothing is tuned to "
            "it, because a distribution matched by tuning says nothing."
        ),
    )
    shells: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Neighbour-shell mix the designed pairs are drawn from, e.g. "
            "{'1': 0.5, '2': 0.5}. Shells are numbered from 1, closest first. "
            "Empty means no shell-selected pairs. Selection itself is the "
            "assignment stage's job (assignment.entanglements.shell_weights "
            "and select_by_shells); this key says what the conformation stage "
            "should be handed to route."
        ),
    )
    pairs: list[list[int]] = Field(
        default_factory=list,
        description=(
            "Named pairs to wind, as [chain_a, chain_b, windings]. Chain ids "
            "index the strand order of topon.conformation.strand_plans, which "
            "is the graph's edge order. A request whose windings do not fit in "
            "the strands' contour is refused by name with the minimum DP that "
            "would carry it."
        ),
    )
    controller: EntanglementControllerConfig = Field(
        default_factory=EntanglementControllerConfig
    )

    # Forbidding here, not only on the parent, because a round of this
    # controller is a full relaxation protocol. `"target_z": 0.18` with a
    # lower-case z is dropped by Pydantic's default and `target_Z` stays None,
    # so the loop runs one round with no target at all and reports it as
    # unconverged -- an hour of LAMMPS spent on a typo, with nothing in the
    # output pointing at it.
    model_config = {"extra": "forbid"}

    @field_validator("target_hist")
    @classmethod
    def _hist_is_a_distribution(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("target_hist is empty; use null for no target")
        if any(x < 0 for x in v):
            raise ValueError("target_hist has a negative bin")
        total = float(sum(v))
        if total <= 0:
            raise ValueError("target_hist sums to zero")
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"target_hist should be fractions summing to 1; got {total:.4f}"
            )
        return v

    @field_validator("pairs")
    @classmethod
    def _pairs_are_triples(cls, v):
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(
                    f"pairs[{i}] has {len(row)} entries; each is "
                    f"[chain_a, chain_b, windings]"
                )
            if row[0] == row[1]:
                raise ValueError(
                    f"pairs[{i}] winds chain {row[0]} with itself"
                )
            if row[2] < 1:
                raise ValueError(
                    f"pairs[{i}] asks for {row[2]} windings; ask for at least 1"
                )
        return v

    @field_validator("shells")
    @classmethod
    def _shells_are_numbered(cls, v):
        for k, w in v.items():
            try:
                shell = int(k)
            except (TypeError, ValueError):
                raise ValueError(
                    f"shell key {k!r} is not a shell number; shells are "
                    f"numbered from 1, closest first"
                )
            if shell < 1:
                raise ValueError(f"shell {shell} is below 1")
            if w < 0:
                raise ValueError(f"shell {shell} has a negative weight")
        return v


class ConformationConfig(BaseModel):
    """Stage 5: chain shape, build box and entanglement target.

    The three legacy keys (``overlap_cutoff``, ``overlap_max_iters``,
    ``noise_magnitude``) drive :class:`~topon.conformation.ConformationManager`,
    which rewrites a data file that already has coordinates. The rest drive
    :func:`topon.conformation.place`, which draws a bead-spring build from the
    graph. A config may use either set; they do not interact.

    ``placement`` is the knob that decides the entanglement state, and
    ``build_density`` is the one that decides how far it can be pushed.
    Measured on the N20 reference graph (``REPORT.md`` section 4): random-walk
    placement floors at Z = 0.23 per DP-20 strand at every build density tried
    (0.145 / 0.095 / 0.035 give 0.30 / 0.25 / 0.23), while the meander at
    rho 0.05 lands on the reference distribution exactly. Density moves Z over
    the range shape leaves it, not the other way round.
    """

    # ---- the data-file route (ConformationManager) ----
    overlap_cutoff: float = Field(
        default=0.01, ge=0,
        description="Separation below which two atoms are pushed apart, in the "
                    "units of the data file being conformed",
    )
    overlap_max_iters: int = Field(
        default=10, ge=0,
        description="Passes of the overlap resolver before it gives up",
    )
    noise_magnitude: float = Field(
        default=1e-4, ge=0,
        description="Uniform jitter applied to every atom to break degeneracy",
    )

    # ---- the bead-spring build route (place) ----
    placement: Literal["straight", "meander", "walk"] = Field(
        default="meander",
        description=(
            "Chain shape at build. 'straight' is the chord with a jitter, "
            "'meander' waves the chord out to the contour the beads need, "
            "'walk' is a random walk that closes on the far junction."
        ),
    )
    coil_ratio: Optional[float] = Field(
        default=None, gt=0,
        description=(
            "Strand contour over chord, mean over mean, at the build state. "
            "Give this or build_density, not both: the box scales as "
            "rho^(-1/3) and the contour does not move, so they are one knob."
        ),
    )
    build_density: Optional[float] = Field(
        default=None, gt=0,
        description=(
            "Bead density the paths are drawn at. The final density comes from "
            "chemistry.target_density and is reached by the relaxation "
            "protocol's compression stage, not by this."
        ),
    )
    bond: float = Field(
        default=0.97, gt=0,
        description="Design bond length of the build, in sigma",
    )
    meander_waves: float = Field(
        default=6.0, gt=0,
        description=(
            "Waves the meander spends its slack in, before the self-fold gate "
            "halves it. Six at DP 20 is about three beads per wave and folds; "
            "the gate takes it down until every bond clears min_bond and no "
            "bead sits within min_sep of a non-adjacent bead of its own chain. "
            "Fewer waves means fewer kinks per strand, which is the lever "
            "REPORT.md 4.5 identified behind the DP-20 excess Z."
        ),
    )
    min_bond: float = Field(
        default=0.85, gt=0,
        description="Shortest bond a placed strand may carry, in sigma",
    )
    min_self_separation: Optional[float] = Field(
        default=None, ge=0,
        description=(
            "Closest a bead may come to a non-adjacent bead of its own chain, "
            "in sigma. null uses the floor that belongs to the route: 1.0 for "
            "a drawn path, where a sub-sigma contact is a fold the drawing "
            "made, and 0.05 for a walk, where coming back near itself is what "
            "makes it a melt chain and only a hard overlap is a fault."
        ),
    )
    path_jitter: float = Field(
        default=0.02, ge=0,
        description="Gaussian jitter on the interior beads of a straight path, "
                    "in sigma, to break lattice degeneracy",
    )
    junction_shell_spacing: Optional[float] = Field(
        default=None, gt=0,
        description=(
            "Seat the first bead of every chain leaving a junction on a "
            "spread shell at least this far apart, in sigma. All-or-nothing "
            "per junction, since a half-seated junction is more crowded than "
            "an unseated one, and most junctions decline: read "
            "guard_report()['junction_shells'] for what was actually seated. "
            "null leaves every chain where its chord put it."
        ),
    )
    junction_jitter: float = Field(
        default=0.0, ge=0.0, le=0.5,
        description=(
            "Move every junction by a Gaussian offset per axis before the "
            "strands are drawn, as a fraction of the site spacing "
            "((box volume / sites)^(1/3)). On a lattice with several "
            "neighbour shells many chords cross exactly (every pair of sites "
            "symmetric about one point has its midpoint there), and three "
            "strands drawn through one point jam in the push-off; 0.10-0.15 "
            "takes the close chord triples within 0.5 sigma on the N20 graph "
            "from 128 to 2-6. The graph does not change; chords, reach and "
            "coil ratio do (guard_report()['junction_jitter']). An offset that "
            "would take a strand past 0.97 of its contour (or past its "
            "lattice chord, if longer), flip a chord's periodic image or "
            "bring two junctions within 1 sigma is halved. 0 leaves every "
            "junction on its site."
        ),
    )
    settle_clearance: Optional[float] = Field(
        default=None, gt=0,
        description=(
            "After drawing, part the bonds of different strands to this "
            "bond-to-bond distance, in sigma (about 1), holding the junctions, "
            "keeping every bond in the placement band and each strand clear "
            "of itself, and never moving one bond through another (two bonds "
            "drawn touching have no side, and are parted first) "
            "(guard_report()['settle']). Pairs within two bonds of a junction "
            "both strands share are exempt. Use it with junction_jitter on a "
            "lattice whose chords cross exactly: on the N20 build without jitter, "
            "or at 0.02 (up to five strands through one point), it does not "
            "converge, and a settle that would leave a strand "
            "failing the gate puts the whole build back as drawn. null draws "
            "the build as it has always been drawn."
        ),
    )
    entanglement: EntanglementTargetConfig = Field(
        default_factory=EntanglementTargetConfig
    )

    # ---- the atomistic route (Pipeline, chemistry.model_type "atomistic") ----
    atomistic_placement: Optional[Literal["straight", "meander", "walk"]] = Field(
        default="meander",
        description=(
            "Draw each strand's backbone as a chain at the force field's bond "
            "lengths, with 'straight', 'meander' or 'walk' as on the bead-spring "
            "route (meander_waves, min_bond, min_self_separation and "
            "path_jitter keep their meaning, read in units of the backbone bond "
            "over 0.97), settle it (atomistic_clearance), and place every other "
            "atom at its bond length off the backbone. A strand whose chord "
            "reaches 0.97 of its extended length (0.79 of the contour for PDMS) "
            "is drawn straight. The default; a network with POSS nodes falls "
            "back to the historic placement unless this is set. null keeps the "
            "historic placement: every heavy atom evenly on the chord, relaxed "
            "into a molecule by the soft first stage (and the relaxation "
            "defaults to the historic soft_push deck)."
        ),
    )
    atomistic_clearance: float = Field(
        default=1.5, ge=0.0,
        description=(
            "With atomistic_placement: the closest two backbone bonds may be "
            "drawn, in A (bonds that share an atom, or sit within two bonds of "
            "each other along one strand, excepted). Pairs drawn closer are "
            "opened along the line between them before anything else is "
            "placed, so the first stage never has to decide which side of each "
            "other two bonds are on. 0 turns it off."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _one_build_knob(self):
        if self.coil_ratio is not None and self.build_density is not None:
            raise ValueError(
                "set coil_ratio or build_density, not both: they are the same "
                "knob expressed two ways (coil ratio scales as rho^(1/3)), and "
                "two values for one knob cannot both be honoured"
            )
        return self

# =============================================================================
# OUTPUT CONFIG
# =============================================================================

class OutputConfig(BaseModel):
    """Output configuration."""
    lammps_data: bool = Field(default=True, description="Generate LAMMPS data file")
    lammps_convention: Literal["topon", "endlinked"] = Field(
        default="topon",
        description=(
            "Coarse-grained data-file convention. 'topon' is the historic "
            "one: one molecule for the network, atom types from bead_type. "
            "'endlinked' matches `fix bond/create` datasets - type 1 chain "
            "end, 2 interior, 3 junction, one molecule per chain and per "
            "junction, primary loops as rings - which is what refnet.parse "
            "and the Z1+ exporter read. Atomistic output ignores it."
        ),
    )
    lammps_inputs: bool = Field(default=True, description="Generate LAMMPS input scripts")
    visualization: bool = Field(default=True, description="Generate visualization HTML")
    analysis_report: bool = Field(default=True, description="Generate analysis report")
    save_attributed_graph: bool = Field(default=True, description="Save attributed graph as gpickle")
    export_graphml: bool = Field(
        default=False,
        description="Export the dual-graph (chains + entanglement edges) as a GraphML file",
    )
    export_npz: bool = Field(
        default=False,
        description="Export the graph as a compressed NPZ dataset for downstream GNN pipelines",
    )


# =============================================================================
# ANALYSIS CONFIG
# =============================================================================

class Z1PlusConfig(BaseModel):
    """Where Z1+ is installed and how to reach it.

    Z1+ (Kroger's primitive-path analysis) is not part of topon and must not
    be: its licence does not allow redistribution. Install it yourself and
    point ``executable`` at the binary. It ships for Linux, so on Windows it
    runs inside WSL, where ``executable`` is a path in the Linux file system.
    """

    model_config = ConfigDict(extra="forbid")

    executable: str = Field(
        default="~/z1/Z1+",
        description=(
            "Path to the Z1+ binary, in WSL when it runs there. A leading ~/ "
            "is the home directory of the account it runs under."
        ),
    )
    wsl: Literal["auto", "always", "never"] = Field(
        default="auto",
        description=(
            "Run Z1+ inside WSL. 'auto' does on Windows and does not "
            "elsewhere."
        ),
    )
    wsl_distro: Optional[str] = Field(
        default=None,
        description="WSL distribution to run in; null is the default one",
    )
    timeout: float = Field(
        default=7200.0, gt=0,
        description="Seconds one Z1+ run may take before it is abandoned",
    )


class AnalysisConfig(BaseModel):
    """Settings for `topon analyze` and `topon inspect`."""

    model_config = ConfigDict(extra="forbid")

    z1plus: Z1PlusConfig = Field(default_factory=Z1PlusConfig)


# =============================================================================
# MAIN CONFIG
# =============================================================================

class ToponConfig(BaseModel):
    """Complete Topon configuration."""
    study: StudyConfig = Field(default_factory=StudyConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    assignment: AssignmentConfig = Field(default_factory=AssignmentConfig)
    chemistry: ChemistryConfig = Field(default_factory=ChemistryConfig)
    conformation: ConformationConfig = Field(default_factory=ConformationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _crosslink_melt_owns_its_chains(self):
        """A crosslinked melt decides its chains and loops in stage 1.

        The chain cover (``assignment.chains``) would cut the strands into
        chains a second time, and
        the defects stage would add loops and sol the melt already has (its
        loops are intra-chain crosslinks, its sol the chains no crosslink
        reached), so both are refused rather than left to overwrite it.
        """
        if self.topology.source != "crosslink":
            return self
        if self.topology.generator.architecture == "random_crosslinked":
            raise ValueError(
                "topology.source 'crosslink' decides the chains in stage 1; "
                "topology.generator.architecture 'random_crosslinked' (the "
                "chain cover of a sculpted graph) does not apply to it")
        d = self.assignment.defects
        on = [k for k in ("primary_loops", "secondary_loops", "triangles",
                          "four_cycles", "sol_chains") if getattr(d, k).enabled]
        if on:
            raise ValueError(
                f"topology.source 'crosslink' makes its own loops and sol "
                f"(intra-chain crosslinks, chains no crosslink reached); "
                f"assignment.defects {on} would add more on top")
        return self

    @model_validator(mode="after")
    def _chains_have_a_length(self):
        """A random-crosslinked network needs the length of its chains."""
        arch = self.topology.generator.architecture
        if arch == "random_crosslinked" and self.assignment.chains.dp is None:
            raise ValueError(
                "topology.generator.architecture 'random_crosslinked' needs "
                "assignment.chains.dp, the beads per chain the strands are "
                "cut from")
        return self
