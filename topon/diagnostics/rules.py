"""Diagnostics rules for `topon doctor`.

Each rule receives the validated Pydantic `ToponConfig` plus the raw dict
(so we can also inspect schema-gap fields like `conformation`, `simulation`,
`execution`). Rules emit zero or more `Issue` records.

Adding a new rule:
    1. Write a function `check_<name>(cfg, raw) -> list[Issue]`
    2. Append it to RULE_REGISTRY at the bottom of this file
    3. The rule should be self-contained — no chemistry build, no LAMMPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass
class Issue:
    rule: str
    level: str          # "ok" | "warn" | "error"
    message: str
    fix: str | None = None


def _node_type_mapping(cfg) -> dict:
    """Pull the degree->node_type mapping from the active assignment method."""
    assign = cfg.assignment
    if assign.node_types.method == "degree":
        return dict(assign.node_types.degree.mapping or {})
    return {}


# ---------- rules ---------------------------------------------------------

def check_poss_at_internal_junction(cfg, raw) -> List[Issue]:
    """POSS at degree-≥2 junctions crashes LAMMPS stage 1."""
    mapping = _node_type_mapping(cfg)
    if not mapping:
        return []
    poss_keys = {
        tname for tname, mol_cfg in (cfg.chemistry.node_type_map or {}).items()
        if mol_cfg.molecule and mol_cfg.molecule.upper().startswith("POSS")
    }
    if not poss_keys:
        return []
    out: List[Issue] = []
    for degree_str, tname in mapping.items():
        if tname in poss_keys:
            try:
                deg = int(degree_str)
            except (TypeError, ValueError):
                continue
            if deg >= 2:
                out.append(Issue(
                    rule="poss_at_internal_junction",
                    level="warn",
                    message=(
                        f"POSS molecule '{tname}' is mapped to degree-{deg} "
                        f"nodes. POSS at internal junctions (degree >= 2) "
                        f"hit a known bug: bond extent > half periodic "
                        f"box at LAMMPS stage 1."
                    ),
                    fix=(
                        f"Map POSS to degree-1 chain caps only "
                        f'(e.g. `"mapping": {{"1": "{tname}"}}`) and use a '
                        f"plain Si node-type for higher-degree junctions."
                    ),
                ))
    return out


def check_unknown_node_type(cfg, raw) -> List[Issue]:
    """A `node_type` in assignment.degree.mapping must appear in
    `chemistry.node_type_map`. On the atomistic route the chemistry stage
    refuses the build (since 0.4.0; before, it silently built a single Si
    atom); a coarse-grained junction is one bead whatever its molecule, so
    there it is a warning.
    """
    mapping = _node_type_mapping(cfg)
    if not mapping:
        return []
    chem_types = set((cfg.chemistry.node_type_map or {}).keys())
    atomistic = cfg.chemistry.model_type == "atomistic"
    out: List[Issue] = []
    for degree_str, tname in mapping.items():
        if tname not in chem_types:
            out.append(Issue(
                rule="unknown_node_type",
                level="error" if atomistic else "warn",
                message=(
                    f"Assignment maps degree-{degree_str} to '{tname}', but "
                    f"'{tname}' isn't a key in chemistry.node_type_map "
                    f"(present: {sorted(chem_types) or '[]'}). "
                    + ("The atomistic chemistry stage will refuse the build."
                       if atomistic else
                       "A coarse-grained junction is one bead whatever its "
                       "molecule, so the build goes ahead.")
                ),
                fix=(
                    f'Add a "{tname}" entry under chemistry.node_type_map '
                    f'with a `molecule` SMILES or element symbol.'
                ),
            ))
    return out


def check_atomistic_graft_non_pdms(cfg, raw) -> List[Issue]:
    """Grafts asked for on a monomer that cannot carry one.

    An atomistic graft replaces a methyl on a repeat's head atom (the Si of
    a siloxane) with the O its PDMS side chain hangs from. A repeat whose
    monomer has no such methyl is built ungrafted, with a RuntimeWarning.
    Before 0.4.5 every monomer but PDMS itself was; the rule keeps its name.
    The monomers read are the edge type's own and, with copolymers on, those
    of its composition. The side chain itself is always PDMS on this route,
    so a ``side_chain_monomer`` that is not PDMS is warned about as well.
    """
    if cfg.chemistry.model_type != "atomistic":
        return []
    grafts_cfg = cfg.assignment.grafts.per_edge_type or {}
    if not grafts_cfg:
        return []
    from topon.chemistry.builder import _PDMS_SMILES, takes_a_graft

    def no_head_methyl(name) -> bool:
        try:
            return not takes_a_graft(monomers[name].smiles)
        except ValueError:          # a SMILES that does not parse: not this rule's call
            return False

    monomers = cfg.chemistry.monomers or {}
    edge_types = cfg.chemistry.edge_type_map or {}
    copolymer = cfg.assignment.copolymer
    out: List[Issue] = []
    for etype, gcfg in grafts_cfg.items():
        if gcfg.graft_density <= 0:
            continue
        side = monomers.get(gcfg.side_chain_monomer)
        if side is not None and side.smiles != _PDMS_SMILES:
            out.append(Issue(
                rule="atomistic_graft_non_pdms",
                level="warn",
                message=(
                    f"Edge type '{etype}' asks for side chains of "
                    f"'{gcfg.side_chain_monomer}' ({side.smiles}), but atomistic side "
                    f"chains are built of PDMS {_PDMS_SMILES} whatever "
                    f"side_chain_monomer names."
                ),
                fix="Use side_chain_monomer 'PDMS' to say what is built, or the CG model_type.",
            ))
        names = []
        if copolymer.enabled and etype in (copolymer.per_edge_type or {}):
            names = [c.monomer for c in copolymer.per_edge_type[etype].composition]
        if not names and etype in edge_types:
            names = [edge_types[etype].monomer]
        bare = [n for n in dict.fromkeys(names) if n in monomers and no_head_methyl(n)]
        if not bare:
            continue
        out.append(Issue(
            rule="atomistic_graft_non_pdms",
            level="warn",
            message=(
                f"Edge type '{etype}' has graft_density={gcfg.graft_density}, "
                f"but its monomer(s) "
                + ", ".join(f"'{n}' ({monomers[n].smiles})" for n in bare)
                + " have no methyl on the head atom for an atomistic graft to "
                f"replace. Their repeats are built ungrafted, with a "
                f"RuntimeWarning at build time."
            ),
            fix="Use a monomer with a methyl on its head atom (PDMS, a "
                "methylphenylsiloxane), the CG model_type, or set "
                "graft_density=0 for this edge type.",
        ))
    return out


def check_schema_gap_extras(cfg, raw) -> List[Issue]:
    """Top-level `simulation`/`execution` sections aren't
    Pydantic-validated. CLI path handles them via `load_config_full`; direct
    `Pipeline(ToponConfig(...))` construction without `raw_config=...` ignores
    them silently.

    `conformation` left this list in 0.2.0: it is a schema section now
    (:class:`~topon.config.schema.ConformationConfig`) and is validated like
    any other, though `load_config_full` still hands a copy back in the raw
    dict for the callers that have always read it from there.
    """
    extras = [k for k in ("simulation", "execution") if k in (raw or {})]
    if not extras:
        return []
    return [Issue(
        rule="schema_gap_extras",
        level="ok",
        message=(
            f"Config has unvalidated top-level section(s): {extras}. The "
            f"CLI handles these via `load_config_full`; direct API users "
            f"must pass them as `raw_config={{...}}` to `Pipeline(...)`."
        ),
        fix="If using the API directly, do "
            "`config, raw = load_config_full(path); Pipeline(config, raw_config=raw)`.",
    )]


def check_lattice_size_format(cfg, raw) -> List[Issue]:
    """Catch the common 'lattice_size: 5' (int) vs '5x5x5' (string) confusion."""
    if cfg.topology.source != "generate":
        return []
    gen = cfg.topology.generator
    if gen is None:
        return []
    ls = gen.lattice_size
    if not isinstance(ls, str) or "x" not in ls.lower():
        return [Issue(
            rule="lattice_size_format",
            level="error",
            message=(
                f"topology.generator.lattice_size = {ls!r}; expected a "
                f"string like '5x5x5' or '6x6x6'."
            ),
            fix='Set `"lattice_size": "5x5x5"` (string with two `x` separators).',
        )]
    return []


def check_dp_below_kuhn(cfg, raw) -> List[Issue]:
    """DP < 5 produces chains shorter than a Kuhn length — geometrically valid
    but the entanglement/conformation stages have edge cases."""
    dp_cfg = cfg.assignment.dp_distribution
    if dp_cfg is None or dp_cfg.default is None:
        return []
    mean = dp_cfg.default.mean
    if mean is None or mean >= 5:
        return []
    return [Issue(
        rule="dp_below_kuhn",
        level="warn",
        message=(
            f"DP mean = {mean} is shorter than a typical Kuhn length (~5). "
            f"Conformation overlap-resolution and entanglement detection "
            f"may produce poor geometries at this length."
        ),
        fix="Use DP >= 5 unless intentionally testing the short-chain limit.",
    )]


def check_defects_without_endcap_safety(cfg, raw) -> List[Issue]:
    """Reminder: parallel-edge defects were over-valencing end-cap
    nodes pre-2026-05-10. The fix is in `assignment/defects.py` and is always
    applied; this rule just informs the user that defect+endcap is now safe.
    """
    defects = cfg.assignment.defects
    if not (defects.secondary_loops.enabled or defects.primary_loops.enabled):
        return []
    return [Issue(
        rule="defects_endcap_safe",
        level="ok",
        message=(
            "Loop defects are enabled. Eligible pairs and loop junctions "
            "skip degree-1 chain caps (max_degree = max_functionality + "
            "exclude_node_types=('end',)), so the chemistry build stays "
            "chemically valid."
        ),
    )]


def _lattice_dims(gen):
    """``(Nx, Ny, Nz)`` from ``lattice_size``, or None when it is malformed."""
    ls = getattr(gen, "lattice_size", None)
    if not isinstance(ls, str):
        return None
    parts = ls.lower().split("x")
    if len(parts) != 3:
        return None
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError:
        return None
    return dims if all(d > 0 for d in dims) else None


def check_neighbour_cutoff_vs_box(cfg, raw) -> List[Issue]:
    """A candidate-edge range beyond a third of the periodic box.

    Past ``box/3`` three candidate edges can close a cycle around the
    box, so the candidate set carries triangles that come from the box
    rather than from the lattice; past ``box/2`` a pair can be within
    range through two images at once, which a simple graph cannot even
    represent. Open axes have no images, so only periodic ones are
    checked.
    """
    if cfg.topology.source != "generate":
        return []
    gen = cfg.topology.generator
    if gen is None:
        return []
    dims = _lattice_dims(gen)
    if dims is None:
        return []          # lattice_size_format reports the malformed size
    cutoff = float(getattr(gen, "neighbour_cutoff", 1.0) or 1.0)
    per = str(getattr(gen, "periodicity", "111"))
    periodic = [c == "1" for c in per] if len(per) == 3 else [True] * 3
    offending = [
        (axis, n) for axis, n, p in zip("xyz", dims, periodic)
        if p and cutoff > n / 3.0 + 1e-9
    ]
    if not offending:
        return []
    smallest = min(n for _, n in offending)
    axes = "".join(a for a, _ in offending)
    tail = (
        " Above half the box a pair can be within range through two "
        "periodic images at once, which the candidate graph cannot even "
        "represent."
        if cutoff > smallest / 2.0 + 1e-9 else ""
    )
    return [Issue(
        rule="neighbour_cutoff_vs_box",
        level="warn",
        message=(
            f"topology.generator.neighbour_cutoff = {cutoff:g} cell units is "
            f"more than a third of the periodic box along {axes} "
            f"(lattice_size {gen.lattice_size}). Three candidate edges can "
            f"then close a cycle around the box, so the candidate set "
            f"carries triangles that come from the box rather than from "
            f"the lattice.{tail}"
        ),
        fix=(
            f"Use at least {math.ceil(3.0 * cutoff)} cells along every "
            f"periodic axis, or a neighbour_cutoff of at most "
            f"{smallest / 3.0:.2f} on this box."
        ),
    )]


def check_site_count(cfg, raw) -> List[Issue]:
    """The sites a ``degree_distribution`` asks for against the lattice's.

    Absolute counts belong to one site count. For SC, BCC, FCC and
    Diamond that is fixed by the size, and for a MIX by the seed that
    draws it (``count_sites``, the same draw the generator makes). A
    request that cannot fit is an error here rather than a run of failed
    trials. A MIX this config does not fix (no ``seed``, or the C binary,
    which draws its own sites) gets a warning when the expected count
    leaves the request at risk.
    """
    if cfg.topology.source != "generate":
        return []
    gen = cfg.topology.generator
    if gen is None or _lattice_dims(gen) is None:
        return []

    from topon.topology.degree_matching import (
        is_fully_specified,
        parse_degree_distribution,
        resolve_search,
    )
    from topon.topology.generator_python import (
        check_site_count as fits,
        count_sites,
        expected_mix_sites,
    )

    targets, edge_count = parse_degree_distribution(gen.degree_distribution)
    if not any(n >= 0 for n in targets.values()):
        return []
    try:
        search = resolve_search(gen.search, targets, gen.max_functionality,
                                edge_count)
    except ValueError:
        return []          # the generator refuses it, with its own message
    mix = gen.lattice_type == "MIX"
    label = f"{gen.lattice_size} {gen.lattice_type}"
    fix = ("Work the counts out for the lattice that is built: "
           "topon.topology.generator_python.count_sites(config, seed) gives "
           "its sites, and topon.topology.degree_matching."
           "rescale_degree_counts carries a P(f) to them.")

    mean, sd = expected_mix_sites(gen) if mix else (None, 0.0)
    known = not mix or sd == 0.0 or (gen.seed is not None and not gen.exe_path)
    if known:
        n_sites = count_sites(gen, seed=gen.seed if gen.seed is not None else 0)
        try:
            fits(targets, gen.max_functionality, search, n_sites, label, mix=mix)
        except ValueError as exc:
            return [Issue(rule="site_count", level="error",
                          message=str(exc), fix=fix)]
        return []

    full = search == "strict" and is_fully_specified(targets, gen.max_functionality)
    asked = sum(n for d, n in targets.items()
                if n >= 0 and (d >= 1 or search != "exact"))
    if not full and asked <= mean - 3.0 * sd:
        return []
    why = ("the C binary (exe_path) draws its own sites" if gen.exe_path
           else "no topology.generator.seed fixes the draw")
    if full:
        risk = (f"degree_distribution names every degree, so its {asked} "
                f"sites must match the draw exactly and most draws will not; "
                f"those runs end in failed trials.")
        extra = " Or leave degree 0 out, so the vacancies take up the difference."
    else:
        risk = (f"degree_distribution needs {asked} sites, within three "
                f"standard deviations of that, so some draws will not fit.")
        extra = ""
    return [Issue(
        rule="site_count",
        level="warn",
        message=(f"A {label} lattice draws its sites, about {mean:.0f} +/- "
                 f"{sd:.0f} here, and {why}. {risk}"),
        fix=("Set topology.generator.seed (the Python generator) and work the "
             "counts out for that draw: topon.topology.generator_python."
             "count_sites(config, seed) gives its sites, and topon.topology."
             "degree_matching.rescale_degree_counts carries a P(f) to them."
             + extra),
    )]


def check_degree_sum_parity(cfg, raw) -> List[Issue]:
    """An odd degree sum, which no graph has (every edge adds 2).

    A request that names every degree from 0 to ``max_functionality``
    fixes the sum, and both generators and both searches refuse an odd
    one before the first trial, so doctor says so first. A partial request
    is fine in Python (the degrees it leaves out take up the parity), but
    the C binary's strict search refuses any odd sum of the degrees it
    names, so that combination is a warning when ``exe_path`` is set.
    """
    if cfg.topology.source != "generate":
        return []
    gen = cfg.topology.generator
    if gen is None:
        return []

    from topon.topology.degree_matching import (
        is_fully_specified,
        parse_degree_distribution,
        resolve_search,
    )

    targets, edge_count = parse_degree_distribution(gen.degree_distribution)
    max_f = int(gen.max_functionality)
    named_sum = sum(d * n for d, n in targets.items() if 0 < d <= max_f and n > 0)
    if named_sum % 2 == 0:
        return []
    fix = ("Move one site between two odd degrees (e.g. one fewer at degree 1 "
           "and one more at degree 2), or add one site of odd degree; "
           "topon.topology.degree_matching.rescale_degree_counts makes the sum "
           "even when it rescales a P(f).")
    if is_fully_specified(targets, max_f):
        return [Issue(
            rule="degree_sum_parity",
            level="error",
            message=(f"degree_distribution names every degree from 0 to "
                     f"max_functionality ({max_f}) and its degree sum is "
                     f"{named_sum}, which is odd; every edge adds 2 to the sum, "
                     f"so no graph has these counts."),
            fix=fix,
        )]
    try:
        search = resolve_search(gen.search, targets, max_f, edge_count)
    except ValueError:
        return []
    if gen.exe_path and search == "strict" and edge_count < 0:
        return [Issue(
            rule="degree_sum_parity",
            level="warn",
            message=(f"The degrees degree_distribution names sum to {named_sum}, "
                     f"which is odd. The Python generator runs this (the degrees "
                     f"it leaves out take up the parity), but the C binary "
                     f"(exe_path) refuses it before any trial."),
            fix=fix + " Or leave exe_path unset.",
        )]
    return []


def check_diamond_dangling_ends(cfg, raw) -> List[Issue]:
    """Degree-1 sites on the canonical Diamond scaffold, with no room for them.

    Diamond at the default cutoff has four candidate partners per site, so
    a site asked for degree 4 has to bond to all of them. Each dangling
    end leaves three neighbours a bond short, which only vacancies and
    sites below degree 4 can take. With few sites at 4 there is room (a
    216-site cell with 10 dangling ends and 56 four-fold sites builds);
    once most active sites are at 4 there is none, which is where the
    end-linked reference P(f) sits (73 % four-fold, 8 % dangling), and the
    validation found no network for it at any size. The test is the exact
    search's own no-slack rule, the one that stops it after one attempt.
    """
    if cfg.topology.source != "generate":
        return []
    gen = cfg.topology.generator
    if gen is None or gen.lattice_type != "Diamond":
        return []
    if float(gen.neighbour_cutoff) != 1.0 or gen.neighbour_shells is not None:
        return []          # a wider scaffold has spare candidates

    from topon.topology.degree_matching import _PINNED_SHARE, parse_degree_distribution

    targets, _ = parse_degree_distribution(gen.degree_distribution)
    dangling = targets.get(1, 0)
    if dangling <= 0:
        return []
    # The scaffold's four, as in the search. Capped at a lower
    # max_functionality it would warn where every site keeps a spare
    # candidate and the search retries.
    ceiling = 4
    active = sum(n for d, n in targets.items() if d > 0 and n > 0)
    pinned = sum(n for d, n in targets.items()
                 if ceiling <= d <= int(gen.max_functionality) and n > 0)
    if not active or pinned < _PINNED_SHARE * active:
        return []
    return [Issue(
        rule="diamond_dangling_ends",
        level="warn",
        message=(f"Diamond at the default cutoff gives every site four candidate "
                 f"partners, and degree_distribution asks for {pinned} of its "
                 f"{active} active sites at degree {ceiling} alongside "
                 f"{dangling} degree-1 sites. Each dangling end leaves three "
                 f"neighbours a bond short, and with that many sites at the "
                 f"ceiling nothing can take it up; the exact search stops "
                 f"after one attempt on such a request."),
        fix=("Give the scaffold spare candidates (neighbour_cutoff 0.71 on "
             "Diamond admits the second shell, z = 16, or use SC/BCC/FCC at a "
             "wider cutoff), or drop the degree-1 sites. Diamond is not a "
             "scaffold for end-linked networks with dangling ends."),
    )]


def check_deprecated_mix_cutoff(cfg, raw) -> List[Issue]:
    """``mix_cutoff`` still loads, as ``neighbour_cutoff``, but is deprecated."""
    gen_raw = ((raw or {}).get("topology") or {}).get("generator") or {}
    if not isinstance(gen_raw, dict) or "mix_cutoff" not in gen_raw:
        return []
    return [Issue(
        rule="deprecated_mix_cutoff",
        level="warn",
        message=(
            "topology.generator.mix_cutoff is deprecated. It is read as "
            "neighbour_cutoff, which sets the candidate-edge range for "
            "every lattice type, not only MIX."
        ),
        fix="Rename the key to neighbour_cutoff.",
    )]



def check_unknown_config_keys(cfg, raw: dict) -> list:
    """Keys the schema silently ignores, anywhere in the nested config.

    ``ToponConfig`` is ``extra="forbid"``, but that binds at the top
    level only: every nested section takes Pydantic's default and drops
    unknown keys without a word. That is how ``topology.generator.seed``
    validated cleanly while doing nothing before it was a real field
    (0.4.0), which a user reasonably read as having pinned the graph.

    ``GeneratorConfig`` now refuses them outright, so this rule never
    sees that case; a config carrying it fails to load before doctor
    runs. This covers everything else, where forbidding is not safe:
    configs may carry other tools' parameters under
    ``topology``, and several sections outside the schema
    (``conformation``, ``simulation``, ``execution``) are read from the
    raw dict by design, which ``schema_gap_extras`` reports separately.

    Walks the model tree rather than a hard-coded list, so a section
    added later is covered without touching this rule.
    """
    from pydantic import BaseModel

    # No allowlist of "sections read from raw on purpose" here, though an
    # earlier version carried one. It was unreachable: the walk skips any
    # model that forbids extras, and ToponConfig does, so the root level
    # never reports and the list never ran. Root-level pass-through
    # sections are schema_gap_extras' job, and which sections those are
    # changes as the schema grows, so naming them twice would leave one
    # copy to rot. This rule is only about the nested models that still
    # ignore.
    issues = []

    def walk(model_cls, data, path):
        if not isinstance(data, dict) or not isinstance(model_cls, type):
            return
        if not issubclass(model_cls, BaseModel):
            return
        fields = model_cls.model_fields
        forbids = model_cls.model_config.get("extra") == "forbid"
        unknown = [k for k in data if k not in fields]
        if unknown and not forbids:
            where = ".".join(path) if path else "<root>"
            issues.append(Issue(
                rule="unknown_config_keys",
                level="warn",
                message=(
                    f"{where} carries key(s) the schema does not define: "
                    f"{', '.join(sorted(unknown))}. They are ignored "
                    f"silently, so anything you meant them to do is not "
                    f"happening."
                ),
                fix=(
                    "Remove them, or check the spelling against "
                    "docs/USAGE.md Appendix A. A generated graph is pinned "
                    "by topology.generator.seed."
                ),
            ))
        for name, field in fields.items():
            if name not in data:
                continue
            annotation = field.annotation
            # Unwrap Optional[X] / Union[X, None] to reach the model.
            for candidate in getattr(annotation, "__args__", (annotation,)):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    walk(candidate, data[name], path + [name])
                    break

    walk(type(cfg), raw, [])
    return issues


# ---------- registry + runner ---------------------------------------------

RuleFn = Callable[[object, dict], Iterable[Issue]]

def _atomistic(cfg) -> bool:
    chem = getattr(cfg, "chemistry", None)
    return getattr(chem, "model_type", None) == "atomistic"


def check_atomistic_target_placement(cfg, raw) -> List[Issue]:
    """An atomistic `target_Z` with a placement named that it cannot turn.

    On the atomistic route a target is met by the coil radius, searched on
    the build (the coarse-grained calibration table and its coil ratio do
    not apply there), so naming the meander, the walk, straight or the
    historic placement beside it leaves nothing to turn.
    """
    conf = getattr(cfg, "conformation", None)
    if conf is None or not _atomistic(cfg) or conf.entanglement.target_Z is None:
        return []
    if "atomistic_placement" not in conf.model_fields_set:
        return []
    if conf.atomistic_placement == "coil":
        return []
    return [Issue(
        rule="atomistic_target_placement",
        level="error",
        message=(
            f"conformation.entanglement.target_Z is set on the atomistic route, "
            f"where it is met by the coil radius, but atomistic_placement is "
            f"{conf.atomistic_placement!r}."
        ),
        fix="Set conformation.atomistic_placement to 'coil' or leave it out.",
    )]


def check_entanglement_target_below_floor(cfg, raw) -> List[Issue]:
    """A `target_Z` under the lowest value its route has ever reached.

    The controller says the same thing when it runs, but a round of that loop
    is a full relaxation protocol -- about 35 minutes at 8 threads on a
    95 000-bead build -- and this costs nothing.
    """
    conf = getattr(cfg, "conformation", None)
    if conf is None or _atomistic(cfg):
        return []
    target = conf.entanglement.target_Z
    if target is None:
        return []
    dp = int(cfg.assignment.dp_distribution.default.mean)

    from topon.conformation.entanglement import actuator_name, floor_warning

    msg = floor_warning(dp, conf.placement, float(target))
    if not msg:
        return []
    fix = ("Pick a target the route has reached; the warning says whether "
           "the meander's floor at this DP is any lower." if conf.placement == "walk"
           else "Lower conformation.meander_waves, which is the lever "
                "REPORT.md 4.5 identified behind the DP-20 excess, and "
                "expect to calibrate it.")
    return [Issue(
        rule="entanglement_target_below_floor",
        level="warn",
        message=msg,
        fix=fix + f" The knob for this route is conformation."
                  f"{actuator_name(conf.placement)}.",
    )]


def check_conformation_build_knob(cfg, raw) -> List[Issue]:
    """An entanglement target with no build knob and no calibration to seed it.

    `place` refuses to pick a build density on its own, and the controller
    falls back to the shipped table -- which has rows only for the routes and
    DPs that have actually been measured.
    """
    conf = getattr(cfg, "conformation", None)
    if conf is None or _atomistic(cfg):
        return []
    if conf.coil_ratio is not None or conf.build_density is not None:
        return []
    if conf.entanglement.target_Z is None:
        return []
    dp = int(cfg.assignment.dp_distribution.default.mean)

    from topon.conformation.entanglement import calibration_for

    for state in ("final", "build"):
        for protocol in ("limit", "hardcore_min"):
            if calibration_for(dp, conf.placement, state, protocol):
                return []
    return [Issue(
        rule="conformation_build_knob",
        level="error",
        message=(
            f"conformation.entanglement.target_Z is set but neither "
            f"coil_ratio nor build_density is, and nothing has been measured "
            f"for DP {dp} with placement {conf.placement!r} to seed the "
            f"controller from."
        ),
        fix="Set conformation.coil_ratio (meander) or "
            "conformation.build_density (walk) to start the loop from.",
    )]


RULE_REGISTRY: list[RuleFn] = [
    check_lattice_size_format,
    check_neighbour_cutoff_vs_box,
    check_site_count,
    check_degree_sum_parity,
    check_diamond_dangling_ends,
    check_deprecated_mix_cutoff,
    check_unknown_node_type,
    check_poss_at_internal_junction,
    check_atomistic_graft_non_pdms,
    check_dp_below_kuhn,
    check_defects_without_endcap_safety,
    check_entanglement_target_below_floor,
    check_conformation_build_knob,
    check_atomistic_target_placement,
    check_schema_gap_extras,
    check_unknown_config_keys,
]


def run_all_rules(cfg, raw: dict | None = None) -> List[Issue]:
    """Run every rule against ``cfg`` (Pydantic ``ToponConfig``)."""
    raw = raw or {}
    issues: List[Issue] = []
    for rule in RULE_REGISTRY:
        try:
            issues.extend(rule(cfg, raw) or [])
        except Exception as exc:  # rules must never crash doctor
            issues.append(Issue(
                rule=rule.__name__,
                level="warn",
                message=f"Rule crashed: {type(exc).__name__}: {exc}",
            ))
    return issues
