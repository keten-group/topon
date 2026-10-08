"""Step 2 of ``topon fit``: the cell, the neighbour cutoff, and the sweep.

The cell is the smallest cube whose lattice holds the reference's active
sites (junctions with a strand to somewhere else, plus one site per
dangling-chain end). The cutoff is where the connectivity is decided: the
bond/create validation found that the candidate-edge range, not the
sublattice mix, is what makes a sculpted graph read like a reaction-made
one, three to four simple-cubic shells for DP-20 strands and six to eight
for DP 100 (``bond_create_validation/REPORT.md`` section 3). The range that
matches sits at the upper end of the strand's reach, so the rule of thumb is

    cutoff ~ p95(junction-junction separation of the bridges) / site spacing

with the site spacing the reference box over the cell's edge count. On the
references that reads 10.48 / 4.95 = 2.1 for DP 20 (14^3) and
25.43 / 7.70 = 3.3 for DP 100 (9^3), and the matching cutoffs are at or just
below those. So the candidates are the simple-cubic shell cutoffs at or below
the rule, the nearest first, and a short sweep decides between them: each
candidate is built through the pipeline's own stages 1 to 3 (so it is exactly
the graph ``topon generate`` will build for that config and seed), described,
and scored against the reference with the composite of
:func:`topon.analysis.descriptors.compare`. A nearest-neighbour control row
(cutoff 1.0) is scored alongside and never chosen: it is the scale the
chosen range is read against.
"""
from __future__ import annotations

import contextlib
import copy
import io
import math
import tempfile
import time
import warnings
from typing import Callable, Optional, Sequence

import numpy as np

from topon.topology.shells import (
    sc_coordination, sc_shell_cutoff, sc_shell_radius, sc_shells_within)

#: Shells swept when the reference has no coordinates to read a rule from.
DEFAULT_SHELLS = (2, 3, 4, 6, 8)
#: Candidates taken at or below the rule of thumb, nearest first.
N_CANDIDATES = 3
#: The control row: the canonical nearest-neighbour range.
CONTROL_CUTOFF = 1.0


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

def sites_per_cell(lattice: str, mix: Optional[dict] = None) -> tuple[float, float]:
    """Mean and per-cell variance of the site count of one cubic cell."""
    from topon.topology.generator_python import SITES_PER_CELL

    if lattice in SITES_PER_CELL:
        return float(SITES_PER_CELL[lattice]), 0.0
    if lattice != "MIX":
        raise ValueError(f"lattice {lattice!r} has no site rule")
    f_bcc = float((mix or {}).get("BCC", 0.0))
    f_fcc = float((mix or {}).get("FCC", 0.0))
    mean = 1.0 + f_bcc + 3.0 * f_fcc
    var = f_bcc * (1.0 - f_bcc) + 3.0 * f_fcc * (1.0 - f_fcc)
    return mean, var


def choose_cell(n_active: int, lattice: str = "SC",
                mix: Optional[dict] = None) -> dict:
    """The smallest cubic cell that holds ``n_active`` sites.

    For a MIX lattice, whose body and face sites are drawn, the cell must
    hold them three standard deviations below its expected count, so every
    seed a verification draws still fits; the exact count for a given seed
    is :func:`topon.topology.generator_python.count_sites`.
    """
    mean, var = sites_per_cell(lattice, mix)
    n = max(1, int(math.floor((n_active / mean) ** (1.0 / 3.0))))
    while True:
        cells = n ** 3
        if cells * mean - 3.0 * math.sqrt(cells * var) >= n_active:
            break
        n += 1
    n_sites = n ** 3 * mean
    return {"n": n, "lattice_size": f"{n}x{n}x{n}", "lattice_type": lattice,
            "n_sites": int(round(n_sites)) if var == 0 else float(n_sites),
            "n_active": int(n_active),
            "vacancy_fraction": float(1.0 - n_active / n_sites)}


# ---------------------------------------------------------------------------
# The cutoff
# ---------------------------------------------------------------------------

def rule_of_thumb(p95_separation: float, box, n: int) -> float:
    """``p95 / site spacing``, the spacing being the mean box edge over ``n``."""
    spacing = float(np.mean(np.asarray(box, float))) / float(n)
    return float(p95_separation) / spacing


def describe_cutoff(cutoff: float) -> dict:
    """A cutoff with the simple-cubic shells it admits and their coordination."""
    return {"cutoff": float(cutoff), "shells": int(sc_shells_within(cutoff)),
            "z": int(sc_coordination(cutoff))}


def shells_within(rule: float) -> int:
    """Simple-cubic shells whose radius is at or below ``rule``.

    Compared on the radius, not on the rounded cutoff that admits the
    shell: a reference built on a lattice has its p95 exactly on a shell
    (sqrt 2 = 1.4142 for two shells, where the cutoff is 1.42), and that
    shell is within its reach.
    """
    s = 0
    while sc_shell_radius(s + 1) <= rule * (1.0 + 1e-6):
        s += 1
    return s


def candidate_cutoffs(rule: Optional[float], n: int,
                      k: int = N_CANDIDATES) -> list[dict]:
    """The cutoffs to sweep.

    With a rule of thumb: the cutoffs of the ``k`` outermost simple-cubic
    shells within it (:func:`shells_within`), nearest first, and at least
    the first shell. Without one: the shells of :data:`DEFAULT_SHELLS` whose
    radius is within a third of the box (eight shells, radius 3, on a 9^3
    cell, which is the DP-100 range). Nothing is offered at or beyond half
    the box, where a pair would be in range through two images.
    """
    if rule is None:
        out = [sc_shell_cutoff(s) for s in DEFAULT_SHELLS
               if sc_shell_radius(s) <= n / 3.0 + 1e-9]
        return [describe_cutoff(c) for c in (out or [CONTROL_CUTOFF])]
    top = max(1, shells_within(rule))
    cutoffs = [sc_shell_cutoff(s) for s in range(1, top + 1)]
    cutoffs = [c for c in cutoffs if c < n / 2.0] or [CONTROL_CUTOFF]
    return [describe_cutoff(c) for c in reversed(cutoffs[-k:])]


# ---------------------------------------------------------------------------
# Building and scoring one candidate
# ---------------------------------------------------------------------------

def with_cutoff(config: dict, cutoff: float) -> dict:
    """A copy of ``config`` with the generator's range set to ``cutoff``."""
    cfg = copy.deepcopy(config)
    gen = cfg["topology"]["generator"]
    gen.pop("neighbour_shells", None)
    gen["neighbour_cutoff"] = float(cutoff)
    return cfg


def with_seed(config: dict, seed: int) -> dict:
    """A copy of ``config`` pinned to ``seed``: the graph and its defects.

    A crosslinked melt (``topology.source: "crosslink"``) is pinned through
    ``topology.crosslinking.seed``. A chain cover (``assignment.chains``)
    moves with the graph's seed and keeps its offset from it, so a config
    whose cover seed differs from its graph seed regenerates its own build
    at its own seed.
    """
    cfg = copy.deepcopy(config)
    topo = cfg.setdefault("topology", {})
    block = "crosslinking" if topo.get("source") == "crosslink" else "generator"
    old = (topo.get(block) or {}).get("seed")
    topo.setdefault(block, {})["seed"] = int(seed)
    assignment = cfg.get("assignment", {})
    defects = assignment.get("defects")
    if defects is not None:
        defects["seed"] = int(seed)
    chains = assignment.get("chains")
    if chains is not None:
        own = chains.get("seed")
        chains["seed"] = (int(seed) if own is None or old is None
                          else (int(seed) + int(own) - int(old)) % 2 ** 32)
    return cfg


def build_graph(config: dict, seed: Optional[int] = None, quiet: bool = True):
    """Pipeline stages 1 to 3 on ``config``, in a scratch directory.

    Returns ``(G, topology_manifest)``: the graph the chemistry stage would
    receive and the topology section of the run manifest (requested against
    achieved degree counts, the sculpt record, timing). Nothing is left on
    disk. ``seed`` pins the generator and the defects stage through their
    config fields, and the rest of stages 2 and 3 (the DP draw of a
    polydisperse distribution, random types) by seeding the global streams
    with it for the duration of the call; their state is put back after.
    """
    import random

    from topon.config.schema import ToponConfig
    from topon.core.manifest import read_manifest
    from topon.pipeline import Pipeline

    cfg = with_seed(config, seed) if seed is not None else copy.deepcopy(config)
    known = {k: v for k, v in cfg.items() if k in ToponConfig.model_fields}
    saved = (random.getstate(), np.random.get_state())
    with tempfile.TemporaryDirectory(prefix="topon_fit_") as tmp:
        known["study"] = {"name": "graph", "output_dir": tmp}
        model = ToponConfig.model_validate(known)
        sink = io.StringIO()
        try:
            if seed is not None:
                random.seed(int(seed))
                np.random.seed(int(seed))
            with warnings.catch_warnings():
                if quiet:
                    warnings.simplefilter("ignore")
                with (contextlib.redirect_stdout(sink) if quiet
                      else contextlib.nullcontext()):
                    pipe = Pipeline(model, raw_config={})
                    G = pipe.run_graph_stages()
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
        manifest = (read_manifest(pipe.output_dir) or {}).get("stages", {})
    return G, manifest.get("topology", {})


def total_beads(G, dp_default: float) -> Optional[int]:
    """Beads the chemistry stage will build for ``G``, loops and sol included.

    The defects stage's bead budget when it ran, otherwise the same count
    made here (a config with no defects never runs it).
    """
    budget = (G.graph.get("defects") or {}).get("bead_budget") or {}
    if budget.get("total_beads"):
        return int(budget["total_beads"])
    from topon.assignment.defects import bead_budget

    sol = G.graph.get("sol_chains") or {}
    count = int(sol.get("count", 0)) if isinstance(sol, dict) else 0
    return int(bead_budget(G, dp_default=int(round(dp_default)),
                           sol_count=count,
                           sol_dp=sol.get("dp") if count else None)["total_beads"])


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def sweep(config: dict, reference, candidates: Sequence[dict],
          seeds: Sequence[int], control: Optional[float] = CONTROL_CUTOFF,
          log: Optional[Callable[[str], None]] = None) -> dict:
    """Score every candidate cutoff on every seed and pick the closest.

    ``config`` is the fitted config with everything but the cutoff decided;
    ``reference`` the reference's :class:`~topon.analysis.descriptors.
    Descriptors`. Returns ``{"rows": [...], "summary": [...], "chosen":
    {...}}``: one row per build, one summary per cutoff (mean and spread of
    the composite over the seeds), and the chosen cutoff, the one with the
    lowest mean composite among the candidates on which every seed built.
    The composite that chooses divides each term by the larger of the
    reference value and its spread over the sweep's graphs, as
    ``cubic_sweep.py`` did; ``composite_shipped`` beside it divides by the
    reference value alone, :func:`~topon.analysis.descriptors.compare`'s
    default and the number ``--verify`` reports.
    ``chosen["within_scatter"]`` says when the runner-up is closer than the
    larger of the two seed spreads, which is when the choice is a tie that
    the seeds happened to break.
    """
    say = log or (lambda msg: None)
    todo = [dict(c, control=False) for c in candidates]
    if control is not None and all(abs(c["cutoff"] - control) > 1e-9
                                   for c in candidates):
        todo.append(dict(describe_cutoff(control), control=True))

    from topon.analysis.descriptors import compare, describe

    rows, described = [], []
    for cand in todo:
        cfg = with_cutoff(config, cand["cutoff"])
        for s in seeds:
            row = {"cutoff": cand["cutoff"], "shells": cand["shells"],
                   "z": cand["z"], "control": cand["control"], "seed": int(s)}
            t0 = time.perf_counter()
            try:
                G, man = build_graph(cfg, seed=s)
            except Exception as exc:          # a candidate that cannot build
                row["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0]
                rows.append(row)
                say(f"  cutoff {cand['cutoff']:.2f} seed {s}: failed, "
                    f"{row['error']}")
                continue
            t1 = time.perf_counter()
            d = describe(G, heavy=True, seed=0)
            t2 = time.perf_counter()
            defects = G.graph.get("defects") or {}
            dp_mean = (cfg.get("assignment", {}).get("dp_distribution", {})
                       .get("default", {}).get("mean", 25))
            row.update(
                beads=total_beads(G, dp_mean),
                pf_chemical=defects.get("chemical_degree"),
                pf_effective=defects.get("effective_degree"),
                sculpt=(man.get("sculpt") or {}).get("achieved"),
                build_seconds=round(t1 - t0, 2),
                describe_seconds=round(t2 - t1, 2))
            rows.append(row)
            described.append((row, d))
            say(f"  cutoff {cand['cutoff']:.2f} ({cand['shells']} shells, "
                f"z {cand['z']}) seed {s}: built and described in "
                f"{t2 - t0:.0f} s")

    # Two composites per row. The one that chooses is the validation
    # sweep's own (cubic_sweep.py): each size-free term divided by
    # max(|reference|, spread over every graph of the sweep), which keeps a
    # reference value near zero -- the DP-100 assortativity is -0.006 --
    # from turning seed noise into the ranking. The spread is over this
    # sweep's graphs (candidates and control), not the validation's hundred,
    # so the scale moves with the candidate set: --no-control or other
    # --sweep-cutoffs give other numbers. The shipped one, each term over
    # |reference|, is what `topon generate --verify` reports and what the
    # acceptance thresholds are written in.
    population = [d for _r, d in described]
    for row, d in described:
        swept = compare(d, reference, population=population)
        shipped = compare(d, reference)
        row.update(
            composite=swept["composite"],
            composite_shipped=shipped["composite"],
            cycle_js=swept.get("cycle_js"),
            ks_betweenness=swept["ks"].get("betweenness_core"),
            deviations=swept["deviations"],
            deviations_shipped=shipped["deviations"],
            omitted=swept["omitted"])
        say(f"  cutoff {row['cutoff']:.2f} seed {row['seed']}: composite "
            f"{row['composite']:.3f} (over the reference value "
            f"{row['composite_shipped']:.3f})")

    summary = []
    for cand in todo:
        mine = [r for r in rows if r["cutoff"] == cand["cutoff"]]
        ok = [r["composite"] for r in mine if "composite" in r]
        summary.append({
            "cutoff": cand["cutoff"], "shells": cand["shells"], "z": cand["z"],
            "control": cand["control"], "n_seeds": len(mine), "n_ok": len(ok),
            "composite": float(np.mean(ok)) if ok else None,
            "composite_sd": float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0,
            "composite_shipped": _mean(r.get("composite_shipped") for r in mine),
            "cycle_js": _mean(r.get("cycle_js") for r in mine),
            "deviations": _mean_dict([r["deviations"] for r in mine
                                      if "deviations" in r])})

    eligible = [s for s in summary if not s["control"]
                and s["n_ok"] == s["n_seeds"] and s["composite"] is not None]
    chosen = None
    if eligible:
        ranked = sorted(eligible, key=lambda s: s["composite"])
        chosen = dict(ranked[0])
        if len(ranked) > 1:
            second = ranked[1]
            gap = second["composite"] - chosen["composite"]
            spread = max(chosen["composite_sd"], second["composite_sd"])
            chosen["runner_up"] = {"cutoff": second["cutoff"],
                                   "shells": second["shells"],
                                   "composite": second["composite"]}
            chosen["within_scatter"] = bool(gap < spread)
    return {"rows": rows, "summary": summary, "chosen": chosen,
            "seeds": [int(s) for s in seeds]}


def _mean(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _mean_dict(dicts) -> dict:
    keys = sorted({k for d in dicts for k in d})
    return {k: _mean(d.get(k) for d in dicts) for k in keys}
