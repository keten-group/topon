"""Z1+ primitive-path analysis of a bead-spring network.

Z1+ (M. Kroger, Comput. Phys. Commun. 2023 and earlier) reduces every chain to
its shortest path through the others and counts the kinks, Z per chain. It is
not part of topon and must not be copied into it: its licence does not allow
redistribution. Install it and name the binary with ``analysis.z1plus`` in a
config (:class:`topon.config.schema.Z1PlusConfig`) or ``--z1-exe`` on the
command line. It ships for Linux, so on Windows it runs inside WSL. When it
cannot be reached every entry point here raises :class:`Z1PlusUnavailable`,
and :func:`z1plus_available` says so beforehand, so a caller can skip cleanly.

What is exported, and why:

* junction to junction: a bridge is written with the junction bead at both
  ends, a dangling chain with its junction at one, so the strands meet where
  the network does;
* junction beads jittered by 1e-3 sigma: Z1+ stops with "CRASHED. contact mk"
  when two chains share an exact coordinate, which every shared junction is;
* unwrapped bead to bead by the minimum image: Z1+ measures a chain's path,
  and a chain folded at the boundary is not the chain;
* every class (bridge, loop, dangling, free) in the file, since Z1+ measures
  entanglements between whatever it is given, and the per-class numbers are
  split out afterwards in export order.

With ``-SP+`` Z1+ also names the chain responsible for each kink, which gives
the chain-chain entanglement graph. On the DP-100 reference that graph
reproduces the collaborator's NPZ entanglement edges (764 against 748 pairs).

Z1+ counts the kinks of the shortest paths between the current junction
positions, so it is a state property, not a topological invariant: it moves
10-20 % through equilibration and compression with no crossing anywhere.
Compare two measurements only at the same density and temperature.

Ported from ``z1tools.py`` and ``measure_system.py`` of the bond/create
validation scripts; on the DP-20 reference it gives Z 0.167 over all chains,
0.178 per bridge and 600 partner pairs, the numbers in the report.
"""
from __future__ import annotations

import collections
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from topon.analysis.endlinked import (
    CLASSES, EndLinkedSystem, chain_statistics, read_endlinked)
from topon.config.schema import Z1PlusConfig

#: Jitter on the first and last bead of every exported chain, sigma.
JITTER = 1e-3


class Z1PlusUnavailable(RuntimeError):
    """Z1+ is not installed where the configuration says, or cannot run."""


class Z1PlusFailed(RuntimeError):
    """Z1+ ran and did not produce a result."""


@dataclass
class Z1Result:
    """What one Z1+ run returned, per chain in export order."""

    summary: dict                      # the whole-system line of Z1+summary.dat
    Z: np.ndarray                      # kinks per chain
    Lpp: np.ndarray                    # primitive-path length per chain
    Ree: np.ndarray                    # end-to-end distance per chain
    partners: Optional[list] = None    # per chain, partner chain (1-based) per kink
    classes: Optional[np.ndarray] = None
    #: Per chain, its primitive path as rows ``x y z kink partner`` (kink 1
    #: at an entanglement point, partner 0 where there is none), with -SP+.
    paths: Optional[list] = None
    log: str = ""

    @property
    def n_chains(self) -> int:
        return len(self.Z)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_z1(path, chains: Sequence[np.ndarray], box) -> Path:
    """Write chains in Z1+'s input format.

    Line 1 the chain count, line 2 the box, line 3 the bead count of each
    chain, then one ``x y z`` line per bead, chain after chain. Coordinates
    must already be unwrapped.
    """
    Path(path).write_text(_z1_text(chains, box), encoding="utf-8", newline="\n")
    return Path(path)


def _z1_text(chains, box) -> str:
    box = np.asarray(box, float).reshape(3)
    lines = [f"{len(chains)}", f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f}",
             " ".join(str(len(c)) for c in chains)]
    for c in chains:
        lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in c]
    return "\n".join(lines) + "\n"


def jitter_ends(chains, seed: int = 0, amp: float = JITTER) -> list:
    """Copies of ``chains`` with the first and last bead of each moved by a
    Gaussian of ``amp`` per axis, drawn in chain order from ``seed``."""
    rng = np.random.default_rng(seed)
    out = []
    for c in chains:
        c = np.array(c, float)
        c[0] += rng.normal(0, amp, 3)
        c[-1] += rng.normal(0, amp, 3)
        out.append(c)
    return out


def export_system(system: EndLinkedSystem, seed: int = 0,
                  amp: float = JITTER) -> tuple[list, np.ndarray, np.ndarray]:
    """Every chain of an end-linked system, ready for Z1+.

    Returns ``(chains, classes, molecules)``, all in molecule order.
    """
    chains, cls, mols = [], [], []
    for m in sorted(system.strands):
        s = system.strands[m]
        chains.append(system.strand_path(s, junctions=True))
        cls.append(s.cls)
        mols.append(m)
    return jitter_ends(chains, seed, amp), np.array(cls), np.array(mols)


def export_placement(placement, seed: int = 0,
                     amp: float = JITTER) -> tuple[list, np.ndarray]:
    """Every strand of a :class:`~topon.conformation.placement.Placement`.

    The placed paths are already unwrapped and end on their junctions (a
    sol chain, class ``"free"``, has none and is its own beads), so they go
    out as drawn, in placement order (the chain order of
    :func:`topon.writers.write_endlinked`). Returns ``(chains, classes)``.
    """
    chains = [np.asarray(s.path, float) for s in placement.strands]
    cls = np.array([s.plan.kind for s in placement.strands])
    return jitter_ends(chains, seed, amp), cls


# ---------------------------------------------------------------------------
# Running Z1+
# ---------------------------------------------------------------------------

def _config(config) -> Z1PlusConfig:
    if config is None:
        return Z1PlusConfig()
    if isinstance(config, Z1PlusConfig):
        return config
    return Z1PlusConfig(**dict(config))


def _uses_wsl(cfg: Z1PlusConfig) -> bool:
    if cfg.wsl == "always":
        return True
    if cfg.wsl == "never":
        return False
    return os.name == "nt"


def _exe_word(executable: str) -> str:
    """The binary as a bash word, with a leading ~/ left to the shell."""
    if "'" in executable:
        raise ValueError(f"Z1+ path may not contain a quote: {executable!r}")
    if executable.startswith("~/"):
        return "$HOME/" + shlex.quote(executable[2:])
    return shlex.quote(executable)


def _bash(cfg: Z1PlusConfig, script: str) -> list:
    """The argument list that runs ``script`` in bash, in WSL or natively.

    The script never contains a double quote or a backslash, so it reaches
    bash unchanged through the Windows command line wsl.exe is handed.
    """
    if _uses_wsl(cfg):
        cmd = ["wsl.exe"]
        if cfg.wsl_distro:
            cmd += ["-d", cfg.wsl_distro]
        return cmd + ["-e", "bash", "-c", script]
    return ["bash", "-c", script]


def _decode(raw: bytes) -> str:
    # wsl.exe writes its own messages in UTF-16; the program's output is UTF-8.
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


@lru_cache(maxsize=8)
def _probe(cfg_json: str) -> Optional[str]:
    cfg = Z1PlusConfig.model_validate_json(cfg_json)
    script = f"test -x {_exe_word(cfg.executable)} && echo Z1PLUS_OK"
    try:
        p = subprocess.run(_bash(cfg, script), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"cannot start {'WSL' if _uses_wsl(cfg) else 'bash'}: {exc}"
    if "Z1PLUS_OK" not in _decode(p.stdout):
        where = f"WSL{' (' + cfg.wsl_distro + ')' if cfg.wsl_distro else ''}" \
            if _uses_wsl(cfg) else "this machine"
        return (f"no executable Z1+ at {cfg.executable} in {where}. Install "
                f"Z1+ and set analysis.z1plus.executable (or --z1-exe).")
    return None


def z1plus_available(config=None) -> bool:
    """True when the configured Z1+ binary can be run. Cached per config."""
    return _probe(_config(config).model_dump_json()) is None


def why_unavailable(config=None) -> Optional[str]:
    """Why Z1+ cannot run, or None when it can."""
    return _probe(_config(config).model_dump_json())


def run_z1(chains: Sequence[np.ndarray], box, partners: bool = True,
           config=None, classes=None) -> Z1Result:
    """Run Z1+ on ``chains`` (unwrapped, one array of beads per chain).

    The input goes to Z1+ on stdin and Z1+ runs in a fresh temporary
    directory that is removed afterwards, so concurrent runs cannot overwrite
    each other's output and no path has to be translated into WSL.

    Raises:
        Z1PlusUnavailable: the binary cannot be run.
        Z1PlusFailed: Z1+ ran and wrote no summary (the tail of its log is in
            the message).
    """
    cfg = _config(config)
    why = why_unavailable(cfg)
    if why:
        raise Z1PlusUnavailable(why)
    opt = "-SP+ " if partners else ""
    script = (
        "d=$(mktemp -d) && cd $d && trap 'cd / && rm -rf $d' EXIT && "
        f"cat > cfg.Z1 && {_exe_word(cfg.executable)} {opt}cfg.Z1 "
        "> z1.log 2>&1; "
        "if [ -f Z1+summary.dat ]; then "
        "echo @@summary; cat Z1+summary.dat; "
        "echo @@Z; cat Z_values.dat; "
        "echo @@Lpp; cat Lpp_values.dat; "
        "echo @@Ree; cat Ree_values.dat; "
        + ("if [ -f Z1+SP.dat ]; then echo @@SP; cat Z1+SP.dat; fi; "
           if partners else "")
        + "else echo @@crash; tail -20 z1.log; fi")
    try:
        p = subprocess.run(_bash(cfg, script),
                           input=_z1_text(chains, box).encode("utf-8"),
                           capture_output=True, timeout=cfg.timeout)
    except subprocess.TimeoutExpired as exc:
        raise Z1PlusFailed(f"Z1+ did not finish in {cfg.timeout:.0f} s") from exc
    text = _decode(p.stdout)
    blocks = _blocks(text)
    if "crash" in blocks or "summary" not in blocks:
        tail = blocks.get("crash", text + _decode(p.stderr))[-1500:]
        raise Z1PlusFailed(f"Z1+ wrote no summary. Its log ends:\n{tail}")
    return _result(blocks, partners, classes)


def run_z1_file(z1_file, partners: bool = True, config=None) -> Z1Result:
    """Run Z1+ on an existing ``.Z1`` file (see :func:`write_z1`)."""
    chains, box = read_z1(z1_file)
    return run_z1(chains, box, partners=partners, config=config)


def read_z1(path) -> tuple[list, np.ndarray]:
    """Chains and box from a ``.Z1`` file."""
    tok = Path(path).read_text(encoding="utf-8").split()
    n = int(tok[0])
    box = np.array([float(x) for x in tok[1:4]])
    counts = [int(x) for x in tok[4:4 + n]]
    xyz = np.array(tok[4 + n:], float).reshape(-1, 3)
    chains, i = [], 0
    for c in counts:
        chains.append(xyz[i:i + c])
        i += c
    return chains, box


def _blocks(text: str) -> dict:
    out, name, buf = {}, None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("@@"):
            if name is not None:
                out[name] = "\n".join(buf)
            name, buf = s[2:], []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf)
    return out


def parse_summary(text: str) -> dict:
    """The whole-system line of ``Z1+summary.dat``.

    Columns: 2 chains, 3 beads per chain N, 4 <Ree^2>, 5 <Lpp>, 6 <Z>,
    10 Ne from the classical Kuhn estimator (CK), 11 modified Kuhn (MK),
    12 classical coil (CC). The line is kept verbatim as ``summary_line``.
    """
    rows = [ln.split() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    rows = [r for r in rows if len(r) >= 12]
    if not rows:
        raise Z1PlusFailed(f"no summary line in Z1+summary.dat:\n{text[:400]}")
    f = rows[-1]
    return {"n_chains": int(float(f[1])), "N": float(f[2]),
            "Ree2": float(f[3]), "Lpp": float(f[4]), "Zmean": float(f[5]),
            "Ne_CK": float(f[9]), "Ne_MK": float(f[10]),
            "Ne_CC": float(f[11]), "summary_line": " ".join(f)}


def parse_sp(text: str) -> list:
    """Per chain, the partner chain (1-based) of each kink, from ``Z1+SP.dat``.

    The file is the chain count, the box, then per chain its point count and
    one row per point: ``x y z s E``, with ``chain node`` appended when E is
    1 (an entanglement point). A partner of 0 or below is dropped.
    """
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    n = int(float(rows[0][0]))
    i = 2                                         # past the box line
    partners = []
    for _ in range(n):
        m = int(float(rows[i][0]))
        i += 1
        mine = []
        for r in rows[i:i + m]:
            if len(r) >= 7 and int(float(r[4])) == 1:
                p = int(float(r[5]))
                if p > 0:
                    mine.append(p)
        i += m
        partners.append(mine)
    return partners


def parse_sp_paths(text: str) -> list:
    """Per chain, the primitive path of ``Z1+SP.dat`` as an ``(m, 5)`` array.

    Columns ``x y z kink partner``: kink is 1 at an entanglement point, and
    partner the chain (1-based) responsible, 0 where there is none.
    """
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    n = int(float(rows[0][0]))
    i = 2
    out = []
    for _ in range(n):
        m = int(float(rows[i][0]))
        i += 1
        pts = []
        for r in rows[i:i + m]:
            kink = int(float(r[4])) if len(r) >= 5 else 0
            partner = int(float(r[5])) if kink == 1 and len(r) >= 7 else 0
            pts.append([float(r[0]), float(r[1]), float(r[2]), kink, max(partner, 0)])
        i += m
        out.append(np.array(pts, float).reshape(-1, 5))
    return out


def _result(blocks, partners, classes) -> Z1Result:
    summary = parse_summary(blocks["summary"])
    Z = np.array([int(float(x)) for x in blocks["Z"].split()], int)
    Lpp = np.array([float(x) for x in blocks["Lpp"].split()])
    Ree = np.array([float(x) for x in blocks["Ree"].split()])
    has_sp = partners and blocks.get("SP")
    sp = parse_sp(blocks["SP"]) if has_sp else None
    paths = parse_sp_paths(blocks["SP"]) if has_sp else None
    return Z1Result(summary=summary, Z=Z, Lpp=Lpp, Ree=Ree, partners=sp,
                    classes=None if classes is None else np.asarray(classes),
                    paths=paths)


# ---------------------------------------------------------------------------
# Reading a result
# ---------------------------------------------------------------------------

def partner_graph(partners) -> tuple[collections.Counter, np.ndarray]:
    """The chain-chain entanglement multigraph of a ``-SP+`` run.

    Returns ``(pairs, degree)``: ``pairs`` counts reports per unordered chain
    pair (1-based ids), recorded when either chain reports the other, and
    ``degree[i]`` is the number of distinct partners of chain ``i + 1``.
    """
    pairs = collections.Counter()
    for i, plist in enumerate(partners, start=1):
        for j in plist:
            if j is None or j == i:
                continue
            pairs[(min(i, j), max(i, j))] += 1
    deg = collections.Counter()
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    return pairs, np.array([deg.get(i, 0) for i in range(1, len(partners) + 1)])


def summarise(res: Z1Result) -> dict:
    """A result as plain numbers: whole system, per class, partner graph.

    The layout of ``measure_system.py``'s ``z1`` block, so earlier
    measurements and new ones read the same.
    """
    Z = res.Z
    out = {k: v for k, v in res.summary.items() if k != "summary_line"}
    out.update(n_chains=int(len(Z)),
               frac_zero=float((Z == 0).mean()) if len(Z) else None,
               Zmax=int(Z.max()) if len(Z) else 0,
               Zsd=float(Z.std()) if len(Z) else None,
               hist=np.bincount(Z).tolist() if len(Z) else [],
               summary_line=res.summary.get("summary_line"))
    if res.classes is not None and len(res.classes) == len(Z):
        out["by_class"] = {}
        for c in CLASSES:
            m = res.classes == c
            if m.sum():
                out["by_class"][c] = {
                    "n": int(m.sum()), "Zmean": float(Z[m].mean()),
                    "frac_zero": float((Z[m] == 0).mean()),
                    "hist": np.bincount(Z[m]).tolist(),
                    "Zsd": float(Z[m].std()),
                    "Lpp_mean": float(res.Lpp[m].mean()),
                    "Ree_mean": float(res.Ree[m].mean())}
    if res.partners is not None:
        pairs, pdeg = partner_graph(res.partners)
        out["partner_pairs"] = len(pairs)
        out["partner_multiplicity"] = {
            int(k): int(v) for k, v in
            sorted(collections.Counter(pairs.values()).items())}
        out["partner_degree_mean"] = float(pdeg.mean()) if len(pdeg) else 0.0
        out["partner_degree_hist"] = np.bincount(pdeg).tolist()
        if res.classes is not None and len(res.classes) == len(pdeg):
            out["partner_degree_hist_bridge"] = np.bincount(
                pdeg[res.classes == "bridge"]).tolist()
    return out


def measure_system(system: EndLinkedSystem, partners: bool = True,
                   config=None, seed: int = 0) -> tuple[dict, Z1Result]:
    """Whole-system Z1+ of an end-linked system, split by chain class."""
    chains, cls, _mols = export_system(system, seed=seed)
    res = run_z1(chains, system.box, partners=partners, config=config,
                 classes=cls)
    return summarise(res), res


@dataclass
class SeedSummary:
    """Z1+ of one configuration over several seeds of the export's jitter.

    A single run cannot be read pair by pair: the exporter moves every
    junction by :data:`JITTER` so Z1+ does not crash on shared points, and on
    a DP-30 atomistic network another seed for that move changes 10 to 15 of
    55 partner pairs on the identical configuration. ``robust`` are the
    pairs found at every seed, ``seen`` those found at any.
    """

    z_bridge: Optional[float]           # mean over seeds, per bridge
    z_bridge_sd: Optional[float]
    z_all: float                        # mean over seeds, every chain
    robust: set
    seen: set
    seeds: int
    first: tuple                        # (summary dict, Z1Result) of the first seed


def measure_seeds(system: EndLinkedSystem, seeds: int = 8,
                  config=None) -> SeedSummary:
    """:func:`measure_system` at seeds ``0 .. seeds - 1``, summarised."""
    zb, za, sets, first = [], [], [], None
    for seed in range(max(1, int(seeds))):
        z, res = measure_system(system, config=config, seed=seed)
        bridge = (z.get("by_class") or {}).get("bridge")
        if bridge:
            zb.append(bridge["Zmean"])
        za.append(z["Zmean"])
        sets.append(set(partner_graph(res.partners)[0]) if res.partners is not None else set())
        if first is None:
            first = (z, res)
    return SeedSummary(
        z_bridge=float(np.mean(zb)) if zb else None,
        z_bridge_sd=float(np.std(zb)) if zb else None,
        z_all=float(np.mean(za)),
        robust=set.intersection(*sets) if sets else set(),
        seen=set.union(*sets) if sets else set(),
        seeds=len(sets), first=first)


def measure_checkpoint(path, graph: bool = False, z1: bool = True,
                       config=None, label: Optional[str] = None
                       ) -> tuple[dict, dict]:
    """Everything the validation measured on one checkpoint.

    Chain statistics (every bond scanned), chord statistics and orientation,
    Z1+ with partners split by class, and with ``graph`` the connectivity
    descriptors. Returns ``(record, per_chain)``: the record in the layout
    ``measure_system.py`` wrote, and the per-chain arrays (``Z``, ``cls``,
    ``ree``, ``chord``, ``Lpp``, ``partner_degree``) for an npz. When Z1+
    cannot run or fails, the record carries ``z1_error`` instead of ``z1``.
    """
    from topon.analysis.descriptors import describe, spatial

    system = read_endlinked(path)
    record, per_chain = chain_statistics(system)
    record = {"label": label, "data": str(path), **record}
    try:
        record["orientation_eigs"] = spatial(system.graph)[0]["orientation_eigs"]
    except ValueError:
        pass
    if z1:
        try:
            z, res = measure_system(system, config=config)
            record["z1"] = z
            per_chain["Z"] = res.Z
            per_chain["Lpp"] = res.Lpp
            if res.partners is not None:
                per_chain["partner_degree"] = partner_graph(res.partners)[1]
        except (Z1PlusUnavailable, Z1PlusFailed) as exc:
            record["z1_error"] = str(exc)
    if graph:
        record["graph"] = describe(system.graph).scalars
    return record, per_chain


def bridge_z(record: dict) -> Optional[float]:
    """Z per bridge from a :func:`measure_checkpoint` record, else the mean."""
    z = record.get("z1") or {}
    bridge = (z.get("by_class") or {}).get("bridge") or {}
    return bridge.get("Zmean", z.get("Zmean"))
