"""Connectivity descriptors of a polymer network's strand graph.

The set that separated network topologies in the bond/create validation
(``bond_create_validation/REPORT.md`` section 3): the shortest cycle through
each strand, clustering, betweenness, effective resistance, the spectrum. Each
one is defined in docs/USAGE.md section 3.6, which is the place to read what a
number means before reading anything into it.

Input is a NetworkX graph in the strand convention: nodes are junctions and
dangling-chain ends, edges are strands, a primary loop is a self-loop and a
secondary loop is a parallel edge. A node's ``kind`` attribute (``junction``
or ``end``) says which it is. Graphs from topon's own generators carry no
``kind``; there a degree-1 node is the free end of a dangling strand, the rule
:func:`topon.conformation.placement.chains.strand_plans` and the chemistry
builder apply, and a degree-0 node is a lattice site the search left empty,
which is not part of the network and is dropped.

Two views are measured:

``full``
    every junction and end node, self-loops removed, parallel edges merged,
    largest connected component
``core``
    the full view with degree-1 nodes pruned until none are left, i.e. the
    elastically active backbone

Most measures are taken on the core. ``lambda2`` and the mean path length
depend on system size, so they compare only between graphs of matched size;
everything :func:`compare` folds into its composite is size-free.

Ported from ``graph_descriptors.py`` of the validation scripts. The numbers it
returns for the DP-20 reference are the ones in the report.
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
from scipy import stats as sstats
from scipy.sparse.linalg import eigsh

#: Largest core, in nodes, for the dense eigen-decomposition behind the graph
#: energy and the spectral radius (about 1 GB of memory at the limit).
ENERGY_LIMIT = 6000
#: Largest core for the Laplacian pseudo-inverse behind the effective
#: resistance and the Kirchhoff index (dense, cubic in the node count).
RESISTANCE_LIMIT = 4000

#: Cycle lengths the Jensen-Shannon divergence compares, ``[3, 16)``.
CYCLE_RANGE = (3, 16)

#: The size-free scalars :func:`compare` folds into its composite, by the
#: short name the comparison reports them under.
INTENSIVE = {
    "transitivity": "transitivity_core",
    "square_clustering": "square_clustering_core",
    "assortativity": "assortativity_core",
    "betweenness_gini": "betweenness_gini_core",
    "resistance_cv": "edge_eff_resistance_cv",
    "graph_energy": "graph_energy_core_per_node",
    "cycle_mean": "edge_shortest_cycle_mean",
    "cycle_le4": "frac_edges_in_cycle_le4",
}


@dataclass
class Descriptors:
    """Scalars and per-node / per-edge distributions of one graph.

    Unpacks as ``scalars, dists = describe(G)``, the shape the validation
    scripts use.
    """

    scalars: dict
    dists: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.scalars, self.dists))

    def __getitem__(self, key):
        return self.scalars[key]

    def save(self, json_path) -> tuple[Path, Path]:
        """Write the scalars as JSON and the distributions beside it as npz."""
        json_path = Path(json_path)
        npz_path = json_path.with_suffix(".npz")
        json_path.write_text(json.dumps(self.scalars, indent=1, default=to_jsonable),
                             encoding="utf-8")
        np.savez_compressed(npz_path, **{k: np.asarray(v)
                                         for k, v in self.dists.items()})
        return json_path, npz_path

    @classmethod
    def load(cls, json_path) -> "Descriptors":
        """Read what :meth:`save` wrote; the npz is optional."""
        json_path = Path(json_path)
        scalars = json.loads(json_path.read_text(encoding="utf-8"))
        npz_path = json_path.with_suffix(".npz")
        dists = {}
        if npz_path.exists():
            with np.load(npz_path) as z:
                dists = {k: z[k] for k in z.files}
        return cls(scalars, dists)


def to_jsonable(x):
    """``json.dumps`` default for NumPy scalars and arrays."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------------------
# Small measures
# ---------------------------------------------------------------------------

def gini(x) -> float:
    """Gini coefficient of non-negative values (0 even, 1 all on one)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def bimodality(x) -> float:
    """Sarle's bimodality coefficient; above 5/9 hints at two modes."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    g = sstats.skew(x)
    k = sstats.kurtosis(x)
    return float((g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def algebraic_connectivity(H) -> float:
    """Second-smallest Laplacian eigenvalue (0 for a disconnected graph)."""
    n = H.number_of_nodes()
    if n < 2:
        return float("nan")
    Lm = nx.laplacian_matrix(H).astype(float)
    if n > 2:
        try:
            # A fixed start vector: ARPACK's default is random, which leaves
            # the last digits different from one call to the next.
            v0 = np.random.default_rng(0).random(n)
            vals = eigsh(Lm, k=2, which="SM", return_eigenvectors=False,
                         tol=1e-6, maxiter=10000, v0=v0)
            return float(np.sort(vals)[1])
        except Exception:
            pass
    return float(np.sort(np.linalg.eigvalsh(Lm.toarray()))[1])


def shortest_cycle_per_edge(H) -> np.ndarray:
    """Length of the shortest cycle through each edge of a simple graph.

    One plus the shortest path between the edge's ends with the edge itself
    removed; ``inf`` for a bridge, which is on no cycle. Works on a copy.
    """
    H = nx.Graph(H)
    out = []
    for u, v in list(H.edges()):
        H.remove_edge(u, v)
        try:
            out.append(nx.shortest_path_length(H, u, v) + 1)
        except nx.NetworkXNoPath:
            out.append(np.inf)
        H.add_edge(u, v)
    return np.array(out, float)


def node_kind(G, n) -> Optional[str]:
    """``junction``, ``end``, or None for an empty lattice site.

    A declared ``kind`` decides. Otherwise the degree does: 1 is the free end
    of a dangling strand, 0 a vacancy (dropped), anything else a junction.
    """
    declared = G.nodes[n].get("kind")
    if declared in ("junction", "end"):
        return declared
    d = G.degree(n)
    if d == 0:
        return None
    return "end" if d == 1 else "junction"


def network_view(G) -> nx.MultiGraph:
    """``G`` with every node carrying ``kind`` and vacancies removed.

    A copy; the input is never modified.
    """
    H = nx.MultiGraph()
    H.graph.update(G.graph)
    for n, data in G.nodes(data=True):
        k = node_kind(G, n)
        if k is not None:
            H.add_node(n, **{**data, "kind": k})
    multi = G.is_multigraph()
    edges = (G.edges(keys=True, data=True) if multi
             else ((u, v, None, d) for u, v, d in G.edges(data=True)))
    for u, v, _k, data in edges:
        H.add_edge(u, v, **data)
    return H


def _sol_count(G) -> int:
    sol = G.graph.get("sol_chains")
    if isinstance(sol, dict):
        return int(sol.get("count", 0))
    return int(sol or 0)


def _dist_counts(values) -> dict:
    return {int(k): int(v) for k, v in
            sorted(collections.Counter(np.asarray(values).astype(int)).items())}


# ---------------------------------------------------------------------------
# The descriptor set
# ---------------------------------------------------------------------------

def strand_counts(G) -> dict:
    """Junctions, strands by class, loops and the two degree distributions.

    The cheap first block of :func:`describe`, for a caller that wants the
    counts of a large network without the rest.
    """
    out, _eff = _counts(G, network_view(G))
    return out


def _counts(G, V) -> tuple[dict, np.ndarray]:
    out: dict = {}
    kind = {n: d["kind"] for n, d in V.nodes(data=True)}
    junc = [n for n in V.nodes() if kind[n] == "junction"]
    ends = [n for n in V.nodes() if kind[n] == "end"]

    out["n_junctions"], out["n_end_nodes"] = len(junc), len(ends)
    out["n_vacancies"] = G.number_of_nodes() - V.number_of_nodes()
    out["n_edges_total"] = V.number_of_edges()
    out["n_primary_loops"] = sum(1 for u, v in V.edges() if u == v)
    mult = collections.Counter((u, v) if str(u) <= str(v) else (v, u)
                               for u, v in V.edges() if u != v)
    out["n_secondary_loops"] = sum(c - 1 for c in mult.values() if c > 1)
    out["n_bridging_chains"] = sum(
        1 for u, v in V.edges()
        if u != v and kind[u] == "junction" and kind[v] == "junction")
    out["n_dangling_chains"] = sum(1 for u, v in V.edges()
                                   if kind[u] == "end" or kind[v] == "end")
    out["n_sol_chains"] = _sol_count(G)

    # Chemical degree counts a primary loop twice (both of its ends are on the
    # junction); effective degree leaves loops out.
    chem = np.array([V.degree(n) for n in junc], float)
    eff = np.array([sum(1 for u, v in V.edges(n) if u != v) for n in junc],
                   float)
    for name, d in (("chem", chem), ("eff", eff)):
        if not len(d):
            continue
        c = collections.Counter(d.astype(int))
        out[f"deg_{name}_dist"] = {int(k): int(v) for k, v in sorted(c.items())}
        out[f"deg_{name}_mean"] = float(d.mean())
        out[f"deg_{name}_std"] = float(d.std())
        p = np.array(list(c.values()), float) / len(d)
        out[f"deg_{name}_entropy"] = float(-(p * np.log(p)).sum())
        out[f"deg_{name}_skew"] = float(sstats.skew(d)) if d.std() > 0 else 0.0
        out[f"deg_{name}_bimodality"] = bimodality(d)
    return out, eff


def describe(G, sample_bc: int = 400, seed: int = 0,
             heavy: bool = True) -> Descriptors:
    """Every graph descriptor of the strand graph ``G``.

    ``sample_bc`` source nodes estimate node and edge betweenness (exact when
    the core is smaller), 200 the path lengths, both drawn with ``seed``, so a
    graph always gets the same numbers. ``heavy=False`` skips the
    shortest-cycle spectrum, the effective resistance and the edge betweenness,
    which dominate the time on a large network.

    Returns :class:`Descriptors`; ``G`` is not modified.
    """
    V = network_view(G)
    rng = np.random.default_rng(seed)
    out, eff = _counts(G, V)
    dists: dict = {"deg_eff": eff}
    kind = {n: d["kind"] for n, d in V.nodes(data=True)}
    junc = [n for n in V.nodes() if kind[n] == "junction"]

    S = nx.Graph()
    S.add_nodes_from(V.nodes())
    S.add_edges_from((u, v) for u, v in V.edges() if u != v)
    comps = sorted(nx.connected_components(S), key=len, reverse=True)
    out["n_components"] = len(comps)
    if not comps:
        return Descriptors(out, dists)
    out["giant_frac_junctions"] = (
        sum(1 for n in comps[0] if kind[n] == "junction") / max(1, len(junc)))
    H = S.subgraph(comps[0]).copy()
    nH, eH = H.number_of_nodes(), H.number_of_edges()
    out["giant_nodes"], out["giant_edges"] = nH, eH
    out["cycle_rank"] = eH - nH + 1
    out["cycle_rank_per_junction"] = (eH - nH + 1) / max(1, len(junc))

    C = H.copy()
    while True:
        leaves = [n for n, d in C.degree() if d <= 1]
        if not leaves:
            break
        C.remove_nodes_from(leaves)
    nC, eC = C.number_of_nodes(), C.number_of_edges()
    out["core_nodes"], out["core_edges"] = nC, eC
    out["frac_active_junctions"] = nC / max(1, len(junc))
    out["core_cycle_rank_per_node"] = (eC - nC + 1) / max(1, nC)
    cdeg = np.array([d for _, d in C.degree()], float)
    out["core_deg_mean"] = float(cdeg.mean()) if nC else float("nan")
    out["core_deg_dist"] = _dist_counts(cdeg)
    dists["core_deg"] = cdeg

    out["lambda2_full"] = algebraic_connectivity(H)
    out["lambda2_core"] = algebraic_connectivity(C) if nC > 2 else float("nan")
    if nC and nC <= ENERGY_LIMIT:
        A = nx.to_scipy_sparse_array(C, dtype=float)
        ev = np.linalg.eigvalsh(A.toarray())
        out["graph_energy_core_per_node"] = float(np.abs(ev).sum() / nC)
        out["spectral_radius_core"] = float(ev.max())

    if nC:
        srcs = list(rng.choice(list(C.nodes()), size=min(200, nC),
                               replace=False))
        pl, ecc = [], []
        for s in srcs:
            d = nx.single_source_shortest_path_length(C, s)
            v = np.array(list(d.values()))
            pl.append(v[v > 0])
            ecc.append(v.max())
        pl = np.concatenate(pl)
        if len(pl):
            out["avg_path_core"] = float(pl.mean())
            out["diameter_core_est"] = int(max(ecc))
            dists["path_len_core"] = pl
    if not eC:
        return Descriptors(out, dists)

    out["assortativity_core"] = _safe(nx.degree_assortativity_coefficient, C)
    out["transitivity_core"] = float(nx.transitivity(C))
    out["avg_clustering_core"] = float(nx.average_clustering(C))
    out["square_clustering_core"] = float(
        np.mean(list(nx.square_clustering(C).values())))
    out["n_bridges_core"] = sum(1 for _ in nx.bridges(C))
    out["n_articulation_core"] = sum(1 for _ in nx.articulation_points(C))
    kc = nx.core_number(C)
    out["mean_k_core"] = float(np.mean(list(kc.values())))
    out["max_k_core"] = int(max(kc.values()))

    bc = nx.betweenness_centrality(C, k=min(sample_bc, nC), seed=int(seed),
                                   normalized=True)
    b = np.array(list(bc.values()))
    out["max_betweenness_core"] = float(b.max())
    out["mean_betweenness_core"] = float(b.mean())
    out["betweenness_gini_core"] = gini(b)
    dists["betweenness_core"] = b
    try:
        ec = nx.eigenvector_centrality_numpy(C)
        e = np.array(list(ec.values()))
        out["mean_eigvec_cent_core"] = float(e.mean())
        out["eigvec_cent_cv_core"] = float(e.std() / e.mean())
        dists["eigvec_cent_core"] = e
    except Exception:
        pass

    if not heavy:
        return Descriptors(out, dists)

    sc = shortest_cycle_per_edge(C)
    fin = sc[np.isfinite(sc)]
    out["edge_shortest_cycle_mean"] = float(fin.mean()) if len(fin) else float("nan")
    out["edge_shortest_cycle_dist"] = _dist_counts(fin)
    out["frac_edges_in_cycle_le4"] = float((fin <= 4).sum() / len(sc))
    out["frac_edges_in_cycle_le6"] = float((fin <= 6).sum() / len(sc))
    out["frac_odd_cycles"] = (float((fin.astype(int) % 2 == 1).sum() / len(fin))
                              if len(fin) else float("nan"))
    dists["edge_shortest_cycle"] = sc

    if nC <= RESISTANCE_LIMIT:
        Lm = nx.laplacian_matrix(C).toarray().astype(float)
        Lp = np.linalg.pinv(Lm)
        dd = np.diag(Lp)
        idx = {n: i for i, n in enumerate(C.nodes())}
        R = np.array([dd[idx[u]] + dd[idx[v]] - 2 * Lp[idx[u], idx[v]]
                      for u, v in C.edges()])
        out["edge_eff_resistance_mean"] = float(R.mean())
        out["edge_eff_resistance_cv"] = float(R.std() / R.mean())
        out["kirchhoff_per_node"] = float(np.trace(Lp))
        dists["edge_eff_resistance"] = R

    eb = np.array(list(nx.edge_betweenness_centrality(
        C, k=min(sample_bc, nC), seed=int(seed)).values()))
    out["edge_betweenness_gini_core"] = gini(eb)
    out["edge_betweenness_cv_core"] = float(eb.std() / eb.mean())
    dists["edge_betweenness_core"] = eb
    return Descriptors(out, dists)


def _safe(fn, *a) -> float:
    """A NetworkX coefficient, or nan where it is undefined (a regular graph)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return float(fn(*a))
        except (ZeroDivisionError, ValueError):
            return float("nan")


def graph_box(G) -> Optional[np.ndarray]:
    """The periodic cell in the units of the node positions, if recorded.

    Graphs the validation scripts saved keep positions in sigma with the cell
    in ``box_sigma``; topon's generators keep both in lattice units under
    ``box``; the end-linked reader stores the data file's box under ``box``.
    """
    for key in ("box_sigma", "box"):
        if G.graph.get(key) is not None:
            return np.asarray(G.graph[key], float).reshape(3)
    return None


def spatial(G, box=None) -> tuple[dict, np.ndarray]:
    """Chord statistics and orientation tensor of the bridging strands.

    A chord is the minimum-image distance between the two junctions of a
    bridge, in the units of the node positions. The orientation tensor is
    the mean of ``u u^T`` over the chord directions; its three eigenvalues
    are 1/3 each for an isotropic network.

    Returns ``(scalars, chords)``; raises ``ValueError`` when there is no box
    or no bridge.
    """
    V = network_view(G)
    L = np.asarray(box, float) if box is not None else graph_box(G)
    if L is None:
        raise ValueError("spatial descriptors need the periodic cell")
    kind = {n: d["kind"] for n, d in V.nodes(data=True)}
    r, vec = [], []
    for u, v in V.edges():
        if u == v or kind[u] != "junction" or kind[v] != "junction":
            continue
        d = (np.asarray(V.nodes[u]["pos"], float)
             - np.asarray(V.nodes[v]["pos"], float))
        d -= L * np.round(d / L)
        n = np.linalg.norm(d)
        if n > 0:
            r.append(n)
            vec.append(d / n)
    if not r:
        raise ValueError("no bridging strand to measure")
    r = np.array(r)
    vec = np.array(vec)
    Q = np.einsum("ni,nj->ij", vec, vec) / len(vec)
    return {"chord_mean": float(r.mean()), "chord_sd": float(r.std()),
            "chord_cv": float(r.std() / r.mean()),
            "chord_p5": float(np.percentile(r, 5)),
            "chord_p50": float(np.percentile(r, 50)),
            "chord_p95": float(np.percentile(r, 95)),
            "chord_max": float(r.max()),
            "orientation_eigs": [float(x) for x in np.linalg.eigvalsh(Q)]}, r


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def cycle_js(p: dict, q: dict, kmin: int = CYCLE_RANGE[0],
             kmax: int = CYCLE_RANGE[1]) -> float:
    """Jensen-Shannon divergence (base 2, 0 to 1) of two cycle spectra.

    ``p`` and ``q`` are ``{cycle length: strand count}``, the
    ``edge_shortest_cycle_dist`` of two descriptor sets; lengths outside
    ``[kmin, kmax)`` are left out.
    """
    P = np.array([p.get(str(k), p.get(k, 0)) for k in range(kmin, kmax)], float)
    Q = np.array([q.get(str(k), q.get(k, 0)) for k in range(kmin, kmax)], float)
    P /= max(P.sum(), 1)
    Q /= max(Q.sum(), 1)
    M = 0.5 * (P + Q)

    def kl(a, b):
        m = a > 0
        return float((a[m] * np.log2(a[m] / b[m])).sum())

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def _odd_fraction(s: dict) -> Optional[float]:
    if s.get("frac_odd_cycles") is not None:
        return float(s["frac_odd_cycles"])
    cd = s.get("edge_shortest_cycle_dist")
    if not cd:
        return None
    tot = sum(cd.values())
    return sum(v for k, v in cd.items() if int(k) % 2) / tot if tot else None


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def compare(a, b, population: Optional[Iterable] = None) -> dict:
    """How far the descriptors ``a`` sit from the reference ``b``.

    Returns the Jensen-Shannon divergence of the two cycle spectra
    (``cycle_js``), the two-sample Kolmogorov-Smirnov statistic of every
    distribution both carry (``ks``), the normalised deviation of each size-free
    scalar (``deviations``) and their mean, the ``composite``: 0 for identical
    graphs, larger the further apart.

    The composite averages eleven terms, as in the validation sweep: the cycle
    divergence, the KS statistic of the core betweenness, the odd-cycle
    fraction and the eight scalars of :data:`INTENSIVE`, each as
    ``|a - b| / |b|``. ``b`` is the reference, so the measure is not symmetric.
    With ``population`` (the descriptor sets of a sweep) an intensive scalar is
    divided by ``max(|b|, sd over the population)`` instead, which is the
    sweep's own normalisation and reproduces its composites exactly. A term
    the reference has at zero and no population rescues, or that either side
    lacks, is left out and named in ``omitted``.

    ``a`` and ``b`` are :class:`Descriptors` (or ``(scalars, dists)`` pairs).
    """
    sa, da = a
    sb, db = b
    pop_sd = {}
    if population is not None:
        pop = [p.scalars if isinstance(p, Descriptors)
               else p if isinstance(p, dict) else p[0] for p in population]
        for key in INTENSIVE.values():
            vals = [float(p[key]) for p in pop if _finite(p.get(key))]
            if len(vals) > 1:
                pop_sd[key] = float(np.std(vals, ddof=1))

    out: dict = {"deviations": {}, "ks": {}, "omitted": []}
    parts = []

    ca, cb = sa.get("edge_shortest_cycle_dist"), sb.get("edge_shortest_cycle_dist")
    if ca and cb:
        out["cycle_js"] = cycle_js(ca, cb)
        parts.append(out["cycle_js"])
    else:
        out["omitted"].append("cycle_js")

    for name in sorted(set(da) & set(db)):
        x, y = np.asarray(da[name], float), np.asarray(db[name], float)
        if len(x) and len(y):
            out["ks"][name] = float(sstats.ks_2samp(x, y).statistic)
    if "betweenness_core" in out["ks"]:
        parts.append(out["ks"]["betweenness_core"])
    else:
        out["omitted"].append("ks_betweenness")

    for short, key in INTENSIVE.items():
        va, vb = sa.get(key), sb.get(key)
        scale = abs(vb) if _finite(vb) else 0.0
        scale = max(scale, pop_sd.get(key, 0.0))
        if not (_finite(va) and _finite(vb)) or scale <= 1e-12:
            out["omitted"].append(short)
            continue
        out["deviations"][short] = abs(va - vb) / scale
        parts.append(out["deviations"][short])

    oa, ob = _odd_fraction(sa), _odd_fraction(sb)
    if _finite(oa) and _finite(ob) and ob > 1e-12:
        out["deviations"]["odd_fraction"] = abs(oa - ob) / ob
        parts.append(out["deviations"]["odd_fraction"])
    else:
        out["omitted"].append("odd_fraction")

    out["n_terms"] = len(parts)
    out["composite"] = float(np.mean(parts)) if parts else float("nan")
    return out
