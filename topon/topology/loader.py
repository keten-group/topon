"""
Graph loader for Topon.

Handles loading topology from various file formats:

* gpickle / .nodes+.edges -- basic topology (lattice + connectivity).
* graphml / npz           -- post-assignment dual graph: chains + crosslinks
                              with DP, entanglements (multiplicity preserved),
                              and crosslink positions. Suitable for skipping
                              the topology + analysis + assignment stages and
                              going straight into chemistry/conformation/output.
"""

import pickle
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd


def load_graph(
    gpickle_path: Optional[Union[str, Path]] = None,
    nodes_path: Optional[Union[str, Path]] = None,
    edges_path: Optional[Union[str, Path]] = None,
) -> tuple[nx.MultiGraph, Optional[np.ndarray]]:
    """
    Load a topology graph from file(s).
    
    Args:
        gpickle_path: Path to a .gpickle file (takes precedence).
        nodes_path: Path to a .nodes file.
        edges_path: Path to a .edges file.
        
    Returns:
        Tuple of (NetworkX MultiGraph, box dimensions array or None).
        
    Raises:
        ValueError: If no valid file paths provided.
        FileNotFoundError: If specified files don't exist.
    """
    if gpickle_path:
        return _load_from_gpickle(gpickle_path)
    elif nodes_path and edges_path:
        return _load_from_nodes_edges(nodes_path, edges_path)
    else:
        raise ValueError(
            "Must provide either gpickle_path or both nodes_path and edges_path"
        )


def _load_from_gpickle(path: Union[str, Path]) -> tuple[nx.MultiGraph, Optional[np.ndarray]]:
    """
    Load graph from a .gpickle file.
    
    Args:
        path: Path to .gpickle file.
        
    Returns:
        Tuple of (graph, dims).
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Gpickle file not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Handle different gpickle formats
    if isinstance(data, tuple) and len(data) == 2:
        # Format: (graph, dims)
        G, dims = data
    elif isinstance(data, nx.Graph):
        # Format: just the graph
        G = data
        dims = infer_dims_from_graph(G)
    elif isinstance(data, dict) and "graph" in data:
        # Format: dict with 'graph' and optionally 'dims'
        G = data["graph"]
        dims = data.get("dims")
    else:
        raise ValueError(f"Unrecognized gpickle format in {path}")
    
    # Ensure it's a MultiGraph
    if not isinstance(G, nx.MultiGraph):
        G = nx.MultiGraph(G)

    # A recorded box wins over a stored dims: it is the generator's exact
    # cell, whereas a dims saved alongside an older graph may have come
    # from the positional fallback (and be half a cell too large on any
    # lattice with fractional basis sites).
    if G.graph.get("box") is not None:
        dims = infer_dims_from_graph(G)

    # Remove vacancies (degree-0 nodes)
    n_removed = remove_vacancies(G)
    
    print(f"Loaded graph from {path.name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if n_removed > 0:
        print(f"  Removed {n_removed} vacancies (degree-0 nodes)")
    
    return G, dims


def _load_from_nodes_edges(
    nodes_path: Union[str, Path],
    edges_path: Union[str, Path]
) -> tuple[nx.MultiGraph, Optional[np.ndarray]]:
    """
    Load graph from .nodes and .edges files.
    
    File formats:
    - .nodes: NodeID X Y Z Degree (whitespace-separated, # comments)
    - .edges: Node1 Node2 (whitespace-separated, # comments)
    
    Args:
        nodes_path: Path to .nodes file.
        edges_path: Path to .edges file.
        
    Returns:
        Tuple of (graph, dims).
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    
    # Load nodes
    nodes_df = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["id", "x", "y", "z", "degree"]
    )
    
    # Load edges
    edges_df = pd.read_csv(
        edges_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["u", "v"]
    )
    
    # Build graph
    G = nx.MultiGraph()

    for _, row in nodes_df.iterrows():
        G.add_node(
            int(row["id"]),
            pos=(float(row["x"]), float(row["y"]), float(row["z"]))
        )

    for _, row in edges_df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)

    # An optional "# BOX Lx Ly Lz" header carries the true periodic cell.
    # Files written without one fall back to the positional heuristic.
    box = read_box_header(nodes_path)
    if box is not None:
        G.graph["box"] = box

    # "# PERIODICITY 110" records which axes are open. Absent means fully
    # periodic, which is what every file predating the header represents.
    axes = read_periodicity_header(nodes_path)
    if axes is not None:
        G.graph["periodicity"] = axes

    # Infer dimensions from positions
    dims = infer_dims_from_graph(G)
    
    # Remove vacancies (degree-0 nodes)
    n_removed = remove_vacancies(G)
    
    print(f"Loaded graph from {nodes_path.name} + {edges_path.name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if n_removed > 0:
        print(f"  Removed {n_removed} vacancies (degree-0 nodes)")
    
    return G, dims


def remove_vacancies(G: nx.Graph) -> int:
    """
    Remove degree-0 nodes (vacancies) from graph.
    
    Vacancies are lattice positions with no edges - they should not
    become atoms in the simulation.
    
    Args:
        G: Graph to modify in-place.
        
    Returns:
        Number of nodes removed.
    """
    vacancies = [n for n in G.nodes() if G.degree(n) == 0]
    if vacancies:
        G.remove_nodes_from(vacancies)
    return len(vacancies)


# Header line the topology generators write into ``.nodes`` files to record
# the exact periodic cell, e.g. "# BOX 6 6 6". Held as a module constant so
# the Python reader/writer and the C generator agree on one spelling.
BOX_HEADER_KEY = "BOX"


PERIODICITY_HEADER_KEY = "PERIODICITY"


def graph_periodicity(G):
    """Per-axis boundaries a graph was built with, or None if all-periodic.

    Returns None both when nothing was recorded and when every axis is
    periodic, because every consumer treats those identically and the
    None case is the one that reproduces pre-open-boundary behaviour.

    Shared by ``Pipeline`` and the standalone workflows so the three
    cannot drift on what an open axis means.

    Args:
        G: Graph, possibly carrying a ``periodicity`` graph attribute.

    Returns:
        ``(px, py, pz)`` booleans, or None.
    """
    if G is None:
        return None
    axes = G.graph.get("periodicity")
    if axes is None:
        return None
    axes = tuple(bool(a) for a in axes)
    return None if all(axes) else axes


def format_periodicity_header(periodicity) -> str:
    """Render per-axis boundaries as a ``.nodes`` header line.

    Args:
        periodicity: Iterable of three truthy values, one per axis.

    Returns:
        The header line, e.g. ``"# PERIODICITY 110"``, without a newline.
    """
    digits = "".join("1" if p else "0" for p in periodicity)
    return f"# {PERIODICITY_HEADER_KEY} {digits}"


def read_periodicity_header(path: Union[str, Path]):
    """Read the ``# PERIODICITY 110`` header from a .nodes file.

    Returns ``(px, py, pz)`` booleans, or None when absent or malformed.
    Absent means fully periodic, which is what every file written before
    this header existed represents.
    """
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                parts = line.lstrip("#").split()
                if len(parts) == 2 and parts[0].upper() == PERIODICITY_HEADER_KEY:
                    digits = parts[1].strip()
                    if len(digits) == 3 and set(digits) <= {"0", "1"}:
                        return tuple(c == "1" for c in digits)
                    return None
    except OSError:
        return None
    return None


def format_box_header(box) -> str:
    """Render a 3-component box as its ``.nodes`` header line.

    Args:
        box: Iterable of three box lengths in lattice units.

    Returns:
        The header line, without a trailing newline.
    """
    lx, ly, lz = (float(v) for v in box)
    return f"# {BOX_HEADER_KEY} {lx:g} {ly:g} {lz:g}"


def read_box_header(path: Union[str, Path]) -> Optional[tuple[float, float, float]]:
    """Read the ``# BOX Lx Ly Lz`` header from a .nodes file.

    Only the leading comment block is scanned, so this stops after a
    handful of lines on any file. Returns None when the header is absent
    or malformed, which is the expected case for files written before
    generators recorded their box.

    Args:
        path: Path to a ``.nodes`` file.

    Returns:
        ``(Lx, Ly, Lz)`` in lattice units, or None.
    """
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                parts = line.lstrip("#").split()
                if len(parts) == 4 and parts[0].upper() == BOX_HEADER_KEY:
                    try:
                        lx, ly, lz = (float(v) for v in parts[1:4])
                    except ValueError:
                        return None
                    if min(lx, ly, lz) <= 0:
                        return None
                    return (lx, ly, lz)
    except OSError:
        return None
    return None


def save_nodes_edges(
    G: nx.Graph,
    nodes_path: Union[str, Path],
    edges_path: Union[str, Path],
    box=None,
    periodicity=None,
) -> None:
    """Write a graph in the ``.nodes`` / ``.edges`` format.

    Matches what the C generator emits, plus a ``# BOX`` header carrying
    the exact periodic cell so a reload does not have to guess it, and a
    ``# PERIODICITY`` header when any axis is open.

    Args:
        G: Graph whose nodes carry ``pos``.
        nodes_path: Destination ``.nodes`` path.
        edges_path: Destination ``.edges`` path.
        box: Periodic cell to record. Defaults to ``G.graph["box"]``;
             omitted from the header when neither is available.
        periodicity: Per-axis boundaries. Defaults to
             ``G.graph["periodicity"]``. Written only when an axis is
             actually open, so fully periodic files keep the exact
             format they had before this existed.
    """
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)

    if box is None:
        box = G.graph.get("box")
    if periodicity is None:
        periodicity = G.graph.get("periodicity")

    with open(nodes_path, "w") as f:
        if box is not None:
            f.write(format_box_header(box) + "\n")
        if periodicity is not None and not all(periodicity):
            f.write(format_periodicity_header(periodicity) + "\n")
        f.write("# NodeID X Y Z Degree\n")
        for node in sorted(G.nodes()):
            x, y, z = G.nodes[node].get("pos", (0.0, 0.0, 0.0))
            f.write(f"{node} {x:f} {y:f} {z:f} {G.degree(node)}\n")

    with open(edges_path, "w") as f:
        f.write("# Node1 Node2\n")
        for u, v in sorted(G.edges()):
            f.write(f"{u} {v}\n")


def infer_dims_from_graph(G: nx.Graph) -> Optional[np.ndarray]:
    """
    Infer box dimensions from node positions.
    
    Args:
        G: Graph with 'pos' node attributes.
        
    Returns:
        Box dimensions as numpy array, or None if no positions.

    Notes:
        Prefers the exact cell recorded by the generator in
        ``G.graph["box"]``. The ``max - min + 1`` fallback below is only
        correct when every site sits on an integer coordinate with unit
        spacing, i.e. simple cubic. BCC, FCC and Diamond place basis
        sites at fractional offsets, so the fallback overshoots the true
        cell by that offset: a 4x4x4 BCC or FCC lattice reports 4.5
        instead of 4.0. Since this value feeds every minimum-image
        calculation downstream, that overshoot makes roughly a third of
        BCC edges (and a quarter of FCC edges) resolve to the wrong
        periodic replica and be built at twice their true bond length.
        Generators therefore record the true cell explicitly; the
        fallback survives only for graphs written before they did
        (old gpickles, ``.nodes`` files with no ``# BOX`` header).
    """
    box = G.graph.get("box")
    if box is not None:
        arr = np.asarray(box, dtype=float).ravel()
        if arr.size == 3 and np.all(np.isfinite(arr)) and np.all(arr > 0):
            return arr

    positions = []
    for node, data in G.nodes(data=True):
        if "pos" in data:
            positions.append(data["pos"])
    
    if not positions:
        return None
    
    positions = np.array(positions)
    
    # Assume box starts at 0, dimensions are max + 1 (for lattice spacing)
    # This is a heuristic - actual dims should be stored in gpickle
    max_pos = positions.max(axis=0)
    min_pos = positions.min(axis=0)
    
    # For integer lattice positions, dims = max - min + 1
    dims = max_pos - min_pos + 1
    
    return dims


def save_graph(
    G: nx.Graph,
    output_path: Union[str, Path],
    dims: Optional[np.ndarray] = None
) -> None:
    """
    Save graph to a .gpickle file.
    
    Args:
        G: NetworkX graph to save.
        output_path: Path for output file.
        dims: Optional box dimensions to save with graph.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = (G, dims) if dims is not None else G
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved graph to {output_path}")


def get_node_positions(G: nx.Graph) -> dict[int, np.ndarray]:
    """
    Extract node positions as a dictionary.

    Args:
        G: Graph with 'pos' node attributes.

    Returns:
        Dict mapping node ID to position array.
    """
    positions = {}
    for node, data in G.nodes(data=True):
        if "pos" in data:
            positions[node] = np.array(data["pos"])
    return positions


# ======================================================================
# Dual-graph loaders (graphml + npz)
#
# Both formats are written by ``topon.writers.{graphml_writer,npz_writer}``
# in the dual representation:
#   * polymer nodes   (one per topon chain) carry DP via "length"
#   * crosslinker nodes (one per junction) carry pos via COMX/Y/Z
#   * chemical edges connect chain <-> its two end-crosslinks
#   * entanglement edges connect chain <-> chain, replicated `count` times
#     to preserve multiplicity
#
# The loaders here invert that transformation to rebuild the topon
# MultiGraph (junctions = nodes, chains = edges with dp / entangled_with /
# entanglement_count attributes). With that graph plus dims, the chemistry
# + conformation + output stages can be re-run without going through
# topology generation or assignment.
# ======================================================================


def load_graphml(
    path: Union[str, Path],
) -> tuple[nx.MultiGraph, Optional[np.ndarray]]:
    """Load a topon graphml dual graph back into a MultiGraph + dims.

    Reverses :func:`topon.writers.graphml_writer.write_graphml`.

    Args:
        path: Path to the ``.graphml`` file.

    Returns:
        ``(G, dims)`` where ``G`` is a ``nx.MultiGraph`` with crosslink
        nodes (carrying ``pos``) and chain edges (carrying ``dp``,
        ``entangled_with``, ``entanglement_count``). ``dims`` is the
        inferred box dimensions (from graph attributes if present, else
        from node positions).
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    # Parse <key> declarations so we can map data-element key IDs back to
    # their semantic names (e.g. "length", "COMX", "edge_type", "xhi").
    key_meta: dict[str, tuple[str, str, str]] = {}
    for key in root.findall("g:key", ns):
        key_meta[key.get("id")] = (
            key.get("for"),
            key.get("attr.name"),
            key.get("attr.type"),
        )

    graph_el = root.find("g:graph", ns)
    if graph_el is None:
        raise ValueError(f"No <graph> element in {path}")

    polymer_attrs: dict[int, dict] = {}     # chain dual-id -> attrs
    crosslink_attrs: dict[int, dict] = {}   # junction id   -> attrs

    for node in graph_el.findall("g:node", ns):
        nid = int(node.get("id"))
        attrs: dict = {}
        for d in node.findall("g:data", ns):
            kid = d.get("key")
            if kid not in key_meta:
                continue
            _, name, type_ = key_meta[kid]
            txt = d.text
            if txt is None or txt == "NaN":
                val = float("nan")
            elif type_ == "long":
                val = int(txt)
            elif type_ in ("double", "float"):
                val = float(txt)
            else:
                val = txt
            attrs[name] = val
        ntype = attrs.get("type", "")
        if ntype == "polymer":
            polymer_attrs[nid] = attrs
        elif ntype == "crosslinker":
            crosslink_attrs[nid] = attrs
        else:
            # Fall back on whether the node has COMX/Y/Z that aren't NaN
            # (crosslinks carry positions; chains do not).
            comx = attrs.get("COMX")
            if isinstance(comx, float) and not np.isnan(comx):
                crosslink_attrs[nid] = attrs
            else:
                polymer_attrs[nid] = attrs

    # Pull edges, split by edge_type. Chemical edges link a chain to a
    # crosslink; entanglement edges link chain to chain (replicated for
    # multiplicity).
    chemical_pairs: list[tuple[int, int]] = []
    entanglement_pairs: list[tuple[int, int]] = []
    for e in graph_el.findall("g:edge", ns):
        src = int(e.get("source"))
        tgt = int(e.get("target"))
        etype = "chemical"
        for d in e.findall("g:data", ns):
            kid = d.get("key")
            if kid in key_meta and key_meta[kid][1] == "edge_type":
                etype = d.text or "chemical"
        if etype == "chemical":
            chemical_pairs.append((src, tgt))
        elif etype == "entanglement":
            entanglement_pairs.append((src, tgt))

    G, cid_to_edge = _build_multigraph_from_dual(
        polymer_attrs, crosslink_attrs, chemical_pairs, entanglement_pairs
    )

    # Graph-level box bounds (xlo/xhi/ylo/yhi/zlo/zhi). All NaN by default
    # in the writer, so fall back to inferring from positions.
    box: dict[str, float] = {}
    for d in graph_el.findall("g:data", ns):
        kid = d.get("key")
        if kid in key_meta and key_meta[kid][0] == "graph":
            try:
                box[key_meta[kid][1]] = float(d.text)
            except (TypeError, ValueError):
                pass
    dims: Optional[np.ndarray] = None
    needed = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"}
    if needed <= set(box) and not any(np.isnan(box[k]) for k in needed):
        dims = np.array(
            [box["xhi"] - box["xlo"], box["yhi"] - box["ylo"], box["zhi"] - box["zlo"]],
            dtype=float,
        )
    else:
        dims = infer_dims_from_graph(G)

    n_ent_pairs = sum(
        1 for _, _, _, data in G.edges(keys=True, data=True)
        if data.get("entangled_with")
    ) // 2  # symmetric
    print(f"Loaded graphml from {path.name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
          f"Entangled pairs: {n_ent_pairs}")
    return G, dims


def _sc_positions_from_ids(
    xl_ids: list[int],
    box: np.ndarray,
    node_features: np.ndarray,
    node_ids: np.ndarray,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    n_polymer: int,
) -> Optional[dict[int, tuple[float, float, float]]]:
    """Reconstruct simple-cubic lattice coords from node ids + box.

    The C generator lays SC sites out as ``id = x + y*Nx + z*Nx*Ny`` with
    ``coords = (x, y, z)``; ``write_npz`` stores ``box = [0, Nx, 0, Ny,
    0, Nz]``. Inverting is therefore exact -- but only if the graph
    really came from an SC lattice, so we verify: every chain must join
    two sites that are nearest neighbours under periodic boundaries.

    Returns ``{node_id: (x, y, z)}``, or ``None`` when the box is
    unusable, an id falls outside the lattice, or the neighbour check
    fails (e.g. a BCC/FCC/Diamond graph, or a non-lattice topology).
    """
    if box is None or box.size != 6 or bool(np.isnan(box).any()):
        return None
    Nx, Ny, Nz = int(round(float(box[1]))), int(round(float(box[3]))), int(round(float(box[5])))
    if min(Nx, Ny, Nz) <= 0:
        return None
    n_sites = Nx * Ny * Nz
    if not xl_ids or max(xl_ids) >= n_sites or min(xl_ids) < 0:
        return None

    pos = {
        i: (float(i % Nx), float((i // Nx) % Ny), float(i // (Nx * Ny)))
        for i in xl_ids
    }

    # Validate: each chain's two junctions must be lattice neighbours.
    from collections import defaultdict
    chain_to_xl: dict[int, set[int]] = defaultdict(set)
    chem = edge_index[:, edge_type == 0]
    for k in range(chem.shape[1]):
        a, b = int(chem[0, k]), int(chem[1, k])
        if a < n_polymer <= b:
            chain_to_xl[a].add(int(node_ids[b]))
        elif b < n_polymer <= a:
            chain_to_xl[b].add(int(node_ids[a]))
    checked = 0
    for xls in chain_to_xl.values():
        if len(xls) != 2:
            continue
        p, q = (pos[i] for i in xls)
        dims_ = (Nx, Ny, Nz)
        dist = 0.0
        for ax in range(3):
            d = abs(p[ax] - q[ax])
            dist += min(d, dims_[ax] - d)
        if abs(dist - 1.0) > 1e-6:
            return None                      # not an SC nearest-neighbour graph
        checked += 1
    if checked == 0:
        return None
    return pos


#: Node-feature columns of a schema v1 NPZ, the layout of the downstream
#: GNN pipeline's bond/create datasets (``N_20_equil_...npz``). Read off the
#: data rather than the May spec, which names columns 1 and 2 ``length``
#: and ``contour_length``: column 1 is 18 for a DP-20 strand (the interior
#: beads, the two chain ends not counted), and column 2 averages 35.5
#: against 6 Rg^2 = 36.0 on the N20 file, the mean square end-to-end
#: distance, where a contour length would be about 18 for every strand.
NPZ_V1_COLUMNS = ("type", "n_interior", "ree2", "rg", "COMX", "COMY", "COMZ",
                  "chem_degree")


def npz_schema_version(arrays) -> int:
    """The schema of an NPZ dual graph: 1 or 2.

    Files topon writes carry ``schema_version`` (2). The collaborator's
    bond/create datasets carry none and have 8 feature columns, which is
    schema 1; a file with no stamp and 10 columns is taken as 2.

    Raises:
        ValueError: a stamp this reader does not know, or a feature width
            that is neither schema.
    """
    from topon.writers.npz_writer import _FEATURE_COLUMNS, SCHEMA_VERSION

    width = int(np.shape(arrays["node_features"])[1]) if np.ndim(
        arrays["node_features"]) == 2 else -1
    if "schema_version" in arrays:
        version = int(np.asarray(arrays["schema_version"]))
        expected = {1: len(NPZ_V1_COLUMNS), SCHEMA_VERSION: len(_FEATURE_COLUMNS)}
        if version not in expected:
            raise ValueError(
                f"NPZ schema_version {version} is not one this reader knows "
                f"({', '.join(str(v) for v in sorted(expected))}); it may be "
                f"from a newer topon.")
        if width != expected[version]:
            raise ValueError(
                f"NPZ says schema_version {version} but node_features has "
                f"{width} columns, not {expected[version]}.")
        return version
    if width == len(NPZ_V1_COLUMNS):
        return 1
    if width == len(_FEATURE_COLUMNS):
        return SCHEMA_VERSION
    raise ValueError(
        f"NPZ node_features has {width} columns and no schema_version; "
        f"schema 1 has {len(NPZ_V1_COLUMNS)} and schema 2 has "
        f"{len(_FEATURE_COLUMNS)}.")


def upgrade_npz_v1(arrays) -> dict:
    """A schema v1 NPZ (as a dict of arrays) in the schema v2 layout.

    Column by column (v1 -> v2):

    - ``type``, ``rg`` and ``COMX/Y/Z`` are copied.
    - ``n_interior`` -> ``length``: plus 2 on the chain rows, since v2's
      ``length`` is the DP, which counts the two chain ends (a DP-20 strand
      reads 18 in v1); crosslinker rows stay 1.
    - ``ree2`` has no v2 column. It is kept as a separate ``ree2`` array
      (per row, as v1 wrote it), and v2's ``contour_length`` and
      ``frac_ext`` are NaN, the value the v2 writer uses for a measurement
      it does not have.
    - ``chem_degree`` is copied; ``phys_degree`` is counted from the
      ``edge_type`` 1 edges, which in these datasets are the Z1+ ``-SP+``
      partner pairs (entanglements), stored in both directions.

    ``n_polymer`` and ``n_crosslinker`` are recounted from the ``type``
    column: the v1 files carry other numbers there (18 and 0 in the N20
    file, 98 and 0 in N100), so a reader that believed them rebuilt the
    first 18 rows of 7,500. Rows are put in polymer-then-crosslinker order
    if they are not already, with ``node_ids`` and ``edge_index`` moved
    with them. ``strain``, ``stress`` and ``box`` pass through. The result
    carries ``schema_version`` 2 and ``upgraded_from`` 1.
    """
    from topon.writers.npz_writer import _FEATURE_COLUMNS, SCHEMA_VERSION

    v1 = np.asarray(arrays["node_features"], dtype=np.float32)
    if v1.ndim != 2 or v1.shape[1] != len(NPZ_V1_COLUMNS):
        raise ValueError(
            f"a schema v1 node_features has {len(NPZ_V1_COLUMNS)} columns, "
            f"got shape {v1.shape}")
    col = {name: i for i, name in enumerate(NPZ_V1_COLUMNS)}
    node_ids = np.asarray(arrays["node_ids"])
    edge_index = np.asarray(arrays["edge_index"])
    edge_type = np.asarray(arrays["edge_type"])

    kind = v1[:, col["type"]]
    order = np.argsort(kind, kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        v1 = v1[order]
        node_ids = node_ids[order]
        new_row = np.empty_like(order)
        new_row[order] = np.arange(len(order))
        edge_index = new_row[edge_index]
        kind = v1[:, col["type"]]
    chain = kind == 0

    n = len(v1)
    v2 = np.full((n, len(_FEATURE_COLUMNS)), np.nan, dtype=np.float32)
    at = {name: i for i, name in enumerate(_FEATURE_COLUMNS)}
    v2[:, at["type"]] = kind
    v2[:, at["length"]] = np.where(chain, v1[:, col["n_interior"]] + 2.0, 1.0)
    for name in ("rg", "COMX", "COMY", "COMZ", "chem_degree"):
        v2[:, at[name]] = v1[:, col[name]]
    ent = edge_index[:, edge_type == 1]
    v2[:, at["phys_degree"]] = np.bincount(ent[0], minlength=n)[:n] if ent.size else 0.0

    out = {k: np.asarray(v) for k, v in arrays.items()}
    out.update(
        node_features=v2,
        node_ids=node_ids.astype(np.int32),
        edge_index=edge_index.astype(np.int32),
        edge_type=edge_type.astype(np.int32),
        n_polymer=np.int32(int(chain.sum())),
        n_crosslinker=np.int32(int((kind == 1).sum())),
        ree2=v1[:, col["ree2"]].copy(),
        schema_version=np.int32(SCHEMA_VERSION),
        upgraded_from=np.int32(1),
    )
    return out


def read_npz(path: Union[str, Path]) -> dict:
    """The arrays of an NPZ dual graph in the current schema.

    Checks ``schema_version`` (:func:`npz_schema_version`) and upgrades a
    schema 1 file (:func:`upgrade_npz_v1`), so a caller always sees the v2
    columns. For GNN pipelines that want the arrays rather than a graph.
    """
    with np.load(Path(path)) as data:
        arrays = {k: data[k] for k in data.files}
    if npz_schema_version(arrays) == 1:
        arrays = upgrade_npz_v1(arrays)
    return arrays


def load_npz(
    path: Union[str, Path],
) -> tuple[nx.MultiGraph, Optional[np.ndarray]]:
    """Load a topon npz dual graph back into a MultiGraph + dims.

    Reverses :func:`topon.writers.npz_writer.write_npz`. A schema v1 file
    is upgraded first (:func:`read_npz`); its dangling and sol chains have
    fewer than two crosslinks and are counted and skipped, since the dual
    graph holds no node for a free chain end.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        ``(G, dims)`` -- same shape as :func:`load_graphml`.
    """
    path = Path(path)
    data = read_npz(path)
    node_features = data["node_features"]
    node_ids = data["node_ids"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    box = data["box"]
    n_polymer = int(data["n_polymer"])
    n_crosslinker = int(data["n_crosslinker"])

    # Feature columns (must match npz_writer._FEATURE_COLUMNS; a v1 file
    # has been upgraded to these by read_npz):
    #   v2: [type, length, contour_length, rg, COMX, COMY, COMZ,
    #        chem_degree, phys_degree, frac_ext]
    # In v2 the COM columns are deliberately NaN (they are conformation
    # outputs, filled in after a LAMMPS run -- not known at generation
    # time). The downstream chemistry/conformation stages nevertheless
    # need a 3-D embedding of the junctions, so when COM is NaN we
    # reconstruct the ORIGINAL LATTICE COORDINATES from node_ids + box.
    #
    # The C generator numbers simple-cubic sites as
    #     id = x + y*Nx + z*Nx*Ny        coords = (x, y, z)
    # and write_npz stores box = [0, Nx, 0, Ny, 0, Nz], so the mapping
    # inverts exactly. ``_sc_positions_from_ids`` validates the result
    # (every chain must join two lattice neighbours) and returns None if
    # the graph is not an SC lattice, in which case COM/NaN is kept.
    polymer_attrs: dict[int, dict] = {}
    crosslink_attrs: dict[int, dict] = {}

    for i in range(n_polymer):
        nid = int(node_ids[i])
        polymer_attrs[nid] = {"length": int(node_features[i, 1])}

    xl_ids = [int(node_ids[i]) for i in range(n_polymer, n_polymer + n_crosslinker)]
    com_is_nan = (
        n_crosslinker > 0
        and bool(np.isnan(node_features[n_polymer:, 4:7]).all())
    )
    recovered = (
        _sc_positions_from_ids(xl_ids, box, node_features, node_ids,
                               edge_index, edge_type, n_polymer)
        if com_is_nan else None
    )
    if com_is_nan and recovered is None:
        print("  WARNING: COM columns are NaN and lattice positions could "
              "not be reconstructed; junction coordinates will be NaN and "
              "any downstream conformation/LAMMPS build will be invalid.")

    for i in range(n_polymer, n_polymer + n_crosslinker):
        nid = int(node_ids[i])
        if recovered is not None:
            cx, cy, cz = recovered[nid]
        else:
            cx = float(node_features[i, 4])
            cy = float(node_features[i, 5])
            cz = float(node_features[i, 6])
        crosslink_attrs[nid] = {"COMX": cx, "COMY": cy, "COMZ": cz}

    # Edges are stored bi-directionally; reduce to unordered pairs.
    #
    # edge_index holds 0-based ROW POSITIONS into node_features (PyG
    # convention -- see the npz_writer remap bugfix). The rest of this
    # loader keys nodes by their original IDs, so map each edge endpoint
    # back through node_ids: node_ids[row_position] -> original ID.
    chemical_pairs_set: set[tuple[int, int]] = set()
    entanglement_pairs_list: list[tuple[int, int]] = []
    n_edges = edge_index.shape[1]
    for k in range(n_edges):
        src = int(node_ids[int(edge_index[0, k])])
        tgt = int(node_ids[int(edge_index[1, k])])
        et = int(edge_type[k])
        if et == 0:                              # chemical (bidirectional copy)
            chemical_pairs_set.add(tuple(sorted((src, tgt))))
        elif et == 1 and src < tgt:              # entanglement: keep one dir
            entanglement_pairs_list.append((src, tgt))

    G, cid_to_edge = _build_multigraph_from_dual(
        polymer_attrs,
        crosslink_attrs,
        list(chemical_pairs_set),
        entanglement_pairs_list,
    )

    # Box: [xlo, xhi, ylo, yhi, zlo, zhi]
    if box.size == 6 and not np.isnan(box[1]):
        dims = np.array(
            [box[1] - box[0], box[3] - box[2], box[5] - box[4]], dtype=float
        )
    else:
        dims = infer_dims_from_graph(G)

    n_ent_pairs = sum(
        1 for _, _, _, data in G.edges(keys=True, data=True)
        if data.get("entangled_with")
    ) // 2
    print(f"Loaded npz from {path.name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
          f"Entangled pairs: {n_ent_pairs}")
    return G, dims


def _build_multigraph_from_dual(
    polymer_attrs: dict[int, dict],
    crosslink_attrs: dict[int, dict],
    chemical_pairs: list[tuple[int, int]],
    entanglement_pairs: list[tuple[int, int]],
) -> tuple[nx.MultiGraph, dict[int, tuple[int, int, int]]]:
    """Shared dual-graph -> MultiGraph reconstruction (graphml & npz).

    * Each polymer (chain) dual-node becomes an edge in the topon graph;
      its two chemical-edge neighbours are the chain's u/v crosslinks.
    * Crosslink positions come from the COMX/Y/Z fields.
    * Entanglement multiplicity is recovered by counting how many times
      a chain-chain pair appears among ``entanglement_pairs``.

    Returns the rebuilt graph plus a ``{chain_dual_id: (u, v, key)}`` map
    so the caller can wire `entangled_with` after edges are added.
    """
    G = nx.MultiGraph()
    for xid, attrs in crosslink_attrs.items():
        pos = (
            float(attrs.get("COMX", 0.0)),
            float(attrs.get("COMY", 0.0)),
            float(attrs.get("COMZ", 0.0)),
        )
        G.add_node(xid, pos=pos)

    # For each chain, find its two crosslink endpoints from chemical pairs.
    chain_endpoints: dict[int, list[int]] = {}
    for a, b in chemical_pairs:
        chain, junction = (a, b) if a in polymer_attrs else (b, a)
        if chain not in polymer_attrs:
            # Neither end is a known polymer dual-node; skip
            continue
        chain_endpoints.setdefault(chain, []).append(junction)

    # Add edges in chain-id sorted order so the resulting (u, v, key)
    # assignment is deterministic across graphml and npz loads of the
    # same network.
    cid_to_edge: dict[int, tuple[int, int, int]] = {}
    # A chain with one crosslink (dangling) or none (sol) has no second
    # junction to be an edge to; topon's own writers never produce one,
    # but the bond/create datasets carry hundreds, so they are counted and
    # reported once rather than one line each.
    skipped = Counter(len(chain_endpoints.get(cid, ())) for cid in polymer_attrs
                      if len(chain_endpoints.get(cid, ())) != 2)
    for cid in sorted(chain_endpoints):
        endpoints = chain_endpoints[cid]
        if len(endpoints) != 2:
            continue
        # Canonicalise the (u, v) order so the same chain always yields
        # the same edge key regardless of how chemical_pairs were ordered
        # in the source file.
        u, v = sorted(endpoints)
        dp = int(polymer_attrs[cid].get("length", 1))
        key = G.add_edge(u, v, dp=dp)
        cid_to_edge[cid] = (u, v, key)
    if skipped:
        parts = ", ".join(f"{n} with {k} crosslink{'s' if k != 1 else ''}"
                          for k, n in sorted(skipped.items()))
        print(f"  [warn] skipped {sum(skipped.values())} chains that do not "
              f"join two crosslinks ({parts}); the graph has no node for a "
              f"free chain end")

    # Recover entanglement multiplicities by counting chain pairs.
    ent_counts: Counter = Counter()
    for cid1, cid2 in entanglement_pairs:
        pair = tuple(sorted((cid1, cid2)))
        ent_counts[pair] += 1

    for (cid1, cid2), count in ent_counts.items():
        if cid1 not in cid_to_edge or cid2 not in cid_to_edge:
            continue
        e1 = cid_to_edge[cid1]
        e2 = cid_to_edge[cid2]
        G[e1[0]][e1[1]][e1[2]]["entangled_with"] = e2
        G[e1[0]][e1[1]][e1[2]]["entanglement_count"] = count
        G[e2[0]][e2[1]][e2[2]]["entangled_with"] = e1
        G[e2[0]][e2[1]][e2[2]]["entanglement_count"] = count

    return G, cid_to_edge
