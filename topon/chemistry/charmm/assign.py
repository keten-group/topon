"""CHARMM atom typing of a polymer network from RTF residues.

The chemistry stage builds the network as one RDKit molecule: node molecules,
chains of repeat units (the monomer SMILES concatenated ``dp`` times) and,
with ``auto_bridge``, bridge atoms between them. This module cuts that
molecule into residue instances (each node molecule, each repeat unit, each
bridge atom, with their hydrogens), matches every instance to the RTF
residue its config names, and returns each atom's CHARMM type, charge and
atom name, plus the impropers the RTF lists.

Matching. With ``charmm_atom_names`` the heavy atoms are named in SMILES
order and checked against the RTF. Without it, the instance's heavy-atom
graph is matched to the residue's by graph isomorphism (element, hydrogen
count and, for residues that link through ``+``/``-`` atoms, the number of
bonds leaving the residue). Hydrogens follow their heavy atom. If two
isomorphisms would give an atom a different type or charge the match is
ambiguous and topon asks for the names instead of choosing. Instances with
the same graph reuse one match, so a chain of thousands of identical units
is matched once.

CGenFF typing (a new molecule without an RTF) needs the licensed CGenFF
program; run it and pass its stream file. topon does not guess types.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from topon.forcefield.charmm import CharmmParameterSet


class CharmmTypingError(ValueError):
    """An atom that no configured RTF residue accounts for, or an ambiguous one."""


@dataclass
class ResidueInstance:
    resname: str
    heavy: list[int]
    atoms: list[int] = field(default_factory=list)
    kind: str = "monomer"            # monomer | node | bridge
    chain: tuple | None = None
    unit: int | None = None


@dataclass
class CharmmTyping:
    """Per-atom CHARMM data for an RDKit molecule (0-based atom indices)."""
    atom_type: dict[int, str]
    charge: dict[int, float]
    atom_name: dict[int, str]
    residue_of: dict[int, int]
    residues: list[ResidueInstance]
    impropers: list[tuple[int, int, int, int]]
    n_graph_matches: int = 0

    def residue_charges(self) -> dict[str, list[float]]:
        """Net charge of every residue instance, grouped by residue name."""
        out: dict[str, list[float]] = defaultdict(list)
        for inst in self.residues:
            out[inst.resname].append(sum(self.charge[a] for a in inst.atoms))
        return dict(out)


def bundled_data_dir() -> Path:
    return Path(str(resources.files("topon.chemistry.charmm") / "data"))


def resolve_files(files: list[str], base: Path | None = None) -> list[Path]:
    """Paths of the CHARMM files a config names.

    ``bundled:NAME`` is ``topon/chemistry/charmm/data/NAME``; a relative path
    is taken from ``base`` (the config's folder) when given.
    """
    out = []
    for f in files:
        if f.startswith("bundled:"):
            p = bundled_data_dir() / f.split(":", 1)[1]
        else:
            p = Path(f)
            if not p.is_absolute() and base is not None:
                p = base / p
        if not p.exists():
            raise FileNotFoundError(f"CHARMM file not found: {f} ({p})")
        out.append(p)
    return out


def load_parameters(files: list[str], base: Path | None = None) -> CharmmParameterSet:
    paths = resolve_files(files, base)
    if not paths:
        raise CharmmTypingError("force_field 'charmm' needs chemistry.charmm.files "
                                "(RTF and PRM, or a stream file)")
    return CharmmParameterSet.from_files(*paths)


# ── residue templates ────────────────────────────────────────────────────────

@dataclass
class _Template:
    name: str
    heavy: list[str]
    label: dict[str, tuple]              # heavy name -> (element, nH, n_external|None)
    hydrogens: dict[str, list[str]]      # heavy name -> [H names] (RTF order)
    graph: nx.Graph
    atoms: dict[str, tuple[str, float]]
    impropers: list[tuple[str, str, str, str]]
    charge: float                        # the RESI line's total charge


def _template(ps: CharmmParameterSet, resname: str) -> _Template:
    res = ps.residues.get(resname.upper())
    if res is None:
        raise CharmmTypingError(f"RTF residue {resname!r} is not in the CHARMM files "
                                f"(known: {sorted(ps.residues)[:30]} ...)")
    elem = {}
    for name, (atype, _q) in res.atoms.items():
        e = ps.element(atype)
        if e is None:
            raise CharmmTypingError(f"residue {resname}: no mass/element for type {atype}")
        elem[name] = e
    internal = [(a, b) for a, b in res.bonds if a in res.atoms and b in res.atoms]
    ext = res.external_atoms()
    linking = any(a[:1] in "+-" or b[:1] in "+-" for a, b in res.bonds)
    heavy = [n for n in res.atoms if elem[n] != "H"]
    hyd: dict[str, list[str]] = {n: [] for n in heavy}
    g = nx.Graph()
    g.add_nodes_from(heavy)
    for a, b in internal:
        if elem[a] == "H" and elem[b] != "H":
            hyd[b].append(a)
        elif elem[b] == "H" and elem[a] != "H":
            hyd[a].append(b)
        elif elem[a] != "H" and elem[b] != "H":
            g.add_edge(a, b)
    label = {n: (elem[n], len(hyd[n]), ext.get(n, 0) if linking else None) for n in heavy}
    for n in heavy:
        g.nodes[n]["label"] = label[n]
    # What this route cannot write is refused rather than dropped: explicit
    # angle/dihedral lists (NOANG/NODIH) and CMAP cross terms.
    if res.no_angles or res.no_dihedrals:
        raise CharmmTypingError(f"residue {resname}: NOANG/NODIH residues (explicit angle "
                                f"or dihedral lists) are not supported for polymer networks")
    if res.cmaps:
        raise CharmmTypingError(f"residue {resname}: {len(res.cmaps)} CMAP cross term(s); "
                                f"the polymer route writes no CMAP")
    return _Template(resname.upper(), heavy, label, hyd, g, dict(res.atoms),
                     list(res.impropers), res.charge)


# ── typing ───────────────────────────────────────────────────────────────────

def _node_config(chem, graph, node):
    attrs = graph.nodes[node]
    ntype = attrs.get("node_type", attrs.get("type", "A"))
    cfg = chem.node_type_map.get(ntype)
    if cfg is None or not cfg.charmm_residue:
        raise CharmmTypingError(
            f"node type {ntype!r} has no charmm_residue in chemistry.node_type_map")
    return cfg


def _monomer_config(chem, data):
    etype = data.get("edge_type", "A")
    ecfg = chem.edge_type_map.get(etype)
    if ecfg is None or ecfg.monomer not in chem.monomers:
        raise CharmmTypingError(f"edge type {etype!r} has no monomer in the config")
    mcfg = chem.monomers[ecfg.monomer]
    if not mcfg.charmm_residue:
        raise CharmmTypingError(f"monomer {ecfg.monomer!r} has no charmm_residue")
    return mcfg


def _unit_size(smiles: str) -> int:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise CharmmTypingError(f"cannot parse monomer SMILES {smiles!r}")
    return Chem.RemoveHs(m).GetNumAtoms()


def type_network(mol_h, builder, chem, ps: CharmmParameterSet) -> CharmmTyping:
    """Type every atom of ``mol_h`` (the network with hydrogens) from the RTF.

    ``builder`` is the :class:`topon.chemistry.builder.ChemistryBuilder` that
    built the heavy-atom network (its node, chain and bridge bookkeeping is
    what cuts the molecule into residues), ``chem`` the chemistry config.
    """
    graph = builder.G
    if getattr(builder, "graft_atom_map", None):
        raise CharmmTypingError("grafted chains are not supported with force_field 'charmm'")
    if getattr(builder, "sol_atom_map", None):
        raise CharmmTypingError("sol chains are not supported with force_field 'charmm'")

    instances: list[ResidueInstance] = []
    names: dict[tuple, list[str] | None] = {}
    for node, idxs in builder.node_atoms.items():
        cfg = _node_config(chem, graph, node)
        instances.append(ResidueInstance(cfg.charmm_residue.upper(), list(idxs), kind="node"))
        names[("node", len(instances) - 1)] = cfg.charmm_atom_names
    missing_nodes = set(builder.node_map) - set(builder.node_atoms)
    if missing_nodes:
        raise CharmmTypingError(f"{len(missing_nodes)} node(s) are POSS cages or other "
                                f"structures CHARMM typing does not cover")
    for edge, heavy in builder.edge_atom_map.items():
        u, v, key = edge
        data = graph[u][v][key]
        mcfg = _monomer_config(chem, data)
        n = _unit_size(mcfg.smiles)
        if len(heavy) % n:
            raise CharmmTypingError(f"chain {edge}: {len(heavy)} atoms is not a whole "
                                    f"number of {n}-atom repeat units")
        for k in range(len(heavy) // n):
            instances.append(ResidueInstance(mcfg.charmm_residue.upper(),
                                             list(heavy[k * n:(k + 1) * n]),
                                             kind="monomer", chain=edge, unit=k))
            names[("monomer", len(instances) - 1)] = mcfg.charmm_atom_names
    if builder.bridge_atoms:
        bres = chem.charmm.bridge_residue if chem.charmm else None
        if not bres:
            raise CharmmTypingError(
                f"{len(builder.bridge_atoms)} auto_bridge atoms need "
                f"chemistry.charmm.bridge_residue (or set connection.auto_bridge false)")
        for b in builder.bridge_atoms:
            instances.append(ResidueInstance(bres.upper(), [b], kind="bridge"))

    residue_of: dict[int, int] = {}
    for i, inst in enumerate(instances):
        for a in inst.heavy:
            if a in residue_of:
                raise CharmmTypingError(f"atom {a} is in two residues")
            residue_of[a] = i
    heavy_all = [a.GetIdx() for a in mol_h.GetAtoms() if a.GetAtomicNum() > 1]
    loose = [a for a in heavy_all if a not in residue_of]
    if loose:
        sym = sorted({mol_h.GetAtomWithIdx(a).GetSymbol() for a in loose})
        raise CharmmTypingError(f"{len(loose)} heavy atoms ({sym}) belong to no residue")
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            nb = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(nb) != 1 or nb[0] not in residue_of:
                raise CharmmTypingError(f"hydrogen {atom.GetIdx()} is not on a typed atom")
            residue_of[atom.GetIdx()] = residue_of[nb[0]]
    members: dict[int, list[int]] = defaultdict(list)
    for a, r in residue_of.items():
        members[r].append(a)
    for i, inst in enumerate(instances):
        inst.atoms = sorted(members[i])

    templates: dict[str, _Template] = {}
    cache: dict[tuple, list[str]] = {}
    atype, charge, aname = {}, {}, {}
    n_graph = 0
    for i, inst in enumerate(instances):
        t = templates.get(inst.resname) or templates.setdefault(inst.resname,
                                                                _template(ps, inst.resname))
        heavy_names = names.get((inst.kind, i))
        local = _instance_labels(mol_h, inst, residue_of, i, linking=t.label and
                                 next(iter(t.label.values()))[2] is not None)
        if heavy_names:
            mapping = _map_by_names(t, inst, local, heavy_names, mol_h)
        else:
            key = (inst.resname, tuple(local[a] for a in inst.heavy),
                   tuple(sorted(_local_edges(mol_h, inst))))
            if key not in cache:
                cache[key] = _map_by_graph(t, inst, local, mol_h)
                n_graph += 1
            mapping = dict(zip(inst.heavy, cache[key]))
        for a, nm in mapping.items():
            _assign(t, a, nm, atype, charge, aname)
            hs = sorted(n.GetIdx() for n in mol_h.GetAtomWithIdx(a).GetNeighbors()
                        if n.GetAtomicNum() == 1)
            rtf_h = t.hydrogens[nm]
            if len(hs) != len(rtf_h):
                raise CharmmTypingError(f"{inst.resname} {nm}: {len(hs)} H in the network, "
                                        f"{len(rtf_h)} in the RTF")
            if len({t.atoms[h] for h in rtf_h}) > 1:
                raise CharmmTypingError(f"{inst.resname} {nm}: its hydrogens differ in type "
                                        f"or charge, so they cannot be told apart")
            for h, hn in zip(hs, rtf_h):
                _assign(t, h, hn, atype, charge, aname)

    # Every residue carries its RTF charge, and the network an integer one.
    for inst in instances:
        q = sum(charge[a] for a in inst.atoms)
        want = templates[inst.resname].charge
        if abs(q - want) > 1e-6:
            raise CharmmTypingError(f"{inst.resname}: its atoms sum to {q:+.4f} e, the RTF "
                                    f"residue declares {want:+.4f} e")
    total = sum(charge.values())
    if abs(total - round(total)) > 1e-4:
        raise CharmmTypingError(f"the network's net charge is {total:+.4f} e, not an "
                                f"integer; check the residue charges in the RTF")

    impropers = _impropers(instances, templates, aname, mol_h, residue_of)
    return CharmmTyping(atype, charge, aname, residue_of, instances, impropers, n_graph)


def _assign(t, atom, name, atype, charge, aname):
    ty, q = t.atoms[name]
    atype[atom], charge[atom], aname[atom] = ty, q, name


def _instance_labels(mol_h, inst, residue_of, i, linking):
    out = {}
    for a in inst.heavy:
        at = mol_h.GetAtomWithIdx(a)
        n_h = sum(1 for n in at.GetNeighbors() if n.GetAtomicNum() == 1)
        n_ext = sum(1 for n in at.GetNeighbors()
                    if n.GetAtomicNum() > 1 and residue_of.get(n.GetIdx()) != i)
        out[a] = (at.GetSymbol().upper(), n_h, n_ext if linking else None)
    return out


def _local_edges(mol_h, inst):
    pos = {a: k for k, a in enumerate(inst.heavy)}
    for b in mol_h.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            yield tuple(sorted((pos[i], pos[j])))


def _map_by_names(t, inst, local, heavy_names, mol_h):
    names = [n.upper() for n in heavy_names]
    if len(names) != len(inst.heavy):
        raise CharmmTypingError(f"{inst.resname}: {len(names)} charmm_atom_names for "
                                f"{len(inst.heavy)} heavy atoms")
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise CharmmTypingError(f"{inst.resname}: charmm_atom_names repeats {', '.join(dup)}")
    mapping = dict(zip(inst.heavy, names))
    for a, nm in mapping.items():
        if nm not in t.label:
            raise CharmmTypingError(f"{inst.resname}: no heavy atom {nm} in the RTF")
        want, got = t.label[nm], local[a]
        if want[:2] != got[:2] or (want[2] is not None and want[2] != got[2]):
            raise CharmmTypingError(f"{inst.resname} {nm}: RTF has {want}, the network "
                                    f"atom has {got} (element, H count, outside bonds)")
    for i, j in (tuple(sorted(e)) for e in _local_edges(mol_h, inst)):
        if not t.graph.has_edge(names[i], names[j]):
            raise CharmmTypingError(f"{inst.resname}: bond {names[i]}-{names[j]} is not in the RTF")
    return mapping


def _map_by_graph(t, inst, local, mol_h, limit=500):
    g = nx.Graph()
    for k, a in enumerate(inst.heavy):
        g.add_node(k, label=local[a])
    g.add_edges_from(_local_edges(mol_h, inst))
    if g.number_of_nodes() != t.graph.number_of_nodes() or \
            g.number_of_edges() != t.graph.number_of_edges():
        raise CharmmTypingError(
            f"{inst.resname}: the network unit has {g.number_of_nodes()} heavy atoms and "
            f"{g.number_of_edges()} bonds, the RTF residue {t.graph.number_of_nodes()} and "
            f"{t.graph.number_of_edges()}")
    gm = GraphMatcher(g, t.graph, node_match=lambda x, y: x["label"] == y["label"])
    first, seen = None, defaultdict(set)
    for n, iso in enumerate(gm.isomorphisms_iter()):
        if first is None:
            first = iso
        for k, nm in iso.items():
            seen[k].add((t.atoms[nm], tuple(sorted(t.atoms[h] for h in t.hydrogens[nm]))))
        if n + 1 >= limit:
            break
    if first is None:
        labels = [local[a] for a in inst.heavy]
        raise CharmmTypingError(
            f"{inst.resname}: the network unit does not match the RTF residue "
            f"(unit labels {labels}; RTF {t.label}). Check the SMILES, the residue name "
            f"and the atoms bonded outside the unit.")
    ambiguous = [k for k, v in seen.items() if len(v) > 1]
    if ambiguous:
        raise CharmmTypingError(
            f"{inst.resname}: atoms can be matched to the RTF in ways that give different "
            f"types or charges; give charmm_atom_names for this residue")
    return [first[k] for k in range(len(inst.heavy))]


def _impropers(instances, templates, aname, mol_h, residue_of):
    """RTF impropers of every residue instance as atom quads.

    A ``+``/``-`` name is looked up in the next/previous unit of the strand.
    At a strand end that neighbour is the junction (or bridge) residue the
    unit is bonded to, and the name must exist there. An improper that
    cannot be placed stops the build, since dropping it would leave the
    network without a term its RTF defines.
    """
    by_chain_unit = {}
    for i, inst in enumerate(instances):
        if inst.kind == "monomer":
            by_chain_unit[(inst.chain, inst.unit)] = i
    name_to_atom = defaultdict(dict)
    for i, inst in enumerate(instances):
        for a in inst.atoms:
            name_to_atom[i][aname[a]] = a

    def bonded_residues(i):
        out = set()
        for a in instances[i].heavy:
            for n in mol_h.GetAtomWithIdx(a).GetNeighbors():
                j = residue_of.get(n.GetIdx())
                if j is not None and j != i:
                    out.add(j)
        return out

    def neighbour(i, sign, nm):
        inst = instances[i]
        if inst.kind != "monomer":
            raise CharmmTypingError(f"{inst.resname}: an improper names {sign}{nm}, but a "
                                    f"junction or bridge residue has no next or previous unit")
        j = by_chain_unit.get((inst.chain, inst.unit + (1 if sign == "+" else -1)))
        if j is not None:
            return j
        # strand end: the non-strand residue this unit is bonded to
        other = by_chain_unit.get((inst.chain, inst.unit + (-1 if sign == "+" else 1)))
        ends = [j for j in bonded_residues(i) if j != other
                and instances[j].kind != "monomer" and nm in name_to_atom[j]]
        if len(ends) != 1:
            raise CharmmTypingError(
                f"{inst.resname}: an improper names {sign}{nm}, and at the strand end "
                f"{'no' if not ends else 'more than one'} bonded junction residue has an "
                f"atom {nm}. Give the junction residue that atom name, or use an RTF "
                f"residue without inter-residue impropers.")
        return ends[0]

    out = []
    for i, inst in enumerate(instances):
        for quad in templates[inst.resname].impropers:
            ids = []
            for nm in quad:
                j = i
                if nm[:1] in "+-":
                    j = neighbour(i, nm[0], nm[1:])
                    nm = nm[1:]
                a = name_to_atom[j].get(nm)
                if a is None:
                    raise CharmmTypingError(f"{inst.resname}: improper atom {nm} is not in "
                                            f"residue {instances[j].resname}")
                ids.append(a)
            out.append(tuple(ids))
    return out
