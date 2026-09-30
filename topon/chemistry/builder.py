"""
Chemistry Builder for Topon.

Builds molecular structures based on assigned graph attributes.
Supports atomistic (DREIDING) and coarse-grained (Kremer-Grest) models.

Key features:
- Smart connection chemistry (auto-bridge atoms)
- Node type → molecule mapping
- Edge type → monomer mapping
- Copolymer sequences
- Grafted chains
"""

import collections
import functools
import warnings
from typing import Optional
import networkx as nx
import numpy as np

from topon.config.schema import ChemistryConfig


# Feature-detect marker for downstream projects.
# True since the fix that removed the trailing "[O]" placeholder from
# `_create_chain_from_smiles`, which produced a peroxide -O-O- bond at the
# chain tail for O-terminal monomers (e.g. PDMS, PTFPMS) under the atomistic
# auto-bridge path.
_PEROXIDE_FIX_APPLIED = True

#: Valence a bare junction atom of this element is filled to with methyls on
#: the atomistic route. A silicone junction bonds to oxygen and carbon only,
#: so a crosslinker site no strand reached is a methyl (MeSi(O-)3 for a
#: trifunctional junction), never the Si-H that AddHs would otherwise put
#: there. The workflow route (`topon.workflows.atomistic_network`) has always
#: capped this way; the pipeline did not before 0.4.0.
_METHYL_CAPPED = {"Si": 4}


class ChemistryBuilderError(ValueError):
    """The graph asks for a chemistry the configuration does not define."""


def _is_element(symbol: str) -> bool:
    """True for an element symbol ("Si", "C", "Co"), False for SMILES ("CC", "CO").

    A capital, then at most one lower-case letter, that RDKit's periodic
    table knows. The length alone cannot tell them apart: "CO" is carbon
    monoxide as SMILES and "Co" is cobalt.
    """
    return symbol in _element_symbols()


@functools.lru_cache(maxsize=1)
def _element_symbols() -> frozenset:
    # Asking RDKit for an unknown symbol prints a post-condition violation,
    # so the table is read once instead.
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    return frozenset(table.GetElementSymbol(z) for z in range(1, 119))


class ChemistryBuilder:
    """
    Builds molecular structure from attributed graph.
    
    Uses configuration to map:
    - Node types → crosslinker molecules (Si, POSS, custom)
    - Edge types → chain monomers (PDMS, FPDMS, etc.)
    - Auto-detects when bridge atoms are needed
    """
    
    def __init__(
        self, 
        G: nx.MultiGraph, 
        dims: Optional[np.ndarray],
        config: ChemistryConfig
    ):
        """
        Initialize the chemistry builder.
        
        Args:
            G: Attributed graph from assignment module.
            dims: Box dimensions.
            config: Chemistry configuration.
        """
        self.G = G
        self.dims = dims
        self.config = config
        
        # RDKit molecule (lazy import to avoid hard dependency)
        self.chemical_space = None
        
        # Atom mappings
        self.node_map = {}  # node_id -> atom_idx or list of atom_idxs (for POSS)
        self.edge_atom_map = {}  # edge_id -> list of atom_idxs
        self.edge_backbone_map = {}  # edge_id -> (head_idx, tail_idx)
        self.graft_atom_map = {}  # edge_id -> {backbone_pos: [graft_atom_idxs]}

        # For POSS corner tracking
        self.poss_usage = {}  # node_id -> set of used corner indices

        # Entangled pairs
        self.entangled_pairs = []

        # Molecule bookkeeping for the end-linked writer convention:
        # every junction is its own molecule, every chain is one molecule,
        # and a dangling chain's free end site belongs to the chain.
        self.node_is_end_cap = {}   # node_id -> bool
        self.node_mol = {}          # node_id -> molecule id
        self.chain_mol = {}         # (u, v, key) -> molecule id
        self.sol_atom_map = []      # [[atom idxs], ...] for sol chains
        self._next_mol = 0

        # Every heavy atom a node molecule placed (node_map keeps only the
        # attachment atom), and the auto-bridge atoms. Read by the CHARMM
        # typing, which assigns each atom to one RTF residue.
        self.node_atoms = {}        # node_id -> [atom idxs]
        self.bridge_atoms = []      # [atom idx, ...]

        # Per strand, the atoms a strand-level reader needs: the junction
        # atom each end is bonded to, and the backbone from one to the other
        # (bridge atoms included, junction atoms not). `strand_table` turns
        # these into the run manifest's per-strand record.
        self.edge_ends = {}         # edge_id -> (atom at u, atom at v)
        self.edge_backbone_path = {}  # edge_id -> [atom idxs, u side first]
        self.edge_repeat_heads = {}   # edge_id -> [one atom idx per repeat]
        self.cap_atoms = {}         # node_id -> [methyl C idxs added as caps]

    def build(self):
        """
        Build the molecular structure.
        
        Returns:
            RDKit RWMol object with complete molecular structure.
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit is required for chemistry building. Install with: pip install rdkit")
        
        print("Building chemistry...")
        print(f"  Model type: {self.config.model_type}")
        
        # Initialize RWMol
        self.chemical_space = Chem.RWMol()
        
        # Get entangled pairs from graph
        self._extract_entangled_pairs()
        
        # Build nodes (crosslinkers)
        self._build_nodes()
        
        # Build chains (edges)
        self._build_chains()

        # Fill what the strands left of each bare junction's valence
        self._cap_junctions()

        # Report
        print(f"  Total atoms: {self.chemical_space.GetNumAtoms()}")
        print(f"  Total bonds: {self.chemical_space.GetNumBonds()}")
        
        return self.chemical_space
    
    def _extract_entangled_pairs(self):
        """Extract entangled edge pairs from graph attributes."""
        processed = set()
        for u, v, key, data in self.G.edges(keys=True, data=True):
            partner = data.get("entangled_with")
            if partner:
                edge = (u, v, key)
                pair = tuple(sorted([edge, partner]))
                if pair not in processed:
                    self.entangled_pairs.append(pair)
                    processed.add(pair)
        
        if self.entangled_pairs:
            print(f"  Entangled pairs: {len(self.entangled_pairs)}")
    
    def _build_nodes(self):
        """Build crosslinker structures for each node."""
        from rdkit import Chem
        
        print("  Building nodes...")
        if self.config.model_type == "atomistic":
            self._check_node_types()

        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree == 0:
                continue

            # Get node type from graph.  For backward compatibility with
            # legacy callers, accept the attribute name ``type`` as a fallback.  Emit a
            # DeprecationWarning so new code migrates to ``node_type``.
            node_attrs = self.G.nodes[node]
            if "node_type" in node_attrs:
                node_type = node_attrs["node_type"]
            elif "type" in node_attrs:
                import warnings
                warnings.warn(
                    "Graph node attribute 'type' is read for backward-compat; "
                    "please use 'node_type' in new code.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                node_type = node_attrs["type"]
            else:
                node_type = "A"

            # Get molecule config for this type
            node_config = self.config.node_type_map.get(node_type)
            if not node_config:
                # Default to Si
                molecule = "Si"
                is_end_cap = degree == 1
            else:
                molecule = node_config.molecule
                is_end_cap = node_config.is_end_cap
            
            # Build the appropriate structure
            if molecule.upper() == "POSS" or molecule.upper() == "SI8O12":
                self._place_poss_cage(node)
            elif molecule.upper() == "POSS_AM0270":
                self._place_poss_am0270(node)
            elif is_end_cap or degree == 1:
                self._place_end_cap(node, molecule)
            else:
                self._place_simple_atom(node, molecule)

            # A junction is its own molecule and carries the junction role;
            # an end cap is the free end of its one chain, so it waits for
            # that chain to claim it.
            #
            # Which one a node is follows its declared type when it has one:
            # a crosslinker that happens to carry a single strand is still a
            # crosslinker (the DP-20 reference has eight of them), and only
            # an undeclared node falls back to reading degree 1 as a free
            # end. The structure placed above is unaffected: in CG both
            # branches put one bead there.
            declared = next((node_attrs[k] for k in ("node_type", "type", "kind")
                             if k in node_attrs), None)
            if node_config is not None:
                self.node_is_end_cap[node] = bool(node_config.is_end_cap)
            elif declared is not None:
                self.node_is_end_cap[node] = declared in ("end", "END")
            else:
                self.node_is_end_cap[node] = degree == 1
            if not self.node_is_end_cap[node]:
                self._next_mol += 1
                self.node_mol[node] = self._next_mol
                self._tag_atoms(self.node_map[node], self._next_mol, "junction")

        print(f"    Placed {len(self.node_map)} node structures")

    def _check_node_types(self) -> None:
        """Refuse node types the configuration has no molecule for.

        An unmatched type used to fall through to a bare Si, which on
        the atomistic route puts silicon into a network that may have none,
        and a key typed ``"End"`` for ``"end"`` did the same without a word.
        A coarse-grained junction is one bead whatever its molecule, so only
        the atomistic route refuses. All unmatched types are named at once,
        with the keys the map does have.
        """
        missing = {}
        for node in self.G.nodes():
            if self.G.degree(node) == 0:
                continue
            attrs = self.G.nodes[node]
            node_type = attrs.get("node_type", attrs.get("type", "A"))
            if node_type not in self.config.node_type_map:
                missing[node_type] = missing.get(node_type, 0) + 1
        if missing:
            listed = ", ".join(f"{t!r} ({n} nodes)" for t, n in sorted(
                missing.items(), key=lambda kv: str(kv[0])))
            raise ChemistryBuilderError(
                f"chemistry.node_type_map has no molecule for node type(s) "
                f"{listed}; its keys are {sorted(self.config.node_type_map)}. "
                f"Add them to the map, or change assignment.node_types so the "
                f"graph uses the keys the map has.")

    def _tag_atoms(self, atom_ref, mol_id: int, role: str) -> None:
        """Record molecule and role on atoms, for the end-linked writer.

        Inert for the default writer convention, which puts every atom in
        molecule 1 and types beads by ``bead_type``.
        """
        idxs = atom_ref if isinstance(atom_ref, (list, tuple)) else [atom_ref]
        for idx in idxs:
            if idx is None:
                continue
            atom = self.chemical_space.GetAtomWithIdx(int(idx))
            atom.SetIntProp("topon_mol", int(mol_id))
            atom.SetProp("topon_role", role)
    
    def _place_simple_atom(self, node: int, atom_symbol: str = "Si"):
        """Place a simple atom crosslinker."""
        from rdkit import Chem
        
        # Handle SMILES vs atom symbol
        if _is_element(atom_symbol):
            atom = Chem.Atom(atom_symbol)
            idx = self.chemical_space.AddAtom(atom)
            self.node_map[node] = idx
            self.node_atoms[node] = [idx]
        else:
            # It's a SMILES string
            mol = Chem.MolFromSmiles(atom_symbol)
            if mol:
                idxs = [self.chemical_space.AddAtom(a) for a in mol.GetAtoms()]
                for b in mol.GetBonds():
                    self.chemical_space.AddBond(
                        idxs[b.GetBeginAtomIdx()],
                        idxs[b.GetEndAtomIdx()],
                        b.GetBondType()
                    )
                self.node_map[node] = idxs[0]  # Use first atom as attachment point
                self.node_atoms[node] = idxs
            else:
                if self.config.model_type == "atomistic":
                    # The same silent Si as an unmatched node type.
                    raise ChemistryBuilderError(
                        f"node molecule {atom_symbol!r} is neither an element "
                        f"symbol nor a SMILES RDKit can parse")
                # Fallback to Si
                idx = self.chemical_space.AddAtom(Chem.Atom("Si"))
                self.node_map[node] = idx
                self.node_atoms[node] = [idx]
    
    def _place_end_cap(self, node: int, molecule: str):
        """Place an end-cap molecule.

        Atomistic mode: instantiate the full SMILES (e.g. trimethylsilyl
        ``[Si](C)(C)C``). The atomistic Pipeline's pendant-coordinate pass
        propagates positions for the methyl Cs through bond neighbours, so
        leaving them out of ``node_map`` is fine.

        Coarse-grained mode: collapse to a single bead. The CG Pipeline
        emits only nodes / backbone / grafts displacement files (there is
        no pendant pass), so a multi-atom SMILES end cap would leave its
        non-Si atoms with no displacement entry and they'd end up stuck
        at the origin. One bead per node is the CG design intent anyway.
        """
        from rdkit import Chem

        if self.config.model_type == "coarse_grained":
            self._place_simple_atom(node, "Si")
            return
        if _is_element(molecule):
            # A bare element ("Si" on a crosslinker that carries one strand)
            # is an atom, not SMILES: "Si" does not parse as SMILES, and the
            # failed parse used to print an error on every such node.
            # Its free valences are capped after the chains are built.
            self._place_simple_atom(node, molecule)
            return

        mol = Chem.MolFromSmiles(molecule)
        if mol:
            mol = Chem.RemoveHs(mol)
            idxs = [self.chemical_space.AddAtom(a) for a in mol.GetAtoms()]
            for b in mol.GetBonds():
                self.chemical_space.AddBond(
                    idxs[b.GetBeginAtomIdx()],
                    idxs[b.GetEndAtomIdx()],
                    b.GetBondType()
                )
            self.node_atoms[node] = idxs
            # Find attachment point (usually Si)
            for i, a in enumerate(mol.GetAtoms()):
                if a.GetSymbol() == "Si":
                    self.node_map[node] = idxs[i]
                    return
            self.node_map[node] = idxs[0]
        else:
            raise ChemistryBuilderError(
                f"end-cap molecule {molecule!r} is not a SMILES RDKit can parse")
    
    def _place_poss_cage(self, node: int):
        """Place a POSS cage structure (Si8O12)."""
        from rdkit import Chem
        
        # Create 8 Si atoms at corners
        ids = [self.chemical_space.AddAtom(Chem.Atom("Si")) for _ in range(8)]
        
        # Connect with O bridges (cube edges)
        # Cube connectivity: edges of a cube
        conns = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]
        
        for a, b in conns:
            o = self.chemical_space.AddAtom(Chem.Atom("O"))
            self.chemical_space.AddBond(ids[a], o, Chem.BondType.SINGLE)
            self.chemical_space.AddBond(o, ids[b], Chem.BondType.SINGLE)
        
        self.node_map[node] = ids  # List of 8 corner atoms
        self.poss_usage[node] = set()

    def _place_poss_am0270(self, node: int):
        """
        Place AM0270 POSS (AminopropylIsooctyl POSS).
        
        Structure:
        - Si8O12 core (cubic cage)
        - 7 corners (1-7) functionalized with IsoOctyl (2,4,4-trimethylpentyl)
        - 1 corner (0) functionalized with Propyl linker -> connects to network
        
        Corner layout (cube vertices):
            Corner 0: (-1, -1, -1)  <- Propyl linker (network connection)
            Corner 1: (-1, -1, +1)
            Corner 2: (-1, +1, -1)
            Corner 3: (-1, +1, +1)
            Corner 4: (+1, -1, -1)
            Corner 5: (+1, -1, +1)
            Corner 6: (+1, +1, -1)
            Corner 7: (+1, +1, +1)
        
        The IsoOctyl arms extend along the space diagonal (outward from cage center).
        """
        from rdkit import Chem
        
        # Initialize structure tracking if not exists
        if not hasattr(self, 'poss_structure'):
            self.poss_structure = {}
        
        # 1. Build Base Cage (Si8O12)
        corner_si_ids = [self.chemical_space.AddAtom(Chem.Atom("Si")) for _ in range(8)]
        
        # Oxygen bridges (12 edges of cube)
        cage_oxygen_ids = []
        conns = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
            (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
        ]
        
        for a, b in conns:
            o = self.chemical_space.AddAtom(Chem.Atom("O"))
            cage_oxygen_ids.append(o)
            self.chemical_space.AddBond(corner_si_ids[a], o, Chem.BondType.SINGLE)
            self.chemical_space.AddBond(o, corner_si_ids[b], Chem.BondType.SINGLE)
        
        # 2. Prepare Functional Groups
        isooctyl_smiles = "CC(C)CC(C)(C)C"  # 2,4,4-trimethylpentyl (8 carbons)
        propyl_smiles = "CCC"  # Propyl linker
        
        # 3. Attach Propyl to Corner 0 (Network Connection)
        mol_prop = Chem.MolFromSmiles(propyl_smiles)
        mol_prop = Chem.RemoveHs(mol_prop)
        
        propyl_idxs = [self.chemical_space.AddAtom(a) for a in mol_prop.GetAtoms()]
        for b in mol_prop.GetBonds():
            self.chemical_space.AddBond(
                propyl_idxs[b.GetBeginAtomIdx()],
                propyl_idxs[b.GetEndAtomIdx()],
                b.GetBondType()
            )
        
        # Connect first C to Si (Corner 0)
        self.chemical_space.AddBond(corner_si_ids[0], propyl_idxs[0], Chem.BondType.SINGLE)
        
        # Register the LAST carbon as the network attachment point
        self.node_map[node] = propyl_idxs[-1]
        
        # 4. Attach IsoOctyl to Corners 1-7 (Dangling Ends)
        isooctyl_arms = {}  # corner_idx -> list of atom indices
        
        for corner_idx in range(1, 8):
            mol_iso = Chem.MolFromSmiles(isooctyl_smiles)
            mol_iso = Chem.RemoveHs(mol_iso)
            
            iso_idxs = [self.chemical_space.AddAtom(a) for a in mol_iso.GetAtoms()]
            
            for b in mol_iso.GetBonds():
                self.chemical_space.AddBond(
                    iso_idxs[b.GetBeginAtomIdx()],
                    iso_idxs[b.GetEndAtomIdx()],
                    b.GetBondType()
                )
            
            # Connect first C to Si (Corner)
            self.chemical_space.AddBond(corner_si_ids[corner_idx], iso_idxs[0], Chem.BondType.SINGLE)
            
            isooctyl_arms[corner_idx] = iso_idxs
        
        # 5. Store structure metadata for coordinate generation
        self.poss_structure[node] = {
            'corner_si_ids': corner_si_ids,          # [8 Si atom indices]
            'cage_oxygen_ids': cage_oxygen_ids,      # [12 O atom indices]
            'propyl_arm': {
                'corner_idx': 0,
                'atom_ids': propyl_idxs              # [3 C atom indices]
            },
            'isooctyl_arms': isooctyl_arms           # {corner_idx: [8 C atom indices]}
        }

    
    def _get_attachment_atom(self, node: int, vec: np.ndarray) -> int:
        """
        Get the attachment atom for a node.
        
        For POSS, selects corner based on direction vector.
        For simple atoms, returns the atom index directly.
        """
        target = self.node_map.get(node)
        
        if target is None:
            return None
        
        if isinstance(target, int):
            return target
        
        # POSS cage - select corner based on direction
        # Corner vectors (normalized cube corners)
        corner_vecs = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], dtype=float)
        corner_vecs = corner_vecs / np.linalg.norm(corner_vecs, axis=1, keepdims=True)
        
        # Normalize direction
        v_dir = vec / (np.linalg.norm(vec) + 1e-9)
        
        # Find best matching corner not yet used
        used = self.poss_usage.get(node, set())
        dots = corner_vecs @ v_dir
        ranked = np.argsort(dots)[::-1]  # Best match first
        
        for idx in ranked:
            if idx not in used:
                used.add(idx)
                self.poss_usage[node] = used
                return target[idx]
        
        # All used, return first one again
        return target[0]
    
    def _build_chains(self):
        """Build chain structures for each edge."""
        print("  Building chains...")
        
        chain_count = 0
        for u, v, key, data in self.G.edges(keys=True, data=True):
            dp = data.get("dp", 25)
            edge_type = data.get("edge_type", "A")

            # Get monomer for this edge type
            edge_config = self.config.edge_type_map.get(edge_type)
            if edge_config:
                monomer_name = edge_config.monomer
            else:
                monomer_name = "PDMS"  # Default

            # Get monomer definition
            monomer_config = self.config.monomers.get(monomer_name)
            if not monomer_config:
                print(f"    Warning: Monomer {monomer_name} not found, using PDMS")
                monomer_config = self.config.monomers.get("PDMS")

            # Build the chain (pass full edge data for graft/copolymer attributes)
            self._next_mol += 1
            self.chain_mol[(u, v, key)] = self._next_mol
            self._build_chain(u, v, key, dp, monomer_config, data)
            chain_count += 1

        n_loops = sum(1 for u, v in self.G.edges() if u == v)
        if n_loops:
            print(f"      of which {n_loops} primary loops (rings closed on "
                  f"their own junction)")
        self._build_sol_chains()
        print(f"    Built {chain_count} chains")
    
    def _build_chain(self, u: int, v: int, key: int, dp: int, monomer_config, edge_data: dict = None):
        """Build a single chain between two nodes."""
        from rdkit import Chem
        
        # Get positions for direction calculation
        pos_u = self.G.nodes[u].get("pos")
        pos_v = self.G.nodes[v].get("pos")
        
        if pos_u is not None and pos_v is not None:
            pos_u = np.array(pos_u)
            pos_v = np.array(pos_v)
            
            # MIC vector
            if self.dims is not None:
                raw = pos_v - pos_u
                vec = raw - self.dims * np.round(raw / self.dims)
            else:
                vec = pos_v - pos_u
        else:
            vec = np.array([1, 0, 0])  # Default direction

        if u == v:
            # A primary loop: both ends of the strand land on the same
            # junction, so there is no chord to take a direction from.
            # An arbitrary axis is enough; on a POSS cage the two calls
            # below then pick opposite corners, as two strands would.
            vec = np.array([1.0, 0.0, 0.0])

        # Get attachment atoms
        att_u = self._get_attachment_atom(u, vec)
        att_v = self._get_attachment_atom(v, -vec)
        
        if att_u is None or att_v is None:
            return  # Skip if nodes not built
        
        # For coarse-grained, just add beads
        if self.config.model_type == "coarse_grained":
            self._build_chain_cg(u, v, key, dp, att_u, att_v, edge_data or {})
        else:
            self._build_chain_atomistic(
                u, v, key, dp, monomer_config, att_u, att_v, edge_data or {}
            )
        self._tag_chain((u, v, key), u, v)

    def _tag_chain(self, edge_id, u, v) -> None:
        """Give a strand's atoms their molecule id and their chain roles.

        The molecule is the chain; its two chemical ends carry the ``end``
        role and everything between them ``interior``. A dangling strand
        ends on an end-cap node, which joins the chain's molecule and is
        the chain's free end. That is the end-linked convention, where
        every chain is DP beads whether or not it reacted at both ends.
        """
        atoms = list(self.edge_atom_map.get(edge_id, []))
        if not atoms:
            return
        mol_id = self.chain_mol.get(edge_id, 1)

        def cap_atom(node):
            ref = self.node_map.get(node) if self.node_is_end_cap.get(node) else None
            if ref is None or isinstance(ref, int):
                return ref
            return ref[0]

        # The chain's two chemical ends: its backbone head and tail, or the
        # end-cap node at whichever side is free. Atomistic chains carry
        # side atoms in `edge_atom_map` too, so the ends come from
        # `edge_backbone_map` rather than from the list order.
        head, tail = self.edge_backbone_map.get(edge_id, (atoms[0], atoms[-1]))
        ends = {cap_atom(u) if cap_atom(u) is not None else head,
                cap_atom(v) if cap_atom(v) is not None else tail}
        for idx in atoms + [a for a in (cap_atom(u), cap_atom(v)) if a is not None]:
            self._tag_atoms(idx, mol_id, "end" if idx in ends else "interior")
        for _, graft_atoms in self.graft_atom_map.get(edge_id, []):
            self._tag_atoms(graft_atoms, mol_id, "interior")

    def _cap_junctions(self) -> None:
        """Fill a bare junction atom's free valence with methyls.

        A junction built as a single Si has four bonds to give, and a
        junction with fewer than four strands (an effective degree 2 or 3
        site, or a crosslinker that carries one strand) used to have the rest
        filled by ``AddHs`` with hydrogen. PDMS silicon bonds to oxygen and
        carbon, so the site becomes MeSi(O-)3, Me2Si(O-)2 or Me3Si-O-, the
        same rule the workflow route applies. Atomistic only; a junction that
        is a larger molecule (POSS, a SMILES) keeps whatever its molecule
        defines. The caps join the node's atoms, so CHARMM typing counts them
        in the node's residue.
        """
        if self.config.model_type != "atomistic":
            return
        from rdkit import Chem

        n_caps = 0
        for node, ref in self.node_map.items():
            if not isinstance(ref, int) or len(self.node_atoms.get(node, ())) != 1:
                continue
            atom = self.chemical_space.GetAtomWithIdx(ref)
            valence = _METHYL_CAPPED.get(atom.GetSymbol())
            if valence is None:
                continue
            caps = []
            for _ in range(max(0, valence - atom.GetDegree())):
                c = self.chemical_space.AddAtom(Chem.Atom("C"))
                self.chemical_space.AddBond(ref, c, Chem.BondType.SINGLE)
                caps.append(c)
            if caps:
                self.cap_atoms[node] = caps
                self.node_atoms[node] = list(self.node_atoms[node]) + caps
                n_caps += len(caps)
        if n_caps:
            print(f"    Capped {len(self.cap_atoms)} under-coordinated junctions "
                  f"with {n_caps} methyls")

    def strand_table(self, mol=None) -> dict:
        """Which atoms make up each strand and each node, for the manifest.

        Every strand the builder made, with its class, DP, the junction atom
        at each end, its backbone in order from the bonded end (bridge atoms
        included, junction atoms not), one atom per repeat unit, and its
        heavy atoms and hydrogens, as ranges where they are contiguous (they
        are, for strands built from a repeated SMILES) and as lists where
        not. A dangling strand ends on its end-cap node's attachment atom,
        given as ``free_end``. Indices are those of ``mol`` (0-based RDKit);
        the pipeline writes them as LAMMPS ids.

        ``mol`` is the molecule with hydrogens when there is one; without
        it the hydrogens are left empty.
        """
        mol = self.chemical_space if mol is None else mol

        def span(ids):
            ids = sorted(int(i) for i in ids)
            if not ids:
                return {"range": None}
            if ids[-1] - ids[0] + 1 == len(ids):
                return {"range": [ids[0], ids[-1]]}
            return {"ids": ids}

        def hydrogens(heavy):
            out = []
            for i in heavy:
                if i >= mol.GetNumAtoms():
                    continue
                out.extend(n.GetIdx() for n in mol.GetAtomWithIdx(int(i)).GetNeighbors()
                           if n.GetAtomicNum() == 1)
            return out

        strands = []
        for edge_id, atoms in self.edge_atom_map.items():
            u, v, key = edge_id
            att_u, att_v = self.edge_ends.get(edge_id, (None, None))
            cap_u = bool(self.node_is_end_cap.get(u))
            cap_v = bool(self.node_is_end_cap.get(v))
            if u == v:
                cls = "loop"
            elif cap_u and cap_v:
                cls = "free"
            elif cap_u or cap_v:
                cls = "dangling"
            else:
                cls = "bridge"
            backbone = list(self.edge_backbone_path.get(edge_id, atoms))
            heads = list(self.edge_repeat_heads.get(edge_id, backbone))
            ends = [att_u, att_v]
            free_end = free_ends = None
            if cls == "dangling":
                if cap_u:           # bonded end first, as the end-linked reader does
                    backbone.reverse()
                    heads.reverse()
                    ends = [att_v, att_u]
                free_end = ends[1]
                ends = [ends[0], None]
            elif cls == "free":
                # a strand between two end caps, bonded to no junction
                free_ends = [att_u, att_v]
                ends = [None, None]
            heavy = set(atoms) | set(backbone)
            for _, graft in self.graft_atom_map.get(edge_id, []):
                heavy |= set(graft)
            heavy -= {att_u, att_v}
            row = {"edge": [u, v, key], "cls": cls,
                   "dp": int(self.G[u][v][key].get("dp", 0)),
                   "junctions": ends, "backbone": backbone,
                   "repeat_heads": heads}
            if free_end is not None:
                row["free_end"] = free_end
            if free_ends is not None:
                row["free_ends"] = free_ends
            row["heavy"] = span(heavy)
            row["hydrogens"] = span(hydrogens(heavy))
            strands.append(row)

        # Every node the builder placed. A POSS cage has a list of corner
        # attachment atoms (each strand bonds to one), and no node_atoms
        # entry, so its corners stand in for its atoms; a reader groups a
        # strand's end atom into its node through ``attach``.
        nodes = []
        for node, ref in self.node_map.items():
            attach = list(ref) if isinstance(ref, (list, tuple)) else int(ref)
            atom_ids = self.node_atoms.get(node) or (
                attach if isinstance(attach, list) else [attach])
            nodes.append({"node": node,
                          "kind": "end" if self.node_is_end_cap.get(node) else "junction",
                          "attach": attach,
                          "atoms": sorted(int(i) for i in atom_ids),
                          "hydrogens": sorted(hydrogens(atom_ids))})
        return {"strands": strands, "nodes": nodes}

    def _build_sol_chains(self) -> None:
        """Build the sol: chains bonded to no junction.

        They are not in the junction graph (``G.graph["sol_chains"]``
        carries their count and DP, written by the defects stage), but they
        hold beads, and the box is sized from the bead count. Leaving them
        out makes the bridge density, and with it the tensile peak, too
        high by their share (7 % of the beads on the DP-20 reference,
        together with the primary loops). A crosslinked melt whose sol
        chains differ in length lists them as ``dps`` (DP to count), and
        each is built at its own length.
        """
        spec = self.G.graph.get("sol_chains")
        if not spec:
            return
        count = int(spec.get("count", 0))
        dp = int(spec.get("dp", 25))
        lengths = ([int(d) for d, k in sorted(spec["dps"].items()) for _ in range(int(k))]
                   if spec.get("dps") else [dp] * count)
        lengths = [d for d in lengths if d > 0]
        if count <= 0 or not lengths:
            return
        if self.config.model_type != "coarse_grained":
            warnings.warn(
                f"{count} sol chains requested; sol is built for "
                f"coarse-grained systems only, so they are left out of the "
                f"atomistic structure (and of its bead budget).",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        from rdkit import Chem

        bead_type = spec.get("bead_type", "B")
        for length in lengths:
            self._next_mol += 1
            idxs = []
            for _ in range(length):
                bead = Chem.Atom("C")
                bead.SetProp("bead_type", bead_type)
                idxs.append(self.chemical_space.AddAtom(bead))
            for a, b in zip(idxs[:-1], idxs[1:]):
                self.chemical_space.AddBond(a, b, Chem.BondType.SINGLE)
            for i, idx in enumerate(idxs):
                self._tag_atoms(idx, self._next_mol,
                                "end" if i in (0, len(idxs) - 1) else "interior")
            self.sol_atom_map.append(idxs)
        if len(set(lengths)) == 1:
            print(f"    Built {len(lengths)} sol chains of {lengths[0]} beads")
        else:
            print(f"    Built {len(lengths)} sol chains of "
                  + ", ".join(f"{k} x {d}" for d, k in sorted(collections.Counter(lengths).items()))
                  + " beads")
    
    def _build_chain_cg(self, u, v, key, dp, att_u, att_v, edge_data: dict = None):
        """Build a coarse-grained chain (simple bead chain).

        Reads edge attributes (set by AssignmentManager) to:
        - Set per-bead ``bead_type`` from ``monomer_sequence`` if present.
        - Attach graft side chains at positions listed in ``graft_positions``.
        """
        from rdkit import Chem

        if edge_data is None:
            edge_data = {}

        edge_id = (u, v, key)
        chain_idxs = []

        # Per-bead monomer sequence (length == dp when set by copolymer assignment)
        monomer_seq = edge_data.get("monomer_sequence")  # list[str] | None

        # Create backbone beads
        for i in range(dp):
            bead = Chem.Atom("C")  # Use C as generic CG bead element
            bead_type = (monomer_seq[i] if monomer_seq and i < len(monomer_seq) else "B")
            bead.SetProp("bead_type", bead_type)
            idx = self.chemical_space.AddAtom(bead)
            chain_idxs.append(idx)

        # Connect backbone beads
        for i in range(len(chain_idxs) - 1):
            self.chemical_space.AddBond(chain_idxs[i], chain_idxs[i + 1], Chem.BondType.SINGLE)

        # Connect backbone to nodes. A strand of no beads (a crosslink on the
        # bead next to a chain end) is the one bond from its junction to the
        # end bead.
        if chain_idxs:
            self.chemical_space.AddBond(att_u, chain_idxs[0], Chem.BondType.SINGLE)
            self.chemical_space.AddBond(chain_idxs[-1], att_v, Chem.BondType.SINGLE)
        elif att_u != att_v and self.chemical_space.GetBondBetweenAtoms(att_u, att_v) is None:
            self.chemical_space.AddBond(att_u, att_v, Chem.BondType.SINGLE)

        # Graft side chains.
        # Stored as ``[(frac, [side-chain atom ids]), ...]`` to match
        # Pipeline's graft-placement loop (and the atomistic shape). The
        # backbone position ``pos`` is mapped to ``frac = (pos+1)/(dp+1)``
        # so the placement code can interpolate along the backbone vector.
        graft_positions = edge_data.get("graft_positions", [])
        graft_dp = edge_data.get("graft_dp", 5)
        graft_monomer = edge_data.get("graft_monomer", "G")

        edge_grafts: list[tuple[float, list[int]]] = []
        for pos in graft_positions:
            if pos < 0 or pos >= len(chain_idxs):
                continue
            backbone_idx = chain_idxs[pos]
            side_idxs: list[int] = []
            prev = backbone_idx
            for _ in range(graft_dp):
                g_bead = Chem.Atom("C")
                g_bead.SetProp("bead_type", graft_monomer)
                g_idx = self.chemical_space.AddAtom(g_bead)
                self.chemical_space.AddBond(prev, g_idx, Chem.BondType.SINGLE)
                side_idxs.append(g_idx)
                prev = g_idx
            edge_grafts.append(((pos + 1) / (dp + 1), side_idxs))

        self.edge_atom_map[edge_id] = chain_idxs
        self.edge_backbone_map[edge_id] = (chain_idxs[0], chain_idxs[-1]) if chain_idxs else (None, None)
        if edge_grafts:
            self.graft_atom_map[edge_id] = edge_grafts
        self.edge_ends[edge_id] = (att_u, att_v)
        self.edge_backbone_path[edge_id] = list(chain_idxs)
        self.edge_repeat_heads[edge_id] = list(chain_idxs)

    def _build_chain_atomistic(self, u, v, key, dp, monomer_config, att_u, att_v, edge_data: dict = None):
        """Build an atomistic chain with smart bridge detection.

        When ``edge_data['graft_positions']`` is non-empty AND the monomer
        SMILES matches the PDMS reference ``[Si](C)(C)O``, falls back to a
        per-repeat builder (``_build_pdms_chain_with_grafts``) that emits a
        side chain at each marked backbone Si and records the per-edge graft
        atom map. For non-PDMS monomers with grafts, a warning is emitted
        and grafts are skipped — the SMILES-concatenation path can't add
        conditional side chains at specific repeat positions.
        """
        from rdkit import Chem

        if edge_data is None:
            edge_data = {}

        edge_id = (u, v, key)
        smiles = monomer_config.smiles
        chain_head_atom = monomer_config.chain_head
        chain_tail_atom = monomer_config.chain_tail

        graft_positions = list(edge_data.get("graft_positions") or [])
        graft_dp = int(edge_data.get("graft_dp", 5))
        use_per_repeat = bool(graft_positions) and smiles == "[Si](C)(C)O"

        if graft_positions and not use_per_repeat:
            warnings.warn(
                f"Edge ({u},{v}): graft_positions set but monomer SMILES "
                f"is {smiles!r} — atomistic graft is currently implemented "
                f"only for PDMS '[Si](C)(C)O'; grafts skipped.",
                RuntimeWarning, stacklevel=2,
            )
            graft_positions = []

        edge_grafts: list[tuple[float, list[int]]] = []

        if use_per_repeat:
            chain_idxs, edge_grafts, backbone = self._build_pdms_chain_with_grafts(
                dp, graft_positions, graft_dp
            )
            if not chain_idxs:
                return
            chain_head = chain_idxs[0]
            chain_tail = chain_idxs[-1]
        else:
            # SMILES-concatenation path (fast for non-grafted PDMS or any
            # custom monomer where per-position grafting isn't supported).
            try:
                chain_mol = self._create_chain_from_smiles(smiles, dp)
            except ValueError as exc:
                warnings.warn(
                    f"Skipping edge ({u}, {v}): {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            if chain_mol is None:
                return

            chain_idxs = [self.chemical_space.AddAtom(a) for a in chain_mol.GetAtoms()]
            for b in chain_mol.GetBonds():
                self.chemical_space.AddBond(
                    chain_idxs[b.GetBeginAtomIdx()],
                    chain_idxs[b.GetEndAtomIdx()],
                    b.GetBondType()
                )

            if not chain_idxs:
                return

            chain_head = chain_idxs[0]
            chain_tail = chain_idxs[-1]
            # The backbone is the shortest route from the chain's head atom to
            # its tail through its own atoms (Si-O-Si-O... for PDMS), read on
            # the chain before it joins the network, where no shortcut
            # through a junction exists yet.
            local = (list(Chem.GetShortestPath(chain_mol, 0, len(chain_idxs) - 1))
                     if len(chain_idxs) > 1 else [0])
            backbone = [chain_idxs[i] for i in local]

        bridge_u = bridge_v = None

        # Check if need bridge on left side
        node_u_symbol = self.chemical_space.GetAtomWithIdx(att_u).GetSymbol()
        head_symbol = chain_head_atom
        
        if self.config.connection.auto_bridge and node_u_symbol == head_symbol:
            # Same atom type - need bridge
            bridge = self.chemical_space.AddAtom(Chem.Atom(self.config.connection.default_bridge_atom))
            self.bridge_atoms.append(bridge)
            self.chemical_space.AddBond(att_u, bridge, Chem.BondType.SINGLE)
            self.chemical_space.AddBond(bridge, chain_head, Chem.BondType.SINGLE)
            bridge_u = bridge
        else:
            # Direct bond OK
            self.chemical_space.AddBond(att_u, chain_head, Chem.BondType.SINGLE)
        
        # Check if need bridge on right side (usually not for PDMS)
        node_v_symbol = self.chemical_space.GetAtomWithIdx(att_v).GetSymbol()
        tail_symbol = chain_tail_atom
        
        if self.config.connection.auto_bridge and node_v_symbol == tail_symbol:
            # Same atom type - need bridge
            bridge = self.chemical_space.AddAtom(Chem.Atom(self.config.connection.default_bridge_atom))
            self.bridge_atoms.append(bridge)
            self.chemical_space.AddBond(chain_tail, bridge, Chem.BondType.SINGLE)
            self.chemical_space.AddBond(bridge, att_v, Chem.BondType.SINGLE)
            bridge_v = bridge
        else:
            # Direct bond OK
            self.chemical_space.AddBond(chain_tail, att_v, Chem.BondType.SINGLE)

        self.edge_atom_map[edge_id] = chain_idxs
        self.edge_backbone_map[edge_id] = (chain_head, chain_tail)
        if edge_grafts:
            self.graft_atom_map[edge_id] = edge_grafts
        self.edge_ends[edge_id] = (att_u, att_v)
        self.edge_backbone_path[edge_id] = (
            ([bridge_u] if bridge_u is not None else []) + backbone
            + ([bridge_v] if bridge_v is not None else []))
        # One atom per repeat unit: every repeat of a concatenated monomer
        # puts the same number of atoms on the backbone, so the heads are
        # the backbone taken at that stride (Si of each Si-O pair for PDMS,
        # the first C of each C-O-C for PEG).
        stride = len(backbone) // dp if dp > 0 and len(backbone) % dp == 0 else 1
        self.edge_repeat_heads[edge_id] = backbone[::max(stride, 1)]

    def _build_pdms_chain_with_grafts(
        self,
        dp: int,
        graft_positions: list,
        graft_dp: int,
    ) -> tuple[list, list]:
        """Build a PDMS chain repeat-by-repeat, attaching side chains.

        Each backbone repeat is structurally ``Si(C)(C)O`` so the chain is
        the same atom-order as ``[Si](C)(C)O`` × dp (matches the
        SMILES-concatenation path's ``chain_idxs`` ordering, so any
        index-based callers behave identically). For each backbone Si
        whose index ``k`` is in ``graft_positions``, one of the two
        methyl C caps is replaced by a branch O that leads into a side
        chain of ``graft_dp`` repeat units (``Si(C)(C)O`` × graft_dp
        but with the trailing O omitted on the last repeat). This keeps
        every Si at valence 4.

        Returns:
            (chain_idxs, edge_grafts, backbone) where
                chain_idxs   = global RDKit atom indices in build order
                edge_grafts  = [(frac, [side_atom_idx, ...]), ...]
                               frac = (k+1)/(dp+1), matching the canonical
                               workflow's per-edge graft_map shape.
                backbone     = the Si and linker O of every repeat, in order
        """
        from rdkit import Chem

        graft_set = set(int(p) for p in graft_positions)
        M = self.chemical_space
        chain_idxs: list = []
        edge_grafts: list[tuple[float, list[int]]] = []
        backbone: list = []
        prev = None  # previous repeat's linker O

        for k in range(dp):
            si = M.AddAtom(Chem.Atom("Si"))
            chain_idxs.append(si)
            backbone.append(si)
            if prev is not None:
                M.AddBond(prev, si, Chem.BondType.SINGLE)

            if k in graft_set:
                # 1 methyl cap (instead of 2) + branch O + side chain
                c1 = M.AddAtom(Chem.Atom("C"))
                chain_idxs.append(c1)
                M.AddBond(si, c1, Chem.BondType.SINGLE)

                g_o = M.AddAtom(Chem.Atom("O"))
                chain_idxs.append(g_o)
                M.AddBond(si, g_o, Chem.BondType.SINGLE)

                # Side chain: graft_dp repeats of Si(C)(C)O; drop the
                # trailing O on the last repeat so the tail Si caps at
                # valence 3 (Si + 2 methyls + 1 bridge from prev = 4).
                g_atoms = [g_o]
                g_prev = g_o
                for j in range(graft_dp):
                    g_si = M.AddAtom(Chem.Atom("Si"))
                    g_atoms.append(g_si)
                    M.AddBond(g_prev, g_si, Chem.BondType.SINGLE)
                    g_c1 = M.AddAtom(Chem.Atom("C"))
                    g_c2 = M.AddAtom(Chem.Atom("C"))
                    g_atoms.extend([g_c1, g_c2])
                    M.AddBond(g_si, g_c1, Chem.BondType.SINGLE)
                    M.AddBond(g_si, g_c2, Chem.BondType.SINGLE)
                    if j < graft_dp - 1:
                        g_next_o = M.AddAtom(Chem.Atom("O"))
                        g_atoms.append(g_next_o)
                        M.AddBond(g_si, g_next_o, Chem.BondType.SINGLE)
                        g_prev = g_next_o
                edge_grafts.append(((k + 1) / (dp + 1), g_atoms))
            else:
                # Normal repeat: 2 methyl caps
                c1 = M.AddAtom(Chem.Atom("C"))
                c2 = M.AddAtom(Chem.Atom("C"))
                chain_idxs.extend([c1, c2])
                M.AddBond(si, c1, Chem.BondType.SINGLE)
                M.AddBond(si, c2, Chem.BondType.SINGLE)

            # Trailing linker O of this repeat
            o = M.AddAtom(Chem.Atom("O"))
            chain_idxs.append(o)
            backbone.append(o)
            M.AddBond(si, o, Chem.BondType.SINGLE)
            prev = o

        return chain_idxs, edge_grafts, backbone

    def _create_chain_from_smiles(self, smiles: str, dp: int):
        """Create a polymer chain from repeating SMILES unit.

        Returns the RDKit molecule for a chain of *dp* repeat units, or
        ``None`` if the chain cannot be constructed.  Raises ``ValueError``
        if the SMILES concatenation fails and falls back to a single monomer
        (which would silently produce an under-length chain).
        """
        from rdkit import Chem

        # Remove trailing O for linking (if present)
        if smiles.endswith("O"):
            unit = smiles[:-1]
            linker = "O"
        else:
            unit = smiles
            linker = ""

        # Build full chain SMILES.
        # The last repeat unit's linker O serves as the chain tail; the
        # auto-bridge in `_build_chain_atomistic` direct-bonds it to the
        # network/end-cap Si node. Do NOT append a trailing "[O]" — that
        # produces a spurious -O-O- peroxide bond at the chain tail.
        if linker:
            chain_smiles = (unit + linker) * dp
        else:
            chain_smiles = unit * dp

        mol = Chem.MolFromSmiles(chain_smiles)
        if mol is not None:
            return Chem.RemoveHs(mol)

        # Chain SMILES failed — do NOT silently return a single monomer.
        # That would embed a 1-unit chain instead of a dp-unit chain with
        # no warning, corrupting the molecular structure.
        raise ValueError(
            f"_create_chain_from_smiles: failed to parse chain SMILES for "
            f"dp={dp}, smiles={smiles!r}.\n"
            f"  Attempted chain SMILES: {chain_smiles!r}\n"
            f"  The monomer SMILES may not support simple string concatenation "
            f"for chain building. Use a monomer with a terminal 'O' linker, "
            f"or check that repeated concatenation produces valid SMILES."
        )
