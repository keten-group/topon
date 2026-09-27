import os
import itertools
from rdkit import Chem

#: Atom types of the end-linked convention, in the order they are written.
ENDLINKED_TYPES = {"end": 1, "interior": 2, "junction": 3}


class CGWriter:
    """
    Writes LAMMPS data files for Coarse-Grained (Kremer-Grest) systems.

    Two conventions:

    ``topon`` (default)
        One molecule for the whole network; atom types are the distinct
        ``bead_type`` properties, numbered in sorted order.
    ``endlinked``
        The convention of `fix bond/create` datasets: type 1 = chain-end
        bead, 2 = chain interior, 3 = junction; one molecule per chain and
        one per junction; a dangling chain's free end is its DP-th bead and
        a primary loop is a ring whose two ends bond to the same junction.
        This is what ``refnet.parse`` and the Z1+ exporter read, so it is
        what makes topon output directly comparable with an end-linked
        dataset. It needs the molecule and role properties
        :class:`topon.chemistry.builder.ChemistryBuilder` writes
        (``topon_mol``, ``topon_role``); atoms carrying neither are written
        as interior beads of one molecule.
    """
    def __init__(self, mol, output_file, include_angles=True, box_size=None,
                 pair_style="attractive", convention="topon"):
        self.mol = mol
        self.output_file = output_file
        self.include_angles = include_angles
        self.box_size = box_size if box_size else (100.0, 100.0, 100.0)
        self.pair_style = pair_style
        if convention not in ("topon", "endlinked"):
            raise ValueError(
                f"Unknown CG data convention {convention!r} "
                f"(expected 'topon' or 'endlinked')"
            )
        self.convention = convention

        
        # Standard Kremer-Grest Parameters (Epsilon=1.0, Sigma=1.0)
        self.mass = 1.0
        
        cutoff = 1.12246 if pair_style == "repulsive" else 2.5
        
        self.params = {
            'bond': {'style': 'harmonic', 'K': 466.1, 'R0': 0.97}, # Harmonic start
            'angle': {'style': 'harmonic', 'K': 466.1, 'theta0': 180.0}, # Stiff harmonic angle
            'pair': {'epsilon': 1.0, 'sigma': 1.0, 'cutoff': cutoff} 
        }

        # Containers
        self.atom_types = {}    # {'A': 1, 'J': 2}
        self.atom_data = []     # List of dicts
        self.bond_types = {}    # {(type1, type2): 1}
        self.bond_data = []
        self.angle_types = {}   # {(t1, t2, t3): 1}
        self.angle_data = []

    def write(self):
        print(f"Writing CG Data File: {os.path.basename(self.output_file)}")
        print(f"  -> Angles Enabled: {self.include_angles}")
        print(f"  -> Convention: {self.convention}")
        
        self._assign_atom_types()
        self._extract_bonds()
        
        if self.include_angles:
            self._extract_angles()
        
        self._write_file()
        print("Write complete.")

    def _assign_atom_types(self):
        if self.convention == "endlinked":
            self._assign_atom_types_endlinked()
            return
        # 1. Discover all unique bead types from RDKit properties
        unique_types = set()
        for atom in self.mol.GetAtoms():
            # Default to 'A' if no prop found (e.g. from simplistic builds)
            b_type = atom.GetProp("bead_type") if atom.HasProp("bead_type") else "A"
            unique_types.add(b_type)
        
        # 2. Assign IDs (Sort for consistency)
        for i, t in enumerate(sorted(unique_types)):
            self.atom_types[t] = i + 1
            
        # 3. Build Atom Data
        for atom in self.mol.GetAtoms():
            b_type = atom.GetProp("bead_type") if atom.HasProp("bead_type") else "A"
            self.atom_data.append({
                'id': atom.GetIdx() + 1,
                'mol_id': 1, # Single molecule for network
                'type': self.atom_types[b_type],
                'x': 0.0, 'y': 0.0, 'z': 0.0 # Placeholders, will be displaced later
            })

    def _assign_atom_types_endlinked(self):
        """Types and molecules of the end-linked convention.

        Roles come from the chemistry builder; molecule ids are renumbered
        from 1 in the order they first appear, so the file is dense whether
        or not every node built a structure.
        """
        self.atom_types = dict(ENDLINKED_TYPES)
        seen: dict = {}
        missing = 0
        for atom in self.mol.GetAtoms():
            role = atom.GetProp("topon_role") if atom.HasProp("topon_role") else None
            if role not in self.atom_types:
                role = "interior"
                missing += 1
            raw_mol = atom.GetIntProp("topon_mol") if atom.HasProp("topon_mol") else 0
            mol_id = seen.setdefault(raw_mol, len(seen) + 1)
            self.atom_data.append({
                'id': atom.GetIdx() + 1,
                'mol_id': mol_id,
                'type': self.atom_types[role],
                'x': 0.0, 'y': 0.0, 'z': 0.0,
            })
        if missing:
            print(f"  -> [WARN] {missing} atoms carried no chain role and were "
                  f"written as interior beads")
        print(f"  -> Molecules: {len(seen)}")

    def _extract_bonds(self):
        # Generic FENE bond for all connections
        # We can expand this later if you need specific bond types (e.g. A-B vs A-A)
        self.bond_types['default'] = 1 
        
        for bond in self.mol.GetBonds():
            self.bond_data.append((
                len(self.bond_data) + 1,
                1, # Always Type 1 for now
                bond.GetBeginAtomIdx() + 1,
                bond.GetEndAtomIdx() + 1
            ))

    def _extract_angles(self):
        # Simple iteration over all atoms with >= 2 neighbors
        self.angle_types['default'] = 1
        
        for atom in self.mol.GetAtoms():
            if atom.GetDegree() < 2: continue
            
            # Get neighbors and form triplets
            neighbors = [n.GetIdx() + 1 for n in atom.GetNeighbors()]
            center = atom.GetIdx() + 1
            
            for n1, n2 in itertools.combinations(neighbors, 2):
                self.angle_data.append((
                    len(self.angle_data) + 1,
                    1, # Always Type 1
                    n1, center, n2
                ))

    def _write_file(self):
        with open(self.output_file, 'w') as f:
            f.write("LAMMPS data file (Kremer-Grest CG)\n\n")
            f.write(f"{len(self.atom_data)} atoms\n")
            f.write(f"{len(self.bond_data)} bonds\n")
            if self.include_angles:
                f.write(f"{len(self.angle_data)} angles\n")
            
            f.write(f"\n{len(self.atom_types)} atom types\n")
            f.write(f"{len(self.bond_types)} bond types\n")
            if self.include_angles:
                f.write(f"{len(self.angle_types)} angle types\n")
                
            # Box
            lx, ly, lz = self.box_size
            f.write(f"\n{-lx/2.0:.4f} {lx/2.0:.4f} xlo xhi\n")
            f.write(f"{-ly/2.0:.4f} {ly/2.0:.4f} ylo yhi\n")
            f.write(f"{-lz/2.0:.4f} {lz/2.0:.4f} zlo zhi\n\n")
            
            f.write("Masses\n\n")
            types = self.atom_types.items()
            if self.convention == "endlinked":
                types = sorted(types, key=lambda kv: kv[1])
            for t_name, t_id in types:
                f.write(f"{t_id} {self.mass} # {t_name}\n")
                
            f.write("\nPair Coeffs\n\n")
            p = self.params['pair']
            for t_id in self.atom_types.values():
                f.write(f"{t_id} {p['epsilon']} {p['sigma']} {p['cutoff']}\n")
                
            f.write("\nBond Coeffs\n\n")
            b = self.params['bond']
            # Writes Harmonic coeff: K R0
            f.write(f"1 {b['K']} {b['R0']}\n")
            
            if self.include_angles:
                f.write("\nAngle Coeffs\n\n")
                a = self.params['angle']
                f.write(f"1 {a['K']} {a['theta0']}\n")
                
            f.write("\nAtoms # full\n\n")
            for a in self.atom_data:
                # ID Mol Type Charge X Y Z
                f.write(f"{a['id']} {a['mol_id']} {a['type']} 0.0 {a['x']} {a['y']} {a['z']}\n")
                
            if self.bond_data:
                f.write("\nBonds\n\n")
                for d in self.bond_data:
                    f.write(f"{d[0]} {d[1]} {d[2]} {d[3]}\n")
                    
            if self.angle_data:
                f.write("\nAngles\n\n")
                for d in self.angle_data:
                    f.write(f"{d[0]} {d[1]} {d[2]} {d[3]} {d[4]}\n")