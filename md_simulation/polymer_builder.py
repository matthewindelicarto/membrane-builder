"""
Polymer Structure Builder for MD Simulations

Generates realistic all-atom polymer structures for TPU membranes
that can be used with GROMACS or other MD packages.

Polymers:
- Sparsa 1: PEG/PPG polyether + H12MDI urethane
- Sparsa 2: PCL polyester + H12MDI urethane
- Carbosil 1: PDMS + polycarbonate-urethane
- Carbosil 2: PDMS + higher hard segment
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os


@dataclass
class Atom:
    """Represents an atom in the structure"""
    id: int
    name: str
    element: str
    res_name: str
    res_id: int
    x: float
    y: float
    z: float
    charge: float = 0.0
    mass: float = 12.0


@dataclass
class Bond:
    """Represents a bond between atoms"""
    atom1: int
    atom2: int
    bond_type: str = "1"  # single bond


class PolymerChainBuilder:
    """Builds realistic polymer chains with correct chemistry"""

    # Bond lengths in Angstroms
    BOND_LENGTHS = {
        'C-C': 1.54,
        'C-O': 1.43,
        'C=O': 1.23,
        'C-N': 1.47,
        'N-H': 1.01,
        'Si-O': 1.64,
        'Si-C': 1.87,
    }

    # Atomic masses
    MASSES = {
        'C': 12.011,
        'O': 15.999,
        'N': 14.007,
        'H': 1.008,
        'Si': 28.086,
    }

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.atom_id = 1
        self.res_id = 1

    def _add_atom(self, element: str, x: float, y: float, z: float,
                  res_name: str, charge: float = 0.0) -> int:
        """Add an atom and return its ID"""
        atom = Atom(
            id=self.atom_id,
            name=element,
            element=element,
            res_name=res_name,
            res_id=self.res_id,
            x=x, y=y, z=z,
            charge=charge,
            mass=self.MASSES.get(element, 12.0)
        )
        self.atoms.append(atom)
        self.atom_id += 1
        return atom.id

    def _add_bond(self, atom1: int, atom2: int, bond_type: str = "1"):
        """Add a bond between two atoms"""
        self.bonds.append(Bond(atom1, atom2, bond_type))

    def _random_direction(self) -> Tuple[float, float, float]:
        """Generate random unit vector for chain growth"""
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.pi/4, 3*np.pi/4)
        return (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        )

    def _step(self, x: float, y: float, z: float, bond_len: float,
              direction: Tuple[float, float, float],
              wobble: float = 0.3) -> Tuple[float, float, float, Tuple]:
        """Take a step along the chain"""
        dx, dy, dz = direction
        dx += np.random.uniform(-wobble, wobble)
        dy += np.random.uniform(-wobble, wobble)
        dz += np.random.uniform(-wobble, wobble)
        mag = np.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            dx, dy, dz = dx/mag * bond_len, dy/mag * bond_len, dz/mag * bond_len
        new_dir = (dx/bond_len, dy/bond_len, dz/bond_len)
        return x + dx, y + dy, z + dz, new_dir

    def build_peg_segment(self, x: float, y: float, z: float,
                          direction: Tuple, n_units: int = 6) -> Tuple:
        """
        Build PEG segment: -[CH2-CH2-O]n-
        Used in Sparsa 1
        """
        prev_atom = None
        for _ in range(n_units):
            # CH2
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-C'], direction)
            c1 = self._add_atom("C", x, y, z, "PEG")
            if prev_atom:
                self._add_bond(prev_atom, c1)

            # CH2
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-C'], direction)
            c2 = self._add_atom("C", x, y, z, "PEG")
            self._add_bond(c1, c2)

            # O (ether)
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
            o1 = self._add_atom("O", x, y, z, "PEG", charge=-0.4)
            self._add_bond(c2, o1)

            prev_atom = o1
            self.res_id += 1

        return x, y, z, direction, prev_atom

    def build_pcl_segment(self, x: float, y: float, z: float,
                          direction: Tuple, n_units: int = 4) -> Tuple:
        """
        Build PCL (polycaprolactone) segment: -[O-CO-(CH2)5]n-
        Used in Sparsa 2
        """
        prev_atom = None
        for _ in range(n_units):
            # Ester O
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
            o1 = self._add_atom("O", x, y, z, "PCL", charge=-0.33)
            if prev_atom:
                self._add_bond(prev_atom, o1)

            # Carbonyl C
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
            c1 = self._add_atom("C", x, y, z, "PCL", charge=0.51)
            self._add_bond(o1, c1)

            # Carbonyl O (double bond)
            ox = x + np.random.uniform(-0.5, 0.5)
            oy = y + 1.23
            o2 = self._add_atom("O", ox, oy, z, "PCL", charge=-0.43)
            self._add_bond(c1, o2)

            # Methylene chain (CH2)5
            prev_c = c1
            for _ in range(5):
                x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-C'], direction)
                c = self._add_atom("C", x, y, z, "PCL")
                self._add_bond(prev_c, c)
                prev_c = c

            prev_atom = prev_c
            self.res_id += 1

        return x, y, z, direction, prev_atom

    def build_pdms_segment(self, x: float, y: float, z: float,
                           direction: Tuple, n_units: int = 5) -> Tuple:
        """
        Build PDMS segment: -[Si(CH3)2-O]n-
        Used in Carbosil
        """
        prev_atom = None
        for _ in range(n_units):
            # Si
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['Si-O'], direction)
            si = self._add_atom("Si", x, y, z, "PDM", charge=0.4)
            if prev_atom:
                self._add_bond(prev_atom, si)

            # Methyl groups
            c1 = self._add_atom("C", x + 0.9, y + 0.5, z, "PDM")
            c2 = self._add_atom("C", x - 0.9, y + 0.5, z, "PDM")
            self._add_bond(si, c1)
            self._add_bond(si, c2)

            # O
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['Si-O'], direction)
            o1 = self._add_atom("O", x, y, z, "PDM", charge=-0.4)
            self._add_bond(si, o1)

            prev_atom = o1
            self.res_id += 1

        return x, y, z, direction, prev_atom

    def build_urethane_hard(self, x: float, y: float, z: float,
                            direction: Tuple) -> Tuple:
        """
        Build H12MDI urethane hard segment
        -NH-CO-O-[cyclohexyl-CH2-cyclohexyl]-O-CO-NH-
        """
        prev_atom = None

        # First urethane linkage: -NH-CO-O-
        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-N'], direction)
        n1 = self._add_atom("N", x, y, z, "URE", charge=-0.47)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-N'], direction)
        c1 = self._add_atom("C", x, y, z, "URE", charge=0.51)
        self._add_bond(n1, c1)

        # Carbonyl O
        o1 = self._add_atom("O", x + 0.3, y + 1.23, z, "URE", charge=-0.43)
        self._add_bond(c1, o1)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        o2 = self._add_atom("O", x, y, z, "URE", charge=-0.33)
        self._add_bond(c1, o2)

        # Simplified cyclohexyl-CH2-cyclohexyl (H12MDI core)
        prev_c = None
        for i in range(6):
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-C'], direction)
            c = self._add_atom("C", x, y, z, "URE")
            if i == 0:
                self._add_bond(o2, c)
            if prev_c:
                self._add_bond(prev_c, c)
            prev_c = c

        # Second urethane linkage: -O-CO-NH-
        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        o3 = self._add_atom("O", x, y, z, "URE", charge=-0.33)
        self._add_bond(prev_c, o3)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        c2 = self._add_atom("C", x, y, z, "URE", charge=0.51)
        self._add_bond(o3, c2)

        o4 = self._add_atom("O", x + 0.3, y + 1.23, z, "URE", charge=-0.43)
        self._add_bond(c2, o4)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-N'], direction)
        n2 = self._add_atom("N", x, y, z, "URE", charge=-0.47)
        self._add_bond(c2, n2)

        self.res_id += 1
        return x, y, z, direction, n2

    def build_polycarbonate_urethane(self, x: float, y: float, z: float,
                                      direction: Tuple) -> Tuple:
        """
        Build polycarbonate-urethane hard segment for Carbosil
        -O-CO-O-[hexamethylene]-NH-CO-O-
        """
        # Carbonate: -O-CO-O-
        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        o1 = self._add_atom("O", x, y, z, "PCU", charge=-0.33)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        c1 = self._add_atom("C", x, y, z, "PCU", charge=0.51)
        self._add_bond(o1, c1)

        o2 = self._add_atom("O", x + 0.3, y + 1.23, z, "PCU", charge=-0.43)
        self._add_bond(c1, o2)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-O'], direction)
        o3 = self._add_atom("O", x, y, z, "PCU", charge=-0.33)
        self._add_bond(c1, o3)

        # Hexamethylene spacer
        prev_c = None
        for i in range(6):
            x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-C'], direction)
            c = self._add_atom("C", x, y, z, "PCU")
            if i == 0:
                self._add_bond(o3, c)
            if prev_c:
                self._add_bond(prev_c, c)
            prev_c = c

        # Urethane: -NH-CO-O-
        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-N'], direction)
        n1 = self._add_atom("N", x, y, z, "PCU", charge=-0.47)
        self._add_bond(prev_c, n1)

        x, y, z, direction = self._step(x, y, z, self.BOND_LENGTHS['C-N'], direction)
        c2 = self._add_atom("C", x, y, z, "PCU", charge=0.51)
        self._add_bond(n1, c2)

        o4 = self._add_atom("O", x + 0.3, y + 1.23, z, "PCU", charge=-0.43)
        self._add_bond(c2, o4)

        self.res_id += 1
        return x, y, z, direction, c2


class TPUMembraneBuilder:
    """Builds complete TPU membrane systems for MD simulation"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def build_membrane(self, composition: Dict[str, float],
                       box_size: Tuple[float, float, float] = (50, 50, 50),
                       n_chains: int = 20) -> Tuple[List[Atom], List[Bond]]:
        """
        Build a membrane with the specified composition

        Args:
            composition: Dict with keys 'Sparsa1', 'Sparsa2', 'Carbosil1', 'Carbosil2'
            box_size: Box dimensions in Angstroms (x, y, z)
            n_chains: Number of polymer chains to generate

        Returns:
            Tuple of (atoms list, bonds list)
        """
        builder = PolymerChainBuilder(self.seed)

        # Normalize composition
        total = sum(composition.values())
        if total == 0:
            total = 1

        sparsa1_frac = composition.get('Sparsa1', 0) / total
        sparsa2_frac = composition.get('Sparsa2', 0) / total
        carbosil1_frac = composition.get('Carbosil1', 0) / total
        carbosil2_frac = composition.get('Carbosil2', 0) / total

        bx, by, bz = box_size

        for i in range(n_chains):
            # Random starting position
            x = np.random.uniform(-bx/2 + 5, bx/2 - 5)
            y = np.random.uniform(-by/2 + 5, by/2 - 5)
            z = np.random.uniform(-bz/2 + 5, bz/2 - 5)
            direction = builder._random_direction()

            # Determine chain type based on composition
            r = np.random.random()
            if r < sparsa1_frac:
                chain_type = 'sparsa1'
            elif r < sparsa1_frac + sparsa2_frac:
                chain_type = 'sparsa2'
            elif r < sparsa1_frac + sparsa2_frac + carbosil1_frac:
                chain_type = 'carbosil1'
            else:
                chain_type = 'carbosil2'

            # Build chain with 3-5 repeat units
            n_repeats = np.random.randint(3, 6)

            for _ in range(n_repeats):
                if chain_type == 'sparsa1':
                    # PEG soft + urethane hard
                    x, y, z, direction, _ = builder.build_peg_segment(
                        x, y, z, direction, n_units=np.random.randint(5, 10))
                    x, y, z, direction, _ = builder.build_urethane_hard(x, y, z, direction)

                elif chain_type == 'sparsa2':
                    # PCL soft + urethane hard
                    x, y, z, direction, _ = builder.build_pcl_segment(
                        x, y, z, direction, n_units=np.random.randint(3, 6))
                    x, y, z, direction, _ = builder.build_urethane_hard(x, y, z, direction)

                elif chain_type in ('carbosil1', 'carbosil2'):
                    # PDMS soft + PC-urethane hard
                    n_pdms = np.random.randint(4, 8) if chain_type == 'carbosil1' else np.random.randint(3, 5)
                    x, y, z, direction, _ = builder.build_pdms_segment(
                        x, y, z, direction, n_units=n_pdms)
                    x, y, z, direction, _ = builder.build_polycarbonate_urethane(x, y, z, direction)

                # Random direction change for chain tangling
                if np.random.random() < 0.3:
                    direction = builder._random_direction()

        return builder.atoms, builder.bonds

    def write_pdb(self, atoms: List[Atom], filename: str,
                  box_size: Tuple[float, float, float] = (50, 50, 50)):
        """Write atoms to PDB format for visualization and GROMACS"""
        with open(filename, 'w') as f:
            # Write CRYST1 record for box
            bx, by, bz = box_size
            f.write(f"CRYST1{bx:9.3f}{by:9.3f}{bz:9.3f}  90.00  90.00  90.00 P 1           1\n")

            for atom in atoms:
                # PDB ATOM record format
                f.write(f"ATOM  {atom.id:5d} {atom.name:4s} {atom.res_name:3s}  "
                       f"{atom.res_id:4d}    {atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                       f"  1.00  0.00          {atom.element:>2s}\n")
            f.write("END\n")

    def write_gro(self, atoms: List[Atom], filename: str,
                  box_size: Tuple[float, float, float] = (50, 50, 50)):
        """Write atoms to GROMACS .gro format"""
        with open(filename, 'w') as f:
            f.write("TPU Membrane\n")
            f.write(f"{len(atoms)}\n")

            for atom in atoms:
                # GRO format: resid, resname, atomname, atomid, x, y, z (in nm)
                x_nm = atom.x / 10.0  # Convert A to nm
                y_nm = atom.y / 10.0
                z_nm = atom.z / 10.0
                f.write(f"{atom.res_id:5d}{atom.res_name:5s}{atom.name:>5s}"
                       f"{atom.id:5d}{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}\n")

            # Box vectors in nm
            bx, by, bz = [b/10.0 for b in box_size]
            f.write(f"{bx:10.5f}{by:10.5f}{bz:10.5f}\n")


def generate_membrane_structure(composition: Dict[str, float],
                                output_dir: str = ".",
                                box_size: Tuple[float, float, float] = (50, 50, 50),
                                n_chains: int = 20,
                                seed: int = 42) -> str:
    """
    Generate a membrane structure and write to files

    Args:
        composition: Dict with polymer fractions
        output_dir: Directory to write output files
        box_size: Box dimensions in Angstroms
        n_chains: Number of polymer chains
        seed: Random seed

    Returns:
        Path to generated PDB file
    """
    os.makedirs(output_dir, exist_ok=True)

    builder = TPUMembraneBuilder(seed=seed)
    atoms, bonds = builder.build_membrane(composition, box_size, n_chains)

    # Generate filename based on composition
    comp_str = "_".join([f"{k}{int(v*100)}" for k, v in composition.items() if v > 0])
    pdb_file = os.path.join(output_dir, f"membrane_{comp_str}.pdb")
    gro_file = os.path.join(output_dir, f"membrane_{comp_str}.gro")

    builder.write_pdb(atoms, pdb_file, box_size)
    builder.write_gro(atoms, gro_file, box_size)

    print(f"Generated {len(atoms)} atoms, {len(bonds)} bonds")
    print(f"PDB file: {pdb_file}")
    print(f"GRO file: {gro_file}")

    return pdb_file


if __name__ == "__main__":
    # Example: Generate a 70% Carbosil 1, 30% Sparsa 1 membrane
    composition = {
        'Sparsa1': 0.3,
        'Sparsa2': 0.0,
        'Carbosil1': 0.7,
        'Carbosil2': 0.0
    }

    generate_membrane_structure(
        composition=composition,
        output_dir="./structures",
        box_size=(50, 50, 50),
        n_chains=20
    )
