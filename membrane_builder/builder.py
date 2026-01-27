"""
Core membrane building module.
"""

from __future__ import annotations
import os
import math
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

from .config import MembraneConfig, BoxConfig
from .lipids import LipidLibrary, Lipid
from .physics import MembranePhysics, MembraneProperties


@dataclass
class Atom:
    name: str
    element: str
    x: float
    y: float
    z: float
    residue_name: str = "LIP"
    residue_id: int = 1
    chain: str = "A"

    def to_pdb_line(self, atom_id: int) -> str:
        return (
            f"ATOM  {atom_id:5d} {self.name:4s} {self.residue_name:3s} "
            f"{self.chain:1s}{self.residue_id:4d}    "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
            f"  1.00  0.00          {self.element:>2s}"
        )


@dataclass
class PlacedLipid:
    lipid_type: str
    atoms: List[Atom]
    anchor_position: np.ndarray
    leaflet: str  # "top" or "bottom"
    residue_id: int


@dataclass
class BuiltMembrane:
    """The built membrane with all lipids and properties."""
    lipids: List[PlacedLipid]
    box: BoxConfig
    properties: Optional[MembraneProperties] = None
    config: Optional[MembraneConfig] = None

    @property
    def n_atoms(self) -> int:
        return sum(len(lip.atoms) for lip in self.lipids)

    @property
    def n_lipids(self) -> int:
        return len(self.lipids)

    def get_all_atoms(self) -> List[Atom]:
        atoms = []
        for lipid in self.lipids:
            atoms.extend(lipid.atoms)
        return atoms

    def to_pdb_string(self) -> str:
        lines = []
        lines.append("HEADER    LIPID BILAYER MEMBRANE")
        lines.append("TITLE     Membrane built by Membrane Builder")
        lines.append("REMARK   1")
        lines.append("REMARK   1 MEMBRANE BUILDER OUTPUT")
        lines.append("REMARK   1")
        lines.append(f"REMARK   2 TOTAL LIPIDS: {self.n_lipids}")
        lines.append(f"REMARK   2 TOTAL ATOMS:  {self.n_atoms}")

        if self.properties:
            lines.append("REMARK   3")
            lines.append("REMARK   3 PHYSICAL PROPERTIES:")
            lines.append(f"REMARK   3   BENDING MODULUS (KC): {self.properties.bending_modulus:.2f} KT")
            lines.append(f"REMARK   3   BILAYER THICKNESS:    {self.properties.thickness:.1f} ANGSTROMS")
            lines.append(f"REMARK   3   AREA PER LIPID:       {self.properties.area_per_lipid:.1f} ANGSTROMS^2")
            lines.append(f"REMARK   3   NET CHARGE:           {self.properties.net_charge:+.0f} E")

        if self.config:
            lines.append("REMARK   4")
            lines.append("REMARK   4 COMPOSITION:")
            for name, lip_cfg in self.config.lipids.items():
                lines.append(f"REMARK   4   {name}: {lip_cfg.count_top} TOP, {lip_cfg.count_bottom} BOTTOM")

        lines.append(
            f"CRYST1{self.box.Lx:9.3f}{self.box.Ly:9.3f}{self.box.Lz:9.3f}"
            f"  90.00  90.00  90.00 P 1           1"
        )

        atom_id = 1
        for lipid in self.lipids:
            chain = "A" if lipid.leaflet == "top" else "B"
            for atom in lipid.atoms:
                atom.chain = chain
                lines.append(atom.to_pdb_line(atom_id))
                atom_id += 1

        lines.append("END")
        return "\n".join(lines)

    def to_gro_string(self) -> str:
        lines = []
        lines.append("Membrane built by Membrane Builder")
        lines.append(f"{self.n_atoms:5d}")

        atom_id = 1
        for lipid in self.lipids:
            for atom in lipid.atoms:
                x_nm = atom.x / 10.0
                y_nm = atom.y / 10.0
                z_nm = atom.z / 10.0
                lines.append(
                    f"{atom.residue_id:5d}{atom.residue_name:<5s}"
                    f"{atom.name:>5s}{atom_id:5d}"
                    f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"
                )
                atom_id += 1

        lines.append(
            f"   {self.box.Lx/10:.5f}   {self.box.Ly/10:.5f}   {self.box.Lz/10:.5f}"
        )

        return "\n".join(lines)

    def write_pdb(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_pdb_string())

    def write_gro(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_gro_string())

    def write_xyz(self, path: str) -> None:
        atoms = self.get_all_atoms()
        lines = [str(len(atoms)), "Membrane built by Membrane Builder"]
        for atom in atoms:
            lines.append(f"{atom.element}  {atom.x:.3f}  {atom.y:.3f}  {atom.z:.3f}")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def calculate_permeability(
        self,
        molecule: "MoleculeDescriptor",
        membrane_type: str = "BLM",
    ) -> "PermeabilityResults":
        """
        Calculate permeability for a molecule through this membrane.

        Args:
            molecule: Molecule descriptor (from permeability module)
            membrane_type: Type of membrane model for calibration.
                Options: "BLM", "PAMPA-DS", "BBB", "Caco-2"

        Returns:
            PermeabilityResults with log_p and energy profile
        """
        from .permeability import PermeabilityPredictor, MembraneType

        # Get composition from config
        composition = {}
        if self.config:
            composition = self.config.composition_dict

        # Get thickness from properties
        thickness = 36.5
        if self.properties:
            thickness = self.properties.thickness

        predictor = PermeabilityPredictor(
            composition=composition,
            membrane_thickness=thickness,
        )

        mt = MembraneType(membrane_type)
        return predictor.calculate(molecule, mt)


class MembraneBuilder:
    """Builds lipid bilayer membranes."""

    def __init__(self, seed: int = 12345, lipid_library: Optional[LipidLibrary] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.library = lipid_library or LipidLibrary()
        self.physics = MembranePhysics()

    def build(
        self,
        config: MembraneConfig,
        use_templates: bool = False,
        templates_dir: Optional[str] = None,
    ) -> BuiltMembrane:
        """Build membrane from config."""
        errors = config.validate()
        for msg in errors:
            if msg.startswith("ERROR"):
                raise ValueError(msg)

        sites_top = self._generate_sites(
            n_sites=config.top_leaflet_count,
            box=config.box,
            min_dist=config.packing.min_anchor_dist,
            jitter=config.packing.jitter,
            seed_offset=1,
        )

        sites_bot = self._generate_sites(
            n_sites=config.bottom_leaflet_count,
            box=config.box,
            min_dist=config.packing.min_anchor_dist,
            jitter=config.packing.jitter,
            seed_offset=2,
        )

        top_assignments = self._assign_lipids_to_sites(
            sites_top, config.lipids, "top"
        )
        bot_assignments = self._assign_lipids_to_sites(
            sites_bot, config.lipids, "bottom"
        )

        placed_lipids = []
        residue_id = 1

        for site, lipid_name in top_assignments:
            lipid = self._place_lipid(
                lipid_name=lipid_name,
                site=site,
                z_anchor=config.leaflets.z_top,
                flip=False,
                residue_id=residue_id,
                leaflet="top",
                use_templates=use_templates,
                templates_dir=templates_dir,
            )
            placed_lipids.append(lipid)
            residue_id += 1

        for site, lipid_name in bot_assignments:
            lipid = self._place_lipid(
                lipid_name=lipid_name,
                site=site,
                z_anchor=config.leaflets.z_bottom,
                flip=True,
                residue_id=residue_id,
                leaflet="bottom",
                use_templates=use_templates,
                templates_dir=templates_dir,
            )
            placed_lipids.append(lipid)
            residue_id += 1

        composition = config.composition_dict
        properties = self.physics.calculate(
            composition=composition,
            box_area=config.box.area,
            bending_modulus_override=config.physics.bending_modulus,
        )

        return BuiltMembrane(
            lipids=placed_lipids,
            box=config.box,
            properties=properties,
            config=config,
        )

    def _generate_sites(
        self,
        n_sites: int,
        box: BoxConfig,
        min_dist: float,
        jitter: float,
        seed_offset: int,
    ) -> np.ndarray:
        """Generate XY anchor sites with jittered grid."""
        rng = np.random.default_rng(self.seed + seed_offset)

        if n_sites <= 0:
            return np.zeros((0, 2), dtype=float)

        nx = int(math.ceil(math.sqrt(n_sites * box.Lx / box.Ly)))
        ny = int(math.ceil(n_sites / nx))

        dx = box.Lx / nx
        dy = box.Ly / ny

        xs = (np.arange(nx) + 0.5) * dx - box.Lx / 2.0
        ys = (np.arange(ny) + 0.5) * dy - box.Ly / 2.0
        grid = np.array([(x, y) for y in ys for x in xs], dtype=float)

        rng.shuffle(grid)
        grid = grid[:n_sites]

        placed = []

        for i in range(n_sites):
            base = grid[i]
            ok = False

            for _ in range(2000):
                trial = base + rng.uniform(-jitter, jitter, size=2)
                trial[0] = ((trial[0] + box.Lx / 2) % box.Lx) - box.Lx / 2
                trial[1] = ((trial[1] + box.Ly / 2) % box.Ly) - box.Ly / 2

                if self._check_min_distance(trial, placed, min_dist, box):
                    placed.append(trial)
                    ok = True
                    break

            if not ok:
                raise RuntimeError(
                    f"Failed to place site {i+1}/{n_sites}. "
                    "Try increasing box size or reducing lipid count."
                )

        return np.array(placed)

    def _check_min_distance(
        self,
        site: np.ndarray,
        existing: List[np.ndarray],
        min_dist: float,
        box: BoxConfig,
    ) -> bool:
        if not existing:
            return True

        existing_arr = np.array(existing)
        d = site[None, :] - existing_arr
        d[:, 0] -= box.Lx * np.round(d[:, 0] / box.Lx)
        d[:, 1] -= box.Ly * np.round(d[:, 1] / box.Ly)

        distances = np.sqrt(np.sum(d * d, axis=1))
        return np.all(distances >= min_dist)

    def _assign_lipids_to_sites(
        self,
        sites: np.ndarray,
        lipid_configs: dict,
        leaflet: str,
    ) -> List[Tuple[np.ndarray, str]]:
        assignments = []
        lipid_list = []
        for name, lip_cfg in lipid_configs.items():
            count = lip_cfg.count_top if leaflet == "top" else lip_cfg.count_bottom
            lipid_list.extend([name] * count)

        self.rng.shuffle(lipid_list)

        for i, site in enumerate(sites):
            if i < len(lipid_list):
                assignments.append((site, lipid_list[i]))

        return assignments

    def _place_lipid(
        self,
        lipid_name: str,
        site: np.ndarray,
        z_anchor: float,
        flip: bool,
        residue_id: int,
        leaflet: str,
        use_templates: bool,
        templates_dir: Optional[str],
    ) -> PlacedLipid:
        lipid_info = self.library.get(lipid_name)
        if lipid_info is None:
            lipid_info = self.library.get("POPC")  # Fallback

        if use_templates and templates_dir:
            atoms = self._load_template_lipid(
                lipid_name, templates_dir, residue_id
            )
        else:
            atoms = self._generate_simplified_lipid(
                lipid_name, lipid_info, residue_id
            )

        anchor_names = ["P", "O3", "C1"]
        anchor_idx = None
        for aname in anchor_names:
            for i, atom in enumerate(atoms):
                if atom.name == aname:
                    anchor_idx = i
                    break
            if anchor_idx is not None:
                break

        if anchor_idx is None:
            anchor_idx = 0

        anchor_pos = np.array([atoms[anchor_idx].x, atoms[anchor_idx].y, atoms[anchor_idx].z])
        for atom in atoms:
            atom.x -= anchor_pos[0]
            atom.y -= anchor_pos[1]
            atom.z -= anchor_pos[2]

        theta = self.rng.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for atom in atoms:
            x_new = cos_t * atom.x - sin_t * atom.y
            y_new = sin_t * atom.x + cos_t * atom.y
            atom.x, atom.y = x_new, y_new

        if flip:
            for atom in atoms:
                atom.y = -atom.y
                atom.z = -atom.z

        final_anchor = np.array([site[0], site[1], z_anchor])
        for atom in atoms:
            atom.x += site[0]
            atom.y += site[1]
            atom.z += z_anchor

        return PlacedLipid(
            lipid_type=lipid_name,
            atoms=atoms,
            anchor_position=final_anchor,
            leaflet=leaflet,
            residue_id=residue_id,
        )

    def _generate_simplified_lipid(
        self,
        lipid_name: str,
        lipid_info: Lipid,
        residue_id: int,
    ) -> List[Atom]:
        """Generate simplified lipid structure."""
        atoms = []
        resname = lipid_name[:3].upper()

        if lipid_info.category.value == "STEROL":
            atoms = self._generate_sterol_atoms(resname, residue_id)
        else:
            atoms = self._generate_phospholipid_atoms(
                resname, residue_id, lipid_info
            )

        return atoms

    def _generate_phospholipid_atoms(
        self,
        resname: str,
        residue_id: int,
        lipid_info: Lipid,
    ) -> List[Atom]:
        atoms = []

        # Headgroup
        atoms.append(Atom("N", "N", 0.0, 0.0, 5.0, resname, residue_id))
        atoms.append(Atom("P", "P", 0.0, 0.0, 0.0, resname, residue_id))
        atoms.append(Atom("O11", "O", 1.5, 0.0, 0.0, resname, residue_id))
        atoms.append(Atom("O12", "O", -1.5, 0.0, 0.0, resname, residue_id))
        atoms.append(Atom("O13", "O", 0.0, 1.2, 0.5, resname, residue_id))
        atoms.append(Atom("O14", "O", 0.0, -1.2, 0.5, resname, residue_id))

        # Glycerol
        atoms.append(Atom("C1", "C", 0.0, 0.0, -3.0, resname, residue_id))
        atoms.append(Atom("C2", "C", 0.0, 1.5, -4.5, resname, residue_id))
        atoms.append(Atom("C3", "C", 0.0, -1.5, -4.5, resname, residue_id))

        atoms.append(Atom("O21", "O", 0.0, 2.5, -5.5, resname, residue_id))
        atoms.append(Atom("O31", "O", 0.0, -2.5, -5.5, resname, residue_id))

        # Tails
        sn1_carbons = lipid_info.tail_carbons[0]
        for i in range(min(sn1_carbons, 16)):
            z = -7.0 - i * 1.25
            x = 0.3 * np.sin(i * 0.5) + 1.5
            atoms.append(Atom(f"C2{i+1}", "C", x, 1.5, z, resname, residue_id))

        sn2_carbons = lipid_info.tail_carbons[1]
        for i in range(min(sn2_carbons, 18)):
            z = -7.0 - i * 1.25
            x = 0.3 * np.sin(i * 0.5 + np.pi) - 1.5
            atoms.append(Atom(f"C3{i+1}", "C", x, -1.5, z, resname, residue_id))

        return atoms

    def _generate_sterol_atoms(
        self,
        resname: str,
        residue_id: int,
    ) -> List[Atom]:
        atoms = []

        atoms.append(Atom("O3", "O", 0.0, 0.0, 0.0, resname, residue_id))
        atoms.append(Atom("C1", "C", 0.0, 1.2, -1.5, resname, residue_id))
        atoms.append(Atom("C2", "C", 1.2, 1.2, -2.5, resname, residue_id))
        atoms.append(Atom("C3", "C", 1.2, 0.0, -3.5, resname, residue_id))
        atoms.append(Atom("C4", "C", 0.0, -1.2, -3.5, resname, residue_id))
        atoms.append(Atom("C5", "C", -1.2, -1.2, -2.5, resname, residue_id))
        atoms.append(Atom("C6", "C", -1.2, 0.0, -1.5, resname, residue_id))
        atoms.append(Atom("C7", "C", 0.0, 0.0, -5.0, resname, residue_id))
        atoms.append(Atom("C8", "C", 1.2, 0.0, -6.0, resname, residue_id))
        atoms.append(Atom("C9", "C", 1.2, -1.2, -7.0, resname, residue_id))
        atoms.append(Atom("C10", "C", 0.0, -1.2, -8.0, resname, residue_id))
        atoms.append(Atom("C11", "C", -1.2, 0.0, -8.0, resname, residue_id))
        atoms.append(Atom("C12", "C", -1.2, 1.2, -9.0, resname, residue_id))
        atoms.append(Atom("C13", "C", 0.0, 1.2, -10.0, resname, residue_id))
        atoms.append(Atom("C14", "C", 1.2, 0.0, -10.0, resname, residue_id))
        atoms.append(Atom("C15", "C", 0.0, 0.0, -11.5, resname, residue_id))
        atoms.append(Atom("C16", "C", 0.0, 0.0, -13.0, resname, residue_id))
        atoms.append(Atom("C17", "C", 0.0, 0.0, -14.5, resname, residue_id))
        atoms.append(Atom("C18", "C", 1.5, 0.0, -15.5, resname, residue_id))
        atoms.append(Atom("C19", "C", 1.5, 0.0, -17.0, resname, residue_id))
        atoms.append(Atom("C20", "C", 0.0, 0.0, -18.0, resname, residue_id))
        atoms.append(Atom("C21", "C", -1.5, 0.0, -6.0, resname, residue_id))
        atoms.append(Atom("C22", "C", 2.5, 0.0, -9.0, resname, residue_id))

        return atoms

    def _load_template_lipid(
        self,
        lipid_name: str,
        templates_dir: str,
        residue_id: int,
    ) -> List[Atom]:
        pattern = os.path.join(templates_dir, lipid_name.lower(), "conf*", "*.crd")
        files = glob.glob(pattern)

        if not files:
            pattern = os.path.join(templates_dir, lipid_name.lower(), "conf*", "*.pdb")
            files = glob.glob(pattern)

        if not files:
            lipid_info = self.library.get(lipid_name) or self.library.get("POPC")
            return self._generate_simplified_lipid(lipid_name, lipid_info, residue_id)

        conf_file = self.rng.choice(files)

        return self._parse_coordinate_file(conf_file, residue_id)

    def _parse_coordinate_file(
        self,
        filepath: str,
        residue_id: int,
    ) -> List[Atom]:
        atoms = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdb":
            for line in lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    name = line[12:16].strip()
                    resname = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip() if len(line) > 76 else name[0]
                    atoms.append(Atom(name, element, x, y, z, resname, residue_id))

        elif ext == ".crd":
            for line in lines:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        name = parts[4]
                        resname = parts[3]
                        x = float(parts[5])
                        y = float(parts[6])
                        z = float(parts[7])
                        element = name[0]
                        atoms.append(Atom(name, element, x, y, z, resname, residue_id))
                    except (ValueError, IndexError):
                        continue

        return atoms

    @classmethod
    def quick_build(
        cls,
        lipids: Dict[str, Tuple[int, int]],
        box_size: Tuple[float, float] = (80.0, 80.0),
        box_height: float = 120.0,
        bending_modulus: Optional[float] = None,
        seed: int = 12345,
    ) -> BuiltMembrane:
        """Build membrane with minimal config."""
        config = MembraneConfig.create_simple(
            lipids=lipids,
            box_size=(box_size[0], box_size[1], box_height),
            bending_modulus=bending_modulus,
        )

        builder = cls(seed=seed)
        return builder.build(config)
