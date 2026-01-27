"""
Configuration for membrane building.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os
from pathlib import Path


@dataclass
class BoxConfig:
    Lx: float = 80.0
    Ly: float = 80.0
    Lz: float = 120.0

    def to_dict(self) -> dict:
        return {"Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz}

    @classmethod
    def from_dict(cls, data: dict) -> "BoxConfig":
        return cls(**data)

    @property
    def area(self) -> float:
        return self.Lx * self.Ly

    @property
    def volume(self) -> float:
        return self.Lx * self.Ly * self.Lz


@dataclass
class LeafletConfig:
    z_top: float = 18.0
    z_bottom: float = -18.0

    def to_dict(self) -> dict:
        return {"z_top": self.z_top, "z_bottom": self.z_bottom}

    @classmethod
    def from_dict(cls, data: dict) -> "LeafletConfig":
        return cls(**data)

    @property
    def separation(self) -> float:
        return abs(self.z_top - self.z_bottom)


@dataclass
class LipidConfig:
    name: str
    count_top: int
    count_bottom: int
    path: Optional[str] = None
    file_glob: str = "*.crd"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "file_glob": self.file_glob,
            "count_top": self.count_top,
            "count_bottom": self.count_bottom,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "LipidConfig":
        return cls(
            name=name,
            count_top=data.get("count_top", 0),
            count_bottom=data.get("count_bottom", 0),
            path=data.get("path"),
            file_glob=data.get("file_glob", "*.crd"),
        )

    @property
    def total(self) -> int:
        return self.count_top + self.count_bottom


@dataclass
class PackingConfig:
    min_anchor_dist: float = 6.5
    jitter: float = 2.0

    def to_dict(self) -> dict:
        return {"min_anchor_dist": self.min_anchor_dist, "jitter": self.jitter}

    @classmethod
    def from_dict(cls, data: dict) -> "PackingConfig":
        return cls(
            min_anchor_dist=data.get("min_anchor_dist", 6.5),
            jitter=data.get("jitter", 2.0),
        )


@dataclass
class OutputConfig:
    out_dir: str = "Outputs"
    write_pdb: bool = True
    write_gro: bool = True
    write_psf: bool = False

    def to_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "write_pdb": self.write_pdb,
            "write_gro": self.write_gro,
            "write_psf": self.write_psf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OutputConfig":
        return cls(
            out_dir=data.get("out_dir", "Outputs"),
            write_pdb=data.get("write_pdb", True),
            write_gro=data.get("write_gro", True),
            write_psf=data.get("write_psf", False),
        )


@dataclass
class PhysicsConfig:
    temperature: float = 310.15
    bending_modulus: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "bending_modulus": self.bending_modulus,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicsConfig":
        return cls(
            temperature=data.get("temperature", 310.15),
            bending_modulus=data.get("bending_modulus"),
        )


@dataclass
class MembraneConfig:
    """Complete membrane configuration."""

    project: str = "membrane"
    box: BoxConfig = field(default_factory=BoxConfig)
    leaflets: LeafletConfig = field(default_factory=LeafletConfig)
    lipids: Dict[str, LipidConfig] = field(default_factory=dict)
    packing: PackingConfig = field(default_factory=PackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    def to_dict(self) -> dict:
        lipids_dict = {}
        for name, cfg in self.lipids.items():
            lipids_dict[name] = cfg.to_dict()

        return {
            "project": self.project,
            "box": self.box.to_dict(),
            "leaflets": self.leaflets.to_dict(),
            "lipids": lipids_dict,
            "packing": self.packing.to_dict(),
            "output": self.output.to_dict(),
            "physics": self.physics.to_dict(),
        }

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "MembraneConfig":
        lipids = {}
        for name, lipid_data in data.get("lipids", {}).items():
            lipids[name] = LipidConfig.from_dict(name, lipid_data)

        return cls(
            project=data.get("project", "membrane"),
            box=BoxConfig.from_dict(data.get("box", {})),
            leaflets=LeafletConfig.from_dict(data.get("leaflets", {})),
            lipids=lipids,
            packing=PackingConfig.from_dict(data.get("packing", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
            physics=PhysicsConfig.from_dict(data.get("physics", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MembraneConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def create_simple(
        cls,
        lipids: Dict[str, Tuple[int, int]],
        box_size: Tuple[float, float, float] = (80.0, 80.0, 120.0),
        project_name: str = "membrane",
        output_dir: str = "Outputs",
        bending_modulus: Optional[float] = None,
    ) -> "MembraneConfig":
        """
        Quick config creation.

        lipids: {name: (top_count, bottom_count)}
        box_size: (Lx, Ly, Lz) in Angstroms
        """
        lipid_configs = {}
        for name, (top, bottom) in lipids.items():
            lipid_configs[name] = LipidConfig(
                name=name,
                count_top=top,
                count_bottom=bottom,
                path=f"Lipids/{name.lower()}/conf1",
                file_glob=f"{name.lower()}_*.crd",
            )

        return cls(
            project=project_name,
            box=BoxConfig(Lx=box_size[0], Ly=box_size[1], Lz=box_size[2]),
            leaflets=LeafletConfig(),
            lipids=lipid_configs,
            packing=PackingConfig(),
            output=OutputConfig(out_dir=output_dir),
            physics=PhysicsConfig(bending_modulus=bending_modulus),
        )

    @property
    def total_lipids(self) -> int:
        return sum(lip.total for lip in self.lipids.values())

    @property
    def top_leaflet_count(self) -> int:
        return sum(lip.count_top for lip in self.lipids.values())

    @property
    def bottom_leaflet_count(self) -> int:
        return sum(lip.count_bottom for lip in self.lipids.values())

    @property
    def composition_dict(self) -> Dict[str, int]:
        return {name: lip.total for name, lip in self.lipids.items()}

    def validate(self) -> List[str]:
        """Check config and return any issues."""
        from .lipids import LipidLibrary
        library = LipidLibrary()

        messages = []

        if self.box.Lx <= 0 or self.box.Ly <= 0 or self.box.Lz <= 0:
            messages.append("ERROR: Box dimensions must be positive")

        for name in self.lipids:
            if name.upper() not in library:
                messages.append(f"WARNING: Unknown lipid '{name}'")

        if self.total_lipids == 0:
            messages.append("ERROR: No lipids specified")

        top = self.top_leaflet_count
        bottom = self.bottom_leaflet_count
        if abs(top - bottom) > max(top, bottom) * 0.2:
            messages.append(f"WARNING: Leaflet asymmetry: {top} top vs {bottom} bottom")

        area = self.box.area
        min_needed = self.top_leaflet_count * 40
        if min_needed > area:
            messages.append(f"ERROR: Box too small for {self.top_leaflet_count} lipids/leaflet")

        if self.packing.min_anchor_dist < 4.0:
            messages.append("WARNING: min_anchor_dist < 4 A may cause clashes")

        return messages

    def summary(self) -> str:
        lines = [
            "=" * 50,
            f"CONFIG: {self.project}",
            "=" * 50,
            "",
            f"Box: {self.box.Lx} x {self.box.Ly} x {self.box.Lz} A",
            f"Area: {self.box.area:.0f} A^2",
            "",
            "Lipids:",
        ]

        for name, lip in self.lipids.items():
            lines.append(f"  {name}: {lip.count_top} top, {lip.count_bottom} bottom")

        lines.extend([
            "",
            f"Total: {self.total_lipids} lipids",
            "=" * 50,
        ])

        return "\n".join(lines)
