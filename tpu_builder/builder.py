"""
TPU Membrane Builder

Builds thermoplastic polyurethane membranes from CarboSil and Sparsa polymers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .polymers import PolymerLibrary, PolymerProperties, calculate_blend_properties


@dataclass
class TPUMembraneConfig:
    """Configuration for TPU membrane construction"""

    # Polymer composition (weight fractions, must sum to 1)
    polymers: Dict[str, float] = field(default_factory=lambda: {"CarboSil": 1.0})

    # Membrane dimensions
    thickness: float = 100.0  # micrometers
    width: float = 10.0  # mm
    height: float = 10.0  # mm

    # Processing parameters
    annealing_temp: float = 80.0  # Celsius
    processing_method: str = "solvent_cast"  # solvent_cast, extrusion, electrospinning

    @classmethod
    def create_carbosil(cls, thickness: float = 100.0) -> 'TPUMembraneConfig':
        """Create pure CarboSil membrane"""
        return cls(polymers={"CarboSil": 1.0}, thickness=thickness)

    @classmethod
    def create_sparsa(cls, thickness: float = 100.0) -> 'TPUMembraneConfig':
        """Create pure Sparsa membrane"""
        return cls(polymers={"Sparsa": 1.0}, thickness=thickness)

    @classmethod
    def create_blend(
        cls,
        carbosil_fraction: float,
        sparsa_fraction: float,
        thickness: float = 100.0
    ) -> 'TPUMembraneConfig':
        """Create CarboSil/Sparsa blend membrane"""
        total = carbosil_fraction + sparsa_fraction
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Fractions must sum to 1, got {total}")
        return cls(
            polymers={"CarboSil": carbosil_fraction, "Sparsa": sparsa_fraction},
            thickness=thickness
        )


@dataclass
class MembraneProperties:
    """Calculated properties of a TPU membrane"""

    # Composition
    polymer_composition: Dict[str, float]
    thickness_um: float

    # Physical properties
    density: float  # g/cm³
    water_uptake: float  # %
    free_volume_fraction: float
    crystallinity: float
    hydrophilicity: float
    soft_segment_fraction: float

    # Derived transport properties
    diffusivity_factor: float  # Relative to pure PDMS
    solubility_factor: float  # Relative to water
    permeability_factor: float  # Overall transport enhancement

    # Mechanical (estimated)
    estimated_modulus: float  # MPa
    estimated_elongation: float  # %


class TPUMembrane:
    """Represents a constructed TPU membrane"""

    def __init__(
        self,
        config: TPUMembraneConfig,
        properties: MembraneProperties,
        structure: Optional[np.ndarray] = None
    ):
        self.config = config
        self.properties = properties
        self._structure = structure

    @property
    def thickness(self) -> float:
        return self.config.thickness

    @property
    def composition(self) -> Dict[str, float]:
        return self.config.polymers

    def get_structure(self) -> np.ndarray:
        """Get the membrane structure array"""
        if self._structure is None:
            raise ValueError("No structure generated")
        return self._structure

    def to_dict(self) -> dict:
        """Export membrane data as dictionary"""
        return {
            'composition': self.config.polymers,
            'thickness_um': self.config.thickness,
            'width_mm': self.config.width,
            'height_mm': self.config.height,
            'properties': {
                'density': self.properties.density,
                'water_uptake': self.properties.water_uptake,
                'free_volume': self.properties.free_volume_fraction,
                'crystallinity': self.properties.crystallinity,
                'hydrophilicity': self.properties.hydrophilicity,
                'soft_segment_fraction': self.properties.soft_segment_fraction,
                'permeability_factor': self.properties.permeability_factor
            }
        }

    def write_report(self, filepath: str):
        """Write membrane report to file"""
        with open(filepath, 'w') as f:
            f.write("TPU Membrane Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Composition:\n")
            for polymer, fraction in self.config.polymers.items():
                f.write(f"  {polymer}: {fraction*100:.1f}%\n")

            f.write(f"\nDimensions:\n")
            f.write(f"  Thickness: {self.config.thickness} µm\n")
            f.write(f"  Width: {self.config.width} mm\n")
            f.write(f"  Height: {self.config.height} mm\n")

            f.write(f"\nPhysical Properties:\n")
            f.write(f"  Density: {self.properties.density:.3f} g/cm³\n")
            f.write(f"  Water uptake: {self.properties.water_uptake:.1f}%\n")
            f.write(f"  Free volume: {self.properties.free_volume_fraction:.3f}\n")
            f.write(f"  Crystallinity: {self.properties.crystallinity:.3f}\n")
            f.write(f"  Hydrophilicity: {self.properties.hydrophilicity:.2f}\n")

            f.write(f"\nSegment Composition:\n")
            f.write(f"  Soft segment: {self.properties.soft_segment_fraction*100:.1f}%\n")
            f.write(f"  Hard segment: {(1-self.properties.soft_segment_fraction)*100:.1f}%\n")

            f.write(f"\nTransport Properties:\n")
            f.write(f"  Diffusivity factor: {self.properties.diffusivity_factor:.3f}\n")
            f.write(f"  Solubility factor: {self.properties.solubility_factor:.3f}\n")
            f.write(f"  Permeability factor: {self.properties.permeability_factor:.3f}\n")

            f.write(f"\nMechanical Properties (estimated):\n")
            f.write(f"  Modulus: {self.properties.estimated_modulus:.1f} MPa\n")
            f.write(f"  Elongation: {self.properties.estimated_elongation:.0f}%\n")


class TPUMembraneBuilder:
    """Builder for TPU membranes"""

    def __init__(self, seed: Optional[int] = None):
        self.library = PolymerLibrary()
        self.rng = np.random.default_rng(seed)

    def build(self, config: TPUMembraneConfig) -> TPUMembrane:
        """
        Build a TPU membrane from configuration

        Args:
            config: TPUMembraneConfig specifying composition and dimensions

        Returns:
            TPUMembrane object
        """
        # Calculate blend properties
        blend_props = calculate_blend_properties(config.polymers, self.library)

        # Calculate transport properties
        transport = self._calculate_transport_properties(blend_props, config)

        # Calculate mechanical properties
        mechanical = self._calculate_mechanical_properties(config)

        # Create properties object
        properties = MembraneProperties(
            polymer_composition=config.polymers,
            thickness_um=config.thickness,
            density=blend_props['density'],
            water_uptake=blend_props['water_uptake'],
            free_volume_fraction=blend_props['free_volume_fraction'],
            crystallinity=blend_props['crystallinity'],
            hydrophilicity=blend_props['hydrophilicity'],
            soft_segment_fraction=blend_props['soft_segment_fraction'],
            diffusivity_factor=transport['diffusivity'],
            solubility_factor=transport['solubility'],
            permeability_factor=transport['permeability'],
            estimated_modulus=mechanical['modulus'],
            estimated_elongation=mechanical['elongation']
        )

        # Generate structure representation
        structure = self._generate_structure(config)

        return TPUMembrane(config, properties, structure)

    def _calculate_transport_properties(
        self,
        blend_props: dict,
        config: TPUMembraneConfig
    ) -> dict:
        """Calculate transport properties based on blend composition"""

        # Free volume theory for diffusivity
        # D ~ exp(-B/vf) where vf is free volume fraction
        vf = blend_props['free_volume_fraction']
        B = 0.5  # Empirical constant
        diffusivity = np.exp(-B / max(vf, 0.01))

        # Solubility depends on hydrophilicity and water uptake
        hydro = blend_props['hydrophilicity']
        water = blend_props['water_uptake']
        solubility = 0.3 + 0.7 * hydro + 0.02 * water

        # Permeability = D * S (solution-diffusion model)
        # Adjust for crystallinity (crystalline regions are impermeable)
        cryst = blend_props['crystallinity']
        permeability = diffusivity * solubility * (1 - cryst)

        # Normalize relative to pure PDMS
        permeability = permeability / 0.1  # PDMS reference

        return {
            'diffusivity': diffusivity,
            'solubility': solubility,
            'permeability': permeability
        }

    def _calculate_mechanical_properties(self, config: TPUMembraneConfig) -> dict:
        """Estimate mechanical properties from composition"""

        modulus = 0
        elongation = 0

        for name, fraction in config.polymers.items():
            poly = self.library.get(name)
            # Simple rule of mixtures
            if name == "CarboSil":
                modulus += fraction * 15  # Higher modulus
                elongation += fraction * 400
            else:  # Sparsa
                modulus += fraction * 8  # Lower modulus
                elongation += fraction * 600

        return {
            'modulus': modulus,
            'elongation': elongation
        }

    def _generate_structure(self, config: TPUMembraneConfig) -> np.ndarray:
        """
        Generate a simplified structure representation

        Returns 3D array representing phase distribution:
        0 = soft segment
        1 = hard segment
        Values between represent interfaces
        """
        # Create a simple 2D representation of phase separation
        nx = 100
        ny = 100
        nz = int(config.thickness / 10)  # 10 nm resolution

        structure = np.zeros((nx, ny, nz))

        # Generate random hard segment domains
        hard_fraction = 0
        for name, fraction in config.polymers.items():
            poly = self.library.get(name)
            hard_fraction += fraction * poly.hard_segment_fraction

        # Randomly place hard domains
        n_domains = int(nx * ny * nz * hard_fraction * 0.1)
        for _ in range(n_domains):
            x = self.rng.integers(0, nx)
            y = self.rng.integers(0, ny)
            z = self.rng.integers(0, nz)
            # Domain size varies with crystallinity
            size = self.rng.integers(2, 8)
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    for dz in range(-size // 2, size // 2 + 1):
                        xi = (x + dx) % nx
                        yi = (y + dy) % ny
                        zi = min(max(z + dz, 0), nz - 1)
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist <= size:
                            structure[xi, yi, zi] = max(
                                structure[xi, yi, zi],
                                1 - dist / size
                            )

        return structure

    @classmethod
    def quick_build(
        cls,
        polymers: Dict[str, float],
        thickness: float = 100.0
    ) -> TPUMembrane:
        """Quick build method for simple membrane creation"""
        config = TPUMembraneConfig(polymers=polymers, thickness=thickness)
        builder = cls()
        return builder.build(config)
