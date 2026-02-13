"""
TPU Polymer definitions for CarboSil and Sparsa

CarboSil: Segmented silicone-polycarbonate polyurethane
- Soft segments: PDMS (polydimethylsiloxane) diol
- Hard segments: Aliphatic polycarbonate diol
- Linkages: Urethane bonds (HDI or H12MDI-type)
- Properties: High mechanical stiffness, lower permeability, reduced swelling

Sparsa: Amphiphilic polyurethane with flexible soft segment
- Soft segment: Polyether diol (PEG/PPG-like) or very short PDMS diol
- Hard segment: Urethane domains (lower content than CarboSil)
- Properties: Higher permeability, hydrated but not dissolving, amphiphilic
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import numpy as np


class PolymerType(Enum):
    CARBOSIL = "carbosil"
    SPARSA = "sparsa"


@dataclass
class SegmentProperties:
    """Properties of a polymer segment"""
    name: str
    molecular_weight: float  # Da
    glass_transition: float  # Tg in Celsius
    density: float  # g/cm³
    solubility_parameter: float  # (J/cm³)^0.5


@dataclass
class PolymerProperties:
    """Physical properties of a TPU polymer"""
    name: str
    polymer_type: PolymerType

    # Segment composition
    soft_segment_fraction: float  # 0-1
    hard_segment_fraction: float  # 0-1

    # Bulk properties
    density: float  # g/cm³
    water_uptake: float  # % by weight
    shore_hardness: float  # Shore A
    tensile_strength: float  # MPa
    elongation: float  # % at break

    # Permeability-related
    free_volume_fraction: float  # 0-1
    crystallinity: float  # 0-1
    hydrophilicity: float  # 0-1 scale

    # Segment details
    soft_segment: SegmentProperties
    hard_segment: SegmentProperties


# CarboSil polymer definition
CARBOSIL_SOFT = SegmentProperties(
    name="PDMS",
    molecular_weight=2000,  # Da, typical PDMS diol
    glass_transition=-120,  # Very low Tg
    density=0.97,
    solubility_parameter=15.5
)

CARBOSIL_HARD = SegmentProperties(
    name="Polycarbonate-urethane",
    molecular_weight=800,
    glass_transition=60,
    density=1.20,
    solubility_parameter=21.0
)

CarboSil = PolymerProperties(
    name="CarboSil",
    polymer_type=PolymerType.CARBOSIL,
    soft_segment_fraction=0.65,
    hard_segment_fraction=0.35,
    density=1.05,
    water_uptake=0.5,  # Low water uptake
    shore_hardness=80,
    tensile_strength=35,
    elongation=400,
    free_volume_fraction=0.03,
    crystallinity=0.15,
    hydrophilicity=0.2,
    soft_segment=CARBOSIL_SOFT,
    hard_segment=CARBOSIL_HARD
)


# Sparsa polymer definition
SPARSA_SOFT = SegmentProperties(
    name="Polyether",
    molecular_weight=1500,  # Shorter than CarboSil
    glass_transition=-60,
    density=1.05,
    solubility_parameter=18.5
)

SPARSA_HARD = SegmentProperties(
    name="Urethane",
    molecular_weight=400,  # Lower hard segment content
    glass_transition=80,
    density=1.25,
    solubility_parameter=23.0
)

Sparsa = PolymerProperties(
    name="Sparsa",
    polymer_type=PolymerType.SPARSA,
    soft_segment_fraction=0.75,
    hard_segment_fraction=0.25,
    density=1.08,
    water_uptake=8.0,  # Higher water uptake - amphiphilic
    shore_hardness=60,
    tensile_strength=25,
    elongation=600,
    free_volume_fraction=0.08,
    crystallinity=0.05,
    hydrophilicity=0.6,
    soft_segment=SPARSA_SOFT,
    hard_segment=SPARSA_HARD
)

# Sparsa 1 - Original Sparsa formulation (higher permeability)
Sparsa1 = PolymerProperties(
    name="Sparsa1",
    polymer_type=PolymerType.SPARSA,
    soft_segment_fraction=0.75,
    hard_segment_fraction=0.25,
    density=1.08,
    water_uptake=8.0,
    shore_hardness=60,
    tensile_strength=25,
    elongation=600,
    free_volume_fraction=0.08,
    crystallinity=0.05,
    hydrophilicity=0.6,
    soft_segment=SPARSA_SOFT,
    hard_segment=SPARSA_HARD
)

# Sparsa 2 - Modified Sparsa formulation (different permeability profile)
Sparsa2 = PolymerProperties(
    name="Sparsa2",
    polymer_type=PolymerType.SPARSA,
    soft_segment_fraction=0.70,
    hard_segment_fraction=0.30,
    density=1.10,
    water_uptake=6.0,
    shore_hardness=65,
    tensile_strength=28,
    elongation=550,
    free_volume_fraction=0.06,
    crystallinity=0.08,
    hydrophilicity=0.5,
    soft_segment=SPARSA_SOFT,
    hard_segment=SPARSA_HARD
)

# Carbosil 1 - Original Carbosil formulation
Carbosil1 = PolymerProperties(
    name="Carbosil1",
    polymer_type=PolymerType.CARBOSIL,
    soft_segment_fraction=0.65,
    hard_segment_fraction=0.35,
    density=1.05,
    water_uptake=0.5,
    shore_hardness=80,
    tensile_strength=35,
    elongation=400,
    free_volume_fraction=0.03,
    crystallinity=0.15,
    hydrophilicity=0.2,
    soft_segment=CARBOSIL_SOFT,
    hard_segment=CARBOSIL_HARD
)

# Carbosil 2 - Modified Carbosil formulation
Carbosil2 = PolymerProperties(
    name="Carbosil2",
    polymer_type=PolymerType.CARBOSIL,
    soft_segment_fraction=0.60,
    hard_segment_fraction=0.40,
    density=1.07,
    water_uptake=0.3,
    shore_hardness=85,
    tensile_strength=38,
    elongation=350,
    free_volume_fraction=0.025,
    crystallinity=0.18,
    hydrophilicity=0.15,
    soft_segment=CARBOSIL_SOFT,
    hard_segment=CARBOSIL_HARD
)


class PolymerLibrary:
    """Library of available TPU polymers"""

    def __init__(self):
        self._polymers = {
            "CarboSil": CarboSil,
            "Sparsa": Sparsa,
            "Sparsa1": Sparsa1,
            "Sparsa2": Sparsa2,
            "Carbosil1": Carbosil1,
            "Carbosil2": Carbosil2
        }

    def get(self, name: str) -> PolymerProperties:
        """Get polymer by name"""
        if name not in self._polymers:
            raise ValueError(f"Unknown polymer: {name}. Available: {list(self._polymers.keys())}")
        return self._polymers[name]

    def list_polymers(self) -> list:
        """List all available polymers"""
        return list(self._polymers.keys())

    def add_polymer(self, polymer: PolymerProperties):
        """Add a custom polymer to the library"""
        self._polymers[polymer.name] = polymer


def calculate_blend_properties(
    polymers: Dict[str, float],
    library: Optional[PolymerLibrary] = None
) -> dict:
    """
    Calculate properties of a polymer blend

    Args:
        polymers: Dict of polymer name -> weight fraction (must sum to 1)
        library: PolymerLibrary instance

    Returns:
        Dict of blended properties
    """
    if library is None:
        library = PolymerLibrary()

    total_fraction = sum(polymers.values())
    if abs(total_fraction - 1.0) > 0.01:
        raise ValueError(f"Polymer fractions must sum to 1, got {total_fraction}")

    # Initialize weighted properties
    density = 0
    water_uptake = 0
    free_volume = 0
    crystallinity = 0
    hydrophilicity = 0
    soft_fraction = 0

    for name, fraction in polymers.items():
        poly = library.get(name)
        density += fraction * poly.density
        water_uptake += fraction * poly.water_uptake
        free_volume += fraction * poly.free_volume_fraction
        crystallinity += fraction * poly.crystallinity
        hydrophilicity += fraction * poly.hydrophilicity
        soft_fraction += fraction * poly.soft_segment_fraction

    return {
        'density': density,
        'water_uptake': water_uptake,
        'free_volume_fraction': free_volume,
        'crystallinity': crystallinity,
        'hydrophilicity': hydrophilicity,
        'soft_segment_fraction': soft_fraction
    }
