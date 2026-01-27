"""
Membrane Builder - builds lipid bilayers for MD simulations.
"""

__version__ = "1.0.0"

from .lipids import LipidLibrary, Lipid
from .builder import MembraneBuilder
from .physics import MembranePhysics
from .config import MembraneConfig
from .permeability import (
    PermeabilityPredictor,
    MembraneProfileGenerator,
    MoleculeDescriptor,
    PermeabilityResults,
    quick_permeability,
)

__all__ = [
    "LipidLibrary",
    "Lipid",
    "MembraneBuilder",
    "MembranePhysics",
    "MembraneConfig",
    "PermeabilityPredictor",
    "MembraneProfileGenerator",
    "MoleculeDescriptor",
    "PermeabilityResults",
    "quick_permeability",
]
