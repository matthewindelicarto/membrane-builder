"""
MD Simulation Pipeline for TPU Membranes

Modules:
- polymer_builder: Generate all-atom polymer structures
- gromacs_setup: Create GROMACS input files
- md_analysis: Analyze MD trajectories
- ml_permeability_model: Hybrid ML model for permeability

Usage:
    # Set up simulation
    python run_pipeline.py setup --sparsa1 0.3 --carbosil1 0.7

    # Predict permeability
    python run_pipeline.py predict --sparsa1 0.5 --carbosil1 0.5

    # Train/evaluate model
    python run_pipeline.py train
"""

from .polymer_builder import (
    TPUMembraneBuilder,
    PolymerChainBuilder,
    generate_membrane_structure
)

from .gromacs_setup import (
    setup_gromacs_simulation,
    write_topology,
    write_mdp_minimization,
    write_mdp_nvt,
    write_mdp_npt,
    write_mdp_production
)

from .md_analysis import (
    GMXAnalyzer,
    MDAnalysisResult,
    analyze_trajectory
)

from .ml_permeability_model import (
    HybridPermeabilityModel,
    PermeabilityPredictor,
    MDDescriptors
)

__all__ = [
    'TPUMembraneBuilder',
    'PolymerChainBuilder',
    'generate_membrane_structure',
    'setup_gromacs_simulation',
    'GMXAnalyzer',
    'MDAnalysisResult',
    'analyze_trajectory',
    'HybridPermeabilityModel',
    'PermeabilityPredictor',
    'MDDescriptors'
]
