#!/usr/bin/env python3
"""
TPU Membrane MD Simulation Pipeline

Complete workflow for:
1. Building polymer structures
2. Running GROMACS MD simulations
3. Analyzing trajectories
4. Training ML models with MD descriptors

Usage:
    # Set up a new simulation
    python run_pipeline.py setup --sparsa1 0.3 --carbosil1 0.7

    # Analyze completed simulation
    python run_pipeline.py analyze --dir ./md_pipeline/simulation

    # Train ML model with experimental data
    python run_pipeline.py train

    # Predict permeability for new composition
    python run_pipeline.py predict --sparsa1 0.5 --carbosil1 0.5 --molecule phenol
"""

import argparse
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_simulation(args):
    """Set up a new MD simulation"""
    from polymer_builder import generate_membrane_structure
    from gromacs_setup import setup_gromacs_simulation

    composition = {
        'Sparsa1': args.sparsa1,
        'Sparsa2': args.sparsa2,
        'Carbosil1': args.carbosil1,
        'Carbosil2': args.carbosil2
    }

    # Normalize
    total = sum(composition.values())
    if total > 0:
        composition = {k: v/total for k, v in composition.items()}

    print("=" * 60)
    print("TPU Membrane MD Simulation Setup")
    print("=" * 60)
    print(f"\nComposition:")
    for k, v in composition.items():
        if v > 0:
            print(f"  {k}: {v*100:.1f}%")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Generate structure
    print("\n--- Generating polymer structure ---")
    structure_dir = os.path.join(output_dir, "structures")
    pdb_file = generate_membrane_structure(
        composition=composition,
        output_dir=structure_dir,
        box_size=(args.box_size, args.box_size, args.box_size),
        n_chains=args.chains
    )

    # Set up GROMACS
    print("\n--- Setting up GROMACS files ---")
    sim_dir = os.path.join(output_dir, "simulation")
    setup_gromacs_simulation(
        structure_file=pdb_file,
        output_dir=sim_dir,
        temperature=args.temperature,
        production_ns=args.ns
    )

    # Copy structure file
    import shutil
    shutil.copy(pdb_file, sim_dir)

    # Save config
    config = {
        'composition': composition,
        'box_size': args.box_size,
        'n_chains': args.chains,
        'temperature': args.temperature,
        'production_ns': args.ns,
        'structure_file': pdb_file,
        'simulation_dir': sim_dir
    }

    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nSimulation directory: {sim_dir}")
    print(f"\nTo run the simulation:")
    print(f"  cd {sim_dir}")
    print(f"  ./run_simulation.sh")
    print(f"\nOr run each step manually:")
    print(f"  gmx pdb2gmx -f <structure>.pdb -o processed.gro -water none")
    print(f"  gmx editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic")
    print(f"  gmx grompp -f minim.mdp -c box.gro -p topol.top -o em.tpr")
    print(f"  gmx mdrun -v -deffnm em")
    print(f"  ... (see run_simulation.sh for full workflow)")


def analyze_simulation(args):
    """Analyze a completed MD simulation"""
    from md_analysis import analyze_trajectory

    print("=" * 60)
    print("MD Trajectory Analysis")
    print("=" * 60)

    sim_dir = args.dir

    # Check if files exist
    required_files = ['md.tpr', 'md.xtc', 'md.gro', 'md.edr']
    missing = [f for f in required_files if not os.path.exists(os.path.join(sim_dir, f))]

    if missing:
        print(f"\nWarning: Missing files: {missing}")
        print("Analysis may be incomplete.")

    result = analyze_trajectory(sim_dir)

    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"Density:              {result.density:.3f} ± {result.density_std:.3f} g/cm³")
    print(f"Free volume fraction: {result.free_volume_fraction:.4f}")
    print(f"Diffusion coeff:      {result.diffusion_coefficient:.2e} ± {result.diffusion_std:.2e} cm²/s")
    print(f"Radius of gyration:   {result.radius_of_gyration:.2f} nm")
    print(f"End-to-end distance:  {result.end_to_end_distance:.2f} nm")
    print(f"Temperature:          {result.temperature:.1f} K")
    print(f"Pressure:             {result.pressure:.1f} bar")
    print(f"Box volume:           {result.box_volume:.0f} nm³")

    # Save results
    results_file = os.path.join(sim_dir, "analysis_results.json")
    results_dict = {
        'density': result.density,
        'density_std': result.density_std,
        'free_volume_fraction': result.free_volume_fraction,
        'diffusion_coefficient': result.diffusion_coefficient,
        'diffusion_std': result.diffusion_std,
        'radius_of_gyration': result.radius_of_gyration,
        'end_to_end_distance': result.end_to_end_distance,
        'temperature': result.temperature,
        'pressure': result.pressure,
        'box_volume': result.box_volume
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_file}")


def train_model(args):
    """Train and evaluate the ML model"""
    from ml_permeability_model import train_and_evaluate
    train_and_evaluate()


def predict_permeability(args):
    """Predict permeability for a composition"""
    from ml_permeability_model import PermeabilityPredictor

    composition = {
        'Sparsa1': args.sparsa1,
        'Sparsa2': args.sparsa2,
        'Carbosil1': args.carbosil1,
        'Carbosil2': args.carbosil2
    }

    # Normalize
    total = sum(composition.values())
    if total > 0:
        composition = {k: v/total for k, v in composition.items()}

    predictor = PermeabilityPredictor()

    print("=" * 60)
    print("Permeability Prediction")
    print("=" * 60)
    print(f"\nComposition:")
    for k, v in composition.items():
        if v > 0:
            print(f"  {k}: {v*100:.1f}%")
    print(f"Thickness: {args.thickness} um")

    print(f"\nPredictions:")
    print("-" * 40)

    molecules = [args.molecule] if args.molecule != 'all' else ['phenol', 'm-cresol', 'glucose']

    for mol in molecules:
        try:
            result = predictor.predict(mol, composition, args.thickness)
            print(f"{mol:12s}: {result['permeability_cm2_s']:.2e} cm²/s "
                  f"(log P = {result['log_permeability']:.2f})")
        except Exception as e:
            print(f"{mol:12s}: Error - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TPU Membrane MD Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up new simulation')
    setup_parser.add_argument('--sparsa1', type=float, default=0.3,
                             help='Sparsa 1 fraction (default: 0.3)')
    setup_parser.add_argument('--sparsa2', type=float, default=0.0,
                             help='Sparsa 2 fraction (default: 0.0)')
    setup_parser.add_argument('--carbosil1', type=float, default=0.7,
                             help='Carbosil 1 fraction (default: 0.7)')
    setup_parser.add_argument('--carbosil2', type=float, default=0.0,
                             help='Carbosil 2 fraction (default: 0.0)')
    setup_parser.add_argument('--output', '-o', type=str, default='./md_pipeline',
                             help='Output directory (default: ./md_pipeline)')
    setup_parser.add_argument('--box-size', type=float, default=50.0,
                             help='Box size in Angstroms (default: 50)')
    setup_parser.add_argument('--chains', type=int, default=20,
                             help='Number of polymer chains (default: 20)')
    setup_parser.add_argument('--temperature', '-T', type=float, default=310.0,
                             help='Temperature in K (default: 310)')
    setup_parser.add_argument('--ns', type=float, default=10.0,
                             help='Production run length in ns (default: 10)')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze completed simulation')
    analyze_parser.add_argument('--dir', '-d', type=str, required=True,
                               help='Simulation directory')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict permeability')
    predict_parser.add_argument('--sparsa1', type=float, default=0.5,
                               help='Sparsa 1 fraction')
    predict_parser.add_argument('--sparsa2', type=float, default=0.0,
                               help='Sparsa 2 fraction')
    predict_parser.add_argument('--carbosil1', type=float, default=0.5,
                               help='Carbosil 1 fraction')
    predict_parser.add_argument('--carbosil2', type=float, default=0.0,
                               help='Carbosil 2 fraction')
    predict_parser.add_argument('--thickness', '-t', type=float, default=200.0,
                               help='Thickness in um (default: 200)')
    predict_parser.add_argument('--molecule', '-m', type=str, default='all',
                               choices=['phenol', 'm-cresol', 'glucose', 'all'],
                               help='Molecule to predict (default: all)')

    args = parser.parse_args()

    if args.command == 'setup':
        setup_simulation(args)
    elif args.command == 'analyze':
        analyze_simulation(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict_permeability(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
