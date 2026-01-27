"""
Main runner script for membrane permeability analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Dict, List, Optional

from membrane_builder import (
    MembraneBuilder,
    MoleculeDescriptor,
    PermeabilityPredictor,
)

from .visualize import PermeabilityVisualizer
from .analyze import PermeabilityAnalyzer, print_permeability_table


def run_permeability_analysis(
    molecules: Optional[List[MoleculeDescriptor]] = None,
    composition: Optional[Dict[str, int]] = None,
    membrane_type: str = "BLM",
    output_dir: str = "permeability_results",
    show_plots: bool = True,
    save_plots: bool = True,
) -> None:
    """
    Run complete permeability analysis with visualization.

    Args:
        molecules: List of molecules to analyze (uses defaults if None)
        composition: Membrane composition (uses POPC if None)
        membrane_type: Membrane type for calibration
        output_dir: Directory to save results
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Default molecules if none provided
    if molecules is None:
        molecules = [
            MoleculeDescriptor.water(),
            MoleculeDescriptor.ethanol(),
            MoleculeDescriptor.glucose(),
            MoleculeDescriptor.simple(
                name="caffeine",
                molecular_weight=194.2,
                total_asa=150.0,
                n_hbd=0,
                n_hba=3,
            ),
            MoleculeDescriptor.simple(
                name="aspirin",
                molecular_weight=180.2,
                total_asa=140.0,
                n_hbd=1,
                n_hba=4,
                charge=-1.0,
                pka=3.5,
            ),
        ]

    # Default composition
    if composition is None:
        composition = {"POPC": 128}

    print("\nMembrane Permeability Analysis")
    print("-" * 60)
    print(f"Membrane composition: {composition}")
    print(f"Membrane type: {membrane_type}")
    print(f"Molecules: {len(molecules)}")
    print()

    # Create analyzer
    analyzer = PermeabilityAnalyzer(composition=composition)

    # Print comparison table
    print_permeability_table(molecules, analyzer)

    # Analyze each molecule
    results = {}
    for mol in molecules:
        print(f"\nAnalyzing: {mol.name}")
        result = analyzer.predictor.calculate(mol, membrane_type)
        results[mol.name] = result
        print(f"  log P = {result.log_p:.2f} cm/s")
        print(f"  Binding energy = {result.membrane_bound_energy:.1f} kJ/mol")

    # Create visualizations
    print("\nGenerating figures...")

    try:
        viz = PermeabilityVisualizer()

        # Individual molecule analysis
        for mol in molecules:
            result = results[mol.name]
            save_path = output_path / f"{mol.name}_summary.png" if save_plots else None

            print(f"  Creating summary for {mol.name}...")
            viz.create_summary_figure(
                result,
                molecule_name=mol.name,
                save_path=str(save_path) if save_path else None,
                show=show_plots,
            )

        # Comparison plot
        print("  Creating comparison plot...")
        save_path = output_path / "comparison.png" if save_plots else None
        viz.plot_comparison(
            results,
            title=f"Permeability Comparison ({membrane_type})",
            save_path=str(save_path) if save_path else None,
            show=show_plots,
        )

        print(f"\nResults saved to: {output_path.absolute()}")

    except ImportError:
        print("\n  [WARNING] matplotlib not installed - skipping visualizations")
        print("  Install with: pip install matplotlib")

    # Generate text report
    report_path = output_path / "permeability_report.txt"
    with open(report_path, "w") as f:
        f.write("Membrane Permeability Analysis\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Membrane composition: {composition}\n")
        f.write(f"Membrane type: {membrane_type}\n\n")

        f.write("Results summary\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Molecule':<15} {'log P':>10} {'P (cm/s)':>15} {'Class':>12}\n")
        f.write("-" * 60 + "\n")

        for mol in molecules:
            result = results[mol.name]
            if result.log_p > -6:
                ranking = "high"
            elif result.log_p > -8:
                ranking = "moderate"
            else:
                ranking = "low"
            f.write(f"{mol.name:<15} {result.log_p:>10.2f} {result.permeability_cm_s:>15.2e} {ranking:>12}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("\nDetailed results\n")

        for mol in molecules:
            result = results[mol.name]
            f.write(f"\n{mol.name}\n")
            f.write(f"  Molecular weight: {mol.molecular_weight:.1f} Da\n")
            f.write(f"  Accessible surface area: {mol.total_asa:.0f} Å²\n")
            f.write(f"  log P: {result.log_p:.2f}\n")
            f.write(f"  Permeability: {result.permeability_cm_s:.2e} cm/s\n")
            f.write(f"  Membrane-bound energy: {result.membrane_bound_energy:.1f} kJ/mol\n")
            f.write(f"  Binding position: {result.binding_position:.1f} Å from center\n")

    print(f"Report saved to: {report_path}")

    print("\nAnalysis complete.")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze membrane permeability for molecules"
    )
    parser.add_argument(
        "--membrane-type",
        "-m",
        default="BLM",
        choices=["BLM", "PAMPA-DS", "BBB", "Caco-2"],
        help="Membrane type for calibration",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="permeability_results",
        help="Output directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save)",
    )
    parser.add_argument(
        "--cholesterol",
        "-c",
        type=float,
        default=0.0,
        help="Cholesterol fraction (0-0.5)",
    )

    args = parser.parse_args()

    # Build composition
    total = 128
    chol_count = int(total * args.cholesterol)
    popc_count = total - chol_count
    composition = {"POPC": popc_count}
    if chol_count > 0:
        composition["CHOL"] = chol_count

    run_permeability_analysis(
        composition=composition,
        membrane_type=args.membrane_type,
        output_dir=args.output,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    main()
