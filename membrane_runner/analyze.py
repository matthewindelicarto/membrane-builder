"""
Analysis tools for membrane permeability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from membrane_builder import (
    MoleculeDescriptor,
    PermeabilityPredictor,
    MembraneBuilder,
)


@dataclass
class PermeabilityReport:
    """Complete permeability analysis report."""
    molecule_name: str
    membrane_composition: Dict[str, int]
    membrane_thickness: float
    results_by_membrane_type: Dict[str, "PermeabilityResults"]
    ranking: str  # "high", "moderate", "low"
    drug_likeness: Dict[str, bool]

    def __str__(self) -> str:
        lines = [
            "-" * 60,
            f"Permeability Report: {self.molecule_name}",
            "-" * 60,
            "",
            "Membrane parameters:",
            f"  Composition: {self.membrane_composition}",
            f"  Thickness: {self.membrane_thickness:.1f} Å",
            "",
            "Permeability by membrane type:",
        ]

        for mtype, result in self.results_by_membrane_type.items():
            lines.append(f"  {mtype:12s}: log P = {result.log_p:6.2f} cm/s")

        lines.extend([
            "",
            f"Classification: {self.ranking}",
            "",
            "Drug-likeness assessment:",
        ])

        for criterion, passed in self.drug_likeness.items():
            status = "pass" if passed else "fail"
            lines.append(f"  {criterion}: {status}")

        lines.append("-" * 60)
        return "\n".join(lines)


class PermeabilityAnalyzer:
    """Analyze membrane permeability for molecules."""

    # Permeability thresholds (log P in cm/s)
    HIGH_PERM_THRESHOLD = -6.0
    LOW_PERM_THRESHOLD = -8.0

    def __init__(
        self,
        composition: Optional[Dict[str, int]] = None,
        membrane_thickness: float = 36.5,
    ):
        """
        Initialize analyzer.

        Args:
            composition: Lipid composition {name: count}
            membrane_thickness: Membrane thickness in Angstroms
        """
        self.composition = composition or {"POPC": 128}
        self.membrane_thickness = membrane_thickness
        self.predictor = PermeabilityPredictor(
            composition=self.composition,
            membrane_thickness=membrane_thickness,
        )

    @classmethod
    def from_membrane(cls, membrane: "BuiltMembrane") -> "PermeabilityAnalyzer":
        """Create analyzer from a built membrane."""
        composition = {}
        if membrane.config:
            composition = membrane.config.composition_dict

        thickness = 36.5
        if membrane.properties:
            thickness = membrane.properties.thickness

        return cls(composition=composition, membrane_thickness=thickness)

    def analyze(
        self,
        molecule: MoleculeDescriptor,
        membrane_types: Optional[List[str]] = None,
    ) -> PermeabilityReport:
        """
        Perform complete permeability analysis.

        Args:
            molecule: Molecule to analyze
            membrane_types: List of membrane types to test

        Returns:
            PermeabilityReport with all results
        """
        if membrane_types is None:
            membrane_types = ["BLM", "PAMPA-DS", "BBB", "Caco-2"]

        results = {}
        for mtype in membrane_types:
            results[mtype] = self.predictor.calculate(molecule, mtype)

        # Determine ranking based on BLM (or first available)
        primary_result = results.get("BLM", list(results.values())[0])
        log_p = primary_result.log_p

        if log_p > self.HIGH_PERM_THRESHOLD:
            ranking = "high"
        elif log_p > self.LOW_PERM_THRESHOLD:
            ranking = "moderate"
        else:
            ranking = "low"

        # Drug-likeness assessment
        drug_likeness = self._assess_drug_likeness(molecule, results)

        return PermeabilityReport(
            molecule_name=molecule.name,
            membrane_composition=self.composition,
            membrane_thickness=self.membrane_thickness,
            results_by_membrane_type=results,
            ranking=ranking,
            drug_likeness=drug_likeness,
        )

    def _assess_drug_likeness(
        self,
        molecule: MoleculeDescriptor,
        results: Dict[str, "PermeabilityResults"],
    ) -> Dict[str, bool]:
        """Assess drug-likeness based on permeability criteria."""
        assessments = {}

        # Lipinski-related
        assessments["MW < 500"] = molecule.molecular_weight < 500
        assessments["HBD <= 5"] = len([g for g in molecule.polar_groups if "HBD" in g.name]) <= 5
        assessments["HBA <= 10"] = len([g for g in molecule.polar_groups if "HBA" in g.name]) <= 10

        # Permeability-based
        if "Caco-2" in results:
            assessments["Caco-2 > 10⁻⁶"] = results["Caco-2"].log_p > -6
        if "BBB" in results:
            assessments["BBB penetrant"] = results["BBB"].log_p > -5

        # Membrane binding
        blm = results.get("BLM", list(results.values())[0])
        assessments["Not membrane trapped"] = blm.membrane_bound_energy > -20  # kJ/mol

        return assessments

    def compare_molecules(
        self,
        molecules: List[MoleculeDescriptor],
        membrane_type: str = "BLM",
    ) -> List[Tuple[str, float, str]]:
        """
        Compare permeability of multiple molecules.

        Returns:
            List of (name, log_p, ranking) tuples, sorted by permeability
        """
        results = []
        for mol in molecules:
            result = self.predictor.calculate(mol, membrane_type)
            log_p = result.log_p

            if log_p > self.HIGH_PERM_THRESHOLD:
                ranking = "high"
            elif log_p > self.LOW_PERM_THRESHOLD:
                ranking = "moderate"
            else:
                ranking = "low"

            results.append((mol.name, log_p, ranking))

        # Sort by permeability (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def scan_composition_effects(
        self,
        molecule: MoleculeDescriptor,
        base_composition: Dict[str, int],
        variable_lipid: str,
        fractions: List[float],
    ) -> Dict[float, float]:
        """
        Scan how varying a lipid affects permeability.

        Args:
            molecule: Test molecule
            base_composition: Starting composition
            variable_lipid: Lipid to vary (e.g., "CHOL")
            fractions: Mole fractions to test (0.0 to 0.5)

        Returns:
            Dict mapping fraction to log_p
        """
        results = {}
        total = sum(base_composition.values())

        for frac in fractions:
            # Adjust composition
            comp = {}
            var_count = int(total * frac)
            remaining = total - var_count

            # Scale other lipids
            other_total = sum(
                c for n, c in base_composition.items() if n != variable_lipid
            )

            for name, count in base_composition.items():
                if name == variable_lipid:
                    comp[name] = var_count
                elif other_total > 0:
                    comp[name] = int(remaining * count / other_total)

            if var_count > 0:
                comp[variable_lipid] = var_count

            # Calculate permeability
            predictor = PermeabilityPredictor(composition=comp)
            result = predictor.calculate(molecule, "BLM")
            results[frac] = result.log_p

        return results


def print_permeability_table(
    molecules: List[MoleculeDescriptor],
    analyzer: Optional[PermeabilityAnalyzer] = None,
) -> None:
    """Print a formatted table of permeability results."""
    if analyzer is None:
        analyzer = PermeabilityAnalyzer()

    print("\nPermeability comparison")
    print("-" * 70)
    print(f"{'Molecule':<15} {'MW (Da)':>10} {'ASA (Å²)':>10} {'log P':>10} {'Class':>12}")
    print("-" * 70)

    for mol in molecules:
        result = analyzer.predictor.calculate(mol, "BLM")
        log_p = result.log_p

        if log_p > -6:
            ranking = "high"
        elif log_p > -8:
            ranking = "moderate"
        else:
            ranking = "low"

        print(f"{mol.name:<15} {mol.molecular_weight:>10.1f} {mol.total_asa:>10.0f} {log_p:>10.2f} {ranking:>12}")

    print("-" * 70)
