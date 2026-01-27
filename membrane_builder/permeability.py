"""
Membrane permeability prediction based on the PerMM method.

Implementation of physics-based permeability calculation following
Lomize & Pogozheva (2019) "Positioning of Proteins in Membranes
for Membrane Permeability".

References:
    Lomize AL, Pogozheva ID. J Chem Inf Model. 2019;59(7):3198-3213.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math
import numpy as np

from .lipids import LipidLibrary, LipidCategory


# Physical constants
R = 8.314  # J/(mol*K)
KB = 1.380649e-23  # J/K
T_STANDARD = 310.15  # K (37C)
RT = R * T_STANDARD / 1000  # kJ/mol at 37C


class AtomType(Enum):
    """Atom types for solvation parameter lookup."""
    C_SP3 = "C_sp3"
    C_SP2 = "C_sp2"
    C_AROMATIC = "C_aromatic"
    N_AMINE = "N_amine"
    N_AMIDE = "N_amide"
    N_AROMATIC = "N_aromatic"
    O_HYDROXYL = "O_hydroxyl"
    O_ETHER = "O_ether"
    O_CARBONYL = "O_carbonyl"
    O_CARBOXYL = "O_carboxyl"
    S_THIOL = "S_thiol"
    S_THIOETHER = "S_thioether"
    F = "F"
    CL = "Cl"
    BR = "Br"
    P = "P"


class MembraneType(Enum):
    """Membrane types for calibration."""
    BLM = "BLM"  # Black lipid membrane
    PAMPA_DS = "PAMPA-DS"  # Double-sink PAMPA
    BBB = "BBB"  # Blood-brain barrier
    CACO2 = "Caco-2"  # Intestinal epithelium


@dataclass
class AtomDescriptor:
    """Represents an atom in a molecule for permeability calculation."""
    element: str
    atom_type: AtomType
    x: float
    y: float
    z: float
    asa: float = 0.0  # Accessible surface area in A^2
    charge: float = 0.0
    # Solvation parameters (from eq. 14)
    e_param: float = 0.0  # Dielectric contribution
    a_param: float = 0.0  # H-bond donor contribution
    b_param: float = 0.0  # H-bond acceptor contribution


@dataclass
class PolarGroup:
    """Represents a polar group with dipole moment."""
    name: str
    dipole_moment: float  # Debye
    center_x: float
    center_y: float
    center_z: float


@dataclass
class IonizableGroup:
    """Represents an ionizable group."""
    name: str
    pka: float
    charge_when_ionized: float
    center_x: float
    center_y: float
    center_z: float
    is_acid: bool = True  # True for acids (lose H+), False for bases (gain H+)


@dataclass
class MoleculeDescriptor:
    """
    Complete descriptor for a molecule to calculate permeability.

    Can be constructed manually or from simplified inputs.
    """
    name: str
    atoms: List[AtomDescriptor] = field(default_factory=list)
    polar_groups: List[PolarGroup] = field(default_factory=list)
    ionizable_groups: List[IonizableGroup] = field(default_factory=list)
    molecular_weight: float = 0.0
    total_asa: float = 0.0

    def __post_init__(self):
        if self.total_asa == 0.0 and self.atoms:
            self.total_asa = sum(a.asa for a in self.atoms)

    @classmethod
    def simple(
        cls,
        name: str,
        molecular_weight: float,
        total_asa: float,
        n_hbd: int = 0,  # H-bond donors
        n_hba: int = 0,  # H-bond acceptors
        charge: float = 0.0,
        pka: Optional[float] = None,
    ) -> "MoleculeDescriptor":
        """
        Create a simplified molecule descriptor.

        For quick estimates without full atom coordinates.
        """
        mol = cls(
            name=name,
            molecular_weight=molecular_weight,
            total_asa=total_asa,
        )

        # Create synthetic polar groups based on H-bond counts
        for i in range(n_hbd):
            mol.polar_groups.append(PolarGroup(
                name=f"HBD_{i}",
                dipole_moment=1.5,  # Typical OH/NH
                center_x=0, center_y=0, center_z=0
            ))

        for i in range(n_hba):
            mol.polar_groups.append(PolarGroup(
                name=f"HBA_{i}",
                dipole_moment=1.2,
                center_x=0, center_y=0, center_z=0
            ))

        # Add ionizable group if pKa provided
        if pka is not None:
            is_acid = charge < 0
            mol.ionizable_groups.append(IonizableGroup(
                name="ionizable",
                pka=pka,
                charge_when_ionized=charge,
                center_x=0, center_y=0, center_z=0,
                is_acid=is_acid
            ))

        return mol

    @classmethod
    def water(cls) -> "MoleculeDescriptor":
        """Create water molecule descriptor."""
        return cls.simple(
            name="water",
            molecular_weight=18.015,
            total_asa=40.0,
            n_hbd=2,
            n_hba=1,
        )

    @classmethod
    def glucose(cls) -> "MoleculeDescriptor":
        """Create glucose molecule descriptor."""
        return cls.simple(
            name="glucose",
            molecular_weight=180.16,
            total_asa=180.0,
            n_hbd=5,
            n_hba=6,
        )

    @classmethod
    def ethanol(cls) -> "MoleculeDescriptor":
        """Create ethanol molecule descriptor."""
        return cls.simple(
            name="ethanol",
            molecular_weight=46.07,
            total_asa=80.0,
            n_hbd=1,
            n_hba=1,
        )


@dataclass
class MembraneProfiles:
    """
    Z-dependent membrane profiles.

    Based on experimental X-ray and neutron scattering data.
    Z = 0 is at membrane center.
    """
    z_positions: np.ndarray  # A from center
    epsilon: np.ndarray  # Dielectric constant
    alpha: np.ndarray  # H-bond donor capacity
    beta: np.ndarray  # H-bond acceptor capacity

    # Hydrocarbon core boundaries
    core_boundary: float = 15.0  # A from center

    def get_epsilon_at_z(self, z: float) -> float:
        """Interpolate dielectric constant at position z."""
        return float(np.interp(abs(z), self.z_positions, self.epsilon))

    def get_alpha_at_z(self, z: float) -> float:
        """Interpolate H-bond donor capacity at position z."""
        return float(np.interp(abs(z), self.z_positions, self.alpha))

    def get_beta_at_z(self, z: float) -> float:
        """Interpolate H-bond acceptor capacity at position z."""
        return float(np.interp(abs(z), self.z_positions, self.beta))


@dataclass
class PermeabilityResults:
    """Results of permeability calculation."""
    log_p: float  # log10(permeability in cm/s)
    permeability_cm_s: float  # Permeability in cm/s
    energy_profile: Dict[str, np.ndarray]  # z vs energy
    partition_coefficients: Dict[str, np.ndarray]  # z vs K
    membrane_type: str
    membrane_bound_energy: float  # Minimum energy in membrane
    binding_position: float  # Z position of minimum energy

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "Permeability prediction",
            "-" * 40,
            f"Membrane type:     {self.membrane_type}",
            f"log P:             {self.log_p:.2f}",
            f"Permeability:      {self.permeability_cm_s:.2e} cm/s",
            "",
            f"Binding energy:    {self.membrane_bound_energy:.1f} kJ/mol",
            f"Binding position:  {self.binding_position:.1f} Å from center",
        ]
        return "\n".join(lines)


# Atomic solvation parameters (Table S1 from paper, eq. 14)
# sigma = sigma_0 + e*(1/eps_bil - 1/eps_wat) + a*(alpha_bil - alpha_wat) + b*(beta_bil - beta_wat)
ATOMIC_SOLVATION_PARAMS = {
    AtomType.C_SP3: {"sigma_0": 0.012, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.C_SP2: {"sigma_0": 0.004, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.C_AROMATIC: {"sigma_0": 0.000, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.N_AMINE: {"sigma_0": -0.100, "e": -3.0, "a": 5.5, "b": 0.0},
    AtomType.N_AMIDE: {"sigma_0": -0.060, "e": -2.0, "a": 3.0, "b": 0.0},
    AtomType.N_AROMATIC: {"sigma_0": -0.040, "e": -1.5, "a": 0.0, "b": 2.0},
    AtomType.O_HYDROXYL: {"sigma_0": -0.070, "e": -2.0, "a": 4.0, "b": 3.0},
    AtomType.O_ETHER: {"sigma_0": -0.020, "e": -1.0, "a": 0.0, "b": 2.5},
    AtomType.O_CARBONYL: {"sigma_0": -0.050, "e": -1.5, "a": 0.0, "b": 4.0},
    AtomType.O_CARBOXYL: {"sigma_0": -0.080, "e": -2.5, "a": 2.0, "b": 4.5},
    AtomType.S_THIOL: {"sigma_0": 0.005, "e": -0.5, "a": 1.0, "b": 0.5},
    AtomType.S_THIOETHER: {"sigma_0": 0.008, "e": -0.3, "a": 0.0, "b": 0.5},
    AtomType.F: {"sigma_0": -0.010, "e": -0.8, "a": 0.0, "b": 0.5},
    AtomType.CL: {"sigma_0": 0.005, "e": -0.5, "a": 0.0, "b": 0.3},
    AtomType.BR: {"sigma_0": 0.008, "e": -0.3, "a": 0.0, "b": 0.2},
    AtomType.P: {"sigma_0": -0.030, "e": -1.0, "a": 0.0, "b": 2.0},
}

# Membrane-specific calibration constants (eqs. 15, 21-23)
CALIBRATION_PARAMS = {
    MembraneType.BLM: {"slope": 1.063, "intercept": 3.669},
    MembraneType.PAMPA_DS: {"slope": 0.981, "intercept": 2.159},
    MembraneType.BBB: {"slope": 0.375, "intercept": -1.600},
    MembraneType.CACO2: {"slope": 0.272, "intercept": -2.541},
}


class MembraneProfileGenerator:
    """
    Generates membrane profiles based on lipid composition.

    Default profiles are for DOPC bilayer from experimental data.
    Composition adjustments modify the profiles.
    """

    def __init__(self):
        self.library = LipidLibrary()

    def get_default_profiles(self) -> MembraneProfiles:
        """
        Return default DOPC bilayer profiles.

        Based on Figure S1 from Lomize & Pogozheva (2019).
        """
        # Z positions from center (use positive half, symmetric)
        z = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

        # Dielectric constant profile (eq. 11)
        # Core: ~2, transitions to water: 78.4
        epsilon = np.array([
            2.0, 2.0, 2.0, 2.1, 2.3, 2.8,  # Core (0-10 A)
            4.5, 8.0, 15.0, 25.0,  # Transition (12-18 A)
            40.0, 55.0, 68.0, 75.0, 77.5, 78.4  # Headgroup/water (20-30 A)
        ])

        # H-bond donor capacity (alpha)
        # Core: 0, water: 0.82
        alpha = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.02,  # Core
            0.08, 0.20, 0.35, 0.50,  # Transition
            0.62, 0.72, 0.78, 0.80, 0.81, 0.82  # Water
        ])

        # H-bond acceptor capacity (beta)
        # Core: 0, water: 0.82
        beta = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
            0.08, 0.20, 0.35, 0.50,
            0.62, 0.72, 0.78, 0.80, 0.81, 0.82
        ])

        return MembraneProfiles(
            z_positions=z,
            epsilon=epsilon,
            alpha=alpha,
            beta=beta,
            core_boundary=15.0
        )

    def get_profiles_for_composition(
        self,
        composition: Dict[str, int],
        thickness: float = 36.5,
    ) -> MembraneProfiles:
        """
        Generate composition-adjusted membrane profiles.

        Args:
            composition: {lipid_name: count}
            thickness: Membrane thickness in A
        """
        profiles = self.get_default_profiles()

        if not composition:
            return profiles

        # Calculate composition fractions
        total = sum(composition.values())
        if total == 0:
            return profiles

        # Get lipid properties
        chol_frac = 0.0
        sm_frac = 0.0
        unsat_frac = 0.0
        charged_frac = 0.0

        for name, count in composition.items():
            frac = count / total
            lipid = self.library.get(name)
            if lipid is None:
                continue

            if lipid.category == LipidCategory.STEROL:
                chol_frac += frac
            elif lipid.category == LipidCategory.SPHINGOMYELIN:
                sm_frac += frac

            # Count unsaturations
            total_unsat = sum(lipid.tail_unsaturations)
            if total_unsat > 0:
                unsat_frac += frac

            if lipid.charge != 0:
                charged_frac += frac

        # Adjust profiles based on composition

        # Cholesterol effect: narrows transition region, increases order
        if chol_frac > 0:
            # Sharpen dielectric transition
            transition_factor = 1.0 - 0.3 * chol_frac
            profiles.epsilon = self._sharpen_profile(
                profiles.epsilon, profiles.z_positions, transition_factor
            )
            profiles.alpha = self._sharpen_profile(
                profiles.alpha, profiles.z_positions, transition_factor
            )
            profiles.beta = self._sharpen_profile(
                profiles.beta, profiles.z_positions, transition_factor
            )

        # Sphingomyelin: tighter packing
        if sm_frac > 0:
            # Shift transition region slightly inward
            shift = -1.0 * sm_frac
            profiles.z_positions = profiles.z_positions + shift

        # Unsaturation: increases disorder, broadens transition
        if unsat_frac > 0.5:
            broaden_factor = 1.0 + 0.2 * (unsat_frac - 0.5)
            profiles.z_positions = profiles.z_positions * broaden_factor

        # Scale to actual thickness
        thickness_ratio = thickness / 36.5  # Ratio to default
        profiles.z_positions = profiles.z_positions * (thickness_ratio ** 0.5)
        profiles.core_boundary = 15.0 * (thickness_ratio ** 0.5)

        return profiles

    def _sharpen_profile(
        self,
        profile: np.ndarray,
        z: np.ndarray,
        factor: float,
    ) -> np.ndarray:
        """Sharpen a sigmoidal profile by adjusting steepness."""
        # Find midpoint of transition
        mid_val = (profile[0] + profile[-1]) / 2
        mid_idx = np.argmin(np.abs(profile - mid_val))

        # Compress z-scale around midpoint
        z_shifted = z - z[mid_idx]
        z_compressed = z_shifted * factor + z[mid_idx]

        # Reinterpolate
        return np.interp(z, z_compressed, profile)


class TransferEnergyCalculator:
    """
    Calculates transfer free energy for molecules in membrane.

    Based on eq. 7 from Lomize & Pogozheva (2019).
    """

    # Water reference values
    EPSILON_WATER = 78.4
    ALPHA_WATER = 0.82
    BETA_WATER = 0.82

    def __init__(self, profiles: MembraneProfiles):
        self.profiles = profiles

    def calculate_transfer_energy(
        self,
        molecule: MoleculeDescriptor,
        z_position: float,
        pH: float = 7.4,
        temperature: float = T_STANDARD,
    ) -> float:
        """
        Calculate transfer free energy at position z.

        Args:
            molecule: Molecule descriptor
            z_position: Position along membrane normal (0 = center)
            pH: Solution pH for ionization state
            temperature: Temperature in K

        Returns:
            Transfer energy in kJ/mol
        """
        energy = 0.0

        # 1. ASA-dependent solvation (eq. 14)
        if molecule.atoms:
            for atom in molecule.atoms:
                sigma = self._get_solvation_parameter(atom, z_position)
                energy += sigma * atom.asa
        else:
            # Use simplified model based on total ASA and polar groups
            energy += self._simplified_solvation_energy(molecule, z_position)

        # 2. Dipolar contributions
        for group in molecule.polar_groups:
            eta = self._get_dipole_penalty(z_position)
            energy += eta * group.dipole_moment

        # 3. Ionization state optimization
        for ionizable in molecule.ionizable_groups:
            energy += self._ionization_energy(
                ionizable, z_position, pH, temperature
            )

        return energy

    def _get_solvation_parameter(
        self,
        atom: AtomDescriptor,
        z: float,
    ) -> float:
        """Calculate position-dependent solvation parameter (eq. 14)."""
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        alpha_bil = self.profiles.get_alpha_at_z(z)
        beta_bil = self.profiles.get_beta_at_z(z)

        params = ATOMIC_SOLVATION_PARAMS.get(atom.atom_type)
        if params is None:
            return 0.0

        sigma = params["sigma_0"] + params["e"] * (
            1/epsilon_bil - 1/self.EPSILON_WATER
        ) + params["a"] * (
            alpha_bil - self.ALPHA_WATER
        ) + params["b"] * (
            beta_bil - self.BETA_WATER
        )

        return sigma

    def _simplified_solvation_energy(
        self,
        molecule: MoleculeDescriptor,
        z: float,
    ) -> float:
        """
        Simplified solvation energy when full atom coords unavailable.

        Uses hydrophobic effect estimate.
        """
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        alpha_bil = self.profiles.get_alpha_at_z(z)
        beta_bil = self.profiles.get_beta_at_z(z)

        # Hydrophobic contribution (favorable in core)
        # ~0.012 kJ/mol per A^2 of hydrophobic ASA
        hydrophobic_asa = molecule.total_asa * 0.6  # Estimate 60% hydrophobic
        polar_asa = molecule.total_asa * 0.4

        # Hydrophobic effect
        hydrophobic_energy = -0.012 * hydrophobic_asa * (
            1.0 - epsilon_bil / self.EPSILON_WATER
        )

        # Polar penalty in core
        polar_penalty = 0.08 * polar_asa * (
            1.0 - alpha_bil / self.ALPHA_WATER
        )

        return hydrophobic_energy + polar_penalty

    def _get_dipole_penalty(self, z: float) -> float:
        """Get position-dependent dipole penalty."""
        epsilon_bil = self.profiles.get_epsilon_at_z(z)

        # Penalty scales inversely with dielectric
        # ~0.5 kJ/mol per Debye in hydrocarbon core
        return 0.5 * (self.EPSILON_WATER / epsilon_bil - 1.0)

    def _ionization_energy(
        self,
        group: IonizableGroup,
        z: float,
        pH: float,
        temperature: float,
    ) -> float:
        """
        Calculate optimal ionization state energy (eq. 9).

        Uses Henderson-Hasselbalch to find lowest energy state.
        """
        RT_local = R * temperature / 1000  # kJ/mol

        # Deionization free energy
        if group.is_acid:
            delta_G_ion = 2.303 * RT_local * (pH - group.pka)
        else:
            delta_G_ion = 2.303 * RT_local * (group.pka - pH)

        # Born energy for charged state in membrane
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        # Simplified Born model
        born_energy = 332.0 * (group.charge_when_ionized ** 2) * (
            1/epsilon_bil - 1/self.EPSILON_WATER
        ) / 4.0  # Approximate ion radius of 4 A

        # Choose lowest energy state
        neutral_energy = 0.0
        ionized_energy = born_energy + delta_G_ion

        return min(neutral_energy, ionized_energy)


class PermeabilityPredictor:
    """
    Predicts membrane permeability using the PerMM method.

    Implements the solubility-diffusion model with
    physics-based transfer free energy calculations.
    """

    def __init__(
        self,
        composition: Optional[Dict[str, int]] = None,
        membrane_thickness: float = 36.5,
        temperature: float = T_STANDARD,
    ):
        """
        Initialize predictor.

        Args:
            composition: Lipid composition {name: count}
            membrane_thickness: Membrane thickness in A
            temperature: Temperature in K
        """
        self.composition = composition or {}
        self.thickness = membrane_thickness
        self.temperature = temperature

        # Generate membrane profiles
        profile_gen = MembraneProfileGenerator()
        if composition:
            self.profiles = profile_gen.get_profiles_for_composition(
                composition, membrane_thickness
            )
        else:
            self.profiles = profile_gen.get_default_profiles()

        self.energy_calc = TransferEnergyCalculator(self.profiles)

    def calculate(
        self,
        molecule: MoleculeDescriptor,
        membrane_type: Union[MembraneType, str] = MembraneType.BLM,
        pH: float = 7.4,
        n_points: int = 31,
    ) -> PermeabilityResults:
        """
        Calculate permeability for a molecule.

        Args:
            molecule: Molecule descriptor
            membrane_type: Type of membrane for calibration (MembraneType or string)
            pH: Solution pH
            n_points: Number of points for integration

        Returns:
            PermeabilityResults with predictions
        """
        # Convert string to enum if needed
        if isinstance(membrane_type, str):
            membrane_type = MembraneType(membrane_type)

        # Integration range: through hydrocarbon core
        z_range = np.linspace(
            -self.profiles.core_boundary,
            self.profiles.core_boundary,
            n_points
        )

        # Calculate energy profile
        energy_profile = np.array([
            self.energy_calc.calculate_transfer_energy(
                molecule, z, pH, self.temperature
            )
            for z in z_range
        ])

        # Partition coefficient K(z) = exp(-deltaG/RT)
        RT = R * self.temperature / 1000  # kJ/mol
        K_profile = np.exp(-energy_profile / RT)

        # Solubility-diffusion integral (eq. 6)
        # P = D / integral(1/K dz)
        # log P_sigma = -log10(integral) - log10(ASA)

        # Avoid division by zero
        K_safe = np.maximum(K_profile, 1e-10)
        integral = np.trapz(1.0 / K_safe, z_range)

        # Size-dependent correction using ASA
        asa = molecule.total_asa if molecule.total_asa > 0 else 100.0
        log_P_sigma = -np.log10(max(integral, 1e-20)) - np.log10(asa)

        # Apply membrane-specific calibration
        calib = CALIBRATION_PARAMS[membrane_type]
        log_P_calc = calib["slope"] * log_P_sigma + calib["intercept"]

        # Find membrane-bound state
        min_energy_idx = np.argmin(energy_profile)
        membrane_bound_energy = energy_profile[min_energy_idx]
        binding_position = z_range[min_energy_idx]

        return PermeabilityResults(
            log_p=log_P_calc,
            permeability_cm_s=10 ** log_P_calc,
            energy_profile={
                "z": z_range,
                "energy_kJ_mol": energy_profile,
            },
            partition_coefficients={
                "z": z_range,
                "K": K_profile,
            },
            membrane_type=membrane_type.value,
            membrane_bound_energy=membrane_bound_energy,
            binding_position=binding_position,
        )

    @classmethod
    def quick_calculate(
        cls,
        molecule: MoleculeDescriptor,
        membrane_type: MembraneType = MembraneType.BLM,
    ) -> float:
        """Quick permeability calculation with default parameters."""
        predictor = cls()
        result = predictor.calculate(molecule, membrane_type)
        return result.log_p


class MembraneCompositionOptimizer:
    """
    Optimizes membrane composition for target permeability.

    Combines neural network-based predictions with physics model.
    """

    def __init__(
        self,
        target_molecule: MoleculeDescriptor,
        base_composition: Optional[Dict[str, int]] = None,
    ):
        self.target_molecule = target_molecule
        self.base_composition = base_composition or {"POPC": 128}
        self.library = LipidLibrary()

    def predict_permeability_for_composition(
        self,
        composition: Dict[str, int],
        membrane_type: MembraneType = MembraneType.BLM,
    ) -> float:
        """Predict permeability for a given composition."""
        # Calculate membrane thickness from composition
        total = sum(composition.values())
        if total == 0:
            return -10.0  # Very low permeability

        weighted_thickness = 0.0
        for name, count in composition.items():
            lipid = self.library.get(name)
            if lipid:
                weighted_thickness += lipid.thickness_contribution * count

        thickness = weighted_thickness / total if total > 0 else 36.5

        # Create predictor with composition
        predictor = PermeabilityPredictor(
            composition=composition,
            membrane_thickness=thickness,
        )

        result = predictor.calculate(self.target_molecule, membrane_type)
        return result.log_p

    def optimize_for_target_permeability(
        self,
        target_log_p: float,
        membrane_type: MembraneType = MembraneType.BLM,
        max_iterations: int = 100,
    ) -> Tuple[Dict[str, int], float]:
        """
        Find composition that achieves target permeability.

        Simple gradient-free optimization by adjusting cholesterol content.
        """
        best_composition = dict(self.base_composition)
        best_diff = float("inf")

        # Try different cholesterol fractions
        for chol_frac in np.linspace(0, 0.45, 10):
            composition = self._adjust_cholesterol(
                self.base_composition, chol_frac
            )
            log_p = self.predict_permeability_for_composition(
                composition, membrane_type
            )
            diff = abs(log_p - target_log_p)

            if diff < best_diff:
                best_diff = diff
                best_composition = composition

        return best_composition, self.predict_permeability_for_composition(
            best_composition, membrane_type
        )

    def _adjust_cholesterol(
        self,
        base: Dict[str, int],
        chol_fraction: float,
    ) -> Dict[str, int]:
        """Adjust cholesterol content while maintaining total lipid count."""
        total = sum(base.values())
        chol_count = int(total * chol_fraction)
        remaining = total - chol_count

        # Scale other lipids proportionally
        result = {}
        other_total = sum(
            c for n, c in base.items()
            if self.library.get(n) and
            self.library.get(n).category != LipidCategory.STEROL
        )

        for name, count in base.items():
            lipid = self.library.get(name)
            if lipid and lipid.category == LipidCategory.STEROL:
                result[name] = chol_count
            elif other_total > 0:
                result[name] = int(remaining * count / other_total)

        return result


def quick_permeability(
    molecule: MoleculeDescriptor,
    composition: Optional[Dict[str, int]] = None,
    membrane_type: str = "BLM",
) -> float:
    """
    Quick permeability calculation.

    Args:
        molecule: Molecule descriptor
        composition: Optional lipid composition
        membrane_type: "BLM", "PAMPA-DS", "BBB", or "Caco-2"

    Returns:
        log P in cm/s
    """
    mt = MembraneType(membrane_type)

    if composition:
        predictor = PermeabilityPredictor(composition=composition)
    else:
        predictor = PermeabilityPredictor()

    return predictor.calculate(molecule, mt).log_p
