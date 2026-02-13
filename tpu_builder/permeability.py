"""
Permeability prediction for TPU membranes

Uses regression model trained on experimental Franz cell data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from .polymers import PolymerLibrary, calculate_blend_properties


# Experimental training data from Franz cell experiments
# Format: (sparsa1_wt%, sparsa2_wt%, carbosil1_wt%, carbosil2_wt%, thickness_cm, permeability_cm2_s)
# Only includes validated data points (non-red rows)

PHENOL_TRAINING_DATA = [
    # M-01: 100% Sparsa 1
    (100, 0, 0, 0, 0.0254, 1.60618e-06),
    # M-02: 100% Sparsa 2
    (0, 100, 0, 0, 0.037, 7.55954e-07),
    # M-03: 100% Carbosil 1
    (0, 0, 100, 0, 0.021, 1.68063e-07),
    # M-05: 60% Sparsa 1, 40% Sparsa 2
    (60, 40, 0, 0, 0.0202, 5.75051e-07),
    # M-07: 30% Sparsa 1, 70% Sparsa 2
    (30, 70, 0, 0, 0.0208, 3.39749e-08),
    # M-11: 10% Sparsa 1, 20% Sparsa 2, 70% Carbosil 1
    (10, 20, 70, 0, 0.016, 1.59367e-07),
]

MCRESOL_TRAINING_DATA = [
    # M-02 (2): 100% Sparsa 2
    (0, 100, 0, 0, 0.018, 1.0215e-07),
    # M-03 (2): 100% Carbosil 1
    (0, 0, 100, 0, 0.0152, 7.64893e-08),
    # M-07: 30% Sparsa 1, 70% Sparsa 2
    (30, 70, 0, 0, 0.0208, 9.7528e-08),
    # M-11: 10% Sparsa 1, 20% Sparsa 2, 70% Carbosil 1
    (10, 20, 70, 0, 0.016, 1.09746e-07),
]

# Glucose permeability data from Andre's experiments
# Format: (sparsa1_wt%, sparsa2_wt%, carbosil1_wt%, carbosil2_wt%, thickness_cm, permeability_cm2_s)
# Note: This data uses Sparsa 2 and Carbosil 1 blends
# Thickness assumed to be ~0.02 cm (200 um) as typical membrane thickness
GLUCOSE_TRAINING_DATA = [
    # 100% Carbosil 1
    (0, 0, 100, 0, 0.02, 1.00e-13),
    # 30% Sparsa 2, 70% Carbosil 1
    (0, 30, 70, 0, 0.02, 8.30e-11),
    # 40% Sparsa 2, 60% Carbosil 1
    (0, 40, 60, 0, 0.02, 7.80e-10),
    # 60% Sparsa 2, 40% Carbosil 1
    (0, 60, 40, 0, 0.02, 9.68e-09),
    # 80% Sparsa 2, 20% Carbosil 1
    (0, 80, 20, 0, 0.02, 2.12e-08),
    # 100% Sparsa 2
    (0, 100, 0, 0, 0.02, 2.19e-08),
]


class PermeabilityRegressor:
    """
    Regression model for permeability prediction based on experimental data.
    Uses inverse distance weighting interpolation for small datasets.
    """

    def __init__(self, molecule_type: str = 'phenol'):
        self.molecule_type = molecule_type
        self._training_data = None
        self._load_data()

    def _load_data(self):
        """Load training data for the molecule type"""
        if self.molecule_type == 'phenol':
            self._training_data = PHENOL_TRAINING_DATA
        elif self.molecule_type == 'm-cresol':
            self._training_data = MCRESOL_TRAINING_DATA
        elif self.molecule_type == 'glucose':
            self._training_data = GLUCOSE_TRAINING_DATA
        else:
            self._training_data = PHENOL_TRAINING_DATA

    def predict(self, sparsa1_frac: float, sparsa2_frac: float,
                carbosil1_frac: float, carbosil2_frac: float,
                thickness_cm: float) -> float:
        """
        Predict permeability using inverse distance weighted interpolation.

        Args:
            sparsa1_frac: Fraction of Sparsa 1 (0-1)
            sparsa2_frac: Fraction of Sparsa 2 (0-1)
            carbosil1_frac: Fraction of Carbosil 1 (0-1)
            carbosil2_frac: Fraction of Carbosil 2 (0-1)
            thickness_cm: Membrane thickness in cm

        Returns:
            Predicted permeability in cm^2/s
        """
        if not self._training_data:
            return 1e-7  # Default fallback

        # Query point (composition only, thickness handled separately)
        query = np.array([sparsa1_frac, sparsa2_frac, carbosil1_frac, carbosil2_frac])

        # Calculate weighted average based on composition similarity
        weights = []
        log_perms = []
        thickness_ratios = []

        for sparsa1, sparsa2, carbosil1, carbosil2, thick, perm in self._training_data:
            total = sparsa1 + sparsa2 + carbosil1 + carbosil2
            if total == 0:
                total = 100

            point = np.array([sparsa1/total, sparsa2/total, carbosil1/total, carbosil2/total])

            # Euclidean distance in composition space
            dist = np.linalg.norm(query - point)

            # Inverse distance weight (with small epsilon to avoid division by zero)
            weight = 1.0 / (dist + 0.01) ** 2

            weights.append(weight)
            log_perms.append(np.log10(perm))
            thickness_ratios.append(thick)

        weights = np.array(weights)
        log_perms = np.array(log_perms)
        thickness_ratios = np.array(thickness_ratios)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average of log permeability
        base_log_perm = np.sum(weights * log_perms)

        # Weighted average reference thickness
        ref_thickness = np.sum(weights * thickness_ratios)

        # Adjust for thickness difference (permeability scales inversely with thickness)
        # P ~ 1/L, so log(P) ~ -log(L)
        thickness_adjustment = np.log10(ref_thickness / thickness_cm) if thickness_cm > 0 else 0

        log_perm = base_log_perm + thickness_adjustment

        # Clamp to reasonable range
        log_perm = np.clip(log_perm, -12, -4)

        return 10 ** log_perm


@dataclass
class MoleculeDescriptor:
    """Descriptor for a permeating molecule"""
    name: str
    molecular_weight: float  # Da
    molar_volume: float  # cm³/mol
    solubility_parameter: float  # (J/cm³)^0.5
    n_hbd: int  # H-bond donors
    n_hba: int  # H-bond acceptors
    log_p: float  # Octanol-water partition coefficient
    charge: float = 0

    @classmethod
    def simple(
        cls,
        name: str,
        molecular_weight: float,
        log_p: float = 0,
        n_hbd: int = 0,
        n_hba: int = 0
    ) -> 'MoleculeDescriptor':
        """Create a molecule with estimated properties"""
        # Estimate molar volume from MW (rough correlation)
        molar_volume = molecular_weight * 0.9  # Approximate

        # Estimate solubility parameter from log_p
        # Hydrophobic molecules have lower solubility parameters
        solubility_parameter = 23 - log_p * 2

        return cls(
            name=name,
            molecular_weight=molecular_weight,
            molar_volume=molar_volume,
            solubility_parameter=solubility_parameter,
            n_hbd=n_hbd,
            n_hba=n_hba,
            log_p=log_p
        )


# Molecule presets - only phenol, m-cresol, glucose, oxygen
MOLECULE_PRESETS = {
    'phenol': MoleculeDescriptor(
        name='phenol',
        molecular_weight=94.11,
        molar_volume=89.0,
        solubility_parameter=24.1,
        n_hbd=1,
        n_hba=1,
        log_p=1.46
    ),
    'm-cresol': MoleculeDescriptor(
        name='m-cresol',
        molecular_weight=108.14,
        molar_volume=105.0,
        solubility_parameter=23.3,
        n_hbd=1,
        n_hba=1,
        log_p=1.96
    ),
    'glucose': MoleculeDescriptor(
        name='glucose',
        molecular_weight=180.16,
        molar_volume=115.0,
        solubility_parameter=35.0,
        n_hbd=5,
        n_hba=6,
        log_p=-3.0
    ),
    'oxygen': MoleculeDescriptor(
        name='oxygen',
        molecular_weight=32.0,
        molar_volume=25.6,
        solubility_parameter=8.2,
        n_hbd=0,
        n_hba=0,
        log_p=0.65
    ),
}


@dataclass
class PermeabilityResult:
    """Result of permeability calculation"""
    molecule_name: str
    permeability_cm_s: float  # cm/s
    log_permeability: float  # log10(P)
    diffusivity_cm2_s: float  # cm²/s
    solubility: float  # dimensionless partition coefficient
    flux_mol_cm2_s: float  # mol/(cm²·s) at 1M gradient
    classification: str  # high, moderate, low


class TPUPermeabilityPredictor:
    """
    Predicts molecular permeability through TPU membranes

    Uses regression model trained on experimental Franz cell data
    for phenol, m-cresol, and glucose. Falls back to solution-diffusion
    model for other molecules (e.g., oxygen).
    """

    # Reference values for normalization
    D_REF = 1e-7  # cm²/s, reference diffusivity
    K_REF = 1.0  # Reference partition coefficient

    def __init__(
        self,
        composition: Optional[Dict[str, float]] = None,
        thickness_um: float = 100.0,
        temperature: float = 310.15  # 37°C in Kelvin
    ):
        self.library = PolymerLibrary()
        self.composition = composition or {"CarboSil": 1.0}
        self.thickness_um = thickness_um
        self.thickness_cm = thickness_um * 1e-4  # Convert to cm
        self.temperature = temperature

        # Calculate blend properties
        self.blend_props = calculate_blend_properties(self.composition, self.library)

        # Initialize regression models for phenol, m-cresol, and glucose
        self._phenol_regressor = PermeabilityRegressor('phenol')
        self._mcresol_regressor = PermeabilityRegressor('m-cresol')
        self._glucose_regressor = PermeabilityRegressor('glucose')

        # Parse composition into Sparsa/Carbosil fractions
        self._parse_composition()

    def _parse_composition(self):
        """Parse composition dict into Sparsa1, Sparsa2, Carbosil1, Carbosil2 fractions"""
        # Map old naming to new 4-component system
        # "Sparsa" -> Sparsa 1, "CarboSil" -> Carbosil 1 for backwards compatibility
        total = 0
        self.sparsa1_frac = 0
        self.sparsa2_frac = 0
        self.carbosil1_frac = 0
        self.carbosil2_frac = 0

        for name, frac in self.composition.items():
            name_lower = name.lower()
            if 'sparsa' in name_lower:
                if '2' in name_lower or 'sparsa2' in name_lower:
                    self.sparsa2_frac = frac
                else:
                    self.sparsa1_frac = frac
            elif 'carbosil' in name_lower or 'carbo' in name_lower:
                if '2' in name_lower or 'carbosil2' in name_lower:
                    self.carbosil2_frac = frac
                else:
                    self.carbosil1_frac = frac
            total += frac

        # Normalize if needed
        if total > 0 and abs(total - 1.0) > 0.01:
            self.sparsa1_frac /= total
            self.sparsa2_frac /= total
            self.carbosil1_frac /= total
            self.carbosil2_frac /= total

    def calculate(
        self,
        molecule: MoleculeDescriptor
    ) -> PermeabilityResult:
        """
        Calculate permeability for a molecule through the membrane

        Uses trained regression model for phenol and m-cresol,
        falls back to solution-diffusion model for other molecules.

        Args:
            molecule: MoleculeDescriptor for the permeating species

        Returns:
            PermeabilityResult with all calculated values
        """
        mol_name = molecule.name.lower()

        # Use regression model for phenol, m-cresol, and glucose
        if mol_name == 'phenol':
            P = self._phenol_regressor.predict(
                self.sparsa1_frac, self.sparsa2_frac,
                self.carbosil1_frac, self.carbosil2_frac,
                self.thickness_cm
            )
            D = P * self.thickness_cm  # Estimate diffusivity
            K = 1.0  # Not separately estimated from regression
        elif mol_name == 'm-cresol':
            P = self._mcresol_regressor.predict(
                self.sparsa1_frac, self.sparsa2_frac,
                self.carbosil1_frac, self.carbosil2_frac,
                self.thickness_cm
            )
            D = P * self.thickness_cm
            K = 1.0
        elif mol_name == 'glucose':
            P = self._glucose_regressor.predict(
                self.sparsa1_frac, self.sparsa2_frac,
                self.carbosil1_frac, self.carbosil2_frac,
                self.thickness_cm
            )
            D = P * self.thickness_cm
            K = 1.0
        else:
            # Fall back to theoretical model for oxygen, etc.
            D = self._calculate_diffusivity(molecule)
            K = self._calculate_solubility(molecule)
            P = D * K / self.thickness_cm

        # Calculate flux at unit concentration gradient
        flux = P  # mol/(cm²·s) for 1 mol/cm³ gradient

        # Log permeability
        log_P = np.log10(max(P, 1e-20))

        # Classification based on experimental data ranges
        if log_P > -6:
            classification = "high"
        elif log_P > -7.5:
            classification = "moderate"
        else:
            classification = "low"

        return PermeabilityResult(
            molecule_name=molecule.name,
            permeability_cm_s=P,
            log_permeability=log_P,
            diffusivity_cm2_s=D,
            solubility=K,
            flux_mol_cm2_s=flux,
            classification=classification
        )

    def _calculate_diffusivity(self, molecule: MoleculeDescriptor) -> float:
        """
        Calculate diffusivity using free volume theory

        D = D0 * exp(-Bd * V / Vf)

        where:
        - D0 = reference diffusivity
        - Bd = empirical constant
        - V = molecular volume
        - Vf = free volume of polymer
        """
        # Base diffusivity (Stokes-Einstein scaling)
        MW = molecule.molecular_weight
        D0 = self.D_REF * (100 / MW) ** 0.5

        # Free volume effect
        Vf = self.blend_props['free_volume_fraction']
        Vm = molecule.molar_volume / 100  # Normalize
        Bd = 0.5

        D_fv = D0 * np.exp(-Bd * Vm / max(Vf, 0.01))

        # Crystallinity reduction (tortuous path)
        cryst = self.blend_props['crystallinity']
        tau = 1 + cryst * 2  # Tortuosity factor
        D_cryst = D_fv / tau

        # Temperature dependence (Arrhenius)
        Ea = 20000  # J/mol, activation energy
        R = 8.314
        T_ref = 298.15
        D_temp = D_cryst * np.exp(-Ea / R * (1 / self.temperature - 1 / T_ref))

        return D_temp

    def _calculate_solubility(self, molecule: MoleculeDescriptor) -> float:
        """
        Calculate partition coefficient using solubility parameter theory

        K ~ exp(-V * (δ1 - δ2)² / RT)

        where:
        - V = molar volume
        - δ1, δ2 = solubility parameters of molecule and polymer
        """
        # Effective polymer solubility parameter
        delta_poly = 18.0  # Approximate for TPU blend
        if self.blend_props['hydrophilicity'] > 0.5:
            delta_poly = 22.0  # More hydrophilic

        delta_mol = molecule.solubility_parameter

        # Flory-Huggins interaction parameter
        R = 8.314
        V = molecule.molar_volume * 1e-6  # Convert to m³/mol
        chi = V * (delta_mol - delta_poly) ** 2 / (R * self.temperature)

        # Partition coefficient
        K = np.exp(-chi)

        # Hydrophilicity boost for polar molecules
        hydro = self.blend_props['hydrophilicity']
        if molecule.n_hbd > 0 or molecule.n_hba > 0:
            polar_factor = 1 + hydro * (molecule.n_hbd + molecule.n_hba) * 0.1
            K *= polar_factor

        # Water uptake effect (swollen membrane is more permeable to hydrophilics)
        water = self.blend_props['water_uptake']
        if molecule.log_p < 0:  # Hydrophilic molecule
            K *= (1 + water * 0.05)

        return min(K, 10)  # Cap at reasonable value

    def get_preset_molecule(self, name: str) -> MoleculeDescriptor:
        """Get a preset molecule by name"""
        if name not in MOLECULE_PRESETS:
            raise ValueError(f"Unknown molecule: {name}. Available: {list(MOLECULE_PRESETS.keys())}")
        return MOLECULE_PRESETS[name]

    def calculate_preset(self, molecule_name: str) -> PermeabilityResult:
        """Calculate permeability for a preset molecule"""
        molecule = self.get_preset_molecule(molecule_name)
        return self.calculate(molecule)

    def compare_molecules(self, molecules: list) -> Dict[str, PermeabilityResult]:
        """Calculate permeability for multiple molecules"""
        results = {}
        for mol_name in molecules:
            if isinstance(mol_name, str):
                mol = self.get_preset_molecule(mol_name)
            else:
                mol = mol_name
            results[mol.name] = self.calculate(mol)
        return results
