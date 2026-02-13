"""
Machine Learning Model for Permeability Prediction

Combines MD-derived descriptors with experimental Franz cell data
to predict permeability for new compositions.

Features:
- Composition features (polymer fractions)
- MD-derived descriptors (free volume, density, diffusivity estimates)
- Hybrid model using both experimental and simulation data
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os


@dataclass
class MDDescriptors:
    """Descriptors extracted from MD simulation"""
    density: float                    # g/cm³
    free_volume_fraction: float       # Fractional free volume
    diffusion_coefficient: float      # cm²/s (from MSD)
    rg: float                         # Radius of gyration
    end_to_end_distance: float        # End-to-end distance
    h_bond_count: float              # Average H-bonds
    soft_segment_fraction: float     # Fraction of soft segments


@dataclass
class CompositionFeatures:
    """Features derived from composition"""
    sparsa1_frac: float
    sparsa2_frac: float
    carbosil1_frac: float
    carbosil2_frac: float
    total_sparsa: float
    total_carbosil: float
    soft_segment_ratio: float  # Estimated from composition


# Experimental training data from Franz cell experiments
# Format: (composition_dict, thickness_cm, permeability_cm2_s, molecule)
EXPERIMENTAL_DATA = {
    'phenol': [
        ({'Sparsa1': 1.0, 'Sparsa2': 0, 'Carbosil1': 0, 'Carbosil2': 0}, 0.0254, 1.60618e-06),
        ({'Sparsa1': 0, 'Sparsa2': 1.0, 'Carbosil1': 0, 'Carbosil2': 0}, 0.037, 7.55954e-07),
        ({'Sparsa1': 0, 'Sparsa2': 0, 'Carbosil1': 1.0, 'Carbosil2': 0}, 0.021, 1.68063e-07),
        ({'Sparsa1': 0.6, 'Sparsa2': 0.4, 'Carbosil1': 0, 'Carbosil2': 0}, 0.0202, 5.75051e-07),
        ({'Sparsa1': 0.3, 'Sparsa2': 0.7, 'Carbosil1': 0, 'Carbosil2': 0}, 0.0208, 3.39749e-08),
        ({'Sparsa1': 0.1, 'Sparsa2': 0.2, 'Carbosil1': 0.7, 'Carbosil2': 0}, 0.016, 1.59367e-07),
    ],
    'm-cresol': [
        ({'Sparsa1': 0, 'Sparsa2': 1.0, 'Carbosil1': 0, 'Carbosil2': 0}, 0.018, 1.0215e-07),
        ({'Sparsa1': 0, 'Sparsa2': 0, 'Carbosil1': 1.0, 'Carbosil2': 0}, 0.0152, 7.64893e-08),
        ({'Sparsa1': 0.3, 'Sparsa2': 0.7, 'Carbosil1': 0, 'Carbosil2': 0}, 0.0208, 9.7528e-08),
        ({'Sparsa1': 0.1, 'Sparsa2': 0.2, 'Carbosil1': 0.7, 'Carbosil2': 0}, 0.016, 1.09746e-07),
    ],
    'glucose': [
        ({'Sparsa1': 0, 'Sparsa2': 0, 'Carbosil1': 1.0, 'Carbosil2': 0}, 0.02, 1.00e-13),
        ({'Sparsa1': 0, 'Sparsa2': 0.3, 'Carbosil1': 0.7, 'Carbosil2': 0}, 0.02, 8.30e-11),
        ({'Sparsa1': 0, 'Sparsa2': 0.4, 'Carbosil1': 0.6, 'Carbosil2': 0}, 0.02, 7.80e-10),
        ({'Sparsa1': 0, 'Sparsa2': 0.6, 'Carbosil1': 0.4, 'Carbosil2': 0}, 0.02, 9.68e-09),
        ({'Sparsa1': 0, 'Sparsa2': 0.8, 'Carbosil1': 0.2, 'Carbosil2': 0}, 0.02, 2.12e-08),
        ({'Sparsa1': 0, 'Sparsa2': 1.0, 'Carbosil1': 0, 'Carbosil2': 0}, 0.02, 2.19e-08),
    ]
}

# Polymer properties for feature estimation
POLYMER_PROPERTIES = {
    'Sparsa1': {
        'density': 1.05,
        'soft_segment_frac': 0.75,
        'Tg': -60,  # Glass transition (°C)
        'free_volume_base': 0.08,
    },
    'Sparsa2': {
        'density': 1.10,
        'soft_segment_frac': 0.70,
        'Tg': -50,
        'free_volume_base': 0.06,
    },
    'Carbosil1': {
        'density': 1.08,
        'soft_segment_frac': 0.65,
        'Tg': -110,  # PDMS is very flexible
        'free_volume_base': 0.05,
    },
    'Carbosil2': {
        'density': 1.12,
        'soft_segment_frac': 0.55,
        'Tg': -90,
        'free_volume_base': 0.04,
    }
}


class HybridPermeabilityModel:
    """
    Hybrid ML model combining experimental data and MD descriptors

    Uses:
    1. Inverse distance weighting for interpolation in composition space
    2. MD-derived corrections for free volume and diffusivity
    3. Molecule-specific models
    """

    def __init__(self, molecule: str = 'phenol'):
        self.molecule = molecule
        self.training_data = EXPERIMENTAL_DATA.get(molecule, [])
        self.md_descriptors_cache: Dict[str, MDDescriptors] = {}

    def _composition_to_features(self, composition: Dict[str, float]) -> np.ndarray:
        """Convert composition dict to feature vector"""
        total = sum(composition.values())
        if total == 0:
            total = 1

        s1 = composition.get('Sparsa1', 0) / total
        s2 = composition.get('Sparsa2', 0) / total
        c1 = composition.get('Carbosil1', 0) / total
        c2 = composition.get('Carbosil2', 0) / total

        # Calculate derived features
        total_sparsa = s1 + s2
        total_carbosil = c1 + c2

        # Estimate soft segment ratio from composition
        soft_seg = (
            s1 * POLYMER_PROPERTIES['Sparsa1']['soft_segment_frac'] +
            s2 * POLYMER_PROPERTIES['Sparsa2']['soft_segment_frac'] +
            c1 * POLYMER_PROPERTIES['Carbosil1']['soft_segment_frac'] +
            c2 * POLYMER_PROPERTIES['Carbosil2']['soft_segment_frac']
        )

        # Estimate free volume from composition
        free_vol = (
            s1 * POLYMER_PROPERTIES['Sparsa1']['free_volume_base'] +
            s2 * POLYMER_PROPERTIES['Sparsa2']['free_volume_base'] +
            c1 * POLYMER_PROPERTIES['Carbosil1']['free_volume_base'] +
            c2 * POLYMER_PROPERTIES['Carbosil2']['free_volume_base']
        )

        # Estimate density
        density = (
            s1 * POLYMER_PROPERTIES['Sparsa1']['density'] +
            s2 * POLYMER_PROPERTIES['Sparsa2']['density'] +
            c1 * POLYMER_PROPERTIES['Carbosil1']['density'] +
            c2 * POLYMER_PROPERTIES['Carbosil2']['density']
        )

        return np.array([s1, s2, c1, c2, total_sparsa, total_carbosil,
                        soft_seg, free_vol, density])

    def _distance(self, comp1: Dict, comp2: Dict) -> float:
        """Calculate distance between two compositions in feature space"""
        f1 = self._composition_to_features(comp1)
        f2 = self._composition_to_features(comp2)
        return np.linalg.norm(f1 - f2)

    def predict_idw(self, composition: Dict[str, float],
                    thickness_cm: float, power: float = 2.0) -> float:
        """
        Predict permeability using inverse distance weighting

        Args:
            composition: Polymer composition dict
            thickness_cm: Membrane thickness in cm
            power: Power parameter for IDW (higher = more local)

        Returns:
            Predicted permeability in cm²/s
        """
        if not self.training_data:
            raise ValueError(f"No training data for molecule: {self.molecule}")

        # Calculate distances to all training points
        distances = []
        permeabilities = []
        ref_thicknesses = []

        for train_comp, train_thick, train_perm in self.training_data:
            d = self._distance(composition, train_comp)
            distances.append(d)
            permeabilities.append(train_perm)
            ref_thicknesses.append(train_thick)

        distances = np.array(distances)
        permeabilities = np.array(permeabilities)
        ref_thicknesses = np.array(ref_thicknesses)

        # Handle exact matches
        min_dist = np.min(distances)
        if min_dist < 1e-10:
            idx = np.argmin(distances)
            base_perm = permeabilities[idx]
            thickness_ratio = ref_thicknesses[idx] / thickness_cm
            return base_perm * thickness_ratio

        # IDW weights
        weights = 1.0 / (distances ** power)
        weights = weights / np.sum(weights)

        # Weighted average in log space (permeability spans orders of magnitude)
        log_perms = np.log10(permeabilities)
        log_pred = np.sum(weights * log_perms)

        # Thickness correction
        avg_ref_thickness = np.sum(weights * ref_thicknesses)
        thickness_correction = np.log10(avg_ref_thickness / thickness_cm)

        return 10 ** (log_pred + thickness_correction)

    def predict_with_md(self, composition: Dict[str, float],
                        thickness_cm: float,
                        md_descriptors: Optional[MDDescriptors] = None) -> float:
        """
        Predict permeability with optional MD descriptor corrections

        Args:
            composition: Polymer composition dict
            thickness_cm: Membrane thickness
            md_descriptors: Optional MD-derived descriptors for correction

        Returns:
            Predicted permeability
        """
        # Base prediction from experimental data
        base_pred = self.predict_idw(composition, thickness_cm)

        if md_descriptors is None:
            return base_pred

        # Apply MD-based corrections
        # Free volume affects diffusivity exponentially
        estimated_fv = self._composition_to_features(composition)[7]
        fv_ratio = md_descriptors.free_volume_fraction / estimated_fv

        # Diffusivity correction based on free volume
        fv_correction = np.exp(0.5 * (fv_ratio - 1))

        # Density correction (denser = slower diffusion)
        estimated_density = self._composition_to_features(composition)[8]
        density_ratio = estimated_density / md_descriptors.density
        density_correction = density_ratio ** 0.3

        # Combined correction
        correction_factor = fv_correction * density_correction

        return base_pred * correction_factor

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance based on correlation with permeability
        """
        if len(self.training_data) < 3:
            return {}

        features_list = []
        perms = []

        for comp, thick, perm in self.training_data:
            features_list.append(self._composition_to_features(comp))
            perms.append(np.log10(perm))

        features = np.array(features_list)
        perms = np.array(perms)

        # Calculate correlations
        feature_names = ['Sparsa1', 'Sparsa2', 'Carbosil1', 'Carbosil2',
                        'Total_Sparsa', 'Total_Carbosil', 'Soft_Segment',
                        'Free_Volume', 'Density']

        importance = {}
        for i, name in enumerate(feature_names):
            if np.std(features[:, i]) > 1e-10:
                corr = np.corrcoef(features[:, i], perms)[0, 1]
                importance[name] = abs(corr)
            else:
                importance[name] = 0.0

        return importance


class PermeabilityPredictor:
    """
    Main interface for permeability prediction

    Combines multiple molecule-specific models
    """

    def __init__(self):
        self.models = {
            'phenol': HybridPermeabilityModel('phenol'),
            'm-cresol': HybridPermeabilityModel('m-cresol'),
            'glucose': HybridPermeabilityModel('glucose'),
        }

    def predict(self, molecule: str, composition: Dict[str, float],
                thickness_um: float,
                md_descriptors: Optional[MDDescriptors] = None) -> Dict:
        """
        Predict permeability for a molecule

        Args:
            molecule: Molecule name ('phenol', 'm-cresol', 'glucose')
            composition: Polymer composition dict
            thickness_um: Membrane thickness in micrometers
            md_descriptors: Optional MD descriptors

        Returns:
            Dict with prediction results
        """
        thickness_cm = thickness_um * 1e-4

        if molecule not in self.models:
            raise ValueError(f"Unknown molecule: {molecule}")

        model = self.models[molecule]

        if md_descriptors:
            permeability = model.predict_with_md(composition, thickness_cm, md_descriptors)
        else:
            permeability = model.predict_idw(composition, thickness_cm)

        # Get feature importance
        importance = model.get_feature_importance()

        return {
            'molecule': molecule,
            'permeability_cm2_s': permeability,
            'log_permeability': np.log10(permeability),
            'composition': composition,
            'thickness_um': thickness_um,
            'feature_importance': importance,
            'used_md_correction': md_descriptors is not None
        }

    def cross_validate(self, molecule: str) -> Dict:
        """
        Leave-one-out cross-validation for a molecule

        Returns:
            Dict with CV results
        """
        if molecule not in self.models:
            raise ValueError(f"Unknown molecule: {molecule}")

        data = EXPERIMENTAL_DATA.get(molecule, [])
        if len(data) < 3:
            return {'error': 'Not enough data for CV'}

        errors = []
        for i in range(len(data)):
            # Leave out point i
            train_data = data[:i] + data[i+1:]
            test_comp, test_thick, test_perm = data[i]

            # Create temporary model
            temp_model = HybridPermeabilityModel(molecule)
            temp_model.training_data = train_data

            # Predict
            pred = temp_model.predict_idw(test_comp, test_thick)

            # Log error
            log_error = abs(np.log10(pred) - np.log10(test_perm))
            errors.append(log_error)

        return {
            'molecule': molecule,
            'n_samples': len(data),
            'mean_log_error': np.mean(errors),
            'std_log_error': np.std(errors),
            'max_log_error': np.max(errors),
            'errors': errors
        }


def train_and_evaluate():
    """Train models and show evaluation results"""
    predictor = PermeabilityPredictor()

    print("=" * 60)
    print("Hybrid Permeability Model - Training and Evaluation")
    print("=" * 60)

    for molecule in ['phenol', 'm-cresol', 'glucose']:
        print(f"\n--- {molecule.upper()} ---")

        # Cross-validation
        cv_results = predictor.cross_validate(molecule)
        if 'error' not in cv_results:
            print(f"Leave-one-out CV (n={cv_results['n_samples']}):")
            print(f"  Mean log error: {cv_results['mean_log_error']:.3f}")
            print(f"  Std log error:  {cv_results['std_log_error']:.3f}")

        # Feature importance
        model = predictor.models[molecule]
        importance = model.get_feature_importance()
        if importance:
            print("Feature importance (correlation with log P):")
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for name, imp in sorted_imp[:5]:
                print(f"  {name}: {imp:.3f}")

    # Example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    test_compositions = [
        {'Sparsa1': 0.5, 'Sparsa2': 0.0, 'Carbosil1': 0.5, 'Carbosil2': 0.0},
        {'Sparsa1': 0.0, 'Sparsa2': 0.5, 'Carbosil1': 0.5, 'Carbosil2': 0.0},
        {'Sparsa1': 0.25, 'Sparsa2': 0.25, 'Carbosil1': 0.25, 'Carbosil2': 0.25},
    ]

    for comp in test_compositions:
        print(f"\nComposition: {comp}")
        for mol in ['phenol', 'glucose']:
            result = predictor.predict(mol, comp, thickness_um=200)
            print(f"  {mol}: {result['permeability_cm2_s']:.2e} cm²/s "
                  f"(log P = {result['log_permeability']:.2f})")


if __name__ == "__main__":
    train_and_evaluate()
