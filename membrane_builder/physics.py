"""
Calculates physical properties of lipid bilayers.

References:
  Nagle & Tristram-Nagle (2000) BBA
  Rawicz et al. (2000) Biophys J
  Marsh (2006) Chem Phys Lipids
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from .lipids import Lipid, LipidLibrary, LipidCategory


KB = 1.380649e-23  # J/K
T_STANDARD = 310.15  # K (37C)


@dataclass
class MembraneProperties:
    """Physical properties of a membrane."""
    bending_modulus: float  # Kc in kT
    bending_modulus_joules: float
    gaussian_modulus: float  # Kbar in kT
    area_compressibility: float  # Ka in kT/A^2
    thickness: float  # Angstroms
    area_per_lipid: float  # Angstroms^2
    total_area: float
    total_lipids: int
    net_charge: float
    charge_density: float
    spontaneous_curvature: float  # 1/Angstroms
    temperature: float

    def to_dict(self) -> dict:
        return {
            "bending_modulus_kT": self.bending_modulus,
            "bending_modulus_J": self.bending_modulus_joules,
            "gaussian_modulus_kT": self.gaussian_modulus,
            "area_compressibility_kT_per_A2": self.area_compressibility,
            "thickness_A": self.thickness,
            "area_per_lipid_A2": self.area_per_lipid,
            "total_area_A2": self.total_area,
            "total_lipids": self.total_lipids,
            "net_charge": self.net_charge,
            "charge_density_e_per_A2": self.charge_density,
            "spontaneous_curvature_per_A": self.spontaneous_curvature,
            "temperature_K": self.temperature,
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "MEMBRANE PROPERTIES",
            "=" * 50,
            "",
            f"Bending modulus (Kc):  {self.bending_modulus:.2f} kT",
            f"Gaussian modulus:      {self.gaussian_modulus:.2f} kT",
            f"Area compressibility:  {self.area_compressibility:.2f} kT/A^2",
            "",
            f"Thickness:        {self.thickness:.1f} A",
            f"Area per lipid:   {self.area_per_lipid:.1f} A^2",
            f"Total lipids:     {self.total_lipids}",
            f"Net charge:       {self.net_charge:+.1f} e",
            "",
            f"Temperature: {self.temperature:.1f} K",
            "=" * 50,
        ]
        return "\n".join(lines)


class MembranePhysics:
    """
    Computes membrane properties from lipid composition.
    Uses experimentally-derived values and mixing rules.
    """

    def __init__(self, temperature: float = 310.15):
        self.temperature = temperature
        self.kT = KB * temperature
        self.library = LipidLibrary()
        self._chol_stiffening = 0.5  # kT per mol% cholesterol

    def calculate(
        self,
        composition: Dict[str, int],
        box_area: Optional[float] = None,
        bending_modulus_override: Optional[float] = None,
    ) -> MembraneProperties:
        """Calculate membrane properties from lipid counts."""

        if not composition:
            raise ValueError("Empty composition")

        # Get lipid objects
        lipids_data = []
        for name, count in composition.items():
            if count <= 0:
                continue
            lipid = self.library.get(name)
            if lipid is None:
                raise ValueError(f"Unknown lipid: {name}")
            lipids_data.append((lipid, count))

        if not lipids_data:
            raise ValueError("No valid lipids")

        total_lipids = sum(count for _, count in lipids_data)

        # Weighted average area
        area_per_lipid = self._weighted_avg(lipids_data, lambda l: l.area_per_lipid)
        thickness = self._calc_thickness(lipids_data)

        if box_area is None:
            total_area = (total_lipids / 2) * area_per_lipid
        else:
            total_area = box_area

        # Bending modulus
        if bending_modulus_override is not None:
            bending_kT = bending_modulus_override
        else:
            bending_kT = self._calc_bending_modulus(lipids_data)

        bending_joules = bending_kT * self.kT

        # Gaussian modulus ~ -0.8 * Kc
        gaussian_kT = -0.8 * bending_kT

        # Area compressibility from Ka ~ 48*Kc/h^2
        h = thickness - 10  # hydrophobic thickness
        if h > 0:
            area_comp = 48 * bending_kT / (h * h)
        else:
            area_comp = 0.24

        net_charge = sum(lip.charge * count for lip, count in lipids_data)
        charge_density = net_charge / total_area if total_area > 0 else 0

        spont_curv = self._calc_spontaneous_curvature(lipids_data)

        return MembraneProperties(
            bending_modulus=bending_kT,
            bending_modulus_joules=bending_joules,
            gaussian_modulus=gaussian_kT,
            area_compressibility=area_comp,
            thickness=thickness,
            area_per_lipid=area_per_lipid,
            total_area=total_area,
            total_lipids=total_lipids,
            net_charge=net_charge,
            charge_density=charge_density,
            spontaneous_curvature=spont_curv,
            temperature=self.temperature,
        )

    def _weighted_avg(self, lipids_data, prop_func) -> float:
        total = sum(count for _, count in lipids_data)
        if total == 0:
            return 0.0
        return sum(prop_func(lip) * count for lip, count in lipids_data) / total

    def _calc_thickness(self, lipids_data) -> float:
        """Thickness with sterol ordering effects."""
        sterols = [(l, c) for l, c in lipids_data if l.category == LipidCategory.STEROL]
        non_sterols = [(l, c) for l, c in lipids_data if l.category != LipidCategory.STEROL]

        if not non_sterols:
            return 40.0

        base = self._weighted_avg(non_sterols, lambda l: l.thickness_contribution)

        total = sum(c for _, c in lipids_data)
        sterol_count = sum(c for _, c in sterols)
        sterol_frac = sterol_count / total if total > 0 else 0

        # Sterols condense neighbors, max ~10% increase at 50% sterol
        increase = base * 0.2 * sterol_frac * (1 - sterol_frac) * 4

        return base + increase

    def _calc_bending_modulus(self, lipids_data) -> float:
        """Bending modulus with cholesterol stiffening."""
        base_kc = self._weighted_avg(lipids_data, lambda l: l.bending_modulus)

        total = sum(c for _, c in lipids_data)
        sterols = [(l, c) for l, c in lipids_data if l.category == LipidCategory.STEROL]
        sterol_count = sum(c for _, c in sterols)
        sterol_pct = 100 * sterol_count / total if total > 0 else 0

        # Linear increase up to ~40 mol%
        eff_pct = min(sterol_pct, 40)
        enhancement = self._chol_stiffening * eff_pct

        return base_kc + enhancement

    def _calc_spontaneous_curvature(self, lipids_data) -> float:
        """Spontaneous curvature from lipid shape."""
        c0_values = {
            LipidCategory.PHOSPHATIDYLCHOLINE: 0.0,
            LipidCategory.PHOSPHATIDYLETHANOLAMINE: -0.003,
            LipidCategory.PHOSPHATIDYLSERINE: -0.001,
            LipidCategory.PHOSPHATIDYLGLYCEROL: 0.0,
            LipidCategory.PHOSPHATIDYLINOSITOL: 0.001,
            LipidCategory.SPHINGOMYELIN: 0.0,
            LipidCategory.STEROL: -0.002,
            LipidCategory.CARDIOLIPIN: -0.005,
            LipidCategory.CERAMIDE: -0.003,
        }

        total = sum(c for _, c in lipids_data)
        if total == 0:
            return 0.0

        weighted = sum(c0_values.get(l.category, 0.0) * c for l, c in lipids_data)
        return weighted / total

    def estimate_lipid_count(
        self,
        box_dims: Tuple[float, float],
        fractions: Dict[str, float],
    ) -> Dict[str, int]:
        """Estimate lipid counts to fill a box given mole fractions."""
        total_frac = sum(fractions.values())
        if abs(total_frac - 1.0) > 0.01:
            raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")

        lipids = []
        for name, frac in fractions.items():
            lipid = self.library.get(name)
            if lipid is None:
                raise ValueError(f"Unknown lipid: {name}")
            lipids.append((lipid, frac))

        avg_area = sum(l.area_per_lipid * f for l, f in lipids)
        box_area = box_dims[0] * box_dims[1]
        total = int(box_area / avg_area)

        counts = {}
        assigned = 0
        items = list(fractions.items())

        for name, frac in items[:-1]:
            count = round(total * frac)
            counts[name] = count
            assigned += count

        # Last one gets remainder
        counts[items[-1][0]] = total - assigned

        return counts


def quick_bending_modulus(composition: Dict[str, int]) -> float:
    """Quick Kc calculation."""
    return MembranePhysics().calculate(composition).bending_modulus
