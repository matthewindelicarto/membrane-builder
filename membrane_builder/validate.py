"""
Validates membrane configurations before building.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

from .lipids import LipidLibrary, LipidCategory


class MessageLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationMessage:
    level: MessageLevel
    category: str
    message: str
    suggestion: str = ""

    def __str__(self) -> str:
        s = f"[{self.level.value}] {self.category}: {self.message}"
        if self.suggestion:
            s += f"\n         -> {self.suggestion}"
        return s


class MembraneValidator:
    """Checks membrane configs for issues before building."""

    def __init__(self):
        self.library = LipidLibrary()
        self.messages: List[ValidationMessage] = []

    def validate(
        self,
        lipids: Dict[str, Tuple[int, int]],
        box_x: float,
        box_y: float,
        min_anchor_dist: float = 6.5,
    ) -> List[ValidationMessage]:
        """Run all validation checks."""
        self.messages = []
        box_area = box_x * box_y

        self._check_lipid_names(lipids)
        stats = self._calc_stats(lipids, box_area)
        self._check_packing(stats, box_area, min_anchor_dist)
        self._check_leaflet_balance(stats)
        self._check_charge(stats, box_area)
        self._check_cholesterol(stats)
        self._check_composition(stats)
        self._check_anionic_placement(lipids)

        return self.messages

    def _calc_stats(self, lipids: Dict[str, Tuple[int, int]], box_area: float) -> dict:
        total_top = sum(top for top, _ in lipids.values())
        total_bottom = sum(bot for _, bot in lipids.values())
        total = total_top + total_bottom

        weighted_area_top = 0.0
        weighted_area_bottom = 0.0
        charge_top = 0.0
        charge_bottom = 0.0
        chol_count = 0
        anionic_count = 0
        cats_top: Dict[str, int] = {}
        cats_bottom: Dict[str, int] = {}

        for name, (top, bot) in lipids.items():
            lip = self.library.get(name)
            if lip:
                weighted_area_top += lip.area_per_lipid * top
                weighted_area_bottom += lip.area_per_lipid * bot
                charge_top += lip.charge * top
                charge_bottom += lip.charge * bot

                if lip.category == LipidCategory.STEROL:
                    chol_count += top + bot
                if lip.charge < 0:
                    anionic_count += top + bot

                cat = lip.category.name
                cats_top[cat] = cats_top.get(cat, 0) + top
                cats_bottom[cat] = cats_bottom.get(cat, 0) + bot

        avg_area_top = weighted_area_top / total_top if total_top > 0 else 0
        avg_area_bottom = weighted_area_bottom / total_bottom if total_bottom > 0 else 0

        return {
            "total": total,
            "total_top": total_top,
            "total_bottom": total_bottom,
            "avg_area_top": avg_area_top,
            "avg_area_bottom": avg_area_bottom,
            "charge_top": charge_top,
            "charge_bottom": charge_bottom,
            "net_charge": charge_top + charge_bottom,
            "chol_count": chol_count,
            "chol_frac": chol_count / total if total > 0 else 0,
            "anionic_count": anionic_count,
            "anionic_frac": anionic_count / total if total > 0 else 0,
            "cats_top": cats_top,
            "cats_bottom": cats_bottom,
            "box_area": box_area,
        }

    def _check_lipid_names(self, lipids: Dict[str, Tuple[int, int]]):
        for name in lipids:
            if self.library.get(name) is None:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.ERROR,
                    category="Lipid",
                    message=f"Unknown lipid: '{name}'",
                    suggestion=f"Available: {', '.join(self.library.list_all())}"
                ))

    def _check_packing(self, stats: dict, box_area: float, min_dist: float):
        if stats["total_top"] > 0:
            actual = box_area / stats["total_top"]
            expected = stats["avg_area_top"]

            if actual < expected * 0.7:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.ERROR,
                    category="Packing Density",
                    message=f"Top leaflet overpacked: {actual:.1f} A^2/lipid (expected ~{expected:.1f})",
                    suggestion="Reduce lipid count or increase box size"
                ))
            elif actual < expected * 0.85:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.WARNING,
                    category="Packing Density",
                    message=f"Top leaflet tight: {actual:.1f} A^2/lipid (typical ~{expected:.1f})",
                    suggestion="May need longer equilibration"
                ))
            elif actual > expected * 1.3:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.WARNING,
                    category="Packing Density",
                    message=f"Top leaflet underpacked: {actual:.1f} A^2/lipid (typical ~{expected:.1f})",
                    suggestion="Consider adding more lipids or reducing box size"
                ))

        if stats["total_bottom"] > 0:
            actual = box_area / stats["total_bottom"]
            expected = stats["avg_area_bottom"]

            if actual < expected * 0.7:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.ERROR,
                    category="Packing Density",
                    message=f"Bottom leaflet overpacked: {actual:.1f} A^2/lipid (expected ~{expected:.1f})",
                    suggestion="Reduce lipid count or increase box size"
                ))
            elif actual < expected * 0.85:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.WARNING,
                    category="Packing Density",
                    message=f"Bottom leaflet tight: {actual:.1f} A^2/lipid (typical ~{expected:.1f})",
                    suggestion="May need longer equilibration"
                ))

        # Check geometry
        min_area = (min_dist ** 2) * 0.866  # hex packing
        if stats["total_top"] > 0:
            actual = box_area / stats["total_top"]
            if actual < min_area:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.ERROR,
                    category="Packing Geometry",
                    message=f"Can't fit lipids with min_anchor_distance={min_dist} A",
                    suggestion=f"Need {min_area:.1f} A^2/lipid, have {actual:.1f}"
                ))

    def _check_leaflet_balance(self, stats: dict):
        top = stats["total_top"]
        bot = stats["total_bottom"]

        if top == 0 or bot == 0:
            if top == 0 and bot == 0:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.ERROR,
                    category="Composition",
                    message="No lipids specified",
                    suggestion="Add lipids to the configuration"
                ))
            else:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.WARNING,
                    category="Leaflet Balance",
                    message="Building a monolayer (one leaflet empty)",
                    suggestion="Confirm this is intentional"
                ))
            return

        imbalance = abs(top - bot) / max(top, bot)
        if imbalance > 0.2:
            self.messages.append(ValidationMessage(
                level=MessageLevel.WARNING,
                category="Leaflet Balance",
                message=f"Large asymmetry: {top} top vs {bot} bottom ({imbalance*100:.0f}%)",
                suggestion="May cause curvature or instability"
            ))
        elif imbalance > 0.1:
            self.messages.append(ValidationMessage(
                level=MessageLevel.INFO,
                category="Leaflet Balance",
                message=f"Slight asymmetry: {top} top vs {bot} bottom",
                suggestion="Often biologically realistic"
            ))

    def _check_charge(self, stats: dict, box_area: float):
        net = stats["net_charge"]
        density = abs(net) / box_area * 1000

        if abs(net) > 50:
            self.messages.append(ValidationMessage(
                level=MessageLevel.WARNING,
                category="Charge Balance",
                message=f"High net charge: {net:+.0f} e",
                suggestion="Consider adding counterions"
            ))
        elif abs(net) > 20:
            self.messages.append(ValidationMessage(
                level=MessageLevel.INFO,
                category="Charge Balance",
                message=f"Moderate net charge: {net:+.0f} e",
                suggestion="Ensure adequate ionic strength"
            ))

        if density > 0.5:
            self.messages.append(ValidationMessage(
                level=MessageLevel.WARNING,
                category="Charge Density",
                message=f"High charge density: {density:.2f} e per 1000 A^2",
                suggestion="May need careful equilibration"
            ))

        if stats["charge_top"] != 0 or stats["charge_bottom"] != 0:
            if abs(stats["charge_top"] - stats["charge_bottom"]) > 10:
                self.messages.append(ValidationMessage(
                    level=MessageLevel.INFO,
                    category="Charge Distribution",
                    message=f"Asymmetric charge: top={stats['charge_top']:+.0f}, bottom={stats['charge_bottom']:+.0f}",
                    suggestion="Mimics biological membranes (PS in inner leaflet)"
                ))

    def _check_cholesterol(self, stats: dict):
        frac = stats["chol_frac"]
        if frac > 0.5:
            self.messages.append(ValidationMessage(
                level=MessageLevel.WARNING,
                category="Cholesterol",
                message=f"Very high cholesterol: {frac*100:.0f}%",
                suggestion=">50% may cause phase separation"
            ))
        elif frac > 0.4:
            self.messages.append(ValidationMessage(
                level=MessageLevel.INFO,
                category="Cholesterol",
                message=f"High cholesterol: {frac*100:.0f}%",
                suggestion="Upper range of physiological"
            ))

    def _check_composition(self, stats: dict):
        cats_top = stats["cats_top"]
        cats_bottom = stats["cats_bottom"]

        sterols = cats_top.get("STEROL", 0) + cats_bottom.get("STEROL", 0)
        non_sterols = stats["total"] - sterols

        if non_sterols == 0 and sterols > 0:
            self.messages.append(ValidationMessage(
                level=MessageLevel.ERROR,
                category="Composition",
                message="Pure sterol membrane - sterols can't form bilayers alone",
                suggestion="Add phospholipids (PC, PE, etc.)"
            ))

        pc_pe = (
            cats_top.get("PHOSPHATIDYLCHOLINE", 0) +
            cats_bottom.get("PHOSPHATIDYLCHOLINE", 0) +
            cats_top.get("PHOSPHATIDYLETHANOLAMINE", 0) +
            cats_bottom.get("PHOSPHATIDYLETHANOLAMINE", 0)
        )
        if pc_pe == 0 and stats["total"] > 0:
            self.messages.append(ValidationMessage(
                level=MessageLevel.INFO,
                category="Composition",
                message="No PC or PE lipids",
                suggestion="Most membranes are PC/PE-based"
            ))

    def _check_anionic_placement(self, lipids: Dict[str, Tuple[int, int]]):
        anionic_top = 0
        anionic_bot = 0

        for name, (top, bot) in lipids.items():
            lip = self.library.get(name)
            if lip and lip.charge < 0:
                anionic_top += top
                anionic_bot += bot

        if anionic_top > 0 and anionic_bot == 0:
            self.messages.append(ValidationMessage(
                level=MessageLevel.INFO,
                category="Lipid Distribution",
                message="Anionic lipids only in top leaflet",
                suggestion="In cells, PS is typically in the inner leaflet"
            ))

    def print_report(self) -> str:
        if not self.messages:
            return ""

        lines = []
        errors = [m for m in self.messages if m.level == MessageLevel.ERROR]
        warnings = [m for m in self.messages if m.level == MessageLevel.WARNING]
        infos = [m for m in self.messages if m.level == MessageLevel.INFO]

        for msg in errors:
            lines.append(f"  [ERROR] {msg.category}: {msg.message}")
            if msg.suggestion:
                lines.append(f"          {msg.suggestion}")

        for msg in warnings:
            lines.append(f"  [WARN]  {msg.category}: {msg.message}")
            if msg.suggestion:
                lines.append(f"          {msg.suggestion}")

        for msg in infos:
            lines.append(f"  [INFO]  {msg.category}: {msg.message}")
            if msg.suggestion:
                lines.append(f"          {msg.suggestion}")

        return "\n".join(lines)

    def has_errors(self) -> bool:
        return any(m.level == MessageLevel.ERROR for m in self.messages)

    def has_warnings(self) -> bool:
        return any(m.level == MessageLevel.WARNING for m in self.messages)
