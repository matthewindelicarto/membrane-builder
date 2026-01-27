"""
Lipid library with physical properties for common membrane lipids.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class LipidCategory(Enum):
    PHOSPHATIDYLCHOLINE = "PC"
    PHOSPHATIDYLETHANOLAMINE = "PE"
    PHOSPHATIDYLSERINE = "PS"
    PHOSPHATIDYLGLYCEROL = "PG"
    PHOSPHATIDYLINOSITOL = "PI"
    SPHINGOMYELIN = "SM"
    STEROL = "STEROL"
    CARDIOLIPIN = "CL"
    CERAMIDE = "CER"


@dataclass
class Lipid:
    """A lipid with its physical properties."""
    name: str
    full_name: str
    category: LipidCategory
    molecular_weight: float
    charge: float
    area_per_lipid: float  # Angstroms^2
    thickness_contribution: float  # Angstroms
    bending_modulus: float  # kT
    tail_carbons: Tuple[int, int] = (16, 18)
    tail_unsaturations: Tuple[int, int] = (0, 1)
    headgroup_atoms: int = 8
    anchor_atom: str = "P"
    description: str = ""
    color: str = "#3498db"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "category": self.category.value,
            "molecular_weight": self.molecular_weight,
            "charge": self.charge,
            "area_per_lipid": self.area_per_lipid,
            "thickness_contribution": self.thickness_contribution,
            "bending_modulus": self.bending_modulus,
            "tail_carbons": list(self.tail_carbons),
            "tail_unsaturations": list(self.tail_unsaturations),
            "headgroup_atoms": self.headgroup_atoms,
            "anchor_atom": self.anchor_atom,
            "description": self.description,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lipid":
        data = data.copy()
        data["category"] = LipidCategory(data["category"])
        data["tail_carbons"] = tuple(data["tail_carbons"])
        data["tail_unsaturations"] = tuple(data["tail_unsaturations"])
        return cls(**data)


class LipidLibrary:
    """Library of membrane lipids with their properties."""

    def __init__(self):
        self._lipids: Dict[str, Lipid] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load built-in lipid definitions."""

        # PC lipids
        self.add(Lipid(
            name="POPC",
            full_name="1-palmitoyl-2-oleoyl-sn-glycero-3-phosphocholine",
            category=LipidCategory.PHOSPHATIDYLCHOLINE,
            molecular_weight=760.08,
            charge=0.0,
            area_per_lipid=68.3,
            thickness_contribution=36.5,
            bending_modulus=20.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=8,
            anchor_atom="P",
            description="Common PC, one saturated and one unsaturated tail",
            color="#3498db",
        ))

        self.add(Lipid(
            name="DPPC",
            full_name="1,2-dipalmitoyl-sn-glycero-3-phosphocholine",
            category=LipidCategory.PHOSPHATIDYLCHOLINE,
            molecular_weight=734.04,
            charge=0.0,
            area_per_lipid=63.0,
            thickness_contribution=38.0,
            bending_modulus=25.0,
            tail_carbons=(16, 16),
            tail_unsaturations=(0, 0),
            headgroup_atoms=8,
            anchor_atom="P",
            description="Saturated PC, gel phase at room temp",
            color="#2980b9",
        ))

        self.add(Lipid(
            name="DOPC",
            full_name="1,2-dioleoyl-sn-glycero-3-phosphocholine",
            category=LipidCategory.PHOSPHATIDYLCHOLINE,
            molecular_weight=786.11,
            charge=0.0,
            area_per_lipid=72.5,
            thickness_contribution=35.0,
            bending_modulus=18.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(1, 1),
            headgroup_atoms=8,
            anchor_atom="P",
            description="Unsaturated PC, very fluid",
            color="#1abc9c",
        ))

        self.add(Lipid(
            name="DMPC",
            full_name="1,2-dimyristoyl-sn-glycero-3-phosphocholine",
            category=LipidCategory.PHOSPHATIDYLCHOLINE,
            molecular_weight=677.93,
            charge=0.0,
            area_per_lipid=60.6,
            thickness_contribution=35.3,
            bending_modulus=22.0,
            tail_carbons=(14, 14),
            tail_unsaturations=(0, 0),
            headgroup_atoms=8,
            anchor_atom="P",
            description="Short saturated PC",
            color="#5dade2",
        ))

        self.add(Lipid(
            name="SOPC",
            full_name="1-stearoyl-2-oleoyl-sn-glycero-3-phosphocholine",
            category=LipidCategory.PHOSPHATIDYLCHOLINE,
            molecular_weight=788.13,
            charge=0.0,
            area_per_lipid=67.0,
            thickness_contribution=37.5,
            bending_modulus=21.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=8,
            anchor_atom="P",
            description="Like POPC but longer saturated tail",
            color="#85c1e9",
        ))

        # PE lipids
        self.add(Lipid(
            name="POPE",
            full_name="1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoethanolamine",
            category=LipidCategory.PHOSPHATIDYLETHANOLAMINE,
            molecular_weight=718.00,
            charge=0.0,
            area_per_lipid=58.8,
            thickness_contribution=38.5,
            bending_modulus=24.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=5,
            anchor_atom="P",
            description="Common in bacteria and mitochondria",
            color="#e74c3c",
        ))

        self.add(Lipid(
            name="DOPE",
            full_name="1,2-dioleoyl-sn-glycero-3-phosphoethanolamine",
            category=LipidCategory.PHOSPHATIDYLETHANOLAMINE,
            molecular_weight=744.03,
            charge=0.0,
            area_per_lipid=65.0,
            thickness_contribution=36.0,
            bending_modulus=20.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(1, 1),
            headgroup_atoms=5,
            anchor_atom="P",
            description="Promotes negative curvature",
            color="#c0392b",
        ))

        # PS lipids (anionic)
        self.add(Lipid(
            name="POPS",
            full_name="1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoserine",
            category=LipidCategory.PHOSPHATIDYLSERINE,
            molecular_weight=762.00,
            charge=-1.0,
            area_per_lipid=55.0,
            thickness_contribution=37.0,
            bending_modulus=22.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=7,
            anchor_atom="P",
            description="Anionic, normally inner leaflet",
            color="#9b59b6",
        ))

        self.add(Lipid(
            name="DOPS",
            full_name="1,2-dioleoyl-sn-glycero-3-phosphoserine",
            category=LipidCategory.PHOSPHATIDYLSERINE,
            molecular_weight=788.03,
            charge=-1.0,
            area_per_lipid=65.3,
            thickness_contribution=35.5,
            bending_modulus=19.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(1, 1),
            headgroup_atoms=7,
            anchor_atom="P",
            description="Unsaturated anionic lipid",
            color="#8e44ad",
        ))

        # PG lipids (bacterial)
        self.add(Lipid(
            name="POPG",
            full_name="1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoglycerol",
            category=LipidCategory.PHOSPHATIDYLGLYCEROL,
            molecular_weight=748.97,
            charge=-1.0,
            area_per_lipid=66.0,
            thickness_contribution=36.0,
            bending_modulus=20.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=6,
            anchor_atom="P",
            description="Major bacterial anionic lipid",
            color="#27ae60",
        ))

        self.add(Lipid(
            name="DOPG",
            full_name="1,2-dioleoyl-sn-glycero-3-phosphoglycerol",
            category=LipidCategory.PHOSPHATIDYLGLYCEROL,
            molecular_weight=775.00,
            charge=-1.0,
            area_per_lipid=70.0,
            thickness_contribution=34.5,
            bending_modulus=17.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(1, 1),
            headgroup_atoms=6,
            anchor_atom="P",
            description="Unsaturated PG for bacterial membranes",
            color="#2ecc71",
        ))

        # Sphingomyelins
        self.add(Lipid(
            name="PSM",
            full_name="N-palmitoyl-sphingomyelin",
            category=LipidCategory.SPHINGOMYELIN,
            molecular_weight=702.97,
            charge=0.0,
            area_per_lipid=52.0,
            thickness_contribution=42.0,
            bending_modulus=30.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=9,
            anchor_atom="P",
            description="Forms rafts with cholesterol",
            color="#f39c12",
        ))

        self.add(Lipid(
            name="SSM",
            full_name="N-stearoyl-sphingomyelin",
            category=LipidCategory.SPHINGOMYELIN,
            molecular_weight=731.02,
            charge=0.0,
            area_per_lipid=50.0,
            thickness_contribution=44.0,
            bending_modulus=32.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=9,
            anchor_atom="P",
            description="Long-chain SM, very ordered",
            color="#e67e22",
        ))

        # Sterols
        self.add(Lipid(
            name="CHOL",
            full_name="Cholesterol",
            category=LipidCategory.STEROL,
            molecular_weight=386.65,
            charge=0.0,
            area_per_lipid=40.0,
            thickness_contribution=0.0,
            bending_modulus=5.0,
            tail_carbons=(0, 0),
            tail_unsaturations=(0, 0),
            headgroup_atoms=1,
            anchor_atom="O3",
            description="Modulates fluidity, forms rafts",
            color="#f1c40f",
        ))

        self.add(Lipid(
            name="ERGO",
            full_name="Ergosterol",
            category=LipidCategory.STEROL,
            molecular_weight=396.65,
            charge=0.0,
            area_per_lipid=38.0,
            thickness_contribution=0.0,
            bending_modulus=6.0,
            tail_carbons=(0, 0),
            tail_unsaturations=(0, 0),
            headgroup_atoms=1,
            anchor_atom="O3",
            description="Fungal sterol, antifungal target",
            color="#d4ac0d",
        ))

        # PI
        self.add(Lipid(
            name="POPI",
            full_name="1-palmitoyl-2-oleoyl-sn-glycero-3-phosphoinositol",
            category=LipidCategory.PHOSPHATIDYLINOSITOL,
            molecular_weight=854.05,
            charge=-1.0,
            area_per_lipid=70.0,
            thickness_contribution=35.0,
            bending_modulus=18.0,
            tail_carbons=(16, 18),
            tail_unsaturations=(0, 1),
            headgroup_atoms=12,
            anchor_atom="P",
            description="Signaling lipid precursor",
            color="#16a085",
        ))

        # Cardiolipin
        self.add(Lipid(
            name="TOCL",
            full_name="Tetraoleoyl cardiolipin",
            category=LipidCategory.CARDIOLIPIN,
            molecular_weight=1501.98,
            charge=-2.0,
            area_per_lipid=130.0,
            thickness_contribution=34.0,
            bending_modulus=15.0,
            tail_carbons=(18, 18),
            tail_unsaturations=(1, 1),
            headgroup_atoms=10,
            anchor_atom="P",
            description="Mitochondrial lipid, 4 tails",
            color="#1e8449",
        ))

    def add(self, lipid: Lipid) -> None:
        self._lipids[lipid.name.upper()] = lipid

    def get(self, name: str) -> Optional[Lipid]:
        return self._lipids.get(name.upper())

    def get_by_category(self, category: LipidCategory) -> List[Lipid]:
        return [lip for lip in self._lipids.values() if lip.category == category]

    def list_all(self) -> List[str]:
        return sorted(self._lipids.keys())

    def list_categories(self) -> List[str]:
        return [cat.name for cat in LipidCategory]

    def search(self, query: str) -> List[Lipid]:
        query = query.lower()
        results = []
        for lipid in self._lipids.values():
            if (query in lipid.name.lower() or
                query in lipid.full_name.lower() or
                query in lipid.description.lower()):
                results.append(lipid)
        return results

    def to_dict(self) -> Dict[str, dict]:
        return {name: lip.to_dict() for name, lip in self._lipids.items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __len__(self) -> int:
        return len(self._lipids)

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._lipids

    def __iter__(self):
        return iter(self._lipids.values())


def get_lipid(name: str) -> Optional[Lipid]:
    """Get a lipid from the default library."""
    return LipidLibrary().get(name)
