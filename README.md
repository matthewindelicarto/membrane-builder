# Membrane Builder

Build lipid bilayer membranes and predict molecular permeability.

## Quick Start

Build a membrane:
```bash
python run.py
```

Predict permeability:
```bash
python membrane_runner/run_analysis.py
```

Install dependencies:
```bash
pip install numpy matplotlib
```

---

## Building Membranes

Edit `args.txt` to configure your membrane:

```
box_x = 80.0
box_y = 80.0
box_z = 120.0

POPC = 64, 64
CHOL = 32, 32
```

The format for lipids is `NAME = top_leaflet, bottom_leaflet`.

Run `python run.py` to generate:
- `Outputs/membrane.pdb` — view in PyMOL
- `Outputs/membrane.gro` — for GROMACS
- `Outputs/membrane_report.txt` — properties

### Common Lipids

| Name | Type | Notes |
|------|------|-------|
| POPC | PC | Standard membrane lipid |
| DOPC | PC | Unsaturated, fluid |
| DPPC | PC | Saturated, rigid |
| POPE | PE | Bacterial membranes |
| POPS | PS | Anionic, inner leaflet |
| CHOL | Sterol | Stiffens membrane |
| PSM | SM | Sphingomyelin |

### Example: Plasma Membrane

```
box_x = 100.0
box_y = 100.0
box_z = 120.0

POPC = 50, 20
PSM = 20, 0
POPE = 0, 40
POPS = 0, 20
CHOL = 30, 30
```

### Python API

```python
from membrane_builder import MembraneBuilder

membrane = MembraneBuilder.quick_build(
    lipids={"POPC": (64, 64), "CHOL": (32, 32)},
    box_size=(80, 80),
)

membrane.write_pdb("membrane.pdb")
print(f"Thickness: {membrane.properties.thickness} Å")
```

---

## Permeability Prediction

Predicts how easily molecules cross membranes.

```bash
python membrane_runner/run_analysis.py
python membrane_runner/run_analysis.py --membrane-type BBB --cholesterol 0.3
```

Results go to `permeability_results/`:
- PNG plots showing energy profiles
- Text report with log P values

### Python API

```python
from membrane_builder import MoleculeDescriptor, PermeabilityPredictor

caffeine = MoleculeDescriptor.simple(
    name="caffeine",
    molecular_weight=194.2,
    total_asa=150.0,
    n_hbd=0,
    n_hba=3,
)

predictor = PermeabilityPredictor()
result = predictor.calculate(caffeine, "Caco-2")

print(f"log P = {result.log_p:.2f}")
```

### Membrane Types

| Type | Use |
|------|-----|
| BLM | General permeability |
| PAMPA-DS | Drug screening |
| BBB | Brain penetration |
| Caco-2 | Oral absorption |

---

## How It Works

### What is membrane permeability?

When a molecule crosses a membrane, it has to pass through the middle part called the hydrocarbon core. This region is about 30 Å thick and has very different properties than water:

- No water molecules to form hydrogen bonds with
- Low dielectric constant (ε ≈ 2 instead of 78 in water)

Polar molecules (ones with OH groups, NH groups, or charges) don't like being in this environment. They lose all their hydrogen bonds and electrostatic stabilization. This creates an energy barrier that slows them down.

Nonpolar molecules (hydrophobic ones) actually prefer this environment. They cross easily.

### How we calculate it

We calculate the "transfer energy" at each position as the molecule moves through the membrane. This tells us how much the molecule likes or dislikes being at that depth.

The transfer energy has three parts:

1. Solvation energy: Each atom on the molecule surface contributes. Carbon atoms like the membrane core (negative energy, favorable). Oxygen and nitrogen atoms dislike it (positive energy, unfavorable) because they lose hydrogen bonds.

2. Dipole penalty: If the molecule has polar groups with dipole moments, they pay an extra penalty in the low-dielectric core.

3. Ionization: If the molecule can gain or lose a proton (like acids and bases), we check whether it's better to be charged or neutral at each position. Being charged in the membrane core is very unfavorable due to the Born energy.

Once we have the energy at each position, we calculate the partition coefficient:

```
K(z) = exp(-ΔG(z) / RT)
```

This tells us how much the molecule prefers that position compared to water. We then integrate across the membrane to get the permeability.

### Membrane structure

The membrane isn't uniform. Properties change with depth:

| Region | Distance from center | Dielectric | H-bonding |
|--------|---------------------|------------|-----------|
| Core | 0–10 Å | ~2 | None |
| Interface | 10–15 Å | 3–10 | Low |
| Headgroups | 15–25 Å | 20–40 | Medium |
| Water | >25 Å | 78 | High |

These profiles come from X-ray and neutron scattering experiments on real lipid bilayers.

### How lipid composition affects permeability

Cholesterol orders the lipid tails and makes the membrane more rigid. This reduces permeability by 2-5x at 30 mol%.

Unsaturated lipids (like DOPC with double bonds) make the membrane more disordered. More disorder means higher permeability.

Sphingomyelin packs tightly with cholesterol, further reducing permeability.

### Calibration

The raw calculations are calibrated against experimental data from different assay types:

| Membrane | What it models |
|----------|----------------|
| BLM | Black lipid membrane, simple bilayer |
| PAMPA-DS | Artificial membrane used in drug screening |
| BBB | Blood-brain barrier (tighter, more cholesterol) |
| Caco-2 | Intestinal cells (includes some active transport effects) |

Each has different calibration parameters because they have different lipid compositions and effective barrier thicknesses.

---

## Examples

### Why glucose doesn't cross membranes

Ethanol has one OH group. Glucose has five.

```python
from membrane_builder import MoleculeDescriptor, quick_permeability

ethanol = MoleculeDescriptor.ethanol()   # MW 46, 1 OH
glucose = MoleculeDescriptor.glucose()   # MW 180, 5 OH

print(f"Ethanol: log P = {quick_permeability(ethanol):.1f}")  # -9.1
print(f"Glucose: log P = {quick_permeability(glucose):.1f}")  # -10.9
```

Glucose is about 60x less permeable. Each OH group that enters the membrane core loses hydrogen bonds it was making with water. Five OH groups means a huge energy penalty.

This is why cells need glucose transporters — glucose can't get in by passive diffusion.

### How cholesterol changes things

```python
from membrane_builder import MoleculeDescriptor, PermeabilityPredictor

drug = MoleculeDescriptor.simple(
    name="drug",
    molecular_weight=300,
    total_asa=200,
    n_hbd=2,
    n_hba=4
)

# Pure POPC membrane
p1 = PermeabilityPredictor(composition={"POPC": 128})
r1 = p1.calculate(drug, "BLM")

# Add 30% cholesterol
p2 = PermeabilityPredictor(composition={"POPC": 90, "CHOL": 38})
r2 = p2.calculate(drug, "BLM")

print(f"No cholesterol:  log P = {r1.log_p:.2f}")
print(f"30% cholesterol: log P = {r2.log_p:.2f}")
```

Adding cholesterol stiffens the membrane and slightly reduces permeability.

### Where drugs like to sit

The energy profile tells you where a molecule prefers to be:

```python
result = predictor.calculate(drug, "BLM")

print(f"Binding energy: {result.membrane_bound_energy:.1f} kJ/mol")
print(f"Binding position: {result.binding_position:.1f} Å from center")
```

Many drugs have an energy minimum at the interface (around 12-15 Å from center). They can bury their hydrophobic parts in the membrane while keeping polar groups near the headgroup region where there's still some water and H-bonding.

---

## What this model doesn't capture

- Active transport: Pumps and transporters that move molecules against their gradient
- Efflux: P-glycoprotein and other pumps that kick molecules back out
- Paracellular transport: Molecules sneaking between cells instead of through them
- Conformational flexibility: We use one shape instead of sampling all possible conformations
- Membrane heterogeneity: Real membranes have lipid rafts and domains

---

## References

1. Lomize & Pogozheva. Physics-based method for modeling passive membrane permeability. J Chem Inf Model 2019.

2. Nagle & Tristram-Nagle. Structure of lipid bilayers. Biochim Biophys Acta 2000.
