"""
GROMACS Setup for TPU Membrane Simulations

Creates topology files, MDP parameter files, and run scripts
for molecular dynamics simulations of TPU membranes.
"""

import os
from typing import Dict, List, Tuple


def write_topology(output_dir: str, system_name: str = "TPU_membrane"):
    """
    Write GROMACS topology file (.top)
    Uses OPLS-AA force field parameters for polymers
    """
    top_content = f"""; Topology file for {system_name}
; Generated for TPU membrane simulation

; Force field
#include "oplsaa.ff/forcefield.itp"

; Custom molecule types for TPU segments
[ moleculetype ]
; Name    nrexcl
TPU       3

[ atoms ]
; nr  type  resnr residue atom cgnr charge  mass
; This will be filled by pdb2gmx or custom parameters

; Include position restraints if needed
#ifdef POSRES
#include "posre.itp"
#endif

[ system ]
; Name
{system_name}

[ molecules ]
; Compound  #mols
TPU         1
"""

    top_file = os.path.join(output_dir, "topol.top")
    with open(top_file, 'w') as f:
        f.write(top_content)

    print(f"Topology template: {top_file}")
    return top_file


def write_mdp_minimization(output_dir: str):
    """Write MDP file for energy minimization"""
    mdp_content = """; Energy Minimization Parameters
; minim.mdp

integrator  = steep         ; Steepest descent minimization
emtol       = 1000.0        ; Stop when max force < 1000 kJ/mol/nm
emstep      = 0.01          ; Energy step size
nsteps      = 50000         ; Maximum number of steps

; Neighbor searching
nstlist     = 10            ; Update neighbor list every 10 steps
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2

; Electrostatics
coulombtype = PME
rcoulomb    = 1.2

; VdW
vdwtype     = Cut-off
rvdw        = 1.2

; Periodic boundary conditions
pbc         = xyz

; Output control
nstxout     = 0
nstvout     = 0
nstenergy   = 500
nstlog      = 500
"""

    mdp_file = os.path.join(output_dir, "minim.mdp")
    with open(mdp_file, 'w') as f:
        f.write(mdp_content)

    print(f"Minimization MDP: {mdp_file}")
    return mdp_file


def write_mdp_nvt(output_dir: str, temperature: float = 310.0):
    """Write MDP file for NVT equilibration"""
    mdp_content = f"""; NVT Equilibration Parameters
; nvt.mdp

integrator  = md            ; Leap-frog integrator
nsteps      = 50000         ; 100 ps (2 fs timestep)
dt          = 0.002         ; 2 fs timestep

; Output control
nstxout     = 5000          ; Save coordinates every 10 ps
nstvout     = 5000          ; Save velocities every 10 ps
nstenergy   = 500           ; Save energies every 1 ps
nstlog      = 500           ; Update log every 1 ps
nstxout-compressed = 5000   ; Save compressed trajectory

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2

; Electrostatics
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16

; VdW
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres      ; Long-range dispersion correction

; Temperature coupling
tcoupl      = V-rescale     ; Velocity rescaling thermostat
tc-grps     = System
tau_t       = 0.1           ; Time constant (ps)
ref_t       = {temperature}  ; Reference temperature (K)

; Pressure coupling (off for NVT)
pcoupl      = no

; Periodic boundary conditions
pbc         = xyz

; Velocity generation
gen_vel     = yes           ; Generate velocities
gen_temp    = {temperature}  ; Temperature for velocity generation
gen_seed    = -1            ; Random seed

; Constraints
constraints = h-bonds       ; Constrain H-bonds
constraint_algorithm = lincs
lincs_iter  = 1
lincs_order = 4
"""

    mdp_file = os.path.join(output_dir, "nvt.mdp")
    with open(mdp_file, 'w') as f:
        f.write(mdp_content)

    print(f"NVT equilibration MDP: {mdp_file}")
    return mdp_file


def write_mdp_npt(output_dir: str, temperature: float = 310.0, pressure: float = 1.0):
    """Write MDP file for NPT equilibration"""
    mdp_content = f"""; NPT Equilibration Parameters
; npt.mdp

integrator  = md
nsteps      = 50000         ; 100 ps
dt          = 0.002

; Output control
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500
nstxout-compressed = 5000

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2

; Electrostatics
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16

; VdW
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres

; Temperature coupling
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}

; Pressure coupling
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0           ; Time constant (ps)
ref_p       = {pressure}     ; Reference pressure (bar)
compressibility = 4.5e-5    ; Compressibility (bar^-1)
refcoord_scaling = com

; Periodic boundary conditions
pbc         = xyz

; Velocity generation (continue from NVT)
gen_vel     = no

; Constraints
constraints = h-bonds
constraint_algorithm = lincs
lincs_iter  = 1
lincs_order = 4
"""

    mdp_file = os.path.join(output_dir, "npt.mdp")
    with open(mdp_file, 'w') as f:
        f.write(mdp_content)

    print(f"NPT equilibration MDP: {mdp_file}")
    return mdp_file


def write_mdp_production(output_dir: str, temperature: float = 310.0,
                         pressure: float = 1.0, nsteps: int = 5000000):
    """Write MDP file for production MD run"""
    mdp_content = f"""; Production MD Parameters
; md.mdp

integrator  = md
nsteps      = {nsteps}      ; {nsteps * 0.002 / 1000:.0f} ns total
dt          = 0.002         ; 2 fs timestep

; Output control
nstxout     = 0             ; Don't save full precision coords
nstvout     = 0             ; Don't save velocities
nstenergy   = 5000          ; Save energies every 10 ps
nstlog      = 5000          ; Update log every 10 ps
nstxout-compressed = 5000   ; Save compressed trajectory every 10 ps

; Neighbor searching
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2

; Electrostatics
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16

; VdW
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres

; Temperature coupling
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}

; Pressure coupling
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = {pressure}
compressibility = 4.5e-5
refcoord_scaling = com

; Periodic boundary conditions
pbc         = xyz

; Velocity generation
gen_vel     = no

; Constraints
constraints = h-bonds
constraint_algorithm = lincs
lincs_iter  = 1
lincs_order = 4
"""

    mdp_file = os.path.join(output_dir, "md.mdp")
    with open(mdp_file, 'w') as f:
        f.write(mdp_content)

    print(f"Production MD MDP: {mdp_file}")
    return mdp_file


def write_run_script(output_dir: str, structure_file: str):
    """Write bash script to run the full GROMACS simulation workflow"""
    script = f"""#!/bin/bash
# GROMACS simulation workflow for TPU membrane
# Run from the simulation directory

# Set GROMACS path if needed
# source /path/to/gromacs/bin/GMXRC

STRUCTURE="{os.path.basename(structure_file)}"
NAME="${{STRUCTURE%.pdb}}"

echo "=== TPU Membrane MD Simulation ==="
echo "Structure: $STRUCTURE"

# Step 1: Generate topology using pdb2gmx
echo "Step 1: Generating topology..."
gmx pdb2gmx -f $STRUCTURE -o processed.gro -water none -ff oplsaa << EOF
1
EOF

# Step 2: Define simulation box
echo "Step 2: Defining box..."
gmx editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic

# Step 3: Energy minimization
echo "Step 3: Energy minimization..."
gmx grompp -f minim.mdp -c box.gro -p topol.top -o em.tpr -maxwarn 10
gmx mdrun -v -deffnm em

# Step 4: NVT equilibration
echo "Step 4: NVT equilibration..."
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 10
gmx mdrun -v -deffnm nvt

# Step 5: NPT equilibration
echo "Step 5: NPT equilibration..."
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr -maxwarn 10
gmx mdrun -v -deffnm npt

# Step 6: Production MD
echo "Step 6: Production MD..."
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 10
gmx mdrun -v -deffnm md

echo "=== Simulation Complete ==="
echo "Trajectory: md.xtc"
echo "Final structure: md.gro"
"""

    script_file = os.path.join(output_dir, "run_simulation.sh")
    with open(script_file, 'w') as f:
        f.write(script)
    os.chmod(script_file, 0o755)

    print(f"Run script: {script_file}")
    return script_file


def setup_gromacs_simulation(structure_file: str, output_dir: str,
                             temperature: float = 310.0,
                             production_ns: float = 10.0):
    """
    Set up complete GROMACS simulation files

    Args:
        structure_file: Path to input PDB/GRO structure
        output_dir: Directory to write simulation files
        temperature: Simulation temperature in K (default 310 K = 37°C)
        production_ns: Production run length in nanoseconds
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate production steps (2 fs timestep)
    production_steps = int(production_ns * 1000 / 0.002)

    print(f"Setting up GROMACS simulation in: {output_dir}")
    print(f"Temperature: {temperature} K")
    print(f"Production run: {production_ns} ns ({production_steps} steps)")

    # Write all MDP files
    write_topology(output_dir)
    write_mdp_minimization(output_dir)
    write_mdp_nvt(output_dir, temperature)
    write_mdp_npt(output_dir, temperature)
    write_mdp_production(output_dir, temperature, nsteps=production_steps)
    write_run_script(output_dir, structure_file)

    print("\n=== Setup Complete ===")
    print(f"To run simulation:")
    print(f"  cd {output_dir}")
    print(f"  ./run_simulation.sh")


if __name__ == "__main__":
    # Example setup
    setup_gromacs_simulation(
        structure_file="./structures/membrane_Sparsa130_Carbosil170.pdb",
        output_dir="./simulation",
        temperature=310.0,
        production_ns=10.0
    )
