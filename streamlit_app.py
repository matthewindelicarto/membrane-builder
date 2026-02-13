import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import zipfile
from io import BytesIO

from membrane_builder import MembraneBuilder, MembraneConfig, MoleculeDescriptor, PermeabilityPredictor
from tpu_builder import (
    TPUMembraneBuilder,
    TPUMembraneConfig,
    TPUPermeabilityPredictor,
    MoleculeDescriptor as TPUMoleculeDescriptor
)

st.set_page_config(
    page_title="Membrane Builder",
    layout="wide"
)

# Create main tabs
tab_lipid, tab_tpu, tab_md = st.tabs(["Lipid Membranes", "TPU Membranes", "MD Simulation"])

# ============== SESSION STATE ==============
if 'pdb_data' not in st.session_state:
    st.session_state.pdb_data = None
if 'membrane_props' not in st.session_state:
    st.session_state.membrane_props = None
if 'perm_result' not in st.session_state:
    st.session_state.perm_result = None
if 'tpu_membrane' not in st.session_state:
    st.session_state.tpu_membrane = None
if 'tpu_perm_result' not in st.session_state:
    st.session_state.tpu_perm_result = None
if 'md_files' not in st.session_state:
    st.session_state.md_files = None

# ============== LIPID MEMBRANE HELPER FUNCTIONS ==============

STANDARD_APL = {
    "POPC": 68.3, "DOPC": 72.4, "DPPC": 63.0, "POPE": 58.8,
    "POPG": 66.0, "POPS": 65.0, "CHOL": 40.0, "PSM": 52.0,
}

def calculate_effective_apl(lipid_counts, chol_count, total_lipids):
    if total_lipids == 0:
        return 65.0
    chol_fraction = chol_count / total_lipids if total_lipids > 0 else 0
    condensation_factor = max(0.75, 1.0 - (0.4 * chol_fraction))
    total_area = 0
    for lip, count in lipid_counts.items():
        if lip == "CHOL":
            total_area += count * STANDARD_APL.get("CHOL", 40.0)
        else:
            base_apl = STANDARD_APL.get(lip, 65.0)
            total_area += count * base_apl * condensation_factor
    return total_area / total_lipids if total_lipids > 0 else 65.0

def calculate_box_dimensions(lipid_values):
    top_count = sum(v[0] for v in lipid_values.values())
    bottom_count = sum(v[1] for v in lipid_values.values())
    max_leaflet = max(top_count, bottom_count)
    if max_leaflet == 0:
        return 80, 80
    total_counts = {lip: v[0] + v[1] for lip, v in lipid_values.items()}
    total_lipids = sum(total_counts.values())
    chol_count = total_counts.get("CHOL", 0)
    effective_apl = calculate_effective_apl(total_counts, chol_count, total_lipids)
    required_area = max_leaflet * effective_apl
    box_side = int(np.ceil(np.sqrt(required_area) / 5) * 5) + 5
    return max(40, min(200, box_side)), max(40, min(200, box_side))

def render_lipid_3dmol(pdb_data, style="stick", molecule_info=None):
    mol_colors = {"water": "0x3498db", "ethanol": "0x9b59b6", "caffeine": "0x8B4513",
                  "aspirin": "0xe74c3c", "glucose": "0xf39c12", "custom": "0x1abc9c"}
    mol_sphere_js = ""
    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "0x1abc9c")
        z_pos = molecule_info['z']
        mol_sphere_js = f"viewer.addSphere({{center: {{x: 0, y: 0, z: {z_pos}}}, radius: 3, color: {color}, opacity: 0.9}});"

    style_js = {'stick': 'viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.25}});',
                'line': 'viewer.setStyle({}, {line: {linewidth: 1.5}});',
                'sphere': 'viewer.setStyle({}, {sphere: {scale: 0.4}});'}.get(style, '')
    pdb_escaped = pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer_lipid" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer_lipid", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");
        {style_js}
        {mol_sphere_js}
        viewer.zoomTo(); viewer.render();
    </script>
    """
    components.html(html, height=520)

# ============== TPU MEMBRANE HELPER FUNCTIONS ==============

def generate_tpu_polymer_chains(membrane, sparsa_frac, carbosil_frac):
    atoms = []
    atom_id = 1
    res_id = 1
    np.random.seed(42)
    box_x, box_y, box_z = 40, 40, 15

    def add_atom(element, x, y, z, res_name):
        nonlocal atom_id, res_id
        x, y, z = max(-box_x/2, min(box_x/2, x)), max(-box_y/2, min(box_y/2, y)), max(-box_z/2, min(box_z/2, z))
        atoms.append({'id': atom_id, 'name': element, 'res_name': res_name, 'res_id': res_id, 'x': x, 'y': y, 'z': z, 'element': element})
        atom_id += 1

    def random_direction():
        theta, phi = np.random.uniform(0, 2*np.pi), np.random.uniform(np.pi/3, 2*np.pi/3)
        return (np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)*0.3)

    def step(x, y, z, bond_len, direction, wobble=0.4):
        dx, dy, dz = direction
        dx += np.random.uniform(-wobble, wobble)
        dy += np.random.uniform(-wobble, wobble)
        dz += np.random.uniform(-wobble*0.3, wobble*0.3)
        mag = np.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            dx, dy, dz = dx/mag*bond_len, dy/mag*bond_len, dz/mag*bond_len
        new_x, new_y, new_z = x + dx, y + dy, z + dz
        if abs(new_x) > box_x/2: dx, new_x = -dx, x + (-dx)
        if abs(new_y) > box_y/2: dy, new_y = -dy, y + (-dy)
        if abs(new_z) > box_z/2: dz, new_z = -dz, z + (-dz)
        return new_x, new_y, new_z, (dx/bond_len if bond_len > 0 else 0, dy/bond_len if bond_len > 0 else 0, dz/bond_len if bond_len > 0 else 0)

    def generate_peg_segment(x, y, z, direction, n_units=6):
        nonlocal res_id
        for _ in range(n_units):
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "PEG")
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "PEG")
            x, y, z, direction = step(x, y, z, 1.43, direction)
            add_atom("O", x, y, z, "PEG")
            res_id += 1
        return x, y, z, direction

    def generate_pdms_segment(x, y, z, direction, n_units=5):
        nonlocal res_id
        for _ in range(n_units):
            x, y, z, direction = step(x, y, z, 1.64, direction)
            add_atom("SI", x, y, z, "PDM")
            add_atom("C", x + np.random.uniform(0.6, 1.0), y + np.random.uniform(-0.8, 0.8), z, "PDM")
            add_atom("C", x + np.random.uniform(-1.0, -0.6), y + np.random.uniform(-0.8, 0.8), z, "PDM")
            x, y, z, direction = step(x, y, z, 1.64, direction)
            add_atom("O", x, y, z, "PDM")
            res_id += 1
        return x, y, z, direction

    def generate_urethane_hard(x, y, z, direction):
        nonlocal res_id
        x, y, z, direction = step(x, y, z, 1.47, direction)
        add_atom("N", x, y, z, "URE")
        x, y, z, direction = step(x, y, z, 1.33, direction)
        add_atom("C", x, y, z, "URE")
        add_atom("O", x + np.random.uniform(-0.4, 0.4), y + 0.9, z, "URE")
        x, y, z, direction = step(x, y, z, 1.43, direction)
        add_atom("O", x, y, z, "URE")
        for _ in range(3):
            x, y, z, direction = step(x, y, z, 1.54, direction)
            add_atom("C", x, y, z, "URE")
        res_id += 1
        return x, y, z, direction

    sparsa_total, carbosil_total = sparsa_frac, carbosil_frac
    total = sparsa_total + carbosil_total
    if total == 0:
        total, sparsa_total, carbosil_total = 1, 0.5, 0.5

    n_chains, grid_nx, grid_ny, grid_nz = 60, 5, 5, 3
    chain_count = 0
    for gx in range(grid_nx):
        for gy in range(grid_ny):
            for gz in range(grid_nz):
                if chain_count >= n_chains:
                    break
                x = -box_x/2 + (gx + 0.5) * box_x/grid_nx + np.random.uniform(-2, 2)
                y = -box_y/2 + (gy + 0.5) * box_y/grid_ny + np.random.uniform(-2, 2)
                z = -box_z/2 + (gz + 0.5) * box_z/grid_nz + np.random.uniform(-1, 1)
                direction = random_direction()
                is_carbosil = np.random.random() < (carbosil_total / total)
                for _ in range(np.random.randint(3, 6)):
                    if is_carbosil:
                        x, y, z, direction = generate_pdms_segment(x, y, z, direction, n_units=np.random.randint(4, 7))
                    else:
                        x, y, z, direction = generate_peg_segment(x, y, z, direction, n_units=np.random.randint(5, 8))
                    x, y, z, direction = generate_urethane_hard(x, y, z, direction)
                    if np.random.random() < 0.3:
                        direction = random_direction()
                chain_count += 1
    return atoms

def render_tpu_3dmol(atoms, carbosil_frac, style, molecule_info=None):
    pdb_lines = []
    for atom in atoms:
        line = f"ATOM  {atom['id']:5d} {atom['name']:4s} {atom['res_name']:3s}  {atom['res_id']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {atom['element']:>2s}"
        pdb_lines.append(line)
    pdb_lines.append("END")
    pdb_data = "\n".join(pdb_lines)
    pdb_escaped = pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    if style == "stick":
        color_scheme = """
        viewer.setStyle({resn: 'PEG'}, {stick: {radius: 0.12, colorscheme: 'greenCarbon'}});
        viewer.setStyle({resn: 'PDM'}, {stick: {radius: 0.14, colorscheme: 'blueCarbon'}});
        viewer.setStyle({resn: 'URE'}, {stick: {radius: 0.14, colorscheme: 'orangeCarbon'}});
        viewer.setStyle({elem: 'N'}, {stick: {radius: 0.14}, sphere: {scale: 0.25, color: '0x3498db'}});
        viewer.setStyle({elem: 'SI'}, {stick: {radius: 0.16}, sphere: {scale: 0.3, color: '0xf1c40f'}});
        """
    elif style == "sphere":
        color_scheme = """
        viewer.setStyle({resn: 'PEG'}, {sphere: {scale: 0.3, colorscheme: 'greenCarbon'}});
        viewer.setStyle({resn: 'PDM'}, {sphere: {scale: 0.32, colorscheme: 'blueCarbon'}});
        viewer.setStyle({resn: 'URE'}, {sphere: {scale: 0.32, colorscheme: 'orangeCarbon'}});
        viewer.setStyle({elem: 'SI'}, {sphere: {scale: 0.4, color: '0xf1c40f'}});
        """
    else:
        color_scheme = """
        viewer.setStyle({resn: 'PEG'}, {line: {linewidth: 2.5, color: '0x27ae60'}});
        viewer.setStyle({resn: 'PDM'}, {line: {linewidth: 3, color: '0x2980b9'}});
        viewer.setStyle({resn: 'URE'}, {line: {linewidth: 3, color: '0xe74c3c'}});
        """

    mol_colors = {"phenol": "0xe74c3c", "m-cresol": "0x9b59b6", "glucose": "0xf39c12", "oxygen": "0x3498db"}
    mol_sphere_js = ""
    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "0x1abc9c")
        mol_sphere_js = f"viewer.addSphere({{center: {{x: 0, y: 0, z: 0}}, radius: 2.5, color: {color}, opacity: 0.95}});"

    box_x, box_y, box_z = 40, 40, 15
    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer_tpu" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer_tpu", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");
        {color_scheme}
        var hx = {box_x}/2, hy = {box_y}/2, hz = {box_z}/2;
        var boxColor = 0x555555; var boxWidth = 1.5;
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: -hy, z: -hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: -hy, z: hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: -hy, z: -hz}}, end: {{x: -hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: -hy, z: -hz}}, end: {{x: hx, y: -hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: hx, y: hy, z: -hz}}, end: {{x: hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        viewer.addLine({{start: {{x: -hx, y: hy, z: -hz}}, end: {{x: -hx, y: hy, z: hz}}, color: boxColor, linewidth: boxWidth}});
        {mol_sphere_js}
        viewer.zoomTo(); viewer.zoom(0.5);
        viewer.rotate(20, {{x: 1, y: 0, z: 0}}); viewer.rotate(-15, {{x: 0, y: 1, z: 0}});
        viewer.render();
    </script>
    """
    components.html(html, height=520)

# ============== MD SIMULATION HELPER FUNCTIONS ==============

def generate_md_structure(composition, box_size=50, n_chains=20, membrane_type="tpu", seed=42):
    np.random.seed(seed)
    atoms, bonds = [], []
    atom_id, res_id = 1, 1

    if membrane_type == "tpu":
        total = sum(composition.values())
        if total == 0: total = 1
        s1 = composition.get('Sparsa_27G26', 0) / total
        s2 = composition.get('Sparsa_30G25', 0) / total
        c1 = composition.get('Carbosil_2080A', 0) / total
        c2 = composition.get('Carbosil_2090A', 0) / total

    bx = box_size

    def add_atom(element, x, y, z, res_name, charge=0.0):
        nonlocal atom_id
        masses = {'C': 12.011, 'O': 15.999, 'N': 14.007, 'H': 1.008, 'Si': 28.086, 'P': 30.974}
        atoms.append({'id': atom_id, 'name': element, 'element': element, 'res_name': res_name, 'res_id': res_id,
                      'x': x, 'y': y, 'z': z, 'charge': charge, 'mass': masses.get(element, 12.0)})
        atom_id += 1
        return atom_id - 1

    def random_dir():
        theta, phi = np.random.uniform(0, 2*np.pi), np.random.uniform(np.pi/4, 3*np.pi/4)
        return (np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi))

    def step(x, y, z, bond_len, direction, wobble=0.3):
        dx, dy, dz = direction
        dx += np.random.uniform(-wobble, wobble)
        dy += np.random.uniform(-wobble, wobble)
        dz += np.random.uniform(-wobble, wobble)
        mag = np.sqrt(dx*dx + dy*dy + dz*dz)
        if mag > 0:
            dx, dy, dz = dx/mag*bond_len, dy/mag*bond_len, dz/mag*bond_len
        return x + dx, y + dy, z + dz, (dx/bond_len, dy/bond_len, dz/bond_len)

    if membrane_type == "tpu":
        for i in range(n_chains):
            x = np.random.uniform(-bx/2 + 5, bx/2 - 5)
            y = np.random.uniform(-bx/2 + 5, bx/2 - 5)
            z = np.random.uniform(-bx/2 + 5, bx/2 - 5)
            direction = random_dir()
            r = np.random.random()
            if r < s1: chain_type = 'sparsa1'
            elif r < s1 + s2: chain_type = 'sparsa2'
            elif r < s1 + s2 + c1: chain_type = 'carbosil1'
            else: chain_type = 'carbosil2'
            prev_atom = None
            for _ in range(np.random.randint(3, 6)):
                for _ in range(np.random.randint(4, 8)):
                    x, y, z, direction = step(x, y, z, 1.54, direction)
                    if chain_type in ('carbosil1', 'carbosil2'):
                        a = add_atom("Si", x, y, z, "PDM", 0.4)
                    else:
                        a = add_atom("C", x, y, z, "PEG")
                    if prev_atom: bonds.append((prev_atom, a))
                    prev_atom = a
                    x, y, z, direction = step(x, y, z, 1.43, direction)
                    a = add_atom("O", x, y, z, "PEG" if chain_type.startswith('sparsa') else "PDM", -0.4)
                    bonds.append((prev_atom, a))
                    prev_atom = a
                x, y, z, direction = step(x, y, z, 1.47, direction)
                a = add_atom("N", x, y, z, "URE", -0.47)
                if prev_atom: bonds.append((prev_atom, a))
                prev_atom = a
                x, y, z, direction = step(x, y, z, 1.33, direction)
                a = add_atom("C", x, y, z, "URE", 0.51)
                bonds.append((prev_atom, a))
                prev_atom = a
                x, y, z, direction = step(x, y, z, 1.43, direction)
                a = add_atom("O", x, y, z, "URE", -0.33)
                bonds.append((prev_atom, a))
                prev_atom = a
                res_id += 1
                if np.random.random() < 0.3: direction = random_dir()
    else:
        for lip_name, count in composition.items():
            for _ in range(count):
                x = np.random.uniform(-bx/2 + 3, bx/2 - 3)
                y = np.random.uniform(-bx/2 + 3, bx/2 - 3)
                z = np.random.uniform(-5, 5)
                add_atom("P", x, y, z + 10, lip_name[:3], -1.0)
                add_atom("N", x + 1, y, z + 12, lip_name[:3], 1.0)
                prev = None
                for t in range(8):
                    a = add_atom("C", x, y + t*0.5, z + 8 - t*2, lip_name[:3])
                    if prev: bonds.append((prev, a))
                    prev = a
                res_id += 1

    return atoms, bonds, box_size

def write_pdb_string(atoms, box_size):
    lines = [f"CRYST1{box_size:9.3f}{box_size:9.3f}{box_size:9.3f}  90.00  90.00  90.00 P 1           1"]
    for atom in atoms:
        lines.append(f"ATOM  {atom['id']:5d} {atom['name']:4s} {atom['res_name']:3s}  {atom['res_id']:4d}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {atom['element']:>2s}")
    lines.append("END")
    return "\n".join(lines)

def write_gro_string(atoms, box_size):
    lines = ["Membrane System", str(len(atoms))]
    for atom in atoms:
        x_nm, y_nm, z_nm = atom['x']/10, atom['y']/10, atom['z']/10
        lines.append(f"{atom['res_id']:5d}{atom['res_name']:5s}{atom['name']:>5s}{atom['id']:5d}{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}")
    bx = box_size / 10
    lines.append(f"{bx:10.5f}{bx:10.5f}{bx:10.5f}")
    return "\n".join(lines)

def write_mdp_minim():
    return """; Energy Minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
vdwtype     = Cut-off
rvdw        = 1.2
pbc         = xyz
"""

def write_mdp_nvt(temperature=310):
    return f"""; NVT Equilibration
integrator  = md
nsteps      = 50000
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500
nstxout-compressed = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
pcoupl      = no
pbc         = xyz
gen_vel     = yes
gen_temp    = {temperature}
gen_seed    = -1
constraints = h-bonds
constraint_algorithm = lincs
"""

def write_mdp_npt(temperature=310, pressure=1.0):
    return f"""; NPT Equilibration
integrator  = md
nsteps      = 50000
dt          = 0.002
nstxout     = 5000
nstvout     = 5000
nstenergy   = 500
nstlog      = 500
nstxout-compressed = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = {pressure}
compressibility = 4.5e-5
refcoord_scaling = com
pbc         = xyz
gen_vel     = no
constraints = h-bonds
constraint_algorithm = lincs
"""

def write_mdp_production(temperature=310, nsteps=5000000):
    return f"""; Production MD
integrator  = md
nsteps      = {nsteps}
dt          = 0.002
nstxout     = 0
nstvout     = 0
nstenergy   = 5000
nstlog      = 5000
nstxout-compressed = 5000
nstlist     = 10
cutoff-scheme = Verlet
ns_type     = grid
rlist       = 1.2
coulombtype = PME
rcoulomb    = 1.2
pme_order   = 4
fourierspacing = 0.16
vdwtype     = Cut-off
rvdw        = 1.2
DispCorr    = EnerPres
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5
refcoord_scaling = com
pbc         = xyz
gen_vel     = no
constraints = h-bonds
constraint_algorithm = lincs
"""

def write_run_script(structure_name):
    return f"""#!/bin/bash
# GROMACS simulation workflow

STRUCTURE="{structure_name}"

echo "=== Membrane MD Simulation ==="

# Step 1: Generate topology
echo "Step 1: Generating topology..."
gmx pdb2gmx -f $STRUCTURE -o processed.gro -water none -ff oplsaa << EOF
1
EOF

# Step 2: Define box
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
"""

def create_md_zip(composition, box_size, n_chains, temperature, production_ns, membrane_type="tpu"):
    atoms, bonds, box = generate_md_structure(composition, box_size, n_chains, membrane_type)
    production_steps = int(production_ns * 1000 / 0.002)

    if membrane_type == "tpu":
        comp_str = "_".join([f"{k}{int(v*100)}" for k, v in composition.items() if v > 0])
    else:
        comp_str = "_".join([f"{k}{v}" for k, v in composition.items() if v > 0])

    structure_name = f"membrane_{comp_str}.pdb"

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(structure_name, write_pdb_string(atoms, box))
        zf.writestr(f"membrane_{comp_str}.gro", write_gro_string(atoms, box))
        zf.writestr("minim.mdp", write_mdp_minim())
        zf.writestr("nvt.mdp", write_mdp_nvt(temperature))
        zf.writestr("npt.mdp", write_mdp_npt(temperature))
        zf.writestr("md.mdp", write_mdp_production(temperature, production_steps))
        zf.writestr("run_simulation.sh", write_run_script(structure_name))

        readme = f"""Membrane MD Simulation Files
============================

Membrane Type: {membrane_type.upper()}
Composition: {composition}

Parameters:
- Box size: {box_size} Angstroms
- Number of chains/lipids: {n_chains}
- Temperature: {temperature} K
- Production run: {production_ns} ns

Files:
- {structure_name}: Initial structure (PDB format)
- membrane_{comp_str}.gro: Initial structure (GROMACS format)
- minim.mdp: Energy minimization parameters
- nvt.mdp: NVT equilibration parameters
- npt.mdp: NPT equilibration parameters
- md.mdp: Production MD parameters
- run_simulation.sh: Bash script to run full workflow

To run:
1. Make sure GROMACS is installed and in your PATH
2. chmod +x run_simulation.sh
3. ./run_simulation.sh
"""
        zf.writestr("README.txt", readme)

    zip_buffer.seek(0)
    return zip_buffer, len(atoms), len(bonds)


# ============== TAB 1: LIPID MEMBRANES ==============

with tab_lipid:
    st.title("Lipid Membrane Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Membrane Composition")
        st.markdown("**Lipids (Top / Bottom)**")

        lipid_names = ["POPC", "POPE", "POPS", "CHOL", "POPG"]
        lipid_values = {}

        for lip in lipid_names:
            c1, c2 = st.columns(2)
            with c1:
                top = st.number_input(f"{lip} Top", value=64 if lip == "POPC" else 0, min_value=0, key=f"{lip}_top")
            with c2:
                bottom = st.number_input(f"{lip} Bottom", value=64 if lip == "POPC" else 0, min_value=0, key=f"{lip}_bottom")
            lipid_values[lip] = (top, bottom)

        suggested_x, suggested_y = calculate_box_dimensions(lipid_values)
        total_counts = {lip: v[0] + v[1] for lip, v in lipid_values.items()}
        total_lipids = sum(total_counts.values())
        chol_count = total_counts.get("CHOL", 0)

        if total_lipids > 0:
            effective_apl = calculate_effective_apl(total_counts, chol_count, total_lipids)
            chol_pct = (chol_count / total_lipids) * 100
            st.divider()
            st.markdown("**Box Dimensions**")
            auto_calc = st.checkbox("Auto-calculate from Area per Lipid", value=True)
            if auto_calc:
                box_x, box_y = suggested_x, suggested_y
                st.caption(f"Suggested: {suggested_x} x {suggested_y} A")
                st.caption(f"Effective Area per Lipid: {effective_apl:.1f} A^2")
                if chol_pct > 5:
                    st.caption(f"Cholesterol: {chol_pct:.0f}% (condensation applied)")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    box_x = st.number_input("Box X (A)", value=suggested_x, min_value=40, max_value=200)
                with c2:
                    box_y = st.number_input("Box Y (A)", value=suggested_y, min_value=40, max_value=200)
        else:
            st.divider()
            st.markdown("**Box Dimensions**")
            c1, c2 = st.columns(2)
            with c1:
                box_x = st.number_input("Box X (A)", value=80, min_value=40, max_value=200)
            with c2:
                box_y = st.number_input("Box Y (A)", value=80, min_value=40, max_value=200)

        if st.button("Build Membrane", type="primary", use_container_width=True, key="build_lipid"):
            lipids = {k: v for k, v in lipid_values.items() if v[0] > 0 or v[1] > 0}
            if not lipids:
                st.error("Add at least one lipid")
            else:
                with st.spinner("Building membrane..."):
                    try:
                        config = MembraneConfig.create_simple(lipids=lipids, box_size=(float(box_x), float(box_y), 120.0))
                        builder = MembraneBuilder(seed=12345)
                        membrane = builder.build(config, use_templates=True, templates_dir="lipids")
                        st.session_state.pdb_data = membrane.to_pdb_string()
                        st.session_state.membrane_props = {
                            'thickness': round(membrane.properties.thickness, 1),
                            'area_per_lipid': round(membrane.properties.area_per_lipid, 1),
                            'bending_modulus': round(membrane.properties.bending_modulus, 2),
                            'total_lipids': membrane.properties.total_lipids,
                            'n_atoms': membrane.n_atoms
                        }
                        st.session_state.perm_result = None
                        st.success("Membrane built!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.pdb_data:
            st.download_button("Download PDB", st.session_state.pdb_data, file_name="membrane.pdb",
                              mime="text/plain", use_container_width=True)

        st.divider()
        st.subheader("Permeability Calculator")

        mol_presets = {
            "Water": {"mw": 18, "asa": 40, "hbd": 2, "hba": 1, "charge": 0, "pka": None},
            "Ethanol": {"mw": 46, "asa": 80, "hbd": 1, "hba": 1, "charge": 0, "pka": None},
            "Glucose": {"mw": 180, "asa": 180, "hbd": 5, "hba": 6, "charge": 0, "pka": None},
            "Caffeine": {"mw": 194, "asa": 150, "hbd": 0, "hba": 3, "charge": 0, "pka": None},
            "Aspirin": {"mw": 180, "asa": 140, "hbd": 1, "hba": 4, "charge": -1, "pka": 3.5},
        }

        selected_mol = st.selectbox("Preset Molecule", ["Custom"] + list(mol_presets.keys()))

        if selected_mol != "Custom":
            preset = mol_presets[selected_mol]
            mol_name, mol_mw, mol_asa = selected_mol.lower(), preset["mw"], preset["asa"]
            mol_hbd, mol_hba, mol_charge, mol_pka = preset["hbd"], preset["hba"], preset["charge"], preset["pka"]
        else:
            mol_name, mol_mw, mol_asa, mol_hbd, mol_hba, mol_charge, mol_pka = "custom", 100, 100, 0, 0, 0, None

        c1, c2 = st.columns(2)
        with c1:
            mol_mw = st.number_input("Molecular Weight", value=mol_mw, min_value=1)
            mol_hbd = st.number_input("H-Bond Donors", value=mol_hbd, min_value=0)
            mol_charge = st.number_input("Charge", value=float(mol_charge), step=0.1)
        with c2:
            mol_asa = st.number_input("Surface Area (A^2)", value=mol_asa, min_value=1)
            mol_hba = st.number_input("H-Bond Acceptors", value=mol_hba, min_value=0)
            pka_input = st.text_input("pKa", value=str(mol_pka) if mol_pka else "")

        if st.button("Calculate Permeability", type="primary", use_container_width=True, key="calc_lipid"):
            if not st.session_state.pdb_data:
                st.error("Build a membrane first")
            else:
                with st.spinner("Calculating..."):
                    try:
                        pka_val = float(pka_input) if pka_input else None
                        mol = MoleculeDescriptor.simple(name=mol_name, molecular_weight=float(mol_mw),
                            total_asa=float(mol_asa), n_hbd=int(mol_hbd), n_hba=int(mol_hba),
                            charge=float(mol_charge), pka=pka_val)
                        composition = {lip: top + bottom for lip, (top, bottom) in lipid_values.items() if top + bottom > 0}
                        if not composition: composition = {"POPC": 128}
                        predictor = PermeabilityPredictor(composition=composition)
                        result = predictor.calculate(mol, "BLM")
                        classification = "high" if result.log_p > -6 else "moderate" if result.log_p > -8 else "low"
                        st.session_state.perm_result = {
                            'log_p': round(result.log_p, 2), 'permeability': f"{result.permeability_cm_s:.2e}",
                            'binding_energy': round(result.membrane_bound_energy, 1),
                            'binding_position': round(result.binding_position, 1),
                            'classification': classification, 'mol_name': mol_name
                        }
                        st.success("Calculated!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.subheader("3D Viewer")
        if st.session_state.pdb_data:
            style = st.radio("Style", ["stick", "line", "sphere"], horizontal=True, key="lipid_style")

            mol_info = {'name': st.session_state.perm_result['mol_name'], 'z': st.session_state.perm_result['binding_position']} if st.session_state.perm_result else None
            render_lipid_3dmol(st.session_state.pdb_data, style, mol_info)

            if st.session_state.membrane_props:
                st.markdown("**Membrane Properties**")
                props = st.session_state.membrane_props
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Lipids", props['total_lipids'])
                c2.metric("Atoms", f"{props['n_atoms']:,}")
                c3.metric("Thickness", f"{props['thickness']} A")
                c4.metric("Area/Lipid", f"{props['area_per_lipid']} A^2")
                c5.metric("Bending Mod.", f"{props['bending_modulus']} kT")

            if st.session_state.perm_result:
                st.divider()
                st.markdown("**Permeability Results**")
                res = st.session_state.perm_result
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("log P", res['log_p'])
                c2.metric("P (cm/s)", res['permeability'])
                c3.metric("Binding", f"{res['binding_energy']} kJ/mol")
                class_colors = {"high": "green", "moderate": "orange", "low": "red"}
                c4.markdown(f"**Classification**")
                c4.markdown(f":{class_colors[res['classification']]}[{res['classification'].upper()}]")
        else:
            st.info("Build a membrane to see the 3D structure")


# ============== TAB 2: TPU MEMBRANES ==============

with tab_tpu:
    st.title("TPU Membrane Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Membrane Composition")
        thickness = st.number_input("Thickness (um)", value=200, min_value=10, max_value=500, step=10)

        st.markdown("**Sparsa Polymers (%)**")
        c1, c2 = st.columns(2)
        with c1:
            sparsa1_pct = st.number_input("Sparsa 1 (27G26)", value=30, min_value=0, max_value=100, key="tpu_sparsa1")
        with c2:
            sparsa2_pct = st.number_input("Sparsa 2 (30G25)", value=0, min_value=0, max_value=100, key="tpu_sparsa2")

        st.markdown("**Carbosil Polymers (%)**")
        c3, c4 = st.columns(2)
        with c3:
            carbosil1_pct = st.number_input("Carbosil 1 (2080A)", value=70, min_value=0, max_value=100, key="tpu_carbosil1")
        with c4:
            carbosil2_pct = st.number_input("Carbosil 2 (2090A)", value=0, min_value=0, max_value=100, key="tpu_carbosil2")

        total = sparsa1_pct + sparsa2_pct + carbosil1_pct + carbosil2_pct
        if total > 0:
            sparsa1_frac, sparsa2_frac = sparsa1_pct / total, sparsa2_pct / total
            carbosil1_frac, carbosil2_frac = carbosil1_pct / total, carbosil2_pct / total
        else:
            sparsa1_frac = sparsa2_frac = carbosil1_frac = carbosil2_frac = 0.25

        carbosil_frac = carbosil1_frac + carbosil2_frac
        sparsa_frac = sparsa1_frac + sparsa2_frac
        st.caption(f"Total Sparsa: {sparsa_frac*100:.0f}% | Total Carbosil: {carbosil_frac*100:.0f}%")

        if st.button("Build Membrane", type="primary", use_container_width=True, key="build_tpu"):
            with st.spinner("Building membrane..."):
                try:
                    config = TPUMembraneConfig(
                        polymers={"Sparsa_27G26": sparsa1_frac, "Sparsa_30G25": sparsa2_frac,
                                  "Carbosil_2080A": carbosil1_frac, "Carbosil_2090A": carbosil2_frac},
                        thickness=float(thickness)
                    )
                    builder = TPUMembraneBuilder(seed=12345)
                    st.session_state.tpu_membrane = builder.build(config)
                    st.session_state.tpu_perm_result = None
                    st.success("Membrane built!")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.tpu_membrane:
            report = ["TPU Membrane Report", "=" * 40,
                     f"Sparsa 1 (27G26): {sparsa1_frac*100:.1f}%", f"Sparsa 2 (30G25): {sparsa2_frac*100:.1f}%",
                     f"Carbosil 1 (2080A): {carbosil1_frac*100:.1f}%", f"Carbosil 2 (2090A): {carbosil2_frac*100:.1f}%",
                     f"Thickness: {thickness} um"]
            props = st.session_state.tpu_membrane.properties
            report.extend([f"Density: {props.density:.3f} g/cm3", f"Water uptake: {props.water_uptake:.1f}%"])
            st.download_button("Download Report", "\n".join(report), file_name="tpu_membrane_report.txt",
                              mime="text/plain", use_container_width=True)

        st.divider()
        st.subheader("Permeability Calculator")
        tpu_mol_presets = {"Phenol": "phenol", "m-Cresol": "m-cresol", "Glucose": "glucose", "Oxygen": "oxygen"}
        selected_tpu_mol = st.selectbox("Molecule", list(tpu_mol_presets.keys()), key="tpu_mol")

        if st.button("Calculate Permeability", type="primary", use_container_width=True, key="calc_tpu"):
            if st.session_state.tpu_membrane is None:
                st.error("Build a membrane first")
            else:
                with st.spinner("Calculating..."):
                    try:
                        predictor = TPUPermeabilityPredictor(
                            composition=st.session_state.tpu_membrane.composition,
                            thickness_um=st.session_state.tpu_membrane.thickness
                        )
                        mol_name = tpu_mol_presets[selected_tpu_mol]
                        result = predictor.calculate_preset(mol_name)
                        st.session_state.tpu_perm_result = {'permeability': f"{result.permeability_cm_s:.2e}", 'mol_name': mol_name}
                        st.success("Calculated!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.subheader("3D Viewer")
        if st.session_state.tpu_membrane:
            membrane = st.session_state.tpu_membrane
            props = membrane.properties

            tpu_style = st.radio("Style", ["stick", "sphere", "line"], horizontal=True, key="tpu_style")

            mol_info = {'name': st.session_state.tpu_perm_result['mol_name']} if st.session_state.tpu_perm_result else None

            comp = membrane.composition
            sparsa_frac = comp.get("Sparsa_27G26", 0) + comp.get("Sparsa_30G25", 0)
            carbosil_frac = comp.get("Carbosil_2080A", 0) + comp.get("Carbosil_2090A", 0)

            atoms = generate_tpu_polymer_chains(membrane, sparsa_frac, carbosil_frac)
            render_tpu_3dmol(atoms, carbosil_frac, tpu_style, mol_info)

            st.markdown("**Membrane Properties**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Thickness", f"{props.thickness_um} um")
            c2.metric("Density", f"{props.density:.2f} g/cm3")
            c3.metric("Water Uptake", f"{props.water_uptake:.1f}%")
            c4.metric("Free Volume", f"{props.free_volume_fraction:.3f}")
            c5.metric("Soft Seg.", f"{props.soft_segment_fraction*100:.0f}%")

            if st.session_state.tpu_perm_result:
                st.divider()
                res = st.session_state.tpu_perm_result
                st.metric("Permeability (cm/s)", res['permeability'])
        else:
            st.info("Build a membrane to see the 3D structure")


# ============== TAB 3: MD SIMULATION ==============

with tab_md:
    st.title("MD Simulation Setup")
    st.markdown("Generate GROMACS input files for molecular dynamics simulation.")

    md_type = st.radio("Membrane Type", ["TPU", "Lipid"], horizontal=True, key="md_type")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Composition")
        if md_type == "TPU":
            md_s1 = st.slider("Sparsa 1 (27G26) %", 0, 100, 30, key="md_s1")
            md_s2 = st.slider("Sparsa 2 (30G25) %", 0, 100, 0, key="md_s2")
            md_c1 = st.slider("Carbosil 1 (2080A) %", 0, 100, 70, key="md_c1")
            md_c2 = st.slider("Carbosil 2 (2090A) %", 0, 100, 0, key="md_c2")
            md_total = md_s1 + md_s2 + md_c1 + md_c2
            if md_total > 0:
                st.caption(f"Normalized: S1={md_s1/md_total*100:.0f}%, S2={md_s2/md_total*100:.0f}%, C1={md_c1/md_total*100:.0f}%, C2={md_c2/md_total*100:.0f}%")
        else:
            md_popc = st.slider("POPC", 0, 200, 64, key="md_popc")
            md_pope = st.slider("POPE", 0, 200, 0, key="md_pope")
            md_chol = st.slider("CHOL", 0, 200, 0, key="md_chol")
            md_pops = st.slider("POPS", 0, 200, 0, key="md_pops")
            st.caption(f"Total lipids: {md_popc + md_pope + md_chol + md_pops}")

    with col2:
        st.subheader("Simulation Parameters")
        md_box_size = st.number_input("Box Size (Angstroms)", value=50, min_value=30, max_value=100, step=10, key="md_box")
        md_n_chains = st.number_input("Number of Chains/Lipids", value=20, min_value=5, max_value=100, step=5, key="md_chains")
        md_temp = st.number_input("Temperature (K)", value=310, min_value=270, max_value=400, step=10, key="md_temp")
        md_ns = st.number_input("Production Run (ns)", value=10.0, min_value=1.0, max_value=100.0, step=1.0, key="md_ns")

    st.divider()

    if st.button("Generate MD Files", type="primary", use_container_width=True, key="gen_md"):
        with st.spinner("Generating simulation files..."):
            if md_type == "TPU":
                md_total = md_s1 + md_s2 + md_c1 + md_c2
                if md_total == 0: md_total = 1
                composition = {'Sparsa_27G26': md_s1/md_total, 'Sparsa_30G25': md_s2/md_total,
                              'Carbosil_2080A': md_c1/md_total, 'Carbosil_2090A': md_c2/md_total}
                membrane_type = "tpu"
            else:
                composition = {'POPC': md_popc, 'POPE': md_pope, 'CHOL': md_chol, 'POPS': md_pops}
                composition = {k: v for k, v in composition.items() if v > 0}
                membrane_type = "lipid"

            zip_buffer, n_atoms, n_bonds = create_md_zip(composition, md_box_size, md_n_chains, md_temp, md_ns, membrane_type)
            st.session_state.md_files = zip_buffer
            st.success(f"Generated {n_atoms} atoms and {n_bonds} bonds!")

    if st.session_state.md_files:
        st.download_button("Download MD Simulation Package (.zip)", st.session_state.md_files,
                          file_name="md_simulation.zip", mime="application/zip", use_container_width=True)

        st.divider()
        st.subheader("How to Run")
        st.markdown("""
        1. **Extract** the ZIP file to a directory
        2. **Make sure GROMACS is installed** and available in your PATH
        3. **Run the simulation**:
        ```bash
        cd md_simulation
        chmod +x run_simulation.sh
        ./run_simulation.sh
        ```

        Or run steps individually:
        ```bash
        # Generate topology
        gmx pdb2gmx -f membrane_*.pdb -o processed.gro -water none -ff oplsaa

        # Define box
        gmx editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic

        # Energy minimization
        gmx grompp -f minim.mdp -c box.gro -p topol.top -o em.tpr
        gmx mdrun -v -deffnm em

        # NVT equilibration
        gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
        gmx mdrun -v -deffnm nvt

        # NPT equilibration
        gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr
        gmx mdrun -v -deffnm npt

        # Production MD
        gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
        gmx mdrun -v -deffnm md
        ```
        """)
