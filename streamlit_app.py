import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from membrane_builder import MembraneBuilder, MembraneConfig, MoleculeDescriptor, PermeabilityPredictor

# Page config
st.set_page_config(
    page_title="Membrane Builder",
    layout="wide"
)

st.title("Membrane Builder")

# Standard area per lipid values (Angstrom^2)
# From experimental data and MD simulations
STANDARD_APL = {
    "POPC": 68.3,   # Kucerka et al. 2011
    "DOPC": 72.4,   # Kucerka et al. 2008
    "DPPC": 63.0,   # Nagle & Tristram-Nagle 2000
    "POPE": 58.8,   # Kucerka et al. 2012
    "POPG": 66.0,   # Pan et al. 2014
    "POPS": 65.0,   # Petrache et al. 2004
    "CHOL": 40.0,   # Hung et al. 2007
    "PSM": 52.0,    # Venable et al. 2014
}

def calculate_effective_apl(lipid_counts, chol_count, total_lipids):
    """
    Calculate effective APL accounting for cholesterol condensation effect.
    Cholesterol reduces the APL of neighboring phospholipids.
    """
    if total_lipids == 0:
        return 65.0  # Default

    # Calculate cholesterol mole fraction
    chol_fraction = chol_count / total_lipids if total_lipids > 0 else 0

    # Condensation factor: cholesterol reduces APL of phospholipids
    # Based on Hung et al. 2007 and Alwarawrah et al. 2010
    # At ~40% cholesterol, APL can decrease by ~15-20%
    condensation_factor = 1.0 - (0.4 * chol_fraction)  # Linear approximation
    condensation_factor = max(0.75, condensation_factor)  # Cap at 25% reduction

    # Calculate weighted average APL
    total_area = 0
    for lip, count in lipid_counts.items():
        if lip == "CHOL":
            # Cholesterol has its own small APL
            total_area += count * STANDARD_APL.get("CHOL", 40.0)
        else:
            # Phospholipids get condensed by cholesterol
            base_apl = STANDARD_APL.get(lip, 65.0)
            effective_apl = base_apl * condensation_factor
            total_area += count * effective_apl

    return total_area / total_lipids if total_lipids > 0 else 65.0


def calculate_box_dimensions(lipid_values):
    """Calculate optimal box dimensions based on lipid composition."""
    # Count lipids per leaflet
    top_count = sum(v[0] for v in lipid_values.values())
    bottom_count = sum(v[1] for v in lipid_values.values())

    # Use the leaflet with more lipids for sizing
    max_leaflet = max(top_count, bottom_count)

    if max_leaflet == 0:
        return 80, 80

    # Get total counts for APL calculation
    total_counts = {lip: v[0] + v[1] for lip, v in lipid_values.items()}
    total_lipids = sum(total_counts.values())
    chol_count = total_counts.get("CHOL", 0)

    # Calculate effective APL with cholesterol condensation
    effective_apl = calculate_effective_apl(total_counts, chol_count, total_lipids)

    # Calculate required area for one leaflet
    required_area = max_leaflet * effective_apl

    # Calculate box side (square box)
    box_side = np.sqrt(required_area)

    # Round up to nearest 5 Angstroms and add small buffer
    box_side = int(np.ceil(box_side / 5) * 5) + 5

    # Clamp to reasonable range
    box_side = max(40, min(200, box_side))

    return box_side, box_side


# Initialize session state
if 'pdb_data' not in st.session_state:
    st.session_state.pdb_data = None
if 'membrane_props' not in st.session_state:
    st.session_state.membrane_props = None
if 'perm_result' not in st.session_state:
    st.session_state.perm_result = None

def render_3dmol(pdb_data, style="stick", molecule_info=None, animate=False):
    """Render 3D structure using py3Dmol via HTML component"""

    mol_colors = {
        "water": "0x3498db",
        "ethanol": "0x9b59b6",
        "caffeine": "0x8B4513",
        "aspirin": "0xe74c3c",
        "glucose": "0xf39c12",
        "custom": "0x1abc9c"
    }

    mol_sphere_js = ""
    animation_js = ""

    if molecule_info:
        color = mol_colors.get(molecule_info['name'], "0x1abc9c")
        z_pos = molecule_info['z']

        if animate:
            # Animation code - molecule passes through membrane
            animation_js = f"""
            var sphereId = null;
            var startZ = 40;
            var endZ = -40;
            var bindingZ = {z_pos};
            var duration = 4000;
            var startTime = Date.now();
            var color = {color};

            function animatePermeation() {{
                var elapsed = Date.now() - startTime;
                var progress = elapsed / duration;

                if (progress >= 1) {{
                    // Reset to binding position
                    if (sphereId !== null) viewer.removeShape(sphereId);
                    sphereId = viewer.addSphere({{
                        center: {{x: 0, y: 0, z: bindingZ}},
                        radius: 3,
                        color: color,
                        opacity: 0.9
                    }});
                    viewer.render();
                    return;
                }}

                // Calculate z position with slowdown at membrane
                var z;
                if (progress < 0.3) {{
                    z = startZ - (startZ - bindingZ) * (progress / 0.3);
                }} else if (progress < 0.7) {{
                    var membraneProgress = (progress - 0.3) / 0.4;
                    z = bindingZ - (bindingZ * 2) * membraneProgress;
                }} else {{
                    var exitProgress = (progress - 0.7) / 0.3;
                    z = -bindingZ - (endZ + bindingZ) * exitProgress;
                }}

                if (sphereId !== null) viewer.removeShape(sphereId);
                sphereId = viewer.addSphere({{
                    center: {{x: 0, y: 0, z: z}},
                    radius: 3,
                    color: color,
                    opacity: 0.9
                }});
                viewer.render();
                requestAnimationFrame(animatePermeation);
            }}

            animatePermeation();
            """
        else:
            mol_sphere_js = f"""
            viewer.addSphere({{
                center: {{x: 0, y: 0, z: {z_pos}}},
                radius: 3,
                color: {color},
                opacity: 0.9
            }});
            """

    style_js = ""
    if style == "stick":
        style_js = 'viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.25}});'
    elif style == "line":
        style_js = 'viewer.setStyle({}, {line: {linewidth: 1.5}});'
    else:
        style_js = 'viewer.setStyle({}, {sphere: {scale: 0.4}});'

    # Escape PDB data for JavaScript
    pdb_escaped = pdb_data.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')

    html = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="viewer" style="width: 100%; height: 500px; position: relative;"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x1a1a1a"}});
        var pdb = `{pdb_escaped}`;
        viewer.addModel(pdb, "pdb");
        {style_js}
        {mol_sphere_js}
        viewer.zoomTo();
        viewer.render();
        {animation_js}
    </script>
    """
    components.html(html, height=520)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Membrane Composition")

    st.markdown("**Lipids (Top / Bottom)**")

    # Lipid inputs
    lipid_names = ["POPC", "POPE", "POPS", "CHOL", "POPG"]
    lipid_values = {}

    for lip in lipid_names:
        c1, c2 = st.columns(2)
        with c1:
            top = st.number_input(f"{lip} Top", value=64 if lip == "POPC" else 0, min_value=0, key=f"{lip}_top")
        with c2:
            bottom = st.number_input(f"{lip} Bottom", value=64 if lip == "POPC" else 0, min_value=0, key=f"{lip}_bottom")
        lipid_values[lip] = (top, bottom)

    # Calculate suggested box dimensions
    suggested_x, suggested_y = calculate_box_dimensions(lipid_values)

    # Show calculated APL info
    total_counts = {lip: v[0] + v[1] for lip, v in lipid_values.items()}
    total_lipids = sum(total_counts.values())
    chol_count = total_counts.get("CHOL", 0)

    if total_lipids > 0:
        effective_apl = calculate_effective_apl(total_counts, chol_count, total_lipids)
        chol_pct = (chol_count / total_lipids) * 100

        st.divider()
        st.markdown("**Box Dimensions**")

        # Auto-calculate toggle
        auto_calc = st.checkbox("Auto-calculate from Area per Lipid", value=True)

        if auto_calc:
            box_x = suggested_x
            box_y = suggested_y
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

    # Build button
    if st.button("Build Membrane", type="primary", use_container_width=True):
        lipids = {k: v for k, v in lipid_values.items() if v[0] > 0 or v[1] > 0}

        if not lipids:
            st.error("Add at least one lipid")
        else:
            with st.spinner("Building membrane..."):
                try:
                    config = MembraneConfig.create_simple(
                        lipids=lipids,
                        box_size=(float(box_x), float(box_y), 120.0),
                    )

                    builder = MembraneBuilder(seed=12345)
                    membrane = builder.build(
                        config,
                        use_templates=True,
                        templates_dir="Lipids"
                    )

                    st.session_state.pdb_data = membrane.to_pdb_string()
                    st.session_state.membrane_props = {
                        'thickness': round(membrane.properties.thickness, 1),
                        'area_per_lipid': round(membrane.properties.area_per_lipid, 1),
                        'bending_modulus': round(membrane.properties.bending_modulus, 2),
                        'total_lipids': membrane.properties.total_lipids,
                        'n_atoms': membrane.n_atoms
                    }
                    st.session_state.perm_result = None  # Reset permeability
                    st.success("Membrane built!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Download button
    if st.session_state.pdb_data:
        st.download_button(
            "Download PDB",
            st.session_state.pdb_data,
            file_name="membrane.pdb",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # Permeability section
    st.subheader("Permeability Calculator")

    # Molecule presets
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
        mol_name = selected_mol.lower()
        mol_mw = preset["mw"]
        mol_asa = preset["asa"]
        mol_hbd = preset["hbd"]
        mol_hba = preset["hba"]
        mol_charge = preset["charge"]
        mol_pka = preset["pka"]
    else:
        mol_name = "custom"
        mol_mw = 100
        mol_asa = 100
        mol_hbd = 0
        mol_hba = 0
        mol_charge = 0
        mol_pka = None

    c1, c2 = st.columns(2)
    with c1:
        mol_mw = st.number_input("Molecular Weight", value=mol_mw, min_value=1)
        mol_hbd = st.number_input("H-Bond Donors", value=mol_hbd, min_value=0)
        mol_charge = st.number_input("Charge", value=float(mol_charge), step=0.1)
    with c2:
        mol_asa = st.number_input("Surface Area (A^2)", value=mol_asa, min_value=1)
        mol_hba = st.number_input("H-Bond Acceptors", value=mol_hba, min_value=0)
        pka_input = st.text_input("pKa", value=str(mol_pka) if mol_pka else "")

    if st.button("Calculate Permeability", type="primary", use_container_width=True):
        if not st.session_state.pdb_data:
            st.error("Build a membrane first")
        else:
            with st.spinner("Calculating..."):
                try:
                    pka_val = float(pka_input) if pka_input else None

                    mol = MoleculeDescriptor.simple(
                        name=mol_name,
                        molecular_weight=float(mol_mw),
                        total_asa=float(mol_asa),
                        n_hbd=int(mol_hbd),
                        n_hba=int(mol_hba),
                        charge=float(mol_charge),
                        pka=pka_val
                    )

                    # Get composition from current lipids
                    composition = {}
                    for lip, (top, bottom) in lipid_values.items():
                        total = top + bottom
                        if total > 0:
                            composition[lip] = total

                    if not composition:
                        composition = {"POPC": 128}

                    predictor = PermeabilityPredictor(composition=composition)
                    result = predictor.calculate(mol, "BLM")

                    classification = "low"
                    if result.log_p > -6:
                        classification = "high"
                    elif result.log_p > -8:
                        classification = "moderate"

                    st.session_state.perm_result = {
                        'log_p': round(result.log_p, 2),
                        'permeability': f"{result.permeability_cm_s:.2e}",
                        'binding_energy': round(result.membrane_bound_energy, 1),
                        'binding_position': round(result.binding_position, 1),
                        'classification': classification,
                        'mol_name': mol_name
                    }
                    st.success("Calculated!")
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("3D Viewer")

    if st.session_state.pdb_data:
        # Style selector and animate button
        c1, c2 = st.columns([3, 1])
        with c1:
            style = st.radio("Style", ["stick", "line", "sphere"], horizontal=True)
        with c2:
            animate = False
            if st.session_state.perm_result:
                animate = st.button("Animate", use_container_width=True)

        # Molecule info for sphere
        mol_info = None
        if st.session_state.perm_result:
            mol_info = {
                'name': st.session_state.perm_result['mol_name'],
                'z': st.session_state.perm_result['binding_position']
            }

        # Render 3D viewer
        render_3dmol(st.session_state.pdb_data, style, mol_info, animate)

        # Membrane properties
        if st.session_state.membrane_props:
            st.markdown("**Membrane Properties**")
            props = st.session_state.membrane_props
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Lipids", props['total_lipids'])
            c2.metric("Atoms", f"{props['n_atoms']:,}")
            c3.metric("Thickness", f"{props['thickness']} A")
            c4.metric("Area/Lipid", f"{props['area_per_lipid']} A^2")
            c5.metric("Kc", f"{props['bending_modulus']} kT")

        # Permeability results
        if st.session_state.perm_result:
            st.divider()
            st.markdown("**Permeability Results**")
            res = st.session_state.perm_result
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("log P", res['log_p'])
            c2.metric("P (cm/s)", res['permeability'])
            c3.metric("Binding", f"{res['binding_energy']} kJ/mol")

            # Classification badge
            class_colors = {"high": "green", "moderate": "orange", "low": "red"}
            c4.markdown(f"**Classification**")
            c4.markdown(f":{class_colors[res['classification']]}[{res['classification'].upper()}]")
    else:
        st.info("Build a membrane to see the 3D structure")
