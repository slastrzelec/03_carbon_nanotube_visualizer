import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

def create_carbon_nanotube(nanotube_type='armchair', radius=2, length=12, 
                          atoms_per_ring=10, num_rings=8, n=5, m=5):
    """
    Creates coordinates of carbon atoms and bonds for a carbon nanotube.
    
    Parameters:
    - nanotube_type: 'armchair', 'zigzag', 'chiral', or 'vector'
    - radius: nanotube radius (used only if not 'vector' type)
    - length: nanotube length
    - atoms_per_ring: number of atoms in one ring (used only if not 'vector' type)
    - num_rings: number of rings
    - n, m: chiral vector indices (n,m) - used when nanotube_type='vector'
    """
    atoms = []
    bonds = []
    
    # Calculate parameters from chiral vector if type is 'vector'
    if nanotube_type == 'vector':
        # Calculate radius from chiral vector (n,m)
        # Carbon-carbon bond length in graphene (Angstroms)
        a_cc = 1.42  
        # Lattice constant
        a = np.sqrt(3) * a_cc
        # Circumference
        C = a * np.sqrt(n**2 + m**2 + n*m)
        # Radius
        radius = C / (2 * np.pi)
        
        # Determine number of atoms per ring based on (n,m)
        # GCD for unit cell
        from math import gcd
        d = gcd(n, m)
        atoms_per_ring = 2 * (n**2 + m**2 + n*m) // d
        
        # Chiral angle
        if n == m:
            chiral_angle = np.pi / 6  # 30 degrees - armchair
        elif m == 0:
            chiral_angle = 0  # zigzag
        else:
            chiral_angle = np.arctan(np.sqrt(3) * m / (2*n + m))
    
    # Generate atoms
    for ring in range(num_rings):
        z = (ring / (num_rings - 1)) * length - length / 2
        
        for i in range(atoms_per_ring):
            angle = (i / atoms_per_ring) * 2 * np.pi
            
            # Angle modification depending on nanotube type
            if nanotube_type == 'zigzag' and ring % 2 == 1:
                angle += np.pi / atoms_per_ring
            elif nanotube_type == 'chiral':
                angle += ring * 0.3
            elif nanotube_type == 'vector':
                # Apply chiral angle for vector-defined nanotubes
                angle += ring * chiral_angle * 0.5
            
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            
            atoms.append([x, y, z, ring, i])
    
    atoms = np.array(atoms)
    
    # Generate bonds
    for idx, atom in enumerate(atoms):
        x, y, z, ring, i = atom
        
        # Bonds within the ring
        next_in_ring = None
        next_idx = int((i + 1) % atoms_per_ring)
        for j, a in enumerate(atoms):
            if a[3] == ring and a[4] == next_idx:
                next_in_ring = j
                break
        
        if next_in_ring is not None:
            bonds.append([idx, next_in_ring])
        
        # Bonds between rings
        if ring < num_rings - 1:
            for j, a in enumerate(atoms):
                if a[3] == ring + 1 and abs(a[4] - i) <= 1:
                    bonds.append([idx, j])
                    break
    
    return atoms[:, :3], bonds


def calculate_atom_properties(atoms, bonds, property_type='distance'):
    """
    Calculates properties for coloring atoms.
    
    Parameters:
    - atoms: array of atom coordinates
    - bonds: list of bonds
    - property_type: 'distance', 'coordination', 'charge'
    
    Returns: array of values for coloring
    """
    n_atoms = len(atoms)
    
    if property_type == 'distance':
        # Average distance to neighboring atoms
        distances = np.zeros(n_atoms)
        counts = np.zeros(n_atoms)
        
        for bond in bonds:
            idx1, idx2 = bond
            dist = np.linalg.norm(atoms[idx1] - atoms[idx2])
            distances[idx1] += dist
            distances[idx2] += dist
            counts[idx1] += 1
            counts[idx2] += 1
        
        # Average distance
        avg_distances = np.divide(distances, counts, where=counts > 0)
        return avg_distances
    
    elif property_type == 'coordination':
        # Number of bonds per atom
        coordination = np.zeros(n_atoms)
        for bond in bonds:
            coordination[bond[0]] += 1
            coordination[bond[1]] += 1
        return coordination
    
    elif property_type == 'charge':
        # Simplified charge based on z-position (example: edge effects)
        z_coords = atoms[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        # Normalize to simulate charge distribution
        normalized_z = (z_coords - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_coords)
        # Simulate charge: edges have different charge
        charge = np.abs(normalized_z - 0.5) * 2  # Higher at edges
        return charge
    
    return np.ones(n_atoms)


def visualize_nanotube(nanotube_type='armchair', radius=2, atoms_per_ring=10, 
                      num_rings=8, elevation=20, azimuth=45, render_style='ball_and_stick',
                      color_by='none', n=5, m=5, dark_mode=False):
    """
    Visualizes carbon nanotube in 3D.
    """
    # Create nanotube
    atoms, bonds = create_carbon_nanotube(
        nanotube_type=nanotube_type,
        radius=radius,
        atoms_per_ring=atoms_per_ring,
        num_rings=num_rings,
        n=n,
        m=m
    )
    
    # Calculate coloring properties
    if color_by != 'none':
        color_values = calculate_atom_properties(atoms, bonds, color_by)
        colormap = plt.cm.viridis
        atom_colors = colormap((color_values - color_values.min()) / 
                              (color_values.max() - color_values.min()) 
                              if color_values.max() > color_values.min() else 0.5)
    else:
        atom_colors = 'black'
    
    # Color scheme based on mode
    if dark_mode:
        bg_color = '#1a1a1a'
        plot_bg_color = '#2d2d2d'
        text_color = 'white'
        grid_color = 'gray'
        grid_alpha = 0.2
        bond_color = '#888888'
        atom_color = 'white' if color_by == 'none' else None
    else:
        bg_color = 'white'
        plot_bg_color = '#f8f9fa'
        text_color = 'black'
        grid_color = 'gray'
        grid_alpha = 0.3
        bond_color = 'gray'
        atom_color = 'black' if color_by == 'none' else None
    
    # Create plot
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(plot_bg_color)
    
    # Rendering based on style
    if render_style == 'ball_and_stick':
        # Draw bonds
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     bond_color, linewidth=1.5, alpha=0.6)
        
        # Draw atoms
        if color_by != 'none':
            scatter = ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=color_values, cmap='viridis', s=150, alpha=0.9, 
                      edgecolors='white', linewidth=1)
            plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8, label=color_by.capitalize())
        else:
            ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=atom_color, s=150, alpha=0.9, edgecolors=bond_color, linewidth=1)
    
    elif render_style == 'wireframe':
        # Only bonds, no atoms
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     atom_color if atom_color else bond_color, linewidth=2, alpha=0.8)
    
    elif render_style == 'space_filling':
        # Large spheres, no bonds
        if color_by != 'none':
            scatter = ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=color_values, cmap='viridis', s=400, alpha=0.85, edgecolors='none')
            plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8, label=color_by.capitalize())
        else:
            ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=atom_color, s=400, alpha=0.85, edgecolors='none')
    
    elif render_style == 'stick':
        # Thick bonds only
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     atom_color if atom_color else bond_color, linewidth=3, alpha=0.7)
        
        # Small atoms
        if color_by != 'none':
            scatter = ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=color_values, cmap='viridis', s=50, alpha=0.8)
            plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8, label=color_by.capitalize())
        else:
            ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                      c=atom_color, s=50, alpha=0.8)
    
    # Plot settings
    ax.set_xlabel('X [Å]', fontsize=11, labelpad=10, color=text_color)
    ax.set_ylabel('Y [Å]', fontsize=11, labelpad=10, color=text_color)
    ax.set_zlabel('Z [Å]', fontsize=11, labelpad=10, color=text_color)
    
    # Set tick colors
    ax.tick_params(colors=text_color)
    
    # Set pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    if dark_mode:
        ax.xaxis.pane.set_edgecolor(grid_color)
        ax.yaxis.pane.set_edgecolor(grid_color)
        ax.zaxis.pane.set_edgecolor(grid_color)
    
    # Title with description
    type_descriptions = {
        'armchair': 'Armchair - atoms arranged horizontally to the axis',
        'zigzag': 'Zigzag - atoms form a zigzag pattern',
        'chiral': 'Chiral - spiral arrangement of atoms',
        'vector': f'Chiral Vector ({n},{m}) - Custom nanotube'
    }
    
    title = f'Carbon Nanotube: {nanotube_type.capitalize()}\n{type_descriptions[nanotube_type]}'
    if nanotube_type == 'vector':
        # Calculate additional info
        a_cc = 1.42
        a = np.sqrt(3) * a_cc
        C = a * np.sqrt(n**2 + m**2 + n*m)
        diameter = C / np.pi
        
        # Determine type from (n,m)
        if n == m:
            cnt_type = "Armchair"
        elif m == 0:
            cnt_type = "Zigzag"
        else:
            cnt_type = "Chiral"
        
        title = f'Carbon Nanotube ({n},{m}) - {cnt_type}\nDiameter: {diameter:.2f} Å'
    
    ax.set_title(title, fontsize=13, pad=20, fontweight='bold', color=text_color)
    
    # Set axis proportions
    max_range = radius + 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-7, 7])
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Grid
    ax.grid(True, alpha=grid_alpha, color=grid_color)
    
    plt.tight_layout()
    return fig


# Streamlit page configuration
st.set_page_config(
    page_title="Carbon Nanotube Visualization",
    page_icon="⚛️",
    layout="wide"
)

# Application title
st.title("⚛️ Carbon Nanotube Visualization")
st.markdown("""
Interactive visualization of different types of carbon nanotubes. 
Carbon nanotubes are cylindrical structures built from carbon atoms arranged in a hexagonal lattice.
""")

# Sidebar with controls
st.sidebar.header("⚙️ Visualization Parameters")

nanotube_type = st.sidebar.selectbox(
    "Nanotube type:",
    options=['armchair', 'zigzag', 'chiral', 'vector'],
    index=0
)

# Chiral vector input (only for 'vector' type)
if nanotube_type == 'vector':
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔢 Chiral Vector (n,m)")
    
    col_n, col_m = st.sidebar.columns(2)
    with col_n:
        n = st.number_input("n:", min_value=1, max_value=20, value=5, step=1)
    with col_m:
        m = st.number_input("m:", min_value=0, max_value=20, value=5, step=1)
    
    # Show what type it is
    if n == m:
        st.sidebar.success("✓ This is an **Armchair** nanotube")
    elif m == 0:
        st.sidebar.success("✓ This is a **Zigzag** nanotube")
    else:
        st.sidebar.success("✓ This is a **Chiral** nanotube")
    
    # Calculate diameter
    a_cc = 1.42
    a = np.sqrt(3) * a_cc
    C = a * np.sqrt(n**2 + m**2 + n*m)
    diameter = C / np.pi
    st.sidebar.info(f"📏 Calculated diameter: **{diameter:.2f} Å**")
else:
    n, m = 5, 5  # Default values

st.sidebar.markdown("---")
st.sidebar.subheader("Rendering style")

render_style = st.sidebar.selectbox(
    "Visualization style:",
    options=['ball_and_stick', 'wireframe', 'space_filling', 'stick'],
    index=0,
    help="Choose how the nanotube should be rendered"
)

# Style descriptions
style_descriptions = {
    'ball_and_stick': '⚫ Ball and Stick - Classic molecular representation',
    'wireframe': '📐 Wireframe - Bonds only, minimal style',
    'space_filling': '🔵 Space Filling - Van der Waals spheres',
    'stick': '📏 Stick - Thick bonds with small atoms'
}

st.sidebar.info(style_descriptions[render_style])

st.sidebar.markdown("---")
st.sidebar.subheader("Theme")

dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=False,
    help="Switch between light and dark theme"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Color mapping")

color_by = st.sidebar.selectbox(
    "Color atoms by:",
    options=['none', 'distance', 'coordination', 'charge'],
    index=0,
    help="Choose property for atom coloring"
)

# Color descriptions
color_descriptions = {
    'none': '⚫ Default - All atoms black',
    'distance': '📏 Distance - Average bond length',
    'coordination': '🔗 Coordination - Number of bonds',
    'charge': '⚡ Charge - Simulated charge distribution'
}

if color_by != 'none':
    st.sidebar.info(color_descriptions[color_by])

st.sidebar.markdown("---")
st.sidebar.subheader("Structural parameters")

# Only show manual parameters if not using vector mode
if nanotube_type != 'vector':
    radius = st.sidebar.slider(
        "Radius [Å]:",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.5
    )

    if nanotube_type == 'zigzag':
        default_atoms = 8
    else:
        default_atoms = 10

    atoms_per_ring = st.sidebar.slider(
        "Atoms per ring:",
        min_value=6,
        max_value=16,
        value=default_atoms,
        step=2
    )
else:
    # For vector mode, these are calculated automatically
    radius = 2.0  # Will be overridden
    atoms_per_ring = 10  # Will be overridden
    st.sidebar.info("ℹ️ Radius and atoms per ring are calculated from (n,m)")

num_rings = st.sidebar.slider(
    "Number of rings:",
    min_value=5,
    max_value=15,
    value=8,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Viewing angle")

elevation = st.sidebar.slider(
    "Vertical angle (elevation):",
    min_value=0,
    max_value=90,
    value=20,
    step=5
)

azimuth = st.sidebar.slider(
    "Horizontal angle (azimuth):",
    min_value=0,
    max_value=360,
    value=45,
    step=15
)

# Columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Generate and display visualization
    with st.spinner('Generating visualization...'):
        fig = visualize_nanotube(
            nanotube_type=nanotube_type,
            radius=radius,
            atoms_per_ring=atoms_per_ring,
            num_rings=num_rings,
            elevation=elevation,
            azimuth=azimuth,
            render_style=render_style,
            color_by=color_by,
            n=n,
            m=m,
            dark_mode=dark_mode
        )
        st.pyplot(fig)
        
        # Export options
        st.markdown("---")
        st.subheader("📥 Export Visualization")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        # Save figure to buffer for download
        from io import BytesIO
        
        # PNG export
        with col_exp1:
            buf_png = BytesIO()
            fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
            buf_png.seek(0)
            st.download_button(
                label="💾 Download PNG",
                data=buf_png,
                file_name=f"nanotube_{nanotube_type}.png",
                mime="image/png",
                use_container_width=True
            )
        
        # SVG export
        with col_exp2:
            buf_svg = BytesIO()
            fig.savefig(buf_svg, format='svg', bbox_inches='tight')
            buf_svg.seek(0)
            st.download_button(
                label="💾 Download SVG",
                data=buf_svg,
                file_name=f"nanotube_{nanotube_type}.svg",
                mime="image/svg+xml",
                use_container_width=True
            )
        
        # PDF export
        with col_exp3:
            buf_pdf = BytesIO()
            fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
            buf_pdf.seek(0)
            st.download_button(
                label="💾 Download PDF",
                data=buf_pdf,
                file_name=f"nanotube_{nanotube_type}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        plt.close()

with col2:
    st.subheader("ℹ️ Information")
    
    # Information about nanotube type
    if nanotube_type == 'armchair':
        st.info("""
        **Armchair Type**
        - Atoms arranged horizontally to the axis
        - Can be a metallic conductor
        - High electrical conductivity
        - Symmetric structure
        """)
    elif nanotube_type == 'zigzag':
        st.info("""
        **Zigzag Type**
        - Atoms form a zigzag pattern
        - Conductivity depends on diameter
        - Can be conductor or semiconductor
        - Characteristic edge pattern
        """)
    elif nanotube_type == 'chiral':
        st.info("""
        **Chiral Type**
        - Spiral arrangement of atoms
        - Usually a semiconductor
        - Asymmetric structure
        - Properties depend on chirality angle
        """)
    else:  # vector
        st.info(f"""
        **Chiral Vector ({n},{m})**
        - Custom nanotube defined by indices
        - Type: {'Armchair' if n == m else 'Zigzag' if m == 0 else 'Chiral'}
        - Properties calculated from (n,m)
        - Precise control over structure
        """)
        
        # Additional calculations for vector mode
        if nanotube_type == 'vector':
            st.markdown("---")
            st.markdown("**📊 Calculated Properties:**")
            
            a_cc = 1.42
            a = np.sqrt(3) * a_cc
            C = a * np.sqrt(n**2 + m**2 + n*m)
            diameter = C / np.pi
            
            # Chiral angle
            if n == m:
                chiral_angle_deg = 30.0
            elif m == 0:
                chiral_angle_deg = 0.0
            else:
                chiral_angle_rad = np.arctan(np.sqrt(3) * m / (2*n + m))
                chiral_angle_deg = np.degrees(chiral_angle_rad)
            
            # Metallic or semiconducting
            mod_diff = (n - m) % 3
            conductivity = "Metallic" if mod_diff == 0 else "Semiconducting"
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Diameter", f"{diameter:.2f} Å")
                st.metric("Chiral angle", f"{chiral_angle_deg:.1f}°")
            with col_b:
                st.metric("Type", conductivity)
                st.metric("(n-m) mod 3", str(mod_diff))
    
    st.markdown("---")
    
    # General information
    st.markdown("""
    **Nanotube properties:**
    - ⚫ Black spheres = carbon atoms
    - ⚪ Gray lines = covalent bonds
    - 📏 Diameter: ~0.7-10 nm
    - 💪 Strength > 100x steel
    - ⚡ Electrical and thermal conductivity
    """)
    
    # Current structure statistics
    total_atoms = atoms_per_ring * num_rings
    st.metric("Number of atoms", total_atoms)
    st.metric("Radius", f"{radius} Å")
    st.metric("Atoms/ring", atoms_per_ring)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>💡 <b>Tip:</b> Use the controls on the left to customize the visualization</p>
    <p>Carbon nanotubes discovered in 1991 by Sumio Iijima</p>
</div>
""", unsafe_allow_html=True)