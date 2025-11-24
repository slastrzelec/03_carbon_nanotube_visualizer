import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

def create_carbon_nanotube(nanotube_type='armchair', radius=2, length=12, 
                          atoms_per_ring=10, num_rings=8):
    """
    Creates coordinates of carbon atoms and bonds for a carbon nanotube.
    
    Parameters:
    - nanotube_type: 'armchair', 'zigzag', or 'chiral'
    - radius: nanotube radius
    - length: nanotube length
    - atoms_per_ring: number of atoms in one ring
    - num_rings: number of rings
    """
    atoms = []
    bonds = []
    
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


def visualize_nanotube(nanotube_type='armchair', radius=2, atoms_per_ring=10, 
                      num_rings=8, elevation=20, azimuth=45, render_style='ball_and_stick'):
    """
    Visualizes carbon nanotube in 3D.
    """
    # Create nanotube
    atoms, bonds = create_carbon_nanotube(
        nanotube_type=nanotube_type,
        radius=radius,
        atoms_per_ring=atoms_per_ring,
        num_rings=num_rings
    )
    
    # Create plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Rendering based on style
    if render_style == 'ball_and_stick':
        # Draw bonds
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     'gray', linewidth=1.5, alpha=0.6)
        
        # Draw atoms
        ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                  c='black', s=150, alpha=0.9, edgecolors='white', linewidth=1)
    
    elif render_style == 'wireframe':
        # Only bonds, no atoms
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     'black', linewidth=2, alpha=0.8)
    
    elif render_style == 'space_filling':
        # Large spheres, no bonds
        ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                  c='black', s=400, alpha=0.85, edgecolors='none')
    
    elif render_style == 'points':
        # Simple points
        ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                  c='black', s=50, alpha=1.0, marker='o')
    
    elif render_style == 'stick':
        # Thick bonds only
        for bond in bonds:
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            ax.plot3D([atom1[0], atom2[0]], 
                     [atom1[1], atom2[1]], 
                     [atom1[2], atom2[2]], 
                     'black', linewidth=3, alpha=0.7)
        
        # Small atoms
        ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                  c='black', s=50, alpha=0.8)
    
    # Plot settings
    ax.set_xlabel('X [Å]', fontsize=11, labelpad=10)
    ax.set_ylabel('Y [Å]', fontsize=11, labelpad=10)
    ax.set_zlabel('Z [Å]', fontsize=11, labelpad=10)
    
    # Title with description
    type_descriptions = {
        'armchair': 'Armchair - atoms arranged horizontally to the axis',
        'zigzag': 'Zigzag - atoms form a zigzag pattern',
        'chiral': 'Chiral - spiral arrangement of atoms'
    }
    
    ax.set_title(f'Carbon Nanotube: {nanotube_type.capitalize()}\n{type_descriptions[nanotube_type]}', 
                fontsize=13, pad=20, fontweight='bold')
    
    # Set axis proportions
    max_range = radius + 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-7, 7])
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    
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
    options=['armchair', 'zigzag', 'chiral'],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Rendering style")

render_style = st.sidebar.selectbox(
    "Visualization style:",
    options=['ball_and_stick', 'wireframe', 'space_filling', 'stick', 'points'],
    index=0,
    help="Choose how the nanotube should be rendered"
)

# Style descriptions
style_descriptions = {
    'ball_and_stick': '⚫ Ball and Stick - Classic molecular representation',
    'wireframe': '📐 Wireframe - Bonds only, minimal style',
    'space_filling': '🔵 Space Filling - Van der Waals spheres',
    'stick': '📏 Stick - Thick bonds with small atoms',
    'points': '• Points - Atomic positions only'
}

st.sidebar.info(style_descriptions[render_style])

st.sidebar.markdown("---")
st.sidebar.subheader("Structural parameters")

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
            render_style=render_style
        )
        st.pyplot(fig)
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
    else:
        st.info("""
        **Chiral Type**
        - Spiral arrangement of atoms
        - Usually a semiconductor
        - Asymmetric structure
        - Properties depend on chirality angle
        """)
    
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