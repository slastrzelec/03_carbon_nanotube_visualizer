import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import streamlit as st
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

# Page configuration
st.set_page_config(
    page_title="Carbon Nanotubes 3D",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #4fd1c5 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def create_carbon_nanotube(nanotube_type='armchair', radius=2, length=12, 
                          atoms_per_ring=10, num_rings=8, n=5, m=5):
    """
    Creates coordinates of carbon atoms and bonds for a carbon nanotube.
    """
    atoms = []
    bonds = []
    
    if nanotube_type == 'vector':
        a_cc = 1.42
        a = np.sqrt(3) * a_cc
        C = a * np.sqrt(n**2 + m**2 + n*m)
        radius = C / (2 * np.pi)
        
        from math import gcd
        d = gcd(n, m)
        atoms_per_ring = 2 * (n**2 + m**2 + n*m) // d
        
        if n == m:
            chiral_angle = np.pi / 6
        elif m == 0:
            chiral_angle = 0
        else:
            chiral_angle = np.arctan(np.sqrt(3) * m / (2*n + m))
    
    for ring in range(num_rings):
        z = (ring / (num_rings - 1)) * length - length / 2
        
        for i in range(atoms_per_ring):
            angle = (i / atoms_per_ring) * 2 * np.pi
            
            if nanotube_type == 'zigzag' and ring % 2 == 1:
                angle += np.pi / atoms_per_ring
            elif nanotube_type == 'chiral':
                angle += ring * 0.3
            elif nanotube_type == 'vector':
                angle += ring * chiral_angle * 0.5
            
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            
            atoms.append([x, y, z, ring, i])
    
    atoms = np.array(atoms)
    
    for idx, atom in enumerate(atoms):
        x, y, z, ring, i = atom
        
        next_in_ring = None
        next_idx = int((i + 1) % atoms_per_ring)
        for j, a in enumerate(atoms):
            if a[3] == ring and a[4] == next_idx:
                next_in_ring = j
                break
        
        if next_in_ring is not None:
            bonds.append([idx, next_in_ring])
        
        if ring < num_rings - 1:
            for j, a in enumerate(atoms):
                if a[3] == ring + 1 and abs(a[4] - i) <= 1:
                    bonds.append([idx, j])
                    break
    
    return atoms[:, :3], bonds


def calculate_atom_properties(atoms, bonds, property_type='distance'):
    """
    Calculates properties for coloring atoms.
    """
    n_atoms = len(atoms)
    
    if property_type == 'distance':
        distances = np.zeros(n_atoms)
        counts = np.zeros(n_atoms)
        
        for bond in bonds:
            idx1, idx2 = bond
            dist = np.linalg.norm(atoms[idx1] - atoms[idx2])
            distances[idx1] += dist
            distances[idx2] += dist
            counts[idx1] += 1
            counts[idx2] += 1
        
        avg_distances = np.divide(distances, counts, where=counts > 0)
        return avg_distances
    
    elif property_type == 'coordination':
        coordination = np.zeros(n_atoms)
        for bond in bonds:
            coordination[bond[0]] += 1
            coordination[bond[1]] += 1
        return coordination
    
    elif property_type == 'charge':
        z_coords = atoms[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        normalized_z = (z_coords - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_coords)
        charge = np.abs(normalized_z - 0.5) * 2
        return charge
    
    elif property_type == 'height':
        return atoms[:, 2]
    
    return np.ones(n_atoms)


def visualize_nanotube(nanotube_type='armchair', radius=2, atoms_per_ring=10, 
                      num_rings=8, elevation=20, azimuth=45, render_style='ball_and_stick',
                      color_by='none', n=5, m=5, theme='dark', quality='high'):
    """
    Visualizes carbon nanotube in 3D with enhanced graphics.
    """
    atoms, bonds = create_carbon_nanotube(
        nanotube_type=nanotube_type,
        radius=radius,
        atoms_per_ring=atoms_per_ring,
        num_rings=num_rings,
        n=n,
        m=m
    )
    
    # Calculate coloring values
    if color_by != 'none':
        color_values = calculate_atom_properties(atoms, bonds, color_by)
        norm_colors = (color_values - color_values.min()) / (color_values.max() - color_values.min()) if color_values.max() > color_values.min() else np.ones_like(color_values) * 0.5
    
    # Color scheme based on theme
    if theme == 'dark':
        fig_bg = '#0a0e27'
        plot_bg = '#1a1f3a'
        text_color = '#e0e6ed'
        grid_color = '#2d3561'
        grid_alpha = 0.15
        bond_colors = ['#4a90e2', '#5ba3f5', '#7cb8ff']
        pane_alpha = 0.05
    elif theme == 'light':
        fig_bg = '#f8f9fa'
        plot_bg = '#ffffff'
        text_color = '#2c3e50'
        grid_color = '#cbd5e0'
        grid_alpha = 0.3
        bond_colors = ['#4a5568', '#718096', '#a0aec0']
        pane_alpha = 0.02
    elif theme == 'neon':
        fig_bg = '#000000'
        plot_bg = '#0d0221'
        text_color = '#00ff9f'
        grid_color = '#ff006e'
        grid_alpha = 0.2
        bond_colors = ['#00ff9f', '#00d9ff', '#ff006e']
        pane_alpha = 0.03
    else:  # gradient
        fig_bg = '#1a1a2e'
        plot_bg = '#16213e'
        text_color = '#f0a500'
        grid_color = '#0f3460'
        grid_alpha = 0.25
        bond_colors = ['#e94560', '#f0a500', '#16a085']
        pane_alpha = 0.04
    
    # Larger figure for better quality
    dpi = 150 if quality == 'high' else 100
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    fig.patch.set_facecolor(fig_bg)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(plot_bg)
    
    # Draw bonds with gradients
    if render_style in ['ball_and_stick', 'wireframe', 'stick']:
        for idx, bond in enumerate(bonds):
            atom1, atom2 = atoms[bond[0]], atoms[bond[1]]
            
            # Gradient colors for bonds
            bond_color = bond_colors[idx % len(bond_colors)]
            
            if render_style == 'wireframe':
                linewidth = 2.5
                alpha = 0.7
            elif render_style == 'stick':
                linewidth = 4.0
                alpha = 0.8
            else:
                linewidth = 2.0
                alpha = 0.6
            
            ax.plot3D(
                [atom1[0], atom2[0]], 
                [atom1[1], atom2[1]], 
                [atom1[2], atom2[2]], 
                color=bond_color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle='round'
            )
    
    # Draw atoms
    if render_style != 'wireframe':
        if render_style == 'space_filling':
            size = 500
        elif render_style == 'stick':
            size = 100
        else:
            size = 200
        
        if color_by != 'none':
            # Choose colormap
            if theme == 'neon':
                cmap = plt.cm.plasma
            elif theme == 'gradient':
                cmap = plt.cm.twilight
            else:
                cmap = plt.cm.viridis
            
            scatter = ax.scatter(
                atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                c=norm_colors,
                cmap=cmap,
                s=size,
                alpha=0.9,
                edgecolors='white',
                linewidth=1.5 if render_style != 'space_filling' else 0,
                depthshade=True
            )
            
            # Colorbar with better style
            cbar = plt.colorbar(scatter, ax=ax, pad=0.12, shrink=0.7, aspect=20)
            cbar.set_label(
                color_by.capitalize(), 
                color=text_color, 
                fontsize=12, 
                fontweight='bold'
            )
            cbar.ax.yaxis.set_tick_params(color=text_color)
            cbar.outline.set_edgecolor(text_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
        else:
            # Gradient colors for atoms by height
            if theme == 'neon':
                atom_colors = plt.cm.plasma((atoms[:, 2] - atoms[:, 2].min()) / (atoms[:, 2].max() - atoms[:, 2].min()))
            elif theme == 'gradient':
                atom_colors = plt.cm.twilight((atoms[:, 2] - atoms[:, 2].min()) / (atoms[:, 2].max() - atoms[:, 2].min()))
            elif theme == 'dark':
                atom_colors = plt.cm.cool((atoms[:, 2] - atoms[:, 2].min()) / (atoms[:, 2].max() - atoms[:, 2].min()))
            else:
                atom_colors = '#2c3e50'
            
            ax.scatter(
                atoms[:, 0], atoms[:, 1], atoms[:, 2], 
                c=atom_colors,
                s=size,
                alpha=0.9,
                edgecolors='white' if theme == 'dark' or theme == 'neon' else 'black',
                linewidth=1.5 if render_style != 'space_filling' else 0,
                depthshade=True
            )
    
    # Axis labels with better style
    ax.set_xlabel('X [Å]', fontsize=13, labelpad=15, color=text_color, fontweight='bold')
    ax.set_ylabel('Y [Å]', fontsize=13, labelpad=15, color=text_color, fontweight='bold')
    ax.set_zlabel('Z [Å]', fontsize=13, labelpad=15, color=text_color, fontweight='bold')
    
    # Tick colors
    ax.tick_params(colors=text_color, labelsize=10)
    
    # Transparent panes with subtle color
    ax.xaxis.pane.set_facecolor(plot_bg)
    ax.yaxis.pane.set_facecolor(plot_bg)
    ax.zaxis.pane.set_facecolor(plot_bg)
    ax.xaxis.pane.set_alpha(pane_alpha)
    ax.yaxis.pane.set_alpha(pane_alpha)
    ax.zaxis.pane.set_alpha(pane_alpha)
    
    ax.xaxis.pane.set_edgecolor(grid_color)
    ax.yaxis.pane.set_edgecolor(grid_color)
    ax.zaxis.pane.set_edgecolor(grid_color)
    
    # Title
    type_descriptions = {
        'armchair': 'Armchair - Metallic Conductor',
        'zigzag': 'Zigzag - Variable Properties',
        'chiral': 'Chiral - Spiral Structure',
        'vector': f'Vector ({n},{m}) - Custom'
    }
    
    title = f'⚛️ Carbon Nanotube: {nanotube_type.upper()}\n{type_descriptions[nanotube_type]}'
    
    if nanotube_type == 'vector':
        a_cc = 1.42
        a = np.sqrt(3) * a_cc
        C = a * np.sqrt(n**2 + m**2 + n*m)
        diameter = C / np.pi
        
        if n == m:
            cnt_type = "Armchair"
        elif m == 0:
            cnt_type = "Zigzag"
        else:
            cnt_type = "Chiral"
        
        mod_diff = (n - m) % 3
        conductivity = "Metallic" if mod_diff == 0 else "Semiconductor"
        
        title = f'⚛️ Nanotube ({n},{m}) - {cnt_type}\n📏 Diameter: {diameter:.2f} Å  |  ⚡ {conductivity}'
    
    title_obj = ax.set_title(title, fontsize=16, pad=25, fontweight='bold', color=text_color)
    
    # Glow effect for title
    if theme in ['neon', 'gradient']:
        title_obj.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='black', alpha=0.8)
        ])
    
    # Axis proportions
    max_range = radius + 2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-7, 7])
    
    # Viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Grid
    ax.grid(True, alpha=grid_alpha, color=grid_color, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    return fig


# ============= STREAMLIT INTERFACE =============

# Header
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("""
    <h1 style='font-size: 48px; margin-bottom: 0;'>
        ⚛️ Carbon Nanotubes 3D
    </h1>
    <p style='font-size: 18px; color: #a0aec0; margin-top: 5px;'>
        Interactive visualization of carbon structures with advanced graphics
    </p>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# ⚙️ Control Panel")
st.sidebar.markdown("---")

# Section: Theme
st.sidebar.markdown("### 🎨 Visual Theme")
theme = st.sidebar.radio(
    "Choose theme:",
    options=['dark', 'light', 'neon', 'gradient'],
    format_func=lambda x: {
        'dark': '🌙 Dark (Classic)',
        'light': '☀️ Light (Minimalist)', 
        'neon': '✨ Neon (Cyberpunk)',
        'gradient': '🌈 Gradient (Artistic)'
    }[x],
    index=0
)

st.sidebar.markdown("---")

# Section: Nanotube type
st.sidebar.markdown("### 🔬 Nanotube Type")
nanotube_type = st.sidebar.selectbox(
    "Structure:",
    options=['armchair', 'zigzag', 'chiral', 'vector'],
    format_func=lambda x: {
        'armchair': '⬡ Armchair',
        'zigzag': '⚡ Zigzag',
        'chiral': '🌀 Chiral',
        'vector': '📐 Vector (n,m)'
    }[x],
    index=0
)

# Chiral vector parameters
if nanotube_type == 'vector':
    st.sidebar.markdown("#### 🔢 Chiral Vector")
    col_n, col_m = st.sidebar.columns(2)
    with col_n:
        n = st.number_input("n:", min_value=1, max_value=20, value=5, step=1)
    with col_m:
        m = st.number_input("m:", min_value=0, max_value=20, value=5, step=1)
    
    # Type info
    if n == m:
        st.sidebar.success("✓ **Armchair** (Metallic)")
    elif m == 0:
        st.sidebar.success("✓ **Zigzag**")
    else:
        mod_diff = (n - m) % 3
        cond = "Metallic" if mod_diff == 0 else "Semiconductor"
        st.sidebar.success(f"✓ **Chiral** ({cond})")
    
    # Calculate diameter
    a_cc = 1.42
    a = np.sqrt(3) * a_cc
    C = a * np.sqrt(n**2 + m**2 + n*m)
    diameter = C / np.pi
    st.sidebar.info(f"📏 Diameter: **{diameter:.2f} Å**")
else:
    n, m = 5, 5

st.sidebar.markdown("---")

# Section: Rendering style
st.sidebar.markdown("### 🎭 Rendering Style")
render_style = st.sidebar.selectbox(
    "Visualization:",
    options=['ball_and_stick', 'wireframe', 'space_filling', 'stick'],
    format_func=lambda x: {
        'ball_and_stick': '⚫ Ball and Stick',
        'wireframe': '📐 Wireframe',
        'space_filling': '🔵 Space Filling',
        'stick': '📏 Stick'
    }[x],
    index=0
)

quality = st.sidebar.select_slider(
    "Render quality:",
    options=['standard', 'high'],
    value='high',
    format_func=lambda x: '⭐ Standard' if x == 'standard' else '⭐⭐ High'
)

st.sidebar.markdown("---")

# Section: Coloring
st.sidebar.markdown("### 🎨 Atom Coloring")
color_by = st.sidebar.selectbox(
    "Color by:",
    options=['none', 'distance', 'coordination', 'charge', 'height'],
    format_func=lambda x: {
        'none': '⚪ Default',
        'distance': '📏 Distance',
        'coordination': '🔗 Coordination',
        'charge': '⚡ Charge',
        'height': '📊 Height'
    }[x],
    index=0
)

st.sidebar.markdown("---")

# Section: Structural parameters
st.sidebar.markdown("### ⚙️ Structure Parameters")

if nanotube_type != 'vector':
    radius = st.sidebar.slider("Radius [Å]:", 1.0, 4.0, 2.0, 0.5)
    atoms_per_ring = st.sidebar.slider("Atoms/ring:", 6, 16, 10, 2)
else:
    radius = 2.0
    atoms_per_ring = 10
    st.sidebar.info("ℹ️ Parameters calculated from (n,m)")

num_rings = st.sidebar.slider("Number of rings:", 5, 15, 8, 1)

st.sidebar.markdown("---")

# Section: Viewing angle
st.sidebar.markdown("### 👁️ Viewing Angle")
elevation = st.sidebar.slider("Elevation (°):", 0, 90, 20, 5)
azimuth = st.sidebar.slider("Azimuth (°):", 0, 360, 45, 15)

# Main content
col1, col2 = st.columns([2.5, 1])

with col1:
    with st.spinner('🔄 Generating visualization...'):
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
            theme=theme,
            quality=quality
        )
        st.pyplot(fig)
        plt.close()
    
    # Export buttons
    st.markdown("---")
    st.markdown("### 💾 Export Visualization")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    from io import BytesIO
    
    with col_exp1:
        buf_png = BytesIO()
        fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf_png.seek(0)
        st.download_button(
            label="📥 PNG (High Quality)",
            data=buf_png,
            file_name=f"nanotube_{nanotube_type}_{theme}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col_exp2:
        buf_svg = BytesIO()
        fig.savefig(buf_svg, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
        buf_svg.seek(0)
        st.download_button(
            label="📥 SVG (Vector)",
            data=buf_svg,
            file_name=f"nanotube_{nanotube_type}_{theme}.svg",
            mime="image/svg+xml",
            use_container_width=True
        )
    
    with col_exp3:
        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format='pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        buf_pdf.seek(0)
        st.download_button(
            label="📥 PDF (Document)",
            data=buf_pdf,
            file_name=f"nanotube_{nanotube_type}_{theme}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with col2:
    st.markdown("### 📊 Information")
    
    # Statistics
    total_atoms = atoms_per_ring * num_rings
    st.metric("💎 Total atoms", f"{total_atoms}")
    st.metric("🔄 Rings", f"{num_rings}")
    st.metric("⭕ Atoms/ring", f"{atoms_per_ring}")
    
    if nanotube_type == 'vector':
        st.metric("📏 Diameter", f"{diameter:.2f} Å")
    else:
        st.metric("📏 Radius", f"{radius:.1f} Å")
    
    st.markdown("---")
    
    # Type info
    if nanotube_type == 'armchair':
        st.info("""
        **⬡ Armchair Type**
        - ⚡ Metallic conductor
        - 🔋 High conductivity
        - ⚖️ Symmetric structure
        - 🎯 Electronically stable
        """)
    elif nanotube_type == 'zigzag':
        st.info("""
        **⚡ Zigzag Type**
        - 🔀 Variable conductivity
        - 📊 Depends on diameter
        - 🔷 Characteristic pattern
        - 🎲 Can be metallic or semiconducting
        """)
    elif nanotube_type == 'chiral':
        st.info("""
        **🌀 Chiral Type**
        - 🌊 Spiral structure
        - 🔌 Usually semiconductor
        - ⚖️ Asymmetric construction
        - 📐 Properties depend on angle
        """)
    else:
        mod_diff = (n - m) % 3
        conductivity = "Metallic" if mod_diff == 0 else "Semiconductor"
        st.info(f"""
        **📐 Vector ({n},{m})**
        - 🔬 {conductivity}
        - 📏 Diameter: {diameter:.2f} Å
        - 🎯 (n-m) mod 3 = {mod_diff}
        - ⚙️ Precise control
        """)
    
    st.markdown("---")
    
    # General information
    st.markdown("""
    ### 🔬 CNT Properties
    
    - 💪 **Strength:** >100× steel
    - ⚡ **Conductivity:** Excellent
    - 🌡️ **Thermal conductivity:** Very high
    - 📏 **Diameter:** 0.7-10 nm
    - 📅 **Discovery:** 1991 (S. Iijima)
    - 🎓 **Applications:** Electronics, composites, sensors
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; opacity: 0.7;'>
    <p style='font-size: 14px;'>
        💡 <b>Tip:</b> Use the controls on the left to customize the visualization<br>
        🎨 Try different themes for unique visual effects
    </p>
</div>
""", unsafe_allow_html=True)