import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import trimesh
from skimage import measure
import plotly.graph_objects as go

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Project Helix – 3D MRI Tumor Analysis",
    layout="wide"
)

st.title("🧠 Project Helix – 3D MRI Tumor Analysis")
st.markdown(
    """
Upload one or more **MRI scans** (`.nii` / `.nii.gz`).  
The app will automatically:
- Process the scans
- Reconstruct a **3D tumor**
- Compute **tumor measurements**
- Allow **interactive peeling**
"""
)

# --------------------------------------------------
# Step 1: Upload MRI files
# --------------------------------------------------
st.subheader("📂 Upload MRI Scans")

uploaded_files = st.file_uploader(
    "Upload MRI scan files",
    type=["nii", "nii.gz"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more MRI scan files to continue.")
    st.stop()

# --------------------------------------------------
# Step 2: Load MRI volumes
# --------------------------------------------------
volumes = []

with st.spinner("Loading MRI scans..."):
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        nii = nib.load(tmp_path)
        vol = nii.get_fdata()
        volumes.append(vol)

st.success(f"{len(volumes)} MRI scan(s) loaded successfully")

# --------------------------------------------------
# Step 3: Combine multiple scans into one volume
# --------------------------------------------------
volume_3d = np.mean(volumes, axis=0)

# Normalize
volume_3d = (volume_3d - volume_3d.min()) / (volume_3d.max() - volume_3d.min())

# --------------------------------------------------
# Step 4: Tumor detection (simple MVP threshold)
# --------------------------------------------------
st.subheader("🧪 Tumor Detection")

tumor_mask = volume_3d > 0.7
tumor_voxels = int(tumor_mask.sum())

st.write("Detected tumor voxels:", tumor_voxels)

if tumor_voxels == 0:
    st.error("No tumor detected. Try different MRI scans.")
    st.stop()

# --------------------------------------------------
# Step 5: 3D Tumor Reconstruction
# --------------------------------------------------
with st.spinner("Reconstructing 3D tumor..."):
    verts, faces, _, _ = measure.marching_cubes(tumor_mask, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

st.success("3D tumor reconstructed")

# --------------------------------------------------
# Step 6: Tumor Measurements
# --------------------------------------------------
st.subheader("📊 Tumor Measurements")

tumor_volume = mesh.volume
bounds = mesh.bounds
diameter = np.linalg.norm(bounds[1] - bounds[0])

col1, col2 = st.columns(2)
col1.metric("Tumor Volume", f"{tumor_volume:.2f} cubic units")
col2.metric("Approx Diameter", f"{diameter:.2f} units")

# --------------------------------------------------
# Step 7: Peel / Slice Control
# --------------------------------------------------
st.subheader("🔍 Peel / Slice Tumor")

z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()

z_cut = st.slider(
    "Peel depth (top → bottom)",
    float(z_min),
    float(z_max),
    float(z_max)
)

verts = mesh.vertices
faces = mesh.faces

mask = verts[:, 2] <= z_cut

index_map = -np.ones(len(verts), dtype=int)
index_map[mask] = np.arange(mask.sum())

face_mask = np.all(mask[faces], axis=1)
new_faces = index_map[faces[face_mask]]
new_verts = verts[mask]

# --------------------------------------------------
# Step 8: Visualization
# --------------------------------------------------
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=new_verts[:, 0],
            y=new_verts[:, 1],
            z=new_verts[:, 2],
            i=new_faces[:, 0],
            j=new_faces[:, 1],
            k=new_faces[:, 2],
            color="red",
            opacity=0.7
        )
    ]
)

fig.update_layout(
    height=650,
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False
    ),
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Usage Instructions
# --------------------------------------------------
st.markdown(
    """
### 🧭 How to use
- **Rotate:** click + drag  
- **Zoom:** scroll  
- **Peel:** move the slider to explore internal tumor layers  

This MVP demonstrates **end‑to‑end MRI → 3D tumor reconstruction**.
"""
)
