import streamlit as st
import numpy as np
import nibabel as nib
import io
import torch
import torch.nn as nn
import trimesh
from skimage import measure
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Project Helix – MRI Tumor Analysis",
    layout="wide"
)

st.title("🧠 Project Helix – MRI Tumor Analysis")
st.markdown("""
Upload one or more **MRI scans** (`.nii` / `.nii.gz`).

The app will:
- Run **ML‑based tumor segmentation**
- Reconstruct **3D brain + tumor**
- Show **slice‑by‑slice 2D MRI**
- Display **tumor measurements**
""")

# --------------------------------------------------
# Simple 3D UNet (Inference wrapper)
# --------------------------------------------------
class SimpleUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@st.cache_resource
def load_model():
    model = SimpleUNet3D()
    state = torch.load("brats_pretrained.pth", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


model = load_model()

# --------------------------------------------------
# Upload MRI files
# --------------------------------------------------
st.subheader("📂 Upload MRI Scans")

uploaded_files = st.file_uploader(
    "Upload MRI scan files",
    type=["nii", "nii.gz"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one MRI file.")
    st.stop()

# --------------------------------------------------
# Load MRI volumes safely (BytesIO)
# --------------------------------------------------
volumes = []

with st.spinner("Loading MRI scans..."):
    for file in uploaded_files:
        bytes_data = file.read()
        file_obj = io.BytesIO(bytes_data)
        nii = nib.load(file_obj)
        vol = nii.get_fdata()
        volumes.append(vol)

st.success(f"{len(volumes)} MRI scan(s) loaded")

# --------------------------------------------------
# Combine & normalize MRI
# --------------------------------------------------
volume_3d = np.mean(volumes, axis=0)
volume_3d = (volume_3d - volume_3d.min()) / (volume_3d.max() - volume_3d.min())

# --------------------------------------------------
# ML‑based tumor segmentation
# --------------------------------------------------
st.subheader("🧠 ML‑Based Tumor Segmentation")

with st.spinner("Running ML inference..."):
    input_tensor = torch.tensor(volume_3d, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

    with torch.no_grad():
        prediction = model(input_tensor)

    tumor_mask = prediction.squeeze().numpy() > 0.5

st.success("Tumor segmentation complete")

if tumor_mask.sum() == 0:
    st.error("No tumor detected in the uploaded MRI.")
    st.stop()

# --------------------------------------------------
# Brain mask (organ)
# --------------------------------------------------
brain_mask = volume_3d > 0.1

# --------------------------------------------------
# 3D Reconstruction
# --------------------------------------------------
with st.spinner("Reconstructing 3D meshes..."):
    # Tumor
    t_verts, t_faces, _, _ = measure.marching_cubes(tumor_mask, level=0)
    tumor_mesh = trimesh.Trimesh(vertices=t_verts, faces=t_faces)

    # Brain
    b_verts, b_faces, _, _ = measure.marching_cubes(brain_mask, level=0)
    brain_mesh = trimesh.Trimesh(vertices=b_verts, faces=b_faces)

st.success("3D reconstruction complete")

# --------------------------------------------------
# Tumor measurements
# --------------------------------------------------
st.subheader("📊 Tumor Measurements")

tumor_volume = tumor_mesh.volume
bounds = tumor_mesh.bounds
diameter = np.linalg.norm(bounds[1] - bounds[0])

c1, c2 = st.columns(2)
c1.metric("Tumor Volume", f"{tumor_volume:.2f} cubic units")
c2.metric("Approx Diameter", f"{diameter:.2f} units")

# --------------------------------------------------
# 3D Visualization (Brain + Tumor)
# --------------------------------------------------
st.subheader("🧠 3D Brain & Tumor View")

fig3d = go.Figure()

# Brain
fig3d.add_trace(
    go.Mesh3d(
        x=brain_mesh.vertices[:, 0],
        y=brain_mesh.vertices[:, 1],
        z=brain_mesh.vertices[:, 2],
        i=brain_mesh.faces[:, 0],
        j=brain_mesh.faces[:, 1],
        k=brain_mesh.faces[:, 2],
        color="lightgray",
        opacity=0.15,
        name="Brain"
    )
)

# Tumor
fig3d.add_trace(
    go.Mesh3d(
        x=tumor_mesh.vertices[:, 0],
        y=tumor_mesh.vertices[:, 1],
        z=tumor_mesh.vertices[:, 2],
        i=tumor_mesh.faces[:, 0],
        j=tumor_mesh.faces[:, 1],
        k=tumor_mesh.faces[:, 2],
        color="red",
        opacity=0.85,
        name="Tumor"
    )
)

fig3d.update_layout(
    height=700,
    scene=dict(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False
    )
)

st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------
# 2D Slice‑by‑Slice View
# --------------------------------------------------
st.subheader("🖼️ 2D Slice‑by‑Slice MRI View")

num_slices = volume_3d.shape[2]
slice_idx = st.slider(
    "Select slice",
    0,
    num_slices - 1,
    num_slices // 2
)

mri_slice = volume_3d[:, :, slice_idx]
tumor_slice = tumor_mask[:, :, slice_idx]

fig2d, ax = plt.subplots(figsize=(5, 5))

ax.imshow(mri_slice.T, cmap="gray", origin="lower")
ax.imshow(
    np.ma.masked_where(tumor_slice.T == 0, tumor_slice.T),
    cmap="Reds",
    alpha=0.6,
    origin="lower"
)

ax.set_title(f"MRI Slice {slice_idx}")
ax.axis("off")

st.pyplot(fig2d)

# --------------------------------------------------
# Instructions
# --------------------------------------------------
st.markdown("""
### 🧭 How to use
- **3D View:** rotate, zoom, inspect tumor location
- **2D View:** scroll slices like a radiologist
- **Overlay:** tumor shown in red on brain anatomy

This is an **ML‑powered MRI tumor analysis MVP**.
""")
