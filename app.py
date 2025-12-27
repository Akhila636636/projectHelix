import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 Project Helix – 3D Tumor Visualization")

mesh = trimesh.load("tumor.glb")
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.dump())

# Measurements
volume = mesh.volume
diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

st.metric("Tumor Volume", f"{volume:.2f}")
st.metric("Approx Diameter", f"{diameter:.2f}")

# Peeling slider
z_min, z_max = mesh.vertices[:,2].min(), mesh.vertices[:,2].max()
z_cut = st.slider("Peel tumor", float(z_min), float(z_max), float(z_max))

verts = mesh.vertices
faces = mesh.faces

mask = verts[:,2] <= z_cut
index_map = -np.ones(len(verts), dtype=int)
index_map[mask] = np.arange(mask.sum())

face_mask = np.all(mask[faces], axis=1)
new_faces = index_map[faces[face_mask]]
new_verts = verts[mask]

fig = go.Figure(data=[
    go.Mesh3d(
        x=new_verts[:,0],
        y=new_verts[:,1],
        z=new_verts[:,2],
        i=new_faces[:,0],
        j=new_faces[:,1],
        k=new_faces[:,2],
        color="red",
        opacity=0.7
    )
])

fig.update_layout(
    scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
    height=650
)

st.plotly_chart(fig, use_container_width=True)
