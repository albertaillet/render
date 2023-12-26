# %%
import numpy as np
from numpy import load

# from jax import numpy as np
from plotly import graph_objects as go

# from matplotlib import pyplot as plt

# %% tiny nerf: https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
data = load('assets/other/tiny_nerf_data.npz')
print(data['images'].shape)
print(data['poses'].shape)


# %%
def get_unit_rays_linspace(h, w, focal):
    f = 100 / focal
    x = np.linspace(-f, f, w)
    y = np.linspace(f, -f, h)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    unit_rays = np.stack((x, y, np.ones(w * h)), axis=-1)
    unit_rays = unit_rays / np.linalg.norm(unit_rays, axis=-1, keepdims=True)
    return unit_rays


def get_unit_rays_arange(h, w, focal):
    # Create a 2D rectangular grid for the rays corresponding to image dimensions
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    i, j = i.flatten(), j.flatten()
    transformed_i = (i - w * 0.5) / focal  # Normalize the x-axis coordinates
    transformed_j = -(j - h * 0.5) / focal  # Normalize the y-axis coordinates
    k = -np.ones_like(i)  # z-axis coordinates
    # Create the unit vectors corresponding to ray directions
    unit_rays = np.stack([transformed_i, transformed_j, k], axis=-1)
    unit_rays = unit_rays / np.linalg.norm(unit_rays, axis=-1, keepdims=True)
    return unit_rays


def get_rays(unit_rays, pose):
    # unit_rays is assumed to be normalized already
    rays = np.einsum("il,kl", unit_rays, pose[:3, :3])
    return rays


h, w = 30, 30
unit_rays = get_unit_rays_linspace(h, w, data["focal"])
empty = [None] * 3
traces = []
traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers"))
poses = data["poses"][:20]

for pose in poses:
    coords = []
    rays = get_rays(unit_rays, pose)
    ray_origin = pose[:3, -1]
    for j in range(h * w):
        coords.extend([ray_origin, rays[j], empty])

    x_coords = [x for x, _, _ in coords]
    y_coords = [y for _, y, _ in coords]
    z_coords = [z for _, _, z in coords]
    traces.append(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode="lines"))

go.Figure(traces)
# %%
