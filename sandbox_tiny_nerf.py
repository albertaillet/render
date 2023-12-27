# %%
import numpy as np
from plotly import graph_objects as go


# %% tiny nerf: https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
data = np.load('assets/other/tiny_nerf_data.npz')
print(*[f'{k}: {data[k].shape},' for k in data.keys()])


# %%
def get_unit_rays_linspace(h, w, focal):
    bx = 0.5 * w / focal
    by = 0.5 * h / focal
    x = np.linspace(bx, -bx, w)
    y = np.linspace(-by, by, h)
    x, y = np.meshgrid(x, y, indexing="xy")
    x, y = x.flatten(), y.flatten()
    unit_rays = np.stack((x, y, -np.ones_like(x)), axis=-1)
    unit_rays = unit_rays / np.linalg.norm(unit_rays, axis=-1, keepdims=True)
    return unit_rays


def get_unit_rays_arange(h, w, focal):
    # Create a 2D rectangular grid for the rays corresponding to image dimensions
    x = np.arange(w)
    y = np.arange(h)
    i, j = np.meshgrid(x, y, indexing="xy")
    i, j = i.flatten(), j.flatten()
    transformed_i = (i - w * 0.5) / focal  # Normalize the x-axis coordinates
    transformed_j = -(j - h * 0.5) / focal  # Normalize the y-axis coordinates
    k = -np.ones_like(i)  # z-axis coordinates
    # Create the unit vectors corresponding to ray directions
    unit_rays = np.stack([transformed_i, transformed_j, k], axis=-1)
    unit_rays = unit_rays / np.linalg.norm(unit_rays, axis=-1, keepdims=True)
    return unit_rays


def get_rays(unit_rays, poses):
    # unit_rays is assumed to be normalized already
    rays = np.einsum("il,nkl->nik", unit_rays, poses[:, :3, :3])
    return rays


def get_pose_fig(ray_origins, ray_directions):
    empty = [None] * 3
    n_poses = len(ray_origins)

    traces = []
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers"))
    for i in range(n_poses):
        coords = []
        ray_origin = ray_origins[i]

        for j in range(ray_directions.shape[1]):
            coords.extend([ray_origin, ray_origin + ray_directions[i, j], empty])

        x_coords = [x for x, _, _ in coords]
        y_coords = [y for _, y, _ in coords]
        z_coords = [z for _, _, z in coords]
        traces.append(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode="lines"))

    return go.Figure(traces)


h, w = 100, 100
poses = data["poses"][:10]
ray_origins = poses[:, :3, -1]
ray_directions = get_rays(get_unit_rays_arange(h, w, data["focal"]), poses)
get_pose_fig(ray_origins, ray_directions).show()
ray_directions = get_rays(get_unit_rays_linspace(h, w, data["focal"]), poses)
get_pose_fig(ray_origins, ray_directions).show()

# %%
