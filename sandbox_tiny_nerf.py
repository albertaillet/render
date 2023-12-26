# %%
import numpy as np
from numpy import load

# from jax import numpy as np
# from plotly import graph_objects as go
from matplotlib import pyplot as plt

# %% tiny nerf: https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
data = load('assets/other/tiny_nerf_data.npz')
coord_names = 'xyz'


print(data['images'].shape)
print(data['poses'].shape)
# print(focal)

focal = 100 / data['focal']
h, w = 2, 2

x = np.linspace(-focal, focal, w)
y = np.linspace(focal, -focal, h)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
unit_rays = np.stack((x, y, np.ones(w * h)), axis=-1)
unit_rays = unit_rays / np.sqrt(np.square(unit_rays).sum(axis=-1, keepdims=True))

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for m in data['poses']:
    rot = m[:3, :3]
    pos = m[:3, -1]
    v = (unit_rays[:, None, :] * rot[None, :, :]).sum(axis=-1)
    dir = pos[None, 1] + v
    coord = lambda i, j: [pos[i % 3], dir[j, i % 3]]  # noqa: E731

    for j in range(h * w):
        for i in range(3):
            ax[i].plot(coord(j, i), coord(j, i + 1), label=f'{j}')
for i in range(3):
    title = coord_names[i] + coord_names[(i + 1) % 3]
    ax[i].set_title(title)
plt.show()

# %%
