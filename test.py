# %%
import matplotlib.pyplot as plt
from jax import jit, grad, value_and_grad, vmap, numpy as np, random
from jax.nn import sigmoid
from utils.mlp import init_mlp_params, MLP
from utils.linalg import normalize, norm, min, smoothmin, softmax
from raymarch import (
    Planes,
    Camera,
    Scene,
    raymarch,
    cast_shadow,
    shade_f,
    render_scene,
    LIGHT_DIR,
)
from functools import partial
from optax import adam, apply_updates

# typing
from typing import Tuple, Sequence
from jax import Array


camera = Camera(
    up=np.array([0.0, 1.0, 0.0]),
    position=np.array([3.0, 5.0, 3.0]),
    target=np.array([0.0, 0.0, 0.0]),
)

plane = Planes(
    position=np.zeros((3, 3)),
    normal=np.eye(3),
    color=np.eye(3) / 2.0 + 0.5,
)

scene = Scene(objects=(plane,), camera=camera)

# %%
@partial(jit, static_argnames=('view_size'))
def render_mlp(
    mlps: Tuple[MLP, MLP],
    camera: Camera,
    view_size: Tuple[int, int],
    light_dir: Array = LIGHT_DIR,
) -> Array:
    sdf_mlp, color_mlp = mlps
    w, h = view_size
    ray_dir = camera.rays(view_size)

    def sdf(p: Array) -> Array:
        return sdf_mlp(p)[0]

    def colorf(p: Array) -> Array:
        return color_mlp(p)

    hit_pos = vmap(partial(raymarch, sdf, camera.position))(ray_dir)
    surface_color = vmap(colorf)(hit_pos)
    raw_normal = vmap(grad(sdf))(hit_pos)

    light_dir = normalize(light_dir)
    shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
    color = vmap(partial(shade_f, light_dir=light_dir))(
        surface_color, raw_normal, ray_dir, shadow
    )

    color = color ** (1.0 / 2.2)  # gamma correction
    out = color.reshape((w, h, 3)).transpose((1, 0, 2))
    return out


# %%
def batch_generator(key: Array, batch_size: int, minval: float, maxval: float):
    while True:
        key, subkey = random.split(key)
        yield random.uniform(subkey, (batch_size, 3), minval=minval, maxval=maxval)


def sdf_loss_fn(mlp: MLP, scene: Scene, p: Array) -> Array:
    mlp_dists = vmap(lambda p: mlp(p))(p)
    scene_dists = vmap(lambda p: smoothmin(scene.sdf(p)))(p)
    return np.mean(np.abs(mlp_dists - scene_dists))


def color_loss_fn(mlp: MLP, scene: Scene, p: Array) -> Array:
    mlp_colors = vmap(lambda p: sigmoid(mlp(p)))(p)
    scene_colors = vmap(scene.color)(p)
    return np.sqrt(np.mean(np.square(mlp_colors - scene_colors)))


# %%
sdf_params = init_mlp_params([3, 32, 32, 32, 32, 32, 1], key=random.PRNGKey(0))
color_params = init_mlp_params([3, 32, 32, 3], key=random.PRNGKey(1))
sdf_mlp = MLP(sdf_params)
color_mlp = MLP(color_params)

# %%
sdf_lr, color_lr = 1e-3, 1e-3

sdf_opt = adam(sdf_lr)
sdf_opt_state = sdf_opt.init(sdf_mlp)

color_opt = adam(color_lr)
color_opt_state = color_opt.init(color_mlp)

# %%
batch_size = 2**10
print(f'batch_size: {batch_size}')
for i, batch in enumerate(batch_generator(random.PRNGKey(0), batch_size, -20.0, 20.0)):

    sdf_loss, sdf_grads = value_and_grad(sdf_loss_fn)(sdf_mlp, scene, batch)
    sdf_updates, sdf_opt_state = sdf_opt.update(sdf_grads, sdf_opt_state)
    sdf_mlp = apply_updates(sdf_mlp, sdf_updates)

    color_loss, color_grads = value_and_grad(color_loss_fn)(color_mlp, scene, batch)
    color_updates, color_opt_state = color_opt.update(color_grads, color_opt_state)
    color_mlp = apply_updates(color_mlp, color_updates)

    print(f'[{i}] sdf_loss: {sdf_loss:.2f}, color_loss: {color_loss:.2f}', end='\r')
    if i == 100:
        break

# %%
# plot slice of the sdf
def get_slice(minval: float, maxval: float, n: int, s: float = 0.0):
    x = np.linspace(minval, maxval, n)
    y = np.linspace(minval, maxval, n)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * s
    p = np.stack([X, Y, Z], axis=-1).reshape((n * n, 3))
    mlp_dists = vmap(lambda p: sdf_mlp(p))(p)
    dists = vmap(lambda p: smoothmin(scene.sdf(p)))(p)
    return mlp_dists.reshape((n, n)), dists.reshape((n, n))


mlp_dists, dists = get_slice(-10.0, 10.0, 20, 10.0)
vmin = np.minimum(mlp_dists.min(), dists.min())
vmax = np.maximum(mlp_dists.max(), dists.max())
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(mlp_dists, vmin=vmin, vmax=vmax)
ax[1].imshow(dists, vmin=vmin, vmax=vmax)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(ax[0].imshow(mlp_dists, vmin=vmin, vmax=vmax), cax=cbar_ax)


# %%
view_size = (100, 100)
mlp_img = render_mlp((sdf_params, color_params), camera, view_size)
scene_img = render_scene(scene, view_size, (-1, 1))
plt.subplot(1, 2, 1)
plt.imshow(mlp_img[::-1])
plt.subplot(1, 2, 2)
plt.imshow(scene_img[::-1])
plt.show()
# %%
