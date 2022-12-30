# %%
# Import
from jax import numpy as np
from jax import lax, nn, vmap, grad, jit
from jax.random import PRNGKey, uniform
from functools import partial
from plotly import express as px


# typing
from typing import NamedTuple, Callable
from jax import Array
from jax.random import PRNGKeyArray


def norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Array:
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x: Array, axis: int = -1, eps: float = 1e-20) -> Array:
    return x / norm(x, axis=axis, keepdims=True, eps=eps)


class Spheres(NamedTuple):
    pos: Array
    radii: Array

    def sdf(self, p: Array) -> Array:
        return norm(p - self.pos) - self.radii


class Planes(NamedTuple):
    pos: Array
    normal: Array

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.pos, self.normal)


def create_spheres(*, key: PRNGKeyArray, n: int = 16, s: float = 3.0):
    pos = s * uniform(key, (n, 3))  # position of the sphere
    radii = uniform(key, (n,))  # radius of each sphere
    return Spheres(pos, radii)  # returns a spheres named tuple


def create_planes():
    pos = np.zeros((3, 3))  # all planes pass through origin
    normal = normalize(np.eye(3))  # each direction
    return Planes(pos, normal)


def scene_sdf(
    spheres: Spheres, planes: Planes, p: Array, c: float = 8.0, union='softmin'
):
    sphere_dists = spheres.sdf(p)
    plane_dists = planes.sdf(p)
    dists = np.concatenate([sphere_dists, plane_dists])
    if union == 'min':
        # distance to the closest object using min
        dist = dists.min()
    elif union == 'softmin':
        # distance using softmin
        dist = -nn.logsumexp(-c * dists) / c
    else:
        raise ValueError('union must be either min or softmin')
    return dist


def raymarch(sdf: Callable, p0: Array, dir: Array, n_steps: int = 50) -> Array:
    def march_step(_, p):
        return p + sdf(p) * dir

    return lax.fori_loop(0, n_steps, march_step, p0)


def camera_rays(forward: Array, view_size: tuple[int, int], fx: float = 0.6) -> Array:
    right = np.cross(forward, WORLD_UP)
    down = np.cross(right, forward)
    R = normalize(np.vstack([right, down, forward]))
    w, h = view_size
    fy = fx / w * h
    y, x = np.mgrid[fy : -fy : h * 1j, -fx : fx : w * 1j].reshape(2, -1)
    return normalize(np.c_[x, y, np.ones_like(x)]) @ R


def cast_shadow(
    sdf: Callable, light_dir: Array, p0: Array, n_steps: int = 50, hardness: float = 4.0
) -> Array:
    def shade_step(_, carry):
        t, shadow = carry
        h = sdf(p0 + light_dir * t)
        return t + h, np.clip(hardness * h / t, 0.0, shadow)

    _, shadow = lax.fori_loop(0, n_steps, shade_step, (1e-2, 1.0))
    return shadow


def shade_f(
    raw_normal: Array, ray_dir: Array, shadow: Array, light_dir: Array
) -> Array:
    ambient = norm(raw_normal)
    normal = raw_normal / ambient
    diffuse = normal.dot(light_dir).clip(0.0) * shadow
    half = normalize(light_dir - ray_dir)
    spec = 0.3 * shadow * half.dot(normal).clip(0.0) ** 200.0
    light = 0.8 * diffuse + 0.2 * ambient
    return light + spec


# %%
SPHERES_KEY = PRNGKey(0)
WORLD_UP = np.array([0.0, 1.0, 0.0])
CAMERA_POS = np.array([3.0, 5.0, 3.0])
LIGHT_DIR = normalize(np.array([1.5, 1.0, 0.2]))

w, h = 480, 320

spheres = create_spheres(key=SPHERES_KEY, n=6, s=2.7)
planes = create_planes()
ray_dir = camera_rays(-CAMERA_POS, view_size=(w, h))
sdf = jit(partial(scene_sdf, spheres, planes, c=8.0, union='softmin'))

# %%
hit_pos = vmap(partial(raymarch, sdf, CAMERA_POS))(ray_dir)
raw_normal = vmap(grad(sdf))(hit_pos)
shadow = vmap(partial(cast_shadow, sdf, LIGHT_DIR))(hit_pos)
frame = vmap(partial(shade_f, light_dir=LIGHT_DIR))(raw_normal, ray_dir, shadow)
frame = frame ** (1.0 / 2.2)  # gamma correction

# %%
fig = px.imshow(frame.reshape(h, w), color_continuous_scale='gray', zmin=0.0, zmax=1.0)
fig.update_layout(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=0, b=0)
)

# %%
