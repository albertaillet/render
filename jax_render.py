# %%
# Import
from jax import numpy as np
from jax import lax, nn, vmap, grad
from jax.random import PRNGKey, uniform
from functools import partial
from plotly import express as px


# typing
from typing import NamedTuple, Callable, Sequence
from jax import Array
from jax.random import PRNGKeyArray


def norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Array:
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x: Array, axis: int = -1, eps: float = 1e-20) -> Array:
    return x / norm(x, axis=axis, keepdims=True, eps=eps)


class Spheres(NamedTuple):
    pos: Array
    color: Array
    radii: Array

    def sdf(self, p: Array):
        return norm(p - self.pos) - self.radii


def create_spheres(*, key: PRNGKeyArray, n: int = 16, R: float = 3.0):
    pos, color = uniform(
        key, (2, n, 3)
    )  # creates a 2D array of 3D vectors, one for each sphere
    # pos is position of the sphere, color is the color of the sphere
    radii = uniform(key, (n,))  # creates a 1D array of radii, one for each sphere
    pos = (pos - 0.5) * R  # centers the spheres around the origin
    return Spheres(pos, color, radii)  # returns a spheres named tuple


def scene_sdf(spheres: Spheres, p: Array, c: float = 8.0):
    dists = spheres.sdf(p)
    # distance to the closest sphere
    # spheres_dist = dists.min()
    # distance to the closest sphere using softmin
    spheres_dist = -nn.logsumexp(-c * dists) / c
    p_x, p_y, p_z = p
    floor_dist = p_y - FLOOR_Y  # distance to the floor
    return np.minimum(spheres_dist, floor_dist)  # distance to the closest object


def raymarch(sdf: Callable, p0: Array, dir: Array, n_steps: int = 50) -> Array:
    def march_step(_, p):
        return p + sdf(p) * dir

    return lax.fori_loop(0, n_steps, march_step, p0)


def camera_rays(forward: Array, view_size: Sequence[int], fx: float = 0.6) -> Array:
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
    return SURFACE_COLOR * light + spec


# %%
SPHERES_KEY = PRNGKey(123)
FLOOR_Y = -3.0
WORLD_UP = np.array([0.0, 1.0, 0.0])
SURFACE_COLOR = np.array([0.9, 0.9, 0.9])
CAMERA_POS = np.float32([3.0, 5.0, 4.0])
w, h = 480, 320

spheres = create_spheres(key=SPHERES_KEY)
ray_dir = camera_rays(-CAMERA_POS, view_size=(w, h))

# %%
sdf = partial(scene_sdf, spheres, c=8.0)
hit_pos = vmap(partial(raymarch, sdf, CAMERA_POS))(ray_dir)
raw_normal = vmap(grad(sdf))(hit_pos)
px.imshow(raw_normal.reshape(h, w, 3) % 1.0)

# %%
light_dir = normalize(np.array([1.1, 1.0, 0.2]))
shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
px.imshow(shadow.reshape(h, w), color_continuous_scale='gray')

# %%
frame = vmap(partial(shade_f, light_dir=light_dir))(raw_normal, ray_dir, shadow)
frame = frame ** (1.0 / 2.2) # gamma correction
px.imshow(frame.reshape(h, w, 3).clip(0, 1), color_continuous_scale='gray')

# %%
