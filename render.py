from jax import numpy as np
from jax import lax, nn, vmap, grad, jit
from functools import partial

# typing
from typing import NamedTuple, Callable
from jax import Array


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


def create_spheres(n: int = 16) -> Spheres:
    pos = np.stack([np.zeros(n), np.arange(n), np.ones(n)], axis=1)
    radii = 0.5 * np.ones(n)  # radius of each sphere
    return Spheres(pos, radii)  # returns a spheres named tuple


def create_planes() -> Planes:
    pos = np.zeros((1, 3))  # all planes pass through origin
    normal = np.array([[0, 1, 0]])  # each direction
    return Planes(pos, normal)


def scene_sdf(
    spheres: Spheres, planes: Planes, p: Array, c: float = 8.0, union='softmin'
) -> Array:
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
    x = np.linspace(-fx, fx, w)
    y = np.linspace(-fy, fy, h)
    x, y = np.meshgrid(x, y, indexing='ij')
    x, y = x.flatten(), y.flatten()
    rays = np.stack([x, y, np.ones(w * h)], axis=-1)
    return normalize(rays) @ R


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


WORLD_UP = np.array([0.0, 1.0, 0.0])
CAMERA_POS = np.array([3.0, 5.0, 3.0])
LIGHT_DIR = normalize(np.array([1.5, 1.0, 0.2]))
spheres = create_spheres(n=9)
planes = create_planes()
sdf = jit(partial(scene_sdf, spheres, planes, c=8.0, union='softmin'))


def render_scene(w: int, h: int, x0: int, y0: int) -> Array:
    w, h = int(w), int(h)
    ray_dir = camera_rays(-CAMERA_POS, view_size=(w, h))
    hit_pos = vmap(partial(raymarch, sdf, CAMERA_POS))(ray_dir)
    raw_normal = vmap(grad(sdf))(hit_pos)

    if x0 != -1:
        light_dir = normalize(raw_normal[x0 * h + y0])
    else:
        light_dir = LIGHT_DIR
    shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
    frame = vmap(partial(shade_f, light_dir=light_dir))(raw_normal, ray_dir, shadow)
    frame = frame ** (1.0 / 2.2)  # gamma correction
    return frame.reshape(w, h).T
