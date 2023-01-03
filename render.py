from jax import numpy as np
from jax import lax, vmap, grad, jit
from functools import partial
from jax_utils import norm, normalize, min, softmin

# typing
from typing import Callable, Tuple
from jax import Array
from objects import Scene


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
    surface_color: Array,
    raw_normal: Array,
    ray_dir: Array,
    shadow: Array,
    light_dir: Array,
) -> Array:
    ambient = norm(raw_normal)
    normal = raw_normal / ambient
    diffuse = normal.dot(light_dir).clip(0.0) * shadow
    half = normalize(light_dir - ray_dir)
    spec = 0.3 * shadow * half.dot(normal).clip(0.0) ** 200.0
    light = 0.8 * diffuse + 0.2 * ambient
    return surface_color * light + spec


WORLD_UP = np.array([0.0, 1.0, 0.0])
CAMERA_POS = np.array([3.0, 5.0, 3.0])
TARGET_POS = np.array([0.0, 0.0, 0.0])
LIGHT_DIR = normalize(np.array([1.5, 1.0, 0.2]))


@partial(jit, static_argnames=('view_size'))
def render_scene(
    scene: Scene,
    view_size: Tuple[int, int],
    click: Array,
    light_dir: Array = LIGHT_DIR,
    camera_pos: Array = CAMERA_POS,
    target_pos: Array = TARGET_POS,
) -> Array:
    w, h = view_size
    i, j = click
    ray_dir = camera_rays(target_pos - camera_pos, view_size=view_size)

    def sdf(p: Array) -> Array:
        return softmin(scene.sdf(p))

    hit_pos = vmap(partial(raymarch, sdf, camera_pos))(ray_dir)
    surface_color = vmap(scene.color)(hit_pos)
    raw_normal = vmap(grad(sdf))(hit_pos)

    light_dir = np.where(i == -1, light_dir, raw_normal[i * h - j])
    light_dir = normalize(light_dir)
    shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
    color = vmap(partial(shade_f, light_dir=light_dir))(
        surface_color, raw_normal, ray_dir, shadow
    )

    color = color ** (1.0 / 2.2)  # gamma correction

    def to_rgb_image(img: Array) -> Array:
        img = np.uint8(255.0 * img.clip(0.0, 1.0))
        return img.reshape((w, h, 3)).transpose((1, 0, 2))[::-1]

    return to_rgb_image(color)
