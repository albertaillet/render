from jax import numpy as np
from jax import lax, vmap, grad, jit
from functools import partial
from utils.linalg import norm, normalize, softmax, relu, min, smoothmin

# typing
from typing import Callable, Tuple, Dict, NamedTuple
from jaxtyping import Array, Float, Int8

Vec3 = Float[Array, '3']
Vec3s = Float[Array, 'n 3']
Scalar = Float[Array, '']


def sdf_sphere(position: Vec3, radius: Vec3, p: Vec3) -> Scalar:
    r, *_ = radius
    return norm(p - position) - r


def sdf_plane(position: Vec3, normal: Vec3, p: Vec3) -> Scalar:
    n = normalize(normal)
    return np.sum((p - position) * n)


def sdf_box(position: Vec3, size: Vec3, p: Vec3) -> Scalar:
    q = np.abs(p - position) - size
    return norm(relu(q)) + min(relu(q))


OBJECT_IDX = {
    'Box': 0,
    'Sphere': 1,
    'Plane': 2,
}
BRANCHES = [sdf_box, sdf_sphere, sdf_plane]


class Camera(NamedTuple):
    up: Vec3
    position: Vec3
    target: Vec3

    def rays(self, view_size: Tuple[int, int], fx: float = 0.6) -> Vec3s:
        forward = self.target - self.position
        right = np.cross(forward, self.up)
        down = np.cross(right, forward)
        R = normalize(np.vstack([right, down, forward]))
        h, w = view_size
        fy = fx / w * h
        x = np.linspace(-fx, fx, w)
        y = np.linspace(fy, -fy, h)
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        rays = np.stack((x, y, np.ones(w * h)), axis=-1)
        return normalize(rays) @ R


class Scene(NamedTuple):
    objects: Int8
    positions: Vec3s
    objattr: Vec3
    colors: Vec3s
    camera: Camera

    def sdf(self, p: Array) -> Array:
        def switch(obj_idx, pos, attr):
            return lax.switch(obj_idx, BRANCHES, pos, attr, p)

        return vmap(switch)(self.objects, self.positions, self.objattr)

    def color(self, p: Array) -> Array:
        dists = self.sdf(p)
        color = softmax(-8.0 * dists) @ self.colors
        return color


def raymarch(sdf: Callable, p0: Array, dir: Array, n_steps: int = 50) -> Array:
    def march_step(_, p):
        return p + sdf(p) * dir

    return lax.fori_loop(0, n_steps, march_step, p0)


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


LIGHT_DIR = normalize(np.array([1.5, 1.0, 0.2]))


@partial(jit, static_argnames=('view_size'))
def render_scene(
    scene: Scene,
    view_size: Tuple[int, int],
    click: Tuple[int, int],
    light_dir: Array = LIGHT_DIR,
) -> Dict[str, Array]:
    h, w = view_size
    i, j = click
    ray_dir = scene.camera.rays(view_size)

    def sdf(p: Array) -> Array:
        return smoothmin(scene.sdf(p), axis=0)

    hit_pos = vmap(partial(raymarch, sdf, scene.camera.position))(ray_dir)
    surface_color = vmap(scene.color)(hit_pos)
    raw_normal = vmap(grad(sdf))(hit_pos)

    light_dir = np.where(i == -1, light_dir, raw_normal[i * w + j])
    light_dir = normalize(light_dir)
    shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
    image = vmap(partial(shade_f, light_dir=light_dir))(surface_color, raw_normal, ray_dir, shadow)

    image = image ** (1.0 / 2.2)  # gamma correction

    distances = norm(hit_pos - scene.camera.position, axis=1)

    return {
        'image': image.reshape((h, w, 3)),
        'raw_normal': raw_normal.reshape((h, w, 3)),
        'distance': distances.reshape((h, w)),
    }


if __name__ == '__main__':
    from builder import build_scene
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)

    scene, view_size = build_scene(scene_dict)

    out = render_scene(scene, view_size, (-1, -1))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(out['image'].clip(0.0, 1.0))
    ax[1].imshow(out['raw_normal'].clip(0.0, 1.0))
    ax[2].imshow(out['distance'])
    for a in ax:
        a.axis('off')
    plt.show()
