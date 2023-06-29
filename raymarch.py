from jax import numpy as np
from jax import lax, vmap, grad, jit
from functools import partial
from utils.linalg import norm, normalize, softmax, relu, smoothmin, Rxyz

# typing
from typing import Callable, Tuple, Dict, NamedTuple
from jaxtyping import Array, Float32, UInt8

Vec3 = Float32[Array, '3']
Vec3s = Float32[Array, 'n 3']
Scalar = Float32[Array, '']
Scalars = Float32[Array, 'n']
UInts = UInt8[Array, 'n']


def sdf_sphere(p: Vec3, r: Vec3) -> Scalar:
    return norm(p) - r[0]


def sdf_plane(p: Vec3, n: Vec3) -> Scalar:
    return np.sum(p * normalize(n))


def sdf_box(p: Vec3, b: Vec3) -> Scalar:
    q = np.abs(p) - b
    return norm(relu(q)) + np.minimum(np.max(q), 0)


def sdf_torus(p: Vec3, t: Vec3) -> Scalar:
    q = np.array([norm(p[:2]) - t[0], p[2]])
    return norm(q) - t[1]


OBJECT_IDX = {
    'Box': 0,
    'Sphere': 1,
    'Plane': 2,
    'Torus': 3,
}
BRANCHES = [sdf_box, sdf_sphere, sdf_plane, sdf_torus]


class Camera(NamedTuple):
    up: Vec3
    position: Vec3
    target: Vec3
    f: Scalar

    def __call__(self, view_size: Tuple[int, int]) -> Vec3s:
        forward = self.target - self.position
        right = np.cross(forward, self.up)
        down = np.cross(right, forward)
        R = normalize(np.vstack([right, down, forward]))
        h, w = view_size
        fx, fy = self.f, self.f * h / w
        x = np.linspace(-fx, fx, w)
        y = np.linspace(fy, -fy, h)
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        rays = np.stack((x, y, np.ones(w * h)), axis=-1)
        return normalize(rays) @ R


class Scene(NamedTuple):
    objects: UInts
    positions: Vec3s
    attributes: Vec3s
    rotations: Vec3s
    colors: Vec3s
    roundings: Scalars
    smoothing: Scalar
    camera: Camera

    def sdfs(self, p: Vec3) -> Scalars:
        def switch(p: Vec3, obj_idx: UInt8, pos: Vec3, attr: Vec3, rot: Vec3):
            return lax.switch(obj_idx, BRANCHES, (p - pos) @ Rxyz(rot), attr)

        dists = vmap(partial(switch, p))(
            self.objects,
            self.positions,
            self.attributes,
            self.rotations,
        )
        return dists - self.roundings

    def sdf(self, p: Vec3) -> Scalar:
        return smoothmin(self.sdfs(p), self.smoothing)

    def color(self, p: Vec3) -> Scalar:
        return softmax(-self.sdfs(p) / self.smoothing) @ self.colors


def raymarch(sdf: Callable, p0: Vec3, dir: Vec3, n_steps: int = 50) -> Vec3:
    def march_step(_, p):
        return p + sdf(p) * dir

    return lax.fori_loop(0, n_steps, march_step, p0)


def cast_shadow(
    sdf: Callable, light_dir: Vec3, p0: Vec3, n_steps: int = 50, hardness: float = 4.0
) -> Scalar:
    def shade_step(_, carry):
        t, shadow = carry
        h = sdf(p0 + light_dir * t)
        return t + h, np.clip(hardness * h / t, 0.0, shadow)

    _, shadow = lax.fori_loop(0, n_steps, shade_step, (1e-2, 1.0))
    return shadow


def shade_f(
    surface_color: Vec3,
    raw_normal: Vec3,
    ray_dir: Vec3,
    shadow: Vec3,
    light_dir: Vec3,
) -> Scalar:
    ambient = norm(raw_normal)
    normal = raw_normal / ambient
    diffuse = light_dir.dot(normal).clip(0.0) * shadow
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
    light_dir: Vec3 = LIGHT_DIR,
) -> Dict[str, Array]:
    h, w = view_size
    i, j = click
    ray_dir = scene.camera(view_size)

    hit_pos = vmap(partial(raymarch, scene.sdf, scene.camera.position))(ray_dir)
    surface_color = vmap(scene.color)(hit_pos)
    raw_normal = vmap(grad(scene.sdf))(hit_pos)

    light_dir = np.where(i == -1, light_dir, raw_normal[i * w + j])
    light_dir = normalize(light_dir)
    shadow = vmap(partial(cast_shadow, scene.sdf, light_dir))(hit_pos)
    image = vmap(partial(shade_f, light_dir=light_dir))(surface_color, raw_normal, ray_dir, shadow)

    image = image ** (1.0 / 2.2)  # gamma correction

    distances = norm(hit_pos - scene.camera.position, axis=1)

    return {
        'image': image.reshape((h, w, 3)),
        'normal': normalize(raw_normal).reshape((h, w, 3)),
        'coordinate': (hit_pos % 1.0).reshape((h, w, 3)),
        'distance': (distances / distances.max()).reshape((h, w)),
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
