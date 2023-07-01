from jax import vmap, grad, jit, lax, numpy as np
from functools import partial
from utils.linalg import norm, normalize, softmax, relu, smoothmin, Rxyz

# typing
from typing import Callable, Tuple, Dict, NamedTuple
from jaxtyping import Array, Float32, UInt8, Bool

Vec3 = Float32[Array, '3']
Vec3s = Float32[Array, 'n 3']
Bool3 = Bool[Array, '3']
Bool3s = Bool[Array, 'n 3']
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
BRANCHES = (sdf_box, sdf_sphere, sdf_plane, sdf_torus)


class Camera(NamedTuple):
    up: Vec3
    position: Vec3
    target: Vec3
    fov: Scalar

    def __call__(self, view_size: Tuple[int, int]) -> Vec3s:
        forward = self.target - self.position
        right = np.cross(forward, self.up)
        down = np.cross(right, forward)
        R = normalize(np.vstack([right, down, forward]))
        h, w = view_size
        fx, fy = self.fov, self.fov * h / w
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
    mirrorings: Bool3s
    roundings: Scalars
    smoothing: Scalar

    def sdfs(self, p: Vec3) -> Scalars:
        def switch(
            p: Vec3, obj_idx: UInt8, pos: Vec3, attr: Vec3, rot: Vec3, mirror: Bool3
        ) -> Scalar:
            p = np.where(mirror, np.abs(p), p)
            p = (p - pos) @ Rxyz(rot)
            return lax.switch(obj_idx, BRANCHES, p, attr)

        dists = vmap(partial(switch, p))(
            self.objects,
            self.positions,
            self.attributes,
            self.rotations,
            self.mirrorings,
        )
        return dists - self.roundings

    def sdf(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return lax.cond(
            self.smoothing > 0,
            lambda: smoothmin(dists, self.smoothing),
            lambda: np.min(dists),
        )

    def color(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return lax.cond(
            self.smoothing > 0,
            lambda: softmax(-dists / self.smoothing) @ self.colors,
            lambda: self.colors[np.argmin(dists)],
        )


def raymarch(sdf: Callable, p0: Vec3, dir: Vec3, n_steps: int = 50) -> Vec3:
    def march_step(_, p):
        return p + sdf(p) * dir

    return lax.fori_loop(0, n_steps, march_step, p0)


def shade(sdf: Callable, light_dir: Vec3, p0: Vec3, n_steps: int = 50, k: float = 4.0) -> Scalar:
    def shade_step(_, carry):
        res, t = carry
        h = sdf(p0 + light_dir * t)
        return np.clip(k * h / t, 0.0, res), t + h

    return lax.fori_loop(0, n_steps, shade_step, (1.0, 1e-2))[0]


LIGHT_DIR = np.array([0, 0, 1])
IMAGE_NAMES = (
    'image',
    'normal',
    'color',
    'shadow',
    'diffuse',
    'ambient',
    'specularity',
    'coordinate',
    'depth',
)


@partial(jit, static_argnames=('view_size'))
def render_scene(
    scene: Scene,
    camera: Camera,
    view_size: Tuple[int, int],
    click: Tuple[int, int],
) -> Dict[str, Array]:
    h, w = view_size
    i, j = click
    rays = camera(view_size)

    hits = vmap(partial(raymarch, scene.sdf, camera.position))(rays)
    color = vmap(scene.color)(hits)
    raw_normal = vmap(grad(scene.sdf))(hits)

    ambient = norm(raw_normal, keepdims=True)
    normal: Array = raw_normal / ambient

    light_dir = np.where(i == -1, LIGHT_DIR, normal[i * w + j])
    shadow = vmap(partial(shade, scene.sdf, light_dir))(hits).reshape(-1, 1)

    diffuse = normal.dot(light_dir).clip(0.0).reshape(-1, 1)
    specularity = (normal * normalize(light_dir - rays)).sum(axis=1, keepdims=True).clip(0.0) ** 200

    light = 0.8 * shadow * diffuse + 0.2 * ambient
    image = color * light + 0.3 * shadow * specularity
    image = image ** (1.0 / 2.2)  # gamma correction

    depth = norm(hits - camera.position)

    return {
        'image': image.reshape(h, w, 3),
        'normal': (0.5 * normal + 0.5).reshape(h, w, 3),
        'coordinate': (hits % 1).reshape(h, w, 3),
        'shadow': shadow.reshape(h, w),
        'depth': (depth / depth.max()).reshape(h, w),
        'specularity': specularity.reshape(h, w),
        'diffuse': diffuse.reshape(h, w),
        'ambient': ambient.reshape(h, w),
        'color': color.reshape(h, w, 3),
    }


if __name__ == '__main__':
    from sys import argv
    from builder import build_scene
    from matplotlib import pyplot as plt
    from utils.plot import load_yaml, to_rgb

    plt.style.use('grayscale')
    file = argv[1] if len(argv) > 1 else 'scene'

    out = render_scene(**build_scene(load_yaml(f'scenes/{file}.yml')), click=(-1, -1))

    rows = 2
    cols = len(IMAGE_NAMES) // rows + (len(IMAGE_NAMES) % rows > 0)
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, name in zip(axs.flatten(), IMAGE_NAMES):
        ax.imshow(to_rgb(out[name]))
        ax.set_title(name.capitalize())
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()
