# %%
from jax import jit, grad, value_and_grad, vmap, numpy as np, random
from jax.nn import sigmoid
from optax import OptState, apply_updates, adam
from utils.mlp import init_mlp_params, MLP
from utils.linalg import normalize, norm, smoothmin, softmax, Rxyz
from utils.plot import to_rgb, fromarray
from raymarch import Vec3, Vec3s, Scalar, Scalars, sdf_box, RenderedImages, raymarch, shade
from functools import partial
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from numpy import load, hstack

# typing
from typing import Tuple, Sequence, NamedTuple
from jax import Array

VIEW_SIZE = (100, 100)


def imrow(*images: Sequence[Array]) -> Image.Image:
    return fromarray(hstack([to_rgb(image) for image in images]))  # type: ignore


class DifferentiableObjects(NamedTuple):
    positions: Vec3s
    sides: Vec3s
    rotations: Vec3s
    colors: Vec3s
    roundings: Scalars
    smoothing: Scalar

    def sdfs(self, p: Vec3) -> Scalars:
        p = p - self.positions
        p = vmap(lambda p, r: p @ Rxyz(r))(p, self.rotations)
        dists = sdf_box(p, self.sides)
        return dists - self.roundings

    def sdf(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return smoothmin(dists, self.smoothing)

    def color(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return softmax(-dists / self.smoothing) @ self.colors


# %% tiny nerf: https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
data = load('assets/other/tiny_nerf_data.npz')
print(*[f'{k}: {data[k].shape},' for k in data.keys()])  # noqa
imrow(*list(data['images'][:5]))


# %%
def create_objects(seed: Array, n: int) -> DifferentiableObjects:
    p_key, s_key, r_key, c_key = random.split(seed, 4)
    shape = (n, 3)
    positions = 0.1 * random.normal(p_key, shape)
    sides = 0.05 * random.uniform(s_key, shape)
    rotations = random.uniform(r_key, shape)
    colors = random.uniform(c_key, shape)
    roundings = 0.0 * random.uniform(r_key, (n,))
    smoothing = np.array(0.01)
    return DifferentiableObjects(
        positions=positions,
        sides=sides,
        rotations=rotations,
        colors=colors,
        roundings=roundings,
        smoothing=smoothing,
    )


objects = create_objects(random.PRNGKey(0), 20)


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


def get_rays(unit_rays, poses):
    # unit_rays is assumed to be normalized already
    rays = np.einsum("il,nkl", unit_rays, poses[:, :3, :3])
    return rays


@partial(jit, static_argnames=('view_size'))
def render_scene(
    objects: DifferentiableObjects,
    rays: Vec3s,
    view_size: Tuple[int, int],
    light_dir: Vec3,
) -> RenderedImages:
    h, w = view_size

    hits = vmap(partial(raymarch, objects.sdf, camera.position, n_steps=10))(rays)
    color = vmap(objects.color)(hits)
    raw_normal = vmap(grad(objects.sdf))(hits)

    ambient = norm(raw_normal, keepdims=True)
    normal = raw_normal / ambient

    shadow = vmap(partial(shade, objects.sdf, light_dir))(hits).reshape(-1, 1)

    diffuse = normal.dot(light_dir).clip(0.0).reshape(-1, 1)
    specularity = (normal * normalize(light_dir - rays)).sum(axis=1, keepdims=True).clip(0.0) ** 200

    light = 0.8 * shadow * diffuse + 0.2 * ambient
    image = color * light + 0.3 * shadow * specularity
    image = image ** (1.0 / 2.2)  # gamma correction

    depth = norm(hits - camera.position)

    return RenderedImages(
        image=image.reshape(h, w, 3),
        normal=(0.5 * normal + 0.5).reshape(h, w, 3),
        color=color.reshape(h, w, 3),
        coordinate=(hits % 1).reshape(h, w, 3),
        shadow=shadow.reshape(h, w),
        diffuse=diffuse.reshape(h, w),
        ambient=ambient.reshape(h, w),
        specularity=specularity.reshape(h, w),
        depth=(depth / depth.max()).reshape(h, w),
    )


# %%
def get_depth_contour(depth: Array, camera: Camera) -> Array:
    rays = camera.rays(view_size=depth.shape).reshape(*depth.shape, 3)
    assert np.allclose(norm(rays, axis=-1, keepdims=True), 1.0)
    depth = np.where(depth == np.inf, np.nan, depth)
    points = rays * depth[:, :, None]
    return points + camera.position


def show_slice(sdf, z=0.0, w=400, r=3.5, y_c=0.0, points=None):
    y, x = np.mgrid[-r : r : w * 1j, -r : r : w * 1j].reshape(2, -1)
    y = y + y_c
    z = z * np.ones_like(x)
    p = np.c_[x, y, z]
    d = vmap(sdf)(p).reshape(w, w)
    plt.figure(figsize=(5, 5))
    kw = dict(extent=(-r, r, -r + y_c, r + y_c), vmin=-r, vmax=r)
    plt.contourf(d, 16, cmap='bwr', **kw)
    plt.contour(d, levels=[0.0], colors='black', **kw)
    if points is not None:
        plt.plot(points[:, 50, 0], points[:, 50, 1], 'k')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


depth_points = get_depth_contour(depth_gt, camera)
show_slice(learnt_scene.sdf, z=0.0, w=400, r=1.5, y_c=0.5, points=depth_points)


# %%
@jit
def loss_fn(
    scene: Scene,
    camera: Camera,
) -> Tuple[Array, Array]:
    # render scene
    image, depth = render_scene(scene, camera)
    # compute l1 loss
    img_loss = np.where(foreground[:, :, None], np.abs(image - img_gt), 0.0).mean()
    # depth_loss = np.where(foreground, np.abs(depth - depth_gt), 0.0).mean()
    loss = img_loss  # + depth_loss
    return loss, (image, depth)


@jit
def update_fn(
    scene: Scene,
    camera: Camera,
    opt_state: OptState,
) -> Tuple[Scene, Array, Array, OptState]:
    (loss, (learnt_img, learnt_depth)), grads = value_and_grad(loss_fn, has_aux=True)(scene, camera)
    updates, opt_state = opt_update(grads, opt_state)
    scene = apply_updates(scene, updates)
    return scene, loss, learnt_img, learnt_depth, opt_state


# %%
lr = 0.01
opt_init, opt_update = adam(lr)
opt_state = opt_init(learnt_scene)
frames = []
pbar = tqdm(range(10_000))

# %%
for i in pbar:
    learnt_scene, loss, learnt_img, learnt_depth, opt_state = update_fn(
        learnt_scene, camera, opt_state
    )
    pbar.set_description(f'loss: {loss:.4f}')
    frames.append(learnt_img)
    if i % 10 == 0:
        imshow(learnt_img, learnt_depth)
        show_slice(learnt_scene.sdf, z=0.0, w=400, r=1.5, y_c=0.5)

# %%
imgs = [to_pil_image(img) for img in frames]
imgs[0].save(
    'assets/other/snowman.gif',
    save_all=True,
    append_images=imgs[1:],
    duration=100,
    loop=0,
)

# %%
