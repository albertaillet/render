# %%
from jax import jit, grad, value_and_grad, vmap, numpy as np, random, lax
from jax.nn import softplus, relu
from optax import OptState, apply_updates, adam
from utils.linalg import normalize, norm, smoothmin, softmax, Rxyz
from utils.plot import to_rgb, fromarray
from IPython.display import display
from raymarch import Vec3, Vec3s, Scalar, Scalars, sdf_box, RenderedImages, raymarch, shade, Image3
from functools import partial
from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy import load, hstack

# typing
from typing import Tuple, Sequence, NamedTuple
from jax import Array

# %% tiny nerf: https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
data = load('assets/other/tiny_nerf_data.npz')
print(*[f'{k}: {data[k].shape},' for k in data.keys()])  # noqa
view_size = data['images'].shape[1:3]


def imrow(*images: Sequence[Array]):
    return fromarray(hstack([to_rgb(image) for image in images]))  # type: ignore


display(imrow(*list(data['images'][:5])))


# %%
class DifferentiableObjects(NamedTuple):
    positions: Vec3s
    sides: Vec3s
    rotations: Vec3s
    colors: Vec3s
    roundings: Scalars
    smoothing: Scalar
    outer: Scalar

    def sdfs(self, p: Vec3) -> Scalars:
        p = p - self.positions
        p = vmap(lambda p, r: p @ Rxyz(r))(p, self.rotations)
        dists = sdf_box(p, self.sides)
        return dists - relu(self.roundings)

    def sdf(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return smoothmin(dists, softplus(self.smoothing - 2))

    def color(self, p: Vec3) -> Scalar:
        dists = self.sdfs(p)
        return lax.cond(
            np.max(dists) > self.outer,
            lambda: np.zeros(3),
            lambda: softmax(-dists / softplus(self.smoothing - 2)) @ softplus(self.colors),
        )


def create_objects(seed: Array, n: int) -> DifferentiableObjects:
    p_key, s_key, r_key, c_key = random.split(seed, 4)
    shape = (n, 3)
    positions = 0.1 * random.normal(p_key, shape)
    sides = 0.1 * random.uniform(s_key, shape)
    rotations = random.uniform(r_key, shape)
    colors = random.uniform(c_key, shape)
    roundings = np.zeros(n)
    smoothing = np.array(0.0)
    outer = np.array(10.0)  # outer radius
    return DifferentiableObjects(
        positions=positions,
        sides=sides,
        rotations=rotations,
        colors=colors,
        roundings=roundings,
        smoothing=smoothing,
        outer=outer,
    )


def get_unit_rays_linspace(view_size: Tuple[int, int], focal: float) -> Vec3s:
    h, w = view_size
    bx = 0.5 * w / focal
    by = 0.5 * h / focal
    x = np.linspace(bx, -bx, w)
    y = np.linspace(by, -by, h)
    x, y = np.meshgrid(x, y, indexing="xy")
    x, y = x.flatten(), y.flatten()
    unit_rays = np.stack((x, y, -np.ones_like(x)), axis=-1)
    unit_rays = unit_rays / np.linalg.norm(unit_rays, axis=-1, keepdims=True)
    return unit_rays


def get_rays(unit_rays: Vec3s, poses: Array) -> Array:
    # unit_rays is assumed to be normalized already
    rays = np.einsum("il,nkl->nik", unit_rays, poses[:, :3, :3])
    return rays


@jit
def render(
    objects: DifferentiableObjects,
    position: Vec3,
    rays: Vec3s,
    light_dir: Vec3,
) -> RenderedImages:
    h, w = view_size

    hits = vmap(partial(raymarch, objects.sdf, position, n_steps=20))(rays)
    color = vmap(objects.color)(hits)
    raw_normal = vmap(grad(objects.sdf))(hits)

    ambient = norm(raw_normal, keepdims=True)
    normal = raw_normal / ambient

    # shadow = vmap(partial(shade, objects.sdf, light_dir))(hits).reshape(-1, 1)
    shadow = np.zeros(view_size)  # dummy
    diffuse = normal.dot(light_dir).clip(0.0).reshape(-1, 1)
    specularity = (normal * normalize(light_dir - rays)).sum(axis=1, keepdims=True).clip(0.0) ** 200

    # light = shadow * diffuse  # * 0.8 + 0.2 * ambient
    image = color  # * light  # + 0.3 * shadow * specularity
    # image = image ** (1.0 / 2.2)  # gamma correction

    depth = norm(hits - position)

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


@jit
def loss_fn(
    objects: DifferentiableObjects,
    position: Vec3,
    rays: Vec3s,
    gt_image: Image3,
) -> Tuple[Array, RenderedImages]:
    # render scene
    images = render(objects, position, rays, light_dir)
    # compute l1 loss
    img_loss = np.abs(images.image - gt_image).mean()
    # depth_loss = np.where(foreground, np.abs(depth - depth_gt), 0.0).mean()
    loss = img_loss  # + depth_loss
    return loss, images


@jit
def update_fn(
    objects: DifferentiableObjects,
    position: Vec3,
    rays: Vec3s,
    gt_image: Image3,
    opt_state: OptState,
) -> Tuple[DifferentiableObjects, Array, RenderedImages, OptState]:
    (loss, images), grads = value_and_grad(loss_fn, has_aux=True)(objects, position, rays, gt_image)
    updates, opt_state = opt_update(grads, opt_state)
    objects = apply_updates(objects, updates)  # type: ignore
    return objects, loss, images, opt_state


# %%
gt_images = np.array(data['images'])
unit_rays = get_unit_rays_linspace(view_size, focal=float(data['focal']))
rays = get_rays(unit_rays, data['poses'])
positions = data['poses'][:, :3, -1]
light_dir = normalize(np.array([0.0, 0.0, 1.0]))
objects = create_objects(random.PRNGKey(0), 40)
for i in range(3):
    images = render(objects, positions[i], rays[i], light_dir)
    display(imrow(*images))
print(*images._fields, sep=', ')  # type: ignore

# show_slice(objects.sdf, z=0.0, w=200, r=5, y_c=0.5)

# %%
lr = 0.001
opt_init, opt_update = adam(lr)
opt_state = opt_init(objects)
pbar = tqdm(range(100))
frames = []
train_size = 100

# %%
pbar = tqdm(range(100))
for i in pbar:
    i = i % train_size
    objects, loss, images, opt_state = update_fn(
        objects, positions[i], rays[i], gt_images[i], opt_state
    )
    pbar.set_description(f'loss: {loss:.4f}')
    display(imrow(*images))
    if i % 10 == 0:
        images = render(objects, positions[0], rays[0], light_dir)
        frames.append(images)
        display(imrow(*images))


# %%
imgs = [fromarray(to_rgb(imgs.image)) for imgs in frames]
imgs[0].save(
    'assets/other/lego.gif',
    save_all=True,
    append_images=imgs[1:],
    duration=100,
    loop=0,
)

# %%
