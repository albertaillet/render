# %%
from jax import jit, grad, value_and_grad, vmap, numpy as np, random
from jax.nn import sigmoid
from optax import OptState, apply_updates, adam
from utils.mlp import init_mlp_params, MLP
from utils.linalg import normalize, norm, min, smoothmin, softmax
from raymarch import (
    Planes,
    Camera,
    render_scene,
    LIGHT_DIR,
)
from functools import partial
from tqdm import tqdm
from numpy import uint8
from PIL import Image
from matplotlib import pyplot as plt

# typing
from typing import Tuple, Sequence, NamedTuple, Iterable
from jax import Array

VIEW_SIZE = (100, 100)
render_scene = partial(
    render_scene, view_size=VIEW_SIZE, click=(-1, -1), light_dir=LIGHT_DIR
)


def imshow(*imgs: Sequence[Array]) -> None:
    _, axes = plt.subplots(1, len(imgs))
    if not isinstance(axes, Iterable):
        axes = [axes]
    for ax, img in zip(axes, imgs):
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def to_pil_image(im: Array) -> Image:
    return Image.fromarray(uint8(255.0 * im.clip(0.0, 1.0)))


# %%
pil_img_gt = Image.open('assets/other/little_snowman.png').resize(VIEW_SIZE)
transparency = np.array(pil_img_gt)[:, :, 4] == 0
foreground = 1 - transparency

img_gt = np.float32(pil_img_gt)[:, :, :3] / 255.0
img_gt = np.where(transparency[:, :, None], np.inf, img_gt)

pil_depth_gt = Image.open('assets/other/little_snowman_depth.png').resize(VIEW_SIZE)
depth_gt = 5 * np.float32(pil_depth_gt) / 255.0
depth_gt = np.where(transparency, np.inf, depth_gt)
imshow(img_gt, depth_gt)

# %%
camera = Camera(
    up=np.array([0.0, 1.0, 0.0]),
    position=np.array([1.0, 1.0, 0.0]),
    target=np.array([0.0, 0.25, 0.0]),
)

planes = Planes(
    position=np.array([[0.0, 0.0, 0.0]]),  # , [-2.0, 0.0, 0.0]]),
    normal=np.array([[0.0, 1.0, 0.0]]),  # , [1.0, 0.0, 0.0]]),
    color=np.ones((1, 3)),
)


class Scene(NamedTuple):
    objects: Tuple[NamedTuple, ...]
    color: MLP

    def sdf(self, p: Array) -> Array:
        return smoothmin(np.concatenate([o.sdf(p) for o in self.objects]))


class LearntObjects(NamedTuple):
    position: Array
    radius: Array

    def sdf(self, p: Array) -> Array:
        return norm(p - self.position) - self.radius


p_key, r_key, c_key = random.split(random.PRNGKey(0), 3)

n = 20
learnt_scene = Scene(
    objects=(
        LearntObjects(
            position=0.1 * random.normal(p_key, (n, 3)),
            radius=0.05 * random.uniform(r_key, (n,)),
        ),
        planes,
    ),
    color=MLP(init_mlp_params([3, 9, 9, 3], key=c_key)),
)

learnt_img, depth = render_scene(learnt_scene, camera)
imshow(img_gt, depth_gt, learnt_img, depth)


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
    (loss, (learnt_img, learnt_depth)), grads = value_and_grad(loss_fn, has_aux=True)(
        scene, camera
    )
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
