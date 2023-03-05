# %%
import matplotlib.pyplot as plt
from jax import jit, grad, value_and_grad, vmap, numpy as np, random
from jax.nn import sigmoid
from utils.mlp import init_mlp_params, MLP
from utils.linalg import normalize, norm, min, smoothmin, softmax
from raymarch import (
    Planes,
    Spheres,
    Camera,
    Scene,
    render_scene,
    LIGHT_DIR,
)
from functools import partial
import optax
from tqdm import tqdm
import numpy as onp
from PIL import Image

# typing
from typing import Tuple, Sequence, NamedTuple, Callable
from jax import Array

VIEW_SIZE = (100, 100)
render_scene = partial(
    render_scene, view_size=VIEW_SIZE, click=(-1, -1), light_dir=LIGHT_DIR
)

# %%
gt_pil_img = Image.open("snowman.png").resize(VIEW_SIZE).transpose(Image.FLIP_LEFT_RIGHT)
transparency = np.array(gt_pil_img)[:, :, 4] == 0

gt_img = np.float32(gt_pil_img)[:, :, :3] / 255.0
gt_img = np.where(transparency[:, :, None], 1.0, gt_img)
gt_pil_img

# %%
camera = Camera(
    up=np.array([0.0, 1.0, 0.0]),
    position=np.array([3.0, 4.0, 3.0]),
    target=np.array([0.0, 0.0, 0.0]),
)

planes = Planes(
    position=np.zeros((1, 3)),
    normal=np.array([[0.0, 1.0, 0.0]]),
    color=np.ones((1, 3)),
)


class LearntObjects(NamedTuple):
    position: Array
    radius: Array
    color: MLP

    def sdf(self, p: Array) -> Array:
        return norm(p - self.position) - self.radius

    def colorf(self, p: Array) -> Array:
        n = self.radius.shape[0]
        c = sigmoid(self.color(p))
        c = np.tile(c, (n, 1))
        return c


p_key, r_key, c_key = random.split(random.PRNGKey(0), 3)

n = 10
p_scale = 2.0
r_scale = 0.1
learnt_objects = LearntObjects(
    position=p_scale * random.normal(p_key, (n, 3)),
    radius=r_scale * random.uniform(r_key, (n,)),
    color=MLP(init_mlp_params([3, 32, 32, 3], key=c_key)),
)
learnt_scene = Scene((learnt_objects, planes))

# %%
plt.imshow(gt_img)
plt.show()
learnt_img = render_scene(learnt_scene, camera)
plt.imshow(learnt_img)
plt.show()

# %%
@jit
def loss_fn(
    scene: Scene,
    camera: Camera,
    gt_image: Array,
) -> Tuple[Array, Array]:
    # render scene
    image = render_scene(scene, camera)
    # compute l1 loss
    loss = np.mean(np.abs(image - gt_image))
    return loss, image


@jit
def update_fn(
    scene: Scene,
    camera: Camera,
    gt_image: Array,
    opt_state: optax.OptState,
) -> Tuple[Scene, Array, Array, optax.OptState]:
    (loss, learnt_img), grads = value_and_grad(loss_fn, has_aux=True)(
        scene, camera, gt_image
    )
    updates, opt_state = opt_update(grads, opt_state)
    scene = optax.apply_updates(scene, updates)
    return scene, loss, learnt_img, opt_state


# %%
lr = 0.001
opt_init, opt_update = optax.adam(lr)
opt_state = opt_init(learnt_scene)
frames = []

# %%
pbar = tqdm(range(1_000))
for i in pbar:
    learnt_scene, loss, learnt_img, opt_state = update_fn(
        learnt_scene, camera, gt_img, opt_state
    )
    pbar.set_description(f"loss: {loss:.4f}")
    frames.append(learnt_img)

# %%
imgs = [Image.fromarray((255 * onp.asarray(img)).astype(np.uint8)) for img in frames]
imgs[0].save("snowman.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)

# %%
