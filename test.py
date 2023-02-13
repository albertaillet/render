# %%
from jax import jit, grad, vmap, numpy as np, random
from utils.mlp import init_mlp_params, forward_mlp
from utils.linalg import normalize, norm, min, smoothmin
from raymarch import raymarch, camera_rays, cast_shadow, shade_f, LIGHT_DIR
from utils.plot import imshow, to_base_64
from plotly import graph_objects as go
from objects import Planes, Camera, softmax
from functools import partial

# typing
from typing import Tuple, Optional, Sequence, Any, Union
from jax import Array


camera = Camera(
    up=np.array([0.0, 1.0, 0.0]),
    position=np.array([3.0, 5.0, 3.0]),
    target=np.array([0.0, 0.0, 0.0]),
)

plane = Planes(
    position=np.zeros((3, 3)),
    normal=np.eye(3),
    color=np.eye(3) / 2.0 + 0.5,
)


@partial(jit, static_argnames=('view_size'))
def render_scene(
    params: Sequence[Tuple[Array, Array]],
    camera: Camera,
    plane: Planes,
    view_size: Tuple[int, int],
    light_dir: Array = LIGHT_DIR,
) -> Array:
    w, h = view_size
    ray_dir = camera_rays(
        camera.target - camera.position, camera.up, view_size=view_size
    )

    def sdf(p: Array) -> Array:
        d = smoothmin(plane.sdf(p))
        #mlp_d = forward_mlp(params, p)[0]
        #d = np.minimum(d, mlp_d)
        return d

    def color(p: Array) -> Array:
        dists = plane.sdf(p)
        colors = plane.color
        return softmax(-8.0 * dists) @ colors

    hit_pos = vmap(partial(raymarch, sdf, camera.position))(ray_dir)
    surface_color = vmap(color)(hit_pos)
    raw_normal = vmap(grad(sdf))(hit_pos)

    light_dir = normalize(light_dir)
    shadow = vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
    # color = vmap(partial(shade_f, light_dir=light_dir))(
    #     surface_color, raw_normal, ray_dir, shadow
    # )

    # color = color ** (1.0 / 2.2)  # gamma correction

    def to_rgb_image(img: Array) -> Array:
        img = np.uint8(255.0 * img.clip(0.0, 1.0))
        #return img.reshape((w, h, 3)).transpose((1, 0, 2))
        return img.reshape((w, h)).T

    return to_rgb_image(shadow)


# %%
def imshow(im: Optional[Array] = None, view_size: Tuple[int, int] = (0, 0)) -> dict:
    h, w = view_size
    return {
        'data': [
            {
                'source': im if im is None else to_base_64(im),
                'type': 'image',
                #'x': 0,
                #'y': h,
                #'sizex': w,
                #'sizey': h,
                #'sizing': 'stretch',
                #'layer': 'below',
                #'showscale': False,
                #'hoverinfo': 'none',
            },
        ],
        'layout': {
            'xaxis': {
                'visible': False,
                'range': [0, w],
            },
            'yaxis': {
                'visible': False,
                'range': [0, h],
                'scaleanchor': 'x',
            },
            'margin': {
                'l': 0,
                'r': 0,
                'b': 0,
                't': 0,
                'pad': 0,
            },
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'dragmode': False,
            #'height': '100%',
            #'width': '100%',
        },
    }


view_size = (300, 300)
params = init_mlp_params([3, 32, 32, 1], key=random.PRNGKey(0))

im = render_scene(params, camera, plane, view_size)
go.Figure(**imshow(im, view_size))

# %%
