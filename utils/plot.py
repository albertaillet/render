from PIL import Image
from io import BytesIO
from base64 import b64encode
from numpy import isnan, uint8
from yaml import SafeLoader, load
from plotly import graph_objects as go


# typing
from numpy.typing import ArrayLike
from typing import Optional, Tuple


def imshow(im: Optional[ArrayLike] = None, view_size: Tuple[int, int] = (0, 0)) -> go.Figure:
    h, w = view_size

    return go.Figure(
        data=go.Image(
            source=im if im is None else to_base_64(to_pil(to_rgb(im))),
            hoverinfo='none',
        ),
        layout=go.Layout(
            xaxis=dict(visible=False, range=(0, w)),
            yaxis=dict(visible=False, range=(h, 0)),
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            dragmode=False,
        ),
    )


def to_rgb(im: ArrayLike) -> ArrayLike:
    if isnan(im).any():
        im = fill_nan(im)
    return uint8(255 * im.clip(0.0, 1.0))


def to_pil(im: ArrayLike) -> Image:
    return Image.fromarray(im)


def to_base_64(im: Image) -> str:
    with BytesIO() as buffer:
        im.save(buffer, format='png')
        return 'data:image/png;base64,' + b64encode(buffer.getvalue()).decode()


def load_yaml(path: str) -> dict:
    return load(open(path, 'r'), SafeLoader)


def fill_nan(im: ArrayLike) -> ArrayLike:
    # highlight the nan pixels in red
    from jax import numpy as np, vmap

    def color_nan_pixel(x: ArrayLike) -> ArrayLike:
        return np.where(np.isnan(x).any(), np.array([1, 0, 0]), x)

    if len(im.shape) == 2:
        im = np.tile(im.reshape(*im.shape, 1), (1, 1, 3))

    return vmap(vmap(color_nan_pixel))(im)
