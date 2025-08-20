from io import BytesIO
from base64 import b64encode
from numpy import isnan, uint8
from PIL.Image import Image, fromarray


# typing
from jax import Array
from numpy.typing import ArrayLike as NpArray


def imshow(im: Array) -> str:
    return to_base_64(fromarray(to_rgb(im)))


def to_rgb(im: Array) -> NpArray:
    if len(im.shape) == 2:
        im = im.reshape(*im.shape, 1).repeat(3, axis=2)
    if isnan(im).any():
        im = fill_nan(im)
    return uint8(255 * im.clip(0.0, 1.0))


def to_base_64(im: Image) -> str:
    with BytesIO() as buffer:
        im.save(buffer, format='png')
        return 'data:image/png;base64,' + b64encode(buffer.getvalue()).decode()


def fill_nan(im: Array) -> Array:
    # highlight the nan pixels in red
    from jax import numpy as np, vmap

    def color_nan_pixel(x: Array) -> Array:
        return np.where(np.isnan(x).any(), np.array([1, 0, 0]), x)

    return vmap(vmap(color_nan_pixel))(im)
