import io
import base64
import numpy as np
from PIL import Image

# typing
from typing import Optional, Tuple
from jax import Array


def imshow(im: Optional[Array] = None, view_size: Tuple[int, int] = (0, 0)) -> dict:
    h, w = view_size
    return {
        'data': [
            {
                'source': im if im is None else to_base_64(im),
                'type': 'image',
                'xref': 'x',
                'yref': 'y',
                'x': 0,
                'y': h,
                'sizex': w,
                'sizey': h,
                'sizing': 'stretch',
                'layer': 'below',
                'showscale': False,
                'hoverinfo': 'none',
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
            'height': '100vh',
            'width': '100vw',
        },
    }


def to_base_64(im: Array) -> str:
    im = Image.fromarray(np.asarray(im))
    with io.BytesIO() as buffer:
        im.save(buffer, format='png')
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
