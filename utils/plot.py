import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode
from plotly import graph_objects as go


# typing
from numpy.typing import ArrayLike
from typing import Optional, Tuple


def imshow(
    im: Optional[ArrayLike] = None, view_size: Tuple[int, int] = (0, 0)
) -> go.Figure:
    h, w = view_size

    return go.Figure(
        data=go.Image(
            source=im if im is None else to_base_64(to_rgb(im)),
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
    return np.uint8(255.0 * im.clip(0.0, 1.0))


def to_base_64(im: ArrayLike) -> str:
    with BytesIO() as buffer:
        Image.fromarray(im).save(buffer, format='png')
        return 'data:image/png;base64,' + b64encode(buffer.getvalue()).decode()
