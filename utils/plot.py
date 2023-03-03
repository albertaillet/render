import io
import base64
import numpy as np
from PIL import Image

# typing
from typing import Optional, Tuple
from jaxtyping import Array


def imshow(im: Optional[Array] = None, view_size: Tuple[int, int] = (0, 0)) -> dict:
    h, w = view_size

    ''' The returned figure is equivalent to the following:
    from plotly import graph_objects as go
    return go.Figure(
        data=go.Image(
            source=im if im is None else to_base_64(im),
            hoverinfo='none',
        ),
        layout=go.Layout(
            xaxis=dict(visible=False, range=(0, w)),
            yaxis=dict(visible=False, range=(0, h)),
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            dragmode=False,
        ),
    )
    '''

    return {
        'data': [
            {
                'type': 'image',
                'source': im if im is None else to_base_64(im),
                'hoverinfo': 'none',
            },
        ],
        'layout': {
            'xaxis': {'visible': False, 'range': (0, w)},
            'yaxis': {'visible': False, 'range': (h, 0)},
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0, 'pad': 0},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'dragmode': False,
        },
    }


def to_base_64(im: Array) -> str:
    with io.BytesIO() as buffer:
        Image.fromarray(np.asarray(im)).save(buffer, format='png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
