from jax import numpy as np
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

# typing
from jax import Array

RESOLUTION_SLIDER_ID = 'resolution-slider'
RADIUS_SLIDER_ID = 'radius-slider'
SPHERE_GRAPH_ID = 'sphere-graph'

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.title = 'Sphere'

server = app.server

app.layout = html.Div(
    [
        dbc.Container(
            [
                html.H2('Sphere'),
                html.Div(
                    [
                        'Resultion',
                        dcc.Slider(
                            id=RESOLUTION_SLIDER_ID,
                            min=2**4,
                            max=2**7,
                            step=2,
                            value=48,
                            marks={i: str(i) for i in range(2**4, 2**7, 2**4)},
                        ),
                    ]
                ),
                html.Div(
                    [
                        'Radius',
                        dcc.Slider(
                            id=RADIUS_SLIDER_ID,
                            min=0.1,
                            max=1,
                            step=0.01,
                            value=0.5,
                            marks={float(i): f'{i:.1f}' for i in np.linspace(0, 1, 11)},
                        ),
                    ]
                ),
            ]
        ),
        html.Center(
            dcc.Graph(
                id=SPHERE_GRAPH_ID,
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'doubleClick': False,
                },
            ),
            style={'width': '100%'},
        ),
    ]
)


@app.callback(
    Output(SPHERE_GRAPH_ID, 'figure'),
    Input(RESOLUTION_SLIDER_ID, 'value'),
    Input(RADIUS_SLIDER_ID, 'value'),
    Input(SPHERE_GRAPH_ID, 'clickData'),
)
def render(resultion: int, r: float, click_data: dict) -> dict:
    try:
        x0, y0 = click_data['points'][0]['x'], click_data['points'][0]['y']
    except TypeError:
        x0, y0 = 0, 0

    space = np.linspace(-1, 1, resultion)
    x, y = np.meshgrid(space, space)

    z0 = np.sqrt(np.clip(r**2 - x0**2 - y0**2, 0, None))
    z = np.sqrt(np.clip(r**2 - x**2 - y**2, 0, None))
    b = np.clip((x * x0 + y * y0 + z * z0) / r**2, 0, None)
    return imshow(space, space, b)


def imshow(x: Array, y: Array, z: Array) -> dict:
    return {
        'data': [
            {
                'x': x,
                'y': y,
                'z': z,
                'type': 'heatmap',
                'colorscale': 'Greys',
                'showscale': False,
                'hoverinfo': 'none',
            }
        ],
        'layout': {
            'xaxis': {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'scaleanchor': 'y',
                'scaleratio': 1,
            },
            'yaxis': {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
            },
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'margin': {
                'l': 0,
                'r': 0,
                'b': 0,
                't': 0,
            },
            'dragmode': False,
            'height': '100%',
        },
    }


if __name__ == '__main__':
    app.run_server(debug=True)
