import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

RESOLUTION_SLIDER_ID = 'resolution-slider'
RADIUS_SLIDER_ID = 'radius-slider'
SPHERE_GRAPH_ID = 'sphere-graph'

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

server = app.server

app.layout = html.Div(
    [
        html.H2('Sphere'),
        html.Div(
            [
                'Resultion',
                dcc.Slider(
                    id=RESOLUTION_SLIDER_ID,
                    min=32,
                    max=500,
                    step=32,
                    value=40,
                ),
            ]
        ),
        html.Div(
            [
                'Radius',
                dcc.Slider(
                    id=RADIUS_SLIDER_ID,
                    min=0,
                    max=100,
                    step=4,
                    value=40,
                ),
            ]
        ),
        html.Center(
            dcc.Graph(id=SPHERE_GRAPH_ID),
        ),
    ]
)


@app.callback(
    Output(SPHERE_GRAPH_ID, 'figure'),
    Input(RESOLUTION_SLIDER_ID, 'value'),
    Input(RADIUS_SLIDER_ID, 'value'),
    Input(SPHERE_GRAPH_ID, 'clickData'),
)
def render(size: float, r: float, click_data: dict) -> dict:
    try:
        x0, y0 = click_data['points'][0]['x'], click_data['points'][0]['y']
        z0 = r**2 - x0**2 - y0**2
        if z0 < 0:
            raise ValueError
        z0 = np.sqrt(z0)
    except (TypeError, ValueError):
        x0, y0, z0 = 0, 0, r

    range = np.arange(-size, size + 1)
    x, y = np.meshgrid(range, range)

    h = r**2 - x**2 - y**2

    z = np.sqrt(h)
    b = (x * x0 + y * y0 + z * z0) / r**2

    return {
        'data': [
            {
                'x': range,
                'y': range,
                'z': b,
                'type': 'heatmap',
                'colorscale': 'Greys',
                'showscale': False,
            }
        ],
        'layout': {
            'xaxis': {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
            },
            'yaxis': {
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
            },
            'width': 1000,
            'height': 1000,
            'plot_bgcolor': 'black',
        },
    }


if __name__ == '__main__':
    app.run_server(debug=True)
