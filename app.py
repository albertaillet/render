import numpy as np
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

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
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.5,
                            marks={i: f'{i:.1f}' for i in np.linspace(0, 1, 11)},
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
def render(size: float, r: float, click_data: dict) -> dict:
    try:
        x0, y0 = click_data['points'][0]['x'], click_data['points'][0]['y']
        z0 = r**2 - x0**2 - y0**2
        if z0 < 0:
            raise ValueError
        z0 = np.sqrt(z0)
    except (TypeError, ValueError):
        x0, y0, z0 = 0, 0, r

    space = np.linspace(-1, 1, size)
    x, y = np.meshgrid(space, space)

    h = r**2 - x**2 - y**2

    h = np.where(h > 0, h, 0)
    z = np.sqrt(h)
    b = (x * x0 + y * y0 + z * z0) / r**2
    b = np.where(b > 0, b, 0)

    return {
        'data': [
            {
                'x': space,
                'y': space,
                'z': b,
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
