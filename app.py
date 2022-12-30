from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from render import render_scene

# typing
from jax import Array

RESOLUTION_SLIDER_ID = 'resolution-slider'
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
                            value=128,
                            marks={i: str(i) for i in range(2**4, 2**7, 2**4)},
                            persistence_type='session'
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
    Input(SPHERE_GRAPH_ID, 'clickData'),
)
def render(resultion: int, click_data: dict) -> dict:
    try:
        x0, y0 = click_data['points'][0]['x'], click_data['points'][0]['y']
    except TypeError:
        x0, y0 = -1, -1
    im = render_scene(resultion, resultion, x0, y0)
    return imshow(im)


def imshow(im: Array) -> dict:
    return {
        'data': [
            {
                'z': im,
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
