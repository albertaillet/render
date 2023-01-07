import yaml
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
from raymarch import render_scene
from objects import check_scene_dict, get_scene
from utils.plot import imshow

# typing
from typing import Tuple

SCENE_GRAPH_ID = 'scene-graph'
SCENE_STORE_ID = 'scene-store'
SCENE_EDIT_ACCESS_BUTTON_ID = 'scene-edit-access-button'
SCENE_EDIT_OFFCANVAS_ID = 'scene-edit-offcanvas'
SCENE_EDIT_CODE_ID = 'scene-edit-code'
SCENE_EDIT_POPOVER_ID = 'scene-edit-popover'
SCENE_EDIT_POPOVERHEADER_ID = 'scene-edit-popoverheader'
SCENE_EDIT_POPOVERBODY_ID = 'scene-edit-popoverbody'

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.title = 'Render'

server = app.server

app.layout = html.Div(
    [
        dbc.Container(
            [
                html.H2('Render'),
                dbc.Button('Edit Scene', id=SCENE_EDIT_ACCESS_BUTTON_ID, n_clicks=0),
            ]
        ),
        html.Center(
            dcc.Graph(
                figure=imshow([]),
                id=SCENE_GRAPH_ID,
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'doubleClick': False,
                },
            ),
            style={'width': '100%'},
        ),
        dbc.Offcanvas(
            [
                dbc.Textarea(
                    value=open('scenes/scene.yml', 'r').read(),
                    placeholder='Scene data',
                    id=SCENE_EDIT_CODE_ID,
                    size='sm',
                    wrap=True,
                    required=True,
                    style={
                        'width': '100%',
                        'height': '100%',
                        'background-color': '#343a40',
                        'color': '#fff',
                    },
                ),
                dbc.Popover(
                    [
                        dbc.PopoverHeader(id=SCENE_EDIT_POPOVERHEADER_ID),
                        dbc.PopoverBody(id=SCENE_EDIT_POPOVERBODY_ID),
                    ],
                    target=SCENE_EDIT_CODE_ID,
                    id=SCENE_EDIT_POPOVER_ID,
                    is_open=False,
                ),
            ],
            autofocus=True,
            id=SCENE_EDIT_OFFCANVAS_ID,
            title='Edit Scene',
        ),
        dcc.Store(
            id=SCENE_STORE_ID,
            data=yaml.load(open('scenes/scene.yml', 'r'), yaml.SafeLoader),
        ),
    ]
)


@app.callback(
    Output(SCENE_EDIT_OFFCANVAS_ID, 'is_open'),
    Input(SCENE_EDIT_ACCESS_BUTTON_ID, 'n_clicks'),
    State(SCENE_EDIT_OFFCANVAS_ID, 'is_open'),
)
def toggle_edit_offcanvas(n_clicks: int, is_open: bool) -> bool:
    return not is_open if n_clicks else is_open


@app.callback(
    Output(SCENE_STORE_ID, 'data'),
    Output(SCENE_EDIT_CODE_ID, 'invalid'),
    Output(SCENE_EDIT_POPOVER_ID, 'is_open'),
    Output(SCENE_EDIT_POPOVERHEADER_ID, 'children'),
    Output(SCENE_EDIT_POPOVERBODY_ID, 'children'),
    Input(SCENE_EDIT_CODE_ID, 'value'),
)
def save_code_to_store(scene_yml_str: str) -> Tuple[dict, bool]:
    try:
        scene_dict = yaml.load(scene_yml_str, Loader=yaml.SafeLoader)
        check_scene_dict(scene_dict)
        return scene_dict, False, False, no_update, no_update
    except Exception as e:
        return no_update, True, True, type(e).__name__, str(e)


@app.callback(
    Output(SCENE_GRAPH_ID, 'figure'),
    Input(SCENE_GRAPH_ID, 'clickData'),
    Input(SCENE_STORE_ID, 'data'),
)
def render(click_data: dict, scene_dict: dict) -> dict:
    try:
        click = click_data['points'][0]['x'], click_data['points'][0]['y']
    except TypeError:
        click = (-1, -1)
    scene, view_size = get_scene(scene_dict)
    im = render_scene(scene, view_size, click)
    return imshow(im)


if __name__ == '__main__':
    app.run_server(debug=True)
