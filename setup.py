import dash_bootstrap_components as dbc
from dash import Input, Output, State, clientside_callback, no_update, callback_context, ALL
from dash.dcc import Graph, Store
from yaml import SafeLoader, load
from raymarch import render_scene, IMAGE_NAMES
from builder import check_scene_dict, build_scene
from utils.plot import imshow
from pathlib import Path

# typing
from typing import Tuple

GRAPH_ID = 'graph'
GRAPH_DOWNLOAD_BUTTON_ID = 'graph-download-button'
VIEW_CHOICE_ID = 'view-choice'
STORE_ID = 'store'
EDIT_CODE_ID = 'edit-code'
EDIT_POPOVER_ID = 'edit-popover'
EDIT_POPOVERHEADER_ID = 'edit-popoverheader'
EDIT_POPOVERBODY_ID = 'edit-popoverbody'
FILE_LOAD_DROPDOWN_ID = 'file-load-dropdown'


def setup(app) -> None:
    FILES = sorted(Path('scenes').glob('*.yml'))
    app.title = 'Render'

    app.layout = dbc.Container(
        [
            dbc.Button(
                'Download Image',
                id=GRAPH_DOWNLOAD_BUTTON_ID,
                style={'margin': '10px 5px 10px 5px'},
            ),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem(file.stem, id={'type': FILE_LOAD_DROPDOWN_ID, 'index': i})
                    for i, file in enumerate(FILES)
                ],
                label='Load Scene from File',
                id=FILE_LOAD_DROPDOWN_ID,
                style={'margin': '10px 5px 10px 5px', 'display': 'inline-block'},
            ),
            dbc.RadioItems(
                options=[{'label': name.capitalize(), 'value': name} for name in IMAGE_NAMES],
                value=IMAGE_NAMES[0],
                inline=True,
                id=VIEW_CHOICE_ID,
                persistence=True,
                style={'margin': '0px 5px 10px 5px'},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Textarea(
                                placeholder='Scene data',
                                id=EDIT_CODE_ID,
                                size='sm',
                                required=True,
                                style={
                                    'width': '100%',
                                    'height': '100%',
                                    'backgroundColor': '#343a40',
                                    'color': '#fff',
                                },
                                persisted_props=['value'],
                            ),
                            dbc.Popover(
                                [
                                    dbc.PopoverHeader(id=EDIT_POPOVERHEADER_ID),
                                    dbc.PopoverBody(id=EDIT_POPOVERBODY_ID),
                                ],
                                target=EDIT_CODE_ID,
                                id=EDIT_POPOVER_ID,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        Graph(
                            figure=imshow(),
                            id=GRAPH_ID,
                            config={
                                'displayModeBar': False,
                                'scrollZoom': False,
                                'doubleClick': False,
                            },
                        )
                    ),
                ],
                style={'height': 'calc(100vh - 175px)'},
            ),
            Store(
                id=STORE_ID,
                storage_type='session',
            ),
        ]
    )

    @app.callback(
        Output(STORE_ID, 'data'),
        Output(EDIT_CODE_ID, 'invalid'),
        Output(EDIT_POPOVER_ID, 'is_open'),
        Output(EDIT_POPOVERHEADER_ID, 'children'),
        Output(EDIT_POPOVERBODY_ID, 'children'),
        Input(EDIT_CODE_ID, 'value'),
    )
    def save_code_to_store(scene_str: str) -> Tuple[dict, bool, bool, str, str]:
        '''Saves the editable scene config to the store if it is valid.
        Otherwise, shows an error popover.'''
        try:
            scene_dict = check_scene_dict(load(scene_str, Loader=SafeLoader))
            store = {'scene_dict': scene_dict, 'scene_str': scene_str}
            return store, False, False, no_update, no_update
        except Exception as e:
            return no_update, True, True, type(e).__name__, str(e)

    @app.callback(
        Output(EDIT_CODE_ID, 'value'),
        Input({'type': FILE_LOAD_DROPDOWN_ID, 'index': ALL}, 'n_clicks'),
        State(STORE_ID, 'data'),
    )
    def load_scene_str_to_testarea(_, store: dict) -> str:
        '''Load the editable scene config from file or fill using the store on intial call.'''
        triggered_prop_ids = callback_context.triggered_prop_ids
        if triggered_prop_ids:  # if a file was clicked
            idx = next(iter(triggered_prop_ids.values()))['index']
            return open(FILES[idx], 'r').read()
        elif store:  # if the store is not empty
            return store['scene_str']
        return open(FILES[0], 'r').read()  # else load the first file

    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(GRAPH_ID, 'clickData'),
        Input(VIEW_CHOICE_ID, 'value'),
        Input(STORE_ID, 'data'),
        prevent_initial_call=True,
    )
    def render(click_data: dict, view: str, store: dict) -> dict:
        '''Renders the scene with the given click, view choice and scene data.'''
        try:
            point = click_data['points'][0]
            click = point['y'], point['x']
        except TypeError:
            click = (-1, -1)
        args = build_scene(store['scene_dict'])
        im = render_scene(**args, click=click).get(view)
        return imshow(im, args['view_size'])

    clientside_callback(
        '''
        function(n_clicks, figure){
            if(n_clicks > 0){
                let dlink = document.createElement('a');
                dlink.href = figure.data[0].source;
                dlink.download = 'render.png';
                dlink.click();
            }
        }
        ''',
        Output(GRAPH_DOWNLOAD_BUTTON_ID, 'n_clicks'),
        Input(GRAPH_DOWNLOAD_BUTTON_ID, 'n_clicks'),
        State(GRAPH_ID, 'figure'),
    )
