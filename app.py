from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div(
    [
        html.H2('Hello World'),
        dcc.Dropdown(
            ['Monkey Ball', 'Monkey Bar', 'Monkey Banana', 'Monkey Bread', 'Monkeh'],
            'Monkey Ball',
            id='dropdown',
        ),
        html.Div(id='display-value'),
    ]
)


@app.callback(Output('display-value', 'children'), [Input('dropdown', 'value')])
def display_value(value):
    return f'You have selected {value}'


if __name__ == '__main__':
    app.run_server(debug=True)
