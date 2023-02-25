from dash import Dash
from setup import setup
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

setup(app)

server = app.server

app.run_server(debug=True)
