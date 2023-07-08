from dash import Dash
from setup import setup
from dash_bootstrap_components import themes


app = Dash(__name__, external_stylesheets=[themes.DARKLY])

setup(app)

app.run_server(debug=True)
