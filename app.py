from dash import Dash
from setup import setup


app = Dash(__name__)
setup(app)
app.run_server(debug=True)
