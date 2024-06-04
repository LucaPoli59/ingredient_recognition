import os
import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import html, Dash, dcc, callback, Input, Output, Patch, dash_table


dash.register_page(__name__, path="/", name="Home", title="Home", is_index=True, order=0, nav=True)

layout = dbc.Container(fluid=True, children=[
    dcc.Location(id='url', refresh=False),
    html.Center(html.H1("Home", className="display-3 my-4")),
])
