import dash
from dash import html, Dash, dcc, callback, Input, Output, Patch, dash_table, DiskcacheManager
import diskcache
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc  # todo migrare a 0.14
import os
import sys
from flask import send_from_directory
from whitenoise import WhiteNoise

from src.dashboards._commons import (DASH_PORT, OPTUNA_PORT, TENSORBOARD_PORT, DASH_PAGES_APP, DASH_CACHE,
                                     DASH_STATIC)
from settings.config import DATA_PATH

cache = diskcache.Cache(os.path.join(DASH_CACHE, "_cache"))
background_callback_manager = DiskcacheManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]

app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets,
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
           suppress_callback_exceptions=True, background_callback_manager=background_callback_manager,
           pages_folder=DASH_PAGES_APP
           )
server = app.server


@server.get('/assets/data/<path:resource>')
def serve_dataset_asset(resource):
    """Serve one dataset image without making the complete dataset static."""
    return send_from_directory(DATA_PATH, resource)


server.wsgi_app = WhiteNoise(server.wsgi_app, root=DASH_STATIC)


app.layout = html.Div([
    html.Div([
        dbc.NavbarSimple(
            children=[
                         dbc.NavItem(dbc.NavLink(page['name'], href=page['relative_path']))
                         for page in dash.page_registry.values() if page['nav'] is True] + [
                         dbc.NavItem(dbc.NavLink("Optuna Dashboard", href=f"http://127.0.0.1:{OPTUNA_PORT}",
                                                 external_link=True, target="_blank")),
                         dbc.NavItem(dbc.NavLink("Tensorboard", href=f"http://127.0.0.1:{TENSORBOARD_PORT}",
                                                 external_link=True, target="_blank"))

                     ],
            brand="Project: Ingredients Recognition",
            brand_href="/",
            color="dark",
            dark=True,
        )
    ]),

    dash.page_container,

    dmc.Footer(
        fixed=False,
        height=60,
        children=[
            html.P("Master Thesis: Ingredients Recognition, Luca Poli [852027]"),
        ],
        style={"background-color": "#333333", "color": "white", "text-align": "center", "padding-top": "20px",
               "margin-top": "20px"}
    )
])

# @app.server.route('/assets/<image_path>.jpg')
# def serve_assets(resource):
#     print("ciao")
#     return flask.send_from_directory(DATA_PATH, resource)


if __name__ == '__main__':
    # Werkzeug cannot safely reconstruct PyCharm's debugger command on Windows
    # when its installation path contains spaces. Keep hot reload for normal
    # runs, but disable the extra reloader process while a debugger is attached.
    app.run(debug=True, port=DASH_PORT, use_reloader=sys.gettrace() is None)
