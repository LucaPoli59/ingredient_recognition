import dash
from dash import html, Dash, dcc, callback, Input, Output, Patch, dash_table, DiskcacheManager
import diskcache
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc  # todo migrare a 0.14
import os
from whitenoise import WhiteNoise

from src.dashboards._commons import DASH_PORT, OPTUNA_PORT, TENSORBOARD_PORT, DASH_PAGES_APP, DASH_CACHE
from settings.config import PROJECT_PATH, BLANK_IMG_PATH

cache = diskcache.Cache(os.path.join(DASH_CACHE, "_cache"))
background_callback_manager = DiskcacheManager(cache)

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]

app = Dash(__name__, use_pages=True, external_stylesheets=external_stylesheets,
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
           suppress_callback_exceptions=True, background_callback_manager=background_callback_manager,
           pages_folder=DASH_PAGES_APP
           )
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')


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
    app.run(debug=True, port=DASH_PORT)
