import os

import dash
from typing import Any, Tuple, List, Optional
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import jsonpickle
import numpy as np
from dash_iconify import DashIconify
import yaml
from dash import html, Dash, dcc, callback, Input, Output, Patch, dash_table, State
import dash_uploader
import base64
import plotly.express as px
import cv2
import pandas as pd
import torch
import io
import timeit
from PIL import Image
import io

from dash.exceptions import PreventUpdate
from settings.config import EXPERIMENTS_PATH, HTUNER_CONFIG_FILE, BLANK_IMG_PATH, PROJECT_PATH
from src.dashboards._commons import recursive_listdir, DASH_CACHE, open_img, dash_get_asset_url, img_from_ndarray
from src.commons.exp_config import ExpConfig, HTunerExpConfig
from src.data_processing.data_handling import LightImagesRecipesDataset, get_transform_plain, MultiLabelBinarizerRobust
from src.commons.visualizations import gradcam, feature_factorization, correct_legend_factor
from src.commons.utils import pred_digits_to_values

DEVICE = "cpu"  # needed since the library uses the model on CPU
MODEL_CACHE_PATH = os.path.join(DASH_CACHE, "model_cache.pt")
IMG_WIDTH, IMG_HEIGHT = "600px", "400px"
TABLES_PARAMS = dict(bordered=True, hover=True, responsive=True, striped=True, size='lg')
DEF_GRADCAM_TARGET = [{"value": None, "label": "Most Probable"}]

dash.register_page(__name__, path="/model_visualization", name="Model Visualization", title="Model Visualization",
                   order=3, nav=True)

experiments = recursive_listdir(EXPERIMENTS_PATH)
experiments_str = [exp.replace(EXPERIMENTS_PATH, "")[1:] for exp in experiments]

layout = dbc.Container(fluid=True, children=[
    dcc.Location(id='url', refresh=False),
    dbc.Container(children=[
        html.Center(html.H1("Presentation of results", className="display-3 my-3")),

        html.H5("Load an experiment to visualize the results:"),
        html.Div([
            dmc.Select(id='load_exp_select', placeholder="Select an experiment",
                       data=[{'label': exp_str, 'value': exp} for exp, exp_str in zip(experiments, experiments_str)],
                       searchable=True, nothingFound="No experiments found", className="w-50"),
            dmc.Select(id='load_htrial_select', placeholder="Select a trial",
                       searchable=True, nothingFound="No trials found", className="w-20", disabled=True),
            dmc.Button("Load", color="primary", className="ml-3", id="load_exp_button"),

            # TODO add uploader from file system
            dcc.Upload(id='load_exp_upload', multiple=False, className="ml-3", accept=".ckpt, .yaml",
                       children=dmc.Button('Drag and Drop or Select Files', id="load_exp_upload_inner_button",
                                           color="secondary")),
            # dash_uploader.Upload(id='load_exp_upload', maxFiles=1, multiple=False, accept=".ckpt, .yaml",
        ], className="my-4 d-flex align-items-center gap-2"),

    ]),
    dbc.Toast(id="loading_notify", is_open=False, header="Loading experiment", duration=10000, dismissable=True,
              class_name="position-absolute top-0 end-0 me-3", style={"margin-top": "60px"}),

    dmc.Divider(size="xl", className="my-5"),

    dbc.Container(children=[
        html.Center(html.Div(className="d-inline-flex align-items-center gap-5", children=[
            html.H3("Visualize prediction for an image"),
            html.Div(className="d-inline-flex align-items-center gap-2", children=[
                dmc.Slider(id="img_preds_weight_slider", min=0, max=1, step=0.1, value=0.5, size="xs",
                           showLabelOnHover=False, marks=[{"value": 0, "label": "0"}, {"value": 1, "label": "1"}],
                           style={"width": "80px"}),
                dbc.Tooltip("Masks opacity", target="img_preds_weight_slider", placement="bottom",
                            style={"font-size": "0.6rem"}),
                dmc.Select(id="gradcam_target_select", placeholder="Select target for GradCAM",
                           disabled=True, searchable=True, style={"width": "200px"},
                           data=DEF_GRADCAM_TARGET, value=DEF_GRADCAM_TARGET[0]['value']),
                dbc.Button("Load predictions", color="primary", id="load_preds_btn", disabled=True),
            ])

        ])),
        html.Div(className="d-flex justify-content-between mt-5", children=[
            html.Div(className="align-items-center", children=[
                html.Img(id="img_display", src=dash_get_asset_url(BLANK_IMG_PATH), alt="Current image",
                         style={"width": IMG_WIDTH, "height": IMG_HEIGHT}),
                dmc.Button(DashIconify(icon="simple-line-icons:arrow-left", color="black", width=24), variant="subtle",
                           size="lg", className="position-absolute top-50 start-0", id="img_bt_prev", disabled=True),
                dbc.Tooltip("Previous Image",
                            target="img_bt_prev", placement="top", style={"font-size": "0.6rem"}),
                dmc.Button(DashIconify(icon="simple-line-icons:arrow-right", color="black", width=24), variant="subtle",
                           size="lg", className="position-absolute top-50 end-0", id="img_bt_next", disabled=True),
                dbc.Tooltip("Next Image", target="img_bt_next", placement="top", style={"font-size": "0.6rem"}),
            ], style={"position": "relative", "width": IMG_WIDTH, "max-width": IMG_WIDTH}),

            html.Div(id="img_labels_table", style={"overflow": "auto"}),
            html.Div(id="img_preds_table", style={"overflow": "auto"}),

        ], style={"height": IMG_HEIGHT, "max-height": IMG_HEIGHT}),

        dmc.Divider(size="xs", className="my-3"),
        dbc.Row([
            dbc.Col(dcc.Loading(dcc.Graph(id="preds_gradcam"), type="circle"), className="col-5"),
            dbc.Col(dcc.Loading(dcc.Graph(id="preds_factors"), type="circle"), className="col-7"),
        ], className="justify-content-between aligns-items-center"),

    ]),

    dbc.Toast(id="inference_notify", is_open=False, header="Prediction notifications", duration=10000,
              dismissable=True, class_name="position-absolute top-0 end-0 me-3", style={"margin-top": "60px"}),

    dcc.Store(id='store_data_loader', data={}, storage_type="memory"),  # todo put in session
    dcc.Store(id='store_label_encoder', data={}, storage_type="memory"),
    dcc.Store(id='store_imgs_shape', data={}, storage_type="memory"),
    # Make this to the transform method loaded similar to the config
    dcc.Store(id='store_imgs_data', data={}, storage_type="memory"),
    dcc.Store(id='store_img_index', data={"curr": None, "max": None}, storage_type="memory"),
    dcc.Store(id='store_model_loaded', data=False, storage_type="memory"),
])


@callback(Output('load_htrial_select', 'data'),
          Output('load_htrial_select', 'disabled'),
          Output('load_htrial_select', 'value'),
          Input('load_exp_select', 'value'), prevent_initial_call=True)
def update_trial_selector(selected_exp):
    if selected_exp is None:
        return dash.no_update, dash.no_update, dash.no_update

    elems = os.listdir(selected_exp)
    if HTUNER_CONFIG_FILE not in elems:
        return [], True, None

    paths = [os.path.join(selected_exp, elem) for elem in elems if elem.startswith("trial_")]
    paths_str = [os.path.basename(elem).replace("trial_", "").upper() for elem in paths]
    return [{'label': exp_str, 'value': exp} for exp, exp_str in zip(paths, paths_str)], False, paths[-1]


@callback(Output("store_data_loader", "data"),
          Output("store_label_encoder", "data"),
          Output("store_imgs_shape", "data"),
          Output("store_model_loaded", "data", allow_duplicate=True),
          Output('gradcam_target_select', 'data'),
          Output('gradcam_target_select', 'value'),
          Output('loading_notify', 'is_open', allow_duplicate=True),
          Output('loading_notify', 'children'),
          Output('loading_notify', 'icon'),
          Input('load_exp_button', 'n_clicks'),
          State('load_exp_select', 'value'),
          State('load_htrial_select', 'value'),
          Input('load_exp_upload', 'contents'),
          State('load_exp_upload', 'filename'), prevent_initial_call=True, background=False,
          # running=[
          #     (Output('load_exp_button', 'disabled'), True, False),
          #     (Output('load_exp_upload', 'disabled'), True, False),
          #     (Output('load_exp_upload_inner_button', 'disabled'), True, False),
          #     (Output('load_exp_select', 'disabled'), True, False),
          #     (Output('load_htrial_select', 'disabled'), True, False),
          #     (Output('load_exp_upload', 'disabled'), True, False),
          #     (Output('loading_notify', 'is_open'), True, False),
          #     (Output('loading_notify', 'children'),
          #      html.P("Loading experiment"), html.P("")),
          #     (Output('loading_notify', 'icon'), "info", "")
          # ]
          )
def load_experiment(_, selected_exp, selected_htrial, upload_contents, upload_filenames):
    try:
        if dash.callback_context.triggered_id == "load_exp_button":
            if selected_exp is None:
                raise ValueError("No experiment selected")
            exp_config, model, feedback = _load_exp_from_select(selected_exp, selected_htrial)

        elif dash.callback_context.triggered_id == "load_exp_upload":
            # files = _load_exp_from_upload(upload_contents, upload_filenames)
            raise NotImplementedError("Upload not implemented yet")
        else:
            raise ValueError("Unknown trigger")

    except Exception as e:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                True, f"Error: {e}", "danger")

    dm_config = exp_config.datamodule
    dm_type = dm_config["type"]
    datamodule = dm_type.load_from_config(dm_config, batch_size=exp_config.lgn_model["batch_size"])
    datamodule.prepare_data()
    datamodule.setup()

    dataset = datamodule.val_dataloader().dataset.to_light_dataset(datamodule.label_encoder)
    label_encoder = jsonpickle.encode(datamodule.label_encoder)

    ingredients = pd.Series(dataset.label_data).explode().value_counts().index
    target_options = (DEF_GRADCAM_TARGET +
                      [{"value": datamodule.label_encoder.get_index(ingredient), "label": ingredient}
                       for ingredient in ingredients])

    imgs_shape = datamodule.image_shape
    torch.save(model.model, MODEL_CACHE_PATH)

    return (dataset.to_json(), label_encoder, imgs_shape, True, target_options, target_options[1]['value'],
            True, feedback, "success")


@callback(Output('store_imgs_data', 'data'),
          Output('store_img_index', 'data', allow_duplicate=True),
          Output('img_display', 'src', allow_duplicate=True),
          Output('img_bt_prev', 'disabled'),
          Output('img_bt_next', 'disabled'),
          Output('img_labels_table', 'children', allow_duplicate=True),
          Output('img_preds_table', 'children', allow_duplicate=True),
          Output('store_model_loaded', 'data', allow_duplicate=True),
          Input('store_data_loader', 'data'), prevent_initial_call=True)
def load_images(store_data_loader):
    if store_data_loader is None or store_data_loader == {}:
        raise PreventUpdate
    dataset = LightImagesRecipesDataset.from_json(store_data_loader)
    data = [{"img": str(img_path), "labels": labels}
            for img_path, labels in zip(dataset.images_paths, dataset.label_data)]

    labels_table = _create_labels_tables(data[0]["labels"])
    return (data, {"curr": 0, "max": len(data) - 1}, dash_get_asset_url(data[0]["img"]), False, False, labels_table,
            dbc.Table(), True)


@callback(Output('img_display', 'src', allow_duplicate=True),
          Output('store_img_index', 'data', allow_duplicate=True),
          Output('img_labels_table', 'children', allow_duplicate=True),
          Output('img_preds_table', 'children', allow_duplicate=True),
          Input('img_bt_next', 'n_clicks'),
          Input('img_bt_prev', 'n_clicks'),
          State('store_img_index', 'data'),
          State('store_imgs_data', 'data'), prevent_initial_call=True)
def scroll_img(_, __, index_data, imgs_data):
    idx_mod = 1 if dash.callback_context.triggered_id == "img_bt_next" else -1
    curr_idx, max_idx = index_data["curr"], index_data["max"]
    next_idx = (curr_idx + idx_mod) % (max_idx + 1)

    labels_table = _create_labels_tables(imgs_data[next_idx]["labels"])
    return dash_get_asset_url(imgs_data[next_idx]["img"]), {"curr": next_idx, "max": max_idx}, labels_table, dbc.Table()


@callback(Output('load_preds_btn', 'disabled'),
          Output("gradcam_target_select", "disabled"),
          Input('store_model_loaded', 'data'), prevent_initial_call=True)
def enable_model_use(model_loaded):
    return [not model_loaded] * 2


@callback(Output('preds_gradcam', 'figure', allow_duplicate=True),
          Output('preds_factors', 'figure', allow_duplicate=True),
          Output('img_preds_table', 'children', allow_duplicate=True),
          Output('loading_notify', 'is_open', allow_duplicate=True),
          Output('inference_notify', 'is_open'),
          Output('inference_notify', 'children'),
          Output('inference_notify', 'icon'),
          Input('load_preds_btn', 'n_clicks'),
          State('gradcam_target_select', 'value'),
          State('store_imgs_shape', 'data'),
          State('store_img_index', 'data'),
          State('store_imgs_data', 'data'),
          State("img_preds_weight_slider", 'value'),
          State("store_label_encoder", "data"), prevent_initial_call=True)
def make_inference(_, target, imgs_shape, img_index_data, imgs_data, img_weight, label_encoder, device=DEVICE):
    try:
        model = _load_model().to(device)
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, f"Error: {e}", "danger"

    targets = [target] if target is not None else None
    model.eval()
    imgs_transform = get_transform_plain(imgs_shape)
    img_path = imgs_data[img_index_data["curr"]]["img"]
    img = imgs_transform(Image.open(img_path))
    label_encoder = jsonpickle.decode(label_encoder)

    gradcam_imgs, gradcam_masks, gradcam_target, outputs = gradcam(model, model.conv_target_layer, img.to(device),
                                                                   targets=targets, img_weight=1 - img_weight)
    gradcam_img, gradcam_mask, gradcam_target, output = gradcam_imgs[0], gradcam_masks[0], gradcam_target[0], outputs[0]

    if not isinstance(gradcam_target, str):
        gradcam_target = label_encoder.decode_labels([[int(gradcam_target)]])[0][0]

    factors_img = feature_factorization(model, model.conv_target_layer, model.classifier_target_layer,
                                        img.to(device), img_weight=1 - img_weight, label_encoder=label_encoder)[0]
    gradcam_img = _create_img_plot(gradcam_img
                                   ).add_annotation(x=0.95, y=0.99, text=f"Target: {gradcam_target}", showarrow=False,
                                                    font_size=20, font_color="black", xref="paper", yref="paper")
    factors_img = _create_img_plot(correct_legend_factor(factors_img, ratio=0.75))
    preds_table = _create_preds_table(torch.sigmoid(output).cpu().detach().numpy(), label_encoder=label_encoder)

    return gradcam_img, factors_img, preds_table, False, True, "Inference completed", "success"


def _create_img_plot(img_array: np.ndarray):
    return px.imshow(img_array).update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0)
                                              ).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)


def _load_exp_from_select(select_value, selected_htrial, device=DEVICE) -> Tuple[ExpConfig, torch.nn.Module, str]:
    if selected_htrial is None:
        ckpt_path = _find_checkpoint_from_trial(select_value)
        if ckpt_path is None:
            raise ValueError("No checkpoint found in the selected trial")

        exp_config = ExpConfig.load_from_ckpt_data(torch.load(ckpt_path))
        model = exp_config.lgn_model['lgn_model_type'].load_from_config(exp_config.lgn_model).to(device)
        model.load_weights_from_checkpoint(ckpt_path)

        output = f"Loaded experiment from {os.path.basename(ckpt_path)}"

    else:
        ckpt_weights_path = _find_checkpoint_from_trial(selected_htrial)
        config_path = os.path.join(select_value, HTUNER_CONFIG_FILE)
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found in {selected_htrial}")

        exp_config = HTunerExpConfig.load_from_file(str(config_path))
        model = exp_config.lgn_model['lgn_model_type'].load_from_config(exp_config.lgn_model).to(device)
        model.load_weights_from_checkpoint(ckpt_weights_path)

        output = (f"Loaded experiment from "
                  f"{os.path.join(os.path.basename(selected_htrial), os.path.basename(ckpt_weights_path))}")

    return exp_config, model, output


def _find_checkpoint_from_trial(trial_path: str | os.PathLike) -> str | None:
    ckpt_path = os.path.join(trial_path, "best_model.ckpt")
    if os.path.exists(ckpt_path):
        return ckpt_path

    if not os.path.exists(os.path.join(trial_path, "checkpoints")):
        return None

    ckpt_path = os.path.join(trial_path, "checkpoints", "last.ckpt")
    if os.path.exists(ckpt_path):
        return ckpt_path

    ckpts = [ckpt for ckpt in os.listdir(os.path.join(trial_path, "checkpoints")) if ckpt.endswith(".ckpt")]
    if len(ckpts) == 0:
        return None
    return os.path.join(trial_path, "checkpoints", ckpts[-1])


def _create_labels_tables(labels: List[str] | List[int],
                          label_encoder: Optional[MultiLabelBinarizerRobust] = None) -> dbc.Table:
    if label_encoder is not None and len(labels) > 0 and isinstance(labels[0], int):
        labels = label_encoder.decode_labels(labels)

    labels_df = pd.DataFrame(labels, columns=["Ingredients"])
    labels_table = dbc.Table.from_dataframe(labels_df, **TABLES_PARAMS)

    return labels_table


def _create_preds_table(preds: np.ndarray, label_encoder: Optional[MultiLabelBinarizerRobust] = None) -> dbc.Table:
    best_preds = np.argsort(preds)[::-1][:20]
    preds_confidence = preds[best_preds]

    best_preds_decoded = label_encoder.decode_labels([best_preds])[0] if label_encoder is not None else best_preds
    preds_df = pd.DataFrame({"Ingredients": best_preds_decoded, "Confidence": preds_confidence})
    preds_df['Confidence'] = np.round(preds_df['Confidence'], 4)
    preds_table = dbc.Table.from_dataframe(preds_df, **TABLES_PARAMS)
    return preds_table


def _load_model():
    model = torch.load(MODEL_CACHE_PATH)
    if getattr(model, "conv_target_layer", None) is None:
        raise ValueError("The model does not have a convolution target layer for the visualization, "
                         "select another model")

    if getattr(model, "classifier_target_layer", None) is None:
        raise ValueError("The model does not have a classifier target layer for the visualization, "
                         "select another model")
    return model

#
#
# def _load_exp_from_upload(contents_string, upload_filenames) -> Any:
#     print(type(contents_string), "  ", contents_string[:30])
#     contents = base64.b64decode(contents_string)
#     print(type(contents),  "  ", contents[:30])
#     if upload_filenames.endswith(".yaml"):
#         print("yaml")
#         file = yaml.load(contents, Loader=yaml.BaseLoader)
#         print(file)
#
#     elif upload_filenames.endswith(".ckpt"):
#         print("ckpt")
#
#         tmp = torch.load(io.BytesIO(contents))
#         print(tmp.keys())
#     return None


#
# @callback(Output('carousel_images', 'items'),
#           Output("carousel_images", "active_index"),
#           Input('store_data_loader', 'data'), prevent_initial_call=True)
# def load_images(store_data_loader):
#     if store_data_loader is None or store_data_loader == {}:
#         raise PreventUpdate
#     dataset = LightImagesRecipesDataset.from_json(store_data_loader)
#     print(f"New dataset: {dataset}")
#
#     items = [{"key": str(i), "src": dash_get_asset_url(str(image_path)), "alt": f"Image {i}"}
#              for i, image_path in enumerate(dataset.images_paths)]
#     return items, 0
