from typing import Optional, List, Tuple
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_cam_on_image, show_factorization_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data_processing.labels_encoders import LabelEncoderInterface


def gradcam(model: torch.nn.Module,
            target_layer: torch.nn.Module,
            input_x: torch.Tensor,
            targets: Optional[List[int]] = None,
            imgs_show: Optional[torch.Tensor] = None,
            img_weight: float = 0.5,
            ) -> Tuple[List[np.ndarray], np.ndarray, List[int], torch.Tensor]:
    """
    Function that wraps the gradcam class, it performs the forward and backward pass with the model over the input,
    to compute the features maps and the gradients of the target layer,
    then it compute the weighted sum of the features map (gradcam).
    It returns the original images masked by the gradcam masks

    :param model: trained torch model to visualize
    :param target_layer: layer in the model to visualize
    :param input_x: batch or single image to visualize
    :param targets: target classes to visualize (if None, it will use the class with the highest score)
    :param img_weight: weight of the image respect to the gradcam mask (img_weight * img + (1-img_weight) * mask)
    :param imgs_show: images to show in the visualization (if None, it will use the input_x)
    :return: images with the gradcam masked, gradcam mask and the targets selected, and the output of the model
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    input_x_cf, input_x_cl = _manage_input(input_x)

    if imgs_show is None:
        imgs_show = input_x_cl
    else:
        _, imgs_show = _manage_input(imgs_show)

    if targets is not None:
        if len(targets) != len(input_x_cf):
            raise ValueError("The number of targets must be the same as the number of images")

        targets = [ClassifierOutputTarget(target) for target in targets]
    cam_masks = cam(input_tensor=input_x_cf, targets=targets)

    imgs = np.array([show_cam_on_image(x.cpu().numpy(), cam_mask, use_rgb=True, image_weight=img_weight)
                     for x, cam_mask in zip(imgs_show, cam_masks)]) / 255

    if targets is None:
        targets = torch.argmax(cam.outputs.detach(), dim=-1).cpu().numpy()
    else:
        targets = [target.category for target in targets]

    return imgs, cam_masks, targets, cam.outputs


def feature_factorization(model: torch.nn.Module,
                          target_conv: torch.nn.Module,
                          target_classifier: torch.nn.Module,
                          input_x: torch.Tensor,
                          label_encoder: Optional[LabelEncoderInterface] = None,
                          n_components: int = 5,
                          top_k: int = 1,
                          imgs_show: torch.Tensor = None,
                          img_weight: float = 0.5,
                          ) -> np.ndarray:
    """
    Function that extract the factorization components from some images, by targeting the last conv layer and the
    classification layer of the network.

    It wraps the DeepFeatureFactorization class, it performs the forward pass with the model over the input, to compute
    the features maps of the target layer, and the output of the classifier,
    then it computes the factorization of the features maps.
    It returns the original images masked by the factorization components with their labels.

    NB:
    - Concept è tabella dei o componenti in relazione ad ogni feature map del target layer (2 righe, 512 colonne) [nella docs è W]
    - Batch_explanations [nella docs H] esprime la relazione tra ogni pixel e ogni concetto dell'immagine
    - Concept_outputs è la classificazione dei concetti (corrisponde alla probabilità di una classe di essere in quel concetto)


    :param model: trained model to visualize
    :param target_conv: convolutional layer from which to extract the features
    :param target_classifier: classification layer to extract classes probabilities
    :param input_x: batch or single image to visualize
    :param label_encoder: label encoder to decode the labels (if provided)
    :param n_components: number of components to extract (notes: a high value can lead to missed convergence
    :param top_k: number of top classes to show in the labels
    :param img_weight: weight of the image respect to the factorization components
    :param imgs_show: images to show in the visualization (if None, it will use the input_x)
    :return: images with the factorization components masked
    """

    dff = DeepFeatureFactorization(model=model, target_layer=target_conv, computation_on_concepts=target_classifier)

    input_x_cf, input_x_cl = _manage_input(input_x)
    if imgs_show is None:
        imgs_show = input_x_cl
    else:
        _, imgs_show = _manage_input(imgs_show)

    concepts, batch_explanations, concept_outputs = dff(input_x_cf, n_components=n_components)
    concept_labels = _create_labels(concept_outputs, label_encoder=label_encoder, top_k=top_k)
    visualizations = np.array(
        [show_factorization_on_image(x.cpu().numpy(), batch_expl, concept_labels=concept_labels,
                                     image_weight=img_weight)
         for x, batch_expl in zip(imgs_show, batch_explanations)]) / 255

    return visualizations


def correct_legend_factor(img_factors, ratio=0.5):
    """Function that reduce the size of the legend in the factorization image"""
    img_len = int(img_factors.shape[1] / 2)
    img, legend = img_factors[:, :img_len, :], img_factors[:, img_len:, :]  # Split the image from the legend

    legend = _crop_to_box(legend, lines_value=0, lines_channel=0)  # Crop the legend
    legend = cv2.resize(legend, dsize=None, fx=ratio, fy=ratio)  # Resize the legend

    background_l = np.ones((img.shape[0], legend.shape[1], 3))
    background_l[:legend.shape[0], :legend.shape[1]] = legend
    return np.hstack([img, background_l])

def h_stack_imgs(*images: np.ndarray | torch.Tensor, img_sep=None) -> np.ndarray:
    """Function that stacks horizontally the images provided (by using a separator, and managing the channels)"""

    if len(images) == 1:
        img = images[0]
        return img if img.shape[0] == 3 else img.transpose(1, 2, 0)


    # manage type conversion
    images = [img.numpy() if isinstance(img, torch.Tensor) else img for img in images]
    # manage channel order to channel last
    images = [img.transpose(1, 2, 0) if img.shape[0] == 3 else img for img in images]

    if img_sep is None:
        img_sep = np.ones((images[0].shape[1], 10, 3), dtype=np.uint8)

    return np.hstack([item for tup in [(img, img_sep) for img in images[:-1]] for item in tup] + [images[-1]])


def _manage_input(input_x):
    """Function that converts the input images to a batch if needed, and changes the channel order if needed
    by returning the input in both formats (channel first and channel last)"""
    if isinstance(input_x, np.ndarray):
        input_x = torch.from_numpy(input_x)
    if isinstance(input_x, list):
        input_x = torch.Tensor(input_x)

    if len(input_x.shape) == 3:
        input_x = input_x.unsqueeze(0)

    if input_x.shape[1] == 3:
        input_x_cf = input_x
        input_x_cl = input_x.permute(0, 2, 3, 1)
    else:
        input_x_cf = input_x.permute(0, 3, 1, 2)
        input_x_cl = input_x

    return input_x_cf, input_x_cl


def _create_labels(concept_scores, label_encoder=None, top_k=2, char_sep=": ", real_model=True):
    concept_scores = torch.softmax(torch.from_numpy(concept_scores), dim=-1).numpy()  # convert output to probabilities
    concept_categories = np.argsort(concept_scores)[:, ::-1][:, :top_k]  # we take the index of the top k categories
    top_k_scores = np.take_along_axis(concept_scores, concept_categories,
                                      axis=1)  # we extract the scores of the best categories

    if label_encoder is not None:
        concept_categories = label_encoder.decode_labels(concept_categories)
    concept_labels = np.char.add(np.char.add(concept_categories.astype(np.str_), char_sep),
                                 top_k_scores.round(4).astype(np.str_))
    concept_labels = np.array(list(map(lambda x: "\n".join(x), concept_labels)))

    return concept_labels


def _crop_to_box(img_array, lines_value=0, lines_channel=0, border=20):
    """Used to crop around legend labels"""
    points = np.argwhere(img_array[:, :, lines_channel] == lines_value)
    min_x, max_x = points[:, 1].min(), points[:, 1].max()
    min_y, max_y = points[:, 0].min(), points[:, 0].max()
    return img_array[min_y - border:max_y + border, min_x - border:max_x + border]
