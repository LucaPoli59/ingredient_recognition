# Differentiable interpretability in DINOv2: gradients, tokens and spatial representations

**Creation date:** 2026-07-29

## Purpose of the note

This note preserves the machine learning reasoning behind an interpretability problem that arose with DINOv2: why a prediction can be computed correctly while Grad-CAM cannot, and why a Transformer network requires an explicit step from the sequence of tokens to a spatial representation.

It is not a code editing guide. The concrete case involves `DinoV2B14` and a multi-label classification of ingredients, but the principles apply to any Vision Transformer (ViT) used with gradient-based saliency methods.

## Prediction and explanation are different tasks

A prediction in inference evaluates a function:

$$y = f_\theta(x),$$

where $x$ is the image, $\theta$ are the model parameters, and $y$ are the logits. For multi-label classification, each component $y^c$ is the independent score of a class $c$; there is no softmax that makes the classes mutually exclusive.

A Grad-CAM explanation requires more than just $y^c$. It asks how that logit changes as an intermediate representation $A$ changes:

$$\frac{\partial y^c}{\partial A}.$$

Prediction is therefore a *forward pass* problem; Grad-CAM is a forward pass plus backward pass problem. It is possible that the former is perfectly valid while the latter is impossible or undefined in the autograd graph constructed for that inference.

## Grad-CAM as a projection of the gradient onto a feature map

For a CNN, you typically choose a layer with activations:

$$A \in \mathbb{R}^{B \times C \times H \times W}.$$

For a class $c$, Grad-CAM calculates a weight per channel by spatially averaging the gradient:

$$\alpha_k^c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}
\frac{\partial y^c}{\partial A_{kij}}.$$

The relevance map is then:

$$L_{\mathrm{GradCAM}}^c = \operatorname{ReLU}\left(\sum_{k=1}^{C}\alpha_k^c A_k\right).$$

The intuition is precise: a channel receives a high weight when increasing its activations increases the logit of the observed class. The weighted combination preserves the spatial index $(i,j)$, so it can be superimposed on the image.

This definition implies two requirements:

1. there must be a gradient of the logit with respect to the chosen layer;
2. the activations must be able to be interpreted as a spatial map, either directly or via a known transformation.

## Freezing, autograd and activation gradients

Freezing a backbone means setting `requires_grad=False` on its parameters. In training, this choice prevents the gradient from being accumulated in the frozen weights and the optimizer from modifying them.

Furthermore, if the input $x$ requires no gradient, PyTorch has no reason to build a differentiable graph for backbone operations. Its intermediate activations are therefore not differentiable with respect to the logit, despite being numerically available.

This is not a model error—it is the correct optimization for normal inference. It becomes a limitation only when an explanatory method requires a backward pass.

It is useful to distinguish the concepts:

| Concept | Meaning |
| --- | --- |
| Frozen parameter | The weight receives no gradient and is not updated. |
| Differentiable input | You can calculate how the output changes with respect to the input and activations along the way. |
| Gradient for Grad-CAM | It is a temporary gradient used to estimate relevance, not an update of the weights. |

Making the input differentiable is enough to reopen the gradient path through operators with frozen weights. This allows you to calculate $\partial y^c / \partial A$ without transforming the model into a trainable model and without performing an optimization update.

## From patch grid to Transformer sequence

A CNN explicitly maintains `H×W` spatial axes along the network. A ViT instead transforms them into a sequence.

With patch size $P=14$ and a square image $H=W=224$, the number of patches is:

$$N = \frac{H}{P}\frac{W}{P} = 16 \cdot 16 = 256.$$

Each patch is projected into an embedding of size $D=768$. The backbone then works on patch tokens:

$$E \in \mathbb{R}^{B \times N \times D}.
$$

DINOv2 ViT-B/14 with registers adds one class token and four register tokens to the sequence:

$$Z \in \mathbb{R}^{B \times (1+4+256) \times 768}
= \mathbb{R}^{B \times 261 \times 768}.$$

The class token is a learned vector used to aggregate global information. Register tokens are learned memory slots: they do not correspond to image regions and help prevent the model from using background patches as working memory. Both participate in self-attention, but they have no $(i,j)$ coordinate in the patch grid.

## Why a sequence is not a heatmap

A Transformer layer normally returns a `[B, T, D]` tensor containing $T$ tokens. Classical Grad-CAM, by contrast, assumes channels and spatial coordinates: `[B, C, H, W]`.

To obtain a saliency map from a ViT, the spatial correspondence must be made explicit:

```text
[B, 261, 768]
  → remove the class token and register tokens
[B, 256, 768]
  → 256 = 16 × 16
[B, 768, 16, 16]
```

This transformation is not merely a shape convention: it states that each of the 256 remaining tokens represents a specific image patch. After reshaping, the 768 embedding dimensions become the feature-map channels, and Grad-CAM can average gradients across the 256 positions.

If the class token or register tokens were included in the reshape, they would be assigned artificially to image positions and the heatmap would no longer have correct spatial semantics.

## Where to hook Grad-CAM in a Transformer

The selected layer must satisfy three properties:

1. it must be deep enough to encode semantic information relevant to the class;
2. it must lie on the differentiable path to the logit;
3. it must expose patch tokens that can be reconstructed as a grid.

For DINOv2, the LayerNorm before self-attention in the final block (`norm1`) is a natural choice: its features enter the final attention operation and influence the network output. The backbone's final normalization is less suitable as a conceptual hook when it is invoked multiple times to extract several intermediate levels: a single module may then produce multiple activations during the same forward pass, making the association between activation, gradient, and semantic level ambiguous.

A ViT heatmap is not identical to a CNN heatmap. Its native resolution is the patch-grid resolution (`16×16` for a 224-pixel input), so upsampling to `224×224` makes the visualization easier to read but does not create new information below the size of a patch.

## The `_lc` head and multi-level representations

The DINOv2 `_lc` wrapper used in the project performs linear classification on a composite representation. It does not use only the final class token; it concatenates:

$$h = [c_9; c_{10}; c_{11}; c_{12}; \operatorname{mean}(E_{12})]
\in \mathbb{R}^{3840}.$$

The terms $c_9,\ldots,c_{12}$ are the class tokens from the final four blocks; each is a 768-dimensional vector. $\operatorname{mean}(E_{12})$ is the mean of the patch tokens from the final block and is also 768-dimensional. The resulting classification is:

$$y = Wh + b, \qquad W \in \mathbb{R}^{C \times 3840}.$$

This construction combines global representations from several deep levels with a summary of the features distributed across patches. For a multi-label task, each row of $W$ corresponds to an ingredient and produces its logit.

## Deep Feature Factorization and dimensional compatibility

DFF applies non-negative factorization to spatial activations. If the map is `[B, 768, 16, 16]`, each concept produced by the factorization lies in channel space:

$$z_q \in \mathbb{R}^{768}.$$

To assign labels to concepts, DFF applies a classifier to $z_q$. This reveals a fundamental difference between a “local feature” and the “actual head input”: the `_lc` head expects $h \in \mathbb{R}^{3840}$, whereas a patch concept has 768 dimensions.

The portion of the head associated with the patch mean is:

$$W_{\mathrm{patch}} = W[:, -768:] \in \mathbb{R}^{C \times 768}.$$

The projection $W_{\mathrm{patch}}z_q+b$ does not reproduce full inference: it measures only how the patch concept aligns with the segment of the final decision that depends on the patch summary. This is therefore an appropriate interpretation for labeling DFF concepts, but it is not a replacement for the original head.

## Interpretation considerations

- Grad-CAM shows the logit's local sensitivity, not causal proof that the region actually contains the ingredient.
- A class may receive high relevance for context, tableware, or visual composition if those elements correlate with the class in the training data.
- For ingredients that are not visible, are mixed in, or are covered, saliency cannot provide direct visual evidence; the model may rely on learned correlations.
- Saliency depends on the selected target: two ingredients predicted from the same image can produce different maps.
- The DFF explanation based on $W_{\mathrm{patch}}$ describes the contribution of patch concepts, not the complete contribution of the four class tokens.

## Summary

Grad-CAM requires activation gradients, whereas normal inference with a frozen backbone may perform only the forward pass without constructing a differentiable path. A ViT also requires patch tokens to be converted explicitly back into a spatial grid, excluding global tokens such as CLS and registers.

In the DINOv2 `_lc` classifier, the final decision uses a 3,840-dimensional multi-level feature. Techniques based on patch concepts instead operate in the 768-dimensional patch space; interpreting them correctly requires distinguishing the local contribution of patches from the complete global classification.
