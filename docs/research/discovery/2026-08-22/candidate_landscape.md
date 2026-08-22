# Candidate landscape for a reference learnability selector

**Created:** 2026-08-22
**Last updated:** 2026-08-22

## Research question and boundary

Which model and adaptation families could provide a scientifically defensible
reference instrument for deciding whether an ingredient label is learnable from
the frozen `v5` image task?

The target remains one RGB food image mapped to continuous scores for every
label in a fixed recipe-level multi-label vocabulary. The selector must
eventually support per-label train and validation trajectories. Detection,
segmentation, retrieval, open-ended generation, and zero-shot prompting are
adjacent evidence, not interchangeable targets.

This document catalogs candidate families. It does not rate them, freeze an
eligible set, or choose `M_ref`. Those decisions require the R0.2 repository
inventory and the R1 rubric.

## A model is a protocol, not only a backbone

A reference-selector candidate must be identified by the following tuple:

\[
M =
(\text{pretraining},\ \text{backbone},\ \text{adaptation},\
\text{head},\ \text{loss},\ \text{input policy}).
\]

Naming only “DINO,” “ResNet,” or “SigLIP” leaves the scientific measurement
undefined. The same checkpoint answers materially different questions under a
linear probe, full fine-tuning, or text-prototype classifier.

### Pretraining axis

| Code | Family | Information available before `v5` training | Claim boundary |
| --- | --- | --- | --- |
| P0 | Supervised visual | Human class labels such as ImageNet categories | Learnability conditional on a supervised visual prior |
| P1 | Visual self-supervision | Image structure and invariances without class or caption supervision | Learnability conditional on a general visual prior |
| P2 | Generic vision-language | Web image-text associations and language semantics | Learnability conditional on visual and semantic web priors |
| P3 | Food-domain supervised or multimodal | Dish, recipe, ingredient, instruction, or food-pixel supervision | Learnability conditional on domain knowledge close to the target |
| P4 | Random initialization | Only local images and labels | Learnability from the declared local optimization budget; expensive and not automatically a better selector |

P4 is scientifically interpretable but can understate recognizability when the
available data or budget cannot learn general visual primitives. Conversely,
P2 and P3 can overstate what the local dataset teaches. The project must choose
which conditional question is useful; no row is universally correct.

### Adaptation axis

| Code | Mode | What success demonstrates | Principal limitation |
| --- | --- | --- | --- |
| A0 | Frozen encoder + learned linear/sigmoid head | Label information is linearly accessible in the fixed representation | Misses evidence that requires representation adaptation |
| A1 | Frozen spatial features + learned class-query head | Label-specific evidence is accessible after spatial re-pooling | Changes the head capacity and may exploit shared label structure |
| A2 | Partial or parameter-efficient fine-tuning | The representation can adapt within a constrained budget | Results depend on which blocks/parameters are trainable |
| A3 | Full fine-tuning | The end-to-end model can learn the label under the declared budget | Highest compute and optimization sensitivity |
| A4 | Text-defined prototype or prompt scoring | Image features align with the ingredient name or description | Directly imports downstream label semantics and prompt choices |

The modes are not interchangeable replications. If more than one is used, each
must remain a separately named selector protocol.

### Head axis

| Code | Head | Benefit | Measurement risk |
| --- | --- | --- | --- |
| H0 | Independent learned logits after pooling | Low cost, direct per-label scores, clean continuity control | Global pooling may erase small or spatially separated evidence |
| H1 | Learned spatial label queries | Per-label access to local feature maps | Head capacity becomes part of learnability; attention is not proof of localization |
| H2 | Text-initialized queries or prototypes | Makes label semantics available and may help tail labels | Measures transferred semantic alignment in addition to local visual learning |
| H3 | Label-dependency graph or label-state transformer | Exploits real ingredient co-occurrence | Can predict hidden labels or cuisine priors without ingredient pixels |
| H4 | Autoregressive/set decoder | Models output-set structure | Adds ordering/decoding decisions and can amplify co-occurrence |

H0 is the required interpretive control even if a later structured head is
considered. H2–H4 need image-free, shuffled-image, or prior controls before
their gains can be described as visual evidence.

## Candidate-family catalog

### Supervised convolutional continuity

**Representatives:** pretrained ResNet-50; DenseNet-121 as a continuity check.

These models use mature supervised image-classification priors and map cleanly
to H0. The repository already contains torchvision and custom ResNet/DenseNet
implementations, so they provide historical continuity and the lowest
integration uncertainty.

Their main limitation is not age alone. Global pooling and a supervised
single-label pretraining objective may make weak, small, or multi-component
ingredient cues less accessible. DenseNet does not create a distinct
measurement principle from ResNet, so carrying both into later comparison
requires an explicit continuity reason rather than treating architecture count
as evidence breadth.

**R0.1 disposition:** retain the family; R0.2 must identify one continuity
representative and verify its pretrained and random-initialization semantics.

### Modern efficient convolutional families

**Representatives:** ConvNeXt V2 Atto–Tiny; EfficientNetV2-S.

ConvNeXt V2 provides a modern convolutional family across small scales and can
use fully convolutional masked-autoencoder (FCMAE) pretraining. It is therefore
both an architecture alternative and a P1 pretraining alternative. Its official
repository was archived in 2025, so a maintained library implementation and
checkpoint provenance must be preferred over direct dependence on the archived
training repository.

EfficientNetV2 was designed for parameter and training efficiency and supplies
a supervised efficient-CNN control. It does not introduce a new semantic prior,
but may offer a better cost-to-capacity point than ResNet on the 8 GB device.

**R0.1 disposition:** retain both at family level; R0.2 should inventory compact
variants already supplied by the project's maintained PyTorch stack and avoid a
broad sweep of near-equivalent CNNs.

### Hierarchical and patch-based supervised transformers

**Representatives:** Swin V2 Tiny; a compact ViT/DeiT-style baseline when a
maintained checkpoint exists.

Hierarchical/windowed transformers preserve spatial feature maps while limiting
the full-attention cost, making them plausible for multi-component food images.
A plain ViT separates the transformer architecture from DINO-like
self-supervision.

Architecture alone does not define the selector: ImageNet-supervised Swin,
masked-image-pretrained Swin, and DINO-style ViT have different priors. The
family is useful only if R0.2 can retain those distinctions and expose spatial
features without creating a one-off training path.

**R0.1 disposition:** retain one compact hierarchical/patch-based
representative for inventory; do not assume it must enter R2.

### General visual self-supervision

#### DINOv2

DINOv2 models were pretrained on a curated set of 142 million unlabeled images
and expose both global and patch features. The official family includes smaller
and larger ViT variants, with and without register tokens. The repository
currently wraps a ViT-B/14-with-registers path, but the broad candidate is the
protocol, not only that local variant.

- A0 tests linear accessibility of ingredient information.
- A1 tests whether patch features contain ingredient-specific local evidence.
- A2/A3 test adaptation, at a larger compute and configuration cost.

**Interpretation:** P1 adds general visual knowledge but not text-label
semantics. This makes DINOv2 attractive for a selector intended to be visually
sensitive while limiting direct language transfer.

**R0.1 disposition:** retain; R0.2 must compare feasible S/B variants, correct
checkpoint and transform semantics, and genuine freeze/unfreeze modes.

#### DINOv3

DINOv3 scales visual self-supervision to 1.7 billion images and releases ViT and
distilled ConvNeXt families, including compact variants intended for constrained
use. Its emphasis on high-resolution dense features is relevant when ingredient
evidence is local.

The weights require explicit access and use a dedicated DINOv3 licence. The
newer dependency path and the absence of a current local wrapper are material
reproducibility risks, not reasons to infer poor scientific fit.

**R0.1 disposition:** retain behind an access, licence, dependency, and
single-GPU smoke gate.

#### MAE, FCMAE, and I-JEPA

Masked Autoencoders (MAE) reconstruct masked image content; ConvNeXt V2's FCMAE
adapts masked reconstruction to convolutional encoders. I-JEPA predicts target
representations in latent space rather than reconstructing pixels.

These objectives broaden the P1 family beyond DINO self-distillation, but
candidate availability matters:

- MAE/FCMAE offer compact or standard variants with published checkpoints;
- the official I-JEPA release is centered on ViT-H/ViT-g checkpoints, making
  end-to-end adaptation difficult under 8 GB;
- none has project-specific evidence that its objective is more sensitive to
  ingredient cues.

**R0.1 disposition:** retain MAE/FCMAE as comparison concepts; keep I-JEPA as a
research-only lead unless R0.2 finds a credible compact maintained checkpoint.

### Generic vision-language encoders

**Representatives:** CLIP ViT-B; SigLIP/SigLIP 2 ViT-B.

CLIP learns image-text alignment from web pairs. SigLIP replaces global
softmax contrast with a pairwise sigmoid objective. SigLIP 2 additionally uses
captioning, self-distillation, masked prediction, online curation, and releases
multi-resolution/native-aspect variants.

Two downstream protocols must be separated:

1. **P2+A0/A2+H0:** use only the pretrained image encoder during local
   multi-label training and learn ordinary logits;
2. **P2+A4+H2:** score or initialize labels using ingredient text.

The first still contains language-shaped visual features, but local labels
determine the output head. The second can recognize a concept because its name
was learned from web text and must be described as semantic-transfer evidence.
Zero-shot prompt scores can be useful diagnostics, but they do not produce the
same learning trajectories as a locally trained selector.

**R0.1 disposition:** retain one feasible P2 image-encoder transfer path and a
separately labelled text-prototype diagnostic; inventory SigLIP 2 as the
current-family representative and CLIP as the continuity representative.

### Food- and recipe-domain pretraining

#### Food2K supervised representation

Food2K contains over one million images across 2,000 food categories and reports
transfer to several food-vision tasks. A backbone pretrained on Food2K could
encode fine-grained food appearance better than a general ImageNet model.

Its labels are mainly dish categories, not ingredient presence. It can
therefore transfer plating, cuisine, and recipe-family cues that correlate with
hidden ingredients. Public checkpoint availability, terms, taxonomy, and
source-image overlap with the local web corpus require explicit verification.

**R0.1 disposition:** conditional family; not eligible until R0.2 resolves a
traceable checkpoint and provenance path.

#### Recipe1M+, VLPCook, and related cross-modal food encoders

Recipe1M+ aligns more than one million recipes with millions of food images.
VLPCook explicitly pretrains on food images and structured recipe text,
including ingredient-related information, and publishes code and checkpoints
for retrieval.

These encoders are highly relevant but also highly target-adjacent. A positive
label may become accessible because the pretrained model learned recipe text,
not because the ingredient is visible. Web-recipe/image overlap with Yummly
could also create direct or recipe-family contamination and must be audited
rather than assumed absent or present.

**R0.1 disposition:** conditional P3 family. Any later use must record the
pretraining corpus, overlap audit, vocabulary mapping, frozen/fine-tuned mode,
and the claim “learnable under recipe-domain pretraining.”

#### FoodSeg103/ReLeM

FoodSeg103 provides pixel-level annotations for visible food components.
ReLeM combines food image and recipe information to pretrain segmentation
encoders. This line is unusually aligned with the project's notion of direct
visual observability, because the source task requires localized pixels.

The transfer is not clean by default: the taxonomy differs, ReLeM also uses
Recipe1M+ semantics, and a segmentation encoder/head does not directly satisfy
the canonical closed-vocabulary multi-label contract.

**R0.1 disposition:** retain as a conditional visible-evidence prior and as
supporting evidence for spatial heads; require checkpoint, licence, taxonomy,
overlap, and adapter feasibility checks.

#### Emerging generative food VLMs

Food-R1 is a 2026 food-specific generative vision-language system trained for
multiple food and nutrition tasks. It is evidence that domain-specific food VLM
pretraining is becoming available, but its released checkpoint is large and its
generation interface does not natively provide the required per-label
optimization trajectories.

**R0.1 disposition:** catalog as an emerging diagnostic/teacher direction, not
as a credible primary `M_ref` path under the present compute and output
contract.

### Multi-label head families

#### Independent pooled head

The existing pooled-linear-logit design is the mandatory H0 control. It is
cheap, exposes continuous scores, and minimizes the chance that co-occurrence
machinery defines a label as learnable.

It can still exploit context through the backbone, so H0 does not prove direct
ingredient visibility. Observability review and mechanism controls remain
required.

#### ML-Decoder and Query2Label

Both use learned label queries to extract class-related spatial features.
ML-Decoder reduces the cost of a full transformer decoder through group
decoding, while Query2Label is the fuller label-query reference.

For ingredient selection, their value is not a published aggregate gain. They
test whether a label looks unlearnable only because global pooling discarded a
small region. Learned queries (H1) should be distinguished from word-derived
queries (H2).

**R0.1 disposition:** retain ML-Decoder as the compact H1 representative and
Query2Label as the fuller reference family. R0.2 must determine whether the
backbone exposes compatible spatial features and whether the head can preserve
per-label metrics and dashboard hooks.

#### Dependency-aware heads

C-Tran jointly represents labels and visual features; graph models such as
ML-GCN use label relationships. Inverse Cooking and related set decoders model
ingredient-set dependencies more directly.

These methods may improve recipe-level prediction, especially for hidden or
contextual ingredients. That same property makes them risky as the sole
recognizability selector: an ingredient can look learnable because another
label or cuisine prior predicts it.

**R0.1 disposition:** retain as later comparative or mechanism candidates, not
as a sole H0-free selector. Any promotion requires image-free and label-prior
controls.

## Cross-domain protocol checklist

The following checklist is reusable whenever a pretrained model is used to
select learnable targets in this or another domain:

1. name the source data, objective, checkpoint, licence, and checksum;
2. state whether the target labels or their names could appear in pretraining;
3. declare frozen, partial, parameter-efficient, or full adaptation;
4. declare whether downstream label text enters the model;
5. declare independent, spatial-query, dependency-aware, or generative output;
6. require continuous per-label scores and a fixed label order;
7. separate optimization evidence from held-out generalization;
8. add non-visual and shuffled-image controls when semantic/dependency priors
   are present;
9. audit source overlap when pretraining uses related web or domain corpora;
10. phrase the conclusion as conditional on the complete protocol.

## Handoff set for R0.2

The following is a non-ranked inventory set. Inclusion means “inspect the
credible project path,” not “promote to R2”:

| Role | Representative family to inventory | Required boundary |
| --- | --- | --- |
| Continuity supervised CNN | ResNet-50; DenseNet only if needed for historical continuity | P0 + H0 |
| Compact modern CNN | ConvNeXt V2 compact and/or EfficientNetV2-S | Keep supervised and FCMAE checkpoints distinct |
| Compact transformer | Swin V2 Tiny or equivalent maintained ViT | Record P0 versus P1 initialization |
| Visual SSL continuity | DINOv2 ViT-S/B, with register choice explicit | A0 first; A2/A3 separate |
| Current visual SSL | compact DINOv3 ViT or ConvNeXt | Access/licence/dependency gate |
| Generic vision-language | SigLIP 2 ViT-B; CLIP continuity | Learned H0 head separate from text H2 |
| Food-domain transfer | Food2K, Recipe1M+/VLPCook, FoodSeg/ReLeM | Provenance, overlap, terms, and taxonomy gate |
| Spatial multi-label head | ML-Decoder; Query2Label reference | H1 learned queries; no silent word queries |
| Dependency head | C-Tran/graph/set decoder | Supplemental only unless controls isolate image evidence |
| Emerging food VLM | Food-R1 or successor | Diagnostic/teacher only under current output and compute contract |

R0.2 should remove representatives that lack a reproducible checkpoint,
maintained integration path, required instrumentation, or feasible compute. It
must not remove a family merely because it is not currently implemented.

## Uncertainty and open questions

- Does the intended selector prioritize sensitivity to any useful predictive
  signal, or specificity to direct visual evidence? The later R1 rubric must
  make this trade-off explicit.
- Is a frozen probe sufficiently adaptive for the target, or would it
  systematically reject labels that become learnable after modest fine-tuning?
- Can a spatial-query head be standardized across CNN, DINO, and
  vision-language backbones without changing the comparison more than the
  representation itself?
- Can food-domain checkpoints be acquired with traceable terms and an adequate
  Yummly/Recipe1M overlap audit?
- Which compact variants leave enough 8 GB headroom for per-label logging and
  the full single-seed configuration panel?

## Related evidence

- [Primary-source catalog](source_catalog.md)
- [Discovery index and synthesis](README.md)
- [Reference-selector plan](../../../plans/reference_selector_research.md)
- [Label-learnability research](../../topics/label_learnability/learnability_assessment.md)
