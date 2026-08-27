# Reference-selector candidate and instrumentation inventory

**Created:** 2026-08-22
**Last updated:** 2026-08-27
**R0.2 status:** Complete

> **Workspace provenance note (2026-08-27):** the original static audit was
> performed on the Windows `feature/ingredient_selection` workspace and was
> committed as `993aa4f` before being merged into the WSL development branch.
> Its environment observations and code-state statements are retained below as
> snapshot evidence. They are not the current WSL runtime contract: the WSL ML
> environment is operational and the merged DINOv2 wrapper no longer enforces
> the historical physical batch cap of 32. R1 must use the current source and
> repeat its bounded environment/resource smoke checks; the current model
> contract remains authoritative in
> [`implementation_details/models.md`](../../../implementation_details/models.md).

## Purpose and decision boundary

This record maps the broad candidate landscape to the repository as it exists
on 2026-08-22. It answers whether a candidate has a credible route to the
frozen `v5` multi-label task, the Phase 3 evidence profile, and the available
8 GB development GPU.

It does **not** select `M_ref`, freeze the R1 rubric, or claim that a model is
feasible because a paper reports a compact parameter count. The dispositions
below determine which paths are sufficiently concrete to enter the next
eligibility discussion. Candidate comparison remains downstream of the
predeclared R1 gates.

Current behavior is authoritative in the source code and in the
[model implementation contract](../../../implementation_details/models.md).
This dated document is an inventory snapshot and should not be silently
updated to describe later implementations.

## Verification method

The inventory combined:

- direct inspection of the `v5` DataModule, label encoder, `BaseModel`, current
  wrappers, Lightning metric path, loggers, checkpoint callbacks, training
  entry points, dependency declarations, and retained experiment artifacts;
- a read-only environment check on the available workstation;
- the primary-source catalog retained by R0.1; and
- current official model repositories, model cards, and library documentation
  for dependency, checkpoint, access, and licence facts.

No new model was downloaded, instantiated, trained, or evaluated. The active
Python environment lacks `lightning`, does not include `timm`, `transformers`,
`open_clip`, or `big_vision`, and still exposes the known NumPy 2.1 / compiled
NumPy 1.x ABI warning when Torch is imported. These conditions prevent a valid
runtime smoke comparison but do not prevent static contract inspection.

The workstation reports an NVIDIA GeForce RTX 4060 with 8,188 MiB VRAM. All
candidate-specific compute statements below are therefore either historical
project evidence, official model metadata, or an explicit unverified estimate.

## Frozen task and shared interface

### Data contract

New runs default to `ingredients_target_v5_metadata.json` and
`feature_label="ingredients_target"`. The strict training-fitted
`MultiLabelBinarizer` produces a fixed 165-label output without `<UNK>` and
serializes both its ordered `classes` and its `encode_map`. The same fitted
encoder is then used for validation and test data.

`ImagesRecipesBaseDataModule` provides one RGB image and a 165-dimensional
multi-hot target. It uses the frozen train/validation/test metadata, the shared
`imgs/standard` image store, shuffled train batches, and non-shuffled
validation batches. The standard training path calls `fit`, so candidate
selection can remain train/validation-only; the test and `predict` loaders must
not be invoked by the selector workflow.

### Model and loss contract

The maintained model interface expects a square input and each current model
returns `num_classes` continuous logits without a final sigmoid. The Lightning
module applies `BCEWithLogitsLoss`, optionally with train-derived `pos_weight`,
and passes logits directly to TorchMetrics. Model-owned train and validation
transforms are injected into the DataModule during reconstruction.

This contract directly supports a pooled independent head (`H0`) and either a
frozen or fully trainable backbone. It does not yet define:

- a common pooled-feature or spatial-feature output interface;
- parameter groups for partial or parameter-efficient fine-tuning;
- non-square or native-aspect inputs; or
- a standard adapter for label-query, dependency, or text-prototype heads.

Those omissions are material for `A1`, `A2`, and `A4` protocols and for
SigLIP 2 NaFlex. They are not blockers for fixed-resolution `H0` candidates.

## Instrumentation audit

The current instrumentation is shared by all model wrappers. A new backbone
would inherit the same gaps unless the common training path is upgraded first.

| Required Phase 3 evidence | Current state | R0.2 verdict |
| --- | --- | --- |
| Ordered labels | The encoder stores ordered classes and index mapping in DataModule hyperparameters and full checkpoints. Per-label metric column names contain only numeric indices. | **Partial.** The order is recoverable, but every selector artifact must carry or link an explicit immutable index-to-label manifest. |
| Train AP trajectory per label | Average precision is not imported or configured. | **Missing.** |
| Validation AP trajectory per label | Average precision is not imported or configured. | **Missing.** |
| Fixed-policy F1 per label | A per-label F1 configuration exists but is disabled by default. When enabled, columns are index-based and no project-level threshold policy is serialized. | **Partial and unsuitable as-is.** |
| Continuous scores and targets | Logits exist in memory and `predict_step` returns them, but the default prediction limit is one batch and no maintained callback writes selector score/target artifacts. | **Missing as a reproducible run artifact.** |
| Loss and transform provenance | Loss class, weighting flag, model options, DataModule options, and Python transform callables are encoded in hyperparameters/checkpoints. | **Partial.** Callable serialization is project-specific, and external checkpoint preprocessing/revisions are not pinned independently. |
| Data provenance | Metadata filename, data path, feature field, and label encoder are stored. | **Partial.** The run does not persist metadata checksums or a frozen split/vocabulary manifest identifier. |
| Code and environment provenance | Not captured by the training path. | **Missing.** Git revision/dirty state, dependency versions, CUDA/cuDNN state, device identity, and external checkpoint checksum/revision are required. |
| Declared seed and deterministic state | No canonical seed is configured; `cudnn.benchmark=True`; no deterministic setting is recorded. | **Missing.** The one-seed policy cannot currently be reproduced or honestly described. |
| Epoch-level scalar logging | CSV, TensorBoard, and offline W&B loggers are maintained; checkpoints monitor validation loss. | **Available**, but insufficient for the decision profile. |
| Resource evidence | The DINOv2 wrapper declares a physical batch cap of 32 and the trainer can use gradient accumulation. No generic peak-VRAM/time probe is persisted. | **Partial.** Hard-coded limits and paper FLOPs are not hardware measurements. |

### Consequence

No current or newly wrapped candidate can pass the intended R1 evidence gate
without a common selector-observability layer. This is a cross-cutting
engineering prerequisite, not evidence against any architecture.

A proportionate maintained implementation must at least produce:

1. a run manifest with label order, frozen data identifiers, seed,
   deterministic-state declaration, code/environment identity, device, exact
   initialization/checkpoint identity, transforms, loss, optimizer, schedule,
   and trainable-parameter policy;
2. epoch-level train and validation AP for every named label;
3. F1 only under the later frozen validation-derived policy, with the policy
   identifier stored alongside the values;
4. bounded raw-score/target artifacts sufficient to audit and recompute the
   declared diagnostics without using the test split; and
5. a common smoke report containing physical batch size, gradient
   accumulation, precision, peak allocated/reserved VRAM, and elapsed time.

R1 must freeze the exact artifact scope. In particular, it should avoid
requiring every train logit for every epoch when streamed AP states and bounded
checkpoint score artifacts provide equivalent auditability at much lower
storage cost.

## Existing implementation paths

### Torchvision ResNet-50 — `P0 + A3 + H0`

**Path.** `src/models/resnet.py::Resnet50` uses the declared torchvision 0.23
dependency, `ResNet50_Weights.DEFAULT`, weight-compatible transforms, a
165-logit replacement head, standard serialization, and the shared BCE path.
No new library is required.

**Evidence.** Numerous retained ResNet experiments prove that the family has
run through the project pipeline on earlier vocabularies. They do not
constitute a `v5` smoke test or a recorded peak-VRAM measurement on the current
environment.

**Disposition.** **Verified R1 intake path.** It is the lowest-risk supervised
continuity protocol. Before R2 or any campaign it still needs the common
instrumentation, an exact torchvision weight identifier rather than an
unqualified future `DEFAULT`, a measured `v5` resource smoke test, and the
declared one-seed contract.

### Torchvision DenseNet-121 — `P0 + A3 + H0`

**Path.** A wrapper and official torchvision weights exist, but the current
constructor stores its weights in a local variable while the transform
properties read `self.tr_weights`. Accessing the model transforms therefore
raises `AttributeError` before a normal training reconstruction can complete.

**Disposition.** **Do not carry as an R1 representative in the current
state.** The bug is small, but DenseNet supplies no distinct measurement
principle relative to the supervised CNN control. It should return only if
historical continuity provides a concrete reason, after the wrapper is fixed
and smoke-tested.

### DINOv2 ViT-B/14 with registers, frozen — `P1 + A0 + H0`

**Path.** `src/models/dinov2.py::DinoV2B14` loads
`dinov2_vitb14_reg_lc` from the official repository through PyTorch Hub,
replaces the ImageNet classifier with 165 learned logits, freezes the DINO
backbone by default, and uses DINO-specific square transforms. The official
DINOv2 hub path requires only PyTorch and its code is Apache-2.0.

**Verified defects and risks.** The hub call tracks the repository default
branch rather than a pinned revision. The serialized `pretrained` flag is
ignored by the constructor, which always asks the hub entry point for its
default pretrained model. Neither upstream revision nor downloaded checkpoint
checksum is retained. Only the B/14-register variant is wrapped, so S/B and
register choices are not a maintained configuration axis.

**Evidence.** `experiments/dummy/dummy_dino_experiment/trial_2` retains a
50-epoch, frozen-backbone run on an earlier 183-output task with logical batch
128 and the wrapper's physical cap of 32. This is useful integration evidence,
but its hardware identity, external checkpoint revision, raw scores, AP, seed,
and `v5` contract were not recorded.

**Disposition.** **Verified R1 intake path after bounded reproducibility
repair.** Freeze mode, architecture, register policy, backbone rather than
upstream ImageNet head, upstream revision, local checkpoint checksum, and
preprocessing must become explicit. Full fine-tuning is a separate `A3`
candidate and cannot inherit the frozen-probe compute claim.

## Maintained-library candidates without local wrappers

Torchvision 0.23 officially supplies pretrained ConvNeXt, EfficientNetV2,
Swin Transformer, and Vision Transformer classifiers. Its published weight
table reports 28.6 M parameters / 4.46 GFLOPs for ConvNeXt Tiny, 21.5 M / 8.37
GFLOPs for EfficientNetV2-S, and 28.4 M / 5.94 GFLOPs for Swin V2 Tiny. These
figures compare model scale; they do not prove training memory on the project
GPU.

Each of the following can reuse the existing ResNet-wrapper pattern: construct
the official weights, replace the classifier, expose the existing dashboard
hooks, serialize the exact weight enum, and pass logits to the shared BCE
module. None requires a new third-party dependency.

| Candidate protocol | Compatibility and boundary | R0.2 disposition | Missing engineering |
| --- | --- | --- | --- |
| Torchvision ConvNeXt Tiny v1, `P0 + A3 + H0` | Square fixed-resolution inputs and pooled logits fit the current contract. It is **not** ConvNeXt V2 or FCMAE. | **Verified maintained-library option.** | Add and test a wrapper; record exact weights/transforms and measured 8 GB behavior. |
| EfficientNetV2-S, `P0 + A3 + H0` | Same canonical output with a smaller parameter count but a higher official inference FLOP estimate than ConvNeXt Tiny. | **Verified maintained-library option.** | Add and test a wrapper; decide whether its weight-native resolution is affordable and scientifically comparable. |
| Swin V2 Tiny, `P0 + A3 + H0` | Provides a compact hierarchical transformer control while retaining square inputs and independent logits. | **Verified maintained-library option.** | Add and test a wrapper, spatial hook, exact transforms, and measured full-fine-tuning memory. |

**Reduction rule for R1.** These three are credible implementation choices,
not a requirement to carry three near-redundant supervised models. R1 should
predeclare how one modern supervised architecture representative is chosen, or
why both a modern CNN and a supervised transformer are necessary. The choice
must precede candidate-specific performance inspection.

The exact ConvNeXt V2/FCMAE representative from R0.1 is **deferred**. It is not
available in the declared torchvision stack, `timm` is absent, and the official
training repository is archived. Torchvision ConvNeXt Tiny v1 is a maintained
supervised-CNN alternative, not a silent substitute for V2 self-supervision.

## Conditional external candidates

### Compact DINOv3 — `P1 + A0 + H0`

The official release includes ViT-S/16 (21 M parameters) and distilled
ConvNeXt-Tiny backbones. It supports either a local official repository with
explicit checkpoint paths or Hugging Face Transformers 4.56 and later. Model
weights require an access request and both code and weights use the dedicated
DINOv3 licence.

The project currently has neither a wrapper, a declared DINOv3/Transformers
dependency, an accepted local weight, nor a checksum. Frozen compact features
are plausible on 8 GB, but this has not been measured with a 165-logit head and
the required artifacts.

**Disposition.** **Conditional R1 intake path.** Promote only after weight
access and licence acceptance are recorded and one compact variant has a
pinned, checksummed, cache-independent loading design plus a bounded 8 GB smoke
plan. It must not displace DINOv2 merely because it is newer.

### SigLIP 2 Base FixRes 224 image encoder — `P2 + A0 + H0`

The official `google/siglip2-base-patch16-224` model card is Apache-2.0 and the
official Transformers path exposes a vision encoder and pooled output suitable
for a locally learned 165-logit head. The current project does not declare or
install `transformers`. The full released checkpoint contains both vision and
text towers and is listed at roughly 0.4 B parameters; the maintained wrapper
would need to load and retain only the declared image-encoder protocol where
possible.

FixRes 224 fits the current square-input contract. NaFlex does not and is not
the same candidate. Text-prototype scoring is `A4 + H2`, not an alternative
implementation of the learned `H0` head.

**Disposition.** **Conditional R1 intake path.** Promote only with a pinned
Transformers/checkpoint revision and checksum, image-only wrapper, exact
processor contract, frozen-memory smoke test, and an explicit interpretation
that web language supervision is present even when downstream label text is
not used. CLIP remains a continuity lead, but carrying both generic P2 families
is not justified without a separate question.

## Paths deferred before R2

The following families remain relevant research evidence but lack a
proportionate, maintained selector path under the current task and budget.
They are deferred for the stated reason, not merely because they are absent
from `src/models`.

| Family | Blocking evidence at R0.2 | Re-entry condition |
| --- | --- | --- |
| MAE and FCMAE | No declared maintained dependency or local wrapper; no evidence that carrying another masked-image P1 model answers a question not already covered by DINO and the supervised controls. | Identify a compact, licensed, pinned checkpoint and a distinct predeclared selector hypothesis. |
| I-JEPA | Official checkpoints are centered on ViT-H/g, disproportionate for the 8 GB selector budget. | A traceable compact checkpoint and credible smoke plan become available. |
| Food2K transfer | No verified public checkpoint, exact usage terms, taxonomy mapping, or source-overlap audit path was established. | All four are resolved before model integration. |
| Recipe1M+/VLPCook | Published code/checkpoints target cross-modal retrieval through a legacy external stack and directly import recipe/ingredient semantics; Yummly overlap remains unaudited. | Pinned modern adapter, terms, corpus/overlap audit, and a protocol-specific scientific justification. |
| FoodSeg103/ReLeM | Segmentation taxonomy/output differs; ReLeM also imports Recipe1M+ semantics; no canonical classifier adapter is maintained here. | Traceable encoder checkpoint, licence, taxonomy/overlap audit, and independent-logit adapter plan. |
| ML-Decoder and Query2Label | Require a common spatial-feature API and make the head part of learnability. No maintained local package or cross-backbone adapter exists. | `H0` remains a control and R1 explicitly decides that spatial sensitivity justifies a separately named `A1/H1` protocol. |
| C-Tran, graph, and set decoders | Label dependency can predict hidden ingredients from co-occurrence and changes the measurement more than a backbone swap. | Supplemental protocol with image-free/shuffled-image and label-prior controls; never the sole selector. |
| Food-R1 and other generative food VLMs | Large generative interface does not natively expose comparable epoch-level 165-label trajectories and is disproportionate on 8 GB. | Use as a separately governed diagnostic/teacher, not primary `M_ref`, unless the output and compute contract materially changes. |

## R0.2 handoff to R1

R1 starts with three evidence tiers rather than a selected model:

1. **Verified intake:** torchvision ResNet-50; frozen DINOv2 B/14-register after
   bounded reproducibility repair; and the maintained-library pool containing
   ConvNeXt Tiny v1, EfficientNetV2-S, and Swin V2 Tiny.
2. **Conditional intake:** compact DINOv3 and SigLIP 2 Base FixRes 224, only if
   their named access, dependency, checkpoint, and 8 GB gates can be satisfied
   without creating a one-off experiment pipeline.
3. **Deferred:** DenseNet in its broken/redundant current state, exact
   ConvNeXt V2/FCMAE, other masked-image objectives, food-domain transfer,
   structured/dependency heads, and generative food VLMs until their explicit
   re-entry conditions are met.

Before inspecting candidate-specific evidence, R1 must:

- freeze hard gates and a qualitative scale/tie rule;
- decide whether the target measurement favors direct visual-prior neutrality
  or maximum sensitivity under transferable priors;
- decide how many supervised architecture representatives are necessary;
- define the maximum acceptable integration effort and the conditional
  candidate deadline; and
- make the common instrumentation and measured-resource smoke report a gate
  rather than a model-specific preference.

This handoff deliberately leaves `M_ref` open. In particular, “already
implemented,” “newest,” and “smallest official parameter count” are not
selection criteria by themselves.

## Primary and repository sources

### Repository evidence

- `settings/config.py` and `src/commons/exp_config.py`: active metadata,
  feature, loss, metric, and trainer defaults.
- `src/data_processing/images_recipes.py` and
  `src/data_processing/labels_encoders.py`: frozen task, label order, loaders,
  and serialized encoder.
- `src/models/commons.py`, `resnet.py`, `densenet.py`, and `dinov2.py`: model,
  transform, checkpoint-loading, and batch-limit behavior.
- `src/lightning/lgn_models.py`, `lgn_trainers.py`, and
  `custom_callbacks.py`: loss, metric logging, checkpoints, and experiment
  loggers.
- `src/training/commons.py` and `one_shot_exp.py`: reconstruction and training
  entry path.
- `requirements.txt`: declared dependency boundary.
- `experiments/dummy/dummy_dino_experiment/trial_2/`: retained DINOv2
  integration evidence on an earlier task.

### External primary or authoritative sources

- [Torchvision 0.23 models and pretrained-weight table](https://docs.pytorch.org/vision/0.23/models.html)
- [Official DINOv2 repository and hub models](https://github.com/facebookresearch/dinov2)
- [Official DINOv2 hub entry points](https://github.com/facebookresearch/dinov2/blob/main/hubconf.py)
- [Official DINOv3 repository, access paths, variants, and licence link](https://github.com/facebookresearch/dinov3)
- [DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [Official Transformers SigLIP 2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [Official SigLIP 2 Base FixRes 224 model card](https://huggingface.co/google/siglip2-base-patch16-224)
- [R0.1 primary-source catalog](source_catalog.md) for all deferred family-level
  evidence and transfer boundaries.

## Limitations

- Compute has not been measured for any `v5` candidate in the current
  environment.
- Historical experiments establish pipeline feasibility only for their saved
  task, dependency, and artifact state; missing hardware/provenance cannot be
  reconstructed as fact.
- Official parameter counts and FLOPs are inference metadata, not training
  memory or elapsed-time measurements.
- Access, model cards, licences, package APIs, and default weight aliases can
  change. R2 must recheck them when a concrete candidate record is created.
- R0.2 classifies integration credibility. It does not assess candidate quality
  against the still-unfrozen R1 rubric.
