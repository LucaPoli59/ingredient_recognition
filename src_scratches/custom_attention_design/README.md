# Custom attention-design arithmetic

This folder contains the scalar calculator supporting the reviewed
[4A.4.3 compatibility synthesis](../../docs/research/topics/custom_attention_model_design/architecture_compatibility_synthesis.md).
It is not a model implementation, benchmark or resource smoke test.

Run from the repository root with Python 3.10 or later:

```sh
python src_scratches/custom_attention_design/estimate_design_envelope.py
```

The script uses only the standard library and writes its results to stdout.
It imports no ML library, reads no files, constructs no model/tensors, loads no
weights and accesses no dataset or GPU. Its assumptions are the explicit
TorchVision v0.23.0 EfficientNetV2-S configuration and the prospective component
equations documented in the synthesis. Assertions reconcile the scalar trunk
count with the upstream parameter metadata and verify feature resolutions.

MACs count convolution, SE linear operations, linear projections and attention
matrix products only; activation figures are individual/intermediate storage
examples, not peak or upper-bound training memory. The durable document owns
the interpretation, limitations and source links. Update both if formulas or
the scale envelope change. Raw stdout is not retained as another source of truth.
