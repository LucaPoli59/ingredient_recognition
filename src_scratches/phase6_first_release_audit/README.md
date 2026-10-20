# Phase 6 first-release audit

**Created:** 2026-10-20
**Last updated:** 2026-10-20

Reproduce the [reviewed first-release result](../../docs/experiment_results/phase6_first_release.md) from the main repository using its ML Python:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src_scratches/phase6_first_release_audit/audit.py
```

`--output NEW_PATH` saves a new JSON report exclusively; its parent must exist. `summary.json` is the retained 2026-10-20 review, not a live campaign state or training authorization.

The helper reads the five new journals as JSON without replaying a study, checks complete epoch/update accounting, exact source ZIP/current bytes, approved cached weights, frozen train/validation metadata and ordered saved validation arrays. It deserializes only trusted locally generated full checkpoints on CPU to inspect identity, tensors and saved epoch/update/optimizer state. It does not construct a model, read images/test metadata, evaluate test predictions, mutate a study, repair a source or start training. Scalar validation metrics are recomputed from saved arrays only.

A false `all_final_checkpoints_valid` is a retained finding, not an audit crash: in this release all five `last.ckpt` files are identical to the earlier best checkpoint. Do not rerun training or overwrite those files to hide the exception. The result owner and operational plan describe the required disposition and future retention repair.
