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

## Subsequent explicit runtime patch

The original `audit.py` and `summary.json` are preserved byte-for-byte. After registration of the reviewed checkpoint-only patch, direct equality between old source bytes and current code is intentionally no longer true. Do not rewrite the original inventory or audit to pretend otherwise.

```bash
python src_scratches/phase6_first_release_audit/verify_runtime_patch.py
```

This separate read-only verifier checks every original reviewed input hash, validates the additive active source snapshot, loads each trusted existing study without creating a trial, and verifies that journal/live-sampler bytes and one-COMPLETE/99-remaining accounting are unchanged. It never calls `ask`, `suggest`, `optimize`, constructs a model or reads test metadata/images. `runtime_patch_verification.json` is the retained transition verification. This is a registration-cutoff check, not a live campaign monitor: it deliberately rejects later trial/journal/sampler changes, even if made by a subsequently authorized release. The maintained source validator separately permits valid journal append/live-sampler advancement while protecting original evidence. `--output NEW_PATH` exclusively creates another report; no output or original artifact is overwritten. Source registration and this verifier do not approve a budget or activate early stopping.
