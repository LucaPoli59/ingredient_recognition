# Thesis skill validation fixture

**Created:** 2026-10-21  
**Last updated:** 2026-10-21

This bounded fixture exercises the selected bibliography and figure tools using public DOI metadata and explicitly synthetic numerical values. It is not a project experiment, thesis result or independent review of scientific validity.

Run `validate_specialists.py` using the isolated thesis Python environment described in the [specialist integration guide](../../.agents/skills/write-thesis/references/specialists.md):

```text
python validate_specialists.py --skills-root <installed-skills-directory> --output <fresh-local-output-directory>
```

The script requests only the public DOI `10.1109/CVPR.2016.90`, checks formatting without modifying the input, exercises a missing Markdown citation and a corrected passage, records parser edge cases, and renders a synthetic figure with metadata and palette checks. Use a fresh output directory because figure export intentionally refuses implicit overwrite. Each specialist runs in its own process where needed to avoid the unrelated `_common.py` module names colliding.

Accepted output is retained in `results/`. The [manual review](review.md) separately checks the numerical mismatch, split and unsupported seed-stability claim against the raw fixture. The automated citation checker does not perform those scientific checks. The setup report in [thesis-writing research](../../docs/research/topics/thesis_writing/skill_selection.md) records observed outcomes and remaining limits.
