# Specialist integration

Read this when selecting a specialist command, diagnosing a limitation or updating the installation. The [source manifest](skill_sources.json) records the reviewed revision and file hashes. Read the actual selected `SKILL.md` before invoking it.

## Available routes

| Skill | Purpose here | Boundary |
| --- | --- | --- |
| `citation-management` | DOI metadata, stable BibTeX, duplicates, missing references and bibliography review | Does not verify a proposition, numerical result or complete Pandoc syntax. |
| `scientific-visualization` | Matplotlib figure design/export, source manifest, metadata and palette checks | Scientific interpretation and visual inspection remain necessary; journal profiles are not thesis requirements. |
| `pdf` | Paper extraction, source-page inspection and final PDF checks | Use the installed plugin's current entry point. |
| `documents` | A requested Word artifact, conversion/layout support and render-and-verify review | The Markdown-to-Word build remains to be implemented and checked. |
| `maintain-project-docs` | Store research, decisions, outcomes and project status with the correct owner | The manuscript is not another project progress tracker. |

The two external specialists are installed under `$CODEX_HOME/skills` (fall back to the user's `.codex/skills`). Their upstream files are unchanged; the repository's MIT license was added to each package because the installer copies only the skill subtree. Resolve bundled plugin skills from the live inventory rather than installing duplicates.

## Execution environment

Setup on 2026-10-21 created an isolated Python 3.12.14 environment at `$CODEX_HOME/tools/thesis/.venv`, using the same `.codex` fallback. On this Windows host its interpreter is `Scripts/python.exe`. It is separate from WSL training environments. Installed versions are recorded in [requirements-tested.txt](requirements-tested.txt); the freeze describes this Windows smoke environment, not a guarantee for other platforms. Resolve the path before running commands, and use an equivalent isolated environment if working elsewhere.

No account or API key was needed for the exercised DOI and local figure paths. OpenAlex search, PubMed, Google Scholar, Seaborn, Plotly/Kaleido and paid image/search services are not needed for the baseline and were not exercised. Recheck API access before using a new service; do not install optional dependencies or create accounts simply because a generic guide lists them.

## Bibliography commands

Resolve the selected skill's `scripts/` and use argument arrays or proper shell quoting. Substitute actual paths; do not interpolate source metadata into shell code.

```text
python <citation-scripts>/doi_to_bibtex.py 10.1109/CVPR.2016.90 --output new_entry.bib
python <citation-scripts>/format_bibtex.py references.bib --output references_review.bib
python <citation-scripts>/validate_citations.py references.bib --manuscript chapter.md --report citation_report.json
```

Before validation, confirm the input files exist and are readable. Before formatting, check for malformed entries and retain the original. Inspect the entry count, fields and keys in the output; do not use `--in-place`, `--rekey` or automatic deduplication on the authoritative bibliography without reviewing the proposed changes. Do not apply a venue preset or minimum reference count without an actual requirement.

Known limitations of this revision:

- The shared BibTeX parser can skip malformed/unterminated input. A successful formatter/validator exit does not prove the input was fully parsed.
- Citation extraction is regex-based. It misses author-suppressed `[-@key]`, truncates keys containing a dot, and can treat code examples as citations. Inspect such constructs directly; final conversion must also resolve references.
- A failed manuscript read can return an empty citation list. A zero exit status is not an adequate missing-file check.
- Bibliographic checks cannot detect a made-up number, wrong split or unsupported inference. Apply [source and evidence review](sources_and_evidence.md) separately.

For ordinary new keys, use stable alphanumeric identifiers such as `He2016Deep`; preserve existing keys rather than changing them to accommodate a checker. Missing metadata should be investigated, not invented.

## Figure commands

Prefer Matplotlib for this project's static scientific plots. Use the skill's `figure_export.export_figure` with explicit formats, size, DPI, provenance and `write_manifest=True`. Run scripts in a separate process or carefully scope imports: both installed packages contain an unrelated module named `_common.py`, so mixing their script directories in one interpreter can resolve the wrong helper.

```text
python <visualization-scripts>/image_metadata.py figure.pdf --format pdf --output figure_metadata.json
python <visualization-scripts>/palette_audit.py --palette okabe_ito_on_white --background FFFFFF --role graphical
```

Use an explicit fresh output path. The exporter refuses overwrite by default and uses same-directory atomic file publication; multi-format export is not an all-or-nothing transaction. Inspect every produced artifact and its manifest. On a filesystem that does not support the required operations, export to a local staging directory, verify, then copy the accepted output deliberately.

## Updates

Do not silently replace the pinned revision. Review relevant upstream changes, preserve licenses, install through `skill-installer`, compare installed bytes with the reviewed source, and rerun the relevant [validation fixture](../../../../src_scratches/thesis_skill_validation/README.md). Update this guide and the manifest when the supported behavior changes. Specialist names in this file are real dependencies, not a request to activate every external cross-reference they contain.
