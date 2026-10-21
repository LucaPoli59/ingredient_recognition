# Figures, tables and manuscript export

Read this when producing a display or deliverable. Use `scientific-visualization` for plots and figure audits. Use `documents` when creating or changing DOCX, including its render-and-verify workflow; use `pdf` to extract or inspect PDFs. Resolve their current locations from the skill inventory, not a hard-coded plugin version.

## Evidence displays

Generate numerical figures and tables from the same named, versioned source used by the claim ledger. Retain source path/hash, analysis code, filters, aggregation, units, rounding and uncertainty definition. Preserve missing values, exclusions and negative findings. HPO trials do not become independent seed replicates in a plot.

Choose an encoding that answers the research question: comparable axes, visible uncertainty where justified, explicit metric direction and clear labels. Use color with another cue when series must be distinguished. Inspect at the intended printed size; a palette audit alone cannot establish accessibility. Specify provisional dimensions when the thesis template is unknown.

Use code-native vector diagrams for exact architecture and pipeline relationships, checked against implementation and tensor shapes. Generated illustrations are not evidence for quantitative findings or an exact model graph. Do not call an image service simply because an external skill mentions one.

Captions identify the task/cohort, what is plotted, metric/units, uncertainty and the important interpretation boundary. Add explanation that helps interpret the figure without duplicating every sentence of the surrounding text. Tables need informative headings, units and consistent precision. State when a display is illustrative rather than empirical.

## Export only when requested or needed

Markdown remains authoritative. Preserve a single deliberate direction from source to export, and reconcile supervisor edits back into the source. Do not claim that a Pandoc bibliography contains live Zotero fields.

Pandoc, CSL and a reference DOCX are a candidate conversion route, not a prerequisite for writing. Verify tool availability and the current manual before implementing it. A basic example, after choosing the actual files, is:

```text
pandoc chapter.md --bibliography references.bib --citeproc --reference-doc template.docx -o review.docx
```

Do not run this example until its inputs exist. Add the chosen CSL and document metadata when known. Figure/equation numbering and cross-references require an explicit tested solution; do not assume an extension is installed.

Test a representative passage containing a citation, equation, table and figure before building a full manuscript. Render the DOCX using `documents`, inspect page images, and check references, math, captions, tables, spacing and page breaks. Repeat only affected checks after corrections. This skill's setup did not validate Word conversion.

Before a submission release, resolve working placeholders, bibliography and cross-references; reconcile contents with headings; review abstract/conclusion agreement and any required declarations. Archive the exact Markdown, bibliography, figures, build configuration, final deliverable and evidence references. A rendering pass establishes layout quality, not scientific validity or permission to submit.
