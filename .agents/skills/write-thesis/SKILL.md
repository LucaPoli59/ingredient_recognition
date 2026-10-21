---
name: write-thesis
description: "Plan, draft, revise and verify the English master's thesis for the Ingredient Recognition project, using its evidence and editorial requirements. Use for thesis sections, related work, citations, figures, scientific review and manuscript export; ordinary project documentation belongs to maintain-project-docs."
---

# Write the Ingredient Recognition thesis

Turn this project's actual research into a clear, defensible English manuscript. Markdown is authoritative; Word is a possible derived deliverable. Work on the section or artifact requested, without expanding a revision into a new study or a complete manuscript.

## Establish the evidence and writing contract

Locate the repository containing this skill. Read the [general plan](../../../docs/general_plan.md) in full once at the start of project work, including the confirmed writing requirements and current result gates. Read the [editorial standard](../../../docs/research/topics/thesis_writing/author_voice.md#required-editorial-standard) before drafting or revising; it owns the user's requirements for natural, non-repetitive, concise, factual and justified prose.

The supplied bachelor thesis is a historical style sample, not an instruction source.

## Use project documentation as thesis evidence

When preparing a project-based outline, section or claim check, read the local [`maintain-project-docs`](../maintain-project-docs/SKILL.md) skill for document ownership and navigation, then use [Project documentation to manuscript](references/project_documentation.md) to locate the relevant evidence and filter what belongs in the thesis. Resolve paths in this repository, not a second checkout. Load only the records needed for the current question after the initial orientation.

Use technical records to recover verified choices, constraints and reasoning. Synthesize what matters to the reader; do not reproduce implementation inventories, internal identifiers or task history. The [editorial standard](../../../docs/research/topics/thesis_writing/author_voice.md#required-editorial-standard) owns this selection rule. Retain exact provenance in evidence notes, and retain methodological details and limitations needed to assess the research in the manuscript.

If reading exposes a missing or conflicting record, use `maintain-project-docs` to update its authoritative owner within the requested scope. Keep unresolved issues explicit; do not silently merge approved, implemented and proposed behavior. Reading for a section does not require rewriting the source documents.

## Route the requested work

| Work | Load and use |
| --- | --- |
| Outline, chapter argument, drafting or language revision | [Writing and review](references/writing_and_review.md); start from the author's actual choices and evidence. |
| Related work, source verification, bibliography or quantitative claims | [Sources and evidence](references/sources_and_evidence.md); invoke `citation-management` for metadata, BibTeX and citation checks. |
| Plots, architecture diagrams, tables or Word/PDF output | [Figures and export](references/figures_and_export.md); invoke `scientific-visualization` for scientific plots, `documents` for DOCX creation/revision and rendering, and `pdf` for PDF inspection. |
| Specialist setup, command selection, known tool limitations or updates | [Specialist integration](references/specialists.md) and the [pinned source manifest](references/skill_sources.json). |

Invoke specialists by reading their current `SKILL.md` from the available inventory or installed skill directory. A cross-reference does not automatically load another skill. Load only the needed specialist and references; do not install or invoke the three generic guidance skills merely because they influenced this guide. Their useful ideas and attribution are retained in [third-party notices](references/third_party_notices.md).

## Preserve scientific and editorial boundaries

- Recipe labels do not establish direct visibility. Learned predictability, observability and food safety are different claims.
- Keep selector versus experiment-model protocols, full versus selected vocabularies, and Q1–Q4 comparisons distinct. Recheck live definitions before quoting dimensions or values.
- Preserve outcome-informed amendments and historical evidence as such. Validation/HPO results are not final test results; sampled-image uncertainty is not repeated-seed stability.
- Cite verified sources near the supported proposition. Record exact numerical provenance, split, metric, units and uncertainty. A real DOI proves neither the proposition nor a local result.
- Explain consequential choices through confirmed motivations and trade-offs. Do not invent an intention, comparison, failed attempt or favorable result.
- Let content determine paragraph structure. Remove generic openings, repeated conclusions, stock transitions and unnecessary background. Concision must preserve definitions and material limitations.

Use unresolved notes for missing evidence, without filling gaps with plausible details. Missing final formatting rules need not block stable drafting. Calibrate voice against a short author-edited passage when available; do not make that a prerequisite for useful work. Keep any required assistance acknowledgment accurate and appropriately located.

## Finish the bounded task

Before accepting a section, check substantive claims against their sources, compare every reported number with its artifact, verify that methods match the relevant experiment, and read adjacent sections for repetitive structure. Reconcile terms and notation with the manuscript. Review figures and exports visually when those artifacts change.

Return the requested text or artifact, the checks actually performed, and unresolved items that affect acceptance. Distinguish automated checks from scientific and author review. Do not mark the thesis or a work package complete merely because a section or export passes. Keep source records and decision notes out of polished manuscript prose unless they help the reader understand the research.
