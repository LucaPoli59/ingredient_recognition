# Thesis writing skill setup

**Created:** 2026-10-21  
**Last updated:** 2026-10-21

## Scope

Prepare the agent workflow supporting [Macro-section 8 and work package 8.1](../general_plan.md#8-thesis-writing): review the five previously identified external skills, install useful specialists, and create a repository-scoped writing skill. The user's authorization covers selection, installation and creation. Manuscript drafting, final formatting, scientific-policy changes, experiments and Git publication are outside this setup.

## Progress tracker

**Overall status:** Done  
**Current task:** Setup complete  
**Next action:** Use `write-thesis` for the separately pending 8.1 outline and evidence map; implement Word conversion only when needed.

| Task | Status | Evidence or result |
| --- | --- | --- |
| Inspect candidate instructions, dependencies and relevant executable paths | **Done** | Five K-Dense candidates reviewed at `593377bb048752abb0152deab9fbb1083a6087df`; selected citation/visualization paths receive execution checks below. |
| Install selected specialists and create project guidance | **Done** | `citation-management` and `scientific-visualization` installed at the pinned revision; repository [write-thesis](../../.agents/skills/write-thesis/SKILL.md) and isolated tool dependencies created. |
| Validate scripts, references and a representative section review | **Done** | Skill validator, DOI/Markdown citation checks, figure export/metadata and manual number/scope review; [fixture and exact boundaries](../../src_scratches/thesis_skill_validation/review.md). |
| Publish the reviewed selection, usage and verification record | **Done** | [Selection record](../research/topics/thesis_writing/skill_selection.md), indexes and general-plan handoff synchronized. 8.1 and the thesis remain Pending. |

## Implementation and acceptance

Use the existing [research](../research/topics/thesis_writing/tools_and_workflow.md) and [editorial requirements](../research/topics/thesis_writing/author_voice.md). Install only the selected pinned packages through `skill-installer`, keep executable dependencies outside the training environment, and preserve upstream notices. Keep external scripts unchanged unless an observed defect warrants a separately recorded adaptation.

The project skill must route literature/citation work, argument planning, bounded drafting, methodological and numerical review, scientific figures, terminology and eventual Word export. Link existing authorities instead of duplicating them. References are loaded by task, not all at once. Reusable reasoning from the three uninstalled guidance skills may inform the local instructions with provenance; their journal-specific gates are not adopted.

Completion requires valid skill metadata, resolving local references, reproducible installation provenance, exercised specialist paths, and a section-review fixture with a real bibliographic source plus deliberately unsupported material. A passing metadata or rendering check must not be described as verification of scientific claims. Document any remaining export, source-access or calibration limitations.

## Decisions and dependencies

- 2026-10-21: retain the English/Markdown workflow and the mandatory natural, concise, evidence-backed editorial standard. Final template and citation style do not block skill setup.
- 2026-10-21: prefer two specialist installations and one project guide over activating all five overlapping generic workflows. No additional accounts or paid services are necessary for the selected checks.
- 2026-10-21 completion: the [selection record](../research/topics/thesis_writing/skill_selection.md) and skill references now own those findings. Citation-parser limitations are explicit; no source scripts were patched. The exercise is a constructed self-check, with manual scientific review distinguished from automated validation. Final template, author calibration and Word conversion remain later manuscript work.

Final setup verification on 2026-10-21: all 103 scoped local links/anchors resolve, the installed upstream file hashes match the recorded manifest, the project skill passes `quick_validate.py`, and `git diff --check` passes for the affected tracked documentation. These checks do not evaluate the eventual manuscript's quality.
