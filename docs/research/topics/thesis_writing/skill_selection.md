# Thesis skill selection and verification

**Created:** 2026-10-21  
**Last updated:** 2026-10-21

## Question and result

Which previously identified skills add useful capabilities to this English Markdown thesis without duplicating its editorial guidance or imposing irrelevant procedures?

Two specialist skills are installed: **`citation-management`** and **`scientific-visualization`**. The repository's new [**`write-thesis` skill**](../../../../.agents/skills/write-thesis/SKILL.md) coordinates their use with the existing `documents`, `pdf` and `maintain-project-docs` skills. It covers argument planning, research, citations, evidence, drafting, scientific/editorial review, figures and later export. Installation was authorized by the user; the [setup plan](../../../plans/thesis_writing_skills.md) owns completion status. The thesis outline and manuscript have not been written by this task.

## Method and provenance

The 2026-10-20 investigation inspected five external manifests. This follow-up fetched their complete skill subtrees into an inactive review directory and rechecked the repository's main revision on 2026-10-21: it still resolved to `593377bb048752abb0152deab9fbb1083a6087df`. All five entry points and selected relevant references were reviewed. For the two installed packages, the inspection inventoried Python imports and network/file operations and examined the specific parser, DOI, citation-check, formatting, export and local asset-loading paths used or bounded below. This was a targeted review, not an exhaustive security audit of every optional function.

Installation used the official local `skill-installer` helper with that explicit revision. All 22 citation-management files and 18 scientific-visualization files matched the reviewed copies byte-for-byte. The repository-root MIT license was added to each installation; upstream instructions and scripts were otherwise unchanged. The [source manifest](../../../../.agents/skills/write-thesis/references/skill_sources.json) records hashes, and [third-party notices](../../../../.agents/skills/write-thesis/references/third_party_notices.md) preserve the provenance of incorporated guidance.

## Selection

The source for each assessment is the corresponding pinned upstream entry point and its bundled files.

| Candidate | Decision | Reason and scope |
| --- | --- | --- |
| [citation-management](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/citation-management/SKILL.md) | **Installed** | Useful DOI/BibTeX mechanics and Markdown citation checks. Use within documented parser limits; metadata never substitutes for support-of-claim review. |
| [scientific-visualization](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-visualization/SKILL.md) | **Installed** | Useful figure export, dimensions, provenance, palette and metadata checks. Apply thesis dimensions and actual statistical meaning rather than journal defaults. |
| [scientific-writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-writing/SKILL.md) | **Guidance incorporated; not installed** | Reuse evidence/claim separation and consistency reasoning. Its five registries, human-verification declarations and clinical reporting routes would add a parallel workflow; the project uses its existing authorities and a compact claim ledger. |
| [literature-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/literature-review/SKILL.md) | **Guidance incorporated; not installed** | Reuse recorded searches, thematic synthesis and coverage limits. Systematic-review machinery, optional Parallel/OpenRouter integration and XeLaTeX export are unnecessary for targeted computer-science related work. |
| [peer-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/peer-review/SKILL.md) | **Guidance incorporated; not installed** | Reuse location-specific claim/method critique and the distinction between missing reporting and demonstrated inconsistency. The author's self-review does not need a journal-review intake or editorial decision workflow. |

These are task-specific judgments, not a ranking of the libraries' general quality. Installation of the whole collection would not provide evidence that the manuscript is better.

## Environment and observed checks

The specialists are discoverable in the user's Codex skill directory; `write-thesis` resides under `.agents/skills/` in this repository. A separate thesis-tools virtual environment was created outside the WSL training environment. The exercised versions are Python 3.12.14, requests 2.34.2, Matplotlib 3.11.2, Pillow 13.0.0 and pypdf 6.21.1; PyYAML 6.0.3 supports skill validation. The [dependency freeze](../../../../.agents/skills/write-thesis/references/requirements-tested.txt) and [integration guide](../../../../.agents/skills/write-thesis/references/specialists.md) describe reproduction and command selection.

The [rerunnable fixture](../../../../src_scratches/thesis_skill_validation/README.md) uses public bibliographic metadata and synthetic values, with these observed results:

- The DOI for *Deep Residual Learning for Image Recognition* produced a BibTeX record. Formatting preserved the input bytes and existing citation key.
- A Markdown citation absent from the bibliography produced a high-severity error and exit 1. The corrected passage passed that check.
- Assistant review found an unsupported numerical value (`0.470` instead of `0.410`), test/validation confusion and unsupported cross-seed stability. These were manual evidence checks, not capabilities of the citation script.
- A synthetic Matplotlib figure exported as PNG and PDF with a provenance manifest. Both metadata checks and the palette command completed successfully. The PNG was visually inspected; the PDF received metadata inspection only.
- The project skill passes the bundled metadata/scaffold validator. Its local references and documentation links are checked separately; syntactic validity is not proof of writing quality.

The [review record](../../../../src_scratches/thesis_skill_validation/review.md) states exact findings and inspection limits. This was a constructed self-check, not blinded forward testing or a full chapter evaluation.

## Limits that affect use

The bibliography tools need extra care: the parser skips malformed/unterminated entries; citation extraction misses author-suppressed Pandoc citations, truncates dotted keys and can read code examples as citations. The manuscript-read error path can return an empty list. Therefore check input accessibility, preserve originals, inspect parser coverage and output diffs, and verify final rendering. The project guide records these boundaries rather than claiming complete automated citation verification.

The selected DOI path used no account or key. The other search integrations were not exercised. [OpenAlex's current authentication guidance](https://help.openalex.org/api/authentication/) distinguishes basic keyless queries from account-dependent scale and cursor paging; access must be checked for the intended search. No optional search account, paid service, third-party manuscript upload, image-generation service or training dependency change was introduced.

The final template, citation style, author-edited calibration sample, and Markdown-to-Word conversion remain open. Existing `documents` and `pdf` capabilities are routed by the skill, but this setup did not validate a thesis DOCX build. Final experimental writing still depends on the project's Phase 6/7 evidence gates.

## Attribution

The adopted tooling and incorporated guidance come from K-Dense Inc.'s MIT-licensed repository. The library reference is Kassis, Agarwal, He, Patel and Brueckner (2026), [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://doi.org/10.48550/arXiv.2609.00065), whose arXiv record was checked during this review. Attribution is recorded as tooling provenance here and in the skill; no thesis bibliography or scientific claim is inferred from it.
