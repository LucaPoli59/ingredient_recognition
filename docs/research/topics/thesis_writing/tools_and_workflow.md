# Tools and workflow for evidence-based thesis writing

**Created:** 2026-10-20  
**Last updated:** 2026-10-21

## Research question and status

Which tools, agent skills, and writing practices would help turn this project's evidence into a rigorous thesis that retains its author's voice?

This is preparatory research for [Macro-section 8](../../../general_plan.md#8-thesis-writing). The recommended baseline is **English Markdown chapters, a central bibliography, a claim-to-evidence ledger, reproducible figures, and separate scientific and editorial reviews**, with Pandoc as the candidate for later Word export and Zotero as an optional reference-library manager.

The user's [confirmed requirements](../../../general_plan.md#confirmed-writing-requirements) establish the programme, language, Markdown source, permitted AI assistance and a mandatory [editorial standard](author_voice.md#required-editorial-standard). The initial LaTeX-first recommendation is superseded by that clarification. The initial investigation created no manuscript or installation. The [2026-10-21 follow-up](skill_selection.md) subsequently installed two specialists and implemented the [project writing skill](../../../../.agents/skills/write-thesis/SKILL.md). Chapter structure and conversion tooling remain proposals; final submission details remain open.

## Method and source cutoff

Research cutoff: **2026-10-20**. The investigation combined:

1. The repository knowledge map, general plan, documentation governance, research indexes, problem definition, and Q1-Q4 methodology.
2. Text extraction from all 74 pages of the supplied bachelor thesis; close reading of the contents, introduction, representative theory and methods, results, conclusion, and bibliography; visual inspection of selected rendered pages. This is a style/production review, not a complete technical audit of that thesis.
3. Official institutional and product documentation, original tool repositories, and the NeurIPS reporting checklist. Search snippets were used for discovery, not as proof of a manuscript claim.
4. The current session's tool and skill inventory, the OpenAI curated skill listing, plugin searches, and bounded local dependency checks.
5. Inspection of five external K-Dense skill manifests and their file inventory at repository commit `593377bb048752abb0152deab9fbb1083a6087df`. External skill instructions were research material, not activated instructions; their scripts were not executed or comprehensively audited.

The local PDF extraction/rendering path was exercised successfully. LaTeX compilation, Zotero automation, third-party research services, and external skill execution were not tested. No thesis content was uploaded to those services.

## Institutional guidance and its authority

The bachelor thesis identifies Milano-Bicocca, Informatica, academic year 2021-2022. The user subsequently confirmed that the new thesis is for the Informatica master's programme at the same university and will be written in English. The older PDF remains a style sample rather than the current formatting specification.

The official [Informatica master's final-exam page](https://elearning.unimib.it/course/section.php?id=110520&lang=en) links a regulation, title-page material, writing guidance, and an open thesis-writing course. Its [writing guidance](https://elearning.unimib.it/pluginfile.php/564058/course/section/110520/LM-Inf_Indicazioni-Tesi.pdf) recommends a technical, objective account, explicit identification of the author's contribution, numbered/captioned figures, and correspondence between in-text citations and bibliography. It allows Italian or English and recommends impersonal present-tense prose. Check applicability and current submission requirements before using it as the manuscript specification.

The [2024 recommendations on generative AI](https://elearning.unimib.it/pluginfile.php/376465/mod_forum/attachment/405052/Raccomandazioni%20uso%20IA%20nella%20produzione%20scritta.pdf?forcedownload=1), produced by named groups in collaboration with DISCo, promote critical verification, personal reworking, and explaining the purposes of AI use. They are explicitly proposals/recommendations. For this project, proceed on the user's confirmation that AI assistance is permitted; any specific acknowledgment wording remains an open submission detail. The user's confirmation is not presented as evidence of a university-wide policy.

For technical reporting, use the [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) as a review aid: align claims with evidence; state limitations, experimental settings, reproducibility, compute, and the source of uncertainty. It is not an institutional thesis template or a requirement to invent experiments the project cannot afford.

## Recommended tool stack

| Need | Candidate and practical use | Availability and qualification |
| --- | --- | --- |
| Authoritative manuscript | **English Markdown**, one file per chapter, with a central bibliography and separate evidence notes | Source format selected by the user. Specific folder layout and conversion setup remain to be implemented. |
| Reference library | **Zotero** for bibliographic records, source PDFs, notes, and citations | Recommended external application; no connected library was verified. Its [Word/LibreOffice/Google Docs plugins](https://www.zotero.org/support/word_processor_integration) support dynamic citations and bibliography. |
| Bibliography export | **Better BibTeX**, an optional third-party Zotero add-on, for a dedicated collection exported to a tracked `.bib` file usable by Pandoc | Its [automatic-export documentation](https://retorque.re/zotero-better-bibtex/exporting/auto/) describes keeping exports updated. Check citekey stability and use one bibliography authority; do not independently edit both generated exports and library records. |
| Retained alternative | **LaTeX**, BibLaTeX/Biber where supported, and **latexmk** | Superseded as the recommended authoring route by the user's Markdown choice. [latexmk](https://ctan.org/pkg/latexmk) automates compilation; it and the checked TeX tools were not found on local PATHs. No TeX setup is required for the proposed Markdown-to-DOCX path. |
| Shared editing | **Overleaf**, when the supervisor wants browser-based LaTeX collaboration | Its [official starter project](https://docs.overleaf.com/getting-started/your-first-project) supports `.tex`, figures, bibliography, and recompilation. No matching plugin was returned by the catalog search; no account/project access was verified. Keep one authoritative source and an explicit synchronization direction. |
| Word deliverable | **Word** for a possible exported review/submission copy | Markdown stays authoritative. Local `documents` and `pdf` skills can support artifact and rendering checks. Do not assume an exported bibliography has live Zotero fields. |
| Conversion | **Pandoc**, a bibliography/CSL style, and a reference DOCX when formatting is needed | Its [manual](https://pandoc.org/MANUAL.html) documents Markdown-to-DOCX output, citation processing and reference-document styling. Pandoc was not found on the checked PATHs; setup and a representative export test remain pending. |
| Bibliographic verification | **Crossref**, publisher/proceedings pages, and official paper repositories | The [Crossref API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) exposes deposited bibliographic metadata. A valid DOI verifies identity, not that a paper supports a proposition; inspect the actual passage/table. Preserve genuinely absent DOI/page fields. |
| Literature discovery | Existing project source catalogs, web search, **OpenAlex**, and discipline-specific publication repositories | [OpenAlex](https://help.openalex.org/) is a research index. Recheck access and API requirements before automation. Discovery results need deduplication and full-text verification; no dedicated relevant Semantic Scholar plugin was identified in the catalog results. |
| Optional literature assistance | **Elicit**, for a concrete gap in searching or structured extraction | The [vendor documentation](https://elicit.com/solutions/literature-review) describes source-linked screening and extraction. This was not an independent accuracy evaluation or a logged-in test. A subscription is not part of the recommended baseline; account, access, and cost would need checking. |
| Scientific figures | **Matplotlib** and code-generated diagrams, with source tables and a figure manifest | [Matplotlib export](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) supports saved figure artifacts. Prefer vector exports for plots where suitable, inspect at final page size, and retain the code/data that produced them. Generated illustrations are not evidence for measured results or exact architecture graphs. |
| Revision and output QA | Git; optional **latexdiff** for LaTeX; PDF text checks and page rendering | [latexdiff](https://ctan.org/pkg/latexdiff) marks textual changes between LaTeX sources. PDF extraction/rendering was tested locally. Rendering must complement citation, formula, and numerical checks. |

**Selected source format:** Markdown, by user instruction. Keep Word exports as derived artifacts and bring reviewed changes back into Markdown deliberately. The earlier LaTeX/Word comparison is retained as research history rather than an unresolved authoring choice.

### Markdown conventions for the proposed workflow

Use semantic headings, a file per chapter, relative figure paths, and stable bibliographic identifiers. Pandoc supports citations such as `[@citation_key]`, rendered with `--citeproc`, and a DOCX style reference through `--reference-doc`; see the [citation](https://pandoc.org/MANUAL.html#citations) and [reference-document](https://pandoc.org/MANUAL.html#option--reference-doc) documentation. Keep source verification notes separate from manuscript prose. Establish and test figure/equation cross-reference handling before relying on automatic numbering. A representative export should include a citation, formula, table and figure; inspect its rendering before adopting the build recipe. Conversion is a planned capability, not a completed validation.

No suitable Zotero or Overleaf plugin was found in the searched catalog. This is a dated discovery result, not a claim that integrations cannot exist elsewhere. Available built-in file and web tools are sufficient for the preparatory workflow.

## Skills already available

| Skill | Role in this thesis workflow | Boundary |
| --- | --- | --- |
| `maintain-project-docs` | Keep research, decisions, empirical results, and status in their existing authoritative locations | Already used for this research record; do not turn the manuscript into another project tracker. |
| `pdf` | Read papers, extract text, render and inspect thesis pages | Used for the supplied sample. Text extraction alone cannot validate equations or page layout. |
| `documents` | Create/revise DOCX artifacts and check their rendering | Relevant to a later Word export; available capability, not a verified live Word session. |
| `skill-creator` | Build and maintain the project-specific thesis workflow | Used in the [2026-10-21 setup](skill_selection.md); a calibrated author sample can refine the resulting skill. |
| `skill-installer` | Discover and selectively install reviewed third-party skills | Curated listing checked. It contained no dedicated thesis/scientific-writing skill under those names at this cutoff. Plugin-provided skills remain a separate inventory. |
| `openai-docs` / `plugin-management` | Verify current agent capabilities and discover integrations | Used for capability research, not sources for scientific thesis claims. |

`presentations` can support the eventual defence; `visualize` can support discussion of structures and results. Neither is essential for the manuscript pipeline. There is no need to introduce image generation for quantitative plots.

## External skill shortlist

[OpenAI's skill documentation](https://developers.openai.com/plugins/concepts/skills) describes skills as reusable instructions plus optional resources and scripts. A skill is not a new scientific authority or a guarantee of correct prose.

The following were **external candidates, not installed at the initial 2026-10-20 cutoff**. Links identify the inspected revision of [K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/593377bb048752abb0152deab9fbb1083a6087df). The [selection and verification record](skill_selection.md) now owns the follow-up findings and dispositions: citation-management and scientific-visualization are installed; useful guidance from the other three is incorporated locally.

| Candidate | Useful contribution | Adaptation needed |
| --- | --- | --- |
| [scientific-writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-writing/SKILL.md) | Evidence-linked outlines, claim/number consistency, reference and manuscript checks | Its manifest includes human source-verification and journal/reporting workflows. Adapt to an Informatica thesis; clinical checklists and article structure are not automatic defaults. |
| [citation-management](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/citation-management/SKILL.md) | DOI metadata, BibTeX cleanup, duplicate and reference validation | Best first candidate for a bounded trial. Verify current API contracts, preserve established citekeys, and keep support-of-claim review separate from metadata validation. |
| [scientific-visualization](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-visualization/SKILL.md) | Figure sizing, readable palettes, uncertainty labels, export checks | Useful alongside existing plotting scripts. Use actual thesis dimensions and data semantics; a journal preset is not automatically appropriate. |
| [literature-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/literature-review/SKILL.md) | Recorded search strategies, screening, thematic synthesis and citation checking | Use a narrative/targeted related-work workflow unless a systematic review is truly intended. Optional services and AI schematics add dependencies and are unnecessary here. |
| [peer-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/peer-review/SKILL.md) | Structured methods, claim, statistics and reproducibility critique | Adapt to author-requested self-review. It is not an actual external peer review, a second independent experiment, or an editorial decision. |

The initial adoption recommendation was to evaluate `citation-management` and `scientific-visualization` first, and use `scientific-writing` as a design reference for a smaller local workflow. The follow-up implemented that direction with pinned sources and bounded checks. Its recorded limitations govern use; the initial manifest review alone did not certify script quality or compatibility.

## Proposed local skill

This proposal is retained as design history. Its implemented successor is [**`write-thesis`**](../../../../.agents/skills/write-thesis/SKILL.md), which links the existing repository authorities and routes the selected specialists. Actual behavior and validation boundaries are described in the [selection record](skill_selection.md).

Proposed resources:

- a requirements sheet for the confirmed programme, language, template and submission rules;
- the [author voice profile](author_voice.md), calibrated against a new author-edited sample;
- a chapter-to-question map and claim/source ledger;
- a terminology and notation sheet;
- a short scientific/editorial/output review checklist;
- optional deterministic checks for unresolved references, placeholders, bibliography identity, and numbers against an explicit evidence table.

Core behavior: inspect the current evidence; identify unresolved facts; draft or revise one bounded section; preserve claim provenance; keep results and interpretation distinct; apply the [required editorial standard and acceptance checks](author_voice.md#required-editorial-standard); render the selected output; report unresolved issues. Match the degree's actual AI policy and record assistance accurately. A future skill must preserve these requirements without imposing a reusable paragraph template or expanding prose unnecessarily.

The proposed acceptance exercise was a small section containing a numerical claim, a citation and a figure. The [completed bounded exercise](../../../../src_scratches/thesis_skill_validation/review.md) now records missing-source detection, numerical/scope review and figure checks, distinguishing automated findings from assistant review. A full manuscript or Word rendering was not tested.

## Writing workflow and evidence control

1. **Apply confirmed requirements.** Use the English/Markdown direction and permitted AI assistance recorded in the general plan. Track the remaining template, length, deadline, citation-style and acknowledgment details separately; they do not block a provisional outline or stable chapter drafting.
2. **Build the argument.** Write a short statement of the problem, research questions, actual contributions, main limitations, and what remains unmeasured. Organize by scientific questions rather than the chronology of debugging.
3. **Create a source and claim ledger.** A useful row contains claim ID, proposed wording, chapter, source/experiment ID, exact page/section/table or artifact path, version/commit, split/cohort, value/units when relevant, verification status, and limits. Distinguish a real bibliographic record from verified evidential support.
4. **Collect the author's reasoning.** Before each chapter, use brief notes on the choice made, alternatives, evidence, observed difficulties and interpretation. Only record experiences and rationales the author confirms. These notes supply voice and judgment rather than generic filler.
5. **Draft one section at a time.** Start with a small evidence-backed outline, then connected prose following the [required editorial standard](author_voice.md#required-editorial-standard). Include citations and decision rationales while drafting; mark missing material in working notes. Write background, data and stable methodology before final results; refine the introduction and abstract after the argument is complete.
6. **Generate displays from evidence.** Produce tables and figures from named outputs, with fixed units, metric definitions, denominators and rounding. Keep a manifest tying each display to data, analysis code, parameters and the claim it supports.
7. **Review in distinct passes.** Scientific correctness and claim scope first; citations and numeric agreement next; author voice, justified choices and concision next; final layout and submission checks last. Apply the [section acceptance checks](author_voice.md#a-practical-author-in-the-loop-process), including a read across adjacent sections for repetitive structures. A fluent paragraph can still contain an unsupported conclusion.
8. **Freeze a reviewable release.** Archive manuscript sources, bibliography, figures, exact PDF, build environment and evidence references. Remove working placeholders and ensure all references resolve. Preserve revisions and any required record of AI assistance.

This method supports authentic authorship through specific reasoning, accurate evidence, economical explanation and author revision. Editorial quality is a requirement for accepting a section, not a cosmetic pass postponed until final export.

## Proposed structure for this project

This is a working hypothesis for 8.1, not a final outline or a page-count requirement.

| Chapter | Purpose and evidence owner | Readiness |
| --- | --- | --- |
| 1. Introduction | Problem, contributions, scope and Q1-Q4; [problem definition](../../../project_objective/problem_definition.md) and [methodology](../../../project_objective/model_comparison_methodology.md) | Structure can be prepared; final claims await results. |
| 2. Background and related work | Only concepts needed for the problem, model choices, multi-label evaluation and visual ambiguity; [research](../../README.md) and [model deep dives](../../../models_deepdive/README.md) | Substantial source material exists; verify original papers before citing. |
| 3. Dataset and benchmark construction | Yummly supervision, FoodOn-first normalization, support, duplicate handling and split; [benchmark decisions](../../../project_objective/benchmark_decisions.md) | Stable material can be drafted. |
| 4. Ingredient selection and comparison design | Learnability, selector role, D4/D6 history, shared vocabulary, Q1-Q4 controls; [methodology](../../../project_objective/model_comparison_methodology.md) | Distinguish frozen rules from outcome-informed amendments. |
| 5. Models and experimental setup | Baselines, established families, DICA-Net, actual configuration and resource constraints; [portfolio](../../../project_objective/experimental_model_portfolio.md) and [runtime contract](../../../implementation_details/experimental_model_contract.md) | Architecture is available; campaign-dependent details remain provisional. |
| 6. Results | Separate full-task ranking, matched selected-task ranking, reference random controls and local adaptation; [reviewed results](../../../experiment_results/README.md) | Final benchmark claims remain deferred. |
| 7. Discussion and limitations | Answer questions, analyze errors and ambiguity, explain compute/data/single-run boundaries | Prepare questions now; complete with final evidence. |
| 8. Conclusions | Supported contributions and bounded future work | Complete last. |

Appendices can hold detailed configurations, additional plots, reproducibility material and optional observability examples. Adjust chapter granularity after the supervisor's requirements are known.

The evidence ledger must enforce these project-specific boundaries:

- Ingredient labels describe recipes; predicted plausibility is not proof that an ingredient is visible in the image.
- Vocabulary learnability does not certify human observability or food safety.
- Q1/Q2 compare model categories; Q3 tests vocabulary reduction with the reference model; Q4 is separately bounded local adaptation.
- The 384-pixel selector and 224-pixel experiment model are distinct protocols even when they share a backbone family.
- D6 is explicitly outcome-informed; do not rewrite it as a rule fixed before seeing all evidence.
- Validation/HPO evidence is not final test evidence; engineering acceptance is not predictive superiority.
- Bootstrap uncertainty for the evaluated sample is not seed-level training stability. Near-duplicate leakage is not ruled out merely by byte-identical grouping.

Illustrative ledger entries would map the full/selected vocabulary definitions to the versioned methodology and projection artifact, while leaving any claim that DICA-Net outperforms alternatives unresolved until comparable final evidence exists. Avoid constructing a compelling narrative around a desired winner.

## Limitations and next handoff

The recommendation is grounded in one historical writing sample, the user's confirmed requirements and current documentation, not an exhaustive tool benchmark. Prices, account permissions, full-text access, final submission specifications and local conversion integration were not established. User edits to training-related files were present during the investigation; this research neither changes nor certifies that work.

The next bounded deliverable for 8.1 should be a provisional English Markdown outline with an initial evidence map and a short author-revised sample, using the confirmed requirements and implemented writing skill. Bibliography/figure tools are available within their tested limits; conversion setup remains a later task. Final experimental chapters remain governed by the existing Phase 6/7 gates.
