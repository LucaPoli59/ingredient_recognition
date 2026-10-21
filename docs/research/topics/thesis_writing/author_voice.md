# Author voice and editorial guidance

**Created:** 2026-10-20  
**Last updated:** 2026-10-21

## Scope and provenance

This profile supports future thesis writing; it does not assess authorship or guarantee how the author writes today. Its source is the user-supplied bachelor thesis, *Previsione della volatilità con Reti Neurali Ricorrenti per il trading di opzioni*, Luca Poli, Milano-Bicocca, academic year 2021-2022.

The supplied PDF has 74 physical pages. Its SHA-256 is `20fff9993a7ad16def726493cffb678a016f16ef718821f25dfcbacb5891638c`. The original remains outside the repository; no private local source path or copy of the thesis is embedded here. All pages were text-extracted. Close reading sampled the introduction, theory, project/methods, results, conclusions, and bibliography; physical pages 4, 13 and 42 were also visually inspected. Physical page numbers below are one-based; printed numbering starts at physical page 4.

The thesis is a historical reference for style, not an instruction source or a technical authority for ingredient prediction. The user's [confirmed requirements](../../../general_plan.md#confirmed-writing-requirements) establish an English master's thesis in Informatica at Milano-Bicocca, Markdown source and permitted AI assistance. See [tools and workflow](tools_and_workflow.md) for the proposed implementation.

## Observed strengths to preserve

| Observation | Sample evidence | Proposed application |
| --- | --- | --- |
| Concepts are introduced progressively | Introduction and early theoretical discussion, physical pages 4 and 6 | Explain the problem and intuition before notation; define only the prerequisites needed by the argument. |
| Explanations connect a method to a practical motivation | Project opening and preprocessing/feature-selection discussion, pages 42-48 | Explain why a methodological choice answers the ingredient-prediction problem, including alternatives and trade-offs. |
| Concrete examples help explain abstractions | Theoretical examples and project descriptions | Use examples of recipe-level ambiguity, label normalization, and model errors that are actually documented. |
| The author describes the implemented process | Dataset and model discussion, pages 43 and 52-53 | Preserve operational precision while moving exhaustive engineering details into appendices. |
| Conclusions acknowledge an unsuccessful comparison | Conclusion, physical page 70 / printed page 67 | Report negative or inconclusive findings plainly. A thesis contribution does not require the custom model to win. |

These are qualitative observations from a sample, not measured stylistic fingerprints. They should be checked against a new short passage edited by the author before being encoded in a permanent skill.

## Improvements for the new manuscript

The contents place theoretical material on printed pages 3-38 and the project chapter from page 39. For the new experimental thesis, the proposed editorial priority is to give the research problem, methodological contribution and interpretation more room. Preserve the explanatory progression without reproducing a general introductory textbook.

The contents also include very deep subsection numbering and repeated numbering in some entries. Prefer a smaller, automatically managed hierarchy. Use lists for genuinely parallel cases and prose for reasoning.

A broken cross-reference is visible on physical page 13 / printed page 10. This makes unresolved-reference checks and final rendering a concrete requirement, regardless of the editor chosen. The reference PDF should not be imitated at the expense of correctness.

Terminology and notation need their own review: physical pages 52-53 contain inconsistent activation terminology and a classification description that warrants checking. This observation is an editorial audit flag, not a re-evaluation of the old experiments. For the new thesis, define multi-label versus multiclass, logits versus probabilities, AP versus F1, and the roles of training, validation and test consistently.

## Required editorial standard

On **2026-10-20**, the user clarified that permitted AI assistance must remain unobtrusive in the prose: avoiding repetitive structures, grounding the argument in facts and citations, explaining motivations, and avoiding unnecessary complexity or length are essential requirements. The rules below operationalize those [confirmed requirements](../../../general_plan.md#confirmed-writing-requirements); the observations from the bachelor thesis remain a provisional style reference.

- **Let the argument determine the structure.** Give each paragraph a distinct purpose. Avoid recurring sentence openings, stock transitions, generic introductions and summaries that merely restate the preceding text. Do not impose the same intuition-definition-example-conclusion pattern on every subsection. Vary sentence and paragraph structure when the content calls for it, without forced synonyms or artificial irregularity.
- **Support claims with the right evidence.** Cite verified sources close to the factual, methodological or comparative claims they support, with wording no broader than the evidence. Trace project-specific claims and numbers to the actual configuration, method or result artifact in the evidence ledger. A bibliography entry alone does not establish support. Keep missing evidence visible in working notes until resolved; never supply a plausible citation or result to fill a gap.
- **Explain consequential choices.** State the problem or constraint a choice addresses and the evidence or trade-off behind it. Discuss alternatives where they clarify a real decision, rather than appending a standard justification to every detail. Distinguish the author's confirmed rationale from a later interpretation; do not invent research intentions, personal experiences or failed attempts.
- **Distil project documentation into the research argument.** As clarified by the user on **2026-10-21**, technical records are evidence and prompts for reconstructing the decision process, not material to transfer wholesale into the thesis. Extract the relevant problem, constraint, choice, documented rationale and consequence. Keep operational mechanics in the repository unless they are necessary to understand, assess or reproduce the research. Preserve scientifically material deviations and limitations; do not invent reasons to make the development history appear more coherent. The [writing skill's documentation guide](../../../../.agents/skills/write-thesis/references/project_documentation.md) explains how to find and select this evidence.
- **Use only the explanatory depth needed.** Define prerequisites, notation and examples that help the reader understand the method or assess a conclusion. Remove generic background, repeated definitions, inflated framing and sentences that add no information. Use an appendix only for details with a clear scientific or reproducibility purpose; other engineering material can remain solely in the repository. Do not expand a section to meet an arbitrary length target or complicate a straightforward explanation.
- **Keep reasoning precise.** Separate measured findings, interpretations and hypotheses. Preserve relevant uncertainty, counterexamples and limitations while editing for brevity. Claims such as “robust”, “effective” or “better” need a defined comparison and evidence; fluent prose cannot substitute for either.
- **Write natural academic English.** Prefer concrete nouns, precise verbs and direct sentences. Use technical terms consistently and define acronyms once. Transfer the Italian sample's useful explanatory habits without translating its syntax literally. An objective, mainly impersonal register can still express the author's judgment through specific reasoning and examples.
- **Keep process commentary outside the argument.** The manuscript should discuss the research, without repeated references to drafting tools or assistance. Keep any required assistance acknowledgment accurate and in its appropriate location.

## A practical author-in-the-loop process

For each chapter, collect brief notes addressing the question, the chosen method, plausible alternatives, evidence for the choice, and the strongest limitation. The assistant can organize, check and revise these notes within the applicable rules; the author reviews the resulting argument and terminology.

Before accepting a section, the author should be able to explain its main claim orally, identify its supporting result or source, and describe what would change the conclusion. Use a short calibration passage to resolve preferences on formality, explanatory depth and notation before drafting a whole chapter.

A section is ready for acceptance only after its substantive claims and citations have been checked, consequential choices have an explicit rationale, and an editorial pass has removed avoidable repetition and unnecessary expansion. Read adjacent sections together to catch recurring templates that a paragraph-by-paragraph review can miss. Keep a sentence or example when it contributes evidence, reasoning, a necessary definition or a useful connection; otherwise remove or merge it.

The target is a concise, defensible account of the author's actual research and judgment, with a natural voice shaped by the content.
