# Project documentation to manuscript

Read this when using repository knowledge to plan, draft or verify a thesis section. Apply the local [maintain-project-docs skill](../../maintain-project-docs/SKILL.md) to identify owners and navigate the live documentation. Its authority rules and the [editorial standard](../../../../docs/research/topics/thesis_writing/author_voice.md#required-editorial-standard) govern this guide.

## Locate evidence for the question

Start from [repository knowledge](../../../../README_PROJECT_KNOWLEDGE.md) as a map and the [documentation index](../../../../docs/README.md) for category entry points. The general plan, already read by the writing skill, establishes current gates and status. Search the relevant category by concept, decision name or experiment identifier; inspect its README and follow links to the owning section and underlying evidence. Do not load the entire documentation tree.

| Need in the section | Start here and follow the evidence |
| --- | --- |
| Research question, scope, chosen method and rationale | [Project objectives](../../../../docs/project_objective/README.md), especially [problem definition](../../../../docs/project_objective/problem_definition.md), [benchmark decisions](../../../../docs/project_objective/benchmark_decisions.md) and [comparison methodology](../../../../docs/project_objective/model_comparison_methodology.md). |
| Measured result, comparison or limitation | [Reviewed experiment results](../../../../docs/experiment_results/README.md), then the exact linked configuration and result artifact. |
| Alternatives considered and external scientific support | [Research](../../../../docs/research/README.md), including topic records and dated discoveries, then the original papers or official sources. A recommendation is not an adopted decision. |
| Architecture or conceptual explanation | [Model deep dives](../../../../docs/models_deepdive/README.md) and [technical explanations](../../../../docs/technical_details/README.md); verify the relevant model variant, assumptions and original sources. |
| What was actually implemented | [Implementation contracts](../../../../docs/implementation_details/README.md), then code, configuration and tests for the relevant version. |
| When a decision changed or whether a step happened | [General plan](../../../../docs/general_plan.md) and linked [feature plans](../../../../docs/plans/README.md); follow their evidence links rather than treating a task checkbox as proof of a result. |

For each consequential statement, establish the record's role and time: adopted decision, implemented behavior, measured result, interpretation, proposal or superseded history. Match the version to the experiment being described. A newer implementation cannot establish how an older run behaved. Resolve conflicts at the owner using `maintain-project-docs`; if resolution needs missing evidence or an author's rationale, leave a targeted note and continue unaffected work.

Use scratch outputs only through their reviewed record or after checking and documenting their provenance. Internal notes locate evidence; they do not replace primary literature citations. Preserve a source path and section, artifact/version and verification limits in the existing [claim ledger](sources_and_evidence.md#claim-ledger), without creating a parallel registry.

## Select what the reader needs

For a relevant decision, extract the problem or constraint, the adopted choice, the recorded rationale or trade-off, and its consequences or limits. These are evidence-selection questions, not a mandatory paragraph template. A technical detail can reveal a useful motivation without itself belonging in the manuscript. Distinguish contemporaneous reasons from later interpretation; code alone does not establish why the author chose it.

Choose the destination by function:

- **Main text:** what the reader needs to understand the contribution, follow a consequential decision, assess validity or interpret a result. Translate internal phase names and code identifiers into scientific concepts while retaining exact model, dataset and metric identities.
- **Appendix or reproducibility material:** exact settings or procedures that are needed to reproduce or audit the study but would interrupt the argument. Include only useful material and point to it where necessary.
- **Repository or evidence notes only:** routine paths, launcher commands, class inventories, dashboard plumbing, environment troubleshooting, checklists and intermediate debugging. Do not move them automatically into an appendix simply because they exist.

The boundary depends on scientific relevance, not on whether a detail is technical. Preprocessing, split isolation, label construction, training/selection rules, metric definitions and uncertainty can be essential methods. An implementation failure or protocol amendment belongs in the account when it affects evidence quality or interpretation. Do not omit unfavorable findings or outcome-informed changes to simplify the story.

Examples of selection, not ready-made thesis claims:

- A memory-limit record may support an explanation of the feasible training configuration and its trade-off. Verify that link in the decision record and the run configuration; retain relevant effective-batch or precision settings, while leaving shell commands and allocation-debugging history in the repository.
- A data-split record may support explaining how duplicate images were prevented from crossing partitions. Report the verified grouping rule and leakage boundary; function names and metadata storage paths usually remain in the evidence notes. Do not broaden exact-duplicate protection into a claim that all leakage was eliminated.

Before accepting the passage, check both directions: every included claim has adequate support, and removing implementation detail has not removed a condition, limitation or deviation that changes the scientific meaning.
