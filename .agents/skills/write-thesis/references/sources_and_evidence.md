# Sources, bibliography and numerical evidence

Read this when searching literature, adding citations or checking substantive claims. Use `citation-management` for bibliographic mechanics, following the [tested commands and limitations](specialists.md). Bibliographic validity and evidential support are separate checks.

## Search and verify

Start from the project's existing source catalogs, then close the specific gap with original papers, proceedings, official repositories or documentation. Record query, source, date and selection rationale in research notes. Follow useful backward and forward citations. Prefer relevant complementary coverage over an arbitrary database or citation count.

The thesis related-work chapter is a targeted narrative synthesis unless a systematic review is explicitly requested and actually conducted. Do not label it systematic or add PRISMA counts from ordinary web searches. Treat abstracts and search snippets as discovery aids; obtain the passage, equation or table needed for the claim. If full text is unavailable, narrow the claim to the inspected material or mark it unresolved.

Check paper identity, version, publication status, corrections and retractions where relevant. Distinguish preprints from the published version; preserve genuinely absent metadata. Record the exact locator and how the source supports the wording. Label verification performed by the assistant as such; never record a fictitious human sign-off. Human review remains distinct from automated or assistant checks.

Synthesize agreements, differences and transfer limits. A result from another dataset, vocabulary, adaptation regime or budget does not establish superiority on this benchmark.

## Keep one bibliography authority

Use stable citation keys and Pandoc citations such as `[@He2016Deep]`. Maintain one `.bib` source. If a Zotero collection later becomes authoritative, treat its export as generated and bring corrections back to the library. Do not silently rekey existing citations or deduplicate entries that represent distinct versions without checking their identity.

Run bibliography checks on copies or explicit new outputs and inspect the diff. Keep source/evidence notes outside manuscript prose. Use the selected CSL style for final formatting when it becomes known; do not impose a journal's reference count or punctuation on a thesis by default.

## Claim ledger

For each substantive claim or coherent group of claims, retain a compact record in the manuscript's evidence notes. Choose CSV, JSON or a Markdown table to fit the existing workspace; do not require multiple parallel registries. The record needs:

| Field | Content |
| --- | --- |
| Claim ID and manuscript location | Stable identifier, section and current wording or precise statement. |
| Kind and evidence | Literature, method, result or interpretation; source/citation key or versioned project artifact. |
| Locator and identity | Page/section/table or artifact field; version/commit/hash where needed for reproducibility. |
| Verification | What was actually opened or computed, by whom/what, date and unresolved issues. |
| Scope and limits | Population/task, assumptions, supported inference and important exclusions. |

For numbers, also record metric definition and aggregation, split/cohort, vocabulary, checkpoint/selection rule, denominator, units, raw value, display rounding, uncertainty and replication unit where applicable. Recompute comparisons from comparable rows; distinguish a percentage-point difference from a relative percentage improvement. Check displayed text, captions and tables against the same verified value.

Changing a claim's meaning requires checking its support again. An existing ID or matching hash is not proof that the revised statement is true. An unverified result stays out of final prose even when the bibliography is complete.

## Numerical and methodological checks

Read the exact artifact named in the ledger. Compare the reported value, metric, split and cohort; do not accept a matching number from a different run. Recalculate differences and rounding. Trace confidence intervals to their method and independent sampling unit. Keep finite-sample intervals distinct from run-to-run variation.

Check comparability and the applicable Q1–Q4 boundary against the live methodology. Historical runs, smoke tests, HPO trials and final comparisons retain their actual roles. Neither metadata validation nor a generic citation checker can perform this scientific check.
