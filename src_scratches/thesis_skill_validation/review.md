# Bounded section-review exercise

**Created:** 2026-10-21  
**Last updated:** 2026-10-21

The assistant applied the project writing guidance to [draft_bad.md](results/draft_bad.md), the [raw synthetic result](results/result.json) and [bibliography](results/references.bib). This is a transparent self-check on a constructed example, not a blinded independent evaluation or an assessment of actual project performance.

| Finding | Evidence and disposition | Check type |
| --- | --- | --- |
| Missing source | `MissingStudy2026` has no bibliography entry. The validator returned exit 1 and named the missing key; remove the unsupported citation and claim rather than fabricate a paper. | Automated citation check plus assistant review. |
| Unsupported number | Draft reports `0.470`; the fixture's `value` is `0.410`. Correct the number. | Assistant comparison with raw JSON; the citation validator does not detect this. |
| Wrong evaluation split | Draft says test; the artifact says validation. No test result exists in the fixture. | Assistant scope review. |
| Unsupported stability claim | Only seed 42 is present and uncertainty is absent. “Robust generalization across seeds” is unsupported. | Assistant interpretation review. |
| Real background citation | DOI metadata identifies He, Zhang, Ren and Sun's CVPR 2016 paper. The [authors' arXiv abstract](https://arxiv.org/abs/1512.03385) supports the limited statement that it introduces a residual-learning framework. The CVF page returned 403 during web access; no claim of full-paper verification is made. | DOI lookup plus assistant inspection of the abstract. |

The [corrected passage](results/draft_corrected.md) uses the actual fixture value and split, removes the unverified source and limits the interpretation to one supplied seed. The bibliography check then returns exit 0. Neither result validates real thesis claims.

The [PNG](results/fixture.png) was visually inspected: its synthetic status is explicit, the value is `0.410`, the metric label and zero baseline are legible, and no clipping or overlapping labels was observed. The [PDF](results/fixture.pdf) passed metadata inspection; its page layout was not separately visually inspected. No uncertainty interval was invented. Filesystem paths in retained JSON reports were normalized to relative fixture paths after execution; numerical results and return codes were unchanged.

The [machine record](results/validation.json) also confirms the observed parser limitations: `[-@SuppressedMissing]` is missed, `[@Key.WithDot]` becomes `Key`, a code-span handle becomes `CodeExample`, and appending an unterminated BibTeX entry leaves only the original entry parsed. These findings are carried into the [specialist guide](../../.agents/skills/write-thesis/references/specialists.md).
