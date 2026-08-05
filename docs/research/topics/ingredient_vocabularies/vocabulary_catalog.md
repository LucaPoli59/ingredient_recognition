# Controlled food and ingredient vocabulary catalog

**Created:** 2026-08-04  
**Last updated:** 2026-08-04

## Purpose

This document compares the principal reusable vocabularies and food-description resources identified while researching canonical ingredient concepts. It describes their general design and likely roles without assuming the Yummly dataset or a particular machine-learning task. Dataset-specific measurements and recommendations are kept separately in [`controlled_vocabulary_evaluation.md`](../../../plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md).

## Comparison overview

| Resource | Resource type and primary purpose | Structure and identifiers | Offline release | Best general use | Main limitation as an ingredient-label vocabulary |
| --- | --- | --- | --- | --- | --- |
| FoodOn | Actively maintained ontology for food materials, food products, sources, processes, qualities, and related entities | Globally unique OBO-style identifiers, preferred labels, typed synonyms, definitions, relations, and a polyhierarchy | OWL plus a generated tabular hierarchy/synonym export; versioned GitHub releases | Canonical food concepts, semantic integration, explicit aliases, and durable cross-dataset references | Its scope is broader and often finer than recipe ingredients; lexical ambiguity and multiple inheritance prevent automatic selection of a visually meaningful target level |
| LanguaL | Multilingual faceted thesaurus for describing and retrieving food-composition and consumption records | Unique facet-term codes across product type, source, plant/animal part, physical state, treatment, packaging, dietary use, and other facets | XML and tab-delimited exports; latest thesaurus version is LanguaL 2017 | Legacy food-database interoperability and compositional food descriptions | It is a descriptor system rather than one flat ingredient concept list; the maintainers identify FoodOn as its successor |
| FoodEx2 | EFSA classification and description system for food-intake and exposure assessment | Core and extended food terms, parent-child aggregation, multiple domain hierarchies, and additional facets | Distributed through EFSA catalogues and coding tools | Regulatory food coding, exposure aggregation, and domain-specific reporting | A coded food description may require a base term plus facets; its levels reflect exposure use cases rather than visual ingredient recognition |
| FoodData Central | USDA food-composition data platform rather than a controlled ontology | Data-type-specific food records, descriptions, food categories, FDC IDs, NDB numbers, FNDDS food codes, and GTINs | Versioned JSON and CSV downloads plus an API | Nutrient composition, analyzed foods, dietary-survey foods, and branded-product records | Descriptions identify food records and forms, not reusable ingredient concepts; FDC IDs change when a food record is updated and categories differ by data type |

## FoodOn

[FoodOn](https://foodon.org/) is the strongest general-purpose semantic resource among the candidates. Its food-product branch aims to provide generic, unambiguous food product categories rather than manufacturer-specific products. FoodOn also contains organismal sources, anatomy, processes, qualities, and imported terms from other ontologies, so consumers must select an explicit branch rather than index the entire ontology indiscriminately.

The resource provides:

- OBO-style identifiers such as `FOODON_03305833` for `english muffin`;
- preferred labels and typed exact, broad, and narrow synonyms;
- parent relations and multiple inheritance;
- deprecation annotations and replacement links while retaining deprecated terms;
- versioned OWL releases and a generated `foodon-synonyms.tsv` representation suitable for deterministic offline indexing.

FoodOn labels are normally singular, and its curation guidance expects plural recognition to be handled by text-processing tools. Some broader classes deliberately carry a `food product` suffix to disambiguate them from leaf items; exact suffix-free synonyms are intended for text matching. These conventions make a bounded plural rule and label/synonym precedence appropriate, but they do not justify fuzzy matching.

FoodOn is not a ready-made visual-recognition label space. Its product hierarchy is polyhierarchical and still evolving, and parents can express product type, source, processing, or other semantic axes. Automatically climbing a fixed number of parent edges therefore cannot guarantee either a consistent ingredient identity or practical recognizability in a prepared dish.

Primary references:

- [FoodOn repository and release format](https://github.com/FoodOntology/foodon)
- [FoodOn v2025-07-31 release](https://github.com/FoodOntology/foodon/releases/tag/v2025-07-31)
- [Food product hierarchy](https://foodon.org/food-facets/food-product/)
- [Curation rules for labels, plurals, synonyms, and deprecation](https://foodon.org/design/curation-rules/)
- [Tabular hierarchy and synonym export](https://foodon.org/reuse-project/reuse-technical/foodon-instant-formula/)

## LanguaL

[LanguaL](https://www.langual.org/) is a multilingual food-description thesaurus organized as a faceted classification system. A food is represented by a combination of descriptors rather than by selecting a single canonical ingredient node. Its fourteen facets include product type, food source, plant or animal part, physical form, heat treatment, cooking method, preservation, packaging, consumer group, geography, and adjunct characteristics.

LanguaL terms have stable facet-term codes and broader/narrower relationships. The official downloadable LanguaL 2017 export contains 12,605 descriptors in XML and tabular formats and incorporates historical FoodEx2 and GS1 classification material. The official site now presents FoodOn as the successor that carries LanguaL-derived material into an interoperable ontology environment.

LanguaL remains useful when reading older food-composition databases or decomposing a food description into facets. It is less suitable as the sole source of recipe ingredient targets because product identity, source, physical state, and processing live in separate descriptor branches, and matching across all facets can confuse ingredient identities with properties or processes.

Primary references:

- [LanguaL overview and relationship to FoodOn](https://www.langual.org/Default.asp)
- [LanguaL 2017 downloads](https://www.langual.org/langual_downloads.asp)
- [LanguaL systematic facet display](https://www.langual.org/langual_thesaurus.asp)

## FoodEx2

[FoodEx2](https://www.efsa.europa.eu/en/data/data-standardisation) is maintained by the European Food Safety Authority for standardized food classification and description in intake and exposure assessment. It separates a core list from more detailed extended terms, permits aggregation through parent-child relationships, and adds facets that describe properties not captured by the base food term. EFSA's current description reports nine hierarchies: six domain-specific hierarchies, two supporting exposure tools, and one service hierarchy.

FoodEx2 is valuable when a project must exchange or aggregate food-safety and exposure data. It is not a simple synonym dictionary for recipe ingredients. Selecting the correct base term and facets is a coding task, and its core/extended distinction expresses regulatory detail requirements rather than whether an ingredient can be recognized from a photograph.

FoodEx2 should therefore be treated as a possible interoperability reference, not as an automatic target taxonomy. Historical FoodEx2 terms also appear inside LanguaL and FoodOn, but those imports must not be mistaken for the current EFSA catalogue.

Primary references:

- [EFSA FoodEx2 structure and tools](https://www.efsa.europa.eu/en/data/data-standardisation)
- [EFSA Catalogue Browser](https://github.com/openefsa/catalogue-browser/wiki)

## FoodData Central

[FoodData Central](https://fdc.nal.usda.gov/) integrates several USDA food-data types: Foundation Foods, Experimental Foods, FNDDS, Branded Foods, and the final SR Legacy release. Its JSON and CSV downloads are excellent sources for nutrient profiles, analyzed samples, dietary-survey foods, and commercial products.

Its identifiers describe database records rather than ontology concepts. USDA documents that an `FDC_ID` changes when the corresponding food record changes; NDB numbers remain stable for specific Foundation and SR Legacy food forms, while FNDDS and branded data use other identifiers. Food categories also differ by data type and purpose. Consequently, FoodData Central descriptions such as `Hummus, commercial` or detailed raw/cooked product forms are not a controlled synonym space that can be used directly as canonical recipe ingredient labels.

Primary references:

- [FoodData Central data-type documentation](https://fdc.nal.usda.gov/data-documentation/)
- [Downloadable releases](https://fdc.nal.usda.gov/download-datasets/)
- [Identifier and category definitions](https://fdc.nal.usda.gov/help/)

## General conclusions

1. FoodOn is the best starting point when durable concept identity, food-product coverage, synonyms, and offline operation are required together.
2. No investigated resource directly defines visual recognizability. A project using food images still needs an explicit task-specific policy for merging distinctions such as product style, preparation state, or source family.
3. LanguaL and FoodEx2 are compositional coding systems. Treating all of their facet descriptors as interchangeable ingredient labels creates semantic collisions.
4. FoodData Central should complement an ontology with composition or product-record data, not replace the ontology.
5. A safe mapping system must permit unresolved local concepts. External coverage is necessarily incomplete for culturally specific products, colloquial recipe language, and new products.
