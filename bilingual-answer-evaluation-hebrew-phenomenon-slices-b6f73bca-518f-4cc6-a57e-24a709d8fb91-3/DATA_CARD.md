# Hebrew challenge corpus v1

## Source and scope

`cola_lab/resources/hebrew_phenomena.json` is a local, AI-assisted set of constructed fictional object facts. No real people, scraped data, private repository content or external corpus are included. `source` in the JSON records this provenance. Labels are source-derived annotations, not independent human judgments. Independent linguistic review has **not** been performed.

This corpus exercises the existing evaluator; it is not an estimate of naturally occurring Hebrew usage or an exhaustive linguistic test set. The controls use Latin fictional entity identifiers, as the original corpus does. The mixed-language tag concerns replacing a Hebrew content word with English, not simply detecting Latin characters in an identifier.

## Partitions and expected labels

| Split | Entity | Phenomenon | English answer | Hebrew answer | Source attribute |
| --- | --- | --- | --- | --- | --- |
| validation | Alder | inflection | blue | כחול | color |
| validation | Birch | spelling | square | מרובע | description |
| validation | Cedar | mixed_language | red | אדום | color |
| validation | Dune | inflection | gold | זהוב | color |
| test | Ember | inflection | blue | כחול | color |
| test | Fjord | spelling | square | מרובע | description |
| test | Grove | mixed_language | red | אדום | color |
| test | Harbor | inflection | gold | זהוב | color |

Each row expands to two source facts (target and shape distractor) per language, two answerable question forms per language, and one missing-weight question per language. Each split therefore contains 16 documents and 24 cases: 16 answerable, 8 unanswerable. Each Hebrew variant slice contains two inflection cases, one spelling case and one mixed-language case per split. Every answerable form points to the target source ID; weight has `expected_answer: null` and `evidence_ids: []` because no source states weight.

`variant_group` is `entity/phenomenon`, independent of split. Every annotated group must contain exactly English control, English variant, Hebrew control and Hebrew variant in one split. Case IDs include attribute and form, preventing language-pair collisions. Entity-level isolation also keeps the distractor and missing-weight questions out of the other partition. There is no training partition: this evaluator selects a threshold on validation and evaluates test; it fits no learned model.

## What each annotation means

- **inflection**: `מה הצבע של Ember?` becomes `מה צבעו של Ember?`. The latter uses a third-person masculine singular possessive suffix. The fictional referent is treated as a masculine object. This does not test gender inference or broad inflectional morphology.
- **spelling**: `מה התיאור של Fjord?` becomes `מה התאור של Fjord?`. The description noun loses the yod in the deficient spelling. The supporting source retains `התיאור`; both questions still ask for `מרובע`. This does not establish that every spelling variant is interchangeable in every context.
- **mixed_language**: `מה הצבע של Grove?` becomes `מה ה-color של Grove?`. The English attribute appears within Hebrew syntax with a Hebrew definite article and a retained hyphen. This intentionally constructed form is not a frequency claim about natural code-switching.

English variants use `Which color/description does ENTITY have?` instead of `What is the color/description of ENTITY?`. They are paired semantic controls, not examples of Hebrew phenomena. They are not counted in Hebrew slices. Altering both English and Hebrew wording prevents treating aggregate bilingual differences as a causal language experiment.

## Cleaning and duplicate policy

`clean_text` normalizes Unicode to NFC and collapses whitespace. It does not stem, lemmatize, translate, remove letters/marks/punctuation, change script or apply spelling correction. The raw Fjord deficient-spelling question intentionally includes tabs/outer spaces to exercise this path. The loader retains raw strings and the validator checks that cleaning them yields exactly the saved inputs.

Duplicate detection uses `(language, cleaned_text.casefold())` separately for documents and questions across the entire dataset. This rejects accidental copies even under different IDs or whitespace. Deliberately related but nonidentical forms are retained and grouped. No semantic or approximate duplicate detection is claimed. The existing retrieval tokenizer separately case-folds and extracts word tokens; its treatment of hyphens is part of the unchanged word-overlap rule.

## Grammar and label validation

`phenomena.question_target` checks the exact form declared by the annotation; it does not trust a tag without examining the question. It parses the entity, maps the declared form to `color` or `description`, then the existing validator resolves that entity/attribute in the complete same-split/same-language source corpus. The answer and citation must both agree with that fact. Null labels are rejected when a fact exists, including when retrieval misses it.

The challenge additionally supports the explicit description sources `The description of ENTITY is VALUE.` and `התיאור של ENTITY הוא VALUE.`, and feminine shape distractors `הצורה של ENTITY היא VALUE.`. Original source/question templates remain supported. Unknown wording is rejected instead of silently interpreted as an absent answer. Supporting additional natural text requires deliberate grammar/data changes and independent review beyond these software checks.

Tests in `tests/test_phenomena_data.py` specify literal expected queries, values and source sentences separately from parsing logic, then corrupt references, groups, grammar and raw text to exercise rejection. They establish software behavior, not independent linguistic validation.
