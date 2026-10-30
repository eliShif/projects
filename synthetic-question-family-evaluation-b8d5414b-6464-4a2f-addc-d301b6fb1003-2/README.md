# Synthetic question-family evaluation

An offline Python experiment that asks whether lexical features can classify
synthetic questions into eight task families for unseen service entities.
Version **2.1.0** adds explicit train–validation gap analysis to the existing
generator, majority baseline, unigram and unigram-plus-bigram multinomial Naive
Bayes models. It reuses their metrics, entity-isolation checks, validation-only
selection and sensitivity analyses. Source provenance is in [UPSTREAM.md](UPSTREAM.md).
No original remote was changed.

## What changed

Each candidate is now evaluated on the same training cases used to fit it, as
well as on validation. `summary.json` preserves full `training` metrics and a
`train_validation_review` with accuracy and macro F1 for each candidate, signed
gaps and directions. `REPORT.md` explains how to interpret these observations
alongside the existing template and label-distribution audits. The gap function
never receives test or challenge metrics, and the selection rule is unchanged.
No additional runtime library or model is needed.

The focused tests check known positive, zero and negative gaps, invalid scores,
and independence of selection from training scores and changed test wording.
The smoke check independently recalculates aggregate metrics from the saved
confusion counts and rejects corrupted gap values.

### Retained split and class-support safeguards

The old generator chose comparison partners from one global list, then assigned
splits using only the primary entity. For example, an Ember training question
could mention Fjord, which belonged to validation. The revised path assigns
entity pools **before generating questions**, and both operands come from the
same pool. A second validation entity, Ibis, makes distinct-entity comparisons
possible while retaining the original three training and two test entities.

| Split | Service roots | Cases |
|---|---|---:|
| Train | Cobalt, Delta, Ember | 24 |
| Validation | Fjord, Ibis | 16 |
| Test | Garnet, Helix | 16 |

Every pool has at least two distinct names. The validator scans all known names
in each question, including secondary mentions, case variants and the fictional
`Sandbox`/`Mirror` aliases. Aliases belong to their base service's pool. The
post-selection entity folds also remove training cases containing any service
mentioned by their holdout. The shared generator's default `dev` and `test`
outputs remain identical; only this lab opts into the new pools.

Counts always include all eight `FAMILIES`. `validate_cases` lists missing
validation/test families and returns JSON `null` for an undefined imbalance
ratio, rather than omitting zero-count classes. Missing training families are
rejected. The full experiment requires every family in **every split** and
rejects missing support before fitting or writing outputs. This keeps validation
selection and the complete paired diagnostic meaningful. Per-family metric
formulas retain the existing zero-denominator convention, including zero F1 for
absent labels in the fixed eight-family macro average.

## Setup and installation

Run commands from this directory. Python 3.11+ is required; the recorded execution
uses Python 3.14.4. The classifier and tests use only the standard library.
Packaging uses setuptools 78.1.1, already available in this workspace. These
commands install the actual local package without contacting a package index:

```text
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check --no-cache-dir install --no-index --no-build-isolation --no-deps .
```

The system-site option exposes the preinstalled build tool to the virtual
environment; it does not add runtime dependencies to this lab. On another
machine without that build tool, install `setuptools==78.1.1` into the virtual
environment from an available package index or local wheel first. A networked
setup command is:

```text
.venv/bin/python -m pip --isolated --disable-pip-version-check --no-cache-dir install setuptools==78.1.1
```

Source execution also works with `python3` directly and needs no installation.
The other inherited portfolio modules and their requirements files are outside
this experiment's runtime path. No model weights or API credentials are needed.

## Test, build and run

```text
.venv/bin/python -m unittest discover -s tests -p "test_research*.py" -v
.venv/bin/python -m unittest discover -s tests -p test_evidence_agent.py -v
.venv/bin/python -m compileall -q portfolio tests scripts
.venv/bin/python -m portfolio.research_lab --out results/research_lab
.venv/bin/python scripts/smoke_research_lab.py --out results/research_lab --installed
```

The build command checks Python compilation; the installation command also
builds a real wheel. The finite smoke executes the documented module command,
checks artifact hashes, counts, train/validation/test metrics and signed gaps,
repeats it using the saved JSON input, and
compares all six artifacts byte for byte. It also verifies an installed-package
run with Python `-I` (excluding the checkout from import lookup), then confirms
that a leaked secondary entity and a missing validation family fail without
creating result directories. Each child run has a 30-second timeout.

To record installation, both test suites, compilation and smoke output with
actual exit codes, versions and SHA-256 digests, and regenerate `VALIDATION.md`:

```text
.venv/bin/python scripts/verify_research_lab.py --install
```

See [VALIDATION.md](VALIDATION.md) and
[results/research_validation/commands.json](results/research_validation/commands.json).
Raw command output is saved alongside the manifest. These are local automated
checks, not evidence of a person's authorship, understanding or completed review.
The unrelated full repository suite was not run for this adaptation.

## Selection and results

1. Generate or read cases, audit all entity mentions, label consistency and
   complete family support. Fit all three methods on **train only**, then
   evaluate those same fitted objects on train and validation. Training scores
   describe fit on seen data; they are not held-out estimates.
2. Rank by validation macro F1, then validation accuracy, then alphabetical model
   name. Write the decision, validation scores, training IDs, settings and input
   hashes to `selection.json` **before the first test prediction**. Do not refit
   on validation or change the selection after seeing test results. Calculate
   train-minus-validation gaps for description only; they do not rank models.
3. Evaluate all frozen candidates on the same test cases for comparison. Evaluate
   alternate wording for those same entity/family pairs as a diagnostic.
4. Run the existing composition, resampling, score-margin and error analyses.
   They never select or tune the primary model.

The recorded run selects bigram by the deterministic tie-break: unigram and
bigram both have validation macro F1 of 1.0. The selected model gets 16/16 original
test cases correct, versus 2/16 for majority. It gets 10/16 alternate-wording
cases correct; all six failures are listed in
[REPORT.md](results/research_lab/REPORT.md). These measured values describe a tiny
constructed dataset, not a general performance claim or an improvement over the
older dataset. All 16 original test templates also occur in training after
masking entity names.

### Reading the train–validation gaps

For either metric, **gap = training score − validation score**. Values are
fractions: 0.10 is 10 percentage points. Positive means training scored higher;
negative means validation scored higher. JSON retains full precision; the report
rounds to six decimals. The two metrics can have different signs because
accuracy weights cases and macro F1 weights the eight families equally.

The default generated run has zero gaps for all three candidates. Unigram and
bigram have train/validation accuracy and macro F1 of 1.0. Majority has accuracy
1/8 and macro F1 1/36 on both splits. Equal scores alone cannot establish good
generalization: majority is weak on both sets, while all 16 validation templates
also appear in training after masking entities. The separate alternate-wording
diagnostic exposes sensitivity that a zero gap can conceal.

Overfitting means learning training-specific patterns that do not generalize.
High training scores with lower validation scores are consistent with it when
the tasks are comparable. A positive gap may also reflect wording/template
shift, class proportions, difficulty, label noise or small samples. Check the
template audit, label counts and failure examples before attributing a cause.
A negative gap does not guarantee generalization either. There is no automatic
overfitting verdict, significance test, promised gap size or performance gain.
Training scores reuse fitted data, and validation scores have already informed
selection. Further tuning after inspecting test/challenge results would require
a fresh evaluation protocol and new held-out data.

| Artifact | Purpose |
|---|---|
| `cases.json` | Exact 56-case input snapshot |
| `challenge_cases.json` | Sixteen matched alternate-wording cases |
| `protocol.json` | Experiment version, split policy, metrics, source/input hashes and Python version |
| `selection.json` | Persisted validation-only choice before test evaluation |
| `summary.json` | Full train/validation/test metrics, signed train–validation gaps and interpretation, confusion matrices, entity mentions, failures and sensitivity results |
| `REPORT.md` | Readable measured comparisons and limitations |

Replay or inspect a copied input:

```text
.venv/bin/python -m portfolio.research_lab --cases results/research_lab/cases.json --out results/research_lab_replay
```

The five string fields are exactly `id`, `entity`, `question`, `label`, `split`.
The accepted primary names are the catalogue above. Every split must be nonempty;
IDs must be unique; blank fields, contradictory labels for identical questions,
cross-split duplicates and recognized foreign mentions are rejected. Generated
IDs must still match their generator entity and family. Custom IDs are reported
as unmatched in the provenance audit. The challenge requires exactly one test
case per designated entity/family pair. The input must also contain at least two
training `lookup` examples for the retained downsampling diagnostic.
CLI input errors exit nonzero. A failed
validation does not overwrite existing outputs; it does not certify an older
output directory as a result of the failed run.

## Limits and learning

The synthetic English questions share authored templates across splits. Entity
isolation does not remove that shortcut. Alternate wording was authored after
inspecting those templates and is an exploratory diagnostic. Name detection
covers the closed synthetic catalogue, not arbitrary entities in external text;
external corpora need their own audited entity inventory. Generated labels have
no independent human annotation review. Score margins are not calibrated
probabilities, and the variance across folds or reused training samples is not
a confidence interval. Observed feature patterns do not establish error causes.

This is an AI-assisted implementation and preparation resource. It does not
establish personal mastery, independent implementation, supervised experience,
Hebrew or language-model performance. Use [CODE_MAP.md](CODE_MAP.md) and
[EXERCISES.md](EXERCISES.md) to run, change and explain it in a future review.
