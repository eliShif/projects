# Synthetic question-family evaluation

An offline Python experiment that asks whether lexical features can classify
synthetic questions into eight task families for unseen service entities.
Version **2.1.0** adds train–validation learning curves to the existing generator,
majority baseline, unigram and unigram-plus-bigram multinomial Naive Bayes models.
It measures generalization gaps across predeclared training sizes while retaining
the earlier secondary-entity isolation, missing-family checks and frozen model
selection. Source provenance is in [UPSTREAM.md](UPSTREAM.md).
No original remote was changed.

## What changed

The new diagnostic refits the existing three methods on nested subsets of
**8, 16 and 24 training cases**, with seed **19** and all eight families at every
size. Each method sees identical subset IDs and the same 16 validation cases.
It records training and validation accuracy, macro F1, confusion matrices and
signed train-minus-validation gaps. No test labels enter this calculation or
the validation-only selection. A regression reruns the experiment with every
test label permuted, checking that the choice and curve stay identical while
the test scores change.

The prior revision already fixed a generator that chose comparison partners from one global list, then assigned
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
checks artifact hashes and counts, repeats it using the saved JSON input, and
compares all six artifacts byte for byte. It recalculates curve accuracy and
macro F1 from confusion matrices and checks gaps, nested IDs, class support and
entity isolation. It also verifies an installed-package
run with Python `-I` (excluding the checkout from import lookup), then confirms
that a leaked secondary entity, a missing validation family and infeasible
training sizes fail without
creating result directories. Each child run has a 30-second timeout.

To record installation, both test suites, compilation and smoke output with
actual exit codes, versions and SHA-256 digests:

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
   complete family support. Validate the curve schedule and write it to the
   protocol before fitting. Fit all three primary methods on **all train rows only**.
2. Rank by validation macro F1, then validation accuracy, then alphabetical model
   name. Write the decision, validation scores, training IDs, settings and input
   hashes to `selection.json` **before the first test prediction**. Do not refit
   on validation or change the selection after seeing test results.
3. Evaluate all frozen candidates on the same test cases for comparison. Evaluate
   alternate wording for those same entity/family pairs as a diagnostic.
4. Refit separate models for the train–validation learning curve using only
   development data. Run the existing composition, resampling, score-margin and
   error analyses. None select or tune the primary model.

The recorded run selects bigram by the deterministic tie-break: unigram and
bigram both have validation macro F1 of 1.0. The selected model gets 16/16 original
test cases correct, versus 2/16 for majority. It gets 10/16 alternate-wording
cases correct; all six failures are listed in
[REPORT.md](results/research_lab/REPORT.md). These measured values describe a tiny
constructed dataset, not a general performance claim or an improvement over the
older dataset. All 16 original test templates also occur in training after
masking entity names.

In the recorded 2.1.0 learning curve, unigram and bigram each score **1.0 accuracy
and macro F1 on both train and validation at all three sizes**, so their measured
gaps are **0.0**. Majority scores 0.125 accuracy and 1/36 macro F1 on both sets,
also with zero gaps. No positive gap was observed on these shared-template data;
this is not evidence that overfitting is absent on unfamiliar language. The
numerical controls in the tests are separate from these measured results.

| Artifact | Purpose |
|---|---|
| `cases.json` | Exact 56-case input snapshot |
| `challenge_cases.json` | Sixteen matched alternate-wording cases |
| `protocol.json` | Experiment version, split policy, metrics, source/input hashes and Python version |
| `selection.json` | Persisted validation-only choice before test evaluation |
| `summary.json` | All candidate metrics, `learning_curve` subset IDs/counts/hashes, train/validation scores and gaps, confusion matrices, entity mentions, failures and sensitivity results |
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
The curve schedule must contain at least two strictly increasing integer sizes,
each from 8 through the number of available training rows. The default therefore
requires at least 24 training rows; predeclare smaller sizes for a smaller input.
CLI input errors exit nonzero. A failed
validation does not overwrite existing outputs; it does not certify an older
output directory as a result of the failed run.

## Learning curves and overfitting

`learning_curve_subsets` sorts each family's IDs, shuffles with a local seeded
random generator, then interleaves family queues. Prefixes give nested samples
without replacement. With the default balanced data, each family contributes
one, two and three rows. Unbalanced inputs still retain every family. All known
references, including secondary operands, remain inside the original train pool;
a referenced service need not have a primary row in a small subset.

For each size and method, the training metric is **resubstitution**: scoring the
same examples used to fit that model. Validation always uses the complete,
unchanged validation set. A gap is `training metric - validation metric`, with
the same eight-family macro-F1 definition on both sides. Positive means training
scored higher; negative gaps are retained. A gap of 0.10 is ten percentage points.

Overfitting involves fitting training-specific patterns that generalize poorly.
High training performance with lower validation performance is consistent with
it, but sampling variation and distribution differences can also produce a gap.
Low scores on both sets can suggest underfitting. Zero gaps do not rule out
overfitting: these tiny splits reuse the same templates. The alternate-wording
challenge changes the input distribution; a drop there alone does not diagnose
overfitting. Neither a positive gap nor an improvement is required for a valid run.

To predeclare another diagnostic schedule, before examining its output:

```text
.venv/bin/python -m portfolio.research_lab --train-sizes 8 24 --curve-seed 23 --out results/research_curve_alternative
```

Sizes and seed change only this diagnostic. The primary candidates still fit
all training rows and selection still uses validation. Searching sizes or seeds
after looking at results is exploratory; repeated use of this small validation
set does not create an independent benchmark. The report includes every declared
point, rather than choosing a favorable gap.

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
