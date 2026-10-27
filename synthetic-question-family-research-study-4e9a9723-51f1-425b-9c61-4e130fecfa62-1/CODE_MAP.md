# Code map

| Path | Role |
|---|---|
| `portfolio/research_cases.py` | Existing deterministic synthetic service records and eight question families; source reused by the new experiment. |
| `portfolio/research_lab.py` | Group split, input validation and generator label ledger check, normalized label distribution and split comparison, exact template-overlap audit, token features, majority and Naive Bayes models, validation selection, paired wording transitions, group stability with fold variance, imbalance stress, error review, metrics and output files. |
| `tests/test_research_lab.py` | Checks deterministic reproduction, split separation, paired outcomes, diagnostic behavior, failure metrics, and rejection of malformed labels, repeated-question leakage or incomplete or mismatched pairs. |
| `results/research_lab/cases.json` | Generated input snapshot; optional `--cases` reads this schema from disk. |
| `results/research_lab/challenge_cases.json` | Alternate authored wording for the same held-out entity/family pairs. |
| `results/research_lab/protocol.json` | Question, split, model selection rule, metric definitions, versions and source/input hashes. |
| `results/research_lab/summary.json` | Machine-readable quality checks, validation/test metrics, confusion matrices and failure cases. |
| `results/research_lab/REPORT.md` | Short human-readable findings and limits from the actual smoke run. |
| `portfolio/retrieval.py` | Existing scikit-learn TF-IDF retrieval experiment; related but independent from the standard-library classifier. |

Flow: `build_cases("test")` → assign whole entities to train/validation/test →
validate and check generated labels against the source ledger → fit three methods on train → select by validation macro F1 → evaluate
all methods on the frozen test set and a paired wording diagnostic → write results.
After selecting the method, `grouped_stability` retrains that fixed method with
each entity held out in turn. `imbalance_stress` changes only the number of
`lookup` training examples and compares the resulting held-out metrics.
`diagnostic_review` records per-error label counts and unseen feature fractions;
it does not assign a cause. These post-hoc checks are not inputs to selection.
`paired_wording_review` requires one alternate for each original test case,
with the same entity and label. It records correct-to-correct, correct-to-wrong,
wrong-to-correct and wrong-to-wrong transitions plus per-family counts. A score
change alone cannot show which examples changed prediction quality.
`template_overlap` masks known service names, preserving the synthetic
`Sandbox` and `Mirror` suffixes, and marks each evaluation question whose
normalized wording matches a training question. The saved audit observes
16/16 original test matches and 0/16 challenge matches. It detects exact
template reuse only; it does not measure semantic similarity or prove the
cause of any individual prediction.
`label_distribution_review` divides each family count by its split size,
computes the unweighted mean and population variance of each fraction over
train, validation and test, and compares validation/test with training using
total variation distance. The current equal fractions yield zero distance and
variance, while the separate overlap audit still detects repeated wording.
The tests also inject skew and inconsistent counts to verify the calculations
and failure path.
Test and challenge results never choose the winner. The original repeated templates
are a strong shortcut: the selected model drops from 16/16 to 10/16 on the
alternate wording. This difference is descriptive, since the wording was authored
after inspecting the original templates. It shows sensitivity to a changed input
distribution but does not isolate a causal effect of one token or estimate general
language performance. The question, dataset, choice of metrics and limitations are encoded
in `protocol.json` and the report.

## Selected technical source and experimental consequence

The [scikit-learn MultinomialNB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
describes the multinomial model for discrete feature counts and additive
smoothing (`alpha`). The new baseline implements count features and add-one
smoothing in the Python standard library. Smoothing keeps probabilities for
observed vocabulary features finite; this implementation ignores tokens that
were never seen in training. The source informs the method; the
repository does not claim anyone personally read or mastered it. The code is a
teaching-sized implementation, not a byte-for-byte replica of scikit-learn.

Accuracy is the fraction of correct family predictions. Macro F1 gives each
family equal weight so a majority baseline cannot hide weak minority classes.
The report gives label fractions and population variance across fixed splits,
as well as population variance of accuracy across held-out entities. These
statistics are descriptive, not uncertainty intervals. A correlation between
tokens and labels cannot show that those tokens cause a task family; template
construction explains the association here.
