"""Offline supervised question-family experiment over existing synthetic cases."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from .research_cases import build_cases

SPLITS = {"train": ["Cobalt", "Delta", "Ember"],
          "validation": ["Fjord"], "test": ["Garnet", "Helix"]}
FAMILIES = ("lookup", "conversion", "difference", "sum", "ratio",
            "missing_field", "unknown_service", "conflict")
CANDIDATES = ("majority", "unigram", "bigram")

# Authored alternative wording for the eight existing families. This is a
# diagnostic set, not another source for model selection or tuning.
CHALLENGE_TEMPLATES = {
    "lookup": "State the audit log storage duration, in days, for {entity}.",
    "conversion": "Convert the {entity} session expiry interval from minutes into seconds.",
    "difference": "In days, what gap separates the log storage durations of {entity} and {other}?",
    "sum": "Add together the per-day request limits assigned to {entity} and {other}.",
    "ratio": "Divide the monthly charge for {entity} by the monthly charge for {other}.",
    "missing_field": "Give the headcount of engineers assigned to {entity} on-call duty.",
    "unknown_service": "Report the audit log storage duration for {entity}Sandbox, in days.",
    "conflict": "Resolve the log storage duration for {entity}Mirror despite two equally current records.",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def cases_from_generator():
    """Reuse the source project's synthetic questions and hold out entire services."""
    _, cases = build_cases("test")
    assignments = {entity: split for split, entities in SPLITS.items() for entity in entities}
    return [{"id": c["id"], "entity": c["entity"], "question": c["question"],
             "label": c["family"], "split": assignments[c["entity"]]} for c in cases]


def validate_cases(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Cases must be a nonempty JSON array")
    required = {"id", "entity", "question", "label", "split"}
    seen_ids, entity_split, questions = set(), {}, {}
    counts = {s: Counter() for s in SPLITS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("Each case needs exactly id, entity, question, label, split")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in required):
            raise ValueError("Case fields must be nonempty strings")
        if row["id"] in seen_ids:
            raise ValueError("Duplicate case id")
        seen_ids.add(row["id"])
        normalized = " ".join(row["question"].casefold().split())
        if normalized in questions:
            earlier = questions[normalized]
            if earlier["label"] != row["label"]:
                raise ValueError("Conflicting labels for identical question; review annotation")
            if earlier["split"] != row["split"]:
                raise ValueError("Identical question leakage across splits")
        questions[normalized] = row
        if row["split"] not in SPLITS or row["label"] not in FAMILIES:
            raise ValueError("Unknown split or label")
        entity = row["entity"]
        if entity in entity_split and entity_split[entity] != row["split"]:
            raise ValueError("Entity leakage across splits")
        entity_split[entity] = row["split"]
        counts[row["split"]][row["label"]] += 1
    if any(not counts[s] for s in SPLITS):
        raise ValueError("Train, validation and test must be nonempty")
    if set(counts["train"]) != set(FAMILIES):
        raise ValueError("Training data must cover every label")
    return {s: {"cases": sum(counts[s].values()),
                "entities": sorted(e for e, split in entity_split.items() if split == s),
                "label_counts": {f: counts[s][f] for f in FAMILIES},
                "imbalance_ratio": max(counts[s].values()) / min(counts[s].values())}
            for s in SPLITS}


def audit_generator_labels(rows):
    """Check labels and entities against the source generator's ID ledger.

    This is a provenance check, not an independent human annotation review.
    Custom IDs remain usable but are listed as unverifiable against the ledger.
    """
    _, source = build_cases("test")
    ledger = {case["id"]: case for case in source}
    unmatched = []
    matched = 0
    for row in rows:
        original = ledger.get(row["id"])
        if original is None:
            unmatched.append(row["id"])
            continue
        if row["entity"] != original["entity"] or row["label"] != original["family"]:
            raise ValueError(f"Generator label/entity mismatch for {row['id']}")
        matched += 1
    return {"matched_source_ids": matched, "unmatched_ids": sorted(unmatched),
            "scope": "Generator provenance only; no independent annotation audit"}


def challenge_cases(test_rows):
    """Pair each held-out case with a different authored question template."""
    expected = {(entity, label) for entity in SPLITS["test"] for label in FAMILIES}
    pairs = [(r["entity"], r["label"]) for r in test_rows]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Challenge requires one test case per held-out entity and label")
    if any(r["split"] != "test" for r in test_rows):
        raise ValueError("Challenge can only use test cases")
    entities = SPLITS["test"]
    challenge = []
    for row in test_rows:
        entity = row["entity"]
        other = entities[(entities.index(entity) + 1) % len(entities)]
        question = CHALLENGE_TEMPLATES[row["label"]].format(entity=entity, other=other)
        if question.casefold() == row["question"].casefold():
            raise ValueError("Challenge wording duplicates the original question")
        challenge.append({**row, "id": f"challenge-{row['id']}", "question": question})
    return challenge


def features(question, ngram):
    words = re.findall(r"\b\w+\b", question.casefold())
    return words if ngram == 1 else words + [f"{a}__{b}" for a, b in zip(words, words[1:])]


class TextBaseline:
    """Majority label or Laplace-smoothed multinomial Naive Bayes."""
    def __init__(self, name):
        if name not in CANDIDATES:
            raise ValueError("Unknown candidate")
        self.name = name

    def fit(self, rows):
        self.class_counts = Counter(r["label"] for r in rows)
        self.majority = sorted(FAMILIES, key=lambda f: (-self.class_counts[f], f))[0]
        self.word_counts = {f: Counter() for f in FAMILIES}
        self.vocabulary = set()
        self.ngram = 2 if self.name == "bigram" else 1
        for row in rows:
            tokens = features(row["question"], self.ngram)
            self.word_counts[row["label"]].update(tokens)
            self.vocabulary.update(tokens)
        return self

    def class_scores(self, question):
        """Unnormalized log scores; only differences within one model are meaningful."""
        if self.name == "majority":
            raise ValueError("Majority baseline has no class log scores")
        tokens = features(question, self.ngram)
        total = sum(self.class_counts.values())
        size = len(self.vocabulary)
        scores = {}
        for label in FAMILIES:
            count = self.class_counts[label]
            if not count:
                continue
            words = self.word_counts[label]
            denom = sum(words.values()) + size
            scores[label] = math.log(count / total) + sum(
                math.log((words[token] + 1) / denom) for token in tokens if token in self.vocabulary)
        return scores

    def predict(self, question):
        if self.name == "majority":
            return self.majority
        scores = self.class_scores(question)
        return sorted(scores, key=lambda f: (-scores[f], f))[0]


def score_margin_review(model, rows):
    """Describe top-two log-score gaps and observed errors on a fixed case set."""
    if not rows:
        raise ValueError("Score margin review needs nonempty cases")
    if model.name == "majority":
        raise ValueError("Score margin review needs a scored classifier")
    examples = []
    for row in rows:
        ranked = sorted(model.class_scores(row["question"]).items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < 2:
            raise ValueError("Score margin review needs at least two trained labels")
        margin = ranked[0][1] - ranked[1][1]
        examples.append({"id": row["id"], "label": row["label"], "predicted": ranked[0][0],
                         "correct": ranked[0][0] == row["label"], "top_two_log_score_margin": margin})
    def describe(items):
        values = [item["top_two_log_score_margin"] for item in items]
        if not values:
            return {"count": 0, "mean": None, "population_variance": None, "min": None, "max": None}
        mean = sum(values) / len(values)
        return {"count": len(values), "mean": mean,
                "population_variance": sum((value - mean) ** 2 for value in values) / len(values),
                "min": min(values), "max": max(values)}
    return {"all": describe(examples), "correct": describe([item for item in examples if item["correct"]]),
            "wrong": describe([item for item in examples if not item["correct"]]), "examples": examples,
            "scope": "Top minus runner-up unnormalized log score within one fitted model; descriptive, not a calibrated probability or causal explanation"}


def evaluate(model, rows):
    if not rows:
        raise ValueError("Evaluation needs nonempty cases")
    confusion = {f: Counter() for f in FAMILIES}
    by_entity = defaultdict(list)
    failures = []
    for row in rows:
        guess = model.predict(row["question"])
        truth = row["label"]
        confusion[truth][guess] += 1
        by_entity[row["entity"]].append(int(guess == truth))
        if guess != truth:
            failures.append({"id": row["id"], "entity": row["entity"], "expected": truth,
                             "predicted": guess, "question": row["question"]})
    correct = sum(confusion[f][f] for f in FAMILIES)
    per_label = {}
    for f in FAMILIES:
        tp = confusion[f][f]
        fp = sum(confusion[other][f] for other in FAMILIES if other != f)
        fn = sum(confusion[f][other] for other in FAMILIES if other != f)
        support = tp + fn
        per_label[f] = {"support": support, "predicted": tp + fp,
                        "precision": tp / (tp + fp) if tp + fp else 0.0,
                        "recall": tp / support if support else 0.0,
                        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    entity_accuracy = {e: sum(v) / len(v) for e, v in sorted(by_entity.items())}
    entity_mean = sum(entity_accuracy.values()) / len(entity_accuracy)
    return {"accuracy": correct / len(rows), "macro_f1": sum(item["f1"] for item in per_label.values()) / len(FAMILIES),
            "per_label": per_label,
            "labels": list(FAMILIES), "confusion": [[confusion[f][g] for g in FAMILIES] for f in FAMILIES],
            "entity_accuracy": entity_accuracy, "entity_accuracy_mean": entity_mean,
            "entity_accuracy_population_variance": sum((x - entity_mean) ** 2 for x in entity_accuracy.values()) / len(entity_accuracy),
            "failures": failures}


def diagnostic_review(model, training, evaluated, result):
    """Record observable data and feature clues; do not assign causal blame."""
    vocabulary = model.vocabulary
    counts = Counter(row["label"] for row in training)
    failures = []
    by_id = {row["id"]: row for row in evaluated}
    for failure in result["failures"]:
        row = by_id[failure["id"]]
        tokens = features(row["question"], model.ngram)
        unseen = sorted(set(tokens) - vocabulary)
        failures.append({**failure, "training_label_count": counts[row["label"]],
                         "unseen_feature_fraction": len([t for t in tokens if t not in vocabulary]) / len(tokens) if tokens else 0,
                         "unseen_features": unseen,
                         "annotation_status": "synthetic generator label; no independent annotation audit"})
    return {"evaluation_label_counts": dict(sorted(Counter(r["label"] for r in evaluated).items())),
            "training_label_counts": {f: counts[f] for f in FAMILIES},
            "failures": failures,
            "interpretation": "Label counts and unseen features are clues, not proof of whether an error came from annotation, composition or model behavior."}


def paired_wording_review(model, original, alternate):
    """Compare predictions on matched cases; preserve every success and failure."""
    original_by_id = {row["id"]: row for row in original}
    alternate_by_id = {row["id"]: row for row in alternate}
    if len(original_by_id) != len(original) or len(alternate_by_id) != len(alternate):
        raise ValueError("Paired review requires unique case ids")
    if set(alternate_by_id) != {f"challenge-{case_id}" for case_id in original_by_id}:
        raise ValueError("Paired review requires one alternate per original case")
    transitions = Counter()
    by_label = {label: Counter() for label in FAMILIES}
    examples = []
    for case_id, row in sorted(original_by_id.items()):
        changed = alternate_by_id[f"challenge-{case_id}"]
        if (row["entity"], row["label"], row["split"]) != (changed["entity"], changed["label"], changed["split"]):
            raise ValueError("Paired review entity, label and split must match")
        if row["question"].casefold() == changed["question"].casefold():
            raise ValueError("Paired review needs different wording")
        before = model.predict(row["question"])
        after = model.predict(changed["question"])
        transition = ("correct" if before == row["label"] else "wrong") + "_to_" + ("correct" if after == row["label"] else "wrong")
        transitions[transition] += 1
        by_label[row["label"]][transition] += 1
        examples.append({"id": case_id, "entity": row["entity"], "label": row["label"],
                         "original_prediction": before, "alternate_prediction": after,
                         "transition": transition})
    kinds = ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")
    return {"cases": len(examples), "transitions": {kind: transitions[kind] for kind in kinds},
            "by_label": {label: {kind: by_label[label][kind] for kind in kinds} for label in FAMILIES},
            "examples": examples}


def template_overlap(training, evaluated):
    """Audit exact wording reuse after masking known synthetic entity names.

    This detects a specific shortcut in these generated cases, not semantic
    similarity. Entity names are masked as whole words to avoid substring matches.
    """
    if not training or not evaluated:
        raise ValueError("Template audit needs nonempty training and evaluation sets")
    entities = {row["entity"] for row in training + evaluated}
    # The generator also uses EntitySandbox and EntityMirror as fictional names.
    # Retain those suffixes as part of the template rather than masking them.
    pattern = re.compile(r"\b(" + "|".join(re.escape(e) for e in sorted(entities, key=len, reverse=True)) + r")(Sandbox|Mirror)?\b", re.IGNORECASE)

    def normalized(row):
        return " ".join(pattern.sub(lambda match: "<entity>" + (match.group(2) or "").casefold(),
                                    row["question"].casefold()).split())

    known = {normalized(row) for row in training}
    examples = [{"id": row["id"], "label": row["label"], "seen_training_template": normalized(row) in known}
                for row in evaluated]
    counts = Counter(row["label"] for row in evaluated)
    seen = Counter(row["label"] for row in examples if row["seen_training_template"])
    return {"cases": len(examples), "seen": sum(seen.values()),
            "seen_fraction": sum(seen.values()) / len(examples),
            "by_label": {label: {"seen": seen[label], "total": counts[label]} for label in FAMILIES},
            "examples": examples,
            "definition": "Exact normalized question match after replacing known entity names, preserving Sandbox/Mirror suffixes; no semantic matching"}


def grouped_stability(rows, method):
    """Exploratory leave-one-entity-out check with a fixed method, no selection."""
    entities = sorted({r["entity"] for r in rows})
    folds = []
    for entity in entities:
        train = [r for r in rows if r["entity"] != entity]
        held = [r for r in rows if r["entity"] == entity]
        if not train or not held or not set(FAMILIES) <= {r["label"] for r in train}:
            raise ValueError("Each stability fold needs a held-out entity and all training labels")
        score = evaluate(TextBaseline(method).fit(train), held)
        folds.append({"held_out_entity": entity, "train_cases": len(train),
                      "held_out_cases": len(held), "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
    accuracies = [f["accuracy"] for f in folds]
    mean = sum(accuracies) / len(accuracies)
    return {"method": method, "folds": folds, "accuracy_min": min(accuracies),
            "accuracy_max": max(accuracies), "accuracy_mean": mean,
            "accuracy_population_variance": sum((value - mean) ** 2 for value in accuracies) / len(accuracies)}


def resample_stability(training, test, challenge, method, seeds=tuple(range(12))):
    """Refit one fixed method on label-stratified training resamples.

    Evaluation sets and their labels stay frozen. This is a sensitivity check
    on a tiny constructed training set, not a confidence interval.
    """
    if not training or not test or not challenge:
        raise ValueError("Resampling needs nonempty training and evaluation sets")
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Resampling needs distinct integer seeds")
    if any(row["split"] != "train" for row in training) or any(row["split"] != "test" for row in test + challenge):
        raise ValueError("Resampling requires frozen train and test assignments")
    by_label = {label: sorted((row for row in training if row["label"] == label), key=lambda row: row["id"])
                for label in FAMILIES}
    if any(not group for group in by_label.values()):
        raise ValueError("Resampling needs every training label")
    if len({row["id"] for row in test}) != len(test) or len({row["id"] for row in challenge}) != len(challenge):
        raise ValueError("Evaluation case IDs must be unique")
    runs = []
    errors = {row["id"]: 0 for row in test}
    challenge_errors = {row["id"]: 0 for row in challenge}
    for seed in seeds:
        rng = random.Random(seed)
        sample = [rng.choice(by_label[label]) for label in FAMILIES for _ in by_label[label]]
        model = TextBaseline(method).fit(sample)
        held = evaluate(model, test)
        alternate = evaluate(model, challenge)
        for failure in held["failures"]:
            errors[failure["id"]] += 1
        for failure in alternate["failures"]:
            challenge_errors[failure["id"]] += 1
        runs.append({"seed": seed, "sample_ids": [row["id"] for row in sample],
                     "test_accuracy": held["accuracy"], "test_macro_f1": held["macro_f1"],
                     "challenge_accuracy": alternate["accuracy"], "challenge_macro_f1": alternate["macro_f1"]})
    def describe(key):
        values = [run[key] for run in runs]
        mean = sum(values) / len(values)
        return {"mean": mean, "min": min(values), "max": max(values),
                "population_variance": sum((value - mean) ** 2 for value in values) / len(values)}
    return {"method": method, "runs": runs,
            "metrics": {key: describe(key) for key in ("test_accuracy", "test_macro_f1", "challenge_accuracy", "challenge_macro_f1")},
            "test_error_counts": errors, "challenge_error_counts": challenge_errors,
            "scope": "Stratified resampling with replacement of fixed synthetic training cases; frozen evaluation sets; descriptive sensitivity, not independent trials or a confidence interval"}


def label_distribution_review(quality):
    """Compare label proportions, accounting for different split sizes.

    Total variation is half the sum of absolute proportion differences. It is
    descriptive here: these deterministic synthetic splits are not a random
    sample from a population.
    """
    proportions = {}
    for split in SPLITS:
        item = quality[split]
        counts = item["label_counts"]
        total = item["cases"]
        if total <= 0 or any(counts.get(label, 0) < 0 for label in FAMILIES):
            raise ValueError("Label distribution requires nonnegative counts and nonempty splits")
        if sum(counts.get(label, 0) for label in FAMILIES) != total:
            raise ValueError("Label counts must sum to split size")
        proportions[split] = {label: counts[label] / total for label in FAMILIES}
    per_label = {}
    for label in FAMILIES:
        values = [proportions[split][label] for split in SPLITS]
        mean = sum(values) / len(values)
        per_label[label] = {"mean_split_fraction": mean,
                            "population_variance_split_fraction": sum((x - mean) ** 2 for x in values) / len(values)}
    train = proportions["train"]
    distance = {split: sum(abs(train[label] - proportions[split][label]) for label in FAMILIES) / 2
                for split in ("validation", "test")}
    return {"proportions": proportions, "per_label": per_label,
            "total_variation_from_train": distance,
            "scope": "Unweighted mean and population variance over three fixed split fractions; descriptive, not uncertainty or causality"}


def imbalance_stress(train, evaluated, method, minority="lookup"):
    """Deterministically remove all but one training example of a chosen family."""
    if minority not in FAMILIES:
        raise ValueError("Unknown minority label")
    candidates = sorted((r for r in train if r["label"] == minority), key=lambda r: r["id"])
    if len(candidates) < 2:
        raise ValueError("Imbalance stress needs at least two minority examples")
    stressed = [r for r in train if r["label"] != minority] + candidates[:1]
    original = evaluate(TextBaseline(method).fit(train), evaluated)
    changed = evaluate(TextBaseline(method).fit(stressed), evaluated)
    index = FAMILIES.index(minority)
    def recall(score):
        return score["confusion"][index][index] / sum(score["confusion"][index]) if sum(score["confusion"][index]) else None
    return {"method": method, "minority_label": minority,
            "original_training_counts": dict(sorted(Counter(r["label"] for r in train).items())),
            "stressed_training_counts": dict(sorted(Counter(r["label"] for r in stressed).items())),
            "original": {"accuracy": original["accuracy"], "macro_f1": original["macro_f1"], "minority_recall": recall(original)},
            "stressed": {"accuracy": changed["accuracy"], "macro_f1": changed["macro_f1"], "minority_recall": recall(changed)}}


def run(out, cases_path=None):
    out = Path(out)
    rows = json.loads(Path(cases_path).read_text(encoding="utf-8")) if cases_path else cases_from_generator()
    quality = validate_cases(rows)  # Reject bad input before writing results.
    distribution = label_distribution_review(quality)
    label_audit = audit_generator_labels(rows)
    grouped = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    challenge_rows = challenge_cases(grouped["test"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cases.json", rows)
    write_json(out / "challenge_cases.json", challenge_rows)
    protocol = {"question": "Can lexical features classify a synthetic research question's family for unseen service entities?",
                "data": "Existing build_cases('test') generator; synthetic English templates only",
                "split_unit": "service entity", "splits": SPLITS, "label_order": FAMILIES,
                "selection": "highest validation macro F1, then accuracy, then fixed name order; test used after selection",
                "challenge": "After selection, evaluate the same models on one alternate authored wording per held-out entity and label; do not tune on challenge results",
                "diagnostics": "After original validation selection: paired wording transitions, fixed-method leave-one-entity-out folds, 12 seeded stratified training resamples, one-family training downsampling, and feature/label-count review; exploratory, never used to select a model",
                "template_audit": "Exact question-template overlap with training after masking known service names; descriptive only and never used for model selection",
                "candidates": CANDIDATES, "method": "majority versus multinomial Naive Bayes with unigram or unigram+bigram counts and Laplace smoothing",
                "metrics": {"accuracy": "fraction correctly classified", "macro_f1": "arithmetic mean of per-family F1, including zero for absent or unpredicted families",
                            "per_label": "support, predicted count, precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN); undefined denominators reported as zero",
                            "entity_variance": "population variance of accuracy across test entities",
                            "score_margin": "top minus runner-up unnormalized class log score; descriptive mean and population variance on fixed cases",
                            "label_distribution": "split-normalized label fractions, unweighted mean and population variance across fixed splits, and total variation from train"},
                "cases_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "challenge_cases_sha256": hashlib.sha256(json.dumps(challenge_rows, sort_keys=True).encode()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": sys.version.split()[0]}
    write_json(out / "protocol.json", protocol)
    models = {name: TextBaseline(name).fit(grouped["train"]) for name in CANDIDATES}
    validation = {name: evaluate(model, grouped["validation"]) for name, model in models.items()}
    winner = sorted(models, key=lambda n: (-validation[n]["macro_f1"],
                                             -validation[n]["accuracy"], n))[0]
    test = {name: evaluate(model, grouped["test"]) for name, model in models.items()}
    challenge = {name: evaluate(model, challenge_rows) for name, model in models.items()}
    margins = {name: {"validation": score_margin_review(models[name], grouped["validation"]),
                      "test": score_margin_review(models[name], grouped["test"]),
                      "challenge": score_margin_review(models[name], challenge_rows)}
               for name in ("unigram", "bigram")}
    overlap = {"validation": template_overlap(grouped["train"], grouped["validation"]),
               "test": template_overlap(grouped["train"], grouped["test"]),
               "challenge": template_overlap(grouped["train"], challenge_rows)}
    paired = paired_wording_review(models[winner], grouped["test"], challenge_rows)
    stability = grouped_stability(rows, winner)
    resampling = resample_stability(grouped["train"], grouped["test"], challenge_rows, winner)
    imbalance = imbalance_stress(grouped["train"], grouped["test"], winner)
    review = diagnostic_review(models[winner], grouped["train"], challenge_rows, challenge[winner])
    summary = {"quality": quality, "label_distribution": distribution, "score_margins": margins,
               "label_audit": label_audit, "template_overlap": overlap,
               "validation": validation, "selected": winner, "test": test,
               "new_template_challenge": challenge, "paired_wording_review": paired, "grouped_stability": stability,
               "training_resample_stability": resampling,
               "training_imbalance_stress": imbalance, "challenge_error_review": review,
               "limitations": ["The alternate wording was authored after inspecting the original templates; this diagnostic is exploratory, not an independent benchmark.",
                               "One validation and two test entities; variance describes these entities, not a confidence interval.",
                               "Label-fraction statistics use three fixed synthetic splits; equal proportions do not imply equal wording or representative sampling.",
                               "English synthetic questions only; no claim about Hebrew quality or language models.",
                               "Stability folds reuse all entities after the primary selection; they are a sensitivity analysis, not a second independent test.",
                               "Seeded resamples reuse the same 24 constructed training cases and frozen evaluation sets; their spread is not population uncertainty.",
                               "The downsampling check changes training composition and is post-hoc; any score change cannot establish a general effect size.",
                               "Generated labels have no independent human audit; feature and frequency clues cannot prove error cause.",
                               "Feature-label association does not establish causality or independent human mastery."]}
    write_json(out / "summary.json", summary)
    lines = ["# Synthetic question-family experiment", "", protocol["question"], "",
             "## Data quality", "", "| Split | Cases | Entities | Largest/smallest label count |",
             "|---|---:|---:|---:|"]
    for split, q in quality.items():
        lines.append(f"| {split} | {q['cases']} | {len(q['entities'])} | {q['imbalance_ratio']:.2f} |")
    lines += ["", "## Label distribution across splits", "",
              "Fractions account for the different split sizes. Total variation is half the sum of absolute label-fraction differences from training; zero means identical proportions.", "",
              "| Family | Train | Validation | Test | Mean split fraction | Population variance |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in FAMILIES:
        p = distribution["proportions"]
        stat = distribution["per_label"][label]
        lines.append(f"| {label} | {p['train'][label]:.3f} | {p['validation'][label]:.3f} | {p['test'][label]:.3f} | {stat['mean_split_fraction']:.3f} | {stat['population_variance_split_fraction']:.4f} |")
    distance = distribution["total_variation_from_train"]
    lines += ["", f"Total variation from training: validation {distance['validation']:.3f}; test {distance['test']:.3f}.",
              "The mean and variance treat each fixed split equally; they are descriptive and do not estimate population uncertainty. Even identical label proportions cannot rule out wording shift: the separate template audit checks one form of that shift."]
    lines += ["", f"Generator label/entity ledger: {label_audit['matched_source_ids']} matching IDs; {len(label_audit['unmatched_ids'])} custom IDs without a source match. This checks provenance, not independent annotation quality.",
              "", "## Template reuse audit", "",
              "The audit masks known service names, retaining Sandbox/Mirror suffixes, then checks exact normalized wording against training. It detects one constructed shortcut, not semantic similarity.", "",
              "| Evaluation set | Training templates seen | Cases | Fraction |", "|---|---:|---:|---:|"]
    for split in ("validation", "test", "challenge"):
        item = overlap[split]
        lines.append(f"| {split} | {item['seen']} | {item['cases']} | {item['seen_fraction']:.3f} |")
    lines += ["", "A high overlap can make entity-held-out accuracy optimistic; a lower challenge score is consistent with wording sensitivity, without identifying the cause of each error."]
    lines += ["", "## Results", "", "| Model | Validation macro F1 | Test accuracy | Test macro F1 |",
              "|---|---:|---:|---:|"]
    for name in CANDIDATES:
        lines.append(f"| {name} | {validation[name]['macro_f1']:.3f} | {test[name]['accuracy']:.3f} | {test[name]['macro_f1']:.3f} |")
    lines += ["", f"Selected on validation: **{winner}**.", "", "Test errors for selected model:", ""]
    lines += [f"- `{r['id']}`: expected {r['expected']}, predicted {r['predicted']}; {r['question']}"
              for r in test[winner]["failures"]] or ["- None in this small synthetic test."]
    lines += ["", "## Per-family held-out evaluation", "",
              "Support counts true cases; predicted counts assigned cases. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. An undefined denominator is reported as zero. Macro F1 averages all eight family F1 values, including zero for missed families; accuracy weights individual cases.", "",
              "| Model | Family | Support | Predicted | Precision | Recall | F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("majority", winner):
        for label in FAMILIES:
            item = test[name]["per_label"][label]
            lines.append(f"| {name} | {label} | {item['support']} | {item['predicted']} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    lines += ["", "The majority baseline shows why one aggregate accuracy cannot describe which families are missed. These fixed synthetic cases do not estimate population performance."]
    lines += ["", "## Class score margin distribution", "",
              "For each Naive Bayes model, the margin is the largest minus second-largest unnormalized class log score. The table describes fixed cases; larger margins are not calibrated probabilities or proof that a prediction is correct. The majority baseline has no class log scores.", "",
              "| Model | Set | Cases | Mean margin | Population variance | Wrong cases | Wrong-case mean |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("unigram", "bigram"):
        for split in ("validation", "test", "challenge"):
            review_item = margins[name][split]
            all_stats, wrong_stats = review_item["all"], review_item["wrong"]
            wrong_mean = "n/a" if wrong_stats["mean"] is None else f"{wrong_stats['mean']:.3f}"
            lines.append(f"| {name} | {split} | {all_stats['count']} | {all_stats['mean']:.3f} | {all_stats['population_variance']:.3f} | {wrong_stats['count']} | {wrong_mean} |")
    lines += ["", "Per-case margins and correctness flags are in `summary.json`. These scores reflect constructed feature-label associations; this comparison does not identify a causal effect of wording changes. Because question lengths and vocabularies vary, compare margins within a fitted model cautiously."]
    lines += ["", "## New-template diagnostic (same held-out entities and labels)", "",
              "| Model | Original test accuracy | New-template accuracy | Change |",
              "|---|---:|---:|---:|"]
    for name in CANDIDATES:
        delta = challenge[name]["accuracy"] - test[name]["accuracy"]
        lines.append(f"| {name} | {test[name]['accuracy']:.3f} | {challenge[name]['accuracy']:.3f} | {delta:+.3f} |")
    lines += ["", "Paired outcomes for the validation-selected model:", "",
              "| Original → alternate | Cases |", "|---|---:|"]
    for kind, count in paired["transitions"].items():
        lines.append(f"| {kind.replace('_to_', ' → ')} | {count} |")
    lines += ["", "| Family | Original-correct → alternate-wrong |", "|---|---:|"]
    for label in FAMILIES:
        lines.append(f"| {label} | {paired['by_label'][label]['correct_to_wrong']} |")
    lines += ["", "New-template errors for selected model:", ""]
    lines += [f"- `{r['id']}`: expected {r['expected']}, predicted {r['predicted']}; {r['question']}"
              for r in challenge[winner]["failures"]] or ["- None in this small diagnostic set."]
    lines += ["", "## Group and training-composition sensitivity", "",
              "The selected method stays fixed. Leave-one-entity-out folds retrain it on the other five entities; these folds reuse the primary data and are exploratory.", "",
              "| Held-out entity | Accuracy | Macro F1 |", "|---|---:|---:|"]
    for fold in stability["folds"]:
        lines.append(f"| {fold['held_out_entity']} | {fold['accuracy']:.3f} | {fold['macro_f1']:.3f} |")
    lines += ["", f"Accuracy range: {stability['accuracy_min']:.3f}–{stability['accuracy_max']:.3f}; mean: {stability['accuracy_mean']:.3f}; population variance: {stability['accuracy_population_variance']:.4f}.", "",
              "Twelve seeded, label-stratified training resamples refit the validation-selected method while keeping test and alternate wording fixed. Duplicate training cases can occur; these are sensitivity runs, not independent samples or a confidence interval.", "",
              "| Evaluation | Accuracy min | Mean | Max | Population variance |", "|---|---:|---:|---:|---:|"]
    for label, key in (("Original test", "test_accuracy"), ("Alternate wording", "challenge_accuracy")):
        stat = resampling["metrics"][key]
        lines.append(f"| {label} | {stat['min']:.3f} | {stat['mean']:.3f} | {stat['max']:.3f} | {stat['population_variance']:.4f} |")
    lines += ["", "Per-case error counts and sampled training IDs for each seed are in `summary.json`.", "",
              f"Downsampling `{imbalance['minority_label']}` from {imbalance['original_training_counts'][imbalance['minority_label']]} to 1 training case changes held-out macro F1 from {imbalance['original']['macro_f1']:.3f} to {imbalance['stressed']['macro_f1']:.3f}; minority recall changes from {imbalance['original']['minority_recall']:.3f} to {imbalance['stressed']['minority_recall']:.3f}.",
              "This post-hoc stress check does not change model selection or the primary score.", "",
              "## Error review: distinguish observations from possible causes", "",
              "Labels come from the synthetic family generator; there is no independent annotation audit. The held-out challenge label counts and training counts are recorded in `summary.json`. Unseen feature fractions can indicate wording shift, but cannot establish whether a failure is due to label quality, data composition or the classifier.", ""]
    for failure in review["failures"]:
        lines.append(f"- `{failure['id']}`: {failure['expected']} → {failure['predicted']}; unseen feature fraction {failure['unseen_feature_fraction']:.2f}; training examples for true label {failure['training_label_count']}. Annotation unverified independently.")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in summary["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", help="optional JSON case file; validated before evaluation")
    args = parser.parse_args()
    result = run(args.out, args.cases)
    print(json.dumps({"selected": result["selected"], "test_accuracy": result["test"][result["selected"]]["accuracy"],
                      "output": args.out}, sort_keys=True))
