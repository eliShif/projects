"""Bigram scoring, validation-only calibration, and held-out evaluation."""

import hashlib
import json
import math
import platform
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .data import generate_rows, read_rows, write_rows


class BigramModel:
    def __init__(self, sentences):
        self.transitions = Counter()
        self.contexts = Counter()
        vocabulary = set()
        for sentence in sentences:
            tokens = sentence.split() + ["</s>"]
            vocabulary.update(tokens)
            for before, after in zip(["<s>"] + tokens[:-1], tokens):
                self.transitions[before, after] += 1
                self.contexts[before] += 1
        if not self.contexts:
            raise ValueError("Training sentences are required")
        self.vocabulary = vocabulary

    def token_log_probabilities(self, sentence):
        tokens = [word if word in self.vocabulary else "<unk>"
                  for word in sentence.split()] + ["</s>"]
        size = len(self.vocabulary) + 1  # Reserve one unseen-token type.
        return [math.log((self.transitions[before, after] + 1) /
                         (self.contexts[before] + size))
                for before, after in zip(["<s>"] + tokens[:-1], tokens)]


def scores(model, sentence):
    values = model.token_log_probabilities(sentence)
    return {"sum": sum(values), "mean": sum(values) / len(values)}


def roc_auc(labels, values):
    positive = [score for label, score in zip(labels, values) if label == 1]
    negative = [score for label, score in zip(labels, values) if label == 0]
    if not positive or not negative:
        raise ValueError("ROC AUC requires both classes")
    return sum(1 if p > n else 0.5 if p == n else 0
               for p in positive for n in negative) / (len(positive) * len(negative))


def metrics(labels, predictions, values=None):
    counts = Counter((actual, predicted) for actual, predicted in zip(labels, predictions))
    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
    if not labels or tn + fp == 0 or tp + fn == 0:
        raise ValueError("Metrics require both classes")
    result = {"accuracy": (tp + tn) / len(labels),
              "balanced_accuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
              "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    if values is not None:
        result["roc_auc"] = roc_auc(labels, values)
    return result


def choose_threshold(labels, values):
    """Maximize validation balanced accuracy; deterministic conservative tie break."""
    unique = sorted(set(values))
    candidates = [unique[0] - 1] + [(a + b) / 2 for a, b in zip(unique, unique[1:])] + [unique[-1] + 1]
    return max(candidates, key=lambda threshold: (metrics(labels, [int(value >= threshold)
                                          for value in values])["balanced_accuracy"], threshold))


def describe(values):
    """Describe every value in a fixed, observed set (population variance)."""
    if not values:
        raise ValueError("Cannot describe an empty set")
    return {"count": len(values), "mean": statistics.mean(values),
            "population_variance": statistics.pvariance(values),
            "minimum": min(values), "maximum": max(values)}


def training_sensitivity(train, validation, test, draws=24, seed=2026):
    """Refit on seeded group resamples; keep validation/test cases unchanged."""
    if draws < 1:
        raise ValueError("At least one resample is required")
    groups = sorted({row["group_id"] for row in train})
    if not groups:
        raise ValueError("Training groups are required")
    group_rows = {group: [row for row in train if row["group_id"] == group]
                  for group in groups}
    rng = random.Random(seed)
    records = {method: [] for method in ("sum", "mean")}
    error_counts = {method: Counter() for method in records}
    val_labels = [row["label"] for row in validation]
    test_labels = [row["label"] for row in test]
    for _ in range(draws):
        sampled = [rng.choice(groups) for _ in groups]
        positive_sentences = [row["sentence"] for group in sampled
                              for row in group_rows[group] if row["label"] == 1]
        model = BigramModel(positive_sentences)
        for method in records:
            threshold = choose_threshold(val_labels,
                                         [scores(model, row["sentence"])[method]
                                          for row in validation])
            predictions = [int(scores(model, row["sentence"])[method] >= threshold)
                           for row in test]
            result = metrics(test_labels, predictions)
            records[method].append({"sampled_groups": sampled,
                                    "validation_threshold": threshold,
                                    "accuracy": result["accuracy"],
                                    "balanced_accuracy": result["balanced_accuracy"]})
            for row, prediction in zip(test, predictions):
                if prediction != row["label"]:
                    error_counts[method][row["sentence"]] += 1
    return {"protocol": {"draws": draws, "seed": seed,
                         "sampling_unit": "training group, with replacement",
                         "validation_and_test": "fixed; threshold selected on validation each draw"},
            "methods": {method: {"accuracy": describe([r["accuracy"] for r in records[method]]),
                                 "balanced_accuracy": describe([r["balanced_accuracy"]
                                                                 for r in records[method]]),
                                 "error_frequency": [{"sentence": row["sentence"],
                                                      "error_draws": error_counts[method][row["sentence"]]}
                                                     for row in test if error_counts[method][row["sentence"]]],
                                 "draws": records[method]}
                        for method in records}}


def evaluate(rows, dataset_sha256):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    data_quality = {split: {"label_counts": {str(label): sum(row["label"] == label
                                                           for row in by_split[split])
                                                  for label in (0, 1)},
                            "length_counts": {length: sum(row["length"] == length
                                                          for row in by_split[split])
                                              for length in ("short", "long")},
                            "group_count": len({row["group_id"] for row in by_split[split]})}
                    for split in ("train", "validation", "test")}
    model = BigramModel(row["sentence"] for row in by_split["train"] if row["label"] == 1)
    validation = by_split["validation"]
    test = by_split["test"]
    methods = {}
    for method in ("sum", "mean"):
        val_scores = [scores(model, row["sentence"])[method] for row in validation]
        threshold = choose_threshold([row["label"] for row in validation], val_scores)
        test_scores = [scores(model, row["sentence"])[method] for row in test]
        predictions = [int(value >= threshold) for value in test_scores]
        labels = [row["label"] for row in test]
        by_length = {}
        for length in ("short", "long"):
            indices = [i for i, row in enumerate(test) if row["length"] == length]
            by_length[length] = metrics([labels[i] for i in indices],
                                        [predictions[i] for i in indices],
                                        [test_scores[i] for i in indices])
            by_length[length]["mean_score"] = sum(test_scores[i] for i in indices) / len(indices)
            by_length[length]["mean_score_by_label"] = {
                str(label): sum(test_scores[i] for i in indices if labels[i] == label) /
                            sum(labels[i] == label for i in indices)
                for label in (0, 1)}
        failures = [{"sentence": row["sentence"], "actual": row["label"],
                     "predicted": predictions[i], "score": test_scores[i],
                     "length": row["length"]}
                    for i, row in enumerate(test) if row["label"] != predictions[i]]
        methods[method] = {"validation_threshold": threshold,
                           "test": metrics(labels, predictions, test_scores),
                           "by_length": by_length, "errors": failures,
                           "score_distribution": {
                               length: {str(label): describe([test_scores[i] for i, row in enumerate(test)
                                                             if row["length"] == length and row["label"] == label])
                                        for label in (0, 1)}
                               for length in ("short", "long")}}
    test_labels = [row["label"] for row in test]
    majority_label = int(sum(row["label"] for row in by_split["train"]) >=
                         len(by_split["train"]) / 2)
    baseline = metrics(test_labels, [majority_label] * len(test))
    return {"protocol": {"dataset": "synthetic subject-verb agreement, not CoLA",
                         "seed": 17, "dataset_sha256": dataset_sha256,
                         "python": platform.python_version(),
                         "training": "Laplace-smoothed bigram model on positive train sentences",
                         "selection": "threshold chosen on validation balanced accuracy only",
                         "test_use": "held out until thresholds fixed",
                         "split_counts": {split: len(by_split[split]) for split in
                                          ("train", "validation", "test")}},
            "data_quality": data_quality,
            "baseline": {"method": "train majority class", "label": majority_label,
                         "test": baseline}, "methods": methods,
            "training_sensitivity": training_sensitivity(by_split["train"], validation, test),
            "limitations": ["Synthetic templated sentences do not represent CoLA or natural language.",
                            "A bigram model is not GPT-2 and this does not verify earlier project claims.",
                            "One deterministic split and a small test set limit stability claims.",
                            "The grammatical label follows the generation rule, not human annotation."]}


def render_report(report):
    """Render a compact interpretation from computed values, never stored scores."""
    quality = report["data_quality"]
    lines = ["# Synthetic acceptability experiment", "",
             "This report is generated from a fixed synthetic CSV. It is not a CoLA or GPT-2 result.", "",
             "## Data and protocol", "",
             f"CSV SHA-256: `{report['protocol']['dataset_sha256']}`. "
             "The model trains on positive training sentences; thresholds use validation labels; "
             "test labels are used only for final evaluation.", "",
             "| Split | Rows | Label 0 | Label 1 | Groups |", "| --- | ---: | ---: | ---: | ---: |"]
    for split in ("train", "validation", "test"):
        q = quality[split]
        lines.append(f"| {split} | {report['protocol']['split_counts'][split]} | "
                     f"{q['label_counts']['0']} | {q['label_counts']['1']} | {q['group_count']} |")
    lines += ["", "## Held-out comparison", "",
              "| Method | Accuracy | Balanced accuracy | False positives | False negatives | AUC |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    baseline = report["baseline"]["test"]
    lines.append(f"| Train-majority | {baseline['accuracy']:.3f} | "
                 f"{baseline['balanced_accuracy']:.3f} | {baseline['confusion']['fp']} | "
                 f"{baseline['confusion']['fn']} | no score |")
    for method in ("sum", "mean"):
        result = report["methods"][method]["test"]
        lines.append(f"| {method} | {result['accuracy']:.3f} | "
                     f"{result['balanced_accuracy']:.3f} | {result['confusion']['fp']} | "
                     f"{result['confusion']['fn']} | AUC {result['roc_auc']:.3f} |")
    lines += ["", "## Score distributions and failures", "",
              "Population variance describes all scores in each fixed test slice; "
              "it is not uncertainty about future data.", "",
              "| Method | Length | Label | Count | Mean score | Population variance |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in ("sum", "mean"):
        for length in ("short", "long"):
            for label in ("0", "1"):
                d = report["methods"][method]["score_distribution"][length][label]
                lines.append(f"| {method} | {length} | {label} | {d['count']} | "
                             f"{d['mean']:.3f} | {d['population_variance']:.3f} |")
    for method in ("sum", "mean"):
        errors = report["methods"][method]["errors"]
        lines += ["", f"### {method} errors ({len(errors)})", ""]
        if errors:
            for error in errors:
                lines.append(f"- `{error['sentence']}`: actual {error['actual']}, "
                             f"predicted {error['predicted']}, {error['length']}, "
                             f"score {error['score']:.3f}.")
        else:
            lines.append("No errors on these fixed test cases.")
    sensitivity = report["training_sensitivity"]
    lines += ["", "## Training-composition sensitivity", "",
              f"{sensitivity['protocol']['draws']} seeded refits sample training groups with replacement. "
              "Validation and test cases stay fixed; each refit chooses its threshold on validation.", "",
              "| Method | Mean accuracy | Population variance | Minimum | Maximum |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for method in ("sum", "mean"):
        d = sensitivity["methods"][method]["accuracy"]
        lines.append(f"| {method} | {d['mean']:.3f} | {d['population_variance']:.4f} | "
                     f"{d['minimum']:.3f} | {d['maximum']:.3f} |")
    lines += ["", "Most frequent errors across refits:", ""]
    for method in ("sum", "mean"):
        frequent = sorted(sensitivity["methods"][method]["error_frequency"],
                          key=lambda item: (-item["error_draws"], item["sentence"]))[:3]
        for item in frequent:
            lines.append(f"- {method}: `{item['sentence']}` failed in "
                         f"{item['error_draws']}/{sensitivity['protocol']['draws']} refits.")
    lines += ["", "## Interpretation and source note", "",
              "The Python `statistics.pvariance` documentation defines variance over the entire "
              "observed population. This report therefore uses it for each complete fixed slice "
              "and for the finite set of resample outcomes; the values are descriptive, not confidence intervals. "
              "Source: https://docs.python.org/3/library/statistics.html#statistics.pvariance", "",
              "A relationship between sentence length and score or error is observational here. "
              "Length also changes token context under the fixed templates; this comparison does "
              "not identify a causal effect of length. Resampling these training groups checks "
              "sensitivity to this small generated training set, not performance on new language.", "",
              "This note is generated by code for a future reader to review; it does not claim "
              "personal reading, authorship, or mastery.", ""]
    return "\n".join(lines)


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "sentences.csv"
    write_rows(data_path, generate_rows())
    rows = read_rows(data_path)  # The experiment consumes the saved file.
    report = evaluate(rows, hashlib.sha256(data_path.read_bytes()).hexdigest())
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")
    return report_path
