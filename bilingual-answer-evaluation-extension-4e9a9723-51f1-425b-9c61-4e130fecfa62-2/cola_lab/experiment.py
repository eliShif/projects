"""Bigram scoring, validation-only calibration, and held-out evaluation."""

import hashlib
import json
import math
import platform
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


def evaluate(rows, dataset_sha256):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
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
                           "by_length": by_length, "errors": failures}
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
            "baseline": {"method": "train majority class", "label": majority_label,
                         "test": baseline}, "methods": methods,
            "limitations": ["Synthetic templated sentences do not represent CoLA or natural language.",
                            "A bigram model is not GPT-2 and this does not verify earlier project claims.",
                            "One deterministic split and a small test set limit stability claims.",
                            "The grammatical label follows the generation rule, not human annotation."]}


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "sentences.csv"
    write_rows(data_path, generate_rows())
    rows = read_rows(data_path)  # The experiment consumes the saved file.
    report = evaluate(rows, hashlib.sha256(data_path.read_bytes()).hexdigest())
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path
