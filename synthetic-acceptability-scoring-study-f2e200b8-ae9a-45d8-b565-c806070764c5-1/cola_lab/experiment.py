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


def score_histogram(values, rows, bins=5):
    """Count fixed test scores in shared, equal-width bins for four labeled slices."""
    if not values or len(values) != len(rows) or bins < 1:
        raise ValueError("Histogram needs matching nonempty scores and positive bins")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Histogram scores must be finite")
    low, high = min(values), max(values)
    width = (high - low) / bins
    result = [{"lower": low + index * width,
               "upper": low + (index + 1) * width,
               "counts": {length: {str(label): 0 for label in (0, 1)}
                          for length in ("short", "long")}}
              for index in range(bins)]
    for value, row in zip(values, rows):
        if row["length"] not in ("short", "long") or row["label"] not in (0, 1):
            raise ValueError("Histogram needs known length and label values")
        index = min(int((value - low) / width), bins - 1) if width else 0
        result[index]["counts"][row["length"]][str(row["label"])] += 1
    return result


def paired_comparison(rows, sum_predictions, mean_predictions):
    """Compare correctness on the same held-out rows, including discordant cases."""
    if not rows or len(rows) != len(sum_predictions) or len(rows) != len(mean_predictions):
        raise ValueError("Paired comparison needs matching nonempty rows and predictions")
    counts = {key: 0 for key in ("both_correct", "both_wrong", "sum_only", "mean_only")}
    by_length = {length: counts.copy() for length in ("short", "long")}
    disagreements = []
    for row, sum_prediction, mean_prediction in zip(rows, sum_predictions, mean_predictions):
        if row.get("label") not in (0, 1) or row.get("length") not in by_length or any(
                prediction not in (0, 1) for prediction in (sum_prediction, mean_prediction)):
            raise ValueError("Paired comparison needs binary labels/predictions and known lengths")
        sum_right = sum_prediction == row["label"]
        mean_right = mean_prediction == row["label"]
        category = ("both_correct" if sum_right and mean_right else
                    "both_wrong" if not sum_right and not mean_right else
                    "sum_only" if sum_right else "mean_only")
        counts[category] += 1
        by_length[row["length"]][category] += 1
        if category in ("sum_only", "mean_only"):
            disagreements.append({"sentence": row["sentence"], "length": row["length"],
                                  "actual": row["label"], "sum": sum_prediction,
                                  "mean": mean_prediction, "winner": category})
    discordant = counts["sum_only"] + counts["mean_only"]
    # Exact two-sided binomial tail conditional on discordance, under equal win chance.
    minority = min(counts["sum_only"], counts["mean_only"])
    probability = (min(1.0, 2 * sum(math.comb(discordant, k) for k in range(minority + 1)) /
                       2 ** discordant) if discordant else 1.0)
    return {"counts": counts, "by_length": by_length,
            "accuracy_delta_mean_minus_sum": (counts["mean_only"] - counts["sum_only"]) / len(rows),
            "discordant_count": discordant, "exact_two_sided_p": probability,
            "disagreements": disagreements}


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
    predictions_by_method = {}
    for method in ("sum", "mean"):
        val_scores = [scores(model, row["sentence"])[method] for row in validation]
        threshold = choose_threshold([row["label"] for row in validation], val_scores)
        test_scores = [scores(model, row["sentence"])[method] for row in test]
        predictions = [int(value >= threshold) for value in test_scores]
        predictions_by_method[method] = predictions
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
                           "score_histogram": score_histogram(test_scores, test),
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
            "paired_comparison": paired_comparison(test, predictions_by_method["sum"],
                                                   predictions_by_method["mean"]),
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
    paired = report["paired_comparison"]
    c = paired["counts"]
    lines += ["", "## Paired held-out outcomes", "",
              f"On the same test rows: both correct {c['both_correct']}, both wrong {c['both_wrong']}, "
              f"sum only correct {c['sum_only']}, mean only correct {c['mean_only']}. "
              f"Mean minus sum accuracy: {paired['accuracy_delta_mean_minus_sum']:.3f}. "
              f"Exact two-sided conditional binomial p: {paired['exact_two_sided_p']:.3f} "
              f"from {paired['discordant_count']} discordant rows.", "",
              "This probability assumes independent discordant rows and equal chances that either "
              "method wins. Shared templates and groups weaken that assumption; it is a descriptive "
              "calculation, not evidence of population superiority.", "",
              "| Length | Both correct | Both wrong | Sum only | Mean only |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for length in ("short", "long"):
        slice_counts = paired["by_length"][length]
        lines.append(f"| {length} | {slice_counts['both_correct']} | {slice_counts['both_wrong']} | "
                     f"{slice_counts['sum_only']} | {slice_counts['mean_only']} |")
    for item in paired["disagreements"]:
        lines.append(f"- `{item['sentence']}`: {item['winner']} "
                     f"(actual {item['actual']}, sum {item['sum']}, mean {item['mean']}).")
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
    lines += ["", "Equal-width score bins show the observed distribution. Each method uses "
              "one range shared by all four slices; ranges differ between methods. "
              "Bins include their lower endpoint and exclude their upper endpoint, "
              "except the final bin, which includes the maximum.", "",
              "| Method | Score bin | Short label 0 | Short label 1 | Long label 0 | Long label 1 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in ("sum", "mean"):
        histogram = report["methods"][method]["score_histogram"]
        for index, bucket in enumerate(histogram):
            interval = f"[{bucket['lower']:.3f}, {bucket['upper']:.3f}{']' if index == len(histogram) - 1 else ')'}"
            counts = bucket["counts"]
            lines.append(f"| {method} | {interval} | {counts['short']['0']} | "
                         f"{counts['short']['1']} | {counts['long']['0']} | "
                         f"{counts['long']['1']} |")
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
