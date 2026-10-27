"""Offline supervised question-family experiment over existing synthetic cases."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from .research_cases import build_cases

SPLITS = {"train": ["Cobalt", "Delta", "Ember"],
          "validation": ["Fjord"], "test": ["Garnet", "Helix"]}
FAMILIES = ("lookup", "conversion", "difference", "sum", "ratio",
            "missing_field", "unknown_service", "conflict")
CANDIDATES = ("majority", "unigram", "bigram")


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
    seen_ids, entity_split = set(), {}
    counts = {s: Counter() for s in SPLITS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("Each case needs exactly id, entity, question, label, split")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in required):
            raise ValueError("Case fields must be nonempty strings")
        if row["id"] in seen_ids:
            raise ValueError("Duplicate case id")
        seen_ids.add(row["id"])
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

    def predict(self, question):
        if self.name == "majority":
            return self.majority
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
        return sorted(scores, key=lambda f: (-scores[f], f))[0]


def evaluate(model, rows):
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
    f1 = []
    for f in FAMILIES:
        tp = confusion[f][f]
        fp = sum(confusion[other][f] for other in FAMILIES if other != f)
        fn = sum(confusion[f][other] for other in FAMILIES if other != f)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    entity_accuracy = {e: sum(v) / len(v) for e, v in sorted(by_entity.items())}
    entity_mean = sum(entity_accuracy.values()) / len(entity_accuracy)
    return {"accuracy": correct / len(rows), "macro_f1": sum(f1) / len(f1),
            "labels": list(FAMILIES), "confusion": [[confusion[f][g] for g in FAMILIES] for f in FAMILIES],
            "entity_accuracy": entity_accuracy, "entity_accuracy_mean": entity_mean,
            "entity_accuracy_population_variance": sum((x - entity_mean) ** 2 for x in entity_accuracy.values()) / len(entity_accuracy),
            "failures": failures}


def run(out, cases_path=None):
    out = Path(out)
    rows = json.loads(Path(cases_path).read_text(encoding="utf-8")) if cases_path else cases_from_generator()
    quality = validate_cases(rows)  # Reject bad input before writing results.
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cases.json", rows)
    protocol = {"question": "Can lexical features classify a synthetic research question's family for unseen service entities?",
                "data": "Existing build_cases('test') generator; synthetic English templates only",
                "split_unit": "service entity", "splits": SPLITS, "label_order": FAMILIES,
                "selection": "highest validation macro F1, then accuracy, then fixed name order; test used after selection",
                "candidates": CANDIDATES, "method": "majority versus multinomial Naive Bayes with unigram or unigram+bigram counts and Laplace smoothing",
                "metrics": {"accuracy": "fraction correctly classified", "macro_f1": "equal weight per family",
                            "entity_variance": "population variance of accuracy across test entities"},
                "cases_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": sys.version.split()[0]}
    write_json(out / "protocol.json", protocol)
    grouped = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    models = {name: TextBaseline(name).fit(grouped["train"]) for name in CANDIDATES}
    validation = {name: evaluate(model, grouped["validation"]) for name, model in models.items()}
    winner = sorted(models, key=lambda n: (-validation[n]["macro_f1"],
                                             -validation[n]["accuracy"], n))[0]
    test = {name: evaluate(model, grouped["test"]) for name, model in models.items()}
    summary = {"quality": quality, "validation": validation, "selected": winner, "test": test,
               "limitations": ["Authored templates repeat across entities; holdout does not establish new-template generalization.",
                               "One validation and two test entities; variance describes these entities, not a confidence interval.",
                               "English synthetic questions only; no claim about Hebrew quality or language models.",
                               "Feature-label association does not establish causality or independent human mastery."]}
    write_json(out / "summary.json", summary)
    lines = ["# Synthetic question-family experiment", "", protocol["question"], "",
             "## Data quality", "", "| Split | Cases | Entities | Largest/smallest label count |",
             "|---|---:|---:|---:|"]
    for split, q in quality.items():
        lines.append(f"| {split} | {q['cases']} | {len(q['entities'])} | {q['imbalance_ratio']:.2f} |")
    lines += ["", "## Results", "", "| Model | Validation macro F1 | Test accuracy | Test macro F1 |",
              "|---|---:|---:|---:|"]
    for name in CANDIDATES:
        lines.append(f"| {name} | {validation[name]['macro_f1']:.3f} | {test[name]['accuracy']:.3f} | {test[name]['macro_f1']:.3f} |")
    lines += ["", f"Selected on validation: **{winner}**.", "", "Test errors for selected model:", ""]
    lines += [f"- `{r['id']}`: expected {r['expected']}, predicted {r['predicted']}; {r['question']}"
              for r in test[winner]["failures"]] or ["- None in this small synthetic test."]
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
