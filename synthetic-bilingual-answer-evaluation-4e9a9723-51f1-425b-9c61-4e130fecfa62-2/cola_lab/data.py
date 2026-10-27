"""Generate and validate an explicitly synthetic, grouped sentence dataset."""

import csv
import random
from collections import Counter
from pathlib import Path

FIELDS = ("group_id", "split", "sentence", "label", "length")
NOUNS = ("pilot", "artist", "chef", "teacher", "farmer", "singer")
VERBS = (("walks", "walk"), ("reads", "read"), ("sings", "sing"), ("writes", "write"))


def generate_rows(seed=17):
    groups = [(noun, pair) for noun in NOUNS for pair in VERBS]
    random.Random(seed).shuffle(groups)
    rows = []
    for index, (noun, (correct, incorrect)) in enumerate(groups):
        split = "train" if index < 12 else "validation" if index < 18 else "test"
        group_id = f"{noun}:{correct}"
        for length, suffix in (("short", ""), ("long", " near the trees")):
            for label, verb in ((1, correct), (0, incorrect)):
                rows.append({"group_id": group_id, "split": split,
                             "sentence": f"the {noun} {verb}{suffix}",
                             "label": label, "length": length})
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Expected CSV columns: {', '.join(FIELDS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset is empty")
    seen = set()
    group_splits = {}
    labels = {split: Counter() for split in ("train", "validation", "test")}
    for row in rows:
        if any(not row[field].strip() for field in FIELDS):
            raise ValueError("Dataset contains a missing value")
        if row["split"] not in labels or row["label"] not in ("0", "1"):
            raise ValueError("Invalid split or label")
        if row["length"] not in ("short", "long"):
            raise ValueError("Invalid length bucket")
        if (len(row["sentence"].split()) <= 3) != (row["length"] == "short"):
            raise ValueError("Length bucket disagrees with sentence")
        if row["sentence"] in seen:
            raise ValueError("Duplicate sentence across dataset")
        seen.add(row["sentence"])
        group = row["group_id"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError("Group leakage across splits")
        group_splits[group] = row["split"]
        labels[row["split"]][row["label"]] += 1
    if any(labels[split]["0"] == 0 or labels[split]["1"] == 0 for split in labels):
        raise ValueError("Each split needs both labels")
    return [{**row, "label": int(row["label"])} for row in rows]
