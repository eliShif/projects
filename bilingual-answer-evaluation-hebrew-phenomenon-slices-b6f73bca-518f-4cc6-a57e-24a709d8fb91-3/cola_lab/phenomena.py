"""Explicit, bounded Hebrew challenge grammar and corpus hygiene.

The annotations describe constructed examples, not a general linguistic detector.
Reference parsing is deliberately separate from the lexical answerer.
"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


SUITE = "hebrew-phenomena-v1"
CORPUS_PATH = Path(__file__).with_name("resources") / "hebrew_phenomena.json"
CLEANING_POLICY = "nfc-whitespace-v1; NFC and whitespace only; retain letters, marks, punctuation and script"
REFERENCE_POLICY = "hebrew-challenges-v1; explicit form grammar plus source entity/attribute/value checks"
SLICE_POLICY = "hebrew-variants-v1; Hebrew variants only, matched controls, null metrics for empty slices"
PHENOMENA = {
    "inflection": "Possessive inflection: הצבע של (the color of) versus צבעו של (its color, of); masculine fictional object.",
    "spelling": "Full/deficient spelling of description: התיאור versus התאור; the source retains התיאור.",
    "mixed_language": "English content word inside Hebrew syntax: הצבע versus ה-color; Latin entity IDs alone do not define this slice.",
}
ATTRIBUTES = {"inflection": "color", "spelling": "description", "mixed_language": "color"}
# English controls are semantic paraphrases, not Hebrew phenomenon examples.
QUESTION_TEMPLATES = {
    "inflection": {
        "he": {"control": "מה הצבע של {entity}?", "variant": "מה צבעו של {entity}?"},
        "en": {"control": "What is the color of {entity}?", "variant": "Which color does {entity} have?"}},
    "spelling": {
        "he": {"control": "מה התיאור של {entity}?", "variant": "מה התאור של {entity}?"},
        "en": {"control": "What is the description of {entity}?", "variant": "Which description does {entity} have?"}},
    "mixed_language": {
        "he": {"control": "מה הצבע של {entity}?", "variant": "מה ה-color של {entity}?"},
        "en": {"control": "What is the color of {entity}?", "variant": "Which color does {entity} have?"}},
}
DESCRIPTION_FACTS = {
    "en": r"The description of (?P<entity>\w+) is (?P<value>\w+)\.",
    "he": r"התיאור של (?P<entity>\w+) הוא (?P<value>\w+)\."}
ANNOTATION_FIELDS = {"phenomenon", "variant_group", "variant"}


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Raw text must be a nonempty string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def make_dataset():
    """Load the packaged literal examples and retain both raw and cleaned text."""
    dataset = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for collection, field in (("documents", "text"), ("cases", "question")):
        for row in dataset[collection]:
            row["raw_" + field] = row[field]
            row[field] = clean_text(row[field])
    return dataset


def challenge_fact(text, language):
    match = re.fullmatch(DESCRIPTION_FACTS[language], text)
    if match:
        return match["entity"], "description", match["value"]
    if language == "he":
        match = re.fullmatch(r"הצורה של (?P<entity>\w+) היא (?P<value>\w+)\.", text)
        if match:
            return match["entity"], "shape", match["value"]
    return None


def question_target(case):
    """Check the actual wording against the declared phenomenon AND form."""
    phenomenon, variant = case["phenomenon"], case["variant"]
    if (not isinstance(phenomenon, str) or phenomenon not in PHENOMENA
            or not isinstance(variant, str) or variant not in ("control", "variant")):
        raise ValueError("Invalid phenomenon or variant annotation")
    template = QUESTION_TEMPLATES[phenomenon][case["language"]][variant]
    pattern = re.escape(template).replace(re.escape("{entity}"), r"(?P<entity>\w+)")
    match = re.fullmatch(pattern, case["question"])
    if not match:
        raise ValueError("Question does not match declared phenomenon/form grammar")
    entity = match["entity"]
    if case["variant_group"] != f"{entity}/{phenomenon}":
        raise ValueError("Variant group must identify its entity and phenomenon")
    return entity, ATTRIBUTES[phenomenon]


def validate_texts(dataset):
    """Check raw/clean correspondence and duplicates before interpreting labels."""
    for collection, field in (("documents", "text"), ("cases", "question")):
        seen = set()
        for row in dataset[collection]:
            if (not isinstance(row, dict) or row.get("language") not in ("en", "he")
                    or not isinstance(row.get(field), str)):
                raise ValueError("Invalid challenge text fields")
            if clean_text(row.get("raw_" + field)) != row[field]:
                raise ValueError("Cleaned text disagrees with raw text and cleaning policy")
            # Casefold only for detecting duplicates, never for transforming model inputs.
            key = (row["language"], row[field].casefold())
            if key in seen:
                raise ValueError(f"Duplicate normalized {field}")
            seen.add(key)


def validate_structure(dataset):
    """Called after base field validation; isolate complete related groups."""
    groups = defaultdict(list)
    for case in dataset["cases"]:
        if case["phenomenon"] is not None:
            groups[case["variant_group"]].append(case)
    for group, rows in groups.items():
        if len({row["split"] for row in rows}) != 1:
            raise ValueError(f"Variant group leaks across splits: {group}")
        if len(rows) != 4 or {(row["language"], row["variant"]) for row in rows} != {
                (language, form) for language in ("en", "he") for form in ("control", "variant")}:
            raise ValueError(f"Variant group needs both forms and translations: {group}")
        for language in ("en", "he"):
            pair = [row for row in rows if row["language"] == language]
            if (pair[0]["expected_answer"], pair[0]["evidence_ids"]) != (
                    pair[1]["expected_answer"], pair[1]["evidence_ids"]):
                raise ValueError("Variants must share the same answer and source within a language")

