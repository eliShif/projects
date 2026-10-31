"""Literal linguistic examples plus corrupted labels, text and partition fixtures."""

import copy
import tempfile
import unittest
from pathlib import Path

from cola_lab import phenomena
from cola_lab.answer_eval import (_source_fact, dataset_named, evaluate, method_inputs,
                                 run, validate_dataset)
from cola_lab.local_gguf import generation_requests


def find_case(dataset, entity="Fjord", language="he", form="variant"):
    return next(case for case in dataset["cases"] if case["entity"] == entity
                and case["language"] == language and case["variant"] == form)


class PhenomenaDataTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset_named("hebrew-phenomena")

    def test_literal_queries_resolve_to_separately_specified_sources_and_values(self):
        validate_dataset(self.data)
        # Expectations are literals, not produced by the parser or answerer.
        for entity, query, attribute, answer, source in (
                ("Ember", "מה צבעו של Ember?", "color", "כחול", "הצבע של Ember הוא כחול."),
                ("Fjord", "מה התאור של Fjord?", "description", "מרובע", "התיאור של Fjord הוא מרובע."),
                ("Grove", "מה ה-color של Grove?", "color", "אדום", "הצבע של Grove הוא אדום.")):
            with self.subTest(entity=entity):
                case = find_case(self.data, entity)
                self.assertEqual(case["question"], query)
                self.assertEqual(phenomena.question_target(case), (entity, attribute))
                self.assertEqual(case["expected_answer"], answer)
                doc = next(doc for doc in self.data["documents"] if doc["id"] == case["evidence_ids"][0])
                self.assertEqual(doc["text"], source)
                self.assertEqual(_source_fact(source, "he", challenge=True), (entity, attribute, answer))
        self.assertEqual(_source_fact("הצורה של Ember היא עגולה.", "he", challenge=True),
                         ("Ember", "shape", "עגולה"))

    def test_cleaning_preserves_inflection_spelling_script_punctuation_and_marks(self):
        for raw, expected in (("  מה\tהתאור של Fjord?\n", "מה התאור של Fjord?"),
                              (" מה צבעו של Ember? ", "מה צבעו של Ember?"),
                              (" מה ה-color של Grove? ", "מה ה-color של Grove?"),
                              ("  שָׁלוֹם  ", "שָׁלוֹם"), ("cafe\u0301", "café")):
            self.assertEqual(phenomena.clean_text(raw), expected)
            self.assertEqual(phenomena.clean_text(expected), expected)
        self.assertNotEqual(phenomena.clean_text("התאור"), phenomena.clean_text("התיאור"))
        self.assertNotEqual(find_case(self.data)["raw_question"], find_case(self.data)["question"])
        for bad in ("", " \t", None, []):
            with self.assertRaises(ValueError):
                phenomena.clean_text(bad)

    def test_raw_and_cleaned_text_cannot_disagree(self):
        find_case(self.data)["raw_question"] = "מה התיאור של Fjord?"
        with self.assertRaisesRegex(ValueError, "Cleaned text disagrees"):
            validate_dataset(self.data)

    def test_variants_and_translations_stay_in_one_split_with_same_language_sources(self):
        groups = {}
        for case in self.data["cases"]:
            if case["variant_group"]:
                groups.setdefault(case["variant_group"], []).append(case)
        self.assertEqual(len(groups), 8)
        self.assertEqual({row["entity"] for row in self.data["cases"] if row["split"] == "test"},
                         {"Ember", "Fjord", "Grove", "Harbor"})
        for group in groups.values():
            self.assertEqual(len(group), 4)
            self.assertEqual(len({row["split"] for row in group}), 1)
            for language in ("en", "he"):
                forms = [row for row in group if row["language"] == language]
                self.assertEqual(forms[0]["evidence_ids"], forms[1]["evidence_ids"])
                self.assertEqual(forms[0]["expected_answer"], forms[1]["expected_answer"])

    def test_moving_only_a_variant_across_splits_is_rejected(self):
        case = find_case(self.data)
        case["split"] = "validation"
        case["id"] = case["id"].replace("test-", "validation-")
        with self.assertRaisesRegex(ValueError, "Variant group leaks"):
            validate_dataset(self.data)

    def test_forged_group_and_missing_variant_are_rejected(self):
        bad = copy.deepcopy(self.data)
        find_case(bad)["variant_group"] = "Birch/spelling"
        with self.assertRaisesRegex(ValueError, "Variant group"):
            validate_dataset(bad)
        self.data["cases"] = [case for case in self.data["cases"]
                              if not (case["entity"] == "Fjord" and case["variant"] == "variant")]
        with self.assertRaisesRegex(ValueError, "both forms and translations"):
            validate_dataset(self.data)

    def test_duplicate_questions_and_documents_rejected_after_normalization(self):
        for collection, field in (("cases", "question"), ("documents", "text")):
            with self.subTest(collection=collection):
                data = copy.deepcopy(self.data)
                row = copy.deepcopy(data[collection][0])
                row["id"] += "-copy"
                row["raw_" + field] = "  " + row[field].replace(" ", "\t") + "\n"
                data[collection].append(row)
                with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                    validate_dataset(data)

    def test_wrong_value_wrong_attribute_and_false_abstention_fail_in_both_languages(self):
        for language in ("en", "he"):
            for mutation in ("wrong_value", "wrong_attribute", "false_absence"):
                with self.subTest(language=language, mutation=mutation):
                    data = copy.deepcopy(self.data)
                    case = find_case(data, language=language)
                    if mutation == "wrong_value":
                        case["expected_answer"] = "description" if language == "en" else "התיאור"
                    elif mutation == "wrong_attribute":
                        case["evidence_ids"] = [f"test-{language}-Fjord-shape"]
                    else:
                        case.update(expected_answer=None, evidence_ids=[])
                    with self.assertRaises(ValueError):
                        validate_dataset(data)

    def test_changed_entity_or_annotation_is_not_accepted_as_a_new_template(self):
        for edits in ({"question": "מה התאור של Ember?", "raw_question": "מה התאור של Ember?"},
                      {"phenomenon": "inflection"}, {"variant": "control"},
                      {"phenomenon": []}, {"variant_group": []}):
            data = copy.deepcopy(self.data)
            find_case(data).update(edits)
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                validate_dataset(data)

    def test_unknown_templates_and_schema_are_rejected(self):
        for edits in ({"suite": "future-version"}, {"extra": "ignored?"}):
            data = copy.deepcopy(self.data)
            data.update(edits)
            with self.assertRaises(ValueError):
                validate_dataset(data)
        case = find_case(self.data)
        case.update(question="ספר לי על Fjord", raw_question="ספר לי על Fjord")
        with self.assertRaisesRegex(ValueError, "grammar"):
            validate_dataset(self.data)

    def test_invalid_labels_fail_before_requests_or_report_publication(self):
        for language in ("en", "he"):
            find_case(self.data, language=language).update(expected_answer=None, evidence_ids=[])
        with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
            generation_requests(self.data)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run(directory, dataset=self.data)
            self.assertFalse((Path(directory) / "answer_report.json").exists())
            self.assertFalse((Path(directory) / "answer_dataset.json").exists())

    def test_shared_inputs_are_label_free_even_for_annotated_variants(self):
        for case in self.data["cases"]:
            inputs = method_inputs(case, self.data["documents"])
            self.assertEqual(inputs["local_model"], inputs["lexical_retrieval"])
            self.assertEqual(set(inputs["local_model"]), {"question", "documents"})
            self.assertTrue(all(set(doc) == {"id", "text"} for doc in inputs["local_model"]["documents"]))
        original = method_inputs(find_case(self.data), self.data["documents"])
        find_case(self.data)["expected_answer"] = "not exposed"
        self.assertEqual(method_inputs(find_case(self.data), self.data["documents"]), original)
        with self.assertRaises(ValueError):
            generation_requests(self.data)

    def test_threshold_is_not_selected_from_test_reference_values(self):
        original = evaluate(self.data, "fixture")
        # Valid but different held-out source values; validation is untouched.
        for doc in self.data["documents"]:
            if doc["split"] == "test":
                doc["text"] = doc["text"].replace(doc["answer"] + ".", "novelvalue.")
                doc["raw_text"] = doc["text"]
                doc["answer"] = "novelvalue"
        for case in self.data["cases"]:
            if case["split"] == "test" and case["expected_answer"] is not None:
                case["expected_answer"] = "novelvalue"
        changed = evaluate(self.data, "changed-fixture")
        self.assertEqual(original["protocol"]["threshold"], changed["protocol"]["threshold"])


if __name__ == "__main__":
    unittest.main()
