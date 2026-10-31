"""Reference expectations are hand-written, never obtained from the generator/parser."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from cola_lab.answer_eval import make_dataset, run, validate_dataset
from cola_lab.local_gguf import generation_requests


class ReferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = json.loads((Path(__file__).parent / "fixtures/reference_cases.json")
                                  .read_text(encoding="utf-8"))

    def case(self, language, attribute="color"):
        return next(row for row in self.dataset["cases"]
                    if row["id"] == f"test-{language}-Ember-{attribute}-q")

    def document(self, language, attribute="color"):
        return next(row for row in self.dataset["documents"]
                    if row["id"] == f"test-{language}-Ember-{attribute}")

    def test_accepts_literal_references_and_existing_generated_data(self):
        validate_dataset(self.dataset)
        validate_dataset(make_dataset())
        # Explicit expectations also guard the fixture's intended meaning.
        self.assertEqual(self.case("en")["expected_answer"], "blue")
        self.assertEqual(self.case("he")["expected_answer"], "כחול")
        self.assertIsNone(self.case("en", "weight")["expected_answer"])
        self.assertIsNone(self.case("he", "weight")["expected_answer"])

    def test_rejects_word_present_in_source_but_not_its_value(self):
        for language, word in (("en", "color"), ("he", "הצבע")):
            with self.subTest(language=language):
                self.setUp()
                self.document(language)["answer"] = word
                self.case(language)["expected_answer"] = word
                with self.assertRaisesRegex(ValueError, "disagrees with source text"):
                    validate_dataset(self.dataset)

    def test_rejects_answer_to_other_attribute_even_with_consistent_evidence(self):
        for language, wrong in (("en", "round"), ("he", "עגול")):
            with self.subTest(language=language):
                self.setUp()
                self.case(language).update(expected_answer=wrong,
                                           evidence_ids=[f"test-{language}-Ember-shape"])
                with self.assertRaisesRegex(ValueError, "question's entity and attribute"):
                    validate_dataset(self.dataset)

    def test_rejects_answer_of_other_entity_even_when_labels_and_source_agree(self):
        # Blue/כחול is Ember's color, but these questions ask for Fjord's green/ירוק.
        for language, question in (("en", "What is the color of Fjord?"),
                                   ("he", "מה הצבע של Fjord?")):
            with self.subTest(language=language):
                self.setUp()
                self.case(language)["question"] = question
                with self.assertRaisesRegex(ValueError, "Question entity"):
                    validate_dataset(self.dataset)

    def test_rejects_false_abstention_even_if_both_translations_are_changed(self):
        for languages in (("en",), ("he",), ("en", "he")):
            with self.subTest(languages=languages):
                self.setUp()
                for language in languages:
                    self.case(language).update(expected_answer=None, evidence_ids=[])
                with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
                    validate_dataset(self.dataset)

    def test_rejects_answer_for_absent_attribute(self):
        for language, wrong in (("en", "round"), ("he", "עגול")):
            with self.subTest(language=language):
                self.setUp()
                self.case(language, "weight").update(
                    expected_answer=wrong, evidence_ids=[f"test-{language}-Ember-shape"])
                with self.assertRaisesRegex(ValueError, "No source fact"):
                    validate_dataset(self.dataset)

    def test_abstention_uses_all_source_facts_not_a_hardcoded_missing_attribute(self):
        for language, answer, text in (("en", "heavy", "The weight of Ember is heavy."),
                                       ("he", "כבד", "המשקל של Ember הוא כבד.")):
            self.dataset["documents"].append({"id": f"test-{language}-Ember-weight",
                "split": "test", "language": language, "entity": "Ember", "text": text, "answer": answer})
        with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
            validate_dataset(self.dataset)
        for language, answer in (("en", "heavy"), ("he", "כבד")):
            self.case(language, "weight").update(
                expected_answer=answer, evidence_ids=[f"test-{language}-Ember-weight"])
        validate_dataset(self.dataset)

    def test_case_id_intent_cannot_override_question_attribute(self):
        for language, question in (("en", "What is the shape of Ember?"),
                                   ("he", "מה הצורה של Ember?")):
            with self.subTest(language=language):
                self.setUp()
                self.case(language)["question"] = question
                with self.assertRaisesRegex(ValueError, "Case ID"):
                    validate_dataset(self.dataset)

    def test_source_entity_cannot_be_forged_in_metadata(self):
        for language, text in (("en", "The color of Fjord is blue."),
                               ("he", "הצבע של Fjord הוא כחול.")):
            with self.subTest(language=language):
                self.setUp()
                self.document(language)["text"] = text
                with self.assertRaisesRegex(ValueError, "Source entity"):
                    validate_dataset(self.dataset)

    def test_unsupported_sources_or_questions_cannot_imply_absence(self):
        for field, row, text in (("text", "documents", "Ember is reportedly blue."),
                                  ("question", "cases", "Tell me anything about Ember.")):
            with self.subTest(field=field):
                broken = copy.deepcopy(self.dataset)
                broken[row][0][field] = text
                with self.assertRaisesRegex(ValueError, "Unsupported synthetic"):
                    validate_dataset(broken)

    def test_rejects_duplicate_or_conflicting_source_facts(self):
        for answer in ("blue", "red"):
            with self.subTest(answer=answer):
                self.setUp()
                self.dataset["documents"].append({**self.document("en"), "id": "another-fact",
                    "answer": answer, "text": f"The color of Ember is {answer}."})
                with self.assertRaisesRegex(ValueError, "Duplicate or conflicting"):
                    validate_dataset(self.dataset)

    def test_rejects_duplicate_or_malformed_evidence(self):
        for evidence in (["test-en-Ember-color"] * 2, [{}], "test-en-Ember-color"):
            with self.subTest(evidence=evidence):
                self.setUp()
                self.case("en")["evidence_ids"] = evidence
                with self.assertRaises(ValueError):
                    validate_dataset(self.dataset)

    def test_invalid_references_fail_before_generation_and_report_publication(self):
        for language in ("en", "he"):
            self.case(language).update(expected_answer=None, evidence_ids=[])
        with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
            generation_requests(self.dataset)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unanswerable label hides"):
                run(directory, dataset=self.dataset)
            self.assertFalse((Path(directory) / "answer_report.json").exists())


if __name__ == "__main__":
    unittest.main()
