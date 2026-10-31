"""Model-free correctness tests. Ranking fixtures are not model-quality results."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from portfolio.common import sha256_json, write_json
from portfolio.retrieval_experiment import DEFAULT_DATA, ROOT, audit, prepare, run, smoke
from portfolio.retrieval_study import (
    CANDIDATES, compare, evaluate_queries, load_study, paired_differences,
    score_ranking, select_on_validation, summarize, validate_study,
)


class RankingFixture:
    """Hand-constructed evidence for evaluator tests; never a CLI backend."""
    def __init__(self, study, *, correct=True, events=None, label="fixture"):
        self.docs = {d["id"]: d["text"] for d in study.documents}
        self.queries = {q["text"]: q for q in study.queries}
        self.correct, self.events, self.label = correct, events, label

    def search(self, query, k):
        q = self.queries[query]
        if self.events is not None:
            self.events.append((self.label, q["split"], query, k))
        if not self.correct:
            return []
        doc = q["relevant_ids"][0]
        return [{"id": doc, "text": self.docs[doc], "score": 0.75}]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))

    def test_bilingual_data_has_disjoint_topic_groups(self):
        study = validate_study(self.raw)
        self.assertFalse({q["group"] for q in study.split("validation")} &
                         {q["group"] for q in study.split("test")})
        for split in ("validation", "test"):
            self.assertEqual({q["language"] for q in study.split(split)}, {"en", "he"})
        self.assertEqual(study.fingerprint, load_study(DEFAULT_DATA).fingerprint)

    def test_duplicate_ids_and_normalized_text_are_rejected(self):
        for field, value in (("id", self.raw["documents"][0]["id"]),
                             ("text", "  " + self.raw["documents"][0]["text"].upper() + "\n")):
            raw = copy.deepcopy(self.raw)
            raw["documents"][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_study(raw)

    def test_query_copy_and_unicode_equivalent_text_are_rejected(self):
        self.raw["queries"][0]["text"] = self.raw["documents"][0]["text"]
        with self.assertRaisesRegex(ValueError, "copy"):
            validate_study(self.raw)
        self.raw["queries"][0]["text"] = "ＡＢＣ"
        self.raw["queries"][1]["text"] = "abc"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_study(self.raw)

    def test_topic_leakage_is_rejected_even_with_different_texts(self):
        held_out = next(q for q in self.raw["queries"] if q["split"] == "test")
        held_out["group"] = self.raw["queries"][0]["group"]
        held_out["relevant_ids"] = self.raw["queries"][0]["relevant_ids"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_study(self.raw)

    def test_invalid_or_conflicting_relevance_labels_are_rejected(self):
        for relevant in ([], ["absent"], ["d-backup", "d-backup"], ["d-refund"], [None]):
            raw = copy.deepcopy(self.raw)
            raw["queries"][0]["relevant_ids"] = relevant
            with self.subTest(relevant=relevant), self.assertRaises(ValueError):
                validate_study(raw)

    def test_bad_schema_empty_text_or_missing_split_is_rejected(self):
        mutations = [lambda r: r.update(schema_version=True), lambda r: r.update(documents=[]),
                     lambda r: r["queries"][0].update(text=" \n"),
                     lambda r: r["documents"][0].update(text="x" * 4001),
                     lambda r: r["queries"][0].update(split="train"),
                     lambda r: r["queries"][0].update(language="unknown"),
                     lambda r: r["documents"][0].update(unexpected=1),
                     lambda r: r.update(queries=[q for q in r["queries"] if q["split"] == "test"])]
        for mutate in mutations:
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(ValueError):
                validate_study(raw)

    def test_json_parse_error_and_file_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"queries":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line"):
                load_study(path)
            path.write_text(" " * 2_000_001, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "budget"):
                load_study(path)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(DEFAULT_DATA)

    def test_hand_computed_partial_relevance_and_empty_ranking(self):
        result = score_ranking(["a", "b"], ["b", "c"], 3)
        self.assertAlmostEqual(result["precision_at_k"], 1 / 3)
        self.assertEqual(result["recall_at_k"], 0.5)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertEqual(result["hit_at_k"], 1)
        self.assertTrue(all(value == 0 for value in score_ranking([], ["b"], 3).values()))

    def test_invalid_ranks_and_k_are_rejected(self):
        for ranking, gold, k in ((["a", "a"], ["a"], 2), (["a"], [], 1),
                                  (["a", "b"], ["a"], 1), ([], ["a"], 0),
                                  ([], ["a"], True), ([None], ["a"], 1)):
            with self.subTest(ranking=ranking, k=k), self.assertRaises(ValueError):
                score_ranking(ranking, gold, k)

    def test_aggregate_and_language_slices_use_actual_denominators(self):
        rows = evaluate_queries(self.study, self.study.split("validation"), RankingFixture(self.study), 3)
        rows[0].update(mrr_at_k=0.0, recall_at_k=0.0, hit_at_k=0.0, precision_at_k=0.0)
        result = summarize(rows)
        self.assertAlmostEqual(result["mrr_at_k"], (len(rows) - 1) / len(rows))
        en_count = sum(r["language"] == "en" for r in rows)
        self.assertAlmostEqual(result["by_language"]["en"]["recall_at_k"], (en_count - 1) / en_count)
        self.assertEqual(result["by_language"]["he"]["recall_at_k"], 1)
        with self.assertRaises(ValueError):
            summarize([])

    def test_unknown_duplicate_unsorted_and_nonfinite_evidence_fails(self):
        first, second = self.study.documents[:2]
        a = {"id": first["id"], "text": first["text"], "score": 0.8}
        b = {"id": second["id"], "text": second["text"], "score": 0.2}
        bad = [[{**a, "id": "unknown"}], [{**a, "text": "altered"}], [a, a],
               [b, a], [{**a, "score": math.nan}], [{**a, "score": math.inf}],
               [{**a, "score": True}], [a, b, a, b]]
        for hits in bad:
            class Broken:
                def search(self, query, k):
                    return hits
            with self.subTest(hits=hits), self.assertRaises(ValueError):
                evaluate_queries(self.study, self.study.split("validation"), Broken(), 3)

    def test_selection_precedes_every_test_query_and_uses_same_k(self):
        events = []
        factories = {name: (lambda n=name: RankingFixture(self.study, events=events, label=n))
                     for name in reversed(CANDIDATES)}
        result = compare(self.study, factories, 3,
                         on_selection=lambda choice: events.append(("selection", choice["selected"])))
        self.assertEqual(result["selection"]["selected"], "tfidf")  # predeclared tie break
        decision_index = next(i for i, event in enumerate(events) if event[0] == "selection")
        self.assertTrue(all(e[1] == "validation" for e in events[:decision_index]))
        self.assertTrue(all(e[1] == "test" for e in events[decision_index + 1:]))
        for name in CANDIDATES:
            observed = [(e[2], e[3]) for e in events if e[0] == name]
            expected = [(q["text"], 3) for split in ("validation", "test") for q in self.study.split(split)]
            self.assertEqual(observed, expected)

    def test_changed_test_labels_do_not_change_selection(self):
        raw = self.study.as_dict()
        held_out = next(q for q in raw["queries"] if q["split"] == "test")
        raw["documents"].append({"id": "test-distractor", "group": held_out["group"],
                                 "language": "en", "text": "An unrelated second paragraph in the same topic group."})
        original = validate_study(raw)
        held_out["relevant_ids"] = ["test-distractor"]
        changed = validate_study(raw)
        factories = {"tfidf": lambda: RankingFixture(original, correct=False),
                     "dense_mean": lambda: RankingFixture(original)}
        before, after = compare(original, factories, 3), compare(changed, factories, 3)
        self.assertEqual(before["selection"], after["selection"])
        self.assertNotEqual(before["test"], after["test"])

    def test_selector_rejects_test_rows_or_unpaired_candidates(self):
        rows = evaluate_queries(self.study, self.study.split("test"), RankingFixture(self.study), 3)
        with self.assertRaisesRegex(ValueError, "validation"):
            select_on_validation({"tfidf": rows})
        rows = evaluate_queries(self.study, self.study.split("validation"), RankingFixture(self.study), 3)
        with self.assertRaisesRegex(ValueError, "identical"):
            select_on_validation({"tfidf": rows, "dense_mean": rows[::-1]})
        rows[0]["mrr_at_k"] = math.nan
        with self.assertRaisesRegex(ValueError, "metric"):
            select_on_validation({"tfidf": rows})

    def test_paired_effects_average_topics_not_independent_paraphrases(self):
        left = [{"query_id": "a", "group": "g1", "split": "test", "mrr_at_k": 0.0},
                {"query_id": "b", "group": "g1", "split": "test", "mrr_at_k": 0.0},
                {"query_id": "c", "group": "g2", "split": "test", "mrr_at_k": 1.0}]
        right = [{**row, "mrr_at_k": 1.0} for row in left]
        result = paired_differences(left, right)
        self.assertAlmostEqual(result["mean_query_difference"], 2 / 3)
        self.assertEqual(result["mean_group_difference"], 0.5)
        with self.assertRaises(ValueError):
            paired_differences(left, right[::-1])

    def test_retriever_failure_aborts_without_test_search(self):
        events = []
        def fail():
            raise RuntimeError("fixture arm failed")
        with self.assertRaisesRegex(RuntimeError, "fixture arm"):
            compare(self.study, {"tfidf": lambda: RankingFixture(self.study, events=events),
                                 "dense_mean": fail}, 3)
        self.assertTrue(all(event[1] == "validation" for event in events))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "run"

    def test_prepare_snapshots_inputs_and_budget_without_model_results(self):
        study, protocol = prepare(DEFAULT_DATA, self.out)
        self.assertEqual(protocol["budget"]["max_encoder_texts"], 2 * (len(study.documents) + len(study.queries)))
        self.assertEqual(protocol["budget"]["training_steps"], 0)
        self.assertEqual(audit(self.out)["status"], "prepared")
        self.assertFalse((self.out / "summary.json").exists())
        self.assertIsNone(protocol["model"]["version"])

    def test_refuses_to_overwrite_previous_artifacts(self):
        prepare(DEFAULT_DATA, self.out)
        before = (self.out / "protocol.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(DEFAULT_DATA, self.out)
        self.assertEqual((self.out / "protocol.json").read_bytes(), before)

    def test_invalid_budget_never_creates_a_run(self):
        for option in ({"k": 0}, {"k": True}, {"batch_size": 0}, {"max_length": 513}, {"max_length": 1}, {"threads": 9}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                prepare(DEFAULT_DATA, self.out, **option)
            self.assertFalse(self.out.exists())

    def test_audit_detects_changed_data_and_false_completion(self):
        prepare(DEFAULT_DATA, self.out)
        write_json(self.out / "summary.json", {"status": "complete"})
        with self.assertRaisesRegex(ValueError, "Prepared"):
            audit(self.out)
        (self.out / "summary.json").unlink()
        raw = json.loads((self.out / "study.json").read_text(encoding="utf-8"))
        raw["documents"][0]["text"] += " Additional sentence."
        write_json(self.out / "study.json", raw)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            audit(self.out)

    def test_dependency_failure_is_recorded_and_cannot_pass_audit(self):
        # Force a missing dependency regardless of the test host's installed stack.
        with patch.dict(sys.modules, {"portfolio.retrieval": None}):
            with self.assertRaises(ImportError):
                run(DEFAULT_DATA, self.out, mode="baseline")
        self.assertEqual(json.loads((self.out / "status.json").read_text())["status"], "failed")
        self.assertTrue((self.out / "failure.json").exists())
        self.assertFalse((self.out / "summary.json").exists())
        with self.assertRaisesRegex(ValueError, "failed"):
            audit(self.out)

    def test_documented_cli_preparation_and_invalid_input_exit_codes(self):
        command = [sys.executable, "-m", "portfolio.retrieval_experiment", "prepare", "--out", str(self.out)]
        success = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertFalse(json.loads(success.stdout)["model_evaluated"])
        duplicate = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("never overwritten", duplicate.stderr)

    def test_finite_smoke_exercises_cli_and_tamper_detection(self):
        result = smoke()
        self.assertEqual(result["smoke"], "ok")
        self.assertEqual(result["status"], "prepared")
        self.assertIn("no model scored", result["scope"])

    def test_prepared_protocol_with_invalid_budget_fails_audit(self):
        prepare(DEFAULT_DATA, self.out)
        protocol = json.loads((self.out / "protocol.json").read_text())
        protocol["budget"]["max_encoder_forward_calls"] = 1
        write_json(self.out / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "budget"):
            audit(self.out)

    def write_ranking_artifact_fixture(self):
        # Deliberately hand-built rankings, only in a temporary test directory.
        # This checks artifact arithmetic, not whether an inference occurred.
        study, protocol = prepare(DEFAULT_DATA, self.out, mode="baseline")
        result = compare(study, {"tfidf": lambda: RankingFixture(study)}, 3,
                         on_selection=lambda decision: write_json(self.out / "selection.json", decision))
        rankings = {"validation": result.pop("validation_rows"), "test": result.pop("test_rows")}
        summary = {"status": "complete", "mode": "baseline", "k": 3,
                   "data_sha256": study.fingerprint, "protocol_sha256": sha256_json(protocol),
                   "rankings_sha256": sha256_json(rankings), "model_evaluated": False,
                   "encoder_version": None, **result}
        write_json(self.out / "rankings.json", rankings)
        write_json(self.out / "summary.json", summary)
        write_json(self.out / "status.json", {"status": "complete", "model_evaluated": False})
        return rankings, summary

    def test_completed_artifact_audit_recomputes_metrics(self):
        rankings, summary = self.write_ranking_artifact_fixture()
        self.assertEqual(audit(self.out)["status"], "complete")
        rankings["test"]["tfidf"][0]["mrr_at_k"] = 0.0
        write_json(self.out / "rankings.json", rankings)
        summary["rankings_sha256"] = sha256_json(rankings)
        write_json(self.out / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "metrics"):
            audit(self.out)

    def test_completed_artifact_audit_rejects_test_driven_selection(self):
        _, summary = self.write_ranking_artifact_fixture()
        summary["selection"]["selected"] = "dense_cls"
        write_json(self.out / "summary.json", summary)
        write_json(self.out / "selection.json", summary["selection"])
        with self.assertRaisesRegex(ValueError, "Selection"):
            audit(self.out)


if __name__ == "__main__":
    unittest.main()
