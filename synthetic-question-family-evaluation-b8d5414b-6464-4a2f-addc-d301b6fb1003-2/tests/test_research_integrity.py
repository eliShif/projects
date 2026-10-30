"""Success and failure paths for entity isolation, class support and selection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from portfolio.research_cases import build_cases
from portfolio.research_lab import (SPLITS, FAMILIES, CANDIDATES, TextBaseline,
                                    cases_from_generator, challenge_cases, validate_cases,
                                    mentioned_entities, evaluate, fingerprint, run)
from scripts.smoke_research_lab import check_artifacts


class ResearchIntegrityTests(unittest.TestCase):
    def test_all_mentions_are_separated_in_original_and_challenge_cases(self):
        rows = cases_from_generator()
        validate_cases(rows)
        challenge = challenge_cases([r for r in rows if r["split"] == "test"])
        observed = {split: set() for split in SPLITS}
        for row in rows + challenge:
            references = mentioned_entities(row["question"])
            self.assertIn(row["entity"], references)
            self.assertLessEqual(references, set(SPLITS[row["split"]]))
            if row["label"] in ("sum", "ratio", "difference"):
                self.assertEqual(len(references), 2)
            observed[row["split"]].update(references)
        self.assertFalse(observed["train"] & observed["validation"])
        self.assertFalse(observed["train"] & observed["test"])
        self.assertFalse(observed["validation"] & observed["test"])

    def test_legacy_global_ring_is_rejected_by_secondary_mentions(self):
        _, legacy = build_cases("test")
        pools = {e: split for split, names in SPLITS.items() for e in names}
        rows = [{"id": r["id"], "entity": r["entity"], "label": r["family"],
                 "question": r["question"], "split": pools[r["entity"]]} for r in legacy]
        with self.assertRaisesRegex(ValueError, "mentions.*Ember-difference.*Fjord"):
            validate_cases(rows)

    def test_secondary_leak_and_aliases_fail_before_any_output(self):
        for foreign in ("Fjord", "gArNeT", "GarnetMirror", "GarnetSandbox"):
            with self.subTest(foreign=foreign), TemporaryDirectory() as directory:
                rows = cases_from_generator()
                row = next(r for r in rows if r["entity"] == "Cobalt" and r["label"] == "sum")
                # Keep primary entity, split and label intact: only the second
                # mention is changed, so the old primary-only audit missed it.
                row["question"] = f"What is the combined quota for Cobalt and {foreign}?"
                source = Path(directory) / "cases.json"
                source.write_text(json.dumps(rows), encoding="utf-8")
                out = Path(directory) / "out"
                with self.assertRaisesRegex(ValueError, "leakage through question mentions"):
                    run(out, source)
                self.assertFalse(out.exists())

    def test_aliases_share_root_and_matching_respects_word_boundaries(self):
        self.assertEqual(mentioned_entities("COBALT's quota with DeltaMirror and EmberSandbox"),
                         {"Cobalt", "Delta", "Ember"})
        self.assertEqual(mentioned_entities("Cobaltite and Garnetized are not service names"), set())
        rows = cases_from_generator()
        rows[0]["question"] = "What is Cobaltite storage?"
        with self.assertRaisesRegex(ValueError, "Primary entity must be mentioned"):
            validate_cases(rows)

    def test_all_family_counts_include_zero_and_ratio_is_explicitly_undefined(self):
        for split in ("validation", "test"):
            with self.subTest(split=split):
                rows = [r for r in cases_from_generator()
                        if not (r["split"] == split and r["label"] == "ratio")]
                item = validate_cases(rows)[split]
                self.assertEqual(set(item["label_counts"]), set(FAMILIES))
                self.assertEqual(item["label_counts"]["ratio"], 0)
                self.assertEqual(item["missing_labels"], ["ratio"])
                self.assertIsNone(item["imbalance_ratio"])
                # No Infinity or NaN may escape into JSON.
                self.assertIn('"imbalance_ratio": null', json.dumps(item, allow_nan=False))

    def test_missing_family_in_each_split_stops_run_before_fitting_or_writing(self):
        for split in SPLITS:
            with self.subTest(split=split), TemporaryDirectory() as directory:
                rows = [r for r in cases_from_generator()
                        if not (r["split"] == split and r["label"] == "ratio")]
                source = Path(directory) / "cases.json"
                source.write_text(json.dumps(rows), encoding="utf-8")
                out = Path(directory) / "out"
                with patch.object(TextBaseline, "fit") as fit:
                    with self.assertRaisesRegex(ValueError, "missing: ratio"):
                        run(out, source)
                    fit.assert_not_called()
                self.assertFalse(out.exists())

    def test_missing_family_metrics_still_average_all_eight_labels(self):
        train = [r for r in cases_from_generator() if r["split"] == "train"]
        cases = [{"id": "single", "entity": "Garnet", "label": "conflict", "question": "Garnet"}]
        score = evaluate(TextBaseline("majority").fit(train), cases)
        self.assertEqual(score["accuracy"], 1)
        self.assertEqual(score["macro_f1"], 1 / 8)
        self.assertEqual(score["per_label"]["ratio"],
                         {"support": 0, "predicted": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            TextBaseline("bigram").fit([])

    def test_insufficient_downsampling_support_is_rejected_before_fitting(self):
        rows = [r for r in cases_from_generator()
                if not (r["split"] == "train" and r["label"] == "lookup" and r["entity"] != "Cobalt")]
        self.assertEqual(validate_cases(rows)["train"]["label_counts"]["lookup"], 1)
        with TemporaryDirectory() as directory, patch.object(TextBaseline, "fit") as fit:
            source = Path(directory) / "cases.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            out = Path(directory) / "out"
            with self.assertRaisesRegex(ValueError, "at least two lookup"):
                run(out, source)
            fit.assert_not_called()
            self.assertFalse(out.exists())

    def test_selection_is_persisted_before_test_and_uses_training_only(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            calls, fits = [], []
            original_fit = TextBaseline.fit

            def trace_fit(model, rows):
                fits.append((model.name, {r["split"] for r in rows}))
                return original_fit(model, rows)

            def trace_evaluate(model, rows):
                phase = rows[0]["split"]
                if len(calls) < 9:
                    calls.append((model.name, phase))
                    if phase == "test":
                        decision = json.loads((out / "selection.json").read_text())
                        self.assertEqual(len(fits), 3)  # No refit before primary evaluation.
                        self.assertEqual(decision["selection_split"], "validation")
                        expected = min(CANDIDATES, key=lambda n: (-decision["candidate_metrics"][n]["macro_f1"],
                                                               -decision["candidate_metrics"][n]["accuracy"], n))
                        self.assertEqual(decision["selected"], expected)
                return evaluate(model, rows)

            with patch.object(TextBaseline, "fit", trace_fit), patch("portfolio.research_lab.evaluate", trace_evaluate):
                summary = run(out)
            self.assertEqual(calls, [(name, phase) for phase in ("train", "validation", "test") for name in CANDIDATES])
            self.assertEqual(fits[:3], [(name, {"train"}) for name in CANDIDATES])
            decision = json.loads((out / "selection.json").read_text())
            self.assertEqual(summary["selected"], decision["selected"])
            self.assertEqual(summary["selection_sha256"], fingerprint(decision))
            self.assertFalse(set(decision["fit_case_ids"]) & set(decision["validation_case_ids"]))
            protocol = json.loads((out / "protocol.json").read_text())
            for path, digest in protocol["source_hashes"].items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_controlled_validation_choice_is_not_overridden_by_better_test_scores(self):
        # This is a control-flow fixture, not a measured experiment result.
        def validation_fixture(model, rows):
            score = evaluate(model, rows)
            if all(r["split"] == "validation" for r in rows):
                score["macro_f1"] = score["accuracy"] = float(model.name == "majority")
            return score

        with TemporaryDirectory() as directory, patch("portfolio.research_lab.evaluate", validation_fixture):
            summary = run(directory)
            self.assertEqual(summary["selected"], "majority")
            self.assertGreater(summary["test"]["bigram"]["accuracy"], summary["test"]["majority"]["accuracy"])

    def test_cli_rejects_malformed_file_without_output(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "broken.json"
            source.write_text("{broken", encoding="utf-8")
            out = Path(directory) / "out"
            completed = subprocess.run([sys.executable, "-m", "portfolio.research_lab", "--cases", str(source),
                                        "--out", str(out)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("error:", completed.stderr)
            self.assertFalse(out.exists())

    def test_smoke_artifact_audit_accepts_real_output_and_rejects_tampering(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            summary = run(out)
            winner = summary["selected"]
            printed = {"selected": winner, "test_accuracy": summary["test"][winner]["accuracy"],
                       "challenge_accuracy": summary["new_template_challenge"][winner]["accuracy"]}
            check_artifacts(out, printed)
            path = out / "cases.json"
            rows = json.loads(path.read_text())
            rows[0]["question"] += " Changed after evaluation."
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Input hash mismatch"):
                check_artifacts(out, printed)


if __name__ == "__main__":
    unittest.main()
