"""Regression checks for the shared generator and the new split-aware mode."""
import hashlib
import json
import unittest

from portfolio.research_cases import build_cases, entity_assignments
from portfolio.research_lab import SPLITS


class ResearchCaseTests(unittest.TestCase):
    def test_default_datasets_remain_identical_to_inspected_source(self):
        # Captured from the pinned implementation before the pool change.
        expected = {
            "dev": "9404c0c331683ffb6ba6a2f8d18925bd622deda4832462d488b7af12e7ce3db9",
            "test": "c54e0c5b9325b024e53f3559c264cccaccf0951750e8b12d8622ba5df572e125",
        }
        for split, digest in expected.items():
            with self.subTest(split=split):
                records, cases = build_cases(split)
                payload = json.dumps([[r.as_dict() for r in records], cases], sort_keys=True)
                self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), digest)

    def test_comparison_operands_and_answers_stay_in_the_assigned_pool(self):
        records, cases = build_cases("test", entity_groups=SPLITS)
        records_by_id = {r.id: r for r in records}
        pools = entity_assignments(SPLITS)
        for row in cases:
            if row["family"] not in ("sum", "difference", "ratio"):
                continue
            with self.subTest(case=row["id"]):
                a, b = (records_by_id[key] for key in row["evidence_ids"])
                self.assertNotEqual(a.service, b.service)
                self.assertEqual(pools[a.service], pools[b.service])
                self.assertEqual(a.service, row["entity"])
                self.assertIn(b.service, row["question"])
                expected = {"sum": a.facts["daily_quota"] + b.facts["daily_quota"],
                            "difference": a.facts["retention_days"] - b.facts["retention_days"],
                            "ratio": a.facts["monthly_cost"] / b.facts["monthly_cost"]}[row["family"]]
                self.assertEqual(row["expected"], expected)
        # These were boundary crossings in the old global ring of names.
        by_id = {r["id"]: r for r in cases}
        self.assertIn("Cobalt", by_id["test-Ember-sum"]["question"])
        self.assertIn("Ibis", by_id["test-Fjord-sum"]["question"])
        self.assertIn("Garnet", by_id["test-Helix-sum"]["question"])

    def test_invalid_pools_fail_before_generation(self):
        invalid = [({}, "nonempty"), ({"train": ["Cobalt"]}, "at least two"),
                   ({"train": ["Cobalt", "cobalt"]}, "disjoint"),
                   ({"train": ["Cobalt", "Delta"], "test": ["Cobalt", "Ember"]}, "disjoint"),
                   ({"train": ["Cobalt", "CobaltMirror"]}, "disjoint"),
                   ({"train": ["Cobalt", "bad name"]}, "alphanumeric")]
        for pools, message in invalid:
            with self.subTest(pools=pools), self.assertRaisesRegex(ValueError, message):
                build_cases("test", entity_groups=pools)


if __name__ == "__main__":
    unittest.main()
