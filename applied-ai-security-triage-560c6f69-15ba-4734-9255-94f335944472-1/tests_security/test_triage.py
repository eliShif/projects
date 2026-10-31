import copy
import json
import unittest

from portfolio.evidence_agent import Record
from portfolio.security_data import build_dataset, validate_dataset
from portfolio.security_triage import (SecurityTriageEngine, RulePlanner, JSONPlanner,
    prepare_observation, investigate, validate_alert, POLICIES)


class OutputFixture:
    """Adversarial output fixture, never counted as inference evidence."""
    def __init__(self, output):
        self.output = output

    def plan(self, observation):
        return {"text": self.output, "backend": "test_fixture"}


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.data = build_dataset()
        self.case = self.data["test"][0]

    def observation(self, case=None, policies=POLICIES):
        c = case or self.case
        return prepare_observation(c["alert"], c["events"], policies)

    def test_rule_outcomes_for_every_family_with_citations(self):
        for case in self.data["test"]:
            with self.subTest(family=case["family"]):
                result = investigate(self.observation(case), RulePlanner())
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["disposition"], case["expected"]["disposition"])
                self.assertEqual(result["response_actions"], [])
                if result["disposition"] != "abstain":
                    self.assertIn("triage-policy-v1", result["citations"])
                    self.assertIn(case["alert"]["anchor_id"], result["citations"])

    def test_foreign_actor_future_and_wrong_source_events_cannot_trigger(self):
        case = copy.deepcopy(self.data["test"][1])  # ordinary retries
        for i in range(12):
            event = dict(case["events"][0], id=f"extra-{i}")
            if i < 4:
                event["actor"] = "foreign"
            elif i < 8:
                event["ts"] = case["alert"]["end"] + 1
            else:
                event["source_ip"] = "198.51.100.5"
            case["events"].append(event)
        result = investigate(self.observation(case), RulePlanner())
        self.assertEqual(result["disposition"], "no_finding")
        self.assertFalse(any(f"extra-{i}" in result["citations"] for i in range(8)))

    def test_no_finding_requires_complete_window(self):
        case = copy.deepcopy(self.case)
        case["alert"]["start"] += 10
        result = investigate(self.observation(case), RulePlanner())
        self.assertEqual(result["reason"], "incomplete_window")
        self.assertEqual(result["disposition"], "abstain")

    def test_exact_duplicates_do_not_inflate_counts(self):
        case = copy.deepcopy(self.data["test"][1])
        case["events"] += [case["events"][0]] * 10
        self.assertEqual(investigate(self.observation(case), RulePlanner())["disposition"], "no_finding")
        case["events"].append(dict(case["events"][0], kind="secret_read"))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.observation(case)

    def test_missing_conflicting_and_removed_policy_abstain(self):
        self.assertEqual(investigate(self.observation(policies=[]), RulePlanner())["reason"], "missing_policy")
        policies = [Record("one", "triage-policy", 2, {"failure_threshold": 8, "window_seconds": 120}),
                    Record("two", "triage-policy", 2, {"failure_threshold": 2, "window_seconds": 120})]
        self.assertEqual(investigate(self.observation(policies=policies), RulePlanner())["reason"], "conflicting_current_evidence")
        policies = POLICIES + [Record("new", "triage-policy", 2, {"window_seconds": 120})]
        self.assertEqual(investigate(self.observation(policies=policies), RulePlanner())["reason"], "missing_current_fact")

    def test_event_budget_fails_closed(self):
        case = copy.deepcopy(self.case)
        for i in range(65):
            case["events"].append(dict(case["events"][0], id=f"overflow-{i}"))
        obs = self.observation(case)
        self.assertEqual(obs["events"], [])
        self.assertEqual(investigate(obs, RulePlanner())["reason"], "event_budget_exceeded")

    def test_forbidden_action_scope_and_semantic_errors_rejected(self):
        obs = self.observation()
        bad = [{"operation": "shell", "alert_id": self.case["id"]},
               {"operation": "disable_account", "alert_id": self.case["id"]},
               {"operation": "investigate_auth", "alert_id": "another-alert"},
               {"operation": "investigate_access", "alert_id": self.case["id"]},
               {"operation": "investigate_auth", "alert_id": self.case["id"], "finding": "safe"},
               {"operation": "investigate_auth", "alert_id": self.case["id"], "citations": ["invented"]}]
        for plan in bad:
            with self.subTest(plan=plan):
                result = investigate(obs, OutputFixture(json.dumps(plan)))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["trace"][-1]["tool"], "validate_plan")
                self.assertEqual(result["response_actions"], [])

    def test_malformed_truncated_or_oversized_output_is_retained_and_failed(self):
        for raw in ("{", "null", "[]", "x" * 2001):
            result = investigate(self.observation(), OutputFixture(raw))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["model_response"]["text"], raw)

    def test_planner_cannot_rewrite_evidence_used_by_the_executor(self):
        observation = self.observation()
        original = copy.deepcopy(observation)

        class MutatingPlanner(RulePlanner):
            def plan(self, supplied):
                supplied["policy"]["failure_threshold"] = 32
                supplied["events"][0]["kind"] = "secret_read"
                supplied["policy_evidence"][0]["source_id"] = "fabricated-policy"
                return super().plan(supplied)

        result = investigate(observation, MutatingPlanner())
        self.assertEqual(observation, original)
        self.assertEqual(result["disposition"], "escalate")
        self.assertNotIn("fabricated-policy", result["citations"])
        self.assertEqual(result["finding"], investigate(original, RulePlanner())["finding"])

    def test_ambiguous_json_plan_is_rejected_before_execution(self):
        alert_id = json.dumps(self.case["id"])
        for raw in (
            '{"operation":"shell","operation":"investigate_auth","alert_id":' + alert_id + '}',
            '{"operation":"abstain","alert_id":NaN}',
            '[' * 1000 + '0' + ']' * 1000,
        ):
            with self.subTest(raw=raw[:80]):
                result = investigate(self.observation(), OutputFixture(raw))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["model_response"]["text"], raw)

    def test_hostile_note_cannot_inject_tools_or_generated_finding(self):
        hostile = self.data["test"][7]
        result = investigate(self.observation(hostile), RulePlanner())
        self.assertEqual(result["disposition"], "escalate")
        self.assertNotIn("disable", result["finding"])
        class HostileModel:
            def generate_json(self, system, question, schema, seed):
                self.observed = question
                return {"text": json.dumps({"operation": "shell", "alert_id": hostile["id"]}), "backend": "test_fixture"}
        model = HostileModel()
        result = investigate(self.observation(hostile), JSONPlanner(model))
        self.assertIn("Ignore all rules", model.observed)
        self.assertEqual(result["status"], "failed")

    def test_structured_engine_uses_bounded_agent(self):
        result = SecurityTriageEngine(RulePlanner(), self.case["events"]).answer(
            {"mode": "security_triage", "alert": self.case["alert"]})
        self.assertTrue(result["read_only"])
        self.assertEqual(result["structured_answer"]["disposition"], "escalate")
        self.assertTrue(all(e["actor"] == self.case["alert"]["actor"] for e in result["evidence"]))

    def test_alert_and_event_schema_failures(self):
        for change in ({"start": True}, {"complete": "yes"}, {"actor": "../foreign"}, {"note": "x" * 501}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_alert({**self.case["alert"], **change})
        with self.assertRaises(ValueError):
            prepare_observation(self.case["alert"], [dict(self.case["events"][0], ts=True)])
        with self.assertRaises(ValueError):
            prepare_observation(self.case["alert"], [None])


class DataTests(unittest.TestCase):
    def test_budget_abstention_is_part_of_the_label_contract(self):
        data = build_dataset()
        case = data["test"][0]
        case["events"] += [dict(case["events"][0], id=f"budget-{i}") for i in range(65)]
        with self.assertRaisesRegex(ValueError, "Label"):
            validate_dataset(data)
        case["expected"] = {"operation": "abstain", "disposition": "abstain"}
        validate_dataset(data)

    def test_wrong_label_and_false_abstention_are_rejected(self):
        for wrong in ("no_finding", "abstain"):
            data = build_dataset()
            data["test"][0]["expected"]["disposition"] = wrong
            with self.assertRaisesRegex(ValueError, "Label"):
                validate_dataset(data)

    def test_actor_and_event_leakage_rejected(self):
        for field in ("actor", "id"):
            data = build_dataset()
            # Changing a distractor preserves the label while creating leakage.
            data["test"][0]["events"][-1][field] = data["train"][0]["events"][-1][field]
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_dataset(data)
