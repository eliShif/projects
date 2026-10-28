import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from portfolio.bilingual_eval import dataset, evaluate, lexical_baseline, model_predictions, run


class BilingualEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.corpus, self.cases = dataset()
        self.predictions = lexical_baseline(self.corpus, self.cases)

    def test_paired_questions_and_offline_run(self):
        by_language = {language: {case['pair_id'] for case in self.cases if case['language'] == language}
                       for language in ('en', 'he')}
        self.assertEqual(by_language['en'], by_language['he'])
        with tempfile.TemporaryDirectory() as directory:
            summary = run(directory)
            self.assertEqual(summary['en']['cases'], 3)
            self.assertEqual(summary['he']['cases'], 3)
            for name in ('protocol.json', 'corpus.json', 'cases.json', 'predictions.json', 'runs.json', 'summary.json'):
                self.assertTrue((Path(directory) / name).is_file())
            protocol = json.loads((Path(directory) / 'protocol.json').read_text(encoding='utf-8'))
            self.assertIn('predictions_sha256', protocol)
            self.assertIn('bilingual_eval.py', protocol['source_hashes'])

    def test_supplied_predictions_run_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'answers.json'
            path.write_text(json.dumps(self.predictions), encoding='utf-8')
            output = Path(directory) / 'output'
            summary = run(output, path)
            protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol['mode'], 'supplied')
            self.assertEqual(sum(item['cases'] for item in summary.values()), len(self.cases))

    def test_wrong_value_missing_citation_and_bad_abstention(self):
        predictions = [{"case_id": c['id'], "status": "abstained", "value": None, "citations": []}
                       if c['expected'] is None else
                       {"case_id": c['id'], "status": "answered", "value": c['expected'], "citations": c['evidence_ids']}
                       for c in self.cases]
        predictions[0]['value'] = 999
        predictions[1]['citations'] = []
        predictions[5] = {"case_id": self.cases[5]['id'], "status": "answered", "value": 23,
                          "citations": ['cedar-current']}
        rows, summary = evaluate(self.corpus, self.cases, predictions)
        self.assertEqual(rows[0]['error_type'], 'wrong_value_or_abstention')
        self.assertEqual(rows[1]['error_type'], 'missing_expected_evidence')
        self.assertEqual(rows[5]['error_type'], 'wrong_value_or_abstention')
        self.assertEqual(summary['en']['grounded_correct'], 1)
        self.assertEqual(summary['he']['grounded_correct'], 2)

    def test_rejects_incomplete_duplicate_and_unknown_citations(self):
        with self.assertRaises(ValueError):
            evaluate(self.corpus, self.cases, self.predictions[:-1])
        with self.assertRaises(ValueError):
            evaluate(self.corpus, self.cases, self.predictions + [self.predictions[0]])
        predictions = [dict(p) for p in self.predictions]
        predictions[0]['citations'] = ['invented']
        with self.assertRaises(ValueError):
            evaluate(self.corpus, self.cases, predictions)

    def test_lexical_baseline_is_not_oracle(self):
        predictions = lexical_baseline(self.corpus, self.cases)
        altered_cases = [dict(case) for case in self.cases]
        altered_cases[0]['expected'] = 999
        self.assertEqual(predictions, lexical_baseline(self.corpus, altered_cases))
        rows, _ = evaluate(self.corpus, altered_cases, predictions)
        self.assertFalse(rows[0]['correct'])

    def test_local_model_predictions_and_error_capture(self):
        class StubModel:
            def __init__(self):
                self.calls = []

            def generate_json(self, prompt, request, schema, seed):
                self.calls.append((prompt, json.loads(request), schema, seed))
                index = seed - 4200
                if index == 1:
                    return {'text': '{bad json'}
                if index == 2:
                    return {'text': json.dumps({'value': 17, 'citations': ['dune-current']})}
                return {'text': json.dumps({'value': None, 'citations': []})}

        model = StubModel()
        predictions, generations = model_predictions(self.corpus, self.cases, model)
        self.assertEqual(len(model.calls), len(self.cases))
        self.assertIn('מה ערך', model.calls[3][1]['question'])
        self.assertEqual(predictions[1]['status'], 'failed')
        self.assertEqual(predictions[2]['status'], 'failed')
        self.assertEqual(generations[1]['response']['text'], '{bad json')
        rows, _ = evaluate(self.corpus, self.cases, predictions)
        self.assertEqual(rows[1]['error_type'], 'generation_failure')

    def test_comparison_outputs_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            run(directory)
            comparison = json.loads((Path(directory) / 'comparison.json').read_text())
            errors = json.loads((Path(directory) / 'errors.json').read_text())
            self.assertEqual(comparison['en']['paired_difference'], 0)
            self.assertTrue(all('question' in row and 'error_type' in row for row in errors))
            with self.assertRaises(FileNotFoundError):
                run(Path(directory) / 'missing', model_path=Path(directory) / 'no-model.gguf')

    def test_model_run_writes_provenance_and_generations(self):
        class StubGGUF:
            model_name = 'test-double.gguf'
            sha256 = 'test-double-only'

            def __init__(self, path):
                pass

            def generate_json(self, prompt, request, schema, seed):
                return {'text': '{"value": null, "citations": []}'}

        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / 'test-double.gguf'
            model_file.write_bytes(b'test double; not model weights')
            output = Path(directory) / 'out'
            with patch('portfolio.gguf_lm.GGUFModel', StubGGUF):
                run(output, model_path=model_file)
            protocol = json.loads((output / 'protocol.json').read_text())
            generations = json.loads((output / 'generations.json').read_text())
            self.assertEqual(protocol['mode'], 'local_gguf')
            self.assertEqual(protocol['model']['sha256'], 'test-double-only')
            self.assertEqual(len(generations), len(self.cases))
            self.assertEqual(generations[0]['response']['text'], '{"value": null, "citations": []}')

    def test_rejects_nonfinite_predictions(self):
        predictions = [dict(p) for p in self.predictions]
        predictions[0]['value'] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(self.corpus, self.cases, predictions)


if __name__ == '__main__':
    unittest.main()
