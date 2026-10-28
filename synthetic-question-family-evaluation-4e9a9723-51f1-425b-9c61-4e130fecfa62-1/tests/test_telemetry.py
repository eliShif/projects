import unittest
from portfolio.telemetry import make_dataset,normalize,analyze,features


class TelemetryTests(unittest.TestCase):
    def test_deduplication_and_conflict(self):
        events,_=make_dataset(1,4)
        self.assertEqual(len(normalize(events+events)),len(events))
        with self.assertRaises(ValueError):normalize(events+[{**events[0],'country':'FR'}])

    def test_labels_are_not_model_features(self):
        events,labels=make_dataset(1,4)
        self.assertTrue(all('label' not in event and 'scenario' not in event for event in events))
        self.assertEqual(features(events,[r['actor'] for r in labels]).shape,(4,7))

    def test_actor_and_time_splits_are_disjoint(self):
        tr,_=make_dataset(17,240,'train');va,_=make_dataset(29,120,'validation');te,_=make_dataset(43,240,'test')
        self.assertLess(max(r['ts'] for r in tr),min(r['ts'] for r in va))
        self.assertLess(max(r['ts'] for r in va),min(r['ts'] for r in te))
        self.assertFalse({r['actor'] for r in tr}&{r['actor'] for r in te})

    def test_alert_evidence_exists_and_is_same_actor(self):
        events,_=make_dataset(1,40);byid={r['id']:r for r in events}
        for alert in analyze(events):
            self.assertTrue(all(i in byid and byid[i]['actor']==alert['actor'] for i in alert['evidence_ids']))

    def test_order_and_duplicate_invariance(self):
        events,_=make_dataset(1,24)
        self.assertEqual(analyze(events),analyze(list(reversed(events))+events))

    def test_negative_controls_and_unseen_slow_behavior(self):
        events,labels=make_dataset(43,80,'test');detected={r['actor'] for r in analyze(events)}
        false_negatives=[r for r in labels if r['label'] and r['actor'] not in detected]
        false_positives=[r for r in labels if not r['label'] and r['actor'] in detected]
        self.assertTrue(false_negatives);self.assertTrue(false_positives)
