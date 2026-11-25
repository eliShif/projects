import unittest

from sigvision.perf import nearest_rank_percentile, peak_alloc_bytes, time_calls_us


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_values(self):
        values = list(range(10, 0, -1))
        self.assertEqual(nearest_rank_percentile(values, 50.0), 5.0)
        self.assertEqual(nearest_rank_percentile(values, 95.0), 10.0)
        self.assertEqual(nearest_rank_percentile(values, 100.0), 10.0)
        self.assertEqual(nearest_rank_percentile(values, 1.0), 1.0)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            nearest_rank_percentile([], 50.0)
        with self.assertRaises(ValueError):
            nearest_rank_percentile([1.0], 0.0)
        with self.assertRaises(ValueError):
            nearest_rank_percentile([1.0], 101.0)


class TimingTests(unittest.TestCase):
    def test_time_calls_counts_invocations_and_orders_percentiles(self):
        calls = []
        result = time_calls_us(lambda: calls.append(1), repeats=7, warmup=2)
        self.assertEqual(len(calls), 9)
        self.assertEqual(result["repeats"], 7.0)
        self.assertGreaterEqual(result["p95_us"], result["p50_us"])
        self.assertGreaterEqual(result["p50_us"], 0.0)

    def test_time_calls_rejects_zero_repeats(self):
        with self.assertRaises(ValueError):
            time_calls_us(lambda: None, repeats=0)

    def test_peak_alloc_sees_a_large_allocation(self):
        peak = peak_alloc_bytes(lambda: bytearray(2_000_000))
        self.assertGreater(peak, 1_000_000)


if __name__ == "__main__":
    unittest.main()
