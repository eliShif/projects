import unittest

import numpy as np

from sigvision.metrics import match_points, match_scalars
from sigvision.vision import (
    SOBEL_X,
    SOBEL_Y,
    conv2d,
    edge_map,
    harris_corners,
    harris_response,
    make_scene,
    rect_corners,
    sobel_gradients,
)

TWO_RECTS = [(10, 12, 30, 34, 1.0), (40, 36, 58, 60, 0.9)]


def naive_convolution(image, kernel):
    """Direct translation of the definition, independent of conv2d's slicing."""
    h, w = image.shape
    kh, kw = kernel.shape
    ci, cj = kh // 2, kw // 2
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for i in range(kh):
                for j in range(kw):
                    yy, xx = y + ci - i, x + cj - j
                    if 0 <= yy < h and 0 <= xx < w:
                        acc += kernel[i, j] * image[yy, xx]
            out[y, x] = acc
    return out


class Conv2dTests(unittest.TestCase):
    def test_matches_naive_definition_for_asymmetric_kernel(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=(9, 11))
        ker = rng.normal(size=(3, 5))
        np.testing.assert_allclose(conv2d(img, ker), naive_convolution(img, ker), atol=1e-12)

    def test_identity_kernel_and_delta_response(self):
        rng = np.random.default_rng(1)
        img = rng.normal(size=(6, 7))
        identity = np.zeros((3, 3))
        identity[1, 1] = 1.0
        np.testing.assert_allclose(conv2d(img, identity), img, atol=1e-15)
        delta = np.zeros((7, 7))
        delta[3, 3] = 1.0
        ker = np.arange(1.0, 10.0).reshape(3, 3)
        np.testing.assert_allclose(conv2d(delta, ker)[2:5, 2:5], ker, atol=1e-15)

    def test_rejects_even_kernel_and_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            conv2d(np.ones((4, 4)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            conv2d(np.ones(4), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            conv2d(np.ones((4, 4)), np.ones(3))
        with self.assertRaises(ValueError):
            conv2d(np.zeros((0, 3)), np.ones((3, 3)))


class SobelEdgeTests(unittest.TestCase):
    def test_gradient_direction_on_ramps(self):
        h = w = 8
        xramp = np.tile(np.arange(w, dtype=float), (h, 1))
        gx, gy, _ = sobel_gradients(xramp)
        np.testing.assert_allclose(gx[1:-1, 1:-1], 8.0)
        np.testing.assert_allclose(gy[1:-1, 1:-1], 0.0, atol=1e-12)
        gx2, gy2, _ = sobel_gradients(xramp.T)
        np.testing.assert_allclose(gy2[1:-1, 1:-1], 8.0)
        np.testing.assert_allclose(gx2[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_kernels_are_separable_derivative_smoothers(self):
        smooth = np.array([1.0, 2.0, 1.0])
        deriv = np.array([1.0, 0.0, -1.0])
        np.testing.assert_array_equal(SOBEL_X, np.outer(smooth, deriv))
        np.testing.assert_array_equal(SOBEL_Y, np.outer(deriv, smooth))
        self.assertAlmostEqual(float(SOBEL_X.sum()), 0.0)
        self.assertAlmostEqual(float(SOBEL_Y.sum()), 0.0)

    def test_edge_map_marks_boundary_not_interior_or_background(self):
        img = make_scene((64, 64), [(10, 12, 30, 34, 1.0)])
        edges = edge_map(img)
        self.assertTrue(edges[10, 20] or edges[9, 20])  # top edge
        self.assertTrue(edges[20, 12] or edges[20, 11])  # left edge
        self.assertFalse(edges[20, 23])  # interior
        self.assertFalse(edges[2, 2])  # background

    def test_edge_map_of_blank_image_is_empty_and_threshold_validated(self):
        self.assertFalse(edge_map(np.zeros((8, 8))).any())
        with self.assertRaises(ValueError):
            edge_map(np.zeros((8, 8)), threshold_ratio=0.0)


class HarrisTests(unittest.TestCase):
    def test_all_rectangle_corners_found_without_false_corners(self):
        img = make_scene((64, 64), TWO_RECTS)
        truth = rect_corners(TWO_RECTS)
        self.assertEqual(len(truth), 8)
        detected = harris_corners(img)
        matched, false = match_points(truth, detected, tol=3.0)
        self.assertEqual(matched, 8)
        self.assertEqual(false, 0)

    def test_corners_found_with_mild_noise(self):
        img = make_scene((64, 64), TWO_RECTS, noise_std=0.01, seed=3)
        matched, false = match_points(rect_corners(TWO_RECTS), harris_corners(img), tol=3.0)
        self.assertEqual(matched, 8)
        self.assertEqual(false, 0)

    def test_edges_have_negative_response_corners_positive(self):
        img = make_scene((64, 64), [(10, 12, 30, 34, 1.0)])
        resp = harris_response(img)
        self.assertLess(resp[20, 12], 0.0)  # middle of left edge
        self.assertGreater(resp.max(), 0.0)

    def test_blank_image_has_no_corners(self):
        self.assertEqual(harris_corners(np.zeros((32, 32))), [])

    def test_invalid_parameters(self):
        img = make_scene((32, 32), [(8, 8, 20, 20, 1.0)])
        with self.assertRaises(ValueError):
            harris_corners(img, rel_threshold=0.0)
        with self.assertRaises(ValueError):
            harris_corners(img, min_distance=-1)
        with self.assertRaises(ValueError):
            harris_response(img, window=4)
        with self.assertRaises(ValueError):
            harris_response(img, k=0.0)


class SceneAndMetricTests(unittest.TestCase):
    def test_scene_validation_and_determinism(self):
        with self.assertRaises(ValueError):
            make_scene((10, 10), [(5, 5, 11, 8, 1.0)])
        with self.assertRaises(ValueError):
            make_scene((10, 10), [(5, 5, 5, 8, 1.0)])
        a = make_scene((16, 16), [(4, 4, 10, 10, 1.0)], noise_std=0.1, seed=2)
        b = make_scene((16, 16), [(4, 4, 10, 10, 1.0)], noise_std=0.1, seed=2)
        np.testing.assert_array_equal(a, b)

    def test_rect_corners(self):
        self.assertEqual(
            rect_corners([(2, 3, 6, 9, 1.0)]), [(2, 3), (2, 8), (5, 3), (5, 8)]
        )

    def test_match_points_counts_misses_and_false_detections(self):
        truth = [(0, 0), (10, 10)]
        matched, false = match_points(truth, [(1, 1), (30, 30)], tol=2.0)
        self.assertEqual((matched, false), (1, 1))
        with self.assertRaises(ValueError):
            match_points(truth, [], tol=-1.0)

    def test_match_scalars(self):
        self.assertEqual(match_scalars([10.0, 20.0], [10.4, 50.0], tol=1.0), (1, 1))
        self.assertEqual(match_scalars([], [], tol=1.0), (0, 0))


if __name__ == "__main__":
    unittest.main()
