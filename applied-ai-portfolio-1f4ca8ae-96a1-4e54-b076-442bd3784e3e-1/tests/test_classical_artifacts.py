"""Regression: the native program is built outside the repository and no binary is left in the tree."""
from pathlib import Path
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

from portfolio import classical_native as native


class StaleArtifactTests(unittest.TestCase):
    def test_stale_binary_is_removed_and_absence_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / 'classical_filters.bin'
            fake.write_text('stale')
            with patch.object(native, 'BINARY', fake):
                self.assertTrue(native.remove_stale_artifact())
                self.assertFalse(fake.exists())
                self.assertFalse(native.remove_stale_artifact())

    def test_unbuilt_program_without_build_permission_is_unavailable(self):
        with patch.dict(native._fallback_build, {}, clear=True):
            with self.assertRaises(native.NativeUnavailable):
                native.binary_path(build_if_missing=False)

    def test_missing_compiler_is_reported_as_unavailable(self):
        with patch.dict(native._fallback_build, {}, clear=True), patch.object(native.shutil, 'which', return_value=None):
            with self.assertRaises(native.NativeUnavailable):
                native.binary_path()


class BuildLocationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.path = native.binary_path()
        except native.NativeUnavailable as exc:
            self.skipTest('g++ unavailable: %s' % exc)

    def test_binary_is_outside_the_repository(self):
        self.assertNotIn(native.ROOT, self.path.resolve().parents)
        self.assertTrue(self.path.is_file())

    def test_build_check_succeeds_and_leaves_no_binary_in_cpp(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(native.main(), 0)
        self.assertIn('native build ok', out.getvalue())
        self.assertEqual(sorted(p.name for p in (native.ROOT / 'cpp').glob('*.bin')), [])


if __name__ == '__main__':
    unittest.main()
