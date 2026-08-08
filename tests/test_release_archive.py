from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_release_archive import verify_archive


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArchiveTests(unittest.TestCase):
    def test_head_archive_is_self_contained_and_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vphysbench-archive-test-") as directory:
            issues = verify_archive(ROOT, Path(directory))

        self.assertEqual([], issues)


if __name__ == "__main__":
    unittest.main()
