from __future__ import annotations

import unittest

from _paths import FIXTURES
from physbench.io import load_jsonl
from physbench.splitters import build_view_a, build_view_b


class SplitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(FIXTURES / "cases.jsonl")

    def test_view_a_is_disjoint_and_complete(self) -> None:
        split = build_view_a(self.cases)
        case_ids = [
            case_id
            for scene in split["scenes"].values()
            for group in scene.values()
            for case_id in group
        ]
        self.assertEqual(len(self.cases), len(case_ids))
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_view_b_is_reproducible_and_balanced(self) -> None:
        first = build_view_b(self.cases, groups=2, seed=42)
        second = build_view_b(self.cases, groups=2, seed=42)
        self.assertEqual(first, second)
        for scene in first["scenes"].values():
            sizes = [len(group) for group in scene.values()]
            self.assertLessEqual(max(sizes) - min(sizes), 1)


if __name__ == "__main__":
    unittest.main()
