from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from _paths import ROOT


class ReferenceObservationCurationReviewTests(unittest.TestCase):
    @staticmethod
    def _review_api():
        from physbench.reference_observations.curation import review

        return review

    @staticmethod
    def _visualization_api():
        from physbench.reference_observations.curation import visualization

        return visualization

    def test_review_ledger_rejects_duplicate_case_rows(self) -> None:
        review = self._review_api()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            row = {
                "schema_version": "1.0",
                "case_id": "a",
                "scene_id": "pendulum",
                "anchor_decision": "pass",
                "tube_decision": "pass",
                "evidence": ["anchor.png", "contact.png"],
                "reviewer": "codex",
                "notes": "checked",
                "issues": [],
                "repairs": [],
                "final_asset_digests": {},
            }
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate Case"):
                review.read_review_ledger(path, expected_case_ids={"a"})

    def test_review_ledger_rejects_missing_case_rows(self) -> None:
        review = self._review_api()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing Case"):
                review.read_review_ledger(path, expected_case_ids={"a", "b"})

    def test_review_ledger_round_trip_preserves_release_order(self) -> None:
        review = self._review_api()
        decisions = [
            review.ReviewDecision(
                case_id="b",
                scene_id="collision_1d",
                anchor_decision="repair",
                tube_decision="needs_dense_review",
                evidence=("b-anchor.png", "b-dense.png"),
                reviewer="codex",
                notes="reversed numbering",
                issues=("collision_left_to_right_order",),
                repairs=("case_wide_identity_remap",),
                final_asset_digests={},
            ),
            review.ReviewDecision(
                case_id="a",
                scene_id="pendulum",
                anchor_decision="pass",
                tube_decision="pass",
                evidence=("a-anchor.png", "a-contact.png"),
                reviewer="codex",
                notes="identity and tube checked",
                issues=(),
                repairs=(),
                final_asset_digests={"manifest.json": "a" * 64},
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"

            review.write_review_ledger(
                path,
                decisions,
                expected_case_ids=("a", "b"),
            )
            loaded = review.read_review_ledger(
                path,
                expected_case_ids={"a", "b"},
            )

            self.assertEqual(["a", "b"], [value.case_id for value in loaded])
            self.assertEqual("repair", loaded[1].anchor_decision)
            self.assertEqual(("case_wide_identity_remap",), loaded[1].repairs)

    def test_review_decision_requires_visual_evidence_for_both_checks(self) -> None:
        review = self._review_api()

        with self.assertRaisesRegex(ValueError, "at least two evidence"):
            review.ReviewDecision(
                case_id="a",
                scene_id="pendulum",
                anchor_decision="pass",
                tube_decision="pass",
                evidence=("only-one.png",),
                reviewer="codex",
                notes="checked",
                issues=(),
                repairs=(),
                final_asset_digests={},
            )

    def test_anchor_sheet_distinguishes_existing_and_candidate_boundaries(self) -> None:
        visualization = self._visualization_api()
        image = np.full((32, 32, 3), 40, dtype=np.uint8)
        existing = np.zeros((32, 32), dtype=np.uint8)
        candidate = np.zeros((32, 32), dtype=np.uint8)
        existing[4:12, 4:12] = 1
        candidate[5:13, 5:13] = 1

        rendered = visualization.render_anchor_sheet(
            image,
            existing_masks={"object_1": existing},
            candidate_masks={"object_1": candidate},
            metadata={"case_id": "case_a", "scene_id": "pendulum"},
        )

        self.assertEqual(np.uint8, rendered.dtype)
        self.assertEqual(3, rendered.ndim)
        colors = {tuple(value) for value in rendered.reshape(-1, 3).tolist()}
        self.assertIn((0, 255, 0), colors)
        self.assertIn((255, 0, 255), colors)

    def test_dense_event_sheet_renders_requested_observation_indices(self) -> None:
        visualization = self._visualization_api()
        frames = [np.full((24, 32, 3), index * 20, np.uint8) for index in range(4)]
        masks = np.zeros((4, 24, 32), np.uint8)
        for index in range(4):
            masks[index, 6:12, 4 + index : 10 + index] = 1

        rendered = visualization.render_dense_event_sheet(
            frames,
            masks_by_object={"object_1": masks},
            states_by_object={"object_1": np.zeros(4, np.uint8)},
            observation_indices=(1, 3),
            source_indices=(10, 30),
            columns=2,
        )

        self.assertEqual(np.uint8, rendered.dtype)
        self.assertGreater(rendered.shape[0], 24)
        self.assertGreaterEqual(rendered.shape[1], 64)

    def test_render_review_queue_writes_real_case_anchor_evidence(self) -> None:
        script = ROOT / "scripts" / "reference_observations" / "render_review_queue.py"
        dataset = ROOT / "datasets" / "releases" / "14.0.0" / "dataset.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "anchor.png"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    str(dataset),
                    "--case-id",
                    "circular_r1_silver02cm_img_0370",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertGreater(rendered.shape[0], 1080)
            self.assertEqual(1080, rendered.shape[1])


if __name__ == "__main__":
    unittest.main()
