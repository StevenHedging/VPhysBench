from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _script():
    path = ROOT / "scripts/reference_observations/rebuild_collision_sam31.py"
    spec = importlib.util.spec_from_file_location("rebuild_collision_sam31", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sam31GtCliTests(unittest.TestCase):
    def test_shards_cover_sorted_collision_cases_once(self) -> None:
        cli = _script()

        shards = cli.shard_case_ids(("case_c", "case_a", "case_d", "case_b"), 3)

        flattened = [case_id for shard in shards for case_id in shard]
        self.assertEqual(["case_a", "case_b", "case_c", "case_d"], sorted(flattened))
        self.assertEqual(4, len(set(flattened)))

    def test_collision_selection_rejects_requested_noncollision_case(self) -> None:
        cli = _script()
        cases = (
            SimpleNamespace(case_id="collision_a", scene_id="collision_1d"),
            SimpleNamespace(case_id="pendulum_a", scene_id="pendulum"),
        )

        with self.assertRaisesRegex(ValueError, "not collision_1d"):
            cli.select_collision_cases(cases, requested_ids={"pendulum_a"})

    def test_resume_validates_candidate_bytes_before_skipping(self) -> None:
        cli = _script()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            payload = root / "payload.bin"
            payload.write_bytes(b"valid")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "candidate_bundle.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "case_id": "case_a",
                        "required_targets": ["payload.bin"],
                        "files": [
                            {
                                "candidate_path": "payload.bin",
                                "target_path": "payload.bin",
                                "size_bytes": 5,
                                "sha256": digest,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(cli.candidate_is_complete(manifest, case_id="case_a"))
            payload.write_bytes(b"corrupt")
            self.assertFalse(cli.candidate_is_complete(manifest, case_id="case_a"))

    def test_decode_timeline_frames_preserves_repeated_source_indices(self) -> None:
        cli = _script()
        with tempfile.TemporaryDirectory() as value:
            video = Path(value) / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for intensity in (0, 60, 120):
                writer.write(np.full((16, 16, 3), intensity, np.uint8))
            writer.release()

            frames = cli.decode_timeline_frames(video, source_frame_indices=(0, 0, 2))

            self.assertEqual(3, len(frames))
            np.testing.assert_array_equal(frames[0], frames[1])
            self.assertLess(float(frames[1].mean()), float(frames[2].mean()))


if __name__ == "__main__":
    unittest.main()
