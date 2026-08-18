from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / "reference_observations" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceObservationCurationCliTests(unittest.TestCase):
    def test_gpu_shards_cover_each_case_exactly_once(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        shards = rebuild.shard_case_ids(["c", "a", "b", "d"], shard_count=3)
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(["a", "b", "c", "d"], sorted(flattened))
        self.assertEqual(4, len(set(flattened)))

    def test_install_authorization_requires_accepted_matching_candidate(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "anchor_decision": "pass",
                        "tube_decision": "repair",
                        "candidate_digest": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "accepted review decision"):
                rebuild.require_install_authorization(
                    ledger, case_id="case_a", candidate_manifest=candidate
                )

    def test_install_authorization_rejects_stale_candidate_digest(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text("{}", encoding="utf-8")
            ledger = root / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "case_id": "case_a",
                        "anchor_decision": "pass",
                        "tube_decision": "pass",
                        "candidate_digest": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                rebuild.require_install_authorization(
                    ledger, case_id="case_a", candidate_manifest=candidate
                )

    def test_audit_records_are_written_in_release_order(self) -> None:
        audit = _load_script("audit_v14.py")
        self.assertEqual((0, 3, 6, 9), audit.shard_release_indices(10, 0, 3))
        self.assertEqual((1, 4, 7), audit.shard_release_indices(10, 1, 3))
        self.assertEqual((2, 5, 8), audit.shard_release_indices(10, 2, 3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostics.jsonl"
            audit.write_diagnostic_records(
                output,
                [
                    {"case_id": "case_b", "release_index": 1},
                    {"case_id": "case_a", "release_index": 0},
                ],
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(["case_a", "case_b"], [row["case_id"] for row in rows])

    def test_rebuild_decodes_only_timeline_source_frames(self) -> None:
        rebuild = _load_script("rebuild_cases.py")
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "fixture.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 40, 80, 120, 160):
                writer.write(np.full((16, 16, 3), value, np.uint8))
            writer.release()
            frames = rebuild._decode_video(video, source_frame_indices=(0, 2, 4))
            self.assertEqual(3, len(frames))
            self.assertLess(float(frames[0].mean()), float(frames[1].mean()))
            self.assertLess(float(frames[1].mean()), float(frames[2].mean()))

    def test_audit_evidence_decode_does_not_depend_on_random_seek(self) -> None:
        audit = _load_script("audit_v14.py")

        class SequentialCapture:
            def __init__(self, _path: str):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, _property):
                return 5

            def set(self, _property, _value):
                raise AssertionError("random seeking is not reliable")

            def read(self):
                if self.index == 5:
                    return False, None
                frame = np.full((8, 8, 3), self.index, np.uint8)
                self.index += 1
                return True, frame

            def release(self):
                return None

        with patch.object(audit.cv2, "VideoCapture", SequentialCapture):
            frames = audit._spaced_video_frames(Path("fixture.mp4"), count=3)
        self.assertEqual([0, 2, 4], [int(frame[0, 0, 0]) for frame in frames])


if __name__ == "__main__":
    unittest.main()
