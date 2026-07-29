from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from _paths import ROOT
from physbench.data_layout import V4_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.media import (
    VideoInfo,
    VideoProtocolError,
    probe_video,
    sample_video,
)
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.pendulum.evaluator import (
    _timeline_provenance,
    build_pendulum_timeline,
)


PROTOCOL_ROOT = ROOT / "configs" / "evaluation" / "protocols"
V1_RAW_SHA256 = "b3712e0930c3a729c65a8774c2984c407cf3559b52030b0403fcc7468c8d37fc"
V1_CANONICAL_FINGERPRINT = (
    "7e40b69368a60282109e4e34fc99d380b186b5e65e537e113e601a3a066fa352"
)
V1_EVALUATOR_FINGERPRINTS = {
    "pendulum": "ffebe28bfe88ccfd83484ebdded2ab8446f0e4c56be17677d984156dd2aacddf",
    "collision_1d": "d276255b037ec7a96c1dd7a7b95ebef42c36e5b9b082ace5b4ce6ca2e7556a85",
    "free_fall": "2056e3479d08c33c56ba8e61b640745a26f72361225b562796a3fdffeba71d7a",
    "inclined_plane_slide": "00b672e45e57470680853110fa4ef8b4d938032ccd31946d1c0006b60f9c88d1",
    "uniform_circular_motion": "9a60d9272a6efa2a022de89a6fdd1d249876c9de6da5ebf3ddc567c426a1caba",
}


class EvaluationProtocolV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_evaluation_protocol("scene_default_v1")
        cls.v2 = load_evaluation_protocol("scene_default_v2")

    def test_v1_bytes_and_fingerprint_are_frozen(self) -> None:
        path = PROTOCOL_ROOT / "scene_default_v1.json"
        self.assertEqual(
            V1_RAW_SHA256,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(V1_CANONICAL_FINGERPRINT, self.v1["fingerprint"])

    def test_v2_changes_timeline_and_explicitly_pins_forward_decode(self) -> None:
        self.assertEqual("scene_default_v2", self.v2["protocol_id"])
        for scene_id, config in self.v1["scenes"].items():
            if scene_id != "pendulum":
                comparable = deepcopy(self.v2["scenes"][scene_id])
                self.assertEqual(
                    "sequential_forward",
                    comparable["timeline"].pop("decode_policy"),
                )
                self.assertEqual(config, comparable)
        pendulum = self.v2["scenes"]["pendulum"]
        self.assertEqual("pendulum_state_v2", pendulum["type"])
        self.assertEqual(
            {
                "fps": 16.0,
                "maximum_duration_s": 5.0,
                "minimum_duration_s": 4.8,
                "minimum_source_fps": 8.0,
                "duration_tolerance_s": 0.02,
                "decode_policy": "sequential_forward",
            },
            pendulum["timeline"],
        )

    def test_protocol_schema_preserves_v1_v2_and_accepts_v3(self) -> None:
        schema = json.loads(
            (
                ROOT
                / "schemas"
                / "v2"
                / "evaluation_protocol.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [
                "pendulum_state_v1",
                "pendulum_state_v2",
                "pendulum_state_v3",
            ],
            schema["$defs"]["pendulum"]["properties"]["type"]["enum"],
        )
        branches = schema["$defs"]["pendulum"]["oneOf"]
        self.assertEqual(
            {
                "pendulum_state_v1",
                "pendulum_state_v2",
                "pendulum_state_v3",
            },
            {
                branch["properties"]["type"]["const"]
                for branch in branches
            },
        )

    def test_official_tasks_pin_v3_with_new_identity(self) -> None:
        expected_ids = {
            "five_scene_finetune_eval.json": "five_scene_finetune_eval_v6",
            "five_scene_direct_eval.json": "five_scene_direct_eval_v6",
        }
        for name, expected_id in expected_ids.items():
            task = json.loads(
                (
                    ROOT / "tasks" / "official" / name
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                "scene_default_v3",
                task["evaluation"]["protocol"],
            )
            self.assertEqual(expected_id, task["task_id"])

    def test_registry_freezes_v1_identity_and_separates_v2(self) -> None:
        v1_registry = SceneEvaluatorRegistry(self.v1)
        v2_registry = SceneEvaluatorRegistry(self.v2)
        for scene_id, expected in V1_EVALUATOR_FINGERPRINTS.items():
            legacy = v1_registry.resolve(scene_id).describe()
            sequential = v2_registry.resolve(scene_id).describe()
            self.assertEqual("1.1", legacy["version"])
            self.assertEqual(expected, legacy["fingerprint"])
            self.assertEqual("1.2", sequential["version"])
            self.assertNotEqual(legacy["fingerprint"], sequential["fingerprint"])

    def test_v1_timeline_remains_fixed_at_five_seconds(self) -> None:
        with patch(
            "physbench.evaluation.scenes.pendulum.evaluator.reference_timeline"
        ) as bounded:
            times, policy = build_pendulum_timeline(
                Path("unused.mp4"),
                self.v1["scenes"]["pendulum"],
            )
        bounded.assert_not_called()
        self.assertEqual("fixed_duration_legacy", policy)
        self.assertEqual(81, len(times))
        self.assertEqual(0.0, times[0])
        self.assertEqual(5.0, times[-1])

    def test_v1_serialized_timeline_provenance_is_frozen(self) -> None:
        legacy = _timeline_provenance(
            self.v1["scenes"]["pendulum"],
            fps=16.0,
            duration_s=5.0,
            frame_count=81,
            policy="fixed_duration_legacy",
        )
        self.assertEqual(
            {
                "fps": 16.0,
                "duration_s": 5.0,
                "frame_count": 81,
                "prediction_frame_zero_injected": False,
            },
            legacy,
        )
        sequential = _timeline_provenance(
            self.v2["scenes"]["pendulum"],
            fps=16.0,
            duration_s=4.8125,
            frame_count=78,
            policy="case_reference_bounded",
        )
        self.assertEqual(
            {
                "fps": 16.0,
                "duration_s": 4.8125,
                "frame_count": 78,
                "policy": "case_reference_bounded",
                "prediction_frame_zero_injected": False,
            },
            sequential,
        )

    def test_v2_rejects_too_short_reference(self) -> None:
        info = VideoInfo(576, 120.0, 480, 832, 4.79)
        with (
            patch(
                "physbench.evaluation.common.media.probe_video",
                return_value=info,
            ),
            self.assertRaises(VideoProtocolError) as raised,
        ):
            build_pendulum_timeline(
                Path("too-short.mp4"),
                self.v2["scenes"]["pendulum"],
            )
        self.assertEqual("reference_too_short", raised.exception.code)

    def test_v2_accepts_short_valid_reference_and_bounds_to_sample_grid(
        self,
    ) -> None:
        info = VideoInfo(584, 120.0, 480, 832, 4.858333333333333)
        with patch(
            "physbench.evaluation.common.media.probe_video",
            return_value=info,
        ):
            times, policy = build_pendulum_timeline(
                Path("short-valid.mp4"),
                self.v2["scenes"]["pendulum"],
            )
        self.assertEqual("case_reference_bounded", policy)
        self.assertEqual(78, len(times))
        self.assertEqual(4.8125, times[-1])
        self.assertLessEqual(times[-1], info.last_frame_time_s)

    def test_v2_caps_long_reference_at_five_seconds(self) -> None:
        info = VideoInfo(721, 120.0, 480, 832, 6.0)
        with patch(
            "physbench.evaluation.common.media.probe_video",
            return_value=info,
        ):
            times, policy = build_pendulum_timeline(
                Path("long.mp4"),
                self.v2["scenes"]["pendulum"],
            )
        self.assertEqual("case_reference_bounded", policy)
        self.assertEqual(81, len(times))
        self.assertEqual(5.0, times[-1])

    def test_frozen_dataset_pendulum_references_pass_v2_preflight(
        self,
    ) -> None:
        dataset = load_dataset(V4_DATASET)
        pendulum_root = dataset.asset_root / "assets" / "pendulum"
        if not pendulum_root.is_dir():
            self.skipTest("canonical pendulum assets are not materialized")
        config = self.v2["scenes"]["pendulum"]
        paths = sorted({
            dataset.asset_root / case["assets"]["physics_reference_video"]
            for case in dataset.cases
            if case["scene_id"] == "pendulum"
        })
        self.assertTrue(paths)
        shortest: tuple[float, Path] | None = None
        for path in paths:
            with self.subTest(path=path):
                info = probe_video(path)
                shortest = min(
                    shortest or (info.last_frame_time_s, path),
                    (info.last_frame_time_s, path),
                )
                self.assertGreaterEqual(info.last_frame_time_s, 4.8)
                times, policy = build_pendulum_timeline(path, config)
                self.assertEqual("case_reference_bounded", policy)
                self.assertLessEqual(times[-1], 5.0)
                self.assertLessEqual(times[-1], info.last_frame_time_s)
        assert shortest is not None
        self.assertAlmostEqual(4.858333333333333, shortest[0], places=6)
        self.assertIn(
            "pendulum_r2_ltot0130mm_lrope0120mm_r010mm_a020deg",
            str(shortest[1]),
        )

    def test_real_hevc_keeps_legacy_and_forward_decode_identities_distinct(
        self,
    ) -> None:
        dataset = load_dataset(V4_DATASET)
        case = next(
            item
            for item in dataset.cases
            if item["case_id"]
            == (
                "collision_r2_glass_marble_glass_marble_"
                "glass_marble_v04374"
            )
        )
        path = (
            dataset.asset_root
            / case["assets"]["physics_reference_video"]
        )
        if not path.is_file():
            self.skipTest("canonical HEVC asset is not materialized")
        capture = cv2.VideoCapture(str(path))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((fourcc >> (8 * index)) & 0xFF)
            for index in range(4)
        )
        capture.release()
        self.assertEqual("hevc", codec)

        info = probe_video(path)
        kwargs = {
            "sample_times_s": [3.625],
            "width": info.width,
            "height": info.height,
        }
        legacy = sample_video(
            path,
            **kwargs,
            decode_policy="legacy_random_seek",
        )
        forward = sample_video(
            path,
            **kwargs,
            decode_policy="sequential_forward",
        )
        self.assertEqual([771], legacy.source_indices)
        self.assertEqual(legacy.source_indices, forward.source_indices)
        self.assertFalse(
            np.array_equal(legacy.frames[0], forward.frames[0]),
            "the frozen HEVC fixture is expected to expose long-GOP seek "
            "semantics; if the decoder changes, bump the evaluator identity "
            "and update this regression deliberately",
        )


if __name__ == "__main__":
    unittest.main()
