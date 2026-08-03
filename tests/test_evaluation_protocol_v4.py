from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from _paths import ROOT
from physbench.evaluation.common.masks.sam2 import (
    MaskPrompt,
    Sam2VideoSegmenter,
)
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.collision.observation import (
    build_multiframe_collision_prompts,
)
from physbench.evaluation.scenes.collision.scoring import (
    extract_collision_trace,
    score_collision,
)
from physbench.evaluation.scenes.collision.visualization import (
    write_collision_visualization,
)
from physbench.io import canonical_sha256


class _TensorLogits:
    def __init__(self, value: torch.Tensor):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.numpy()


class _FakePredictor:
    def __init__(self, frame_count: int, height: int, width: int):
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.seed = 0

    def init_state(self, **_):
        return {}

    def reset_state(self, _):
        return None

    def add_new_points_or_box(self, *, frame_idx, **_):
        self.seed = int(frame_idx)

    def _logits(self):
        value = torch.full(
            (3, 1, self.height, self.width), -1.0, dtype=torch.float32
        )
        value[0, 0, 2:6, 2:7] = 2.0
        value[1, 0, 2:6, 6:11] = 3.0
        value[2, 0, 2:6, 10:15] = 4.0
        return _TensorLogits(value)

    def propagate_in_video(
        self, _, *, start_frame_idx=None, reverse=False
    ):
        seed = self.seed if start_frame_idx is None else int(start_frame_idx)
        indices = (
            range(seed, -1, -1)
            if reverse
            else range(seed, self.frame_count)
        )
        for frame_index in indices:
            yield frame_index, [1, 2, 3], self._logits()


class EvaluationProtocolV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = load_evaluation_protocol("scene_default_v3")
        cls.v4 = load_evaluation_protocol("scene_default_v4")

    def test_v4_isolated_collision_upgrade_preserves_v3(self) -> None:
        self.assertEqual("collision_1d_state_v2", self.v3["scenes"][
            "collision_1d"
        ]["type"])
        self.assertEqual("collision_1d_state_v3", self.v4["scenes"][
            "collision_1d"
        ]["type"])
        self.assertNotEqual(self.v3["fingerprint"], self.v4["fingerprint"])
        v3_collision = SceneEvaluatorRegistry(self.v3).resolve(
            "collision_1d"
        ).describe()
        v4_registry = SceneEvaluatorRegistry(self.v4)
        self.assertEqual("1.3", v3_collision["version"])
        self.assertEqual(
            "1.4", v4_registry.resolve("collision_1d").describe()["version"]
        )
        for scene_id in (
            "pendulum",
            "free_fall",
            "inclined_plane_slide",
            "uniform_circular_motion",
        ):
            self.assertEqual(
                self.v3["scenes"][scene_id],
                self.v4["scenes"][scene_id],
            )
            self.assertEqual(
                "1.3", v4_registry.resolve(scene_id).describe()["version"]
            )

    def test_schema_lists_collision_v3(self) -> None:
        schema = json.loads(
            (
                ROOT / "schemas" / "v2" / "evaluation_protocol.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn(
            "collision_1d_state_v3",
            schema["$defs"]["referenceScene"]["properties"]["type"]["enum"],
        )

    @staticmethod
    def _draw_ball(
        frame: np.ndarray, center: tuple[int, int], radius: int
    ) -> None:
        cv2.circle(frame, center, radius, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(
            frame, center, radius - 2, (105, 105, 105), -1, cv2.LINE_AA
        )
        cv2.circle(
            frame,
            (center[0] - radius // 3, center[1] - radius // 3),
            max(2, radius // 4),
            (245, 245, 245),
            -1,
            cv2.LINE_AA,
        )

    def test_multiframe_seed_does_not_require_frame_zero_striker(self) -> None:
        config = self.v4["scenes"]["collision_1d"][
            "multi_frame_observation"
        ]
        frames = []
        for frame_index in range(16):
            frame = np.full((540, 960, 3), 180, dtype=np.uint8)
            cv2.line(frame, (0, 500), (959, 500), (80, 80, 80), 3)
            if frame_index > 0:
                self._draw_ball(
                    frame, (30 + 24 * frame_index, 455), 20
                )
            self._draw_ball(frame, (520, 455), 18)
            self._draw_ball(frame, (558, 455), 16)
            frames.append(frame)
        prompts, metadata = build_multiframe_collision_prompts(
            frames, config=config
        )
        self.assertGreater(metadata["seed_frame"], 0)
        self.assertEqual(
            ["striker", "target_1", "target_2"],
            [prompt.metadata["role"] for prompt in prompts],
        )
        self.assertTrue(
            all(
                prompt.frame_index == metadata["seed_frame"]
                for prompt in prompts
            )
        )
        self.assertEqual(
            sorted(prompt.points_xy[0, 0] for prompt in prompts),
            [prompt.points_xy[0, 0] for prompt in prompts],
        )

    def test_multi_object_sam_uses_reverse_and_exclusive_masks(self) -> None:
        frames = [
            np.zeros((12, 18, 3), dtype=np.uint8) for _ in range(5)
        ]
        prompts = [
            MaskPrompt(
                frame_index=2,
                box_xyxy=np.asarray([1, 1, 8, 8], dtype=np.float32),
                points_xy=np.asarray([[4, 4]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                metadata={"role": role},
            )
            for role in ("striker", "target_1", "target_2")
        ]
        segmenter = Sam2VideoSegmenter(
            {"model_id": "unused", "device": "cpu"}
        )
        segmenter._predictor = _FakePredictor(5, 12, 18)
        segmenter._torch = SimpleNamespace(
            inference_mode=lambda: nullcontext()
        )
        segmenter.device = "cpu"
        masks, metadata = segmenter.segment_instances(
            frames, prompts=prompts, exclusive_masks=True
        )
        self.assertEqual(2, metadata["seed_frame"])
        self.assertEqual(3, metadata["forward_frames"])
        self.assertEqual(3, metadata["reverse_frames"])
        for frame_index in range(5):
            total = sum(
                (masks[object_index][frame_index] > 0).astype(np.uint8)
                for object_index in range(3)
            )
            self.assertLessEqual(int(total.max()), 1)

    def test_legacy_frame_zero_masks_remain_independently_thresholded(
        self,
    ) -> None:
        frames = [
            np.zeros((12, 18, 3), dtype=np.uint8) for _ in range(3)
        ]
        prompts = [
            MaskPrompt(
                frame_index=0,
                box_xyxy=np.asarray([1, 1, 8, 8], dtype=np.float32),
                points_xy=np.asarray([[4, 4]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                metadata={"role": role},
            )
            for role in ("striker", "target_1", "target_2")
        ]
        segmenter = Sam2VideoSegmenter(
            {"model_id": "unused", "device": "cpu"}
        )
        segmenter._predictor = _FakePredictor(3, 12, 18)
        segmenter._torch = SimpleNamespace(
            inference_mode=lambda: nullcontext()
        )
        segmenter.device = "cpu"
        masks, metadata = segmenter.segment_instances(
            frames, prompts=prompts
        )
        self.assertNotIn("overlap_policy", metadata)
        self.assertNotIn("reverse_frames", metadata)
        total = sum(
            (masks[object_index][0] > 0).astype(np.uint8)
            for object_index in range(3)
        )
        self.assertGreater(int(total.max()), 1)

    def test_semantic_role_swap_is_not_an_identity_match(self) -> None:
        times = np.arange(33, dtype=np.float64) / 16.0
        before = np.minimum(times, 1.0)
        after = np.maximum(times - 1.0, 0.0)
        scalar = np.column_stack(
            [
                10.0 + 40.0 * before,
                55.0 + 8.0 * after,
                65.0 + 40.0 * after,
            ]
        )
        xy = np.stack([scalar, np.full_like(scalar, 30.0)], axis=2)
        trace = extract_collision_trace(
            xy,
            np.ones((len(times), 3), dtype=bool),
            times.tolist(),
            masses_kg=np.ones(3),
            minimum_span_px=10.0,
            velocity_window_fraction=0.2,
        )
        swapped = replace(
            trace,
            normalized_position=trace.normalized_position[:, [0, 2, 1]],
            pre_velocity_normalized_s=(
                trace.pre_velocity_normalized_s[[0, 2, 1]]
            ),
            post_velocity_normalized_s=(
                trace.post_velocity_normalized_s[[0, 2, 1]]
            ),
        )
        result = score_collision(
            trace,
            swapped,
            config=self.v4["scenes"]["collision_1d"]["scoring"],
        )
        self.assertLess(result["score"], 0.95)
        self.assertLess(
            result["components"]["instance_trajectories"], 1.0
        )

    def test_visualization_is_run_owned_and_locally_manifested(self) -> None:
        times = [0.0, 0.1, 0.2, 0.3]
        frames = [
            np.full((100, 160, 3), 120, dtype=np.uint8)
            for _ in times
        ]
        instance_masks = [
            [np.zeros((100, 160), dtype=np.uint8) for _ in times]
            for _ in range(3)
        ]
        xy = np.zeros((len(times), 3, 2), dtype=np.float64)
        for frame_index in range(len(times)):
            for object_index in range(3):
                x = 24 + 32 * object_index + 4 * frame_index
                y = 60
                cv2.circle(
                    instance_masks[object_index][frame_index],
                    (x, y),
                    5,
                    255,
                    -1,
                )
                xy[frame_index, object_index] = [x, y]
        union = []
        for frame_index in range(len(times)):
            value = np.zeros((100, 160), dtype=np.uint8)
            for object_index in range(3):
                value = cv2.bitwise_or(
                    value, instance_masks[object_index][frame_index]
                )
            union.append(value)
        normalized = xy[:, :, 0] / 160.0
        rows = [
            {"frame": index, "time_s": time_s, "physical_subject_iou": 1.0}
            for index, time_s in enumerate(times)
        ]
        observation = {
            "prompt_builder": {
                "seed_frame": 1,
                "seed_source": "synthetic",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "sealed" / "case",
                case={"case_id": "collision_case"},
                job={"job_id": "job_collision_case"},
                evaluator_config={"type": "collision_1d_state_v3"},
                run_id="collision_v4_test_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            request.artifact_dir.mkdir(parents=True)
            artifacts = write_collision_visualization(
                request,
                config={
                    "enabled": True,
                    "codec": "h264",
                    "fps": 10.0,
                    "panel_width": 160,
                    "panel_height": 100,
                },
                times_s=times,
                reference_frames=frames,
                prediction_frames=frames,
                reference_masks=instance_masks,
                prediction_masks=instance_masks,
                reference_union=union,
                prediction_union=union,
                reference_xy=xy,
                prediction_xy=xy,
                reference_normalized=normalized,
                prediction_normalized=normalized,
                union_ious=[1.0] * len(times),
                instance_ious=[[1.0, 1.0, 1.0] for _ in times],
                rows=rows,
                reference_observation=observation,
                prediction_observation=observation,
                reference_event_frame=2,
                prediction_event_frame=2,
            )
            manifest_path = Path(
                artifacts["collision_visualization_manifest"]
            )
            self.assertTrue(manifest_path.is_relative_to(request.artifact_dir))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(
                canonical_sha256(request.evaluator_config),
                manifest["evaluator_config_sha256"],
            )
            self.assertFalse(manifest_path.is_symlink())
            for record in manifest["files"].values():
                path = Path(record["path"])
                self.assertTrue(path.is_file())
                self.assertTrue(
                    path.is_relative_to(
                        root / "run" / "evaluation" / "visualizations"
                    )
                )
                self.assertGreater(record["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
