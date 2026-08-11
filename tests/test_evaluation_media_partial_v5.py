from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from physbench.baseline_runtime import build_i2v_media_contract
from physbench.evaluation.common import base, media
from physbench.evaluation.common.base import (
    ReferenceCaseEvaluator,
    SceneAnalysis,
)
from physbench.evaluation.common.media import (
    SampledVideo,
    VideoInfo,
    VideoProtocolError,
    sample_video,
)
from physbench.evaluation.contracts import CaseEvaluationRequest


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fail_at: int | None = None,
    ):
        self.frames = [frame.copy() for frame in frames]
        self.fail_at = fail_at
        self.position = 0
        self.read_calls = 0
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        index = self.position
        self.position += 1
        self.read_calls += 1
        if index == self.fail_at or index >= len(self.frames):
            return False, None
        return True, self.frames[index].copy()

    def release(self) -> None:
        self.released = True


def _info(*, frame_count: int = 3, fps: float = 10.0) -> VideoInfo:
    return VideoInfo(
        frame_count=frame_count,
        fps=fps,
        width=4,
        height=2,
        last_frame_time_s=(frame_count - 1) / fps,
    )


class PartialVideoSamplingTests(unittest.TestCase):
    def test_strict_default_preserves_insufficient_duration_error(
        self,
    ) -> None:
        with (
            patch.object(media, "probe_video", return_value=_info()),
            patch.object(media.cv2, "VideoCapture") as capture,
            self.assertRaises(VideoProtocolError) as raised,
        ):
            sample_video(
                Path("short.mp4"),
                sample_times_s=[0.0, 0.5],
                width=4,
                height=2,
            )
        self.assertEqual("insufficient_duration", raised.exception.code)
        capture.assert_not_called()

    def test_partial_random_seek_emits_neutral_unavailable_cells(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        capture = _FakeCapture(frames)
        with (
            patch.object(media, "probe_video", return_value=_info()),
            patch.object(
                media.cv2,
                "VideoCapture",
                return_value=capture,
            ),
        ):
            sampled = sample_video(
                Path("short.mp4"),
                sample_times_s=[0.0, 0.2, 0.3, 0.5],
                width=4,
                height=2,
                pad_value=7,
                allow_partial=True,
            )

        self.assertEqual([0, 2, 3, 5], sampled.source_indices)
        self.assertEqual([True, True, False, False], sampled.available)
        self.assertEqual(2, capture.read_calls)
        self.assertEqual(
            [
                (media.cv2.CAP_PROP_POS_FRAMES, 0.0),
                (media.cv2.CAP_PROP_POS_FRAMES, 2.0),
            ],
            capture.set_calls,
        )
        np.testing.assert_array_equal(frames[0], sampled.frames[0])
        np.testing.assert_array_equal(frames[2], sampled.frames[1])
        for frame in sampled.frames[2:]:
            np.testing.assert_array_equal(
                np.full((2, 4, 3), 7, dtype=np.uint8),
                frame,
            )
        self.assertIsNot(sampled.frames[2], sampled.frames[3])
        self.assertTrue(capture.released)

    def test_partial_does_not_hold_last_frame_past_video_duration(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        capture = _FakeCapture(frames)
        with (
            patch.object(media, "probe_video", return_value=_info()),
            patch.object(
                media.cv2,
                "VideoCapture",
                return_value=capture,
            ),
        ):
            sampled = sample_video(
                Path("short.mp4"),
                sample_times_s=[0.2, 0.219, 0.23],
                width=4,
                height=2,
                duration_tolerance_s=0.02,
                allow_partial=True,
            )

        # All three timestamps round to source frame 2.  The final timestamp
        # is nevertheless outside the permitted physical duration and must
        # not silently hold/copy that last source frame.
        self.assertEqual([2, 2, 2], sampled.source_indices)
        self.assertEqual([True, True, False], sampled.available)
        self.assertEqual(2, capture.read_calls)
        np.testing.assert_array_equal(frames[2], sampled.frames[0])
        np.testing.assert_array_equal(frames[2], sampled.frames[1])
        np.testing.assert_array_equal(
            np.zeros((2, 4, 3), dtype=np.uint8),
            sampled.frames[2],
        )

    def test_partial_forward_decode_never_walks_to_out_of_range_index(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        capture = _FakeCapture(frames)
        with (
            patch.object(media, "probe_video", return_value=_info()),
            patch.object(
                media.cv2,
                "VideoCapture",
                return_value=capture,
            ),
        ):
            sampled = sample_video(
                Path("short.mp4"),
                sample_times_s=[0.4, 0.1, 0.1, 0.3, 0.0],
                width=4,
                height=2,
                decode_policy="sequential_forward",
                allow_partial=True,
            )

        self.assertEqual([False, True, True, False, True], sampled.available)
        self.assertEqual(2, capture.read_calls)
        self.assertEqual([], capture.set_calls)
        self.assertIsNot(sampled.frames[1], sampled.frames[2])
        np.testing.assert_array_equal(frames[1], sampled.frames[1])
        np.testing.assert_array_equal(frames[0], sampled.frames[-1])

    def test_partial_forward_decode_failure_marks_remaining_cells_missing(
        self,
    ) -> None:
        frames = [
            np.full((2, 4, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40, 50)
        ]
        capture = _FakeCapture(frames, fail_at=1)
        with (
            patch.object(
                media,
                "probe_video",
                return_value=_info(frame_count=5),
            ),
            patch.object(
                media.cv2,
                "VideoCapture",
                return_value=capture,
            ),
        ):
            sampled = sample_video(
                Path("truncated.mp4"),
                sample_times_s=[0.0, 0.1, 0.3],
                width=4,
                height=2,
                decode_policy="sequential_forward",
                allow_partial=True,
            )
        self.assertEqual([True, False, False], sampled.available)
        self.assertEqual(2, capture.read_calls)
        self.assertTrue(capture.released)

    def test_full_sampling_always_reports_available_cells(self) -> None:
        frames = [
            np.full((2, 4, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        capture = _FakeCapture(frames)
        with (
            patch.object(media, "probe_video", return_value=_info()),
            patch.object(
                media.cv2,
                "VideoCapture",
                return_value=capture,
            ),
        ):
            sampled = sample_video(
                Path("complete.mp4"),
                sample_times_s=[0.0, 0.1, 0.2],
                width=4,
                height=2,
            )
        self.assertEqual([True, True, True], sampled.available)

    def test_sampled_video_validates_optional_availability_contract(
        self,
    ) -> None:
        kwargs = {
            "frames": [np.zeros((2, 4, 3), dtype=np.uint8)],
            "info": _info(),
            "sample_times_s": [0.0],
            "source_indices": [0],
            "spatial_transform": {},
        }
        self.assertIsNone(SampledVideo(**kwargs).available)
        with self.assertRaisesRegex(ValueError, "sampled frame"):
            SampledVideo(**kwargs, available=[])
        with self.assertRaisesRegex(ValueError, "boolean"):
            SampledVideo(**kwargs, available=[1])


class _SamplingEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "sampling_test"
    evaluator_version = "1"
    sequential_evaluator_version = "1"
    robust_evaluator_version = "1"
    scene_id = "collision_1d"
    primary_score = "sampling_score"

    def analyze(
        self,
        request,
        *,
        times_s,
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        return SceneAnalysis(
            score=0.5,
            metrics={"sampling_score": 0.5},
            quality={},
        )


class _PartialSamplingEvaluator(_SamplingEvaluator):
    allow_partial_prediction = True


class ReferenceEvaluatorPartialSamplingTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "timeline": {
                "fps": 2.0,
                "maximum_duration_s": 1.0,
                "minimum_duration_s": 0.1,
                "minimum_source_fps": 1.0,
            },
            "spatial": {"width": 4, "height": 2},
        }

    def test_only_opted_in_evaluator_allows_partial_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.mp4"
            prediction_path = root / "prediction.mp4"
            reference_path.touch()
            prediction_path.touch()
            request = CaseEvaluationRequest(
                job={"job_id": "job"},
                case={
                    "case_id": "case",
                    "scene_id": "collision_1d",
                },
                case_catalog={},
                prediction={
                    "status": "complete",
                    "video_path": str(prediction_path),
                },
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config={},
            )
            video_info = VideoInfo(2, 2.0, 4, 2, 0.5)
            reference_sample = SampledVideo(
                frames=[
                    np.zeros((2, 4, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                info=video_info,
                sample_times_s=[0.0, 0.5, 1.0],
                source_indices=[0, 1, 2],
                spatial_transform={},
                available=[True, True, True],
            )
            prediction_sample = SampledVideo(
                frames=[
                    np.zeros((2, 4, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                info=video_info,
                sample_times_s=[0.0, 0.5, 1.0],
                source_indices=[0, 1, 2],
                spatial_transform={},
                available=[True, True, False],
            )

            def run(evaluator_type):
                calls: list[dict] = []

                def fake_sample(_path, **kwargs):
                    calls.append(kwargs)
                    return (
                        reference_sample
                        if len(calls) == 1
                        else prediction_sample
                    )

                with (
                    patch.object(
                        base,
                        "resolve_physics_reference",
                        return_value=(reference_path, "same_case_gt", None),
                    ),
                    patch.object(
                        base,
                        "reference_timeline",
                        return_value=[0.0, 0.5, 1.0],
                    ),
                    patch.object(base, "sample_video", side_effect=fake_sample),
                ):
                    result = evaluator_type(self._config()).evaluate(request)
                return result, calls

            strict, strict_calls = run(_SamplingEvaluator)
            partial, partial_calls = run(_PartialSamplingEvaluator)

        self.assertNotIn("allow_partial", strict_calls[0])
        self.assertFalse(strict_calls[1]["allow_partial"])
        self.assertNotIn("allow_partial", partial_calls[0])
        self.assertTrue(partial_calls[1]["allow_partial"])
        self.assertEqual("evaluated", strict.status)
        self.assertEqual(1.0, strict.quality["temporal_coverage"])
        self.assertNotIn(
            "available",
            strict.provenance["sampling"]["prediction"],
        )
        self.assertEqual("evaluated", partial.status)
        self.assertAlmostEqual(0.75, partial.quality["temporal_coverage"])
        self.assertEqual(2, partial.quality["available_prediction_frames"])
        self.assertEqual(
            [True, True, False],
            partial.provenance["sampling"]["prediction"]["available"],
        )

    def test_overlap_duration_is_reported_but_does_not_scale_score(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.mp4"
            prediction_path = root / "prediction.mp4"
            reference_path.touch()
            prediction_path.touch()
            request = CaseEvaluationRequest(
                job={"job_id": "job"},
                case={
                    "case_id": "case",
                    "scene_id": "collision_1d",
                    "temporal": {"encoded_to_physical_speed": 1.0},
                },
                case_catalog={},
                prediction={
                    "status": "complete",
                    "video_path": str(prediction_path),
                },
                asset_root=root,
                artifact_dir=root / "artifacts",
                evaluator_config={},
            )
            reference_info = VideoInfo(21, 10.0, 4, 2, 2.0)
            prediction_info = VideoInfo(6, 10.0, 4, 2, 0.5)

            def fake_probe(path):
                return (
                    reference_info
                    if Path(path) == reference_path
                    else prediction_info
                )

            def fake_sample(path, **kwargs):
                times = list(kwargs["sample_times_s"])
                info = (
                    reference_info
                    if Path(path) == reference_path
                    else prediction_info
                )
                return SampledVideo(
                    frames=[
                        np.zeros((2, 4, 3), dtype=np.uint8)
                        for _ in times
                    ],
                    info=info,
                    sample_times_s=times,
                    source_indices=list(range(len(times))),
                    spatial_transform={},
                    available=[True] * len(times),
                )

            config = {
                "timeline": {
                    "policy": "physical_overlap_common_fps_v1",
                    "fps": 10.0,
                    "minimum_evaluation_fps": 1.0,
                    "minimum_source_fps": 1.0,
                },
                "spatial": {"width": 4, "height": 2},
            }
            with (
                patch.object(
                    base,
                    "resolve_physics_reference",
                    return_value=(
                        reference_path,
                        "same_case_gt",
                        None,
                    ),
                ),
                patch.object(base, "probe_video", side_effect=fake_probe),
                patch.object(base, "sample_video", side_effect=fake_sample),
            ):
                result = _SamplingEvaluator(config).evaluate(request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.5, result.score)
        self.assertEqual(0.25, result.quality["temporal_coverage"])
        self.assertEqual(
            0.5,
            result.quality["evaluated_physical_duration_s"],
        )
        self.assertFalse(result.quality["duration_mismatch_penalized"])


class ReferenceEvaluatorNoPadTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "evaluator_contract": "robust_subject_v3",
            "timeline": {
                "fps": 2.0,
                "maximum_duration_s": 1.0,
                "minimum_duration_s": 0.1,
                "minimum_source_fps": 1.0,
            },
            "spatial": {
                "width": 480,
                "height": 832,
                "policy": "shared_reference_content_no_pad_v1",
            },
        }

    @staticmethod
    def _contract() -> dict:
        return build_i2v_media_contract(
            conditioning_asset="first.png",
            width=480,
            height=832,
            temporal={"fps": 24, "num_frames": 121},
        )

    @staticmethod
    def _request(
        root: Path,
        *,
        contract: dict | None,
        has_real_reference: bool = True,
    ) -> CaseEvaluationRequest:
        reference = root / "reference.mp4"
        prediction = root / "prediction.mp4"
        condition = root / "first.png"
        reference.touch()
        prediction.touch()
        condition.touch()
        return CaseEvaluationRequest(
            job={"job_id": "job"},
            case={
                "case_id": "case",
                "scene_id": "collision_1d",
                "assets": {"first_frame": "first.png"},
                "has_real_reference_video": has_real_reference,
            },
            case_catalog={},
            prediction={
                "status": "complete",
                "video_path": str(prediction),
                **(
                    {"media_contract": contract}
                    if contract is not None
                    else {}
                ),
            },
            asset_root=root,
            artifact_dir=root / "artifacts",
            evaluator_config={},
        )

    def test_missing_contract_with_different_aspect_is_robust_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root, contract=None)
            infos = [
                VideoInfo(10, 10.0, 16, 9, 0.9),
                VideoInfo(10, 10.0, 4, 3, 0.9),
            ]
            with (
                patch.object(
                    base,
                    "resolve_physics_reference",
                    return_value=(
                        root / "reference.mp4",
                        "same_case_reference",
                        None,
                    ),
                ),
                patch.object(base, "reference_timeline", return_value=[0.0]),
                patch.object(base, "probe_video", side_effect=infos),
            ):
                result = _SamplingEvaluator(self._config()).evaluate(request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual(
            "prediction_media_contract_missing",
            result.reason_code,
        )

    def test_prediction_canvas_mismatch_is_robust_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root, contract=self._contract())
            infos = [
                VideoInfo(10, 10.0, 1080, 1920, 0.9),
                VideoInfo(10, 10.0, 832, 480, 0.9),
            ]
            with (
                patch.object(
                    base,
                    "resolve_physics_reference",
                    return_value=(
                        root / "reference.mp4",
                        "same_case_reference",
                        None,
                    ),
                ),
                patch.object(base, "reference_timeline", return_value=[0.0]),
                patch.object(base, "probe_video", side_effect=infos),
                patch.object(
                    base,
                    "probe_image_size",
                    return_value=(1080, 1920),
                ),
            ):
                result = _SamplingEvaluator(self._config()).evaluate(request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual(
            "prediction_canvas_mismatch",
            result.reason_code,
        )

    def test_unreadable_conditioning_asset_is_reference_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(root, contract=self._contract())
            with (
                patch.object(
                    base,
                    "resolve_physics_reference",
                    return_value=(
                        root / "reference.mp4",
                        "same_case_reference",
                        None,
                    ),
                ),
                patch.object(base, "reference_timeline", return_value=[0.0]),
                patch.object(
                    base,
                    "probe_video",
                    return_value=VideoInfo(10, 10.0, 1080, 1920, 0.9),
                ),
                patch.object(
                    base,
                    "probe_image_size",
                    side_effect=VideoProtocolError(
                        "conditioning_image_unreadable",
                        "fixture unreadable",
                    ),
                ),
            ):
                result = _SamplingEvaluator(self._config()).evaluate(request)

        self.assertEqual("unavailable", result.status)
        self.assertIsNone(result.score)
        self.assertEqual(
            "reference_conditioning_image_unreadable",
            result.reason_code,
        )

    def test_parent_reference_is_explicitly_rejected_by_no_pad_protocol(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self._request(
                root,
                contract=self._contract(),
                has_real_reference=False,
            )
            with (
                patch.object(
                    base,
                    "resolve_physics_reference",
                    return_value=(
                        root / "reference.mp4",
                        "parent_physics_reference",
                        "parent-case",
                    ),
                ),
                patch.object(base, "reference_timeline", return_value=[0.0]),
            ):
                result = _SamplingEvaluator(self._config()).evaluate(request)

        self.assertEqual("unavailable", result.status)
        self.assertIsNone(result.score)
        self.assertEqual(
            "reference_parent_media_contract_unsupported",
            result.reason_code,
        )


if __name__ == "__main__":
    unittest.main()
