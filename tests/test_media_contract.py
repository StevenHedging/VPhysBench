from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from physbench.baseline_runtime.media_contract import (
    MediaContractError,
    build_i2v_media_contract,
    materialize_i2v_conditioning,
    plan_generation_timeline,
    validate_prediction_video,
)
from physbench.baseline_runtime.plugin import (
    _audit_prediction_media_contract,
)
from physbench.evaluation.task_evaluator import evaluate_task


def _ffmpeg(*arguments: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments],
        check=True,
    )


class ManagedI2VMediaContractTests(unittest.TestCase):
    def test_bounded_timeline_uses_shortest_legal_sequence_covering_target(
        self,
    ) -> None:
        plan = plan_generation_timeline(
            {
                "fps": 16,
                "min_frames": 5,
                "max_frames": 81,
                "valid_frame_rule": "4n+1",
            },
            target_physical_duration_s=0.325,
        )

        self.assertEqual(9, plan["requested_num_frames"])
        self.assertEqual(0.5, plan["requested_physical_duration_s"])
        self.assertEqual(
            "covers_target_with_native_tail",
            plan["duration_alignment"],
        )

    def test_fixed_timeline_keeps_model_native_length(self) -> None:
        plan = plan_generation_timeline(
            {"fps": 24, "num_frames": 121},
            target_physical_duration_s=0.5,
        )

        self.assertEqual(121, plan["requested_num_frames"])
        self.assertEqual(5.0, plan["requested_physical_duration_s"])
        self.assertEqual(
            "covers_target_with_native_tail",
            plan["duration_alignment"],
        )

    def test_duration_aware_contract_seals_requested_frame_count(self) -> None:
        contract = build_i2v_media_contract(
            conditioning_asset="first.png",
            width=64,
            height=64,
            temporal={
                "fps": 16,
                "min_frames": 5,
                "max_frames": 81,
                "valid_frame_rule": "4n+1",
            },
            target_physical_duration_s=0.325,
        )

        self.assertEqual("1.1", contract["schema_version"])
        self.assertEqual(
            {"rule": "fixed", "value": 9},
            contract["output"]["timeline"]["frame_count"],
        )
        self.assertEqual(
            0.325,
            contract["evaluation"]["target_physical_duration_s"],
        )

        tampered = copy.deepcopy(contract)
        tampered["evaluation"]["duration_alignment"] = "exact"
        with self.assertRaises(MediaContractError):
            validate_prediction_video(Path("missing.mp4"), tampered)

    def test_temporal_contract_has_one_supported_frame_count_rule(self) -> None:
        with self.assertRaises(MediaContractError):
            build_i2v_media_contract(
                conditioning_asset="first.png",
                width=64,
                height=64,
                temporal={
                    "fps": 8,
                    "num_frames": 9,
                    "max_frames": 17,
                },
            )
        with self.assertRaises(MediaContractError):
            build_i2v_media_contract(
                conditioning_asset="first.png",
                width=64,
                height=64,
                temporal={
                    "fps": 8,
                    "num_frames": 9,
                    "valid_frame_rule": "odd",
                },
            )

    def test_conditioning_materialization_preserves_full_exact_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "conditioning.png"
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=1920x1080",
                "-frames:v",
                "1",
                str(source),
            )
            contract = build_i2v_media_contract(
                conditioning_asset="source.png",
                width=832,
                height=480,
                temporal={
                    "fps": 24,
                    "num_frames": 121,
                    "valid_frame_rule": "4n+1",
                },
            )

            audit = materialize_i2v_conditioning(
                source,
                output,
                contract,
            )

            self.assertEqual([0, 6, 832, 468], audit["content_rect_xywh"])
            self.assertEqual(832, audit["output_probe"]["width"])
            self.assertEqual(480, audit["output_probe"]["height"])

    def test_prediction_video_must_match_canvas_fps_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.mp4"
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=64x64:rate=8",
                "-frames:v",
                "9",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "8",
                str(prediction),
            )
            contract = build_i2v_media_contract(
                conditioning_asset="first.png",
                width=64,
                height=64,
                temporal={
                    "fps": 8,
                    "num_frames": 9,
                    "valid_frame_rule": "4n+1",
                },
            )

            audit = validate_prediction_video(prediction, contract)

            self.assertEqual("valid", audit["status"])
            self.assertEqual(9, audit["probe"]["frames"])
            self.assertEqual(0.0, audit["probe"]["start_time_s"])

            contract["output"]["canvas"]["width"] = 96
            with self.assertRaises(MediaContractError) as raised:
                validate_prediction_video(prediction, contract)
            self.assertEqual("prediction_canvas_mismatch", raised.exception.code)

    def test_prediction_timeline_must_start_at_physical_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary) / "offset.mp4"
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=64x64:rate=8",
                "-frames:v",
                "9",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-output_ts_offset",
                "1",
                str(prediction),
            )
            contract = build_i2v_media_contract(
                conditioning_asset="first.png",
                width=64,
                height=64,
                temporal={"fps": 8, "num_frames": 9},
            )

            with self.assertRaises(MediaContractError) as raised:
                validate_prediction_video(prediction, contract)

            self.assertEqual(
                "prediction_start_time_mismatch",
                raised.exception.code,
            )

    def test_bounded_frame_rule_accepts_only_4n_plus_1(self) -> None:
        contract = build_i2v_media_contract(
            conditioning_asset="first.png",
            width=64,
            height=64,
            temporal={
                "fps": 24,
                "min_frames": 5,
                "max_frames": 121,
                "valid_frame_rule": "4n+1",
            },
        )
        self.assertEqual(
            {
                "rule": "bounded",
                "minimum": 5,
                "maximum": 121,
                "modulus": 4,
                "remainder": 1,
            },
            contract["output"]["timeline"]["frame_count"],
        )

    def test_runtime_marks_malformed_output_as_protocol_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prediction_path = Path(temporary) / "prediction.mp4"
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=64x64:rate=8",
                "-frames:v",
                "9",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(prediction_path),
            )
            prediction = {
                "status": "complete",
                "video_path": str(prediction_path),
                "media_contract": build_i2v_media_contract(
                    conditioning_asset="first.png",
                    width=96,
                    height=64,
                    temporal={"fps": 8, "num_frames": 9},
                ),
            }

            _audit_prediction_media_contract([prediction])

            self.assertEqual("protocol_error", prediction["status"])
            self.assertEqual(
                "prediction_canvas_mismatch",
                prediction["protocol_error"]["code"],
            )

    def test_task_preflight_stops_before_scene_evaluator(self) -> None:
        class RejectSceneEvaluationRegistry:
            def resolve(self, scene_id):
                raise AssertionError("scene evaluator must not run")

            def describe(self):
                return {}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "prediction.mp4"
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=64x64:rate=8",
                "-frames:v",
                "9",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(prediction_path),
            )
            job = {
                "job_id": "job",
                "case_id": "case",
                "scene_id": "pendulum",
                "evaluation_partition": "test",
                "seed": 7,
            }
            prediction = {
                **job,
                "baseline_id": "fixture",
                "status": "complete",
                "video_path": str(prediction_path),
                "media_contract": build_i2v_media_contract(
                    conditioning_asset="first.png",
                    width=96,
                    height=64,
                    temporal={"fps": 8, "num_frames": 9},
                ),
            }
            results, summary = evaluate_task(
                plan={
                    "task_id": "fixture",
                    "family": "direct_eval",
                    "scene_ids": ["pendulum"],
                    "jobs": [job],
                },
                cases=[{"case_id": "case", "scene_id": "pendulum"}],
                predictions=[prediction],
                asset_root=root,
                protocol={
                    "protocol_id": "fixture",
                    "fingerprint": "f" * 64,
                    "path": "fixture.json",
                    "scenes": {"pendulum": {"type": "fixture"}},
                },
                output_dir=root / "evaluation",
                registry=RejectSceneEvaluationRegistry(),
            )

            self.assertEqual("protocol_error", results[0]["status"])
            self.assertEqual(
                "prediction_canvas_mismatch",
                results[0]["reason_code"],
            )
            self.assertEqual(
                {"protocol_error": 1},
                summary["status_counts"],
            )


if __name__ == "__main__":
    unittest.main()
