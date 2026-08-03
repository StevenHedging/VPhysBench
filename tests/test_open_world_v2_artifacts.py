from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.artifacts import open_world_v2 as visual
from physbench.evaluation.common.entities.contracts import (
    EntitySpec,
    LifecyclePolicy,
    ReferenceCapability,
)
from physbench.evaluation.common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
)
from physbench.evaluation.common.entities.timeline import (
    build_common_time_grid,
)
from physbench.evaluation.common.entities.v2 import (
    ExpectedEntityTimeline,
    compare_open_world_v2,
)


def _mask(x: int, y: int) -> np.ndarray:
    output = np.zeros((48, 64), dtype=np.uint8)
    output[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = 255
    return output


def _timeline() -> ExpectedEntityTimeline:
    entity = EntitySpec(
        entity_id="body_0",
        role_id="subject",
        entity_class="rigid_body",
        lifecycle=LifecyclePolicy.PERSISTENT,
    )
    xy = np.asarray(
        [[10.0, 20.0], [16.0, 20.0], [22.0, 20.0]],
        dtype=np.float64,
    )
    return ExpectedEntityTimeline(
        entity=entity,
        capability=ReferenceCapability.SAME_CASE_GT,
        expected_exists=np.ones(3, dtype=bool),
        existence_supervised=np.ones(3, dtype=bool),
        localization_supervised=np.ones(3, dtype=bool),
        association_supervised=np.ones(3, dtype=bool),
        reference_xy=xy,
        reference_area_px2=np.full(3, 25.0),
        reference_masks=tuple(
            _mask(round(point[0]), round(point[1])) for point in xy
        ),
    )


def _track(
    track_id: str,
    samples: dict[int, tuple[float, float]],
) -> OpenWorldTrack:
    detections = []
    for frame_index, xy in sorted(samples.items()):
        mask_xy = (
            (58, 42)
            if xy[0] >= 60.0
            else (round(xy[0]), round(xy[1]))
        )
        detections.append(
            ObjectDetection(
                frame_index=frame_index,
                detection_id=f"{track_id}_{frame_index}",
                xy=np.asarray(xy, dtype=np.float64),
                area_px2=25.0,
                entity_class="rigid_body",
                mask=_mask(*mask_xy),
                confidence=1.0,
                evidence_tier=EvidenceTier.PARTICIPANT,
                sources=("unit_test",),
            )
        )
    return OpenWorldTrack(
        track_id=track_id,
        detections=tuple(detections),
        confirmed=True,
        evidence_tier=EvidenceTier.PARTICIPANT,
    )


def _inputs() -> dict:
    times_s = [0.0, 0.1, 0.2]
    grid = build_common_time_grid(
        times_s,
        interval_start_s=0.0,
        interval_end_s=0.3,
    )
    timeline = _timeline()
    observation = OpenWorldObservation(
        tracks=(
            _track("original", {0: (10.0, 20.0)}),
            _track("far_extra", {1: (80.0, 45.0)}),
            _track("replacement", {2: (22.0, 20.0)}),
        ),
        overflow_counts=np.zeros(3, dtype=np.float64),
        diagnostics={"adapter": "unit_test"},
    )
    comparison = compare_open_world_v2(
        expected_timelines=[timeline],
        prediction_observation=observation,
        time_grid=grid,
        frame_diagonal_px=80.0,
        minimum_match_position_similarity=0.1,
    )
    reference_frames = [
        np.full((48, 64, 3), 40 + index * 5, dtype=np.uint8)
        for index in range(3)
    ]
    prediction_frames = [
        np.full((48, 64, 3), 25 + index * 5, dtype=np.uint8)
        for index in range(3)
    ]
    return {
        "scene_name": "Test rigid body",
        "times_s": times_s,
        "reference_frames": reference_frames,
        "prediction_frames": prediction_frames,
        "expected_timelines": [timeline],
        "prediction_observation": observation,
        "comparison": comparison,
    }


class OpenWorldV2ArtifactTest(unittest.TestCase):
    def test_issues_only_policy_skips_clean_case_deterministically(self) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "clean", "scene_id": "free_fall"},
                job={"job_id": "clean_job"},
                evaluator_config={"type": "free_fall_state_v7"},
                prediction={"video_sha256": "0" * 64},
                run_id="baseline_clean_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            artifacts = visual.write_open_world_v2_artifacts(
                request,
                config={
                    "enabled": True,
                    "mode": "issues_only",
                },
                score_summary={"score": 1.0},
                has_issues=False,
                **values,
            )
            manifest = json.loads(
                Path(
                    artifacts["open_world_v2_artifact_manifest"]
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("skipped", manifest["status"])
            self.assertEqual("no_issue_detected", manifest["reason"])
            self.assertFalse(request.visualization_root.exists())

    def test_quad_layout_records_score_and_scene_diagnostics(self) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "quad", "scene_id": "free_fall"},
                job={"job_id": "quad_job"},
                evaluator_config={"type": "free_fall_state_v7"},
                prediction={"video_sha256": "1" * 64},
                run_id="baseline_quad_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            artifacts = visual.write_open_world_v2_artifacts(
                request,
                config={
                    "enabled": True,
                    "mode": "all",
                    "layout": "quad",
                    "panel_width": 128,
                    "panel_height": 96,
                    "fps": 10.0,
                },
                score_summary={
                    "score": 0.75,
                    "components": {"physics": 0.8},
                },
                per_frame_diagnostics=[
                    {"axis progress": index / 2} for index in range(3)
                ],
                **values,
            )
            manifest = json.loads(
                Path(
                    artifacts["open_world_v2_artifact_manifest"]
                ).read_text(encoding="utf-8")
            )
            video_path = Path(manifest["files"]["overlay_video"]["path"])
            capture = cv2.VideoCapture(str(video_path))
            self.assertEqual(256, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.assertEqual(192, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            capture.release()
            audit = json.loads(
                Path(manifest["files"]["audit_json"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(0.75, audit["score_summary"]["score"])
            self.assertEqual(
                0.5,
                audit["per_frame"][1]["scene_diagnostics"][
                    "axis progress"
                ],
            )

    def test_run_owned_overlay_and_lossless_audit_are_locally_manifested(
        self,
    ) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "run" / "evaluation" / "cases" / "job_0",
                case={
                    "case_id": "case_with_events",
                    "scene_id": "free_fall",
                },
                job={"job_id": "job_0"},
                evaluator_config={"type": "free_fall_state_v6"},
                run_id="baseline_task_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            artifacts = visual.write_open_world_v2_artifacts(
                request,
                config={
                    "panel_width": 128,
                    "panel_height": 96,
                    "footer_height": 96,
                    "fps": 10.0,
                },
                **values,
            )
            manifest_path = Path(
                artifacts["open_world_v2_artifact_manifest"]
            )
            self.assertTrue(manifest_path.is_relative_to(request.artifact_dir))
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(
                "open_world_v2_artifacts",
                manifest["artifact_protocol"]["id"],
            )
            video_path = Path(
                manifest["files"]["overlay_video"]["path"]
            )
            audit_path = Path(manifest["files"]["audit_json"]["path"])
            relative = Path(manifest["visualization_directory"]).relative_to(
                root / "run" / "evaluation" / "visualizations"
            )
            self.assertEqual(
                ("free_fall", "case_with_events"),
                relative.parts[:2],
            )
            self.assertEqual("visualization.mp4", video_path.name)
            self.assertEqual("audit.json", audit_path.name)
            self.assertTrue(video_path.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertIn(
                "free_fall/case_with_events/",
                manifest["files"]["overlay_video"][
                    "visualization_path"
                ],
            )
            for record in manifest["files"].values():
                self.assertEqual(64, len(record["sha256"]))
                self.assertGreater(record["size_bytes"], 0)

            capture = cv2.VideoCapture(str(video_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(3, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.assertEqual(256, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.assertEqual(192, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            capture.release()

            probe = json.loads(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,pix_fmt",
                        "-of",
                        "json",
                        str(video_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            self.assertEqual("h264", probe["streams"][0]["codec_name"])
            self.assertEqual("yuv420p", probe["streams"][0]["pix_fmt"])
            payload = video_path.read_bytes()
            self.assertGreater(payload.find(b"mdat"), payload.find(b"moov"))

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ["body_0"],
                [
                    row["entity_id"]
                    for row in audit["expected_entities"]
                ],
            )
            self.assertEqual(
                {"original", "far_extra", "replacement"},
                {
                    row["track_id"]
                    for row in audit["prediction_observation"]["tracks"]
                },
            )
            self.assertEqual(
                ["body_0"],
                audit["per_frame"][1]["missing_entity_ids"],
            )
            self.assertEqual(
                ["far_extra"],
                audit["per_frame"][1]["extra_track_ids"],
            )
            self.assertTrue(
                audit["per_frame"][1]["rejected_candidate_matches"]
            )
            self.assertTrue(audit["per_frame"][2]["id_switches"])
            self.assertEqual(
                [1.0, 0.0, 1.0],
                [
                    row["full_subject_mask_iou"]
                    for row in audit["per_frame"]
                ],
            )

    def test_run_policy_disables_run_owned_video_by_default(self) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "disabled", "scene_id": "free_fall"},
                job={"job_id": "disabled_job"},
                evaluator_config={"type": "free_fall_state_v7"},
                run_id="baseline_run",
                save_visualizations=False,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            artifacts = visual.write_open_world_v2_artifacts(
                request,
                config={
                    "enabled": True,
                },
                **values,
            )
            manifest = artifacts["open_world_v2_artifacts"]
            self.assertEqual("disabled", manifest["status"])
            self.assertEqual(
                "runtime_save_visualizations_false",
                manifest["reason"],
            )
            self.assertFalse(request.visualization_root.exists())

    def test_renderer_failure_is_a_score_safe_failed_manifest(self) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "case_0", "scene_id": "pendulum"},
                job={"job_id": "job_0"},
                evaluator_config={"type": "pendulum_state_v6"},
                run_id="baseline_failure_run",
                save_visualizations=True,
                visualization_root=root / "run" / "evaluation" / "visualizations",
            )
            with patch.object(
                visual,
                "render_open_world_v2_overlay",
                side_effect=PermissionError("run visualization unavailable"),
            ):
                artifacts = visual.write_open_world_v2_artifacts(
                    request,
                    config={"enabled": True},
                    **values,
                )
            manifest = json.loads(
                Path(
                    artifacts["open_world_v2_artifact_manifest"]
                ).read_text(encoding="utf-8")
            )
            self.assertEqual("failed", manifest["status"])
            self.assertEqual(
                "best_effort_never_changes_case_score",
                manifest["storage_policy"],
            )
            self.assertEqual("PermissionError", manifest["error"]["type"])
            self.assertIn(
                "run visualization unavailable",
                manifest["error"]["message"],
            )


if __name__ == "__main__":
    unittest.main()
