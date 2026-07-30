from __future__ import annotations

import json
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
    def test_external_overlay_and_lossless_audit_are_locally_manifested(
        self,
    ) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local" / "case",
                case={
                    "case_id": "case_with_events",
                    "scene_id": "free_fall",
                },
                job={"job_id": "job_0"},
                evaluator_config={"type": "free_fall_state_v6"},
            )
            artifacts = visual.write_open_world_v2_artifacts(
                request,
                config={
                    "external_root": str(root / "external"),
                    "external_root_env": "UNSET_OPEN_WORLD_V2_TEST_ROOT",
                    "repository_link": "visualizations",
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
            self.assertTrue(video_path.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertIn(
                "visualizations/",
                manifest["files"]["overlay_video"][
                    "repository_path"
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

    def test_renderer_failure_is_a_score_safe_failed_manifest(self) -> None:
        values = _inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "local",
                case={"case_id": "case_0", "scene_id": "pendulum"},
                job={"job_id": "job_0"},
                evaluator_config={"type": "pendulum_state_v6"},
            )
            with patch.object(
                visual,
                "render_open_world_v2_overlay",
                side_effect=PermissionError("external volume unavailable"),
            ):
                artifacts = visual.write_open_world_v2_artifacts(
                    request,
                    config={"external_root": str(root / "external")},
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
                "external volume unavailable",
                manifest["error"]["message"],
            )


if __name__ == "__main__":
    unittest.main()
