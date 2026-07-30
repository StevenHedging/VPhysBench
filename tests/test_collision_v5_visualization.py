from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.entities import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
)
from physbench.evaluation.scenes.collision.nbody import (
    extract_nbody_collision_state,
)
from physbench.evaluation.scenes.collision import v5_visualization as visual


def _state(body_count: int):
    times = [0.0, 0.1, 0.2]
    centers = np.zeros((len(times), body_count, 2), dtype=np.float64)
    for frame_index in range(len(times)):
        for body_index in range(body_count):
            centers[frame_index, body_index] = [
                12.0 + 14.0 * body_index + frame_index,
                28.0,
            ]
    return extract_nbody_collision_state(
        centers,
        np.ones((len(times), body_count), dtype=bool),
        times,
        entity_ids=[f"ball_{index + 1}" for index in range(body_count)],
        radii_px=np.full((len(times), body_count), 3.0),
        masses_kg=np.full(body_count, 0.02),
        axis_origin_xy=np.asarray([0.0, 28.0]),
        axis_direction_xy=np.asarray([1.0, 0.0]),
    )


def _inputs(body_count: int) -> dict:
    times = [0.0, 0.1, 0.2]
    height, width = 56, 96
    entity_ids = [f"ball_{index + 1}" for index in range(body_count)]
    frames = [
        np.full((height, width, 3), 90, dtype=np.uint8)
        for _ in times
    ]
    masks = [
        [np.zeros((height, width), dtype=np.uint8) for _ in times]
        for _ in entity_ids
    ]
    xy = np.zeros((len(times), body_count, 2), dtype=np.float64)
    detections: list[OpenWorldTrack] = []
    for body_index, entity_id in enumerate(entity_ids):
        track_detections = []
        for frame_index in range(len(times)):
            center = (
                12 + 14 * body_index + frame_index,
                28,
            )
            cv2.circle(
                masks[body_index][frame_index],
                center,
                3,
                255,
                -1,
            )
            xy[frame_index, body_index] = center
            track_detections.append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=f"p{body_index}_{frame_index}",
                    xy=np.asarray(center, dtype=np.float64),
                    area_px2=29.0,
                    entity_class="ball",
                    mask=masks[body_index][frame_index],
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("directed",),
                )
            )
        detections.append(
            OpenWorldTrack(
                track_id=f"track_{body_index + 1}",
                detections=tuple(track_detections),
                confirmed=True,
                evidence_tier=EvidenceTier.PARTICIPANT,
            )
        )
    extra_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(extra_mask, (80, 20), 3, 255, -1)
    detections.append(
        OpenWorldTrack(
            track_id="extra",
            detections=(
                ObjectDetection(
                    frame_index=1,
                    detection_id="extra_1",
                    xy=np.asarray([80.0, 20.0]),
                    area_px2=29.0,
                    entity_class="ball",
                    mask=extra_mask,
                    confidence=1.0,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                    sources=("residual",),
                ),
            ),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )
    )
    comparison = []
    for frame_index, time_s in enumerate(times):
        matched_count = body_count - (1 if frame_index == 1 else 0)
        comparison.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "matches": [
                    {
                        "entity_id": entity_ids[index],
                        "prediction_track_id": f"track_{index + 1}",
                        "position_score": 0.9,
                    }
                    for index in range(matched_count)
                ],
                "missing_entity_ids": (
                    [entity_ids[-1]] if frame_index == 1 else []
                ),
                "residual_track_ids": (
                    ["extra"] if frame_index == 1 else []
                ),
                "overflow_count": 1.0 if frame_index == 1 else 0.0,
            }
        )
    union = []
    for frame_index in range(len(times)):
        merged = np.zeros((height, width), dtype=np.uint8)
        for body_index in range(body_count):
            merged = cv2.bitwise_or(
                merged,
                masks[body_index][frame_index],
            )
        union.append(merged)
    return {
        "times_s": times,
        "reference_frames": frames,
        "prediction_frames": frames,
        "reference_masks": masks,
        "reference_xy": xy,
        "reference_valid": np.ones(
            (len(times), body_count),
            dtype=bool,
        ),
        "entity_ids": entity_ids,
        "prediction_observation": OpenWorldObservation(
            tracks=tuple(detections),
            overflow_counts=np.asarray([0.0, 1.0, 0.0]),
        ),
        "comparison": comparison,
        "reference_union": union,
        "prediction_union": union,
        "union_ious": [1.0, 0.7, 1.0],
        "reference_state": _state(body_count),
        "prediction_state": None,
        "prediction_available": [True, True, False],
    }


class _MemoryWriter:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class CollisionV5VisualizationTests(unittest.TestCase):
    def test_video_renderer_supports_two_and_four_entities_with_missing_extra(
        self,
    ) -> None:
        for body_count in (2, 4):
            with self.subTest(body_count=body_count):
                writer = _MemoryWriter()
                with patch.object(
                    visual.cv2,
                    "VideoWriter",
                    return_value=writer,
                ):
                    visual._write_video(
                        Path("unused.mp4"),
                        **_inputs(body_count),
                        config={
                            "fps": 10.0,
                            "panel_width": 120,
                            "panel_height": 72,
                        },
                    )
                self.assertEqual(3, len(writer.frames))
                self.assertTrue(writer.released)
                self.assertTrue(
                    all(
                        frame.shape == (144, 240, 3)
                        for frame in writer.frames
                    )
                )
                self.assertTrue(
                    all(np.any(frame != frame[0, 0]) for frame in writer.frames)
                )

    def test_prediction_panel_reports_missing_extra_and_overflow(self) -> None:
        values = _inputs(4)
        detections = visual._prediction_detections(
            values["prediction_observation"]
        )
        headers: list[tuple[str, str | None]] = []

        def capture_header(
            _frame,
            text,
            *,
            secondary=None,
        ):
            headers.append((text, secondary))

        with patch.object(visual, "_header", side_effect=capture_header):
            visual._prediction_panel(
                values["prediction_frames"][1],
                frame_index=1,
                detections=detections,
                audit=values["comparison"][1],
                available=False,
            )
        self.assertIn("N=5", headers[0][0])
        self.assertIn("MEDIA UNAVAILABLE", headers[0][0])
        self.assertIn("missing=1", headers[0][1])
        self.assertIn("extra=1", headers[0][1])
        self.assertIn("overflow=1", headers[0][1])

    def test_complete_external_manifest_records_degraded_prediction_state(
        self,
    ) -> None:
        values = _inputs(4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "sealed" / "case",
                case={"case_id": "four_body"},
                job={"job_id": "job_four_body"},
                evaluator_config={"type": "collision_1d_state_v5"},
            )

            def fake_video(path: Path, **_kwargs) -> None:
                path.write_bytes(b"synthetic-video")

            with patch.object(
                visual,
                "_write_video",
                side_effect=fake_video,
            ):
                artifacts = visual.write_collision_v5_visualization(
                    request,
                    config={
                        "enabled": True,
                        "external_root": str(root / "external"),
                        "external_root_env": (
                            "UNSET_PHYSBENCH_V5_VIS_TEST_ROOT"
                        ),
                        "repository_link": "visualizations",
                        "namespace": "test_v5",
                    },
                    **values,
                )

            manifest_path = Path(
                artifacts["collision_v5_visualization_manifest"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("complete", manifest["status"])
            audit_path = Path(manifest["files"]["audit_json"]["path"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(4, len(audit["entity_ids"]))
            self.assertFalse(audit["prediction_state_available"])
            self.assertIsNone(audit["prediction_body_count"])
            self.assertEqual([], audit["prediction_contact_events"])
            self.assertEqual(
                ["ball_4"],
                audit["per_frame"][1]["missing_entity_ids"],
            )
            self.assertEqual(
                ["extra"],
                audit["per_frame"][1]["residual_track_ids"],
            )
            evidence = {
                row["track_id"]: row
                for row in audit["prediction_track_evidence"]
            }
            self.assertEqual(
                "participant",
                evidence["extra"]["formal_tier"],
            )
            self.assertEqual(
                ["residual"],
                evidence["extra"]["sources"],
            )
            for record in manifest["files"].values():
                self.assertTrue(Path(record["path"]).is_file())
                self.assertEqual(64, len(record["sha256"]))

    def test_external_failure_is_sealed_as_best_effort_manifest(self) -> None:
        values = _inputs(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = SimpleNamespace(
                artifact_dir=root / "sealed" / "case",
                case={"case_id": "two_body"},
                job={"job_id": "job_two_body"},
                evaluator_config={"type": "collision_1d_state_v5"},
            )
            with patch.object(
                visual,
                "_artifact_directory",
                side_effect=PermissionError("external root is read-only"),
            ):
                artifacts = visual.write_collision_v5_visualization(
                    request,
                    config={"enabled": True, "external_root": str(root)},
                    **values,
                )
            manifest_path = Path(
                artifacts["collision_v5_visualization_manifest"]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("failed", manifest["status"])
            self.assertEqual(
                "best_effort_never_changes_case_score",
                manifest["storage_policy"],
            )
            self.assertEqual("PermissionError", manifest["error"]["type"])
            self.assertIn(
                "external root is read-only",
                manifest["error"]["message"],
            )


if __name__ == "__main__":
    unittest.main()
