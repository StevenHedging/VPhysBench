"""Official SAM 3.1 sessions used by offline GT curation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...evaluation.common.csti.observation import PromptGroupConfig
from ...evaluation.common.masks.sam31_text import Sam31TextVideoSegmenter


PredictorFactory = Callable[[], Any]


@dataclass(frozen=True)
class Sam31PointSeed:
    semantic_id: str
    backend_object_id: int
    point_xy: tuple[float, float]


@dataclass(frozen=True)
class Sam31BoxSeed:
    semantic_id: str
    text: str
    reference_mask: np.ndarray
    box_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class Sam31GtTrack:
    semantic_id: str
    backend_object_id: int
    masks: np.ndarray
    boxes_xywh: np.ndarray
    confidences: np.ndarray
    prompt: str
    source: str


class Sam31GtPredictor:
    """Reuse the evaluator's pinned model loader with curation-only prompts."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        self._segmenter = Sam31TextVideoSegmenter(
            config,
            predictor_factory=predictor_factory,
        )

    def discover(
        self,
        frames: Sequence[np.ndarray],
        *,
        prompts: Sequence[str],
    ) -> tuple[Sam31GtTrack, ...]:
        prompt_values = tuple(str(value).strip() for value in prompts)
        if not prompt_values or any(not value for value in prompt_values):
            raise ValueError("SAM3.1 GT discovery prompts must be non-empty")
        if len(set(prompt_values)) != len(prompt_values):
            raise ValueError("SAM3.1 GT discovery prompts must be unique")
        groups = tuple(
            PromptGroupConfig(
                group_id=f"prompt_{index:02d}",
                text=prompt,
                entity_classes=("ball",),
            )
            for index, prompt in enumerate(prompt_values)
        )
        candidates = self._segmenter.segment(frames, groups)
        prompt_by_group = {group.group_id: group.text for group in groups}
        return tuple(
            Sam31GtTrack(
                semantic_id=candidate.candidate_id,
                backend_object_id=candidate.backend_object_id,
                masks=candidate.masks,
                boxes_xywh=candidate.boxes_xywh,
                confidences=candidate.confidences,
                prompt=prompt_by_group[candidate.prompt_group_id],
                source="full_frame_text",
            )
            for candidate in candidates
        )

    @staticmethod
    def _validate_seeds(
        seeds: Sequence[Sam31PointSeed],
        *,
        frame_shape: tuple[int, int],
    ) -> tuple[Sam31PointSeed, ...]:
        values = tuple(seeds)
        if not values:
            raise ValueError("SAM3.1 GT point tracking requires at least one seed")
        semantic_ids = tuple(value.semantic_id for value in values)
        backend_ids = tuple(int(value.backend_object_id) for value in values)
        if (
            any(not value for value in semantic_ids)
            or len(set(semantic_ids)) != len(values)
            or len(set(backend_ids)) != len(values)
        ):
            raise ValueError("SAM3.1 GT point seeds require unique identities")
        height, width = frame_shape
        for seed in values:
            point = np.asarray(seed.point_xy, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("SAM3.1 GT point seed must be finite XY")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("SAM3.1 GT point seed lies outside the video frame")
        return values

    def track_points(
        self,
        frames: Sequence[np.ndarray],
        seeds: Sequence[Sam31PointSeed],
    ) -> dict[str, Sam31GtTrack]:
        normalized = self._segmenter._validate_frames(frames)
        frame_shape = normalized[0].shape[:2]
        seed_values = self._validate_seeds(seeds, frame_shape=frame_shape)
        frame_count = len(normalized)
        with self._segmenter._lock:
            self._segmenter._load()
            predictor = self._segmenter._predictor
            assert predictor is not None
            with tempfile.TemporaryDirectory(prefix="vphysbench-sam31-gt-") as value:
                directory = Path(value)
                self._segmenter._write_frames(normalized, directory)
                result: dict[str, Sam31GtTrack] = {}
                for seed in seed_values:
                    response = predictor.handle_request(
                        {
                            "type": "start_session",
                            "resource_path": str(directory),
                            "offload_video_to_cpu": True,
                            "offload_state_to_cpu": False,
                        }
                    )
                    session_id = str(response["session_id"])
                    masks = np.zeros((frame_count, *frame_shape), dtype=bool)
                    boxes = np.zeros((frame_count, 4), dtype=np.float32)
                    probabilities = np.zeros(frame_count, dtype=np.float32)

                    def consume(output: Mapping[str, Any]) -> None:
                        frame_index = int(output["frame_index"])
                        if not 0 <= frame_index < frame_count:
                            raise ValueError(
                                f"SAM3.1 GT returned frame {frame_index} outside the video"
                            )
                        ids, output_masks, output_boxes, output_probabilities = (
                            self._segmenter._response_arrays(
                                output,
                                frame_shape=frame_shape,
                            )
                        )
                        matches = np.flatnonzero(ids == int(seed.backend_object_id))
                        if len(matches) > 1:
                            raise ValueError(
                                f"SAM3.1 GT duplicated backend object {seed.backend_object_id}"
                            )
                        if not len(matches):
                            return
                        source_index = int(matches[0])
                        masks[frame_index] = output_masks[source_index]
                        boxes[frame_index] = output_boxes[source_index]
                        probabilities[frame_index] = output_probabilities[source_index]

                    negative_points = [
                        [float(other.point_xy[0]), float(other.point_xy[1])]
                        for other in seed_values
                        if other.semantic_id != seed.semantic_id
                    ]
                    try:
                        consume(
                            predictor.handle_request(
                                {
                                    "type": "add_prompt",
                                    "session_id": session_id,
                                    "frame_index": 0,
                                    "points": [
                                        [
                                            float(seed.point_xy[0]),
                                            float(seed.point_xy[1]),
                                        ],
                                        *negative_points,
                                    ],
                                    "point_labels": [1, *([0] * len(negative_points))],
                                    "clear_old_points": True,
                                    "obj_id": int(seed.backend_object_id),
                                    "rel_coordinates": False,
                                    "output_prob_thresh": (
                                        self._segmenter.output_probability_threshold
                                    ),
                                }
                            )
                        )
                        for output in predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "propagation_direction": "forward",
                                "start_frame_index": 0,
                                "output_prob_thresh": (
                                    self._segmenter.output_probability_threshold
                                ),
                            }
                        ):
                            consume(output)
                    finally:
                        predictor.handle_request(
                            {
                                "type": "close_session",
                                "session_id": session_id,
                                "run_gc_collect": True,
                            }
                        )
                    result[seed.semantic_id] = Sam31GtTrack(
                        semantic_id=seed.semantic_id,
                        backend_object_id=seed.backend_object_id,
                        masks=masks,
                        boxes_xywh=boxes,
                        confidences=probabilities,
                        prompt="positive_point_with_other_subject_negatives",
                        source="independent_full_frame_point_session",
                    )
        if any(not track.masks[0].any() for track in result.values()):
            missing = sorted(
                semantic_id
                for semantic_id, track in result.items()
                if not track.masks[0].any()
            )
            raise ValueError(f"SAM3.1 GT point initialization missed {missing}")
        return result

    @staticmethod
    def _validate_box_seeds(
        seeds: Sequence[Sam31BoxSeed],
        *,
        frame_shape: tuple[int, int],
    ) -> tuple[Sam31BoxSeed, ...]:
        values = tuple(seeds)
        if not values:
            raise ValueError("SAM3.1 GT box tracking requires at least one seed")
        semantic_ids = tuple(value.semantic_id for value in values)
        if any(not value for value in semantic_ids) or len(set(semantic_ids)) != len(values):
            raise ValueError("SAM3.1 GT box seeds require unique identities")
        for seed in values:
            if not seed.text.strip():
                raise ValueError("SAM3.1 GT box seed text must be non-empty")
            mask = np.asarray(seed.reference_mask)
            if mask.shape != frame_shape or not np.asarray(mask, dtype=bool).any():
                raise ValueError("SAM3.1 GT box seed reference mask is invalid")
            box = np.asarray(seed.box_xywh, dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("SAM3.1 GT box seed must be finite XYWH")
            x, y, width, height = box
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > 1 + 1e-6
                or y + height > 1 + 1e-6
            ):
                raise ValueError("SAM3.1 GT box seed must fit normalized frame bounds")
        return values

    def track_boxes(
        self,
        frames: Sequence[np.ndarray],
        seeds: Sequence[Sam31BoxSeed],
        *,
        initial_iou_threshold: float,
        output_probability_threshold: float,
    ) -> dict[str, Sam31GtTrack]:
        """Track each locked subject in an independent text-plus-box session."""

        if not 0 <= initial_iou_threshold <= 1:
            raise ValueError("SAM3.1 GT initial box IoU threshold must lie in [0,1]")
        if not 0 <= output_probability_threshold <= 1:
            raise ValueError("SAM3.1 GT tracking probability threshold must lie in [0,1]")
        normalized = self._segmenter._validate_frames(frames)
        frame_shape = normalized[0].shape[:2]
        seed_values = self._validate_box_seeds(seeds, frame_shape=frame_shape)
        frame_count = len(normalized)
        with self._segmenter._lock:
            self._segmenter._load()
            predictor = self._segmenter._predictor
            assert predictor is not None
            with tempfile.TemporaryDirectory(prefix="vphysbench-sam31-gt-box-") as value:
                directory = Path(value)
                self._segmenter._write_frames(normalized, directory)
                result: dict[str, Sam31GtTrack] = {}
                for seed in seed_values:
                    response = predictor.handle_request(
                        {
                            "type": "start_session",
                            "resource_path": str(directory),
                            "offload_video_to_cpu": True,
                            "offload_state_to_cpu": False,
                        }
                    )
                    session_id = str(response["session_id"])
                    masks = np.zeros((frame_count, *frame_shape), dtype=bool)
                    boxes = np.zeros((frame_count, 4), dtype=np.float32)
                    probabilities = np.zeros(frame_count, dtype=np.float32)
                    selected_backend_id: int | None = None

                    def consume(output: Mapping[str, Any], *, initialize: bool = False) -> None:
                        nonlocal selected_backend_id
                        frame_index = int(output["frame_index"])
                        if not 0 <= frame_index < frame_count:
                            raise ValueError(
                                f"SAM3.1 GT returned frame {frame_index} outside the video"
                            )
                        ids, output_masks, output_boxes, output_probabilities = (
                            self._segmenter._response_arrays(
                                output,
                                frame_shape=frame_shape,
                            )
                        )
                        if initialize:
                            if frame_index != 0 or not len(ids):
                                raise ValueError(
                                    f"SAM3.1 GT box initialization missed {seed.semantic_id}"
                                )
                            reference = np.asarray(seed.reference_mask, dtype=bool)
                            intersections = np.logical_and(output_masks, reference).reshape(
                                len(ids), -1
                            ).sum(axis=1)
                            unions = np.logical_or(output_masks, reference).reshape(
                                len(ids), -1
                            ).sum(axis=1)
                            ious = np.divide(
                                intersections,
                                unions,
                                out=np.zeros(len(ids), dtype=np.float64),
                                where=unions > 0,
                            )
                            best = int(np.argmax(ious))
                            if float(ious[best]) < initial_iou_threshold:
                                raise ValueError(
                                    f"SAM3.1 GT box initialization IoU for {seed.semantic_id} "
                                    f"is {float(ious[best]):.6f}, below {initial_iou_threshold:.6f}"
                                )
                            selected_backend_id = int(ids[best])
                        if selected_backend_id is None:
                            raise RuntimeError("SAM3.1 GT box identity was not initialized")
                        matches = np.flatnonzero(ids == selected_backend_id)
                        if len(matches) > 1:
                            raise ValueError(
                                f"SAM3.1 GT duplicated backend object {selected_backend_id}"
                            )
                        if not len(matches):
                            return
                        source_index = int(matches[0])
                        masks[frame_index] = output_masks[source_index]
                        boxes[frame_index] = output_boxes[source_index]
                        probabilities[frame_index] = output_probabilities[source_index]

                    other_boxes = [
                        list(other.box_xywh)
                        for other in seed_values
                        if other.semantic_id != seed.semantic_id
                    ]
                    try:
                        consume(
                            predictor.handle_request(
                                {
                                    "type": "add_prompt",
                                    "session_id": session_id,
                                    "frame_index": 0,
                                    "text": seed.text,
                                    "bounding_boxes": [list(seed.box_xywh), *other_boxes],
                                    "bounding_box_labels": [1, *([0] * len(other_boxes))],
                                    "output_prob_thresh": (
                                        output_probability_threshold
                                    ),
                                }
                            ),
                            initialize=True,
                        )
                        for output in predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session_id,
                                "propagation_direction": "forward",
                                "start_frame_index": 0,
                                "output_prob_thresh": (
                                    output_probability_threshold
                                ),
                            }
                        ):
                            consume(output)
                    finally:
                        predictor.handle_request(
                            {
                                "type": "close_session",
                                "session_id": session_id,
                                "run_gc_collect": True,
                            }
                        )
                    assert selected_backend_id is not None
                    result[seed.semantic_id] = Sam31GtTrack(
                        semantic_id=seed.semantic_id,
                        backend_object_id=selected_backend_id,
                        masks=masks,
                        boxes_xywh=boxes,
                        confidences=probabilities,
                        prompt=seed.text,
                        source="independent_full_frame_text_box_session",
                    )
        return result

    def describe(self) -> dict[str, Any]:
        value = self._segmenter.describe()
        value.update(
            {
                "curation_policy": "text_discovery_then_locked_point_tracking",
                "mask_overlap_policy": "independent_nonexclusive",
            }
        )
        return value


__all__ = [
    "Sam31BoxSeed",
    "Sam31GtPredictor",
    "Sam31GtTrack",
    "Sam31PointSeed",
]
