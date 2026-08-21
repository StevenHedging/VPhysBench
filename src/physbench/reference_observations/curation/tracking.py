from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from ...evaluation.common.masks.sam2 import MaskPrompt
from .anchors import AnchorCandidate
from .overrides import CorrectionPrompt, LifecycleOverride


@dataclass(frozen=True)
class TrackingCandidate:
    masks_by_object: dict[str, np.ndarray]
    states_by_object: dict[str, np.ndarray]
    seed_frame_by_observation: np.ndarray
    propagation_metadata: tuple[dict[str, Any], ...]


def isolate_compact_round_subject(
    masks: np.ndarray,
    states: np.ndarray,
    *,
    radius_scale: float = 1.45,
    minimum_core_radius: float = 2.0,
    maximum_internal_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the thick round body while removing an attached thin tether.

    The distance-transform maximum lies inside the bob or spring mass even when
    SAM also selects a string, loop, or spring. Short interior tracking gaps are
    filled by translating the preceding reviewed shape between neighbouring
    centers; boundary/lifecycle gaps are never synthesized.
    """
    values = (np.asarray(masks) > 0).astype(np.uint8)
    result_states = np.asarray(states, dtype=np.uint8).copy()
    if values.ndim != 3 or result_states.shape != (values.shape[0],):
        raise ValueError("round-subject refinement requires aligned THW/T arrays")
    if radius_scale <= 1.0 or minimum_core_radius <= 0 or maximum_internal_gap < 0:
        raise ValueError("round-subject refinement parameters are invalid")
    refined = np.zeros_like(values)
    centers = np.full((len(values), 2), np.nan, np.float64)
    yy, xx = np.ogrid[: values.shape[1], : values.shape[2]]
    for index, mask in enumerate(values):
        if not mask.any():
            continue
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, radius, _, center = cv2.minMaxLoc(distance)
        if radius < minimum_core_radius:
            result_states[index] = 3
            continue
        center_x, center_y = center
        neighborhood = (
            (xx - center_x) ** 2 + (yy - center_y) ** 2
            <= (radius_scale * radius) ** 2
        )
        cropped = np.logical_and(mask > 0, neighborhood).astype(np.uint8)
        count, labels = cv2.connectedComponents(cropped, 8)
        label = int(labels[center_y, center_x])
        if count <= 1 or label == 0:
            result_states[index] = 3
            continue
        component = labels == label
        refined[index, component] = 1
        ys, xs = np.nonzero(component)
        centers[index] = (float(xs.mean()), float(ys.mean()))
        result_states[index] = 0

    present = refined.reshape(len(refined), -1).any(axis=1)
    index = 0
    while index < len(refined):
        if present[index]:
            index += 1
            continue
        start = index
        while index < len(refined) and not present[index]:
            index += 1
        end = index - 1
        length = end - start + 1
        if (
            length > maximum_internal_gap
            or start == 0
            or index == len(refined)
            or np.any(result_states[start : end + 1] != 3)
        ):
            continue
        left = start - 1
        right = index
        for missing in range(start, end + 1):
            fraction = (missing - left) / (right - left)
            target = centers[left] + fraction * (centers[right] - centers[left])
            delta = target - centers[left]
            transform = np.asarray(
                [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]], np.float32
            )
            refined[missing] = cv2.warpAffine(
                refined[left],
                transform,
                (refined.shape[2], refined.shape[1]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            result_states[missing] = 0
    return refined, result_states


def resolve_trailing_boundary_exit(
    masks: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Classify an empty suffix after the last boundary sliver as out-of-frame."""
    values = np.asarray(masks) > 0
    result = np.asarray(states, dtype=np.uint8).copy()
    if values.ndim != 3 or result.shape != (values.shape[0],):
        raise ValueError("boundary lifecycle inference requires aligned THW/T arrays")
    present = values.reshape(values.shape[0], -1).any(axis=1)
    visible = np.flatnonzero(present)
    if not len(visible) or visible[-1] == len(result) - 1:
        return result
    last = int(visible[-1])
    if np.any(result[last + 1 :] != 3):
        return result
    ys, xs = np.nonzero(values[last])
    height, width = values.shape[1:]
    touches_boundary = bool(
        xs.min() == 0
        or xs.max() == width - 1
        or ys.min() == 0
        or ys.max() == height - 1
    )
    if touches_boundary:
        result[last + 1 :] = 2
    return result


def resolve_trailing_predicted_exit(
    masks: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Classify a trailing absence when a stable trajectory crosses the boundary.

    The reference timeline can sample a fast object sparsely enough that its last
    visible mask does not touch an image edge.  This inference is deliberately
    conservative: it requires three consecutive visible observations, consistent
    motion and scale, and a constant-acceleration prediction that places the
    entire last bounding box outside the next observation.
    """
    values = np.asarray(masks) > 0
    result = np.asarray(states, dtype=np.uint8).copy()
    if values.ndim != 3 or result.shape != (values.shape[0],):
        raise ValueError("predicted lifecycle inference requires aligned THW/T arrays")
    present = values.reshape(values.shape[0], -1).any(axis=1)
    visible = np.flatnonzero(present)
    if len(visible) < 3 or visible[-1] == len(result) - 1:
        return result
    last = int(visible[-1])
    if last < 2 or not np.all(present[last - 2 : last + 1]):
        return result
    if np.any(result[last + 1 :] != 3):
        return result

    centroids: list[np.ndarray] = []
    areas: list[int] = []
    last_bbox: np.ndarray | None = None
    for index in range(last - 2, last + 1):
        ys, xs = np.nonzero(values[index])
        centroids.append(np.asarray([xs.mean(), ys.mean()], dtype=np.float64))
        areas.append(int(len(xs)))
        if index == last:
            last_bbox = np.asarray(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64
            )
    if max(areas) / min(areas) > 1.75:
        return result

    previous_velocity = centroids[1] - centroids[0]
    latest_velocity = centroids[2] - centroids[1]
    previous_speed = float(np.linalg.norm(previous_velocity))
    latest_speed = float(np.linalg.norm(latest_velocity))
    if previous_speed < 1.0 or latest_speed < 1.0:
        return result
    cosine = float(
        np.dot(previous_velocity, latest_velocity) / (previous_speed * latest_speed)
    )
    if cosine < 0.9:
        return result

    acceleration = latest_velocity - previous_velocity
    if float(np.linalg.norm(acceleration)) > 0.75 * max(previous_speed, latest_speed):
        return result
    predicted_delta = latest_velocity + acceleration
    if float(np.dot(predicted_delta, latest_velocity)) <= 0:
        return result

    assert last_bbox is not None
    predicted_bbox = last_bbox + np.asarray(
        [predicted_delta[0], predicted_delta[1], predicted_delta[0], predicted_delta[1]]
    )
    height, width = values.shape[1:]
    fully_outside = bool(
        predicted_bbox[2] < 0
        or predicted_bbox[0] >= width
        or predicted_bbox[3] < 0
        or predicted_bbox[1] >= height
    )
    if fully_outside:
        result[last + 1 :] = 2
    return result


def _candidate_prompt(anchor: AnchorCandidate, frame_index: int) -> MaskPrompt:
    x, y = anchor.centroid_xy
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=np.asarray(anchor.bbox_xyxy, dtype=np.float32),
        points_xy=np.asarray([[x, y]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        metadata={"object_id": anchor.object_id, "source": "independent_anchor"},
    )


def _correction_prompt(prompt: CorrectionPrompt) -> MaskPrompt:
    return MaskPrompt(
        frame_index=prompt.frame_index,
        box_xyxy=np.asarray(prompt.box_xyxy, dtype=np.float32),
        points_xy=np.asarray(prompt.points_xy, dtype=np.float32),
        point_labels=np.asarray(prompt.point_labels, dtype=np.int32),
        metadata={"object_id": prompt.object_id, "source": "reviewed_correction"},
    )


def _mask_prompt(object_id: str, frame_index: int, mask: np.ndarray) -> MaskPrompt:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"cannot reseed empty mask for {object_id} at {frame_index}")
    return MaskPrompt(
        frame_index=frame_index,
        box_xyxy=np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], np.float32),
        points_xy=np.asarray([[xs.mean(), ys.mean()]], np.float32),
        point_labels=np.asarray([1], np.int32),
        metadata={"object_id": object_id, "source": "propagated_reseed"},
    )


class CuratedSam2Tracker:
    def __init__(self, segmenter: Any):
        self.segmenter = segmenter

    def track(
        self,
        frames: Sequence[np.ndarray],
        *,
        anchors: Sequence[AnchorCandidate],
        corrections: Sequence[CorrectionPrompt],
        lifecycle: Sequence[LifecycleOverride],
    ) -> TrackingCandidate:
        if not frames or not anchors:
            raise ValueError("tracking requires frames and independent anchors")
        object_ids = tuple(anchor.object_id for anchor in anchors)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("tracking anchors must have unique object ids")
        unknown = {item.object_id for item in (*corrections, *lifecycle)} - set(object_ids)
        if unknown:
            raise ValueError(f"tracking overrides reference unknown objects: {sorted(unknown)}")
        frame_count = len(frames)
        corrections_by_seed = {
            seed: {item.object_id: item for item in corrections if item.frame_index == seed}
            for seed in sorted({item.frame_index for item in corrections})
        }
        if any(seed >= frame_count for seed in corrections_by_seed):
            raise ValueError("correction prompt frame is outside the video")

        seed_tubes: dict[int, dict[str, np.ndarray]] = {}
        metadata: list[dict[str, Any]] = []
        initial_prompts = [_candidate_prompt(anchor, 0) for anchor in anchors]
        tubes, details = self.segmenter.segment_instances(
            list(frames), prompts=initial_prompts, exclusive_masks=True
        )
        seed_tubes[0] = {
            object_id: np.asarray(tube, dtype=np.uint8) > 0
            for object_id, tube in zip(object_ids, tubes, strict=True)
        }
        metadata.append(dict(details))

        anchor_by_id = {anchor.object_id: anchor for anchor in anchors}
        for seed, reviewed in corrections_by_seed.items():
            prompts = []
            for object_id in object_ids:
                correction = reviewed.get(object_id)
                if correction is not None:
                    prompts.append(_correction_prompt(correction))
                else:
                    earlier = max(existing for existing in seed_tubes if existing < seed)
                    mask = seed_tubes[earlier][object_id][seed]
                    if mask.any():
                        prompts.append(_mask_prompt(object_id, seed, mask))
                    else:
                        prompts.append(_candidate_prompt(anchor_by_id[object_id], seed))
            tubes, details = self.segmenter.segment_instances(
                list(frames), prompts=prompts, exclusive_masks=True
            )
            seed_tubes[seed] = {
                object_id: np.asarray(tube, dtype=np.uint8) > 0
                for object_id, tube in zip(object_ids, tubes, strict=True)
            }
            metadata.append(dict(details))

        seeds = np.asarray(sorted(seed_tubes), dtype=np.int32)
        selected = np.asarray(
            [int(seeds[np.argmin(np.abs(seeds - index))]) for index in range(frame_count)],
            dtype=np.int32,
        )
        masks: dict[str, np.ndarray] = {}
        states: dict[str, np.ndarray] = {}
        for object_id in object_ids:
            tube = np.stack(
                [seed_tubes[int(seed)][object_id][index] for index, seed in enumerate(selected)]
            ).astype(np.uint8)
            state = np.where(tube.reshape(frame_count, -1).any(axis=1), 0, 3).astype(np.int8)
            masks[object_id] = tube
            state = resolve_trailing_boundary_exit(tube, state)
            states[object_id] = resolve_trailing_predicted_exit(tube, state)
        for override in lifecycle:
            if override.end_index >= frame_count:
                raise ValueError("lifecycle override range is outside the video")
            selection = slice(override.start_index, override.end_index + 1)
            states[override.object_id][selection] = override.state
            if override.state != 0:
                masks[override.object_id][selection] = 0
        return TrackingCandidate(
            masks_by_object=masks,
            states_by_object=states,
            seed_frame_by_observation=selected,
            propagation_metadata=tuple(metadata),
        )
