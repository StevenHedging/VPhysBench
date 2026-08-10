"""Motion-validated frame-zero identity binding for collision v6."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.frozen_subject import load_frozen_subject_anchor
from ...common.masks.sam2 import MaskPrompt
from ...contracts import CaseEvaluationRequest
from .observation import _circle_prompt, _hough_candidates, _nms_circles


@dataclass(frozen=True)
class CollisionIdentityAnchor:
    logical_entity_id: str
    dataset_object_id: str
    mask: np.ndarray
    centroid_xy: np.ndarray
    equivalent_radius_px: float
    histogram: np.ndarray
    provenance: Mapping[str, object]


def load_collision_identity_anchors(
    request: CaseEvaluationRequest,
    *,
    entity_ids: Sequence[str],
    reference_frame: np.ndarray,
    spatial_transform: Mapping[str, object],
) -> tuple[CollisionIdentityAnchor, ...]:
    """Load every ordered collision role under both frozen ID namespaces."""

    ids = tuple(str(value) for value in entity_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("collision entity_ids must contain unique non-empty roles")
    frame = np.asarray(reference_frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("reference_frame must be an HWC BGR image")
    output: list[CollisionIdentityAnchor] = []
    for order, logical_entity_id in enumerate(ids, start=1):
        dataset_object_id = f"object_{order}"
        anchor = load_frozen_subject_anchor(
            request,
            logical_entity_id=logical_entity_id,
            dataset_object_id=dataset_object_id,
            entity_class="ball",
            spatial_transform=spatial_transform,
            error_namespace="reference_collision_subject",
        )
        if anchor.mask.shape != frame.shape[:2]:
            raise ValueError(
                "frozen collision mask and reference frame use different canvases"
            )
        histogram = _mask_histogram(frame, anchor.mask)
        histogram.setflags(write=False)
        output.append(
            CollisionIdentityAnchor(
                logical_entity_id=logical_entity_id,
                dataset_object_id=dataset_object_id,
                mask=anchor.mask,
                centroid_xy=anchor.centroid_xy,
                equivalent_radius_px=anchor.equivalent_radius_px,
                histogram=histogram,
                provenance=dict(anchor.provenance),
            )
        )
    return tuple(output)


def localize_collision_reference_candidates(
    *,
    entity_ids: Sequence[str],
    reference_frames: Sequence[np.ndarray],
    candidates: Sequence[tuple[float, float, float, float]],
    config: Mapping[str, object],
) -> tuple[tuple[CollisionIdentityAnchor, ...], dict[str, Any]]:
    """Select a unique frame-zero ball set using reference-only motion."""

    ids = tuple(str(value) for value in entity_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("collision entity_ids must contain unique non-empty roles")
    frames = [np.asarray(value) for value in reference_frames]
    if len(frames) < 2 or any(
        value.shape != frames[0].shape or value.ndim != 3 or value.shape[2] != 3
        for value in frames
    ):
        raise ValueError(
            "reference motion localization requires at least two equal HWC frames"
        )
    first = frames[0]
    differences = [
        cv2.cvtColor(cv2.absdiff(frame, first), cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]
    ]
    change = cv2.GaussianBlur(
        np.percentile(np.stack(differences), 75, axis=0).astype(np.float32),
        (3, 3),
        0,
    )
    candidate_records = []
    for raw in candidates:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise TypeError(
                "reference collision candidate must be (x, y, radius, strength)"
            )
        circle = _validated_circle(raw[:3])
        strength = float(raw[3])
        if not math.isfinite(strength):
            raise ValueError("reference collision candidate strength must be finite")
        motion = _circle_motion_metrics(change, circle)
        contrast = _local_circle_contrast(first, circle)
        candidate_records.append(
            {
                "circle": circle,
                "detector_strength": strength,
                "motion_fraction": motion["fraction"],
                "mean_change": motion["mean_change"],
                "ring_change": motion["ring_change"],
                "motion_score": motion["score"],
                "local_contrast": contrast,
                "selection_score": (
                    motion["score"] + 0.5 * strength + 0.1 * contrast
                ),
            }
        )
    viable = [
        value
        for value in candidate_records
        if value["motion_fraction"]
        >= float(config["minimum_motion_fraction"])
        and value["mean_change"] >= float(config["minimum_mean_change"])
    ]
    viable = _deduplicate_reference_candidates(viable)
    if len(viable) < len(ids):
        raise SceneAnalysisError(
            "reference_collision_frame_zero_entities_missing",
            "reference frame zero has fewer motion-validated balls than the "
            f"declared collision entity count {len(ids)}",
        )
    ranked_sets: list[
        tuple[float, tuple[dict[str, Any], ...]]
    ] = []
    for combination in itertools.combinations(viable, len(ids)):
        ordered = tuple(
            sorted(combination, key=lambda value: value["circle"][0])
        )
        y = np.asarray([value["circle"][1] for value in ordered])
        radii = np.asarray([value["circle"][2] for value in ordered])
        x = np.asarray([value["circle"][0] for value in ordered])
        if float(np.ptp(y)) > float(config["maximum_entity_y_spread_px"]):
            continue
        if float(np.max(radii) / max(np.min(radii), 1e-12)) > float(
            config["maximum_radius_ratio"]
        ):
            continue
        gaps = np.diff(x)
        radius_sums = radii[:-1] + radii[1:]
        minimum_gap_fraction = (
            float(
                config.get(
                    "minimum_two_body_gap_radius_fraction", 0.75
                )
            )
            if len(ids) == 2
            else float(config["minimum_gap_radius_fraction"])
        )
        if np.any(
            gaps < minimum_gap_fraction * radius_sums
        ):
            continue
        structure_score = 0.0
        if len(ids) >= 3:
            striker_separation = gaps[0] / max(radius_sums[0], 1.0)
            target_contact = gaps[-1] / max(radius_sums[-1], 1.0)
            structure_score = (
                5.0 * min(float(striker_separation), 5.0)
                - 10.0
                * abs(
                    math.log(max(float(target_contact), 0.1) / 1.1)
                )
            )
        ranked_sets.append(
            (
                float(
                    np.mean(
                        [value["selection_score"] for value in ordered]
                    )
                    + structure_score
                ),
                ordered,
            )
        )
    if not ranked_sets:
        raise SceneAnalysisError(
            "reference_collision_identity_unmatched",
            "no motion-validated reference candidate set satisfies collision "
            "geometry",
        )
    ranked_sets.sort(
        key=lambda item: (
            -item[0],
            tuple(value["circle"] for value in item[1]),
        )
    )
    best_score, selected = ranked_sets[0]
    second_score = ranked_sets[1][0] if len(ranked_sets) > 1 else None
    second_selected = ranked_sets[1][1] if len(ranked_sets) > 1 else None
    margin = 1.0 if second_score is None else float(best_score - second_score)
    minimum_margin = float(config["minimum_set_score_margin"])
    if margin < minimum_margin:
        raise SceneAnalysisError(
            "reference_collision_identity_ambiguous",
            "reference frame-zero ball set is not unique: "
            f"margin {margin:.6g} < {minimum_margin:.6g}",
        )
    anchors: list[CollisionIdentityAnchor] = []
    for order, (logical_entity_id, record) in enumerate(
        zip(ids, selected), start=1
    ):
        circle = record["circle"]
        mask = _circle_mask(first.shape[:2], circle)
        histogram = _mask_histogram(first, mask)
        mask.setflags(write=False)
        histogram.setflags(write=False)
        anchors.append(
            CollisionIdentityAnchor(
                logical_entity_id=logical_entity_id,
                dataset_object_id=f"object_{order}",
                mask=mask,
                centroid_xy=np.asarray(circle[:2], dtype=np.float64),
                equivalent_radius_px=float(circle[2]),
                histogram=histogram,
                provenance={
                    "policy": "reference_motion_validated_frame_zero_v1",
                    "logical_entity_id": logical_entity_id,
                    "dataset_object_id": f"object_{order}",
                    "motion_validated": True,
                    "circle_xyr": list(circle),
                    "motion_fraction": record["motion_fraction"],
                    "mean_change": record["mean_change"],
                    "ring_change": record["ring_change"],
                    "detector_strength": record["detector_strength"],
                    "local_contrast": record["local_contrast"],
                },
            )
        )
    return tuple(anchors), {
        "backend": "reference_motion_validated_frame_zero_hough_v1",
        "seed_frame": 0,
        "seed_source": "reference_motion_validated_frame_zero_v1",
        "expected_count": len(ids),
        "selected_entity_ids": list(ids),
        "selected_circles_xyr": [
            list(value["circle"]) for value in selected
        ],
        "selected_candidate_diagnostics": [dict(value) for value in selected],
        "candidate_count": len(candidate_records),
        "motion_valid_candidate_count": len(viable),
        "set_score": best_score,
        "second_set_score": second_score,
        "second_set_circles_xyr": (
            [list(value["circle"]) for value in second_selected]
            if second_selected is not None
            else None
        ),
        "set_score_margin": margin,
        "minimum_set_score_margin": minimum_margin,
    }


def build_motion_validated_collision_reference_anchors(
    *,
    entity_ids: Sequence[str],
    reference_frames: Sequence[np.ndarray],
    observation_config: Mapping[str, object],
    localization_config: Mapping[str, object],
) -> tuple[tuple[CollisionIdentityAnchor, ...], dict[str, Any]]:
    """Detect frame-zero candidates and validate them with reference motion."""

    if not reference_frames:
        raise ValueError("reference_frames must not be empty")
    raw: list[tuple[float, float, float, float]] = []
    attempts = []
    for threshold_value in observation_config["hough_accumulator_thresholds"]:
        threshold = float(threshold_value)
        circles, band = _hough_candidates(
            np.asarray(reference_frames[0]),
            config=dict(observation_config),
            accumulator_threshold=threshold,
        )
        raw.extend((*circle, threshold) for circle in circles)
        attempts.append(
            {
                "frame": 0,
                "accumulator_threshold": threshold,
                "candidate_count": len(circles),
                "track_band_y": list(band),
            }
        )
    anchors, metadata = localize_collision_reference_candidates(
        entity_ids=entity_ids,
        reference_frames=reference_frames,
        candidates=raw,
        config=localization_config,
    )
    return anchors, {**metadata, "candidate_attempts": attempts}


def reference_prompts_from_collision_anchors(
    anchors: Sequence[CollisionIdentityAnchor],
    *,
    config: Mapping[str, object],
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Create causal reference prompts from verified frame-zero anchors."""

    values = _validate_anchors(anchors)
    policies = {
        str(value.provenance.get("policy", "unknown")) for value in values
    }
    seed_source = next(iter(policies)) if len(policies) == 1 else "mixed"
    prompts = [
        _circle_prompt(
            (
                float(anchor.centroid_xy[0]),
                float(anchor.centroid_xy[1]),
                float(anchor.equivalent_radius_px),
            ),
            frame_index=0,
            frame_shape=(*anchor.mask.shape, 3),
            expand=float(config["box_expand"]),
            minimum_side=int(config["minimum_box_side"]),
            metadata={
                "entity_id": anchor.logical_entity_id,
                "dataset_object_id": anchor.dataset_object_id,
                "source": seed_source,
                "circle_xyr": [
                    float(anchor.centroid_xy[0]),
                    float(anchor.centroid_xy[1]),
                    float(anchor.equivalent_radius_px),
                ],
                "anchor_provenance": dict(anchor.provenance),
            },
        )
        for anchor in values
    ]
    return prompts, {
        "backend": "collision_frame_zero_anchors_plus_forward_sam2_v1",
        "seed_frame": 0,
        "seed_source": seed_source,
        "expected_count": len(values),
        "selected_entity_ids": [value.logical_entity_id for value in values],
        "selected_dataset_object_ids": [
            value.dataset_object_id for value in values
        ],
        "anchor_provenance": [dict(value.provenance) for value in values],
    }


def assign_collision_frame_zero_candidates(
    anchors: Sequence[CollisionIdentityAnchor],
    *,
    prediction_frame: np.ndarray,
    candidates: Sequence[tuple[float, float, float]],
    config: Mapping[str, object],
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Bind prediction pixels to all verified roles or fail closed."""

    values = _validate_anchors(anchors)
    frame = np.asarray(prediction_frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("prediction_frame must be an HWC BGR image")
    if any(value.mask.shape != frame.shape[:2] for value in values):
        raise ValueError(
            "prediction and reference anchors must share the evaluator canvas"
        )
    circles = tuple(_validated_circle(value) for value in candidates)
    if len(circles) < len(values):
        raise SceneAnalysisError(
            "collision_prediction_frame_zero_entities_missing",
            "prediction frame zero contains fewer credible balls than the "
            f"declared collision entity count {len(values)}",
        )
    edge_details: list[list[dict[str, float]]] = []
    for anchor in values:
        row = []
        for circle in circles:
            candidate_mask = _circle_mask(frame.shape[:2], circle)
            row.append(
                _binding_components(
                    anchor,
                    candidate_histogram=_mask_histogram(
                        frame, candidate_mask
                    ),
                    circle=circle,
                    config=config,
                )
            )
        edge_details.append(row)
    ranked: list[tuple[float, tuple[int, ...]]] = []
    for assignment in itertools.permutations(
        range(len(circles)), len(values)
    ):
        score = float(
            np.mean(
                [
                    edge_details[index][candidate_index]["score"]
                    for index, candidate_index in enumerate(assignment)
                ]
            )
        )
        ranked.append((score, assignment))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_assignment = ranked[0]
    selected_edges = [
        edge_details[index][candidate_index]
        for index, candidate_index in enumerate(best_assignment)
    ]
    if not all(_edge_is_accepted(edge, config=config) for edge in selected_edges):
        raise SceneAnalysisError(
            "collision_prediction_identity_unmatched",
            "prediction frame-zero candidates do not satisfy every verified "
            "collision identity gate",
        )
    second_score = ranked[1][0] if len(ranked) > 1 else None
    assignment_margin = (
        1.0 if second_score is None else float(best_score - second_score)
    )
    minimum_margin = float(config["minimum_assignment_margin"])
    if assignment_margin < minimum_margin:
        raise SceneAnalysisError(
            "collision_prediction_identity_ambiguous",
            "prediction frame-zero global identity assignment is not unique: "
            f"margin {assignment_margin:.6g} < {minimum_margin:.6g}",
        )
    selected_circles = [circles[index] for index in best_assignment]
    prompts = [
        _circle_prompt(
            circle,
            frame_index=0,
            frame_shape=frame.shape,
            expand=float(config["box_expand"]),
            minimum_side=int(config["minimum_box_side"]),
            metadata={
                "entity_id": anchor.logical_entity_id,
                "dataset_object_id": anchor.dataset_object_id,
                "source": "prediction_frame_zero_unique_binding",
                "circle_xyr": list(circle),
                "binding": dict(edge),
            },
        )
        for anchor, circle, edge in zip(values, selected_circles, selected_edges)
    ]
    assigned_indices = set(best_assignment)
    return prompts, {
        "backend": "prediction_frame_zero_unique_global_binding_v1",
        "seed_frame": 0,
        "seed_source": "prediction_frame_zero_unique_binding",
        "expected_count": len(values),
        "candidate_count": len(circles),
        "selected_entity_ids": [value.logical_entity_id for value in values],
        "assigned_candidate_indices": list(best_assignment),
        "assigned_circles_xyr": [list(value) for value in selected_circles],
        "unassigned_circles_xyr": [
            list(circle)
            for index, circle in enumerate(circles)
            if index not in assigned_indices
        ],
        "binding_edges": selected_edges,
        "assignment_score": best_score,
        "second_assignment_score": second_score,
        "assignment_margin": assignment_margin,
        "minimum_assignment_margin": minimum_margin,
        "accepted": True,
    }


def build_prediction_collision_identity_prompts(
    anchors: Sequence[CollisionIdentityAnchor],
    *,
    prediction_frame: np.ndarray,
    observation_config: Mapping[str, object],
    binding_config: Mapping[str, object],
) -> tuple[list[MaskPrompt], dict[str, Any]]:
    """Discover prediction balls only at frame zero, then bind globally."""

    all_circles: list[tuple[float, float, float]] = []
    attempts: list[dict[str, object]] = []
    thresholds = [
        float(value)
        for value in observation_config["hough_accumulator_thresholds"]
    ]
    track_band = (0, int(np.asarray(prediction_frame).shape[0]))
    for threshold in thresholds:
        circles, track_band = _hough_candidates(
            np.asarray(prediction_frame),
            config=dict(observation_config),
            accumulator_threshold=threshold,
        )
        attempts.append(
            {
                "frame": 0,
                "source": "hough_circle",
                "accumulator_threshold": threshold,
                "candidate_count": len(circles),
            }
        )
        all_circles.extend(circles)
    circles = _nms_circles(
        all_circles,
        minimum_center_distance=float(
            observation_config["minimum_circle_center_distance_px"]
        ),
        overlap_distance_factor=float(
            observation_config.get("circle_nms_overlap_distance_factor", 0.55)
        ),
        maximum_candidates=int(observation_config["maximum_candidates"]),
    )
    prompts, metadata = assign_collision_frame_zero_candidates(
        anchors,
        prediction_frame=np.asarray(prediction_frame),
        candidates=circles,
        config=binding_config,
    )
    return prompts, {
        **metadata,
        "track_band_y": list(track_band),
        "candidate_attempts": attempts,
    }


def latch_collision_identity_masks(
    masks: Sequence[Sequence[np.ndarray]],
    prompts: Sequence[MaskPrompt],
    *,
    config: Mapping[str, object],
) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """Terminate a role permanently at its first unconfirmed frame."""

    instances = [list(value) for value in masks]
    prompt_values = list(prompts)
    if len(instances) != len(prompt_values):
        raise ValueError("collision masks and identity prompts must align")
    if not instances:
        return [], {
            "policy": "causal_terminal_identity_latch_v1",
            "roles": [],
        }
    frame_count = len(instances[0])
    if frame_count == 0 or any(
        len(instance) != frame_count for instance in instances
    ):
        raise ValueError("collision identity masks must share a nonempty timeline")
    shape = np.asarray(instances[0][0]).shape
    if len(shape) != 2 or any(
        np.asarray(mask).shape != shape
        for instance in instances
        for mask in instance
    ):
        raise ValueError("collision identity masks must share one 2D canvas")
    if any(int(prompt.frame_index) != 0 for prompt in prompt_values):
        raise ValueError("fail-closed collision identity requires frame-zero prompts")

    minimum_area = int(config["minimum_mask_pixels"])
    maximum_area = int(
        round(
            shape[0]
            * shape[1]
            * float(config["maximum_mask_area_ratio"])
        )
    )
    maximum_jump = float(config["maximum_centroid_jump_px"])
    if minimum_area <= 0 or maximum_area < minimum_area or maximum_jump <= 0.0:
        raise ValueError("collision identity tracking gates are invalid")

    output = [
        [np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)]
        for _ in instances
    ]
    role_diagnostics: list[dict[str, Any]] = []
    for instance, prompt in zip(instances, prompt_values):
        anchor = np.asarray(prompt.points_xy[0], dtype=np.float64)
        terminated_frame = None
        termination_reason = None
        last_confirmed_frame = None
        confirmed_frames = 0
        for frame_index, raw_mask in enumerate(instance):
            if terminated_frame is not None:
                continue
            mask = (np.asarray(raw_mask) > 0).astype(np.uint8) * 255
            area = int(np.count_nonzero(mask))
            if area < minimum_area or area > maximum_area:
                terminated_frame = frame_index
                termination_reason = "mask_missing_or_invalid"
                continue
            moments = cv2.moments(mask, binaryImage=True)
            if abs(float(moments["m00"])) <= 1e-12:
                terminated_frame = frame_index
                termination_reason = "mask_centroid_unavailable"
                continue
            centroid = np.asarray(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=np.float64,
            )
            if float(np.linalg.norm(centroid - anchor)) > maximum_jump:
                terminated_frame = frame_index
                termination_reason = "maximum_identity_jump_exceeded"
                continue
            output[len(role_diagnostics)][frame_index] = mask
            anchor = centroid
            last_confirmed_frame = frame_index
            confirmed_frames += 1
        role_diagnostics.append(
            {
                "entity_id": prompt.metadata.get("entity_id"),
                "confirmed_frames": confirmed_frames,
                "last_confirmed_frame": last_confirmed_frame,
                "terminated_frame": terminated_frame,
                "termination_reason": termination_reason,
            }
        )
    return output, {
        "policy": "causal_terminal_identity_latch_v1",
        "minimum_mask_pixels": minimum_area,
        "maximum_mask_pixels": maximum_area,
        "maximum_centroid_jump_px": maximum_jump,
        "roles": role_diagnostics,
    }


def _validate_anchors(
    anchors: Sequence[CollisionIdentityAnchor],
) -> tuple[CollisionIdentityAnchor, ...]:
    values = tuple(anchors)
    if len(values) < 2:
        raise ValueError("collision requires at least two identity anchors")
    ids = [value.logical_entity_id for value in values]
    objects = [value.dataset_object_id for value in values]
    if len(set(ids)) != len(ids) or len(set(objects)) != len(objects):
        raise ValueError("collision anchor identities must be unique")
    return values


def _validated_circle(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError("collision candidate must be an (x, y, radius) triple")
    circle = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in circle) or circle[2] <= 0.0:
        raise ValueError("collision candidate circle must be finite and positive")
    return circle


def _circle_mask(
    shape: tuple[int, int], circle: tuple[float, float, float]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        tuple(np.rint(circle[:2]).astype(int).tolist()),
        max(1, int(round(0.85 * circle[2]))),
        255,
        -1,
    )
    return mask


def _circle_motion_metrics(
    change: np.ndarray, circle: tuple[float, float, float]
) -> dict[str, float]:
    x, y, radius = circle
    height, width = change.shape
    y0 = max(0, int(math.floor(y - 2.0 * radius)))
    y1 = min(height, int(math.ceil(y + 2.0 * radius + 1.0)))
    x0 = max(0, int(math.floor(x - 2.0 * radius)))
    x1 = min(width, int(math.ceil(x + 2.0 * radius + 1.0)))
    patch = change[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    squared = (xx - x) ** 2 + (yy - y) ** 2
    core = squared <= 0.85 * radius * radius
    ring = (squared >= 1.25 * radius * radius) & (
        squared <= 1.90 * radius * radius
    )
    if not np.any(core) or not np.any(ring):
        return {
            "fraction": 0.0,
            "mean_change": 0.0,
            "ring_change": 0.0,
            "score": 0.0,
        }
    mean_change = float(np.mean(patch[core]))
    fraction = float(np.mean(patch[core] >= 12.0))
    ring_change = float(np.mean(patch[ring]))
    return {
        "fraction": fraction,
        "mean_change": mean_change,
        "ring_change": ring_change,
        "score": mean_change + 35.0 * fraction - 0.25 * ring_change,
    }


def _local_circle_contrast(
    frame: np.ndarray, circle: tuple[float, float, float]
) -> float:
    gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY)
    x, y, radius = circle
    height, width = gray.shape
    y0 = max(0, int(math.floor(y - 2.2 * radius)))
    y1 = min(height, int(math.ceil(y + 2.2 * radius + 1.0)))
    x0 = max(0, int(math.floor(x - 2.2 * radius)))
    x1 = min(width, int(math.ceil(x + 2.2 * radius + 1.0)))
    patch = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    squared = (xx - x) ** 2 + (yy - y) ** 2
    core = squared <= 0.70 * radius * radius
    ring = (squared >= 1.35 * radius * radius) & (
        squared <= 2.10 * radius * radius
    )
    if not np.any(core) or not np.any(ring):
        return 0.0
    return abs(float(np.mean(patch[core])) - float(np.median(patch[ring])))


def _deduplicate_reference_candidates(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for value in sorted(
        records,
        key=lambda item: (
            -float(item["selection_score"]),
            tuple(item["circle"]),
        ),
    ):
        x, y, radius = value["circle"]
        if any(
            math.hypot(x - old["circle"][0], y - old["circle"][1])
            < 0.9 * max(radius, old["circle"][2])
            for old in retained
        ):
            continue
        retained.append(value)
    return sorted(retained, key=lambda item: item["circle"][0])


def _mask_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        return np.zeros(512, dtype=np.float64)
    lab = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2LAB)
    histogram = cv2.calcHist(
        [lab], [0, 1, 2], binary.astype(np.uint8), [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    ).astype(np.float64).ravel()
    total = float(np.sum(histogram))
    return histogram / total if total > 0.0 else histogram


def _histogram_intersection(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("collision identity histograms must share shape")
    return float(np.clip(np.minimum(first, second).sum(), 0.0, 1.0))


def _binding_components(
    anchor: CollisionIdentityAnchor,
    *,
    candidate_histogram: np.ndarray,
    circle: tuple[float, float, float],
    config: Mapping[str, object],
) -> dict[str, float]:
    distance = float(
        np.linalg.norm(
            np.asarray(circle[:2], dtype=np.float64) - anchor.centroid_xy
        )
    )
    distance_scale = max(
        1.0,
        float(config["maximum_center_distance_radii"])
        * anchor.equivalent_radius_px,
    )
    position = 1.0 / (1.0 + (distance / distance_scale) ** 2)
    scale = math.exp(
        -abs(
            math.log(
                circle[2] / max(anchor.equivalent_radius_px, 1e-12)
            )
        )
    )
    appearance = _histogram_intersection(
        anchor.histogram, candidate_histogram
    )
    weights = config["weights"]
    if not isinstance(weights, Mapping):
        raise TypeError("collision identity weights must be an object")
    denominator = sum(float(value) for value in weights.values())
    if denominator <= 0.0:
        raise ValueError("collision identity weights must sum positive")
    score = (
        float(weights["position"]) * position
        + float(weights["scale"]) * scale
        + float(weights["appearance"]) * appearance
    ) / denominator
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "position": float(np.clip(position, 0.0, 1.0)),
        "scale": float(np.clip(scale, 0.0, 1.0)),
        "appearance": float(np.clip(appearance, 0.0, 1.0)),
        "center_distance_px": distance,
    }


def _edge_is_accepted(
    edge: Mapping[str, float], *, config: Mapping[str, object]
) -> bool:
    return bool(
        edge["position"] >= float(config["minimum_position_similarity"])
        and edge["scale"] >= float(config["minimum_scale_similarity"])
        and edge["appearance"]
        >= float(config["minimum_appearance_similarity"])
        and edge["score"] >= float(config["minimum_binding_score"])
    )


__all__ = [
    "CollisionIdentityAnchor",
    "assign_collision_frame_zero_candidates",
    "build_motion_validated_collision_reference_anchors",
    "build_prediction_collision_identity_prompts",
    "latch_collision_identity_masks",
    "load_collision_identity_anchors",
    "localize_collision_reference_candidates",
    "reference_prompts_from_collision_anchors",
]
