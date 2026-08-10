"""Fail-closed subject identity helpers for ``pendulum_state_v8``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ....io import load_json, sha256_file
from ...common.entities import (
    EntitySpec,
    FrozenAssignment,
    OpenWorldObservation,
)
from ...common.errors import ReferenceAnalysisError
from ...contracts import CaseEvaluationRequest
from .open_world import PendulumStructureSpec


@dataclass(frozen=True)
class PendulumSubjectAnchor:
    case_id: str
    entity_id: str
    entity_class: str
    manifest_path: Path
    npz_path: Path
    source_mask: np.ndarray
    mask: np.ndarray
    centroid_xy: np.ndarray
    area_px2: float
    equivalent_radius_px: float
    provenance: Mapping[str, object]


class SubjectIdentityState(str, Enum):
    CONFIRMED = "confirmed"
    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"
    REFERENCE_INVALID = "reference_invalid"


@dataclass(frozen=True)
class SubjectIdentityDecision:
    state: SubjectIdentityState
    entity_id: str
    track_id: str | None
    reason_code: str | None
    candidates: tuple[Mapping[str, object], ...]
    winning_margin: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "reason_code": self.reason_code,
            "candidates": [dict(candidate) for candidate in self.candidates],
            "winning_margin": self.winning_margin,
        }


def transform_evaluator_mask(
    mask: np.ndarray,
    spatial_transform: Mapping[str, object],
) -> np.ndarray:
    source = np.asarray(mask)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("evaluator mask must be a non-empty 2D array")
    source_binary = np.where(source > 0, 255, 0).astype(np.uint8)
    source_height, source_width = source_binary.shape
    declared_source = _integer_pair(
        spatial_transform.get("source_size"),
        name="source_size",
    )
    if declared_source != (source_width, source_height):
        raise ValueError(
            "mask shape differs from spatial transform source_size"
        )
    target_width, target_height = _integer_pair(
        spatial_transform.get("target_size"),
        name="target_size",
    )
    policy = spatial_transform.get("policy")
    if policy == "reference_content_crop_resize_no_pad":
        crop = _integer_quad(
            spatial_transform.get("crop_xywh"),
            name="crop_xywh",
        )
        x, y, crop_width, crop_height = crop
        if (
            x < 0
            or y < 0
            or crop_width <= 0
            or crop_height <= 0
            or x + crop_width > source_width
            or y + crop_height > source_height
            or crop_width * target_height
            != crop_height * target_width
        ):
            raise ValueError("invalid no-pad mask crop")
        scale = _finite_positive(
            spatial_transform.get("scale"),
            name="scale",
        )
        if not math.isclose(
            scale,
            target_width / crop_width,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("no-pad mask scale differs from crop geometry")
        cropped = source_binary[y : y + crop_height, x : x + crop_width]
        output = cv2.resize(
            cropped,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    elif policy == "preserve_aspect_ratio_letterbox":
        scale = _finite_positive(
            spatial_transform.get("scale"),
            name="scale",
        )
        offset_x, offset_y = _integer_pair(
            spatial_transform.get("offset_xy"),
            name="offset_xy",
            allow_zero=True,
        )
        expected_scale = min(
            target_width / source_width,
            target_height / source_height,
        )
        if not math.isclose(
            scale,
            expected_scale,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "letterbox mask scale differs from source/target geometry"
            )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        expected_offset = (
            (target_width - resized_width) // 2,
            (target_height - resized_height) // 2,
        )
        if (offset_x, offset_y) != expected_offset:
            raise ValueError(
                "letterbox mask offset differs from centered geometry"
            )
        if (
            offset_x < 0
            or offset_y < 0
            or offset_x + resized_width > target_width
            or offset_y + resized_height > target_height
        ):
            raise ValueError("letterbox mask placement exceeds target canvas")
        resized = cv2.resize(
            source_binary,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        output = np.zeros((target_height, target_width), dtype=np.uint8)
        output[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
    else:
        raise ValueError(f"unsupported mask spatial policy: {policy!r}")
    return np.where(output > 0, 255, 0).astype(np.uint8)


def load_pendulum_subject_anchor(
    request: CaseEvaluationRequest,
    *,
    entity_id: str,
    spatial_transform: Mapping[str, object],
) -> PendulumSubjectAnchor:
    root = request.asset_root.resolve()
    manifest_value = request.case.get("assets", {}).get(
        "first_frame_mask_manifest"
    )
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_manifest_missing_v8",
            "Case does not declare assets.first_frame_mask_manifest",
        )
    manifest_path = _resolve_asset(
        root,
        manifest_value,
        path_escape_code=(
            "reference_pendulum_subject_mask_manifest_path_escape_v8"
        ),
        label="pendulum subject mask manifest",
    )
    if not manifest_path.is_file():
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_manifest_missing_v8",
            f"pendulum subject mask manifest does not exist: {manifest_path}",
        )
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_manifest_invalid_v8",
            "cannot parse pendulum subject mask manifest: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        instance, image_shape = _validate_manifest(
            manifest,
            request=request,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_manifest_invalid_v8",
            "pendulum subject mask manifest violates schema 1.2: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    npz_path = _resolve_asset(
        root,
        instance.get("npz_asset"),
        path_escape_code="reference_pendulum_subject_mask_path_escape_v8",
        label="pendulum subject mask NPZ",
    )
    if not npz_path.is_file():
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_missing_v8",
            f"pendulum subject mask NPZ does not exist: {npz_path}",
        )
    try:
        source_mask = _load_subject_npz(
            npz_path,
            instance=instance,
            entity_id=entity_id,
            image_shape=image_shape,
        )
        _validate_geometry(source_mask, instance=instance)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_invalid_v8",
            "pendulum subject mask NPZ violates the per-object contract: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        transformed = transform_evaluator_mask(
            source_mask,
            spatial_transform,
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_transform_invalid_v8",
            "pendulum subject mask cannot use the reference transform: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    ys, xs = np.where(transformed > 0)
    if xs.size == 0:
        raise ReferenceAnalysisError(
            "reference_pendulum_subject_mask_transform_invalid_v8",
            "pendulum subject mask is empty after the reference transform",
        )
    centroid = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
    source_mask.setflags(write=False)
    transformed.setflags(write=False)
    area = float(xs.size)
    return PendulumSubjectAnchor(
        case_id=str(request.case["case_id"]),
        entity_id=str(entity_id),
        entity_class="pendulum_bob",
        manifest_path=manifest_path,
        npz_path=npz_path,
        source_mask=source_mask,
        mask=transformed,
        centroid_xy=centroid,
        area_px2=area,
        equivalent_radius_px=float(math.sqrt(area / math.pi)),
        provenance={
            "policy": "frozen_dataset_subject_annotation_v1",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "mask_id": str(instance["mask_id"]),
            "physics_object_id": str(instance["object_id"]),
            "source_shape_hw": list(source_mask.shape),
            "target_shape_hw": list(transformed.shape),
            "spatial_transform": dict(spatial_transform),
        },
    )


def decide_pendulum_identity_v8(
    *,
    entity: EntitySpec,
    observation: OpenWorldObservation,
    anchor: PendulumSubjectAnchor,
    structure: PendulumStructureSpec,
    anchor_frame: np.ndarray,
    observed_frame: np.ndarray,
    config: Mapping[str, Any],
    reference: bool,
) -> SubjectIdentityDecision:
    """Confirm one frame-zero bob track or fail closed without fallback."""

    if entity.entity_id != anchor.entity_id:
        raise ValueError("entity and frozen subject anchor IDs differ")
    if entity.entity_class != anchor.entity_class:
        raise ValueError("entity and frozen subject anchor classes differ")
    anchor_mask = np.asarray(anchor.mask) > 0
    if (
        anchor_mask.ndim != 2
        or not np.any(anchor_mask)
        or structure.bob_mask.shape != anchor_mask.shape
    ):
        raise ValueError("identity inputs do not share one non-empty canvas")
    anchor_image = _identity_frame(
        anchor_frame,
        mask_shape=anchor_mask.shape,
        name="anchor_frame",
    )
    observed_image = _identity_frame(
        observed_frame,
        mask_shape=anchor_mask.shape,
        name="observed_frame",
    )

    minimum_iou = _unit_interval(
        config.get("minimum_anchor_iou", 0.10),
        name="minimum_anchor_iou",
    )
    maximum_distance = _finite_positive(
        config.get("maximum_center_distance_radii", 1.75),
        name="maximum_center_distance_radii",
    )
    minimum_area = _finite_positive(
        config.get("minimum_area_ratio", 0.35),
        name="minimum_area_ratio",
    )
    maximum_area = _finite_positive(
        config.get("maximum_area_ratio", 2.5),
        name="maximum_area_ratio",
    )
    minimum_length = _finite_positive(
        config.get("minimum_length_ratio", 0.60),
        name="minimum_length_ratio",
    )
    maximum_length = _finite_positive(
        config.get("maximum_length_ratio", 1.35),
        name="maximum_length_ratio",
    )
    if minimum_area > maximum_area:
        raise ValueError("minimum_area_ratio exceeds maximum_area_ratio")
    if minimum_length > maximum_length:
        raise ValueError("minimum_length_ratio exceeds maximum_length_ratio")
    minimum_color = _unit_interval(
        config.get("minimum_color_similarity", 0.20),
        name="minimum_color_similarity",
    )
    minimum_score = _unit_interval(
        config.get("minimum_score", 0.45),
        name="minimum_score",
    )
    ambiguity_margin = _unit_interval(
        config.get("ambiguity_margin", 0.05),
        name="ambiguity_margin",
    )
    anchor_histogram = _lab_mask_histogram(anchor_image, anchor_mask)

    audited: list[dict[str, object]] = []
    for track in observation.tracks:
        for detection in track.detections:
            if detection.frame_index != 0:
                continue
            audited.append(
                _audit_identity_candidate(
                    track_id=track.track_id,
                    track_confirmed=track.confirmed,
                    detection=detection,
                    entity=entity,
                    anchor=anchor,
                    anchor_mask=anchor_mask,
                    anchor_histogram=anchor_histogram,
                    observed_frame=observed_image,
                    minimum_iou=minimum_iou,
                    maximum_distance=maximum_distance,
                    minimum_area=minimum_area,
                    maximum_area=maximum_area,
                    minimum_length=minimum_length,
                    maximum_length=maximum_length,
                    minimum_color=minimum_color,
                    minimum_score=minimum_score,
                )
            )

    eligible = [candidate for candidate in audited if candidate["eligible"]]
    eligible.sort(
        key=lambda candidate: (
            float(candidate["score"]),
            float(candidate["anchor_iou"]),
            float(candidate["color_similarity"]),
            str(candidate["track_id"]),
        ),
        reverse=True,
    )
    if not eligible:
        return _unconfirmed_identity_decision(
            entity_id=entity.entity_id,
            candidates=audited,
            reference=reference,
            ambiguous=False,
            winning_margin=None,
        )

    winner = eligible[0]
    distinct = [
        candidate
        for candidate in eligible[1:]
        if _identity_candidates_are_distinct(winner, candidate)
    ]
    runner_score = float(distinct[0]["score"]) if distinct else 0.0
    winning_margin = float(winner["score"]) - runner_score
    if distinct and winning_margin < ambiguity_margin:
        return _unconfirmed_identity_decision(
            entity_id=entity.entity_id,
            candidates=audited,
            reference=reference,
            ambiguous=True,
            winning_margin=winning_margin,
        )
    return SubjectIdentityDecision(
        state=SubjectIdentityState.CONFIRMED,
        entity_id=entity.entity_id,
        track_id=str(winner["track_id"]),
        reason_code=None,
        candidates=tuple(audited),
        winning_margin=winning_margin,
    )


def identity_decision_to_frozen_assignment(
    decision: SubjectIdentityDecision,
    *,
    entity_id: str,
    observation: OpenWorldObservation,
) -> FrozenAssignment | None:
    """Convert only a confirmed identity into one immutable track mapping."""

    if decision.state is not SubjectIdentityState.CONFIRMED:
        return None
    if decision.entity_id != entity_id or decision.track_id is None:
        raise ValueError("confirmed identity decision violates its entity contract")
    track_ids = tuple(track.track_id for track in observation.tracks)
    if decision.track_id not in track_ids:
        raise ValueError("confirmed identity track is absent from the observation")
    return FrozenAssignment(
        entity_to_track={entity_id: decision.track_id},
        residual_track_ids=tuple(
            track_id
            for track_id in track_ids
            if track_id != decision.track_id
        ),
        unmatched_entity_ids=(),
        total_cost=0.0,
    )


def _audit_identity_candidate(
    *,
    track_id: str,
    track_confirmed: bool,
    detection: Any,
    entity: EntitySpec,
    anchor: PendulumSubjectAnchor,
    anchor_mask: np.ndarray,
    anchor_histogram: np.ndarray,
    observed_frame: np.ndarray,
    minimum_iou: float,
    maximum_distance: float,
    minimum_area: float,
    maximum_area: float,
    minimum_length: float,
    maximum_length: float,
    minimum_color: float,
    minimum_score: float,
) -> dict[str, object]:
    base: dict[str, object] = {
        "track_id": track_id,
        "track_confirmed": bool(track_confirmed),
        "detection_id": detection.detection_id,
        "entity_class": detection.entity_class,
        "sources": list(detection.sources),
        "confidence": float(detection.confidence),
        "center_xy": detection.xy.tolist(),
        "area_px2": float(detection.area_px2),
    }
    mask = detection.mask
    if mask is None or np.asarray(mask).shape != anchor_mask.shape:
        return {
            **base,
            "eligible": False,
            "malformed": True,
            "malformed_reason": "missing or canvas-incompatible mask",
            "gates": _all_identity_gates(False),
        }
    candidate_mask = np.asarray(mask) > 0
    candidate_area = int(np.count_nonzero(candidate_mask))
    if candidate_area == 0:
        return {
            **base,
            "eligible": False,
            "malformed": True,
            "malformed_reason": "empty candidate mask",
            "gates": _all_identity_gates(False),
        }
    intersection = int(np.count_nonzero(anchor_mask & candidate_mask))
    union = int(np.count_nonzero(anchor_mask | candidate_mask))
    anchor_iou = float(intersection / max(union, 1))
    candidate_radius = float(math.sqrt(candidate_area / math.pi))
    distance_radii = float(
        np.linalg.norm(detection.xy - anchor.centroid_xy)
        / max(anchor.equivalent_radius_px, candidate_radius, 1.0)
    )
    area_ratio = float(candidate_area / anchor.area_px2)
    raw_length = detection.metadata.get("pivot_distance_ratio")
    length_ratio = (
        float(raw_length)
        if isinstance(raw_length, (int, float))
        and not isinstance(raw_length, bool)
        and math.isfinite(float(raw_length))
        else math.nan
    )
    color_similarity = _histogram_intersection(
        anchor_histogram,
        _lab_mask_histogram(observed_frame, candidate_mask),
    )
    center_score = math.exp(-distance_radii)
    area_score = math.exp(-abs(math.log(max(area_ratio, 1e-12))))
    length_score = (
        math.exp(-abs(math.log(max(length_ratio, 1e-12))))
        if math.isfinite(length_ratio) and length_ratio > 0.0
        else 0.0
    )
    score = float(
        0.30 * anchor_iou
        + 0.15 * center_score
        + 0.15 * area_score
        + 0.15 * length_score
        + 0.15 * color_similarity
        + 0.10 * float(detection.confidence)
    )
    gates = {
        "entity_class": detection.entity_class == entity.entity_class,
        "identity_anchor_valid": (
            detection.metadata.get("identity_anchor_valid") is not False
        ),
        "directed_sam_source": (
            "condition_directed_sam2" in detection.sources
        ),
        "anchor_iou": anchor_iou >= minimum_iou,
        "center_distance": distance_radii <= maximum_distance,
        "area_ratio": minimum_area <= area_ratio <= maximum_area,
        "length_ratio": (
            math.isfinite(length_ratio)
            and minimum_length <= length_ratio <= maximum_length
        ),
        "color_similarity": color_similarity >= minimum_color,
        "minimum_score": score >= minimum_score,
    }
    return {
        **base,
        "eligible": all(gates.values()),
        "malformed": False,
        "anchor_iou": anchor_iou,
        "center_distance_radii": distance_radii,
        "area_ratio": area_ratio,
        "pivot_distance_ratio": (
            length_ratio if math.isfinite(length_ratio) else None
        ),
        "color_similarity": color_similarity,
        "candidate_equivalent_radius_px": candidate_radius,
        "score": score,
        "gates": gates,
    }


def _all_identity_gates(value: bool) -> dict[str, bool]:
    return {
        name: value
        for name in (
            "entity_class",
            "identity_anchor_valid",
            "directed_sam_source",
            "anchor_iou",
            "center_distance",
            "area_ratio",
            "length_ratio",
            "color_similarity",
            "minimum_score",
        )
    }


def _unconfirmed_identity_decision(
    *,
    entity_id: str,
    candidates: list[Mapping[str, object]],
    reference: bool,
    ambiguous: bool,
    winning_margin: float | None,
) -> SubjectIdentityDecision:
    if reference:
        state = SubjectIdentityState.REFERENCE_INVALID
        reason = "reference_pendulum_identity_unconfirmed_v8"
    elif ambiguous:
        state = SubjectIdentityState.AMBIGUOUS
        reason = "prediction_pendulum_identity_ambiguous_v8"
    else:
        state = SubjectIdentityState.ABSENT
        reason = "prediction_pendulum_identity_absent_v8"
    return SubjectIdentityDecision(
        state=state,
        entity_id=entity_id,
        track_id=None,
        reason_code=reason,
        candidates=tuple(candidates),
        winning_margin=winning_margin,
    )


def _identity_candidates_are_distinct(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    distance = float(
        np.linalg.norm(
            np.asarray(first["center_xy"], dtype=np.float64)
            - np.asarray(second["center_xy"], dtype=np.float64)
        )
    )
    scale = max(
        float(first["candidate_equivalent_radius_px"]),
        float(second["candidate_equivalent_radius_px"]),
        1.0,
    )
    return distance > 0.75 * scale


def _identity_frame(
    value: np.ndarray,
    *,
    mask_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    frame = np.asarray(value)
    if (
        frame.shape != (*mask_shape, 3)
        or frame.dtype != np.uint8
    ):
        raise ValueError(f"{name} must be uint8 BGR on the identity canvas")
    return frame


def _lab_mask_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(512, dtype=np.float64)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    histogram = cv2.calcHist(
        [lab],
        [0, 1, 2],
        binary,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    ).reshape(-1)
    total = float(histogram.sum())
    if total <= 0.0:
        return np.zeros(512, dtype=np.float64)
    return histogram.astype(np.float64) / total


def _histogram_intersection(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(np.minimum(first, second).sum())


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or not 0.0 <= output <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return output


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return output


def _integer_pair(
    value: object,
    *,
    name: str,
    allow_zero: bool = False,
) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must contain two integers")
    output = (int(value[0]), int(value[1]))
    minimum = 0 if allow_zero else 1
    if min(output) < minimum:
        raise ValueError(f"{name} entries must be >= {minimum}")
    return output


def _integer_quad(value: object, *, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must contain four integers")
    return tuple(int(item) for item in value)


def _resolve_asset(
    root: Path,
    value: object,
    *,
    path_escape_code: str,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(
            path_escape_code,
            f"{label} must be a non-empty relative path",
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReferenceAnalysisError(
            path_escape_code,
            f"{label} escapes the Dataset root: {value}",
        )
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            path_escape_code,
            f"{label} escapes the Dataset root: {value}",
        ) from exc
    return path


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    request: CaseEvaluationRequest,
) -> tuple[Mapping[str, Any], tuple[int, int]]:
    if manifest.get("schema_version") != "1.2":
        raise ValueError("schema_version must be '1.2'")
    if manifest.get("case_id") != request.case.get("case_id"):
        raise ValueError("case_id differs from the evaluated Case")
    if manifest.get("scene_id") != "pendulum":
        raise ValueError("scene_id must be 'pendulum'")
    if manifest.get("frame_index") != 0:
        raise ValueError("frame_index must be zero")
    if manifest.get("frame_scope") != "first_frame_only":
        raise ValueError("frame_scope must be 'first_frame_only'")
    if manifest.get("source_first_frame") != request.case.get("assets", {}).get(
        "first_frame"
    ):
        raise ValueError("source_first_frame differs from the Case asset")
    image_shape = _integer_pair(
        manifest.get("image_shape_hw"),
        name="image_shape_hw",
    )
    storage = manifest.get("storage")
    if not isinstance(storage, Mapping):
        raise TypeError("storage must be an object")
    model = storage.get("model")
    if not isinstance(model, Mapping) or (
        model.get("array_key") != "masks"
        or model.get("layout") != "1HW"
        or model.get("dtype") != "uint8"
        or model.get("values") != [0, 1]
    ):
        raise ValueError("storage.model differs from the per-object contract")
    instances = manifest.get("instances")
    if not isinstance(instances, list):
        raise TypeError("instances must be a list")
    bobs = [
        item
        for item in instances
        if isinstance(item, Mapping)
        and item.get("entity_class") == "pendulum_bob"
    ]
    if len(bobs) != 1:
        raise ValueError("manifest must declare exactly one pendulum_bob")
    instance = bobs[0]
    for key in (
        "mask_id",
        "object_id",
        "npz_asset",
        "area_pixels",
        "bbox_xyxy",
        "centroid_xy",
    ):
        if key not in instance:
            raise KeyError(key)
    if not isinstance(instance["mask_id"], str) or not instance["mask_id"]:
        raise TypeError("mask_id must be a non-empty string")
    if not isinstance(instance["object_id"], str) or not instance["object_id"]:
        raise TypeError("object_id must be a non-empty string")
    return instance, image_shape


def _load_subject_npz(
    path: Path,
    *,
    instance: Mapping[str, Any],
    entity_id: str,
    image_shape: tuple[int, int],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "masks",
            "mask_ids",
            "object_ids",
            "frame_index",
        }:
            raise ValueError("NPZ keys differ from the per-object contract")
        masks = payload["masks"]
        mask_ids = payload["mask_ids"]
        object_ids = payload["object_ids"]
        frame_index = payload["frame_index"]
        if masks.dtype != np.uint8 or masks.shape != (1, *image_shape):
            raise ValueError("masks must be uint8 [1,H,W]")
        values = set(int(value) for value in np.unique(masks))
        if not values <= {0, 1}:
            raise ValueError("masks values must be binary {0,1}")
        if (
            mask_ids.shape != (1,)
            or mask_ids.dtype.kind != "U"
            or str(mask_ids[0]) != instance["mask_id"]
        ):
            raise ValueError("mask_ids does not identify this mask")
        if (
            object_ids.shape != (1,)
            or object_ids.dtype.kind != "U"
            or str(object_ids[0]) != entity_id
        ):
            raise ValueError("object_ids does not identify the declared entity")
        if (
            frame_index.shape != ()
            or frame_index.dtype != np.int64
            or int(frame_index) != 0
        ):
            raise ValueError("frame_index must be int64 scalar zero")
        output = np.array(masks[0], copy=True)
    if not np.any(output):
        raise ValueError("subject mask must not be empty")
    return output


def _validate_geometry(
    mask: np.ndarray,
    *,
    instance: Mapping[str, Any],
) -> None:
    ys, xs = np.where(mask > 0)
    area = int(xs.size)
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]
    centroid = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
    if instance["area_pixels"] != area:
        raise ValueError("area_pixels differs from the NPZ mask")
    if instance["bbox_xyxy"] != bbox:
        raise ValueError("bbox_xyxy differs from the NPZ mask")
    declared_centroid = np.asarray(instance["centroid_xy"], dtype=np.float64)
    if declared_centroid.shape != (2,) or not np.allclose(
        centroid,
        declared_centroid,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("centroid_xy differs from the NPZ mask")


__all__ = [
    "PendulumSubjectAnchor",
    "SubjectIdentityDecision",
    "SubjectIdentityState",
    "decide_pendulum_identity_v8",
    "identity_decision_to_frozen_assignment",
    "load_pendulum_subject_anchor",
    "transform_evaluator_mask",
]
