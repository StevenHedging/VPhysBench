from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file, write_json
from ...common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ...common.artifacts.open_world_v2 import (
    write_open_world_v2_artifacts,
)
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.entities import (
    ExpectedPositionSample,
    ObjectCentricComparisonV2,
    PositionEvidence,
    ReferenceCapability,
    build_common_time_grid,
    compose_object_centric_v2,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
    safe_compare_open_world_v2,
)
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.subject import (
    SubjectComparison,
    compare_subjects,
    infer_reference_mode,
    subject_weights,
    write_subject_artifacts,
)
from ...contracts import CaseEvaluationRequest
from .open_world import (
    CircularOpenWorldObservation,
    empty_open_world_observation,
    freeze_reference_identities,
    freeze_prediction_identity,
    identity_anchors_from_observation,
    matched_prediction_inputs,
    observe_circular_objects,
    polar_position_similarity,
    reference_instance_tracks,
    score_open_world_orbits,
)


class CircularMotionOpenWorldCaseEvaluator(ReferenceCaseEvaluator):
    """Shadow-v6 circular evaluator with open-world object accounting."""

    evaluator_id = "uniform_circular_motion_open_world"
    evaluator_version = "2.0"
    sequential_evaluator_version = "2.0"
    robust_evaluator_version = "2.0"
    scene_id = "uniform_circular_motion"
    primary_score = "uniform_circular_motion_open_world_similarity"
    allow_partial_prediction = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
                "entity_contract": "manifest_v1",
                "observer": "open_world_v2_adapter",
                "distance": "relative_polar_v1",
                "composition": "integrity_gate_times_content_v1",
            }
        )

    def describe_observation(self) -> dict[str, Any]:
        assignment = self.config.get("object_centric_scoring", {}).get(
            "assignment", {}
        )
        return {
            "subject": "manifest_declared_orbiters_plus_all_disk_residuals",
            "segmentation": "green_disk_interior_color_contrast_components",
            "apparatus_exclusion": "eroded_green_disk",
            "candidate_policy": (
                "retain_all_candidates_up_to_safety_cap_then_overflow_exposure"
            ),
            "identity": (
                "condition_or_initial_window_appearance_radius_phase_anchors_"
                "plus_causal_tracks"
            ),
            "assignment": {
                "method": "evidence_tiered_hungarian_with_null",
                "minimum_match_position_similarity": float(
                    assignment.get(
                        "minimum_match_position_similarity", 0.1
                    )
                ),
            },
            "coordinate_system": (
                "per_video_disk_normalized_relative_polar_with_"
                "cartesian_center_fallback"
            ),
            "lifecycle": "persistent_orbiters",
        }

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        try:
            manifest = materialize_entity_manifest(request.case)
            entities = _ordered_entities(manifest)
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_invalid",
                f"circular entity manifest is invalid: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        time_grid = build_common_time_grid(times_s)
        observation_config = self.config["color_observation"]
        try:
            reference_observation = observe_circular_objects(
                reference_video.frames,
                time_grid=time_grid,
                config=observation_config,
            )
            minimum_disk_ratio = float(
                observation_config.get(
                    "open_world_minimum_reference_disk_valid_ratio", 0.8
                )
            )
            if (
                float(
                    reference_observation.diagnostics[
                        "disk_valid_frame_ratio"
                    ]
                )
                < minimum_disk_ratio
            ):
                raise ValueError(
                    "reference rotating-platform coverage is below "
                    f"{minimum_disk_ratio:g}"
                )
            frozen_reference = freeze_reference_identities(
                reference_observation,
                entities=entities,
                time_grid=time_grid,
                minimum_valid_ratio=float(
                    observation_config.get(
                        "open_world_minimum_reference_track_ratio", 0.5
                    )
                ),
            )
            reference_tracks = reference_instance_tracks(frozen_reference)
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_circular_open_world_observation_failed",
                f"reference circular observation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        reference_mode = infer_reference_mode(request.case)
        condition_frame: np.ndarray | None = None
        condition_mask: np.ndarray | None = None
        condition_observation: CircularOpenWorldObservation | None = None
        condition_provenance: dict[str, Any] = {}
        try:
            if (
                manifest.reference_capability
                is ReferenceCapability.SAME_CASE_GT
            ):
                condition_anchors = frozen_reference.anchors
            else:
                condition_frame, condition_provenance = (
                    _load_condition_frame(
                        request,
                        target_shape=reference_video.frames[0].shape,
                    )
                )
                condition_grid = build_common_time_grid([0.0])
                condition_observation = observe_circular_objects(
                    [condition_frame],
                    time_grid=condition_grid,
                    config=observation_config,
                )
                condition_anchors = identity_anchors_from_observation(
                    condition_observation,
                    entities=entities,
                )
                condition_mask = condition_observation.union_masks[0]
                condition_provenance.update(
                    {
                        "condition_mask_source": (
                            "condition_frame_disk_interior_objects"
                        ),
                        "condition_identity_anchors": list(
                            condition_anchors
                        ),
                        "condition_observation": dict(
                            condition_observation.diagnostics
                        ),
                    }
                )
            expected_timelines = _expected_timelines(
                entities=entities,
                frozen_reference=frozen_reference,
                condition_observation=condition_observation,
                condition_anchors=condition_anchors,
                capability=manifest.reference_capability,
                time_grid=time_grid,
            )
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_circular_expected_timeline_failed",
                f"circular expected timeline failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        (
            prediction_frames,
            prediction_available,
            timeline_failure,
        ) = _normalized_prediction_timeline(
            getattr(prediction_video, "frames", []),
            available=getattr(prediction_video, "available", None),
            frame_count=len(times_s),
            frame_shape=reference_video.frames[0].shape,
        )
        prediction_failures: list[dict[str, str]] = []
        if timeline_failure is not None:
            prediction_failures.append(timeline_failure)
        prediction_observation: CircularOpenWorldObservation | None = None
        try:
            prediction_observation = observe_circular_objects(
                prediction_frames,
                time_grid=time_grid,
                config=observation_config,
                available=prediction_available,
            )
            prediction_objects = prediction_observation.objects
            prediction_union = list(prediction_observation.union_masks)
            prediction_observation_diagnostics = dict(
                prediction_observation.diagnostics
            )
        except Exception as exc:
            failure = {
                "code": "prediction_circular_open_world_observation_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            prediction_objects = empty_open_world_observation(
                len(times_s), code=failure["code"], reason=failure["reason"]
            )
            shape = reference_video.frames[0].shape[:2]
            prediction_union = [
                np.zeros(shape, dtype=np.uint8) for _ in times_s
            ]
            prediction_observation_diagnostics = dict(
                prediction_objects.diagnostics
            )

        assignment_config = self.config.get(
            "object_centric_scoring", {}
        ).get("assignment", {})
        try:
            frozen_prediction = freeze_prediction_identity(
                prediction_objects,
                entities=entities,
                condition_anchors=condition_anchors,
                initial_window_frames=int(
                    assignment_config.get("initial_window_frames", 3)
                ),
                maximum_match_cost=float(
                    assignment_config.get(
                        "maximum_condition_identity_cost", 3.0
                    )
                ),
            )
            polar_config = {
                **self.config["scoring"],
                **self.config.get("object_centric_scoring", {}).get(
                    "polar_distance", {}
                ),
            }

            def position_callback(
                timeline,
                detection,
                frame_index,
                _frame_diagonal_px,
            ):
                value = polar_position_similarity(
                    timeline.reference_xy[frame_index],
                    detection.xy,
                    center_fallback_radius_ratio=float(
                        polar_config.get(
                            "center_fallback_radius_ratio", 0.08
                        )
                    ),
                    radial_scale_ratio=float(
                        polar_config.get(
                            "polar_radial_scale_ratio", 0.15
                        )
                    ),
                    angular_scale_rad=float(
                        polar_config.get(
                            "polar_angular_scale_rad", 0.35
                        )
                    ),
                    cartesian_scale_ratio=float(
                        polar_config.get(
                            "center_cartesian_scale_ratio", 0.15
                        )
                    ),
                )
                return PositionEvidence(
                    score=float(value["score"]),
                    normalized_distance=float(
                        value["normalized_distance"]
                    ),
                    diagnostics={"mode": str(value["mode"])},
                )

            comparison = safe_compare_open_world_v2(
                expected_timelines=expected_timelines,
                prediction_factory=lambda: prediction_objects,
                time_grid=time_grid,
                frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
                frozen_identity=frozen_prediction,
                minimum_match_position_similarity=float(
                    assignment_config.get(
                        "minimum_match_position_similarity", 0.1
                    )
                ),
                position_similarity_callback=position_callback,
            )
            if comparison.failed:
                prediction_failures.append(
                    {
                        "code": (
                            "prediction_circular_open_world_v2_failed_closed"
                        ),
                        "reason": str(comparison.failure_reason),
                    }
                )
        except Exception as exc:
            failure = {
                "code": "prediction_circular_open_world_comparison_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            prediction_objects = empty_open_world_observation(
                len(times_s), code=failure["code"], reason=failure["reason"]
            )
            frozen_prediction = freeze_prediction_identity(
                prediction_objects,
                entities=entities,
                condition_anchors=condition_anchors,
            )
            comparison = safe_compare_open_world_v2(
                expected_timelines=expected_timelines,
                prediction_factory=lambda: (_ for _ in ()).throw(exc),
                time_grid=time_grid,
                frame_diagonal_px=2.0 * math.sqrt(2.0) * 100.0,
                frozen_identity=frozen_prediction,
            )

        prediction_tracks, matched_prediction_union = (
            matched_prediction_inputs(
                reference=frozen_reference,
                prediction=prediction_objects,
                matches=comparison.matches,
                frame_count=len(times_s),
            )
        )
        try:
            orbit_score = score_open_world_orbits(
                reference_tracks,
                prediction_tracks,
                matches=comparison.matches,
                reference_object_tracks=frozen_reference.tracks,
                prediction_observation=prediction_objects,
                times_s=times_s,
                time_grid=time_grid,
                config={
                    **self.config["scoring"],
                    **self.config.get("object_centric_scoring", {}).get(
                        "polar_distance", {}
                    ),
                },
            )
        except Exception as exc:
            failure = {
                "code": "prediction_circular_orbit_scoring_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            orbit_score = _degraded_orbit_score(failure)

        reference_union = reference_tracks.union_masks
        union_ious = [
            observed_mask_iou(reference_union[index], prediction_union[index])
            for index in range(len(times_s))
        ]
        try:
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=reference_union,
                prediction_masks=matched_prediction_union,
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
                condition_frame=condition_frame,
                condition_mask=condition_mask,
            )
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            failure = {
                "code": "prediction_circular_subject_comparison_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            subject = _degraded_subject_comparison(
                len(times_s), reference_mode=reference_mode
            )

        content_components, content_weights = _content_contract(
            orbit_score=orbit_score,
            subject=subject,
            reference_mode=reference_mode,
            config=self.config.get("object_centric_scoring", {}),
        )
        composition = compose_object_centric_v2(
            comparison,
            content_components=content_components,
            content_weights=content_weights,
        )
        score = float(composition.score or 0.0)
        if not math.isfinite(score):
            prediction_failures.append(
                {
                    "code": "prediction_non_finite_case_score",
                    "reason": "case composition returned a non-finite score",
                }
            )
            score = 0.0

        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        rows = _per_frame_rows(
            times_s=times_s,
            comparison=comparison,
            union_ious=union_ious,
            subject=subject,
            orbit_score=orbit_score,
            reference_tracks=frozen_reference.tracks,
            prediction_objects=prediction_objects,
        )
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        position_path = request.artifact_dir / "entity_position_curve.png"
        cardinality_path = (
            request.artifact_dir / "object_cardinality_timeline.png"
        )
        angle_path = request.artifact_dir / "angular_trajectory_curve.png"
        audit_path = request.artifact_dir / "object_centric_audit.json"
        write_rows_csv(csv_path, rows)
        write_json(
            audit_path,
            {
                **comparison.to_dict(),
                "expected_timelines": [
                    value.to_dict() for value in expected_timelines
                ],
                "condition_frozen_identity": {
                    "entity_to_track": dict(
                        frozen_prediction.entity_to_track
                    ),
                    "residual_track_ids": list(
                        frozen_prediction.residual_track_ids
                    ),
                    "unmatched_entity_ids": list(
                        frozen_prediction.unmatched_entity_ids
                    ),
                    "total_cost": frozen_prediction.total_cost,
                },
            },
        )
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=union_ious,
            case_id=request.case["case_id"],
            scene_name="Open-world uniform circular motion",
        )
        position_values = [
            float(row["polar_position_similarity"]) for row in rows
        ]
        save_iou_curve(
            position_path,
            times_s=times_s,
            ious=position_values,
            case_id=request.case["case_id"],
            scene_name="Open-world circular entity position",
            series_label="Matched polar position similarity",
            y_label="Position similarity",
            metric_name="score",
        )
        expected_counts = [
            float(len(frozen_reference.entity_ids)) for _ in times_s
        ]
        predicted_counts = [
            float(row["formal_prediction_cardinality"]) for row in rows
        ]
        save_series_comparison(
            cardinality_path,
            times_s=times_s,
            reference=expected_counts,
            prediction=predicted_counts,
            ylabel="Observed physical objects",
            title=(
                "Circular-motion object cardinality — "
                f"{request.case['case_id']}"
            ),
        )
        _write_angle_curve(
            angle_path,
            request=request,
            times_s=times_s,
            reference_tracks=reference_tracks,
            prediction_tracks=prediction_tracks,
        )
        subject_artifacts = write_subject_artifacts(
            directory=request.artifact_dir,
            times_s=times_s,
            comparison=subject,
            case_id=request.case["case_id"],
            scene_name="Open-world uniform circular motion",
        )
        visualization_artifacts = write_open_world_v2_artifacts(
            request,
            scene_name="Open-world uniform circular motion",
            times_s=times_s,
            reference_frames=reference_video.frames,
            prediction_frames=prediction_frames,
            expected_timelines=expected_timelines,
            prediction_observation=prediction_objects,
            comparison=comparison,
            config=self.config.get("visualization"),
            reference_union_masks=reference_union,
            prediction_union_masks=prediction_union,
            full_subject_ious=union_ious,
            prediction_available=prediction_available,
        )

        integrity = comparison.integrity.to_dict()
        primary_metric = {
            **composition.to_dict(),
            "score": score,
            "content_components": content_components,
            "reference_mode": reference_mode,
        }
        return SceneAnalysis(
            score=score,
            metrics={
                "scene_subject_state_similarity": dict(primary_metric),
                "uniform_circular_motion_open_world_similarity": (
                    primary_metric
                ),
                "object_centric_integrity": integrity,
                "uniform_circular_motion_state_similarity": orbit_score,
                "physical_subject_similarity": subject.to_metric(
                    weights=(
                        {"appearance": 1.0}
                        if reference_mode == "parent_physics_reference"
                        else subject_weights(self.config["subject_scoring"])
                    )
                ),
                "physical_subject_mask_iou": {
                    **summarize_mask_ious(union_ious),
                    "role": "required_full_set_visual_diagnostic",
                },
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": bool(prediction_failures),
                "degradation_codes": [
                    value["code"] for value in prediction_failures
                ],
                "degradation_reason": (
                    "; ".join(
                        value["reason"] for value in prediction_failures
                    )
                    if prediction_failures
                    else None
                ),
                "expected_entity_count": len(frozen_reference.entity_ids),
                "prediction_track_count": len(prediction_objects.tracks),
                "reference_identity_anchors": list(
                    frozen_reference.anchors
                ),
                "condition_frozen_identity": {
                    "entity_to_track": dict(
                        frozen_prediction.entity_to_track
                    ),
                    "residual_track_ids": list(
                        frozen_prediction.residual_track_ids
                    ),
                    "unmatched_entity_ids": list(
                        frozen_prediction.unmatched_entity_ids
                    ),
                    "total_cost": frozen_prediction.total_cost,
                },
                "reference_open_world_observation": dict(
                    reference_observation.diagnostics
                ),
                "prediction_open_world_observation": (
                    prediction_observation_diagnostics
                ),
                "available_prediction_frames": int(
                    np.count_nonzero(prediction_available)
                ),
                "expected_prediction_frames": len(times_s),
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "entity_position_curve": str(position_path),
                "object_cardinality_timeline": str(cardinality_path),
                "angular_trajectory_curve": str(angle_path),
                "object_centric_audit": str(audit_path),
                **subject_artifacts,
                **visualization_artifacts,
            },
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                },
                "observation": {
                    "reference": dict(reference_observation.diagnostics),
                    "prediction": prediction_observation_diagnostics,
                },
                "identity_policy": (
                    "open_world_v2_condition_frozen_appearance_radius_phase_"
                    "anchors_plus_causal_track_ids"
                ),
                "expected_timelines": [
                    value.to_dict() for value in expected_timelines
                ],
                "open_world_protocol": {
                    "id": comparison.protocol_id,
                    "version": comparison.protocol_version,
                    "failed": comparison.failed,
                    "failure_reason": comparison.failure_reason,
                },
                "position_kernel": (
                    "relative_polar_cauchy_with_cartesian_center_fallback"
                ),
                "apparatus_policy": "eroded_green_disk_excluded",
                "subject_reference_mode": reference_mode,
                **condition_provenance,
            },
        )


def _expected_timelines(
    *,
    entities: Sequence[Any],
    frozen_reference,
    condition_observation: CircularOpenWorldObservation | None,
    condition_anchors: Sequence[Mapping[str, object]],
    capability: ReferenceCapability,
    time_grid,
):
    reference_by_id = {
        track.matched_entity_id: track
        for track in frozen_reference.tracks
    }
    reference_masks_by_id = {
        entity_id: masks
        for entity_id, masks in zip(
            frozen_reference.entity_ids,
            frozen_reference.instance_masks,
        )
    }
    anchors = {
        str(value["entity_id"]): value for value in condition_anchors
    }
    condition_tracks = (
        {
            track.track_id: track
            for track in condition_observation.objects.tracks
        }
        if condition_observation is not None
        else {}
    )
    timelines = []
    for entity in entities:
        reference_track = reference_by_id[entity.entity_id]
        condition_position = None
        if capability is not ReferenceCapability.SAME_CASE_GT:
            anchor = anchors[entity.entity_id]
            condition_track = condition_tracks[str(anchor["track_id"])]
            detection = condition_track.detections[0]
            condition_position = ExpectedPositionSample(
                xy=detection.xy,
                area_px2=detection.area_px2,
                mask=detection.mask,
            )
        timelines.append(
            resolve_expected_entity_timeline(
                entity.to_entity_spec(),
                time_grid=time_grid,
                capability=capability,
                reference_track=reference_track,
                condition_position=condition_position,
                condition_identity_supervised=(
                    capability is not ReferenceCapability.SAME_CASE_GT
                ),
                reference_masks=(
                    reference_masks_by_id[entity.entity_id]
                    if capability is ReferenceCapability.SAME_CASE_GT
                    else None
                ),
                metadata={
                    "identity_anchor": dict(anchors[entity.entity_id]),
                    "coordinate_system": (
                        "disk_normalized_relative_polar"
                    ),
                },
            )
        )
    return tuple(timelines)


def _ordered_entities(manifest) -> list[Any]:
    if manifest.scene_id != "uniform_circular_motion":
        raise ValueError(
            "circular evaluator requires uniform_circular_motion manifest"
        )
    entities = list(manifest.entities)
    if not entities:
        raise ValueError("circular manifest declares no orbiters")
    if any(entity.entity_class != "orbiter" for entity in entities):
        raise ValueError("all circular manifest entities must be orbiters")
    return entities


def _content_contract(
    *,
    orbit_score: Mapping[str, Any],
    subject: SubjectComparison,
    reference_mode: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, float]]:
    configured = config.get("content_weights", {})
    if reference_mode == "parent_physics_reference":
        components: dict[str, float | None] = {
            "orbit_physics": float(orbit_score["score"]),
            "appearance": subject.components.get("appearance"),
        }
        defaults = {"orbit_physics": 0.7, "appearance": 0.3}
    else:
        components = {
            "orbit_physics": float(orbit_score["score"]),
            "shape": subject.components.get("shape"),
            "appearance": subject.components.get("appearance"),
        }
        defaults = {
            "orbit_physics": 0.55,
            "shape": 0.15,
            "appearance": 0.30,
        }
    weights = {
        name: float(configured.get(name, defaults[name]))
        for name in components
    }
    return components, weights


def _normalized_prediction_timeline(
    frames: Sequence[np.ndarray],
    *,
    available: Sequence[bool] | None,
    frame_count: int,
    frame_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, str] | None]:
    values = list(frames[:frame_count])
    source_available = (
        [True] * len(values)
        if available is None
        else [bool(value) for value in list(available)[: len(values)]]
    )
    normalized: list[np.ndarray] = []
    normalized_available: list[bool] = []
    for index, frame in enumerate(values):
        value = np.asarray(frame)
        usable = (
            source_available[index]
            and value.shape == frame_shape
            and value.dtype == np.uint8
        )
        normalized.append(
            value if usable else np.zeros(frame_shape, dtype=np.uint8)
        )
        normalized_available.append(usable)
    while len(normalized) < frame_count:
        normalized.append(np.zeros(frame_shape, dtype=np.uint8))
        normalized_available.append(False)
    mask = np.asarray(normalized_available, dtype=bool)
    if bool(np.all(mask)):
        return normalized, mask, None
    return normalized, mask, {
        "code": "prediction_partial_timeline",
        "reason": (
            f"prediction supplies {int(np.count_nonzero(mask))}/"
            f"{frame_count} usable common timeline samples"
        ),
    }


def _load_condition_frame(
    request: CaseEvaluationRequest,
    *,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    value = request.case.get("assets", {}).get("first_frame")
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(
            "reference_condition_frame_missing",
            "parent-reference circular case has no first_frame asset",
        )
    asset_root = request.asset_root.resolve()
    path = (asset_root / value).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            "reference_condition_path_escape",
            f"condition frame escapes dataset root: {value}",
        ) from exc
    source = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if source is None:
        raise ReferenceAnalysisError(
            "reference_condition_frame_unreadable",
            f"cannot read condition frame: {path}",
        )
    target_height, target_width = target_shape[:2]
    source_height, source_width = source.shape[:2]
    scale = min(
        target_width / max(source_width, 1),
        target_height / max(source_height, 1),
    )
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        source,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    output = np.full(
        (target_height, target_width, 3),
        int(request.evaluator_config["spatial"].get("pad_value", 0)),
        dtype=np.uint8,
    )
    offset_x = (target_width - width) // 2
    offset_y = (target_height - height) // 2
    output[offset_y : offset_y + height, offset_x : offset_x + width] = (
        resized
    )
    return output, {
        "condition_frame": str(path),
        "condition_frame_sha256": sha256_file(path),
        "condition_spatial_transform": {
            "scale": scale,
            "offset_xy": [offset_x, offset_y],
            "source_size": [source_width, source_height],
            "target_size": [target_width, target_height],
        },
    }


def _degraded_subject_comparison(
    frame_count: int,
    *,
    reference_mode: str,
) -> SubjectComparison:
    return SubjectComparison(
        score=0.0,
        components={
            "position": 0.0,
            "shape": 0.0,
            "appearance": 0.0,
        },
        per_frame=[
            {
                "frame": index,
                "position": 0.0,
                "shape": 0.0,
                "appearance": 0.0,
                "subject": 0.0,
            }
            for index in range(frame_count)
        ],
        reference_observed_ratio=1.0,
        prediction_observed_ratio=0.0,
        comparable_reference=(
            "same_case_ground_truth_video"
            if reference_mode == "same_case_reference"
            else "case_condition_first_frame_appearance_only"
        ),
    )


def _degraded_orbit_score(failure: Mapping[str, str]) -> dict[str, Any]:
    return {
        "score": 0.0,
        "components": {
            "legacy_orbit_state": 0.0,
            "polar_position": 0.0,
            "matched_evidence_recall": 0.0,
        },
        "legacy_orbit_state": {
            "score": 0.0,
            "degraded": True,
            "degradation_code": failure["code"],
            "degradation_reason": failure["reason"],
        },
        "polar_matches": [],
    }


def _per_frame_rows(
    *,
    times_s: Sequence[float],
    comparison: ObjectCentricComparisonV2,
    union_ious: Sequence[float | None],
    subject: SubjectComparison,
    orbit_score: Mapping[str, Any],
    reference_tracks: Sequence[Any],
    prediction_objects,
) -> list[dict[str, Any]]:
    polar_by_frame: dict[int, list[float]] = {}
    for value in orbit_score.get("polar_matches", []):
        polar_by_frame.setdefault(int(value["frame"]), []).append(
            float(value["score"])
        )
    reference_by_id = {
        track.matched_entity_id: track for track in reference_tracks
    }
    prediction_detections = {
        (track.track_id, detection.frame_index): detection
        for track in prediction_objects.tracks
        for detection in track.detections
    }
    rows: list[dict[str, Any]] = []
    for frame_index, time_s in enumerate(times_s):
        audit = comparison.per_frame[frame_index]
        expected_count = int(audit["expected_cardinality"])
        polar_values = polar_by_frame.get(frame_index, [])
        polar = (
            float(sum(polar_values) / max(expected_count, 1))
            if expected_count
            else 0.0
        )
        subject_row = subject.per_frame[frame_index]
        match_by_entity = {
            str(value["entity_id"]): value for value in audit["matches"]
        }
        entity_trajectories: dict[str, dict[str, object]] = {}
        for entity_id in audit["expected_entity_ids"]:
            reference_track = reference_by_id[str(entity_id)]
            match = match_by_entity.get(str(entity_id))
            detection = (
                None
                if match is None
                else prediction_detections.get(
                    (
                        str(match["prediction_track_id"]),
                        frame_index,
                    )
                )
            )
            entity_trajectories[str(entity_id)] = {
                "reference_canonical_xy": (
                    reference_track.xy[frame_index].tolist()
                ),
                "prediction_canonical_xy": (
                    None if detection is None else detection.xy.tolist()
                ),
                "prediction_track_id": (
                    None
                    if match is None
                    else str(match["prediction_track_id"])
                ),
                "position_score": (
                    None
                    if match is None
                    else match.get("position_score")
                ),
            }
        rows.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "physical_subject_iou": union_ious[frame_index],
                "polar_position_similarity": polar,
                "matched_count": len(audit["matches"]),
                "missing_count": len(audit["missing_entity_ids"]),
                "extra_count": len(audit["extra_track_ids"]),
                "formal_prediction_cardinality": audit[
                    "formal_prediction_cardinality"
                ],
                "overflow_count": audit["overflow_count"],
                "missing_entity_ids": json.dumps(
                    audit["missing_entity_ids"], separators=(",", ":")
                ),
                "extra_track_ids": json.dumps(
                    audit["extra_track_ids"], separators=(",", ":")
                ),
                "birth_track_ids": json.dumps(
                    audit["birth_track_ids"], separators=(",", ":")
                ),
                "death_track_ids": json.dumps(
                    audit["death_track_ids"], separators=(",", ":")
                ),
                "id_switches": json.dumps(
                    audit["id_switches"],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "entity_trajectories": json.dumps(
                    entity_trajectories,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "rejected_candidate_matches": json.dumps(
                    audit["rejected_candidate_matches"],
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "subject_shape": subject_row.get("shape"),
                "subject_appearance": subject_row.get("appearance"),
            }
        )
    return rows


def _write_angle_curve(
    path: Path,
    *,
    request: CaseEvaluationRequest,
    times_s: Sequence[float],
    reference_tracks,
    prediction_tracks,
) -> None:
    reference_xy = reference_tracks.xy[:, -1]
    reference_angle = np.unwrap(
        np.arctan2(reference_xy[:, 1], reference_xy[:, 0])
    )
    reference_angle -= reference_angle[0]
    if (
        prediction_tracks.xy.shape == reference_tracks.xy.shape
        and np.isfinite(prediction_tracks.xy[:, -1]).all()
    ):
        prediction_xy = prediction_tracks.xy[:, -1]
        prediction_angle = np.unwrap(
            np.arctan2(prediction_xy[:, 1], prediction_xy[:, 0])
        )
        prediction_angle -= prediction_angle[0]
    else:
        prediction_angle = np.zeros(len(times_s), dtype=np.float64)
    save_series_comparison(
        path,
        times_s=list(times_s),
        reference=reference_angle,
        prediction=prediction_angle,
        ylabel="Relative unwrapped angle (rad)",
        title=(
            "Open-world circular angular trajectory — "
            f"{request.case['case_id']}"
        ),
    )


__all__ = ["CircularMotionOpenWorldCaseEvaluator"]
