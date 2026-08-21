from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file
from ....datasets.physics import flat_physics_quantities
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
    ObjectTrack,
    ReferenceCapability,
    VisibilityState,
    build_common_time_grid,
    compare_open_world_v2,
    compose_object_centric_v2,
    fail_closed_open_world_v2,
    freeze_condition_identity,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
)
from ...common.entities.observer import OpenWorldObservation
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.subject import compare_subjects, infer_reference_mode
from ...contracts import CaseEvaluationRequest
from .open_world import (
    PENDULUM_STRUCTURE_SPEC_VERSION,
    PendulumStructureSpec,
    audit_pendulum_topology,
    detect_condition_structure,
    discover_pendulum_objects,
    extract_bob_masks,
    letterbox_condition_image,
    prompt_from_structure,
    structure_from_reference_trace,
)
from .scoring import TraceQualityError, extract_trace, score_traces
from .segmentation import Sam2PendulumSegmenter, SegmentationError


class PendulumOpenWorldCaseEvaluator(ReferenceCaseEvaluator):
    """v6 pendulum evaluator with a condition-frozen bob identity."""

    evaluator_id = "pendulum_open_world_structure"
    evaluator_version = "2.0"
    sequential_evaluator_version = "2.0"
    robust_evaluator_version = "2.0"
    scene_id = "pendulum"
    primary_score = "pendulum_open_world_similarity"
    allow_partial_prediction = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2PendulumSegmenter(config["sam2"])
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
                "entity_contract": "manifest_v1",
                "observer": "open_world_v2",
                "structure_spec": PENDULUM_STRUCTURE_SPEC_VERSION,
                "composition": "integrity_gate_times_scene_content_v1",
            }
        )

    def describe_observation(self) -> dict[str, Any]:
        assignment = self.config["object_centric_scoring"]["assignment"]
        return {
            "subject": "manifest_bob_identity_plus_string_topology",
            "directed_channel": (
                "current_condition_pivot_string_bob_prompt_plus_sam2"
            ),
            "residual_channel": (
                "circle_string_condition_change_multi_source_tracking"
            ),
            "apparatus_exclusion": "pivot_support_and_vertical_rod_corridor",
            "identity": (
                "condition_frame_position_size_appearance_frozen_before_"
                "future_scoring"
            ),
            "assignment": {
                "method": "open_world_v2_hungarian_null_plus_frozen_identity",
                "minimum_match_position_similarity": float(
                    assignment["minimum_match_position_similarity"]
                ),
            },
            "topology": (
                f"PendulumStructureSpec/{PENDULUM_STRUCTURE_SPEC_VERSION}"
            ),
            "physics_parent_policy": (
                "condition_existence_appearance_identity_plus_parent_"
                "normalized_dynamics_no_parent_future_pixels"
            ),
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
            if manifest.scene_id != self.scene_id or len(manifest.entities) != 1:
                raise ValueError(
                    "pendulum requires exactly one declared bob identity"
                )
            declaration = manifest.entities[0]
            if declaration.entity_class != "pendulum_bob":
                raise ValueError(
                    "pendulum entity class must be 'pendulum_bob'"
                )
            entity = declaration.to_entity_spec()
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_invalid",
                f"pendulum entity manifest is invalid: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        time_grid = build_common_time_grid(times_s)
        proposal_config = self.config["motion_proposal"]
        try:
            (
                reference_assembly,
                reference_segmentation,
                _,
            ) = self._segmenter.segment(
                reference_video.frames,
                proposal_config=proposal_config,
            )
            reference_trace = extract_trace(
                reference_assembly,
                times_s,
                quality_config=self.config["quality"],
                period_config=self.config["period"],
            )
            (
                reference_bobs,
                reference_bob_xy,
                reference_bob_radii,
            ) = extract_bob_masks(
                reference_assembly,
                minimum_pixels=int(
                    self.config["open_world_observation"].get(
                        "minimum_bob_pixels", 8
                    )
                ),
            )
            reference_track = _object_track_from_bob_masks(
                entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                masks=reference_bobs,
                xy=reference_bob_xy,
                radii=reference_bob_radii,
                time_grid=time_grid,
                reference=True,
            )
        except (SegmentationError, TraceQualityError, Exception) as exc:
            # The broad boundary is reference-only.  A bad GT observation is
            # unavailable, never silently converted into a prediction zero.
            if isinstance(exc, ReferenceAnalysisError):
                raise
            raise ReferenceAnalysisError(
                "reference_pendulum_open_world_observation_failed",
                f"reference pendulum observation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        condition_frame, condition_path, condition_transform = (
            _load_condition_frame(request, config=self.config)
        )
        reference_mode = infer_reference_mode(request.case)
        try:
            if reference_mode == "same_case_reference":
                condition_structure = structure_from_reference_trace(
                    pivot_xy=reference_trace.frame_pivot_xy[0],
                    bob_xy=reference_trace.bob_xy[0],
                    assembly_mask=reference_assembly[0],
                )
            else:
                condition_structure = detect_condition_structure(
                    condition_frame,
                    config=self.config["open_world_observation"],
                    expected_radius_length_ratio=(
                        _pendulum_radius_length_ratio(request.case)
                    ),
                )
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_condition_pendulum_structure_failed",
                f"condition pendulum structure failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        capability = manifest.reference_capability
        timeline = resolve_expected_entity_timeline(
            entity,
            time_grid=time_grid,
            capability=capability,
            reference_track=reference_track,
            condition_position=ExpectedPositionSample(
                xy=condition_structure.bob_xy,
                area_px2=float(
                    np.count_nonzero(condition_structure.bob_mask)
                ),
                mask=condition_structure.bob_mask,
            ),
            condition_identity_supervised=True,
            reference_masks=(
                reference_bobs
                if capability is ReferenceCapability.SAME_CASE_GT
                else [condition_structure.bob_mask]
                + [None] * (len(times_s) - 1)
            ),
            metadata={
                "condition_structure_source": condition_structure.source,
                "physics_parent_future_pixel_supervision": False,
            },
        )

        prediction_frames, prediction_available, timeline_failure = (
            _normalized_prediction_timeline(
                getattr(prediction_video, "frames", []),
                available=getattr(prediction_video, "available", None),
                frame_count=len(times_s),
                frame_shape=reference_video.frames[0].shape,
            )
        )
        prediction_failures: list[dict[str, str]] = []
        if timeline_failure is not None:
            prediction_failures.append(timeline_failure)

        empty_mask = np.zeros(reference_video.frames[0].shape[:2], dtype=np.uint8)
        prediction_assembly = [
            np.array(empty_mask, copy=True) for _ in times_s
        ]
        prediction_bobs = [
            np.array(empty_mask, copy=True) for _ in times_s
        ]
        prediction_bob_xy = np.full(
            (len(times_s), 2), np.nan, dtype=np.float64
        )
        prediction_segmentation: Mapping[str, Any] = {
            "status": "not_run"
        }
        observation: OpenWorldObservation | None = None
        frozen_identity = None
        comparison = None
        try:
            condition_prompt = prompt_from_structure(condition_structure)
            (
                prediction_assembly,
                prediction_segmentation,
                _,
            ) = self._segmenter.segment(
                prediction_frames,
                proposal_config=proposal_config,
                prompt=condition_prompt,
                prompt_source="case_condition_structure",
            )
            for index, is_available in enumerate(prediction_available):
                if not is_available:
                    prediction_assembly[index] = np.zeros_like(
                        prediction_assembly[index]
                    )
            (
                prediction_bobs,
                prediction_bob_xy,
                _,
            ) = extract_bob_masks(
                prediction_assembly,
                expected_radius_px=condition_structure.bob_radius_px,
                minimum_pixels=int(
                    self.config["open_world_observation"].get(
                        "minimum_bob_pixels", 8
                    )
                ),
            )
            observation = discover_pendulum_objects(
                prediction_frames,
                directed_bob_masks=prediction_bobs,
                condition_frame=condition_frame,
                structure=condition_structure,
                time_grid=time_grid,
                config=self.config["open_world_observation"],
                available=prediction_available,
            )
            frozen_identity = _freeze_bob_identity(
                entity=entity,
                observation=observation,
                structure=condition_structure,
                frame=condition_frame,
                maximum_match_cost=float(
                    self.config["object_centric_scoring"]["assignment"].get(
                        "maximum_condition_identity_cost", 3.0
                    )
                ),
            )
            comparison = compare_open_world_v2(
                expected_timelines=[timeline],
                prediction_observation=observation,
                time_grid=time_grid,
                frame_diagonal_px=float(
                    math.hypot(
                        reference_video.frames[0].shape[1],
                        reference_video.frames[0].shape[0],
                    )
                ),
                frozen_identity=frozen_identity,
                minimum_match_position_similarity=float(
                    self.config["object_centric_scoring"]["assignment"][
                        "minimum_match_position_similarity"
                    ]
                ),
            )
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_open_world_observation_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            comparison = fail_closed_open_world_v2(
                expected_timelines=[timeline],
                time_grid=time_grid,
                reason=failure["reason"],
            )
        assert comparison is not None

        matched_bobs = [
            (
                np.array(mask, copy=True)
                if mask is not None
                else np.zeros_like(empty_mask)
            )
            for mask in comparison.matched_masks
        ]
        matched_xy, matched_observed = _centroids(matched_bobs)
        observed_coverage = _time_coverage(
            matched_observed & timeline.expected_exists,
            timeline.expected_exists & timeline.existence_supervised,
            time_grid.cell_weights_s,
        )
        try:
            lenient_quality = dict(self.config["quality"])
            lenient_quality["minimum_valid_frame_ratio"] = 0.0
            prediction_trace = extract_trace(
                prediction_assembly,
                times_s,
                quality_config=lenient_quality,
                period_config=self.config["period"],
            )
            raw_state = score_traces(
                reference_trace,
                prediction_trace,
                scoring_config=self.config["scoring"],
            )
            state_score = {
                **raw_state,
                "raw_interpolated_score": float(raw_state["score"]),
                "physical_evidence_coverage": observed_coverage,
                "score": float(raw_state["score"]) * observed_coverage,
                "missing_frames_are_not_free": True,
            }
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_physics_scoring_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            state_score = {
                "score": 0.0,
                "raw_interpolated_score": 0.0,
                "physical_evidence_coverage": observed_coverage,
                "failure": failure,
                "missing_frames_are_not_free": True,
            }

        try:
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=reference_bobs,
                prediction_masks=matched_bobs,
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
                condition_frame=(
                    condition_frame
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
                condition_mask=(
                    condition_structure.bob_mask
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
            )
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_subject_scoring_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=reference_bobs,
                prediction_masks=[
                    np.zeros_like(value) for value in reference_bobs
                ],
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
                condition_frame=(
                    condition_frame
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
                condition_mask=(
                    condition_structure.bob_mask
                    if reference_mode == "parent_physics_reference"
                    else None
                ),
            )

        topology = audit_pendulum_topology(
            prediction_assembly,
            pivot_xy=condition_structure.pivot_xy,
            bob_xy=matched_xy,
            bob_observed=matched_observed,
            bob_masks=matched_bobs,
            frame_weights=time_grid.cell_weights_s,
            string_half_width_px=int(
                self.config["topology"].get("string_half_width_px", 2)
            ),
            minimum_string_occupancy=float(
                self.config["topology"]["minimum_string_occupancy"]
            ),
            branch_penalty_weight=float(
                self.config["topology"].get(
                    "branch_string_penalty_weight", 0.65
                )
            ),
            branch_minimum_length_ratio=float(
                self.config["topology"].get(
                    "branch_minimum_length_ratio", 0.12
                )
            ),
            branch_minimum_elongation=float(
                self.config["topology"].get(
                    "branch_minimum_elongation", 3.0
                )
            ),
            branch_maximum_thickness_radius_ratio=float(
                self.config["topology"].get(
                    "branch_maximum_thickness_radius_ratio", 0.75
                )
            ),
            branch_bob_exclusion_margin_ratio=float(
                self.config["topology"].get(
                    "branch_bob_exclusion_margin_ratio", 0.40
                )
            ),
            branch_support_half_width_radius_ratio=float(
                self.config["topology"].get(
                    "branch_support_half_width_radius_ratio", 1.25
                )
            ),
            branch_support_verticality_threshold=float(
                self.config["topology"].get(
                    "branch_support_verticality_threshold", 0.90
                )
            ),
            branch_pivot_support_verticality_threshold=float(
                self.config["topology"].get(
                    "branch_pivot_support_verticality_threshold", 0.65
                )
            ),
        )
        extra_ratio = _extra_exposure_ratio(
            comparison.per_frame,
            time_grid.cell_weights_s,
        )
        topology_score = float(topology["score"]) * max(
            0.0,
            1.0
            - float(self.config["topology"].get("branch_penalty_weight", 0.5))
            * extra_ratio,
        )
        topology_metric = {
            **topology,
            "score_before_branch_penalty": topology["score"],
            "extra_branch_exposure_ratio": extra_ratio,
            "score": topology_score,
        }
        content, weights = _content_contract(
            state_score=float(state_score["score"]),
            subject_components=subject.components,
            subject_score=subject.score,
            topology_score=topology_score,
            reference_mode=reference_mode,
            config=self.config["object_centric_scoring"],
        )
        composition = compose_object_centric_v2(
            comparison,
            content_components=content,
            content_weights=weights,
        )
        score = float(composition.score or 0.0)
        if not math.isfinite(score):
            score = 0.0
            prediction_failures.append(
                {
                    "code": "prediction_non_finite_case_score",
                    "reason": "v6 composition produced a non-finite score",
                }
            )

        prediction_union = _full_prediction_union(
            prediction_assembly,
            observation=observation,
        )
        if reference_mode == "parent_physics_reference":
            full_reference_union: list[np.ndarray | None] = [
                condition_structure.subject_mask,
                *([None] * (len(times_s) - 1)),
            ]
            full_ious: list[float | None] = [
                observed_mask_iou(
                    condition_structure.subject_mask,
                    prediction_union[0],
                ),
                *([None] * (len(times_s) - 1)),
            ]
        else:
            full_reference_union = list(reference_assembly)
            full_ious = [
                observed_mask_iou(
                    reference_assembly[index], prediction_union[index]
                )
                for index in range(len(times_s))
            ]
        artifacts = _write_artifacts(
            request,
            times_s=times_s,
            reference_assembly=reference_assembly,
            prediction_union=prediction_union,
            full_ious=full_ious,
            comparison=comparison,
            observation=observation,
            subject_rows=subject.per_frame,
            topology_rows=topology["per_frame"],
        )
        artifacts.update(
            write_open_world_v2_artifacts(
                request,
                scene_name="Open-world pendulum",
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                expected_timelines=[timeline],
                prediction_observation=observation,
                comparison=comparison,
                config=self.config.get("visualization"),
                reference_union_masks=full_reference_union,
                prediction_union_masks=prediction_union,
                full_subject_ious=full_ious,
                prediction_available=prediction_available,
            )
        )
        primary_metric = {
            **composition.to_dict(),
            "score": score,
            "content_components": content,
            "content_weights": weights,
        }
        return SceneAnalysis(
            score=score,
            metrics={
                "scene_subject_state_similarity": dict(primary_metric),
                "pendulum_open_world_similarity": primary_metric,
                "object_centric_integrity": (
                    comparison.integrity.to_dict()
                ),
                "pendulum_state_similarity": state_score,
                "pendulum_structure_integrity": topology_metric,
                "physical_subject_similarity": subject.to_metric(
                    weights=(
                        {"appearance": 1.0}
                        if reference_mode == "parent_physics_reference"
                        else {
                            key: float(value)
                            for key, value in self.config[
                                "subject_scoring"
                            ]["weights"].items()
                        }
                    )
                ),
                "physical_subject_mask_iou": {
                    **summarize_mask_ious(full_ious),
                    "role": (
                        "required_full_physical_subject_union_diagnostic"
                    ),
                    "includes_residual_prediction_entities": True,
                    "parent_reference_policy": (
                        "current_condition_frame_only_no_parent_future_pixels"
                        if reference_mode == "parent_physics_reference"
                        else "same_case_ground_truth_all_frames"
                    ),
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
                "reference_valid_mask_ratio": (
                    reference_trace.valid_ratio
                ),
                "prediction_observed_bob_ratio": float(
                    np.mean(matched_observed)
                ),
                "physical_evidence_coverage": observed_coverage,
                "expected_entity_count": 1,
                "prediction_track_count": (
                    len(observation.tracks)
                    if observation is not None
                    else 0
                ),
                "open_world_comparison_failed": comparison.failed,
            },
            artifacts=artifacts,
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                    "reference_capability": capability.value,
                },
                "condition_frame": str(condition_path),
                "condition_frame_sha256": sha256_file(condition_path),
                "condition_transform": condition_transform,
                "condition_structure": condition_structure.to_dict(),
                "condition_mask_source": (
                    "current_case_condition_structure_not_prediction_roi"
                ),
                "physics_parent_future_pixel_supervision": False,
                "identity_assignment": (
                    None
                    if frozen_identity is None
                    else {
                        "entity_to_track": dict(
                            frozen_identity.entity_to_track
                        ),
                        "residual_track_ids": list(
                            frozen_identity.residual_track_ids
                        ),
                        "unmatched_entity_ids": list(
                            frozen_identity.unmatched_entity_ids
                        ),
                        "total_cost": frozen_identity.total_cost,
                    }
                ),
                "segmentation": {
                    "reference": reference_segmentation,
                    "prediction": dict(prediction_segmentation),
                },
                "open_world": comparison.to_dict(),
            },
        )


def _load_condition_frame(
    request: CaseEvaluationRequest,
    *,
    config: Mapping[str, Any],
    normalized_reference_frame: np.ndarray | None = None,
    normalized_reference_transform: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, Path, dict[str, object]]:
    value = request.case.get("assets", {}).get("first_frame")
    if not isinstance(value, str) or not value:
        raise ReferenceAnalysisError(
            "reference_condition_frame_missing",
            "pendulum Case has no first_frame condition asset",
        )
    root = request.asset_root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            "reference_condition_frame_path_escape",
            f"condition asset escapes dataset root: {value}",
        ) from exc
    if not path.is_file():
        raise ReferenceAnalysisError(
            "reference_condition_frame_missing",
            f"condition frame does not exist: {path}",
        )
    spatial = config["spatial"]
    if spatial.get("policy") == "shared_reference_content_no_pad_v1":
        if (
            normalized_reference_frame is None
            or normalized_reference_transform is None
        ):
            raise ReferenceAnalysisError(
                "reference_condition_alignment_missing",
                "no-pad pendulum evaluation requires the normalized "
                "reference frame-zero coordinate system",
            )
        return (
            np.array(normalized_reference_frame, copy=True),
            path,
            dict(normalized_reference_transform),
        )
    frame, transform = letterbox_condition_image(
        path,
        width=int(spatial["width"]),
        height=int(spatial["height"]),
        pad_value=int(spatial.get("pad_value", 0)),
    )
    return frame, path, transform


def _pendulum_radius_length_ratio(case: Mapping[str, Any]) -> float:
    physics = flat_physics_quantities(case)
    try:
        radius = float(physics["bob_radius"]["value"])
        length = float(physics["string_length"]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceAnalysisError(
            "reference_pendulum_physics_incomplete",
            "pendulum condition identity requires bob_radius and "
            "string_length quantities",
        ) from exc
    if not math.isfinite(radius) or not math.isfinite(length) or min(
        radius, length
    ) <= 0.0:
        raise ReferenceAnalysisError(
            "reference_pendulum_physics_invalid",
            "pendulum radius and length must be finite and positive",
        )
    # The visual geometry measures from the pivot to the bob centre.  The
    # declared string terminates at the bob surface, so the corresponding
    # centre distance is the string length plus one bob radius.
    return radius / (length + radius)


def _object_track_from_bob_masks(
    *,
    entity_id: str,
    entity_class: str,
    masks: Sequence[np.ndarray],
    xy: np.ndarray,
    radii: np.ndarray,
    time_grid,
    reference: bool,
) -> ObjectTrack:
    frame_count = len(masks)
    centers = np.asarray(xy, dtype=np.float64)
    observed = np.asarray(
        [
            np.isfinite(centers[index]).all()
            and np.count_nonzero(mask) >= 4
            for index, mask in enumerate(masks)
        ],
        dtype=bool,
    )
    areas = np.asarray(
        [
            float(np.count_nonzero(mask)) if observed[index] else np.nan
            for index, mask in enumerate(masks)
        ],
        dtype=np.float64,
    )
    confidence = np.where(observed, 1.0, 0.0)
    return ObjectTrack(
        track_id=("gt_" if reference else "condition_") + entity_id,
        matched_entity_id=entity_id if reference else None,
        xy=centers,
        observed=observed,
        visibility=tuple(
            VisibilityState.VISIBLE if value else VisibilityState.UNKNOWN
            for value in observed
        ),
        areas_px2=areas,
        confidence=confidence,
        existence_observed=observed,
        localization_eligible=observed,
        association_eligible=observed,
        time_weights_s=time_grid.cell_weights_s,
        metadata={
            "entity_class": entity_class,
            "radii_px": np.asarray(radii, dtype=np.float64).tolist(),
        },
    )


def _freeze_bob_identity(
    *,
    entity,
    observation: OpenWorldObservation,
    structure: PendulumStructureSpec,
    frame: np.ndarray,
    maximum_match_cost: float,
):
    del frame  # Reserved for a versioned appearance descriptor extension.
    initial = {
        track.track_id: next(
            (
                value
                for value in track.detections
                if value.frame_index == 0
            ),
            None,
        )
        for track in observation.tracks
    }
    track_ids = [
        track_id
        for track_id, detection in initial.items()
        if detection is not None
        and detection.entity_class == entity.entity_class
        and detection.metadata.get("identity_anchor_valid") is not False
        and "condition_directed_sam2" in detection.sources
    ]
    condition_area = max(
        float(np.count_nonzero(structure.bob_mask)), 1.0
    )

    def cost(_entity, track_id: str) -> float:
        detection = initial[track_id]
        assert detection is not None
        distance = float(
            np.linalg.norm(detection.xy - structure.bob_xy)
            / max(2.0 * structure.bob_radius_px, 1.0)
        )
        area = abs(math.log(detection.area_px2 / condition_area))
        direct_bonus = (
            -0.15
            if "condition_directed_sam2" in detection.sources
            else 0.0
        )
        return max(distance + 0.25 * area + direct_bonus, 0.0)

    return freeze_condition_identity(
        [entity],
        track_ids,
        identity_cost_hook=cost,
        maximum_match_cost=maximum_match_cost,
        unmatched_entity_cost=1.0,
        unmatched_track_cost=1.0,
        maximum_entities=4,
    )


def _normalized_prediction_timeline(
    frames: Sequence[np.ndarray],
    *,
    available: Sequence[bool] | None,
    frame_count: int,
    frame_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], list[bool], dict[str, str] | None]:
    source_frames = list(frames)
    source_available = (
        [True] * len(source_frames)
        if available is None
        else list(available)
    )
    valid_availability = (
        len(source_available) == len(source_frames)
        and all(isinstance(value, (bool, np.bool_)) for value in source_available)
    )
    if not valid_availability:
        source_available = [False] * len(source_frames)
    output: list[np.ndarray] = []
    output_available: list[bool] = []
    malformed = not valid_availability
    for index in range(frame_count):
        if index >= len(source_frames):
            output.append(np.zeros(frame_shape, dtype=np.uint8))
            output_available.append(False)
            continue
        value = np.asarray(source_frames[index])
        valid = (
            value.shape == frame_shape
            and value.dtype == np.uint8
            and bool(source_available[index])
        )
        if valid:
            output.append(value)
            output_available.append(True)
        else:
            output.append(np.zeros(frame_shape, dtype=np.uint8))
            output_available.append(False)
            malformed = True
    failure = (
        {
            "code": "prediction_timeline_normalized_with_missing_frames",
            "reason": (
                "prediction timeline was short, unavailable, or malformed; "
                "missing samples were retained as explicit absence"
            ),
        }
        if malformed or not all(output_available)
        else None
    )
    return output, output_available, failure


def _centroids(
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.full((len(masks), 2), np.nan, dtype=np.float64)
    observed = np.zeros(len(masks), dtype=bool)
    for index, mask in enumerate(masks):
        moments = cv2.moments(
            np.where(np.asarray(mask) > 0, 1, 0).astype(np.uint8),
            binaryImage=True,
        )
        if abs(float(moments["m00"])) <= 1e-12:
            continue
        xy[index] = [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ]
        observed[index] = True
    return xy, observed


def _time_coverage(
    observed: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = float(np.dot(expected.astype(np.float64), weights))
    if denominator <= 1e-12:
        return float(np.mean(observed[expected])) if np.any(expected) else 1.0
    numerator = float(
        np.dot((observed & expected).astype(np.float64), weights)
    )
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _extra_exposure_ratio(
    rows: Sequence[Mapping[str, object]],
    weights: np.ndarray,
) -> float:
    expected = float(
        sum(
            float(row.get("expected_cardinality", 0.0)) * weights[index]
            for index, row in enumerate(rows)
        )
    )
    extra = float(
        sum(float(row.get("extra_exposure_s", 0.0)) for row in rows)
    )
    return float(np.clip(extra / max(expected, 1e-12), 0.0, 1.0))


def _content_contract(
    *,
    state_score: float,
    subject_components: Mapping[str, float | None],
    subject_score: float,
    topology_score: float,
    reference_mode: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, float | None], dict[str, float]]:
    configured = {
        key: float(value)
        for key, value in config["content_weights"].items()
    }
    if reference_mode == "parent_physics_reference":
        content = {
            "physics_state": state_score,
            "appearance": subject_components.get(
                "appearance", subject_score
            ),
            "topology": topology_score,
        }
    else:
        content = {
            "physics_state": state_score,
            "shape": subject_components.get("shape"),
            "appearance": subject_components.get("appearance"),
            "topology": topology_score,
        }
    return content, {key: configured[key] for key in content}


def _full_prediction_union(
    assembly_masks: Sequence[np.ndarray],
    *,
    observation: OpenWorldObservation | None,
) -> list[np.ndarray]:
    output = [
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        for mask in assembly_masks
    ]
    if observation is None:
        return output
    for track in observation.tracks:
        if track.formal_exposure_weight <= 0.0:
            continue
        for detection in track.detections:
            if detection.mask is not None:
                output[detection.frame_index] = np.maximum(
                    output[detection.frame_index],
                    np.asarray(detection.mask, dtype=np.uint8),
                )
    return output


def _track_payload(
    observation: OpenWorldObservation | None,
) -> list[dict[str, object]]:
    if observation is None:
        return []
    return [
        {
            "track_id": track.track_id,
            "entity_class": track.entity_class,
            "evidence_tier": track.evidence_tier.value,
            "formal_exposure_weight": track.formal_exposure_weight,
            "confirmed": track.confirmed,
            "detections": [
                {
                    "frame": detection.frame_index,
                    "detection_id": detection.detection_id,
                    "xy": detection.xy.tolist(),
                    "area_px2": detection.area_px2,
                    "confidence": detection.confidence,
                    "sources": list(detection.sources),
                    "metadata": dict(detection.metadata),
                }
                for detection in track.detections
            ],
        }
        for track in observation.tracks
    ]


def _write_artifacts(
    request: CaseEvaluationRequest,
    *,
    times_s: list[float],
    reference_assembly: Sequence[np.ndarray],
    prediction_union: Sequence[np.ndarray],
    full_ious: Sequence[float | None],
    comparison,
    observation: OpenWorldObservation | None,
    subject_rows: Sequence[Mapping[str, Any]],
    topology_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, audit in enumerate(comparison.per_frame):
        subject = subject_rows[index]
        topology = topology_rows[index]
        rows.append(
            {
                "frame": index,
                "time_s": times_s[index],
                "physical_subject_iou": full_ious[index],
                "position_score": audit.get("position_score"),
                "expected_cardinality": audit.get(
                    "expected_cardinality"
                ),
                "prediction_cardinality": audit.get(
                    "formal_prediction_cardinality"
                ),
                "missing_entity_ids": json.dumps(
                    audit.get("missing_entity_ids", [])
                ),
                "extra_track_ids": json.dumps(
                    audit.get("extra_track_ids", [])
                ),
                "birth_track_ids": json.dumps(
                    audit.get("birth_track_ids", [])
                ),
                "death_track_ids": json.dumps(
                    audit.get("death_track_ids", [])
                ),
                "id_switches": json.dumps(audit.get("id_switches", [])),
                "rejected_matches": json.dumps(
                    audit.get("rejected_candidate_matches", [])
                ),
                "subject_score": subject.get("subject"),
                "appearance_score": subject.get("appearance"),
                "string_intact_score": topology.get(
                    "string_intact_score"
                ),
                "broken_string": topology.get("broken_string"),
                "branch_string_evidence": topology.get(
                    "branch_string_evidence"
                ),
                "branch_string_detected": topology.get(
                    "branch_string_detected"
                ),
                "branch_string_component_count": topology.get(
                    "branch_string_component_count"
                ),
                "branch_string_candidate_pixel_count": topology.get(
                    "branch_string_candidate_pixel_count"
                ),
                "branch_string_penalty": topology.get(
                    "branch_string_penalty"
                ),
                "topology_frame_score": topology.get(
                    "topology_frame_score"
                ),
                "branch_components": json.dumps(
                    topology.get("branch_components", [])
                ),
                "branch_rejected_components": json.dumps(
                    topology.get("branch_rejected_components", [])
                ),
                "reference_mask_area": int(
                    np.count_nonzero(reference_assembly[index])
                ),
                "prediction_union_mask_area": int(
                    np.count_nonzero(prediction_union[index])
                ),
            }
        )
    csv_path = request.artifact_dir / "per_frame.csv"
    iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
    position_path = request.artifact_dir / "entity_position_curve.png"
    cardinality_path = (
        request.artifact_dir / "object_cardinality_timeline.png"
    )
    audit_path = request.artifact_dir / "open_world_audit.json"
    tracks_path = request.artifact_dir / "entity_tracks.json"
    write_rows_csv(csv_path, rows)
    save_iou_curve(
        iou_path,
        times_s=times_s,
        ious=list(full_ious),
        case_id=request.case["case_id"],
        scene_name="Open-world pendulum full subject",
    )
    save_iou_curve(
        position_path,
        times_s=times_s,
        ious=[
            (
                float(row["position_score"])
                if row.get("position_score") is not None
                else 0.0
            )
            for row in comparison.per_frame
        ],
        case_id=request.case["case_id"],
        scene_name="Open-world pendulum position",
        series_label="Matched bob continuous position similarity",
        y_label="Position similarity",
        metric_name="score",
    )
    save_series_comparison(
        cardinality_path,
        times_s=times_s,
        reference=[
            float(row["expected_cardinality"])
            for row in comparison.per_frame
        ],
        prediction=[
            float(row["formal_prediction_cardinality"])
            for row in comparison.per_frame
        ],
        ylabel="Physical bob count",
        title=f"Pendulum cardinality — {request.case['case_id']}",
    )
    audit_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tracks_path.write_text(
        json.dumps(
            {
                "observer_diagnostics": (
                    dict(observation.diagnostics)
                    if observation is not None
                    else {"status": "failed_closed"}
                ),
                "tracks": _track_payload(observation),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "per_frame_csv": str(csv_path),
        "physical_subject_iou_curve": str(iou_path),
        "entity_position_curve": str(position_path),
        "object_cardinality_timeline": str(cardinality_path),
        "open_world_audit": str(audit_path),
        "entity_tracks": str(tracks_path),
    }


__all__ = ["PendulumOpenWorldCaseEvaluator"]
