from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file
from ...common.artifacts import (
    save_iou_curve,
    save_series_comparison,
    write_rows_csv,
)
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.csti import build_csti_input_from_frame_matches
from ...common.entities import (
    ObjectTrack,
    ReferenceCapability,
    build_common_time_grid,
    compose_gated_case_score,
    materialize_entity_manifest,
)
from ...common.entities.observer import (
    OpenWorldObservation,
    compare_open_world_tracks,
)
from ...common.errors import ReferenceAnalysisError, SceneAnalysisError
from ...common.geometry import AxisModel, fit_axis
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.masks.sam2 import MaskPrompt, Sam2VideoSegmenter
from ...common.subject import (
    SubjectComparison,
    compare_subjects,
    infer_reference_mode,
)
from ...contracts import CaseEvaluationRequest
from .nbody import (
    NBodyExtractionConfig,
    NBodyScoringConfig,
    extract_nbody_collision_state,
    score_nbody_collision,
)
from .observation import stabilize_collision_instance_masks
from .open_world import (
    build_multiframe_collision_entity_prompts,
    discover_prediction_objects,
    reference_tracks_from_instance_masks,
)
from .v5_visualization import write_collision_v5_visualization


class CollisionOpenWorldCaseEvaluator(ReferenceCaseEvaluator):
    """Shadow v5 evaluator with manifest cardinality and residual discovery."""

    evaluator_id = "collision_1d_open_world_nbody"
    evaluator_version = "2.2"
    sequential_evaluator_version = "2.2"
    robust_evaluator_version = "2.2"
    scene_id = "collision_1d"
    primary_score = "collision_1d_open_world_similarity"
    allow_partial_prediction = True
    fail_closed_on_directed_identity = False
    identity_policy = (
        "causal_track_ids_no_future_renaming_hota_association"
    )

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
                "entity_contract": "manifest_v1",
                "observer": "open_world_v1.2",
                "physics": "nbody_v1.2",
            }
        )

    def describe_observation(self) -> dict[str, Any]:
        assignment = self.config["object_centric_scoring"]["assignment"]
        return {
            "subject": "manifest_declared_n_body_instances",
            "segmentation": self._segmenter.describe(),
            "expected_channel": (
                "manifest_cardinality_multiframe_prompts_plus_sam2"
            ),
            "residual_channel": (
                "per_frame_motion_and_high_confidence_circle_discovery"
            ),
            "identity": (
                "causal_track_ids_plus_evidence_tiered_hungarian_with_null"
            ),
            "assignment": {
                "method": "evidence_tiered_hungarian_with_null",
                "position_kernel": "reference_scaled_cauchy",
                "minimum_match_position_similarity": float(
                    assignment["minimum_match_position_similarity"]
                ),
                "rejected_edge_policy": "missing_reference_plus_extra_prediction",
            },
            "coordinate_system": "frozen_reference_track_axis",
            "cardinality": "case_entity_manifest_not_scene_constant",
        }

    def _prepare_identity_context(
        self,
        request: CaseEvaluationRequest,
        *,
        reference_frames: Sequence[np.ndarray],
        entity_ids: Sequence[str],
    ) -> object | None:
        del request, reference_frames, entity_ids
        return None

    def _observe_expected_role(
        self,
        frames: list[np.ndarray],
        *,
        expected_count: int,
        entity_ids: Sequence[str],
        observation_role: str,
        identity_context: object | None,
        available: Sequence[bool] | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[np.ndarray]],
        list[np.ndarray],
        dict[str, Any],
    ]:
        del identity_context
        return self._observe_expected(
            frames,
            expected_count=expected_count,
            entity_ids=entity_ids,
            available=available,
            observation_role=observation_role,
        )

    def _postprocess_expected_masks(
        self,
        masks: list[list[np.ndarray]],
        prompts: list[MaskPrompt],
        *,
        observation_role: str,
    ) -> tuple[list[list[np.ndarray]], dict[str, Any] | None]:
        del prompts, observation_role
        return masks, None

    def _fixed_prediction_track_ids(
        self, entity_ids: Sequence[str]
    ) -> Mapping[str, str] | None:
        del entity_ids
        return None

    def _terminal_prediction_observation_failure(
        self,
        observation: OpenWorldObservation,
        *,
        entity_ids: Sequence[str],
        frame_count: int,
    ) -> tuple[dict[str, str], dict[str, Any]] | None:
        del observation, entity_ids, frame_count
        return None

    def _observe_expected(
        self,
        frames: list[np.ndarray],
        *,
        expected_count: int,
        entity_ids: Sequence[str],
        available: Sequence[bool] | None = None,
        prompts_override: Sequence[MaskPrompt] | None = None,
        prompt_metadata_override: Mapping[str, Any] | None = None,
        observation_role: str = "unspecified",
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[np.ndarray]],
        list[np.ndarray],
        dict[str, Any],
    ]:
        config = self.config["multi_frame_observation"]
        if not frames:
            raise SceneAnalysisError(
                "collision_empty_observation",
                "collision video has no sampled frames",
            )
        if len(entity_ids) != expected_count:
            raise ValueError(
                "entity ID count differs from expected collision cardinality"
            )
        if prompts_override is None:
            prompts, prompt_metadata = (
                build_multiframe_collision_entity_prompts(
                    frames,
                    expected_count=expected_count,
                    config=config,
                    entity_ids=entity_ids,
                    available=available,
                )
            )
        else:
            prompts = list(prompts_override)
            prompt_metadata = dict(prompt_metadata_override or {})
        if len(prompts) != expected_count:
            raise SceneAnalysisError(
                "collision_prompt_cardinality_mismatch",
                f"prompt builder returned {len(prompts)} channels for "
                f"{expected_count} entities",
            )
        masks, segmentation = self._segmenter.segment_instances(
            frames,
            prompts=prompts,
            temporary_prefix="physbench_collision_v5_",
            exclusive_masks=True,
        )
        if len(masks) != expected_count or any(
            len(instance) != len(frames) for instance in masks
        ):
            raise SceneAnalysisError(
                "collision_segmentation_cardinality_mismatch",
                "SAM2 instance channels differ from manifest cardinality "
                "or common timeline",
            )
        expected_shape = frames[0].shape[:2]
        if any(
            np.asarray(mask).shape != expected_shape
            for instance in masks
            for mask in instance
        ):
            raise SceneAnalysisError(
                "collision_segmentation_shape_mismatch",
                "SAM2 masks differ from the normalized video canvas",
            )
        masks, stabilization = stabilize_collision_instance_masks(
            masks,
            prompts,
            config=self.config["quality"],
        )
        if available is not None:
            if len(available) != len(frames):
                raise ValueError(
                    "availability mask differs from sampled frame count"
                )
            for instance in masks:
                for index, is_available in enumerate(available):
                    if not is_available:
                        instance[index] = np.zeros_like(instance[index])
        masks, identity_latch = self._postprocess_expected_masks(
            masks,
            prompts,
            observation_role=observation_role,
        )
        quality = self.config["quality"]
        minimum_area = int(quality["minimum_mask_pixels"])
        maximum_area = int(
            frames[0].shape[0]
            * frames[0].shape[1]
            * float(quality["maximum_mask_area_ratio"])
        )
        frame_count = len(frames)
        xy = np.full(
            (frame_count, expected_count, 2),
            np.nan,
            dtype=np.float64,
        )
        valid = np.zeros((frame_count, expected_count), dtype=bool)
        radii = np.full(
            (frame_count, expected_count),
            np.nan,
            dtype=np.float64,
        )
        union_masks: list[np.ndarray] = []
        for frame_index in range(frame_count):
            union = np.zeros(frames[0].shape[:2], dtype=np.uint8)
            for object_index in range(expected_count):
                mask = masks[object_index][frame_index]
                area = int(np.count_nonzero(mask))
                union = cv2.bitwise_or(union, mask)
                if area < minimum_area or area > maximum_area:
                    continue
                moments = cv2.moments(mask, binaryImage=True)
                if abs(float(moments["m00"])) <= 1e-12:
                    continue
                xy[frame_index, object_index] = [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ]
                radii[frame_index, object_index] = math.sqrt(
                    area / math.pi
                )
                valid[frame_index, object_index] = True
            union_masks.append(union)
        radii = _fill_positive_radii(
            radii,
            valid,
            fallback=np.asarray(
                [
                    float(
                        prompt.metadata.get(
                            "circle_xyr",
                            [0.0, 0.0, 8.0],
                        )[2]
                    )
                    for prompt in prompts
                ]
            ),
        )
        return (
            xy,
            valid,
            radii,
            masks,
            union_masks,
            {
                "prompt_builder": prompt_metadata,
                "segmentation": segmentation,
                "mask_stabilization": stabilization,
                "valid_track_ratios": np.mean(valid, axis=0).tolist(),
                **(
                    {"identity_latch": identity_latch}
                    if identity_latch is not None
                    else {}
                ),
            },
        )

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
            entities = _ordered_collision_entities(manifest)
        except ReferenceAnalysisError:
            raise
        except ValueError as exc:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_invalid",
                f"collision entity manifest is invalid: {exc}",
            ) from exc
        entity_ids = [entity.entity_id for entity in entities]
        if len(entities) < 2:
            raise ReferenceAnalysisError(
                "reference_collision_entity_count_invalid",
                "collision manifest must declare at least two entities",
            )
        time_grid = build_common_time_grid(times_s)
        quality = self.config["quality"]
        extraction_config = NBodyExtractionConfig(
            **self.config.get("nbody_extraction", {})
        )
        scoring_config = NBodyScoringConfig(
            **self.config.get("nbody_scoring", {})
        )
        try:
            identity_context = self._prepare_identity_context(
                request,
                reference_frames=reference_video.frames,
                entity_ids=entity_ids,
            )
            (
                reference_xy,
                reference_valid,
                reference_radii,
                reference_masks,
                reference_union,
                reference_observation,
            ) = self._observe_expected_role(
                reference_video.frames,
                expected_count=len(entities),
                entity_ids=entity_ids,
                observation_role="reference",
                identity_context=identity_context,
            )
            minimum_reference_ratio = float(
                quality.get("minimum_reference_valid_frame_ratio", 0.2)
            )
            reference_ratios = np.mean(reference_valid, axis=0)
            if np.any(reference_ratios < minimum_reference_ratio):
                raise SceneAnalysisError(
                    "insufficient_reference_entity_tracks",
                    "reference per-entity coverage "
                    f"{reference_ratios.tolist()} is below "
                    f"{minimum_reference_ratio:g}",
                )
            reference_tracks = reference_tracks_from_instance_masks(
                reference_masks,
                entity_ids=entity_ids,
                entity_class="ball",
                time_grid=time_grid,
                minimum_area=int(quality["minimum_mask_pixels"]),
                maximum_area_ratio=float(
                    quality["maximum_mask_area_ratio"]
                ),
            )
            axis = _reference_axis(reference_xy, reference_valid)
            masses = np.asarray(
                [
                    _entity_attribute(
                        entity,
                        "mass",
                        expected_unit="kg",
                        positive=True,
                    )
                    for entity in entities
                ],
                dtype=np.float64,
            )
            reference_state = extract_nbody_collision_state(
                reference_xy,
                reference_valid,
                times_s,
                entity_ids=entity_ids,
                radii_px=reference_radii,
                masses_kg=masses,
                axis_origin_xy=axis.origin_xy,
                axis_direction_xy=axis.direction_xy,
                config=extraction_config,
            )
        except (SceneAnalysisError, ValueError, RuntimeError) as exc:
            raise ReferenceAnalysisError(
                "reference_collision_open_world_observation_failed",
                f"reference collision observation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        (
            prediction_frames,
            prediction_available,
            prediction_timeline_failure,
        ) = _normalized_prediction_timeline(
            getattr(prediction_video, "frames", []),
            available=getattr(prediction_video, "available", None),
            frame_count=len(times_s),
            frame_shape=reference_video.frames[0].shape,
        )
        prediction_failures: list[dict[str, str]] = []
        directed_identity_failure: dict[str, str] | None = None
        residual_after_identity_failure: dict[str, Any] | None = None
        terminal_prediction_observation: dict[str, Any] | None = None
        if prediction_timeline_failure is not None:
            prediction_failures.append(prediction_timeline_failure)
        try:
            (
                _prediction_direct_xy,
                _prediction_direct_valid,
                _prediction_direct_radii,
                prediction_masks,
                _prediction_direct_union,
                prediction_direct_observation,
            ) = self._observe_expected_role(
                prediction_frames,
                expected_count=len(entities),
                entity_ids=entity_ids,
                observation_role="prediction",
                identity_context=identity_context,
                available=prediction_available,
            )
        except Exception as exc:
            prediction_failure = {
                "code": getattr(
                    exc,
                    "code",
                    "prediction_directed_observation_failed",
                ),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(prediction_failure)
            directed_identity_failure = prediction_failure
            height, width = reference_video.frames[0].shape[:2]
            prediction_masks = [
                [
                    np.zeros((height, width), dtype=np.uint8)
                    for _ in times_s
                ]
                for _ in entities
            ]
            prediction_direct_observation = {
                "status": (
                    "failed_terminal_residual_discovery_diagnostic_only"
                    if self.fail_closed_on_directed_identity
                    else "failed_but_residual_discovery_continued"
                ),
                **prediction_failure,
            }

        try:
            prediction_objects = discover_prediction_objects(
                prediction_frames,
                directed_instance_masks=prediction_masks,
                time_grid=time_grid,
                observation_config=self.config[
                    "multi_frame_observation"
                ],
                quality_config=quality,
                available=prediction_available,
            )
        except Exception as exc:
            prediction_failure = {
                "code": "prediction_residual_observation_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(prediction_failure)
            # Open-world discovery is the only channel that can establish
            # that no additional physical object exists. Falling back to the
            # directed manifest-cardinality masks would therefore fail open:
            # a prediction with hidden extras could receive a perfect score.
            # The protocol freezes prediction observation failures as an
            # evaluated conservative zero, while retaining the directed
            # observation diagnostics above for provenance.
            prediction_objects = _empty_prediction_observation(
                len(times_s),
                code=prediction_failure["code"],
                reason=prediction_failure["reason"],
            )

        if (
            self.fail_closed_on_directed_identity
            and directed_identity_failure is not None
        ):
            residual_after_identity_failure = {
                "status": "diagnostic_only_not_eligible_for_role_recovery",
                "track_count": len(prediction_objects.tracks),
                "overflow_counts": prediction_objects.overflow_counts.tolist(),
                "diagnostics": prediction_objects.diagnostics,
            }
            prediction_objects = _empty_prediction_observation(
                len(times_s),
                code=directed_identity_failure["code"],
                reason=(
                    "directed collision identity is unconfirmed; residual "
                    "tracks are diagnostic-only and cannot recover roles: "
                    + directed_identity_failure["reason"]
                ),
            )

        if directed_identity_failure is None:
            terminal_observation = self._terminal_prediction_observation_failure(
                prediction_objects,
                entity_ids=entity_ids,
                frame_count=len(times_s),
            )
            if terminal_observation is not None:
                failure, terminal_prediction_observation = terminal_observation
                prediction_failures.append(failure)
                prediction_objects = _empty_prediction_observation(
                    len(times_s),
                    code=failure["code"],
                    reason=failure["reason"],
                )

        frame_diagonal = float(
            np.hypot(
                reference_video.frames[0].shape[1],
                reference_video.frames[0].shape[0],
            )
        )
        minimum_match_position_similarity = float(
            self.config["object_centric_scoring"]["assignment"][
                "minimum_match_position_similarity"
            ]
        )
        fixed_prediction_track_ids = self._fixed_prediction_track_ids(
            entity_ids
        )
        try:
            comparison = compare_open_world_tracks(
                reference_tracks=reference_tracks,
                prediction_observation=prediction_objects,
                time_grid=time_grid,
                frame_diagonal_px=frame_diagonal,
                minimum_match_position_similarity=(
                    minimum_match_position_similarity
                ),
                fixed_entity_track_ids=fixed_prediction_track_ids,
            )
        except Exception as exc:
            failure = {
                "code": "prediction_open_world_comparison_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            prediction_objects = _empty_prediction_observation(
                len(times_s),
                code=failure["code"],
                reason=failure["reason"],
            )
            comparison = compare_open_world_tracks(
                reference_tracks=reference_tracks,
                prediction_observation=prediction_objects,
                time_grid=time_grid,
                frame_diagonal_px=frame_diagonal,
                minimum_match_position_similarity=(
                    minimum_match_position_similarity
                ),
                fixed_entity_track_ids=fixed_prediction_track_ids,
            )

        prediction_entity_ids = list(entity_ids)
        prediction_state = None
        try:
            (
                prediction_xy,
                prediction_valid,
                prediction_radii,
                prediction_entity_ids,
                prediction_masses,
            ) = _prediction_nbody_inputs(
                comparison_matches=comparison.matches,
                observation=prediction_objects,
                entity_ids=entity_ids,
                reference_radii=reference_radii,
                reference_masses=masses,
                time_grid=time_grid,
            )
            prediction_state = extract_nbody_collision_state(
                prediction_xy,
                prediction_valid,
                times_s,
                entity_ids=prediction_entity_ids,
                radii_px=prediction_radii,
                masses_kg=prediction_masses,
                axis_origin_xy=axis.origin_xy,
                axis_direction_xy=axis.direction_xy,
                config=extraction_config,
            )
            nbody_score = score_nbody_collision(
                reference_state,
                prediction_state,
                frame_diagonal_px=frame_diagonal,
                config=scoring_config,
            )
            if not math.isfinite(float(nbody_score["score"])):
                raise ValueError("N-body scorer returned a non-finite score")
        except Exception as exc:
            failure = {
                "code": "prediction_nbody_scoring_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            nbody_score = _degraded_nbody_score(failure)

        prediction_union = _observation_union_masks(
            prediction_objects,
            frame_count=len(times_s),
            shape=reference_union[0].shape,
        )
        union_ious = [
            observed_mask_iou(reference_union[index], prediction_union[index])
            for index in range(len(times_s))
        ]
        reference_mode = infer_reference_mode(request.case)
        condition_provenance: dict[str, Any] = {}
        try:
            (
                matched_reference_masks,
                matched_prediction_masks,
            ) = _matched_subject_masks(
                reference_masks=reference_masks,
                entity_ids=entity_ids,
                prediction_observation=prediction_objects,
                comparison_matches=comparison.matches,
                frame_count=len(times_s),
                shape=reference_union[0].shape,
            )
            subject_reference_masks = matched_reference_masks
            if not any(
                np.count_nonzero(mask) for mask in subject_reference_masks
            ):
                # With no match there is no matched-only appearance content
                # to compare. Use the observable GT union against empty
                # prediction masks to obtain a conservative finite zero; the
                # integrity gate is already zero in this branch.
                subject_reference_masks = reference_union
            condition_frame = None
            condition_mask = None
            if reference_mode == "parent_physics_reference":
                condition_frame, condition_provenance = (
                    _load_condition_frame(
                        request,
                        target_shape=prediction_frames[0].shape,
                    )
                )
                condition_mask = matched_prediction_masks[0]
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=subject_reference_masks,
                prediction_masks=matched_prediction_masks,
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
                condition_frame=condition_frame,
                condition_mask=condition_mask,
            )
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            failure = {
                "code": "prediction_subject_comparison_failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            subject = _degraded_subject_comparison(
                len(times_s),
                reference_mode=reference_mode,
            )
        content_components = {
            "nbody_physics": float(nbody_score["score"]),
            "shape": subject.components.get("shape"),
            "appearance": subject.components.get("appearance"),
        }
        content_weights = self.config["object_centric_scoring"][
            "content_weights"
        ]
        composition = compose_gated_case_score(
            comparison.integrity.integrity_gate,
            content_components,
            {
                key: float(value)
                for key, value in content_weights.items()
            },
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

        rows = _per_frame_rows(
            times_s=times_s,
            comparison=comparison.per_frame,
            union_ious=union_ious,
            subject_rows=subject.per_frame,
        )
        csv_path = request.artifact_dir / "per_frame.csv"
        iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
        distance_path = request.artifact_dir / "entity_position_curve.png"
        cardinality_path = (
            request.artifact_dir / "object_cardinality_timeline.png"
        )
        write_rows_csv(csv_path, rows)
        save_iou_curve(
            iou_path,
            times_s=times_s,
            ious=union_ious,
            case_id=request.case["case_id"],
            scene_name="Open-world N-body collision",
        )
        per_frame_position = [
            float(row["position_diagnostic_score"])
            for row in comparison.per_frame
        ]
        save_iou_curve(
            distance_path,
            times_s=times_s,
            ious=per_frame_position,
            case_id=request.case["case_id"],
            scene_name="Open-world N-body collision position",
            series_label=(
                "Matched/nearest-rejected continuous position score"
            ),
            y_label="Position similarity",
            metric_name="score",
        )
        reference_counts = [
            float(
                len(row["matches"]) + len(row["missing_entity_ids"])
            )
            for row in comparison.per_frame
        ]
        prediction_counts = [
            float(row["formal_prediction_cardinality"])
            for row in comparison.per_frame
        ]
        save_series_comparison(
            cardinality_path,
            times_s=times_s,
            reference=reference_counts,
            prediction=prediction_counts,
            ylabel="Observed physical objects",
            title=(
                "Collision object cardinality — "
                f"{request.case['case_id']}"
            ),
        )
        visualization_artifacts: dict[str, Any] = {}
        visualization_config = self.config.get("visualization")
        if visualization_config is not None:
            visualization_artifacts = write_collision_v5_visualization(
                request,
                config=visualization_config,
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_frames,
                reference_masks=reference_masks,
                reference_xy=reference_xy,
                reference_valid=reference_valid,
                entity_ids=entity_ids,
                prediction_observation=prediction_objects,
                comparison=comparison.per_frame,
                reference_union=reference_union,
                prediction_union=prediction_union,
                union_ious=union_ious,
                reference_state=reference_state,
                prediction_state=prediction_state,
                prediction_available=prediction_available,
                reference_role=(
                    "PHYSICS REFERENCE"
                    if reference_mode == "parent_physics_reference"
                    else "REFERENCE"
                ),
                score_summary={
                    "score": score,
                    "components": {
                        "physics": float(nbody_score["score"]),
                        "integrity": float(comparison.integrity.score),
                        **{
                            key: float(value)
                            for key, value in nbody_score.get(
                                "components", {}
                            ).items()
                            if value is not None
                        },
                    },
                },
                has_issues=bool(
                    prediction_failures
                    or getattr(comparison, "failed", False)
                ),
            )
        integrity = comparison.integrity.to_dict()
        primary_metric = {
            **composition.to_dict(),
            "score": score,
            "content_components": content_components,
        }
        csti_input = (
            build_csti_input_from_frame_matches(
                reference_capability=manifest.reference_capability,
                times_s=times_s,
                frame_shape=reference_union[0].shape,
                expected_entities=tuple(
                    (entity.entity_id, entity.role_id)
                    for entity in entities
                ),
                reference_masks_by_entity={
                    entity_id: tuple(reference_masks[index])
                    for index, entity_id in enumerate(entity_ids)
                },
                prediction_observation=prediction_objects,
                matches=comparison.matches,
            )
            if (
                getattr(self, "csti_enabled", False)
                and manifest.reference_capability
                is ReferenceCapability.SAME_CASE_GT
            )
            else None
        )
        return SceneAnalysis(
            score=score,
            metrics={
                # ``robust_subject_v3`` advertises this stable Task-level
                # metric name through ``ReferenceCaseEvaluator.describe``.
                # Keep the collision-specific name as an exact diagnostic
                # alias, but never publish a primary_score that is absent
                # from a normal evaluated result.
                "scene_subject_state_similarity": dict(primary_metric),
                "collision_1d_open_world_similarity": primary_metric,
                "object_centric_integrity": integrity,
                "collision_nbody_state_similarity": nbody_score,
                "physical_subject_similarity": subject.to_metric(
                    weights={
                        (
                            "appearance"
                            if reference_mode
                            == "parent_physics_reference"
                            else key
                        ): (
                            1.0
                            if reference_mode
                            == "parent_physics_reference"
                            else float(value)
                        )
                        for key, value in (
                            {"appearance": 1.0}.items()
                            if reference_mode
                            == "parent_physics_reference"
                            else self.config["subject_scoring"][
                                "weights"
                            ].items()
                        )
                    },
                ),
                "physical_subject_mask_iou": {
                    **summarize_mask_ious(union_ious),
                    "role": (
                        "required_full-set_visual_diagnostic_not_content_gate"
                    ),
                },
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": bool(prediction_failures),
                "degradation_codes": [
                    failure["code"] for failure in prediction_failures
                ],
                "degradation_reason": (
                    "; ".join(
                        failure["reason"]
                        for failure in prediction_failures
                    )
                    if prediction_failures
                    else None
                ),
                "reference_valid_track_ratios": np.mean(
                    reference_valid, axis=0
                ).tolist(),
                "prediction_directed_observation": (
                    prediction_direct_observation
                ),
                "prediction_open_world_observation": (
                    prediction_objects.diagnostics
                ),
                "expected_entity_count": len(entities),
                "prediction_track_count": len(prediction_objects.tracks),
                "prediction_residual_body_count": max(
                    len(prediction_entity_ids) - len(entity_ids),
                    0,
                ),
                "available_prediction_frames": int(
                    np.count_nonzero(prediction_available)
                ),
                "expected_prediction_frames": len(times_s),
                **(
                    {
                        "directed_identity_failure_terminal": (
                            directed_identity_failure is not None
                        ),
                        "prediction_residual_after_identity_failure": (
                            residual_after_identity_failure
                        ),
                        "terminal_prediction_observation": (
                            terminal_prediction_observation
                        ),
                    }
                    if self.fail_closed_on_directed_identity
                    else {}
                ),
            },
            artifacts={
                "per_frame_csv": str(csv_path),
                "physical_subject_iou_curve": str(iou_path),
                "entity_position_curve": str(distance_path),
                "object_cardinality_timeline": str(cardinality_path),
                **visualization_artifacts,
            },
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                },
                "observation": {
                    "reference": reference_observation,
                    "prediction_directed": prediction_direct_observation,
                    "prediction_open_world": (
                        prediction_objects.diagnostics
                    ),
                },
                "identity_policy": self.identity_policy,
                "position_kernel": (
                    "reference_scaled_anisotropic_cauchy"
                ),
                "subject_content_mask_policy": (
                    "per_frame_matched_entity_union_only"
                ),
                "subject_reference_mode": reference_mode,
                **condition_provenance,
            },
            csti_input=csti_input,
        )


def _ordered_collision_entities(manifest) -> list[Any]:
    if manifest.scene_id != "collision_1d":
        raise ReferenceAnalysisError(
            "reference_entity_manifest_scene_mismatch",
            "collision evaluator requires a collision_1d entity manifest",
        )
    entities = list(manifest.entities)
    indexed: list[tuple[int, Any]] = []
    for entity in entities:
        if entity.entity_class != "ball":
            raise ReferenceAnalysisError(
                "reference_collision_entity_class_invalid",
                f"collision entity {entity.entity_id} has unsupported class "
                f"{entity.entity_class!r}; expected 'ball'",
            )
        raw_order = entity.condition_anchor.get("initial_order_index")
        if (
            isinstance(raw_order, bool)
            or not isinstance(raw_order, int)
            or raw_order < 0
        ):
            raise ReferenceAnalysisError(
                "reference_collision_entity_anchor_missing",
                f"collision entity {entity.entity_id} lacks a non-negative "
                "integer condition_anchor.initial_order_index",
            )
        _entity_attribute(
            entity, "mass", expected_unit="kg", positive=True
        )
        _entity_attribute(
            entity, "radius", expected_unit="m", positive=True
        )
        _entity_attribute(
            entity,
            "initial_velocity",
            expected_unit="m/s",
            positive=False,
        )
        indexed.append((raw_order, entity))
    orders = [order for order, _ in indexed]
    if sorted(orders) != list(range(len(entities))):
        raise ReferenceAnalysisError(
            "reference_collision_entity_anchor_mismatch",
            "collision initial_order_index values must uniquely cover "
            f"0..{len(entities) - 1}",
        )
    return [entity for _, entity in sorted(indexed)]


def _normalized_prediction_timeline(
    frames: Sequence[np.ndarray],
    *,
    available: Sequence[bool] | None,
    frame_count: int,
    frame_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], list[bool], dict[str, str] | None]:
    """Pad malformed/short prediction samples with unavailable neutral frames."""

    source_frames = list(frames)
    availability_valid = True
    if available is None:
        source_available = [True] * len(source_frames)
    else:
        source_available = list(available)
        availability_valid = (
            len(source_available) == len(source_frames)
            and all(
                isinstance(value, (bool, np.bool_))
                for value in source_available
            )
        )
        if not availability_valid:
            source_available = [False] * len(source_frames)
    normalized: list[np.ndarray] = []
    normalized_available: list[bool] = []
    invalid_frame = False
    for frame_index in range(frame_count):
        if frame_index >= len(source_frames):
            normalized.append(np.zeros(frame_shape, dtype=np.uint8))
            normalized_available.append(False)
            continue
        value = np.asarray(source_frames[frame_index])
        usable = (
            bool(source_available[frame_index])
            and value.shape == frame_shape
            and np.issubdtype(value.dtype, np.number)
            and np.isfinite(value).all()
        )
        if usable:
            normalized.append(
                np.clip(value, 0, 255).astype(np.uint8, copy=False)
            )
            normalized_available.append(True)
        else:
            normalized.append(np.zeros(frame_shape, dtype=np.uint8))
            normalized_available.append(False)
            invalid_frame = True
    timeline_degraded = (
        len(source_frames) != frame_count
        or not availability_valid
        or invalid_frame
        or not all(normalized_available)
    )
    if not timeline_degraded:
        return normalized, normalized_available, None
    return (
        normalized,
        normalized_available,
        {
            "code": "prediction_partial_timeline",
            "reason": (
                "prediction supplies "
                f"{sum(normalized_available)}/{frame_count} usable common "
                "timeline samples; unavailable samples are scored as missing"
            ),
        },
    )


def _empty_prediction_observation(
    frame_count: int,
    *,
    code: str,
    reason: str,
) -> OpenWorldObservation:
    return OpenWorldObservation(
        tracks=(),
        overflow_counts=np.zeros(frame_count, dtype=np.float64),
        diagnostics={
            "status": "failed_as_empty_prediction",
            "code": code,
            "reason": reason,
        },
    )


def _degraded_nbody_score(failure: dict[str, str]) -> dict[str, Any]:
    components = {
        "track_position": 0.0,
        "contact_graph": 0.0,
        "velocity": 0.0,
        "momentum": 0.0,
        "nonpenetration": 0.0,
    }
    return {
        "score": 0.0,
        "components": components,
        "details": {
            "degraded": True,
            "degradation_code": failure["code"],
            "degradation_reason": failure["reason"],
        },
        "weights_used": {},
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
            "parent-reference collision case has no first_frame asset",
        )
    asset_root = request.asset_root.resolve()
    path = (asset_root / value).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise ReferenceAnalysisError(
            "reference_condition_path_escape",
            f"conditioned first frame escapes dataset root: {value}",
        ) from exc
    source = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if source is None:
        raise ReferenceAnalysisError(
            "reference_condition_frame_unreadable",
            f"cannot read conditioned first frame: {path}",
        )
    target_height, target_width = target_shape[:2]
    source_height, source_width = source.shape[:2]
    scale = min(
        target_width / max(source_width, 1),
        target_height / max(source_height, 1),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        source,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    output = np.full(
        (target_height, target_width, 3),
        int(request.evaluator_config["spatial"].get("pad_value", 0)),
        dtype=np.uint8,
    )
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    output[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    return output, {
        "condition_frame": str(path),
        "condition_frame_sha256": sha256_file(path),
        "condition_mask_source": "matched_prediction_frame_zero_roi",
        "condition_spatial_transform": {
            "policy": "preserve_aspect_ratio_letterbox",
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
                "frame": frame_index,
                "position": 0.0,
                "shape": 0.0,
                "appearance": 0.0,
                "subject": 0.0,
            }
            for frame_index in range(frame_count)
        ],
        reference_observed_ratio=1.0,
        prediction_observed_ratio=0.0,
        comparable_reference=(
            "same_case_ground_truth_video"
            if reference_mode == "same_case_reference"
            else "case_condition_first_frame_appearance_only"
        ),
    )


def _entity_attribute(
    entity,
    name: str,
    *,
    expected_unit: str,
    positive: bool,
) -> float:
    values = {
        attribute.name: attribute for attribute in entity.physical_attributes
    }
    if name not in values:
        raise ReferenceAnalysisError(
            "reference_entity_physics_missing",
            f"entity {entity.entity_id} lacks physical attribute {name}",
        )
    value = values[name]
    if not value.annotated:
        raise ReferenceAnalysisError(
            "reference_entity_physics_unannotated",
            f"entity {entity.entity_id} attribute {name} is unannotated",
        )
    if value.unit != expected_unit:
        raise ReferenceAnalysisError(
            "reference_entity_physics_unit_invalid",
            f"entity {entity.entity_id} attribute {name} uses "
            f"{value.unit!r}; expected {expected_unit!r}",
        )
    numeric = float(value.value)
    if not math.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "positive" if positive else "finite"
        raise ReferenceAnalysisError(
            "reference_entity_physics_value_invalid",
            f"entity {entity.entity_id} attribute {name} must be {qualifier}",
        )
    return numeric


def _fill_positive_radii(
    radii: np.ndarray,
    valid: np.ndarray,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    output = np.asarray(radii, dtype=np.float64).copy()
    for object_index in range(output.shape[1]):
        observed = output[valid[:, object_index], object_index]
        fill = (
            float(np.median(observed))
            if len(observed)
            else float(fallback[object_index])
        )
        output[~np.isfinite(output[:, object_index]), object_index] = max(
            fill, 1.0
        )
    return output


def _reference_axis(
    xy: np.ndarray,
    valid: np.ndarray,
) -> AxisModel:
    points = xy[valid]
    if len(points) < 4:
        raise SceneAnalysisError(
            "insufficient_reference_axis_points",
            "reference has fewer than four valid entity centers",
        )
    axis = fit_axis(points, orient_with_time=False)
    if axis.direction_xy[0] < 0.0:
        axis = AxisModel(
            origin_xy=axis.origin_xy,
            direction_xy=-axis.direction_xy,
            normal_xy=-axis.normal_xy,
            explained_ratio=axis.explained_ratio,
        )
    return axis


def _prediction_nbody_inputs(
    *,
    comparison_matches,
    observation: OpenWorldObservation,
    entity_ids: Sequence[str],
    reference_radii: np.ndarray,
    reference_masses: np.ndarray,
    time_grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    frame_count = len(time_grid.times_s)
    track_objects = {
        track.track_id: track.to_object_track(
            frame_count=frame_count,
            time_weights_s=time_grid.cell_weights_s,
        )
        for track in observation.tracks
    }
    expected_count = len(entity_ids)
    centers: list[np.ndarray] = [
        np.full((frame_count, 2), np.nan, dtype=np.float64)
        for _ in entity_ids
    ]
    valid: list[np.ndarray] = [
        np.zeros(frame_count, dtype=bool) for _ in entity_ids
    ]
    radii: list[np.ndarray] = [
        np.full(frame_count, np.nan, dtype=np.float64)
        for _ in entity_ids
    ]
    entity_lookup = {
        entity_id: index for index, entity_id in enumerate(entity_ids)
    }
    matched_frames_by_track: dict[str, set[int]] = {}
    for match in comparison_matches:
        entity_index = entity_lookup[match.entity_id]
        track = track_objects[match.track_id]
        frame_index = match.frame_index
        centers[entity_index][frame_index] = track.xy[frame_index]
        valid[entity_index][frame_index] = True
        if (
            track.areas_px2 is not None
            and np.isfinite(track.areas_px2[frame_index])
        ):
            radii[entity_index][frame_index] = math.sqrt(
                float(track.areas_px2[frame_index]) / math.pi
            )
        matched_frames_by_track.setdefault(match.track_id, set()).add(
            frame_index
        )

    output_ids = list(entity_ids)
    masses = [float(value) for value in reference_masses]
    median_mass = float(np.median(reference_masses))
    reference_radius_fill = np.median(reference_radii, axis=0)
    for index in range(expected_count):
        radii[index][~np.isfinite(radii[index])] = max(
            float(reference_radius_fill[index]), 1.0
        )
    for track_id, track in track_objects.items():
        if (
            # Weak static-circle evidence still pays an existence penalty.
            # Only a full participant can create a body or contact event in
            # the scene-specific N-body model.
            float(
                track.metadata.get("formal_exposure_weight", 0.0)
            )
            < 1.0 - 1e-12
        ):
            continue
        unmatched = track.observed.copy()
        for frame_index in matched_frames_by_track.get(track_id, ()):
            unmatched[frame_index] = False
        unmatched_indices = np.flatnonzero(unmatched)
        if not len(unmatched_indices):
            continue
        # Keep one residual channel per persistent prediction track. Internal
        # gaps are legitimate: those detections were assigned to an expected
        # entity at that instant. This gives a strict slot partition even for
        # alternating match/unmatch patterns and retains singleton residual
        # evidence for penetration/contact diagnostics where applicable.
        segment_valid = unmatched
        segment_xy = np.full(
            (frame_count, 2),
            np.nan,
            dtype=np.float64,
        )
        segment_xy[segment_valid] = track.xy[segment_valid]
        output_ids.append(f"residual:{track_id}")
        centers.append(segment_xy)
        valid.append(segment_valid)
        radius = np.full(frame_count, np.nan, dtype=np.float64)
        assert track.areas_px2 is not None
        finite = segment_valid & np.isfinite(track.areas_px2)
        radius[finite] = np.sqrt(track.areas_px2[finite] / math.pi)
        fill = (
            float(np.median(radius[finite]))
            if finite.any()
            else float(np.median(reference_radius_fill))
        )
        radius[~np.isfinite(radius)] = max(fill, 1.0)
        radii.append(radius)
        masses.append(median_mass)
    return (
        np.stack(centers, axis=1),
        np.stack(valid, axis=1),
        np.stack(radii, axis=1),
        output_ids,
        np.asarray(masses, dtype=np.float64),
    )


def _matched_subject_masks(
    *,
    reference_masks: Sequence[Sequence[np.ndarray]],
    entity_ids: Sequence[str],
    prediction_observation: OpenWorldObservation,
    comparison_matches,
    frame_count: int,
    shape: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(reference_masks) != len(entity_ids):
        raise ValueError(
            "reference masks and entity IDs have different cardinality"
        )
    reference_lookup = {
        entity_id: index for index, entity_id in enumerate(entity_ids)
    }
    prediction_masks: dict[tuple[str, int], np.ndarray] = {}
    for track in prediction_observation.tracks:
        for detection in track.detections:
            if detection.frame_index >= frame_count:
                raise ValueError(
                    "prediction detection lies outside the common timeline"
                )
            if detection.mask is not None:
                prediction_masks[
                    (track.track_id, detection.frame_index)
                ] = np.asarray(detection.mask, dtype=np.uint8)
    reference_output = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    prediction_output = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    seen_reference_slots: set[tuple[int, str]] = set()
    seen_prediction_slots: set[tuple[int, str]] = set()
    for match in comparison_matches:
        frame_index = int(match.frame_index)
        reference_slot = frame_index, match.entity_id
        prediction_slot = frame_index, match.track_id
        if (
            reference_slot in seen_reference_slots
            or prediction_slot in seen_prediction_slots
        ):
            raise ValueError(
                "matched subject masks require one-to-one frame matches"
            )
        seen_reference_slots.add(reference_slot)
        seen_prediction_slots.add(prediction_slot)
        reference_index = reference_lookup[match.entity_id]
        reference_mask = np.asarray(
            reference_masks[reference_index][frame_index],
            dtype=np.uint8,
        )
        if reference_mask.shape != shape:
            raise ValueError(
                "reference entity mask differs from normalized canvas"
            )
        reference_output[frame_index] = cv2.bitwise_or(
            reference_output[frame_index],
            reference_mask,
        )
        prediction_mask = prediction_masks.get(
            (match.track_id, frame_index)
        )
        if prediction_mask is None:
            continue
        if prediction_mask.shape != shape:
            raise ValueError(
                "prediction entity mask differs from normalized canvas"
            )
        prediction_output[frame_index] = cv2.bitwise_or(
            prediction_output[frame_index],
            prediction_mask,
        )
    return reference_output, prediction_output


def _observation_union_masks(
    observation: OpenWorldObservation,
    *,
    frame_count: int,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    output = [
        np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)
    ]
    for track in observation.tracks:
        if track.formal_exposure_weight <= 0.0:
            continue
        for detection in track.detections:
            if detection.mask is None:
                continue
            output[detection.frame_index] = cv2.bitwise_or(
                output[detection.frame_index],
                np.asarray(detection.mask, dtype=np.uint8),
            )
    return output


def _per_frame_rows(
    *,
    times_s: Sequence[float],
    comparison,
    union_ious,
    subject_rows,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, time_s in enumerate(times_s):
        audit = comparison[frame_index]
        rows.append(
            {
                "frame": frame_index,
                "time_s": time_s,
                "physical_subject_iou": union_ious[frame_index],
                "matched_count": len(audit["matches"]),
                "missing_count": len(audit["missing_entity_ids"]),
                "residual_count": len(audit["residual_track_ids"]),
                "ambiguous_candidate_count": len(
                    audit["ambiguous_candidate_track_ids"]
                ),
                "formal_prediction_cardinality": audit[
                    "formal_prediction_cardinality"
                ],
                "participant_prediction_count": audit[
                    "participant_prediction_count"
                ],
                "raw_prediction_candidate_count": audit[
                    "raw_prediction_candidate_count"
                ],
                "overflow_count": audit["overflow_count"],
                "missing_entity_ids": json.dumps(
                    audit["missing_entity_ids"],
                    separators=(",", ":"),
                ),
                "residual_track_ids": json.dumps(
                    audit["residual_track_ids"],
                    separators=(",", ":"),
                ),
                "ambiguous_candidate_track_ids": json.dumps(
                    audit["ambiguous_candidate_track_ids"],
                    separators=(",", ":"),
                ),
                "rejected_candidate_matches": json.dumps(
                    audit["rejected_candidate_matches"],
                    separators=(",", ":"),
                ),
                "matches": json.dumps(
                    audit["matches"],
                    separators=(",", ":"),
                ),
                "position_score": audit["position_diagnostic_score"],
                "position_diagnostic_score": audit[
                    "position_diagnostic_score"
                ],
                "matched_position_score": audit[
                    "matched_position_score"
                ],
                "shape_score": subject_rows[frame_index].get("shape"),
                "appearance_score": subject_rows[frame_index].get(
                    "appearance"
                ),
            }
        )
    return rows


__all__ = ["CollisionOpenWorldCaseEvaluator"]
