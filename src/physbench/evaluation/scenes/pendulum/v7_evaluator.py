from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ....io import canonical_sha256, sha256_file
from ...common.artifacts.open_world_v2 import write_open_world_v2_artifacts
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.csti import (
    CSTIInput,
    build_csti_input_from_aligned_masks,
    build_csti_input_from_frame_matches,
)
from ...common.entities import (
    ExpectedPositionSample,
    ReferenceCapability,
    build_common_time_grid,
    compare_open_world_v2,
    compose_object_centric_v2,
    fail_closed_open_world_v2,
    materialize_entity_manifest,
    resolve_expected_entity_timeline,
)
from ...common.entities.observer import OpenWorldObservation, OpenWorldTrack
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import observed_mask_iou, summarize_mask_ious
from ...common.subject import compare_subjects, infer_reference_mode
from ...contracts import CaseEvaluationRequest
from ....datasets.physics import flat_physics_quantities
from .open_world import (
    PENDULUM_STRUCTURE_SPEC_VERSION,
    extract_bob_masks,
    prompt_from_structure,
)
from .scoring import TraceQualityError, extract_trace, score_traces
from .segmentation import Sam2PendulumSegmenter, SegmentationError
from .v6_evaluator import (
    _centroids,
    _content_contract,
    _freeze_bob_identity,
    _full_prediction_union,
    _load_condition_frame,
    _normalized_prediction_timeline,
    _pendulum_radius_length_ratio,
    _time_coverage,
    _write_artifacts,
)
from .v7_open_world import (
    PENDULUM_V7_OBSERVER_VERSION,
    PENDULUM_V7_TOPOLOGY_VERSION,
    compare_pendulum_topology_v7,
    detect_condition_structure_v7,
    discover_pendulum_objects_v7,
    extract_bob_trace_v7,
    observe_pendulum_topology_v7,
    track_masks,
)


class PendulumOpenWorldCaseEvaluatorV7(ReferenceCaseEvaluator):
    """Condition-causal, symmetric open-world pendulum evaluator."""

    evaluator_id = "pendulum_open_world_structure"
    evaluator_version = "2.1"
    sequential_evaluator_version = "2.1"
    robust_evaluator_version = "2.1"
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
                "observer": PENDULUM_V7_OBSERVER_VERSION,
                "structure_spec": PENDULUM_STRUCTURE_SPEC_VERSION,
                "topology": PENDULUM_V7_TOPOLOGY_VERSION,
                "composition": "integrity_gate_times_scene_content_v1",
                "physics_parent_future_pixel_supervision": False,
            }
        )

    def describe_observation(self) -> dict[str, Any]:
        assignment = self.config["object_centric_scoring"]["assignment"]
        return {
            "subject": "condition_only_frozen_bob_plus_symmetric_topology",
            "condition_anchor": (
                "hough_circle_plus_string_plus_physics_ratio_and_angle_plus_"
                "raw_edge_and_body_evidence"
            ),
            "reference_prediction_symmetry": (
                "same_case_reference_and_prediction_share_the_exact_"
                "condition_frozen_prompt_observer_and_topology_path"
            ),
            "proposal_fusion": (
                "directed_and_residual_sources_are_fused_before_tracking"
            ),
            "residual_policy": (
                "independent_evidence_then_multi_frame_persistence"
            ),
            "assignment": {
                "method": "open_world_v2_hungarian_null_plus_frozen_identity",
                "minimum_match_position_similarity": float(
                    assignment["minimum_match_position_similarity"]
                ),
            },
            "topology": PENDULUM_V7_TOPOLOGY_VERSION,
            "physics_parent_policy": (
                "child_condition_identity_plus_parent_normalized_dynamics_"
                "without_parent_future_pixel_identity_or_localization"
            ),
        }

    @staticmethod
    def _build_csti_input(
        *,
        capability: ReferenceCapability,
        times_s: list[float],
        frame_shape: tuple[int, int],
        entity: Any,
        reference_bobs,
        prediction_observation: OpenWorldObservation | None,
        comparison: Any,
    ) -> CSTIInput:
        expected_entities = ((entity.entity_id, entity.role_id),)
        references = {entity.entity_id: tuple(reference_bobs)}
        if prediction_observation is None:
            return build_csti_input_from_aligned_masks(
                reference_capability=capability,
                times_s=times_s,
                frame_shape=frame_shape,
                expected_entities=expected_entities,
                reference_masks_by_entity=references,
                prediction_masks_by_entity={entity.entity_id: None},
                matched_track_ids_by_entity={entity.entity_id: ()},
            )
        return build_csti_input_from_frame_matches(
            reference_capability=capability,
            times_s=times_s,
            frame_shape=frame_shape,
            expected_entities=expected_entities,
            reference_masks_by_entity=references,
            prediction_observation=prediction_observation,
            matches=comparison.matches,
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
                "pendulum entity manifest is invalid: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        time_grid = build_common_time_grid(times_s)
        proposal_config = self.config["motion_proposal"]
        observation_config = self.config["open_world_observation"]
        condition_frame, condition_path, condition_transform = (
            _load_condition_frame(
                request,
                config=self.config,
                normalized_reference_frame=reference_video.frames[0],
                normalized_reference_transform=(
                    reference_video.spatial_transform
                ),
            )
        )
        try:
            initial_angle_deg = float(
                flat_physics_quantities(request.case)["initial_angle"]["value"]
            )
            condition_decision = detect_condition_structure_v7(
                condition_frame,
                config=observation_config,
                expected_radius_length_ratio=(
                    _pendulum_radius_length_ratio(request.case)
                ),
                expected_initial_angle_deg=initial_angle_deg,
            )
            condition_structure = condition_decision.structure
            condition_prompt = prompt_from_structure(condition_structure)
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_condition_pendulum_structure_failed_v7",
                "condition-only pendulum structure failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        reference_mode = infer_reference_mode(request.case)
        capability = manifest.reference_capability
        reference_segmentation: Mapping[str, Any]
        reference_observation: OpenWorldObservation | None = None
        reference_frozen_identity = None
        reference_track = None
        reference_topology: Mapping[str, object] | None = None
        try:
            if reference_mode == "same_case_reference":
                (
                    reference_assembly,
                    reference_segmentation,
                    _,
                ) = self._segmenter.segment(
                    reference_video.frames,
                    proposal_config=proposal_config,
                    prompt=condition_prompt,
                    prompt_source="case_condition_structure_v7_symmetric",
                )
                (
                    directed_reference_bobs,
                    _,
                    _,
                ) = extract_bob_masks(
                    reference_assembly,
                    expected_radius_px=condition_structure.bob_radius_px,
                    minimum_pixels=int(
                        observation_config.get("minimum_bob_pixels", 8)
                    ),
                )
                reference_observation = discover_pendulum_objects_v7(
                    reference_video.frames,
                    directed_bob_masks=directed_reference_bobs,
                    directed_subject_masks=reference_assembly,
                    condition_frame=condition_frame,
                    structure=condition_structure,
                    time_grid=time_grid,
                    config=observation_config,
                )
                reference_frozen_identity = _freeze_bob_identity(
                    entity=entity,
                    observation=reference_observation,
                    structure=condition_structure,
                    frame=condition_frame,
                    maximum_match_cost=float(
                        self.config["object_centric_scoring"][
                            "assignment"
                        ].get("maximum_condition_identity_cost", 3.0)
                    ),
                )
                reference_open_track = _assigned_track(
                    reference_observation,
                    reference_frozen_identity.entity_to_track.get(
                        entity.entity_id
                    ),
                )
                if reference_open_track is None:
                    raise ReferenceAnalysisError(
                        "reference_condition_identity_unmatched_v7",
                        "condition-only reference observer could not freeze "
                        "the declared bob identity",
                    )
                (
                    reference_bobs,
                    reference_bob_xy,
                    reference_bob_observed,
                ) = track_masks(
                    reference_open_track,
                    frame_count=len(times_s),
                    shape=reference_video.frames[0].shape[:2],
                )
                reference_track = reference_open_track.to_object_track(
                    frame_count=len(times_s),
                    time_weights_s=time_grid.cell_weights_s,
                    matched_entity_id=entity.entity_id,
                )
                reference_trace = extract_bob_trace_v7(
                    reference_bob_xy,
                    reference_bob_observed,
                    times_s,
                    pivot_xy=condition_structure.pivot_xy,
                    period_config=self.config["period"],
                )
                reference_topology = observe_pendulum_topology_v7(
                    reference_video.frames,
                    reference_assembly,
                    pivot_xy=condition_structure.pivot_xy,
                    bob_xy=reference_bob_xy,
                    bob_observed=reference_bob_observed,
                    bob_masks=reference_bobs,
                    frame_weights=time_grid.cell_weights_s,
                    config=self.config["topology"],
                )
            else:
                # A physics parent may provide normalized dynamics, but never
                # the child identity, lifecycle, future position, or ROI.
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
                    _,
                ) = extract_bob_masks(
                    reference_assembly,
                    minimum_pixels=int(
                        observation_config.get("minimum_bob_pixels", 8)
                    ),
                )
                reference_bob_observed = np.isfinite(
                    reference_bob_xy
                ).all(axis=1)
        except ReferenceAnalysisError:
            raise
        except (SegmentationError, TraceQualityError, Exception) as exc:
            raise ReferenceAnalysisError(
                "reference_pendulum_open_world_observation_failed_v7",
                "reference pendulum v7 observation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

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
                "observer": PENDULUM_V7_OBSERVER_VERSION,
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
        if condition_decision.confidence_margin < float(
            observation_config.get(
                "v7_condition_low_margin_warning", 0.03
            )
        ):
            prediction_failures.append(
                {
                    "code": "condition_structure_low_margin_v7",
                    "reason": (
                        "condition-only structure winner has confidence "
                        f"margin {condition_decision.confidence_margin:.4f}"
                    ),
                }
            )

        empty_mask = np.zeros(
            reference_video.frames[0].shape[:2], dtype=np.uint8
        )
        prediction_assembly = [
            np.array(empty_mask, copy=True) for _ in times_s
        ]
        prediction_segmentation: Mapping[str, Any] = {"status": "not_run"}
        prediction_observation: OpenWorldObservation | None = None
        prediction_frozen_identity = None
        comparison = None
        exact_same_sampled_frames = bool(
            reference_mode == "same_case_reference"
            and np.all(prediction_available)
            and len(prediction_frames) == len(reference_video.frames)
            and all(
                np.array_equal(prediction_frame, reference_frame)
                for prediction_frame, reference_frame in zip(
                    prediction_frames,
                    reference_video.frames,
                    strict=True,
                )
            )
        )
        try:
            if exact_same_sampled_frames:
                prediction_assembly = [
                    np.array(value, copy=True)
                    for value in reference_assembly
                ]
                prediction_segmentation = {
                    **dict(reference_segmentation),
                    "status": "reused_exact_same_sampled_frames",
                    "reuse_is_pixel_exact": True,
                }
                prediction_observation = reference_observation
                prediction_frozen_identity = reference_frozen_identity
            else:
                (
                    prediction_assembly,
                    prediction_segmentation,
                    _,
                ) = self._segmenter.segment(
                    prediction_frames,
                    proposal_config=proposal_config,
                    prompt=condition_prompt,
                    prompt_source="case_condition_structure_v7_symmetric",
                )
                for index, is_available in enumerate(prediction_available):
                    if not is_available:
                        prediction_assembly[index] = np.zeros_like(
                            prediction_assembly[index]
                        )
                directed_prediction_bobs, _, _ = extract_bob_masks(
                    prediction_assembly,
                    expected_radius_px=condition_structure.bob_radius_px,
                    minimum_pixels=int(
                        observation_config.get("minimum_bob_pixels", 8)
                    ),
                )
                prediction_observation = discover_pendulum_objects_v7(
                    prediction_frames,
                    directed_bob_masks=directed_prediction_bobs,
                    directed_subject_masks=prediction_assembly,
                    condition_frame=condition_frame,
                    structure=condition_structure,
                    time_grid=time_grid,
                    config=observation_config,
                    available=prediction_available,
                )
                prediction_frozen_identity = _freeze_bob_identity(
                    entity=entity,
                    observation=prediction_observation,
                    structure=condition_structure,
                    frame=condition_frame,
                    maximum_match_cost=float(
                        self.config["object_centric_scoring"]["assignment"].get(
                            "maximum_condition_identity_cost", 3.0
                        )
                    )
                )
            comparison = compare_open_world_v2(
                expected_timelines=[timeline],
                prediction_observation=prediction_observation,
                time_grid=time_grid,
                frame_diagonal_px=float(
                    math.hypot(
                        reference_video.frames[0].shape[1],
                        reference_video.frames[0].shape[0],
                    )
                ),
                frozen_identity=prediction_frozen_identity,
                minimum_match_position_similarity=float(
                    self.config["object_centric_scoring"]["assignment"][
                        "minimum_match_position_similarity"
                    ]
                ),
            )
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_open_world_observation_failed_v7",
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
            prediction_trace = extract_bob_trace_v7(
                matched_xy,
                matched_observed,
                times_s,
                pivot_xy=condition_structure.pivot_xy,
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
                "trace_source": "matched_condition_frozen_bob_v7",
            }
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_physics_scoring_failed_v7",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            state_score = {
                "score": 0.0,
                "raw_interpolated_score": 0.0,
                "physical_evidence_coverage": observed_coverage,
                "failure": failure,
                "missing_frames_are_not_free": True,
                "trace_source": "matched_condition_frozen_bob_v7",
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
                "code": "prediction_pendulum_subject_scoring_failed_v7",
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

        try:
            if exact_same_sampled_frames and reference_topology is not None:
                prediction_topology_observation = reference_topology
            else:
                prediction_topology_observation = observe_pendulum_topology_v7(
                    prediction_frames,
                    prediction_assembly,
                    pivot_xy=condition_structure.pivot_xy,
                    bob_xy=matched_xy,
                    bob_observed=matched_observed,
                    bob_masks=matched_bobs,
                    frame_weights=time_grid.cell_weights_s,
                    config=self.config["topology"],
                )
            topology_metric = compare_pendulum_topology_v7(
                reference_topology,
                prediction_topology_observation,
                frame_weights=time_grid.cell_weights_s,
            )
            topology_rows = prediction_topology_observation["per_frame"]
            if bool(topology_metric["uncertain"]):
                prediction_failures.append(
                    {
                        "code": "prediction_topology_source_disagreement_v7",
                        "reason": (
                            "mask and raw-image string evidence disagree "
                            f"at ratio {float(topology_metric['source_disagreement_ratio']):.3f}"
                        ),
                    }
                )
        except Exception as exc:
            failure = {
                "code": "prediction_pendulum_topology_failed_v7",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            prediction_failures.append(failure)
            topology_metric = {
                "version": PENDULUM_V7_TOPOLOGY_VERSION,
                "score": 0.0,
                "failure": failure,
            }
            topology_rows = _empty_topology_rows(len(times_s))

        if observed_coverage < float(
            observation_config.get("v7_low_matched_coverage_warning", 0.80)
        ):
            prediction_failures.append(
                {
                    "code": "prediction_low_matched_bob_coverage_v7",
                    "reason": (
                        f"matched bob coverage is {observed_coverage:.3f}"
                    ),
                }
            )
        if (
            prediction_observation is not None
            and bool(
                prediction_observation.diagnostics.get(
                    "track_saturated", False
                )
            )
        ):
            prediction_failures.append(
                {
                    "code": "prediction_pendulum_track_saturation_v7",
                    "reason": "open-world pendulum track cap was reached",
                }
            )
        if prediction_observation is not None:
            residual_formal = [
                track
                for track in prediction_observation.tracks
                if track.formal_exposure_weight > 0.0
                and not any(
                    "condition_directed_sam2" in detection.sources
                    for detection in track.detections
                )
            ]
            if len(residual_formal) > int(
                observation_config.get(
                    "v7_residual_fragmentation_warning_tracks", 3
                )
            ):
                prediction_failures.append(
                    {
                        "code": "prediction_residual_fragmentation_v7",
                        "reason": (
                            f"{len(residual_formal)} formal residual tracks "
                            "survived proposal fusion"
                        ),
                    }
                )

        content, weights = _content_contract(
            state_score=float(state_score["score"]),
            subject_components=subject.components,
            subject_score=subject.score,
            topology_score=float(topology_metric["score"]),
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
                    "code": "prediction_non_finite_case_score_v7",
                    "reason": "v7 composition produced a non-finite score",
                }
            )

        prediction_union = _full_prediction_union(
            prediction_assembly,
            observation=prediction_observation,
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
            artifact_reference_assembly = [
                np.array(condition_structure.subject_mask, copy=True),
                *[
                    np.zeros_like(condition_structure.subject_mask)
                    for _ in range(len(times_s) - 1)
                ],
            ]
            visualization_reference_frames = [
                np.array(condition_frame, copy=True) for _ in times_s
            ]
            visualization_scene_name = (
                "Open-world pendulum v7 "
                "(condition anchor; no same-case GT)"
            )
        else:
            symmetric_reference_union = _full_prediction_union(
                reference_assembly,
                observation=reference_observation,
            )
            full_reference_union = list(symmetric_reference_union)
            full_ious = [
                observed_mask_iou(
                    symmetric_reference_union[index],
                    prediction_union[index],
                )
                for index in range(len(times_s))
            ]
            artifact_reference_assembly = symmetric_reference_union
            visualization_reference_frames = reference_video.frames
            visualization_scene_name = "Open-world pendulum v7"
        artifacts = _write_artifacts(
            request,
            times_s=times_s,
            reference_assembly=artifact_reference_assembly,
            prediction_union=prediction_union,
            full_ious=full_ious,
            comparison=comparison,
            observation=prediction_observation,
            subject_rows=subject.per_frame,
            topology_rows=topology_rows,
        )
        artifacts.update(
            write_open_world_v2_artifacts(
                request,
                scene_name=visualization_scene_name,
                times_s=times_s,
                reference_frames=visualization_reference_frames,
                prediction_frames=prediction_frames,
                expected_timelines=[timeline],
                prediction_observation=prediction_observation,
                comparison=comparison,
                config=self.config.get("visualization"),
                reference_union_masks=full_reference_union,
                prediction_union_masks=prediction_union,
                full_subject_ious=full_ious,
                prediction_available=prediction_available,
                reference_role=(
                    "CONDITION ANCHOR"
                    if reference_mode == "parent_physics_reference"
                    else "REFERENCE"
                ),
                score_summary={
                    "score": score,
                    "components": {
                        "physics": float(state_score["score"]),
                        "subject": float(subject.score),
                        "topology": float(topology_metric["score"]),
                        "integrity": float(comparison.integrity.score),
                    },
                },
                per_frame_diagnostics=[
                    {
                        "position": row.get("position_score"),
                        "string intact": topology.get("string_intact_score"),
                        "branch detected": topology.get(
                            "branch_string_detected"
                        ),
                    }
                    for row, topology in zip(
                        comparison.per_frame,
                        topology_rows,
                    )
                ],
                has_issues=bool(prediction_failures or comparison.failed),
            )
        )
        primary_metric = {
            **composition.to_dict(),
            "score": score,
            "content_components": content,
            "content_weights": weights,
        }
        csti_input = (
            self._build_csti_input(
                capability=capability,
                times_s=times_s,
                frame_shape=reference_video.frames[0].shape[:2],
                entity=entity,
                reference_bobs=reference_bobs,
                prediction_observation=prediction_observation,
                comparison=comparison,
            )
            if (
                self.csti_enabled
                and capability is ReferenceCapability.SAME_CASE_GT
            )
            else None
        )
        return SceneAnalysis(
            score=score,
            metrics={
                "scene_subject_state_similarity": dict(primary_metric),
                "pendulum_open_world_similarity": primary_metric,
                "object_centric_integrity": comparison.integrity.to_dict(),
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
                        (
                            "condition_anchor_subject_overlap_only_"
                            "no_same_case_gt_curve"
                        )
                        if reference_mode == "parent_physics_reference"
                        else (
                            "required_full_physical_subject_overlap_curve"
                        )
                    ),
                    "temporal_scope": (
                        "condition_frame_zero_only"
                        if reference_mode == "parent_physics_reference"
                        else "full_common_timeline"
                    ),
                    "future_frames_supervised": (
                        reference_mode != "parent_physics_reference"
                    ),
                },
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": bool(prediction_failures),
                "degradation_codes": _unique_codes(prediction_failures),
                "degradation_reason": (
                    "; ".join(
                        value["reason"] for value in prediction_failures
                    )
                    if prediction_failures
                    else None
                ),
                "reference_valid_mask_ratio": reference_trace.valid_ratio,
                "prediction_observed_bob_ratio": float(
                    np.mean(matched_observed)
                ),
                "physical_evidence_coverage": observed_coverage,
                "expected_entity_count": 1,
                "prediction_track_count": (
                    len(prediction_observation.tracks)
                    if prediction_observation is not None
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
                "condition_structure": condition_decision.to_dict(),
                "condition_mask_source": (
                    "current_case_condition_only_not_prediction_roi"
                ),
                "reference_prediction_observer_symmetry": (
                    reference_mode == "same_case_reference"
                ),
                "exact_same_sampled_frames_reused": (
                    exact_same_sampled_frames
                ),
                "physics_parent_future_pixel_supervision": False,
                "visualization_reference_role": (
                    (
                        "repeated_child_condition_frame_non_scoring_"
                        "parent_pixels_not_rendered"
                    )
                    if reference_mode == "parent_physics_reference"
                    else "same_case_ground_truth"
                ),
                "reference_identity_assignment": (
                    None
                    if reference_frozen_identity is None
                    else {
                        "entity_to_track": dict(
                            reference_frozen_identity.entity_to_track
                        ),
                        "residual_track_ids": list(
                            reference_frozen_identity.residual_track_ids
                        ),
                        "unmatched_entity_ids": list(
                            reference_frozen_identity.unmatched_entity_ids
                        ),
                    }
                ),
                "prediction_identity_assignment": (
                    None
                    if prediction_frozen_identity is None
                    else {
                        "entity_to_track": dict(
                            prediction_frozen_identity.entity_to_track
                        ),
                        "residual_track_ids": list(
                            prediction_frozen_identity.residual_track_ids
                        ),
                        "unmatched_entity_ids": list(
                            prediction_frozen_identity.unmatched_entity_ids
                        ),
                    }
                ),
                "segmentation": {
                    "reference": dict(reference_segmentation),
                    "prediction": dict(prediction_segmentation),
                },
                "reference_observer": (
                    None
                    if reference_observation is None
                    else dict(reference_observation.diagnostics)
                ),
                "prediction_observer": (
                    None
                    if prediction_observation is None
                    else dict(prediction_observation.diagnostics)
                ),
                "open_world": comparison.to_dict(),
            },
            csti_input=csti_input,
        )


def _assigned_track(
    observation: OpenWorldObservation,
    track_id: str | None,
) -> OpenWorldTrack | None:
    if track_id is None:
        return None
    return next(
        (
            track
            for track in observation.tracks
            if track.track_id == track_id
        ),
        None,
    )


def _empty_topology_rows(frame_count: int) -> list[dict[str, object]]:
    return [
        {
            "frame": index,
            "string_intact_score": 0.0,
            "broken_string": True,
            "branch_string_evidence": 0.0,
            "branch_string_detected": False,
            "branch_string_component_count": 0,
            "branch_string_candidate_pixel_count": 0,
            "branch_string_penalty": 0.0,
            "topology_frame_score": 0.0,
            "branch_components": [],
            "branch_rejected_components": [],
        }
        for index in range(frame_count)
    ]


def _unique_codes(
    failures: list[Mapping[str, str]],
) -> list[str]:
    return list(dict.fromkeys(value["code"] for value in failures))


PendulumOpenWorldV7CaseEvaluator = PendulumOpenWorldCaseEvaluatorV7


__all__ = [
    "PendulumOpenWorldCaseEvaluatorV7",
    "PendulumOpenWorldV7CaseEvaluator",
]
