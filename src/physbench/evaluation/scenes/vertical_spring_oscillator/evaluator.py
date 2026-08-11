"""Fail-closed evaluator for one frozen vertical spring-ball identity."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ....io import write_json
from ...common.artifacts import save_series_comparison, write_rows_csv
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.csti import build_csti_input_from_aligned_masks
from ...common.entities import (
    LifecyclePolicy,
    ReferenceCapability,
    materialize_entity_manifest,
)
from ...common.errors import ReferenceAnalysisError, SceneAnalysisError
from ...common.frozen_subject import load_frozen_subject_anchor
from ...common.masks.sam2 import Sam2VideoSegmenter
from ...common.subject import (
    compare_subjects,
    compose_subject_and_state_score,
    infer_reference_mode,
    subject_weights,
    write_subject_artifacts,
)
from ...contracts import CaseEvaluationRequest
from .observation import (
    SpringTopology,
    observe_spring_topology,
    prompt_from_anchor,
    validate_mask_tube,
    validate_prediction_identity,
)
from .scoring import (
    SpringTrace,
    SpringTraceError,
    extract_spring_trace,
    score_spring_traces,
)


_MASK_QUALITY_KEYS = (
    "minimum_mask_pixels",
    "maximum_mask_area_ratio",
    "minimum_anchor_area_ratio",
    "maximum_anchor_area_ratio",
)
_CONTENT_WEIGHT_KEYS = frozenset({"physics_state", "subject", "topology"})


def _exact_sampled_frames(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(left_frame, right_frame)
        for left_frame, right_frame in zip(left, right)
    )


def _metadata(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {"value": repr(value)}


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value >= 1.0 - 1e-8:
        return 1.0
    return float(min(1.0, max(0.0, value)))


def _normalized_content_weights(config: Mapping[str, Any]) -> dict[str, float]:
    if set(config) != _CONTENT_WEIGHT_KEYS:
        raise ValueError(
            "content_weights must name exactly physics_state, subject, and topology"
        )
    try:
        values = {name: float(config[name]) for name in _CONTENT_WEIGHT_KEYS}
    except (TypeError, ValueError) as exc:
        raise ValueError("content_weights must be numeric") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("content_weights must be finite and non-negative")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("content_weights must contain a positive weight")
    return {name: values[name] / total for name in sorted(values)}


def _compare_topology(
    reference: SpringTopology,
    prediction: SpringTopology,
) -> dict[str, Any]:
    lengths = {
        len(reference.valid),
        len(prediction.valid),
        len(reference.row_coverage),
        len(prediction.row_coverage),
        len(reference.endpoint_support),
        len(prediction.endpoint_support),
    }
    if len(lengths) != 1 or not reference.valid.any():
        raise ValueError("reference spring topology has no comparable samples")
    eligible = np.asarray(reference.valid, dtype=bool)
    prediction_valid = np.asarray(prediction.valid, dtype=bool)
    reference_signal = np.clip(
        0.5
        * (
            np.asarray(reference.row_coverage, dtype=np.float64)
            + np.asarray(reference.endpoint_support, dtype=np.float64)
        ),
        0.0,
        1.0,
    )
    prediction_signal = np.clip(
        0.5
        * (
            np.asarray(prediction.row_coverage, dtype=np.float64)
            + np.asarray(prediction.endpoint_support, dtype=np.float64)
        ),
        0.0,
        1.0,
    )
    prediction_signal = np.where(prediction_valid, prediction_signal, 0.0)
    mean_absolute_error = float(
        np.mean(np.abs(prediction_signal[eligible] - reference_signal[eligible]))
    )
    agreement = _bounded(math.exp(-mean_absolute_error / 0.20))
    reference_support = float(np.mean(reference_signal[eligible]))
    prediction_support = float(np.mean(prediction_signal[eligible]))
    support_ratio = _bounded(
        prediction_support / max(reference_support, 1e-12)
    )
    observed_ratio = float(np.mean(prediction_valid[eligible]))
    score = _bounded(agreement * support_ratio * observed_ratio)
    return {
        "score": score,
        "components": {
            "signal_agreement": agreement,
            "absolute_support_ratio": support_ratio,
            "prediction_observed_ratio": observed_ratio,
        },
        "diagnostics": {
            "mean_absolute_signal_error": mean_absolute_error,
            "reference_mean_support": reference_support,
            "prediction_mean_support": prediction_support,
            "reference_raw_score": float(reference.score),
            "prediction_raw_score": float(prediction.score),
        },
        "role": "integrity_factor_and_weighted_content_component",
    }


def _trace_rows(
    *,
    times_s: Sequence[float],
    reference: SpringTrace,
    prediction: SpringTrace,
    reference_topology: SpringTopology,
    prediction_topology: SpringTopology,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(times_s):
        rows.append(
            {
                "time_s": float(time_s),
                "reference_x_px": float(reference.xy[index, 0]),
                "reference_y_px": float(reference.xy[index, 1]),
                "reference_valid": bool(reference.valid[index]),
                "reference_area_px2": float(reference.area_px2[index]),
                "reference_displacement_px": float(
                    reference.xy[index, 1] - reference.equilibrium_y_px
                ),
                "reference_topology_valid": bool(reference_topology.valid[index]),
                "reference_topology_row_coverage": float(
                    reference_topology.row_coverage[index]
                ),
                "reference_topology_endpoint_support": float(
                    reference_topology.endpoint_support[index]
                ),
                "prediction_x_px": float(prediction.xy[index, 0]),
                "prediction_y_px": float(prediction.xy[index, 1]),
                "prediction_valid": bool(prediction.valid[index]),
                "prediction_area_px2": float(prediction.area_px2[index]),
                "prediction_displacement_px": float(
                    prediction.xy[index, 1] - prediction.equilibrium_y_px
                ),
                "prediction_topology_valid": bool(prediction_topology.valid[index]),
                "prediction_topology_row_coverage": float(
                    prediction_topology.row_coverage[index]
                ),
                "prediction_topology_endpoint_support": float(
                    prediction_topology.endpoint_support[index]
                ),
            }
        )
    return rows


class VerticalSpringOscillatorCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "vertical_spring_oscillator_expert"
    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"
    scene_id = "vertical_spring_oscillator"
    primary_score = "vertical_spring_oscillator_similarity"
    allow_partial_prediction = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._segmenter = Sam2VideoSegmenter(config["sam2"])

    def describe_observation(self) -> dict[str, Any]:
        return {
            "subject": "frozen_dataset_steel_ball_identity",
            "segmentation": "condition_causal_sam2_video_tube",
            "reference_prediction_symmetry": (
                "independent_segmentation_with_exact_sample_reuse"
            ),
            "dynamics": "shared_timeline_vertical_spring_trace_no_warping",
            "topology": "current_ball_corridor_connected_edge_support",
            "identity_failure_policy": "robust_subject_v3_evaluated_zero",
            "future_gt_usage": False,
        }

    def _validate_manifest(self, request: CaseEvaluationRequest):
        try:
            manifest = materialize_entity_manifest(request.case)
            if manifest.scene_id != self.scene_id:
                raise ValueError("spring manifest scene differs from evaluator")
            if len(manifest.entities) != 1 or len(manifest.apparatus) != 1:
                raise ValueError(
                    "vertical spring requires one dynamic entity and one apparatus"
                )
            entity = manifest.entities[0]
            if (
                entity.entity_id != "oscillator_ball"
                or entity.role_id != "spring_oscillator"
                or entity.entity_class != "steel_ball"
                or entity.parts != ("spring",)
                or entity.lifecycle is not LifecyclePolicy.PERSISTENT
            ):
                raise ValueError("vertical spring entity contract differs from v1")
            apparatus = manifest.apparatus[0]
            if (
                apparatus.apparatus_id != "spring_support"
                or apparatus.apparatus_class != "vertical_spring_and_support"
            ):
                raise ValueError("vertical spring apparatus contract differs from v1")
            if manifest.reference_capability is not ReferenceCapability.SAME_CASE_GT:
                raise ValueError("vertical spring v1 requires SAME_CASE_GT")
            entity_attributes = {
                value.name: float(value.value)
                for value in entity.physical_attributes
            }
            apparatus_attributes = {
                value.name: float(value.value)
                for value in apparatus.physical_attributes
            }
            if set(entity_attributes) != {"initial_displacement", "mass", "radius"}:
                raise ValueError("vertical spring entity attributes differ from v1")
            if set(apparatus_attributes) != {
                "natural_spring_length",
                "spring_stiffness",
                "gravity_acceleration",
            }:
                raise ValueError("vertical spring apparatus attributes differ from v1")
            return manifest, entity, entity_attributes, apparatus_attributes
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_entity_manifest_invalid",
                "vertical spring entity manifest is invalid: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def _mask_quality_config(self) -> dict[str, Any]:
        quality = self.config["quality"]
        return {name: quality[name] for name in _MASK_QUALITY_KEYS}

    def _trace_quality_config(self) -> dict[str, Any]:
        return {**self.config["quality"], **self.config["period"]}

    def _segment_reference(
        self,
        frames: list[np.ndarray],
        *,
        prompt,
    ) -> tuple[Sequence[np.ndarray], dict[str, Any]]:
        try:
            masks, metadata = self._segmenter.segment(
                frames,
                prompt=prompt,
                temporary_prefix="physbench_vertical_spring_reference_",
            )
            return masks, _metadata(metadata)
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_spring_segmentation_failed",
                "reference vertical spring segmentation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def _segment_prediction(
        self,
        frames: list[np.ndarray],
        *,
        prompt,
    ) -> tuple[Sequence[np.ndarray], dict[str, Any]]:
        try:
            masks, metadata = self._segmenter.segment(
                frames,
                prompt=prompt,
                temporary_prefix="physbench_vertical_spring_prediction_",
            )
            return masks, _metadata(metadata)
        except Exception as exc:
            raise SceneAnalysisError(
                "prediction_spring_segmentation_failed",
                "prediction vertical spring segmentation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def _write_artifacts(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_trace: SpringTrace,
        prediction_trace: SpringTrace,
        reference_topology: SpringTopology,
        prediction_topology: SpringTopology,
        subject,
        audit: dict[str, Any],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        artifacts: dict[str, str] = {}
        failures: list[dict[str, str]] = []
        try:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            csv_path = request.artifact_dir / "vertical_spring_per_frame.csv"
            trajectory_path = request.artifact_dir / "vertical_spring_trajectory.png"
            audit_path = request.artifact_dir / "vertical_spring_audit.json"
            write_rows_csv(
                csv_path,
                _trace_rows(
                    times_s=times_s,
                    reference=reference_trace,
                    prediction=prediction_trace,
                    reference_topology=reference_topology,
                    prediction_topology=prediction_topology,
                ),
            )
            save_series_comparison(
                trajectory_path,
                times_s=times_s,
                reference=np.asarray(reference_trace.xy[:, 1]),
                prediction=np.asarray(prediction_trace.xy[:, 1]),
                ylabel="Vertical centroid y (px)",
                title=(
                    "Vertical spring trajectory — "
                    f"{request.case['case_id']}"
                ),
            )
            subject_artifacts = write_subject_artifacts(
                directory=request.artifact_dir,
                times_s=times_s,
                comparison=subject,
                case_id=str(request.case["case_id"]),
                scene_name="Vertical spring oscillator",
            )
            write_json(audit_path, audit)
            artifacts = {
                "per_frame_csv": str(csv_path),
                "trajectory_curve": str(trajectory_path),
                **subject_artifacts,
                "audit_json": str(audit_path),
            }
        except Exception as exc:
            failures.append(
                {
                    "code": "vertical_spring_artifact_write_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            artifacts = {}
        return artifacts, failures

    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video,
        prediction_video,
    ) -> SceneAnalysis:
        (
            manifest,
            entity,
            entity_attributes,
            apparatus_attributes,
        ) = self._validate_manifest(request)
        if (
            not times_s
            or len(reference_video.frames) != len(times_s)
            or len(prediction_video.frames) != len(times_s)
        ):
            raise ReferenceAnalysisError(
                "reference_spring_timeline_invalid",
                "vertical spring analysis requires equal non-empty sampled timelines",
            )
        if (
            reference_video.frames[0].shape
            != prediction_video.frames[0].shape
        ):
            raise SceneAnalysisError(
                "prediction_spring_canvas_mismatch",
                "prediction and reference spring canvases differ",
            )

        try:
            anchor = load_frozen_subject_anchor(
                request,
                logical_entity_id=entity.entity_id,
                entity_class=entity.entity_class,
                spatial_transform=reference_video.spatial_transform,
                error_namespace="reference_spring_subject",
            )
            prompt = prompt_from_anchor(anchor)
        except ReferenceAnalysisError:
            raise
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_spring_anchor_invalid",
                "frozen vertical spring subject anchor is invalid: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        raw_reference_masks, reference_segmentation = self._segment_reference(
            reference_video.frames,
            prompt=prompt,
        )
        exact_reuse = _exact_sampled_frames(
            reference_video.frames,
            prediction_video.frames,
        )
        if exact_reuse:
            raw_prediction_masks = raw_reference_masks
            prediction_segmentation = {
                **reference_segmentation,
                "reuse": "exact_sampled_reference_frames",
            }
        else:
            raw_prediction_masks, prediction_segmentation = (
                self._segment_prediction(
                    prediction_video.frames,
                    prompt=prompt,
                )
            )

        mask_quality = self._mask_quality_config()
        try:
            reference_masks = validate_mask_tube(
                raw_reference_masks,
                availability=[True] * len(times_s),
                anchor=anchor,
                config=mask_quality,
            )
            reference_identity = validate_prediction_identity(
                anchor_mask=anchor.mask,
                prediction_mask=reference_masks[0],
                config=self.config["identity"],
            )
            if not bool(reference_identity["accepted"]):
                raise ValueError("reference frame-zero mask failed frozen identity")
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_spring_identity_rejected",
                "reference vertical spring identity validation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        prediction_available = prediction_video.available or [True] * len(times_s)
        try:
            prediction_masks = validate_mask_tube(
                raw_prediction_masks,
                availability=prediction_available,
                anchor=anchor,
                config=mask_quality,
            )
            prediction_identity = validate_prediction_identity(
                anchor_mask=anchor.mask,
                prediction_mask=prediction_masks[0],
                config=self.config["identity"],
            )
            if not bool(prediction_identity["accepted"]):
                raise SceneAnalysisError(
                    "prediction_spring_identity_rejected",
                    "prediction frame-zero mask failed the frozen steel-ball identity gate",
                )
        except SceneAnalysisError:
            raise
        except Exception as exc:
            raise SceneAnalysisError(
                "prediction_spring_identity_rejected",
                "prediction vertical spring identity validation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        trace_quality = self._trace_quality_config()
        try:
            reference_trace = extract_spring_trace(
                reference_masks,
                times_s,
                quality_config=trace_quality,
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, SpringTraceError) else type(exc).__name__
            raise ReferenceAnalysisError(
                "reference_spring_trace_invalid",
                f"reference vertical spring trace failed ({code}): {exc}",
            ) from exc
        try:
            prediction_trace = extract_spring_trace(
                prediction_masks,
                times_s,
                quality_config=trace_quality,
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, SpringTraceError) else type(exc).__name__
            raise SceneAnalysisError(
                "prediction_spring_trace_invalid",
                f"prediction vertical spring trace failed ({code}): {exc}",
            ) from exc

        try:
            reference_topology = observe_spring_topology(
                reference_video.frames,
                reference_masks,
                availability=[True] * len(times_s),
                config=self.config["topology"],
            )
            if not reference_topology.valid.any():
                raise ValueError("reference topology contains no valid corridor")
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_spring_topology_invalid",
                "reference vertical spring topology failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        try:
            prediction_topology = (
                reference_topology
                if exact_reuse and all(prediction_available)
                else observe_spring_topology(
                    prediction_video.frames,
                    prediction_masks,
                    availability=prediction_available,
                    config=self.config["topology"],
                )
            )
            topology_metric = _compare_topology(
                reference_topology,
                prediction_topology,
            )
        except Exception as exc:
            raise SceneAnalysisError(
                "prediction_spring_topology_invalid",
                "prediction vertical spring topology failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            physics_metric = score_spring_traces(
                reference_trace,
                prediction_trace,
                mass_kg=entity_attributes["mass"],
                stiffness_n_m=apparatus_attributes["spring_stiffness"],
                scoring_config=self.config["scoring"],
            )
            reference_mode = infer_reference_mode(request.case)
            subject = compare_subjects(
                reference_frames=reference_video.frames,
                prediction_frames=prediction_video.frames,
                reference_masks=list(reference_masks),
                prediction_masks=list(prediction_masks),
                reference_mode=reference_mode,
                config=self.config["subject_scoring"],
            )
            content_weights = _normalized_content_weights(
                self.config["content_weights"]
            )
            subject_state = compose_subject_and_state_score(
                state_score=float(physics_metric["score"]),
                subject=subject,
                reference_mode=reference_mode,
                config={
                    **self.config["subject_scoring"],
                    "case_weights": {
                        "physics_state": content_weights["physics_state"],
                        "subject": content_weights["subject"],
                    },
                },
            )
        except Exception as exc:
            raise SceneAnalysisError(
                "prediction_spring_scoring_failed",
                "vertical spring prediction scoring failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        subject_state_weight = (
            content_weights["physics_state"] + content_weights["subject"]
        )
        content_score = _bounded(
            subject_state_weight * float(subject_state["score"])
            + content_weights["topology"] * float(topology_metric["score"])
        )
        integrity_factor = float(topology_metric["score"])
        score = _bounded(content_score * integrity_factor)
        primary_metric = {
            "score": score,
            "raw_content_score": content_score,
            "integrity_factor": integrity_factor,
            "components": {
                "physics_state": float(physics_metric["score"]),
                "subject": float(subject.score),
                "topology": float(topology_metric["score"]),
            },
            "content_weights": content_weights,
            "subject_state_composition": subject_state,
            "composition": "topology_integrity_times_weighted_content_v1",
        }

        audit = {
            "case_id": str(request.case["case_id"]),
            "entity_manifest_digest": manifest.digest,
            "entity_manifest_materializer_id": manifest.materializer_id,
            "frozen_anchor": dict(anchor.provenance),
            "segmentation": {
                "reference": reference_segmentation,
                "prediction": prediction_segmentation,
            },
            "identity": {
                "reference": dict(reference_identity),
                "prediction": dict(prediction_identity),
            },
            "physics": physics_metric,
            "topology": topology_metric,
            "primary": primary_metric,
            "failures": [],
        }
        artifacts, artifact_failures = self._write_artifacts(
            request,
            times_s=times_s,
            reference_trace=reference_trace,
            prediction_trace=prediction_trace,
            reference_topology=reference_topology,
            prediction_topology=prediction_topology,
            subject=subject,
            audit=audit,
        )
        csti_input = (
            build_csti_input_from_aligned_masks(
                reference_capability=manifest.reference_capability,
                times_s=times_s,
                frame_shape=reference_video.frames[0].shape[:2],
                expected_entities=((entity.entity_id, entity.role_id),),
                reference_masks_by_entity={
                    entity.entity_id: tuple(reference_masks)
                },
                prediction_masks_by_entity={
                    entity.entity_id: tuple(prediction_masks)
                },
                matched_track_ids_by_entity={
                    entity.entity_id: ("bound_spring_ball",)
                },
            )
            if (
                self.csti_enabled
                and manifest.reference_capability
                is ReferenceCapability.SAME_CASE_GT
            )
            else None
        )
        return SceneAnalysis(
            score=score,
            metrics={
                "scene_subject_state_similarity": dict(primary_metric),
                "vertical_spring_oscillator_similarity": primary_metric,
                "vertical_spring_dynamics_similarity": physics_metric,
                "vertical_spring_topology_integrity": topology_metric,
                "physical_subject_similarity": subject.to_metric(
                    weights=subject_weights(self.config["subject_scoring"])
                ),
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": False,
                "degradation_codes": [],
                "degradation_reason": None,
                "artifact_failures": artifact_failures,
                "reference_valid_mask_ratio": float(reference_trace.valid_ratio),
                "prediction_valid_mask_ratio": float(prediction_trace.valid_ratio),
                "reference_topology_valid_ratio": float(
                    np.mean(reference_topology.valid)
                ),
                "prediction_topology_valid_ratio": float(
                    np.mean(prediction_topology.valid)
                ),
                "prediction_identity_accepted": True,
                "expected_entity_count": 1,
            },
            artifacts=artifacts,
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                    "reference_capability": manifest.reference_capability.value,
                },
                "frozen_anchor": dict(anchor.provenance),
                "segmentation": {
                    "adapter": self._segmenter.describe(),
                    "reference": reference_segmentation,
                    "prediction": prediction_segmentation,
                    "exact_same_sampled_frames_reused": exact_reuse,
                },
                "identity": {
                    "reference": dict(reference_identity),
                    "prediction": dict(prediction_identity),
                    "matched_prediction_track_id": "bound_spring_ball",
                },
                "time_alignment": "common_physical_timeline_no_dtw",
                "future_reference_pixels_used_for_prediction_localization": False,
            },
            csti_input=csti_input,
        )


__all__ = ["VerticalSpringOscillatorCaseEvaluator"]
