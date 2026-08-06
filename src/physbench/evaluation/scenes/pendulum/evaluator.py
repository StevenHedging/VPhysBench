from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....io import canonical_sha256, sha256_file
from ...common.base import SceneAnalysis
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import summarize_mask_ious
from ...common.robustness import (
    add_subject_comparison,
    degraded_prediction_analysis,
    prediction_failure_code,
    reference_failure,
)
from ...common.subject import degraded_subject_metric, infer_reference_mode
from ...contracts import CaseEvaluationRequest, CaseEvaluationResult
from .scoring import (
    TraceQualityError,
    extract_trace,
    save_iou_curve,
    score_traces,
    write_per_frame_csv,
)
from .segmentation import Sam2PendulumSegmenter, SegmentationError
from .timeline import VideoProtocolError, reference_timeline, sample_video


def build_pendulum_timeline(
    reference_path: Path,
    config: dict[str, Any],
) -> tuple[list[float], str]:
    """Build the protocol-specific pendulum sampling timeline."""
    timeline = config["timeline"]
    fps = float(timeline["fps"])
    evaluator_type = config.get("type")
    if evaluator_type == "pendulum_state_v1":
        duration_s = float(timeline["duration_s"])
        frame_count = int(round(duration_s * fps)) + 1
        return (
            (np.arange(frame_count, dtype=np.float64) / fps).tolist(),
            "fixed_duration_legacy",
        )
    if evaluator_type in {"pendulum_state_v2", "pendulum_state_v3"}:
        return (
            reference_timeline(
                reference_path,
                fps=fps,
                max_duration_s=float(timeline["maximum_duration_s"]),
                minimum_duration_s=float(timeline["minimum_duration_s"]),
            ),
            "case_reference_bounded",
        )
    raise ValueError(f"unsupported pendulum evaluator type: {evaluator_type!r}")


def _timeline_provenance(
    config: dict[str, Any],
    *,
    fps: float,
    duration_s: float,
    frame_count: int,
    policy: str,
) -> dict[str, Any]:
    value = {
        "fps": fps,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "prediction_frame_zero_injected": False,
    }
    if config.get("type") in {"pendulum_state_v2", "pendulum_state_v3"}:
        value["policy"] = policy
    return value


class PendulumCaseEvaluator:
    evaluator_id = "pendulum_state"
    evaluator_version = "1.1"
    sequential_evaluator_version = "1.2"
    robust_evaluator_version = "1.3"
    scene_id = "pendulum"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.robust_subject = (
            config.get("evaluator_contract") == "robust_subject_v3"
        )
        decode_policy = config.get("timeline", {}).get(
            "decode_policy",
            "legacy_random_seek",
        )
        evaluator_type = config.get("type")
        expected_policy = (
            "legacy_random_seek"
            if evaluator_type == "pendulum_state_v1"
            else "sequential_forward"
        )
        if decode_policy != expected_policy:
            raise ValueError(
                f"{evaluator_type} requires decode_policy={expected_policy!r}; "
                f"got {decode_policy!r}"
            )
        if decode_policy == "sequential_forward":
            self.evaluator_version = self.sequential_evaluator_version
        if self.robust_subject:
            self.evaluator_version = self.robust_evaluator_version
        self._segmenter = Sam2PendulumSegmenter(config["sam2"])
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
            }
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.evaluator_id,
            "version": self.evaluator_version,
            "scene_id": self.scene_id,
            "implemented": True,
            "fingerprint": self.fingerprint,
            "primary_score": (
                "scene_subject_state_similarity"
                if self.robust_subject
                else "pendulum_state_similarity"
            ),
            "diagnostic": "physical_subject_mask_iou_curve",
            "segmentation": self._segmenter.describe(),
        }

    @staticmethod
    def _unavailable(
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        code: str,
        reason: str,
    ) -> CaseEvaluationResult:
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=evaluator,
            status="unavailable",
            score=None,
            reason_code=code,
            reason=reason,
        )

    @staticmethod
    def _error(
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        code: str,
        reason: str,
    ) -> CaseEvaluationResult:
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=evaluator,
            status="error",
            score=None,
            reason_code=code,
            reason=reason,
        )

    def _degraded(
        self,
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        *,
        code: str,
        reason: str,
        provenance: dict[str, Any] | None = None,
    ) -> CaseEvaluationResult:
        reference_mode = infer_reference_mode(request.case)
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=evaluator,
            status="evaluated",
            score=0.0,
            reason_code=code,
            reason=reason,
            metrics={
                "scene_subject_state_similarity": degraded_subject_metric(
                    reference_mode=reference_mode,
                    code=code,
                    reason=reason,
                )
            },
            quality={
                "degraded": True,
                "degradation_codes": [code],
                "temporal_coverage": 0.0,
                "reference_mode": reference_mode,
            },
            provenance={
                "degradation": {
                    "origin": "prediction",
                    "code": code,
                    "reason": reason,
                    "policy": "conservative_zero_not_evaluator_failure",
                },
                **(provenance or {}),
            },
        )

    @staticmethod
    def _resolve_asset(asset_root: Path, value: str) -> Path:
        path = (asset_root / value).resolve()
        try:
            path.relative_to(asset_root.resolve())
        except ValueError as exc:
            raise VideoProtocolError(
                "reference_path_escape",
                f"reference asset escapes dataset root: {value}",
            ) from exc
        return path

    def _reference(
        self, request: CaseEvaluationRequest
    ) -> tuple[Path, str, str | None]:
        case = request.case
        value = case["assets"].get("reference_video")
        if not value:
            raise VideoProtocolError(
                "no_trustworthy_physics_reference",
                "case has no reference_video",
            )
        path = self._resolve_asset(request.asset_root, value)
        if not path.is_file():
            raise VideoProtocolError(
                "reference_video_missing", f"reference video not found: {path}"
            )
        return path, "same_case_reference", None

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
        if self.robust_subject:
            return self._evaluate_robust(request)
        evaluator = self.describe()
        if request.case["scene_id"] != self.scene_id:
            return self._error(
                request,
                evaluator,
                "wrong_scene",
                f"pendulum evaluator received {request.case['scene_id']}",
            )
        prediction = request.prediction
        if prediction is None:
            return self._unavailable(
                request,
                evaluator,
                "prediction_record_missing",
                "planned job has no prediction record",
            )
        if prediction.get("status") != "complete":
            return self._unavailable(
                request,
                evaluator,
                "prediction_incomplete",
                f"prediction status is {prediction.get('status')!r}",
            )
        video_value = prediction.get("video_path")
        if not video_value:
            return self._unavailable(
                request,
                evaluator,
                "prediction_video_missing",
                "complete prediction has no video_path",
            )
        prediction_path = Path(video_value).resolve()
        if not prediction_path.is_file():
            return self._unavailable(
                request,
                evaluator,
                "prediction_video_missing",
                f"prediction video not found: {prediction_path}",
            )
        try:
            reference_path, reference_mode, parent_id = self._reference(request)
            timeline = self.config["timeline"]
            fps = float(timeline["fps"])
            times_s, timeline_policy = build_pendulum_timeline(
                reference_path,
                self.config,
            )
            duration_s = times_s[-1]
            spatial = self.config["spatial"]
            sampling_kwargs = {
                "sample_times_s": times_s,
                "width": int(spatial["width"]),
                "height": int(spatial["height"]),
                "pad_value": int(spatial.get("pad_value", 0)),
                "min_source_fps": float(timeline["minimum_source_fps"]),
                "duration_tolerance_s": float(
                    timeline.get("duration_tolerance_s", 0.02)
                ),
                "decode_policy": str(
                    timeline.get("decode_policy", "legacy_random_seek")
                ),
            }
            reference_video = sample_video(reference_path, **sampling_kwargs)
            prediction_video = sample_video(prediction_path, **sampling_kwargs)
            proposal = self.config["motion_proposal"]
            (
                reference_masks,
                reference_segmentation,
                reference_prompt,
            ) = self._segmenter.segment(
                reference_video.frames, proposal_config=proposal
            )
            try:
                (
                    prediction_masks,
                    prediction_segmentation,
                    _,
                ) = self._segmenter.segment(
                    prediction_video.frames, proposal_config=proposal
                )
            except SegmentationError as exc:
                if exc.code != "motion_prompt_failed":
                    raise
                (
                    prediction_masks,
                    prediction_segmentation,
                    _,
                ) = self._segmenter.segment(
                    prediction_video.frames,
                    proposal_config=proposal,
                    prompt=reference_prompt,
                    prompt_source="reference_geometry_fallback",
                )
            reference_trace = extract_trace(
                reference_masks,
                times_s,
                quality_config=self.config["quality"],
                period_config=self.config["period"],
            )
            prediction_trace = extract_trace(
                prediction_masks,
                times_s,
                quality_config=self.config["quality"],
                period_config=self.config["period"],
            )
            state_score = score_traces(
                reference_trace,
                prediction_trace,
                scoring_config=self.config["scoring"],
            )

            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            per_frame_path = request.artifact_dir / "per_frame.csv"
            curve_path = request.artifact_dir / "physical_subject_iou_curve.png"
            ious = write_per_frame_csv(
                per_frame_path,
                times_s=times_s,
                reference_masks=reference_masks,
                prediction_masks=prediction_masks,
                reference=reference_trace,
                prediction=prediction_trace,
            )
            save_iou_curve(
                curve_path,
                times_s=times_s,
                ious=ious,
                case_id=request.case["case_id"],
            )
            iou_summary = summarize_mask_ious(ious)
        except VideoProtocolError as exc:
            return self._unavailable(
                request, evaluator, exc.code, str(exc)
            )
        except (SegmentationError, TraceQualityError) as exc:
            return self._error(request, evaluator, exc.code, str(exc))

        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=self.describe(),
            status="evaluated",
            score=state_score["score"],
            metrics={
                "pendulum_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    **iou_summary,
                    "definition": "intersection / union of SAM2-segmented reference and generation physical-subject masks",
                    "role": "diagnostic_not_primary_score",
                },
            },
            quality={
                "evaluated_frames": len(times_s),
                "reference_valid_mask_ratio": reference_trace.valid_ratio,
                "prediction_valid_mask_ratio": prediction_trace.valid_ratio,
                "temporal_coverage": 1.0,
                "reference_mode": reference_mode,
            },
            artifacts={
                "per_frame_csv": str(per_frame_path),
                "physical_subject_iou_curve": str(curve_path),
            },
            provenance={
                "reference_video": str(reference_path),
                "reference_video_sha256": sha256_file(reference_path),
                "prediction_video": str(prediction_path),
                "prediction_video_sha256": sha256_file(prediction_path),
                "parent_case_id": parent_id,
                "timeline": _timeline_provenance(
                    self.config,
                    fps=fps,
                    duration_s=duration_s,
                    frame_count=len(times_s),
                    policy=timeline_policy,
                ),
                "sampling": {
                    "reference": {
                        "source": reference_video.info.to_dict(),
                        "source_indices": reference_video.source_indices,
                        "spatial_transform": reference_video.spatial_transform,
                    },
                    "prediction": {
                        "source": prediction_video.info.to_dict(),
                        "source_indices": prediction_video.source_indices,
                        "spatial_transform": prediction_video.spatial_transform,
                    },
                },
                "segmentation": {
                    "reference": reference_segmentation,
                    "prediction": prediction_segmentation,
                },
            },
        )

    def _evaluate_robust(
        self, request: CaseEvaluationRequest
    ) -> CaseEvaluationResult:
        evaluator = self.describe()
        if request.case["scene_id"] != self.scene_id:
            return self._error(
                request,
                evaluator,
                "wrong_scene",
                f"pendulum evaluator received {request.case['scene_id']}",
            )
        prediction = request.prediction
        if prediction is None:
            return self._degraded(
                request,
                evaluator,
                code="prediction_record_missing",
                reason="planned job has no prediction record",
            )
        if prediction.get("status") != "complete":
            return self._degraded(
                request,
                evaluator,
                code="prediction_incomplete",
                reason=f"prediction status is {prediction.get('status')!r}",
            )
        video_value = prediction.get("video_path")
        if not video_value:
            return self._degraded(
                request,
                evaluator,
                code="prediction_video_missing",
                reason="complete prediction has no video_path",
            )
        prediction_path = Path(video_value).resolve()
        if not prediction_path.is_file():
            return self._degraded(
                request,
                evaluator,
                code="prediction_video_missing",
                reason=f"prediction video not found: {prediction_path}",
            )

        try:
            reference_path, reference_mode, parent_id = self._reference(request)
            timeline = self.config["timeline"]
            fps = float(timeline["fps"])
            times_s, timeline_policy = build_pendulum_timeline(
                reference_path,
                self.config,
            )
            duration_s = times_s[-1]
            spatial = self.config["spatial"]
            sampling_kwargs = {
                "sample_times_s": times_s,
                "width": int(spatial["width"]),
                "height": int(spatial["height"]),
                "pad_value": int(spatial.get("pad_value", 0)),
                "min_source_fps": float(timeline["minimum_source_fps"]),
                "duration_tolerance_s": float(
                    timeline.get("duration_tolerance_s", 0.02)
                ),
                "decode_policy": str(timeline["decode_policy"]),
            }
            reference_video = sample_video(reference_path, **sampling_kwargs)
        except VideoProtocolError as exc:
            return self._unavailable(
                request, evaluator, exc.code, str(exc)
            )
        try:
            prediction_video = sample_video(
                prediction_path, **sampling_kwargs
            )
        except VideoProtocolError as exc:
            return self._degraded(
                request,
                evaluator,
                code=exc.code,
                reason=str(exc),
                provenance={
                    "reference_video": str(reference_path),
                    "reference_video_sha256": sha256_file(reference_path),
                    "prediction_video": str(prediction_path),
                    "prediction_video_sha256": sha256_file(prediction_path),
                },
            )

        proposal = self.config["motion_proposal"]
        try:
            (
                reference_masks,
                reference_segmentation,
                reference_prompt,
            ) = self._segmenter.segment(
                reference_video.frames, proposal_config=proposal
            )
            reference_trace = extract_trace(
                reference_masks,
                times_s,
                quality_config=self.config["quality"],
                period_config=self.config["period"],
            )
        except (SegmentationError, TraceQualityError) as exc:
            failure = reference_failure(
                exc, stage="pendulum observation"
            )
            return self._unavailable(
                request, evaluator, failure.code, str(failure)
            )

        try:
            try:
                (
                    prediction_masks,
                    prediction_segmentation,
                    _,
                ) = self._segmenter.segment(
                    prediction_video.frames,
                    proposal_config=proposal,
                )
            except SegmentationError as exc:
                if exc.code != "motion_prompt_failed":
                    raise
                (
                    prediction_masks,
                    prediction_segmentation,
                    _,
                ) = self._segmenter.segment(
                    prediction_video.frames,
                    proposal_config=proposal,
                    prompt=reference_prompt,
                    prompt_source="reference_geometry_fallback",
                )
            prediction_trace = extract_trace(
                prediction_masks,
                times_s,
                quality_config=self.config["quality"],
                period_config=self.config["period"],
            )
            state_score = score_traces(
                reference_trace,
                prediction_trace,
                scoring_config=self.config["scoring"],
            )
        except (SegmentationError, TraceQualityError) as exc:
            code, reason = prediction_failure_code(
                exc, stage="pendulum_observation"
            )
            analysis = degraded_prediction_analysis(
                request,
                times_s=times_s,
                reference_masks=reference_masks,
                code=code,
                reason=reason,
                scene_name="Pendulum",
            )
            return self._result_from_analysis(
                request,
                analysis=analysis,
                reference_path=reference_path,
                prediction_path=prediction_path,
                reference_mode=reference_mode,
                parent_id=parent_id,
                times_s=times_s,
                fps=fps,
                duration_s=duration_s,
                timeline_policy=timeline_policy,
                reference_video=reference_video,
                prediction_video=prediction_video,
                segmentation={
                    "reference": reference_segmentation,
                    "prediction": {
                        "status": "failed",
                        "reason_code": code,
                        "reason": reason,
                    },
                },
                reason_code=code,
                reason=reason,
            )

        request.artifact_dir.mkdir(parents=True, exist_ok=True)
        per_frame_path = request.artifact_dir / "per_frame.csv"
        curve_path = request.artifact_dir / "physical_subject_iou_curve.png"
        ious = write_per_frame_csv(
            per_frame_path,
            times_s=times_s,
            reference_masks=reference_masks,
            prediction_masks=prediction_masks,
            reference=reference_trace,
            prediction=prediction_trace,
        )
        save_iou_curve(
            curve_path,
            times_s=times_s,
            ious=ious,
            case_id=request.case["case_id"],
        )
        iou_summary = summarize_mask_ious(ious)
        analysis = SceneAnalysis(
            score=float(state_score["score"]),
            metrics={
                "pendulum_state_similarity": state_score,
                "physical_subject_mask_iou": {
                    **iou_summary,
                    "definition": (
                        "intersection / union of SAM2-segmented reference "
                        "and generation physical-subject masks"
                    ),
                    "role": "diagnostic_not_primary_score",
                },
            },
            quality={
                "reference_valid_mask_ratio": reference_trace.valid_ratio,
                "prediction_valid_mask_ratio": prediction_trace.valid_ratio,
            },
            artifacts={
                "per_frame_csv": str(per_frame_path),
                "physical_subject_iou_curve": str(curve_path),
            },
            provenance={
                "segmentation": {
                    "reference": reference_segmentation,
                    "prediction": prediction_segmentation,
                }
            },
        )
        try:
            analysis = add_subject_comparison(
                analysis,
                request,
                times_s=times_s,
                reference_frames=reference_video.frames,
                prediction_frames=prediction_video.frames,
                reference_masks=reference_masks,
                prediction_masks=prediction_masks,
                scene_name="Pendulum",
            )
        except ReferenceAnalysisError as exc:
            return self._unavailable(
                request, evaluator, exc.code, str(exc)
            )
        return self._result_from_analysis(
            request,
            analysis=analysis,
            reference_path=reference_path,
            prediction_path=prediction_path,
            reference_mode=reference_mode,
            parent_id=parent_id,
            times_s=times_s,
            fps=fps,
            duration_s=duration_s,
            timeline_policy=timeline_policy,
            reference_video=reference_video,
            prediction_video=prediction_video,
            segmentation={
                "reference": reference_segmentation,
                "prediction": prediction_segmentation,
            },
        )

    def _result_from_analysis(
        self,
        request: CaseEvaluationRequest,
        *,
        analysis: SceneAnalysis,
        reference_path: Path,
        prediction_path: Path,
        reference_mode: str,
        parent_id: str | None,
        times_s: list[float],
        fps: float,
        duration_s: float,
        timeline_policy: str,
        reference_video,
        prediction_video,
        segmentation: dict[str, Any],
        reason_code: str | None = None,
        reason: str | None = None,
    ) -> CaseEvaluationResult:
        analysis.quality.setdefault("evaluated_frames", len(times_s))
        analysis.quality.setdefault("temporal_coverage", 1.0)
        analysis.quality["reference_mode"] = reference_mode
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=self.describe(),
            status="evaluated",
            score=float(analysis.score),
            reason_code=reason_code,
            reason=reason,
            metrics=analysis.metrics,
            quality=analysis.quality,
            artifacts=analysis.artifacts,
            provenance={
                "reference_video": str(reference_path),
                "reference_video_sha256": sha256_file(reference_path),
                "prediction_video": str(prediction_path),
                "prediction_video_sha256": sha256_file(prediction_path),
                "parent_case_id": parent_id,
                "timeline": _timeline_provenance(
                    self.config,
                    fps=fps,
                    duration_s=duration_s,
                    frame_count=len(times_s),
                    policy=timeline_policy,
                ),
                "sampling": {
                    "reference": {
                        "source": reference_video.info.to_dict(),
                        "source_indices": reference_video.source_indices,
                        "spatial_transform": (
                            reference_video.spatial_transform
                        ),
                    },
                    "prediction": {
                        "source": prediction_video.info.to_dict(),
                        "source_indices": prediction_video.source_indices,
                        "spatial_transform": (
                            prediction_video.spatial_transform
                        ),
                    },
                },
                "segmentation": segmentation,
                **analysis.provenance,
            },
        )
