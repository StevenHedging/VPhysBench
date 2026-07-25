from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....io import canonical_sha256, sha256_file
from ...contracts import CaseEvaluationRequest, CaseEvaluationResult
from .scoring import (
    TraceQualityError,
    extract_trace,
    save_iou_curve,
    score_traces,
    write_per_frame_csv,
)
from .segmentation import Sam2PendulumSegmenter, SegmentationError
from .timeline import VideoProtocolError, sample_video


class PendulumCaseEvaluator:
    evaluator_id = "pendulum_state"
    evaluator_version = "1.0"
    scene_id = "pendulum"

    def __init__(self, config: dict[str, Any]):
        self.config = config
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
            "primary_score": "pendulum_state_similarity",
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
        value = case["assets"].get("physics_reference_video")
        if not value:
            raise VideoProtocolError(
                "no_trustworthy_physics_reference",
                "case has no physics reference video",
            )
        parent_id = case.get("provenance", {}).get("parent_case_id")
        if case.get("ood", {}).get("level") == "ood1":
            if not parent_id:
                raise VideoProtocolError(
                    "ood_parent_missing",
                    "OOD1 pendulum case has no parent case",
                )
            parent = request.case_catalog.get(parent_id)
            if parent is None:
                raise VideoProtocolError(
                    "ood_parent_missing",
                    f"OOD1 parent case is absent from dataset: {parent_id}",
                )
            if parent.get("physics") != case.get("physics"):
                raise VideoProtocolError(
                    "ood_parent_physics_mismatch",
                    "OOD1 case and parent do not have identical physics labels",
                )
            parent_reference = parent.get("assets", {}).get(
                "physics_reference_video"
            )
            if value != parent_reference:
                raise VideoProtocolError(
                    "ood_parent_reference_mismatch",
                    "OOD1 physics reference is not the parent reference",
                )
            mode = "parent_physics_reference"
        else:
            mode = "same_case_reference"
        path = self._resolve_asset(request.asset_root, value)
        if not path.is_file():
            raise VideoProtocolError(
                "reference_video_missing", f"reference video not found: {path}"
            )
        return path, mode, parent_id

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
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
            duration_s = float(timeline["duration_s"])
            frame_count = int(round(duration_s * fps)) + 1
            times_s = (np.arange(frame_count) / fps).tolist()
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
                    "mean": float(np.mean(ious)),
                    "minimum": float(np.min(ious)),
                    "maximum": float(np.max(ious)),
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
                "timeline": {
                    "fps": fps,
                    "duration_s": duration_s,
                    "frame_count": len(times_s),
                    "prediction_frame_zero_injected": False,
                },
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
