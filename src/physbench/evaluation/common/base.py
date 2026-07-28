from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...io import canonical_sha256, sha256_file
from ..contracts import CaseEvaluationRequest, CaseEvaluationResult
from .errors import SceneAnalysisError
from .media import (
    SampledVideo,
    VideoProtocolError,
    reference_timeline,
    sample_video,
)
from .reference import resolve_physics_reference


@dataclass
class SceneAnalysis:
    score: float
    metrics: dict[str, Any]
    quality: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


class ReferenceCaseEvaluator(ABC):
    """Shared lifecycle for scene evaluators that compare against real dynamics."""

    evaluator_id: str
    evaluator_version: str
    sequential_evaluator_version = "1.2"
    scene_id: str
    primary_score: str

    def __init__(self, config: dict[str, Any]):
        self.config = config
        decode_policy = config.get("timeline", {}).get(
            "decode_policy",
            "legacy_random_seek",
        )
        if decode_policy == "sequential_forward":
            self.evaluator_version = self.sequential_evaluator_version
        elif decode_policy != "legacy_random_seek":
            raise ValueError(
                f"unsupported evaluator decode policy: {decode_policy!r}"
            )
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
            }
        )

    def describe(self) -> dict[str, Any]:
        value = {
            "id": self.evaluator_id,
            "version": self.evaluator_version,
            "scene_id": self.scene_id,
            "implemented": True,
            "fingerprint": self.fingerprint,
            "primary_score": self.primary_score,
            "diagnostic": "physical_subject_mask_iou_curve",
        }
        observation = self.describe_observation()
        if observation:
            value["observation"] = observation
        return value

    def describe_observation(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def _outcome(
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        *,
        status: str,
        code: str,
        reason: str,
    ) -> CaseEvaluationResult:
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=evaluator,
            status=status,
            score=None,
            reason_code=code,
            reason=reason,
        )

    @abstractmethod
    def analyze(
        self,
        request: CaseEvaluationRequest,
        *,
        times_s: list[float],
        reference_video: SampledVideo,
        prediction_video: SampledVideo,
    ) -> SceneAnalysis:
        ...

    def evaluate(self, request: CaseEvaluationRequest) -> CaseEvaluationResult:
        evaluator = self.describe()
        if request.case.get("scene_id") != self.scene_id:
            return self._outcome(
                request,
                evaluator,
                status="error",
                code="wrong_scene",
                reason=(
                    f"{self.scene_id} evaluator received "
                    f"{request.case.get('scene_id')}"
                ),
            )
        prediction = request.prediction
        if prediction is None:
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_record_missing",
                reason="planned job has no prediction record",
            )
        if prediction.get("status") != "complete":
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_incomplete",
                reason=f"prediction status is {prediction.get('status')!r}",
            )
        video_value = prediction.get("video_path")
        if not video_value:
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_video_missing",
                reason="complete prediction has no video_path",
            )
        prediction_path = Path(video_value).resolve()
        if not prediction_path.is_file():
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_video_missing",
                reason=f"prediction video not found: {prediction_path}",
            )
        try:
            reference_path, reference_mode, parent_id = (
                resolve_physics_reference(request)
            )
            timeline = self.config["timeline"]
            times_s = reference_timeline(
                reference_path,
                fps=float(timeline["fps"]),
                max_duration_s=float(timeline["maximum_duration_s"]),
                minimum_duration_s=float(timeline["minimum_duration_s"]),
            )
            spatial = self.config["spatial"]
            sampling = {
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
            reference_video = sample_video(reference_path, **sampling)
            prediction_video = sample_video(prediction_path, **sampling)
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            analysis = self.analyze(
                request,
                times_s=times_s,
                reference_video=reference_video,
                prediction_video=prediction_video,
            )
        except VideoProtocolError as exc:
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code=exc.code,
                reason=str(exc),
            )
        except SceneAnalysisError as exc:
            return self._outcome(
                request,
                evaluator,
                status="error",
                code=exc.code,
                reason=str(exc),
            )

        analysis.quality.setdefault("evaluated_frames", len(times_s))
        analysis.quality.setdefault("temporal_coverage", 1.0)
        analysis.quality["reference_mode"] = reference_mode
        provenance = {
            "reference_video": str(reference_path),
            "reference_video_sha256": sha256_file(reference_path),
            "prediction_video": str(prediction_path),
            "prediction_video_sha256": sha256_file(prediction_path),
            "parent_case_id": parent_id,
            "timeline": {
                "fps": float(timeline["fps"]),
                "duration_s": times_s[-1],
                "frame_count": len(times_s),
                "policy": "case_reference_bounded",
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
            **analysis.provenance,
        }
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=self.describe(),
            status="evaluated",
            score=float(analysis.score),
            metrics=analysis.metrics,
            quality=analysis.quality,
            artifacts=analysis.artifacts,
            provenance=provenance,
        )
