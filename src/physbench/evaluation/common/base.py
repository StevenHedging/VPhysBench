from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...io import canonical_sha256, sha256_file
from ..contracts import CaseEvaluationRequest, CaseEvaluationResult
from .csti import (
    CSTIConfig,
    CSTIContractError,
    CSTIInput,
    evaluate_csti,
    not_applicable_csti_metric,
    zero_csti_metric,
)
from .entities import ReferenceCapability, materialize_entity_manifest
from .entities.timeline import build_common_time_grid
from .errors import ReferenceAnalysisError, SceneAnalysisError
from .media import (
    EvaluationTimelinePlan,
    SampledVideo,
    VideoProtocolError,
    probe_image_size,
    probe_video,
    reference_timeline,
    resolve_evaluation_timeline,
    resolve_shared_spatial_plan,
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
    csti_input: CSTIInput | None = None


class ReferenceCaseEvaluator(ABC):
    """Shared lifecycle for scene evaluators that compare against real dynamics."""

    evaluator_id: str
    evaluator_version: str
    sequential_evaluator_version = "1.2"
    robust_evaluator_version = "1.3"
    scene_id: str
    primary_score: str
    allow_partial_prediction = False

    def __init__(self, config: dict[str, Any]):
        self.config = config
        csti_value = config.get("general_metrics", {}).get("csti")
        self.csti_config = (
            CSTIConfig.from_mapping(csti_value)
            if csti_value is not None
            else None
        )
        self.csti_enabled = self.csti_config is not None
        self.robust_subject = (
            config.get("evaluator_contract") == "robust_subject_v3"
        )
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
        if self.robust_subject:
            self.evaluator_version = self.robust_evaluator_version
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
            "primary_score": (
                "scene_subject_state_similarity"
                if self.robust_subject
                else self.primary_score
            ),
            "diagnostic": "physical_subject_mask_iou_curve",
        }
        observation = self.describe_observation()
        if observation:
            value["observation"] = observation
        if self.csti_enabled:
            value["general_metrics"] = {
                "csti": self.config["general_metrics"]["csti"]
            }
        return value

    def describe_observation(self) -> dict[str, Any]:
        return {}

    def _attach_csti_metric(
        self,
        request: CaseEvaluationRequest,
        analysis: SceneAnalysis,
    ) -> dict[str, Any]:
        if self.csti_config is None:
            raise CSTIContractError(
                "csti_config_missing",
                "CSTI is not configured",
            )
        try:
            manifest = materialize_entity_manifest(request.case)
        except (TypeError, ValueError) as exc:
            raise CSTIContractError(
                "csti_manifest_invalid",
                f"cannot materialize the CSTI entity manifest: {exc}",
            ) from exc
        expected_entities = tuple(
            (entity.entity_id, entity.role_id)
            for entity in manifest.entities
        )
        if (
            manifest.reference_capability
            is not ReferenceCapability.SAME_CASE_GT
        ):
            return not_applicable_csti_metric(
                config=self.csti_config,
                reason_code="csti_requires_same_case_gt",
            )
        if analysis.csti_input is None:
            raise CSTIContractError(
                "csti_input_missing",
                "CSTI is enabled but the Scene analysis did not provide "
                "Tube input",
            )
        if (
            analysis.csti_input.reference_capability
            is not manifest.reference_capability
        ):
            raise CSTIContractError(
                "csti_reference_capability_mismatch",
                "CSTI input reference capability differs from the Case manifest",
            )
        analysis.provenance["csti"] = {
            "algorithm": self.csti_config.algorithm,
            "spatial_tolerance_fraction": (
                self.csti_config.spatial_tolerance_fraction
            ),
            "temporal_tolerance_s": self.csti_config.temporal_tolerance_s,
            "condition_frame_policy": (
                self.csti_config.condition_frame_policy
            ),
            "initial_frames_excluded": (
                self.csti_config.initial_frames_excluded
            ),
            "entity_manifest_materializer_id": manifest.materializer_id,
            "entity_manifest_digest": manifest.digest,
        }
        return evaluate_csti(
            analysis.csti_input,
            expected_entities=expected_entities,
            config=self.csti_config,
        )

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

    def _prediction_media_failure(
        self,
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        *,
        code: str,
        reason: str,
        times_s: list[float] | None = None,
        reference_path: Path | None = None,
        prediction_path: Path | None = None,
    ) -> CaseEvaluationResult:
        if self.robust_subject:
            return self._degraded_output(
                request,
                evaluator,
                code=code,
                reason=reason,
                times_s=times_s,
                reference_path=reference_path,
                prediction_path=prediction_path,
            )
        return self._outcome(
            request,
            evaluator,
            status="protocol_error",
            code=code,
            reason=reason,
        )

    def _degraded_output(
        self,
        request: CaseEvaluationRequest,
        evaluator: dict[str, Any],
        *,
        code: str,
        reason: str,
        times_s: list[float] | None = None,
        reference_path: Path | None = None,
        prediction_path: Path | None = None,
        artifact_failures: list[dict[str, str]] | None = None,
    ) -> CaseEvaluationResult:
        from .artifacts import save_iou_curve
        from .subject import degraded_subject_metric, infer_reference_mode

        reference_mode = infer_reference_mode(request.case)
        artifacts: dict[str, Any] = {}
        audited_artifact_failures = list(artifact_failures or [])
        if times_s:
            try:
                request.artifact_dir.mkdir(parents=True, exist_ok=True)
                curve_path = (
                    request.artifact_dir / "physical_subject_iou_curve.png"
                )
                save_iou_curve(
                    curve_path,
                    times_s=times_s,
                    ious=[0.0] * len(times_s),
                    case_id=request.case["case_id"],
                    scene_name=self.scene_id,
                )
                artifacts["physical_subject_iou_curve"] = str(curve_path)
            except Exception as exc:
                audited_artifact_failures.append(
                    {
                        "code": "degraded_artifact_write_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        provenance: dict[str, Any] = {
            "degradation": {
                "origin": "prediction",
                "code": code,
                "reason": reason,
                "policy": "conservative_zero_not_evaluator_failure",
            },
            "artifact_failures": audited_artifact_failures,
        }
        metrics = {
            "scene_subject_state_similarity": degraded_subject_metric(
                reference_mode=reference_mode,
                code=code,
                reason=reason,
            )
        }
        if self.csti_config is not None:
            manifest = materialize_entity_manifest(request.case)
            if (
                manifest.reference_capability
                is ReferenceCapability.SAME_CASE_GT
            ):
                metrics["csti"] = zero_csti_metric(
                    expected_entities=tuple(
                        (entity.entity_id, entity.role_id)
                        for entity in manifest.entities
                    ),
                    config=self.csti_config,
                    degradation_code=code,
                    degradation_reason=reason,
                )
            else:
                metrics["csti"] = not_applicable_csti_metric(
                    config=self.csti_config,
                    reason_code="csti_requires_same_case_gt",
                )
        if reference_path is not None and reference_path.is_file():
            provenance.update(
                {
                    "reference_video": str(reference_path),
                    "reference_video_sha256": sha256_file(reference_path),
                }
            )
        if prediction_path is not None and prediction_path.is_file():
            provenance.update(
                {
                    "prediction_video": str(prediction_path),
                    "prediction_video_sha256": sha256_file(prediction_path),
                }
            )
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=evaluator,
            status="evaluated",
            score=0.0,
            reason_code=code,
            reason=reason,
            metrics=metrics,
            quality={
                "degraded": True,
                "degradation_codes": [code],
                "temporal_coverage": 0.0,
                "reference_mode": reference_mode,
                "artifact_failures": audited_artifact_failures,
            },
            artifacts=artifacts,
            provenance=provenance,
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
            if self.robust_subject:
                return self._degraded_output(
                    request,
                    evaluator,
                    code="prediction_record_missing",
                    reason="planned job has no prediction record",
                )
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_record_missing",
                reason="planned job has no prediction record",
            )
        if prediction.get("status") != "complete":
            if self.robust_subject:
                return self._degraded_output(
                    request,
                    evaluator,
                    code="prediction_incomplete",
                    reason=f"prediction status is {prediction.get('status')!r}",
                )
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_incomplete",
                reason=f"prediction status is {prediction.get('status')!r}",
            )
        video_value = prediction.get("video_path")
        if not video_value:
            if self.robust_subject:
                return self._degraded_output(
                    request,
                    evaluator,
                    code="prediction_video_missing",
                    reason="complete prediction has no video_path",
                )
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_video_missing",
                reason="complete prediction has no video_path",
            )
        prediction_path = Path(video_value).resolve()
        if not prediction_path.is_file():
            if self.robust_subject:
                return self._degraded_output(
                    request,
                    evaluator,
                    code="prediction_video_missing",
                    reason=f"prediction video not found: {prediction_path}",
                    prediction_path=prediction_path,
                )
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code="prediction_video_missing",
                reason=f"prediction video not found: {prediction_path}",
            )
        reference_video: SampledVideo | None = None
        reference_path: Path | None = None
        times_s: list[float] = []
        shared_media_provenance: dict[str, Any] | None = None
        timeline_plan: EvaluationTimelinePlan | None = None
        reference_info = None
        prediction_info = None
        try:
            reference_path, reference_mode, parent_id = (
                resolve_physics_reference(request)
            )
            timeline = self.config["timeline"]
            timeline_policy = str(
                timeline.get("policy", "fixed_reference_cap_v1")
            )
            if timeline_policy in {
                "physical_reference_full_common_fps_v1",
                "physical_overlap_common_fps_v1",
            }:
                try:
                    reference_info = probe_video(reference_path)
                except VideoProtocolError as exc:
                    raise VideoProtocolError(
                        f"reference_{exc.code}",
                        f"cannot inspect physics reference video: {exc}",
                    ) from exc
                try:
                    prediction_info = probe_video(prediction_path)
                except VideoProtocolError as exc:
                    raise VideoProtocolError(
                        f"prediction_{exc.code}",
                        f"cannot inspect prediction video: {exc}",
                    ) from exc
                reference_case = (
                    request.case_catalog.get(parent_id, request.case)
                    if parent_id is not None
                    else request.case
                )
                temporal = reference_case.get("temporal", {})
                reference_source_time_scale = float(
                    temporal.get("encoded_to_physical_speed", 1.0)
                )
                timeline_plan = resolve_evaluation_timeline(
                    reference_info=reference_info,
                    prediction_info=prediction_info,
                    config=timeline,
                    reference_source_time_scale=(
                        reference_source_time_scale
                    ),
                )
                times_s = list(timeline_plan.sample_times_s)
                resolved_timeline_fps = timeline_plan.fps
                reference_source_time_scale = (
                    timeline_plan.reference_source_time_scale
                )
            else:
                times_s = reference_timeline(
                    reference_path,
                    fps=float(timeline["fps"]),
                    max_duration_s=float(timeline["maximum_duration_s"]),
                    minimum_duration_s=float(timeline["minimum_duration_s"]),
                )
                resolved_timeline_fps = float(timeline["fps"])
                reference_source_time_scale = 1.0
            spatial = self.config["spatial"]
            spatial_policy = str(
                spatial.get(
                    "policy",
                    "preserve_aspect_ratio_letterbox",
                )
            )
            common_sampling = {
                "sample_times_s": times_s,
                "pad_value": int(spatial.get("pad_value", 0)),
                "min_source_fps": float(timeline["minimum_source_fps"]),
                "duration_tolerance_s": float(
                    timeline.get("duration_tolerance_s", 0.02)
                ),
                "decode_policy": str(
                    timeline.get("decode_policy", "legacy_random_seek")
                ),
            }
        except VideoProtocolError as exc:
            if exc.code.startswith("prediction_"):
                return self._prediction_media_failure(
                    request,
                    evaluator,
                    code=exc.code,
                    reason=str(exc),
                    times_s=times_s,
                    reference_path=reference_path,
                    prediction_path=prediction_path,
                )
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code=exc.code,
                reason=str(exc),
            )

        if spatial_policy == "preserve_aspect_ratio_letterbox":
            reference_sampling = {
                **common_sampling,
                "width": int(spatial["width"]),
                "height": int(spatial["height"]),
                "source_time_scale": reference_source_time_scale,
            }
            prediction_sampling = {
                **common_sampling,
                "width": int(spatial["width"]),
                "height": int(spatial["height"]),
                "source_time_scale": 1.0,
            }
        elif spatial_policy == "shared_reference_content_no_pad_v1":
            try:
                if reference_mode == "parent_physics_reference":
                    raise VideoProtocolError(
                        "reference_parent_media_contract_unsupported",
                        "the no-padding protocol requires a same-Case physical "
                        "reference; parent-reference coordinate mapping is not "
                        "yet defined without cropping, padding, or content-based "
                        "registration",
                    )
                if reference_info is None:
                    try:
                        reference_info = probe_video(reference_path)
                    except VideoProtocolError as exc:
                        raise VideoProtocolError(
                            f"reference_{exc.code}",
                            f"cannot inspect physics reference video: {exc}",
                        ) from exc
                first_frame_asset = request.case.get("assets", {}).get(
                    "first_frame"
                )
                contract = prediction.get("media_contract")
                if contract is not None and not isinstance(contract, dict):
                    raise VideoProtocolError(
                        "media_contract_invalid",
                        "prediction media_contract must be an object",
                    )
                if contract is not None:
                    if not isinstance(first_frame_asset, str) or not first_frame_asset:
                        raise VideoProtocolError(
                            "reference_conditioning_asset_missing",
                            "I2V spatial alignment requires a Case first_frame asset",
                        )
                    asset_root = request.asset_root.resolve()
                    conditioning_path = (asset_root / first_frame_asset).resolve()
                    try:
                        conditioning_path.relative_to(asset_root)
                    except ValueError as exc:
                        raise VideoProtocolError(
                            "reference_conditioning_asset_path_escape",
                            "Case first_frame escapes the Dataset asset root",
                        ) from exc
                    try:
                        conditioning_width, conditioning_height = (
                            probe_image_size(conditioning_path)
                        )
                    except VideoProtocolError as exc:
                        raise VideoProtocolError(
                            f"reference_{exc.code}",
                            f"cannot inspect Case conditioning image: {exc}",
                        ) from exc
                    if (
                        conditioning_width * reference_info.height
                        != conditioning_height * reference_info.width
                    ):
                        raise VideoProtocolError(
                            "reference_conditioning_aspect_mismatch",
                            "Case first_frame and physics reference video have "
                            "different aspect ratios",
                        )
                if prediction_info is None:
                    prediction_info = probe_video(prediction_path)
                spatial_plan = resolve_shared_spatial_plan(
                    reference_info=reference_info,
                    prediction_info=prediction_info,
                    maximum_width=int(spatial["width"]),
                    maximum_height=int(spatial["height"]),
                    media_contract=contract,
                    expected_conditioning_asset=(
                        first_frame_asset
                        if isinstance(first_frame_asset, str)
                        else None
                    ),
                )
                shared_media_provenance = spatial_plan.provenance
                reference_sampling = {
                    **common_sampling,
                    "width": spatial_plan.width,
                    "height": spatial_plan.height,
                    "spatial_policy": "reference_content_crop_resize_no_pad",
                    "crop_xywh": spatial_plan.reference_crop_xywh,
                    "source_time_scale": reference_source_time_scale,
                }
                prediction_sampling = {
                    **common_sampling,
                    "width": spatial_plan.width,
                    "height": spatial_plan.height,
                    "spatial_policy": "reference_content_crop_resize_no_pad",
                    "crop_xywh": spatial_plan.prediction_crop_xywh,
                    "source_time_scale": 1.0,
                }
            except VideoProtocolError as exc:
                if exc.code.startswith("reference_"):
                    return self._outcome(
                        request,
                        evaluator,
                        status="unavailable",
                        code=exc.code,
                        reason=str(exc),
                    )
                return self._prediction_media_failure(
                    request,
                    evaluator,
                    code=exc.code,
                    reason=str(exc),
                    times_s=times_s,
                    reference_path=reference_path,
                    prediction_path=prediction_path,
                )
        else:
            return self._outcome(
                request,
                evaluator,
                status="error",
                code="invalid_spatial_policy",
                reason=f"unsupported spatial policy: {spatial_policy!r}",
            )

        try:
            reference_video = sample_video(
                reference_path,
                **reference_sampling,
            )
        except VideoProtocolError as exc:
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code=exc.code,
                reason=str(exc),
            )

        try:
            prediction_video = sample_video(
                prediction_path,
                **prediction_sampling,
                allow_partial=self.allow_partial_prediction,
            )
        except VideoProtocolError as exc:
            return self._prediction_media_failure(
                request,
                evaluator,
                code=exc.code,
                reason=str(exc),
                times_s=times_s,
                reference_path=reference_path,
                prediction_path=prediction_path,
            )

        artifact_failures: list[dict[str, str]] = []
        try:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            artifact_failures.append(
                {
                    "code": "artifact_directory_setup_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            assert reference_video is not None
            analysis = self.analyze(
                request,
                times_s=times_s,
                reference_video=reference_video,
                prediction_video=prediction_video,
            )
        except ReferenceAnalysisError as exc:
            return self._outcome(
                request,
                evaluator,
                status="unavailable",
                code=exc.code,
                reason=str(exc),
            )
        except SceneAnalysisError as exc:
            if self.robust_subject:
                return self._degraded_output(
                    request,
                    evaluator,
                    code=exc.code,
                    reason=str(exc),
                    times_s=times_s,
                    reference_path=reference_path,
                    prediction_path=prediction_path,
                    artifact_failures=artifact_failures,
                )
            return self._outcome(
                request,
                evaluator,
                status="error",
                code=exc.code,
                reason=str(exc),
            )

        if artifact_failures:
            analysis.quality["artifact_failures"] = [
                *artifact_failures,
                *list(analysis.quality.get("artifact_failures", [])),
            ]
            analysis.provenance["artifact_failures"] = [
                *artifact_failures,
                *list(analysis.provenance.get("artifact_failures", [])),
            ]
        if self.csti_enabled:
            try:
                analysis.metrics["csti"] = self._attach_csti_metric(
                    request,
                    analysis,
                )
            except CSTIContractError as exc:
                return self._outcome(
                    request,
                    evaluator,
                    status="error",
                    code=exc.code,
                    reason=f"CSTI contract failed: {exc}",
                )

        analysis.quality.setdefault("evaluated_frames", len(times_s))
        if self.allow_partial_prediction:
            availability = prediction_video.available or [
                True
            ] * len(times_s)
            time_grid = build_common_time_grid(times_s)
            temporal_coverage = (
                time_grid.integrate(availability) / time_grid.duration_s
                if time_grid.duration_s > 1e-12
                else float(sum(availability) / len(availability))
            )
            analysis.quality.setdefault(
                "temporal_coverage",
                temporal_coverage,
            )
            analysis.quality.setdefault(
                "available_prediction_frames",
                int(sum(availability)),
            )
        else:
            analysis.quality.setdefault("temporal_coverage", 1.0)
        if (
            timeline_plan is not None
            and timeline_plan.policy == "physical_overlap_common_fps_v1"
        ):
            resolution = timeline_plan.provenance
            analysis.quality["temporal_coverage"] = float(
                resolution["prediction_temporal_coverage"]
            )
            analysis.quality["reference_physical_duration_s"] = float(
                resolution["reference_physical_duration_s"]
            )
            analysis.quality["prediction_physical_duration_s"] = float(
                resolution["prediction_physical_duration_s"]
            )
            analysis.quality["evaluated_physical_duration_s"] = float(
                timeline_plan.duration_s
            )
            analysis.quality["duration_mismatch_penalized"] = False
        analysis.quality["reference_mode"] = reference_mode
        provenance = {
            "reference_video": str(reference_path),
            "reference_video_sha256": sha256_file(reference_path),
            "prediction_video": str(prediction_path),
            "prediction_video_sha256": sha256_file(prediction_path),
            "parent_case_id": parent_id,
            "timeline": {
                "fps": float(resolved_timeline_fps),
                "duration_s": times_s[-1],
                "frame_count": len(times_s),
                "policy": (
                    timeline_plan.policy
                    if timeline_plan is not None
                    else "case_reference_bounded"
                ),
                "prediction_frame_zero_injected": False,
                **(
                    {"resolution": timeline_plan.provenance}
                    if timeline_plan is not None
                    else {}
                ),
            },
            "sampling": {
                "reference": {
                    "source": reference_video.info.to_dict(),
                    "source_indices": reference_video.source_indices,
                    "spatial_transform": reference_video.spatial_transform,
                    "temporal_transform": (
                        reference_video.temporal_transform
                    ),
                },
                "prediction": {
                    "source": prediction_video.info.to_dict(),
                    "source_indices": prediction_video.source_indices,
                    "spatial_transform": prediction_video.spatial_transform,
                    "temporal_transform": (
                        prediction_video.temporal_transform
                    ),
                    **(
                        {"available": prediction_video.available}
                        if self.allow_partial_prediction
                        else {}
                    ),
                },
            },
            **(
                {"shared_media_contract": shared_media_provenance}
                if shared_media_provenance is not None
                else {}
            ),
            **analysis.provenance,
        }
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id=request.case["scene_id"],
            evaluator=self.describe(),
            status="evaluated",
            score=float(analysis.score),
            reason_code=(
                analysis.quality.get("degradation_codes", [None])[0]
                if analysis.quality.get("degraded") is True
                else None
            ),
            reason=(
                analysis.quality.get("degradation_reason")
                if analysis.quality.get("degraded") is True
                else None
            ),
            metrics=analysis.metrics,
            quality=analysis.quality,
            artifacts=analysis.artifacts,
            provenance=provenance,
        )
