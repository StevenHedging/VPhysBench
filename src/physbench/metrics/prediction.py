from __future__ import annotations

from typing import Any

from .base import MetricResult, manual_score


def evaluate(case: dict[str, Any], prediction: dict[str, Any], scene: dict[str, Any], plugin: dict[str, Any]) -> MetricResult:
    assets = case.get("assets", {})
    reference = (
        prediction.get("evaluation_reference_video")
        or assets.get("reference_video")
        or assets.get("physics_reference_video")
    )
    if not reference:
        return MetricResult("prediction", "not_applicable", None, "no physics reference video")
    score = manual_score(prediction, "prediction")
    details = {
        "reference": reference,
        "subjects": scene.get("metric_spec", {}).get("prediction_subjects", []),
        "quantities": scene.get("metric_spec", {}).get("prediction_quantities", []),
    }
    if score is not None:
        return MetricResult("prediction", "evaluated", score, "externally supplied masked-prediction score", details)
    if prediction.get("status") != "complete" or not prediction.get("video_path"):
        return MetricResult("prediction", "unavailable", None, "generated video is unavailable", details)
    return MetricResult(
        "prediction", "unavailable", None,
        "scene segmenter/mask evaluator is not configured", {**details, "plugin": plugin},
    )
