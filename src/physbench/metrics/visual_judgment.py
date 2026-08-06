from __future__ import annotations

from typing import Any

from .base import MetricResult, manual_score


def evaluate(case: dict[str, Any], prediction: dict[str, Any], scene: dict[str, Any], plugin: dict[str, Any]) -> MetricResult:
    reference = prediction.get("visual_reference_video") or case.get("assets", {}).get("reference_video")
    if case.get("has_real_reference_video") is False or not reference:
        return MetricResult(
            "visual_judgment", "not_applicable", None,
            "no real continuation video matching this case appearance",
        )
    score = manual_score(prediction, "visual_judgment")
    details = {
        "reference": reference,
        "attributes": scene.get("metric_spec", {}).get("visual_attributes", []),
    }
    if score is not None:
        return MetricResult("visual_judgment", "evaluated", score, "externally supplied perceptual score", details)
    if prediction.get("status") != "complete" or not prediction.get("video_path"):
        return MetricResult("visual_judgment", "unavailable", None, "generated video is unavailable", details)
    return MetricResult(
        "visual_judgment", "unavailable", None,
        "SSIM/LPIPS/VLM perceptual plugin is not configured", {**details, "plugin": plugin},
    )
