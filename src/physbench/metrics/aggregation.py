from __future__ import annotations

from typing import Any


def aggregate(metric_results: dict[str, dict[str, Any]], config: dict[str, Any]) -> tuple[float | None, float]:
    weights = config.get("weights", {})
    applicable = {
        name: float(weights.get(name, 0.0))
        for name, result in metric_results.items()
        if result["status"] != "not_applicable" and float(weights.get(name, 0.0)) > 0
    }
    denominator = sum(applicable.values())
    if denominator <= 0:
        return None, 0.0
    evaluated = {
        name: weight for name, weight in applicable.items()
        if metric_results[name]["status"] == "evaluated"
    }
    coverage = sum(evaluated.values()) / denominator
    if not evaluated:
        return None, coverage
    policy = config.get("aggregation_policy", "require_all_applicable")
    if policy == "require_all_applicable" and coverage < 1.0 - 1e-12:
        return None, coverage
    if policy not in {"require_all_applicable", "reweight_available"}:
        raise ValueError(f"unknown aggregation policy {policy}")
    score = sum(metric_results[name]["score"] * weight for name, weight in evaluated.items()) / sum(evaluated.values())
    return float(score), float(coverage)

