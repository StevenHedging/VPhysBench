from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from . import common_sense, prediction, visual_judgment
from .aggregation import aggregate


def evaluate_cases(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    metric_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case = {case["case_id"]: case for case in cases}
    results = []
    plugins = metric_config.get("plugins", {})
    for item in predictions:
        case = by_case[item["case_id"]]
        scene = scenes[case["scene_id"]]
        metric_objects = [
            common_sense.evaluate(case, item, scene, plugins.get("common_sense", {})),
            prediction.evaluate(case, item, scene, plugins.get("prediction", {})),
            visual_judgment.evaluate(case, item, scene, plugins.get("visual_judgment", {})),
        ]
        metrics = {metric.name: metric.to_dict() for metric in metric_objects}
        score, coverage = aggregate(metrics, metric_config)
        conditioning = item.get(
            "conditioning", item.get("prompt_profile_id", "unprofiled")
        )
        partition = item.get(
            "evaluation_partition", case.get("view_a_split", "unspecified")
        )
        results.append({
            "job_id": item["job_id"],
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "conditioning": conditioning,
            "prompt_profile_id": item.get("prompt_profile_id", conditioning),
            "view_a_split": case.get("view_a_split", partition),
            "evaluation_partition": partition,
            "ood_level": case["ood"]["level"],
            "ood_factors": case["ood"]["factors"],
            "metrics": metrics,
            "final_score": score,
            "metric_coverage": coverage,
        })

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(
            result["scene_id"],
            result["evaluation_partition"],
            result["conditioning"],
        )].append(result)
    breakdown = {}
    for (scene_id, partition, conditioning), items in sorted(groups.items()):
        scores = [item["final_score"] for item in items if item["final_score"] is not None]
        breakdown[f"{scene_id}/{partition}/{conditioning}"] = {
            "jobs": len(items),
            "scored_jobs": len(scores),
            "mean_score": mean(scores) if scores else None,
            "mean_metric_coverage": mean(item["metric_coverage"] for item in items),
        }
    # train_seen is an auxiliary memorization diagnostic, not a generalization set.
    # Keep its per-partition breakdown, but never let it change the official ID/OOD summary.
    official_results = [
        item for item in results if item["evaluation_partition"] != "train_seen"
    ]
    auxiliary_results = [
        item for item in results if item["evaluation_partition"] == "train_seen"
    ]
    all_scores = [
        item["final_score"] for item in official_results if item["final_score"] is not None
    ]
    auxiliary_scores = [
        item["final_score"] for item in auxiliary_results if item["final_score"] is not None
    ]
    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in official_results:
        prompt_groups[item["conditioning"]].append(item)
    prompt_breakdown = {}
    for prompt_profile_id, items in sorted(prompt_groups.items()):
        scores = [item["final_score"] for item in items if item["final_score"] is not None]
        prompt_breakdown[prompt_profile_id] = {
            "jobs": len(items),
            "scored_jobs": len(scores),
            "mean_score": mean(scores) if scores else None,
            "mean_metric_coverage": mean(item["metric_coverage"] for item in items),
        }
    summary = {
        "jobs": len(official_results),
        "all_jobs": len(results),
        "auxiliary_train_seen_jobs": len(auxiliary_results),
        "scored_jobs": len(all_scores),
        "mean_score": mean(all_scores) if all_scores else None,
        "mean_metric_coverage": (
            mean(item["metric_coverage"] for item in official_results) if official_results else 0.0
        ),
        "auxiliary_train_seen_scored_jobs": len(auxiliary_scores),
        "auxiliary_train_seen_mean_score": mean(auxiliary_scores) if auxiliary_scores else None,
        "auxiliary_train_seen_mean_metric_coverage": (
            mean(item["metric_coverage"] for item in auxiliary_results) if auxiliary_results else 0.0
        ),
        "prompt_breakdown": prompt_breakdown,
        "conditioning_breakdown": prompt_breakdown,
        "breakdown": breakdown,
    }
    return results, summary
