from __future__ import annotations

from typing import Any


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_report(run: dict[str, Any], plan: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark run: {run['run_id']}\n\n",
        f"- Task: `{run['task_id']}` (`{plan['mode']}`, view `{plan['view']}`)\n",
        f"- Baseline: `{run['baseline_id']}`\n",
        f"- Dataset/manifest SHA-256: "
        f"`{run.get('dataset_digest', run.get('manifest_sha256'))}`\n",
        f"- Status: `{run['status']}`\n",
        f"- Train cases: {len(plan['train_case_ids'])}\n",
        f"- Evaluation jobs: {len(plan['jobs'])}\n",
        f"- Official ID/OOD jobs: {summary['jobs']}\n",
        f"- Auxiliary train-seen jobs: {summary.get('auxiliary_train_seen_jobs', 0)}\n",
        f"- Train prompt profile: `{plan.get('prompt_profiles', {}).get('train')}`\n",
        f"- Evaluation prompt profiles: "
        f"`{', '.join(plan.get('prompt_profiles', {}).get('eval', []))}`\n",
        f"- Official scored jobs: {summary['scored_jobs']}/{summary['jobs']}\n",
        f"- Mean score: {_score(summary['mean_score'])}\n",
        f"- Mean metric coverage: {summary['mean_metric_coverage']:.4f}\n\n",
        "## Breakdown\n\n",
        "| Scene / partition / prompt | Jobs | Scored | Mean score | Metric coverage |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for name, item in summary["breakdown"].items():
        lines.append(
            f"| `{name}` | {item['jobs']} | {item['scored_jobs']} | "
            f"{_score(item['mean_score'])} | {item['mean_metric_coverage']:.4f} |\n"
        )
    if summary.get("prompt_breakdown"):
        lines.extend([
            "\n## Prompt comparison\n\n",
            "| Prompt profile | Jobs | Scored | Mean score | Metric coverage |\n",
            "|---|---:|---:|---:|---:|\n",
        ])
        for name, item in summary["prompt_breakdown"].items():
            lines.append(
                f"| `{name}` | {item['jobs']} | {item['scored_jobs']} | "
                f"{_score(item['mean_score'])} | {item['mean_metric_coverage']:.4f} |\n"
            )
    if summary["scored_jobs"] == 0:
        lines.extend([
            "\n## Interpretation\n\n",
            "本次没有可聚合分数。预测视频是否生成成功应以 `predictions.jsonl` 为准；"
            "当前 VLM、分割器和感知评价器尚未接入，因此即使视频已生成，指标仍保持"
            " unavailable/not_applicable，不把缺失的评价能力记成零分。\n",
        ])
    return "".join(lines)
