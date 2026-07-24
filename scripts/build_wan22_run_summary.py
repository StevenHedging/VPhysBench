#!/usr/bin/env python3
"""Write a compact Chinese record for a completed WAN Task-1 benchmark run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run = read_json(run_dir / "run.json", {})
    baseline = read_json(run_dir / "frozen_baseline.json", {})
    plan = read_json(run_dir / "plan.json", {})
    cases = {item["case_id"]: item for item in read_jsonl(run_dir / "frozen_cases.jsonl")}
    sampling = read_json(run_dir / "artifacts" / "wan22" / "training_sampling_plan.json", {})
    loss = read_json(run_dir / "artifacts" / "wan22" / "loss_analysis" / "loss_summary.json", {})
    workers = read_json(run_dir / "artifacts" / "wan22" / "inference_workers" / "summary.json", {})
    summary = read_json(run_dir / "summary.json", {})
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    train_counts = Counter(cases[case_id]["scene_id"] for case_id in plan.get("train_case_ids", []))
    job_counts = Counter((item["scene_id"], item["evaluation_partition"]) for item in plan.get("jobs", []))
    prediction_counts = Counter(
        (
            item.get("prompt_profile_id", "unprofiled"),
            item.get("evaluation_partition"),
            item.get("status"),
        )
        for item in predictions
    )
    lora = baseline.get("lora", {})
    media = baseline.get("media_adapter", {})
    buckets = media.get("aspect_ratio_buckets", {}).get("buckets", {})

    lines = [
        "# WAN2.2 + LoRA 三场景分桶实验记录\n\n",
        f"- Run ID：`{run.get('run_id', run_dir.name)}`\n",
        f"- 状态：`{run.get('status', 'unknown')}`\n",
        f"- 基线：`{baseline.get('baseline_id')}`\n",
        f"- Task：`{plan.get('task_id')}`（视图 A，训练后评测 ID/OOD1）\n\n",
        "## 提示词条件\n\n",
        f"- 训练 profile：`{plan.get('prompt_profiles', {}).get('train')}`\n",
        f"- 评测 profiles：`"
        f"{', '.join(plan.get('prompt_profiles', {}).get('eval', []))}`\n",
        "- 每个评测 profile 复用同一份 Adapter、首帧、seed 与生成超参；"
        "仅文本条件不同。\n\n",
        "## 数据与任务\n\n",
        "| 场景 | 唯一训练 case | 平衡后 metadata 行 | test_id jobs | test_ood1 jobs | train_seen jobs |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    balanced = sampling.get("balanced_scene_counts", {})
    for scene in sorted(train_counts):
        lines.append(
            f"| `{scene}` | {train_counts[scene]} | {balanced.get(scene, 0)} | "
            f"{job_counts[(scene, 'test_id')]} | {job_counts[(scene, 'test_ood1')]} | "
            f"{job_counts[(scene, 'train_seen')]} |\n"
        )
    lines.extend([
        "\n`train_seen` 是辅助记忆能力检查，不计入官方 ID/OOD 聚合分数。\n\n",
        "## 模型侧横竖屏分桶\n\n",
    ])
    for name, bucket in buckets.items():
        lines.append(
            f"- `{name}`：`{bucket['width']}×{bucket['height']}`，scene="
            f"`{','.join(bucket.get('scene_ids', []))}`\n"
        )
    lines.extend([
        "\n原始资产保持只读；分辨率、FPS 与帧数适配均发生在该 run 的 WAN 私有派生缓存中。\n\n",
        "## 训练配置\n\n",
        "| 项目 | 配置 |\n|---|---|\n",
        f"| 基础模型 | WAN2.2-TI2V-5B |\n",
        f"| 算法 | {lora.get('algorithm')} |\n",
        f"| LoRA | rank {lora.get('rank')}；`{lora.get('target_modules')}` |\n",
        f"| 优化器 / Scheduler | {lora.get('optimizer')} / {lora.get('scheduler')} |\n",
        f"| LR / Weight decay | `{lora.get('learning_rate')}` / `{lora.get('weight_decay')}` |\n",
        f"| Repeat × Epoch | {lora.get('dataset_repeat')} × {lora.get('num_epochs')} |\n",
        f"| 8 卡优化步数 | {sampling.get('expected_optimizer_steps_per_epoch')} steps/epoch；"
        f"共 {sampling.get('expected_total_optimizer_steps')} |\n",
        f"| 精度 / Seed | {lora.get('precision')} / {lora.get('seed')} |\n",
        f"| 时间规格 | {media.get('fps')} FPS；每条最多 {media.get('max_frames')} 帧；不循环补帧 |\n\n",
        "## Loss\n\n",
    ])
    if loss:
        lines.extend([
            f"- 记录 step：`{loss.get('recorded_steps')}/{loss.get('expected_total_steps')}`\n",
            f"- 均值 / 中位数：`{loss.get('mean', 0):.6f}` / `{loss.get('median', 0):.6f}`\n",
            f"- 范围：`{loss.get('min', 0):.6f}–{loss.get('max', 0):.6f}`\n",
            f"- 前/后 100 step 均值：`{loss.get('first_100_mean', 0):.6f}` / "
            f"`{loss.get('last_100_mean', 0):.6f}`\n\n",
            "![训练 loss](artifacts/wan22/loss_analysis/loss_curve.png)\n\n",
        ])
    else:
        lines.append("Loss 导出尚不可用。\n\n")
    lines.extend([
        "## 推理产物\n\n",
        f"- 计划 / 已记录 / 完成 / 失败：`{workers.get('planned_jobs', len(plan.get('jobs', [])))}` / "
        f"`{workers.get('recorded_jobs', len(predictions))}` / `{workers.get('completed_jobs', 0)}` / "
        f"`{workers.get('failed_jobs', 0)}`\n",
        f"- 官方评测 jobs：`{summary.get('jobs', 0)}`；辅助 train_seen jobs："
        f"`{summary.get('auxiliary_train_seen_jobs', 0)}`\n",
    ])
    for (prompt_profile_id, partition, status), count in sorted(prediction_counts.items()):
        lines.append(f"- `{prompt_profile_id}/{partition}/{status}`：{count}\n")
    if plan.get("train_preview", {}).get("enabled"):
        lines.extend(["\n训练样本展示：[train_seen_gallery.md](train_seen_gallery.md)\n\n"])
    lines.extend([
        "## 关键路径\n\n",
        "- Adapter：`artifacts/wan22/checkpoints/`\n",
        "- 训练日志：`artifacts/wan22/checkpoints/train.log`\n",
        "- Loss：`artifacts/wan22/loss_analysis/`\n",
        "- 生成视频：`predictions/`\n",
        "- 冻结配置与划分：`frozen_baseline.json`、`frozen_split.json`、`plan.json`\n",
        "- 冻结提示词：`frozen_prompt_profiles.json`、`resolved_prompts.jsonl`\n",
        "- 汇总指标：`summary.json`、`case_metrics.jsonl`、`report.md`\n",
    ])
    output = run_dir / "experiment_summary.md"
    output.write_text("".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
