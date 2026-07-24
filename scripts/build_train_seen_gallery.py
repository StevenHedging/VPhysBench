#!/usr/bin/env python3
"""Build an auditable storyboard gallery for optional View-A train_seen jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_frames(path: Path) -> int:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames", "-of", "json", str(path),
    ], check=True, text=True, capture_output=True)
    stream = json.loads(result.stdout)["streams"][0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if value in {None, "N/A"}:
        raise ValueError(f"cannot determine frame count: {path}")
    return int(value)


def storyboard(video: Path, output: Path, count: int = 8) -> None:
    frames = probe_frames(video)
    indices = sorted({round(index * (frames - 1) / max(1, count - 1)) for index in range(count)})
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", (
            f"select='{expression}',"
            "scale=240:240:force_original_aspect_ratio=decrease,"
            "pad=240:240:(ow-iw)/2:(oh-ih)/2:color=black,"
            "tile=4x2:padding=4:margin=4"
        ),
        "-frames:v", "1", "-q:v", "2", str(output),
    ], check=True)


def parameters(case: dict[str, Any]) -> str:
    values = []
    for name, item in sorted(case.get("physical_parameters", {}).items()):
        if item.get("value") is None:
            continue
        unit = str(item.get("unit", "")).strip()
        values.append(f"`{name}={item['value']}{unit}`")
    return "、".join(values) if values else "—"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    plan = load_json(run_dir / "plan.json")
    cases = {item["case_id"]: item for item in load_jsonl(run_dir / "frozen_cases.jsonl")}
    predictions = {
        item["job_id"]: item for item in load_jsonl(run_dir / "predictions.jsonl")
        if item.get("evaluation_partition") == "train_seen"
    }
    selected_jobs = [
        item for item in plan["jobs"] if item.get("evaluation_partition") == "train_seen"
    ]
    gallery_dir = run_dir / "artifacts" / "train_seen_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 已见训练样本推理展示\n\n",
        f"抽样 seed：`{plan.get('train_preview', {}).get('seed')}`；",
        f"每个场景最多 `{plan.get('train_preview', {}).get('per_scene')}` 条。",
        "本分区只用于直观检查模型是否记住训练样本，不与 ID/OOD1 泛化分数混合。\n\n",
    ]
    completed = 0
    for job in sorted(
        selected_jobs,
        key=lambda item: (
            item["scene_id"], item["case_id"], item["prompt_profile_id"], item["seed"]
        ),
    ):
        prediction = predictions.get(job["job_id"], {})
        case = cases[job["case_id"]]
        prompt_profile_id = job["prompt_profile_id"]
        lines.extend([
            f"## {job['scene_id']} / `{job['case_id']}` / "
            f"`{prompt_profile_id}` / seed {job['seed']}\n\n",
            f"- 物理参数：{parameters(case)}\n",
            f"- 推理状态：`{prediction.get('status', 'missing')}`\n",
        ])
        video_value = prediction.get("video_path")
        if prediction.get("status") != "complete" or not video_value or not Path(video_value).is_file():
            lines.append("\n该 job 没有可用生成视频。\n\n")
            continue
        video = Path(video_value)
        generated_board = gallery_dir / (
            f"{job['case_id']}__{prompt_profile_id}__seed{job['seed']:06d}__generated.jpg"
        )
        storyboard(video, generated_board)
        lines.extend([
            f"- [生成视频]({video.relative_to(run_dir).as_posix()})\n\n",
            "生成视频均匀采样分镜：\n\n",
            f"![generated]({generated_board.relative_to(run_dir).as_posix()})\n\n",
        ])
        reference_value = prediction.get("visual_reference_video") or prediction.get("evaluation_reference_video")
        if reference_value and Path(reference_value).is_file():
            reference = Path(reference_value)
            reference_board = gallery_dir / f"{job['case_id']}__reference.jpg"
            storyboard(reference, reference_board)
            lines.extend([
                "对应训练参考视频均匀采样分镜：\n\n",
                f"![reference]({reference_board.relative_to(run_dir).as_posix()})\n\n",
            ])
        completed += 1
    output = run_dir / "train_seen_gallery.md"
    output.write_text("".join(lines), encoding="utf-8")
    summary = {
        "planned_train_seen_jobs": len(selected_jobs),
        "completed_train_seen_jobs": completed,
        "gallery": str(output),
    }
    (gallery_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
