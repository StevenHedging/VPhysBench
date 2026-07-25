#!/usr/bin/env python3
"""Run Task-1 training, loss export, and parallel WAN inference as one supervised workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V1_CASES, V1_VIEW_A  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--task", type=Path,
        default=ROOT / "configs" / "tasks" / "view_a_three_scene_finetune.json",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "configs" / "baselines" / "wan22_ti2v_5b_lora_three_scene_8gpu_buckets.json",
    )
    parser.add_argument("--manifest", type=Path, default=V1_CASES)
    parser.add_argument("--split", type=Path, default=V1_VIEW_A)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--train-preview-per-scene", type=int, default=2)
    parser.add_argument("--train-preview-seed", type=int, default=42)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    run_dir = (args.output_root / args.run_id).resolve()
    status_path = args.output_root / f"{args.run_id}.supervisor.json"
    status = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "started_at": now(),
        "stage": "training_and_staging",
        "commands": {},
    }
    write_status(status_path, status)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    train_command = [
        sys.executable, "-m", "physbench", "run",
        "--task", str(args.task.resolve()),
        "--baseline", str(args.baseline.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--split", str(args.split.resolve()),
        "--scene-config-dir", str(ROOT / "configs" / "scenes"),
        "--metrics", str(ROOT / "configs" / "metrics" / "default.json"),
        "--output-root", str(args.output_root.resolve()),
        "--run-id", args.run_id,
        "--train-preview-per-scene", str(args.train_preview_per_scene),
        "--train-preview-seed", str(args.train_preview_seed),
        "--execute", "--stop-after-training",
    ]
    status["commands"]["training_and_staging"] = train_command
    write_status(status_path, status)
    train_code = subprocess.run(train_command, cwd=ROOT, env=env).returncode
    status["training_and_staging_return_code"] = train_code
    if train_code:
        status.update(stage="failed_training_or_staging", finished_at=now())
        write_status(status_path, status)
        return train_code

    status["stage"] = "loss_export"
    write_status(status_path, status)
    baseline = json.loads((run_dir / "frozen_baseline.json").read_text(encoding="utf-8"))
    model_python = baseline["runtime"]["python"]
    loss_command = [
        model_python, str(ROOT / "scripts" / "plot_wan22_loss.py"), "--run-dir", str(run_dir),
    ]
    status["commands"]["loss_export"] = loss_command
    loss_code = subprocess.run(loss_command, cwd=ROOT).returncode
    status["loss_export_return_code"] = loss_code
    write_status(status_path, status)

    status["stage"] = "parallel_inference"
    inference_command = [
        sys.executable, str(ROOT / "scripts" / "run_wan22_parallel_inference.py"),
        "--run-dir", str(run_dir), "--gpus", args.gpus,
    ]
    status["commands"]["parallel_inference"] = inference_command
    write_status(status_path, status)
    inference_code = subprocess.run(inference_command, cwd=ROOT, env=env).returncode
    status["parallel_inference_return_code"] = inference_code
    status["stage"] = "complete" if inference_code == 0 else "inference_incomplete"
    status["finished_at"] = now()
    write_status(status_path, status)
    return inference_code


if __name__ == "__main__":
    raise SystemExit(main())
