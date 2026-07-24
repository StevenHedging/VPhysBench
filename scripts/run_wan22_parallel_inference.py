#!/usr/bin/env python3
"""Run staged WAN jobs with one persistent model instance per selected GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import load_json, load_jsonl, write_json, write_jsonl  # noqa: E402
from physbench.runner import reevaluate_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    worker_root = run_dir / "artifacts" / "wan22" / "inference_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, str, subprocess.Popen, object]] = []
    worker_script = ROOT / "scripts" / "wan22_generate_batch.py"
    python = load_json(run_dir / "frozen_baseline.json")["runtime"]["python"]
    for index, gpu in enumerate(gpus):
        result = worker_root / f"worker_{index:02d}.jsonl"
        log = (worker_root / f"worker_{index:02d}.log").open("w", encoding="utf-8")
        command = [
            python, str(worker_script), "--run-dir", str(run_dir),
            "--worker-index", str(index), "--worker-count", str(len(gpus)),
            "--result", str(result),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        processes.append((index, gpu, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    return_codes = {}
    for index, gpu, process, log in processes:
        return_codes[f"worker_{index:02d}_gpu_{gpu}"] = process.wait()
        log.close()

    records = []
    for path in sorted(worker_root.glob("worker_*.jsonl")):
        records.extend(load_jsonl(path))
    records.sort(key=lambda item: item["job_id"])
    planned_jobs = sorted((run_dir / "jobs").glob("*.json"))
    write_jsonl(run_dir / "predictions.jsonl", records)
    write_json(worker_root / "summary.json", {
        "gpu_assignments": gpus,
        "worker_return_codes": return_codes,
        "planned_jobs": len(planned_jobs),
        "recorded_jobs": len(records),
        "completed_jobs": sum(item["status"] == "complete" for item in records),
        "failed_jobs": sum(item["status"] != "complete" for item in records),
    })
    run = load_json(run_dir / "run.json")
    all_complete = len(records) == len(planned_jobs) and all(
        item["status"] == "complete" for item in records
    )
    run["status"] = "complete" if all_complete else "inference_incomplete"
    write_json(run_dir / "run.json", run)
    reevaluate_run(run_dir, ROOT / "configs" / "scenes")
    gallery_return_code = None
    if any(job.get("evaluation_partition") == "train_seen" for job in (
        load_json(path) for path in planned_jobs
    )):
        gallery_return_code = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "build_train_seen_gallery.py"),
            "--run-dir", str(run_dir),
        ]).returncode
        worker_summary = load_json(worker_root / "summary.json")
        worker_summary["train_seen_gallery_return_code"] = gallery_return_code
        write_json(worker_root / "summary.json", worker_summary)
    experiment_summary_return_code = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "build_wan22_run_summary.py"),
        "--run-dir", str(run_dir),
    ]).returncode
    worker_summary = load_json(worker_root / "summary.json")
    worker_summary["experiment_summary_return_code"] = experiment_summary_return_code
    write_json(worker_root / "summary.json", worker_summary)
    print(json.dumps(load_json(worker_root / "summary.json"), ensure_ascii=False, indent=2))
    return 0 if (
        all_complete
        and gallery_return_code in {None, 0}
        and experiment_summary_return_code == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
