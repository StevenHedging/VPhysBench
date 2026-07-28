#!/usr/bin/env python3
"""Persistent single-GPU WAN2.2 worker for a shard of frozen benchmark jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def emit(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def main() -> int:
    args = parse_args()
    all_jobs = sorted((args.run_dir / "jobs").glob("*.json"))
    job_paths = all_jobs[args.worker_index :: args.worker_count]
    if not job_paths:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text("", encoding="utf-8")
        return 0
    first = json.loads(job_paths[0].read_text(encoding="utf-8"))
    runtime = first["wan22"]["runtime"]
    project_root = Path(runtime["project_root"])
    python_env_root = Path(sys.executable).resolve().parents[1]
    os.environ.setdefault("CUDA_HOME", str(python_env_root))
    os.environ["PATH"] = f"{python_env_root / 'bin'}:{os.environ.get('PATH', '')}"
    sys.path.insert(0, str(project_root / "vendor" / "DiffSynth-Studio"))
    model_base = Path(runtime.get("model_base", project_root / "models"))
    model_dir = model_base / "Wan-AI" / "Wan2.2-TI2V-5B"
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(model_base))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    shards = sorted(model_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 3:
        raise FileNotFoundError(f"WAN DiT shards incomplete: {len(shards)}/3")
    started = time.monotonic()
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=[str(path) for path in shards]),
            ModelConfig(path=str(model_dir / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(model_dir / "google" / "umt5-xxl")),
    )
    checkpoint = first.get("checkpoint")
    if checkpoint:
        pipe.load_lora(
            pipe.dit, checkpoint,
            alpha=float(first["wan22"]["generation"].get("lora_alpha", 1.0)),
        )
    model_load_seconds = time.monotonic() - started

    args.result.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.result.open("w", encoding="utf-8") as handle:
        for job_path in job_paths:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            generation = job["wan22"]["generation"]
            item_started = time.monotonic()
            try:
                if job.get("checkpoint") != checkpoint:
                    raise ValueError(
                        "all jobs in one persistent worker must use the same frozen checkpoint"
                    )
                first_frame = Image.open(job["model_input"]["first_frame"]).convert("RGB")
                video = pipe(
                    prompt=job["model_input"]["prompt"],
                    negative_prompt=generation.get("negative_prompt", ""),
                    input_image=first_frame,
                    seed=int(job["seed"]),
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    num_frames=int(generation["num_frames"]),
                    num_inference_steps=int(generation.get("num_inference_steps", 50)),
                    cfg_scale=float(generation.get("cfg_scale", 5.0)),
                    tiled=bool(generation.get("tiled", True)),
                )
                output = Path(job["output_video"])
                output.parent.mkdir(parents=True, exist_ok=True)
                save_video(
                    video, str(output), fps=int(generation["fps"]),
                    quality=int(generation.get("quality", 5)),
                )
                emit(handle, {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": job["baseline_id"],
                    "evaluation_partition": job["evaluation_partition"],
                    "status": "complete",
                    "video_path": str(output),
                    "manual_scores": {},
                    "seed": int(job["seed"]),
                    "worker_index": args.worker_index,
                    "model_load_seconds": model_load_seconds,
                    "generation_seconds": time.monotonic() - item_started,
                })
                del video
                torch.cuda.empty_cache()
            except Exception as exc:  # keep the remaining frozen jobs running
                failures += 1
                emit(handle, {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": job["baseline_id"],
                    "evaluation_partition": job["evaluation_partition"],
                    "status": "failed",
                    "video_path": None,
                    "manual_scores": {},
                    "seed": int(job["seed"]),
                    "worker_index": args.worker_index,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "generation_seconds": time.monotonic() - item_started,
                })
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
