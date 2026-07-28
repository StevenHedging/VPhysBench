#!/usr/bin/env python3
"""Persistent single-GPU worker for WAN quantity-embedding jobs."""

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
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    )
    handle.flush()


def main() -> int:
    args = parse_args()
    all_jobs = sorted((args.run_dir / "jobs").glob("*.json"))
    job_paths = all_jobs[args.worker_index :: args.worker_count]
    if not job_paths:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text("", encoding="utf-8")
        return 0
    # Every worker uses the same global first job as its load-time anchor.
    # This prevents different GPU shards from silently choosing different
    # checkpoints, runtimes, model roots, encoder configs, or LoRA scales.
    first = json.loads(all_jobs[0].read_text(encoding="utf-8"))
    runtime = first["wan22"]["runtime"]
    project_root = Path(runtime["project_root"])
    benchmark_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(benchmark_root / "src"))
    sys.path.insert(
        0,
        str(project_root / "vendor" / "DiffSynth-Studio"),
    )
    python_env_root = Path(sys.executable).resolve().parents[1]
    os.environ.setdefault("CUDA_HOME", str(python_env_root))
    os.environ["PATH"] = (
        f"{python_env_root / 'bin'}:{os.environ.get('PATH', '')}"
    )
    model_base = Path(
        runtime.get("model_base", project_root / "models")
    )
    model_dir = model_base / "Wan-AI" / "Wan2.2-TI2V-5B"
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(model_base))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import (
        ModelConfig,
        WanVideoPipeline,
    )
    from diffsynth.utils.data import save_video
    from physbench.baselines.wan22_quantity_model import (
        QuantityEncoder,
        install_quantity_prompt_unit,
        load_verified_combined_quantity_checkpoint,
        quantity_inference_conditioning,
        quantity_pipeline_shared_config,
        quantity_pipeline_shared_fingerprint,
        verify_quantity_checkpoint_manifest,
    )
    from physbench.io import write_json

    shards = sorted(
        model_dir.glob("diffusion_pytorch_model-*-of-*.safetensors")
    )
    if len(shards) != 3:
        raise FileNotFoundError(
            f"WAN DiT shards incomplete: {len(shards)}/3"
        )
    checkpoint = first.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError(
            "quantity-embedding generation requires a combined checkpoint"
        )
    checkpoint_manifest = first.get("checkpoint_manifest")
    if not checkpoint_manifest:
        raise FileNotFoundError(
            "quantity-embedding generation requires a checkpoint manifest"
        )
    first_shared_config = quantity_pipeline_shared_config(first)
    shared_config_fingerprint = quantity_pipeline_shared_fingerprint(first)
    for candidate_path in all_jobs:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if quantity_pipeline_shared_config(candidate) != first_shared_config:
            raise ValueError(
                "persistent inference batch contains inconsistent "
                "pipeline-shared configuration across GPU shards"
            )
    early_checkpoint_verification = verify_quantity_checkpoint_manifest(
        checkpoint,
        checkpoint_manifest,
    )
    checkpoint = early_checkpoint_verification["checkpoint"]
    checkpoint_manifest = early_checkpoint_verification["manifest"]
    started = time.monotonic()
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path=str(
                    model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
                )
            ),
            ModelConfig(path=[str(path) for path in shards]),
            ModelConfig(path=str(model_dir / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(
            path=str(model_dir / "google" / "umt5-xxl")
        ),
    )
    encoder = QuantityEncoder(
        first["wan22"]["quantity_encoder"]
    ).to(device=pipe.device, dtype=torch.float32)
    checkpoint_load = load_verified_combined_quantity_checkpoint(
        pipe,
        encoder,
        checkpoint,
        checkpoint_manifest,
        lora_alpha=float(
            first["wan22"]["generation"].get("lora_alpha", 1.0)
        ),
    )
    checkpoint_verification = checkpoint_load[
        "checkpoint_verification"
    ]
    encoder.eval()
    install_quantity_prompt_unit(pipe, encoder)
    model_load_seconds = time.monotonic() - started

    args.result.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.result.open("w", encoding="utf-8") as handle:
        for job_path in job_paths:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            generation = job["wan22"]["generation"]
            item_started = time.monotonic()
            try:
                job_shared_config = quantity_pipeline_shared_config(job)
                if job_shared_config != first_shared_config:
                    raise ValueError(
                        "persistent worker jobs changed pipeline-shared "
                        "configuration (checkpoint, manifest, runtime/base "
                        "assets, quantity encoder, LoRA alpha, or another "
                        "load-time generation option)"
                    )
                prompt = job["model_input"]["prompt"]
                payload = job["model_input"]["quantity_payload"]
                first_frame = Image.open(
                    job["model_input"]["first_frame"]
                ).convert("RGB")
                with (
                    quantity_inference_conditioning(
                        pipe,
                        payload["quantities"],
                    ),
                    torch.inference_mode(),
                ):
                    video = pipe(
                        prompt=prompt,
                        negative_prompt=generation.get(
                            "negative_prompt",
                            "",
                        ),
                        input_image=first_frame,
                        seed=int(job["seed"]),
                        height=int(generation["height"]),
                        width=int(generation["width"]),
                        num_frames=int(generation["num_frames"]),
                        num_inference_steps=int(
                            generation.get("num_inference_steps", 50)
                        ),
                        cfg_scale=float(
                            generation.get("cfg_scale", 5.0)
                        ),
                        tiled=bool(generation.get("tiled", True)),
                    )
                output = Path(job["output_video"])
                output.parent.mkdir(parents=True, exist_ok=True)
                save_video(
                    video,
                    str(output),
                    fps=int(generation["fps"]),
                    quality=int(generation.get("quality", 5)),
                )
                audit_path = Path(job["quantity_token_audit"])
                write_json(audit_path, {
                    "schema_version": "1.0",
                    "case_id": job["case_id"],
                    "job_id": job["job_id"],
                    "registry_id": payload["registry_id"],
                    "registry_fingerprint": payload[
                        "registry_fingerprint"
                    ],
                    "prompt": prompt,
                    "audited_prompt": job["model_input"][
                        "audited_prompt"
                    ],
                    "quantities": pipe._last_quantity_token_audit,
                    "checkpoint_verification": checkpoint_verification,
                    "load_boundary_verified": checkpoint_load[
                        "load_boundary_verified"
                    ],
                    "checkpoint_load_mode": checkpoint_load[
                        "checkpoint_load_mode"
                    ],
                    "pipeline_shared_config_fingerprint": (
                        shared_config_fingerprint
                    ),
                })
                emit(handle, {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": job["baseline_id"],
                    "evaluation_partition": job[
                        "evaluation_partition"
                    ],
                    "status": "complete",
                    "video_path": str(output),
                    "manual_scores": {},
                    "seed": int(job["seed"]),
                    "worker_index": args.worker_index,
                    "model_load_seconds": model_load_seconds,
                    "generation_seconds": (
                        time.monotonic() - item_started
                    ),
                    "quantity_token_audit": str(audit_path),
                    "quantity_count": len(payload["quantities"]),
                    "checkpoint_sha256": checkpoint_verification[
                        "checkpoint_sha256"
                    ],
                    "checkpoint_size": checkpoint_verification[
                        "checkpoint_size"
                    ],
                    "checkpoint_manifest": checkpoint_verification[
                        "manifest"
                    ],
                    "load_boundary_verified": checkpoint_load[
                        "load_boundary_verified"
                    ],
                    "checkpoint_load_mode": checkpoint_load[
                        "checkpoint_load_mode"
                    ],
                    "pipeline_shared_config_fingerprint": (
                        shared_config_fingerprint
                    ),
                })
                del video
                torch.cuda.empty_cache()
            except Exception as exc:
                failures += 1
                emit(handle, {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": job["baseline_id"],
                    "evaluation_partition": job[
                        "evaluation_partition"
                    ],
                    "status": "failed",
                    "video_path": None,
                    "manual_scores": {},
                    "seed": int(job["seed"]),
                    "worker_index": args.worker_index,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "generation_seconds": (
                        time.monotonic() - item_started
                    ),
                })
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
