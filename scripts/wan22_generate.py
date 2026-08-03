#!/usr/bin/env python3
"""Execute one frozen WAN2.2 TI2V/T2V benchmark generation job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.job.open(encoding="utf-8") as handle:
        job = json.load(handle)
    runtime = job["wan22"]["runtime"]
    generation = job["wan22"]["generation"]
    project_root = Path(runtime["project_root"])
    python_env_root = Path(sys.executable).resolve().parents[1]
    os.environ.setdefault("CUDA_HOME", str(python_env_root))
    os.environ["PATH"] = f"{python_env_root / 'bin'}:{os.environ.get('PATH', '')}"
    vendor = project_root / "vendor" / "DiffSynth-Studio"
    sys.path.insert(0, str(vendor))
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
    required = [
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "Wan2.2_VAE.pth",
        model_dir / "google" / "umt5-xxl" / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or len(shards) != 3:
        raise FileNotFoundError(f"WAN2.2 model incomplete: missing={missing}, shards={len(shards)}/3")

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
    checkpoint = job.get("checkpoint")
    if checkpoint:
        pipe.load_lora(pipe.dit, str(checkpoint), alpha=float(generation.get("lora_alpha", 1.0)))
    first_frame_path = job["model_input"].get("first_frame")
    first_frame = Image.open(first_frame_path).convert("RGB") if first_frame_path else None
    expected_size = (int(generation["width"]), int(generation["height"]))
    if first_frame is not None and first_frame.size != expected_size:
        raise ValueError(
            "WAN conditioning image must already match the native canvas; "
            f"image={first_frame.size}, expected={expected_size}"
        )
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
    save_video(video, str(output), fps=int(generation["fps"]), quality=int(generation.get("quality", 5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
