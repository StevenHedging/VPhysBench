#!/usr/bin/env python3
"""Execute one frozen WAN2.2 quantity-embedding generation job."""

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
    job = json.loads(args.job.read_text(encoding="utf-8"))
    runtime = job["wan22"]["runtime"]
    generation = job["wan22"]["generation"]
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
        load_combined_quantity_checkpoint,
        quantity_inference_conditioning,
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
    checkpoint = job.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError(
            "quantity-embedding generation requires a combined checkpoint"
        )
    checkpoint_manifest = job.get("checkpoint_manifest")
    if not checkpoint_manifest:
        raise FileNotFoundError(
            "quantity-embedding generation requires a checkpoint manifest"
        )
    checkpoint_verification = verify_quantity_checkpoint_manifest(
        checkpoint,
        checkpoint_manifest,
    )
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
        job["wan22"]["quantity_encoder"]
    ).to(device=pipe.device, dtype=torch.float32)
    load_combined_quantity_checkpoint(
        pipe,
        encoder,
        checkpoint,
        lora_alpha=float(generation.get("lora_alpha", 1.0)),
    )
    encoder.eval()
    install_quantity_prompt_unit(pipe, encoder)
    prompt = job["model_input"]["prompt"]
    quantity_payload = job["model_input"]["quantity_payload"]
    first_frame = Image.open(
        job["model_input"]["first_frame"]
    ).convert("RGB")
    with (
        quantity_inference_conditioning(
            pipe,
            quantity_payload["quantities"],
        ),
        torch.inference_mode(),
    ):
        video = pipe(
            prompt=prompt,
            negative_prompt=generation.get("negative_prompt", ""),
            input_image=first_frame,
            seed=int(job["seed"]),
            height=int(generation["height"]),
            width=int(generation["width"]),
            num_frames=int(generation["num_frames"]),
            num_inference_steps=int(
                generation.get("num_inference_steps", 50)
            ),
            cfg_scale=float(generation.get("cfg_scale", 5.0)),
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
    write_json(Path(job["quantity_token_audit"]), {
        "schema_version": "1.0",
        "case_id": job["case_id"],
        "job_id": job["job_id"],
        "registry_id": quantity_payload["registry_id"],
        "registry_fingerprint": quantity_payload[
            "registry_fingerprint"
        ],
        "prompt": prompt,
        "audited_prompt": job["model_input"]["audited_prompt"],
        "quantities": pipe._last_quantity_token_audit,
        "checkpoint_verification": checkpoint_verification,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
