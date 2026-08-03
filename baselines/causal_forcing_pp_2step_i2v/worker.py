from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--low-memory", action="store_true")
    return parser.parse_args()


def _load_pipeline(args: argparse.Namespace):
    framework_root = Path(args.framework_root).resolve()
    sys.path.insert(0, str(framework_root))

    import torch
    from omegaconf import OmegaConf

    from demo_utils.memory import DynamicSwapInstaller, gpu
    from pipeline import CausalInferencePipeline

    config = OmegaConf.merge(
        OmegaConf.load(framework_root / "configs" / "default_config.yaml"),
        OmegaConf.load(args.config),
    )
    pipeline = CausalInferencePipeline(config, device=torch.device("cuda"))
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    generator = state_dict[args.checkpoint_key]
    try:
        pipeline.generator.load_state_dict(generator)
    except RuntimeError:
        fixed = {}
        for key, value in generator.items():
            if key.startswith("model._fsdp_wrapped_module."):
                key = key.replace(
                    "model._fsdp_wrapped_module.", "model.", 1
                )
            fixed[key] = value
        pipeline.generator.load_state_dict(fixed, strict=False)
    del state_dict, generator
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if args.low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    pipeline.eval()
    return pipeline


def _generate(pipeline, spec: dict) -> dict:
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.io import write_video

    from utils.misc import set_seed

    started = time.monotonic()
    set_seed(int(spec["seed"]))
    width = int(spec["width"])
    height = int(spec["height"])
    latent_frames = int(spec["latent_frames"])
    expected_frames = 1 + 4 * (latent_frames - 1)
    if expected_frames != int(spec["num_frames"]):
        raise ValueError(
            f"latent/pixel frame mismatch: {latent_frames} -> "
            f"{expected_frames}, expected {spec['num_frames']}"
        )
    image = Image.open(spec["first_frame"]).convert("RGB")
    if image.size != (width, height):
        raise ValueError(
            "canonical conditioning canvas changed before the model "
            f"boundary: image={image.size}, expected={(width, height)}"
        )
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(image)
    pixel = (
        tensor.unsqueeze(0)
        .unsqueeze(2)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    initial_latent = pipeline.vae.encode_to_latent(pixel).to(
        device="cuda", dtype=torch.bfloat16
    )
    noise = torch.randn(
        [
            1,
            latent_frames - 1,
            16,
            height // 8,
            width // 8,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        video = pipeline.inference(
            noise=noise,
            text_prompts=[spec["prompt"]],
            initial_latent=initial_latent,
            return_latents=False,
        )
    frames = (
        video[0]
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    if frames.shape[0] != int(spec["num_frames"]):
        raise RuntimeError(
            f"model returned {frames.shape[0]} frames, expected "
            f"{spec['num_frames']}"
        )
    output = Path(spec["output_video"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(output),
        frames.permute(0, 2, 3, 1),
        fps=int(spec["fps"]),
        video_codec="h264",
        options={"crf": "18"},
    )
    pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()
    return {
        "return_code": 0,
        "elapsed_seconds": time.monotonic() - started,
        "generated_frames": int(frames.shape[0]),
        "generated_width": int(frames.shape[-1]),
        "generated_height": int(frames.shape[-2]),
        "generated_fps": int(spec["fps"]),
    }


def main() -> int:
    args = _parse_args()
    with Path(args.jobs).open(encoding="utf-8") as stream:
        jobs = json.load(stream)["jobs"]
    results: dict[str, dict] = {}
    pipeline = _load_pipeline(args)
    for spec in jobs:
        try:
            results[spec["job_id"]] = _generate(pipeline, spec)
        except Exception as exc:
            traceback.print_exc()
            results[spec["job_id"]] = {
                "return_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
    output = Path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"jobs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if all(
        item["return_code"] == 0 for item in results.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
