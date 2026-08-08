#!/usr/bin/env python3
"""Generate a static protocol-test video; this is not a model baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _shape(job_spec: Path) -> tuple[int, int, float, int]:
    value = json.loads(job_spec.read_text(encoding="utf-8"))
    output = value["media_contract"]["output"]
    canvas = output["canvas"]
    timeline = output["timeline"]
    frame_count = timeline["frame_count"]
    frames = (
        int(frame_count["value"])
        if frame_count["rule"] == "fixed"
        else int(frame_count["maximum"])
    )
    width = int(canvas["width"])
    height = int(canvas["height"])
    fps = float(timeline["fps"])
    if min(width, height, frames) <= 0 or fps <= 0:
        raise ValueError("job spec contains a non-positive media shape")
    return width, height, fps, frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--job-spec", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required by the protocol fixture")
    image = Path(args.image).resolve(strict=True)
    job_spec = Path(args.job_spec).resolve(strict=True)
    output = Path(args.output).resolve()
    width, height, fps, frames = _shape(job_spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image),
        "-vf",
        f"scale={width}:{height}:flags=neighbor,format=yuv420p",
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return int(completed.returncode)
    print(
        "warning: dummy_i2v_command is not a benchmark baseline; its scores "
        "are meaningless",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
