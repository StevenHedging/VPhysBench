#!/usr/bin/env python3
"""Create full-resolution boundary crops around manually bracketed entry frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "cases.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=40)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    centers = json.loads(args.centers.read_text(encoding="utf-8"))
    cases = {
        case["case_id"]: case
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and (case := json.loads(line))["scene_id"] == "collision_1d"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (case_id, center) in enumerate(centers.items(), 1):
        case = cases[case_id]
        source_value = case["assets"].get("source_video") or case["assets"]["reference_video"]
        source = (manifest.parent / source_value).resolve()
        start = max(0, center - args.radius)
        stop = center + args.radius
        sample_count = (stop - start) // args.stride + 1
        vf = (
            f"select='between(n\\,{start}\\,{stop})*not(mod(n-{start}\\,{args.stride}))',"
            "crop=280:300:0:700,scale=392:420:flags=neighbor,"
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            f"text='source frame %{{eif\\:n*{args.stride}+{start}\\:d}}':"
            "x=8:y=8:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.70,"
            f"tile=7x6:nb_frames={sample_count}:padding=2:margin=2"
        )
        destination = args.output / f"{case_id}.png"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", vf, "-frames:v", "1", str(destination),
        ], check=True)
        print(f"[{index:02d}/{len(centers)}] {destination.name}", flush=True)


if __name__ == "__main__":
    main()
