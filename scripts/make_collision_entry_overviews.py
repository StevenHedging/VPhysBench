#!/usr/bin/env python3
"""Create coarse frame-index overviews for manual collision-entry bracketing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V1_CASES as DEFAULT_MANIFEST  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--samples", type=int, default=28)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    cases = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["scene_id"] == "collision_1d"
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        assets = case["assets"]
        source_value = assets.get("source_video") or assets["reference_video"]
        source = (manifest.parent / source_value).resolve()
        maximum = args.stride * (args.samples - 1)
        # The crop covers the full left entry region and the complete ball height.
        # Frame labels refer to decoded source-frame indices, not timestamps.
        vf = (
            f"select='lte(n\\,{maximum})*not(mod(n\\,{args.stride}))',"
            "crop=900:420:0:620,scale=360:168:flags=lanczos,"
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
            "text='frame %{n}':x=8:y=8:fontsize=22:fontcolor=white:"
            "box=1:boxcolor=black@0.65,"
            f"tile=7x4:nb_frames={args.samples}:padding=2:margin=2"
        )
        destination = args.output / f"{case['case_id']}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source),
                "-vf", vf, "-frames:v", "1", "-q:v", "2", str(destination),
            ],
            check=True,
        )
        print(f"[{index:02d}/{len(cases)}] {destination.name}", flush=True)


if __name__ == "__main__":
    main()
