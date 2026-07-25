#!/usr/bin/env python3
"""Render adjacent-frame evidence for final collision entry choices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V1_CASES as DEFAULT_MANIFEST  # noqa: E402

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    reviewed = json.loads(args.reviewed.read_text(encoding="utf-8"))["frames"]
    cases = {
        case["case_id"]: case
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and (case := json.loads(line))["scene_id"] == "collision_1d"
    }
    frame_root = args.output / "frames"
    sheet_root = args.output / "sheets"
    frame_root.mkdir(parents=True, exist_ok=True)
    sheet_root.mkdir(parents=True, exist_ok=True)
    offsets = (-2, -1, 0, 1)
    records = []
    for case_id, selected in reviewed.items():
        case = cases[case_id]
        source_value = case["assets"].get("source_video") or case["assets"]["reference_video"]
        source = (manifest.parent / source_value).resolve()
        case_dir = frame_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        indices = sorted(set(max(0, selected + offset) for offset in offsets))
        expression = "+".join(f"eq(n\\,{value})" for value in indices)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", f"select='{expression}'", "-vsync", "0", str(case_dir / "%02d.png"),
        ], check=True)
        extracted = sorted(case_dir.glob("*.png"))
        labeled = []
        for path, frame_index in zip(extracted, indices):
            destination = case_dir / f"source_frame_{frame_index:05d}.png"
            path.rename(destination)
            labeled.append((destination, frame_index))
        records.append((case_id, selected, labeled))

    font = ImageFont.truetype(FONT_PATH, 18)
    for page_start in range(0, len(records), 4):
        page = records[page_start:page_start + 4]
        canvas = Image.new("RGB", (4 * 600, 4 * 410), (18, 18, 18))
        for row, (case_id, selected, labeled) in enumerate(page):
            for column, (path, frame_index) in enumerate(labeled):
                image = Image.open(path).convert("RGB").crop((0, 700, 320, 1020))
                image = image.resize((600, 360), Image.Resampling.NEAREST)
                cell = Image.new("RGB", (600, 410), (18, 18, 18))
                cell.paste(image, (0, 0))
                status = "SELECTED" if frame_index == selected else "context"
                ImageDraw.Draw(cell).text(
                    (8, 368), f"{case_id} | frame {frame_index} | {status}",
                    font=font, fill=(255, 220, 60) if status == "SELECTED" else "white",
                )
                canvas.paste(cell, (column * 600, row * 410))
        canvas.save(sheet_root / f"page_{page_start // 4:02d}.jpg", quality=95)


if __name__ == "__main__":
    main()
