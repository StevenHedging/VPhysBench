#!/usr/bin/env python3
"""Detect and materialize frame-exact collision videos whose striker is fully visible at frame 0.

Detection is intentionally conservative: a static first-frame background is compared with
every decoded frame in the track band, and the first left-edge component whose height is
consistent with the annotated striker radius is proposed.  Detection never edits the
manifest.  Apply mode consumes a reviewed candidate/override file and writes new aligned
derivatives while retaining the byte-preserved source MOV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V1_CASES as DEFAULT_MANIFEST  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ANALYSIS_WIDTH = 400
ANALYSIS_HEIGHT = 240
TRACK_CROP_Y = 600
TRACK_CROP_HEIGHT = 480


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,pix_fmt",
        "-of", "json", str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "avg_frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "duration_s": float(stream["duration"]),
    }


def source_path(case: dict[str, Any], manifest: Path) -> Path:
    assets = case["assets"]
    value = assets.get("source_video") or assets["reference_video"]
    return (manifest.parent / value).resolve()


def components_for_frame(frame: np.ndarray, reference: np.ndarray) -> list[dict[str, Any]]:
    delta = frame.astype(np.int16) - reference
    delta = delta - int(np.median(delta))
    difference = np.abs(delta)
    mask = difference > 12
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, _ = ndimage.label(mask)
    candidates = []
    for label, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        ys, xs = slices
        width = xs.stop - xs.start
        height = ys.stop - ys.start
        area = int((labels[slices] == label).sum())
        if not (5 <= height <= 70 and width >= 3 and area >= 20):
            continue
        if xs.start >= int(ANALYSIS_WIDTH * 0.26):
            continue
        fill = area / float(width * height)
        if fill < 0.20:
            continue
        candidates.append({
            "x0": int(xs.start), "x1": int(xs.stop),
            "y0": int(ys.start), "y1": int(ys.stop),
            "width": int(width), "height": int(height),
            "area": area, "fill": fill,
            "mean_difference": float(difference[slices].mean()),
        })
    return candidates


def detect_case(case: dict[str, Any], manifest: Path) -> dict[str, Any]:
    source = source_path(case, manifest)
    info = probe(source)
    if (info["width"], info["height"]) != (1920, 1080):
        raise RuntimeError(f"unsupported collision dimensions for {case['case_id']}: {info}")
    command = [
        "ffmpeg", "-v", "error", "-i", str(source),
        "-vf", (
            f"crop=800:{TRACK_CROP_HEIGHT}:0:{TRACK_CROP_Y},"
            f"scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT},format=gray"
        ),
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = ANALYSIS_WIDTH * ANALYSIS_HEIGHT
    initial = []
    for _ in range(16):
        data = process.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            raise RuntimeError(f"short decode while building background: {case['case_id']}")
        initial.append(np.frombuffer(data, dtype=np.uint8).reshape(ANALYSIS_HEIGHT, ANALYSIS_WIDTH).copy())
    reference = np.median(initial, axis=0).astype(np.int16)
    components: list[list[dict[str, Any]]] = [[] for _ in range(16)]
    while True:
        data = process.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            break
        frame = np.frombuffer(data, dtype=np.uint8).reshape(ANALYSIS_HEIGHT, ANALYSIS_WIDTH)
        components.append(components_for_frame(frame, reference))
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg analysis failed for {case['case_id']}: {return_code}")

    radius = float(case["physical_parameters"]["ball_1_radius"]["value"])
    # Calibrated from the stationary track scale: a 10 mm radius ball is ~31 px tall
    # in the half-resolution analysis strip.
    expected_height = 31.0 * radius / 0.010
    # Select only radius-consistent components in the true entry zone.  Unlike the
    # old tallest-component heuristic, this explicitly follows a component that
    # starts at the left border and then moves monotonically into the image.
    entry_components: list[dict[str, Any] | None] = []
    for frame_components in components:
        plausible = [
            item for item in frame_components
            if item["x0"] < 120
            and expected_height * 0.45 <= item["height"] <= expected_height * 1.65
            and item["width"] <= expected_height * 2.2
        ]
        entry_components.append(
            min(plausible, key=lambda item: (item["x0"], -item["area"]))
            if plausible else None
        )

    border_frames = [
        index for index, item in enumerate(entry_components)
        if item is not None and item["x0"] <= 2
    ]
    if not border_frames:
        raise RuntimeError(f"no left-border entry component for {case['case_id']}")
    first_border = border_frames[0]
    local_heights = [
        item["height"] for item in entry_components[first_border:first_border + 80]
        if item is not None
    ]
    stable_height = float(np.percentile(local_heights, 80))
    minimum_height = max(7, math.floor(stable_height * 0.82))
    minimum_width = max(4, math.floor(stable_height * 0.50))
    qualifying = []
    for index in range(first_border, min(len(entry_components), first_border + 100)):
        component = entry_components[index]
        if component is None or component["x0"] < 3:
            continue
        if component["height"] < minimum_height or component["width"] < minimum_width:
            continue
        following = [item for item in entry_components[index:index + 5] if item is not None]
        if len(following) < 4:
            continue
        centers = [(item["x0"] + item["x1"]) / 2.0 for item in following]
        if sum(b >= a - 2 for a, b in zip(centers, centers[1:])) < 3:
            continue
        qualifying.append(index)
    if not qualifying:
        raise RuntimeError(
            f"no fully-visible candidate for {case['case_id']}; expected_height={expected_height:.2f}"
        )
    candidate = qualifying[0]
    component = entry_components[candidate]
    assert component is not None
    return {
        "case_id": case["case_id"],
        "view_a_split": case["view_a_split"],
        "source": str(source),
        "source_sha256": sha256(source),
        "source_probe": info,
        "candidate_frame": candidate,
        "candidate_time_s": candidate / info["fps"],
        "expected_ball_height_analysis_px": expected_height,
        "stable_ball_height_analysis_px": stable_height,
        "minimum_height_analysis_px": minimum_height,
        "minimum_width_analysis_px": minimum_width,
        "candidate_component": component,
        "frames_decoded": len(components),
        "review_status": "pending_visual_review",
    }


def extract_audit_frames(records: list[dict[str, Any]], output: Path) -> None:
    frame_root = output / "audit_frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    offsets = (-8, -1, 0, 4)
    for record in records:
        case_dir = frame_root / record["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        indexes = [max(0, int(record["candidate_frame"]) + value) for value in offsets]
        expression = "+".join(f"eq(n\\,{value})" for value in indexes)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", record["source"],
            "-vf", f"select='{expression}'", "-vsync", "0",
            str(case_dir / "frame_%02d.png"),
        ], check=True)
        extracted = sorted(case_dir.glob("frame_*.png"))
        for path, frame_index, offset in zip(extracted, indexes, offsets):
            destination = case_dir / f"offset_{offset:+03d}_frame_{frame_index:05d}.png"
            path.rename(destination)

    font = ImageFont.truetype(FONT, 18)
    sheets = output / "audit_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for sheet_index in range(0, len(records), 4):
        selected = records[sheet_index:sheet_index + 4]
        canvas = Image.new("RGB", (4 * 720, len(selected) * 260), (20, 20, 20))
        for row, record in enumerate(selected):
            images = sorted((frame_root / record["case_id"]).glob("*.png"))
            for column, path in enumerate(images):
                image = Image.open(path).convert("RGB")
                # Enlarge the left entry region and track band where clipping is decided.
                image = image.crop((0, 700, 720, 1020)).resize((720, 220), Image.Resampling.LANCZOS)
                cell = Image.new("RGB", (720, 260), (20, 20, 20))
                cell.paste(image, (0, 0))
                label = f"{record['case_id']} | {path.stem}"
                ImageDraw.Draw(cell).text((8, 228), label, font=font, fill="white")
                canvas.paste(cell, (column * 720, row * 260))
        canvas.save(sheets / f"sheet_{sheet_index // 4:02d}.jpg", quality=94)


def detect(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    cases = [case for case in load_jsonl(manifest) if case["scene_id"] == "collision_1d"]
    records = []
    for index, case in enumerate(cases, 1):
        record = detect_case(case, manifest)
        records.append(record)
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} frame={record['candidate_frame']} "
            f"time={record['candidate_time_s']:.6f}s component={record['candidate_component']}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "candidates.json", {
        "schema_version": "1.0",
        "policy": "first full-height left-entry component after static background",
        "manifest": str(manifest),
        "records": records,
    })
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "split", "candidate_frame", "candidate_time_s", "height", "width", "x0"])
        for record in records:
            component = record["candidate_component"]
            writer.writerow([
                record["case_id"], record["view_a_split"], record["candidate_frame"],
                f"{record['candidate_time_s']:.9f}", component["height"],
                component["width"], component["x0"],
            ])
    extract_audit_frames(records, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    detect(build_parser().parse_args())
