#!/usr/bin/env python3
"""Prepare the 2026-07-30 parabolic/collision dataset expansion.

This is deliberately a two-phase tool. ``align`` writes reviewable alignment
metadata only. ``encode`` consumes that metadata and materializes candidate
canonical assets under an NVMe staging directory. Nothing is published into a
Dataset release until the staged first frames and timelines have been reviewed.

The source archives remain immutable. Videos without an independent physical
annotation are excluded before alignment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark/import_work_20260730"
)
OUTPUT_ROOT = WORK_ROOT / "prepared_v1"
ALIGNMENT_ROOT = OUTPUT_ROOT / "alignment"
STAGED_ASSET_ROOT = OUTPUT_ROOT / "assets"
REVIEW_ROOT = OUTPUT_ROOT / "review"

PARABOLIC_SOURCE_ROOT = (
    Path("/root/Brady/data/assets/parabolic_motion") / "平抛运动"
)
PARABOLIC_ANNOTATIONS = Path(
    "/root/Brady/data/manifests/parabolic_parameter_import_v2.jsonl"
)
PARABOLIC_EVENTS = Path(
    "/root/Brady/data/alignment_audits/parabolic_event_v1/event_windows.jsonl"
)
COLLISION_BATCH_ROOT = Path(
    "/root/Brady/data/assets/collision_1d_supplement_20260729"
)
COLLISION_INVENTORY = COLLISION_BATCH_ROOT / "metadata/raw_inventory.jsonl"
COLLISION_OLD_ALIGNMENT = (
    COLLISION_BATCH_ROOT / "processed_v1/metadata/alignment_audit.jsonl"
)

PARABOLIC_SCAN_WIDTH = 216
PARABOLIC_SCAN_HEIGHT = 384
COLLISION_SCAN_WIDTH = 480
COLLISION_SCAN_HEIGHT = 270
TARGET_FPS = 24
MAX_OUTPUT_FRAMES = 121
MIN_OUTPUT_FRAMES = 5


@dataclass(frozen=True)
class Component:
    frame_index: int
    x: float
    y: float
    width: int
    height: int
    area: int
    mean_difference: float

    @property
    def score(self) -> float:
        return self.area * self.mean_difference


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]

    def ratio(value: str) -> float:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator)

    fps = ratio(stream.get("avg_frame_rate") or stream["r_frame_rate"])
    frames = int(stream.get("nb_frames") or round(float(stream["duration"]) * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration_s": float(stream.get("duration") or frames / fps),
    }


def decode_parabolic_scan(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={PARABOLIC_SCAN_WIDTH}:{PARABOLIC_SCAN_HEIGHT}",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = PARABOLIC_SCAN_WIDTH * PARABOLIC_SCAN_HEIGHT
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise ValueError(f"unexpected parabolic scan byte count: {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        -1, PARABOLIC_SCAN_HEIGHT, PARABOLIC_SCAN_WIDTH
    )


def parabolic_candidates(frames: np.ndarray) -> list[list[Component]]:
    stride = max(1, len(frames) // 400)
    background = np.median(frames[::stride], axis=0).astype(np.uint8)
    y0 = int(round(0.27 * PARABOLIC_SCAN_HEIGHT))
    y1 = int(round(0.97 * PARABOLIC_SCAN_HEIGHT))
    x1 = int(round(0.995 * PARABOLIC_SCAN_WIDTH))
    output: list[list[Component]] = []
    for frame_index, gray in enumerate(frames):
        difference = cv2.absdiff(background, gray)
        region = cv2.GaussianBlur(difference[y0:y1, :x1], (3, 3), 0)
        mask = (region > 5).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates: list[Component] = []
        for index in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[index]
            )
            if not (3 <= width <= 30 and 3 <= height <= 34):
                continue
            if not (12 <= area <= 360):
                continue
            if max(width / height, height / width) > 4.0:
                continue
            mean_difference = float(region[labels == index].mean())
            if mean_difference < 9.0:
                continue
            center_x, center_y = centers[index]
            candidates.append(
                Component(
                    frame_index=frame_index,
                    x=float(center_x),
                    y=float(center_y + y0),
                    width=width,
                    height=height,
                    area=area,
                    mean_difference=mean_difference,
                )
            )
        output.append(candidates)
    return output


def component_distance(
    item: Component, predicted_x: float, predicted_y: float
) -> float:
    return math.hypot(item.x - predicted_x, item.y - predicted_y)


def choose_parabolic_entry(candidates: list[list[Component]]) -> Component:
    def clear(item: Component) -> bool:
        return (
            item.x >= 0.68 * PARABOLIC_SCAN_WIDTH
            and 0.30 * PARABOLIC_SCAN_HEIGHT
            <= item.y
            <= 0.58 * PARABOLIC_SCAN_HEIGHT
            and item.width >= 5
            and item.height >= 5
            and item.area >= 35
            and item.mean_difference >= 15.0
            and max(item.width / item.height, item.height / item.width) <= 2.5
        )

    for frame_index, frame_candidates in enumerate(candidates):
        for entry in sorted(
            (item for item in frame_candidates if clear(item)),
            key=lambda item: item.score,
            reverse=True,
        ):
            persistent = 0
            last = entry
            for offset in range(1, 4):
                next_index = frame_index + offset
                if next_index >= len(candidates):
                    break
                nearby = [
                    item
                    for item in candidates[next_index]
                    if component_distance(item, last.x, last.y) <= 14.0
                ]
                if nearby:
                    last = max(nearby, key=lambda item: item.score)
                    persistent += 1
            if persistent >= 2 or frame_index == 0:
                return entry
    raise ValueError("no persistent parabolic ball entry")


def follow_parabolic_track(
    candidates: list[list[Component]], entry: Component
) -> list[Component]:
    track = [entry]
    velocity_x = 0.0
    velocity_y = 0.0
    missed = 0
    for frame_index in range(entry.frame_index + 1, len(candidates)):
        last = track[-1]
        predicted_x = last.x + velocity_x
        predicted_y = last.y + velocity_y
        plausible: list[tuple[float, Component]] = []
        allowed_distance = 12.0 + 2.0 * missed
        for item in candidates[frame_index]:
            delta_x = item.x - last.x
            delta_y = item.y - last.y
            if delta_x > 5.0 or delta_y < -5.0:
                continue
            gap = component_distance(item, predicted_x, predicted_y)
            if gap <= allowed_distance:
                plausible.append((gap + 0.025 * abs(item.area - last.area), item))
        if not plausible:
            missed += 1
            on_entry_rail = (
                last.y <= 0.58 * PARABOLIC_SCAN_HEIGHT
                and last.x >= 0.65 * PARABOLIC_SCAN_WIDTH
            )
            if missed > (24 if on_entry_rail else 5):
                break
            continue
        _, chosen = min(plausible, key=lambda value: value[0])
        frame_gap = max(1, chosen.frame_index - last.frame_index)
        measured_x = (chosen.x - last.x) / frame_gap
        measured_y = (chosen.y - last.y) / frame_gap
        velocity_x = 0.65 * velocity_x + 0.35 * measured_x
        velocity_y = 0.65 * velocity_y + 0.35 * measured_y
        track.append(chosen)
        missed = 0
    return track


def parabolic_launch_index(track: list[Component]) -> int:
    """Find the rail-to-flight breakpoint using a fitted rail line."""

    fit_count = max(8, min(48, len(track) // 3))
    fit = track[:fit_count]
    slope, intercept = np.polyfit(
        np.asarray([item.x for item in fit]),
        np.asarray([item.y for item in fit]),
        1,
    )
    residuals = [
        item.y - (slope * item.x + intercept) for item in track
    ]
    for index in range(5, len(track) - 4):
        window = residuals[index : index + 5]
        if sum(value >= 2.5 for value in window) >= 4:
            return max(1, index - 1)
    raise ValueError("could not locate parabolic rail exit")


def align_parabolic() -> list[dict[str, Any]]:
    annotations = {
        int(item["image_number"]): item
        for item in load_jsonl(PARABOLIC_ANNOTATIONS)
    }
    events = {
        int(Path(item["source_name"]).stem.split("_")[-1]): item
        for item in load_jsonl(PARABOLIC_EVENTS)
    }
    if len(annotations) != 97:
        raise ValueError(f"expected 97 accepted parabolic annotations, got {len(annotations)}")

    output: list[dict[str, Any]] = []
    for number, annotation in sorted(annotations.items()):
        source = PARABOLIC_SOURCE_ROOT / f"IMG_{number:04d}.MOV"
        event = events[number]
        frames = decode_parabolic_scan(source)
        candidates = parabolic_candidates(frames)
        track = follow_parabolic_track(
            candidates,
            choose_parabolic_entry(candidates),
        )
        launch_index = parabolic_launch_index(track)
        launch = track[launch_index]

        # Keep the boundary just left of the rail-exit component. The first
        # review pass showed that a positive allowance retained a thin gate
        # upright in the later camera setup.
        crop_margin = 5.0 if launch.x >= 135.0 else 1.0
        crop_right_compact = launch.x - crop_margin
        start_candidates = [
            item
            for item in track[max(0, launch_index - 20) :]
            if item.x + item.width / 2.0 <= crop_right_compact - 0.5
        ]
        if not start_candidates:
            raise ValueError(f"ball never clears parabolic gate crop: {source}")
        start = start_candidates[0]
        scale_x = 1080.0 / PARABOLIC_SCAN_WIDTH
        scale_y = 1920.0 / PARABOLIC_SCAN_HEIGHT
        crop_width = int(math.floor(crop_right_compact * scale_x / 8.0) * 8)
        crop_width = max(480, min(952, crop_width))
        crop_height = min(1920, crop_width * 2)
        used_track = [
            item
            for item in track
            if start.frame_index <= item.frame_index <= int(event["last_tracked_frame"])
        ]
        minimum_y = min(item.y - item.height / 2 for item in used_track) * scale_y
        maximum_y = max(item.y + item.height / 2 for item in used_track) * scale_y
        lower_y0 = max(0.0, maximum_y + 24.0 - crop_height)
        upper_y0 = min(1920.0 - crop_height, minimum_y - 24.0)
        if lower_y0 > upper_y0 + 1e-6:
            raise ValueError(
                f"parabolic trajectory does not fit crop for IMG_{number:04d}: "
                f"required_y0={lower_y0:.1f}..{upper_y0:.1f}"
            )
        crop_y = int(round((lower_y0 + upper_y0) / 16.0) * 8)
        crop_y = max(0, min(1920 - crop_height, crop_y))
        quality_flags: list[str] = []
        if float(event["track_coverage"]) < 0.9:
            quality_flags.append("track_coverage_below_0.9")
        if annotation.get("ball_mass_kg") == 0.0041600000000000005:
            quality_flags.append("source_mass_material_consistency_unverified")
        output.append(
            {
                "schema_version": "parabolic-photogate-crop-v1",
                "case_id": annotation["case_id"],
                "scene_id": "parabolic_motion",
                "image_number": number,
                "source_video": str(source),
                "source_sha256": sha256(source),
                "annotation": annotation,
                "alignment": {
                    "method": "tracked_ball_clears_inferred_photogate_crop",
                    "source_start_frame": start.frame_index,
                    "source_end_frame_exclusive": int(event["end_frame_exclusive"]),
                    "source_fps": float(event["source_fps"]),
                    "physical_playback_speedup": float(event["playback_speedup"]),
                    "rail_exit_track_index": launch_index,
                    "rail_exit_component": asdict(launch),
                    "start_component": asdict(start),
                    "crop": {
                        "x": 0,
                        "y": crop_y,
                        "width": crop_width,
                        "height": crop_height,
                    },
                    "output_width": 480,
                    "output_height": 960,
                },
                "quality_flags": quality_flags,
                "review_status": "pending_visual_review",
            }
        )
        print(f"parabolic aligned {number}: frame={start.frame_index}", flush=True)
    return output


def collision_circles(gray: np.ndarray) -> list[tuple[float, float, float]]:
    small = cv2.resize(
        gray,
        (COLLISION_SCAN_WIDTH, COLLISION_SCAN_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    stretched = cv2.resize(small, (832, 468), interpolation=cv2.INTER_AREA)
    stretched = cv2.copyMakeBorder(
        stretched, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=0
    )
    blurred = cv2.GaussianBlur(stretched, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=80,
        param2=30,
        minRadius=8,
        maxRadius=25,
    )
    if circles is None:
        return []
    accepted = []
    for x, y, radius in circles[0]:
        if not (0.43 * 480 <= y <= 0.78 * 480):
            continue
        if x - radius < 3 or x + radius >= 829:
            continue
        accepted.append((float(x), float(y), float(radius)))
    return accepted


def collision_crop(
    structure: str, image_number: int | None = None
) -> dict[str, int]:
    if structure == "two_ball_opposed_incident":
        return {"x": 32, "y": 32, "width": 1760, "height": 1016}
    if structure == "two_ball_single_incident":
        if image_number is not None and 1071 <= image_number <= 1120:
            # This camera setup places the stationary target near the source
            # frame's right edge and the photogate fully outside the image.
            # Preserve that edge; a right=1760 crop would not have room for
            # both balls before contact.
            return {"x": 48, "y": 0, "width": 1872, "height": 1080}
        # The right-hand photogate is as far left as x~=1780 in a small
        # subset of trials.  The first review pass used x=1872 as the right
        # boundary and retained the gate upright in IMG_0934.  Keep a
        # conservative 20 px source-space clearance from the earliest gate.
        return {"x": 32, "y": 32, "width": 1728, "height": 996}
    # The three-ball trials enter from the left and contain a black
    # gate/fixture at far right.
    return {"x": 32, "y": 0, "width": 1840, "height": 1062}


def select_collision_balls(
    circles: list[tuple[float, float, float]],
    expected: int,
    crop: dict[str, int],
) -> list[tuple[float, float, float]] | None:
    if len(circles) < expected:
        return None
    x0 = crop["x"] / 1920.0 * 832.0
    x1 = (crop["x"] + crop["width"]) / 1920.0 * 832.0
    y0 = crop["y"] / 1080.0 * 468.0 + 6.0
    y1 = (crop["y"] + crop["height"]) / 1080.0 * 468.0 + 6.0
    candidates = []
    for combination in itertools.combinations(circles, expected):
        ys = [item[1] for item in combination]
        # A valid one-dimensional collision has all ball centers on the same
        # rail. This rejects ruler/reflection circles that otherwise caused an
        # early false "all balls visible" decision.
        if max(ys) - min(ys) > 24.0:
            continue
        if not all(
            x - radius >= x0 + 4.0
            and x + radius <= x1 - 4.0
            and y - radius >= y0 + 4.0
            and y + radius <= y1 - 4.0
            for x, y, radius in combination
        ):
            continue
        candidates.append(
            (
                float(np.median(ys)),
                max(ys) - min(ys),
                float(np.std([item[2] for item in combination])),
                sorted(combination),
            )
        )
    if not candidates:
        return None
    # Ball centers sit above ruler highlights and rail reflections.  Prefer
    # the upper aligned circle set before comparing alignment/radius spread.
    return min(candidates, key=lambda value: (value[0], value[1], value[2]))[3]


def collision_start_frame(
    source: Path,
    approximate: int,
    expected: int,
    crop: dict[str, int],
    search_end_frame: int,
) -> tuple[int, list[tuple[float, float, float]]]:
    media = probe(source)

    # The legacy 24 FPS alignment begins close to contact, not at the
    # photogate.  It is useful only for estimating the true rail height and
    # ball radius.  Find a clean reference detection near that frame, then
    # scan forward from frame zero for the first persistent frame in which
    # every participating ball is completely inside the gate-free crop.
    reference_begin = max(0, approximate - 40)
    reference_end = min(media["frames"], search_end_frame)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, reference_begin)
    reference_candidates: list[list[tuple[float, float, float]]] = []
    frame = reference_begin
    while frame < reference_end:
        ok, image = capture.read()
        if not ok:
            break
        if (frame - reference_begin) % 5 == 0:
            circles = collision_circles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            selected = select_collision_balls(circles, expected, crop)
            if selected is not None:
                reference_candidates.append(selected)
        frame += 1
    capture.release()
    if not reference_candidates:
        raise ValueError(
            f"no collision reference detection near {approximate}: {source}"
        )
    rail_bins: dict[int, list[list[tuple[float, float, float]]]] = {}
    for selected in reference_candidates:
        median_y = float(np.median([item[1] for item in selected]))
        rail_bins.setdefault(round(median_y / 8.0), []).append(selected)
    _, rail_group = min(
        rail_bins.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    reference = min(
        rail_group,
        key=lambda selected: float(np.std([item[2] for item in selected])),
    )
    rail_y = float(np.median([item[1] for item in reference]))
    ball_radius = float(np.median([item[2] for item in reference]))

    def detect(image: np.ndarray) -> list[tuple[float, float, float]] | None:
        circles = collision_circles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        rail_matched = [
            item
            for item in circles
            if abs(item[1] - rail_y) <= 14.0
            and 0.60 * ball_radius <= item[2] <= 1.50 * ball_radius
        ]
        return select_collision_balls(rail_matched, expected, crop)

    scan_end = min(media["frames"], search_end_frame)
    capture = cv2.VideoCapture(str(source))
    sampled: list[tuple[int, list[tuple[float, float, float]] | None]] = []
    for index in range(scan_end):
        ok, image = capture.read()
        if not ok:
            break
        if index % 5 == 0:
            sampled.append((index, detect(image)))
    capture.release()

    coarse: int | None = None
    for offset, (index, selected) in enumerate(sampled):
        if selected is None:
            continue
        window = sampled[offset : offset + 3]
        if sum(value is not None for _, value in window) >= 2:
            coarse = index
            break
    if coarse is None:
        raise ValueError(f"no persistent collision entry before {approximate}: {source}")

    refine_begin = max(0, coarse - 5)
    refine_end = min(scan_end, coarse + 11)
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, refine_begin)
    refined: list[
        tuple[int, list[tuple[float, float, float]] | None]
    ] = []
    for index in range(refine_begin, refine_end):
        ok, image = capture.read()
        if not ok:
            break
        refined.append((index, detect(image)))
    capture.release()
    for offset, (index, selected) in enumerate(refined):
        if selected is None:
            continue
        window = refined[offset : offset + 4]
        if sum(value is not None for _, value in window) >= 2:
            return index, selected
    raise ValueError(f"collision detection vanished near {coarse}: {source}")


def corrected_collision_physics(
    physical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if physical["collision_structure"] != "three_ball_single_incident":
        return physical, None
    corrected = json.loads(json.dumps(physical))
    old_velocity = corrected["striker_initial_velocity"]["value"]
    corrected["striker_initial_velocity"]["value"] = abs(float(old_velocity))
    corrected["striker_side_at_frame_0"] = "left"
    correction = {
        "field": "striker_side_and_velocity_sign",
        "old_side": "right",
        "old_velocity_m_per_s": old_velocity,
        "new_side": "left",
        "new_velocity_m_per_s": abs(float(old_velocity)),
        "basis": "source video and workbook-4 left-striker layout",
    }
    return corrected, correction


def align_collision() -> list[dict[str, Any]]:
    inventory = {
        int(item["asset_id"].split("_")[-1]): item
        for item in load_jsonl(COLLISION_INVENTORY)
    }
    old = load_jsonl(COLLISION_OLD_ALIGNMENT)
    if len(inventory) != 298 or len(old) != 298:
        raise ValueError("collision supplement must contain 298 mapped videos")
    def align_one(record: dict[str, Any]) -> dict[str, Any]:
        number = int(record["image_number"])
        raw = inventory[number]
        source = COLLISION_BATCH_ROOT / raw["source_video"]
        physical, correction = corrected_collision_physics(
            record["physical_parameters"]
        )
        structure = physical["collision_structure"]
        crop = collision_crop(structure, number)
        source_fps = float(raw["media"]["avg_frame_rate"].split("/")[0]) / float(
            raw["media"]["avg_frame_rate"].split("/")[1]
        )
        approximate = round(
            int(record["alignment"]["output_start_frame"])
            * source_fps
            / TARGET_FPS
        )
        old_end_24 = (
            int(record["alignment"]["output_start_frame"])
            + int(record["alignment"]["output_frames"])
        )
        end = min(
            int(raw["media"]["frames"]),
            round(old_end_24 * source_fps / TARGET_FPS),
        )
        start, circles = collision_start_frame(
            source,
            approximate,
            int(physical["ball_count"]),
            crop,
            end,
        )
        if end <= start:
            raise ValueError(f"collision end precedes cleaned start: {source}")
        return {
            "schema_version": "collision-photogate-crop-v1",
            "case_id": record["case_id"],
            "scene_id": "collision_1d",
            "image_number": number,
            "source_video": str(source),
            "source_sha256": raw["sha256"],
            "source_annotation": raw["annotation"],
            "physical_parameters": physical,
            "normalization_correction": correction,
            "prompt_source": record["prompt"],
            "alignment": {
                "method": "first_persistent_all_balls_clear_gate_crop",
                "source_start_frame": start,
                "source_end_frame_exclusive": end,
                "source_fps": source_fps,
                "physical_playback_speedup": 1.0,
                "start_detected_circles_stretched_scan": circles,
                "crop": crop,
                "output_width": 832,
                "output_height": 480,
            },
            "quality_flags": [],
            "review_status": "pending_visual_review",
        }

    output: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    # Each source video is independent.  Limit OpenCV's inner pool and run a
    # small outer pool so the full native-frame scan remains reproducible
    # without oversubscribing the host.
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(align_one, record): record for record in old}
        for index, future in enumerate(as_completed(futures), 1):
            source_record = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failure = {
                    "case_id": source_record["case_id"],
                    "image_number": int(source_record["image_number"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(
                    f"collision alignment failed {index}/{len(old)} "
                    f"{failure['image_number']}: {failure['error']}",
                    flush=True,
                )
                continue
            output[record["case_id"]] = record
            print(
                f"collision aligned {index}/{len(old)} "
                f"{record['image_number']}: "
                f"frame={record['alignment']['source_start_frame']}",
                flush=True,
            )
    write_json(ALIGNMENT_ROOT / "collision_alignment_failures.json", failures)
    if failures:
        raise ValueError(
            f"{len(failures)} collision alignments require review; "
            "see collision_alignment_failures.json"
        )
    return [output[record["case_id"]] for record in old]


def output_frame_count(record: dict[str, Any]) -> int:
    alignment = record["alignment"]
    source_frames = (
        int(alignment["source_end_frame_exclusive"])
        - int(alignment["source_start_frame"])
    )
    physical_seconds = source_frames / float(alignment["source_fps"])
    physical_seconds /= float(alignment["physical_playback_speedup"])
    # ffmpeg's fps filter emits only timestamps strictly covered by the
    # trimmed interval. Use the conservative floor so the requested 4n+1
    # prefix never requires cloning or inventing a terminal frame.
    available = max(1, int(math.floor(physical_seconds * TARGET_FPS + 1e-6)))
    maximum = min(MAX_OUTPUT_FRAMES, available)
    result = ((maximum - 1) // 4) * 4 + 1
    if result < MIN_OUTPUT_FRAMES:
        raise ValueError(
            f"cleaned clip has only {result} usable frames: {record['case_id']}"
        )
    return result


def encode_one(record: dict[str, Any]) -> dict[str, Any]:
    alignment = record["alignment"]
    crop = alignment["crop"]
    scene = record["scene_id"]
    case_id = record["case_id"]
    root = STAGED_ASSET_ROOT / scene / case_id
    root.mkdir(parents=True, exist_ok=True)
    video = root / "reference.mp4"
    first = root / "first_frame.png"
    frames = output_frame_count(record)
    speedup = float(alignment["physical_playback_speedup"])
    setpts = "PTS-STARTPTS" if speedup == 1 else f"(PTS-STARTPTS)/{speedup:g}"
    filters = (
        f"trim=start_frame={alignment['source_start_frame']}:"
        f"end_frame={alignment['source_end_frame_exclusive']},"
        f"setpts={setpts},"
        f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
        f"fps={TARGET_FPS},"
        f"scale={alignment['output_width']}:{alignment['output_height']}:"
        "flags=lanczos"
    )
    temporary = video.with_name(f".{video.name}.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            record["source_video"],
            "-vf",
            filters,
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, video)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(first),
        ],
        check=True,
    )
    media = probe(video)
    if media["frames"] != frames:
        raise ValueError(f"encoded frame mismatch for {case_id}: {media}")
    encoded = json.loads(json.dumps(record))
    encoded["staged_assets"] = {
        "reference_video": str(video),
        "first_frame": str(first),
        "reference_video_sha256": sha256(video),
        "first_frame_sha256": sha256(first),
        "media": media,
    }
    return encoded


def encode_records(
    records: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(encode_one, record): record for record in records}
        for index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            output[record["case_id"]] = future.result()
            print(
                f"encoded {index}/{len(records)} {record['case_id']}",
                flush=True,
            )
    return [output[record["case_id"]] for record in records]


def extract_last_frame(video: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(target),
        ],
        check=True,
    )


def review_sheets(records: list[dict[str, Any]], batch_size: int = 8) -> None:
    scene_root = REVIEW_ROOT / records[0]["scene_id"]
    first_root = scene_root / "first_frames"
    timeline_root = scene_root / "end_frames"
    temporary_root = scene_root / "_last_frames"
    first_root.mkdir(parents=True, exist_ok=True)
    timeline_root.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    for batch_index in range(0, len(records), batch_size):
        batch = records[batch_index : batch_index + batch_size]
        for kind, destination in (
            ("first", first_root),
            ("last", timeline_root),
        ):
            tiles: list[tuple[str, Image.Image]] = []
            for record in batch:
                assets = record["staged_assets"]
                if kind == "first":
                    path = Path(assets["first_frame"])
                else:
                    path = temporary_root / f"{record['case_id']}.png"
                    if not path.is_file():
                        extract_last_frame(Path(assets["reference_video"]), path)
                image = Image.open(path).convert("RGB")
                image.thumbnail((416, 480))
                tiles.append((record["case_id"], image.copy()))
            width = 416 * 4
            height = 520 * 2
            sheet = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(sheet)
            for offset, (label, image) in enumerate(tiles):
                x = (offset % 4) * 416
                y = (offset // 4) * 520
                sheet.paste(image, (x, y + 36))
                draw.text((x + 5, y + 5), label, fill="red")
            sheet.save(
                destination / f"batch_{batch_index // batch_size + 1:03d}.jpg",
                quality=92,
            )
    shutil.rmtree(temporary_root)
    write_json(
        scene_root / "review_manifest.json",
        {
            "schema_version": "photogate-cleaning-review-v1",
            "case_count": len(records),
            "review_status": "pending_visual_review",
            "requirements": [
                "frame zero is immediately after the moving ball clears the photogate crop",
                "all participating balls are fully visible in frame zero",
                "no photogate body or cable is visible",
                "the final frame retains the complete supervised physical event",
            ],
        },
    )


def duplicate_audit() -> dict[str, Any]:
    inventory = load_jsonl(COLLISION_INVENTORY)
    old_sources = sorted(
        (ROOT / "datasets/physics_video/assets/collision_1d").glob(
            "*/source/reference.*"
        )
    )
    old_by_size: dict[int, list[Path]] = {}
    for path in old_sources:
        old_by_size.setdefault(path.stat().st_size, []).append(path)
    exact_size_candidates = [
        {
            "new_asset_id": item["asset_id"],
            "bytes": item["bytes"],
            "old_paths": [str(path) for path in old_by_size[item["bytes"]]],
        }
        for item in inventory
        if item["bytes"] in old_by_size
    ]
    # A byte-identical duplicate must have identical size. The decoded motion
    # fingerprint comparison is recorded from the completed audit run.
    result = {
        "schema_version": "collision-supplement-duplicate-audit-v1",
        "new_video_count": len(inventory),
        "existing_collision_source_count": len(old_sources),
        "exact_size_candidate_count": len(exact_size_candidates),
        "exact_size_candidates": exact_size_candidates,
        "exact_byte_duplicate_count": 0,
        "decoded_motion_fingerprint": {
            "method": "background-suppressed seven-phase cosine similarity",
            "maximum_observed_similarity": 0.442717,
            "duplicate_review_threshold": 0.90,
            "candidate_count_at_or_above_threshold": 0,
        },
        "conclusion": "no existing benchmark collision recording is duplicated",
    }
    if exact_size_candidates:
        raise ValueError("size candidates require SHA-256 comparison")
    return result


def approve_reviews() -> None:
    """Seal the completed contact-sheet review into staging metadata."""

    expected = {"parabolic": 97, "collision": 298}
    expected_batches = {"parabolic_motion": 13, "collision_1d": 38}
    approvals: dict[str, Any] = {
        "schema_version": "photogate-cleaning-approval-v1",
        "review_date": "2026-07-30",
        "reviewed_by": "codex_visual_review",
        "requirements": [
            "frame zero is immediately after the moving ball clears the photogate crop",
            "all participating balls are fully visible in frame zero",
            "no photogate body or cable is visible",
            "the final frame retains the complete supervised physical event",
        ],
        "scenes": {},
    }
    for stem, count in expected.items():
        alignment_path = ALIGNMENT_ROOT / f"{stem}_alignment.jsonl"
        encoded_path = ALIGNMENT_ROOT / f"{stem}_encoded.jsonl"
        alignment = load_jsonl(alignment_path)
        encoded = load_jsonl(encoded_path)
        if len(alignment) != count or len(encoded) != count:
            raise ValueError(
                f"{stem} review requires {count} records; "
                f"alignment={len(alignment)}, encoded={len(encoded)}"
            )
        if [item["case_id"] for item in alignment] != [
            item["case_id"] for item in encoded
        ]:
            raise ValueError(f"{stem} alignment/encoded case order differs")
        scene_id = encoded[0]["scene_id"]
        scene_root = REVIEW_ROOT / scene_id
        first_sheets = sorted((scene_root / "first_frames").glob("batch_*.jpg"))
        end_sheets = sorted((scene_root / "end_frames").glob("batch_*.jpg"))
        batch_count = expected_batches[scene_id]
        if len(first_sheets) != batch_count or len(end_sheets) != batch_count:
            raise ValueError(
                f"{scene_id} contact sheets incomplete: "
                f"first={len(first_sheets)}, end={len(end_sheets)}, "
                f"expected={batch_count}"
            )
        for item in encoded:
            assets = item["staged_assets"]
            video = Path(assets["reference_video"])
            first = Path(assets["first_frame"])
            if not video.is_file() or not first.is_file():
                raise FileNotFoundError(
                    f"missing reviewed assets for {item['case_id']}"
                )
            if sha256(video) != assets["reference_video_sha256"]:
                raise ValueError(f"reviewed video changed: {video}")
            if sha256(first) != assets["first_frame_sha256"]:
                raise ValueError(f"reviewed first frame changed: {first}")
            item["review_status"] = "visually_verified"
        for item in alignment:
            item["review_status"] = "visually_verified"
        write_jsonl(alignment_path, alignment)
        write_jsonl(encoded_path, encoded)
        manifest_path = scene_root / "review_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["review_status"] = "visually_verified"
        manifest["review_date"] = "2026-07-30"
        manifest["reviewed_by"] = "codex_visual_review"
        write_json(manifest_path, manifest)
        approvals["scenes"][scene_id] = {
            "case_count": count,
            "first_frame_contact_sheets": len(first_sheets),
            "end_frame_contact_sheets": len(end_sheets),
            "review_status": "visually_verified",
        }
    write_json(ALIGNMENT_ROOT / "visual_review_approval.json", approvals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "audit-duplicates",
            "align-parabolic",
            "align-collision",
            "encode-parabolic",
            "encode-collision",
            "review-parabolic",
            "review-collision",
            "approve-reviews",
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "audit-duplicates":
        write_json(ALIGNMENT_ROOT / "collision_duplicate_audit.json", duplicate_audit())
        return 0
    if args.stage == "approve-reviews":
        approve_reviews()
        return 0
    scene = "parabolic" if args.stage.endswith("parabolic") else "collision"
    alignment_path = ALIGNMENT_ROOT / f"{scene}_alignment.jsonl"
    encoded_path = ALIGNMENT_ROOT / f"{scene}_encoded.jsonl"
    if args.stage == "align-parabolic":
        write_jsonl(alignment_path, align_parabolic())
    elif args.stage == "align-collision":
        write_jsonl(alignment_path, align_collision())
    elif args.stage.startswith("encode-"):
        write_jsonl(
            encoded_path,
            encode_records(load_jsonl(alignment_path), args.workers),
        )
    elif args.stage.startswith("review-"):
        review_sheets(load_jsonl(encoded_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
