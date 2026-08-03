#!/usr/bin/env python3
"""Restore source timing after the required dataset-side spatial/event trims.

Dataset media must not be adapted to a model's FPS or frame-count contract.
This script:

* preserves every source frame in an approved [start, end) event window;
* preserves source timestamps/FPS while retaining the approved crop and scale;
* restores the original slow-motion files for free fall without re-encoding;
* records any encoded-to-physical speed factor for baseline-side adaptation;
* refreshes first frames, mapping hashes, and a machine-readable timing audit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "physics_video"
RELEASE_ROOT = DATA_ROOT / "releases" / "5.1.0"
CASES_PATH = RELEASE_ROOT / "cases.jsonl"
MAPPING_PATH = RELEASE_ROOT / "asset_directory_mapping.json"
MIGRATION_AUDIT_PATH = RELEASE_ROOT / "migration_audit.json"
PARABOLIC_SOURCE_AUDIT = (
    DATA_ROOT
    / "provenance/alignment/photogate_cleaning_20260730_v1"
    / "parabolic_encoded.jsonl"
)
TIMING_AUDIT_DIR = (
    DATA_ROOT / "provenance/alignment/native_timing_20260731_v1"
)
TIMING_AUDIT_PATH = TIMING_AUDIT_DIR / "audit.jsonl"
TIMING_AUDIT_REFERENCE = (
    "provenance/alignment/native_timing_20260731_v1/audit.jsonl"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.native-timing.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate(value: str) -> float:
    numerator, denominator = value.split("/")
    return float(numerator) / float(denominator)


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,"
                "duration,time_base"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    frames = stream.get("nb_frames")
    if frames in {None, "N/A"}:
        raise ValueError(f"video does not declare nb_frames: {path}")
    return {
        "avg_frame_rate": stream["avg_frame_rate"],
        "fps": rate(stream["avg_frame_rate"]),
        "nominal_frame_rate": stream["r_frame_rate"],
        "nominal_fps": rate(stream["r_frame_rate"]),
        "frames": int(frames),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": float(stream["duration"]),
        "time_base": stream["time_base"],
    }


def extract_first_frame(video: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.native-timing.tmp{destination.suffix}"
    )
    if temporary.exists():
        temporary.unlink()
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
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, destination)


def restore_trimmed_case(case: dict[str, Any]) -> dict[str, Any]:
    alignment = case["alignment"]
    source = DATA_ROOT / case["assets"]["source_video"]
    destination = DATA_ROOT / case["assets"]["reference_video"]
    first_frame = DATA_ROOT / case["assets"]["first_frame"]
    crop = alignment["source_crop"]
    output = alignment["output_size"]
    start = int(alignment["source_start_frame"])
    end = int(alignment["source_end_frame_exclusive"])
    expected_frames = end - start
    if expected_frames < 1:
        raise ValueError(f"invalid trim window: {case['case_id']}")

    before = probe(destination)
    source_probe = probe(source)
    temporary = destination.with_name(
        f".{destination.name}.native-timing.tmp.mp4"
    )
    if temporary.exists():
        temporary.unlink()
    filters = (
        f"trim=start_frame={start}:end_frame={end},"
        "setpts=PTS-STARTPTS,"
        f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
        f"scale={output['width']}:{output['height']}:flags=lanczos"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            filters,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "0",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    after = probe(temporary)
    if after["frames"] != expected_frames:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"frame loss in {case['case_id']}: "
            f"{after['frames']} != {expected_frames}"
        )
    if after["nominal_frame_rate"] != source_probe["nominal_frame_rate"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"nominal FPS changed in {case['case_id']}: "
            f"{source_probe['nominal_frame_rate']} -> "
            f"{after['nominal_frame_rate']}"
        )
    os.replace(temporary, destination)
    extract_first_frame(destination, first_frame)
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "kind": "trim_crop_scale_preserve_native_timing",
        "source_asset": str(source.relative_to(DATA_ROOT)),
        "reference_asset": str(destination.relative_to(DATA_ROOT)),
        "first_frame_asset": str(first_frame.relative_to(DATA_ROOT)),
        "source_probe": source_probe,
        "previous_reference_probe": before,
        "reference_probe": after,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "expected_reference_frames": expected_frames,
        "frame_count_verified": True,
        "nominal_frame_rate_verified": True,
    }


def restore_freefall_source(case: dict[str, Any]) -> dict[str, Any]:
    source = DATA_ROOT / case["assets"]["source_video"]
    destination = DATA_ROOT / case["assets"]["reference_video"]
    first_frame = DATA_ROOT / case["assets"]["first_frame"]
    before = probe(destination)
    source_probe = probe(source)
    if not os.path.samefile(source, destination):
        temporary = destination.with_name(
            f".{destination.name}.native-timing.tmp"
        )
        temporary.unlink(missing_ok=True)
        os.link(source, temporary)
        os.replace(temporary, destination)
    after = probe(destination)
    if after != source_probe:
        raise ValueError(f"free-fall source timing was not preserved: {case['case_id']}")
    extract_first_frame(destination, first_frame)
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "kind": "byte_identical_source_timing_hardlink",
        "source_asset": str(source.relative_to(DATA_ROOT)),
        "reference_asset": str(destination.relative_to(DATA_ROOT)),
        "first_frame_asset": str(first_frame.relative_to(DATA_ROOT)),
        "source_probe": source_probe,
        "previous_reference_probe": before,
        "reference_probe": after,
        "expected_reference_frames": source_probe["frames"],
        "frame_count_verified": True,
        "nominal_frame_rate_verified": True,
        "same_inode_as_source": os.path.samefile(source, destination),
    }


def update_case_timing(
    cases: list[dict[str, Any]],
    restored_ids: set[str],
) -> None:
    parabolic_speed = {
        record["case_id"]: float(
            record["alignment"]["physical_playback_speedup"]
        )
        for record in load_jsonl(PARABOLIC_SOURCE_AUDIT)
    }
    for case in cases:
        if case["case_id"] not in restored_ids:
            continue
        if case["scene_id"] == "free_fall":
            speed = 8.0
        elif case["scene_id"] == "parabolic_motion":
            speed = parabolic_speed[case["case_id"]]
        else:
            speed = 1.0
        case["temporal"] = {
            "encoded_to_physical_speed": speed,
            "time_scale": "source_timing",
            "annotation_source": (
                "source frame rate and all frames in the retained source "
                "window are preserved; baseline adapters handle timing"
            ),
        }
        if "alignment" in case:
            alignment = case["alignment"]
            alignment["timing_policy"] = (
                "preserve_source_fps_and_all_trimmed_frames"
            )
            alignment["timing_audit_record"] = TIMING_AUDIT_REFERENCE
            alignment["output_frames"] = (
                int(alignment["source_end_frame_exclusive"])
                - int(alignment["source_start_frame"])
            )
            alignment["output_fps"] = float(alignment["source_fps"])
            if case["scene_id"] == "parabolic_motion":
                alignment["physical_playback_speedup"] = speed


def refresh_mapping_hashes(restored_ids: set[str]) -> None:
    mapping = load_json(MAPPING_PATH)
    touched = 0
    for record in mapping["records"]:
        if record["case_id"] not in restored_ids:
            continue
        touched += 1
        for path_record in record["paths"].values():
            corrected = DATA_ROOT / path_record["corrected"]
            path_record["sha256"] = sha256(corrected)
    if touched != len(restored_ids):
        raise ValueError(
            f"mapping coverage mismatch: {touched} != {len(restored_ids)}"
        )
    write_json(MAPPING_PATH, mapping)


def main() -> int:
    cases = load_jsonl(CASES_PATH)
    trimmed = [
        case
        for case in cases
        if (
            case["scene_id"] == "parabolic_motion"
            or (
                case["scene_id"] == "collision_1d"
                and case["alignment"]["method"]
                == "first_persistent_all_balls_clear_gate_crop"
            )
        )
    ]
    freefall = [
        case for case in cases if case["scene_id"] == "free_fall"
    ]
    if len(trimmed) != 395 or len(freefall) != 11:
        raise ValueError(
            f"unexpected restore scope: trimmed={len(trimmed)} "
            f"freefall={len(freefall)}"
        )

    audit_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(restore_trimmed_case, case): case["case_id"]
            for case in trimmed
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            audit_by_id[record["case_id"]] = record
            if index % 25 == 0 or index == len(futures):
                print(
                    f"native timing encoded {index}/{len(futures)}",
                    flush=True,
                )

    for case in freefall:
        record = restore_freefall_source(case)
        audit_by_id[record["case_id"]] = record

    restored_ids = set(audit_by_id)
    if len(restored_ids) != 406:
        raise ValueError(f"unexpected restored case count: {len(restored_ids)}")
    update_case_timing(cases, restored_ids)
    write_jsonl(CASES_PATH, cases)
    refresh_mapping_hashes(restored_ids)

    TIMING_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        TIMING_AUDIT_PATH,
        [audit_by_id[case_id] for case_id in sorted(audit_by_id)],
    )
    migration = load_json(MIGRATION_AUDIT_PATH)
    migration.update({
        "native_timing_policy": (
            "preserve_source_fps_and_all_frames_in_retained_window"
        ),
        "native_timing_audit": TIMING_AUDIT_REFERENCE,
        "native_timing_case_count": len(restored_ids),
        "native_timing_scene_case_counts": {
            "collision_1d": 298,
            "free_fall": 11,
            "parabolic_motion": 97,
        },
        "model_frame_constraint_in_dataset_assets": False,
    })
    write_json(MIGRATION_AUDIT_PATH, migration)
    print(
        f"restored={len(restored_ids)} "
        f"audit={TIMING_AUDIT_PATH.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
