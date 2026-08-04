#!/usr/bin/env python3
"""Materialize visually accepted 2026-08-04 pendulum supplement assets.

This script consumes the staging analysis, frame-accurately removes the first
half-cycle, preserves every remaining source frame at the source nominal FPS,
generates canonical first frames, and writes auditable Case drafts.  It does
not publish a Dataset release; publication remains a separate atomic step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from zipfile import ZipFile

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets/physics_video"
ARCHIVE = Path("/root/Steven/补充_小球单摆实验.zip")
WORKBOOK_MEMBER = "补充_小球单摆实验/钟摆实验.xlsx"
IMPORT_ID = "pendulum_supplement_20260804"
PROMPT = (
    "A pendulum bob starts at a turning point, swings down through the lowest "
    "point to the opposite side, and continues oscillating back and forth about "
    "the fixed pivot."
)
BACKGROUND_IDS = {"白色卡纸": "white_card", "绿色卡纸": "green_card"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
            "nb_frames,duration:stream_tags=rotate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "rotation_deg": int(stream.get("tags", {}).get("rotate", 0)),
        "nominal_frame_rate": stream["r_frame_rate"],
        "average_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream.get("duration", payload["format"]["duration"])),
    }


def _length_mm(value_m: float) -> int:
    return int(round(value_m * 1000.0))


def _angle_id(value_deg: float) -> str:
    if float(value_deg).is_integer():
        return f"{int(value_deg):03d}deg"
    return f"{value_deg:g}deg".replace(".", "p")


def _asset_directory(record: dict[str, Any]) -> str:
    physics = record["normalized_physics"]
    total = _length_mm(physics["pendulum_length"]["value"])
    string = _length_mm(physics["string_length"]["value"])
    radius = _length_mm(physics["bob_radius"]["value"])
    angle = _angle_id(physics["initial_angle"]["value"])
    return (
        f"pendulum_l{total:03d}mm_ls{string:03d}mm_m31p5g_"
        f"r{radius:02d}mm_a{angle}_id{record['source_sha256'][:8]}"
    )


def _case_id(record: dict[str, Any]) -> str:
    physics = record["normalized_physics"]
    total = _length_mm(physics["pendulum_length"]["value"])
    string = _length_mm(physics["string_length"]["value"])
    radius = _length_mm(physics["bob_radius"]["value"])
    angle = _angle_id(physics["initial_angle"]["value"])
    stem = record["annotation"]["source_stem"].lower().replace("_", "")
    return (
        f"pendulum_s3_ltot{total:04d}mm_lrope{string:04d}mm_"
        f"m031p5g_r{radius:03d}mm_a{angle}_{stem}"
    )


def _normalized_physics(record: dict[str, Any]) -> dict[str, Any]:
    physics = json.loads(json.dumps(record["normalized_physics"]))
    for quantity in physics.values():
        quantity["value"] = round(float(quantity["value"]), 6)
    return physics


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _extract_frame(path: Path, frame_index: int, output: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ]
    )


def _raw_rgb_digest(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _psnr(source_image: Path, canonical_image: Path) -> float:
    source = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    canonical = cv2.imread(str(canonical_image), cv2.IMREAD_COLOR)
    if source is None or canonical is None or source.shape != canonical.shape:
        raise ValueError(
            f"cannot compare aligned frames: {source_image}, {canonical_image}"
        )
    error = np.mean((source.astype(np.float64) - canonical.astype(np.float64)) ** 2)
    return math.inf if error == 0.0 else float(10.0 * math.log10(255.0**2 / error))


def _materialize_one(arguments: tuple[dict[str, Any], str]) -> dict[str, Any]:
    record, asset_root_value = arguments
    asset_root = Path(asset_root_value)
    source = Path(record["source_path"])
    directory = _asset_directory(record)
    canonical_dir = asset_root / directory / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    reference = canonical_dir / "reference.mp4"
    first_frame = canonical_dir / "first_frame.png"
    source_start = int(record["alignment"]["source_start_frame"])
    expected_frames = int(record["alignment"]["output_frame_count"])
    source_rate = record["source_probe"]["nominal_frame_rate"]
    if source_rate != "240/1":
        raise ValueError(f"unexpected source nominal FPS for {source}: {source_rate}")

    if not reference.is_file():
        temporary = canonical_dir / "reference.tmp.mp4"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"trim=start_frame={source_start},setpts=N/(240*TB)",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-threads",
                "8",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-vsync",
                "0",
                "-metadata:s:v:0",
                "rotate=0",
                "-y",
                str(temporary),
            ]
        )
        os.replace(temporary, reference)

    output_probe = _probe(reference)
    if output_probe["frame_count"] != expected_frames:
        raise ValueError(
            f"frame-count mismatch for {reference}: "
            f"{output_probe['frame_count']} != {expected_frames}"
        )
    if output_probe["nominal_frame_rate"] != source_rate:
        raise ValueError(
            f"FPS mismatch for {reference}: "
            f"{output_probe['nominal_frame_rate']} != {source_rate}"
        )
    if (output_probe["width"], output_probe["height"]) != (1080, 1920):
        raise ValueError(f"unexpected canonical dimensions: {reference}: {output_probe}")

    temporary_png = canonical_dir / "first_frame.tmp.png"
    _extract_frame(reference, 0, temporary_png)
    os.replace(temporary_png, first_frame)
    if _raw_rgb_digest(reference) != _raw_rgb_digest(first_frame):
        raise ValueError(f"canonical first-frame pixel mismatch: {reference}")

    source_png = canonical_dir / "source_start.audit.png"
    _extract_frame(source, source_start, source_png)
    start_psnr = _psnr(source_png, first_frame)
    source_png.unlink()
    if start_psnr < 35.0:
        raise ValueError(f"poor source/canonical start-frame alignment: {start_psnr:.3f} dB")

    relative_root = asset_root.relative_to(DATASET_ROOT)
    reference_relative = str(relative_root / directory / "canonical/reference.mp4")
    first_relative = str(relative_root / directory / "canonical/first_frame.png")
    background = BACKGROUND_IDS[record["annotation"]["background"]]
    physics = _normalized_physics(record)
    case = {
        "schema_version": "4.0",
        "case_id": _case_id(record),
        "scene_id": "pendulum",
        "text": {
            "schema_version": "1.0",
            "prompt": PROMPT,
            "language": "en",
            "annotation_source": IMPORT_ID,
        },
        "assets": {
            "first_frame": first_relative,
            "reference_video": reference_relative,
            "physics_reference_video": reference_relative,
            "subject_mask": None,
        },
        "physics": physics,
        "appearance": {
            "background": background,
            "bob_material": "not_documented",
            "camera": "fixed",
            "capture_session": "supplement_20260731",
            "support": "laboratory_pendulum_rig",
        },
        "temporal": {
            "time_scale": "real_time",
            "encoded_to_physical_speed": 1,
            "annotation_source": IMPORT_ID,
            "nominal_frame_rate": source_rate,
            "frame_count_policy": "all source frames from the selected start through EOF",
        },
        "alignment": {
            "canonical_first_frame_event": record["alignment"][
                "canonical_first_frame_event"
            ],
            "source_start_frame": source_start,
            "source_end_frame_exclusive": record["alignment"][
                "source_end_frame_exclusive"
            ],
            "spatial_crop": None,
        },
        "provenance": {
            "source_kind": "real_capture",
            "parent_case_id": None,
            "import_id": IMPORT_ID,
            "source_member": record["annotation"]["video_member"],
            "source_workbook_row": record["annotation"]["source_row"],
            "source_sha256": record["source_sha256"],
        },
        "has_real_reference_video": True,
    }
    return {
        "case": case,
        "asset_directory": directory,
        "source_path": str(source),
        "source_sha256": record["source_sha256"],
        "source_probe": record["source_probe"],
        "canonical_reference_sha256": sha256(reference),
        "canonical_first_frame_sha256": sha256(first_frame),
        "canonical_probe": output_probe,
        "source_to_canonical_start_frame_psnr_db": start_psnr,
        "frame_count_preserved_after_trim": True,
        "fps_preserved": True,
        "spatial_crop": None,
        "visual_review": "accepted_start_turning_triptych_and_full_span_five_frame_sheet",
        "status": "staged_accepted_not_yet_published",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _preserve_sources(archive: Path) -> tuple[Path, Path]:
    archive_target = (
        DATASET_ROOT
        / "assets/source_archives/20260804_new_data/pendulum_supplement.zip"
    )
    archive_target.parent.mkdir(parents=True, exist_ok=True)
    if not archive_target.exists():
        os.link(archive, archive_target)
    if sha256(archive_target) != sha256(archive):
        raise ValueError(f"source archive preservation mismatch: {archive_target}")

    docs = DATASET_ROOT / "provenance/source_docs/20260804_pendulum_supplement"
    docs.mkdir(parents=True, exist_ok=True)
    workbook_target = docs / "钟摆实验.xlsx"
    with ZipFile(archive) as source:
        workbook = source.read(WORKBOOK_MEMBER)
    if not workbook_target.exists():
        workbook_target.write_bytes(workbook)
    if workbook_target.read_bytes() != workbook:
        raise ValueError(f"source workbook preservation mismatch: {workbook_target}")
    return archive_target, workbook_target


def _copy_review_evidence(review_dir: Path) -> list[dict[str, Any]]:
    pages = sorted(review_dir.glob("page_*.jpg"))
    if len(pages) != 11:
        raise ValueError(f"expected 11 full-span review pages, found {len(pages)}")
    target = DATASET_ROOT / "provenance/alignment" / IMPORT_ID / "full_span_review"
    target.mkdir(parents=True, exist_ok=True)
    evidence: list[dict[str, Any]] = []
    for page in pages:
        copied = target / page.name
        shutil.copy2(page, copied)
        evidence.append(
            {
                "path": str(copied.relative_to(DATASET_ROOT)),
                "sha256": sha256(copied),
                "status": "visually_reviewed_accepted",
            }
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    records = analysis["records"]
    if len(records) != 65 or len(analysis["video_quality_exclusions"]) != 35:
        raise ValueError("unexpected reviewed pendulum supplement population")
    source_hashes = [record["source_sha256"] for record in records]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("duplicate source-video SHA-256 within accepted supplement")

    current_cases = DATASET_ROOT / "releases/7.0.0/cases.jsonl"
    existing_first_frames = []
    for line in current_cases.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["scene_id"] == "pendulum":
            existing_first_frames.append(DATASET_ROOT / case["assets"]["first_frame"])
    existing_first_frames.sort()
    # Existing R1/R2 uses a visibly different freestanding rig.  Exact hashes
    # supplement the all-frame contact-sheet review without treating independent
    # repeated trials at the same physical settings as duplicates.
    existing_hashes = {sha256(path) for path in existing_first_frames}

    archive_target, workbook_target = _preserve_sources(args.archive)
    review_evidence = _copy_review_evidence(args.review_dir)
    asset_root = DATASET_ROOT / "assets/pendulum"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        materialized = list(
            executor.map(
                _materialize_one,
                [(record, str(asset_root)) for record in records],
            )
        )
    materialized.sort(key=lambda item: item["case"]["case_id"])
    if any(
        item["canonical_first_frame_sha256"] in existing_hashes
        for item in materialized
    ):
        raise ValueError("exact first-frame duplicate found against existing pendulum set")

    docs = DATASET_ROOT / "provenance/source_docs/20260804_pendulum_supplement"
    import_dir = DATASET_ROOT / "provenance/imports"
    case_drafts = [item["case"] for item in materialized]
    _write_jsonl(docs / "case_drafts.jsonl", case_drafts)
    _write_jsonl(import_dir / f"{IMPORT_ID}_import_audit.jsonl", materialized)
    _write_json(
        import_dir / f"{IMPORT_ID}_exclusions.json",
        {
            "schema_version": "1.0",
            "import_id": IMPORT_ID,
            "ignored_annotations_missing_video": analysis["ignored_rows_missing_media"],
            "ignored_annotations_missing_video_count": len(
                analysis["ignored_rows_missing_media"]
            ),
            "excluded_video_quality": analysis["video_quality_exclusions"],
            "excluded_video_quality_count": len(analysis["video_quality_exclusions"]),
        },
    )
    normalized = dict(analysis)
    for record in normalized["records"]:
        record["prompt"] = PROMPT
        record["normalized_physics"] = _normalized_physics(record)
        record["normalized_appearance"]["bob_material"] = "not_documented"
        record["status"] = "visually_verified_and_materialized"
    _write_json(docs / "normalized_annotations.json", normalized)
    _write_json(
        import_dir / f"{IMPORT_ID}_summary.json",
        {
            "schema_version": "1.0",
            "import_id": IMPORT_ID,
            "source_archive": str(archive_target.relative_to(DATASET_ROOT)),
            "source_archive_sha256": sha256(archive_target),
            "source_workbook": str(workbook_target.relative_to(DATASET_ROOT)),
            "accepted_case_drafts": len(case_drafts),
            "video_quality_exclusions": len(analysis["video_quality_exclusions"]),
            "ignored_annotations_missing_video": len(
                analysis["ignored_rows_missing_media"]
            ),
            "prompt": PROMPT,
            "physics_decisions": {
                "bob_radius": "user-confirmed 1 cm, normalized to 0.01 m",
                "bob_mass": "user-confirmed 31.5 g, normalized to 0.0315 kg",
                "string_length": "workbook 线长 converted from cm to m",
                "pendulum_length": "derived as string_length + bob_radius",
                "initial_angle": "workbook 角度 retained in degrees",
            },
            "appearance_decisions": {
                "background": "normalized under appearance only",
                "bob_material": "not_documented; metallic appearance was not treated as a material annotation",
            },
            "duplicate_audit": {
                "accepted_source_sha256_unique": True,
                "existing_pendulum_case_count": len(existing_first_frames),
                "exact_first_frame_duplicates_against_existing": 0,
                "visual_existing_rig_comparison": "distinct apparatus and capture setup; no near-duplicate candidate",
            },
            "review_evidence": review_evidence,
            "release_status": (
                "staged_not_published_pending_atomic_release_with_new push-bottle scene"
            ),
        },
    )


if __name__ == "__main__":
    main()
