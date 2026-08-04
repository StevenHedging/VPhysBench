#!/usr/bin/env python3
"""Materialize the reviewed 2026-08-04 push-bottle import.

The source video is the canonical reference: no temporal trim, spatial crop,
resampling, frame-rate change, or re-encoding is performed.  A PNG first frame
is decoded from that byte-preserved reference and checked against its decoded
frame zero.  Publication into a Dataset release remains a separate step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets/physics_video"
DEFAULT_ANALYSIS = Path(
    "/mnt/nvme1/physics_video_benchmark_ingest/20260804_new_data/"
    "push_bottle_analysis.json"
)
VIDEO_ARCHIVE = Path("/root/Steven/推水瓶.zip")
ANNOTATION_ARCHIVE = Path("/root/Steven/推水瓶实验物理标注.zip")
REVIEW_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark_ingest/20260804_new_data/"
    "review/push_full_span"
)
IMPORT_ID = "push_bottle_20260804"
PROMPT = (
    "An upright bottle is pushed near its top, tips over, and falls onto its side."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decoded_zip_name(value: str) -> str:
    try:
        return value.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _number_token(value: float, *, digits: int = 6) -> str:
    rendered = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return rendered.replace("-", "neg").replace(".", "p")


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    physics = record["normalized_physics"]
    mass_g = float(physics["bottle_mass"]["value"]) * 1000.0
    height_mm = int(round(float(physics["bottle_height"]["value"]) * 1000.0))
    peak_n = float(physics["peak_applied_force"]["value"])
    mean_n = float(physics["mean_applied_force"]["value"])
    stem = str(record["annotation"]["source_sheet"]).lower().replace("_", "")
    physical = (
        f"m{_number_token(mass_g, digits=3)}g_h{height_mm:03d}mm_"
        f"fmax{_number_token(peak_n)}n_fmean{_number_token(mean_n)}n"
    )
    directory = f"push_bottle_{physical}_{stem}"
    case_id = f"push_bottle_{physical}_{stem}"
    return directory, case_id


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
        "stored_width": int(stream["width"]),
        "stored_height": int(stream["height"]),
        "rotation_deg": int(stream.get("tags", {}).get("rotate", 0)),
        "nominal_frame_rate": stream["r_frame_rate"],
        "average_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream.get("duration", payload["format"]["duration"])),
    }


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


def _copy_byte_preserving(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected_sha256:
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"byte-preserving copy hash mismatch: {source}")
    os.replace(temporary, target)


def _materialize_one(arguments: tuple[dict[str, Any], str]) -> dict[str, Any]:
    record, asset_root_value = arguments
    asset_root = Path(asset_root_value)
    source = Path(record["source_path"])
    directory, case_id = _identity(record)
    canonical = asset_root / directory / "canonical"
    reference = canonical / "reference.mov"
    first_frame = canonical / "first_frame.png"
    _copy_byte_preserving(source, reference, record["source_sha256"])

    canonical.mkdir(parents=True, exist_ok=True)
    temporary_frame = canonical / "first_frame.tmp.png"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(reference),
            "-frames:v",
            "1",
            "-y",
            str(temporary_frame),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    os.replace(temporary_frame, first_frame)
    if _raw_rgb_digest(reference) != _raw_rgb_digest(first_frame):
        raise ValueError(f"first-frame pixel mismatch: {reference}")

    output_probe = _probe(reference)
    if output_probe != record["source_probe"]:
        raise ValueError(f"byte-preserved reference probe changed: {reference}")
    relative_root = asset_root.relative_to(DATASET_ROOT)
    reference_relative = str(relative_root / directory / "canonical/reference.mov")
    first_relative = str(relative_root / directory / "canonical/first_frame.png")
    video_archive_relative = (
        "assets/source_archives/20260804_new_data/push_bottle.zip"
    )
    annotation_archive_relative = (
        "assets/source_archives/20260804_new_data/"
        "push_bottle_physics_annotations.zip"
    )
    physics = json.loads(json.dumps(record["normalized_physics"]))
    case = {
        "schema_version": "4.0",
        "case_id": case_id,
        "scene_id": "push_bottle",
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
            "source_video": reference_relative,
            "source_archive": video_archive_relative,
            "annotation_archive": annotation_archive_relative,
            "subject_mask": None,
        },
        "physics": physics,
        "appearance": json.loads(json.dumps(record["normalized_appearance"])),
        "temporal": {
            "time_scale": "real_time",
            "encoded_to_physical_speed": 1,
            "annotation_source": IMPORT_ID,
            "nominal_frame_rate": output_probe["nominal_frame_rate"],
            "frame_count_policy": "source video retained byte-for-byte",
        },
        "alignment": {
            "canonical_first_frame_event": "upright bottle before the push develops",
            "source_start_frame": 0,
            "source_end_frame_exclusive": output_probe["frame_count"],
            "spatial_crop": None,
            "temporal_trim": None,
            "review_status": "visually_verified_full_span_five_frame_sheet",
        },
        "provenance": {
            "source_kind": "real_capture",
            "parent_case_id": None,
            "import_id": IMPORT_ID,
            "source_sha256": record["source_sha256"],
            "source_locator": {
                "archive": video_archive_relative,
                "member": record["annotation"]["video_member"],
                "annotation_archive": annotation_archive_relative,
                "annotation_workbook": record["annotation"][
                    "source_workbook_member"
                ],
                "sheet": record["annotation"]["source_sheet"],
                "normalized_annotation": (
                    "provenance/source_docs/20260804_push_bottle/"
                    "normalized_annotations.json"
                ),
            },
        },
        "has_real_reference_video": True,
    }
    return {
        "case": case,
        "asset_directory": directory,
        "source_path": str(source),
        "source_sha256": record["source_sha256"],
        "canonical_reference_sha256": sha256(reference),
        "canonical_first_frame_sha256": sha256(first_frame),
        "source_probe": record["source_probe"],
        "canonical_probe": output_probe,
        "copy_policy": "byte_preserving_no_trim_no_crop_no_reencode",
        "first_frame_matches_canonical_frame_zero": True,
        "visual_review": "accepted_full_span_five_frame_contact_sheet",
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


def _preserve_file(source: Path, target: Path) -> None:
    _copy_byte_preserving(source, target, sha256(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--video-archive", type=Path, default=VIDEO_ARCHIVE)
    parser.add_argument(
        "--annotation-archive", type=Path, default=ANNOTATION_ARCHIVE
    )
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--visually-reviewed", action="store_true")
    args = parser.parse_args()
    if not args.visually_reviewed:
        raise ValueError(
            "refusing to materialize without --visually-reviewed; all 141 matched "
            "videos must have a current full-span visual review"
        )
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if analysis.get("intake_id") != IMPORT_ID:
        raise ValueError("analysis import ID mismatch")
    if analysis.get("videos_missing_annotation") != ["IMG_0076"]:
        raise ValueError("unexpected missing-annotation set")
    if analysis.get("annotations_missing_video"):
        raise ValueError("annotations without video require a fresh intake decision")
    records = analysis["records"]
    if len(records) != 141 or any(
        record.get("status") != "candidate_needs_visual_review"
        for record in records
    ):
        raise ValueError("expected exactly 141 reviewed intake candidates")

    archive_root = DATASET_ROOT / "assets/source_archives/20260804_new_data"
    video_archive_target = archive_root / "push_bottle.zip"
    annotation_archive_target = archive_root / "push_bottle_physics_annotations.zip"
    _preserve_file(args.video_archive, video_archive_target)
    _preserve_file(args.annotation_archive, annotation_archive_target)

    docs = DATASET_ROOT / "provenance/source_docs/20260804_push_bottle"
    docs.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.annotation_archive) as archive:
        by_decoded_name = {
            _decoded_zip_name(item.filename): item
            for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).suffix == ".xlsx"
        }
        for decoded, item in sorted(by_decoded_name.items()):
            target = docs / PurePosixPath(decoded).name
            payload = archive.read(item)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite changed XLSX: {target}")
            target.write_bytes(payload)

    review_target = DATASET_ROOT / "provenance/reviews/20260804_push_bottle"
    review_target.mkdir(parents=True, exist_ok=True)
    review_files = sorted(args.review_root.glob("page_*.jpg"))
    if len(review_files) != 24:
        raise ValueError("expected 24 push-bottle full-span review pages")
    review_manifest = []
    for source in review_files:
        target = review_target / source.name
        _preserve_file(source, target)
        review_manifest.append(
            {
                "path": str(target.relative_to(DATASET_ROOT)),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    asset_root = DATASET_ROOT / "assets/push_bottle"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        materialized = list(
            executor.map(
                _materialize_one,
                [(record, str(asset_root)) for record in records],
            )
        )
    materialized.sort(key=lambda item: item["case"]["case_id"])
    case_drafts = [item["case"] for item in materialized]
    if len({case["case_id"] for case in case_drafts}) != 141:
        raise ValueError("push-bottle case IDs are not unique")
    if any(
        item["source_sha256"] != item["canonical_reference_sha256"]
        for item in materialized
    ):
        raise ValueError("a canonical push-bottle reference changed source bytes")

    normalized = json.loads(json.dumps(analysis))
    normalized["visual_review"] = {
        "status": "all_141_matched_videos_accepted",
        "method": "five decoded times spanning every complete video",
        "evidence": review_manifest,
        "decision": (
            "Each accepted video shows one upright bottle pushed near its top, "
            "tipping over, and falling onto its side without abnormal truncation."
        ),
    }
    for record in normalized["records"]:
        record["status"] = "visually_verified_and_materialized"
    _write_json(docs / "normalized_annotations.json", normalized)
    _write_jsonl(docs / "case_drafts.jsonl", case_drafts)
    import_root = DATASET_ROOT / "provenance/imports"
    _write_jsonl(
        import_root / f"{IMPORT_ID}_import_audit.jsonl",
        materialized,
    )
    _write_json(
        import_root / f"{IMPORT_ID}_exclusions.json",
        {
            "schema_version": "1.0",
            "import_id": IMPORT_ID,
            "excluded_videos": [
                {
                    "source_member": "推水瓶/IMG_0076.MOV",
                    "source_stem": "IMG_0076",
                    "reason_code": "missing_annotation",
                    "reason": "No XLSX sheet uniquely maps physical annotations to this video.",
                }
            ],
            "annotations_missing_video": [],
        },
    )
    corrected = [
        record
        for record in records
        if record["annotation"].get("unit_corrections")
    ]
    _write_json(
        import_root / f"{IMPORT_ID}_summary.json",
        {
            "schema_version": "1.0",
            "import_id": IMPORT_ID,
            "accepted_case_drafts": len(case_drafts),
            "excluded_missing_annotation": 1,
            "annotations_missing_video": 0,
            "source_video_archive": str(
                video_archive_target.relative_to(DATASET_ROOT)
            ),
            "source_video_archive_sha256": sha256(video_archive_target),
            "source_annotation_archive": str(
                annotation_archive_target.relative_to(DATASET_ROOT)
            ),
            "source_annotation_archive_sha256": sha256(annotation_archive_target),
            "prompt": PROMPT,
            "media_policy": "byte preserving; no crop, trim, resampling, or re-encode",
            "physics_fields": [
                "bottle_mass",
                "bottle_height",
                "peak_applied_force",
                "mean_applied_force",
            ],
            "force_units": {
                "source_N_cases": sum(
                    record["force_annotation"]["source_force_unit"] == "N"
                    for record in records
                ),
                "source_kgf_cases": sum(
                    record["force_annotation"]["source_force_unit"] == "Kgf"
                    for record in records
                ),
                "canonical_unit": "N",
                "kgf_to_N": 9.80665,
            },
            "annotation_corrections": [
                {
                    "workbook": record["annotation"]["source_workbook_member"],
                    "corrections": record["annotation"]["unit_corrections"],
                }
                for record in corrected
            ],
            "visual_review_evidence": review_manifest,
            "release_status": "staged_not_published",
        },
    )
    print(
        json.dumps(
            {
                "accepted": len(case_drafts),
                "excluded_missing_annotation": ["IMG_0076"],
                "asset_root": str(asset_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
