#!/usr/bin/env python3
"""Materialize reviewed, frame-exact collision entry alignment.

The byte-preserved ``reference.mov`` remains the source of truth.  This script creates a
lossless HEVC ``reference_aligned.mp4`` beginning at the reviewed decoded source frame,
derives its frame-0 PNG, verifies decoded-frame identity and frame counts, then updates
the case manifest and frozen split views atomically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import load_jsonl, write_json  # noqa: E402
from physbench.splitters import build_view_a, build_view_b  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "cases.jsonl"
DEFAULT_REVIEW = ROOT / "data" / "alignment_audits" / "collision_entry_v1" / "reviewed_frames.json"
DEFAULT_AUDIT_DIR = ROOT / "data" / "alignment_audits" / "collision_entry_v1"
VIEW_A = ROOT / "data" / "splits" / "view_a.json"
VIEW_B = ROOT / "data" / "splits" / "view_b_seed42_g5.json"
ALIGNMENT_VERSION = "collision_entry_v1"
PROMPT_PREFIX = (
    "A fixed-camera real-world laboratory video of a one-dimensional central collision "
    "among three aligned balls. At frame 0, all three balls are visible, with ball 1 on "
    "the left. "
)
OLD_PROMPT_PREFIX = (
    "A fixed-camera real-world laboratory video of a one-dimensional central collision "
    "among three aligned balls. "
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,duration",
        "-of", "json", str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "time_base": stream.get("time_base"),
        "frames": int(stream["nb_frames"]),
        "duration_s": float(stream["duration"]),
    }


def decoded_frame_hash(path: Path, frame: int) -> str:
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if frame:
        command.extend(["-vf", f"select=eq(n\\,{frame})"])
    command.extend([
        "-frames:v", "1", "-an", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ])
    payload = subprocess.check_output(command)
    if not payload:
        raise RuntimeError(f"failed to decode frame {frame} from {path}")
    return hashlib.sha256(payload).hexdigest()


def encoded_png_hash(path: Path, frame: int) -> str:
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if frame:
        command.extend(["-vf", f"select=eq(n\\,{frame})"])
    command.extend(["-frames:v", "1", "-an", "-c:v", "png", "-f", "image2pipe", "pipe:1"])
    payload = subprocess.check_output(command)
    if not payload:
        raise RuntimeError(f"failed to render PNG for frame {frame} from {path}")
    return hashlib.sha256(payload).hexdigest()


def relative_asset(case_dir: Path, name: str, manifest: Path) -> str:
    return os.path.relpath(case_dir / name, manifest.parent)


def run_ffmpeg(source: Path, temporary: Path, start_frame: int, threads: int) -> list[str]:
    if start_frame == 0:
        command = [
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
            "-c:v", "copy", "-video_track_timescale", "240000", "-movflags", "+faststart",
            str(temporary),
        ]
    else:
        command = [
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", f"trim=start_frame={start_frame},setpts=PTS-STARTPTS",
            "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", f"lossless=1:pools={threads}:frame-threads=4:log-level=error",
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-vsync", "0",
            "-video_track_timescale", "240000", "-movflags", "+faststart", str(temporary),
        ]
    subprocess.run(command, check=True)
    return command


def process_case(
    case: dict[str, Any], manifest: Path, start_frame: int, threads: int, overwrite: bool,
) -> dict[str, Any]:
    assets = case["assets"]
    source_value = assets.get("source_video") or assets["reference_video"]
    source = (manifest.parent / source_value).resolve()
    case_dir = source.parent
    destination = case_dir / "reference_aligned.mp4"
    temporary = case_dir / ".reference_aligned.partial.mp4"
    first_frame = case_dir / "first_frame.png"
    source_first_frame = case_dir / "first_frame_source.png"
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.name != "reference.mov":
        canonical_source = case_dir / "reference.mov"
        if canonical_source.is_file():
            source = canonical_source
    source_info = probe(source)
    if start_frame < 0 or start_frame >= source_info["frames"]:
        raise ValueError(f"invalid start frame {start_frame} for {case['case_id']}")

    if first_frame.is_file() and not source_first_frame.exists():
        shutil.copy2(first_frame, source_first_frame)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
    temporary.unlink(missing_ok=True)
    command = run_ffmpeg(source, temporary, start_frame, threads)
    output_info = probe(temporary)
    expected_frames = source_info["frames"] - start_frame
    if output_info["frames"] != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"frame count mismatch for {case['case_id']}: {output_info['frames']} != {expected_frames}"
        )
    if (output_info["width"], output_info["height"]) != (source_info["width"], source_info["height"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"spatial dimensions changed for {case['case_id']}")
    source_frame_hash = decoded_frame_hash(source, start_frame)
    output_frame_hash = decoded_frame_hash(temporary, 0)
    if source_frame_hash != output_frame_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"lossless frame-0 verification failed for {case['case_id']}")
    os.replace(temporary, destination)

    first_temporary = case_dir / ".first_frame.partial.png"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(destination), "-frames:v", "1",
        "-c:v", "png", str(first_temporary),
    ], check=True)
    if encoded_png_hash(destination, 0) != sha256(first_temporary):
        first_temporary.unlink(missing_ok=True)
        raise RuntimeError(f"derived first-frame verification failed for {case['case_id']}")
    os.replace(first_temporary, first_frame)

    frame_rate = Fraction(source_info["avg_frame_rate"])
    return {
        "schema_version": "1.0",
        "alignment_version": ALIGNMENT_VERSION,
        "case_id": case["case_id"],
        "source_video": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "source_probe": source_info,
        "source_start_frame": start_frame,
        "source_start_time_s_approx": float(start_frame / frame_rate),
        "aligned_video": str(destination.relative_to(ROOT)),
        "aligned_sha256": sha256(destination),
        "aligned_probe": output_info,
        "first_frame": str(first_frame.relative_to(ROOT)),
        "first_frame_sha256": sha256(first_frame),
        "decoded_source_frame_sha256": source_frame_hash,
        "decoded_aligned_frame0_sha256": output_frame_hash,
        "frame_count_expected": expected_frames,
        "frame_count_verified": True,
        "frame0_identity_verified": True,
        "review_status": "visually_verified",
        "source_frame_zero_exception": start_frame == 0,
        "encoder_policy": "stream-copy for N=0; otherwise lossless libx265, original PTS intervals retained",
        "ffmpeg_command": command,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_threads": threads,
    }


def updated_prompt(prompt: str) -> str:
    if prompt.startswith(PROMPT_PREFIX):
        return prompt
    if not prompt.startswith(OLD_PROMPT_PREFIX):
        raise ValueError(f"unexpected collision prompt prefix: {prompt[:100]!r}")
    return PROMPT_PREFIX + prompt[len(OLD_PROMPT_PREFIX):]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def update_case(case: dict[str, Any], record: dict[str, Any], manifest: Path) -> None:
    case_dir = ROOT / Path(record["source_video"]).parent
    case["assets"]["source_video"] = relative_asset(case_dir, "reference.mov", manifest)
    aligned = relative_asset(case_dir, "reference_aligned.mp4", manifest)
    case["assets"]["reference_video"] = aligned
    case["assets"]["physics_reference_video"] = aligned
    first = relative_asset(case_dir, "first_frame.png", manifest)
    case["assets"]["first_frame"] = first
    case["text"]["prompt"] = updated_prompt(case["text"]["prompt"])
    for view in case["input_views"].values():
        if "prompt" in view:
            view["prompt"] = updated_prompt(view["prompt"])
        if "first_frame" in view:
            view["first_frame"] = first
    case["alignment"] = {
        "version": ALIGNMENT_VERSION,
        "method": "decoded_source_frame_trim",
        "source_start_frame": record["source_start_frame"],
        "source_start_time_s_approx": record["source_start_time_s_approx"],
        "review_status": "visually_verified",
        "audit_record": os.path.relpath(
            DEFAULT_AUDIT_DIR / "alignment_audit.jsonl", manifest.parent,
        ),
        "source_frame_zero_exception": record["source_frame_zero_exception"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-job", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--case-id", help="Process one case only; manifest is not updated.")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    reviewed_frames = review["frames"]
    cases = load_jsonl(manifest)
    collision = {case["case_id"]: case for case in cases if case["scene_id"] == "collision_1d"}
    if set(collision) != set(reviewed_frames):
        raise ValueError(
            f"review/manifest mismatch: missing={sorted(set(collision)-set(reviewed_frames))}, "
            f"extra={sorted(set(reviewed_frames)-set(collision))}"
        )
    selected_ids = [args.case_id] if args.case_id else sorted(collision)
    if any(case_id not in collision for case_id in selected_ids):
        raise KeyError(args.case_id)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(selected_ids))) as pool:
        futures = {
            pool.submit(
                process_case, collision[case_id], manifest, int(reviewed_frames[case_id]),
                args.threads_per_job, args.overwrite,
            ): case_id
            for index, case_id in enumerate(selected_ids)
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            case_id = futures[future]
            record = future.result()
            records.append(record)
            print(
                f"[{index:02d}/{len(selected_ids)}] {case_id} start={record['source_start_frame']} "
                f"frames={record['aligned_probe']['frames']} threads={record['encoder_threads']}",
                flush=True,
            )
    records.sort(key=lambda row: row["case_id"])
    if args.case_id:
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
        return 0

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    before = args.audit_dir / "cases_before_alignment.jsonl"
    if not before.exists():
        shutil.copy2(manifest, before)
    write_jsonl_atomic(args.audit_dir / "alignment_audit.jsonl", records)
    with (args.audit_dir / "alignment_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_id", "source_start_frame", "source_start_time_s_approx", "source_frames",
            "aligned_frames", "source_sha256", "aligned_sha256", "frame0_identity_verified",
            "source_frame_zero_exception",
        ])
        for row in records:
            writer.writerow([
                row["case_id"], row["source_start_frame"], f"{row['source_start_time_s_approx']:.9f}",
                row["source_probe"]["frames"], row["aligned_probe"]["frames"],
                row["source_sha256"], row["aligned_sha256"], row["frame0_identity_verified"],
                row["source_frame_zero_exception"],
            ])
    by_id = {row["case_id"]: row for row in records}
    for case in cases:
        if case["case_id"] in by_id:
            update_case(case, by_id[case["case_id"]], manifest)
    write_jsonl_atomic(manifest, cases)
    write_json(VIEW_A, build_view_a(cases))
    write_json(VIEW_B, build_view_b(cases, groups=5, seed=42))
    print(json.dumps({
        "aligned_cases": len(records),
        "manifest": str(manifest),
        "audit": str(args.audit_dir / "alignment_audit.jsonl"),
        "view_a": str(VIEW_A),
        "view_b": str(VIEW_B),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
