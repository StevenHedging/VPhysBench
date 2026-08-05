#!/usr/bin/env python3
"""Find and materialize the inclined-plane release frame for each annotated clip.

The source archive is immutable.  For every annotation-backed member this tool:

1. estimates the first sustained movement of the block in a fixed upper-plane ROI;
2. rewinds a small number of frames so canonical frame zero is immediately before
   visible displacement;
3. writes a before/start/after review strip;
4. optionally encodes the exact decoded-frame trim and extracts canonical frame 0.

The proposal pass is intentionally separate from ``--accept-proposals``.  A human
or visual agent must inspect the review sheets before freezing reviewed_frames.json.
Run with the ``phybench`` Conda interpreter because OpenCV and NumPy are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"
ARCHIVE = (
    DATA_ROOT / "assets" / "source_archives" / "20260723_new_scenes"
    / "inclined_plane_raw.zip"
)
ANNOTATIONS = (
    DATA_ROOT / "provenance" / "source_docs" / "20260723_new_scenes"
    / "inclined_plane_annotations.xlsx"
)
ALIGNMENT_ROOT = (
    DATA_ROOT / "provenance" / "alignment" / "inclined_plane_start_v1"
)
PROPOSALS = ALIGNMENT_ROOT / "proposed_frames.json"
REVIEWED = ALIGNMENT_ROOT / "reviewed_frames.json"
AUDIT = ALIGNMENT_ROOT / "alignment_audit.jsonl"
REVIEW_ROOT = ALIGNMENT_ROOT / "review"
ASSET_ROOT = DATA_ROOT / "assets" / "inclined_plane_slide"

XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str | None]]:
    """Read cached XLSX cell values without adding an openpyxl dependency."""
    with zipfile.ZipFile(path) as workbook:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.iterfind(".//a:t", XML_NS))
                for item in root.findall("a:si", XML_NS)
            ]
        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        sheet = next(
            item
            for item in book.find("a:sheets", XML_NS) or []
            if item.attrib["name"] == sheet_name
        )
        relation_key = f"{{{XML_NS['r']}}}id"
        target = targets[sheet.attrib[relation_key]]
        member = (
            target.lstrip("/")
            if target.startswith("/")
            else str(PurePosixPath("xl") / target)
        )
        root = ET.fromstring(workbook.read(member))
        rows: list[dict[str, str | None]] = []
        for row in root.findall(".//a:sheetData/a:row", XML_NS):
            values: dict[str, str | None] = {}
            for cell in row.findall("a:c", XML_NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"])
                if column is None:
                    continue
                value_node = cell.find("a:v", XML_NS)
                kind = cell.attrib.get("t")
                value: str | None = None
                if kind == "s" and value_node is not None:
                    value = strings[int(value_node.text or "0")]
                elif kind == "inlineStr":
                    inline = cell.find("a:is", XML_NS)
                    if inline is not None:
                        value = "".join(
                            node.text or ""
                            for node in inline.iterfind(".//a:t", XML_NS)
                        )
                elif value_node is not None:
                    value = value_node.text
                values[column.group()] = value
            rows.append(values)
        return rows


def normalized_stem(value: str) -> str:
    match = re.search(r"(\d+)", value)
    if match is None:
        raise ValueError(f"filename has no numeric identifier: {value}")
    return f"IMG_{int(match.group(1)):04d}"


def annotation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in _xlsx_rows(ANNOTATIONS, "Trial记录"):
        filename = value.get("B")
        angle = value.get("C")
        if not filename or not angle or not re.search(r"\d", filename):
            continue
        background_zh = (value.get("G") or "").strip()
        background = {
            "": "default_white",
            "油画": "oil_painting",
            "黑色泡沫板": "black_foam_board",
            "绿色卡纸": "green_cardstock",
        }.get(background_zh)
        if background is None:
            raise ValueError(f"unknown background annotation: {background_zh}")
        rows.append(
            {
                "trial_id": value.get("A"),
                "source_stem": normalized_stem(filename),
                "angle_deg": int(float(angle)),
                "friction_force_n": float(value["D"] or "nan"),
                "theoretical_acceleration_m_s2": float(value["E"] or "nan"),
                "background": background,
                "background_source_value": background_zh or "default",
            }
        )
    if len(rows) != 100:
        raise ValueError(f"expected 100 annotation rows, found {len(rows)}")
    return rows


def complete_members() -> tuple[dict[str, str], dict[str, str]]:
    complete: dict[str, str] = {}
    partial: dict[str, str] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            stem = normalized_stem(Path(item.filename).stem)
            target = partial if item.filename.endswith(".drivedownload") else complete
            if stem in target:
                raise ValueError(f"duplicate normalized archive member: {stem}")
            target[stem] = item.filename
    return complete, partial


def accepted_sources() -> list[dict[str, Any]]:
    rows = annotation_rows()
    complete, _ = complete_members()
    accepted = [
        {**row, "archive_member": complete[row["source_stem"]]}
        for row in rows
        if row["source_stem"] in complete
    ]
    if len(accepted) != 95:
        raise ValueError(f"expected 95 complete annotated clips, found {len(accepted)}")
    return accepted


def case_id(row: dict[str, Any]) -> str:
    backgrounds = {
        "default_white": "bgwhite",
        "oil_painting": "bgoil",
        "black_foam_board": "bgblack",
        "green_cardstock": "bggreen",
    }
    return (
        f"incline_r1_a{row['angle_deg']:02d}deg_"
        f"{backgrounds[row['background']]}_{row['source_stem'].lower()}"
    )


def extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _gray_frame(frame: np.ndarray, width: int = 320) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def detect_start(video: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid FPS for {video}: {fps}")
    limit = min(frame_count, max(1, round(8.0 * fps)))
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_gray_frame(frame))
    capture.release()
    if len(frames) < max(12, round(0.15 * fps)):
        raise RuntimeError(f"too few decoded frames in {video}")

    reference_count = min(len(frames), max(8, round(0.10 * fps)))
    reference = np.median(np.stack(frames[:reference_count]), axis=0).astype(
        np.uint8
    )
    height, width = reference.shape
    # The apparatus is fixed and the block release point is in the upper/right
    # half.  Excluding the lower half prevents an entering hand from dominating.
    roi = (slice(0, round(height * 0.58)), slice(round(width * 0.36), width))
    mean_scores: list[float] = []
    active_scores: list[float] = []
    for frame in frames:
        difference = cv2.absdiff(frame[roi], reference[roi])
        mean_scores.append(float(difference.mean()))
        active_scores.append(float((difference > 12).mean()))

    early_index = min(len(frames) - 1, max(1, round(0.10 * fps)))
    early_difference = cv2.absdiff(frames[early_index][roi], frames[0][roi])
    early_mean = float(early_difference.mean())
    early_active = float((early_difference > 12).mean())
    trend_count = min(len(frames), max(12, round(0.30 * fps)))
    trend_x = np.arange(trend_count, dtype=np.float64)
    trend_y = np.asarray(mean_scores[:trend_count], dtype=np.float64)
    trend_slope_per_frame = float(np.polyfit(trend_x, trend_y, 1)[0])
    trend_correlation = float(np.corrcoef(trend_x, trend_y)[0, 1])
    edge_count = min(10, max(1, trend_count // 4))
    trend_delta = float(
        np.mean(trend_y[-edge_count:]) - np.mean(trend_y[:edge_count])
    )
    # Some source clips already begin within a few frames of release.  In that
    # situation frame zero is the best available canonical boundary and must not
    # be rejected merely because a stationary reference window is unavailable.
    source_frame_zero_is_start = (
        (early_mean >= 2.0 and early_active >= 0.015)
        or (
            trend_slope_per_frame >= 0.018
            and trend_correlation >= 0.85
            and trend_delta >= 1.2
        )
    )

    baseline_end = min(len(frames), max(reference_count, round(0.25 * fps)))
    baseline_mean = np.asarray(mean_scores[:baseline_end])
    baseline_active = np.asarray(active_scores[:baseline_end])
    mean_median = float(np.median(baseline_mean))
    active_median = float(np.median(baseline_active))
    mean_mad = float(np.median(np.abs(baseline_mean - mean_median)))
    active_mad = float(np.median(np.abs(baseline_active - active_median)))
    mean_threshold = mean_median + max(1.0, 7.0 * mean_mad)
    active_threshold = active_median + max(0.018, 7.0 * active_mad)

    window = max(6, round(0.04 * fps))
    first_candidate: int | None = 0 if source_frame_zero_is_start else None
    search_start = max(reference_count, round(0.10 * fps))
    if first_candidate is None:
        for index in range(search_start, len(frames) - window + 1):
            mean_hits = sum(
                score >= mean_threshold
                for score in mean_scores[index : index + window]
            )
            active_hits = sum(
                score >= active_threshold
                for score in active_scores[index : index + window]
            )
            required = math.ceil(window * 0.75)
            if mean_hits >= required and active_hits >= required:
                first_candidate = index
                break
    if first_candidate is None:
        raise RuntimeError(
            f"no sustained start found in first {len(frames) / fps:.2f}s of {video}"
        )

    rewind = max(2, round(0.05 * fps))
    start_frame = max(0, first_candidate - rewind)
    confidence_margin = (
        min(early_mean / 2.0, early_active / 0.015)
        if source_frame_zero_is_start
        else min(
            mean_scores[first_candidate] / max(mean_threshold, 1e-9),
            active_scores[first_candidate] / max(active_threshold, 1e-9),
        )
    )
    return {
        "fps": fps,
        "source_frames": frame_count,
        "decoded_analysis_frames": len(frames),
        "detected_motion_frame": first_candidate,
        "source_start_frame": start_frame,
        "source_start_time_s_approx": start_frame / fps,
        "rewind_frames": rewind,
        "source_frame_zero_is_start": source_frame_zero_is_start,
        "early_mean_score": early_mean,
        "early_active_score": early_active,
        "early_trend_slope_per_frame": trend_slope_per_frame,
        "early_trend_correlation": trend_correlation,
        "early_trend_delta": trend_delta,
        "mean_threshold": mean_threshold,
        "active_threshold": active_threshold,
        "candidate_mean_score": mean_scores[first_candidate],
        "candidate_active_score": active_scores[first_candidate],
        "confidence_margin": confidence_margin,
    }


def decoded_frame(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index} from {video}")
    return frame


def make_review_strip(
    video: Path, start_frame: int, fps: float, destination: Path
) -> None:
    offset = max(3, round(0.08 * fps))
    indices = [
        max(0, start_frame - offset),
        start_frame,
        start_frame + offset,
    ]
    labels = ["before", "canonical frame 0", "after"]
    rendered: list[np.ndarray] = []
    for index, label in zip(indices, labels):
        frame = decoded_frame(video, index)
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (width, 45), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            f"{label}: source frame {index}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.concatenate(rendered, axis=1)):
        raise RuntimeError(f"cannot write review strip {destination}")


def proposal_for(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identifier = case_id(row)
    with tempfile.TemporaryDirectory(prefix="physbench_incline_proposal_") as raw:
        source = Path(raw) / f"{row['source_stem']}.mov"
        with zipfile.ZipFile(ARCHIVE) as archive:
            extract_member(archive, row["archive_member"], source)
        detection = detect_start(source)
        make_review_strip(
            source,
            detection["source_start_frame"],
            detection["fps"],
            REVIEW_ROOT / f"{identifier}.jpg",
        )
    return identifier, {
        "case_id": identifier,
        "source_stem": row["source_stem"],
        "archive_member": row["archive_member"],
        "angle_deg": row["angle_deg"],
        "background": row["background"],
        **detection,
    }


def build_proposals(workers: int) -> dict[str, dict[str, Any]]:
    rows = accepted_sources()
    proposals: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(proposal_for, row): row for row in rows}
        for completed, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                identifier, proposal = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case_id(row),
                        "source_stem": row["source_stem"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{completed:03d}/{len(rows):03d}] {case_id(row)} FAILED: {exc}",
                    flush=True,
                )
                continue
            proposals[identifier] = proposal
            print(
                f"[{completed:03d}/{len(rows):03d}] {identifier} "
                f"frame={proposal['source_start_frame']} "
                f"margin={proposal['confidence_margin']:.2f}",
                flush=True,
            )
    if failures:
        write_json(
            ALIGNMENT_ROOT / "proposal_failures.json",
            {"schema_version": "1.0", "failures": failures},
        )
        raise RuntimeError(
            f"{len(failures)} proposal(s) failed; see "
            f"{ALIGNMENT_ROOT / 'proposal_failures.json'}"
        )
    ordered = {key: proposals[key] for key in sorted(proposals)}
    write_json(
        PROPOSALS,
        {
            "schema_version": "1.0",
            "method": "upper_plane_reference_difference_v1",
            "archive": str(ARCHIVE.relative_to(ROOT)),
            "case_count": len(ordered),
            "proposals": ordered,
        },
    )
    return ordered


def accept_proposals() -> None:
    if not PROPOSALS.is_file():
        raise FileNotFoundError(f"run proposal pass first: {PROPOSALS}")
    value = json.loads(PROPOSALS.read_text(encoding="utf-8"))
    proposals = value["proposals"]
    review_files = {path.stem for path in REVIEW_ROOT.glob("*.jpg")}
    if set(proposals) != review_files:
        raise ValueError("review strips do not exactly cover proposal cases")
    write_json(
        REVIEWED,
        {
            "schema_version": "1.0",
            "method": value["method"],
            "review_status": "visually_verified",
            "review_artifact": "review/<case_id>.jpg",
            "case_count": len(proposals),
            "frames": {
                key: {
                    "source_start_frame": item["source_start_frame"],
                    "detected_motion_frame": item["detected_motion_frame"],
                    "fps": item["fps"],
                }
                for key, item in sorted(proposals.items())
            },
        },
    )
    print(REVIEWED)


def probe(video: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "frames": (
            int(stream["nb_frames"])
            if stream.get("nb_frames") not in {None, "N/A"}
            else None
        ),
        "duration_s": (
            float(stream["duration"])
            if stream.get("duration") not in {None, "N/A"}
            else None
        ),
    }


def encode_aligned(
    source: Path, destination: Path, start_frame: int, fps: float
) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"trim=start_frame={start_frame},setpts=PTS-STARTPTS",
        "-c:v",
        "libx265",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-tag:v",
        "hvc1",
        "-x265-params",
        "log-level=error",
        "-vsync",
        "0",
        "-video_track_timescale",
        "240000",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)
    return command


def first_frame(video: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
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


def materialize_one(
    row: dict[str, Any], frame: dict[str, Any], overwrite: bool
) -> dict[str, Any]:
    identifier = case_id(row)
    case_root = ASSET_ROOT / identifier
    destination = case_root / "canonical" / "reference.mp4"
    image = case_root / "canonical" / "first_frame.png"
    if destination.exists() and image.exists() and not overwrite:
        output_probe = probe(destination)
        with zipfile.ZipFile(ARCHIVE) as source_archive:
            source_crc32 = (
                f"{source_archive.getinfo(row['archive_member']).CRC:08x}"
            )
        return {
            "case_id": identifier,
            "status": "reused",
            "archive_member": row["archive_member"],
            "source_member_crc32": source_crc32,
            "source_start_frame": frame["source_start_frame"],
            "source_start_time_s_approx": (
                frame["source_start_frame"] / float(frame["fps"])
            ),
            "destination": str(destination.relative_to(ROOT)),
            "destination_sha256": sha256(destination),
            "first_frame": str(image.relative_to(ROOT)),
            "first_frame_sha256": sha256(image),
            "output_probe": output_probe,
        }
    with tempfile.TemporaryDirectory(prefix="physbench_incline_encode_") as raw:
        source = Path(raw) / f"{row['source_stem']}.mov"
        with zipfile.ZipFile(ARCHIVE) as archive:
            extract_member(archive, row["archive_member"], source)
            source_crc32 = f"{archive.getinfo(row['archive_member']).CRC:08x}"
        source_probe = probe(source)
        command = encode_aligned(
            source,
            destination,
            int(frame["source_start_frame"]),
            float(frame["fps"]),
        )
        first_frame(destination, image)
        output_probe = probe(destination)
        expected_frames = (
            source_probe["frames"] - int(frame["source_start_frame"])
            if source_probe["frames"] is not None
            else None
        )
        if expected_frames is not None and output_probe["frames"] != expected_frames:
            raise RuntimeError(
                f"{identifier}: expected {expected_frames} output frames, "
                f"got {output_probe['frames']}"
            )
        source_zero = decoded_frame(source, int(frame["source_start_frame"]))
        canonical_zero = decoded_frame(destination, 0)
        if source_zero.shape != canonical_zero.shape:
            raise RuntimeError(f"{identifier}: frame-zero dimensions changed")
        # Re-encoding is intentionally lossy but the canonical frame must still be
        # visually equivalent to the selected source frame.
        mean_absolute_error = float(
            np.abs(
                source_zero.astype(np.int16) - canonical_zero.astype(np.int16)
            ).mean()
        )
        if mean_absolute_error > 5.0:
            raise RuntimeError(
                f"{identifier}: canonical frame-zero MAE too high: "
                f"{mean_absolute_error:.3f}"
            )
        return {
            "case_id": identifier,
            "status": "encoded",
            "archive_member": row["archive_member"],
            "source_member_crc32": source_crc32,
            "source_probe": source_probe,
            "source_start_frame": frame["source_start_frame"],
            "source_start_time_s_approx": (
                frame["source_start_frame"] / float(frame["fps"])
            ),
            "destination": str(destination.relative_to(ROOT)),
            "destination_sha256": sha256(destination),
            "first_frame": str(image.relative_to(ROOT)),
            "first_frame_sha256": sha256(image),
            "output_probe": output_probe,
            "frame_zero_mean_absolute_error": mean_absolute_error,
            "command": command,
        }


def materialize(workers: int, overwrite: bool) -> None:
    if not REVIEWED.is_file():
        raise FileNotFoundError(
            f"reviewed mapping required before encoding: {REVIEWED}"
        )
    reviewed = json.loads(REVIEWED.read_text(encoding="utf-8"))
    frames = reviewed["frames"]
    rows = accepted_sources()
    expected = {case_id(row) for row in rows}
    if set(frames) != expected:
        raise ValueError("reviewed frame map does not exactly cover accepted cases")
    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(materialize_one, row, frames[case_id(row)], overwrite): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            audit = future.result()
            audits.append(audit)
            print(
                f"[{completed:03d}/{len(rows):03d}] {audit['case_id']} "
                f"{audit['status']}",
                flush=True,
            )
    write_jsonl(AUDIT, sorted(audits, key=lambda item: item["case_id"]))
    print(AUDIT)


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--propose", action="store_true")
    action.add_argument("--accept-proposals", action="store_true")
    action.add_argument("--materialize", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.propose:
        build_proposals(args.workers)
    elif args.accept_proposals:
        accept_proposals()
    else:
        materialize(args.workers, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
