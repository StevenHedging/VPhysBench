#!/usr/bin/env python3
"""Audit and stage the 2026-08-04 supplementary pendulum capture.

The source workbook explicitly names each video.  This script therefore never
pairs rows and media by sort order.  It reads the XLSX container without
rewriting it, selects only exact filename matches, tracks the steel bob, and
locates the first opposite-side turning point after release.  The resulting
JSON is an ingest intermediate; it does not mutate a Dataset release.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


ARCHIVE = Path("/root/Steven/补充_小球单摆实验.zip")
WORKBOOK_MEMBER = "补充_小球单摆实验/钟摆实验.xlsx"
VIDEO_MEMBER_PREFIX = "补充_小球单摆实验/"
DEFAULT_MEDIA_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark_ingest/20260804_new_data/pendulum/"
    "补充_小球单摆实验"
)
BOB_RADIUS_M = 0.01
BOB_MASS_KG = 0.0315
PROMPT = (
    "A pendulum bob starts at a turning point, swings down through the lowest "
    "point to the opposite side, and continues oscillating back and forth about "
    "the fixed pivot."
)

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class Annotation:
    source_row: int
    trial_number: int
    source_stem: str
    video_member: str
    string_length_m: float
    pendulum_length_m: float
    initial_angle_deg: float
    background: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xlsx_rows(payload: bytes) -> list[dict[str, str]]:
    ns = {"m": MAIN_NS, "r": REL_NS, "pr": PACKAGE_REL_NS}
    with ZipFile(BytesIO(payload)) as workbook:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append(
                    "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                )

        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("pr:Relationship", ns)
        }
        sheets = book.findall("m:sheets/m:sheet", ns)
        if len(sheets) != 1:
            raise ValueError(f"expected one worksheet, found {len(sheets)}")
        target = targets[sheets[0].attrib[f"{{{REL_NS}}}id"]]
        sheet_path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        sheet = ET.fromstring(workbook.read(sheet_path))

        rows: list[dict[str, str]] = []
        for row in sheet.findall("m:sheetData/m:row", ns):
            values: dict[str, str] = {"_row": row.attrib["r"]}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib["r"]
                column = re.match(r"[A-Z]+", ref).group(0)  # type: ignore[union-attr]
                raw = cell.find("m:v", ns)
                if raw is None or raw.text is None:
                    continue
                value = raw.text
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                values[column] = value
            rows.append(values)
        return rows


def load_annotations(archive: Path) -> tuple[list[Annotation], list[str], list[str]]:
    with ZipFile(archive) as source:
        members = [item.filename for item in source.infolist() if not item.is_dir()]
        workbook = source.read(WORKBOOK_MEMBER)
    video_members = sorted(
        member for member in members if Path(member).suffix.lower() == ".mov"
    )
    video_by_stem = {Path(member).stem.upper(): member for member in video_members}
    if len(video_by_stem) != len(video_members):
        raise ValueError("duplicate video stems in source archive")

    annotations: list[Annotation] = []
    annotated_stems: set[str] = set()
    for row in _xlsx_rows(workbook):
        if row.get("A") == "序号" or not row.get("B"):
            continue
        stem = row["B"].strip().upper()
        annotated_stems.add(stem)
        if stem not in video_by_stem:
            continue
        length_match = re.fullmatch(r"线长\s*([0-9.]+)\s*cm", row.get("D", ""))
        angle_match = re.fullmatch(r"([0-9.]+)\s*°", row.get("F", ""))
        if not length_match or not angle_match:
            raise ValueError(f"unrecognized physics annotation at row {row['_row']}: {row}")
        string_length_m = float(length_match.group(1)) / 100.0
        annotations.append(
            Annotation(
                source_row=int(row["_row"]),
                trial_number=int(float(row["A"])),
                source_stem=stem,
                video_member=video_by_stem[stem],
                string_length_m=string_length_m,
                pendulum_length_m=round(string_length_m + BOB_RADIUS_M, 6),
                initial_angle_deg=float(angle_match.group(1)),
                background=row.get("H", "").strip(),
            )
        )

    missing_media = sorted(annotated_stems - set(video_by_stem))
    missing_annotation = sorted(set(video_by_stem) - annotated_stems)
    return annotations, missing_media, missing_annotation


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
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "stored_width": int(stream["width"]),
        "stored_height": int(stream["height"]),
        "rotation_deg": int(stream.get("tags", {}).get("rotate", 0)),
        "nominal_frame_rate": stream["r_frame_rate"],
        "average_frame_rate": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream.get("duration", value["format"]["duration"])),
    }


def _ball_candidate(frame: np.ndarray, *, long_string: bool) -> tuple[float, float] | None:
    """Return the most steel-ball-like Hough proposal in the pendulum annulus."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = frame[700:1900]
    reduced = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    reduced_gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    reduced_gray = cv2.GaussianBlur(reduced_gray, (7, 7), 1.3)
    circles = cv2.HoughCircles(
        reduced_gray,
        cv2.HOUGH_GRADIENT,
        1.2,
        20,
        param1=90,
        param2=17,
        minRadius=8,
        maxRadius=32,
    )
    if circles is None:
        return None

    proposals: list[tuple[float, float, float]] = []
    for reduced_x, reduced_y, reduced_radius in circles[0]:
        x = float(reduced_x * 2.0)
        y = float(reduced_y * 2.0 + 700.0)
        radius = float(reduced_radius * 2.0)
        pivot_distance = math.hypot(x - 500.0, y - 800.0)
        if long_string:
            if not 540.0 < pivot_distance < 960.0:
                continue
        elif not 150.0 < pivot_distance < 400.0:
            continue

        x0, x1 = max(0, int(x - radius)), min(frame.shape[1], int(x + radius + 1))
        y0, y1 = max(0, int(y - radius)), min(frame.shape[0], int(y + radius + 1))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= max(radius - 3.0, 3.0) ** 2
        local_hsv = hsv[y0:y1, x0:x1]
        local_gray = gray[y0:y1, x0:x1]
        saturation = float(local_hsv[:, :, 1][mask].mean())
        contrast = float(local_gray[mask].std())
        # The polished steel bob has high local contrast and substantially lower
        # saturation than a hand.  Radius is only a weak tie breaker.
        score = contrast - 0.25 * saturation - 0.02 * abs(radius - 38.0)
        proposals.append((score, x, y))
    if not proposals:
        return None
    _, x, y = max(proposals)
    return x, y


def _coarse_track(
    path: Path, *, long_string: bool
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        raise ValueError(f"invalid FPS: {path}")
    sample_step = 4
    maximum_frame = int(math.ceil(fps * 3.0))
    frame_indices: list[int] = []
    positions: list[float] = []
    vertical_positions: list[float] = []
    skin_counts: list[int] = []
    early_skin_counts: list[int] = []
    index = -1
    while True:
        ok, frame = capture.read()
        index += 1
        if not ok or index >= maximum_frame:
            break
        if index % sample_step:
            continue
        hsv = cv2.cvtColor(frame[700:], cv2.COLOR_BGR2HSV)
        skin = (
            ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 40)
            & (hsv[:, :, 2] > 50)
        )
        skin_count = int(skin.sum())
        skin_counts.append(skin_count)
        if index < int(round(fps * 0.20)):
            early_skin_counts.append(skin_count)
        proposal = _ball_candidate(frame, long_string=long_string)
        frame_indices.append(index)
        positions.append(float("nan") if proposal is None else proposal[0])
        vertical_positions.append(float("nan") if proposal is None else proposal[1])
    capture.release()
    values = np.asarray(positions, dtype=float)
    observed = np.isfinite(values)
    if observed.sum() < 20:
        raise ValueError(f"insufficient bob detections in first 3 seconds: {path}")
    values = np.interp(np.arange(len(values)), np.flatnonzero(observed), values[observed])
    window = min(11, len(values) if len(values) % 2 else len(values) - 1)
    smoothed = savgol_filter(values, window, 3)
    # The white/green cards have low/red-excluding chroma, so a sizeable warm
    # region in the lower board is a conservative indicator of the releasing
    # hand.  A held bob defines the release side more reliably than the first
    # few noisy displacement samples.
    hand_present = bool(early_skin_counts and max(early_skin_counts) >= 2_000)
    return (
        fps,
        np.asarray(frame_indices, dtype=int),
        smoothed,
        np.asarray(vertical_positions, dtype=float),
        np.asarray(skin_counts, dtype=int),
        hand_present,
    )


def _find_opposite_turning_point(
    fps: float,
    frame_indices: np.ndarray,
    x: np.ndarray,
    *,
    detected_y: np.ndarray,
    skin_counts: np.ndarray,
    hand_present: bool,
) -> tuple[int, str, dict[str, Any]]:
    trajectory_range = float(x.max() - x.min())
    if trajectory_range < 20.0:
        raise ValueError(f"detected trajectory is too small: {trajectory_range:.3f}px")
    # At 240 fps with a four-frame sampling stride, seven samples already span
    # enough of a short-pendulum half-cycle to cross the centre.  Limit the
    # initial-side estimate to the first three samples so an immediately
    # released bob cannot invert the inferred direction.
    initial_window = min(3, len(x))
    initial_x = float(np.median(x[:initial_window]))
    departure_threshold = max(12.0, trajectory_range * 0.05)
    departures = np.flatnonzero(np.abs(x - initial_x) > departure_threshold)
    if not len(departures):
        raise ValueError("bob never departs from its initial side")
    departure = int(departures[0])
    observed_direction = "right" if x[departure] > initial_x else "left"
    if hand_present:
        trajectory_centre = float((x.min() + x.max()) / 2.0)
        direction = "left" if initial_x > trajectory_centre else "right"
        direction_basis = "opposite_to_hand_held_initial_side"
    else:
        direction = observed_direction
        direction_basis = "first_observed_motion_after_source_start"
    prominence = max(8.0, trajectory_range * 0.08)
    distance = max(8, int(round(fps * 0.12 / 4.0)))
    candidates, _ = find_peaks(
        x if direction == "right" else -x,
        prominence=prominence,
        distance=distance,
    )
    candidates = candidates[candidates >= departure]
    if not len(candidates):
        raise ValueError("no opposite-side turning point found")
    candidate_audit: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons: list[str] = []
        y = float(detected_y[candidate])
        if not math.isfinite(y):
            reasons.append("bob_not_detected")
        if not 55.0 <= float(x[candidate]) <= 1025.0:
            reasons.append("bob_not_fully_inside_horizontal_frame")
        if math.isfinite(y) and not 55.0 <= y <= 1865.0:
            reasons.append("bob_not_fully_inside_vertical_frame")
        if int(skin_counts[candidate]) >= 5_000:
            reasons.append("hand_still_visible")
        candidate_audit.append({
            "frame": int(frame_indices[candidate]),
            "x_px": float(x[candidate]),
            "y_px": y if math.isfinite(y) else None,
            "skin_pixel_count": int(skin_counts[candidate]),
            "accepted": not reasons,
            "rejection_reasons": reasons,
        })
    # The workbook angle describes the release amplitude.  Skipping a full
    # extra period to wait for a hand to leave would silently relabel a damped
    # amplitude as the original angle.  Therefore only the *first* opposite
    # turning point may enter the Dataset.
    first_reasons = candidate_audit[0]["rejection_reasons"]
    if first_reasons:
        raise ValueError(
            "first opposite-side turning point is unusable: "
            + ", ".join(first_reasons)
        )
    coarse_index = int(candidates[0])
    return (
        int(frame_indices[coarse_index]),
        "maximum_x" if direction == "right" else "minimum_x",
        {
            "initial_x_px": initial_x,
            "trajectory_range_px": trajectory_range,
            "departure_frame": int(frame_indices[departure]),
            "selected_opposite_direction": direction,
            "observed_departure_direction": observed_direction,
            "direction_basis": direction_basis,
            "releasing_hand_detected": hand_present,
            "coarse_turning_frame": int(frame_indices[coarse_index]),
            "coarse_turning_x_px": float(x[coarse_index]),
            "turning_candidate_audit": candidate_audit,
        },
    )


def _refine_turning_frame(
    path: Path,
    *,
    long_string: bool,
    coarse_frame: int,
    extremum: str,
) -> tuple[int, dict[str, Any]]:
    start = max(0, coarse_frame - 24)
    end = coarse_frame + 25
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[int] = []
    positions: list[float] = []
    for frame_index in range(start, end):
        ok, frame = capture.read()
        if not ok:
            break
        proposal = _ball_candidate(frame, long_string=long_string)
        frames.append(frame_index)
        positions.append(float("nan") if proposal is None else proposal[0])
    capture.release()
    frame_array = np.asarray(frames, dtype=float)
    x = np.asarray(positions, dtype=float)
    observed = np.isfinite(x)
    if observed.sum() < 12:
        raise ValueError(f"insufficient detections near turning point: {path}")
    fit_frames = frame_array[observed]
    fit_x = x[observed]
    coefficients = np.polyfit(fit_frames, fit_x, 2)
    a, b, _ = coefficients
    if abs(a) < 1e-9:
        raise ValueError(f"degenerate turning-point fit: {path}")
    vertex = float(-b / (2.0 * a))
    if not start <= vertex < end:
        raise ValueError(f"fitted turning point outside refinement window: {path}")
    if extremum == "maximum_x" and a >= 0.0:
        raise ValueError(f"expected maximum but fitted minimum: {path}")
    if extremum == "minimum_x" and a <= 0.0:
        raise ValueError(f"expected minimum but fitted maximum: {path}")
    selected = int(round(vertex))
    return selected, {
        "refinement_window": [start, end],
        "observed_frames": int(observed.sum()),
        "quadratic_coefficients": [float(value) for value in coefficients],
        "quadratic_vertex_frame": vertex,
        "selected_start_frame": selected,
    }


def _warm_pixel_count(frame: np.ndarray) -> int:
    """Conservative releasing-hand indicator for the board below the pivot."""

    hsv = cv2.cvtColor(frame[700:], cv2.COLOR_BGR2HSV)
    skin = (
        ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 40)
        & (hsv[:, :, 2] > 50)
    )
    return int(skin.sum())


def _audit_post_start_hand(
    path: Path,
    *,
    start_frame: int,
    fps: float,
    horizon_s: float = 0.75,
) -> dict[str, Any]:
    """Ensure the hand does not remain in the canonical opening interval."""

    sample_step = 4
    end_frame = start_frame + int(round(fps * horizon_s))
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    samples: list[dict[str, int]] = []
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % sample_step:
            continue
        samples.append(
            {
                "frame": frame_index,
                "warm_pixel_count": _warm_pixel_count(frame),
            }
        )
    capture.release()
    if not samples:
        raise ValueError(f"unable to audit post-start hand interval: {path}")
    maximum = max(item["warm_pixel_count"] for item in samples)
    if maximum >= 5_000:
        raise ValueError(
            "hand visible at or after first opposite-side turning point: "
            f"maximum warm-pixel count {maximum}"
        )
    return {
        "horizon_s": horizon_s,
        "sample_step_frames": sample_step,
        "threshold_pixels": 5_000,
        "maximum_warm_pixel_count": maximum,
        "samples": samples,
        "accepted": True,
    }


def analyze_one(arguments: tuple[Annotation, str]) -> dict[str, Any]:
    annotation, media_root_value = arguments
    media_root = Path(media_root_value)
    source = media_root / f"{annotation.source_stem}.MOV"
    if not source.is_file():
        raise FileNotFoundError(source)
    long_string = annotation.string_length_m > 0.1
    probe = _probe(source)
    (
        fps,
        frame_indices,
        trajectory,
        detected_y,
        skin_counts,
        hand_present,
    ) = _coarse_track(
        source, long_string=long_string
    )
    coarse_frame, extremum, tracking = _find_opposite_turning_point(
        fps,
        frame_indices,
        trajectory,
        detected_y=detected_y,
        skin_counts=skin_counts,
        hand_present=hand_present,
    )
    selected_frame, refinement = _refine_turning_frame(
        source,
        long_string=long_string,
        coarse_frame=coarse_frame,
        extremum=extremum,
    )
    post_start_hand_audit = _audit_post_start_hand(
        source,
        start_frame=selected_frame,
        fps=fps,
    )
    return {
        "annotation": asdict(annotation),
        "normalized_physics": {
            "bob_mass": {"value": BOB_MASS_KG, "unit": "kg", "annotated": True},
            "bob_radius": {"value": BOB_RADIUS_M, "unit": "m", "annotated": True},
            "initial_angle": {
                "value": annotation.initial_angle_deg,
                "unit": "deg",
                "annotated": True,
            },
            "pendulum_length": {
                "value": annotation.pendulum_length_m,
                "unit": "m",
                "annotated": True,
            },
            "string_length": {
                "value": annotation.string_length_m,
                "unit": "m",
                "annotated": True,
            },
        },
        "normalized_appearance": {
            "background": annotation.background,
            "bob_material": "not_documented",
            "camera": "fixed",
            "capture_session": "supplement_20260731",
            "support": "laboratory_pendulum_rig",
        },
        "prompt": PROMPT,
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": sha256(source),
        "source_probe": probe,
        "alignment": {
            "canonical_first_frame_event": "first opposite-side turning point after release",
            "source_start_frame": selected_frame,
            "source_end_frame_exclusive": probe["frame_count"],
            "output_frame_count": probe["frame_count"] - selected_frame,
            "spatial_crop": None,
            "tracking": tracking,
            "refinement": refinement,
            "post_start_hand_audit": post_start_hand_audit,
        },
        "status": "candidate_needs_visual_review",
    }


def analyze_one_safe(arguments: tuple[Annotation, str]) -> dict[str, Any]:
    annotation, media_root_value = arguments
    try:
        return {"kind": "record", "value": analyze_one(arguments)}
    except Exception as error:  # keep one unusable trial from hiding the other 99
        source = Path(media_root_value) / f"{annotation.source_stem}.MOV"
        return {
            "kind": "exclusion",
            "value": {
                "annotation": asdict(annotation),
                "source_path": str(source),
                "source_size": source.stat().st_size if source.is_file() else None,
                "source_sha256": sha256(source) if source.is_file() else None,
                "source_probe": _probe(source) if source.is_file() else None,
                "status": "excluded_video_quality",
                "reason": str(error),
            },
        }


def _read_display_frames(path: Path, frame_indices: tuple[int, ...]) -> list[np.ndarray]:
    """Decode requested frames sequentially; HEVC random seeks are not frame exact."""

    capture = cv2.VideoCapture(str(path))
    wanted = set(frame_indices)
    frames: dict[int, np.ndarray] = {}
    last = max(frame_indices)
    frame_index = -1
    while frame_index < last:
        ok, frame = capture.read()
        frame_index += 1
        if not ok:
            break
        if frame_index in wanted:
            frames[frame_index] = frame
    capture.release()
    missing = [index for index in frame_indices if index not in frames]
    if missing:
        raise ValueError(f"unable to decode frames {missing}: {path}")
    return [frames[index] for index in frame_indices]


def _prepare_review_row(record: dict[str, Any]) -> dict[str, Any]:
    cv2.setNumThreads(1)
    start = int(record["alignment"]["source_start_frame"])
    end = int(record["alignment"]["source_end_frame_exclusive"])
    span = end - start
    indices = (
        start,
        start + span // 4,
        start + span // 2,
        start + 3 * span // 4,
        end - 1,
    )
    frames = _read_display_frames(Path(record["source_path"]), indices)
    return {
        "source_stem": record["annotation"]["source_stem"],
        "indices": indices,
        "frames": [
            cv2.resize(frame, (216, 384), interpolation=cv2.INTER_AREA)
            for frame in frames
        ],
    }


def write_review_sheets(records: list[dict[str, Any]], review_dir: Path) -> None:
    """Write start/interior/end contact sheets for the required visual audit."""

    review_dir.mkdir(parents=True, exist_ok=True)
    labels = ("start", "quarter", "middle", "three_quarters", "end")
    rows_per_page = 6
    cell_width, cell_height = 216, 384
    label_height = 28
    with ProcessPoolExecutor(max_workers=min(8, len(records))) as executor:
        review_rows = list(executor.map(_prepare_review_row, records))
    for page_start in range(0, len(review_rows), rows_per_page):
        page_records = review_rows[page_start : page_start + rows_per_page]
        canvas = np.zeros(
            (len(page_records) * (cell_height + label_height), len(labels) * cell_width, 3),
            dtype=np.uint8,
        )
        for row, record in enumerate(page_records):
            indices = record["indices"]
            decoded = record["frames"]
            for column, (label, frame_index, frame) in enumerate(
                zip(labels, indices, decoded)
            ):
                x0 = column * cell_width
                y0 = row * (cell_height + label_height) + label_height
                canvas[y0 : y0 + cell_height, x0 : x0 + cell_width] = frame
                caption = (
                    f"{record['source_stem']} {label} f{frame_index}"
                    if column == 0
                    else f"{label} f{frame_index}"
                )
                cv2.putText(
                    canvas,
                    caption,
                    (x0 + 4, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        page_number = page_start // rows_per_page + 1
        output = review_dir / f"page_{page_number:02d}.jpg"
        if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"unable to write review sheet: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    annotations, missing_media, missing_annotation = load_annotations(args.archive)
    if missing_annotation:
        raise ValueError(f"videos without workbook rows: {missing_annotation}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        analyzed = list(
            executor.map(
                analyze_one_safe,
                [(annotation, str(args.media_root)) for annotation in annotations],
            )
        )
    records = [item["value"] for item in analyzed if item["kind"] == "record"]
    exclusions = [item["value"] for item in analyzed if item["kind"] == "exclusion"]
    records.sort(key=lambda item: item["annotation"]["source_stem"])
    exclusions.sort(key=lambda item: item["annotation"]["source_stem"])
    result = {
        "schema_version": "1.0",
        "import_id": "pendulum_supplement_20260804",
        "source_archive": str(args.archive),
        "source_archive_size": args.archive.stat().st_size,
        "source_archive_sha256": sha256(args.archive),
        "source_workbook_member": WORKBOOK_MEMBER,
        "field_decisions": {
            "视频": "exact source-member mapping key",
            "长度": "physics.string_length; cm converted to m",
            "角度": "physics.initial_angle; degrees retained",
            "球半径": "user-confirmed 1 cm; physics.bob_radius",
            "球质量": "user-confirmed 31.5 g; physics.bob_mass",
            "总摆长": "derived string_length + bob_radius; physics.pendulum_length",
            "背景": "appearance.background only; excluded from structured physics and prompt",
        },
        "workbook_annotation_rows": len(annotations) + len(missing_media),
        "matched_video_rows": len(records),
        "excluded_video_rows": len(exclusions),
        "ignored_rows_missing_media": missing_media,
        "videos_missing_annotation": missing_annotation,
        "video_quality_exclusions": exclusions,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.review_dir is not None:
        write_review_sheets(records, args.review_dir)


if __name__ == "__main__":
    main()
