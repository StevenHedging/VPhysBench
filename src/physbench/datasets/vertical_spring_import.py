from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Literal, Sequence
import zipfile

import cv2
import numpy as np
from openpyxl import load_workbook


Direction = Literal["above", "below"]


@dataclass(frozen=True)
class TrialAnnotation:
    trial_id: str
    spring_id: str
    video_name: str
    displacement_mm: float
    workbook_row: int

    @property
    def direction(self) -> Direction:
        return "below" if self.displacement_mm > 0 else "above"

    @property
    def displacement_m(self) -> float:
        return abs(self.displacement_mm) / 1000.0


@dataclass(frozen=True)
class SourceMember:
    member: str
    basename: str
    size: int
    crc32: int


@dataclass(frozen=True)
class MappedTrial:
    trial: TrialAnnotation
    source: SourceMember
    duplicate_members: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntakeExclusion:
    trial_id: str
    video_name: str
    reason: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntakeResult:
    accepted: tuple[MappedTrial, ...]
    exclusions: tuple[IntakeExclusion, ...]


@dataclass(frozen=True)
class BallDetection:
    center_x: float
    center_y: float
    radius: float
    score: float


@dataclass(frozen=True)
class TrackSample:
    frame_index: int
    time_s: float
    center_x: float
    center_y: float
    radius: float
    score: float


@dataclass(frozen=True)
class TurningFrameCandidate:
    frame_index: int
    time_s: float
    center_x: float
    center_y: float
    radius: float
    observed_period_s: float
    track_coverage: float


@dataclass(frozen=True)
class VideoAnalysis:
    candidate: TurningFrameCandidate
    full_resolution_ball: BallDetection
    frame_count: int
    displayed_width: int
    displayed_height: int
    analysis_stride: int


@dataclass(frozen=True)
class CaseDraft:
    case_id: str
    case_directory: Path
    audit: dict[str, object]


def select_spring_test_ids(
    records: Sequence[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[str]:
    """Select a bounded, balanced, source-group-safe ID test subset."""
    if limit < 1:
        raise ValueError("spring test limit must be positive")
    strata: dict[tuple[str, float], dict[str, list[str]]] = {}
    for record in records:
        stratum = (
            str(record["direction"]),
            abs(float(record["signed_displacement_mm"])),
        )
        source_group = str(record["source_group"])
        strata.setdefault(stratum, {}).setdefault(source_group, []).append(
            str(record["case_id"])
        )
    queues = {
        stratum: [sorted(case_ids) for _, case_ids in sorted(groups.items())]
        for stratum, groups in sorted(strata.items())
    }
    strata_by_direction: dict[str, list[tuple[str, float]]] = {}
    for stratum in queues:
        strata_by_direction.setdefault(stratum[0], []).append(stratum)
    for direction, direction_strata in strata_by_direction.items():
        ordered = sorted(direction_strata, key=lambda item: item[1])
        spread: list[tuple[str, float]] = []
        while ordered:
            spread.append(ordered.pop(0))
            if ordered:
                spread.append(ordered.pop())
        strata_by_direction[direction] = spread
    stratum_order: list[tuple[str, float]] = []
    depth = 0
    while any(depth < len(items) for items in strata_by_direction.values()):
        for direction in sorted(strata_by_direction):
            items = strata_by_direction[direction]
            if depth < len(items):
                stratum_order.append(items[depth])
        depth += 1
    selected: list[str] = []
    while len(selected) < limit:
        progressed = False
        for stratum in stratum_order:
            groups = queues[stratum]
            while len(groups) > 1 and len(selected) + len(groups[0]) > limit:
                groups.pop(0)
            if len(groups) <= 1:
                continue
            selected.extend(groups.pop(0))
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _normalized_header(value: object) -> str:
    return "".join(str(value or "").strip().lower().split())


def _column_index(headers: Sequence[object], *names: str) -> int:
    normalized = [_normalized_header(value) for value in headers]
    for name in names:
        target = _normalized_header(name)
        if target in normalized:
            return normalized.index(target)
    raise ValueError(f"missing required workbook column: {names[0]}")


def load_trial_annotations(workbook: Path) -> list[TrialAnnotation]:
    document = load_workbook(workbook, data_only=True, read_only=True)
    if "Trial记录" not in document.sheetnames:
        raise ValueError("workbook has no Trial记录 sheet")
    sheet = document["Trial记录"]
    rows = list(sheet.iter_rows(values_only=True))
    header_offset = next(
        (
            index
            for index, row in enumerate(rows)
            if _normalized_header("Trial编号")
            in {_normalized_header(value) for value in row}
            or _normalized_header("Trial ID")
            in {_normalized_header(value) for value in row}
        ),
        None,
    )
    if header_offset is None:
        raise ValueError("Trial记录 sheet has no recognized header row")
    headers = rows[header_offset]
    trial_col = _column_index(headers, "Trial ID", "TrialID", "Trial编号")
    spring_col = _column_index(headers, "弹簧ID", "Spring ID", "弹簧编号")
    video_col = _column_index(
        headers,
        "视频文件",
        "视频名",
        "Video",
        "视频文件名",
    )
    displacement_col = _column_index(
        headers,
        "初始位移 (mm)",
        "初始位移(mm)",
        "位移(mm)",
        "释放长度-平衡长度_mm",
    )
    result: list[TrialAnnotation] = []
    for workbook_row, row in enumerate(
        rows[header_offset + 1 :],
        start=header_offset + 2,
    ):
        trial_id = str(row[trial_col] or "").strip()
        spring_id = str(row[spring_col] or "").strip()
        video_name = str(row[video_col] or "").strip()
        displacement = row[displacement_col]
        if not trial_id:
            continue
        if not video_name and displacement in (None, ""):
            continue
        if not spring_id or not video_name or displacement in (None, ""):
            raise ValueError(f"incomplete required Trial row {workbook_row}")
        if spring_id != "S01":
            raise ValueError(
                f"unsupported spring ID {spring_id!r} at row {workbook_row}"
            )
        displacement_mm = float(displacement)
        if not math.isfinite(displacement_mm) or displacement_mm == 0:
            raise ValueError(f"invalid displacement at row {workbook_row}")
        normalized_video_name = PurePosixPath(video_name).name
        if not PurePosixPath(normalized_video_name).suffix:
            normalized_video_name = f"{normalized_video_name}.MOV"
        result.append(
            TrialAnnotation(
                trial_id=trial_id,
                spring_id=spring_id,
                video_name=normalized_video_name,
                displacement_mm=displacement_mm,
                workbook_row=workbook_row,
            )
        )
    return result


def inventory_archive(archive: Path) -> list[SourceMember]:
    with zipfile.ZipFile(archive) as handle:
        members = [
            SourceMember(
                member=info.filename,
                basename=PurePosixPath(info.filename).name,
                size=info.file_size,
                crc32=info.CRC,
            )
            for info in handle.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".mov")
        ]
    return sorted(members, key=lambda item: (item.basename.lower(), item.member))


def map_trials_to_sources(
    trials: Sequence[TrialAnnotation],
    sources: Sequence[SourceMember],
) -> IntakeResult:
    sources_by_name: dict[str, list[SourceMember]] = {}
    for source in sources:
        sources_by_name.setdefault(source.basename.casefold(), []).append(source)
    accepted: list[MappedTrial] = []
    exclusions: list[IntakeExclusion] = []
    for trial in trials:
        matches = sources_by_name.get(trial.video_name.casefold(), [])
        if not matches:
            exclusions.append(
                IntakeExclusion(
                    trial.trial_id,
                    trial.video_name,
                    "workbook_video_missing_from_archive",
                )
            )
            continue
        signatures = {(item.size, item.crc32) for item in matches}
        if len(signatures) != 1:
            exclusions.append(
                IntakeExclusion(
                    trial.trial_id,
                    trial.video_name,
                    "ambiguous_source_members",
                    tuple(item.member for item in matches),
                )
            )
            continue
        selected = sorted(matches, key=lambda item: item.member)[0]
        accepted.append(
            MappedTrial(
                trial=trial,
                source=selected,
                duplicate_members=tuple(
                    item.member for item in matches if item.member != selected.member
                ),
            )
        )
    return IntakeResult(tuple(accepted), tuple(exclusions))


def detect_ball(
    frame: np.ndarray,
    previous: BallDetection | None = None,
) -> BallDetection:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("ball detection requires a BGR frame")
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    minimum_radius = max(8, round(width * 0.07))
    maximum_radius = max(minimum_radius + 2, round(width * 0.15))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, height // 12),
        param1=100,
        param2=24,
        minRadius=minimum_radius,
        maxRadius=maximum_radius,
    )
    if circles is None:
        raise ValueError("ball_not_detected")
    candidates: list[BallDetection] = []
    for center_x, center_y, radius in circles[0]:
        if not (0.25 * width <= center_x <= 0.85 * width):
            continue
        if not (0.25 * height <= center_y <= 0.95 * height):
            continue
        if previous is None:
            score = (
                float(radius)
                - abs(float(center_x) - 0.60 * width) * 0.03
                + float(center_y) * 0.002
            )
        else:
            distance = math.hypot(
                float(center_x) - previous.center_x,
                float(center_y) - previous.center_y,
            )
            radius_delta = abs(float(radius) - previous.radius)
            score = -distance - 2.0 * radius_delta
        candidates.append(
            BallDetection(
                float(center_x),
                float(center_y),
                float(radius),
                float(score),
            )
        )
    if not candidates:
        raise ValueError("ball_not_detected")
    return max(candidates, key=lambda item: item.score)


def _median_filter(values: np.ndarray, width: int = 5) -> np.ndarray:
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + width]) for index in range(len(values))],
        dtype=float,
    )


def _odd_window(duration_s: float, sample_interval_s: float, maximum: int) -> int:
    width = max(1, min(maximum, round(duration_s / sample_interval_s)))
    if width % 2 == 0:
        width = width - 1 if width == maximum else width + 1
    return max(1, width)


def detect_release_return(
    track: Sequence[TrackSample],
    direction: Direction,
    theoretical_period_s: float = 0.789924128416829,
) -> TurningFrameCandidate:
    if direction not in ("above", "below"):
        raise ValueError(f"unsupported release direction: {direction!r}")
    if len(track) < 3:
        raise ValueError("insufficient_track_samples")
    ordered = sorted(track, key=lambda item: item.frame_index)
    times = np.asarray([item.time_s for item in ordered], dtype=float)
    centers_y = np.asarray([item.center_y for item in ordered], dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("invalid_track_timestamps")
    vertical_span = float(np.ptp(centers_y))
    if vertical_span < 10.0:
        raise ValueError("insufficient_vertical_motion")
    frame_deltas = np.diff([item.frame_index for item in ordered])
    analysis_stride = max(1, round(float(np.median(frame_deltas))))
    expected_samples = (
        (ordered[-1].frame_index - ordered[0].frame_index) // analysis_stride
    ) + 1
    coverage = len(ordered) / expected_samples
    if coverage < 0.90:
        raise ValueError("insufficient_track_coverage")

    sample_interval_s = float(np.median(np.diff(times)))
    median_width = _odd_window(0.025, sample_interval_s, 5)
    average_width = _odd_window(0.0375, sample_interval_s, 9)
    smooth = _median_filter(centers_y, median_width)
    average_radius = average_width // 2
    smooth = np.convolve(
        np.pad(smooth, (average_radius, average_radius), mode="edge"),
        np.ones(average_width, dtype=float) / average_width,
        mode="valid",
    )
    anchor_window = times <= times[0] + 0.25 * theoretical_period_s
    anchor = float(np.median(smooth[anchor_window]))
    departure = np.abs(smooth - anchor) >= max(3.0, 0.04 * vertical_span)
    sustained = np.convolve(
        departure.astype(np.uint8),
        np.ones(3, dtype=np.uint8),
        mode="valid",
    )
    onset_candidates = np.flatnonzero(sustained == 3)
    if not len(onset_candidates):
        raise ValueError("motion_onset_not_detected")
    onset_index = max(0, int(onset_candidates[0]) - 2)
    start_time = float(times[onset_index])
    opposite = (times >= start_time + 0.25 * theoretical_period_s) & (
        times <= start_time + 0.75 * theoretical_period_s
    )
    same_side = (times >= start_time + 0.60 * theoretical_period_s) & (
        times <= start_time + 1.35 * theoretical_period_s
    )
    if not np.any(opposite) or not np.any(same_side):
        raise ValueError("insufficient_cycle_coverage")
    opposite_values = smooth[opposite]
    same_values = smooth[same_side]
    if direction == "below":
        opposite_extreme = float(np.min(opposite_values))
        coarse_local = int(np.argmax(same_values))
        same_extreme = float(same_values[coarse_local])
        separation = same_extreme - opposite_extreme
    else:
        opposite_extreme = float(np.max(opposite_values))
        coarse_local = int(np.argmin(same_values))
        same_extreme = float(same_values[coarse_local])
        separation = opposite_extreme - same_extreme
    if separation < max(8.0, 0.45 * vertical_span):
        raise ValueError("insufficient_cycle_extrema")
    same_indices = np.flatnonzero(same_side)
    coarse_index = int(same_indices[coarse_local])
    lower = max(0, coarse_index - 8)
    upper = min(len(ordered), coarse_index + 9)
    local = centers_y[lower:upper]
    refined = lower + int(np.argmax(local) if direction == "below" else np.argmin(local))
    selected = ordered[refined]
    return TurningFrameCandidate(
        frame_index=selected.frame_index,
        time_s=selected.time_s,
        center_x=selected.center_x,
        center_y=selected.center_y,
        radius=selected.radius,
        observed_period_s=selected.time_s - start_time,
        track_coverage=coverage,
    )


def ball_mask(
    shape: tuple[int, int],
    detection: BallDetection,
) -> np.ndarray:
    height, width = shape
    if height <= 0 or width <= 0 or detection.radius <= 0:
        raise ValueError("invalid ball mask geometry")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(
        mask,
        (round(detection.center_x), round(detection.center_y)),
        round(detection.radius),
        1,
        thickness=-1,
        lineType=cv2.LINE_8,
    )
    return mask


def build_physics(case_id: str, signed_displacement_mm: float) -> dict[str, object]:
    displacement_m = abs(float(signed_displacement_mm)) / 1000.0
    if not case_id or not math.isfinite(displacement_m) or displacement_m <= 0:
        raise ValueError("physics requires a Case ID and positive displacement")
    return {
        "case_id": case_id,
        "physics": {
            "environment": {
                "gravity_acceleration": {
                    "symbol": "g",
                    "unit": "m/s^2",
                    "value": 9.80665,
                },
                "natural_spring_length": {
                    "symbol": "L_0",
                    "unit": "m",
                    "value": 0.068,
                },
                "spring_stiffness": {
                    "symbol": "k",
                    "unit": "N/m",
                    "value": 32.6213467096774,
                },
            },
            "objects": {
                "object_1": {
                    "initial_displacement": {
                        "symbol": "x_0",
                        "unit": "m",
                        "value": displacement_m,
                    },
                    "mass": {
                        "symbol": "m",
                        "unit": "kg",
                        "value": 0.5156,
                    },
                    "radius": {
                        "symbol": "r",
                        "unit": "m",
                        "value": 0.025,
                    },
                }
            },
        },
        "scene_id": "vertical_spring_oscillator",
    }


def build_caption(case_id: str, direction: Direction) -> dict[str, str]:
    if not case_id:
        raise ValueError("caption requires a Case ID")
    if direction not in ("above", "below"):
        raise ValueError(f"unsupported release direction: {direction!r}")
    return {
        "case_id": case_id,
        "scene_id": "vertical_spring_oscillator",
        "caption": (
            "A vertically suspended oscillator of total moving mass m and ball "
            "radius r is attached to a spring with stiffness k and natural "
            "length L_0 under gravitational acceleration g. At the first "
            f"frame, the oscillator is at displacement x_0 {direction} "
            "equilibrium and starts from rest, then undergoes vertical free "
            "oscillation."
        ),
    }


def probe_frame_timestamps(video: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in payload.get("frames", [])
        if "best_effort_timestamp_time" in frame
    ]
    if not timestamps or not all(
        later > earlier for earlier, later in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(f"video has invalid presentation timestamps: {video}")
    return timestamps


def trim_video_exact(source: Path, start_frame: int, target: Path) -> None:
    if start_frame < 0:
        raise ValueError("start frame must be non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"trim=start_frame={start_frame},setpts=PTS-STARTPTS",
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-metadata:s:v:0",
            "rotate=0",
            str(target),
        ],
        check=True,
    )


def analyze_video(
    video: Path,
    direction: Direction,
    *,
    analysis_stride: int = 8,
    theoretical_period_s: float = 0.789924128416829,
) -> VideoAnalysis:
    if analysis_stride < 1:
        raise ValueError("analysis stride must be positive")
    timestamps = probe_frame_timestamps(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video}")
    track: list[TrackSample] = []
    previous: BallDetection | None = None
    source_index = 0
    analysis_width = 270
    analysis_height = 480
    while source_index < len(timestamps):
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % analysis_stride == 0:
            small = cv2.resize(
                frame,
                (analysis_width, analysis_height),
                interpolation=cv2.INTER_AREA,
            )
            try:
                detected = detect_ball(small, previous)
            except ValueError:
                pass
            else:
                previous = detected
                track.append(
                    TrackSample(
                        source_index,
                        timestamps[source_index],
                        detected.center_x,
                        detected.center_y,
                        detected.radius,
                        detected.score,
                    )
                )
        source_index += 1
    capture.release()
    if source_index != len(timestamps):
        raise ValueError(
            f"decoded frame count {source_index} does not match timestamps "
            f"{len(timestamps)}"
        )
    candidate = detect_release_return(
        track,
        direction,
        theoretical_period_s,
    )

    refinement_start = max(0, candidate.frame_index - analysis_stride)
    refinement_end = min(
        len(timestamps) - 1,
        candidate.frame_index + analysis_stride,
    )
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, refinement_start)
    refined: list[TrackSample] = []
    seed = BallDetection(
        candidate.center_x,
        candidate.center_y,
        candidate.radius,
        1.0,
    )
    for frame_index in range(refinement_start, refinement_end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(
            frame,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
        try:
            detected = detect_ball(small, seed)
        except ValueError:
            continue
        refined.append(
            TrackSample(
                frame_index,
                timestamps[frame_index],
                detected.center_x,
                detected.center_y,
                detected.radius,
                detected.score,
            )
        )
    capture.release()
    if not refined:
        raise ValueError("turning_frame_refinement_failed")
    selected = (
        max(refined, key=lambda item: item.center_y)
        if direction == "below"
        else min(refined, key=lambda item: item.center_y)
    )
    candidate = replace(
        candidate,
        frame_index=selected.frame_index,
        time_s=selected.time_s,
        center_x=selected.center_x,
        center_y=selected.center_y,
        radius=selected.radius,
    )

    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
    ok, full_frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("could not decode selected full-resolution frame")
    displayed_height, displayed_width = full_frame.shape[:2]
    full_seed = BallDetection(
        candidate.center_x * displayed_width / analysis_width,
        candidate.center_y * displayed_height / analysis_height,
        candidate.radius
        * min(
            displayed_width / analysis_width,
            displayed_height / analysis_height,
        ),
        candidate.track_coverage,
    )
    full_detection = detect_ball(full_frame, full_seed)
    return VideoAnalysis(
        candidate=candidate,
        full_resolution_ball=full_detection,
        frame_count=len(timestamps),
        displayed_width=displayed_width,
        displayed_height=displayed_height,
        analysis_stride=analysis_stride,
    )


def _number_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _source_number(source: SourceMember) -> str:
    stem = PurePosixPath(source.basename).stem
    normalized = stem.split("(", 1)[0]
    if normalized.upper().startswith("IMG_"):
        normalized = normalized[4:]
    if not normalized.isdigit():
        raise ValueError(f"source member has no IMG number: {source.basename}")
    return normalized


def _write_json_document(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_case(
    source_video: Path,
    trial: TrialAnnotation,
    source: SourceMember,
    analysis: VideoAnalysis,
    repo_root: Path,
) -> CaseDraft:
    magnitude = _number_token(abs(trial.displacement_mm))
    image_number = _source_number(source)
    case_id = (
        f"vertical_spring_s01_x{magnitude}mm_{trial.direction}_img_"
        f"{image_number}"
    )
    directory_name = (
        f"spring_m515p6g_r25mm_x{magnitude}mm_{trial.direction}_img"
        f"{image_number}"
    )
    case_directory = (
        repo_root
        / "datasets"
        / "assets"
        / "vertical_spring_oscillator"
        / directory_name
    )
    canonical = case_directory / "canonical"
    masks = canonical / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    reference = canonical / "reference.mp4"
    trim_video_exact(
        source_video,
        analysis.candidate.frame_index,
        reference,
    )
    source_times = probe_frame_timestamps(source_video)
    canonical_times = probe_frame_timestamps(reference)
    expected_count = len(source_times) - analysis.candidate.frame_index
    if len(canonical_times) != expected_count:
        raise ValueError(
            f"canonical frame count {len(canonical_times)} != {expected_count}"
        )
    capture = cv2.VideoCapture(str(reference))
    ok, first_frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("could not decode canonical frame zero")
    first_frame_path = canonical / "first_frame.png"
    if not cv2.imwrite(str(first_frame_path), first_frame):
        raise ValueError("could not write canonical first frame")
    frame_height, frame_width = first_frame.shape[:2]
    scale_x = frame_width / analysis.displayed_width
    scale_y = frame_height / analysis.displayed_height
    detection = BallDetection(
        analysis.full_resolution_ball.center_x * scale_x,
        analysis.full_resolution_ball.center_y * scale_y,
        analysis.full_resolution_ball.radius * min(scale_x, scale_y),
        analysis.full_resolution_ball.score,
    )
    binary = ball_mask((frame_height, frame_width), detection)
    cv2.imwrite(str(masks / "01.png"), binary * 255)
    np.savez_compressed(
        masks / "01.npz",
        masks=binary[np.newaxis, ...],
        mask_ids=np.asarray(["01"]),
        object_ids=np.asarray(["object_1"]),
        frame_index=np.asarray(0, dtype=np.int64),
    )
    ys, xs = np.nonzero(binary)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    relative_case = case_directory.relative_to(repo_root / "datasets").as_posix()
    manifest = {
        "case_id": case_id,
        "frame_index": 0,
        "frame_scope": "first_frame_only",
        "generator": {
            "id": "reviewed_hough_circle_steel_ball_v1",
            "model": "geometric circle fit",
        },
        "image_shape_hw": [frame_height, frame_width],
        "instances": [
            {
                "area_pixels": int(binary.sum()),
                "asset": f"{relative_case}/canonical/masks/01.png",
                "bbox_xyxy": bbox,
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "entity_class": "steel_ball",
                "mask_id": "01",
                "npz_asset": f"{relative_case}/canonical/masks/01.npz",
                "object_id": "object_1",
                "physics_keys": [
                    "objects.object_1.initial_displacement",
                    "objects.object_1.mass",
                    "objects.object_1.radius",
                ],
                "segmentation": {
                    "anchor_circle_xyr": [
                        detection.center_x,
                        detection.center_y,
                        detection.radius,
                    ]
                },
            }
        ],
        "ordering": "single_object",
        "scene_id": "vertical_spring_oscillator",
        "schema_version": "1.2",
        "source_first_frame": f"{relative_case}/canonical/first_frame.png",
        "storage": {
            "model": {
                "array_key": "masks",
                "asset_pattern": (
                    f"{relative_case}/canonical/masks/{{mask_id}}.npz"
                ),
                "dtype": "uint8",
                "layout": "1HW",
                "values": [0, 1],
            },
            "visualization": {
                "asset_pattern": (
                    f"{relative_case}/canonical/masks/{{mask_id}}.png"
                ),
                "dtype": "uint8",
                "values": [0, 255],
            },
        },
    }
    _write_json_document(masks / "manifest.json", manifest)
    _write_json_document(case_directory / "physics.json", build_physics(case_id, trial.displacement_mm))
    _write_json_document(case_directory / "caption.json", build_caption(case_id, trial.direction))
    audit: dict[str, object] = {
        "asset_directory": case_directory.relative_to(
            repo_root / "datasets"
        ).as_posix(),
        "case_id": case_id,
        "scene_id": "vertical_spring_oscillator",
        "trial_id": trial.trial_id,
        "workbook_row": trial.workbook_row,
        "source_member": source.member,
        "source_size": source.size,
        "source_crc32": source.crc32,
        "source_group": source.member,
        "source_start_frame": analysis.candidate.frame_index,
        "source_start_time_s": analysis.candidate.time_s,
        "source_frame_count": analysis.frame_count,
        "canonical_frame_count": len(canonical_times),
        "direction": trial.direction,
        "signed_displacement_mm": trial.displacement_mm,
        "detector": {
            "analysis_stride": analysis.analysis_stride,
            "full_resolution_ball_xyr": [
                analysis.full_resolution_ball.center_x,
                analysis.full_resolution_ball.center_y,
                analysis.full_resolution_ball.radius,
            ],
            "observed_period_s": analysis.candidate.observed_period_s,
            "track_coverage": analysis.candidate.track_coverage,
        },
        "alignment": {
            "canonical_first_frame_event": (
                "first return to the release-side turning point after one "
                "complete oscillation"
            ),
            "source_end_frame_exclusive": analysis.frame_count,
            "source_start_frame": analysis.candidate.frame_index,
            "spatial_crop": None,
            "tail_trim": None,
        },
    }
    return CaseDraft(case_id, case_directory, audit)
