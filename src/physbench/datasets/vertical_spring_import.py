from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
from typing import Literal, Sequence
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
    minimum_radius = max(8, round(height * 0.025))
    maximum_radius = max(minimum_radius + 2, round(height * 0.085))
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
