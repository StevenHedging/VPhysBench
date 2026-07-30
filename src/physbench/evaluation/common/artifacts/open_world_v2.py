"""Best-effort visual and audit artifacts for open-world v2 evaluators.

The scoring protocol deliberately does not depend on this module.  Scene
evaluators should finish their numeric comparison first and then call
``write_open_world_v2_artifacts``.  Rendering failures are sealed in the
local manifest and never alter the already-computed case score.

The external MP4 shows the reference and prediction side by side.  It keeps
all expected entity identities and all observer tracks visible, including
unmatched and tentative tracks, and adds a compact cardinality/lifecycle
dashboard.  The companion JSON is intentionally scene-neutral and contains
the lossless (apart from masks) object-centric audit.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ....io import canonical_sha256, sha256_file, write_json
from ..entities.observer import (
    OpenWorldObservation,
    OpenWorldTrack,
)
from ..entities.v2 import (
    ExpectedEntityTimeline,
    ObjectCentricComparisonV2,
)


DEFAULT_EXTERNAL_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark/evaluation_visualizations"
)
DEFAULT_EXTERNAL_ROOT_ENV = "PHYSBENCH_VISUALIZATION_ROOT"
DEFAULT_REPOSITORY_LINK = "visualizations"
ARTIFACT_PROTOCOL_ID = "open_world_v2_artifacts"
ARTIFACT_PROTOCOL_VERSION = "1.0"
LOCAL_MANIFEST_NAME = "open_world_v2_artifact_manifest.json"

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: object, *, fallback: str) -> str:
    """Return one non-traversing path component with a stable suffix."""

    raw = str(value or "").strip()
    if not raw:
        return fallback
    normalized = _SAFE_COMPONENT.sub("_", raw).strip("._")
    normalized = normalized[:96]
    if not normalized:
        normalized = fallback
    if normalized == raw and normalized not in {".", ".."}:
        return normalized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def _artifact_directory(
    request: Any,
    *,
    config: Mapping[str, Any],
) -> tuple[Path, Path, str]:
    environment_name = str(
        config.get("external_root_env", DEFAULT_EXTERNAL_ROOT_ENV)
    ).strip()
    configured_root = (
        config.get("external_root", DEFAULT_EXTERNAL_ROOT)
        or DEFAULT_EXTERNAL_ROOT
    )
    root_value = (
        os.environ.get(environment_name)
        if environment_name
        else None
    ) or configured_root
    root = Path(str(root_value)).expanduser().resolve()
    namespace = _safe_component(
        config.get("namespace", "scene_default_v6"),
        fallback="scene_default_v6",
    )
    scene_id = _safe_component(
        request.case.get("scene_id", "unknown_scene"),
        fallback="unknown_scene",
    )
    case_id = _safe_component(
        request.case.get("case_id", "unknown_case"),
        fallback="unknown_case",
    )
    job_id = _safe_component(
        request.job.get("job_id", "unknown_job"),
        fallback="unknown_job",
    )
    identity = hashlib.sha256(
        str(request.artifact_dir.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    relative = (
        Path(namespace)
        / scene_id
        / case_id
        / f"{job_id}-{identity}"
    )
    return root / relative, relative, str(root)


def _json_safe(value: Any) -> Any:
    """Convert evaluator diagnostics to deterministic JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _identity_color(value: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    hue = int(digest[0]) * 179 // 255
    hsv = np.asarray([[[hue, 205, 245]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("video frames must be grayscale, BGR, or BGRA")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("video frames must have positive dimensions")
    return np.array(array, copy=True)


def _normalize_mask(
    mask: np.ndarray | None,
    *,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if mask is None:
        return None
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("object masks must be two-dimensional")
    output = binary.astype(np.uint8) * 255
    if output.shape != shape:
        output = cv2.resize(
            output,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return output


def _letterbox(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    target_width = max(1, int(round(source_width * scale)))
    target_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=interpolation,
    )
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - target_width) // 2
    y = (height - target_height) // 2
    output[y : y + target_height, x : x + target_width] = resized
    return output, scale, (x, y), (target_width, target_height)


def _mask_to_canvas(
    mask: np.ndarray | None,
    *,
    source_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    offset_xy: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray | None:
    normalized = _normalize_mask(mask, shape=source_shape)
    if normalized is None:
        return None
    target_width, target_height = target_size
    resized = cv2.resize(
        normalized,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros(canvas_shape, dtype=np.uint8)
    x, y = offset_xy
    output[y : y + target_height, x : x + target_width] = resized
    return output


def _points_to_canvas(
    points: np.ndarray,
    *,
    scale: float,
    offset_xy: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    return values * scale + np.asarray(offset_xy, dtype=np.float64)


def _point_to_canvas(
    point: Sequence[float],
    *,
    scale: float,
    offset_xy: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (2,):
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return value * scale + np.asarray(offset_xy, dtype=np.float64)


def _blend_mask(
    frame: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.32,
) -> np.ndarray:
    if mask is None:
        return frame
    binary = _normalize_mask(mask, shape=frame.shape[:2])
    assert binary is not None
    selected = binary > 0
    if not np.any(selected):
        return frame
    output = frame.copy()
    color_array = np.asarray(color, dtype=np.float64)
    output[selected] = np.clip(
        (1.0 - alpha) * output[selected].astype(np.float64)
        + alpha * color_array,
        0,
        255,
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output


def _ascii(value: object, *, maximum: int = 180) -> str:
    text = str(value).encode("ascii", "replace").decode("ascii")
    if len(text) <= maximum:
        return text
    return text[: max(maximum - 3, 0)] + "..."


def _label(
    frame: np.ndarray,
    text: str,
    xy: Sequence[float],
    color: tuple[int, int, int],
) -> None:
    point = np.asarray(xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return
    x = int(np.clip(round(float(point[0])), 0, frame.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, frame.shape[0] - 1))
    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        _ascii(text, maximum=72),
        (min(x + 7, max(frame.shape[1] - 220, 0)), max(y - 7, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        color,
        1,
        cv2.LINE_AA,
    )


def _header(
    frame: np.ndarray,
    primary: str,
    secondary: str = "",
) -> None:
    height = 56 if secondary else 34
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        _ascii(primary),
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if secondary:
        cv2.putText(
            frame,
            _ascii(secondary),
            (10, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )


def _finite_points(
    values: np.ndarray,
    *,
    through: int,
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)[: through + 1, :2]
    if rows.ndim != 2 or rows.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    return rows[np.isfinite(rows).all(axis=1)]


def _draw_trajectory(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return
    clipped = np.asarray(points, dtype=np.float64).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, frame.shape[1] - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, frame.shape[0] - 1)
    cv2.polylines(
        frame,
        [np.rint(clipped).astype(np.int32)],
        False,
        color,
        2,
        cv2.LINE_AA,
    )


def _reference_panel(
    frame: np.ndarray,
    *,
    frame_index: int,
    timelines: Sequence[ExpectedEntityTimeline],
    audit: Mapping[str, Any],
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    source = _normalize_frame(frame)
    output, scale, offset, target_size = _letterbox(
        source,
        width=panel_width,
        height=panel_height,
    )
    for timeline in timelines:
        entity_id = timeline.entity_id
        color = _identity_color(f"entity:{entity_id}")
        expected_now = bool(timeline.expected_exists[frame_index])
        mask = (
            timeline.reference_masks[frame_index]
            if expected_now
            else None
        )
        canvas_mask = _mask_to_canvas(
            mask,
            source_shape=source.shape[:2],
            canvas_shape=output.shape[:2],
            offset_xy=offset,
            target_size=target_size,
        )
        output = _blend_mask(output, canvas_mask, color)
        points = _finite_points(timeline.reference_xy, through=frame_index)
        _draw_trajectory(
            output,
            _points_to_canvas(
                points,
                scale=scale,
                offset_xy=offset,
            ),
            color,
        )
        xy = timeline.reference_xy[frame_index, :2]
        if expected_now and np.isfinite(xy).all():
            _label(
                output,
                f"GT:{entity_id}",
                _point_to_canvas(
                    xy,
                    scale=scale,
                    offset_xy=offset,
                ),
                color,
            )
    absent = audit.get("legally_absent_entity_ids", ())
    _header(
        output,
        (
            f"REFERENCE | frame={frame_index} | "
            f"expected={audit.get('expected_cardinality', 0)}"
        ),
        "legal absent=" + (",".join(map(str, absent)) or "none"),
    )
    return output


def _detections_by_frame(
    observation: OpenWorldObservation | None,
) -> dict[int, list[tuple[OpenWorldTrack, Any]]]:
    output: dict[int, list[tuple[OpenWorldTrack, Any]]] = {}
    if observation is None:
        return output
    for track in observation.tracks:
        for detection in track.detections:
            output.setdefault(detection.frame_index, []).append(
                (track, detection)
            )
    return output


def _prediction_panel(
    frame: np.ndarray,
    *,
    frame_index: int,
    audit: Mapping[str, Any],
    detections: Mapping[int, Sequence[tuple[OpenWorldTrack, Any]]],
    observation_available: bool,
    media_available: bool,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    source = _normalize_frame(frame)
    output, scale, offset, target_size = _letterbox(
        source,
        width=panel_width,
        height=panel_height,
    )
    matches = {
        str(row.get("prediction_track_id")): str(row.get("entity_id"))
        for row in audit.get("matches", ())
    }
    extra = {str(value) for value in audit.get("extra_track_ids", ())}
    ambiguous = {
        str(value)
        for value in audit.get("ambiguous_candidate_track_ids", ())
    }
    rejected = {
        str(row.get("prediction_track_id"))
        for row in audit.get("rejected_candidate_matches", ())
    }
    for track, detection in detections.get(frame_index, ()):
        track_id = track.track_id
        color = _identity_color(f"track:{track_id}")
        canvas_mask = _mask_to_canvas(
            detection.mask,
            source_shape=source.shape[:2],
            canvas_shape=output.shape[:2],
            offset_xy=offset,
            target_size=target_size,
        )
        output = _blend_mask(
            output,
            canvas_mask,
            color,
            alpha=0.20 if track_id in ambiguous else 0.32,
        )
        current_history = np.asarray(
            [
                item.xy
                for item in track.detections
                if item.frame_index <= frame_index
            ],
            dtype=np.float64,
        )
        if current_history.size:
            _draw_trajectory(
                output,
                _points_to_canvas(
                    current_history,
                    scale=scale,
                    offset_xy=offset,
                ),
                color,
            )
        status: list[str] = []
        if track_id in matches:
            status.append(f"->{matches[track_id]}")
        if track_id in extra:
            status.append("EXTRA")
        if track_id in ambiguous:
            status.append("CANDIDATE")
        if track_id in rejected:
            status.append("REJECTED")
        _label(
            output,
            f"P:{track_id} {' '.join(status)}".rstrip(),
            _point_to_canvas(
                detection.xy,
                scale=scale,
                offset_xy=offset,
            ),
            color,
        )
    state: list[str] = []
    if not observation_available:
        state.append("OBSERVER UNAVAILABLE")
    if not media_available:
        state.append("MEDIA UNAVAILABLE")
    if not state:
        state.append(
            "missing="
            f"{len(audit.get('missing_entity_ids', ()))} "
            "extra="
            f"{len(audit.get('extra_track_ids', ()))} "
            "overflow="
            f"{float(audit.get('overflow_count', 0.0)):g}"
        )
    _header(
        output,
        (
            f"PREDICTION | frame={frame_index} | "
            "observed="
            f"{float(audit.get('formal_prediction_cardinality', 0.0)):g}"
        ),
        " | ".join(state),
    )
    return output


def _format_ids(values: object, *, maximum: int = 88) -> str:
    if not isinstance(values, (list, tuple, set)):
        return "none"
    text = ", ".join(str(value) for value in values) or "none"
    return _ascii(text, maximum=maximum)


def _rejected_text(rows: object) -> str:
    if not isinstance(rows, (list, tuple)):
        return "none"
    values = []
    for row in rows[:4]:
        if not isinstance(row, Mapping):
            continue
        entity = row.get("entity_id", "?")
        track = row.get("prediction_track_id", "?")
        score = row.get("position_score")
        score_text = (
            "?"
            if not isinstance(score, (int, float))
            else f"{float(score):.2f}"
        )
        values.append(f"{entity}<-{track}@{score_text}")
    if len(rows) > 4:
        values.append(f"+{len(rows) - 4}")
    return ", ".join(values) or "none"


def _switch_text(rows: object) -> str:
    if not isinstance(rows, (list, tuple)):
        return "none"
    values = []
    for row in rows[:4]:
        if not isinstance(row, Mapping):
            continue
        values.append(
            f"{row.get('entity_id', '?')}:"
            f"{row.get('from_track_id', '?')}->"
            f"{row.get('to_track_id', '?')}"
        )
    if len(rows) > 4:
        values.append(f"+{len(rows) - 4}")
    return ", ".join(values) or "none"


def _put_line(
    frame: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    color: tuple[int, int, int] = (215, 215, 215),
    scale: float = 0.43,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        _ascii(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_cardinality_history(
    frame: np.ndarray,
    *,
    rows: Sequence[Mapping[str, Any]],
    through: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + width, y0 + height),
        (58, 58, 58),
        1,
    )
    expected = np.asarray(
        [float(row.get("expected_cardinality", 0.0)) for row in rows],
        dtype=np.float64,
    )
    prediction = np.asarray(
        [
            float(row.get("formal_prediction_cardinality", 0.0))
            for row in rows
        ],
        dtype=np.float64,
    )
    maximum = max(
        float(np.max(expected, initial=0.0)),
        float(np.max(prediction, initial=0.0)),
        1.0,
    )
    right = x0 + width
    bottom = y0 + height

    def points(values: np.ndarray) -> np.ndarray:
        indices = np.arange(through + 1, dtype=np.float64)
        denominator = max(len(rows) - 1, 1)
        x = x0 + indices / denominator * width
        y = bottom - values[: through + 1] / maximum * height
        return np.rint(np.column_stack((x, y))).astype(np.int32)

    expected_points = points(expected)
    prediction_points = points(prediction)
    if len(expected_points) >= 2:
        cv2.polylines(
            frame,
            [expected_points],
            False,
            (80, 230, 90),
            2,
            cv2.LINE_AA,
        )
        cv2.polylines(
            frame,
            [prediction_points],
            False,
            (40, 170, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(
            frame,
            tuple(expected_points[0]),
            3,
            (80, 230, 90),
            -1,
        )
        cv2.circle(
            frame,
            tuple(prediction_points[0]),
            3,
            (40, 170, 255),
            -1,
        )
    _put_line(
        frame,
        f"N max={maximum:g}",
        x=x0 + 5,
        y=y0 + 16,
        scale=0.38,
    )
    _put_line(
        frame,
        "GT",
        x=right - 82,
        y=y0 + 16,
        color=(80, 230, 90),
        scale=0.38,
    )
    _put_line(
        frame,
        "PRED",
        x=right - 48,
        y=y0 + 16,
        color=(40, 170, 255),
        scale=0.38,
    )


def _dashboard(
    *,
    width: int,
    height: int,
    frame_index: int,
    time_s: float,
    rows: Sequence[Mapping[str, Any]],
    full_subject_iou: float | None,
    scene_name: str,
) -> np.ndarray:
    output = np.full((height, width, 3), 22, dtype=np.uint8)
    row = rows[frame_index]
    iou_text = (
        "n/a" if full_subject_iou is None else f"{full_subject_iou:.3f}"
    )
    matched = len(row.get("matches", ()))
    lines = [
        f"OPEN-WORLD V2 AUDIT | {scene_name}",
        (
            f"t={time_s:.3f}s  expected="
            f"{row.get('expected_cardinality', 0)}  predicted="
            f"{float(row.get('formal_prediction_cardinality', 0.0)):g}  "
            f"matched={matched}  full-subject IoU={iou_text}"
        ),
        "missing: " + _format_ids(row.get("missing_entity_ids", ())),
        "extra: " + _format_ids(row.get("extra_track_ids", ())),
        "rejected: " + _rejected_text(
            row.get("rejected_candidate_matches", ())
        ),
        "ID switches: " + _switch_text(row.get("id_switches", ())),
        (
            "birth/death: "
            + _format_ids(row.get("birth_track_ids", ()), maximum=42)
            + " / "
            + _format_ids(row.get("death_track_ids", ()), maximum=42)
        ),
    ]
    graph_x = min(
        max(int(width * 0.59), 8),
        max(width - 88, 8),
    )
    line_width = max(graph_x - 30, 120)
    y = 27
    for index, line in enumerate(lines):
        _put_line(
            output,
            _ascii(line, maximum=max(line_width // 7, 24)),
            x=15,
            y=y,
            color=(
                (255, 255, 255)
                if index == 0
                else (215, 215, 215)
            ),
            scale=0.50 if index == 0 else 0.41,
            thickness=2 if index == 0 else 1,
        )
        y += 26
    graph_width = max(width - graph_x - 18, 80)
    graph_height = max(height - 38, 60)
    _draw_cardinality_history(
        output,
        rows=rows,
        through=frame_index,
        x0=graph_x,
        y0=20,
        width=graph_width,
        height=graph_height,
    )
    return output


def _normalize_union_overrides(
    values: Sequence[np.ndarray | None] | None,
    *,
    frame_shapes: Sequence[tuple[int, int]],
) -> tuple[list[np.ndarray | None], list[bool]] | None:
    if values is None:
        return None
    if len(values) != len(frame_shapes):
        raise ValueError("union masks must have one value per time sample")
    masks = [
        _normalize_mask(mask, shape=shape)
        for mask, shape in zip(values, frame_shapes)
    ]
    return masks, [mask is not None for mask in masks]


def _reference_unions(
    timelines: Sequence[ExpectedEntityTimeline],
    *,
    frame_shapes: Sequence[tuple[int, int]],
) -> tuple[list[np.ndarray | None], list[bool]]:
    masks: list[np.ndarray | None] = []
    complete: list[bool] = []
    for frame_index, shape in enumerate(frame_shapes):
        expected = [
            timeline
            for timeline in timelines
            if timeline.existence_supervised[frame_index]
            and timeline.expected_exists[frame_index]
        ]
        union = np.zeros(shape, dtype=np.uint8)
        frame_complete = True
        for timeline in expected:
            mask = _normalize_mask(
                timeline.reference_masks[frame_index],
                shape=shape,
            )
            if mask is None:
                frame_complete = False
                continue
            union = np.maximum(union, mask)
        masks.append(union)
        complete.append(frame_complete)
    return masks, complete


def _prediction_unions(
    observation: OpenWorldObservation | None,
    *,
    frame_shapes: Sequence[tuple[int, int]],
    media_available: Sequence[bool],
) -> tuple[list[np.ndarray | None], list[bool]]:
    if observation is None:
        return [None] * len(frame_shapes), [False] * len(frame_shapes)
    detections = _detections_by_frame(observation)
    masks: list[np.ndarray | None] = []
    complete: list[bool] = []
    for frame_index, shape in enumerate(frame_shapes):
        union = np.zeros(shape, dtype=np.uint8)
        frame_complete = bool(media_available[frame_index])
        for track, detection in detections.get(frame_index, ()):
            if track.formal_exposure_weight <= 0.0:
                continue
            mask = _normalize_mask(detection.mask, shape=shape)
            if mask is None:
                frame_complete = False
                continue
            union = np.maximum(union, mask)
        if observation.overflow_counts[frame_index] > 0.0:
            frame_complete = False
        masks.append(union)
        complete.append(frame_complete)
    return masks, complete


def _mask_iou(
    reference: np.ndarray | None,
    prediction: np.ndarray | None,
    *,
    complete: bool,
) -> float | None:
    if reference is None or prediction is None or not complete:
        return None
    first = np.asarray(reference) > 0
    second = np.asarray(prediction) > 0
    if first.shape != second.shape:
        return None
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _validated_ious(
    values: Sequence[float | None] | None,
    *,
    frame_count: int,
) -> list[float | None] | None:
    if values is None:
        return None
    if len(values) != frame_count:
        raise ValueError(
            "full_subject_ious must have one value per time sample"
        )
    output: list[float | None] = []
    for value in values:
        if value is None:
            output.append(None)
            continue
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("full-subject IoU values must be in [0, 1]")
        output.append(score)
    return output


def _observation_payload(
    observation: OpenWorldObservation | None,
) -> dict[str, Any]:
    if observation is None:
        return {
            "available": False,
            "diagnostics": {},
            "overflow_counts": [],
            "tracks": [],
        }
    tracks = []
    for track in observation.tracks:
        tracks.append(
            {
                "track_id": track.track_id,
                "entity_class": track.entity_class,
                "confirmed": track.confirmed,
                "evidence_tier": track.evidence_tier.value,
                "formal_exposure_weight": (
                    track.formal_exposure_weight
                ),
                "detections": [
                    {
                        "frame": detection.frame_index,
                        "detection_id": detection.detection_id,
                        "xy": detection.xy.tolist(),
                        "area_px2": detection.area_px2,
                        "confidence": detection.confidence,
                        "evidence_tier": detection.evidence_tier.value,
                        "sources": list(detection.sources),
                        "mask_area_px2": (
                            None
                            if detection.mask is None
                            else int(np.count_nonzero(detection.mask))
                        ),
                        "metadata": _json_safe(detection.metadata),
                    }
                    for detection in track.detections
                ],
            }
        )
    return {
        "available": True,
        "diagnostics": _json_safe(observation.diagnostics),
        "overflow_counts": observation.overflow_counts.tolist(),
        "tracks": tracks,
    }


def _timeline_payload(
    timelines: Sequence[ExpectedEntityTimeline],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for timeline in timelines:
        payload = timeline.to_dict()
        payload["trajectory"] = [
            {
                "frame": index,
                "xy": [
                    (
                        float(value)
                        if math.isfinite(float(value))
                        else None
                    )
                    for value in timeline.reference_xy[index, :2]
                ],
                "mask_area_px2": (
                    None
                    if timeline.reference_masks[index] is None
                    else int(
                        np.count_nonzero(
                            timeline.reference_masks[index]
                        )
                    )
                ),
            }
            for index in range(timeline.frame_count)
        ]
        output.append(_json_safe(payload))
    return output


def _audit_payload(
    request: Any,
    *,
    scene_name: str,
    times_s: Sequence[float],
    timelines: Sequence[ExpectedEntityTimeline],
    observation: OpenWorldObservation | None,
    comparison: ObjectCentricComparisonV2,
    reference_unions: Sequence[np.ndarray | None],
    prediction_unions: Sequence[np.ndarray | None],
    reference_complete: Sequence[bool],
    prediction_complete: Sequence[bool],
    full_subject_ious: Sequence[float | None],
    prediction_available: Sequence[bool],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame_index, source in enumerate(comparison.per_frame):
        reference_mask = reference_unions[frame_index]
        prediction_mask = prediction_unions[frame_index]
        rows.append(
            {
                **_json_safe(source),
                "reference_full_subject_mask_area_px2": (
                    None
                    if reference_mask is None
                    else int(np.count_nonzero(reference_mask))
                ),
                "prediction_full_subject_mask_area_px2": (
                    None
                    if prediction_mask is None
                    else int(np.count_nonzero(prediction_mask))
                ),
                "reference_full_subject_mask_complete": bool(
                    reference_complete[frame_index]
                ),
                "prediction_full_subject_mask_complete": bool(
                    prediction_complete[frame_index]
                ),
                "full_subject_mask_iou": (
                    full_subject_ious[frame_index]
                ),
                "prediction_media_available": bool(
                    prediction_available[frame_index]
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "artifact_protocol": {
            "id": ARTIFACT_PROTOCOL_ID,
            "version": ARTIFACT_PROTOCOL_VERSION,
        },
        "case_id": request.case.get("case_id"),
        "job_id": request.job.get("job_id"),
        "scene_id": request.case.get("scene_id"),
        "scene_name": scene_name,
        "times_s": [float(value) for value in times_s],
        "expected_entities": _timeline_payload(timelines),
        "prediction_observation": _observation_payload(observation),
        "comparison": comparison.to_dict(),
        "per_frame": rows,
    }


def _validate_inputs(
    *,
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    timelines: Sequence[ExpectedEntityTimeline],
    observation: OpenWorldObservation | None,
    comparison: ObjectCentricComparisonV2,
    prediction_available: Sequence[bool] | None,
) -> list[bool]:
    frame_count = len(times_s)
    if frame_count < 1:
        raise ValueError("artifact rendering requires at least one frame")
    times = np.asarray(times_s, dtype=np.float64)
    if (
        times.shape != (frame_count,)
        or not np.isfinite(times).all()
        or np.any(times < 0.0)
        or (frame_count > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError(
            "times_s must be finite, non-negative, and strictly increasing"
        )
    if len(reference_frames) != frame_count:
        raise ValueError(
            "reference_frames must have one value per time sample"
        )
    if len(prediction_frames) != frame_count:
        raise ValueError(
            "prediction_frames must have one value per time sample"
        )
    if len(comparison.per_frame) != frame_count:
        raise ValueError(
            "comparison must have one audit row per time sample"
        )
    if any(timeline.frame_count != frame_count for timeline in timelines):
        raise ValueError(
            "expected timelines must match the artifact time grid"
        )
    if (
        observation is not None
        and observation.overflow_counts.shape != (frame_count,)
    ):
        raise ValueError(
            "observation overflow counts must match the artifact time grid"
        )
    if prediction_available is None:
        return [True] * frame_count
    if len(prediction_available) != frame_count:
        raise ValueError(
            "prediction_available must have one value per time sample"
        )
    return [bool(value) for value in prediction_available]


def _derived_fps(times_s: Sequence[float]) -> float:
    if len(times_s) < 2:
        return 8.0
    step = float(np.median(np.diff(np.asarray(times_s, dtype=np.float64))))
    if not math.isfinite(step) or step <= 0.0:
        return 8.0
    return float(np.clip(1.0 / step, 1.0, 60.0))


def render_open_world_v2_overlay(
    path: Path,
    *,
    scene_name: str,
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    expected_timelines: Sequence[ExpectedEntityTimeline],
    prediction_observation: OpenWorldObservation | None,
    comparison: ObjectCentricComparisonV2,
    full_subject_ious: Sequence[float | None],
    prediction_available: Sequence[bool],
    config: Mapping[str, Any],
) -> None:
    """Render the side-by-side overlay.

    This low-level function intentionally raises on invalid inputs or codec
    failures.  ``write_open_world_v2_artifacts`` is the score-safe boundary.
    """

    panel_width = int(config.get("panel_width", 640))
    panel_height = int(config.get("panel_height", 360))
    footer_height = int(config.get("footer_height", 205))
    if min(panel_width, panel_height, footer_height) < 64:
        raise ValueError("visualization dimensions must each be >= 64")
    configured_fps = config.get("fps")
    fps = (
        _derived_fps(times_s)
        if configured_fps is None
        else float(configured_fps)
    )
    if not math.isfinite(fps) or fps <= 0.0 or fps > 120.0:
        raise ValueError("visualization fps must be in (0, 120]")
    codec = str(config.get("codec", "mp4v"))
    if len(codec) != 4 or not codec.isascii():
        raise ValueError("visualization codec must contain four ASCII bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (2 * panel_width, panel_height + footer_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create open-world overlay: {path}")
    rows = [dict(row) for row in comparison.per_frame]
    detections = _detections_by_frame(prediction_observation)
    try:
        for frame_index, time_s in enumerate(times_s):
            audit = rows[frame_index]
            reference = _reference_panel(
                reference_frames[frame_index],
                frame_index=frame_index,
                timelines=expected_timelines,
                audit=audit,
                panel_width=panel_width,
                panel_height=panel_height,
            )
            prediction = _prediction_panel(
                prediction_frames[frame_index],
                frame_index=frame_index,
                audit=audit,
                detections=detections,
                observation_available=prediction_observation is not None,
                media_available=bool(
                    prediction_available[frame_index]
                ),
                panel_width=panel_width,
                panel_height=panel_height,
            )
            dashboard = _dashboard(
                width=2 * panel_width,
                height=footer_height,
                frame_index=frame_index,
                time_s=float(time_s),
                rows=rows,
                full_subject_iou=full_subject_ious[frame_index],
                scene_name=scene_name,
            )
            writer.write(
                np.vstack(
                    (
                        np.hstack(
                            (
                                reference,
                                prediction,
                            )
                        ),
                        dashboard,
                    )
                )
            )
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(
            f"open-world overlay writer produced no output: {path}"
        )


def write_open_world_v2_artifacts(
    request: Any,
    *,
    scene_name: str,
    times_s: Sequence[float],
    reference_frames: Sequence[np.ndarray],
    prediction_frames: Sequence[np.ndarray],
    expected_timelines: Sequence[ExpectedEntityTimeline],
    prediction_observation: OpenWorldObservation | None,
    comparison: ObjectCentricComparisonV2,
    config: Mapping[str, Any] | None = None,
    reference_union_masks: Sequence[np.ndarray | None] | None = None,
    prediction_union_masks: Sequence[np.ndarray | None] | None = None,
    full_subject_ious: Sequence[float | None] | None = None,
    prediction_available: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Write one external overlay/audit pair and a local hashed manifest.

    The returned mapping can be merged directly into
    ``CaseEvaluationResult.artifacts``.  All external failures become a
    ``status=failed`` manifest.  The function only raises if even the local
    artifact directory/manifest cannot be written.  Union-mask overrides must
    contain the complete physical-subject set, including unmatched formal
    prediction tracks; leaving them unset derives that anti-hack union from
    the open-world observation.
    """

    artifact_config = dict(config or {})
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = request.artifact_dir / LOCAL_MANIFEST_NAME
    if not bool(artifact_config.get("enabled", True)):
        manifest = {
            "schema_version": "1.0",
            "artifact_protocol": {
                "id": ARTIFACT_PROTOCOL_ID,
                "version": ARTIFACT_PROTOCOL_VERSION,
            },
            "status": "disabled",
            "storage_policy": "external_unsealed_diagnostic",
        }
        write_json(local_manifest, manifest)
        return {
            "open_world_v2_artifact_manifest": str(local_manifest),
            "open_world_v2_artifacts": manifest,
        }

    try:
        available = _validate_inputs(
            times_s=times_s,
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            timelines=expected_timelines,
            observation=prediction_observation,
            comparison=comparison,
            prediction_available=prediction_available,
        )
        reference_shapes = [
            _normalize_frame(frame).shape[:2]
            for frame in reference_frames
        ]
        prediction_shapes = [
            _normalize_frame(frame).shape[:2]
            for frame in prediction_frames
        ]
        reference_override = _normalize_union_overrides(
            reference_union_masks,
            frame_shapes=reference_shapes,
        )
        prediction_override = _normalize_union_overrides(
            prediction_union_masks,
            frame_shapes=prediction_shapes,
        )
        if reference_override is None:
            reference_unions, reference_complete = _reference_unions(
                expected_timelines,
                frame_shapes=reference_shapes,
            )
        else:
            reference_unions, reference_complete = reference_override
        if prediction_override is None:
            prediction_unions, prediction_complete = _prediction_unions(
                prediction_observation,
                frame_shapes=prediction_shapes,
                media_available=available,
            )
        else:
            prediction_unions, prediction_complete = prediction_override
            prediction_complete = [
                complete and media
                for complete, media in zip(
                    prediction_complete,
                    available,
                )
            ]
        supplied_ious = _validated_ious(
            full_subject_ious,
            frame_count=len(times_s),
        )
        if supplied_ious is None:
            ious = [
                _mask_iou(
                    reference_unions[index],
                    prediction_unions[index],
                    complete=(
                        reference_complete[index]
                        and prediction_complete[index]
                    ),
                )
                for index in range(len(times_s))
            ]
        else:
            ious = supplied_ious

        directory, relative, external_root = _artifact_directory(
            request,
            config=artifact_config,
        )
        directory.mkdir(parents=True, exist_ok=True)
        video_path = directory / "open_world_v2_overlay.mp4"
        audit_path = directory / "open_world_v2_audit.json"
        render_open_world_v2_overlay(
            video_path,
            scene_name=scene_name,
            times_s=times_s,
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            expected_timelines=expected_timelines,
            prediction_observation=prediction_observation,
            comparison=comparison,
            full_subject_ious=ious,
            prediction_available=available,
            config=artifact_config,
        )
        audit = _audit_payload(
            request,
            scene_name=scene_name,
            times_s=times_s,
            timelines=expected_timelines,
            observation=prediction_observation,
            comparison=comparison,
            reference_unions=reference_unions,
            prediction_unions=prediction_unions,
            reference_complete=reference_complete,
            prediction_complete=prediction_complete,
            full_subject_ious=ious,
            prediction_available=available,
        )
        write_json(audit_path, _json_safe(audit))

        repository_link = str(
            artifact_config.get(
                "repository_link",
                DEFAULT_REPOSITORY_LINK,
            )
        ).strip() or DEFAULT_REPOSITORY_LINK
        paths = {
            "overlay_video": video_path,
            "audit_json": audit_path,
        }
        files = {
            name: {
                "path": str(path.resolve()),
                "repository_path": (
                    Path(repository_link) / relative / path.name
                ).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        }
        manifest = {
            "schema_version": "1.0",
            "artifact_protocol": {
                "id": ARTIFACT_PROTOCOL_ID,
                "version": ARTIFACT_PROTOCOL_VERSION,
            },
            "status": "complete",
            "evaluator_config_sha256": canonical_sha256(
                _json_safe(request.evaluator_config)
            ),
            "storage_policy": (
                "external_unsealed_diagnostic_with_local_hashed_manifest"
            ),
            "external_root": external_root,
            "external_directory": str(directory.resolve()),
            "repository_directory": (
                Path(repository_link) / relative
            ).as_posix(),
            "files": files,
        }
    except Exception as exc:
        manifest = {
            "schema_version": "1.0",
            "artifact_protocol": {
                "id": ARTIFACT_PROTOCOL_ID,
                "version": ARTIFACT_PROTOCOL_VERSION,
            },
            "status": "failed",
            "storage_policy": "best_effort_never_changes_case_score",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    write_json(local_manifest, manifest)
    return {
        "open_world_v2_artifact_manifest": str(local_manifest),
        "open_world_v2_artifacts": manifest,
    }


__all__ = [
    "ARTIFACT_PROTOCOL_ID",
    "ARTIFACT_PROTOCOL_VERSION",
    "DEFAULT_EXTERNAL_ROOT",
    "DEFAULT_EXTERNAL_ROOT_ENV",
    "DEFAULT_REPOSITORY_LINK",
    "LOCAL_MANIFEST_NAME",
    "render_open_world_v2_overlay",
    "write_open_world_v2_artifacts",
]
