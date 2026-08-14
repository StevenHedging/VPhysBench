"""Canonical physical-time sampling for frozen reference observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import ReferenceTimeline, TimelineSample


def _positive_finite(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def build_timeline(
    *,
    frame_count: int,
    source_fps: float,
    encoded_to_physical_speed: float,
    sampling_rate_hz: float = 24.0,
    source_width: int | None = None,
    source_height: int | None = None,
) -> ReferenceTimeline:
    """Map a physical-time grid to exact encoded source-frame indices."""

    if isinstance(frame_count, bool) or int(frame_count) != frame_count:
        raise ValueError("frame_count must be a positive integer")
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    fps = _positive_finite(source_fps, label="source_fps")
    speed = _positive_finite(
        encoded_to_physical_speed,
        label="encoded_to_physical_speed",
    )
    rate = _positive_finite(sampling_rate_hz, label="sampling_rate_hz")
    for value, label in (
        (source_width, "source_width"),
        (source_height, "source_height"),
    ):
        if value is not None and (
            isinstance(value, bool) or int(value) != value or int(value) <= 0
        ):
            raise ValueError(f"{label} must be a positive integer")

    encoded_duration = (frame_count - 1) / fps
    physical_duration = encoded_duration / speed
    regular_count = int(math.floor(physical_duration * rate + 1e-9)) + 1
    physical_times = [index / rate for index in range(regular_count)]
    tolerance = max(1e-12, 1e-9 * max(1.0, physical_duration))
    if physical_duration - physical_times[-1] > tolerance:
        physical_times.append(physical_duration)
    else:
        physical_times[-1] = physical_duration

    samples: list[TimelineSample] = []
    for index, physical_time in enumerate(physical_times):
        source_time = min(encoded_duration, physical_time * speed)
        source_frame = int(np.rint(source_time * fps))
        source_frame = min(frame_count - 1, max(0, source_frame))
        samples.append(
            TimelineSample(
                observation_index=index,
                encoded_time_seconds=float(source_time),
                physical_time_seconds=float(physical_time),
                source_frame_index=source_frame,
                source_time_seconds=float(source_time),
            )
        )
    return ReferenceTimeline(
        sampling_rate_hz=rate,
        source_frame_count=frame_count,
        source_fps=fps,
        encoded_to_physical_speed=speed,
        samples=tuple(samples),
        source_width=None if source_width is None else int(source_width),
        source_height=None if source_height is None else int(source_height),
    )


def timeline_to_dict(case_id: str, timeline: ReferenceTimeline) -> dict[str, Any]:
    if timeline.source_width is None or timeline.source_height is None:
        raise ValueError("timeline source width and height are required for storage")
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "sampling_rate_hz": timeline.sampling_rate_hz,
        "source_video": {
            "frame_count": timeline.source_frame_count,
            "width": timeline.source_width,
            "height": timeline.source_height,
            "fps": timeline.source_fps,
            "duration_seconds": timeline.source_duration_seconds,
        },
        "temporal": {
            "encoded_to_physical_speed": timeline.encoded_to_physical_speed,
        },
        "samples": [
            {
                "observation_index": sample.observation_index,
                "encoded_time_seconds": sample.encoded_time_seconds,
                "physical_time_seconds": sample.physical_time_seconds,
                "source_frame_index": sample.source_frame_index,
                "source_time_seconds": sample.source_time_seconds,
            }
            for sample in timeline.samples
        ],
    }


def timeline_from_dict(value: dict[str, Any]) -> ReferenceTimeline:
    required = {
        "schema_version",
        "case_id",
        "sampling_rate_hz",
        "source_video",
        "temporal",
        "samples",
    }
    if set(value) != required or value.get("schema_version") != "1.0":
        raise ValueError("reference observation timeline has invalid fields")
    source = value["source_video"]
    temporal = value["temporal"]
    if not isinstance(source, dict) or set(source) != {
        "frame_count",
        "width",
        "height",
        "fps",
        "duration_seconds",
    }:
        raise ValueError("reference observation source_video is invalid")
    if not isinstance(temporal, dict) or set(temporal) != {
        "encoded_to_physical_speed"
    }:
        raise ValueError("reference observation temporal mapping is invalid")
    timeline = build_timeline(
        frame_count=source["frame_count"],
        source_fps=source["fps"],
        encoded_to_physical_speed=temporal["encoded_to_physical_speed"],
        sampling_rate_hz=value["sampling_rate_hz"],
        source_width=source["width"],
        source_height=source["height"],
    )
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("reference observation timeline samples must be non-empty")
    samples: list[TimelineSample] = []
    previous_physical = -math.inf
    previous_source = -1
    expected_fields = {
        "observation_index",
        "encoded_time_seconds",
        "physical_time_seconds",
        "source_frame_index",
        "source_time_seconds",
    }
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError(f"reference observation timeline sample {index} is invalid")
        if raw["observation_index"] != index:
            raise ValueError("reference observation indices must be contiguous")
        physical = float(raw["physical_time_seconds"])
        source_index = int(raw["source_frame_index"])
        numbers = (
            physical,
            float(raw["encoded_time_seconds"]),
            float(raw["source_time_seconds"]),
        )
        if any(not math.isfinite(number) or number < 0.0 for number in numbers):
            raise ValueError("reference observation timeline times must be finite")
        if physical <= previous_physical and index > 0:
            raise ValueError("reference observation physical times must increase")
        if source_index < previous_source or not 0 <= source_index < timeline.source_frame_count:
            raise ValueError("reference observation source indices are invalid")
        samples.append(
            TimelineSample(
                observation_index=index,
                encoded_time_seconds=numbers[1],
                physical_time_seconds=physical,
                source_frame_index=source_index,
                source_time_seconds=numbers[2],
            )
        )
        previous_physical = physical
        previous_source = source_index
    return ReferenceTimeline(
        sampling_rate_hz=timeline.sampling_rate_hz,
        source_frame_count=timeline.source_frame_count,
        source_fps=timeline.source_fps,
        encoded_to_physical_speed=timeline.encoded_to_physical_speed,
        samples=tuple(samples),
        source_width=timeline.source_width,
        source_height=timeline.source_height,
    )


__all__ = ["build_timeline", "timeline_from_dict", "timeline_to_dict"]
