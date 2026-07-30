from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoProtocolError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int
    last_frame_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SampledVideo:
    frames: list[np.ndarray]
    info: VideoInfo
    sample_times_s: list[float]
    source_indices: list[int]
    spatial_transform: dict
    available: list[bool] | None = None

    def __post_init__(self) -> None:
        if self.available is None:
            return
        if len(self.available) != len(self.frames):
            raise ValueError(
                "available must have one value per sampled frame"
            )
        if len(self.available) != len(self.sample_times_s):
            raise ValueError(
                "available must have one value per sample timestamp"
            )
        if any(
            not isinstance(value, (bool, np.bool_))
            for value in self.available
        ):
            raise ValueError("available must contain boolean values")
        object.__setattr__(
            self,
            "available",
            [bool(value) for value in self.available],
        )


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoProtocolError("video_open_failed", f"cannot open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise VideoProtocolError(
            "invalid_video_metadata",
            f"invalid video metadata for {path}: frames={count}, fps={fps}, "
            f"size={width}x{height}",
        )
    return VideoInfo(
        frame_count=count,
        fps=fps,
        width=width,
        height=height,
        last_frame_time_s=(count - 1) / fps,
    )


def _letterbox(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    pad_value: int,
) -> tuple[np.ndarray, dict]:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=interpolation
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas, {
        "policy": "preserve_aspect_ratio_letterbox",
        "scale": scale,
        "offset_xy": [x, y],
        "source_size": [source_width, source_height],
        "target_size": [width, height],
    }


def sample_video(
    path: Path,
    *,
    sample_times_s: list[float],
    width: int,
    height: int,
    pad_value: int = 0,
    min_source_fps: float = 1.0,
    duration_tolerance_s: float = 0.02,
    decode_policy: str = "legacy_random_seek",
    allow_partial: bool = False,
) -> SampledVideo:
    if not sample_times_s:
        raise VideoProtocolError("empty_timeline", "sample timeline is empty")
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be a boolean")
    info = probe_video(path)
    if info.fps < min_source_fps:
        raise VideoProtocolError(
            "source_fps_too_low",
            f"{path} has {info.fps:g} FPS; minimum is {min_source_fps:g}",
        )
    required_end = max(sample_times_s)
    if (
        not allow_partial
        and info.last_frame_time_s + duration_tolerance_s < required_end
    ):
        raise VideoProtocolError(
            "insufficient_duration",
            f"{path} ends at frame time {info.last_frame_time_s:.6f}s, "
            f"but protocol requires {required_end:.6f}s",
        )
    raw_indices = np.rint(np.asarray(sample_times_s) * info.fps).astype(int)
    available_by_metadata = (
        (raw_indices >= 0)
        & (raw_indices < info.frame_count)
        & (
            np.asarray(sample_times_s, dtype=np.float64)
            <= info.last_frame_time_s + duration_tolerance_s
        )
    )
    if decode_policy not in {"legacy_random_seek", "sequential_forward"}:
        raise VideoProtocolError(
            "invalid_decode_policy",
            f"unsupported video decode policy: {decode_policy!r}",
        )
    if (
        (decode_policy == "sequential_forward" or allow_partial)
        and raw_indices.min(initial=0) < 0
    ):
        raise VideoProtocolError(
            "invalid_timeline",
            "sample timeline contains a negative source-frame index",
        )
    if (
        not allow_partial
        and raw_indices.max(initial=0) >= info.frame_count
    ):
        raise VideoProtocolError(
            "insufficient_duration",
            f"{path} has no source frame for t={required_end:.6f}s",
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoProtocolError("video_open_failed", f"cannot open video: {path}")
    if decode_policy == "legacy_random_seek":
        frames: list[np.ndarray] = []
        available: list[bool] = []
        transform: dict | None = None
        for index, metadata_available in zip(
            raw_indices,
            available_by_metadata,
        ):
            if allow_partial and not bool(metadata_available):
                frames.append(
                    _unavailable_frame(
                        width=width,
                        height=height,
                        pad_value=pad_value,
                    )
                )
                available.append(False)
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                if allow_partial:
                    frames.append(
                        _unavailable_frame(
                            width=width,
                            height=height,
                            pad_value=pad_value,
                        )
                    )
                    available.append(False)
                    continue
                capture.release()
                raise VideoProtocolError(
                    "video_decode_failed",
                    f"failed to decode frame {index} from {path}",
                )
            normalized, current_transform = _letterbox(
                frame, width=width, height=height, pad_value=pad_value
            )
            transform = transform or current_transform
            frames.append(normalized)
            available.append(True)
        capture.release()
        return SampledVideo(
            frames=frames,
            info=info,
            sample_times_s=list(sample_times_s),
            source_indices=raw_indices.astype(int).tolist(),
            spatial_transform=transform
            or _spatial_transform_from_info(
                info,
                width=width,
                height=height,
            ),
            available=available,
        )

    requested = {
        int(index)
        for index, metadata_available in zip(
            raw_indices,
            available_by_metadata,
        )
        if bool(metadata_available)
    }
    normalized_by_index: dict[int, np.ndarray] = {}
    transform_by_index: dict[int, dict] = {}
    try:
        decode_end = max(requested, default=-1)
        for decoded_index in range(decode_end + 1):
            ok, frame = capture.read()
            if not ok:
                if allow_partial:
                    break
                failed_index = next(
                    int(index)
                    for index in raw_indices
                    if int(index) not in normalized_by_index
                )
                raise VideoProtocolError(
                    "video_decode_failed",
                    f"failed to decode frame {failed_index} from {path}",
                )
            if decoded_index not in requested:
                continue
            normalized, current_transform = _letterbox(
                frame, width=width, height=height, pad_value=pad_value
            )
            normalized_by_index[decoded_index] = normalized
            transform_by_index[decoded_index] = current_transform
    finally:
        capture.release()

    frames: list[np.ndarray] = []
    available = []
    emitted: set[int] = set()
    for index, metadata_available in zip(
        raw_indices,
        available_by_metadata,
    ):
        source_index = int(index)
        if (
            not bool(metadata_available)
            or source_index not in normalized_by_index
        ):
            if not allow_partial:
                raise VideoProtocolError(
                    "video_decode_failed",
                    f"failed to decode frame {source_index} from {path}",
                )
            frames.append(
                _unavailable_frame(
                    width=width,
                    height=height,
                    pad_value=pad_value,
                )
            )
            available.append(False)
            continue
        normalized = normalized_by_index[source_index]
        frames.append(
            normalized.copy() if source_index in emitted else normalized
        )
        available.append(True)
        emitted.add(source_index)
    first_available_index = next(
        (
            int(index)
            for index in raw_indices
            if int(index) in transform_by_index
        ),
        None,
    )
    return SampledVideo(
        frames=frames,
        info=info,
        sample_times_s=list(sample_times_s),
        source_indices=raw_indices.astype(int).tolist(),
        spatial_transform=(
            transform_by_index[first_available_index]
            if first_available_index is not None
            else _spatial_transform_from_info(
                info,
                width=width,
                height=height,
            )
        ),
        available=available,
    )


def _unavailable_frame(
    *,
    width: int,
    height: int,
    pad_value: int,
) -> np.ndarray:
    """Return neutral pixels for an unavailable time cell, never an extrapolation."""
    return np.full((height, width, 3), pad_value, dtype=np.uint8)


def _spatial_transform_from_info(
    info: VideoInfo,
    *,
    width: int,
    height: int,
) -> dict:
    scale = min(width / info.width, height / info.height)
    resized_width = max(1, int(round(info.width * scale)))
    resized_height = max(1, int(round(info.height * scale)))
    return {
        "policy": "preserve_aspect_ratio_letterbox",
        "scale": scale,
        "offset_xy": [
            (width - resized_width) // 2,
            (height - resized_height) // 2,
        ],
        "source_size": [info.width, info.height],
        "target_size": [width, height],
    }


def reference_timeline(
    path: Path,
    *,
    fps: float,
    max_duration_s: float,
    minimum_duration_s: float,
) -> list[float]:
    """Build an inclusive, case-specific timeline bounded by the reference."""
    info = probe_video(path)
    duration = min(float(max_duration_s), info.last_frame_time_s)
    if duration < float(minimum_duration_s):
        raise VideoProtocolError(
            "reference_too_short",
            f"{path} covers {duration:.6f}s; minimum is {minimum_duration_s:g}s",
        )
    count = int(np.floor(duration * float(fps) + 1e-9)) + 1
    if count < 2:
        raise VideoProtocolError(
            "reference_too_short", f"{path} yields fewer than two samples"
        )
    return (np.arange(count, dtype=np.float64) / float(fps)).tolist()
