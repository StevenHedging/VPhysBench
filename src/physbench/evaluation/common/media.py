from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...baseline_runtime.media_contract import (
    MediaContractError,
    centered_contain_rect,
    validate_media_contract,
)

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


@dataclass(frozen=True)
class SharedSpatialPlan:
    width: int
    height: int
    reference_crop_xywh: tuple[int, int, int, int]
    prediction_crop_xywh: tuple[int, int, int, int]
    provenance: dict[str, Any]


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


def probe_image_size(path: Path) -> tuple[int, int]:
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim < 2:
        raise VideoProtocolError(
            "conditioning_image_unreadable",
            f"cannot read conditioning image: {path}",
        )
    height, width = source.shape[:2]
    if width <= 0 or height <= 0:
        raise VideoProtocolError(
            "conditioning_image_invalid",
            f"conditioning image has invalid size: {path}",
        )
    return int(width), int(height)


def _largest_exact_aspect_size(
    source_width: int,
    source_height: int,
    *,
    maximum_width: int,
    maximum_height: int,
) -> tuple[int, int]:
    divisor = gcd(source_width, source_height)
    unit_width = source_width // divisor
    unit_height = source_height // divisor
    multiplier = min(
        maximum_width // unit_width,
        maximum_height // unit_height,
    )
    if multiplier <= 0:
        raise VideoProtocolError(
            "spatial_target_too_small",
            "protocol bounds cannot represent the reference aspect ratio "
            f"without distortion: reference={source_width}x{source_height}, "
            f"bounds={maximum_width}x{maximum_height}",
        )
    return unit_width * multiplier, unit_height * multiplier


def resolve_shared_spatial_plan(
    *,
    reference_info: VideoInfo,
    prediction_info: VideoInfo,
    maximum_width: int,
    maximum_height: int,
    media_contract: dict[str, Any] | None,
    expected_conditioning_asset: str | None,
) -> SharedSpatialPlan:
    """Resolve one no-padding coordinate system for reference and prediction.

    An I2V prediction may use a model canvas whose unused margin came from an
    aspect-preserving contain transform.  Only that declared margin is removed;
    physical reference content is never cropped.  With no contract, the two
    videos must already have exactly the same aspect ratio.
    """
    target_width, target_height = _largest_exact_aspect_size(
        reference_info.width,
        reference_info.height,
        maximum_width=maximum_width,
        maximum_height=maximum_height,
    )
    reference_crop = (
        0,
        0,
        reference_info.width,
        reference_info.height,
    )
    if media_contract is None:
        if (
            reference_info.width * prediction_info.height
            != reference_info.height * prediction_info.width
        ):
            raise VideoProtocolError(
                "prediction_media_contract_missing",
                "reference and prediction aspect ratios differ and the "
                "prediction has no sealed I2V media contract",
            )
        prediction_crop = (
            0,
            0,
            prediction_info.width,
            prediction_info.height,
        )
        mode = "already_exact_full_frame"
        contract_value = None
    else:
        try:
            contract = validate_media_contract(media_contract)
        except MediaContractError as exc:
            raise VideoProtocolError(exc.code, str(exc)) from exc
        if (
            expected_conditioning_asset is None
            or contract["conditioning"]["asset"]
            != expected_conditioning_asset
        ):
            raise VideoProtocolError(
                "prediction_conditioning_asset_mismatch",
                "prediction spatial contract is not bound to this Case's "
                "first-frame asset",
            )
        canvas = contract["output"]["canvas"]
        canvas_width = canvas["width"]
        canvas_height = canvas["height"]
        if (
            canvas_width != prediction_info.width
            or canvas_height != prediction_info.height
        ):
            raise VideoProtocolError(
                "prediction_canvas_mismatch",
                "prediction video dimensions differ from its sealed "
                f"canvas: video={prediction_info.width}x{prediction_info.height}, "
                f"contract={canvas_width}x{canvas_height}",
            )
        try:
            prediction_crop = centered_contain_rect(
                reference_info.width,
                reference_info.height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
        except MediaContractError as exc:
            raise VideoProtocolError(exc.code, str(exc)) from exc
        mode = "sealed_i2v_conditioning_content"
        contract_value = contract
    return SharedSpatialPlan(
        width=target_width,
        height=target_height,
        reference_crop_xywh=reference_crop,
        prediction_crop_xywh=prediction_crop,
        provenance={
            "policy": "shared_reference_content_no_pad_v1",
            "mode": mode,
            "reference_source_size": [
                reference_info.width,
                reference_info.height,
            ],
            "prediction_source_size": [
                prediction_info.width,
                prediction_info.height,
            ],
            "reference_crop_xywh": list(reference_crop),
            "prediction_crop_xywh": list(prediction_crop),
            "target_size": [target_width, target_height],
            "padding_used_for_evaluation": False,
            "aspect_ratio_distortion": False,
            "physical_reference_content_cropped": False,
            "media_contract": contract_value,
        },
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


def _crop_resize_no_pad(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    crop_xywh: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict]:
    """Crop a declared physical view and resize it without padding/distortion."""
    source_height, source_width = frame.shape[:2]
    x, y, crop_width, crop_height = crop_xywh
    if (
        x < 0
        or y < 0
        or crop_width <= 0
        or crop_height <= 0
        or x + crop_width > source_width
        or y + crop_height > source_height
    ):
        raise VideoProtocolError(
            "invalid_spatial_crop",
            "spatial crop lies outside decoded frame: "
            f"crop={crop_xywh}, frame={source_width}x{source_height}",
        )
    if crop_width * height != crop_height * width:
        raise VideoProtocolError(
            "spatial_aspect_ratio_mismatch",
            "crop and target must have exactly the same aspect ratio: "
            f"crop={crop_width}x{crop_height}, target={width}x{height}",
        )
    cropped = frame[y : y + crop_height, x : x + crop_width]
    scale = width / crop_width
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        cropped,
        (width, height),
        interpolation=interpolation,
    )
    return resized, {
        "policy": "reference_content_crop_resize_no_pad",
        "crop_xywh": [x, y, crop_width, crop_height],
        "scale": scale,
        "source_size": [source_width, source_height],
        "target_size": [width, height],
        "padding": None,
    }


def _normalize_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    pad_value: int,
    spatial_policy: str,
    crop_xywh: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, dict]:
    if spatial_policy == "preserve_aspect_ratio_letterbox":
        return _letterbox(
            frame,
            width=width,
            height=height,
            pad_value=pad_value,
        )
    if spatial_policy == "reference_content_crop_resize_no_pad":
        if crop_xywh is None:
            raise VideoProtocolError(
                "spatial_crop_missing",
                "no-pad spatial normalization requires crop_xywh",
            )
        return _crop_resize_no_pad(
            frame,
            width=width,
            height=height,
            crop_xywh=crop_xywh,
        )
    raise VideoProtocolError(
        "invalid_spatial_policy",
        f"unsupported spatial normalization policy: {spatial_policy!r}",
    )


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
    spatial_policy: str = "preserve_aspect_ratio_letterbox",
    crop_xywh: tuple[int, int, int, int] | None = None,
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
            normalized, current_transform = _normalize_frame(
                frame,
                width=width,
                height=height,
                pad_value=pad_value,
                spatial_policy=spatial_policy,
                crop_xywh=crop_xywh,
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
                spatial_policy=spatial_policy,
                crop_xywh=crop_xywh,
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
            normalized, current_transform = _normalize_frame(
                frame,
                width=width,
                height=height,
                pad_value=pad_value,
                spatial_policy=spatial_policy,
                crop_xywh=crop_xywh,
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
                spatial_policy=spatial_policy,
                crop_xywh=crop_xywh,
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
    spatial_policy: str = "preserve_aspect_ratio_letterbox",
    crop_xywh: tuple[int, int, int, int] | None = None,
) -> dict:
    if spatial_policy == "reference_content_crop_resize_no_pad":
        if crop_xywh is None:
            raise VideoProtocolError(
                "spatial_crop_missing",
                "no-pad spatial normalization requires crop_xywh",
            )
        x, y, crop_width, crop_height = crop_xywh
        if (
            x < 0
            or y < 0
            or crop_width <= 0
            or crop_height <= 0
            or x + crop_width > info.width
            or y + crop_height > info.height
            or crop_width * height != crop_height * width
        ):
            raise VideoProtocolError(
                "invalid_spatial_crop",
                "no-pad crop is incompatible with source/target metadata",
            )
        return {
            "policy": spatial_policy,
            "crop_xywh": list(crop_xywh),
            "scale": width / crop_width,
            "source_size": [info.width, info.height],
            "target_size": [width, height],
            "padding": None,
        }
    if spatial_policy != "preserve_aspect_ratio_letterbox":
        raise VideoProtocolError(
            "invalid_spatial_policy",
            f"unsupported spatial normalization policy: {spatial_policy!r}",
        )
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
