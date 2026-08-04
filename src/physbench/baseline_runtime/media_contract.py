"""One media boundary shared by every managed I2V Baseline.

The Dataset remains native.  A Baseline declares only its output canvas and
timeline; this module materializes the corresponding full-view conditioning
image and validates the generated video before any scene evaluator sees it.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


MEDIA_CONTRACT_SCHEMA_VERSION = "1.1"
LEGACY_MEDIA_CONTRACT_SCHEMA_VERSION = "1.0"
MEDIA_CONTRACT_POLICY = "standard_i2v_media_v1"


class MediaContractError(ValueError):
    """A stable, Baseline-attributable prediction protocol violation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MediaContractError(
            "media_contract_invalid",
            f"{label} must be a positive integer",
        )
    return value


def _positive_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise MediaContractError(
            "media_contract_invalid",
            f"{label} must be a positive finite number",
        )
    return float(value)


def plan_generation_timeline(
    temporal: Mapping[str, Any],
    *,
    target_physical_duration_s: float | None = None,
) -> dict[str, Any]:
    """Resolve one model-native frame count for a requested duration.

    The Baseline keeps its declared FPS.  For a bounded frame-count model we
    choose the shortest legal sequence whose last timestamp covers the target;
    if the model cannot reach the target, its maximum legal length is used.
    Fixed-length models retain their native frame count.
    """

    fps = _positive_number(temporal.get("fps"), label="output.timeline.fps")
    has_fixed_count = "num_frames" in temporal
    has_bounded_count = "max_frames" in temporal
    if has_fixed_count == has_bounded_count:
        raise MediaContractError(
            "media_contract_invalid",
            "I2V temporal policy requires exactly one of num_frames or "
            "max_frames",
        )
    valid_frame_rule = temporal.get("valid_frame_rule")
    if valid_frame_rule not in {None, "4n+1"}:
        raise MediaContractError(
            "media_contract_unsupported",
            f"unsupported I2V frame-count rule: {valid_frame_rule!r}",
        )
    if has_fixed_count:
        minimum = maximum = _positive_int(
            temporal["num_frames"],
            label="output.timeline.frame_count.value",
        )
        capability = {"rule": "fixed", "value": minimum}
    else:
        minimum = _positive_int(
            temporal.get("min_frames", 1),
            label="output.timeline.frame_count.minimum",
        )
        maximum = _positive_int(
            temporal["max_frames"],
            label="output.timeline.frame_count.maximum",
        )
        if minimum > maximum:
            raise MediaContractError(
                "media_contract_invalid",
                "output.timeline frame-count minimum exceeds maximum",
            )
        capability = {
            "rule": "bounded",
            "minimum": minimum,
            "maximum": maximum,
        }
    modulus = remainder = None
    if valid_frame_rule == "4n+1":
        modulus, remainder = 4, 1
        if minimum % modulus != remainder or maximum % modulus != remainder:
            raise MediaContractError(
                "media_contract_invalid",
                "frame-count bounds violate their modular rule",
            )
        capability.update({"modulus": modulus, "remainder": remainder})

    if target_physical_duration_s is None:
        requested = minimum if has_fixed_count else maximum
        target = None
    else:
        target = _positive_number(
            target_physical_duration_s,
            label="evaluation.target_physical_duration_s",
        )
        if has_fixed_count:
            requested = minimum
        else:
            requested = int(math.ceil(target * fps - 1e-9)) + 1
            if modulus is not None:
                requested += (remainder - requested) % modulus
            requested = min(maximum, max(minimum, requested))

    requested_duration = (requested - 1) / fps
    if target is None:
        alignment = "target_not_declared"
    elif abs(requested_duration - target) <= 1e-9:
        alignment = "exact"
    elif requested_duration > target:
        alignment = "covers_target_with_native_tail"
    else:
        alignment = "model_maximum_shorter_than_target"
    return {
        "fps": fps,
        "frame_count_capability": capability,
        "requested_num_frames": requested,
        "requested_physical_duration_s": requested_duration,
        "target_physical_duration_s": target,
        "duration_alignment": alignment,
    }


def build_i2v_media_contract(
    *,
    conditioning_asset: str,
    width: int,
    height: int,
    temporal: Mapping[str, Any],
    target_physical_duration_s: float | None = None,
) -> dict[str, Any]:
    """Build the only supported managed-I2V submission contract."""

    if not isinstance(conditioning_asset, str) or not conditioning_asset:
        raise MediaContractError(
            "media_contract_invalid",
            "conditioning_asset must be a non-empty string",
        )
    width = _positive_int(width, label="output.canvas.width")
    height = _positive_int(height, label="output.canvas.height")
    timeline_plan = plan_generation_timeline(
        temporal,
        target_physical_duration_s=target_physical_duration_s,
    )
    fps = timeline_plan["fps"]
    if target_physical_duration_s is None:
        schema_version = LEGACY_MEDIA_CONTRACT_SCHEMA_VERSION
        frame_count = timeline_plan["frame_count_capability"]
        output_timeline = {
            "fps": fps,
            "start_time_s": 0.0,
            "frame_count": frame_count,
        }
        evaluation = {
            "spatial_view": "conditioning_content",
            "temporal_view": "physical_time_from_frame_zero",
        }
    else:
        schema_version = MEDIA_CONTRACT_SCHEMA_VERSION
        frame_count = {
            "rule": "fixed",
            "value": timeline_plan["requested_num_frames"],
        }
        output_timeline = {
            "fps": fps,
            "start_time_s": 0.0,
            "frame_count": frame_count,
            "requested_physical_duration_s": timeline_plan[
                "requested_physical_duration_s"
            ],
        }
        evaluation = {
            "spatial_view": "conditioning_content",
            "temporal_view": "physical_time_from_frame_zero",
            "target_physical_duration_s": timeline_plan[
                "target_physical_duration_s"
            ],
            "duration_alignment": timeline_plan["duration_alignment"],
            "long_prediction_policy": (
                "evaluate_reference_physical_duration"
            ),
            "short_prediction_policy": (
                "evaluate_prediction_physical_duration"
            ),
        }

    contract = {
        "schema_version": schema_version,
        "policy": MEDIA_CONTRACT_POLICY,
        "conditioning": {
            "asset": conditioning_asset,
            "transform": "centered_contain",
            "margin_fill": "edge_replicate",
        },
        "output": {
            "canvas": {"width": width, "height": height},
            "timeline": output_timeline,
        },
        "evaluation": evaluation,
    }
    validate_media_contract(contract)
    return contract


def validate_media_contract(value: Any) -> dict[str, Any]:
    """Validate and return a plain copy of the standard I2V contract."""

    if not isinstance(value, Mapping):
        raise MediaContractError(
            "media_contract_missing",
            "managed I2V input requires native_inputs.media_contract",
        )
    required = {
        "schema_version",
        "policy",
        "conditioning",
        "output",
        "evaluation",
    }
    if set(value) != required:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract fields must be exactly " + repr(sorted(required)),
        )
    schema_version = value.get("schema_version")
    if (
        schema_version
        not in {
            LEGACY_MEDIA_CONTRACT_SCHEMA_VERSION,
            MEDIA_CONTRACT_SCHEMA_VERSION,
        }
        or value.get("policy") != MEDIA_CONTRACT_POLICY
    ):
        raise MediaContractError(
            "media_contract_unsupported",
            "unsupported managed I2V media contract",
        )

    conditioning = value.get("conditioning")
    if not isinstance(conditioning, Mapping) or set(conditioning) != {
        "asset",
        "transform",
        "margin_fill",
    }:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract.conditioning fields are invalid",
        )
    if (
        not isinstance(conditioning.get("asset"), str)
        or not conditioning["asset"]
        or conditioning.get("transform") != "centered_contain"
        or conditioning.get("margin_fill") != "edge_replicate"
    ):
        raise MediaContractError(
            "media_contract_unsupported",
            "I2V conditioning must use centered contain with edge replication",
        )

    output = value.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "canvas",
        "timeline",
    }:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract.output fields are invalid",
        )
    canvas = output.get("canvas")
    if not isinstance(canvas, Mapping) or set(canvas) != {"width", "height"}:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract.output.canvas fields are invalid",
        )
    _positive_int(canvas.get("width"), label="output.canvas.width")
    _positive_int(canvas.get("height"), label="output.canvas.height")

    timeline = output.get("timeline")
    timeline_fields = {"fps", "start_time_s", "frame_count"}
    if schema_version == MEDIA_CONTRACT_SCHEMA_VERSION:
        timeline_fields.add("requested_physical_duration_s")
    if not isinstance(timeline, Mapping) or set(timeline) != timeline_fields:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract.output.timeline fields are invalid",
        )
    timeline_fps = _positive_number(
        timeline.get("fps"), label="output.timeline.fps"
    )
    if timeline.get("start_time_s") != 0.0:
        raise MediaContractError(
            "media_contract_unsupported",
            "prediction timeline must begin at physical time zero",
        )
    frame_count = timeline.get("frame_count")
    if not isinstance(frame_count, Mapping):
        raise MediaContractError(
            "media_contract_invalid",
            "output.timeline.frame_count must be an object",
        )
    rule = frame_count.get("rule")
    common = {"rule"}
    if "modulus" in frame_count or "remainder" in frame_count:
        common |= {"modulus", "remainder"}
        modulus = _positive_int(
            frame_count.get("modulus"),
            label="output.timeline.frame_count.modulus",
        )
        remainder = frame_count.get("remainder")
        if (
            isinstance(remainder, bool)
            or not isinstance(remainder, int)
            or remainder < 0
            or remainder >= modulus
        ):
            raise MediaContractError(
                "media_contract_invalid",
                "frame-count remainder must be in [0, modulus)",
            )
    if rule == "fixed":
        if set(frame_count) != common | {"value"}:
            raise MediaContractError(
                "media_contract_invalid",
                "fixed frame-count fields are invalid",
            )
        minimum = maximum = _positive_int(
            frame_count.get("value"),
            label="output.timeline.frame_count.value",
        )
    elif rule == "bounded":
        if set(frame_count) != common | {"minimum", "maximum"}:
            raise MediaContractError(
                "media_contract_invalid",
                "bounded frame-count fields are invalid",
            )
        minimum = _positive_int(
            frame_count.get("minimum"),
            label="output.timeline.frame_count.minimum",
        )
        maximum = _positive_int(
            frame_count.get("maximum"),
            label="output.timeline.frame_count.maximum",
        )
        if minimum > maximum:
            raise MediaContractError(
                "media_contract_invalid",
                "frame-count minimum exceeds maximum",
            )
    else:
        raise MediaContractError(
            "media_contract_invalid",
            "frame-count rule must be fixed or bounded",
        )
    if "modulus" in frame_count:
        modulus = int(frame_count["modulus"])
        remainder = int(frame_count["remainder"])
        if minimum % modulus != remainder or maximum % modulus != remainder:
            raise MediaContractError(
                "media_contract_invalid",
                "frame-count bounds violate their modular rule",
            )

    if schema_version == MEDIA_CONTRACT_SCHEMA_VERSION:
        requested_duration = _positive_number(
            timeline.get("requested_physical_duration_s"),
            label="output.timeline.requested_physical_duration_s",
        )
        expected_duration = (minimum - 1) / timeline_fps
        if maximum != minimum or abs(requested_duration - expected_duration) > 1e-9:
            raise MediaContractError(
                "media_contract_invalid",
                "requested physical duration must equal the fixed output "
                "timeline's last-frame timestamp",
            )

    evaluation = value.get("evaluation")
    evaluation_fields = {"spatial_view", "temporal_view"}
    if schema_version == MEDIA_CONTRACT_SCHEMA_VERSION:
        evaluation_fields |= {
            "target_physical_duration_s",
            "duration_alignment",
            "long_prediction_policy",
            "short_prediction_policy",
        }
    if not isinstance(evaluation, Mapping) or set(evaluation) != evaluation_fields:
        raise MediaContractError(
            "media_contract_invalid",
            "media_contract.evaluation fields are invalid",
        )
    if (
        evaluation.get("spatial_view") != "conditioning_content"
        or evaluation.get("temporal_view")
        != "physical_time_from_frame_zero"
    ):
        raise MediaContractError(
            "media_contract_unsupported",
            "unsupported I2V evaluation view",
        )
    if schema_version == MEDIA_CONTRACT_SCHEMA_VERSION:
        target_duration = _positive_number(
            evaluation.get("target_physical_duration_s"),
            label="evaluation.target_physical_duration_s",
        )
        if abs(requested_duration - target_duration) <= 1e-9:
            expected_alignment = "exact"
        elif requested_duration > target_duration:
            expected_alignment = "covers_target_with_native_tail"
        else:
            expected_alignment = "model_maximum_shorter_than_target"
        if evaluation.get("duration_alignment") != expected_alignment:
            raise MediaContractError(
                "media_contract_invalid",
                "evaluation.duration_alignment does not describe the "
                "requested and target physical durations",
            )
        if (
            evaluation.get("long_prediction_policy")
            != "evaluate_reference_physical_duration"
            or evaluation.get("short_prediction_policy")
            != "evaluate_prediction_physical_duration"
        ):
            raise MediaContractError(
                "media_contract_unsupported",
                "unsupported prediction/reference duration policy",
            )
    return json.loads(json.dumps(value))


def centered_contain_rect(
    source_width: int,
    source_height: int,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Return the exact integer content rectangle for centered contain."""

    source_width = _positive_int(source_width, label="source_width")
    source_height = _positive_int(source_height, label="source_height")
    canvas_width = _positive_int(canvas_width, label="canvas_width")
    canvas_height = _positive_int(canvas_height, label="canvas_height")
    scale = min(
        canvas_width / source_width,
        canvas_height / source_height,
    )
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    if width * source_height != height * source_width:
        raise MediaContractError(
            "conditioning_aspect_not_exact",
            "model canvas cannot represent the conditioning aspect ratio "
            "exactly at integer pixels: "
            f"source={source_width}x{source_height}, "
            f"canvas={canvas_width}x{canvas_height}, "
            f"contained={width}x{height}",
        )
    return (
        (canvas_width - width) // 2,
        (canvas_height - height) // 2,
        width,
        height,
    )


def edge_replicated_contain_filter(
    source_width: int,
    source_height: int,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[str, tuple[int, int, int, int]]:
    """Return the canonical ffmpeg filter and its content rectangle."""

    left, top, width, height = centered_contain_rect(
        source_width,
        source_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    right = canvas_width - width - left
    bottom = canvas_height - height - top
    return (
        f"scale={width}:{height},"
        f"pad={canvas_width}:{canvas_height}:{left}:{top}:color=black,"
        f"fillborders=left={left}:right={right}:top={top}:"
        f"bottom={bottom}:mode=smear,setsar=1",
        (left, top, width, height),
    )


def _fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return (
            float(numerator) / float(denominator)
            if float(denominator)
            else None
        )
    return float(value)


def require_media_tools() -> None:
    missing = [
        name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None
    ]
    if missing:
        raise FileNotFoundError(f"required media tools are missing: {missing}")


def probe_media(path: str | Path, *, count_frames: bool = True) -> dict[str, Any]:
    """Probe an image or video without importing evaluator/model libraries."""

    require_media_tools()
    if not isinstance(path, (str, Path)) or not str(path):
        raise MediaContractError(
            "prediction_video_missing",
            "media path must be a non-empty string or Path",
        )
    path = Path(path)
    if not path.is_file():
        raise MediaContractError(
            "prediction_video_missing",
            f"media file does not exist: {path}",
        )
    count_args = ["-count_frames"] if count_frames else []
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *count_args,
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,r_frame_rate,"
                "nb_frames,nb_read_frames,start_time,duration:"
                "format=start_time,duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise MediaContractError(
            "prediction_video_unreadable",
            f"ffprobe cannot read {path}: {completed.stderr.strip()}",
        )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise MediaContractError(
            "prediction_video_metadata_invalid",
            f"media has no valid video stream: {path}",
        ) from exc
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    duration_value = stream.get("duration") or payload.get("format", {}).get(
        "duration"
    )
    start_value = stream.get("start_time")
    if start_value in {None, "N/A"}:
        start_value = payload.get("format", {}).get("start_time")
    try:
        fps = _fraction(
            stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        )
        frames = (
            int(frame_value)
            if frame_value not in {None, "N/A"}
            else None
        )
        duration = (
            float(duration_value)
            if duration_value not in {None, "N/A"}
            else None
        )
        start_time = (
            float(start_value)
            if start_value not in {None, "N/A"}
            else None
        )
    except (TypeError, ValueError) as exc:
        raise MediaContractError(
            "prediction_video_metadata_invalid",
            f"media has malformed numeric metadata: {path}",
        ) from exc
    if (
        (fps is not None and (not math.isfinite(fps) or fps <= 0))
        or (
            duration is not None
            and (not math.isfinite(duration) or duration < 0)
        )
        or (start_time is not None and not math.isfinite(start_time))
    ):
        raise MediaContractError(
            "prediction_video_metadata_invalid",
            f"media has non-finite or non-positive metadata: {path}",
        )
    if width <= 0 or height <= 0:
        raise MediaContractError(
            "prediction_video_metadata_invalid",
            f"media has invalid dimensions: {width}x{height}",
        )
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "start_time_s": start_time,
        "duration_s": duration,
    }


def materialize_i2v_conditioning(
    source: str | Path,
    output: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one canonical contain derivative and verify its dimensions."""

    contract = validate_media_contract(contract)
    source = Path(source)
    output = Path(output)
    source_probe = probe_media(source, count_frames=False)
    canvas = contract["output"]["canvas"]
    media_filter, rect = edge_replicated_contain_filter(
        int(source_probe["width"]),
        int(source_probe["height"]),
        canvas_width=int(canvas["width"]),
        canvas_height=int(canvas["height"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        media_filter,
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output),
    ]
    subprocess.run(command, check=True)
    output_probe = probe_media(output, count_frames=False)
    if (
        output_probe["width"] != canvas["width"]
        or output_probe["height"] != canvas["height"]
    ):
        raise RuntimeError(
            "canonical conditioning materialization produced the wrong "
            f"canvas: {output_probe}"
        )
    return {
        "policy": MEDIA_CONTRACT_POLICY,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_probe": source_probe,
        "output_probe": output_probe,
        "content_rect_xywh": list(rect),
        "command": command,
    }


def validate_prediction_video(
    path: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when generated media differs from its sealed contract."""

    contract = validate_media_contract(contract)
    probe = probe_media(path, count_frames=True)
    canvas = contract["output"]["canvas"]
    if (
        probe["width"] != canvas["width"]
        or probe["height"] != canvas["height"]
    ):
        raise MediaContractError(
            "prediction_canvas_mismatch",
            "prediction dimensions differ from the sealed model canvas: "
            f"video={probe['width']}x{probe['height']}, "
            f"contract={canvas['width']}x{canvas['height']}",
        )
    timeline = contract["output"]["timeline"]
    actual_start = probe.get("start_time_s")
    if actual_start is None or abs(float(actual_start)) > 1e-3:
        raise MediaContractError(
            "prediction_start_time_mismatch",
            "prediction timeline does not begin at physical time zero: "
            f"video={actual_start}, contract=0.0",
        )
    expected_fps = float(timeline["fps"])
    actual_fps = probe.get("fps")
    if actual_fps is None or abs(float(actual_fps) - expected_fps) > max(
        0.01,
        expected_fps * 1e-3,
    ):
        raise MediaContractError(
            "prediction_fps_mismatch",
            "prediction FPS differs from the sealed physical timeline: "
            f"video={actual_fps}, contract={expected_fps}",
        )
    frames = probe.get("frames")
    if frames is None or frames <= 0:
        raise MediaContractError(
            "prediction_frame_count_missing",
            "prediction frame count cannot be verified",
        )
    frame_count = timeline["frame_count"]
    if frame_count["rule"] == "fixed":
        minimum = maximum = int(frame_count["value"])
    else:
        minimum = int(frame_count["minimum"])
        maximum = int(frame_count["maximum"])
    if not minimum <= frames <= maximum:
        raise MediaContractError(
            "prediction_frame_count_mismatch",
            "prediction frame count is outside the sealed range: "
            f"video={frames}, contract=[{minimum}, {maximum}]",
        )
    if "modulus" in frame_count and (
        frames % int(frame_count["modulus"])
        != int(frame_count["remainder"])
    ):
        raise MediaContractError(
            "prediction_frame_count_rule_mismatch",
            "prediction frame count violates the sealed modular rule: "
            f"video={frames}, rule={frame_count}",
        )
    return {
        "status": "valid",
        "policy": MEDIA_CONTRACT_POLICY,
        "probe": probe,
    }


__all__ = [
    "MEDIA_CONTRACT_POLICY",
    "MEDIA_CONTRACT_SCHEMA_VERSION",
    "MediaContractError",
    "build_i2v_media_contract",
    "centered_contain_rect",
    "edge_replicated_contain_filter",
    "materialize_i2v_conditioning",
    "plan_generation_timeline",
    "probe_media",
    "require_media_tools",
    "validate_media_contract",
    "validate_prediction_video",
]
