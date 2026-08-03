from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator) if float(denominator) else None
    return float(value)


class Wan22MediaAdapter:
    """Create WAN-specific, auditable media derivatives without touching source assets."""

    def __init__(self, policy: dict[str, Any]):
        self.policy = policy
        self.width = int(policy["width"])
        self.height = int(policy["height"])
        self.fps = int(policy["fps"])
        self.max_frames = int(policy["max_frames"])
        self.min_frames = int(policy.get("min_frames", 5))
        self.pad_color = str(policy.get("pad_color", "black"))
        self.pad_mode = str(policy.get("pad_mode", "color"))
        if self.pad_mode not in {"color", "edge"}:
            raise ValueError("WAN pad_mode must be color or edge")
        if self.width % 32 or self.height % 32:
            raise ValueError(
                "WAN I2V target width and height must be divisible by 32"
            )
        bucket_config = policy.get("aspect_ratio_buckets", {})
        self.bucket_enabled = bool(bucket_config.get("enabled", False))
        self.buckets: dict[str, dict[str, Any]] = {}
        self.scene_buckets: dict[str, str] = {}
        if self.bucket_enabled:
            for name, value in bucket_config.get("buckets", {}).items():
                width, height = int(value["width"]), int(value["height"])
                if width % 32 or height % 32:
                    raise ValueError(
                        f"WAN I2V bucket {name} dimensions must be "
                        "divisible by 32"
                    )
                self.buckets[name] = {"name": name, "width": width, "height": height}
                for scene_id in value.get("scene_ids", []):
                    if scene_id in self.scene_buckets:
                        raise ValueError(f"scene {scene_id} is assigned to multiple WAN buckets")
                    self.scene_buckets[str(scene_id)] = name
            if not self.buckets:
                raise ValueError("enabled aspect_ratio_buckets requires at least one bucket")
        if self.max_frames % 4 != 1 or self.min_frames % 4 != 1:
            raise ValueError("WAN frame limits must satisfy 4n+1")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames cannot exceed max_frames")

    @property
    def dynamic_resolution(self) -> bool:
        return self.bucket_enabled

    @property
    def max_pixels(self) -> int:
        profiles = self.buckets.values() if self.bucket_enabled else [self.profile(None)]
        return max(int(item["width"]) * int(item["height"]) for item in profiles)

    def profile(self, scene_id: str | None) -> dict[str, Any]:
        if not self.bucket_enabled:
            return {"name": "default", "width": self.width, "height": self.height}
        name = self.scene_buckets.get(str(scene_id))
        if name is None:
            raise ValueError(f"no WAN aspect-ratio bucket configured for scene {scene_id}")
        return dict(self.buckets[name])

    @staticmethod
    def require_tools() -> None:
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            raise FileNotFoundError(f"missing media tools: {missing}")

    @staticmethod
    def probe(path: str | Path, *, count_frames: bool = True) -> dict[str, Any]:
        count_args = ["-count_frames"] if count_frames else []
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                *count_args, "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
                "-of", "json", str(path),
            ],
            check=True, text=True, capture_output=True,
        )
        payload = json.loads(result.stdout)
        if not payload.get("streams"):
            raise ValueError(f"no video/image stream: {path}")
        stream = payload["streams"][0]
        duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
        frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
            "frames": int(frame_value) if frame_value not in {None, "N/A"} else None,
            "duration_s": float(duration_value) if duration_value not in {None, "N/A"} else None,
        }

    def frame_count(self, source_probe: dict[str, Any], *, speed_factor: float = 1.0) -> int:
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        duration = source_probe.get("duration_s")
        if duration is None:
            frames, fps = source_probe.get("frames"), source_probe.get("fps")
            if frames is None or not fps:
                raise ValueError("cannot infer source duration")
            duration = frames / fps
        physical_duration = float(duration) / speed_factor
        available = int(math.floor(physical_duration * self.fps + 1e-6))
        frames = min(self.max_frames, available)
        frames -= (frames - 1) % 4
        if frames < self.min_frames:
            raise ValueError(
                f"source is too short for WAN: encoded duration {duration:.4f}s at "
                f"{speed_factor:g}x gives {physical_duration:.4f}s and {frames} frames at {self.fps} fps; "
                f"minimum is {self.min_frames}"
            )
        return frames

    def generation_frame_count(
        self,
        source_probe: dict[str, Any],
        *,
        speed_factor: float = 1.0,
    ) -> int:
        """Return a valid model length whose last timestamp covers the source.

        A video with N frames at ``fps`` ends at timestamp ``(N-1)/fps``.
        Reusing the number of safely decodable reference frames for generation
        can therefore leave the prediction one timestamp short. Generation has
        no source-frame decoding constraint, so round upward to the next
        ``4n+1`` length, capped by the model maximum.
        """
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        duration = source_probe.get("duration_s")
        if duration is None:
            frames, fps = source_probe.get("frames"), source_probe.get("fps")
            if frames is None or not fps:
                raise ValueError("cannot infer source duration")
            duration = max(0.0, (float(frames) - 1.0) / float(fps))
        physical_duration = float(duration) / speed_factor
        timestamp_steps = int(
            math.ceil(physical_duration * self.fps - 1e-9)
        )
        frames = timestamp_steps + 1
        remainder = (frames - 1) % 4
        if remainder:
            frames += 4 - remainder
        frames = min(self.max_frames, frames)
        if frames < self.min_frames:
            frames = self.min_frames
        return frames

    @staticmethod
    def _contain_geometry(
        source_width: int,
        source_height: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, int, int]:
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        left = (width - resized_width) // 2
        right = width - resized_width - left
        top = (height - resized_height) // 2
        bottom = height - resized_height - top
        return resized_width, resized_height, left, right, top, bottom

    def _spatial_filter(
        self,
        width: int,
        height: int,
        *,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> str:
        if self.pad_mode == "color":
            return (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:"
                f"color={self.pad_color},setsar=1"
            )
        if source_width is None or source_height is None:
            raise ValueError(
                "edge-replicated WAN contain requires source dimensions"
            )
        (
            resized_width,
            resized_height,
            left,
            right,
            top,
            bottom,
        ) = self._contain_geometry(
            source_width,
            source_height,
            width,
            height,
        )
        return (
            f"scale={resized_width}:{resized_height},"
            f"pad={width}:{height}:{left}:{top}:color=black,"
            f"fillborders=left={left}:right={right}:top={top}:"
            f"bottom={bottom}:mode=smear,setsar=1"
        )

    def normalize_video(
        self,
        source: Path,
        output: Path,
        *,
        materialize: bool,
        speed_factor: float = 1.0,
        scene_id: str | None = None,
    ) -> dict[str, Any]:
        if speed_factor <= 0:
            raise ValueError("speed_factor must be positive")
        exists = source.is_file()
        if materialize and not exists:
            raise FileNotFoundError(f"source video not found: {source}")
        # Duration is sufficient for target-frame calculation. Avoid decoding an entire
        # lossless source merely to count frames; output verification still uses an exact count.
        source_probe = self.probe(source, count_frames=False) if exists else None
        frames = self.frame_count(source_probe, speed_factor=speed_factor) if source_probe else self.max_frames
        generation_frames = (
            self.generation_frame_count(
                source_probe, speed_factor=speed_factor
            )
            if source_probe
            else self.max_frames
        )
        profile = self.profile(scene_id)
        width, height = int(profile["width"]), int(profile["height"])
        temporal_filters = []
        if abs(speed_factor - 1.0) > 1e-12:
            temporal_filters.append(f"setpts=PTS/{speed_factor:g}")
        temporal_filters.append(f"fps={self.fps}")
        temporal_filters.append(
            self._spatial_filter(
                width,
                height,
                source_width=(
                    int(source_probe["width"]) if source_probe else None
                ),
                source_height=(
                    int(source_probe["height"]) if source_probe else None
                ),
            )
        )
        command = [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", ",".join(temporal_filters),
            "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(self.fps),
            "-movflags", "+faststart", str(output),
        ]
        output_probe = None
        if materialize:
            self.require_tools()
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.is_file():
                subprocess.run(command, check=True)
            output_probe = self.probe(output)
            if (
                output_probe["width"] != width
                or output_probe["height"] != height
                or output_probe["frames"] != frames
                or abs(float(output_probe["fps"] or 0) - self.fps) > 1e-6
            ):
                raise RuntimeError(f"normalized video failed verification: {output_probe}")
        return {
            "kind": "video",
            "status": "materialized" if materialize else "planned",
            "source": str(source),
            "source_probe": source_probe,
            "output": str(output),
            "output_probe": output_probe,
            "target_frames": frames,
            "generation_target_frames": generation_frames,
            "aspect_ratio_bucket": profile,
            "time_mapping": {
                "start_s": 0.0,
                "target_fps": self.fps,
                "encoded_to_physical_speed": speed_factor,
                "encoded_duration_s": source_probe.get("duration_s") if source_probe else None,
                "physical_duration_s": (
                    float(source_probe["duration_s"]) / speed_factor
                    if source_probe and source_probe.get("duration_s") is not None else None
                ),
                "policy": "restore_physical_time_then_timestamp_resample_prefix_without_loop",
            },
            "spatial_mapping": {
                "width": width,
                "height": height,
                "policy": (
                    "aspect_preserving_contain_edge_replicate"
                    if self.pad_mode == "edge"
                    else "aspect_preserving_fit_and_pad"
                ),
                "pad_mode": self.pad_mode,
                **(
                    {"pad_color": self.pad_color}
                    if self.pad_mode == "color"
                    else {}
                ),
            },
            "command": command,
        }

    def normalize_first_frame(
        self,
        source: Path,
        output: Path,
        *,
        source_is_video: bool,
        materialize: bool,
        scene_id: str | None = None,
    ) -> dict[str, Any]:
        exists = source.is_file()
        if materialize and not exists:
            raise FileNotFoundError(f"first-frame source not found: {source}")
        source_probe = self.probe(source) if exists else None
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
        profile = self.profile(scene_id)
        width, height = int(profile["width"]), int(profile["height"])
        command += [
            "-vf",
            self._spatial_filter(
                width,
                height,
                source_width=(
                    int(source_probe["width"]) if source_probe else None
                ),
                source_height=(
                    int(source_probe["height"]) if source_probe else None
                ),
            ),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(output),
        ]
        output_probe = None
        if materialize:
            self.require_tools()
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.is_file():
                subprocess.run(command, check=True)
            output_probe = self.probe(output)
            if (
                output_probe["width"] != width
                or output_probe["height"] != height
            ):
                raise RuntimeError(
                    "normalized first frame failed verification: "
                    f"{output_probe}"
                )
        return {
            "kind": (
                "first_frame_from_video"
                if source_is_video
                else "first_frame_from_image"
            ),
            "status": "materialized" if materialize else "planned",
            "source": str(source),
            "output": str(output),
            "output_probe": output_probe,
            "aspect_ratio_bucket": profile,
            "spatial_mapping": {
                "width": width,
                "height": height,
                "policy": (
                    "aspect_preserving_contain_edge_replicate"
                    if self.pad_mode == "edge"
                    else "aspect_preserving_fit_and_pad"
                ),
                "pad_mode": self.pad_mode,
                **(
                    {"pad_color": self.pad_color}
                    if self.pad_mode == "color"
                    else {}
                ),
            },
            "command": command,
        }
