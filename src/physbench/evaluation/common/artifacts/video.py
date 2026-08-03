"""Browser-compatible, streaming MP4 writer for evaluation diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np


class CompatibleMp4Writer:
    """Stream BGR frames to FFmpeg and atomically publish a fast-start MP4."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        size: tuple[int, int],
        codec: str = "h264",
    ) -> None:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("video dimensions must be positive")
        if width % 2 or height % 2:
            raise ValueError("yuv420p video dimensions must be even")
        if not np.isfinite(fps) or fps <= 0.0 or fps > 120.0:
            raise ValueError("video fps must be in (0, 120]")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for visualization videos")
        encoder = {
            "h264": "libx264",
            "avc1": "libx264",
        }.get(codec)
        if encoder is None:
            raise ValueError(
                "visualization codec must be h264 or avc1"
            )

        self.path = path
        self.width = width
        self.height = height
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = path.with_name(
            f".{path.stem}.{uuid.uuid4().hex}.tmp.mp4"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        command.extend(["-preset", "veryfast", "-crf", "20"])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self._temporary),
            ]
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None or self._process.stderr is None:
            self.abort()
            raise RuntimeError("failed to open FFmpeg visualization pipes")

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed visualization video")
        array = np.asarray(frame)
        expected = (self.height, self.width, 3)
        if array.shape != expected:
            raise ValueError(
                f"visualization frame has shape {array.shape}, expected {expected}"
            )
        if array.dtype != np.uint8:
            raise ValueError("visualization frame must use uint8 BGR pixels")
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            message = self._finish_failed_process()
            raise RuntimeError(
                f"FFmpeg visualization writer failed: {message}"
            ) from exc

    def _finish_failed_process(self) -> str:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        assert self._process.stderr is not None
        message = self._process.stderr.read().decode("utf-8", errors="replace")
        self._process.stderr.close()
        self._process.wait()
        self._closed = True
        self._temporary.unlink(missing_ok=True)
        return message.strip() or f"exit code {self._process.returncode}"

    def close(self) -> None:
        if self._closed:
            return
        assert self._process.stdin is not None
        assert self._process.stderr is not None
        self._process.stdin.close()
        message = self._process.stderr.read().decode("utf-8", errors="replace")
        self._process.stderr.close()
        return_code = self._process.wait()
        self._closed = True
        if return_code != 0:
            self._temporary.unlink(missing_ok=True)
            detail = message.strip() or f"exit code {return_code}"
            raise RuntimeError(f"FFmpeg visualization writer failed: {detail}")
        if not self._temporary.is_file() or self._temporary.stat().st_size <= 0:
            self._temporary.unlink(missing_ok=True)
            raise RuntimeError("FFmpeg produced no visualization video")
        os.replace(self._temporary, self.path)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        if self._process.stderr is not None and not self._process.stderr.closed:
            self._process.stderr.close()
        self._temporary.unlink(missing_ok=True)


__all__ = ["CompatibleMp4Writer"]
