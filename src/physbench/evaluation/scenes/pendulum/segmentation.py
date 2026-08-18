from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...common.errors import SceneAnalysisError
from ...common.masks.sam2 import MaskPrompt, Sam2VideoSegmenter


class SegmentationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MotionPrompt:
    frame_index: int
    box_xyxy: np.ndarray
    motion_box_xyxy: np.ndarray
    points_xy: np.ndarray
    point_labels: np.ndarray
    proposal_score: float


def estimate_background(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise SegmentationError("empty_timeline", "cannot estimate an empty video")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _positive_points(component: np.ndarray, count: int = 6) -> np.ndarray:
    ys, xs = np.where(component)
    if not len(xs):
        raise SegmentationError(
            "empty_motion_proposal", "motion proposal contains no pixels"
        )
    y0, y1 = int(ys.min()), int(ys.max())
    band = max(3, int(round((y1 - y0 + 1) * 0.012)))
    points: list[tuple[float, float]] = []
    for fraction in np.linspace(0.08, 0.94, count):
        target_y = int(round(y0 + fraction * (y1 - y0)))
        selected = np.abs(ys - target_y) <= band
        if selected.any():
            points.append(
                (float(np.median(xs[selected])), float(np.median(ys[selected])))
            )
    if not points:
        points.append((float(np.median(xs)), float(np.median(ys))))
    return np.asarray(points, dtype=np.float32)


def build_motion_prompt(
    frames: list[np.ndarray],
    *,
    threshold: float,
    min_area: int,
    expand: float,
    min_box_side: int,
) -> MotionPrompt:
    """Find a bright moving rope/bob assembly without assuming a fixed frame."""
    background = cv2.GaussianBlur(estimate_background(frames), (5, 5), 0)
    background_gray = cv2.cvtColor(
        background, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    height, width = background.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    margin_x = max(2, int(width * 0.015))
    margin_y = max(2, int(height * 0.015))
    best: tuple[
        float, int, np.ndarray, tuple[int, int, int, int]
    ] | None = None
    for frame_index, frame in enumerate(frames):
        current = cv2.GaussianBlur(frame, (5, 5), 0)
        current_gray = cv2.cvtColor(
            current, cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        rough = (
            (current_gray - background_gray >= threshold).astype(np.uint8) * 255
        )
        rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(rough)
        for label in range(1, count):
            x, y, box_width, box_height, area = (
                int(value) for value in stats[label]
            )
            if (
                area < min_area
                or box_height < max(min_box_side, int(height * 0.08))
            ):
                continue
            if (
                x <= margin_x
                or y <= margin_y
                or x + box_width >= width - margin_x
                or y + box_height >= height - margin_y
            ):
                continue
            component = labels == label
            score = float(
                area
                * (1.0 + 3.0 * box_height / height)
                * (1.0 + 2.0 * box_width / width)
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    frame_index,
                    component.copy(),
                    (x, y, box_width, box_height),
                )
    if best is None:
        raise SegmentationError(
            "motion_prompt_failed",
            "could not locate a moving pendulum proposal",
        )

    score, frame_index, component, (x, y, box_width, box_height) = best
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    expanded_width = max(float(min_box_side), box_width * expand)
    expanded_height = max(float(min_box_side), box_height * expand)
    box = np.asarray(
        [
            max(0.0, center_x - expanded_width / 2),
            max(0.0, center_y - expanded_height / 2),
            min(width - 1.0, center_x + expanded_width / 2),
            min(height - 1.0, center_y + expanded_height / 2),
        ],
        dtype=np.float32,
    )
    points = _positive_points(component)
    labels_out = np.ones(len(points), dtype=np.int32)

    ys, xs = np.where(component)
    top = ys <= np.quantile(ys, 0.08)
    pivot_x = float(np.median(xs[top]))
    pole_y = float(y + 0.58 * box_height)
    nearest = points[np.argmin(np.abs(points[:, 1] - pole_y))]
    if abs(float(nearest[0]) - pivot_x) >= max(12.0, width * 0.015):
        points = np.vstack(
            [points, np.asarray([[pivot_x, pole_y]], dtype=np.float32)]
        )
        labels_out = np.concatenate(
            [labels_out, np.asarray([0], dtype=np.int32)]
        )
    motion_box = np.asarray(
        [x, y, x + box_width - 1, y + box_height - 1],
        dtype=np.float32,
    )
    return MotionPrompt(
        frame_index=frame_index,
        box_xyxy=box,
        motion_box_xyxy=motion_box,
        points_xy=points,
        point_labels=labels_out,
        proposal_score=score,
    )


class Sam2PendulumSegmenter:
    """Pendulum-specific prompt builder backed by the shared SAM2 adapter."""

    def __init__(self, config: dict[str, Any]):
        self._backend = Sam2VideoSegmenter(config)

    def describe(self) -> dict[str, Any]:
        return self._backend.describe()

    def segment(
        self,
        frames: list[np.ndarray],
        *,
        proposal_config: dict[str, Any],
        prompt: MotionPrompt | None = None,
        prompt_source: str = "independent_motion_proposal",
    ) -> tuple[list[np.ndarray], dict[str, Any], MotionPrompt]:
        if prompt is None:
            prompt = build_motion_prompt(
                frames,
                threshold=float(proposal_config["threshold"]),
                min_area=int(proposal_config["min_area"]),
                expand=float(proposal_config["box_expand"]),
                min_box_side=int(proposal_config["min_box_side"]),
            )
        common_prompt = MaskPrompt(
            frame_index=prompt.frame_index,
            box_xyxy=prompt.box_xyxy,
            points_xy=prompt.points_xy,
            point_labels=prompt.point_labels,
            metadata={
                "source": prompt_source,
                "motion_box_xyxy": prompt.motion_box_xyxy.tolist(),
                "proposal_score": prompt.proposal_score,
            },
        )
        try:
            masks, backend_metadata = self._backend.segment(
                frames,
                prompt=common_prompt,
                temporary_prefix="physbench_pendulum_",
            )
        except SceneAnalysisError as exc:
            raise SegmentationError(exc.code, str(exc)) from exc

        # SAM2 may expose masks backed by a read-only decoded buffer.  The
        # pendulum crop below is intentionally in-place, so first detach every
        # mask from backend-owned storage.
        masks = [np.array(mask, copy=True) for mask in masks]
        motion_top = max(0, int(round(prompt.motion_box_xyxy[1])))
        if motion_top:
            for mask in masks:
                mask[:motion_top] = 0
        return masks, {
            "prompt_frame": prompt.frame_index,
            "prompt_source": prompt_source,
            "prompt_box_xyxy": prompt.box_xyxy.tolist(),
            "motion_box_xyxy": prompt.motion_box_xyxy.tolist(),
            "prompt_points_xy": prompt.points_xy.tolist(),
            "prompt_point_labels": prompt.point_labels.tolist(),
            "proposal_score": prompt.proposal_score,
            **backend_metadata,
        }, prompt
