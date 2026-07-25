from __future__ import annotations

import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
    """Lazy, reusable SAM2.1 backend."""

    def __init__(self, config: dict[str, Any]):
        self.model_id = str(config["model_id"])
        self.requested_device = str(config.get("device", "auto"))
        self._predictor: Any | None = None
        self._torch: Any | None = None
        self.device: str | None = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except ImportError as exc:
            raise SegmentationError(
                "sam2_dependency_missing",
                "SAM2 evaluation dependencies are not installed",
            ) from exc
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SegmentationError(
                "cuda_unavailable", "SAM2 evaluator requested CUDA but none is available"
            )
        try:
            predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id, device=device
            )
        except Exception as exc:
            raise SegmentationError(
                "sam2_model_load_failed",
                f"failed to load SAM2 model {self.model_id}: {exc}",
            ) from exc
        self._torch = torch
        self._predictor = predictor
        self.device = device

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "sam2_video_predictor",
            "model_id": self.model_id,
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "reuse_policy": "one_model_instance_per_task_evaluator",
        }

    def segment(
        self,
        frames: list[np.ndarray],
        *,
        proposal_config: dict[str, Any],
        prompt: MotionPrompt | None = None,
        prompt_source: str = "independent_motion_proposal",
    ) -> tuple[list[np.ndarray], dict[str, Any], MotionPrompt]:
        self._load()
        assert self._predictor is not None
        assert self._torch is not None
        if prompt is None:
            prompt = build_motion_prompt(
                frames,
                threshold=float(proposal_config["threshold"]),
                min_area=int(proposal_config["min_area"]),
                expand=float(proposal_config["box_expand"]),
                min_box_side=int(proposal_config["min_box_side"]),
            )
        height, width = frames[0].shape[:2]
        masks = [np.zeros((height, width), np.uint8) for _ in frames]
        with tempfile.TemporaryDirectory(prefix="physbench_pendulum_") as temporary:
            frame_directory = Path(temporary)
            for index, frame in enumerate(frames):
                written = cv2.imwrite(
                    str(frame_directory / f"{index:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not written:
                    raise SegmentationError(
                        "temporary_frame_write_failed",
                        f"failed to write temporary SAM2 frame {index}",
                    )
            state = self._predictor.init_state(
                video_path=str(frame_directory),
                offload_video_to_cpu=True,
                offload_state_to_cpu=(self.device == "cpu"),
            )
            self._predictor.reset_state(state)
            amp = (
                self._torch.autocast("cuda", dtype=self._torch.bfloat16)
                if self.device == "cuda"
                else nullcontext()
            )
            try:
                with self._torch.inference_mode(), amp:
                    self._predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=prompt.frame_index,
                        obj_id=1,
                        box=prompt.box_xyxy,
                        points=prompt.points_xy,
                        labels=prompt.point_labels,
                    )
                    forward_count = 0
                    for frame_index, _, logits in (
                        self._predictor.propagate_in_video(state)
                    ):
                        mask = (
                            (logits[0] > 0.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                            .astype(np.uint8)
                            * 255
                        )
                        if mask.shape != (height, width):
                            mask = cv2.resize(
                                mask,
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        masks[int(frame_index)] = mask
                        forward_count += 1
                    reverse_count = 0
                    if prompt.frame_index > 0:
                        for frame_index, _, logits in (
                            self._predictor.propagate_in_video(
                                state,
                                start_frame_idx=prompt.frame_index,
                                reverse=True,
                            )
                        ):
                            mask = (
                                (logits[0] > 0.0)
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                                .astype(np.uint8)
                                * 255
                            )
                            if mask.shape != (height, width):
                                mask = cv2.resize(
                                    mask,
                                    (width, height),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            masks[int(frame_index)] = mask
                            reverse_count += 1
            except SegmentationError:
                raise
            except Exception as exc:
                raise SegmentationError(
                    "sam2_propagation_failed",
                    f"SAM2 mask propagation failed: {exc}",
                ) from exc

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
            "forward_frames": forward_count,
            "reverse_frames": reverse_count,
            **self.describe(),
        }, prompt
