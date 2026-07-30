from __future__ import annotations

import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..errors import SceneAnalysisError


@dataclass(frozen=True)
class MaskPrompt:
    frame_index: int
    box_xyxy: np.ndarray
    points_xy: np.ndarray
    point_labels: np.ndarray
    metadata: dict[str, Any]


class Sam2VideoSegmenter:
    """Lazy SAM2.1 adapter shared by all scene-specific prompt builders."""

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
            raise SceneAnalysisError(
                "sam2_dependency_missing",
                "SAM2 evaluation dependencies are not installed",
            ) from exc
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SceneAnalysisError(
                "cuda_unavailable", "SAM2 evaluator requested CUDA but none is available"
            )
        try:
            predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id, device=device
            )
        except Exception as exc:
            raise SceneAnalysisError(
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
            "reuse_policy": "one_model_instance_per_scene_evaluator",
        }

    @staticmethod
    def _logit_mask(logit: Any, *, width: int, height: int) -> np.ndarray:
        mask = (
            (logit > 0.0)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
            .astype(np.uint8)
            * 255
        )
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        return mask

    def segment(
        self,
        frames: list[np.ndarray],
        *,
        prompt: MaskPrompt,
        temporary_prefix: str = "physbench_sam2_",
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        if not frames:
            raise SceneAnalysisError("empty_timeline", "cannot segment an empty video")
        self._load()
        assert self._predictor is not None
        assert self._torch is not None
        height, width = frames[0].shape[:2]
        masks = [np.zeros((height, width), np.uint8) for _ in frames]
        with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
            frame_directory = Path(temporary)
            for index, frame in enumerate(frames):
                written = cv2.imwrite(
                    str(frame_directory / f"{index:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not written:
                    raise SceneAnalysisError(
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
                    for frame_index, _, logits in self._predictor.propagate_in_video(
                        state
                    ):
                        masks[int(frame_index)] = self._logit_mask(
                            logits[0], width=width, height=height
                        )
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
                            masks[int(frame_index)] = self._logit_mask(
                                logits[0], width=width, height=height
                            )
                            reverse_count += 1
            except SceneAnalysisError:
                raise
            except Exception as exc:
                raise SceneAnalysisError(
                    "sam2_propagation_failed",
                    f"SAM2 mask propagation failed: {exc}",
                ) from exc
        return masks, {
            "prompt_frame": prompt.frame_index,
            "prompt_box_xyxy": prompt.box_xyxy.tolist(),
            "prompt_points_xy": prompt.points_xy.tolist(),
            "prompt_point_labels": prompt.point_labels.tolist(),
            "prompt_metadata": prompt.metadata,
            "forward_frames": forward_count,
            "reverse_frames": reverse_count,
            **self.describe(),
        }

    def segment_instances(
        self,
        frames: list[np.ndarray],
        *,
        prompts: list[MaskPrompt],
        temporary_prefix: str = "physbench_sam2_instances_",
        exclusive_masks: bool = False,
    ) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
        """Propagate several subjects from one shared seed frame.

        A non-zero seed is propagated both forward and backward.  New
        evaluators may request joint logit competition with
        ``exclusive_masks``; the default retains the historical independent
        thresholding used by frozen v1-v3 protocols.
        """
        if not frames or not prompts:
            raise SceneAnalysisError(
                "empty_instance_prompts",
                "multi-instance segmentation requires frames and prompts",
            )
        seed_frames = {int(prompt.frame_index) for prompt in prompts}
        if len(seed_frames) != 1:
            raise SceneAnalysisError(
                "unsupported_instance_prompt_timeline",
                "multi-instance prompts must share one seed frame",
            )
        seed_frame = next(iter(seed_frames))
        if seed_frame < 0 or seed_frame >= len(frames):
            raise SceneAnalysisError(
                "invalid_instance_prompt_frame",
                f"multi-instance seed frame {seed_frame} is outside the video",
            )
        self._load()
        assert self._predictor is not None
        assert self._torch is not None
        height, width = frames[0].shape[:2]
        instance_masks = [
            [np.zeros((height, width), np.uint8) for _ in frames]
            for _ in prompts
        ]
        with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
            frame_directory = Path(temporary)
            for index, frame in enumerate(frames):
                written = cv2.imwrite(
                    str(frame_directory / f"{index:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not written:
                    raise SceneAnalysisError(
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
                    for object_index, prompt in enumerate(prompts, start=1):
                        self._predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=seed_frame,
                            obj_id=object_index,
                            box=prompt.box_xyxy,
                            points=prompt.points_xy,
                            labels=prompt.point_labels,
                        )

                    def consume(
                        frame_index: int,
                        object_ids: Any,
                        logits: Any,
                    ) -> None:
                        ids = [int(value) for value in object_ids]
                        raw = logits.detach().float().cpu().numpy()
                        if raw.ndim == 4:
                            raw = raw[:, 0]
                        elif raw.ndim != 3:
                            raise SceneAnalysisError(
                                "sam2_invalid_logits",
                                f"unexpected SAM2 logits shape {raw.shape}",
                            )
                        if raw.shape[1:] != (height, width):
                            raw = np.stack(
                                [
                                    cv2.resize(
                                        value,
                                        (width, height),
                                        interpolation=cv2.INTER_LINEAR,
                                    )
                                    for value in raw
                                ]
                            )
                        if exclusive_masks:
                            # The old independent thresholding allowed the
                            # same touching pixels to be counted by several
                            # roles. V4 opts into SAM2 instance competition.
                            winner = np.argmax(raw, axis=0)
                            positive = np.max(raw, axis=0) > 0.0
                            binary_masks = [
                                (winner == logit_index) & positive
                                for logit_index in range(len(ids))
                            ]
                        else:
                            # Preserve the v1-v3 evaluator behavior exactly.
                            binary_masks = [
                                raw[logit_index] > 0.0
                                for logit_index in range(len(ids))
                            ]
                        for binary, object_id in zip(binary_masks, ids):
                            object_number = object_id - 1
                            if 0 <= object_number < len(instance_masks):
                                instance_masks[object_number][
                                    int(frame_index)
                                ] = binary.astype(np.uint8) * 255

                    forward_count = 0
                    forward = (
                        self._predictor.propagate_in_video(state)
                        if seed_frame == 0
                        else self._predictor.propagate_in_video(
                            state, start_frame_idx=seed_frame
                        )
                    )
                    for frame_index, object_ids, logits in forward:
                        consume(int(frame_index), object_ids, logits)
                        forward_count += 1
                    reverse_count = 0
                    if seed_frame > 0:
                        for frame_index, object_ids, logits in (
                            self._predictor.propagate_in_video(
                                state,
                                start_frame_idx=seed_frame,
                                reverse=True,
                            )
                        ):
                            consume(int(frame_index), object_ids, logits)
                            reverse_count += 1
            except SceneAnalysisError:
                raise
            except Exception as exc:
                raise SceneAnalysisError(
                    "sam2_propagation_failed",
                    f"SAM2 multi-instance propagation failed: {exc}",
                ) from exc
        metadata = {
            "prompt_frames": [seed_frame for _ in prompts],
            "prompt_boxes_xyxy": [
                prompt.box_xyxy.tolist() for prompt in prompts
            ],
            "prompt_points_xy": [
                prompt.points_xy.tolist() for prompt in prompts
            ],
            "prompt_metadata": [prompt.metadata for prompt in prompts],
            "objects": len(prompts),
            "forward_frames": forward_count,
            **self.describe(),
        }
        if seed_frame > 0 or exclusive_masks:
            metadata.update(
                {
                    "seed_frame": seed_frame,
                    "reverse_frames": reverse_count,
                    "overlap_policy": (
                        "highest_positive_logit_wins"
                        if exclusive_masks
                        else "independent_positive_logits"
                    ),
                }
            )
        return instance_masks, metadata
