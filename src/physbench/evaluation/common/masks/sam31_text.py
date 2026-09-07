"""Official SAM 3.1 text-only stateful video tracking adapter."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from ..csti.observation import PromptGroupConfig, SemanticCandidateTube
from ..errors import SceneAnalysisError
from .sam31_compat import BIRTH_CONDITIONING_POLICY, install_birth_conditioning_fix


PredictorFactory = Callable[[], Any]
SAM31_SOURCE_REVISION = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
SAM31_CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)


class Sam31TextVideoSegmenter:
    """Detect on frame zero from text, then track official object IDs forward."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        predictor_factory: PredictorFactory | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("SAM3.1 segmenter config must be a mapping")
        self.config = dict(config)
        self.checkpoint_path = str(config.get("checkpoint_path", "")).strip()
        self.checkpoint_path_env = str(
            config.get(
                "checkpoint_path_env", "VPHYSBENCH_SAM31_CHECKPOINT"
            )
        ).strip()
        if not self.checkpoint_path and not self.checkpoint_path_env:
            raise ValueError(
                "SAM3.1 checkpoint_path or checkpoint_path_env is required"
            )
        self.checkpoint_sha256 = str(
            config.get("checkpoint_sha256", SAM31_CHECKPOINT_SHA256)
        ).strip()
        if (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("SAM3.1 checkpoint_sha256 must be lowercase hex")
        self.requested_device = str(config.get("device", "auto"))
        self.precision = str(config.get("precision", "bfloat16"))
        if self.precision != "bfloat16":
            raise ValueError(
                "The official SAM3.1 multiplex predictor currently supports "
                "bfloat16 inference only"
            )
        self.output_probability_threshold = float(
            config.get("output_probability_threshold", 0.5)
        )
        if not 0.0 <= self.output_probability_threshold <= 1.0:
            raise ValueError(
                "SAM3.1 output_probability_threshold must lie in [0,1]"
            )
        self.initial_detection: dict[str, float] | None = None
        if "initial_detection" in config:
            initial_detection = config["initial_detection"]
            if not isinstance(initial_detection, Mapping) or set(
                initial_detection
            ) != {"score_threshold", "new_object_threshold"}:
                raise ValueError(
                    "SAM3.1 initial_detection must contain only "
                    "score_threshold and new_object_threshold"
                )
            score_threshold = self._unit_float(
                initial_detection["score_threshold"],
                "initial_detection.score_threshold",
            )
            new_object_threshold = self._unit_float(
                initial_detection["new_object_threshold"],
                "initial_detection.new_object_threshold",
            )
            if new_object_threshold < score_threshold:
                raise ValueError(
                    "SAM3.1 initial_detection.new_object_threshold must be "
                    "greater than or equal to score_threshold"
                )
            self.initial_detection = {
                "score_threshold": score_threshold,
                "new_object_threshold": new_object_threshold,
            }
        self.masklet_confirmation_enable: bool | None = None
        if "masklet_confirmation_enable" in config:
            masklet_confirmation_enable = config["masklet_confirmation_enable"]
            if not isinstance(masklet_confirmation_enable, bool):
                raise ValueError(
                    "SAM3.1 masklet_confirmation_enable must be boolean"
                )
            self.masklet_confirmation_enable = masklet_confirmation_enable
        self.discovery_pruning_policy = config.get(
            "discovery_pruning_policy", "backend_native_v1"
        )
        if not isinstance(self.discovery_pruning_policy, str) or (
            self.discovery_pruning_policy
            not in {
                "backend_native_v1",
                "fixed_initial_ids_v1",
            }
        ):
            raise ValueError(
                "SAM3.1 discovery_pruning_policy must be "
                "backend_native_v1 or fixed_initial_ids_v1"
            )
        self.segmenter_policy_revision = (
            2
            if self.initial_detection is not None
            or self.masklet_confirmation_enable is not None
            or self.discovery_pruning_policy == "fixed_initial_ids_v1"
            else 1
        )
        self.max_num_objects = int(config.get("max_num_objects", 16))
        self.multiplex_count = int(config.get("multiplex_count", 16))
        if self.max_num_objects < 1 or self.multiplex_count < 1:
            raise ValueError("SAM3.1 object limits must be positive")
        self.compile_model = bool(config.get("compile", False))
        self.warm_up = bool(config.get("warm_up", False))
        self.use_fa3 = bool(config.get("use_fa3", False))
        self.use_rope_real = bool(config.get("use_rope_real", True))
        self.async_loading_frames = bool(
            config.get("async_loading_frames", False)
        )
        self._predictor_factory = predictor_factory
        self._predictor: Any | None = None
        self._lock = threading.RLock()
        self.resolved_device: str | None = None
        self.compatibility_filtered_session_keywords: tuple[str, ...] = ()
        self.model_load_output_summary: dict[str, Any] = {}
        self.birth_conditioning_policy = "not_loaded"
        self._native_detection: dict[str, float] | None = None
        self._native_masklet_confirmation: bool | None = None
        self._native_discovery_pruning: dict[str, Any] | None = None

    @staticmethod
    def _unit_float(value: Any, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"SAM3.1 {label} must be a finite number in [0,1]")
        return float(value)

    @staticmethod
    def _backend_gate(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def _record_native_detection_policy(self, model: Any) -> None:
        if all(
            hasattr(model, name)
            for name in ("score_threshold_detection", "new_det_thresh")
        ):
            score_threshold = self._backend_gate(
                model.score_threshold_detection
            )
            new_object_threshold = self._backend_gate(model.new_det_thresh)
            if score_threshold is not None and new_object_threshold is not None:
                self._native_detection = {
                    "score_threshold": score_threshold,
                    "new_object_threshold": new_object_threshold,
                }

    def _record_native_backend_policy(self) -> Any:
        assert self._predictor is not None
        model = getattr(self._predictor, "model", None)
        if model is None:
            return None
        self._record_native_detection_policy(model)
        if hasattr(model, "masklet_confirmation_enable") and isinstance(
            model.masklet_confirmation_enable, bool
        ):
            self._native_masklet_confirmation = (
                model.masklet_confirmation_enable
            )
        return model

    def _required_model_attribute(self, model: Any, attribute: str) -> Any:
        if model is None or not hasattr(model, attribute):
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                "SAM3.1 predictor lacks required backend policy attribute "
                f"{attribute!r}",
            )
        return getattr(model, attribute)

    def _record_native_discovery_pruning(self, model: Any) -> None:
        if not all(
            hasattr(model, name)
            for name in (
                "hotstart_delay",
                "suppress_unmatched_only_within_hotstart",
            )
        ):
            return
        hotstart_delay = model.hotstart_delay
        suppress_unmatched = model.suppress_unmatched_only_within_hotstart
        if (
            isinstance(hotstart_delay, bool)
            or not isinstance(hotstart_delay, int)
            or hotstart_delay < 0
            or not isinstance(suppress_unmatched, bool)
        ):
            return
        self._native_discovery_pruning = {
            "hotstart_delay": hotstart_delay,
            "suppress_unmatched_only_within_hotstart": suppress_unmatched,
        }

    @contextlib.contextmanager
    def _discovery_pruning_policy(self):
        assert self._predictor is not None
        model = getattr(self._predictor, "model", None)
        self._record_native_discovery_pruning(model)
        if self.discovery_pruning_policy == "backend_native_v1":
            yield
            return
        previous_hotstart = self._required_model_attribute(
            model, "hotstart_delay"
        )
        previous_suppress_unmatched = self._required_model_attribute(
            model, "suppress_unmatched_only_within_hotstart"
        )
        if (
            isinstance(previous_hotstart, bool)
            or not isinstance(previous_hotstart, int)
            or previous_hotstart < 0
            or not isinstance(previous_suppress_unmatched, bool)
        ):
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                "SAM3.1 discovery-pruning backend attributes have "
                "incompatible types",
            )

        def restore() -> None:
            model.suppress_unmatched_only_within_hotstart = (
                previous_suppress_unmatched
            )
            model.hotstart_delay = previous_hotstart

        try:
            model.hotstart_delay = 0
            model.suppress_unmatched_only_within_hotstart = True
        except Exception as exc:
            restore()
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                f"Cannot apply SAM3.1 discovery pruning policy: {exc}",
            ) from exc
        try:
            yield
        finally:
            restore()

    @contextlib.contextmanager
    def _initial_detection_policy(self):
        if self.initial_detection is None:
            yield
            return
        assert self._predictor is not None
        model = getattr(self._predictor, "model", None)
        self._record_native_detection_policy(model)
        previous_score = self._required_model_attribute(
            model, "score_threshold_detection"
        )
        previous_new_object = self._required_model_attribute(
            model, "new_det_thresh"
        )
        try:
            model.score_threshold_detection = self.initial_detection[
                "score_threshold"
            ]
            model.new_det_thresh = self.initial_detection[
                "new_object_threshold"
            ]
        except Exception as exc:
            model.score_threshold_detection = previous_score
            model.new_det_thresh = previous_new_object
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                f"Cannot apply SAM3.1 initial detection policy: {exc}",
            ) from exc
        try:
            yield
        finally:
            model.score_threshold_detection = previous_score
            model.new_det_thresh = previous_new_object

    @contextlib.contextmanager
    def _masklet_confirmation_policy(self):
        model = self._record_native_backend_policy()
        if self.masklet_confirmation_enable is None:
            yield
            return
        previous = self._required_model_attribute(
            model, "masklet_confirmation_enable"
        )
        if not isinstance(previous, bool):
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                "SAM3.1 masklet_confirmation_enable backend attribute must "
                "be boolean",
            )
        try:
            model.masklet_confirmation_enable = (
                self.masklet_confirmation_enable
            )
        except Exception as exc:
            model.masklet_confirmation_enable = previous
            raise SceneAnalysisError(
                "sam31_model_interface_incompatible",
                f"Cannot apply SAM3.1 masklet confirmation policy: {exc}",
            ) from exc
        try:
            yield
        finally:
            model.masklet_confirmation_enable = previous

    def _install_birth_conditioning_compatibility(self) -> None:
        try:
            installed = install_birth_conditioning_fix(self._predictor)
        except RuntimeError as exc:
            self._predictor = None
            raise SceneAnalysisError(
                "sam31_tracker_interface_incompatible", str(exc)
            ) from exc
        if installed:
            self.birth_conditioning_policy = BIRTH_CONDITIONING_POLICY
        elif self._predictor_factory is not None:
            self.birth_conditioning_policy = "not_installed_fixture"
        else:
            self._predictor = None
            raise SceneAnalysisError(
                "sam31_tracker_interface_incompatible",
                "SAM3.1 predictor lacks the pinned birth-conditioning tracker interface",
            )

    def _install_session_compatibility(self) -> None:
        assert self._predictor is not None
        model = getattr(self._predictor, "model", None)
        init_state = getattr(model, "init_state", None)
        if init_state is None:
            return
        signature = inspect.signature(init_state)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return
        accepted = frozenset(signature.parameters)
        possible = {
            "resource_path",
            "offload_video_to_cpu",
            "offload_state_to_cpu",
            "async_loading_frames",
            "video_loader_type",
        }
        filtered = tuple(sorted(possible - accepted))
        if not filtered:
            return

        def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
            return init_state(
                *args,
                **{key: value for key, value in kwargs.items() if key in accepted},
            )

        model.init_state = compatible_init_state
        self.compatibility_filtered_session_keywords = filtered

    def _load(self) -> None:
        if self._predictor is not None:
            return
        if self._predictor_factory is not None:
            self._predictor = self._predictor_factory()
            self.resolved_device = "fixture"
            self._install_session_compatibility()
            self._install_birth_conditioning_compatibility()
            return
        checkpoint_value = self.checkpoint_path
        if not checkpoint_value:
            checkpoint_value = os.environ.get(self.checkpoint_path_env, "").strip()
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            raise SceneAnalysisError(
                "sam31_checkpoint_missing",
                "SAM3.1 checkpoint is missing; configure checkpoint_path or "
                f"{self.checkpoint_path_env!r}",
            )
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.checkpoint_sha256:
            raise SceneAnalysisError(
                "sam31_checkpoint_digest_mismatch",
                "SAM3.1 checkpoint SHA-256 differs from the protocol",
            )
        try:
            import torch
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:
            raise SceneAnalysisError(
                "sam31_dependency_missing",
                "SAM3.1 evaluator dependencies are not installed",
            ) from exc
        if not torch.cuda.is_available():
            raise SceneAnalysisError(
                "sam31_cuda_required",
                "The official SAM3.1 multiplex predictor requires CUDA",
            )
        device = self.requested_device
        if device == "auto":
            device = "cuda"
        if not device.startswith("cuda"):
            raise SceneAnalysisError(
                "sam31_cuda_required",
                "The official SAM3.1 multiplex predictor requires a CUDA device",
            )
        if device != "cuda":
            torch.cuda.set_device(torch.device(device))
        try:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self._predictor = build_sam3_multiplex_video_predictor(
                    checkpoint_path=str(checkpoint),
                    max_num_objects=self.max_num_objects,
                    multiplex_count=self.multiplex_count,
                    use_fa3=self.use_fa3,
                    use_rope_real=self.use_rope_real,
                    compile=self.compile_model,
                    warm_up=self.warm_up,
                    default_output_prob_thresh=(
                        self.output_probability_threshold
                    ),
                    async_loading_frames=self.async_loading_frames,
                )
            model_output = captured.getvalue()
            self.model_load_output_summary = {
                "line_count": len(model_output.splitlines()),
                "sha256": hashlib.sha256(model_output.encode()).hexdigest(),
                "reported_missing_keys": "Missing keys" in model_output,
            }
        except Exception as exc:
            raise SceneAnalysisError(
                "sam31_model_load_failed",
                f"Failed to load the SAM3.1 predictor: {exc}",
            ) from exc
        self.checkpoint_path = str(checkpoint)
        self.resolved_device = device
        self._install_session_compatibility()
        self._install_birth_conditioning_compatibility()

    @staticmethod
    def _validate_frames(
        frames: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        values = tuple(frames)
        if not values:
            raise ValueError("SAM3.1 video requires at least one frame")
        shape = values[0].shape
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("SAM3.1 frames must use HWC three-channel layout")
        for frame in values:
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.shape != shape
            ):
                raise ValueError(
                    "SAM3.1 frames must be equal-sized uint8 HWC arrays"
                )
        return values

    @staticmethod
    def _write_frames(frames: tuple[np.ndarray, ...], directory: Path) -> None:
        for index, frame in enumerate(frames):
            path = directory / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SceneAnalysisError(
                    "sam31_temporary_frame_write_failed",
                    f"Failed to write aligned SAM3.1 frame {index}",
                )

    @staticmethod
    def _response_arrays(
        response: Mapping[str, Any],
        *,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = response.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise SceneAnalysisError(
                "sam31_output_invalid", "SAM3.1 outputs must be a mapping"
            )
        object_ids = np.asarray(
            outputs.get("out_obj_ids", np.zeros(0, dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        masks = np.asarray(
            outputs.get(
                "out_binary_masks",
                np.zeros((0, *frame_shape), dtype=bool),
            )
        )
        if masks.ndim == 2 and len(object_ids) == 1:
            masks = masks[None]
        boxes = np.asarray(
            outputs.get(
                "out_boxes_xywh",
                np.zeros((len(object_ids), 4), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        probabilities = np.asarray(
            outputs.get(
                "out_probs", np.ones(len(object_ids), dtype=np.float32)
            ),
            dtype=np.float32,
        ).reshape(-1)
        if masks.shape != (len(object_ids), *frame_shape):
            raise SceneAnalysisError(
                "sam31_mask_shape_invalid",
                "SAM3.1 object IDs and masks have incompatible shapes",
            )
        if boxes.shape != (len(object_ids), 4) or not np.isfinite(boxes).all():
            raise SceneAnalysisError(
                "sam31_box_shape_invalid",
                "SAM3.1 object IDs and boxes have incompatible shapes",
            )
        if (
            probabilities.shape != (len(object_ids),)
            or not np.isfinite(probabilities).all()
        ):
            raise SceneAnalysisError(
                "sam31_probability_shape_invalid",
                "SAM3.1 object IDs and probabilities have incompatible shapes",
            )
        return object_ids, masks.astype(bool), boxes, np.clip(probabilities, 0, 1)

    def _run_group(
        self,
        *,
        frame_directory: Path,
        frame_count: int,
        frame_shape: tuple[int, int],
        group: PromptGroupConfig,
    ) -> tuple[SemanticCandidateTube, ...]:
        assert self._predictor is not None
        object_ids: tuple[int, ...] = ()
        masks_by_id: dict[int, np.ndarray] = {}
        boxes_by_id: dict[int, np.ndarray] = {}
        probabilities_by_id: dict[int, np.ndarray] = {}

        def consume(response: Mapping[str, Any], *, initialize: bool = False) -> None:
            nonlocal object_ids
            frame_index = int(response["frame_index"])
            if not 0 <= frame_index < frame_count:
                raise SceneAnalysisError(
                    "sam31_frame_index_invalid",
                    f"SAM3.1 returned frame {frame_index} outside the video",
                )
            ids, masks, boxes, probabilities = self._response_arrays(
                response, frame_shape=frame_shape
            )
            if initialize:
                object_ids = tuple(sorted(int(value) for value in ids.tolist()))
                for object_id in object_ids:
                    masks_by_id[object_id] = np.zeros(
                        (frame_count, *frame_shape), dtype=bool
                    )
                    boxes_by_id[object_id] = np.zeros(
                        (frame_count, 4), dtype=np.float32
                    )
                    probabilities_by_id[object_id] = np.zeros(
                        frame_count, dtype=np.float32
                    )
            indices = {int(value): index for index, value in enumerate(ids)}
            for object_id in object_ids:
                output_index = indices.get(object_id)
                if output_index is None:
                    continue
                masks_by_id[object_id][frame_index] = masks[output_index]
                boxes_by_id[object_id][frame_index] = boxes[output_index]
                probabilities_by_id[object_id][frame_index] = probabilities[
                    output_index
                ]

        with (
            self._discovery_pruning_policy(),
            self._masklet_confirmation_policy(),
        ):
            start_response = self._predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_directory),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = str(start_response["session_id"])
            try:
                with self._initial_detection_policy():
                    prompt_response = self._predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": group.text,
                            "output_prob_thresh": (
                                self.output_probability_threshold
                            ),
                        }
                    )
                consume(prompt_response, initialize=True)
                if not object_ids:
                    return ()
                for response in self._predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "output_prob_thresh": (
                            self.output_probability_threshold
                        ),
                    }
                ):
                    consume(response)
            finally:
                self._predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
        return tuple(
            SemanticCandidateTube(
                candidate_id=f"{group.group_id}:{object_id}",
                prompt_group_id=group.group_id,
                backend_object_id=object_id,
                masks=masks_by_id[object_id],
                boxes_xywh=boxes_by_id[object_id],
                confidences=probabilities_by_id[object_id],
            )
            for object_id in object_ids
        )

    def segment(
        self,
        frames: Sequence[np.ndarray],
        prompt_groups: Sequence[PromptGroupConfig],
    ) -> tuple[SemanticCandidateTube, ...]:
        normalized_frames = self._validate_frames(frames)
        groups = tuple(prompt_groups)
        if not groups or len({group.group_id for group in groups}) != len(groups):
            raise ValueError("SAM3.1 prompt groups must be non-empty and unique")
        with self._lock:
            self._load()
            with tempfile.TemporaryDirectory(prefix="vphysbench-sam31-") as value:
                directory = Path(value)
                self._write_frames(normalized_frames, directory)
                candidates = tuple(
                    candidate
                    for group in groups
                    for candidate in self._run_group(
                        frame_directory=directory,
                        frame_count=len(normalized_frames),
                        frame_shape=normalized_frames[0].shape[:2],
                        group=group,
                    )
                )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    def describe(self) -> dict[str, Any]:
        native_detection = self._native_detection
        initial_detection = self.initial_detection or native_detection
        initial_detection_description = {
            "policy": (
                "frame_zero_override_v1"
                if self.initial_detection is not None
                else "backend_native_v1"
            ),
            "score_threshold": (
                initial_detection["score_threshold"]
                if initial_detection is not None
                else None
            ),
            "new_object_threshold": (
                initial_detection["new_object_threshold"]
                if initial_detection is not None
                else None
            ),
        }
        return {
            "backend": "sam3.1_multiplex_text_video",
            "segmenter_policy_revision": self.segmenter_policy_revision,
            "source_revision": SAM31_SOURCE_REVISION,
            "birth_conditioning_policy": self.birth_conditioning_policy,
            "checkpoint_path_env": self.checkpoint_path_env,
            "checkpoint_sha256": self.checkpoint_sha256,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "precision": self.precision,
            "output_probability_threshold": self.output_probability_threshold,
            "output_probability_threshold_policy": (
                "backend_request_not_guaranteed_export_filter_v1"
            ),
            "initial_detection": initial_detection_description,
            "propagation_detection": {
                "policy": "backend_native_v1",
                "score_threshold": (
                    native_detection["score_threshold"]
                    if native_detection is not None
                    else None
                ),
                "new_object_threshold": (
                    native_detection["new_object_threshold"]
                    if native_detection is not None
                    else None
                ),
            },
            "masklet_confirmation": {
                "policy": (
                    "whole_session_override_v1"
                    if self.masklet_confirmation_enable is not None
                    else "backend_native_v1"
                ),
                "requested": self.masklet_confirmation_enable,
                "native": self._native_masklet_confirmation,
                "effective": (
                    self.masklet_confirmation_enable
                    if self.masklet_confirmation_enable is not None
                    else self._native_masklet_confirmation
                ),
            },
            "discovery_pruning": {
                "policy": self.discovery_pruning_policy,
                "requested": (
                    {
                        "hotstart_delay": 0,
                        "suppress_unmatched_only_within_hotstart": True,
                    }
                    if self.discovery_pruning_policy == "fixed_initial_ids_v1"
                    else None
                ),
                "native": (
                    dict(self._native_discovery_pruning)
                    if self._native_discovery_pruning is not None
                    else None
                ),
                "effective": (
                    {
                        "hotstart_delay": 0,
                        "suppress_unmatched_only_within_hotstart": True,
                    }
                    if self.discovery_pruning_policy == "fixed_initial_ids_v1"
                    else (
                        dict(self._native_discovery_pruning)
                        if self._native_discovery_pruning is not None
                        else None
                    )
                ),
            },
            "prompt_policy": "text_only_frame_zero",
            "propagation_direction": "forward",
            "identity_policy": "external_first_frame_hungarian_locked",
            "compatibility_filtered_session_keywords": list(
                self.compatibility_filtered_session_keywords
            ),
            "model_load_output_summary": dict(self.model_load_output_summary),
        }


_SHARED_LOCK = threading.Lock()
_SHARED_SEGMENTERS: dict[tuple[str, int | None], Sam31TextVideoSegmenter] = {}


def get_shared_sam31_text_segmenter(
    config: Mapping[str, Any],
    *,
    predictor_factory: PredictorFactory | None = None,
) -> Sam31TextVideoSegmenter:
    """Reuse the heavy predictor for one immutable process-local config."""

    serialized = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    key = (serialized, id(predictor_factory) if predictor_factory is not None else None)
    with _SHARED_LOCK:
        segmenter = _SHARED_SEGMENTERS.get(key)
        if segmenter is None:
            segmenter = Sam31TextVideoSegmenter(
                config, predictor_factory=predictor_factory
            )
            _SHARED_SEGMENTERS[key] = segmenter
        return segmenter


__all__ = [
    "SAM31_CHECKPOINT_SHA256",
    "SAM31_SOURCE_REVISION",
    "Sam31TextVideoSegmenter",
    "get_shared_sam31_text_segmenter",
]
