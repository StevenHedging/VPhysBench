"""Fail-closed parabolic evaluator bound to a frozen projectile identity."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from ...common.frozen_reference import load_frozen_reference_observation
from ...common.frozen_subject import load_frozen_subject_anchor
from ...common.errors import ReferenceAnalysisError
from ...contracts import CaseEvaluationRequest
from .evaluator import (
    BallCandidate,
    ParabolicMotionCaseEvaluator,
    ParabolicObservation,
    _mask_histogram,
    observe_projectile,
)


class ParabolicMotionCaseEvaluatorV2(ParabolicMotionCaseEvaluator):
    evaluator_id = "parabolic_motion_fail_closed"
    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"
    prediction_missing_code = "prediction_projectile_entity_missing_v2"
    prediction_unmatched_code = "prediction_projectile_entity_unmatched_v2"
    prediction_uncertain_code = "prediction_projectile_identity_uncertain_v2"
    identity_policy = (
        "frozen_frame_zero_anchor_then_unique_sequential_assignment_or_"
        "latched_null_v2"
    )

    def describe_observation(self) -> dict[str, Any]:
        return {
            "protocol": "open_world_v2",
            "reference_discovery": (
                "frozen_dataset_reference_observation_v1"
            ),
            "prediction_discovery": (
                "frozen_anchor_binding_then_independent_unique_candidates_v2"
            ),
            "segmentation": (
                "frozen_reference_tube_then_prediction_local_candidates"
            ),
            "tracking": "unique_assignment_margin_with_latched_identity_loss",
            "lifecycle": "reference_frozen_may_exit",
            "distance": "scene_specific_empirical_projectile_trajectory_v1",
            "physics_composition": "strict_multiplicative_v2",
            "future_gt_usage": False,
        }

    def _legacy_reference_observation(
        self,
        request: CaseEvaluationRequest,
        *,
        entity_id: str,
        frames: list[np.ndarray],
        available: np.ndarray,
        spatial_transform: Mapping[str, object],
    ) -> ParabolicObservation:
        anchor = load_frozen_subject_anchor(
            request,
            logical_entity_id=entity_id,
            entity_class="ball",
            spatial_transform=spatial_transform,
            error_namespace="reference_projectile_subject",
        )
        frame_zero = np.asarray(frames[0])
        if frame_zero.shape != (*anchor.mask.shape, 3):
            raise ValueError(
                "frozen projectile mask and video frame use different canvases"
            )
        gray = cv2.cvtColor(frame_zero, cv2.COLOR_BGR2GRAY)
        foreground = gray[anchor.mask > 0]
        background_ring = cv2.dilate(
            anchor.mask,
            np.ones((7, 7), dtype=np.uint8),
        )
        background_ring = (background_ring > 0) & (anchor.mask == 0)
        contrast = (
            0.0
            if foreground.size == 0 or not np.any(background_ring)
            else abs(
                float(np.mean(foreground))
                - float(np.median(gray[background_ring]))
            )
        )
        seed = BallCandidate(
            xy=np.asarray(anchor.centroid_xy, dtype=np.float64),
            radius=float(anchor.equivalent_radius_px),
            mask=np.asarray(anchor.mask),
            histogram=_mask_histogram(frame_zero, anchor.mask),
            contrast=contrast,
            source="frozen_subject_anchor_v2",
        )
        return observe_projectile(
            frames,
            available=available,
            config=dict(self.config.get("open_world_observation", {})),
            seed_override=seed,
            seed_provenance={
                **dict(anchor.provenance),
                "observation_role": "reference",
            },
        )

    def _observe_projectile(
        self,
        request: CaseEvaluationRequest,
        *,
        entity_id: str,
        frames: list[np.ndarray],
        times_s: list[float] | None = None,
        available: np.ndarray,
        spatial_transform: Mapping[str, object],
        reference: bool,
    ) -> ParabolicObservation:
        if not frames:
            raise ValueError("projectile observation requires frames")
        if not reference:
            return observe_projectile(
                frames,
                available=available,
                config=dict(self.config.get("open_world_observation", {})),
            )
        if self.config.get("reference_observation_policy") != (
            "frozen_dataset_reference_observation_v1"
        ):
            return self._legacy_reference_observation(
                request,
                entity_id=entity_id,
                frames=frames,
                available=available,
                spatial_transform=spatial_transform,
            )
        frozen = load_frozen_reference_observation(
            request,
            times_s=(
                times_s
                if times_s is not None
                else [float(index) for index in range(len(frames))]
            ),
            spatial_transform=spatial_transform,
            expected_entity_ids=[entity_id],
        )
        value = frozen.entities[entity_id]
        if len(value.masks) != len(frames):
            raise ValueError("frozen projectile timeline differs from video")
        radii = np.sqrt(
            np.asarray(value.area_pixels, dtype=np.float64) / np.pi
        )
        radii[~value.visible] = np.nan
        histograms = tuple(
            _mask_histogram(frame, mask) if visible else None
            for frame, mask, visible in zip(
                frames,
                value.masks,
                value.visible,
                strict=True,
            )
        )
        if not bool(value.visible[0]):
            raise ValueError("frozen projectile is not visible at frame zero")
        frame_zero = np.asarray(frames[0])
        mask_zero = np.asarray(value.masks[0], dtype=np.uint8)
        gray = cv2.cvtColor(frame_zero, cv2.COLOR_BGR2GRAY)
        foreground = gray[mask_zero > 0]
        background_ring = cv2.dilate(
            mask_zero,
            np.ones((7, 7), dtype=np.uint8),
        )
        background_ring = (background_ring > 0) & (mask_zero == 0)
        contrast = 0.0 if foreground.size == 0 or not np.any(
            background_ring
        ) else abs(
            float(np.mean(foreground))
            - float(np.median(gray[background_ring]))
        )
        seed = BallCandidate(
            xy=np.asarray(value.centroid_xy[0], dtype=np.float64),
            radius=float(radii[0]),
            mask=mask_zero,
            histogram=np.asarray(histograms[0]),
            contrast=contrast,
            source="frozen_dataset_reference_observation_v1",
        )
        return ParabolicObservation(
            xy=np.asarray(value.centroid_xy, dtype=np.float64),
            radii=radii,
            observed=np.asarray(value.visible, dtype=bool),
            interpolated=np.zeros(len(frames), dtype=bool),
            masks=tuple(np.asarray(mask) for mask in value.masks),
            histograms=histograms,
            cardinality=np.asarray(value.visible, dtype=np.int64),
            seed=seed,
            diagnostics=dict(frozen.provenance),
        )


__all__ = ["ParabolicMotionCaseEvaluatorV2"]
