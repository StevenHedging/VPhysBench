"""Fail-closed parabolic evaluator bound to a frozen projectile identity."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from ...common.frozen_subject import load_frozen_subject_anchor
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
    evaluator_version = "2.0"
    sequential_evaluator_version = "2.0"
    robust_evaluator_version = "2.0"
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
            "reference_discovery": "frozen_dataset_projectile_anchor_v2",
            "prediction_discovery": (
                "frozen_anchor_binding_then_independent_unique_candidates_v2"
            ),
            "segmentation": "frozen_seed_then_local_compact_ball_candidates",
            "tracking": "unique_assignment_margin_with_latched_identity_loss",
            "lifecycle": "reference_frozen_may_exit",
            "distance": "scene_specific_empirical_projectile_trajectory_v1",
            "physics_composition": "strict_multiplicative_v2",
            "future_gt_usage": False,
        }

    def _observe_projectile(
        self,
        request: CaseEvaluationRequest,
        *,
        entity_id: str,
        frames: list[np.ndarray],
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
                "observation_role": (
                    "reference" if reference else "prediction"
                ),
            },
        )


__all__ = ["ParabolicMotionCaseEvaluatorV2"]
