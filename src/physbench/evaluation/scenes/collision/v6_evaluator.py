from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ....io import canonical_sha256
from ...common.entities.observer import EvidenceTier, OpenWorldObservation
from ...common.errors import SceneAnalysisError
from ...common.masks.sam2 import MaskPrompt
from ...contracts import CaseEvaluationRequest
from .v5_evaluator import CollisionOpenWorldCaseEvaluator
from .v6_identity import (
    CollisionIdentityAnchor,
    build_motion_validated_collision_reference_anchors,
    build_prediction_collision_identity_prompts,
    latch_collision_identity_masks,
    reference_prompts_from_collision_anchors,
)


@dataclass(frozen=True)
class CollisionIdentityContext:
    anchors: tuple[CollisionIdentityAnchor, ...]
    reference_localization: dict[str, Any]


class CollisionFailClosedCaseEvaluator(CollisionOpenWorldCaseEvaluator):
    """Collision evaluator with causal, terminal subject identity."""

    evaluator_id = "collision_1d_fail_closed_nbody"
    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"
    fail_closed_on_directed_identity = True
    identity_policy = (
        "motion_validated_reference_frame_zero_plus_unique_prediction_"
        "frame_zero_binding_terminal_latch_v1"
    )

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.fingerprint = canonical_sha256(
            {
                "id": self.evaluator_id,
                "version": self.evaluator_version,
                "config": config,
                "entity_contract": "manifest_v1",
                "observer": "open_world_fail_closed_identity_v1",
                "physics": "nbody_v1.2",
            }
        )

    def describe_observation(self) -> dict[str, Any]:
        value = super().describe_observation()
        value.update(
            {
                "expected_channel": (
                    "motion_validated_reference_frame_zero_and_unique_"
                    "prediction_frame_zero_binding_plus_forward_sam2"
                ),
                "identity": self.identity_policy,
                "identity_failure_policy": (
                    "terminal_zero_residual_diagnostics_only"
                ),
            }
        )
        return value

    def _prepare_identity_context(
        self,
        request: CaseEvaluationRequest,
        *,
        reference_frames: Sequence[np.ndarray],
        entity_ids: Sequence[str],
    ) -> CollisionIdentityContext:
        identity = self.config["subject_identity"]
        if (
            identity.get("anchor_policy")
            != "reference_motion_validated_frame_zero_v1"
            or identity.get("failure_policy") != "fail_closed_v1"
        ):
            raise SceneAnalysisError(
                "reference_collision_identity_policy_invalid",
                "collision v6 requires the frozen motion-validated, "
                "fail-closed identity policy",
            )
        anchors, metadata = (
            build_motion_validated_collision_reference_anchors(
                entity_ids=entity_ids,
                reference_frames=reference_frames,
                observation_config=self.config["multi_frame_observation"],
                localization_config=identity["reference_localization"],
            )
        )
        return CollisionIdentityContext(
            anchors=anchors,
            reference_localization=metadata,
        )

    def _observe_expected_role(
        self,
        frames: list[np.ndarray],
        *,
        expected_count: int,
        entity_ids: Sequence[str],
        observation_role: str,
        identity_context: object | None,
        available: Sequence[bool] | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[np.ndarray]],
        list[np.ndarray],
        dict[str, Any],
    ]:
        if not isinstance(identity_context, CollisionIdentityContext):
            raise SceneAnalysisError(
                "collision_identity_context_missing",
                "collision v6 has no confirmed reference identity context",
            )
        anchors = identity_context.anchors
        if (
            len(anchors) != expected_count
            or tuple(value.logical_entity_id for value in anchors)
            != tuple(entity_ids)
        ):
            raise SceneAnalysisError(
                "collision_identity_context_mismatch",
                "collision v6 identity context differs from the entity manifest",
            )
        identity = self.config["subject_identity"]
        if observation_role == "reference":
            prompts, prompt_metadata = reference_prompts_from_collision_anchors(
                anchors,
                config=identity["prediction_binding"],
            )
            prompt_metadata = {
                **prompt_metadata,
                "reference_localization": (
                    identity_context.reference_localization
                ),
            }
        elif observation_role == "prediction":
            if not frames:
                raise SceneAnalysisError(
                    "collision_prediction_frame_zero_unavailable",
                    "prediction has no frame zero for identity binding",
                )
            if available is not None and (
                len(available) != len(frames) or not bool(available[0])
            ):
                raise SceneAnalysisError(
                    "collision_prediction_frame_zero_unavailable",
                    "prediction frame zero is unavailable for identity binding",
                )
            prompts, prompt_metadata = (
                build_prediction_collision_identity_prompts(
                    anchors,
                    prediction_frame=frames[0],
                    observation_config=self.config[
                        "multi_frame_observation"
                    ],
                    binding_config=identity["prediction_binding"],
                )
            )
        else:
            raise ValueError(
                f"unsupported collision observation role: {observation_role!r}"
            )
        observation = self._observe_expected(
            frames,
            expected_count=expected_count,
            entity_ids=entity_ids,
            available=available,
            prompts_override=prompts,
            prompt_metadata_override=prompt_metadata,
            observation_role=observation_role,
        )
        valid = observation[1]
        if valid.shape != (len(frames), expected_count) or not bool(
            np.all(valid[0])
        ):
            code_prefix = (
                "reference" if observation_role == "reference" else "prediction"
            )
            raise SceneAnalysisError(
                f"collision_{code_prefix}_frame_zero_segmentation_unconfirmed",
                "not every bound collision identity has a valid frame-zero mask",
            )
        return observation

    def _postprocess_expected_masks(
        self,
        masks: list[list[np.ndarray]],
        prompts: list[MaskPrompt],
        *,
        observation_role: str,
    ) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
        if observation_role not in {"reference", "prediction"}:
            raise ValueError("collision v6 identity latch requires a role")
        return latch_collision_identity_masks(
            masks,
            prompts,
            config=self.config["subject_identity"]["tracking"],
        )

    def _fixed_prediction_track_ids(
        self, entity_ids: Sequence[str]
    ) -> Mapping[str, str]:
        return {
            str(entity_id): f"direct_{index:03d}"
            for index, entity_id in enumerate(entity_ids)
        }

    def _terminal_prediction_observation_failure(
        self,
        observation: OpenWorldObservation,
        *,
        entity_ids: Sequence[str],
        frame_count: int,
    ) -> tuple[dict[str, str], dict[str, Any]] | None:
        direct_ids = {
            f"direct_{index:03d}" for index in range(len(entity_ids))
        }
        config = self.config["subject_identity"]["unexpected_participant"]
        minimum_frames = int(config["minimum_observed_frames"])
        minimum_fraction = float(config["minimum_observed_frame_fraction"])
        offending = []
        for track in observation.tracks:
            observed_frames = len(track.detections)
            observed_fraction = observed_frames / max(frame_count, 1)
            if (
                track.track_id not in direct_ids
                and track.evidence_tier is EvidenceTier.PARTICIPANT
                and track.confirmed
                and observed_frames >= minimum_frames
                and observed_fraction >= minimum_fraction
            ):
                offending.append(
                    {
                        "track_id": track.track_id,
                        "observed_frames": observed_frames,
                        "observed_frame_fraction": observed_fraction,
                        "first_frame": track.detections[0].frame_index,
                        "last_frame": track.detections[-1].frame_index,
                    }
                )
        if not offending:
            return None
        return (
            {
                "code": "collision_prediction_unexpected_participant",
                "reason": (
                    "prediction contains a persistent high-confidence physical "
                    "participant outside the frame-zero-bound collision roles"
                ),
            },
            {
                "status": "failed_closed_on_unexpected_participant",
                "minimum_observed_frames": minimum_frames,
                "minimum_observed_frame_fraction": minimum_fraction,
                "offending_tracks": offending,
                "original_observation": observation.diagnostics,
            },
        )


__all__ = [
    "CollisionFailClosedCaseEvaluator",
    "CollisionIdentityContext",
]
