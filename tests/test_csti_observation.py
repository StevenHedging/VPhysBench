from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.csti.observation import (
    CSTIObserverConfig,
    SemanticCandidateTube,
    match_initial_identities,
)
from physbench.evaluation.common.entities import EntitySpec


def _mask(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    shape: tuple[int, int] = (8, 12),
) -> np.ndarray:
    value = np.zeros(shape, dtype=bool)
    value[top:bottom, left:right] = True
    return value


def _entity(entity_id: str, entity_class: str) -> EntitySpec:
    return EntitySpec(
        entity_id=entity_id,
        role_id=entity_id,
        entity_class=entity_class,
    )


def _candidate(
    candidate_id: str,
    prompt_group_id: str,
    first_mask: np.ndarray,
    *,
    frame_count: int = 4,
) -> SemanticCandidateTube:
    masks = np.repeat(first_mask[None], frame_count, axis=0)
    return SemanticCandidateTube(
        candidate_id=candidate_id,
        prompt_group_id=prompt_group_id,
        backend_object_id=int(candidate_id.rsplit(":", 1)[-1]),
        masks=masks,
        boxes_xywh=np.zeros((frame_count, 4), dtype=np.float32),
        confidences=np.ones(frame_count, dtype=np.float32),
    )


def _config(
    *,
    threshold: float = 0.5,
    ambiguity_margin: float = 0.0,
) -> CSTIObserverConfig:
    return CSTIObserverConfig.from_mapping(
        {
            "prompt_groups": [
                {
                    "id": "ball",
                    "text": "ball",
                    "entity_classes": ["ball"],
                },
                {
                    "id": "block",
                    "text": "sliding block",
                    "entity_classes": ["block"],
                },
            ],
            "initial_match_iou_threshold": threshold,
            "initial_match_ambiguity_margin": ambiguity_margin,
            "termination_patience": 3,
            "minimum_mask_pixels": 1,
            "minimum_observation_confidence": 0.0,
        }
    )


class CSTIInitialMatchingTest(unittest.TestCase):
    def test_hungarian_matching_uses_first_frame_iou_not_candidate_order(self) -> None:
        left = _mask(1, 2, 4, 5)
        right = _mask(8, 2, 11, 5)
        result = match_initial_identities(
            entities=(_entity("left", "ball"), _entity("right", "ball")),
            reference_masks_by_entity={"left": left, "right": right},
            candidates=(
                _candidate("ball:9", "ball", right),
                _candidate("ball:4", "ball", left),
            ),
            config=_config(),
        )

        self.assertTrue(result.success)
        self.assertEqual(
            {"left": "ball:4", "right": "ball:9"},
            result.initial_matching,
        )
        self.assertEqual({"left": 1.0, "right": 1.0}, result.matching_iou)

    def test_extra_prediction_is_ignored_after_complete_matching(self) -> None:
        target = _mask(1, 2, 4, 5)
        extra = _mask(5, 2, 7, 5)
        result = match_initial_identities(
            entities=(_entity("body", "ball"),),
            reference_masks_by_entity={"body": target},
            candidates=(
                _candidate("ball:1", "ball", target),
                _candidate("ball:2", "ball", extra),
            ),
            config=_config(),
        )

        self.assertTrue(result.success)
        self.assertEqual({"body": "ball:1"}, result.initial_matching)
        self.assertEqual(("ball:2",), result.ignored_candidate_ids)

    def test_semantically_incompatible_candidate_cannot_match(self) -> None:
        target = _mask(1, 2, 4, 5)
        result = match_initial_identities(
            entities=(_entity("body", "ball"),),
            reference_masks_by_entity={"body": target},
            candidates=(_candidate("block:1", "block", target),),
            config=_config(),
        )

        self.assertFalse(result.success)
        self.assertEqual("insufficient_semantic_candidates", result.failure.code)
        self.assertEqual({}, result.initial_matching)

    def test_detection_shortage_is_initialization_failure_not_partial_match(self) -> None:
        left = _mask(1, 2, 4, 5)
        right = _mask(8, 2, 11, 5)
        result = match_initial_identities(
            entities=(_entity("left", "ball"), _entity("right", "ball")),
            reference_masks_by_entity={"left": left, "right": right},
            candidates=(_candidate("ball:1", "ball", left),),
            config=_config(),
        )

        self.assertFalse(result.success)
        self.assertEqual("insufficient_semantic_candidates", result.failure.code)
        self.assertEqual({}, result.initial_matching)

    def test_match_below_threshold_is_initialization_failure(self) -> None:
        target = _mask(1, 2, 4, 5)
        far = _mask(8, 2, 11, 5)
        result = match_initial_identities(
            entities=(_entity("body", "ball"),),
            reference_masks_by_entity={"body": target},
            candidates=(_candidate("ball:1", "ball", far),),
            config=_config(threshold=0.25),
        )

        self.assertFalse(result.success)
        self.assertEqual("initial_match_iou_below_threshold", result.failure.code)
        self.assertEqual(0.0, result.failure.details["matching_iou"]["body"])

    def test_near_tied_complete_assignments_are_rejected_as_ambiguous(self) -> None:
        shared = _mask(3, 2, 9, 6)
        result = match_initial_identities(
            entities=(_entity("one", "ball"), _entity("two", "ball")),
            reference_masks_by_entity={"one": shared, "two": shared.copy()},
            candidates=(
                _candidate("ball:1", "ball", shared),
                _candidate("ball:2", "ball", shared.copy()),
            ),
            config=_config(threshold=0.5, ambiguity_margin=0.01),
        )

        self.assertFalse(result.success)
        self.assertEqual("initial_assignment_ambiguous", result.failure.code)
        self.assertEqual(0.0, result.failure.details["assignment_margin"])


if __name__ == "__main__":
    unittest.main()
