from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.csti import (
    CSTIContractError,
    build_csti_input_from_aligned_masks,
    build_csti_input_from_frame_matches,
    normalize_observer_mask,
)
from physbench.evaluation.common.entities import (
    EntityMatch,
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    ReferenceCapability,
)


class CSTIAdaptersTest(unittest.TestCase):
    def test_frame_matches_preserve_gt_entities_and_normalize_255_masks(self) -> None:
        observed = np.zeros((9, 11), dtype=np.uint8)
        observed[3:5, 4:7] = 255
        observation = OpenWorldObservation(
            tracks=(
                OpenWorldTrack(
                    track_id="track_b",
                    detections=(
                        ObjectDetection(
                            frame_index=1,
                            detection_id="d1",
                            xy=np.array([5.0, 4.0]),
                            area_px2=6.0,
                            entity_class="block",
                            mask=observed,
                        ),
                    ),
                    confirmed=True,
                    evidence_tier=EvidenceTier.PARTICIPANT,
                ),
            ),
            overflow_counts=np.zeros(2),
        )
        empty = tuple(np.zeros((9, 11), dtype=bool) for _ in range(2))
        reference_b = list(empty)
        reference_b[1] = observed > 0

        value = build_csti_input_from_frame_matches(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0, 0.0625),
            frame_shape=(9, 11),
            expected_entities=(("a", "left"), ("b", "right")),
            reference_masks_by_entity={"a": empty, "b": tuple(reference_b)},
            prediction_observation=observation,
            matches=(EntityMatch(1, "b", "track_b", 1.0, 0.0),),
        )

        self.assertEqual(("a", "b"), tuple(item.entity_id for item in value.entities))
        self.assertIsNone(value.entities[0].prediction_masks)
        prediction_masks = value.entities[1].prediction_masks
        assert prediction_masks is not None
        self.assertEqual(np.dtype(bool), prediction_masks[1].dtype)
        self.assertFalse(prediction_masks[1].flags.writeable)
        np.testing.assert_array_equal(prediction_masks[1], observed > 0)
        self.assertEqual(("track_b",), value.entities[1].matched_prediction_track_ids)

    def test_normalization_rejects_float_and_wrong_shape(self) -> None:
        with self.assertRaises(CSTIContractError):
            normalize_observer_mask(np.zeros((3, 4), dtype=np.float32), frame_shape=(3, 4))
        with self.assertRaises(CSTIContractError):
            normalize_observer_mask(np.zeros((3, 5), dtype=np.uint8), frame_shape=(3, 4))

    def test_match_with_missing_mask_is_matched_but_frame_stays_empty(self) -> None:
        observation = self._observation(
            self._track("track_a", frame_index=0, mask=None),
            frame_count=2,
        )
        empty = tuple(np.zeros((5, 7), dtype=bool) for _ in range(2))

        value = build_csti_input_from_frame_matches(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0, 1.0),
            frame_shape=(5, 7),
            expected_entities=(("a", "subject"),),
            reference_masks_by_entity={"a": empty},
            prediction_observation=observation,
            matches=(EntityMatch(0, "a", "track_a", 0.5, 0.5),),
        )

        prediction = value.entities[0].prediction_masks
        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertFalse(np.any(prediction[0]))
        self.assertFalse(np.any(prediction[1]))
        self.assertEqual(("track_a",), value.entities[0].matched_prediction_track_ids)

    def test_frame_match_contract_rejects_unknown_and_duplicate_slots(self) -> None:
        observation = self._observation(
            self._track("track_a", frame_index=0),
            self._track("track_b", frame_index=0),
            frame_count=1,
        )
        empty = (np.zeros((5, 7), dtype=bool),)
        kwargs = {
            "reference_capability": ReferenceCapability.SAME_CASE_GT,
            "times_s": (0.0,),
            "frame_shape": (5, 7),
            "expected_entities": (("a", "left"), ("b", "right")),
            "reference_masks_by_entity": {"a": empty, "b": empty},
            "prediction_observation": observation,
        }
        invalid_matches = (
            (EntityMatch(0, "unknown", "track_a", 1.0, 0.0),),
            (EntityMatch(0, "a", "unknown", 1.0, 0.0),),
            (
                EntityMatch(0, "a", "track_a", 1.0, 0.0),
                EntityMatch(0, "a", "track_b", 1.0, 0.0),
            ),
            (
                EntityMatch(0, "a", "track_a", 1.0, 0.0),
                EntityMatch(0, "b", "track_a", 1.0, 0.0),
            ),
        )

        for matches in invalid_matches:
            with self.subTest(matches=matches), self.assertRaises(CSTIContractError):
                build_csti_input_from_frame_matches(matches=matches, **kwargs)

    def test_out_of_range_detection_is_rejected(self) -> None:
        observation = self._observation(
            self._track("track_a", frame_index=2),
            frame_count=2,
        )
        empty = tuple(np.zeros((5, 7), dtype=bool) for _ in range(2))

        with self.assertRaises(CSTIContractError):
            build_csti_input_from_frame_matches(
                reference_capability=ReferenceCapability.SAME_CASE_GT,
                times_s=(0.0, 1.0),
                frame_shape=(5, 7),
                expected_entities=(("a", "subject"),),
                reference_masks_by_entity={"a": empty},
                prediction_observation=observation,
                matches=(),
            )

    def test_aligned_masks_preserve_manifest_order_and_sort_track_ids(self) -> None:
        empty = tuple(np.zeros((5, 7), dtype=np.uint8) for _ in range(2))
        observed = list(empty)
        observed[1] = observed[1].copy()
        observed[1][2:4, 3:5] = 255

        value = build_csti_input_from_aligned_masks(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0, 1.0),
            frame_shape=(5, 7),
            expected_entities=(("a", "left"), ("b", "right")),
            reference_masks_by_entity={"b": tuple(observed), "a": empty},
            prediction_masks_by_entity={"b": tuple(observed), "a": None},
            matched_track_ids_by_entity={"b": ("z", "a", "z"), "a": ()},
        )

        self.assertEqual(("a", "b"), tuple(item.entity_id for item in value.entities))
        self.assertIsNone(value.entities[0].prediction_masks)
        self.assertEqual(("a", "z"), value.entities[1].matched_prediction_track_ids)
        prediction = value.entities[1].prediction_masks
        assert prediction is not None
        self.assertEqual(np.dtype(bool), prediction[1].dtype)
        np.testing.assert_array_equal(prediction[1], observed[1] > 0)

    def test_aligned_masks_require_exact_entity_coverage_and_lengths(self) -> None:
        empty = tuple(np.zeros((5, 7), dtype=bool) for _ in range(2))
        common = {
            "reference_capability": ReferenceCapability.SAME_CASE_GT,
            "times_s": (0.0, 1.0),
            "frame_shape": (5, 7),
            "expected_entities": (("a", "subject"),),
            "prediction_masks_by_entity": {"a": empty},
            "matched_track_ids_by_entity": {"a": ()},
        }
        with self.assertRaises(CSTIContractError):
            build_csti_input_from_aligned_masks(reference_masks_by_entity={}, **common)
        with self.assertRaises(CSTIContractError):
            build_csti_input_from_aligned_masks(
                reference_masks_by_entity={"a": empty[:1]},
                **common,
            )

    def test_adapters_reject_coercible_timestamps_ids_and_track_ids(self) -> None:
        empty = (np.zeros((5, 7), dtype=bool),)
        common = {
            "reference_capability": ReferenceCapability.SAME_CASE_GT,
            "frame_shape": (5, 7),
            "reference_masks_by_entity": {"a": empty},
            "prediction_masks_by_entity": {"a": empty},
        }
        invalid = (
            {
                **common,
                "times_s": ("0.0",),
                "expected_entities": (("a", "subject"),),
                "matched_track_ids_by_entity": {"a": ("track_a",)},
            },
            {
                **common,
                "times_s": (0.0,),
                "expected_entities": ((1, "subject"),),
                "reference_masks_by_entity": {1: empty},
                "prediction_masks_by_entity": {1: empty},
                "matched_track_ids_by_entity": {1: ("track_a",)},
            },
            {
                **common,
                "times_s": (0.0,),
                "expected_entities": (("a", "subject"),),
                "matched_track_ids_by_entity": {"a": (7,)},
            },
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(CSTIContractError):
                build_csti_input_from_aligned_masks(**kwargs)

    @staticmethod
    def _track(
        track_id: str,
        *,
        frame_index: int,
        mask: np.ndarray | None = None,
    ) -> OpenWorldTrack:
        if mask is None and frame_index != 0:
            mask = np.ones((5, 7), dtype=np.uint8) * 255
        return OpenWorldTrack(
            track_id=track_id,
            detections=(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=f"{track_id}_{frame_index}",
                    xy=np.array([3.0, 2.0]),
                    area_px2=1.0,
                    entity_class="block",
                    mask=mask,
                ),
            ),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )

    @staticmethod
    def _observation(*tracks: OpenWorldTrack, frame_count: int) -> OpenWorldObservation:
        return OpenWorldObservation(
            tracks=tuple(tracks),
            overflow_counts=np.zeros(frame_count),
        )


if __name__ == "__main__":
    unittest.main()
