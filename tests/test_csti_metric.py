from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import physbench.evaluation.common.csti.metric as csti_metric

from physbench.evaluation.common.csti import (
    CSTIConfig,
    CSTIContractError,
    CSTIEntityTube,
    CSTIInput,
    evaluate_csti,
    not_applicable_csti_metric,
    score_postcondition_tube,
    score_postcondition_tube_reference,
    zero_csti_metric,
)
from physbench.evaluation.common.entities import ReferenceCapability


FIXED_CONFIG = {
    "enabled": True,
    "algorithm": "exact_full_tube_edt",
    "spatial_tolerance_fraction": 0.005,
    "temporal_tolerance_s": 0.05,
    "condition_frame_policy": "exclude_initial_samples",
    "initial_frames_excluded": 1,
    "score_aggregation": "full_tube",
    "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
    "case_aggregation": "mean_gt_entities",
    "timeline_policy": "physical_overlap",
    "mask_resolution": "scene_analysis_native",
}

ADAPTIVE_CONFIG = {
    **{
        key: value
        for key, value in FIXED_CONFIG.items()
        if key != "spatial_tolerance_fraction"
    },
    "spatial_tolerance_policy": (
        "reference_tube_equivalent_diameter_v1"
    ),
    "spatial_tolerance_radius_ratio": 0.5,
}

CONFIG = CSTIConfig.from_mapping(FIXED_CONFIG)
TIMES = (0.0, 0.0625, 0.125, 0.1875)


def point_tube(*, x_offset: int = 0, delay: int = 0) -> tuple[np.ndarray, ...]:
    tube = np.zeros((4, 41, 41), dtype=bool)
    for frame in range(delay, 4):
        tube[frame, 20, 10 + frame - delay + x_offset] = True
    return tuple(tube)


def config_with(**changes: object) -> CSTIConfig:
    value = dict(FIXED_CONFIG)
    value.update(changes)
    return CSTIConfig.from_mapping(value)


class CSTIMetricTest(unittest.TestCase):
    def _adaptive_config(self) -> CSTIConfig:
        try:
            return CSTIConfig.from_mapping(ADAPTIVE_CONFIG)
        except CSTIContractError as exc:
            self.fail(f"adaptive CSTI config was rejected: {exc}")

    def test_adaptive_spatial_tolerance_uses_reference_tube_area(self) -> None:
        config = self._adaptive_config()
        reference = []
        prediction = []
        for index, _ in enumerate(TIMES):
            reference_mask = np.zeros((41, 41), dtype=bool)
            if index == 0:
                # The condition frame is excluded from both scoring and the
                # postcondition Tube scale estimate.
                reference_mask[10:31, 10:31] = True
            else:
                reference_mask[18:21, 18:21] = True
                # A remote outlier must affect area by one pixel, not turn the
                # bounding-box width into the Tube diameter.
                reference_mask[2, 38] = True
            prediction_mask = np.roll(reference_mask, 1, axis=1)
            reference.append(reference_mask)
            prediction.append(prediction_mask)
        value = CSTIInput(
            ReferenceCapability.SAME_CASE_GT,
            TIMES,
            (41, 41),
            (
                CSTIEntityTube(
                    "a",
                    "subject",
                    tuple(reference),
                    tuple(prediction),
                    ("track_a",),
                ),
            ),
        )

        result = evaluate_csti(
            value,
            expected_entities=(("a", "subject"),),
            config=config,
        )

        tolerance = result["objects"][0]["spatial_tolerance"]
        expected_diameter = 2.0 * math.sqrt(10.0 / math.pi)
        self.assertEqual(
            "reference_tube_equivalent_diameter_v1",
            tolerance["policy"],
        )
        self.assertEqual(3, tolerance["reference_nonempty_frame_count"])
        self.assertAlmostEqual(
            expected_diameter,
            tolerance["reference_tube_diameter_px"],
            places=12,
        )
        self.assertAlmostEqual(
            0.5 * expected_diameter,
            tolerance["effective_radius_px"],
            places=12,
        )

    def test_wider_reference_tube_receives_wider_spatial_support(self) -> None:
        config = self._adaptive_config()
        small_reference = np.zeros((4, 41, 41), dtype=bool)
        small_prediction = np.zeros_like(small_reference)
        small_reference[:, 20, 10] = True
        small_prediction[:, 20, 12] = True
        wide_reference = np.zeros((4, 41, 41), dtype=bool)
        wide_prediction = np.zeros_like(wide_reference)
        wide_reference[:, 16:25, 5:14] = True
        wide_prediction[:, 16:25, 17:26] = True

        small, _ = score_postcondition_tube(
            tuple(small_reference),
            tuple(small_prediction),
            times_s=TIMES,
            config=config,
        )
        wide, _ = score_postcondition_tube(
            tuple(wide_reference),
            tuple(wide_prediction),
            times_s=TIMES,
            config=config,
        )

        self.assertEqual(0.0, small)
        self.assertGreater(wide, 0.0)

    def test_fixed_config_parses(self) -> None:
        config = CSTIConfig.from_mapping(FIXED_CONFIG)

        self.assertTrue(config.enabled)
        self.assertEqual("exact_full_tube_edt", config.algorithm)
        self.assertEqual(0.005, config.spatial_tolerance_fraction)
        self.assertEqual(0.05, config.temporal_tolerance_s)
        self.assertEqual("exclude_initial_samples", config.condition_frame_policy)
        self.assertEqual(1, config.initial_frames_excluded)
        self.assertEqual("full_tube", config.score_aggregation)
        self.assertEqual(
            (0.25, 0.5, 0.75, 1.0),
            config.diagnostic_prefix_fractions,
        )
        self.assertEqual("mean_gt_entities", config.case_aggregation)
        self.assertEqual("physical_overlap", config.timeline_policy)
        self.assertEqual("scene_analysis_native", config.mask_resolution)

    def test_direct_config_construction_cannot_bypass_validation(self) -> None:
        fields = {
            **FIXED_CONFIG,
            "diagnostic_prefix_fractions": (0.25, 0.5, 0.75, 1.0),
        }
        for field, value in (
            ("enabled", False),
            ("algorithm", "approximate"),
            ("spatial_tolerance_fraction", -1.0),
            ("temporal_tolerance_s", 0.0),
            ("condition_frame_policy", "include_first"),
            ("initial_frames_excluded", 0),
            ("initial_frames_excluded", True),
            ("score_aggregation", "prefix_mean"),
            ("diagnostic_prefix_fractions", (0.5, 1.0)),
        ):
            with self.subTest(field=field):
                with self.assertRaises(CSTIContractError):
                    CSTIConfig(**{**fields, field: value})

    def test_identity_postcondition_score_and_curve_are_one(self) -> None:
        score, curve = score_postcondition_tube(
            point_tube(), point_tube(), times_s=TIMES, config=CONFIG
        )

        self.assertEqual(1.0, score)
        self.assertTrue(all(point["score"] == 1.0 for point in curve))

    def test_condition_frame_cannot_affect_score_or_diagnostics(self) -> None:
        first_reference = list(point_tube())
        first_prediction = list(point_tube())
        second_reference = list(point_tube())
        second_prediction = list(point_tube())
        second_reference[0] = np.flip(second_reference[0], axis=1)
        second_prediction[0] = np.flip(second_prediction[0], axis=0)

        self.assertEqual(
            score_postcondition_tube(
                first_reference,
                first_prediction,
                times_s=TIMES,
                config=CONFIG,
            ),
            score_postcondition_tube(
                second_reference,
                second_prediction,
                times_s=TIMES,
                config=CONFIG,
            ),
        )

    def test_configured_initial_samples_cannot_affect_score_or_diagnostics(self) -> None:
        for excluded in (1, 2, 3):
            with self.subTest(excluded=excluded):
                first_reference = list(point_tube())
                first_prediction = list(point_tube())
                second_reference = list(point_tube())
                second_prediction = list(point_tube())
                for index in range(excluded):
                    second_reference[index] = np.flip(
                        second_reference[index], axis=1
                    )
                    second_prediction[index] = np.flip(
                        second_prediction[index], axis=0
                    )
                config = config_with(initial_frames_excluded=excluded)
                self.assertEqual(
                    score_postcondition_tube(
                        first_reference,
                        first_prediction,
                        times_s=TIMES,
                        config=config,
                    ),
                    score_postcondition_tube(
                        second_reference,
                        second_prediction,
                        times_s=TIMES,
                        config=config,
                    ),
                )

    def test_identity_is_one_for_each_supported_initial_exclusion(self) -> None:
        for excluded in (1, 2, 3):
            with self.subTest(excluded=excluded):
                score, curve = score_postcondition_tube(
                    point_tube(),
                    point_tube(),
                    times_s=TIMES,
                    config=config_with(initial_frames_excluded=excluded),
                )
                self.assertEqual(1.0, score)
                self.assertTrue(all(point["score"] == 1.0 for point in curve))
                self.assertEqual(3, curve[-1]["end_frame_index"])

    def test_initial_exclusion_cannot_consume_whole_tube(self) -> None:
        with self.assertRaises(CSTIContractError) as raised:
            score_postcondition_tube(
                point_tube(),
                point_tube(),
                times_s=TIMES,
                config=config_with(initial_frames_excluded=4),
            )

        self.assertEqual("csti_initial_exclusion_invalid", raised.exception.code)

    def test_condition_only_copy_scores_zero_when_gt_continues(self) -> None:
        reference = point_tube()
        prediction = (reference[0],) + tuple(
            np.zeros_like(mask) for mask in reference[1:]
        )

        score, curve = score_postcondition_tube(
            reference,
            prediction,
            times_s=TIMES,
            config=CONFIG,
        )

        self.assertEqual(0.0, score)
        self.assertEqual(0.0, curve[-1]["score"])

    def test_postcondition_empty_policies_are_explicit(self) -> None:
        empty = tuple(np.zeros((5, 7), dtype=bool) for _ in range(3))
        nonempty = list(empty)
        nonempty[1] = nonempty[1].copy()
        nonempty[1][2, 3] = True
        times = (0.0, 0.1, 0.2)

        both_score, _ = score_postcondition_tube(
            empty, empty, times_s=times, config=CONFIG
        )
        one_score, _ = score_postcondition_tube(
            empty, tuple(nonempty), times_s=times, config=CONFIG
        )

        self.assertEqual(1.0, both_score)
        self.assertEqual(0.0, one_score)

    def test_near_offset_scores_above_far_offset(self) -> None:
        config = config_with(spatial_tolerance_fraction=0.05)
        near, _ = score_postcondition_tube(
            point_tube(), point_tube(x_offset=1), times_s=TIMES, config=config
        )
        far, _ = score_postcondition_tube(
            point_tube(), point_tube(x_offset=6), times_s=TIMES, config=config
        )

        self.assertGreater(near, far)

    def test_spatial_tolerance_does_not_reduce_near_offset_score(self) -> None:
        reference = point_tube()
        prediction = point_tube(x_offset=1)
        narrow, _ = score_postcondition_tube(
            reference,
            prediction,
            times_s=TIMES,
            config=config_with(spatial_tolerance_fraction=0.025),
        )
        wide, _ = score_postcondition_tube(
            reference,
            prediction,
            times_s=TIMES,
            config=config_with(spatial_tolerance_fraction=0.1),
        )

        self.assertGreaterEqual(wide, narrow)

    def test_temporal_tolerance_controls_physical_time_offset(self) -> None:
        reference = [np.zeros((5, 5), dtype=bool) for _ in range(4)]
        prediction = [np.zeros((5, 5), dtype=bool) for _ in range(4)]
        reference[1][2, 2] = True
        prediction[2][2, 2] = True
        times = (0.0, 0.025, 0.05, 0.075)

        narrow, _ = score_postcondition_tube(
            reference,
            prediction,
            times_s=times,
            config=config_with(temporal_tolerance_s=0.03),
        )
        wide, _ = score_postcondition_tube(
            reference,
            prediction,
            times_s=times,
            config=config_with(temporal_tolerance_s=0.1),
        )

        self.assertGreater(wide, narrow)

    def test_same_time_offset_is_independent_of_total_video_duration(self) -> None:
        short_reference = np.zeros((5, 5, 5), dtype=bool)
        short_prediction = np.zeros_like(short_reference)
        short_reference[1, 2, 2] = True
        short_prediction[2, 2, 2] = True
        long_reference = np.zeros((9, 5, 5), dtype=bool)
        long_prediction = np.zeros_like(long_reference)
        long_reference[1, 2, 2] = True
        long_prediction[2, 2, 2] = True
        short_times = tuple(index * 0.025 for index in range(5))
        long_times = tuple(index * 0.025 for index in range(9))

        short, _ = score_postcondition_tube(
            tuple(short_reference),
            tuple(short_prediction),
            times_s=short_times,
            config=CONFIG,
        )
        long, _ = score_postcondition_tube(
            tuple(long_reference),
            tuple(long_prediction),
            times_s=long_times,
            config=CONFIG,
        )

        self.assertAlmostEqual(short, long, places=12)

    def test_final_checkpoint_is_formal_score(self) -> None:
        score, curve = score_postcondition_tube(
            point_tube(),
            point_tube(x_offset=1),
            times_s=TIMES,
            config=CONFIG,
        )

        self.assertEqual(1.0, curve[-1]["fraction"])
        self.assertEqual(3, curve[-1]["end_frame_index"])
        self.assertEqual(TIMES[3], curve[-1]["end_time_s"])
        self.assertEqual(score, curve[-1]["score"])

    def test_short_timeline_deduplicates_checkpoints_and_keeps_final(self) -> None:
        tube = tuple(np.ones((3, 3), dtype=bool) for _ in range(2))

        score, curve = score_postcondition_tube(
            tube,
            tube,
            times_s=(0.0, 0.0625),
            config=CONFIG,
        )

        self.assertEqual(1.0, score)
        self.assertEqual(
            (
                {
                    "fraction": 1.0,
                    "end_frame_index": 1,
                    "end_time_s": 0.0625,
                    "score": 1.0,
                },
            ),
            curve,
        )

    def test_support_crop_equals_full_domain_oracle(self) -> None:
        cases = (
            (point_tube(), point_tube(x_offset=1)),
            (point_tube(), point_tube(delay=2)),
            (
                tuple(np.eye(9, dtype=bool) for _ in range(4)),
                tuple(np.fliplr(np.eye(9, dtype=bool)) for _ in range(4)),
            ),
        )
        for reference, prediction in cases:
            with self.subTest(shape=np.asarray(reference).shape):
                actual_score, actual_curve = score_postcondition_tube(
                    reference,
                    prediction,
                    times_s=TIMES,
                    config=CONFIG,
                )
                oracle_score, oracle_curve = score_postcondition_tube_reference(
                    reference,
                    prediction,
                    times_s=TIMES,
                    config=CONFIG,
                )
                np.testing.assert_allclose(actual_score, oracle_score, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(
                    [point["score"] for point in actual_curve],
                    [point["score"] for point in oracle_curve],
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_default_backend_limits_edt_to_finite_spatial_support(self) -> None:
        with patch.object(
            csti_metric,
            "distance_transform_edt",
            wraps=csti_metric.distance_transform_edt,
        ) as edt:
            score_postcondition_tube(
                point_tube(),
                point_tube(x_offset=1),
                times_s=TIMES,
                config=CONFIG,
            )

        self.assertTrue(edt.call_args_list)
        self.assertTrue(all(call.args[0].shape[1] < 41 for call in edt.call_args_list))
        self.assertTrue(all(call.args[0].shape[2] < 41 for call in edt.call_args_list))

    def test_tracking_gap_reduces_score(self) -> None:
        reference = point_tube()
        prediction = list(reference)
        prediction[2] = np.zeros_like(prediction[2])

        identity, _ = score_postcondition_tube(
            reference, reference, times_s=TIMES, config=CONFIG
        )
        with_gap, _ = score_postcondition_tube(
            reference, prediction, times_s=TIMES, config=CONFIG
        )

        self.assertLess(with_gap, identity)

    def test_bool_and_integer_zero_one_inputs_are_equivalent(self) -> None:
        masks = point_tube()
        bool_score, _ = score_postcondition_tube(
            masks, masks, times_s=TIMES, config=CONFIG
        )
        for dtype in (np.uint8, np.int32):
            with self.subTest(dtype=dtype):
                values = tuple(mask.astype(dtype) for mask in masks)
                score, _ = score_postcondition_tube(
                    values, values, times_s=TIMES, config=CONFIG
                )
                self.assertEqual(bool_score, score)

    def test_uint8_255_and_float_inputs_are_rejected(self) -> None:
        bad_255 = tuple(mask.astype(np.uint8) * 255 for mask in point_tube())
        bad_float = tuple(mask.astype(np.float32) for mask in point_tube())

        with self.assertRaises(CSTIContractError):
            score_postcondition_tube(
                bad_255, bad_255, times_s=TIMES, config=CONFIG
            )
        with self.assertRaises(CSTIContractError):
            score_postcondition_tube(
                bad_float, bad_float, times_s=TIMES, config=CONFIG
            )

    def test_invalid_mask_dimensions_and_shapes_are_rejected(self) -> None:
        with self.assertRaises(CSTIContractError):
            score_postcondition_tube(
                (np.zeros(4, dtype=bool),) * 2,
                (np.zeros(4, dtype=bool),) * 2,
                times_s=(0.0, 0.1),
                config=CONFIG,
            )
        with self.assertRaises(CSTIContractError):
            score_postcondition_tube(
                (np.zeros((3, 4), dtype=bool),) * 2,
                (np.zeros((3, 5), dtype=bool),) * 2,
                times_s=(0.0, 0.1),
                config=CONFIG,
            )

    def test_nonuniform_timeline_and_single_frame_are_rejected(self) -> None:
        masks = point_tube()
        with self.assertRaisesRegex(CSTIContractError, "uniform"):
            score_postcondition_tube(
                masks,
                masks,
                times_s=(0.0, 0.05, 0.15, 0.2),
                config=CONFIG,
            )
        with self.assertRaises(CSTIContractError):
            score_postcondition_tube(
                (masks[0],),
                (masks[0],),
                times_s=(0.0,),
                config=CONFIG,
            )

    def test_short_off_grid_terminal_endpoint_is_excluded_before_edt(self) -> None:
        reference = point_tube()
        prediction = point_tube(x_offset=1)
        regular = score_postcondition_tube(
            reference[:3],
            prediction[:3],
            times_s=(0.0, 0.0625, 0.125),
            config=CONFIG,
        )

        with_endpoint = score_postcondition_tube(
            reference,
            prediction,
            times_s=(0.0, 0.0625, 0.125, 0.15),
            config=CONFIG,
        )

        self.assertEqual(regular, with_endpoint)
        self.assertEqual(2, with_endpoint[1][-1]["end_frame_index"])
        self.assertEqual(0.125, with_endpoint[1][-1]["end_time_s"])

    def test_nonfinite_or_nonpositive_config_numbers_are_rejected(self) -> None:
        for field, value in (
            ("spatial_tolerance_fraction", math.nan),
            ("spatial_tolerance_fraction", 0.0),
            ("temporal_tolerance_s", math.inf),
            ("temporal_tolerance_s", -0.1),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(CSTIContractError):
                config_with(**{field: value})

    def test_score_and_diagnostics_remain_in_unit_interval(self) -> None:
        score, curve = score_postcondition_tube(
            point_tube(),
            point_tube(x_offset=1),
            times_s=TIMES,
            config=CONFIG,
        )

        self.assertTrue(0.0 <= score <= 1.0 and math.isfinite(score))
        self.assertTrue(
            all(0.0 <= point["score"] <= 1.0 for point in curve)
        )

    def test_tiny_tube_matches_manual_soft_iou(self) -> None:
        reference = np.zeros((2, 1, 3), dtype=bool)
        prediction = np.zeros_like(reference)
        reference[1, 0, 0] = True
        prediction[1, 0, 1] = True

        score, _ = score_postcondition_tube(
            tuple(reference),
            tuple(prediction),
            times_s=(0.0, 0.1),
            config=config_with(spatial_tolerance_fraction=1.0),
        )

        self.assertAlmostEqual(0.4, score, places=12)

    def test_case_score_uses_manifest_order_and_unmatched_zero(self) -> None:
        masks = point_tube()
        value = CSTIInput(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=TIMES,
            frame_shape=(41, 41),
            entities=(
                CSTIEntityTube("b", "right", masks, None),
                CSTIEntityTube(
                    "a", "left", masks, masks, ("track_a",)
                ),
            ),
        )

        result = evaluate_csti(
            value,
            expected_entities=(("a", "left"), ("b", "right")),
            config=CONFIG,
        )

        self.assertEqual("evaluated", result["status"])
        self.assertEqual(0.5, result["score"])
        self.assertEqual(4, result["frame_count"])
        self.assertEqual(0, result["terminal_frames_excluded"])
        self.assertEqual(1, result["initial_frames_excluded"])
        self.assertEqual(3, result["scored_frame_count"])
        self.assertEqual(["a", "b"], [item["entity_id"] for item in result["objects"]])
        self.assertEqual(1.0, result["objects"][0]["score"])
        self.assertEqual(
            1.0,
            result["objects"][0]["diagnostic_prefix_curve"][-1]["score"],
        )
        self.assertNotIn("curve", result["objects"][0])
        self.assertTrue(result["objects"][0]["matched"])
        self.assertEqual(["track_a"], result["objects"][0]["matched_prediction_track_ids"])
        self.assertEqual(0.0, result["objects"][1]["score"])
        self.assertIsNone(result["objects"][1]["diagnostic_prefix_curve"])
        self.assertFalse(result["objects"][1]["matched"])

    def test_case_metadata_records_excluded_off_grid_terminal_endpoint(self) -> None:
        masks = point_tube()
        value = CSTIInput(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0, 0.0625, 0.125, 0.15),
            frame_shape=(41, 41),
            entities=(
                CSTIEntityTube("a", "subject", masks, masks, ("track_a",)),
            ),
        )

        result = evaluate_csti(
            value,
            expected_entities=(("a", "subject"),),
            config=CONFIG,
        )

        self.assertEqual(4, result["frame_count"])
        self.assertEqual(1, result["terminal_frames_excluded"])
        self.assertEqual(2, result["scored_frame_count"])
        self.assertEqual(
            "exclude_off_grid_endpoint",
            result["parameters"]["terminal_frame_policy"],
        )

    def test_invalid_capability_and_matched_metadata_are_rejected(self) -> None:
        masks = point_tube()
        with self.assertRaises(CSTIContractError):
            CSTIInput(
                reference_capability="same_case_gt",  # type: ignore[arg-type]
                times_s=TIMES,
                frame_shape=(41, 41),
                entities=(),
            )

        value = CSTIInput(
            ReferenceCapability.SAME_CASE_GT,
            TIMES,
            (41, 41),
            (CSTIEntityTube("a", "left", masks, masks),),
        )
        with self.assertRaises(CSTIContractError):
            evaluate_csti(
                value,
                expected_entities=(("a", "left"),),
                config=CONFIG,
            )

    def test_case_contract_rejects_entity_role_timeline_and_shape_mismatches(self) -> None:
        masks = point_tube()
        entity = CSTIEntityTube("a", "left", masks, masks, ("track_a",))
        valid = CSTIInput(
            ReferenceCapability.SAME_CASE_GT,
            TIMES,
            (41, 41),
            (entity,),
        )
        cases = (
            (valid, (("b", "left"),)),
            (valid, (("a", "right"),)),
            (
                CSTIInput(
                    ReferenceCapability.SAME_CASE_GT,
                    (0.0, 0.1, 0.2, 0.4),
                    (41, 41),
                    (entity,),
                ),
                (("a", "left"),),
            ),
            (
                CSTIInput(
                    ReferenceCapability.SAME_CASE_GT,
                    TIMES,
                    (40, 41),
                    (entity,),
                ),
                (("a", "left"),),
            ),
        )

        for value, expected in cases:
            with self.subTest(expected=expected), self.assertRaises(CSTIContractError):
                evaluate_csti(value, expected_entities=expected, config=CONFIG)

    def test_zero_and_not_applicable_builders_are_explicit(self) -> None:
        zero = zero_csti_metric(
            expected_entities=(("a", "left"), ("b", "right")),
            config=CONFIG,
            degradation_code="prediction_missing",
            degradation_reason="No generated video",
        )
        unavailable = not_applicable_csti_metric(
            config=CONFIG,
            reason_code="csti_requires_same_case_gt",
        )

        self.assertEqual("evaluated", zero["status"])
        self.assertEqual(0.0, zero["score"])
        self.assertEqual(["a", "b"], [item["entity_id"] for item in zero["objects"]])
        self.assertTrue(
            all(
                item["score"] == 0.0
                and item["diagnostic_prefix_curve"] is None
                for item in zero["objects"]
            )
        )
        self.assertEqual("prediction_missing", zero["degradation"]["code"])
        self.assertEqual(
            {
                "status": "not_applicable",
                "score": None,
                "reason_code": "csti_requires_same_case_gt",
                "algorithm": "exact_full_tube_edt",
                "parameters": {
                    "spatial_tolerance_fraction": 0.005,
                    "temporal_tolerance_s": 0.05,
                    "condition_frame_policy": "exclude_initial_samples",
                    "initial_frames_excluded": 1,
                    "terminal_frame_policy": "exclude_off_grid_endpoint",
                },
                "aggregation": {
                    "time": "full_tube",
                    "entities": "mean_all_gt_entities",
                },
            },
            unavailable,
        )


if __name__ == "__main__":
    unittest.main()
