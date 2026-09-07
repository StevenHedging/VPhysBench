from __future__ import annotations

import unittest

import numpy as np

from physbench.evaluation.common.csti.observation import (
    CSTIObserverConfig,
    InitialIdentityMatch,
    PromptGroupConfig,
    SemanticCandidateTube,
    build_locked_prediction_tubes,
    decorate_csti_metric,
    evaluator_init_failure_metric,
    is_valid_track_observation,
    match_initial_identities,
    observe_csti_tubes,
)
from physbench.evaluation.common.csti import (
    CSTIConfig,
    CSTIContractError,
    CSTIEntityTube,
    CSTIInput,
    evaluate_csti,
)
from physbench.evaluation.common.entities import EntitySpec, ReferenceCapability


_MISSING = object()


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


def _candidate_frames(
    candidate_id: str,
    prompt_group_id: str,
    frames: list[np.ndarray],
    *,
    confidences: list[float] | None = None,
) -> SemanticCandidateTube:
    frame_count = len(frames)
    return SemanticCandidateTube(
        candidate_id=candidate_id,
        prompt_group_id=prompt_group_id,
        backend_object_id=int(candidate_id.rsplit(":", 1)[-1]),
        masks=np.stack(frames),
        boxes_xywh=np.zeros((frame_count, 4), dtype=np.float32),
        confidences=np.asarray(
            confidences if confidences is not None else [1.0] * frame_count,
            dtype=np.float32,
        ),
    )


def _disjoint_hundred_pixel_masks(
    entity_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    shape = (10, 10 * len(entity_ids))
    result: dict[str, np.ndarray] = {}
    for index, entity_id in enumerate(entity_ids):
        mask = np.zeros(shape, dtype=bool)
        mask[:, index * 10 : (index + 1) * 10] = True
        result[entity_id] = mask
    return result


def _candidate_from_disjoint_counts(
    candidate_id: str,
    counts: tuple[int, ...],
    *,
    prompt_group_id: str = "ball",
) -> SemanticCandidateTube:
    shape = (10, 10 * len(counts))
    mask = np.zeros(shape, dtype=bool)
    for index, count in enumerate(counts):
        for offset in range(count):
            row, column = divmod(offset, 10)
            mask[row, index * 10 + column] = True
    return SemanticCandidateTube(
        candidate_id=candidate_id,
        prompt_group_id=prompt_group_id,
        backend_object_id=sum(ord(character) for character in candidate_id),
        masks=np.repeat(mask[None], 4, axis=0),
        boxes_xywh=np.zeros((4, 4), dtype=np.float32),
        confidences=np.ones(4, dtype=np.float32),
    )


def _config(
    *,
    threshold: float = 0.5,
    ambiguity_margin: float = 0.0,
    policy: str | None = None,
    observer_revision: object = _MISSING,
    segmenter: dict[str, object] | None = None,
) -> CSTIObserverConfig:
    mapping = {
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
    if policy is not None:
        mapping["initial_matching_policy"] = policy
    if observer_revision is not _MISSING:
        mapping["observer_revision"] = observer_revision
    if segmenter is not None:
        mapping["segmenter"] = segmenter
    return CSTIObserverConfig.from_mapping(mapping)


def _metric_config() -> CSTIConfig:
    return CSTIConfig.from_mapping(
        {
            "enabled": True,
            "algorithm": "exact_full_tube_edt",
            "spatial_tolerance_policy": "reference_tube_equivalent_diameter_v1",
            "spatial_tolerance_radius_ratio": 0.5,
            "temporal_tolerance_s": 0.025,
            "condition_frame_policy": "exclude_initial_samples",
            "initial_frames_excluded": 1,
            "score_aggregation": "full_tube",
            "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
    )


class CSTIInitialMatchingTest(unittest.TestCase):
    def test_config_preserves_legacy_matching_policy_default(self) -> None:
        self.assertEqual(
            "maximum_total_iou_v1", _config().initial_matching_policy
        )

    def test_config_derives_revision_two_from_each_new_component_policy(self) -> None:
        configs = (
            _config(policy="threshold_feasible_v2"),
            _config(
                segmenter={
                    "initial_detection": {
                        "score_threshold": 0.2,
                        "new_object_threshold": 0.2,
                    }
                }
            ),
            _config(segmenter={"masklet_confirmation_enable": False}),
            _config(segmenter={"discovery_pruning_policy": "fixed_initial_ids_v1"}),
        )

        self.assertEqual((2, 2, 2, 2), tuple(x.observer_revision for x in configs))

    def test_backend_native_discovery_policy_keeps_implicit_revision_one(
        self,
    ) -> None:
        config = _config(
            segmenter={"discovery_pruning_policy": "backend_native_v1"}
        )

        self.assertEqual(1, config.observer_revision)

    def test_config_rejects_explicit_revision_one_with_new_component_policy(
        self,
    ) -> None:
        configurations = (
            {"policy": "threshold_feasible_v2"},
            {
                "segmenter": {
                    "initial_detection": {
                        "score_threshold": 0.2,
                        "new_object_threshold": 0.2,
                    }
                }
            },
            {"segmenter": {"masklet_confirmation_enable": False}},
            {
                "segmenter": {
                    "discovery_pruning_policy": "fixed_initial_ids_v1"
                }
            },
        )
        for options in configurations:
            with self.subTest(options=options):
                with self.assertRaisesRegex(CSTIContractError, "revision 1"):
                    _config(observer_revision=1, **options)

    def test_explicit_revision_two_allows_legacy_component_ablation(self) -> None:
        config = _config(observer_revision=2)

        self.assertEqual(2, config.observer_revision)
        self.assertEqual("maximum_total_iou_v1", config.initial_matching_policy)
        self.assertEqual({}, config.segmenter)

    def test_mapping_rejects_explicit_null_observer_revision(self) -> None:
        with self.assertRaisesRegex(CSTIContractError, "revision"):
            _config(observer_revision=None)

    def test_config_preserves_legacy_positional_constructor_order(self) -> None:
        config = CSTIObserverConfig(
            (
                PromptGroupConfig(
                    group_id="ball",
                    text="ball",
                    entity_classes=("ball",),
                ),
            ),
            0.25,
            0.02,
            3,
            4,
            0.0,
            {"backend": "legacy"},
            True,
        )

        self.assertEqual({"backend": "legacy"}, config.segmenter)
        self.assertTrue(config.debug_outputs)
        self.assertEqual("maximum_total_iou_v1", config.initial_matching_policy)
        self.assertEqual(1, config.observer_revision)

    def test_config_rejects_unknown_matching_policy(self) -> None:
        with self.assertRaisesRegex(CSTIContractError, "matching policy"):
            _config(policy="threshold_feasible_v3")

    def test_config_rejects_unknown_observer_revision(self) -> None:
        for invalid in (True, 0, 3, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(CSTIContractError, "revision"):
                    CSTIObserverConfig(
                        prompt_groups=(
                            PromptGroupConfig(
                                group_id="ball",
                                text="ball",
                                entity_classes=("ball",),
                            ),
                        ),
                        initial_match_iou_threshold=0.25,
                        initial_match_ambiguity_margin=0.02,
                        termination_patience=3,
                        minimum_mask_pixels=4,
                        minimum_observation_confidence=0.0,
                        observer_revision=invalid,  # type: ignore[arg-type]
                    )

    def test_config_rejects_entity_class_shared_by_prompt_groups(self) -> None:
        mapping = {
            "prompt_groups": [
                {"id": "round", "text": "ball", "entity_classes": ["ball"]},
                {
                    "id": "metal",
                    "text": "metal ball",
                    "entity_classes": ["ball"],
                },
            ],
            "initial_match_iou_threshold": 0.25,
            "initial_match_ambiguity_margin": 0.02,
            "termination_patience": 3,
            "minimum_mask_pixels": 4,
            "minimum_observation_confidence": 0.0,
        }

        with self.assertRaisesRegex(
            CSTIContractError, "exactly one prompt group"
        ):
            CSTIObserverConfig.from_mapping(mapping)

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

    def test_single_entity_single_candidate_has_no_false_alternative(self) -> None:
        target = _mask(2, 2, 6, 6)

        result = match_initial_identities(
            entities=(_entity("body", "ball"),),
            reference_masks_by_entity={"body": target},
            candidates=(_candidate("ball:4", "ball", target),),
            config=_config(ambiguity_margin=0.02),
        )

        self.assertTrue(result.success)
        self.assertEqual({"body": "ball:4"}, result.initial_matching)
        self.assertEqual({"body": 1.0}, result.matching_iou)

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

    def test_threshold_feasible_policy_finds_valid_assignment_legacy_misses(
        self,
    ) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        candidates = (
            _candidate_from_disjoint_counts("x", (100, 55)),
            _candidate_from_disjoint_counts("y", (26, 0)),
        )
        entities = (_entity("a", "ball"), _entity("b", "ball"))

        legacy = match_initial_identities(
            entities=entities,
            reference_masks_by_entity=references,
            candidates=candidates,
            config=_config(threshold=0.25),
        )
        feasible = match_initial_identities(
            entities=entities,
            reference_masks_by_entity=references,
            candidates=candidates,
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertFalse(legacy.success)
        self.assertEqual("initial_match_iou_below_threshold", legacy.failure.code)
        self.assertTrue(feasible.success)
        self.assertEqual({"a": "y", "b": "x"}, feasible.initial_matching)
        self.assertEqual({"a": 0.26, "b": 0.275}, feasible.matching_iou)

    def test_threshold_feasible_policy_reports_hall_conflict_as_init_failure(
        self,
    ) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        result = match_initial_identities(
            entities=(_entity("a", "ball"), _entity("b", "ball")),
            reference_masks_by_entity=references,
            candidates=(
                _candidate_from_disjoint_counts("shared", (100, 100)),
                _candidate_from_disjoint_counts("empty", (0, 0)),
            ),
            config=_config(threshold=0.5, policy="threshold_feasible_v2"),
        )

        self.assertFalse(result.success)
        self.assertEqual("no_feasible_initial_assignment", result.failure.code)
        self.assertEqual({}, result.initial_matching)

    def test_threshold_feasible_policy_rejects_an_all_zero_entity_row(self) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        result = match_initial_identities(
            entities=(_entity("a", "ball"), _entity("b", "ball")),
            reference_masks_by_entity=references,
            candidates=(
                _candidate_from_disjoint_counts("a-full", (100, 0)),
                _candidate_from_disjoint_counts("a-part", (50, 0)),
            ),
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertFalse(result.success)
        self.assertEqual("no_feasible_initial_assignment", result.failure.code)

    def test_threshold_equality_is_a_feasible_edge(self) -> None:
        references = _disjoint_hundred_pixel_masks(("a",))
        result = match_initial_identities(
            entities=(_entity("a", "ball"),),
            reference_masks_by_entity=references,
            candidates=(_candidate_from_disjoint_counts("quarter", (25,)),),
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertTrue(result.success)
        self.assertEqual({"a": "quarter"}, result.initial_matching)
        self.assertEqual({"a": 0.25}, result.matching_iou)

    def test_rectangular_feasible_matching_ignores_only_the_extra_candidate(
        self,
    ) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        candidates = (
            _candidate_from_disjoint_counts("x", (100, 55)),
            _candidate_from_disjoint_counts("y", (26, 0)),
            _candidate_from_disjoint_counts("extra", (0, 0)),
        )
        result = match_initial_identities(
            entities=(_entity("a", "ball"), _entity("b", "ball")),
            reference_masks_by_entity=references,
            candidates=candidates,
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertTrue(result.success)
        self.assertEqual({"a": "y", "b": "x"}, result.initial_matching)
        self.assertEqual(("extra",), result.ignored_candidate_ids)

    def test_feasible_assignment_is_invariant_to_entity_and_candidate_order(
        self,
    ) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        entities = (_entity("a", "ball"), _entity("b", "ball"))
        candidates = (
            _candidate_from_disjoint_counts("x", (100, 55)),
            _candidate_from_disjoint_counts("y", (26, 0)),
        )

        for ordered_entities in (entities, tuple(reversed(entities))):
            for ordered_candidates in (candidates, tuple(reversed(candidates))):
                with self.subTest(
                    entities=tuple(item.entity_id for item in ordered_entities),
                    candidates=tuple(
                        item.candidate_id for item in ordered_candidates
                    ),
                ):
                    result = match_initial_identities(
                        entities=ordered_entities,
                        reference_masks_by_entity=references,
                        candidates=ordered_candidates,
                        config=_config(
                            threshold=0.25,
                            policy="threshold_feasible_v2",
                        ),
                    )
                    self.assertTrue(result.success)
                    self.assertEqual(
                        {"a": "y", "b": "x"}, result.initial_matching
                    )

    def test_ambiguity_uses_second_best_feasible_assignment(self) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b", "c"))
        result = match_initial_identities(
            entities=(
                _entity("a", "ball"),
                _entity("b", "ball"),
                _entity("c", "ball"),
            ),
            reference_masks_by_entity=references,
            candidates=(
                _candidate_from_disjoint_counts("first", (85, 37, 74)),
                _candidate_from_disjoint_counts("second", (68, 5, 63)),
                _candidate_from_disjoint_counts("third", (19, 54, 92)),
            ),
            config=_config(
                threshold=0.25,
                ambiguity_margin=0.01,
                policy="threshold_feasible_v2",
            ),
        )

        self.assertFalse(result.success)
        self.assertEqual("initial_assignment_ambiguous", result.failure.code)
        self.assertAlmostEqual(
            0.00963673783715977,
            result.failure.details["assignment_margin"],
        )

    def test_full_iou_diagnostics_include_candidate_ids_on_success(self) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        result = match_initial_identities(
            entities=(_entity("a", "ball"), _entity("b", "ball")),
            reference_masks_by_entity=references,
            candidates=(
                _candidate_from_disjoint_counts("x", (100, 55)),
                _candidate_from_disjoint_counts("y", (26, 0)),
                _candidate_from_disjoint_counts("extra", (0, 0)),
            ),
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertEqual(
            {
                "ball": {
                    "entity_ids": ("a", "b"),
                    "candidate_ids": ("x", "y", "extra"),
                    "iou_matrix": (
                        (100 / 155, 0.26, 0.0),
                        (0.275, 0.0, 0.0),
                    ),
                }
            },
            result.initial_iou_diagnostics,
        )

    def test_shortage_diagnostics_retain_rows_and_zero_candidate_columns(
        self,
    ) -> None:
        references = _disjoint_hundred_pixel_masks(("a", "b"))
        result = match_initial_identities(
            entities=(_entity("a", "ball"), _entity("b", "ball")),
            reference_masks_by_entity=references,
            candidates=(),
            config=_config(threshold=0.25, policy="threshold_feasible_v2"),
        )

        self.assertFalse(result.success)
        self.assertEqual(
            {
                "ball": {
                    "entity_ids": ("a", "b"),
                    "candidate_ids": (),
                    "iou_matrix": ((), ()),
                }
            },
            result.initial_iou_diagnostics,
        )


class CSTILockedTubeLifecycleTest(unittest.TestCase):
    def _build(
        self,
        frames: list[np.ndarray],
        *,
        patience: int = 3,
    ):
        candidate = _candidate_frames("ball:1", "ball", frames)
        config = CSTIObserverConfig.from_mapping(
            {
                "prompt_groups": [
                    {
                        "id": "ball",
                        "text": "ball",
                        "entity_classes": ["ball"],
                    }
                ],
                "initial_match_iou_threshold": 0.5,
                "initial_match_ambiguity_margin": 0.0,
                "termination_patience": patience,
                "minimum_mask_pixels": 1,
                "minimum_observation_confidence": 0.0,
            }
        )
        initial = InitialIdentityMatch(
            success=True,
            initial_matching={"body": "ball:1"},
            matching_iou={"body": 1.0},
            ignored_candidate_ids=(),
        )
        return build_locked_prediction_tubes(
            entities=(_entity("body", "ball"),),
            candidates=(candidate,),
            initial_match=initial,
            frame_count=len(frames),
            frame_shape=frames[0].shape,
            config=config,
        )

    def test_one_invalid_frame_then_recovery_does_not_terminate(self) -> None:
        valid = _mask(2, 2, 5, 5)
        empty = np.zeros_like(valid)
        result = self._build([valid, empty, valid, valid])

        self.assertIsNone(result.termination_frame_per_subject["body"])
        self.assertFalse(np.any(result.prediction_masks_by_entity["body"][1]))
        self.assertTrue(np.array_equal(valid, result.prediction_masks_by_entity["body"][2]))
        self.assertEqual("VIDEO_END", result.final_state_per_subject["body"])

    def test_two_invalid_frames_then_recovery_does_not_terminate(self) -> None:
        valid = _mask(2, 2, 5, 5)
        empty = np.zeros_like(valid)
        result = self._build([valid, empty, empty, valid, valid])

        self.assertIsNone(result.termination_frame_per_subject["body"])
        self.assertTrue(np.array_equal(valid, result.prediction_masks_by_entity["body"][3]))

    def test_three_invalid_frames_confirm_and_backdate_termination(self) -> None:
        valid = _mask(2, 2, 5, 5)
        empty = np.zeros_like(valid)
        result = self._build([valid, valid, empty, empty, empty, valid])

        self.assertEqual(2, result.termination_frame_per_subject["body"])
        self.assertEqual("TERMINATED", result.final_state_per_subject["body"])
        self.assertFalse(
            np.any(np.stack(result.prediction_masks_by_entity["body"])[2:])
        )

    def test_pending_invalid_tail_stays_empty_without_claiming_termination(self) -> None:
        valid = _mask(2, 2, 5, 5)
        empty = np.zeros_like(valid)
        result = self._build([valid, valid, empty, empty])

        self.assertIsNone(result.termination_frame_per_subject["body"])
        self.assertEqual("VIDEO_END", result.final_state_per_subject["body"])
        self.assertFalse(
            np.any(np.stack(result.prediction_masks_by_entity["body"])[2:])
        )
        self.assertEqual(2, result.pending_invalid_frames_per_subject["body"])

    def test_locked_backend_id_is_not_rematched_when_same_class_positions_swap(self) -> None:
        left = _mask(1, 2, 4, 5)
        right = _mask(8, 2, 11, 5)
        candidates = (
            _candidate_frames("ball:1", "ball", [left, right, right]),
            _candidate_frames("ball:2", "ball", [right, left, left]),
        )
        initial = match_initial_identities(
            entities=(_entity("left", "ball"), _entity("right", "ball")),
            reference_masks_by_entity={"left": left, "right": right},
            candidates=candidates,
            config=_config(policy="threshold_feasible_v2"),
        )

        result = build_locked_prediction_tubes(
            entities=(_entity("left", "ball"), _entity("right", "ball")),
            candidates=candidates,
            initial_match=initial,
            frame_count=3,
            frame_shape=left.shape,
            config=_config(policy="threshold_feasible_v2"),
        )

        self.assertTrue(np.array_equal(right, result.prediction_masks_by_entity["left"][1]))
        self.assertTrue(np.array_equal(left, result.prediction_masks_by_entity["right"][1]))
        self.assertEqual({"left": "ball:1", "right": "ball:2"}, initial.initial_matching)

    def test_validity_checks_shape_area_and_confidence(self) -> None:
        valid = _mask(2, 2, 5, 5)
        config = _config()

        self.assertTrue(
            is_valid_track_observation(
                valid,
                confidence=1.0,
                frame_shape=valid.shape,
                config=config,
            ).valid
        )
        self.assertEqual(
            "mask_empty",
            is_valid_track_observation(
                np.zeros_like(valid),
                confidence=1.0,
                frame_shape=valid.shape,
                config=config,
            ).reason,
        )
        self.assertEqual(
            "mask_shape_invalid",
            is_valid_track_observation(
                np.zeros((2, 2), dtype=bool),
                confidence=1.0,
                frame_shape=valid.shape,
                config=config,
            ).reason,
        )

    def test_incomplete_locked_mapping_is_an_explicit_contract_error(self) -> None:
        valid = _mask(2, 2, 5, 5)
        with self.assertRaisesRegex(CSTIContractError, "cover every entity"):
            build_locked_prediction_tubes(
                entities=(_entity("body", "ball"),),
                candidates=(_candidate_frames("ball:1", "ball", [valid]),),
                initial_match=InitialIdentityMatch(
                    success=True,
                    initial_matching={},
                    matching_iou={},
                    ignored_candidate_ids=("ball:1",),
                ),
                frame_count=1,
                frame_shape=valid.shape,
                config=_config(),
            )


class CSTIObservationAdapterTest(unittest.TestCase):
    def _aligned_input(self) -> tuple[CSTIInput, tuple[EntitySpec, ...]]:
        first = _mask(1, 2, 4, 5)
        second = _mask(7, 1, 11, 6)
        first_tube = tuple(
            np.roll(first, shift=index, axis=1) for index in range(4)
        )
        second_tube = tuple(second.copy() for _ in range(4))
        entities = (_entity("small", "ball"), _entity("large", "block"))
        value = CSTIInput(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0, 1 / 24, 2 / 24, 3 / 24),
            frame_shape=first.shape,
            entities=(
                CSTIEntityTube(
                    entity_id="small",
                    role_id="small",
                    reference_masks=first_tube,
                    prediction_masks=first_tube,
                    matched_prediction_track_ids=("legacy-small",),
                ),
                CSTIEntityTube(
                    entity_id="large",
                    role_id="large",
                    reference_masks=second_tube,
                    prediction_masks=second_tube,
                    matched_prediction_track_ids=("legacy-large",),
                ),
            ),
        )
        return value, entities

    def test_adapter_preserves_reference_contract_and_replaces_only_prediction(self) -> None:
        value, entities = self._aligned_input()
        candidates = (
            _candidate_frames(
                "ball:1",
                "ball",
                [mask.copy() for mask in value.entities[0].reference_masks],
            ),
            _candidate_frames(
                "block:2",
                "block",
                [mask.copy() for mask in value.entities[1].reference_masks],
            ),
        )

        observed = observe_csti_tubes(
            aligned_input=value,
            entities=entities,
            candidates=candidates,
            config=_config(),
        )

        self.assertTrue(observed.evaluator_init_success)
        self.assertEqual(value.times_s, observed.csti_input.times_s)
        self.assertEqual(value.frame_shape, observed.csti_input.frame_shape)
        self.assertIs(
            value.reference_capability,
            observed.csti_input.reference_capability,
        )
        self.assertEqual(
            [entity.reference_masks for entity in value.entities],
            [entity.reference_masks for entity in observed.csti_input.entities],
        )
        self.assertEqual(
            [("ball:1",), ("block:2",)],
            [
                entity.matched_prediction_track_ids
                for entity in observed.csti_input.entities
            ],
        )

    def test_adapter_keeps_improved_csti_scores_bit_identical_for_same_tubes(self) -> None:
        value, entities = self._aligned_input()
        candidates = (
            _candidate_frames(
                "ball:1", "ball", list(value.entities[0].prediction_masks)
            ),
            _candidate_frames(
                "block:2", "block", list(value.entities[1].prediction_masks)
            ),
        )
        observed = observe_csti_tubes(
            aligned_input=value,
            entities=entities,
            candidates=candidates,
            config=_config(),
        )
        expected_entities = (("small", "small"), ("large", "large"))

        before = evaluate_csti(
            value,
            expected_entities=expected_entities,
            config=_metric_config(),
        )
        after = evaluate_csti(
            observed.csti_input,
            expected_entities=expected_entities,
            config=_metric_config(),
        )

        self.assertEqual(before["score"], after["score"])
        self.assertEqual(
            [item["score"] for item in before["objects"]],
            [item["score"] for item in after["objects"]],
        )

    def test_result_decoration_exposes_subject_macro_and_identity_audit(self) -> None:
        value, entities = self._aligned_input()
        candidates = (
            _candidate_frames(
                "ball:1", "ball", list(value.entities[0].prediction_masks)
            ),
            _candidate_frames(
                "block:2", "block", list(value.entities[1].prediction_masks)
            ),
        )
        observed = observe_csti_tubes(
            aligned_input=value,
            entities=entities,
            candidates=candidates,
            config=_config(),
        )
        raw = evaluate_csti(
            observed.csti_input,
            expected_entities=(("small", "small"), ("large", "large")),
            config=_metric_config(),
        )

        result = decorate_csti_metric(raw, observation=observed)

        self.assertEqual(result["score"], result["csti_video"])
        self.assertEqual(
            result["score"],
            np.mean(list(result["csti_per_subject"].values())),
        )
        self.assertEqual(2, result["subject_count"])
        self.assertTrue(result["evaluator_init_success"])
        self.assertIsNone(result["evaluator_init_failure_reason"])
        self.assertEqual(
            {"small": "ball:1", "large": "block:2"},
            result["initial_matching"],
        )
        self.assertEqual(
            {
                "ball": {
                    "entity_ids": ("small",),
                    "candidate_ids": ("ball:1",),
                    "iou_matrix": ((1.0,),),
                },
                "block": {
                    "entity_ids": ("large",),
                    "candidate_ids": ("block:2",),
                    "iou_matrix": ((1.0,),),
                },
            },
            result["observer"]["initial_iou_diagnostics"],
        )

    def test_initialization_failure_metric_is_null_and_auditable_not_zero(self) -> None:
        value, entities = self._aligned_input()
        observed = observe_csti_tubes(
            aligned_input=value,
            entities=entities,
            candidates=(
                _candidate_frames(
                    "ball:1", "ball", list(value.entities[0].reference_masks)
                ),
            ),
            config=_config(),
        )

        result = evaluator_init_failure_metric(
            observation=observed,
            expected_entities=(("small", "small"), ("large", "large")),
            config=_metric_config(),
        )

        self.assertEqual("evaluator_init_failure", result["status"])
        self.assertIsNone(result["score"])
        self.assertFalse(result["evaluator_init_success"])
        self.assertEqual(2, result["subject_count"])
        self.assertEqual(
            "insufficient_semantic_candidates",
            result["evaluator_init_failure_reason"]["code"],
        )
        self.assertEqual(
            {
                "entity_ids": ("large",),
                "candidate_ids": (),
                "iou_matrix": ((),),
            },
            result["observer"]["initial_iou_diagnostics"]["block"],
        )


if __name__ == "__main__":
    unittest.main()
