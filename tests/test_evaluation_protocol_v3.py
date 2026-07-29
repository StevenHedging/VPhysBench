from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from physbench.evaluation.common.errors import SceneAnalysisError
from physbench.evaluation.common.subject import compare_subjects
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.task_evaluator import evaluate_task


class EvaluationProtocolV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_evaluation_protocol("scene_default_v1")
        cls.v2 = load_evaluation_protocol("scene_default_v2")
        cls.v3 = load_evaluation_protocol("scene_default_v3")

    @staticmethod
    def _subject_fixture(
        *,
        color: tuple[int, int, int] = (20, 80, 220),
        shift_x: int = 0,
        empty: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in range(4):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            mask = np.zeros((64, 96), dtype=np.uint8)
            if not empty:
                x0 = 20 + index * 4 + shift_x
                frame[22:38, x0 : x0 + 14] = color
                mask[22:38, x0 : x0 + 14] = 255
            frames.append(frame)
            masks.append(mask)
        return frames, masks

    @staticmethod
    def _subject_config() -> dict:
        return {
            "minimum_observed_pixels": 4,
            "position_distance_scale": 0.08,
            "canonical_crop_size": 64,
            "boundary_tolerance_px": 2,
            "weights": {
                "position": 0.5,
                "shape": 0.2,
                "appearance": 0.3,
            },
        }

    def test_protocol_identity_and_evaluator_versions_are_isolated(self) -> None:
        self.assertEqual("scene_default_v3", self.v3["protocol_id"])
        self.assertEqual(
            "84733a984480b4ec8a66b2d31976bd72ddca0a284f525b9d66525436e88dfe88",
            self.v2["fingerprint"],
        )
        self.assertNotEqual(self.v2["fingerprint"], self.v3["fingerprint"])
        registry = SceneEvaluatorRegistry(self.v3)
        for scene_id in self.v3["scenes"]:
            description = registry.resolve(scene_id).describe()
            self.assertEqual("1.3", description["version"])
            self.assertEqual(
                "scene_subject_state_similarity",
                description["primary_score"],
            )

    def test_identical_subject_position_shape_and_appearance_score_one(
        self,
    ) -> None:
        frames, masks = self._subject_fixture()
        result = compare_subjects(
            reference_frames=frames,
            prediction_frames=[frame.copy() for frame in frames],
            reference_masks=masks,
            prediction_masks=[mask.copy() for mask in masks],
            reference_mode="same_case_reference",
            config=self._subject_config(),
        )
        self.assertAlmostEqual(1.0, result.score, places=7)
        self.assertAlmostEqual(1.0, result.components["position"], places=7)
        self.assertAlmostEqual(1.0, result.components["shape"], places=7)
        self.assertAlmostEqual(1.0, result.components["appearance"], places=7)

    def test_position_and_appearance_changes_are_independently_sensitive(
        self,
    ) -> None:
        reference_frames, reference_masks = self._subject_fixture()
        shifted_frames, shifted_masks = self._subject_fixture(shift_x=20)
        shifted = compare_subjects(
            reference_frames=reference_frames,
            prediction_frames=shifted_frames,
            reference_masks=reference_masks,
            prediction_masks=shifted_masks,
            reference_mode="same_case_reference",
            config=self._subject_config(),
        )
        recolored_frames, recolored_masks = self._subject_fixture(
            color=(220, 60, 20)
        )
        recolored = compare_subjects(
            reference_frames=reference_frames,
            prediction_frames=recolored_frames,
            reference_masks=reference_masks,
            prediction_masks=recolored_masks,
            reference_mode="same_case_reference",
            config=self._subject_config(),
        )
        self.assertLess(shifted.components["position"], 1.0)
        self.assertAlmostEqual(1.0, shifted.components["shape"], places=7)
        self.assertLess(recolored.components["appearance"], 1.0)
        self.assertAlmostEqual(1.0, recolored.components["position"], places=7)
        self.assertLess(shifted.score, 1.0)
        self.assertLess(recolored.score, 1.0)

    def test_missing_prediction_subject_is_a_finite_zero_not_failure(
        self,
    ) -> None:
        reference_frames, reference_masks = self._subject_fixture()
        prediction_frames, prediction_masks = self._subject_fixture(empty=True)
        result = compare_subjects(
            reference_frames=reference_frames,
            prediction_frames=prediction_frames,
            reference_masks=reference_masks,
            prediction_masks=prediction_masks,
            reference_mode="same_case_reference",
            config=self._subject_config(),
        )
        self.assertEqual(0.0, result.score)
        self.assertEqual(0.0, result.prediction_observed_ratio)
        self.assertTrue(
            all(row["subject"] == 0.0 for row in result.per_frame)
        )

    def test_parent_reference_does_not_score_parent_appearance(self) -> None:
        parent_frames, parent_masks = self._subject_fixture(
            color=(255, 255, 255)
        )
        prediction_frames, prediction_masks = self._subject_fixture(
            color=(5, 120, 220)
        )
        result = compare_subjects(
            reference_frames=parent_frames,
            prediction_frames=prediction_frames,
            reference_masks=parent_masks,
            prediction_masks=prediction_masks,
            reference_mode="parent_physics_reference",
            config=self._subject_config(),
        )
        self.assertAlmostEqual(1.0, result.score, places=7)
        self.assertNotIn("position", result.components)
        self.assertEqual(
            "prediction_frame_zero_appearance_fallback",
            result.comparable_reference,
        )

    def test_parent_reference_scores_case_condition_not_parent_pixels(
        self,
    ) -> None:
        parent_frames, parent_masks = self._subject_fixture(
            color=(255, 255, 255)
        )
        prediction_frames, prediction_masks = self._subject_fixture(
            color=(5, 120, 220)
        )
        matching_condition = prediction_frames[0].copy()
        matching_mask = prediction_masks[0].copy()
        matching = compare_subjects(
            reference_frames=parent_frames,
            prediction_frames=prediction_frames,
            reference_masks=parent_masks,
            prediction_masks=prediction_masks,
            reference_mode="parent_physics_reference",
            config=self._subject_config(),
            condition_frame=matching_condition,
            condition_mask=matching_mask,
        )
        mismatching_condition = matching_condition.copy()
        mismatching_condition[matching_mask > 0] = (220, 20, 20)
        mismatching = compare_subjects(
            reference_frames=parent_frames,
            prediction_frames=prediction_frames,
            reference_masks=parent_masks,
            prediction_masks=prediction_masks,
            reference_mode="parent_physics_reference",
            config=self._subject_config(),
            condition_frame=mismatching_condition,
            condition_mask=matching_mask,
        )
        self.assertAlmostEqual(1.0, matching.score, places=7)
        self.assertLess(mismatching.score, matching.score)
        self.assertEqual(
            "case_condition_first_frame_appearance_only",
            matching.comparable_reference,
        )

    def test_unobservable_reference_remains_reference_failure(self) -> None:
        reference_frames, reference_masks = self._subject_fixture(empty=True)
        prediction_frames, prediction_masks = self._subject_fixture()
        with self.assertRaises(SceneAnalysisError) as raised:
            compare_subjects(
                reference_frames=reference_frames,
                prediction_frames=prediction_frames,
                reference_masks=reference_masks,
                prediction_masks=prediction_masks,
                reference_mode="same_case_reference",
                config=self._subject_config(),
            )
        self.assertEqual("reference_subject_unobserved", raised.exception.code)

    def test_missing_prediction_record_scores_zero_with_full_coverage(
        self,
    ) -> None:
        plan = {
            "task_id": "robust-test",
            "family": "direct_eval",
            "scene_ids": ["pendulum"],
            "jobs": [
                {
                    "job_id": "job-a",
                    "case_id": "case-a",
                    "scene_id": "pendulum",
                    "evaluation_partition": "group_1",
                    "seed": 42,
                }
            ],
        }
        cases = [
            {
                "case_id": "case-a",
                "scene_id": "pendulum",
                "has_real_reference_video": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            results, summary = evaluate_task(
                plan=plan,
                cases=cases,
                predictions=[],
                asset_root=temporary,
                protocol=self.v3,
                output_dir=Path(temporary) / "evaluation",
            )
        self.assertEqual("evaluated", results[0]["status"])
        self.assertEqual(0.0, results[0]["score"])
        self.assertTrue(results[0]["quality"]["degraded"])
        self.assertEqual(1.0, summary["coverage"])
        self.assertEqual(0.0, summary["score"])
        self.assertEqual(
            1, summary["robustness"]["degraded_evaluated_jobs"]
        )
        self.assertEqual(0, summary["robustness"]["evaluator_error_jobs"])


if __name__ == "__main__":
    unittest.main()
