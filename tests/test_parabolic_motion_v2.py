from __future__ import annotations

import copy
from contextlib import ExitStack
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, evaluate_csti
from physbench.evaluation.common.entities import materialize_entity_manifest
from physbench.evaluation.common.errors import ReferenceAnalysisError
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.scenes.parabolic_motion.evaluator import (
    BallCandidate,
    ParabolicObservation,
    _mask_histogram,
    observe_projectile,
)
from physbench.evaluation.scenes.parabolic_motion.v2_evaluator import (
    ParabolicMotionCaseEvaluatorV2,
)
from physbench.io import write_json


_CSTI_MAPPING = {
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


def _frames() -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in range(12):
        frame = np.full((160, 240, 3), 224, dtype=np.uint8)
        x = 218 - 12 * index
        y = 28 + round(0.65 * index * index)
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (x, y), 8, (25, 25, 25), -1)
            cv2.circle(frame, (x - 2, y - 2), 2, (210, 210, 210), -1)
        output.append(frame)
    return output


def _transform() -> dict[str, object]:
    return {
        "policy": "reference_content_crop_resize_no_pad",
        "crop_xywh": [0, 0, 240, 160],
        "scale": 1.0,
        "source_size": [240, 160],
        "target_size": [240, 160],
        "padding": None,
    }


def _config() -> dict:
    config = copy.deepcopy(
        load_evaluation_protocol("scene_default_v12")["scenes"]
        ["parabolic_motion"]
    )
    config["type"] = "parabolic_motion_state_v2"
    config["reference_observation_policy"] = (
        "frozen_dataset_subject_identity"
    )
    config["subject_identity"] = {
        "anchor_policy": "first_frame_subject_mask_manifest_v1",
        "failure_policy": "fail_closed_v1",
    }
    config["open_world_observation"].update(
        {
            "minimum_assignment_cost_margin": 0.15,
            "latch_identity_loss": True,
        }
    )
    config["scoring"].update(
        {
            "composition": "strict_multiplicative_v2",
            "minimum_binding_score": 0.55,
            "minimum_binding_appearance": 0.20,
        }
    )
    config["general_metrics"] = {"csti": _CSTI_MAPPING}
    return config


def _write_case(root: Path) -> dict[str, object]:
    mask = np.zeros((160, 240), dtype=np.uint8)
    cv2.circle(mask, (218, 28), 8, 1, -1)
    ys, xs = np.where(mask > 0)
    manifest_relative = "case/canonical/masks/manifest.json"
    npz_relative = "case/canonical/masks/01.npz"
    npz_path = root / npz_relative
    npz_path.parent.mkdir(parents=True)
    np.savez_compressed(
        npz_path,
        masks=mask[None, ...],
        mask_ids=np.asarray(["01"]),
        object_ids=np.asarray(["projectile_ball"]),
        frame_index=np.asarray(0, dtype=np.int64),
    )
    write_json(
        root / manifest_relative,
        {
            "schema_version": "1.2",
            "case_id": "synthetic_parabolic_v2",
            "scene_id": "parabolic_motion",
            "frame_index": 0,
            "frame_scope": "first_frame_only",
            "source_first_frame": "case/canonical/first_frame.png",
            "image_shape_hw": [160, 240],
            "instances": [
                {
                    "mask_id": "01",
                    "object_id": "object_1",
                    "entity_class": "ball",
                    "npz_asset": npz_relative,
                    "area_pixels": int(xs.size),
                    "bbox_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                }
            ],
            "storage": {
                "model": {
                    "array_key": "masks",
                    "layout": "1HW",
                    "dtype": "uint8",
                    "values": [0, 1],
                }
            },
        },
    )
    return {
        "case_id": "synthetic_parabolic_v2",
        "scene_id": "parabolic_motion",
        "physics": {
            "ball_mass": {"value": 0.014, "unit": "kg", "annotated": True},
            "ball_radius": {
                "value": 0.0075,
                "unit": "m",
                "annotated": True,
            },
            "initial_horizontal_velocity": {
                "value": 1.0,
                "unit": "m/s",
                "annotated": True,
            },
            "launch_height": {
                "value": 0.77,
                "unit": "m",
                "annotated": True,
            },
        },
        "appearance": {
            "ball_material": "steel",
            "ball_size_class": "small",
        },
        "assets": {
            "reference_video": "case/canonical/reference.mp4",
            "first_frame": "case/canonical/first_frame.png",
            "first_frame_mask_manifest": manifest_relative,
        },
    }


class ParabolicMotionV2Tests(unittest.TestCase):
    @staticmethod
    def _video(frames: list[np.ndarray]) -> SimpleNamespace:
        return SimpleNamespace(
            frames=frames,
            available=np.ones(len(frames), dtype=bool),
            spatial_transform=_transform(),
        )

    @staticmethod
    def _request(
        root: Path,
        case: dict[str, object],
        config: dict,
    ) -> CaseEvaluationRequest:
        return CaseEvaluationRequest(
            job={"job_id": "parabolic_v2_test"},
            case=case,
            case_catalog={case["case_id"]: case},
            prediction={"status": "complete", "video_path": "mock.mp4"},
            asset_root=root,
            artifact_dir=root / "artifacts",
            evaluator_config=config,
        )

    @staticmethod
    def _artifact_patches() -> ExitStack:
        stack = ExitStack()
        prefix = "physbench.evaluation.scenes.parabolic_motion.evaluator."
        for name in ("write_rows_csv", "write_json", "save_iou_curve"):
            stack.enter_context(patch(prefix + name))
        stack.enter_context(patch(prefix + "save_series_comparison"))
        stack.enter_context(
            patch(prefix + "write_parabolic_visualization", return_value={})
        )
        return stack

    def test_prediction_discovers_subject_without_transforming_gt_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _write_case(root)
            config = _config()
            evaluator = ParabolicMotionCaseEvaluatorV2(config)
            frames = _frames()
            prediction_transform = {
                **_transform(),
                "source_size": [480, 832],
                "crop_xywh": [32, 0, 416, 832],
                "scale": 240 / 416,
            }

            observation = evaluator._observe_projectile(
                self._request(root, case, config),
                entity_id="projectile_ball",
                frames=frames,
                available=np.ones(len(frames), dtype=bool),
                spatial_transform=prediction_transform,
                reference=False,
            )

        self.assertIsNotNone(observation.seed)
        self.assertNotEqual(
            "frozen_subject_anchor_v2", observation.seed.source
        )
        self.assertEqual({}, observation.diagnostics["seed_provenance"])

    def test_projected_boundary_exit_is_not_identity_uncertainty(self) -> None:
        frames: list[np.ndarray] = []
        for x in (70, 40, 10, None, None):
            frame = np.full((160, 240, 3), 224, dtype=np.uint8)
            if x is not None:
                cv2.circle(frame, (x, 80), 8, (25, 25, 25), -1)
            frames.append(frame)
        seed_mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        cv2.circle(seed_mask, (70, 80), 8, 255, -1)
        seed = BallCandidate(
            xy=np.asarray([70.0, 80.0]),
            radius=8.0,
            mask=seed_mask,
            histogram=_mask_histogram(frames[0], seed_mask),
            contrast=199.0,
            source="frozen_subject_anchor_v2",
        )

        observation = observe_projectile(
            frames,
            available=np.ones(len(frames), dtype=bool),
            config=_config()["open_world_observation"],
            seed_override=seed,
        )

        self.assertEqual("exited", observation.diagnostics["identity_state"])
        self.assertEqual(
            "projected_outside_canvas",
            observation.diagnostics["identity_termination_reason"],
        )

    def test_exact_reference_is_unit_expert_and_csti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _write_case(root)
            config = _config()
            evaluator = ParabolicMotionCaseEvaluatorV2(config)
            frames = _frames()
            video = self._video(frames)
            times_s = (np.arange(len(frames)) / 24.0).tolist()
            with self._artifact_patches():
                analysis = evaluator.analyze(
                    self._request(root, case, config),
                    times_s=times_s,
                    reference_video=video,
                    prediction_video=video,
                )

        self.assertEqual(1.0, analysis.score)
        self.assertEqual(
            "strict_multiplicative_v2",
            analysis.metrics["parabolic_motion_state_similarity"]
            ["composition"],
        )
        manifest = materialize_entity_manifest(case)
        csti = evaluate_csti(
            analysis.csti_input,
            expected_entities=tuple(
                (entity.entity_id, entity.role_id)
                for entity in manifest.entities
            ),
            config=CSTIConfig.from_mapping(_CSTI_MAPPING),
        )
        self.assertEqual(1.0, csti["score"])

    def test_missing_frozen_anchor_is_reference_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _write_case(root)
            del case["assets"]["first_frame_mask_manifest"]
            config = _config()
            evaluator = ParabolicMotionCaseEvaluatorV2(config)
            frames = _frames()
            video = self._video(frames)

            with self.assertRaises(ReferenceAnalysisError) as captured:
                with self._artifact_patches():
                    evaluator.analyze(
                        self._request(root, case, config),
                        times_s=(np.arange(len(frames)) / 24.0).tolist(),
                        reference_video=video,
                        prediction_video=video,
                    )

        self.assertEqual(
            "reference_projectile_subject_manifest_missing",
            captured.exception.code,
        )

    def test_blank_anchor_region_rejects_prediction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _write_case(root)
            config = _config()
            evaluator = ParabolicMotionCaseEvaluatorV2(config)
            reference_frames = _frames()
            prediction_frames = [
                np.full_like(frame, 224) for frame in reference_frames
            ]
            with self._artifact_patches():
                analysis = evaluator.analyze(
                    self._request(root, case, config),
                    times_s=(
                        np.arange(len(reference_frames)) / 24.0
                    ).tolist(),
                    reference_video=self._video(reference_frames),
                    prediction_video=self._video(prediction_frames),
                )

        self.assertEqual(0.0, analysis.score)
        self.assertFalse(
            analysis.metrics["physical_subject_similarity"]["components"]
            ["frame_zero_binding"]["accepted"]
        )
        self.assertIn(
            "prediction_projectile_entity_missing_v2",
            analysis.quality["degradation_codes"],
        )

    def test_terminated_identity_exposes_stable_v2_degradation(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        cv2.circle(mask, (5, 5), 2, 255, -1)
        histogram = np.zeros(24, dtype=np.float64)
        histogram[0] = 1.0
        seed = BallCandidate(
            xy=np.asarray([5.0, 5.0]),
            radius=2.0,
            mask=mask,
            histogram=histogram,
            contrast=20.0,
            source="fixture",
        )
        observation = ParabolicObservation(
            xy=np.asarray([[5.0, 5.0]]),
            radii=np.asarray([2.0]),
            observed=np.asarray([True]),
            interpolated=np.asarray([False]),
            masks=(mask,),
            histograms=(histogram,),
            cardinality=np.asarray([1]),
            seed=seed,
            diagnostics={
                "identity_state": "terminated_uncertain",
                "identity_termination_frame": 1,
                "identity_termination_reason": "ambiguous_assignment",
            },
        )
        evaluator = ParabolicMotionCaseEvaluatorV2(_config())

        failures = evaluator._projectile_identity_failures(
            observation,
            binding={"accepted": True},
        )

        self.assertEqual(
            ["prediction_projectile_identity_uncertain_v2"],
            [failure["code"] for failure in failures],
        )
        self.assertIn("frame 1", failures[0]["reason"])
        self.assertIn("ambiguous_assignment", failures[0]["reason"])


@unittest.skipUnless(
    (
        Path(__file__).resolve().parents[1]
        / "scripts/regress_parabolic_identity_v2.py"
    ).is_file(),
    "clean release excludes the model-specific parabolic regression script",
)
class ParabolicV2RealRegressionContractTests(unittest.TestCase):
    def test_reuse_requires_matching_evaluator_and_video_hashes(self) -> None:
        module = importlib.import_module("scripts.regress_parabolic_identity_v2")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_path = root / "prediction.mp4"
            reference_path = root / "reference.mp4"
            prediction_path.write_bytes(b"prediction-v1")
            reference_path.write_bytes(b"reference-v1")
            artifact_dir = root / "cases" / "job"
            write_json(
                artifact_dir / "result.json",
                {
                    "job_id": "job",
                    "case_id": "case",
                    "scene_id": "parabolic_motion",
                    "status": "evaluated",
                    "regression_variant": "false_high_negative",
                    "evaluator": {"fingerprint": "evaluator-v1"},
                    "provenance": {
                        "prediction_video_sha256": module.sha256_file(
                            prediction_path
                        ),
                        "reference_video_sha256": module.sha256_file(
                            reference_path
                        ),
                    },
                },
            )

            reusable = module._load_reusable_result(
                artifact_dir=artifact_dir,
                job_id="job",
                case_id="case",
                variant="false_high_negative",
                evaluator_fingerprint="evaluator-v1",
                prediction_path=prediction_path,
                reference_path=reference_path,
            )
            self.assertIsNotNone(reusable)
            self.assertIsNone(
                module._load_reusable_result(
                    artifact_dir=artifact_dir,
                    job_id="job",
                    case_id="case",
                    variant="false_high_negative",
                    evaluator_fingerprint="evaluator-v2",
                    prediction_path=prediction_path,
                    reference_path=reference_path,
                )
            )
            prediction_path.write_bytes(b"prediction-v2")
            self.assertIsNone(
                module._load_reusable_result(
                    artifact_dir=artifact_dir,
                    job_id="job",
                    case_id="case",
                    variant="false_high_negative",
                    evaluator_fingerprint="evaluator-v1",
                    prediction_path=prediction_path,
                    reference_path=reference_path,
                )
            )

    def test_script_freezes_reviewed_negatives_and_positive_assets(self) -> None:
        module = importlib.import_module("scripts.regress_parabolic_identity_v2")
        expected_negatives = {
            "parabolic_img_0592": (
                0.5656459893608466,
                "seven_scene_symbol_value_finetune_eval_v13__"
                "parabolic_img_0592__seed000042.mp4",
            ),
            "parabolic_img_0623": (
                0.5270753640342365,
                "seven_scene_symbol_value_finetune_eval_v13__"
                "parabolic_img_0623__seed000042.mp4",
            ),
        }
        self.assertEqual(set(expected_negatives), set(module.REGRESSION_CASES))
        dataset = load_dataset(LATEST_DATASET, check_assets=True)
        cases = {case["case_id"]: case for case in dataset.cases}
        for case_id, (old_score, filename) in expected_negatives.items():
            with self.subTest(case_id=case_id):
                spec = module.REGRESSION_CASES[case_id]
                self.assertEqual(old_score, spec["old_expert_score"])
                self.assertEqual(0.25, spec["maximum_score"])
                self.assertEqual(filename, Path(spec["prediction_path"]).name)
                self.assertTrue(Path(spec["prediction_path"]).is_file())
                manifest = (
                    dataset.asset_root
                    / cases[case_id]["assets"]["first_frame_mask_manifest"]
                )
                self.assertTrue(manifest.is_file())
        self.assertEqual(
            {"parabolic_img_0577"}, set(module.REVIEWED_GOOD_CONTROLS)
        )
        positive = module.REVIEWED_GOOD_CONTROLS["parabolic_img_0577"]
        self.assertEqual(0.8473541946432471, positive["old_expert_score"])
        self.assertEqual(0.35, positive["minimum_expert_score"])


if __name__ == "__main__":
    unittest.main()
