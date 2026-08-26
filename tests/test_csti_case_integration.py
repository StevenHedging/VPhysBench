from __future__ import annotations

import tempfile
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from physbench.evaluation.common.base import ReferenceCaseEvaluator, SceneAnalysis
from physbench.evaluation.common.csti import (
    CSTIEntityTube,
    CSTIInput,
    SemanticCandidateTube,
)
from physbench.evaluation.common.entities import ReferenceCapability
from physbench.evaluation.common.errors import (
    ReferenceAnalysisError,
    SceneAnalysisError,
)
from physbench.evaluation.contracts import CaseEvaluationRequest
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.task_evaluator import aggregate_task_results


FIXED_CSTI = {
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


def evaluator_config(
    *, csti: bool = True, csti_observer: bool = False
) -> dict[str, object]:
    value: dict[str, object] = {
        "type": "fake_v1",
        "evaluator_contract": "robust_subject_v3",
        "timeline": {
            "policy": "fixed_reference_cap_v1",
            "fps": 1.0,
            "maximum_duration_s": 1.0,
            "minimum_duration_s": 0.0,
            "minimum_source_fps": 1.0,
        },
        "spatial": {
            "policy": "preserve_aspect_ratio_letterbox",
            "width": 4,
            "height": 3,
            "pad_value": 0,
        },
    }
    if csti:
        value["general_metrics"] = {"csti": FIXED_CSTI}
    if csti_observer:
        value["csti_observer"] = {
            "prompt_groups": [
                {
                    "id": "subject",
                    "text": "rigid body",
                    "entity_classes": ["rigid_body"],
                }
            ],
            "initial_match_iou_threshold": 0.5,
            "initial_match_ambiguity_margin": 0.0,
            "termination_patience": 3,
            "minimum_mask_pixels": 1,
            "minimum_observation_confidence": 0.0,
            "segmenter": {},
            "debug_outputs": False,
        }
    return value


class _FakeTextSegmenter:
    def __init__(self, *, initialize: bool = True) -> None:
        self.initialize = initialize
        self.calls = 0

    def segment(self, frames, prompt_groups):
        self.calls += 1
        if not self.initialize:
            return ()
        frame_count = len(frames)
        masks = np.ones((frame_count, 3, 4), dtype=bool)
        return (
            SemanticCandidateTube(
                candidate_id="subject:7",
                prompt_group_id="subject",
                backend_object_id=7,
                masks=masks,
                boxes_xywh=np.zeros((frame_count, 4), dtype=np.float32),
                confidences=np.ones(frame_count, dtype=np.float32),
            ),
        )

    def describe(self):
        return {"backend": "fake_sam31_text"}


class _FakeCSTIEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "fake_csti"
    evaluator_version = "1.0"
    scene_id = "custom_scene"
    primary_score = "expert"

    def analyze(self, request, *, times_s, reference_video, prediction_video):
        if request.case.get("test_scene_prediction_failure", False):
            raise SceneAnalysisError(
                "synthetic_prediction_observation_failure",
                "synthetic scene observer failed",
            )
        masks = tuple(np.ones((3, 4), dtype=bool) for _ in times_s)
        capability = ReferenceCapability(
            request.case.get("test_reference_capability", "same_case_gt")
        )
        csti_input = None
        if not request.case.get("test_missing_csti_input", False):
            csti_input = CSTIInput(
                reference_capability=capability,
                times_s=tuple(times_s),
                frame_shape=(3, 4),
                entities=(
                    CSTIEntityTube(
                        "body",
                        "participant",
                        masks,
                        masks,
                        ("track_body",),
                    ),
                ),
            )
        return SceneAnalysis(
            score=0.625,
            metrics={"expert": {"score": 0.625}},
            quality={},
            csti_input=csti_input,
        )


class CSTICaseIntegrationTest(unittest.TestCase):
    def test_enabled_config_is_parsed_once(self) -> None:
        evaluator = _FakeCSTIEvaluator(evaluator_config())

        self.assertTrue(evaluator.csti_enabled)
        self.assertEqual("exact_full_tube_edt", evaluator.csti_config.algorithm)

    def test_evaluate_attaches_csti_without_changing_expert_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            evaluator = _FakeCSTIEvaluator(evaluator_config())

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.625, result.score)
        self.assertEqual(0.625, result.metrics["expert"]["score"])
        self.assertEqual(1.0, result.metrics["csti"]["score"])
        self.assertEqual("evaluated", result.metrics["csti"]["status"])
        self.assertNotIn("csti_input", result.to_dict())
        self.assertEqual("exact_full_tube_edt", result.evaluator["general_metrics"]["csti"]["algorithm"])
        self.assertIn("entity_manifest_digest", result.provenance["csti"])

    def test_disabled_config_preserves_existing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            evaluator = _FakeCSTIEvaluator(evaluator_config(csti=False))

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual({"expert": {"score": 0.625}}, result.metrics)
        self.assertNotIn("general_metrics", result.evaluator)

    def test_v10_v11_synthetic_case_preserves_expert_result_and_adds_only_csti_dimension(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            v10 = self._evaluate(
                _FakeCSTIEvaluator(evaluator_config(csti=False)),
                request,
                reference_path,
            )
            v11 = self._evaluate(
                _FakeCSTIEvaluator(evaluator_config(csti=True)),
                request,
                reference_path,
            )

        self.assertEqual(v10.status, v11.status)
        self.assertEqual(v10.score, v11.score)
        self.assertEqual(v10.quality, v11.quality)
        self.assertEqual(v10.metrics["expert"], v11.metrics["expert"])
        self.assertNotIn("csti", v10.metrics)
        self.assertIn("csti", v11.metrics)

        plan = {
            "task_id": "compatibility_audit",
            "family": "direct_eval",
            "scene_ids": ["custom_scene"],
        }
        v10_record = {
            **v10.to_dict(),
            "evaluation_partition": "test",
        }
        v11_record = {
            **v11.to_dict(),
            "evaluation_partition": "test",
        }
        v10_task = aggregate_task_results(
            plan=plan,
            case_results=[v10_record],
        )
        v11_task = aggregate_task_results(
            plan=plan,
            case_results=[v11_record],
            general_metrics={"csti": FIXED_CSTI},
        )

        self.assertEqual(v10_task["status"], v11_task["status"])
        self.assertEqual(v10_task["score"], v11_task["score"])
        self.assertEqual(v10_task["by_scene"], v11_task["by_scene"])
        self.assertNotIn("dimensions", v10_task)
        self.assertEqual(
            v11_task["score"],
            v11_task["dimensions"]["expert"]["score"],
        )
        self.assertEqual(1.0, v11_task["dimensions"]["csti"]["score"])

    def test_physics_parent_is_explicitly_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(
                root,
                reference_capability=ReferenceCapability.PHYSICS_PARENT,
            )
            evaluator = _FakeCSTIEvaluator(evaluator_config())

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.625, result.score)
        self.assertEqual("not_applicable", result.metrics["csti"]["status"])
        self.assertIsNone(result.metrics["csti"]["score"])
        self.assertEqual("csti_requires_same_case_gt", result.metrics["csti"]["reason_code"])

    def test_missing_or_mismatched_same_case_input_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            evaluator = _FakeCSTIEvaluator(evaluator_config())

            request.case["test_missing_csti_input"] = True
            missing = self._evaluate(evaluator, request, reference_path)
            request.case.pop("test_missing_csti_input")
            request.case["test_reference_capability"] = "physics_parent"
            mismatched = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("error", missing.status)
        self.assertEqual("csti_input_missing", missing.reason_code)
        self.assertTrue(missing.reason.startswith("CSTI contract failed:"))
        self.assertEqual("error", mismatched.status)
        self.assertEqual("csti_reference_capability_mismatch", mismatched.reason_code)

    def test_prediction_degradation_includes_manifest_complete_csti_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, _ = self._request(root, prediction=False)
            evaluator = _FakeCSTIEvaluator(evaluator_config())

            result = evaluator.evaluate(request)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual(0.0, result.metrics["csti"]["score"])
        self.assertEqual(["body"], [item["entity_id"] for item in result.metrics["csti"]["objects"]])
        self.assertEqual("prediction_record_missing", result.metrics["csti"]["degradation"]["code"])

    def test_registry_injects_general_metrics_without_mutating_protocol(self) -> None:
        protocol = {
            "general_metrics": {"csti": FIXED_CSTI},
            "scenes": {"custom_scene": {"type": "unsupported"}},
        }
        original = copy.deepcopy(protocol)

        evaluator = SceneEvaluatorRegistry(protocol).resolve("custom_scene")

        self.assertEqual({"csti": FIXED_CSTI}, evaluator.config["general_metrics"])
        self.assertEqual(original, protocol)

    def test_scene_analysis_carries_in_memory_csti_input(self) -> None:
        masks = (np.ones((3, 4), dtype=bool),)
        value = CSTIInput(
            reference_capability=ReferenceCapability.SAME_CASE_GT,
            times_s=(0.0,),
            frame_shape=(3, 4),
            entities=(CSTIEntityTube("body", "participant", masks, masks),),
        )

        analysis = SceneAnalysis(
            score=0.625,
            metrics={"expert": {"score": 0.625}},
            quality={},
            csti_input=value,
        )

        self.assertIs(value, analysis.csti_input)

    def test_text_observer_replaces_csti_prediction_without_changing_expert_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            segmenter = _FakeTextSegmenter()
            evaluator = _FakeCSTIEvaluator(
                evaluator_config(csti_observer=True),
                csti_segmenter=segmenter,
            )

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.625, result.score)
        self.assertEqual(1.0, result.metrics["csti"]["score"])
        self.assertEqual(
            {"body": "subject:7"},
            result.metrics["csti"]["initial_matching"],
        )
        self.assertTrue(result.metrics["csti"]["evaluator_init_success"])
        self.assertEqual(
            "exact_full_tube_edt", result.provenance["csti"]["algorithm"]
        )
        self.assertIn("entity_manifest_digest", result.provenance["csti"])
        self.assertEqual(
            "sam31_text_frame_zero_hungarian_locked_v1",
            result.provenance["csti_observer"]["policy"],
        )
        self.assertEqual(1, segmenter.calls)

    def test_scene_prediction_failure_keeps_independently_observed_csti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            request.case["test_scene_prediction_failure"] = True
            evaluator = _FakeCSTIEvaluator(
                evaluator_config(csti_observer=True),
                csti_segmenter=_FakeTextSegmenter(),
            )

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.0, result.score)
        self.assertEqual(1.0, result.metrics["csti"]["score"])
        self.assertEqual(
            "synthetic_prediction_observation_failure", result.reason_code
        )

    def test_text_observer_init_failure_is_null_csti_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            evaluator = _FakeCSTIEvaluator(
                evaluator_config(csti_observer=True),
                csti_segmenter=_FakeTextSegmenter(initialize=False),
            )

            result = self._evaluate(evaluator, request, reference_path)

        self.assertEqual("evaluated", result.status)
        self.assertEqual(0.625, result.score)
        self.assertEqual(
            "evaluator_init_failure", result.metrics["csti"]["status"]
        )
        self.assertIsNone(result.metrics["csti"]["score"])
        self.assertFalse(result.metrics["csti"]["evaluator_init_success"])

    def test_text_observer_reference_failure_remains_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, reference_path = self._request(root)
            evaluator = _FakeCSTIEvaluator(
                evaluator_config(csti_observer=True),
                csti_segmenter=_FakeTextSegmenter(),
            )

            result = self._evaluate(
                evaluator,
                request,
                reference_path,
                frozen_reference_error=ReferenceAnalysisError(
                    "synthetic_reference_failure", "frozen GT is invalid"
                ),
            )

        self.assertEqual("unavailable", result.status)
        self.assertEqual("synthetic_reference_failure", result.reason_code)

    @staticmethod
    def _case(*, reference_capability: ReferenceCapability = ReferenceCapability.SAME_CASE_GT) -> dict:
        case = {
            "case_id": "case_1",
            "scene_id": "custom_scene",
            "physics": {},
            "entities": [
                {
                    "entity_id": "body",
                    "role_id": "participant",
                    "entity_class": "rigid_body",
                    "physical_attributes": {},
                    "condition_anchor": {},
                    "lifecycle": "persistent",
                }
            ],
            "apparatus": [],
        }
        if reference_capability is ReferenceCapability.SAME_CASE_GT:
            case["assets"] = {"reference_video": "reference.mp4"}
        elif reference_capability is ReferenceCapability.PHYSICS_PARENT:
            case["provenance"] = {"parent_case_id": "parent_1"}
        case["test_reference_capability"] = reference_capability.value
        return case

    @classmethod
    def _request(
        cls,
        root: Path,
        *,
        reference_capability: ReferenceCapability = ReferenceCapability.SAME_CASE_GT,
        prediction: bool = True,
    ) -> tuple[CaseEvaluationRequest, Path]:
        reference_path = root / "reference.mp4"
        prediction_path = root / "prediction.mp4"
        reference_path.write_bytes(b"reference")
        prediction_path.write_bytes(b"prediction")
        request = CaseEvaluationRequest(
            job={"job_id": "job_1"},
            case=cls._case(reference_capability=reference_capability),
            case_catalog={},
            prediction=(
                {"status": "complete", "video_path": str(prediction_path)}
                if prediction
                else None
            ),
            asset_root=root,
            artifact_dir=root / "artifacts",
            evaluator_config={},
        )
        return request, reference_path

    @staticmethod
    def _evaluate(
        evaluator: _FakeCSTIEvaluator,
        request: CaseEvaluationRequest,
        reference_path: Path,
        frozen_reference_error: Exception | None = None,
    ):
        info = SimpleNamespace(to_dict=lambda: {"width": 4, "height": 3})
        video = SimpleNamespace(
            info=info,
            frames=[
                np.zeros((3, 4, 3), dtype=np.uint8),
                np.zeros((3, 4, 3), dtype=np.uint8),
            ],
            source_indices=[0, 1],
            spatial_transform={},
            temporal_transform={},
            available=[True, True],
        )
        frozen = SimpleNamespace(
            entities={
                "body": SimpleNamespace(
                    masks=np.ones((2, 3, 4), dtype=np.uint8)
                )
            },
            provenance={"policy": "synthetic_frozen_reference"},
        )
        frozen_side_effect = frozen_reference_error or frozen
        with (
            patch(
                "physbench.evaluation.common.base.resolve_physics_reference",
                return_value=(reference_path, "same_case_reference", None),
            ),
            patch(
                "physbench.evaluation.common.base.reference_timeline",
                return_value=[0.0, 1.0],
            ),
            patch(
                "physbench.evaluation.common.base.sample_video",
                side_effect=[video, video],
            ),
            patch(
                "physbench.evaluation.common.base.load_frozen_reference_observation",
                side_effect=(
                    frozen_side_effect
                    if isinstance(frozen_side_effect, Exception)
                    else None
                ),
                return_value=(
                    None
                    if isinstance(frozen_side_effect, Exception)
                    else frozen_side_effect
                ),
            ),
        ):
            return evaluator.evaluate(request)


if __name__ == "__main__":
    unittest.main()
