from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import audit_collision_evaluator_v5 as audit


class _FakeResult:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return dict(self.value)


class _FakeEvaluator:
    def __init__(self):
        self.requests = []

    def evaluate(self, request):
        self.requests.append(request)
        return _FakeResult(
            {
                "case_id": request.case["case_id"],
                "scene_id": request.case["scene_id"],
                "status": "evaluated",
                "score": 0.75,
            }
        )


class CollisionV5AuditScriptTests(unittest.TestCase):
    def test_prediction_mapping_is_strict_and_preserves_equals_in_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "prediction=variant.mp4"
            video.touch()
            mapping = audit._prediction_mapping(
                [f"case_a={video}"]
            )
            self.assertEqual(video.resolve(), mapping["case_a"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit._prediction_mapping(
                    [f"case_a={video}", f"case_a={video}"]
                )
            with self.assertRaisesRegex(ValueError, "exact form"):
                audit._prediction_mapping(["malformed_without_separator"])
            with self.assertRaises(FileNotFoundError):
                audit._prediction_mapping(
                    [f"case_b={video.with_name('missing.mp4')}"]
                )

    def test_selection_never_silently_creates_a_zero_record_audit(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "select at least one"):
            audit._selected_case_ids([], {}, self_check=False)
        with self.assertRaisesRegex(ValueError, "require --self-check"):
            audit._selected_case_ids(
                ["case_without_video"],
                {},
                self_check=False,
            )
        selected = audit._selected_case_ids(
            ["self_case", "self_case"],
            {"prediction_case": Path("prediction.mp4")},
            self_check=True,
        )
        self.assertEqual(["self_case", "prediction_case"], selected)

    def test_protocol_guard_rejects_accidental_v3_or_v4_audit(self) -> None:
        v5 = {
            "scenes": {
                "collision_1d": {
                    "type": "collision_1d_state_v5",
                }
            }
        }
        self.assertIs(
            v5["scenes"]["collision_1d"],
            audit._collision_v5_config(v5),
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires collision_1d_state_v5",
        ):
            audit._collision_v5_config(
                {
                    "scenes": {
                        "collision_1d": {
                            "type": "collision_1d_state_v3",
                        }
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "no collision_1d"):
            audit._collision_v5_config({"scenes": {}})

    def test_reference_resolution_supports_physics_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "assets" / "reference.mp4"
            reference.parent.mkdir()
            reference.touch()
            physics = {"speed": {"value": 1.0, "unit": "m/s"}}
            parent = {
                "case_id": "parent",
                "scene_id": "collision_1d",
                "physics": physics,
                "assets": {
                    "physics_reference_video": (
                        "assets/reference.mp4"
                    )
                },
                "has_real_reference_video": True,
                "provenance": {"parent_case_id": None},
            }
            child = {
                "case_id": "ood_child",
                "scene_id": "collision_1d",
                "physics": physics,
                "assets": {},
                "has_real_reference_video": False,
                "provenance": {"parent_case_id": "parent"},
            }
            catalog = {"parent": parent, "ood_child": child}
            resolved = audit._reference_video(
                case=child,
                catalog=catalog,
                asset_root=root,
            )
            self.assertEqual(reference.resolve(), resolved)

            child["physics"] = {
                "speed": {"value": 2.0, "unit": "m/s"}
            }
            with self.assertRaisesRegex(
                Exception,
                "identical physics",
            ):
                audit._reference_video(
                    case=child,
                    catalog=catalog,
                    asset_root=root,
                )

    def test_evaluate_passes_arbitrary_entity_manifest_without_rewriting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "prediction.mp4"
            video.touch()
            entities = [
                {
                    "entity_id": f"ball_{index}",
                    "entity_class": "ball",
                }
                for index in range(1, 5)
            ]
            case = {
                "case_id": "four_body",
                "scene_id": "collision_1d",
                "entities": entities,
            }
            evaluator = _FakeEvaluator()
            result = audit._evaluate(
                evaluator,
                case=case,
                catalog={"four_body": case},
                asset_root=root,
                video_path=video,
                output=root / "audit",
                variant="prediction",
                evaluator_config={"type": "collision_1d_state_v5"},
            )

            self.assertEqual("prediction", result["audit_variant"])
            self.assertEqual(entities, evaluator.requests[0].case["entities"])
            self.assertEqual(
                str(video),
                evaluator.requests[0].prediction["video_path"],
            )
            result_path = (
                root
                / "audit"
                / "cases"
                / audit._job_id("four_body", "prediction")
                / "result.json"
            )
            self.assertTrue(result_path.is_file())

    def test_failure_record_is_finite_and_serializable(self) -> None:
        result = audit._failure_record(
            case_id="case",
            variant="prediction",
            exc=RuntimeError("unexpected audit failure"),
        )
        self.assertEqual("error", result["status"])
        self.assertIsNone(result["score"])
        self.assertEqual(
            "audit_execution_failed",
            result["reason_code"],
        )
        self.assertEqual("RuntimeError", result["audit_error"]["type"])


if __name__ == "__main__":
    unittest.main()
