from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import audit_open_world_evaluator_v6 as audit


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


class OpenWorldV6AuditScriptTests(unittest.TestCase):
    def test_selection_and_prediction_mapping_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "select at least one"):
            audit._selected_case_ids([], {}, self_check=False)
        with self.assertRaisesRegex(ValueError, "require --self-check"):
            audit._selected_case_ids(["unscored"], {}, self_check=False)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "prediction=variant.mp4"
            video.touch()
            mapping = audit._prediction_mapping([f"case_a={video}"])
            self.assertEqual(video.resolve(), mapping["case_a"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit._prediction_mapping(
                    [f"case_a={video}", f"case_a={video}"]
                )

    def test_protocol_guard_is_scene_specific_and_sets_device(self) -> None:
        protocol = {
            "scenes": {
                "free_fall": {
                    "type": "free_fall_state_v6",
                    "observer_protocol": "open_world_v2",
                    "sam2": {"device": "auto"},
                },
                "collision_1d": {
                    "type": "collision_1d_state_v5",
                },
            }
        }
        config = audit._scene_config(
            protocol,
            scene_id="free_fall",
            device="cpu",
        )
        self.assertEqual("cpu", config["sam2"]["device"])
        with self.assertRaisesRegex(ValueError, "non-collision"):
            audit._scene_config(
                protocol,
                scene_id="collision_1d",
                device="cpu",
            )
        protocol["scenes"]["free_fall"]["observer_protocol"] = (
            "open_world_v1"
        )
        with self.assertRaisesRegex(ValueError, "open_world_v2"):
            audit._scene_config(
                protocol,
                scene_id="free_fall",
                device="cpu",
            )

    def test_reference_resolution_reports_physics_parent_capability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "assets" / "reference.mp4"
            reference.parent.mkdir()
            reference.touch()
            physics = {"height": {"value": 1.0, "unit": "m"}}
            parent = {
                "case_id": "parent",
                "scene_id": "free_fall",
                "physics": physics,
                "assets": {
                    "physics_reference_video": "assets/reference.mp4"
                },
                "has_real_reference_video": True,
                "provenance": {"parent_case_id": None},
            }
            child = {
                "case_id": "ood_child",
                "scene_id": "free_fall",
                "physics": physics,
                "assets": {},
                "has_real_reference_video": False,
                "provenance": {"parent_case_id": "parent"},
            }
            resolved, mode, parent_id = audit._reference_resolution(
                case=child,
                catalog={"parent": parent, "ood_child": child},
                asset_root=root,
            )
            self.assertEqual(reference.resolve(), resolved)
            self.assertEqual("parent_physics_reference", mode)
            self.assertEqual("parent", parent_id)

    def test_evaluate_preserves_case_and_reference_audit_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "prediction.mp4"
            video.touch()
            case = {
                "case_id": "freefall_case",
                "scene_id": "free_fall",
                "entities": [
                    {"entity_id": "ball_1", "entity_class": "ball"}
                ],
            }
            evaluator = _FakeEvaluator()
            result = audit._evaluate(
                evaluator,
                case=case,
                catalog={case["case_id"]: case},
                asset_root=root,
                video_path=video,
                output=root / "audit",
                variant="gt_self",
                evaluator_config={"type": "free_fall_state_v6"},
                reference_mode="same_case_reference",
            )
            self.assertEqual(case, evaluator.requests[0].case)
            self.assertEqual("gt_self", result["audit_variant"])
            self.assertEqual(
                "same_case_reference",
                result["audit_reference"]["mode"],
            )
            result_path = (
                root
                / "audit"
                / "cases"
                / audit._job_id("freefall_case", "gt_self")
                / "result.json"
            )
            self.assertTrue(result_path.is_file())


if __name__ == "__main__":
    unittest.main()
