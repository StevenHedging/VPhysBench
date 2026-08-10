from __future__ import annotations

import copy
import hashlib
import json
import unittest

from _paths import ROOT
from physbench.evaluation.common.csti import CSTIConfig, CSTIContractError
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.tasks.planner import load_task


V10_PROTOCOL_FILE_SHA256 = (
    "c33c8d2a3625aa78035ef60977d7e27bc876af3998f207d6fed30a8226183d68"
)
V13_DIRECT_TASK_FILE_SHA256 = (
    "a1e52fa047333e737761a12caacac860fd04880d0e00fc5951c402a7fdeb9c97"
)
V13_FINETUNE_TASK_FILE_SHA256 = (
    "533b38f5c9f1529a7d70b48add9952cbc532bbe1b95629aa67d6ce327e04a4fb"
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationProtocolV11Tests(unittest.TestCase):
    def test_v11_enables_the_fixed_csti_contract(self) -> None:
        protocol = load_evaluation_protocol("scene_default_v11")

        self.assertEqual("scene_default_v11", protocol["protocol_id"])
        self.assertEqual(
            {
                "enabled": True,
                "algorithm": "exact_full_tube_edt",
                "spatial_tolerance_fraction": 0.004204482076268572,
                "temporal_tolerance_s": 0.025,
                "condition_frame_policy": "exclude_initial_samples",
                "initial_frames_excluded": 3,
                "score_aggregation": "full_tube",
                "diagnostic_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
                "case_aggregation": "mean_gt_entities",
                "timeline_policy": "physical_overlap",
                "mask_resolution": "scene_analysis_native",
            },
            protocol["general_metrics"]["csti"],
        )
        self.assertEqual(
            CSTIConfig.from_mapping(protocol["general_metrics"]["csti"]),
            CSTIConfig(
                enabled=True,
                algorithm="exact_full_tube_edt",
                spatial_tolerance_fraction=0.004204482076268572,
                temporal_tolerance_s=0.025,
                condition_frame_policy="exclude_initial_samples",
                initial_frames_excluded=3,
                score_aggregation="full_tube",
                diagnostic_prefix_fractions=(0.25, 0.5, 0.75, 1.0),
                case_aggregation="mean_gt_entities",
                timeline_policy="physical_overlap",
                mask_resolution="scene_analysis_native",
            ),
        )

    def test_v11_changes_only_csti_sampling_in_scene_contracts(self) -> None:
        v10 = load_evaluation_protocol("scene_default_v10")
        v11 = load_evaluation_protocol("scene_default_v11")

        self.assertEqual(v10["robustness"], v11["robustness"])
        expected_scenes = copy.deepcopy(v10["scenes"])
        for scene_id in (
            "collision_1d",
            "inclined_plane_slide",
            "parabolic_motion",
            "pendulum",
            "uniform_circular_motion",
        ):
            expected_scenes[scene_id]["timeline"]["fps"] = 24.0
            visualization = expected_scenes[scene_id].get("visualization", {})
            if isinstance(visualization.get("fps"), (int, float)):
                visualization["fps"] = 24.0
        self.assertEqual(expected_scenes, v11["scenes"])

    def test_csti_tasks_change_only_identity_and_protocol(self) -> None:
        pairs = (
            (
                "five_scene_direct_eval.json",
                "five_scene_direct_eval_csti.json",
                "five_scene_direct_eval_v13_csti",
            ),
            (
                "five_scene_finetune_eval.json",
                "five_scene_finetune_eval_csti.json",
                "five_scene_finetune_eval_v13_csti",
            ),
        )
        for legacy_name, csti_name, expected_id in pairs:
            with self.subTest(task=csti_name):
                legacy = load_task(ROOT / "tasks" / "official" / legacy_name)
                csti = load_task(ROOT / "tasks" / "official" / csti_name)
                self.assertEqual(expected_id, csti.task_id)
                self.assertEqual("scene_default_v11", csti.value["evaluation"]["protocol"])
                self.assertEqual("scene_default_v10", legacy.value["evaluation"]["protocol"])

                expected = copy.deepcopy(legacy.value)
                expected["task_id"] = expected_id
                expected["evaluation"]["protocol"] = "scene_default_v11"
                self.assertEqual(expected, csti.value)

    def test_legacy_protocol_and_tasks_remain_byte_frozen(self) -> None:
        self.assertEqual(
            V10_PROTOCOL_FILE_SHA256,
            _sha256(ROOT / "configs/evaluation/protocols/scene_default_v10.json"),
        )
        self.assertEqual(
            V13_DIRECT_TASK_FILE_SHA256,
            _sha256(ROOT / "tasks/official/five_scene_direct_eval.json"),
        )
        self.assertEqual(
            V13_FINETUNE_TASK_FILE_SHA256,
            _sha256(ROOT / "tasks/official/five_scene_finetune_eval.json"),
        )

    def test_protocol_schema_declares_a_strict_optional_csti_contract(self) -> None:
        schema = _read_json(ROOT / "schemas/v3/evaluation_protocol.schema.json")
        self.assertEqual(
            "https://json-schema.org/draft/2020-12/schema", schema["$schema"]
        )
        self.assertNotIn("general_metrics", schema["required"])
        self.assertEqual(
            {"$ref": "#/$defs/generalMetrics"},
            schema["properties"]["general_metrics"],
        )

        metrics = schema["$defs"]["generalMetrics"]
        self.assertEqual(["csti"], metrics["required"])
        self.assertFalse(metrics["additionalProperties"])
        csti = schema["$defs"]["csti"]
        self.assertEqual(
            {
                "enabled",
                "algorithm",
                "spatial_tolerance_fraction",
                "temporal_tolerance_s",
                "condition_frame_policy",
                "initial_frames_excluded",
                "score_aggregation",
                "diagnostic_prefix_fractions",
                "case_aggregation",
                "timeline_policy",
                "mask_resolution",
            },
            set(csti["required"]),
        )
        self.assertFalse(csti["additionalProperties"])
        expected_constants = {
            "enabled": True,
            "algorithm": "exact_full_tube_edt",
            "condition_frame_policy": "exclude_initial_samples",
            "score_aggregation": "full_tube",
            "case_aggregation": "mean_gt_entities",
            "timeline_policy": "physical_overlap",
            "mask_resolution": "scene_analysis_native",
        }
        for field, expected in expected_constants.items():
            with self.subTest(field=field):
                self.assertEqual(expected, csti["properties"][field]["const"])
        self.assertEqual(
            [0.25, 0.5, 0.75, 1.0],
            [
                item["const"]
                for item in csti["properties"]["diagnostic_prefix_fractions"]["prefixItems"]
            ],
        )
        for field in ("spatial_tolerance_fraction", "temporal_tolerance_s"):
            with self.subTest(field=field):
                self.assertEqual("number", csti["properties"][field]["type"])
                self.assertEqual(0, csti["properties"][field]["exclusiveMinimum"])
        self.assertEqual("integer", csti["properties"]["initial_frames_excluded"]["type"])
        self.assertEqual(1, csti["properties"]["initial_frames_excluded"]["minimum"])

    def test_runtime_config_rejects_unsupported_or_invalid_values(self) -> None:
        valid = load_evaluation_protocol("scene_default_v11")["general_metrics"]["csti"]
        mutations = {
            "enabled": False,
            "algorithm": "approximate",
            "condition_frame_policy": "include_first",
            "score_aggregation": "prefix_mean",
            "diagnostic_prefix_fractions": [0.5, 1.0],
            "case_aggregation": "mean_matched_entities",
            "timeline_policy": "source_duration",
            "mask_resolution": "fixed_512",
            "spatial_tolerance_fraction": 0.0,
            "temporal_tolerance_s": -0.1,
            "initial_frames_excluded": 0,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                mutated = dict(valid)
                mutated[field] = value
                with self.assertRaises(CSTIContractError):
                    CSTIConfig.from_mapping(mutated)

    def test_task_result_schema_documents_independent_dimensions(self) -> None:
        schema = _read_json(ROOT / "schemas/v3/task_evaluation.schema.json")
        self.assertNotIn("dimensions", schema["required"])
        dimensions = schema["properties"]["dimensions"]
        self.assertEqual({"expert", "csti"}, set(dimensions["required"]))
        self.assertFalse(dimensions["additionalProperties"])
        self.assertEqual(
            {"score", "source"},
            set(dimensions["properties"]["expert"]["required"]),
        )
        self.assertTrue(
            {"status", "score", "coverage", "by_scene", "breakdown"}.issubset(
                dimensions["properties"]["csti"]["required"]
            )
        )


if __name__ == "__main__":
    unittest.main()
