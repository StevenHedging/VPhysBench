from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from physbench.baseline_runtime.compiler import ManagedTaskBuilder
from physbench.datasets.loader import load_dataset
from physbench.io import canonical_sha256, load_json, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


class DatasetContractV6Tests(unittest.TestCase):
    @staticmethod
    def _case() -> dict:
        return {
            "case_id": "pendulum_case_1",
            "scene_id": "pendulum",
            "text": {"prompt": "A reviewed pendulum trial."},
            "assets": {
                "caption": "caption.json",
                "first_frame": "frame.bin",
                "first_frame_mask_manifest": "mask_manifest.json",
                "physics_annotation": "physics.json",
                "reference_video": "frame.bin",
                "reference_observation_manifest": "reference_observation.json",
                "reference_observation_visualization_manifest": (
                    "reference_observation_visualization.json"
                ),
            },
            "physics": {
                "objects": {
                    "object_1": {
                        "mass": {
                            "value": 0.2,
                            "unit": "kg",
                            "symbol": "m",
                        },
                        "radius": {
                            "value": 0.01,
                            "unit": "m",
                            "symbol": "r",
                        },
                        "initial_angle": {
                            "value": 0.2,
                            "unit": "rad",
                            "symbol": "theta_0",
                        }
                    }
                },
                "environment": {
                    "string_length": {
                        "value": 0.5,
                        "unit": "m",
                        "symbol": "L",
                    }
                },
            },
            "appearance": {},
            "temporal": {"encoded_to_physical_speed": 1.0},
        }

    def _write_dataset(self, root: Path, *, case: dict | None = None) -> Path:
        materialized = copy.deepcopy(case or self._case())
        assets = root / "assets"
        assets.mkdir(parents=True)
        (assets / "frame.bin").write_bytes(b"fixture-media")
        for name in (
            "mask_manifest.json",
            "reference_observation.json",
            "reference_observation_visualization.json",
        ):
            write_json(assets / name, {"fixture": name})
        write_json(
            assets / "caption.json",
            {
                "case_id": materialized["case_id"],
                "scene_id": materialized["scene_id"],
                "caption": materialized["text"]["prompt"],
            },
        )
        write_json(
            assets / "physics.json",
            {
                "case_id": materialized["case_id"],
                "scene_id": materialized["scene_id"],
                "physics": materialized["physics"],
            },
        )
        indexed = copy.deepcopy(materialized)
        indexed.pop("text")
        indexed.pop("physics")
        write_jsonl(root / "cases.jsonl", [indexed])

        (root / "scenes").mkdir()
        write_json(
            root / "scenes" / "pendulum.json",
            {
                "schema_version": "2.0",
                "scene_id": "pendulum",
                "display_name": "Pendulum",
                "structured_physics_parameters": [
                    "objects.object_1.mass",
                    "objects.object_1.radius",
                    "objects.object_1.initial_angle",
                    "environment.string_length",
                ],
                "generalization_factors": [],
                "constraints": [],
                "metric_spec": {},
            },
        )
        (root / "views").mkdir()
        case_digest = canonical_sha256([materialized["case_id"]])
        write_json(
            root / "views" / "view_a.json",
            {
                "schema_version": "3.0",
                "view_id": "view_a",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "split_semantics": {
                    "primary_partitions": ["train", "test"],
                    "generalization_regimes": ["id", "ood", "mixed"],
                    "regime_is_relative_to": "view_a.train",
                },
                "scenes": {
                    "pendulum": {
                        "train": [materialized["case_id"]],
                        "test": [],
                    }
                },
                "test_annotations": {},
            },
        )
        write_json(
            root / "views" / "view_b.json",
            {
                "schema_version": "2.0",
                "view_id": "view_b",
                "coverage": "complete",
                "case_set_sha256": case_digest,
                "scenes": {
                    "pendulum": {"group_1": [materialized["case_id"]]}
                },
            },
        )
        write_json(
            root / "dataset.json",
            {
                "schema_version": "6.0",
                "dataset_id": "frozen_observation_fixture",
                "release": "14.0.0",
                "cases": "cases.jsonl",
                "asset_root": "assets",
                "scene_catalog": "scenes",
                "views": {
                    "view_a": "views/view_a.json",
                    "view_b": "views/view_b.json",
                },
            },
        )
        return root / "dataset.json"

    def test_schema_v6_materializes_caption_and_physics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = load_dataset(
                self._write_dataset(Path(temporary)),
                check_assets=True,
            )

        self.assertEqual("6.0", snapshot.descriptor["schema_version"])
        self.assertEqual(
            "A reviewed pendulum trial.",
            snapshot.cases[0]["text"]["prompt"],
        )
        self.assertEqual(
            0.2,
            snapshot.cases[0]["physics"]["objects"]["object_1"]["mass"][
                "value"
            ],
        )

    def test_schema_v6_requires_both_reference_observation_roles(self) -> None:
        for missing_role in (
            "reference_observation_manifest",
            "reference_observation_visualization_manifest",
        ):
            with self.subTest(missing_role=missing_role):
                case = self._case()
                case["assets"].pop(missing_role)
                with tempfile.TemporaryDirectory() as temporary:
                    descriptor = self._write_dataset(
                        Path(temporary),
                        case=case,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "invalid current assets",
                    ):
                        load_dataset(descriptor, check_assets=False)

    def test_managed_baseline_projection_hides_frozen_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = load_dataset(
                self._write_dataset(Path(temporary)),
                check_assets=False,
            )
            projected = ManagedTaskBuilder._adapter_case(snapshot.cases[0])

        self.assertNotIn("reference_video", projected["assets"])
        self.assertNotIn("first_frame_mask_manifest", projected["assets"])
        self.assertNotIn("reference_observation_manifest", projected["assets"])
        self.assertNotIn(
            "reference_observation_visualization_manifest",
            projected["assets"],
        )

    def test_v6_json_schemas_accept_the_materialized_case(self) -> None:
        case_schema = load_json(ROOT / "schemas/v6/case.schema.json")
        dataset_schema = load_json(ROOT / "schemas/v6/dataset.schema.json")
        Draft202012Validator.check_schema(case_schema)
        Draft202012Validator.check_schema(dataset_schema)
        Draft202012Validator(case_schema).validate(self._case())
        Draft202012Validator(dataset_schema).validate(
            {
                "schema_version": "6.0",
                "dataset_id": "frozen_observation_fixture",
                "release": "14.0.0",
                "cases": "cases.jsonl",
                "asset_root": "assets",
                "scene_catalog": "scenes",
                "views": {
                    "view_a": "views/view_a.json",
                    "view_b": "views/view_b.json",
                },
            }
        )

    def test_v6_reference_observation_schemas_accept_closed_examples(self) -> None:
        digest = "a" * 64
        file_record = {
            "path": "canonical/reference_observation/timeline.json",
            "size_bytes": 123,
            "sha256": digest,
        }
        examples = {
            "reference_observation_timeline.schema.json": {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "sampling_rate_hz": 24.0,
                "source_video": {
                    "frame_count": 241,
                    "width": 1080,
                    "height": 1920,
                    "fps": 240.0,
                    "duration_seconds": 1.0,
                },
                "temporal": {"encoded_to_physical_speed": 1.0},
                "samples": [
                    {
                        "observation_index": 0,
                        "encoded_time_seconds": 0.0,
                        "physical_time_seconds": 0.0,
                        "source_frame_index": 0,
                        "source_time_seconds": 0.0,
                    }
                ],
            },
            "reference_observation_quality.schema.json": {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "status": "pass",
                "config_fingerprint": digest,
                "checks": [],
                "entities": {},
                "failures": [],
                "warnings": [],
            },
            "reference_observation_review.schema.json": {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "decision": "approved",
                "scope": "full_video",
                "reviewer": "reviewer_01",
                "observation_manifest_sha256": digest,
                "visualization_manifest_sha256": digest,
                "checks": {},
                "entity_reviews": {},
                "issues": [],
            },
            "reference_observation_visualization.schema.json": {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "observation_manifest_sha256": digest,
                "renderer": {
                    "id": "reference_observation_renderer_v1",
                    "config_fingerprint": digest,
                },
                "video": {
                    "width": 1536,
                    "height": 864,
                    "fps": 24.0,
                    "frame_count": 25,
                },
                "files": [
                    {
                        **file_record,
                        "path": "canonical/reference_observation/visualization/overview.mp4",
                    },
                    {
                        **file_record,
                        "path": "canonical/reference_observation/visualization/contact_sheet.png",
                    },
                    {
                        **file_record,
                        "path": "canonical/reference_observation/visualization/trajectory.png",
                    },
                ],
            },
            "reference_observation.schema.json": {
                "schema_version": "1.0",
                "case_id": "pendulum_case_1",
                "scene_id": "pendulum",
                "source": {
                    "reference_video": file_record,
                    "first_frame_mask_manifest": file_record,
                },
                "generator": {
                    "id": "reference_observation_generator_v1",
                    "code_revision": "658ad86",
                    "model_id": "facebook/sam2.1-hiera-tiny",
                    "config_fingerprint": digest,
                },
                "timeline": file_record,
                "quality": file_record,
                "entities": [
                    {
                        "object_id": "object_1",
                        "mask_id": "01",
                        "mask_tube": file_record,
                        "trajectory": file_record,
                    }
                ],
            },
        }
        for name, example in examples.items():
            with self.subTest(schema=name):
                schema = load_json(ROOT / "schemas/v6" / name)
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema).validate(example)


if __name__ == "__main__":
    unittest.main()
