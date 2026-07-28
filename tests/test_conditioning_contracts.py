from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from physbench.baseline_api import (
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime import (
    create_baseline_scaffold,
    load_data_adapter,
    validate_adaptation_record,
)
from physbench.baseline_runtime.adapter import StandardDataAdapter
from physbench.domain import (
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from physbench.io import canonical_sha256, write_json


DRIVER_SOURCE = """\
from pathlib import Path

from physbench.baseline_runtime import DirectManagedDriver


class Driver(DirectManagedDriver):
    def __init__(self, bundle):
        super().__init__(bundle)
        self.received_case = None

    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        self.received_case = case
        return {
            "job_id": job["job_id"],
            "output_video": str(
                run_dir / "predictions" / f"{job['job_id']}.mp4"
            ),
        }

    def execute_job(self, spec, *, log_path):
        raise AssertionError("fixture driver must not execute")
"""


ADAPTER_SOURCE = """\
from physbench.baseline_api.interfaces import DataAdapter


class FixtureAdapter(DataAdapter):
    def __init__(self, bundle):
        self.bundle = bundle
        self.received_has_physics = []

    @property
    def fingerprint(self):
        return "a" * 64

    @property
    def materialization_fingerprint(self):
        return "b" * 64

    def describe(self):
        return {
            "type": "fixture_python_adapter_v1",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": (
                self.materialization_fingerprint
            ),
            "generation_mode": "i2v",
            "text_conditioning_required": True,
        }

    def adapt_case(self, case, conditioning, *, role):
        has_physics = "physics" in case
        self.received_has_physics.append(has_physics)
        if conditioning == "generic":
            if has_physics:
                raise AssertionError(
                    "generic custom adapter received physics"
                )
            used_parameters = {}
            physics_channels = []
            prompt = "A pendulum moves."
        else:
            quantity = case["physics"]["string_length"]
            used_parameters = {"string_length": quantity}
            physics_channels = [{
                "id": "physics_text",
                "representation": "structured_text",
                "binding": "native_inputs.text.prompt",
                "transport": "inline_text",
                "used_parameters": ["string_length"],
            }]
            prompt = (
                "A pendulum moves with string_length="
                f"{quantity['value']} {quantity['unit']}."
            )
        first_frame = case["assets"]["first_frame"]
        return {
            "schema_version": "2.0",
            "case_id": case["case_id"],
            "conditioning": conditioning,
            "role": role,
            "used_parameters": used_parameters,
            "input_contract": {
                "schema_version": "1.0",
                "generation_mode": "i2v",
                "text": {
                    "required": True,
                    "binding": "native_inputs.text.prompt",
                },
                "media_channels": [{
                    "id": "initial_frame",
                    "kind": "image",
                    "asset_key": "first_frame",
                    "binding": (
                        "native_inputs.vision.first_frame_asset"
                    ),
                }],
                "physics_channels": physics_channels,
                "asset_access": ["first_frame"],
            },
            "native_inputs": {
                "text": {"prompt": prompt},
                "vision": {"first_frame_asset": first_frame},
                "custom_adapter": True,
            },
        }


def create_adapter(bundle):
    return FixtureAdapter(bundle)
"""


def _standard_adapter_config() -> dict:
    return {
        "preset": "standard_i2v_v1",
        "profile_set": "five_scene_i2v_v1",
        "first_frame_policy": "require_asset",
        "spatial": {
            "scene_profiles": {
                "pendulum": {"width": 64, "height": 64},
            },
        },
        "temporal": {
            "fps": 8,
            "num_frames": 9,
            "valid_frame_rule": "4n+1",
        },
    }


def _manifest(
    baseline_id: str,
    *,
    adapter: dict,
    custom_capabilities: bool,
) -> dict:
    capabilities = {
        "task_families": ["direct_eval"],
        "conditioning": ["generic", "physics"],
        "train": False,
        "finetune": False,
        "generate": True,
    }
    if custom_capabilities:
        capabilities.update({
            "generation_modes": ["i2v"],
            "physics_representations": ["structured_text"],
        })
    return {
        "schema_version": "4.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "implementation": {
            "kind": "managed",
            "driver": "driver.py",
            "fingerprint_paths": [],
        },
        "supported_scenes": ["pendulum"],
        "capabilities": capabilities,
        "model": {},
        "runtime": {},
        "adapter": adapter,
        "runner": {"type": "fixture_v1", "config": {}},
    }


def _create_bundle(
    parent: Path,
    *,
    name: str = "fixture",
    adapter_source: str | None = ADAPTER_SOURCE,
    adapter: dict | None = None,
) -> Path:
    root = parent / name
    root.mkdir(parents=True)
    (root / "driver.py").write_text(DRIVER_SOURCE, encoding="utf-8")
    if adapter is None:
        adapter = {
            "kind": "python",
            "entrypoint": "adapter.py",
            "config": {},
        }
    if adapter_source is not None:
        (root / "adapter.py").write_text(
            adapter_source,
            encoding="utf-8",
        )
    write_json(
        root / "baseline.json",
        _manifest(
            name,
            adapter=adapter,
            custom_capabilities=adapter.get("kind") == "python",
        ),
    )
    return root


def _case() -> dict:
    return {
        "schema_version": "2.0",
        "case_id": "pendulum_case",
        "scene_id": "pendulum",
        "assets": {
            "first_frame": "assets/first.png",
            "input_video": "assets/input.mp4",
            "reference_video": "assets/reference.mp4",
            "physics_reference_video": "assets/physics_reference.mp4",
            "source_video": "assets/source.mp4",
            "source_archive": "assets/source.zip",
            "subject_mask": "assets/mask.png",
        },
        "physics": {
            "string_length": {
                "value": 0.25,
                "unit": "m",
                "annotated": True,
            },
        },
        "appearance": {"background": "fixture"},
        "temporal": {"fps": 8},
        "alignment": None,
        "provenance": {
            "source_kind": "fixture",
            "parent_case_id": None,
        },
        "ood": {"level": "id", "factors": []},
        "has_real_reference_video": True,
    }


def _dataset(root: Path) -> DatasetSnapshot:
    case = _case()
    descriptor = {"dataset_id": "conditioning_contract_fixture"}
    views = {
        "view_a": {
            "schema_version": "2.0",
            "coverage": "complete",
            "scenes": {
                "pendulum": {
                    "train": [],
                    "test_id": ["pendulum_case"],
                    "test_ood1": [],
                },
            },
        },
        "view_b": {
            "schema_version": "2.0",
            "coverage": "complete",
            "scenes": {
                "pendulum": {"group_1": ["pendulum_case"]},
            },
        },
    }
    digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": [case],
        "views": views,
    })
    return DatasetSnapshot(
        root=root,
        descriptor=descriptor,
        cases=(case,),
        views=views,
        scene_configs={
            "pendulum": {
                "schema_version": "2.0",
                "scene_id": "pendulum",
            },
        },
        asset_lock=None,
        digest=digest,
        asset_root=root,
    )


def _task(root: Path, *, conditioning: str = "generic") -> TaskSpec:
    value = {
        "schema_version": "2.0",
        "task_id": f"fixture_direct_{conditioning}",
        "family": "direct_eval",
        "conditioning": conditioning,
        "dataset_id": "conditioning_contract_fixture",
        "dataset_view": "view_b",
        "selection": {
            "scene_ids": ["pendulum"],
            "groups": "all",
        },
        "ood2": {"enabled": False},
        "seeds": {"training": [], "inference": [7]},
    }
    return TaskSpec(
        path=root / f"{value['task_id']}.json",
        value=value,
        digest=canonical_sha256(value),
    )


def _generic_record(*, mode: str, kind: str, asset_key: str) -> dict:
    media_field = (
        "first_frame_asset" if kind == "image" else "input_video_asset"
    )
    return {
        "used_parameters": {},
        "native_inputs": {
            "text": {"prompt": "A text-conditioned physical scene."},
            "media": {media_field: f"assets/{asset_key}"},
        },
        "input_contract": {
            "schema_version": "1.0",
            "generation_mode": mode,
            "text": {
                "required": True,
                "binding": "native_inputs.text.prompt",
            },
            "media_channels": [{
                "id": "media",
                "kind": kind,
                "asset_key": asset_key,
                "binding": f"native_inputs.media.{media_field}",
            }],
            "physics_channels": [],
            "asset_access": [asset_key],
        },
    }


class AdapterFactoryTests(unittest.TestCase):
    def test_legacy_v4_adapter_without_kind_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(
                Path(temporary),
                adapter_source=None,
                adapter=_standard_adapter_config(),
            )
            bundle = load_baseline_bundle(root)
            adapter = load_data_adapter(bundle)

            self.assertNotIn("kind", bundle.value["adapter"])
            self.assertIsInstance(adapter, StandardDataAdapter)
            self.assertEqual("i2v", adapter.describe()["generation_mode"])

    def test_python_factory_loads_and_entrypoint_changes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            first_bundle = load_baseline_bundle(root)
            adapter = load_data_adapter(first_bundle)

            self.assertEqual(
                "fixture_python_adapter_v1",
                adapter.describe()["type"],
            )
            path = root / "adapter.py"
            path.write_text(
                path.read_text(encoding="utf-8") + "\n# digest change\n",
                encoding="utf-8",
            )
            second_bundle = load_baseline_bundle(root)
            self.assertNotEqual(first_bundle.digest, second_bundle.digest)

    def test_python_factory_rejects_wrong_return_type(self) -> None:
        source = "def create_adapter(bundle):\n    return object()\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(
                Path(temporary),
                adapter_source=source,
            )
            bundle = load_baseline_bundle(root)
            with self.assertRaisesRegex(
                TypeError,
                "must return a DataAdapter",
            ):
                load_data_adapter(bundle)

    def test_python_factory_rejects_entrypoint_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            manifest = _manifest(
                "fixture",
                adapter={
                    "kind": "python",
                    "entrypoint": "../outside.py",
                    "config": {},
                },
                custom_capabilities=True,
            )
            write_json(root / "baseline.json", manifest)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(root)


class InputContractTests(unittest.TestCase):
    def test_text_binding_must_resolve_to_non_empty_text(self) -> None:
        capabilities = {"physics_representations": ["structured_text"]}
        valid = _generic_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        validate_adaptation_record(
            valid,
            conditioning="generic",
            capabilities=capabilities,
        )

        invalid = copy.deepcopy(valid)
        invalid["native_inputs"]["text"]["prompt"] = ""
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_adaptation_record(
                invalid,
                conditioning="generic",
                capabilities=capabilities,
            )

    def test_i2v_and_v2v_require_the_matching_media_kind(self) -> None:
        capabilities = {"physics_representations": ["structured_text"]}
        i2v = _generic_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        v2v = _generic_record(
            mode="v2v",
            kind="video",
            asset_key="input_video",
        )
        for record in (i2v, v2v):
            validate_adaptation_record(
                record,
                conditioning="generic",
                capabilities=capabilities,
            )

        wrong = _generic_record(
            mode="i2v",
            kind="video",
            asset_key="input_video",
        )
        with self.assertRaisesRegex(ValueError, "requires at least one image"):
            validate_adaptation_record(
                wrong,
                conditioning="generic",
                capabilities=capabilities,
            )

        leaked_gt = _generic_record(
            mode="v2v",
            kind="video",
            asset_key="reference_video",
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_adaptation_record(
                leaked_gt,
                conditioning="generic",
                capabilities=capabilities,
            )

    def test_large_physics_payload_requires_artifact_reference(self) -> None:
        capabilities = {
            "physics_representations": ["optical_flow"],
        }
        record = _generic_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        record["used_parameters"] = {
            "gravity": {
                "value": 9.81,
                "unit": "m/s^2",
                "annotated": True,
            },
        }
        record["native_inputs"]["controls"] = {
            "flow": "artifact://controls/case/flow.npz",
        }
        record["input_contract"]["physics_channels"] = [{
            "id": "flow",
            "representation": "optical_flow",
            "binding": "native_inputs.controls.flow",
            "transport": "artifact_ref",
            "used_parameters": ["gravity"],
        }]
        validate_adaptation_record(
            record,
            conditioning="physics",
            capabilities=capabilities,
        )

        inline = copy.deepcopy(record)
        inline["native_inputs"]["controls"]["flow"] = [[0.0, 1.0]]
        with self.assertRaisesRegex(
            ValueError,
            "artifact:// or cache://",
        ):
            validate_adaptation_record(
                inline,
                conditioning="physics",
                capabilities=capabilities,
            )

    def test_generic_arm_rejects_all_physics_consumption(self) -> None:
        capabilities = {
            "physics_representations": ["structured_text"],
        }
        record = _generic_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        record["used_parameters"] = {
            "string_length": {"value": 0.25, "unit": "m"},
        }
        record["input_contract"]["physics_channels"] = [{
            "id": "physics_text",
            "representation": "structured_text",
            "binding": "native_inputs.text.prompt",
            "transport": "inline_text",
            "used_parameters": ["string_length"],
        }]
        with self.assertRaisesRegex(
            ValueError,
            "generic adaptation record.used_parameters",
        ):
            validate_adaptation_record(
                record,
                conditioning="generic",
                capabilities=capabilities,
            )

    def test_standard_v2v_uses_only_explicit_conditioning_video(self) -> None:
        config = _standard_adapter_config()
        config["preset"] = "standard_v2v_v1"
        config.pop("first_frame_policy")
        config["video_asset_key"] = "input_video"
        adapter = StandardDataAdapter(config)
        record = adapter.adapt_case(
            _case(),
            "generic",
            role="eval",
        )
        self.assertEqual(
            "assets/input.mp4",
            record["native_inputs"]["vision"]["input_video_asset"],
        )
        self.assertEqual(
            ["input_video"],
            record["input_contract"]["asset_access"],
        )
        self.assertNotIn(
            "reference",
            str(record["native_inputs"]).lower(),
        )
        with self.assertRaisesRegex(
            ValueError,
            "explicit conditioning asset",
        ):
            StandardDataAdapter({
                **config,
                "video_asset_key": "physics_reference_video",
            })


class CompilerAndDriverIsolationTests(unittest.TestCase):
    def _compiled_fixture(self, root: Path):
        bundle_root = _create_bundle(root)
        plugin = load_baseline_plugin(
            load_baseline_bundle(bundle_root)
        )
        instance = plugin.task_builder.build(
            _dataset(root),
            _task(root),
        )
        return plugin, instance

    def test_compiler_uses_custom_adapter_and_hides_generic_physics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plugin, instance = self._compiled_fixture(Path(temporary))

            self.assertEqual([False], plugin.data_adapter.received_has_physics)
            adaptation = instance.value["adaptations"][0]
            self.assertTrue(
                adaptation["native_inputs"]["custom_adapter"]
            )
            self.assertEqual({}, adaptation["used_parameters"])
            self.assertNotIn(
                "physics",
                instance.value["source"]["cases"][0],
            )

    def test_driver_receives_only_contract_authorized_case_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, instance = self._compiled_fixture(root)
            run_dir = root / "run"
            plugin.run_task(
                instance=instance,
                run_dir=run_dir,
                execute=False,
                stop_after_training=False,
            )

            received = plugin.driver.received_case
            self.assertIsNotNone(received)
            self.assertNotIn("physics", received)
            self.assertNotIn("has_real_reference_video", received)
            self.assertEqual(
                {"first_frame"},
                set(received["assets"]),
            )
            self.assertEqual(
                set(
                    instance.value["adaptations"][0][
                        "input_contract"
                    ]["asset_access"]
                ),
                set(received["assets"]),
            )
            for forbidden in (
                "reference_video",
                "physics_reference_video",
                "source_video",
            ):
                self.assertNotIn(forbidden, received["assets"])

    def test_physics_arm_rejects_non_annotated_fields_and_hides_raw_case(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root)
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            dataset = _dataset(root)
            dataset.cases[0]["physics"]["string_length"][
                "annotated"
            ] = False
            with self.assertRaisesRegex(
                ValueError,
                "non-annotated physics",
            ):
                plugin.task_builder.build(
                    dataset,
                    _task(root, conditioning="physics"),
                )

            dataset.cases[0]["physics"]["string_length"][
                "annotated"
            ] = True
            instance = plugin.task_builder.build(
                dataset,
                _task(root, conditioning="physics"),
            )
            plugin.run_task(
                instance=instance,
                run_dir=root / "physics-run",
                execute=False,
                stop_after_training=False,
            )
            self.assertNotIn("physics", plugin.driver.received_case)


class TaskInstanceContractTests(unittest.TestCase):
    def test_missing_envelope_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root)
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            value = plugin.task_builder.build(
                _dataset(root),
                _task(root),
            ).value
            value.pop("source")
            with self.assertRaisesRegex(
                ValueError,
                "missing required field.*source",
            ):
                BaselineTaskInstance.seal(value)

    def test_unknown_adaptation_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root)
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            value = plugin.task_builder.build(
                _dataset(root),
                _task(root),
            ).value
            value["inference"]["jobs"][0][
                "adaptation_id"
            ] = "missing-adaptation"
            with self.assertRaisesRegex(
                ValueError,
                "references unknown adaptation_id",
            ):
                BaselineTaskInstance.seal(value)


class ScaffoldContractTests(unittest.TestCase):
    def test_managed_v2v_scaffold_loads_through_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_baseline_scaffold(
                name="fixture_v2v",
                backend="managed-v2v",
                root=temporary,
            )
            bundle = load_baseline_bundle(root)
            plugin = load_baseline_plugin(bundle)

            self.assertEqual(
                ["v2v"],
                bundle.value["capabilities"]["generation_modes"],
            )
            self.assertEqual(
                "standard_v2v_v1",
                bundle.value["adapter"]["preset"],
            )
            self.assertEqual(
                "v2v",
                plugin.data_adapter.describe()["generation_mode"],
            )


if __name__ == "__main__":
    unittest.main()
