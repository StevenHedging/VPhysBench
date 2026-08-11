from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from physbench.baseline_api import (
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime import (
    build_i2v_media_contract,
    create_baseline_scaffold,
    load_data_adapter,
    validate_adaptation_record,
)
from physbench.baseline_runtime.adapter import StandardDataAdapter
from physbench.baseline_runtime.drivers.subprocess_i2v import (
    StandardI2VCLIDriver,
)
from physbench.baseline_runtime.compiler import ManagedTaskBuilder
from physbench.domain import (
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from physbench.io import canonical_sha256, write_json
from physbench.orchestration import compile_task_instance


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
from physbench.baseline_runtime import build_i2v_media_contract


class FixtureAdapter(DataAdapter):
    def __init__(self, bundle):
        self.bundle = bundle
        self.received_has_physics = []
        self.received_physics_keys = []
        self.received_case_keys = []

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
            "input_policy": self.bundle.value["input_policy"],
            "physics_representations": self.bundle.value[
                "input_policy"
            ]["physics"]["representations"],
            "text_conditioning_required": True,
        }

    def adapt_case(self, case, *, role):
        has_physics = "physics" in case
        self.received_has_physics.append(has_physics)
        self.received_physics_keys.append(
            sorted(case.get("physics", {}))
        )
        self.received_case_keys.append(sorted(case))
        physics_usage = self.bundle.value[
            "input_policy"
        ]["physics"]["usage"]
        base_prompt = case["text"]["prompt"]
        if physics_usage == "ignored":
            used_parameters = {}
            physics_channels = []
            prompt = base_prompt
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
                base_prompt + " Physical parameters: string_length="
                f"{quantity['value']} {quantity['unit']}."
            )
        first_frame = case["assets"]["first_frame"]
        return {
            "schema_version": "3.0",
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
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
                "media_contract": build_i2v_media_contract(
                    conditioning_asset=first_frame,
                    width=64,
                    height=64,
                    temporal={
                        "fps": 8,
                        "num_frames": 9,
                        "valid_frame_rule": "4n+1",
                    },
                ),
                "custom_adapter": True,
            },
        }


def create_adapter(bundle):
    return FixtureAdapter(bundle)
"""


def _standard_adapter_config() -> dict:
    return {
        "preset": "standard_i2v_v1",
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


def _sealed_standard_adapter_config() -> dict:
    config = _standard_adapter_config()
    config["spatial"].update({
        "conditioning_transform": "aspect_preserving_contain",
        "conditioning_margin_fill": "edge_replicate",
    })
    return config


def _input_policy(
    usage: str = "ignored",
    representations: list[str] | None = None,
) -> dict:
    if representations is None:
        representations = (
            [] if usage == "ignored" else ["structured_text"]
        )
    return {
        "schema_version": "1.0",
        "case_view": "conditionable_case_v1",
        "text": {
            "source": "case.text.prompt",
            "usage": "required",
        },
        "physics": {
            "source": "case.physics",
            "usage": usage,
            "representations": representations,
        },
    }


def _manifest(
    baseline_id: str,
    *,
    adapter: dict,
    input_policy: dict | None = None,
) -> dict:
    generation_mode = {
        "standard_t2v_v1": "t2v",
        "standard_i2v_v1": "i2v",
        "standard_v2v_v1": "v2v",
    }.get(adapter.get("preset"), "i2v")
    capabilities = {
        "task_families": ["direct_eval"],
        "generation_modes": [generation_mode],
        "train": False,
        "finetune": False,
        "generate": True,
    }
    return {
        "schema_version": "5.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "implementation": {
            "kind": "managed",
            "driver": "driver.py",
            "fingerprint_paths": [],
        },
        "supported_scenes": ["pendulum"],
        "capabilities": capabilities,
        "input_policy": input_policy or _input_policy(),
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
    input_policy: dict | None = None,
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
            input_policy=input_policy,
        ),
    )
    return root


def _case() -> dict:
    return {
        "schema_version": "3.0",
        "case_id": "pendulum_case",
        "scene_id": "pendulum",
        "text": {
            "schema_version": "1.0",
            "language": "en",
            "prompt": "A pendulum moves.",
            "annotation_source": "fixture",
        },
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
            "bob_radius": {
                "value": 0.02,
                "unit": "m",
                "annotated": True,
            },
            "initial_angle": {
                "value": 30.0,
                "unit": "deg",
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
    descriptor = {
        "schema_version": "3.0",
        "dataset_id": "conditioning_contract_fixture",
    }
    views = {
        "view_a": {
            "schema_version": "3.0",
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
            "schema_version": "3.0",
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
                "schema_version": "3.0",
                "scene_id": "pendulum",
            },
        },
        asset_lock=None,
        digest=digest,
        asset_root=root,
    )


def _task(root: Path) -> TaskSpec:
    value = {
        "schema_version": "1.0",
        "task_id": "fixture_direct",
        "family": "direct_eval",
        "dataset_id": "conditioning_contract_fixture",
        "selection": {
            "evaluation_scene_ids": ["pendulum"],
            "groups": "all",
        },
        "seeds": {"training": [], "inference": [7]},
        "evaluation": {
            "protocol": "scene_default_v1",
            "reporting": {
                "primary_score": "overall_test",
                "breakdowns": [],
                "minimum_subgroup_jobs": 1,
            },
        },
    }
    return TaskSpec(
        path=root / f"{value['task_id']}.json",
        value=value,
        digest=canonical_sha256(value),
    )


def _base_record(*, mode: str, kind: str, asset_key: str) -> dict:
    media_field = (
        "first_frame_asset" if kind == "image" else "input_video_asset"
    )
    record = {
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
    if mode == "i2v":
        record["native_inputs"]["media_contract"] = (
            build_i2v_media_contract(
                conditioning_asset=f"assets/{asset_key}",
                width=64,
                height=64,
                temporal={
                    "fps": 8,
                    "num_frames": 9,
                    "valid_frame_rule": "4n+1",
                },
            )
        )
    return record


def _record_with_structured_physics() -> dict:
    record = _base_record(
        mode="i2v",
        kind="image",
        asset_key="first_frame",
    )
    record["used_parameters"] = {
        "string_length": {
            "value": 0.25,
            "unit": "m",
        },
    }
    record["native_inputs"]["text"]["prompt"] += (
        " The string length is 0.25 m."
    )
    record["input_contract"]["physics_channels"] = [{
        "id": "physics_text",
        "representation": "structured_text",
        "binding": "native_inputs.text.prompt",
        "transport": "inline_text",
        "used_parameters": ["string_length"],
    }]
    return record


class AdapterFactoryTests(unittest.TestCase):
    def test_standard_i2v_adapter_seals_full_content_spatial_contract(
        self,
    ) -> None:
        adapter = StandardDataAdapter(
            _sealed_standard_adapter_config(),
            _input_policy(),
        )
        record = adapter.adapt_case(_case(), role="eval")

        self.assertEqual(
            build_i2v_media_contract(
                conditioning_asset="assets/first.png",
                width=64,
                height=64,
                temporal={
                    "fps": 8,
                    "num_frames": 9,
                    "valid_frame_rule": "4n+1",
                },
            ),
            record["native_inputs"]["media_contract"],
        )

    def test_sealed_i2v_adapter_rejects_black_canvas_margin(self) -> None:
        config = _sealed_standard_adapter_config()
        config["spatial"]["conditioning_margin_fill"] = "black"
        with self.assertRaisesRegex(ValueError, "edge_replicate"):
            StandardDataAdapter(config, _input_policy())

    def test_standard_v5_adapter_without_kind_loads(self) -> None:
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

    def test_schema_v4_manifest_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            manifest = _manifest(
                "fixture",
                adapter={
                    "kind": "python",
                    "entrypoint": "adapter.py",
                    "config": {},
                },
            )
            manifest["schema_version"] = "4.0"
            write_json(root / "baseline.json", manifest)

            with self.assertRaisesRegex(
                ValueError,
                "schema_version=5.0",
            ):
                load_baseline_bundle(root)

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

    def test_python_adapter_relative_helper_is_loaded_and_fingerprinted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            (root / "helper.py").write_text(
                "TOKEN = 'first'\n",
                encoding="utf-8",
            )
            adapter_path = root / "adapter.py"
            adapter_path.write_text(
                "from .helper import TOKEN\n" + ADAPTER_SOURCE,
                encoding="utf-8",
            )
            first = load_baseline_bundle(root)
            self.assertIsNotNone(load_data_adapter(first))

            (root / "helper.py").write_text(
                "TOKEN = 'second'\n",
                encoding="utf-8",
            )
            second = load_baseline_bundle(root)
            self.assertNotEqual(first.digest, second.digest)

    def test_nested_adapter_executes_package_initializer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            package = root / "adapter_pkg"
            package.mkdir()
            (package / "__init__.py").write_text(
                "TOKEN = 'package-initialized'\n",
                encoding="utf-8",
            )
            (package / "adapter.py").write_text(
                "from . import TOKEN\n" + ADAPTER_SOURCE,
                encoding="utf-8",
            )
            write_json(
                root / "baseline.json",
                _manifest(
                    "fixture",
                    adapter={
                        "kind": "python",
                        "entrypoint": "adapter_pkg/adapter.py",
                        "config": {},
                    },
                ),
            )

            adapter = load_data_adapter(load_baseline_bundle(root))
            self.assertEqual(
                "fixture_python_adapter_v1",
                adapter.describe()["type"],
            )

    def test_identical_bundle_copies_use_distinct_module_namespaces(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first_root = _create_bundle(parent / "first")
            (parent / "second").mkdir()
            second_root = parent / "second" / "fixture"
            shutil.copytree(first_root, second_root)

            first_bundle = load_baseline_bundle(first_root)
            second_bundle = load_baseline_bundle(second_root)
            first_adapter = load_data_adapter(first_bundle)
            second_adapter = load_data_adapter(second_bundle)

            self.assertEqual(first_bundle.digest, second_bundle.digest)
            self.assertNotEqual(
                first_adapter.__class__.__module__,
                second_adapter.__class__.__module__,
            )

    def test_driver_relative_import_and_helper_enter_bundle_digest(
        self,
    ) -> None:
        helper_source = """\
from dataclasses import dataclass
from physbench.baseline_runtime import DirectManagedDriver


@dataclass(frozen=True)
class Marker:
    value: str


class BaseDriver(DirectManagedDriver):
    marker = Marker("loaded")

    def prepare_job(
        self, *, job, case, adaptation, source_root, run_dir
    ):
        return {
            "job_id": job["job_id"],
            "output_video": str(
                run_dir / "predictions" / f"{job['job_id']}.mp4"
            ),
        }

    def execute_job(self, spec, *, log_path):
        raise AssertionError("fixture driver must not execute")
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = _create_bundle(Path(temporary))
            package = root / "runtime"
            package.mkdir()
            helper = package / "helper.py"
            helper.write_text(helper_source, encoding="utf-8")
            (package / "driver.py").write_text(
                "from .helper import BaseDriver\n"
                "class Driver(BaseDriver):\n"
                "    pass\n",
                encoding="utf-8",
            )
            manifest = _manifest(
                "fixture",
                adapter={
                    "kind": "python",
                    "entrypoint": "adapter.py",
                    "config": {},
                },
            )
            manifest["implementation"]["driver"] = "runtime/driver.py"
            write_json(root / "baseline.json", manifest)

            first = load_baseline_bundle(root)
            plugin = load_baseline_plugin(first)
            self.assertEqual("loaded", plugin.driver.marker.value)

            helper.write_text(
                helper_source.replace('"loaded"', '"changed"'),
                encoding="utf-8",
            )
            second = load_baseline_bundle(root)
            self.assertNotEqual(first.digest, second.digest)

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
            )
            write_json(root / "baseline.json", manifest)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(root)


class InputContractTests(unittest.TestCase):
    def test_input_contract_rejects_evaluator_only_mask_manifest(
        self,
    ) -> None:
        record = _base_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        record["input_contract"]["asset_access"].append(
            "first_frame_mask_manifest"
        )

        with self.assertRaisesRegex(ValueError, "reserved asset key"):
            validate_adaptation_record(
                record,
                input_policy=_input_policy(),
            )

    def test_text_binding_must_resolve_to_non_empty_text(self) -> None:
        valid = _base_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        validate_adaptation_record(
            valid,
            input_policy=_input_policy(),
        )

        invalid = copy.deepcopy(valid)
        invalid["native_inputs"]["text"]["prompt"] = ""
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_adaptation_record(
                invalid,
                input_policy=_input_policy(),
            )

    def test_i2v_and_v2v_require_the_matching_media_kind(self) -> None:
        i2v = _base_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        v2v = _base_record(
            mode="v2v",
            kind="video",
            asset_key="input_video",
        )
        for record in (i2v, v2v):
            validate_adaptation_record(
                record,
                input_policy=_input_policy(),
            )

        wrong = _base_record(
            mode="i2v",
            kind="video",
            asset_key="input_video",
        )
        with self.assertRaisesRegex(ValueError, "requires one or more image"):
            validate_adaptation_record(
                wrong,
                input_policy=_input_policy(),
            )

        leaked_gt = _base_record(
            mode="v2v",
            kind="video",
            asset_key="reference_video",
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_adaptation_record(
                leaked_gt,
                input_policy=_input_policy(),
            )

        qualified_gt = _base_record(
            mode="v2v",
            kind="video",
            asset_key="assets.reference_video",
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_adaptation_record(
                qualified_gt,
                input_policy=_input_policy(),
            )

    def test_generation_modes_have_unambiguous_media_modalities(
        self,
    ) -> None:
        t2v_with_image = _base_record(
            mode="t2v",
            kind="image",
            asset_key="first_frame",
        )
        with self.assertRaisesRegex(ValueError, "must not declare media"):
            validate_adaptation_record(
                t2v_with_image,
                input_policy=_input_policy(),
            )

        hybrid = _base_record(
            mode="hybrid",
            kind="image",
            asset_key="first_frame",
        )
        with self.assertRaisesRegex(ValueError, "both image and video"):
            validate_adaptation_record(
                hybrid,
                input_policy=_input_policy(),
            )
        hybrid["native_inputs"]["media"][
            "input_video_asset"
        ] = "assets/input_video"
        hybrid["input_contract"]["media_channels"].append({
            "id": "video",
            "kind": "video",
            "asset_key": "input_video",
            "binding": "native_inputs.media.input_video_asset",
        })
        hybrid["input_contract"]["asset_access"].append("input_video")
        validate_adaptation_record(
            hybrid,
            input_policy=_input_policy(),
        )

    def test_derived_artifact_can_supply_v2v_conditioning(self) -> None:
        record = _base_record(
            mode="v2v",
            kind="video",
            asset_key="input_video",
        )
        digest = "c" * 64
        record["native_inputs"]["media"][
            "input_video_asset"
        ] = f"artifact://sha256/{digest}"
        record["input_contract"]["media_channels"] = [{
            "id": "proxy_video",
            "kind": "video",
            "origin": "derived_artifact",
            "binding": "native_inputs.media.input_video_asset",
            "artifact_provenance": {
                "content_sha256": digest,
                "producer_fingerprint": "d" * 64,
                "source_digest": "e" * 64,
            },
        }]
        record["input_contract"]["asset_access"] = []
        validate_adaptation_record(
            record,
            input_policy=_input_policy(),
        )

    def test_large_physics_payload_requires_artifact_reference(self) -> None:
        policy = _input_policy(
            "required",
            ["optical_flow"],
        )
        record = _base_record(
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
        content_sha256 = "c" * 64
        record["native_inputs"]["controls"] = {
            "flow": f"artifact://sha256/{content_sha256}",
        }
        record["input_contract"]["physics_channels"] = [{
            "id": "flow",
            "representation": "optical_flow",
            "binding": "native_inputs.controls.flow",
            "transport": "artifact_ref",
            "used_parameters": ["gravity"],
            "artifact_provenance": {
                "content_sha256": content_sha256,
                "producer_fingerprint": "d" * 64,
                "source_digest": "e" * 64,
            },
        }]
        validate_adaptation_record(
            record,
            input_policy=policy,
        )

        inline = copy.deepcopy(record)
        inline["native_inputs"]["controls"]["flow"] = [[0.0, 1.0]]
        with self.assertRaisesRegex(
            ValueError,
            "artifact://sha256",
        ):
            validate_adaptation_record(
                inline,
                input_policy=policy,
            )

    def test_ignored_policy_rejects_all_physics_consumption(self) -> None:
        record = _record_with_structured_physics()
        record["input_contract"]["physics_channels"] = []
        with self.assertRaisesRegex(
            ValueError,
            "physics-ignored adaptation record.used_parameters",
        ):
            validate_adaptation_record(
                record,
                input_policy=_input_policy("ignored"),
            )

    def test_required_policy_must_consume_a_parameter(self) -> None:
        valid = _record_with_structured_physics()
        validate_adaptation_record(
            valid,
            input_policy=_input_policy("required"),
        )

        record = _base_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        record["input_contract"]["physics_channels"] = [{
            "id": "empty_physics",
            "representation": "structured_text",
            "binding": "native_inputs.text.prompt",
            "transport": "inline_text",
            "used_parameters": [],
        }]
        with self.assertRaisesRegex(
            ValueError,
            "must consume at least one",
        ):
            validate_adaptation_record(
                record,
                input_policy=_input_policy("required"),
            )

    def test_optional_policy_accepts_zero_or_audited_physics_use(
        self,
    ) -> None:
        policy = _input_policy("optional")
        unused = _base_record(
            mode="i2v",
            kind="image",
            asset_key="first_frame",
        )
        used = _record_with_structured_physics()
        validate_adaptation_record(unused, input_policy=policy)
        validate_adaptation_record(used, input_policy=policy)

        inconsistent = copy.deepcopy(used)
        inconsistent["used_parameters"] = {}
        with self.assertRaisesRegex(
            ValueError,
            "declare channels exactly when",
        ):
            validate_adaptation_record(
                inconsistent,
                input_policy=policy,
            )

    def test_standard_v2v_uses_only_explicit_conditioning_video(self) -> None:
        config = _standard_adapter_config()
        config["preset"] = "standard_v2v_v1"
        config.pop("first_frame_policy")
        config["video_asset_key"] = "input_video"
        adapter = StandardDataAdapter(config, _input_policy())
        record = adapter.adapt_case(
            _case(),
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
            StandardDataAdapter(
                {
                    **config,
                    "video_asset_key": "physics_reference_video",
                },
                _input_policy(),
            )


class CompilerAndDriverIsolationTests(unittest.TestCase):
    def _compiled_fixture(self, root: Path):
        bundle_root = _create_bundle(root)
        plugin = load_baseline_plugin(
            load_baseline_bundle(bundle_root)
        )
        instance = compile_task_instance(plugin,
            _dataset(root),
            _task(root),
        )
        return plugin, instance

    def test_evaluator_only_mask_manifest_never_reaches_adapter_case(
        self,
    ) -> None:
        case = _case()
        case["assets"]["first_frame_mask_manifest"] = (
            "assets/pendulum/canonical/masks/manifest.json"
        )

        projected = ManagedTaskBuilder._adapter_case(case)

        self.assertNotIn(
            "first_frame_mask_manifest",
            projected["assets"],
        )

    def test_compiler_gives_adapter_annotated_physics_but_ignored_policy_does_not_consume_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plugin, instance = self._compiled_fixture(Path(temporary))

            self.assertEqual([True], plugin.data_adapter.received_has_physics)
            self.assertEqual(
                ["bob_radius", "initial_angle", "string_length"],
                plugin.data_adapter.received_physics_keys[0],
            )
            self.assertNotIn(
                "provenance",
                plugin.data_adapter.received_case_keys[0],
            )
            adaptation = instance.value["adaptations"][0]
            self.assertTrue(
                adaptation["native_inputs"]["custom_adapter"]
            )
            self.assertEqual({}, adaptation["used_parameters"])
            self.assertEqual(
                "A pendulum moves.",
                adaptation["native_inputs"]["text"]["prompt"],
            )
            value = instance.value
            self.assertEqual("3.0", value["schema_version"])
            self.assertNotIn("conditioning", value)
            self.assertNotIn("conditioning", value["semantics"])
            self.assertIn("physics", value["source"]["cases"][0])
            self.assertIn("text", value["source"]["cases"][0])

    def test_physics_ignored_baseline_is_counterfactually_invariant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ignored_plugin = load_baseline_plugin(
                load_baseline_bundle(
                    _create_bundle(root, name="ignored_fixture")
                )
            )
            required_plugin = load_baseline_plugin(
                load_baseline_bundle(
                    _create_bundle(
                        root,
                        name="required_fixture",
                        input_policy=_input_policy("required"),
                    )
                )
            )
            original = _case()
            counterfactual = copy.deepcopy(original)
            counterfactual["physics"]["string_length"]["value"] = 9.0

            ignored_original = ignored_plugin.data_adapter.adapt_case(
                original,
                role="eval",
            )
            ignored_counterfactual = (
                ignored_plugin.data_adapter.adapt_case(
                    counterfactual,
                    role="eval",
                )
            )
            self.assertEqual(
                ignored_original["native_inputs"],
                ignored_counterfactual["native_inputs"],
            )
            self.assertEqual({}, ignored_original["used_parameters"])

            required_original = required_plugin.data_adapter.adapt_case(
                original,
                role="eval",
            )
            required_counterfactual = (
                required_plugin.data_adapter.adapt_case(
                    counterfactual,
                    role="eval",
                )
            )
            self.assertNotEqual(
                required_original["native_inputs"],
                required_counterfactual["native_inputs"],
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

    def test_driver_cannot_override_compiler_sealed_media_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(
                root,
                adapter_source=None,
                adapter=_sealed_standard_adapter_config(),
            )
            plugin = load_baseline_plugin(load_baseline_bundle(bundle_root))
            instance = compile_task_instance(plugin, _dataset(root), _task(root))
            job = instance.value["inference"]["jobs"][0]
            expected = job["native_inputs"]["media_contract"]
            tampered = copy.deepcopy(expected)
            tampered["output"]["canvas"] = {
                "width": 832,
                "height": 480,
            }

            plugin.driver.run_task = mock.Mock(return_value=(
                {
                    "operation_id": "train",
                    "status": "not_requested",
                },
                [{
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "baseline_id": plugin.bundle.baseline_id,
                    "evaluation_partition": job["evaluation_partition"],
                    "seed": job["seed"],
                    "status": "planned",
                    "video_path": None,
                    "media_contract": tampered,
                }],
            ))
            with self.assertRaisesRegex(
                ValueError,
                "driver media contract differs",
            ):
                plugin.run_task(
                    instance=instance,
                    run_dir=root / "run",
                    execute=False,
                    stop_after_training=False,
                )

    def test_adapter_receives_only_annotated_physics_and_driver_hides_raw_case(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(
                root,
                input_policy=_input_policy("required"),
            )
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            dataset = _dataset(root)
            dataset.cases[0]["physics"]["derived_period"] = {
                "value": 1.0,
                "unit": "s",
                "annotated": False,
            }
            instance = compile_task_instance(plugin,
                dataset,
                _task(root),
            )
            self.assertEqual(
                ["bob_radius", "initial_angle", "string_length"],
                plugin.data_adapter.received_physics_keys[-1],
            )
            plugin.run_task(
                instance=instance,
                run_dir=root / "physics-run",
                execute=False,
                stop_after_training=False,
            )
            self.assertNotIn("physics", plugin.driver.received_case)

    def test_physics_values_must_match_dataset_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = load_baseline_plugin(
                load_baseline_bundle(
                    _create_bundle(
                        root,
                        input_policy=_input_policy("required"),
                    )
                )
            )
            original = plugin.data_adapter.adapt_case

            def tampered_adapter(case, *, role):
                record = copy.deepcopy(
                    original(case, role=role)
                )
                record["used_parameters"]["string_length"][
                    "value"
                ] = 999.0
                return record

            plugin.data_adapter.adapt_case = tampered_adapter
            with self.assertRaisesRegex(
                ValueError,
                "differ from Dataset truth",
            ):
                compile_task_instance(plugin,
                    _dataset(root),
                    _task(root),
                )

    def test_v2v_rejects_cross_case_reference_digest_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _standard_adapter_config()
            config["preset"] = "standard_v2v_v1"
            config.pop("first_frame_policy")
            config["video_asset_key"] = "input_video"
            bundle_root = _create_bundle(
                root,
                name="v2v_fixture",
                adapter_source=None,
                adapter=config,
            )
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )

            base = _dataset(root)
            first = copy.deepcopy(base.cases[0])
            second = copy.deepcopy(first)
            second["case_id"] = "pendulum_other"
            second["assets"][
                "reference_video"
            ] = "assets/other-reference.mp4"
            digest = "f" * 64
            dataset = DatasetSnapshot(
                root=base.root,
                descriptor=base.descriptor,
                cases=(first, second),
                views=base.views,
                scene_configs=base.scene_configs,
                asset_lock={
                    "files": [
                        {
                            "path": first["assets"]["input_video"],
                            "sha256": digest,
                        },
                        {
                            "path": second["assets"]["reference_video"],
                            "sha256": digest,
                        },
                    ],
                },
                digest=base.digest,
                asset_root=base.asset_root,
            )
            with self.assertRaisesRegex(
                ValueError,
                "anywhere in the Dataset",
            ):
                compile_task_instance(plugin, dataset, _task(root))

    def test_runtime_revalidates_managed_contract_and_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, instance = self._compiled_fixture(root)

            contract_tamper = instance.value
            source_case = contract_tamper["source"]["cases"][0]
            source_case["assets"][
                "reference_video"
            ] = "assets/reference.mp4"
            adaptation = contract_tamper["adaptations"][0]
            adaptation["input_contract"][
                "asset_access"
            ] = ["reference_video"]
            adaptation["input_contract"]["media_channels"][0][
                "asset_key"
            ] = "reference_video"
            adaptation["native_inputs"]["vision"][
                "first_frame_asset"
            ] = "assets/reference.mp4"
            contract_tamper["inference"]["jobs"][0][
                "native_inputs"
            ] = copy.deepcopy(adaptation["native_inputs"])
            resealed = BaselineTaskInstance.seal(contract_tamper)
            with self.assertRaisesRegex(
                ValueError,
                "exposes evaluator/source assets",
            ):
                plugin.run_task(
                    instance=resealed,
                    run_dir=root / "contract-tamper",
                    execute=False,
                    stop_after_training=False,
                )

            recipe_tamper = instance.value
            recipe_tamper["inference"]["predictor"]["config"][
                "tampered"
            ] = True
            resealed = BaselineTaskInstance.seal(recipe_tamper)
            with self.assertRaisesRegex(
                ValueError,
                "differs from active Baseline runner recipe",
            ):
                plugin.run_task(
                    instance=resealed,
                    run_dir=root / "recipe-tamper",
                    execute=False,
                    stop_after_training=False,
                )

    def test_runtime_rejects_escaping_dataset_asset_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, instance = self._compiled_fixture(root)
            value = instance.value
            value["source"]["cases"][0]["assets"][
                "first_frame"
            ] = "../outside.png"
            adaptation = value["adaptations"][0]
            adaptation["native_inputs"]["vision"][
                "first_frame_asset"
            ] = "../outside.png"
            adaptation["native_inputs"]["media_contract"]["conditioning"][
                "asset"
            ] = "../outside.png"
            value["inference"]["jobs"][0][
                "native_inputs"
            ] = copy.deepcopy(adaptation["native_inputs"])
            resealed = BaselineTaskInstance.seal(value)
            with self.assertRaisesRegex(
                ValueError,
                "inside the Dataset asset root",
            ):
                plugin.run_task(
                    instance=resealed,
                    run_dir=root / "path-tamper",
                    execute=False,
                    stop_after_training=False,
                )

    def test_finetune_runtime_separates_targets_from_model_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root)
            manifest = _manifest(
                "fixture",
                adapter={
                    "kind": "python",
                    "entrypoint": "adapter.py",
                    "config": {},
                },
            )
            manifest["capabilities"]["task_families"] = ["finetune_eval"]
            manifest["capabilities"]["train"] = True
            manifest["capabilities"]["finetune"] = True
            manifest["trainer"] = {"type": "fixture_trainer_v1"}
            write_json(bundle_root / "baseline.json", manifest)
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )

            train_case = _case()
            train_case["case_id"] = "pendulum_train"
            eval_case = copy.deepcopy(_case())
            eval_case["case_id"] = "pendulum_eval"
            views = {
                "view_a": {
                    "schema_version": "3.0",
                    "coverage": "complete",
                    "scenes": {
                        "pendulum": {
                            "train": ["pendulum_train"],
                            "test": ["pendulum_eval"],
                        },
                    },
                    "test_annotations": {
                        "pendulum_eval": {
                            "generalization_regime": "id",
                            "ood_factors": [],
                        },
                    },
                },
                "view_b": {
                    "schema_version": "3.0",
                    "coverage": "complete",
                    "scenes": {"pendulum": {"group_1": []}},
                },
            }
            descriptor = {
                "schema_version": "3.0",
                "dataset_id": "conditioning_contract_fixture"
            }
            dataset = DatasetSnapshot(
                root=root,
                descriptor=descriptor,
                cases=(train_case, eval_case),
                views=views,
                scene_configs={
                    "pendulum": {
                        "schema_version": "3.0",
                        "scene_id": "pendulum",
                    },
                },
                asset_lock=None,
                digest=canonical_sha256({
                    "descriptor": descriptor,
                    "cases": [train_case, eval_case],
                    "views": views,
                }),
                asset_root=root,
            )
            task_value = {
                "schema_version": "1.0",
                "task_id": "fixture_finetune",
                "family": "finetune_eval",
                "dataset_id": "conditioning_contract_fixture",
                "selection": {
                    "training_scene_ids": ["pendulum"],
                    "evaluation_scene_ids": ["pendulum"],
                    "test_regimes": ["id"],
                },
                "seeds": {"training": [11], "inference": [7]},
                "evaluation": {
                    "protocol": "scene_default_v1",
                    "reporting": {
                        "primary_score": "overall_test",
                        "breakdowns": [],
                        "minimum_subgroup_jobs": 1,
                    },
                },
            }
            task = TaskSpec(
                path=root / "fixture_finetune.json",
                value=task_value,
                digest=canonical_sha256(task_value),
            )

            instance = compile_task_instance(plugin, dataset, task).value
            source = {
                case["case_id"]: case
                for case in instance["source"]["cases"]
            }
            self.assertIn("physics", source["pendulum_train"])
            self.assertIn("text", source["pendulum_train"])
            self.assertNotIn(
                "reference_video",
                source["pendulum_train"]["assets"],
            )
            self.assertEqual(
                "training_target_only",
                source["pendulum_train"]["supervised_targets"][
                    "video"
                ]["role"],
            )
            self.assertNotIn(
                "supervised_targets",
                source["pendulum_eval"],
            )


class TaskInstanceContractTests(unittest.TestCase):
    def test_missing_envelope_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root)
            plugin = load_baseline_plugin(
                load_baseline_bundle(bundle_root)
            )
            value = compile_task_instance(plugin,
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
            value = compile_task_instance(plugin,
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

    def test_inference_job_must_match_canonical_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = load_baseline_plugin(
                load_baseline_bundle(_create_bundle(root))
            )
            value = compile_task_instance(plugin,
                _dataset(root),
                _task(root),
            ).value
            value["inference"]["jobs"][0]["seed"] += 1
            with self.assertRaisesRegex(
                ValueError,
                "does not match canonical_plan job field.*seed",
            ):
                BaselineTaskInstance.seal(value)

    def test_inference_native_inputs_must_match_adaptation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = load_baseline_plugin(
                load_baseline_bundle(_create_bundle(root))
            )
            value = compile_task_instance(plugin,
                _dataset(root),
                _task(root),
            ).value
            value["inference"]["jobs"][0]["native_inputs"]["text"][
                "prompt"
            ] = "tampered"
            with self.assertRaisesRegex(
                ValueError,
                "must equal the referenced adaptation.native_inputs",
            ):
                BaselineTaskInstance.seal(value)

    def test_execution_graph_must_cover_inference_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = load_baseline_plugin(
                load_baseline_bundle(_create_bundle(root))
            )
            value = compile_task_instance(plugin,
                _dataset(root),
                _task(root),
            ).value
            for operation in value["execution_graph"]["operations"]:
                if operation["kind"] == "evaluate":
                    operation["job_ids"] = []
            with self.assertRaisesRegex(
                ValueError,
                "evaluate operations must cover every inference job",
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

    def test_i2v_job_spec_flag_is_opt_in_and_new_scaffolds_enable_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_baseline_scaffold(
                name="fixture_i2v",
                backend="managed-i2v",
                root=temporary,
            )
            bundle = load_baseline_bundle(root)
            config = bundle.value["runner"]["config"]
            self.assertEqual("--job-spec", config["job_spec_arg"])
            driver = StandardI2VCLIDriver(bundle)
            spec = {
                "prompt": "fixture",
                "first_frame": "/tmp/frame.png",
                "output_video": "/tmp/output.mp4",
                "seed": 7,
                "job_spec": "/tmp/job.json",
            }
            with mock.patch(
                "physbench.baseline_runtime.drivers.subprocess_i2v."
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                driver.execute_job(
                    spec,
                    log_path=Path(temporary) / "new.log",
                )
            command = run.call_args.args[0]
            self.assertIn("--job-spec", command)
            self.assertIn("/tmp/job.json", command)

            config.pop("job_spec_arg")
            with mock.patch(
                "physbench.baseline_runtime.drivers.subprocess_i2v."
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                driver.execute_job(
                    spec,
                    log_path=Path(temporary) / "legacy.log",
                )
            self.assertNotIn("--job-spec", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
