from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physbench.baseline_api import (
    discover_baseline_bundles,
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.data_layout import V2_DATASET
from physbench.datasets import load_dataset_v2
from physbench.io import canonical_sha256, load_json, write_json
from physbench.tasks import load_task_v2

from _paths import ROOT


MOCK_COMMAND = r'''
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "physbench-baseline-v1"
TASK_BUILDER_FP = "a" * 64
DATA_ADAPTER_FP = "b" * 64
MATERIALIZATION_FP = "c" * 64


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal(value):
    document = dict(value)
    document.pop("instance_digest", None)
    document["instance_digest"] = canonical_sha256(document)
    return document


def dispatch(request):
    operation = request["operation"]
    payload = request["payload"]
    bundle = request["bundle"]
    if operation == "describe":
        return {
            "task_builder": {
                "type": "fixture_builder",
                "fingerprint": TASK_BUILDER_FP,
            },
            "data_adapter": {
                "type": "fixture_adapter",
                "fingerprint": DATA_ADAPTER_FP,
                "materialization_fingerprint": MATERIALIZATION_FP,
            },
        }
    if operation == "adapt_case":
        return {
            "case_id": payload["case"]["case_id"],
            "role": payload["role"],
            "conditioning": payload["conditioning"],
            "used_parameters": {},
            "native_inputs": {"fixture": True},
        }
    if operation == "build_task_instance":
        dataset = payload["dataset"]
        task = payload["task"]
        plan = payload["canonical_plan"]
        return seal({
            "schema_version": "2.1",
            "instance_id": "fixture-instance",
            "identity": {
                "dataset": {
                    "dataset_id": dataset["descriptor"]["dataset_id"],
                    "digest": dataset["digest"],
                },
                "task": {
                    "task_id": task["value"]["task_id"],
                    "digest": task["digest"],
                },
                "baseline": {
                    "baseline_id": bundle["value"]["baseline_id"],
                    "baseline_version": bundle["value"]["baseline_version"],
                    "digest": bundle["digest"],
                    "deployment_digest": bundle["deployment_digest"],
                },
                "task_builder": {
                    "type": "fixture_builder",
                    "fingerprint": TASK_BUILDER_FP,
                },
                "data_adapter": {
                    "fingerprint": DATA_ADAPTER_FP,
                    "materialization_fingerprint": MATERIALIZATION_FP,
                },
                "canonical_plan_digest": canonical_sha256(plan),
            },
            "canonical_plan": plan,
        })
    if operation == "run_task":
        return {
            "training": {"status": "not_requested"},
            "predictions": [],
        }
    raise ValueError(operation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    response = {
        "protocol": PROTOCOL,
        "operation": request["operation"],
        "ok": True,
        "result": dispatch(request),
    }
    Path(args.response).write_text(
        json.dumps(response), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
'''


def _manifest(baseline_id: str) -> dict:
    return {
        "schema_version": "3.0",
        "baseline_id": baseline_id,
        "baseline_version": "0.1.0",
        "implementation": {
            "kind": "command",
            "protocol": "physbench-baseline-v1",
            "entrypoint": ["{python}", "plugin/main.py"],
            "fingerprint_paths": ["plugin/*.py"],
        },
        "capabilities": {
            "task_families": ["finetune_eval", "direct_eval"],
            "conditioning": ["generic", "physics"],
        },
        "runtime": {"device": "cpu"},
        "model": {"checkpoint": None},
    }


def _create_bundle(root: Path, name: str, baseline_id: str) -> Path:
    bundle = root / name
    plugin = bundle / "plugin"
    plugin.mkdir(parents=True)
    (plugin / "main.py").write_text(MOCK_COMMAND, encoding="utf-8")
    write_json(bundle / "baseline.json", _manifest(baseline_id))
    return bundle


class BaselineBundleV3Tests(unittest.TestCase):
    def test_directory_discovery_and_id_resolution_require_no_registry_edit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture_command")
            discovered = discover_baseline_bundles(root)
            self.assertEqual(
                bundle_root / "baseline.json",
                discovered["fixture_command"],
            )
            bundle = load_baseline_bundle(
                "fixture_command", baselines_root=root
            )
            plugin = load_baseline_plugin(bundle)
            self.assertEqual("fixture_command", bundle.baseline_id)
            self.assertEqual("a" * 64, plugin.task_builder.fingerprint)

    def test_duplicate_baseline_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _create_bundle(root, "first", "duplicate")
            _create_bundle(root, "second", "duplicate")
            with self.assertRaisesRegex(ValueError, "duplicate baseline_id"):
                discover_baseline_bundles(root)

    def test_implementation_change_changes_portable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            first = load_baseline_bundle(bundle_root)
            script = bundle_root / "plugin" / "main.py"
            script.write_text(
                script.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )
            second = load_baseline_bundle(bundle_root)
            self.assertNotEqual(first.digest, second.digest)

    def test_local_override_changes_only_deployment_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            first = load_baseline_bundle(bundle_root)
            write_json(bundle_root / "baseline.local.json", {
                "runtime": {"device": "cuda:7"},
                "model": {"checkpoint": "/models/fixture.bin"},
            })
            second = load_baseline_bundle(bundle_root)
            self.assertEqual(first.digest, second.digest)
            self.assertNotEqual(
                first.deployment_digest, second.deployment_digest
            )
            self.assertEqual("cuda:7", second.value["runtime"]["device"])

    def test_entrypoint_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            value = load_json(bundle_root / "baseline.json")
            value["implementation"]["entrypoint"] = [
                "{python}",
                "../outside.py",
            ]
            write_json(bundle_root / "baseline.json", value)
            with self.assertRaisesRegex(ValueError, "bundle-relative"):
                load_baseline_bundle(bundle_root)

    def test_command_builder_cannot_replace_canonical_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            bundle = load_baseline_bundle(bundle_root)
            plugin = load_baseline_plugin(bundle)
            dataset = load_dataset_v2(V2_DATASET, check_assets=False)
            task = load_task_v2(
                ROOT / "tasks" / "official" / "direct_eval_generic.json"
            )
            before = canonical_sha256({
                "descriptor": dataset.descriptor,
                "cases": dataset.cases,
                "views": dataset.views,
            })
            instance = plugin.task_builder.build(dataset, task)
            after = canonical_sha256({
                "descriptor": dataset.descriptor,
                "cases": dataset.cases,
                "views": dataset.views,
            })
            self.assertEqual(before, after)
            self.assertEqual(
                canonical_sha256(instance.value["canonical_plan"]),
                instance.value["identity"]["canonical_plan_digest"],
            )

    def test_host_rejects_command_that_replaces_canonical_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            script = bundle_root / "plugin" / "main.py"
            value = script.read_text(encoding="utf-8").replace(
                'plan = payload["canonical_plan"]',
                'plan = dict(payload["canonical_plan"])\n'
                '        plan["jobs"] = []',
            )
            script.write_text(value, encoding="utf-8")
            bundle = load_baseline_bundle(bundle_root)
            plugin = load_baseline_plugin(bundle)
            dataset = load_dataset_v2(V2_DATASET, check_assets=False)
            task = load_task_v2(
                ROOT / "tasks" / "official" / "direct_eval_generic.json"
            )
            with self.assertRaisesRegex(
                RuntimeError, "changed the frozen build identity"
            ):
                plugin.task_builder.build(dataset, task)

    def test_local_override_rejects_contract_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_root = _create_bundle(root, "fixture", "fixture")
            write_json(bundle_root / "baseline.local.json", {
                "capabilities": {"conditioning": ["generic"]},
            })
            with self.assertRaisesRegex(ValueError, "may only override"):
                load_baseline_bundle(bundle_root)


if __name__ == "__main__":
    unittest.main()
