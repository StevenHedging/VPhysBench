from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physbench.cli import main
from physbench.baseline_runtime import create_baseline_scaffold
from physbench.evaluation import load_evaluation_protocol
from physbench.io import (
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from physbench.orchestration import reevaluate_atomic_variant
from physbench.orchestration.atomic_runner import run_atomic
from physbench.orchestration.evaluation_variants import (
    _load_asset_lock,
    _validate_cases,
    _validate_reference_assets,
)


def _asset_record(
    path: Path,
    *,
    asset_root: Path,
    case_ids: list[str],
    roles: list[str],
) -> dict:
    return {
        "path": path.relative_to(asset_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "case_ids": case_ids,
        "roles": roles,
    }


def _view(view_id: str, groups: dict[str, list[str]]) -> dict:
    case_ids = [
        case_id
        for members in groups.values()
        for case_id in members
    ]
    return {
        "schema_version": "2.0",
        "view_id": view_id,
        "coverage": "complete",
        "case_set_sha256": canonical_sha256(sorted(case_ids)),
        "scenes": {"pendulum": groups},
    }


def _build_atomic_run(container: Path, *, run_id: str = "fixture-run") -> Path:
    asset_root = container / "assets"
    dataset_root = asset_root / "releases" / "1.0.0"
    scene_root = dataset_root / "scenes"
    view_root = dataset_root / "views"
    media_root = asset_root / "fixture"
    for directory in (scene_root, view_root, media_root):
        directory.mkdir(parents=True, exist_ok=True)

    parent_frame = media_root / "parent.png"
    child_frame = media_root / "child.png"
    reference = media_root / "reference.mp4"
    parent_frame.write_bytes(b"fixture-parent-frame")
    child_frame.write_bytes(b"fixture-child-frame")
    reference.write_bytes(b"fixture-reference-video")

    physics = {
        "length": {
            "annotated": True,
            "unit": "m",
            "value": 0.8,
        },
        "gravity": {
            "annotated": True,
            "unit": "m/s^2",
            "value": 9.8,
        },
    }
    text = {
        "schema_version": "1.0",
        "prompt": "A laboratory pendulum swings from its initial position.",
        "language": "en",
        "annotation_source": "evaluation_variant_fixture",
    }
    parent = {
        "schema_version": "3.0",
        "case_id": "pendulum_parent",
        "scene_id": "pendulum",
        "text": text,
        "assets": {
            "first_frame": "fixture/parent.png",
            "physics_reference_video": "fixture/reference.mp4",
            "reference_video": "fixture/reference.mp4",
        },
        "physics": physics,
        "appearance": {
            "background": "fixture_lab",
            "apparatus": "fixture_stand",
        },
        "temporal": {"time_scale": "real_time"},
        "provenance": {
            "source_kind": "fixture",
            "parent_case_id": None,
        },
        "ood": {"level": "id", "factors": []},
        "has_real_reference_video": True,
    }
    child = {
        "schema_version": "3.0",
        "case_id": "pendulum_ood_child",
        "scene_id": "pendulum",
        "text": text,
        "assets": {
            "first_frame": "fixture/child.png",
            "physics_reference_video": "fixture/reference.mp4",
            "reference_video": None,
        },
        "physics": physics,
        "appearance": {
            "background": "fixture_ood_lab",
            "apparatus": "fixture_ood_stand",
        },
        "temporal": {"time_scale": "real_time"},
        "provenance": {
            "source_kind": "derived_ood",
            "parent_case_id": "pendulum_parent",
        },
        "ood": {
            "level": "ood1",
            "factors": ["background", "apparatus"],
        },
        "has_real_reference_video": False,
    }
    cases = [parent, child]
    case_ids = [case["case_id"] for case in cases]
    views = {
        "view_a": _view("view_a", {"test_id": case_ids}),
        "view_b": _view("view_b", {"group_1": case_ids}),
    }
    scene = {"schema_version": "1.0", "scene_id": "pendulum"}
    write_jsonl(dataset_root / "cases.jsonl", cases)
    write_json(scene_root / "pendulum.json", scene)
    write_json(view_root / "view_a.json", views["view_a"])
    write_json(view_root / "view_b.json", views["view_b"])

    files = sorted(
        [
            _asset_record(
                parent_frame,
                asset_root=asset_root,
                case_ids=["pendulum_parent"],
                roles=["first_frame"],
            ),
            _asset_record(
                child_frame,
                asset_root=asset_root,
                case_ids=["pendulum_ood_child"],
                roles=["first_frame"],
            ),
            _asset_record(
                reference,
                asset_root=asset_root,
                case_ids=case_ids,
                roles=[
                    "physics_reference_video",
                    "reference_video",
                ],
            ),
        ],
        key=lambda item: item["path"],
    )
    asset_lock = {
        "schema_version": "1.0",
        "dataset_id": "evaluation_variant_fixture",
        "release": "1.0.0",
        "files": files,
        "files_digest": canonical_sha256(files),
    }
    write_json(dataset_root / "assets.lock.json", asset_lock)
    descriptor = {
        "schema_version": "3.0",
        "dataset_id": "evaluation_variant_fixture",
        "release": "1.0.0",
        "cases": "cases.jsonl",
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "release_manifest": "release.json",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
    }
    write_json(dataset_root / "dataset.json", descriptor)
    dataset_digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": tuple(cases),
        "views": views,
        "scenes": {"pendulum": scene},
        "asset_lock": asset_lock,
    })
    write_json(dataset_root / "release.json", {
        "schema_version": "1.0",
        "dataset_id": descriptor["dataset_id"],
        "release": descriptor["release"],
        "dataset_digest": dataset_digest,
        "asset_files_digest": asset_lock["files_digest"],
    })
    task = {
        "schema_version": "3.0",
        "task_id": "evaluation_variant_direct",
        "family": "direct_eval",
        "dataset_id": descriptor["dataset_id"],
        "dataset_view": "view_b",
        "selection": {
            "scene_ids": ["pendulum"],
            "groups": "all",
            "case_ids": case_ids,
        },
        "ood2": {"enabled": False},
        "seeds": {"training": [], "inference": [42]},
        "evaluation": {"protocol": "scene_default_v1"},
    }
    task_path = container / "task.json"
    write_json(task_path, task)
    baseline = create_baseline_scaffold(
        name="evaluation_variant_submission",
        backend="submission",
        root=container / "baselines",
    )
    return run_atomic(
        dataset_path=dataset_root / "dataset.json",
        task_path=task_path,
        baseline_path=baseline,
        output_root=container / "runs",
        run_id=run_id,
        execute=False,
    )


def _canonical_snapshot(run_dir: Path) -> dict[str, bytes]:
    paths = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and "reevaluations" not in path.relative_to(run_dir).parts
    ]
    return {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in paths
    }


def _assert_snapshot_unchanged(
    testcase: unittest.TestCase,
    run_dir: Path,
    expected: dict[str, bytes],
) -> None:
    observed = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and "reevaluations" not in path.relative_to(run_dir).parts
    }
    testcase.assertEqual(expected, observed)


def _variant_destination(
    run_dir: Path,
    *,
    evaluation_id: str,
    protocol_id: str = "scene_default_v2",
) -> Path:
    protocol = load_evaluation_protocol(protocol_id)
    return (
        run_dir
        / "reevaluations"
        / protocol_id
        / protocol["fingerprint"]
        / evaluation_id
    )


class EvaluationVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sam_patch = patch(
            "physbench.orchestration.evaluation_variants."
            "_sam_model_integrity",
            return_value={
                "models": [],
                "models_digest": canonical_sha256([]),
            },
        )
        self.sam_patch.start()

    def tearDown(self) -> None:
        self.sam_patch.stop()
        self.temporary.cleanup()

    def test_partial_variant_is_coexisting_and_fully_manifested(self) -> None:
        run_dir = _build_atomic_run(self.root)
        canonical = _canonical_snapshot(run_dir)
        result = reevaluate_atomic_variant(
            run_dir,
            protocol_id="scene_default_v2",
            evaluation_id="protocol-v2-audit",
        )

        destination = _variant_destination(
            run_dir,
            evaluation_id="protocol-v2-audit",
        )
        self.assertEqual(str(destination), result["variant_path"])
        self.assertEqual("complete", result["reevaluation"]["workflow_status"])
        self.assertEqual("partial", result["task_result"]["status"])
        self.assertEqual(0.0, result["task_result"]["coverage"])
        _assert_snapshot_unchanged(self, run_dir, canonical)

        references = load_json(destination / "reference_assets.json")
        self.assertEqual(0, references["unique_reference_assets"])
        self.assertEqual([], references["records"])

        source = load_json(destination / "source_integrity.json")
        self.assertEqual(
            "scene_default_v1",
            source["native_protocol"]["id"],
        )
        self.assertEqual(
            "scene_default_v2",
            source["target_protocol"]["id"],
        )
        self.assertIn("tree_sha256", source["evaluator_source"])
        self.assertIn("git", source["evaluator_source"])
        self.assertIn("python", source["dependencies"])
        self.assertEqual(
            "recomputed_release_digest_matches_sealed_task_instance",
            source["dataset_authentication"]["policy"],
        )
        self.assertEqual(
            "self_consistent_runtime_output_not_externally_sealed",
            source["prediction_authentication"]["authenticity_boundary"],
        )

        manifest = load_json(destination / "artifact_manifest.json")
        self.assertNotIn(
            "artifact_manifest.json",
            {item["path"] for item in manifest["files"]},
        )
        for item in manifest["files"]:
            path = destination / item["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(sha256_file(path), item["sha256"])
        self.assertEqual(
            canonical_sha256(manifest["files"]),
            manifest["files_digest"],
        )

    def test_duplicate_and_unsafe_ids_are_rejected(self) -> None:
        run_dir = _build_atomic_run(self.root)
        reevaluate_atomic_variant(
            run_dir,
            protocol_id="scene_default_v2",
            evaluation_id="one",
        )
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="scene_default_v2",
                evaluation_id="one",
            )
        for invalid in ("../escape", "nested/id", "."):
            with self.subTest(evaluation_id=invalid):
                with self.assertRaises(ValueError):
                    reevaluate_atomic_variant(
                        run_dir,
                        protocol_id="scene_default_v2",
                        evaluation_id=invalid,
                    )
        with self.assertRaises(ValueError):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="../scene_default_v2",
                evaluation_id="two",
            )

    def test_reevaluations_symlink_is_rejected(self) -> None:
        run_dir = _build_atomic_run(self.root)
        external = self.root / "external-variants"
        external.mkdir()
        os.symlink(external, run_dir / "reevaluations")
        with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="scene_default_v2",
                evaluation_id="symlink-attempt",
            )
        self.assertEqual([], list(external.iterdir()))

    def test_canonical_source_symlink_is_rejected_preflight(self) -> None:
        run_dir = _build_atomic_run(self.root)
        report = run_dir / "report.md"
        external = self.root / "external-report.md"
        external.write_bytes(report.read_bytes())
        report.unlink()
        os.symlink(external, report)
        with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="scene_default_v2",
                evaluation_id="must-not-exist",
            )
        self.assertFalse((run_dir / "reevaluations").exists())

    def test_sealed_plan_and_task_tampering_fail_before_creation(self) -> None:
        mutations = {
            "sealed": self._tamper_sealed_instance,
            "plan": self._tamper_plan,
            "task": self._tamper_task,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                run_dir = _build_atomic_run(
                    self.root / label,
                    run_id=f"fixture-{label}",
                )
                mutate(run_dir)
                with self.assertRaises(ValueError):
                    reevaluate_atomic_variant(
                        run_dir,
                        protocol_id="scene_default_v2",
                        evaluation_id="must-not-exist",
                    )
                self.assertFalse((run_dir / "reevaluations").exists())

    def test_missing_external_release_authenticity_anchor_is_rejected(
        self,
    ) -> None:
        run_dir = _build_atomic_run(self.root)
        (
            self.root
            / "assets"
            / "releases"
            / "1.0.0"
            / "dataset.json"
        ).unlink()
        with self.assertRaisesRegex(
            ValueError,
            "cannot authenticate frozen Dataset",
        ):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="scene_default_v2",
                evaluation_id="must-not-exist",
            )
        self.assertFalse((run_dir / "reevaluations").exists())

    def test_prediction_component_and_native_tampering_are_rejected(self) -> None:
        mutations = {
            "predictions": self._tamper_predictions,
            "prediction_manifest": self._tamper_prediction_manifest,
            "component": self._tamper_component,
            "native_results": self._tamper_native_results,
            "native_protocol": self._tamper_native_protocol,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                run_dir = _build_atomic_run(
                    self.root / label,
                    run_id=f"fixture-{label}",
                )
                mutate(run_dir)
                with self.assertRaises(ValueError):
                    reevaluate_atomic_variant(
                        run_dir,
                        protocol_id="scene_default_v2",
                        evaluation_id="must-not-exist",
                    )
                self.assertFalse((run_dir / "reevaluations").exists())

    def test_sealed_dataset_and_native_aggregation_reject_coordinated_tampering(
        self,
    ) -> None:
        mutations = {
            "frozen_cases": (
                self._tamper_frozen_cases,
                "frozen Dataset files differ",
            ),
            "native_aggregation": (
                self._tamper_native_result_every_projection,
                "fresh aggregation",
            ),
            "reference_and_lock": (
                self._tamper_reference_and_frozen_lock,
                "frozen Dataset files differ",
            ),
        }
        for label, (mutate, expected) in mutations.items():
            with self.subTest(label=label):
                container = self.root / label
                run_dir = _build_atomic_run(
                    container,
                    run_id=f"fixture-{label}",
                )
                mutate(run_dir, container)
                with self.assertRaisesRegex(ValueError, expected):
                    reevaluate_atomic_variant(
                        run_dir,
                        protocol_id="scene_default_v2",
                        evaluation_id="must-not-exist",
                    )
                self.assertFalse((run_dir / "reevaluations").exists())

    def test_incomplete_predictions_do_not_preflight_unused_references(
        self,
    ) -> None:
        run_dir = _build_atomic_run(self.root)
        reference = self.root / "assets" / "fixture" / "reference.mp4"
        original = reference.read_bytes()
        reference.write_bytes(b"x" * len(original))
        result = reevaluate_atomic_variant(
            run_dir,
            protocol_id="scene_default_v2",
            evaluation_id="unused-reference",
        )
        self.assertEqual("partial", result["task_result"]["status"])
        references = load_json(
            Path(result["variant_path"]) / "reference_assets.json"
        )
        self.assertEqual(0, references["unique_reference_assets"])

    def test_complete_predictions_preflight_consumed_parent_reference(
        self,
    ) -> None:
        run_dir = _build_atomic_run(self.root)
        plan = load_json(run_dir / "plan.json")
        cases = load_jsonl(run_dir / "frozen" / "cases.jsonl")
        catalog = _validate_cases(cases, plan)
        dataset = load_json(run_dir / "frozen" / "dataset.json")
        _, locked = _load_asset_lock(run_dir, dataset=dataset)
        predictions = load_jsonl(run_dir / "predictions.jsonl")
        for prediction in predictions:
            prediction["status"] = "complete"
        reference = self.root / "assets" / "fixture" / "reference.mp4"
        reference.write_bytes(b"x" * reference.stat().st_size)

        with self.assertRaisesRegex(
            ValueError,
            "reference asset SHA-256 differs",
        ):
            _validate_reference_assets(
                plan=plan,
                predictions=predictions,
                protocol=load_evaluation_protocol("scene_default_v2"),
                catalog=catalog,
                asset_root=self.root / "assets",
                locked_by_path=locked,
            )

    def test_workflow_failure_is_audited_without_canonical_mutation(
        self,
    ) -> None:
        run_dir = _build_atomic_run(self.root)
        canonical = _canonical_snapshot(run_dir)
        destination = _variant_destination(
            run_dir,
            evaluation_id="failed-workflow",
        )

        def fail_after_partial_output(**kwargs: object) -> None:
            output_dir = Path(str(kwargs["output_dir"]))
            write_json(
                output_dir / "partial_summary.json",
                {"status": "partial-before-failure"},
            )
            raise RuntimeError("fixture evaluator failure")

        with patch(
            "physbench.orchestration.evaluation_variants.evaluate_task",
            side_effect=fail_after_partial_output,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "fixture evaluator failure",
            ):
                reevaluate_atomic_variant(
                    run_dir,
                    protocol_id="scene_default_v2",
                    evaluation_id="failed-workflow",
                )

        self.assertTrue(destination.is_dir())
        record = load_json(destination / "reevaluation.json")
        self.assertEqual("failed", record["workflow_status"])
        self.assertEqual("RuntimeError", record["error"]["type"])
        self.assertTrue(
            (destination / "artifact_manifest.json").is_file()
        )
        self.assertEqual(
            {"status": "partial-before-failure"},
            load_json(
                destination
                / "evaluation"
                / "partial_summary.json"
            ),
        )
        manifest = load_json(destination / "artifact_manifest.json")
        self.assertIn(
            "evaluation/partial_summary.json",
            {item["path"] for item in manifest["files"]},
        )
        _assert_snapshot_unchanged(self, run_dir, canonical)
        with self.assertRaises(FileExistsError):
            reevaluate_atomic_variant(
                run_dir,
                protocol_id="scene_default_v2",
                evaluation_id="failed-workflow",
            )

    def test_cli_requires_pair_and_creates_variant(self) -> None:
        run_dir = _build_atomic_run(self.root)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main([
                "evaluate",
                "--run-dir",
                str(run_dir),
            ])
        self.assertEqual(2, status)
        self.assertIn("--protocol-id", stderr.getvalue())

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main([
                "evaluate",
                "--run-dir",
                str(run_dir),
                "--protocol-id",
                "scene_default_v2",
                "--evaluation-id",
                "cli-evaluation",
            ])
        self.assertEqual(0, status)
        self.assertIn('"workflow_status": "complete"', stdout.getvalue())
        self.assertTrue(
            _variant_destination(
                run_dir,
                evaluation_id="cli-evaluation",
            ).is_dir()
        )

    def test_cli_never_falls_back_when_atomic_manifest_is_missing(self) -> None:
        run_dir = _build_atomic_run(self.root)
        (run_dir / "task_instance" / "manifest.json").unlink()
        stderr = io.StringIO()
        with (
            patch("physbench.cli.reevaluate_run") as legacy,
            contextlib.redirect_stderr(stderr),
        ):
            status = main([
                "evaluate",
                "--run-dir",
                str(run_dir),
            ])
        self.assertEqual(2, status)
        self.assertIn("--protocol-id", stderr.getvalue())
        legacy.assert_not_called()

    @staticmethod
    def _tamper_sealed_instance(run_dir: Path) -> None:
        path = run_dir / "task_instance" / "manifest.json"
        value = load_json(path)
        value["semantics"]["scene_ids"].append("invented")
        write_json(path, value)

    @staticmethod
    def _tamper_plan(run_dir: Path) -> None:
        path = run_dir / "plan.json"
        value = load_json(path)
        value["jobs"][0]["seed"] = 7
        write_json(path, value)

    @staticmethod
    def _tamper_task(run_dir: Path) -> None:
        path = run_dir / "frozen" / "task.json"
        value = load_json(path)
        value["task_id"] = "modified-task"
        write_json(path, value)

    @staticmethod
    def _tamper_predictions(run_dir: Path) -> None:
        path = run_dir / "predictions.jsonl"
        value = load_jsonl(path)
        value[0]["seed"] = 7
        write_jsonl(path, value)

    @staticmethod
    def _tamper_prediction_manifest(run_dir: Path) -> None:
        path = run_dir / "artifacts" / "prediction_artifacts.json"
        value = load_json(path)
        value["policy"]["external_prediction_references_allowed"] = True
        write_json(path, value)

    @staticmethod
    def _tamper_component(run_dir: Path) -> None:
        path = run_dir / "component_fingerprints.json"
        value = load_json(path)
        value["dataset"] = "0" * 64
        write_json(path, value)

    @staticmethod
    def _tamper_native_results(run_dir: Path) -> None:
        for name in ("case_results.jsonl", "case_metrics.jsonl"):
            path = run_dir / "evaluation" / name
            value = load_jsonl(path)
            write_jsonl(path, value[:-1])

    @staticmethod
    def _tamper_native_protocol(run_dir: Path) -> None:
        for name in ("task_result.json", "summary.json"):
            path = run_dir / "evaluation" / name
            value = load_json(path)
            value["protocol"]["fingerprint"] = "0" * 64
            write_json(path, value)

    @staticmethod
    def _tamper_frozen_cases(run_dir: Path, _container: Path) -> None:
        path = run_dir / "frozen" / "cases.jsonl"
        cases = load_jsonl(path)
        for case in cases:
            case["physics"]["gravity"]["value"] = 1.234
        write_jsonl(path, cases)

    @staticmethod
    def _tamper_native_result_every_projection(
        run_dir: Path,
        _container: Path,
    ) -> None:
        job_id = load_json(run_dir / "plan.json")["jobs"][0]["job_id"]
        for path in (
            run_dir / "evaluation" / "case_results.jsonl",
            run_dir / "evaluation" / "case_metrics.jsonl",
        ):
            records = load_jsonl(path)
            records[0].update({
                "status": "evaluated",
                "score": 0.999,
                "reason_code": None,
                "reason": None,
            })
            write_jsonl(path, records)
        artifact = (
            run_dir
            / "evaluation"
            / "cases"
            / job_id
            / "result.json"
        )
        result = load_json(artifact)
        result.update({
            "status": "evaluated",
            "score": 0.999,
            "reason_code": None,
            "reason": None,
        })
        write_json(artifact, result)

    @staticmethod
    def _tamper_reference_and_frozen_lock(
        run_dir: Path,
        container: Path,
    ) -> None:
        reference = container / "assets" / "fixture" / "reference.mp4"
        reference.write_bytes(b"x" * reference.stat().st_size)
        path = run_dir / "frozen" / "assets.lock.json"
        lock = load_json(path)
        record = next(
            item
            for item in lock["files"]
            if item["path"] == "fixture/reference.mp4"
        )
        record["sha256"] = sha256_file(reference)
        lock["files_digest"] = canonical_sha256(lock["files"])
        write_json(path, lock)


if __name__ == "__main__":
    unittest.main()
