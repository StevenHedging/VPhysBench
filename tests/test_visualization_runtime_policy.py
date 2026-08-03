from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physbench.cli import build_parser
from physbench.evaluation.contracts import (
    CaseEvaluationRequest,
    CaseEvaluationResult,
)
from physbench.evaluation.task_evaluator import evaluate_task
from physbench.orchestration.evaluation_variants import _artifact_manifest


class _CapturingEvaluator:
    def __init__(self) -> None:
        self.requests = []

    def describe(self):
        return {"id": "capture", "version": "1.0", "scene_id": "scene"}

    def evaluate(self, request):
        self.requests.append(request)
        return CaseEvaluationResult(
            job_id=request.job["job_id"],
            case_id=request.case["case_id"],
            scene_id="scene",
            evaluator=self.describe(),
            status="evaluated",
            score=1.0,
        )


class _CapturingRegistry:
    def __init__(self, evaluator: _CapturingEvaluator) -> None:
        self.evaluator = evaluator

    def resolve(self, _scene_id):
        return self.evaluator

    def describe(self):
        return {"scene": self.evaluator.describe()}


class VisualizationRuntimePolicyTest(unittest.TestCase):
    def test_atomic_cli_defaults_off_and_explicit_flag_enables(self) -> None:
        parser = build_parser()
        base = [
            "atomic-run",
            "--dataset",
            "dataset.json",
            "--task",
            "task.json",
            "--baseline",
            "baseline.json",
        ]
        self.assertFalse(parser.parse_args(base).save_visualizations)
        self.assertTrue(
            parser.parse_args([*base, "--save-visualizations"])
            .save_visualizations
        )

        matrix = [
            "matrix-run",
            "--dataset",
            "dataset.json",
            "--task",
            "task.json",
            "--baseline",
            "baseline-a.json",
            "--baseline",
            "baseline-b.json",
            "--matrix-id",
            "matrix",
        ]
        self.assertFalse(parser.parse_args(matrix).save_visualizations)
        self.assertTrue(
            parser.parse_args([*matrix, "--save-visualizations"])
            .save_visualizations
        )

        evaluate = [
            "evaluate",
            "--run-dir",
            "run",
            "--protocol-id",
            "protocol",
            "--evaluation-id",
            "evaluation",
        ]
        self.assertFalse(parser.parse_args(evaluate).save_visualizations)
        self.assertTrue(
            parser.parse_args([*evaluate, "--save-visualizations"])
            .save_visualizations
        )

    def test_case_request_contract_also_defaults_off(self) -> None:
        request = CaseEvaluationRequest(
            job={},
            case={},
            case_catalog={},
            prediction=None,
            asset_root=Path("."),
            artifact_dir=Path("."),
            evaluator_config={},
        )
        self.assertFalse(request.save_visualizations)

    def test_task_evaluator_propagates_run_scoped_policy(self) -> None:
        plan = {
            "task_id": "task",
            "family": "direct_eval",
            "scene_ids": ["scene"],
            "jobs": [
                {
                    "job_id": "job",
                    "case_id": "case",
                    "scene_id": "scene",
                    "evaluation_partition": "test",
                    "seed": 42,
                }
            ],
        }
        cases = [{"case_id": "case", "scene_id": "scene"}]
        predictions = [{"job_id": "job", "status": "complete"}]
        protocol = {
            "protocol_id": "protocol",
            "fingerprint": "f" * 64,
            "path": "/protocol.json",
            "scenes": {"scene": {"type": "capture"}},
        }
        evaluator = _CapturingEvaluator()
        with tempfile.TemporaryDirectory() as temporary:
            evaluate_task(
                plan=plan,
                cases=cases,
                predictions=predictions,
                asset_root=temporary,
                protocol=protocol,
                output_dir=Path(temporary) / "evaluation",
                registry=_CapturingRegistry(evaluator),
                run_id="baseline_task_run",
            )
            self.assertEqual("baseline_task_run", evaluator.requests[-1].run_id)
            self.assertFalse(evaluator.requests[-1].save_visualizations)
            self.assertEqual(
                (Path(temporary) / "evaluation" / "visualizations").resolve(),
                evaluator.requests[-1].visualization_root.resolve(),
            )

            evaluate_task(
                plan=plan,
                cases=cases,
                predictions=predictions,
                asset_root=temporary,
                protocol=protocol,
                output_dir=Path(temporary) / "enabled_evaluation",
                registry=_CapturingRegistry(evaluator),
                run_id="baseline_task_run",
                save_visualizations=True,
            )
            self.assertTrue(evaluator.requests[-1].save_visualizations)
            self.assertEqual(
                (
                    Path(temporary)
                    / "enabled_evaluation"
                    / "visualizations"
                ).resolve(),
                evaluator.requests[-1].visualization_root.resolve(),
            )

    def test_reevaluation_manifest_seals_run_owned_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = (
                root
                / "evaluation"
                / "visualizations"
                / "scene"
                / "case"
                / "evaluation-deadbeef"
                / "visualization.mp4"
            )
            video.parent.mkdir(parents=True)
            video.write_bytes(b"run-owned-visualization")
            manifest = _artifact_manifest(root)
            paths = {record["path"] for record in manifest["files"]}
            self.assertIn(
                "evaluation/visualizations/scene/case/"
                "evaluation-deadbeef/visualization.mp4",
                paths,
            )


if __name__ == "__main__":
    unittest.main()
