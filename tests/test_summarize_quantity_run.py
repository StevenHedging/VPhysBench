from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_quantity_run import (
    OUTPUT_JSON,
    OUTPUT_MARKDOWN,
    _canonical_sha256,
    _file_sha256,
    main,
    summarize_run,
)


class QuantityRunSummaryTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, values: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )

    def _fixture_run(self, root: Path) -> Path:
        run_dir = root / "atomic_run"
        task = {
            "schema_version": "3.0",
            "task_id": "quantity_fixture_task",
            "family": "finetune_eval",
            "selection": {
                "scene_ids": ["pendulum", "free_fall"],
                "eval_partitions": ["test_id", "test_ood1"],
            },
        }
        jobs = [
            {
                "job_id": "pendulum_id_1",
                "case_id": "pendulum_case_1",
                "scene_id": "pendulum",
                "evaluation_partition": "test_id",
                "seed": 42,
            },
            {
                "job_id": "pendulum_id_2",
                "case_id": "pendulum_case_2",
                "scene_id": "pendulum",
                "evaluation_partition": "test_id",
                "seed": 42,
            },
            {
                "job_id": "pendulum_ood_unavailable",
                "case_id": "pendulum_case_3",
                "scene_id": "pendulum",
                "evaluation_partition": "test_ood1",
                "seed": 42,
            },
            {
                "job_id": "pendulum_ood_missing",
                "case_id": "pendulum_case_4",
                "scene_id": "pendulum",
                "evaluation_partition": "test_ood1",
                "seed": 42,
            },
            {
                "job_id": "free_fall_id_failed",
                "case_id": "free_fall_case_1",
                "scene_id": "free_fall",
                "evaluation_partition": "test_id",
                "seed": 42,
            },
        ]
        plan = {
            "task_id": "quantity_fixture_task",
            "family": "finetune_eval",
            "scene_ids": ["pendulum", "free_fall"],
            "train_case_ids": [],
            "jobs": jobs,
        }
        run = {
            "schema_version": "2.0",
            "run_id": "quantity_fixture_run",
            "status": "complete",
            "dataset_id": "fixture_dataset",
            "dataset_digest": "dataset-digest",
            "task_id": "quantity_fixture_task",
            "task_family": "finetune_eval",
            "baseline_id": "wan22_quantity_fixture",
            "baseline_version": "1.0.0",
            "baseline_digest": "baseline-digest",
            "baseline_deployment_digest": "deployment-digest",
            "task_instance_id": "fixture_instance",
            "task_instance_digest": "instance-digest",
        }
        predictions = [
            {
                **jobs[0],
                "baseline_id": "wan22_quantity_fixture",
                "status": "complete",
            },
            {
                **jobs[1],
                "baseline_id": "wan22_quantity_fixture",
                "status": "complete",
            },
            {
                **jobs[2],
                "baseline_id": "wan22_quantity_fixture",
                "status": "complete",
            },
            {
                **jobs[4],
                "baseline_id": "wan22_quantity_fixture",
                "status": "failed",
            },
        ]
        case_results = [
            {
                **jobs[0],
                "status": "evaluated",
                "score": 0.2,
            },
            {
                **jobs[1],
                "status": "evaluated",
                "score": 0.4,
            },
            {
                **jobs[2],
                "status": "unavailable",
                "score": None,
                "reason_code": "subject_not_found",
            },
            {
                **jobs[4],
                "status": "error",
                "score": None,
                "reason_code": "video_decode_failed",
            },
        ]
        task_result = {
            "schema_version": "1.0",
            "task_id": "quantity_fixture_task",
            "task_family": "finetune_eval",
            "status": "partial",
            "coverage": 0.4,
            "score": None,
            "observed_mean_score": 0.3,
            "aggregation_policy": "strict_complete_coverage",
        }
        components = {
            "dataset": "dataset-digest",
            "task": _canonical_sha256(task),
            "baseline": "baseline-digest",
            "baseline_deployment": "deployment-digest",
            "task_instance": "instance-digest",
        }

        self._write_json(run_dir / "run.json", run)
        self._write_json(run_dir / "plan.json", plan)
        self._write_json(run_dir / "frozen" / "task.json", task)
        self._write_json(
            run_dir / "component_fingerprints.json",
            components,
        )
        self._write_jsonl(run_dir / "predictions.jsonl", predictions)
        self._write_jsonl(
            run_dir / "evaluation" / "case_results.jsonl",
            case_results,
        )
        self._write_json(
            run_dir / "evaluation" / "task_result.json",
            task_result,
        )

        checkpoint = (
            run_dir
            / "artifacts"
            / "wan22"
            / "checkpoints"
            / "step-1450.safetensors"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"combined quantity checkpoint")
        self._write_json(
            run_dir / "artifacts" / "wan22" / "checkpoint.json",
            {
                "schema_version": "2.0",
                "status": "complete",
                "source": "task1_finetune",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "inventory": {
                    "lora_tensor_count": 12,
                    "quantity_encoder_tensor_count": 8,
                },
            },
        )
        self._write_json(
            run_dir
            / "artifacts"
            / "wan22"
            / "loss_analysis"
            / "loss_summary.json",
            {
                "recorded_steps": 1450,
                "expected_total_steps": 1450,
                "mean": 0.1,
                "median": 0.08,
                "first_100_mean": 0.15,
                "last_100_mean": 0.05,
                "last_over_first_100_mean": 1 / 3,
            },
        )
        self._write_json(
            checkpoint.parent / "gradient_audit.json",
            {
                "sample_count": 1450,
                "positive_quantity_gradient_count": 1450,
                "min_positive_quantity_gradient_l2": 0.01,
                "max_quantity_gradient_l2": 1.5,
                "text_encoder_gradient_tensor_count_max": 0,
                "samples": [],
            },
        )
        return run_dir

    @staticmethod
    def _row(
        summary: dict,
        scene_id: str,
        partition: str,
    ) -> dict:
        return next(
            row
            for row in summary["scene_partition"]
            if row["scene_id"] == scene_id
            and row["partition"] == partition
        )

    def test_summary_preserves_na_and_never_imputes_missing_as_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))

            first = summarize_run(run_dir)
            second = summarize_run(run_dir)

            pendulum_id = self._row(first, "pendulum", "test_id")
            self.assertEqual(2, pendulum_id["expected"])
            self.assertEqual(2, pendulum_id["completed"])
            self.assertEqual(2, pendulum_id["evaluated"])
            self.assertAlmostEqual(0.3, pendulum_id["mean"])
            self.assertAlmostEqual(0.3, pendulum_id["median"])
            self.assertAlmostEqual(0.1, pendulum_id["std"])
            self.assertEqual(
                pendulum_id["ci95"],
                self._row(second, "pendulum", "test_id")["ci95"],
            )

            pendulum_ood = self._row(first, "pendulum", "test_ood1")
            self.assertEqual(2, pendulum_ood["expected"])
            self.assertEqual(1, pendulum_ood["completed"])
            self.assertEqual(0, pendulum_ood["evaluated"])
            self.assertEqual(1, pendulum_ood["unavailable"])
            self.assertEqual(1, pendulum_ood["missing_prediction"])
            self.assertEqual(1, pendulum_ood["missing_evaluation"])
            self.assertIsNone(pendulum_ood["mean"])
            self.assertIsNone(pendulum_ood["ci95"])

            free_fall_id = self._row(first, "free_fall", "test_id")
            self.assertEqual(1, free_fall_id["failed"])
            self.assertEqual(0, free_fall_id["unavailable"])
            free_fall_ood = self._row(
                first,
                "free_fall",
                "test_ood1",
            )
            self.assertEqual(0, free_fall_ood["expected"])
            self.assertIsNone(free_fall_ood["coverage"])
            self.assertIsNone(free_fall_ood["mean"])

            overall = first["overall"]
            self.assertEqual(5, overall["expected"])
            self.assertEqual(3, overall["completed"])
            self.assertEqual(2, overall["evaluated"])
            self.assertEqual(1, overall["failed"])
            self.assertEqual(1, overall["unavailable"])
            self.assertAlmostEqual(0.3, overall["mean"])
            self.assertIsNone(first["official_task_result"]["score"])
            self.assertEqual([], first["integrity_issues"])

    def test_cli_writes_json_markdown_and_training_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._fixture_run(root)
            output = root / "report"
            run_before = (run_dir / "run.json").read_bytes()

            self.assertEqual(
                0,
                main([
                    "--run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(output),
                ]),
            )

            payload = json.loads(
                (output / OUTPUT_JSON).read_text(encoding="utf-8")
            )
            markdown = (output / OUTPUT_MARKDOWN).read_text(
                encoding="utf-8"
            )
            self.assertTrue(
                payload["provenance"]["checkpoint"]["digest_verified"]
            )
            self.assertIsNotNone(
                payload["provenance"]["checkpoint"]["digest"]
            )
            self.assertTrue(
                payload["training_evidence"]["loss_summary"]["available"]
            )
            self.assertEqual(
                1450,
                payload["training_evidence"]["loss_summary"]["summary"][
                    "recorded_steps"
                ],
            )
            self.assertTrue(
                payload["training_evidence"]["gradient_audit"]["available"]
            )
            self.assertIn(
                "`free_fall` | `test_ood1` | 0 | 0 | 0",
                markdown,
            )
            self.assertIn("N/A", markdown)
            self.assertEqual(
                run_before,
                (run_dir / "run.json").read_bytes(),
            )

    def test_non_complete_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._fixture_run(Path(temporary))
            run = json.loads(
                (run_dir / "run.json").read_text(encoding="utf-8")
            )
            run["status"] = "inference_incomplete"
            self._write_json(run_dir / "run.json", run)

            with self.assertRaisesRegex(ValueError, "status=complete"):
                summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
