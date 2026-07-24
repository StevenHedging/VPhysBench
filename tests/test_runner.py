from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _paths import FIXTURES, METRICS, ROOT, SCENES
from physbench.io import load_json, load_jsonl
from physbench.runner import reevaluate_run, run_benchmark


class RunnerTests(unittest.TestCase):
    def test_dummy_run_is_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = run_benchmark(
                task_path=FIXTURES / "task_view_a.json",
                baseline_path=ROOT / "configs" / "baselines" / "dummy_i2v.json",
                manifest_path=FIXTURES / "cases.jsonl",
                split_path=FIXTURES / "view_a.json",
                scene_config_dir=SCENES,
                metric_config_path=METRICS,
                output_root=temporary,
                run_id="test-run",
            )
            expected = {
                "run.json", "data_context.json", "frozen_task.json", "frozen_baseline.json", "frozen_split.json",
                "frozen_metrics.json", "frozen_cases.jsonl", "plan.json", "training_job.json", "training_stage.json",
                "frozen_prompt_profiles.json", "resolved_prompts.jsonl",
                "predictions.jsonl", "case_metrics.jsonl", "summary.json", "report.md",
            }
            self.assertTrue(expected <= {path.name for path in Path(run_dir).iterdir()})
            self.assertEqual(4, len(load_jsonl(Path(run_dir) / "predictions.jsonl")))
            self.assertEqual(
                {"physics_natural"},
                {
                    item["prompt_profile_id"]
                    for item in load_jsonl(Path(run_dir) / "predictions.jsonl")
                },
            )
            self.assertEqual("complete_with_placeholders", load_json(Path(run_dir) / "run.json")["status"])
            summary = reevaluate_run(run_dir, SCENES)
            self.assertEqual(0, summary["scored_jobs"])


if __name__ == "__main__":
    unittest.main()
