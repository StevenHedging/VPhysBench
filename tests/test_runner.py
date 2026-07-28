from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _paths import FIXTURES, METRICS, SCENES
from physbench.io import (
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from physbench.runner import reevaluate_run


class LegacyReevaluationTests(unittest.TestCase):
    def test_frozen_schema_v1_run_can_still_be_reevaluated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            cases = load_jsonl(FIXTURES / "cases.jsonl")
            prediction = {
                "job_id": "legacy-job",
                "case_id": "pend_id_001",
                "baseline_id": "legacy-baseline",
                "prompt_profile_id": "generic",
                "evaluation_partition": "test_id",
                "status": "placeholder",
                "video_path": None,
                "manual_scores": {},
            }
            write_jsonl(run_dir / "frozen_cases.jsonl", cases)
            write_jsonl(run_dir / "predictions.jsonl", [prediction])
            write_json(run_dir / "frozen_metrics.json", load_json(METRICS))
            write_json(run_dir / "run.json", {
                "schema_version": "1.0",
                "run_id": "legacy-run",
                "task_id": "legacy-task",
                "baseline_id": "legacy-baseline",
                "manifest_sha256": "0" * 64,
                "status": "complete_with_placeholders",
            })
            write_json(run_dir / "plan.json", {
                "schema_version": "1.0",
                "mode": "zero_shot_eval",
                "view": "B",
                "train_case_ids": [],
                "jobs": [prediction],
                "prompt_profiles": {
                    "train": None,
                    "eval": ["generic"],
                },
            })

            summary = reevaluate_run(run_dir, SCENES)

            self.assertEqual(0, summary["scored_jobs"])
            self.assertTrue((run_dir / "case_metrics.jsonl").is_file())
            self.assertTrue((run_dir / "summary.json").is_file())
            self.assertIn(
                "legacy-run",
                (run_dir / "report.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
