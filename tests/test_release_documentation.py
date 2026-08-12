from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path

from _paths import ROOT


REQUIRED_DOCS = (
    "GETTING_STARTED.md",
    "CUSTOM_BASELINE_QUICKSTART.md",
    "SUBMISSION_QUICKSTART.md",
    "BENCHMARK_PROTOCOL.md",
    "RUN_LAYOUT.md",
    "REPRODUCIBILITY.md",
)
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "BASELINE_INTEGRATION.md",
    ROOT / "docs" / "EVALUATION.md",
    ROOT / "docs" / "OPERATIONS.md",
    ROOT / "docs" / "TASKS.md",
)
MODEL_MARKERS = (
    "wan22",
    "cosmos3",
    "causal_forcing",
    "quantity_embedding",
    "symbol_value_cross_attention",
)


class ReleaseDocumentationTests(unittest.TestCase):
    def test_required_manuals_exist(self) -> None:
        self.assertEqual(
            [],
            [name for name in REQUIRED_DOCS if not (ROOT / "docs" / name).is_file()],
        )

    def test_public_manuals_have_no_model_or_machine_specific_content(self) -> None:
        paths = [*PUBLIC_DOCS, *(ROOT / "docs" / name for name in REQUIRED_DOCS)]
        violations: list[str] = []
        for path in paths:
            content = path.read_text(encoding="utf-8")
            lowered = content.casefold()
            for marker in MODEL_MARKERS:
                if marker in lowered:
                    violations.append(f"{path.name}: {marker}")
            if re.search(r"(?<![A-Za-z0-9])/(?:root|mnt)/", content):
                violations.append(f"{path.name}: machine path")
        self.assertEqual([], violations)

    def test_onboarding_names_the_preserved_workspace_contracts(self) -> None:
        content = (ROOT / "docs" / "GETTING_STARTED.md").read_text(
            encoding="utf-8"
        )
        integration = (
            ROOT / "docs" / "CUSTOM_BASELINE_QUICKSTART.md"
        ).read_text(encoding="utf-8")
        run_layout = (ROOT / "docs" / "RUN_LAYOUT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("baselines/<baseline_id>", integration)
        self.assertIn("run/<run_id>/predictions", run_layout)
        self.assertIn("physbench dataset pull", content)
        self.assertIn("physbench doctor", content)

    def test_main_navigation_exposes_baseline_adapter_and_result_guides(self) -> None:
        content = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/BASELINE_INTEGRATION.md", content)
        self.assertIn("docs/DATA_ADAPTER.md", content)
        self.assertRegex(
            content,
            r"\[[^\]]*(?:结果|result)[^\]]*\]\(docs/RUN_LAYOUT\.md\)",
        )

    def test_protocol_declares_six_scored_and_one_unsupported_scene(self) -> None:
        content = (ROOT / "docs" / "BENCHMARK_PROTOCOL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Six scored scenes", content)
        self.assertIn("One Dataset-only unsupported scene", content)

    def test_v2v_documentation_matches_the_current_dataset_asset_boundary(self) -> None:
        cases = [
            json.loads(line)
            for line in (
                ROOT / "datasets" / "releases" / "13.0.0" / "cases.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(916, len(cases))
        self.assertFalse(any(
            "input_video" in case.get("assets", {})
            for case in cases
        ))
        content = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "README.md",
                ROOT / "docs" / "CUSTOM_BASELINE_QUICKSTART.md",
                ROOT / "docs" / "BASELINE_INTEGRATION.md",
            )
        )
        self.assertIn("cropped conditioning-prefix video", content)
        self.assertIn("not supported", content)
        self.assertIn("must not", content)
        self.assertIn("reference video", content)

    def test_data_adapter_manual_has_no_retired_model_or_scene_residue(self) -> None:
        content = (ROOT / "docs" / "DATA_ADAPTER.md").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(re.search(r"\bWAN\b", content))
        self.assertIsNone(re.search(r"\bCosmos\b", content))
        self.assertNotIn("free_fall", content)

    def test_run_layout_explains_official_result_fields(self) -> None:
        content = (ROOT / "docs" / "RUN_LAYOUT.md").read_text(
            encoding="utf-8"
        )
        for field in (
            "evaluation/case_results.jsonl",
            "evaluation/task_result.json",
            "coverage",
            "status_counts",
            "expert score",
            "CSTI",
            "degradation",
            "observed_mean_score",
        ):
            self.assertIn(field, content)
        self.assertRegex(content, r"coverage[^\n]*1")
        self.assertRegex(
            content,
            r"observed_mean_score[^\n]*(?:not official|非正式|不能作为正式)",
        )

    def test_release_manifest_matches_dataset_binding(self) -> None:
        release = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        binding = json.loads((ROOT / "datasets" / "huggingface.json").read_text())
        self.assertEqual("2026-08-11", release["branch"])
        self.assertEqual("1.0", release["task_schema"])
        self.assertEqual("scene_default_v1", release["evaluation_protocol"])
        self.assertEqual(
            [
                "tasks/official/six_scene_direct_eval_v1.json",
                "tasks/official/six_scene_train_six_scene_eval_v1.json",
            ],
            release["official_tasks"],
        )
        self.assertEqual(
            {
                "six_scene_direct_eval_v1": {
                    "training_cases": 0,
                    "evaluation_jobs": 775,
                },
                "six_scene_train_six_scene_eval_v1": {
                    "training_cases": 679,
                    "evaluation_jobs": 96,
                },
            },
            release["task_counts"],
        )
        self.assertEqual(binding["repo_id"], release["dataset"]["repo_id"])
        self.assertEqual(binding["revision"], release["dataset"]["revision"])
        self.assertEqual(6, len(release["scored_scenes"]))
        self.assertEqual(["push_bottle"], release["preview_scenes"])

    def test_evaluator_extra_pins_the_documented_sam2_revision(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        release = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        requirements = project["project"]["optional-dependencies"][
            "scene-evaluation"
        ]
        self.assertTrue(any(
            release["sam2_source_revision"] in requirement
            for requirement in requirements
        ))


if __name__ == "__main__":
    unittest.main()
