from __future__ import annotations

import json
import re
import subprocess
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
    ROOT / "datasets" / "HF_DATASET_CARD.md",
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

        dataset_card = (ROOT / "datasets" / "HF_DATASET_CARD.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Six scenes are part of the scored benchmark", dataset_card)
        self.assertIn("push_bottle", dataset_card)
        self.assertNotIn("vertical_spring_oscillator` are", dataset_card)

    def test_v2v_documentation_matches_the_current_dataset_asset_boundary(self) -> None:
        cases = [
            json.loads(line)
            for line in (
                ROOT / "datasets" / "releases" / "14.0.0" / "cases.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(916, len(cases))
        self.assertFalse(any(
            "input_video" in case.get("assets", {})
            for case in cases
        ))
        v2v_manuals = (
            ROOT / "docs" / "CUSTOM_BASELINE_QUICKSTART.md",
            ROOT / "docs" / "BASELINE_INTEGRATION.md",
            ROOT / "docs" / "DATA_ADAPTER.md",
        )
        for path in v2v_manuals:
            content = path.read_text(encoding="utf-8").casefold()
            self.assertRegex(content, r"(?:not supported|not an official|不能用于)")
            self.assertRegex(content, r"(?:reference|参考)")
            self.assertRegex(content, r"(?:must not|do not|禁止|不能|不得)")

    def test_protocol_distinguishes_degraded_zero_from_contract_errors(self) -> None:
        content = (ROOT / "docs" / "BENCHMARK_PROTOCOL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("sealed media/record contract", content)
        self.assertIn("protocol_error", content)
        self.assertRegex(content, r"protocol_error[\s\S]{0,100}coverage")
        self.assertIn("pre-evaluation validation", content)
        self.assertIn("不会产生 publishable Task result", content)

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
        self.assertEqual("2026-08-31", release["branch"])
        self.assertEqual(
            "56585745c3ee05d99502f5fe2e21d280dc5a789e",
            release["source_commit"],
        )
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

    def test_bootstrap_documentation_states_profiles_and_portability_boundary(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        getting_started = (ROOT / "docs" / "GETTING_STARTED.md").read_text(
            encoding="utf-8"
        )
        operations = (ROOT / "docs" / "OPERATIONS.md").read_text(
            encoding="utf-8"
        )
        combined = "\n".join((readme, getting_started, operations))

        for command in (
            "scripts/bootstrap_env.sh --profile metadata",
            "scripts/bootstrap_env.sh --profile evaluation",
            "scripts/bootstrap_env.sh --profile metadata --dry-run",
        ):
            self.assertIn(command, combined)
        for requirement in (
            "Python 3.11",
            "Python 3.12",
            "PyTorch 2.10.0",
            "CUDA 12.8",
            "VPHYSBENCH_SAM31_CHECKPOINT",
            "baseline.local.json",
            "Dataset media",
            "checkpoints",
            "credentials",
        ):
            self.assertIn(requirement, combined)
        self.assertIn("make portable-release-check", operations)

    def test_ci_covers_asset_free_evaluator_and_portable_release_gates(self) -> None:
        content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("make portable-release-check", content)
        self.assertIn('python-version: "3.12"', content)
        self.assertIn('python -m pip install --editable ".[scene-evaluation]"', content)
        self.assertIn("make test-evaluation", content)
        self.assertNotIn("sam31-evaluation", content)
        self.assertNotIn("physbench dataset pull", content)
        self.assertNotIn("VPHYSBENCH_SAM31_CHECKPOINT", content)

    def test_ci_owned_interface_target_runs_bootstrap_safety_suite(self) -> None:
        """Dropping bootstrap regressions from the Make target must fail."""
        completed = subprocess.run(
            ["make", "--dry-run", "test-interface"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("tests.test_bootstrap_env", completed.stdout)

    def test_reproducibility_guide_agrees_with_release_manifest_branch(self) -> None:
        release = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        content = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"`{release['branch']}`", content)
        self.assertIn("`source_commit`", content)
        self.assertIn("pre-publication source revision", content)
        self.assertIn("final self-containing release commit", content)

    def test_real_case_guidance_requires_full_doctor_and_checkpoint(self) -> None:
        content = (ROOT / "docs" / "GETTING_STARTED.md").read_text(
            encoding="utf-8"
        )
        real_case = content[content.index("## 5. Run one real case"):]
        self.assertIn("VPHYSBENCH_SAM31_CHECKPOINT", real_case)
        self.assertIn("full doctor", real_case)
        self.assertIn("exit 0", real_case)

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

    def test_csti_manual_documents_sam31_identity_and_failure_semantics(self) -> None:
        evaluation = (ROOT / "docs" / "EVALUATION.md").read_text(
            encoding="utf-8"
        )
        run_layout = (ROOT / "docs" / "RUN_LAYOUT.md").read_text(
            encoding="utf-8"
        )
        for field in (
            "VPHYSBENCH_SAM31_CHECKPOINT",
            "text-only",
            "Hungarian",
            "termination_patience",
            "evaluator_init_failure",
            "init_coverage",
            "exact_full_tube_edt",
        ):
            self.assertIn(field, evaluation)
        for field in (
            "csti_video",
            "csti_per_subject",
            "initial_matching",
            "termination_frame_per_subject",
            "evaluator_init_failure",
            "init_coverage",
        ):
            self.assertIn(field, run_layout)

    def test_sam31_extra_pins_the_official_source_revision(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        requirements = project["project"]["optional-dependencies"][
            "sam31-evaluation"
        ]
        self.assertTrue(
            any(
                "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da" in requirement
                for requirement in requirements
            )
        )

    def test_evaluator_extras_exclude_opencv_5(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        extras = project["project"]["optional-dependencies"]

        for extra in ("scene-evaluation", "pendulum-evaluation"):
            with self.subTest(extra=extra):
                self.assertIn(
                    "opencv-python-headless>=4.10,<5",
                    extras[extra],
                )


if __name__ == "__main__":
    unittest.main()
