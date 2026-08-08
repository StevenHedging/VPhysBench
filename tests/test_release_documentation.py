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

    def test_protocol_declares_five_scored_and_two_preview_scenes(self) -> None:
        content = (ROOT / "docs" / "BENCHMARK_PROTOCOL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Five scored scenes", content)
        self.assertIn("Two preview scenes", content)

    def test_release_manifest_matches_dataset_binding(self) -> None:
        release = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        binding = json.loads((ROOT / "datasets" / "huggingface.json").read_text())
        self.assertEqual("2026-08-08", release["branch"])
        self.assertEqual(binding["repo_id"], release["dataset"]["repo_id"])
        self.assertEqual(binding["revision"], release["dataset"]["revision"])
        self.assertEqual(5, len(release["scored_scenes"]))
        self.assertEqual(2, len(release["preview_scenes"]))

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
