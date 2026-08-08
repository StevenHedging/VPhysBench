from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.release_audit import audit_release


class ReleaseAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def issues(self) -> list[str]:
        files = sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        )
        return audit_release(self.root, tracked_files=files)

    def assert_issue(self, expected: str) -> None:
        self.assertTrue(
            any(expected in issue for issue in self.issues()),
            self.issues(),
        )

    def test_rejects_registered_baseline_bundle(self) -> None:
        self.write("baselines/model/baseline.json", "{}")
        self.assert_issue("tracked baseline bundle")

    def test_rejects_model_weight(self) -> None:
        self.write("models/model.safetensors", "fixture")
        self.assert_issue("model weight or checkpoint")

    def test_rejects_runtime_outputs_and_local_overrides(self) -> None:
        self.write("run/example/predictions/output.mp4", "fixture")
        self.write("baselines/model/baseline.local.json", "{}")
        self.assert_issue("tracked runtime output")
        self.assert_issue("local-only file")

    def test_rejects_model_markers_absolute_paths_and_credentials(self) -> None:
        self.write("docs/model.md", "wan22 lives at /root/models")
        self.write("config.json", '{"token": "hf_' + "a" * 32 + '"}')
        self.assert_issue("concrete model marker")
        self.assert_issue("machine-local absolute path")
        self.assert_issue("credential-like content")

    def test_accepts_release_skeleton(self) -> None:
        self.write("baselines/README.md", "# Baseline integrations\n")
        self.write("run/README.md", "# AtomicRun outputs\n")
        self.write("src/physbench/core.py", "VALUE = 'generic'\n")
        self.assertEqual([], self.issues())

    def test_policy_files_may_name_forbidden_markers(self) -> None:
        self.write(
            "scripts/release_audit.py",
            "FORBIDDEN = ('wan22', '/root/', 'hf_" + "a" * 32 + "')\n",
        )
        self.write(
            "tests/test_release_audit.py",
            "MARKER = ('cosmos3', '/mnt/')\n",
        )
        self.write(
            "tests/test_repository_portability.py",
            "FORBIDDEN = ('/root/', '/mnt/')\n",
        )
        self.assertEqual([], self.issues())


if __name__ == "__main__":
    unittest.main()
