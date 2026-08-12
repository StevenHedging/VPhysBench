from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from _paths import ROOT


class CSTIBenchmarkTests(unittest.TestCase):
    def test_benchmark_cli_emits_deterministic_shape_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/benchmark_csti.py"),
                    "--frames",
                    "5",
                    "--height",
                    "9",
                    "--width",
                    "11",
                    "--objects",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "algorithm",
                    "sampling_fps",
                    "spatial_tolerance_policy",
                    "spatial_tolerance_radius_ratio",
                    "temporal_tolerance_s",
                    "frames",
                    "initial_frames_excluded",
                    "scored_frames",
                    "height",
                    "width",
                    "objects",
                    "scores",
                    "diagnostic_points",
                    "wall_time_s",
                    "peak_rss_kib",
                },
                set(payload),
            )
            self.assertEqual("exact_full_tube_edt", payload["algorithm"])
            self.assertEqual(24.0, payload["sampling_fps"])
            self.assertEqual(
                "reference_tube_equivalent_diameter_v1",
                payload["spatial_tolerance_policy"],
            )
            self.assertEqual(0.5, payload["spatial_tolerance_radius_ratio"])
            self.assertEqual(0.025, payload["temporal_tolerance_s"])
            self.assertEqual(5, payload["frames"])
            self.assertEqual(1, payload["initial_frames_excluded"])
            self.assertEqual(4, payload["scored_frames"])
            self.assertEqual(9, payload["height"])
            self.assertEqual(11, payload["width"])
            self.assertEqual(1, payload["objects"])
            self.assertEqual(1, len(payload["scores"]))
            self.assertEqual(1, len(payload["diagnostic_points"]))
            self.assertLessEqual(payload["diagnostic_points"][0], 4)
            self.assertTrue(0.0 <= payload["scores"][0] <= 1.0)
            self.assertGreaterEqual(payload["wall_time_s"], 0.0)
            self.assertGreater(payload["peak_rss_kib"], 0)


if __name__ == "__main__":
    unittest.main()
