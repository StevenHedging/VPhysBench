from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _paths import ROOT
from physbench.baseline_api import discover_baseline_bundles
from physbench.baseline_runtime.media_contract import probe_media


PPM = b"P6\n2 2\n255\n" + bytes([
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
])


class CleanBaselineSmokeTests(unittest.TestCase):
    def test_dummy_command_writes_contract_shaped_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "first.ppm"
            output = root / "run" / "smoke" / "predictions" / "video.mp4"
            job_spec = root / "job.json"
            image.write_bytes(PPM)
            job_spec.write_text(json.dumps({
                "media_contract": {
                    "output": {
                        "canvas": {"width": 32, "height": 24},
                        "timeline": {
                            "fps": 8,
                            "frame_count": {"rule": "fixed", "value": 5},
                        },
                    },
                },
            }), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "dummy_i2v_command.py"),
                    "--prompt",
                    "protocol fixture",
                    "--image",
                    str(image),
                    "--output",
                    str(output),
                    "--seed",
                    "42",
                    "--job-spec",
                    str(job_spec),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("not a benchmark baseline", completed.stderr)
            self.assertEqual(
                {"width": 32, "height": 24, "fps": 8.0, "frames": 5},
                {
                    key: probe_media(output)[key]
                    for key in ("width", "height", "fps", "frames")
                },
            )

    def test_repository_still_has_no_discoverable_baseline(self) -> None:
        self.assertEqual({}, discover_baseline_bundles(ROOT / "baselines"))


if __name__ == "__main__":
    unittest.main()
