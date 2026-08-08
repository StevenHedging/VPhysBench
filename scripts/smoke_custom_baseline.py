#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from physbench.baseline_api import load_baseline_bundle, load_baseline_plugin
from physbench.baseline_runtime import create_baseline_scaffold
from physbench.baseline_runtime.media_contract import probe_media
from physbench.io import load_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPM = b"P6\n2 2\n255\n" + bytes([
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 255,
])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="vphysbench-interface-") as temporary:
        root = Path(temporary)
        bundle = create_baseline_scaffold(
            name="smoke_fixture",
            backend="managed-i2v",
            root=root / "baselines",
        )
        descriptor = load_json(bundle / "baseline.json")
        descriptor["runner"]["config"]["command"] = [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "dummy_i2v_command.py"),
        ]
        write_json(bundle / "baseline.json", descriptor)
        load_baseline_plugin(load_baseline_bundle(bundle))

        image = root / "first.ppm"
        image.write_bytes(PPM)
        output = root / "run" / "smoke" / "predictions" / "video.mp4"
        job_spec = root / "job.json"
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
        completed = subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "examples" / "dummy_i2v_command.py"),
            "--prompt", "protocol fixture",
            "--image", str(image),
            "--output", str(output),
            "--seed", "42",
            "--job-spec", str(job_spec),
        ], check=False)
        if completed.returncode:
            return int(completed.returncode)
        probe = probe_media(output)
        if probe["frames"] != 5 or not output.is_relative_to(root / "run"):
            raise RuntimeError(f"unexpected smoke output: {probe}")
    print("smoke_interface=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
