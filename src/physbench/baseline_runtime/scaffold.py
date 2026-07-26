from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..io import write_json


SCENES = [
    "pendulum",
    "free_fall",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion",
]


def _adapter() -> dict[str, Any]:
    return {
        "preset": "standard_i2v_v1",
        "profile_set": "five_scene_i2v_v1",
        "first_frame_policy": "require_asset",
        "spatial": {
            "scene_profiles": {
                scene_id: {"width": 832, "height": 480}
                for scene_id in SCENES
            }
        },
        "temporal": {
            "fps": 24,
            "num_frames": 121,
            "valid_frame_rule": "4n+1",
        },
    }


def _common(name: str) -> dict[str, Any]:
    return {
        "schema_version": "4.0",
        "baseline_id": name,
        "baseline_version": "0.1.0",
        "description": f"Physics Video Benchmark Baseline: {name}.",
        "supported_scenes": SCENES,
        "capabilities": {
            "task_families": ["direct_eval"],
            "conditioning": ["generic", "physics"],
            "train": False,
            "finetune": False,
            "generate": True,
        },
        "model": {"model_id": name, "checkpoint": None},
        "runtime": {},
        "adapter": _adapter(),
    }


def create_baseline_scaffold(
    *,
    name: str,
    backend: str,
    root: str | Path,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(
            "baseline name may only contain letters, digits, '.', '_' and '-'"
        )
    if backend not in {"managed-i2v", "submission"}:
        raise ValueError(
            "baseline backend must be managed-i2v or submission"
        )
    target = (Path(root).resolve() / name).resolve()
    if target.exists():
        raise FileExistsError(
            f"baseline scaffold target already exists: {target}"
        )
    target.mkdir(parents=True)
    value = _common(name)
    if backend == "managed-i2v":
        value["implementation"] = {
            "kind": "managed",
            "driver": "driver.py",
            "fingerprint_paths": ["*.py"],
        }
        value["runner"] = {
            "type": "standard_i2v_cli_v1",
            "config": {
                "command": ["python", "inference.py"],
                "extra_args": [],
            },
        }
        (target / "driver.py").write_text(
            "from physbench.baseline_runtime.drivers.subprocess_i2v import "
            "StandardI2VCLIDriver as Driver\n",
            encoding="utf-8",
        )
    else:
        value["implementation"] = {
            "kind": "submission",
            "fingerprint_paths": [],
        }
    write_json(target / "baseline.json", value)
    write_json(target / "baseline.local.example.json", {
        "model": {"checkpoint": "/absolute/path/to/checkpoint"},
        "runtime": (
            {}
            if backend == "managed-i2v"
            else {
                "submission_manifest": (
                    "/absolute/path/to/submission.jsonl"
                )
            }
        ),
    })
    (target / ".gitignore").write_text(
        "baseline.local.json\n__pycache__/\n",
        encoding="utf-8",
    )
    readme = (
        f"# {name}\n\n"
        f"Backend: `{backend}`.\n\n"
        "Copy `baseline.local.example.json` to `baseline.local.json` and "
        "fill only machine-local paths. See the repository operations guide "
        "for the input/output contract.\n"
    )
    (target / "README.md").write_text(readme, encoding="utf-8")
    return target
