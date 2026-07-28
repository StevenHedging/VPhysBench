from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..driver import DirectManagedDriver


class StandardV2VCLIDriver(DirectManagedDriver):
    """Portable CLI boundary for text-conditioned video-to-video models.

    The command receives ``--prompt``, ``--video``, ``--output`` and
    ``--seed``. The input video must be an explicitly declared conditioning
    asset; evaluator references and raw source videos are rejected upstream.
    """

    def dependency_paths(self) -> dict[str, Path]:
        return {
            "src/physbench/baseline_runtime/drivers/"
            "subprocess_v2v.py": Path(__file__),
        }

    def prepare_job(
        self,
        *,
        job: dict[str, Any],
        case: dict[str, Any],
        adaptation: dict[str, Any],
        source_root: Path,
        run_dir: Path,
    ) -> dict[str, Any]:
        native = job["native_inputs"]
        input_video_asset = native["vision"].get("input_video_asset")
        if not input_video_asset:
            raise ValueError(
                f"standard V2V job has no input-video asset: {job['job_id']}"
            )
        input_video = (source_root / input_video_asset).resolve()
        if not input_video.is_file():
            raise FileNotFoundError(
                f"standard V2V input video not found: {input_video}"
            )
        output = (
            run_dir
            / "predictions"
            / adaptation["conditioning"]
            / f"{job['job_id']}.mp4"
        ).resolve()
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "seed": int(job["seed"]),
            "prompt": native["text"]["prompt"],
            "input_video": str(input_video),
            "output_video": str(output),
            "generation_shape": native["generation_shape"],
        }

    def execute_job(
        self,
        spec: dict[str, Any],
        *,
        log_path: Path,
    ) -> dict[str, Any]:
        config = self.bundle.value["runner"]["config"]
        configured = config.get("command")
        if (
            not isinstance(configured, list)
            or not configured
            or any(not isinstance(item, str) or not item for item in configured)
        ):
            raise ValueError(
                "standard_v2v_cli_v1 requires runner.config.command"
            )
        command = [
            *configured,
            "--prompt",
            spec["prompt"],
            "--video",
            spec["input_video"],
            "--output",
            spec["output_video"],
            "--seed",
            str(spec["seed"]),
            *[str(item) for item in config.get("extra_args", [])],
        ]
        Path(spec["output_video"]).parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=self.bundle.value.get("runtime", {}).get(
                    "working_directory", self.bundle.root
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            "return_code": completed.returncode,
            "command": command,
            "log_path": str(log_path),
        }
