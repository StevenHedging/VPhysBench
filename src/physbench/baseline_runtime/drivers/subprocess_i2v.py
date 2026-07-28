from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..driver import DirectManagedDriver
from ..input_contract import resolve_dataset_asset_path


class StandardI2VCLIDriver(DirectManagedDriver):
    """Portable CLI contract for ordinary image-to-video generators.

    The v1 command always receives ``--prompt``, ``--image``, ``--output`` and
    ``--seed``. Set ``runner.config.job_spec_arg`` (new scaffolds use
    ``--job-spec``) to also pass the JSON job specification containing the
    requested generation shape. Extra model-specific flags belong in
    ``runner.config.extra_args``.
    """

    def dependency_paths(self) -> dict[str, Path]:
        return {
            "src/physbench/baseline_runtime/drivers/"
            "subprocess_i2v.py": Path(__file__),
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
        first_frame_asset = native["vision"].get("first_frame_asset")
        if not first_frame_asset:
            raise ValueError(
                f"standard I2V job has no first-frame asset: {job['job_id']}"
            )
        expected_asset = case.get("assets", {}).get("first_frame")
        if first_frame_asset != expected_asset:
            raise ValueError(
                "standard I2V native input does not match the contract-"
                "authorized first-frame asset"
            )
        first_frame = resolve_dataset_asset_path(
            source_root,
            first_frame_asset,
            label="standard I2V first frame",
        )
        if not first_frame.is_file():
            raise FileNotFoundError(
                f"standard I2V first frame not found: {first_frame}"
            )
        output = (
            run_dir
            / "predictions"
            / f"{job['job_id']}.mp4"
        ).resolve()
        job_spec = (
            run_dir / "jobs" / f"{job['job_id']}.json"
        ).resolve()
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "seed": int(job["seed"]),
            "prompt": native["text"]["prompt"],
            "first_frame": str(first_frame),
            "output_video": str(output),
            "job_spec": str(job_spec),
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
                "standard_i2v_cli_v1 requires runner.config.command"
            )
        command = [
            *configured,
            "--prompt",
            spec["prompt"],
            "--image",
            spec["first_frame"],
            "--output",
            spec["output_video"],
            "--seed",
            str(spec["seed"]),
        ]
        job_spec_arg = config.get("job_spec_arg")
        if job_spec_arg is not None:
            if not isinstance(job_spec_arg, str) or not job_spec_arg:
                raise ValueError(
                    "runner.config.job_spec_arg must be a non-empty string"
                )
            command.extend([job_spec_arg, spec["job_spec"]])
        command.extend(
            str(item) for item in config.get("extra_args", [])
        )
        output = Path(spec["output_video"])
        output.parent.mkdir(parents=True, exist_ok=True)
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
