from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from physbench.baseline_runtime import DirectManagedDriver
from physbench.io import sha256_file, write_json


class Driver(DirectManagedDriver):
    """Benchmark boundary for Causal Forcing++ frame-wise I2V."""

    def _gpu_ids(self) -> list[str]:
        raw = str(self.bundle.value["runtime"]["cuda_visible_devices"])
        gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
        if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
            raise ValueError(
                "Causal Forcing cuda_visible_devices must be comma-separated "
                "GPU IDs"
            )
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("Causal Forcing GPU IDs must be unique")
        return gpu_ids

    def _worker_index(self, job_id: str) -> int:
        digest = hashlib.sha256(job_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self._gpu_ids())

    def _framework_root(self) -> Path:
        return Path(self.bundle.value["runtime"]["framework_root"]).resolve()

    def _checkpoint_path(self) -> Path:
        model = self.bundle.value["model"]
        return (Path(model["checkpoint"]) / model["checkpoint_file"]).resolve()

    def _config_path(self) -> Path:
        relative = self.bundle.value["runner"]["config"]["official_config"]
        return (self.bundle.root / relative).resolve()

    def validate_deployment(self) -> None:
        model = self.bundle.value["model"]
        checkpoint_root = model.get("checkpoint")
        if not checkpoint_root or not Path(checkpoint_root).is_dir():
            raise FileNotFoundError(
                f"Causal Forcing checkpoint directory not found: {checkpoint_root}"
            )
        checkpoint = self._checkpoint_path()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Causal Forcing checkpoint file not found: {checkpoint}"
            )
        actual_checkpoint = sha256_file(checkpoint)
        if actual_checkpoint != model["checkpoint_sha256"]:
            raise ValueError(
                "Causal Forcing checkpoint identity mismatch: "
                f"expected={model['checkpoint_sha256']}, "
                f"actual={actual_checkpoint}"
            )

        base_model = model.get("base_model")
        if not base_model or not Path(base_model).is_dir():
            raise FileNotFoundError(
                f"Wan2.1 base model directory not found: {base_model}"
            )
        base_root = Path(base_model)
        for relative, expected_size in model["base_required_files"].items():
            path = base_root / relative
            actual_size = path.stat().st_size if path.is_file() else None
            if actual_size != int(expected_size):
                raise FileNotFoundError(
                    f"Wan2.1 required file is missing or incomplete: {path}; "
                    f"expected_size={expected_size}, actual_size={actual_size}"
                )
        for relative, expected in model["base_identity_files"].items():
            path = base_root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Wan2.1 base identity file missing: {path}"
                )
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"Wan2.1 base identity mismatch for {relative}: "
                    f"expected={expected}, actual={actual}"
                )

        runtime = self.bundle.value["runtime"]
        python = Path(runtime["python"])
        if not python.is_file():
            raise FileNotFoundError(
                f"Causal Forcing Python runtime not found: {python}"
            )
        identity_program = (
            "import json,platform,av,diffusers,flash_attn,omegaconf,torch,"
            "torchvision,transformers;"
            "print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'torchvision':torchvision.__version__,"
            "'flash_attn':flash_attn.__version__,"
            "'diffusers':diffusers.__version__,"
            "'transformers':transformers.__version__,'av':av.__version__,"
            "'omegaconf':omegaconf.__version__}))"
        )
        environment_check = subprocess.run(
            [str(python), "-c", identity_program],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if environment_check.returncode:
            raise RuntimeError(
                "Causal Forcing environment identity check failed: "
                f"{environment_check.stderr.strip()}"
            )
        actual_environment = json.loads(environment_check.stdout)
        expected_environment = runtime["environment_identity"]
        if actual_environment != expected_environment:
            raise ValueError(
                "Causal Forcing environment identity mismatch: "
                f"expected={expected_environment}, "
                f"actual={actual_environment}"
            )
        framework = self._framework_root()
        required = (
            framework / "pipeline" / "causal_inference.py",
            framework / "utils" / "wan_wrapper.py",
            framework / "wan" / "modules" / "causal_model.py",
            framework / "configs" / "default_config.yaml",
            self._config_path(),
            self.bundle.root / "worker.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Causal Forcing deployment files missing: {missing}"
            )
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=framework,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        actual_commit = completed.stdout.strip()
        expected_commit = runtime["causal_forcing_commit"]
        if completed.returncode or actual_commit != expected_commit:
            raise ValueError(
                "Causal Forcing source revision mismatch: "
                f"expected={expected_commit}, actual={actual_commit or 'unavailable'}"
            )
        self._gpu_ids()

    def dependency_paths(self) -> dict[str, Path]:
        framework = self._framework_root()
        paths: dict[str, Path] = {}
        for directory in ("pipeline", "utils", "wan", "demo_utils"):
            for path in sorted((framework / directory).rglob("*.py")):
                relative = path.relative_to(framework).as_posix()
                paths[f"external/Causal-Forcing/{relative}"] = path
        paths[
            "external/Causal-Forcing/configs/default_config.yaml"
        ] = framework / "configs" / "default_config.yaml"
        return paths

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
        first_frame = (
            source_root / native["vision"]["first_frame_asset"]
        ).resolve()
        if not first_frame.is_file():
            raise FileNotFoundError(
                f"Causal Forcing first frame not found: {first_frame}"
            )
        shape = native["generation_shape"]
        width = int(shape["width"])
        height = int(shape["height"])
        if sorted((width, height)) != [480, 832]:
            raise ValueError(
                "Causal Forcing requires a 480x832 fixed token budget; "
                f"got {width}x{height}"
            )
        frames = int(shape["num_frames"])
        if frames != 81 or (frames - 1) % 4:
            raise ValueError(
                f"Causal Forcing expects 81 pixel frames, got {frames}"
            )
        worker_index = self._worker_index(job["job_id"])
        output_video = (
            run_dir / "predictions" / f"{job['job_id']}.mp4"
        ).resolve()
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "scene_id": case["scene_id"],
            "seed": int(job["seed"]),
            "prompt": native["text"]["prompt"],
            "first_frame": str(first_frame),
            "output_video": str(output_video),
            "width": width,
            "height": height,
            "fps": int(shape["fps"]),
            "num_frames": frames,
            "latent_frames": int(
                self.bundle.value["runner"]["config"]["latent_frames"]
            ),
            "resize_policy": self.bundle.value["runner"]["config"][
                "pixel_resize"
            ],
            "worker_index": worker_index,
        }

    def _execution_environment(self, gpu_id: str) -> dict[str, str]:
        runtime = self.bundle.value["runtime"]
        framework = self._framework_root()
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "HF_HOME": str(runtime["hf_home"]),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": os.pathsep.join(
                item for item in (str(framework), existing_pythonpath) if item
            ),
        })
        if runtime.get("offline", True):
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    @staticmethod
    def _ensure_base_model_link(workspace: Path, base_model: Path) -> None:
        link = workspace / "wan_models" / "Wan2.1-T2V-1.3B"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != base_model.resolve():
                raise ValueError(
                    f"worker base-model link targets {link.resolve()}, "
                    f"expected {base_model.resolve()}"
                )
            return
        if link.exists():
            raise FileExistsError(
                f"worker base-model link path is occupied: {link}"
            )
        link.symlink_to(base_model.resolve(), target_is_directory=True)

    def _execute_batch(
        self,
        specs: list[dict[str, Any]],
        *,
        gpu_id: str,
        worker_index: int,
        run_dir: Path,
        log_path: Path,
    ) -> dict[str, dict[str, Any]]:
        if not specs:
            return {}
        worker_root = run_dir / "jobs" / "_workers" / f"worker_{worker_index:02d}"
        workspace = worker_root / "runtime"
        workspace.mkdir(parents=True, exist_ok=True)
        self._ensure_base_model_link(
            workspace,
            Path(self.bundle.value["model"]["base_model"]),
        )
        jobs_path = worker_root / "jobs.json"
        results_path = worker_root / "results.json"
        write_json(jobs_path, {"jobs": specs})
        command = [
            str(self.bundle.value["runtime"]["python"]),
            str((self.bundle.root / "worker.py").resolve()),
            "--framework-root",
            str(self._framework_root()),
            "--config",
            str(self._config_path()),
            "--checkpoint",
            str(self._checkpoint_path()),
            "--checkpoint-key",
            str(self.bundle.value["model"]["checkpoint_key"]),
            "--jobs",
            str(jobs_path),
            "--results",
            str(results_path),
        ]
        if self.bundle.value["runtime"].get("force_low_memory", True):
            command.append("--low-memory")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=self._execution_environment(gpu_id),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        payload: dict[str, Any] = {}
        if results_path.is_file():
            with results_path.open(encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                payload = loaded.get("jobs", {})
        output: dict[str, dict[str, Any]] = {}
        for spec in specs:
            job_id = spec["job_id"]
            worker_result = payload.get(job_id, {})
            return_code = int(
                worker_result.get("return_code", completed.returncode or 1)
            )
            output[job_id] = {
                **{
                    key: value
                    for key, value in worker_result.items()
                    if key != "return_code"
                },
                "return_code": return_code,
                "command": command,
                "log_path": str(log_path),
                "worker_index": worker_index,
                "worker_gpu_id": gpu_id,
                "persistent_model_batch_size": len(specs),
                "architecture": "framewise_causal_autoregressive_diffusion",
            }
        return output

    def execute_job(
        self,
        spec: dict[str, Any],
        *,
        log_path: Path,
    ) -> dict[str, Any]:
        worker_index = int(spec["worker_index"])
        gpu_id = self._gpu_ids()[worker_index]
        run_dir = Path(spec["output_video"]).resolve().parents[1]
        return self._execute_batch(
            [spec],
            gpu_id=gpu_id,
            worker_index=worker_index,
            run_dir=run_dir,
            log_path=log_path,
        )[spec["job_id"]]

    def execute_jobs(
        self,
        specs: list[dict[str, Any]],
        *,
        run_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        gpu_ids = self._gpu_ids()
        by_worker = {
            index: [
                spec
                for spec in specs
                if int(spec["worker_index"]) == index
            ]
            for index in range(len(gpu_ids))
        }
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(
                    self._execute_batch,
                    worker_specs,
                    gpu_id=gpu_ids[index],
                    worker_index=index,
                    run_dir=run_dir,
                    log_path=(
                        run_dir
                        / "logs"
                        / self.bundle.baseline_id
                        / f"worker_{index:02d}.log"
                    ),
                ): index
                for index, worker_specs in by_worker.items()
                if worker_specs
            }
            for future in as_completed(futures):
                batch = future.result()
                overlap = set(results) & set(batch)
                if overlap:
                    raise ValueError(
                        f"duplicate Causal Forcing results: {sorted(overlap)}"
                    )
                results.update(batch)
        return results
