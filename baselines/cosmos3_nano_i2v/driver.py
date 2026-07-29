from __future__ import annotations

import hashlib
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from physbench.baseline_runtime import DirectManagedDriver
from physbench.io import sha256_file, write_json


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Driver(DirectManagedDriver):
    """Cosmos-specific model boundary beneath the managed runtime."""

    @staticmethod
    def _environment_root(python: Path) -> Path:
        # Keep the virtual-environment boundary.  ``.venv/bin/python`` is
        # commonly a symlink into uv's managed base Python; resolving it would
        # drop both ``.venv/bin`` (including uvx) and its site-packages.
        return python.parent.parent

    def _gpu_groups(self) -> list[list[str]]:
        runtime = self.bundle.value["runtime"]
        gpu_ids = [
            item.strip()
            for item in str(runtime["cuda_visible_devices"]).split(",")
            if item.strip()
        ]
        if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
            raise ValueError(
                "Cosmos3 cuda_visible_devices must be comma-separated GPU IDs"
            )
        per_worker = int(runtime.get("gpus_per_worker", len(gpu_ids)))
        if per_worker <= 0:
            raise ValueError("Cosmos3 gpus_per_worker must be positive")
        if len(gpu_ids) % per_worker:
            raise ValueError(
                "Cosmos3 cuda_visible_devices count must be divisible by "
                "gpus_per_worker"
            )
        return [
            gpu_ids[index:index + per_worker]
            for index in range(0, len(gpu_ids), per_worker)
        ]

    def _worker_index(self, job_id: str) -> int:
        digest = hashlib.sha256(job_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self._gpu_groups())

    def _inference_entry(self) -> Path:
        return (
            Path(self.bundle.value["runtime"]["framework_root"])
            / "cosmos_framework"
            / "scripts"
            / "inference.py"
        ).resolve()

    def validate_deployment(self) -> None:
        model = self.bundle.value["model"]
        checkpoint = model.get("checkpoint")
        if not checkpoint or not Path(checkpoint).is_dir():
            raise FileNotFoundError(
                f"Cosmos3 checkpoint directory not found: {checkpoint}"
            )
        checkpoint_root = Path(checkpoint)
        for relative, expected in model["identity_files"].items():
            path = checkpoint_root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cosmos3 checkpoint identity file missing: {path}"
                )
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"Cosmos3 checkpoint identity mismatch for {relative}: "
                    f"expected={expected}, actual={actual}"
                )
        runtime = self.bundle.value["runtime"]
        for key in ("python", "torchrun"):
            path = Path(runtime[key])
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cosmos3 runtime {key} not found: {path}"
                )
        if not self._inference_entry().is_file():
            raise FileNotFoundError(
                f"Cosmos3 inference entry not found: "
                f"{self._inference_entry()}"
            )

    def dependency_paths(self) -> dict[str, Path]:
        return {
            "external/cosmos_framework/scripts/inference.py": (
                self._inference_entry()
            )
        }

    @staticmethod
    def _cuda_library_path(python: Path) -> str:
        environment_root = Driver._environment_root(python)
        paths = sorted(
            str(path)
            for path in environment_root.glob(
                "lib/python*/site-packages/nvidia/*/lib"
            )
            if path.is_dir()
        )
        return os.pathsep.join(paths)

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
                f"Cosmos3 first frame not found: {first_frame}"
            )
        predictor = self.bundle.value["runner"]["config"]
        shape = native["generation_shape"]
        worker_index = self._worker_index(job["job_id"])
        output_root = (
            run_dir
            / "predictions"
            / "_workers"
            / f"worker_{worker_index:02d}"
        ).resolve()
        output_video = output_root / job["job_id"] / "vision.mp4"
        payload_path = run_dir / "jobs" / f"{job['job_id']}.payload.json"
        payload = {
            "model_mode": "image2video",
            "name": job["job_id"],
            "prompt": native["text"]["prompt"],
            "negative_prompt": predictor["negative_prompt"],
            "vision_path": str(first_frame),
            "enable_sound": False,
            "num_steps": int(predictor["num_inference_steps"]),
            "guidance": float(predictor["guidance"]),
            "shift": float(predictor["shift"]),
            "fps": int(shape["fps"]),
            "num_frames": int(shape["num_frames"]),
            "resolution": str(shape["resolution"]),
            "aspect_ratio": shape["aspect_ratio"],
            "seed": int(job["seed"]),
        }
        write_json(payload_path, payload)
        return {
            "job_id": job["job_id"],
            "case_id": job["case_id"],
            "seed": int(job["seed"]),
            "payload_path": str(payload_path),
            "output_root": str(output_root),
            "output_video": str(output_video),
            "first_frame": str(first_frame),
            "worker_index": worker_index,
        }

    def _execution_environment(
        self,
        gpu_ids: list[str],
    ) -> dict[str, str]:
        runtime = self.bundle.value["runtime"]
        python = Path(runtime["python"])
        environment_root = self._environment_root(python)
        env = os.environ.copy()
        previous_libraries = env.get("LD_LIBRARY_PATH", "")
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "HF_HOME": str(runtime["hf_home"]),
            "UV_CACHE_DIR": str(runtime["uv_cache_dir"]),
            "PATH": f"{environment_root / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": os.pathsep.join(
                item
                for item in (
                    self._cuda_library_path(python),
                    previous_libraries,
                )
                if item
            ),
        })
        if runtime.get("offline", True):
            env["HF_HUB_OFFLINE"] = "1"
        return env

    def _execute_batch(
        self,
        specs: list[dict[str, Any]],
        *,
        gpu_ids: list[str],
        log_path: Path,
    ) -> dict[str, dict[str, Any]]:
        if not specs:
            return {}
        runtime = self.bundle.value["runtime"]
        output_roots = {spec["output_root"] for spec in specs}
        if len(output_roots) != 1:
            raise ValueError("Cosmos3 worker batch has multiple output roots")
        seeds = {int(spec["seed"]) for spec in specs}
        if len(seeds) != 1:
            raise ValueError("Cosmos3 worker batch has multiple setup seeds")
        command = [
            str(runtime["torchrun"]),
            f"--nproc-per-node={len(gpu_ids)}",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            (
                "--parallelism-preset="
                f"{runtime.get('parallelism_preset', 'throughput')}"
            ),
            "-i",
            *(spec["payload_path"] for spec in specs),
            "-o",
            next(iter(output_roots)),
            "--checkpoint-path",
            str(self.bundle.value["model"]["checkpoint"]),
            f"--seed={next(iter(seeds))}",
            (
                "--guardrails"
                if runtime.get("guardrails", False)
                else "--no-guardrails"
            ),
            (
                "--use-torch-compile"
                if runtime.get("compile", False)
                else "--no-use-torch-compile"
            ),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=runtime["framework_root"],
                env=self._execution_environment(gpu_ids),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {
            spec["job_id"]: {
                "return_code": completed.returncode,
                "command": command,
                "log_path": str(log_path),
                "worker_index": int(spec["worker_index"]),
                "worker_gpu_ids": list(gpu_ids),
                "persistent_model_batch_size": len(specs),
            }
            for spec in specs
        }

    def execute_job(
        self,
        spec: dict[str, Any],
        *,
        log_path: Path,
    ) -> dict[str, Any]:
        worker_index = int(spec["worker_index"])
        groups = self._gpu_groups()
        return self._execute_batch(
            [spec],
            gpu_ids=groups[worker_index],
            log_path=log_path,
        )[spec["job_id"]]

    def execute_jobs(
        self,
        specs: list[dict[str, Any]],
        *,
        run_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        groups = self._gpu_groups()
        by_worker = {
            index: [
                spec
                for spec in specs
                if int(spec["worker_index"]) == index
            ]
            for index in range(len(groups))
        }
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    self._execute_batch,
                    worker_specs,
                    gpu_ids=groups[index],
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
                        f"duplicate Cosmos3 batch results: {sorted(overlap)}"
                    )
                results.update(batch)
        return results
