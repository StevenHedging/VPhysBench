from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from physbench.baseline_api.interfaces import BaselinePlugin, DataAdapter, TaskBuilder
from physbench.domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)
from physbench.io import canonical_sha256, sha256_file, write_json
from physbench.prompts import PromptRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = (
    PROJECT_ROOT / "src" / "physbench" / "baseline_plugins" / "resources"
)
VALID_ASPECT_RATIOS = {"16,9", "4,3", "1,1", "3,4", "9,16"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Cosmos3DataAdapter(DataAdapter):
    """Convert immutable benchmark Cases into Cosmos3-Nano I2V inputs."""

    REQUIRED_STAGES = {"spatial", "temporal", "paradigm", "text", "physics"}

    def __init__(self, config: dict[str, Any], implementation_digest: str):
        self.config = config
        self.implementation_digest = implementation_digest
        missing = self.REQUIRED_STAGES - set(config)
        if missing:
            raise ValueError(
                f"Cosmos3 data adapter missing stages: {sorted(missing)}"
            )
        profile_set = config["text"]["profile_set"]
        profile_dir = RESOURCE_ROOT / profile_set
        if not profile_dir.is_dir():
            raise FileNotFoundError(
                f"Cosmos3 data-adapter profile set not found: {profile_dir}"
            )
        self.registry = PromptRegistry(profile_dir)
        if set(self.registry.profiles) != {"generic", "physics"}:
            raise ValueError(
                "Cosmos3 profiles must define exactly generic and physics"
            )
        temporal = config["temporal"]
        frames = int(temporal["num_frames"])
        if frames < 5 or (frames - 1) % 4:
            raise ValueError(
                "Cosmos3 temporal.num_frames must be at least 5 and 1 modulo 4"
            )
        for profile in config["spatial"]["scene_profiles"].values():
            if profile["aspect_ratio"] not in VALID_ASPECT_RATIOS:
                raise ValueError(
                    f"invalid Cosmos3 aspect ratio {profile['aspect_ratio']}"
                )

    @property
    def stage_fingerprints(self) -> dict[str, str]:
        profiles = self.registry.snapshot(["generic", "physics"])["profiles"]
        return {
            "spatial": canonical_sha256({
                "config": self.config["spatial"],
                "implementation": self.implementation_digest,
            }),
            "temporal": canonical_sha256({
                "config": self.config["temporal"],
                "implementation": self.implementation_digest,
            }),
            "paradigm": canonical_sha256({
                "config": self.config["paradigm"],
                "implementation": self.implementation_digest,
            }),
            "text": canonical_sha256({
                "config": self.config["text"],
                "generic_profile": profiles["generic"],
                "implementation": self.implementation_digest,
            }),
            "physics": canonical_sha256({
                "config": self.config["physics"],
                "physics_profile": profiles["physics"],
                "implementation": self.implementation_digest,
            }),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": "cosmos3_nano_i2v_case_data_v1",
            "stages": self.stage_fingerprints,
        })

    @property
    def materialization_fingerprint(self) -> str:
        return canonical_sha256({
            "type": "cosmos3_source_asset_passthrough_v1",
            "spatial": self.stage_fingerprints["spatial"],
            "temporal": self.stage_fingerprints["temporal"],
            "paradigm": self.stage_fingerprints["paradigm"],
        })

    def _spatial_profile(self, scene_id: str) -> dict[str, Any]:
        try:
            return dict(self.config["spatial"]["scene_profiles"][scene_id])
        except KeyError as exc:
            raise ValueError(
                f"no Cosmos3 spatial profile configured for scene {scene_id}"
            ) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "type": "cosmos3_nano_i2v_case_data_v1",
            "fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "stage_fingerprints": self.stage_fingerprints,
            "stages": [
                "spatial",
                "temporal",
                "paradigm",
                "text",
                "physics",
            ],
            "ownership": "baseline",
            "source_assets_mutated": False,
            "native_inputs_are_opaque_to_benchmark": True,
            "config": self.config,
        }

    def adapt_case(
        self, case: dict[str, Any], conditioning: str, *, role: str
    ) -> dict[str, Any]:
        if conditioning not in {"generic", "physics"}:
            raise ValueError(
                f"unsupported Cosmos3 conditioning {conditioning}"
            )
        first_frame = case["assets"].get("first_frame")
        if not first_frame:
            raise ValueError(
                f"Cosmos3 I2V case has no first-frame asset: {case['case_id']}"
            )
        generic_case = {
            "case_id": case["case_id"],
            "scene_id": case["scene_id"],
            "physical_parameters": {},
        }
        generic = self.registry.resolve(generic_case, "generic", role=role)
        if generic["used_parameters"]:
            raise AssertionError(
                "Cosmos3 generic text adaptation leaked physical parameters"
            )
        record = (
            self.registry.resolve(
                {
                    "case_id": case["case_id"],
                    "scene_id": case["scene_id"],
                    "physical_parameters": case["physics"],
                },
                "physics",
                role=role,
            )
            if conditioning == "physics"
            else generic
        )
        spatial = self._spatial_profile(case["scene_id"])
        temporal = self.config["temporal"]
        record.update({
            "schema_version": "2.0",
            "conditioning": conditioning,
            "data_adapter_fingerprint": self.fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "stages": {
                "spatial": {
                    "type": self.config["spatial"]["type"],
                    "resolution": spatial["resolution"],
                    "aspect_ratio": spatial["aspect_ratio"],
                    "materialization": "native_cosmos_preprocessor",
                },
                "temporal": {
                    "type": temporal["type"],
                    "fps": temporal["fps"],
                    "num_frames": temporal["num_frames"],
                    "valid_frame_rule": "4n+1",
                },
                "paradigm": {
                    "type": self.config["paradigm"]["type"],
                    "mode": "image2video",
                    "source": "assets.first_frame",
                },
                "text": {
                    "type": self.config["text"]["type"],
                    "generic_description": generic["prompt"],
                    "contains_detailed_physics": False,
                },
                "physics": {
                    "type": self.config["physics"]["type"],
                    "enabled": conditioning == "physics",
                    "strategy": (
                        self.config["physics"]["strategy"]
                        if conditioning == "physics"
                        else "disabled"
                    ),
                    "used_parameters": record["used_parameters"],
                },
            },
            "native_inputs": {
                "vision": {
                    "paradigm": "image2video",
                    "first_frame_asset": first_frame,
                },
                "text": {"prompt": record["prompt"]},
                "generation_shape": {
                    "resolution": str(spatial["resolution"]),
                    "aspect_ratio": spatial["aspect_ratio"],
                    "fps": int(temporal["fps"]),
                    "num_frames": int(temporal["num_frames"]),
                },
            },
        })
        return record


class Cosmos3TaskBuilder(TaskBuilder):
    TYPE = "cosmos3_nano_i2v_task_builder_v1"

    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self._dependency_fingerprints = self._compute_dependency_fingerprints()
        implementation_digest = canonical_sha256(
            self._dependency_fingerprints
        )
        config = bundle.value["components"]["task_builder"]["config"][
            "data_adapter"
        ]
        self.data_adapter = Cosmos3DataAdapter(config, implementation_digest)

    def _compute_dependency_fingerprints(self) -> dict[str, str]:
        paths = {
            "plugin/implementation.py": Path(__file__),
            "src/physbench/baseline_api/endpoint.py": (
                PROJECT_ROOT / "src" / "physbench" / "baseline_api"
                / "endpoint.py"
            ),
            "src/physbench/baseline_plugins/resources/"
            "five_scene_i2v_v1/generic.json": (
                RESOURCE_ROOT / "five_scene_i2v_v1" / "generic.json"
            ),
            "src/physbench/baseline_plugins/resources/"
            "five_scene_i2v_v1/physics.json": (
                RESOURCE_ROOT / "five_scene_i2v_v1" / "physics.json"
            ),
        }
        framework_root = Path(
            self.bundle.value["runtime"]["framework_root"]
        )
        inference_entry = (
            framework_root / "cosmos_framework" / "scripts" / "inference.py"
        )
        if inference_entry.is_file():
            paths["external/cosmos_framework/scripts/inference.py"] = (
                inference_entry
            )
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cosmos3 runtime dependencies missing: {missing}"
            )
        return {
            name: sha256_file(path)
            for name, path in sorted(paths.items())
        }

    @property
    def dependency_fingerprints(self) -> dict[str, str]:
        return dict(self._dependency_fingerprints)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256({
            "type": self.TYPE,
            "data_adapter": self.data_adapter.fingerprint,
            "predictor": self.bundle.value["components"]["predictor"],
            "model": self.bundle.value["model"],
            "bundle_digest": self.bundle.digest,
            "deployment_digest": self.bundle.deployment_digest,
            "runtime_dependencies": self.dependency_fingerprints,
        })

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.TYPE,
            "fingerprint": self.fingerprint,
            "ownership": "baseline",
            "build_is_side_effect_free": True,
            "canonical_plan_owner": "benchmark",
            "runtime_dependency_fingerprints": self.dependency_fingerprints,
            "data_adapter": self.data_adapter.describe(),
            "output": "BaselineTaskInstance",
        }

    def _validate_compatibility(
        self, task: TaskSpec, plan: AtomicPlan
    ) -> None:
        capabilities = self.bundle.value["capabilities"]
        if task.family not in capabilities["task_families"]:
            raise ValueError(
                f"baseline does not support task family {task.family}"
            )
        if task.conditioning not in capabilities["conditioning"]:
            raise ValueError(
                f"baseline does not support conditioning {task.conditioning}"
            )
        supported = set(self.bundle.value["supported_scenes"])
        requested = set(plan.value["scene_ids"]) | {
            job["scene_id"] for job in plan.jobs
        }
        unknown = requested - supported
        if unknown:
            raise ValueError(
                f"baseline does not support scenes {sorted(unknown)}"
            )
        model = self.bundle.value["model"]
        checkpoint = model.get("checkpoint")
        if not checkpoint or not Path(checkpoint).is_dir():
            raise FileNotFoundError(
                f"Cosmos3 checkpoint directory not found: {checkpoint}"
            )
        checkpoint_root = Path(checkpoint)
        expected = model["identity_files"]
        for relative, expected_digest in expected.items():
            path = checkpoint_root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cosmos3 checkpoint identity file missing: {path}"
                )
            actual = sha256_file(path)
            if actual != expected_digest:
                raise ValueError(
                    f"Cosmos3 checkpoint identity mismatch for {relative}: "
                    f"expected={expected_digest}, actual={actual}"
                )

    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        self._validate_compatibility(task, canonical_plan)
        by_id = {case["case_id"]: case for case in dataset.cases}
        eval_ids = sorted({job["case_id"] for job in canonical_plan.jobs})
        adaptations = []
        by_case = {}
        for case_id in eval_ids:
            record = self.data_adapter.adapt_case(
                by_id[case_id], task.conditioning, role="eval"
            )
            record["adaptation_id"] = (
                f"{case_id}::eval::{task.conditioning}"
            )
            adaptations.append(record)
            by_case[case_id] = record
        jobs = []
        for job in canonical_plan.jobs:
            adaptation = by_case[job["case_id"]]
            jobs.append({
                **job,
                "adaptation_id": adaptation["adaptation_id"],
                "model_ref": "baseline://frozen_model",
                "native_inputs": adaptation["native_inputs"],
            })
        jobs.sort(key=lambda item: item["job_id"])
        plan_digest = canonical_sha256(canonical_plan.value)
        instance_id = (
            f"{task.task_id}__{self.bundle.baseline_id}"
            f"__{self.fingerprint[:12]}"
        )
        return BaselineTaskInstance.seal({
            "schema_version": "2.1",
            "instance_id": instance_id,
            "identity": {
                "dataset": {
                    "dataset_id": dataset.dataset_id,
                    "digest": dataset.digest,
                },
                "task": {"task_id": task.task_id, "digest": task.digest},
                "baseline": {
                    "baseline_id": self.bundle.baseline_id,
                    "baseline_version": self.bundle.baseline_version,
                    "digest": self.bundle.digest,
                    "deployment_digest": self.bundle.deployment_digest,
                },
                "task_builder": {
                    "type": self.TYPE,
                    "fingerprint": self.fingerprint,
                },
                "data_adapter": {
                    "fingerprint": self.data_adapter.fingerprint,
                    "materialization_fingerprint": (
                        self.data_adapter.materialization_fingerprint
                    ),
                },
                "canonical_plan_digest": plan_digest,
            },
            "semantics": {
                "family": task.family,
                "conditioning": task.conditioning,
                "scene_ids": canonical_plan.value["scene_ids"],
            },
            "canonical_plan": canonical_plan.value,
            "source": {
                "asset_root": str(dataset.asset_root),
                "cases": [by_id[case_id] for case_id in eval_ids],
            },
            "adaptations": adaptations,
            "training": None,
            "inference": {
                "predictor": self.bundle.value["components"]["predictor"],
                "jobs": jobs,
            },
            "execution_graph": {
                "operations": [
                    {
                        "operation_id": "infer",
                        "kind": "infer",
                        "depends_on": [],
                        "model_ref": "baseline://frozen_model",
                        "job_ids": [job["job_id"] for job in jobs],
                    },
                    {
                        "operation_id": "evaluate",
                        "kind": "evaluate",
                        "depends_on": ["infer"],
                        "job_ids": [job["job_id"] for job in jobs],
                    },
                ]
            },
            "cache_bindings": [],
            "baseline_payload": {
                "type": "cosmos3_nano_i2v_task_v1",
                "model": self.bundle.value["model"],
                "runtime": self.bundle.value["runtime"],
            },
        })


class Cosmos3BaselinePlugin(BaselinePlugin):
    def __init__(self, bundle: BaselineBundle):
        self.bundle = bundle
        self.task_builder = Cosmos3TaskBuilder(bundle)

    def _verify_instance(self, instance: BaselineTaskInstance) -> None:
        instance.verify()
        identity = instance.value["identity"]
        baseline = identity["baseline"]
        if baseline["baseline_id"] != self.bundle.baseline_id:
            raise ValueError("task instance targets a different baseline")
        if baseline["digest"] != self.bundle.digest:
            raise ValueError("task instance baseline snapshot digest mismatch")
        if baseline["deployment_digest"] != self.bundle.deployment_digest:
            raise ValueError("task instance baseline deployment digest mismatch")
        if (
            identity["task_builder"]["fingerprint"]
            != self.task_builder.fingerprint
        ):
            raise ValueError("task instance TaskBuilder fingerprint mismatch")

    @staticmethod
    def _cuda_library_path(python: Path) -> str:
        environment_root = python.resolve().parents[1]
        paths = sorted(
            str(path)
            for path in environment_root.glob(
                "lib/python*/site-packages/nvidia/*/lib"
            )
            if path.is_dir()
        )
        return os.pathsep.join(paths)

    def _execute_job(
        self, spec: dict[str, Any], *, log_path: Path
    ) -> tuple[int, list[str]]:
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
        torchrun = Path(runtime["torchrun"])
        framework_root = Path(runtime["framework_root"])
        if not torchrun.is_file():
            raise FileNotFoundError(f"Cosmos3 torchrun not found: {torchrun}")
        command = [
            str(torchrun),
            f"--nproc-per-node={len(gpu_ids)}",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_port()}",
            "-m",
            "cosmos_framework.scripts.inference",
            f"--parallelism-preset={runtime.get('parallelism_preset', 'throughput')}",
            "-i",
            spec["payload_path"],
            "-o",
            spec["output_root"],
            "--checkpoint-path",
            str(self.bundle.value["model"]["checkpoint"]),
            f"--seed={spec['seed']}",
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
        python = Path(runtime["python"])
        environment_root = python.resolve().parents[1]
        env = os.environ.copy()
        cuda_libraries = self._cuda_library_path(python)
        previous_libraries = env.get("LD_LIBRARY_PATH", "")
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "HF_HOME": str(runtime["hf_home"]),
            "UV_CACHE_DIR": str(runtime["uv_cache_dir"]),
            "PATH": f"{environment_root / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": os.pathsep.join(
                item
                for item in (cuda_libraries, previous_libraries)
                if item
            ),
        })
        if runtime.get("offline", True):
            env["HF_HUB_OFFLINE"] = "1"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=framework_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return completed.returncode, command

    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self._verify_instance(instance)
        value = instance.value
        source_root = Path(value["source"]["asset_root"])
        cases = {
            case["case_id"]: case for case in value["source"]["cases"]
        }
        adaptations = {
            item["adaptation_id"]: item for item in value["adaptations"]
        }
        predictor = value["inference"]["predictor"]["config"]
        conditioning = value["semantics"]["conditioning"]
        training = {
            "operation_id": "train",
            "status": "not_requested",
            "note": "Frozen Cosmos3-Nano direct-eval baseline.",
        }
        predictions = []
        for job in value["inference"]["jobs"]:
            case = cases[job["case_id"]]
            adaptation = adaptations[job["adaptation_id"]]
            native = job["native_inputs"]
            first_frame = (
                source_root / native["vision"]["first_frame_asset"]
            ).resolve()
            if not first_frame.is_file():
                raise FileNotFoundError(
                    f"Cosmos3 first frame not found: {first_frame}"
                )
            shape = native["generation_shape"]
            output_root = (
                run_dir / "predictions" / conditioning
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
            spec = {
                **job,
                "baseline_id": self.bundle.baseline_id,
                "conditioning": conditioning,
                "prompt_profile_id": conditioning,
                "payload_path": str(payload_path),
                "output_root": str(output_root),
                "output_video": str(output_video),
                "first_frame": str(first_frame),
                "prompt_record": adaptation,
                "seed": int(job["seed"]),
            }
            spec_path = run_dir / "jobs" / f"{job['job_id']}.json"
            write_json(spec_path, spec)
            common = {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "baseline_id": self.bundle.baseline_id,
                "conditioning": conditioning,
                "prompt_profile_id": conditioning,
                "evaluation_partition": job["evaluation_partition"],
                "evaluation_reference_video": (
                    str(source_root / case["assets"]["physics_reference_video"])
                    if case["assets"].get("physics_reference_video")
                    else None
                ),
                "visual_reference_video": (
                    str(source_root / case["assets"]["reference_video"])
                    if (
                        case["has_real_reference_video"]
                        and case["assets"].get("reference_video")
                    )
                    else None
                ),
                "manual_scores": {},
                "job_spec": str(spec_path),
            }
            if stop_after_training:
                predictions.append({
                    **common,
                    "status": "staged",
                    "video_path": None,
                })
            elif not execute:
                predictions.append({
                    **common,
                    "status": "planned",
                    "video_path": None,
                    "note": "Cosmos3 generation is planned; execution is disabled.",
                })
            else:
                return_code, command = self._execute_job(
                    spec,
                    log_path=(
                        run_dir / "logs" / "cosmos3"
                        / f"{job['job_id']}.log"
                    ),
                )
                predictions.append({
                    **common,
                    "status": (
                        "complete"
                        if return_code == 0 and output_video.is_file()
                        else "failed"
                    ),
                    "video_path": (
                        str(output_video) if output_video.is_file() else None
                    ),
                    "return_code": return_code,
                    "command": command,
                })
        return training, predictions
